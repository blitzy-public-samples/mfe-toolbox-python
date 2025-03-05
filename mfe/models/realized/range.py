# mfe/models/realized/range.py
"""
Realized Range estimator for quadratic variation.

This module implements the Realized Range estimator, which uses high-low ranges
within bins rather than squared returns to estimate quadratic variation. The range-based
estimator is more efficient than standard realized variance when high and low prices
are available within each sampling interval.

The implementation follows the class-based architecture of the MFE Toolbox with
comprehensive type hints, parameter validation, and Numba acceleration for
performance-critical calculations. It supports both NumPy arrays and Pandas DataFrames
with datetime indices for convenient time series handling.

Classes:
    RealizedRangeConfig: Configuration parameters for Realized Range estimation
    RealizedRange: Realized Range estimator implementation

References:
    Christensen, K., & Podolskij, M. (2007). Realized range-based estimation of 
    integrated variance. Journal of Econometrics, 141(2), 323-349.
    
    Martens, M., & van Dijk, D. (2007). Measuring volatility with the realized range.
    Journal of Econometrics, 138(1), 181-207.
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Literal, Optional, Sequence, 
    Tuple, Type, TypeVar, Union, cast, overload
)
import numpy as np
import pandas as pd
from scipy import stats

from ...core.parameters import ParameterBase, validate_positive, validate_non_negative
from ...core.exceptions import ParameterError, DimensionError, NumericError
from .base import BaseRealizedEstimator, RealizedEstimatorConfig, RealizedEstimatorResult

# Set up module-level logger
logger = logging.getLogger("mfe.models.realized.range")

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
    logger.debug("Numba available for Realized Range acceleration")
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    HAS_NUMBA = False
    logger.info("Numba not available. Realized Range will use pure NumPy implementation.")


@dataclass
class RealizedRangeConfig(RealizedEstimatorConfig):
    """Configuration parameters for Realized Range estimation.
    
    This class extends RealizedEstimatorConfig with parameters specific to
    the Realized Range estimator.
    
    Attributes:
        sampling_frequency: Sampling frequency for price data (e.g., '5min', 300)
        annualize: Whether to annualize the volatility estimate
        annualization_factor: Factor to use for annualization (e.g., 252 for daily data)
        return_type: Type of returns to compute ('log', 'simple')
        use_subsampling: Whether to use subsampling for noise reduction
        subsampling_factor: Factor for subsampling (number of subsamples)
        apply_noise_correction: Whether to apply microstructure noise correction
        kernel_type: Type of kernel for kernel-based estimators
        bandwidth: Bandwidth parameter for kernel-based estimators
        time_unit: Unit of time for high-frequency data ('seconds', 'minutes', etc.)
        scale_factor: Scale factor to correct for discretization bias (default: 4 log(2))
        use_parkinson: Whether to use Parkinson's scaling (default: True)
        min_observations: Minimum number of observations required in each bin
    """
    
    scale_factor: Optional[float] = None
    use_parkinson: bool = True
    min_observations: int = 2
    
    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        super().__post_init__()
        
        # Validate min_observations
        if self.min_observations < 2:
            raise ParameterError(f"min_observations must be at least 2 for range calculation, got {self.min_observations}")
        
        # Set default scale factor if not provided
        if self.scale_factor is None:
            if self.use_parkinson:
                # Parkinson's scaling factor: 4 log(2)
                self.scale_factor = 4.0 * np.log(2.0)
            else:
                # Default scale factor (theoretical scaling for Brownian motion)
                self.scale_factor = 1.0


@jit(nopython=True, cache=True)
def _compute_realized_range_numba(
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    scale_factor: float,
    min_observations: int
) -> float:
    """
    Numba-accelerated implementation of realized range computation.
    
    Args:
        high_prices: Array of high prices for each bin
        low_prices: Array of low prices for each bin
        scale_factor: Scale factor to correct for discretization bias
        min_observations: Minimum number of observations required in each bin
        
    Returns:
        Realized range estimate
    """
    n_bins = len(high_prices)
    realized_range = 0.0
    valid_bins = 0
    
    for i in range(n_bins):
        # Skip bins with invalid prices
        if np.isnan(high_prices[i]) or np.isnan(low_prices[i]):
            continue
        
        # Skip bins where high <= low (should not happen with proper data)
        if high_prices[i] <= low_prices[i]:
            continue
        
        # Compute squared range for this bin
        squared_range = (np.log(high_prices[i]) - np.log(low_prices[i]))**2
        
        # Add to the total
        realized_range += squared_range
        valid_bins += 1
    
    # Apply scaling factor
    if valid_bins > 0:
        realized_range = realized_range / (scale_factor * valid_bins)
    else:
        realized_range = np.nan
    
    return realized_range


def _compute_bin_high_low(
    prices: np.ndarray,
    times: np.ndarray,
    bin_edges: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute high and low prices within each bin.
    
    Args:
        prices: Array of price data
        times: Array of time points corresponding to prices
        bin_edges: Array of bin edge time points
        
    Returns:
        Tuple of (bin_highs, bin_lows, bin_counts, bin_centers)
    """
    n_bins = len(bin_edges) - 1
    bin_highs = np.full(n_bins, np.nan)
    bin_lows = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins, dtype=int)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Assign each price to a bin
    bin_indices = np.digitize(times, bin_edges) - 1
    
    # Filter out points outside the bins
    valid_mask = (bin_indices >= 0) & (bin_indices < n_bins)
    valid_prices = prices[valid_mask]
    valid_indices = bin_indices[valid_mask]
    
    # Compute high and low for each bin
    for i in range(n_bins):
        bin_mask = (valid_indices == i)
        bin_prices = valid_prices[bin_mask]
        bin_counts[i] = len(bin_prices)
        
        if bin_counts[i] > 0:
            bin_highs[i] = np.max(bin_prices)
            bin_lows[i] = np.min(bin_prices)
    
    return bin_highs, bin_lows, bin_counts, bin_centers


class RealizedRange(BaseRealizedEstimator):
    """Realized Range estimator for quadratic variation.
    
    This class implements the Realized Range estimator, which uses high-low ranges
    within bins rather than squared returns to estimate quadratic variation. The
    range-based estimator is more efficient than standard realized variance when
    high and low prices are available within each sampling interval.
    
    Attributes:
        config: Configuration parameters for the estimator
        name: A descriptive name for the estimator
    """
    
    def __init__(self, config: Optional[RealizedRangeConfig] = None, name: str = "RealizedRange"):
        """Initialize the Realized Range estimator.
        
        Args:
            config: Configuration parameters for the estimator
            name: A descriptive name for the estimator
        """
        super().__init__(config=config if config is not None else RealizedRangeConfig(), name=name)
        self._bin_highs: Optional[np.ndarray] = None
        self._bin_lows: Optional[np.ndarray] = None
        self._bin_counts: Optional[np.ndarray] = None
        self._bin_centers: Optional[np.ndarray] = None
    
    @property
    def config(self) -> RealizedRangeConfig:
        """Get the estimator configuration.
        
        Returns:
            RealizedRangeConfig: The estimator configuration
        """
        return cast(RealizedRangeConfig, self._config)
    
    @config.setter
    def config(self, config: RealizedRangeConfig) -> None:
        """Set the estimator configuration.
        
        Args:
            config: New configuration parameters
        
        Raises:
            TypeError: If config is not a RealizedRangeConfig
        """
        if not isinstance(config, RealizedRangeConfig):
            raise TypeError(f"config must be a RealizedRangeConfig, got {type(config)}")
        
        self._config = config
        self._fitted = False  # Reset fitted state when configuration changes
    
    @property
    def bin_highs(self) -> Optional[np.ndarray]:
        """Get the high prices for each bin.
        
        Returns:
            Optional[np.ndarray]: High prices if the estimator has been fitted,
                                 None otherwise
        """
        return self._bin_highs
    
    @property
    def bin_lows(self) -> Optional[np.ndarray]:
        """Get the low prices for each bin.
        
        Returns:
            Optional[np.ndarray]: Low prices if the estimator has been fitted,
                                 None otherwise
        """
        return self._bin_lows
    
    @property
    def bin_counts(self) -> Optional[np.ndarray]:
        """Get the number of observations in each bin.
        
        Returns:
            Optional[np.ndarray]: Observation counts if the estimator has been fitted,
                                 None otherwise
        """
        return self._bin_counts
    
    @property
    def bin_centers(self) -> Optional[np.ndarray]:
        """Get the center time points of each bin.
        
        Returns:
            Optional[np.ndarray]: Bin centers if the estimator has been fitted,
                                 None otherwise
        """
        return self._bin_centers
    
    def _compute_realized_measure(self, 
                                 prices: np.ndarray, 
                                 times: np.ndarray, 
                                 returns: np.ndarray,
                                 **kwargs: Any) -> np.ndarray:
        """Compute the realized range from the preprocessed data.
        
        Args:
            prices: Preprocessed price data
            times: Preprocessed time points
            returns: Returns computed from prices
            **kwargs: Additional keyword arguments for computation
        
        Returns:
            np.ndarray: Realized range measure
            
        Raises:
            ValueError: If computation fails
        """
        # Get configuration parameters
        config = self.config
        scale_factor = config.scale_factor
        min_observations = config.min_observations
        
        # Create bins based on sampling frequency
        if config.sampling_frequency is not None:
            # Use price_filter to create regular bins
            from .price_filter import price_filter
            
            try:
                # Get filtered times to use as bin centers
                _, filtered_times = price_filter(
                    prices, times, 
                    sample_freq=config.sampling_frequency,
                    time_unit=config.time_unit,
                    sampling_scheme='calendar',
                    interpolation_method='previous'
                )
                
                # Create bin edges from filtered times
                if len(filtered_times) > 1:
                    bin_width = np.mean(np.diff(filtered_times))
                    bin_edges = np.concatenate([
                        [filtered_times[0] - bin_width/2],
                        filtered_times[:-1] + np.diff(filtered_times)/2,
                        [filtered_times[-1] + bin_width/2]
                    ])
                else:
                    # Not enough filtered times to create bins
                    raise ValueError("Insufficient filtered time points to create bins")
            except Exception as e:
                logger.warning(f"Failed to create bins using price_filter: {str(e)}")
                logger.warning("Falling back to uniform binning")
                
                # Create uniform bins
                n_bins = max(10, int(len(prices) / min_observations))
                bin_edges = np.linspace(times[0], times[-1], n_bins + 1)
        else:
            # Create uniform bins
            n_bins = max(10, int(len(prices) / min_observations))
            bin_edges = np.linspace(times[0], times[-1], n_bins + 1)
        
        # Compute high and low prices within each bin
        bin_highs, bin_lows, bin_counts, bin_centers = _compute_bin_high_low(
            prices, times, bin_edges
        )
        
        # Store bin information
        self._bin_highs = bin_highs
        self._bin_lows = bin_lows
        self._bin_counts = bin_counts
        self._bin_centers = bin_centers
        
        # Filter bins with insufficient observations
        valid_bins = bin_counts >= min_observations
        if not np.any(valid_bins):
            raise ValueError(f"No bins have at least {min_observations} observations")
        
        filtered_highs = bin_highs[valid_bins]
        filtered_lows = bin_lows[valid_bins]
        
        # Compute realized range
        if HAS_NUMBA:
            # Use Numba-accelerated implementation
            realized_range = _compute_realized_range_numba(
                filtered_highs, filtered_lows, scale_factor, min_observations
            )
        else:
            # Pure NumPy implementation
            # Compute squared ranges
            squared_ranges = (np.log(filtered_highs) - np.log(filtered_lows))**2
            
            # Remove any invalid values
            valid_ranges = ~np.isnan(squared_ranges)
            if not np.any(valid_ranges):
                raise ValueError("No valid ranges computed")
            
            # Compute realized range
            realized_range = np.sum(squared_ranges[valid_ranges]) / (scale_factor * np.sum(valid_ranges))
        
        # Apply subsampling if requested
        if config.use_subsampling and config.subsampling_factor > 1:
            # Implement subsampling by shifting the bins
            subsample_ranges = []
            subsample_ranges.append(realized_range)  # Add the original estimate
            
            for i in range(1, config.subsampling_factor):
                # Shift bin edges
                shift_amount = i * (bin_edges[1] - bin_edges[0]) / config.subsampling_factor
                shifted_edges = bin_edges + shift_amount
                
                # Compute high and low prices for shifted bins
                shifted_highs, shifted_lows, shifted_counts, _ = _compute_bin_high_low(
                    prices, times, shifted_edges
                )
                
                # Filter bins with insufficient observations
                valid_shifted = shifted_counts >= min_observations
                if not np.any(valid_shifted):
                    continue
                
                filtered_shifted_highs = shifted_highs[valid_shifted]
                filtered_shifted_lows = shifted_lows[valid_shifted]
                
                # Compute realized range for this subsample
                if HAS_NUMBA:
                    subsample_range = _compute_realized_range_numba(
                        filtered_shifted_highs, filtered_shifted_lows, scale_factor, min_observations
                    )
                else:
                    # Compute squared ranges
                    shifted_squared_ranges = (np.log(filtered_shifted_highs) - np.log(filtered_shifted_lows))**2
                    
                    # Remove any invalid values
                    valid_shifted_ranges = ~np.isnan(shifted_squared_ranges)
                    if not np.any(valid_shifted_ranges):
                        continue
                    
                    # Compute realized range for this subsample
                    subsample_range = np.sum(shifted_squared_ranges[valid_shifted_ranges]) / (
                        scale_factor * np.sum(valid_shifted_ranges)
                    )
                
                # Add to the list of subsample estimates
                if not np.isnan(subsample_range):
                    subsample_ranges.append(subsample_range)
            
            # Average across subsamples
            if subsample_ranges:
                realized_range = np.mean(subsample_ranges)
        
        # Return as a single-element array for consistency with other estimators
        return np.array([realized_range])
    
    def fit(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> RealizedEstimatorResult:
        """Fit the Realized Range estimator to the provided data.
        
        This method extends the base fit method to handle Pandas DataFrame inputs
        with datetime indices, making it more convenient for time series analysis.
        
        Args:
            data: The data to fit the estimator to, as a tuple of (prices, times)
                 or a pandas DataFrame with a datetime index
            **kwargs: Additional keyword arguments for estimation
        
        Returns:
            RealizedEstimatorResult: The estimation results
        
        Raises:
            ValueError: If the data is invalid
            RuntimeError: If the estimation fails
        """
        # Handle pandas DataFrame input
        if isinstance(data, pd.DataFrame):
            if len(data.columns) != 1:
                raise ValueError("DataFrame must have exactly one column of price data")
            
            prices = data.iloc[:, 0].values
            times = data.index
            
            # Convert to numpy arrays
            if isinstance(times, pd.DatetimeIndex):
                # Convert to seconds since first observation
                times_array = times.astype(np.int64) / 1e9
                times_array = times_array - times_array[0]
            else:
                times_array = np.asarray(times)
            
            data_tuple = (prices, times_array)
            return super().fit(data_tuple, **kwargs)
        
        # Handle pandas Series input
        elif isinstance(data, pd.Series):
            prices = data.values
            times = data.index
            
            # Convert to numpy arrays
            if isinstance(times, pd.DatetimeIndex):
                # Convert to seconds since first observation
                times_array = times.astype(np.int64) / 1e9
                times_array = times_array - times_array[0]
            else:
                times_array = np.asarray(times)
            
            data_tuple = (prices, times_array)
            return super().fit(data_tuple, **kwargs)
        
        # Handle tuple input
        elif isinstance(data, tuple):
            # Check if the first element is a pandas Series or DataFrame
            if isinstance(data[0], (pd.Series, pd.DataFrame)):
                if isinstance(data[0], pd.Series):
                    prices = data[0].values
                else:  # DataFrame
                    if len(data[0].columns) != 1:
                        raise ValueError("DataFrame must have exactly one column of price data")
                    prices = data[0].iloc[:, 0].values
                
                # Check if the second element is a pandas Series, DatetimeIndex, or array-like
                if isinstance(data[1], pd.Series):
                    times = data[1].values
                elif isinstance(data[1], pd.DatetimeIndex):
                    # Convert to seconds since first observation
                    times_array = data[1].astype(np.int64) / 1e9
                    times = times_array - times_array[0]
                else:
                    times = np.asarray(data[1])
                
                data_tuple = (prices, times)
                return super().fit(data_tuple, **kwargs)
            else:
                # Standard tuple input
                return super().fit(data, **kwargs)
        
        else:
            raise TypeError("data must be a tuple of (prices, times), a pandas DataFrame, or a pandas Series")
    
    async def fit_async(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> RealizedEstimatorResult:
        """Asynchronously fit the Realized Range estimator to the provided data.
        
        This method provides an asynchronous interface to the fit method,
        allowing for non-blocking estimation in UI contexts.
        
        Args:
            data: The data to fit the estimator to, as a tuple of (prices, times)
                 or a pandas DataFrame with a datetime index
            **kwargs: Additional keyword arguments for estimation
        
        Returns:
            RealizedEstimatorResult: The estimation results
        
        Raises:
            ValueError: If the data is invalid
            RuntimeError: If the estimation fails
        """
        # Use the synchronous implementation for now
        # In a real implementation, this would be truly asynchronous
        return self.fit(data, **kwargs)
    
    def calibrate(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> RealizedRangeConfig:
        """Calibrate the estimator configuration based on the provided data.
        
        This method analyzes the input data and determines optimal configuration
        parameters for the Realized Range estimator.
        
        Args:
            data: The data to calibrate the estimator with, as a tuple of (prices, times)
            **kwargs: Additional keyword arguments for calibration
        
        Returns:
            RealizedRangeConfig: Calibrated configuration
        
        Raises:
            ValueError: If calibration fails
        """
        # Validate input data
        self.validate_data(data)
        prices, times = data
        
        # Create a copy of the current configuration
        calibrated_config = self.config.copy()
        
        # Determine optimal sampling frequency
        try:
            from .utils import compute_optimal_sampling
            optimal_freq = compute_optimal_sampling(prices, times, method='signature')
            calibrated_config.sampling_frequency = optimal_freq
            logger.info(f"Calibrated sampling frequency: {optimal_freq}")
        except Exception as e:
            logger.warning(f"Failed to determine optimal sampling frequency: {str(e)}")
        
        # Determine whether to use subsampling based on data characteristics
        # (e.g., if there's evidence of microstructure noise)
        try:
            from .utils import noise_variance
            returns = np.diff(np.log(prices))
            noise_var = noise_variance(returns)
            
            # If noise variance is significant, enable subsampling
            if noise_var > 1e-6:
                calibrated_config.use_subsampling = True
                
                # Set subsampling factor based on noise level
                noise_level = noise_var / np.var(returns)
                if noise_level > 0.1:
                    calibrated_config.subsampling_factor = 5
                elif noise_level > 0.05:
                    calibrated_config.subsampling_factor = 3
                else:
                    calibrated_config.subsampling_factor = 2
                
                logger.info(f"Enabled subsampling with factor {calibrated_config.subsampling_factor} "
                           f"due to noise level {noise_level:.4f}")
            else:
                calibrated_config.use_subsampling = False
                logger.info("Disabled subsampling due to low noise level")
        except Exception as e:
            logger.warning(f"Failed to determine optimal subsampling: {str(e)}")
        
        # Determine minimum observations per bin based on data density
        avg_obs_per_bin = len(prices) / 20  # Assuming 20 bins as a starting point
        calibrated_config.min_observations = max(2, int(avg_obs_per_bin / 5))
        logger.info(f"Calibrated minimum observations per bin: {calibrated_config.min_observations}")
        
        return calibrated_config
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert estimation results to a pandas DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame containing bin information and realized range
        
        Raises:
            RuntimeError: If the estimator has not been fitted
        """
        if not self._fitted:
            raise RuntimeError("Estimator has not been fitted. Call fit() first.")
        
        if (self._bin_centers is None or self._bin_highs is None or 
            self._bin_lows is None or self._bin_counts is None):
            raise RuntimeError("Bin information is not available")
        
        # Create DataFrame with bin information
        df = pd.DataFrame({
            'bin_center': self._bin_centers,
            'high': self._bin_highs,
            'low': self._bin_lows,
            'count': self._bin_counts
        })
        
        # Add realized range
        if self._realized_measure is not None:
            df['realized_range'] = self._realized_measure[0]
        
        return df
    
    def plot_bins(self, figsize: Tuple[int, int] = (10, 6), 
                 title: str = 'Realized Range Bins',
                 save_path: Optional[str] = None) -> Optional[Any]:
        """Plot the high-low ranges for each bin.
        
        Args:
            figsize: Figure size as (width, height) in inches
            title: Plot title
            save_path: Path to save the figure (if None, figure is displayed)
            
        Returns:
            Matplotlib figure object if matplotlib is available, None otherwise
            
        Raises:
            RuntimeError: If the estimator has not been fitted
        """
        if not self._fitted:
            raise RuntimeError("Estimator has not been fitted. Call fit() first.")
        
        if (self._bin_centers is None or self._bin_highs is None or 
            self._bin_lows is None or self._bin_counts is None):
            raise RuntimeError("Bin information is not available")
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib is not available. Cannot create visualization.")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot high-low ranges
        for i, (center, high, low, count) in enumerate(zip(
            self._bin_centers, self._bin_highs, self._bin_lows, self._bin_counts
        )):
            if count >= self.config.min_observations and not np.isnan(high) and not np.isnan(low):
                # Plot vertical line for the range
                ax.plot([center, center], [low, high], 'b-', alpha=0.7)
                # Plot markers for high and low
                ax.plot(center, high, 'b^', markersize=4)
                ax.plot(center, low, 'bv', markersize=4)
        
        # Add labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.set_title(title)
        
        # Add realized range value as text
        if self._realized_measure is not None:
            realized_range = self._realized_measure[0]
            ax.text(0.02, 0.98, f'Realized Range: {realized_range:.6f}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Save or display the figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.tight_layout()
        
        return fig
    
    def summary(self) -> str:
        """Generate a text summary of the Realized Range estimator.
        
        Returns:
            str: A formatted string containing the estimator summary
        
        Raises:
            RuntimeError: If the estimator has not been fitted
        """
        if not self._fitted:
            return f"Realized Range Estimator: {self._name} (not fitted)"
        
        if self._results is None:
            return f"Realized Range Estimator: {self._name} (fitted, but no results available)"
        
        # Get base summary
        base_summary = self._results.summary()
        
        # Add Realized Range specific information
        rr_summary = "\nRealized Range Specific Information:\n"
        
        if self._bin_counts is not None:
            valid_bins = np.sum(self._bin_counts >= self.config.min_observations)
            total_bins = len(self._bin_counts)
            rr_summary += f"Valid Bins: {valid_bins} / {total_bins} ({valid_bins/total_bins*100:.1f}%)\n"
        
        if self.config.scale_factor is not None:
            rr_summary += f"Scale Factor: {self.config.scale_factor:.4f}\n"
        
        rr_summary += f"Parkinson Scaling: {'Enabled' if self.config.use_parkinson else 'Disabled'}\n"
        rr_summary += f"Minimum Observations per Bin: {self.config.min_observations}\n"
        
        return base_summary + rr_summary
    
    def __str__(self) -> str:
        """Generate a string representation of the estimator.
        
        Returns:
            str: A string representation of the estimator
        """
        return self.summary()
    
    def __repr__(self) -> str:
        """Generate a detailed string representation of the estimator.
        
        Returns:
            str: A detailed string representation of the estimator
        """
        return f"RealizedRange(name='{self._name}', fitted={self._fitted})"