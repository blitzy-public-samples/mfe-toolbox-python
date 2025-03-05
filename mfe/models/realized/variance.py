"""
Realized Variance Estimator

This module implements the standard realized variance estimator, which is the sum of
squared high-frequency returns. This is the most fundamental volatility measure and
serves as the basis for many other estimators in high-frequency financial econometrics.

The realized variance estimator provides a non-parametric measure of volatility that
converges to the integrated variance as the sampling frequency increases, under ideal
conditions. It is widely used in financial applications for risk management, option
pricing, and volatility forecasting.

The implementation supports various sampling schemes, subsampling for noise reduction,
and visualization capabilities, with comprehensive type hints and parameter validation.
Performance-critical calculations are accelerated using Numba's JIT compilation.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, cast
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from ...core.base import ModelBase
from ...core.parameters import ParameterBase, ParameterError, validate_positive, validate_non_negative
from ...core.exceptions import DimensionError, NumericError
from ..realized.base import BaseRealizedEstimator, RealizedEstimatorConfig, RealizedEstimatorResult
from ..realized.utils import (
    compute_returns, compute_realized_variance, compute_subsampled_measure,
    sample_prices, align_time, seconds2unit, unit2seconds
)

# Set up module-level logger
logger = logging.getLogger("mfe.models.realized.variance")

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
    logger.debug("Numba available for realized variance acceleration")
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    HAS_NUMBA = False
    logger.info("Numba not available. Realized variance will use pure NumPy implementation.")


@jit(nopython=True, cache=True)
def _compute_realized_variance_numba(returns: np.ndarray) -> float:
    """
    Numba-accelerated implementation of realized variance computation.
    
    Args:
        returns: Array of returns
        
    Returns:
        float: Realized variance (sum of squared returns)
    """
    return np.sum(returns**2)


@dataclass
class RealizedVarianceConfig(RealizedEstimatorConfig):
    """Configuration parameters for realized variance estimator.
    
    This class extends RealizedEstimatorConfig with parameters specific to
    the realized variance estimator.
    
    Attributes:
        sampling_frequency: Sampling frequency for price data (e.g., '5min', 300)
        annualize: Whether to annualize the volatility estimate
        annualization_factor: Factor to use for annualization (e.g., 252 for daily data)
        return_type: Type of returns to compute ('log', 'simple')
        use_subsampling: Whether to use subsampling for noise reduction
        subsampling_factor: Factor for subsampling (number of subsamples)
        apply_noise_correction: Whether to apply microstructure noise correction
        time_unit: Unit of time for high-frequency data ('seconds', 'minutes', etc.)
        interpolation_method: Method for interpolating prices ('previous', 'linear', 'cubic')
    """
    
    interpolation_method: str = 'previous'
    
    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        super().__post_init__()
        
        # Validate interpolation_method
        if self.interpolation_method not in ['previous', 'linear', 'cubic']:
            raise ParameterError(
                f"interpolation_method must be 'previous', 'linear', or 'cubic', "
                f"got {self.interpolation_method}"
            )


class RealizedVariance(BaseRealizedEstimator):
    """Realized Variance Estimator.
    
    This class implements the standard realized variance estimator, which is the sum of
    squared high-frequency returns. It provides methods for estimating realized variance
    from high-frequency price data, with support for various sampling schemes, subsampling
    for noise reduction, and visualization capabilities.
    
    The realized variance estimator is the most fundamental volatility measure and serves
    as the basis for many other estimators in high-frequency financial econometrics.
    
    Attributes:
        config: Configuration parameters for the estimator
        name: A descriptive name for the estimator
    """
    
    def __init__(self, 
                config: Optional[RealizedVarianceConfig] = None, 
                name: str = "RealizedVariance"):
        """Initialize the realized variance estimator.
        
        Args:
            config: Configuration parameters for the estimator
            name: A descriptive name for the estimator
        """
        # Use RealizedVarianceConfig if no config is provided
        if config is None:
            config = RealizedVarianceConfig()
        elif not isinstance(config, RealizedVarianceConfig):
            # If a RealizedEstimatorConfig is provided, convert it to RealizedVarianceConfig
            if isinstance(config, RealizedEstimatorConfig):
                config_dict = config.to_dict()
                config_dict.setdefault('interpolation_method', 'previous')
                config = RealizedVarianceConfig(**config_dict)
            else:
                raise TypeError(f"config must be a RealizedVarianceConfig, got {type(config)}")
        
        super().__init__(config=config, name=name)
    
    def _compute_realized_measure(self, 
                                 prices: np.ndarray, 
                                 times: np.ndarray, 
                                 returns: np.ndarray,
                                 **kwargs: Any) -> np.ndarray:
        """Compute realized variance from the preprocessed data.
        
        This method implements the core realized variance calculation, which is
        the sum of squared returns. It supports subsampling for noise reduction
        and can handle various sampling schemes.
        
        Args:
            prices: Preprocessed price data
            times: Preprocessed time points
            returns: Returns computed from prices
            **kwargs: Additional keyword arguments for computation
        
        Returns:
            np.ndarray: Realized variance
        
        Raises:
            ValueError: If computation fails
        """
        # Check if subsampling should be used
        if self._config.use_subsampling and self._config.subsampling_factor > 1:
            try:
                # Compute subsampled realized variance
                rv = compute_subsampled_measure(returns, self._config.subsampling_factor)
                logger.debug(
                    f"Computed subsampled realized variance with {self._config.subsampling_factor} subsamples"
                )
            except Exception as e:
                logger.warning(f"Subsampling failed: {str(e)}. Using standard realized variance.")
                # Fall back to standard realized variance
                rv = compute_realized_variance(returns)
        else:
            # Compute standard realized variance
            if HAS_NUMBA:
                # Use Numba-accelerated implementation
                rv = _compute_realized_variance_numba(returns)
            else:
                # Use pure NumPy implementation
                rv = compute_realized_variance(returns)
        
        # Apply noise correction if requested
        if self._config.apply_noise_correction:
            try:
                from .utils import noise_variance
                # Estimate noise variance
                noise_var = noise_variance(returns)
                # Correct for noise bias (2n * noise_var)
                n = len(returns)
                correction = 2 * n * noise_var
                # Subtract correction (ensure result is non-negative)
                rv = max(0, rv - correction)
                logger.debug(f"Applied noise correction: {correction:.6e}")
            except Exception as e:
                logger.warning(f"Noise correction failed: {str(e)}. Using uncorrected realized variance.")
        
        # Return realized variance as a scalar in a numpy array
        return np.array([rv])
    
    def fit(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> RealizedEstimatorResult:
        """Fit the realized variance estimator to the provided data.
        
        This method validates the input data, preprocesses it according to the
        estimator configuration, and then computes the realized variance.
        
        Args:
            data: The data to fit the estimator to, as a tuple of (prices, times)
            **kwargs: Additional keyword arguments for estimation
        
        Returns:
            RealizedEstimatorResult: The estimation results
        
        Raises:
            ValueError: If the data is invalid
            RuntimeError: If the estimation fails
        """
        return super().fit(data, **kwargs)
    
    async def fit_async(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> RealizedEstimatorResult:
        """Asynchronously fit the realized variance estimator to the provided data.
        
        This method provides an asynchronous interface to the fit method,
        allowing for non-blocking estimation in UI contexts.
        
        Args:
            data: The data to fit the estimator to, as a tuple of (prices, times)
            **kwargs: Additional keyword arguments for estimation
        
        Returns:
            RealizedEstimatorResult: The estimation results
        
        Raises:
            ValueError: If the data is invalid
            RuntimeError: If the estimation fails
        """
        # This implementation uses Python's async/await pattern for asynchronous execution
        import asyncio
        
        # Create a coroutine that runs the synchronous fit method in a thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: self.fit(data, **kwargs)
        )
        
        return result
    
    def calibrate(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> RealizedVarianceConfig:
        """Calibrate the estimator configuration based on the provided data.
        
        This method analyzes the input data and determines optimal configuration
        parameters for the realized variance estimator, such as sampling frequency
        and subsampling factor.
        
        Args:
            data: The data to calibrate the estimator with, as a tuple of (prices, times)
            **kwargs: Additional keyword arguments for calibration
        
        Returns:
            RealizedVarianceConfig: Calibrated configuration
        
        Raises:
            ValueError: If the data is invalid
        """
        # Validate input data
        self.validate_data(data)
        prices, times = data
        
        # Create a copy of the current configuration
        calibrated_config = cast(RealizedVarianceConfig, self._config.copy())
        
        # Determine optimal sampling frequency
        try:
            from .utils import compute_optimal_sampling
            optimal_freq = compute_optimal_sampling(prices, times, method='signature')
            
            # Convert to string representation if needed
            if isinstance(self._config.sampling_frequency, str):
                # Determine the unit from the current sampling_frequency
                import re
                match = re.match(r'(\d+)([a-zA-Z]+)', self._config.sampling_frequency)
                if match:
                    _, unit = match.groups()
                    # Convert optimal_freq to the same unit
                    if unit.lower() in ['s', 'sec', 'second', 'seconds']:
                        optimal_freq_str = f"{int(optimal_freq)}s"
                    elif unit.lower() in ['m', 'min', 'minute', 'minutes']:
                        optimal_freq_str = f"{int(optimal_freq / 60)}m"
                    elif unit.lower() in ['h', 'hour', 'hours']:
                        optimal_freq_str = f"{int(optimal_freq / 3600)}h"
                    else:
                        optimal_freq_str = f"{int(optimal_freq)}s"
                else:
                    optimal_freq_str = f"{int(optimal_freq)}s"
                
                calibrated_config.sampling_frequency = optimal_freq_str
            else:
                calibrated_config.sampling_frequency = optimal_freq
            
            logger.info(f"Calibrated sampling frequency: {calibrated_config.sampling_frequency}")
        except Exception as e:
            logger.warning(f"Sampling frequency calibration failed: {str(e)}")
        
        # Determine whether to use subsampling
        try:
            from .utils import noise_variance
            # Compute returns
            returns = compute_returns(prices, self._config.return_type)
            # Estimate noise variance
            noise_var = noise_variance(returns)
            # If noise variance is significant, enable subsampling
            if noise_var > 1e-6:
                calibrated_config.use_subsampling = True
                # Determine subsampling factor based on noise level
                # Higher noise -> more subsamples
                noise_level = noise_var / np.var(returns)
                if noise_level > 0.1:
                    calibrated_config.subsampling_factor = 10
                elif noise_level > 0.05:
                    calibrated_config.subsampling_factor = 5
                else:
                    calibrated_config.subsampling_factor = 3
                
                logger.info(
                    f"Calibrated subsampling: enabled with factor {calibrated_config.subsampling_factor}"
                )
            else:
                calibrated_config.use_subsampling = False
                logger.info("Calibrated subsampling: disabled (low noise)")
        except Exception as e:
            logger.warning(f"Subsampling calibration failed: {str(e)}")
        
        # Determine whether to apply noise correction
        try:
            # If noise variance is very significant, enable noise correction
            if noise_var > 1e-5:
                calibrated_config.apply_noise_correction = True
                logger.info("Calibrated noise correction: enabled (high noise)")
            else:
                calibrated_config.apply_noise_correction = False
                logger.info("Calibrated noise correction: disabled (low noise)")
        except Exception as e:
            logger.warning(f"Noise correction calibration failed: {str(e)}")
        
        return calibrated_config
    
    def plot_volatility(self, 
                       result: Optional[RealizedEstimatorResult] = None,
                       annualize: Optional[bool] = None,
                       figsize: Tuple[int, int] = (10, 6),
                       title: Optional[str] = None,
                       **kwargs: Any) -> Any:
        """Plot realized volatility from estimation results.
        
        Args:
            result: Estimation results (if None, uses the most recent results)
            annualize: Whether to annualize the volatility (overrides config)
            figsize: Figure size as (width, height)
            title: Plot title (if None, a default title is used)
            **kwargs: Additional keyword arguments for plotting
        
        Returns:
            matplotlib.figure.Figure: The created figure
        
        Raises:
            RuntimeError: If no results are available
            ImportError: If matplotlib is not installed
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib is required for plotting. Install it with 'pip install matplotlib'.")
        
        # Get results
        if result is None:
            if self._results is None:
                raise RuntimeError("No results available. Call fit() first or provide result.")
            result = cast(RealizedEstimatorResult, self._results)
        
        # Determine whether to annualize
        do_annualize = self._config.annualize if annualize is None else annualize
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get volatility (square root of realized measure)
        if do_annualize:
            volatility = np.sqrt(result.realized_measure * self._config.annualization_factor)
            ylabel = "Annualized Realized Volatility"
        else:
            volatility = np.sqrt(result.realized_measure)
            ylabel = "Realized Volatility"
        
        # Create x-axis values (timestamps if available, otherwise indices)
        if isinstance(result.times, np.ndarray) and len(result.times) > 0:
            # Try to convert times to datetime
            try:
                import pandas as pd
                # Check if times are already datetime-like
                if isinstance(result.times[0], (pd.Timestamp, np.datetime64)):
                    x = result.times
                else:
                    # Try to interpret as seconds since epoch
                    x = pd.to_datetime(result.times, unit='s')
            except:
                # Fall back to using times as is
                x = result.times
        else:
            # Use indices if times are not available
            x = np.arange(len(volatility))
        
        # Plot volatility
        ax.plot(x, volatility, **kwargs)
        
        # Set labels and title
        ax.set_xlabel("Time")
        ax.set_ylabel(ylabel)
        if title is None:
            title = f"{self._name} - {ylabel}"
        ax.set_title(title)
        
        # Format x-axis if datetime
        if hasattr(ax, 'xaxis') and hasattr(ax.xaxis, 'set_major_formatter'):
            try:
                from matplotlib.dates import DateFormatter
                ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
                fig.autofmt_xdate()
            except:
                pass
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        fig.tight_layout()
        
        return fig
    
    def to_pandas(self, 
                 result: Optional[RealizedEstimatorResult] = None,
                 annualize: Optional[bool] = None) -> pd.DataFrame:
        """Convert realized volatility results to a pandas DataFrame.
        
        Args:
            result: Estimation results (if None, uses the most recent results)
            annualize: Whether to annualize the volatility (overrides config)
        
        Returns:
            pd.DataFrame: DataFrame containing realized volatility results
        
        Raises:
            RuntimeError: If no results are available
        """
        # Get results
        if result is None:
            if self._results is None:
                raise RuntimeError("No results available. Call fit() first or provide result.")
            result = cast(RealizedEstimatorResult, self._results)
        
        # Determine whether to annualize
        do_annualize = self._config.annualize if annualize is None else annualize
        
        # Create DataFrame
        if isinstance(result.times, np.ndarray) and len(result.times) > 0:
            # Try to convert times to datetime index
            try:
                import pandas as pd
                # Check if times are already datetime-like
                if isinstance(result.times[0], (pd.Timestamp, np.datetime64)):
                    index = pd.DatetimeIndex(result.times)
                else:
                    # Try to interpret as seconds since epoch
                    index = pd.to_datetime(result.times, unit='s')
            except:
                # Fall back to using times as is
                index = result.times
        else:
            # Use range index if times are not available
            index = pd.RangeIndex(len(result.realized_measure))
        
        # Create DataFrame with realized measure
        df = pd.DataFrame(
            {'realized_variance': result.realized_measure},
            index=index
        )
        
        # Add realized volatility (square root of realized measure)
        df['realized_volatility'] = np.sqrt(df['realized_variance'])
        
        # Add annualized measures if requested
        if do_annualize:
            annualization_factor = self._config.annualization_factor
            df['annualized_variance'] = df['realized_variance'] * annualization_factor
            df['annualized_volatility'] = df['realized_volatility'] * np.sqrt(annualization_factor)
        
        # Add returns if available
        if result.returns is not None and len(result.returns) == len(df):
            df['returns'] = result.returns
        
        return df
    
    @classmethod
    def from_pandas(cls, 
                   data: pd.DataFrame,
                   price_col: str = 'price',
                   time_col: Optional[str] = None,
                   **kwargs: Any) -> 'RealizedVariance':
        """Create a realized variance estimator from pandas DataFrame.
        
        Args:
            data: DataFrame containing price and time data
            price_col: Name of the column containing price data
            time_col: Name of the column containing time data (if None, uses index)
            **kwargs: Additional keyword arguments for RealizedVarianceConfig
        
        Returns:
            RealizedVariance: Realized variance estimator
        
        Raises:
            ValueError: If the DataFrame doesn't contain the required columns
        """
        # Validate DataFrame
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        
        if price_col not in data.columns:
            raise ValueError(f"DataFrame must contain a '{price_col}' column")
        
        # Extract prices
        prices = data[price_col].values
        
        # Extract times
        if time_col is not None:
            if time_col not in data.columns:
                raise ValueError(f"DataFrame must contain a '{time_col}' column")
            times = data[time_col].values
        else:
            # Use index as times
            if isinstance(data.index, pd.DatetimeIndex):
                # Convert datetime index to Unix timestamps (seconds since epoch)
                times = data.index.astype('int64') / 1e9
            else:
                # Use index values directly
                times = data.index.values
        
        # Create configuration
        config = RealizedVarianceConfig(**kwargs)
        
        # Create estimator
        estimator = cls(config=config)
        
        return estimator
    
    def __str__(self) -> str:
        """Generate a string representation of the estimator.
        
        Returns:
            str: A string representation of the estimator
        """
        if not self._fitted:
            return f"RealizedVariance(fitted=False, config={self._config})"
        
        # Include basic results if available
        if self._results is not None:
            rv = self._results.realized_measure[0]
            vol = np.sqrt(rv)
            if self._config.annualize:
                ann_factor = self._config.annualization_factor
                ann_vol = vol * np.sqrt(ann_factor)
                return (f"RealizedVariance(fitted=True, "
                        f"RV={rv:.6f}, Vol={vol:.6f}, AnnVol={ann_vol:.6f})")
            else:
                return f"RealizedVariance(fitted=True, RV={rv:.6f}, Vol={vol:.6f})"
        
        return f"RealizedVariance(fitted=True)"
    
    def __repr__(self) -> str:
        """Generate a detailed string representation of the estimator.
        
        Returns:
            str: A detailed string representation of the estimator
        """
        return (f"RealizedVariance(name='{self._name}', fitted={self._fitted}, "
                f"config={self._config})")


# Register Numba-accelerated functions if available
def _register_numba_functions() -> None:
    """
    Register Numba JIT-compiled functions for realized variance.
    
    This function is called during module initialization to register
    performance-critical functions for JIT compilation if Numba is available.
    """
    if HAS_NUMBA:
        # The functions are already decorated with @jit, so no additional
        # registration is needed here. This function is kept for consistency
        # with the module structure and potential future enhancements.
        logger.debug("Realized variance Numba JIT functions registered")
    else:
        logger.info("Numba not available. Realized variance will use pure NumPy implementations.")


# Initialize the module
_register_numba_functions()
