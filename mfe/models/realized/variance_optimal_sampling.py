'''
Optimal sampling for realized variance estimation based on the Bandi-Russell methodology.

This module implements the Bandi-Russell approach for determining the optimal sampling
frequency in realized volatility estimation. The method automatically balances the
tradeoff between microstructure noise (which dominates at high frequencies) and
 discretization error (which dominates at low frequencies) to minimize the mean squared
 error of the realized variance estimator.

The implementation leverages NumPy for efficient numerical operations, SciPy for
optimization, and Pandas for handling time series data with datetime indices.
Performance-critical calculations are accelerated using Numba's JIT compilation.
The module provides comprehensive type hints, parameter validation, and visualization
capabilities for optimal sampling selection.

Classes:
    OptimalSamplingConfig: Configuration parameters for optimal sampling
    RealizedVarianceOptimalSampling: Estimator for realized variance with optimal sampling
'''

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, cast
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy import optimize

from ...core.base import ModelBase
from ...core.parameters import ParameterBase, ParameterError, validate_positive, validate_non_negative
from ...core.exceptions import DimensionError, NumericError
from .base import BaseRealizedEstimator, RealizedEstimatorConfig, RealizedEstimatorResult
from .variance import RealizedVariance, RealizedVarianceConfig
from .utils import (
    compute_returns, compute_realized_variance, compute_realized_quarticity,
    noise_variance, sample_prices, align_time, seconds2unit, unit2seconds
)
from .price_filter import price_filter

# Set up module-level logger
logger = logging.getLogger("mfe.models.realized.variance_optimal_sampling")

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
    logger.debug("Numba available for optimal sampling acceleration")
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    HAS_NUMBA = False
    logger.info("Numba not available. Optimal sampling will use pure NumPy implementation.")

# Try to import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.info("Matplotlib not available. Visualization functions will be disabled.")


@dataclass
class OptimalSamplingConfig(RealizedVarianceConfig):
    """Configuration parameters for optimal sampling realized variance estimator.
    
    This class extends RealizedVarianceConfig with parameters specific to
    the optimal sampling methodology.
    
    Attributes:
        sampling_frequency: Sampling frequency for price data (if None, determined automatically)
        annualize: Whether to annualize the volatility estimate
        annualization_factor: Factor to use for annualization (e.g., 252 for daily data)
        return_type: Type of returns to compute ('log', 'simple')
        use_subsampling: Whether to use subsampling for noise reduction
        subsampling_factor: Factor for subsampling (number of subsamples)
        apply_noise_correction: Whether to apply microstructure noise correction
        time_unit: Unit of time for high-frequency data ('seconds', 'minutes', etc.)
        interpolation_method: Method for interpolating prices ('previous', 'linear', 'cubic')
        optimization_method: Method for determining optimal sampling ('mse', 'signature', 'scale')
        grid_points: Number of sampling frequencies to evaluate in grid search
        min_observations: Minimum number of observations required after sampling
        adaptive_sampling: Whether to use adaptive sampling based on local characteristics
        local_window_size: Window size for local characteristic estimation in adaptive sampling
    """
    
    optimization_method: str = 'mse'
    grid_points: int = 20
    min_observations: int = 10
    adaptive_sampling: bool = False
    local_window_size: int = 100
    
    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        super().__post_init__()
        
        # Validate optimization_method
        if self.optimization_method not in ['mse', 'signature', 'scale']:
            raise ParameterError(
                f"optimization_method must be 'mse', 'signature', or 'scale', "
                f"got {self.optimization_method}"
            )
        
        # Validate grid_points
        if not isinstance(self.grid_points, int) or self.grid_points < 2:
            raise ParameterError(f"grid_points must be an integer >= 2, got {self.grid_points}")
        
        # Validate min_observations
        if not isinstance(self.min_observations, int) or self.min_observations < 2:
            raise ParameterError(f"min_observations must be an integer >= 2, got {self.min_observations}")
        
        # Validate local_window_size if adaptive_sampling is True
        if self.adaptive_sampling:
            if not isinstance(self.local_window_size, int) or self.local_window_size < 10:
                raise ParameterError(
                    f"local_window_size must be an integer >= 10 when adaptive_sampling is True, "
                    f"got {self.local_window_size}"
                )


@jit(nopython=True, cache=True)
def _estimate_noise_variance_numba(returns: np.ndarray) -> float:
    """
    Numba-accelerated implementation of noise variance estimation.
    
    Args:
        returns: Array of returns
        
    Returns:
        Estimated noise variance
    """
    # Estimate noise variance using the autocovariance method
    # Noise variance is -0.5 * first-order autocovariance
    n = len(returns)
    acov = 0.0
    
    for i in range(n-1):
        acov += returns[i] * returns[i+1]
    
    acov /= (n-1)
    noise_var = -0.5 * acov
    
    # If noise_var is negative, use alternative method
    if noise_var <= 0:
        # Use first-order method: 0.5 * mean squared returns
        noise_var = 0.0
        for i in range(n):
            noise_var += returns[i]**2
        
        noise_var *= 0.5 / n
    
    return noise_var


@jit(nopython=True, cache=True)
def _estimate_integrated_quarticity_numba(returns: np.ndarray) -> float:
    """
    Numba-accelerated implementation of integrated quarticity estimation.
    
    Args:
        returns: Array of returns
        
    Returns:
        Estimated integrated quarticity
    """
    n = len(returns)
    quarticity = 0.0
    
    for i in range(n):
        quarticity += returns[i]**4
    
    # Scale by n/3 for consistency with asymptotic theory
    quarticity *= n / 3
    
    return quarticity


@jit(nopython=True, cache=True)
def _compute_optimal_sampling_mse_numba(
    noise_var: float, 
    integrated_quarticity: float, 
    n: int
) -> float:
    """
    Numba-accelerated implementation of optimal sampling frequency computation.
    
    Args:
        noise_var: Estimated noise variance
        integrated_quarticity: Estimated integrated quarticity
        n: Number of observations
        
    Returns:
        Optimal sampling frequency (number of observations to skip)
    """
    # Bandi-Russell formula for optimal sampling frequency
    # Optimal sampling minimizes MSE = (bias^2 + variance)
    # bias = 2*n*noise_var/k, variance = 2*integrated_quarticity*k/n^2
    # Optimal k = (n^2 * noise_var / integrated_quarticity)^(1/3)
    
    if integrated_quarticity <= 0 or noise_var <= 0:
        # Default to sqrt(n) if parameters are invalid
        return max(1, int(np.sqrt(n)))
    
    # Compute optimal number of observations to skip
    k_opt = (n**2 * noise_var / integrated_quarticity)**(1/3)
    
    # Ensure k_opt is at least 1
    k_opt = max(1, k_opt)
    
    # Round to nearest integer
    return round(k_opt)


class RealizedVarianceOptimalSampling(BaseRealizedEstimator):
    """Realized Variance Estimator with Optimal Sampling.
    
    This class implements the Bandi-Russell approach for determining the optimal
    sampling frequency in realized volatility estimation. The method automatically
    balances the tradeoff between microstructure noise and discretization error
    to minimize the mean squared error of the realized variance estimator.
    
    The estimator can operate in automatic mode, where it determines the optimal
    sampling frequency based on the data characteristics, or with a user-specified
    sampling frequency. It also supports adaptive sampling, where the sampling
    frequency varies based on local data characteristics.
    
    Attributes:
        config: Configuration parameters for the estimator
        name: A descriptive name for the estimator
    """
    
    def __init__(self, 
                config: Optional[OptimalSamplingConfig] = None, 
                name: str = "RealizedVarianceOptimalSampling"):
        """Initialize the realized variance estimator with optimal sampling.
        
        Args:
            config: Configuration parameters for the estimator
            name: A descriptive name for the estimator
        """
        # Use OptimalSamplingConfig if no config is provided
        if config is None:
            config = OptimalSamplingConfig()
        elif not isinstance(config, OptimalSamplingConfig):
            # If a RealizedEstimatorConfig is provided, convert it to OptimalSamplingConfig
            if isinstance(config, RealizedEstimatorConfig):
                config_dict = config.to_dict()
                config_dict.setdefault('optimization_method', 'mse')
                config_dict.setdefault('grid_points', 20)
                config_dict.setdefault('min_observations', 10)
                config_dict.setdefault('adaptive_sampling', False)
                config_dict.setdefault('local_window_size', 100)
                config = OptimalSamplingConfig(**config_dict)
            else:
                raise TypeError(f"config must be an OptimalSamplingConfig, got {type(config)}")
        
        super().__init__(config=config, name=name)
        self._optimal_sampling_frequency: Optional[float] = None
        self._noise_variance: Optional[float] = None
        self._integrated_quarticity: Optional[float] = None
        self._sampling_grid: Optional[np.ndarray] = None
        self._mse_values: Optional[np.ndarray] = None
        self._adaptive_frequencies: Optional[np.ndarray] = None
    
    @property
    def optimal_sampling_frequency(self) -> Optional[float]:
        """Get the optimal sampling frequency.
        
        Returns:
            Optional[float]: The optimal sampling frequency if determined,
                            None otherwise
        """
        return self._optimal_sampling_frequency
    
    @property
    def noise_variance(self) -> Optional[float]:
        """Get the estimated noise variance.
        
        Returns:
            Optional[float]: The estimated noise variance if available,
                            None otherwise
        """
        return self._noise_variance
    
    @property
    def integrated_quarticity(self) -> Optional[float]:
        """Get the estimated integrated quarticity.
        
        Returns:
            Optional[float]: The estimated integrated quarticity if available,
                            None otherwise
        """
        return self._integrated_quarticity
    
    def _compute_realized_measure(self, 
                                 prices: np.ndarray, 
                                 times: np.ndarray, 
                                 returns: np.ndarray,
                                 **kwargs: Any) -> np.ndarray:
        """Compute realized variance with optimal sampling from the preprocessed data.
        
        This method implements the core realized variance calculation with optimal
        sampling. It determines the optimal sampling frequency based on the data
        characteristics and then computes the realized variance using that frequency.
        
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
        # If sampling_frequency is already specified in config, use it
        if self._config.sampling_frequency is not None:
            # Use the standard RealizedVariance implementation
            rv_estimator = RealizedVariance(config=self._config)
            rv = rv_estimator._compute_realized_measure(prices, times, returns, **kwargs)
            return rv
        
        # Otherwise, determine the optimal sampling frequency
        if self._config.adaptive_sampling:
            # Adaptive sampling based on local characteristics
            rv = self._compute_adaptive_sampling(prices, times, returns, **kwargs)
        else:
            # Global optimal sampling
            optimal_freq = self._determine_optimal_sampling(returns, **kwargs)
            
            # Store the optimal sampling frequency
            self._optimal_sampling_frequency = optimal_freq
            
            # Apply the optimal sampling
            try:
                # Convert optimal_freq to appropriate format based on time_unit
                if isinstance(optimal_freq, (int, float)):
                    # For numeric optimal_freq, interpret as number of observations to skip
                    skip = max(1, int(optimal_freq))
                    sampled_returns = returns[::skip]
                    
                    # Compute realized variance from sampled returns
                    rv_value = compute_realized_variance(sampled_returns)
                    
                    # Scale by the sampling factor to maintain consistency
                    rv_value *= (len(returns) / len(sampled_returns))
                    
                    rv = np.array([rv_value])
                else:
                    # For string optimal_freq, use price_filter
                    sampled_prices, sampled_times = price_filter(
                        prices, times, 
                        sample_freq=optimal_freq,
                        time_unit=self._config.time_unit,
                        interpolation_method=self._config.interpolation_method
                    )
                    
                    # Compute returns from sampled prices
                    sampled_returns = compute_returns(sampled_prices, self._config.return_type)
                    
                    # Compute realized variance from sampled returns
                    rv_value = compute_realized_variance(sampled_returns)
                    
                    # Scale by the sampling factor to maintain consistency
                    rv_value *= (len(returns) / len(sampled_returns))
                    
                    rv = np.array([rv_value])
            except Exception as e:
                logger.warning(f"Optimal sampling application failed: {str(e)}. Using original returns.")
                rv = np.array([compute_realized_variance(returns)])
        
        # Apply noise correction if requested
        if self._config.apply_noise_correction and self._noise_variance is not None:
            # Correct for noise bias (2n * noise_var)
            n = len(returns)
            correction = 2 * n * self._noise_variance
            # Subtract correction (ensure result is non-negative)
            rv_corrected = max(0, rv[0] - correction)
            rv = np.array([rv_corrected])
            logger.debug(f"Applied noise correction: {correction:.6e}")
        
        return rv
    
    def _determine_optimal_sampling(self, returns: np.ndarray, **kwargs: Any) -> Union[float, str]:
        """Determine the optimal sampling frequency based on the data characteristics.
        
        Args:
            returns: Returns computed from prices
            **kwargs: Additional keyword arguments for computation
        
        Returns:
            Union[float, str]: Optimal sampling frequency
        
        Raises:
            ValueError: If computation fails
        """
        # Estimate noise variance
        if HAS_NUMBA:
            self._noise_variance = _estimate_noise_variance_numba(returns)
        else:
            self._noise_variance = noise_variance(returns, method='autocovariance')
        
        logger.debug(f"Estimated noise variance: {self._noise_variance:.6e}")
        
        # Use the specified optimization method
        if self._config.optimization_method == 'mse':
            # Mean squared error method (Bandi-Russell)
            
            # Estimate integrated quarticity
            if HAS_NUMBA:
                self._integrated_quarticity = _estimate_integrated_quarticity_numba(returns)
            else:
                self._integrated_quarticity = compute_realized_quarticity(returns)
            
            logger.debug(f"Estimated integrated quarticity: {self._integrated_quarticity:.6e}")
            
            # Compute optimal sampling frequency
            n = len(returns)
            
            if HAS_NUMBA:
                optimal_skip = _compute_optimal_sampling_mse_numba(
                    self._noise_variance, self._integrated_quarticity, n
                )
            else:
                # Bandi-Russell formula for optimal sampling frequency
                # Optimal sampling minimizes MSE = (bias^2 + variance)
                # bias = 2*n*noise_var/k, variance = 2*integrated_quarticity*k/n^2
                # Optimal k = (n^2 * noise_var / integrated_quarticity)^(1/3)
                
                if self._integrated_quarticity <= 0 or self._noise_variance <= 0:
                    # Default to sqrt(n) if parameters are invalid
                    optimal_skip = max(1, int(np.sqrt(n)))
                else:
                    # Compute optimal number of observations to skip
                    k_opt = (n**2 * self._noise_variance / self._integrated_quarticity)**(1/3)
                    
                    # Ensure k_opt is at least 1
                    k_opt = max(1, k_opt)
                    
                    # Round to nearest integer
                    optimal_skip = round(k_opt)
            
            logger.info(f"Optimal sampling: skip every {optimal_skip} observations")
            
            # Store the grid of sampling frequencies and MSE values for visualization
            self._sampling_grid = np.arange(1, self._config.grid_points + 1)
            self._mse_values = np.zeros(self._config.grid_points)
            
            for i, k in enumerate(self._sampling_grid):
                # Compute MSE for each sampling frequency
                bias_squared = (2 * n * self._noise_variance / k)**2
                variance = 2 * self._integrated_quarticity * k / n**2
                self._mse_values[i] = bias_squared + variance
            
            # Return the optimal sampling frequency
            return float(optimal_skip)
        
        elif self._config.optimization_method == 'signature':
            # Signature plot method
            from .utils import compute_optimal_sampling
            
            # Compute optimal sampling frequency using the signature plot method
            try:
                optimal_freq = compute_optimal_sampling(
                    self._prices, self._times, method='signature', max_points=self._config.grid_points
                )
                
                logger.info(f"Optimal sampling frequency (signature): {optimal_freq}")
                
                return optimal_freq
            except Exception as e:
                logger.warning(f"Signature plot method failed: {str(e)}. Using default sampling.")
                # Default to sqrt(n) observations
                n = len(returns)
                return max(1, int(np.sqrt(n)))
        
        elif self._config.optimization_method == 'scale':
            # Scale-based method
            from .utils import compute_optimal_sampling
            
            # Compute optimal sampling frequency using the scale-based method
            try:
                optimal_freq = compute_optimal_sampling(
                    self._prices, self._times, method='scale', max_points=self._config.grid_points
                )
                
                logger.info(f"Optimal sampling frequency (scale): {optimal_freq}")
                
                return optimal_freq
            except Exception as e:
                logger.warning(f"Scale-based method failed: {str(e)}. Using default sampling.")
                # Default to sqrt(n) observations
                n = len(returns)
                return max(1, int(np.sqrt(n)))
        
        else:
            raise ValueError(f"Unrecognized optimization method: {self._config.optimization_method}")
    
    def _compute_adaptive_sampling(self, 
                                  prices: np.ndarray, 
                                  times: np.ndarray, 
                                  returns: np.ndarray,
                                  **kwargs: Any) -> np.ndarray:
        """Compute realized variance with adaptive sampling.
        
        This method implements adaptive sampling, where the sampling frequency
        varies based on local data characteristics. It divides the data into
        windows and determines the optimal sampling frequency for each window.
        
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
        n = len(returns)
        window_size = min(self._config.local_window_size, n // 5)
        
        if window_size < 10:
            logger.warning("Not enough data for adaptive sampling. Using global optimal sampling.")
            # Fall back to global optimal sampling
            return self._compute_realized_measure(prices, times, returns, **kwargs)
        
        # Determine number of windows
        n_windows = max(2, n // window_size)
        
        # Initialize arrays for adaptive sampling
        window_rv = np.zeros(n_windows)
        window_weights = np.zeros(n_windows)
        self._adaptive_frequencies = np.zeros(n_windows)
        
        # Process each window
        for i in range(n_windows):
            # Determine window boundaries
            start_idx = i * window_size
            end_idx = min((i + 1) * window_size, n)
            
            # Extract window data
            window_returns = returns[start_idx:end_idx]
            window_prices = prices[start_idx:end_idx]
            window_times = times[start_idx:end_idx]
            
            # Skip windows with too few observations
            if len(window_returns) < self._config.min_observations:
                continue
            
            # Determine optimal sampling for this window
            try:
                # Estimate noise variance for this window
                window_noise_var = noise_variance(window_returns)
                
                # Estimate integrated quarticity for this window
                window_quarticity = compute_realized_quarticity(window_returns)
                
                # Compute optimal sampling frequency for this window
                window_n = len(window_returns)
                
                if window_quarticity <= 0 or window_noise_var <= 0:
                    # Default to sqrt(n) if parameters are invalid
                    window_optimal_skip = max(1, int(np.sqrt(window_n)))
                else:
                    # Compute optimal number of observations to skip
                    k_opt = (window_n**2 * window_noise_var / window_quarticity)**(1/3)
                    
                    # Ensure k_opt is at least 1
                    k_opt = max(1, k_opt)
                    
                    # Round to nearest integer
                    window_optimal_skip = round(k_opt)
                
                # Store the optimal sampling frequency for this window
                self._adaptive_frequencies[i] = window_optimal_skip
                
                # Apply the optimal sampling to this window
                window_sampled_returns = window_returns[::window_optimal_skip]
                
                # Compute realized variance for this window
                window_rv[i] = compute_realized_variance(window_sampled_returns)
                
                # Scale by the sampling factor to maintain consistency
                window_rv[i] *= (len(window_returns) / len(window_sampled_returns))
                
                # Set weight based on window size
                window_weights[i] = len(window_returns)
            
            except Exception as e:
                logger.warning(f"Adaptive sampling failed for window {i}: {str(e)}. Skipping window.")
                continue
        
        # Combine window estimates (weighted average)
        if np.sum(window_weights) > 0:
            # Normalize weights
            window_weights /= np.sum(window_weights)
            
            # Compute weighted average
            rv_value = np.sum(window_rv * window_weights)
            
            logger.info(f"Adaptive sampling: {n_windows} windows, average frequency: {np.mean(self._adaptive_frequencies[self._adaptive_frequencies > 0]):.2f}")
            
            return np.array([rv_value])
        else:
            logger.warning("Adaptive sampling failed for all windows. Using global optimal sampling.")
            # Fall back to global optimal sampling
            return self._compute_realized_measure(prices, times, returns, **kwargs)
    
    def fit(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> RealizedEstimatorResult:
        """Fit the realized variance estimator with optimal sampling to the provided data.
        
        This method validates the input data, preprocesses it according to the
        estimator configuration, and then computes the realized variance with
        optimal sampling.
        
        Args:
            data: The data to fit the estimator to, as a tuple of (prices, times)
            **kwargs: Additional keyword arguments for estimation
        
        Returns:
            RealizedEstimatorResult: The estimation results
        
        Raises:
            ValueError: If the data is invalid
            RuntimeError: If the estimation fails
        """
        result = super().fit(data, **kwargs)
        
        # Add optimal sampling information to the result
        if hasattr(result, 'noise_variance') and self._noise_variance is not None:
            result.noise_variance = self._noise_variance
        
        return result
    
    async def fit_async(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> RealizedEstimatorResult:
        """Asynchronously fit the realized variance estimator with optimal sampling.
        
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
    
    def calibrate(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> OptimalSamplingConfig:
        """Calibrate the estimator configuration based on the provided data.
        
        This method analyzes the input data and determines optimal configuration
        parameters for the realized variance estimator with optimal sampling.
        
        Args:
            data: The data to calibrate the estimator with, as a tuple of (prices, times)
            **kwargs: Additional keyword arguments for calibration
        
        Returns:
            OptimalSamplingConfig: Calibrated configuration
        
        Raises:
            ValueError: If the data is invalid
        """
        # Validate input data
        self.validate_data(data)
        prices, times = data
        
        # Create a copy of the current configuration
        calibrated_config = cast(OptimalSamplingConfig, self._config.copy())
        
        # Determine optimal sampling method based on data characteristics
        try:
            # Compute returns
            returns = compute_returns(prices, self._config.return_type)
            
            # Estimate noise variance
            noise_var = noise_variance(returns)
            
            # Estimate integrated quarticity
            quarticity = compute_realized_quarticity(returns)
            
            # Determine optimal sampling method based on noise level
            noise_level = noise_var / np.var(returns)
            
            if noise_level > 0.1:
                # High noise, use MSE method
                calibrated_config.optimization_method = 'mse'
                logger.info("Calibrated optimization method: 'mse' (high noise)")
            elif noise_level > 0.05:
                # Moderate noise, use signature plot method
                calibrated_config.optimization_method = 'signature'
                logger.info("Calibrated optimization method: 'signature' (moderate noise)")
            else:
                # Low noise, use scale-based method
                calibrated_config.optimization_method = 'scale'
                logger.info("Calibrated optimization method: 'scale' (low noise)")
            
            # Determine whether to use adaptive sampling
            n = len(returns)
            
            if n > 500 and np.std(np.abs(returns)) / np.mean(np.abs(returns)) > 2.0:
                # Large dataset with high variability, use adaptive sampling
                calibrated_config.adaptive_sampling = True
                
                # Set appropriate window size
                if n > 2000:
                    calibrated_config.local_window_size = 200
                elif n > 1000:
                    calibrated_config.local_window_size = 100
                else:
                    calibrated_config.local_window_size = 50
                
                logger.info(f"Calibrated adaptive sampling: enabled with window size {calibrated_config.local_window_size}")
            else:
                # Small dataset or low variability, use global sampling
                calibrated_config.adaptive_sampling = False
                logger.info("Calibrated adaptive sampling: disabled")
            
            # Determine whether to apply noise correction
            if noise_level > 0.05:
                # Significant noise, enable correction
                calibrated_config.apply_noise_correction = True
                logger.info("Calibrated noise correction: enabled (significant noise)")
            else:
                # Low noise, disable correction
                calibrated_config.apply_noise_correction = False
                logger.info("Calibrated noise correction: disabled (low noise)")
            
            # Determine whether to use subsampling
            if noise_level > 0.1:
                # High noise, enable subsampling
                calibrated_config.use_subsampling = True
                
                # Set subsampling factor based on noise level
                if noise_level > 0.2:
                    calibrated_config.subsampling_factor = 10
                elif noise_level > 0.15:
                    calibrated_config.subsampling_factor = 5
                else:
                    calibrated_config.subsampling_factor = 3
                
                logger.info(f"Calibrated subsampling: enabled with factor {calibrated_config.subsampling_factor}")
            else:
                # Low noise, disable subsampling
                calibrated_config.use_subsampling = False
                logger.info("Calibrated subsampling: disabled (low noise)")
        
        except Exception as e:
            logger.warning(f"Calibration failed: {str(e)}. Using default configuration.")
        
        return calibrated_config
    
    def plot_mse_curve(self, 
                      figsize: Tuple[int, int] = (10, 6),
                      title: Optional[str] = None,
                      save_path: Optional[str] = None) -> Optional[Any]:
        """Plot the mean squared error curve for different sampling frequencies.
        
        Args:
            figsize: Figure size as (width, height) in inches
            title: Plot title (if None, a default title is used)
            save_path: Path to save the figure (if None, figure is displayed)
            
        Returns:
            Matplotlib figure object if matplotlib is available, None otherwise
            
        Raises:
            RuntimeError: If the estimator has not been fitted or if MSE values are not available
        """
        if not HAS_MATPLOTLIB:
            logger.warning("Matplotlib is not available. Cannot create visualization.")
            return None
        
        if not self._fitted:
            raise RuntimeError("Estimator has not been fitted. Call fit() first.")
        
        if self._sampling_grid is None or self._mse_values is None:
            raise RuntimeError("MSE values are not available. Use 'mse' optimization method.")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot MSE curve
        ax.plot(self._sampling_grid, self._mse_values, 'o-', linewidth=2)
        
        # Mark optimal sampling frequency
        if self._optimal_sampling_frequency is not None:
            optimal_idx = np.argmin(self._mse_values)
            ax.plot(self._sampling_grid[optimal_idx], self._mse_values[optimal_idx], 'ro', 
                   markersize=10, label=f'Optimal: {self._optimal_sampling_frequency:.0f}')
        
        # Set labels and title
        ax.set_xlabel('Sampling Frequency (observations to skip)')
        ax.set_ylabel('Mean Squared Error')
        if title is None:
            title = 'Mean Squared Error vs. Sampling Frequency'
        ax.set_title(title)
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        if self._optimal_sampling_frequency is not None:
            ax.legend()
        
        # Set x-axis to integer values
        ax.set_xticks(self._sampling_grid)
        
        # Tight layout
        fig.tight_layout()
        
        # Save or display the figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        return fig
    
    def plot_adaptive_sampling(self, 
                              figsize: Tuple[int, int] = (10, 6),
                              title: Optional[str] = None,
                              save_path: Optional[str] = None) -> Optional[Any]:
        """Plot the adaptive sampling frequencies across different windows.
        
        Args:
            figsize: Figure size as (width, height) in inches
            title: Plot title (if None, a default title is used)
            save_path: Path to save the figure (if None, figure is displayed)
            
        Returns:
            Matplotlib figure object if matplotlib is available, None otherwise
            
        Raises:
            RuntimeError: If the estimator has not been fitted or if adaptive frequencies are not available
        """
        if not HAS_MATPLOTLIB:
            logger.warning("Matplotlib is not available. Cannot create visualization.")
            return None
        
        if not self._fitted:
            raise RuntimeError("Estimator has not been fitted. Call fit() first.")
        
        if self._adaptive_frequencies is None or not self._config.adaptive_sampling:
            raise RuntimeError("Adaptive sampling frequencies are not available. Use adaptive sampling.")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot adaptive sampling frequencies
        valid_freqs = self._adaptive_frequencies[self._adaptive_frequencies > 0]
        window_indices = np.arange(len(valid_freqs))
        
        ax.bar(window_indices, valid_freqs, alpha=0.7)
        
        # Add horizontal line for average frequency
        avg_freq = np.mean(valid_freqs)
        ax.axhline(avg_freq, color='r', linestyle='--', 
                  label=f'Average: {avg_freq:.2f}')
        
        # Set labels and title
        ax.set_xlabel('Window Index')
        ax.set_ylabel('Optimal Sampling Frequency')
        if title is None:
            title = 'Adaptive Sampling Frequencies Across Windows'
        ax.set_title(title)
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Tight layout
        fig.tight_layout()
        
        # Save or display the figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        
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
        
        # Add optimal sampling information
        if self._optimal_sampling_frequency is not None:
            df['optimal_sampling_frequency'] = self._optimal_sampling_frequency
        
        if self._noise_variance is not None:
            df['noise_variance'] = self._noise_variance
        
        if self._integrated_quarticity is not None:
            df['integrated_quarticity'] = self._integrated_quarticity
        
        return df
    
    @classmethod
    def from_pandas(cls, 
                   data: pd.DataFrame,
                   price_col: str = 'price',
                   time_col: Optional[str] = None,
                   **kwargs: Any) -> 'RealizedVarianceOptimalSampling':
        """Create a realized variance estimator with optimal sampling from pandas DataFrame.
        
        Args:
            data: DataFrame containing price and time data
            price_col: Name of the column containing price data
            time_col: Name of the column containing time data (if None, uses index)
            **kwargs: Additional keyword arguments for OptimalSamplingConfig
        
        Returns:
            RealizedVarianceOptimalSampling: Realized variance estimator with optimal sampling
        
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
        config = OptimalSamplingConfig(**kwargs)
        
        # Create estimator
        estimator = cls(config=config)
        
        return estimator
    
    def __str__(self) -> str:
        """Generate a string representation of the estimator.
        
        Returns:
            str: A string representation of the estimator
        """
        if not self._fitted:
            return f"RealizedVarianceOptimalSampling(fitted=False, config={self._config})"
        
        # Include basic results if available
        if self._results is not None:
            rv = self._results.realized_measure[0]
            vol = np.sqrt(rv)
            
            if self._optimal_sampling_frequency is not None:
                freq_str = f", OptFreq={self._optimal_sampling_frequency:.2f}"
            else:
                freq_str = ""
            
            if self._noise_variance is not None:
                noise_str = f", NoiseVar={self._noise_variance:.6e}"
            else:
                noise_str = ""
            
            if self._config.annualize:
                ann_factor = self._config.annualization_factor
                ann_vol = vol * np.sqrt(ann_factor)
                return (f"RealizedVarianceOptimalSampling(fitted=True, "
                        f"RV={rv:.6f}, Vol={vol:.6f}, AnnVol={ann_vol:.6f}{freq_str}{noise_str})")
            else:
                return (f"RealizedVarianceOptimalSampling(fitted=True, "
                        f"RV={rv:.6f}, Vol={vol:.6f}{freq_str}{noise_str})")
        
        return f"RealizedVarianceOptimalSampling(fitted=True)"
    
    def __repr__(self) -> str:
        """Generate a detailed string representation of the estimator.
        
        Returns:
            str: A detailed string representation of the estimator
        """
        return (f"RealizedVarianceOptimalSampling(name='{self._name}', fitted={self._fitted}, "
                f"config={self._config})")


# Register Numba-accelerated functions if available
def _register_numba_functions() -> None:
    """
    Register Numba JIT-compiled functions for optimal sampling.
    
    This function is called during module initialization to register
    performance-critical functions for JIT compilation if Numba is available.
    """
    if HAS_NUMBA:
        # The functions are already decorated with @jit, so no additional
        # registration is needed here. This function is kept for consistency
        # with the module structure and potential future enhancements.
        logger.debug("Optimal sampling Numba JIT functions registered")
    else:
        logger.info("Numba not available. Optimal sampling will use pure NumPy implementations.")


# Initialize the module
_register_numba_functions()
