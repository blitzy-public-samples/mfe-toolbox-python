# mfe/models/realized/preaveraged_variance.py
"""
Preaveraged realized variance estimator for noise-robust volatility measurement.

This module implements the preaveraged realized variance estimator, which uses weighted
moving averages to mitigate market microstructure noise effects in high-frequency data.
The preaveraging approach reduces the impact of microstructure noise by computing weighted
averages of returns over local windows, effectively smoothing out the noise while preserving
the underlying volatility signal.

The implementation leverages Numba's JIT compilation for performance-critical calculations
and supports various configuration options including different window sizes for preaveraging,
subsampling for additional noise reduction, and asynchronous processing for long computations.
The estimator inherits from NoiseRobustEstimator to provide comprehensive functionality for
handling microstructure noise in high-frequency data.
"""

import logging
import warnings
from typing import Any, Dict, Optional, Tuple, Union, cast, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .base import NoiseRobustEstimator, RealizedEstimatorConfig, RealizedEstimatorResult
from ._numba_core import _preaveraged_returns_core, _preaveraged_variance_core
from ...core.exceptions import ParameterError, NumericError
from ...core.parameters import validate_positive, validate_non_negative

# Set up module-level logger
logger = logging.getLogger("mfe.models.realized.preaveraged_variance")

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
    logger.debug("Numba available for preaveraged variance acceleration")
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    HAS_NUMBA = False
    logger.info("Numba not available. Preaveraged variance will use pure NumPy implementation.")


@jit(nopython=True, cache=True)
def _preaveraged_variance_core_impl(returns: np.ndarray, window_size: int) -> float:
    """
    Numba-accelerated core implementation of preaveraged variance.
    
    Args:
        returns: Array of returns
        window_size: Window size for preaveraging
        
    Returns:
        Preaveraged variance estimate
    """
    # Compute preaveraged returns
    n = len(returns)
    n_preaveraged = n - window_size + 1
    preaveraged = np.zeros(n_preaveraged)
    
    # Apply triangular kernel for preaveraging
    for i in range(n_preaveraged):
        for j in range(window_size):
            # Triangular kernel weight: j * (window_size - j) / window_size^2
            weight = j * (window_size - j) / (window_size**2)
            preaveraged[i] += weight * returns[i + j]
    
    # Compute variance of preaveraged returns
    pav = np.sum(preaveraged**2)
    
    # Scaling factor for asymptotic consistency
    # For triangular kernel, theta = 1/12
    theta = 1.0 / 12.0
    scale_factor = 1.0 / (window_size * (1.0 - 2.0 * theta))
    
    return pav * scale_factor


class PreaveragedVariance(NoiseRobustEstimator):
    """
    Preaveraged realized variance estimator for noise-robust volatility measurement.
    
    This class implements the preaveraged realized variance estimator, which uses weighted
    moving averages to mitigate market microstructure noise effects in high-frequency data.
    The preaveraging approach reduces the impact of microstructure noise by computing weighted
    averages of returns over local windows, effectively smoothing out the noise while preserving
    the underlying volatility signal.
    
    The estimator inherits from NoiseRobustEstimator to provide comprehensive functionality
    for handling microstructure noise in high-frequency data.
    
    Attributes:
        config: Configuration parameters for the estimator
        window_size: Window size for preaveraging
        preaveraged_returns: Preaveraged returns if the estimator has been fitted
        noise_variance: Estimated noise variance (if noise estimation was performed)
    """
    
    def __init__(self, config: Optional[RealizedEstimatorConfig] = None):
        """
        Initialize the preaveraged variance estimator.
        
        Args:
            config: Configuration parameters for the estimator
        """
        super().__init__(config=config, name="Preaveraged Variance")
        
        # Additional attributes specific to preaveraged variance
        self._window_size: int = 10  # Default window size
        self._preaveraged_returns: Optional[np.ndarray] = None
        self._debiased: bool = True  # Whether to apply finite sample bias correction
    
    @property
    def window_size(self) -> int:
        """
        Get the window size for preaveraging.
        
        Returns:
            int: The window size
        """
        return self._window_size
    
    @window_size.setter
    def window_size(self, value: int) -> None:
        """
        Set the window size for preaveraging.
        
        Args:
            value: New window size
            
        Raises:
            ValueError: If window_size is not a positive integer
        """
        if not isinstance(value, int) or value <= 0:
            raise ValueError("window_size must be a positive integer")
        
        self._window_size = value
        self._fitted = False  # Reset fitted state when window size changes
    
    @property
    def preaveraged_returns(self) -> Optional[np.ndarray]:
        """
        Get the preaveraged returns from the fitted estimator.
        
        Returns:
            Optional[np.ndarray]: The preaveraged returns if the estimator has been fitted,
                                 None otherwise
        
        Raises:
            RuntimeError: If the estimator has not been fitted
        """
        if not self._fitted:
            raise RuntimeError("Estimator has not been fitted. Call fit() first.")
        return self._preaveraged_returns
    
    def _compute_realized_measure(self, 
                                 prices: np.ndarray, 
                                 times: np.ndarray, 
                                 returns: np.ndarray,
                                 window_size: Optional[int] = None,
                                 debiased: bool = True,
                                 estimate_noise: bool = True,
                                 **kwargs: Any) -> np.ndarray:
        """
        Compute the preaveraged variance from the preprocessed data.
        
        This method implements the core preaveraged variance calculation, with
        options for debiasing, subsampling, and noise estimation.
        
        Args:
            prices: Preprocessed price data
            times: Preprocessed time points
            returns: Returns computed from prices
            window_size: Window size for preaveraging (overrides instance attribute)
            debiased: Whether to apply finite sample bias correction
            estimate_noise: Whether to estimate microstructure noise variance
            **kwargs: Additional keyword arguments for computation
        
        Returns:
            np.ndarray: Preaveraged variance
        
        Raises:
            ValueError: If computation fails
            NumericError: If numerical issues are encountered
        """
        # Use provided window_size or instance attribute
        if window_size is not None:
            if not isinstance(window_size, int) or window_size <= 0:
                raise ValueError("window_size must be a positive integer")
            self._window_size = window_size
        
        # Store debiased flag
        self._debiased = debiased
        
        try:
            # Estimate noise variance if requested
            if estimate_noise:
                noise_var = self.estimate_noise_variance(returns)
                self._noise_variance = noise_var
                logger.info(f"Estimated noise variance: {noise_var:.6e}")
            
            # Compute preaveraged returns
            preaveraged_returns = self._compute_preaveraged_returns(returns, self._window_size)
            self._preaveraged_returns = preaveraged_returns
            
            # Check if subsampling is enabled
            if self._config.use_subsampling and self._config.subsampling_factor > 1:
                # Compute subsampled preaveraged variance
                pav = self._compute_subsampled_preaveraged_variance(
                    returns, self._window_size, self._config.subsampling_factor
                )
            else:
                # Compute standard preaveraged variance
                pav = self._compute_preaveraged_variance(
                    returns, self._window_size, debiased
                )
            
            return pav
            
        except Exception as e:
            logger.error(f"Preaveraged variance computation failed: {str(e)}")
            raise ValueError(f"Preaveraged variance computation failed: {str(e)}") from e
    
    def _compute_preaveraged_returns(self, returns: np.ndarray, window_size: int) -> np.ndarray:
        """
        Compute preaveraged returns using a triangular kernel.
        
        Args:
            returns: Return series
            window_size: Window size for preaveraging
        
        Returns:
            np.ndarray: Preaveraged returns
        
        Raises:
            ValueError: If window_size is invalid
            NumericError: If numerical issues are encountered
        """
        # Validate window_size
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        if window_size >= len(returns):
            raise ValueError(f"window_size ({window_size}) must be less than returns length ({len(returns)})")
        
        try:
            # Use Numba-accelerated implementation
            preaveraged = _preaveraged_returns_core(returns, window_size)
            return preaveraged
            
        except Exception as e:
            logger.error(f"Preaveraged returns computation failed: {str(e)}")
            
            # Fallback to pure NumPy implementation if Numba fails
            logger.warning("Falling back to pure NumPy implementation")
            
            n = len(returns)
            n_preaveraged = n - window_size + 1
            preaveraged = np.zeros(n_preaveraged)
            
            # Apply triangular kernel for preaveraging
            for i in range(n_preaveraged):
                for j in range(window_size):
                    # Triangular kernel weight: j * (window_size - j) / window_size^2
                    weight = j * (window_size - j) / (window_size**2)
                    preaveraged[i] += weight * returns[i + j]
            
            return preaveraged
    
    def _compute_preaveraged_variance(self, 
                                     returns: np.ndarray, 
                                     window_size: int,
                                     debiased: bool = True) -> float:
        """
        Compute the standard preaveraged variance.
        
        Args:
            returns: Return series
            window_size: Window size for preaveraging
            debiased: Whether to apply finite sample bias correction
        
        Returns:
            float: Preaveraged variance
        
        Raises:
            NumericError: If numerical issues are encountered
        """
        try:
            # Use Numba-accelerated implementation
            if HAS_NUMBA:
                pav = _preaveraged_variance_core(returns, window_size)
            else:
                pav = _preaveraged_variance_core_impl(returns, window_size)
            
            # Apply finite sample bias correction if requested
            if not debiased:
                # Remove the finite sample correction applied in the core function
                n_preaveraged = len(returns) - window_size + 1
                pav = pav * ((n_preaveraged - 1) / n_preaveraged)
            
            return pav
            
        except Exception as e:
            logger.error(f"Preaveraged variance core computation failed: {str(e)}")
            
            # Fallback to pure NumPy implementation if Numba fails
            logger.warning("Falling back to pure NumPy implementation")
            
            # Compute preaveraged returns
            preaveraged = self._compute_preaveraged_returns(returns, window_size)
            
            # Compute variance of preaveraged returns
            n_preaveraged = len(preaveraged)
            pav = np.sum(preaveraged**2)
            
            # Apply finite sample correction if requested
            correction = n_preaveraged / (n_preaveraged - 1) if debiased else 1.0
            
            # Additional scaling for preaveraging
            # For triangular kernel, theta = 1.0 / 12
            theta = 1.0 / 12.0
            scale_factor = 1.0 / (window_size * (1.0 - 2.0 * theta))
            
            return correction * pav * scale_factor
    
    def _compute_subsampled_preaveraged_variance(self, 
                                               returns: np.ndarray, 
                                               window_size: int,
                                               subsample_factor: int) -> float:
        """
        Compute subsampled preaveraged variance for noise reduction.
        
        Args:
            returns: Return series
            window_size: Window size for preaveraging
            subsample_factor: Number of subsamples to use
        
        Returns:
            float: Subsampled preaveraged variance
        
        Raises:
            ValueError: If subsample_factor is invalid
            NumericError: If numerical issues are encountered
        """
        # Validate subsample_factor
        if not isinstance(subsample_factor, int) or subsample_factor < 1:
            raise ParameterError(f"subsample_factor must be a positive integer, got {subsample_factor}")
        
        if subsample_factor == 1:
            # No subsampling needed
            return self._compute_preaveraged_variance(returns, window_size, self._debiased)
        
        try:
            n = len(returns)
            subsampled_pav = 0.0
            
            for i in range(subsample_factor):
                # Extract i-th subsample
                subsample = returns[i::subsample_factor]
                
                # Skip if subsample is too short
                if len(subsample) < window_size + 1:
                    continue
                
                # Compute preaveraged variance for this subsample
                subsample_pav = self._compute_preaveraged_variance(
                    subsample, window_size, self._debiased
                )
                
                # Scale by the number of observations
                scaled_pav = subsample_pav * (n / len(subsample))
                
                # Add to the total
                subsampled_pav += scaled_pav
            
            # Average across subsamples
            return subsampled_pav / subsample_factor
            
        except Exception as e:
            logger.error(f"Subsampled preaveraged variance computation failed: {str(e)}")
            raise ValueError(f"Subsampled preaveraged variance computation failed: {str(e)}") from e
    
    def _compute_optimal_window_size(self, returns: np.ndarray) -> int:
        """
        Compute the optimal window size for preaveraging based on noise variance.
        
        Args:
            returns: Return series
        
        Returns:
            int: Optimal window size
        """
        # Estimate noise variance if not already done
        if self._noise_variance is None:
            self._noise_variance = self.estimate_noise_variance(returns)
        
        # Compute optimal window size based on noise variance
        # Theoretical optimal window size is proportional to noise^(-1/3)
        n = len(returns)
        c = 1.0  # Constant factor
        
        # Avoid division by zero
        noise_var = max(self._noise_variance, 1e-10)
        
        # Compute optimal window size
        optimal_window = int(c * n**(1/3) * noise_var**(-1/3))
        
        # Ensure window size is at least 2 and at most n/10
        optimal_window = max(2, min(optimal_window, n // 10))
        
        return optimal_window
    
    async def fit_async(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> RealizedEstimatorResult:
        """
        Asynchronously fit the preaveraged variance estimator to the provided data.
        
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
        import asyncio
        
        # Validate input data
        self.validate_data(data)
        prices, times = data
        
        # Preprocess data
        processed_prices, processed_times, returns = self.preprocess_data(prices, times)
        
        # Create a coroutine for the computation
        async def compute_async():
            import time
            start_time = time.time()
            
            try:
                # Compute realized measure
                realized_measure = self._compute_realized_measure(
                    processed_prices, processed_times, returns, **kwargs
                )
                
                # Update instance state
                self._realized_measure = realized_measure
                self._fitted = True
                
                # Create result object
                result = RealizedEstimatorResult(
                    model_name=self._name,
                    realized_measure=realized_measure,
                    prices=prices,
                    times=times,
                    sampling_frequency=self._config.sampling_frequency,
                    kernel_type=self._config.kernel_type,
                    bandwidth=self._config.bandwidth,
                    subsampling=self._config.use_subsampling,
                    noise_correction=self._config.apply_noise_correction,
                    annualization_factor=self._config.annualization_factor if self._config.annualize else None,
                    returns=returns,
                    noise_variance=self._noise_variance,
                    computation_time=time.time() - start_time,
                    config=self._config.to_dict()
                )
                
                # Store result
                self._results = result
                
                return result
                
            except Exception as e:
                logger.error(f"Asynchronous estimation failed: {str(e)}")
                raise RuntimeError(f"Preaveraged variance estimation failed: {str(e)}") from e
        
        # Run the computation asynchronously
        return await compute_async()
    
    def calibrate(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> RealizedEstimatorConfig:
        """
        Calibrate the estimator configuration based on the provided data.
        
        This method analyzes the input data and determines optimal configuration
        parameters for the preaveraged variance estimator, such as window size,
        sampling frequency, and subsampling factor.
        
        Args:
            data: The data to calibrate the estimator with, as a tuple of (prices, times)
            **kwargs: Additional keyword arguments for calibration
        
        Returns:
            RealizedEstimatorConfig: Calibrated configuration
        
        Raises:
            ValueError: If calibration fails
        """
        # Validate input data
        self.validate_data(data)
        prices, times = data
        
        try:
            # Compute returns
            if self._config.return_type == 'log':
                returns = np.diff(np.log(prices))
            else:  # 'simple'
                returns = np.diff(prices) / prices[:-1]
            
            # Estimate noise variance
            noise_var = self.estimate_noise_variance(returns)
            self._noise_variance = noise_var
            
            # Compute optimal window size
            optimal_window = self._compute_optimal_window_size(returns)
            
            # Determine optimal sampling frequency
            from .utils import compute_optimal_sampling
            optimal_freq = compute_optimal_sampling(
                prices, times, method='signature', max_points=20
            )
            
            # Determine optimal subsampling factor
            # For preaveraged variance, a moderate subsampling factor is usually sufficient
            optimal_subsample = min(5, len(returns) // 100)
            optimal_subsample = max(1, optimal_subsample)  # Ensure at least 1
            
            # Create calibrated configuration
            calibrated_config = RealizedEstimatorConfig(
                sampling_frequency=optimal_freq,
                annualize=self._config.annualize,
                annualization_factor=self._config.annualization_factor,
                return_type=self._config.return_type,
                use_subsampling=optimal_subsample > 1,
                subsampling_factor=optimal_subsample,
                apply_noise_correction=True,  # Enable noise correction
                time_unit=self._config.time_unit
            )
            
            # Update window size
            self._window_size = optimal_window
            
            logger.info(
                f"Calibrated configuration: window_size={optimal_window}, "
                f"sampling_frequency={optimal_freq}, subsampling_factor={optimal_subsample}"
            )
            
            return calibrated_config
            
        except Exception as e:
            logger.error(f"Calibration failed: {str(e)}")
            raise ValueError(f"Preaveraged variance calibration failed: {str(e)}") from e
    
    def get_noise_robust_ratio(self) -> Optional[float]:
        """
        Get the ratio of noise-robust variance to standard realized variance.
        
        Returns:
            Optional[float]: The noise-robust ratio if both measures are available, None otherwise
        
        Raises:
            RuntimeError: If the estimator has not been fitted
        """
        if not self._fitted or self._realized_measure is None or self._returns is None:
            raise RuntimeError("Estimator has not been fitted. Call fit() first.")
        
        # Compute standard realized variance
        rv = np.sum(self._returns ** 2)
        
        # Compute noise-robust ratio
        noise_robust_ratio = self._realized_measure / rv
        
        return noise_robust_ratio
    
    def plot_preaveraged_returns(self, figsize: Tuple[int, int] = (10, 6)) -> Any:
        """
        Plot original returns and preaveraged returns.
        
        Args:
            figsize: Figure size as (width, height) in inches
        
        Returns:
            matplotlib.figure.Figure: The created figure
        
        Raises:
            RuntimeError: If the estimator has not been fitted
            ImportError: If matplotlib is not available
        """
        if not self._fitted or self._returns is None or self._preaveraged_returns is None:
            raise RuntimeError("Estimator has not been fitted. Call fit() first.")
        
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
            
            # Plot original returns
            ax1.plot(self._returns, 'b-', alpha=0.7, label='Original Returns')
            ax1.set_title('Original Returns')
            ax1.set_ylabel('Return')
            ax1.grid(True, alpha=0.3)
            
            # Plot preaveraged returns
            # Create x-axis for preaveraged returns (centered in the window)
            offset = self._window_size // 2
            x_preaveraged = np.arange(offset, offset + len(self._preaveraged_returns))
            
            ax2.plot(x_preaveraged, self._preaveraged_returns, 'g-', alpha=0.7, 
                    label=f'Preaveraged Returns (window={self._window_size})')
            
            ax2.set_title('Preaveraged Returns')
            ax2.set_xlabel('Observation')
            ax2.set_ylabel('Preaveraged Return')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except ImportError:
            logger.warning("Matplotlib is required for plotting")
            raise ImportError("Matplotlib is required for plotting")
    
    def plot_noise_impact(self, figsize: Tuple[int, int] = (10, 6)) -> Any:
        """
        Plot the impact of noise on volatility estimation.
        
        Args:
            figsize: Figure size as (width, height) in inches
        
        Returns:
            matplotlib.figure.Figure: The created figure
        
        Raises:
            RuntimeError: If the estimator has not been fitted
            ImportError: If matplotlib is not available
        """
        if not self._fitted or self._returns is None:
            raise RuntimeError("Estimator has not been fitted. Call fit() first.")
        
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # Compute standard realized variance
            rv = np.sum(self._returns ** 2)
            
            # Compute preaveraged variance with different window sizes
            window_sizes = np.arange(2, min(30, len(self._returns) // 5))
            pav_values = []
            
            for window in window_sizes:
                pav = self._compute_preaveraged_variance(
                    self._returns, window, self._debiased
                )
                pav_values.append(pav)
            
            # Plot results
            ax.plot(window_sizes, pav_values, 'b-', marker='o', label='Preaveraged Variance')
            ax.axhline(y=rv, color='r', linestyle='--', label='Standard Realized Variance')
            
            if self._realized_measure is not None:
                ax.axhline(y=self._realized_measure, color='g', linestyle='-', 
                          label=f'Selected PAV (window={self._window_size})')
            
            ax.set_title('Impact of Window Size on Preaveraged Variance')
            ax.set_xlabel('Window Size')
            ax.set_ylabel('Volatility Measure')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except ImportError:
            logger.warning("Matplotlib is required for plotting")
            raise ImportError("Matplotlib is required for plotting")
    
    def summary(self) -> str:
        """
        Generate a text summary of the preaveraged variance estimator.
        
        Returns:
            str: A formatted string containing the estimator summary
        
        Raises:
            RuntimeError: If the estimator has not been fitted
        """
        if not self._fitted:
            return f"Preaveraged Variance Estimator (not fitted)"
        
        if self._results is None:
            return f"Preaveraged Variance Estimator (fitted, but no results available)"
        
        base_summary = self._results.summary()
        
        # Add preaveraged variance-specific information
        additional_info = ""
        
        # Add window size information
        additional_info += f"Window Size: {self._window_size}\n"
        
        # Add noise-robust ratio if available
        noise_robust_ratio = self.get_noise_robust_ratio()
        if noise_robust_ratio is not None:
            additional_info += f"Noise-Robust Ratio: {noise_robust_ratio:.4f}\n"
        
        # Add signal-to-noise ratio if available
        signal_to_noise = self.get_signal_to_noise_ratio()
        if signal_to_noise is not None:
            additional_info += f"Signal-to-Noise Ratio: {signal_to_noise:.4f}\n"
        
        # Add debiasing information
        additional_info += f"Debiased: {self._debiased}\n"
        
        if additional_info:
            additional_info = "\nPreaveraged Variance Information:\n" + additional_info
        
        return base_summary + additional_info
    
    def __repr__(self) -> str:
        """
        Generate a detailed string representation of the estimator.
        
        Returns:
            str: A detailed string representation of the estimator
        """
        fitted_str = "fitted" if self._fitted else "not fitted"
        window_str = f", window_size={self._window_size}"
        config_str = f", config={self._config}" if self._config else ""
        return f"PreaveragedVariance({fitted_str}{window_str}{config_str})"