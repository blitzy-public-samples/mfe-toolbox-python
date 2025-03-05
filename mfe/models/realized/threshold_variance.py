# mfe/models/realized/threshold_variance.py
"""
Threshold realized variance estimator for robust volatility measurement.

This module implements the threshold realized variance estimator, which uses local
truncation to reduce the impact of jumps on volatility estimation. By applying
adaptive thresholding to filter out large returns before computing variance,
this estimator provides more robust volatility measurements in the presence of jumps.

The implementation follows a class-based design inheriting from JumpRobustEstimator,
with comprehensive type hints, parameter validation, and Numba-accelerated core
calculations for optimal performance. The estimator supports various configuration
options including different threshold methods, subsampling, and visualization
capabilities for threshold effects.

This approach provides a more robust alternative to standard realized variance
estimators when dealing with financial time series containing price jumps.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd

from .base import JumpRobustEstimator, RealizedEstimatorConfig, RealizedEstimatorResult
from ...core.exceptions import ParameterError, NumericError, DimensionError

# Set up module-level logger
logger = logging.getLogger("mfe.models.realized.threshold_variance")

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
    logger.debug("Numba available for threshold variance acceleration")
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    HAS_NUMBA = False
    logger.info("Numba not available. Threshold variance will use pure NumPy implementation.")


# Numba-accelerated core implementation
@jit(nopython=True, cache=True)
def _threshold_variance_core(returns: np.ndarray, threshold: float) -> float:
    """
    Numba-accelerated core implementation of threshold realized variance.
    
    Args:
        returns: Array of returns
        threshold: Threshold for jump detection
        
    Returns:
        Threshold realized variance
    """
    n = len(returns)
    variance_sum = 0.0
    
    # Compute variance with thresholding
    for i in range(n):
        # Apply threshold to return
        if abs(returns[i]) <= threshold:
            variance_sum += returns[i] ** 2
    
    return variance_sum


class ThresholdVariance(JumpRobustEstimator):
    """
    Threshold realized variance estimator for robust volatility measurement.
    
    This class implements the threshold realized variance estimator, which applies
    local truncation to reduce the impact of jumps on volatility estimation.
    By filtering out returns that exceed a threshold before computing variance,
    this estimator provides more robust volatility measurements in the presence of jumps.
    
    The estimator inherits from JumpRobustEstimator, providing specialized
    functionality for jump detection and separation of continuous and jump
    components of volatility.
    
    Attributes:
        config: Configuration parameters for the estimator
        jump_threshold: Threshold used for jump detection
        jump_indicators: Boolean array indicating detected jumps
        threshold_method: Method used for threshold determination
    """
    
    def __init__(self, 
                config: Optional[RealizedEstimatorConfig] = None,
                threshold_method: str = "adaptive"):
        """
        Initialize the threshold realized variance estimator.
        
        Args:
            config: Configuration parameters for the estimator
            threshold_method: Method for determining threshold ('fixed', 'adaptive', 'quantile')
        """
        super().__init__(config=config, name="Threshold Realized Variance")
        
        # Set threshold method
        self._threshold_method = threshold_method
        
        # Additional attributes
        self._threshold_value: Optional[float] = None
        self._threshold_returns: Optional[np.ndarray] = None
        self._continuous_variance: Optional[float] = None
        self._jump_variance: Optional[float] = None
    
    @property
    def threshold_method(self) -> str:
        """
        Get the threshold method.
        
        Returns:
            str: Threshold method
        """
        return self._threshold_method
    
    @threshold_method.setter
    def threshold_method(self, method: str) -> None:
        """
        Set the threshold method.
        
        Args:
            method: Threshold method ('fixed', 'adaptive', 'quantile')
            
        Raises:
            ParameterError: If method is invalid
        """
        if method not in ['fixed', 'adaptive', 'quantile']:
            raise ParameterError(
                f"threshold_method must be one of 'fixed', 'adaptive', 'quantile', got {method}"
            )
        
        self._threshold_method = method
        self._fitted = False  # Reset fitted state when method changes
    
    @property
    def threshold_value(self) -> Optional[float]:
        """
        Get the threshold value used for estimation.
        
        Returns:
            Optional[float]: Threshold value if the estimator has been fitted, None otherwise
        """
        return self._threshold_value
    
    @property
    def continuous_variance(self) -> Optional[float]:
        """
        Get the continuous component of variance (without jumps).
        
        Returns:
            Optional[float]: Continuous variance if the estimator has been fitted, None otherwise
        """
        return self._continuous_variance
    
    @property
    def jump_variance(self) -> Optional[float]:
        """
        Get the jump component of variance.
        
        Returns:
            Optional[float]: Jump variance if the estimator has been fitted, None otherwise
        """
        return self._jump_variance
    
    def _compute_threshold(self, 
                          returns: np.ndarray, 
                          method: str = "adaptive",
                          fixed_value: Optional[float] = None,
                          quantile: float = 0.99,
                          multiplier: float = 3.0) -> float:
        """
        Compute threshold for filtering returns.
        
        Args:
            returns: Array of returns
            method: Method for determining threshold ('fixed', 'adaptive', 'quantile')
            fixed_value: Fixed threshold value (used if method is 'fixed')
            quantile: Quantile for threshold determination (used if method is 'quantile')
            multiplier: Multiplier for adaptive threshold (used if method is 'adaptive')
            
        Returns:
            float: Computed threshold
            
        Raises:
            ParameterError: If method is invalid or parameters are inconsistent
            ValueError: If computation fails
        """
        if method == "fixed":
            # Use fixed threshold value
            if fixed_value is None:
                raise ParameterError("fixed_value must be provided when method is 'fixed'")
            
            if fixed_value <= 0:
                raise ParameterError("fixed_value must be positive")
            
            return fixed_value
        
        elif method == "quantile":
            # Use quantile of absolute returns
            if not 0 < quantile < 1:
                raise ParameterError("quantile must be between 0 and 1")
            
            abs_returns = np.abs(returns)
            threshold = np.quantile(abs_returns, quantile)
            
            return threshold
        
        elif method == "adaptive":
            # Use adaptive threshold based on local volatility
            try:
                # Estimate local volatility using bipower variation
                from .utils import detect_jumps
                _, threshold = detect_jumps(returns, threshold_multiplier=multiplier, method='bipower')
                
                return threshold
                
            except Exception as e:
                logger.warning(f"Adaptive threshold computation failed: {str(e)}. "
                              f"Falling back to quantile method.")
                
                # Fallback to quantile method
                return self._compute_threshold(returns, "quantile", None, quantile)
        
        else:
            raise ParameterError(
                f"method must be one of 'fixed', 'adaptive', 'quantile', got {method}"
            )
    
    def _apply_threshold(self, returns: np.ndarray, threshold: float) -> np.ndarray:
        """
        Apply threshold to returns.
        
        Args:
            returns: Array of returns
            threshold: Threshold value
            
        Returns:
            np.ndarray: Thresholded returns (returns where |r| > threshold are set to 0)
        """
        # Create a copy of returns
        thresholded_returns = returns.copy()
        
        # Set returns exceeding threshold to zero
        thresholded_returns[np.abs(thresholded_returns) > threshold] = 0.0
        
        return thresholded_returns
    
    def _compute_realized_measure(self, 
                                 prices: np.ndarray, 
                                 times: np.ndarray, 
                                 returns: np.ndarray,
                                 threshold_method: Optional[str] = None,
                                 fixed_threshold: Optional[float] = None,
                                 threshold_quantile: float = 0.99,
                                 threshold_multiplier: float = 3.0,
                                 **kwargs: Any) -> np.ndarray:
        """
        Compute the threshold realized variance from the preprocessed data.
        
        This method implements the core threshold realized variance calculation,
        with options for different threshold methods and subsampling.
        
        Args:
            prices: Preprocessed price data
            times: Preprocessed time points
            returns: Returns computed from prices
            threshold_method: Method for determining threshold ('fixed', 'adaptive', 'quantile')
            fixed_threshold: Fixed threshold value (used if threshold_method is 'fixed')
            threshold_quantile: Quantile for threshold determination (used if threshold_method is 'quantile')
            threshold_multiplier: Multiplier for adaptive threshold (used if threshold_method is 'adaptive')
            **kwargs: Additional keyword arguments for computation
        
        Returns:
            np.ndarray: Threshold realized variance
        
        Raises:
            ValueError: If computation fails
            NumericError: If numerical issues are encountered
        """
        # Use provided threshold method or instance default
        method = threshold_method if threshold_method is not None else self._threshold_method
        
        try:
            # Compute threshold
            threshold = self._compute_threshold(
                returns, method, fixed_threshold, threshold_quantile, threshold_multiplier
            )
            
            # Store threshold value
            self._threshold_value = threshold
            
            # Apply threshold to returns
            thresholded_returns = self._apply_threshold(returns, threshold)
            
            # Store thresholded returns
            self._threshold_returns = thresholded_returns
            
            # Detect jumps based on threshold
            jump_indicators = np.abs(returns) > threshold
            
            # Store jump detection results
            self._jump_threshold = threshold
            self._jump_indicators = jump_indicators
            
            # Check if subsampling is enabled
            if self._config.use_subsampling and self._config.subsampling_factor > 1:
                # Compute subsampled threshold variance
                trv = self._compute_subsampled_threshold_variance(
                    returns, threshold, self._config.subsampling_factor
                )
            else:
                # Compute standard threshold variance
                if HAS_NUMBA:
                    # Use Numba-accelerated implementation
                    trv = _threshold_variance_core(returns, threshold)
                else:
                    # Use pure NumPy implementation
                    trv = np.sum(returns[np.abs(returns) <= threshold] ** 2)
            
            # Compute continuous and jump components
            self._continuous_variance = trv
            self._jump_variance = np.sum(returns[jump_indicators] ** 2)
            
            # Log jump detection results
            jump_count = np.sum(jump_indicators)
            logger.info(
                f"Detected {jump_count} jumps ({jump_count / len(returns) * 100:.2f}%) "
                f"with threshold {threshold:.6f}"
            )
            
            return trv
            
        except Exception as e:
            logger.error(f"Threshold variance computation failed: {str(e)}")
            raise ValueError(f"Threshold variance computation failed: {str(e)}") from e
    
    def _compute_subsampled_threshold_variance(self, 
                                             returns: np.ndarray, 
                                             threshold: float,
                                             subsample_factor: int) -> float:
        """
        Compute subsampled threshold variance for noise reduction.
        
        Args:
            returns: Return series
            threshold: Threshold value
            subsample_factor: Number of subsamples to use
        
        Returns:
            float: Subsampled threshold variance
        
        Raises:
            ValueError: If subsample_factor is invalid
            NumericError: If numerical issues are encountered
        """
        # Validate subsample_factor
        if not isinstance(subsample_factor, int) or subsample_factor < 1:
            raise ParameterError(f"subsample_factor must be a positive integer, got {subsample_factor}")
        
        if subsample_factor == 1:
            # No subsampling needed
            if HAS_NUMBA:
                return _threshold_variance_core(returns, threshold)
            else:
                return np.sum(returns[np.abs(returns) <= threshold] ** 2)
        
        try:
            n = len(returns)
            subsampled_trv = 0.0
            
            for i in range(subsample_factor):
                # Extract i-th subsample
                subsample = returns[i::subsample_factor]
                
                # Compute threshold variance for this subsample
                if HAS_NUMBA:
                    subsample_trv = _threshold_variance_core(subsample, threshold)
                else:
                    subsample_trv = np.sum(subsample[np.abs(subsample) <= threshold] ** 2)
                
                # Scale by the number of observations
                scaled_trv = subsample_trv * (n / len(subsample))
                
                # Add to the total
                subsampled_trv += scaled_trv
            
            # Average across subsamples
            return subsampled_trv / subsample_factor
            
        except Exception as e:
            logger.error(f"Subsampled threshold variance computation failed: {str(e)}")
            raise ValueError(f"Subsampled threshold variance computation failed: {str(e)}") from e
    
    async def fit_async(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> RealizedEstimatorResult:
        """
        Asynchronously fit the threshold realized variance estimator to the provided data.
        
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
                    jump_threshold=self._jump_threshold,
                    jump_indicators=self._jump_indicators,
                    computation_time=time.time() - start_time,
                    config=self._config.to_dict()
                )
                
                # Store result
                self._results = result
                
                return result
                
            except Exception as e:
                logger.error(f"Asynchronous estimation failed: {str(e)}")
                raise RuntimeError(f"Threshold variance estimation failed: {str(e)}") from e
        
        # Run the computation asynchronously
        return await compute_async()
    
    def calibrate(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> RealizedEstimatorConfig:
        """
        Calibrate the estimator configuration based on the provided data.
        
        This method analyzes the input data and determines optimal configuration
        parameters for the threshold realized variance estimator, such as
        sampling frequency and subsampling factor.
        
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
            
            # Determine optimal sampling frequency
            from .utils import compute_optimal_sampling
            optimal_freq = compute_optimal_sampling(
                prices, times, method='signature', max_points=20
            )
            
            # Determine optimal subsampling factor
            # For threshold variance, a moderate subsampling factor is usually sufficient
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
                apply_noise_correction=False,  # Threshold variance doesn't use noise correction
                time_unit=self._config.time_unit
            )
            
            logger.info(
                f"Calibrated configuration: sampling_frequency={optimal_freq}, "
                f"subsampling_factor={optimal_subsample}"
            )
            
            return calibrated_config
            
        except Exception as e:
            logger.error(f"Calibration failed: {str(e)}")
            raise ValueError(f"Threshold variance calibration failed: {str(e)}") from e
    
    def get_jump_ratio(self) -> Optional[float]:
        """
        Get the ratio of variance attributed to jumps.
        
        Returns:
            Optional[float]: The jump ratio if the estimator has been fitted, None otherwise
        
        Raises:
            RuntimeError: If the estimator has not been fitted
        """
        if not self._fitted or self._realized_measure is None or self._jump_variance is None:
            raise RuntimeError("Estimator has not been fitted. Call fit() first.")
        
        # Compute total variance (continuous + jumps)
        total_variance = self._continuous_variance + self._jump_variance
        
        # Compute jump ratio
        if total_variance > 0:
            jump_ratio = self._jump_variance / total_variance
        else:
            jump_ratio = 0.0
        
        return jump_ratio
    
    def get_threshold_effect(self) -> Optional[float]:
        """
        Get the effect of thresholding on the realized measure.
        
        Returns:
            Optional[float]: The threshold effect if the estimator has been fitted, None otherwise
        
        Raises:
            RuntimeError: If the estimator has not been fitted
        """
        if not self._fitted or self._returns is None or self._realized_measure is None:
            raise RuntimeError("Estimator has not been fitted. Call fit() first.")
        
        try:
            # Compute standard realized variance without thresholding
            standard_rv = np.sum(self._returns**2)
            
            # Compute threshold effect as ratio
            threshold_effect = self._realized_measure / standard_rv
            
            return threshold_effect
            
        except Exception as e:
            logger.warning(f"Threshold effect computation failed: {str(e)}")
            return None
    
    def plot_returns_with_threshold(self, figsize: Tuple[int, int] = (10, 6)) -> Any:
        """
        Plot returns with threshold highlighted.
        
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
            
            # Plot all returns
            ax.plot(self._returns, 'b-', alpha=0.7, label='Returns')
            
            # Highlight thresholded returns if available
            if self._jump_indicators is not None and self._jump_threshold is not None:
                jump_indices = np.where(self._jump_indicators)[0]
                jump_returns = self._returns[jump_indices]
                ax.scatter(jump_indices, jump_returns, color='r', s=50, 
                          marker='o', label='Exceeds Threshold')
                
                # Add threshold lines
                ax.axhline(y=self._jump_threshold, color='r', linestyle='--', 
                          alpha=0.5, label=f'Threshold (+{self._jump_threshold:.4f})')
                ax.axhline(y=-self._jump_threshold, color='r', linestyle='--', 
                          alpha=0.5, label=f'Threshold (-{self._jump_threshold:.4f})')
            
            ax.set_title('Returns with Threshold')
            ax.set_xlabel('Observation')
            ax.set_ylabel('Return')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except ImportError:
            logger.warning("Matplotlib is required for plotting")
            raise ImportError("Matplotlib is required for plotting")
    
    def plot_threshold_comparison(self, figsize: Tuple[int, int] = (12, 8)) -> Any:
        """
        Plot comparison of original and thresholded returns.
        
        Args:
            figsize: Figure size as (width, height) in inches
        
        Returns:
            matplotlib.figure.Figure: The created figure
        
        Raises:
            RuntimeError: If the estimator has not been fitted
            ImportError: If matplotlib is not available
        """
        if not self._fitted or self._returns is None or self._threshold_returns is None:
            raise RuntimeError("Estimator has not been fitted. Call fit() first.")
        
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
            
            # Plot original returns
            ax1.plot(self._returns, 'b-', alpha=0.7)
            ax1.set_title('Original Returns')
            ax1.set_ylabel('Return')
            ax1.grid(True, alpha=0.3)
            
            # Plot thresholded returns
            ax2.plot(self._threshold_returns, 'g-', alpha=0.7)
            ax2.set_title('Thresholded Returns')
            ax2.set_xlabel('Observation')
            ax2.set_ylabel('Return')
            ax2.grid(True, alpha=0.3)
            
            # Add threshold lines if available
            if self._jump_threshold is not None:
                ax1.axhline(y=self._jump_threshold, color='r', linestyle='--', 
                           alpha=0.5, label=f'Threshold (±{self._jump_threshold:.4f})')
                ax1.axhline(y=-self._jump_threshold, color='r', linestyle='--', 
                           alpha=0.5)
                ax1.legend()
            
            plt.tight_layout()
            return fig
            
        except ImportError:
            logger.warning("Matplotlib is required for plotting")
            raise ImportError("Matplotlib is required for plotting")
    
    def plot_variance_decomposition(self, figsize: Tuple[int, int] = (8, 6)) -> Any:
        """
        Plot decomposition of variance into continuous and jump components.
        
        Args:
            figsize: Figure size as (width, height) in inches
        
        Returns:
            matplotlib.figure.Figure: The created figure
        
        Raises:
            RuntimeError: If the estimator has not been fitted
            ImportError: If matplotlib is not available
        """
        if not self._fitted or self._continuous_variance is None or self._jump_variance is None:
            raise RuntimeError("Estimator has not been fitted. Call fit() first.")
        
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # Create data for pie chart
            labels = ['Continuous', 'Jumps']
            sizes = [self._continuous_variance, self._jump_variance]
            
            # Create pie chart
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
                  colors=['#4CAF50', '#F44336'], explode=(0, 0.1))
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            
            ax.set_title('Variance Decomposition')
            
            plt.tight_layout()
            return fig
            
        except ImportError:
            logger.warning("Matplotlib is required for plotting")
            raise ImportError("Matplotlib is required for plotting")
    
    def summary(self) -> str:
        """
        Generate a text summary of the threshold realized variance estimator.
        
        Returns:
            str: A formatted string containing the estimator summary
        
        Raises:
            RuntimeError: If the estimator has not been fitted
        """
        if not self._fitted:
            return f"Threshold Realized Variance Estimator (not fitted)"
        
        if self._results is None:
            return f"Threshold Realized Variance Estimator (fitted, but no results available)"
        
        base_summary = self._results.summary()
        
        # Add threshold variance-specific information
        additional_info = ""
        
        # Add threshold method information
        additional_info += f"Threshold Method: {self._threshold_method}\n"
        
        # Add threshold value if available
        if self._threshold_value is not None:
            additional_info += f"Threshold Value: {self._threshold_value:.6f}\n"
        
        # Add variance decomposition if available
        if self._continuous_variance is not None and self._jump_variance is not None:
            total_variance = self._continuous_variance + self._jump_variance
            continuous_pct = (self._continuous_variance / total_variance) * 100 if total_variance > 0 else 0
            jump_pct = (self._jump_variance / total_variance) * 100 if total_variance > 0 else 0
            
            additional_info += f"Continuous Variance: {self._continuous_variance:.6e} ({continuous_pct:.2f}%)\n"
            additional_info += f"Jump Variance: {self._jump_variance:.6e} ({jump_pct:.2f}%)\n"
        
        # Add threshold effect if available
        threshold_effect = self.get_threshold_effect()
        if threshold_effect is not None:
            additional_info += f"Threshold Effect: {threshold_effect:.4f}\n"
        
        if additional_info:
            additional_info = "\nThreshold Variance Information:\n" + additional_info
        
        return base_summary + additional_info
    
    def __repr__(self) -> str:
        """
        Generate a detailed string representation of the estimator.
        
        Returns:
            str: A detailed string representation of the estimator
        """
        fitted_str = "fitted" if self._fitted else "not fitted"
        threshold_str = f"threshold_method='{self._threshold_method}'"
        config_str = f", config={self._config}" if self._config else ""
        return f"ThresholdVariance({fitted_str}, {threshold_str}{config_str})"