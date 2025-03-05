'''
Threshold multipower variation estimator for robust volatility measurement.

This module implements threshold multipower variation estimators that combine
thresholding techniques with multipower variation for robust volatility estimation.
These estimators provide superior robustness to both jumps and noise in high-frequency
financial data by applying thresholds to filter out large returns before computing
multipower variation.

The implementation follows a class-based design inheriting from JumpRobustEstimator,
with comprehensive type hints, parameter validation, and Numba-accelerated core
calculations for optimal performance. The estimator supports various configuration
options including different power variations, threshold methods, and visualization
capabilities for threshold effects.

This approach provides a more robust alternative to standard realized volatility
estimators when dealing with both jumps and microstructure noise simultaneously.
'''
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd

from .base import JumpRobustEstimator, RealizedEstimatorConfig, RealizedEstimatorResult
from ._numba_core import _multipower_variation_core
from ...core.exceptions import ParameterError, NumericError, DimensionError

# Set up module-level logger
logger = logging.getLogger("mfe.models.realized.threshold_multipower_variation")


@np.vectorize
def _threshold_function(x: float, threshold: float) -> float:
    """
    Apply threshold to a value.
    
    Args:
        x: Input value
        threshold: Threshold value
        
    Returns:
        Thresholded value (x if |x| <= threshold, 0 otherwise)
    """
    return x if abs(x) <= threshold else 0.0


# Numba-accelerated core implementation
from numba import jit, float64, boolean

@jit(float64(float64[:], float64[:], float64), nopython=True, cache=True)
def _threshold_multipower_variation_core(returns: np.ndarray, powers: np.ndarray, threshold: float) -> float:
    """
    Numba-accelerated core implementation of threshold multipower variation.
    
    Args:
        returns: Array of returns
        powers: Array of powers for each lag
        threshold: Threshold for jump detection
        
    Returns:
        Threshold multipower variation
    """
    n = len(returns)
    m = len(powers)
    mpv_sum = 0.0
    
    # Compute multipower variation with thresholding
    for i in range(n - m + 1):
        term = 1.0
        valid_term = True
        
        for j in range(m):
            # Apply threshold to return
            if abs(returns[i + j]) > threshold:
                valid_term = False
                break
            
            # Compute power term
            term *= abs(returns[i + j]) ** powers[j]
        
        # Add term to sum if all returns are below threshold
        if valid_term:
            mpv_sum += term
    
    # Compute scaling factor
    scaling = 1.0
    for p in powers:
        # mu_p = E[|Z|^p] where Z ~ N(0,1)
        mu_p = 2**(p/2) * np.exp(np.log(np.pi) / 2) / np.exp(np.log(2) / 2)
        scaling *= mu_p
    
    # Apply finite sample correction
    correction = n / (n - m + 1)
    
    return correction * mpv_sum / scaling


class ThresholdMultipowerVariation(JumpRobustEstimator):
    """
    Threshold multipower variation estimator for robust volatility measurement.
    
    This class implements threshold multipower variation estimators that combine
    thresholding techniques with multipower variation for robust volatility estimation.
    These estimators provide superior robustness to both jumps and noise by applying
    thresholds to filter out large returns before computing multipower variation.
    
    The estimator inherits from JumpRobustEstimator, providing specialized
    functionality for jump detection and separation of continuous and jump
    components of volatility.
    
    Attributes:
        config: Configuration parameters for the estimator
        jump_threshold: Threshold used for jump detection
        jump_indicators: Boolean array indicating detected jumps
        powers: Array of powers used for multipower variation
        threshold_method: Method used for threshold determination
    """
    
    def __init__(self, 
                config: Optional[RealizedEstimatorConfig] = None,
                powers: Optional[np.ndarray] = None,
                threshold_method: str = "adaptive"):
        """
        Initialize the threshold multipower variation estimator.
        
        Args:
            config: Configuration parameters for the estimator
            powers: Array of powers for multipower variation (default: [2/3, 2/3, 2/3] for tripower)
            threshold_method: Method for determining threshold ('fixed', 'adaptive', 'quantile')
        """
        super().__init__(config=config, name="Threshold Multipower Variation")
        
        # Set default powers if not provided (tripower variation)
        if powers is None:
            self._powers = np.array([2/3, 2/3, 2/3])
        else:
            self._powers = np.asarray(powers)
        
        # Validate powers
        if len(self._powers) < 1:
            raise ParameterError("powers must have at least one element")
        
        # Set threshold method
        self._threshold_method = threshold_method
        
        # Additional attributes
        self._threshold_value: Optional[float] = None
        self._threshold_returns: Optional[np.ndarray] = None
        self._debiased: bool = True  # Whether to apply finite sample bias correction
    
    @property
    def powers(self) -> np.ndarray:
        """
        Get the powers used for multipower variation.
        
        Returns:
            np.ndarray: Array of powers
        """
        return self._powers
    
    @powers.setter
    def powers(self, powers: np.ndarray) -> None:
        """
        Set the powers used for multipower variation.
        
        Args:
            powers: Array of powers
            
        Raises:
            ParameterError: If powers is invalid
        """
        powers = np.asarray(powers)
        
        if len(powers) < 1:
            raise ParameterError("powers must have at least one element")
        
        self._powers = powers
        self._fitted = False  # Reset fitted state when powers change
    
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
                from .bipower_variation import BipowerVariation
                bv_estimator = BipowerVariation(self._config)
                bv_result = bv_estimator.fit((np.exp(np.cumsum(np.insert(returns, 0, 0))), np.arange(len(returns) + 1)))
                
                # Compute threshold as multiplier * sqrt(bipower variation)
                threshold = multiplier * np.sqrt(bv_result.realized_measure)
                
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
            np.ndarray: Thresholded returns
        """
        # Apply threshold using vectorized function
        thresholded_returns = _threshold_function(returns, threshold)
        
        return thresholded_returns
    
    def _compute_realized_measure(self, 
                                 prices: np.ndarray, 
                                 times: np.ndarray, 
                                 returns: np.ndarray,
                                 threshold_method: Optional[str] = None,
                                 fixed_threshold: Optional[float] = None,
                                 threshold_quantile: float = 0.99,
                                 threshold_multiplier: float = 3.0,
                                 debiased: bool = True,
                                 **kwargs: Any) -> np.ndarray:
        """
        Compute the threshold multipower variation from the preprocessed data.
        
        This method implements the core threshold multipower variation calculation,
        with options for different threshold methods and subsampling.
        
        Args:
            prices: Preprocessed price data
            times: Preprocessed time points
            returns: Returns computed from prices
            threshold_method: Method for determining threshold ('fixed', 'adaptive', 'quantile')
            fixed_threshold: Fixed threshold value (used if threshold_method is 'fixed')
            threshold_quantile: Quantile for threshold determination (used if threshold_method is 'quantile')
            threshold_multiplier: Multiplier for adaptive threshold (used if threshold_method is 'adaptive')
            debiased: Whether to apply finite sample bias correction
            **kwargs: Additional keyword arguments for computation
        
        Returns:
            np.ndarray: Threshold multipower variation
        
        Raises:
            ValueError: If computation fails
            NumericError: If numerical issues are encountered
        """
        # Store debiased flag
        self._debiased = debiased
        
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
            
            # Check if subsampling is enabled
            if self._config.use_subsampling and self._config.subsampling_factor > 1:
                # Compute subsampled threshold multipower variation
                tmpv = self._compute_subsampled_threshold_multipower_variation(
                    returns, threshold, self._config.subsampling_factor
                )
            else:
                # Compute standard threshold multipower variation
                tmpv = self._compute_threshold_multipower_variation(
                    returns, threshold, debiased
                )
            
            # Detect jumps based on threshold
            jump_indicators = np.abs(returns) > threshold
            
            # Store jump detection results
            self._jump_threshold = threshold
            self._jump_indicators = jump_indicators
            
            # Log jump detection results
            jump_count = np.sum(jump_indicators)
            logger.info(
                f"Detected {jump_count} jumps ({jump_count / len(returns) * 100:.2f}%) "
                f"with threshold {threshold:.6f}"
            )
            
            return tmpv
            
        except Exception as e:
            logger.error(f"Threshold multipower variation computation failed: {str(e)}")
            raise ValueError(f"Threshold multipower variation computation failed: {str(e)}") from e
    
    def _compute_threshold_multipower_variation(self, 
                                              returns: np.ndarray, 
                                              threshold: float,
                                              debiased: bool = True) -> float:
        """
        Compute the standard threshold multipower variation.
        
        Args:
            returns: Return series
            threshold: Threshold value
            debiased: Whether to apply finite sample bias correction
        
        Returns:
            float: Threshold multipower variation
        
        Raises:
            NumericError: If numerical issues are encountered
        """
        try:
            # Use Numba-accelerated implementation
            tmpv = _threshold_multipower_variation_core(returns, self._powers, threshold)
            
            # Apply finite sample bias correction if requested
            if not debiased:
                # Remove the finite sample correction applied in the core function
                n = len(returns)
                m = len(self._powers)
                tmpv = tmpv * ((n - m + 1) / n)
            
            return tmpv
            
        except Exception as e:
            logger.error(f"Threshold multipower variation core computation failed: {str(e)}")
            
            # Fallback to pure NumPy implementation if Numba fails
            logger.warning("Falling back to pure NumPy implementation")
            
            n = len(returns)
            m = len(self._powers)
            mpv_sum = 0.0
            
            # Compute multipower variation with thresholding
            for i in range(n - m + 1):
                term = 1.0
                valid_term = True
                
                for j in range(m):
                    # Apply threshold to return
                    if abs(returns[i + j]) > threshold:
                        valid_term = False
                        break
                    
                    # Compute power term
                    term *= abs(returns[i + j]) ** self._powers[j]
                
                # Add term to sum if all returns are below threshold
                if valid_term:
                    mpv_sum += term
            
            # Compute scaling factor
            scaling = 1.0
            for p in self._powers:
                # mu_p = E[|Z|^p] where Z ~ N(0,1)
                mu_p = 2**(p/2) * np.exp(np.log(np.pi) / 2) / np.exp(np.log(2) / 2)
                scaling *= mu_p
            
            # Apply finite sample correction if requested
            correction = n / (n - m + 1) if debiased else 1.0
            
            return correction * mpv_sum / scaling
    
    def _compute_subsampled_threshold_multipower_variation(self, 
                                                         returns: np.ndarray, 
                                                         threshold: float,
                                                         subsample_factor: int) -> float:
        """
        Compute subsampled threshold multipower variation for noise reduction.
        
        Args:
            returns: Return series
            threshold: Threshold value
            subsample_factor: Number of subsamples to use
        
        Returns:
            float: Subsampled threshold multipower variation
        
        Raises:
            ValueError: If subsample_factor is invalid
            NumericError: If numerical issues are encountered
        """
        # Validate subsample_factor
        if not isinstance(subsample_factor, int) or subsample_factor < 1:
            raise ParameterError(f"subsample_factor must be a positive integer, got {subsample_factor}")
        
        if subsample_factor == 1:
            # No subsampling needed
            return self._compute_threshold_multipower_variation(returns, threshold, self._debiased)
        
        try:
            n = len(returns)
            subsampled_tmpv = 0.0
            
            for i in range(subsample_factor):
                # Extract i-th subsample
                subsample = returns[i::subsample_factor]
                
                # Skip if subsample is too short
                if len(subsample) <= len(self._powers):
                    continue
                
                # Compute threshold multipower variation for this subsample
                subsample_tmpv = self._compute_threshold_multipower_variation(
                    subsample, threshold, self._debiased
                )
                
                # Scale by the number of observations
                scaled_tmpv = subsample_tmpv * (n / len(subsample))
                
                # Add to the total
                subsampled_tmpv += scaled_tmpv
            
            # Average across subsamples
            return subsampled_tmpv / subsample_factor
            
        except Exception as e:
            logger.error(f"Subsampled threshold multipower variation computation failed: {str(e)}")
            raise ValueError(f"Subsampled threshold multipower variation computation failed: {str(e)}") from e
    
    async def fit_async(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> RealizedEstimatorResult:
        """
        Asynchronously fit the threshold multipower variation estimator to the provided data.
        
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
                raise RuntimeError(f"Threshold multipower variation estimation failed: {str(e)}") from e
        
        # Run the computation asynchronously
        return await compute_async()
    
    def calibrate(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> RealizedEstimatorConfig:
        """
        Calibrate the estimator configuration based on the provided data.
        
        This method analyzes the input data and determines optimal configuration
        parameters for the threshold multipower variation estimator, such as
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
            # For threshold multipower variation, a moderate subsampling factor is usually sufficient
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
                apply_noise_correction=False,  # Threshold multipower variation doesn't use noise correction
                time_unit=self._config.time_unit
            )
            
            logger.info(
                f"Calibrated configuration: sampling_frequency={optimal_freq}, "
                f"subsampling_factor={optimal_subsample}"
            )
            
            return calibrated_config
            
        except Exception as e:
            logger.error(f"Calibration failed: {str(e)}")
            raise ValueError(f"Threshold multipower variation calibration failed: {str(e)}") from e
    
    def get_threshold_ratio(self) -> Optional[float]:
        """
        Get the ratio of returns exceeding the threshold.
        
        Returns:
            Optional[float]: The threshold ratio if the estimator has been fitted, None otherwise
        
        Raises:
            RuntimeError: If the estimator has not been fitted
        """
        if not self._fitted or self._returns is None or self._jump_indicators is None:
            raise RuntimeError("Estimator has not been fitted. Call fit() first.")
        
        # Compute threshold ratio
        threshold_ratio = np.sum(self._jump_indicators) / len(self._returns)
        
        return threshold_ratio
    
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
            # Compute standard multipower variation without thresholding
            from ._numba_core import _multipower_variation_core
            standard_mpv = _multipower_variation_core(self._returns, self._powers)
            
            # Compute threshold effect as ratio
            threshold_effect = self._realized_measure / standard_mpv
            
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
    
    def summary(self) -> str:
        """
        Generate a text summary of the threshold multipower variation estimator.
        
        Returns:
            str: A formatted string containing the estimator summary
        
        Raises:
            RuntimeError: If the estimator has not been fitted
        """
        if not self._fitted:
            return f"Threshold Multipower Variation Estimator (not fitted)"
        
        if self._results is None:
            return f"Threshold Multipower Variation Estimator (fitted, but no results available)"
        
        base_summary = self._results.summary()
        
        # Add threshold multipower-specific information
        additional_info = ""
        
        # Add powers information
        powers_str = ", ".join([f"{p:.4f}" for p in self._powers])
        additional_info += f"Powers: [{powers_str}]\n"
        
        # Add threshold method information
        additional_info += f"Threshold Method: {self._threshold_method}\n"
        
        # Add threshold value if available
        if self._threshold_value is not None:
            additional_info += f"Threshold Value: {self._threshold_value:.6f}\n"
        
        # Add threshold ratio if jumps were detected
        threshold_ratio = self.get_threshold_ratio()
        if threshold_ratio is not None:
            additional_info += f"Threshold Ratio: {threshold_ratio:.4f} ({threshold_ratio * 100:.2f}%)\n"
        
        # Add threshold effect if available
        threshold_effect = self.get_threshold_effect()
        if threshold_effect is not None:
            additional_info += f"Threshold Effect: {threshold_effect:.4f}\n"
        
        # Add debiasing information
        additional_info += f"Debiased: {self._debiased}\n"
        
        if additional_info:
            additional_info = "\nThreshold Multipower Variation Information:\n" + additional_info
        
        return base_summary + additional_info
    
    def __repr__(self) -> str:
        """
        Generate a detailed string representation of the estimator.
        
        Returns:
            str: A detailed string representation of the estimator
        """
        fitted_str = "fitted" if self._fitted else "not fitted"
        powers_str = f"powers=[{', '.join([f'{p:.4f}' for p in self._powers])}]"
        threshold_str = f"threshold_method='{self._threshold_method}'"
        config_str = f", config={self._config}" if self._config else ""
        return f"ThresholdMultipowerVariation({fitted_str}, {powers_str}, {threshold_str}{config_str})"