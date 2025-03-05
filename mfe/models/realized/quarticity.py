"""
Realized quarticity estimators for volatility of volatility measurement.

This module implements realized quarticity estimators, which measure the variability
of volatility itself (effectively the volatility of volatility). These measures are
essential for statistical inference about volatility, enabling the construction of
confidence intervals and hypothesis tests.

The implementation provides multiple quarticity estimation methods including standard
realized quarticity, tri-power quarticity, and quad-power quarticity, with Numba
acceleration for performance-critical calculations. The estimators support various
configuration options including subsampling for noise reduction and visualization
methods for quarticity estimates.
"""

import logging
import warnings
from typing import Any, Dict, Literal, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd

from .base import BaseRealizedEstimator, RealizedEstimatorConfig, RealizedEstimatorResult
from ._numba_core import _realized_quarticity_core, _tri_quarticity_core
from ...core.exceptions import ParameterError, NumericError
from ...core.parameters import validate_positive, validate_non_negative

# Set up module-level logger
logger = logging.getLogger("mfe.models.realized.quarticity")


class RealizedQuarticity(BaseRealizedEstimator):
    """
    Realized quarticity estimator for volatility of volatility measurement.
    
    This class implements realized quarticity estimators, which measure the variability
    of volatility itself (effectively the volatility of volatility). These measures are
    essential for statistical inference about volatility, enabling the construction of
    confidence intervals and hypothesis tests.
    
    The implementation provides multiple quarticity estimation methods including standard
    realized quarticity, tri-power quarticity, and quad-power quarticity, with Numba
    acceleration for performance-critical calculations.
    
    Attributes:
        config: Configuration parameters for the estimator
        method: Method used for quarticity estimation ('standard', 'tripower', 'quadpower')
    """
    
    def __init__(self, 
                config: Optional[RealizedEstimatorConfig] = None,
                method: Literal['standard', 'tripower', 'quadpower'] = 'standard'):
        """
        Initialize the realized quarticity estimator.
        
        Args:
            config: Configuration parameters for the estimator
            method: Method to use for quarticity estimation:
                   'standard': Standard realized quarticity (sum of fourth powers)
                   'tripower': Tri-power quarticity (robust to jumps)
                   'quadpower': Quad-power quarticity (more robust to jumps)
        
        Raises:
            ValueError: If method is not one of the supported methods
        """
        super().__init__(config=config, name=f"Realized Quarticity ({method})")
        
        # Validate method
        if method not in ['standard', 'tripower', 'quadpower']:
            raise ValueError(
                f"method must be one of 'standard', 'tripower', 'quadpower', got {method}"
            )
        
        self._method = method
        self._debiased = True  # Whether to apply finite sample bias correction
    
    @property
    def method(self) -> str:
        """
        Get the quarticity estimation method.
        
        Returns:
            str: The quarticity estimation method
        """
        return self._method
    
    @method.setter
    def method(self, method: Literal['standard', 'tripower', 'quadpower']) -> None:
        """
        Set the quarticity estimation method.
        
        Args:
            method: Method to use for quarticity estimation
        
        Raises:
            ValueError: If method is not one of the supported methods
        """
        if method not in ['standard', 'tripower', 'quadpower']:
            raise ValueError(
                f"method must be one of 'standard', 'tripower', 'quadpower', got {method}"
            )
        
        self._method = method
        self._name = f"Realized Quarticity ({method})"
        self._fitted = False  # Reset fitted state when method changes
    
    def _compute_realized_measure(self, 
                                 prices: np.ndarray, 
                                 times: np.ndarray, 
                                 returns: np.ndarray,
                                 debiased: bool = True,
                                 **kwargs: Any) -> np.ndarray:
        """
        Compute the realized quarticity from the preprocessed data.
        
        This method implements the core quarticity calculation, with options for
        different estimation methods and subsampling.
        
        Args:
            prices: Preprocessed price data
            times: Preprocessed time points
            returns: Returns computed from prices
            debiased: Whether to apply finite sample bias correction
            **kwargs: Additional keyword arguments for computation
        
        Returns:
            np.ndarray: Realized quarticity
        
        Raises:
            ValueError: If computation fails
            NumericError: If numerical issues are encountered
        """
        # Store debiased flag
        self._debiased = debiased
        
        try:
            # Check if subsampling is enabled
            if self._config.use_subsampling and self._config.subsampling_factor > 1:
                # Compute subsampled quarticity
                quarticity = self._compute_subsampled_quarticity(
                    returns, self._config.subsampling_factor
                )
            else:
                # Compute standard quarticity based on selected method
                if self._method == 'standard':
                    quarticity = self._compute_standard_quarticity(returns, debiased)
                elif self._method == 'tripower':
                    quarticity = self._compute_tripower_quarticity(returns, debiased)
                elif self._method == 'quadpower':
                    quarticity = self._compute_quadpower_quarticity(returns, debiased)
                else:
                    raise ValueError(f"Unsupported method: {self._method}")
            
            return quarticity
            
        except Exception as e:
            logger.error(f"Quarticity computation failed: {str(e)}")
            raise ValueError(f"Quarticity computation failed: {str(e)}") from e
    
    def _compute_standard_quarticity(self, returns: np.ndarray, debiased: bool = True) -> float:
        """
        Compute the standard realized quarticity.
        
        Args:
            returns: Return series
            debiased: Whether to apply finite sample bias correction
        
        Returns:
            float: Standard realized quarticity
        
        Raises:
            NumericError: If numerical issues are encountered
        """
        try:
            # Use Numba-accelerated implementation
            quarticity = _realized_quarticity_core(returns)
            
            # Apply finite sample bias correction if requested
            if not debiased:
                # Remove the finite sample correction applied in the core function
                n = len(returns)
                quarticity = quarticity * (3 / n)
            
            return quarticity
            
        except Exception as e:
            logger.error(f"Standard quarticity computation failed: {str(e)}")
            
            # Fallback to pure NumPy implementation if Numba fails
            logger.warning("Falling back to pure NumPy implementation")
            
            n = len(returns)
            quarticity = np.sum(returns**4)
            
            # Apply finite sample correction if requested
            correction = n / 3 if debiased else 1.0
            
            return correction * quarticity
    
    def _compute_tripower_quarticity(self, returns: np.ndarray, debiased: bool = True) -> float:
        """
        Compute the tri-power quarticity (robust to jumps).
        
        Args:
            returns: Return series
            debiased: Whether to apply finite sample bias correction
        
        Returns:
            float: Tri-power quarticity
        
        Raises:
            NumericError: If numerical issues are encountered
        """
        try:
            # Use Numba-accelerated implementation
            quarticity = _tri_quarticity_core(returns)
            
            # Apply finite sample bias correction if requested
            if not debiased:
                # Remove the finite sample correction applied in the core function
                n = len(returns)
                quarticity = quarticity * ((n - 2) * (n - 1)) / n**2
            
            return quarticity
            
        except Exception as e:
            logger.error(f"Tri-power quarticity computation failed: {str(e)}")
            
            # Fallback to pure NumPy implementation if Numba fails
            logger.warning("Falling back to pure NumPy implementation")
            
            n = len(returns)
            abs_returns = np.abs(returns)
            tri_quarticity_sum = 0.0
            
            for i in range(n-2):
                tri_quarticity_sum += (abs_returns[i]**(4/3)) * (abs_returns[i+1]**(4/3)) * (abs_returns[i+2]**(4/3))
            
            # Scaling factor for asymptotic consistency
            scaling = (2**(4/3)) * (np.pi**(2/3)) / (np.pi + np.pi/2)
            
            # Apply finite sample correction if requested
            correction = n**2 / ((n-2) * (n-1)) if debiased else 1.0
            
            return correction * tri_quarticity_sum * (scaling**3)
    
    def _compute_quadpower_quarticity(self, returns: np.ndarray, debiased: bool = True) -> float:
        """
        Compute the quad-power quarticity (more robust to jumps).
        
        Args:
            returns: Return series
            debiased: Whether to apply finite sample bias correction
        
        Returns:
            float: Quad-power quarticity
        
        Raises:
            NumericError: If numerical issues are encountered
        """
        try:
            n = len(returns)
            abs_returns = np.abs(returns)
            quad_quarticity_sum = 0.0
            
            # Numba acceleration would be ideal here, but we'll implement in pure NumPy
            # for now and add a Numba-accelerated version in _numba_core.py later
            for i in range(n-3):
                quad_quarticity_sum += (abs_returns[i]) * (abs_returns[i+1]) * (abs_returns[i+2]) * (abs_returns[i+3])
            
            # Scaling factor for asymptotic consistency
            scaling = np.pi / 2
            
            # Apply finite sample correction if requested
            correction = n**3 / ((n-3) * (n-2) * (n-1)) if debiased else 1.0
            
            return correction * quad_quarticity_sum * (scaling**4)
            
        except Exception as e:
            logger.error(f"Quad-power quarticity computation failed: {str(e)}")
            raise NumericError(f"Quad-power quarticity computation failed: {str(e)}") from e
    
    def _compute_subsampled_quarticity(self, returns: np.ndarray, subsample_factor: int) -> float:
        """
        Compute subsampled quarticity for noise reduction.
        
        Args:
            returns: Return series
            subsample_factor: Number of subsamples to use
        
        Returns:
            float: Subsampled quarticity
        
        Raises:
            ValueError: If subsample_factor is invalid
            NumericError: If numerical issues are encountered
        """
        # Validate subsample_factor
        if not isinstance(subsample_factor, int) or subsample_factor < 1:
            raise ParameterError(f"subsample_factor must be a positive integer, got {subsample_factor}")
        
        if subsample_factor == 1:
            # No subsampling needed
            if self._method == 'standard':
                return self._compute_standard_quarticity(returns, self._debiased)
            elif self._method == 'tripower':
                return self._compute_tripower_quarticity(returns, self._debiased)
            elif self._method == 'quadpower':
                return self._compute_quadpower_quarticity(returns, self._debiased)
            else:
                raise ValueError(f"Unsupported method: {self._method}")
        
        try:
            n = len(returns)
            subsampled_quarticity = 0.0
            
            for i in range(subsample_factor):
                # Extract i-th subsample
                subsample = returns[i::subsample_factor]
                
                # Skip if subsample is too short
                if self._method == 'standard' and len(subsample) < 1:
                    continue
                elif self._method == 'tripower' and len(subsample) < 3:
                    continue
                elif self._method == 'quadpower' and len(subsample) < 4:
                    continue
                
                # Compute quarticity for this subsample
                if self._method == 'standard':
                    subsample_quarticity = self._compute_standard_quarticity(subsample, self._debiased)
                elif self._method == 'tripower':
                    subsample_quarticity = self._compute_tripower_quarticity(subsample, self._debiased)
                elif self._method == 'quadpower':
                    subsample_quarticity = self._compute_quadpower_quarticity(subsample, self._debiased)
                else:
                    raise ValueError(f"Unsupported method: {self._method}")
                
                # Scale by the number of observations
                scaled_quarticity = subsample_quarticity * (n / len(subsample))
                
                # Add to the total
                subsampled_quarticity += scaled_quarticity
            
            # Average across subsamples
            return subsampled_quarticity / subsample_factor
            
        except Exception as e:
            logger.error(f"Subsampled quarticity computation failed: {str(e)}")
            raise ValueError(f"Subsampled quarticity computation failed: {str(e)}") from e
    
    async def fit_async(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> RealizedEstimatorResult:
        """
        Asynchronously fit the quarticity estimator to the provided data.
        
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
                    computation_time=time.time() - start_time,
                    config=self._config.to_dict()
                )
                
                # Store result
                self._results = result
                
                return result
                
            except Exception as e:
                logger.error(f"Asynchronous estimation failed: {str(e)}")
                raise RuntimeError(f"Quarticity estimation failed: {str(e)}") from e
        
        # Run the computation asynchronously
        return await compute_async()
    
    def calibrate(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> RealizedEstimatorConfig:
        """
        Calibrate the estimator configuration based on the provided data.
        
        This method analyzes the input data and determines optimal configuration
        parameters for the quarticity estimator, such as sampling frequency
        and subsampling factor.
        
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
            
            # Determine optimal subsampling factor based on method
            if self._method == 'standard':
                # Standard quarticity benefits from more subsampling
                optimal_subsample = min(10, len(returns) // 100)
            elif self._method == 'tripower':
                # Tripower quarticity already has some robustness
                optimal_subsample = min(5, len(returns) // 100)
            else:  # 'quadpower'
                # Quadpower quarticity is most robust
                optimal_subsample = min(3, len(returns) // 100)
            
            optimal_subsample = max(1, optimal_subsample)  # Ensure at least 1
            
            # Create calibrated configuration
            calibrated_config = RealizedEstimatorConfig(
                sampling_frequency=optimal_freq,
                annualize=self._config.annualize,
                annualization_factor=self._config.annualization_factor,
                return_type=self._config.return_type,
                use_subsampling=optimal_subsample > 1,
                subsampling_factor=optimal_subsample,
                apply_noise_correction=False,  # Quarticity doesn't use noise correction
                time_unit=self._config.time_unit
            )
            
            logger.info(
                f"Calibrated configuration: sampling_frequency={optimal_freq}, "
                f"subsampling_factor={optimal_subsample}"
            )
            
            return calibrated_config
            
        except Exception as e:
            logger.error(f"Calibration failed: {str(e)}")
            raise ValueError(f"Quarticity calibration failed: {str(e)}") from e
    
    def get_asymptotic_variance(self, realized_variance: Optional[float] = None) -> float:
        """
        Get the asymptotic variance of realized variance based on quarticity.
        
        This method computes the asymptotic variance of realized variance,
        which is essential for constructing confidence intervals and conducting
        hypothesis tests about volatility.
        
        Args:
            realized_variance: Realized variance estimate (if None, will be computed from returns)
        
        Returns:
            float: Asymptotic variance of realized variance
        
        Raises:
            RuntimeError: If the estimator has not been fitted
        """
        if not self._fitted or self._realized_measure is None:
            raise RuntimeError("Estimator has not been fitted. Call fit() first.")
        
        if self._returns is None:
            raise RuntimeError("Returns are not available")
        
        # Compute realized variance if not provided
        if realized_variance is None:
            realized_variance = np.sum(self._returns ** 2)
        
        # Compute asymptotic variance based on quarticity
        n = len(self._returns)
        asymptotic_variance = self._realized_measure / (n ** 2)
        
        return asymptotic_variance
    
    def get_confidence_interval(self, 
                               realized_variance: Optional[float] = None,
                               confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Get confidence interval for integrated variance based on quarticity.
        
        Args:
            realized_variance: Realized variance estimate (if None, will be computed from returns)
            confidence_level: Confidence level (default: 0.95)
        
        Returns:
            Tuple[float, float]: Lower and upper bounds of the confidence interval
        
        Raises:
            RuntimeError: If the estimator has not been fitted
            ValueError: If confidence_level is not in (0, 1)
        """
        if not (0 < confidence_level < 1):
            raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")
        
        if not self._fitted or self._realized_measure is None:
            raise RuntimeError("Estimator has not been fitted. Call fit() first.")
        
        if self._returns is None:
            raise RuntimeError("Returns are not available")
        
        # Compute realized variance if not provided
        if realized_variance is None:
            realized_variance = np.sum(self._returns ** 2)
        
        # Compute asymptotic variance
        asymptotic_variance = self.get_asymptotic_variance(realized_variance)
        
        # Compute standard error
        standard_error = np.sqrt(asymptotic_variance)
        
        # Compute critical value
        from scipy import stats
        alpha = 1 - confidence_level
        critical_value = stats.norm.ppf(1 - alpha / 2)
        
        # Compute confidence interval
        lower_bound = realized_variance - critical_value * standard_error
        upper_bound = realized_variance + critical_value * standard_error
        
        # Ensure lower bound is non-negative
        lower_bound = max(0, lower_bound)
        
        return lower_bound, upper_bound
    
    def plot_quarticity(self, figsize: Tuple[int, int] = (10, 6)) -> Any:
        """
        Plot quarticity estimate with confidence bands.
        
        Args:
            figsize: Figure size as (width, height) in inches
        
        Returns:
            matplotlib.figure.Figure: The created figure
        
        Raises:
            RuntimeError: If the estimator has not been fitted
            ImportError: If matplotlib is not available
        """
        if not self._fitted or self._realized_measure is None:
            raise RuntimeError("Estimator has not been fitted. Call fit() first.")
        
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot quarticity estimate
            ax.axhline(y=self._realized_measure, color='b', linestyle='-', 
                      label=f'Quarticity ({self._method})')
            
            # Add confidence bands if returns are available
            if self._returns is not None:
                # Compute standard error
                asymptotic_variance = self.get_asymptotic_variance()
                standard_error = np.sqrt(asymptotic_variance)
                
                # Compute 95% confidence bands
                lower_band = self._realized_measure - 1.96 * standard_error
                upper_band = self._realized_measure + 1.96 * standard_error
                
                # Ensure lower band is non-negative
                lower_band = max(0, lower_band)
                
                # Plot confidence bands
                ax.axhline(y=lower_band, color='r', linestyle='--', 
                          alpha=0.5, label='95% Confidence Band')
                ax.axhline(y=upper_band, color='r', linestyle='--', 
                          alpha=0.5)
            
            ax.set_title(f'Realized Quarticity Estimate ({self._method})')
            ax.set_xlabel('Observation')
            ax.set_ylabel('Quarticity')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except ImportError:
            logger.warning("Matplotlib is required for plotting")
            raise ImportError("Matplotlib is required for plotting")
    
    def summary(self) -> str:
        """
        Generate a text summary of the quarticity estimator.
        
        Returns:
            str: A formatted string containing the estimator summary
        
        Raises:
            RuntimeError: If the estimator has not been fitted
        """
        if not self._fitted:
            return f"Realized Quarticity Estimator ({self._method}, not fitted)"
        
        if self._results is None:
            return f"Realized Quarticity Estimator ({self._method}, fitted, but no results available)"
        
        base_summary = self._results.summary()
        
        # Add quarticity-specific information
        additional_info = ""
        
        # Add method information
        additional_info += f"Method: {self._method}\n"
        
        # Add debiasing information
        additional_info += f"Debiased: {self._debiased}\n"
        
        # Add asymptotic variance if returns are available
        if self._returns is not None:
            asymptotic_variance = self.get_asymptotic_variance()
            additional_info += f"Asymptotic Variance: {asymptotic_variance:.6e}\n"
            
            # Add 95% confidence interval for realized variance
            realized_variance = np.sum(self._returns ** 2)
            lower_bound, upper_bound = self.get_confidence_interval(realized_variance)
            additional_info += f"95% CI for Realized Variance: [{lower_bound:.6f}, {upper_bound:.6f}]\n"
        
        if additional_info:
            additional_info = "\nQuarticity Information:\n" + additional_info
        
        return base_summary + additional_info
    
    def __repr__(self) -> str:
        """
        Generate a detailed string representation of the estimator.
        
        Returns:
            str: A detailed string representation of the estimator
        """
        fitted_str = "fitted" if self._fitted else "not fitted"
        method_str = f", method='{self._method}'"
        config_str = f", config={self._config}" if self._config else ""
        return f"RealizedQuarticity({fitted_str}{method_str}{config_str})"