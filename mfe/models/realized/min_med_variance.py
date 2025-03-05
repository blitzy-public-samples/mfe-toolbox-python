# mfe/models/realized/min_med_variance.py
"""
Truncated realized variance estimators using median and minimum methods.

This module implements the truncated realized variance estimators proposed by
Andersen, Dobrev, and Schaumburg (2012) that use median and minimum methods
to provide robust volatility estimation in the presence of jumps. These estimators
are alternatives to bipower variation that offer improved finite-sample properties
and robustness to jumps.

The implementation follows a class-based design inheriting from JumpRobustEstimator,
with comprehensive type hints, parameter validation, and Numba-accelerated core
calculations for optimal performance. Both NumPy arrays and Pandas DataFrame inputs
are supported, with visualization methods for comparing truncation effects.

References:
    Andersen, T. G., Dobrev, D., & Schaumburg, E. (2012). Jump-robust volatility
    estimation using nearest neighbor truncation. Journal of Econometrics, 169(1), 75-93.
"""

import logging
import warnings
from typing import Any, Dict, Optional, Tuple, Union, cast, List

import numpy as np
import pandas as pd
from numba import jit, njit

from .base import JumpRobustEstimator, RealizedEstimatorConfig, RealizedEstimatorResult
from ._numba_core import _compute_subsampled_measure_numba
from ...core.exceptions import ParameterError, NumericError

# Set up module-level logger
logger = logging.getLogger("mfe.models.realized.min_med_variance")


@njit(cache=True)
def _minrv_core(returns: np.ndarray) -> float:
    """
    Numba-accelerated core implementation of MinRV estimator.
    
    Args:
        returns: Array of returns
        
    Returns:
        MinRV estimate
    """
    n = len(returns)
    min_rv_sum = 0.0
    
    # Scaling factor for asymptotic consistency
    scaling = np.pi / (np.pi - 2)
    
    # Apply finite sample correction
    correction = n / (n - 1)
    
    # Compute MinRV
    for i in range(n - 1):
        # Take minimum of adjacent squared returns
        min_squared = min(returns[i]**2, returns[i+1]**2)
        min_rv_sum += min_squared
    
    return correction * scaling * min_rv_sum


@njit(cache=True)
def _medrv_core(returns: np.ndarray) -> float:
    """
    Numba-accelerated core implementation of MedRV estimator.
    
    Args:
        returns: Array of returns
        
    Returns:
        MedRV estimate
    """
    n = len(returns)
    med_rv_sum = 0.0
    
    # Scaling factor for asymptotic consistency
    scaling = np.pi / (6 - 4 * np.sqrt(3) + np.pi)
    
    # Apply finite sample correction
    correction = n / (n - 2)
    
    # Compute MedRV
    for i in range(1, n - 1):
        # Take median of three consecutive squared returns
        med_squared = np.median([
            returns[i-1]**2, 
            returns[i]**2, 
            returns[i+1]**2
        ])
        med_rv_sum += med_squared
    
    return correction * scaling * med_rv_sum


class MinMedVariance(JumpRobustEstimator):
    """
    Truncated realized variance estimators using median and minimum methods.
    
    This class implements the MinRV and MedRV estimators proposed by Andersen,
    Dobrev, and Schaumburg (2012) that provide robust volatility estimation in
    the presence of jumps. These estimators use the minimum or median of adjacent
    squared returns to reduce the impact of jumps on volatility estimation.
    
    The estimator inherits from JumpRobustEstimator, providing specialized
    functionality for jump detection and separation of continuous and jump
    components of volatility.
    
    Attributes:
        config: Configuration parameters for the estimator
        jump_threshold: Threshold used for jump detection (if jumps were detected)
        jump_indicators: Boolean array indicating detected jumps (if jumps were detected)
        estimator_type: Type of estimator ('min' or 'med')
    """
    
    def __init__(self, 
                config: Optional[RealizedEstimatorConfig] = None,
                estimator_type: str = 'med'):
        """
        Initialize the MinMedVariance estimator.
        
        Args:
            config: Configuration parameters for the estimator
            estimator_type: Type of estimator to use ('min' or 'med')
        
        Raises:
            ValueError: If estimator_type is not 'min' or 'med'
        """
        # Validate estimator_type
        if estimator_type.lower() not in ['min', 'med']:
            raise ValueError(f"estimator_type must be 'min' or 'med', got {estimator_type}")
        
        # Set name based on estimator type
        name = "MinRV" if estimator_type.lower() == 'min' else "MedRV"
        
        super().__init__(config=config, name=name)
        
        # Store estimator type
        self._estimator_type = estimator_type.lower()
    
    @property
    def estimator_type(self) -> str:
        """
        Get the type of estimator ('min' or 'med').
        
        Returns:
            str: The estimator type
        """
        return self._estimator_type
    
    @estimator_type.setter
    def estimator_type(self, value: str) -> None:
        """
        Set the type of estimator.
        
        Args:
            value: Type of estimator ('min' or 'med')
            
        Raises:
            ValueError: If value is not 'min' or 'med'
        """
        if value.lower() not in ['min', 'med']:
            raise ValueError(f"estimator_type must be 'min' or 'med', got {value}")
        
        self._estimator_type = value.lower()
        self._name = "MinRV" if value.lower() == 'min' else "MedRV"
        self._fitted = False  # Reset fitted state when estimator type changes
    
    def _compute_realized_measure(self, 
                                 prices: np.ndarray, 
                                 times: np.ndarray, 
                                 returns: np.ndarray,
                                 detect_jumps: bool = False,
                                 threshold_multiplier: float = 3.0,
                                 **kwargs: Any) -> np.ndarray:
        """
        Compute the truncated realized variance from the preprocessed data.
        
        This method implements the core MinRV or MedRV calculation, with
        options for subsampling. It can also detect jumps in the return series
        if requested.
        
        Args:
            prices: Preprocessed price data
            times: Preprocessed time points
            returns: Returns computed from prices
            detect_jumps: Whether to detect jumps in the return series
            threshold_multiplier: Multiplier for jump detection threshold
            **kwargs: Additional keyword arguments for computation
        
        Returns:
            np.ndarray: Truncated realized variance
        
        Raises:
            ValueError: If computation fails
            NumericError: If numerical issues are encountered
        """
        try:
            # Check if subsampling is enabled
            if self._config.use_subsampling and self._config.subsampling_factor > 1:
                # Compute subsampled truncated variance
                rv = self._compute_subsampled_truncated_variance(
                    returns, self._config.subsampling_factor
                )
            else:
                # Compute standard truncated variance
                rv = self._compute_truncated_variance(returns)
            
            # Detect jumps if requested
            if detect_jumps:
                jump_indicators, threshold = self.detect_jumps(
                    returns, threshold_multiplier, **kwargs
                )
                
                # Store jump detection results
                self._jump_threshold = threshold
                self._jump_indicators = jump_indicators
                
                # Log jump detection results
                jump_count = np.sum(jump_indicators)
                logger.info(
                    f"Detected {jump_count} jumps ({jump_count / len(returns) * 100:.2f}%) "
                    f"with threshold {threshold:.6f}"
                )
            
            return rv
            
        except Exception as e:
            logger.error(f"Truncated variance computation failed: {str(e)}")
            raise ValueError(f"Truncated variance computation failed: {str(e)}") from e
    
    def _compute_truncated_variance(self, returns: np.ndarray) -> float:
        """
        Compute the truncated variance using the selected method.
        
        Args:
            returns: Return series
        
        Returns:
            float: Truncated variance
        
        Raises:
            NumericError: If numerical issues are encountered
        """
        try:
            # Use appropriate Numba-accelerated implementation based on estimator type
            if self._estimator_type == 'min':
                rv = _minrv_core(returns)
            else:  # 'med'
                rv = _medrv_core(returns)
            
            return rv
            
        except Exception as e:
            logger.error(f"Truncated variance core computation failed: {str(e)}")
            
            # Fallback to pure NumPy implementation if Numba fails
            logger.warning("Falling back to pure NumPy implementation")
            
            n = len(returns)
            
            if self._estimator_type == 'min':
                # MinRV implementation
                min_rv_sum = 0.0
                for i in range(n - 1):
                    min_squared = min(returns[i]**2, returns[i+1]**2)
                    min_rv_sum += min_squared
                
                # Scaling factor for asymptotic consistency
                scaling = np.pi / (np.pi - 2)
                
                # Apply finite sample correction
                correction = n / (n - 1)
                
                return correction * scaling * min_rv_sum
            
            else:  # 'med'
                # MedRV implementation
                med_rv_sum = 0.0
                for i in range(1, n - 1):
                    med_rv_sum += np.median([
                        returns[i-1]**2, 
                        returns[i]**2, 
                        returns[i+1]**2
                    ])
                
                # Scaling factor for asymptotic consistency
                scaling = np.pi / (6 - 4 * np.sqrt(3) + np.pi)
                
                # Apply finite sample correction
                correction = n / (n - 2)
                
                return correction * scaling * med_rv_sum
    
    def _compute_subsampled_truncated_variance(self, returns: np.ndarray, subsample_factor: int) -> float:
        """
        Compute subsampled truncated variance for noise reduction.
        
        Args:
            returns: Return series
            subsample_factor: Number of subsamples to use
        
        Returns:
            float: Subsampled truncated variance
        
        Raises:
            ValueError: If subsample_factor is invalid
            NumericError: If numerical issues are encountered
        """
        # Validate subsample_factor
        if not isinstance(subsample_factor, int) or subsample_factor < 1:
            raise ParameterError(f"subsample_factor must be a positive integer, got {subsample_factor}")
        
        if subsample_factor == 1:
            # No subsampling needed
            return self._compute_truncated_variance(returns)
        
        try:
            n = len(returns)
            subsampled_rv = 0.0
            
            for i in range(subsample_factor):
                # Extract i-th subsample
                subsample = returns[i::subsample_factor]
                
                # Skip if subsample is too short
                if (self._estimator_type == 'min' and len(subsample) < 2) or \
                   (self._estimator_type == 'med' and len(subsample) < 3):
                    continue
                
                # Compute truncated variance for this subsample
                subsample_rv = self._compute_truncated_variance(subsample)
                
                # Scale by the number of observations
                scaled_rv = subsample_rv * (n / len(subsample))
                
                # Add to the total
                subsampled_rv += scaled_rv
            
            # Average across subsamples
            return subsampled_rv / subsample_factor
            
        except Exception as e:
            logger.error(f"Subsampled truncated variance computation failed: {str(e)}")
            raise ValueError(f"Subsampled truncated variance computation failed: {str(e)}") from e
    
    async def fit_async(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> RealizedEstimatorResult:
        """
        Asynchronously fit the truncated variance estimator to the provided data.
        
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
                raise RuntimeError(f"Truncated variance estimation failed: {str(e)}") from e
        
        # Run the computation asynchronously
        return await compute_async()
    
    def calibrate(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> RealizedEstimatorConfig:
        """
        Calibrate the estimator configuration based on the provided data.
        
        This method analyzes the input data and determines optimal configuration
        parameters for the truncated variance estimator, such as sampling frequency
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
            
            # Determine optimal subsampling factor
            # For truncated variance, a moderate subsampling factor is usually sufficient
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
                apply_noise_correction=False,  # Truncated variance doesn't use noise correction
                time_unit=self._config.time_unit
            )
            
            logger.info(
                f"Calibrated configuration: sampling_frequency={optimal_freq}, "
                f"subsampling_factor={optimal_subsample}"
            )
            
            return calibrated_config
            
        except Exception as e:
            logger.error(f"Calibration failed: {str(e)}")
            raise ValueError(f"Truncated variance calibration failed: {str(e)}") from e
    
    def get_jump_ratio(self) -> Optional[float]:
        """
        Get the ratio of jump variation to total variation.
        
        Returns:
            Optional[float]: The jump ratio if jumps were detected, None otherwise
        
        Raises:
            RuntimeError: If the estimator has not been fitted
        """
        if not self._fitted or self._realized_measure is None:
            raise RuntimeError("Estimator has not been fitted. Call fit() first.")
        
        if self._jump_indicators is None or self._returns is None:
            return None
        
        # Get jump variation
        jump_variation = self.get_jump_variation()
        
        if jump_variation is None:
            return None
        
        # Compute total variation (realized variance)
        total_variation = np.sum(self._returns ** 2)
        
        # Compute jump ratio
        jump_ratio = jump_variation / total_variation
        
        return jump_ratio
    
    def get_continuous_ratio(self) -> Optional[float]:
        """
        Get the ratio of continuous variation to total variation.
        
        Returns:
            Optional[float]: The continuous ratio if jumps were detected, None otherwise
        
        Raises:
            RuntimeError: If the estimator has not been fitted
        """
        jump_ratio = self.get_jump_ratio()
        
        if jump_ratio is None:
            return None
        
        return 1.0 - jump_ratio
    
    def compare_with_rv(self, figsize: Tuple[int, int] = (10, 6)) -> Any:
        """
        Compare truncated variance with standard realized variance.
        
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
            
            # Compute standard realized variance
            rv = np.sum(self._returns ** 2)
            
            # Get truncated variance
            trunc_rv = self._realized_measure
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Create bar chart
            labels = ['Realized Variance', self._name]
            values = [rv, trunc_rv]
            
            ax.bar(labels, values, color=['blue', 'green'])
            
            # Add values on top of bars
            for i, v in enumerate(values):
                ax.text(i, v * 1.01, f'{v:.6f}', ha='center')
            
            # Add title and labels
            ax.set_title(f'Comparison of Standard RV and {self._name}')
            ax.set_ylabel('Variance')
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except ImportError:
            logger.warning("Matplotlib is required for plotting")
            raise ImportError("Matplotlib is required for plotting")
    
    def plot_returns_with_jumps(self, figsize: Tuple[int, int] = (10, 6)) -> Any:
        """
        Plot returns with detected jumps highlighted.
        
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
            
            # Highlight jumps if detected
            if self._jump_indicators is not None:
                jump_indices = np.where(self._jump_indicators)[0]
                jump_returns = self._returns[jump_indices]
                ax.scatter(jump_indices, jump_returns, color='r', s=50, 
                          marker='o', label='Jumps')
                
                # Add threshold lines if available
                if self._jump_threshold is not None:
                    ax.axhline(y=self._jump_threshold, color='r', linestyle='--', 
                              alpha=0.5, label=f'Threshold (+{self._jump_threshold:.4f})')
                    ax.axhline(y=-self._jump_threshold, color='r', linestyle='--', 
                              alpha=0.5, label=f'Threshold (-{self._jump_threshold:.4f})')
            
            ax.set_title(f'Returns with Detected Jumps ({self._name})')
            ax.set_xlabel('Observation')
            ax.set_ylabel('Return')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except ImportError:
            logger.warning("Matplotlib is required for plotting")
            raise ImportError("Matplotlib is required for plotting")
    
    def compare_estimators(self, returns: np.ndarray, figsize: Tuple[int, int] = (10, 6)) -> Any:
        """
        Compare MinRV and MedRV estimators on the same data.
        
        Args:
            returns: Return series to compare estimators on
            figsize: Figure size as (width, height) in inches
        
        Returns:
            matplotlib.figure.Figure: The created figure
        
        Raises:
            ImportError: If matplotlib is not available
        """
        try:
            import matplotlib.pyplot as plt
            
            # Create temporary estimators
            min_estimator = MinMedVariance(estimator_type='min')
            med_estimator = MinMedVariance(estimator_type='med')
            
            # Compute estimates
            min_rv = min_estimator._compute_truncated_variance(returns)
            med_rv = med_estimator._compute_truncated_variance(returns)
            rv = np.sum(returns ** 2)
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Create bar chart
            labels = ['Realized Variance', 'MinRV', 'MedRV']
            values = [rv, min_rv, med_rv]
            
            ax.bar(labels, values, color=['blue', 'green', 'orange'])
            
            # Add values on top of bars
            for i, v in enumerate(values):
                ax.text(i, v * 1.01, f'{v:.6f}', ha='center')
            
            # Add title and labels
            ax.set_title('Comparison of Variance Estimators')
            ax.set_ylabel('Variance')
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except ImportError:
            logger.warning("Matplotlib is required for plotting")
            raise ImportError("Matplotlib is required for plotting")
    
    def summary(self) -> str:
        """
        Generate a text summary of the truncated variance estimator.
        
        Returns:
            str: A formatted string containing the estimator summary
        
        Raises:
            RuntimeError: If the estimator has not been fitted
        """
        if not self._fitted:
            return f"{self._name} Estimator (not fitted)"
        
        if self._results is None:
            return f"{self._name} Estimator (fitted, but no results available)"
        
        base_summary = self._results.summary()
        
        # Add estimator-specific information
        additional_info = f"Estimator Type: {self._estimator_type.upper()}\n"
        
        # Add jump ratio if jumps were detected
        jump_ratio = self.get_jump_ratio()
        if jump_ratio is not None:
            additional_info += f"Jump Ratio: {jump_ratio:.4f} ({jump_ratio * 100:.2f}%)\n"
            additional_info += f"Continuous Ratio: {1 - jump_ratio:.4f} ({(1 - jump_ratio) * 100:.2f}%)\n"
        
        if additional_info:
            additional_info = f"\n{self._name} Information:\n" + additional_info
        
        return base_summary + additional_info
    
    def __repr__(self) -> str:
        """
        Generate a detailed string representation of the estimator.
        
        Returns:
            str: A detailed string representation of the estimator
        """
        fitted_str = "fitted" if self._fitted else "not fitted"
        config_str = f", config={self._config}" if self._config else ""
        return f"MinMedVariance(type='{self._estimator_type}', {fitted_str}{config_str})"