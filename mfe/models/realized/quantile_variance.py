"""
Quantile Realized Variance (RQ) estimator for robust volatility measurement.

This module implements the Quantile Realized Variance (RQ) estimator, which provides
a robust measure of volatility that is less sensitive to outliers and jumps than
standard realized variance. The quantile variance is computed based on the quantiles
of the return distribution, offering an alternative approach to jump-robust estimation.

The estimator inherits from JumpRobustEstimator, providing specialized functionality
for jump detection and robust volatility estimation. Performance-critical calculations
are accelerated using Numba's JIT compilation for efficient processing of high-frequency
data.

The implementation supports various configuration options including subsampling
for noise reduction, different return types, and asynchronous processing for
long computations. It also provides visualization methods for examining the
contributions of different quantiles to the overall variance estimate.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd

from .base import JumpRobustEstimator, RealizedEstimatorConfig, RealizedEstimatorResult
from ._numba_core import _compute_subsampled_measure_numba
from ...core.exceptions import ParameterError, NumericError

# Set up module-level logger
logger = logging.getLogger("mfe.models.realized.quantile_variance")

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
    logger.debug("Numba available for quantile variance acceleration")
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    HAS_NUMBA = False
    logger.info("Numba not available. Quantile variance will use pure NumPy implementations.")


@jit(nopython=True, cache=True)
def _quantile_variance_core(returns: np.ndarray, alpha: float = 0.5) -> float:
    """
    Numba-accelerated core implementation of quantile variance computation.
    
    Args:
        returns: Array of returns
        alpha: Quantile level (0.5 for median)
        
    Returns:
        Quantile variance estimate
    """
    n = len(returns)
    
    # Sort returns by absolute value
    abs_returns = np.abs(returns)
    sorted_indices = np.argsort(abs_returns)
    
    # Compute quantile index
    q_index = int(np.floor(alpha * n))
    
    # Ensure index is valid
    q_index = max(0, min(q_index, n - 1))
    
    # Get quantile value
    quantile = abs_returns[sorted_indices[q_index]]
    
    # Compute scaling factor for consistency with realized variance
    # For alpha=0.5 (median), the scaling factor is approximately 1.57
    # For normal distribution, the scaling factor is 1/(Φ^(-1)(0.5 + α/2))^2
    # where Φ^(-1) is the inverse of the standard normal CDF
    if alpha == 0.5:
        # Hardcoded scaling for median (most common case)
        scaling = 1.57
    else:
        # Approximate scaling for other quantiles
        # This is an approximation of the theoretical scaling factor
        scaling = 1.0 / (alpha * (1.0 - alpha))
    
    # Compute quantile variance
    return n * quantile**2 * scaling


@jit(nopython=True, cache=True)
def _quantile_variance_multiple_core(returns: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """
    Numba-accelerated core implementation of multiple quantile variance computation.
    
    Args:
        returns: Array of returns
        alphas: Array of quantile levels
        
    Returns:
        Array of quantile variance estimates for each alpha
    """
    n = len(returns)
    n_alphas = len(alphas)
    results = np.zeros(n_alphas)
    
    # Sort returns by absolute value
    abs_returns = np.abs(returns)
    sorted_abs_returns = np.sort(abs_returns)
    
    for i, alpha in enumerate(alphas):
        # Compute quantile index
        q_index = int(np.floor(alpha * n))
        
        # Ensure index is valid
        q_index = max(0, min(q_index, n - 1))
        
        # Get quantile value
        quantile = sorted_abs_returns[q_index]
        
        # Compute scaling factor
        if alpha == 0.5:
            # Hardcoded scaling for median (most common case)
            scaling = 1.57
        else:
            # Approximate scaling for other quantiles
            scaling = 1.0 / (alpha * (1.0 - alpha))
        
        # Compute quantile variance
        results[i] = n * quantile**2 * scaling
    
    return results


@jit(nopython=True, cache=True)
def _subsampled_quantile_variance_core(returns: np.ndarray, alpha: float, subsample_factor: int) -> float:
    """
    Numba-accelerated core implementation of subsampled quantile variance.
    
    Args:
        returns: Array of returns
        alpha: Quantile level
        subsample_factor: Number of subsamples
        
    Returns:
        Subsampled quantile variance estimate
    """
    n = len(returns)
    subsampled_qv = 0.0
    
    for i in range(subsample_factor):
        # Extract i-th subsample
        subsample = returns[i::subsample_factor]
        
        # Skip if subsample is too short
        if len(subsample) < 2:
            continue
        
        # Compute quantile variance for this subsample
        subsample_qv = _quantile_variance_core(subsample, alpha)
        
        # Scale by the number of observations
        scaled_qv = subsample_qv * (n / len(subsample))
        
        # Add to the total
        subsampled_qv += scaled_qv
    
    # Average across subsamples
    return subsampled_qv / subsample_factor


class QuantileVariance(JumpRobustEstimator):
    """
    Quantile Realized Variance (RQ) estimator for robust volatility measurement.
    
    This class implements the Quantile Realized Variance (RQ) estimator, which provides
    a robust measure of volatility that is less sensitive to outliers and jumps than
    standard realized variance. The quantile variance is computed based on the quantiles
    of the return distribution, offering an alternative approach to jump-robust estimation.
    
    The estimator inherits from JumpRobustEstimator, providing specialized
    functionality for jump detection and robust volatility estimation.
    
    Attributes:
        config: Configuration parameters for the estimator
        jump_threshold: Threshold used for jump detection (if jumps were detected)
        jump_indicators: Boolean array indicating detected jumps (if jumps were detected)
        alpha: Quantile level used for estimation (default: 0.5 for median)
        quantile_contributions: Contributions of different quantiles to the variance estimate
    """
    
    def __init__(self, config: Optional[RealizedEstimatorConfig] = None, alpha: float = 0.5):
        """
        Initialize the quantile variance estimator.
        
        Args:
            config: Configuration parameters for the estimator
            alpha: Quantile level (0.5 for median)
        """
        super().__init__(config=config, name="Quantile Variance")
        
        # Validate alpha
        if not 0 < alpha < 1:
            raise ParameterError(f"alpha must be between 0 and 1, got {alpha}")
        
        # Additional attributes specific to quantile variance
        self._alpha: float = alpha
        self._quantile_contributions: Optional[Dict[float, float]] = None
    
    @property
    def alpha(self) -> float:
        """
        Get the quantile level used for estimation.
        
        Returns:
            float: The quantile level
        """
        return self._alpha
    
    @alpha.setter
    def alpha(self, value: float) -> None:
        """
        Set the quantile level used for estimation.
        
        Args:
            value: New quantile level (must be between 0 and 1)
            
        Raises:
            ParameterError: If value is not between 0 and 1
        """
        if not 0 < value < 1:
            raise ParameterError(f"alpha must be between 0 and 1, got {value}")
        
        self._alpha = value
        self._fitted = False  # Reset fitted state when alpha changes
    
    @property
    def quantile_contributions(self) -> Optional[Dict[float, float]]:
        """
        Get the contributions of different quantiles to the variance estimate.
        
        Returns:
            Optional[Dict[float, float]]: Dictionary mapping quantile levels to variance contributions,
                                         or None if not computed
        """
        return self._quantile_contributions
    
    def _compute_realized_measure(self, 
                                 prices: np.ndarray, 
                                 times: np.ndarray, 
                                 returns: np.ndarray,
                                 alpha: Optional[float] = None,
                                 compute_contributions: bool = False,
                                 detect_jumps: bool = False,
                                 threshold_multiplier: float = 3.0,
                                 **kwargs: Any) -> np.ndarray:
        """
        Compute the quantile variance from the preprocessed data.
        
        This method implements the core quantile variance calculation, with
        options for different quantile levels, subsampling, and jump detection.
        
        Args:
            prices: Preprocessed price data
            times: Preprocessed time points
            returns: Returns computed from prices
            alpha: Quantile level (overrides the instance attribute if provided)
            compute_contributions: Whether to compute contributions of different quantiles
            detect_jumps: Whether to detect jumps in the return series
            threshold_multiplier: Multiplier for jump detection threshold
            **kwargs: Additional keyword arguments for computation
        
        Returns:
            np.ndarray: Quantile variance
        
        Raises:
            ValueError: If computation fails
            NumericError: If numerical issues are encountered
        """
        # Use provided alpha or instance attribute
        alpha_value = alpha if alpha is not None else self._alpha
        
        # Validate alpha
        if not 0 < alpha_value < 1:
            raise ParameterError(f"alpha must be between 0 and 1, got {alpha_value}")
        
        try:
            # Check if subsampling is enabled
            if self._config.use_subsampling and self._config.subsampling_factor > 1:
                # Compute subsampled quantile variance
                qv = self._compute_subsampled_quantile_variance(
                    returns, alpha_value, self._config.subsampling_factor
                )
            else:
                # Compute standard quantile variance
                qv = self._compute_quantile_variance(returns, alpha_value)
            
            # Compute contributions of different quantiles if requested
            if compute_contributions:
                self._compute_quantile_contributions(returns)
            
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
            
            return qv
            
        except Exception as e:
            logger.error(f"Quantile variance computation failed: {str(e)}")
            raise ValueError(f"Quantile variance computation failed: {str(e)}") from e
    
    def _compute_quantile_variance(self, returns: np.ndarray, alpha: float = 0.5) -> float:
        """
        Compute the standard quantile variance.
        
        Args:
            returns: Return series
            alpha: Quantile level
        
        Returns:
            float: Quantile variance
        
        Raises:
            NumericError: If numerical issues are encountered
        """
        try:
            # Use Numba-accelerated implementation if available
            if HAS_NUMBA:
                qv = _quantile_variance_core(returns, alpha)
                return qv
            
            # Pure NumPy implementation if Numba is not available
            n = len(returns)
            
            # Sort returns by absolute value
            abs_returns = np.abs(returns)
            sorted_indices = np.argsort(abs_returns)
            
            # Compute quantile index
            q_index = int(np.floor(alpha * n))
            
            # Ensure index is valid
            q_index = max(0, min(q_index, n - 1))
            
            # Get quantile value
            quantile = abs_returns[sorted_indices[q_index]]
            
            # Compute scaling factor for consistency with realized variance
            if alpha == 0.5:
                # Hardcoded scaling for median (most common case)
                scaling = 1.57
            else:
                # Approximate scaling for other quantiles
                scaling = 1.0 / (alpha * (1.0 - alpha))
            
            # Compute quantile variance
            return n * quantile**2 * scaling
            
        except Exception as e:
            logger.error(f"Quantile variance core computation failed: {str(e)}")
            raise NumericError(f"Quantile variance computation failed: {str(e)}") from e
    
    def _compute_subsampled_quantile_variance(self, returns: np.ndarray, 
                                             alpha: float = 0.5,
                                             subsample_factor: int = 5) -> float:
        """
        Compute subsampled quantile variance for noise reduction.
        
        Args:
            returns: Return series
            alpha: Quantile level
            subsample_factor: Number of subsamples to use
        
        Returns:
            float: Subsampled quantile variance
        
        Raises:
            ValueError: If subsample_factor is invalid
            NumericError: If numerical issues are encountered
        """
        # Validate subsample_factor
        if not isinstance(subsample_factor, int) or subsample_factor < 1:
            raise ParameterError(f"subsample_factor must be a positive integer, got {subsample_factor}")
        
        if subsample_factor == 1:
            # No subsampling needed
            return self._compute_quantile_variance(returns, alpha)
        
        try:
            # Use Numba-accelerated implementation if available
            if HAS_NUMBA:
                qv = _subsampled_quantile_variance_core(returns, alpha, subsample_factor)
                return qv
            
            # Pure NumPy implementation if Numba is not available
            n = len(returns)
            subsampled_qv = 0.0
            
            for i in range(subsample_factor):
                # Extract i-th subsample
                subsample = returns[i::subsample_factor]
                
                # Skip if subsample is too short
                if len(subsample) < 2:
                    continue
                
                # Compute quantile variance for this subsample
                subsample_qv = self._compute_quantile_variance(subsample, alpha)
                
                # Scale by the number of observations
                scaled_qv = subsample_qv * (n / len(subsample))
                
                # Add to the total
                subsampled_qv += scaled_qv
            
            # Average across subsamples
            return subsampled_qv / subsample_factor
            
        except Exception as e:
            logger.error(f"Subsampled quantile variance computation failed: {str(e)}")
            raise ValueError(f"Subsampled quantile variance computation failed: {str(e)}") from e
    
    def _compute_quantile_contributions(self, returns: np.ndarray, 
                                       num_quantiles: int = 10) -> Dict[float, float]:
        """
        Compute contributions of different quantiles to the variance estimate.
        
        Args:
            returns: Return series
            num_quantiles: Number of quantiles to compute
        
        Returns:
            Dict[float, float]: Dictionary mapping quantile levels to variance contributions
        
        Raises:
            NumericError: If numerical issues are encountered
        """
        try:
            # Generate quantile levels
            alphas = np.linspace(0.1, 0.9, num_quantiles)
            
            # Compute quantile variances for each alpha
            if HAS_NUMBA:
                # Use Numba-accelerated implementation for multiple quantiles
                qvs = _quantile_variance_multiple_core(returns, alphas)
            else:
                # Pure NumPy implementation
                qvs = np.array([self._compute_quantile_variance(returns, alpha) for alpha in alphas])
            
            # Create dictionary of contributions
            contributions = {alpha: qv for alpha, qv in zip(alphas, qvs)}
            
            # Store contributions
            self._quantile_contributions = contributions
            
            return contributions
            
        except Exception as e:
            logger.error(f"Quantile contributions computation failed: {str(e)}")
            raise NumericError(f"Quantile contributions computation failed: {str(e)}") from e
    
    async def fit_async(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> RealizedEstimatorResult:
        """
        Asynchronously fit the quantile variance estimator to the provided data.
        
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
                raise RuntimeError(f"Quantile variance estimation failed: {str(e)}") from e
        
        # Run the computation asynchronously
        return await compute_async()
    
    def calibrate(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> RealizedEstimatorConfig:
        """
        Calibrate the estimator configuration based on the provided data.
        
        This method analyzes the input data and determines optimal configuration
        parameters for the quantile variance estimator, such as sampling frequency,
        subsampling factor, and quantile level.
        
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
            # For quantile variance, a moderate subsampling factor is usually sufficient
            optimal_subsample = min(5, len(returns) // 100)
            optimal_subsample = max(1, optimal_subsample)  # Ensure at least 1
            
            # Determine optimal alpha (quantile level)
            # Test different alpha values and choose the one with lowest variance
            alphas = [0.25, 0.5, 0.75]
            variances = []
            
            for alpha in alphas:
                # Compute quantile variance for this alpha
                qv = self._compute_quantile_variance(returns, alpha)
                
                # Compute variance of the estimator (approximate)
                # This is a simplified approach to assess estimator stability
                bootstrap_qvs = []
                n_bootstrap = 10
                sample_size = len(returns) // 2
                
                for _ in range(n_bootstrap):
                    # Sample with replacement
                    indices = np.random.choice(len(returns), sample_size, replace=True)
                    bootstrap_sample = returns[indices]
                    
                    # Compute quantile variance for bootstrap sample
                    bootstrap_qv = self._compute_quantile_variance(bootstrap_sample, alpha)
                    bootstrap_qvs.append(bootstrap_qv)
                
                # Compute variance of bootstrap estimates
                var_estimator = np.var(bootstrap_qvs)
                variances.append(var_estimator)
            
            # Choose alpha with lowest variance
            optimal_alpha = alphas[np.argmin(variances)]
            
            # Update instance alpha
            self._alpha = optimal_alpha
            
            # Create calibrated configuration
            calibrated_config = RealizedEstimatorConfig(
                sampling_frequency=optimal_freq,
                annualize=self._config.annualize,
                annualization_factor=self._config.annualization_factor,
                return_type=self._config.return_type,
                use_subsampling=optimal_subsample > 1,
                subsampling_factor=optimal_subsample,
                apply_noise_correction=False,  # Quantile variance doesn't use noise correction
                time_unit=self._config.time_unit
            )
            
            logger.info(
                f"Calibrated configuration: sampling_frequency={optimal_freq}, "
                f"subsampling_factor={optimal_subsample}, alpha={optimal_alpha}"
            )
            
            return calibrated_config
            
        except Exception as e:
            logger.error(f"Calibration failed: {str(e)}")
            raise ValueError(f"Quantile variance calibration failed: {str(e)}") from e
    
    def plot_quantile_contributions(self, figsize: Tuple[int, int] = (10, 6)) -> Any:
        """
        Plot contributions of different quantiles to the variance estimate.
        
        Args:
            figsize: Figure size as (width, height) in inches
        
        Returns:
            matplotlib.figure.Figure: The created figure
        
        Raises:
            RuntimeError: If quantile contributions have not been computed
            ImportError: If matplotlib is not available
        """
        if self._quantile_contributions is None:
            raise RuntimeError(
                "Quantile contributions have not been computed. "
                "Call fit() with compute_contributions=True first."
            )
        
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # Extract quantile levels and contributions
            alphas = list(self._quantile_contributions.keys())
            contributions = list(self._quantile_contributions.values())
            
            # Plot contributions
            ax.plot(alphas, contributions, 'b-', marker='o')
            
            # Highlight current alpha
            current_alpha_idx = np.argmin(np.abs(np.array(alphas) - self._alpha))
            ax.scatter([alphas[current_alpha_idx]], [contributions[current_alpha_idx]], 
                      color='r', s=100, marker='*', 
                      label=f'Current α={self._alpha:.2f}')
            
            ax.set_title('Quantile Contributions to Variance Estimate')
            ax.set_xlabel('Quantile Level (α)')
            ax.set_ylabel('Variance Contribution')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            return fig
            
        except ImportError:
            logger.warning("Matplotlib is required for plotting")
            raise ImportError("Matplotlib is required for plotting")
    
    def plot_returns_with_quantiles(self, figsize: Tuple[int, int] = (10, 6)) -> Any:
        """
        Plot returns with quantile thresholds.
        
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
            
            # Compute quantile threshold
            abs_returns = np.abs(self._returns)
            sorted_abs_returns = np.sort(abs_returns)
            q_index = int(np.floor(self._alpha * len(self._returns)))
            q_index = max(0, min(q_index, len(self._returns) - 1))
            quantile = sorted_abs_returns[q_index]
            
            # Add quantile threshold lines
            ax.axhline(y=quantile, color='r', linestyle='--', 
                      alpha=0.5, label=f'α={self._alpha:.2f} Quantile (+{quantile:.4f})')
            ax.axhline(y=-quantile, color='r', linestyle='--', 
                      alpha=0.5, label=f'α={self._alpha:.2f} Quantile (-{quantile:.4f})')
            
            # Highlight returns above quantile threshold
            above_threshold = np.abs(self._returns) > quantile
            ax.scatter(np.where(above_threshold)[0], self._returns[above_threshold], 
                      color='r', s=50, marker='o', 
                      label=f'Above {self._alpha:.2f} Quantile')
            
            ax.set_title('Returns with Quantile Thresholds')
            ax.set_xlabel('Observation')
            ax.set_ylabel('Return')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except ImportError:
            logger.warning("Matplotlib is required for plotting")
            raise ImportError("Matplotlib is required for plotting")
    
    def compare_with_standard_variance(self, figsize: Tuple[int, int] = (10, 6)) -> Any:
        """
        Compare quantile variance with standard realized variance.
        
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
            rv = np.sum(self._returns**2)
            
            # Get quantile variance
            qv = self._realized_measure
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # Create bar chart
            labels = ['Realized Variance', f'Quantile Variance (α={self._alpha:.2f})']
            values = [rv, qv]
            
            ax.bar(labels, values, color=['blue', 'orange'])
            
            # Add values on top of bars
            for i, v in enumerate(values):
                ax.text(i, v * 1.01, f'{v:.6f}', ha='center')
            
            # Add ratio
            ratio = qv / rv
            ax.text(0.5, max(values) * 1.1, f'Ratio (QV/RV): {ratio:.4f}', 
                   ha='center', fontsize=12, fontweight='bold')
            
            ax.set_title('Comparison of Variance Estimators')
            ax.set_ylabel('Variance Estimate')
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            return fig
            
        except ImportError:
            logger.warning("Matplotlib is required for plotting")
            raise ImportError("Matplotlib is required for plotting")
    
    def summary(self) -> str:
        """
        Generate a text summary of the quantile variance estimator.
        
        Returns:
            str: A formatted string containing the estimator summary
        
        Raises:
            RuntimeError: If the estimator has not been fitted
        """
        if not self._fitted:
            return f"Quantile Variance Estimator (not fitted)"
        
        if self._results is None:
            return f"Quantile Variance Estimator (fitted, but no results available)"
        
        base_summary = self._results.summary()
        
        # Add quantile-specific information
        additional_info = f"Quantile Level (α): {self._alpha:.4f}\n"
        
        # Add jump ratio if jumps were detected
        jump_ratio = self.get_jump_ratio()
        if jump_ratio is not None:
            additional_info += f"Jump Ratio: {jump_ratio:.4f} ({jump_ratio * 100:.2f}%)\n"
            additional_info += f"Continuous Ratio: {1 - jump_ratio:.4f} ({(1 - jump_ratio) * 100:.2f}%)\n"
        
        # Add subsampling information
        if self._config.use_subsampling:
            additional_info += f"Subsampling Factor: {self._config.subsampling_factor}\n"
        
        # Add quantile contributions information
        if self._quantile_contributions is not None:
            additional_info += f"Quantile Contributions: Computed for {len(self._quantile_contributions)} levels\n"
        
        if additional_info:
            additional_info = "\nQuantile Variance Information:\n" + additional_info
        
        return base_summary + additional_info
    
    def __repr__(self) -> str:
        """
        Generate a detailed string representation of the estimator.
        
        Returns:
            str: A detailed string representation of the estimator
        """
        fitted_str = "fitted" if self._fitted else "not fitted"
        alpha_str = f", alpha={self._alpha}"
        config_str = f", config={self._config}" if self._config else ""
        return f"QuantileVariance({fitted_str}{alpha_str}{config_str})"