# mfe/models/realized/bipower_variation.py
"""
Realized bipower variation estimator for robust volatility measurement.

This module implements the realized bipower variation estimator, which provides
a robust measure of volatility that is less sensitive to price jumps than standard
realized variance. The bipower variation is computed as the sum of products of
adjacent absolute returns, with appropriate scaling to ensure consistency.

The estimator inherits from JumpRobustEstimator, providing specialized functionality
for jump detection and separation of continuous and jump components of volatility.
Performance-critical calculations are accelerated using Numba's JIT compilation
for efficient processing of high-frequency data.

The implementation supports various configuration options including subsampling
for noise reduction, different return types, and asynchronous processing for
long computations.
"""

import logging
import warnings
from typing import Any, Dict, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd

from .base import JumpRobustEstimator, RealizedEstimatorConfig, RealizedEstimatorResult
from ._numba_core import _bipower_variation_core, _compute_subsampled_measure_numba
from ...core.exceptions import ParameterError, NumericError

# Set up module-level logger
logger = logging.getLogger("mfe.models.realized.bipower_variation")


class BipowerVariation(JumpRobustEstimator):
    """
    Realized bipower variation estimator for robust volatility measurement.
    
    This class implements the realized bipower variation estimator, which provides
    a robust measure of volatility that is less sensitive to price jumps than
    standard realized variance. The bipower variation is computed as the sum of
    products of adjacent absolute returns, with appropriate scaling to ensure
    consistency with integrated variance in the absence of jumps.
    
    The estimator inherits from JumpRobustEstimator, providing specialized
    functionality for jump detection and separation of continuous and jump
    components of volatility.
    
    Attributes:
        config: Configuration parameters for the estimator
        jump_threshold: Threshold used for jump detection (if jumps were detected)
        jump_indicators: Boolean array indicating detected jumps (if jumps were detected)
    """
    
    def __init__(self, config: Optional[RealizedEstimatorConfig] = None):
        """
        Initialize the bipower variation estimator.
        
        Args:
            config: Configuration parameters for the estimator
        """
        super().__init__(config=config, name="Bipower Variation")
        
        # Additional attributes specific to bipower variation
        self._correction_factor: float = np.pi / 2  # Asymptotic correction factor
        self._debiased: bool = True  # Whether to apply finite sample bias correction
    
    def _compute_realized_measure(self, 
                                 prices: np.ndarray, 
                                 times: np.ndarray, 
                                 returns: np.ndarray,
                                 debiased: bool = True,
                                 detect_jumps: bool = False,
                                 threshold_multiplier: float = 3.0,
                                 **kwargs: Any) -> np.ndarray:
        """
        Compute the bipower variation from the preprocessed data.
        
        This method implements the core bipower variation calculation, with
        options for debiasing and subsampling. It can also detect jumps
        in the return series if requested.
        
        Args:
            prices: Preprocessed price data
            times: Preprocessed time points
            returns: Returns computed from prices
            debiased: Whether to apply finite sample bias correction
            detect_jumps: Whether to detect jumps in the return series
            threshold_multiplier: Multiplier for jump detection threshold
            **kwargs: Additional keyword arguments for computation
        
        Returns:
            np.ndarray: Bipower variation
        
        Raises:
            ValueError: If computation fails
            NumericError: If numerical issues are encountered
        """
        # Store debiased flag
        self._debiased = debiased
        
        try:
            # Check if subsampling is enabled
            if self._config.use_subsampling and self._config.subsampling_factor > 1:
                # Compute subsampled bipower variation
                bv = self._compute_subsampled_bipower_variation(
                    returns, self._config.subsampling_factor
                )
            else:
                # Compute standard bipower variation
                bv = self._compute_bipower_variation(returns, debiased)
            
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
            
            return bv
            
        except Exception as e:
            logger.error(f"Bipower variation computation failed: {str(e)}")
            raise ValueError(f"Bipower variation computation failed: {str(e)}") from e
    
    def _compute_bipower_variation(self, returns: np.ndarray, debiased: bool = True) -> float:
        """
        Compute the standard bipower variation.
        
        Args:
            returns: Return series
            debiased: Whether to apply finite sample bias correction
        
        Returns:
            float: Bipower variation
        
        Raises:
            NumericError: If numerical issues are encountered
        """
        try:
            # Use Numba-accelerated implementation
            bv = _bipower_variation_core(returns)
            
            # Apply finite sample bias correction if requested
            if not debiased:
                # Remove the finite sample correction applied in the core function
                n = len(returns)
                bv = bv * ((n - 1) / n)
            
            return bv
            
        except Exception as e:
            logger.error(f"Bipower variation core computation failed: {str(e)}")
            
            # Fallback to pure NumPy implementation if Numba fails
            logger.warning("Falling back to pure NumPy implementation")
            
            n = len(returns)
            abs_returns = np.abs(returns)
            bipower_sum = np.sum(abs_returns[:-1] * abs_returns[1:])
            
            # Apply scaling factor (π/2)^2 for asymptotic consistency
            scaling = (np.pi / 2) ** 2
            
            # Apply finite sample correction if requested
            correction = n / (n - 1) if debiased else 1.0
            
            return correction * bipower_sum * scaling
    
    def _compute_subsampled_bipower_variation(self, returns: np.ndarray, subsample_factor: int) -> float:
        """
        Compute subsampled bipower variation for noise reduction.
        
        Args:
            returns: Return series
            subsample_factor: Number of subsamples to use
        
        Returns:
            float: Subsampled bipower variation
        
        Raises:
            ValueError: If subsample_factor is invalid
            NumericError: If numerical issues are encountered
        """
        # Validate subsample_factor
        if not isinstance(subsample_factor, int) or subsample_factor < 1:
            raise ParameterError(f"subsample_factor must be a positive integer, got {subsample_factor}")
        
        if subsample_factor == 1:
            # No subsampling needed
            return self._compute_bipower_variation(returns, self._debiased)
        
        try:
            n = len(returns)
            subsampled_bv = 0.0
            
            for i in range(subsample_factor):
                # Extract i-th subsample
                subsample = returns[i::subsample_factor]
                
                # Skip if subsample is too short
                if len(subsample) < 2:
                    continue
                
                # Compute bipower variation for this subsample
                subsample_bv = self._compute_bipower_variation(subsample, self._debiased)
                
                # Scale by the number of observations
                scaled_bv = subsample_bv * (n / len(subsample))
                
                # Add to the total
                subsampled_bv += scaled_bv
            
            # Average across subsamples
            return subsampled_bv / subsample_factor
            
        except Exception as e:
            logger.error(f"Subsampled bipower variation computation failed: {str(e)}")
            raise ValueError(f"Subsampled bipower variation computation failed: {str(e)}") from e
    
    async def fit_async(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> RealizedEstimatorResult:
        """
        Asynchronously fit the bipower variation estimator to the provided data.
        
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
                raise RuntimeError(f"Bipower variation estimation failed: {str(e)}") from e
        
        # Run the computation asynchronously
        return await compute_async()
    
    def calibrate(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> RealizedEstimatorConfig:
        """
        Calibrate the estimator configuration based on the provided data.
        
        This method analyzes the input data and determines optimal configuration
        parameters for the bipower variation estimator, such as sampling frequency
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
            # For bipower variation, a moderate subsampling factor is usually sufficient
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
                apply_noise_correction=False,  # Bipower variation doesn't use noise correction
                time_unit=self._config.time_unit
            )
            
            logger.info(
                f"Calibrated configuration: sampling_frequency={optimal_freq}, "
                f"subsampling_factor={optimal_subsample}"
            )
            
            return calibrated_config
            
        except Exception as e:
            logger.error(f"Calibration failed: {str(e)}")
            raise ValueError(f"Bipower variation calibration failed: {str(e)}") from e
    
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
            
            ax.set_title('Returns with Detected Jumps')
            ax.set_xlabel('Observation')
            ax.set_ylabel('Return')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except ImportError:
            logger.warning("Matplotlib is required for plotting")
            raise ImportError("Matplotlib is required for plotting")
    
    def summary(self) -> str:
        """
        Generate a text summary of the bipower variation estimator.
        
        Returns:
            str: A formatted string containing the estimator summary
        
        Raises:
            RuntimeError: If the estimator has not been fitted
        """
        if not self._fitted:
            return f"Bipower Variation Estimator (not fitted)"
        
        if self._results is None:
            return f"Bipower Variation Estimator (fitted, but no results available)"
        
        base_summary = self._results.summary()
        
        # Add bipower-specific information
        additional_info = ""
        
        # Add jump ratio if jumps were detected
        jump_ratio = self.get_jump_ratio()
        if jump_ratio is not None:
            additional_info += f"Jump Ratio: {jump_ratio:.4f} ({jump_ratio * 100:.2f}%)\n"
            additional_info += f"Continuous Ratio: {1 - jump_ratio:.4f} ({(1 - jump_ratio) * 100:.2f}%)\n"
        
        # Add debiasing information
        additional_info += f"Debiased: {self._debiased}\n"
        
        if additional_info:
            additional_info = "\nBipower Variation Information:\n" + additional_info
        
        return base_summary + additional_info
    
    def __repr__(self) -> str:
        """
        Generate a detailed string representation of the estimator.
        
        Returns:
            str: A detailed string representation of the estimator
        """
        fitted_str = "fitted" if self._fitted else "not fitted"
        config_str = f", config={self._config}" if self._config else ""
        return f"BipowerVariation({fitted_str}{config_str})"