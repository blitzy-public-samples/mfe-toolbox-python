"""
Multiscale Realized Variance Estimator

This module implements the multiscale realized variance estimator, which combines
multiple sampling frequencies to mitigate the effects of market microstructure noise.
The estimator is particularly useful for analyzing very high-frequency data where
microstructure effects can significantly bias traditional realized variance estimators.

The implementation follows the approach of Zhang, Mykland, and Aït-Sahalia (2005),
combining information from multiple time scales to produce a noise-robust estimator.
Performance-critical calculations are accelerated using Numba's JIT compilation,
and the implementation supports both synchronous and asynchronous processing for
large datasets.

References:
    Zhang, L., Mykland, P. A., & Aït-Sahalia, Y. (2005). A tale of two time scales:
    Determining integrated volatility with noisy high-frequency data.
    Journal of the American Statistical Association, 100(472), 1394-1411.
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import (
    Any, Dict, List, Literal, Optional, Sequence, Tuple, Union, cast
)

import numpy as np
import pandas as pd
from scipy import stats, optimize
import matplotlib.pyplot as plt

from ...core.parameters import ParameterBase, ParameterError, validate_positive, validate_non_negative
from ...core.exceptions import DimensionError, NumericError
from .base import BaseRealizedEstimator, RealizedEstimatorConfig, RealizedEstimatorResult, NoiseRobustEstimator
from .utils import compute_returns, noise_variance, compute_realized_variance
from ._numba_core import _multiscale_variance_core

# Set up module-level logger
logger = logging.getLogger("mfe.models.realized.multiscale_variance")

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
    logger.debug("Numba available for multiscale variance acceleration")
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    HAS_NUMBA = False
    logger.info("Numba not available. Multiscale variance will use pure NumPy implementation.")


@dataclass
class MultiscaleVarianceConfig(RealizedEstimatorConfig):
    """Configuration parameters for multiscale realized variance estimator.
    
    This class extends RealizedEstimatorConfig with parameters specific to
    the multiscale realized variance estimator.
    
    Attributes:
        min_scale: Minimum scale to use (default: 1)
        max_scale: Maximum scale to use (default: None, determined automatically)
        num_scales: Number of scales to use (default: 10)
        scale_type: Type of scale progression ('linear', 'log', 'sqrt') (default: 'linear')
        weight_type: Type of weights to use ('optimal', 'equal', 'linear') (default: 'optimal')
        bias_correction: Whether to apply bias correction (default: True)
        estimate_noise: Whether to estimate noise variance (default: True)
        noise_variance: Pre-specified noise variance (default: None)
    """
    
    min_scale: int = 1
    max_scale: Optional[int] = None
    num_scales: int = 10
    scale_type: Literal['linear', 'log', 'sqrt'] = 'linear'
    weight_type: Literal['optimal', 'equal', 'linear'] = 'optimal'
    bias_correction: bool = True
    estimate_noise: bool = True
    noise_variance: Optional[float] = None
    
    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        super().__post_init__()
        self.validate_multiscale()
    
    def validate_multiscale(self) -> None:
        """Validate multiscale-specific configuration parameters.
        
        Raises:
            ParameterError: If any parameter constraints are violated
        """
        # Validate min_scale
        if not isinstance(self.min_scale, int):
            raise ParameterError(f"min_scale must be an integer, got {type(self.min_scale)}")
        validate_positive(self.min_scale, "min_scale")
        
        # Validate max_scale if provided
        if self.max_scale is not None:
            if not isinstance(self.max_scale, int):
                raise ParameterError(f"max_scale must be an integer, got {type(self.max_scale)}")
            validate_positive(self.max_scale, "max_scale")
            if self.max_scale < self.min_scale:
                raise ParameterError(f"max_scale ({self.max_scale}) must be greater than or equal to min_scale ({self.min_scale})")
        
        # Validate num_scales
        if not isinstance(self.num_scales, int):
            raise ParameterError(f"num_scales must be an integer, got {type(self.num_scales)}")
        validate_positive(self.num_scales, "num_scales")
        
        # Validate scale_type
        if self.scale_type not in ['linear', 'log', 'sqrt']:
            raise ParameterError(f"scale_type must be 'linear', 'log', or 'sqrt', got {self.scale_type}")
        
        # Validate weight_type
        if self.weight_type not in ['optimal', 'equal', 'linear']:
            raise ParameterError(f"weight_type must be 'optimal', 'equal', or 'linear', got {self.weight_type}")
        
        # Validate noise_variance if provided
        if self.noise_variance is not None:
            validate_non_negative(self.noise_variance, "noise_variance")


@dataclass
class MultiscaleVarianceResult(RealizedEstimatorResult):
    """Result container for multiscale realized variance estimator.
    
    This class extends RealizedEstimatorResult to provide specialized functionality
    for multiscale realized variance estimator results, including scale-specific
    information and visualization methods.
    
    Attributes:
        scales: Scales used for estimation
        weights: Weights used for each scale
        scale_variances: Realized variance at each scale
        noise_variance: Estimated noise variance
        bias_correction: Whether bias correction was applied
        scale_contributions: Contribution of each scale to the final estimate
    """
    
    scales: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None
    scale_variances: Optional[np.ndarray] = None
    bias_correction: Optional[bool] = None
    scale_contributions: Optional[np.ndarray] = None
    
    def __post_init__(self) -> None:
        """Validate result object after initialization."""
        super().__post_init__()
        
        # Ensure arrays are NumPy arrays if provided
        if self.scales is not None and not isinstance(self.scales, np.ndarray):
            self.scales = np.array(self.scales)
        
        if self.weights is not None and not isinstance(self.weights, np.ndarray):
            self.weights = np.array(self.weights)
        
        if self.scale_variances is not None and not isinstance(self.scale_variances, np.ndarray):
            self.scale_variances = np.array(self.scale_variances)
        
        if self.scale_contributions is not None and not isinstance(self.scale_contributions, np.ndarray):
            self.scale_contributions = np.array(self.scale_contributions)
    
    def summary(self) -> str:
        """Generate a text summary of the multiscale realized variance results.
        
        Returns:
            str: A formatted string containing the multiscale realized variance results summary
        """
        base_summary = super().summary()
        
        additional_info = ""
        if self.scales is not None:
            additional_info += f"Number of Scales: {len(self.scales)}\n"
            additional_info += f"Scale Range: {self.scales[0]} to {self.scales[-1]}\n"
        
        if self.bias_correction is not None:
            additional_info += f"Bias Correction Applied: {self.bias_correction}\n"
        
        if additional_info:
            additional_info = "Multiscale Information:\n" + additional_info + "\n"
        
        return base_summary + additional_info
    
    def plot_scale_contributions(self, figsize: Tuple[int, int] = (10, 6), 
                               title: str = "Scale Contributions to Multiscale Variance") -> plt.Figure:
        """Plot the contribution of each scale to the multiscale variance estimate.
        
        Args:
            figsize: Figure size (width, height) in inches
            title: Plot title
            
        Returns:
            matplotlib.figure.Figure: The generated figure
            
        Raises:
            ValueError: If scale information is not available
        """
        if self.scales is None or self.scale_contributions is None:
            raise ValueError("Scale information is not available")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot scale contributions
        ax.bar(self.scales, self.scale_contributions, alpha=0.7)
        
        # Add labels and title
        ax.set_xlabel("Scale")
        ax.set_ylabel("Contribution")
        ax.set_title(title)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add text with total realized measure
        ax.text(0.95, 0.95, f"Total: {self.realized_measure:.6f}",
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def plot_scale_variances(self, figsize: Tuple[int, int] = (10, 6),
                           title: str = "Realized Variance at Different Scales") -> plt.Figure:
        """Plot the realized variance at each scale.
        
        Args:
            figsize: Figure size (width, height) in inches
            title: Plot title
            
        Returns:
            matplotlib.figure.Figure: The generated figure
            
        Raises:
            ValueError: If scale information is not available
        """
        if self.scales is None or self.scale_variances is None:
            raise ValueError("Scale information is not available")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot scale variances
        ax.plot(self.scales, self.scale_variances, 'o-', linewidth=2, markersize=8)
        
        # Add labels and title
        ax.set_xlabel("Scale")
        ax.set_ylabel("Realized Variance")
        ax.set_title(title)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add text with noise variance if available
        if self.noise_variance is not None:
            ax.text(0.95, 0.95, f"Noise Variance: {self.noise_variance:.6e}",
                    transform=ax.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert multiscale variance results to a pandas DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame containing multiscale variance results
        """
        df = super().to_dataframe()
        
        # Add scale information if available
        if self.scales is not None and self.scale_variances is not None:
            scale_df = pd.DataFrame({
                'scale': self.scales,
                'variance': self.scale_variances
            })
            
            if self.weights is not None:
                scale_df['weight'] = self.weights
            
            if self.scale_contributions is not None:
                scale_df['contribution'] = self.scale_contributions
            
            # Return both dataframes as a dictionary
            return {'main': df, 'scales': scale_df}
        
        return df


@jit(nopython=True, cache=True)
def _compute_multiscale_weights(scales: np.ndarray, weight_type: str, noise_var: float) -> np.ndarray:
    """
    Numba-accelerated implementation of multiscale weight computation.
    
    Args:
        scales: Array of scales
        weight_type: Type of weights to use ('optimal', 'equal', 'linear')
        noise_var: Estimated noise variance
        
    Returns:
        Array of weights for each scale
    """
    n_scales = len(scales)
    weights = np.zeros(n_scales)
    
    if weight_type == 'optimal':
        # Compute optimal weights based on Zhang et al. (2005)
        # Weights are proportional to (scale - noise_var/scale)
        for i in range(n_scales):
            scale = scales[i]
            weights[i] = scale - noise_var / scale
        
        # Normalize weights to sum to 1
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            # Fallback to equal weights if optimal weights are invalid
            weights = np.ones(n_scales) / n_scales
    
    elif weight_type == 'equal':
        # Equal weights
        weights = np.ones(n_scales) / n_scales
    
    elif weight_type == 'linear':
        # Linear weights (higher weight for larger scales)
        for i in range(n_scales):
            weights[i] = scales[i]
        
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)
    
    else:
        # Default to equal weights
        weights = np.ones(n_scales) / n_scales
    
    return weights


class MultiscaleVariance(NoiseRobustEstimator):
    """Multiscale realized variance estimator.
    
    This class implements the multiscale realized variance estimator, which combines
    multiple sampling frequencies to mitigate the effects of market microstructure noise.
    The estimator is particularly useful for analyzing very high-frequency data where
    microstructure effects can significantly bias traditional realized variance estimators.
    
    The implementation follows the approach of Zhang, Mykland, and Aït-Sahalia (2005),
    combining information from multiple time scales to produce a noise-robust estimator.
    Performance-critical calculations are accelerated using Numba's JIT compilation,
    and the implementation supports both synchronous and asynchronous processing for
    large datasets.
    
    Attributes:
        config: Configuration parameters for the estimator
        scales: Scales used for estimation
        weights: Weights used for each scale
        scale_variances: Realized variance at each scale
        scale_contributions: Contribution of each scale to the final estimate
    """
    
    def __init__(self, config: Optional[MultiscaleVarianceConfig] = None, name: str = "MultiscaleVariance"):
        """Initialize the multiscale realized variance estimator.
        
        Args:
            config: Configuration parameters for the estimator
            name: A descriptive name for the estimator
        """
        super().__init__(config=config if config is not None else MultiscaleVarianceConfig(), name=name)
        self._scales: Optional[np.ndarray] = None
        self._weights: Optional[np.ndarray] = None
        self._scale_variances: Optional[np.ndarray] = None
        self._scale_contributions: Optional[np.ndarray] = None
    
    @property
    def config(self) -> MultiscaleVarianceConfig:
        """Get the estimator configuration.
        
        Returns:
            MultiscaleVarianceConfig: The estimator configuration
        """
        return cast(MultiscaleVarianceConfig, self._config)
    
    @config.setter
    def config(self, config: MultiscaleVarianceConfig) -> None:
        """Set the estimator configuration.
        
        Args:
            config: New configuration parameters
        
        Raises:
            TypeError: If config is not a MultiscaleVarianceConfig
        """
        if not isinstance(config, MultiscaleVarianceConfig):
            raise TypeError(f"config must be a MultiscaleVarianceConfig, got {type(config)}")
        
        self._config = config
        self._fitted = False  # Reset fitted state when configuration changes
    
    @property
    def scales(self) -> Optional[np.ndarray]:
        """Get the scales used for estimation.
        
        Returns:
            Optional[np.ndarray]: The scales if the estimator has been fitted,
                                 None otherwise
        """
        return self._scales
    
    @property
    def weights(self) -> Optional[np.ndarray]:
        """Get the weights used for each scale.
        
        Returns:
            Optional[np.ndarray]: The weights if the estimator has been fitted,
                                 None otherwise
        """
        return self._weights
    
    @property
    def scale_variances(self) -> Optional[np.ndarray]:
        """Get the realized variance at each scale.
        
        Returns:
            Optional[np.ndarray]: The scale variances if the estimator has been fitted,
                                 None otherwise
        """
        return self._scale_variances
    
    @property
    def scale_contributions(self) -> Optional[np.ndarray]:
        """Get the contribution of each scale to the final estimate.
        
        Returns:
            Optional[np.ndarray]: The scale contributions if the estimator has been fitted,
                                 None otherwise
        """
        return self._scale_contributions
    
    def _generate_scales(self, n: int) -> np.ndarray:
        """Generate scales for multiscale estimation.
        
        Args:
            n: Length of the return series
            
        Returns:
            np.ndarray: Array of scales
        """
        min_scale = self.config.min_scale
        max_scale = self.config.max_scale if self.config.max_scale is not None else max(min(n // 4, 100), min_scale + 1)
        num_scales = self.config.num_scales
        scale_type = self.config.scale_type
        
        # Ensure max_scale is valid
        max_scale = min(max_scale, n - 1)
        
        # Ensure num_scales is valid
        num_scales = min(num_scales, max_scale - min_scale + 1)
        
        # Generate scales based on scale_type
        if scale_type == 'linear':
            # Linear spacing
            scales = np.linspace(min_scale, max_scale, num_scales, dtype=int)
        elif scale_type == 'log':
            # Logarithmic spacing
            scales = np.logspace(np.log10(min_scale), np.log10(max_scale), num_scales, dtype=int)
        elif scale_type == 'sqrt':
            # Square root spacing
            scales = np.linspace(np.sqrt(min_scale), np.sqrt(max_scale), num_scales)**2
            scales = np.unique(scales.astype(int))  # Remove duplicates due to rounding
        else:
            # Default to linear spacing
            scales = np.linspace(min_scale, max_scale, num_scales, dtype=int)
        
        # Ensure scales are unique and sorted
        scales = np.unique(scales)
        
        return scales
    
    def _compute_scale_variances(self, returns: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """Compute realized variance at each scale.
        
        Args:
            returns: Array of returns
            scales: Array of scales
            
        Returns:
            np.ndarray: Array of realized variances at each scale
        """
        n = len(returns)
        n_scales = len(scales)
        scale_variances = np.zeros(n_scales)
        
        for i, scale in enumerate(scales):
            # Skip if scale is too large
            if scale >= n:
                continue
            
            # Compute returns at this scale
            scale_returns = np.zeros(n - scale)
            for j in range(n - scale):
                for k in range(scale):
                    scale_returns[j] += returns[j + k]
            
            # Compute realized variance at this scale
            scale_variances[i] = np.sum(scale_returns**2) / (n - scale)
        
        return scale_variances
    
    def _compute_weights(self, scales: np.ndarray, noise_var: float) -> np.ndarray:
        """Compute weights for each scale.
        
        Args:
            scales: Array of scales
            noise_var: Estimated noise variance
            
        Returns:
            np.ndarray: Array of weights for each scale
        """
        weight_type = self.config.weight_type
        
        # Use Numba-accelerated implementation if available
        if HAS_NUMBA:
            return _compute_multiscale_weights(scales, weight_type, noise_var)
        
        # Pure NumPy implementation
        n_scales = len(scales)
        weights = np.zeros(n_scales)
        
        if weight_type == 'optimal':
            # Compute optimal weights based on Zhang et al. (2005)
            # Weights are proportional to (scale - noise_var/scale)
            for i in range(n_scales):
                scale = scales[i]
                weights[i] = scale - noise_var / scale
            
            # Normalize weights to sum to 1
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                # Fallback to equal weights if optimal weights are invalid
                weights = np.ones(n_scales) / n_scales
        
        elif weight_type == 'equal':
            # Equal weights
            weights = np.ones(n_scales) / n_scales
        
        elif weight_type == 'linear':
            # Linear weights (higher weight for larger scales)
            weights = scales.astype(float)
            
            # Normalize weights to sum to 1
            weights = weights / np.sum(weights)
        
        else:
            # Default to equal weights
            weights = np.ones(n_scales) / n_scales
        
        return weights
    
    def _compute_realized_measure(self, 
                                 prices: np.ndarray, 
                                 times: np.ndarray, 
                                 returns: np.ndarray,
                                 **kwargs: Any) -> np.ndarray:
        """Compute the multiscale realized variance from the preprocessed data.
        
        Args:
            prices: Preprocessed price data
            times: Preprocessed time points
            returns: Returns computed from prices
            **kwargs: Additional keyword arguments for computation
            
        Returns:
            np.ndarray: Multiscale realized variance
            
        Raises:
            ValueError: If computation fails
        """
        n = len(returns)
        
        # Generate scales
        scales = self._generate_scales(n)
        self._scales = scales
        
        # Estimate noise variance if needed
        if self.config.estimate_noise:
            noise_var = self.estimate_noise_variance(returns)
        else:
            noise_var = self.config.noise_variance if self.config.noise_variance is not None else 0.0
        
        self._noise_variance = noise_var
        
        # Compute realized variance at each scale
        if HAS_NUMBA:
            # Use Numba-accelerated implementation for scale variances
            scale_variances = np.zeros(len(scales))
            for i, scale in enumerate(scales):
                # Skip if scale is too large
                if scale >= n:
                    continue
                
                # Compute returns at this scale
                scale_returns = np.zeros(n - scale)
                for j in range(n - scale):
                    for k in range(scale):
                        scale_returns[j] += returns[j + k]
                
                # Compute realized variance at this scale
                scale_variances[i] = np.sum(scale_returns**2) / (n - scale)
        else:
            # Use pure NumPy implementation
            scale_variances = self._compute_scale_variances(returns, scales)
        
        self._scale_variances = scale_variances
        
        # Compute weights for each scale
        weights = self._compute_weights(scales, noise_var)
        self._weights = weights
        
        # Compute weighted sum of realized variances
        msv = np.sum(weights * scale_variances)
        
        # Apply bias correction if enabled
        if self.config.bias_correction and noise_var > 0:
            # Bias correction based on Zhang et al. (2005)
            # Correction term is 2 * n * noise_var / avg_scale
            avg_scale = np.sum(weights * scales)
            correction = 2 * n * noise_var / avg_scale
            msv -= correction
        
        # Ensure non-negativity
        msv = max(0, msv)
        
        # Compute scale contributions
        self._scale_contributions = weights * scale_variances / msv if msv > 0 else weights
        
        return np.array([msv])
    
    async def _compute_realized_measure_async(self, 
                                            prices: np.ndarray, 
                                            times: np.ndarray, 
                                            returns: np.ndarray,
                                            progress_callback: Optional[Callable[[float, str], None]] = None,
                                            **kwargs: Any) -> np.ndarray:
        """Asynchronously compute the multiscale realized variance from the preprocessed data.
        
        Args:
            prices: Preprocessed price data
            times: Preprocessed time points
            returns: Returns computed from prices
            progress_callback: Optional callback function for reporting progress
            **kwargs: Additional keyword arguments for computation
            
        Returns:
            np.ndarray: Multiscale realized variance
            
        Raises:
            ValueError: If computation fails
        """
        import asyncio
        
        n = len(returns)
        
        # Report progress
        if progress_callback:
            await progress_callback(0.1, "Generating scales...")
        
        # Generate scales
        scales = self._generate_scales(n)
        self._scales = scales
        
        # Report progress
        if progress_callback:
            await progress_callback(0.2, "Estimating noise variance...")
        
        # Estimate noise variance if needed
        if self.config.estimate_noise:
            noise_var = self.estimate_noise_variance(returns)
        else:
            noise_var = self.config.noise_variance if self.config.noise_variance is not None else 0.0
        
        self._noise_variance = noise_var
        
        # Report progress
        if progress_callback:
            await progress_callback(0.3, "Computing scale variances...")
        
        # Compute realized variance at each scale
        scale_variances = np.zeros(len(scales))
        
        # Process scales in chunks to allow for progress updates
        chunk_size = max(1, len(scales) // 10)
        for i in range(0, len(scales), chunk_size):
            chunk_end = min(i + chunk_size, len(scales))
            chunk_scales = scales[i:chunk_end]
            
            # Process each scale in the chunk
            for j, scale in enumerate(chunk_scales):
                scale_idx = i + j
                
                # Skip if scale is too large
                if scale >= n:
                    continue
                
                # Compute returns at this scale
                scale_returns = np.zeros(n - scale)
                for k in range(n - scale):
                    for l in range(scale):
                        scale_returns[k] += returns[k + l]
                
                # Compute realized variance at this scale
                scale_variances[scale_idx] = np.sum(scale_returns**2) / (n - scale)
            
            # Report progress
            if progress_callback:
                progress = 0.3 + 0.4 * (chunk_end / len(scales))
                await progress_callback(progress, f"Computing scale variances ({chunk_end}/{len(scales)})...")
            
            # Yield control to allow other tasks to run
            await asyncio.sleep(0)
        
        self._scale_variances = scale_variances
        
        # Report progress
        if progress_callback:
            await progress_callback(0.7, "Computing weights...")
        
        # Compute weights for each scale
        weights = self._compute_weights(scales, noise_var)
        self._weights = weights
        
        # Report progress
        if progress_callback:
            await progress_callback(0.8, "Computing final estimate...")
        
        # Compute weighted sum of realized variances
        msv = np.sum(weights * scale_variances)
        
        # Apply bias correction if enabled
        if self.config.bias_correction and noise_var > 0:
            # Bias correction based on Zhang et al. (2005)
            # Correction term is 2 * n * noise_var / avg_scale
            avg_scale = np.sum(weights * scales)
            correction = 2 * n * noise_var / avg_scale
            msv -= correction
        
        # Ensure non-negativity
        msv = max(0, msv)
        
        # Compute scale contributions
        self._scale_contributions = weights * scale_variances / msv if msv > 0 else weights
        
        # Report progress
        if progress_callback:
            await progress_callback(1.0, "Multiscale variance estimation complete")
        
        return np.array([msv])
    
    async def fit_async(self, data: Tuple[np.ndarray, np.ndarray], 
                       progress_callback: Optional[Callable[[float, str], None]] = None,
                       **kwargs: Any) -> MultiscaleVarianceResult:
        """Asynchronously fit the multiscale realized variance estimator to the provided data.
        
        This method provides an asynchronous interface to the fit method,
        allowing for non-blocking estimation in UI contexts with progress reporting.
        
        Args:
            data: The data to fit the estimator to, as a tuple of (prices, times)
            progress_callback: Optional callback function for reporting progress
            **kwargs: Additional keyword arguments for estimation
            
        Returns:
            MultiscaleVarianceResult: The estimation results
            
        Raises:
            ValueError: If the data is invalid
            RuntimeError: If the estimation fails
        """
        import time
        import asyncio
        
        start_time = time.time()
        
        # Report initial progress
        if progress_callback:
            await progress_callback(0.0, "Starting multiscale variance estimation...")
        
        # Validate input data
        self.validate_data(data)
        prices, times = data
        
        # Report progress
        if progress_callback:
            await progress_callback(0.05, "Preprocessing data...")
        
        # Preprocess data
        processed_prices, processed_times, returns = self.preprocess_data(prices, times)
        
        try:
            # Compute realized measure asynchronously
            realized_measure = await self._compute_realized_measure_async(
                processed_prices, processed_times, returns, 
                progress_callback=progress_callback, **kwargs
            )
            
            # Update instance state
            self._realized_measure = realized_measure
            self._fitted = True
            
            # Create result object
            result = MultiscaleVarianceResult(
                model_name=self._name,
                realized_measure=realized_measure[0],
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
                config=self._config.to_dict(),
                scales=self._scales,
                weights=self._weights,
                scale_variances=self._scale_variances,
                bias_correction=self.config.bias_correction,
                scale_contributions=self._scale_contributions
            )
            
            # Store result
            self._results = result
            
            return result
        
        except Exception as e:
            logger.error(f"Multiscale variance estimation failed: {str(e)}")
            raise RuntimeError(f"Multiscale variance estimation failed: {str(e)}") from e
    
    def fit(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> MultiscaleVarianceResult:
        """Fit the multiscale realized variance estimator to the provided data.
        
        Args:
            data: The data to fit the estimator to, as a tuple of (prices, times)
            **kwargs: Additional keyword arguments for estimation
            
        Returns:
            MultiscaleVarianceResult: The estimation results
            
        Raises:
            ValueError: If the data is invalid
            RuntimeError: If the estimation fails
        """
        import time
        
        start_time = time.time()
        
        # Validate input data
        self.validate_data(data)
        prices, times = data
        
        # Preprocess data
        processed_prices, processed_times, returns = self.preprocess_data(prices, times)
        
        try:
            # Compute realized measure
            realized_measure = self._compute_realized_measure(
                processed_prices, processed_times, returns, **kwargs
            )
            
            # Update instance state
            self._realized_measure = realized_measure
            self._fitted = True
            
            # Create result object
            result = MultiscaleVarianceResult(
                model_name=self._name,
                realized_measure=realized_measure[0],
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
                config=self._config.to_dict(),
                scales=self._scales,
                weights=self._weights,
                scale_variances=self._scale_variances,
                bias_correction=self.config.bias_correction,
                scale_contributions=self._scale_contributions
            )
            
            # Store result
            self._results = result
            
            return result
        
        except Exception as e:
            logger.error(f"Multiscale variance estimation failed: {str(e)}")
            raise RuntimeError(f"Multiscale variance estimation failed: {str(e)}") from e
    
    def calibrate(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> MultiscaleVarianceConfig:
        """Calibrate the estimator configuration based on the provided data.
        
        This method analyzes the input data and determines optimal configuration
        parameters for the multiscale realized variance estimator.
        
        Args:
            data: The data to calibrate the estimator with, as a tuple of (prices, times)
            **kwargs: Additional keyword arguments for calibration
            
        Returns:
            MultiscaleVarianceConfig: Calibrated configuration
        """
        # Validate input data
        self.validate_data(data)
        prices, times = data
        
        # Preprocess data
        processed_prices, processed_times, returns = self.preprocess_data(prices, times)
        
        # Estimate noise variance
        noise_var = self.estimate_noise_variance(returns)
        
        # Determine optimal number of scales based on data length
        n = len(returns)
        num_scales = min(20, max(5, int(np.sqrt(n) / 2)))
        
        # Determine optimal max_scale based on data length
        max_scale = min(n // 4, 100)
        
        # Create calibrated configuration
        calibrated_config = MultiscaleVarianceConfig(
            min_scale=1,
            max_scale=max_scale,
            num_scales=num_scales,
            scale_type='linear',
            weight_type='optimal',
            bias_correction=True,
            estimate_noise=False,
            noise_variance=noise_var,
            sampling_frequency=self._config.sampling_frequency,
            annualize=self._config.annualize,
            annualization_factor=self._config.annualization_factor,
            return_type=self._config.return_type
        )
        
        return calibrated_config
    
    def plot_results(self, figsize: Tuple[int, int] = (12, 10)) -> Dict[str, plt.Figure]:
        """Plot comprehensive results of the multiscale variance estimation.
        
        Args:
            figsize: Base figure size (width, height) in inches
            
        Returns:
            Dict[str, plt.Figure]: Dictionary of generated figures
            
        Raises:
            RuntimeError: If the estimator has not been fitted
        """
        if not self._fitted:
            raise RuntimeError("Estimator has not been fitted. Call fit() first.")
        
        if not isinstance(self._results, MultiscaleVarianceResult):
            raise RuntimeError("Results are not available or not of the correct type.")
        
        # Create dictionary to store figures
        figures = {}
        
        # Plot scale contributions
        figures['scale_contributions'] = self._results.plot_scale_contributions(
            figsize=figsize, 
            title=f"Scale Contributions to Multiscale Variance - {self._name}"
        )
        
        # Plot scale variances
        figures['scale_variances'] = self._results.plot_scale_variances(
            figsize=figsize, 
            title=f"Realized Variance at Different Scales - {self._name}"
        )
        
        return figures
    
    def __str__(self) -> str:
        """Generate a string representation of the estimator.
        
        Returns:
            str: A string representation of the estimator
        """
        if not self._fitted:
            return f"MultiscaleVariance(name='{self._name}', fitted=False)""
        
        return (f"MultiscaleVariance(name='{self._name}', fitted=True, "
                f"realized_measure={self._realized_measure[0]:.6f}, "
                f"num_scales={len(self._scales) if self._scales is not None else 'N/A'})")
