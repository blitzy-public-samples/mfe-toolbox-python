# mfe/models/realized/twoscale_variance.py
"""
Two-scale realized variance estimator for high-frequency financial data.

This module implements the two-scale realized variance estimator, which combines
two different sampling frequencies to correct for market microstructure noise.
The estimator is based on the work of Zhang, Mykland, and Aït-Sahalia (2005) and
provides a robust approach to volatility estimation in the presence of noise.

The two-scale estimator uses a combination of a slow time scale (sparse sampling)
and a fast time scale (frequent sampling) to separate the true volatility from
the noise component. This approach provides a consistent estimator of integrated
variance even when high-frequency data is contaminated with microstructure noise.

The implementation leverages NumPy's efficient array operations with Numba
acceleration for performance-critical calculations. It supports both raw NumPy
arrays and Pandas DataFrames with datetime indices for convenient time series analysis.

Classes:
    TwoScaleVarianceConfig: Configuration parameters for two-scale variance estimation
    TwoScaleVarianceResult: Result container for two-scale variance estimation
    TwoScaleVarianceEstimator: Class for estimating two-scale realized variance

References:
    Zhang, L., Mykland, P. A., & Aït-Sahalia, Y. (2005). A tale of two time scales:
    Determining integrated volatility with noisy high-frequency data. Journal of the
    American Statistical Association, 100(472), 1394-1411.
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Literal, Optional, Sequence, 
    Tuple, Type, TypeVar, Union, cast, overload
)
import numpy as np
import pandas as pd
from scipy import stats, optimize

from ...core.base import ModelBase
from ...core.parameters import ParameterBase, ParameterError, validate_positive, validate_non_negative
from ...core.exceptions import (
    DimensionError, NumericError, ParameterError, RealizedVolatilityError,
    raise_dimension_error, raise_numeric_error, warn_numeric
)
from .base import BaseRealizedEstimator, RealizedEstimatorConfig, RealizedEstimatorResult, NoiseRobustEstimator
from .utils import compute_realized_variance, compute_subsampled_measure
from .noise_estimate import noise_variance

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    HAS_NUMBA = False

# Set up module-level logger
logger = logging.getLogger("mfe.models.realized.twoscale_variance")

# Try to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.info("Matplotlib not available. Plotting functions will be disabled.")


@dataclass
class TwoScaleVarianceConfig(RealizedEstimatorConfig):
    """Configuration parameters for two-scale realized variance estimation.
    
    This class extends RealizedEstimatorConfig to provide specialized configuration
    parameters for two-scale realized variance estimation.
    
    Attributes:
        slow_scale: Sampling frequency for the slow time scale (sparse sampling)
        fast_scale: Sampling frequency for the fast time scale (frequent sampling)
        auto_scale: Whether to automatically determine optimal scales
        max_scales: Maximum number of scales to consider for auto-scale
        bias_correction: Whether to apply bias correction
        estimate_noise: Whether to estimate noise variance
        noise_method: Method for estimating noise variance
        plot: Whether to generate diagnostic plots
    """
    
    slow_scale: Optional[int] = None
    fast_scale: Optional[int] = 1
    auto_scale: bool = True
    max_scales: int = 30
    bias_correction: bool = True
    estimate_noise: bool = True
    noise_method: Literal['ac1', 'bandi-russell', 'signature', 'ml', 'auto'] = 'auto'
    plot: bool = False
    
    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        super().__post_init__()
        
        # Validate slow_scale and fast_scale if provided
        if self.slow_scale is not None:
            validate_positive(self.slow_scale, "slow_scale")
        
        if self.fast_scale is not None:
            validate_positive(self.fast_scale, "fast_scale")
        
        # Validate max_scales
        validate_positive(self.max_scales, "max_scales")
        
        # Validate noise_method
        valid_methods = ['ac1', 'bandi-russell', 'signature', 'ml', 'auto']
        if self.noise_method not in valid_methods:
            raise ParameterError(f"noise_method must be one of {valid_methods}, got {self.noise_method}")
        
        # Check that slow_scale > fast_scale if both are provided
        if self.slow_scale is not None and self.fast_scale is not None:
            if self.slow_scale <= self.fast_scale:
                raise ParameterError(
                    f"slow_scale ({self.slow_scale}) must be greater than fast_scale ({self.fast_scale})",
                    param_name="slow_scale",
                    param_value=self.slow_scale,
                    constraint=f"slow_scale > fast_scale ({self.fast_scale})"
                )


@dataclass
class TwoScaleVarianceResult(RealizedEstimatorResult):
    """Result container for two-scale realized variance estimation.
    
    This class extends RealizedEstimatorResult to provide specialized functionality
    for two-scale realized variance estimation results, including additional metadata
    and diagnostic information specific to two-scale estimation.
    
    Attributes:
        slow_scale: Sampling frequency for the slow time scale (sparse sampling)
        fast_scale: Sampling frequency for the fast time scale (frequent sampling)
        slow_rv: Realized variance at the slow time scale
        fast_rv: Realized variance at the fast time scale
        noise_variance: Estimated noise variance
        bias_correction: Whether bias correction was applied
        scales_tested: List of scales tested for auto-scale selection
        rv_by_scale: Realized variance at each tested scale
    """
    
    slow_scale: Optional[int] = None
    fast_scale: Optional[int] = None
    slow_rv: Optional[float] = None
    fast_rv: Optional[float] = None
    bias_correction: Optional[bool] = None
    scales_tested: Optional[List[int]] = None
    rv_by_scale: Optional[Dict[int, float]] = None
    
    def __post_init__(self) -> None:
        """Validate result object after initialization."""
        super().__post_init__()
    
    def summary(self) -> str:
        """Generate a text summary of the two-scale realized variance results.
        
        Returns:
            str: A formatted string containing the two-scale realized variance results summary
        """
        base_summary = super().summary()
        
        additional_info = f"Two-Scale Realized Variance Results:\n"
        additional_info += f"  Slow Scale: {self.slow_scale}\n"
        additional_info += f"  Fast Scale: {self.fast_scale}\n"
        
        if self.slow_rv is not None:
            additional_info += f"  Slow-Scale RV: {self.slow_rv:.6e}\n"
        
        if self.fast_rv is not None:
            additional_info += f"  Fast-Scale RV: {self.fast_rv:.6e}\n"
        
        if self.noise_variance is not None:
            additional_info += f"  Estimated Noise Variance: {self.noise_variance:.6e}\n"
        
        if self.bias_correction is not None:
            additional_info += f"  Bias Correction Applied: {self.bias_correction}\n"
        
        return base_summary + additional_info
    
    def plot(self, figsize: Tuple[int, int] = (10, 6)) -> Optional[Any]:
        """Plot two-scale realized variance results.
        
        Args:
            figsize: Figure size (width, height) in inches
            
        Returns:
            Optional[Any]: Matplotlib figure if matplotlib is available, None otherwise
            
        Raises:
            ImportError: If matplotlib is not available
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib is required for plotting. Please install matplotlib.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot realized variance by scale if available
        if self.scales_tested is not None and self.rv_by_scale is not None:
            scales = self.scales_tested
            rv_values = [self.rv_by_scale[scale] for scale in scales]
            
            ax.plot(scales, rv_values, 'o-', label='Realized Variance by Scale')
            
            # Mark the selected scales
            if self.slow_scale is not None:
                ax.axvline(x=self.slow_scale, color='r', linestyle='--', 
                          label=f'Slow Scale ({self.slow_scale})')
            
            if self.fast_scale is not None:
                ax.axvline(x=self.fast_scale, color='g', linestyle='--', 
                          label=f'Fast Scale ({self.fast_scale})')
            
            ax.set_xlabel('Scale')
            ax.set_ylabel('Realized Variance')
            ax.set_title('Two-Scale Realized Variance Estimation')
            ax.legend()
            
            # Add two-scale RV estimate as text
            if self.realized_measure is not None:
                ax.text(0.05, 0.95, f'Two-Scale RV: {self.realized_measure[0]:.6e}',
                        transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # If no scale data, just show the returns
        else:
            if self.returns is not None:
                # Plot returns
                ax.plot(self.returns, 'b-', alpha=0.5)
                ax.set_title('Returns with Two-Scale Realized Variance Estimate')
                ax.set_xlabel('Observation')
                ax.set_ylabel('Return')
                
                # Add two-scale RV estimate as text
                if self.realized_measure is not None:
                    ax.text(0.05, 0.95, 
                            f'Two-Scale RV: {self.realized_measure[0]:.6e}\n'
                            f'Slow Scale: {self.slow_scale}\n'
                            f'Fast Scale: {self.fast_scale}',
                            transform=ax.transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig


@jit(nopython=True, cache=True)
def _compute_twoscale_variance_numba(returns: np.ndarray, 
                                    slow_scale: int, 
                                    fast_scale: int,
                                    bias_correction: bool = True) -> Tuple[float, float, float]:
    """
    Numba-accelerated implementation of two-scale realized variance estimation.
    
    Args:
        returns: Array of returns
        slow_scale: Sampling frequency for the slow time scale (sparse sampling)
        fast_scale: Sampling frequency for the fast time scale (frequent sampling)
        bias_correction: Whether to apply bias correction
        
    Returns:
        Tuple of (two_scale_rv, slow_rv, fast_rv)
    """
    n = len(returns)
    
    # Compute slow-scale realized variance
    slow_rv = 0.0
    slow_count = 0
    for i in range(0, n, slow_scale):
        if i + slow_scale < n:
            # Aggregate returns over the slow scale
            agg_return = 0.0
            for j in range(slow_scale):
                agg_return += returns[i + j]
            slow_rv += agg_return * agg_return
            slow_count += 1
    
    # Scale by the number of observations
    if slow_count > 0:
        slow_rv /= slow_count
    
    # Compute fast-scale realized variance
    fast_rv = 0.0
    fast_count = 0
    for i in range(0, n, fast_scale):
        if i + fast_scale < n:
            # Aggregate returns over the fast scale
            agg_return = 0.0
            for j in range(fast_scale):
                agg_return += returns[i + j]
            fast_rv += agg_return * agg_return
            fast_count += 1
    
    # Scale by the number of observations
    if fast_count > 0:
        fast_rv /= fast_count
    
    # Compute adjustment factor
    adjustment = float(slow_scale) / float(fast_scale)
    
    # Compute two-scale realized variance
    two_scale_rv = slow_rv - (fast_rv / adjustment)
    
    # Apply bias correction if requested
    if bias_correction:
        # Adjust for small sample bias
        two_scale_rv *= (1.0 + 1.0 / n)
    
    return two_scale_rv, slow_rv, fast_rv


def compute_twoscale_variance(returns: np.ndarray, 
                             slow_scale: int, 
                             fast_scale: int = 1,
                             bias_correction: bool = True) -> Tuple[float, float, float]:
    """
    Compute two-scale realized variance.
    
    This function implements the two-scale realized variance estimator, which combines
    two different sampling frequencies to correct for market microstructure noise.
    
    Args:
        returns: Array of returns
        slow_scale: Sampling frequency for the slow time scale (sparse sampling)
        fast_scale: Sampling frequency for the fast time scale (frequent sampling)
        bias_correction: Whether to apply bias correction
        
    Returns:
        Tuple of (two_scale_rv, slow_rv, fast_rv)
        
    Raises:
        ValueError: If returns is not a 1D array or contains invalid values
        ValueError: If slow_scale <= fast_scale
        
    References:
        Zhang, L., Mykland, P. A., & Aït-Sahalia, Y. (2005). A tale of two time scales:
        Determining integrated volatility with noisy high-frequency data. Journal of the
        American Statistical Association, 100(472), 1394-1411.
    
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.twoscale_variance import compute_twoscale_variance
        >>> np.random.seed(42)
        >>> returns = np.random.normal(0, 0.01, 1000)
        >>> two_scale_rv, slow_rv, fast_rv = compute_twoscale_variance(returns, 10, 1)
        >>> two_scale_rv
        0.0009...
    """
    # Convert to numpy array if not already
    returns = np.asarray(returns)
    
    # Validate inputs
    if returns.ndim != 1:
        raise ValueError("returns must be a 1D array")
    
    if np.isnan(returns).any() or np.isinf(returns).any():
        raise ValueError("returns contains NaN or infinite values")
    
    if slow_scale <= fast_scale:
        raise ValueError(f"slow_scale ({slow_scale}) must be greater than fast_scale ({fast_scale})")
    
    # Use Numba-accelerated implementation if available
    if HAS_NUMBA:
        return _compute_twoscale_variance_numba(returns, slow_scale, fast_scale, bias_correction)
    
    # Pure NumPy implementation
    n = len(returns)
    
    # Compute slow-scale realized variance
    slow_returns = np.zeros(n // slow_scale)
    for i in range(len(slow_returns)):
        start_idx = i * slow_scale
        end_idx = min((i + 1) * slow_scale, n)
        slow_returns[i] = np.sum(returns[start_idx:end_idx])
    
    slow_rv = np.mean(slow_returns**2)
    
    # Compute fast-scale realized variance
    fast_returns = np.zeros(n // fast_scale)
    for i in range(len(fast_returns)):
        start_idx = i * fast_scale
        end_idx = min((i + 1) * fast_scale, n)
        fast_returns[i] = np.sum(returns[start_idx:end_idx])
    
    fast_rv = np.mean(fast_returns**2)
    
    # Compute adjustment factor
    adjustment = float(slow_scale) / float(fast_scale)
    
    # Compute two-scale realized variance
    two_scale_rv = slow_rv - (fast_rv / adjustment)
    
    # Apply bias correction if requested
    if bias_correction:
        # Adjust for small sample bias
        two_scale_rv *= (1.0 + 1.0 / n)
    
    return two_scale_rv, slow_rv, fast_rv


def select_optimal_scales(returns: np.ndarray, 
                         max_scales: int = 30,
                         min_slow_scale: int = 5) -> Tuple[int, int]:
    """
    Select optimal scales for two-scale realized variance estimation.
    
    This function analyzes the behavior of realized variance at different scales
    to determine the optimal slow and fast scales for two-scale estimation.
    
    Args:
        returns: Array of returns
        max_scales: Maximum number of scales to consider
        min_slow_scale: Minimum value for the slow scale
        
    Returns:
        Tuple of (optimal_slow_scale, optimal_fast_scale)
        
    Raises:
        ValueError: If returns is not a 1D array or contains invalid values
    
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.twoscale_variance import select_optimal_scales
        >>> np.random.seed(42)
        >>> returns = np.random.normal(0, 0.01, 1000)
        >>> slow_scale, fast_scale = select_optimal_scales(returns)
        >>> slow_scale, fast_scale
        (10, 1)
    """
    # Convert to numpy array if not already
    returns = np.asarray(returns)
    
    # Validate inputs
    if returns.ndim != 1:
        raise ValueError("returns must be a 1D array")
    
    if np.isnan(returns).any() or np.isinf(returns).any():
        raise ValueError("returns contains NaN or infinite values")
    
    n = len(returns)
    max_scales = min(max_scales, n // 10)  # Ensure max_scales is not too large
    
    # Compute realized variance at different scales
    scales = np.arange(1, max_scales + 1)
    rv = np.zeros(len(scales))
    
    for i, scale in enumerate(scales):
        # Skip every scale-th observation
        sampled_returns = np.zeros(n // scale)
        for j in range(len(sampled_returns)):
            start_idx = j * scale
            end_idx = min((j + 1) * scale, n)
            sampled_returns[j] = np.sum(returns[start_idx:end_idx])
        
        rv[i] = np.mean(sampled_returns**2)
    
    # Analyze the behavior of realized variance across scales
    # The slow scale should be where the realized variance stabilizes
    # The fast scale is typically 1 (highest frequency)
    
    # Compute first differences of realized variance
    rv_diff = np.diff(rv)
    
    # Normalize differences
    if np.max(np.abs(rv_diff)) > 0:
        norm_diff = np.abs(rv_diff) / np.max(np.abs(rv_diff))
    else:
        norm_diff = np.abs(rv_diff)
    
    # Find the point where the realized variance stabilizes
    # (where the normalized difference is below a threshold)
    threshold = 0.1
    stable_points = np.where(norm_diff < threshold)[0]
    
    if len(stable_points) > 0:
        # Add 1 because diff reduces the length by 1, and add 1 more for 1-based scale
        optimal_slow_scale = stable_points[0] + 2
    else:
        # If no stable point is found, use a default value
        optimal_slow_scale = max(min_slow_scale, max_scales // 3)
    
    # Ensure optimal_slow_scale is at least min_slow_scale
    optimal_slow_scale = max(optimal_slow_scale, min_slow_scale)
    
    # Fast scale is typically 1 (highest frequency)
    optimal_fast_scale = 1
    
    return optimal_slow_scale, optimal_fast_scale


class TwoScaleVarianceEstimator(NoiseRobustEstimator):
    """Estimator for two-scale realized variance in high-frequency financial data.
    
    This class implements the two-scale realized variance estimator, which combines
    two different sampling frequencies to correct for market microstructure noise.
    The estimator is based on the work of Zhang, Mykland, and Aït-Sahalia (2005) and
    provides a robust approach to volatility estimation in the presence of noise.
    
    The two-scale estimator uses a combination of a slow time scale (sparse sampling)
    and a fast time scale (frequent sampling) to separate the true volatility from
    the noise component. This approach provides a consistent estimator of integrated
    variance even when high-frequency data is contaminated with microstructure noise.
    
    Attributes:
        config: Configuration parameters for the estimator
        name: A descriptive name for the estimator
    """
    
    def __init__(self, config: Optional[TwoScaleVarianceConfig] = None, name: str = "TwoScaleVarianceEstimator"):
        """Initialize the two-scale realized variance estimator.
        
        Args:
            config: Configuration parameters for the estimator
            name: A descriptive name for the estimator
        """
        config_to_use = config if config is not None else TwoScaleVarianceConfig()
        super().__init__(config=config_to_use, name=name)
        self._slow_scale: Optional[int] = None
        self._fast_scale: Optional[int] = None
        self._slow_rv: Optional[float] = None
        self._fast_rv: Optional[float] = None
        self._scales_tested: Optional[List[int]] = None
        self._rv_by_scale: Optional[Dict[int, float]] = None
    
    @property
    def config(self) -> TwoScaleVarianceConfig:
        """Get the estimator configuration.
        
        Returns:
            TwoScaleVarianceConfig: The estimator configuration
        """
        return cast(TwoScaleVarianceConfig, self._config)
    
    @config.setter
    def config(self, config: TwoScaleVarianceConfig) -> None:
        """Set the estimator configuration.
        
        Args:
            config: New configuration parameters
            
        Raises:
            TypeError: If config is not a TwoScaleVarianceConfig
        """
        if not isinstance(config, TwoScaleVarianceConfig):
            raise TypeError(f"config must be a TwoScaleVarianceConfig, got {type(config)}")
        
        self._config = config
        self._fitted = False  # Reset fitted state when configuration changes
    
    @property
    def slow_scale(self) -> Optional[int]:
        """Get the slow scale used for estimation.
        
        Returns:
            Optional[int]: The slow scale if the estimator has been fitted,
                          None otherwise
        """
        return self._slow_scale
    
    @property
    def fast_scale(self) -> Optional[int]:
        """Get the fast scale used for estimation.
        
        Returns:
            Optional[int]: The fast scale if the estimator has been fitted,
                          None otherwise
        """
        return self._fast_scale
    
    @property
    def slow_rv(self) -> Optional[float]:
        """Get the realized variance at the slow scale.
        
        Returns:
            Optional[float]: The slow-scale realized variance if the estimator has been fitted,
                            None otherwise
        """
        return self._slow_rv
    
    @property
    def fast_rv(self) -> Optional[float]:
        """Get the realized variance at the fast scale.
        
        Returns:
            Optional[float]: The fast-scale realized variance if the estimator has been fitted,
                            None otherwise
        """
        return self._fast_rv
    
    def _compute_realized_measure(self, 
                                 prices: np.ndarray, 
                                 times: np.ndarray, 
                                 returns: np.ndarray,
                                 **kwargs: Any) -> np.ndarray:
        """Compute the two-scale realized variance from the preprocessed data.
        
        Args:
            prices: Preprocessed price data
            times: Preprocessed time points
            returns: Returns computed from prices
            **kwargs: Additional keyword arguments for computation
            
        Returns:
            np.ndarray: Two-scale realized variance estimate
            
        Raises:
            ValueError: If computation fails
        """
        # Determine scales to use
        if self.config.auto_scale:
            # Automatically determine optimal scales
            try:
                slow_scale, fast_scale = select_optimal_scales(
                    returns, 
                    max_scales=self.config.max_scales,
                    min_slow_scale=5
                )
                
                logger.info(f"Automatically selected scales: slow_scale={slow_scale}, fast_scale={fast_scale}")
            except Exception as e:
                logger.warning(f"Automatic scale selection failed: {str(e)}. Using default scales.")
                slow_scale = 10  # Default slow scale
                fast_scale = 1   # Default fast scale
        else:
            # Use user-specified scales
            if self.config.slow_scale is None:
                raise ValueError("slow_scale must be specified when auto_scale is False")
            
            slow_scale = self.config.slow_scale
            fast_scale = self.config.fast_scale if self.config.fast_scale is not None else 1
        
        # Store scales
        self._slow_scale = slow_scale
        self._fast_scale = fast_scale
        
        # Compute realized variance at different scales for diagnostics
        if self.config.auto_scale or self.config.plot:
            max_scales = min(self.config.max_scales, len(returns) // 10)
            scales = list(range(1, max_scales + 1))
            rv_by_scale = {}
            
            for scale in scales:
                # Skip every scale-th observation
                sampled_returns = np.zeros(len(returns) // scale)
                for i in range(len(sampled_returns)):
                    start_idx = i * scale
                    end_idx = min((i + 1) * scale, len(returns))
                    sampled_returns[i] = np.sum(returns[start_idx:end_idx])
                
                rv_by_scale[scale] = np.mean(sampled_returns**2)
            
            self._scales_tested = scales
            self._rv_by_scale = rv_by_scale
        
        # Compute two-scale realized variance
        try:
            two_scale_rv, slow_rv, fast_rv = compute_twoscale_variance(
                returns, 
                slow_scale, 
                fast_scale,
                bias_correction=self.config.bias_correction
            )
            
            # Store results
            self._slow_rv = slow_rv
            self._fast_rv = fast_rv
            
            # Estimate noise variance if requested
            if self.config.estimate_noise:
                try:
                    noise_var = noise_variance(
                        returns, 
                        method=self.config.noise_method,
                        bias_correction=self.config.bias_correction
                    )
                    
                    self._noise_variance = noise_var
                except Exception as e:
                    logger.warning(f"Noise variance estimation failed: {str(e)}")
                    self._noise_variance = None
            
            # Create result object
            result = TwoScaleVarianceResult(
                model_name=self._name,
                realized_measure=np.array([two_scale_rv]),
                prices=prices,
                times=times,
                returns=returns,
                slow_scale=slow_scale,
                fast_scale=fast_scale,
                slow_rv=slow_rv,
                fast_rv=fast_rv,
                noise_variance=self._noise_variance,
                bias_correction=self.config.bias_correction,
                scales_tested=self._scales_tested,
                rv_by_scale=self._rv_by_scale
            )
            
            # Store result
            self._results = result
            
            # Generate plot if requested
            if self.config.plot and HAS_MATPLOTLIB:
                result.plot()
            
            return np.array([two_scale_rv])
        
        except Exception as e:
            logger.error(f"Two-scale realized variance estimation failed: {str(e)}")
            raise RealizedVolatilityError(
                f"Two-scale realized variance estimation failed: {str(e)}",
                estimator_type="TwoScaleVariance",
                issue=str(e)
            ) from e
    
    def fit(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> TwoScaleVarianceResult:
        """Fit the two-scale realized variance estimator to the provided data.
        
        Args:
            data: The data to fit the estimator to, as a tuple of (prices, times)
            **kwargs: Additional keyword arguments for estimation
            
        Returns:
            TwoScaleVarianceResult: The estimation results
            
        Raises:
            ValueError: If the data is invalid
            RuntimeError: If the estimation fails
        """
        result = super().fit(data, **kwargs)
        return cast(TwoScaleVarianceResult, result)
    
    async def fit_async(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> TwoScaleVarianceResult:
        """Asynchronously fit the two-scale realized variance estimator to the provided data.
        
        Args:
            data: The data to fit the estimator to, as a tuple of (prices, times)
            **kwargs: Additional keyword arguments for estimation
            
        Returns:
            TwoScaleVarianceResult: The estimation results
            
        Raises:
            ValueError: If the data is invalid
            RuntimeError: If the estimation fails
        """
        result = await super().fit_async(data, **kwargs)
        return cast(TwoScaleVarianceResult, result)
    
    def calibrate(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> TwoScaleVarianceConfig:
        """Calibrate the estimator configuration based on the provided data.
        
        This method analyzes the input data and determines optimal configuration
        parameters for the two-scale realized variance estimator.
        
        Args:
            data: The data to calibrate the estimator with, as a tuple of (prices, times)
            **kwargs: Additional keyword arguments for calibration
            
        Returns:
            TwoScaleVarianceConfig: Calibrated configuration
        """
        # Validate input data
        self.validate_data(data)
        prices, times = data
        
        # Preprocess data
        processed_prices, processed_times, returns = self.preprocess_data(prices, times)
        
        # Determine optimal scales
        try:
            slow_scale, fast_scale = select_optimal_scales(
                returns, 
                max_scales=self.config.max_scales,
                min_slow_scale=5
            )
            
            logger.info(f"Calibration selected scales: slow_scale={slow_scale}, fast_scale={fast_scale}")
        except Exception as e:
            logger.warning(f"Optimal scale selection failed: {str(e)}. Using default scales.")
            slow_scale = 10  # Default slow scale
            fast_scale = 1   # Default fast scale
        
        # Create calibrated configuration
        calibrated_config = TwoScaleVarianceConfig(
            slow_scale=slow_scale,
            fast_scale=fast_scale,
            auto_scale=False,  # Set to False since we've already determined optimal scales
            max_scales=self.config.max_scales,
            bias_correction=True,
            estimate_noise=True,
            noise_method='auto',
            plot=self.config.plot
        )
        
        return calibrated_config
    
    def compare_scales(self, 
                      data: Tuple[np.ndarray, np.ndarray], 
                      max_scales: int = 30,
                      figsize: Tuple[int, int] = (10, 6)) -> Optional[Any]:
        """Compare realized variance at different scales.
        
        This method computes realized variance at different scales and visualizes
        the results to help understand the behavior of realized variance across scales.
        
        Args:
            data: The data to analyze, as a tuple of (prices, times)
            max_scales: Maximum number of scales to consider
            figsize: Figure size (width, height) in inches
            
        Returns:
            Optional[Any]: Matplotlib figure if matplotlib is available, None otherwise
            
        Raises:
            ImportError: If matplotlib is not available
            ValueError: If the data is invalid
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib is required for plotting. Please install matplotlib.")
        
        # Validate input data
        self.validate_data(data)
        prices, times = data
        
        # Preprocess data
        processed_prices, processed_times, returns = self.preprocess_data(prices, times)
        
        # Compute realized variance at different scales
        n = len(returns)
        max_scales = min(max_scales, n // 10)  # Ensure max_scales is not too large
        scales = list(range(1, max_scales + 1))
        rv = np.zeros(len(scales))
        
        for i, scale in enumerate(scales):
            # Skip every scale-th observation
            sampled_returns = np.zeros(n // scale)
            for j in range(len(sampled_returns)):
                start_idx = j * scale
                end_idx = min((j + 1) * scale, n)
                sampled_returns[j] = np.sum(returns[start_idx:end_idx])
            
            rv[i] = np.mean(sampled_returns**2)
        
        # Determine optimal scales
        try:
            slow_scale, fast_scale = select_optimal_scales(
                returns, 
                max_scales=max_scales,
                min_slow_scale=5
            )
        except Exception as e:
            logger.warning(f"Optimal scale selection failed: {str(e)}. Using default scales.")
            slow_scale = 10  # Default slow scale
            fast_scale = 1   # Default fast scale
        
        # Compute two-scale realized variance
        two_scale_rv, _, _ = compute_twoscale_variance(
            returns, 
            slow_scale, 
            fast_scale,
            bias_correction=self.config.bias_correction
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot realized variance by scale
        ax.plot(scales, rv, 'o-', label='Realized Variance by Scale')
        
        # Mark the selected scales
        ax.axvline(x=slow_scale, color='r', linestyle='--', 
                  label=f'Slow Scale ({slow_scale})')
        
        ax.axvline(x=fast_scale, color='g', linestyle='--', 
                  label=f'Fast Scale ({fast_scale})')
        
        ax.set_xlabel('Scale')
        ax.set_ylabel('Realized Variance')
        ax.set_title('Two-Scale Realized Variance Estimation')
        ax.legend()
        
        # Add two-scale RV estimate as text
        ax.text(0.05, 0.95, f'Two-Scale RV: {two_scale_rv:.6e}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig


# Register Numba-accelerated functions if available
def _register_numba_functions() -> None:
    """
    Register Numba JIT-compiled functions for two-scale realized variance estimation.
    
    This function is called during module initialization to register
    performance-critical functions for JIT compilation if Numba is available.
    """
    if HAS_NUMBA:
        # The functions are already decorated with @jit, so no additional
        # registration is needed here. This function is kept for consistency
        # with the module structure and potential future enhancements.
        logger.debug("Two-scale realized variance Numba JIT functions registered")
    else:
        logger.info("Numba not available. Two-scale realized variance will use pure NumPy implementations.")


# Initialize the module
_register_numba_functions()


# mfe/models/realized/twoscale_variance.py
"""
Two-scale realized variance estimator for high-frequency financial data.

This module implements the two-scale realized variance estimator, which combines
two different sampling frequencies to correct for market microstructure noise.
The estimator is based on the work of Zhang, Mykland, and Aït-Sahalia (2005) and
provides a robust approach to volatility estimation in the presence of noise.

The two-scale estimator uses a combination of a slow time scale (sparse sampling)
and a fast time scale (frequent sampling) to separate the true volatility from
the noise component. This approach provides a consistent estimator of integrated
variance even when high-frequency data is contaminated with microstructure noise.

The implementation leverages NumPy's efficient array operations with Numba
acceleration for performance-critical calculations. It supports both raw NumPy
arrays and Pandas DataFrames with datetime indices for convenient time series analysis.

Classes:
    TwoScaleVarianceConfig: Configuration parameters for two-scale variance estimation
    TwoScaleVarianceResult: Result container for two-scale variance estimation
    TwoScaleVarianceEstimator: Class for estimating two-scale realized variance

References:
    Zhang, L., Mykland, P. A., & Aït-Sahalia, Y. (2005). A tale of two time scales:
    Determining integrated volatility with noisy high-frequency data. Journal of the
    American Statistical Association, 100(472), 1394-1411.
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Literal, Optional, Sequence, 
    Tuple, Type, TypeVar, Union, cast, overload
)
import numpy as np
import pandas as pd
from scipy import stats, optimize

from ...core.base import ModelBase
from ...core.parameters import ParameterBase, ParameterError, validate_positive, validate_non_negative
from ...core.exceptions import (
    DimensionError, NumericError, ParameterError, RealizedVolatilityError,
    raise_dimension_error, raise_numeric_error, warn_numeric
)
from .base import BaseRealizedEstimator, RealizedEstimatorConfig, RealizedEstimatorResult, NoiseRobustEstimator
from .utils import compute_realized_variance, compute_subsampled_measure
from .noise_estimate import noise_variance

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    HAS_NUMBA = False

# Set up module-level logger
logger = logging.getLogger("mfe.models.realized.twoscale_variance")

# Try to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.info("Matplotlib not available. Plotting functions will be disabled.")


@dataclass
class TwoScaleVarianceConfig(RealizedEstimatorConfig):
    """Configuration parameters for two-scale realized variance estimation.
    
    This class extends RealizedEstimatorConfig to provide specialized configuration
    parameters for two-scale realized variance estimation.
    
    Attributes:
        slow_scale: Sampling frequency for the slow time scale (sparse sampling)
        fast_scale: Sampling frequency for the fast time scale (frequent sampling)
        auto_scale: Whether to automatically determine optimal scales
        max_scales: Maximum number of scales to consider for auto-scale
        bias_correction: Whether to apply bias correction
        estimate_noise: Whether to estimate noise variance
        noise_method: Method for estimating noise variance
        plot: Whether to generate diagnostic plots
    """
    
    slow_scale: Optional[int] = None
    fast_scale: Optional[int] = 1
    auto_scale: bool = True
    max_scales: int = 30
    bias_correction: bool = True
    estimate_noise: bool = True
    noise_method: Literal['ac1', 'bandi-russell', 'signature', 'ml', 'auto'] = 'auto'
    plot: bool = False
    
    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        super().__post_init__()
        
        # Validate slow_scale and fast_scale if provided
        if self.slow_scale is not None:
            validate_positive(self.slow_scale, "slow_scale")
        
        if self.fast_scale is not None:
            validate_positive(self.fast_scale, "fast_scale")
        
        # Validate max_scales
        validate_positive(self.max_scales, "max_scales")
        
        # Validate noise_method
        valid_methods = ['ac1', 'bandi-russell', 'signature', 'ml', 'auto']
        if self.noise_method not in valid_methods:
            raise ParameterError(f"noise_method must be one of {valid_methods}, got {self.noise_method}")
        
        # Check that slow_scale > fast_scale if both are provided
        if self.slow_scale is not None and self.fast_scale is not None:
            if self.slow_scale <= self.fast_scale:
                raise ParameterError(
                    f"slow_scale ({self.slow_scale}) must be greater than fast_scale ({self.fast_scale})",
                    param_name="slow_scale",
                    param_value=self.slow_scale,
                    constraint=f"slow_scale > fast_scale ({self.fast_scale})"
                )


@dataclass
class TwoScaleVarianceResult(RealizedEstimatorResult):
    """Result container for two-scale realized variance estimation.
    
    This class extends RealizedEstimatorResult to provide specialized functionality
    for two-scale realized variance estimation results, including additional metadata
    and diagnostic information specific to two-scale estimation.
    
    Attributes:
        slow_scale: Sampling frequency for the slow time scale (sparse sampling)
        fast_scale: Sampling frequency for the fast time scale (frequent sampling)
        slow_rv: Realized variance at the slow time scale
        fast_rv: Realized variance at the fast time scale
        noise_variance: Estimated noise variance
        bias_correction: Whether bias correction was applied
        scales_tested: List of scales tested for auto-scale selection
        rv_by_scale: Realized variance at each tested scale
    """
    
    slow_scale: Optional[int] = None
    fast_scale: Optional[int] = None
    slow_rv: Optional[float] = None
    fast_rv: Optional[float] = None
    bias_correction: Optional[bool] = None
    scales_tested: Optional[List[int]] = None
    rv_by_scale: Optional[Dict[int, float]] = None
    
    def __post_init__(self) -> None:
        """Validate result object after initialization."""
        super().__post_init__()
    
    def summary(self) -> str:
        """Generate a text summary of the two-scale realized variance results.
        
        Returns:
            str: A formatted string containing the two-scale realized variance results summary
        """
        base_summary = super().summary()
        
        additional_info = f"Two-Scale Realized Variance Results:\n"
        additional_info += f"  Slow Scale: {self.slow_scale}\n"
        additional_info += f"  Fast Scale: {self.fast_scale}\n"
        
        if self.slow_rv is not None:
            additional_info += f"  Slow-Scale RV: {self.slow_rv:.6e}\n"
        
        if self.fast_rv is not None:
            additional_info += f"  Fast-Scale RV: {self.fast_rv:.6e}\n"
        
        if self.noise_variance is not None:
            additional_info += f"  Estimated Noise Variance: {self.noise_variance:.6e}\n"
        
        if self.bias_correction is not None:
            additional_info += f"  Bias Correction Applied: {self.bias_correction}\n"
        
        return base_summary + additional_info
    
    def plot(self, figsize: Tuple[int, int] = (10, 6)) -> Optional[Any]:
        """Plot two-scale realized variance results.
        
        Args:
            figsize: Figure size (width, height) in inches
            
        Returns:
            Optional[Any]: Matplotlib figure if matplotlib is available, None otherwise
            
        Raises:
            ImportError: If matplotlib is not available
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib is required for plotting. Please install matplotlib.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot realized variance by scale if available
        if self.scales_tested is not None and self.rv_by_scale is not None:
            scales = self.scales_tested
            rv_values = [self.rv_by_scale[scale] for scale in scales]
            
            ax.plot(scales, rv_values, 'o-', label='Realized Variance by Scale')
            
            # Mark the selected scales
            if self.slow_scale is not None:
                ax.axvline(x=self.slow_scale, color='r', linestyle='--', 
                          label=f'Slow Scale ({self.slow_scale})')
            
            if self.fast_scale is not None:
                ax.axvline(x=self.fast_scale, color='g', linestyle='--', 
                          label=f'Fast Scale ({self.fast_scale})')
            
            ax.set_xlabel('Scale')
            ax.set_ylabel('Realized Variance')
            ax.set_title('Two-Scale Realized Variance Estimation')
            ax.legend()
            
            # Add two-scale RV estimate as text
            if self.realized_measure is not None:
                ax.text(0.05, 0.95, f'Two-Scale RV: {self.realized_measure[0]:.6e}',
                        transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # If no scale data, just show the returns
        else:
            if self.returns is not None:
                # Plot returns
                ax.plot(self.returns, 'b-', alpha=0.5)
                ax.set_title('Returns with Two-Scale Realized Variance Estimate')
                ax.set_xlabel('Observation')
                ax.set_ylabel('Return')
                
                # Add two-scale RV estimate as text
                if self.realized_measure is not None:
                    ax.text(0.05, 0.95, 
                            f'Two-Scale RV: {self.realized_measure[0]:.6e}\n'
                            f'Slow Scale: {self.slow_scale}\n'
                            f'Fast Scale: {self.fast_scale}',
                            transform=ax.transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig


@jit(nopython=True, cache=True)
def _compute_twoscale_variance_numba(returns: np.ndarray, 
                                    slow_scale: int, 
                                    fast_scale: int,
                                    bias_correction: bool = True) -> Tuple[float, float, float]:
    """
    Numba-accelerated implementation of two-scale realized variance estimation.
    
    Args:
        returns: Array of returns
        slow_scale: Sampling frequency for the slow time scale (sparse sampling)
        fast_scale: Sampling frequency for the fast time scale (frequent sampling)
        bias_correction: Whether to apply bias correction
        
    Returns:
        Tuple of (two_scale_rv, slow_rv, fast_rv)
    """
    n = len(returns)
    
    # Compute slow-scale realized variance
    slow_rv = 0.0
    slow_count = 0
    for i in range(0, n, slow_scale):
        if i + slow_scale < n:
            # Aggregate returns over the slow scale
            agg_return = 0.0
            for j in range(slow_scale):
                agg_return += returns[i + j]
            slow_rv += agg_return * agg_return
            slow_count += 1
    
    # Scale by the number of observations
    if slow_count > 0:
        slow_rv /= slow_count
    
    # Compute fast-scale realized variance
    fast_rv = 0.0
    fast_count = 0
    for i in range(0, n, fast_scale):
        if i + fast_scale < n:
            # Aggregate returns over the fast scale
            agg_return = 0.0
            for j in range(fast_scale):
                agg_return += returns[i + j]
            fast_rv += agg_return * agg_return
            fast_count += 1
    
    # Scale by the number of observations
    if fast_count > 0:
        fast_rv /= fast_count
    
    # Compute adjustment factor
    adjustment = float(slow_scale) / float(fast_scale)
    
    # Compute two-scale realized variance
    two_scale_rv = slow_rv - (fast_rv / adjustment)
    
    # Apply bias correction if requested
    if bias_correction:
        # Adjust for small sample bias
        two_scale_rv *= (1.0 + 1.0 / n)
    
    return two_scale_rv, slow_rv, fast_rv


def compute_twoscale_variance(returns: np.ndarray, 
                             slow_scale: int, 
                             fast_scale: int = 1,
                             bias_correction: bool = True) -> Tuple[float, float, float]:
    """
    Compute two-scale realized variance.
    
    This function implements the two-scale realized variance estimator, which combines
    two different sampling frequencies to correct for market microstructure noise.
    
    Args:
        returns: Array of returns
        slow_scale: Sampling frequency for the slow time scale (sparse sampling)
        fast_scale: Sampling frequency for the fast time scale (frequent sampling)
        bias_correction: Whether to apply bias correction
        
    Returns:
        Tuple of (two_scale_rv, slow_rv, fast_rv)
        
    Raises:
        ValueError: If returns is not a 1D array or contains invalid values
        ValueError: If slow_scale <= fast_scale
        
    References:
        Zhang, L., Mykland, P. A., & Aït-Sahalia, Y. (2005). A tale of two time scales:
        Determining integrated volatility with noisy high-frequency data. Journal of the
        American Statistical Association, 100(472), 1394-1411.
    
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.twoscale_variance import compute_twoscale_variance
        >>> np.random.seed(42)
        >>> returns = np.random.normal(0, 0.01, 1000)
        >>> two_scale_rv, slow_rv, fast_rv = compute_twoscale_variance(returns, 10, 1)
        >>> two_scale_rv
        0.0009...
    """
    # Convert to numpy array if not already
    returns = np.asarray(returns)
    
    # Validate inputs
    if returns.ndim != 1:
        raise ValueError("returns must be a 1D array")
    
    if np.isnan(returns).any() or np.isinf(returns).any():
        raise ValueError("returns contains NaN or infinite values")
    
    if slow_scale <= fast_scale:
        raise ValueError(f"slow_scale ({slow_scale}) must be greater than fast_scale ({fast_scale})")
    
    # Use Numba-accelerated implementation if available
    if HAS_NUMBA:
        return _compute_twoscale_variance_numba(returns, slow_scale, fast_scale, bias_correction)
    
    # Pure NumPy implementation
    n = len(returns)
    
    # Compute slow-scale realized variance
    slow_returns = np.zeros(n // slow_scale)
    for i in range(len(slow_returns)):
        start_idx = i * slow_scale
        end_idx = min((i + 1) * slow_scale, n)
        slow_returns[i] = np.sum(returns[start_idx:end_idx])
    
    slow_rv = np.mean(slow_returns**2)
    
    # Compute fast-scale realized variance
    fast_returns = np.zeros(n // fast_scale)
    for i in range(len(fast_returns)):
        start_idx = i * fast_scale
        end_idx = min((i + 1) * fast_scale, n)
        fast_returns[i] = np.sum(returns[start_idx:end_idx])
    
    fast_rv = np.mean(fast_returns**2)
    
    # Compute adjustment factor
    adjustment = float(slow_scale) / float(fast_scale)
    
    # Compute two-scale realized variance
    two_scale_rv = slow_rv - (fast_rv / adjustment)
    
    # Apply bias correction if requested
    if bias_correction:
        # Adjust for small sample bias
        two_scale_rv *= (1.0 + 1.0 / n)
    
    return two_scale_rv, slow_rv, fast_rv


def select_optimal_scales(returns: np.ndarray, 
                         max_scales: int = 30,
                         min_slow_scale: int = 5) -> Tuple[int, int]:
    """
    Select optimal scales for two-scale realized variance estimation.
    
    This function analyzes the behavior of realized variance at different scales
    to determine the optimal slow and fast scales for two-scale estimation.
    
    Args:
        returns: Array of returns
        max_scales: Maximum number of scales to consider
        min_slow_scale: Minimum value for the slow scale
        
    Returns:
        Tuple of (optimal_slow_scale, optimal_fast_scale)
        
    Raises:
        ValueError: If returns is not a 1D array or contains invalid values
    
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.twoscale_variance import select_optimal_scales
        >>> np.random.seed(42)
        >>> returns = np.random.normal(0, 0.01, 1000)
        >>> slow_scale, fast_scale = select_optimal_scales(returns)
        >>> slow_scale, fast_scale
        (10, 1)
    """
    # Convert to numpy array if not already
    returns = np.asarray(returns)
    
    # Validate inputs
    if returns.ndim != 1:
        raise ValueError("returns must be a 1D array")
    
    if np.isnan(returns).any() or np.isinf(returns).any():
        raise ValueError("returns contains NaN or infinite values")
    
    n = len(returns)
    max_scales = min(max_scales, n // 10)  # Ensure max_scales is not too large
    
    # Compute realized variance at different scales
    scales = np.arange(1, max_scales + 1)
    rv = np.zeros(len(scales))
    
    for i, scale in enumerate(scales):
        # Skip every scale-th observation
        sampled_returns = np.zeros(n // scale)
        for j in range(len(sampled_returns)):
            start_idx = j * scale
            end_idx = min((j + 1) * scale, n)
            sampled_returns[j] = np.sum(returns[start_idx:end_idx])
        
        rv[i] = np.mean(sampled_returns**2)
    
    # Analyze the behavior of realized variance across scales
    # The slow scale should be where the realized variance stabilizes
    # The fast scale is typically 1 (highest frequency)
    
    # Compute first differences of realized variance
    rv_diff = np.diff(rv)
    
    # Normalize differences
    if np.max(np.abs(rv_diff)) > 0:
        norm_diff = np.abs(rv_diff) / np.max(np.abs(rv_diff))
    else:
        norm_diff = np.abs(rv_diff)
    
    # Find the point where the realized variance stabilizes
    # (where the normalized difference is below a threshold)
    threshold = 0.1
    stable_points = np.where(norm_diff < threshold)[0]
    
    if len(stable_points) > 0:
        # Add 1 because diff reduces the length by 1, and add 1 more for 1-based scale
        optimal_slow_scale = stable_points[0] + 2
    else:
        # If no stable point is found, use a default value
        optimal_slow_scale = max(min_slow_scale, max_scales // 3)
    
    # Ensure optimal_slow_scale is at least min_slow_scale
    optimal_slow_scale = max(optimal_slow_scale, min_slow_scale)
    
    # Fast scale is typically 1 (highest frequency)
    optimal_fast_scale = 1
    
    return optimal_slow_scale, optimal_fast_scale


class TwoScaleVarianceEstimator(NoiseRobustEstimator):
    """Estimator for two-scale realized variance in high-frequency financial data.
    
    This class implements the two-scale realized variance estimator, which combines
    two different sampling frequencies to correct for market microstructure noise.
    The estimator is based on the work of Zhang, Mykland, and Aït-Sahalia (2005) and
    provides a robust approach to volatility estimation in the presence of noise.
    
    The two-scale estimator uses a combination of a slow time scale (sparse sampling)
    and a fast time scale (frequent sampling) to separate the true volatility from
    the noise component. This approach provides a consistent estimator of integrated
    variance even when high-frequency data is contaminated with microstructure noise.
    
    Attributes:
        config: Configuration parameters for the estimator
        name: A descriptive name for the estimator
    """
    
    def __init__(self, config: Optional[TwoScaleVarianceConfig] = None, name: str = "TwoScaleVarianceEstimator"):
        """Initialize the two-scale realized variance estimator.
        
        Args:
            config: Configuration parameters for the estimator
            name: A descriptive name for the estimator
        """
        config_to_use = config if config is not None else TwoScaleVarianceConfig()
        super().__init__(config=config_to_use, name=name)
        self._slow_scale: Optional[int] = None
        self._fast_scale: Optional[int] = None
        self._slow_rv: Optional[float] = None
        self._fast_rv: Optional[float] = None
        self._scales_tested: Optional[List[int]] = None
        self._rv_by_scale: Optional[Dict[int, float]] = None
    
    @property
    def config(self) -> TwoScaleVarianceConfig:
        """Get the estimator configuration.
        
        Returns:
            TwoScaleVarianceConfig: The estimator configuration
        """
        return cast(TwoScaleVarianceConfig, self._config)
    
    @config.setter
    def config(self, config: TwoScaleVarianceConfig) -> None:
        """Set the estimator configuration.
        
        Args:
            config: New configuration parameters
            
        Raises:
            TypeError: If config is not a TwoScaleVarianceConfig
        """
        if not isinstance(config, TwoScaleVarianceConfig):
            raise TypeError(f"config must be a TwoScaleVarianceConfig, got {type(config)}")
        
        self._config = config
        self._fitted = False  # Reset fitted state when configuration changes
    
    @property
    def slow_scale(self) -> Optional[int]:
        """Get the slow scale used for estimation.
        
        Returns:
            Optional[int]: The slow scale if the estimator has been fitted,
                          None otherwise
        """
        return self._slow_scale
    
    @property
    def fast_scale(self) -> Optional[int]:
        """Get the fast scale used for estimation.
        
        Returns:
            Optional[int]: The fast scale if the estimator has been fitted,
                          None otherwise
        """
        return self._fast_scale
    
    @property
    def slow_rv(self) -> Optional[float]:
        """Get the realized variance at the slow scale.
        
        Returns:
            Optional[float]: The slow-scale realized variance if the estimator has been fitted,
                            None otherwise
        """
        return self._slow_rv
    
    @property
    def fast_rv(self) -> Optional[float]:
        """Get the realized variance at the fast scale.
        
        Returns:
            Optional[float]: The fast-scale realized variance if the estimator has been fitted,
                            None otherwise
        """
        return self._fast_rv
    
    def _compute_realized_measure(self, 
                                 prices: np.ndarray, 
                                 times: np.ndarray, 
                                 returns: np.ndarray,
                                 **kwargs: Any) -> np.ndarray:
        """Compute the two-scale realized variance from the preprocessed data.
        
        Args:
            prices: Preprocessed price data
            times: Preprocessed time points
            returns: Returns computed from prices
            **kwargs: Additional keyword arguments for computation
            
        Returns:
            np.ndarray: Two-scale realized variance estimate
            
        Raises:
            ValueError: If computation fails
        """
        # Determine scales to use
        if self.config.auto_scale:
            # Automatically determine optimal scales
            try:
                slow_scale, fast_scale = select_optimal_scales(
                    returns, 
                    max_scales=self.config.max_scales,
                    min_slow_scale=5
                )
                
                logger.info(f"Automatically selected scales: slow_scale={slow_scale}, fast_scale={fast_scale}")
            except Exception as e:
                logger.warning(f"Automatic scale selection failed: {str(e)}. Using default scales.")
                slow_scale = 10  # Default slow scale
                fast_scale = 1   # Default fast scale
        else:
            # Use user-specified scales
            if self.config.slow_scale is None:
                raise ValueError("slow_scale must be specified when auto_scale is False")
            
            slow_scale = self.config.slow_scale
            fast_scale = self.config.fast_scale if self.config.fast_scale is not None else 1
        
        # Store scales
        self._slow_scale = slow_scale
        self._fast_scale = fast_scale
        
        # Compute realized variance at different scales for diagnostics
        if self.config.auto_scale or self.config.plot:
            max_scales = min(self.config.max_scales, len(returns) // 10)
            scales = list(range(1, max_scales + 1))
            rv_by_scale = {}
            
            for scale in scales:
                # Skip every scale-th observation
                sampled_returns = np.zeros(len(returns) // scale)
                for i in range(len(sampled_returns)):
                    start_idx = i * scale
                    end_idx = min((i + 1) * scale, len(returns))
                    sampled_returns[i] = np.sum(returns[start_idx:end_idx])
                
                rv_by_scale[scale] = np.mean(sampled_returns**2)
            
            self._scales_tested = scales
            self._rv_by_scale = rv_by_scale
        
        # Compute two-scale realized variance
        try:
            two_scale_rv, slow_rv, fast_rv = compute_twoscale_variance(
                returns, 
                slow_scale, 
                fast_scale,
                bias_correction=self.config.bias_correction
            )
            
            # Store results
            self._slow_rv = slow_rv
            self._fast_rv = fast_rv
            
            # Estimate noise variance if requested
            if self.config.estimate_noise:
                try:
                    noise_var = noise_variance(
                        returns, 
                        method=self.config.noise_method,
                        bias_correction=self.config.bias_correction
                    )
                    
                    self._noise_variance = noise_var
                except Exception as e:
                    logger.warning(f"Noise variance estimation failed: {str(e)}")
                    self._noise_variance = None
            
            # Create result object
            result = TwoScaleVarianceResult(
                model_name=self._name,
                realized_measure=np.array([two_scale_rv]),
                prices=prices,
                times=times,
                returns=returns,
                slow_scale=slow_scale,
                fast_scale=fast_scale,
                slow_rv=slow_rv,
                fast_rv=fast_rv,
                noise_variance=self._noise_variance,
                bias_correction=self.config.bias_correction,
                scales_tested=self._scales_tested,
                rv_by_scale=self._rv_by_scale
            )
            
            # Store result
            self._results = result
            
            # Generate plot if requested
            if self.config.plot and HAS_MATPLOTLIB:
                result.plot()
            
            return np.array([two_scale_rv])
        
        except Exception as e:
            logger.error(f"Two-scale realized variance estimation failed: {str(e)}")
            raise RealizedVolatilityError(
                f"Two-scale realized variance estimation failed: {str(e)}",
                estimator_type="TwoScaleVariance",
                issue=str(e)
            ) from e
    
    def fit(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> TwoScaleVarianceResult:
        """Fit the two-scale realized variance estimator to the provided data.
        
        Args:
            data: The data to fit the estimator to, as a tuple of (prices, times)
            **kwargs: Additional keyword arguments for estimation
            
        Returns:
            TwoScaleVarianceResult: The estimation results
            
        Raises:
            ValueError: If the data is invalid
            RuntimeError: If the estimation fails
        """
        result = super().fit(data, **kwargs)
        return cast(TwoScaleVarianceResult, result)
    
    async def fit_async(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> TwoScaleVarianceResult:
        """Asynchronously fit the two-scale realized variance estimator to the provided data.
        
        Args:
            data: The data to fit the estimator to, as a tuple of (prices, times)
            **kwargs: Additional keyword arguments for estimation
            
        Returns:
            TwoScaleVarianceResult: The estimation results
            
        Raises:
            ValueError: If the data is invalid
            RuntimeError: If the estimation fails
        """
        result = await super().fit_async(data, **kwargs)
        return cast(TwoScaleVarianceResult, result)
    
    def calibrate(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> TwoScaleVarianceConfig:
        """Calibrate the estimator configuration based on the provided data.
        
        This method analyzes the input data and determines optimal configuration
        parameters for the two-scale realized variance estimator.
        
        Args:
            data: The data to calibrate the estimator with, as a tuple of (prices, times)
            **kwargs: Additional keyword arguments for calibration
            
        Returns:
            TwoScaleVarianceConfig: Calibrated configuration
        """
        # Validate input data
        self.validate_data(data)
        prices, times = data
        
        # Preprocess data
        processed_prices, processed_times, returns = self.preprocess_data(prices, times)
        
        # Determine optimal scales
        try:
            slow_scale, fast_scale = select_optimal_scales(
                returns, 
                max_scales=self.config.max_scales,
                min_slow_scale=5
            )
            
            logger.info(f"Calibration selected scales: slow_scale={slow_scale}, fast_scale={fast_scale}")
        except Exception as e:
            logger.warning(f"Optimal scale selection failed: {str(e)}. Using default scales.")
            slow_scale = 10  # Default slow scale
            fast_scale = 1   # Default fast scale
        
        # Create calibrated configuration
        calibrated_config = TwoScaleVarianceConfig(
            slow_scale=slow_scale,
            fast_scale=fast_scale,
            auto_scale=False,  # Set to False since we've already determined optimal scales
            max_scales=self.config.max_scales,
            bias_correction=True,
            estimate_noise=True,
            noise_method='auto',
            plot=self.config.plot
        )
        
        return calibrated_config
    
    def compare_scales(self, 
                      data: Tuple[np.ndarray, np.ndarray], 
                      max_scales: int = 30,
                      figsize: Tuple[int, int] = (10, 6)) -> Optional[Any]:
        """Compare realized variance at different scales.
        
        This method computes realized variance at different scales and visualizes
        the results to help understand the behavior of realized variance across scales.
        
        Args:
            data: The data to analyze, as a tuple of (prices, times)
            max_scales: Maximum number of scales to consider
            figsize: Figure size (width, height) in inches
            
        Returns:
            Optional[Any]: Matplotlib figure if matplotlib is available, None otherwise
            
        Raises:
            ImportError: If matplotlib is not available
            ValueError: If the data is invalid
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib is required for plotting. Please install matplotlib.")
        
        # Validate input data
        self.validate_data(data)
        prices, times = data
        
        # Preprocess data
        processed_prices, processed_times, returns = self.preprocess_data(prices, times)
        
        # Compute realized variance at different scales
        n = len(returns)
        max_scales = min(max_scales, n // 10)  # Ensure max_scales is not too large
        scales = list(range(1, max_scales + 1))
        rv = np.zeros(len(scales))
        
        for i, scale in enumerate(scales):
            # Skip every scale-th observation
            sampled_returns = np.zeros(n // scale)
            for j in range(len(sampled_returns)):
                start_idx = j * scale
                end_idx = min((j + 1) * scale, n)
                sampled_returns[j] = np.sum(returns[start_idx:end_idx])
            
            rv[i] = np.mean(sampled_returns**2)
        
        # Determine optimal scales
        try:
            slow_scale, fast_scale = select_optimal_scales(
                returns, 
                max_scales=max_scales,
                min_slow_scale=5
            )
        except Exception as e:
            logger.warning(f"Optimal scale selection failed: {str(e)}. Using default scales.")
            slow_scale = 10  # Default slow scale
            fast_scale = 1   # Default fast scale
        
        # Compute two-scale realized variance
        two_scale_rv, _, _ = compute_twoscale_variance(
            returns, 
            slow_scale, 
            fast_scale,
            bias_correction=self.config.bias_correction
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot realized variance by scale
        ax.plot(scales, rv, 'o-', label='Realized Variance by Scale')
        
        # Mark the selected scales
        ax.axvline(x=slow_scale, color='r', linestyle='--', 
                  label=f'Slow Scale ({slow_scale})')
        
        ax.axvline(x=fast_scale, color='g', linestyle='--', 
                  label=f'Fast Scale ({fast_scale})')
        
        ax.set_xlabel('Scale')
        ax.set_ylabel('Realized Variance')
        ax.set_title('Two-Scale Realized Variance Estimation')
        ax.legend()
        
        # Add two-scale RV estimate as text
        ax.text(0.05, 0.95, f'Two-Scale RV: {two_scale_rv:.6e}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig


# Register Numba-accelerated functions if available
def _register_numba_functions() -> None:
    """
    Register Numba JIT-compiled functions for two-scale realized variance estimation.
    
    This function is called during module initialization to register
    performance-critical functions for JIT compilation if Numba is available.
    """
    if HAS_NUMBA:
        # The functions are already decorated with @jit, so no additional
        # registration is needed here. This function is kept for consistency
        # with the module structure and potential future enhancements.
        logger.debug("Two-scale realized variance Numba JIT functions registered")
    else:
        logger.info("Numba not available. Two-scale realized variance will use pure NumPy implementations.")


# Initialize the module
_register_numba_functions()
