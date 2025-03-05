'''
Realized Kernel Volatility Estimator

This module implements realized kernel volatility estimators that are robust to market
microstructure noise. It provides both raw and bias-adjusted realized kernel estimates
with support for different kernel types.

The realized kernel estimator is a noise-robust estimator that uses kernel weighting
of autocovariances to estimate integrated variance in the presence of market microstructure
noise. It extends the NoiseRobustEstimator base class and provides specialized functionality
for kernel-based estimation.

Classes:
    KernelEstimator: Base class for realized kernel estimators
    BartlettKernelEstimator: Realized kernel estimator with Bartlett kernel
    ParzenKernelEstimator: Realized kernel estimator with Parzen kernel
    TukeyHanningKernelEstimator: Realized kernel estimator with Tukey-Hanning kernel
    QuadraticKernelEstimator: Realized kernel estimator with Quadratic kernel
    FlatTopKernelEstimator: Realized kernel estimator with Flat-Top kernel
'''
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

from ...core.parameters import ParameterBase, ParameterError, validate_positive, validate_non_negative
from ...core.exceptions import DimensionError, NumericError
from .base import NoiseRobustEstimator, RealizedEstimatorConfig, RealizedEstimatorResult

# Set up module-level logger
logger = logging.getLogger("mfe.models.realized.kernel")

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
    logger.debug("Numba available for kernel estimator acceleration")
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    HAS_NUMBA = False
    logger.info("Numba not available. Kernel estimator will use pure NumPy implementations.")


@dataclass
class KernelEstimatorConfig(RealizedEstimatorConfig):
    """Configuration parameters for realized kernel estimators.
    
    This class extends RealizedEstimatorConfig to provide specialized configuration
    parameters for realized kernel estimators, including kernel type, bandwidth,
    and bias correction options.
    
    Attributes:
        kernel_type: Type of kernel function ('bartlett', 'parzen', 'tukey-hanning', 'quadratic', 'flat-top')
        bandwidth: Bandwidth parameter for kernel function (H)
        bias_correction: Whether to apply bias correction
        jitter_correction: Whether to apply jitter correction
        max_lags: Maximum number of lags to consider (if None, determined by bandwidth)
        auto_bandwidth: Whether to automatically determine optimal bandwidth
        subsampling: Whether to use subsampling for noise reduction
        subsampling_factor: Number of subsamples to use if subsampling is enabled
    """
    
    kernel_type: str = 'bartlett'
    bandwidth: Optional[float] = None
    bias_correction: bool = True
    jitter_correction: bool = False
    max_lags: Optional[int] = None
    auto_bandwidth: bool = True
    subsampling: bool = False
    subsampling_factor: int = 1
    
    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        super().__post_init__()
        
        # Validate kernel_type
        valid_kernels = ['bartlett', 'parzen', 'tukey-hanning', 'tukey', 'hanning', 
                         'quadratic', 'flat-top']
        if self.kernel_type.lower() not in valid_kernels:
            raise ParameterError(
                f"kernel_type must be one of {valid_kernels}, got {self.kernel_type}"
            )
        
        # Validate bandwidth if provided
        if self.bandwidth is not None:
            validate_positive(self.bandwidth, "bandwidth")
        
        # Validate max_lags if provided
        if self.max_lags is not None:
            if not isinstance(self.max_lags, int):
                raise ParameterError(f"max_lags must be an integer, got {type(self.max_lags)}")
            validate_positive(self.max_lags, "max_lags")
        
        # Validate subsampling_factor if subsampling is enabled
        if self.subsampling:
            if not isinstance(self.subsampling_factor, int):
                raise ParameterError(
                    f"subsampling_factor must be an integer, got {type(self.subsampling_factor)}"
                )
            validate_positive(self.subsampling_factor, "subsampling_factor")


@dataclass
class KernelEstimatorResult(RealizedEstimatorResult):
    """Result container for realized kernel estimators.
    
    This class extends RealizedEstimatorResult to provide specialized functionality
    for realized kernel estimator results, including kernel-specific metadata and
    diagnostic information.
    
    Attributes:
        kernel_type: Type of kernel function used
        bandwidth: Bandwidth parameter used
        kernel_weights: Kernel weights used for estimation
        bias_correction: Whether bias correction was applied
        jitter_correction: Whether jitter correction was applied
        max_lags: Maximum number of lags used
        autocovariances: Autocovariances used in estimation
        subsampling: Whether subsampling was used
        subsampling_factor: Number of subsamples used
        raw_measure: Raw realized measure before corrections
        bias_corrected_measure: Bias-corrected realized measure
    """
    
    kernel_type: Optional[str] = None
    bandwidth: Optional[float] = None
    kernel_weights: Optional[np.ndarray] = None
    bias_correction: Optional[bool] = None
    jitter_correction: Optional[bool] = None
    max_lags: Optional[int] = None
    autocovariances: Optional[np.ndarray] = None
    subsampling: Optional[bool] = None
    subsampling_factor: Optional[int] = None
    raw_measure: Optional[float] = None
    bias_corrected_measure: Optional[float] = None
    
    def __post_init__(self) -> None:
        """Validate result object after initialization."""
        super().__post_init__()
        
        # Ensure arrays are NumPy arrays if provided
        if self.kernel_weights is not None and not isinstance(self.kernel_weights, np.ndarray):
            self.kernel_weights = np.array(self.kernel_weights)
        
        if self.autocovariances is not None and not isinstance(self.autocovariances, np.ndarray):
            self.autocovariances = np.array(self.autocovariances)
    
    def summary(self) -> str:
        """Generate a text summary of the realized kernel results.
        
        Returns:
            str: A formatted string containing the realized kernel results summary
        """
        base_summary = super().summary()
        
        kernel_info = f"Kernel Type: {self.kernel_type}\n"
        kernel_info += f"Bandwidth: {self.bandwidth:.2f}\n"
        
        if self.max_lags is not None:
            kernel_info += f"Maximum Lags: {self.max_lags}\n"
        
        if self.bias_correction is not None:
            kernel_info += f"Bias Correction: {'Applied' if self.bias_correction else 'Not Applied'}\n"
        
        if self.jitter_correction is not None:
            kernel_info += f"Jitter Correction: {'Applied' if self.jitter_correction else 'Not Applied'}\n"
        
        if self.subsampling is not None and self.subsampling:
            kernel_info += f"Subsampling: Applied with {self.subsampling_factor} subsamples\n"
        
        if self.raw_measure is not None and self.bias_corrected_measure is not None:
            kernel_info += f"Raw Measure: {self.raw_measure:.6e}\n"
            kernel_info += f"Bias-Corrected Measure: {self.bias_corrected_measure:.6e}\n"
            
            # Calculate bias correction percentage
            if self.raw_measure > 0:
                bias_pct = (self.bias_corrected_measure - self.raw_measure) / self.raw_measure * 100
                kernel_info += f"Bias Correction: {bias_pct:.2f}%\n"
        
        return base_summary + "\nKernel Estimator Details:\n" + kernel_info
    
    def plot_kernel_weights(self, figsize: Tuple[int, int] = (10, 6)) -> Any:
        """Plot the kernel weights used in estimation.
        
        Args:
            figsize: Figure size as (width, height) in inches
            
        Returns:
            matplotlib.figure.Figure: The generated figure
            
        Raises:
            ImportError: If matplotlib is not available
            ValueError: If kernel weights are not available
        """
        if self.kernel_weights is None:
            raise ValueError("Kernel weights are not available")
        
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=figsize)
            
            lags = np.arange(len(self.kernel_weights))
            ax.stem(lags, self.kernel_weights, basefmt=' ')
            
            ax.set_title(f"{self.kernel_type.capitalize()} Kernel Weights (H = {self.bandwidth:.2f})")
            ax.set_xlabel("Lag")
            ax.set_ylabel("Weight")
            ax.grid(True, alpha=0.3)
            
            return fig
        except ImportError:
            raise ImportError("Matplotlib is required for plotting kernel weights")
    
    def plot_autocovariances(self, figsize: Tuple[int, int] = (10, 6)) -> Any:
        """Plot the autocovariances used in estimation.
        
        Args:
            figsize: Figure size as (width, height) in inches
            
        Returns:
            matplotlib.figure.Figure: The generated figure
            
        Raises:
            ImportError: If matplotlib is not available
            ValueError: If autocovariances are not available
        """
        if self.autocovariances is None:
            raise ValueError("Autocovariances are not available")
        
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=figsize)
            
            lags = np.arange(len(self.autocovariances))
            ax.stem(lags, self.autocovariances, basefmt=' ')
            
            ax.set_title("Return Autocovariances")
            ax.set_xlabel("Lag")
            ax.set_ylabel("Autocovariance")
            ax.grid(True, alpha=0.3)
            
            return fig
        except ImportError:
            raise ImportError("Matplotlib is required for plotting autocovariances")


@jit(nopython=True, cache=True)
def _compute_autocovariances_numba(returns: np.ndarray, max_lags: int) -> np.ndarray:
    """Numba-accelerated computation of autocovariances.
    
    Args:
        returns: Array of returns
        max_lags: Maximum number of lags to compute
        
    Returns:
        Array of autocovariances from lag 0 to max_lags
    """
    n = len(returns)
    autocovariances = np.zeros(max_lags + 1)
    
    # Compute mean return
    mean_return = 0.0
    for i in range(n):
        mean_return += returns[i]
    mean_return /= n
    
    # Compute autocovariances
    for h in range(max_lags + 1):
        cov_sum = 0.0
        for t in range(n - h):
            cov_sum += (returns[t] - mean_return) * (returns[t + h] - mean_return)
        autocovariances[h] = cov_sum / n
    
    return autocovariances


def compute_autocovariances(returns: np.ndarray, max_lags: int) -> np.ndarray:
    """Compute autocovariances of returns up to max_lags.
    
    Args:
        returns: Array of returns
        max_lags: Maximum number of lags to compute
        
    Returns:
        Array of autocovariances from lag 0 to max_lags
        
    Raises:
        ValueError: If max_lags is negative or if returns has invalid dimensions
    """
    # Convert to numpy array if not already
    returns = np.asarray(returns)
    
    # Validate inputs
    if returns.ndim != 1:
        raise ValueError("returns must be a 1D array")
    if max_lags < 0:
        raise ValueError("max_lags must be non-negative")
    if max_lags >= len(returns):
        max_lags = len(returns) - 1
        logger.warning(f"max_lags reduced to {max_lags} (length of returns - 1)")
    
    # Use Numba-accelerated implementation if available
    if HAS_NUMBA:
        return _compute_autocovariances_numba(returns, max_lags)
    
    # Pure NumPy implementation
    n = len(returns)
    autocovariances = np.zeros(max_lags + 1)
    
    # Compute mean return
    mean_return = np.mean(returns)
    
    # Compute autocovariances
    for h in range(max_lags + 1):
        autocovariances[h] = np.mean((returns[:(n-h)] - mean_return) * 
                                     (returns[h:] - mean_return))
    
    return autocovariances


@jit(nopython=True, cache=True)
def _compute_kernel_estimate_numba(autocovariances: np.ndarray, 
                                  kernel_weights: np.ndarray) -> float:
    """Numba-accelerated computation of realized kernel estimate.
    
    Args:
        autocovariances: Array of autocovariances
        kernel_weights: Array of kernel weights
        
    Returns:
        Realized kernel estimate
    """
    n_lags = min(len(autocovariances) - 1, len(kernel_weights) - 1)
    
    # Initialize with the variance term (lag 0)
    kernel_estimate = autocovariances[0]
    
    # Add weighted autocovariance terms
    for h in range(1, n_lags + 1):
        kernel_estimate += 2 * kernel_weights[h] * autocovariances[h]
    
    return kernel_estimate


def compute_kernel_estimate(autocovariances: np.ndarray, 
                           kernel_weights: np.ndarray) -> float:
    """Compute realized kernel estimate from autocovariances and kernel weights.
    
    Args:
        autocovariances: Array of autocovariances
        kernel_weights: Array of kernel weights
        
    Returns:
        Realized kernel estimate
        
    Raises:
        ValueError: If inputs have invalid dimensions
    """
    # Convert to numpy arrays if not already
    autocovariances = np.asarray(autocovariances)
    kernel_weights = np.asarray(kernel_weights)
    
    # Validate inputs
    if autocovariances.ndim != 1:
        raise ValueError("autocovariances must be a 1D array")
    if kernel_weights.ndim != 1:
        raise ValueError("kernel_weights must be a 1D array")
    
    # Determine number of lags to use
    n_lags = min(len(autocovariances) - 1, len(kernel_weights) - 1)
    
    # Use Numba-accelerated implementation if available
    if HAS_NUMBA:
        return _compute_kernel_estimate_numba(autocovariances, kernel_weights)
    
    # Pure NumPy implementation
    # Initialize with the variance term (lag 0)
    kernel_estimate = autocovariances[0]
    
    # Add weighted autocovariance terms
    for h in range(1, n_lags + 1):
        kernel_estimate += 2 * kernel_weights[h] * autocovariances[h]
    
    return kernel_estimate


def compute_optimal_bandwidth(returns: np.ndarray, kernel_type: str = 'bartlett') -> float:
    """Compute optimal bandwidth for realized kernel estimator.
    
    Args:
        returns: Array of returns
        kernel_type: Type of kernel function
        
    Returns:
        Optimal bandwidth
        
    Raises:
        ValueError: If kernel_type is not recognized or if returns has invalid dimensions
    """
    from .utils import compute_optimal_bandwidth as utils_compute_optimal_bandwidth
    
    # Use the utility function for optimal bandwidth computation
    return utils_compute_optimal_bandwidth(returns, kernel_type)


def compute_kernel_weights(n_lags: int, kernel_type: str = 'bartlett', 
                         bandwidth: Optional[float] = None) -> np.ndarray:
    """Compute kernel weights for realized kernel estimator.
    
    Args:
        n_lags: Number of lags (including lag 0)
        kernel_type: Type of kernel function
        bandwidth: Bandwidth parameter (if None, defaults to sqrt(n_lags))
        
    Returns:
        Array of kernel weights
        
    Raises:
        ValueError: If kernel_type is not recognized or if n_lags is not positive
    """
    from .utils import compute_kernel_weights as utils_compute_kernel_weights
    
    # Use the utility function for kernel weights computation
    return utils_compute_kernel_weights(n_lags, kernel_type, bandwidth)


class KernelEstimator(NoiseRobustEstimator):
    """Base class for realized kernel volatility estimators.
    
    This class implements the realized kernel estimator, which uses kernel weighting
    of autocovariances to estimate integrated variance in the presence of market
    microstructure noise. It extends the NoiseRobustEstimator base class and provides
    specialized functionality for kernel-based estimation.
    
    The realized kernel estimator is defined as:
    
    RK = γ₀ + 2 * Σ_{h=1}^H k(h/H) * γₕ
    
    where γₕ is the h-th order autocovariance of returns, k(·) is the kernel function,
    and H is the bandwidth parameter.
    
    Attributes:
        config: Configuration parameters for the estimator
        kernel_type: Type of kernel function
        bandwidth: Bandwidth parameter
        max_lags: Maximum number of lags to consider
        kernel_weights: Kernel weights used for estimation
        autocovariances: Autocovariances used in estimation
    """
    
    def __init__(self, config: Optional[KernelEstimatorConfig] = None, 
                name: str = "KernelEstimator"):
        """Initialize the realized kernel estimator.
        
        Args:
            config: Configuration parameters for the estimator
            name: A descriptive name for the estimator
        """
        # Use default config if not provided
        if config is None:
            config = KernelEstimatorConfig()
        
        # Initialize base class
        super().__init__(config=config, name=name)
        
        # Store kernel-specific attributes
        self._kernel_type = config.kernel_type
        self._bandwidth = config.bandwidth
        self._max_lags = config.max_lags
        self._bias_correction = config.bias_correction
        self._jitter_correction = config.jitter_correction
        self._auto_bandwidth = config.auto_bandwidth
        self._subsampling = config.subsampling
        self._subsampling_factor = config.subsampling_factor
        
        # Initialize result attributes
        self._kernel_weights: Optional[np.ndarray] = None
        self._autocovariances: Optional[np.ndarray] = None
        self._raw_measure: Optional[float] = None
        self._bias_corrected_measure: Optional[float] = None
    
    @property
    def kernel_type(self) -> str:
        """Get the kernel type.
        
        Returns:
            str: The kernel type
        """
        return self._kernel_type
    
    @property
    def bandwidth(self) -> Optional[float]:
        """Get the bandwidth parameter.
        
        Returns:
            Optional[float]: The bandwidth parameter if set, None otherwise
        """
        return self._bandwidth
    
    @property
    def max_lags(self) -> Optional[int]:
        """Get the maximum number of lags.
        
        Returns:
            Optional[int]: The maximum number of lags if set, None otherwise
        """
        return self._max_lags
    
    @property
    def kernel_weights(self) -> Optional[np.ndarray]:
        """Get the kernel weights.
        
        Returns:
            Optional[np.ndarray]: The kernel weights if computed, None otherwise
        """
        return self._kernel_weights
    
    @property
    def autocovariances(self) -> Optional[np.ndarray]:
        """Get the autocovariances.
        
        Returns:
            Optional[np.ndarray]: The autocovariances if computed, None otherwise
        """
        return self._autocovariances
    
    @property
    def raw_measure(self) -> Optional[float]:
        """Get the raw realized measure.
        
        Returns:
            Optional[float]: The raw realized measure if computed, None otherwise
        """
        return self._raw_measure
    
    @property
    def bias_corrected_measure(self) -> Optional[float]:
        """Get the bias-corrected realized measure.
        
        Returns:
            Optional[float]: The bias-corrected realized measure if computed, None otherwise
        """
        return self._bias_corrected_measure
    
    def _compute_realized_measure(self, 
                                 prices: np.ndarray, 
                                 times: np.ndarray, 
                                 returns: np.ndarray,
                                 **kwargs: Any) -> np.ndarray:
        """Compute the realized kernel measure from the preprocessed data.
        
        This method implements the core computation of the realized kernel estimator,
        including bandwidth determination, autocovariance computation, kernel weighting,
        and bias correction.
        
        Args:
            prices: Preprocessed price data
            times: Preprocessed time points
            returns: Returns computed from prices
            **kwargs: Additional keyword arguments for computation
            
        Returns:
            np.ndarray: Realized kernel measure
            
        Raises:
            ValueError: If computation fails
        """
        # Get number of observations
        n = len(returns)
        
        # Determine bandwidth if not provided or if auto_bandwidth is True
        if self._bandwidth is None or self._auto_bandwidth:
            self._bandwidth = compute_optimal_bandwidth(returns, self._kernel_type)
            logger.info(f"Using optimal bandwidth: {self._bandwidth:.2f}")
        
        # Determine maximum lags if not provided
        if self._max_lags is None:
            # Default to bandwidth or n/5, whichever is smaller
            self._max_lags = min(int(np.ceil(self._bandwidth)), n // 5)
            logger.info(f"Using maximum lags: {self._max_lags}")
        
        # Ensure max_lags is not too large
        if self._max_lags >= n:
            self._max_lags = n - 1
            logger.warning(f"Maximum lags reduced to {self._max_lags} (length of returns - 1)")
        
        # Compute autocovariances
        self._autocovariances = compute_autocovariances(returns, self._max_lags)
        
        # Compute kernel weights
        self._kernel_weights = compute_kernel_weights(
            self._max_lags + 1, self._kernel_type, self._bandwidth
        )
        
        # Compute raw kernel estimate
        self._raw_measure = compute_kernel_estimate(
            self._autocovariances, self._kernel_weights
        )
        
        # Apply bias correction if enabled
        if self._bias_correction:
            # Estimate noise variance
            noise_var = self.estimate_noise_variance(returns)
            self._noise_variance = noise_var
            
            # Compute bias correction
            # The bias is approximately 2 * n * noise_variance for realized kernel
            bias_correction = 2 * n * noise_var
            
            # Apply correction
            self._bias_corrected_measure = self._raw_measure - bias_correction
            
            # Ensure non-negative result
            if self._bias_corrected_measure < 0:
                logger.warning(
                    "Bias correction resulted in negative measure. "
                    "Setting to a small positive value."
                )
                self._bias_corrected_measure = 1e-10
            
            # Use bias-corrected measure as the final result
            realized_measure = self._bias_corrected_measure
        else:
            # Use raw measure as the final result
            realized_measure = self._raw_measure
            self._bias_corrected_measure = self._raw_measure
        
        # Apply jitter correction if enabled
        if self._jitter_correction:
            # Jitter correction is not implemented yet
            # This is a placeholder for future implementation
            logger.warning("Jitter correction is not implemented yet. Ignoring.")
        
        # Apply subsampling if enabled
        if self._subsampling and self._subsampling_factor > 1:
            from .utils import compute_subsampled_measure
            
            # Compute subsampled measure
            subsampled_measure = compute_subsampled_measure(
                returns, self._subsampling_factor
            )
            
            # Use subsampled measure as the final result
            realized_measure = subsampled_measure
        
        return np.array([realized_measure])
    
    async def _compute_realized_measure_async(self, 
                                            prices: np.ndarray, 
                                            times: np.ndarray, 
                                            returns: np.ndarray,
                                            **kwargs: Any) -> np.ndarray:
        """Asynchronously compute the realized kernel measure from the preprocessed data.
        
        This method provides an asynchronous implementation of the realized kernel
        estimator computation, allowing for non-blocking estimation in UI contexts.
        
        Args:
            prices: Preprocessed price data
            times: Preprocessed time points
            returns: Returns computed from prices
            **kwargs: Additional keyword arguments for computation
            
        Returns:
            np.ndarray: Realized kernel measure
            
        Raises:
            ValueError: If computation fails
        """
        # This is a simple asynchronous wrapper around the synchronous implementation
        # In a real implementation, this could be optimized for truly asynchronous execution
        return self._compute_realized_measure(prices, times, returns, **kwargs)
    
    async def fit_async(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> KernelEstimatorResult:
        """Asynchronously fit the realized kernel estimator to the provided data.
        
        This method provides an asynchronous interface to the fit method,
        allowing for non-blocking estimation in UI contexts.
        
        Args:
            data: The data to fit the estimator to, as a tuple of (prices, times)
            **kwargs: Additional keyword arguments for estimation
            
        Returns:
            KernelEstimatorResult: The estimation results
            
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
            # Compute realized measure asynchronously
            realized_measure = await self._compute_realized_measure_async(
                processed_prices, processed_times, returns, **kwargs
            )
            
            # Update instance state
            self._realized_measure = realized_measure
            self._fitted = True
            
            # Create result object
            result = KernelEstimatorResult(
                model_name=self._name,
                realized_measure=realized_measure,
                prices=prices,
                times=times,
                sampling_frequency=self._config.sampling_frequency,
                kernel_type=self._kernel_type,
                bandwidth=self._bandwidth,
                bias_correction=self._bias_correction,
                jitter_correction=self._jitter_correction,
                max_lags=self._max_lags,
                autocovariances=self._autocovariances,
                subsampling=self._subsampling,
                subsampling_factor=self._subsampling_factor,
                raw_measure=self._raw_measure,
                bias_corrected_measure=self._bias_corrected_measure,
                noise_variance=self._noise_variance,
                computation_time=time.time() - start_time,
                config=self._config.to_dict(),
                kernel_weights=self._kernel_weights
            )
            
            # Store result
            self._results = result
            
            return result
        
        except Exception as e:
            logger.error(f"Estimation failed: {str(e)}")
            raise RuntimeError(f"Realized kernel estimation failed: {str(e)}") from e
    
    def fit(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> KernelEstimatorResult:
        """Fit the realized kernel estimator to the provided data.
        
        This method validates the input data, preprocesses it according to the
        estimator configuration, and then calls the _compute_realized_measure
        method to perform the actual estimation.
        
        Args:
            data: The data to fit the estimator to, as a tuple of (prices, times)
            **kwargs: Additional keyword arguments for estimation
            
        Returns:
            KernelEstimatorResult: The estimation results
            
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
            result = KernelEstimatorResult(
                model_name=self._name,
                realized_measure=realized_measure,
                prices=prices,
                times=times,
                sampling_frequency=self._config.sampling_frequency,
                kernel_type=self._kernel_type,
                bandwidth=self._bandwidth,
                bias_correction=self._bias_correction,
                jitter_correction=self._jitter_correction,
                max_lags=self._max_lags,
                autocovariances=self._autocovariances,
                subsampling=self._subsampling,
                subsampling_factor=self._subsampling_factor,
                raw_measure=self._raw_measure,
                bias_corrected_measure=self._bias_corrected_measure,
                noise_variance=self._noise_variance,
                computation_time=time.time() - start_time,
                config=self._config.to_dict(),
                kernel_weights=self._kernel_weights
            )
            
            # Store result
            self._results = result
            
            return result
        
        except Exception as e:
            logger.error(f"Estimation failed: {str(e)}")
            raise RuntimeError(f"Realized kernel estimation failed: {str(e)}") from e
    
    def calibrate(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> KernelEstimatorConfig:
        """Calibrate the estimator configuration based on the provided data.
        
        This method analyzes the input data and determines optimal configuration
        parameters for the estimator, such as bandwidth and maximum lags.
        
        Args:
            data: The data to calibrate the estimator with, as a tuple of (prices, times)
            **kwargs: Additional keyword arguments for calibration
            
        Returns:
            KernelEstimatorConfig: Calibrated configuration
        """
        # Validate input data
        self.validate_data(data)
        prices, times = data
        
        # Preprocess data
        processed_prices, processed_times, returns = self.preprocess_data(prices, times)
        
        # Determine optimal bandwidth
        optimal_bandwidth = compute_optimal_bandwidth(returns, self._kernel_type)
        
        # Determine optimal maximum lags
        n = len(returns)
        optimal_max_lags = min(int(np.ceil(optimal_bandwidth)), n // 5)
        
        # Create calibrated configuration
        calibrated_config = KernelEstimatorConfig(
            kernel_type=self._kernel_type,
            bandwidth=optimal_bandwidth,
            max_lags=optimal_max_lags,
            bias_correction=self._bias_correction,
            jitter_correction=self._jitter_correction,
            auto_bandwidth=False,  # Set to False since we've already determined the optimal bandwidth
            subsampling=self._subsampling,
            subsampling_factor=self._subsampling_factor,
            sampling_frequency=self._config.sampling_frequency,
            annualize=self._config.annualize,
            annualization_factor=self._config.annualization_factor,
            return_type=self._config.return_type,
            apply_noise_correction=self._config.apply_noise_correction,
            time_unit=self._config.time_unit
        )
        
        return calibrated_config
    
    def plot_kernel_weights(self, figsize: Tuple[int, int] = (10, 6)) -> Any:
        """Plot the kernel weights used in estimation.
        
        Args:
            figsize: Figure size as (width, height) in inches
            
        Returns:
            matplotlib.figure.Figure: The generated figure
            
        Raises:
            ImportError: If matplotlib is not available
            RuntimeError: If the estimator has not been fitted
        """
        if not self._fitted:
            raise RuntimeError("Estimator has not been fitted. Call fit() first.")
        
        if self._kernel_weights is None:
            raise RuntimeError("Kernel weights are not available")
        
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=figsize)
            
            lags = np.arange(len(self._kernel_weights))
            ax.stem(lags, self._kernel_weights, basefmt=' ')
            
            ax.set_title(f"{self._kernel_type.capitalize()} Kernel Weights (H = {self._bandwidth:.2f})")
            ax.set_xlabel("Lag")
            ax.set_ylabel("Weight")
            ax.grid(True, alpha=0.3)
            
            return fig
        except ImportError:
            raise ImportError("Matplotlib is required for plotting kernel weights")
    
    def plot_autocovariances(self, figsize: Tuple[int, int] = (10, 6)) -> Any:
        """Plot the autocovariances used in estimation.
        
        Args:
            figsize: Figure size as (width, height) in inches
            
        Returns:
            matplotlib.figure.Figure: The generated figure
            
        Raises:
            ImportError: If matplotlib is not available
            RuntimeError: If the estimator has not been fitted
        """
        if not self._fitted:
            raise RuntimeError("Estimator has not been fitted. Call fit() first.")
        
        if self._autocovariances is None:
            raise RuntimeError("Autocovariances are not available")
        
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=figsize)
            
            lags = np.arange(len(self._autocovariances))
            ax.stem(lags, self._autocovariances, basefmt=' ')
            
            ax.set_title("Return Autocovariances")
            ax.set_xlabel("Lag")
            ax.set_ylabel("Autocovariance")
            ax.grid(True, alpha=0.3)
            
            return fig
        except ImportError:
            raise ImportError("Matplotlib is required for plotting autocovariances")


class BartlettKernelEstimator(KernelEstimator):
    """Realized kernel estimator with Bartlett kernel.
    
    This class implements the realized kernel estimator with the Bartlett kernel,
    which is a linear kernel that decreases from 1 at lag 0 to 0 at lag H+1.
    
    The Bartlett kernel is defined as:
    
    k(x) = 1 - x  for 0 ≤ x ≤ 1
    k(x) = 0      for x > 1
    
    where x = h/H, h is the lag, and H is the bandwidth parameter.
    """
    
    def __init__(self, config: Optional[KernelEstimatorConfig] = None):
        """Initialize the Bartlett kernel estimator.
        
        Args:
            config: Configuration parameters for the estimator
        """
        # Create default config if not provided
        if config is None:
            config = KernelEstimatorConfig(kernel_type='bartlett')
        else:
            # Ensure kernel_type is set to 'bartlett'
            config.kernel_type = 'bartlett'
        
        # Initialize base class
        super().__init__(config=config, name="BartlettKernelEstimator")


class ParzenKernelEstimator(KernelEstimator):
    """Realized kernel estimator with Parzen kernel.
    
    This class implements the realized kernel estimator with the Parzen kernel,
    which is a smooth kernel that provides better bias-variance tradeoff than
    the Bartlett kernel.
    
    The Parzen kernel is defined as:
    
    k(x) = 1 - 6x^2 + 6x^3  for 0 ≤ x ≤ 0.5
    k(x) = 2(1 - x)^3       for 0.5 < x ≤ 1
    k(x) = 0                for x > 1
    
    where x = h/H, h is the lag, and H is the bandwidth parameter.
    """
    
    def __init__(self, config: Optional[KernelEstimatorConfig] = None):
        """Initialize the Parzen kernel estimator.
        
        Args:
            config: Configuration parameters for the estimator
        """
        # Create default config if not provided
        if config is None:
            config = KernelEstimatorConfig(kernel_type='parzen')
        else:
            # Ensure kernel_type is set to 'parzen'
            config.kernel_type = 'parzen'
        
        # Initialize base class
        super().__init__(config=config, name="ParzenKernelEstimator")


class TukeyHanningKernelEstimator(KernelEstimator):
    """Realized kernel estimator with Tukey-Hanning kernel.
    
    This class implements the realized kernel estimator with the Tukey-Hanning kernel,
    which is a smooth kernel based on the cosine function.
    
    The Tukey-Hanning kernel is defined as:
    
    k(x) = 0.5 * (1 + cos(πx))  for 0 ≤ x ≤ 1
    k(x) = 0                    for x > 1
    
    where x = h/H, h is the lag, and H is the bandwidth parameter.
    """
    
    def __init__(self, config: Optional[KernelEstimatorConfig] = None):
        """Initialize the Tukey-Hanning kernel estimator.
        
        Args:
            config: Configuration parameters for the estimator
        """
        # Create default config if not provided
        if config is None:
            config = KernelEstimatorConfig(kernel_type='tukey-hanning')
        else:
            # Ensure kernel_type is set to 'tukey-hanning'
            config.kernel_type = 'tukey-hanning'
        
        # Initialize base class
        super().__init__(config=config, name="TukeyHanningKernelEstimator")


class QuadraticKernelEstimator(KernelEstimator):
    """Realized kernel estimator with Quadratic kernel.
    
    This class implements the realized kernel estimator with the Quadratic kernel,
    which is a smooth kernel that provides good bias-variance tradeoff.
    
    The Quadratic kernel is defined as:
    
    k(x) = (1 - x^2)^2  for 0 ≤ x ≤ 1
    k(x) = 0            for x > 1
    
    where x = h/H, h is the lag, and H is the bandwidth parameter.
    """
    
    def __init__(self, config: Optional[KernelEstimatorConfig] = None):
        """Initialize the Quadratic kernel estimator.
        
        Args:
            config: Configuration parameters for the estimator
        """
        # Create default config if not provided
        if config is None:
            config = KernelEstimatorConfig(kernel_type='quadratic')
        else:
            # Ensure kernel_type is set to 'quadratic'
            config.kernel_type = 'quadratic'
        
        # Initialize base class
        super().__init__(config=config, name="QuadraticKernelEstimator")


class FlatTopKernelEstimator(KernelEstimator):
    """Realized kernel estimator with Flat-Top kernel.
    
    This class implements the realized kernel estimator with the Flat-Top kernel,
    which is designed to minimize asymptotic bias.
    
    The Flat-Top kernel is defined as:
    
    k(x) = 1                    for 0 ≤ x ≤ 0.1
    k(x) = 1.1 - x              for 0.1 < x ≤ 1.1
    k(x) = 0                    for x > 1.1
    
    where x = h/H, h is the lag, and H is the bandwidth parameter.
    
    Note: This is a simplified implementation of the Flat-Top kernel.
    """
    
    def __init__(self, config: Optional[KernelEstimatorConfig] = None):
        """Initialize the Flat-Top kernel estimator.
        
        Args:
            config: Configuration parameters for the estimator
        """
        # Create default config if not provided
        if config is None:
            config = KernelEstimatorConfig(kernel_type='flat-top')
        else:
            # Ensure kernel_type is set to 'flat-top'
            config.kernel_type = 'flat-top'
        
        # Initialize base class
        super().__init__(config=config, name="FlatTopKernelEstimator")
    
    def _compute_realized_measure(self, 
                                 prices: np.ndarray, 
                                 times: np.ndarray, 
                                 returns: np.ndarray,
                                 **kwargs: Any) -> np.ndarray:
        """Compute the realized kernel measure with Flat-Top kernel.
        
        This method overrides the base implementation to provide a specialized
        implementation for the Flat-Top kernel, which requires custom handling.
        
        Args:
            prices: Preprocessed price data
            times: Preprocessed time points
            returns: Returns computed from prices
            **kwargs: Additional keyword arguments for computation
            
        Returns:
            np.ndarray: Realized kernel measure
            
        Raises:
            ValueError: If computation fails
        """
        # For now, use the base implementation
        # In a real implementation, this would be customized for the Flat-Top kernel
        logger.warning(
            "Specialized implementation for Flat-Top kernel is not available. "
            "Using generic kernel implementation."
        )
        return super()._compute_realized_measure(prices, times, returns, **kwargs)


# Create a dictionary mapping kernel types to estimator classes
KERNEL_ESTIMATOR_CLASSES = {
    'bartlett': BartlettKernelEstimator,
    'parzen': ParzenKernelEstimator,
    'tukey-hanning': TukeyHanningKernelEstimator,
    'tukey': TukeyHanningKernelEstimator,
    'hanning': TukeyHanningKernelEstimator,
    'quadratic': QuadraticKernelEstimator,
    'flat-top': FlatTopKernelEstimator
}


def create_kernel_estimator(kernel_type: str, 
                          config: Optional[KernelEstimatorConfig] = None) -> KernelEstimator:
    """Create a kernel estimator of the specified type.
    
    Args:
        kernel_type: Type of kernel function
        config: Configuration parameters for the estimator
        
    Returns:
        KernelEstimator: A kernel estimator of the specified type
        
    Raises:
        ValueError: If kernel_type is not recognized
    """
    # Validate kernel_type
    kernel_type_lower = kernel_type.lower()
    if kernel_type_lower not in KERNEL_ESTIMATOR_CLASSES:
        valid_kernels = list(KERNEL_ESTIMATOR_CLASSES.keys())
        raise ValueError(f"Unrecognized kernel type: {kernel_type}. "
                         f"Supported types are {valid_kernels}.")
    
    # Create default config if not provided
    if config is None:
        config = KernelEstimatorConfig(kernel_type=kernel_type_lower)
    else:
        # Ensure kernel_type is set correctly
        config.kernel_type = kernel_type_lower
    
    # Create and return the appropriate estimator
    estimator_class = KERNEL_ESTIMATOR_CLASSES[kernel_type_lower]
    return estimator_class(config=config)