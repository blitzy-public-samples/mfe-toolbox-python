# mfe/models/realized/multivariate_kernel.py

"""
Multivariate Realized Kernel Estimator

This module implements the multivariate realized kernel estimator for covariance matrix
estimation from high-frequency financial data. The estimator is designed to handle
market microstructure noise while preserving the positive semi-definiteness of the
covariance matrix, making it suitable for portfolio optimization and risk management.

The implementation supports various kernel types, bandwidth selection methods, and
data synchronization techniques for handling asynchronous trading. Performance-critical
calculations are accelerated using Numba's JIT compilation for efficient processing
of large high-frequency datasets.

Classes:
    MultivariateKernelConfig: Configuration parameters for multivariate kernel estimator
    MultivariateKernelResult: Result container for multivariate kernel estimation
    MultivariateKernelEstimator: Base class for multivariate realized kernel estimators
    BartlettMultivariateKernelEstimator: Multivariate kernel estimator with Bartlett kernel
    ParzenMultivariateKernelEstimator: Multivariate kernel estimator with Parzen kernel
    TukeyHanningMultivariateKernelEstimator: Multivariate kernel estimator with Tukey-Hanning kernel
"""

import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, cast, overload

import numpy as np
import pandas as pd
from scipy import stats, optimize

from ...core.exceptions import ParameterError, DimensionError, NumericError
from ...utils.matrix_ops import ensure_symmetric, is_positive_definite, nearest_positive_definite
from .base import MultivariateRealizedEstimator, RealizedEstimatorConfig, RealizedEstimatorResult
from .kernel import KernelEstimatorConfig, compute_kernel_weights, compute_optimal_bandwidth
from .covariance import _ensure_psd, _synchronize_prices

# Set up module-level logger
logger = logging.getLogger("mfe.models.realized.multivariate_kernel")

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
    logger.debug("Numba available for multivariate kernel estimator acceleration")
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    HAS_NUMBA = False
    logger.info("Numba not available. Multivariate kernel estimator will use pure NumPy implementations.")


@dataclass
class MultivariateKernelConfig(KernelEstimatorConfig):
    """Configuration parameters for multivariate realized kernel estimators.

    This class extends KernelEstimatorConfig to provide specialized configuration
    parameters for multivariate realized kernel estimators, including options for
    ensuring positive semi-definiteness and data synchronization.

    Attributes:
        kernel_type: Type of kernel function ('bartlett', 'parzen', 'tukey-hanning', etc.)
        bandwidth: Bandwidth parameter for kernel function (H)
        bias_correction: Whether to apply bias correction
        jitter_correction: Whether to apply jitter correction
        max_lags: Maximum number of lags to consider (if None, determined by bandwidth)
        auto_bandwidth: Whether to automatically determine optimal bandwidth
        subsampling: Whether to use subsampling for noise reduction
        subsampling_factor: Number of subsamples to use if subsampling is enabled
        ensure_psd: Whether to ensure the resulting covariance matrix is positive semi-definite
        psd_method: Method to use for ensuring positive semi-definiteness ('nearest' or 'eigenvalue')
        eigenvalue_threshold: Threshold for eigenvalue adjustment when psd_method='eigenvalue'
        synchronize_data: Whether to synchronize asynchronous price data
        synchronization_method: Method to use for synchronization ('previous' or 'linear')
        return_correlation: Whether to also compute and return the correlation matrix
        refresh_time: Whether to use refresh time synchronization for asynchronous data
    """

    ensure_psd: bool = True
    psd_method: str = "nearest"
    eigenvalue_threshold: float = 1e-8
    synchronize_data: bool = True
    synchronization_method: str = "previous"
    return_correlation: bool = True
    refresh_time: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        super().__post_init__()

        # Validate psd_method
        if self.psd_method not in ["nearest", "eigenvalue"]:
            raise ParameterError(
                f"psd_method must be 'nearest' or 'eigenvalue', got {self.psd_method}"
            )

        # Validate eigenvalue_threshold
        if self.eigenvalue_threshold <= 0:
            raise ParameterError(
                f"eigenvalue_threshold must be positive, got {self.eigenvalue_threshold}"
            )

        # Validate synchronization_method
        if self.synchronization_method not in ["previous", "linear"]:
            raise ParameterError(
                f"synchronization_method must be 'previous' or 'linear', got {self.synchronization_method}"
            )


@dataclass
class MultivariateKernelResult(RealizedEstimatorResult):
    """Result container for multivariate realized kernel estimators.

    This class extends RealizedEstimatorResult to provide specialized functionality
    for multivariate realized kernel estimator results, including correlation matrices,
    eigenvalue information, and kernel-specific metadata.

    Attributes:
        realized_measure: Computed realized covariance matrix
        kernel_type: Type of kernel function used
        bandwidth: Bandwidth parameter used
        kernel_weights: Kernel weights used for estimation
        bias_correction: Whether bias correction was applied
        jitter_correction: Whether jitter correction was applied
        max_lags: Maximum number of lags used
        autocovariances: Autocovariances used in estimation
        correlation_matrix: Correlation matrix derived from the realized covariance
        eigenvalues: Eigenvalues of the realized covariance matrix
        is_psd: Whether the realized covariance matrix is positive semi-definite
        psd_adjustment: Adjustment made to ensure positive semi-definiteness
        synchronization_info: Information about data synchronization
        raw_measure: Raw realized measure before corrections
        bias_corrected_measure: Bias-corrected realized measure
    """

    kernel_type: Optional[str] = None
    bandwidth: Optional[float] = None
    kernel_weights: Optional[np.ndarray] = None
    bias_correction: Optional[bool] = None
    jitter_correction: Optional[bool] = None
    max_lags: Optional[int] = None
    autocovariances: Optional[List[np.ndarray]] = None
    correlation_matrix: Optional[np.ndarray] = None
    eigenvalues: Optional[np.ndarray] = None
    is_psd: Optional[bool] = None
    psd_adjustment: Optional[str] = None
    synchronization_info: Optional[Dict[str, Any]] = None
    raw_measure: Optional[np.ndarray] = None
    bias_corrected_measure: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        """Validate result object after initialization."""
        super().__post_init__()

        # Ensure arrays are NumPy arrays if provided
        if self.kernel_weights is not None and not isinstance(self.kernel_weights, np.ndarray):
            self.kernel_weights = np.array(self.kernel_weights)

        if self.correlation_matrix is not None and not isinstance(self.correlation_matrix, np.ndarray):
            self.correlation_matrix = np.array(self.correlation_matrix)

        if self.eigenvalues is not None and not isinstance(self.eigenvalues, np.ndarray):
            self.eigenvalues = np.array(self.eigenvalues)

        if self.raw_measure is not None and not isinstance(self.raw_measure, np.ndarray):
            self.raw_measure = np.array(self.raw_measure)

        if self.bias_corrected_measure is not None and not isinstance(self.bias_corrected_measure, np.ndarray):
            self.bias_corrected_measure = np.array(self.bias_corrected_measure)

    def summary(self) -> str:
        """Generate a text summary of the multivariate realized kernel results.

        Returns:
            str: A formatted string containing the multivariate realized kernel results summary
        """
        base_summary = super().summary()

        additional_info = ""

        # Add kernel information
        if self.kernel_type is not None:
            additional_info += f"Kernel Type: {self.kernel_type}\n"

        if self.bandwidth is not None:
            additional_info += f"Bandwidth: {self.bandwidth:.2f}\n"

        if self.max_lags is not None:
            additional_info += f"Maximum Lags: {self.max_lags}\n"

        if self.bias_correction is not None:
            additional_info += f"Bias Correction: {'Applied' if self.bias_correction else 'Not Applied'}\n"

        if self.jitter_correction is not None:
            additional_info += f"Jitter Correction: {'Applied' if self.jitter_correction else 'Not Applied'}\n"

        # Add eigenvalue information
        if self.eigenvalues is not None:
            additional_info += "Eigenvalue Information:\n"
            additional_info += f"  Minimum Eigenvalue: {np.min(self.eigenvalues):.6e}\n"
            additional_info += f"  Maximum Eigenvalue: {np.max(self.eigenvalues):.6e}\n"
            additional_info += f"  Condition Number: {np.max(self.eigenvalues) / max(np.min(self.eigenvalues), 1e-15):.6e}\n"

        # Add PSD information
        if self.is_psd is not None:
            additional_info += f"Positive Semi-Definite: {'Yes' if self.is_psd else 'No'}\n"
            if self.psd_adjustment is not None:
                additional_info += f"PSD Adjustment: {self.psd_adjustment}\n"

        # Add synchronization information
        if self.synchronization_info is not None:
            additional_info += "Synchronization Information:\n"
            for key, value in self.synchronization_info.items():
                additional_info += f"  {key}: {value}\n"

        if additional_info:
            additional_info = "Additional Information:\n" + additional_info + "\n"

        return base_summary + additional_info

    def get_correlation_matrix(self) -> np.ndarray:
        """Get the correlation matrix derived from the realized covariance.

        Returns:
            np.ndarray: Correlation matrix

        Raises:
            ValueError: If correlation matrix is not available
        """
        if self.correlation_matrix is None:
            raise ValueError("Correlation matrix is not available")

        return self.correlation_matrix

    def plot_correlation_heatmap(self, **kwargs: Any) -> Any:
        """Plot a heatmap of the correlation matrix.

        Args:
            **kwargs: Additional keyword arguments for plotting

        Returns:
            Any: Plot object

        Raises:
            ValueError: If correlation matrix is not available
            ImportError: If matplotlib and seaborn are not installed
        """
        if self.correlation_matrix is None:
            raise ValueError("Correlation matrix is not available")

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            raise ImportError("Matplotlib and seaborn are required for plotting")

        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 8)))

        # Create heatmap
        sns.heatmap(
            self.correlation_matrix,
            annot=kwargs.get('annot', True),
            cmap=kwargs.get('cmap', "coolwarm"),
            vmin=kwargs.get('vmin', -1),
            vmax=kwargs.get('vmax', 1),
            ax=ax
        )

        ax.set_title(kwargs.get('title', "Realized Correlation Matrix"))

        return fig

    def plot_covariance_heatmap(self, **kwargs: Any) -> Any:
        """Plot a heatmap of the realized covariance matrix.

        Args:
            **kwargs: Additional keyword arguments for plotting

        Returns:
            Any: Plot object

        Raises:
            ImportError: If matplotlib and seaborn are not installed
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            raise ImportError("Matplotlib and seaborn are required for plotting")

        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 8)))

        # Create heatmap
        sns.heatmap(
            self.realized_measure,
            annot=kwargs.get('annot', True),
            cmap=kwargs.get('cmap', "viridis"),
            ax=ax
        )

        ax.set_title(kwargs.get('title', "Realized Covariance Matrix"))

        return fig

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


@jit(nopython=True, cache=True)
def _compute_autocovariance_matrices_numba(returns: np.ndarray, max_lags: int) -> List[np.ndarray]:
    """Numba-accelerated computation of autocovariance matrices.

    Args:
        returns: Matrix of returns (n_obs, n_assets)
        max_lags: Maximum number of lags to compute

    Returns:
        List of autocovariance matrices from lag 0 to max_lags
    """
    n_obs, n_assets = returns.shape

    # Initialize list of autocovariance matrices
    # Note: Numba doesn't support list comprehensions with np.zeros inside JIT functions
    autocovariances = []
    for _ in range(max_lags + 1):
        autocovariances.append(np.zeros((n_assets, n_assets)))

    # Compute mean returns
    mean_returns = np.zeros(n_assets)
    for i in range(n_assets):
        for t in range(n_obs):
            mean_returns[i] += returns[t, i]
        mean_returns[i] /= n_obs

    # Compute autocovariance matrices
    for h in range(max_lags + 1):
        for t in range(n_obs - h):
            for i in range(n_assets):
                for j in range(n_assets):
                    autocovariances[h][i, j] += (returns[t, i] - mean_returns[i]) * \
                        (returns[t + h, j] - mean_returns[j])

        # Normalize by number of observations
        for i in range(n_assets):
            for j in range(n_assets):
                autocovariances[h][i, j] /= n_obs

    return autocovariances


def compute_autocovariance_matrices(returns: np.ndarray, max_lags: int) -> List[np.ndarray]:
    """Compute autocovariance matrices of returns up to max_lags.

    Args:
        returns: Matrix of returns (n_obs, n_assets)
        max_lags: Maximum number of lags to compute

    Returns:
        List of autocovariance matrices from lag 0 to max_lags

    Raises:
        ValueError: If max_lags is negative or if returns has invalid dimensions
    """
    # Convert to numpy array if not already
    returns = np.asarray(returns)

    # Validate inputs
    if returns.ndim != 2:
        raise ValueError("returns must be a 2D array (n_obs, n_assets)")
    if max_lags < 0:
        raise ValueError("max_lags must be non-negative")

    n_obs, n_assets = returns.shape

    if max_lags >= n_obs:
        max_lags = n_obs - 1
        logger.warning(f"max_lags reduced to {max_lags} (length of returns - 1)")

    # Use Numba-accelerated implementation if available
    if HAS_NUMBA:
        return _compute_autocovariance_matrices_numba(returns, max_lags)

    # Pure NumPy implementation
    # Initialize list of autocovariance matrices
    autocovariances = [np.zeros((n_assets, n_assets)) for _ in range(max_lags + 1)]

    # Compute mean returns
    mean_returns = np.mean(returns, axis=0)

    # Compute autocovariance matrices
    for h in range(max_lags + 1):
        # Compute cross-products of demeaned returns
        demeaned_t = returns[:-h] - mean_returns if h > 0 else returns - mean_returns
        demeaned_t_plus_h = returns[h:] - mean_returns if h > 0 else returns - mean_returns

        # Compute autocovariance matrix for lag h
        for t in range(len(demeaned_t)):
            autocovariances[h] += np.outer(demeaned_t[t], demeaned_t_plus_h[t])

        # Normalize by number of observations
        autocovariances[h] /= n_obs

    return autocovariances


@jit(nopython=True, cache=True)
def _compute_multivariate_kernel_estimate_numba(autocovariances: List[np.ndarray],
                                                kernel_weights: np.ndarray) -> np.ndarray:
    """Numba-accelerated computation of multivariate realized kernel estimate.

    Args:
        autocovariances: List of autocovariance matrices
        kernel_weights: Array of kernel weights

    Returns:
        np.ndarray: Realized kernel covariance matrix
    """
    n_lags = min(len(autocovariances) - 1, len(kernel_weights) - 1)
    n_assets = autocovariances[0].shape[0]

    # Initialize with the variance term (lag 0)
    kernel_estimate = autocovariances[0].copy()

    # Add weighted autocovariance terms
    for h in range(1, n_lags + 1):
        weight = kernel_weights[h]
        for i in range(n_assets):
            for j in range(n_assets):
                # Add h-th lag autocovariance and its transpose (for symmetry)
                kernel_estimate[i, j] += weight * (autocovariances[h][i, j] + autocovariances[h][j, i])

    return kernel_estimate


def compute_multivariate_kernel_estimate(autocovariances: List[np.ndarray],
                                         kernel_weights: np.ndarray) -> np.ndarray:
    """Compute multivariate realized kernel estimate from autocovariances and kernel weights.

    Args:
        autocovariances: List of autocovariance matrices
        kernel_weights: Array of kernel weights

    Returns:
        np.ndarray: Realized kernel covariance matrix

    Raises:
        ValueError: If inputs have invalid dimensions
    """
    # Validate inputs
    if not isinstance(autocovariances, list) or not autocovariances:
        raise ValueError("autocovariances must be a non-empty list of matrices")

    for i, acov in enumerate(autocovariances):
        if not isinstance(acov, np.ndarray) or acov.ndim != 2 or acov.shape[0] != acov.shape[1]:
            raise ValueError(f"autocovariances[{i}] must be a square matrix")

    kernel_weights = np.asarray(kernel_weights)
    if kernel_weights.ndim != 1:
        raise ValueError("kernel_weights must be a 1D array")

    # Determine number of lags to use
    n_lags = min(len(autocovariances) - 1, len(kernel_weights) - 1)
    n_assets = autocovariances[0].shape[0]

    # Use Numba-accelerated implementation if available
    if HAS_NUMBA:
        return _compute_multivariate_kernel_estimate_numba(autocovariances, kernel_weights)

    # Pure NumPy implementation
    # Initialize with the variance term (lag 0)
    kernel_estimate = autocovariances[0].copy()

    # Add weighted autocovariance terms
    for h in range(1, n_lags + 1):
        # Add h-th lag autocovariance and its transpose (for symmetry)
        kernel_estimate += kernel_weights[h] * (autocovariances[h] + autocovariances[h].T)

    return kernel_estimate


def refresh_time_synchronization(prices: np.ndarray, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Synchronize asynchronous price data using the refresh time algorithm.

    The refresh time algorithm identifies points in time where all assets have
    been traded at least once since the previous refresh time, creating a
    synchronized time grid for multivariate analysis.

    Args:
        prices: Matrix of prices (n_obs, n_assets)
        times: Array of observation times

    Returns:
        Tuple containing:
        - Synchronized prices
        - Synchronized times
        - Synchronization information
    """
    n_obs, n_assets = prices.shape

    # Check if data is already synchronized (no NaN values)
    if not np.isnan(prices).any():
        return prices, times, {"synchronized": False, "reason": "Data already synchronized"}

    # Create a mask of valid (non-NaN) prices
    valid_mask = ~np.isnan(prices)

    # Count NaN values before synchronization
    nan_count_before = np.isnan(prices).sum()

    # Find refresh times
    refresh_times = []
    last_observed = np.full(n_assets, -1)  # Last observation index for each asset

    # Add first time point
    refresh_times.append(0)

    # Update last observed for first time point
    for j in range(n_assets):
        if valid_mask[0, j]:
            last_observed[j] = 0

    # Find subsequent refresh times
    for i in range(1, n_obs):
        # Update last observed for current time point
        for j in range(n_assets):
            if valid_mask[i, j]:
                last_observed[j] = i

        # Check if all assets have been observed since last refresh time
        if np.all(last_observed >= 0):
            refresh_times.append(i)

            # Reset last observed
            last_observed = np.full(n_assets, -1)

            # Update last observed for current time point
            for j in range(n_assets):
                if valid_mask[i, j]:
                    last_observed[j] = i

    # Create synchronized arrays
    n_refresh = len(refresh_times)
    sync_prices = np.zeros((n_refresh, n_assets))
    sync_times = np.zeros(n_refresh)

    # Fill synchronized arrays
    for i, rt_idx in enumerate(refresh_times):
        sync_times[i] = times[rt_idx]

        # For each asset, use the most recent valid price
        for j in range(n_assets):
            # Find the most recent valid price for this asset
            valid_indices = np.where(valid_mask[:rt_idx+1, j])[0]

            if len(valid_indices) > 0:
                # Use the most recent valid price
                sync_prices[i, j] = prices[valid_indices[-1], j]
            else:
                # No valid price found, use NaN
                sync_prices[i, j] = np.nan

    # Count NaN values after synchronization
    nan_count_after = np.isnan(sync_prices).sum()

    # Create synchronization info
    sync_info = {
        "synchronized": True,
        "method": "refresh_time",
        "original_observations": n_obs,
        "synchronized_observations": n_refresh,
        "reduction_percent": 100 * (1 - n_refresh / n_obs),
        "nan_count_before": int(nan_count_before),
        "nan_count_after": int(nan_count_after),
        "nan_reduction_percent": 100 * (1 - nan_count_after / max(nan_count_before, 1))
    }

    return sync_prices, sync_times, sync_info


class MultivariateKernelEstimator(MultivariateRealizedEstimator):
    """Multivariate realized kernel estimator for high-frequency financial data.

    This class implements the multivariate realized kernel estimator, which computes
    a noise-robust covariance matrix from high-frequency returns using kernel-weighted
    autocovariance matrices. The estimator is designed to handle market microstructure
    noise while preserving the positive semi-definiteness of the covariance matrix.

    The estimator supports various kernel types, bandwidth selection methods, and
    data synchronization techniques for handling asynchronous trading.

    Attributes:
        config: Configuration parameters for the estimator
        _n_assets: Number of assets in the data
        _realized_measure: Computed realized covariance matrix
        _kernel_weights: Kernel weights used for estimation
        _autocovariances: Autocovariance matrices used in estimation
        _correlation_matrix: Correlation matrix derived from the covariance matrix
    """

    def __init__(self, config: Optional[MultivariateKernelConfig] = None,
                 name: str = "MultivariateKernelEstimator"):
        """Initialize the multivariate realized kernel estimator.

        Args:
            config: Configuration parameters for the estimator
            name: A descriptive name for the estimator
        """
        config = config if config is not None else MultivariateKernelConfig()
        super().__init__(config=config, name=name)

        # Store kernel-specific attributes
        self._kernel_type = config.kernel_type
        self._bandwidth = config.bandwidth
        self._max_lags = config.max_lags
        self._bias_correction = config.bias_correction
        self._jitter_correction = config.jitter_correction
        self._auto_bandwidth = config.auto_bandwidth

        # Initialize result attributes
        self._kernel_weights: Optional[np.ndarray] = None
        self._autocovariances: Optional[List[np.ndarray]] = None
        self._raw_measure: Optional[np.ndarray] = None
        self._bias_corrected_measure: Optional[np.ndarray] = None
        self._correlation_matrix: Optional[np.ndarray] = None
        self._eigenvalues: Optional[np.ndarray] = None
        self._is_psd: Optional[bool] = None
        self._psd_adjustment: Optional[str] = None
        self._synchronization_info: Optional[Dict[str, Any]] = None

    @property
    def config(self) -> MultivariateKernelConfig:
        """Get the estimator configuration.

        Returns:
            MultivariateKernelConfig: The estimator configuration
        """
        return cast(MultivariateKernelConfig, self._config)

    @config.setter
    def config(self, config: MultivariateKernelConfig) -> None:
        """Set the estimator configuration.

        Args:
            config: New configuration parameters

        Raises:
            TypeError: If config is not a MultivariateKernelConfig
        """
        if not isinstance(config, MultivariateKernelConfig):
            raise TypeError(f"config must be a MultivariateKernelConfig, got {type(config)}")

        self._config = config
        self._kernel_type = config.kernel_type
        self._bandwidth = config.bandwidth
        self._max_lags = config.max_lags
        self._bias_correction = config.bias_correction
        self._jitter_correction = config.jitter_correction
        self._auto_bandwidth = config.auto_bandwidth

        self._fitted = False  # Reset fitted state when configuration changes

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
    def autocovariances(self) -> Optional[List[np.ndarray]]:
        """Get the autocovariance matrices.

        Returns:
            Optional[List[np.ndarray]]: The autocovariance matrices if computed, None otherwise
        """
        return self._autocovariances

    @property
    def raw_measure(self) -> Optional[np.ndarray]:
        """Get the raw realized measure.

        Returns:
            Optional[np.ndarray]: The raw realized measure if computed, None otherwise
        """
        return self._raw_measure

    @property
    def bias_corrected_measure(self) -> Optional[np.ndarray]:
        """Get the bias-corrected realized measure.

        Returns:
            Optional[np.ndarray]: The bias-corrected realized measure if computed, None otherwise
        """
        return self._bias_corrected_measure

    @property
    def correlation_matrix(self) -> Optional[np.ndarray]:
        """Get the correlation matrix.

        Returns:
            Optional[np.ndarray]: The correlation matrix if computed, None otherwise
        """
        return self._correlation_matrix

    def _compute_realized_measure(self,
                                  prices: np.ndarray,
                                  times: np.ndarray,
                                  returns: np.ndarray,
                                  **kwargs: Any) -> np.ndarray:
        """Compute the multivariate realized kernel measure from the preprocessed data.

        This method implements the core computation of the multivariate realized kernel
        estimator, including bandwidth determination, autocovariance computation,
        kernel weighting, and bias correction.

        Args:
            prices: Preprocessed price data (n_obs, n_assets)
            times: Preprocessed time points
            returns: Returns computed from prices (n_obs-1, n_assets)
            **kwargs: Additional keyword arguments for computation

        Returns:
            np.ndarray: Realized kernel covariance matrix (n_assets, n_assets)

        Raises:
            ValueError: If computation fails
        """
        n_obs, n_assets = returns.shape

        # Store number of assets
        self._n_assets = n_assets

        # Determine bandwidth if not provided or if auto_bandwidth is True
        if self._bandwidth is None or self._auto_bandwidth:
            # For multivariate case, compute optimal bandwidth for each asset
            # and take the average
            bandwidths = []
            for j in range(n_assets):
                asset_returns = returns[:, j]
                # Skip assets with all NaN returns
                if np.all(np.isnan(asset_returns)):
                    continue
                # Replace remaining NaNs with zeros
                asset_returns = np.nan_to_num(asset_returns, nan=0.0)
                bandwidths.append(compute_optimal_bandwidth(asset_returns, self._kernel_type))

            if bandwidths:
                self._bandwidth = np.mean(bandwidths)
            else:
                # Default to sqrt(n) if no valid bandwidths
                self._bandwidth = np.sqrt(n_obs)

            logger.info(f"Using optimal bandwidth: {self._bandwidth:.2f}")

        # Determine maximum lags if not provided
        if self._max_lags is None:
            # Default to bandwidth or n/5, whichever is smaller
            self._max_lags = min(int(np.ceil(self._bandwidth)), n_obs // 5)
            logger.info(f"Using maximum lags: {self._max_lags}")

        # Ensure max_lags is not too large
        if self._max_lags >= n_obs:
            self._max_lags = n_obs - 1
            logger.warning(f"Maximum lags reduced to {self._max_lags} (length of returns - 1)")

        # Replace NaNs with zeros for autocovariance computation
        # This is a simple approach; more sophisticated methods could be used
        returns_no_nan = np.nan_to_num(returns, nan=0.0)

        # Compute autocovariance matrices
        self._autocovariances = compute_autocovariance_matrices(returns_no_nan, self._max_lags)

        # Compute kernel weights
        self._kernel_weights = compute_kernel_weights(
            self._max_lags + 1, self._kernel_type, self._bandwidth
        )

        # Compute raw kernel estimate
        self._raw_measure = compute_multivariate_kernel_estimate(
            self._autocovariances, self._kernel_weights
        )

        # Ensure the matrix is symmetric (to handle numerical precision issues)
        self._raw_measure = ensure_symmetric(self._raw_measure)

        # Apply bias correction if enabled
        if self._bias_correction:
            # Estimate noise variance for each asset
            noise_vars = np.zeros(n_assets)
            for j in range(n_assets):
                asset_returns = returns[:, j]
                # Skip assets with all NaN returns
                if np.all(np.isnan(asset_returns)):
                    continue
                # Replace remaining NaNs with zeros
                asset_returns = np.nan_to_num(asset_returns, nan=0.0)
                from .utils import noise_variance
                noise_vars[j] = noise_variance(asset_returns)

            # Compute bias correction matrix
            # The bias is approximately 2 * n * noise_variance for realized kernel
            bias_correction = np.zeros((n_assets, n_assets))
            for i in range(n_assets):
                for j in range(n_assets):
                    # For off-diagonal elements, use geometric mean of noise variances
                    if i != j:
                        bias_correction[i, j] = 2 * n_obs * np.sqrt(noise_vars[i] * noise_vars[j])
                    else:
                        bias_correction[i, j] = 2 * n_obs * noise_vars[i]

            # Apply correction
            self._bias_corrected_measure = self._raw_measure - bias_correction

            # Ensure the matrix is symmetric
            self._bias_corrected_measure = ensure_symmetric(self._bias_corrected_measure)

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

        # Ensure positive semi-definiteness if requested
        if self.config.ensure_psd:
            realized_measure, is_psd, psd_adjustment, eigenvalues = _ensure_psd(
                realized_measure,
                method=self.config.psd_method,
                threshold=self.config.eigenvalue_threshold
            )

            # Store PSD information for result
            self._is_psd = is_psd
            self._psd_adjustment = psd_adjustment
            self._eigenvalues = eigenvalues
        else:
            # Compute eigenvalues for information
            self._eigenvalues = np.linalg.eigvalsh(realized_measure)
            self._is_psd = np.all(self._eigenvalues > -1e-10)
            self._psd_adjustment = "No adjustment applied"

        # Compute correlation matrix if requested
        if self.config.return_correlation:
            # Extract standard deviations from diagonal
            std_devs = np.sqrt(np.diag(realized_measure))

            # Avoid division by zero
            std_devs = np.maximum(std_devs, np.finfo(float).eps)

            # Compute correlation matrix
            corr = np.zeros_like(realized_measure)
            for i in range(n_assets):
                for j in range(n_assets):
                    corr[i, j] = realized_measure[i, j] / (std_devs[i] * std_devs[j])

            # Ensure diagonal is exactly 1
            np.fill_diagonal(corr, 1.0)

            # Ensure the matrix is symmetric
            corr = ensure_symmetric(corr)

            # Store correlation matrix for result
            self._correlation_matrix = corr
        else:
            self._correlation_matrix = None

        return realized_measure

    async def _compute_realized_measure_async(self,
                                              prices: np.ndarray,
                                              times: np.ndarray,
                                              returns: np.ndarray,
                                              **kwargs: Any) -> np.ndarray:
        """Asynchronously compute the multivariate realized kernel measure.

        This method provides an asynchronous implementation of the multivariate
        realized kernel estimator computation, allowing for non-blocking estimation
        in UI contexts.

        Args:
            prices: Preprocessed price data (n_obs, n_assets)
            times: Preprocessed time points
            returns: Returns computed from prices (n_obs-1, n_assets)
            **kwargs: Additional keyword arguments for computation

        Returns:
            np.ndarray: Realized kernel covariance matrix (n_assets, n_assets)

        Raises:
            ValueError: If computation fails
        """
        # This is a simple asynchronous wrapper around the synchronous implementation
        # In a real implementation, this could be optimized for truly asynchronous execution
        import asyncio

        # Run the computation in a separate thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,  # Use default executor
            lambda: self._compute_realized_measure(prices, times, returns, **kwargs)
        )

        return result

    async def fit_async(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> MultivariateKernelResult:
        """Asynchronously fit the multivariate realized kernel estimator to the provided data.

        This method provides an asynchronous interface to the fit method,
        allowing for non-blocking estimation in UI contexts.

        Args:
            data: The data to fit the estimator to, as a tuple of (prices, times)
            **kwargs: Additional keyword arguments for estimation

        Returns:
            MultivariateKernelResult: The estimation results

        Raises:
            ValueError: If the data is invalid
            RuntimeError: If the estimation fails
        """
        import time
        start_time = time.time()

        # Validate input data
        self.validate_data(data)
        prices, times = data

        # Synchronize data if requested and needed
        if self.config.synchronize_data and np.isnan(prices).any():
            if self.config.refresh_time:
                # Use refresh time synchronization
                prices, times, sync_info = refresh_time_synchronization(prices, times)
            else:
                # Use standard synchronization
                prices, times, sync_info = _synchronize_prices(
                    prices,
                    times,
                    method=self.config.synchronization_method
                )
            self._synchronization_info = sync_info
        else:
            self._synchronization_info = {"synchronized": False,
                                          "reason": "Synchronization not requested or not needed"}

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
            result = MultivariateKernelResult(
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
                subsampling=self._config.use_subsampling,
                noise_correction=self._config.apply_noise_correction,
                annualization_factor=self._config.annualization_factor if self._config.annualize else None,
                returns=returns,
                computation_time=time.time() - start_time,
                config=self._config.to_dict(),
                correlation_matrix=self._correlation_matrix,
                eigenvalues=self._eigenvalues,
                is_psd=self._is_psd,
                psd_adjustment=self._psd_adjustment,
                synchronization_info=self._synchronization_info,
                raw_measure=self._raw_measure,
                bias_corrected_measure=self._bias_corrected_measure,
                kernel_weights=self._kernel_weights
            )

            # Store result
            self._results = result

            return result

        except Exception as e:
            logger.error(f"Estimation failed: {str(e)}")
            raise RuntimeError(f"Multivariate realized kernel estimation failed: {str(e)}") from e

    def fit(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> MultivariateKernelResult:
        """Fit the multivariate realized kernel estimator to the provided data.

        This method validates the input data, preprocesses it according to the
        estimator configuration, and then computes the multivariate realized kernel.

        Args:
            data: The data to fit the estimator to, as a tuple of (prices, times)
            **kwargs: Additional keyword arguments for estimation

        Returns:
            MultivariateKernelResult: The estimation results

        Raises:
            ValueError: If the data is invalid
            RuntimeError: If the estimation fails
        """
        import time
        start_time = time.time()

        # Validate input data
        self.validate_data(data)
        prices, times = data

        # Synchronize data if requested and needed
        if self.config.synchronize_data and np.isnan(prices).any():
            if self.config.refresh_time:
                # Use refresh time synchronization
                prices, times, sync_info = refresh_time_synchronization(prices, times)
            else:
                # Use standard synchronization
                prices, times, sync_info = _synchronize_prices(
                    prices,
                    times,
                    method=self.config.synchronization_method
                )
            self._synchronization_info = sync_info
        else:
            self._synchronization_info = {"synchronized": False,
                                          "reason": "Synchronization not requested or not needed"}

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
            result = MultivariateKernelResult(
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
                subsampling=self._config.use_subsampling,
                noise_correction=self._config.apply_noise_correction,
                annualization_factor=self._config.annualization_factor if self._config.annualize else None,
                returns=returns,
                computation_time=time.time() - start_time,
                config=self._config.to_dict(),
                correlation_matrix=self._correlation_matrix,
                eigenvalues=self._eigenvalues,
                is_psd=self._is_psd,
                psd_adjustment=self._psd_adjustment,
                synchronization_info=self._synchronization_info,
                raw_measure=self._raw_measure,
                bias_corrected_measure=self._bias_corrected_measure,
                kernel_weights=self._kernel_weights
            )

            # Store result
            self._results = result

            return result

        except Exception as e:
            logger.error(f"Estimation failed: {str(e)}")
            raise RuntimeError(f"Multivariate realized kernel estimation failed: {str(e)}") from e

    def calibrate(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> MultivariateKernelConfig:
        """Calibrate the estimator configuration based on the provided data.

        This method analyzes the input data and determines optimal configuration
        parameters for the estimator, such as bandwidth, synchronization method,
        and PSD approach.

        Args:
            data: The data to calibrate the estimator with, as a tuple of (prices, times)
            **kwargs: Additional keyword arguments for calibration

        Returns:
            MultivariateKernelConfig: Calibrated configuration

        Raises:
            ValueError: If the data is invalid
        """
        # Validate input data
        self.validate_data(data)
        prices, times = data

        # Create a copy of the current configuration
        calibrated_config = MultivariateKernelConfig(**self.config.to_dict())

        # Check if data needs synchronization
        if np.isnan(prices).any():
            calibrated_config.synchronize_data = True

            # Try both synchronization methods and refresh time
            # Compare the number of observations and NaN values

            # Standard synchronization with previous method
            prices_prev, times_prev, _ = _synchronize_prices(prices, times, method="previous")

            # Standard synchronization with linear method
            prices_linear, times_linear, _ = _synchronize_prices(prices, times, method="linear")

            # Refresh time synchronization
            prices_refresh, times_refresh, _ = refresh_time_synchronization(prices, times)

            # Count NaN values after each synchronization
            nan_prev = np.isnan(prices_prev).sum()
            nan_linear = np.isnan(prices_linear).sum()
            nan_refresh = np.isnan(prices_refresh).sum()

            # Compare number of observations
            n_prev = len(times_prev)
            n_linear = len(times_linear)
            n_refresh = len(times_refresh)

            # Choose method based on NaN count and observation count
            # Prefer method with fewer NaNs and more observations
            if nan_refresh == 0 and n_refresh >= min(n_prev, n_linear) * 0.5:
                # Refresh time is best if it eliminates all NaNs and keeps at least 50% of observations
                calibrated_config.refresh_time = True
            elif nan_prev <= nan_linear:
                # Previous method is better than linear
                calibrated_config.refresh_time = False
                calibrated_config.synchronization_method = "previous"
            else:
                # Linear method is better
                calibrated_config.refresh_time = False
                calibrated_config.synchronization_method = "linear"
        else:
            calibrated_config.synchronize_data = False

        # Determine optimal bandwidth
        # For multivariate case, compute optimal bandwidth for each asset
        # and take the average
        n_obs, n_assets = prices.shape
        bandwidths = []

        # Preprocess data to get returns
        processed_prices, processed_times, returns = self.preprocess_data(prices, times)

        for j in range(n_assets):
            asset_returns = returns[:, j]
            # Skip assets with all NaN returns
            if np.all(np.isnan(asset_returns)):
                continue
            # Replace remaining NaNs with zeros
            asset_returns = np.nan_to_num(asset_returns, nan=0.0)
            bandwidths.append(compute_optimal_bandwidth(asset_returns, self._kernel_type))

        if bandwidths:
            calibrated_config.bandwidth = np.mean(bandwidths)
            calibrated_config.auto_bandwidth = False
        else:
            # Default to sqrt(n) if no valid bandwidths
            calibrated_config.bandwidth = np.sqrt(n_obs)
            calibrated_config.auto_bandwidth = False

        # Determine if PSD enforcement is needed
        # Compute a simple realized covariance to check
        returns_no_nan = np.nan_to_num(returns, nan=0.0)
        rcov = returns_no_nan.T @ returns_no_nan

        # Check if already PSD
        is_psd = is_positive_definite(rcov)
        calibrated_config.ensure_psd = not is_psd

        # Choose PSD method based on eigenvalue distribution
        if not is_psd:
            eigvals = np.linalg.eigvalsh(rcov)
            min_eigval = np.min(eigvals)

            # If minimum eigenvalue is close to zero, use eigenvalue adjustment
            # Otherwise, use nearest PSD
            if min_eigval > -0.01 * np.max(eigvals):
                calibrated_config.psd_method = "eigenvalue"
                calibrated_config.eigenvalue_threshold = max(abs(min_eigval) * 1.1, 1e-8)
            else:
                calibrated_config.psd_method = "nearest"

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

    def plot_covariance_heatmap(self, **kwargs: Any) -> Any:
        """Plot a heatmap of the realized covariance matrix.

        Args:
            **kwargs: Additional keyword arguments for plotting

        Returns:
            Any: Plot object

        Raises:
            RuntimeError: If the estimator has not been fitted
            ImportError: If matplotlib and seaborn are not installed
        """
        if not self._fitted or self._realized_measure is None:
            raise RuntimeError("Estimator has not been fitted. Call fit() first.")

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            raise ImportError("Matplotlib and seaborn are required for plotting")

        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 8)))

        # Create heatmap
        sns.heatmap(
            self._realized_measure,
            annot=kwargs.get('annot', True),
            cmap=kwargs.get('cmap', "viridis"),
            ax=ax
        )

        ax.set_title(kwargs.get('title', f"Realized Covariance Matrix from {self._name}"))

        return fig

    def plot_correlation_heatmap(self, **kwargs: Any) -> Any:
        """Plot a heatmap of the correlation matrix.

        Args:
            **kwargs: Additional keyword arguments for plotting

        Returns:
            Any: Plot object

        Raises:
            RuntimeError: If the estimator has not been fitted or correlation matrix is not available
            ImportError: If matplotlib and seaborn are not installed
        """
        if not self._fitted:
            raise RuntimeError("Estimator has not been fitted. Call fit() first.")

        if self._correlation_matrix is None:
            raise RuntimeError("Correlation matrix is not available. Set return_correlation=True in config.")

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            raise ImportError("Matplotlib and seaborn are required for plotting")

        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 8)))

        # Create heatmap
        sns.heatmap(
            self._correlation_matrix,
            annot=kwargs.get('annot', True),
            cmap=kwargs.get('cmap', "coolwarm"),
            vmin=kwargs.get('vmin', -1),
            vmax=kwargs.get('vmax', 1),
            ax=ax
        )

        ax.set_title(kwargs.get('title', f"Realized Correlation Matrix from {self._name}"))

        return fig

    def plot_eigenvalue_distribution(self, **kwargs: Any) -> Any:
        """Plot the distribution of eigenvalues of the realized covariance matrix.

        Args:
            **kwargs: Additional keyword arguments for plotting

        Returns:
            Any: Plot object

        Raises:
            RuntimeError: If the estimator has not been fitted
            ImportError: If matplotlib is not installed
        """
        if not self._fitted or self._eigenvalues is None:
            raise RuntimeError("Estimator has not been fitted. Call fit() first.")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib is required for plotting")

        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))

        # Plot eigenvalues
        ax.bar(range(1, len(self._eigenvalues) + 1), np.sort(self._eigenvalues)[::-1])

        ax.set_title(kwargs.get('title', f"Eigenvalue Distribution of Realized Covariance Matrix from {self._name}"))
        ax.set_xlabel("Eigenvalue Index")
        ax.set_ylabel("Eigenvalue")

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

        return fig


class BartlettMultivariateKernelEstimator(MultivariateKernelEstimator):
    """Multivariate realized kernel estimator with Bartlett kernel.

    This class implements the multivariate realized kernel estimator with the Bartlett
    kernel, which is a linear kernel that decreases from 1 at lag 0 to 0 at lag H+1.

    The Bartlett kernel is defined as:

    k(x) = 1 - x  for 0 ≤ x ≤ 1
    k(x) = 0      for x > 1

    where x = h/H, h is the lag, and H is the bandwidth parameter.
    """

    def __init__(self, config: Optional[MultivariateKernelConfig] = None):
        """Initialize the Bartlett multivariate kernel estimator.

        Args:
            config: Configuration parameters for the estimator
        """
        # Create default config if not provided
        if config is None:
            config = MultivariateKernelConfig(kernel_type='bartlett')
        else:
            # Ensure kernel_type is set to 'bartlett'
            config.kernel_type = 'bartlett'

        # Initialize base class
        super().__init__(config=config, name="BartlettMultivariateKernelEstimator")


class ParzenMultivariateKernelEstimator(MultivariateKernelEstimator):
    """Multivariate realized kernel estimator with Parzen kernel.

    This class implements the multivariate realized kernel estimator with the Parzen
    kernel, which is a smooth kernel that provides better bias-variance tradeoff than
    the Bartlett kernel.

    The Parzen kernel is defined as:

    k(x) = 1 - 6x^2 + 6x^3  for 0 ≤ x ≤ 0.5
    k(x) = 2(1 - x)^3       for 0.5 < x ≤ 1
    k(x) = 0                for x > 1

    where x = h/H, h is the lag, and H is the bandwidth parameter.
    """

    def __init__(self, config: Optional[MultivariateKernelConfig] = None):
        """Initialize the Parzen multivariate kernel estimator.

        Args:
            config: Configuration parameters for the estimator
        """
        # Create default config if not provided
        if config is None:
            config = MultivariateKernelConfig(kernel_type='parzen')
        else:
            # Ensure kernel_type is set to 'parzen'
            config.kernel_type = 'parzen'

        # Initialize base class
        super().__init__(config=config, name="ParzenMultivariateKernelEstimator")


class TukeyHanningMultivariateKernelEstimator(MultivariateKernelEstimator):
    """Multivariate realized kernel estimator with Tukey-Hanning kernel.

    This class implements the multivariate realized kernel estimator with the
    Tukey-Hanning kernel, which is a smooth kernel based on the cosine function.

    The Tukey-Hanning kernel is defined as:

    k(x) = 0.5 * (1 + cos(πx))  for 0 ≤ x ≤ 1
    k(x) = 0                    for x > 1

    where x = h/H, h is the lag, and H is the bandwidth parameter.
    """

    def __init__(self, config: Optional[MultivariateKernelConfig] = None):
        """Initialize the Tukey-Hanning multivariate kernel estimator.

        Args:
            config: Configuration parameters for the estimator
        """
        # Create default config if not provided
        if config is None:
            config = MultivariateKernelConfig(kernel_type='tukey-hanning')
        else:
            # Ensure kernel_type is set to 'tukey-hanning'
            config.kernel_type = 'tukey-hanning'

        # Initialize base class
        super().__init__(config=config, name="TukeyHanningMultivariateKernelEstimator")


# Create a dictionary mapping kernel types to estimator classes
MULTIVARIATE_KERNEL_ESTIMATOR_CLASSES = {
    'bartlett': BartlettMultivariateKernelEstimator,
    'parzen': ParzenMultivariateKernelEstimator,
    'tukey-hanning': TukeyHanningMultivariateKernelEstimator,
    'tukey': TukeyHanningMultivariateKernelEstimator,
    'hanning': TukeyHanningMultivariateKernelEstimator
}


def create_multivariate_kernel_estimator(kernel_type: str,
                                         config: Optional[MultivariateKernelConfig] = None) -> MultivariateKernelEstimator:
    """Create a multivariate kernel estimator of the specified type.

    Args:
        kernel_type: Type of kernel function
        config: Configuration parameters for the estimator

    Returns:
        MultivariateKernelEstimator: A multivariate kernel estimator of the specified type

    Raises:
        ValueError: If kernel_type is not recognized
    """
    # Validate kernel_type
    kernel_type_lower = kernel_type.lower()
    if kernel_type_lower not in MULTIVARIATE_KERNEL_ESTIMATOR_CLASSES:
        valid_kernels = list(MULTIVARIATE_KERNEL_ESTIMATOR_CLASSES.keys())
        raise ValueError(f"Unrecognized kernel type: {kernel_type}. "
                         f"Supported types are {valid_kernels}.")

    # Create default config if not provided
    if config is None:
        config = MultivariateKernelConfig(kernel_type=kernel_type_lower)
    else:
        # Ensure kernel_type is set correctly
        config.kernel_type = kernel_type_lower

    # Create and return the appropriate estimator
    estimator_class = MULTIVARIATE_KERNEL_ESTIMATOR_CLASSES[kernel_type_lower]
    return estimator_class(config=config)
