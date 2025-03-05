'''
Hayashi-Yoshida Covariance Estimator

This module implements the Hayashi-Yoshida estimator for computing quadratic covariation
between asynchronous asset price series. This estimator is uniquely capable of handling
non-synchronous trading times without requiring data alignment through interpolation.

The Hayashi-Yoshida estimator computes covariance by summing the products of returns
from overlapping time intervals, providing a consistent estimator of integrated covariance
even when observations occur at different times for different assets. This makes it
particularly valuable for high-frequency financial data where trading times are rarely
synchronized across assets.

The implementation leverages Numba's JIT compilation for performance optimization and
supports both NumPy arrays and Pandas DataFrames with datetime indices. It also provides
support for K-lead-and-lag variations with asynchronous processing for improved
robustness to microstructure noise.

References:
    Hayashi, T., & Yoshida, N. (2005). On covariance estimation of non-synchronously
    observed diffusion processes. Bernoulli, 11(2), 359-379.
'''

import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, cast, overload, Callable

import numpy as np
import pandas as pd
from scipy import stats

from ...core.exceptions import ParameterError, DimensionError, NumericError
from ...utils.matrix_ops import ensure_symmetric
from ..realized.base import MultivariateRealizedEstimator, RealizedEstimatorConfig, RealizedEstimatorResult
from ._numba_core import _hayashi_yoshida_core

# Set up module-level logger
logger = logging.getLogger("mfe.models.realized.hayashi_yoshida")

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
    logger.debug("Numba available for Hayashi-Yoshida acceleration")
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    HAS_NUMBA = False
    logger.info("Numba not available. Hayashi-Yoshida estimator will use pure NumPy implementation.")


@dataclass
class HayashiYoshidaConfig(RealizedEstimatorConfig):
    """Configuration parameters for Hayashi-Yoshida estimator.
    
    This class extends RealizedEstimatorConfig to provide specialized configuration
    parameters for the Hayashi-Yoshida estimator.
    
    Attributes:
        lead_lag: Number of leads and lags to include (0 for standard HY estimator)
        use_returns: Whether to use pre-computed returns instead of prices
        return_full_matrix: Whether to return the full covariance matrix or just pairwise covariances
        adjust_overlap: Whether to adjust for overlapping intervals in lead-lag version
        min_overlap: Minimum overlap required between intervals (as fraction of interval length)
        max_gap: Maximum gap allowed between intervals (as fraction of interval length)
        use_weights: Whether to use weights based on overlap length
        bias_correction: Whether to apply bias correction for microstructure noise
    """
    
    lead_lag: int = 0
    use_returns: bool = False
    return_full_matrix: bool = True
    adjust_overlap: bool = True
    min_overlap: float = 0.0
    max_gap: float = float('inf')
    use_weights: bool = False
    bias_correction: bool = False
    
    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        super().__post_init__()
        
        # Validate lead_lag
        if not isinstance(self.lead_lag, int) or self.lead_lag < 0:
            raise ParameterError(f"lead_lag must be a non-negative integer, got {self.lead_lag}")
        
        # Validate min_overlap
        if not 0.0 <= self.min_overlap <= 1.0:
            raise ParameterError(f"min_overlap must be between 0 and 1, got {self.min_overlap}")
        
        # Validate max_gap
        if self.max_gap <= 0:
            raise ParameterError(f"max_gap must be positive, got {self.max_gap}")


@dataclass
class HayashiYoshidaResult(RealizedEstimatorResult):
    """Result container for Hayashi-Yoshida estimator.
    
    This class extends RealizedEstimatorResult to provide specialized functionality
    for Hayashi-Yoshida estimator results, including overlap statistics and
    lead-lag information.
    
    Attributes:
        overlap_statistics: Statistics about interval overlaps
        lead_lag_info: Information about lead-lag adjustments
        pairwise_covariances: Matrix of pairwise covariances
        correlation_matrix: Correlation matrix derived from the covariance matrix
        bias_correction_info: Information about bias correction
    """
    
    overlap_statistics: Optional[Dict[str, Any]] = None
    lead_lag_info: Optional[Dict[str, Any]] = None
    pairwise_covariances: Optional[np.ndarray] = None
    correlation_matrix: Optional[np.ndarray] = None
    bias_correction_info: Optional[Dict[str, Any]] = None
    
    def __post_init__(self) -> None:
        """Validate result object after initialization."""
        super().__post_init__()
        
        # Ensure arrays are NumPy arrays if provided
        if self.pairwise_covariances is not None and not isinstance(self.pairwise_covariances, np.ndarray):
            self.pairwise_covariances = np.array(self.pairwise_covariances)
        
        if self.correlation_matrix is not None and not isinstance(self.correlation_matrix, np.ndarray):
            self.correlation_matrix = np.array(self.correlation_matrix)
    
    def summary(self) -> str:
        """Generate a text summary of the Hayashi-Yoshida results.
        
        Returns:
            str: A formatted string containing the Hayashi-Yoshida results summary
        """
        base_summary = super().summary()
        
        additional_info = ""
        
        # Add overlap statistics
        if self.overlap_statistics is not None:
            additional_info += "Overlap Statistics:\n"
            for key, value in self.overlap_statistics.items():
                additional_info += f"  {key}: {value}\n"
        
        # Add lead-lag information
        if self.lead_lag_info is not None:
            additional_info += "Lead-Lag Information:\n"
            for key, value in self.lead_lag_info.items():
                additional_info += f"  {key}: {value}\n"
        
        # Add bias correction information
        if self.bias_correction_info is not None:
            additional_info += "Bias Correction Information:\n"
            for key, value in self.bias_correction_info.items():
                additional_info += f"  {key}: {value}\n"
        
        if additional_info:
            additional_info = "Additional Information:\n" + additional_info + "\n"
        
        return base_summary + additional_info
    
    def get_correlation_matrix(self) -> np.ndarray:
        """Get the correlation matrix derived from the covariance matrix.
        
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
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(
            self.correlation_matrix,
            annot=True,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            ax=ax
        )
        
        ax.set_title("Hayashi-Yoshida Correlation Matrix")
        
        return fig
    
    def plot_pairwise_covariances(self, **kwargs: Any) -> Any:
        """Plot a heatmap of the pairwise covariances.
        
        Args:
            **kwargs: Additional keyword arguments for plotting
        
        Returns:
            Any: Plot object
        
        Raises:
            ValueError: If pairwise covariances are not available
            ImportError: If matplotlib and seaborn are not installed
        """
        if self.pairwise_covariances is None:
            raise ValueError("Pairwise covariances are not available")
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            raise ImportError("Matplotlib and seaborn are required for plotting")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(
            self.pairwise_covariances,
            annot=True,
            cmap="viridis",
            ax=ax
        )
        
        ax.set_title("Hayashi-Yoshida Pairwise Covariances")
        
        return fig


@jit(nopython=True, cache=True)
def _compute_hayashi_yoshida_numba(returns1: np.ndarray, times1: np.ndarray, 
                                  returns2: np.ndarray, times2: np.ndarray,
                                  min_overlap: float = 0.0, max_gap: float = float('inf'),
                                  use_weights: bool = False) -> Tuple[float, Dict[str, float]]:
    """Numba-accelerated implementation of Hayashi-Yoshida covariance estimator.
    
    Args:
        returns1: Array of returns for first asset
        times1: Array of times for first asset
        returns2: Array of returns for second asset
        times2: Array of times for second asset
        min_overlap: Minimum overlap required between intervals (as fraction of interval length)
        max_gap: Maximum gap allowed between intervals (as fraction of interval length)
        use_weights: Whether to use weights based on overlap length
        
    Returns:
        Tuple containing:
        - Hayashi-Yoshida covariance estimate
        - Dictionary of overlap statistics
    """
    n1 = len(returns1)
    n2 = len(returns2)
    
    # Compute end times for each return interval
    end_times1 = times1[1:]
    end_times2 = times2[1:]
    
    # Start times are the original times except the last one
    start_times1 = times1[:-1]
    start_times2 = times2[:-1]
    
    # Initialize statistics
    total_overlaps = 0
    total_overlap_length = 0.0
    max_overlap_length = 0.0
    min_overlap_length = float('inf')
    total_intervals = n1 * n2
    
    # Compute Hayashi-Yoshida estimator
    hy_cov = 0.0
    
    for i in range(n1):
        for j in range(n2):
            # Check if intervals overlap
            if (start_times1[i] < end_times2[j] and end_times1[i] > start_times2[j]):
                # Compute overlap length
                overlap_start = max(start_times1[i], start_times2[j])
                overlap_end = min(end_times1[i], end_times2[j])
                overlap_length = overlap_end - overlap_start
                
                # Compute interval lengths
                interval1_length = end_times1[i] - start_times1[i]
                interval2_length = end_times2[j] - start_times2[j]
                
                # Check minimum overlap requirement
                if overlap_length >= min_overlap * max(interval1_length, interval2_length):
                    # Check maximum gap requirement
                    gap1 = max(0, start_times1[i] - end_times2[j])
                    gap2 = max(0, start_times2[j] - end_times1[i])
                    max_gap_actual = max(gap1, gap2)
                    
                    if max_gap_actual <= max_gap * max(interval1_length, interval2_length):
                        # Update statistics
                        total_overlaps += 1
                        total_overlap_length += overlap_length
                        max_overlap_length = max(max_overlap_length, overlap_length)
                        min_overlap_length = min(min_overlap_length, overlap_length)
                        
                        # Compute contribution to covariance
                        if use_weights:
                            # Use overlap length as weight
                            weight = overlap_length / max(interval1_length, interval2_length)
                            hy_cov += weight * returns1[i] * returns2[j]
                        else:
                            # Standard Hayashi-Yoshida
                            hy_cov += returns1[i] * returns2[j]
    
    # Finalize statistics
    if total_overlaps > 0:
        avg_overlap_length = total_overlap_length / total_overlaps
    else:
        avg_overlap_length = 0.0
        min_overlap_length = 0.0
    
    # Create statistics dictionary (will be converted to Python dict outside Numba)
    stats = {
        "total_overlaps": total_overlaps,
        "total_intervals": total_intervals,
        "overlap_ratio": total_overlaps / max(total_intervals, 1),
        "avg_overlap_length": avg_overlap_length,
        "max_overlap_length": max_overlap_length,
        "min_overlap_length": min_overlap_length if total_overlaps > 0 else 0.0
    }
    
    return hy_cov, stats


def _compute_hayashi_yoshida_numpy(returns1: np.ndarray, times1: np.ndarray, 
                                  returns2: np.ndarray, times2: np.ndarray,
                                  min_overlap: float = 0.0, max_gap: float = float('inf'),
                                  use_weights: bool = False) -> Tuple[float, Dict[str, float]]:
    """NumPy implementation of Hayashi-Yoshida covariance estimator.
    
    Args:
        returns1: Array of returns for first asset
        times1: Array of times for first asset
        returns2: Array of returns for second asset
        times2: Array of times for second asset
        min_overlap: Minimum overlap required between intervals (as fraction of interval length)
        max_gap: Maximum gap allowed between intervals (as fraction of interval length)
        use_weights: Whether to use weights based on overlap length
        
    Returns:
        Tuple containing:
        - Hayashi-Yoshida covariance estimate
        - Dictionary of overlap statistics
    """
    n1 = len(returns1)
    n2 = len(returns2)
    
    # Compute end times for each return interval
    end_times1 = times1[1:]
    end_times2 = times2[1:]
    
    # Start times are the original times except the last one
    start_times1 = times1[:-1]
    start_times2 = times2[:-1]
    
    # Initialize statistics
    total_overlaps = 0
    total_overlap_length = 0.0
    max_overlap_length = 0.0
    min_overlap_length = float('inf')
    total_intervals = n1 * n2
    
    # Compute Hayashi-Yoshida estimator
    hy_cov = 0.0
    
    for i in range(n1):
        for j in range(n2):
            # Check if intervals overlap
            if (start_times1[i] < end_times2[j] and end_times1[i] > start_times2[j]):
                # Compute overlap length
                overlap_start = max(start_times1[i], start_times2[j])
                overlap_end = min(end_times1[i], end_times2[j])
                overlap_length = overlap_end - overlap_start
                
                # Compute interval lengths
                interval1_length = end_times1[i] - start_times1[i]
                interval2_length = end_times2[j] - start_times2[j]
                
                # Check minimum overlap requirement
                if overlap_length >= min_overlap * max(interval1_length, interval2_length):
                    # Check maximum gap requirement
                    gap1 = max(0, start_times1[i] - end_times2[j])
                    gap2 = max(0, start_times2[j] - end_times1[i])
                    max_gap_actual = max(gap1, gap2)
                    
                    if max_gap_actual <= max_gap * max(interval1_length, interval2_length):
                        # Update statistics
                        total_overlaps += 1
                        total_overlap_length += overlap_length
                        max_overlap_length = max(max_overlap_length, overlap_length)
                        min_overlap_length = min(min_overlap_length, overlap_length)
                        
                        # Compute contribution to covariance
                        if use_weights:
                            # Use overlap length as weight
                            weight = overlap_length / max(interval1_length, interval2_length)
                            hy_cov += weight * returns1[i] * returns2[j]
                        else:
                            # Standard Hayashi-Yoshida
                            hy_cov += returns1[i] * returns2[j]
    
    # Finalize statistics
    if total_overlaps > 0:
        avg_overlap_length = total_overlap_length / total_overlaps
    else:
        avg_overlap_length = 0.0
        min_overlap_length = 0.0
    
    # Create statistics dictionary
    stats = {
        "total_overlaps": total_overlaps,
        "total_intervals": total_intervals,
        "overlap_ratio": total_overlaps / max(total_intervals, 1),
        "avg_overlap_length": avg_overlap_length,
        "max_overlap_length": max_overlap_length,
        "min_overlap_length": min_overlap_length if total_overlaps > 0 else 0.0
    }
    
    return hy_cov, stats


def _compute_lead_lag_hayashi_yoshida(returns1: np.ndarray, times1: np.ndarray, 
                                     returns2: np.ndarray, times2: np.ndarray,
                                     lead_lag: int, adjust_overlap: bool = True,
                                     min_overlap: float = 0.0, max_gap: float = float('inf'),
                                     use_weights: bool = False) -> Tuple[float, Dict[str, Any]]:
    """Compute lead-lag adjusted Hayashi-Yoshida covariance estimator.
    
    Args:
        returns1: Array of returns for first asset
        times1: Array of times for first asset
        returns2: Array of returns for second asset
        times2: Array of times for second asset
        lead_lag: Number of leads and lags to include
        adjust_overlap: Whether to adjust for overlapping intervals
        min_overlap: Minimum overlap required between intervals (as fraction of interval length)
        max_gap: Maximum gap allowed between intervals (as fraction of interval length)
        use_weights: Whether to use weights based on overlap length
        
    Returns:
        Tuple containing:
        - Lead-lag adjusted Hayashi-Yoshida covariance estimate
        - Dictionary of lead-lag information
    """
    # Compute standard Hayashi-Yoshida estimator
    if HAS_NUMBA:
        hy_cov, stats = _compute_hayashi_yoshida_numba(
            returns1, times1, returns2, times2, min_overlap, max_gap, use_weights
        )
    else:
        hy_cov, stats = _compute_hayashi_yoshida_numpy(
            returns1, times1, returns2, times2, min_overlap, max_gap, use_weights
        )
    
    # If lead_lag is 0, return standard estimator
    if lead_lag == 0:
        return hy_cov, {"lead_lag": 0, "components": [hy_cov], "weights": [1.0]}
    
    # Initialize lead-lag components and weights
    components = [hy_cov]
    weights = [1.0]
    
    # Compute lead and lag components
    for k in range(1, lead_lag + 1):
        # Skip if not enough data for this lead/lag
        if k >= len(returns1) or k >= len(returns2):
            continue
        
        # Compute lead component (asset 1 leads asset 2)
        lead_returns1 = returns1[:-k]
        lead_times1 = times1[:-k]
        lead_returns2 = returns2[k:]
        lead_times2 = times2[k:]
        
        if HAS_NUMBA:
            lead_cov, _ = _compute_hayashi_yoshida_numba(
                lead_returns1, lead_times1, lead_returns2, lead_times2,
                min_overlap, max_gap, use_weights
            )
        else:
            lead_cov, _ = _compute_hayashi_yoshida_numpy(
                lead_returns1, lead_times1, lead_returns2, lead_times2,
                min_overlap, max_gap, use_weights
            )
        
        # Compute lag component (asset 2 leads asset 1)
        lag_returns1 = returns1[k:]
        lag_times1 = times1[k:]
        lag_returns2 = returns2[:-k]
        lag_times2 = times2[:-k]
        
        if HAS_NUMBA:
            lag_cov, _ = _compute_hayashi_yoshida_numba(
                lag_returns1, lag_times1, lag_returns2, lag_times2,
                min_overlap, max_gap, use_weights
            )
        else:
            lag_cov, _ = _compute_hayashi_yoshida_numpy(
                lag_returns1, lag_times1, lag_returns2, lag_times2,
                min_overlap, max_gap, use_weights
            )
        
        # Add components
        components.extend([lead_cov, lag_cov])
        
        # Compute weights (decreasing with lag)
        if adjust_overlap:
            # Bartlett kernel weights
            weight = 1.0 - k / (lead_lag + 1)
        else:
            # Equal weights
            weight = 1.0
        
        weights.extend([weight, weight])
    
    # Compute weighted sum
    ll_hy_cov = 0.0
    total_weight = 0.0
    
    for i, (component, weight) in enumerate(zip(components, weights)):
        ll_hy_cov += component * weight
        total_weight += weight
    
    # Normalize by total weight
    if total_weight > 0:
        ll_hy_cov /= total_weight
    
    # Create lead-lag information
    ll_info = {
        "lead_lag": lead_lag,
        "components": components,
        "weights": weights,
        "total_weight": total_weight,
        "adjust_overlap": adjust_overlap
    }
    
    return ll_hy_cov, ll_info


def _compute_bias_correction(returns1: np.ndarray, returns2: np.ndarray, 
                           noise_var1: Optional[float] = None, 
                           noise_var2: Optional[float] = None) -> Tuple[float, Dict[str, float]]:
    """Compute bias correction for Hayashi-Yoshida estimator.
    
    Args:
        returns1: Array of returns for first asset
        returns2: Array of returns for second asset
        noise_var1: Estimated noise variance for first asset (if None, it will be estimated)
        noise_var2: Estimated noise variance for second asset (if None, it will be estimated)
        
    Returns:
        Tuple containing:
        - Bias correction factor
        - Dictionary of bias correction information
    """
    from .utils import noise_variance
    
    # Estimate noise variances if not provided
    if noise_var1 is None:
        noise_var1 = noise_variance(returns1)
    
    if noise_var2 is None:
        noise_var2 = noise_variance(returns2)
    
    # Compute bias correction
    # The bias is proportional to the product of noise variances
    # and the number of overlapping intervals
    n1 = len(returns1)
    n2 = len(returns2)
    
    # Simplified bias correction (assumes all intervals overlap)
    # In practice, this is an upper bound on the bias
    bias_correction = 2 * noise_var1 * noise_var2 * min(n1, n2)
    
    # Create bias correction information
    bias_info = {
        "noise_var1": noise_var1,
        "noise_var2": noise_var2,
        "bias_correction": bias_correction
    }
    
    return bias_correction, bias_info


class HayashiYoshida(MultivariateRealizedEstimator):
    """Hayashi-Yoshida estimator for asynchronous covariance estimation.
    
    This class implements the Hayashi-Yoshida estimator for computing quadratic
    covariation between asynchronous asset price series. The estimator is uniquely
    capable of handling non-synchronous trading times without requiring data
    alignment through interpolation.
    
    The implementation supports both standard Hayashi-Yoshida estimation and
    lead-lag adjustments for improved robustness to microstructure noise.
    It also provides options for minimum overlap requirements, maximum gap
    constraints, and overlap-based weighting.
    
    Attributes:
        config: Configuration parameters for the estimator
        _n_assets: Number of assets in the data
        _realized_measure: Computed realized covariance matrix
        _pairwise_covariances: Matrix of pairwise covariances
        _correlation_matrix: Correlation matrix derived from the covariance matrix
        _overlap_statistics: Statistics about interval overlaps
        _lead_lag_info: Information about lead-lag adjustments
        _bias_correction_info: Information about bias correction
    """
    
    def __init__(self, config: Optional[HayashiYoshidaConfig] = None, name: str = "HayashiYoshida"):
        """Initialize the Hayashi-Yoshida estimator.
        
        Args:
            config: Configuration parameters for the estimator
            name: A descriptive name for the estimator
        """
        config = config if config is not None else HayashiYoshidaConfig()
        super().__init__(config=config, name=name)
        self._pairwise_covariances: Optional[np.ndarray] = None
        self._correlation_matrix: Optional[np.ndarray] = None
        self._overlap_statistics: Optional[Dict[str, Dict[str, float]]] = None
        self._lead_lag_info: Optional[Dict[str, Dict[str, Any]]] = None
        self._bias_correction_info: Optional[Dict[str, Dict[str, float]]] = None
    
    @property
    def config(self) -> HayashiYoshidaConfig:
        """Get the estimator configuration.
        
        Returns:
            HayashiYoshidaConfig: The estimator configuration
        """
        return cast(HayashiYoshidaConfig, self._config)
    
    @config.setter
    def config(self, config: HayashiYoshidaConfig) -> None:
        """Set the estimator configuration.
        
        Args:
            config: New configuration parameters
        
        Raises:
            TypeError: If config is not a HayashiYoshidaConfig
        """
        if not isinstance(config, HayashiYoshidaConfig):
            raise TypeError(f"config must be a HayashiYoshidaConfig, got {type(config)}")
        
        self._config = config
        self._fitted = False  # Reset fitted state when configuration changes
    
    def _compute_realized_measure(self, 
                                 prices: np.ndarray, 
                                 times: np.ndarray, 
                                 returns: np.ndarray,
                                 **kwargs: Any) -> np.ndarray:
        """Compute the Hayashi-Yoshida covariance matrix from the preprocessed data.
        
        Args:
            prices: Preprocessed price data (n_obs, n_assets)
            times: Preprocessed time points
            returns: Returns computed from prices (n_obs-1, n_assets)
            **kwargs: Additional keyword arguments for computation
        
        Returns:
            np.ndarray: Realized covariance matrix (n_assets, n_assets)
        
        Raises:
            ValueError: If computation fails
        """
        n_obs, n_assets = returns.shape
        
        # Store number of assets
        self._n_assets = n_assets
        
        # Initialize covariance matrix
        cov_matrix = np.zeros((n_assets, n_assets))
        
        # Initialize storage for statistics and information
        self._overlap_statistics = {}
        self._lead_lag_info = {}
        self._bias_correction_info = {}
        
        # Compute pairwise covariances
        pairwise_cov = np.zeros((n_assets, n_assets))
        
        # Compute variances (diagonal elements)
        for i in range(n_assets):
            # Variance is just the sum of squared returns
            pairwise_cov[i, i] = np.sum(returns[:, i]**2)
            cov_matrix[i, i] = pairwise_cov[i, i]
        
        # Compute covariances (off-diagonal elements)
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                # Extract returns and times for assets i and j
                returns_i = returns[:, i]
                returns_j = returns[:, j]
                
                # Compute Hayashi-Yoshida estimator
                if self.config.lead_lag > 0:
                    # Lead-lag adjusted estimator
                    hy_cov, ll_info = _compute_lead_lag_hayashi_yoshida(
                        returns_i, times[:-1], returns_j, times[:-1],
                        self.config.lead_lag, self.config.adjust_overlap,
                        self.config.min_overlap, self.config.max_gap,
                        self.config.use_weights
                    )
                    self._lead_lag_info[(i, j)] = ll_info
                else:
                    # Standard Hayashi-Yoshida estimator
                    if HAS_NUMBA:
                        hy_cov, stats = _compute_hayashi_yoshida_numba(
                            returns_i, times[:-1], returns_j, times[:-1],
                            self.config.min_overlap, self.config.max_gap,
                            self.config.use_weights
                        )
                    else:
                        hy_cov, stats = _compute_hayashi_yoshida_numpy(
                            returns_i, times[:-1], returns_j, times[:-1],
                            self.config.min_overlap, self.config.max_gap,
                            self.config.use_weights
                        )
                    self._overlap_statistics[(i, j)] = stats
                
                # Apply bias correction if requested
                if self.config.bias_correction:
                    bias_correction, bias_info = _compute_bias_correction(
                        returns_i, returns_j
                    )
                    self._bias_correction_info[(i, j)] = bias_info
                    
                    # Adjust covariance
                    hy_cov -= bias_correction
                
                # Store covariance
                pairwise_cov[i, j] = hy_cov
                pairwise_cov[j, i] = hy_cov  # Symmetric
                
                # Update covariance matrix
                cov_matrix[i, j] = hy_cov
                cov_matrix[j, i] = hy_cov  # Symmetric
        
        # Store pairwise covariances
        self._pairwise_covariances = pairwise_cov
        
        # Compute correlation matrix
        corr_matrix = np.zeros_like(cov_matrix)
        for i in range(n_assets):
            for j in range(n_assets):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    # Compute correlation coefficient
                    var_i = cov_matrix[i, i]
                    var_j = cov_matrix[j, j]
                    
                    # Avoid division by zero
                    if var_i > 0 and var_j > 0:
                        corr_matrix[i, j] = cov_matrix[i, j] / np.sqrt(var_i * var_j)
                    else:
                        corr_matrix[i, j] = 0.0
        
        # Store correlation matrix
        self._correlation_matrix = corr_matrix
        
        return cov_matrix
    
    def fit(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> HayashiYoshidaResult:
        """Fit the Hayashi-Yoshida estimator to the provided data.
        
        This method validates the input data, preprocesses it according to the
        estimator configuration, and then computes the Hayashi-Yoshida covariance matrix.
        
        Args:
            data: The data to fit the estimator to, as a tuple of (prices, times)
            **kwargs: Additional keyword arguments for estimation
        
        Returns:
            HayashiYoshidaResult: The estimation results
        
        Raises:
            ValueError: If the data is invalid
            RuntimeError: If the estimation fails
        """
        import time
        start_time = time.time()
        
        # Validate input data
        self.validate_data(data)
        prices, times = data
        
        # Handle pandas DataFrame/Series inputs
        if isinstance(prices, pd.DataFrame):
            # Convert DataFrame to NumPy array
            prices_np = prices.values
            
            # If times is a DatetimeIndex, convert to numeric values
            if isinstance(prices.index, pd.DatetimeIndex):
                times_np = prices.index.astype(np.int64) / 1e9  # Convert to seconds
            else:
                times_np = np.array(prices.index)
            
            # Update data tuple
            data = (prices_np, times_np)
        
        # Preprocess data
        processed_prices, processed_times, returns = self.preprocess_data(*data)
        
        try:
            # Compute realized measure
            realized_measure = self._compute_realized_measure(
                processed_prices, processed_times, returns, **kwargs
            )
            
            # Update instance state
            self._realized_measure = realized_measure
            self._fitted = True
            
            # Create result object
            result = HayashiYoshidaResult(
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
                config=self._config.to_dict(),
                overlap_statistics=self._overlap_statistics,
                lead_lag_info=self._lead_lag_info,
                pairwise_covariances=self._pairwise_covariances,
                correlation_matrix=self._correlation_matrix,
                bias_correction_info=self._bias_correction_info
            )
            
            # Store result
            self._results = result
            
            return result
        
        except Exception as e:
            logger.error(f"Estimation failed: {str(e)}")
            raise RuntimeError(f"Hayashi-Yoshida estimation failed: {str(e)}") from e
    
    async def fit_async(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> HayashiYoshidaResult:
        """Asynchronously fit the Hayashi-Yoshida estimator to the provided data.
        
        This method provides an asynchronous interface to the fit method,
        allowing for non-blocking estimation in UI contexts.
        
        Args:
            data: The data to fit the estimator to, as a tuple of (prices, times)
            **kwargs: Additional keyword arguments for estimation
        
        Returns:
            HayashiYoshidaResult: The estimation results
        
        Raises:
            ValueError: If the data is invalid
            RuntimeError: If the estimation fails
        """
        # This implementation runs the synchronous version in a separate thread
        # to avoid blocking the main thread
        import asyncio
        import concurrent.futures
        
        # Create a thread pool executor
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Run the fit method in a separate thread
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                executor,
                lambda: self.fit(data, **kwargs)
            )
            
            return result
    
    def calibrate(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> HayashiYoshidaConfig:
        """Calibrate the estimator configuration based on the provided data.
        
        This method analyzes the input data and determines optimal configuration
        parameters for the estimator, such as lead-lag value and overlap requirements.
        
        Args:
            data: The data to calibrate the estimator with, as a tuple of (prices, times)
            **kwargs: Additional keyword arguments for calibration
        
        Returns:
            HayashiYoshidaConfig: Calibrated configuration
        
        Raises:
            ValueError: If the data is invalid
        """
        # Validate input data
        self.validate_data(data)
        prices, times = data
        
        # Create a copy of the current configuration
        calibrated_config = HayashiYoshidaConfig(**self.config.to_dict())
        
        # Preprocess data
        processed_prices, processed_times, returns = self.preprocess_data(prices, times)
        
        # Determine optimal lead-lag value
        # Try different lead-lag values and choose the one that maximizes
        # the average absolute correlation
        max_lead_lag = min(10, returns.shape[0] // 10)  # Limit to 10 or 10% of data
        lead_lag_values = list(range(0, max_lead_lag + 1))
        avg_abs_corr = []
        
        n_assets = returns.shape[1]
        
        for lead_lag in lead_lag_values:
            # Create temporary estimator with this lead-lag value
            temp_config = HayashiYoshidaConfig(**self.config.to_dict())
            temp_config.lead_lag = lead_lag
            temp_estimator = HayashiYoshida(config=temp_config)
            
            try:
                # Fit estimator
                temp_estimator.fit((processed_prices, processed_times))
                
                # Get correlation matrix
                corr_matrix = temp_estimator._correlation_matrix
                
                # Compute average absolute correlation (excluding diagonal)
                abs_corr_sum = 0.0
                count = 0
                
                for i in range(n_assets):
                    for j in range(i+1, n_assets):
                        abs_corr_sum += abs(corr_matrix[i, j])
                        count += 1
                
                if count > 0:
                    avg_abs_corr.append(abs_corr_sum / count)
                else:
                    avg_abs_corr.append(0.0)
            except Exception as e:
                logger.warning(f"Error fitting with lead_lag={lead_lag}: {str(e)}")
                avg_abs_corr.append(0.0)
        
        # Choose lead-lag value that maximizes average absolute correlation
        if avg_abs_corr:
            best_lead_lag = lead_lag_values[np.argmax(avg_abs_corr)]
            calibrated_config.lead_lag = best_lead_lag
        
        # Determine if bias correction is needed
        # Estimate noise variance for each asset
        noise_variances = []
        for i in range(n_assets):
            from .utils import noise_variance
            try:
                noise_var = noise_variance(returns[:, i])
                noise_variances.append(noise_var)
            except Exception as e:
                noise_variances.append(0.0)
        
        # If average noise variance is significant, enable bias correction
        avg_noise_var = np.mean(noise_variances)
        avg_return_var = np.mean([np.var(returns[:, i]) for i in range(n_assets)])
        
        if avg_noise_var > 0.05 * avg_return_var:
            calibrated_config.bias_correction = True
        else:
            calibrated_config.bias_correction = False
        
        # Determine if weighting is beneficial
        # Try with and without weighting and choose the one with lower variance
        temp_config_no_weight = HayashiYoshidaConfig(**calibrated_config.to_dict())
        temp_config_no_weight.use_weights = False
        temp_estimator_no_weight = HayashiYoshida(config=temp_config_no_weight)
        
        temp_config_weight = HayashiYoshidaConfig(**calibrated_config.to_dict())
        temp_config_weight.use_weights = True
        temp_estimator_weight = HayashiYoshida(config=temp_config_weight)
        
        try:
            # Fit both estimators
            temp_estimator_no_weight.fit((processed_prices, processed_times))
            temp_estimator_weight.fit((processed_prices, processed_times))
            
            # Compare stability of correlation matrices
            corr_no_weight = temp_estimator_no_weight._correlation_matrix
            corr_weight = temp_estimator_weight._correlation_matrix
            
            # Compute variance of off-diagonal elements
            var_no_weight = 0.0
            var_weight = 0.0
            count = 0
            
            for i in range(n_assets):
                for j in range(i+1, n_assets):
                    var_no_weight += corr_no_weight[i, j]**2
                    var_weight += corr_weight[i, j]**2
                    count += 1
            
            if count > 0:
                var_no_weight /= count
                var_weight /= count
            
            # Choose weighting based on lower variance
            calibrated_config.use_weights = var_weight < var_no_weight
        except Exception as e:
            # Default to no weighting if comparison fails
            calibrated_config.use_weights = False
        
        return calibrated_config
    
    def get_pairwise_covariances(self) -> np.ndarray:
        """Get the matrix of pairwise covariances.
        
        Returns:
            np.ndarray: Matrix of pairwise covariances
        
        Raises:
            RuntimeError: If the estimator has not been fitted
        """
        if not self._fitted or self._pairwise_covariances is None:
            raise RuntimeError("Estimator has not been fitted. Call fit() first.")
        
        return self._pairwise_covariances
    
    def get_correlation_matrix(self) -> np.ndarray:
        """Get the correlation matrix derived from the covariance matrix.
        
        Returns:
            np.ndarray: Correlation matrix
        
        Raises:
            RuntimeError: If the estimator has not been fitted
        """
        if not self._fitted or self._correlation_matrix is None:
            raise RuntimeError("Estimator has not been fitted. Call fit() first.")
        
        return self._correlation_matrix
    
    def get_overlap_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics about interval overlaps.
        
        Returns:
            Dict[str, Dict[str, float]]: Dictionary of overlap statistics
        
        Raises:
            RuntimeError: If the estimator has not been fitted
        """
        if not self._fitted or self._overlap_statistics is None:
            raise RuntimeError("Estimator has not been fitted. Call fit() first.")
        
        return self._overlap_statistics
    
    def plot_correlation_heatmap(self, **kwargs: Any) -> Any:
        """Plot a heatmap of the correlation matrix.
        
        Args:
            **kwargs: Additional keyword arguments for plotting
        
        Returns:
            Any: Plot object
        
        Raises:
            RuntimeError: If the estimator has not been fitted
            ImportError: If matplotlib and seaborn are not installed
        """
        if not self._fitted or self._correlation_matrix is None:
            raise RuntimeError("Estimator has not been fitted. Call fit() first.")
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            raise ImportError("Matplotlib and seaborn are required for plotting")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(
            self._correlation_matrix,
            annot=True,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            ax=ax
        )
        
        ax.set_title(f"Hayashi-Yoshida Correlation Matrix from {self._name}")
        
        return fig
    
    def plot_pairwise_covariances(self, **kwargs: Any) -> Any:
        """Plot a heatmap of the pairwise covariances.
        
        Args:
            **kwargs: Additional keyword arguments for plotting
        
        Returns:
            Any: Plot object
        
        Raises:
            RuntimeError: If the estimator has not been fitted
            ImportError: If matplotlib and seaborn are not installed
        """
        if not self._fitted or self._pairwise_covariances is None:
            raise RuntimeError("Estimator has not been fitted. Call fit() first.")
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            raise ImportError("Matplotlib and seaborn are required for plotting")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(
            self._pairwise_covariances,
            annot=True,
            cmap="viridis",
            ax=ax
        )
        
        ax.set_title(f"Hayashi-Yoshida Pairwise Covariances from {self._name}")
        
        return fig
    
    def plot_lead_lag_analysis(self, **kwargs: Any) -> Any:
        """Plot analysis of lead-lag components.
        
        Args:
            **kwargs: Additional keyword arguments for plotting
        
        Returns:
            Any: Plot object
        
        Raises:
            RuntimeError: If the estimator has not been fitted or lead-lag is not used
            ImportError: If matplotlib is not installed
        """
        if not self._fitted or self._lead_lag_info is None or not self._lead_lag_info:
            raise RuntimeError("Estimator has not been fitted or lead-lag is not used.")
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib is required for plotting")
        
        # Get number of asset pairs
        n_pairs = len(self._lead_lag_info)
        
        # Create figure with one subplot per asset pair
        fig, axes = plt.subplots(n_pairs, 1, figsize=(10, 4 * n_pairs), sharex=True)
        
        # Handle case with only one asset pair
        if n_pairs == 1:
            axes = [axes]
        
        # Plot lead-lag components for each asset pair
        for i, ((asset1, asset2), ll_info) in enumerate(self._lead_lag_info.items()):
            components = ll_info["components"]
            weights = ll_info["weights"]
            
            # Create labels for components
            labels = ["Contemporaneous"]
            for k in range(1, self.config.lead_lag + 1):
                labels.extend([f"Lead {k}", f"Lag {k}"])
            
            # Truncate labels if there are fewer components
            labels = labels[:len(components)]
            
            # Plot components
            axes[i].bar(range(len(components)), components, alpha=0.7)
            
            # Plot weighted components
            weighted_components = [c * w for c, w in zip(components, weights)]
            axes[i].bar(range(len(weighted_components)), weighted_components, alpha=0.4, color='red')
            
            # Add labels and title
            axes[i].set_title(f"Lead-Lag Components for Assets {asset1+1} and {asset2+1}")
            axes[i].set_ylabel("Covariance")
            axes[i].set_xticks(range(len(labels)))
            axes[i].set_xticklabels(labels, rotation=45)
            axes[i].legend(["Raw Components", "Weighted Components"])
            axes[i].grid(True, linestyle='--', alpha=0.7)
        
        # Set common x-axis label
        axes[-1].set_xlabel("Component Type")
        
        plt.tight_layout()
        
        return fig
