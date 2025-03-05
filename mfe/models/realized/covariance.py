# mfe/models/realized/covariance.py
"""
Realized Covariance Estimator

This module implements realized covariance estimators for multivariate high-frequency
financial data. It provides classes for computing covariance matrices from synchronous
or asynchronous price observations, with support for various sampling schemes and
noise reduction techniques.

The module includes both standard realized covariance estimators and subsampled variants
for improved accuracy in the presence of market microstructure noise. All estimators
inherit from the MultivariateRealizedEstimator base class, ensuring a consistent
interface and robust error handling.

Classes:
    RealizedCovariance: Standard realized covariance estimator
    SubsampledRealizedCovariance: Subsampled realized covariance estimator
"""

import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
from scipy import stats

from ...core.exceptions import ParameterError, DimensionError, NumericError
from ...utils.matrix_ops import ensure_symmetric, is_positive_definite, nearest_positive_definite
from ..realized.base import (
    MultivariateRealizedEstimator, 
    RealizedEstimatorConfig, 
    RealizedEstimatorResult
)

# Set up module-level logger
logger = logging.getLogger("mfe.models.realized.covariance")

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
    logger.debug("Numba available for realized covariance acceleration")
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    HAS_NUMBA = False
    logger.info("Numba not available. Realized covariance will use pure NumPy implementations.")


@dataclass
class RealizedCovarianceConfig(RealizedEstimatorConfig):
    """Configuration parameters for realized covariance estimators.
    
    This class extends RealizedEstimatorConfig to provide specialized configuration
    parameters for realized covariance estimators.
    
    Attributes:
        ensure_psd: Whether to ensure the resulting covariance matrix is positive semi-definite
        psd_method: Method to use for ensuring positive semi-definiteness ('nearest' or 'eigenvalue')
        eigenvalue_threshold: Threshold for eigenvalue adjustment when psd_method='eigenvalue'
        synchronize_data: Whether to synchronize asynchronous price data
        synchronization_method: Method to use for synchronization ('previous' or 'linear')
        return_correlation: Whether to also compute and return the correlation matrix
    """
    
    ensure_psd: bool = True
    psd_method: str = "nearest"
    eigenvalue_threshold: float = 1e-8
    synchronize_data: bool = True
    synchronization_method: str = "previous"
    return_correlation: bool = False
    
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
class RealizedCovarianceResult(RealizedEstimatorResult):
    """Result container for realized covariance estimators.
    
    This class extends RealizedEstimatorResult to provide specialized functionality
    for realized covariance estimator results, including correlation matrices and
    eigenvalue information.
    
    Attributes:
        correlation_matrix: Correlation matrix derived from the realized covariance
        eigenvalues: Eigenvalues of the realized covariance matrix
        is_psd: Whether the realized covariance matrix is positive semi-definite
        psd_adjustment: Adjustment made to ensure positive semi-definiteness
        synchronization_info: Information about data synchronization
    """
    
    correlation_matrix: Optional[np.ndarray] = None
    eigenvalues: Optional[np.ndarray] = None
    is_psd: Optional[bool] = None
    psd_adjustment: Optional[str] = None
    synchronization_info: Optional[Dict[str, Any]] = None
    
    def __post_init__(self) -> None:
        """Validate result object after initialization."""
        super().__post_init__()
        
        # Ensure arrays are NumPy arrays if provided
        if self.correlation_matrix is not None and not isinstance(self.correlation_matrix, np.ndarray):
            self.correlation_matrix = np.array(self.correlation_matrix)
        
        if self.eigenvalues is not None and not isinstance(self.eigenvalues, np.ndarray):
            self.eigenvalues = np.array(self.eigenvalues)
    
    def summary(self) -> str:
        """Generate a text summary of the realized covariance results.
        
        Returns:
            str: A formatted string containing the realized covariance results summary
        """
        base_summary = super().summary()
        
        additional_info = ""
        
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
        
        ax.set_title("Realized Correlation Matrix")
        
        return fig


@jit(nopython=True, cache=True)
def _compute_realized_covariance_numba(returns: np.ndarray) -> np.ndarray:
    """Numba-accelerated implementation of realized covariance computation.
    
    Args:
        returns: Matrix of returns (n_obs, n_assets)
        
    Returns:
        np.ndarray: Realized covariance matrix (n_assets, n_assets)
    """
    n_obs, n_assets = returns.shape
    cov = np.zeros((n_assets, n_assets))
    
    # Compute outer products and sum
    for t in range(n_obs):
        for i in range(n_assets):
            for j in range(n_assets):
                cov[i, j] += returns[t, i] * returns[t, j]
    
    return cov


def _compute_realized_covariance_numpy(returns: np.ndarray) -> np.ndarray:
    """NumPy implementation of realized covariance computation.
    
    Args:
        returns: Matrix of returns (n_obs, n_assets)
        
    Returns:
        np.ndarray: Realized covariance matrix (n_assets, n_assets)
    """
    # Compute realized covariance as sum of outer products
    return returns.T @ returns


def _ensure_psd(matrix: np.ndarray, method: str = "nearest", threshold: float = 1e-8) -> Tuple[np.ndarray, bool, str, np.ndarray]:
    """Ensure a matrix is positive semi-definite.
    
    Args:
        matrix: Matrix to ensure is positive semi-definite
        method: Method to use ('nearest' or 'eigenvalue')
        threshold: Threshold for eigenvalue adjustment
        
    Returns:
        Tuple containing:
        - Positive semi-definite matrix
        - Whether the original matrix was positive semi-definite
        - Description of adjustment made
        - Eigenvalues of the adjusted matrix
    """
    # Check if matrix is already PSD
    is_psd = is_positive_definite(matrix)
    
    if is_psd:
        # Compute eigenvalues for information
        eigenvalues = np.linalg.eigvalsh(matrix)
        return matrix, True, "No adjustment needed", eigenvalues
    
    # Matrix is not PSD, apply adjustment
    if method == "nearest":
        # Find nearest PSD matrix
        adjusted = nearest_positive_definite(matrix)
        eigenvalues = np.linalg.eigvalsh(adjusted)
        return adjusted, False, "Nearest positive definite matrix", eigenvalues
    
    elif method == "eigenvalue":
        # Adjust eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        
        # Replace negative eigenvalues with threshold
        neg_indices = eigenvalues < threshold
        if np.any(neg_indices):
            eigenvalues[neg_indices] = threshold
            
            # Reconstruct matrix
            adjusted = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            
            # Ensure symmetry
            adjusted = ensure_symmetric(adjusted)
            
            return adjusted, False, f"Eigenvalue adjustment (min: {np.min(eigenvalues):.6e})", eigenvalues
        
        # This shouldn't happen if is_psd was False, but just in case
        return matrix, True, "No adjustment needed", eigenvalues
    
    else:
        raise ValueError(f"Unknown PSD method: {method}")


def _synchronize_prices(prices: np.ndarray, times: np.ndarray, method: str = "previous") -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Synchronize asynchronous price data.
    
    Args:
        prices: Matrix of prices (n_obs, n_assets)
        times: Array of observation times
        method: Synchronization method ('previous' or 'linear')
        
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
    
    # Create synchronized arrays
    sync_prices = np.zeros_like(prices)
    
    # Count NaN values before synchronization
    nan_count_before = np.isnan(prices).sum()
    
    if method == "previous":
        # Fill forward (use previous valid value)
        for j in range(n_assets):
            # Initialize with first valid value
            last_valid_idx = np.where(valid_mask[:, j])[0]
            if len(last_valid_idx) == 0:
                # No valid values for this asset
                sync_prices[:, j] = np.nan
                continue
                
            first_valid = last_valid_idx[0]
            last_valid_value = prices[first_valid, j]
            
            # Fill initial NaNs with first valid value
            sync_prices[:first_valid+1, j] = last_valid_value
            
            # Fill forward for the rest
            for i in range(first_valid+1, n_obs):
                if valid_mask[i, j]:
                    last_valid_value = prices[i, j]
                sync_prices[i, j] = last_valid_value
    
    elif method == "linear":
        # Linear interpolation
        for j in range(n_assets):
            # Get indices of valid values
            valid_indices = np.where(valid_mask[:, j])[0]
            
            if len(valid_indices) == 0:
                # No valid values for this asset
                sync_prices[:, j] = np.nan
                continue
                
            if len(valid_indices) == 1:
                # Only one valid value, use it for all
                sync_prices[:, j] = prices[valid_indices[0], j]
                continue
                
            # Get valid values and their times
            valid_values = prices[valid_indices, j]
            valid_times = times[valid_indices]
            
            # Interpolate
            sync_prices[:, j] = np.interp(
                times,
                valid_times,
                valid_values,
                left=valid_values[0],  # Use first valid value for extrapolation
                right=valid_values[-1]  # Use last valid value for extrapolation
            )
    
    else:
        raise ValueError(f"Unknown synchronization method: {method}")
    
    # Count NaN values after synchronization
    nan_count_after = np.isnan(sync_prices).sum()
    
    # Create synchronization info
    sync_info = {
        "synchronized": True,
        "method": method,
        "nan_count_before": int(nan_count_before),
        "nan_count_after": int(nan_count_after),
        "nan_reduction_percent": 100 * (1 - nan_count_after / max(nan_count_before, 1))
    }
    
    return sync_prices, times, sync_info


class RealizedCovariance(MultivariateRealizedEstimator):
    """Realized covariance estimator for multivariate high-frequency data.
    
    This class implements the standard realized covariance estimator, which computes
    the covariance matrix as the sum of outer products of high-frequency returns.
    
    The estimator supports both synchronous and asynchronous price data, with
    options for data synchronization, subsampling, and ensuring positive
    semi-definiteness of the resulting covariance matrix.
    
    Attributes:
        config: Configuration parameters for the estimator
        _n_assets: Number of assets in the data
        _realized_measure: Computed realized covariance matrix
    """
    
    def __init__(self, config: Optional[RealizedCovarianceConfig] = None, name: str = "RealizedCovariance"):
        """Initialize the realized covariance estimator.
        
        Args:
            config: Configuration parameters for the estimator
            name: A descriptive name for the estimator
        """
        config = config if config is not None else RealizedCovarianceConfig()
        super().__init__(config=config, name=name)
    
    @property
    def config(self) -> RealizedCovarianceConfig:
        """Get the estimator configuration.
        
        Returns:
            RealizedCovarianceConfig: The estimator configuration
        """
        return cast(RealizedCovarianceConfig, self._config)
    
    @config.setter
    def config(self, config: RealizedCovarianceConfig) -> None:
        """Set the estimator configuration.
        
        Args:
            config: New configuration parameters
        
        Raises:
            TypeError: If config is not a RealizedCovarianceConfig
        """
        if not isinstance(config, RealizedCovarianceConfig):
            raise TypeError(f"config must be a RealizedCovarianceConfig, got {type(config)}")
        
        self._config = config
        self._fitted = False  # Reset fitted state when configuration changes
    
    def _compute_realized_measure(self, 
                                 prices: np.ndarray, 
                                 times: np.ndarray, 
                                 returns: np.ndarray,
                                 **kwargs: Any) -> np.ndarray:
        """Compute the realized covariance matrix from the preprocessed data.
        
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
        
        # Compute realized covariance
        if HAS_NUMBA:
            # Use Numba-accelerated implementation
            rcov = _compute_realized_covariance_numba(returns)
        else:
            # Use NumPy implementation
            rcov = _compute_realized_covariance_numpy(returns)
        
        # Ensure the matrix is symmetric (to handle numerical precision issues)
        rcov = ensure_symmetric(rcov)
        
        # Ensure positive semi-definiteness if requested
        if self.config.ensure_psd:
            rcov, is_psd, psd_adjustment, eigenvalues = _ensure_psd(
                rcov,
                method=self.config.psd_method,
                threshold=self.config.eigenvalue_threshold
            )
            
            # Store PSD information for result
            self._is_psd = is_psd
            self._psd_adjustment = psd_adjustment
            self._eigenvalues = eigenvalues
        else:
            # Compute eigenvalues for information
            self._eigenvalues = np.linalg.eigvalsh(rcov)
            self._is_psd = np.all(self._eigenvalues > -1e-10)
            self._psd_adjustment = "No adjustment applied"
        
        # Compute correlation matrix if requested
        if self.config.return_correlation:
            # Extract standard deviations from diagonal
            std_devs = np.sqrt(np.diag(rcov))
            
            # Avoid division by zero
            std_devs = np.maximum(std_devs, np.finfo(float).eps)
            
            # Compute correlation matrix
            corr = np.zeros_like(rcov)
            for i in range(n_assets):
                for j in range(n_assets):
                    corr[i, j] = rcov[i, j] / (std_devs[i] * std_devs[j])
            
            # Ensure diagonal is exactly 1
            np.fill_diagonal(corr, 1.0)
            
            # Ensure the matrix is symmetric
            corr = ensure_symmetric(corr)
            
            # Store correlation matrix for result
            self._correlation_matrix = corr
        else:
            self._correlation_matrix = None
        
        return rcov
    
    def fit(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> RealizedCovarianceResult:
        """Fit the realized covariance estimator to the provided data.
        
        This method validates the input data, preprocesses it according to the
        estimator configuration, and then computes the realized covariance matrix.
        
        Args:
            data: The data to fit the estimator to, as a tuple of (prices, times)
            **kwargs: Additional keyword arguments for estimation
        
        Returns:
            RealizedCovarianceResult: The estimation results
        
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
            prices, times, sync_info = _synchronize_prices(
                prices, 
                times, 
                method=self.config.synchronization_method
            )
            self._synchronization_info = sync_info
        else:
            self._synchronization_info = {"synchronized": False, "reason": "Synchronization not requested or not needed"}
        
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
            result = RealizedCovarianceResult(
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
                correlation_matrix=self._correlation_matrix,
                eigenvalues=self._eigenvalues,
                is_psd=self._is_psd,
                psd_adjustment=self._psd_adjustment,
                synchronization_info=self._synchronization_info
            )
            
            # Store result
            self._results = result
            
            return result
        
        except Exception as e:
            logger.error(f"Estimation failed: {str(e)}")
            raise RuntimeError(f"Realized covariance estimation failed: {str(e)}") from e
    
    async def fit_async(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> RealizedCovarianceResult:
        """Asynchronously fit the realized covariance estimator to the provided data.
        
        This method provides an asynchronous interface to the fit method,
        allowing for non-blocking estimation in UI contexts.
        
        Args:
            data: The data to fit the estimator to, as a tuple of (prices, times)
            **kwargs: Additional keyword arguments for estimation
        
        Returns:
            RealizedCovarianceResult: The estimation results
        
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
    
    def calibrate(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> RealizedCovarianceConfig:
        """Calibrate the estimator configuration based on the provided data.
        
        This method analyzes the input data and determines optimal configuration
        parameters for the estimator, such as synchronization method and PSD approach.
        
        Args:
            data: The data to calibrate the estimator with, as a tuple of (prices, times)
            **kwargs: Additional keyword arguments for calibration
        
        Returns:
            RealizedCovarianceConfig: Calibrated configuration
        
        Raises:
            ValueError: If the data is invalid
        """
        # Validate input data
        self.validate_data(data)
        prices, times = data
        
        # Create a copy of the current configuration
        calibrated_config = RealizedCovarianceConfig(**self.config.to_dict())
        
        # Check if data needs synchronization
        if np.isnan(prices).any():
            calibrated_config.synchronize_data = True
            
            # Try both synchronization methods and choose the better one
            prices_prev, _, _ = _synchronize_prices(prices, times, method="previous")
            prices_linear, _, _ = _synchronize_prices(prices, times, method="linear")
            
            # Compute returns for both methods
            if self.config.return_type == 'log':
                returns_prev = np.diff(np.log(prices_prev), axis=0)
                returns_linear = np.diff(np.log(prices_linear), axis=0)
            else:  # 'simple'
                returns_prev = np.diff(prices_prev, axis=0) / prices_prev[:-1, :]
                returns_linear = np.diff(prices_linear, axis=0) / prices_linear[:-1, :]
            
            # Compute covariance matrices for both methods
            cov_prev = _compute_realized_covariance_numpy(returns_prev)
            cov_linear = _compute_realized_covariance_numpy(returns_linear)
            
            # Check positive definiteness
            is_psd_prev = is_positive_definite(cov_prev)
            is_psd_linear = is_positive_definite(cov_linear)
            
            # Choose method based on positive definiteness and condition number
            if is_psd_prev and not is_psd_linear:
                calibrated_config.synchronization_method = "previous"
            elif is_psd_linear and not is_psd_prev:
                calibrated_config.synchronization_method = "linear"
            else:
                # Both or neither are PSD, choose based on condition number
                eigvals_prev = np.linalg.eigvalsh(cov_prev)
                eigvals_linear = np.linalg.eigvalsh(cov_linear)
                
                cond_prev = np.max(eigvals_prev) / max(np.min(eigvals_prev), 1e-15)
                cond_linear = np.max(eigvals_linear) / max(np.min(eigvals_linear), 1e-15)
                
                calibrated_config.synchronization_method = "linear" if cond_linear < cond_prev else "previous"
        else:
            calibrated_config.synchronize_data = False
        
        # Determine if PSD enforcement is needed
        # Preprocess data
        processed_prices, processed_times, returns = self.preprocess_data(prices, times)
        
        # Compute realized covariance
        rcov = _compute_realized_covariance_numpy(returns)
        
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
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(
            self._realized_measure,
            annot=True,
            cmap="viridis",
            ax=ax
        )
        
        ax.set_title(f"Realized Covariance Matrix from {self._name}")
        
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
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot eigenvalues
        ax.bar(range(1, len(self._eigenvalues) + 1), np.sort(self._eigenvalues)[::-1])
        
        ax.set_title(f"Eigenvalue Distribution of Realized Covariance Matrix from {self._name}")
        ax.set_xlabel("Eigenvalue Index")
        ax.set_ylabel("Eigenvalue")
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        return fig


class SubsampledRealizedCovariance(RealizedCovariance):
    """Subsampled realized covariance estimator for multivariate high-frequency data.
    
    This class extends the standard realized covariance estimator to implement
    subsampling for improved accuracy in the presence of market microstructure noise.
    
    The estimator computes multiple realized covariance matrices using different
    subsamples of the data and then averages them to produce the final estimate.
    
    Attributes:
        config: Configuration parameters for the estimator
        _n_assets: Number of assets in the data
        _realized_measure: Computed realized covariance matrix
        _subsampled_measures: List of realized covariance matrices from each subsample
    """
    
    def __init__(self, config: Optional[RealizedCovarianceConfig] = None, name: str = "SubsampledRealizedCovariance"):
        """Initialize the subsampled realized covariance estimator.
        
        Args:
            config: Configuration parameters for the estimator
            name: A descriptive name for the estimator
        """
        config = config if config is not None else RealizedCovarianceConfig(use_subsampling=True, subsampling_factor=5)
        
        # Ensure subsampling is enabled
        if not config.use_subsampling:
            logger.warning("Subsampling was disabled in config. Enabling it for SubsampledRealizedCovariance.")
            config.use_subsampling = True
        
        # Ensure subsampling_factor is at least 2
        if config.subsampling_factor < 2:
            logger.warning(f"Subsampling factor {config.subsampling_factor} is too small. Setting to 2.")
            config.subsampling_factor = 2
        
        super().__init__(config=config, name=name)
        self._subsampled_measures: List[np.ndarray] = []
    
    def _compute_realized_measure(self, 
                                 prices: np.ndarray, 
                                 times: np.ndarray, 
                                 returns: np.ndarray,
                                 **kwargs: Any) -> np.ndarray:
        """Compute the subsampled realized covariance matrix from the preprocessed data.
        
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
        
        # Get subsampling factor
        subsampling_factor = self.config.subsampling_factor
        
        # Initialize list to store subsampled measures
        self._subsampled_measures = []
        
        # Compute realized covariance for each subsample
        for i in range(subsampling_factor):
            # Create subsample by taking every subsampling_factor-th observation
            # starting from offset i
            subsample = returns[i::subsampling_factor]
            
            # Skip if subsample is too small
            if len(subsample) < 2:
                continue
            
            # Compute realized covariance for this subsample
            if HAS_NUMBA:
                # Use Numba-accelerated implementation
                rcov_subsample = _compute_realized_covariance_numba(subsample)
            else:
                # Use NumPy implementation
                rcov_subsample = _compute_realized_covariance_numpy(subsample)
            
            # Scale by subsampling factor to account for fewer observations
            rcov_subsample *= subsampling_factor
            
            # Ensure the matrix is symmetric
            rcov_subsample = ensure_symmetric(rcov_subsample)
            
            # Add to list of subsampled measures
            self._subsampled_measures.append(rcov_subsample)
        
        # Average the subsampled measures
        if not self._subsampled_measures:
            raise ValueError("No valid subsamples could be created. Try reducing the subsampling factor.")
        
        rcov = np.mean(self._subsampled_measures, axis=0)
        
        # Ensure the matrix is symmetric
        rcov = ensure_symmetric(rcov)
        
        # Ensure positive semi-definiteness if requested
        if self.config.ensure_psd:
            rcov, is_psd, psd_adjustment, eigenvalues = _ensure_psd(
                rcov,
                method=self.config.psd_method,
                threshold=self.config.eigenvalue_threshold
            )
            
            # Store PSD information for result
            self._is_psd = is_psd
            self._psd_adjustment = psd_adjustment
            self._eigenvalues = eigenvalues
        else:
            # Compute eigenvalues for information
            self._eigenvalues = np.linalg.eigvalsh(rcov)
            self._is_psd = np.all(self._eigenvalues > -1e-10)
            self._psd_adjustment = "No adjustment applied"
        
        # Compute correlation matrix if requested
        if self.config.return_correlation:
            # Extract standard deviations from diagonal
            std_devs = np.sqrt(np.diag(rcov))
            
            # Avoid division by zero
            std_devs = np.maximum(std_devs, np.finfo(float).eps)
            
            # Compute correlation matrix
            corr = np.zeros_like(rcov)
            for i in range(n_assets):
                for j in range(n_assets):
                    corr[i, j] = rcov[i, j] / (std_devs[i] * std_devs[j])
            
            # Ensure diagonal is exactly 1
            np.fill_diagonal(corr, 1.0)
            
            # Ensure the matrix is symmetric
            corr = ensure_symmetric(corr)
            
            # Store correlation matrix for result
            self._correlation_matrix = corr
        else:
            self._correlation_matrix = None
        
        return rcov
    
    def get_subsampled_measures(self) -> List[np.ndarray]:
        """Get the realized covariance matrices from each subsample.
        
        Returns:
            List[np.ndarray]: List of realized covariance matrices
        
        Raises:
            RuntimeError: If the estimator has not been fitted
        """
        if not self._fitted:
            raise RuntimeError("Estimator has not been fitted. Call fit() first.")
        
        return self._subsampled_measures
    
    def plot_subsample_variation(self, **kwargs: Any) -> Any:
        """Plot the variation in realized covariance estimates across subsamples.
        
        Args:
            **kwargs: Additional keyword arguments for plotting
        
        Returns:
            Any: Plot object
        
        Raises:
            RuntimeError: If the estimator has not been fitted
            ImportError: If matplotlib is not installed
        """
        if not self._fitted or not self._subsampled_measures:
            raise RuntimeError("Estimator has not been fitted. Call fit() first.")
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib is required for plotting")
        
        # Extract diagonal elements (variances) from each subsample
        n_subsamples = len(self._subsampled_measures)
        n_assets = self._n_assets
        
        # Create figure with one subplot per asset
        fig, axes = plt.subplots(n_assets, 1, figsize=(10, 3 * n_assets), sharex=True)
        
        # Handle case with only one asset
        if n_assets == 1:
            axes = [axes]
        
        # Plot variance estimates for each asset
        for i in range(n_assets):
            variances = [subsample[i, i] for subsample in self._subsampled_measures]
            axes[i].bar(range(1, n_subsamples + 1), variances)
            axes[i].axhline(y=self._realized_measure[i, i], color='r', linestyle='--', label='Average')
            axes[i].set_title(f"Asset {i+1} Variance Estimates Across Subsamples")
            axes[i].set_ylabel("Variance")
            axes[i].legend()
        
        # Set common x-axis label
        axes[-1].set_xlabel("Subsample Index")
        
        plt.tight_layout()
        
        return fig