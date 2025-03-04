# mfe/utils/covariance.py

"""
Covariance Estimation Module

This module provides robust covariance estimation methods for time series data,
including Newey-West estimator, VAR-based estimators, and robust variance-covariance
calculation for parameter inference. These functions are essential for accurate
statistical inference in the presence of heteroskedasticity and autocorrelation.

The module implements optimized versions of covariance estimators using NumPy's
efficient array operations and Numba's JIT compilation for performance-critical
functions. All functions include comprehensive type hints and input validation
to ensure reliability and proper error handling.

Functions:
    covnw: Newey-West covariance estimator for time series
    covvar: VAR-based covariance estimator
    robustvcv: Robust variance-covariance matrix for parameter inference
    kernel_weight: Compute kernel weights for covariance estimation
    kernel_optimal_bandwidth: Determine optimal bandwidth for kernel estimators
"""

import logging
import warnings
from typing import Callable, Dict, Optional, Tuple, Union, cast

import numpy as np
from scipy import linalg

from mfe.core.types import (
    Matrix, Vector, CovarianceMatrix, PositiveDefiniteMatrix
)
from mfe.core.exceptions import (
    DimensionError, NumericError, raise_dimension_error,
    raise_numeric_error, warn_numeric
)

# Set up module-level logger
logger = logging.getLogger("mfe.utils.covariance")

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
    logger.debug("Numba available for covariance estimation acceleration")
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    HAS_NUMBA = False
    logger.info("Numba not available. Covariance estimators will use pure NumPy implementations.")


@jit(nopython=True, cache=True)
def _covnw_core(x: np.ndarray, lags: int, weights: np.ndarray) -> np.ndarray:
    """
    Numba-accelerated core implementation of Newey-West covariance estimator.

    Args:
        x: Data matrix (T x K)
        lags: Number of lags to include
        weights: Kernel weights for each lag

    Returns:
        Newey-West covariance matrix estimate
    """
    T, K = x.shape

    # Compute the base covariance matrix (X'X / T)
    cov = np.zeros((K, K))
    for i in range(T):
        xi = x[i]
        for j in range(K):
            for k in range(K):
                cov[j, k] += xi[j] * xi[k]
    cov = cov / T

    # Add weighted autocovariance terms
    for lag in range(1, lags + 1):
        weight = weights[lag - 1]

        # Compute autocovariance for this lag
        acov = np.zeros((K, K))
        for t in range(lag, T):
            xt = x[t]
            xt_lag = x[t - lag]
            for j in range(K):
                for k in range(K):
                    acov[j, k] += xt[j] * xt_lag[k]

        # Scale and weight the autocovariance
        acov = acov / T

        # Add to the covariance matrix (symmetrically)
        for j in range(K):
            for k in range(K):
                cov[j, k] += weight * (acov[j, k] + acov[k, j])

    return cov


def covnw(x: Matrix, lags: Optional[int] = None,
          kernel: str = "bartlett", bandwidth: Optional[float] = None) -> CovarianceMatrix:
    """
    Compute Newey-West heteroskedasticity and autocorrelation consistent (HAC) 
    covariance matrix estimator.

    This function computes the Newey-West HAC covariance matrix estimator for time
    series data, which is robust to heteroskedasticity and autocorrelation. The
    estimator includes weighted autocovariance terms up to the specified lag order.

    Args:
        x: Data matrix (T x K) where T is the number of observations and K is the
           number of variables
        lags: Number of lags to include in the estimator. If None, a default value
              is chosen based on the sample size: floor(4 * (T/100)^(2/9))
        kernel: Kernel function to use for weighting the autocovariances.
                Options are "bartlett" (default), "parzen", "quadratic-spectral"
        bandwidth: Bandwidth parameter for the kernel. If None, an optimal value
                  is chosen based on the kernel type and sample size

    Returns:
        Newey-West HAC covariance matrix estimate

    Raises:
        DimensionError: If x is not a 2D array
        ValueError: If kernel is not one of the supported types

    Examples:
        >>> import numpy as np
        >>> from mfe.utils.covariance import covnw
        >>> np.random.seed(123)
        >>> x = np.random.randn(100, 3)
        >>> cov_nw = covnw(x, lags=5)
    """
    # Convert to numpy array if not already
    x = np.asarray(x)

    # Check if x is 2D
    if x.ndim != 2:
        raise_dimension_error(
            "Input must be a 2D array",
            array_name="x",
            expected_shape="(T, K)",
            actual_shape=x.shape
        )

    T, K = x.shape

    # Determine the number of lags if not provided
    if lags is None:
        # Default rule: floor(4 * (T/100)^(2/9))
        lags = int(np.floor(4 * (T / 100) ** (2 / 9)))
        logger.debug(f"Using default lag order: {lags}")

    # Ensure lags is not too large
    lags = min(lags, T - 1)

    # Compute kernel weights
    weights = kernel_weight(lags, kernel, bandwidth)

    # Use Numba-accelerated implementation if available
    if HAS_NUMBA:
        return _covnw_core(x, lags, weights)

    # Pure NumPy implementation
    # Compute the base covariance matrix (X'X / T)
    cov = x.T @ x / T

    # Add weighted autocovariance terms
    for lag in range(1, lags + 1):
        weight = weights[lag - 1]

        # Compute autocovariance for this lag
        x_lag = x[lag:, :]
        x_early = x[:-lag, :]
        acov = x_lag.T @ x_early / T

        # Add to the covariance matrix (symmetrically)
        cov += weight * (acov + acov.T)

    # Ensure the result is symmetric (to handle numerical precision issues)
    cov = (cov + cov.T) / 2

    return cov


@jit(nopython=True, cache=True)
def _covvar_core(x: np.ndarray, lags: int) -> np.ndarray:
    """
    Numba-accelerated core implementation of VAR-based covariance estimator.

    Args:
        x: Data matrix (T x K)
        lags: Number of lags to include in the VAR model

    Returns:
        VAR-based covariance matrix estimate
    """
    T, K = x.shape

    # Compute the base covariance matrix (X'X / T)
    cov = np.zeros((K, K))
    for i in range(T):
        xi = x[i]
        for j in range(K):
            for k in range(K):
                cov[j, k] += xi[j] * xi[k]
    cov = cov / T

    # Compute VAR coefficients for each lag
    var_coefs = np.zeros((lags, K, K))

    # Create lagged data matrices
    x_lagged = np.zeros((T - lags, lags * K))
    for i in range(lags):
        x_lagged[:, i*K:(i+1)*K] = x[lags-i-1:T-i-1, :]

    # Dependent variable
    y = x[lags:, :]

    # Compute VAR coefficients using OLS
    # (X'X)^(-1) X'Y for each column of Y
    xtx = x_lagged.T @ x_lagged
    xtx_inv = np.linalg.inv(xtx)

    for k in range(K):
        beta = xtx_inv @ x_lagged.T @ y[:, k]
        for i in range(lags):
            var_coefs[i, k, :] = beta[i*K:(i+1)*K]

    # Compute the long-run covariance matrix
    # I - A(1) where A(1) = sum of VAR coefficient matrices
    a1 = np.zeros((K, K))
    for i in range(lags):
        a1 += var_coefs[i]

    i_minus_a1 = np.eye(K) - a1
    i_minus_a1_inv = np.linalg.inv(i_minus_a1)

    # Long-run covariance = (I - A(1))^(-1) * Sigma * (I - A(1))^(-1)'
    lrc = i_minus_a1_inv @ cov @ i_minus_a1_inv.T

    return lrc


def covvar(x: Matrix, lags: int = 1) -> CovarianceMatrix:
    """
    Compute VAR-based long-run covariance matrix estimator.

    This function computes a long-run covariance matrix estimator based on a
    Vector Autoregression (VAR) model. It is useful for capturing persistent
    autocorrelation in multivariate time series.

    Args:
        x: Data matrix (T x K) where T is the number of observations and K is the
           number of variables
        lags: Number of lags to include in the VAR model

    Returns:
        VAR-based long-run covariance matrix estimate

    Raises:
        DimensionError: If x is not a 2D array
        ValueError: If lags is not a positive integer

    Examples:
        >>> import numpy as np
        >>> from mfe.utils.covariance import covvar
        >>> np.random.seed(123)
        >>> x = np.random.randn(100, 3)
        >>> cov_var = covvar(x, lags=2)
    """
    # Convert to numpy array if not already
    x = np.asarray(x)

    # Check if x is 2D
    if x.ndim != 2:
        raise_dimension_error(
            "Input must be a 2D array",
            array_name="x",
            expected_shape="(T, K)",
            actual_shape=x.shape
        )

    # Check if lags is valid
    if not isinstance(lags, int) or lags < 1:
        raise ValueError("lags must be a positive integer")

    T, K = x.shape

    # Ensure lags is not too large
    if lags >= T:
        raise ValueError(f"lags ({lags}) must be less than the number of observations ({T})")

    # Use Numba-accelerated implementation if available
    if HAS_NUMBA:
        return _covvar_core(x, lags)

    # Pure NumPy implementation
    # Compute the base covariance matrix (X'X / T)
    cov = x.T @ x / T

    # Create lagged data matrices
    x_lagged = np.zeros((T - lags, lags * K))
    for i in range(lags):
        x_lagged[:, i*K:(i+1)*K] = x[lags-i-1:T-i-1, :]

    # Dependent variable
    y = x[lags:, :]

    # Compute VAR coefficients using OLS
    # (X'X)^(-1) X'Y for each column of Y
    try:
        xtx = x_lagged.T @ x_lagged
        xtx_inv = linalg.inv(xtx)

        var_coefs = np.zeros((lags, K, K))
        for k in range(K):
            beta = xtx_inv @ x_lagged.T @ y[:, k]
            for i in range(lags):
                var_coefs[i, k, :] = beta[i*K:(i+1)*K]

        # Compute the long-run covariance matrix
        # I - A(1) where A(1) = sum of VAR coefficient matrices
        a1 = np.sum(var_coefs, axis=0)
        i_minus_a1 = np.eye(K) - a1
        i_minus_a1_inv = linalg.inv(i_minus_a1)

        # Long-run covariance = (I - A(1))^(-1) * Sigma * (I - A(1))^(-1)'
        lrc = i_minus_a1_inv @ cov @ i_minus_a1_inv.T

        # Ensure the result is symmetric (to handle numerical precision issues)
        lrc = (lrc + lrc.T) / 2

        return lrc

    except linalg.LinAlgError as e:
        # Handle potential numerical issues
        raise_numeric_error(
            "Failed to compute VAR-based covariance matrix due to linear algebra error",
            operation="covvar",
            error_type="singular_matrix",
            details=str(e)
        )


def robustvcv(scores: Matrix, jacobian: Matrix,
              method: str = "white", lags: Optional[int] = None) -> CovarianceMatrix:
    """
    Compute robust variance-covariance matrix for parameter inference.

    This function computes a robust variance-covariance matrix for parameter
    estimates based on score contributions and the Jacobian matrix. It supports
    various methods including White's heteroskedasticity-consistent estimator
    and Newey-West HAC estimator.

    Args:
        scores: Score matrix (T x K) where each row is a score contribution
                from one observation and each column corresponds to a parameter
        jacobian: Jacobian matrix (K x K) of the parameter transformation
        method: Method to use for computing the covariance matrix.
                Options are "white" (default), "newey-west", "var"
        lags: Number of lags to include when using "newey-west" or "var" methods.
              If None, an default value is chosen based on the sample size

    Returns:
        Robust variance-covariance matrix

    Raises:
        DimensionError: If scores or jacobian have incorrect dimensions
        ValueError: If method is not one of the supported types

    Examples:
        >>> import numpy as np
        >>> from mfe.utils.covariance import robustvcv
        >>> np.random.seed(123)
        >>> scores = np.random.randn(100, 3)
        >>> jacobian = np.eye(3)
        >>> vcv = robustvcv(scores, jacobian, method="newey-west", lags=5)
    """
    # Convert to numpy arrays if not already
    scores = np.asarray(scores)
    jacobian = np.asarray(jacobian)

    # Check if scores is 2D
    if scores.ndim != 2:
        raise_dimension_error(
            "scores must be a 2D array",
            array_name="scores",
            expected_shape="(T, K)",
            actual_shape=scores.shape
        )

    # Check if jacobian is 2D
    if jacobian.ndim != 2:
        raise_dimension_error(
            "jacobian must be a 2D array",
            array_name="jacobian",
            expected_shape="(K, K)",
            actual_shape=jacobian.shape
        )

    T, K = scores.shape

    # Check if jacobian has correct dimensions
    if (jacobian.shape) != (K, K):
        raise_dimension_error(
            "jacobian must have shape (K, K) where K is the number of parameters",
            array_name="jacobian",
            expected_shape=f"({K}, {K})",
            actual_shape=jacobian.shape
        )

    # Compute the middle term based on the specified method
    if method.lower() == "white":
        # White's heteroskedasticity-consistent estimator
        # S = (1/T) * sum(s_t * s_t')
        S = scores.T @ scores / T

    elif method.lower() == "newey-west":
        # Newey-West HAC estimator
        S = covnw(scores, lags=lags)

    elif method.lower() == "var":
        # VAR-based long-run covariance estimator
        S = covvar(scores, lags=lags if lags is not None else 1)

    else:
        raise ValueError(f"Unknown method: {method}. Supported methods are 'white', 'newey-west', 'var'")

    # Compute the robust variance-covariance matrix
    # VCV = (J'J)^(-1) * J' * S * J * (J'J)^(-1)
    # For symmetric J, this simplifies to J^(-1) * S * J^(-1)'
    try:
        # Check if jacobian is invertible
        jj = jacobian.T @ jacobian
        jj_inv = linalg.inv(jj)

        # Compute the robust VCV
        vcv = jj_inv @ jacobian.T @ S @ jacobian @ jj_inv

        # Ensure the result is symmetric (to handle numerical precision issues)
        vcv = (vcv + vcv.T) / 2

        return vcv

    except linalg.LinAlgError as e:
        # Handle potential numerical issues
        raise_numeric_error(
            "Failed to compute robust variance-covariance matrix due to linear algebra error",
            operation="robustvcv",
            error_type="singular_jacobian",
            details=str(e)
        )


def kernel_weight(lags: int, kernel: str = "bartlett",
                  bandwidth: Optional[float] = None) -> Vector:
    """
    Compute kernel weights for covariance estimation.

    This function computes weights for autocovariance terms in HAC covariance
    estimators based on various kernel functions.

    Args:
        lags: Number of lags to compute weights for
        kernel: Kernel function to use for weighting.
                Options are "bartlett" (default), "parzen", "quadratic-spectral"
        bandwidth: Bandwidth parameter for the kernel. If None, an optimal value
                  is chosen based on the kernel type

    Returns:
        Vector of weights for each lag (length = lags)

    Raises:
        ValueError: If kernel is not one of the supported types

    Examples:
        >>> from mfe.utils.covariance import kernel_weight
        >>> weights = kernel_weight(5, kernel="bartlett")
        >>> weights
        array([0.8, 0.6, 0.4, 0.2, 0. ])
    """
    # Check if lags is valid
    if not isinstance(lags, int) or lags < 1:
        raise ValueError("lags must be a positive integer")

    # Determine bandwidth if not provided
    if bandwidth is None:
        bandwidth = kernel_optimal_bandwidth(lags, kernel)

    # Compute weights based on the specified kernel
    weights = np.zeros(lags)

    if kernel.lower() == "bartlett":
        # Bartlett kernel: w(j) = 1 - j/(m+1) for j <= m, 0 otherwise
        for j in range(1, lags + 1):
            if j <= bandwidth:
                weights[j - 1] = 1 - j / (bandwidth + 1)

    elif kernel.lower() == "parzen":
        # Parzen kernel
        for j in range(1, lags + 1):
            q = j / bandwidth
            if q <= 0.5:
                weights[j - 1] = 1 - 6 * q**2 + 6 * q**3
            elif q <= 1:
                weights[j - 1] = 2 * (1 - q)**3

    elif kernel.lower() == "quadratic-spectral" or kernel.lower() == "qs":
        # Quadratic Spectral kernel
        for j in range(1, lags + 1):
            q = 6 * np.pi * j / (5 * bandwidth)
            if j == 0:
                weights[j - 1] = 1
            else:
                weights[j - 1] = 3 * (np.sin(q) / q - np.cos(q)) / q**2

    else:
        raise ValueError(f"Unknown kernel: {kernel}. Supported kernels are 'bartlett', 'parzen', 'quadratic-spectral'")

    return weights


def kernel_optimal_bandwidth(sample_size: int, kernel: str = "bartlett") -> float:
    """
    Determine optimal bandwidth for kernel estimators.

    This function computes an optimal bandwidth parameter for kernel-based
    covariance estimators based on the sample size and kernel type.

    Args:
        sample_size: Number of observations in the sample
        kernel: Kernel function type.
                Options are "bartlett" (default), "parzen", "quadratic-spectral"

    Returns:
        Optimal bandwidth parameter

    Raises:
        ValueError: If kernel is not one of the supported types

    Examples:
        >>> from mfe.utils.covariance import kernel_optimal_bandwidth
        >>> bandwidth = kernel_optimal_bandwidth(100, kernel="bartlett")
        >>> bandwidth
        5.0
    """
    # Check if sample_size is valid
    if not isinstance(sample_size, int) or sample_size < 2:
        raise ValueError("sample_size must be an integer greater than 1")

    # Compute optimal bandwidth based on the kernel type
    if kernel.lower() == "bartlett":
        # Optimal bandwidth for Bartlett kernel: floor(4 * (T/100)^(2/9))
        bandwidth = np.floor(4 * (sample_size / 100) ** (2 / 9))

    elif kernel.lower() == "parzen":
        # Optimal bandwidth for Parzen kernel: floor(4 * (T/100)^(4/25))
        bandwidth = np.floor(4 * (sample_size / 100) ** (4 / 25))

    elif kernel.lower() == "quadratic-spectral" or kernel.lower() == "qs":
        # Optimal bandwidth for Quadratic Spectral kernel: 1.3 * (T/100)^(2/25)
        bandwidth = 1.3 * (sample_size / 100) ** (2 / 25)

    else:
        raise ValueError(f"Unknown kernel: {kernel}. Supported kernels are 'bartlett', 'parzen', 'quadratic-spectral'")

    # Ensure bandwidth is at least 1
    bandwidth = max(1, bandwidth)

    return bandwidth


# Register Numba-accelerated functions if available
def _register_numba_functions() -> None:
    """
    Register Numba JIT-compiled functions for covariance estimation.

    This function is called during module initialization to register
    performance-critical functions for JIT compilation if Numba is available.
    """
    if HAS_NUMBA:
        # The functions are already decorated with @jit, so no additional
        # registration is needed here. This function is kept for consistency
        # with the module structure and potential future enhancements.
        logger.debug("Covariance estimation Numba JIT functions registered")
    else:
        logger.info("Numba not available. Covariance estimators will use pure NumPy implementations.")


# Initialize the module
_register_numba_functions()
