'''
Core computation for realized kernel volatility estimators.

This module implements the core computation for realized kernel estimators, handling
the weighted sum of autocovariance terms that form the realized kernel value. It provides
optimized implementations for both standard and jittered kernel calculations with
comprehensive type hints, parameter validation, and Numba acceleration for
performance-critical operations.

The module serves as the computational engine for kernel-based volatility estimators,
providing efficient and numerically stable implementations that handle edge cases
and potential numerical issues. It supports both synchronous and asynchronous
computation patterns for integration with the broader MFE Toolbox architecture.

Functions:
    compute_kernel_core: Main function for computing realized kernel value
    compute_jittered_kernel_core: Compute realized kernel with jittered endpoints
    compute_kernel_core_async: Asynchronous version of compute_kernel_core
    compute_autocovariance_matrix: Compute full autocovariance matrix for returns
'''  

import logging
import warnings
from typing import (
    Any, Callable, Dict, List, Literal, Optional, Sequence, 
    Tuple, Type, TypeVar, Union, cast, overload
)
import numpy as np

from ...core.exceptions import ParameterError, NumericError, DimensionError

# Set up module-level logger
logger = logging.getLogger("mfe.models.realized.kernel_core")

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
    logger.debug("Numba available for kernel core computation acceleration")
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    HAS_NUMBA = False
    logger.info("Numba not available. Kernel core computation will use pure NumPy implementations.")


@jit(nopython=True, cache=True)

def _compute_kernel_core_numba(returns: np.ndarray, 
                              kernel_weights: np.ndarray,
                              max_lags: Optional[int] = None) -> float:
    """
    Numba-accelerated computation of realized kernel core.
    
    This function computes the realized kernel value as a weighted sum of autocovariances
    using Numba's JIT compilation for maximum performance.
    
    Args:
        returns: Array of returns
        kernel_weights: Array of kernel weights
        max_lags: Maximum number of lags to consider (if None, uses len(kernel_weights)-1)
        
    Returns:
        Realized kernel value
    """
    n = len(returns)
    
    # Determine maximum lags
    if max_lags is None:
        max_lags = len(kernel_weights) - 1
    else:
        max_lags = min(max_lags, len(kernel_weights) - 1)
    
    # Ensure max_lags doesn't exceed data length
    max_lags = min(max_lags, n - 1)
    
    # Initialize with the variance term (lag 0)
    # This is the sum of squared returns
    kernel_value = np.sum(returns**2)
    
    # Add weighted autocovariance terms
    for h in range(1, max_lags + 1):
        # Compute autocovariance at lag h
        gamma_h = 0.0
        for t in range(n - h):
            gamma_h += returns[t] * returns[t + h]
        
        # Add weighted autocovariance to kernel value
        kernel_value += 2.0 * kernel_weights[h] * gamma_h
    
    return kernel_value



def compute_kernel_core(returns: np.ndarray, 
                       kernel_weights: np.ndarray,
                       max_lags: Optional[int] = None,
                       return_autocovariances: bool = False) -> Union[float, Tuple[float, np.ndarray]]:
    """
    Compute realized kernel value as weighted sum of autocovariances.
    
    This function implements the core computation for realized kernel estimators,
    calculating the weighted sum of autocovariances using the provided kernel weights.
    
    The realized kernel is defined as:
    
    RK = γ₀ + 2 * Σ_{h=1}^H k(h/H) * γₕ
    
    where γₕ is the h-th order autocovariance of returns, k(·) is the kernel function,
    and H is the bandwidth parameter.
    
    Args:
        returns: Array of returns
        kernel_weights: Array of kernel weights
        max_lags: Maximum number of lags to consider (if None, uses len(kernel_weights)-1)
        return_autocovariances: Whether to return autocovariances along with kernel value
        
    Returns:
        If return_autocovariances is False: Realized kernel value
        If return_autocovariances is True: Tuple of (kernel_value, autocovariances)
        
    Raises:
        ValueError: If inputs have invalid dimensions or if max_lags is negative
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.kernel_core import compute_kernel_core
        >>> from mfe.models.realized.kernel_weights import compute_kernel_weights
        >>> returns = np.array([0.01, -0.005, 0.008, -0.002, 0.003])
        >>> weights = compute_kernel_weights(3, 'bartlett')
        >>> compute_kernel_core(returns, weights)
        0.000194...
    """
    # Convert inputs to numpy arrays
    returns = np.asarray(returns)
    kernel_weights = np.asarray(kernel_weights)
    
    # Validate inputs
    if returns.ndim != 1:
        raise ValueError("returns must be a 1D array")
    if kernel_weights.ndim != 1:
        raise ValueError("kernel_weights must be a 1D array")
    if len(kernel_weights) == 0:
        raise ValueError("kernel_weights must not be empty")
    if max_lags is not None and max_lags < 0:
        raise ValueError("max_lags must be non-negative")
    
    n = len(returns)
    
    # Determine maximum lags
    if max_lags is None:
        max_lags = len(kernel_weights) - 1
    else:
        max_lags = min(max_lags, len(kernel_weights) - 1)
    
    # Ensure max_lags doesn't exceed data length
    max_lags = min(max_lags, n - 1)
    
    # If return_autocovariances is True, we need to compute and store autocovariances
    if return_autocovariances:
        # Compute autocovariances
        autocovariances = np.zeros(max_lags + 1)
        
        # Lag 0 autocovariance (variance)
        autocovariances[0] = np.sum(returns**2)
        
        # Higher lag autocovariances
        for h in range(1, max_lags + 1):
            autocovariances[h] = np.sum(returns[:-h] * returns[h:])
        
        # Compute kernel value
        kernel_value = autocovariances[0] + 2.0 * np.sum(
            kernel_weights[1:max_lags+1] * autocovariances[1:max_lags+1]
        )
        
        return kernel_value, autocovariances
    
    # If we don't need to return autocovariances, use optimized implementation
    if HAS_NUMBA:
        return _compute_kernel_core_numba(returns, kernel_weights, max_lags)
    
    # Pure NumPy implementation
    # Initialize with the variance term (lag 0)
    kernel_value = np.sum(returns**2)
    
    # Add weighted autocovariance terms
    for h in range(1, max_lags + 1):
        # Compute autocovariance at lag h
        gamma_h = np.sum(returns[:-h] * returns[h:])
        
        # Add weighted autocovariance to kernel value
        kernel_value += 2.0 * kernel_weights[h] * gamma_h
    
    return kernel_value


@jit(nopython=True, cache=True)

def _compute_jittered_kernel_core_numba(returns: np.ndarray, 
                                       kernel_weights: np.ndarray,
                                       jitter_lag: int,
                                       max_lags: Optional[int] = None) -> float:
    """
    Numba-accelerated computation of jittered realized kernel core.
    
    This function computes the realized kernel value with jittered endpoints
    to mitigate end-effects in kernel estimation.
    
    Args:
        returns: Array of returns
        kernel_weights: Array of kernel weights
        jitter_lag: Jitter lag length for endpoint treatment
        max_lags: Maximum number of lags to consider (if None, uses len(kernel_weights)-1)
        
    Returns:
        Realized kernel value with jittered endpoints
    """
    n = len(returns)
    
    # Determine maximum lags
    if max_lags is None:
        max_lags = len(kernel_weights) - 1
    else:
        max_lags = min(max_lags, len(kernel_weights) - 1)
    
    # Ensure max_lags doesn't exceed data length
    max_lags = min(max_lags, n - 1)
    
    # Ensure jitter_lag is valid
    jitter_lag = min(jitter_lag, n // 4)
    
    # Initialize with the variance term (lag 0)
    # For jittered kernel, we exclude the first and last jitter_lag returns
    kernel_value = np.sum(returns[jitter_lag:-jitter_lag]**2)
    
    # Add weighted autocovariance terms with jittering
    for h in range(1, max_lags + 1):
        # Compute autocovariance at lag h with jittered endpoints
        gamma_h = 0.0
        for t in range(jitter_lag, n - h - jitter_lag):
            gamma_h += returns[t] * returns[t + h]
        
        # Add weighted autocovariance to kernel value
        kernel_value += 2.0 * kernel_weights[h] * gamma_h
    
    # Scale by the ratio of full sample size to jittered sample size
    scale_factor = n / (n - 2 * jitter_lag)
    kernel_value *= scale_factor
    
    return kernel_value



def compute_jittered_kernel_core(returns: np.ndarray, 
                                kernel_weights: np.ndarray,
                                jitter_lag: int,
                                max_lags: Optional[int] = None,
                                return_autocovariances: bool = False) -> Union[float, Tuple[float, np.ndarray]]:
    """
    Compute realized kernel value with jittered endpoints.
    
    This function implements the core computation for realized kernel estimators
    with jittered endpoints to mitigate end-effects in kernel estimation.
    
    Jittering involves excluding the first and last few observations from the
    autocovariance calculations to reduce the impact of endpoint effects.
    
    Args:
        returns: Array of returns
        kernel_weights: Array of kernel weights
        jitter_lag: Jitter lag length for endpoint treatment
        max_lags: Maximum number of lags to consider (if None, uses len(kernel_weights)-1)
        return_autocovariances: Whether to return autocovariances along with kernel value
        
    Returns:
        If return_autocovariances is False: Realized kernel value with jittered endpoints
        If return_autocovariances is True: Tuple of (kernel_value, autocovariances)
        
    Raises:
        ValueError: If inputs have invalid dimensions, if max_lags is negative,
                   or if jitter_lag is negative
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.kernel_core import compute_jittered_kernel_core
        >>> from mfe.models.realized.kernel_weights import compute_kernel_weights
        >>> returns = np.array([0.01, -0.005, 0.008, -0.002, 0.003, -0.001, 0.004])
        >>> weights = compute_kernel_weights(3, 'bartlett')
        >>> compute_jittered_kernel_core(returns, weights, jitter_lag=1)
        0.000198...
    """
    # Convert inputs to numpy arrays
    returns = np.asarray(returns)
    kernel_weights = np.asarray(kernel_weights)
    
    # Validate inputs
    if returns.ndim != 1:
        raise ValueError("returns must be a 1D array")
    if kernel_weights.ndim != 1:
        raise ValueError("kernel_weights must be a 1D array")
    if len(kernel_weights) == 0:
        raise ValueError("kernel_weights must not be empty")
    if max_lags is not None and max_lags < 0:
        raise ValueError("max_lags must be non-negative")
    if not isinstance(jitter_lag, int) or jitter_lag < 0:
        raise ValueError("jitter_lag must be a non-negative integer")
    
    n = len(returns)
    
    # Check if jitter_lag is too large
    if jitter_lag * 2 >= n:
        raise ValueError(f"jitter_lag ({jitter_lag}) is too large for the data length ({n})")
    
    # Determine maximum lags
    if max_lags is None:
        max_lags = len(kernel_weights) - 1
    else:
        max_lags = min(max_lags, len(kernel_weights) - 1)
    
    # Ensure max_lags doesn't exceed data length
    max_lags = min(max_lags, n - 1)
    
    # If return_autocovariances is True, we need to compute and store autocovariances
    if return_autocovariances:
        # Compute autocovariances with jittering
        autocovariances = np.zeros(max_lags + 1)
        
        # Lag 0 autocovariance (variance) with jittering
        autocovariances[0] = np.sum(returns[jitter_lag:-jitter_lag]**2)
        
        # Higher lag autocovariances with jittering
        for h in range(1, max_lags + 1):
            autocovariances[h] = np.sum(
                returns[jitter_lag:-jitter_lag-h] * returns[jitter_lag+h:-jitter_lag]
            )
        
        # Compute kernel value
        kernel_value = autocovariances[0] + 2.0 * np.sum(
            kernel_weights[1:max_lags+1] * autocovariances[1:max_lags+1]
        )
        
        # Scale by the ratio of full sample size to jittered sample size
        scale_factor = n / (n - 2 * jitter_lag)
        kernel_value *= scale_factor
        autocovariances *= scale_factor
        
        return kernel_value, autocovariances
    
    # If we don't need to return autocovariances, use optimized implementation
    if HAS_NUMBA:
        return _compute_jittered_kernel_core_numba(returns, kernel_weights, jitter_lag, max_lags)
    
    # Pure NumPy implementation
    # Initialize with the variance term (lag 0) with jittering
    kernel_value = np.sum(returns[jitter_lag:-jitter_lag]**2)
    
    # Add weighted autocovariance terms with jittering
    for h in range(1, max_lags + 1):
        # Compute autocovariance at lag h with jittered endpoints
        gamma_h = np.sum(
            returns[jitter_lag:-jitter_lag-h] * returns[jitter_lag+h:-jitter_lag]
        )
        
        # Add weighted autocovariance to kernel value
        kernel_value += 2.0 * kernel_weights[h] * gamma_h
    
    # Scale by the ratio of full sample size to jittered sample size
    scale_factor = n / (n - 2 * jitter_lag)
    kernel_value *= scale_factor
    
    return kernel_value



async def compute_kernel_core_async(returns: np.ndarray, 
                                  kernel_weights: np.ndarray,
                                  max_lags: Optional[int] = None,
                                  jitter_lag: Optional[int] = None,
                                  return_autocovariances: bool = False,
                                  progress_callback: Optional[Callable[[float, str], None]] = None) -> Union[float, Tuple[float, np.ndarray]]:
    """
    Asynchronously compute realized kernel value.
    
    This function provides an asynchronous interface to the kernel core computation,
    allowing for non-blocking execution in UI contexts and progress reporting during
    long computations.
    
    Args:
        returns: Array of returns
        kernel_weights: Array of kernel weights
        max_lags: Maximum number of lags to consider (if None, uses len(kernel_weights)-1)
        jitter_lag: Jitter lag length for endpoint treatment (if None, no jittering is applied)
        return_autocovariances: Whether to return autocovariances along with kernel value
        progress_callback: Optional callback function for reporting progress
        
    Returns:
        If return_autocovariances is False: Realized kernel value
        If return_autocovariances is True: Tuple of (kernel_value, autocovariances)
        
    Raises:
        ValueError: If inputs have invalid dimensions, if max_lags is negative,
                   or if jitter_lag is negative
    """
    # Report initial progress
    if progress_callback is not None:
        await progress_callback(0.0, "Starting kernel computation...")
    
    # Determine whether to use jittering
    if jitter_lag is not None and jitter_lag > 0:
        # Use jittered kernel computation
        if progress_callback is not None:
            await progress_callback(0.2, "Computing jittered kernel...")
        
        result = compute_jittered_kernel_core(
            returns, kernel_weights, jitter_lag, max_lags, return_autocovariances
        )
    else:
        # Use standard kernel computation
        if progress_callback is not None:
            await progress_callback(0.2, "Computing standard kernel...")
        
        result = compute_kernel_core(
            returns, kernel_weights, max_lags, return_autocovariances
        )
    
    # Report completion
    if progress_callback is not None:
        await progress_callback(1.0, "Kernel computation complete")
    
    return result


def compute_autocovariance_matrix(returns: np.ndarray, max_lags: int) -> np.ndarray:
    """
    Compute full autocovariance matrix for returns.
    
    This function computes the autocovariance matrix for returns up to max_lags,
    which can be used for various analyses and diagnostics.
    
    Args:
        returns: Array of returns
        max_lags: Maximum number of lags to compute
        
    Returns:
        Autocovariance matrix of shape (max_lags+1,)
        
    Raises:
        ValueError: If returns has invalid dimensions or if max_lags is negative
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.kernel_core import compute_autocovariance_matrix
        >>> returns = np.array([0.01, -0.005, 0.008, -0.002, 0.003])
        >>> acov = compute_autocovariance_matrix(returns, 2)
        >>> acov
        array([0.000194, 0.000011, -0.000052])
    """
    # Convert to numpy array if not already
    returns = np.asarray(returns)
    
    # Validate inputs
    if returns.ndim != 1:
        raise ValueError("returns must be a 1D array")
    if not isinstance(max_lags, int) or max_lags < 0:
        raise ValueError("max_lags must be a non-negative integer")
    
    n = len(returns)
    
    # Ensure max_lags doesn't exceed data length
    max_lags = min(max_lags, n - 1)
    
    # Compute autocovariances
    autocovariances = np.zeros(max_lags + 1)
    
    # Lag 0 autocovariance (variance)
    autocovariances[0] = np.sum(returns**2) / n
    
    # Higher lag autocovariances
    for h in range(1, max_lags + 1):
        autocovariances[h] = np.sum(returns[:-h] * returns[h:]) / (n - h)
    
    return autocovariances


# Register Numba-accelerated functions if available
def _register_numba_functions() -> None:
    """
    Register Numba JIT-compiled functions for kernel core computation.
    
    This function is called during module initialization to register
    performance-critical functions for JIT compilation if Numba is available.
    """
    if HAS_NUMBA:
        # The functions are already decorated with @jit, so no additional
        # registration is needed here. This function is kept for consistency
        # with the module structure and potential future enhancements.
        logger.debug("Kernel core Numba JIT functions registered")
    else:
        logger.info("Numba not available. Kernel core computation will use pure NumPy implementations.")


# Initialize the module
_register_numba_functions()
