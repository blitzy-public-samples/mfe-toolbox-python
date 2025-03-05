# mfe/models/realized/kernel_weights.py
"""
Kernel weight computation for realized kernel volatility estimators.

This module provides functions for computing weight vectors for realized kernels
based on kernel type and bandwidth. These weights are crucial for the accuracy of
kernel-based volatility estimators, as they determine how autocovariances at different
lags are weighted in the realized kernel estimator.

The module implements various kernel types including Bartlett, Parzen, Tukey-Hanning,
Quadratic, and Flat-Top kernels, with optimized computation using NumPy's vectorized
operations and Numba acceleration for performance-critical calculations.

Functions:
    compute_kernel_weights: Main function for computing kernel weights
    compute_bartlett_weights: Compute weights for Bartlett kernel
    compute_parzen_weights: Compute weights for Parzen kernel
    compute_tukey_hanning_weights: Compute weights for Tukey-Hanning kernel
    compute_quadratic_weights: Compute weights for Quadratic kernel
    compute_flat_top_weights: Compute weights for Flat-Top kernel
    plot_kernel_weights: Visualize kernel weight functions
"""

import logging
import warnings
from typing import (
    Any, Callable, Dict, List, Literal, Optional, Sequence, 
    Tuple, Type, TypeVar, Union, cast, overload
)
import numpy as np

from ...core.exceptions import ParameterError

# Set up module-level logger
logger = logging.getLogger("mfe.models.realized.kernel_weights")

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
    logger.debug("Numba available for kernel weights computation acceleration")
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    HAS_NUMBA = False
    logger.info("Numba not available. Kernel weights computation will use pure NumPy implementations.")


@jit(nopython=True, cache=True)
def _compute_bartlett_weights_numba(n: int, bandwidth: float) -> np.ndarray:
    """
    Numba-accelerated computation of Bartlett kernel weights.
    
    The Bartlett kernel is a linear kernel that decreases from 1 at lag 0 to 0 at lag H+1.
    
    Args:
        n: Number of weights to compute (including lag 0)
        bandwidth: Bandwidth parameter (H)
        
    Returns:
        Array of Bartlett kernel weights
    """
    weights = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        x = i / bandwidth
        if x <= 1.0:
            weights[i] = 1.0 - x
        else:
            weights[i] = 0.0
    
    return weights


def compute_bartlett_weights(n: int, bandwidth: float) -> np.ndarray:
    """
    Compute weights for Bartlett kernel.
    
    The Bartlett kernel is a linear kernel that decreases from 1 at lag 0 to 0 at lag H+1.
    
    k(x) = 1 - x  for 0 ≤ x ≤ 1
    k(x) = 0      for x > 1
    
    where x = h/H, h is the lag, and H is the bandwidth parameter.
    
    Args:
        n: Number of weights to compute (including lag 0)
        bandwidth: Bandwidth parameter (H)
        
    Returns:
        Array of Bartlett kernel weights
        
    Raises:
        ValueError: If n is not positive or if bandwidth is not positive
        
    Examples:
        >>> from mfe.models.realized.kernel_weights import compute_bartlett_weights
        >>> weights = compute_bartlett_weights(5, 4)
        >>> weights
        array([1.   , 0.75 , 0.5  , 0.25 , 0.   ])
    """
    # Validate inputs
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    
    # Use Numba-accelerated implementation if available
    if HAS_NUMBA:
        return _compute_bartlett_weights_numba(n, bandwidth)
    
    # Pure NumPy implementation
    lags = np.arange(n)
    x = lags / bandwidth
    weights = np.where(x <= 1.0, 1.0 - x, 0.0)
    
    return weights


@jit(nopython=True, cache=True)
def _compute_parzen_weights_numba(n: int, bandwidth: float) -> np.ndarray:
    """
    Numba-accelerated computation of Parzen kernel weights.
    
    The Parzen kernel is a smooth kernel that provides better bias-variance tradeoff
    than the Bartlett kernel.
    
    Args:
        n: Number of weights to compute (including lag 0)
        bandwidth: Bandwidth parameter (H)
        
    Returns:
        Array of Parzen kernel weights
    """
    weights = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        x = i / bandwidth
        if x <= 0.5:
            weights[i] = 1.0 - 6.0 * x**2 + 6.0 * x**3
        elif x <= 1.0:
            weights[i] = 2.0 * (1.0 - x)**3
        else:
            weights[i] = 0.0
    
    return weights


def compute_parzen_weights(n: int, bandwidth: float) -> np.ndarray:
    """
    Compute weights for Parzen kernel.
    
    The Parzen kernel is a smooth kernel that provides better bias-variance tradeoff
    than the Bartlett kernel.
    
    k(x) = 1 - 6x^2 + 6x^3  for 0 ≤ x ≤ 0.5
    k(x) = 2(1 - x)^3       for 0.5 < x ≤ 1
    k(x) = 0                for x > 1
    
    where x = h/H, h is the lag, and H is the bandwidth parameter.
    
    Args:
        n: Number of weights to compute (including lag 0)
        bandwidth: Bandwidth parameter (H)
        
    Returns:
        Array of Parzen kernel weights
        
    Raises:
        ValueError: If n is not positive or if bandwidth is not positive
        
    Examples:
        >>> from mfe.models.realized.kernel_weights import compute_parzen_weights
        >>> weights = compute_parzen_weights(5, 4)
        >>> weights
        array([1.        , 0.84375   , 0.5       , 0.15625   , 0.        ])
    """
    # Validate inputs
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    
    # Use Numba-accelerated implementation if available
    if HAS_NUMBA:
        return _compute_parzen_weights_numba(n, bandwidth)
    
    # Pure NumPy implementation
    lags = np.arange(n)
    x = lags / bandwidth
    
    # Initialize weights array
    weights = np.zeros(n)
    
    # Apply Parzen kernel formula
    mask1 = x <= 0.5
    mask2 = (x > 0.5) & (x <= 1.0)
    
    weights[mask1] = 1.0 - 6.0 * x[mask1]**2 + 6.0 * x[mask1]**3
    weights[mask2] = 2.0 * (1.0 - x[mask2])**3
    
    return weights


@jit(nopython=True, cache=True)
def _compute_tukey_hanning_weights_numba(n: int, bandwidth: float) -> np.ndarray:
    """
    Numba-accelerated computation of Tukey-Hanning kernel weights.
    
    The Tukey-Hanning kernel is a smooth kernel based on the cosine function.
    
    Args:
        n: Number of weights to compute (including lag 0)
        bandwidth: Bandwidth parameter (H)
        
    Returns:
        Array of Tukey-Hanning kernel weights
    """
    weights = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        x = i / bandwidth
        if x <= 1.0:
            weights[i] = 0.5 * (1.0 + np.cos(np.pi * x))
        else:
            weights[i] = 0.0
    
    return weights


def compute_tukey_hanning_weights(n: int, bandwidth: float) -> np.ndarray:
    """
    Compute weights for Tukey-Hanning kernel.
    
    The Tukey-Hanning kernel is a smooth kernel based on the cosine function.
    
    k(x) = 0.5 * (1 + cos(πx))  for 0 ≤ x ≤ 1
    k(x) = 0                    for x > 1
    
    where x = h/H, h is the lag, and H is the bandwidth parameter.
    
    Args:
        n: Number of weights to compute (including lag 0)
        bandwidth: Bandwidth parameter (H)
        
    Returns:
        Array of Tukey-Hanning kernel weights
        
    Raises:
        ValueError: If n is not positive or if bandwidth is not positive
        
    Examples:
        >>> from mfe.models.realized.kernel_weights import compute_tukey_hanning_weights
        >>> weights = compute_tukey_hanning_weights(5, 4)
        >>> weights
        array([1.        , 0.85355339, 0.5       , 0.14644661, 0.        ])
    """
    # Validate inputs
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    
    # Use Numba-accelerated implementation if available
    if HAS_NUMBA:
        return _compute_tukey_hanning_weights_numba(n, bandwidth)
    
    # Pure NumPy implementation
    lags = np.arange(n)
    x = lags / bandwidth
    
    # Apply Tukey-Hanning kernel formula
    weights = np.where(x <= 1.0, 0.5 * (1.0 + np.cos(np.pi * x)), 0.0)
    
    return weights


@jit(nopython=True, cache=True)
def _compute_quadratic_weights_numba(n: int, bandwidth: float) -> np.ndarray:
    """
    Numba-accelerated computation of Quadratic kernel weights.
    
    The Quadratic kernel is a smooth kernel that provides good bias-variance tradeoff.
    
    Args:
        n: Number of weights to compute (including lag 0)
        bandwidth: Bandwidth parameter (H)
        
    Returns:
        Array of Quadratic kernel weights
    """
    weights = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        x = i / bandwidth
        if x <= 1.0:
            weights[i] = (1.0 - x**2)**2
        else:
            weights[i] = 0.0
    
    return weights


def compute_quadratic_weights(n: int, bandwidth: float) -> np.ndarray:
    """
    Compute weights for Quadratic kernel.
    
    The Quadratic kernel is a smooth kernel that provides good bias-variance tradeoff.
    
    k(x) = (1 - x^2)^2  for 0 ≤ x ≤ 1
    k(x) = 0            for x > 1
    
    where x = h/H, h is the lag, and H is the bandwidth parameter.
    
    Args:
        n: Number of weights to compute (including lag 0)
        bandwidth: Bandwidth parameter (H)
        
    Returns:
        Array of Quadratic kernel weights
        
    Raises:
        ValueError: If n is not positive or if bandwidth is not positive
        
    Examples:
        >>> from mfe.models.realized.kernel_weights import compute_quadratic_weights
        >>> weights = compute_quadratic_weights(5, 4)
        >>> weights
        array([1.        , 0.76562   , 0.39062   , 0.10352   , 0.        ])
    """
    # Validate inputs
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    
    # Use Numba-accelerated implementation if available
    if HAS_NUMBA:
        return _compute_quadratic_weights_numba(n, bandwidth)
    
    # Pure NumPy implementation
    lags = np.arange(n)
    x = lags / bandwidth
    
    # Apply Quadratic kernel formula
    weights = np.where(x <= 1.0, (1.0 - x**2)**2, 0.0)
    
    return weights


@jit(nopython=True, cache=True)
def _compute_flat_top_weights_numba(n: int, bandwidth: float) -> np.ndarray:
    """
    Numba-accelerated computation of Flat-Top kernel weights.
    
    The Flat-Top kernel is designed to minimize asymptotic bias.
    
    Args:
        n: Number of weights to compute (including lag 0)
        bandwidth: Bandwidth parameter (H)
        
    Returns:
        Array of Flat-Top kernel weights
    """
    weights = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        x = i / bandwidth
        if x <= 0.1:
            weights[i] = 1.0
        elif x <= 1.1:
            weights[i] = 1.1 - x
        else:
            weights[i] = 0.0
    
    return weights


def compute_flat_top_weights(n: int, bandwidth: float) -> np.ndarray:
    """
    Compute weights for Flat-Top kernel.
    
    The Flat-Top kernel is designed to minimize asymptotic bias.
    
    k(x) = 1                    for 0 ≤ x ≤ 0.1
    k(x) = 1.1 - x              for 0.1 < x ≤ 1.1
    k(x) = 0                    for x > 1.1
    
    where x = h/H, h is the lag, and H is the bandwidth parameter.
    
    Args:
        n: Number of weights to compute (including lag 0)
        bandwidth: Bandwidth parameter (H)
        
    Returns:
        Array of Flat-Top kernel weights
        
    Raises:
        ValueError: If n is not positive or if bandwidth is not positive
        
    Examples:
        >>> from mfe.models.realized.kernel_weights import compute_flat_top_weights
        >>> weights = compute_flat_top_weights(5, 4)
        >>> weights
        array([1.   , 1.   , 0.6  , 0.35 , 0.1  ])
    """
    # Validate inputs
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    
    # Use Numba-accelerated implementation if available
    if HAS_NUMBA:
        return _compute_flat_top_weights_numba(n, bandwidth)
    
    # Pure NumPy implementation
    lags = np.arange(n)
    x = lags / bandwidth
    
    # Initialize weights array
    weights = np.zeros(n)
    
    # Apply Flat-Top kernel formula
    mask1 = x <= 0.1
    mask2 = (x > 0.1) & (x <= 1.1)
    
    weights[mask1] = 1.0
    weights[mask2] = 1.1 - x[mask2]
    
    return weights


# Dictionary mapping kernel types to weight computation functions
KERNEL_WEIGHT_FUNCTIONS = {
    'bartlett': compute_bartlett_weights,
    'parzen': compute_parzen_weights,
    'tukey-hanning': compute_tukey_hanning_weights,
    'tukey': compute_tukey_hanning_weights,
    'hanning': compute_tukey_hanning_weights,
    'quadratic': compute_quadratic_weights,
    'flat-top': compute_flat_top_weights
}


def compute_kernel_weights(n: int, kernel_type: str = 'bartlett', 
                         bandwidth: Optional[float] = None) -> np.ndarray:
    """
    Compute weights for kernel-based estimators.
    
    This is the main function for computing kernel weights, which dispatches to the
    appropriate kernel-specific function based on the specified kernel type.
    
    Args:
        n: Number of weights to compute (including lag 0)
        kernel_type: Type of kernel function ('bartlett', 'parzen', 'tukey-hanning', 
                    'tukey', 'hanning', 'quadratic', 'flat-top')
        bandwidth: Bandwidth parameter (if None, defaults to sqrt(n))
        
    Returns:
        Array of kernel weights
        
    Raises:
        ValueError: If n is not positive, if bandwidth is not positive,
                   or if kernel_type is not recognized
        
    Examples:
        >>> from mfe.models.realized.kernel_weights import compute_kernel_weights
        >>> weights = compute_kernel_weights(5, 'bartlett', 4)
        >>> weights
        array([1.   , 0.75 , 0.5  , 0.25 , 0.   ])
        
        >>> weights = compute_kernel_weights(5, 'parzen')
        >>> weights
        array([1.        , 0.84375   , 0.5       , 0.15625   , 0.        ])
    """
    # Validate inputs
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    
    # Set default bandwidth if not provided
    if bandwidth is None:
        bandwidth = np.sqrt(n)
    elif bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    
    # Normalize kernel type to lowercase
    kernel_type_lower = kernel_type.lower()
    
    # Check if kernel type is recognized
    if kernel_type_lower not in KERNEL_WEIGHT_FUNCTIONS:
        valid_kernels = list(KERNEL_WEIGHT_FUNCTIONS.keys())
        raise ValueError(f"Unrecognized kernel type: {kernel_type}. "
                         f"Supported types are {valid_kernels}.")
    
    # Dispatch to appropriate kernel weight function
    weight_function = KERNEL_WEIGHT_FUNCTIONS[kernel_type_lower]
    weights = weight_function(n, bandwidth)
    
    return weights


def plot_kernel_weights(kernel_type: str = 'bartlett', 
                      bandwidth: float = 10.0, 
                      n: int = 20,
                      figsize: Tuple[int, int] = (10, 6),
                      compare: bool = False) -> Any:
    """
    Visualize kernel weight functions.
    
    This function plots the weights for a specified kernel type, or compares
    multiple kernel types if compare=True.
    
    Args:
        kernel_type: Type of kernel function ('bartlett', 'parzen', 'tukey-hanning', 
                    'tukey', 'hanning', 'quadratic', 'flat-top')
        bandwidth: Bandwidth parameter
        n: Number of weights to compute (including lag 0)
        figsize: Figure size as (width, height) in inches
        compare: Whether to compare multiple kernel types
        
    Returns:
        matplotlib.figure.Figure: The generated figure
        
    Raises:
        ImportError: If matplotlib is not available
        ValueError: If kernel_type is not recognized
        
    Examples:
        >>> from mfe.models.realized.kernel_weights import plot_kernel_weights
        >>> fig = plot_kernel_weights('bartlett', 10.0, 20)
        >>> fig = plot_kernel_weights(compare=True)  # Compare all kernel types
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Matplotlib is required for plotting kernel weights")
    
    if compare:
        # Compare multiple kernel types
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get list of kernel types (excluding aliases)
        kernel_types = ['bartlett', 'parzen', 'tukey-hanning', 'quadratic', 'flat-top']
        
        # Compute and plot weights for each kernel type
        lags = np.arange(n)
        for kernel in kernel_types:
            weights = compute_kernel_weights(n, kernel, bandwidth)
            ax.plot(lags, weights, label=kernel.capitalize())
        
        ax.set_title(f"Comparison of Kernel Weight Functions (H = {bandwidth:.2f})")
        ax.set_xlabel("Lag")
        ax.set_ylabel("Weight")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
    else:
        # Plot a single kernel type
        fig, ax = plt.subplots(figsize=figsize)
        
        # Compute and plot weights
        weights = compute_kernel_weights(n, kernel_type, bandwidth)
        lags = np.arange(n)
        
        ax.stem(lags, weights, basefmt=' ')
        
        ax.set_title(f"{kernel_type.capitalize()} Kernel Weights (H = {bandwidth:.2f})")
        ax.set_xlabel("Lag")
        ax.set_ylabel("Weight")
        ax.grid(True, alpha=0.3)
    
    return fig



# Register Numba-accelerated functions if available
def _register_numba_functions() -> None:
    """
    Register Numba JIT-compiled functions for kernel weights computation.
    
    This function is called during module initialization to register
    performance-critical functions for JIT compilation if Numba is available.
    """
    if HAS_NUMBA:
        # The functions are already decorated with @jit, so no additional
        # registration is needed here. This function is kept for consistency
        # with the module structure and potential future enhancements.
        logger.debug("Kernel weights Numba JIT functions registered")
    else:
        logger.info("Numba not available. Kernel weights computation will use pure NumPy implementations.")



# Initialize the module
_register_numba_functions()
