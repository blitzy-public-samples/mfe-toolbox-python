# mfe/models/realized/kernel_bandwidth.py
"""
Optimal bandwidth computation for realized kernel estimators.

This module provides functions for computing the optimal bandwidth for realized kernel
estimators, which is a critical parameter that balances bias and variance in kernel-based
volatility estimation. The optimal bandwidth depends on the kernel type, the properties
of the return series, and the noise characteristics of the data.

The module implements several methods for bandwidth selection, including rule-of-thumb
approaches based on asymptotic theory, data-driven methods that minimize mean squared
error, and adaptive methods that account for the specific characteristics of the data.
All implementations include comprehensive type hints, parameter validation, and
Numba acceleration for performance-critical calculations.

Functions:
    compute_optimal_bandwidth: Main function for computing optimal bandwidth
    compute_optimal_bandwidth_asymptotic: Compute bandwidth based on asymptotic theory
    compute_optimal_bandwidth_mse: Compute bandwidth by minimizing mean squared error
    compute_optimal_bandwidth_adaptive: Compute bandwidth using adaptive methods
    get_kernel_parameters: Get kernel-specific parameters for bandwidth computation
"""

import logging
import warnings
from typing import (
    Any, Callable, Dict, List, Literal, Optional, Sequence, 
    Tuple, Type, TypeVar, Union, cast, overload
)
import numpy as np
from scipy import stats, optimize

from ...core.exceptions import ParameterError, NumericError
from .utils import noise_variance

# Set up module-level logger
logger = logging.getLogger("mfe.models.realized.kernel_bandwidth")

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
    logger.debug("Numba available for kernel bandwidth computation acceleration")
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    HAS_NUMBA = False
    logger.info("Numba not available. Kernel bandwidth computation will use pure NumPy implementations.")



def get_kernel_parameters(kernel_type: str) -> Dict[str, float]:
    """
    Get kernel-specific parameters for bandwidth computation.
    
    Different kernel types have different optimal bandwidth formulas based on their
    properties. This function returns the appropriate parameters for each kernel type.
    
    Args:
        kernel_type: Type of kernel function ('bartlett', 'parzen', 'tukey-hanning', 
                    'tukey', 'hanning', 'quadratic', 'flat-top')
        
    Returns:
        Dictionary of kernel parameters
        
    Raises:
        ValueError: If kernel_type is not recognized
    """
    # Normalize kernel type to lowercase
    kernel_type_lower = kernel_type.lower()
    
    # Define kernel parameters dictionary
    kernel_params = {
        'bartlett': {
            'c_star': 1.0,
            'gamma_star': 1.0,
            'rate_exponent': 1/3,
            'description': 'Linear kernel with first-order bias'
        },
        'parzen': {
            'c_star': 2.6614,
            'gamma_star': 0.5,
            'rate_exponent': 1/5,
            'description': 'Smooth kernel with second-order bias'
        },
        'tukey-hanning': {
            'c_star': 3.4375,
            'gamma_star': 0.5,
            'rate_exponent': 1/5,
            'description': 'Cosine-based kernel with second-order bias'
        },
        'tukey': {
            'c_star': 3.4375,
            'gamma_star': 0.5,
            'rate_exponent': 1/5,
            'description': 'Alias for tukey-hanning'
        },
        'hanning': {
            'c_star': 3.4375,
            'gamma_star': 0.5,
            'rate_exponent': 1/5,
            'description': 'Alias for tukey-hanning'
        },
        'quadratic': {
            'c_star': 2.7308,
            'gamma_star': 0.5,
            'rate_exponent': 1/5,
            'description': 'Quadratic kernel with second-order bias'
        },
        'flat-top': {
            'c_star': 1.1,
            'gamma_star': 0.0,
            'rate_exponent': 1/4,
            'description': 'Flat-top kernel with minimal asymptotic bias'
        }
    }
    
    # Check if kernel type is recognized
    if kernel_type_lower not in kernel_params:
        valid_kernels = list(kernel_params.keys())
        raise ValueError(f"Unrecognized kernel type: {kernel_type}. "
                         f"Supported types are {valid_kernels}.")
    
    return kernel_params[kernel_type_lower]


@jit(nopython=True, cache=True)
def _compute_integrated_quarticity_numba(returns: np.ndarray) -> float:
    """
    Numba-accelerated computation of integrated quarticity.
    
    Integrated quarticity is used in bandwidth selection formulas to account for
    the volatility of volatility.
    
    Args:
        returns: Array of returns
        
    Returns:
        Integrated quarticity estimate
    """
    n = len(returns)
    
    # Compute realized quarticity (n/3 * sum of returns^4)
    realized_quarticity = (n / 3) * np.sum(returns**4)
    
    return realized_quarticity



def compute_integrated_quarticity(returns: np.ndarray) -> float:
    """
    Compute integrated quarticity from returns.
    
    Integrated quarticity is used in bandwidth selection formulas to account for
    the volatility of volatility.
    
    Args:
        returns: Array of returns
        
    Returns:
        Integrated quarticity estimate
        
    Raises:
        ValueError: If returns has invalid dimensions
    """
    # Convert to numpy array if not already
    returns = np.asarray(returns)
    
    # Validate inputs
    if returns.ndim != 1:
        raise ValueError("returns must be a 1D array")
    
    # Use Numba-accelerated implementation if available
    if HAS_NUMBA:
        return _compute_integrated_quarticity_numba(returns)
    
    # Pure NumPy implementation
    n = len(returns)
    realized_quarticity = (n / 3) * np.sum(returns**4)
    
    return realized_quarticity


@jit(nopython=True, cache=True)
def _compute_autocorrelation_decay_numba(returns: np.ndarray, max_lag: int) -> float:
    """
    Numba-accelerated computation of autocorrelation decay rate.
    
    The autocorrelation decay rate is used to adjust bandwidth selection based on
    the serial dependence structure of returns.
    
    Args:
        returns: Array of returns
        max_lag: Maximum lag to consider
        
    Returns:
        Autocorrelation decay rate
    """
    n = len(returns)
    
    # Compute mean and variance
    mean_return = np.mean(returns)
    var_return = np.var(returns)
    
    if var_return == 0:
        return 0.0
    
    # Compute autocorrelations
    acf = np.zeros(max_lag)
    
    for lag in range(max_lag):
        cov_sum = 0.0
        for t in range(n - lag - 1):
            cov_sum += (returns[t] - mean_return) * (returns[t + lag + 1] - mean_return)
        
        acf[lag] = cov_sum / ((n - lag - 1) * var_return)
    
    # Compute decay rate (average absolute difference between consecutive autocorrelations)
    decay_rate = np.mean(np.abs(np.diff(acf)))
    
    return decay_rate



def compute_autocorrelation_decay(returns: np.ndarray, max_lag: int = 20) -> float:
    """
    Compute autocorrelation decay rate from returns.
    
    The autocorrelation decay rate is used to adjust bandwidth selection based on
    the serial dependence structure of returns.
    
    Args:
        returns: Array of returns
        max_lag: Maximum lag to consider
        
    Returns:
        Autocorrelation decay rate
        
    Raises:
        ValueError: If returns has invalid dimensions or if max_lag is not positive
    """
    # Convert to numpy array if not already
    returns = np.asarray(returns)
    
    # Validate inputs
    if returns.ndim != 1:
        raise ValueError("returns must be a 1D array")
    if max_lag <= 0:
        raise ValueError("max_lag must be positive")
    
    # Adjust max_lag if necessary
    max_lag = min(max_lag, len(returns) // 4)
    
    # Use Numba-accelerated implementation if available
    if HAS_NUMBA:
        return _compute_autocorrelation_decay_numba(returns, max_lag)
    
    # Pure NumPy implementation
    # Compute autocorrelations
    acf = np.zeros(max_lag)
    
    for lag in range(max_lag):
        acf[lag] = np.corrcoef(returns[:-lag-1], returns[lag+1:])[0, 1]
    
    # Compute decay rate (average absolute difference between consecutive autocorrelations)
    decay_rate = np.mean(np.abs(np.diff(acf)))
    
    return decay_rate



def compute_optimal_bandwidth_asymptotic(returns: np.ndarray, 
                                        kernel_type: str = 'bartlett',
                                        noise_var: Optional[float] = None) -> float:
    """
    Compute optimal bandwidth based on asymptotic theory.
    
    This method uses asymptotic formulas derived from minimizing the asymptotic
    mean squared error of the realized kernel estimator.
    
    Args:
        returns: Array of returns
        kernel_type: Type of kernel function
        noise_var: Noise variance (if None, it will be estimated)
        
    Returns:
        Optimal bandwidth
        
    Raises:
        ValueError: If returns has invalid dimensions or if kernel_type is not recognized
    """
    # Convert to numpy array if not already
    returns = np.asarray(returns)
    
    # Validate inputs
    if returns.ndim != 1:
        raise ValueError("returns must be a 1D array")
    
    # Get kernel parameters
    kernel_params = get_kernel_parameters(kernel_type)
    c_star = kernel_params['c_star']
    gamma_star = kernel_params['gamma_star']
    rate_exponent = kernel_params['rate_exponent']
    
    # Get number of observations
    n = len(returns)
    
    # Estimate noise variance if not provided
    if noise_var is None:
        noise_var = noise_variance(returns)
    
    # Compute integrated quarticity
    quarticity = compute_integrated_quarticity(returns)
    
    # Compute integrated variance (approximated by realized variance)
    integrated_var = np.sum(returns**2)
    
    # Compute signal-to-noise ratio
    if noise_var > 0 and integrated_var > 0:
        xi = noise_var / np.sqrt(integrated_var)
    else:
        # Default value if estimation fails
        xi = 0.1
        logger.warning("Signal-to-noise ratio estimation failed. Using default value.")
    
    # Compute optimal bandwidth
    if gamma_star == 0:
        # For flat-top kernel
        h_star = c_star * (xi**2 * n)**(rate_exponent)
    else:
        # For other kernels
        h_star = c_star * ((xi**2 * quarticity**(1/4) * n) / 
                          (gamma_star * integrated_var**(1/2)))**(rate_exponent)
    
    # Ensure bandwidth is at least 1
    h_star = max(1, h_star)
    
    return h_star



def compute_optimal_bandwidth_mse(returns: np.ndarray, 
                                kernel_type: str = 'bartlett',
                                noise_var: Optional[float] = None,
                                grid_size: int = 20) -> float:
    """
    Compute optimal bandwidth by minimizing mean squared error.
    
    This method uses a grid search to find the bandwidth that minimizes the
    estimated mean squared error of the realized kernel estimator.
    
    Args:
        returns: Array of returns
        kernel_type: Type of kernel function
        noise_var: Noise variance (if None, it will be estimated)
        grid_size: Number of grid points for bandwidth search
        
    Returns:
        Optimal bandwidth
        
    Raises:
        ValueError: If returns has invalid dimensions or if kernel_type is not recognized
    """
    # Convert to numpy array if not already
    returns = np.asarray(returns)
    
    # Validate inputs
    if returns.ndim != 1:
        raise ValueError("returns must be a 1D array")
    
    # Get kernel parameters
    kernel_params = get_kernel_parameters(kernel_type)
    
    # Get number of observations
    n = len(returns)
    
    # Estimate noise variance if not provided
    if noise_var is None:
        noise_var = noise_variance(returns)
    
    # Compute integrated variance (approximated by realized variance)
    integrated_var = np.sum(returns**2)
    
    # Define a grid of bandwidths to search
    min_h = max(1, n**(1/5))
    max_h = max(10, n**(1/2))
    h_grid = np.linspace(min_h, max_h, grid_size)
    
    # Define MSE function to minimize
    def mse_function(h: float) -> float:
        # Compute bias term
        if kernel_params['gamma_star'] == 0:
            # For flat-top kernel
            bias = 0
        else:
            # For other kernels
            bias = kernel_params['gamma_star'] * (h / n) * integrated_var
        
        # Compute variance term
        variance = (2 * noise_var**2) / h
        
        # Compute MSE
        mse = bias**2 + variance
        
        return mse
    
    # Evaluate MSE for each bandwidth in the grid
    mse_values = np.array([mse_function(h) for h in h_grid])
    
    # Find bandwidth with minimum MSE
    optimal_idx = np.argmin(mse_values)
    h_star = h_grid[optimal_idx]
    
    # Ensure bandwidth is at least 1
    h_star = max(1, h_star)
    
    return h_star



def compute_optimal_bandwidth_adaptive(returns: np.ndarray, 
                                     kernel_type: str = 'bartlett',
                                     noise_var: Optional[float] = None) -> float:
    """
    Compute optimal bandwidth using adaptive methods.
    
    This method adjusts the asymptotic bandwidth based on the autocorrelation
    structure of the returns, providing a more data-driven approach to bandwidth selection.
    
    Args:
        returns: Array of returns
        kernel_type: Type of kernel function
        noise_var: Noise variance (if None, it will be estimated)
        
    Returns:
        Optimal bandwidth
        
    Raises:
        ValueError: If returns has invalid dimensions or if kernel_type is not recognized
    """
    # Convert to numpy array if not already
    returns = np.asarray(returns)
    
    # Validate inputs
    if returns.ndim != 1:
        raise ValueError("returns must be a 1D array")
    
    # Compute base bandwidth using asymptotic formula
    h_base = compute_optimal_bandwidth_asymptotic(returns, kernel_type, noise_var)
    
    # Compute autocorrelation decay rate
    decay_rate = compute_autocorrelation_decay(returns)
    
    # Adjust bandwidth based on autocorrelation structure
    if decay_rate > 0.1:
        # Fast decay (weak autocorrelation) -> smaller bandwidth
        adjustment_factor = 0.8
    elif decay_rate < 0.05:
        # Slow decay (strong autocorrelation) -> larger bandwidth
        adjustment_factor = 1.2
    else:
        # Moderate decay -> no adjustment
        adjustment_factor = 1.0
    
    # Apply adjustment
    h_star = h_base * adjustment_factor
    
    # Ensure bandwidth is at least 1
    h_star = max(1, h_star)
    
    # Round to nearest integer for practical implementation
    h_star = round(h_star)
    
    return h_star



def compute_optimal_bandwidth(returns: np.ndarray, 
                            kernel_type: str = 'bartlett',
                            method: str = 'adaptive',
                            noise_var: Optional[float] = None,
                            **kwargs: Any) -> float:
    """
    Compute optimal bandwidth for realized kernel estimator.
    
    This is the main function for bandwidth selection, which dispatches to the
    appropriate method based on the specified approach.
    
    Args:
        returns: Array of returns
        kernel_type: Type of kernel function
        method: Method for bandwidth selection ('asymptotic', 'mse', 'adaptive')
        noise_var: Noise variance (if None, it will be estimated)
        **kwargs: Additional keyword arguments for specific methods
        
    Returns:
        Optimal bandwidth
        
    Raises:
        ValueError: If returns has invalid dimensions, if kernel_type is not recognized,
                   or if method is not recognized
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.kernel_bandwidth import compute_optimal_bandwidth
        >>> np.random.seed(42)
        >>> returns = np.random.normal(0, 0.01, 1000)
        >>> compute_optimal_bandwidth(returns, 'bartlett')
        4.0
    """
    # Convert to numpy array if not already
    returns = np.asarray(returns)
    
    # Validate inputs
    if returns.ndim != 1:
        raise ValueError("returns must be a 1D array")
    
    # Normalize method to lowercase
    method_lower = method.lower()
    
    # Dispatch to appropriate method
    if method_lower == 'asymptotic':
        return compute_optimal_bandwidth_asymptotic(returns, kernel_type, noise_var)
    
    elif method_lower == 'mse':
        grid_size = kwargs.get('grid_size', 20)
        return compute_optimal_bandwidth_mse(returns, kernel_type, noise_var, grid_size)
    
    elif method_lower == 'adaptive':
        return compute_optimal_bandwidth_adaptive(returns, kernel_type, noise_var)
    
    else:
        raise ValueError(f"Unrecognized bandwidth selection method: {method}. "
                         f"Supported methods are 'asymptotic', 'mse', 'adaptive'.")


# Register Numba-accelerated functions if available
def _register_numba_functions() -> None:
    """
    Register Numba JIT-compiled functions for kernel bandwidth computation.
    
    This function is called during module initialization to register
    performance-critical functions for JIT compilation if Numba is available.
    """
    if HAS_NUMBA:
        # The functions are already decorated with @jit, so no additional
        # registration is needed here. This function is kept for consistency
        # with the module structure and potential future enhancements.
        logger.debug("Kernel bandwidth Numba JIT functions registered")
    else:
        logger.info("Numba not available. Kernel bandwidth computation will use pure NumPy implementations.")


# Initialize the module
_register_numba_functions()
