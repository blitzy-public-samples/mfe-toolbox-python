"""
Numerical Differentiation Module

This module provides numerical differentiation utilities for gradient and Hessian
calculation, essential for optimization-based model estimation throughout the MFE Toolbox.
These functions compute finite-difference approximations used in maximum likelihood
estimation, optimization, and sensitivity analysis.

The module implements efficient numerical differentiation using NumPy's array operations
and Numba's JIT compilation for performance-critical sections. All functions include
comprehensive type hints and input validation to ensure reliability and proper error handling.

Functions:
    gradient_2sided: Compute two-sided numerical gradient of a function
    hessian_2sided: Compute two-sided numerical Hessian of a function
    jacobian: Compute the Jacobian matrix of a vector-valued function
    numerical_derivative: Compute numerical derivative of a function
    numerical_hessian: Compute numerical Hessian of a function with custom step sizes
"""

import asyncio
import logging
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np
from scipy import optimize

from mfe.core.exceptions import (
    DimensionError, NumericError, raise_dimension_error, 
    raise_numeric_error, warn_numeric
)
from mfe.core.types import (
    Vector, Matrix, ObjectiveFunction, GradientFunction, 
    HessianFunction, AsyncCallback
)

# Set up module-level logger
logger = logging.getLogger("mfe.utils.differentiation")

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
    logger.debug("Numba available for differentiation acceleration")
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    HAS_NUMBA = False
    logger.info("Numba not available. Differentiation will use pure NumPy implementations.")


@jit(nopython=True, cache=True)
def _gradient_2sided_numba(func: Callable[[np.ndarray], float], 
                           x: np.ndarray, 
                           epsilon: float) -> np.ndarray:
    """
    Numba-accelerated implementation of two-sided numerical gradient.
    
    Args:
        func: Function to differentiate
        x: Point at which to compute the gradient
        epsilon: Step size for finite difference
        
    Returns:
        Gradient vector
    """
    n = x.shape[0]
    grad = np.zeros(n, dtype=np.float64)
    x_plus = x.copy()
    x_minus = x.copy()
    
    for i in range(n):
        # Forward step
        x_plus[i] = x[i] + epsilon
        f_plus = func(x_plus)
        
        # Backward step
        x_minus[i] = x[i] - epsilon
        f_minus = func(x_minus)
        
        # Central difference
        grad[i] = (f_plus - f_minus) / (2.0 * epsilon)
        
        # Reset
        x_plus[i] = x[i]
        x_minus[i] = x[i]
    
    return grad



def gradient_2sided(func: ObjectiveFunction, 
                    x: Vector, 
                    epsilon: Optional[float] = None, 
                    args: Tuple = ()) -> Vector:
    """
    Compute two-sided numerical gradient of a function.
    
    This function calculates the gradient of a scalar function using the two-sided
    (central) finite difference method, which provides better accuracy than one-sided
    differences. For a function f(x), the gradient is computed as:
    
    ∂f/∂x_i ≈ [f(x + ε*e_i) - f(x - ε*e_i)] / (2*ε)
    
    where e_i is the i-th unit vector and ε is a small step size.
    
    Args:
        func: Function to differentiate, should take a vector and return a scalar
        x: Point at which to compute the gradient
        epsilon: Step size for finite difference. If None, an appropriate value
                is selected based on machine precision and input scale
        args: Additional arguments to pass to the function
        
    Returns:
        Gradient vector of the same shape as x
        
    Raises:
        DimensionError: If x is not a 1D array
        NumericError: If the function evaluation produces NaN or Inf values
        
    Examples:
        >>> import numpy as np
        >>> from mfe.utils.differentiation import gradient_2sided
        >>> def f(x): return x[0]**2 + x[1]**2  # Simple quadratic function
        >>> x = np.array([1.0, 2.0])
        >>> gradient_2sided(f, x)
        array([2., 4.])
    """
    # Convert to numpy array if not already
    x = np.asarray(x, dtype=float)
    
    # Check if x is 1D
    if x.ndim != 1:
        raise_dimension_error(
            "Input must be a 1D vector",
            array_name="x",
            expected_shape="(n,)",
            actual_shape=x.shape
        )
    
    # Determine appropriate epsilon if not provided
    if epsilon is None:
        # Scale epsilon based on the magnitude of x
        x_abs = np.abs(x)
        # Use machine epsilon as a base
        eps = np.finfo(float).eps
        # For each element, use either a scaled value or a minimum threshold
        epsilon_vec = np.maximum(x_abs, 1e-8) * np.sqrt(eps)
        # Use the minimum non-zero value to ensure we don't have too small steps
        epsilon = np.min(epsilon_vec[epsilon_vec > 0])
        # Ensure epsilon is not too small
        epsilon = max(epsilon, np.sqrt(eps))
    
    # Create a wrapper function that includes the additional arguments
    def func_wrapper(x_vec):
        try:
            return func(x_vec, *args)
        except Exception as e:
            raise_numeric_error(
                f"Function evaluation failed in gradient_2sided: {str(e)}",
                operation="gradient_2sided",
                values=x_vec,
                error_type="function_evaluation_error"
            )
    
    # Use Numba-accelerated implementation if available
    if HAS_NUMBA:
        try:
            return _gradient_2sided_numba(func_wrapper, x, epsilon)
        except Exception as e:
            # If Numba implementation fails, fall back to NumPy
            logger.warning(f"Numba gradient calculation failed, falling back to NumPy: {str(e)}")
    
    # Pure NumPy implementation
    n = x.shape[0]
    grad = np.zeros(n, dtype=float)
    
    # Create copies for forward and backward steps
    x_plus = x.copy()
    x_minus = x.copy()
    
    for i in range(n):
        # Forward step
        x_plus[i] = x[i] + epsilon
        try:
            f_plus = func_wrapper(x_plus)
        except Exception as e:
            raise_numeric_error(
                f"Forward difference evaluation failed at index {i}: {str(e)}",
                operation="gradient_2sided",
                values=x_plus,
                error_type="forward_difference_error"
            )
        
        # Backward step
        x_minus[i] = x[i] - epsilon
        try:
            f_minus = func_wrapper(x_minus)
        except Exception as e:
            raise_numeric_error(
                f"Backward difference evaluation failed at index {i}: {str(e)}",
                operation="gradient_2sided",
                values=x_minus,
                error_type="backward_difference_error"
            )
        
        # Central difference
        grad[i] = (f_plus - f_minus) / (2.0 * epsilon)
        
        # Check for NaN or Inf
        if not np.isfinite(grad[i]):
            warn_numeric(
                f"Non-finite gradient detected at index {i}",
                operation="gradient_2sided",
                issue="non_finite_gradient",
                value=grad[i]
            )
        
        # Reset for next iteration
        x_plus[i] = x[i]
        x_minus[i] = x[i]
    
    return grad


@jit(nopython=True, cache=True)
def _hessian_2sided_numba(func: Callable[[np.ndarray], float], 
                          x: np.ndarray, 
                          epsilon: float) -> np.ndarray:
    """
    Numba-accelerated implementation of two-sided numerical Hessian.
    
    Args:
        func: Function to differentiate
        x: Point at which to compute the Hessian
        epsilon: Step size for finite difference
        
    Returns:
        Hessian matrix
    """
    n = x.shape[0]
    hess = np.zeros((n, n), dtype=np.float64)
    x_pp = x.copy()
    x_pm = x.copy()
    x_mp = x.copy()
    x_mm = x.copy()
    
    # Diagonal elements (second derivatives)
    for i in range(n):
        # f(x + 2*epsilon*e_i)
        x_pp[i] = x[i] + 2.0 * epsilon
        f_pp = func(x_pp)
        
        # f(x)
        f = func(x)
        
        # f(x - 2*epsilon*e_i)
        x_mm[i] = x[i] - 2.0 * epsilon
        f_mm = func(x_mm)
        
        # Second derivative approximation
        hess[i, i] = (f_pp - 2.0 * f + f_mm) / (4.0 * epsilon * epsilon)
        
        # Reset
        x_pp[i] = x[i]
        x_mm[i] = x[i]
    
    # Off-diagonal elements (mixed partial derivatives)
    for i in range(n):
        for j in range(i+1, n):
            # f(x + epsilon*e_i + epsilon*e_j)
            x_pp[i] = x[i] + epsilon
            x_pp[j] = x[j] + epsilon
            f_pp = func(x_pp)
            
            # f(x + epsilon*e_i - epsilon*e_j)
            x_pm[i] = x[i] + epsilon
            x_pm[j] = x[j] - epsilon
            f_pm = func(x_pm)
            
            # f(x - epsilon*e_i + epsilon*e_j)
            x_mp[i] = x[i] - epsilon
            x_mp[j] = x[j] + epsilon
            f_mp = func(x_mp)
            
            # f(x - epsilon*e_i - epsilon*e_j)
            x_mm[i] = x[i] - epsilon
            x_mm[j] = x[j] - epsilon
            f_mm = func(x_mm)
            
            # Mixed partial derivative approximation
            hess[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * epsilon * epsilon)
            # Symmetric matrix
            hess[j, i] = hess[i, j]
            
            # Reset
            x_pp[i] = x[i]
            x_pp[j] = x[j]
            x_pm[i] = x[i]
            x_pm[j] = x[j]
            x_mp[i] = x[i]
            x_mp[j] = x[j]
            x_mm[i] = x[i]
            x_mm[j] = x[j]
    
    return hess



def hessian_2sided(func: ObjectiveFunction, 
                   x: Vector, 
                   epsilon: Optional[float] = None, 
                   args: Tuple = ()) -> Matrix:
    """
    Compute two-sided numerical Hessian of a function.
    
    This function calculates the Hessian matrix (matrix of second derivatives)
    of a scalar function using the two-sided finite difference method. For a
    function f(x), the Hessian elements are computed as:
    
    ∂²f/∂x_i² ≈ [f(x + 2ε*e_i) - 2f(x) + f(x - 2ε*e_i)] / (4*ε²)
    ∂²f/∂x_i∂x_j ≈ [f(x + ε*e_i + ε*e_j) - f(x + ε*e_i - ε*e_j) - f(x - ε*e_i + ε*e_j) + f(x - ε*e_i - ε*e_j)] / (4*ε²)
    
    where e_i is the i-th unit vector and ε is a small step size.
    
    Args:
        func: Function to differentiate, should take a vector and return a scalar
        x: Point at which to compute the Hessian
        epsilon: Step size for finite difference. If None, an appropriate value
                is selected based on machine precision and input scale
        args: Additional arguments to pass to the function
        
    Returns:
        Hessian matrix of shape (n, n) where n is the length of x
        
    Raises:
        DimensionError: If x is not a 1D array
        NumericError: If the function evaluation produces NaN or Inf values
        
    Examples:
        >>> import numpy as np
        >>> from mfe.utils.differentiation import hessian_2sided
        >>> def f(x): return x[0]**2 + x[1]**2  # Simple quadratic function
        >>> x = np.array([1.0, 2.0])
        >>> hessian_2sided(f, x)
        array([[2., 0.],
               [0., 2.]])
    """
    # Convert to numpy array if not already
    x = np.asarray(x, dtype=float)
    
    # Check if x is 1D
    if x.ndim != 1:
        raise_dimension_error(
            "Input must be a 1D vector",
            array_name="x",
            expected_shape="(n,)",
            actual_shape=x.shape
        )
    
    # Determine appropriate epsilon if not provided
    if epsilon is None:
        # Scale epsilon based on the magnitude of x
        x_abs = np.abs(x)
        # Use machine epsilon as a base
        eps = np.finfo(float).eps
        # For each element, use either a scaled value or a minimum threshold
        epsilon_vec = np.maximum(x_abs, 1e-8) * np.power(eps, 1/3)
        # Use the minimum non-zero value to ensure we don't have too small steps
        epsilon = np.min(epsilon_vec[epsilon_vec > 0])
        # Ensure epsilon is not too small
        epsilon = max(epsilon, np.power(eps, 1/3))
    
    # Create a wrapper function that includes the additional arguments
    def func_wrapper(x_vec):
        try:
            return func(x_vec, *args)
        except Exception as e:
            raise_numeric_error(
                f"Function evaluation failed in hessian_2sided: {str(e)}",
                operation="hessian_2sided",
                values=x_vec,
                error_type="function_evaluation_error"
            )
    
    # Use Numba-accelerated implementation if available
    if HAS_NUMBA:
        try:
            return _hessian_2sided_numba(func_wrapper, x, epsilon)
        except Exception as e:
            # If Numba implementation fails, fall back to NumPy
            logger.warning(f"Numba Hessian calculation failed, falling back to NumPy: {str(e)}")
    
    # Pure NumPy implementation
    n = x.shape[0]
    hess = np.zeros((n, n), dtype=float)
    
    # Create copies for different perturbations
    x_pp = x.copy()  # x + epsilon_i + epsilon_j
    x_pm = x.copy()  # x + epsilon_i - epsilon_j
    x_mp = x.copy()  # x - epsilon_i + epsilon_j
    x_mm = x.copy()  # x - epsilon_i - epsilon_j
    
    # Compute diagonal elements (second derivatives)
    for i in range(n):
        # f(x + 2*epsilon*e_i)
        x_pp[i] = x[i] + 2.0 * epsilon
        try:
            f_pp = func_wrapper(x_pp)
        except Exception as e:
            raise_numeric_error(
                f"Function evaluation failed for diagonal element {i}: {str(e)}",
                operation="hessian_2sided",
                values=x_pp,
                error_type="diagonal_evaluation_error"
            )
        
        # f(x)
        try:
            f = func_wrapper(x)
        except Exception as e:
            raise_numeric_error(
                f"Function evaluation failed at center point: {str(e)}",
                operation="hessian_2sided",
                values=x,
                error_type="center_evaluation_error"
            )
        
        # f(x - 2*epsilon*e_i)
        x_mm[i] = x[i] - 2.0 * epsilon
        try:
            f_mm = func_wrapper(x_mm)
        except Exception as e:
            raise_numeric_error(
                f"Function evaluation failed for diagonal element {i}: {str(e)}",
                operation="hessian_2sided",
                values=x_mm,
                error_type="diagonal_evaluation_error"
            )
        
        # Second derivative approximation
        hess[i, i] = (f_pp - 2.0 * f + f_mm) / (4.0 * epsilon * epsilon)
        
        # Check for NaN or Inf
        if not np.isfinite(hess[i, i]):
            warn_numeric(
                f"Non-finite Hessian diagonal element detected at index ({i}, {i})",
                operation="hessian_2sided",
                issue="non_finite_hessian",
                value=hess[i, i]
            )
        
        # Reset for next iteration
        x_pp[i] = x[i]
        x_mm[i] = x[i]
    
    # Compute off-diagonal elements (mixed partial derivatives)
    for i in range(n):
        for j in range(i+1, n):
            # f(x + epsilon*e_i + epsilon*e_j)
            x_pp[i] = x[i] + epsilon
            x_pp[j] = x[j] + epsilon
            try:
                f_pp = func_wrapper(x_pp)
            except Exception as e:
                raise_numeric_error(
                    f"Function evaluation failed for mixed derivative ({i}, {j}): {str(e)}",
                    operation="hessian_2sided",
                    values=x_pp,
                    error_type="mixed_derivative_error"
                )
            
            # f(x + epsilon*e_i - epsilon*e_j)
            x_pm[i] = x[i] + epsilon
            x_pm[j] = x[j] - epsilon
            try:
                f_pm = func_wrapper(x_pm)
            except Exception as e:
                raise_numeric_error(
                    f"Function evaluation failed for mixed derivative ({i}, {j}): {str(e)}",
                    operation="hessian_2sided",
                    values=x_pm,
                    error_type="mixed_derivative_error"
                )
            
            # f(x - epsilon*e_i + epsilon*e_j)
            x_mp[i] = x[i] - epsilon
            x_mp[j] = x[j] + epsilon
            try:
                f_mp = func_wrapper(x_mp)
            except Exception as e:
                raise_numeric_error(
                    f"Function evaluation failed for mixed derivative ({i}, {j}): {str(e)}",
                    operation="hessian_2sided",
                    values=x_mp,
                    error_type="mixed_derivative_error"
                )
            
            # f(x - epsilon*e_i - epsilon*e_j)
            x_mm[i] = x[i] - epsilon
            x_mm[j] = x[j] - epsilon
            try:
                f_mm = func_wrapper(x_mm)
            except Exception as e:
                raise_numeric_error(
                    f"Function evaluation failed for mixed derivative ({i}, {j}): {str(e)}",
                    operation="hessian_2sided",
                    values=x_mm,
                    error_type="mixed_derivative_error"
                )
            
            # Mixed partial derivative approximation
            hess[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * epsilon * epsilon)
            # Symmetric matrix
            hess[j, i] = hess[i, j]
            
            # Check for NaN or Inf
            if not np.isfinite(hess[i, j]):
                warn_numeric(
                    f"Non-finite Hessian off-diagonal element detected at indices ({i}, {j}) and ({j}, {i})",
                    operation="hessian_2sided",
                    issue="non_finite_hessian",
                    value=hess[i, j]
                )
            
            # Reset for next iteration
            x_pp[i] = x[i]
            x_pp[j] = x[j]
            x_pm[i] = x[i]
            x_pm[j] = x[j]
            x_mp[i] = x[i]
            x_mp[j] = x[j]
            x_mm[i] = x[i]
            x_mm[j] = x[j]
    
    return hess



async def hessian_2sided_async(func: ObjectiveFunction, 
                              x: Vector, 
                              epsilon: Optional[float] = None, 
                              args: Tuple = (),
                              progress_callback: Optional[AsyncCallback] = None) -> Matrix:
    """
    Compute two-sided numerical Hessian of a function asynchronously.
    
    This function provides an asynchronous version of hessian_2sided, which is
    particularly useful for large-scale problems where Hessian computation can be
    time-consuming. It allows for progress reporting during computation.
    
    Args:
        func: Function to differentiate, should take a vector and return a scalar
        x: Point at which to compute the Hessian
        epsilon: Step size for finite difference. If None, an appropriate value
                is selected based on machine precision and input scale
        args: Additional arguments to pass to the function
        progress_callback: Optional async callback function to report progress
        
    Returns:
        Hessian matrix of shape (n, n) where n is the length of x
        
    Raises:
        DimensionError: If x is not a 1D array
        NumericError: If the function evaluation produces NaN or Inf values
        
    Examples:
        >>> import numpy as np
        >>> import asyncio
        >>> from mfe.utils.differentiation import hessian_2sided_async
        >>> 
        >>> async def progress(percent, message):
        ...     print(f"{percent:.1f}% complete: {message}")
        >>> 
        >>> def f(x): return x[0]**2 + x[1]**2  # Simple quadratic function
        >>> 
        >>> async def main():
        ...     x = np.array([1.0, 2.0])
        ...     hess = await hessian_2sided_async(f, x, progress_callback=progress)
        ...     print(hess)
        >>> 
        >>> asyncio.run(main())
        0.0% complete: Starting Hessian computation
        33.3% complete: Computing diagonal elements
        66.7% complete: Computing off-diagonal elements
        100.0% complete: Hessian computation complete
        array([[2., 0.],
               [0., 2.]])
    """
    # Convert to numpy array if not already
    x = np.asarray(x, dtype=float)
    
    # Check if x is 1D
    if x.ndim != 1:
        raise_dimension_error(
            "Input must be a 1D vector",
            array_name="x",
            expected_shape="(n,)",
            actual_shape=x.shape
        )
    
    # Determine appropriate epsilon if not provided
    if epsilon is None:
        # Scale epsilon based on the magnitude of x
        x_abs = np.abs(x)
        # Use machine epsilon as a base
        eps = np.finfo(float).eps
        # For each element, use either a scaled value or a minimum threshold
        epsilon_vec = np.maximum(x_abs, 1e-8) * np.power(eps, 1/3)
        # Use the minimum non-zero value to ensure we don't have too small steps
        epsilon = np.min(epsilon_vec[epsilon_vec > 0])
        # Ensure epsilon is not too small
        epsilon = max(epsilon, np.power(eps, 1/3))
    
    # Create a wrapper function that includes the additional arguments
    def func_wrapper(x_vec):
        try:
            return func(x_vec, *args)
        except Exception as e:
            raise_numeric_error(
                f"Function evaluation failed in hessian_2sided_async: {str(e)}",
                operation="hessian_2sided_async",
                values=x_vec,
                error_type="function_evaluation_error"
            )
    
    # Report progress if callback is provided
    if progress_callback:
        await progress_callback(0.0, "Starting Hessian computation")
    
    # Pure NumPy implementation (Numba doesn't work well with async)
    n = x.shape[0]
    hess = np.zeros((n, n), dtype=float)
    
    # Create copies for different perturbations
    x_pp = x.copy()  # x + epsilon_i + epsilon_j
    x_pm = x.copy()  # x + epsilon_i - epsilon_j
    x_mp = x.copy()  # x - epsilon_i + epsilon_j
    x_mm = x.copy()  # x - epsilon_i - epsilon_j
    
    # Compute diagonal elements (second derivatives)
    for i in range(n):
        # f(x + 2*epsilon*e_i)
        x_pp[i] = x[i] + 2.0 * epsilon
        try:
            f_pp = func_wrapper(x_pp)
        except Exception as e:
            raise_numeric_error(
                f"Function evaluation failed for diagonal element {i}: {str(e)}",
                operation="hessian_2sided_async",
                values=x_pp,
                error_type="diagonal_evaluation_error"
            )
        
        # f(x)
        try:
            f = func_wrapper(x)
        except Exception as e:
            raise_numeric_error(
                f"Function evaluation failed at center point: {str(e)}",
                operation="hessian_2sided_async",
                values=x,
                error_type="center_evaluation_error"
            )
        
        # f(x - 2*epsilon*e_i)
        x_mm[i] = x[i] - 2.0 * epsilon
        try:
            f_mm = func_wrapper(x_mm)
        except Exception as e:
            raise_numeric_error(
                f"Function evaluation failed for diagonal element {i}: {str(e)}",
                operation="hessian_2sided_async",
                values=x_mm,
                error_type="diagonal_evaluation_error"
            )
        
        # Second derivative approximation
        hess[i, i] = (f_pp - 2.0 * f + f_mm) / (4.0 * epsilon * epsilon)
        
        # Check for NaN or Inf
        if not np.isfinite(hess[i, i]):
            warn_numeric(
                f"Non-finite Hessian diagonal element detected at index ({i}, {i})",
                operation="hessian_2sided_async",
                issue="non_finite_hessian",
                value=hess[i, i]
            )
        
        # Reset for next iteration
        x_pp[i] = x[i]
        x_mm[i] = x[i]
    
    # Report progress after diagonal elements
    if progress_callback:
        await progress_callback(33.3, "Computing diagonal elements")
    
    # Allow other tasks to run
    await asyncio.sleep(0)
    
    # Compute off-diagonal elements (mixed partial derivatives)
    total_off_diag = n * (n - 1) // 2
    off_diag_count = 0
    
    for i in range(n):
        for j in range(i+1, n):
            # f(x + epsilon*e_i + epsilon*e_j)
            x_pp[i] = x[i] + epsilon
            x_pp[j] = x[j] + epsilon
            try:
                f_pp = func_wrapper(x_pp)
            except Exception as e:
                raise_numeric_error(
                    f"Function evaluation failed for mixed derivative ({i}, {j}): {str(e)}",
                    operation="hessian_2sided_async",
                    values=x_pp,
                    error_type="mixed_derivative_error"
                )
            
            # f(x + epsilon*e_i - epsilon*e_j)
            x_pm[i] = x[i] + epsilon
            x_pm[j] = x[j] - epsilon
            try:
                f_pm = func_wrapper(x_pm)
            except Exception as e:
                raise_numeric_error(
                    f"Function evaluation failed for mixed derivative ({i}, {j}): {str(e)}",
                    operation="hessian_2sided_async",
                    values=x_pm,
                    error_type="mixed_derivative_error"
                )
            
            # f(x - epsilon*e_i + epsilon*e_j)
            x_mp[i] = x[i] - epsilon
            x_mp[j] = x[j] + epsilon
            try:
                f_mp = func_wrapper(x_mp)
            except Exception as e:
                raise_numeric_error(
                    f"Function evaluation failed for mixed derivative ({i}, {j}): {str(e)}",
                    operation="hessian_2sided_async",
                    values=x_mp,
                    error_type="mixed_derivative_error"
                )
            
            # f(x - epsilon*e_i - epsilon*e_j)
            x_mm[i] = x[i] - epsilon
            x_mm[j] = x[j] - epsilon
            try:
                f_mm = func_wrapper(x_mm)
            except Exception as e:
                raise_numeric_error(
                    f"Function evaluation failed for mixed derivative ({i}, {j}): {str(e)}",
                    operation="hessian_2sided_async",
                    values=x_mm,
                    error_type="mixed_derivative_error"
                )
            
            # Mixed partial derivative approximation
            hess[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * epsilon * epsilon)
            # Symmetric matrix
            hess[j, i] = hess[i, j]
            
            # Check for NaN or Inf
            if not np.isfinite(hess[i, j]):
                warn_numeric(
                    f"Non-finite Hessian off-diagonal element detected at indices ({i}, {j}) and ({j}, {i})",
                    operation="hessian_2sided_async",
                    issue="non_finite_hessian",
                    value=hess[i, j]
                )
            
            # Reset for next iteration
            x_pp[i] = x[i]
            x_pp[j] = x[j]
            x_pm[i] = x[i]
            x_pm[j] = x[j]
            x_mp[i] = x[i]
            x_mp[j] = x[j]
            x_mm[i] = x[i]
            x_mm[j] = x[j]
            
            # Update progress
            off_diag_count += 1
            if progress_callback and off_diag_count % max(1, total_off_diag // 10) == 0:
                progress_percent = 33.3 + 66.7 * (off_diag_count / total_off_diag)
                await progress_callback(progress_percent, "Computing off-diagonal elements")
            
            # Allow other tasks to run periodically
            if off_diag_count % 10 == 0:
                await asyncio.sleep(0)
    
    # Final progress report
    if progress_callback:
        await progress_callback(100.0, "Hessian computation complete")
    
    return hess



def jacobian(func: Callable[[Vector], Vector], 
             x: Vector, 
             epsilon: Optional[float] = None, 
             args: Tuple = ()) -> Matrix:
    """
    Compute the Jacobian matrix of a vector-valued function.
    
    This function calculates the Jacobian matrix of a vector-valued function using
    the two-sided finite difference method. For a function f(x) that returns a
    vector of length m, the Jacobian is an m×n matrix where each element (i,j) is
    the partial derivative of f_i with respect to x_j.
    
    Args:
        func: Function to differentiate, should take a vector and return a vector
        x: Point at which to compute the Jacobian
        epsilon: Step size for finite difference. If None, an appropriate value
                is selected based on machine precision and input scale
        args: Additional arguments to pass to the function
        
    Returns:
        Jacobian matrix of shape (m, n) where m is the length of func(x) and
        n is the length of x
        
    Raises:
        DimensionError: If x is not a 1D array or if func(x) is not a 1D array
        NumericError: If the function evaluation produces NaN or Inf values
        
    Examples:
        >>> import numpy as np
        >>> from mfe.utils.differentiation import jacobian
        >>> def f(x): return np.array([x[0]**2, x[0]*x[1], x[1]**2])
        >>> x = np.array([1.0, 2.0])
        >>> jacobian(f, x)
        array([[2., 0.],
               [2., 1.],
               [0., 4.]])
    """
    # Convert to numpy array if not already
    x = np.asarray(x, dtype=float)
    
    # Check if x is 1D
    if x.ndim != 1:
        raise_dimension_error(
            "Input must be a 1D vector",
            array_name="x",
            expected_shape="(n,)",
            actual_shape=x.shape
        )
    
    # Create a wrapper function that includes the additional arguments
    def func_wrapper(x_vec):
        try:
            result = func(x_vec, *args)
            # Ensure result is a 1D array
            result = np.asarray(result, dtype=float)
            if result.ndim != 1:
                raise_dimension_error(
                    "Function must return a 1D vector",
                    array_name="func(x)",
                    expected_shape="(m,)",
                    actual_shape=result.shape
                )
            return result
        except Exception as e:
            if isinstance(e, DimensionError):
                raise
            raise_numeric_error(
                f"Function evaluation failed in jacobian: {str(e)}",
                operation="jacobian",
                values=x_vec,
                error_type="function_evaluation_error"
            )
    
    # Evaluate function at x to determine output dimension
    f_x = func_wrapper(x)
    m = f_x.shape[0]  # Output dimension
    n = x.shape[0]    # Input dimension
    
    # Determine appropriate epsilon if not provided
    if epsilon is None:
        # Scale epsilon based on the magnitude of x
        x_abs = np.abs(x)
        # Use machine epsilon as a base
        eps = np.finfo(float).eps
        # For each element, use either a scaled value or a minimum threshold
        epsilon_vec = np.maximum(x_abs, 1e-8) * np.sqrt(eps)
        # Use the minimum non-zero value to ensure we don't have too small steps
        epsilon = np.min(epsilon_vec[epsilon_vec > 0])
        # Ensure epsilon is not too small
        epsilon = max(epsilon, np.sqrt(eps))
    
    # Initialize Jacobian matrix
    jac = np.zeros((m, n), dtype=float)
    
    # Create copies for forward and backward steps
    x_plus = x.copy()
    x_minus = x.copy()
    
    # Compute Jacobian using two-sided finite differences
    for j in range(n):
        # Forward step
        x_plus[j] = x[j] + epsilon
        try:
            f_plus = func_wrapper(x_plus)
        except Exception as e:
            raise_numeric_error(
                f"Forward difference evaluation failed at index {j}: {str(e)}",
                operation="jacobian",
                values=x_plus,
                error_type="forward_difference_error"
            )
        
        # Backward step
        x_minus[j] = x[j] - epsilon
        try:
            f_minus = func_wrapper(x_minus)
        except Exception as e:
            raise_numeric_error(
                f"Backward difference evaluation failed at index {j}: {str(e)}",
                operation="jacobian",
                values=x_minus,
                error_type="backward_difference_error"
            )
        
        # Central difference for each output component
        jac[:, j] = (f_plus - f_minus) / (2.0 * epsilon)
        
        # Check for NaN or Inf
        if not np.all(np.isfinite(jac[:, j])):
            warn_numeric(
                f"Non-finite Jacobian elements detected in column {j}",
                operation="jacobian",
                issue="non_finite_jacobian",
                value=jac[:, j]
            )
        
        # Reset for next iteration
        x_plus[j] = x[j]
        x_minus[j] = x[j]
    
    return jac



def numerical_derivative(func: Callable[[float], float], 
                         x: float, 
                         order: int = 1, 
                         epsilon: Optional[float] = None) -> float:
    """
    Compute numerical derivative of a scalar function of a scalar variable.
    
    This function calculates the derivative of a scalar function with respect to
    a scalar variable using finite differences. It supports derivatives of
    different orders.
    
    Args:
        func: Function to differentiate, should take a scalar and return a scalar
        x: Point at which to compute the derivative
        order: Order of the derivative (1 for first derivative, 2 for second, etc.)
        epsilon: Step size for finite difference. If None, an appropriate value
                is selected based on machine precision and input scale
        
    Returns:
        Numerical derivative of the specified order
        
    Raises:
        ValueError: If order is less than 1
        NumericError: If the function evaluation produces NaN or Inf values
        
    Examples:
        >>> from mfe.utils.differentiation import numerical_derivative
        >>> def f(x): return x**3  # Cubic function
        >>> numerical_derivative(f, 2.0)  # First derivative at x=2
        12.000000000000064
        >>> numerical_derivative(f, 2.0, order=2)  # Second derivative at x=2
        12.000000000000668
    """
    # Check order
    if order < 1:
        raise ValueError("Order must be at least 1")
    
    # Determine appropriate epsilon if not provided
    if epsilon is None:
        # Use machine epsilon as a base
        eps = np.finfo(float).eps
        # Scale epsilon based on the magnitude of x and the order
        x_abs = abs(x) if x != 0 else 1.0
        # Higher order derivatives need larger epsilon to avoid numerical issues
        epsilon = x_abs * np.power(eps, 1/(order+1))
        # Ensure epsilon is not too small
        epsilon = max(epsilon, np.power(eps, 1/(order+1)))
    
    # First-order derivative using central difference
    if order == 1:
        try:
            f_plus = func(x + epsilon)
            f_minus = func(x - epsilon)
            return (f_plus - f_minus) / (2.0 * epsilon)
        except Exception as e:
            raise_numeric_error(
                f"Function evaluation failed in numerical_derivative: {str(e)}",
                operation="numerical_derivative",
                values=[x - epsilon, x + epsilon],
                error_type="function_evaluation_error"
            )
    
    # Second-order derivative using central difference
    elif order == 2:
        try:
            f_plus = func(x + epsilon)
            f = func(x)
            f_minus = func(x - epsilon)
            return (f_plus - 2.0 * f + f_minus) / (epsilon * epsilon)
        except Exception as e:
            raise_numeric_error(
                f"Function evaluation failed in numerical_derivative: {str(e)}",
                operation="numerical_derivative",
                values=[x - epsilon, x, x + epsilon],
                error_type="function_evaluation_error"
            )
    
    # Higher-order derivatives using recursive approach
    else:
        # Compute (order-1)th derivative at x+epsilon and x-epsilon
        deriv_plus = numerical_derivative(func, x + epsilon, order - 1, epsilon)
        deriv_minus = numerical_derivative(func, x - epsilon, order - 1, epsilon)
        # Central difference of lower-order derivatives
        return (deriv_plus - deriv_minus) / (2.0 * epsilon)



def numerical_hessian(func: ObjectiveFunction, 
                      x: Vector, 
                      step_sizes: Optional[Vector] = None, 
                      args: Tuple = ()) -> Matrix:
    """
    Compute numerical Hessian of a function with custom step sizes.
    
    This function calculates the Hessian matrix using the two-sided finite
    difference method, but allows for different step sizes for each parameter.
    This can be useful when parameters have very different scales.
    
    Args:
        func: Function to differentiate, should take a vector and return a scalar
        x: Point at which to compute the Hessian
        step_sizes: Vector of step sizes for each parameter. If None, appropriate
                   values are selected based on machine precision and input scale
        args: Additional arguments to pass to the function
        
    Returns:
        Hessian matrix of shape (n, n) where n is the length of x
        
    Raises:
        DimensionError: If x is not a 1D array or if step_sizes is provided but
                       has a different length than x
        NumericError: If the function evaluation produces NaN or Inf values
        
    Examples:
        >>> import numpy as np
        >>> from mfe.utils.differentiation import numerical_hessian
        >>> def f(x): return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2  # Rosenbrock function
        >>> x = np.array([1.0, 1.0])
        >>> # Use different step sizes for each parameter
        >>> step_sizes = np.array([1e-5, 1e-4])
        >>> numerical_hessian(f, x, step_sizes)
        array([[802., -400.],
               [-400.,  200.]])
    """
    # Convert to numpy array if not already
    x = np.asarray(x, dtype=float)
    
    # Check if x is 1D
    if x.ndim != 1:
        raise_dimension_error(
            "Input must be a 1D vector",
            array_name="x",
            expected_shape="(n,)",
            actual_shape=x.shape
        )
    
    n = x.shape[0]
    
    # Determine appropriate step sizes if not provided
    if step_sizes is None:
        # Scale epsilon based on the magnitude of x
        x_abs = np.abs(x)
        # Use machine epsilon as a base
        eps = np.finfo(float).eps
        # For each element, use either a scaled value or a minimum threshold
        step_sizes = np.maximum(x_abs, 1e-8) * np.power(eps, 1/3)
        # Ensure step sizes are not too small
        step_sizes = np.maximum(step_sizes, np.power(eps, 1/3))
    else:
        # Convert to numpy array if not already
        step_sizes = np.asarray(step_sizes, dtype=float)
        
        # Check if step_sizes is 1D and has the same length as x
        if step_sizes.ndim != 1:
            raise_dimension_error(
                "Step sizes must be a 1D vector",
                array_name="step_sizes",
                expected_shape="(n,)",
                actual_shape=step_sizes.shape
            )
        
        if step_sizes.shape[0] != n:
            raise_dimension_error(
                "Step sizes must have the same length as x",
                array_name="step_sizes",
                expected_shape=f"({n},)",
                actual_shape=step_sizes.shape
            )
    
    # Create a wrapper function that includes the additional arguments
    def func_wrapper(x_vec):
        try:
            return func(x_vec, *args)
        except Exception as e:
            raise_numeric_error(
                f"Function evaluation failed in numerical_hessian: {str(e)}",
                operation="numerical_hessian",
                values=x_vec,
                error_type="function_evaluation_error"
            )
    
    # Initialize Hessian matrix
    hess = np.zeros((n, n), dtype=float)
    
    # Create copies for different perturbations
    x_pp = x.copy()  # x + h_i + h_j
    x_pm = x.copy()  # x + h_i - h_j
    x_mp = x.copy()  # x - h_i + h_j
    x_mm = x.copy()  # x - h_i - h_j
    
    # Compute diagonal elements (second derivatives)
    for i in range(n):
        h_i = step_sizes[i]
        
        # f(x + 2*h_i*e_i)
        x_pp[i] = x[i] + 2.0 * h_i
        try:
            f_pp = func_wrapper(x_pp)
        except Exception as e:
            raise_numeric_error(
                f"Function evaluation failed for diagonal element {i}: {str(e)}",
                operation="numerical_hessian",
                values=x_pp,
                error_type="diagonal_evaluation_error"
            )
        
        # f(x)
        try:
            f = func_wrapper(x)
        except Exception as e:
            raise_numeric_error(
                f"Function evaluation failed at center point: {str(e)}",
                operation="numerical_hessian",
                values=x,
                error_type="center_evaluation_error"
            )
        
        # f(x - 2*h_i*e_i)
        x_mm[i] = x[i] - 2.0 * h_i
        try:
            f_mm = func_wrapper(x_mm)
        except Exception as e:
            raise_numeric_error(
                f"Function evaluation failed for diagonal element {i}: {str(e)}",
                operation="numerical_hessian",
                values=x_mm,
                error_type="diagonal_evaluation_error"
            )
        
        # Second derivative approximation
        hess[i, i] = (f_pp - 2.0 * f + f_mm) / (4.0 * h_i * h_i)
        
        # Check for NaN or Inf
        if not np.isfinite(hess[i, i]):
            warn_numeric(
                f"Non-finite Hessian diagonal element detected at index ({i}, {i})",
                operation="numerical_hessian",
                issue="non_finite_hessian",
                value=hess[i, i]
            )
        
        # Reset for next iteration
        x_pp[i] = x[i]
        x_mm[i] = x[i]
    
    # Compute off-diagonal elements (mixed partial derivatives)
    for i in range(n):
        h_i = step_sizes[i]
        for j in range(i+1, n):
            h_j = step_sizes[j]
            
            # f(x + h_i*e_i + h_j*e_j)
            x_pp[i] = x[i] + h_i
            x_pp[j] = x[j] + h_j
            try:
                f_pp = func_wrapper(x_pp)
            except Exception as e:
                raise_numeric_error(
                    f"Function evaluation failed for mixed derivative ({i}, {j}): {str(e)}",
                    operation="numerical_hessian",
                    values=x_pp,
                    error_type="mixed_derivative_error"
                )
            
            # f(x + h_i*e_i - h_j*e_j)
            x_pm[i] = x[i] + h_i
            x_pm[j] = x[j] - h_j
            try:
                f_pm = func_wrapper(x_pm)
            except Exception as e:
                raise_numeric_error(
                    f"Function evaluation failed for mixed derivative ({i}, {j}): {str(e)}",
                    operation="numerical_hessian",
                    values=x_pm,
                    error_type="mixed_derivative_error"
                )
            
            # f(x - h_i*e_i + h_j*e_j)
            x_mp[i] = x[i] - h_i
            x_mp[j] = x[j] + h_j
            try:
                f_mp = func_wrapper(x_mp)
            except Exception as e:
                raise_numeric_error(
                    f"Function evaluation failed for mixed derivative ({i}, {j}): {str(e)}",
                    operation="numerical_hessian",
                    values=x_mp,
                    error_type="mixed_derivative_error"
                )
            
            # f(x - h_i*e_i - h_j*e_j)
            x_mm[i] = x[i] - h_i
            x_mm[j] = x[j] - h_j
            try:
                f_mm = func_wrapper(x_mm)
            except Exception as e:
                raise_numeric_error(
                    f"Function evaluation failed for mixed derivative ({i}, {j}): {str(e)}",
                    operation="numerical_hessian",
                    values=x_mm,
                    error_type="mixed_derivative_error"
                )
            
            # Mixed partial derivative approximation
            hess[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h_i * h_j)
            # Symmetric matrix
            hess[j, i] = hess[i, j]
            
            # Check for NaN or Inf
            if not np.isfinite(hess[i, j]):
                warn_numeric(
                    f"Non-finite Hessian off-diagonal element detected at indices ({i}, {j}) and ({j}, {i})",
                    operation="numerical_hessian",
                    issue="non_finite_hessian",
                    value=hess[i, j]
                )
            
            # Reset for next iteration
            x_pp[i] = x[i]
            x_pp[j] = x[j]
            x_pm[i] = x[i]
            x_pm[j] = x[j]
            x_mp[i] = x[i]
            x_mp[j] = x[j]
            x_mm[i] = x[i]
            x_mm[j] = x[j]
    
    return hess



# Register Numba-accelerated functions if available
def _register_numba_functions() -> None:
    """
    Register Numba JIT-compiled functions for differentiation.
    
    This function is called during module initialization to register
    performance-critical functions for JIT compilation if Numba is available.
    """
    if HAS_NUMBA:
        # The functions are already decorated with @jit, so no additional
        # registration is needed here. This function is kept for consistency
        # with the module structure and potential future enhancements.
        logger.debug("Differentiation Numba JIT functions registered")
    else:
        logger.info("Numba not available. Differentiation will use pure NumPy implementations.")


# Initialize the module
_register_numba_functions()
