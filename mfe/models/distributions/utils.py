# mfe/models/distributions/utils.py
"""
Utility functions for distribution implementations in the MFE Toolbox.

This module provides a comprehensive set of utility functions for implementing
and working with probability distributions in the MFE Toolbox. It includes
functions for input validation, parameter transformations, array dimension
checking, and numerical methods optimized for distribution calculations.

The utilities in this module ensure consistent behavior across all distribution
implementations and enable proper error handling and parameter validation.
Many performance-critical functions are accelerated using Numba's JIT compilation
for optimal performance.
"""

import math
import warnings
from functools import wraps
from typing import (
    Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, TypeVar, 
    Union, cast, overload
)

import numpy as np
from numba import jit
from scipy import stats, special

from mfe.core.exceptions import (
    DistributionError, NumericError, ParameterError, DimensionError,
    raise_parameter_error, warn_numeric
)
from mfe.core.parameters import (
    validate_positive, validate_range, validate_degrees_of_freedom,
    transform_positive, inverse_transform_positive,
    transform_correlation, inverse_transform_correlation
)
from mfe.core.types import (
    Vector, DistributionType, DistributionLike, ParameterVector,
    PDFFunction, CDFFunction, PPFFunction, RVSFunction
)

# Type variables for generic functions
T = TypeVar('T')  # Generic type
P = TypeVar('P')  # Parameter type
D = TypeVar('D')  # Distribution type


def check_array_dimensions(x: np.ndarray, expected_ndim: int,
                          param_name: str = "input") -> np.ndarray:
    """Check that an array has the expected number of dimensions.
    
    Args:
        x: Array to check
        expected_ndim: Expected number of dimensions
        param_name: Name of the parameter for error messages
    
    Returns:
        np.ndarray: The input array, converted to numpy array if needed
    
    Raises:
        DimensionError: If the array doesn't have the expected number of dimensions
    """
    # Convert to numpy array if needed
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    
    # Check dimensions
    if x.ndim != expected_ndim:
        raise DimensionError(
            f"Expected {param_name} to have {expected_ndim} dimensions, got {x.ndim}",
            array_name=param_name,
            expected_shape=f"{expected_ndim}D array",
            actual_shape=x.shape
        )
    
    return x


def check_array_shape(x: np.ndarray, expected_shape: Tuple[int, ...],
                     param_name: str = "input") -> np.ndarray:
    """Check that an array has the expected shape.
    
    Args:
        x: Array to check
        expected_shape: Expected shape (can include None for any size)
        param_name: Name of the parameter for error messages
    
    Returns:
        np.ndarray: The input array, converted to numpy array if needed
    
    Raises:
        DimensionError: If the array doesn't have the expected shape
    """
    # Convert to numpy array if needed
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    
    # Check shape
    if len(x.shape) != len(expected_shape):
        raise DimensionError(
            f"Expected {param_name} to have shape {expected_shape}, got {x.shape}",
            array_name=param_name,
            expected_shape=expected_shape,
            actual_shape=x.shape
        )
    
    for i, (actual, expected) in enumerate(zip(x.shape, expected_shape)):
        if expected is not None and actual != expected:
            raise DimensionError(
                f"Expected {param_name} to have shape {expected_shape}, got {x.shape}",
                array_name=param_name,
                expected_shape=expected_shape,
                actual_shape=x.shape
            )
    
    return x


def check_array_finite(x: np.ndarray, param_name: str = "input") -> np.ndarray:
    """Check that an array contains only finite values.
    
    Args:
        x: Array to check
        param_name: Name of the parameter for error messages
    
    Returns:
        np.ndarray: The input array
    
    Raises:
        ValueError: If the array contains NaN or infinite values
    """
    if np.isnan(x).any() or np.isinf(x).any():
        raise ValueError(f"{param_name} contains NaN or infinite values")
    
    return x


def check_probability_array(q: np.ndarray, param_name: str = "probability") -> np.ndarray:
    """Check that an array contains valid probability values (between 0 and 1).
    
    Args:
        q: Array of probability values to check
        param_name: Name of the parameter for error messages
    
    Returns:
        np.ndarray: The input array, converted to numpy array if needed
    
    Raises:
        ValueError: If the array contains values outside [0, 1]
    """
    # Convert to numpy array if needed
    if not isinstance(q, np.ndarray):
        q = np.asarray(q)
    
    # Check for NaN or infinite values
    check_array_finite(q, param_name)
    
    # Check probability range
    if np.any((q < 0) | (q > 1)):
        raise ValueError(f"{param_name} values must be between 0 and 1")
    
    return q


def check_size_parameter(size: Union[int, Tuple[int, ...]]) -> Tuple[int, ...]:
    """Check and normalize the size parameter for random number generation.
    
    Args:
        size: Size parameter (int or tuple of ints)
    
    Returns:
        Tuple[int, ...]: Normalized size parameter as a tuple
    
    Raises:
        ValueError: If the size parameter is invalid
    """
    if isinstance(size, int):
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        return (size,)
    elif isinstance(size, tuple):
        if not all(isinstance(s, int) and s > 0 for s in size):
            raise ValueError(f"all elements of size must be positive integers, got {size}")
        return size
    else:
        raise ValueError(f"size must be an int or tuple of ints, got {type(size)}")


def is_compatible(x: np.ndarray, y: np.ndarray) -> bool:
    """Check if two arrays are compatible for broadcasting.
    
    Args:
        x: First array
        y: Second array
    
    Returns:
        bool: True if the arrays are compatible for broadcasting
    """
    try:
        # Try to broadcast the shapes
        np.broadcast_shapes(x.shape, y.shape)
        return True
    except ValueError:
        return False


def validate_distribution_parameters(func: Callable) -> Callable:
    """Decorator to validate distribution parameters.
    
    This decorator checks that the distribution parameters are set and valid
    before executing the decorated function.
    
    Args:
        func: Function to decorate
    
    Returns:
        Callable: Decorated function
    
    Raises:
        DistributionError: If parameters are not set
        ParameterError: If parameters are invalid
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Check that parameters are set
        if self._params is None:
            raise DistributionError(
                "Parameters are not set",
                distribution_type=self.name
            )
        
        # Validate parameters
        try:
            self._params.validate()
        except ParameterError as e:
            raise DistributionError(
                f"Invalid parameters for {self.name} distribution: {str(e)}",
                distribution_type=self.name,
                parameter=getattr(e, "param_name", None),
                value=getattr(e, "param_value", None),
                issue=str(e)
            ) from e
        
        # Call the original function
        return func(self, *args, **kwargs)
    
    return wrapper


def validate_input_array(ndim: int = 1, finite: bool = True) -> Callable:
    """Decorator to validate input arrays.
    
    This decorator checks that the first argument to the decorated function
    is a valid numpy array with the expected number of dimensions.
    
    Args:
        ndim: Expected number of dimensions
        finite: Whether to check that the array contains only finite values
    
    Returns:
        Callable: Decorator function
    
    Raises:
        DimensionError: If the array doesn't have the expected number of dimensions
        ValueError: If the array contains NaN or infinite values
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, x, *args, **kwargs):
            # Convert to numpy array if needed
            if not isinstance(x, np.ndarray):
                x = np.asarray(x)
            
            # Check dimensions
            if x.ndim != ndim:
                raise DimensionError(
                    f"Expected input to have {ndim} dimensions, got {x.ndim}",
                    array_name="input",
                    expected_shape=f"{ndim}D array",
                    actual_shape=x.shape
                )
            
            # Check for NaN or infinite values
            if finite and (np.isnan(x).any() or np.isinf(x).any()):
                raise ValueError("Input contains NaN or infinite values")
            
            # Call the original function
            return func(self, x, *args, **kwargs)
        
        return wrapper
    
    return decorator


def validate_probability_input(func: Callable) -> Callable:
    """Decorator to validate probability input arrays.
    
    This decorator checks that the first argument to the decorated function
    is a valid numpy array containing probability values (between 0 and 1).
    
    Args:
        func: Function to decorate
    
    Returns:
        Callable: Decorated function
    
    Raises:
        DimensionError: If the array doesn't have the expected number of dimensions
        ValueError: If the array contains NaN, infinite, or out-of-range values
    """
    @wraps(func)
    def wrapper(self, q, *args, **kwargs):
        # Convert to numpy array if needed
        if not isinstance(q, np.ndarray):
            q = np.asarray(q)
        
        # Check for NaN or infinite values
        if np.isnan(q).any() or np.isinf(q).any():
            raise ValueError("Input contains NaN or infinite values")
        
        # Check probability range
        if np.any((q < 0) | (q > 1)):
            raise ValueError("Probability values must be between 0 and 1")
        
        # Call the original function
        return func(self, q, *args, **kwargs)
    
    return wrapper


def validate_size_parameter(func: Callable) -> Callable:
    """Decorator to validate the size parameter for random number generation.
    
    This decorator checks that the size parameter is a valid int or tuple of ints.
    
    Args:
        func: Function to decorate
    
    Returns:
        Callable: Decorated function
    
    Raises:
        ValueError: If the size parameter is invalid
    """
    @wraps(func)
    def wrapper(self, size, *args, **kwargs):
        # Normalize size parameter
        size_tuple = check_size_parameter(size)
        
        # Call the original function
        return func(self, size_tuple, *args, **kwargs)
    
    return wrapper


@jit(nopython=True, cache=True)
def _standardized_t_pdf(x: np.ndarray, df: float) -> np.ndarray:
    """Numba-accelerated PDF for standardized Student's t distribution.
    
    The standardized t distribution has mean 0 and variance 1 for df > 2.
    
    Args:
        x: Values to compute the PDF for
        df: Degrees of freedom
    
    Returns:
        np.ndarray: PDF values
    """
    # Compute scaling factor for standardization
    scale = np.sqrt(df / (df - 2))
    
    # Compute PDF using the formula
    norm_const = special.gamma((df + 1) / 2) / (special.gamma(df / 2) * np.sqrt(df * np.pi))
    return norm_const * (1 + (x * scale)**2 / df)**(-(df + 1) / 2) * scale


@jit(nopython=True, cache=True)
def _standardized_t_loglikelihood(x: np.ndarray, df: float) -> float:
    """Numba-accelerated log-likelihood for standardized Student's t distribution.
    
    Args:
        x: Data to compute the log-likelihood for
        df: Degrees of freedom
    
    Returns:
        float: Log-likelihood value
    """
    n = len(x)
    scale = np.sqrt(df / (df - 2))
    x_scaled = x * scale
    
    # Compute log-likelihood using the formula
    log_norm_const = (
        special.gammaln((df + 1) / 2) - 
        special.gammaln(df / 2) - 
        0.5 * np.log(df * np.pi)
    )
    
    log_pdf = log_norm_const + np.log(scale) - ((df + 1) / 2) * np.log(1 + x_scaled**2 / df)
    
    return np.sum(log_pdf)


@jit(nopython=True, cache=True)
def _ged_pdf(x: np.ndarray, nu: float) -> np.ndarray:
    """Numba-accelerated PDF for Generalized Error Distribution (GED).
    
    Args:
        x: Values to compute the PDF for
        nu: Shape parameter
    
    Returns:
        np.ndarray: PDF values
    """
    # Compute normalization constant
    lambda_val = np.sqrt(2**(-2/nu) * special.gamma(1/nu) / special.gamma(3/nu))
    norm_const = nu / (2 * lambda_val * special.gamma(1/nu))
    
    # Compute PDF
    return norm_const * np.exp(-(np.abs(x) / lambda_val)**nu / 2)


@jit(nopython=True, cache=True)
def _ged_loglikelihood(x: np.ndarray, nu: float) -> float:
    """Numba-accelerated log-likelihood for Generalized Error Distribution (GED).
    
    Args:
        x: Data to compute the log-likelihood for
        nu: Shape parameter
    
    Returns:
        float: Log-likelihood value
    """
    n = len(x)
    lambda_val = np.sqrt(2**(-2/nu) * special.gamma(1/nu) / special.gamma(3/nu))
    
    # Compute log-likelihood
    log_norm_const = np.log(nu) - np.log(2 * lambda_val) - special.gammaln(1/nu)
    log_pdf = log_norm_const - (np.abs(x) / lambda_val)**nu / 2
    
    return np.sum(log_pdf)


@jit(nopython=True, cache=True)
def _skewed_t_pdf(x: np.ndarray, df: float, lambda_: float) -> np.ndarray:
    """Numba-accelerated PDF for Hansen's skewed t distribution.
    
    Args:
        x: Values to compute the PDF for
        df: Degrees of freedom
        lambda_: Skewness parameter (between -1 and 1)
    
    Returns:
        np.ndarray: PDF values
    """
    # Constants for the distribution
    a = 4 * lambda_ * (df - 2) / ((1 - lambda_**2) * (df - 1))
    b = np.sqrt(1 + 3 * lambda_**2 - a**2)
    
    # Initialize output array
    pdf = np.zeros_like(x, dtype=np.float64)
    
    # Compute PDF for x < -a/b
    mask_left = x < -a/b
    if np.any(mask_left):
        x_left = x[mask_left]
        pdf[mask_left] = (
            b * special.gamma((df + 1) / 2) / 
            (special.gamma(df / 2) * np.sqrt(np.pi * (df - 2)) * (1 + lambda_)) * 
            (1 + 1/(df - 2) * ((b * x_left + a) / (1 - lambda_))**2)**(-(df + 1) / 2)
        )
    
    # Compute PDF for x >= -a/b
    mask_right = ~mask_left
    if np.any(mask_right):
        x_right = x[mask_right]
        pdf[mask_right] = (
            b * special.gamma((df + 1) / 2) / 
            (special.gamma(df / 2) * np.sqrt(np.pi * (df - 2)) * (1 + lambda_)) * 
            (1 + 1/(df - 2) * ((b * x_right + a) / (1 + lambda_))**2)**(-(df + 1) / 2)
        )
    
    return pdf


@jit(nopython=True, cache=True)
def _skewed_t_loglikelihood(x: np.ndarray, df: float, lambda_: float) -> float:
    """Numba-accelerated log-likelihood for Hansen's skewed t distribution.
    
    Args:
        x: Data to compute the log-likelihood for
        df: Degrees of freedom
        lambda_: Skewness parameter (between -1 and 1)
    
    Returns:
        float: Log-likelihood value
    """
    # Constants for the distribution
    a = 4 * lambda_ * (df - 2) / ((1 - lambda_**2) * (df - 1))
    b = np.sqrt(1 + 3 * lambda_**2 - a**2)
    
    # Common term in log-likelihood
    log_common = (
        np.log(b) + 
        special.gammaln((df + 1) / 2) - 
        special.gammaln(df / 2) - 
        0.5 * np.log(np.pi * (df - 2)) - 
        np.log(1 + lambda_)
    )
    
    # Initialize log-likelihood
    log_likelihood = 0.0
    
    # Compute log-likelihood for x < -a/b
    mask_left = x < -a/b
    if np.any(mask_left):
        x_left = x[mask_left]
        log_pdf_left = (
            log_common - 
            (df + 1) / 2 * np.log(1 + 1/(df - 2) * ((b * x_left + a) / (1 - lambda_))**2)
        )
        log_likelihood += np.sum(log_pdf_left)
    
    # Compute log-likelihood for x >= -a/b
    mask_right = ~mask_left
    if np.any(mask_right):
        x_right = x[mask_right]
        log_pdf_right = (
            log_common - 
            (df + 1) / 2 * np.log(1 + 1/(df - 2) * ((b * x_right + a) / (1 + lambda_))**2)
        )
        log_likelihood += np.sum(log_pdf_right)
    
    return log_likelihood


def standardize_data(data: np.ndarray) -> np.ndarray:
    """Standardize data to have mean 0 and variance 1.
    
    Args:
        data: Data to standardize
    
    Returns:
        np.ndarray: Standardized data
    """
    # Convert to numpy array if needed
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    
    # Check for NaN or infinite values
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Data contains NaN or infinite values")
    
    # Standardize data
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    
    if std == 0:
        raise ValueError("Cannot standardize data with zero standard deviation")
    
    return (data - mean) / std


def estimate_distribution_parameters(data: np.ndarray, 
                                    dist_type: DistributionType = "normal") -> Dict[str, float]:
    """Estimate distribution parameters from data.
    
    Args:
        data: Data to estimate parameters from
        dist_type: Type of distribution to estimate parameters for
    
    Returns:
        Dict[str, float]: Estimated parameters
    
    Raises:
        ValueError: If dist_type is not supported
    """
    # Convert to numpy array if needed
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    
    # Check for NaN or infinite values
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Data contains NaN or infinite values")
    
    # Estimate parameters based on distribution type
    if dist_type == "normal":
        # Normal distribution parameters (mu, sigma)
        mu = np.mean(data)
        sigma = np.std(data, ddof=1)
        return {"mu": mu, "sigma": sigma}
    
    elif dist_type == "t":
        # Student's t distribution parameter (df)
        # Estimate df using method of moments based on kurtosis
        kurtosis = stats.kurtosis(data, fisher=True)
        if kurtosis <= 0:
            # If kurtosis is too low, use a default value
            df = 30.0
        else:
            df = 4 + 6 / kurtosis
            # Ensure df > 2 for finite variance
            df = max(df, 2.1)
        
        return {"df": df}
    
    elif dist_type == "skewed_t":
        # Hansen's skewed t distribution parameters (df, lambda_)
        # Estimate df using method of moments based on kurtosis
        kurtosis = stats.kurtosis(data, fisher=True)
        if kurtosis <= 0:
            # If kurtosis is too low, use a default value
            df = 30.0
        else:
            df = 4 + 6 / kurtosis
            # Ensure df > 2 for finite variance
            df = max(df, 2.1)
        
        # Estimate lambda_ based on skewness
        skewness = stats.skew(data)
        # Simple heuristic mapping from skewness to lambda
        lambda_ = np.clip(skewness / 2, -0.99, 0.99)
        
        return {"df": df, "lambda_": lambda_}
    
    elif dist_type == "ged":
        # Generalized Error Distribution parameter (nu)
        # Estimate nu using method of moments based on kurtosis
        kurtosis = stats.kurtosis(data, fisher=True)
        if kurtosis <= 0:
            # If kurtosis is too low, use a default value (normal distribution)
            nu = 2.0
        else:
            # Solve for nu based on kurtosis formula
            # For GED, kurtosis = Gamma(5/nu) * Gamma(1/nu) / (Gamma(3/nu)^2) - 3
            # This is a complex relationship, so we use a simple approximation
            if kurtosis < 3:
                # For kurtosis < 3, nu > 2 (lighter tails than normal)
                nu = 2 + kurtosis / 3
            else:
                # For kurtosis > 3, nu < 2 (heavier tails than normal)
                nu = 2 / (1 + kurtosis / 6)
        
        return {"nu": nu}
    
    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")


def compute_quantile_based_risk_measures(data: np.ndarray, 
                                        alpha: float = 0.05) -> Dict[str, float]:
    """Compute quantile-based risk measures from data.
    
    Args:
        data: Data to compute risk measures from
        alpha: Significance level for risk measures (default: 0.05 for 95% confidence)
    
    Returns:
        Dict[str, float]: Risk measures including Value at Risk (VaR) and 
                         Expected Shortfall (ES)
    
    Raises:
        ValueError: If alpha is not between 0 and 1
    """
    # Validate alpha
    if alpha <= 0 or alpha >= 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    
    # Convert to numpy array if needed
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    
    # Check for NaN or infinite values
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Data contains NaN or infinite values")
    
    # Compute Value at Risk (VaR)
    var = np.quantile(data, alpha)
    
    # Compute Expected Shortfall (ES)
    es = np.mean(data[data <= var])
    
    return {
        "VaR": var,
        "ES": es,
        "alpha": alpha
    }


def compare_distributions(data: np.ndarray, 
                         dist_types: List[DistributionType]) -> Dict[str, Dict[str, Any]]:
    """Compare different distributions for fitting data.
    
    Args:
        data: Data to fit distributions to
        dist_types: List of distribution types to compare
    
    Returns:
        Dict[str, Dict[str, Any]]: Comparison results including parameters,
                                  log-likelihood, AIC, and BIC for each distribution
    
    Raises:
        ValueError: If dist_types is empty or contains unsupported distribution types
    """
    # Validate dist_types
    if not dist_types:
        raise ValueError("dist_types must not be empty")
    
    # Convert to numpy array if needed
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    
    # Check for NaN or infinite values
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Data contains NaN or infinite values")
    
    # Initialize results dictionary
    results = {}
    
    # Fit each distribution and compute metrics
    for dist_type in dist_types:
        # Estimate parameters
        try:
            params = estimate_distribution_parameters(data, dist_type)
        except ValueError as e:
            results[dist_type] = {"error": str(e)}
            continue
        
        # Compute log-likelihood
        try:
            if dist_type == "normal":
                log_likelihood = np.sum(stats.norm.logpdf(
                    data, loc=params["mu"], scale=params["sigma"]
                ))
                num_params = 2
            
            elif dist_type == "t":
                # Standardize data for t distribution
                data_std = (data - np.mean(data)) / np.std(data, ddof=1)
                log_likelihood = _standardized_t_loglikelihood(data_std, params["df"])
                num_params = 3  # df, loc, scale
            
            elif dist_type == "skewed_t":
                # Standardize data for skewed t distribution
                data_std = (data - np.mean(data)) / np.std(data, ddof=1)
                log_likelihood = _skewed_t_loglikelihood(
                    data_std, params["df"], params["lambda_"]
                )
                num_params = 4  # df, lambda_, loc, scale
            
            elif dist_type == "ged":
                # Standardize data for GED
                data_std = (data - np.mean(data)) / np.std(data, ddof=1)
                log_likelihood = _ged_loglikelihood(data_std, params["nu"])
                num_params = 3  # nu, loc, scale
            
            else:
                results[dist_type] = {"error": f"Unsupported distribution type: {dist_type}"}
                continue
            
            # Compute AIC and BIC
            n = len(data)
            aic = -2 * log_likelihood + 2 * num_params
            bic = -2 * log_likelihood + num_params * np.log(n)
            
            # Store results
            results[dist_type] = {
                "parameters": params,
                "log_likelihood": log_likelihood,
                "AIC": aic,
                "BIC": bic,
                "num_params": num_params
            }
            
        except Exception as e:
            results[dist_type] = {"error": str(e)}
    
    return results


def create_qq_plot_data(data: np.ndarray, 
                       dist_type: DistributionType = "normal",
                       params: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Create data for a quantile-quantile (Q-Q) plot.
    
    Args:
        data: Data to create Q-Q plot for
        dist_type: Distribution type for theoretical quantiles
        params: Distribution parameters (if None, estimated from data)
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Theoretical quantiles and sample quantiles
    
    Raises:
        ValueError: If dist_type is not supported
    """
    # Convert to numpy array if needed
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    
    # Check for NaN or infinite values
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Data contains NaN or infinite values")
    
    # Sort data
    data_sorted = np.sort(data)
    
    # Generate probabilities
    n = len(data)
    p = np.arange(1, n + 1) / (n + 1)  # Use (i)/(n+1) for probabilities
    
    # Estimate parameters if not provided
    if params is None:
        params = estimate_distribution_parameters(data, dist_type)
    
    # Generate theoretical quantiles based on distribution type
    if dist_type == "normal":
        mu = params.get("mu", np.mean(data))
        sigma = params.get("sigma", np.std(data, ddof=1))
        theoretical_quantiles = stats.norm.ppf(p, loc=mu, scale=sigma)
    
    elif dist_type == "t":
        df = params.get("df", 5.0)
        # Standardize t distribution
        scale = np.sqrt(df / (df - 2))
        loc = np.mean(data)
        scale_adj = np.std(data, ddof=1) / scale
        theoretical_quantiles = loc + scale_adj * stats.t.ppf(p, df=df)
    
    elif dist_type == "ged":
        nu = params.get("nu", 2.0)
        # This is an approximation as there's no direct ppf for GED in scipy
        # We use a numerical approximation based on the CDF
        theoretical_quantiles = np.zeros_like(p)
        for i, pi in enumerate(p):
            # Find x such that CDF(x) = pi using numerical optimization
            def objective(x):
                # Standardize GED
                lambda_val = np.sqrt(2**(-2/nu) * special.gamma(1/nu) / special.gamma(3/nu))
                # Compute CDF using numerical integration
                cdf = 0.5 + 0.5 * np.sign(x) * special.gammainc(
                    1/nu, (np.abs(x) / lambda_val)**nu / 2
                )
                return (cdf - pi)**2
            
            # Initial guess based on normal distribution
            x0 = stats.norm.ppf(pi)
            result = optimize.minimize_scalar(objective, method='brent')
            theoretical_quantiles[i] = result.x
        
        # Scale and shift to match data
        loc = np.mean(data)
        scale = np.std(data, ddof=1)
        theoretical_quantiles = loc + scale * theoretical_quantiles
    
    elif dist_type == "skewed_t":
        df = params.get("df", 5.0)
        lambda_ = params.get("lambda_", 0.0)
        
        # This is an approximation as there's no direct ppf for skewed t in scipy
        # We use a numerical approximation based on the CDF
        theoretical_quantiles = np.zeros_like(p)
        
        # Constants for the distribution
        a = 4 * lambda_ * (df - 2) / ((1 - lambda_**2) * (df - 1))
        b = np.sqrt(1 + 3 * lambda_**2 - a**2)
        
        for i, pi in enumerate(p):
            # Find x such that CDF(x) = pi using numerical optimization
            def objective(x):
                # Compute CDF using the formula
                if x < -a/b:
                    cdf = (1 - lambda_) * stats.t.cdf(
                        (b * x + a) / (1 - lambda_), df=df
                    )
                else:
                    cdf = (1 - lambda_) + (1 + lambda_) * (
                        stats.t.cdf((b * x + a) / (1 + lambda_), df=df) - 0.5
                    )
                return (cdf - pi)**2
            
            # Initial guess based on normal distribution
            x0 = stats.norm.ppf(pi)
            result = optimize.minimize_scalar(objective, method='brent')
            theoretical_quantiles[i] = result.x
        
        # Scale and shift to match data
        loc = np.mean(data)
        scale = np.std(data, ddof=1)
        theoretical_quantiles = loc + scale * theoretical_quantiles
    
    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")
    
    return theoretical_quantiles, data_sorted


def transform_to_uniform(data: np.ndarray, 
                        dist_type: DistributionType = "normal",
                        params: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Transform data to uniform distribution using probability integral transform.
    
    Args:
        data: Data to transform
        dist_type: Distribution type for the transformation
        params: Distribution parameters (if None, estimated from data)
    
    Returns:
        np.ndarray: Transformed data (should be uniformly distributed if the
                   distribution is correctly specified)
    
    Raises:
        ValueError: If dist_type is not supported
    """
    # Convert to numpy array if needed
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    
    # Check for NaN or infinite values
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Data contains NaN or infinite values")
    
    # Estimate parameters if not provided
    if params is None:
        params = estimate_distribution_parameters(data, dist_type)
    
    # Transform data based on distribution type
    if dist_type == "normal":
        mu = params.get("mu", np.mean(data))
        sigma = params.get("sigma", np.std(data, ddof=1))
        transformed = stats.norm.cdf(data, loc=mu, scale=sigma)
    
    elif dist_type == "t":
        df = params.get("df", 5.0)
        # Standardize data
        data_std = (data - np.mean(data)) / np.std(data, ddof=1)
        # Apply t CDF
        transformed = stats.t.cdf(data_std, df=df)
    
    elif dist_type == "ged":
        nu = params.get("nu", 2.0)
        # Standardize data
        data_std = (data - np.mean(data)) / np.std(data, ddof=1)
        
        # Compute CDF for GED
        transformed = np.zeros_like(data_std)
        lambda_val = np.sqrt(2**(-2/nu) * special.gamma(1/nu) / special.gamma(3/nu))
        
        for i, x in enumerate(data_std):
            # Compute CDF using the formula
            transformed[i] = 0.5 + 0.5 * np.sign(x) * special.gammainc(
                1/nu, (np.abs(x) / lambda_val)**nu / 2
            )
    
    elif dist_type == "skewed_t":
        df = params.get("df", 5.0)
        lambda_ = params.get("lambda_", 0.0)
        
        # Standardize data
        data_std = (data - np.mean(data)) / np.std(data, ddof=1)
        
        # Constants for the distribution
        a = 4 * lambda_ * (df - 2) / ((1 - lambda_**2) * (df - 1))
        b = np.sqrt(1 + 3 * lambda_**2 - a**2)
        
        # Compute CDF for skewed t
        transformed = np.zeros_like(data_std)
        for i, x in enumerate(data_std):
            if x < -a/b:
                transformed[i] = (1 - lambda_) * stats.t.cdf(
                    (b * x + a) / (1 - lambda_), df=df
                )
            else:
                transformed[i] = (1 - lambda_) + (1 + lambda_) * (
                    stats.t.cdf((b * x + a) / (1 + lambda_), df=df) - 0.5
                )
    
    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")
    
    return transformed


def berkowitz_test(data: np.ndarray, 
                  dist_type: DistributionType = "normal",
                  params: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Perform Berkowitz test for distribution specification.
    
    The Berkowitz test transforms data to normal using the probability integral
    transform and tests for normality using an AR(1) model.
    
    Args:
        data: Data to test
        dist_type: Distribution type for the transformation
        params: Distribution parameters (if None, estimated from data)
    
    Returns:
        Dict[str, Any]: Test results including test statistic, p-value, and parameters
    
    Raises:
        ValueError: If dist_type is not supported
    """
    # Convert to numpy array if needed
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    
    # Check for NaN or infinite values
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Data contains NaN or infinite values")
    
    # Transform data to uniform using probability integral transform
    u = transform_to_uniform(data, dist_type, params)
    
    # Transform uniform to normal
    z = stats.norm.ppf(u)
    
    # Check for infinite values (can happen if u is exactly 0 or 1)
    z = np.clip(z, -8, 8)
    
    # Fit AR(1) model to z
    # z_t = mu + rho * z_{t-1} + e_t, e_t ~ N(0, sigma^2)
    z_lag = z[:-1]
    z_curr = z[1:]
    
    # Add constant column for regression
    X = np.column_stack((np.ones_like(z_lag), z_lag))
    
    # Estimate parameters using OLS
    beta = np.linalg.lstsq(X, z_curr, rcond=None)[0]
    mu = beta[0]
    rho = beta[1]
    
    # Compute residuals
    residuals = z_curr - (mu + rho * z_lag)
    
    # Estimate sigma^2
    sigma2 = np.mean(residuals**2)
    
    # Compute log-likelihood for the AR(1) model
    log_likelihood_ar1 = -0.5 * len(z_curr) * (np.log(2 * np.pi) + np.log(sigma2)) - 0.5 * np.sum(residuals**2) / sigma2
    
    # Compute log-likelihood for the null model (iid standard normal)
    log_likelihood_null = -0.5 * len(z) * np.log(2 * np.pi) - 0.5 * np.sum(z**2)
    
    # Compute likelihood ratio test statistic
    lr_stat = -2 * (log_likelihood_null - log_likelihood_ar1)
    
    # Compute p-value (chi-squared with 3 degrees of freedom)
    p_value = 1 - stats.chi2.cdf(lr_stat, 3)
    
    return {
        "test_statistic": lr_stat,
        "p_value": p_value,
        "mu": mu,
        "rho": rho,
        "sigma2": sigma2,
        "null_hypothesis": "Data follows the specified distribution"
    }


def kolmogorov_smirnov_test(data: np.ndarray, 
                           dist_type: DistributionType = "normal",
                           params: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Perform Kolmogorov-Smirnov test for distribution specification.
    
    Args:
        data: Data to test
        dist_type: Distribution type for the test
        params: Distribution parameters (if None, estimated from data)
    
    Returns:
        Dict[str, Any]: Test results including test statistic, p-value, and parameters
    
    Raises:
        ValueError: If dist_type is not supported
    """
    # Convert to numpy array if needed
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    
    # Check for NaN or infinite values
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Data contains NaN or infinite values")
    
    # Estimate parameters if not provided
    if params is None:
        params = estimate_distribution_parameters(data, dist_type)
    
    # Create CDF function based on distribution type
    if dist_type == "normal":
        mu = params.get("mu", np.mean(data))
        sigma = params.get("sigma", np.std(data, ddof=1))
        cdf = lambda x: stats.norm.cdf(x, loc=mu, scale=sigma)
    
    elif dist_type == "t":
        df = params.get("df", 5.0)
        # Standardize t distribution
        scale = np.sqrt(df / (df - 2))
        loc = np.mean(data)
        scale_adj = np.std(data, ddof=1) / scale
        cdf = lambda x: stats.t.cdf((x - loc) / scale_adj, df=df)
    
    elif dist_type == "ged":
        nu = params.get("nu", 2.0)
        # Standardize data
        loc = np.mean(data)
        scale = np.std(data, ddof=1)
        
        def cdf(x):
            # Standardize x
            x_std = (x - loc) / scale
            # Compute CDF for GED
            lambda_val = np.sqrt(2**(-2/nu) * special.gamma(1/nu) / special.gamma(3/nu))
            return 0.5 + 0.5 * np.sign(x_std) * special.gammainc(
                1/nu, (np.abs(x_std) / lambda_val)**nu / 2
            )
    
    elif dist_type == "skewed_t":
        df = params.get("df", 5.0)
        lambda_ = params.get("lambda_", 0.0)
        
        # Standardize data
        loc = np.mean(data)
        scale = np.std(data, ddof=1)
        
        # Constants for the distribution
        a = 4 * lambda_ * (df - 2) / ((1 - lambda_**2) * (df - 1))
        b = np.sqrt(1 + 3 * lambda_**2 - a**2)
        
        def cdf(x):
            # Standardize x
            x_std = (x - loc) / scale
            # Compute CDF for skewed t
            if x_std < -a/b:
                return (1 - lambda_) * stats.t.cdf(
                    (b * x_std + a) / (1 - lambda_), df=df
                )
            else:
                return (1 - lambda_) + (1 + lambda_) * (
                    stats.t.cdf((b * x_std + a) / (1 + lambda_), df=df) - 0.5
                )
    
    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")
    
    # Perform Kolmogorov-Smirnov test
    # Note: Since we're estimating parameters from the data, the p-value is approximate
    ks_stat, p_value = stats.kstest(data, cdf)
    
    return {
        "test_statistic": ks_stat,
        "p_value": p_value,
        "parameters": params,
        "null_hypothesis": "Data follows the specified distribution"
    }


def anderson_darling_test(data: np.ndarray, 
                         dist_type: DistributionType = "normal") -> Dict[str, Any]:
    """Perform Anderson-Darling test for distribution specification.
    
    Args:
        data: Data to test
        dist_type: Distribution type for the test
    
    Returns:
        Dict[str, Any]: Test results including test statistic, critical values, and significance
    
    Raises:
        ValueError: If dist_type is not supported
    """
    # Convert to numpy array if needed
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    
    # Check for NaN or infinite values
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Data contains NaN or infinite values")
    
    # Currently, scipy only supports 'normal', 'expon', 'logistic', 'gumbel', 'gumbel_l', 'gumbel_r', 'extreme1'
    # We'll implement only for normal and raise errors for others
    if dist_type == "normal":
        result = stats.anderson(data, dist='norm')
        
        # Extract results
        statistic = result.statistic
        critical_values = result.critical_values
        significance_levels = result.significance_level / 100  # Convert from percentage
        
        # Determine if null hypothesis is rejected at different significance levels
        results = {}
        for sig_level, crit_val in zip(significance_levels, critical_values):
            results[f"rejected_at_{sig_level:.3f}"] = statistic > crit_val
        
        return {
            "test_statistic": statistic,
            "critical_values": dict(zip(significance_levels, critical_values)),
            "results": results,
            "null_hypothesis": "Data follows the normal distribution"
        }
    else:
        raise ValueError(f"Anderson-Darling test currently only supports 'normal' distribution, got {dist_type}")


def jarque_bera_test(data: np.ndarray) -> Dict[str, Any]:
    """Perform Jarque-Bera test for normality.
    
    Args:
        data: Data to test
    
    Returns:
        Dict[str, Any]: Test results including test statistic, p-value, skewness, and kurtosis
    """
    # Convert to numpy array if needed
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    
    # Check for NaN or infinite values
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Data contains NaN or infinite values")
    
    # Compute test statistic and p-value
    jb_stat, p_value = stats.jarque_bera(data)
    
    # Compute skewness and kurtosis
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data, fisher=True)  # Fisher's definition (excess kurtosis)
    
    return {
        "test_statistic": jb_stat,
        "p_value": p_value,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "null_hypothesis": "Data follows the normal distribution"
    }


def shapiro_wilk_test(data: np.ndarray) -> Dict[str, Any]:
    """Perform Shapiro-Wilk test for normality.
    
    Args:
        data: Data to test
    
    Returns:
        Dict[str, Any]: Test results including test statistic and p-value
    """
    # Convert to numpy array if needed
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    
    # Check for NaN or infinite values
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Data contains NaN or infinite values")
    
    # Check data size (Shapiro-Wilk is recommended for n between 3 and 5000)
    n = len(data)
    if n < 3 or n > 5000:
        warnings.warn(
            f"Shapiro-Wilk test is recommended for sample sizes between 3 and 5000, got {n}",
            UserWarning
        )
    
    # Compute test statistic and p-value
    w_stat, p_value = stats.shapiro(data)
    
    return {
        "test_statistic": w_stat,
        "p_value": p_value,
        "null_hypothesis": "Data follows the normal distribution"
    }


def lilliefors_test(data: np.ndarray) -> Dict[str, Any]:
    """Perform Lilliefors test for normality.
    
    The Lilliefors test is a modification of the Kolmogorov-Smirnov test
    that accounts for the fact that parameters are estimated from the data.
    
    Args:
        data: Data to test
    
    Returns:
        Dict[str, Any]: Test results including test statistic and p-value
    """
    # Convert to numpy array if needed
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    
    # Check for NaN or infinite values
    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Data contains NaN or infinite values")
    
    # Standardize data
    data_std = (data - np.mean(data)) / np.std(data, ddof=1)
    
    # Compute Kolmogorov-Smirnov statistic
    ks_stat = stats.kstest(data_std, 'norm').statistic
    
    # Compute p-value using Monte Carlo simulation
    # (This is an approximation as the exact distribution is complex)
    n = len(data)
    n_simulations = 1000
    ks_stats_sim = np.zeros(n_simulations)
    
    for i in range(n_simulations):
        # Generate random normal data
        x_sim = np.random.normal(size=n)
        # Standardize
        x_sim_std = (x_sim - np.mean(x_sim)) / np.std(x_sim, ddof=1)
        # Compute KS statistic
        ks_stats_sim[i] = stats.kstest(x_sim_std, 'norm').statistic
    
    # Compute p-value as the proportion of simulated statistics >= observed statistic
    p_value = np.mean(ks_stats_sim >= ks_stat)
    
    return {
        "test_statistic": ks_stat,
        "p_value": p_value,
        "null_hypothesis": "Data follows the normal distribution"
    }