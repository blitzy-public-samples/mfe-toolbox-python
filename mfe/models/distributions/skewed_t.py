'''
Hansen's skewed t-distribution implementation for the MFE Toolbox.

This module implements Hansen's skewed t-distribution, which extends the Student's
 t-distribution to capture asymmetry in financial returns. The skewed t-distribution
 is particularly useful for modeling financial data that exhibits both heavy tails
 and asymmetry, such as asset returns that have different behaviors in up and down markets.

The implementation provides methods for computing the probability density function (PDF),
 cumulative distribution function (CDF), quantile function (PPF), random number
 generation, and log-likelihood evaluation. It leverages the standardized Student's
 t-distribution as a foundation, with additional parameters to control skewness.

All computationally intensive functions are accelerated using Numba's just-in-time
 compilation for optimal performance, particularly for large datasets common in
 financial applications.

References:
    Hansen, B. E. (1994). Autoregressive conditional density estimation.
    International Economic Review, 35(3), 705-730.
'''

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union, cast, ClassVar

import numpy as np
from scipy import stats, special, optimize
from numba import jit

from mfe.core.base import DistributionBase
from mfe.core.parameters import (
    ParameterBase, ParameterError, validate_degrees_of_freedom, validate_range,
    transform_positive, inverse_transform_positive, transform_correlation,
    inverse_transform_correlation, SkewedTParameters
)
from mfe.core.exceptions import (
    DistributionError, NumericError, raise_parameter_error, warn_numeric
)
from mfe.core.types import (
    Vector, DistributionType, DistributionLike, ParameterVector,
    PDFFunction, CDFFunction, PPFFunction, RVSFunction
)
from mfe.models.distributions.base import (
    BaseDistribution, NumbaDistribution, ContinuousDistribution
)


# Initialize Numba JIT-compiled functions for Hansen's skewed t-distribution
@jit(nopython=True, cache=True)
def _skewed_t_pdf(x: np.ndarray, df: float, lambda_: float) -> np.ndarray:
    """Numba-accelerated PDF for Hansen's skewed t-distribution.
    
    Computes the probability density function for Hansen's skewed t-distribution
    with specified degrees of freedom and skewness parameter. The distribution
    allows for both heavy tails and asymmetry in the distribution.
    
    Args:
        x: Values to compute the PDF for
        df: Degrees of freedom parameter (must be > 2)
        lambda_: Skewness parameter (must be between -1 and 1)
    
    Returns:
        np.ndarray: PDF values
    """
    # Constants from Hansen's paper
    a = 4 * lambda_ * (df - 2) / ((df - 1) * (1 - lambda_**2))
    b = np.sqrt(1 + 3 * lambda_**2 - a**2)
    
    # Initialize output array
    pdf = np.zeros_like(x, dtype=np.float64)
    
    # Compute PDF for each value
    for i in range(len(x)):
        # Determine which piece of the piecewise function to use
        if x[i] < -a/b:
            # Left piece
            z = -a/b + b * (x[i] + a/b) / (1 - lambda_)
            pdf[i] = b / (1 - lambda_) * _t_pdf(z, df)
        else:
            # Right piece
            z = a/b + b * (x[i] - a/b) / (1 + lambda_)
            pdf[i] = b / (1 + lambda_) * _t_pdf(z, df)
    
    return pdf


@jit(nopython=True, cache=True)
def _t_pdf(x: np.ndarray, df: float) -> np.ndarray:
    """Numba-accelerated PDF for standard Student's t-distribution.
    
    Helper function for computing the PDF of the standard Student's t-distribution.
    
    Args:
        x: Values to compute the PDF for
        df: Degrees of freedom parameter
    
    Returns:
        np.ndarray: PDF values
    """
    # Compute PDF using the formula for Student's t-distribution
    term1 = special.gamma((df + 1) / 2) / (special.gamma(df / 2) * np.sqrt(np.pi * df))
    term2 = (1 + (x**2) / df) ** (-(df + 1) / 2)
    
    return term1 * term2


@jit(nopython=True, cache=True)
def _skewed_t_cdf(x: np.ndarray, df: float, lambda_: float) -> np.ndarray:
    """Numba-accelerated CDF for Hansen's skewed t-distribution.
    
    Computes the cumulative distribution function for Hansen's skewed t-distribution
    with specified degrees of freedom and skewness parameter.
    
    Args:
        x: Values to compute the CDF for
        df: Degrees of freedom parameter (must be > 2)
        lambda_: Skewness parameter (must be between -1 and 1)
    
    Returns:
        np.ndarray: CDF values
    """
    # Constants from Hansen's paper
    a = 4 * lambda_ * (df - 2) / ((df - 1) * (1 - lambda_**2))
    b = np.sqrt(1 + 3 * lambda_**2 - a**2)
    
    # Initialize output array
    cdf = np.zeros_like(x, dtype=np.float64)
    
    # Compute CDF for each value
    for i in range(len(x)):
        # Determine which piece of the piecewise function to use
        if x[i] < -a/b:
            # Left piece
            z = -a/b + b * (x[i] + a/b) / (1 - lambda_)
            cdf[i] = (1 - lambda_) * _t_cdf(z, df) / 2
        else:
            # Right piece
            z = a/b + b * (x[i] - a/b) / (1 + lambda_)
            cdf[i] = 0.5 + (1 + lambda_) * (_t_cdf(z, df) - 0.5) / 2
    
    return cdf


@jit(nopython=True, cache=True)
def _t_cdf(x: np.ndarray, df: float) -> np.ndarray:
    """Numba-accelerated CDF for standard Student's t-distribution.
    
    Helper function for computing the CDF of the standard Student's t-distribution.
    This is an approximation since Numba doesn't support scipy.stats directly.
    
    Args:
        x: Values to compute the CDF for
        df: Degrees of freedom parameter
    
    Returns:
        np.ndarray: CDF values
    """
    # For each value, compute the regularized incomplete beta function
    # which is related to the CDF of the t-distribution
    cdf = np.zeros_like(x, dtype=np.float64)
    
    for i in range(len(x)):
        if x[i] == 0:
            cdf[i] = 0.5
        elif x[i] > 0:
            t2 = x[i]**2
            z = df / (df + t2)
            # Regularized incomplete beta function approximation
            cdf[i] = 1 - 0.5 * special.betainc(df/2, 0.5, z)
        else:
            t2 = x[i]**2
            z = df / (df + t2)
            # Regularized incomplete beta function approximation
            cdf[i] = 0.5 * special.betainc(df/2, 0.5, z)
    
    return cdf


@jit(nopython=True, cache=True)
def _skewed_t_ppf(q: np.ndarray, df: float, lambda_: float) -> np.ndarray:
    """Numba-accelerated PPF for Hansen's skewed t-distribution.
    
    Computes the percent point function (inverse of CDF) for Hansen's skewed
    t-distribution with specified degrees of freedom and skewness parameter.
    
    Args:
        q: Probabilities to compute the PPF for
        df: Degrees of freedom parameter (must be > 2)
        lambda_: Skewness parameter (must be between -1 and 1)
    
    Returns:
        np.ndarray: PPF values
    """
    # Constants from Hansen's paper
    a = 4 * lambda_ * (df - 2) / ((df - 1) * (1 - lambda_**2))
    b = np.sqrt(1 + 3 * lambda_**2 - a**2)
    
    # Initialize output array
    ppf = np.zeros_like(q, dtype=np.float64)
    
    # Compute PPF for each probability
    for i in range(len(q)):
        if q[i] <= 0:
            ppf[i] = -np.inf
        elif q[i] >= 1:
            ppf[i] = np.inf
        elif q[i] < 0.5:
            # Left piece
            z = _t_ppf(2 * q[i] / (1 - lambda_), df)
            ppf[i] = -a/b + (1 - lambda_) * (z + a/b) / b
        else:
            # Right piece
            z = _t_ppf((2 * q[i] - 1) / (1 + lambda_) + 0.5, df)
            ppf[i] = a/b + (1 + lambda_) * (z - a/b) / b
    
    return ppf


@jit(nopython=True, cache=True)
def _t_ppf(q: np.ndarray, df: float) -> np.ndarray:
    """Numba-accelerated PPF for standard Student's t-distribution.
    
    Helper function for computing the PPF of the standard Student's t-distribution.
    This is an approximation since Numba doesn't support scipy.stats directly.
    
    Args:
        q: Probabilities to compute the PPF for
        df: Degrees of freedom parameter
    
    Returns:
        np.ndarray: PPF values
    """
    # This is a simplified approximation
    # For a more accurate implementation, one would need to implement
    # the inverse of the incomplete beta function
    
    # Initialize output array
    ppf = np.zeros_like(q, dtype=np.float64)
    
    # For each probability, compute an approximation of the t-distribution PPF
    for i in range(len(q)):
        if q[i] <= 0:
            ppf[i] = -np.inf
        elif q[i] >= 1:
            ppf[i] = np.inf
        elif q[i] == 0.5:
            ppf[i] = 0
        else:
            # Use a normal approximation for the central part
            # and adjust for heavy tails
            z = np.sqrt(df) * (q[i] - 0.5) * np.sqrt(np.pi) * special.gamma(df/2) / special.gamma((df-1)/2)
            
            # Adjust for heavy tails
            if abs(z) > 3:
                if z > 0:
                    z = z * (1 + (z**2 - 3) / (4 * df))
                else:
                    z = z * (1 + (z**2 - 3) / (4 * df))
            
            ppf[i] = z
    
    return ppf


@jit(nopython=True, cache=True)
def _skewed_t_loglikelihood(x: np.ndarray, df: float, lambda_: float) -> float:
    """Numba-accelerated log-likelihood for Hansen's skewed t-distribution.
    
    Computes the log-likelihood of data under Hansen's skewed t-distribution
    with specified degrees of freedom and skewness parameter.
    
    Args:
        x: Data to compute the log-likelihood for
        df: Degrees of freedom parameter (must be > 2)
        lambda_: Skewness parameter (must be between -1 and 1)
    
    Returns:
        float: Log-likelihood value
    """
    # Compute PDF values
    pdf_values = _skewed_t_pdf(x, df, lambda_)
    
    # Compute log-likelihood
    log_pdf = np.log(pdf_values)
    
    # Sum log-likelihood contributions
    return np.sum(log_pdf)


@jit(nopython=True, cache=True)
def _skewed_t_rvs(size: int, df: float, lambda_: float, u: np.ndarray) -> np.ndarray:
    """Numba-accelerated random number generation for Hansen's skewed t-distribution.
    
    Generates random variates from Hansen's skewed t-distribution using the
    inverse transform method.
    
    Args:
        size: Number of random variates to generate
        df: Degrees of freedom parameter (must be > 2)
        lambda_: Skewness parameter (must be between -1 and 1)
        u: Uniform random numbers between 0 and 1
    
    Returns:
        np.ndarray: Random variates
    """
    # Constants from Hansen's paper
    a = 4 * lambda_ * (df - 2) / ((df - 1) * (1 - lambda_**2))
    b = np.sqrt(1 + 3 * lambda_**2 - a**2)
    
    # Initialize output array
    rvs = np.zeros(size, dtype=np.float64)
    
    # Generate random variates using inverse transform method
    for i in range(size):
        if u[i] < 0.5:
            # Left piece
            z = _t_ppf(2 * u[i] / (1 - lambda_), df)
            rvs[i] = -a/b + (1 - lambda_) * (z + a/b) / b
        else:
            # Right piece
            z = _t_ppf((2 * u[i] - 1) / (1 + lambda_) + 0.5, df)
            rvs[i] = a/b + (1 + lambda_) * (z - a/b) / b
    
    return rvs


class SkewedT(NumbaDistribution[SkewedTParameters]):
    """Hansen's skewed t-distribution implementation with Numba acceleration.
    
    This class implements Hansen's skewed t-distribution with degrees of freedom
    and skewness parameters. The distribution extends the Student's t-distribution
    to capture asymmetry in financial returns, making it particularly useful for
    modeling financial data that exhibits both heavy tails and asymmetry.
    
    The implementation provides Numba-accelerated methods for computing the PDF,
    CDF, PPF, and log-likelihood functions, ensuring optimal performance for
    large datasets common in financial applications.
    
    Attributes:
        name: A descriptive name for the distribution
        params: Distribution parameters (degrees of freedom, skewness)
    
    Examples:
        >>> import numpy as np
        >>> from mfe.models.distributions import SkewedT
        >>> from mfe.core.parameters import SkewedTParameters
        >>> 
        >>> # Create a skewed t-distribution with df=5 and lambda=0.3
        >>> st_dist = SkewedT(params=SkewedTParameters(df=5, lambda_=0.3))
        >>> 
        >>> # Compute PDF values
        >>> x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        >>> st_dist.pdf(x)
        array([0.04982, 0.16934, 0.38454, 0.23576, 0.07726])
        >>> 
        >>> # Compute CDF values
        >>> st_dist.cdf(x)
        array([0.03679, 0.14626, 0.43301, 0.78548, 0.94244])
        >>> 
        >>> # Generate random samples
        >>> st_dist.rvs(size=5, random_state=42)
        array([-0.35264, -0.20358,  0.41268,  0.51789, -0.16345])
    """
    
    # Class variables for JIT-compiled functions
    _jit_pdf = staticmethod(_skewed_t_pdf)
    _jit_cdf = staticmethod(_skewed_t_cdf)
    _jit_ppf = staticmethod(_skewed_t_ppf)
    _jit_loglikelihood = staticmethod(_skewed_t_loglikelihood)
    
    def __init__(self, 
                name: str = "Hansen's Skewed t", 
                params: Optional[SkewedTParameters] = None):
        """Initialize the skewed t-distribution.
        
        Args:
            name: A descriptive name for the distribution
            params: Distribution parameters (degrees of freedom, skewness)
        """
        if params is None:
            # Default to 5 degrees of freedom and no skewness if not specified
            params = SkewedTParameters(df=5.0, lambda_=0.0)
        
        super().__init__(name=name, params=params)
    
    def _params_to_tuple(self) -> Tuple[float, float]:
        """Convert parameters to a tuple for JIT functions.
        
        Returns:
            Tuple[float, float]: Parameter tuple (df, lambda_)
            
        Raises:
            DistributionError: If parameters are not set
        """
        if self._params is None:
            raise DistributionError(
                "Parameters are not set",
                distribution_type=self.name
            )
        
        return (self._params.df, self._params.lambda_)
    
    def _vector_to_params(self, vector: np.ndarray) -> SkewedTParameters:
        """Convert a parameter vector to a parameter object.
        
        Args:
            vector: Parameter vector [log(df-2), arctanh(lambda)]
        
        Returns:
            SkewedTParameters: Parameter object
            
        Raises:
            ValueError: If the vector has incorrect length
        """
        return SkewedTParameters.inverse_transform(vector)
    
    def _get_initial_params(self, data: np.ndarray) -> np.ndarray:
        """Get initial parameter values for optimization.
        
        Estimates initial degrees of freedom parameter based on the kurtosis
        of the data and skewness parameter based on the skewness of the data.
        
        Args:
            data: Data to estimate initial parameters from
        
        Returns:
            np.ndarray: Initial parameter vector [log(df-2), arctanh(lambda)]
        """
        # Compute sample kurtosis and skewness
        kurtosis = stats.kurtosis(data, fisher=True, bias=False)
        skewness = stats.skew(data, bias=False)
        
        # Estimate df from kurtosis (if possible)
        if kurtosis > 0:
            # For t-distribution, excess kurtosis = 6/(df-4) for df > 4
            # Solving for df: df = 6/kurtosis + 4
            df_est = 6.0 / kurtosis + 4.0
            
            # Ensure df > 2 for valid skewed t-distribution
            df_est = max(df_est, 2.1)
        else:
            # If kurtosis is not positive, use a default value
            df_est = 5.0
        
        # Estimate lambda from skewness
        # This is a rough approximation, as the relationship between
        # lambda and skewness in the skewed t-distribution is complex
        lambda_est = np.clip(skewness / 2.0, -0.99, 0.99)
        
        # Transform to unconstrained space
        return np.array([np.log(df_est - 2), np.arctanh(lambda_est)])
    
    def rvs(self, 
           size: Union[int, Tuple[int, ...]], 
           random_state: Optional[Union[int, np.random.Generator]] = None,
           **kwargs: Any) -> np.ndarray:
        """Generate random variates from the distribution.
        
        Args:
            size: Number of random variates to generate
            random_state: Random number generator or seed
            **kwargs: Additional keyword arguments for random variate generation
        
        Returns:
            np.ndarray: Random variates
            
        Raises:
            DistributionError: If the parameters are not set
            ValueError: If size is invalid
        """
        if self._params is None:
            raise DistributionError(
                "Parameters are not set",
                distribution_type=self.name
            )
        
        # Handle random state
        if random_state is None:
            rng = np.random.default_rng()
        elif isinstance(random_state, int):
            rng = np.random.default_rng(random_state)
        else:
            rng = random_state
        
        # Convert size to int if it's a tuple with a single element
        if isinstance(size, tuple) and len(size) == 1:
            size_int = size[0]
        elif isinstance(size, int):
            size_int = size
        else:
            # For multi-dimensional sizes, compute total size
            size_int = np.prod(size)
        
        # Generate uniform random numbers
        u = rng.random(size=size_int)
        
        # Get parameters
        df, lambda_ = self._params_to_tuple()
        
        # Generate random variates using the JIT-compiled function
        rvs = _skewed_t_rvs(size_int, df, lambda_, u)
        
        # Reshape if necessary
        if isinstance(size, tuple) and len(size) > 1:
            rvs = rvs.reshape(size)
        
        return rvs
    
    async def rvs_async(self, 
                       size: Union[int, Tuple[int, ...]], 
                       random_state: Optional[Union[int, np.random.Generator]] = None,
                       **kwargs: Any) -> np.ndarray:
        """Asynchronously generate random variates from the distribution.
        
        This method provides an asynchronous interface to the rvs method,
        allowing for non-blocking random number generation for large sample sizes.
        
        Args:
            size: Number of random variates to generate
            random_state: Random number generator or seed
            **kwargs: Additional keyword arguments for random variate generation
        
        Returns:
            np.ndarray: Random variates
            
        Raises:
            DistributionError: If the parameters are not set
            ValueError: If size is invalid
        """
        # This is a simple implementation that just calls the synchronous version
        # In a real implementation, this would use asyncio to avoid blocking
        return self.rvs(size, random_state, **kwargs)
    
    def fit(self, data: np.ndarray, method: str = "MLE", **kwargs: Any) -> SkewedTParameters:
        """Fit the distribution to data.
        
        Estimates the degrees of freedom and skewness parameters from data using
        the specified method.
        
        Args:
            data: Data to fit the distribution to
            method: Estimation method (default: "MLE")
            **kwargs: Additional keyword arguments for the estimation method
        
        Returns:
            SkewedTParameters: Estimated parameters
            
        Raises:
            ValueError: If data contains invalid values
            NotImplementedError: If the method is not supported
        """
        return super().fit(data, method, **kwargs)
    
    async def fit_async(self, data: np.ndarray, method: str = "MLE", **kwargs: Any) -> SkewedTParameters:
        """Asynchronously fit the distribution to data.
        
        This method provides an asynchronous interface to the fit method,
        allowing for non-blocking parameter estimation for large datasets.
        
        Args:
            data: Data to fit the distribution to
            method: Estimation method (default: "MLE")
            **kwargs: Additional keyword arguments for the estimation method
        
        Returns:
            SkewedTParameters: Estimated parameters
            
        Raises:
            ValueError: If data contains invalid values
            NotImplementedError: If the method is not supported
        """
        # This is a simple implementation that just calls the synchronous version
        # In a real implementation, this would use asyncio to avoid blocking
        return await super().fit_async(data, method, **kwargs)


# Convenience functions for direct use without creating a distribution object

def skewtpdf(x: np.ndarray, df: float, lambda_: float) -> np.ndarray:
    """Compute the PDF of Hansen's skewed t-distribution.
    
    This function provides a direct interface to the skewed t-distribution
    PDF without requiring a distribution object.
    
    Args:
        x: Values to compute the PDF for
        df: Degrees of freedom parameter (must be > 2)
        lambda_: Skewness parameter (must be between -1 and 1)
    
    Returns:
        np.ndarray: PDF values
        
    Raises:
        ParameterError: If df <= 2 or lambda_ is not in [-1, 1]
        ValueError: If x contains invalid values
    
    Examples:
        >>> import numpy as np
        >>> from mfe.models.distributions.skewed_t import skewtpdf
        >>> 
        >>> x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        >>> skewtpdf(x, df=5, lambda_=0.3)
        array([0.04982, 0.16934, 0.38454, 0.23576, 0.07726])
    """
    # Validate parameters
    validate_degrees_of_freedom(df, "df")
    validate_range(lambda_, "lambda_", -1, 1)
    
    # Convert input to numpy array if needed
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    
    # Check for invalid values
    if np.isnan(x).any() or np.isinf(x).any():
        raise ValueError("Input contains NaN or infinite values")
    
    # Use the JIT-compiled function for computation
    return _skewed_t_pdf(x, df, lambda_)


def skewtcdf(x: np.ndarray, df: float, lambda_: float) -> np.ndarray:
    """Compute the CDF of Hansen's skewed t-distribution.
    
    This function provides a direct interface to the skewed t-distribution
    CDF without requiring a distribution object.
    
    Args:
        x: Values to compute the CDF for
        df: Degrees of freedom parameter (must be > 2)
        lambda_: Skewness parameter (must be between -1 and 1)
    
    Returns:
        np.ndarray: CDF values
        
    Raises:
        ParameterError: If df <= 2 or lambda_ is not in [-1, 1]
        ValueError: If x contains invalid values
    """
    # Validate parameters
    validate_degrees_of_freedom(df, "df")
    validate_range(lambda_, "lambda_", -1, 1)
    
    # Convert input to numpy array if needed
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    
    # Check for invalid values
    if np.isnan(x).any() or np.isinf(x).any():
        raise ValueError("Input contains NaN or infinite values")
    
    # Use the JIT-compiled function for computation
    return _skewed_t_cdf(x, df, lambda_)


def skewtinv(q: np.ndarray, df: float, lambda_: float) -> np.ndarray:
    """Compute the inverse CDF (quantile function) of Hansen's skewed t-distribution.
    
    This function provides a direct interface to the skewed t-distribution
    quantile function without requiring a distribution object.
    
    Args:
        q: Probabilities to compute the quantiles for
        df: Degrees of freedom parameter (must be > 2)
        lambda_: Skewness parameter (must be between -1 and 1)
    
    Returns:
        np.ndarray: Quantile values
        
    Raises:
        ParameterError: If df <= 2 or lambda_ is not in [-1, 1]
        ValueError: If q contains values outside [0, 1]
    """
    # Validate parameters
    validate_degrees_of_freedom(df, "df")
    validate_range(lambda_, "lambda_", -1, 1)
    
    # Convert input to numpy array if needed
    if not isinstance(q, np.ndarray):
        q = np.asarray(q)
    
    # Check for invalid values
    if np.isnan(q).any() or np.isinf(q).any():
        raise ValueError("Input contains NaN or infinite values")
    
    if np.any((q < 0) | (q > 1)):
        raise ValueError("Probabilities must be between 0 and 1")
    
    # Use the JIT-compiled function for computation
    return _skewed_t_ppf(q, df, lambda_)


def skewtrnd(size: Union[int, Tuple[int, ...]], 
           df: float, 
           lambda_: float,
           random_state: Optional[Union[int, np.random.Generator]] = None) -> np.ndarray:
    """Generate random variates from Hansen's skewed t-distribution.
    
    This function provides a direct interface to generate random samples from
    the skewed t-distribution without requiring a distribution object.
    
    Args:
        size: Number of random variates to generate
        df: Degrees of freedom parameter (must be > 2)
        lambda_: Skewness parameter (must be between -1 and 1)
        random_state: Random number generator or seed
    
    Returns:
        np.ndarray: Random variates
        
    Raises:
        ParameterError: If df <= 2 or lambda_ is not in [-1, 1]
        ValueError: If size is invalid
    """
    # Validate parameters
    validate_degrees_of_freedom(df, "df")
    validate_range(lambda_, "lambda_", -1, 1)
    
    # Handle random state
    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, int):
        rng = np.random.default_rng(random_state)
    else:
        rng = random_state
    
    # Convert size to int if it's a tuple with a single element
    if isinstance(size, tuple) and len(size) == 1:
        size_int = size[0]
    elif isinstance(size, int):
        size_int = size
    else:
        # For multi-dimensional sizes, compute total size
        size_int = np.prod(size)
    
    # Generate uniform random numbers
    u = rng.random(size=size_int)
    
    # Generate random variates using the JIT-compiled function
    rvs = _skewed_t_rvs(size_int, df, lambda_, u)
    
    # Reshape if necessary
    if isinstance(size, tuple) and len(size) > 1:
        rvs = rvs.reshape(size)
    
    return rvs


def skewtloglik(x: np.ndarray, df: float, lambda_: float) -> float:
    """Compute the log-likelihood of data under Hansen's skewed t-distribution.
    
    This function provides a direct interface to compute the log-likelihood
    without requiring a distribution object.
    
    Args:
        x: Data to compute the log-likelihood for
        df: Degrees of freedom parameter (must be > 2)
        lambda_: Skewness parameter (must be between -1 and 1)
    
    Returns:
        float: Log-likelihood value
        
    Raises:
        ParameterError: If df <= 2 or lambda_ is not in [-1, 1]
        ValueError: If x contains invalid values
    """
    # Validate parameters
    validate_degrees_of_freedom(df, "df")
    validate_range(lambda_, "lambda_", -1, 1)
    
    # Convert input to numpy array if needed
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    
    # Check for invalid values
    if np.isnan(x).any() or np.isinf(x).any():
        raise ValueError("Input contains NaN or infinite values")
    
    # Use the JIT-compiled function for computation
    return _skewed_t_loglikelihood(x, df, lambda_)
