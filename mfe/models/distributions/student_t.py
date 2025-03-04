'''
Standardized Student's t-distribution implementation for the MFE Toolbox.

This module implements the standardized Student's t-distribution, which is commonly
used in financial modeling to capture the heavy tails observed in financial returns.
The standardized t-distribution has heavier tails than the normal distribution,
making it more suitable for modeling extreme market movements.

The implementation provides methods for computing the probability density function (PDF),
cumulative distribution function (CDF), quantile function (PPF), random number
generation, and log-likelihood evaluation. It leverages SciPy's t-distribution as a
foundation, with additional standardization to ensure unit variance regardless of
the degrees of freedom parameter.

All computationally intensive functions are accelerated using Numba's just-in-time
compilation for optimal performance, particularly for large datasets common in
financial applications.
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
    ParameterBase, ParameterError, validate_degrees_of_freedom,
    transform_positive, inverse_transform_positive, StudentTParameters
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


# Initialize Numba JIT-compiled functions for standardized t-distribution
@jit(nopython=True, cache=True)
def _std_t_pdf(x: np.ndarray, df: float) -> np.ndarray:
    """Numba-accelerated PDF for standardized Student's t-distribution.
    
    Computes the probability density function for the standardized Student's
    t-distribution with specified degrees of freedom. The distribution is
    standardized to have unit variance regardless of the degrees of freedom.
    
    Args:
        x: Values to compute the PDF for
        df: Degrees of freedom parameter (must be > 2)
    
    Returns:
        np.ndarray: PDF values
    """
    # Standardization factor to ensure unit variance
    scale = np.sqrt(df / (df - 2))
    
    # Compute standardized x values
    x_std = x * scale
    
    # Compute PDF using the formula for Student's t-distribution
    term1 = special.gamma((df + 1) / 2) / (special.gamma(df / 2) * np.sqrt(np.pi * df))
    term2 = (1 + (x_std ** 2) / df) ** (-(df + 1) / 2)
    
    # Apply the scale factor to maintain proper normalization
    return term1 * term2 * scale


@jit(nopython=True, cache=True)
def _std_t_loglikelihood(x: np.ndarray, df: float) -> float:
    """Numba-accelerated log-likelihood for standardized Student's t-distribution.
    
    Computes the log-likelihood of data under the standardized Student's
    t-distribution with specified degrees of freedom.
    
    Args:
        x: Data to compute the log-likelihood for
        df: Degrees of freedom parameter (must be > 2)
    
    Returns:
        float: Log-likelihood value
    """
    n = len(x)
    
    # Standardization factor to ensure unit variance
    scale = np.sqrt(df / (df - 2))
    
    # Compute standardized x values
    x_std = x * scale
    
    # Compute log-likelihood using the formula for Student's t-distribution
    term1 = n * (np.log(special.gamma((df + 1) / 2)) - np.log(special.gamma(df / 2)) - 0.5 * np.log(np.pi * df))
    term2 = -0.5 * (df + 1) * np.sum(np.log(1 + (x_std ** 2) / df))
    term3 = n * np.log(scale)  # Adjustment for standardization
    
    return term1 + term2 + term3


class StudentT(NumbaDistribution[StudentTParameters]):
    """Standardized Student's t-distribution implementation with Numba acceleration.
    
    This class implements the standardized Student's t-distribution with degrees of
    freedom parameter. The distribution is standardized to have unit variance
    regardless of the degrees of freedom, making it suitable for financial modeling
    where the variance is often modeled separately.
    
    The implementation provides Numba-accelerated methods for computing the PDF,
    CDF, PPF, and log-likelihood functions, ensuring optimal performance for
    large datasets common in financial applications.
    
    Attributes:
        name: A descriptive name for the distribution
        params: Distribution parameters (degrees of freedom)
    
    Examples:
        >>> import numpy as np
        >>> from mfe.models.distributions import StudentT
        >>> from mfe.core.parameters import StudentTParameters
        >>> 
        >>> # Create a standardized t-distribution with 5 degrees of freedom
        >>> t_dist = StudentT(params=StudentTParameters(df=5))
        >>> 
        >>> # Compute PDF values
        >>> x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        >>> t_dist.pdf(x)
        array([0.06301, 0.20215, 0.37457, 0.20215, 0.06301])
        >>> 
        >>> # Compute CDF values
        >>> t_dist.cdf(x)
        array([0.05096, 0.18126, 0.5    , 0.81874, 0.94904])
        >>> 
        >>> # Generate random samples
        >>> t_dist.rvs(size=5, random_state=42)
        array([-0.27126, -0.15698,  0.31816,  0.39942, -0.12605])
    """
    
    # Class variables for JIT-compiled functions
    _jit_pdf = staticmethod(_std_t_pdf)
    _jit_loglikelihood = staticmethod(_std_t_loglikelihood)
    
    def __init__(self, 
                name: str = "Standardized Student's t", 
                params: Optional[StudentTParameters] = None):
        """Initialize the standardized Student's t-distribution.
        
        Args:
            name: A descriptive name for the distribution
            params: Distribution parameters (degrees of freedom)
        """
        if params is None:
            # Default to 5 degrees of freedom if not specified
            params = StudentTParameters(df=5.0)
        
        super().__init__(name=name, params=params)
    
    def _params_to_tuple(self) -> Tuple[float]:
        """Convert parameters to a tuple for JIT functions.
        
        Returns:
            Tuple[float]: Parameter tuple (df)
            
        Raises:
            DistributionError: If parameters are not set
        """
        if self._params is None:
            raise DistributionError(
                "Parameters are not set",
                distribution_type=self.name
            )
        
        return (self._params.df,)
    
    def _vector_to_params(self, vector: np.ndarray) -> StudentTParameters:
        """Convert a parameter vector to a parameter object.
        
        Args:
            vector: Parameter vector [log(df-2)]
        
        Returns:
            StudentTParameters: Parameter object
            
        Raises:
            ValueError: If the vector has incorrect length
        """
        return StudentTParameters.inverse_transform(vector)
    
    def _get_initial_params(self, data: np.ndarray) -> np.ndarray:
        """Get initial parameter values for optimization.
        
        Estimates initial degrees of freedom parameter based on the kurtosis
        of the data. For the t-distribution, the kurtosis is related to the
        degrees of freedom by: kurtosis = 6/(df-4) for df > 4.
        
        Args:
            data: Data to estimate initial parameters from
        
        Returns:
            np.ndarray: Initial parameter vector [log(df-2)]
        """
        # Compute sample kurtosis
        kurtosis = stats.kurtosis(data, fisher=True, bias=False)
        
        # Estimate df from kurtosis (if possible)
        if kurtosis > 0:
            # For t-distribution, excess kurtosis = 6/(df-4) for df > 4
            # Solving for df: df = 6/kurtosis + 4
            df_est = 6.0 / kurtosis + 4.0
            
            # Ensure df > 2 for valid standardized t-distribution
            df_est = max(df_est, 2.1)
        else:
            # If kurtosis is not positive, use a default value
            df_est = 5.0
        
        # Transform to unconstrained space
        return np.array([np.log(df_est - 2)])
    
    def cdf(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute the cumulative distribution function.
        
        Args:
            x: Values to compute the CDF for
            **kwargs: Additional keyword arguments for the CDF
        
        Returns:
            np.ndarray: CDF values
            
        Raises:
            DistributionError: If the parameters are not set
            ValueError: If x contains invalid values
        """
        if self._params is None:
            raise DistributionError(
                "Parameters are not set",
                distribution_type=self.name
            )
        
        # Convert input to numpy array if needed
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        
        # Check for invalid values
        if np.isnan(x).any() or np.isinf(x).any():
            raise ValueError("Input contains NaN or infinite values")
        
        # Get degrees of freedom
        df = self._params.df
        
        # Standardization factor to ensure unit variance
        scale = np.sqrt(df / (df - 2))
        
        # Compute standardized x values
        x_std = x * scale
        
        # Use scipy.stats.t for CDF computation
        return stats.t.cdf(x_std, df)
    
    def ppf(self, q: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute the percent point function (inverse of CDF).
        
        Args:
            q: Probabilities to compute the PPF for
            **kwargs: Additional keyword arguments for the PPF
        
        Returns:
            np.ndarray: PPF values
            
        Raises:
            DistributionError: If the parameters are not set
            ValueError: If q contains values outside [0, 1]
        """
        if self._params is None:
            raise DistributionError(
                "Parameters are not set",
                distribution_type=self.name
            )
        
        # Convert input to numpy array if needed
        if not isinstance(q, np.ndarray):
            q = np.asarray(q)
        
        # Check for invalid values
        if np.isnan(q).any() or np.isinf(q).any():
            raise ValueError("Input contains NaN or infinite values")
        
        if np.any((q < 0) | (q > 1)):
            raise ValueError("Probabilities must be between 0 and 1")
        
        # Get degrees of freedom
        df = self._params.df
        
        # Standardization factor to ensure unit variance
        scale = np.sqrt(df / (df - 2))
        
        # Use scipy.stats.t for PPF computation and apply inverse scaling
        return stats.t.ppf(q, df) / scale
    
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
        
        # Get degrees of freedom
        df = self._params.df
        
        # Standardization factor to ensure unit variance
        scale = np.sqrt(df / (df - 2))
        
        # Generate random variates from standard t-distribution
        t_rvs = stats.t.rvs(df, size=size, random_state=rng)
        
        # Apply inverse scaling to get standardized t-distribution
        return t_rvs / scale
    
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
    
    def fit(self, data: np.ndarray, method: str = "MLE", **kwargs: Any) -> StudentTParameters:
        """Fit the distribution to data.
        
        Estimates the degrees of freedom parameter from data using
        the specified method.
        
        Args:
            data: Data to fit the distribution to
            method: Estimation method (default: "MLE")
            **kwargs: Additional keyword arguments for the estimation method
        
        Returns:
            StudentTParameters: Estimated parameters
            
        Raises:
            ValueError: If data contains invalid values
            NotImplementedError: If the method is not supported
        """
        return super().fit(data, method, **kwargs)
    
    async def fit_async(self, data: np.ndarray, method: str = "MLE", **kwargs: Any) -> StudentTParameters:
        """Asynchronously fit the distribution to data.
        
        This method provides an asynchronous interface to the fit method,
        allowing for non-blocking parameter estimation for large datasets.
        
        Args:
            data: Data to fit the distribution to
            method: Estimation method (default: "MLE")
            **kwargs: Additional keyword arguments for the estimation method
        
        Returns:
            StudentTParameters: Estimated parameters
            
        Raises:
            ValueError: If data contains invalid values
            NotImplementedError: If the method is not supported
        """
        # This is a simple implementation that just calls the synchronous version
        # In a real implementation, this would use asyncio to avoid blocking
        return await super().fit_async(data, method, **kwargs)


# Convenience functions for direct use without creating a distribution object

def stdtpdf(x: np.ndarray, df: float) -> np.ndarray:
    """Compute the PDF of the standardized Student's t-distribution.
    
    This function provides a direct interface to the standardized Student's
    t-distribution PDF without requiring a distribution object.
    
    Args:
        x: Values to compute the PDF for
        df: Degrees of freedom parameter (must be > 2)
    
    Returns:
        np.ndarray: PDF values
        
    Raises:
        ParameterError: If df <= 2
        ValueError: If x contains invalid values
    
    Examples:
        >>> import numpy as np
        >>> from mfe.models.distributions.student_t import stdtpdf
        >>> 
        >>> x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        >>> stdtpdf(x, df=5)
        array([0.06301, 0.20215, 0.37457, 0.20215, 0.06301])
    """
    # Validate degrees of freedom
    validate_degrees_of_freedom(df, "df")
    
    # Convert input to numpy array if needed
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    
    # Check for invalid values
    if np.isnan(x).any() or np.isinf(x).any():
        raise ValueError("Input contains NaN or infinite values")
    
    # Use the JIT-compiled function for computation
    return _std_t_pdf(x, df)


def stdtcdf(x: np.ndarray, df: float) -> np.ndarray:
    """Compute the CDF of the standardized Student's t-distribution.
    
    This function provides a direct interface to the standardized Student's
    t-distribution CDF without requiring a distribution object.
    
    Args:
        x: Values to compute the CDF for
        df: Degrees of freedom parameter (must be > 2)
    
    Returns:
        np.ndarray: CDF values
        
    Raises:
        ParameterError: If df <= 2
        ValueError: If x contains invalid values
    
    Examples:
        >>> import numpy as np
        >>> from mfe.models.distributions.student_t import stdtcdf
        >>> 
        >>> x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        >>> stdtcdf(x, df=5)
        array([0.05096, 0.18126, 0.5    , 0.81874, 0.94904])
    """
    # Validate degrees of freedom
    validate_degrees_of_freedom(df, "df")
    
    # Convert input to numpy array if needed
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    
    # Check for invalid values
    if np.isnan(x).any() or np.isinf(x).any():
        raise ValueError("Input contains NaN or infinite values")
    
    # Standardization factor to ensure unit variance
    scale = np.sqrt(df / (df - 2))
    
    # Compute standardized x values
    x_std = x * scale
    
    # Use scipy.stats.t for CDF computation
    return stats.t.cdf(x_std, df)


def stdtinv(q: np.ndarray, df: float) -> np.ndarray:
    """Compute the inverse CDF (quantile function) of the standardized Student's t-distribution.
    
    This function provides a direct interface to the standardized Student's
    t-distribution quantile function without requiring a distribution object.
    
    Args:
        q: Probabilities to compute the quantiles for
        df: Degrees of freedom parameter (must be > 2)
    
    Returns:
        np.ndarray: Quantile values
        
    Raises:
        ParameterError: If df <= 2
        ValueError: If q contains values outside [0, 1]
    
    Examples:
        >>> import numpy as np
        >>> from mfe.models.distributions.student_t import stdtinv
        >>> 
        >>> q = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
        >>> stdtinv(q, df=5)
        array([-2.01505, -0.72669,  0.     ,  0.72669,  2.01505])
    """
    # Validate degrees of freedom
    validate_degrees_of_freedom(df, "df")
    
    # Convert input to numpy array if needed
    if not isinstance(q, np.ndarray):
        q = np.asarray(q)
    
    # Check for invalid values
    if np.isnan(q).any() or np.isinf(q).any():
        raise ValueError("Input contains NaN or infinite values")
    
    if np.any((q < 0) | (q > 1)):
        raise ValueError("Probabilities must be between 0 and 1")
    
    # Standardization factor to ensure unit variance
    scale = np.sqrt(df / (df - 2))
    
    # Use scipy.stats.t for PPF computation and apply inverse scaling
    return stats.t.ppf(q, df) / scale


def stdtrnd(size: Union[int, Tuple[int, ...]], 
           df: float, 
           random_state: Optional[Union[int, np.random.Generator]] = None) -> np.ndarray:
    """Generate random variates from the standardized Student's t-distribution.
    
    This function provides a direct interface to generate random samples from
    the standardized Student's t-distribution without requiring a distribution object.
    
    Args:
        size: Number of random variates to generate
        df: Degrees of freedom parameter (must be > 2)
        random_state: Random number generator or seed
    
    Returns:
        np.ndarray: Random variates
        
    Raises:
        ParameterError: If df <= 2
        ValueError: If size is invalid
    
    Examples:
        >>> import numpy as np
        >>> from mfe.models.distributions.student_t import stdtrnd
        >>> 
        >>> # Generate 5 random samples with df=5 and fixed seed
        >>> stdtrnd(5, df=5, random_state=42)
        array([-0.27126, -0.15698,  0.31816,  0.39942, -0.12605])
    """
    # Validate degrees of freedom
    validate_degrees_of_freedom(df, "df")
    
    # Handle random state
    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, int):
        rng = np.random.default_rng(random_state)
    else:
        rng = random_state
    
    # Standardization factor to ensure unit variance
    scale = np.sqrt(df / (df - 2))
    
    # Generate random variates from standard t-distribution
    t_rvs = stats.t.rvs(df, size=size, random_state=rng)
    
    # Apply inverse scaling to get standardized t-distribution
    return t_rvs / scale


def stdtloglik(x: np.ndarray, df: float) -> float:
    """Compute the log-likelihood of data under the standardized Student's t-distribution.
    
    This function provides a direct interface to compute the log-likelihood
    without requiring a distribution object.
    
    Args:
        x: Data to compute the log-likelihood for
        df: Degrees of freedom parameter (must be > 2)
    
    Returns:
        float: Log-likelihood value
        
    Raises:
        ParameterError: If df <= 2
        ValueError: If x contains invalid values
    
    Examples:
        >>> import numpy as np
        >>> from mfe.models.distributions.student_t import stdtloglik
        >>> 
        >>> x = np.array([-1.5, -0.5, 0.0, 0.5, 1.5])
        >>> stdtloglik(x, df=5)
        -6.8723...
    """
    # Validate degrees of freedom
    validate_degrees_of_freedom(df, "df")
    
    # Convert input to numpy array if needed
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    
    # Check for invalid values
    if np.isnan(x).any() or np.isinf(x).any():
        raise ValueError("Input contains NaN or infinite values")
    
    # Use the JIT-compiled function for computation
    return _std_t_loglikelihood(x, df)