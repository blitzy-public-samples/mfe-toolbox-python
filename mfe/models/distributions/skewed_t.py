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
from numba import jit, njit
import pandas as pd
import numba

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
from mfe.models.distributions.student_t import _std_t_pdf, _std_t_cdf, _std_t_ppf


# Initialize Numba JIT-compiled functions for Hansen's skewed t-distribution
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
    # Initialize output array
    pdf = np.zeros_like(x)
    
    # For large df, use normal approximation
    if df > 100:
        # Use normal approximation
        pdf = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
        return pdf
    
    # For smaller df, use t-distribution formula
    # The normalizing constant can be precomputed for efficiency
    # We use the fact that for integer df:
    # gamma((df+1)/2) / (gamma(df/2) * sqrt(pi*df)) = 
    # prod(i=1 to df) sqrt((i-1)/i) / sqrt(pi*df)
    
    # Compute normalizing constant
    norm_const = 1.0
    for i in range(1, int(df)):
        norm_const *= np.sqrt((i - 1.0) / i)
    norm_const /= np.sqrt(np.pi * df)
    
    # Compute PDF values
    pdf = norm_const * (1 + x**2 / df)**(-(df + 1) / 2)
    
    return pdf


@jit(nopython=True, cache=True)
def _skewed_t_pdf(x: np.ndarray, df: float, lambda_: float) -> np.ndarray:
    """Compute the PDF of the skewed Student's t distribution.

    Parameters
    ----------
    x : np.ndarray
        Points at which to evaluate the PDF.
    df : float
        Degrees of freedom (must be greater than 2).
    lambda_ : float
        Skewness parameter (must be between -1 and 1).

    Returns
    -------
    np.ndarray
        PDF values at the specified points.
    """
    # Initialize output array
    pdf = np.zeros_like(x, dtype=np.float64)
    
    # Compute standardization factor
    scale = np.sqrt(df / (df - 2)) if df > 2 else 1.0
    x_std = x / scale  # Divide by scale instead of multiply to match standard form

    # Constants for the t-distribution part
    const = _gamma((df + 1) / 2) / (np.sqrt(df * np.pi) * _gamma(df / 2))
    
    # Normalization factor for the skewed t-distribution
    # The correct normalization ensures the PDF integrates to 1
    norm_factor = 2.0 / (1.0 + lambda_**2) if abs(lambda_) > 1e-10 else 1.0
    
    for i in range(len(x)):
        # Base t-distribution PDF
        base_pdf = const * (1 + x_std[i]**2 / df)**(-(df + 1) / 2)
        
        # Skewness adjustment
        if abs(lambda_) < 1e-10:  # Effectively zero
            pdf[i] = base_pdf / scale  # Adjust for the scale change
        else:
            # Apply skewness transformation with correct normalization
            pdf[i] = norm_factor * base_pdf * (1 + lambda_ * np.sign(x_std[i])) / scale
    
    return pdf


@jit(nopython=True, cache=True)
def _gamma(x: float) -> float:
    """Lanczos approximation of the gamma function.
    
    This is a simplified version that works well for x > 0.5
    """
    g = 7
    p = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
         771.32342877765313, -176.61502916214059, 12.507343278686905,
         -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
    
    if x < 0.5:
        return np.pi / (np.sin(np.pi * x) * _gamma(1 - x))
    
    x -= 1
    a = p[0]
    t = x + g + 0.5
    for i in range(1, 9):
        a += p[i] / (x + i)
    
    return np.sqrt(2 * np.pi) * np.power(t, x + 0.5) * np.exp(-t) * a


@jit(nopython=True, cache=True)
def _beta(a: float, b: float) -> float:
    """Compute the beta function using the gamma function."""
    return _gamma(a) * _gamma(b) / _gamma(a + b)


@jit(nopython=True, cache=True)
def _betainc(a: float, b: float, x: float) -> float:
    """Compute the regularized incomplete beta function.
    
    This is a simplified implementation that works well for our use case.
    """
    if x < 0 or x > 1:
        return 0.0
    
    # Use series expansion for x near 0 or 1
    if x < 0.1:
        s = 1.0
        t = 1.0
        for i in range(100):  # Arbitrary limit
            t *= (a + i) * x / ((a + b + i) * (i + 1))
            s += t
            if abs(t) < 1e-10:
                break
        return s * x**a * (1 - x)**b / (a * _beta(a, b))
    elif x > 0.9:
        return 1.0 - _betainc(b, a, 1 - x)
    
    # For middle values, use continued fraction
    fpmin = 1e-30
    m = 100  # Maximum number of iterations
    qab = a + b
    qap = a + 1
    qam = a - 1
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    
    for i in range(1, m + 1):
        m2 = 2 * i
        aa = i * (b - i) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        dl = d * c
        h *= dl
        if abs(dl - 1.0) < 1e-10:
            break
    
    return h * x**a * (1 - x)**b / (a * _beta(a, b))


@jit(nopython=True, cache=True)
def _t_cdf(x: np.ndarray, df: float) -> np.ndarray:
    """Numba-accelerated CDF for standard Student's t-distribution.
    
    Helper function for computing the CDF of the standard Student's t-distribution.
    This uses numerical integration of the PDF.
    
    Args:
        x: Values to compute the CDF for
        df: Degrees of freedom parameter
    
    Returns:
        np.ndarray: CDF values
    """
    # Initialize output array
    cdf = np.zeros_like(x)
    
    # For large df, use normal approximation
    if df > 100:
        # Use normal approximation
        cdf = 0.5 * (1 + np.tanh(x / np.sqrt(2)))
        return cdf
    
    # For each value, integrate the PDF numerically
    for i in range(len(x)):
        if x[i] == 0:
            cdf[i] = 0.5
        elif x[i] == np.inf:
            cdf[i] = 1.0
        elif x[i] == -np.inf:
            cdf[i] = 0.0
        else:
            # For positive x, integrate from -10 (or x, whichever is smaller) to x
            if x[i] > 0:
                start = min(-10.0, x[i])
                t = np.arange(start, x[i] + 0.01, 0.01)
                pdf_vals = _t_pdf(t, df)
                cdf[i] = 0.5 + 0.01 * np.sum(pdf_vals)
            else:
                # For negative x, integrate from x to 10 (or -x, whichever is larger)
                end = max(10.0, -x[i])
                t = np.arange(x[i], end + 0.01, 0.01)
                pdf_vals = _t_pdf(t, df)
                cdf[i] = 0.5 - 0.01 * np.sum(pdf_vals)
    
    # Ensure CDF stays within [0,1]
    cdf = np.clip(cdf, 0.0, 1.0)
    
    return cdf


@jit(nopython=True, cache=True)
def _skewed_t_cdf(x: np.ndarray, df: float, lambda_: float) -> np.ndarray:
    """Compute the CDF of the skewed Student's t distribution.

    Parameters
    ----------
    x : np.ndarray
        Points at which to evaluate the CDF.
    df : float
        Degrees of freedom (must be greater than 2).
    lambda_ : float
        Skewness parameter (must be between -1 and 1).

    Returns
    -------
    np.ndarray
        CDF values at the specified points.
    """
    # Initialize output array
    cdf = np.zeros_like(x, dtype=np.float64)
    
    # For each point, compute the CDF
    for i in range(len(x)):
        # Handle special cases
        if x[i] == np.inf:
            cdf[i] = 1.0
        elif x[i] == -np.inf:
            cdf[i] = 0.0
        else:
            # For symmetric case (lambda_ ≈ 0), use standard Student's t CDF
            if abs(lambda_) < 1e-10:
                # Use the beta incomplete function for the standard t-distribution
                t = x[i]**2 / (df + x[i]**2)
                if x[i] > 0:
                    cdf[i] = 0.5 + 0.5 * _betainc(0.5, df/2, t)
                else:
                    cdf[i] = 0.5 * _betainc(0.5, df/2, t)
            else:
                # For skewed case, use Hansen's formula
                # Calculate the CDF based on the skewness parameter
                if x[i] < 0:
                    # For negative x
                    t = x[i]**2 / (df + x[i]**2)
                    base_cdf = _betainc(0.5, df/2, t)
                    cdf[i] = 0.5 * (1 - lambda_) * base_cdf
                else:
                    # For positive x
                    t = x[i]**2 / (df + x[i]**2)
                    base_cdf = _betainc(0.5, df/2, t)
                    cdf[i] = 0.5 + 0.5 * (1 + lambda_) * (1 - base_cdf)
    
    # Ensure values are in [0, 1]
    result = np.clip(cdf, 0.0, 1.0)
    
    return result


@jit(nopython=True, cache=True)
def _erf(x: float) -> float:
    """Compute the error function using a Taylor series approximation."""
    # Constants for the approximation
    a = [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429]
    p = 0.3275911

    # Save the sign of x
    sign = 1
    if x < 0:
        sign = -1
    x = abs(x)

    # Formula 7.1.26 from Abramowitz and Stegun
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a[4] * t + a[3]) * t + a[2]) * t + a[1]) * t + a[0]) * t * np.exp(-x * x))

    return sign * y


@jit(nopython=True, cache=True)
def _erfinv(x: float) -> float:
    """Compute the inverse error function using a rational approximation."""
    # Handle edge cases
    if x >= 1.0:
        return np.inf
    if x <= -1.0:
        return -np.inf
    if abs(x) < 1e-10:
        return 0.0

    # Constants for the approximation
    a = [0.886226899, -1.645349621, 0.914624893, -0.140543331]
    b = [-2.118377725, 1.442710462, -0.329097515, 0.012229801]
    c = [-1.970840454, -1.624906493, 3.429567803, 1.641345311]
    d = [3.543889200, 1.637067800]

    y = x * x
    if abs(x) <= 0.7:
        z = x * (((a[3] * y + a[2]) * y + a[1]) * y + a[0]) / \
            ((((b[3] * y + b[2]) * y + b[1]) * y + b[0]) * y + 1.0)
    else:
        z = np.sign(x) * np.sqrt(-np.log(1.0 - abs(x))) * \
            (((c[3] * y + c[2]) * y + c[1]) * y + c[0]) / \
            ((d[1] * y + d[0]) * y + 1.0)

    # One iteration of Newton's method to improve accuracy
    z = z - (_erf(z) - x) / (2.0 / np.sqrt(np.pi) * np.exp(-z * z))
    
    return z


@jit(nopython=True, cache=True)
def _skewed_t_ppf(q: np.ndarray, df: float, lambda_: float) -> np.ndarray:
    """Compute the percent point function (PPF) of the skewed Student's t distribution.

    Parameters
    ----------
    q : np.ndarray
        Probabilities at which to evaluate the PPF.
    df : float
        Degrees of freedom (must be greater than 2).
    lambda_ : float
        Skewness parameter (must be between -1 and 1).

    Returns
    -------
    np.ndarray
        PPF values at the specified probabilities.
    """
    # Initialize output array
    x = np.zeros_like(q, dtype=np.float64)
    
    # For each probability, compute the PPF
    for i in range(len(q)):
        p = q[i]
        
        # Handle special cases
        if p <= 0:
            x[i] = -np.inf
            continue
        elif p >= 1:
            x[i] = np.inf
            continue
        
        # For symmetric case (lambda_ ≈ 0), use standard t approximation
        if abs(lambda_) < 1e-10:
            x[i] = _t_quantile_approx(p, df)
            continue
        
        # For skewed case, transform the probability
        if p < 0.5:
            # Left tail
            p_adj = p / (0.5 * (1 - lambda_))
            z = _t_quantile_approx(p_adj, df)
            x[i] = z
        else:
            # Right tail
            p_adj = (p - 0.5) / (0.5 * (1 + lambda_))
            z = _t_quantile_approx(p_adj, df)
            x[i] = z
    
    return x


@jit(nopython=True, cache=True)
def _t_quantile_approx(p: float, df: float) -> float:
    """Approximate quantile function for Student's t-distribution.
    
    This is a simple approximation that works well for most cases.
    
    Args:
        p: Probability, must be in [0, 1]
        df: Degrees of freedom
        
    Returns:
        float: Approximate quantile
    """
    # For p = 0.5, return 0 (median of t-distribution)
    if abs(p - 0.5) < 1e-10:
        return 0.0
    
    # Transform p to standard normal quantile
    z = _normal_quantile_approx(p)
    
    # Adjust for degrees of freedom
    if df > 100:
        # For large df, t-distribution is close to normal
        return z
    else:
        # Simple approximation for t-distribution
        correction = (z**3 + z) / (4 * df)
        return z + correction


@jit(nopython=True, cache=True)
def _normal_quantile_approx(p: float) -> float:
    """Approximate quantile function for standard normal distribution.
    
    This uses a simple approximation that works well for most cases.
    
    Args:
        p: Probability, must be in [0, 1]
        
    Returns:
        float: Approximate quantile
    """
    # Handle special cases
    if p <= 0.0:
        return -10.0
    elif p >= 1.0:
        return 10.0
    elif abs(p - 0.5) < 1e-10:
        return 0.0
    
    # Transform p to be in (0, 1)
    p = max(0.001, min(0.999, p))
    
    # Approximation for normal quantile
    if p < 0.5:
        t = np.sqrt(-2.0 * np.log(p))
        return -t + (2.515517 + 0.802853*t + 0.010328*t*t) / (1.0 + 1.432788*t + 0.189269*t*t + 0.001308*t*t*t)
    else:
        t = np.sqrt(-2.0 * np.log(1.0 - p))
        return t - (2.515517 + 0.802853*t + 0.010328*t*t) / (1.0 + 1.432788*t + 0.189269*t*t + 0.001308*t*t*t)


@jit(nopython=True, cache=True)
def _skewed_t_loglikelihood(x: np.ndarray, df: float, lambda_: float) -> float:
    """Compute the log-likelihood for the skewed Student's t-distribution.
    
    Args:
        x: Data points
        df: Degrees of freedom parameter
        lambda_: Skewness parameter
    
    Returns:
        float: Log-likelihood value
    """
    n = len(x)
    
    # Compute the log-likelihood
    log_lik = n * (_log_gamma((df + 1) / 2) - _log_gamma(df / 2) - 0.5 * np.log(np.pi * df))
    
    # Compute the standardized data
    if lambda_ == 0:
        # For symmetric case, use standard t-distribution
        log_lik -= 0.5 * (df + 1) * np.sum(np.log(1 + x**2 / df))
    else:
        # For skewed case, compute the log-likelihood with skewness
        # Constants for the skewed t-distribution
        c = 2 / (1 + lambda_**2)
        a = lambda_ * np.sqrt(c)
        b = np.sqrt(c)
        
        # Compute the log-likelihood for each data point
        for i in range(n):
            xi = x[i]
            if xi < 0:
                # Left tail
                z = b * xi - a
                log_lik -= 0.5 * (df + 1) * np.log(1 + z**2 / df)
            else:
                # Right tail
                z = a * xi + b
                log_lik -= 0.5 * (df + 1) * np.log(1 + z**2 / df)
        
        # Add the log of the normalization constant
        log_lik += n * np.log(2 / (1 + lambda_**2))
    
    return log_lik


@numba.jit(nopython=True)
def _skewed_t_rvs(size: int, df: float, lambda_: float, u: np.ndarray) -> np.ndarray:
    """
    Generate random variates from a skewed Student's t distribution.
    
    Parameters
    ----------
    size : int
        Number of random variates to generate.
    df : float
        Degrees of freedom.
    lambda_ : float
        Skewness parameter.
    u : np.ndarray
        Array of uniform random numbers between 0 and 1.
    
    Returns
    -------
    np.ndarray
        Array of random variates from the skewed Student's t distribution.
    """
    rvs = np.zeros(size, dtype=np.float64)
    
    # Compute the skewness transformation parameter
    delta = lambda_ / np.sqrt(1 + lambda_**2)
    
    # Generate standard Student's t random variates
    # First generate standard normal random variates using Box-Muller transform
    z = np.zeros(size)
    for i in range(size):
        # Use pairs of uniform random numbers to generate normal random numbers
        if i < size - 1:
            u1 = u[i]
            u2 = u[i + 1]
            r = np.sqrt(-2.0 * np.log(u1))
            theta = 2.0 * np.pi * u2
            z[i] = r * np.cos(theta)
        else:
            # For odd size, use the last uniform random number
            u1 = u[i]
            u2 = 0.5  # Use a constant for the second uniform
            r = np.sqrt(-2.0 * np.log(u1))
            theta = 2.0 * np.pi * u2
            z[i] = r * np.cos(theta)
    
    # Generate chi-square random variates with df degrees of freedom
    chi2 = np.zeros(size)
    for i in range(size):
        # Use the sum of squares of standard normal variables
        # We'll generate these from more uniform random numbers
        chi2_sum = 0.0
        for j in range(int(df)):
            # Use more uniform random numbers to generate normal random numbers
            idx = (i * int(df) + j) % size
            u1 = u[idx]
            idx2 = (i * int(df) + j + 1) % size
            u2 = u[idx2]
            
            r = np.sqrt(-2.0 * np.log(u1))
            theta = 2.0 * np.pi * u2
            x = r * np.cos(theta)
            
            chi2_sum += x * x
        chi2[i] = chi2_sum
    
    # Compute t random variates
    t = np.zeros(size)
    for i in range(size):
        t[i] = z[i] / np.sqrt(chi2[i] / df)
    
    # Apply skewness transformation
    for i in range(size):
        w = t[i]
        if w >= 0:
            rvs[i] = delta * w + np.sqrt((1 - delta**2) * (w**2 + df) / (df + 1))
        else:
            rvs[i] = delta * w - np.sqrt((1 - delta**2) * (w**2 + df) / (df + 1))
    
    return rvs


class SkewedT(DistributionBase):
    """
    Skewed Student's t distribution.

    This class implements Hansen's skewed Student's t distribution with df degrees of freedom
    and lambda_ skewness parameter.

    Parameters
    ----------
    params : SkewedTParameters, optional
        Parameters of the distribution. If not provided, default parameters are used.
    """

    def __init__(self, params: Optional[SkewedTParameters] = None, name: str = "Distribution"):
        """
        Initialize a skewed Student's t distribution.

        Parameters
        ----------
        params : SkewedTParameters, optional
            Parameters for the distribution. Default is None.
        name : str, optional
            Name of the distribution. Default is "Distribution".

        Raises
        ------
        ParameterError
            If the degrees of freedom parameter is not positive or if the skewness parameter
            is not in the range [-1, 1].
        """
        super().__init__(name=name)

        # Initialize with default parameters if none provided
        if params is None:
            params = SkewedTParameters()
        
        self.params = params
        
        if params is not None:
            if params.df <= 0:
                raise ParameterError(
                    "Parameter df (degrees of freedom) must be positive, "
                    f"got {params.df}"
                )
            
            if params.lambda_ < -1 or params.lambda_ > 1:
                raise ParameterError(
                    "Parameter lambda_ (skewness) must be in the range [-1, 1], "
                    f"got {params.lambda_}"
                )
        
        self._jit_pdf = _skewed_t_pdf
        self._jit_cdf = _skewed_t_cdf
        self._jit_ppf = _skewed_t_ppf
        self._jit_rvs = _skewed_t_rvs

    def pdf(self, x: Union[float, np.ndarray, pd.Series, pd.DataFrame]) -> Union[float, np.ndarray, pd.Series, pd.DataFrame]:
        """
        Compute the probability density function (PDF) at the given points.

        Parameters
        ----------
        x : float, ndarray, Series, or DataFrame
            Points at which to evaluate the PDF.

        Returns
        -------
        float, ndarray, Series, or DataFrame
            PDF values.

        Raises
        ------
        ParameterError
            If parameters are not provided.
        ValueError
            If the input contains NaN values or if the input array is empty.
        """
        if self.params is None:
            raise ParameterError("Parameters must be provided")
            
        # Handle different input types
        if isinstance(x, pd.DataFrame):
            # Check for NaN values
            if x.isna().any().any():
                raise ValueError("Input contains NaN values")
                
            # Check for empty DataFrame
            if x.empty:
                raise ValueError("Input DataFrame is empty")
                
            # Create a new DataFrame to store the results
            result = pd.DataFrame(index=x.index)
            
            # Compute PDF for each column
            for col in x.columns:
                result[col] = skewedtpdf(x[col].values, self.params.df, self.params.lambda_)
                
            return result
            
        elif isinstance(x, pd.Series):
            # Check for NaN values
            if x.isna().any():
                raise ValueError("Input contains NaN values")
                
            # Check for empty Series
            if len(x) == 0:
                raise ValueError("Input Series is empty")
                
            return pd.Series(skewedtpdf(x.values, self.params.df, self.params.lambda_), index=x.index)
            
        else:
            # Convert to numpy array if not already
            if not isinstance(x, np.ndarray):
                x = np.array([x])
                
            # Check for NaN values
            if np.isnan(x).any():
                raise ValueError("Input contains NaN values")
                
            # Check for empty array
            if len(x) == 0:
                raise ValueError("Input array is empty")
                
            return skewedtpdf(x, self.params.df, self.params.lambda_)
            
    def cdf(self, x: Union[float, np.ndarray, pd.Series, pd.DataFrame]) -> Union[float, np.ndarray, pd.Series, pd.DataFrame]:
        """
        Compute the cumulative distribution function (CDF) at the given points.

        Parameters
        ----------
        x : float, ndarray, Series, or DataFrame
            Points at which to evaluate the CDF.

        Returns
        -------
        float, ndarray, Series, or DataFrame
            CDF values.

        Raises
        ------
        ParameterError
            If parameters are not provided.
        ValueError
            If the input contains NaN values or if the input array is empty.
        """
        if self.params is None:
            raise ParameterError("Parameters must be provided")
            
        # Handle different input types
        if isinstance(x, pd.DataFrame):
            # Check for NaN values
            if x.isna().any().any():
                raise ValueError("Input contains NaN values")
                
            # Check for empty DataFrame
            if x.empty:
                raise ValueError("Input DataFrame is empty")
                
            # Create a new DataFrame to store the results
            result = pd.DataFrame(index=x.index)
            
            # Compute CDF for each column
            for col in x.columns:
                result[col] = skewedtcdf(x[col].values, self.params.df, self.params.lambda_)
                
            return result
            
        elif isinstance(x, pd.Series):
            # Check for NaN values
            if x.isna().any():
                raise ValueError("Input contains NaN values")
                
            # Check for empty Series
            if len(x) == 0:
                raise ValueError("Input Series is empty")
                
            return pd.Series(skewedtcdf(x.values, self.params.df, self.params.lambda_), index=x.index)
            
        else:
            # Convert to numpy array if not already
            if not isinstance(x, np.ndarray):
                x = np.array([x])
                
            # Check for NaN values
            if np.isnan(x).any():
                raise ValueError("Input contains NaN values")
                
            # Check for empty array
            if len(x) == 0:
                raise ValueError("Input array is empty")
                
            return skewedtcdf(x, self.params.df, self.params.lambda_)
            
    def ppf(self, q: Union[float, np.ndarray, pd.Series, pd.DataFrame]) -> Union[float, np.ndarray, pd.Series, pd.DataFrame]:
        """
        Compute the percent point function (PPF) at the given points.
        
        The PPF is the inverse of the CDF.

        Parameters
        ----------
        q : float, ndarray, Series, or DataFrame
            Points at which to evaluate the PPF. Values should be in the range [0, 1].

        Returns
        -------
        float, ndarray, Series, or DataFrame
            PPF values.

        Raises
        ------
        ParameterError
            If parameters are not provided.
        ValueError
            If the input contains NaN values, if the input array is empty, or if any values are outside the range [0, 1].
        """
        if self.params is None:
            raise ParameterError("Parameters must be provided")
            
        # Handle different input types
        if isinstance(q, pd.DataFrame):
            # Check for NaN values
            if q.isna().any().any():
                raise ValueError("Input contains NaN values")
                
            # Check for empty DataFrame
            if q.empty:
                raise ValueError("Input DataFrame is empty")
                
            # Check for values outside [0, 1]
            if ((q < 0) | (q > 1)).any().any():
                raise ValueError("Input values must be in the range [0, 1]")
                
            # Create a new DataFrame to store the results
            result = pd.DataFrame(index=q.index)
            
            # Compute PPF for each column
            for col in q.columns:
                result[col] = skewedtinv(q[col].values, self.params.df, self.params.lambda_)
                
            return result
            
        elif isinstance(q, pd.Series):
            # Check for NaN values
            if q.isna().any():
                raise ValueError("Input contains NaN values")
                
            # Check for empty Series
            if len(q) == 0:
                raise ValueError("Input Series is empty")
                
            # Check for values outside [0, 1]
            if ((q < 0) | (q > 1)).any():
                raise ValueError("Input values must be in the range [0, 1]")
                
            return pd.Series(skewedtinv(q.values, self.params.df, self.params.lambda_), index=q.index)
            
        else:
            # Convert to numpy array if not already
            if not isinstance(q, np.ndarray):
                q = np.array([q])
                
            # Check for NaN values
            if np.isnan(q).any():
                raise ValueError("Input contains NaN values")
                
            # Check for empty array
            if len(q) == 0:
                raise ValueError("Input array is empty")
                
            # Check for values outside [0, 1]
            if ((q < 0) | (q > 1)).any():
                raise ValueError("Input values must be in the range [0, 1]")
                
            return skewedtinv(q, self.params.df, self.params.lambda_)

    def rvs(self, size: Union[int, tuple] = 1, random_state: Optional[Union[int, np.random.Generator]] = None) -> np.ndarray:
        """
        Generate random samples from the skewed Student's t distribution.

        Parameters
        ----------
        size : int or tuple, optional
            Number of samples to generate or shape of output array. Default is 1.
        random_state : int or np.random.Generator, optional
            Seed for the random number generator or a random number generator. Default is None.

        Returns
        -------
        ndarray
            Random samples from the distribution.
            
        Raises
        ------
        ParameterError
            If parameters are not provided.
        ValueError
            If size is 0 or empty tuple.
        """
        if self.params is None:
            raise ParameterError("Parameters must be provided")
            
        # Check for empty size
        if isinstance(size, int) and size <= 0:
            raise ValueError("Size must be a positive integer")
        elif isinstance(size, tuple) and (len(size) == 0 or any(dim <= 0 for dim in size)):
            raise ValueError("Size tuple must have positive dimensions")
        
        # Use the skewedtrnd function which handles random_state properly
        return skewedtrnd(size=size, df=self.params.df, lambda_=self.params.lambda_, random_state=random_state)

    def loglikelihood(self, x: Union[float, np.ndarray, pd.Series, pd.DataFrame]) -> float:
        """
        Compute the log-likelihood of the data given the distribution parameters.

        Parameters
        ----------
        x : float, ndarray, Series, or DataFrame
            Data points for which to compute the log-likelihood.

        Returns
        -------
        float
            Log-likelihood value.

        Raises
        ------
        ParameterError
            If parameters are not provided.
        ValueError
            If the input contains NaN values or if the input array is empty.
        """
        if self.params is None:
            raise ParameterError("Parameters must be provided")

        # Convert to numpy array if not already
        if isinstance(x, (pd.Series, pd.DataFrame)):
            x_array = x.values.flatten()
        else:
            x_array = np.asarray(x).flatten()

        # Check for empty array
        if len(x_array) == 0:
            raise ValueError("Input array is empty")

        # Check for NaN or infinite values
        if np.isnan(x_array).any() or np.isinf(x_array).any():
            raise ValueError("Input contains NaN or infinite values")

        return skewedtloglik(x_array, self.params.df, self.params.lambda_)
        
    def fit(self, data: Union[np.ndarray, pd.Series, pd.DataFrame], method: str = "MLE", **kwargs: Any) -> SkewedTParameters:
        """
        Fit the distribution to data.
        
        Parameters
        ----------
        data : ndarray, Series, or DataFrame
            Data to fit the distribution to.
        method : str, optional
            Estimation method. Default is "MLE".
        **kwargs : dict
            Additional keyword arguments for the estimation method.
            
        Returns
        -------
        SkewedTParameters
            Estimated parameters.
            
        Raises
        ------
        ValueError
            If data contains invalid values.
        NotImplementedError
            If the method is not supported.
        """
        # Convert data to numpy array
        if isinstance(data, pd.DataFrame):
            if data.shape[1] != 1:
                raise ValueError("DataFrame must have exactly one column")
            data_array = data.iloc[:, 0].values
        elif isinstance(data, pd.Series):
            data_array = data.values
        else:
            data_array = np.asarray(data)
            
        # Check for invalid values
        if np.isnan(data_array).any() or np.isinf(data_array).any():
            raise ValueError("Data contains NaN or infinite values")
            
        if method.upper() == "MLE":
            # Define negative log-likelihood function for optimization
            def neg_loglik(params):
                df, lambda_ = params
                
                # Ensure parameters are valid
                if df <= 2.1 or abs(lambda_) >= 0.99:
                    return 1e10  # Return a large value for invalid parameters
                
                # Compute log-likelihood
                try:
                    log_lik = _skewed_t_loglikelihood(data_array, df, lambda_)
                    # Check for invalid log-likelihood
                    if np.isnan(log_lik) or np.isinf(log_lik):
                        return 1e10
                    return -log_lik
                except:
                    return 1e10
            
            # Initial parameter values
            # Use method of moments for initial estimates
            mean = np.mean(data_array)
            var = np.var(data_array)
            skew = stats.skew(data_array)
            
            # Initial df estimate (higher for less heavy tails)
            init_df = 5.0
            
            # Initial lambda estimate based on skewness
            # Limit to [-0.9, 0.9] to avoid boundary issues
            init_lambda = np.clip(skew / 2, -0.9, 0.9)
            
            # Initial parameter vector
            x0 = np.array([init_df, init_lambda])
            
            # Parameter bounds
            bounds = [(2.1, 100.0), (-0.99, 0.99)]
            
            # Optimize using L-BFGS-B method
            result = optimize.minimize(
                neg_loglik,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'disp': False, 'maxiter': 1000}
            )
            
            # Check if optimization was successful
            if not result.success:
                warnings.warn(f"Optimization did not converge: {result.message}")
                
            # Extract estimated parameters
            est_df, est_lambda = result.x
            
            # Create parameter object
            params = SkewedTParameters(df=est_df, lambda_=est_lambda)
            
            # Update distribution parameters
            self.params = params
            
            return params
        else:
            raise NotImplementedError(
                f"Method {method} is not supported"
            )
            
    async def fit_async(self, data: Union[np.ndarray, pd.Series, pd.DataFrame], method: str = "MLE", **kwargs: Any) -> SkewedTParameters:
        """
        Asynchronously fit the distribution to data.
        
        This method provides an asynchronous interface to the fit method,
        allowing for non-blocking parameter estimation for large datasets.
        
        Parameters
        ----------
        data : ndarray, Series, or DataFrame
            Data to fit the distribution to.
        method : str, optional
            Estimation method. Default is "MLE".
        **kwargs : dict
            Additional keyword arguments for the estimation method.
            
        Returns
        -------
        SkewedTParameters
            Estimated parameters.
            
        Raises
        ------
        ValueError
            If data contains invalid values.
        NotImplementedError
            If the method is not supported.
        """
        # This is a simple implementation that just calls the synchronous version
        return self.fit(data, method, **kwargs)
        
    async def rvs_async(self, size: Union[int, tuple] = 1, random_state: Optional[Union[int, np.random.Generator]] = None) -> np.ndarray:
        """
        Generate random samples from the skewed Student's t distribution asynchronously.

        This is an async wrapper around the rvs method.

        Parameters
        ----------
        size : int or tuple, optional
            Number of samples to generate or shape of output array. Default is 1.
        random_state : int or np.random.Generator, optional
            Seed for the random number generator or a random number generator. Default is None.

        Returns
        -------
        ndarray
            Random samples from the distribution.
        """
        return self.rvs(size=size, random_state=random_state)

    async def loglikelihood_async(self, x: Union[float, np.ndarray, pd.Series, pd.DataFrame]) -> float:
        """
        Compute the log-likelihood of the data under the skewed Student's t distribution asynchronously.

        This is an async wrapper around the loglikelihood method.

        Parameters
        ----------
        x : float, ndarray, Series, or DataFrame
            Data points at which to evaluate the log-likelihood.

        Returns
        -------
        float
            Log-likelihood value.
        """
        return self.loglikelihood(x)


# Convenience functions for direct use without creating a distribution object

def skewedtpdf(x: np.ndarray, df: float, lambda_: float) -> np.ndarray:
    """Compute the PDF of the skewed Student's t-distribution.
    
    This function provides a direct interface to the skewed Student's t-distribution
    PDF without requiring a distribution object.
    
    Args:
        x: Values to compute the PDF for
        df: Degrees of freedom parameter (must be > 2)
        lambda_: Skewness parameter (must be between -1 and 1)
    
    Returns:
        np.ndarray: PDF values
        
    Raises:
        ParameterError: If df <= 2 or |lambda_| >= 1
        ValueError: If x contains invalid values
    """
    # Validate parameters
    if df <= 2:
        raise ParameterError(
            "Parameter df (degrees of freedom) must be greater than 2, "
            f"got {df}"
        )
    if abs(lambda_) >= 1:
        raise ParameterError(
            "Parameter lambda_ (skewness) must be between -1 and 1, "
            f"got {lambda_}"
        )
    
    # Convert input to numpy array if needed
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    
    # Check for invalid values
    if np.isnan(x).any() or np.isinf(x).any():
        raise ValueError("Input contains NaN or infinite values")
    
    # Use the JIT-compiled function for computation
    return _skewed_t_pdf(x, df, lambda_)


def skewedtcdf(x: np.ndarray, df: float, lambda_: float) -> np.ndarray:
    """Compute the CDF of the skewed Student's t-distribution.
    
    This function provides a direct interface to the skewed Student's t-distribution
    CDF without requiring a distribution object.
    
    Args:
        x: Values to compute the CDF for
        df: Degrees of freedom parameter (must be > 2)
        lambda_: Skewness parameter (must be between -1 and 1)
    
    Returns:
        np.ndarray: CDF values
        
    Raises:
        ParameterError: If df <= 2 or |lambda_| >= 1
        ValueError: If x contains NaN values
    """
    # Validate parameters
    if df <= 2:
        raise ParameterError(
            "Parameter df (degrees of freedom) must be greater than 2, "
            f"got {df}"
        )
    if abs(lambda_) >= 1:
        raise ParameterError(
            "Parameter lambda_ (skewness) must be between -1 and 1, "
            f"got {lambda_}"
        )
    
    # Convert input to numpy array if needed
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    
    # Check for invalid values (only NaN, allow inf)
    if np.isnan(x).any():
        raise ValueError("Input contains NaN values")
    
    # Apply scale factor to match scipy.stats implementation
    scale = np.sqrt(df / (df - 2)) if df > 2 else 1.0
    x_scaled = x * scale
    
    # Use the JIT-compiled function for computation
    return _skewed_t_cdf(x_scaled, df, lambda_)


def skewedtinv(q: np.ndarray, df: float, lambda_: float) -> np.ndarray:
    """Compute the inverse CDF (quantile function) of the skewed Student's t-distribution.
    
    This function provides a direct interface to the skewed Student's t-distribution
    quantile function without requiring a distribution object.
    
    Args:
        q: Probabilities to compute the PPF for
        df: Degrees of freedom parameter (must be > 2)
        lambda_: Skewness parameter (must be between -1 and 1)
    
    Returns:
        np.ndarray: PPF values
        
    Raises:
        ParameterError: If df <= 2 or |lambda_| >= 1
        ValueError: If q contains values outside [0, 1]
    """
    # Validate parameters
    if df <= 2:
        raise ParameterError(
            "Parameter df (degrees of freedom) must be greater than 2, "
            f"got {df}"
        )
    if abs(lambda_) >= 1:
        raise ParameterError(
            "Parameter lambda_ (skewness) must be between -1 and 1, "
            f"got {lambda_}"
        )
    
    # Convert input to numpy array if needed
    if not isinstance(q, np.ndarray):
        q = np.asarray(q)
    
    # Check for invalid values
    if np.isnan(q).any() or np.isinf(q).any():
        raise ValueError("Input contains NaN or infinite values")
    
    if np.any((q < 0) | (q > 1)):
        raise ValueError("Probabilities must be between 0 and 1")
    
    # Get raw quantiles from _skewed_t_ppf
    raw_quantiles = _skewed_t_ppf(q, df, lambda_)
    
    # Apply scale factor to match scipy.stats implementation
    # This ensures consistency with the Student's t implementation
    scale = np.sqrt(df / (df - 2)) if df > 2 else 1.0
    return raw_quantiles / scale


def skewedtrnd(size: Union[int, Tuple[int, ...]], 
             df: float, 
             lambda_: float = 0.0,
             random_state: Optional[Union[int, np.random.Generator]] = None) -> np.ndarray:
    """Generate random variates from the skewed Student's t-distribution.
    
    This function provides a direct interface to generate random samples from
    the skewed Student's t-distribution without requiring a distribution object.
    
    Args:
        size: Number of random variates to generate
        df: Degrees of freedom parameter (must be > 2)
        lambda_: Skewness parameter, must be in [-1, 1]
        random_state: Random number generator or seed
    
    Returns:
        np.ndarray: Random variates
        
    Raises:
        ParameterError: If df <= 2 or lambda_ is outside [-1, 1]
        ValueError: If size is invalid
    """
    # Validate parameters
    if df <= 2:
        raise ParameterError(
            "Parameter df (degrees of freedom) must be greater than 2, "
            f"got {df}"
        )
    
    if lambda_ < -1 or lambda_ > 1:
        raise ParameterError(
            "Parameter lambda_ (skewness) must be in [-1, 1], "
            f"got {lambda_}"
        )
    
    # Handle random state
    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, int):
        rng = np.random.default_rng(random_state)
    else:
        rng = random_state
    
    # Convert size to appropriate format
    if isinstance(size, tuple):
        size_tuple = size
    else:
        size_tuple = (size,)
    
    # Total number of samples
    n_samples = np.prod(size_tuple)
    
    # For lambda_ = 0, use standard Student's t
    if abs(lambda_) < 1e-10:
        from scipy import stats
        rvs = stats.t.rvs(df=df, size=size_tuple, random_state=rng)
        # Standardize to ensure variance is 1 for df > 2
        if df > 2:
            scale = np.sqrt((df - 2) / df)
            rvs = rvs * scale
        return rvs
    
    # For skewed t, use the method from Fernandez & Steel (1998)
    # First generate standard t random variates
    from scipy import stats
    z = stats.t.rvs(df=df, size=n_samples, random_state=rng)
    
    # Apply the skewing transformation
    # For positive lambda_, we want positive skewness (longer right tail)
    # For negative lambda_, we want negative skewness (longer left tail)
    # Note: lambda_ is the inverse of the original Fernandez & Steel parameter
    if lambda_ > 0:
        # Right skew (positive skewness)
        # For positive values, multiply by 1/lambda_
        # For negative values, multiply by lambda_
        lambda_inv = 1.0 / lambda_
        rvs = np.where(z >= 0, lambda_inv * z, lambda_ * z)
    else:
        # Left skew (negative skewness)
        # For positive values, multiply by abs(lambda_)
        # For negative values, multiply by 1/abs(lambda_)
        lambda_abs = abs(lambda_)
        lambda_inv = 1.0 / lambda_abs
        rvs = np.where(z >= 0, lambda_abs * z, lambda_inv * z)
    
    # Standardize to ensure variance is 1 for df > 2
    if df > 2:
        # Compute the variance adjustment factor
        c = np.sqrt((df - 2) / df)
        
        # Compute the mean to center the distribution
        # For the Fernandez & Steel parameterization, the mean is:
        # mean = (1/lambda - lambda) * gamma((df+1)/2) / (sqrt(df*pi) * gamma(df/2))
        if lambda_ != 0:
            if lambda_ > 0:
                lambda_param = lambda_
                lambda_inv = 1.0 / lambda_param
            else:
                lambda_param = abs(lambda_)
                lambda_inv = 1.0 / lambda_param
            
            # Calculate the mean adjustment
            mean_factor = (lambda_inv - lambda_param) * _gamma((df + 1) / 2) / (np.sqrt(df * np.pi) * _gamma(df / 2))
            
            # Center and scale
            rvs = c * (rvs - mean_factor)
        else:
            # For lambda_ = 0, the mean is 0
            rvs = c * rvs
    
    # Reshape if necessary
    if len(size_tuple) > 1:
        rvs = rvs.reshape(size_tuple)
    
    return rvs


def skewedtloglik(x: np.ndarray, df: float, lambda_: float) -> float:
    """Compute the log-likelihood of data under the skewed Student's t-distribution.
    
    This function provides a direct interface to compute the log-likelihood
    without requiring a distribution object.
    
    Args:
        x: Data to compute the log-likelihood for
        df: Degrees of freedom parameter (must be > 2)
        lambda_: Skewness parameter (must be between -1 and 1)
    
    Returns:
        float: Log-likelihood value
        
    Raises:
        ParameterError: If df <= 2 or |lambda_| >= 1
        ValueError: If x contains invalid values
    """
    # Validate parameters
    if df <= 2:
        raise ParameterError(
            "Parameter df (degrees of freedom) must be greater than 2, "
            f"got {df}"
        )
    if abs(lambda_) >= 1:
        raise ParameterError(
            "Parameter lambda_ (skewness) must be between -1 and 1, "
            f"got {lambda_}"
        )
    
    # Convert input to numpy array if needed
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    
    # Check for invalid values
    if np.isnan(x).any() or np.isinf(x).any():
        raise ValueError("Input contains NaN or infinite values")
    
    # Use the JIT-compiled function for computation
    return _skewed_t_loglikelihood(x, df, lambda_)


# Create aliases for backward compatibility
SkewedTDistribution = SkewedT
SkewedTParams = SkewedTParameters


@njit
def _log_gamma(x):
    """
    Compute the natural logarithm of the gamma function.
    
    This function uses Stirling's approximation for large values of x
    and a recurrence relation for smaller values.
    
    Parameters
    ----------
    x : float
        Input value, must be positive.
        
    Returns
    -------
    float
        The natural logarithm of the gamma function at x.
    """
    if x <= 0:
        return np.nan
    
    # For large x, use Stirling's approximation
    if x > 10:
        return (x - 0.5) * np.log(x) - x + 0.5 * np.log(2 * np.pi) + 1/(12*x) - 1/(360*x**3)
    
    # For smaller x, use the recurrence relation: Gamma(x+1) = x * Gamma(x)
    # So log(Gamma(x)) = log(Gamma(x+1)) - log(x)
    if x < 1:
        return _log_gamma(x + 1) - np.log(x)
    
    # For x in [1, 10], use a lookup table or compute directly
    if x == 1 or x == 2:
        return 0.0  # log(Gamma(1)) = log(1) = 0, log(Gamma(2)) = log(1) = 0
    
    # For integer values
    if x == np.floor(x):
        result = 0.0
        for i in range(2, int(x)):
            result += np.log(i)
        return result
    
    # For non-integer values in [1, 10], use the recurrence relation
    return _log_gamma(x + 1) - np.log(x)

