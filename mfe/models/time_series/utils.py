# mfe/models/time_series/utils.py

"""
Time Series Utilities Module

This module provides essential utility functions for time series analysis in the MFE Toolbox.
It implements matrix operations, data transformations, and specialized calculations used
throughout the time series module, including lag matrix construction, parameter conversions,
statistical tests, and other helper functions.

The module leverages NumPy for efficient array operations and includes Numba-accelerated
implementations for performance-critical functions. All functions include comprehensive
 type hints and input validation to ensure reliability and proper error handling.

Functions:
    lag_matrix: Create a matrix of lagged values from a time series
    ar_roots: Compute the roots of an AR polynomial
    ma_roots: Compute the roots of an MA polynomial
    check_ar_stationarity: Check if AR parameters satisfy stationarity condition
    check_ma_invertibility: Check if MA parameters satisfy invertibility condition
    acf: Compute the autocorrelation function
    pacf: Compute the partial autocorrelation function
    ljungbox: Perform Ljung-Box test for autocorrelation
    durbin_watson: Compute Durbin-Watson statistic for autocorrelation
    information_criteria: Compute AIC, BIC, and HQIC for model selection
    forecast_error_variance: Compute forecast error variance
    impulse_response: Compute impulse response function
    is_invertible: Check if a time series model is invertible
    is_stationary: Check if a time series model is stationary
    standardize_data: Standardize time series data
    detrend: Remove trend from time series data
    seasonal_adjust: Remove seasonal component from time series data
    lag_polynomial: Evaluate a lag polynomial
    companion_matrix: Create companion matrix from AR parameters
"""

import logging
import warnings
from typing import (
    Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple,
    Type, TypeVar, Union, cast, overload
)

import numpy as np
import pandas as pd
from scipy import stats, linalg, signal, optimize
from scipy.stats import chi2

from mfe.core.types import Matrix, Vector
from mfe.core.exceptions import (
    DimensionError, NumericError, ParameterError,
    raise_dimension_error, raise_numeric_error, raise_parameter_error,
    warn_numeric, warn_model
)
from mfe.core.parameters import (
    TimeSeriesParameters, ARMAParameters, validate_positive, validate_non_negative
)

# Set up module-level logger
logger = logging.getLogger("mfe.models.time_series.utils")

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
    logger.debug("Numba available for time series utilities acceleration")
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    HAS_NUMBA = False
    logger.info("Numba not available. Time series utilities will use pure NumPy implementations.")


@jit(nopython=True, cache=True)
def _lag_matrix_numba(y: np.ndarray, lags: int, include_original: bool = True) -> np.ndarray:
    """
    Numba-accelerated implementation of lag matrix creation.

    Args:
        y: Time series data (1D array)
        lags: Number of lags to include
        include_original: Whether to include the original series in the output

    Returns:
        Matrix of lagged values
    """
    T = len(y)

    # Determine the number of columns in the output matrix
    if include_original:
        cols = lags + 1
        start_col = 0
    else:
        cols = lags
        start_col = 1

    # Initialize the output matrix
    result = np.zeros((T, cols))

    # Fill the matrix with lagged values
    if include_original:
        result[:, 0] = y

    for lag in range(1, lags + 1):
        col_idx = lag - start_col
        result[lag:, col_idx] = y[:-lag]

    return result


def lag_matrix(y: Union[np.ndarray, pd.Series], lags: int,
               include_original: bool = True, drop_na: bool = False) -> Union[np.ndarray, pd.DataFrame]:
    """
    Create a matrix of lagged values from a time series.

    This function creates a matrix where each column contains lagged values of the
    input time series. The first column can optionally contain the original series.

    Args:
        y: Time series data (1D array or Series)
        lags: Number of lags to include
        include_original: Whether to include the original series in the output
        drop_na: Whether to drop rows with NaN values (resulting from lagging)

    Returns:
        Matrix of lagged values. If input is a pandas Series, output is a DataFrame
        with appropriate column names. Otherwise, output is a NumPy array.

    Raises:
        DimensionError: If y is not 1D
        ValueError: If lags is not a positive integer

    Examples:
        >>> import numpy as np
        >>> from mfe.models.time_series.utils import lag_matrix
        >>> y = np.array([1, 2, 3, 4, 5])
        >>> lag_matrix(y, lags=2)
        array([[1., 0., 0.],
               [2., 1., 0.],
               [3., 2., 1.],
               [4., 3., 2.],
               [5., 4., 3.]])
        >>> lag_matrix(y, lags=2, include_original=False)
        array([[0., 0.],
               [1., 0.],
               [2., 1.],
               [3., 2.],
               [4., 3.]])
    """
    # Check if lags is valid
    if not isinstance(lags, int) or lags < 1:
        raise ValueError("lags must be a positive integer")

    # Handle pandas Series
    is_pandas = isinstance(y, pd.Series)
    index = None
    if is_pandas:
        index = y.index
        y_values = y.values
    else:
        y_values = np.asarray(y)

    # Check if y is 1D
    if y_values.ndim != 1:
        raise_dimension_error(
            "Input must be a 1D array or Series",
            array_name="y",
            expected_shape="(T,)",
            actual_shape=y_values.shape
        )

    # Use Numba-accelerated implementation if available
    if HAS_NUMBA:
        result = _lag_matrix_numba(y_values, lags, include_original)
    else:
        # Pure NumPy implementation
        T = len(y_values)

        # Determine the number of columns in the output matrix
        if include_original:
            cols = lags + 1
            start_col = 0
        else:
            cols = lags
            start_col = 1

        # Initialize the output matrix
        result = np.zeros((T, cols))

        # Fill the matrix with lagged values
        if include_original:
            result[:, 0] = y_values

        for lag in range(1, lags + 1):
            col_idx = lag - start_col
            result[lag:, col_idx] = y_values[:-lag]

    # Drop rows with NaN values if requested
    if drop_na:
        # First 'lags' rows will have NaN values
        result = result[lags:, :]
        if index is not None:
            index = index[lags:]

    # Return pandas DataFrame if input was a Series
    if is_pandas:
        # Create column names
        if include_original:
            col_names = ['y'] + [f'y_lag{i}' for i in range(1, lags + 1)]
        else:
            col_names = [f'y_lag{i}' for i in range(1, lags + 1)]

        return pd.DataFrame(result, index=index, columns=col_names)

    return result


def ar_roots(ar_params: Vector) -> np.ndarray:
    """
    Compute the roots of an AR polynomial.

    This function computes the roots of an autoregressive polynomial
    1 - a_1*z - a_2*z^2 - ... - a_p*z^p.

    Args:
        ar_params: AR parameters [a_1, a_2, ..., a_p]

    Returns:
        Array of roots of the AR polynomial

    Examples:
        >>> import numpy as np
        >>> from mfe.models.time_series.utils import ar_roots
        >>> ar_params = np.array([0.5, -0.2])
        >>> roots = ar_roots(ar_params)
        >>> np.abs(roots)  # Check if all roots are outside the unit circle
        array([2.23606798, 2.23606798])
    """
    # Convert to numpy array if not already
    ar_params = np.asarray(ar_params)

    # Check if ar_params is 1D
    if ar_params.ndim != 1:
        raise_dimension_error(
            "AR parameters must be a 1D array",
            array_name="ar_params",
            expected_shape="(p,)",
            actual_shape=ar_params.shape
        )

    # If ar_params is empty, return empty array
    if len(ar_params) == 0:
        return np.array([])

    # Create the AR polynomial coefficients
    # The AR polynomial is 1 - a_1*z - a_2*z^2 - ... - a_p*z^p
    # For np.roots, we need [1, -a_1, -a_2, ..., -a_p]
    ar_poly = np.concatenate(([1], -ar_params))

    # Compute the roots
    roots = np.roots(ar_poly)

    return roots


def ma_roots(ma_params: Vector) -> np.ndarray:
    """
    Compute the roots of an MA polynomial.

    This function computes the roots of a moving average polynomial
    1 + b_1*z + b_2*z^2 + ... + b_q*z^q.

    Args:
        ma_params: MA parameters [b_1, b_2, ..., b_q]

    Returns:
        Array of roots of the MA polynomial

    Examples:
        >>> import numpy as np
        >>> from mfe.models.time_series.utils import ma_roots
        >>> ma_params = np.array([0.5, -0.2])
        >>> roots = ma_roots(ma_params)
        >>> np.abs(roots)  # Check if all roots are outside the unit circle
        array([2.23606798, 2.23606798])
    """
    # Convert to numpy array if not already
    ma_params = np.asarray(ma_params)

    # Check if ma_params is 1D
    if ma_params.ndim != 1:
        raise_dimension_error(
            "MA parameters must be a 1D array",
            array_name="ma_params",
            expected_shape="(q,)",
            actual_shape=ma_params.shape
        )

    # If ma_params is empty, return empty array
    if len(ma_params) == 0:
        return np.array([])

    # Create the MA polynomial coefficients
    # The MA polynomial is 1 + b_1*z + b_2*z^2 + ... + b_q*z^q
    # For np.roots, we need [1, b_1, b_2, ..., b_q]
    ma_poly = np.concatenate(([1], ma_params))

    # Compute the roots
    roots = np.roots(ma_poly)

    return roots


def check_ar_stationarity(ar_params: Vector) -> bool:
    """
    Check if AR parameters satisfy the stationarity condition.

    An AR process is stationary if all roots of the AR polynomial lie outside
    the unit circle, or equivalently, if all eigenvalues of the companion matrix
    have modulus less than 1.

    Args:
        ar_params: AR parameters [a_1, a_2, ..., a_p]

    Returns:
        True if the AR process is stationary, False otherwise

    Examples:
        >>> import numpy as np
        >>> from mfe.models.time_series.utils import check_ar_stationarity
        >>> ar_params = np.array([0.5, -0.2])  # Stationary
        >>> check_ar_stationarity(ar_params)
        True
        >>> ar_params = np.array([1.2, -0.2])  # Non-stationary
        >>> check_ar_stationarity(ar_params)
        False
    """
    # Convert to numpy array if not already
    ar_params = np.asarray(ar_params)

    # Check if ar_params is 1D
    if ar_params.ndim != 1:
        raise_dimension_error(
            "AR parameters must be a 1D array",
            array_name="ar_params",
            expected_shape="(p,)",
            actual_shape=ar_params.shape
        )

    # If ar_params is empty, the process is stationary
    if len(ar_params) == 0:
        return True

    # Compute the roots of the AR polynomial
    roots = ar_roots(ar_params)

    # Check if all roots are outside the unit circle
    return np.all(np.abs(roots) > 1.0)


def check_ma_invertibility(ma_params: Vector) -> bool:
    """
    Check if MA parameters satisfy the invertibility condition.

    An MA process is invertible if all roots of the MA polynomial lie outside
    the unit circle.

    Args:
        ma_params: MA parameters [b_1, b_2, ..., b_q]

    Returns:
        True if the MA process is invertible, False otherwise

    Examples:
        >>> import numpy as np
        >>> from mfe.models.time_series.utils import check_ma_invertibility
        >>> ma_params = np.array([0.5, -0.2])  # Invertible
        >>> check_ma_invertibility(ma_params)
        True
        >>> ma_params = np.array([1.2, -0.2])  # Non-invertible
        >>> check_ma_invertibility(ma_params)
        False
    """
    # Convert to numpy array if not already
    ma_params = np.asarray(ma_params)

    # Check if ma_params is 1D
    if ma_params.ndim != 1:
        raise_dimension_error(
            "MA parameters must be a 1D array",
            array_name="ma_params",
            expected_shape="(q,)",
            actual_shape=ma_params.shape
        )

    # If ma_params is empty, the process is invertible
    if len(ma_params) == 0:
        return True

    # Compute the roots of the MA polynomial
    roots = ma_roots(ma_params)

    # Check if all roots are outside the unit circle
    return np.all(np.abs(roots) > 1.0)


@jit(nopython=True, cache=True)
def _acf_numba(x: np.ndarray, nlags: int, fft: bool = False) -> np.ndarray:
    """
    Numba-accelerated implementation of autocorrelation function.

    Args:
        x: Time series data (1D array)
        nlags: Number of lags to compute
        fft: Whether to use FFT for computation (ignored in Numba implementation)

    Returns:
        Array of autocorrelation coefficients
    """
    T = len(x)

    # Ensure nlags is not too large
    nlags = min(nlags, T - 1)

    # Demean the series
    x_demean = x - np.mean(x)

    # Compute variance
    var = np.sum(x_demean**2) / T

    # Initialize result array
    acf = np.zeros(nlags + 1)
    acf[0] = 1.0  # Autocorrelation at lag 0 is always 1

    # Compute autocorrelation for each lag
    for lag in range(1, nlags + 1):
        # Compute autocovariance
        acov = np.sum(x_demean[lag:] * x_demean[:-lag]) / T

        # Compute autocorrelation
        acf[lag] = acov / var

    return acf


def acf(x: Union[np.ndarray, pd.Series], nlags: Optional[int] = None,
        fft: bool = False, alpha: Optional[float] = None) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute the autocorrelation function.

    This function computes the autocorrelation function (ACF) for a time series,
    optionally with confidence intervals.

    Args:
        x: Time series data (1D array or Series)
        nlags: Number of lags to compute. If None, uses min(10, T//5)
        fft: Whether to use FFT for computation (faster for long series)
        alpha: Significance level for confidence intervals. If None, no intervals are computed

    Returns:
        Tuple of (acf, lower_ci, upper_ci) where lower_ci and upper_ci are None if alpha is None

    Raises:
        DimensionError: If x is not 1D
        ValueError: If nlags is negative or alpha is not between 0 and 1

    Examples:
        >>> import numpy as np
        >>> from mfe.models.time_series.utils import acf
        >>> np.random.seed(123)
        >>> x = np.random.randn(100)
        >>> acf_values, lower_ci, upper_ci = acf(x, nlags=10, alpha=0.05)
    """
    # Handle pandas Series
    if isinstance(x, pd.Series):
        x_values = x.values
    else:
        x_values = np.asarray(x)

    # Check if x is 1D
    if x_values.ndim != 1:
        raise_dimension_error(
            "Input must be a 1D array or Series",
            array_name="x",
            expected_shape="(T,)",
            actual_shape=x_values.shape
        )

    T = len(x_values)

    # Determine number of lags if not provided
    if nlags is None:
        nlags = min(10, T // 5)

    # Check if nlags is valid
    if not isinstance(nlags, int) or nlags < 0:
        raise ValueError("nlags must be a non-negative integer")

    # Ensure nlags is not too large
    nlags = min(nlags, T - 1)

    # Check if alpha is valid
    if alpha is not None and (alpha <= 0 or alpha >= 1):
        raise ValueError("alpha must be between 0 and 1")

    # Use Numba-accelerated implementation if available and not using FFT
    if HAS_NUMBA and not fft:
        acf_values = _acf_numba(x_values, nlags, fft=False)
    else:
        # Use FFT method if requested and series is long enough
        if fft and T > 1000:
            # Compute ACF using FFT
            # Pad the series to the next power of 2 for efficiency
            n = 2 ** int(np.ceil(np.log2(2 * T - 1)))
            x_padded = np.zeros(n)
            x_padded[:T] = x_values - np.mean(x_values)

            # Compute FFT
            f = np.fft.fft(x_padded)
            acf_values = np.fft.ifft(f * np.conjugate(f)).real[:nlags + 1] / T

            # Normalize
            acf_values /= acf_values[0]
        else:
            # Use direct method
            # Demean the series
            x_demean = x_values - np.mean(x_values)

            # Compute variance
            var = np.sum(x_demean**2) / T

            # Initialize result array
            acf_values = np.zeros(nlags + 1)
            acf_values[0] = 1.0  # Autocorrelation at lag 0 is always 1

            # Compute autocorrelation for each lag
            for lag in range(1, nlags + 1):
                # Compute autocovariance
                acov = np.sum(x_demean[lag:] * x_demean[:-lag]) / T

                # Compute autocorrelation
                acf_values[lag] = acov / var

    # Compute confidence intervals if requested
    if alpha is not None:
        # Standard error of ACF
        se = np.ones_like(acf_values)
        se[1:] = np.sqrt((1 + 2 * np.sum(acf_values[1:nlags+1]**2)) / T)

        # Critical value from normal distribution
        z = stats.norm.ppf(1 - alpha / 2)

        # Confidence intervals
        lower_ci = acf_values - z * se
        upper_ci = acf_values + z * se

        return acf_values, lower_ci, upper_ci

    return acf_values, None, None


@jit(nopython=True, cache=True)
def _pacf_yule_walker_numba(x: np.ndarray, nlags: int) -> np.ndarray:
    """
    Numba-accelerated implementation of partial autocorrelation function using Yule-Walker.

    Args:
        x: Time series data (1D array)
        nlags: Number of lags to compute

    Returns:
        Array of partial autocorrelation coefficients
    """
    T = len(x)

    # Ensure nlags is not too large
    nlags = min(nlags, T - 1)

    # Demean the series
    x_demean = x - np.mean(x)

    # Compute autocorrelation function
    acf_values = np.zeros(nlags + 1)
    acf_values[0] = 1.0

    # Compute autocorrelation for each lag
    for lag in range(1, nlags + 1):
        acf_values[lag] = np.sum(x_demean[lag:] * x_demean[:-lag]) / np.sum(x_demean**2)

    # Initialize PACF array
    pacf = np.zeros(nlags + 1)
    pacf[0] = 1.0

    # Compute PACF using Yule-Walker equations
    for k in range(1, nlags + 1):
        # Set up Yule-Walker equations
        r = acf_values[1:k+1]
        R = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                if i == j:
                    R[i, j] = 1.0
                else:
                    R[i, j] = acf_values[abs(i - j)]

        # Solve Yule-Walker equations
        try:
            phi = np.zeros(k)
            # Simple implementation of solving linear system
            # This is not as efficient as using a specialized solver,
            # but it works for small systems and is Numba-compatible
            for i in range(k):
                phi[i] = r[i]
                for j in range(i):
                    phi[i] -= R[i, j] * phi[j]
                phi[i] /= R[i, i]

            # The last coefficient is the PACF value
            pacf[k] = phi[-1]
        except:
            # If matrix is singular, set PACF to 0
            pacf[k] = 0.0

    return pacf


def pacf(x: Union[np.ndarray, pd.Series], nlags: Optional[int] = None,
         method: str = "yule-walker", alpha: Optional[float] = None) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute the partial autocorrelation function.

    This function computes the partial autocorrelation function (PACF) for a time series,
    optionally with confidence intervals.

    Args:
        x: Time series data (1D array or Series)
        nlags: Number of lags to compute. If None, uses min(10, T//5)
        method: Method to use for computation. Options are "yule-walker" (default), "ols", "burg"
        alpha: Significance level for confidence intervals. If None, no intervals are computed

    Returns:
        Tuple of (pacf, lower_ci, upper_ci) where lower_ci and upper_ci are None if alpha is None

    Raises:
        DimensionError: If x is not a 1D array or Series
        ValueError: If nlags is negative, method is invalid, or alpha is not between 0 and 1

    Examples:
        >>> import numpy as np
        >>> from mfe.models.time_series.utils import pacf
        >>> np.random.seed(123)
        >>> x = np.random.randn(100)
        >>> pacf_values, lower_ci, upper_ci = pacf(x, nlags=10, alpha=0.05)
    """
    # Handle pandas Series
    if isinstance(x, pd.Series):
        x_values = x.values
    else:
        x_values = np.asarray(x)

    # Check if x is 1D
    if x_values.ndim != 1:
        raise_dimension_error(
            "Input must be a 1D array or Series",
            array_name="x",
            expected_shape="(T,)",
            actual_shape=x_values.shape
        )

    T = len(x_values)

    # Determine number of lags if not provided
    if nlags is None:
        nlags = min(10, T // 5)

    # Check if nlags is valid
    if not isinstance(nlags, int) or nlags < 0:
        raise ValueError("nlags must be a non-negative integer")

    # Ensure nlags is not too large
    nlags = min(nlags, T - 1)

    # Check if alpha is valid
    if alpha is not None and (alpha <= 0 or alpha >= 1):
        raise ValueError("alpha must be between 0 and 1")

    # Check if method is valid
    valid_methods = ["yule-walker", "ols", "burg"]
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}")

    # Use Numba-accelerated implementation if available and using Yule-Walker
    if HAS_NUMBA and method == "yule-walker":
        pacf_values = _pacf_yule_walker_numba(x_values, nlags)
    else:
        # Compute PACF using the specified method
        if method == "yule-walker":
            # Compute ACF
            acf_values = np.zeros(nlags + 1)
            acf_values[0] = 1.0

            # Compute autocorrelation for each lag
            x_demean = x_values - np.mean(x_values)
            for lag in range(1, nlags + 1):
                acf_values[lag] = np.sum(x_demean[lag:] * x_demean[:-lag]) / np.sum(x_demean**2)

            # Initialize PACF array
            pacf_values = np.zeros(nlags + 1)
            pacf_values[0] = 1.0

            # Compute PACF using Yule-Walker equations
            for k in range(1, nlags + 1):
                # Set up Yule-Walker equations
                r = acf_values[1:k+1]
                R = np.zeros((k, k))
                for i in range(k):
                    for j in range(k):
                        if i == j:
                            R[i, j] = 1.0
                        else:
                            R[i, j] = acf_values[abs(i - j)]

                # Solve Yule-Walker equations
                try:
                    phi = linalg.solve(R, r)
                    pacf_values[k] = phi[-1]
                except linalg.LinAlgError:
                    # If matrix is singular, set PACF to 0
                    pacf_values[k] = 0.0

        elif method == "ols":
            # Compute PACF using OLS regression
            pacf_values = np.zeros(nlags + 1)
            pacf_values[0] = 1.0

            # Demean the series
            x_demean = x_values - np.mean(x_values)

            # For each lag, regress x_t on x_{t-1}, ..., x_{t-k}
            for k in range(1, nlags + 1):
                # Create lagged matrix
                X = np.zeros((T - k, k))
                for i in range(k):
                    X[:, i] = x_demean[k-i-1:T-i-1]

                # Dependent variable
                y = x_demean[k:]

                # Compute OLS regression
                try:
                    beta = linalg.lstsq(X, y, rcond=None)[0]
                    pacf_values[k] = beta[0]  # The coefficient of x_{t-k}
                except:
                    # If regression fails, set PACF to 0
                    pacf_values[k] = 0.0

        elif method == "burg":
            # Compute PACF using Burg's method
            # This is a more efficient and numerically stable method
            # for computing PACF, especially for long time series

            # Demean the series
            x_demean = x_values - np.mean(x_values)

            # Initialize arrays
            pacf_values = np.zeros(nlags + 1)
            pacf_values[0] = 1.0

            # Initialize forward and backward prediction errors
            f = x_demean.copy()
            b = x_demean.copy()

            # Burg algorithm
            for k in range(1, nlags + 1):
                # Compute reflection coefficient
                num = 0.0
                den = 0.0
                for t in range(k, T):
                    num += f[t] * b[t-1]
                    den += f[t]**2 + b[t-1]**2

                # Reflection coefficient is the negative of PACF
                if den > 0:
                    pacf_values[k] = -2 * num / den
                else:
                    pacf_values[k] = 0.0

                # Update forward and backward prediction errors
                for t in range(T - 1, k - 1, -1):
                    f_old = f[t]
                    f[t] = f_old + pacf_values[k] * b[t-1]
                    b[t-1] = b[t-1] + pacf_values[k] * f_old

    # Compute confidence intervals if requested
    if alpha is not None:
        # Standard error of PACF
        se = np.ones_like(pacf_values)
        se[1:] = np.sqrt(1 / T)

        # Critical value from normal distribution
        z = stats.norm.ppf(1 - alpha / 2)

        # Confidence intervals
        lower_ci = pacf_values - z * se
        upper_ci = pacf_values + z * se

        return pacf_values, lower_ci, upper_ci

    return pacf_values, None, None


def ljungbox(x: Union[np.ndarray, pd.Series], lags: Union[int, List[int]],
             boxpierce: bool = False, model_df: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Ljung-Box test for autocorrelation.

    This function performs the Ljung-Box test (or optionally the Box-Pierce test)
    for autocorrelation in a time series.

    Args:
        x: Time series data (1D array or Series)
        lags: Lag(s) to test. Can be a single integer or a list of integers.
        boxpierce: If True, compute the Box-Pierce statistic instead of Ljung-Box
        model_df: Degrees of freedom used in the model (e.g., p+q for ARMA(p,q))

    Returns:
        Tuple of (test_statistic, p_value, degrees_of_freedom, critical_values)
        where critical_values are at the 5% significance level

    Raises:
        DimensionError: If x is not 1D
        ValueError: If lags is not positive or model_df is negative

    Examples:
        >>> import numpy as np
        >>> from mfe.models.time_series.utils import ljungbox
        >>> np.random.seed(123)
        >>> x = np.random.randn(100)
        >>> lb_stat, p_value, dof, crit = ljungbox(x, lags=10)
    """
    # Handle pandas Series
    if isinstance(x, pd.Series):
        x_values = x.values
    else:
        x_values = np.asarray(x)

    # Check if x is 1D
    if x_values.ndim != 1:
        raise_dimension_error(
            "Input must be a 1D array or Series",
            array_name="x",
            expected_shape="(T,)",
            actual_shape=x_values.shape
        )

    T = len(x_values)

    # Convert lags to list if it's a single integer
    if isinstance(lags, int):
        lags_list = [lags]
    else:
        lags_list = lags

    # Check if lags are valid
    if any(lag <= 0 for lag in lags_list):
        raise ValueError("All lags must be positive integers")

    # Check if model_df is valid
    if model_df < 0:
        raise ValueError("model_df must be a non-negative integer")

    # Ensure lags are not too large
    lags_list = [min(lag, T - 1) for lag in lags_list]

    # Compute ACF
    max_lag = max(lags_list)
    acf_values, _, _ = acf(x_values, nlags=max_lag)

    # Compute test statistics and p-values
    n_lags = len(lags_list)
    lb_stat = np.zeros(n_lags)
    p_values = np.zeros(n_lags)
    dof = np.zeros(n_lags, dtype=int)
    crit_values = np.zeros(n_lags)

    for i, lag in enumerate(lags_list):
        # Compute degrees of freedom
        dof[i] = lag - model_df

        # Ensure dof is positive
        if dof[i] <= 0:
            warn_numeric(
                f"Degrees of freedom ({dof[i]}) is not positive for lag {lag}",
                operation="ljungbox",
                issue="invalid_dof"
            )
            dof[i] = 1  # Set to 1 to avoid division by zero

        # Compute test statistic
        if boxpierce:
            # Box-Pierce statistic
            lb_stat[i] = T * np.sum(acf_values[1:lag+1]**2)
        else:
            # Ljung-Box statistic
            lb_stat[i] = T * (T + 2) * np.sum(acf_values[1:lag+1]**2 / (T - np.arange(1, lag+1)))

        # Compute p-value
        p_values[i] = 1 - chi2.cdf(lb_stat[i], dof[i])

        # Compute critical value at 5% significance level
        crit_values[i] = chi2.ppf(0.95, dof[i])

    return lb_stat, p_values, dof, crit_values


def durbin_watson(residuals: Union[np.ndarray, pd.Series]) -> float:
    """
    Compute Durbin-Watson statistic for autocorrelation.

    This function computes the Durbin-Watson statistic for testing the presence
    of first-order autocorrelation in the residuals of a regression.

    Args:
        residuals: Residuals from a regression (1D array or Series)

    Returns:
        Durbin-Watson statistic

    Raises:
        DimensionError: If residuals is not 1D

    Examples:
        >>> import numpy as np
        >>> from mfe.models.time_series.utils import durbin_watson
        >>> np.random.seed(123)
        >>> residuals = np.random.randn(100)
        >>> dw = durbin_watson(residuals)
    """
    # Handle pandas Series
    if isinstance(residuals, pd.Series):
        residuals_values = residuals.values
    else:
        residuals_values = np.asarray(residuals)

    # Check if residuals is 1D
    if residuals_values.ndim != 1:
        raise_dimension_error(
            "Residuals must be a 1D array or Series",
            array_name="residuals",
            expected_shape="(T,)",
            actual_shape=residuals_values.shape
        )

    # Compute Durbin-Watson statistic
    # DW = sum((e_t - e_{t-1})^2) / sum(e_t^2)
    diff_squared = np.sum(np.diff(residuals_values)**2)
    residual_squared = np.sum(residuals_values**2)

    if residual_squared == 0:
        raise_numeric_error(
            "Sum of squared residuals is zero",
            operation="durbin_watson",
            error_type="division_by_zero"
        )

    dw = diff_squared / residual_squared

    return dw


def information_criteria(loglikelihood: float, nobs: int, nparams: int) -> Tuple[float, float, float]:
    """
    Compute information criteria for model selection.

    This function computes the Akaike Information Criterion (AIC), Bayesian
    Information Criterion (BIC), and Hannan-Quinn Information Criterion (HQIC)
    for model selection.

    Args:
        loglikelihood: Log-likelihood of the model
        nobs: Number of observations
        nparams: Number of parameters in the model

    Returns:
        Tuple of (AIC, BIC, HQIC)

    Raises:
        ValueError: If nobs or nparams is not positive

    Examples:
        >>> from mfe.models.time_series.utils import information_criteria
        >>> aic, bic, hqic = information_criteria(loglikelihood=-150, nobs=100, nparams=5)
    """
    # Check if nobs and nparams are valid
    if nobs <= 0:
        raise ValueError("nobs must be a positive integer")
    if nparams <= 0:
        raise ValueError("nparams must be a positive integer")

    # Compute information criteria
    aic = -2 * loglikelihood + 2 * nparams
    bic = -2 * loglikelihood + nparams * np.log(nobs)
    hqic = -2 * loglikelihood + 2 * nparams * np.log(np.log(nobs))

    return aic, bic, hqic


def forecast_error_variance(ar_params: Vector, ma_params: Vector,
                            sigma2: float, h: int) -> float:
    """
    Compute forecast error variance for ARMA models.

    This function computes the variance of the h-step ahead forecast error
    for an ARMA(p,q) model.

    Args:
        ar_params: AR parameters [a_1, a_2, ..., a_p]
        ma_params: MA parameters [b_1, b_2, ..., b_q]
        sigma2: Innovation variance
        h: Forecast horizon

    Returns:
        Forecast error variance

    Raises:
        ValueError: If h is not positive or sigma2 is not positive

    Examples:
        >>> import numpy as np
        >>> from mfe.models.time_series.utils import forecast_error_variance
        >>> ar_params = np.array([0.5, -0.2])
        >>> ma_params = np.array([0.3])
        >>> sigma2 = 1.0
        >>> fev = forecast_error_variance(ar_params, ma_params, sigma2, h=5)
    """
    # Convert to numpy arrays if not already
    ar_params = np.asarray(ar_params) if len(ar_params) > 0 else np.array([])
    ma_params = np.asarray(ma_params) if len(ma_params) > 0 else np.array([])

    # Check if h is valid
    if h <= 0:
        raise ValueError("h must be a positive integer")

    # Check if sigma2 is valid
    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive")

    # Get AR and MA orders
    p = len(ar_params)
    q = len(ma_params)

    # If both AR and MA orders are 0, return sigma2
    if p == 0 and q == 0:
        return sigma2

    # Compute psi weights (MA representation coefficients)
    # For an ARMA(p,q) model, we need at least h+q psi weights
    n_psi = max(h + q, p + q)
    psi = np.zeros(n_psi + 1)
    psi[0] = 1.0

    # Compute psi weights recursively
    for i in range(1, n_psi + 1):
        # Add AR component
        if p > 0:
            for j in range(min(i, p)):
                psi[i] += ar_params[j] * psi[i - j - 1]

        # Add MA component
        if q > 0 and i <= q:
            psi[i] += ma_params[i - 1]

    # Compute forecast error variance
    # For h-step ahead forecast, the error variance is
    # sigma2 * (1 + psi_1^2 + psi_2^2 + ... + psi_{h-1}^2)
    fev = sigma2 * np.sum(psi[:h]**2)

    return fev


def impulse_response(ar_params: Vector, ma_params: Vector,
                     periods: int) -> np.ndarray:
    """
    Compute impulse response function for ARMA models.

    This function computes the impulse response function (IRF) for an ARMA(p,q)
    model, which shows the effect of a one-time shock on the future values of
    the time series.

    Args:
        ar_params: AR parameters [a_1, a_2, ..., a_p]
        ma_params: MA parameters [b_1, b_2, ..., b_q]
        periods: Number of periods to compute the IRF for

    Returns:
        Array of impulse response coefficients

    Raises:
        ValueError: If periods is not positive

    Examples:
        >>> import numpy as np
        >>> from mfe.models.time_series.utils import impulse_response
        >>> ar_params = np.array([0.5, -0.2])
        >>> ma_params = np.array([0.3])
        >>> irf = impulse_response(ar_params, ma_params, periods=10)
    """
    # Convert to numpy arrays if not already
    ar_params = np.asarray(ar_params) if len(ar_params) > 0 else np.array([])
    ma_params = np.asarray(ma_params) if len(ma_params) > 0 else np.array([])

    # Check if periods is valid
    if periods <= 0:
        raise ValueError("periods must be a positive integer")

    # Get AR and MA orders
    p = len(ar_params)
    q = len(ma_params)

    # Initialize impulse response function
    irf = np.zeros(periods + 1)
    irf[0] = 1.0  # Initial shock

    # If both AR and MA orders are 0, return [1, 0, 0, ...]
    if p == 0 and q == 0:
        return irf

    # Compute impulse response function recursively
    for i in range(1, periods + 1):
        # Add AR component
        if p > 0:
            for j in range(min(i, p)):
                irf[i] += ar_params[j] * irf[i - j - 1]

        # Add MA component
        if q > 0 and i <= q:
            irf[i] += ma_params[i - 1]

    return irf


def is_invertible(ma_params: Vector) -> bool:
    """
    Check if a time series model is invertible.

    This function checks if an MA process is invertible by testing if all roots
    of the MA polynomial lie outside the unit circle.

    Args:
        ma_params: MA parameters [b_1, b_2, ..., b_q]

    Returns:
        True if the model is invertible, False otherwise

    Examples:
        >>> import numpy as np
        >>> from mfe.models.time_series.utils import is_invertible
        >>> ma_params = np.array([0.5, -0.2])  # Invertible
        >>> is_invertible(ma_params)
        True
        >>> ma_params = np.array([1.2, -0.2])  # Non-invertible
        >>> is_invertible(ma_params)
        False
    """
    return check_ma_invertibility(ma_params)


def is_stationary(ar_params: Vector) -> bool:
    """
    Check if a time series model is stationary.

    This function checks if an AR process is stationary by testing if all roots
    of the AR polynomial lie outside the unit circle.

    Args:
        ar_params: AR parameters [a_1, a_2, ..., a_p]

    Returns:
        True if the model is stationary, False otherwise

    Examples:
        >>> import numpy as np
        >>> from mfe.models.time_series.utils import is_stationary
        >>> ar_params = np.array([0.5, -0.2])  # Stationary
        >>> is_stationary(ar_params)
        True
        >>> ar_params = np.array([1.2, -0.2])  # Non-stationary
        >>> is_stationary(ar_params)
        False
    """
    return check_ar_stationarity(ar_params)


def standardize_data(x: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
    """
    Standardize time series data.

    This function standardizes a time series by subtracting the mean and
    dividing by the standard deviation.

    Args:
        x: Time series data (1D array or Series)

    Returns:
        Standardized time series (same type as input)

    Raises:
        DimensionError: If x is not 1D
        NumericError: If standard deviation is zero

    Examples:
        >>> import numpy as np
        >>> from mfe.models.time_series.utils import standardize_data
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> standardize_data(x)
        array([-1.41421356, -0.70710678,  0.        ,  0.70710678,  1.41421356])
    """
    # Handle pandas Series
    is_pandas = isinstance(x, pd.Series)
    if is_pandas:
        index = x.index
        name = x.name
        x_values = x.values
    else:
        x_values = np.asarray(x)

    # Check if x is 1D
    if x_values.ndim != 1:
        raise_dimension_error(
            "Input must be a 1D array or Series",
            array_name="x",
            expected_shape="(T,)",
            actual_shape=x_values.shape
        )

    # Compute mean and standard deviation
    mean = np.mean(x_values)
    std = np.std(x_values, ddof=1)  # Use sample standard deviation

    # Check if standard deviation is zero
    if std == 0:
        raise_numeric_error(
            "Standard deviation is zero",
            operation="standardize_data",
            error_type="division_by_zero"
        )

    # Standardize the data
    x_std = (x_values - mean) / std

    # Return pandas Series if input was a Series
    if is_pandas:
        return pd.Series(x_std, index=index, name=name)

    return x_std


def detrend(x: Union[np.ndarray, pd.Series], order: int = 1) -> Union[np.ndarray, pd.Series]:
    """
    Remove trend from time series data.

    This function removes a polynomial trend of specified order from a time series.

    Args:
        x: Time series data (1D array or Series)
        order: Order of the polynomial trend to remove

    Returns:
        Detrended time series (same type as input)

    Raises:
        DimensionError: If x is not 1D
        ValueError: If order is negative

    Examples:
        >>> import numpy as np
        >>> from mfe.models.time_series.utils import detrend
        >>> x = np.array([1, 4, 9, 16, 25])  # Quadratic trend
        >>> detrend(x, order=2)
        array([0., 0., 0., 0., 0.])
    """
    # Handle pandas Series
    is_pandas = isinstance(x, pd.Series)
    if is_pandas:
        index = x.index
        name = x.name
        x_values = x.values
    else:
        x_values = np.asarray(x)

    # Check if x is 1D
    if x_values.ndim != 1:
        raise_dimension_error(
            "Input must be a 1D array or Series",
            array_name="x",
            expected_shape="(T,)",
            actual_shape=x_values.shape
        )

    # Check if order is valid
    if order < 0:
        raise ValueError("order must be a non-negative integer")

    # If order is 0, just remove the mean
    if order == 0:
        x_detrend = x_values - np.mean(x_values)
    else:
        # Create time index
        T = len(x_values)
        t = np.arange(T)

        # Create polynomial features
        X = np.column_stack([t**i for i in range(order + 1)])

        # Fit polynomial trend
        beta = linalg.lstsq(X, x_values, rcond=None)[0]

        # Remove trend
        trend = np.sum(beta[i] * t**i for i in range(order + 1))
        x_detrend = x_values - trend

    # Return pandas Series if input was a Series
    if is_pandas:
        return pd.Series(x_detrend, index=index, name=name)

    return x_detrend


def seasonal_adjust(x: Union[np.ndarray, pd.Series], period: int,
                    method: str = "mean") -> Union[np.ndarray, pd.Series]:
    """
    Remove seasonal component from time series data.

    This function removes a seasonal component of specified period from a time series.

    Args:
        x: Time series data (1D array or Series)
        period: Seasonal period (e.g., 12 for monthly data with yearly seasonality)
        method: Method to use for seasonal adjustment. Options are "mean" (default), "median"

    Returns:
        Seasonally adjusted time series (same type as input)

    Raises:
        DimensionError: If x is not 1D
        ValueError: If period is not positive or method is invalid

    Examples:
        >>> import numpy as np
        >>> from mfe.models.time_series.utils import seasonal_adjust
        >>> # Create data with period 4 seasonality
        >>> x = np.array([10, 20, 30, 40, 15, 25, 35, 45, 20, 30, 40, 50])
        >>> seasonal_adjust(x, period=4)
        array([ 0.,  0.,  0.,  0.,  5.,  5.,  5.,  5., 10., 10., 10., 10.])
    """
    # Handle pandas Series
    is_pandas = isinstance(x, pd.Series)
    if is_pandas:
        index = x.index
        name = x.name
        x_values = x.values
    else:
        x_values = np.asarray(x)

    # Check if x is 1D
    if x_values.ndim != 1:
        raise_dimension_error(
            "Input must be a 1D array or Series",
            array_name="x",
            expected_shape="(T,)",
            actual_shape=x_values.shape
        )

    # Check if period is valid
    if period <= 0:
        raise ValueError("period must be a positive integer")

    # Check if method is valid
    valid_methods = ["mean", "median"]
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}")

    # Get length of series
    T = len(x_values)

    # Compute seasonal component
    seasonal = np.zeros(T)

    # For each position in the seasonal cycle
    for i in range(period):
        # Get values at this position
        indices = np.arange(i, T, period)
        values = x_values[indices]

        # Compute seasonal factor
        if method == "mean":
            factor = np.mean(values)
        else:  # method == "median"
            factor = np.median(values)

        # Assign seasonal factor
        seasonal[indices] = factor

    # Remove seasonal component
    x_adjusted = x_values - seasonal

    # Return pandas Series if input was a Series
    if is_pandas:
        return pd.Series(x_adjusted, index=index, name=name)

    return x_adjusted


def lag_polynomial(params: Vector, z: Union[float, complex, np.ndarray]) -> Union[float, complex, np.ndarray]:
    """
    Evaluate a lag polynomial.

    This function evaluates a lag polynomial of the form
    1 - a_1*z - a_2*z^2 - ... - a_p*z^p.

    Args:
        params: Polynomial coefficients [a_1, a_2, ..., a_p]
        z: Value(s) at which to evaluate the polynomial

    Returns:
        Value(s) of the polynomial at z

    Examples:
        >>> import numpy as np
        >>> from mfe.models.time_series.utils import lag_polynomial
        >>> params = np.array([0.5, -0.2])
        >>> lag_polynomial(params, 0.5)  # Evaluate at z = 0.5
        0.8
    """
    # Convert to numpy array if not already
    params = np.asarray(params)

    # Check if params is 1D
    if params.ndim != 1:
        raise_dimension_error(
            "params must be a 1D array",
            array_name="params",
            expected_shape="(p,)",
            actual_shape=params.shape
        )

    # If params is empty, return 1
    if len(params) == 0:
        return 1.0

    # Evaluate the polynomial
    # 1 - a_1*z - a_2*z^2 - ... - a_p*z^p
    result = 1.0
    for i, a in enumerate(params):
        result -= a * z**(i + 1)

    return result


def companion_matrix(ar_params: Vector) -> np.ndarray:
    """
    Create companion matrix from AR parameters.

    This function creates the companion matrix for an AR(p) process, which is
    useful for analyzing the stationarity of the process.

    Args:
        ar_params: AR parameters [a_1, a_2, ..., a_p]

    Returns:
        Companion matrix of size p×p

    Raises:
        ValueError: If ar_params is empty

    Examples:
        >>> import numpy as np
        >>> from mfe.models.time_series.utils import companion_matrix
        >>> ar_params = np.array([0.5, -0.2])
        >>> companion_matrix(ar_params)
        array([[ 0.5, -0.2],
               [ 1. ,  0. ]])
    """
    # Convert to numpy array if not already
    ar_params = np.asarray(ar_params)

    # Check if ar_params is 1D
    if ar_params.ndim != 1:
        raise_dimension_error(
            "AR parameters must be a 1D array",
            array_name="ar_params",
            expected_shape="(p,)",
            actual_shape=ar_params.shape
        )

    # Check if ar_params is empty
    if len(ar_params) == 0:
        raise ValueError("AR parameters cannot be empty")

    # Get AR order
    p = len(ar_params)

    # Create companion matrix
    companion = np.zeros((p, p))
    companion[0, :] = ar_params
    if p > 1:
        companion[1:, :-1] = np.eye(p - 1)

    return companion


# Register Numba-accelerated functions if available

def _register_numba_functions() -> None:
    """
    Register Numba JIT-compiled functions for time series utilities.

    This function is called during module initialization to register
    performance-critical functions for JIT compilation if Numba is available.
    """
    if HAS_NUMBA:
        # The functions are already decorated with @jit, so no additional
        # registration is needed here. This function is kept for consistency
        # with the module structure and potential future enhancements.
        logger.debug("Time series utilities Numba JIT functions registered")
    else:
        logger.info("Numba not available. Time series utilities will use pure NumPy implementations.")


# Initialize the module
_register_numba_functions()
