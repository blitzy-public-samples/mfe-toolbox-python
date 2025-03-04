'''
Data Transformation Module

This module provides essential data preprocessing functions for financial time series,
including demeaning, standardization, and multivariate standardization. These functions
are critical for preparing data before model estimation and analysis.

The module implements optimized versions of common data transformations using NumPy's
efficient array operations and provides seamless integration with Pandas objects.
All functions include comprehensive type hints and input validation to ensure
reliability and proper error handling.

Functions:
    standardize: Standardize a time series to have zero mean and unit variance
    mvstandardize: Standardize multiple time series to have zero mean and unit variance
    demean: Remove the mean from a time series
    lag_matrix: Create a matrix of lagged values from a time series
    lag_series: Create lagged versions of a time series
    rolling_window: Create rolling windows of a time series
    rolling_mean: Compute rolling mean of a time series
    rolling_variance: Compute rolling variance of a time series
    rolling_skewness: Compute rolling skewness of a time series
    rolling_kurtosis: Compute rolling kurtosis of a time series
'''

import logging
from typing import List, Optional, Tuple, Union, cast, overload

import numpy as np
import pandas as pd
from scipy import stats

from mfe.core.types import (
    TimeSeriesData, TimeSeriesDataFrame, Vector, Matrix
)
from mfe.core.exceptions import (
    DimensionError, DataError, raise_dimension_error, raise_data_error
)
from mfe.core.validation import (
    validate_time_series, validate_vector, validate_matrix_shape,
    validate_input_time_series, validate_input_bounds
)

# Set up module-level logger
logger = logging.getLogger("mfe.utils.data_transformations")

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
    logger.debug("Numba available for data transformation acceleration")
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    HAS_NUMBA = False
    logger.info("Numba not available. Data transformations will use pure NumPy implementations.")


@validate_input_time_series(0)
def standardize(
    data: TimeSeriesData,
    ddof: int = 1,
    inplace: bool = False,
    return_params: bool = False
) -> Union[TimeSeriesData, Tuple[TimeSeriesData, float, float]]:
    """
    Standardize a time series to have zero mean and unit variance.
    
    This function standardizes a time series by subtracting the mean and
    dividing by the standard deviation. The result has zero mean and unit variance.
    
    Args:
        data: Time series data to standardize (NumPy array, Pandas Series, or DataFrame)
        ddof: Delta degrees of freedom for standard deviation calculation
        inplace: If True and data is a Pandas object, modify it in place
        return_params: If True, return the mean and standard deviation along with
                      the standardized data
        
    Returns:
        If return_params is False:
            Standardized time series with the same type as the input
        If return_params is True:
            Tuple of (standardized_data, mean, std)
        
    Raises:
        DataError: If the data has zero variance
        
    Examples:
        >>> import numpy as np
        >>> from mfe.utils.data_transformations import standardize
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> standardized_x = standardize(x)
        >>> np.allclose(standardized_x.mean(), 0)
        True
        >>> np.allclose(standardized_x.std(ddof=1), 1)
        True
        
        >>> import pandas as pd
        >>> s = pd.Series([1, 2, 3, 4, 5], index=pd.date_range('2020-01-01', periods=5))
        >>> standardized_s, mean_s, std_s = standardize(s, return_params=True)
        >>> mean_s
        3.0
        >>> std_s
        1.5811388300841898
    """
    # Handle Pandas Series
    if isinstance(data, pd.Series):
        if inplace and not return_params:
            mean_val = data.mean()
            std_val = data.std(ddof=ddof)
            
            if std_val <= 0:
                raise_data_error(
                    "Cannot standardize data with zero or negative standard deviation",
                    data_name="data",
                    issue="zero_variance"
                )
            
            data = (data - mean_val) / std_val
            return data if not return_params else (data, mean_val, std_val)
        else:
            mean_val = data.mean()
            std_val = data.std(ddof=ddof)
            
            if std_val <= 0:
                raise_data_error(
                    "Cannot standardize data with zero or negative standard deviation",
                    data_name="data",
                    issue="zero_variance"
                )
            
            result = (data - mean_val) / std_val
            return result if not return_params else (result, mean_val, std_val)
    
    # Handle Pandas DataFrame (standardize each column)
    elif isinstance(data, pd.DataFrame):
        if inplace and not return_params:
            mean_vals = data.mean()
            std_vals = data.std(ddof=ddof)
            
            if (std_vals <= 0).any():
                zero_cols = std_vals.index[std_vals <= 0].tolist()
                raise_data_error(
                    f"Cannot standardize columns with zero or negative standard deviation: {zero_cols}",
                    data_name="data",
                    issue="zero_variance"
                )
            
            for col in data.columns:
                data[col] = (data[col] - mean_vals[col]) / std_vals[col]
            
            return data if not return_params else (data, mean_vals, std_vals)
        else:
            mean_vals = data.mean()
            std_vals = data.std(ddof=ddof)
            
            if (std_vals <= 0).any():
                zero_cols = std_vals.index[std_vals <= 0].tolist()
                raise_data_error(
                    f"Cannot standardize columns with zero or negative standard deviation: {zero_cols}",
                    data_name="data",
                    issue="zero_variance"
                )
            
            result = data.copy()
            for col in result.columns:
                result[col] = (result[col] - mean_vals[col]) / std_vals[col]
            
            return result if not return_params else (result, mean_vals, std_vals)
    
    # Handle NumPy array
    else:
        # Convert to numpy array if not already
        data_array = np.asarray(data)
        
        # Handle 1D arrays
        if data_array.ndim == 1:
            mean_val = np.mean(data_array)
            std_val = np.std(data_array, ddof=ddof)
            
            if std_val <= 0:
                raise_data_error(
                    "Cannot standardize data with zero or negative standard deviation",
                    data_name="data",
                    issue="zero_variance"
                )
            
            result = (data_array - mean_val) / std_val
            return result if not return_params else (result, mean_val, std_val)
        
        # Handle 2D arrays (standardize each column)
        elif data_array.ndim == 2:
            mean_vals = np.mean(data_array, axis=0)
            std_vals = np.std(data_array, axis=0, ddof=ddof)
            
            if np.any(std_vals <= 0):
                zero_cols = np.where(std_vals <= 0)[0].tolist()
                raise_data_error(
                    f"Cannot standardize columns with zero or negative standard deviation: {zero_cols}",
                    data_name="data",
                    issue="zero_variance"
                )
            
            result = (data_array - mean_vals) / std_vals
            return result if not return_params else (result, mean_vals, std_vals)
        
        # Handle higher dimensional arrays
        else:
            raise_dimension_error(
                "Input data must be 1D or 2D",
                array_name="data",
                expected_shape="(n,) or (n, p)",
                actual_shape=data_array.shape
            )


@validate_input_time_series(0)

def mvstandardize(
    data: TimeSeriesDataFrame,
    ddof: int = 1,
    inplace: bool = False,
    return_params: bool = False
) -> Union[TimeSeriesDataFrame, Tuple[TimeSeriesDataFrame, Vector, Matrix]]:
    """
    Standardize multiple time series to have zero mean and unit variance.
    
    This function standardizes multiple time series by subtracting the mean and
    dividing by the standard deviation for each series. The result has zero mean
    and unit variance for each series.
    
    Args:
        data: Multivariate time series data to standardize (NumPy array or Pandas DataFrame)
        ddof: Delta degrees of freedom for standard deviation calculation
        inplace: If True and data is a Pandas DataFrame, modify it in place
        return_params: If True, return the means and standard deviations along with
                      the standardized data
        
    Returns:
        If return_params is False:
            Standardized time series with the same type as the input
        If return_params is True:
            Tuple of (standardized_data, means, stds)
        
    Raises:
        DimensionError: If the input is not 2D
        DataError: If any series has zero variance
        
    Examples:
        >>> import numpy as np
        >>> from mfe.utils.data_transformations import mvstandardize
        >>> X = np.array([[1, 4], [2, 5], [3, 6], [4, 7], [5, 8]])
        >>> standardized_X = mvstandardize(X)
        >>> np.allclose(standardized_X.mean(axis=0), [0, 0])
        True
        >>> np.allclose(standardized_X.std(axis=0, ddof=1), [1, 1])
        True
        
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'A': [1, 2, 3, 4, 5],
        ...     'B': [4, 5, 6, 7, 8]
        ... }, index=pd.date_range('2020-01-01', periods=5))
        >>> standardized_df, means, stds = mvstandardize(df, return_params=True)
        >>> means
        A    3.0
        B    6.0
        dtype: float64
        >>> stds
        A    1.581139
        B    1.581139
        dtype: float64
    """
    # Ensure data is 2D
    if isinstance(data, np.ndarray) and data.ndim != 2:
        raise_dimension_error(
            "Input data must be 2D",
            array_name="data",
            expected_shape="(n, p)",
            actual_shape=data.shape
        )
    
    # Handle Pandas DataFrame
    if isinstance(data, pd.DataFrame):
        if inplace and not return_params:
            mean_vals = data.mean()
            std_vals = data.std(ddof=ddof)
            
            if (std_vals <= 0).any():
                zero_cols = std_vals.index[std_vals <= 0].tolist()
                raise_data_error(
                    f"Cannot standardize columns with zero or negative standard deviation: {zero_cols}",
                    data_name="data",
                    issue="zero_variance"
                )
            
            for col in data.columns:
                data[col] = (data[col] - mean_vals[col]) / std_vals[col]
            
            return data if not return_params else (data, mean_vals, std_vals)
        else:
            mean_vals = data.mean()
            std_vals = data.std(ddof=ddof)
            
            if (std_vals <= 0).any():
                zero_cols = std_vals.index[std_vals <= 0].tolist()
                raise_data_error(
                    f"Cannot standardize columns with zero or negative standard deviation: {zero_cols}",
                    data_name="data",
                    issue="zero_variance"
                )
            
            result = data.copy()
            for col in result.columns:
                result[col] = (result[col] - mean_vals[col]) / std_vals[col]
            
            return result if not return_params else (result, mean_vals, std_vals)
    
    # Handle NumPy array
    else:
        # Convert to numpy array if not already
        data_array = np.asarray(data)
        
        mean_vals = np.mean(data_array, axis=0)
        std_vals = np.std(data_array, axis=0, ddof=ddof)
        
        if np.any(std_vals <= 0):
            zero_cols = np.where(std_vals <= 0)[0].tolist()
            raise_data_error(
                f"Cannot standardize columns with zero or negative standard deviation: {zero_cols}",
                data_name="data",
                issue="zero_variance"
            )
        
        result = (data_array - mean_vals) / std_vals
        return result if not return_params else (result, mean_vals, std_vals)


@validate_input_time_series(0)

def demean(
    data: TimeSeriesData,
    inplace: bool = False,
    return_mean: bool = False
) -> Union[TimeSeriesData, Tuple[TimeSeriesData, Union[float, Vector]]]:
    """
    Remove the mean from a time series.
    
    This function subtracts the mean from a time series, resulting in a
    zero-mean series.
    
    Args:
        data: Time series data to demean (NumPy array, Pandas Series, or DataFrame)
        inplace: If True and data is a Pandas object, modify it in place
        return_mean: If True, return the mean along with the demeaned data
        
    Returns:
        If return_mean is False:
            Demeaned time series with the same type as the input
        If return_mean is True:
            Tuple of (demeaned_data, mean)
        
    Examples:
        >>> import numpy as np
        >>> from mfe.utils.data_transformations import demean
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> demeaned_x = demean(x)
        >>> np.allclose(demeaned_x.mean(), 0)
        True
        >>> np.allclose(demeaned_x, [-2, -1, 0, 1, 2])
        True
        
        >>> import pandas as pd
        >>> s = pd.Series([1, 2, 3, 4, 5], index=pd.date_range('2020-01-01', periods=5))
        >>> demeaned_s, mean_s = demean(s, return_mean=True)
        >>> mean_s
        3.0
    """
    # Handle Pandas Series
    if isinstance(data, pd.Series):
        mean_val = data.mean()
        
        if inplace:
            data = data - mean_val
            return data if not return_mean else (data, mean_val)
        else:
            result = data - mean_val
            return result if not return_mean else (result, mean_val)
    
    # Handle Pandas DataFrame (demean each column)
    elif isinstance(data, pd.DataFrame):
        mean_vals = data.mean()
        
        if inplace:
            for col in data.columns:
                data[col] = data[col] - mean_vals[col]
            
            return data if not return_mean else (data, mean_vals)
        else:
            result = data.copy()
            for col in result.columns:
                result[col] = result[col] - mean_vals[col]
            
            return result if not return_mean else (result, mean_vals)
    
    # Handle NumPy array
    else:
        # Convert to numpy array if not already
        data_array = np.asarray(data)
        
        # Handle 1D arrays
        if data_array.ndim == 1:
            mean_val = np.mean(data_array)
            result = data_array - mean_val
            return result if not return_mean else (result, mean_val)
        
        # Handle 2D arrays (demean each column)
        elif data_array.ndim == 2:
            mean_vals = np.mean(data_array, axis=0)
            result = data_array - mean_vals
            return result if not return_mean else (result, mean_vals)
        
        # Handle higher dimensional arrays
        else:
            raise_dimension_error(
                "Input data must be 1D or 2D",
                array_name="data",
                expected_shape="(n,) or (n, p)",
                actual_shape=data_array.shape
            )


@jit(nopython=True, cache=True)

def _lag_matrix_numba(data: np.ndarray, lags: int, include_original: bool = True) -> np.ndarray:
    """
    Numba-accelerated implementation of lag_matrix.
    
    Args:
        data: 1D array of time series data
        lags: Number of lags to include
        include_original: Whether to include the original series
        
    Returns:
        2D array with lagged series as columns
    """
    n = len(data)
    
    # Determine number of columns in result
    if include_original:
        n_cols = lags + 1
        start_col = 0
    else:
        n_cols = lags
        start_col = 1
    
    # Initialize result matrix
    result = np.zeros((n, n_cols))
    
    # Fill the matrix
    if include_original:
        result[:, 0] = data
    
    for lag in range(1, lags + 1):
        col_idx = lag if include_original else lag - 1
        result[lag:, col_idx] = data[:-lag]
    
    return result


@validate_input_time_series(0)
@validate_input_bounds(1, lower_bound=1, param_name="lags")

def lag_matrix(
    data: TimeSeriesData,
    lags: int,
    include_original: bool = True,
    fill_value: Optional[float] = None
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Create a matrix of lagged values from a time series.
    
    This function creates a matrix where each column is a lagged version of the
    input time series. The first column can be the original series, followed by
    columns with increasing lags.
    
    Args:
        data: Time series data (NumPy array or Pandas Series)
        lags: Number of lags to include
        include_original: Whether to include the original series as the first column
        fill_value: Value to use for missing values due to lagging (None for NaN)
        
    Returns:
        Matrix with lagged series as columns, with the same type as the input
        
    Raises:
        DimensionError: If the input is not 1D
        ValueError: If lags is less than 1
        
    Examples:
        >>> import numpy as np
        >>> from mfe.utils.data_transformations import lag_matrix
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> lag_matrix(x, lags=2)
        array([[1., 0., 0.],
               [2., 1., 0.],
               [3., 2., 1.],
               [4., 3., 2.],
               [5., 4., 3.]])
        >>> lag_matrix(x, lags=2, include_original=False)
        array([[0., 0.],
               [1., 0.],
               [2., 1.],
               [3., 2.],
               [4., 3.]])
        
        >>> import pandas as pd
        >>> s = pd.Series([1, 2, 3, 4, 5], index=pd.date_range('2020-01-01', periods=5))
        >>> lag_matrix(s, lags=2, fill_value=0)
        <BLANKLINE>
                   0  1  2
        2020-01-01  1  0  0
        2020-01-02  2  1  0
        2020-01-03  3  2  1
        2020-01-04  4  3  2
        2020-01-05  5  4  3
        """
    # Ensure data is 1D
    if isinstance(data, np.ndarray) and data.ndim != 1:
        raise_dimension_error(
            "Input data must be 1D",
            array_name="data",
            expected_shape="(n,)",
            actual_shape=data.shape
        )
    
    # Handle Pandas Series
    if isinstance(data, pd.Series):
        # Convert to numpy array for processing
        data_array = data.values
        
        # Use Numba-accelerated implementation if available
        if HAS_NUMBA:
            result_array = _lag_matrix_numba(data_array, lags, include_original)
        else:
            # Pure NumPy implementation
            n = len(data_array)
            
            # Determine number of columns in result
            if include_original:
                n_cols = lags + 1
                start_col = 0
            else:
                n_cols = lags
                start_col = 1
            
            # Initialize result matrix
            result_array = np.zeros((n, n_cols))
            
            # Fill the matrix
            if include_original:
                result_array[:, 0] = data_array
            
            for lag in range(1, lags + 1):
                col_idx = lag if include_original else lag - 1
                result_array[lag:, col_idx] = data_array[:-lag]
        
        # Replace zeros with fill_value if specified
        if fill_value is not None:
            result_array[result_array == 0] = fill_value
        
        # Convert back to DataFrame with original index
        col_names = list(range(result_array.shape[1]))
        result_df = pd.DataFrame(result_array, index=data.index, columns=col_names)
        
        return result_df
    
    # Handle NumPy array
    else:
        # Convert to numpy array if not already
        data_array = np.asarray(data)
        
        # Use Numba-accelerated implementation if available
        if HAS_NUMBA:
            result = _lag_matrix_numba(data_array, lags, include_original)
        else:
            # Pure NumPy implementation
            n = len(data_array)
            
            # Determine number of columns in result
            if include_original:
                n_cols = lags + 1
                start_col = 0
            else:
                n_cols = lags
                start_col = 1
            
            # Initialize result matrix
            result = np.zeros((n, n_cols))
            
            # Fill the matrix
            if include_original:
                result[:, 0] = data_array
            
            for lag in range(1, lags + 1):
                col_idx = lag if include_original else lag - 1
                result[lag:, col_idx] = data_array[:-lag]
        
        # Replace zeros with fill_value if specified
        if fill_value is not None:
            result[result == 0] = fill_value
        
        return result


@validate_input_time_series(0)
@validate_input_bounds(1, lower_bound=1, param_name="lags")

def lag_series(
    data: TimeSeriesData,
    lags: Union[int, List[int]],
    include_original: bool = True,
    fill_value: Optional[float] = None
) -> Union[List[np.ndarray], List[pd.Series], pd.DataFrame]:
    """
    Create lagged versions of a time series.
    
    This function creates lagged versions of the input time series, either as
    a list of series or as a DataFrame where each column is a lagged series.
    
    Args:
        data: Time series data (NumPy array or Pandas Series)
        lags: Number of lags to include or list of specific lags
        include_original: Whether to include the original series
        fill_value: Value to use for missing values due to lagging (None for NaN)
        
    Returns:
        If data is a NumPy array:
            List of arrays with lagged series
        If data is a Pandas Series:
            DataFrame with lagged series as columns
        
    Raises:
        DimensionError: If the input is not 1D
        ValueError: If lags is less than 1 or contains non-positive values
        
    Examples:
        >>> import numpy as np
        >>> from mfe.utils.data_transformations import lag_series
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> lagged = lag_series(x, lags=2)
        >>> len(lagged)
        3
        >>> np.array_equal(lagged[0], [1, 2, 3, 4, 5])
        True
        >>> np.array_equal(lagged[1], [0, 1, 2, 3, 4])
        True
        >>> np.array_equal(lagged[2], [0, 0, 1, 2, 3])
        True
        
        >>> import pandas as pd
        >>> s = pd.Series([1, 2, 3, 4, 5], index=pd.date_range('2020-01-01', periods=5))
        >>> lag_series(s, lags=[1, 3], fill_value=0)
        <BLANKLINE>
                   0  1  3
        2020-01-01  1  0  0
        2020-01-02  2  1  0
        2020-01-03  3  2  0
        2020-01-04  4  3  1
        2020-01-05  5  4  2
        """
    # Ensure data is 1D
    if isinstance(data, np.ndarray) and data.ndim != 1:
        raise_dimension_error(
            "Input data must be 1D",
            array_name="data",
            expected_shape="(n,)",
            actual_shape=data.shape
        )
    
    # Convert single lag to list
    if isinstance(lags, int):
        lag_list = list(range(1, lags + 1))
    else:
        # Validate lag values
        if any(lag < 1 for lag in lags):
            raise ValueError("All lag values must be positive integers")
        lag_list = sorted(lags)
    
    # Handle Pandas Series
    if isinstance(data, pd.Series):
        result_dict = {}
        
        # Include original series if requested
        if include_original:
            result_dict[0] = data
        
        # Create lagged series
        for lag in lag_list:
            lagged = data.shift(lag)
            if fill_value is not None:
                lagged = lagged.fillna(fill_value)
            result_dict[lag] = lagged
        
        # Combine into DataFrame
        return pd.DataFrame(result_dict)
    
    # Handle NumPy array
    else:
        # Convert to numpy array if not already
        data_array = np.asarray(data)
        n = len(data_array)
        
        result_list = []
        
        # Include original series if requested
        if include_original:
            result_list.append(data_array.copy())
        
        # Create lagged series
        for lag in lag_list:
            lagged = np.zeros_like(data_array)
            lagged[lag:] = data_array[:-lag]
            
            # Replace zeros with fill_value if specified
            if fill_value is not None:
                lagged[:lag] = fill_value
            
            result_list.append(lagged)
        
        return result_list


@jit(nopython=True, cache=True)

def _rolling_window_numba(data: np.ndarray, window_size: int, step: int = 1) -> np.ndarray:
    """
    Numba-accelerated implementation of rolling_window.
    
    Args:
        data: 1D array of time series data
        window_size: Size of each window
        step: Step size between windows
        
    Returns:
        2D array with each row being a window
    """
    n = len(data)
    num_windows = max(0, (n - window_size) // step + 1)
    
    # Initialize result matrix
    result = np.zeros((num_windows, window_size))
    
    # Fill the matrix
    for i in range(num_windows):
        start_idx = i * step
        end_idx = start_idx + window_size
        result[i, :] = data[start_idx:end_idx]
    
    return result


@validate_input_time_series(0)
@validate_input_bounds(1, lower_bound=1, param_name="window_size")
@validate_input_bounds(2, lower_bound=1, param_name="step")

def rolling_window(
    data: TimeSeriesData,
    window_size: int,
    step: int = 1
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Create rolling windows of a time series.
    
    This function creates a matrix where each row is a window of the input time
    series. The windows can overlap depending on the step size.
    
    Args:
        data: Time series data (NumPy array or Pandas Series)
        window_size: Size of each window
        step: Step size between windows
        
    Returns:
        Matrix with each row being a window, with the same type as the input
        
    Raises:
        DimensionError: If the input is not 1D
        ValueError: If window_size or step is less than 1
        
    Examples:
        >>> import numpy as np
        >>> from mfe.utils.data_transformations import rolling_window
        >>> x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> rolling_window(x, window_size=3, step=2)
        array([[ 1.,  2.,  3.],
               [ 3.,  4.,  5.],
               [ 5.,  6.,  7.],
               [ 7.,  8.,  9.]])
        
        >>> import pandas as pd
        >>> s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
        ...               index=pd.date_range('2020-01-01', periods=10))
        >>> rolling_window(s, window_size=3, step=3)
        <BLANKLINE>
                      0    1    2
        2020-01-01  1.0  2.0  3.0
        2020-01-04  4.0  5.0  6.0
        2020-01-07  7.0  8.0  9.0
        """
    # Ensure data is 1D
    if isinstance(data, np.ndarray) and data.ndim != 1:
        raise_dimension_error(
            "Input data must be 1D",
            array_name="data",
            expected_shape="(n,)",
            actual_shape=data.shape
        )
    
    # Handle Pandas Series
    if isinstance(data, pd.Series):
        # Convert to numpy array for processing
        data_array = data.values
        n = len(data_array)
        num_windows = max(0, (n - window_size) // step + 1)
        
        # Use Numba-accelerated implementation if available
        if HAS_NUMBA:
            result_array = _rolling_window_numba(data_array, window_size, step)
        else:
            # Pure NumPy implementation
            # Initialize result matrix
            result_array = np.zeros((num_windows, window_size))
            
            # Fill the matrix
            for i in range(num_windows):
                start_idx = i * step
                end_idx = start_idx + window_size
                result_array[i, :] = data_array[start_idx:end_idx]
        
        # Create index for result DataFrame
        if num_windows > 0:
            result_index = data.index[::step][:num_windows]
            col_names = list(range(window_size))
            result_df = pd.DataFrame(result_array, index=result_index, columns=col_names)
            return result_df
        else:
            # Return empty DataFrame with correct columns
            col_names = list(range(window_size))
            return pd.DataFrame(columns=col_names)
    
    # Handle NumPy array
    else:
        # Convert to numpy array if not already
        data_array = np.asarray(data)
        
        # Use Numba-accelerated implementation if available
        if HAS_NUMBA:
            result = _rolling_window_numba(data_array, window_size, step)
        else:
            # Pure NumPy implementation
            n = len(data_array)
            num_windows = max(0, (n - window_size) // step + 1)
            
            # Initialize result matrix
            result = np.zeros((num_windows, window_size))
            
            # Fill the matrix
            for i in range(num_windows):
                start_idx = i * step
                end_idx = start_idx + window_size
                result[i, :] = data_array[start_idx:end_idx]
        
        return result


@validate_input_time_series(0)
@validate_input_bounds(1, lower_bound=1, param_name="window_size")

def rolling_mean(
    data: TimeSeriesData,
    window_size: int,
    center: bool = False
) -> TimeSeriesData:
    """
    Compute rolling mean of a time series.
    
    This function computes the rolling mean of a time series with a specified
    window size.
    
    Args:
        data: Time series data (NumPy array, Pandas Series, or DataFrame)
        window_size: Size of the rolling window
        center: If True, the window is centered on each point; otherwise,
               the window ends at each point
        
    Returns:
        Rolling mean with the same type as the input
        
    Raises:
        ValueError: If window_size is less than 1
        
    Examples:
        >>> import numpy as np
        >>> from mfe.utils.data_transformations import rolling_mean
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> rolling_mean(x, window_size=3)
        array([nan, nan,  2.,  3.,  4.])
        >>> rolling_mean(x, window_size=3, center=True)
        array([nan,  2.,  3.,  4., nan])
        
        >>> import pandas as pd
        >>> s = pd.Series([1, 2, 3, 4, 5], index=pd.date_range('2020-01-01', periods=5))
        >>> rolling_mean(s, window_size=3)
        2020-01-01    NaN
        2020-01-02    NaN
        2020-01-03    2.0
        2020-01-04    3.0
        2020-01-05    4.0
        Freq: D, dtype: float64
    """
    # Handle Pandas Series
    if isinstance(data, pd.Series):
        return data.rolling(window=window_size, center=center).mean()
    
    # Handle Pandas DataFrame
    elif isinstance(data, pd.DataFrame):
        return data.rolling(window=window_size, center=center).mean()
    
    # Handle NumPy array
    else:
        # Convert to numpy array if not already
        data_array = np.asarray(data)
        
        # Handle 1D arrays
        if data_array.ndim == 1:
            n = len(data_array)
            result = np.full_like(data_array, np.nan, dtype=float)
            
            if center:
                offset = window_size // 2
                for i in range(offset, n - offset):
                    start_idx = i - offset
                    end_idx = i + offset + (1 if window_size % 2 == 1 else 0)
                    result[i] = np.mean(data_array[start_idx:end_idx])
            else:
                for i in range(window_size - 1, n):
                    result[i] = np.mean(data_array[i - window_size + 1:i + 1])
            
            return result
        
        # Handle 2D arrays (compute rolling mean for each column)
        elif data_array.ndim == 2:
            n, p = data_array.shape
            result = np.full_like(data_array, np.nan, dtype=float)
            
            for j in range(p):
                if center:
                    offset = window_size // 2
                    for i in range(offset, n - offset):
                        start_idx = i - offset
                        end_idx = i + offset + (1 if window_size % 2 == 1 else 0)
                        result[i, j] = np.mean(data_array[start_idx:end_idx, j])
                else:
                    for i in range(window_size - 1, n):
                        result[i, j] = np.mean(data_array[i - window_size + 1:i + 1, j])
            
            return result
        
        # Handle higher dimensional arrays
        else:
            raise_dimension_error(
                "Input data must be 1D or 2D",
                array_name="data",
                expected_shape="(n,) or (n, p)",
                actual_shape=data_array.shape
            )


@validate_input_time_series(0)
@validate_input_bounds(1, lower_bound=1, param_name="window_size")

def rolling_variance(
    data: TimeSeriesData,
    window_size: int,
    center: bool = False,
    ddof: int = 1
) -> TimeSeriesData:
    """
    Compute rolling variance of a time series.
    
    This function computes the rolling variance of a time series with a specified
    window size.
    
    Args:
        data: Time series data (NumPy array, Pandas Series, or DataFrame)
        window_size: Size of the rolling window
        center: If True, the window is centered on each point; otherwise,
               the window ends at each point
        ddof: Delta degrees of freedom for variance calculation
        
    Returns:
        Rolling variance with the same type as the input
        
    Raises:
        ValueError: If window_size is less than 1
        
    Examples:
        >>> import numpy as np
        >>> from mfe.utils.data_transformations import rolling_variance
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> rolling_variance(x, window_size=3)
        array([nan, nan,  1.,  1.,  1.])
        >>> rolling_variance(x, window_size=3, center=True)
        array([nan,  1.,  1.,  1., nan])
        
        >>> import pandas as pd
        >>> s = pd.Series([1, 2, 3, 4, 5], index=pd.date_range('2020-01-01', periods=5))
        >>> rolling_variance(s, window_size=3)
        2020-01-01    NaN
        2020-01-02    NaN
        2020-01-03    1.0
        2020-01-04    1.0
        2020-01-05    1.0
        Freq: D, dtype: float64
    """
    # Handle Pandas Series
    if isinstance(data, pd.Series):
        return data.rolling(window=window_size, center=center).var(ddof=ddof)
    
    # Handle Pandas DataFrame
    elif isinstance(data, pd.DataFrame):
        return data.rolling(window=window_size, center=center).var(ddof=ddof)
    
    # Handle NumPy array
    else:
        # Convert to numpy array if not already
        data_array = np.asarray(data)
        
        # Handle 1D arrays
        if data_array.ndim == 1:
            n = len(data_array)
            result = np.full_like(data_array, np.nan, dtype=float)
            
            if center:
                offset = window_size // 2
                for i in range(offset, n - offset):
                    start_idx = i - offset
                    end_idx = i + offset + (1 if window_size % 2 == 1 else 0)
                    result[i] = np.var(data_array[start_idx:end_idx], ddof=ddof)
            else:
                for i in range(window_size - 1, n):
                    result[i] = np.var(data_array[i - window_size + 1:i + 1], ddof=ddof)
            
            return result
        
        # Handle 2D arrays (compute rolling variance for each column)
        elif data_array.ndim == 2:
            n, p = data_array.shape
            result = np.full_like(data_array, np.nan, dtype=float)
            
            for j in range(p):
                if center:
                    offset = window_size // 2
                    for i in range(offset, n - offset):
                        start_idx = i - offset
                        end_idx = i + offset + (1 if window_size % 2 == 1 else 0)
                        result[i, j] = np.var(data_array[start_idx:end_idx, j], ddof=ddof)
                else:
                    for i in range(window_size - 1, n):
                        result[i, j] = np.var(data_array[i - window_size + 1:i + 1, j], ddof=ddof)
            
            return result
        
        # Handle higher dimensional arrays
        else:
            raise_dimension_error(
                "Input data must be 1D or 2D",
                array_name="data",
                expected_shape="(n,) or (n, p)",
                actual_shape=data_array.shape
            )


@validate_input_time_series(0)
@validate_input_bounds(1, lower_bound=1, param_name="window_size")

def rolling_skewness(
    data: TimeSeriesData,
    window_size: int,
    center: bool = False,
    bias: bool = False
) -> TimeSeriesData:
    """
    Compute rolling skewness of a time series.
    
    This function computes the rolling skewness of a time series with a specified
    window size.
    
    Args:
        data: Time series data (NumPy array, Pandas Series, or DataFrame)
        window_size: Size of the rolling window
        center: If True, the window is centered on each point; otherwise,
               the window ends at each point
        bias: Whether to use the biased or unbiased estimator
        
    Returns:
        Rolling skewness with the same type as the input
        
    Raises:
        ValueError: If window_size is less than 1
        
    Examples:
        >>> import numpy as np
        >>> from mfe.utils.data_transformations import rolling_skewness
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> rolling_skewness(x, window_size=3)
        array([nan, nan,  0.,  0.,  0.])
        >>> rolling_skewness(x, window_size=3, center=True)
        array([nan,  0.,  0.,  0., nan])
        
        >>> import pandas as pd
        >>> s = pd.Series([1, 2, 3, 4, 5], index=pd.date_range('2020-01-01', periods=5))
        >>> rolling_skewness(s, window_size=3)
        2020-01-01    NaN
        2020-01-02    NaN
        2020-01-03    0.0
        2020-01-04    0.0
        2020-01-05    0.0
        Freq: D, dtype: float64
    """
    # Handle Pandas Series
    if isinstance(data, pd.Series):
        return data.rolling(window=window_size, center=center).skew(bias=bias)
    
    # Handle Pandas DataFrame
    elif isinstance(data, pd.DataFrame):
        return data.rolling(window=window_size, center=center).skew(bias=bias)
    
    # Handle NumPy array
    else:
        # Convert to numpy array if not already
        data_array = np.asarray(data)
        
        # Handle 1D arrays
        if data_array.ndim == 1:
            n = len(data_array)
            result = np.full_like(data_array, np.nan, dtype=float)
            
            if center:
                offset = window_size // 2
                for i in range(offset, n - offset):
                    start_idx = i - offset
                    end_idx = i + offset + (1 if window_size % 2 == 1 else 0)
                    result[i] = stats.skew(data_array[start_idx:end_idx], bias=bias)
            else:
                for i in range(window_size - 1, n):
                    result[i] = stats.skew(data_array[i - window_size + 1:i + 1], bias=bias)
            
            return result
        
        # Handle 2D arrays (compute rolling skewness for each column)
        elif data_array.ndim == 2:
            n, p = data_array.shape
            result = np.full_like(data_array, np.nan, dtype=float)
            
            for j in range(p):
                if center:
                    offset = window_size // 2
                    for i in range(offset, n - offset):
                        start_idx = i - offset
                        end_idx = i + offset + (1 if window_size % 2 == 1 else 0)
                        result[i, j] = stats.skew(data_array[start_idx:end_idx, j], bias=bias)
                else:
                    for i in range(window_size - 1, n):
                        result[i, j] = stats.skew(data_array[i - window_size + 1:i + 1, j], bias=bias)
            
            return result
        
        # Handle higher dimensional arrays
        else:
            raise_dimension_error(
                "Input data must be 1D or 2D",
                array_name="data",
                expected_shape="(n,) or (n, p)",
                actual_shape=data_array.shape
            )


@validate_input_time_series(0)
@validate_input_bounds(1, lower_bound=1, param_name="window_size")

def rolling_kurtosis(
    data: TimeSeriesData,
    window_size: int,
    center: bool = False,
    bias: bool = False,
    excess: bool = True
) -> TimeSeriesData:
    """
    Compute rolling kurtosis of a time series.
    
    This function computes the rolling kurtosis of a time series with a specified
    window size.
    
    Args:
        data: Time series data (NumPy array, Pandas Series, or DataFrame)
        window_size: Size of the rolling window
        center: If True, the window is centered on each point; otherwise,
               the window ends at each point
        bias: Whether to use the biased or unbiased estimator
        excess: If True, compute excess kurtosis (kurtosis - 3)
        
    Returns:
        Rolling kurtosis with the same type as the input
        
    Raises:
        ValueError: If window_size is less than 1
        
    Examples:
        >>> import numpy as np
        >>> from mfe.utils.data_transformations import rolling_kurtosis
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> rolling_kurtosis(x, window_size=3)
        array([nan, nan, -1.5, -1.5, -1.5])
        >>> rolling_kurtosis(x, window_size=3, center=True)
        array([nan, -1.5, -1.5, -1.5, nan])
        
        >>> import pandas as pd
        >>> s = pd.Series([1, 2, 3, 4, 5], index=pd.date_range('2020-01-01', periods=5))
        >>> rolling_kurtosis(s, window_size=3)
        2020-01-01    NaN
        2020-01-02    NaN
        2020-01-03   -1.5
        2020-01-04   -1.5
        2020-01-05   -1.5
        Freq: D, dtype: float64
    """
    # Handle Pandas Series
    if isinstance(data, pd.Series):
        return data.rolling(window=window_size, center=center).kurt(bias=bias)
    
    # Handle Pandas DataFrame
    elif isinstance(data, pd.DataFrame):
        return data.rolling(window=window_size, center=center).kurt(bias=bias)
    
    # Handle NumPy array
    else:
        # Convert to numpy array if not already
        data_array = np.asarray(data)
        
        # Handle 1D arrays
        if data_array.ndim == 1:
            n = len(data_array)
            result = np.full_like(data_array, np.nan, dtype=float)
            
            if center:
                offset = window_size // 2
                for i in range(offset, n - offset):
                    start_idx = i - offset
                    end_idx = i + offset + (1 if window_size % 2 == 1 else 0)
                    result[i] = stats.kurtosis(data_array[start_idx:end_idx], bias=bias, fisher=excess)
            else:
                for i in range(window_size - 1, n):
                    result[i] = stats.kurtosis(data_array[i - window_size + 1:i + 1], bias=bias, fisher=excess)
            
            return result
        
        # Handle 2D arrays (compute rolling kurtosis for each column)
        elif data_array.ndim == 2:
            n, p = data_array.shape
            result = np.full_like(data_array, np.nan, dtype=float)
            
            for j in range(p):
                if center:
                    offset = window_size // 2
                    for i in range(offset, n - offset):
                        start_idx = i - offset
                        end_idx = i + offset + (1 if window_size % 2 == 1 else 0)
                        result[i, j] = stats.kurtosis(data_array[start_idx:end_idx, j], bias=bias, fisher=excess)
                else:
                    for i in range(window_size - 1, n):
                        result[i, j] = stats.kurtosis(data_array[i - window_size + 1:i + 1, j], bias=bias, fisher=excess)
            
            return result
        
        # Handle higher dimensional arrays
        else:
            raise_dimension_error(
                "Input data must be 1D or 2D",
                array_name="data",
                expected_shape="(n,) or (n, p)",
                actual_shape=data_array.shape
            )


# Register Numba-accelerated functions if available
def _register_numba_functions() -> None:
    """
    Register Numba JIT-compiled functions for data transformations.
    
    This function is called during module initialization to register
    performance-critical functions for JIT compilation if Numba is available.
    """
    if HAS_NUMBA:
        # The functions are already decorated with @jit, so no additional
        # registration is needed here. This function is kept for consistency
        # with the module structure and potential future enhancements.
        logger.debug("Data transformation Numba JIT functions registered")
    else:
        logger.info("Numba not available. Data transformations will use pure NumPy implementations.")


# Initialize the module
_register_numba_functions()
