# mfe/models/realized/compute_median.py
"""
Compute median prices for each unique timestamp in high-frequency data.

This module provides functionality to compute median prices (or other aggregation methods)
for each unique timestamp in high-frequency financial data. This is crucial for preprocessing
tick data where multiple observations can occur at the same time. The implementation leverages
Pandas' efficient groupby operations and time handling capabilities for optimal performance.

Functions:
    compute_median: Compute median prices for each unique timestamp
"""

import logging
import warnings
from typing import Callable, Dict, Literal, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
from numba import jit

from ...core.exceptions import ParameterError, DimensionError

# Set up module-level logger
logger = logging.getLogger("mfe.models.realized.compute_median")

# Define valid aggregation methods
VALID_AGGREGATION_METHODS = ['median', 'mean', 'first', 'last', 'min', 'max']


@jit(nopython=True, cache=True)
def _compute_median_numba(prices: np.ndarray, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba-accelerated implementation to compute median prices for unique timestamps.
    
    This function is used as a fallback when Pandas is not available or for very large datasets
    where memory efficiency is critical.
    
    Args:
        prices: Array of price data
        times: Array of time points corresponding to prices
        
    Returns:
        Tuple of (unique_times, median_prices)
    """
    # Find unique times and their indices
    unique_times, indices = np.unique(times, return_inverse=True)
    n_unique = len(unique_times)
    
    # Initialize array for median prices
    median_prices = np.zeros(n_unique, dtype=np.float64)
    
    # Compute median for each unique time
    for i in range(n_unique):
        # Get prices for this time
        mask = indices == i
        time_prices = prices[mask]
        
        # Compute median
        if len(time_prices) % 2 == 1:
            # Odd number of elements
            median_prices[i] = np.sort(time_prices)[len(time_prices) // 2]
        else:
            # Even number of elements
            sorted_prices = np.sort(time_prices)
            middle_idx = len(time_prices) // 2
            median_prices[i] = (sorted_prices[middle_idx - 1] + sorted_prices[middle_idx]) / 2
    
    return unique_times, median_prices



def compute_median(
    prices: Union[np.ndarray, pd.Series],
    times: Union[np.ndarray, pd.Series, pd.DatetimeIndex],
    method: Union[str, Callable] = 'median',
    return_index: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute median (or other aggregation) of prices for each unique timestamp.
    
    This function aggregates price data when multiple observations occur at the same timestamp,
    which is common in high-frequency financial data. It leverages Pandas' efficient groupby
    operations for optimal performance and supports various aggregation methods.
    
    Args:
        prices: Array or Series of price data
        times: Array, Series, or DatetimeIndex of time points corresponding to prices
        method: Aggregation method ('median', 'mean', 'first', 'last', 'min', 'max') or custom function
        return_index: Whether to return the indices of unique times in the original array
        
    Returns:
        If return_index=False:
            Tuple of (unique_times, aggregated_prices)
        If return_index=True:
            Tuple of (unique_times, aggregated_prices, unique_indices)
        
    Raises:
        ParameterError: If method is not recognized
        DimensionError: If prices and times have different lengths
        ValueError: If inputs are invalid
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.compute_median import compute_median
        >>> prices = np.array([100.0, 100.5, 101.0, 101.5, 102.0])
        >>> times = np.array([1, 1, 2, 3, 3])
        >>> unique_times, median_prices = compute_median(prices, times)
        >>> unique_times
        array([1, 2, 3])
        >>> median_prices
        array([100.25, 101.  , 101.75])
        
        >>> import pandas as pd
        >>> times = pd.to_datetime(['2023-01-01 09:30:00', '2023-01-01 09:30:00',
        ...                         '2023-01-01 09:31:00', '2023-01-01 09:32:00',
        ...                         '2023-01-01 09:32:00'])
        >>> unique_times, mean_prices = compute_median(prices, times, method='mean')
        >>> mean_prices
        array([100.25, 101.  , 101.75])
    """
    # Validate inputs
    if not isinstance(prices, (np.ndarray, pd.Series)):
        raise TypeError("prices must be a NumPy array or Pandas Series")
    
    if not isinstance(times, (np.ndarray, pd.Series, pd.DatetimeIndex)):
        raise TypeError("times must be a NumPy array, Pandas Series, or DatetimeIndex")
    
    # Convert to numpy arrays if needed
    prices_array = np.asarray(prices)
    
    # Handle DatetimeIndex
    if isinstance(times, pd.DatetimeIndex):
        times_array = times.values
    else:
        times_array = np.asarray(times)
    
    # Check dimensions
    if prices_array.ndim != 1:
        raise DimensionError("prices must be 1-dimensional")
    
    if times_array.ndim != 1:
        raise DimensionError("times must be 1-dimensional")
    
    if len(prices_array) != len(times_array):
        raise DimensionError(
            f"prices length ({len(prices_array)}) must match times length ({len(times_array)})"
        )
    
    # Validate method
    if isinstance(method, str):
        if method.lower() not in VALID_AGGREGATION_METHODS:
            raise ParameterError(
                f"method must be one of {VALID_AGGREGATION_METHODS} or a callable, got {method}"
            )
    elif not callable(method):
        raise ParameterError("method must be a string or callable")
    
    # Use Pandas for efficient groupby operations
    try:
        # Create DataFrame
        df = pd.DataFrame({'price': prices_array, 'time': times_array})
        
        # Group by time and apply aggregation method
        if isinstance(method, str):
            if method.lower() == 'median':
                grouped = df.groupby('time')['price'].median()
            elif method.lower() == 'mean':
                grouped = df.groupby('time')['price'].mean()
            elif method.lower() == 'first':
                grouped = df.groupby('time')['price'].first()
            elif method.lower() == 'last':
                grouped = df.groupby('time')['price'].last()
            elif method.lower() == 'min':
                grouped = df.groupby('time')['price'].min()
            elif method.lower() == 'max':
                grouped = df.groupby('time')['price'].max()
        else:
            # Custom aggregation function
            grouped = df.groupby('time')['price'].apply(method)
        
        # Extract unique times and aggregated prices
        unique_times = np.array(grouped.index)
        aggregated_prices = np.array(grouped.values)
        
        # Return indices if requested
        if return_index:
            # Find indices of first occurrence of each unique time
            _, unique_indices = np.unique(times_array, return_index=True)
            # Sort indices to match the order of unique_times
            unique_indices = unique_indices[np.argsort(times_array[unique_indices])]
            return unique_times, aggregated_prices, unique_indices
        
        return unique_times, aggregated_prices
    
    except Exception as e:
        logger.warning(f"Pandas groupby operation failed: {str(e)}. Falling back to Numba implementation.")
        
        # Fall back to Numba implementation
        unique_times, aggregated_prices = _compute_median_numba(prices_array, times_array)
        
        if return_index:
            # Find indices of first occurrence of each unique time
            _, unique_indices = np.unique(times_array, return_index=True)
            # Sort indices to match the order of unique_times
            unique_indices = unique_indices[np.argsort(times_array[unique_indices])]
            return unique_times, aggregated_prices, unique_indices
        
        return unique_times, aggregated_prices


# mfe/models/realized/compute_median.py
"""
Compute median prices for each unique timestamp in high-frequency data.

This module provides functionality to compute median prices (or other aggregation methods)
for each unique timestamp in high-frequency financial data. This is crucial for preprocessing
tick data where multiple observations can occur at the same time. The implementation leverages
Pandas' efficient groupby operations and time handling capabilities for optimal performance.

Functions:
    compute_median: Compute median prices for each unique timestamp
"""

import logging
import warnings
from typing import Callable, Dict, Literal, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
from numba import jit

from ...core.exceptions import ParameterError, DimensionError

# Set up module-level logger
logger = logging.getLogger("mfe.models.realized.compute_median")

# Define valid aggregation methods
VALID_AGGREGATION_METHODS = ['median', 'mean', 'first', 'last', 'min', 'max']


@jit(nopython=True, cache=True)
def _compute_median_numba(prices: np.ndarray, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba-accelerated implementation to compute median prices for unique timestamps.
    
    This function is used as a fallback when Pandas is not available or for very large datasets
    where memory efficiency is critical.
    
    Args:
        prices: Array of price data
        times: Array of time points corresponding to prices
        
    Returns:
        Tuple of (unique_times, median_prices)
    """
    # Find unique times and their indices
    unique_times, indices = np.unique(times, return_inverse=True)
    n_unique = len(unique_times)
    
    # Initialize array for median prices
    median_prices = np.zeros(n_unique, dtype=np.float64)
    
    # Compute median for each unique time
    for i in range(n_unique):
        # Get prices for this time
        mask = indices == i
        time_prices = prices[mask]
        
        # Compute median
        if len(time_prices) % 2 == 1:
            # Odd number of elements
            median_prices[i] = np.sort(time_prices)[len(time_prices) // 2]
        else:
            # Even number of elements
            sorted_prices = np.sort(time_prices)
            middle_idx = len(time_prices) // 2
            median_prices[i] = (sorted_prices[middle_idx - 1] + sorted_prices[middle_idx]) / 2
    
    return unique_times, median_prices



def compute_median(
    prices: Union[np.ndarray, pd.Series],
    times: Union[np.ndarray, pd.Series, pd.DatetimeIndex],
    method: Union[str, Callable] = 'median',
    return_index: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute median (or other aggregation) of prices for each unique timestamp.
    
    This function aggregates price data when multiple observations occur at the same timestamp,
    which is common in high-frequency financial data. It leverages Pandas' efficient groupby
    operations for optimal performance and supports various aggregation methods.
    
    Args:
        prices: Array or Series of price data
        times: Array, Series, or DatetimeIndex of time points corresponding to prices
        method: Aggregation method ('median', 'mean', 'first', 'last', 'min', 'max') or custom function
        return_index: Whether to return the indices of unique times in the original array
        
    Returns:
        If return_index=False:
            Tuple of (unique_times, aggregated_prices)
        If return_index=True:
            Tuple of (unique_times, aggregated_prices, unique_indices)
        
    Raises:
        ParameterError: If method is not recognized
        DimensionError: If prices and times have different lengths
        ValueError: If inputs are invalid
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.compute_median import compute_median
        >>> prices = np.array([100.0, 100.5, 101.0, 101.5, 102.0])
        >>> times = np.array([1, 1, 2, 3, 3])
        >>> unique_times, median_prices = compute_median(prices, times)
        >>> unique_times
        array([1, 2, 3])
        >>> median_prices
        array([100.25, 101.  , 101.75])
        
        >>> import pandas as pd
        >>> times = pd.to_datetime(['2023-01-01 09:30:00', '2023-01-01 09:30:00',
        ...                         '2023-01-01 09:31:00', '2023-01-01 09:32:00',
        ...                         '2023-01-01 09:32:00'])
        >>> unique_times, mean_prices = compute_median(prices, times, method='mean')
        >>> mean_prices
        array([100.25, 101.  , 101.75])
    """
    # Validate inputs
    if not isinstance(prices, (np.ndarray, pd.Series)):
        raise TypeError("prices must be a NumPy array or Pandas Series")
    
    if not isinstance(times, (np.ndarray, pd.Series, pd.DatetimeIndex)):
        raise TypeError("times must be a NumPy array, Pandas Series, or DatetimeIndex")
    
    # Convert to numpy arrays if needed
    prices_array = np.asarray(prices)
    
    # Handle DatetimeIndex
    if isinstance(times, pd.DatetimeIndex):
        times_array = times.values
    else:
        times_array = np.asarray(times)
    
    # Check dimensions
    if prices_array.ndim != 1:
        raise DimensionError("prices must be 1-dimensional")
    
    if times_array.ndim != 1:
        raise DimensionError("times must be 1-dimensional")
    
    if len(prices_array) != len(times_array):
        raise DimensionError(
            f"prices length ({len(prices_array)}) must match times length ({len(times_array)})"
        )
    
    # Validate method
    if isinstance(method, str):
        if method.lower() not in VALID_AGGREGATION_METHODS:
            raise ParameterError(
                f"method must be one of {VALID_AGGREGATION_METHODS} or a callable, got {method}"
            )
    elif not callable(method):
        raise ParameterError("method must be a string or callable")
    
    # Use Pandas for efficient groupby operations
    try:
        # Create DataFrame
        df = pd.DataFrame({'price': prices_array, 'time': times_array})
        
        # Group by time and apply aggregation method
        if isinstance(method, str):
            if method.lower() == 'median':
                grouped = df.groupby('time')['price'].median()
            elif method.lower() == 'mean':
                grouped = df.groupby('time')['price'].mean()
            elif method.lower() == 'first':
                grouped = df.groupby('time')['price'].first()
            elif method.lower() == 'last':
                grouped = df.groupby('time')['price'].last()
            elif method.lower() == 'min':
                grouped = df.groupby('time')['price'].min()
            elif method.lower() == 'max':
                grouped = df.groupby('time')['price'].max()
        else:
            # Custom aggregation function
            grouped = df.groupby('time')['price'].apply(method)
        
        # Extract unique times and aggregated prices
        unique_times = np.array(grouped.index)
        aggregated_prices = np.array(grouped.values)
        
        # Return indices if requested
        if (return_index):
            # Find indices of first occurrence of each unique time
            _, unique_indices = np.unique(times_array, return_index=True)
            # Sort indices to match the order of unique_times
            unique_indices = unique_indices[np.argsort(times_array[unique_indices])]
            return unique_times, aggregated_prices, unique_indices
        
        return unique_times, aggregated_prices
    
    except Exception as e:
        logger.warning(f"Pandas groupby operation failed: {str(e)}. Falling back to Numba implementation.")
        
        # Fall back to Numba implementation
        unique_times, aggregated_prices = _compute_median_numba(prices_array, times_array)
        
        if (return_index):
            # Find indices of first occurrence of each unique time
            _, unique_indices = np.unique(times_array, return_index=True)
            # Sort indices to match the order of unique_times
            unique_indices = unique_indices[np.argsort(times_array[unique_indices])]
            return unique_times, aggregated_prices, unique_indices
        
        return unique_times, aggregated_prices