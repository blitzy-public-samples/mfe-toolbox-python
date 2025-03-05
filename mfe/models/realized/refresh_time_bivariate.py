# mfe/models/realized/refresh_time_bivariate.py
"""
Optimized refresh time algorithm for bivariate high-frequency data.

This module implements a specialized refresh time algorithm for synchronizing
asynchronous high-frequency financial data between two assets. The refresh time
method aligns observations by creating a common time grid where each point represents
the time when both assets have experienced at least one price update.

While the general refresh_time function in the refresh_time module handles any
number of assets, this bivariate implementation is optimized specifically for
the common case of two assets, providing better performance and memory efficiency.
The implementation leverages Pandas' powerful time series capabilities for efficient
data manipulation and provides comprehensive visualization tools for analyzing the
synchronization process. The module supports various time formats with microsecond
precision for ultra-high-frequency data.

Functions:
    refresh_time_bivariate: Compute refresh times for bivariate high-frequency data
    refresh_time_bivariate_prices: Synchronize price data using bivariate refresh times
    refresh_time_bivariate_returns: Compute synchronized returns using bivariate refresh times
    visualize_refresh_time_bivariate: Create visualization of bivariate refresh time synchronization
    refresh_time_bivariate_statistics: Compute statistics about bivariate refresh time sampling
"""

import logging
import warnings
from typing import (
    Any, Dict, List, Literal, Optional, Tuple, Union, cast, overload
)
import numpy as np
import pandas as pd

from ...core.exceptions import (
    DimensionError, NumericError, ParameterError, 
    raise_dimension_error, raise_numeric_error, warn_numeric
)
from .utils import (
    seconds2unit, unit2seconds, wall2seconds, seconds2wall,
    unit2wall, wall2unit, align_time
)

# Set up module-level logger
logger = logging.getLogger("mfe.models.realized.refresh_time_bivariate")

# Try to import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.info("Matplotlib not available. Visualization functions will be disabled.")


def refresh_time_bivariate(
    times1: Union[np.ndarray, pd.Series, pd.DatetimeIndex],
    times2: Union[np.ndarray, pd.Series, pd.DatetimeIndex],
    convert_to_seconds: bool = True,
    base_time: Optional[Union[str, pd.Timestamp]] = None,
    return_indices: bool = False,
    return_pandas: bool = False
) -> Union[
    np.ndarray,
    pd.DatetimeIndex,
    Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    Tuple[pd.DatetimeIndex, Tuple[np.ndarray, np.ndarray]]
]:
    """
    Compute refresh times for bivariate high-frequency data.
    
    This function is an optimized implementation of the refresh time algorithm
    specifically for the common case of two assets. It identifies points where
    both assets have experienced at least one price update, creating a synchronized
    time grid for bivariate analysis.
    
    Args:
        times1: Time series for the first asset
        times2: Time series for the second asset
        convert_to_seconds: Whether to convert all times to seconds for processing
        base_time: Base time for datetime conversion (required if times are datetime and convert_to_seconds=True)
        return_indices: Whether to return indices of refresh times in original series
        return_pandas: Whether to return a pandas DatetimeIndex instead of numpy array
        
    Returns:
        Refresh times as numpy array or pandas DatetimeIndex, optionally with indices
        
    Raises:
        ValueError: If inputs have invalid dimensions or if parameters are invalid
        TypeError: If inputs have incorrect types
        
    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from mfe.models.realized.refresh_time_bivariate import refresh_time_bivariate
        >>> # Example with numeric times
        >>> times1 = np.array([1, 3, 5, 7, 9])
        >>> times2 = np.array([2, 4, 6, 8, 10])
        >>> refresh_times = refresh_time_bivariate(times1, times2)
        >>> refresh_times
        array([ 2.,  4.,  6.,  8., 10.])
        
        >>> # Example with datetime times
        >>> times1 = pd.to_datetime(['2023-01-01 09:30:00', '2023-01-01 09:31:00', 
        ...                          '2023-01-01 09:32:00'])
        >>> times2 = pd.to_datetime(['2023-01-01 09:30:30', '2023-01-01 09:31:30', 
        ...                          '2023-01-01 09:32:30'])
        >>> refresh_times = refresh_time_bivariate(times1, times2, return_pandas=True)
        >>> refresh_times
        DatetimeIndex(['2023-01-01 09:30:30', '2023-01-01 09:31:30', 
                       '2023-01-01 09:32:30'],
                      dtype='datetime64[ns]', freq=None)
    """
    # Process time series for first asset
    is_datetime = False
    
    # Process first time series
    if isinstance(times1, pd.DatetimeIndex):
        is_datetime = True
        if convert_to_seconds:
            if base_time is None:
                # Use the first time point as base time
                base_time = times1[0]
            times1_processed = wall2seconds(times1, pd.Timestamp(base_time).timestamp())
        else:
            # Keep as DatetimeIndex
            times1_processed = times1
    elif isinstance(times1, pd.Series):
        if pd.api.types.is_datetime64_dtype(times1):
            is_datetime = True
            datetime_idx = pd.DatetimeIndex(times1)
            if convert_to_seconds:
                if base_time is None:
                    # Use the first time point as base time
                    base_time = datetime_idx[0]
                times1_processed = wall2seconds(datetime_idx, pd.Timestamp(base_time).timestamp())
            else:
                # Keep as DatetimeIndex
                times1_processed = datetime_idx
        else:
            # Numeric Series
            times1_processed = np.asarray(times1)
    else:
        # Assume numpy array or list of numeric values
        try:
            times1_processed = np.asarray(times1, dtype=float)
        except (ValueError, TypeError):
            # Try to convert to datetime
            try:
                datetime_idx = pd.DatetimeIndex(pd.to_datetime(times1))
                is_datetime = True
                if convert_to_seconds:
                    if base_time is None:
                        # Use the first time point as base time
                        base_time = datetime_idx[0]
                    times1_processed = wall2seconds(datetime_idx, pd.Timestamp(base_time).timestamp())
                else:
                    # Keep as DatetimeIndex
                    times1_processed = datetime_idx
            except (ValueError, TypeError):
                raise ValueError("Could not process first time series: not recognized as numeric or datetime")
    
    # Process second time series
    if isinstance(times2, pd.DatetimeIndex):
        is_datetime_2 = True
        if convert_to_seconds:
            if base_time is None:
                # Use the first time point as base time
                base_time = times2[0]
            times2_processed = wall2seconds(times2, pd.Timestamp(base_time).timestamp())
        else:
            # Keep as DatetimeIndex
            times2_processed = times2
    elif isinstance(times2, pd.Series):
        if pd.api.types.is_datetime64_dtype(times2):
            is_datetime_2 = True
            datetime_idx = pd.DatetimeIndex(times2)
            if convert_to_seconds:
                if base_time is None:
                    # Use the first time point as base time
                    base_time = datetime_idx[0]
                times2_processed = wall2seconds(datetime_idx, pd.Timestamp(base_time).timestamp())
            else:
                # Keep as DatetimeIndex
                times2_processed = datetime_idx
        else:
            # Numeric Series
            times2_processed = np.asarray(times2)
            is_datetime_2 = False
    else:
        # Assume numpy array or list of numeric values
        try:
            times2_processed = np.asarray(times2, dtype=float)
            is_datetime_2 = False
        except (ValueError, TypeError):
            # Try to convert to datetime
            try:
                datetime_idx = pd.DatetimeIndex(pd.to_datetime(times2))
                is_datetime_2 = True
                if convert_to_seconds:
                    if base_time is None:
                        # Use the first time point as base time
                        base_time = datetime_idx[0]
                    times2_processed = wall2seconds(datetime_idx, pd.Timestamp(base_time).timestamp())
                else:
                    # Keep as DatetimeIndex
                    times2_processed = datetime_idx
            except (ValueError, TypeError):
                raise ValueError("Could not process second time series: not recognized as numeric or datetime")
    
    # Check for consistency in datetime status
    if is_datetime != is_datetime_2:
        raise ValueError("Both time series must be of the same type (either both datetime or both numeric)")
    
    # Check if we have a mix of datetime and numeric
    if is_datetime and convert_to_seconds:
        # All datetime series have been converted to seconds
        is_datetime = False
    
    # Compute refresh times
    if is_datetime and not convert_to_seconds:
        # Working with datetime indices
        # Convert to pandas Series for easier manipulation
        time_series1 = pd.Series(1, index=times1_processed)
        time_series2 = pd.Series(1, index=times2_processed)
        
        # Combine both series and resample to get refresh times
        combined = pd.concat([time_series1, time_series2], axis=1, join='outer')
        combined = combined.fillna(0)
        
        # Find points where both assets have been observed
        cumulative = combined.cumsum()
        refresh_mask = (cumulative > 0).all(axis=1)
        refresh_times_idx = combined.index[refresh_mask]
        
        # Get indices of refresh times in original series
        if return_indices:
            # Find indices of times that are less than or equal to each refresh time
            idx1 = np.zeros(len(refresh_times_idx), dtype=int)
            idx2 = np.zeros(len(refresh_times_idx), dtype=int)
            
            for j, rt in enumerate(refresh_times_idx):
                # Find the last time point less than or equal to the refresh time
                idx1[j] = np.searchsorted(times1_processed, rt, side='right') - 1
                idx2[j] = np.searchsorted(times2_processed, rt, side='right') - 1
            
            if return_pandas:
                return refresh_times_idx, (idx1, idx2)
            else:
                # Convert to numpy array of timestamps
                refresh_times_array = np.array(refresh_times_idx.astype(np.int64) / 1e9)
                return refresh_times_array, (idx1, idx2)
        else:
            if return_pandas:
                return refresh_times_idx
            else:
                # Convert to numpy array of timestamps
                refresh_times_array = np.array(refresh_times_idx.astype(np.int64) / 1e9)
                return refresh_times_array
    else:
        # Working with numeric time arrays
        # Convert to numpy arrays
        times1_array = np.asarray(times1_processed, dtype=float)
        times2_array = np.asarray(times2_processed, dtype=float)
        
        # Ensure arrays are sorted
        if not np.all(np.diff(times1_array) >= 0):
            raise ValueError("First time series is not monotonically increasing")
        if not np.all(np.diff(times2_array) >= 0):
            raise ValueError("Second time series is not monotonically increasing")
        
        # Optimized bivariate refresh time algorithm
        # This is more efficient than the general algorithm for just two assets
        i1, i2 = 0, 0
        n1, n2 = len(times1_array), len(times2_array)
        refresh_times = []
        
        while i1 < n1 and i2 < n2:
            t1, t2 = times1_array[i1], times2_array[i2]
            
            # The refresh time is the maximum of the current times
            refresh_time = max(t1, t2)
            refresh_times.append(refresh_time)
            
            # Advance indices to the next observation after the refresh time
            while i1 < n1 and times1_array[i1] <= refresh_time:
                i1 += 1
            while i2 < n2 and times2_array[i2] <= refresh_time:
                i2 += 1
        
        # Convert to numpy array
        refresh_times_array = np.array(refresh_times)
        
        # Get indices of refresh times in original series
        if return_indices:
            # Find indices of times that are less than or equal to each refresh time
            idx1 = np.zeros(len(refresh_times_array), dtype=int)
            idx2 = np.zeros(len(refresh_times_array), dtype=int)
            
            for j, rt in enumerate(refresh_times_array):
                # Find the last time point less than or equal to the refresh time
                idx1[j] = np.searchsorted(times1_array, rt, side='right') - 1
                idx2[j] = np.searchsorted(times2_array, rt, side='right') - 1
            
            if return_pandas and base_time is not None:
                # Convert to DatetimeIndex
                refresh_times_idx = seconds2wall(refresh_times_array, base_time)
                return refresh_times_idx, (idx1, idx2)
            else:
                return refresh_times_array, (idx1, idx2)
        else:
            if return_pandas and base_time is not None:
                # Convert to DatetimeIndex
                refresh_times_idx = seconds2wall(refresh_times_array, base_time)
                return refresh_times_idx
            else:
                return refresh_times_array


def refresh_time_bivariate_prices(
    prices1: Union[np.ndarray, pd.Series],
    times1: Union[np.ndarray, pd.Series, pd.DatetimeIndex],
    prices2: Union[np.ndarray, pd.Series],
    times2: Union[np.ndarray, pd.Series, pd.DatetimeIndex],
    interpolation_method: str = 'previous',
    convert_to_seconds: bool = True,
    base_time: Optional[Union[str, pd.Timestamp]] = None,
    return_pandas: bool = False
) -> Union[
    Tuple[np.ndarray, np.ndarray],
    Tuple[pd.DataFrame, pd.DatetimeIndex]
]:
    """
    Synchronize bivariate price data using refresh times.
    
    This function computes refresh times for the given bivariate time series and
    aligns price data to these synchronized time points.
    
    Args:
        prices1: Price series for the first asset
        times1: Time points for the first asset
        prices2: Price series for the second asset
        times2: Time points for the second asset
        interpolation_method: Method for interpolating prices ('previous', 'linear', 'cubic')
        convert_to_seconds: Whether to convert all times to seconds for processing
        base_time: Base time for datetime conversion
        return_pandas: Whether to return pandas objects instead of numpy arrays
        
    Returns:
        Tuple of (synchronized_prices, refresh_times) where synchronized_prices
        is a 2D array or DataFrame with each column representing an asset
        
    Raises:
        ValueError: If inputs have invalid dimensions or if parameters are invalid
        TypeError: If inputs have incorrect types
        
    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from mfe.models.realized.refresh_time_bivariate import refresh_time_bivariate_prices
        >>> # Example with numeric times
        >>> prices1 = np.array([100, 101, 102, 103, 104])
        >>> times1 = np.array([1, 3, 5, 7, 9])
        >>> prices2 = np.array([50, 51, 52, 53, 54])
        >>> times2 = np.array([2, 4, 6, 8, 10])
        >>> synced_prices, refresh_times = refresh_time_bivariate_prices(
        ...     prices1, times1, prices2, times2
        ... )
        >>> refresh_times
        array([ 2.,  4.,  6.,  8., 10.])
        >>> synced_prices
        array([[100.,  50.],
               [101.,  51.],
               [102.,  52.],
               [103.,  53.],
               [104.,  54.]])
    """
    # Validate inputs
    prices1_array = np.asarray(prices1)
    prices2_array = np.asarray(prices2)
    
    if isinstance(times1, pd.DatetimeIndex):
        if len(prices1_array) != len(times1):
            raise ValueError(f"First price series length ({len(prices1_array)}) "
                            f"does not match time series length ({len(times1)})")
    else:
        times1_array = np.asarray(times1)
        if len(prices1_array) != len(times1_array):
            raise ValueError(f"First price series length ({len(prices1_array)}) "
                            f"does not match time series length ({len(times1_array)})")
    
    if isinstance(times2, pd.DatetimeIndex):
        if len(prices2_array) != len(times2):
            raise ValueError(f"Second price series length ({len(prices2_array)}) "
                            f"does not match time series length ({len(times2)})")
    else:
        times2_array = np.asarray(times2)
        if len(prices2_array) != len(times2_array):
            raise ValueError(f"Second price series length ({len(prices2_array)}) "
                            f"does not match time series length ({len(times2_array)})")
    
    # Compute refresh times
    refresh_times_result = refresh_time_bivariate(
        times1, 
        times2,
        convert_to_seconds=convert_to_seconds,
        base_time=base_time,
        return_indices=False,
        return_pandas=return_pandas
    )
    
    # Align price data to refresh times
    if return_pandas and isinstance(refresh_times_result, pd.DatetimeIndex):
        # Working with pandas objects
        refresh_times_idx = refresh_times_result
        
        # Create a DataFrame with aligned price series
        aligned_prices = pd.DataFrame(index=refresh_times_idx)
        
        # Process first asset
        if isinstance(prices1, pd.Series) and isinstance(prices1.index, pd.DatetimeIndex):
            # Series already has datetime index
            price_series1 = prices1
        elif isinstance(times1, pd.DatetimeIndex):
            # Create Series with DatetimeIndex
            price_series1 = pd.Series(prices1, index=times1)
        else:
            # Convert times to DatetimeIndex if possible
            try:
                datetime_idx = pd.DatetimeIndex(pd.to_datetime(times1))
                price_series1 = pd.Series(prices1, index=datetime_idx)
            except (ValueError, TypeError):
                if base_time is None:
                    raise ValueError("base_time must be provided to convert numeric times to datetime")
                # Convert numeric times to datetime
                datetime_idx = seconds2wall(np.asarray(times1), base_time)
                price_series1 = pd.Series(prices1, index=datetime_idx)
        
        # Process second asset
        if isinstance(prices2, pd.Series) and isinstance(prices2.index, pd.DatetimeIndex):
            # Series already has datetime index
            price_series2 = prices2
        elif isinstance(times2, pd.DatetimeIndex):
            # Create Series with DatetimeIndex
            price_series2 = pd.Series(prices2, index=times2)
        else:
            # Convert times to DatetimeIndex if possible
            try:
                datetime_idx = pd.DatetimeIndex(pd.to_datetime(times2))
                price_series2 = pd.Series(prices2, index=datetime_idx)
            except (ValueError, TypeError):
                if base_time is None:
                    raise ValueError("base_time must be provided to convert numeric times to datetime")
                # Convert numeric times to datetime
                datetime_idx = seconds2wall(np.asarray(times2), base_time)
                price_series2 = pd.Series(prices2, index=datetime_idx)
        
        # Align to refresh times
        if interpolation_method == 'previous':
            # Use asof to get the last value before each refresh time
            aligned_series1 = price_series1.reindex(
                refresh_times_idx, method='pad'
            )
            aligned_series2 = price_series2.reindex(
                refresh_times_idx, method='pad'
            )
        elif interpolation_method == 'linear':
            # Linear interpolation
            aligned_series1 = price_series1.reindex(
                refresh_times_idx, method='linear'
            )
            aligned_series2 = price_series2.reindex(
                refresh_times_idx, method='linear'
            )
        elif interpolation_method == 'cubic':
            # Cubic interpolation
            aligned_series1 = price_series1.reindex(
                refresh_times_idx).interpolate(method='cubic')
            aligned_series2 = price_series2.reindex(
                refresh_times_idx).interpolate(method='cubic')
        else:
            raise ValueError(f"Unrecognized interpolation method: {interpolation_method}. "
                            f"Supported methods are 'previous', 'linear', 'cubic'.")
        
        # Add to DataFrame
        aligned_prices['Asset_1'] = aligned_series1
        aligned_prices['Asset_2'] = aligned_series2
        
        return aligned_prices, refresh_times_idx
    else:
        # Working with numpy arrays
        refresh_times_array = refresh_times_result
        
        # Initialize aligned prices array
        aligned_prices = np.zeros((len(refresh_times_array), 2))
        
        # Process first asset
        if isinstance(times1, pd.DatetimeIndex):
            if convert_to_seconds:
                if base_time is None:
                    # Use the first time point as base time
                    base_time = times1[0]
                times1_array = wall2seconds(times1, pd.Timestamp(base_time).timestamp())
            else:
                # Convert refresh times to DatetimeIndex for alignment
                if not isinstance(refresh_times_array, pd.DatetimeIndex):
                    if base_time is None:
                        raise ValueError("base_time must be provided to convert refresh times to datetime")
                    refresh_times_idx = seconds2wall(refresh_times_array, base_time)
                    # Align using pandas
                    price_series = pd.Series(prices1_array, index=times1)
                    if interpolation_method == 'previous':
                        aligned_series = price_series.reindex(
                            refresh_times_idx, method='pad'
                        )
                    elif interpolation_method == 'linear':
                        aligned_series = price_series.reindex(
                            refresh_times_idx, method='linear'
                        )
                    elif interpolation_method == 'cubic':
                        aligned_series = price_series.reindex(
                            refresh_times_idx).interpolate(method='cubic')
                    else:
                        raise ValueError(f"Unrecognized interpolation method: {interpolation_method}")
                    
                    aligned_prices[:, 0] = aligned_series.values
                    
                    # Process second asset similarly
                    price_series = pd.Series(prices2_array, index=times2)
                    if interpolation_method == 'previous':
                        aligned_series = price_series.reindex(
                            refresh_times_idx, method='pad'
                        )
                    elif interpolation_method == 'linear':
                        aligned_series = price_series.reindex(
                            refresh_times_idx, method='linear'
                        )
                    elif interpolation_method == 'cubic':
                        aligned_series = price_series.reindex(
                            refresh_times_idx).interpolate(method='cubic')
                    
                    aligned_prices[:, 1] = aligned_series.values
                    
                    return aligned_prices, refresh_times_array
        else:
            times1_array = np.asarray(times1)
        
        # Process second asset
        if isinstance(times2, pd.DatetimeIndex):
            if convert_to_seconds:
                if base_time is None:
                    # Use the first time point as base time
                    base_time = times2[0]
                times2_array = wall2seconds(times2, pd.Timestamp(base_time).timestamp())
            else:
                times2_array = np.asarray(times2)
        else:
            times2_array = np.asarray(times2)
        
        # Align prices to refresh times
        aligned_prices[:, 0] = align_time(
            prices1_array, times1_array, refresh_times_array, interpolation_method
        )
        aligned_prices[:, 1] = align_time(
            prices2_array, times2_array, refresh_times_array, interpolation_method
        )
        
        return aligned_prices, refresh_times_array


def refresh_time_bivariate_returns(
    prices1: Union[np.ndarray, pd.Series],
    times1: Union[np.ndarray, pd.Series, pd.DatetimeIndex],
    prices2: Union[np.ndarray, pd.Series],
    times2: Union[np.ndarray, pd.Series, pd.DatetimeIndex],
    return_type: str = 'log',
    interpolation_method: str = 'previous',
    convert_to_seconds: bool = True,
    base_time: Optional[Union[str, pd.Timestamp]] = None,
    return_pandas: bool = False
) -> Union[
    Tuple[np.ndarray, np.ndarray],
    Tuple[pd.DataFrame, pd.DatetimeIndex]
]:
    """
    Compute synchronized returns using bivariate refresh times.
    
    This function computes refresh times for the given bivariate time series,
    aligns price data to these synchronized time points, and then
    calculates returns from the synchronized prices.
    
    Args:
        prices1: Price series for the first asset
        times1: Time points for the first asset
        prices2: Price series for the second asset
        times2: Time points for the second asset
        return_type: Type of returns to compute ('log', 'simple')
        interpolation_method: Method for interpolating prices ('previous', 'linear', 'cubic')
        convert_to_seconds: Whether to convert all times to seconds for processing
        base_time: Base time for datetime conversion
        return_pandas: Whether to return pandas objects instead of numpy arrays
        
    Returns:
        Tuple of (synchronized_returns, refresh_times) where synchronized_returns
        is a 2D array or DataFrame with each column representing an asset
        
    Raises:
        ValueError: If inputs have invalid dimensions or if parameters are invalid
        TypeError: If inputs have incorrect types
        
    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from mfe.models.realized.refresh_time_bivariate import refresh_time_bivariate_returns
        >>> # Example with numeric times
        >>> prices1 = np.array([100, 101, 102, 103, 104])
        >>> times1 = np.array([1, 3, 5, 7, 9])
        >>> prices2 = np.array([50, 51, 52, 53, 54])
        >>> times2 = np.array([2, 4, 6, 8, 10])
        >>> synced_returns, refresh_times = refresh_time_bivariate_returns(
        ...     prices1, times1, prices2, times2
        ... )
        >>> refresh_times
        array([ 2.,  4.,  6.,  8., 10.])
        >>> synced_returns  # First row is NaN because returns require two prices
        array([[      nan,       nan],
               [0.00995033, 0.01980263],
               [0.00985222, 0.01960784],
               [0.00975709, 0.01941748],
               [0.00966184, 0.01923077]])
    """
    # Get synchronized prices
    synced_prices, refresh_times = refresh_time_bivariate_prices(
        prices1, times1, prices2, times2, 
        interpolation_method=interpolation_method,
        convert_to_seconds=convert_to_seconds,
        base_time=base_time,
        return_pandas=return_pandas
    )
    
    # Compute returns
    if return_pandas and isinstance(synced_prices, pd.DataFrame):
        # Working with pandas objects
        if return_type.lower() == 'log':
            returns = np.log(synced_prices).diff()
        else:  # 'simple'
            returns = synced_prices.pct_change()
        
        return returns, refresh_times
    else:
        # Working with numpy arrays
        synced_prices_array = np.asarray(synced_prices)
        n_times, n_assets = synced_prices_array.shape
        
        # Initialize returns array
        returns = np.full((n_times, n_assets), np.nan)
        
        # Compute returns
        if return_type.lower() == 'log':
            # Log returns
            returns[1:] = np.diff(np.log(synced_prices_array), axis=0)
        else:  # 'simple'
            # Simple returns
            returns[1:] = np.diff(synced_prices_array, axis=0) / synced_prices_array[:-1]
        
        return returns, refresh_times


def visualize_refresh_time_bivariate(
    prices1: Union[np.ndarray, pd.Series],
    times1: Union[np.ndarray, pd.Series, pd.DatetimeIndex],
    prices2: Union[np.ndarray, pd.Series],
    times2: Union[np.ndarray, pd.Series, pd.DatetimeIndex],
    refresh_times: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None,
    asset_names: Optional[List[str]] = None,
    title: str = 'Bivariate Refresh Time Synchronization',
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    show_grid: bool = True,
    convert_to_seconds: bool = True,
    base_time: Optional[Union[str, pd.Timestamp]] = None
) -> Optional[Any]:
    """
    Create visualization of bivariate refresh time synchronization.
    
    This function creates a plot showing the original price series and
    the synchronized prices at refresh times for two assets.
    
    Args:
        prices1: Price series for the first asset
        times1: Time points for the first asset
        prices2: Price series for the second asset
        times2: Time points for the second asset
        refresh_times: Pre-computed refresh times (if None, they will be computed)
        asset_names: Names of assets for the legend
        title: Plot title
        figsize: Figure size as (width, height) in inches
        save_path: Path to save the figure (if None, figure is displayed)
        show_grid: Whether to show grid lines
        convert_to_seconds: Whether to convert all times to seconds for processing
        base_time: Base time for datetime conversion
        
    Returns:
        Matplotlib figure object if matplotlib is available, None otherwise
        
    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from mfe.models.realized.refresh_time_bivariate import visualize_refresh_time_bivariate
        >>> # Example with numeric times
        >>> prices1 = np.array([100, 101, 102, 103, 104])
        >>> times1 = np.array([1, 3, 5, 7, 9])
        >>> prices2 = np.array([50, 51, 52, 53, 54])
        >>> times2 = np.array([2, 4, 6, 8, 10])
        >>> fig = visualize_refresh_time_bivariate(prices1, times1, prices2, times2)
    """
    if not HAS_MATPLOTLIB:
        logger.warning("Matplotlib is not available. Cannot create visualization.")
        return None
    
    # Compute refresh times if not provided
    if refresh_times is None:
        refresh_times = refresh_time_bivariate(
            times1, 
            times2,
            convert_to_seconds=convert_to_seconds,
            base_time=base_time,
            return_indices=False,
            return_pandas=False
        )
    
    # Get synchronized prices
    synced_prices, _ = refresh_time_bivariate_prices(
        prices1, times1, prices2, times2, 
        interpolation_method='previous',
        convert_to_seconds=convert_to_seconds,
        base_time=base_time,
        return_pandas=False
    )
    
    # Determine if we're working with datetime
    is_datetime = False
    if isinstance(times1, pd.DatetimeIndex):
        is_datetime = True
    elif isinstance(times1, pd.Series) and pd.api.types.is_datetime64_dtype(times1):
        is_datetime = True
    elif isinstance(times1, (list, np.ndarray)) and len(times1) > 0:
        if isinstance(times1[0], (str, pd.Timestamp, np.datetime64)):
            is_datetime = True
    
    # Convert times to datetime for plotting if needed
    if is_datetime and not isinstance(refresh_times, pd.DatetimeIndex):
        if base_time is None:
            raise ValueError("base_time must be provided to convert refresh times to datetime")
        refresh_times_dt = seconds2wall(refresh_times, base_time)
    elif isinstance(refresh_times, pd.DatetimeIndex):
        refresh_times_dt = refresh_times
    else:
        # Use numeric times
        refresh_times_dt = None
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # Set asset names if not provided
    if asset_names is None:
        asset_names = ["Asset 1", "Asset 2"]
    
    # Plot first asset's original prices
    ax = axes[0]
    
    # Convert to arrays
    prices1_array = np.asarray(prices1)
    
    if is_datetime:
        # Convert to datetime for plotting
        if isinstance(times1, pd.DatetimeIndex):
            times1_dt = times1
        elif isinstance(times1, pd.Series) and pd.api.types.is_datetime64_dtype(times1):
            times1_dt = pd.DatetimeIndex(times1)
        else:
            try:
                times1_dt = pd.DatetimeIndex(pd.to_datetime(times1))
            except (ValueError, TypeError):
                if base_time is None:
                    raise ValueError("base_time must be provided to convert times to datetime")
                times1_dt = seconds2wall(np.asarray(times1), base_time)
        
        # Plot original prices
        ax.plot(times1_dt, prices1_array, 'o-', label=f'Original {asset_names[0]}', alpha=0.5)
        
        # Plot synchronized prices at refresh times
        if refresh_times_dt is not None:
            ax.plot(refresh_times_dt, synced_prices[:, 0], 's-', 
                    label=f'Synchronized {asset_names[0]}', linewidth=2)
    else:
        # Numeric times
        times1_array = np.asarray(times1)
        
        # Plot original prices
        ax.plot(times1_array, prices1_array, 'o-', label=f'Original {asset_names[0]}', alpha=0.5)
        
        # Plot synchronized prices at refresh times
        ax.plot(refresh_times, synced_prices[:, 0], 's-', 
                label=f'Synchronized {asset_names[0]}', linewidth=2)
    
    # Add legend and grid
    ax.legend(loc='best')
    ax.set_ylabel(f'{asset_names[0]} Price')
    if show_grid:
        ax.grid(True, alpha=0.3)
    
    # Plot second asset's original prices
    ax = axes[1]
    
    # Convert to arrays
    prices2_array = np.asarray(prices2)
    
    if is_datetime:
        # Convert to datetime for plotting
        if isinstance(times2, pd.DatetimeIndex):
            times2_dt = times2
        elif isinstance(times2, pd.Series) and pd.api.types.is_datetime64_dtype(times2):
            times2_dt = pd.DatetimeIndex(times2)
        else:
            try:
                times2_dt = pd.DatetimeIndex(pd.to_datetime(times2))
            except (ValueError, TypeError):
                if base_time is None:
                    raise ValueError("base_time must be provided to convert times to datetime")
                times2_dt = seconds2wall(np.asarray(times2), base_time)
        
        # Plot original prices
        ax.plot(times2_dt, prices2_array, 'o-', label=f'Original {asset_names[1]}', alpha=0.5)
        
        # Plot synchronized prices at refresh times
        if refresh_times_dt is not None:
            ax.plot(refresh_times_dt, synced_prices[:, 1], 's-', 
                    label=f'Synchronized {asset_names[1]}', linewidth=2)
    else:
        # Numeric times
        times2_array = np.asarray(times2)
        
        # Plot original prices
        ax.plot(times2_array, prices2_array, 'o-', label=f'Original {asset_names[1]}', alpha=0.5)
        
        # Plot synchronized prices at refresh times
        ax.plot(refresh_times, synced_prices[:, 1], 's-', 
                label=f'Synchronized {asset_names[1]}', linewidth=2)
    
    # Add legend and grid
    ax.legend(loc='best')
    ax.set_ylabel(f'{asset_names[1]} Price')
    if show_grid:
        ax.grid(True, alpha=0.3)
    
    # Plot refresh times in the bottom panel
    ax = axes[2]
    
    # Create a step function to visualize refresh time points
    if is_datetime and refresh_times_dt is not None:
        # Create a step function for refresh times
        for i, rt in enumerate(refresh_times_dt):
            ax.axvline(x=rt, color='r', linestyle='--', alpha=0.5)
        
        # Add markers at refresh times
        y_pos = np.ones(len(refresh_times_dt))
        ax.plot(refresh_times_dt, y_pos, 'ro', label='Refresh Times')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        fig.autofmt_xdate()
    else:
        # Create a step function for refresh times
        for i, rt in enumerate(refresh_times):
            ax.axvline(x=rt, color='r', linestyle='--', alpha=0.5)
        
        # Add markers at refresh times
        y_pos = np.ones(len(refresh_times))
        ax.plot(refresh_times, y_pos, 'ro', label='Refresh Times')
    
    # Set y-axis limits and hide y-ticks for the refresh time panel
    ax.set_ylim(0.5, 1.5)
    ax.set_yticks([])
    ax.set_ylabel('Refresh\nTimes')
    ax.legend(loc='best')
    if show_grid:
        ax.grid(True, alpha=0.3)
    
    # Set overall title and x-label
    fig.suptitle(title, fontsize=16)
    ax.set_xlabel('Time')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save or display the figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def refresh_time_bivariate_statistics(
    times1: Union[np.ndarray, pd.Series, pd.DatetimeIndex],
    times2: Union[np.ndarray, pd.Series, pd.DatetimeIndex],
    refresh_times: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None,
    convert_to_seconds: bool = True,
    base_time: Optional[Union[str, pd.Timestamp]] = None,
    return_pandas: bool = False
) -> Union[Dict[str, Any], pd.DataFrame]:
    """
    Compute statistics about bivariate refresh time sampling.
    
    This function analyzes the bivariate refresh time synchronization process,
    providing statistics about sampling frequency, data loss, and
    other relevant metrics.
    
    Args:
        times1: Time series for the first asset
        times2: Time series for the second asset
        refresh_times: Pre-computed refresh times (if None, they will be computed)
        convert_to_seconds: Whether to convert all times to seconds for processing
        base_time: Base time for datetime conversion
        return_pandas: Whether to return a pandas DataFrame instead of a dictionary
        
    Returns:
        Dictionary or DataFrame containing refresh time statistics
        
    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from mfe.models.realized.refresh_time_bivariate import refresh_time_bivariate_statistics
        >>> # Example with numeric times
        >>> times1 = np.array([1, 3, 5, 7, 9])
        >>> times2 = np.array([2, 4, 6, 8, 10])
        >>> stats = refresh_time_bivariate_statistics(times1, times2)
        >>> stats['total_observations']
        10
        >>> stats['refresh_time_count']
        5
        >>> stats['data_retention_pct']
        50.0
    """
    # Compute refresh times if not provided
    if refresh_times is None:
        refresh_times = refresh_time_bivariate(
            times1, 
            times2,
            convert_to_seconds=convert_to_seconds,
            base_time=base_time,
            return_indices=False,
            return_pandas=False
        )
    
    # Convert refresh_times to numpy array if it's a DatetimeIndex
    if isinstance(refresh_times, pd.DatetimeIndex):
        if convert_to_seconds:
            if base_time is None:
                # Use the first time point as base time
                base_time = refresh_times[0]
            refresh_times_array = wall2seconds(refresh_times, pd.Timestamp(base_time).timestamp())
        else:
            # Keep as DatetimeIndex for calculations
            refresh_times_array = np.array(refresh_times.astype(np.int64) / 1e9)
    else:
        refresh_times_array = np.asarray(refresh_times)
    
    # Process first time series
    if isinstance(times1, pd.DatetimeIndex):
        if convert_to_seconds:
            if base_time is None:
                # Use the first time point as base time
                base_time = times1[0]
            times1_array = wall2seconds(times1, pd.Timestamp(base_time).timestamp())
        else:
            # Convert to seconds for calculations
            times1_array = np.array(times1.astype(np.int64) / 1e9)
    elif isinstance(times1, pd.Series) and pd.api.types.is_datetime64_dtype(times1):
        datetime_idx = pd.DatetimeIndex(times1)
        if convert_to_seconds:
            if base_time is None:
                # Use the first time point as base time
                base_time = datetime_idx[0]
            times1_array = wall2seconds(datetime_idx, pd.Timestamp(base_time).timestamp())
        else:
            # Convert to seconds for calculations
            times1_array = np.array(datetime_idx.astype(np.int64) / 1e9)
    else:
        # Assume numpy array or list of numeric values
        times1_array = np.asarray(times1, dtype=float)
    
    # Process second time series
    if isinstance(times2, pd.DatetimeIndex):
        if convert_to_seconds:
            if base_time is None:
                # Use the first time point as base time
                base_time = times2[0]
            times2_array = wall2seconds(times2, pd.Timestamp(base_time).timestamp())
        else:
            # Convert to seconds for calculations
            times2_array = np.array(times2.astype(np.int64) / 1e9)
    elif isinstance(times2, pd.Series) and pd.api.types.is_datetime64_dtype(times2):
        datetime_idx = pd.DatetimeIndex(times2)
        if convert_to_seconds:
            if base_time is None:
                # Use the first time point as base time
                base_time = datetime_idx[0]
            times2_array = wall2seconds(datetime_idx, pd.Timestamp(base_time).timestamp())
        else:
            # Convert to seconds for calculations
            times2_array = np.array(datetime_idx.astype(np.int64) / 1e9)
    else:
        # Assume numpy array or list of numeric values
        times2_array = np.asarray(times2, dtype=float)
    
    # Compute statistics
    stats = {}
    
    # Total number of observations across both assets
    total_obs = len(times1_array) + len(times2_array)
    stats['total_observations'] = total_obs
    
    # Number of refresh times
    refresh_count = len(refresh_times_array)
    stats['refresh_time_count'] = refresh_count
    
    # Data retention percentage
    retention_pct = (refresh_count * 2 / total_obs) * 100
    stats['data_retention_pct'] = retention_pct
    
    # Average time between refresh times
    if refresh_count > 1:
        avg_interval = np.mean(np.diff(refresh_times_array))
        stats['avg_refresh_interval'] = avg_interval
    else:
        stats['avg_refresh_interval'] = np.nan
    
    # Standard deviation of intervals
    if refresh_count > 2:
        std_interval = np.std(np.diff(refresh_times_array))
        stats['std_refresh_interval'] = std_interval
    else:
        stats['std_refresh_interval'] = np.nan
    
    # Min and max intervals
    if refresh_count > 1:
        min_interval = np.min(np.diff(refresh_times_array))
        max_interval = np.max(np.diff(refresh_times_array))
        stats['min_refresh_interval'] = min_interval
        stats['max_refresh_interval'] = max_interval
    else:
        stats['min_refresh_interval'] = np.nan
        stats['max_refresh_interval'] = np.nan
    
    # Statistics for each asset
    asset_stats = []
    
    # First asset
    asset_stat1 = {}
    asset_stat1['asset_index'] = 0
    asset_stat1['asset_name'] = 'Asset_1'
    asset_stat1['original_observations'] = len(times1_array)
    
    # Compute average interval in original data
    if len(times1_array) > 1:
        asset_stat1['avg_original_interval'] = np.mean(np.diff(times1_array))
    else:
        asset_stat1['avg_original_interval'] = np.nan
    
    # Compute data retention for this asset
    asset_stat1['retention_pct'] = (refresh_count / len(times1_array)) * 100
    
    asset_stats.append(asset_stat1)
    
    # Second asset
    asset_stat2 = {}
    asset_stat2['asset_index'] = 1
    asset_stat2['asset_name'] = 'Asset_2'
    asset_stat2['original_observations'] = len(times2_array)
    
    # Compute average interval in original data
    if len(times2_array) > 1:
        asset_stat2['avg_original_interval'] = np.mean(np.diff(times2_array))
    else:
        asset_stat2['avg_original_interval'] = np.nan
    
    # Compute data retention for this asset
    asset_stat2['retention_pct'] = (refresh_count / len(times2_array)) * 100
    
    asset_stats.append(asset_stat2)
    
    stats['asset_statistics'] = asset_stats
    
    # Return as pandas DataFrame if requested
    if return_pandas:
        # Create main stats DataFrame
        main_stats = pd.DataFrame({k: [v] for k, v in stats.items() 
                                  if k != 'asset_statistics'})
        
        # Create asset stats DataFrame
        asset_df = pd.DataFrame(asset_stats)
        
        # Return both DataFrames
        return pd.concat([main_stats, asset_df], keys=['overall', 'by_asset'])
    else:
        return stats
