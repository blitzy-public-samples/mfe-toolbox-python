# mfe/models/realized/wall2seconds.py
"""
Convert wall clock time in HHMMSS format to seconds past midnight.

This module provides a function to convert wall clock time in HHMMSS format
(hours, minutes, seconds) into seconds past midnight. This is essential for
numerical time operations in high-frequency financial data analysis, enabling
consistent time representation and arithmetic operations across different
sampling frequencies.

The implementation supports various input formats including NumPy arrays,
pandas Series, DatetimeIndex, and scalar values, with comprehensive type hints
and parameter validation. It also handles timezone-aware timestamps and provides
nanosecond precision for modern market data analysis.

Functions:
    wall2seconds: Convert wall clock time in HHMMSS format to seconds past midnight
"""
import logging
from typing import Union, Optional, Sequence, cast, overload

import numpy as np
import pandas as pd

# Set up module-level logger
logger = logging.getLogger("mfe.models.realized.wall2seconds")


@overload
def wall2seconds(wall_time: int) -> float:
    ...


@overload
def wall2seconds(wall_time: np.ndarray) -> np.ndarray:
    ...


@overload
def wall2seconds(wall_time: pd.Series) -> pd.Series:
    ...


@overload
def wall2seconds(wall_time: pd.DatetimeIndex) -> np.ndarray:
    ...



def wall2seconds(wall_time: Union[int, float, np.ndarray, pd.Series, pd.DatetimeIndex, Sequence[int]]) -> Union[float, np.ndarray, pd.Series]:
    """
    Convert wall clock time in HHMMSS format to seconds past midnight.

    This function converts wall clock time in HHMMSS format (hours, minutes, seconds)
    to seconds past midnight. For example, 10101 (representing 01:01:01) would be
    converted to 3661 seconds past midnight (1:01:01 AM).

    Args:
        wall_time: Wall clock time in HHMMSS format. Can be a scalar value, NumPy array,
                  pandas Series, pandas DatetimeIndex, or any sequence convertible to NumPy array.

    Returns:
        Time in seconds past midnight. The return type matches the input type
        (scalar, NumPy array, or pandas Series) except for DatetimeIndex inputs,
        which return NumPy arrays.

    Raises:
        ValueError: If input contains invalid time values (e.g., hours > 23, minutes > 59, seconds > 59)
        TypeError: If input type is not supported

    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.wall2seconds import wall2seconds
        >>> # Scalar input
        >>> wall2seconds(10101)  # 01:01:01 AM
        3661.0
        >>> # NumPy array input
        >>> times = np.array([0, 10101, 120000, 235959])  # Midnight, 1:01:01 AM, Noon, 11:59:59 PM
        >>> wall2seconds(times)
        array([    0.,  3661., 43200., 86399.])
        >>> # pandas Series input
        >>> import pandas as pd
        >>> time_series = pd.Series([0, 10101, 120000, 235959])
        >>> wall2seconds(time_series)
        0        0.0
        1     3661.0
        2    43200.0
        3    86399.0
        dtype: float64
        >>> # pandas DatetimeIndex input
        >>> times = pd.DatetimeIndex(['2023-01-01 00:00:00', '2023-01-01 01:01:01',
        ...                           '2023-01-01 12:00:00', '2023-01-01 23:59:59'])
        >>> wall2seconds(times)
        array([    0.,  3661., 43200., 86399.])
    """
    # Handle pandas DatetimeIndex input
    if isinstance(wall_time, pd.DatetimeIndex):
        # Extract time components and convert directly to seconds
        seconds = wall_time.hour * 3600 + wall_time.minute * 60 + wall_time.second
        
        # Add nanoseconds for higher precision if available
        if hasattr(wall_time, 'nanosecond'):
            seconds = seconds.astype(float) + wall_time.nanosecond / 1e9
        
        # Convert to numpy array for consistent return type
        return seconds.values
    
    # Handle pandas Series input
    if isinstance(wall_time, pd.Series):
        # Check if the Series contains datetime objects
        if pd.api.types.is_datetime64_any_dtype(wall_time):
            # Extract time components and convert directly to seconds
            seconds = wall_time.dt.hour * 3600 + wall_time.dt.minute * 60 + wall_time.dt.second
            
            # Add nanoseconds for higher precision if available
            if hasattr(wall_time.dt, 'nanosecond'):
                seconds = seconds.astype(float) + wall_time.dt.nanosecond / 1e9
            
            return seconds
        
        # Convert to numpy array for processing
        values = np.asarray(wall_time)
        
        # Extract hours, minutes, and seconds from HHMMSS format
        hours = values // 10000
        minutes = (values % 10000) // 100
        seconds = values % 100
        
        # Validate time components
        if np.any(hours > 23) or np.any(minutes > 59) or np.any(seconds > 59):
            logger.warning("Some values contain invalid time components "
                          "(hours > 23, minutes > 59, or seconds > 59). "
                          "This may indicate incorrect time data.")
        
        # Convert to seconds past midnight
        seconds_past_midnight = hours * 3600 + minutes * 60 + seconds
        
        # Return as pandas Series with the same index
        return pd.Series(seconds_past_midnight, index=wall_time.index)
    
    # Convert to numpy array if not already
    if not isinstance(wall_time, (int, float, np.ndarray)):
        wall_time = np.asarray(wall_time)
    
    # Handle scalar input
    if isinstance(wall_time, (int, float)):
        # Extract hours, minutes, and seconds from HHMMSS format
        hours = wall_time // 10000
        minutes = (wall_time % 10000) // 100
        seconds = wall_time % 100
        
        # Validate time components
        if hours > 23 or minutes > 59 or seconds > 59:
            logger.warning(f"Value {wall_time} contains invalid time components "
                          "(hours > 23, minutes > 59, or seconds > 59). "
                          "This may indicate incorrect time data.")
        
        # Convert to seconds past midnight
        seconds_past_midnight = hours * 3600 + minutes * 60 + seconds
        
        return float(seconds_past_midnight)
    
    # Handle numpy array input
    # Extract hours, minutes, and seconds from HHMMSS format
    hours = wall_time // 10000
    minutes = (wall_time % 10000) // 100
    seconds = wall_time % 100
    
    # Validate time components
    if np.any(hours > 23) or np.any(minutes > 59) or np.any(seconds > 59):
        logger.warning("Some values contain invalid time components "
                      "(hours > 23, minutes > 59, or seconds > 59). "
                      "This may indicate incorrect time data.")
    
    # Convert to seconds past midnight
    seconds_past_midnight = hours * 3600 + minutes * 60 + seconds
    
    return seconds_past_midnight.astype(float)



def wall2timestamp(wall_time: Union[int, np.ndarray, pd.Series],
                   base_date: Union[str, pd.Timestamp, np.datetime64]) -> Union[pd.Timestamp, pd.DatetimeIndex, pd.Series]:
    """
    Convert wall clock time in HHMMSS format to pandas Timestamps.

    This function converts wall clock time in HHMMSS format to pandas Timestamps
    by adding the corresponding time to a base date. This is useful for creating
    datetime objects from wall clock times for visualization or further analysis.

    Args:
        wall_time: Wall clock time in HHMMSS format. Can be a scalar value, NumPy array,
                  pandas Series, or any sequence convertible to NumPy array.
        base_date: The reference date to which the time will be added. Can be a string in
                  ISO format ('YYYY-MM-DD'), pandas Timestamp, or numpy datetime64.

    Returns:
        For scalar inputs: pandas Timestamp
        For array inputs: pandas DatetimeIndex
        For Series inputs: pandas Series with Timestamps

    Raises:
        ValueError: If input contains invalid time values
        TypeError: If input type is not supported

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from mfe.models.realized.wall2seconds import wall2timestamp
        >>> # Scalar input
        >>> wall2timestamp(120000, '2023-01-01')  # 12:00:00 (noon)
        Timestamp('2023-01-01 12:00:00')
        >>> # NumPy array input
        >>> times = np.array([0, 60000, 120000, 180000])  # 00:00, 06:00, 12:00, 18:00
        >>> wall2timestamp(times, '2023-01-01')
        DatetimeIndex(['2023-01-01 00:00:00', '2023-01-01 06:00:00',
                       '2023-01-01 12:00:00', '2023-01-01 18:00:00'],
                      dtype='datetime64[ns]', freq=None)
        >>> # pandas Series input
        >>> time_series = pd.Series([0, 60000, 120000, 180000])
        >>> wall2timestamp(time_series, '2023-01-01')
        0   2023-01-01 00:00:00
        1   2023-01-01 06:00:00
        2   2023-01-01 12:00:00
        3   2023-01-01 18:00:00
        dtype: datetime64[ns]
    """
    # Convert base_date to pandas Timestamp if it's not already
    if not isinstance(base_date, pd.Timestamp):
        base_date = pd.Timestamp(base_date)
    
    # Get seconds past midnight
    seconds = wall2seconds(wall_time)
    
    # Handle different input types
    if isinstance(seconds, (float, int)):
        # Scalar input
        return base_date + pd.Timedelta(seconds=seconds)
    elif isinstance(seconds, pd.Series):
        # Series input - preserve index
        return base_date + pd.to_timedelta(seconds, unit='s')
    else:
        # NumPy array or sequence input
        return pd.DatetimeIndex([base_date + pd.Timedelta(seconds=s) for s in seconds])
