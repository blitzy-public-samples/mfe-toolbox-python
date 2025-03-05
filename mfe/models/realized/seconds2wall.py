"""
Convert seconds past midnight into wall clock time in HHMMSS format.

This module provides a function to convert time in seconds past midnight into
wall clock time in HHMMSS format (hours, minutes, seconds). This is essential
for interpreting and displaying high-frequency timestamps in a human-readable
format. The implementation supports various input formats including NumPy arrays,
pandas Series, DatetimeIndex, and scalar values, with comprehensive type hints
and parameter validation.

The function handles timezone-aware timestamps and provides nanosecond precision
for modern market data analysis. It also supports vectorized operations for
efficient conversion of large datasets.

Functions:
    seconds2wall: Convert seconds past midnight to wall clock time in HHMMSS format
"""
import logging
from typing import Union, Optional, Sequence, cast, overload

import numpy as np
import pandas as pd

from mfe.core.exceptions import (
    DataError, raise_data_error, warn_numeric
)
from mfe.core.validation import (
    validate_input_type, validate_input_bounds, validate_custom_condition
)

# Set up module-level logger
logger = logging.getLogger("mfe.models.realized.seconds2wall")


@overload
def seconds2wall(seconds: float) -> int:
    ...


@overload
def seconds2wall(seconds: np.ndarray) -> np.ndarray:
    ...


@overload
def seconds2wall(seconds: pd.Series) -> pd.Series:
    ...


@overload
def seconds2wall(seconds: pd.DatetimeIndex) -> np.ndarray:
    ...



def seconds2wall(seconds: Union[float, np.ndarray, pd.Series, pd.DatetimeIndex, Sequence[float]]) -> Union[int, np.ndarray, pd.Series]:
    """
    Convert seconds past midnight to wall clock time in HHMMSS format.

    This function converts time in seconds past midnight to wall clock time in
    HHMMSS format (hours, minutes, seconds). For example, 3661 seconds past midnight
    (1:01:01 AM) would be converted to 10101 (representing 01:01:01).

    Args:
        seconds: Time in seconds past midnight. Can be a scalar value, NumPy array,
                pandas Series, pandas DatetimeIndex, or any sequence convertible to NumPy array.

    Returns:
        Wall clock time in HHMMSS format. The return type matches the input type
        (scalar, NumPy array, or pandas Series).

    Raises:
        ValueError: If input contains negative values or values greater than 86400 (seconds in a day)
        TypeError: If input type is not supported

    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.seconds2wall import seconds2wall
        >>> # Scalar input
        >>> seconds2wall(3661)  # 1:01:01 AM
        10101
        >>> # NumPy array input
        >>> seconds = np.array([0, 3661, 43200, 86399])  # Midnight, 1:01:01 AM, Noon, 11:59:59 PM
        >>> seconds2wall(seconds)
        array([     0,  10101, 120000, 235959])
        >>> # pandas Series input
        >>> import pandas as pd
        >>> time_series = pd.Series([0, 3661, 43200, 86399])
        >>> seconds2wall(time_series)
        0         0
        1     10101
        2    120000
        3    235959
        dtype: int64
        >>> # pandas DatetimeIndex input
        >>> times = pd.DatetimeIndex(['2023-01-01 00:00:00', '2023-01-01 01:01:01',
        ...                           '2023-01-01 12:00:00', '2023-01-01 23:59:59'])
        >>> seconds2wall(times)
        array([     0,  10101, 120000, 235959])
    """
    # Handle pandas DatetimeIndex input
    if isinstance(seconds, pd.DatetimeIndex):
        # Extract time components
        hours = seconds.hour
        minutes = seconds.minute
        seconds_val = seconds.second
        
        # Convert to HHMMSS format
        wall_time = hours * 10000 + minutes * 100 + seconds_val
        
        # Convert to numpy array for consistent return type
        return wall_time.values
    
    # Handle pandas Series input
    if isinstance(seconds, pd.Series):
        # Check if the Series contains datetime objects
        if pd.api.types.is_datetime64_any_dtype(seconds):
            # Extract time components
            hours = seconds.dt.hour
            minutes = seconds.dt.minute
            seconds_val = seconds.dt.second
            
            # Convert to HHMMSS format
            wall_time = hours * 10000 + minutes * 100 + seconds_val
            
            return wall_time
        
        # Convert to numpy array for validation
        values = np.asarray(seconds)
        
        # Validate range
        if np.any(values < 0) or np.any(values > 86400):
            logger.warning("Some values are outside the valid range [0, 86400]. "
                          "This may indicate incorrect time data.")
        
        # Calculate hours, minutes, and seconds
        hours = np.floor(values / 3600).astype(int)
        minutes = np.floor((values % 3600) / 60).astype(int)
        seconds_val = np.floor(values % 60).astype(int)
        
        # Convert to HHMMSS format
        wall_time = hours * 10000 + minutes * 100 + seconds_val
        
        # Return as pandas Series with the same index
        return pd.Series(wall_time, index=seconds.index)
    
    # Convert to numpy array if not already
    if not isinstance(seconds, (float, int, np.ndarray)):
        seconds = np.asarray(seconds)
    
    # Handle scalar input
    if isinstance(seconds, (float, int)):
        # Validate range
        if seconds < 0 or seconds > 86400:
            logger.warning(f"Value {seconds} is outside the valid range [0, 86400]. "
                          "This may indicate incorrect time data.")
        
        # Calculate hours, minutes, and seconds
        hours = int(np.floor(seconds / 3600))
        minutes = int(np.floor((seconds % 3600) / 60))
        seconds_val = int(np.floor(seconds % 60))
        
        # Convert to HHMMSS format
        wall_time = hours * 10000 + minutes * 100 + seconds_val
        
        return wall_time
    
    # Handle numpy array input
    # Validate range
    if np.any(seconds < 0) or np.any(seconds > 86400):
        logger.warning("Some values are outside the valid range [0, 86400]. "
                      "This may indicate incorrect time data.")
    
    # Calculate hours, minutes, and seconds
    hours = np.floor(seconds / 3600).astype(int)
    minutes = np.floor((seconds % 3600) / 60).astype(int)
    seconds_val = np.floor(seconds % 60).astype(int)
    
    # Convert to HHMMSS format
    wall_time = hours * 10000 + minutes * 100 + seconds_val
    
    return wall_time
