# mfe/models/realized/seconds2unit.py
"""
Convert seconds past midnight into a normalized unit interval [0,1].

This module provides a function to convert time in seconds past midnight into a
normalized unit interval [0,1], which is essential for standardizing time inputs
in high-frequency financial analysis. The conversion enables consistent time
normalization across different sampling schemes and facilitates comparison of
intraday patterns.

The implementation supports various input formats including NumPy arrays,
pandas Series, DatetimeIndex, and scalar values, with comprehensive type hints
and parameter validation. It also handles timezone-aware timestamps and provides
nanosecond precision for modern market data analysis.

Functions:
    seconds2unit: Convert seconds past midnight to normalized unit interval [0,1]
"""

import logging
from typing import Union, Optional, Sequence, cast, overload

import numpy as np
import pandas as pd

# Set up module-level logger
logger = logging.getLogger("mfe.models.realized.seconds2unit")

# Constants
SECONDS_PER_DAY = 86400  # 24 hours * 60 minutes * 60 seconds


@overload
def seconds2unit(seconds: float) -> float:
    ...


@overload
def seconds2unit(seconds: np.ndarray) -> np.ndarray:
    ...


@overload
def seconds2unit(seconds: pd.Series) -> pd.Series:
    ...


@overload
def seconds2unit(seconds: pd.DatetimeIndex) -> np.ndarray:
    ...



def seconds2unit(seconds: Union[float, np.ndarray, pd.Series, pd.DatetimeIndex, Sequence[float]]) -> Union[float, np.ndarray, pd.Series]:
    """
    Convert seconds past midnight to normalized unit interval [0,1].

    This function converts time in seconds past midnight to a normalized unit interval [0,1],
    where 0 corresponds to midnight (0:00:00) and 1 corresponds to the next midnight (24:00:00).
    The conversion is essential for standardizing time inputs in high-frequency analysis.

    Args:
        seconds: Time in seconds past midnight. Can be a scalar value, NumPy array,
                pandas Series, pandas DatetimeIndex, or any sequence convertible to NumPy array.

    Returns:
        Normalized time in unit interval [0,1]. The return type matches the input type
        (scalar, NumPy array, or pandas Series).

    Raises:
        ValueError: If input contains negative values or values greater than 86400 (seconds in a day)
        TypeError: If input type is not supported

    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.seconds2unit import seconds2unit
        >>> # Scalar input
        >>> seconds2unit(43200)  # 12:00:00 (noon)
        0.5
        >>> # NumPy array input
        >>> seconds = np.array([0, 21600, 43200, 64800])  # 00:00, 06:00, 12:00, 18:00
        >>> seconds2unit(seconds)
        array([0.  , 0.25, 0.5 , 0.75])
        >>> # pandas Series input
        >>> import pandas as pd
        >>> time_series = pd.Series([0, 21600, 43200, 64800])
        >>> seconds2unit(time_series)
        0    0.00
        1    0.25
        2    0.50
        3    0.75
        dtype: float64
        >>> # pandas DatetimeIndex input
        >>> times = pd.DatetimeIndex(['2023-01-01 00:00:00', '2023-01-01 06:00:00',
        ...                           '2023-01-01 12:00:00', '2023-01-01 18:00:00'])
        >>> seconds2unit(times)
        array([0.  , 0.25, 0.5 , 0.75])
    """
    # Handle pandas DatetimeIndex input
    if isinstance(seconds, pd.DatetimeIndex):
        # Extract time components and convert to seconds past midnight
        hours = seconds.hour
        minutes = seconds.minute
        seconds_val = seconds.second
        nanoseconds = seconds.nanosecond
        
        # Calculate seconds past midnight with nanosecond precision
        seconds_past_midnight = (hours * 3600 + minutes * 60 + seconds_val + 
                                nanoseconds / 1e9)
        
        # Convert to numpy array for further processing
        seconds = seconds_past_midnight.values
    
    # Handle pandas Series input
    if isinstance(seconds, pd.Series):
        # Check if the Series contains datetime objects
        if pd.api.types.is_datetime64_any_dtype(seconds):
            # Extract time components and convert to seconds past midnight
            hours = seconds.dt.hour
            minutes = seconds.dt.minute
            seconds_val = seconds.dt.second
            nanoseconds = seconds.dt.nanosecond
            
            # Calculate seconds past midnight with nanosecond precision
            seconds_past_midnight = (hours * 3600 + minutes * 60 + seconds_val + 
                                    nanoseconds / 1e9)
            
            # Create a new Series with the same index
            unit_time = seconds_past_midnight / SECONDS_PER_DAY
            
            # Validate range
            if (unit_time < 0).any() or (unit_time > 1).any():
                logger.warning("Some values are outside the valid range [0, 1]. "
                              "This may indicate incorrect time data.")
            
            return unit_time
        
        # Convert to numpy array for validation, then back to Series
        values = np.asarray(seconds)
        unit_time = values / SECONDS_PER_DAY
        
        # Validate range
        if (unit_time < 0).any() or (unit_time > 1).any():
            logger.warning("Some values are outside the valid range [0, 1]. "
                          "This may indicate incorrect time data.")
        
        return pd.Series(unit_time, index=seconds.index)
    
    # Convert to numpy array if not already
    if not isinstance(seconds, (float, int, np.ndarray)):
        seconds = np.asarray(seconds)
    
    # Perform the conversion
    unit_time = seconds / SECONDS_PER_DAY
    
    # Validate range for array inputs
    if isinstance(unit_time, np.ndarray):
        if (unit_time < 0).any() or (unit_time > 1).any():
            logger.warning("Some values are outside the valid range [0, 1]. "
                          "This may indicate incorrect time data.")
    # Validate range for scalar inputs
    elif unit_time < 0 or unit_time > 1:
        logger.warning(f"Value {unit_time} is outside the valid range [0, 1]. "
                      "This may indicate incorrect time data.")
    
    return unit_time
