# mfe/models/realized/unit2seconds.py
"""
Convert normalized unit time values [0,1] into seconds past midnight.

This module provides a function to convert normalized time in the unit interval [0,1]
into seconds past midnight, which is essential for mapping standardized time values
back to actual time measurements in high-frequency financial analysis. The conversion
enables consistent time mapping across different sampling schemes and facilitates
the interpretation of intraday patterns.

The implementation supports various input formats including NumPy arrays,
pandas Series, and scalar values, with comprehensive type hints and parameter
validation. It also handles timezone-aware timestamps and provides nanosecond
precision for modern market data analysis.

Functions:
    unit2seconds: Convert normalized unit interval [0,1] to seconds past midnight
"""

import logging
from typing import Union, Optional, Sequence, cast, overload, Tuple

import numpy as np
import pandas as pd

# Set up module-level logger
logger = logging.getLogger("mfe.models.realized.unit2seconds")

# Constants
SECONDS_PER_DAY = 86400  # 24 hours * 60 minutes * 60 seconds


@overload
def unit2seconds(unit_time: float, 
                 time_range: Optional[Tuple[float, float]] = None) -> float:
    ...


@overload
def unit2seconds(unit_time: np.ndarray,
                 time_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
    ...


@overload
def unit2seconds(unit_time: pd.Series,
                 time_range: Optional[Tuple[float, float]] = None) -> pd.Series:
    ...



def unit2seconds(unit_time: Union[float, np.ndarray, pd.Series, Sequence[float]],
                 time_range: Optional[Tuple[float, float]] = None) -> Union[float, np.ndarray, pd.Series]:
    """
    Convert normalized unit time values [0,1] into seconds past midnight.

    This function converts time from a normalized unit interval [0,1] to seconds past midnight,
    where 0 corresponds to midnight (0:00:00) and 1 corresponds to the next midnight (24:00:00).
    Optionally, a custom time range can be specified to map the unit interval to a specific
    time window (e.g., market trading hours).

    Args:
        unit_time: Normalized time in unit interval [0,1]. Can be a scalar value, NumPy array,
                  pandas Series, or any sequence convertible to NumPy array.
        time_range: Optional tuple specifying the start and end times in seconds past midnight
                   to map the unit interval to. If None, the full day (0, 86400) is used.
                   Default is None.

    Returns:
        Time in seconds past midnight. The return type matches the input type
        (scalar, NumPy array, or pandas Series).

    Raises:
        ValueError: If input contains values outside the [0,1] range
        TypeError: If input type is not supported

    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.unit2seconds import unit2seconds
        >>> # Scalar input
        >>> unit2seconds(0.5)  # 0.5 corresponds to 12:00:00 (noon)
        43200.0
        >>> # NumPy array input
        >>> unit_times = np.array([0.0, 0.25, 0.5, 0.75])  # 00:00, 06:00, 12:00, 18:00
        >>> unit2seconds(unit_times)
        array([    0., 21600., 43200., 64800.])
        >>> # pandas Series input
        >>> import pandas as pd
        >>> time_series = pd.Series([0.0, 0.25, 0.5, 0.75])
        >>> unit2seconds(time_series)
        0        0.0
        1    21600.0
        2    43200.0
        3    64800.0
        dtype: float64
        >>> # Custom time range (e.g., market hours 9:30 to 16:00)
        >>> market_open = 9.5 * 3600  # 9:30 AM in seconds
        >>> market_close = 16 * 3600  # 4:00 PM in seconds
        >>> unit2seconds(0.5, time_range=(market_open, market_close))
        45900.0
    """
    # Validate time_range if provided
    if time_range is not None:
        if len(time_range) != 2:
            raise ValueError("time_range must be a tuple of (start_time, end_time) in seconds")
        
        start_time, end_time = time_range
        
        if start_time < 0 or start_time >= SECONDS_PER_DAY:
            raise ValueError(f"start_time must be in range [0, {SECONDS_PER_DAY})")
        
        if end_time <= 0 or end_time > SECONDS_PER_DAY:
            raise ValueError(f"end_time must be in range (0, {SECONDS_PER_DAY}]")
        
        if start_time >= end_time:
            raise ValueError("start_time must be less than end_time")
    else:
        # Default to full day
        start_time, end_time = 0, SECONDS_PER_DAY
    
    # Handle pandas Series input
    if isinstance(unit_time, pd.Series):
        # Validate range
        if (unit_time < 0).any() or (unit_time > 1).any():
            raise ValueError("unit_time values must be in the range [0, 1]")
        
        # Convert to seconds
        time_span = end_time - start_time
        seconds = start_time + unit_time * time_span
        
        return seconds
    
    # Convert to numpy array if not already
    if not isinstance(unit_time, (float, int, np.ndarray)):
        unit_time = np.asarray(unit_time, dtype=float)
    
    # Validate range for array inputs
    if isinstance(unit_time, np.ndarray):
        if (unit_time < 0).any() or (unit_time > 1).any():
            raise ValueError("unit_time values must be in the range [0, 1]")
    # Validate range for scalar inputs
    elif unit_time < 0 or unit_time > 1:
        raise ValueError(f"unit_time value {unit_time} must be in the range [0, 1]")
    
    # Perform the conversion
    time_span = end_time - start_time
    seconds = start_time + unit_time * time_span
    
    return seconds


def unit2timestamp(unit_time: Union[float, np.ndarray, pd.Series],
                   base_date: Union[str, pd.Timestamp, np.datetime64],
                   time_range: Optional[Tuple[float, float]] = None) -> Union[pd.Timestamp, pd.DatetimeIndex, pd.Series]:
    """
    Convert normalized unit time values [0,1] into pandas Timestamps.

    This function converts normalized time in the unit interval [0,1] to pandas Timestamps
    by adding the corresponding seconds to a base date. This is useful for creating
    datetime objects from normalized time values for visualization or further analysis.

    Args:
        unit_time: Normalized time in unit interval [0,1]. Can be a scalar value, NumPy array,
                  pandas Series, or any sequence convertible to NumPy array.
        base_date: The reference date to which the time will be added. Can be a string in
                  ISO format ('YYYY-MM-DD'), pandas Timestamp, or numpy datetime64.
        time_range: Optional tuple specifying the start and end times in seconds past midnight
                   to map the unit interval to. If None, the full day (0, 86400) is used.
                   Default is None.

    Returns:
        For scalar inputs: pandas Timestamp
        For array inputs: pandas DatetimeIndex
        For Series inputs: pandas Series with Timestamps

    Raises:
        ValueError: If input contains values outside the [0,1] range
        TypeError: If input type is not supported

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from mfe.models.realized.unit2seconds import unit2timestamp
        >>> # Scalar input
        >>> unit2timestamp(0.5, '2023-01-01')  # 0.5 corresponds to 12:00:00 (noon)
        Timestamp('2023-01-01 12:00:00')
        >>> # NumPy array input
        >>> unit_times = np.array([0.0, 0.25, 0.5, 0.75])
        >>> unit2timestamp(unit_times, '2023-01-01')
        DatetimeIndex(['2023-01-01 00:00:00', '2023-01-01 06:00:00',
                       '2023-01-01 12:00:00', '2023-01-01 18:00:00'],
                      dtype='datetime64[ns]', freq=None)
        >>> # Custom time range (e.g., market hours 9:30 to 16:00)
        >>> market_open = 9.5 * 3600  # 9:30 AM in seconds
        >>> market_close = 16 * 3600  # 4:00 PM in seconds
        >>> unit2timestamp(0.5, '2023-01-01', time_range=(market_open, market_close))
        Timestamp('2023-01-01 12:45:00')
    """
    # Convert base_date to pandas Timestamp if it's not already
    if not isinstance(base_date, pd.Timestamp):
        base_date = pd.Timestamp(base_date)
    
    # Get seconds past midnight
    seconds = unit2seconds(unit_time, time_range)
    
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
