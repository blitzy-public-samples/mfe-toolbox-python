# mfe/models/realized/wall2unit.py

"""
Convert wall clock time in HHMMSS format to a normalized unit interval [0,1].

This module provides a function to convert wall clock time in HHMMSS format
(hours, minutes, seconds) into a normalized unit interval [0,1], which is
essential for standardizing time inputs in high-frequency financial analysis.
The conversion enables consistent time normalization across different sampling
schemes and facilitates comparison of intraday patterns.

The implementation supports various input formats including NumPy arrays,
pandas Series, DatetimeIndex, and scalar values, with comprehensive type hints
and parameter validation. It also handles timezone-aware timestamps and provides
nanosecond precision for modern market data analysis.

Functions:
    wall2unit: Convert wall clock time in HHMMSS format to normalized unit interval [0,1]
"""

import logging
from typing import Union, Optional, Sequence, cast, overload, Tuple

import numpy as np
import pandas as pd

from mfe.core.exceptions import (
    DataError, raise_data_error, warn_numeric
)
from mfe.core.validation import (
    validate_input_type, validate_input_bounds, validate_custom_condition
)
from mfe.models.realized.wall2seconds import wall2seconds

# Set up module-level logger
logger = logging.getLogger("mfe.models.realized.wall2unit")

# Constants
SECONDS_PER_DAY = 86400  # 24 hours * 60 minutes * 60 seconds


@overload
def wall2unit(wall_time: int,
              time_range: Optional[Tuple[float, float]] = None) -> float:
    ...


@overload
def wall2unit(wall_time: np.ndarray,
              time_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
    ...


@overload
def wall2unit(wall_time: pd.Series,
              time_range: Optional[Tuple[float, float]] = None) -> pd.Series:
    ...


@overload
def wall2unit(wall_time: pd.DatetimeIndex,
              time_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
    ...


@validate_input_type(0, (int, float, np.ndarray, pd.Series, pd.DatetimeIndex, list, tuple))
def wall2unit(wall_time: Union[int, float, np.ndarray, pd.Series, pd.DatetimeIndex, Sequence[int]],
              time_range: Optional[Tuple[float, float]] = None) -> Union[float, np.ndarray, pd.Series]:
    """
    Convert wall clock time in HHMMSS format to normalized unit interval [0,1].

    This function converts wall clock time in HHMMSS format (hours, minutes, seconds)
    to a normalized unit interval [0,1], where 0 corresponds to midnight (00:00:00)
    and 1 corresponds to the next midnight (24:00:00). For example, 120000 (representing
    12:00:00) would be converted to 0.5. Optionally, a custom time range can be specified
    to map a specific time window to the unit interval (e.g., market trading hours).

    Args:
        wall_time: Wall clock time in HHMMSS format. Can be a scalar value, NumPy array,
                  pandas Series, pandas DatetimeIndex, or any sequence convertible to NumPy array.
        time_range: Optional tuple specifying the start and end times in seconds past midnight
                   to map to the unit interval [0,1]. If None, the full day (0, 86400) is used.
                   Default is None.

    Returns:
        Normalized time in unit interval [0,1]. The return type matches the input type
        (scalar, NumPy array, or pandas Series) except for DatetimeIndex inputs,
        which return NumPy arrays.

    Raises:
        ValueError: If input contains invalid time values or time_range is invalid
        TypeError: If input type is not supported

    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.wall2unit import wall2unit
        >>> # Scalar input
        >>> wall2unit(120000)  # 12:00:00 (noon)
        0.5
        >>> # NumPy array input
        >>> times = np.array([0, 60000, 120000, 180000])  # 00:00, 06:00, 12:00, 18:00
        >>> wall2unit(times)
        array([0.  , 0.25, 0.5 , 0.75])
        >>> # pandas Series input
        >>> import pandas as pd
        >>> time_series = pd.Series([0, 60000, 120000, 180000])
        >>> wall2unit(time_series)
        0    0.00
        1    0.25
        2    0.50
        3    0.75
        dtype: float64
        >>> # pandas DatetimeIndex input
        >>> times = pd.DatetimeIndex(['2023-01-01 00:00:00', '2023-01-01 06:00:00',
        ...                           '2023-01-01 12:00:00', '2023-01-01 18:00:00'])
        >>> wall2unit(times)
        array([0.  , 0.25, 0.5 , 0.75])
        >>> # Custom time range (e.g., market hours 9:30 to 16:00)
        >>> market_open = 9.5 * 3600  # 9:30 AM in seconds
        >>> market_close = 16 * 3600  # 4:00 PM in seconds
        >>> wall2unit(124500, time_range=(market_open, market_close))  # 12:45:00
        0.5
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

    # First convert wall time to seconds past midnight
    seconds = wall2seconds(wall_time)

    # Then convert seconds to unit time
    time_span = end_time - start_time

    # Handle different input types
    if isinstance(seconds, (float, int, np.number)):
        # Scalar input
        # Check if seconds is within the valid range
        if seconds < start_time or seconds > end_time:
            logger.warning(f"Wall time {wall_time} (seconds: {seconds}) is outside the specified "
                           f"time range [{start_time}, {end_time}]")
            # Clamp to valid range
            seconds = max(start_time, min(seconds, end_time))

        # Convert to unit time
        unit_time = (seconds - start_time) / time_span
        return unit_time

    elif isinstance(seconds, pd.Series):
        # Series input
        # Check if any values are outside the valid range
        if (seconds < start_time).any() or (seconds > end_time).any():
            logger.warning("Some wall times are outside the specified time range "
                           f"[{start_time}, {end_time}]")
            # Clamp to valid range
            seconds = seconds.clip(lower=start_time, upper=end_time)

        # Convert to unit time
        unit_time = (seconds - start_time) / time_span
        return unit_time

    else:  # numpy array
        # Ensure we have a numpy array
        seconds_array = np.asarray(seconds)

        # Check if any values are outside the valid range
        if (seconds_array < start_time).any() or (seconds_array > end_time).any():
            logger.warning("Some wall times are outside the specified time range "
                           f"[{start_time}, {end_time}]")
            # Clamp to valid range
            seconds_array = np.clip(seconds_array, start_time, end_time)

        # Convert to unit time
        unit_time = (seconds_array - start_time) / time_span
        return unit_time


def wall2timestamp_unit(wall_time: Union[int, np.ndarray, pd.Series, pd.DatetimeIndex],
                        base_date: Union[str, pd.Timestamp, np.datetime64],
                        time_range: Optional[Tuple[float, float]] = None) -> Union[pd.Timestamp, pd.DatetimeIndex, pd.Series]:
    """
    Convert wall clock time in HHMMSS format to pandas Timestamps with normalized unit time.

    This function converts wall clock time in HHMMSS format to pandas Timestamps
    by adding the corresponding time to a base date, and also returns the normalized
    unit time. This is useful for creating datetime objects with associated normalized
    time values for visualization or further analysis.

    Args:
        wall_time: Wall clock time in HHMMSS format. Can be a scalar value, NumPy array,
                  pandas Series, pandas DatetimeIndex, or any sequence convertible to NumPy array.
        base_date: The reference date to which the time will be added. Can be a string in
                  ISO format ('YYYY-MM-DD'), pandas Timestamp, or numpy datetime64.
        time_range: Optional tuple specifying the start and end times in seconds past midnight
                   to map to the unit interval [0,1]. If None, the full day (0, 86400) is used.
                   Default is None.

    Returns:
        For scalar inputs: tuple of (pandas Timestamp, unit_time)
        For array inputs: tuple of (pandas DatetimeIndex, numpy array of unit times)
        For Series inputs: tuple of (pandas Series with Timestamps, pandas Series with unit times)

    Raises:
        ValueError: If input contains invalid time values or time_range is invalid
        TypeError: If input type is not supported

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from mfe.models.realized.wall2unit import wall2timestamp_unit
        >>> # Scalar input
        >>> timestamp, unit_time = wall2timestamp_unit(120000, '2023-01-01')
        >>> timestamp
        Timestamp('2023-01-01 12:00:00')
        >>> unit_time
        0.5
        >>> # NumPy array input
        >>> times = np.array([0, 60000, 120000, 180000])
        >>> timestamps, unit_times = wall2timestamp_unit(times, '2023-01-01')
        >>> timestamps
        DatetimeIndex(['2023-01-01 00:00:00', '2023-01-01 06:00:00',
                       '2023-01-01 12:00:00', '2023-01-01 18:00:00'],
                      dtype='datetime64[ns]', freq=None)
        >>> unit_times
        array([0.  , 0.25, 0.5 , 0.75])
        >>> # Custom time range (e.g., market hours 9:30 to 16:00)
        >>> market_open = 9.5 * 3600  # 9:30 AM in seconds
        >>> market_close = 16 * 3600  # 4:00 PM in seconds
        >>> timestamp, unit_time = wall2timestamp_unit(124500, '2023-01-01', 
        ...                                           time_range=(market_open, market_close))
        >>> timestamp
        Timestamp('2023-01-01 12:45:00')
        >>> unit_time
        0.5
    """
    # Convert base_date to pandas Timestamp if it's not already
    if not isinstance(base_date, pd.Timestamp):
        base_date = pd.Timestamp(base_date)

    # Get unit time
    unit_time = wall2unit(wall_time, time_range)

    # Get seconds past midnight
    seconds = wall2seconds(wall_time)

    # Handle different input types
    if isinstance(seconds, (float, int, np.number)):
        # Scalar input
        timestamp = base_date + pd.Timedelta(seconds=float(seconds))
        return timestamp, unit_time
    elif isinstance(seconds, pd.Series):
        # Series input - preserve index
        timestamps = base_date + pd.to_timedelta(seconds, unit='s')
        return timestamps, unit_time
    else:
        # NumPy array or sequence input
        timestamps = pd.DatetimeIndex([base_date + pd.Timedelta(seconds=float(s)) for s in seconds])
        return timestamps, unit_time
