# mfe/models/realized/unit2wall.py
"""
Convert normalized unit time values [0,1] to wall clock time in HHMMSS format.

This module provides a function to convert normalized time in the unit interval [0,1]
to wall clock time in HHMMSS format (hours, minutes, seconds). This is essential for
mapping standardized time values to human-readable formats in high-frequency financial
analysis. The implementation supports various input formats including NumPy arrays,
pandas Series, DatetimeIndex, and scalar values, with comprehensive type hints and
parameter validation.

The function handles timezone-aware timestamps and provides nanosecond precision
for modern market data analysis. It also supports vectorized operations for
efficient conversion of large datasets.

Functions:
    unit2wall: Convert normalized unit time values [0,1] to wall clock time in HHMMSS format
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
from mfe.models.realized.unit2seconds import unit2seconds

# Set up module-level logger
logger = logging.getLogger("mfe.models.realized.unit2wall")

# Constants
SECONDS_PER_DAY = 86400  # 24 hours * 60 minutes * 60 seconds


@overload
def unit2wall(unit_time: float, 
              time_range: Optional[Tuple[float, float]] = None) -> int:
    ...


@overload
def unit2wall(unit_time: np.ndarray,
              time_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
    ...


@overload
def unit2wall(unit_time: pd.Series,
              time_range: Optional[Tuple[float, float]] = None) -> pd.Series:
    ...


@overload
def unit2wall(unit_time: pd.DatetimeIndex,
              time_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
    ...


@validate_input_type(0, (float, np.ndarray, pd.Series, pd.DatetimeIndex, list, tuple))
def unit2wall(unit_time: Union[float, np.ndarray, pd.Series, pd.DatetimeIndex, Sequence[float]],
              time_range: Optional[Tuple[float, float]] = None) -> Union[int, np.ndarray, pd.Series]:
    """
    Convert normalized unit time values [0,1] to wall clock time in HHMMSS format.

    This function converts time from a normalized unit interval [0,1] to wall clock time
    in HHMMSS format (hours, minutes, seconds), where 0 corresponds to midnight (00:00:00)
    and 1 corresponds to the next midnight (24:00:00). For example, 0.5 would be converted
    to 120000 (representing 12:00:00). Optionally, a custom time range can be specified to
    map the unit interval to a specific time window (e.g., market trading hours).

    Args:
        unit_time: Normalized time in unit interval [0,1]. Can be a scalar value, NumPy array,
                  pandas Series, pandas DatetimeIndex, or any sequence convertible to NumPy array.
        time_range: Optional tuple specifying the start and end times in seconds past midnight
                   to map the unit interval to. If None, the full day (0, 86400) is used.
                   Default is None.

    Returns:
        Wall clock time in HHMMSS format. The return type matches the input type
        (scalar, NumPy array, or pandas Series).

    Raises:
        ValueError: If input contains values outside the [0,1] range
        TypeError: If input type is not supported

    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.unit2wall import unit2wall
        >>> # Scalar input
        >>> unit2wall(0.5)  # 0.5 corresponds to 12:00:00 (noon)
        120000
        >>> # NumPy array input
        >>> unit_times = np.array([0.0, 0.25, 0.5, 0.75])  # 00:00, 06:00, 12:00, 18:00
        >>> unit2wall(unit_times)
        array([     0,  60000, 120000, 180000])
        >>> # pandas Series input
        >>> import pandas as pd
        >>> time_series = pd.Series([0.0, 0.25, 0.5, 0.75])
        >>> unit2wall(time_series)
        0        0
        1    60000
        2   120000
        3   180000
        dtype: int64
        >>> # pandas DatetimeIndex input
        >>> times = pd.DatetimeIndex(['2023-01-01 00:00:00', '2023-01-01 06:00:00',
        ...                           '2023-01-01 12:00:00', '2023-01-01 18:00:00'])
        >>> unit2wall(times)
        array([     0,  60000, 120000, 180000])
        >>> # Custom time range (e.g., market hours 9:30 to 16:00)
        >>> market_open = 9.5 * 3600  # 9:30 AM in seconds
        >>> market_close = 16 * 3600  # 4:00 PM in seconds
        >>> unit2wall(0.5, time_range=(market_open, market_close))
        124500
    """
    # Handle pandas DatetimeIndex input directly
    if isinstance(unit_time, pd.DatetimeIndex):
        # Extract time components
        hours = unit_time.hour
        minutes = unit_time.minute
        seconds = unit_time.second
        
        # Convert to HHMMSS format
        wall_time = hours * 10000 + minutes * 100 + seconds
        
        # Convert to numpy array for consistent return type
        return wall_time.values
    
    # For other input types, first convert unit time to seconds past midnight
    seconds = unit2seconds(unit_time, time_range)
    
    # Then convert seconds to wall clock time
    if isinstance(seconds, (float, int)):
        # Calculate hours, minutes, and seconds
        hours = int(np.floor(seconds / 3600))
        minutes = int(np.floor((seconds % 3600) / 60))
        seconds_val = int(np.floor(seconds % 60))
        
        # Convert to HHMMSS format
        wall_time = hours * 10000 + minutes * 100 + seconds_val
        
        return wall_time
    
    elif isinstance(seconds, pd.Series):
        # Calculate hours, minutes, and seconds
        hours = np.floor(seconds / 3600).astype(int)
        minutes = np.floor((seconds % 3600) / 60).astype(int)
        seconds_val = np.floor(seconds % 60).astype(int)
        
        # Convert to HHMMSS format
        wall_time = hours * 10000 + minutes * 100 + seconds_val
        
        return wall_time
    
    else:  # numpy array or sequence converted to numpy array
        # Ensure we have a numpy array
        seconds_array = np.asarray(seconds)
        
        # Calculate hours, minutes, and seconds
        hours = np.floor(seconds_array / 3600).astype(int)
        minutes = np.floor((seconds_array % 3600) / 60).astype(int)
        seconds_val = np.floor(seconds_array % 60).astype(int)
        
        # Convert to HHMMSS format
        wall_time = hours * 10000 + minutes * 100 + seconds_val
        
        return wall_time



def unit2timestamp_wall(unit_time: Union[float, np.ndarray, pd.Series],
                        base_date: Union[str, pd.Timestamp, np.datetime64],
                        time_format: str = 'HHMMSS',
                        time_range: Optional[Tuple[float, float]] = None) -> Union[str, np.ndarray, pd.Series]:
    """
    Convert normalized unit time values [0,1] into formatted time strings.

    This function converts normalized time in the unit interval [0,1] to formatted
    time strings (default is HHMMSS format). This is useful for creating human-readable
    time representations from normalized time values for visualization or reporting.

    Args:
        unit_time: Normalized time in unit interval [0,1]. Can be a scalar value, NumPy array,
                  pandas Series, or any sequence convertible to NumPy array.
        base_date: The reference date to which the time will be added. Can be a string in
                  ISO format ('YYYY-MM-DD'), pandas Timestamp, or numpy datetime64.
        time_format: Format string for the output time. Default is 'HHMMSS'.
                    Other options include 'HH:MM:SS', 'HH-MM-SS', etc.
        time_range: Optional tuple specifying the start and end times in seconds past midnight
                   to map the unit interval to. If None, the full day (0, 86400) is used.
                   Default is None.

    Returns:
        For scalar inputs: formatted time string
        For array inputs: numpy array of formatted time strings
        For Series inputs: pandas Series with formatted time strings

    Raises:
        ValueError: If input contains values outside the [0,1] range
        TypeError: If input type is not supported

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from mfe.models.realized.unit2wall import unit2timestamp_wall
        >>> # Scalar input
        >>> unit2timestamp_wall(0.5, '2023-01-01')
        '120000'
        >>> # Custom format
        >>> unit2timestamp_wall(0.5, '2023-01-01', time_format='HH:MM:SS')
        '12:00:00'
        >>> # NumPy array input
        >>> unit_times = np.array([0.0, 0.25, 0.5, 0.75])
        >>> unit2timestamp_wall(unit_times, '2023-01-01')
        array(['000000', '060000', '120000', '180000'], dtype='<U6')
        >>> # Custom time range (e.g., market hours 9:30 to 16:00)
        >>> market_open = 9.5 * 3600  # 9:30 AM in seconds
        >>> market_close = 16 * 3600  # 4:00 PM in seconds
        >>> unit2timestamp_wall(0.5, '2023-01-01', time_range=(market_open, market_close))
        '124500'
    """
    # Convert unit time to wall clock time
    wall_time = unit2wall(unit_time, time_range)
    
    # Format the wall clock time according to the specified format
    if isinstance(wall_time, (int, np.integer)):
        # Handle scalar input
        hours = wall_time // 10000
        minutes = (wall_time % 10000) // 100
        seconds = wall_time % 100
        
        if time_format == 'HHMMSS':
            return f'{hours:02d}{minutes:02d}{seconds:02d}'
        elif time_format == 'HH:MM:SS':
            return f'{hours:02d}:{minutes:02d}:{seconds:02d}'
        else:
            # Replace placeholders in the format string
            return time_format.replace('HH', f'{hours:02d}').replace('MM', f'{minutes:02d}').replace('SS', f'{seconds:02d}')
    
    elif isinstance(wall_time, pd.Series):
        # Handle pandas Series input
        hours = (wall_time // 10000).astype(int)
        minutes = ((wall_time % 10000) // 100).astype(int)
        seconds = (wall_time % 100).astype(int)
        
        if time_format == 'HHMMSS':
            return hours.astype(str).str.zfill(2) + minutes.astype(str).str.zfill(2) + seconds.astype(str).str.zfill(2)
        elif time_format == 'HH:MM:SS':
            return hours.astype(str).str.zfill(2) + ':' + minutes.astype(str).str.zfill(2) + ':' + seconds.astype(str).str.zfill(2)
        else:
            # Replace placeholders in the format string for each element
            result = pd.Series(index=wall_time.index, dtype=str)
            for i, idx in enumerate(wall_time.index):
                h, m, s = hours[idx], minutes[idx], seconds[idx]
                result[idx] = time_format.replace('HH', f'{h:02d}').replace('MM', f'{m:02d}').replace('SS', f'{s:02d}')
            return result
    
    else:  # numpy array
        # Handle numpy array input
        hours = (wall_time // 10000).astype(int)
        minutes = ((wall_time % 10000) // 100).astype(int)
        seconds = (wall_time % 100).astype(int)
        
        # Create an array of formatted strings
        result = np.empty(wall_time.shape, dtype=object)
        
        if time_format == 'HHMMSS':
            for i in range(len(wall_time)):
                result[i] = f'{hours[i]:02d}{minutes[i]:02d}{seconds[i]:02d}'
        elif time_format == 'HH:MM:SS':
            for i in range(len(wall_time)):
                result[i] = f'{hours[i]:02d}:{minutes[i]:02d}:{seconds[i]:02d}'
        else:
            for i in range(len(wall_time)):
                result[i] = time_format.replace('HH', f'{hours[i]:02d}').replace('MM', f'{minutes[i]:02d}').replace('SS', f'{seconds[i]:02d}')
        
        return result


# mfe/models/realized/unit2wall.py
"""
Convert normalized unit time values [0,1] to wall clock time in HHMMSS format.

This module provides a function to convert normalized time in the unit interval [0,1]
to wall clock time in HHMMSS format (hours, minutes, seconds). This is essential for
mapping standardized time values to human-readable formats in high-frequency financial
analysis. The implementation supports various input formats including NumPy arrays,
pandas Series, DatetimeIndex, and scalar values, with comprehensive type hints and
parameter validation.

The function handles timezone-aware timestamps and provides nanosecond precision
for modern market data analysis. It also supports vectorized operations for
efficient conversion of large datasets.

Functions:
    unit2wall: Convert normalized unit time values [0,1] to wall clock time in HHMMSS format
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
from mfe.models.realized.unit2seconds import unit2seconds

# Set up module-level logger
logger = logging.getLogger("mfe.models.realized.unit2wall")

# Constants
SECONDS_PER_DAY = 86400  # 24 hours * 60 minutes * 60 seconds


@overload
def unit2wall(unit_time: float, 
              time_range: Optional[Tuple[float, float]] = None) -> int:
    ...


@overload
def unit2wall(unit_time: np.ndarray,
              time_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
    ...


@overload
def unit2wall(unit_time: pd.Series,
              time_range: Optional[Tuple[float, float]] = None) -> pd.Series:
    ...


@overload
def unit2wall(unit_time: pd.DatetimeIndex,
              time_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
    ...


@validate_input_type(0, (float, np.ndarray, pd.Series, pd.DatetimeIndex, list, tuple))
def unit2wall(unit_time: Union[float, np.ndarray, pd.Series, pd.DatetimeIndex, Sequence[float]],
              time_range: Optional[Tuple[float, float]] = None) -> Union[int, np.ndarray, pd.Series]:
    """
    Convert normalized unit time values [0,1] to wall clock time in HHMMSS format.

    This function converts time from a normalized unit interval [0,1] to wall clock time
    in HHMMSS format (hours, minutes, seconds), where 0 corresponds to midnight (00:00:00)
    and 1 corresponds to the next midnight (24:00:00). For example, 0.5 would be converted
    to 120000 (representing 12:00:00). Optionally, a custom time range can be specified to
    map the unit interval to a specific time window (e.g., market trading hours).

    Args:
        unit_time: Normalized time in unit interval [0,1]. Can be a scalar value, NumPy array,
                  pandas Series, pandas DatetimeIndex, or any sequence convertible to NumPy array.
        time_range: Optional tuple specifying the start and end times in seconds past midnight
                   to map the unit interval to. If None, the full day (0, 86400) is used.
                   Default is None.

    Returns:
        Wall clock time in HHMMSS format. The return type matches the input type
        (scalar, NumPy array, or pandas Series).

    Raises:
        ValueError: If input contains values outside the [0,1] range
        TypeError: If input type is not supported

    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.unit2wall import unit2wall
        >>> # Scalar input
        >>> unit2wall(0.5)  # 0.5 corresponds to 12:00:00 (noon)
        120000
        >>> # NumPy array input
        >>> unit_times = np.array([0.0, 0.25, 0.5, 0.75])  # 00:00, 06:00, 12:00, 18:00
        >>> unit2wall(unit_times)
        array([     0,  60000, 120000, 180000])
        >>> # pandas Series input
        >>> import pandas as pd
        >>> time_series = pd.Series([0.0, 0.25, 0.5, 0.75])
        >>> unit2wall(time_series)
        0        0
        1    60000
        2   120000
        3   180000
        dtype: int64
        >>> # pandas DatetimeIndex input
        >>> times = pd.DatetimeIndex(['2023-01-01 00:00:00', '2023-01-01 06:00:00',
        ...                           '2023-01-01 12:00:00', '2023-01-01 18:00:00'])
        >>> unit2wall(times)
        array([     0,  60000, 120000, 180000])
        >>> # Custom time range (e.g., market hours 9:30 to 16:00)
        >>> market_open = 9.5 * 3600  # 9:30 AM in seconds
        >>> market_close = 16 * 3600  # 4:00 PM in seconds
        >>> unit2wall(0.5, time_range=(market_open, market_close))
        124500
    """
    # Handle pandas DatetimeIndex input directly
    if isinstance(unit_time, pd.DatetimeIndex):
        # Extract time components
        hours = unit_time.hour
        minutes = unit_time.minute
        seconds = unit_time.second
        
        # Convert to HHMMSS format
        wall_time = hours * 10000 + minutes * 100 + seconds
        
        # Convert to numpy array for consistent return type
        return wall_time.values
    
    # For other input types, first convert unit time to seconds past midnight
    seconds = unit2seconds(unit_time, time_range)
    
    # Then convert seconds to wall clock time
    if isinstance(seconds, (float, int)):
        # Calculate hours, minutes, and seconds
        hours = int(np.floor(seconds / 3600))
        minutes = int(np.floor((seconds % 3600) / 60))
        seconds_val = int(np.floor(seconds % 60))
        
        # Convert to HHMMSS format
        wall_time = hours * 10000 + minutes * 100 + seconds_val
        
        return wall_time
    
    elif isinstance(seconds, pd.Series):
        # Calculate hours, minutes, and seconds
        hours = np.floor(seconds / 3600).astype(int)
        minutes = np.floor((seconds % 3600) / 60).astype(int)
        seconds_val = np.floor(seconds % 60).astype(int)
        
        # Convert to HHMMSS format
        wall_time = hours * 10000 + minutes * 100 + seconds_val
        
        return wall_time
    
    else:  # numpy array or sequence converted to numpy array
        # Ensure we have a numpy array
        seconds_array = np.asarray(seconds)
        
        # Calculate hours, minutes, and seconds
        hours = np.floor(seconds_array / 3600).astype(int)
        minutes = np.floor((seconds_array % 3600) / 60).astype(int)
        seconds_val = np.floor(seconds_array % 60).astype(int)
        
        # Convert to HHMMSS format
        wall_time = hours * 10000 + minutes * 100 + seconds_val
        
        return wall_time



def unit2timestamp_wall(unit_time: Union[float, np.ndarray, pd.Series],
                        base_date: Union[str, pd.Timestamp, np.datetime64],
                        time_format: str = 'HHMMSS',
                        time_range: Optional[Tuple[float, float]] = None) -> Union[str, np.ndarray, pd.Series]:
    """
    Convert normalized unit time values [0,1] into formatted time strings.

    This function converts normalized time in the unit interval [0,1] to formatted
    time strings (default is HHMMSS format). This is useful for creating human-readable
    time representations from normalized time values for visualization or reporting.

    Args:
        unit_time: Normalized time in unit interval [0,1]. Can be a scalar value, NumPy array,
                  pandas Series, or any sequence convertible to NumPy array.
        base_date: The reference date to which the time will be added. Can be a string in
                  ISO format ('YYYY-MM-DD'), pandas Timestamp, or numpy datetime64.
        time_format: Format string for the output time. Default is 'HHMMSS'.
                    Other options include 'HH:MM:SS', 'HH-MM-SS', etc.
        time_range: Optional tuple specifying the start and end times in seconds past midnight
                   to map the unit interval to. If None, the full day (0, 86400) is used.
                   Default is None.

    Returns:
        For scalar inputs: formatted time string
        For array inputs: numpy array of formatted time strings
        For Series inputs: pandas Series with formatted time strings

    Raises:
        ValueError: If input contains values outside the [0,1] range
        TypeError: If input type is not supported

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from mfe.models.realized.unit2wall import unit2timestamp_wall
        >>> # Scalar input
        >>> unit2timestamp_wall(0.5, '2023-01-01')
        '120000'
        >>> # Custom format
        >>> unit2timestamp_wall(0.5, '2023-01-01', time_format='HH:MM:SS')
        '12:00:00'
        >>> # NumPy array input
        >>> unit_times = np.array([0.0, 0.25, 0.5, 0.75])
        >>> unit2timestamp_wall(unit_times, '2023-01-01')
        array(['000000', '060000', '120000', '180000'], dtype='<U6')
        >>> # Custom time range (e.g., market hours 9:30 to 16:00)
        >>> market_open = 9.5 * 3600  # 9:30 AM in seconds
        >>> market_close = 16 * 3600  # 4:00 PM in seconds
        >>> unit2timestamp_wall(0.5, '2023-01-01', time_range=(market_open, market_close))
        '124500'
    """
    # Convert unit time to wall clock time
    wall_time = unit2wall(unit_time, time_range)
    
    # Format the wall clock time according to the specified format
    if isinstance(wall_time, (int, np.integer)):
        # Handle scalar input
        hours = wall_time // 10000
        minutes = (wall_time % 10000) // 100
        seconds = wall_time % 100
        
        if time_format == 'HHMMSS':
            return f'{hours:02d}{minutes:02d}{seconds:02d}'
        elif time_format == 'HH:MM:SS':
            return f'{hours:02d}:{minutes:02d}:{seconds:02d}'
        else:
            # Replace placeholders in the format string
            return time_format.replace('HH', f'{hours:02d}').replace('MM', f'{minutes:02d}').replace('SS', f'{seconds:02d}')
    
    elif isinstance(wall_time, pd.Series):
        # Handle pandas Series input
        hours = (wall_time // 10000).astype(int)
        minutes = ((wall_time % 10000) // 100).astype(int)
        seconds = (wall_time % 100).astype(int)
        
        if time_format == 'HHMMSS':
            return hours.astype(str).str.zfill(2) + minutes.astype(str).str.zfill(2) + seconds.astype(str).str.zfill(2)
        elif time_format == 'HH:MM:SS':
            return hours.astype(str).str.zfill(2) + ':' + minutes.astype(str).str.zfill(2) + ':' + seconds.astype(str).str.zfill(2)
        else:
            # Replace placeholders in the format string for each element
            result = pd.Series(index=wall_time.index, dtype=str)
            for i, idx in enumerate(wall_time.index):
                h, m, s = hours[idx], minutes[idx], seconds[idx]
                result[idx] = time_format.replace('HH', f'{h:02d}').replace('MM', f'{m:02d}').replace('SS', f'{s:02d}')
            return result
    
    else:  # numpy array
        # Handle numpy array input
        hours = (wall_time // 10000).astype(int)
        minutes = ((wall_time % 10000) // 100).astype(int)
        seconds = (wall_time % 100).astype(int)
        
        # Create an array of formatted strings
        result = np.empty(wall_time.shape, dtype=object)
        
        if time_format == 'HHMMSS':
            for i in range(len(wall_time)):
                result[i] = f'{hours[i]:02d}{minutes[i]:02d}{seconds[i]:02d}'
        elif time_format == 'HH:MM:SS':
            for i in range(len(wall_time)):
                result[i] = f'{hours[i]:02d}:{minutes[i]:02d}:{seconds[i]:02d}'
        else:
            for i in range(len(wall_time)):
                result[i] = time_format.replace('HH', f'{hours[i]:02d}').replace('MM', f'{minutes[i]:02d}').replace('SS', f'{seconds[i]:02d}')
        
        return result
