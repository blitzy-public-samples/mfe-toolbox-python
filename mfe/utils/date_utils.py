import calendar
import datetime
from datetime import datetime as dt
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast, overload

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import BDay, Day, Hour, Minute, MonthEnd, QuarterEnd, Second, Week, YearEnd

from mfe.core.exceptions import (
    DataError, raise_data_error, warn_numeric
)
from mfe.core.validation import (
    validate_input_type, validate_custom_condition
)

# Type aliases for date-related types
DateType = Union[str, dt, np.datetime64, pd.Timestamp]
DateIndex = Union[pd.DatetimeIndex, List[DateType], np.ndarray]
DateRange = Union[pd.DatetimeIndex, List[DateType]]
FrequencyType = Union[str, pd.DateOffset]

def date_to_index(dates: Union[DateType, Sequence[DateType]], 
                  base_date: Optional[DateType] = None,
                  unit: str = 'D') -> Union[float, np.ndarray]:
    """Convert dates to numeric indices relative to a base date.
    
    This function converts dates to numeric indices representing the time difference
    from a base date in the specified unit. This is useful for converting dates to
    a format suitable for numerical analysis.
    
    Args:
        dates: Date or sequence of dates to convert
        base_date: Reference date for the conversion (default: earliest date in dates)
        unit: Time unit for the result ('D' for days, 'B' for business days,
              'W' for weeks, 'M' for months, 'Q' for quarters, 'Y' for years,
              'h' for hours, 'm' for minutes, 's' for seconds)
    
    Returns:
        float or np.ndarray: Numeric indices representing time differences from base_date
    
    Raises:
        TypeError: If dates or base_date are not valid date types
        ValueError: If unit is not a valid time unit
    """
    # Convert single date to list for uniform processing
    single_date = not isinstance(dates, (list, tuple, np.ndarray, pd.DatetimeIndex, pd.Series))
    dates_seq = [dates] if single_date else dates
    
    # Convert all dates to pandas Timestamps for consistent handling
    try:
        pd_dates = pd.to_datetime(dates_seq)
    except Exception as e:
        raise TypeError(f"Could not convert dates to datetime: {e}")
    
    # Determine base date if not provided
    if base_date is None:
        if len(pd_dates) == 0:
            raise ValueError("Cannot determine base_date from empty dates sequence")
        base_date = pd_dates.min()
    else:
        try:
            base_date = pd.Timestamp(base_date)
        except Exception as e:
            raise TypeError(f"Could not convert base_date to datetime: {e}")
    
    # Calculate time difference based on the specified unit
    if unit == 'D':  # Days
        result = (pd_dates - base_date).total_seconds() / (24 * 3600)
    elif unit == 'B':  # Business days
        # Create a range of business days and map dates to indices
        min_date = min(pd_dates.min(), base_date)
        max_date = max(pd_dates.max(), base_date)
        # Add buffer days to ensure we capture all dates
        min_date = min_date - pd.Timedelta(days=5)
        max_date = max_date + pd.Timedelta(days=5)
        
        bdays = pd.date_range(min_date, max_date, freq='B')
        bday_indices = {day: idx for idx, day in enumerate(bdays)}
        
        # Get the base date index
        base_idx = None
        for day in bdays:
            if day.date() == base_date.date():
                base_idx = bday_indices[day]
                break
        
        if base_idx is None:
            # Base date is not a business day, find the next one
            next_bday = pd.date_range(base_date, periods=5, freq='B')[0]
            base_idx = bday_indices[next_bday]
        
        # Map each date to its business day index
        result = np.array([
            bday_indices.get(pd.Timestamp(d.date()), np.nan) - base_idx
            for d in pd_dates
        ])
        
        # Handle non-business days
        mask = np.isnan(result)
        if mask.any():
            # For non-business days, find the next business day
            for i, is_nan in enumerate(mask):
                if is_nan:
                    d = pd_dates[i]
                    next_bday = pd.date_range(d, periods=5, freq='B')[0]
                    result[i] = bday_indices[next_bday] - base_idx - 0.5
                    warn_numeric(
                        f"Date {d} is not a business day, using next business day with offset",
                        operation="date_to_index",
                        issue="non-business day",
                        value=d
                    )
    elif unit == 'W':  # Weeks
        result = (pd_dates - base_date).total_seconds() / (7 * 24 * 3600)
    elif unit == 'M':  # Months
        # Calculate month difference plus fractional part for days within month
        years_diff = pd_dates.year - base_date.year
        months_diff = pd_dates.month - base_date.month
        total_months = years_diff * 12 + months_diff
        
        # Add fractional part based on day of month
        days_in_month = pd.Series([
            calendar.monthrange(d.year, d.month)[1] for d in pd_dates
        ])
        base_days_in_month = calendar.monthrange(base_date.year, base_date.month)[1]
        
        day_fraction = (pd_dates.day - 1) / days_in_month
        base_day_fraction = (base_date.day - 1) / base_days_in_month
        
        result = total_months + (day_fraction - base_day_fraction)
    elif unit == 'Q':  # Quarters
        # Calculate quarter difference
        years_diff = pd_dates.year - base_date.year
        quarters_diff = (pd_dates.month - 1) // 3 - (base_date.month - 1) // 3
        total_quarters = years_diff * 4 + quarters_diff
        
        # Add fractional part based on position within quarter
        quarter_month = pd_dates.month - (pd_dates.month - 1) % 3 - 1  # First month of the quarter (0-based)
        days_in_quarter = pd.Series([
            sum(calendar.monthrange(d.year, m)[1] for m in range(quarter_month + 1, quarter_month + 4))
            for d, quarter_month in zip(pd_dates, quarter_month)
        ])
        
        base_quarter_month = base_date.month - (base_date.month - 1) % 3 - 1
        base_days_in_quarter = sum(
            calendar.monthrange(base_date.year, m)[1] 
            for m in range(base_quarter_month + 1, base_quarter_month + 4)
        )
        
        # Calculate days from start of quarter
        days_from_quarter_start = pd.Series([
            (d - pd.Timestamp(d.year, quarter_month + 1, 1)).days
            for d, quarter_month in zip(pd_dates, quarter_month)
        ])
        base_days_from_quarter_start = (
            base_date - pd.Timestamp(base_date.year, base_quarter_month + 1, 1)
        ).days
        
        quarter_fraction = days_from_quarter_start / days_in_quarter
        base_quarter_fraction = base_days_from_quarter_start / base_days_in_quarter
        
        result = total_quarters + (quarter_fraction - base_quarter_fraction)
    elif unit == 'Y':  # Years
        # Calculate year difference plus fractional part for days within year
        years_diff = pd_dates.year - base_date.year
        
        # Add fractional part based on day of year
        is_leap_year = pd.Series([
            calendar.isleap(d.year) for d in pd_dates
        ])
        days_in_year = np.where(is_leap_year, 366, 365)
        
        base_is_leap_year = calendar.isleap(base_date.year)
        base_days_in_year = 366 if base_is_leap_year else 365
        
        day_of_year = pd.Series([d.dayofyear for d in pd_dates])
        base_day_of_year = base_date.dayofyear
        
        year_fraction = (day_of_year - 1) / days_in_year
        base_year_fraction = (base_day_of_year - 1) / base_days_in_year
        
        result = years_diff + (year_fraction - base_year_fraction)
    elif unit == 'h':  # Hours
        result = (pd_dates - base_date).total_seconds() / 3600
    elif unit == 'm':  # Minutes
        result = (pd_dates - base_date).total_seconds() / 60
    elif unit == 's':  # Seconds
        result = (pd_dates - base_date).total_seconds()
    else:
        raise ValueError(f"Invalid unit: {unit}. Valid units are 'D', 'B', 'W', 'M', 'Q', 'Y', 'h', 'm', 's'")
    
    # Return scalar for single date input
    if single_date:
        return float(result[0])
    
    return np.array(result)

def index_to_date(indices: Union[float, Sequence[float]],
                  base_date: DateType,
                  unit: str = 'D') -> Union[pd.Timestamp, pd.DatetimeIndex]:
    """Convert numeric indices to dates relative to a base date.
    
    This function converts numeric indices representing time differences from a base date
    to actual dates. It is the inverse operation of date_to_index.
    
    Args:
        indices: Numeric index or sequence of indices to convert
        base_date: Reference date for the conversion
        unit: Time unit for the indices ('D' for days, 'B' for business days,
              'W' for weeks, 'M' for months, 'Q' for quarters, 'Y' for years,
              'h' for hours, 'm' for minutes, 's' for seconds)
    
    Returns:
        pd.Timestamp or pd.DatetimeIndex: Date(s) corresponding to the indices
    
    Raises:
        TypeError: If indices is not a numeric type or base_date is not a valid date type
        ValueError: If unit is not a valid time unit
    """
    # Convert single index to list for uniform processing
    single_index = isinstance(indices, (int, float)) or (
        isinstance(indices, np.ndarray) and indices.size == 1
    )
    indices_seq = [float(indices)] if single_index else indices
    
    # Convert base_date to pandas Timestamp
    try:
        base_date = pd.Timestamp(base_date)
    except Exception as e:
        raise TypeError(f"Could not convert base_date to datetime: {e}")
    
    # Convert indices to dates based on the specified unit
    if unit == 'D':  # Days
        result = [base_date + pd.Timedelta(days=idx) for idx in indices_seq]
    elif unit == 'B':  # Business days
        # Handle fractional business days (non-business days)
        whole_indices = []
        for idx in indices_seq:
            if idx % 1 == 0:
                whole_indices.append(int(idx))
            else:
                # For fractional indices, use the floor and add a flag
                whole_indices.append(int(idx - 0.5))
        
        # Create business day offsets
        result = [base_date + BDay(idx) for idx in whole_indices]
    elif unit == 'W':  # Weeks
        result = [base_date + pd.Timedelta(weeks=idx) for idx in indices_seq]
    elif unit == 'M':  # Months
        result = []
        for idx in indices_seq:
            # Split into whole months and fractional part
            whole_months = int(idx)
            fraction = idx - whole_months
            
            # Add whole months
            new_date = base_date + pd.DateOffset(months=whole_months)
            
            # Add fractional part as days
            days_in_month = calendar.monthrange(new_date.year, new_date.month)[1]
            days_to_add = int(round(fraction * days_in_month))
            new_date = new_date + pd.Timedelta(days=days_to_add)
            
            result.append(new_date)
    elif unit == 'Q':  # Quarters
        result = []
        for idx in indices_seq:
            # Split into whole quarters and fractional part
            whole_quarters = int(idx)
            fraction = idx - whole_quarters
            
            # Add whole quarters
            new_date = base_date + pd.DateOffset(months=whole_quarters * 3)
            
            # Add fractional part as days
            quarter_month = new_date.month - (new_date.month - 1) % 3 - 1  # First month of the quarter (0-based)
            days_in_quarter = sum(
                calendar.monthrange(new_date.year, m)[1] 
                for m in range(quarter_month + 1, quarter_month + 4)
            )
            days_to_add = int(round(fraction * days_in_quarter))
            new_date = new_date + pd.Timedelta(days=days_to_add)
            
            result.append(new_date)
    elif unit == 'Y':  # Years
        result = []
        for idx in indices_seq:
            # Split into whole years and fractional part
            whole_years = int(idx)
            fraction = idx - whole_years
            
            # Add whole years
            new_date = base_date + pd.DateOffset(years=whole_years)
            
            # Add fractional part as days
            days_in_year = 366 if calendar.isleap(new_date.year) else 365
            days_to_add = int(round(fraction * days_in_year))
            new_date = new_date + pd.Timedelta(days=days_to_add)
            
            result.append(new_date)
    elif unit == 'h':  # Hours
        result = [base_date + pd.Timedelta(hours=idx) for idx in indices_seq]
    elif unit == 'm':  # Minutes
        result = [base_date + pd.Timedelta(minutes=idx) for idx in indices_seq]
    elif unit == 's':  # Seconds
        result = [base_date + pd.Timedelta(seconds=idx) for idx in indices_seq]
    else:
        raise ValueError(f"Invalid unit: {unit}. Valid units are 'D', 'B', 'W', 'M', 'Q', 'Y', 'h', 'm', 's'")
    
    # Return scalar for single index input
    if single_index:
        return result[0]
    
    return pd.DatetimeIndex(result)


@validate_input_type(0, (str, pd.Timestamp, dt, np.datetime64))
@validate_input_type(1, (str, pd.Timestamp, dt, np.datetime64))
def date_range(start_date: DateType,
               end_date: DateType,
               freq: str = 'D',
               inclusive: str = 'both',
               calendar: Optional[pd.AbstractHolidayCalendar] = None) -> pd.DatetimeIndex:
    """Create a range of dates with the specified frequency.
    
    This function creates a DatetimeIndex with dates ranging from start_date to end_date
    with the specified frequency. It supports various frequency options and can
    handle business day calendars.
    
    Args:
        start_date: Start date of the range
        end_date: End date of the range
        freq: Frequency of the date range ('D' for daily, 'B' for business days,
              'W' for weekly, 'M' for month end, 'Q' for quarter end, 'Y' for year end,
              'h' for hourly, 'm' for minutely, 's' for secondly, or any pandas frequency string)
        inclusive: Include boundaries ('left', 'right', 'both', 'neither')
        calendar: Optional holiday calendar for business day calculations
    
    Returns:
        pd.DatetimeIndex: Range of dates with the specified frequency
    
    Raises:
        TypeError: If start_date or end_date are not valid date types
        ValueError: If freq is not a valid frequency string
    """
    # Convert dates to pandas Timestamps
    try:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
    except Exception as e:
        raise TypeError(f"Could not convert dates to datetime: {e}")
    
    # Validate inclusive parameter
    if inclusive not in ('left', 'right', 'both', 'neither'):
        raise ValueError("inclusive must be one of 'left', 'right', 'both', 'neither'")
    
    # Map common frequency strings to pandas frequency strings
    freq_map = {
        'D': 'D',       # Daily
        'B': 'B',       # Business days
        'W': 'W-MON',   # Weekly (Mondays)
        'M': 'M',       # Month end
        'Q': 'Q',       # Quarter end
        'Y': 'Y',       # Year end
        'h': 'H',       # Hourly
        'm': 'T',       # Minutely (T is the pandas code for minutes)
        's': 'S',       # Secondly
    }
    
    # Convert frequency to pandas frequency
    pd_freq = freq_map.get(freq, freq)
    
    # Create date range
    try:
        if calendar is not None and pd_freq == 'B':
            # Use custom business day calendar
            custom_bday = pd.offsets.CustomBusinessDay(calendar=calendar)
            date_index = pd.date_range(start, end, freq=custom_bday, inclusive=inclusive)
        else:
            date_index = pd.date_range(start, end, freq=pd_freq, inclusive=inclusive)
    except Exception as e:
        raise ValueError(f"Error creating date range: {e}")
    
    return date_index


@validate_input_type(0, (str, pd.Timestamp, dt, np.datetime64))
@validate_input_type(1, (str, pd.Timestamp, dt, np.datetime64))
def business_day_count(start_date: DateType,
                       end_date: DateType,
                       calendar: Optional[pd.AbstractHolidayCalendar] = None) -> int:
    """Count the number of business days between two dates.
    
    This function counts the number of business days (excluding weekends and holidays)
    between start_date and end_date, inclusive.
    
    Args:
        start_date: Start date
        end_date: End date
        calendar: Optional holiday calendar (default: USFederalHolidayCalendar)
    
    Returns:
        int: Number of business days between start_date and end_date, inclusive
    
    Raises:
        TypeError: If start_date or end_date are not valid date types
    """
    # Convert dates to pandas Timestamps
    try:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
    except Exception as e:
        raise TypeError(f"Could not convert dates to datetime: {e}")
    
    # Use default calendar if none provided
    if calendar is None:
        calendar = USFederalHolidayCalendar()
    
    # Create custom business day with the specified calendar
    custom_bday = pd.offsets.CustomBusinessDay(calendar=calendar)
    
    # Create business day range and count days
    bday_range = pd.date_range(start, end, freq=custom_bday, inclusive='both')
    return len(bday_range)


@validate_input_type(0, (str, pd.Timestamp, dt, np.datetime64))
def is_business_day(date: DateType,
                    calendar: Optional[pd.AbstractHolidayCalendar] = None) -> bool:
    """Check if a date is a business day.
    
    This function checks if the given date is a business day (not a weekend or holiday).
    
    Args:
        date: Date to check
        calendar: Optional holiday calendar (default: USFederalHolidayCalendar)
    
    Returns:
        bool: True if the date is a business day, False otherwise
    
    Raises:
        TypeError: If date is not a valid date type
    """
    # Convert date to pandas Timestamp
    try:
        pd_date = pd.Timestamp(date)
    except Exception as e:
        raise TypeError(f"Could not convert date to datetime: {e}")
    
    # Check if date is a weekend
    if pd_date.dayofweek >= 5:  # 5 = Saturday, 6 = Sunday
        return False
    
    # Use default calendar if none provided
    if calendar is None:
        calendar = USFederalHolidayCalendar()
    
    # Check if date is a holiday
    holidays = calendar.holidays(pd_date, pd_date)
    return len(holidays) == 0


@validate_input_type(0, (str, pd.Timestamp, dt, np.datetime64))
def next_business_day(date: DateType,
                      calendar: Optional[pd.AbstractHolidayCalendar] = None) -> pd.Timestamp:
    """Get the next business day after the given date.
    
    This function returns the next business day (not a weekend or holiday)
    after the given date.
    
    Args:
        date: Reference date
        calendar: Optional holiday calendar (default: USFederalHolidayCalendar)
    
    Returns:
        pd.Timestamp: Next business day
    
    Raises:
        TypeError: If date is not a valid date type
    """
    # Convert date to pandas Timestamp
    try:
        pd_date = pd.Timestamp(date)
    except Exception as e:
        raise TypeError(f"Could not convert date to datetime: {e}")
    
    # Use default calendar if none provided
    if calendar is None:
        calendar = USFederalHolidayCalendar()
    
    # Create custom business day with the specified calendar
    custom_bday = pd.offsets.CustomBusinessDay(calendar=calendar)
    
    # Get next business day
    next_day = pd_date + custom_bday
    return next_day


@validate_input_type(0, (str, pd.Timestamp, dt, np.datetime64))
def previous_business_day(date: DateType,
                          calendar: Optional[pd.AbstractHolidayCalendar] = None) -> pd.Timestamp:
    """Get the previous business day before the given date.
    
    This function returns the previous business day (not a weekend or holiday)
    before the given date.
    
    Args:
        date: Reference date
        calendar: Optional holiday calendar (default: USFederalHolidayCalendar)
    
    Returns:
        pd.Timestamp: Previous business day
    
    Raises:
        TypeError: If date is not a valid date type
    """
    # Convert date to pandas Timestamp
    try:
        pd_date = pd.Timestamp(date)
    except Exception as e:
        raise TypeError(f"Could not convert date to datetime: {e}")
    
    # Use default calendar if none provided
    if calendar is None:
        calendar = USFederalHolidayCalendar()
    
    # Create custom business day with the specified calendar
    custom_bday = pd.offsets.CustomBusinessDay(calendar=calendar)
    
    # Get previous business day
    prev_day = pd_date - custom_bday
    return prev_day


def align_time_series(*series: Union[pd.Series, pd.DataFrame],
                      method: str = 'outer',
                      fill_value: Optional[float] = None) -> List[Union[pd.Series, pd.DataFrame]]:
    """Align multiple time series to a common date index.
    
    This function aligns multiple time series (Series or DataFrames) to a common
    date index using the specified join method.
    
    Args:
        *series: Time series to align
        method: Join method ('outer', 'inner', 'left', 'right')
        fill_value: Value to use for filling missing values (default: None)
    
    Returns:
        List[Union[pd.Series, pd.DataFrame]]: Aligned time series
    
    Raises:
        TypeError: If any input is not a pandas Series or DataFrame
        ValueError: If method is not a valid join method
    """
    # Validate inputs
    if not series:
        raise ValueError("At least one time series must be provided")
    
    for i, s in enumerate(series):
        if not isinstance(s, (pd.Series, pd.DataFrame)):
            raise TypeError(f"Input {i} must be a pandas Series or DataFrame")
    
    # Validate method
    if method not in ('outer', 'inner', 'left', 'right'):
        raise ValueError("method must be one of 'outer', 'inner', 'left', 'right'")
    
    # Align time series
    if len(series) == 1:
        # Single series, nothing to align
        return list(series)
    
    # Align all series to a common index
    aligned_series = pd.concat(series, axis=1, join=method)
    
    # Fill missing values if specified
    if fill_value is not None:
        aligned_series = aligned_series.fillna(fill_value)
    
    # Split back into separate series
    result = []
    col_idx = 0
    for s in series:
        if isinstance(s, pd.Series):
            # Extract single column for Series
            result.append(aligned_series.iloc[:, col_idx])
            col_idx += 1
        else:
            # Extract multiple columns for DataFrame
            n_cols = s.shape[1]
            result.append(aligned_series.iloc[:, col_idx:col_idx + n_cols])
            col_idx += n_cols
    
    return result


def parse_date(date_str: str, 
               format: Optional[str] = None, 
               dayfirst: bool = False,
               yearfirst: bool = False,
               exact: bool = True) -> pd.Timestamp:
    """Parse a date string into a pandas Timestamp.
    
    This function provides flexible date parsing with various format options.
    
    Args:
        date_str: Date string to parse
        format: Date format string (if None, tries to infer format)
        dayfirst: Whether to interpret the first value as the day
        yearfirst: Whether to interpret the first value as the year
        exact: Whether the format must match exactly
    
    Returns:
        pd.Timestamp: Parsed date
    
    Raises:
        ValueError: If the date string cannot be parsed
    """
    try:
        if format is not None:
            # Parse with specified format
            return pd.to_datetime(date_str, format=format, exact=exact)
        else:
            # Try to infer format
            return pd.to_datetime(date_str, dayfirst=dayfirst, yearfirst=yearfirst)
    except Exception as e:
        raise ValueError(f"Could not parse date string '{date_str}': {e}")


def convert_to_datetime_index(dates: Union[DateType, Sequence[DateType]]) -> pd.DatetimeIndex:
    """Convert various date formats to a pandas DatetimeIndex.
    
    This function converts various date formats (strings, datetime objects,
    numpy datetime64, pandas Timestamps) to a pandas DatetimeIndex.
    
    Args:
        dates: Date or sequence of dates to convert
    
    Returns:
        pd.DatetimeIndex: DatetimeIndex representation of the dates
    
    Raises:
        TypeError: If dates cannot be converted to datetime
    """
    try:
        if isinstance(dates, (str, dt, np.datetime64, pd.Timestamp)):
            # Single date
            return pd.DatetimeIndex([pd.Timestamp(dates)])
        else:
            # Sequence of dates
            return pd.DatetimeIndex(pd.to_datetime(dates))
    except Exception as e:
        raise TypeError(f"Could not convert dates to DatetimeIndex: {e}")


def add_time_offset(date: DateType, 
                    offset: Union[int, float], 
                    unit: str = 'D') -> pd.Timestamp:
    """Add a time offset to a date.
    
    This function adds a time offset to a date using the specified unit.
    
    Args:
        date: Base date
        offset: Amount to add
        unit: Time unit ('D' for days, 'B' for business days,
              'W' for weeks, 'M' for months, 'Q' for quarters, 'Y' for years,
              'h' for hours, 'm' for minutes, 's' for seconds)
    
    Returns:
        pd.Timestamp: Date with the offset added
    
    Raises:
        TypeError: If date is not a valid date type
        ValueError: If unit is not a valid time unit
    """
    # Convert date to pandas Timestamp
    try:
        pd_date = pd.Timestamp(date)
    except Exception as e:
        raise TypeError(f"Could not convert date to datetime: {e}")
    
    # Add offset based on the specified unit
    if unit == 'D':  # Days
        return pd_date + pd.Timedelta(days=offset)
    elif unit == 'B':  # Business days
        return pd_date + BDay(offset)
    elif unit == 'W':  # Weeks
        return pd_date + Week(offset)
    elif unit == 'M':  # Months
        return pd_date + MonthEnd(offset)
    elif unit == 'Q':  # Quarters
        return pd_date + QuarterEnd(offset)
    elif unit == 'Y':  # Years
        return pd_date + YearEnd(offset)
    elif unit == 'h':  # Hours
        return pd_date + Hour(offset)
    elif unit == 'm':  # Minutes
        return pd_date + Minute(offset)
    elif unit == 's':  # Seconds
        return pd_date + Second(offset)
    else:
        raise ValueError(f"Invalid unit: {unit}. Valid units are 'D', 'B', 'W', 'M', 'Q', 'Y', 'h', 'm', 's'")


def date_difference(start_date: DateType, 
                    end_date: DateType, 
                    unit: str = 'D') -> float:
    """Calculate the time difference between two dates.
    
    This function calculates the time difference between two dates in the specified unit.
    
    Args:
        start_date: Start date
        end_date: End date
        unit: Time unit ('D' for days, 'B' for business days,
              'W' for weeks, 'M' for months, 'Q' for quarters, 'Y' for years,
              'h' for hours, 'm' for minutes, 's' for seconds)
    
    Returns:
        float: Time difference in the specified unit
    
    Raises:
        TypeError: If start_date or end_date are not valid date types
        ValueError: If unit is not a valid time unit
    """
    # Use date_to_index with end_date as base_date
    return date_to_index(start_date, end_date, unit) * -1


def get_month_end(date: DateType) -> pd.Timestamp:
    """Get the last day of the month for a given date.
    
    Args:
        date: Reference date
    
    Returns:
        pd.Timestamp: Last day of the month
    
    Raises:
        TypeError: If date is not a valid date type
    """
    try:
        pd_date = pd.Timestamp(date)
    except Exception as e:
        raise TypeError(f"Could not convert date to datetime: {e}")
    
    # Get the last day of the month
    last_day = calendar.monthrange(pd_date.year, pd_date.month)[1]
    return pd.Timestamp(pd_date.year, pd_date.month, last_day)


def get_quarter_end(date: DateType) -> pd.Timestamp:
    """Get the last day of the quarter for a given date.
    
    Args:
        date: Reference date
    
    Returns:
        pd.Timestamp: Last day of the quarter
    
    Raises:
        TypeError: If date is not a valid date type
    """
    try:
        pd_date = pd.Timestamp(date)
    except Exception as e:
        raise TypeError(f"Could not convert date to datetime: {e}")
    
    # Calculate the quarter end month
    quarter = (pd_date.month - 1) // 3
    quarter_end_month = (quarter + 1) * 3
    
    # Get the last day of the quarter end month
    last_day = calendar.monthrange(pd_date.year, quarter_end_month)[1]
    return pd.Timestamp(pd_date.year, quarter_end_month, last_day)


def get_year_end(date: DateType) -> pd.Timestamp:
    """Get the last day of the year for a given date.
    
    Args:
        date: Reference date
    
    Returns:
        pd.Timestamp: Last day of the year
    
    Raises:
        TypeError: If date is not a valid date type
    """
    try:
        pd_date = pd.Timestamp(date)
    except Exception as e:
        raise TypeError(f"Could not convert date to datetime: {e}")
    
    # Get the last day of the year
    return pd.Timestamp(pd_date.year, 12, 31)


def convert_timezone(date: DateType, 
                     from_tz: Optional[str] = None, 
                     to_tz: str = 'UTC') -> pd.Timestamp:
    """Convert a date from one timezone to another.
    
    Args:
        date: Date to convert
        from_tz: Source timezone (if None, assumes date is timezone-naive)
        to_tz: Target timezone
    
    Returns:
        pd.Timestamp: Date converted to the target timezone
    
    Raises:
        TypeError: If date is not a valid date type
        ValueError: If timezone is not valid
    """
    try:
        pd_date = pd.Timestamp(date)
    except Exception as e:
        raise TypeError(f"Could not convert date to datetime: {e}")
    
    # Apply source timezone if specified
    if from_tz is not None:
        if pd_date.tzinfo is not None:
            raise ValueError("Date already has timezone information")
        pd_date = pd_date.tz_localize(from_tz)
    
    # Convert to target timezone
    if pd_date.tzinfo is None:
        # Timezone-naive date, localize to target timezone
        return pd_date.tz_localize(to_tz)
    else:
        # Timezone-aware date, convert to target timezone
        return pd_date.tz_convert(to_tz)


def create_business_day_calendar(holidays: Optional[List[DateType]] = None,
                                 weekend_mask: Optional[str] = None) -> pd.AbstractHolidayCalendar:
    """Create a custom business day calendar.
    
    This function creates a custom business day calendar with the specified
    holidays and weekend mask.
    
    Args:
        holidays: List of holiday dates
        weekend_mask: String with 7 characters indicating which days are weekend
                     (0 = Monday, 6 = Sunday, '1' = weekend, '0' = business day)
                     Default is '0011100' (Saturday and Sunday are weekends)
    
    Returns:
        pd.AbstractHolidayCalendar: Custom business day calendar
    
    Raises:
        ValueError: If weekend_mask is not a valid mask
    """
    # Create custom calendar class
    class CustomCalendar(pd.AbstractHolidayCalendar):
        rules = []
    
    # Add holidays if specified
    if holidays is not None:
        try:
            holiday_dates = pd.DatetimeIndex(pd.to_datetime(holidays))
            CustomCalendar.rules = [pd.Holiday(f"Holiday {i}", year=d.year, month=d.month, day=d.day)
                                   for i, d in enumerate(holiday_dates)]
        except Exception as e:
            raise ValueError(f"Could not convert holidays to datetime: {e}")
    
    # Create calendar instance
    calendar = CustomCalendar()
    
    # Apply custom weekend mask if specified
    if weekend_mask is not None:
        if not isinstance(weekend_mask, str) or len(weekend_mask) != 7:
            raise ValueError("weekend_mask must be a string with 7 characters")
        if not all(c in '01' for c in weekend_mask):
            raise ValueError("weekend_mask must contain only '0' and '1'")
        
        # Create custom business day with the specified weekend mask
        calendar.weekmask = weekend_mask
    
    return calendar


def date_to_period(date: DateType, 
                   freq: str = 'M') -> pd.Period:
    """Convert a date to a pandas Period with the specified frequency.
    
    Args:
        date: Date to convert
        freq: Period frequency ('D' for daily, 'W' for weekly, 'M' for monthly,
              'Q' for quarterly, 'Y' for yearly)
    
    Returns:
        pd.Period: Period representation of the date
    
    Raises:
        TypeError: If date is not a valid date type
        ValueError: If freq is not a valid frequency
    """
    try:
        pd_date = pd.Timestamp(date)
    except Exception as e:
        raise TypeError(f"Could not convert date to datetime: {e}")
    
    # Map common frequency strings to pandas frequency strings
    freq_map = {
        'D': 'D',       # Daily
        'W': 'W',       # Weekly
        'M': 'M',       # Monthly
        'Q': 'Q',       # Quarterly
        'Y': 'Y',       # Yearly
    }
    
    # Convert frequency to pandas frequency
    pd_freq = freq_map.get(freq, freq)
    
    try:
        return pd.Period(pd_date, freq=pd_freq)
    except Exception as e:
        raise ValueError(f"Could not convert date to period with frequency '{freq}': {e}")


def period_to_date(period: pd.Period, 
                   position: str = 'end') -> pd.Timestamp:
    """Convert a pandas Period to a date.
    
    Args:
        period: Period to convert
        position: Position within the period ('start', 'end', 'middle')
    
    Returns:
        pd.Timestamp: Date representation of the period
    
    Raises:
        TypeError: If period is not a pandas Period
        ValueError: If position is not valid
    """
    if not isinstance(period, pd.Period):
        raise TypeError("period must be a pandas Period")
    
    if position == 'start':
        return period.start_time
    elif position == 'end':
        return period.end_time
    elif position == 'middle':
        # Calculate the middle of the period
        start = period.start_time
        end = period.end_time
        middle = start + (end - start) / 2
        return middle
    else:
        raise ValueError("position must be one of 'start', 'end', 'middle'")


def get_fiscal_year_dates(date: DateType, 
                          fiscal_start_month: int = 10) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Get the start and end dates of the fiscal year containing the given date.
    
    Args:
        date: Reference date
        fiscal_start_month: Month when the fiscal year starts (1-12)
    
    Returns:
        Tuple[pd.Timestamp, pd.Timestamp]: Start and end dates of the fiscal year
    
    Raises:
        TypeError: If date is not a valid date type
        ValueError: If fiscal_start_month is not valid
    """
    try:
        pd_date = pd.Timestamp(date)
    except Exception as e:
        raise TypeError(f"Could not convert date to datetime: {e}")
    
    # Validate fiscal_start_month
    if not isinstance(fiscal_start_month, int) or fiscal_start_month < 1 or fiscal_start_month > 12:
        raise ValueError("fiscal_start_month must be an integer between 1 and 12")
    
    # Calculate fiscal year
    if pd_date.month >= fiscal_start_month:
        # Date is in the first part of the fiscal year
        fiscal_year = pd_date.year
    else:
        # Date is in the last part of the fiscal year
        fiscal_year = pd_date.year - 1
    
    # Calculate start and end dates
    fiscal_start = pd.Timestamp(fiscal_year, fiscal_start_month, 1)
    fiscal_end_year = fiscal_year + 1
    fiscal_end_month = fiscal_start_month - 1
    if fiscal_end_month == 0:
        fiscal_end_month = 12
        fiscal_end_year -= 1
    
    fiscal_end = get_month_end(pd.Timestamp(fiscal_end_year, fiscal_end_month, 1))
    
    return fiscal_start, fiscal_end


def get_trading_days(start_date: DateType,
                     end_date: DateType,
                     calendar: Optional[pd.AbstractHolidayCalendar] = None) -> pd.DatetimeIndex:
    """Get all trading days (business days) between two dates.
    
    Args:
        start_date: Start date
        end_date: End date
        calendar: Optional holiday calendar (default: USFederalHolidayCalendar)
    
    Returns:
        pd.DatetimeIndex: All trading days between start_date and end_date, inclusive
    
    Raises:
        TypeError: If start_date or end_date are not valid date types
    """
    # Use date_range with business day frequency
    return date_range(start_date, end_date, freq='B', inclusive='both', calendar=calendar)


def get_day_of_week(date: DateType) -> int:
    """Get the day of the week for a given date (0=Monday, 6=Sunday).
    
    Args:
        date: Date to check
    
    Returns:
        int: Day of the week (0=Monday, 6=Sunday)
    
    Raises:
        TypeError: If date is not a valid date type
    """
    try:
        pd_date = pd.Timestamp(date)
    except Exception as e:
        raise TypeError(f"Could not convert date to datetime: {e}")
    
    return pd_date.dayofweek


def get_day_of_year(date: DateType) -> int:
    """Get the day of the year for a given date (1-366).
    
    Args:
        date: Date to check
    
    Returns:
        int: Day of the year (1-366)
    
    Raises:
        TypeError: If date is not a valid date type
    """
    try:
        pd_date = pd.Timestamp(date)
    except Exception as e:
        raise TypeError(f"Could not convert date to datetime: {e}")
    
    return pd_date.dayofyear


def get_week_of_year(date: DateType) -> int:
    """Get the week of the year for a given date (1-53).
    
    Args:
        date: Date to check
    
    Returns:
        int: Week of the year (1-53)
    
    Raises:
        TypeError: If date is not a valid date type
    """
    try:
        pd_date = pd.Timestamp(date)
    except Exception as e:
        raise TypeError(f"Could not convert date to datetime: {e}")
    
    return pd_date.week


def get_quarter(date: DateType) -> int:
    """Get the quarter for a given date (1-4).
    
    Args:
        date: Date to check
    
    Returns:
        int: Quarter (1-4)
    
    Raises:
        TypeError: If date is not a valid date type
    """
    try:
        pd_date = pd.Timestamp(date)
    except Exception as e:
        raise TypeError(f"Could not convert date to datetime: {e}")
    
    return pd_date.quarter


def format_date(date: DateType, 
                format_str: str = '%Y-%m-%d') -> str:
    """Format a date as a string using the specified format.
    
    Args:
        date: Date to format
        format_str: Format string (strftime format)
    
    Returns:
        str: Formatted date string
    
    Raises:
        TypeError: If date is not a valid date type
    """
    try:
        pd_date = pd.Timestamp(date)
    except Exception as e:
        raise TypeError(f"Could not convert date to datetime: {e}")
    
    return pd_date.strftime(format_str)


def is_leap_year(year: Union[int, DateType]) -> bool:
    """Check if a year is a leap year.
    
    Args:
        year: Year to check (as integer or date)
    
    Returns:
        bool: True if the year is a leap year, False otherwise
    
    Raises:
        TypeError: If year is not a valid year or date type
    """
    if isinstance(year, (str, dt, np.datetime64, pd.Timestamp)):
        try:
            pd_date = pd.Timestamp(year)
            year = pd_date.year
        except Exception as e:
            raise TypeError(f"Could not convert date to datetime: {e}")
    
    if not isinstance(year, int):
        raise TypeError("year must be an integer or a valid date type")
    
    return calendar.isleap(year)


def days_in_month(year: int, month: int) -> int:
    """Get the number of days in a month.
    
    Args:
        year: Year
        month: Month (1-12)
    
    Returns:
        int: Number of days in the month
    
    Raises:
        ValueError: If month is not valid
    """
    if not isinstance(year, int) or not isinstance(month, int):
        raise TypeError("year and month must be integers")
    
    if month < 1 or month > 12:
        raise ValueError("month must be between 1 and 12")
    
    return calendar.monthrange(year, month)[1]


def days_in_year(year: int) -> int:
    """Get the number of days in a year.
    
    Args:
        year: Year
    
    Returns:
        int: Number of days in the year (365 or 366)
    """
    if not isinstance(year, int):
        raise TypeError("year must be an integer")
    
    return 366 if calendar.isleap(year) else 365
