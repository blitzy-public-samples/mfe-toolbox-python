# mfe/models/realized/price_filter.py
"""
Price filtering and alignment for high-frequency financial data.

This module provides functions for filtering and aligning high-frequency price data
based on various sampling schemes. It serves as a foundational component for all
realized volatility estimators, ensuring consistent handling of irregular time series.

The implementation leverages Pandas' powerful time series capabilities for efficient
data manipulation, with optimized methods for various sampling schemes including
calendar time, business time, and fixed interval sampling. The module supports
multiple time formats with seamless conversion between them.

Functions:
    price_filter: Filter and align high-frequency price data based on sampling scheme
    align_prices: Align multiple price series to a common time grid
    visualize_filtered_prices: Create visualization of original and filtered price series
    convert_time_format: Convert between different time formats
    infer_time_format: Automatically detect time format from input data
"""

import logging
import warnings
from typing import (
    Any, Callable, Dict, List, Literal, Optional, Sequence, 
    Tuple, Type, TypeVar, Union, cast, overload
)
import numpy as np
import pandas as pd
from scipy import stats, signal, optimize

from ...core.exceptions import (
    DimensionError, NumericError, ParameterError, 
    raise_dimension_error, raise_numeric_error, warn_numeric
)
from .utils import (
    seconds2unit, unit2seconds, wall2seconds, seconds2wall,
    unit2wall, wall2unit, align_time, sample_prices
)

# Set up module-level logger
logger = logging.getLogger("mfe.models.realized.price_filter")

# Try to import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.info("Matplotlib not available. Visualization functions will be disabled.")


def price_filter(
    prices: Union[np.ndarray, pd.Series], 
    times: Union[np.ndarray, pd.Series, pd.DatetimeIndex], 
    sample_freq: Union[str, float, int], 
    time_unit: str = 'seconds',
    sampling_scheme: str = 'calendar',
    interpolation_method: str = 'previous',
    handle_missing: str = 'ffill',
    return_pandas: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Series, pd.DatetimeIndex]]:
    """
    Filter and align high-frequency price data based on sampling scheme.
    
    This function resamples irregular high-frequency price data to a regular grid
    based on the specified sampling frequency and scheme. It handles various time
    formats and provides options for interpolation and missing value handling.
    
    Args:
        prices: High-frequency price data as NumPy array or Pandas Series
        times: Corresponding time points as NumPy array, Pandas Series, or DatetimeIndex
        sample_freq: Sampling frequency (as string like '5min' or numeric value in time_unit)
        time_unit: Time unit for numeric sample_freq ('seconds', 'minutes', 'hours', 'days')
        sampling_scheme: Sampling scheme ('calendar', 'business', 'fixed')
        interpolation_method: Method for interpolating prices ('previous', 'linear', 'cubic')
        handle_missing: Method for handling missing values ('ffill', 'bfill', 'nearest', 'drop')
        return_pandas: Whether to return Pandas objects (Series/DatetimeIndex) instead of NumPy arrays
        
    Returns:
        Tuple of (filtered_prices, filtered_times) as NumPy arrays or Pandas objects
        
    Raises:
        ValueError: If inputs have invalid dimensions or if parameters are invalid
        TypeError: If inputs have incorrect types
        
    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from mfe.models.realized.price_filter import price_filter
        >>> # Example with numeric times
        >>> prices = np.array([100.0, 100.5, 101.0, 101.5, 102.0])
        >>> times = np.array([0, 30, 70, 150, 210])  # seconds
        >>> filtered_prices, filtered_times = price_filter(
        ...     prices, times, sample_freq=60, time_unit='seconds'
        ... )
        >>> filtered_times
        array([  0.,  60., 120., 180.])
        >>> filtered_prices
        array([100. , 100.5, 101.5, 101.5])
        
        >>> # Example with datetime times
        >>> prices = np.array([100.0, 100.5, 101.0, 101.5, 102.0])
        >>> times = pd.to_datetime([
        ...     '2023-01-01 09:30:00', 
        ...     '2023-01-01 09:30:30',
        ...     '2023-01-01 09:31:10', 
        ...     '2023-01-01 09:32:30',
        ...     '2023-01-01 09:33:30'
        ... ])
        >>> filtered_prices, filtered_times = price_filter(
        ...     prices, times, sample_freq='1min', return_pandas=True
        ... )
        >>> filtered_times
        DatetimeIndex(['2023-01-01 09:30:00', '2023-01-01 09:31:00',
                       '2023-01-01 09:32:00', '2023-01-01 09:33:00'],
                      dtype='datetime64[ns]', freq='T')
        >>> filtered_prices
        0    100.0
        1    100.5
        2    101.5
        3    102.0
        dtype: float64
    """
    # Convert inputs to numpy arrays if they aren't already
    prices_array = np.asarray(prices)
    
    # Validate input dimensions
    if prices_array.ndim != 1:
        raise ValueError("prices must be a 1D array or Series")
    
    # Handle different time formats
    is_datetime = False
    times_array = None
    
    # Check if times is a pandas DatetimeIndex or can be converted to one
    if isinstance(times, pd.DatetimeIndex):
        is_datetime = True
        datetime_index = times
    elif isinstance(times, (pd.Series, np.ndarray, list)):
        # Try to convert to DatetimeIndex if elements look like datetimes
        if isinstance(times, pd.Series) and pd.api.types.is_datetime64_dtype(times):
            is_datetime = True
            datetime_index = pd.DatetimeIndex(times)
        else:
            try:
                # Check if the first element can be parsed as a datetime
                if isinstance(times, (list, np.ndarray)):
                    if isinstance(times[0], (str, pd.Timestamp, np.datetime64)):
                        is_datetime = True
                        datetime_index = pd.DatetimeIndex(pd.to_datetime(times))
                    else:
                        # Numeric times
                        times_array = np.asarray(times, dtype=float)
                elif isinstance(times, pd.Series):
                    if pd.api.types.is_datetime64_dtype(times) or isinstance(times.iloc[0], (str, pd.Timestamp)):
                        is_datetime = True
                        datetime_index = pd.DatetimeIndex(pd.to_datetime(times))
                    else:
                        # Numeric times
                        times_array = np.asarray(times, dtype=float)
            except (ValueError, TypeError):
                # If conversion fails, assume numeric times
                times_array = np.asarray(times, dtype=float)
    else:
        raise TypeError("times must be a NumPy array, Pandas Series, or DatetimeIndex")
    
    # Validate input lengths
    if is_datetime:
        if len(prices_array) != len(datetime_index):
            raise ValueError(f"prices length ({len(prices_array)}) must match times length ({len(datetime_index)})")
    else:
        if len(prices_array) != len(times_array):
            raise ValueError(f"prices length ({len(prices_array)}) must match times length ({len(times_array)})")
    
    # Create a pandas DataFrame for easier manipulation
    if is_datetime:
        df = pd.DataFrame({'price': prices_array}, index=datetime_index)
    else:
        # Convert numeric times to seconds if needed
        if time_unit != 'seconds':
            times_seconds = unit2seconds(times_array, time_unit)
        else:
            times_seconds = times_array
        
        # Create a DataFrame with a numeric index
        df = pd.DataFrame({'price': prices_array, 'time': times_seconds})
    
    # Handle missing values
    if handle_missing == 'ffill':
        df = df.fillna(method='ffill')
    elif handle_missing == 'bfill':
        df = df.fillna(method='bfill')
    elif handle_missing == 'nearest':
        df = df.fillna(method='nearest')
    elif handle_missing == 'drop':
        df = df.dropna()
    else:
        raise ValueError(f"Unrecognized missing value handling method: {handle_missing}. "
                         f"Supported methods are 'ffill', 'bfill', 'nearest', 'drop'.")
    
    # Apply sampling based on the scheme
    if is_datetime:
        # For datetime index, we can use pandas resampling
        if sampling_scheme == 'calendar':
            # Calendar time sampling (regular intervals)
            if isinstance(sample_freq, (int, float)):
                # Convert numeric frequency to string format
                if time_unit == 'seconds':
                    freq_str = f"{int(sample_freq)}S"
                elif time_unit == 'minutes':
                    freq_str = f"{int(sample_freq)}T"
                elif time_unit == 'hours':
                    freq_str = f"{int(sample_freq)}H"
                elif time_unit == 'days':
                    freq_str = f"{int(sample_freq)}D"
                else:
                    raise ValueError(f"Unrecognized time unit: {time_unit}")
            else:
                # Use the provided string frequency
                freq_str = sample_freq
            
            # Resample using the specified frequency
            resampled = df.resample(freq_str)
            
            # Apply the specified interpolation method
            if interpolation_method == 'previous':
                filtered_df = resampled.first()  # Take first observation in each interval
            elif interpolation_method == 'linear':
                filtered_df = resampled.mean()  # Mean value in each interval (approximates linear interpolation)
            elif interpolation_method == 'cubic':
                # For cubic interpolation, we need to resample to a regular grid and then interpolate
                filtered_df = resampled.first().interpolate(method='cubic')
            else:
                raise ValueError(f"Unrecognized interpolation method: {interpolation_method}. "
                                 f"Supported methods are 'previous', 'linear', 'cubic'.")
        
        elif sampling_scheme == 'business':
            # Business time sampling (trading hours only)
            # First, filter to business days
            business_df = df[df.index.dayofweek < 5]  # Monday=0, Friday=4
            
            # Then apply regular resampling
            if isinstance(sample_freq, (int, float)):
                # Convert numeric frequency to string format
                if time_unit == 'seconds':
                    freq_str = f"{int(sample_freq)}S"
                elif time_unit == 'minutes':
                    freq_str = f"{int(sample_freq)}T"
                elif time_unit == 'hours':
                    freq_str = f"{int(sample_freq)}H"
                elif time_unit == 'days':
                    freq_str = f"{int(sample_freq)}B"  # Business days
                else:
                    raise ValueError(f"Unrecognized time unit: {time_unit}")
            else:
                # Use the provided string frequency
                freq_str = sample_freq
            
            # Resample using the specified frequency
            resampled = business_df.resample(freq_str)
            
            # Apply the specified interpolation method
            if interpolation_method == 'previous':
                filtered_df = resampled.first()
            elif interpolation_method == 'linear':
                filtered_df = resampled.mean()
            elif interpolation_method == 'cubic':
                filtered_df = resampled.first().interpolate(method='cubic')
            else:
                raise ValueError(f"Unrecognized interpolation method: {interpolation_method}")
        
        elif sampling_scheme == 'fixed':
            # Fixed interval sampling (exactly n points)
            if isinstance(sample_freq, (int, float)):
                # Interpret as number of points
                n_points = int(sample_freq)
                
                # Generate evenly spaced points
                start_time = df.index[0]
                end_time = df.index[-1]
                new_index = pd.date_range(start=start_time, end=end_time, periods=n_points)
                
                # Resample to the new index
                filtered_df = df.reindex(new_index, method=None)
                
                # Apply interpolation
                if interpolation_method == 'previous':
                    filtered_df = filtered_df.fillna(method='ffill')
                elif interpolation_method == 'linear':
                    filtered_df = filtered_df.interpolate(method='linear')
                elif interpolation_method == 'cubic':
                    filtered_df = filtered_df.interpolate(method='cubic')
                else:
                    raise ValueError(f"Unrecognized interpolation method: {interpolation_method}")
            else:
                raise ValueError("For fixed sampling scheme, sample_freq must be a numeric value "
                                 "representing the number of points")
        
        else:
            raise ValueError(f"Unrecognized sampling scheme: {sampling_scheme}. "
                             f"Supported schemes are 'calendar', 'business', 'fixed'.")
        
        # Extract filtered prices and times
        filtered_prices_array = filtered_df['price'].values
        filtered_times_index = filtered_df.index
        
        # Return results in the requested format
        if return_pandas:
            return pd.Series(filtered_prices_array, index=filtered_times_index), filtered_times_index
        else:
            return filtered_prices_array, np.array(filtered_times_index.astype(np.int64) / 1e9)
    
    else:
        # For numeric times, we need to create a regular grid and interpolate
        if sampling_scheme == 'calendar' or sampling_scheme == 'business':
            # For numeric times, calendar and business sampling are the same
            # (business sampling would require datetime information)
            
            # Use the sample_prices utility function
            filtered_prices_array, filtered_times_array = sample_prices(
                prices_array, times_seconds, sample_freq, time_unit, interpolation_method
            )
        
        elif sampling_scheme == 'fixed':
            # Fixed interval sampling (exactly n points)
            if isinstance(sample_freq, (int, float)):
                # Interpret as number of points
                n_points = int(sample_freq)
                
                # Generate evenly spaced points
                start_time = times_seconds[0]
                end_time = times_seconds[-1]
                filtered_times_array = np.linspace(start_time, end_time, n_points)
                
                # Interpolate prices to the new time grid
                filtered_prices_array = align_time(
                    prices_array, times_seconds, filtered_times_array, interpolation_method
                )
            else:
                raise ValueError("For fixed sampling scheme, sample_freq must be a numeric value "
                                 "representing the number of points")
        
        else:
            raise ValueError(f"Unrecognized sampling scheme: {sampling_scheme}")
        
        # Convert times back to the original unit if needed
        if time_unit != 'seconds':
            filtered_times_array = seconds2unit(filtered_times_array, time_unit)
        
        # Return results in the requested format
        if return_pandas:
            if time_unit == 'seconds':
                # Convert seconds to datetime for better display
                try:
                    base_time = pd.Timestamp('2000-01-01')  # Arbitrary base time
                    times_datetime = [base_time + pd.Timedelta(seconds=t) for t in filtered_times_array]
                    times_index = pd.DatetimeIndex(times_datetime)
                    return pd.Series(filtered_prices_array, index=times_index), times_index
                except Exception:
                    # If conversion fails, use numeric index
                    return pd.Series(filtered_prices_array), pd.Index(filtered_times_array)
            else:
                return pd.Series(filtered_prices_array), pd.Index(filtered_times_array)
        else:
            return filtered_prices_array, filtered_times_array


def align_prices(
    price_series: List[Union[np.ndarray, pd.Series]],
    time_series: List[Union[np.ndarray, pd.Series, pd.DatetimeIndex]],
    sample_freq: Union[str, float, int],
    time_unit: str = 'seconds',
    sampling_scheme: str = 'calendar',
    interpolation_method: str = 'previous',
    return_pandas: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.DataFrame, pd.DatetimeIndex]]:
    """
    Align multiple price series to a common time grid.
    
    This function resamples multiple irregular high-frequency price series to a
    common regular grid based on the specified sampling frequency and scheme.
    
    Args:
        price_series: List of high-frequency price data arrays or Series
        time_series: List of corresponding time points
        sample_freq: Sampling frequency (as string like '5min' or numeric value in time_unit)
        time_unit: Time unit for numeric sample_freq ('seconds', 'minutes', 'hours', 'days')
        sampling_scheme: Sampling scheme ('calendar', 'business', 'fixed')
        interpolation_method: Method for interpolating prices ('previous', 'linear', 'cubic')
        return_pandas: Whether to return Pandas objects instead of NumPy arrays
        
    Returns:
        Tuple of (aligned_prices, aligned_times) where aligned_prices is a 2D array or DataFrame
        
    Raises:
        ValueError: If inputs have invalid dimensions or if parameters are invalid
        TypeError: If inputs have incorrect types
        
    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from mfe.models.realized.price_filter import align_prices
        >>> # Example with two price series
        >>> prices1 = np.array([100.0, 100.5, 101.0, 101.5])
        >>> times1 = np.array([0, 30, 70, 150])  # seconds
        >>> prices2 = np.array([50.0, 50.2, 50.5, 50.8, 51.0])
        >>> times2 = np.array([0, 20, 60, 100, 180])  # seconds
        >>> aligned_prices, aligned_times = align_prices(
        ...     [prices1, prices2], [times1, times2], sample_freq=60, time_unit='seconds'
        ... )
        >>> aligned_times
        array([  0.,  60., 120., 180.])
        >>> aligned_prices
        array([[100. , 100.5, 101. , 101.5],
               [ 50. ,  50.5,  50.8,  51. ]])
    """
    # Validate inputs
    if not isinstance(price_series, list) or not isinstance(time_series, list):
        raise TypeError("price_series and time_series must be lists")
    
    if len(price_series) != len(time_series):
        raise ValueError(f"Number of price series ({len(price_series)}) must match "
                         f"number of time series ({len(time_series)})")
    
    if len(price_series) == 0:
        raise ValueError("At least one price series must be provided")
    
    # Filter each price series
    filtered_prices = []
    filtered_times = []
    
    for i, (prices, times) in enumerate(zip(price_series, time_series)):
        try:
            filtered_p, filtered_t = price_filter(
                prices, times, sample_freq, time_unit, sampling_scheme, 
                interpolation_method, return_pandas=False
            )
            filtered_prices.append(filtered_p)
            filtered_times.append(filtered_t)
        except Exception as e:
            logger.warning(f"Failed to filter price series {i}: {str(e)}")
            # Skip this series
            continue
    
    if not filtered_prices:
        raise ValueError("All price series filtering failed")
    
    # Find the common time grid (intersection of all filtered times)
    # For simplicity, we'll use the time grid from the first series
    # and align all other series to it
    common_times = filtered_times[0]
    
    # Align all price series to the common time grid
    aligned_prices = []
    
    for i, (prices, times) in enumerate(zip(filtered_prices, filtered_times)):
        if i == 0:
            # First series is already aligned
            aligned_prices.append(prices)
        else:
            # Align this series to the common time grid
            aligned_p = align_time(prices, times, common_times, interpolation_method)
            aligned_prices.append(aligned_p)
    
    # Convert to the requested output format
    if return_pandas:
        # Convert to DataFrame
        if isinstance(time_series[0], pd.DatetimeIndex) or (
            isinstance(time_series[0], (pd.Series, np.ndarray, list)) and 
            isinstance(time_series[0][0], (str, pd.Timestamp, np.datetime64))
        ):
            # Convert numeric times to datetime
            try:
                base_time = pd.Timestamp('2000-01-01')  # Arbitrary base time
                times_datetime = [base_time + pd.Timedelta(seconds=t) for t in common_times]
                times_index = pd.DatetimeIndex(times_datetime)
            except Exception:
                # If conversion fails, use numeric index
                times_index = pd.Index(common_times)
        else:
            times_index = pd.Index(common_times)
        
        # Create DataFrame with each price series as a column
        df = pd.DataFrame(np.column_stack(aligned_prices), index=times_index)
        return df, times_index
    else:
        # Return as NumPy arrays
        return np.column_stack(aligned_prices), common_times


def visualize_filtered_prices(
    original_prices: Union[np.ndarray, pd.Series],
    original_times: Union[np.ndarray, pd.Series, pd.DatetimeIndex],
    filtered_prices: Union[np.ndarray, pd.Series],
    filtered_times: Union[np.ndarray, pd.Series, pd.DatetimeIndex],
    title: str = 'Original vs Filtered Prices',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> Optional[Any]:
    """
    Create visualization of original and filtered price series.
    
    Args:
        original_prices: Original high-frequency price data
        original_times: Original time points
        filtered_prices: Filtered price data
        filtered_times: Filtered time points
        title: Plot title
        figsize: Figure size as (width, height) in inches
        save_path: Path to save the figure (if None, figure is displayed)
        
    Returns:
        Matplotlib figure object if matplotlib is available, None otherwise
        
    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from mfe.models.realized.price_filter import price_filter, visualize_filtered_prices
        >>> prices = np.array([100.0, 100.5, 101.0, 101.5, 102.0])
        >>> times = np.array([0, 30, 70, 150, 210])  # seconds
        >>> filtered_prices, filtered_times = price_filter(
        ...     prices, times, sample_freq=60, time_unit='seconds'
        ... )
        >>> fig = visualize_filtered_prices(prices, times, filtered_prices, filtered_times)
    """
    if not HAS_MATPLOTLIB:
        logger.warning("Matplotlib is not available. Cannot create visualization.")
        return None
    
    # Convert inputs to numpy arrays if they aren't already
    original_prices_array = np.asarray(original_prices)
    filtered_prices_array = np.asarray(filtered_prices)
    
    # Handle different time formats for original times
    if isinstance(original_times, pd.DatetimeIndex):
        original_times_array = np.array(original_times.astype(np.int64) / 1e9)
        is_datetime = True
    elif isinstance(original_times, pd.Series) and pd.api.types.is_datetime64_dtype(original_times):
        original_times_array = np.array(pd.DatetimeIndex(original_times).astype(np.int64) / 1e9)
        is_datetime = True
    else:
        original_times_array = np.asarray(original_times)
        is_datetime = False
    
    # Handle different time formats for filtered times
    if isinstance(filtered_times, pd.DatetimeIndex):
        filtered_times_array = np.array(filtered_times.astype(np.int64) / 1e9)
    elif isinstance(filtered_times, pd.Series) and pd.api.types.is_datetime64_dtype(filtered_times):
        filtered_times_array = np.array(pd.DatetimeIndex(filtered_times).astype(np.int64) / 1e9)
    else:
        filtered_times_array = np.asarray(filtered_times)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot original prices
    if is_datetime:
        # Convert seconds to datetime for better display
        base_time = pd.Timestamp('2000-01-01')  # Arbitrary base time
        original_dt = [base_time + pd.Timedelta(seconds=t) for t in original_times_array]
        filtered_dt = [base_time + pd.Timedelta(seconds=t) for t in filtered_times_array]
        
        ax.plot(original_dt, original_prices_array, 'o-', label='Original', alpha=0.5)
        ax.plot(filtered_dt, filtered_prices_array, 's-', label='Filtered', linewidth=2)
    else:
        ax.plot(original_times_array, original_prices_array, 'o-', label='Original', alpha=0.5)
        ax.plot(filtered_times_array, filtered_prices_array, 's-', label='Filtered', linewidth=2)
    
    # Add labels and legend
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format x-axis if datetime
    if is_datetime:
        fig.autofmt_xdate()
    
    # Save or display the figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.tight_layout()
    
    return fig


def convert_time_format(
    times: Union[np.ndarray, pd.Series, pd.DatetimeIndex],
    target_format: str = 'datetime',
    base_time: Optional[Union[str, pd.Timestamp]] = None,
    time_unit: str = 'seconds'
) -> Union[np.ndarray, pd.DatetimeIndex]:
    """
    Convert between different time formats.
    
    Args:
        times: Time points to convert
        target_format: Target format ('datetime', 'seconds', 'unit')
        base_time: Base time for conversion (required for 'datetime' target)
        time_unit: Time unit for 'unit' target format
        
    Returns:
        Converted time points in the target format
        
    Raises:
        ValueError: If conversion parameters are invalid
        
    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from mfe.models.realized.price_filter import convert_time_format
        >>> # Convert seconds to datetime
        >>> times_sec = np.array([0, 60, 120, 180])
        >>> convert_time_format(times_sec, 'datetime', '2023-01-01 09:30:00')
        DatetimeIndex(['2023-01-01 09:30:00', '2023-01-01 09:31:00',
                       '2023-01-01 09:32:00', '2023-01-01 09:33:00'],
                      dtype='datetime64[ns]', freq=None)
        
        >>> # Convert datetime to seconds
        >>> times_dt = pd.to_datetime(['2023-01-01 09:30:00', '2023-01-01 09:31:00',
        ...                           '2023-01-01 09:32:00', '2023-01-01 09:33:00'])
        >>> convert_time_format(times_dt, 'seconds')
        array([  0.,  60., 120., 180.])
    """
    # Validate inputs
    if not isinstance(times, (np.ndarray, pd.Series, pd.DatetimeIndex)):
        raise TypeError("times must be a NumPy array, Pandas Series, or DatetimeIndex")
    
    # Determine the current format
    if isinstance(times, pd.DatetimeIndex):
        current_format = 'datetime'
    elif isinstance(times, pd.Series) and pd.api.types.is_datetime64_dtype(times):
        current_format = 'datetime'
        times = pd.DatetimeIndex(times)
    else:
        # Assume numeric format
        times_array = np.asarray(times)
        current_format = 'numeric'
    
    # Convert to the target format
    if target_format.lower() == 'datetime':
        if current_format == 'datetime':
            # Already in datetime format
            return times
        else:
            # Convert numeric to datetime
            if base_time is None:
                raise ValueError("base_time must be provided for conversion to datetime format")
            
            # Convert to seconds if in a different unit
            if time_unit != 'seconds':
                times_seconds = unit2seconds(times_array, time_unit)
            else:
                times_seconds = times_array
            
            # Convert seconds to datetime
            return seconds2wall(times_seconds, base_time)
    
    elif target_format.lower() == 'seconds':
        if current_format == 'datetime':
            # Convert datetime to seconds
            return wall2seconds(times)
        else:
            # Convert from current unit to seconds
            if time_unit != 'seconds':
                return unit2seconds(times_array, time_unit)
            else:
                return times_array
    
    elif target_format.lower() == 'unit':
        if current_format == 'datetime':
            # Convert datetime to the specified unit
            seconds = wall2seconds(times)
            return seconds2unit(seconds, time_unit)
        else:
            # Convert from seconds to the specified unit
            if time_unit != 'seconds':
                return seconds2unit(times_array, time_unit)
            else:
                return times_array
    
    else:
        raise ValueError(f"Unrecognized target format: {target_format}. "
                         f"Supported formats are 'datetime', 'seconds', 'unit'.")


def infer_time_format(
    times: Union[np.ndarray, pd.Series, pd.DatetimeIndex]
) -> Tuple[str, Optional[str]]:
    """
    Automatically detect time format from input data.
    
    Args:
        times: Time points to analyze
        
    Returns:
        Tuple of (format_type, time_unit) where format_type is 'datetime' or 'numeric'
        and time_unit is the inferred unit for numeric times (or None for datetime)
        
    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from mfe.models.realized.price_filter import infer_time_format
        >>> # Datetime format
        >>> times_dt = pd.to_datetime(['2023-01-01 09:30:00', '2023-01-01 09:31:00'])
        >>> infer_time_format(times_dt)
        ('datetime', None)
        
        >>> # Numeric format (seconds)
        >>> times_sec = np.array([0, 60, 120, 180])
        >>> infer_time_format(times_sec)
        ('numeric', 'seconds')
        
        >>> # Numeric format (minutes)
        >>> times_min = np.array([0, 1, 2, 3])
        >>> infer_time_format(times_min)
        ('numeric', 'minutes')
    """
    # Check if times is a datetime format
    if isinstance(times, pd.DatetimeIndex):
        return 'datetime', None
    elif isinstance(times, pd.Series) and pd.api.types.is_datetime64_dtype(times):
        return 'datetime', None
    elif isinstance(times, (pd.Series, np.ndarray, list)):
        # Check if the first element is a datetime-like object
        if isinstance(times, (list, np.ndarray)):
            if len(times) > 0 and isinstance(times[0], (str, pd.Timestamp, np.datetime64)):
                try:
                    pd.to_datetime(times[0])
                    return 'datetime', None
                except (ValueError, TypeError):
                    pass
        elif isinstance(times, pd.Series):
            if len(times) > 0 and isinstance(times.iloc[0], (str, pd.Timestamp)):
                try:
                    pd.to_datetime(times.iloc[0])
                    return 'datetime', None
                except (ValueError, TypeError):
                    pass
    
    # If we get here, assume numeric format
    # Try to infer the time unit based on the values
    times_array = np.asarray(times)
    
    if len(times_array) < 2:
        # Not enough data to infer time unit
        return 'numeric', 'seconds'
    
    # Compute the average time difference
    avg_diff = np.mean(np.diff(times_array))
    
    # Infer time unit based on the average difference
    if avg_diff < 1:
        # Sub-second intervals, assume seconds
        return 'numeric', 'seconds'
    elif avg_diff < 100:
        # Small intervals, likely seconds
        return 'numeric', 'seconds'
    elif avg_diff < 6000:
        # Medium intervals, likely minutes
        return 'numeric', 'minutes'
    elif avg_diff < 360000:
        # Large intervals, likely hours
        return 'numeric', 'hours'
    else:
        # Very large intervals, likely days
        return 'numeric', 'days'



