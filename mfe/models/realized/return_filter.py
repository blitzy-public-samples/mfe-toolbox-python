"""
Return filtering and processing for high-frequency financial data.

This module provides functions for computing and filtering returns from high-frequency
price data, with support for various return types, sampling schemes, and time formats.
It serves as a critical component for realized volatility estimators, ensuring consistent
return calculation across different sampling schemes.

The implementation leverages Pandas' powerful time series capabilities for efficient
 data manipulation, with optimized methods for handling irregular time series. Performance-critical
 operations are accelerated using Numba's JIT compilation for maximum efficiency.

Functions:
    compute_returns: Calculate returns from price series with various return types
    filter_returns: Filter and align returns based on sampling scheme
    detect_outliers: Identify and handle outliers in return series
    visualize_returns: Create visualization of return characteristics
    compute_return_statistics: Calculate descriptive statistics for returns
    align_returns: Align multiple return series to a common time grid
    infer_return_characteristics: Automatically detect return properties
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
    unit2wall, wall2unit, align_time, compute_returns, filter_outliers
)

# Set up module-level logger
logger = logging.getLogger("mfe.models.realized.return_filter")

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
    logger.debug("Numba available for return filtering acceleration")
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    HAS_NUMBA = False
    logger.info("Numba not available. Return filtering will use pure NumPy/Pandas implementations.")

# Try to import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.info("Matplotlib not available. Visualization functions will be disabled.")


@jit(nopython=True, cache=True)
def _compute_log_returns_numba(prices: np.ndarray) -> np.ndarray:
    """
    Numba-accelerated implementation of log return computation.
    
    Args:
        prices: Array of price data
        
    Returns:
        Array of log returns
    """
    n = len(prices)
    returns = np.empty(n - 1, dtype=np.float64)
    
    for i in range(n - 1):
        returns[i] = np.log(prices[i + 1]) - np.log(prices[i])
    
    return returns


@jit(nopython=True, cache=True)
def _compute_simple_returns_numba(prices: np.ndarray) -> np.ndarray:
    """
    Numba-accelerated implementation of simple return computation.
    
    Args:
        prices: Array of price data
        
    Returns:
        Array of simple returns
    """
    n = len(prices)
    returns = np.empty(n - 1, dtype=np.float64)
    
    for i in range(n - 1):
        returns[i] = (prices[i + 1] - prices[i]) / prices[i]
    
    return returns



def compute_returns_from_prices(
    prices: Union[np.ndarray, pd.Series], 
    return_type: str = 'log'
) -> np.ndarray:
    """
    Calculate returns from price series with various return types.
    
    Args:
        prices: Array or Series of price data
        return_type: Type of returns to compute ('log', 'simple', 'absolute')
        
    Returns:
        Array of returns
        
    Raises:
        ValueError: If return_type is not recognized or if prices contain non-positive values
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.return_filter import compute_returns_from_prices
        >>> prices = np.array([100, 101, 102, 103])
        >>> compute_returns_from_prices(prices, 'log')
        array([0.00995033, 0.00985222, 0.00975709])
        >>> compute_returns_from_prices(prices, 'simple')
        array([0.01, 0.00990099, 0.00980392])
    """
    # Convert to numpy array if not already
    prices_array = np.asarray(prices)
    
    # Validate inputs
    if prices_array.ndim != 1:
        raise ValueError("prices must be a 1D array or Series")
    if len(prices_array) < 2:
        raise ValueError("prices must have at least 2 elements to compute returns")
    
    # Check for non-positive prices
    if np.any(prices_array <= 0):
        raise ValueError("prices must be positive for return calculation")
    
    # Validate return_type
    if return_type.lower() not in ['log', 'simple', 'absolute']:
        raise ValueError(f"Unrecognized return_type: {return_type}. "
                         f"Supported types are 'log', 'simple', 'absolute'.")
    
    # Compute returns based on type
    if return_type.lower() == 'log':
        # Use Numba-accelerated implementation if available
        if HAS_NUMBA:
            returns = _compute_log_returns_numba(prices_array)
        else:
            returns = np.diff(np.log(prices_array))
    
    elif return_type.lower() == 'simple':
        # Use Numba-accelerated implementation if available
        if HAS_NUMBA:
            returns = _compute_simple_returns_numba(prices_array)
        else:
            returns = np.diff(prices_array) / prices_array[:-1]
    
    elif return_type.lower() == 'absolute':
        returns = np.diff(prices_array)
    
    # If input was a pandas Series, preserve the index
    if isinstance(prices, pd.Series):
        index = prices.index[1:]
        return pd.Series(returns, index=index).values
    
    return returns



def filter_returns(
    returns: Union[np.ndarray, pd.Series], 
    times: Union[np.ndarray, pd.Series, pd.DatetimeIndex], 
    sample_freq: Union[str, float, int], 
    time_unit: str = 'seconds',
    sampling_scheme: str = 'calendar',
    handle_outliers: bool = False,
    outlier_method: str = 'std',
    outlier_threshold: float = 3.0,
    return_pandas: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.Series, pd.DatetimeIndex]]:
    """
    Filter and align returns based on sampling scheme.
    
    This function resamples irregular high-frequency returns to a regular grid
    based on the specified sampling frequency and scheme. It handles various time
    formats and provides options for outlier handling.
    
    Args:
        returns: High-frequency return data as NumPy array or Pandas Series
        times: Corresponding time points as NumPy array, Pandas Series, or DatetimeIndex
        sample_freq: Sampling frequency (as string like '5min' or numeric value in time_unit)
        time_unit: Time unit for numeric sample_freq ('seconds', 'minutes', 'hours', 'days')
        sampling_scheme: Sampling scheme ('calendar', 'business', 'fixed')
        handle_outliers: Whether to detect and handle outliers
        outlier_method: Method for identifying outliers ('std', 'iqr', 'mad')
        outlier_threshold: Threshold for outlier detection
        return_pandas: Whether to return Pandas objects (Series/DatetimeIndex) instead of NumPy arrays
        
    Returns:
        Tuple of (filtered_returns, filtered_times) as NumPy arrays or Pandas objects
        
    Raises:
        ValueError: If inputs have invalid dimensions or if parameters are invalid
        TypeError: If inputs have incorrect types
        
    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from mfe.models.realized.return_filter import filter_returns
        >>> # Example with numeric times
        >>> returns = np.array([0.001, 0.002, -0.001, 0.003, 0.002])
        >>> times = np.array([0, 30, 70, 150, 210])  # seconds
        >>> filtered_returns, filtered_times = filter_returns(
        ...     returns, times, sample_freq=60, time_unit='seconds'
        ... )
        >>> filtered_times
        array([  0.,  60., 120., 180.])
        >>> filtered_returns
        array([0.001, 0.001, 0.003, 0.002])
    """
    # Convert inputs to numpy arrays if they aren't already
    returns_array = np.asarray(returns)
    
    # Validate input dimensions
    if returns_array.ndim != 1:
        raise ValueError("returns must be a 1D array or Series")
    
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
        if len(returns_array) != len(datetime_index):
            raise ValueError(f"returns length ({len(returns_array)}) must match times length ({len(datetime_index)})")
    else:
        if len(returns_array) != len(times_array):
            raise ValueError(f"returns length ({len(returns_array)}) must match times length ({len(times_array)})")
    
    # Handle outliers if requested
    if handle_outliers:
        # Detect outliers
        outlier_mask = filter_outliers(returns_array, outlier_method, outlier_threshold)
        
        # Remove outliers
        clean_returns = returns_array[outlier_mask]
        
        if is_datetime:
            clean_times = datetime_index[outlier_mask]
        else:
            clean_times = times_array[outlier_mask]
        
        # Log outlier removal
        n_outliers = len(returns_array) - len(clean_returns)
        if n_outliers > 0:
            logger.info(f"Removed {n_outliers} outliers ({n_outliers/len(returns_array)*100:.1f}%) "
                       f"using {outlier_method} method with threshold {outlier_threshold}")
        
        # Update arrays for further processing
        returns_array = clean_returns
        if is_datetime:
            datetime_index = clean_times
        else:
            times_array = clean_times
    
    # Create a pandas DataFrame for easier manipulation
    if is_datetime:
        df = pd.DataFrame({'return': returns_array}, index=datetime_index)
    else:
        # Convert numeric times to seconds if needed
        if time_unit != 'seconds':
            times_seconds = unit2seconds(times_array, time_unit)
        else:
            times_seconds = times_array
        
        # Create a DataFrame with a numeric index
        df = pd.DataFrame({'return': returns_array, 'time': times_seconds})
    
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
            
            # Sum returns within each interval
            # This is the correct approach for returns (unlike prices where we might use first/mean)
            filtered_df = resampled.sum()
        
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
            
            # Sum returns within each interval
            filtered_df = resampled.sum()
        
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
                # For returns, we need to sum all returns between each pair of new index points
                filtered_df = pd.DataFrame(index=new_index[:-1], columns=['return'])
                
                for i in range(len(new_index) - 1):
                    # Get all returns between this index point and the next
                    mask = (df.index >= new_index[i]) & (df.index < new_index[i+1])
                    if mask.any():
                        filtered_df.iloc[i, 0] = df.loc[mask, 'return'].sum()
                    else:
                        filtered_df.iloc[i, 0] = 0.0
            else:
                raise ValueError("For fixed sampling scheme, sample_freq must be a numeric value "
                                 "representing the number of points")
        
        else:
            raise ValueError(f"Unrecognized sampling scheme: {sampling_scheme}. "
                             f"Supported schemes are 'calendar', 'business', 'fixed'.")
        
        # Extract filtered returns and times
        filtered_returns_array = filtered_df['return'].values
        filtered_times_index = filtered_df.index
        
        # Return results in the requested format
        if return_pandas:
            return pd.Series(filtered_returns_array, index=filtered_times_index), filtered_times_index
        else:
            return filtered_returns_array, np.array(filtered_times_index.astype(np.int64) / 1e9)
    
    else:
        # For numeric times, we need to create a regular grid and aggregate returns
        if sampling_scheme == 'calendar' or sampling_scheme == 'business':
            # For numeric times, calendar and business sampling are the same
            # (business sampling would require datetime information)
            
            # Convert sample_freq to seconds if needed
            if isinstance(sample_freq, str):
                # Parse string frequency
                import re
                match = re.match(r'(\d+)([a-zA-Z]+)', sample_freq)
                if match:
                    value, unit = match.groups()
                    # Convert to seconds
                    if unit.lower() in ['s', 'sec', 'second', 'seconds']:
                        freq_seconds = int(value)
                    elif unit.lower() in ['m', 'min', 'minute', 'minutes']:
                        freq_seconds = int(value) * 60
                    elif unit.lower() in ['h', 'hour', 'hours']:
                        freq_seconds = int(value) * 3600
                    elif unit.lower() in ['d', 'day', 'days']:
                        freq_seconds = int(value) * 86400
                    else:
                        raise ValueError(f"Unrecognized frequency unit: {unit}")
                else:
                    raise ValueError(f"Could not parse frequency string: {sample_freq}")
                
                # Convert to the specified time unit
                if time_unit == 'seconds':
                    freq = freq_seconds
                elif time_unit == 'minutes':
                    freq = freq_seconds / 60.0
                elif time_unit == 'hours':
                    freq = freq_seconds / 3600.0
                elif time_unit == 'days':
                    freq = freq_seconds / 86400.0
                else:
                    raise ValueError(f"Unrecognized time unit: {time_unit}")
            else:
                # Numeric frequency in the specified time unit
                freq = float(sample_freq)
            
            # Generate regular time grid
            start_time = times_seconds[0]
            end_time = times_seconds[-1]
            num_points = int(np.floor((end_time - start_time) / freq)) + 1
            filtered_times_array = np.linspace(start_time, start_time + (num_points - 1) * freq, num_points)
            
            # Aggregate returns within each interval
            filtered_returns_array = np.zeros(len(filtered_times_array) - 1)
            
            for i in range(len(filtered_times_array) - 1):
                # Get all returns between this time point and the next
                mask = (times_seconds >= filtered_times_array[i]) & (times_seconds < filtered_times_array[i+1])
                if mask.any():
                    filtered_returns_array[i] = np.sum(returns_array[mask])
            
            # Adjust filtered_times_array to match filtered_returns_array
            filtered_times_array = filtered_times_array[:-1]
        
        elif sampling_scheme == 'fixed':
            # Fixed interval sampling (exactly n points)
            if isinstance(sample_freq, (int, float)):
                # Interpret as number of points
                n_points = int(sample_freq)
                
                # Generate evenly spaced points
                start_time = times_seconds[0]
                end_time = times_seconds[-1]
                grid_times = np.linspace(start_time, end_time, n_points)
                
                # Aggregate returns within each interval
                filtered_returns_array = np.zeros(len(grid_times) - 1)
                
                for i in range(len(grid_times) - 1):
                    # Get all returns between this time point and the next
                    mask = (times_seconds >= grid_times[i]) & (times_seconds < grid_times[i+1])
                    if mask.any():
                        filtered_returns_array[i] = np.sum(returns_array[mask])
                
                filtered_times_array = grid_times[:-1]
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
                    return pd.Series(filtered_returns_array, index=times_index), times_index
                except Exception:
                    # If conversion fails, use numeric index
                    return pd.Series(filtered_returns_array), pd.Index(filtered_times_array)
            else:
                return pd.Series(filtered_returns_array), pd.Index(filtered_times_array)
        else:
            return filtered_returns_array, filtered_times_array



def align_returns(
    return_series: List[Union[np.ndarray, pd.Series]],
    time_series: List[Union[np.ndarray, pd.Series, pd.DatetimeIndex]],
    sample_freq: Union[str, float, int],
    time_unit: str = 'seconds',
    sampling_scheme: str = 'calendar',
    return_pandas: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.DataFrame, pd.DatetimeIndex]]:
    """
    Align multiple return series to a common time grid.
    
    This function resamples multiple irregular high-frequency return series to a
    common regular grid based on the specified sampling frequency and scheme.
    
    Args:
        return_series: List of high-frequency return data arrays or Series
        time_series: List of corresponding time points
        sample_freq: Sampling frequency (as string like '5min' or numeric value in time_unit)
        time_unit: Time unit for numeric sample_freq ('seconds', 'minutes', 'hours', 'days')
        sampling_scheme: Sampling scheme ('calendar', 'business', 'fixed')
        return_pandas: Whether to return Pandas objects instead of NumPy arrays
        
    Returns:
        Tuple of (aligned_returns, aligned_times) where aligned_returns is a 2D array or DataFrame
        
    Raises:
        ValueError: If inputs have invalid dimensions or if parameters are invalid
        TypeError: If inputs have incorrect types
        
    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from mfe.models.realized.return_filter import align_returns
        >>> # Example with two return series
        >>> returns1 = np.array([0.01, -0.005, 0.008, -0.002])
        >>> times1 = np.array([0, 30, 70, 150])  # seconds
        >>> returns2 = np.array([0.005, -0.002, 0.003, -0.001, 0.004])
        >>> times2 = np.array([0, 20, 60, 100, 180])  # seconds
        >>> aligned_returns, aligned_times = align_returns(
        ...     [returns1, returns2], [times1, times2], sample_freq=60, time_unit='seconds'
        ... )
        >>> aligned_times
        array([  0.,  60., 120., 180.])
        >>> aligned_returns
        array([[ 0.01  ,  0.003 , -0.002 ,  0.    ],
               [ 0.003 ,  0.003 , -0.001 ,  0.004 ]])
    """
    # Validate inputs
    if not isinstance(return_series, list) or not isinstance(time_series, list):
        raise TypeError("return_series and time_series must be lists")
    
    if len(return_series) != len(time_series):
        raise ValueError(f"Number of return series ({len(return_series)}) must match "
                         f"number of time series ({len(time_series)})")
    
    if len(return_series) == 0:
        raise ValueError("At least one return series must be provided")
    
    # Filter each return series
    filtered_returns = []
    filtered_times = []
    
    for i, (returns, times) in enumerate(zip(return_series, time_series)):
        try:
            filtered_r, filtered_t = filter_returns(
                returns, times, sample_freq, time_unit, sampling_scheme, 
                return_pandas=False
            )
            filtered_returns.append(filtered_r)
            filtered_times.append(filtered_t)
        except Exception as e:
            logger.warning(f"Failed to filter return series {i}: {str(e)}")
            # Skip this series
            continue
    
    if not filtered_returns:
        raise ValueError("All return series filtering failed")
    
    # Find the common time grid (intersection of all filtered times)
    # For simplicity, we'll use the time grid from the first series
    # and align all other series to it
    common_times = filtered_times[0]
    
    # Align all return series to the common time grid
    aligned_returns = []
    
    for i, (returns, times) in enumerate(zip(filtered_returns, filtered_times)):
        if i == 0:
            # First series is already aligned
            aligned_returns.append(returns)
        else:
            # For returns, we need to aggregate returns within each interval
            aligned_r = np.zeros_like(common_times)
            
            for j in range(len(common_times)):
                # Find the closest time point in this series
                idx = np.argmin(np.abs(times - common_times[j]))
                
                # If the closest time is within a small tolerance, use its return
                if np.abs(times[idx] - common_times[j]) < 1e-6:
                    aligned_r[j] = returns[idx]
            
            aligned_returns.append(aligned_r)
    
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
        
        # Create DataFrame with each return series as a column
        df = pd.DataFrame(np.column_stack(aligned_returns), index=times_index)
        return df, times_index
    else:
        # Return as NumPy arrays
        return np.column_stack(aligned_returns), common_times



def compute_return_statistics(
    returns: Union[np.ndarray, pd.Series],
    annualize: bool = False,
    annualization_factor: float = 252.0,
    return_dict: bool = True
) -> Union[Dict[str, float], pd.Series]:
    """
    Calculate descriptive statistics for returns.
    
    Args:
        returns: Array or Series of return data
        annualize: Whether to annualize statistics
        annualization_factor: Factor to use for annualization
        return_dict: Whether to return a dictionary (True) or pandas Series (False)
        
    Returns:
        Dictionary or Series of return statistics
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.return_filter import compute_return_statistics
        >>> returns = np.array([0.01, -0.005, 0.008, -0.002, 0.003, -0.001])
        >>> stats = compute_return_statistics(returns)
        >>> stats['mean']
        0.00216...
        >>> stats['volatility']
        0.00556...
    """
    # Convert to numpy array if not already
    returns_array = np.asarray(returns)
    
    # Validate inputs
    if returns_array.ndim != 1:
        raise ValueError("returns must be a 1D array or Series")
    
    # Compute basic statistics
    mean = np.mean(returns_array)
    median = np.median(returns_array)
    std = np.std(returns_array)
    skewness = stats.skew(returns_array)
    kurtosis = stats.kurtosis(returns_array)
    min_return = np.min(returns_array)
    max_return = np.max(returns_array)
    
    # Compute quantiles
    q1 = np.percentile(returns_array, 25)
    q3 = np.percentile(returns_array, 75)
    
    # Compute annualized statistics if requested
    if annualize:
        mean_annual = mean * annualization_factor
        volatility_annual = std * np.sqrt(annualization_factor)
    else:
        mean_annual = None
        volatility_annual = None
    
    # Compute additional statistics
    sharpe_ratio = mean / std if std > 0 else np.nan
    if annualize:
        sharpe_ratio_annual = sharpe_ratio * np.sqrt(annualization_factor)
    else:
        sharpe_ratio_annual = None
    
    # Compute normality test
    _, normality_p_value = stats.normaltest(returns_array)
    
    # Compute autocorrelation
    if len(returns_array) > 1:
        acf_lag1 = np.corrcoef(returns_array[:-1], returns_array[1:])[0, 1]
    else:
        acf_lag1 = np.nan
    
    # Create result dictionary
    result = {
        'mean': mean,
        'median': median,
        'volatility': std,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'min': min_return,
        'max': max_return,
        'q1': q1,
        'q3': q3,
        'sharpe_ratio': sharpe_ratio,
        'normality_p_value': normality_p_value,
        'autocorrelation_lag1': acf_lag1
    }
    
    # Add annualized statistics if computed
    if annualize:
        result['mean_annual'] = mean_annual
        result['volatility_annual'] = volatility_annual
        result['sharpe_ratio_annual'] = sharpe_ratio_annual
    
    # Return as dictionary or Series
    if return_dict:
        return result
    else:
        return pd.Series(result)



def detect_outliers_in_returns(
    returns: Union[np.ndarray, pd.Series],
    method: str = 'std',
    threshold: float = 3.0,
    return_indices: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Identify outliers in return series.
    
    Args:
        returns: Array or Series of return data
        method: Method for identifying outliers ('std', 'iqr', 'mad')
        threshold: Threshold for outlier detection
        return_indices: Whether to return indices of outliers
        
    Returns:
        Boolean array indicating outliers (True) or tuple of (outlier_mask, outlier_indices)
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.return_filter import detect_outliers_in_returns
        >>> returns = np.array([0.001, 0.002, 0.001, 0.05, -0.001, 0.002])
        >>> detect_outliers_in_returns(returns, 'std', 3.0)
        array([False, False, False,  True, False, False])
    """
    # Convert to numpy array if not already
    returns_array = np.asarray(returns)
    
    # Validate inputs
    if returns_array.ndim != 1:
        raise ValueError("returns must be a 1D array or Series")
    
    # Identify outliers using the specified method
    if method.lower() == 'std':
        # Standard deviation method
        mean = np.mean(returns_array)
        std = np.std(returns_array)
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
        outlier_mask = (returns_array < lower_bound) | (returns_array > upper_bound)
    
    elif method.lower() == 'iqr':
        # Interquartile range method
        q1, q3 = np.percentile(returns_array, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        outlier_mask = (returns_array < lower_bound) | (returns_array > upper_bound)
    
    elif method.lower() == 'mad':
        # Median absolute deviation method
        median = np.median(returns_array)
        mad = np.median(np.abs(returns_array - median))
        lower_bound = median - threshold * mad
        upper_bound = median + threshold * mad
        outlier_mask = (returns_array < lower_bound) | (returns_array > upper_bound)
    
    else:
        raise ValueError(f"Unrecognized outlier detection method: {method}. "
                         f"Supported methods are 'std', 'iqr', 'mad'.")
    
    # Return outlier mask or tuple of (mask, indices)
    if return_indices:
        outlier_indices = np.where(outlier_mask)[0]
        return outlier_mask, outlier_indices
    else:
        return outlier_mask



def visualize_returns(
    returns: Union[np.ndarray, pd.Series],
    times: Optional[Union[np.ndarray, pd.Series, pd.DatetimeIndex]] = None,
    title: str = 'Return Series',
    figsize: Tuple[int, int] = (10, 8),
    include_histogram: bool = True,
    include_acf: bool = True,
    include_stats: bool = True,
    highlight_outliers: bool = False,
    outlier_method: str = 'std',
    outlier_threshold: float = 3.0,
    save_path: Optional[str] = None
) -> Optional[Any]:
    """
    Create visualization of return characteristics.
    
    Args:
        returns: Array or Series of return data
        times: Corresponding time points (optional)
        title: Plot title
        figsize: Figure size as (width, height) in inches
        include_histogram: Whether to include return distribution histogram
        include_acf: Whether to include autocorrelation function plot
        include_stats: Whether to include summary statistics
        highlight_outliers: Whether to highlight outliers in the time series plot
        outlier_method: Method for identifying outliers ('std', 'iqr', 'mad')
        outlier_threshold: Threshold for outlier detection
        save_path: Path to save the figure (if None, figure is displayed)
        
    Returns:
        Matplotlib figure object if matplotlib is available, None otherwise
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.return_filter import visualize_returns
        >>> np.random.seed(42)
        >>> returns = np.random.normal(0, 0.01, 100)
        >>> fig = visualize_returns(returns, highlight_outliers=True)
    """
    if not HAS_MATPLOTLIB:
        logger.warning("Matplotlib is not available. Cannot create visualization.")
        return None
    
    # Convert to numpy array if not already
    returns_array = np.asarray(returns)
    
    # Validate inputs
    if returns_array.ndim != 1:
        raise ValueError("returns must be a 1D array or Series")
    
    # Determine number of subplots
    n_plots = 1  # Time series plot
    if include_histogram:
        n_plots += 1
    if include_acf:
        n_plots += 1
    
    # Create figure and subplots
    fig = plt.figure(figsize=figsize)
    
    # Determine subplot layout
    if n_plots <= 2:
        n_rows, n_cols = n_plots, 1
    else:
        n_rows, n_cols = 2, 2
    
    # Create time series plot
    ax1 = fig.add_subplot(n_rows, n_cols, 1)
    
    # Plot returns
    if times is not None:
        # Convert times to appropriate format
        if isinstance(times, pd.DatetimeIndex):
            plot_times = times
        elif isinstance(times, pd.Series) and pd.api.types.is_datetime64_dtype(times):
            plot_times = pd.DatetimeIndex(times)
        else:
            plot_times = np.asarray(times)
        
        # Plot returns vs. times
        ax1.plot(plot_times, returns_array, 'b-', alpha=0.7)
        
        # Format x-axis if datetime
        if isinstance(plot_times, pd.DatetimeIndex):
            fig.autofmt_xdate()
    else:
        # Plot returns vs. index
        ax1.plot(returns_array, 'b-', alpha=0.7)
    
    # Highlight outliers if requested
    if highlight_outliers:
        outlier_mask = detect_outliers_in_returns(returns_array, outlier_method, outlier_threshold)
        if times is not None:
            if isinstance(plot_times, pd.DatetimeIndex):
                outlier_times = plot_times[outlier_mask]
            else:
                outlier_times = plot_times[outlier_mask]
            ax1.plot(outlier_times, returns_array[outlier_mask], 'ro', label='Outliers')
        else:
            outlier_indices = np.where(outlier_mask)[0]
            ax1.plot(outlier_indices, returns_array[outlier_mask], 'ro', label='Outliers')
        
        # Add legend if outliers were found
        if np.any(outlier_mask):
            ax1.legend()
    
    # Add horizontal line at y=0
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Add labels and title
    ax1.set_title('Return Time Series')
    ax1.set_ylabel('Return')
    if times is None:
        ax1.set_xlabel('Index')
    else:
        ax1.set_xlabel('Time')
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    # Create histogram if requested
    if include_histogram:
        ax2 = fig.add_subplot(n_rows, n_cols, 2)
        
        # Plot histogram
        ax2.hist(returns_array, bins=30, alpha=0.7, density=True)
        
        # Add normal distribution curve
        x = np.linspace(min(returns_array), max(returns_array), 100)
        mean = np.mean(returns_array)
        std = np.std(returns_array)
        pdf = stats.norm.pdf(x, mean, std)
        ax2.plot(x, pdf, 'r-', label='Normal')
        
        # Add labels and title
        ax2.set_title('Return Distribution')
        ax2.set_xlabel('Return')
        ax2.set_ylabel('Density')
        ax2.legend()
        
        # Add grid
        ax2.grid(True, alpha=0.3)
    
    # Create ACF plot if requested
    if include_acf:
        ax3 = fig.add_subplot(n_rows, n_cols, 3 if n_plots > 2 else n_plots)
        
        # Compute autocorrelation
        max_lag = min(20, len(returns_array) // 5)
        acf = np.zeros(max_lag)
        
        for lag in range(max_lag):
            if lag == 0:
                acf[lag] = 1.0
            else:
                acf[lag] = np.corrcoef(returns_array[:-lag], returns_array[lag:])[0, 1]
        
        # Plot ACF
        ax3.bar(range(max_lag), acf, alpha=0.7)
        
        # Add confidence bands
        conf_level = 1.96 / np.sqrt(len(returns_array))
        ax3.axhline(y=conf_level, color='r', linestyle='--', alpha=0.5)
        ax3.axhline(y=-conf_level, color='r', linestyle='--', alpha=0.5)
        
        # Add labels and title
        ax3.set_title('Autocorrelation Function')
        ax3.set_xlabel('Lag')
        ax3.set_ylabel('ACF')
        
        # Add grid
        ax3.grid(True, alpha=0.3)
    
    # Add summary statistics if requested
    if include_stats:
        # Compute statistics
        stats_dict = compute_return_statistics(returns_array)
        
        # Create text for statistics
        stats_text = (
            f"Mean: {stats_dict['mean']:.6f}\n"
            f"Volatility: {stats_dict['volatility']:.6f}\n"
            f"Skewness: {stats_dict['skewness']:.4f}\n"
            f"Kurtosis: {stats_dict['kurtosis']:.4f}\n"
            f"Min: {stats_dict['min']:.6f}\n"
            f"Max: {stats_dict['max']:.6f}\n"
            f"Sharpe Ratio: {stats_dict['sharpe_ratio']:.4f}\n"
            f"Normality p-value: {stats_dict['normality_p_value']:.4f}\n"
            f"AC(1): {stats_dict['autocorrelation_lag1']:.4f}"
        )
        
        # Add text to figure
        if n_plots <= 2:
            # Add text to the bottom of the figure
            fig.text(0.5, 0.01, stats_text, ha='center', va='bottom', 
                     bbox=dict(facecolor='white', alpha=0.8))
        else:
            # Add text as a separate subplot
            ax4 = fig.add_subplot(n_rows, n_cols, 4)
            ax4.axis('off')
            ax4.text(0.1, 0.5, stats_text, ha='left', va='center', 
                     transform=ax4.transAxes, fontsize=10,
                     bbox=dict(facecolor='white', alpha=0.8))
    
    # Add main title
    fig.suptitle(title, fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    if include_stats and n_plots <= 2:
        plt.subplots_adjust(bottom=0.2)
    plt.subplots_adjust(top=0.9)
    
    # Save or display the figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    return fig



def infer_return_characteristics(
    returns: Union[np.ndarray, pd.Series],
    times: Optional[Union[np.ndarray, pd.Series, pd.DatetimeIndex]] = None
) -> Dict[str, Any]:
    """
    Automatically detect return properties and characteristics.
    
    Args:
        returns: Array or Series of return data
        times: Corresponding time points (optional)
        
    Returns:
        Dictionary of inferred return characteristics
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.return_filter import infer_return_characteristics
        >>> np.random.seed(42)
        >>> returns = np.random.normal(0, 0.01, 1000)
        >>> characteristics = infer_return_characteristics(returns)
        >>> characteristics['is_normal']
        True
        >>> characteristics['has_outliers']
        False
    """
    # Convert to numpy array if not already
    returns_array = np.asarray(returns)
    
    # Validate inputs
    if returns_array.ndim != 1:
        raise ValueError("returns must be a 1D array or Series")
    
    # Compute basic statistics
    stats_dict = compute_return_statistics(returns_array)
    
    # Infer return characteristics
    result = {}
    
    # Check for normality
    result['is_normal'] = stats_dict['normality_p_value'] > 0.05
    
    # Check for outliers
    outlier_mask = detect_outliers_in_returns(returns_array, 'std', 3.0)
    result['has_outliers'] = np.any(outlier_mask)
    result['outlier_percentage'] = np.sum(outlier_mask) / len(returns_array) * 100
    
    # Check for autocorrelation
    result['has_autocorrelation'] = abs(stats_dict['autocorrelation_lag1']) > 1.96 / np.sqrt(len(returns_array))
    
    # Check for volatility clustering
    squared_returns = returns_array**2
    squared_acf = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
    result['has_volatility_clustering'] = squared_acf > 1.96 / np.sqrt(len(returns_array))
    
    # Check for asymmetry
    result['is_asymmetric'] = abs(stats_dict['skewness']) > 0.5
    
    # Check for heavy tails
    result['has_heavy_tails'] = stats_dict['kurtosis'] > 1.0
    
    # Infer sampling frequency if times are provided
    if times is not None:
        if isinstance(times, pd.DatetimeIndex) or (
            isinstance(times, pd.Series) and pd.api.types.is_datetime64_dtype(times)
        ):
            # Convert to numpy array of timestamps
            if isinstance(times, pd.DatetimeIndex):
                times_array = times.astype(np.int64) / 1e9
            else:
                times_array = pd.DatetimeIndex(times).astype(np.int64) / 1e9
        else:
            times_array = np.asarray(times)
        
        # Compute average time difference
        avg_diff = np.mean(np.diff(times_array))
        
        # Infer sampling frequency
        if avg_diff < 1:
            result['sampling_frequency'] = 'sub-second'
        elif avg_diff < 60:
            result['sampling_frequency'] = f"{avg_diff:.1f} seconds"
        elif avg_diff < 3600:
            result['sampling_frequency'] = f"{avg_diff/60:.1f} minutes"
        elif avg_diff < 86400:
            result['sampling_frequency'] = f"{avg_diff/3600:.1f} hours"
        else:
            result['sampling_frequency'] = f"{avg_diff/86400:.1f} days"
    
    # Add basic statistics to result
    result['statistics'] = stats_dict
    
    return result



# ============================================================================
# Module Initialization
# ============================================================================

# Register Numba-accelerated functions if available
def _register_numba_functions() -> None:
    """
    Register Numba JIT-compiled functions for return filtering.
    
    This function is called during module initialization to register
    performance-critical functions for JIT compilation if Numba is available.
    """
    if HAS_NUMBA:
        # The functions are already decorated with @jit, so no additional
        # registration is needed here. This function is kept for consistency
        # with the module structure and potential future enhancements.
        logger.debug("Return filtering Numba JIT functions registered")
    else:
        logger.info("Numba not available. Return filtering will use pure NumPy/Pandas implementations.")


# Initialize the module
_register_numba_functions()
