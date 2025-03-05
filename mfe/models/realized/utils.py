"""
Realized Volatility Utility Functions

This module provides utility functions used by multiple realized volatility estimators,
including time conversion, filtering, preprocessing, and common calculations. It centralizes
common functionality to avoid duplication across estimator implementations and ensures
consistent behavior throughout the realized volatility module.

The module implements optimized versions of common operations using NumPy's efficient
array operations and Numba's JIT compilation for performance-critical functions.
All functions include comprehensive type hints and input validation to ensure
reliability and proper error handling.

Functions:
    align_time: Align price data to a common time grid
    detect_jumps: Detect jumps in return series using threshold methods
    noise_variance: Estimate microstructure noise variance
    sample_prices: Sample price data at regular intervals
    convert_time_units: Convert between different time units
    compute_returns: Compute returns from price series
    filter_outliers: Filter outliers from price or return series
    compute_realized_variance: Compute basic realized variance
    compute_realized_quarticity: Compute realized quarticity
    compute_optimal_sampling: Determine optimal sampling frequency
    compute_subsampled_measure: Compute subsampled realized measure
    compute_kernel_weights: Compute weights for kernel-based estimators
    compute_optimal_bandwidth: Determine optimal bandwidth for kernel estimators
    compute_bias_correction: Compute bias correction for realized measures
    compute_asymptotic_variance: Compute asymptotic variance of realized measures
    compute_confidence_intervals: Compute confidence intervals for realized measures
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

# Set up module-level logger
logger = logging.getLogger("mfe.models.realized.utils")

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
    logger.debug("Numba available for realized volatility utility acceleration")
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    HAS_NUMBA = False
    logger.info("Numba not available. Realized volatility utilities will use pure NumPy implementations.")


# ============================================================================
# Time Conversion Functions
# ============================================================================

def seconds2unit(seconds: np.ndarray, unit: str = 'seconds') -> np.ndarray:
    """
    Convert time in seconds to the specified unit.
    
    Args:
        seconds: Array of time points in seconds
        unit: Target time unit ('seconds', 'minutes', 'hours', 'days')
        
    Returns:
        Array of time points in the specified unit
        
    Raises:
        ValueError: If unit is not recognized
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.utils import seconds2unit
        >>> seconds = np.array([60, 120, 180, 240])
        >>> seconds2unit(seconds, 'minutes')
        array([1., 2., 3., 4.])
    """
    seconds = np.asarray(seconds)
    
    if unit.lower() == 'seconds':
        return seconds
    elif unit.lower() == 'minutes':
        return seconds / 60.0
    elif unit.lower() == 'hours':
        return seconds / 3600.0
    elif unit.lower() == 'days':
        return seconds / 86400.0
    else:
        raise ValueError(f"Unrecognized time unit: {unit}. "
                         f"Supported units are 'seconds', 'minutes', 'hours', 'days'.")


def unit2seconds(time_points: np.ndarray, unit: str = 'seconds') -> np.ndarray:
    """
    Convert time from the specified unit to seconds.
    
    Args:
        time_points: Array of time points in the specified unit
        unit: Source time unit ('seconds', 'minutes', 'hours', 'days')
        
    Returns:
        Array of time points in seconds
        
    Raises:
        ValueError: If unit is not recognized
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.utils import unit2seconds
        >>> minutes = np.array([1, 2, 3, 4])
        >>> unit2seconds(minutes, 'minutes')
        array([ 60., 120., 180., 240.])
    """
    time_points = np.asarray(time_points)
    
    if unit.lower() == 'seconds':
        return time_points
    elif unit.lower() == 'minutes':
        return time_points * 60.0
    elif unit.lower() == 'hours':
        return time_points * 3600.0
    elif unit.lower() == 'days':
        return time_points * 86400.0
    else:
        raise ValueError(f"Unrecognized time unit: {unit}. "
                         f"Supported units are 'seconds', 'minutes', 'hours', 'days'.")


def wall2seconds(wall_times: np.ndarray, base_time: Optional[float] = None) -> np.ndarray:
    """
    Convert wall clock times to seconds since a base time.
    
    Args:
        wall_times: Array of wall clock times (as timestamps or datetime objects)
        base_time: Base time to measure seconds from (default: first time point)
        
    Returns:
        Array of time points in seconds since base_time
        
    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from mfe.models.realized.utils import wall2seconds
        >>> times = pd.to_datetime(['2023-01-01 09:30:00', '2023-01-01 09:31:00', 
        ...                         '2023-01-01 09:32:30', '2023-01-01 09:35:00'])
        >>> wall2seconds(times)
        array([  0.,  60., 150., 300.])
    """
    # Convert to pandas datetime if not already
    if not isinstance(wall_times, pd.DatetimeIndex):
        try:
            wall_times = pd.DatetimeIndex(wall_times)
        except Exception as e:
            raise ValueError(f"Could not convert wall_times to datetime: {e}")
    
    # Convert to Unix timestamps (seconds since 1970-01-01)
    timestamps = wall_times.astype('int64') / 1e9
    
    # Determine base time if not provided
    if base_time is None:
        base_time = timestamps[0]
    
    # Compute seconds since base time
    seconds = timestamps - base_time
    
    return seconds


def seconds2wall(seconds: np.ndarray, base_time: Union[str, pd.Timestamp, np.datetime64]) -> pd.DatetimeIndex:
    """
    Convert seconds to wall clock times.
    
    Args:
        seconds: Array of time points in seconds since base_time
        base_time: Base time to add seconds to
        
    Returns:
        Array of wall clock times as pandas DatetimeIndex
        
    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from mfe.models.realized.utils import seconds2wall
        >>> seconds = np.array([0, 60, 150, 300])
        >>> seconds2wall(seconds, '2023-01-01 09:30:00')
        DatetimeIndex(['2023-01-01 09:30:00', '2023-01-01 09:31:00',
                       '2023-01-01 09:32:30', '2023-01-01 09:35:00'],
                      dtype='datetime64[ns]', freq=None)
    """
    # Convert base_time to pandas Timestamp if not already
    if not isinstance(base_time, pd.Timestamp):
        base_time = pd.Timestamp(base_time)
    
    # Convert base_time to Unix timestamp (seconds since 1970-01-01)
    base_timestamp = base_time.timestamp()
    
    # Add seconds to base timestamp
    timestamps = base_timestamp + np.asarray(seconds)
    
    # Convert back to pandas DatetimeIndex
    wall_times = pd.DatetimeIndex(pd.to_datetime(timestamps, unit='s'))
    
    return wall_times


def unit2wall(time_points: np.ndarray, unit: str, base_time: Union[str, pd.Timestamp, np.datetime64]) -> pd.DatetimeIndex:
    """
    Convert time from the specified unit to wall clock times.
    
    Args:
        time_points: Array of time points in the specified unit
        unit: Source time unit ('seconds', 'minutes', 'hours', 'days')
        base_time: Base time to add time points to
        
    Returns:
        Array of wall clock times as pandas DatetimeIndex
        
    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from mfe.models.realized.utils import unit2wall
        >>> minutes = np.array([0, 1, 2.5, 5])
        >>> unit2wall(minutes, 'minutes', '2023-01-01 09:30:00')
        DatetimeIndex(['2023-01-01 09:30:00', '2023-01-01 09:31:00',
                       '2023-01-01 09:32:30', '2023-01-01 09:35:00'],
                      dtype='datetime64[ns]', freq=None)
    """
    # Convert to seconds
    seconds = unit2seconds(time_points, unit)
    
    # Convert seconds to wall clock times
    return seconds2wall(seconds, base_time)


def wall2unit(wall_times: np.ndarray, unit: str, base_time: Optional[Union[str, pd.Timestamp, np.datetime64]] = None) -> np.ndarray:
    """
    Convert wall clock times to the specified unit.
    
    Args:
        wall_times: Array of wall clock times (as timestamps or datetime objects)
        unit: Target time unit ('seconds', 'minutes', 'hours', 'days')
        base_time: Base time to measure from (default: first time point)
        
    Returns:
        Array of time points in the specified unit
        
    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from mfe.models.realized.utils import wall2unit
        >>> times = pd.to_datetime(['2023-01-01 09:30:00', '2023-01-01 09:31:00', 
        ...                         '2023-01-01 09:32:30', '2023-01-01 09:35:00'])
        >>> wall2unit(times, 'minutes')
        array([0. , 1. , 2.5, 5. ])
    """
    # Convert to seconds
    seconds = wall2seconds(wall_times, None if base_time is None else pd.Timestamp(base_time).timestamp())
    
    # Convert seconds to the specified unit
    return seconds2unit(seconds, unit)


# ============================================================================
# Data Alignment and Preprocessing Functions
# ============================================================================

def align_time(prices: np.ndarray, times: np.ndarray, target_times: np.ndarray, 
               method: str = 'previous') -> np.ndarray:
    """
    Align price data to a common time grid.
    
    Args:
        prices: Array of price data
        times: Array of time points corresponding to prices
        target_times: Target time grid to align prices to
        method: Interpolation method ('previous', 'linear', 'cubic')
        
    Returns:
        Array of prices aligned to target_times
        
    Raises:
        ValueError: If method is not recognized or if inputs have invalid dimensions
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.utils import align_time
        >>> prices = np.array([100, 101, 102, 103])
        >>> times = np.array([0, 60, 120, 240])
        >>> target_times = np.array([0, 60, 180, 240])
        >>> align_time(prices, times, target_times, 'linear')
        array([100., 101., 102.5, 103.])
    """
    # Convert inputs to numpy arrays
    prices = np.asarray(prices)
    times = np.asarray(times)
    target_times = np.asarray(target_times)
    
    # Validate inputs
    if prices.ndim != 1:
        raise ValueError("prices must be a 1D array")
    if times.ndim != 1:
        raise ValueError("times must be a 1D array")
    if target_times.ndim != 1:
        raise ValueError("target_times must be a 1D array")
    if len(prices) != len(times):
        raise ValueError(f"prices length ({len(prices)}) must match times length ({len(times)})")
    
    # Check if times are sorted
    if not np.all(np.diff(times) >= 0):
        raise ValueError("times must be monotonically increasing")
    if not np.all(np.diff(target_times) >= 0):
        raise ValueError("target_times must be monotonically increasing")
    
    # Check if target_times are within the range of times
    if target_times[0] < times[0] or target_times[-1] > times[-1]:
        logger.warning("target_times extend beyond the range of times. "
                      "Extrapolation may produce unreliable results.")
    
    # Align prices to target_times using the specified method
    if method.lower() == 'previous':
        # Previous value interpolation (forward fill)
        aligned_prices = np.zeros_like(target_times, dtype=float)
        for i, t in enumerate(target_times):
            # Find the index of the last time point less than or equal to t
            idx = np.searchsorted(times, t, side='right') - 1
            # If t is before the first time point, use the first price
            if idx < 0:
                aligned_prices[i] = prices[0]
            else:
                aligned_prices[i] = prices[idx]
    
    elif method.lower() == 'linear':
        # Linear interpolation
        aligned_prices = np.interp(target_times, times, prices)
    
    elif method.lower() == 'cubic':
        # Cubic spline interpolation
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(times, prices)
        aligned_prices = cs(target_times)
    
    else:
        raise ValueError(f"Unrecognized interpolation method: {method}. "
                         f"Supported methods are 'previous', 'linear', 'cubic'.")
    
    return aligned_prices


@jit(nopython=True, cache=True)
def _compute_returns_numba(prices: np.ndarray, return_type: str = 'log') -> np.ndarray:
    """
    Numba-accelerated implementation of return computation.
    
    Args:
        prices: Array of price data
        return_type: Type of returns to compute ('log', 'simple')
        
    Returns:
        Array of returns
    """
    n = len(prices)
    returns = np.empty(n - 1, dtype=np.float64)
    
    if return_type == 'log':
        for i in range(n - 1):
            returns[i] = np.log(prices[i + 1]) - np.log(prices[i])
    else:  # 'simple'
        for i in range(n - 1):
            returns[i] = (prices[i + 1] - prices[i]) / prices[i]
    
    return returns


def compute_returns(prices: np.ndarray, return_type: str = 'log') -> np.ndarray:
    """
    Compute returns from price series.
    
    Args:
        prices: Array of price data
        return_type: Type of returns to compute ('log', 'simple')
        
    Returns:
        Array of returns
        
    Raises:
        ValueError: If return_type is not recognized or if prices contain non-positive values
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.utils import compute_returns
        >>> prices = np.array([100, 101, 102, 103])
        >>> compute_returns(prices, 'log')
        array([0.00995033, 0.00985222, 0.00975709])
        >>> compute_returns(prices, 'simple')
        array([0.01, 0.00990099, 0.00980392])
    """
    # Convert to numpy array if not already
    prices = np.asarray(prices)
    
    # Validate inputs
    if prices.ndim != 1:
        raise ValueError("prices must be a 1D array")
    if len(prices) < 2:
        raise ValueError("prices must have at least 2 elements to compute returns")
    
    # Check for non-positive prices
    if np.any(prices <= 0):
        raise ValueError("prices must be positive for return calculation")
    
    # Validate return_type
    if return_type.lower() not in ['log', 'simple']:
        raise ValueError(f"Unrecognized return_type: {return_type}. "
                         f"Supported types are 'log', 'simple'.")
    
    # Use Numba-accelerated implementation if available
    if HAS_NUMBA:
        return _compute_returns_numba(prices, return_type.lower())
    
    # Pure NumPy implementation
    if return_type.lower() == 'log':
        return np.diff(np.log(prices))
    else:  # 'simple'
        return np.diff(prices) / prices[:-1]


def filter_outliers(data: np.ndarray, method: str = 'std', threshold: float = 3.0) -> np.ndarray:
    """
    Filter outliers from price or return series.
    
    Args:
        data: Array of price or return data
        method: Method for identifying outliers ('std', 'iqr', 'mad')
        threshold: Threshold for outlier detection
        
    Returns:
        Boolean array indicating non-outliers (True) and outliers (False)
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.utils import filter_outliers
        >>> returns = np.array([0.001, 0.002, 0.001, 0.05, -0.001, 0.002])
        >>> filter_outliers(returns, 'std', 3.0)
        array([ True,  True,  True, False,  True,  True])
    """
    # Convert to numpy array if not already
    data = np.asarray(data)
    
    # Validate inputs
    if data.ndim != 1:
        raise ValueError("data must be a 1D array")
    
    # Identify outliers using the specified method
    if method.lower() == 'std':
        # Standard deviation method
        mean = np.mean(data)
        std = np.std(data)
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
        mask = (data >= lower_bound) & (data <= upper_bound)
    
    elif method.lower() == 'iqr':
        # Interquartile range method
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        mask = (data >= lower_bound) & (data <= upper_bound)
    
    elif method.lower() == 'mad':
        # Median absolute deviation method
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        lower_bound = median - threshold * mad
        upper_bound = median + threshold * mad
        mask = (data >= lower_bound) & (data <= upper_bound)
    
    else:
        raise ValueError(f"Unrecognized outlier detection method: {method}. "
                         f"Supported methods are 'std', 'iqr', 'mad'.")
    
    return mask


def sample_prices(prices: np.ndarray, times: np.ndarray, 
                 sample_freq: Union[str, float, int], 
                 time_unit: str = 'seconds',
                 method: str = 'previous') -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample price data at regular intervals.
    
    Args:
        prices: Array of price data
        times: Array of time points corresponding to prices
        sample_freq: Sampling frequency (as string like '5min' or numeric value in time_unit)
        time_unit: Time unit for numeric sample_freq ('seconds', 'minutes', 'hours', 'days')
        method: Interpolation method ('previous', 'linear', 'cubic')
        
    Returns:
        Tuple of (sampled_prices, sampled_times)
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.utils import sample_prices
        >>> prices = np.array([100, 101, 102, 103, 104])
        >>> times = np.array([0, 30, 70, 150, 210])  # seconds
        >>> sampled_prices, sampled_times = sample_prices(prices, times, 60, 'seconds')
        >>> sampled_times
        array([  0.,  60., 120., 180.])
        >>> sampled_prices
        array([100., 101., 103., 103.])
    """
    # Convert inputs to numpy arrays
    prices = np.asarray(prices)
    times = np.asarray(times)
    
    # Validate inputs
    if prices.ndim != 1:
        raise ValueError("prices must be a 1D array")
    if times.ndim != 1:
        raise ValueError("times must be a 1D array")
    if len(prices) != len(times):
        raise ValueError(f"prices length ({len(prices)}) must match times length ({len(times)})")
    
    # Check if times are sorted
    if not np.all(np.diff(times) >= 0):
        raise ValueError("times must be monotonically increasing")
    
    # Parse sampling frequency
    if isinstance(sample_freq, str):
        # Convert string frequency to seconds
        freq_map = {
            's': 1,
            'sec': 1,
            'second': 1,
            'seconds': 1,
            'm': 60,
            'min': 60,
            'minute': 60,
            'minutes': 60,
            'h': 3600,
            'hour': 3600,
            'hours': 3600,
            'd': 86400,
            'day': 86400,
            'days': 86400
        }
        
        # Extract numeric value and unit
        import re
        match = re.match(r'(\d+)([a-zA-Z]+)', sample_freq)
        if match:
            value, unit = match.groups()
            if unit.lower() in freq_map:
                freq_seconds = int(value) * freq_map[unit.lower()]
            else:
                raise ValueError(f"Unrecognized frequency unit: {unit}")
        else:
            raise ValueError(f"Could not parse frequency string: {sample_freq}")
        
        # Convert to the specified time unit
        if time_unit.lower() == 'seconds':
            freq = freq_seconds
        elif time_unit.lower() == 'minutes':
            freq = freq_seconds / 60.0
        elif time_unit.lower() == 'hours':
            freq = freq_seconds / 3600.0
        elif time_unit.lower() == 'days':
            freq = freq_seconds / 86400.0
        else:
            raise ValueError(f"Unrecognized time unit: {time_unit}")
    else:
        # Numeric frequency in the specified time unit
        freq = float(sample_freq)
    
    # Generate regular time grid
    start_time = times[0]
    end_time = times[-1]
    num_points = int(np.floor((end_time - start_time) / freq)) + 1
    sampled_times = np.linspace(start_time, start_time + (num_points - 1) * freq, num_points)
    
    # Align prices to the regular time grid
    sampled_prices = align_time(prices, times, sampled_times, method)
    
    return sampled_prices, sampled_times


# ============================================================================
# Jump Detection and Noise Estimation Functions
# ============================================================================

@jit(nopython=True, cache=True)
def _detect_jumps_numba(returns: np.ndarray, threshold_multiplier: float = 3.0) -> Tuple[np.ndarray, float]:
    """
    Numba-accelerated implementation of jump detection.
    
    Args:
        returns: Array of returns
        threshold_multiplier: Multiplier for the threshold
        
    Returns:
        Tuple of (jump_indicators, threshold)
    """
    n = len(returns)
    jump_indicators = np.zeros(n, dtype=np.bool_)
    
    # Compute local volatility using bipower variation
    abs_returns = np.abs(returns)
    bipower = np.empty(n - 1, dtype=np.float64)
    
    for i in range(n - 1):
        bipower[i] = abs_returns[i] * abs_returns[i + 1]
    
    # Compute mean bipower variation
    mean_bipower = np.mean(bipower) * np.pi / 2
    
    # Compute threshold
    threshold = threshold_multiplier * np.sqrt(mean_bipower)
    
    # Detect jumps
    for i in range(n):
        if np.abs(returns[i]) > threshold:
            jump_indicators[i] = True
    
    return jump_indicators, threshold


def detect_jumps(returns: np.ndarray, threshold_multiplier: float = 3.0, 
                method: str = 'bipower') -> Tuple[np.ndarray, float]:
    """
    Detect jumps in return series using threshold methods.
    
    Args:
        returns: Array of returns
        threshold_multiplier: Multiplier for the threshold
        method: Method for estimating volatility ('bipower', 'quarticity', 'garch')
        
    Returns:
        Tuple of (jump_indicators, threshold)
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.utils import detect_jumps
        >>> np.random.seed(42)
        >>> returns = np.random.normal(0, 0.01, 100)
        >>> returns[50] = 0.1  # Add a jump
        >>> jump_indicators, threshold = detect_jumps(returns)
        >>> jump_indicators[50]
        True
        >>> np.sum(jump_indicators)  # Number of detected jumps
        1
    """
    # Convert to numpy array if not already
    returns = np.asarray(returns)
    
    # Validate inputs
    if returns.ndim != 1:
        raise ValueError("returns must be a 1D array")
    if threshold_multiplier <= 0:
        raise ValueError("threshold_multiplier must be positive")
    
    # Use Numba-accelerated implementation for bipower method if available
    if method.lower() == 'bipower' and HAS_NUMBA:
        return _detect_jumps_numba(returns, threshold_multiplier)
    
    # Compute local volatility using the specified method
    if method.lower() == 'bipower':
        # Bipower variation method
        abs_returns = np.abs(returns)
        bipower = abs_returns[:-1] * abs_returns[1:]
        mean_bipower = np.mean(bipower) * np.pi / 2
        threshold = threshold_multiplier * np.sqrt(mean_bipower)
    
    elif method.lower() == 'quarticity':
        # Quarticity method
        quarticity = np.mean(returns**4) * (np.pi / 2)
        threshold = threshold_multiplier * quarticity**(1/4)
    
    elif method.lower() == 'garch':
        # GARCH method (requires statsmodels)
        try:
            from statsmodels.tsa.api import GARCH
            model = GARCH(returns, vol='GARCH', p=1, q=1)
            result = model.fit(disp='off')
            conditional_vol = result.conditional_volatility
            threshold = threshold_multiplier * conditional_vol
        except ImportError:
            logger.warning("statsmodels not available. Falling back to bipower method.")
            return detect_jumps(returns, threshold_multiplier, 'bipower')
    
    else:
        raise ValueError(f"Unrecognized volatility estimation method: {method}. "
                         f"Supported methods are 'bipower', 'quarticity', 'garch'.")
    
    # Detect jumps
    if method.lower() == 'garch':
        # For GARCH, threshold is time-varying
        jump_indicators = np.abs(returns) > threshold
    else:
        # For other methods, threshold is constant
        jump_indicators = np.abs(returns) > threshold
    
    return jump_indicators, threshold if np.isscalar(threshold) else np.mean(threshold)


def noise_variance(returns: np.ndarray, method: str = 'autocovariance') -> float:
    """
    Estimate microstructure noise variance.
    
    Args:
        returns: Array of returns
        method: Method for estimating noise variance ('autocovariance', 'first_order', 'signature')
        
    Returns:
        Estimated noise variance
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.utils import noise_variance
        >>> np.random.seed(42)
        >>> returns = np.random.normal(0, 0.01, 1000)
        >>> noise_var = noise_variance(returns)
        >>> noise_var
        0.00005...
    """
    # Convert to numpy array if not already
    returns = np.asarray(returns)
    
    # Validate inputs
    if returns.ndim != 1:
        raise ValueError("returns must be a 1D array")
    
    # Estimate noise variance using the specified method
    if method.lower() == 'autocovariance':
        # Autocovariance method
        # Noise variance is -0.5 * first-order autocovariance
        acov = np.mean(returns[:-1] * returns[1:])
        noise_var = -0.5 * acov
    
        # If noise_var is negative, use alternative method
        if noise_var <= 0:
            logger.warning("Autocovariance method produced non-positive noise variance. "
                          "Falling back to first_order method.")
            return noise_variance(returns, 'first_order')
    
    elif method.lower() == 'first_order':
        # First-order method
        # Noise variance is 0.5 * mean squared returns
        noise_var = 0.5 * np.mean(returns**2)
    
    elif method.lower() == 'signature':
        # Signature plot method
        # Compute realized variance at different sampling frequencies
        n = len(returns)
        max_lag = min(20, n // 10)
        lags = np.arange(1, max_lag + 1)
        rv = np.zeros(max_lag)
        
        for i, lag in enumerate(lags):
            # Skip every lag-th observation
            sampled_returns = returns[::lag]
            rv[i] = np.sum(sampled_returns**2)
        
        # Fit linear model: RV(h) = IV + 2*q*h
        # where h is the sampling interval, IV is integrated variance, q is noise variance
        from scipy.stats import linregress
        slope, intercept, _, _, _ = linregress(lags, rv)
        
        # Noise variance is half the slope
        noise_var = slope / 2
        
        # If noise_var is negative, use alternative method
        if noise_var <= 0:
            logger.warning("Signature plot method produced non-positive noise variance. "
                          "Falling back to first_order method.")
            return noise_variance(returns, 'first_order')
    
    else:
        raise ValueError(f"Unrecognized noise estimation method: {method}. "
                         f"Supported methods are 'autocovariance', 'first_order', 'signature'.")
    
    return noise_var


# ============================================================================
# Realized Measure Computation Functions
# ============================================================================

@jit(nopython=True, cache=True)
def _compute_realized_variance_numba(returns: np.ndarray) -> float:
    """
    Numba-accelerated implementation of realized variance computation.
    
    Args:
        returns: Array of returns
        
    Returns:
        Realized variance
    """
    return np.sum(returns**2)


def compute_realized_variance(returns: np.ndarray) -> float:
    """
    Compute basic realized variance.
    
    Args:
        returns: Array of returns
        
    Returns:
        Realized variance
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.utils import compute_realized_variance
        >>> returns = np.array([0.01, -0.005, 0.008, -0.002])
        >>> compute_realized_variance(returns)
        0.000189
    """
    # Convert to numpy array if not already
    returns = np.asarray(returns)
    
    # Validate inputs
    if returns.ndim != 1:
        raise ValueError("returns must be a 1D array")
    
    # Use Numba-accelerated implementation if available
    if HAS_NUMBA:
        return _compute_realized_variance_numba(returns)
    
    # Pure NumPy implementation
    return np.sum(returns**2)


@jit(nopython=True, cache=True)
def _compute_realized_quarticity_numba(returns: np.ndarray) -> float:
    """
    Numba-accelerated implementation of realized quarticity computation.
    
    Args:
        returns: Array of returns
        
    Returns:
        Realized quarticity
    """
    n = len(returns)
    return (n / 3) * np.sum(returns**4)


def compute_realized_quarticity(returns: np.ndarray) -> float:
    """
    Compute realized quarticity.
    
    Args:
        returns: Array of returns
        
    Returns:
        Realized quarticity
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.utils import compute_realized_quarticity
        >>> returns = np.array([0.01, -0.005, 0.008, -0.002])
        >>> compute_realized_quarticity(returns)
        1.3266666666666667e-07
    """
    # Convert to numpy array if not already
    returns = np.asarray(returns)
    
    # Validate inputs
    if returns.ndim != 1:
        raise ValueError("returns must be a 1D array")
    
    # Use Numba-accelerated implementation if available
    if HAS_NUMBA:
        return _compute_realized_quarticity_numba(returns)
    
    # Pure NumPy implementation
    n = len(returns)
    return (n / 3) * np.sum(returns**4)


def compute_optimal_sampling(prices: np.ndarray, times: np.ndarray, 
                           method: str = 'signature', 
                           max_points: int = 20) -> float:
    """
    Determine optimal sampling frequency for realized volatility estimation.
    
    Args:
        prices: Array of price data
        times: Array of time points corresponding to prices
        method: Method for determining optimal sampling ('signature', 'mse', 'scale')
        max_points: Maximum number of sampling points to consider
        
    Returns:
        Optimal sampling frequency in the same units as times
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.utils import compute_optimal_sampling
        >>> np.random.seed(42)
        >>> prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 1000)))
        >>> times = np.arange(1000)
        >>> optimal_freq = compute_optimal_sampling(prices, times)
        >>> optimal_freq
        10...
    """
    # Convert inputs to numpy arrays
    prices = np.asarray(prices)
    times = np.asarray(times)
    
    # Validate inputs
    if prices.ndim != 1:
        raise ValueError("prices must be a 1D array")
    if times.ndim != 1:
        raise ValueError("times must be a 1D array")
    if len(prices) != len(times):
        raise ValueError(f"prices length ({len(prices)}) must match times length ({len(times)})")
    
    # Compute returns
    returns = compute_returns(prices, 'log')
    
    # Determine optimal sampling frequency using the specified method
    if method.lower() == 'signature':
        # Signature plot method
        # Compute realized variance at different sampling frequencies
        n = len(returns)
        max_lag = min(max_points, n // 10)
        lags = np.arange(1, max_lag + 1)
        rv = np.zeros(max_lag)
        
        for i, lag in enumerate(lags):
            # Skip every lag-th observation
            sampled_returns = returns[::lag]
            rv[i] = np.sum(sampled_returns**2)
        
        # Compute average time interval for each lag
        time_intervals = np.zeros(max_lag)
        for i, lag in enumerate(lags):
            sampled_times = times[::lag]
            time_intervals[i] = np.mean(np.diff(sampled_times))
        
        # Find the point where the signature plot stabilizes
        # (where the derivative of RV with respect to time interval becomes small)
        derivatives = np.diff(rv) / np.diff(time_intervals)
        
        # Normalize derivatives
        norm_derivatives = np.abs(derivatives) / np.max(np.abs(derivatives))
        
        # Find the first point where the normalized derivative is below a threshold
        threshold = 0.1
        stable_points = np.where(norm_derivatives < threshold)[0]
        
        if len(stable_points) > 0:
            optimal_idx = stable_points[0] + 1  # +1 because derivatives has length max_lag-1
            optimal_freq = time_intervals[optimal_idx]
        else:
            # If no stable point is found, use the largest lag
            optimal_freq = time_intervals[-1]
    
    elif method.lower() == 'mse':
        # Mean squared error method
        # Estimate noise variance
        noise_var = noise_variance(returns)
        
        # Estimate integrated variance (true volatility)
        # using a low-frequency estimate (e.g., 5-minute returns)
        target_interval = (times[-1] - times[0]) / 100  # Approximately 1% of total time
        sampled_prices, sampled_times = sample_prices(prices, times, target_interval)
        sampled_returns = compute_returns(sampled_prices, 'log')
        integrated_var = compute_realized_variance(sampled_returns)
        
        # Compute optimal sampling frequency
        # Optimal sampling minimizes MSE = (bias^2 + variance)
        # bias = 2*noise_var/dt, variance = 2*integrated_var*dt
        # Optimal dt = sqrt(2*noise_var/integrated_var)
        if integrated_var > 0:
            optimal_dt = np.sqrt(2 * noise_var / integrated_var)
            
            # Convert to sampling frequency
            avg_dt = np.mean(np.diff(times))
            optimal_freq = optimal_dt / avg_dt
            
            # Ensure optimal_freq is at least 1
            optimal_freq = max(1, optimal_freq)
        else:
            # If integrated_var is zero, use a default value
            optimal_freq = 10
    
    elif method.lower() == 'scale':
        # Scale-based method
        # Compute realized variance at different scales
        n = len(returns)
        max_scale = min(max_points, n // 10)
        scales = np.arange(1, max_scale + 1)
        rv = np.zeros(max_scale)
        
        for i, scale in enumerate(scales):
            # Compute returns at different scales
            scaled_returns = np.zeros(n - scale)
            for j in range(n - scale):
                scaled_returns[j] = np.sum(returns[j:j+scale])
            
            # Compute realized variance
            rv[i] = np.sum(scaled_returns**2) / (n - scale)
        
        # Compute average time interval for each scale
        time_intervals = np.zeros(max_scale)
        for i, scale in enumerate(scales):
            time_intervals[i] = np.mean(times[scale:] - times[:-scale])
        
        # Find the scale where the realized variance stabilizes
        # (where the second derivative of RV with respect to scale becomes small)
        first_derivatives = np.diff(rv) / np.diff(scales)
        second_derivatives = np.diff(first_derivatives) / np.diff(scales[:-1])
        
        # Normalize second derivatives
        norm_second_derivatives = np.abs(second_derivatives) / np.max(np.abs(second_derivatives))
        
        # Find the first point where the normalized second derivative is below a threshold
        threshold = 0.1
        stable_points = np.where(norm_second_derivatives < threshold)[0]
        
        if len(stable_points) > 0:
            optimal_idx = stable_points[0] + 2  # +2 because second_derivatives has length max_scale-2
            optimal_freq = time_intervals[optimal_idx]
        else:
            # If no stable point is found, use the largest scale
            optimal_freq = time_intervals[-1]
    
    else:
        raise ValueError(f"Unrecognized optimal sampling method: {method}. "
                         f"Supported methods are 'signature', 'mse', 'scale'.")
    
    return optimal_freq


@jit(nopython=True, cache=True)
def _compute_subsampled_measure_numba(returns: np.ndarray, subsample_factor: int) -> float:
    """
    Numba-accelerated implementation of subsampled realized measure computation.
    
    Args:
        returns: Array of returns
        subsample_factor: Number of subsamples
        
    Returns:
        Subsampled realized measure
    """
    n = len(returns)
    subsampled_rv = 0.0
    
    for i in range(subsample_factor):
        # Extract i-th subsample
        subsample = returns[i::subsample_factor]
        # Compute realized variance for this subsample
        subsample_rv = np.sum(subsample**2)
        # Scale by the number of observations
        scaled_rv = subsample_rv * (n / len(subsample))
        # Add to the total
        subsampled_rv += scaled_rv
    
    # Average across subsamples
    return subsampled_rv / subsample_factor


def compute_subsampled_measure(returns: np.ndarray, subsample_factor: int) -> float:
    """
    Compute subsampled realized measure.
    
    Args:
        returns: Array of returns
        subsample_factor: Number of subsamples
        
    Returns:
        Subsampled realized measure
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.utils import compute_subsampled_measure
        >>> returns = np.array([0.01, -0.005, 0.008, -0.002, 0.003, -0.001])
        >>> compute_subsampled_measure(returns, 3)
        0.000183...
    """
    # Convert to numpy array if not already
    returns = np.asarray(returns)
    
    # Validate inputs
    if returns.ndim != 1:
        raise ValueError("returns must be a 1D array")
    if not isinstance(subsample_factor, int) or subsample_factor < 1:
        raise ValueError("subsample_factor must be a positive integer")
    if subsample_factor > len(returns):
        raise ValueError(f"subsample_factor ({subsample_factor}) must be less than "
                         f"or equal to the length of returns ({len(returns)})")
    
    # Use Numba-accelerated implementation if available
    if HAS_NUMBA:
        return _compute_subsampled_measure_numba(returns, subsample_factor)
    
    # Pure NumPy implementation
    n = len(returns)
    subsampled_rv = 0.0
    
    for i in range(subsample_factor):
        # Extract i-th subsample
        subsample = returns[i::subsample_factor]
        # Compute realized variance for this subsample
        subsample_rv = np.sum(subsample**2)
        # Scale by the number of observations
        scaled_rv = subsample_rv * (n / len(subsample))
        # Add to the total
        subsampled_rv += scaled_rv
    
    # Average across subsamples
    return subsampled_rv / subsample_factor


def compute_kernel_weights(n: int, kernel_type: str = 'bartlett', 
                         bandwidth: Optional[float] = None) -> np.ndarray:
    """
    Compute weights for kernel-based estimators.
    
    Args:
        n: Number of weights to compute
        kernel_type: Type of kernel ('bartlett', 'parzen', 'tukey-hanning', 'quadratic')
        bandwidth: Bandwidth parameter (default: sqrt(n))
        
    Returns:
        Array of kernel weights
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.utils import compute_kernel_weights
        >>> weights = compute_kernel_weights(5, 'bartlett')
        >>> weights
        array([1.        , 0.77777778, 0.55555556, 0.33333333, 0.11111111])
    """
    # Validate inputs
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    
    # Set default bandwidth if not provided
    if bandwidth is None:
        bandwidth = np.sqrt(n)
    
    # Compute kernel weights
    weights = np.zeros(n)
    
    if kernel_type.lower() == 'bartlett':
        # Bartlett kernel (linear)
        for i in range(n):
            if i <= bandwidth:
                weights[i] = 1.0 - i / (bandwidth + 1)
            else:
                weights[i] = 0.0
    
    elif kernel_type.lower() == 'parzen':
        # Parzen kernel
        for i in range(n):
            x = i / bandwidth
            if x <= 0.5:
                weights[i] = 1.0 - 6.0 * x**2 + 6.0 * x**3
            elif x <= 1.0:
                weights[i] = 2.0 * (1.0 - x)**3
            else:
                weights[i] = 0.0
    
    elif kernel_type.lower() in ['tukey-hanning', 'tukey', 'hanning']:
        # Tukey-Hanning kernel
        for i in range(n):
            x = i / bandwidth
            if x <= 1.0:
                weights[i] = 0.5 * (1.0 + np.cos(np.pi * x))
            else:
                weights[i] = 0.0
    
    elif kernel_type.lower() == 'quadratic':
        # Quadratic kernel
        for i in range(n):
            x = i / bandwidth
            if x <= 1.0:
                weights[i] = (1.0 - x**2)**2
            else:
                weights[i] = 0.0
    
    else:
        raise ValueError(f"Unrecognized kernel type: {kernel_type}. "
                         f"Supported types are 'bartlett', 'parzen', 'tukey-hanning', 'quadratic'.")
    
    return weights


def compute_optimal_bandwidth(returns: np.ndarray, kernel_type: str = 'bartlett') -> float:
    """
    Determine optimal bandwidth for kernel estimators.
    
    Args:
        returns: Array of returns
        kernel_type: Type of kernel ('bartlett', 'parzen', 'tukey-hanning', 'quadratic')
        
    Returns:
        Optimal bandwidth
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.utils import compute_optimal_bandwidth
        >>> np.random.seed(42)
        >>> returns = np.random.normal(0, 0.01, 1000)
        >>> optimal_bw = compute_optimal_bandwidth(returns)
        >>> optimal_bw
        4...
    """
    # Convert to numpy array if not already
    returns = np.asarray(returns)
    
    # Validate inputs
    if returns.ndim != 1:
        raise ValueError("returns must be a 1D array")
    
    # Compute autocorrelations
    n = len(returns)
    max_lag = min(30, n // 4)
    acf = np.zeros(max_lag)
    
    for lag in range(max_lag):
        acf[lag] = np.corrcoef(returns[:-lag-1], returns[lag+1:])[0, 1]
    
    # Determine optimal bandwidth based on kernel type
    if kernel_type.lower() == 'bartlett':
        # For Bartlett kernel, optimal bandwidth is approximately n^(1/3)
        optimal_bw = n**(1/3)
    
    elif kernel_type.lower() == 'parzen':
        # For Parzen kernel, optimal bandwidth is approximately n^(1/5)
        optimal_bw = n**(1/5)
    
    elif kernel_type.lower() in ['tukey-hanning', 'tukey', 'hanning']:
        # For Tukey-Hanning kernel, optimal bandwidth is approximately n^(1/5)
        optimal_bw = n**(1/5)
    
    elif kernel_type.lower() == 'quadratic':
        # For Quadratic kernel, optimal bandwidth is approximately n^(1/5)
        optimal_bw = n**(1/5)
    
    else:
        raise ValueError(f"Unrecognized kernel type: {kernel_type}. "
                         f"Supported types are 'bartlett', 'parzen', 'tukey-hanning', 'quadratic'.")
    
    # Adjust bandwidth based on autocorrelation structure
    # If autocorrelations decay quickly, use smaller bandwidth
    # If autocorrelations decay slowly, use larger bandwidth
    acf_decay_rate = np.mean(np.abs(np.diff(acf)))
    if acf_decay_rate > 0.1:
        # Fast decay, use smaller bandwidth
        optimal_bw *= 0.8
    elif acf_decay_rate < 0.05:
        # Slow decay, use larger bandwidth
        optimal_bw *= 1.2
    
    # Ensure bandwidth is at least 1
    optimal_bw = max(1, optimal_bw)
    
    # Round to nearest integer
    return round(optimal_bw)


def compute_bias_correction(returns: np.ndarray, noise_variance: Optional[float] = None) -> float:
    """
    Compute bias correction for realized measures.
    
    Args:
        returns: Array of returns
        noise_variance: Estimated noise variance (if None, it will be estimated)
        
    Returns:
        Bias correction factor
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.utils import compute_bias_correction
        >>> np.random.seed(42)
        >>> returns = np.random.normal(0, 0.01, 1000)
        >>> bias_correction = compute_bias_correction(returns)
        >>> bias_correction
        0.0001...
    """
    # Convert to numpy array if not already
    returns = np.asarray(returns)
    
    # Validate inputs
    if returns.ndim != 1:
        raise ValueError("returns must be a 1D array")
    
    # Estimate noise variance if not provided
    if noise_variance is None:
        noise_var = noise_variance(returns)
    else:
        noise_var = noise_variance
    
    # Compute bias correction
    n = len(returns)
    bias_correction = 2 * n * noise_var
    
    return bias_correction


def compute_asymptotic_variance(returns: np.ndarray, measure_type: str = 'rv') -> float:
    """
    Compute asymptotic variance of realized measures.
    
    Args:
        returns: Array of returns
        measure_type: Type of realized measure ('rv', 'bv', 'kernel')
        
    Returns:
        Asymptotic variance
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.utils import compute_asymptotic_variance
        >>> returns = np.array([0.01, -0.005, 0.008, -0.002, 0.003, -0.001])
        >>> compute_asymptotic_variance(returns)
        1.8e-08
    """
    # Convert to numpy array if not already
    returns = np.asarray(returns)
    
    # Validate inputs
    if returns.ndim != 1:
        raise ValueError("returns must be a 1D array")
    
    # Compute asymptotic variance based on measure type
    if measure_type.lower() == 'rv':
        # For realized variance, asymptotic variance is 2 * integrated quarticity
        quarticity = compute_realized_quarticity(returns)
        asymp_var = 2 * quarticity
    
    elif measure_type.lower() == 'bv':
        # For bipower variation, asymptotic variance is more complex
        # Simplified formula: (pi^2/4 + pi - 3) * integrated quarticity
        quarticity = compute_realized_quarticity(returns)
        asymp_var = (np.pi**2/4 + np.pi - 3) * quarticity
    
    elif measure_type.lower() == 'kernel':
        # For kernel-based estimators, asymptotic variance depends on the kernel
        # Simplified formula: 2 * integrated quarticity * kernel_constant
        quarticity = compute_realized_quarticity(returns)
        kernel_constant = 1.0  # Default value, should be adjusted based on kernel
        asymp_var = 2 * quarticity * kernel_constant
    
    else:
        raise ValueError(f"Unrecognized measure type: {measure_type}. "
                         f"Supported types are 'rv', 'bv', 'kernel'.")
    
    return asymp_var


def compute_confidence_intervals(measure: float, asymp_var: float, n: int, 
                               confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute confidence intervals for realized measures.
    
    Args:
        measure: Realized measure value
        asymp_var: Asymptotic variance
        n: Number of observations
        confidence: Confidence level (0-1)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
        
    Examples:
        >>> from mfe.models.realized.utils import compute_confidence_intervals
        >>> measure = 0.0002
        >>> asymp_var = 1.8e-8
        >>> n = 100
        >>> lower, upper = compute_confidence_intervals(measure, asymp_var, n)
        >>> lower, upper
        (0.00017..., 0.00022...)
    """
    # Validate inputs
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    
    # Compute standard error
    std_error = np.sqrt(asymp_var / n)
    
    # Compute critical value
    alpha = 1 - confidence
    z_critical = stats.norm.ppf(1 - alpha/2)
    
    # Compute confidence interval
    lower_bound = measure - z_critical * std_error
    upper_bound = measure + z_critical * std_error
    
    # Ensure lower bound is non-negative for variance measures
    lower_bound = max(0, lower_bound)
    
    return lower_bound, upper_bound


# ============================================================================
# Module Initialization
# ============================================================================

# Register Numba-accelerated functions if available
def _register_numba_functions() -> None:
    """
    Register Numba JIT-compiled functions for realized volatility utilities.
    
    This function is called during module initialization to register
    performance-critical functions for JIT compilation if Numba is available.
    """
    if HAS_NUMBA:
        # The functions are already decorated with @jit, so no additional
        # registration is needed here. This function is kept for consistency
        # with the module structure and potential future enhancements.
        logger.debug("Realized volatility utility Numba JIT functions registered")
    else:
        logger.info("Numba not available. Realized volatility utilities will use pure NumPy implementations.")


# Initialize the module
_register_numba_functions()
