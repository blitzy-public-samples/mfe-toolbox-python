'''
Subsampling implementation for high-frequency financial data.

This module provides functions for subsampling high-frequency data to improve
the precision of realized volatility estimators. Subsampling averages estimates
from multiple shifted grids, reducing the variance of the estimators while
maintaining their consistency properties.

The implementation leverages NumPy's efficient array operations and Pandas'
time series capabilities, with Numba acceleration for performance-critical
calculations. The module supports both calendar time and tick time subsampling
strategies, with comprehensive type hints and parameter validation.

Functions:
    subsample_grids: Generate subsampling grids for high-frequency data
    subsample_returns: Apply subsampling to return series
    subsample_prices: Apply subsampling to price series
    subsample_times: Generate subsampled time points
    subsample_realized_variance: Compute subsampled realized variance
    visualize_subsampling_grids: Create visualization of subsampling grids
'''

import logging
import warnings
from typing import (
    Any, Callable, Dict, List, Literal, Optional, Sequence, 
    Tuple, Type, TypeVar, Union, cast, overload
)
import numpy as np
import pandas as pd
from scipy import stats

from ...core.exceptions import (
    DimensionError, NumericError, ParameterError, 
    raise_dimension_error, raise_numeric_error, warn_numeric
)
from .utils import (
    compute_returns, compute_realized_variance, compute_subsampled_measure,
    seconds2unit, unit2seconds, wall2seconds, seconds2wall,
    unit2wall, wall2unit, align_time, sample_prices
)

# Set up module-level logger
logger = logging.getLogger("mfe.models.realized.subsample")

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
    logger.debug("Numba available for subsampling acceleration")
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    HAS_NUMBA = False
    logger.info("Numba not available. Subsampling will use pure NumPy implementations.")

# Try to import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.info("Matplotlib not available. Visualization functions will be disabled.")


@jit(nopython=True, cache=True)
def _generate_subsampling_grids_numba(n: int, subsample_factor: int) -> np.ndarray:
    """
    Numba-accelerated implementation of subsampling grid generation.
    
    Args:
        n: Length of the original data
        subsample_factor: Number of subsamples to generate
        
    Returns:
        2D array of indices for each subsample grid
    """
    # Initialize the grid array
    grids = np.zeros((subsample_factor, n // subsample_factor + 1), dtype=np.int64)
    
    # Generate each grid with different starting points
    for i in range(subsample_factor):
        # Start from the i-th point and take every subsample_factor-th point
        idx = 0
        for j in range(i, n, subsample_factor):
            if idx < grids.shape[1]:
                grids[i, idx] = j
                idx += 1
    
    return grids



def subsample_grids(n: int, subsample_factor: int) -> np.ndarray:
    """
    Generate subsampling grids for high-frequency data.
    
    This function creates multiple sampling grids with different starting points,
    which can be used to compute subsampled estimates of realized measures.
    
    Args:
        n: Length of the original data
        subsample_factor: Number of subsamples to generate
        
    Returns:
        2D array of indices for each subsample grid
        
    Raises:
        ValueError: If inputs are invalid
        
    Examples:
        >>> from mfe.models.realized.subsample import subsample_grids
        >>> grids = subsample_grids(10, 3)
        >>> grids
        array([[0, 3, 6, 9, 0],
               [1, 4, 7, 0, 0],
               [2, 5, 8, 0, 0]])
    """
    # Validate inputs
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(subsample_factor, int) or subsample_factor <= 0:
        raise ValueError("subsample_factor must be a positive integer")
    if subsample_factor > n:
        raise ValueError(f"subsample_factor ({subsample_factor}) must be less than or equal to n ({n})")
    
    # Use Numba-accelerated implementation if available
    if HAS_NUMBA:
        grids = _generate_subsampling_grids_numba(n, subsample_factor)
    else:
        # Pure NumPy implementation
        # Initialize the grid array
        max_points = n // subsample_factor + 1
        grids = np.zeros((subsample_factor, max_points), dtype=np.int64)
        
        # Generate each grid with different starting points
        for i in range(subsample_factor):
            # Start from the i-th point and take every subsample_factor-th point
            points = np.arange(i, n, subsample_factor)
            grids[i, :len(points)] = points
    
    return grids


@jit(nopython=True, cache=True)
def _subsample_returns_numba(returns: np.ndarray, subsample_factor: int) -> List[np.ndarray]:
    """
    Numba-accelerated implementation of return subsampling.
    
    Args:
        returns: Array of returns
        subsample_factor: Number of subsamples to generate
        
    Returns:
        List of subsampled return arrays
    """
    n = len(returns)
    subsampled_returns = []
    
    for i in range(subsample_factor):
        # Extract i-th subsample
        subsample = returns[i::subsample_factor]
        subsampled_returns.append(subsample)
    
    return subsampled_returns



def subsample_returns(returns: np.ndarray, subsample_factor: int) -> List[np.ndarray]:
    """
    Apply subsampling to return series.
    
    This function creates multiple subsampled return series by taking every
    subsample_factor-th return starting from different points.
    
    Args:
        returns: Array of returns
        subsample_factor: Number of subsamples to generate
        
    Returns:
        List of subsampled return arrays
        
    Raises:
        ValueError: If inputs are invalid
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.subsample import subsample_returns
        >>> returns = np.array([0.01, 0.02, -0.01, 0.03, -0.02, 0.01])
        >>> subsampled = subsample_returns(returns, 3)
        >>> len(subsampled)
        3
        >>> subsampled[0]  # First subsample (indices 0, 3)
        array([0.01, 0.03])
        >>> subsampled[1]  # Second subsample (indices 1, 4)
        array([ 0.02, -0.02])
        >>> subsampled[2]  # Third subsample (indices 2, 5)
        array([-0.01,  0.01])
    """
    # Convert to numpy array if not already
    returns = np.asarray(returns)
    
    # Validate inputs
    if returns.ndim != 1:
        raise ValueError("returns must be a 1D array")
    if not isinstance(subsample_factor, int) or subsample_factor <= 0:
        raise ValueError("subsample_factor must be a positive integer")
    if subsample_factor > len(returns):
        raise ValueError(f"subsample_factor ({subsample_factor}) must be less than or equal to the length of returns ({len(returns)})")
    
    # Use Numba-accelerated implementation if available
    if HAS_NUMBA:
        return _subsample_returns_numba(returns, subsample_factor)
    
    # Pure NumPy implementation
    subsampled_returns = []
    
    for i in range(subsample_factor):
        # Extract i-th subsample
        subsample = returns[i::subsample_factor]
        subsampled_returns.append(subsample)
    
    return subsampled_returns



def subsample_prices(
    prices: Union[np.ndarray, pd.Series], 
    times: Union[np.ndarray, pd.Series, pd.DatetimeIndex], 
    subsample_factor: int,
    return_type: str = 'log'
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Apply subsampling to price series.
    
    This function creates multiple subsampled price series by taking every
    subsample_factor-th price starting from different points, and computes
    the corresponding returns.
    
    Args:
        prices: Array of prices
        times: Array of time points corresponding to prices
        subsample_factor: Number of subsamples to generate
        return_type: Type of returns to compute ('log', 'simple')
        
    Returns:
        Tuple of (subsampled_prices, subsampled_times, subsampled_returns)
        
    Raises:
        ValueError: If inputs are invalid
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.subsample import subsample_prices
        >>> prices = np.array([100, 101, 102, 103, 104, 105])
        >>> times = np.array([0, 1, 2, 3, 4, 5])
        >>> subsampled_prices, subsampled_times, subsampled_returns = subsample_prices(
        ...     prices, times, 3
        ... )
        >>> subsampled_prices[0]  # First subsample (indices 0, 3)
        array([100, 103])
        >>> subsampled_times[0]  # Times for first subsample
        array([0, 3])
        >>> subsampled_returns[0]  # Returns for first subsample
        array([0.02955...])
    """
    # Convert inputs to numpy arrays
    prices_array = np.asarray(prices)
    
    # Handle different time formats
    is_datetime = False
    times_array = None
    
    if isinstance(times, pd.DatetimeIndex):
        is_datetime = True
        times_array = np.array(times.astype(np.int64) / 1e9)  # Convert to seconds since epoch
    elif isinstance(times, pd.Series) and pd.api.types.is_datetime64_dtype(times):
        is_datetime = True
        times_array = np.array(pd.DatetimeIndex(times).astype(np.int64) / 1e9)
    else:
        times_array = np.asarray(times)
    
    # Validate inputs
    if prices_array.ndim != 1:
        raise ValueError("prices must be a 1D array or Series")
    if times_array.ndim != 1:
        raise ValueError("times must be a 1D array, Series, or DatetimeIndex")
    if len(prices_array) != len(times_array):
        raise ValueError(f"prices length ({len(prices_array)}) must match times length ({len(times_array)})")
    if not isinstance(subsample_factor, int) or subsample_factor <= 0:
        raise ValueError("subsample_factor must be a positive integer")
    if subsample_factor > len(prices_array):
        raise ValueError(f"subsample_factor ({subsample_factor}) must be less than or equal to the length of prices ({len(prices_array)})")
    if return_type not in ['log', 'simple']:
        raise ValueError(f"return_type must be 'log' or 'simple', got {return_type}")
    
    # Generate subsampled price and time series
    subsampled_prices = []
    subsampled_times = []
    subsampled_returns = []
    
    for i in range(subsample_factor):
        # Extract i-th subsample
        subsample_prices = prices_array[i::subsample_factor]
        subsample_times = times_array[i::subsample_factor]
        
        # Compute returns if there are at least 2 prices
        if len(subsample_prices) >= 2:
            if return_type == 'log':
                subsample_returns = np.diff(np.log(subsample_prices))
            else:  # 'simple'
                subsample_returns = np.diff(subsample_prices) / subsample_prices[:-1]
        else:
            subsample_returns = np.array([])
        
        subsampled_prices.append(subsample_prices)
        subsampled_times.append(subsample_times)
        subsampled_returns.append(subsample_returns)
    
    return subsampled_prices, subsampled_times, subsampled_returns



def subsample_times(
    times: Union[np.ndarray, pd.Series, pd.DatetimeIndex], 
    subsample_factor: int
) -> List[np.ndarray]:
    """
    Generate subsampled time points.
    
    This function creates multiple subsampled time series by taking every
    subsample_factor-th time point starting from different points.
    
    Args:
        times: Array of time points
        subsample_factor: Number of subsamples to generate
        
    Returns:
        List of subsampled time arrays
        
    Raises:
        ValueError: If inputs are invalid
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.subsample import subsample_times
        >>> times = np.array([0, 1, 2, 3, 4, 5])
        >>> subsampled = subsample_times(times, 3)
        >>> subsampled[0]  # First subsample (indices 0, 3)
        array([0, 3])
        >>> subsampled[1]  # Second subsample (indices 1, 4)
        array([1, 4])
        >>> subsampled[2]  # Third subsample (indices 2, 5)
        array([2, 5])
    """
    # Handle different time formats
    is_datetime = False
    times_array = None
    
    if isinstance(times, pd.DatetimeIndex):
        is_datetime = True
        times_array = np.array(times.astype(np.int64) / 1e9)  # Convert to seconds since epoch
    elif isinstance(times, pd.Series) and pd.api.types.is_datetime64_dtype(times):
        is_datetime = True
        times_array = np.array(pd.DatetimeIndex(times).astype(np.int64) / 1e9)
    else:
        times_array = np.asarray(times)
    
    # Validate inputs
    if times_array.ndim != 1:
        raise ValueError("times must be a 1D array, Series, or DatetimeIndex")
    if not isinstance(subsample_factor, int) or subsample_factor <= 0:
        raise ValueError("subsample_factor must be a positive integer")
    if subsample_factor > len(times_array):
        raise ValueError(f"subsample_factor ({subsample_factor}) must be less than or equal to the length of times ({len(times_array)})")
    
    # Generate subsampled time series
    subsampled_times = []
    
    for i in range(subsample_factor):
        # Extract i-th subsample
        subsample = times_array[i::subsample_factor]
        subsampled_times.append(subsample)
    
    return subsampled_times



def subsample_realized_variance(
    returns: np.ndarray, 
    subsample_factor: int,
    scale: bool = True
) -> float:
    """
    Compute subsampled realized variance.
    
    This function computes realized variance using subsampling to reduce
    the variance of the estimator. It averages the realized variance
    estimates from multiple subsampled return series.
    
    Args:
        returns: Array of returns
        subsample_factor: Number of subsamples to generate
        scale: Whether to scale the subsampled estimates to match the original scale
        
    Returns:
        Subsampled realized variance
        
    Raises:
        ValueError: If inputs are invalid
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.subsample import subsample_realized_variance
        >>> np.random.seed(42)
        >>> returns = np.random.normal(0, 0.01, 100)
        >>> rv = subsample_realized_variance(returns, 5)
        >>> rv
        0.0099...
    """
    # Convert to numpy array if not already
    returns = np.asarray(returns)
    
    # Validate inputs
    if returns.ndim != 1:
        raise ValueError("returns must be a 1D array")
    if not isinstance(subsample_factor, int) or subsample_factor <= 0:
        raise ValueError("subsample_factor must be a positive integer")
    if subsample_factor > len(returns):
        raise ValueError(f"subsample_factor ({subsample_factor}) must be less than or equal to the length of returns ({len(returns)})")
    
    # Use the compute_subsampled_measure utility function
    return compute_subsampled_measure(returns, subsample_factor)



def visualize_subsampling_grids(
    n: int, 
    subsample_factor: int,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> Optional[Any]:
    """
    Create visualization of subsampling grids.
    
    This function creates a visual representation of the subsampling grids,
    showing how different starting points lead to different subsamples.
    
    Args:
        n: Length of the original data
        subsample_factor: Number of subsamples to generate
        figsize: Figure size as (width, height) in inches
        save_path: Path to save the figure (if None, figure is displayed)
        
    Returns:
        Matplotlib figure object if matplotlib is available, None otherwise
        
    Examples:
        >>> from mfe.models.realized.subsample import visualize_subsampling_grids
        >>> fig = visualize_subsampling_grids(20, 5)
    """
    if not HAS_MATPLOTLIB:
        logger.warning("Matplotlib is not available. Cannot create visualization.")
        return None
    
    # Generate subsampling grids
    grids = subsample_grids(n, subsample_factor)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each grid as a separate line
    for i in range(subsample_factor):
        # Get valid indices (non-zero)
        valid_indices = grids[i] > 0
        valid_indices[0] = True  # Always include the first point
        
        # Plot the grid points
        grid_points = grids[i][valid_indices]
        ax.plot(grid_points, [i] * len(grid_points), 'o-', label=f'Grid {i+1}')
    
    # Add a reference line showing all original points
    ax.plot(np.arange(n), [-1] * n, 'x', label='Original Data')
    
    # Add labels and legend
    ax.set_title(f'Subsampling Grids (n={n}, factor={subsample_factor})')
    ax.set_xlabel('Data Index')
    ax.set_ylabel('Grid Number')
    ax.set_yticks(list(range(-1, subsample_factor)))
    ax.set_yticklabels(['Original'] + [f'Grid {i+1}' for i in range(subsample_factor)])
    ax.grid(True, alpha=0.3)
    
    # Save or display the figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.tight_layout()
    
    return fig



def calendar_time_subsample(
    prices: Union[np.ndarray, pd.Series], 
    times: Union[np.ndarray, pd.Series, pd.DatetimeIndex], 
    subsample_factor: int,
    sampling_interval: Union[str, float, int],
    time_unit: str = 'seconds',
    return_type: str = 'log',
    interpolation_method: str = 'previous'
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Apply calendar time subsampling to price series.
    
    This function creates multiple subsampled price series by shifting the
    sampling grid in calendar time. It first resamples the data to a regular
    grid and then applies subsampling.
    
    Args:
        prices: Array of prices
        times: Array of time points corresponding to prices
        subsample_factor: Number of subsamples to generate
        sampling_interval: Sampling interval (as string like '5min' or numeric value in time_unit)
        time_unit: Time unit for numeric sampling_interval ('seconds', 'minutes', 'hours', 'days')
        return_type: Type of returns to compute ('log', 'simple')
        interpolation_method: Method for interpolating prices ('previous', 'linear', 'cubic')
        
    Returns:
        Tuple of (subsampled_prices, subsampled_times, subsampled_returns)
        
    Raises:
        ValueError: If inputs are invalid
        
    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from mfe.models.realized.subsample import calendar_time_subsample
        >>> # Example with datetime times
        >>> prices = np.array([100.0, 100.5, 101.0, 101.5, 102.0])
        >>> times = pd.to_datetime([
        ...     '2023-01-01 09:30:00', 
        ...     '2023-01-01 09:30:30',
        ...     '2023-01-01 09:31:10', 
        ...     '2023-01-01 09:32:30',
        ...     '2023-01-01 09:33:30'
        ... ])
        >>> subsampled_prices, subsampled_times, subsampled_returns = calendar_time_subsample(
        ...     prices, times, 3, '1min'
        ... )
    """
    # Import price_filter here to avoid circular imports
    from .price_filter import price_filter
    
    # Convert inputs to numpy arrays
    prices_array = np.asarray(prices)
    
    # Validate inputs
    if prices_array.ndim != 1:
        raise ValueError("prices must be a 1D array or Series")
    if not isinstance(subsample_factor, int) or subsample_factor <= 0:
        raise ValueError("subsample_factor must be a positive integer")
    if return_type not in ['log', 'simple']:
        raise ValueError(f"return_type must be 'log' or 'simple', got {return_type}")
    
    # Handle different time formats
    is_datetime = False
    
    if isinstance(times, pd.DatetimeIndex):
        is_datetime = True
    elif isinstance(times, pd.Series) and pd.api.types.is_datetime64_dtype(times):
        is_datetime = True
        times = pd.DatetimeIndex(times)
    
    # For calendar time subsampling, we need to:
    # 1. Determine the base sampling interval
    # 2. Create subsample_factor different grids by shifting the starting point
    # 3. Sample the data to each grid
    
    # Determine the base sampling interval in seconds
    if isinstance(sampling_interval, str):
        # Parse string frequency
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
        match = re.match(r'(\d+)([a-zA-Z]+)', sampling_interval)
        if match:
            value, unit = match.groups()
            if unit.lower() in freq_map:
                interval_seconds = int(value) * freq_map[unit.lower()]
            else:
                raise ValueError(f"Unrecognized frequency unit: {unit}")
        else:
            raise ValueError(f"Could not parse frequency string: {sampling_interval}")
    else:
        # Numeric interval in the specified time unit
        interval_seconds = float(sampling_interval)
        if time_unit != 'seconds':
            interval_seconds = unit2seconds(np.array([interval_seconds]), time_unit)[0]
    
    # Calculate the shift interval for subsampling
    shift_interval = interval_seconds / subsample_factor
    
    # Generate subsampled price and time series
    subsampled_prices = []
    subsampled_times = []
    subsampled_returns = []
    
    for i in range(subsample_factor):
        # Calculate the starting offset for this subsample
        offset = i * shift_interval
        
        if is_datetime:
            # For datetime index, create a shifted starting time
            start_time = times[0] + pd.Timedelta(seconds=offset)
            
            # Create a regular time grid starting from the shifted time
            if isinstance(sampling_interval, str):
                # Use pandas date_range with string frequency
                end_time = times[-1]
                time_grid = pd.date_range(start=start_time, end=end_time, freq=sampling_interval)
            else:
                # Convert interval_seconds to a pandas frequency string
                freq_str = f"{int(interval_seconds)}S"
                end_time = times[-1]
                time_grid = pd.date_range(start=start_time, end=end_time, freq=freq_str)
            
            # Create a DataFrame with the original data
            df = pd.DataFrame({'price': prices_array}, index=times)
            
            # Resample to the new time grid
            if interpolation_method == 'previous':
                resampled = df.reindex(time_grid, method='ffill')
            elif interpolation_method == 'linear':
                resampled = df.reindex(time_grid).interpolate(method='linear')
            elif interpolation_method == 'cubic':
                resampled = df.reindex(time_grid).interpolate(method='cubic')
            else:
                raise ValueError(f"Unrecognized interpolation method: {interpolation_method}")
            
            # Extract resampled prices and times
            subsample_prices = resampled['price'].values
            subsample_times = np.array(resampled.index.astype(np.int64) / 1e9)
        else:
            # For numeric times, create a shifted starting time
            times_array = np.asarray(times)
            start_time = times_array[0] + offset
            
            # Create a regular time grid starting from the shifted time
            end_time = times_array[-1]
            num_points = int(np.floor((end_time - start_time) / interval_seconds)) + 1
            time_grid = np.linspace(start_time, start_time + (num_points - 1) * interval_seconds, num_points)
            
            # Interpolate prices to the new time grid
            subsample_prices = align_time(prices_array, times_array, time_grid, interpolation_method)
            subsample_times = time_grid
        
        # Compute returns if there are at least 2 prices
        if len(subsample_prices) >= 2:
            if return_type == 'log':
                subsample_returns = np.diff(np.log(subsample_prices))
            else:  # 'simple'
                subsample_returns = np.diff(subsample_prices) / subsample_prices[:-1]
        else:
            subsample_returns = np.array([])
        
        subsampled_prices.append(subsample_prices)
        subsampled_times.append(subsample_times)
        subsampled_returns.append(subsample_returns)
    
    return subsampled_prices, subsampled_times, subsampled_returns



def tick_time_subsample(
    prices: Union[np.ndarray, pd.Series], 
    times: Union[np.ndarray, pd.Series, pd.DatetimeIndex], 
    subsample_factor: int,
    return_type: str = 'log'
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Apply tick time subsampling to price series.
    
    This function creates multiple subsampled price series by taking every
    subsample_factor-th price starting from different points. This is also
    known as tick time subsampling because it uses the original tick data
    without resampling to a regular time grid.
    
    Args:
        prices: Array of prices
        times: Array of time points corresponding to prices
        subsample_factor: Number of subsamples to generate
        return_type: Type of returns to compute ('log', 'simple')
        
    Returns:
        Tuple of (subsampled_prices, subsampled_times, subsampled_returns)
        
    Raises:
        ValueError: If inputs are invalid
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.subsample import tick_time_subsample
        >>> prices = np.array([100, 101, 102, 103, 104, 105])
        >>> times = np.array([0, 1, 2, 3, 4, 5])
        >>> subsampled_prices, subsampled_times, subsampled_returns = tick_time_subsample(
        ...     prices, times, 3
        ... )
        >>> subsampled_prices[0]  # First subsample (indices 0, 3)
        array([100, 103])
    """
    # This is essentially the same as subsample_prices
    return subsample_prices(prices, times, subsample_factor, return_type)



def optimal_subsample_factor(
    returns: np.ndarray, 
    max_factor: Optional[int] = None,
    method: str = 'variance'
) -> int:
    """
    Determine the optimal subsampling factor based on data characteristics.
    
    This function analyzes the return series and determines the optimal
    subsampling factor that minimizes the variance of the realized variance
    estimator or optimizes other criteria.
    
    Args:
        returns: Array of returns
        max_factor: Maximum subsampling factor to consider (default: sqrt(n))
        method: Method for determining optimal factor ('variance', 'autocorr', 'noise')
        
    Returns:
        Optimal subsampling factor
        
    Raises:
        ValueError: If inputs are invalid
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.subsample import optimal_subsample_factor
        >>> np.random.seed(42)
        >>> returns = np.random.normal(0, 0.01, 100)
        >>> optimal_factor = optimal_subsample_factor(returns)
        >>> optimal_factor
        5
    """
    # Convert to numpy array if not already
    returns = np.asarray(returns)
    
    # Validate inputs
    if returns.ndim != 1:
        raise ValueError("returns must be a 1D array")
    
    n = len(returns)
    
    # Set default max_factor if not provided
    if max_factor is None:
        max_factor = int(np.sqrt(n))
    
    # Ensure max_factor is valid
    max_factor = min(max_factor, n // 2)
    max_factor = max(max_factor, 2)
    
    # Determine optimal factor based on the specified method
    if method.lower() == 'variance':
        # Compute realized variance for different subsampling factors
        factors = np.arange(1, max_factor + 1)
        variances = np.zeros(len(factors))
        
        for i, factor in enumerate(factors):
            # Compute subsampled realized variance
            rv = subsample_realized_variance(returns, factor)
            
            # Compute variance of the estimator using bootstrap
            bootstrap_samples = 100
            bootstrap_rv = np.zeros(bootstrap_samples)
            
            for j in range(bootstrap_samples):
                # Resample returns with replacement
                bootstrap_returns = returns[np.random.randint(0, n, n)]
                bootstrap_rv[j] = subsample_realized_variance(bootstrap_returns, factor)
            
            # Variance of the bootstrap estimates
            variances[i] = np.var(bootstrap_rv)
        
        # Find the factor that minimizes variance
        optimal_factor = factors[np.argmin(variances)]
    
    elif method.lower() == 'autocorr':
        # Determine optimal factor based on autocorrelation structure
        # Compute autocorrelations
        max_lag = min(30, n // 4)
        acf = np.zeros(max_lag)
        
        for lag in range(max_lag):
            if lag < n - 1:
                acf[lag] = np.corrcoef(returns[:-lag-1], returns[lag+1:])[0, 1]
        
        # Find the lag where autocorrelation becomes insignificant
        # (approximately 2 standard errors)
        std_error = 1.96 / np.sqrt(n)
        significant_lags = np.where(np.abs(acf) > std_error)[0]
        
        if len(significant_lags) > 0:
            # Use the maximum significant lag + 1 as the optimal factor
            optimal_factor = significant_lags[-1] + 1
        else:
            # No significant autocorrelation, use a small factor
            optimal_factor = 2
        
        # Ensure optimal_factor is within bounds
        optimal_factor = max(2, min(optimal_factor, max_factor))
    
    elif method.lower() == 'noise':
        # Determine optimal factor based on noise characteristics
        # Estimate noise variance
        from .utils import noise_variance
        noise_var = noise_variance(returns)
        
        # Compute signal variance (realized variance)
        signal_var = np.sum(returns**2)
        
        # Signal-to-noise ratio
        snr = signal_var / (noise_var * n)
        
        # Optimal factor is approximately proportional to sqrt(1/snr)
        optimal_factor = int(np.sqrt(1 / snr))
        
        # Ensure optimal_factor is within bounds
        optimal_factor = max(2, min(optimal_factor, max_factor))
    
    else:
        raise ValueError(f"Unrecognized method: {method}. "
                         f"Supported methods are 'variance', 'autocorr', 'noise'.")
    
    return optimal_factor
