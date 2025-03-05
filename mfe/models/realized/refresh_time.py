# mfe/models/realized/refresh_time.py

"""
Refresh time synchronization for multivariate high-frequency data.

This module implements the refresh time algorithm for synchronizing asynchronous
high-frequency financial data across multiple assets. The refresh time method
aligns observations by creating a common time grid where each point represents
the time when all assets have experienced at least one price update.

The implementation leverages Pandas' powerful time series capabilities for efficient
data manipulation and provides comprehensive visualization tools for analyzing the
synchronization process. The module supports various time formats with microsecond
precision for ultra-high-frequency data.

Functions:
    refresh_time: Compute refresh times for multivariate high-frequency data
    refresh_time_prices: Synchronize price data using refresh times
    refresh_time_returns: Compute synchronized returns using refresh times
    visualize_refresh_times: Create visualization of refresh time synchronization
    refresh_time_statistics: Compute statistics about refresh time sampling
"""

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
    seconds2unit, unit2seconds, wall2seconds, seconds2wall,
    unit2wall, wall2unit, align_time
)

# Set up module-level logger
logger = logging.getLogger("mfe.models.realized.refresh_time")

# Try to import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.info("Matplotlib not available. Visualization functions will be disabled.")


def refresh_time(
    times_list: List[Union[np.ndarray, pd.Series, pd.DatetimeIndex]],
    convert_to_seconds: bool = True,
    base_time: Optional[Union[str, pd.Timestamp]] = None,
    return_indices: bool = False,
    return_pandas: bool = False
) -> Union[
    np.ndarray,
    pd.DatetimeIndex,
    Tuple[np.ndarray, List[np.ndarray]],
    Tuple[pd.DatetimeIndex, List[np.ndarray]]
]:
    """
    Compute refresh times for multivariate high-frequency data.

    The refresh time algorithm identifies points where all assets have experienced
    at least one price update, creating a synchronized time grid for multivariate analysis.

    Args:
        times_list: List of time series for each asset
        convert_to_seconds: Whether to convert all times to seconds for processing
        base_time: Base time for datetime conversion (required if times are datetime and convert_to_seconds=True)
        return_indices: Whether to return indices of refresh times in original series
        return_pandas: Whether to return a pandas DatetimeIndex instead of numpy array

    Returns:
        Refresh times as numpy array or pandas DatetimeIndex, optionally with indices

    Raises:
        ValueError: If inputs have invalid dimensions or if parameters are invalid
        TypeError: If inputs have incorrect types

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from mfe.models.realized.refresh_time import refresh_time
        >>> # Example with numeric times
        >>> times1 = np.array([1, 3, 5, 7, 9])
        >>> times2 = np.array([2, 4, 6, 8, 10])
        >>> refresh_times = refresh_time([times1, times2])
        >>> refresh_times
        array([ 2.,  4.,  6.,  8., 10.])

        >>> # Example with datetime times
        >>> times1 = pd.to_datetime(['2023-01-01 09:30:00', '2023-01-01 09:31:00', 
        ...                          '2023-01-01 09:32:00'])
        >>> times2 = pd.to_datetime(['2023-01-01 09:30:30', '2023-01-01 09:31:30', 
        ...                          '2023-01-01 09:32:30'])
        >>> refresh_times = refresh_time([times1, times2], return_pandas=True)
        >>> refresh_times
        DatetimeIndex(['2023-01-01 09:30:30', '2023-01-01 09:31:30', 
                       '2023-01-01 09:32:30'],
                      dtype='datetime64[ns]', freq=None)
    """
    # Validate inputs
    if not isinstance(times_list, list):
        raise TypeError("times_list must be a list of time series")

    if len(times_list) < 2:
        raise ValueError("At least two time series must be provided")

    # Process each time series
    processed_times = []
    is_datetime = False
    datetime_index = None

    for i, times in enumerate(times_list):
        # Check if times is a pandas DatetimeIndex or can be converted to one
        if isinstance(times, pd.DatetimeIndex):
            is_datetime = True
            if convert_to_seconds:
                if base_time is None:
                    # Use the first time point as base time
                    base_time = times[0]
                times_seconds = wall2seconds(times, pd.Timestamp(base_time).timestamp())
                processed_times.append(times_seconds)
            else:
                # Keep as DatetimeIndex
                processed_times.append(times)
                if datetime_index is None:
                    datetime_index = times
        elif isinstance(times, pd.Series):
            if pd.api.types.is_datetime64_dtype(times):
                is_datetime = True
                datetime_idx = pd.DatetimeIndex(times)
                if convert_to_seconds:
                    if base_time is None:
                        # Use the first time point as base time
                        base_time = datetime_idx[0]
                    times_seconds = wall2seconds(datetime_idx, pd.Timestamp(base_time).timestamp())
                    processed_times.append(times_seconds)
                else:
                    # Keep as DatetimeIndex
                    processed_times.append(datetime_idx)
                    if datetime_index is None:
                        datetime_index = datetime_idx
            else:
                # Numeric Series
                processed_times.append(np.asarray(times))
        else:
            # Assume numpy array or list of numeric values
            try:
                times_array = np.asarray(times, dtype=float)
                processed_times.append(times_array)
            except (ValueError, TypeError):
                # Try to convert to datetime
                try:
                    datetime_idx = pd.DatetimeIndex(pd.to_datetime(times))
                    is_datetime = True
                    if convert_to_seconds:
                        if base_time is None:
                            # Use the first time point as base time
                            base_time = datetime_idx[0]
                        times_seconds = wall2seconds(datetime_idx, pd.Timestamp(base_time).timestamp())
                        processed_times.append(times_seconds)
                    else:
                        # Keep as DatetimeIndex
                        processed_times.append(datetime_idx)
                        if datetime_index is None:
                            datetime_index = datetime_idx
                except (ValueError, TypeError):
                    raise ValueError(f"Could not process time series {i}: not recognized as numeric or datetime")

    # Check if we have a mix of datetime and numeric
    if is_datetime and convert_to_seconds:
        # All datetime series have been converted to seconds
        is_datetime = False

    if is_datetime and not convert_to_seconds:
        # We're working with datetime indices
        # Convert any numeric series to datetime if needed
        for i, times in enumerate(processed_times):
            if not isinstance(times, pd.DatetimeIndex):
                # Try to convert to datetime using the base_time
                if base_time is None:
                    raise ValueError("base_time must be provided to convert numeric times to datetime")
                try:
                    datetime_idx = seconds2wall(times, base_time)
                    processed_times[i] = datetime_idx
                except Exception as e:
                    raise ValueError(f"Could not convert time series {i} to datetime: {str(e)}")

    # Compute refresh times
    if is_datetime and not convert_to_seconds:
        # Working with datetime indices
        # Convert to pandas Series for easier manipulation
        time_series = [pd.Series(1, index=times) for times in processed_times]

        # Combine all series and resample to get refresh times
        combined = pd.concat(time_series, axis=1, join='outer')
        combined = combined.fillna(0)

        # Find points where all assets have been observed
        cumulative = combined.cumsum()
        refresh_mask = (cumulative > 0).all(axis=1)
        refresh_times_idx = combined.index[refresh_mask]

        # Get indices of refresh times in original series
        if return_indices:
            indices = []
            for times in processed_times:
                # Find indices of times that are less than or equal to each refresh time
                idx = np.zeros(len(refresh_times_idx), dtype=int)
                for j, rt in enumerate(refresh_times_idx):
                    # Find the last time point less than or equal to the refresh time
                    idx[j] = np.searchsorted(times, rt, side='right') - 1
                indices.append(idx)

            if return_pandas:
                return refresh_times_idx, indices
            else:
                # Convert to numpy array of timestamps
                refresh_times_array = np.array(refresh_times_idx.astype(np.int64) / 1e9)
                return refresh_times_array, indices
        else:
            if return_pandas:
                return refresh_times_idx
            else:
                # Convert to numpy array of timestamps
                refresh_times_array = np.array(refresh_times_idx.astype(np.int64) / 1e9)
                return refresh_times_array
    else:
        # Working with numeric time arrays
        # Convert all to numpy arrays
        time_arrays = [np.asarray(times, dtype=float) for times in processed_times]

        # Ensure all arrays are sorted
        for i, times in enumerate(time_arrays):
            if not np.all(np.diff(times) >= 0):
                raise ValueError(f"Time series {i} is not monotonically increasing")

        # Compute refresh times using vectorized operations
        # Combine all unique time points
        all_times = np.unique(np.concatenate(time_arrays))

        # For each asset, find the next observation time for each point in all_times
        next_obs = np.zeros((len(all_times), len(time_arrays)))

        for i, times in enumerate(time_arrays):
            # For each time in all_times, find the next observation in this asset's times
            for j, t in enumerate(all_times):
                # Find the first time point greater than or equal to t
                idx = np.searchsorted(times, t)
                if idx < len(times):
                    next_obs[j, i] = times[idx]
                else:
                    # No next observation, use infinity
                    next_obs[j, i] = np.inf

        # Refresh times are points where the maximum next observation time
        # across all assets changes
        max_next_obs = np.max(next_obs, axis=1)

        # Find points where max_next_obs changes
        refresh_indices = np.where(np.diff(max_next_obs) > 0)[0] + 1

        # Add the first point
        refresh_indices = np.insert(refresh_indices, 0, 0)

        # Get the refresh times
        refresh_times_array = max_next_obs[refresh_indices]

        # Remove any infinite values
        if np.any(np.isinf(refresh_times_array)):
            refresh_times_array = refresh_times_array[~np.isinf(refresh_times_array)]

        # Get indices of refresh times in original series
        if return_indices:
            indices = []
            for times in time_arrays:
                # Find indices of times that are less than or equal to each refresh time
                idx = np.zeros(len(refresh_times_array), dtype=int)
                for j, rt in enumerate(refresh_times_array):
                    # Find the last time point less than or equal to the refresh time
                    idx[j] = np.searchsorted(times, rt, side='right') - 1
                indices.append(idx)

            if return_pandas and base_time is not None:
                # Convert to DatetimeIndex
                refresh_times_idx = seconds2wall(refresh_times_array, base_time)
                return refresh_times_idx, indices
            else:
                return refresh_times_array, indices
        else:
            if return_pandas and base_time is not None:
                # Convert to DatetimeIndex
                refresh_times_idx = seconds2wall(refresh_times_array, base_time)
                return refresh_times_idx
            else:
                return refresh_times_array


def refresh_time_prices(
    prices_list: List[Union[np.ndarray, pd.Series]],
    times_list: List[Union[np.ndarray, pd.Series, pd.DatetimeIndex]],
    interpolation_method: str = 'previous',
    convert_to_seconds: bool = True,
    base_time: Optional[Union[str, pd.Timestamp]] = None,
    return_pandas: bool = False
) -> Union[
    Tuple[np.ndarray, np.ndarray],
    Tuple[pd.DataFrame, pd.DatetimeIndex]
]:
    """
    Synchronize price data using refresh times.

    This function computes refresh times for the given time series and
    aligns price data to these synchronized time points.

    Args:
        prices_list: List of price series for each asset
        times_list: List of corresponding time points
        interpolation_method: Method for interpolating prices ('previous', 'linear', 'cubic')
        convert_to_seconds: Whether to convert all times to seconds for processing
        base_time: Base time for datetime conversion
        return_pandas: Whether to return pandas objects instead of numpy arrays

    Returns:
        Tuple of (synchronized_prices, refresh_times) where synchronized_prices
        is a 2D array or DataFrame with each column representing an asset

    Raises:
        ValueError: If inputs have invalid dimensions or if parameters are invalid
        TypeError: If inputs have incorrect types

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from mfe.models.realized.refresh_time import refresh_time_prices
        >>> # Example with numeric times
        >>> prices1 = np.array([100, 101, 102, 103, 104])
        >>> times1 = np.array([1, 3, 5, 7, 9])
        >>> prices2 = np.array([50, 51, 52, 53, 54])
        >>> times2 = np.array([2, 4, 6, 8, 10])
        >>> synced_prices, refresh_times = refresh_time_prices(
        ...     [prices1, prices2], [times1, times2]
        ... )
        >>> refresh_times
        array([ 2.,  4.,  6.,  8., 10.])
        >>> synced_prices
        array([[100.,  50.],
               [101.,  51.],
               [102.,  52.],
               [103.,  53.],
               [104.,  54.]])
    """
    # Validate inputs
    if not isinstance(prices_list, list) or not isinstance(times_list, list):
        raise TypeError("prices_list and times_list must be lists")

    if len(prices_list) != len(times_list):
        raise ValueError(f"Number of price series ({len(prices_list)}) must match "
                         f"number of time series ({len(times_list)})")

    if len(prices_list) < 2:
        raise ValueError("At least two price series must be provided")

    # Check that each price series matches its time series in length
    for i, (prices, times) in enumerate(zip(prices_list, times_list)):
        prices_array = np.asarray(prices)

        if isinstance(times, pd.DatetimeIndex):
            if len(prices_array) != len(times):
                raise ValueError(f"Price series {i} length ({len(prices_array)}) "
                                 f"does not match time series length ({len(times)})")
        else:
            times_array = np.asarray(times)
            if len(prices_array) != len(times_array):
                raise ValueError(f"Price series {i} length ({len(prices_array)}) "
                                 f"does not match time series length ({len(times_array)})")

    # Compute refresh times
    refresh_times_result = refresh_time(
        times_list,
        convert_to_seconds=convert_to_seconds,
        base_time=base_time,
        return_indices=False,
        return_pandas=return_pandas
    )

    # Align price data to refresh times
    if return_pandas and isinstance(refresh_times_result, pd.DatetimeIndex):
        # Working with pandas objects
        refresh_times_idx = refresh_times_result

        # Create a DataFrame with aligned price series
        aligned_prices = pd.DataFrame(index=refresh_times_idx)

        for i, (prices, times) in enumerate(zip(prices_list, times_list)):
            # Convert to Series if not already
            if isinstance(prices, pd.Series) and isinstance(prices.index, pd.DatetimeIndex):
                # Series already has datetime index
                price_series = prices
            elif isinstance(times, pd.DatetimeIndex):
                # Create Series with DatetimeIndex
                price_series = pd.Series(prices, index=times)
            else:
                # Convert times to DatetimeIndex if possible
                try:
                    datetime_idx = pd.DatetimeIndex(pd.to_datetime(times))
                    price_series = pd.Series(prices, index=datetime_idx)
                except (ValueError, TypeError):
                    if base_time is None:
                        raise ValueError("base_time must be provided to convert numeric times to datetime")
                    # Convert numeric times to datetime
                    datetime_idx = seconds2wall(np.asarray(times), base_time)
                    price_series = pd.Series(prices, index=datetime_idx)

            # Align to refresh times
            if interpolation_method == 'previous':
                # Use asof to get the last value before each refresh time
                aligned_series = price_series.reindex(
                    refresh_times_idx, method='pad'
                )
            elif interpolation_method == 'linear':
                # Linear interpolation
                aligned_series = price_series.reindex(
                    refresh_times_idx, method='linear'
                )
            elif interpolation_method == 'cubic':
                # Cubic interpolation
                aligned_series = price_series.reindex(
                    refresh_times_idx).interpolate(method='cubic')
            else:
                raise ValueError(f"Unrecognized interpolation method: {interpolation_method}. "
                                 f"Supported methods are 'previous', 'linear', 'cubic'.")

            # Add to DataFrame
            aligned_prices[f'Asset_{i}'] = aligned_series

        return aligned_prices, refresh_times_idx
    else:
        # Working with numpy arrays
        refresh_times_array = refresh_times_result

        # Process each price series
        aligned_prices = np.zeros((len(refresh_times_array), len(prices_list)))

        for i, (prices, times) in enumerate(zip(prices_list, times_list)):
            prices_array = np.asarray(prices)

            # Convert times to seconds if needed
            if isinstance(times, pd.DatetimeIndex):
                if convert_to_seconds:
                    if base_time is None:
                        # Use the first time point as base time
                        base_time = times[0]
                    times_array = wall2seconds(times, pd.Timestamp(base_time).timestamp())
                else:
                    # Convert refresh times to DatetimeIndex for alignment
                    if not isinstance(refresh_times_array, pd.DatetimeIndex):
                        if base_time is None:
                            raise ValueError("base_time must be provided to convert refresh times to datetime")
                        refresh_times_idx = seconds2wall(refresh_times_array, base_time)
                        # Align using pandas
                        price_series = pd.Series(prices_array, index=times)
                        if interpolation_method == 'previous':
                            aligned_series = price_series.reindex(
                                refresh_times_idx, method='pad'
                            )
                        elif interpolation_method == 'linear':
                            aligned_series = price_series.reindex(
                                refresh_times_idx, method='linear'
                            )
                        elif interpolation_method == 'cubic':
                            aligned_series = price_series.reindex(
                                refresh_times_idx).interpolate(method='cubic')
                        else:
                            raise ValueError(f"Unrecognized interpolation method: {interpolation_method}")

                        aligned_prices[:, i] = aligned_series.values
                        continue
            else:
                times_array = np.asarray(times)

            # Align prices to refresh times
            aligned_prices[:, i] = align_time(
                prices_array, times_array, refresh_times_array, interpolation_method
            )

        return aligned_prices, refresh_times_array


def refresh_time_returns(
    prices_list: List[Union[np.ndarray, pd.Series]],
    times_list: List[Union[np.ndarray, pd.Series, pd.DatetimeIndex]],
    return_type: str = 'log',
    interpolation_method: str = 'previous',
    convert_to_seconds: bool = True,
    base_time: Optional[Union[str, pd.Timestamp]] = None,
    return_pandas: bool = False
) -> Union[
    Tuple[np.ndarray, np.ndarray],
    Tuple[pd.DataFrame, pd.DatetimeIndex]
]:
    """
    Compute synchronized returns using refresh times.

    This function computes refresh times for the given time series,
    aligns price data to these synchronized time points, and then
    calculates returns from the synchronized prices.

    Args:
        prices_list: List of price series for each asset
        times_list: List of corresponding time points
        return_type: Type of returns to compute ('log', 'simple')
        interpolation_method: Method for interpolating prices ('previous', 'linear', 'cubic')
        convert_to_seconds: Whether to convert all times to seconds for processing
        base_time: Base time for datetime conversion
        return_pandas: Whether to return pandas objects instead of numpy arrays

    Returns:
        Tuple of (synchronized_returns, refresh_times) where synchronized_returns
        is a 2D array or DataFrame with each column representing an asset

    Raises:
        ValueError: If inputs have invalid dimensions or if parameters are invalid
        TypeError: If inputs have incorrect types

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from mfe.models.realized.refresh_time import refresh_time_returns
        >>> # Example with numeric times
        >>> prices1 = np.array([100, 101, 102, 103, 104])
        >>> times1 = np.array([1, 3, 5, 7, 9])
        >>> prices2 = np.array([50, 51, 52, 53, 54])
        >>> times2 = np.array([2, 4, 6, 8, 10])
        >>> synced_returns, refresh_times = refresh_time_returns(
        ...     [prices1, prices2], [times1, times2]
        ... )
        >>> refresh_times
        array([ 2.,  4.,  6.,  8., 10.])
        >>> synced_returns  # First row is NaN because returns require two prices
        array([[      nan,       nan],
               [0.00995033, 0.01980263],
               [0.00985222, 0.01960784],
               [0.00975709, 0.01941748],
               [0.00966184, 0.01923077]])
    """
    # Get synchronized prices
    synced_prices, refresh_times = refresh_time_prices(
        prices_list, times_list,
        interpolation_method=interpolation_method,
        convert_to_seconds=convert_to_seconds,
        base_time=base_time,
        return_pandas=return_pandas
    )

    # Compute returns
    if return_pandas and isinstance(synced_prices, pd.DataFrame):
        # Working with pandas objects
        if return_type.lower() == 'log':
            returns = np.log(synced_prices).diff()
        else:  # 'simple'
            returns = synced_prices.pct_change()

        return returns, refresh_times
    else:
        # Working with numpy arrays
        synced_prices_array = np.asarray(synced_prices)
        n_times, n_assets = synced_prices_array.shape

        # Initialize returns array
        returns = np.full((n_times, n_assets), np.nan)

        # Compute returns
        if return_type.lower() == 'log':
            # Log returns
            returns[1:] = np.diff(np.log(synced_prices_array), axis=0)
        else:  # 'simple'
            # Simple returns
            returns[1:] = np.diff(synced_prices_array, axis=0) / synced_prices_array[:-1]

        return returns, refresh_times


def visualize_refresh_times(
    prices_list: List[Union[np.ndarray, pd.Series]],
    times_list: List[Union[np.ndarray, pd.Series, pd.DatetimeIndex]],
    refresh_times: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None,
    asset_names: Optional[List[str]] = None,
    title: str = 'Refresh Time Synchronization',
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    show_grid: bool = True,
    convert_to_seconds: bool = True,
    base_time: Optional[Union[str, pd.Timestamp]] = None
) -> Optional[Any]:
    """
    Create visualization of refresh time synchronization.

    This function creates a plot showing the original price series and
    the synchronized prices at refresh times.

    Args:
        prices_list: List of price series for each asset
        times_list: List of corresponding time points
        refresh_times: Pre-computed refresh times (if None, they will be computed)
        asset_names: Names of assets for the legend
        title: Plot title
        figsize: Figure size as (width, height) in inches
        save_path: Path to save the figure (if None, figure is displayed)
        show_grid: Whether to show grid lines
        convert_to_seconds: Whether to convert all times to seconds for processing
        base_time: Base time for datetime conversion

    Returns:
        Matplotlib figure object if matplotlib is available, None otherwise

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from mfe.models.realized.refresh_time import visualize_refresh_times
        >>> # Example with numeric times
        >>> prices1 = np.array([100, 101, 102, 103, 104])
        >>> times1 = np.array([1, 3, 5, 7, 9])
        >>> prices2 = np.array([50, 51, 52, 53, 54])
        >>> times2 = np.array([2, 4, 6, 8, 10])
        >>> fig = visualize_refresh_times([prices1, prices2], [times1, times2])
    """
    if not HAS_MATPLOTLIB:
        logger.warning("Matplotlib is not available. Cannot create visualization.")
        return None

    # Compute refresh times if not provided
    if refresh_times is None:
        refresh_times = refresh_time(
            times_list,
            convert_to_seconds=convert_to_seconds,
            base_time=base_time,
            return_indices=False,
            return_pandas=False
        )

    # Get synchronized prices
    synced_prices, _ = refresh_time_prices(
        prices_list, times_list,
        interpolation_method='previous',
        convert_to_seconds=convert_to_seconds,
        base_time=base_time,
        return_pandas=False
    )

    # Determine if we're working with datetime
    is_datetime = False
    if isinstance(times_list[0], pd.DatetimeIndex):
        is_datetime = True
    elif isinstance(times_list[0], pd.Series) and pd.api.types.is_datetime64_dtype(times_list[0]):
        is_datetime = True
    elif isinstance(times_list[0], (list, np.ndarray)) and len(times_list[0]) > 0:
        if isinstance(times_list[0][0], (str, pd.Timestamp, np.datetime64)):
            is_datetime = True

    # Convert times to datetime for plotting if needed
    if is_datetime and not isinstance(refresh_times, pd.DatetimeIndex):
        if base_time is None:
            raise ValueError("base_time must be provided to convert refresh times to datetime")
        refresh_times_dt = seconds2wall(refresh_times, base_time)
    elif isinstance(refresh_times, pd.DatetimeIndex):
        refresh_times_dt = refresh_times
    else:
        # Use numeric times
        refresh_times_dt = None

    # Create figure
    fig, axes = plt.subplots(len(prices_list) + 1, 1, figsize=figsize, sharex=True)

    # Set asset names if not provided
    if asset_names is None:
        asset_names = [f"Asset {i+1}" for i in range(len(prices_list))]

    # Plot each asset's original prices
    for i, (prices, times, name) in enumerate(zip(prices_list, times_list, asset_names)):
        ax = axes[i]

        # Convert to arrays
        prices_array = np.asarray(prices)

        if is_datetime:
            # Convert to datetime for plotting
            if isinstance(times, pd.DatetimeIndex):
                times_dt = times
            elif isinstance(times, pd.Series) and pd.api.types.is_datetime64_dtype(times):
                times_dt = pd.DatetimeIndex(times)
            else:
                try:
                    times_dt = pd.DatetimeIndex(pd.to_datetime(times))
                except (ValueError, TypeError):
                    if base_time is None:
                        raise ValueError("base_time must be provided to convert times to datetime")
                    times_dt = seconds2wall(np.asarray(times), base_time)

            # Plot original prices
            ax.plot(times_dt, prices_array, 'o-', label=f'Original {name}', alpha=0.5)

            # Plot synchronized prices at refresh times
            if refresh_times_dt is not None:
                ax.plot(refresh_times_dt, synced_prices[:, i], 's-',
                        label=f'Synchronized {name}', linewidth=2)
        else:
            # Numeric times
            times_array = np.asarray(times)

            # Plot original prices
            ax.plot(times_array, prices_array, 'o-', label=f'Original {name}', alpha=0.5)

            # Plot synchronized prices at refresh times
            ax.plot(refresh_times, synced_prices[:, i], 's-',
                    label=f'Synchronized {name}', linewidth=2)

        # Add legend and grid
        ax.legend(loc='best')
        ax.set_ylabel(f'{name} Price')
        if show_grid:
            ax.grid(True, alpha=0.3)

    # Plot refresh times in the bottom panel
    ax = axes[-1]

    # Create a step function to visualize refresh time points
    if is_datetime and refresh_times_dt is not None:
        # Create a step function for refresh times
        for i, rt in enumerate(refresh_times_dt):
            ax.axvline(x=rt, color='r', linestyle='--', alpha=0.5)

        # Add markers at refresh times
        y_pos = np.ones(len(refresh_times_dt))
        ax.plot(refresh_times_dt, y_pos, 'ro', label='Refresh Times')

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        fig.autofmt_xdate()
    else:
        # Create a step function for refresh times
        for i, rt in enumerate(refresh_times):
            ax.axvline(x=rt, color='r', linestyle='--', alpha=0.5)

        # Add markers at refresh times
        y_pos = np.ones(len(refresh_times))
        ax.plot(refresh_times, y_pos, 'ro', label='Refresh Times')

    # Set y-axis limits and hide y-ticks for the refresh time panel
    ax.set_ylim(0.5, 1.5)
    ax.set_yticks([])
    ax.set_ylabel('Refresh\nTimes')
    ax.legend(loc='best')
    if show_grid:
        ax.grid(True, alpha=0.3)

    # Set overall title and x-label
    fig.suptitle(title, fontsize=16)
    ax.set_xlabel('Time')

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Save or display the figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    return fig


def refresh_time_statistics(
    times_list: List[Union[np.ndarray, pd.Series, pd.DatetimeIndex]],
    refresh_times: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None,
    convert_to_seconds: bool = True,
    base_time: Optional[Union[str, pd.Timestamp]] = None,
    return_pandas: bool = False
) -> Union[Dict[str, Any], pd.DataFrame]:
    """
    Compute statistics about refresh time sampling.

    This function analyzes the refresh time synchronization process,
    providing statistics about sampling frequency, data loss, and
    other relevant metrics.

    Args:
        times_list: List of time series for each asset
        refresh_times: Pre-computed refresh times (if None, they will be computed)
        convert_to_seconds: Whether to convert all times to seconds for processing
        base_time: Base time for datetime conversion
        return_pandas: Whether to return a pandas DataFrame instead of a dictionary

    Returns:
        Dictionary or DataFrame containing refresh time statistics

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from mfe.models.realized.refresh_time import refresh_time_statistics
        >>> # Example with numeric times
        >>> times1 = np.array([1, 3, 5, 7, 9])
        >>> times2 = np.array([2, 4, 6, 8, 10])
        >>> stats = refresh_time_statistics([times1, times2])
        >>> stats['total_observations']
        10
        >>> stats['refresh_time_count']
        5
        >>> stats['data_retention_pct']
        50.0
    """
    # Compute refresh times if not provided
    if refresh_times is None:
        refresh_times = refresh_time(
            times_list,
            convert_to_seconds=convert_to_seconds,
            base_time=base_time,
            return_indices=False,
            return_pandas=False
        )

    # Convert refresh_times to numpy array if it's a DatetimeIndex
    if isinstance(refresh_times, pd.DatetimeIndex):
        if convert_to_seconds:
            if base_time is None:
                # Use the first time point as base time
                base_time = refresh_times[0]
            refresh_times_array = wall2seconds(refresh_times, pd.Timestamp(base_time).timestamp())
        else:
            # Keep as DatetimeIndex for calculations
            refresh_times_array = np.array(refresh_times.astype(np.int64) / 1e9)
    else:
        refresh_times_array = np.asarray(refresh_times)

    # Process each time series
    processed_times = []

    for times in times_list:
        if isinstance(times, pd.DatetimeIndex):
            if convert_to_seconds:
                if base_time is None:
                    # Use the first time point as base time
                    base_time = times[0]
                times_array = wall2seconds(times, pd.Timestamp(base_time).timestamp())
            else:
                # Convert to seconds for calculations
                times_array = np.array(times.astype(np.int64) / 1e9)
        elif isinstance(times, pd.Series) and pd.api.types.is_datetime64_dtype(times):
            datetime_idx = pd.DatetimeIndex(times)
            if convert_to_seconds:
                if base_time is None:
                    # Use the first time point as base time
                    base_time = datetime_idx[0]
                times_array = wall2seconds(datetime_idx, pd.Timestamp(base_time).timestamp())
            else:
                # Convert to seconds for calculations
                times_array = np.array(datetime_idx.astype(np.int64) / 1e9)
        else:
            # Assume numpy array or list of numeric values
            times_array = np.asarray(times, dtype=float)

        processed_times.append(times_array)

    # Compute statistics
    stats = {}

    # Total number of observations across all assets
    total_obs = sum(len(times) for times in processed_times)
    stats['total_observations'] = total_obs

    # Number of refresh times
    refresh_count = len(refresh_times_array)
    stats['refresh_time_count'] = refresh_count

    # Data retention percentage
    retention_pct = (refresh_count * len(processed_times) / total_obs) * 100
    stats['data_retention_pct'] = retention_pct

    # Average time between refresh times
    if refresh_count > 1:
        avg_interval = np.mean(np.diff(refresh_times_array))
        stats['avg_refresh_interval'] = avg_interval
    else:
        stats['avg_refresh_interval'] = np.nan

    # Standard deviation of intervals
    if refresh_count > 2:
        std_interval = np.std(np.diff(refresh_times_array))
        stats['std_refresh_interval'] = std_interval
    else:
        stats['std_refresh_interval'] = np.nan

    # Min and max intervals
    if refresh_count > 1:
        min_interval = np.min(np.diff(refresh_times_array))
        max_interval = np.max(np.diff(refresh_times_array))
        stats['min_refresh_interval'] = min_interval
        stats['max_refresh_interval'] = max_interval
    else:
        stats['min_refresh_interval'] = np.nan
        stats['max_refresh_interval'] = np.nan

    # Statistics for each asset
    asset_stats = []

    for i, times in enumerate(processed_times):
        asset_stat = {}
        asset_stat['asset_index'] = i
        asset_stat['original_observations'] = len(times)

        # Compute average interval in original data
        if len(times) > 1:
            asset_stat['avg_original_interval'] = np.mean(np.diff(times))
        else:
            asset_stat['avg_original_interval'] = np.nan

        # Compute data retention for this asset
        asset_stat['retention_pct'] = (refresh_count / len(times)) * 100

        asset_stats.append(asset_stat)

    stats['asset_statistics'] = asset_stats

    # Return as pandas DataFrame if requested
    if return_pandas:
        # Create main stats DataFrame
        main_stats = pd.DataFrame({k: [v] for k, v in stats.items()
                                  if k != 'asset_statistics'})

        # Create asset stats DataFrame
        asset_df = pd.DataFrame(asset_stats)

        # Return both DataFrames
        return pd.concat([main_stats, asset_df], keys=['overall', 'by_asset'])
    else:
        return stats
