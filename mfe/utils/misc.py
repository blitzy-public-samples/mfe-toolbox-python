# mfe/utils/misc.py
"""
Miscellaneous Utility Functions Module

This module provides miscellaneous utility functions used throughout the MFE Toolbox,
including directory operations, lag matrix creation, density plotting, and random
character generation. These functions support various operations across the toolbox
that don't fit neatly into other utility categories.

The module implements optimized versions of common utility operations using NumPy's
efficient array operations and Numba's JIT compilation for performance-critical
functions. All functions include comprehensive type hints and input validation to
ensure reliability and proper error handling.

Functions:
    r2z: Transform correlation to Fisher's Z
    z2r: Transform Fisher's Z to correlation
    phi2r: Transform AR(1) parameter to correlation
    r2phi: Transform correlation to AR(1) parameter
    ensure_array: Ensure input is a NumPy array
    ensure_dataframe: Ensure input is a Pandas DataFrame
    ensure_series: Ensure input is a Pandas Series
    progress_bar: Display a progress bar
    format_time: Format time in seconds to a human-readable string
    plot_density: Plot the density of a distribution
    random_chars: Generate random characters
    get_mfe_path: Get the path to the MFE Toolbox installation
    create_directory: Create a directory if it doesn't exist
    list_files: List files in a directory with optional pattern matching
    lag_matrix_extended: Extended version of lag_matrix with additional options
"""

import logging
import os
import random
import re
import string
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast, overload

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from mfe.core.types import (
    TimeSeriesData, TimeSeriesDataFrame, Vector, Matrix, FilePath, DirectoryPath
)
from mfe.core.exceptions import (
    DimensionError, DataError, raise_dimension_error, raise_data_error
)
from mfe.core.validation import (
    validate_time_series, validate_vector, validate_matrix_shape,
    validate_input_time_series, validate_input_bounds
)
from mfe.utils.data_transformations import lag_matrix

# Set up module-level logger
logger = logging.getLogger("mfe.utils.misc")

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
    logger.debug("Numba available for miscellaneous utilities acceleration")
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    HAS_NUMBA = False
    logger.info("Numba not available. Miscellaneous utilities will use pure Python implementations.")


def r2z(r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Transform correlation to Fisher's Z.
    
    This function applies Fisher's Z transformation to correlation coefficients,
    which is useful for hypothesis testing and confidence intervals for correlations.
    The transformation is z = 0.5 * ln((1 + r) / (1 - r)).
    
    Args:
        r: Correlation coefficient(s) in the range [-1, 1]
        
    Returns:
        Fisher's Z transformation of the correlation coefficient(s)
        
    Raises:
        ValueError: If any correlation coefficient is outside the range [-1, 1]
        
    Examples:
        >>> from mfe.utils.misc import r2z
        >>> r2z(0.5)
        0.5493061443340548
        >>> import numpy as np
        >>> r2z(np.array([0.1, 0.5, 0.9]))
        array([0.10033535, 0.54930614, 1.47221948])
    """
    # Convert to numpy array if not already
    if not isinstance(r, np.ndarray):
        r_array = np.array(r)
    else:
        r_array = r
    
    # Check if correlations are in valid range
    if np.any(np.abs(r_array) > 1):
        raise ValueError("Correlation coefficients must be in the range [-1, 1]")
    
    # Apply Fisher's Z transformation
    # Use clipping to avoid numerical issues at exactly -1 or 1
    r_clipped = np.clip(r_array, -0.9999999, 0.9999999)
    z = 0.5 * np.log((1 + r_clipped) / (1 - r_clipped))
    
    # Return scalar if input was scalar
    if np.isscalar(r):
        return float(z.item())
    return z


def z2r(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Transform Fisher's Z to correlation.
    
    This function applies the inverse of Fisher's Z transformation to convert
    Z values back to correlation coefficients. The transformation is
    r = (exp(2z) - 1) / (exp(2z) + 1).
    
    Args:
        z: Fisher's Z value(s)
        
    Returns:
        Correlation coefficient(s) in the range [-1, 1]
        
    Examples:
        >>> from mfe.utils.misc import z2r
        >>> z2r(0.5493061443340548)
        0.5
        >>> import numpy as np
        >>> z2r(np.array([0.1, 0.5, 1.0]))
        array([0.09966799, 0.46211716, 0.76159416])
    """
    # Convert to numpy array if not already
    if not isinstance(z, np.ndarray):
        z_array = np.array(z)
    else:
        z_array = z
    
    # Apply inverse Fisher's Z transformation
    exp_2z = np.exp(2 * z_array)
    r = (exp_2z - 1) / (exp_2z + 1)
    
    # Ensure results are in valid range due to numerical precision
    r = np.clip(r, -1.0, 1.0)
    
    # Return scalar if input was scalar
    if np.isscalar(z):
        return float(r.item())
    return r


def phi2r(phi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Transform AR(1) parameter to correlation.
    
    This function transforms an AR(1) parameter to the corresponding correlation
    coefficient at lag 1. For a stationary AR(1) process, this is simply the
    AR parameter itself.
    
    Args:
        phi: AR(1) parameter(s) in the range [-1, 1] for stationarity
        
    Returns:
        Correlation coefficient(s) at lag 1
        
    Raises:
        ValueError: If any AR parameter is outside the range [-1, 1]
        
    Examples:
        >>> from mfe.utils.misc import phi2r
        >>> phi2r(0.5)
        0.5
        >>> import numpy as np
        >>> phi2r(np.array([0.1, 0.5, 0.9]))
        array([0.1, 0.5, 0.9])
    """
    # Convert to numpy array if not already
    if not isinstance(phi, np.ndarray):
        phi_array = np.array(phi)
    else:
        phi_array = phi
    
    # Check if AR parameters are in valid range for stationarity
    if np.any(np.abs(phi_array) > 1):
        raise ValueError("AR(1) parameters must be in the range [-1, 1] for stationarity")
    
    # For a stationary AR(1) process, the correlation at lag 1 is the AR parameter
    r = phi_array
    
    # Return scalar if input was scalar
    if np.isscalar(phi):
        return float(r.item())
    return r


def r2phi(r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Transform correlation to AR(1) parameter.
    
    This function transforms a correlation coefficient at lag 1 to the
    corresponding AR(1) parameter. For a stationary AR(1) process, this is
    simply the correlation coefficient itself.
    
    Args:
        r: Correlation coefficient(s) at lag 1 in the range [-1, 1]
        
    Returns:
        AR(1) parameter(s)
        
    Raises:
        ValueError: If any correlation coefficient is outside the range [-1, 1]
        
    Examples:
        >>> from mfe.utils.misc import r2phi
        >>> r2phi(0.5)
        0.5
        >>> import numpy as np
        >>> r2phi(np.array([0.1, 0.5, 0.9]))
        array([0.1, 0.5, 0.9])
    """
    # Convert to numpy array if not already
    if not isinstance(r, np.ndarray):
        r_array = np.array(r)
    else:
        r_array = r
    
    # Check if correlations are in valid range
    if np.any(np.abs(r_array) > 1):
        raise ValueError("Correlation coefficients must be in the range [-1, 1]")
    
    # For a stationary AR(1) process, the AR parameter is the correlation at lag 1
    phi = r_array
    
    # Return scalar if input was scalar
    if np.isscalar(r):
        return float(phi.item())
    return phi


def ensure_array(data: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """
    Ensure input is a NumPy array.
    
    This function converts the input to a NumPy array if it isn't already,
    optionally with a specified data type.
    
    Args:
        data: Input data to convert to a NumPy array
        dtype: NumPy data type to use for the array (optional)
        
    Returns:
        Input data as a NumPy array
        
    Examples:
        >>> from mfe.utils.misc import ensure_array
        >>> ensure_array([1, 2, 3])
        array([1, 2, 3])
        >>> import pandas as pd
        >>> ensure_array(pd.Series([1, 2, 3]))
        array([1, 2, 3])
    """
    if isinstance(data, np.ndarray) and (dtype is None or data.dtype == dtype):
        return data
    
    if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        return data.values if dtype is None else data.values.astype(dtype)
    
    return np.array(data) if dtype is None else np.array(data, dtype=dtype)


def ensure_dataframe(data: Any, index: Optional[Any] = None, columns: Optional[Any] = None) -> pd.DataFrame:
    """
    Ensure input is a Pandas DataFrame.
    
    This function converts the input to a Pandas DataFrame if it isn't already,
    optionally with specified index and column labels.
    
    Args:
        data: Input data to convert to a DataFrame
        index: Index to use for the DataFrame (optional)
        columns: Column labels to use for the DataFrame (optional)
        
    Returns:
        Input data as a Pandas DataFrame
        
    Examples:
        >>> from mfe.utils.misc import ensure_dataframe
        >>> ensure_dataframe([1, 2, 3])
           0
        0  1
        1  2
        2  3
        >>> import numpy as np
        >>> ensure_dataframe(np.array([[1, 2], [3, 4]]), columns=['A', 'B'])
           A  B
        0  1  2
        1  3  4
    """
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        if index is not None:
            df.index = index
        if columns is not None:
            df.columns = columns
        return df
    
    if isinstance(data, pd.Series):
        if columns is not None:
            return pd.DataFrame({columns[0]: data}) if isinstance(columns, (list, tuple)) else pd.DataFrame({columns: data})
        return pd.DataFrame(data)
    
    return pd.DataFrame(data, index=index, columns=columns)


def ensure_series(data: Any, index: Optional[Any] = None, name: Optional[str] = None) -> pd.Series:
    """
    Ensure input is a Pandas Series.
    
    This function converts the input to a Pandas Series if it isn't already,
    optionally with a specified index and name.
    
    Args:
        data: Input data to convert to a Series
        index: Index to use for the Series (optional)
        name: Name to use for the Series (optional)
        
    Returns:
        Input data as a Pandas Series
        
    Raises:
        ValueError: If the input is a DataFrame with multiple columns
        
    Examples:
        >>> from mfe.utils.misc import ensure_series
        >>> ensure_series([1, 2, 3])
        0    1
        1    2
        2    3
        dtype: int64
        >>> import numpy as np
        >>> ensure_series(np.array([1, 2, 3]), name='values')
        0    1
        1    2
        2    3
        Name: values, dtype: int64
    """
    if isinstance(data, pd.Series):
        series = data.copy()
        if index is not None:
            series.index = index
        if name is not None:
            series.name = name
        return series
    
    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError("Cannot convert DataFrame with multiple columns to Series")
        series = data.iloc[:, 0].copy()
        if index is not None:
            series.index = index
        if name is not None:
            series.name = name
        return series
    
    return pd.Series(data, index=index, name=name)


def progress_bar(iteration: int, total: int, prefix: str = '', suffix: str = '', 
                 decimals: int = 1, length: int = 50, fill: str = '█', 
                 print_end: str = '\r') -> None:
    """
    Display a progress bar in the console.
    
    This function prints a progress bar to show the progress of an operation.
    
    Args:
        iteration: Current iteration (0-based)
        total: Total iterations
        prefix: Prefix string
        suffix: Suffix string
        decimals: Decimal places for percentage
        length: Character length of the progress bar
        fill: Bar fill character
        print_end: End character (e.g., '\r', '\n')
        
    Examples:
        >>> import time
        >>> from mfe.utils.misc import progress_bar
        >>> # Example usage:
        >>> total = 10
        >>> for i in range(total):
        ...     # Simulate some work
        ...     time.sleep(0.1)
        ...     # Update progress bar
        ...     progress_bar(i + 1, total, prefix='Progress:', suffix='Complete', length=50)
    """
    percent = ('{0:.' + str(decimals) + 'f}').format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    
    # Print new line on completion
    if iteration == total:
        print()


def format_time(seconds: float) -> str:
    """
    Format time in seconds to a human-readable string.
    
    This function converts a time duration in seconds to a formatted string
    showing hours, minutes, and seconds as appropriate.
    
    Args:
        seconds: Time duration in seconds
        
    Returns:
        Formatted time string
        
    Examples:
        >>> from mfe.utils.misc import format_time
        >>> format_time(3661.5)
        '1h 1m 1.5s'
        >>> format_time(125.3)
        '2m 5.3s'
        >>> format_time(45.7)
        '45.7s'
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {seconds:.1f}s"
    
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"


def plot_density(data: Union[np.ndarray, pd.Series], 
                 ax: Optional[plt.Axes] = None,
                 kde: bool = True, 
                 hist: bool = True,
                 bins: Union[int, str, Sequence[float]] = 'auto',
                 hist_kws: Optional[Dict[str, Any]] = None,
                 kde_kws: Optional[Dict[str, Any]] = None,
                 title: Optional[str] = None,
                 xlabel: Optional[str] = None,
                 ylabel: Optional[str] = None) -> Tuple[Figure, Axes]:
    """
    Plot the density of a distribution.
    
    This function creates a plot showing the histogram and/or kernel density
    estimate of the input data.
    
    Args:
        data: Data to plot
        ax: Matplotlib axes to plot on (optional, creates new figure if None)
        kde: Whether to plot the kernel density estimate
        hist: Whether to plot the histogram
        bins: Number of bins for histogram or method to determine bins
        hist_kws: Additional keyword arguments for histogram
        kde_kws: Additional keyword arguments for kernel density estimate
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        
    Returns:
        Tuple of (figure, axes)
        
    Examples:
        >>> import numpy as np
        >>> from mfe.utils.misc import plot_density
        >>> # Example usage:
        >>> data = np.random.normal(0, 1, 1000)
        >>> fig, ax = plot_density(data, title='Normal Distribution', xlabel='Value')
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    # Convert to numpy array if not already
    if isinstance(data, pd.Series):
        data_array = data.values
    else:
        data_array = np.asarray(data)
    
    # Default keyword arguments
    hist_kws = hist_kws or {}
    kde_kws = kde_kws or {}
    
    # Plot histogram if requested
    if hist:
        hist_defaults = {
            'alpha': 0.6,
            'density': True,
            'edgecolor': 'k'
        }
        hist_defaults.update(hist_kws)
        ax.hist(data_array, bins=bins, **hist_defaults)
    
    # Plot kernel density estimate if requested
    if kde:
        kde_defaults = {
            'color': 'r',
            'linewidth': 2
        }
        kde_defaults.update(kde_kws)
        
        # Compute kernel density estimate
        x_grid = np.linspace(np.min(data_array), np.max(data_array), 1000)
        kde_result = stats.gaussian_kde(data_array)
        ax.plot(x_grid, kde_result(x_grid), **kde_defaults)
    
    # Set title and labels if provided
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel('Density')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    return fig, ax


@jit(nopython=True, cache=True)
def _random_chars_numba(length: int, chars: str) -> str:
    """
    Numba-accelerated implementation of random_chars.
    
    Args:
        length: Length of the random string
        chars: String of characters to choose from
        
    Returns:
        Random string of specified length
    """
    result = ""
    n_chars = len(chars)
    
    for _ in range(length):
        idx = int(np.random.random() * n_chars)
        result += chars[idx]
    
    return result


def random_chars(length: int, 
                 use_letters: bool = True, 
                 use_digits: bool = True, 
                 use_punctuation: bool = False,
                 custom_chars: Optional[str] = None) -> str:
    """
    Generate random characters.
    
    This function generates a random string of specified length using
    letters, digits, punctuation, or custom characters.
    
    Args:
        length: Length of the random string
        use_letters: Whether to include letters (a-z, A-Z)
        use_digits: Whether to include digits (0-9)
        use_punctuation: Whether to include punctuation
        custom_chars: Custom string of characters to choose from
        
    Returns:
        Random string of specified length
        
    Raises:
        ValueError: If length is negative or no character set is selected
        
    Examples:
        >>> from mfe.utils.misc import random_chars
        >>> # Generate a random alphanumeric string of length 10
        >>> random_chars(10)  # Example output: 'a7Xp2Kf9Qr'
        >>> # Generate a random string of digits only
        >>> random_chars(5, use_letters=False)  # Example output: '83947'
        >>> # Generate a random string with custom characters
        >>> random_chars(8, use_letters=False, use_digits=False, custom_chars='ABCDEF')  # Example output: 'BFCAEDCB'
    """
    if length < 0:
        raise ValueError("Length must be non-negative")
    
    # Build character set
    chars = ""
    if use_letters:
        chars += string.ascii_letters
    if use_digits:
        chars += string.digits
    if use_punctuation:
        chars += string.punctuation
    if custom_chars:
        chars += custom_chars
    
    if not chars:
        raise ValueError("At least one character set must be selected")
    
    # Use Numba-accelerated implementation if available
    if HAS_NUMBA:
        return _random_chars_numba(length, chars)
    
    # Pure Python implementation
    return ''.join(random.choice(chars) for _ in range(length))


def get_mfe_path() -> Path:
    """
    Get the path to the MFE Toolbox installation.
    
    This function returns the path to the MFE Toolbox installation directory.
    
    Returns:
        Path to the MFE Toolbox installation
        
    Examples:
        >>> from mfe.utils.misc import get_mfe_path
        >>> path = get_mfe_path()
        >>> print(path)  # Example output: '/path/to/mfe'
    """
    # Get the path to the mfe package
    import mfe
    return Path(os.path.dirname(os.path.abspath(mfe.__file__)))


def create_directory(path: Union[str, Path], parents: bool = True, exist_ok: bool = True) -> Path:
    """
    Create a directory if it doesn't exist.
    
    This function creates a directory at the specified path if it doesn't
    already exist.
    
    Args:
        path: Path to the directory to create
        parents: Whether to create parent directories if they don't exist
        exist_ok: Whether to ignore if the directory already exists
        
    Returns:
        Path to the created directory
        
    Raises:
        FileExistsError: If the directory exists and exist_ok is False
        
    Examples:
        >>> from mfe.utils.misc import create_directory
        >>> # Create a directory in the current working directory
        >>> path = create_directory('data')
        >>> print(path)  # Example output: '/current/working/directory/data'
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=parents, exist_ok=exist_ok)
    return path_obj


def list_files(directory: Union[str, Path], 
               pattern: Optional[str] = None, 
               recursive: bool = False,
               full_path: bool = False) -> List[str]:
    """
    List files in a directory with optional pattern matching.
    
    This function lists files in a directory, optionally filtering by a regex
    pattern and recursively searching subdirectories.
    
    Args:
        directory: Path to the directory to search
        pattern: Regular expression pattern to match filenames (optional)
        recursive: Whether to search subdirectories recursively
        full_path: Whether to return full paths or just filenames
        
    Returns:
        List of filenames or paths matching the criteria
        
    Raises:
        FileNotFoundError: If the directory doesn't exist
        
    Examples:
        >>> from mfe.utils.misc import list_files
        >>> # List all Python files in the current directory
        >>> list_files('.', pattern=r'.*\.py$')  # Example output: ['setup.py', 'example.py']
        >>> # List all files recursively with full paths
        >>> list_files('.', recursive=True, full_path=True)  # Example output: ['/path/to/file1.txt', '/path/to/subdir/file2.txt']
    """
    directory_path = Path(directory)
    
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if not directory_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    
    # Compile regex pattern if provided
    pattern_regex = re.compile(pattern) if pattern else None
    
    result = []
    
    if recursive:
        # Walk through directory and subdirectories
        for root, _, files in os.walk(directory_path):
            root_path = Path(root)
            for file in files:
                if pattern_regex and not pattern_regex.match(file):
                    continue
                
                if full_path:
                    result.append(str(root_path / file))
                else:
                    result.append(file)
    else:
        # List files in directory only
        for file in directory_path.iterdir():
            if not file.is_file():
                continue
            
            if pattern_regex and not pattern_regex.match(file.name):
                continue
            
            if full_path:
                result.append(str(file))
            else:
                result.append(file.name)
    
    return sorted(result)


@validate_input_time_series(0)
@validate_input_bounds(1, lower_bound=1, param_name="lags")
def lag_matrix_extended(data: TimeSeriesData,
                        lags: int,
                        include_original: bool = True,
                        fill_value: Optional[float] = None,
                        drop_nan: bool = False,
                        column_names: Optional[List[str]] = None) -> Union[np.ndarray, pd.DataFrame]:
    """
    Extended version of lag_matrix with additional options.
    
    This function creates a matrix where each column is a lagged version of the
    input time series, with additional options for handling missing values and
    naming columns.
    
    Args:
        data: Time series data (NumPy array or Pandas Series)
        lags: Number of lags to include
        include_original: Whether to include the original series as the first column
        fill_value: Value to use for missing values due to lagging (None for NaN)
        drop_nan: Whether to drop rows with NaN values
        column_names: Custom names for columns (optional)
        
    Returns:
        Matrix with lagged series as columns, with the same type as the input
        
    Raises:
        DimensionError: If the input is not 1D
        ValueError: If lags is less than 1
        ValueError: If column_names is provided but has wrong length
        
    Examples:
        >>> import numpy as np
        >>> from mfe.utils.misc import lag_matrix_extended
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> lag_matrix_extended(x, lags=2, drop_nan=True)
        array([[3., 2., 1.],
               [4., 3., 2.],
               [5., 4., 3.]])
        
        >>> import pandas as pd
        >>> s = pd.Series([1, 2, 3, 4, 5], index=pd.date_range('2020-01-01', periods=5))
        >>> lag_matrix_extended(s, lags=2, column_names=['Current', 'Lag1', 'Lag2'])
        <BLANKLINE>
                   Current  Lag1  Lag2
        2020-01-01     1.0   NaN   NaN
        2020-01-02     2.0   1.0   NaN
        2020-01-03     3.0   2.0   1.0
        2020-01-04     4.0   3.0   2.0
        2020-01-05     5.0   4.0   3.0
    """
    # Get the basic lag matrix
    result = lag_matrix(data, lags, include_original, fill_value)
    
    # Determine number of columns
    n_cols = lags + 1 if include_original else lags
    
    # Validate column names if provided
    if column_names is not None:
        if len(column_names) != n_cols:
            raise ValueError(f"Length of column_names ({len(column_names)}) must match number of columns ({n_cols})")
    
    # Handle pandas DataFrame result
    if isinstance(result, pd.DataFrame):
        # Rename columns if names provided
        if column_names is not None:
            result.columns = column_names
        
        # Drop rows with NaN values if requested
        if drop_nan:
            result = result.dropna()
        
        return result
    
    # Handle numpy array result
    else:
        # Drop rows with NaN values if requested
        if drop_nan:
            # Find rows without NaN values
            mask = ~np.isnan(result).any(axis=1)
            result = result[mask]
        
        return result


# Register Numba-accelerated functions if available
 def _register_numba_functions() -> None:
    """
    Register Numba JIT-compiled functions for miscellaneous utilities.
    
    This function is called during module initialization to register
    performance-critical functions for JIT compilation if Numba is available.
    """
    if HAS_NUMBA:
        # The functions are already decorated with @jit, so no additional
        # registration is needed here. This function is kept for consistency
        # with the module structure and potential future enhancements.
        logger.debug("Miscellaneous utilities Numba JIT functions registered")
    else:
        logger.info("Numba not available. Miscellaneous utilities will use pure Python implementations.")


# Initialize the module
_register_numba_functions()
