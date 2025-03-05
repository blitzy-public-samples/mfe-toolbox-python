# mfe/models/time_series/plots.py
"""
Time Series Plotting Module

This module provides specialized plotting functions for time series analysis, offering
a comprehensive set of visualization tools for model diagnostics, forecasting, and data
exploration. It extends matplotlib's capabilities with time series-specific enhancements
and integrates seamlessly with the MFE Toolbox's time series models.

The module includes functions for:
- ACF/PACF plots with confidence intervals
- Residual diagnostics including standardized residuals and QQ plots
- Forecast visualization with prediction intervals
- Model comparison plots
- Time series decomposition visualization
- Impulse response function plots
- Interactive exploratory analysis tools

All functions include comprehensive input validation, support for both NumPy arrays
and Pandas objects, and customizable styling options. The module is designed to work
with the time series models in the MFE Toolbox but can also be used independently
for general time series visualization.

Functions:
    plot_series: Plot one or more time series with proper formatting
    plot_acf_pacf: Plot autocorrelation and partial autocorrelation functions
    plot_residual_diagnostics: Create comprehensive residual diagnostic plots
    plot_forecast: Visualize forecasts with prediction intervals
    plot_model_comparison: Compare multiple model forecasts
    plot_decomposition: Visualize time series decomposition components
    plot_impulse_response: Plot impulse response functions
    plot_qq: Create a Q-Q plot for residual normality assessment
    plot_rolling_statistics: Plot rolling mean and standard deviation
    plot_seasonal: Visualize seasonal patterns in time series
    create_lag_plot: Create a lag plot for time series
    plot_periodogram: Plot the periodogram of a time series
    plot_prediction_error: Visualize prediction errors over time
    plot_cusum: Plot CUSUM test for parameter stability
    plot_model_diagnostics: Create comprehensive model diagnostic plots
"""

import logging
import warnings
from typing import (
    Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, 
    Union, cast, overload
)

import numpy as np
import pandas as pd
from scipy import stats, signal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

from mfe.core.exceptions import (
    ParameterError, DimensionError, NumericError, 
    warn_numeric
)
from mfe.models.time_series.correlation import (
    acf, pacf, cross_correlation, plot_acf, plot_pacf, plot_ccf
)

# Set up module-level logger
logger = logging.getLogger("mfe.models.time_series.plots")


def plot_series(
    data: Union[np.ndarray, pd.Series, pd.DataFrame],
    dates: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None,
    title: Optional[str] = None,
    xlabel: str = "Time",
    ylabel: str = "Value",
    figsize: Tuple[float, float] = (10, 6),
    ax: Optional[plt.Axes] = None,
    style: Optional[Union[str, List[str]]] = None,
    alpha: float = 0.8,
    grid: bool = True,
    legend: bool = True,
    legend_loc: str = "best",
    color_map: Optional[str] = None,
    markers: Optional[Union[str, List[str]]] = None,
    linewidth: Union[float, List[float]] = 1.5,
    highlight_periods: Optional[List[Tuple[int, int, str]]] = None,
    highlight_points: Optional[List[Tuple[int, str, str]]] = None,
    date_format: Optional[str] = None,
    y_lim: Optional[Tuple[float, float]] = None,
    secondary_y: Optional[Union[str, List[str]]] = None,
    fill_between: Optional[Tuple[Union[np.ndarray, pd.Series], Union[np.ndarray, pd.Series], Dict[str, Any]]] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot one or more time series with proper formatting.
    
    This function creates a well-formatted time series plot with support for multiple
    series, date formatting, highlighting specific periods or points, and various
    customization options.
    
    Args:
        data: Time series data to plot (array, Series, or DataFrame)
        dates: Optional dates for x-axis (array or DatetimeIndex)
        title: Plot title
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        figsize: Figure size as (width, height) in inches
        ax: Existing axes to plot on (if None, creates new figure)
        style: Line style(s) for the plot
        alpha: Transparency level for the plot
        grid: Whether to show grid lines
        legend: Whether to show legend
        legend_loc: Location of the legend
        color_map: Matplotlib colormap name for multiple series
        markers: Marker style(s) for the plot
        linewidth: Line width(s) for the plot
        highlight_periods: List of (start, end, color) tuples for highlighting periods
        highlight_points: List of (index, color, label) tuples for highlighting points
        date_format: Format string for date ticks
        y_lim: Y-axis limits as (min, max)
        secondary_y: Column name(s) to plot on secondary y-axis
        fill_between: Tuple of (lower_bound, upper_bound, kwargs) for fill_between
    
    Returns:
        Tuple containing the figure and axes objects
    
    Raises:
        ValueError: If input data is invalid
        
    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from mfe.models.time_series.plots import plot_series
        >>> # Create sample data
        >>> dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        >>> data = pd.DataFrame({
        ...     'Series1': np.cumsum(np.random.normal(0, 1, 100)),
        ...     'Series2': np.cumsum(np.random.normal(0, 2, 100))
        ... }, index=dates)
        >>> # Plot the series
        >>> fig, ax = plot_series(data, title='Time Series Plot', 
        ...                       highlight_periods=[(20, 40, 'lightblue')])
    """
    # Input validation and conversion
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        if dates is None and isinstance(df.index, pd.DatetimeIndex):
            dates = df.index
        series_names = df.columns.tolist()
        is_pandas = True
    elif isinstance(data, pd.Series):
        df = pd.DataFrame(data)
        if dates is None and isinstance(df.index, pd.DatetimeIndex):
            dates = df.index
        series_names = [data.name if data.name is not None else 'Series']
        is_pandas = True
    else:
        # Convert numpy array to DataFrame
        data_array = np.asarray(data)
        if data_array.ndim == 1:
            df = pd.DataFrame(data_array, columns=['Series'])
            series_names = ['Series']
        else:
            df = pd.DataFrame(data_array, columns=[f'Series{i+1}' for i in range(data_array.shape[1])])
            series_names = df.columns.tolist()
        is_pandas = False
    
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Create secondary y-axis if needed
    if secondary_y is not None:
        ax2 = ax.twinx()
        if isinstance(secondary_y, str):
            secondary_y = [secondary_y]
        primary_cols = [col for col in df.columns if col not in secondary_y]
        secondary_cols = [col for col in df.columns if col in secondary_y]
    else:
        ax2 = None
        primary_cols = df.columns.tolist()
        secondary_cols = []
    
    # Set up colors
    n_series = len(primary_cols) + len(secondary_cols)
    if color_map is not None:
        cmap = plt.get_cmap(color_map)
        colors = [cmap(i / max(1, n_series - 1)) for i in range(n_series)]
    else:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # Cycle colors if needed
        if n_series > len(colors):
            colors = colors * (n_series // len(colors) + 1)
        colors = colors[:n_series]
    
    # Set up styles
    if style is None:
        styles = ['-'] * n_series
    elif isinstance(style, str):
        styles = [style] * n_series
    else:
        styles = style
        if len(styles) < n_series:
            styles = styles * (n_series // len(styles) + 1)
        styles = styles[:n_series]
    
    # Set up markers
    if markers is None:
        markers_list = [None] * n_series
    elif isinstance(markers, str):
        markers_list = [markers] * n_series
    else:
        markers_list = markers
        if len(markers_list) < n_series:
            markers_list = markers_list * (n_series // len(markers_list) + 1)
        markers_list = markers_list[:n_series]
    
    # Set up linewidths
    if isinstance(linewidth, (int, float)):
        linewidths = [linewidth] * n_series
    else:
        linewidths = linewidth
        if len(linewidths) < n_series:
            linewidths = linewidths * (n_series // len(linewidths) + 1)
        linewidths = linewidths[:n_series]
    
    # Plot primary y-axis series
    for i, col in enumerate(primary_cols):
        if dates is not None:
            ax.plot(dates, df[col], label=col, color=colors[i], 
                    linestyle=styles[i], marker=markers_list[i], 
                    linewidth=linewidths[i], alpha=alpha)
        else:
            ax.plot(df[col], label=col, color=colors[i], 
                    linestyle=styles[i], marker=markers_list[i], 
                    linewidth=linewidths[i], alpha=alpha)
    
    # Plot secondary y-axis series
    if ax2 is not None:
        for i, col in enumerate(secondary_cols):
            idx = i + len(primary_cols)
            if dates is not None:
                ax2.plot(dates, df[col], label=f"{col} (right)", color=colors[idx], 
                        linestyle=styles[idx], marker=markers_list[idx], 
                        linewidth=linewidths[idx], alpha=alpha)
            else:
                ax2.plot(df[col], label=f"{col} (right)", color=colors[idx], 
                        linestyle=styles[idx], marker=markers_list[idx], 
                        linewidth=linewidths[idx], alpha=alpha)
        ax2.set_ylabel(' / '.join(secondary_cols))
    
    # Add fill_between if provided
    if fill_between is not None:
        lower_bound, upper_bound, kwargs = fill_between
        if dates is not None:
            ax.fill_between(dates, lower_bound, upper_bound, **kwargs)
        else:
            ax.fill_between(range(len(lower_bound)), lower_bound, upper_bound, **kwargs)
    
    # Highlight specific periods if provided
    if highlight_periods is not None:
        for start, end, color in highlight_periods:
            if dates is not None:
                ax.axvspan(dates[start], dates[end], alpha=0.3, color=color)
            else:
                ax.axvspan(start, end, alpha=0.3, color=color)
    
    # Highlight specific points if provided
    if highlight_points is not None:
        for idx, color, label in highlight_points:
            if dates is not None:
                ax.scatter(dates[idx], df.iloc[idx], color=color, s=80, zorder=5, label=label)
            else:
                ax.scatter(idx, df.iloc[idx], color=color, s=80, zorder=5, label=label)
    
    # Set up date formatting if dates are provided
    if dates is not None:
        if isinstance(dates, pd.DatetimeIndex) or np.issubdtype(dates.dtype, np.datetime64):
            # Format date ticks
            if date_format is not None:
                date_formatter = mdates.DateFormatter(date_format)
                ax.xaxis.set_major_formatter(date_formatter)
            else:
                # Auto-select date format based on date range
                locator = mdates.AutoDateLocator()
                formatter = mdates.AutoDateFormatter(locator)
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(formatter)
            
            # Rotate date labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    
    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    
    # Set y-axis limits if provided
    if y_lim is not None:
        ax.set_ylim(y_lim)
    
    # Add grid
    if grid:
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if ax2 is not None:
            handles2, labels2 = ax2.get_legend_handles_labels()
            handles += handles2
            labels += labels2
        ax.legend(handles, labels, loc=legend_loc)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, ax


def plot_acf_pacf(
    data: Union[np.ndarray, pd.Series, pd.DataFrame],
    lags: Optional[int] = None,
    alpha: float = 0.05,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 6),
    method: str = 'yule_walker',
    pacf_method: Optional[str] = None,
    adjusted: bool = False,
    fft: bool = False,
    missing: str = 'none',
    bartlett_confint: bool = True,
    series_names: Optional[List[str]] = None,
    grid: bool = True,
    zero_line: bool = True,
    show_significance: bool = True,
    significance_color: str = 'red',
    acf_kwargs: Optional[Dict[str, Any]] = None,
    pacf_kwargs: Optional[Dict[str, Any]] = None
) -> plt.Figure:
    """
    Plot autocorrelation and partial autocorrelation functions side by side.
    
    This function creates a side-by-side plot of the autocorrelation function (ACF)
    and partial autocorrelation function (PACF) for a time series, including confidence
    intervals and significance lines.
    
    Args:
        data: Time series data (array, Series, or DataFrame)
        lags: Number of lags to include (default: min(10*log10(n), n-1))
        alpha: Significance level for confidence intervals
        title: Plot title
        figsize: Figure size as (width, height) in inches
        method: Method for computing ACF ('standard' or 'robust')
        pacf_method: Method for computing PACF ('yule_walker', 'ols', or 'burg')
        adjusted: Whether to use adjusted sample size for standard errors
        fft: Whether to use FFT for ACF computation
        missing: How to handle missing values ('none', 'drop', or 'raise')
        bartlett_confint: Whether to use Bartlett's formula for confidence intervals
        series_names: Names for series in multivariate case
        grid: Whether to show grid lines
        zero_line: Whether to show horizontal line at zero
        show_significance: Whether to show significance lines
        significance_color: Color for significance lines
        acf_kwargs: Additional keyword arguments for ACF computation
        pacf_kwargs: Additional keyword arguments for PACF computation
    
    Returns:
        Figure object containing the plot
    
    Raises:
        ValueError: If input data is invalid
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.time_series.plots import plot_acf_pacf
        >>> # Create sample data
        >>> np.random.seed(42)
        >>> ar_data = np.zeros(100)
        >>> for i in range(1, 100):
        ...     ar_data[i] = 0.7 * ar_data[i-1] + np.random.normal(0, 1)
        >>> # Plot ACF and PACF
        >>> fig = plot_acf_pacf(ar_data, lags=20, title='AR(1) Process')
    """
    # Set default values for kwargs
    if acf_kwargs is None:
        acf_kwargs = {}
    if pacf_kwargs is None:
        pacf_kwargs = {}
    
    # Set default PACF method if not provided
    if pacf_method is None:
        pacf_method = method if method in ['yule_walker', 'ols'] else 'yule_walker'
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Check if input is multivariate
    is_multivariate = False
    if isinstance(data, pd.DataFrame) and data.shape[1] > 1:
        is_multivariate = True
    elif isinstance(data, np.ndarray) and data.ndim > 1 and data.shape[1] > 1:
        is_multivariate = True
    
    # For multivariate data, we'll only plot ACF/PACF for each series individually
    if is_multivariate:
        if isinstance(data, pd.DataFrame):
            if series_names is None:
                series_names = data.columns.tolist()
            n_series = data.shape[1]
        else:
            if series_names is None:
                series_names = [f'Series {i+1}' for i in range(data.shape[1])]
            n_series = data.shape[1]
        
        # Create a new figure with subplots for each series
        plt.close(fig)
        fig, axes = plt.subplots(n_series, 2, figsize=(12, 5 * n_series))
        
        # Plot ACF/PACF for each series
        for i in range(n_series):
            if isinstance(data, pd.DataFrame):
                series_data = data.iloc[:, i]
            else:
                series_data = data[:, i]
            
            # Compute and plot ACF
            acf_result = acf(
                series_data, nlags=lags, alpha=alpha, adjusted=adjusted,
                fft=fft, missing=missing, demean=True, bartlett_confint=bartlett_confint,
                method=method, **acf_kwargs
            )
            
            # Compute and plot PACF
            pacf_result = pacf(
                series_data, nlags=lags, alpha=alpha, method=pacf_method,
                demean=True, **pacf_kwargs
            )
            
            # Extract results
            acf_values = acf_result['acf']
            acf_confint = acf_result['confint']
            pacf_values = pacf_result['pacf']
            pacf_confint = pacf_result['confint']
            lags_values = acf_result['lags']
            
            # Convert to numpy arrays if pandas
            if isinstance(acf_values, pd.Series):
                acf_values = acf_values.values
            if isinstance(acf_confint, pd.DataFrame):
                acf_confint = acf_confint.values
            if isinstance(pacf_values, pd.Series):
                pacf_values = pacf_values.values
            if isinstance(pacf_confint, pd.DataFrame):
                pacf_confint = pacf_confint.values
            
            # Plot ACF
            ax_acf = axes[i, 0]
            ax_acf.bar(lags_values, acf_values, width=0.3, color='steelblue', alpha=0.8)
            
            # Add confidence intervals
            ax_acf.fill_between(
                lags_values,
                acf_confint[:, 0],
                acf_confint[:, 1],
                color='steelblue',
                alpha=0.2
            )
            
            # Add significance lines
            if show_significance:
                ax_acf.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                sig_level = stats.norm.ppf(1 - alpha / 2) / np.sqrt(len(series_data))
                ax_acf.axhline(y=sig_level, color=significance_color, linestyle='--', linewidth=1)
                ax_acf.axhline(y=-sig_level, color=significance_color, linestyle='--', linewidth=1)
            elif zero_line:
                ax_acf.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            # Set labels and title
            ax_acf.set_xlabel('Lag')
            ax_acf.set_ylabel('Autocorrelation')
            ax_acf.set_title(f'ACF - {series_names[i]}')
            
            # Set y-limits to be symmetric around zero for lags > 0
            y_max = max(1.0, np.max(np.abs(acf_values[1:])) * 1.1)
            ax_acf.set_ylim(-y_max, 1.05)
            
            # Add grid
            if grid:
                ax_acf.grid(True, linestyle='--', alpha=0.7)
            
            # Plot PACF
            ax_pacf = axes[i, 1]
            ax_pacf.bar(lags_values, pacf_values, width=0.3, color='firebrick', alpha=0.8)
            
            # Add confidence intervals
            ax_pacf.fill_between(
                lags_values,
                pacf_confint[:, 0],
                pacf_confint[:, 1],
                color='firebrick',
                alpha=0.2
            )
            
            # Add significance lines
            if show_significance:
                ax_pacf.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                sig_level = stats.norm.ppf(1 - alpha / 2) / np.sqrt(len(series_data))
                ax_pacf.axhline(y=sig_level, color=significance_color, linestyle='--', linewidth=1)
                ax_pacf.axhline(y=-sig_level, color=significance_color, linestyle='--', linewidth=1)
            elif zero_line:
                ax_pacf.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            # Set labels and title
            ax_pacf.set_xlabel('Lag')
            ax_pacf.set_ylabel('Partial Autocorrelation')
            ax_pacf.set_title(f'PACF - {series_names[i]}')
            
            # Set y-limits to be symmetric around zero for lags > 0
            y_max = max(1.0, np.max(np.abs(pacf_values[1:])) * 1.1)
            ax_pacf.set_ylim(-y_max, 1.05)
            
            # Add grid
            if grid:
                ax_pacf.grid(True, linestyle='--', alpha=0.7)
    else:
        # For univariate data, plot ACF/PACF side by side
        # Compute ACF
        acf_result = acf(
            data, nlags=lags, alpha=alpha, adjusted=adjusted,
            fft=fft, missing=missing, demean=True, bartlett_confint=bartlett_confint,
            method=method, **acf_kwargs
        )
        
        # Compute PACF
        pacf_result = pacf(
            data, nlags=lags, alpha=alpha, method=pacf_method,
            demean=True, **pacf_kwargs
        )
        
        # Extract results
        acf_values = acf_result['acf']
        acf_confint = acf_result['confint']
        pacf_values = pacf_result['pacf']
        pacf_confint = pacf_result['confint']
        lags_values = acf_result['lags']
        
        # Convert to numpy arrays if pandas
        if isinstance(acf_values, pd.Series):
            acf_values = acf_values.values
        if isinstance(acf_confint, pd.DataFrame):
            acf_confint = acf_confint.values
        if isinstance(pacf_values, pd.Series):
            pacf_values = pacf_values.values
        if isinstance(pacf_confint, pd.DataFrame):
            pacf_confint = pacf_confint.values
        
        # Plot ACF
        ax_acf = axes[0]
        ax_acf.bar(lags_values, acf_values, width=0.3, color='steelblue', alpha=0.8)
        
        # Add confidence intervals
        ax_acf.fill_between(
            lags_values,
            acf_confint[:, 0],
            acf_confint[:, 1],
            color='steelblue',
            alpha=0.2
        )
        
        # Add significance lines
        if show_significance:
            ax_acf.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            sig_level = stats.norm.ppf(1 - alpha / 2) / np.sqrt(len(data))
            ax_acf.axhline(y=sig_level, color=significance_color, linestyle='--', linewidth=1)
            ax_acf.axhline(y=-sig_level, color=significance_color, linestyle='--', linewidth=1)
        elif zero_line:
            ax_acf.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Set labels and title
        ax_acf.set_xlabel('Lag')
        ax_acf.set_ylabel('Autocorrelation')
        ax_acf.set_title('Autocorrelation Function (ACF)')
        
        # Set y-limits to be symmetric around zero for lags > 0
        y_max = max(1.0, np.max(np.abs(acf_values[1:])) * 1.1)
        ax_acf.set_ylim(-y_max, 1.05)
        
        # Add grid
        if grid:
            ax_acf.grid(True, linestyle='--', alpha=0.7)
        
        # Plot PACF
        ax_pacf = axes[1]
        ax_pacf.bar(lags_values, pacf_values, width=0.3, color='firebrick', alpha=0.8)
        
        # Add confidence intervals
        ax_pacf.fill_between(
            lags_values,
            pacf_confint[:, 0],
            pacf_confint[:, 1],
            color='firebrick',
            alpha=0.2
        )
        
        # Add significance lines
        if show_significance:
            ax_pacf.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            sig_level = stats.norm.ppf(1 - alpha / 2) / np.sqrt(len(data))
            ax_pacf.axhline(y=sig_level, color=significance_color, linestyle='--', linewidth=1)
            ax_pacf.axhline(y=-sig_level, color=significance_color, linestyle='--', linewidth=1)
        elif zero_line:
            ax_pacf.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Set labels and title
        ax_pacf.set_xlabel('Lag')
        ax_pacf.set_ylabel('Partial Autocorrelation')
        ax_pacf.set_title('Partial Autocorrelation Function (PACF)')
        
        # Set y-limits to be symmetric around zero for lags > 0
        y_max = max(1.0, np.max(np.abs(pacf_values[1:])) * 1.1)
        ax_pacf.set_ylim(-y_max, 1.05)
        
        # Add grid
        if grid:
            ax_pacf.grid(True, linestyle='--', alpha=0.7)
    
    # Set overall title if provided
    if title:
        fig.suptitle(title, fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    if title:
        plt.subplots_adjust(top=0.9)
    
    return fig


def plot_residual_diagnostics(
    residuals: Union[np.ndarray, pd.Series],
    standardized: bool = True,
    dates: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 10),
    bins: int = 30,
    acf_lags: Optional[int] = None,
    alpha: float = 0.05,
    grid: bool = True,
    model_name: Optional[str] = None,
    test_normality: bool = True,
    test_autocorrelation: bool = True,
    test_heteroskedasticity: bool = True,
    add_stats: bool = True,
    add_loess: bool = True,
    loess_frac: float = 0.3,
    color: str = 'steelblue',
    hist_kwargs: Optional[Dict[str, Any]] = None,
    qq_kwargs: Optional[Dict[str, Any]] = None,
    acf_kwargs: Optional[Dict[str, Any]] = None
) -> plt.Figure:
    """
    Create comprehensive residual diagnostic plots.
    
    This function creates a set of diagnostic plots for model residuals, including
    time series plot, histogram, Q-Q plot, and autocorrelation function. It also
    optionally performs statistical tests for normality, autocorrelation, and
    heteroskedasticity.
    
    Args:
        residuals: Residuals to analyze
        standardized: Whether to standardize residuals
        dates: Optional dates for time series plot
        title: Overall plot title
        figsize: Figure size as (width, height) in inches
        bins: Number of bins for histogram
        acf_lags: Number of lags for ACF plot
        alpha: Significance level for tests and confidence intervals
        grid: Whether to show grid lines
        model_name: Name of the model for title
        test_normality: Whether to perform normality test
        test_autocorrelation: Whether to perform autocorrelation test
        test_heteroskedasticity: Whether to perform heteroskedasticity test
        add_stats: Whether to add descriptive statistics to plots
        add_loess: Whether to add LOESS smoothing to residual plot
        loess_frac: Fraction of points to use for LOESS smoothing
        color: Base color for plots
        hist_kwargs: Additional keyword arguments for histogram
        qq_kwargs: Additional keyword arguments for Q-Q plot
        acf_kwargs: Additional keyword arguments for ACF plot
    
    Returns:
        Figure object containing the diagnostic plots
    
    Raises:
        ValueError: If input data is invalid
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.time_series.plots import plot_residual_diagnostics
        >>> # Create sample residuals
        >>> np.random.seed(42)
        >>> residuals = np.random.normal(0, 1, 100)
        >>> # Plot diagnostics
        >>> fig = plot_residual_diagnostics(residuals, model_name='ARMA(1,1)')
    """
    # Set default values for kwargs
    if hist_kwargs is None:
        hist_kwargs = {}
    if qq_kwargs is None:
        qq_kwargs = {}
    if acf_kwargs is None:
        acf_kwargs = {}
    
    # Convert to numpy array if needed
    if isinstance(residuals, pd.Series):
        if dates is None and isinstance(residuals.index, pd.DatetimeIndex):
            dates = residuals.index
        residuals_array = residuals.values
    else:
        residuals_array = np.asarray(residuals)
    
    # Standardize residuals if requested
    if standardized:
        mean = np.mean(residuals_array)
        std = np.std(residuals_array)
        if std > 0:
            std_residuals = (residuals_array - mean) / std
        else:
            std_residuals = residuals_array - mean
            warnings.warn("Standard deviation of residuals is zero. Using mean-centered residuals.")
    else:
        std_residuals = residuals_array
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2)
    
    # Plot 1: Residuals over time
    ax1 = plt.subplot(gs[0, 0])
    if dates is not None:
        ax1.plot(dates, std_residuals, 'o-', color=color, alpha=0.7, markersize=4)
        # Format date ticks
        locator = mdates.AutoDateLocator()
        formatter = mdates.AutoDateFormatter(locator)
        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_major_formatter(formatter)
        plt.setp(ax1.get_xticklabels(), rotation=30, ha='right')
    else:
        ax1.plot(std_residuals, 'o-', color=color, alpha=0.7, markersize=4)
    
    # Add horizontal lines at 0 and ±2
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax1.axhline(y=2, color='red', linestyle='--', linewidth=0.8)
    ax1.axhline(y=-2, color='red', linestyle='--', linewidth=0.8)
    
    # Add LOESS smoothing if requested
    if add_loess:
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            x = np.arange(len(std_residuals))
            z = lowess(std_residuals, x, frac=loess_frac)
            ax1.plot(z[:, 0], z[:, 1], 'r-', linewidth=2)
        except ImportError:
            warnings.warn("statsmodels is required for LOESS smoothing. Skipping.")
    
    # Set labels and title
    if standardized:
        ax1.set_ylabel('Standardized Residuals')
        ax1.set_title('Standardized Residuals Over Time')
    else:
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals Over Time')
    ax1.set_xlabel('Time')
    
    # Add grid
    if grid:
        ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Histogram of residuals
    ax2 = plt.subplot(gs[0, 1])
    hist_defaults = {'bins': bins, 'alpha': 0.7, 'color': color, 'density': True}
    hist_defaults.update(hist_kwargs)
    ax2.hist(std_residuals, **hist_defaults)
    
    # Add normal distribution curve
    x = np.linspace(min(std_residuals), max(std_residuals), 100)
    if standardized:
        # For standardized residuals, use standard normal
        ax2.plot(x, stats.norm.pdf(x, 0, 1), 'r-', linewidth=2)
    else:
        # For raw residuals, use estimated mean and std
        mean = np.mean(std_residuals)
        std = np.std(std_residuals)
        ax2.plot(x, stats.norm.pdf(x, mean, std), 'r-', linewidth=2)
    
    # Set labels and title
    if standardized:
        ax2.set_xlabel('Standardized Residuals')
        ax2.set_title('Histogram of Standardized Residuals')
    else:
        ax2.set_xlabel('Residuals')
        ax2.set_title('Histogram of Residuals')
    ax2.set_ylabel('Density')
    
    # Add descriptive statistics if requested
    if add_stats:
        mean = np.mean(std_residuals)
        std = np.std(std_residuals)
        skew = stats.skew(std_residuals)
        kurt = stats.kurtosis(std_residuals, fisher=True)  # Fisher's definition (normal = 0)
        
        stats_text = (
            f"Mean: {mean:.4f}\n"
            f"Std Dev: {std:.4f}\n"
            f"Skewness: {skew:.4f}\n"
            f"Kurtosis: {kurt:.4f}"
        )
        
        # Add normality test if requested
        if test_normality:
            jb_stat, jb_pval = stats.jarque_bera(std_residuals)
            stats_text += f"\nJarque-Bera: {jb_stat:.4f} (p={jb_pval:.4f})"
        
        # Add text box with statistics
        ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add grid
    if grid:
        ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 3: Q-Q plot
    ax3 = plt.subplot(gs[1, 0])
    qq_defaults = {'marker': 'o', 'markersize': 5, 'alpha': 0.7, 'color': color}
    qq_defaults.update(qq_kwargs)
    
    # Create Q-Q plot
    res = stats.probplot(std_residuals, dist="norm", plot=ax3)
    
    # Customize the plot
    ax3.get_lines()[0].set_markerfacecolor(color)
    ax3.get_lines()[0].set_markeredgecolor('none')
    ax3.get_lines()[0].set_alpha(0.7)
    ax3.get_lines()[0].set_markersize(5)
    ax3.get_lines()[1].set_color('red')
    ax3.get_lines()[1].set_linewidth(2)
    
    # Set labels and title
    if standardized:
        ax3.set_title('Q-Q Plot of Standardized Residuals')
    else:
        ax3.set_title('Q-Q Plot of Residuals')
    ax3.set_xlabel('Theoretical Quantiles')
    ax3.set_ylabel('Sample Quantiles')
    
    # Add grid
    if grid:
        ax3.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 4: ACF of residuals
    ax4 = plt.subplot(gs[1, 1])
    
    # Compute ACF
    acf_result = acf(
        std_residuals, nlags=acf_lags, alpha=alpha, 
        adjusted=False, fft=False, demean=True, 
        bartlett_confint=True, method='standard',
        **acf_kwargs
    )
    
    # Extract results
    acf_values = acf_result['acf']
    acf_confint = acf_result['confint']
    lags_values = acf_result['lags']
    
    # Convert to numpy arrays if pandas
    if isinstance(acf_values, pd.Series):
        acf_values = acf_values.values
    if isinstance(acf_confint, pd.DataFrame):
        acf_confint = acf_confint.values
    
    # Plot ACF
    ax4.bar(lags_values, acf_values, width=0.3, color=color, alpha=0.7)
    
    # Add confidence intervals
    ax4.fill_between(
        lags_values,
        acf_confint[:, 0],
        acf_confint[:, 1],
        color=color,
        alpha=0.2
    )
    
    # Add significance lines
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    sig_level = stats.norm.ppf(1 - alpha / 2) / np.sqrt(len(std_residuals))
    ax4.axhline(y=sig_level, color='red', linestyle='--', linewidth=1)
    ax4.axhline(y=-sig_level, color='red', linestyle='--', linewidth=1)
    
    # Set labels and title
    ax4.set_xlabel('Lag')
    ax4.set_ylabel('Autocorrelation')
    ax4.set_title('Autocorrelation Function of Residuals')
    
    # Add autocorrelation test if requested
    if test_autocorrelation and add_stats:
        # Ljung-Box test
        lb_lags = min(10, len(std_residuals) // 5)
        lb_stat, lb_pval = sm.stats.acorr_ljungbox(
            std_residuals, lags=[lb_lags], return_df=False
        )
        
        # Add text box with test results
        lb_text = f"Ljung-Box (lag={lb_lags}):\nQ={lb_stat[0]:.4f} (p={lb_pval[0]:.4f})"
        ax4.text(0.95, 0.95, lb_text, transform=ax4.transAxes,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add grid
    if grid:
        ax4.grid(True, linestyle='--', alpha=0.7)
    
    # Add heteroskedasticity test if requested
    if test_heteroskedasticity and add_stats:
        try:
            from statsmodels.stats.diagnostic import het_arch
            
            # ARCH test
            arch_lags = min(5, len(std_residuals) // 10)
            arch_stat, arch_pval, _, _ = het_arch(std_residuals, nlags=arch_lags)
            
            # Add text box with test results
            arch_text = f"ARCH Test (lag={arch_lags}):\nLM={arch_stat:.4f} (p={arch_pval:.4f})"
            ax1.text(0.95, 0.95, arch_text, transform=ax1.transAxes,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        except ImportError:
            warnings.warn("statsmodels is required for ARCH test. Skipping.")
    
    # Set overall title if provided
    if title:
        fig.suptitle(title, fontsize=14)
    elif model_name:
        if standardized:
            fig.suptitle(f"Residual Diagnostics for {model_name}", fontsize=14)
        else:
            fig.suptitle(f"Residual Diagnostics for {model_name}", fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    if title or model_name:
        plt.subplots_adjust(top=0.9)
    
    return fig


def plot_forecast(
    actual: Union[np.ndarray, pd.Series],
    forecast: Union[np.ndarray, pd.Series],
    lower_bound: Optional[Union[np.ndarray, pd.Series]] = None,
    upper_bound: Optional[Union[np.ndarray, pd.Series]] = None,
    train_actual: Optional[Union[np.ndarray, pd.Series]] = None,
    dates: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None,
    forecast_dates: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None,
    title: Optional[str] = None,
    xlabel: str = "Time",
    ylabel: str = "Value",
    figsize: Tuple[float, float] = (12, 6),
    ax: Optional[plt.Axes] = None,
    grid: bool = True,
    legend: bool = True,
    legend_loc: str = "best",
    ci_alpha: float = 0.3,
    ci_color: str = "lightblue",
    actual_color: str = "steelblue",
    forecast_color: str = "firebrick",
    train_color: str = "darkblue",
    actual_label: str = "Actual",
    forecast_label: str = "Forecast",
    train_label: str = "Training Data",
    ci_label: str = "95% Confidence Interval",
    date_format: Optional[str] = None,
    show_metrics: bool = True,
    metrics: Optional[List[str]] = None,
    vertical_line: bool = True,
    vertical_line_color: str = "black",
    vertical_line_style: str = "--",
    add_error_bars: bool = False
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Visualize forecasts with prediction intervals.
    
    This function creates a plot of actual values and forecasts, optionally with
    prediction intervals and training data. It also computes and displays forecast
    accuracy metrics.
    
    Args:
        actual: Actual values for the forecast period
        forecast: Forecasted values
        lower_bound: Lower bound of prediction interval
        upper_bound: Upper bound of prediction interval
        train_actual: Actual values for the training period
        dates: Dates for the actual values
        forecast_dates: Dates for the forecast values (if different from dates)
        title: Plot title
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        figsize: Figure size as (width, height) in inches
        ax: Existing axes to plot on (if None, creates new figure)
        grid: Whether to show grid lines
        legend: Whether to show legend
        legend_loc: Location of the legend
        ci_alpha: Transparency level for confidence interval
        ci_color: Color for confidence interval
        actual_color: Color for actual values
        forecast_color: Color for forecast values
        train_color: Color for training data
        actual_label: Label for actual values
        forecast_label: Label for forecast values
        train_label: Label for training data
        ci_label: Label for confidence interval
        date_format: Format string for date ticks
        show_metrics: Whether to show forecast accuracy metrics
        metrics: List of metrics to show (default: ['RMSE', 'MAE', 'MAPE'])
        vertical_line: Whether to show vertical line between training and forecast periods
        vertical_line_color: Color for vertical line
        vertical_line_style: Style for vertical line
        add_error_bars: Whether to add error bars to forecast points
    
    Returns:
        Tuple containing the figure and axes objects
    
    Raises:
        ValueError: If input data is invalid
        
    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from mfe.models.time_series.plots import plot_forecast
        >>> # Create sample data
        >>> np.random.seed(42)
        >>> train = np.cumsum(np.random.normal(0, 1, 100))
        >>> actual = np.cumsum(np.random.normal(0, 1, 20)) + train[-1]
        >>> forecast = actual + np.random.normal(0, 0.5, 20)
        >>> lower = forecast - 1.96 * 0.5
        >>> upper = forecast + 1.96 * 0.5
        >>> # Plot forecast
        >>> fig, ax = plot_forecast(actual, forecast, lower, upper, train)
    """
    # Input validation and conversion
    if isinstance(actual, pd.Series):
        actual_array = actual.values
        if dates is None and isinstance(actual.index, pd.DatetimeIndex):
            dates = actual.index
    else:
        actual_array = np.asarray(actual)
    
    if isinstance(forecast, pd.Series):
        forecast_array = forecast.values
        if forecast_dates is None and isinstance(forecast.index, pd.DatetimeIndex):
            forecast_dates = forecast.index
    else:
        forecast_array = np.asarray(forecast)
    
    # Ensure forecast and actual have the same length
    if len(forecast_array) != len(actual_array):
        if len(forecast_array) > len(actual_array):
            forecast_array = forecast_array[:len(actual_array)]
            warnings.warn(f"Forecast length ({len(forecast)}) greater than actual length ({len(actual)}). Truncating forecast.")
        else:
            actual_array = actual_array[:len(forecast_array)]
            warnings.warn(f"Actual length ({len(actual)}) greater than forecast length ({len(forecast)}). Truncating actual.")
    
    # Process confidence intervals if provided
    has_ci = lower_bound is not None and upper_bound is not None
    if has_ci:
        if isinstance(lower_bound, pd.Series):
            lower_array = lower_bound.values
        else:
            lower_array = np.asarray(lower_bound)
        
        if isinstance(upper_bound, pd.Series):
            upper_array = upper_bound.values
        else:
            upper_array = np.asarray(upper_bound)
        
        # Ensure bounds have the same length as forecast
        if len(lower_array) != len(forecast_array):
            lower_array = lower_array[:len(forecast_array)]
        if len(upper_array) != len(forecast_array):
            upper_array = upper_array[:len(forecast_array)]
    
    # Process training data if provided
    has_train = train_actual is not None
    if has_train:
        if isinstance(train_actual, pd.Series):
            train_array = train_actual.values
            if dates is None and isinstance(train_actual.index, pd.DatetimeIndex):
                train_dates = train_actual.index
                if forecast_dates is None:
                    # Try to infer forecast dates from train dates
                    freq = pd.infer_freq(train_dates)
                    if freq is not None:
                        forecast_dates = pd.date_range(
                            start=train_dates[-1] + pd.Timedelta(freq),
                            periods=len(forecast_array),
                            freq=freq
                        )
        else:
            train_array = np.asarray(train_actual)
            train_dates = None
    
    # Set up dates for plotting
    if dates is None and forecast_dates is None:
        # No dates provided, use indices
        if has_train:
            train_indices = np.arange(len(train_array))
            forecast_indices = np.arange(len(train_array), len(train_array) + len(forecast_array))
        else:
            forecast_indices = np.arange(len(forecast_array))
    else:
        # Use provided dates
        if forecast_dates is None:
            # Use dates for both actual and forecast
            if has_train:
                if train_dates is not None:
                    # Use train_dates for training period
                    train_indices = train_dates
                else:
                    # Use first part of dates for training period
                    train_indices = dates[:len(train_array)]
                # Use last part of dates for forecast period
                forecast_indices = dates[-len(forecast_array):]
            else:
                # Use dates for forecast period
                forecast_indices = dates
        else:
            # Use forecast_dates for forecast period
            forecast_indices = forecast_dates
            if has_train:
                if train_dates is not None:
                    # Use train_dates for training period
                    train_indices = train_dates
                elif dates is not None:
                    # Use dates for training period
                    train_indices = dates[:len(train_array)]
                else:
                    # Use indices for training period
                    train_indices = np.arange(len(train_array))
    
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Plot training data if provided
    if has_train:
        ax.plot(train_indices, train_array, color=train_color, label=train_label, linewidth=1.5)
    
    # Plot actual values
    ax.plot(forecast_indices, actual_array, color=actual_color, label=actual_label, linewidth=1.5)
    
    # Plot forecast
    ax.plot(forecast_indices, forecast_array, color=forecast_color, label=forecast_label, linewidth=1.5)
    
    # Add error bars if requested
    if add_error_bars and has_ci:
        ax.errorbar(
            forecast_indices, forecast_array,
            yerr=[forecast_array - lower_array, upper_array - forecast_array],
            fmt='none', ecolor=forecast_color, capsize=3, alpha=0.5
        )
    
    # Add confidence interval if provided
    if has_ci:
        ax.fill_between(
            forecast_indices, lower_array, upper_array,
            color=ci_color, alpha=ci_alpha, label=ci_label
        )
    
    # Add vertical line between training and forecast periods if requested
    if vertical_line and has_train:
        if isinstance(train_indices, (pd.DatetimeIndex, np.ndarray)) and isinstance(train_indices[0], (pd.Timestamp, np.datetime64)):
            # Date-based indices
            if isinstance(forecast_indices, (pd.DatetimeIndex, np.ndarray)) and isinstance(forecast_indices[0], (pd.Timestamp, np.datetime64)):
                ax.axvline(x=train_indices[-1], color=vertical_line_color, linestyle=vertical_line_style)
        else:
            # Numeric indices
            ax.axvline(x=len(train_array) - 0.5, color=vertical_line_color, linestyle=vertical_line_style)
    
    # Set up date formatting if dates are provided
    if isinstance(forecast_indices, pd.DatetimeIndex) or (isinstance(forecast_indices, np.ndarray) and np.issubdtype(forecast_indices.dtype, np.datetime64)):
        # Format date ticks
        if date_format is not None:
            date_formatter = mdates.DateFormatter(date_format)
            ax.xaxis.set_major_formatter(date_formatter)
        else:
            # Auto-select date format based on date range
            locator = mdates.AutoDateLocator()
            formatter = mdates.AutoDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
        
        # Rotate date labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    
    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    
    # Add grid
    if grid:
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    if legend:
        ax.legend(loc=legend_loc)
    
    # Add forecast accuracy metrics if requested
    if show_metrics:
        if metrics is None:
            metrics = ['RMSE', 'MAE', 'MAPE']
        
        # Compute metrics
        metrics_dict = {}
        
        # Root Mean Squared Error
        if 'RMSE' in metrics:
            rmse = np.sqrt(np.mean((actual_array - forecast_array) ** 2))
            metrics_dict['RMSE'] = rmse
        
        # Mean Absolute Error
        if 'MAE' in metrics:
            mae = np.mean(np.abs(actual_array - forecast_array))
            metrics_dict['MAE'] = mae
        
        # Mean Absolute Percentage Error
        if 'MAPE' in metrics and np.all(np.abs(actual_array) > 1e-10):
            mape = np.mean(np.abs((actual_array - forecast_array) / actual_array)) * 100
            metrics_dict['MAPE'] = mape
        
        # Mean Error (Bias)
        if 'ME' in metrics:
            me = np.mean(actual_array - forecast_array)
            metrics_dict['ME'] = me
        
        # Theil's U statistic
        if 'Theil\'s U' in metrics:
            # Compute naive forecast (random walk)
            if has_train:
                naive_forecast = np.full_like(forecast_array, train_array[-1])
            else:
                naive_forecast = np.full_like(forecast_array, actual_array[0])
            
            # Compute Theil's U
            rmse_forecast = np.sqrt(np.mean((actual_array - forecast_array) ** 2))
            rmse_naive = np.sqrt(np.mean((actual_array - naive_forecast) ** 2))
            
            if rmse_naive > 0:
                theil_u = rmse_forecast / rmse_naive
                metrics_dict['Theil\'s U'] = theil_u
        
        # Create metrics text
        metrics_text = "Forecast Metrics:\n"
        for metric, value in metrics_dict.items():
            if metric == 'MAPE':
                metrics_text += f"{metric}: {value:.2f}%\n"
            else:
                metrics_text += f"{metric}: {value:.4f}\n"
        
        # Add text box with metrics
        ax.text(0.02, 0.02, metrics_text, transform=ax.transAxes,
                verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, ax


def plot_model_comparison(
    actual: Union[np.ndarray, pd.Series],
    forecasts: Dict[str, Union[np.ndarray, pd.Series]],
    train_actual: Optional[Union[np.ndarray, pd.Series]] = None,
    dates: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None,
    forecast_dates: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None,
    title: Optional[str] = None,
    xlabel: str = "Time",
    ylabel: str = "Value",
    figsize: Tuple[float, float] = (12, 8),
    grid: bool = True,
    legend: bool = True,
    legend_loc: str = "best",
    actual_color: str = "black",
    train_color: str = "gray",
    forecast_colors: Optional[Dict[str, str]] = None,
    actual_label: str = "Actual",
    train_label: str = "Training Data",
    date_format: Optional[str] = None,
    show_metrics: bool = True,
    metrics: Optional[List[str]] = None,
    vertical_line: bool = True,
    vertical_line_color: str = "black",
    vertical_line_style: str = "--",
    add_table: bool = True,
    plot_diff: bool = False,
    diff_figsize: Optional[Tuple[float, float]] = None
) -> Union[plt.Figure, Tuple[plt.Figure, plt.Figure]]:
    """
    Compare forecasts from multiple models.
    
    This function creates a plot comparing forecasts from multiple models against
    actual values, and optionally computes and displays forecast accuracy metrics.
    
    Args:
        actual: Actual values for the forecast period
        forecasts: Dictionary mapping model names to forecasts
        train_actual: Actual values for the training period
        dates: Dates for the actual values
        forecast_dates: Dates for the forecast values (if different from dates)
        title: Plot title
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        figsize: Figure size as (width, height) in inches
        grid: Whether to show grid lines
        legend: Whether to show legend
        legend_loc: Location of the legend
        actual_color: Color for actual values
        train_color: Color for training data
        forecast_colors: Dictionary mapping model names to colors
        actual_label: Label for actual values
        train_label: Label for training data
        date_format: Format string for date ticks
        show_metrics: Whether to show forecast accuracy metrics
        metrics: List of metrics to show (default: ['RMSE', 'MAE', 'MAPE'])
        vertical_line: Whether to show vertical line between training and forecast periods
        vertical_line_color: Color for vertical line
        vertical_line_style: Style for vertical line
        add_table: Whether to add a table of metrics
        plot_diff: Whether to plot differences between forecasts and actual values
        diff_figsize: Figure size for difference plot
    
    Returns:
        Figure object or tuple of Figure objects if plot_diff is True
    
    Raises:
        ValueError: If input data is invalid
        
    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from mfe.models.time_series.plots import plot_model_comparison
        >>> # Create sample data
        >>> np.random.seed(42)
        >>> train = np.cumsum(np.random.normal(0, 1, 100))
        >>> actual = np.cumsum(np.random.normal(0, 1, 20)) + train[-1]
        >>> forecast1 = actual + np.random.normal(0, 0.5, 20)
        >>> forecast2 = actual + np.random.normal(0, 1.0, 20)
        >>> # Plot comparison
        >>> fig = plot_model_comparison(
        ...     actual, {'Model 1': forecast1, 'Model 2': forecast2}, train
        ... )
    """
    # Input validation and conversion
    if isinstance(actual, pd.Series):
        actual_array = actual.values
        if dates is None and isinstance(actual.index, pd.DatetimeIndex):
            dates = actual.index
    else:
        actual_array = np.asarray(actual)
    
    # Process forecasts
    forecast_arrays = {}
    for model_name, forecast in forecasts.items():
        if isinstance(forecast, pd.Series):
            forecast_arrays[model_name] = forecast.values
            if forecast_dates is None and isinstance(forecast.index, pd.DatetimeIndex):
                forecast_dates = forecast.index
        else:
            forecast_arrays[model_name] = np.asarray(forecast)
    
    # Ensure all forecasts have the same length as actual
    for model_name, forecast_array in forecast_arrays.items():
        if len(forecast_array) != len(actual_array):
            if len(forecast_array) > len(actual_array):
                forecast_arrays[model_name] = forecast_array[:len(actual_array)]
                warnings.warn(f"Forecast length for {model_name} ({len(forecast_array)}) greater than actual length ({len(actual_array)}). Truncating forecast.")
            else:
                actual_array = actual_array[:len(forecast_array)]
                warnings.warn(f"Actual length ({len(actual_array)}) greater than forecast length for {model_name} ({len(forecast_array)}). Truncating actual.")
    
    # Process training data if provided
    has_train = train_actual is not None
    if has_train:
        if isinstance(train_actual, pd.Series):
            train_array = train_actual.values
            if dates is None and isinstance(train_actual.index, pd.DatetimeIndex):
                train_dates = train_actual.index
                if forecast_dates is None:
                    # Try to infer forecast dates from train dates
                    freq = pd.infer_freq(train_dates)
                    if freq is not None:
                        forecast_dates = pd.date_range(
                            start=train_dates[-1] + pd.Timedelta(freq),
                            periods=len(actual_array),
                            freq=freq
                        )
        else:
            train_array = np.asarray(train_actual)
            train_dates = None
    
    # Set up dates for plotting
    if dates is None and forecast_dates is None:
        # No dates provided, use indices
        if has_train:
            train_indices = np.arange(len(train_array))
            forecast_indices = np.arange(len(train_array), len(train_array) + len(actual_array))
        else:
            forecast_indices = np.arange(len(actual_array))
    else:
        # Use provided dates
        if forecast_dates is None:
            # Use dates for both actual and forecast
            if has_train:
                if train_dates is not None:
                    # Use train_dates for training period
                    train_indices = train_dates
                else:
                    # Use first part of dates for training period
                    train_indices = dates[:len(train_array)]
                # Use last part of dates for forecast period
                forecast_indices = dates[-len(actual_array):]
            else:
                # Use dates for forecast period
                forecast_indices = dates
        else:
            # Use forecast_dates for forecast period
            forecast_indices = forecast_dates
            if has_train:
                if train_dates is not None:
                    # Use train_dates for training period
                    train_indices = train_dates
                elif dates is not None:
                    # Use dates for training period
                    train_indices = dates[:len(train_array)]
                else:
                    # Use indices for training period
                    train_indices = np.arange(len(train_array))
    
    # Set up colors for forecasts
    if forecast_colors is None:
        # Use default color cycle
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        forecast_colors = {}
        for i, model_name in enumerate(forecasts.keys()):
            forecast_colors[model_name] = colors[i % len(colors)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot training data if provided
    if has_train:
        ax.plot(train_indices, train_array, color=train_color, label=train_label, linewidth=1.5, alpha=0.7)
    
    # Plot actual values
    ax.plot(forecast_indices, actual_array, color=actual_color, label=actual_label, linewidth=2)
    
    # Plot forecasts
    for model_name, forecast_array in forecast_arrays.items():
        ax.plot(
            forecast_indices, forecast_array,
            color=forecast_colors[model_name], label=model_name,
            linewidth=1.5, alpha=0.8
        )
    
    # Add vertical line between training and forecast periods if requested
    if vertical_line and has_train:
        if isinstance(train_indices, (pd.DatetimeIndex, np.ndarray)) and isinstance(train_indices[0], (pd.Timestamp, np.datetime64)):
            # Date-based indices
            if isinstance(forecast_indices, (pd.DatetimeIndex, np.ndarray)) and isinstance(forecast_indices[0], (pd.Timestamp, np.datetime64)):
                ax.axvline(x=train_indices[-1], color=vertical_line_color, linestyle=vertical_line_style)
        else:
            # Numeric indices
            ax.axvline(x=len(train_array) - 0.5, color=vertical_line_color, linestyle=vertical_line_style)
    
    # Set up date formatting if dates are provided
    if isinstance(forecast_indices, pd.DatetimeIndex) or (isinstance(forecast_indices, np.ndarray) and np.issubdtype(forecast_indices.dtype, np.datetime64)):
        # Format date ticks
        if date_format is not None:
            date_formatter = mdates.DateFormatter(date_format)
            ax.xaxis.set_major_formatter(date_formatter)
        else:
            # Auto-select date format based on date range
            locator = mdates.AutoDateLocator()
            formatter = mdates.AutoDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
        
        # Rotate date labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    
    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Model Forecast Comparison")
    
    # Add grid
    if grid:
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    if legend:
        ax.legend(loc=legend_loc)
    
    # Compute forecast accuracy metrics if requested
    if show_metrics or add_table:
        if metrics is None:
            metrics = ['RMSE', 'MAE', 'MAPE', 'ME']
        
        # Compute metrics for each model
        metrics_dict = {model_name: {} for model_name in forecasts.keys()}
        
        for model_name, forecast_array in forecast_arrays.items():
            # Root Mean Squared Error
            if 'RMSE' in metrics:
                rmse = np.sqrt(np.mean((actual_array - forecast_array) ** 2))
                metrics_dict[model_name]['RMSE'] = rmse
            
            # Mean Absolute Error
            if 'MAE' in metrics:
                mae = np.mean(np.abs(actual_array - forecast_array))
                metrics_dict[model_name]['MAE'] = mae
            
            # Mean Absolute Percentage Error
            if 'MAPE' in metrics and np.all(np.abs(actual_array) > 1e-10):
                mape = np.mean(np.abs((actual_array - forecast_array) / actual_array)) * 100
                metrics_dict[model_name]['MAPE'] = mape
            
            # Mean Error (Bias)
            if 'ME' in metrics:
                me = np.mean(actual_array - forecast_array)
                metrics_dict[model_name]['ME'] = me
            
            # Theil's U statistic
            if 'Theil\'s U' in metrics:
                # Compute naive forecast (random walk)
                if has_train:
                    naive_forecast = np.full_like(forecast_array, train_array[-1])
                else:
                    naive_forecast = np.full_like(forecast_array, actual_array[0])
                
                # Compute Theil's U
                rmse_forecast = np.sqrt(np.mean((actual_array - forecast_array) ** 2))
                rmse_naive = np.sqrt(np.mean((actual_array - naive_forecast) ** 2))
                
                if rmse_naive > 0:
                    theil_u = rmse_forecast / rmse_naive
                    metrics_dict[model_name]['Theil\'s U'] = theil_u
        
        # Add table of metrics if requested
        if add_table:
            # Create table data
            table_data = []
            for metric in metrics:
                if metric in metrics_dict[list(forecasts.keys())[0]]:
                    row = [metric]
                    for model_name in forecasts.keys():
                        if metric == 'MAPE':
                            row.append(f"{metrics_dict[model_name][metric]:.2f}%")
                        else:
                            row.append(f"{metrics_dict[model_name][metric]:.4f}")
                    table_data.append(row)
            
            # Create table
            if table_data:
                # Position the table
                table_ax = plt.axes([0.15, 0.01, 0.7, 0.2], frameon=False)
                table_ax.axis('off')
                
                # Create the table
                table = table_ax.table(
                    cellText=table_data,
                    colLabels=['Metric'] + list(forecasts.keys()),
                    loc='center',
                    cellLoc='center',
                    colColours=['lightgray'] + ['lightblue'] * len(forecasts),
                    colWidths=[0.2] + [0.8 / len(forecasts)] * len(forecasts)
                )
                
                # Style the table
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 1.5)
                
                # Adjust layout to make room for the table
                plt.subplots_adjust(bottom=0.25)
        
        # Add text box with metrics if requested and table not added
        elif show_metrics:
            # Find the best model based on RMSE
            if 'RMSE' in metrics:
                best_model = min(metrics_dict.keys(), key=lambda x: metrics_dict[x]['RMSE'])
                
                # Create metrics text
                metrics_text = f"Best Model: {best_model}\n\n"
                for model_name in forecasts.keys():
                    metrics_text += f"{model_name}:\n"
                    for metric, value in metrics_dict[model_name].items():
                        if metric == 'MAPE':
                            metrics_text += f"  {metric}: {value:.2f}%\n"
                        else:
                            metrics_text += f"  {metric}: {value:.4f}\n"
                    metrics_text += "\n"
                
                # Add text box with metrics
                ax.text(0.02, 0.02, metrics_text, transform=ax.transAxes,
                        verticalalignment='bottom', horizontalalignment='left',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Create difference plot if requested
    if plot_diff:
        if diff_figsize is None:
            diff_figsize = figsize
        
        # Create figure for difference plot
        diff_fig, diff_ax = plt.subplots(figsize=diff_figsize)
        
        # Plot differences
        for model_name, forecast_array in forecast_arrays.items():
            diff = forecast_array - actual_array
            diff_ax.plot(
                forecast_indices, diff,
                color=forecast_colors[model_name], label=model_name,
                linewidth=1.5, alpha=0.8
            )
        
        # Add horizontal line at zero
        diff_ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        # Set up date formatting if dates are provided
        if isinstance(forecast_indices, pd.DatetimeIndex) or (isinstance(forecast_indices, np.ndarray) and np.issubdtype(forecast_indices.dtype, np.datetime64)):
            # Format date ticks
            if date_format is not None:
                date_formatter = mdates.DateFormatter(date_format)
                diff_ax.xaxis.set_major_formatter(date_formatter)
            else:
                # Auto-select date format based on date range
                locator = mdates.AutoDateLocator()
                formatter = mdates.AutoDateFormatter(locator)
                diff_ax.xaxis.set_major_locator(locator)
                diff_ax.xaxis.set_major_formatter(formatter)
            
            # Rotate date labels for better readability
            plt.setp(diff_ax.get_xticklabels(), rotation=30, ha='right')
        
        # Set labels and title
        diff_ax.set_xlabel(xlabel)
        diff_ax.set_ylabel("Forecast - Actual")
        diff_ax.set_title("Forecast Errors")
        
        # Add grid
        if grid:
            diff_ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend
        if legend:
            diff_ax.legend(loc=legend_loc)
        
        # Adjust layout
        diff_fig.tight_layout()
        
        return fig, diff_fig
    
    # Adjust layout
    fig.tight_layout()
    
    return fig


def plot_decomposition(
    result: Union[Dict[str, np.ndarray], Any],
    figsize: Tuple[float, float] = (10, 8),
    dates: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None,
    title: Optional[str] = None,
    xlabel: str = "Time",
    grid: bool = True,
    color: str = "steelblue",
    date_format: Optional[str] = None,
    observed_label: str = "Observed",
    trend_label: str = "Trend",
    seasonal_label: str = "Seasonal",
    residual_label: str = "Residual",
    add_legend: bool = False,
    legend_loc: str = "best",
    sharex: bool = True,
    sharey: bool = False,
    component_titles: bool = True,
    fill_seasonal: bool = True,
    seasonal_color: str = "lightblue",
    seasonal_alpha: float = 0.3
) -> plt.Figure:
    """
    Visualize time series decomposition components.
    
    This function creates a plot of time series decomposition components, including
    the original series, trend, seasonal, and residual components.
    
    Args:
        result: Decomposition result (dictionary or statsmodels DecomposeResult)
        figsize: Figure size as (width, height) in inches
        dates: Dates for the time series
        title: Overall plot title
        xlabel: Label for x-axis
        grid: Whether to show grid lines
        color: Base color for plots
        date_format: Format string for date ticks
        observed_label: Label for observed component
        trend_label: Label for trend component
        seasonal_label: Label for seasonal component
        residual_label: Label for residual component
        add_legend: Whether to add legend to each subplot
        legend_loc: Location of the legend
        sharex: Whether to share x-axis across subplots
        sharey: Whether to share y-axis across subplots
        component_titles: Whether to add titles to component subplots
        fill_seasonal: Whether to fill the area under the seasonal component
        seasonal_color: Color for seasonal component fill
        seasonal_alpha: Transparency for seasonal component fill
    
    Returns:
        Figure object containing the decomposition plots
    
    Raises:
        ValueError: If input data is invalid
        
    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from statsmodels.tsa.seasonal import seasonal_decompose
        >>> from mfe.models.time_series.plots import plot_decomposition
        >>> # Create sample data
        >>> np.random.seed(42)
        >>> dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        >>> trend = np.linspace(0, 10, 100)
        >>> seasonal = 2 * np.sin(np.linspace(0, 8 * np.pi, 100))
        >>> noise = np.random.normal(0, 0.5, 100)
        >>> ts = trend + seasonal + noise
        >>> # Decompose
        >>> result = seasonal_decompose(ts, period=20)
        >>> # Plot decomposition
        >>> fig = plot_decomposition(result, dates=dates)
    """
    # Extract components from result
    if hasattr(result, 'observed') and hasattr(result, 'trend') and hasattr(result, 'seasonal') and hasattr(result, 'resid'):
        # statsmodels DecomposeResult
        observed = result.observed
        trend = result.trend
        seasonal = result.seasonal
        residual = result.resid
    elif isinstance(result, dict) and all(k in result for k in ['observed', 'trend', 'seasonal', 'residual']):
        # Dictionary with components
        observed = result['observed']
        trend = result['trend']
        seasonal = result['seasonal']
        residual = result['residual']
    else:
        raise ValueError("Input must be a statsmodels DecomposeResult or a dictionary with 'observed', 'trend', 'seasonal', and 'residual' keys")
    
    # Convert to numpy arrays if needed
    if isinstance(observed, pd.Series):
        if dates is None and isinstance(observed.index, pd.DatetimeIndex):
            dates = observed.index
        observed = observed.values
    else:
        observed = np.asarray(observed)
    
    if isinstance(trend, pd.Series):
        trend = trend.values
    else:
        trend = np.asarray(trend)
    
    if isinstance(seasonal, pd.Series):
        seasonal = seasonal.values
    else:
        seasonal = np.asarray(seasonal)
    
    if isinstance(residual, pd.Series):
        residual = residual.values
    else:
        residual = np.asarray(residual)
    
    # Create figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=sharex, sharey=sharey)
    
    # Set up x-axis values
    if dates is not None:
        x = dates
    else:
        x = np.arange(len(observed))
    
    # Plot observed component
    axes[0].plot(x, observed, color=color, linewidth=1.5)
    if component_titles:
        axes[0].set_title(observed_label)
    if add_legend:
        axes[0].legend([observed_label], loc=legend_loc)
    
    # Plot trend component
    axes[1].plot(x, trend, color=color, linewidth=1.5)
    if component_titles:
        axes[1].set_title(trend_label)
    if add_legend:
        axes[1].legend([trend_label], loc=legend_loc)
    
    # Plot seasonal component
    axes[2].plot(x, seasonal, color=color, linewidth=1.5)
    if fill_seasonal:
        axes[2].fill_between(x, 0, seasonal, color=seasonal_color, alpha=seasonal_alpha)
    if component_titles:
        axes[2].set_title(seasonal_label)
    if add_legend:
        axes[2].legend([seasonal_label], loc=legend_loc)
    
    # Plot residual component
    axes[3].plot(x, residual, color=color, linewidth=1.5)
    axes[3].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    if component_titles:
        axes[3].set_title(residual_label)
    if add_legend:
        axes[3].legend([residual_label], loc=legend_loc)
    
    # Set up date formatting if dates are provided
    if dates is not None:
        if isinstance(dates, pd.DatetimeIndex) or np.issubdtype(dates.dtype, np.datetime64):
            # Format date ticks
            if date_format is not None:
                date_formatter = mdates.DateFormatter(date_format)
                axes[3].xaxis.set_major_formatter(date_formatter)
            else:
                # Auto-select date format based on date range
                locator = mdates.AutoDateLocator()
                formatter = mdates.AutoDateFormatter(locator)
                axes[3].xaxis.set_major_locator(locator)
                axes[3].xaxis.set_major_formatter(formatter)
            
            # Rotate date labels for better readability
            plt.setp(axes[3].get_xticklabels(), rotation=30, ha='right')
    
    # Set x-label for bottom subplot
    axes[3].set_xlabel(xlabel)
    
    # Add grid to all subplots
    if grid:
        for ax in axes:
            ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set overall title if provided
    if title:
        fig.suptitle(title, fontsize=14)
        plt.subplots_adjust(top=0.9)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def plot_impulse_response(
    irf_result: Union[Dict[str, np.ndarray], Any],
    responses: Optional[List[str]] = None,
    impulses: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (12, 8),
    periods: Optional[int] = None,
    title: Optional[str] = None,
    xlabel: str = "Periods",
    ylabel: str = "Response",
    grid: bool = True,
    color: str = "steelblue",
    ci_color: str = "lightblue",
    ci_alpha: float = 0.3,
    line_style: str = "-",
    marker: Optional[str] = None,
    legend: bool = True,
    legend_loc: str = "best",
    subplot_titles: bool = True,
    zero_line: bool = True,
    cumulative: bool = False,
    orthogonalized: bool = True,
    plot_ci: bool = True,
    ci_level: float = 0.05
) -> plt.Figure:
    """
    Plot impulse response functions.
    
    This function creates a plot of impulse response functions, showing how variables
    respond to shocks in other variables over time.
    
    Args:
        irf_result: Impulse response function result (dictionary or statsmodels IRF result)
        responses: List of response variables to plot (if None, plot all)
        impulses: List of impulse variables to plot (if None, plot all)
        figsize: Figure size as (width, height) in inches
        periods: Number of periods to plot (if None, use all available)
        title: Overall plot title
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        grid: Whether to show grid lines
        color: Base color for plots
        ci_color: Color for confidence intervals
        ci_alpha: Transparency for confidence intervals
        line_style: Line style for impulse response
        marker: Marker style for impulse response
        legend: Whether to show legend
        legend_loc: Location of the legend
        subplot_titles: Whether to add titles to subplots
        zero_line: Whether to show horizontal line at zero
        cumulative: Whether to plot cumulative responses
        orthogonalized: Whether to use orthogonalized responses
        plot_ci: Whether to plot confidence intervals
        ci_level: Confidence level for intervals (e.g., 0.05 for 95% CI)
    
    Returns:
        Figure object containing the impulse response plots
    
    Raises:
        ValueError: If input data is invalid
        
    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from statsmodels.tsa.api import VAR
        >>> from mfe.models.time_series.plots import plot_impulse_response
        >>> # Create sample data
        >>> np.random.seed(42)
        >>> dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        >>> y1 = np.cumsum(np.random.normal(0, 1, 100))
        >>> y2 = 0.5 * y1 + np.cumsum(np.random.normal(0, 1, 100))
        >>> data = pd.DataFrame({'y1': y1, 'y2': y2}, index=dates)
        >>> # Fit VAR model
        >>> model = VAR(data)
        >>> results = model.fit(2)
        >>> # Get impulse response functions
        >>> irf = results.irf(10)
        >>> # Plot impulse responses
        >>> fig = plot_impulse_response(irf)
    """
    # Extract IRF data from result
    if hasattr(irf_result, 'irfs') and hasattr(irf_result, 'stderr'):
        # statsmodels IRF result
        if orthogonalized and hasattr(irf_result, 'orth_irfs'):
            irfs = irf_result.orth_irfs
        else:
            irfs = irf_result.irfs
        
        if cumulative and hasattr(irf_result, 'cum_effects'):
            if orthogonalized and hasattr(irf_result, 'orth_cum_effects'):
                irfs = irf_result.orth_cum_effects
            else:
                irfs = irf_result.cum_effects
        
        # Get variable names
        if hasattr(irf_result, 'names'):
            var_names = irf_result.names
        else:
            var_names = [f"Var{i+1}" for i in range(irfs.shape[1])]
        
        # Get confidence intervals if available
        has_ci = hasattr(irf_result, 'stderr')
        if has_ci and plot_ci:
            stderr = irf_result.stderr
            if orthogonalized and hasattr(irf_result, 'orth_stderr'):
                stderr = irf_result.orth_stderr
            
            if cumulative and hasattr(irf_result, 'cum_stderr'):
                if orthogonalized and hasattr(irf_result, 'orth_cum_stderr'):
                    stderr = irf_result.orth_cum_stderr
                else:
                    stderr = irf_result.cum_stderr
            
            # Compute confidence intervals
            z_value = stats.norm.ppf(1 - ci_level / 2)
            lower = irfs - z_value * stderr
            upper = irfs + z_value * stderr
        else:
            has_ci = False
    elif isinstance(irf_result, dict):
        # Dictionary with IRF data
        if 'irfs' in irf_result:
            irfs = irf_result['irfs']
            if cumulative and 'cum_irfs' in irf_result:
                irfs = irf_result['cum_irfs']
            elif orthogonalized and 'orth_irfs' in irf_result:
                irfs = irf_result['orth_irfs']
                if cumulative and 'orth_cum_irfs' in irf_result:
                    irfs = irf_result['orth_cum_irfs']
        else:
            raise ValueError("Dictionary must contain 'irfs' key")
        
        # Get variable names
        if 'names' in irf_result:
            var_names = irf_result['names']
        else:
            var_names = [f"Var{i+1}" for i in range(irfs.shape[1])]
        
        # Get confidence intervals if available
        has_ci = 'lower' in irf_result and 'upper' in irf_result and plot_ci
        if has_ci:
            lower = irf_result['lower']
            upper = irf_result['upper']
            
            if cumulative and 'cum_lower' in irf_result and 'cum_upper' in irf_result:
                lower = irf_result['cum_lower']
                upper = irf_result['cum_upper']
            elif orthogonalized and 'orth_lower' in irf_result and 'orth_upper' in irf_result:
                lower = irf_result['orth_lower']
                upper = irf_result['orth_upper']
                if cumulative and 'orth_cum_lower' in irf_result and 'orth_cum_upper' in irf_result:
                    lower = irf_result['orth_cum_lower']
                    upper = irf_result['orth_cum_upper']
    else:
        raise ValueError("Input must be a statsmodels IRF result or a dictionary with IRF data")
    
    # Determine number of periods to plot
    if periods is None:
        periods = irfs.shape[0]
    else:
        periods = min(periods, irfs.shape[0])
    
    # Determine which responses and impulses to plot
    if responses is None:
        responses = var_names
    if impulses is None:
        impulses = var_names
    
    # Validate responses and impulses
    for name in responses:
        if name not in var_names:
            raise ValueError(f"Response variable '{name}' not found in variable names: {var_names}")
    for name in impulses:
        if name not in var_names:
            raise ValueError(f"Impulse variable '{name}' not found in variable names: {var_names}")
    
    # Get indices for responses and impulses
    response_indices = [var_names.index(name) for name in responses]
    impulse_indices = [var_names.index(name) for name in impulses]
    
    # Create figure with subplots
    n_responses = len(responses)
    n_impulses = len(impulses)
    fig, axes = plt.subplots(n_responses, n_impulses, figsize=figsize, sharex=True)
    
    # Handle case with only one subplot
    if n_responses == 1 and n_impulses == 1:
        axes = np.array([[axes]])
    elif n_responses == 1:
        axes = axes.reshape(1, -1)
    elif n_impulses == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot impulse responses
    for i, resp_idx in enumerate(response_indices):
        for j, imp_idx in enumerate(impulse_indices):
            ax = axes[i, j]
            
            # Plot impulse response
            ax.plot(
                range(periods),
                irfs[:periods, resp_idx, imp_idx],
                color=color,
                linestyle=line_style,
                marker=marker,
                linewidth=1.5,
                label='Impulse Response'
            )
            
            # Add confidence intervals if available
            if has_ci:
                ax.fill_between(
                    range(periods),
                    lower[:periods, resp_idx, imp_idx],
                    upper[:periods, resp_idx, imp_idx],
                    color=ci_color,
                    alpha=ci_alpha,
                    label=f"{int((1-ci_level)*100)}% CI"
                )
            
            # Add horizontal line at zero
            if zero_line:
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            
            # Add grid
            if grid:
                ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add subplot title
            if subplot_titles:
                if cumulative:
                    ax.set_title(f"{impulses[j]} → {responses[i]} (Cumulative)")
                else:
                    ax.set_title(f"{impulses[j]} → {responses[i]}")
            
            # Add legend to first subplot only
            if legend and i == 0 and j == 0:
                ax.legend(loc=legend_loc)
            
            # Add x-label to bottom row
            if i == n_responses - 1:
                ax.set_xlabel(xlabel)
            
            # Add y-label to leftmost column
            if j == 0:
                ax.set_ylabel(ylabel)
    
    # Set overall title if provided
    if title:
        fig.suptitle(title, fontsize=14)
        plt.subplots_adjust(top=0.9)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def plot_qq(
    data: Union[np.ndarray, pd.Series],
    dist: str = "norm",
    figsize: Tuple[float, float] = (8, 6),
    title: Optional[str] = None,
    xlabel: str = "Theoretical Quantiles",
    ylabel: str = "Sample Quantiles",
    grid: bool = True,
    color: str = "steelblue",
    line_color: str = "red",
    marker: str = "o",
    alpha: float = 0.7,
    markersize: int = 5,
    add_stats: bool = True,
    ax: Optional[plt.Axes] = None,
    dist_kwargs: Optional[Dict[str, Any]] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a Q-Q plot for residual normality assessment.
    
    This function creates a Quantile-Quantile (Q-Q) plot to assess whether a data
    sample follows a specified theoretical distribution (default: normal).
    
    Args:
        data: Data to plot
        dist: Distribution to compare against ('norm', 't', etc.)
        figsize: Figure size as (width, height) in inches
        title: Plot title
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        grid: Whether to show grid lines
        color: Color for data points
        line_color: Color for reference line
        marker: Marker style for data points
        alpha: Transparency level for data points
        markersize: Size of markers
        add_stats: Whether to add descriptive statistics
        ax: Existing axes to plot on (if None, creates new figure)
        dist_kwargs: Additional keyword arguments for distribution
    
    Returns:
        Tuple containing the figure and axes objects
    
    Raises:
        ValueError: If input data is invalid
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.time_series.plots import plot_qq
        >>> # Create sample data
        >>> np.random.seed(42)
        >>> data = np.random.normal(0, 1, 100)
        >>> # Create Q-Q plot
        >>> fig, ax = plot_qq(data, title="Normal Q-Q Plot")
    """
    # Set default values for dist_kwargs
    if dist_kwargs is None:
        dist_kwargs = {}
    
    # Convert to numpy array if needed
    if isinstance(data, pd.Series):
        data_array = data.values
    else:
        data_array = np.asarray(data)
    
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Create Q-Q plot
    res = stats.probplot(data_array, dist=dist, plot=ax, fit=True, **dist_kwargs)
    
    # Customize the plot
    ax.get_lines()[0].set_marker(marker)
    ax.get_lines()[0].set_markerfacecolor(color)
    ax.get_lines()[0].set_markeredgecolor('none')
    ax.get_lines()[0].set_alpha(alpha)
    ax.get_lines()[0].set_markersize(markersize)
    ax.get_lines()[1].set_color(line_color)
    ax.get_lines()[1].set_linewidth(2)
    
    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    else:
        if dist == "norm":
            ax.set_title("Normal Q-Q Plot")
        else:
            ax.set_title(f"{dist.capitalize()} Q-Q Plot")
    
    # Add grid
    if grid:
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add descriptive statistics if requested
    if add_stats:
        mean = np.mean(data_array)
        std = np.std(data_array)
        skew = stats.skew(data_array)
        kurt = stats.kurtosis(data_array, fisher=True)  # Fisher's definition (normal = 0)
        
        stats_text = (
            f"Mean: {mean:.4f}\n"
            f"Std Dev: {std:.4f}\n"
            f"Skewness: {skew:.4f}\n"
            f"Kurtosis: {kurt:.4f}"
        )
        
        # Add normality test
        jb_stat, jb_pval = stats.jarque_bera(data_array)
        stats_text += f"\nJarque-Bera: {jb_stat:.4f} (p={jb_pval:.4f})"
        
        # Add text box with statistics
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, ax


def plot_rolling_statistics(
    data: Union[np.ndarray, pd.Series, pd.DataFrame],
    window: int = 20,
    figsize: Tuple[float, float] = (10, 8),
    dates: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None,
    title: Optional[str] = None,
    xlabel: str = "Time",
    grid: bool = True,
    color: str = "steelblue",
    mean_color: str = "firebrick",
    std_color: str = "forestgreen",
    date_format: Optional[str] = None,
    add_legend: bool = True,
    legend_loc: str = "best",
    plot_original: bool = True,
    plot_mean: bool = True,
    plot_std: bool = True,
    plot_bounds: bool = True,
    bounds_alpha: float = 0.2,
    bounds_color: str = "lightblue",
    n_std: float = 2.0,
    original_label: str = "Original Series",
    mean_label: str = "Rolling Mean",
    std_label: str = "Rolling Std",
    bounds_label: str = "±2σ Bounds",
    add_adf_test: bool = True,
    add_kpss_test: bool = True
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot rolling mean and standard deviation.
    
    This function creates a plot of a time series with its rolling mean and standard
    deviation, which is useful for assessing stationarity and volatility clustering.
    
    Args:
        data: Time series data
        window: Window size for rolling statistics
        figsize: Figure size as (width, height) in inches
        dates: Dates for the time series
        title: Plot title
        xlabel: Label for x-axis
        grid: Whether to show grid lines
        color: Color for original series
        mean_color: Color for rolling mean
        std_color: Color for rolling standard deviation
        date_format: Format string for date ticks
        add_legend: Whether to add legend
        legend_loc: Location of the legend
        plot_original: Whether to plot original series
        plot_mean: Whether to plot rolling mean
        plot_std: Whether to plot rolling standard deviation
        plot_bounds: Whether to plot bounds (mean ± n_std * std)
        bounds_alpha: Transparency level for bounds
        bounds_color: Color for bounds
        n_std: Number of standard deviations for bounds
        original_label: Label for original series
        mean_label: Label for rolling mean
        std_label: Label for rolling standard deviation
        bounds_label: Label for bounds
        add_adf_test: Whether to add ADF test results
        add_kpss_test: Whether to add KPSS test results
    
    Returns:
        Tuple containing the figure and axes objects
    
    Raises:
        ValueError: If input data is invalid
        
    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from mfe.models.time_series.plots import plot_rolling_statistics
        >>> # Create sample data
        >>> np.random.seed(42)
        >>> dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        >>> data = pd.Series(np.cumsum(np.random.normal(0, 1, 100)), index=dates)
        >>> # Plot rolling statistics
        >>> fig, ax = plot_rolling_statistics(data, window=20)
    """
    # Input validation and conversion
    if isinstance(data, pd.DataFrame):
        if data.shape[1] > 1:
            warnings.warn("Multiple columns detected in DataFrame. Using the first column.")
        series = data.iloc[:, 0]
        if dates is None and isinstance(series.index, pd.DatetimeIndex):
            dates = series.index
    elif isinstance(data, pd.Series):
        series = data
        if dates is None and isinstance(series.index, pd.DatetimeIndex):
            dates = series.index
    else:
        # Convert numpy array to Series
        data_array = np.asarray(data)
        if data_array.ndim > 1:
            if data_array.shape[1] > 1:
                warnings.warn("Multiple columns detected in array. Using the first column.")
            data_array = data_array[:, 0]
        series = pd.Series(data_array)
    
    # Compute rolling statistics
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up x-axis values
    if dates is not None:
        x = dates
    else:
        x = np.arange(len(series))
    
    # Plot original series
    if plot_original:
        ax.plot(x, series, color=color, label=original_label, linewidth=1.5)
    
    # Plot rolling mean
    if plot_mean:
        ax.plot(x, rolling_mean, color=mean_color, label=mean_label, linewidth=2)
    
    # Plot rolling standard deviation
    if plot_std:
        ax2 = ax.twinx()
        ax2.plot(x, rolling_std, color=std_color, label=std_label, linewidth=2, linestyle='--')
        ax2.set_ylabel('Standard Deviation', color=std_color)
        ax2.tick_params(axis='y', labelcolor=std_color)
    
    # Plot bounds (mean ± n_std * std)
    if plot_bounds and plot_mean:
        upper_bound = rolling_mean + n_std * rolling_std
        lower_bound = rolling_mean - n_std * rolling_std
        ax.fill_between(x, lower_bound, upper_bound, color=bounds_color, alpha=bounds_alpha, label=bounds_label)
    
    # Set up date formatting if dates are provided
    if dates is not None:
        if isinstance(dates, pd.DatetimeIndex) or np.issubdtype(dates.dtype, np.datetime64):
            # Format date ticks
            if date_format is not None:
                date_formatter = mdates.DateFormatter(date_format)
                ax.xaxis.set_major_formatter(date_formatter)
            else:
                # Auto-select date format based on date range
                locator = mdates.AutoDateLocator()
                formatter = mdates.AutoDateFormatter(locator)
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(formatter)
            
            # Rotate date labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    
    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Value')
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Rolling Statistics (Window = {window})')
    
    # Add grid
    if grid:
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    if add_legend:
        # Combine legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        if plot_std:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc=legend_loc)
        else:
            ax.legend(loc=legend_loc)
    
    # Add stationarity tests if requested
    if add_adf_test or add_kpss_test:
        test_text = "Stationarity Tests:\n"
        
        # ADF test
        if add_adf_test:
            try:
                from statsmodels.tsa.stattools import adfuller
                adf_result = adfuller(series.dropna())
                test_text += f"ADF Test:\n"
                test_text += f"  Statistic: {adf_result[0]:.4f}\n"
                test_text += f"  p-value: {adf_result[1]:.4f}\n"
                test_text += f"  {'Stationary' if adf_result[1] < 0.05 else 'Non-stationary'} at 5% significance\n\n"
            except ImportError:
                test_text += "ADF Test: statsmodels required\n\n"
        
        # KPSS test
        if add_kpss_test:
            try:
                from statsmodels.tsa.stattools import kpss
                kpss_result = kpss(series.dropna())
                test_text += f"KPSS Test:\n"
                test_text += f"  Statistic: {kpss_result[0]:.4f}\n"
                test_text += f"  p-value: {kpss_result[1]:.4f}\n"
                test_text += f"  {'Non-stationary' if kpss_result[1] < 0.05 else 'Stationary'} at 5% significance"
            except ImportError:
                test_text += "KPSS Test: statsmodels required"
        
        # Add text box with test results
        ax.text(0.02, 0.02, test_text, transform=ax.transAxes,
                verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, ax


def plot_seasonal(
    data: Union[np.ndarray, pd.Series, pd.DataFrame],
    period: int,
    figsize: Tuple[float, float] = (12, 8),
    dates: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: str = "Value",
    grid: bool = True,
    color: str = "steelblue",
    alpha: float = 0.7,
    marker: str = "o",
    line_style: str = "-",
    add_legend: bool = True,
    legend_loc: str = "best",
    plot_type: str = "seasonal",
    seasonal_labels: Optional[List[str]] = None,
    add_mean: bool = True,
    mean_color: str = "red",
    mean_line_style: str = "--",
    mean_line_width: float = 2.0,
    mean_label: str = "Mean",
    add_boxplot: bool = False,
    boxplot_figsize: Optional[Tuple[float, float]] = None
) -> Union[plt.Figure, Tuple[plt.Figure, plt.Figure]]:
    """
    Visualize seasonal patterns in time series.
    
    This function creates a plot to visualize seasonal patterns in a time series,
    either by season, by cycle, or as a subseries plot.
    
    Args:
        data: Time series data
        period: Seasonal period (e.g., 12 for monthly, 4 for quarterly)
        figsize: Figure size as (width, height) in inches
        dates: Dates for the time series
        title: Plot title
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        grid: Whether to show grid lines
        color: Base color for plots
        alpha: Transparency level for plots
        marker: Marker style for plots
        line_style: Line style for plots
        add_legend: Whether to add legend
        legend_loc: Location of the legend
        plot_type: Type of plot ('seasonal', 'cycle', or 'subseries')
        seasonal_labels: Labels for seasons (e.g., month names)
        add_mean: Whether to add mean line to seasonal plot
        mean_color: Color for mean line
        mean_line_style: Line style for mean line
        mean_line_width: Line width for mean line
        mean_label: Label for mean line
        add_boxplot: Whether to add boxplot of seasonal values
        boxplot_figsize: Figure size for boxplot
    
    Returns:
        Figure object or tuple of Figure objects if add_boxplot is True
    
    Raises:
        ValueError: If input data is invalid
        
    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from mfe.models.time_series.plots import plot_seasonal
        >>> # Create sample data
        >>> np.random.seed(42)
        >>> dates = pd.date_range(start='2015-01-01', periods=48, freq='M')
        >>> base = np.sin(np.linspace(0, 4*np.pi, 12))
        >>> data = np.tile(base, 4) + np.random.normal(0, 0.2, 48)
        >>> ts = pd.Series(data, index=dates)
        >>> # Plot seasonal pattern
        >>> fig = plot_seasonal(ts, period=12, seasonal_labels=[
        ...     'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
        ...     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
        ... ])
    """
    # Input validation and conversion
    if isinstance(data, pd.DataFrame):
        if data.shape[1] > 1:
            warnings.warn("Multiple columns detected in DataFrame. Using the first column.")
        series = data.iloc[:, 0]
        if dates is None and isinstance(series.index, pd.DatetimeIndex):
            dates = series.index
    elif isinstance(data, pd.Series):
        series = data
        if dates is None and isinstance(series.index, pd.DatetimeIndex):
            dates = series.index
    else:
        # Convert numpy array to Series
        data_array = np.asarray(data)
        if data_array.ndim > 1:
            if data_array.shape[1] > 1:
                warnings.warn("Multiple columns detected in array. Using the first column.")
            data_array = data_array[:, 0]
        series = pd.Series(data_array)
    
    # Validate period
    if period <= 0:
        raise ValueError("Period must be positive")
    if period > len(series) // 2:
        warnings.warn(f"Period ({period}) is large relative to series length ({len(series)}). Results may not be meaningful.")
    
    # Create seasonal labels if not provided
    if seasonal_labels is None:
        seasonal_labels = [f"Season {i+1}" for i in range(period)]
    elif len(seasonal_labels) != period:
        warnings.warn(f"Length of seasonal_labels ({len(seasonal_labels)}) does not match period ({period}). Using default labels.")
        seasonal_labels = [f"Season {i+1}" for i in range(period)]
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Create different plot types
    if plot_type == 'seasonal':
        # Seasonal plot: values grouped by season
        # Reshape data into seasons
        n_cycles = len(series) // period
        remainder = len(series) % period
        
        if remainder > 0:
            # Pad with NaN to make complete cycles
            padded_series = np.concatenate([series.values, np.full(period - remainder, np.nan)])
            n_cycles += 1
        else:
            padded_series = series.values
        
        # Reshape into seasons
        seasonal_data = padded_series.reshape(n_cycles, period)
        
        # Create plot
        ax = fig.add_subplot(111)
        
        # Plot each cycle
        for i in range(n_cycles):
            cycle_data = seasonal_data[i, :]
            if not np.isnan(cycle_data).all():  # Skip if all NaN
                ax.plot(
                    range(period),
                    cycle_data,
                    marker=marker,
                    linestyle=line_style,
                    alpha=alpha,
                    color=color,
                    label=f"Cycle {i+1}" if add_legend else None
                )
        
        # Add mean line if requested
        if add_mean:
            seasonal_mean = np.nanmean(seasonal_data, axis=0)
            ax.plot(
                range(period),
                seasonal_mean,
                marker='o',
                linestyle=mean_line_style,
                linewidth=mean_line_width,
                color=mean_color,
                label=mean_label
            )
        
        # Set x-ticks to seasonal labels
        ax.set_xticks(range(period))
        ax.set_xticklabels(seasonal_labels)
        
        # Set labels and title
        if xlabel is None:
            ax.set_xlabel("Season")
        else:
            ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        else:
            ax.set_title("Seasonal Plot")
        
        # Add grid
        if grid:
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend if requested and there are multiple cycles
        if add_legend and n_cycles > 1:
            if n_cycles > 10:  # Too many cycles for individual legend entries
                # Create custom legend with just the mean
                if add_mean:
                    ax.legend([plt.Line2D([0], [0], color=mean_color, linestyle=mean_line_style, linewidth=mean_line_width)],
                              [mean_label], loc=legend_loc)
            else:
                ax.legend(loc=legend_loc)
        elif add_legend and add_mean:
            ax.legend([plt.Line2D([0], [0], color=mean_color, linestyle=mean_line_style, linewidth=mean_line_width)],
                      [mean_label], loc=legend_loc)
    elif plot_type == 'cycle':
        # Cycle plot: values grouped by cycle
        # Reshape data into cycles
        n_cycles = len(series) // period
        remainder = len(series) % period
        
        if remainder > 0:
            # Pad with NaN to make complete cycles
            padded_series = np.concatenate([series.values, np.full(period - remainder, np.nan)])
            n_cycles += 1
        else:
            padded_series = series.values
        
        # Reshape into cycles
        cycle_data = padded_series.reshape(n_cycles, period)
        
        # Create plot
        ax = fig.add_subplot(111)
        
        # Plot each season
        for i in range(period):
            season_data = cycle_data[:, i]
            if not np.isnan(season_data).all():  # Skip if all NaN
                ax.plot(
                    range(n_cycles),
                    season_data,
                    marker=marker,
                    linestyle=line_style,
                    alpha=alpha,
                    label=seasonal_labels[i]
                )
        
        # Set x-ticks to cycle numbers
        ax.set_xticks(range(n_cycles))
        ax.set_xticklabels([f"Cycle {i+1}" for i in range(n_cycles)])
        
        # Set labels and title
        if xlabel is None:
            ax.set_xlabel("Cycle")
        else:
            ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        else:
            ax.set_title("Cycle Plot")
        
        # Add grid
        if grid:
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend if requested
        if add_legend:
            ax.legend(loc=legend_loc)
    elif plot_type == 'subseries':
        # Subseries plot: one subplot for each season
        # Calculate number of rows and columns for subplots
        n_cols = min(4, period)
        n_rows = (period + n_cols - 1) // n_cols  # Ceiling division
        
        # Create subplots
        axes = fig.subplots(n_rows, n_cols)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Extract seasonal components
        seasonal_values = [series.iloc[i::period] for i in range(period)]
        
        # Plot each season
        for i in range(period):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # Get seasonal data
            season_data = seasonal_values[i]
            
            # Plot seasonal data
            if isinstance(season_data.index, pd.DatetimeIndex) and dates is not None:
                ax.plot(
                    season_data.index,
                    season_data.values,
                    marker=marker,
                    linestyle=line_style,
                    alpha=alpha,
                    color=color
                )
                
                # Format date ticks
                locator = mdates.YearLocator()
                formatter = mdates.DateFormatter('%Y')
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(formatter)
                plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
            else:
                ax.plot(
                    range(len(season_data)),
                    season_data.values,
                    marker=marker,
                    linestyle=line_style,
                    alpha=alpha,
                    color=color
                )
            
            # Add mean line if requested
            if add_mean:
                mean_value = np.mean(season_data.values)
                ax.axhline(
                    y=mean_value,
                    linestyle=mean_line_style,
                    linewidth=mean_line_width,
                    color=mean_color
                )
            
            # Set title for subplot
            ax.set_title(seasonal_labels[i])
            
            # Add grid
            if grid:
                ax.grid(True, linestyle='--', alpha=0.7)
            
            # Set y-label for leftmost subplots
            if col == 0:
                ax.set_ylabel(ylabel)
        
        # Hide empty subplots
        for i in range(period, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)

