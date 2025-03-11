'''
Utility functions for bootstrap methods in the MFE Toolbox.

This module provides a comprehensive set of utility functions to support bootstrap
implementations, result analysis, and visualization. These utilities enhance the
usability of bootstrap methods by providing convenient tools for confidence interval
calculation, p-value computation, result formatting, and visualization.

The implementation leverages NumPy's efficient array operations with performance-critical
sections accelerated using Numba's @jit decorators. This approach provides significant
performance improvements while maintaining the flexibility and readability of Python code.

The module includes functions for:
- Confidence interval calculation with various methods
- Bootstrap p-value computation
- Result formatting and analysis
- Visualization of bootstrap distributions and results
- Input validation and error handling
- Helper functions for bootstrap implementation
'''

from typing import (
    Any, Callable, Dict, List, Literal, Optional, Tuple, Union, cast, overload
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit
import logging
from dataclasses import dataclass, field

from mfe.core.exceptions import (
    BootstrapError, ParameterError, DimensionError, DataError,
    raise_parameter_error, raise_dimension_error, raise_data_error
)
from mfe.core.types import (
    BootstrapIndices, BootstrapSamples, BootstrapStatistic, 
    BootstrapResult, ProgressCallback
)

# Set up module-level logger
logger = logging.getLogger("mfe.models.bootstrap.utils")


@jit(nopython=True, cache=True)
def _compute_percentile_ci(
    bootstrap_statistics: np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    Compute percentile confidence intervals using Numba acceleration.
    
    This function computes confidence intervals using the percentile method,
    which simply takes the empirical quantiles of the bootstrap distribution.
    It is accelerated using Numba's @jit decorator for improved performance.
    
    Args:
        bootstrap_statistics: Bootstrap statistics array
        alpha: Significance level (e.g., 0.05 for 95% confidence)
        
    Returns:
        np.ndarray: Confidence intervals [lower, upper]
    """
    # Calculate percentiles
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    # Handle different dimensions of bootstrap_statistics
    if bootstrap_statistics.ndim == 1:
        # For 1D statistics (scalar statistic)
        lower = np.percentile(bootstrap_statistics, lower_percentile)
        upper = np.percentile(bootstrap_statistics, upper_percentile)
        return np.array([lower, upper])
    else:
        # For multi-dimensional statistics
        n_stats = bootstrap_statistics.shape[1]
        ci = np.zeros((2, n_stats))
        
        for i in range(n_stats):
            ci[0, i] = np.percentile(bootstrap_statistics[:, i], lower_percentile)
            ci[1, i] = np.percentile(bootstrap_statistics[:, i], upper_percentile)
        
        return ci


@jit(nopython=True, cache=True)
def _compute_basic_ci(
    bootstrap_statistics: np.ndarray,
    original_statistic: np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    Compute basic bootstrap confidence intervals using Numba acceleration.
    
    This function computes confidence intervals using the basic bootstrap method,
    which accounts for bias in the bootstrap distribution.
    It is accelerated using Numba's @jit decorator for improved performance.
    
    Args:
        bootstrap_statistics: Bootstrap statistics array
        original_statistic: Original statistic computed on the data
        alpha: Significance level (e.g., 0.05 for 95% confidence)
        
    Returns:
        np.ndarray: Confidence intervals [lower, upper]
    """
    # Calculate percentiles
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    # Handle different dimensions
    if bootstrap_statistics.ndim == 1:
        # For 1D statistics (scalar statistic)
        # Basic CI: 2*original - upper percentile, 2*original - lower percentile
        lower_quantile = np.percentile(bootstrap_statistics, upper_percentile)
        upper_quantile = np.percentile(bootstrap_statistics, lower_percentile)
        
        lower = 2 * original_statistic - lower_quantile
        upper = 2 * original_statistic - upper_quantile
        
        return np.array([lower, upper])
    else:
        # For multi-dimensional statistics
        n_stats = bootstrap_statistics.shape[1]
        ci = np.zeros((2, n_stats))
        
        for i in range(n_stats):
            lower_quantile = np.percentile(bootstrap_statistics[:, i], upper_percentile)
            upper_quantile = np.percentile(bootstrap_statistics[:, i], lower_percentile)
            
            ci[0, i] = 2 * original_statistic[i] - lower_quantile
            ci[1, i] = 2 * original_statistic[i] - upper_quantile
        
        return ci


@jit(nopython=True, cache=True)
def _compute_studentized_ci(
    bootstrap_statistics: np.ndarray,
    bootstrap_standard_errors: np.ndarray,
    original_statistic: np.ndarray,
    original_standard_error: np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    Compute studentized confidence intervals using Numba acceleration.
    
    This function computes confidence intervals using the studentized bootstrap method,
    which accounts for the variability in the bootstrap distribution.
    It is accelerated using Numba's @jit decorator for improved performance.
    
    Args:
        bootstrap_statistics: Bootstrap statistics array
        bootstrap_standard_errors: Standard errors of bootstrap statistics
        original_statistic: Original statistic computed on the data
        original_standard_error: Standard error of the original statistic
        alpha: Significance level (e.g., 0.05 for 95% confidence)
        
    Returns:
        np.ndarray: Confidence intervals [lower, upper]
    """
    # Calculate studentized bootstrap statistics
    n_bootstraps = bootstrap_statistics.shape[0]
    
    # Handle different dimensions
    if bootstrap_statistics.ndim == 1:
        # For 1D statistics (scalar statistic)
        studentized_stats = (bootstrap_statistics - original_statistic) / bootstrap_standard_errors
        
        # Calculate percentiles of studentized statistics
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_quantile = np.percentile(studentized_stats, lower_percentile)
        upper_quantile = np.percentile(studentized_stats, upper_percentile)
        
        # Calculate confidence intervals
        lower = original_statistic - upper_quantile * original_standard_error
        upper = original_statistic - lower_quantile * original_standard_error
        
        return np.array([lower, upper])
    else:
        # For multi-dimensional statistics
        n_stats = bootstrap_statistics.shape[1]
        studentized_stats = np.zeros((n_bootstraps, n_stats))
        ci = np.zeros((2, n_stats))
        
        for i in range(n_stats):
            studentized_stats[:, i] = (bootstrap_statistics[:, i] - original_statistic[i]) / bootstrap_standard_errors[:, i]
            
            # Calculate percentiles of studentized statistics
            lower_percentile = alpha / 2 * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            lower_quantile = np.percentile(studentized_stats[:, i], lower_percentile)
            upper_quantile = np.percentile(studentized_stats[:, i], upper_percentile)
            
            # Calculate confidence intervals
            ci[0, i] = original_statistic[i] - upper_quantile * original_standard_error[i]
            ci[1, i] = original_statistic[i] - lower_quantile * original_standard_error[i]
        
        return ci


@jit(nopython=True, cache=True)
def _compute_bca_ci(
    bootstrap_statistics: np.ndarray,
    original_statistic: np.ndarray,
    jackknife_values: np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    Compute BCa (bias-corrected and accelerated) confidence intervals using Numba acceleration.
    
    This function computes confidence intervals using the BCa method, which adjusts for
    bias and skewness in the bootstrap distribution.
    It is accelerated using Numba's @jit decorator for improved performance.
    
    Args:
        bootstrap_statistics: Bootstrap statistics array
        original_statistic: Original statistic computed on the data
        jackknife_values: Jackknife values for acceleration calculation
        alpha: Significance level (e.g., 0.05 for 95% confidence)
        
    Returns:
        np.ndarray: Confidence intervals [lower, upper]
    """
    # Handle different dimensions
    if bootstrap_statistics.ndim == 1:
        # For 1D statistics (scalar statistic)
        # Calculate bias correction factor
        n_bootstraps = len(bootstrap_statistics)
        z0 = np.sum(bootstrap_statistics < original_statistic) / n_bootstraps
        z0 = np.clip(z0, 1e-10, 1 - 1e-10)  # Avoid extreme values
        z0 = -np.log(1 / z0 - 1) if z0 < 0.5 else np.log(z0 / (1 - z0))
        
        # Calculate acceleration factor
        n_obs = len(jackknife_values)
        jackknife_mean = np.mean(jackknife_values)
        num = np.sum((jackknife_mean - jackknife_values) ** 3)
        den = 6 * (np.sum((jackknife_mean - jackknife_values) ** 2) ** 1.5)
        a = num / (den + 1e-10)  # Add small constant to avoid division by zero
        
        # Calculate adjusted percentiles
        alpha1 = alpha / 2
        alpha2 = 1 - alpha / 2
        
        z1 = -np.log(1 / alpha1 - 1) if alpha1 < 0.5 else np.log(alpha1 / (1 - alpha1))
        z2 = -np.log(1 / alpha2 - 1) if alpha2 < 0.5 else np.log(alpha2 / (1 - alpha2))
        
        p1 = 1 / (1 + np.exp(-(z0 + (z0 + z1) / (1 - a * (z0 + z1)))))
        p2 = 1 / (1 + np.exp(-(z0 + (z0 + z2) / (1 - a * (z0 + z2)))))
        
        # Calculate confidence intervals
        lower = np.percentile(bootstrap_statistics, p1 * 100)
        upper = np.percentile(bootstrap_statistics, p2 * 100)
        
        return np.array([lower, upper])
    else:
        # For multi-dimensional statistics
        n_stats = bootstrap_statistics.shape[1]
        ci = np.zeros((2, n_stats))
        
        for i in range(n_stats):
            # Calculate bias correction factor
            n_bootstraps = len(bootstrap_statistics)
            z0 = np.sum(bootstrap_statistics[:, i] < original_statistic[i]) / n_bootstraps
            z0 = np.clip(z0, 1e-10, 1 - 1e-10)  # Avoid extreme values
            z0 = -np.log(1 / z0 - 1) if z0 < 0.5 else np.log(z0 / (1 - z0))
            
            # Calculate acceleration factor
            n_obs = len(jackknife_values)
            jackknife_mean = np.mean(jackknife_values[:, i])
            num = np.sum((jackknife_mean - jackknife_values[:, i]) ** 3)
            den = 6 * (np.sum((jackknife_mean - jackknife_values[:, i]) ** 2) ** 1.5)
            a = num / (den + 1e-10)  # Add small constant to avoid division by zero
            
            # Calculate adjusted percentiles
            alpha1 = alpha / 2
            alpha2 = 1 - alpha / 2
            
            z1 = -np.log(1 / alpha1 - 1) if alpha1 < 0.5 else np.log(alpha1 / (1 - alpha1))
            z2 = -np.log(1 / alpha2 - 1) if alpha2 < 0.5 else np.log(alpha2 / (1 - alpha2))
            
            p1 = 1 / (1 + np.exp(-(z0 + (z0 + z1) / (1 - a * (z0 + z1)))))
            p2 = 1 / (1 + np.exp(-(z0 + (z0 + z2) / (1 - a * (z0 + z2)))))
            
            # Calculate confidence intervals
            ci[0, i] = np.percentile(bootstrap_statistics[:, i], p1 * 100)
            ci[1, i] = np.percentile(bootstrap_statistics[:, i], p2 * 100)
        
        return ci


def compute_confidence_interval(
    bootstrap_statistics: np.ndarray,
    original_statistic: Optional[Union[float, np.ndarray]] = None,
    confidence_level: float = 0.95,
    method: Literal['percentile', 'basic', 'studentized', 'bca'] = 'percentile',
    bootstrap_standard_errors: Optional[np.ndarray] = None,
    original_standard_error: Optional[Union[float, np.ndarray]] = None,
    jackknife_values: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute confidence intervals from bootstrap statistics.
    
    This function computes confidence intervals using various methods, including
    percentile, basic bootstrap, studentized bootstrap, and BCa (bias-corrected
    and accelerated) methods.
    
    Args:
        bootstrap_statistics: Bootstrap statistics array
        original_statistic: Original statistic computed on the data (required for all methods except percentile)
        confidence_level: Confidence level (between 0 and 1)
        method: Method for computing confidence intervals
        bootstrap_standard_errors: Standard errors of bootstrap statistics (required for studentized method)
        original_standard_error: Standard error of the original statistic (required for studentized method)
        jackknife_values: Jackknife values for acceleration calculation (required for BCa method)
        
    Returns:
        np.ndarray: Confidence intervals [lower, upper]
        
    Raises:
        ParameterError: If parameters are invalid or missing required values
        ValueError: If confidence_level is not between 0 and 1
    """
    # Validate bootstrap_statistics
    if not isinstance(bootstrap_statistics, np.ndarray):
        raise TypeError("bootstrap_statistics must be a NumPy array")
    
    if bootstrap_statistics.size == 0:
        raise ValueError("bootstrap_statistics cannot be empty")
    
    # Validate confidence_level
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    
    # Calculate alpha
    alpha = 1 - confidence_level
    
    # Compute confidence intervals based on the specified method
    if method == 'percentile':
        return _compute_percentile_ci(bootstrap_statistics, alpha)
    
    elif method == 'basic':
        # Validate original_statistic
        if original_statistic is None:
            raise ParameterError(
                "original_statistic is required for basic bootstrap method",
                param_name="original_statistic",
                param_value=None
            )
        
        # Convert original_statistic to numpy array if it's a scalar
        if isinstance(original_statistic, (int, float)):
            original_statistic = np.array(original_statistic)
        
        return _compute_basic_ci(bootstrap_statistics, original_statistic, alpha)
    
    elif method == 'studentized':
        # Validate required parameters
        if original_statistic is None:
            raise ParameterError(
                "original_statistic is required for studentized bootstrap method",
                param_name="original_statistic",
                param_value=None
            )
        
        if bootstrap_standard_errors is None:
            raise ParameterError(
                "bootstrap_standard_errors is required for studentized bootstrap method",
                param_name="bootstrap_standard_errors",
                param_value=None
            )
        
        if original_standard_error is None:
            raise ParameterError(
                "original_standard_error is required for studentized bootstrap method",
                param_name="original_standard_error",
                param_value=None
            )
        
        # Convert to numpy arrays if scalars
        if isinstance(original_statistic, (int, float)):
            original_statistic = np.array(original_statistic)
        
        if isinstance(original_standard_error, (int, float)):
            original_standard_error = np.array(original_standard_error)
        
        return _compute_studentized_ci(
            bootstrap_statistics, bootstrap_standard_errors,
            original_statistic, original_standard_error, alpha
        )
    
    elif method == 'bca':
        # Validate required parameters
        if original_statistic is None:
            raise ParameterError(
                "original_statistic is required for BCa bootstrap method",
                param_name="original_statistic",
                param_value=None
            )
        
        if jackknife_values is None:
            raise ParameterError(
                "jackknife_values is required for BCa bootstrap method",
                param_name="jackknife_values",
                param_value=None
            )
        
        # Convert to numpy array if scalar
        if isinstance(original_statistic, (int, float)):
            original_statistic = np.array(original_statistic)
        
        return _compute_bca_ci(
            bootstrap_statistics, original_statistic, jackknife_values, alpha
        )
    
    else:
        raise ValueError(
            f"Invalid method: {method}. Must be one of: 'percentile', 'basic', 'studentized', 'bca'"
        )


def compute_bootstrap_p_value(
    bootstrap_statistics: np.ndarray,
    original_statistic: Union[float, np.ndarray],
    alternative: Literal['two-sided', 'greater', 'less'] = 'two-sided'
) -> Union[float, np.ndarray]:
    """
    Compute bootstrap p-value.
    
    This function computes the bootstrap p-value for the null hypothesis based on
    the bootstrap distribution and the original statistic.
    
    Args:
        bootstrap_statistics: Bootstrap statistics array
        original_statistic: Original statistic computed on the data
        alternative: Alternative hypothesis ('two-sided', 'greater', or 'less')
        
    Returns:
        Union[float, np.ndarray]: Bootstrap p-value(s)
        
    Raises:
        ValueError: If alternative is not one of 'two-sided', 'greater', or 'less'
    """
    if alternative not in ["two-sided", "greater", "less"]:
        raise ValueError(
            "alternative must be one of 'two-sided', 'greater', or 'less'"
        )
    
    # Handle different dimensions
    if bootstrap_statistics.ndim == 1 and np.isscalar(original_statistic):
        # For 1D statistics (scalar statistic)
        if alternative == "two-sided":
            p_value = np.mean(np.abs(bootstrap_statistics) >= np.abs(original_statistic))
        elif alternative == "greater":
            p_value = np.mean(bootstrap_statistics >= original_statistic)
        else:  # alternative == "less"
            p_value = np.mean(bootstrap_statistics <= original_statistic)
        
        return float(p_value)
    
    elif bootstrap_statistics.ndim == 2 and isinstance(original_statistic, np.ndarray):
        # For multi-dimensional statistics
        n_stats = bootstrap_statistics.shape[1]
        p_values = np.zeros(n_stats)
        
        for i in range(n_stats):
            if alternative == "two-sided":
                p_values[i] = np.mean(np.abs(bootstrap_statistics[:, i]) >= np.abs(original_statistic[i]))
            elif alternative == "greater":
                p_values[i] = np.mean(bootstrap_statistics[:, i] >= original_statistic[i])
            else:  # alternative == "less"
                p_values[i] = np.mean(bootstrap_statistics[:, i] <= original_statistic[i])
        
        return p_values
    
    else:
        # Handle mixed dimensions
        if bootstrap_statistics.ndim == 1:
            # Convert original_statistic to scalar if it's an array with a single element
            if isinstance(original_statistic, np.ndarray) and original_statistic.size == 1:
                original_statistic = float(original_statistic.item())
                return compute_bootstrap_p_value(bootstrap_statistics, original_statistic, alternative)
            else:
                raise ValueError(
                    "Dimension mismatch: bootstrap_statistics is 1D but original_statistic is not a scalar"
                )
        else:  # bootstrap_statistics.ndim == 2
            # Convert original_statistic to array if it's a scalar
            if np.isscalar(original_statistic):
                original_statistic = np.array([original_statistic])
                bootstrap_statistics = bootstrap_statistics[:, np.newaxis]
                return compute_bootstrap_p_value(bootstrap_statistics, original_statistic, alternative)
            else:
                raise ValueError(
                    "Dimension mismatch: bootstrap_statistics is 2D but original_statistic is not an array"
                )


def compute_jackknife_values(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], Union[float, np.ndarray]]
) -> np.ndarray:
    """
    Compute jackknife values for a statistic.
    
    This function computes jackknife values by leaving out one observation at a time
    and computing the statistic on the remaining data.
    
    Args:
        data: Data array
        statistic_func: Function to compute the statistic of interest
        
    Returns:
        np.ndarray: Jackknife values
    """
    n_obs = len(data)
    
    # Compute statistic on full data to determine output shape
    full_stat = statistic_func(data)
    
    # Initialize jackknife values array
    if np.isscalar(full_stat):
        jackknife_values = np.zeros(n_obs)
    else:
        jackknife_values = np.zeros((n_obs, len(full_stat)))
    
    # Compute jackknife values
    for i in range(n_obs):
        # Create leave-one-out sample
        jackknife_sample = np.delete(data, i, axis=0)
        
        # Compute statistic on jackknife sample
        jackknife_stat = statistic_func(jackknife_sample)
        
        # Store jackknife value
        jackknife_values[i] = jackknife_stat
    
    return jackknife_values


def compute_bootstrap_standard_errors(
    bootstrap_statistics: np.ndarray
) -> np.ndarray:
    """
    Compute standard errors from bootstrap statistics.
    
    This function computes the standard errors of bootstrap statistics by taking
    the standard deviation of the bootstrap distribution.
    
    Args:
        bootstrap_statistics: Bootstrap statistics array
        
    Returns:
        np.ndarray: Bootstrap standard errors
    """
    # Handle different dimensions
    if bootstrap_statistics.ndim == 1:
        # For 1D statistics (scalar statistic)
        return np.std(bootstrap_statistics, ddof=1)
    else:
        # For multi-dimensional statistics
        return np.std(bootstrap_statistics, axis=0, ddof=1)


def compute_bootstrap_bias(
    bootstrap_statistics: np.ndarray,
    original_statistic: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Compute bootstrap bias.
    
    This function computes the bootstrap bias, which is the difference between
    the mean of the bootstrap distribution and the original statistic.
    
    Args:
        bootstrap_statistics: Bootstrap statistics array
        original_statistic: Original statistic computed on the data
        
    Returns:
        Union[float, np.ndarray]: Bootstrap bias
    """
    # Handle different dimensions
    if bootstrap_statistics.ndim == 1 and np.isscalar(original_statistic):
        # For 1D statistics (scalar statistic)
        return np.mean(bootstrap_statistics) - original_statistic
    
    elif bootstrap_statistics.ndim == 2 and isinstance(original_statistic, np.ndarray):
        # For multi-dimensional statistics
        return np.mean(bootstrap_statistics, axis=0) - original_statistic
    
    else:
        # Handle mixed dimensions
        if bootstrap_statistics.ndim == 1:
            # Convert original_statistic to scalar if it's an array with a single element
            if isinstance(original_statistic, np.ndarray) and original_statistic.size == 1:
                original_statistic = float(original_statistic.item())
                return compute_bootstrap_bias(bootstrap_statistics, original_statistic)
            else:
                raise ValueError(
                    "Dimension mismatch: bootstrap_statistics is 1D but original_statistic is not a scalar"
                )
        else:  # bootstrap_statistics.ndim == 2
            # Convert original_statistic to array if it's a scalar
            if np.isscalar(original_statistic):
                original_statistic = np.array([original_statistic])
                bootstrap_statistics = bootstrap_statistics[:, np.newaxis]
                return compute_bootstrap_bias(bootstrap_statistics, original_statistic)
            else:
                raise ValueError(
                    "Dimension mismatch: bootstrap_statistics is 2D but original_statistic is not an array"
                )


def compute_bias_corrected_estimate(
    original_statistic: Union[float, np.ndarray],
    bootstrap_bias: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Compute bias-corrected estimate.
    
    This function computes the bias-corrected estimate by subtracting the bootstrap
    bias from the original statistic.
    
    Args:
        original_statistic: Original statistic computed on the data
        bootstrap_bias: Bootstrap bias
        
    Returns:
        Union[float, np.ndarray]: Bias-corrected estimate
    """
    return original_statistic - bootstrap_bias


def plot_bootstrap_distribution(
    bootstrap_statistics: np.ndarray,
    original_statistic: Optional[Union[float, np.ndarray]] = None,
    confidence_level: float = 0.95,
    bins: int = 30,
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    show_ci: bool = True,
    ci_method: Literal['percentile', 'basic', 'studentized', 'bca'] = 'percentile',
    show_original: bool = True,
    show_bias_corrected: bool = False,
    **kwargs: Any
) -> plt.Figure:
    """
    Plot bootstrap distribution with confidence intervals.
    
    This function creates a histogram of the bootstrap distribution with optional
    confidence intervals, original statistic, and bias-corrected estimate.
    
    Args:
        bootstrap_statistics: Bootstrap statistics array
        original_statistic: Original statistic computed on the data
        confidence_level: Confidence level for intervals (between 0 and 1)
        bins: Number of histogram bins
        figsize: Figure size (width, height)
        title: Plot title
        xlabel: x-axis label
        ylabel: y-axis label
        show_ci: Whether to show confidence intervals
        ci_method: Method for computing confidence intervals
        show_original: Whether to show the original statistic
        show_bias_corrected: Whether to show the bias-corrected estimate
        **kwargs: Additional keyword arguments for matplotlib
        
    Returns:
        plt.Figure: Matplotlib figure object
        
    Raises:
        ValueError: If bootstrap_statistics is not 1D
    """
    # Validate bootstrap_statistics
    if bootstrap_statistics.ndim != 1:
        raise ValueError("bootstrap_statistics must be 1D for plotting")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram
    ax.hist(bootstrap_statistics, bins=bins, alpha=0.7, **kwargs)
    
    # Add confidence intervals if requested
    if show_ci:
        if ci_method == 'percentile':
            ci = compute_confidence_interval(
                bootstrap_statistics=bootstrap_statistics,
                confidence_level=confidence_level,
                method='percentile'
            )
        elif ci_method == 'basic':
            if original_statistic is None:
                raise ValueError("original_statistic is required for basic bootstrap method")
            ci = compute_confidence_interval(
                bootstrap_statistics=bootstrap_statistics,
                original_statistic=original_statistic,
                confidence_level=confidence_level,
                method='basic'
            )
        else:
            raise ValueError(
                f"ci_method '{ci_method}' not supported for plotting. "
                f"Use 'percentile' or 'basic'."
            )
        
        # Add vertical lines for confidence intervals
        ax.axvline(x=ci[0], color='r', linestyle='--', 
                  label=f'{confidence_level*100:.1f}% CI Lower')
        ax.axvline(x=ci[1], color='r', linestyle='--', 
                  label=f'{confidence_level*100:.1f}% CI Upper')
    
    # Add original statistic if requested
    if show_original and original_statistic is not None:
        ax.axvline(x=original_statistic, color='g', linestyle='-', 
                  label='Original Statistic')
    
    # Add bias-corrected estimate if requested
    if show_bias_corrected and original_statistic is not None:
        bootstrap_bias = compute_bootstrap_bias(bootstrap_statistics, original_statistic)
        bias_corrected = compute_bias_corrected_estimate(original_statistic, bootstrap_bias)
        ax.axvline(x=bias_corrected, color='b', linestyle='-.', 
                  label='Bias-Corrected Estimate')
    
    # Add labels and title
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel('Statistic Value')
    
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel('Frequency')
    
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title('Bootstrap Distribution')
    
    # Add legend if any lines were added
    if show_ci or (show_original and original_statistic is not None) or (show_bias_corrected and original_statistic is not None):
        ax.legend()
    
    plt.tight_layout()
    return fig


def format_bootstrap_result(
    result: BootstrapResult,
    statistic_name: Optional[str] = None,
    confidence_level: float = 0.95,
    include_bias: bool = False,
    include_standard_error: bool = True,
    include_p_value: bool = False,
    alternative: Literal['two-sided', 'greater', 'less'] = 'two-sided',
    as_dataframe: bool = False
) -> Union[Dict[str, Any], pd.DataFrame]:
    """Format bootstrap results for presentation.
    
    This function formats bootstrap results into a dictionary or DataFrame
    for easy presentation and interpretation.
    
    Args:
        result: Bootstrap result object
        statistic_name: Name of the statistic
        confidence_level: Confidence level for intervals
        include_bias: Whether to include bias estimates
        include_standard_error: Whether to include standard errors
        include_p_value: Whether to include p-values
        alternative: Alternative hypothesis for p-value calculation
        as_dataframe: Whether to return a DataFrame
        
    Returns:
        Union[Dict[str, Any], pd.DataFrame]: Formatted results
    """
    # Extract results
    original_statistic = result.original_statistic
    bootstrap_statistics = result.bootstrap_statistics
    
    # Compute confidence intervals
    confidence_intervals = compute_confidence_interval(
        bootstrap_statistics=bootstrap_statistics,
        original_statistic=original_statistic,
        confidence_level=confidence_level
    )
    
    # Compute standard errors if requested
    standard_errors = None
    if include_standard_error:
        standard_errors = compute_bootstrap_standard_errors(bootstrap_statistics)
    
    # Compute bias if requested
    bias = None
    if include_bias:
        bias = compute_bootstrap_bias(bootstrap_statistics, original_statistic)
    
    # Compute p-value if requested
    p_value = None
    if include_p_value:
        p_value = compute_bootstrap_p_value(
            bootstrap_statistics=bootstrap_statistics,
            original_statistic=original_statistic,
            alternative=alternative
        )
    
    # Format results
    if np.isscalar(original_statistic) or original_statistic.ndim == 0 or original_statistic.size == 1:
        # Scalar statistic
        if statistic_name is None:
            statistic_name = "Statistic"
        
        formatted_result = {
            "Statistic": statistic_name,
            "Estimate": float(original_statistic) if np.isscalar(original_statistic) else float(original_statistic.item()),
            f"CI Lower ({confidence_level*100:.1f}%)": float(confidence_intervals[0]),
            f"CI Upper ({confidence_level*100:.1f}%)": float(confidence_intervals[1]),
        }
        
        if include_standard_error:
            formatted_result["Standard Error"] = float(standard_errors) if np.isscalar(standard_errors) else float(standard_errors.item())
        
        if include_bias:
            formatted_result["Bias"] = float(bias) if np.isscalar(bias) else float(bias.item())
            formatted_result["Bias-Corrected Estimate"] = float(original_statistic - bias) if np.isscalar(original_statistic) else float((original_statistic - bias).item())
        
        if include_p_value:
            formatted_result["p-value"] = float(p_value) if np.isscalar(p_value) else float(p_value.item())
            formatted_result["Alternative"] = alternative
    else:
        # Vector statistic
        if statistic_name is None:
            statistic_name = [f"Statistic {i+1}" for i in range(len(original_statistic))]
        elif isinstance(statistic_name, str):
            statistic_name = [f"{statistic_name} {i+1}" for i in range(len(original_statistic))]
        
        formatted_result = []
        for i in range(len(original_statistic)):
            result_i = {
                "Statistic": statistic_name[i],
                "Estimate": float(original_statistic[i]),
                f"CI Lower ({confidence_level*100:.1f}%)": float(confidence_intervals[0, i]),
                f"CI Upper ({confidence_level*100:.1f}%)": float(confidence_intervals[1, i]),
            }
            
            if include_standard_error:
                result_i["Standard Error"] = float(standard_errors[i])
            
            if include_bias:
                result_i["Bias"] = float(bias[i])
                result_i["Bias-Corrected Estimate"] = float(original_statistic[i] - bias[i])
            
            if include_p_value:
                result_i["p-value"] = float(p_value[i])
                result_i["Alternative"] = alternative
            
            formatted_result.append(result_i)
    
    # Convert to DataFrame if requested
    if as_dataframe:
        if isinstance(formatted_result, list):
            return pd.DataFrame(formatted_result)
        else:
            return pd.DataFrame([formatted_result])
    
    return formatted_result


def validate_bootstrap_data(
    data: np.ndarray,
    min_observations: int = 10
) -> None:
    """
    Validate data for bootstrap methods.
    
    This function validates that the input data is suitable for bootstrap methods,
    checking for appropriate dimensions, sufficient observations, and absence of
    missing or infinite values.
    
    Args:
        data: Data to validate
        min_observations: Minimum number of observations required
        
    Raises:
        TypeError: If data is not a NumPy array
        ValueError: If data is empty or contains invalid values
        DataError: If data has insufficient observations
    """
    # Check type
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a NumPy array")
    
    # Check if empty
    if data.size == 0:
        raise ValueError("Data cannot be empty")
    
    # Check for NaN values
    if np.isnan(data).any():
        raise DataError(
            "Data contains NaN values",
            data_name="data",
            issue="contains NaN values"
        )
    
    # Check for infinite values
    if np.isinf(data).any():
        raise DataError(
            "Data contains infinite values",
            data_name="data",
            issue="contains infinite values"
        )
    
    # Check for sufficient observations
    if len(data) < min_observations:
        raise DataError(
            f"Data has insufficient observations (minimum {min_observations} required)",
            data_name="data",
            issue="insufficient observations",
            index=len(data)
        )


def determine_optimal_block_length(
    data: np.ndarray,
    method: Literal['auto', 'rule_of_thumb', 'politis_white'] = 'auto'
) -> int:
    """
    Determine optimal block length for block bootstrap methods.
    
    This function estimates the optimal block length for block bootstrap methods
    using various heuristics.
    
    Args:
        data: Time series data
        method: Method for determining block length
        
    Returns:
        int: Optimal block length
        
    Raises:
        ValueError: If method is invalid
    """
    n_obs = len(data)
    
    if method == 'auto' or method == 'rule_of_thumb':
        # Rule of thumb: n^(1/3)
        block_length = int(np.ceil(n_obs ** (1/3)))
        logger.info(f"Using rule of thumb for block length: {block_length}")
        return block_length
    
    elif method == 'politis_white':
        # Politis and White (2004) automatic block length selection
        # This is a simplified implementation
        
        # Compute sample autocorrelations
        acf = np.correlate(data - np.mean(data), data - np.mean(data), mode='full')
        acf = acf[n_obs-1:] / acf[n_obs-1]
        
        # Find the lag where autocorrelation becomes insignificant
        # Using 2/sqrt(n) as the threshold
        threshold = 2 / np.sqrt(n_obs)
        significant_lags = np.where(np.abs(acf) > threshold)[0]
        
        if len(significant_lags) == 0:
            # No significant autocorrelation, use minimum block length
            block_length = 1
        else:
            # Use the maximum significant lag
            max_lag = np.max(significant_lags)
            # Block length is approximately 2 * max_lag
            block_length = int(np.ceil(2 * max_lag))
        
        logger.info(f"Using Politis-White method for block length: {block_length}")
        return block_length
    
    else:
        raise ValueError(
            f"Invalid method: {method}. Must be one of: 'auto', 'rule_of_thumb', 'politis_white'"
        )


def bootstrap_confidence_band(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], np.ndarray],
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    bootstrap_method: Literal['block', 'stationary'] = 'stationary',
    block_length: Optional[int] = None,
    random_state: Optional[Union[int, np.random.Generator]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute bootstrap confidence bands for a function of the data.
    
    This function computes bootstrap confidence bands for a function that returns
    a vector of values (e.g., autocorrelation function, impulse response function).
    
    Args:
        data: Time series data
        statistic_func: Function that computes a vector statistic from the data
        n_bootstraps: Number of bootstrap replications
        confidence_level: Confidence level for the bands
        bootstrap_method: Method for bootstrap sampling
        block_length: Block length for block bootstrap methods
        random_state: Random number generator seed for reproducibility
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Original statistic, lower band, upper band
        
    Raises:
        ValueError: If statistic_func does not return a vector
    """
    # Validate data
    validate_bootstrap_data(data)
    
    # Compute original statistic
    original_statistic = statistic_func(data)
    
    # Validate that statistic_func returns a vector
    if not isinstance(original_statistic, np.ndarray) or original_statistic.ndim != 1:
        raise ValueError("statistic_func must return a 1D NumPy array")
    
    # Determine block length if not provided
    if block_length is None:
        block_length = determine_optimal_block_length(data)
    
    # Create bootstrap object
    if bootstrap_method == 'block':
        from mfe.models.bootstrap.block_bootstrap import BlockBootstrap
        bootstrap = BlockBootstrap(
            block_length=block_length,
            n_bootstraps=n_bootstraps,
            random_state=random_state
        )
    elif bootstrap_method == 'stationary':
        from mfe.models.bootstrap.stationary_bootstrap import StationaryBootstrap
        bootstrap = StationaryBootstrap(
            expected_block_length=block_length,
            n_bootstraps=n_bootstraps,
            random_state=random_state
        )
    else:
        raise ValueError(
            f"Invalid bootstrap_method: {bootstrap_method}. Must be one of: 'block', 'stationary'"
        )
    
    # Generate bootstrap indices
    indices = bootstrap.generate_indices(
        data_length=len(data),
        n_bootstraps=n_bootstraps,
        random_state=random_state
    )
    
    # Compute bootstrap statistics
    bootstrap_statistics = np.zeros((n_bootstraps, len(original_statistic)))
    
    for i in range(n_bootstraps):
        # Resample data
        bootstrap_sample = data[indices[i]]
        
        # Compute statistic on bootstrap sample
        bootstrap_statistics[i] = statistic_func(bootstrap_sample)
    
    # Compute confidence bands
    alpha = 1 - confidence_level
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_band = np.percentile(bootstrap_statistics, lower_percentile, axis=0)
    upper_band = np.percentile(bootstrap_statistics, upper_percentile, axis=0)
    
    return original_statistic, lower_band, upper_band


def plot_bootstrap_confidence_band(
    x_values: np.ndarray,
    original_statistic: np.ndarray,
    lower_band: np.ndarray,
    upper_band: np.ndarray,
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    confidence_level: float = 0.95,
    show_zero_line: bool = True,
    **kwargs: Any
) -> plt.Figure:
    """
    Plot bootstrap confidence bands.
    
    This function creates a plot of the original statistic with bootstrap confidence bands.
    
    Args:
        x_values: x-axis values
        original_statistic: Original statistic values
        lower_band: Lower confidence band
        upper_band: Upper confidence band
        figsize: Figure size (width, height)
        title: Plot title
        xlabel: x-axis label
        ylabel: y-axis label
        confidence_level: Confidence level for the bands
        show_zero_line: Whether to show a horizontal line at y=0
        **kwargs: Additional keyword arguments for matplotlib
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot original statistic
    ax.plot(x_values, original_statistic, 'b-', label='Statistic')
    
    # Plot confidence bands
    ax.fill_between(
        x_values, lower_band, upper_band, alpha=0.3, color='b',
        label=f'{confidence_level*100:.1f}% Confidence Band'
    )
    
    # Add zero line if requested
    if show_zero_line:
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Add labels and title
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title('Bootstrap Confidence Bands')
    
    # Add legend
    ax.legend()
    
    plt.tight_layout()
    return fig


def bootstrap_hypothesis_test(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], Union[float, np.ndarray]],
    null_value: Union[float, np.ndarray],
    n_bootstraps: int = 1000,
    alternative: Literal['two-sided', 'greater', 'less'] = 'two-sided',
    bootstrap_method: Literal['block', 'stationary'] = 'stationary',
    block_length: Optional[int] = None,
    random_state: Optional[Union[int, np.random.Generator]] = None
) -> Dict[str, Any]:
    """
    Perform bootstrap hypothesis test.
    
    This function performs a bootstrap hypothesis test for the null hypothesis
    that the statistic equals the null value.
    
    Args:
        data: Time series data
        statistic_func: Function that computes the statistic from the data
        null_value: Null hypothesis value
        n_bootstraps: Number of bootstrap replications
        alternative: Alternative hypothesis
        bootstrap_method: Method for bootstrap sampling
        block_length: Block length for block bootstrap methods
        random_state: Random number generator seed for reproducibility
        
    Returns:
        Dict[str, Any]: Test results including p-value, statistic, and null value
    """
    # Validate data
    validate_bootstrap_data(data)
    
    # Compute original statistic
    original_statistic = statistic_func(data)
    
    # Determine block length if not provided
    if block_length is None:
        block_length = determine_optimal_block_length(data)
    
    # Create bootstrap object
    if bootstrap_method == 'block':
        from mfe.models.bootstrap.block_bootstrap import BlockBootstrap
        bootstrap = BlockBootstrap(
            block_length=block_length,
            n_bootstraps=n_bootstraps,
            random_state=random_state
        )
    elif bootstrap_method == 'stationary':
        from mfe.models.bootstrap.stationary_bootstrap import StationaryBootstrap
        bootstrap = StationaryBootstrap(
            expected_block_length=block_length,
            n_bootstraps=n_bootstraps,
            random_state=random_state
        )
    else:
        raise ValueError(
            f"Invalid bootstrap_method: {bootstrap_method}. Must be one of: 'block', 'stationary'"
        )
    
    # Generate bootstrap indices
    indices = bootstrap.generate_indices(
        data_length=len(data),
        n_bootstraps=n_bootstraps,
        random_state=random_state
    )
    
    # Compute bootstrap statistics
    if np.isscalar(original_statistic):
        bootstrap_statistics = np.zeros(n_bootstraps)
    else:
        bootstrap_statistics = np.zeros((n_bootstraps, len(original_statistic)))
    
    for i in range(n_bootstraps):
        # Resample data
        bootstrap_sample = data[indices[i]]
        
        # Compute statistic on bootstrap sample
        bootstrap_statistics[i] = statistic_func(bootstrap_sample)
    
    # Compute bootstrap p-value
    p_value = compute_bootstrap_p_value(
        bootstrap_statistics=bootstrap_statistics,
        original_statistic=original_statistic - null_value,  # Center at null value
        alternative=alternative
    )
    
    # Prepare result
    result = {
        "statistic": original_statistic,
        "null_value": null_value,
        "p_value": p_value,
        "alternative": alternative,
        "bootstrap_method": bootstrap_method,
        "n_bootstraps": n_bootstraps,
        "block_length": block_length
    }
    
    return result


def summarize_bootstrap_hypothesis_test(
    test_result: Dict[str, Any],
    statistic_name: Optional[str] = None,
    alpha: float = 0.05
) -> str:
    """
    Summarize bootstrap hypothesis test results.
    
    This function creates a formatted summary of bootstrap hypothesis test results.
    
    Args:
        test_result: Test results from bootstrap_hypothesis_test
        statistic_name: Name of the statistic
        alpha: Significance level
        
    Returns:
        str: Formatted summary
    """
    # Extract test results
    statistic = test_result["statistic"]
    null_value = test_result["null_value"]
    p_value = test_result["p_value"]
    alternative = test_result["alternative"]
    bootstrap_method = test_result["bootstrap_method"]
    n_bootstraps = test_result["n_bootstraps"]
    block_length = test_result["block_length"]
    
    # Create statistic name if not provided
    if statistic_name is None:
        statistic_name = "Statistic"
    
    # Determine if we're dealing with a scalar or vector statistic
    is_scalar = np.isscalar(statistic) or (
        isinstance(statistic, np.ndarray) and statistic.size == 1
    )
    
    # Format summary
    summary = [
        "Bootstrap Hypothesis Test",
        "=" * 50,
        f"Bootstrap Method: {bootstrap_method}",
        f"Number of Bootstraps: {n_bootstraps}",
        f"Block Length: {block_length}",
        f"Alternative Hypothesis: {alternative}",
        "-" * 50,
    ]
    
    if is_scalar:
        # Format scalar result
        summary.extend([
            f"Null Hypothesis: {statistic_name} = {null_value}",
            f"Statistic Value: {statistic}",
            f"p-value: {p_value}",
            "-" * 50,
            f"Result: {'Reject' if p_value < alpha else 'Fail to reject'} null hypothesis at {alpha} significance level",
        ])
    else:
        # Format vector result
        n_stats = len(statistic)
        
        summary.append("Results:")
        
        for i in range(n_stats):
            stat_name = f"{statistic_name}_{i+1}" if statistic_name else f"Statistic_{i+1}"
            stat_value = statistic[i]
            null_val = null_value[i] if isinstance(null_value, np.ndarray) else null_value
            p_val = p_value[i] if isinstance(p_value, np.ndarray) else p_value
            
            summary.extend([
                f"  {stat_name}:",
                f"    Null Hypothesis: {stat_name} = {null_val}",
                f"    Statistic Value: {stat_value}",
                f"    p-value: {p_val}",
                f"    Result: {'Reject' if p_val < alpha else 'Fail to reject'} null hypothesis at {alpha} significance level",
                ""
            ])
    
    return "\n".join(summary)


def bootstrap_difference_test(
    data1: np.ndarray,
    data2: np.ndarray,
    statistic_func: Callable[[np.ndarray], Union[float, np.ndarray]],
    n_bootstraps: int = 1000,
    alternative: Literal['two-sided', 'greater', 'less'] = 'two-sided',
    paired: bool = False,
    bootstrap_method: Literal['block', 'stationary'] = 'stationary',
    block_length: Optional[int] = None,
    random_state: Optional[Union[int, np.random.Generator]] = None
) -> Dict[str, Any]:
    """
    Perform bootstrap test for the difference between two samples.
    
    This function performs a bootstrap test for the null hypothesis that the
    difference between the statistics of two samples is zero.
    
    Args:
        data1: First sample
        data2: Second sample
        statistic_func: Function that computes the statistic from the data
        n_bootstraps: Number of bootstrap replications
        alternative: Alternative hypothesis
        paired: Whether the samples are paired
        bootstrap_method: Method for bootstrap sampling
        block_length: Block length for block bootstrap methods
        random_state: Random number generator seed for reproducibility
        
    Returns:
        Dict[str, Any]: Test results including p-value, statistics, and difference
        
    Raises:
        ValueError: If paired=True but samples have different lengths
    """
    # Validate data
    validate_bootstrap_data(data1)
    validate_bootstrap_data(data2)
    
    # Check if paired samples have the same length
    if paired and len(data1) != len(data2):
        raise ValueError("Paired samples must have the same length")
    
    # Compute original statistics
    stat1 = statistic_func(data1)
    stat2 = statistic_func(data2)
    
    # Compute difference
    diff = stat1 - stat2
    
    # Determine block length if not provided
    if block_length is None:
        block_length1 = determine_optimal_block_length(data1)
        block_length2 = determine_optimal_block_length(data2)
        block_length = max(block_length1, block_length2)
    
    # Create random number generator
    if isinstance(random_state, int):
        rng = np.random.default_rng(random_state)
    elif random_state is None:
        rng = np.random.default_rng()
    else:
        rng = random_state
    
    # Create bootstrap objects
    if bootstrap_method == 'block':
        from mfe.models.bootstrap.block_bootstrap import BlockBootstrap
        bootstrap1 = BlockBootstrap(
            block_length=block_length,
            n_bootstraps=n_bootstraps,
            random_state=rng
        )
        bootstrap2 = BlockBootstrap(
            block_length=block_length,
            n_bootstraps=n_bootstraps,
            random_state=rng
        )
    elif bootstrap_method == 'stationary':
        from mfe.models.bootstrap.stationary_bootstrap import StationaryBootstrap
        bootstrap1 = StationaryBootstrap(
            expected_block_length=block_length,
            n_bootstraps=n_bootstraps,
            random_state=rng
        )
        bootstrap2 = StationaryBootstrap(
            expected_block_length=block_length,
            n_bootstraps=n_bootstraps,
            random_state=rng
        )
    else:
        raise ValueError(
            f"Invalid bootstrap_method: {bootstrap_method}. Must be one of: 'block', 'stationary'"
        )
    
    # Generate bootstrap indices
    indices1 = bootstrap1.generate_indices(
        data_length=len(data1),
        n_bootstraps=n_bootstraps,
        random_state=rng
    )
    
    if paired:
        # Use the same indices for both samples in paired case
        indices2 = indices1
    else:
        # Generate separate indices for the second sample
        indices2 = bootstrap2.generate_indices(
            data_length=len(data2),
            n_bootstraps=n_bootstraps,
            random_state=rng
        )
    
    # Compute bootstrap differences
    if np.isscalar(diff):
        bootstrap_diffs = np.zeros(n_bootstraps)
    else:
        bootstrap_diffs = np.zeros((n_bootstraps, len(diff)))
    
    for i in range(n_bootstraps):
        # Resample data
        bootstrap_sample1 = data1[indices1[i]]
        bootstrap_sample2 = data2[indices2[i]]
        
        # Compute statistics on bootstrap samples
        bootstrap_stat1 = statistic_func(bootstrap_sample1)
        bootstrap_stat2 = statistic_func(bootstrap_sample2)
        
        # Compute difference
        bootstrap_diffs[i] = bootstrap_stat1 - bootstrap_stat2
    
    # Compute bootstrap p-value
    p_value = compute_bootstrap_p_value(
        bootstrap_statistics=bootstrap_diffs,
        original_statistic=diff,
        alternative=alternative
    )
    
    # Prepare result
    result = {
        "statistic1": stat1,
        "statistic2": stat2,
        "difference": diff,
        "p_value": p_value,
        "alternative": alternative,
        "paired": paired,
        "bootstrap_method": bootstrap_method,
        "n_bootstraps": n_bootstraps,
        "block_length": block_length
    }
    
    return result


def summarize_bootstrap_difference_test(
    test_result: Dict[str, Any],
    statistic_name: Optional[str] = None,
    sample1_name: str = "Sample 1",
    sample2_name: str = "Sample 2",
    alpha: float = 0.05
) -> str:
    """
    Summarize bootstrap difference test results.
    
    This function creates a formatted summary of bootstrap difference test results.
    
    Args:
        test_result: Test results from bootstrap_difference_test
        statistic_name: Name of the statistic
        sample1_name: Name of the first sample
        sample2_name: Name of the second sample
        alpha: Significance level
        
    Returns:
        str: Formatted summary
    """
    # Extract test results
    stat1 = test_result["statistic1"]
    stat2 = test_result["statistic2"]
    diff = test_result["difference"]
    p_value = test_result["p_value"]
    alternative = test_result["alternative"]
    paired = test_result["paired"]
    bootstrap_method = test_result["bootstrap_method"]
    n_bootstraps = test_result["n_bootstraps"]
    block_length = test_result["block_length"]
    
    # Create statistic name if not provided
    if statistic_name is None:
        statistic_name = "Statistic"
    
    # Determine if we're dealing with a scalar or vector statistic
    is_scalar = np.isscalar(diff) or (
        isinstance(diff, np.ndarray) and diff.size == 1
    )
    
    # Format summary
    summary = [
        "Bootstrap Difference Test",
        "=" * 50,
        f"Bootstrap Method: {bootstrap_method}",
        f"Number of Bootstraps: {n_bootstraps}",
        f"Block Length: {block_length}",
        f"Paired Samples: {'Yes' if paired else 'No'}",
        f"Alternative Hypothesis: {alternative}",
        "-" * 50,
    ]
    
    if is_scalar:
        # Format scalar result
        summary.extend([
            f"Null Hypothesis: {statistic_name}({sample1_name}) = {statistic_name}({sample2_name})",
            f"{statistic_name}({sample1_name}): {stat1}",
            f"{statistic_name}({sample2_name}): {stat2}",
            f"Difference: {diff}",
            f"p-value: {p_value}",
            "-" * 50,
            f"Result: {'Reject' if p_value < alpha else 'Fail to reject'} null hypothesis at {alpha} significance level",
        ])
    else:
        # Format vector result
        n_stats = len(diff)
        
        summary.append("Results:")
        
        for i in range(n_stats):
            stat_name = f"{statistic_name}_{i+1}" if statistic_name else f"Statistic_{i+1}"
            stat1_val = stat1[i]
            stat2_val = stat2[i]
            diff_val = diff[i]
            p_val = p_value[i] if isinstance(p_value, np.ndarray) else p_value
            
            summary.extend([
                f"  {stat_name}:",
                f"    Null Hypothesis: {stat_name}({sample1_name}) = {stat_name}({sample2_name})",
                f"    {stat_name}({sample1_name}): {stat1_val}",
                f"    {stat_name}({sample2_name}): {stat2_val}",
                f"    Difference: {diff_val}",
                f"    p-value: {p_val}",
                f"    Result: {'Reject' if p_val < alpha else 'Fail to reject'} null hypothesis at {alpha} significance level",
                ""
            ])
    
    return "\n".join(summary)


def bootstrap_indices(
    n: int,
    n_bootstraps: int = 1000,
    random_state: Optional[Union[int, np.random.Generator]] = None
) -> np.ndarray:
    """Generate bootstrap indices for standard (iid) bootstrap.
    
    This function generates bootstrap indices for the standard bootstrap method,
    which assumes independent and identically distributed (iid) data.
    
    Args:
        n: Number of observations in the original data
        n_bootstraps: Number of bootstrap samples to generate
        random_state: Random number generator seed for reproducibility
        
    Returns:
        np.ndarray: Bootstrap indices of shape (n_bootstraps, n)
        
    Raises:
        ValueError: If inputs are invalid
    """
    # Validate inputs
    if n <= 0:
        raise ValueError("Number of observations must be positive")
    if n_bootstraps <= 0:
        raise ValueError("Number of bootstrap samples must be positive")
    
    # Set random state
    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, int):
        rng = np.random.default_rng(random_state)
    else:
        rng = random_state
    
    # Generate bootstrap indices
    indices = rng.integers(0, n, size=(n_bootstraps, n))
    
    return indices


def stationary_bootstrap_indices(
    n: int,
    block_length: Union[int, float],
    n_bootstraps: int = 1000,
    random_state: Optional[Union[int, np.random.Generator]] = None
) -> np.ndarray:
    """Generate bootstrap indices for stationary bootstrap.
    
    This function generates bootstrap indices for the stationary bootstrap method,
    which uses random block lengths drawn from a geometric distribution.
    
    Args:
        n: Number of observations in the original data
        block_length: Expected block length (parameter of geometric distribution)
        n_bootstraps: Number of bootstrap samples to generate
        random_state: Random number generator seed for reproducibility
        
    Returns:
        np.ndarray: Bootstrap indices of shape (n_bootstraps, n)
        
    Raises:
        ValueError: If inputs are invalid
    """
    # Validate inputs
    if n <= 0:
        raise ValueError("Number of observations must be positive")
    if block_length <= 0:
        raise ValueError("Block length must be positive")
    if n_bootstraps <= 0:
        raise ValueError("Number of bootstrap samples must be positive")
    
    # Set random state
    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, int):
        rng = np.random.default_rng(random_state)
    else:
        rng = random_state
    
    # Probability of starting a new block
    p = 1.0 / block_length
    
    # Generate bootstrap indices
    indices = np.zeros((n_bootstraps, n), dtype=int)
    
    for i in range(n_bootstraps):
        # Initialize with random starting points
        indices[i, 0] = rng.integers(0, n)
        
        for j in range(1, n):
            # With probability p, start a new block
            if rng.random() < p:
                indices[i, j] = rng.integers(0, n)
            else:
                # Continue the current block
                indices[i, j] = (indices[i, j-1] + 1) % n
    
    return indices


def moving_block_bootstrap_indices(
    n: int,
    block_length: int,
    n_bootstraps: int = 1000,
    random_state: Optional[Union[int, np.random.Generator]] = None
) -> np.ndarray:
    """Generate bootstrap indices for moving block bootstrap.
    
    This function generates bootstrap indices for the moving block bootstrap method,
    which uses fixed-length blocks with overlap.
    
    Args:
        n: Number of observations in the original data
        block_length: Block length
        n_bootstraps: Number of bootstrap samples to generate
        random_state: Random number generator seed for reproducibility
        
    Returns:
        np.ndarray: Bootstrap indices of shape (n_bootstraps, n)
        
    Raises:
        ValueError: If inputs are invalid
    """
    # Validate inputs
    if n <= 0:
        raise ValueError("Number of observations must be positive")
    if block_length <= 0:
        raise ValueError("Block length must be positive")
    if block_length > n:
        raise ValueError("Block length cannot exceed the number of observations")
    if n_bootstraps <= 0:
        raise ValueError("Number of bootstrap samples must be positive")
    
    # Set random state
    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, int):
        rng = np.random.default_rng(random_state)
    else:
        rng = random_state
    
    # Number of blocks needed
    n_blocks_needed = int(np.ceil(n / block_length))
    
    # Number of possible starting positions for blocks
    n_possible_blocks = n - block_length + 1
    
    # Generate bootstrap indices
    indices = np.zeros((n_bootstraps, n), dtype=int)
    
    for i in range(n_bootstraps):
        # Select random blocks
        block_starts = rng.integers(0, n_possible_blocks, size=n_blocks_needed)
        
        # Fill in indices
        idx = 0
        for start in block_starts:
            end = min(idx + block_length, n)
            block_end = start + (end - idx)
            indices[i, idx:end] = np.arange(start, block_end)
            idx = end
            
            if idx >= n:
                break
    
    return indices


def circular_block_bootstrap_indices(
    n: int,
    block_length: int,
    n_bootstraps: int = 1000,
    random_state: Optional[Union[int, np.random.Generator]] = None
) -> np.ndarray:
    """Generate bootstrap indices for circular block bootstrap.
    
    This function generates bootstrap indices for the circular block bootstrap method,
    which uses fixed-length blocks with wrap-around at the end of the series.
    
    Args:
        n: Number of observations in the original data
        block_length: Block length
        n_bootstraps: Number of bootstrap samples to generate
        random_state: Random number generator seed for reproducibility
        
    Returns:
        np.ndarray: Bootstrap indices of shape (n_bootstraps, n)
        
    Raises:
        ValueError: If inputs are invalid
    """
    # Validate inputs
    if n <= 0:
        raise ValueError("Number of observations must be positive")
    if block_length <= 0:
        raise ValueError("Block length must be positive")
    if block_length > n:
        raise ValueError("Block length cannot exceed the number of observations")
    if n_bootstraps <= 0:
        raise ValueError("Number of bootstrap samples must be positive")
    
    # Set random state
    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, int):
        rng = np.random.default_rng(random_state)
    else:
        rng = random_state
    
    # Number of blocks needed
    n_blocks_needed = int(np.ceil(n / block_length))
    
    # Generate bootstrap indices
    indices = np.zeros((n_bootstraps, n), dtype=int)
    
    for i in range(n_bootstraps):
        # Select random blocks
        block_starts = rng.integers(0, n, size=n_blocks_needed)
        
        # Fill in indices
        idx = 0
        for start in block_starts:
            end = min(idx + block_length, n)
            block_indices = np.arange(start, start + (end - idx)) % n
            indices[i, idx:end] = block_indices
            idx = end
            
            if idx >= n:
                break
    
    return indices


def compute_statistic(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], Union[float, np.ndarray]],
    indices: Optional[np.ndarray] = None,
    axis: int = 0
) -> np.ndarray:
    """Compute a statistic on bootstrap samples.
    
    This function computes a statistic on bootstrap samples generated from the
    original data using the provided indices.
    
    Args:
        data: Original data
        statistic_func: Function that computes the statistic
        indices: Bootstrap indices of shape (n_bootstraps, n)
        axis: Axis along which to apply the bootstrap
        
    Returns:
        np.ndarray: Bootstrap statistics
        
    Raises:
        ValueError: If inputs are invalid
    """
    # Validate inputs
    if data.ndim == 0:
        raise ValueError("Data must be at least 1-dimensional")
    
    # If indices are not provided, just compute the statistic on the original data
    if indices is None:
        return statistic_func(data)
    
    # Validate indices
    if indices.ndim != 2:
        raise ValueError("Indices must be 2-dimensional")
    
    n_bootstraps = indices.shape[0]
    
    # Compute statistic on each bootstrap sample
    bootstrap_statistics = []
    for i in range(n_bootstraps):
        # Generate bootstrap sample
        if axis == 0:
            bootstrap_sample = data[indices[i]]
        else:
            bootstrap_sample = np.take(data, indices[i], axis=axis)
        
        # Compute statistic
        statistic = statistic_func(bootstrap_sample)
        bootstrap_statistics.append(statistic)
    
    # Convert to numpy array
    bootstrap_statistics = np.array(bootstrap_statistics)
    
    return bootstrap_statistics


def compute_variance(
    bootstrap_statistics: np.ndarray
) -> Union[float, np.ndarray]:
    """Compute the variance of bootstrap statistics.
    
    Args:
        bootstrap_statistics: Bootstrap statistics
        
    Returns:
        Union[float, np.ndarray]: Variance of bootstrap statistics
    """
    return np.var(bootstrap_statistics, axis=0, ddof=1)


def compute_pvalue(
    bootstrap_statistics: np.ndarray,
    original_statistic: Union[float, np.ndarray],
    alternative: Literal['two-sided', 'greater', 'less'] = 'two-sided'
) -> Union[float, np.ndarray]:
    """Compute bootstrap p-value.
    
    This function computes a bootstrap p-value for a hypothesis test based on
    the distribution of bootstrap statistics.
    
    Args:
        bootstrap_statistics: Bootstrap statistics
        original_statistic: Original statistic computed on the data
        alternative: Alternative hypothesis
            - 'two-sided': H1: statistic != 0
            - 'greater': H1: statistic > 0
            - 'less': H1: statistic < 0
        
    Returns:
        Union[float, np.ndarray]: Bootstrap p-value
        
    Raises:
        ValueError: If inputs are invalid
    """
    # Validate inputs
    if bootstrap_statistics.ndim == 0:
        raise ValueError("Bootstrap statistics must be at least 1-dimensional")
    
    # Compute p-value based on alternative hypothesis
    if alternative == 'two-sided':
        # Two-sided test: p = 2 * min(p(stat >= orig), p(stat <= orig))
        p_upper = np.mean(bootstrap_statistics >= original_statistic, axis=0)
        p_lower = np.mean(bootstrap_statistics <= original_statistic, axis=0)
        p_value = 2.0 * np.minimum(p_upper, p_lower)
        
        # Ensure p-value is at most 1
        p_value = np.minimum(p_value, 1.0)
    elif alternative == 'greater':
        # One-sided test (greater): p = p(stat >= orig)
        p_value = np.mean(bootstrap_statistics >= original_statistic, axis=0)
    elif alternative == 'less':
        # One-sided test (less): p = p(stat <= orig)
        p_value = np.mean(bootstrap_statistics <= original_statistic, axis=0)
    else:
        raise ValueError(
            f"Invalid alternative hypothesis: {alternative}. "
            f"Must be one of: 'two-sided', 'greater', or 'less'."
        )
    
    return p_value
