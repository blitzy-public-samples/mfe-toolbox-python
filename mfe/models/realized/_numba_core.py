# mfe/models/realized/_numba_core.py
"""
Numba-accelerated core functions for realized volatility computations.

This module provides optimized implementations of performance-critical functions
used in realized volatility estimators. These functions are accelerated using
Numba's just-in-time (JIT) compilation to achieve near-native performance while
maintaining the simplicity and readability of Python code.

The module serves as a replacement for the MATLAB MEX files in the original
MFE Toolbox, providing similar performance benefits without requiring separate
compilation steps or platform-specific binaries. Functions are decorated with
Numba's @jit or @njit decorators to enable automatic optimization at runtime.

All functions include proper type hints and documentation to ensure usability
and maintainability, while the Numba acceleration remains transparent to users
of the higher-level realized volatility estimators.
"""

import logging
import warnings
from typing import Tuple, Optional, Union, List, Dict, Any, Callable

import numpy as np

# Set up module-level logger
logger = logging.getLogger("mfe.models.realized._numba_core")

# Try to import numba for JIT compilation
try:
    from numba import jit, njit, prange
    HAS_NUMBA = True
    logger.debug("Numba available for realized volatility acceleration")
except ImportError:
    # Create no-op decorators with the same signatures as jit/njit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    
    def prange(*args):
        return range(*args)
    
    HAS_NUMBA = False
    logger.warning("Numba not available. Realized volatility computations will use pure NumPy implementations.")


# ============================================================================
# Core Realized Variance Functions
# ============================================================================

@njit(cache=True)
def _realized_variance_core(returns: np.ndarray) -> float:
    """
    Numba-accelerated core implementation of realized variance computation.
    
    Args:
        returns: Array of returns
        
    Returns:
        Realized variance (sum of squared returns)
    """
    return np.sum(returns**2)


@njit(cache=True)
def _realized_variance_subsampled_core(returns: np.ndarray, subsample_factor: int) -> float:
    """
    Numba-accelerated core implementation of subsampled realized variance.
    
    Args:
        returns: Array of returns
        subsample_factor: Number of subsamples to use
        
    Returns:
        Subsampled realized variance
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


# ============================================================================
# Core Bipower Variation Functions
# ============================================================================

@njit(cache=True)
def _bipower_variation_core(returns: np.ndarray) -> float:
    """
    Numba-accelerated core implementation of bipower variation.
    
    Args:
        returns: Array of returns
        
    Returns:
        Bipower variation
    """
    n = len(returns)
    abs_returns = np.abs(returns)
    bipower_sum = 0.0
    
    for i in range(n-1):
        bipower_sum += abs_returns[i] * abs_returns[i+1]
    
    # Scaling factor (π/2) for asymptotic consistency
    scaling = np.pi / 2
    
    # Apply finite sample correction
    correction = n / (n - 1)
    
    return correction * bipower_sum * (scaling**2)


@njit(cache=True)
def _tripower_variation_core(returns: np.ndarray) -> float:
    """
    Numba-accelerated core implementation of tripower variation.
    
    Args:
        returns: Array of returns
        
    Returns:
        Tripower variation
    """
    n = len(returns)
    abs_returns = np.abs(returns)
    tripower_sum = 0.0
    
    for i in range(n-2):
        tripower_sum += (abs_returns[i]**(2/3)) * (abs_returns[i+1]**(2/3)) * (abs_returns[i+2]**(2/3))
    
    # Scaling factor for asymptotic consistency
    scaling = (2**(2/3)) * (np.pi**(1/3)) / (np.pi + np.pi/2)
    
    # Apply finite sample correction
    correction = n**2 / ((n-2) * (n-1))
    
    return correction * tripower_sum * (scaling**3)


@njit(cache=True)
def _threshold_bipower_variation_core(returns: np.ndarray, threshold: float) -> float:
    """
    Numba-accelerated core implementation of threshold bipower variation.
    
    Args:
        returns: Array of returns
        threshold: Threshold for jump detection
        
    Returns:
        Threshold bipower variation
    """
    n = len(returns)
    abs_returns = np.abs(returns)
    bipower_sum = 0.0
    
    for i in range(n-1):
        # Apply threshold to both returns
        if abs_returns[i] <= threshold and abs_returns[i+1] <= threshold:
            bipower_sum += abs_returns[i] * abs_returns[i+1]
    
    # Scaling factor (π/2) for asymptotic consistency
    scaling = np.pi / 2
    
    # Apply finite sample correction
    correction = n / (n - 1)
    
    return correction * bipower_sum * (scaling**2)


# ============================================================================
# Core Quarticity Functions
# ============================================================================

@njit(cache=True)
def _realized_quarticity_core(returns: np.ndarray) -> float:
    """
    Numba-accelerated core implementation of realized quarticity.
    
    Args:
        returns: Array of returns
        
    Returns:
        Realized quarticity
    """
    n = len(returns)
    return (n / 3) * np.sum(returns**4)


@njit(cache=True)
def _tri_quarticity_core(returns: np.ndarray) -> float:
    """
    Numba-accelerated core implementation of tri-power quarticity.
    
    Args:
        returns: Array of returns
        
    Returns:
        Tri-power quarticity
    """
    n = len(returns)
    abs_returns = np.abs(returns)
    tri_quarticity_sum = 0.0
    
    for i in range(n-2):
        tri_quarticity_sum += (abs_returns[i]**(4/3)) * (abs_returns[i+1]**(4/3)) * (abs_returns[i+2]**(4/3))
    
    # Scaling factor for asymptotic consistency
    scaling = (2**(4/3)) * (np.pi**(2/3)) / (np.pi + np.pi/2)
    
    # Apply finite sample correction
    correction = n**2 / ((n-2) * (n-1))
    
    return correction * tri_quarticity_sum * (scaling**3)


# ============================================================================
# Core Kernel Functions
# ============================================================================

@njit(cache=True)
def _compute_kernel_weights(n: int, kernel_type: str, bandwidth: float) -> np.ndarray:
    """
    Numba-accelerated implementation of kernel weight computation.
    
    Args:
        n: Number of weights to compute
        kernel_type: Type of kernel ('bartlett', 'parzen', 'tukey-hanning', 'quadratic')
        bandwidth: Bandwidth parameter
        
    Returns:
        Array of kernel weights
    """
    weights = np.zeros(n)
    
    if kernel_type == 'bartlett':
        # Bartlett kernel (linear)
        for i in range(n):
            if i <= bandwidth:
                weights[i] = 1.0 - i / (bandwidth + 1)
            else:
                weights[i] = 0.0
    
    elif kernel_type == 'parzen':
        # Parzen kernel
        for i in range(n):
            x = i / bandwidth
            if x <= 0.5:
                weights[i] = 1.0 - 6.0 * x**2 + 6.0 * x**3
            elif x <= 1.0:
                weights[i] = 2.0 * (1.0 - x)**3
            else:
                weights[i] = 0.0
    
    elif kernel_type in ['tukey-hanning', 'tukey', 'hanning']:
        # Tukey-Hanning kernel
        for i in range(n):
            x = i / bandwidth
            if x <= 1.0:
                weights[i] = 0.5 * (1.0 + np.cos(np.pi * x))
            else:
                weights[i] = 0.0
    
    elif kernel_type == 'quadratic':
        # Quadratic kernel
        for i in range(n):
            x = i / bandwidth
            if x <= 1.0:
                weights[i] = (1.0 - x**2)**2
            else:
                weights[i] = 0.0
    
    else:
        # Default to Bartlett if kernel type is not recognized
        for i in range(n):
            if i <= bandwidth:
                weights[i] = 1.0 - i / (bandwidth + 1)
            else:
                weights[i] = 0.0
    
    return weights


@njit(cache=True)
def _realized_kernel_core(returns: np.ndarray, kernel_weights: np.ndarray) -> float:
    """
    Numba-accelerated core implementation of realized kernel.
    
    Args:
        returns: Array of returns
        kernel_weights: Array of kernel weights
        
    Returns:
        Realized kernel estimate
    """
    n = len(returns)
    max_lag = min(len(kernel_weights) - 1, n - 1)
    
    # Compute realized variance (lag 0 term)
    rk = np.sum(returns**2)
    
    # Add autocovariance terms
    for h in range(1, max_lag + 1):
        # Compute autocovariance at lag h
        gamma_h = 0.0
        for i in range(n - h):
            gamma_h += returns[i] * returns[i + h]
        
        # Add weighted autocovariance to the kernel
        rk += 2 * kernel_weights[h] * gamma_h
    
    return rk


@njit(cache=True)
def _realized_kernel_with_jitter_core(returns: np.ndarray, kernel_weights: np.ndarray, 
                                     jitter: int) -> float:
    """
    Numba-accelerated core implementation of realized kernel with jittering.
    
    Args:
        returns: Array of returns
        kernel_weights: Array of kernel weights
        jitter: Jitter parameter for end-point treatment
        
    Returns:
        Realized kernel estimate with jittering
    """
    n = len(returns)
    max_lag = min(len(kernel_weights) - 1, n - 1)
    
    # Compute realized variance (lag 0 term)
    rk = np.sum(returns**2)
    
    # Add autocovariance terms with jittering
    for h in range(1, max_lag + 1):
        # Compute autocovariance at lag h with jittering
        gamma_h = 0.0
        
        # Start jitter positions
        start_positions = np.arange(jitter)
        
        # Compute jittered autocovariance
        for start in start_positions:
            gamma_h_j = 0.0
            count = 0
            
            for i in range(start, n - h, jitter):
                gamma_h_j += returns[i] * returns[i + h]
                count += 1
            
            if count > 0:
                gamma_h += gamma_h_j / count
        
        # Average over jitter positions
        gamma_h /= jitter
        
        # Add weighted autocovariance to the kernel
        rk += 2 * kernel_weights[h] * gamma_h
    
    return rk


# ============================================================================
# Core Multipower Variation Functions
# ============================================================================

@njit(cache=True)
def _multipower_variation_core(returns: np.ndarray, powers: np.ndarray) -> float:
    """
    Numba-accelerated core implementation of multipower variation.
    
    Args:
        returns: Array of returns
        powers: Array of powers for each lag
        
    Returns:
        Multipower variation
    """
    n = len(returns)
    m = len(powers)
    abs_returns = np.abs(returns)
    mpv_sum = 0.0
    
    # Compute multipower variation
    for i in range(n - m + 1):
        term = 1.0
        for j in range(m):
            term *= abs_returns[i + j] ** powers[j]
        mpv_sum += term
    
    # Compute scaling factor
    scaling = 1.0
    for p in powers:
        # mu_p = E[|Z|^p] where Z ~ N(0,1)
        mu_p = 2**(p/2) * np.exp(np.log(np.pi) / 2) / np.exp(np.log(2) / 2)
        scaling *= mu_p
    
    # Apply finite sample correction
    correction = n / (n - m + 1)
    
    return correction * mpv_sum / scaling


# ============================================================================
# Core Semivariance Functions
# ============================================================================

@njit(cache=True)
def _realized_semivariance_core(returns: np.ndarray) -> Tuple[float, float]:
    """
    Numba-accelerated core implementation of realized semivariance.
    
    Args:
        returns: Array of returns
        
    Returns:
        Tuple of (positive semivariance, negative semivariance)
    """
    positive_sum = 0.0
    negative_sum = 0.0
    
    for r in returns:
        if r > 0:
            positive_sum += r**2
        else:
            negative_sum += r**2
    
    return positive_sum, negative_sum


# ============================================================================
# Core Multiscale Functions
# ============================================================================

@njit(cache=True)
def _multiscale_variance_core(returns: np.ndarray, scales: np.ndarray, 
                             weights: np.ndarray) -> float:
    """
    Numba-accelerated core implementation of multiscale variance.
    
    Args:
        returns: Array of returns
        scales: Array of scales to use
        weights: Array of weights for each scale
        
    Returns:
        Multiscale variance estimate
    """
    n = len(returns)
    n_scales = len(scales)
    
    # Compute weighted sum of realized variances at different scales
    msv = 0.0
    
    for i in range(n_scales):
        scale = scales[i]
        weight = weights[i]
        
        # Skip if scale is too large
        if scale >= n:
            continue
        
        # Compute returns at this scale
        scale_returns = np.zeros(n - scale)
        for j in range(n - scale):
            for k in range(scale):
                scale_returns[j] += returns[j + k]
        
        # Compute realized variance at this scale
        rv_scale = np.sum(scale_returns**2) / (n - scale)
        
        # Add weighted contribution
        msv += weight * rv_scale
    
    return msv


# ============================================================================
# Core Two-Scale Functions
# ============================================================================

@njit(cache=True)
def _twoscale_variance_core(returns: np.ndarray, slow_scale: int) -> float:
    """
    Numba-accelerated core implementation of two-scale realized variance.
    
    Args:
        returns: Array of returns
        slow_scale: Slow scale for noise correction
        
    Returns:
        Two-scale realized variance estimate
    """
    n = len(returns)
    
    # Fast scale (original returns)
    rv_fast = np.sum(returns**2)
    
    # Slow scale
    rv_slow = 0.0
    
    # Compute returns at slow scale
    slow_returns = np.zeros(n - slow_scale + 1)
    for i in range(n - slow_scale + 1):
        for j in range(slow_scale):
            slow_returns[i] += returns[i + j]
    
    # Compute realized variance at slow scale
    rv_slow = np.sum(slow_returns**2) / (n - slow_scale + 1)
    
    # Adjust for noise
    adjustment = (slow_scale - 1) / slow_scale * rv_fast
    
    # Two-scale estimator
    tsrv = rv_slow - adjustment
    
    return tsrv


# ============================================================================
# Core Noise Estimation Functions
# ============================================================================

@njit(cache=True)
def _noise_variance_core(returns: np.ndarray) -> float:
    """
    Numba-accelerated core implementation of noise variance estimation.
    
    Args:
        returns: Array of returns
        
    Returns:
        Estimated noise variance
    """
    n = len(returns)
    
    # Compute first-order autocovariance
    acov = 0.0
    for i in range(n - 1):
        acov += returns[i] * returns[i + 1]
    acov /= (n - 1)
    
    # Noise variance is -0.5 * first-order autocovariance
    noise_var = -0.5 * acov
    
    # If noise_var is negative, use alternative method
    if noise_var <= 0:
        # Use 0.5 * mean squared returns
        noise_var = 0.5 * np.mean(returns**2)
    
    return noise_var


# ============================================================================
# Core Jump Detection Functions
# ============================================================================

@njit(cache=True)
def _detect_jumps_core(returns: np.ndarray, threshold_multiplier: float) -> Tuple[np.ndarray, float]:
    """
    Numba-accelerated core implementation of jump detection.
    
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
    bipower = np.zeros(n - 1)
    
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


# ============================================================================
# Core Range-Based Functions
# ============================================================================

@njit(cache=True)
def _realized_range_core(high_prices: np.ndarray, low_prices: np.ndarray) -> float:
    """
    Numba-accelerated core implementation of realized range.
    
    Args:
        high_prices: Array of high prices
        low_prices: Array of low prices
        
    Returns:
        Realized range estimate
    """
    n = len(high_prices)
    
    # Compute sum of squared log ranges
    range_sum = 0.0
    for i in range(n):
        log_range = np.log(high_prices[i] / low_prices[i])
        range_sum += log_range**2
    
    # Scale factor (4 * log(2)) for asymptotic consistency
    scale_factor = 4.0 * np.log(2.0)
    
    return range_sum / scale_factor


# ============================================================================
# Core Preaveraged Functions
# ============================================================================

@njit(cache=True)
def _preaveraged_returns_core(returns: np.ndarray, window_size: int) -> np.ndarray:
    """
    Numba-accelerated core implementation of preaveraged returns.
    
    Args:
        returns: Array of returns
        window_size: Window size for preaveraging
        
    Returns:
        Array of preaveraged returns
    """
    n = len(returns)
    n_preaveraged = n - window_size + 1
    preaveraged = np.zeros(n_preaveraged)
    
    # Compute preaveraged returns
    for i in range(n_preaveraged):
        for j in range(window_size):
            # Apply triangular kernel
            weight = j * (window_size - j) / (window_size**2)
            preaveraged[i] += weight * returns[i + j]
    
    return preaveraged


@njit(cache=True)
def _preaveraged_variance_core(returns: np.ndarray, window_size: int) -> float:
    """
    Numba-accelerated core implementation of preaveraged variance.
    
    Args:
        returns: Array of returns
        window_size: Window size for preaveraging
        
    Returns:
        Preaveraged variance estimate
    """
    # Compute preaveraged returns
    preaveraged = _preaveraged_returns_core(returns, window_size)
    
    # Compute variance of preaveraged returns
    pav = np.sum(preaveraged**2)
    
    # Scaling factor for asymptotic consistency
    theta = 1.0 / 12.0  # For triangular kernel
    scale_factor = 12.0 / (window_size * (1.0 - 2.0 * theta))
    
    return pav * scale_factor


# ============================================================================
# Core QMLE Functions
# ============================================================================

@njit(cache=True)
def _qmle_variance_core(returns: np.ndarray, noise_var: float) -> float:
    """
    Numba-accelerated core implementation of quasi-maximum likelihood variance.
    
    Args:
        returns: Array of returns
        noise_var: Estimated noise variance
        
    Returns:
        QMLE variance estimate
    """
    n = len(returns)
    
    # Compute realized variance
    rv = np.sum(returns**2)
    
    # Adjust for noise
    qmle = rv - 2 * n * noise_var
    
    # Ensure non-negativity
    if qmle < 0:
        qmle = 0.0
    
    return qmle


# ============================================================================
# Core Refresh Time Functions
# ============================================================================

@njit(cache=True)
def _refresh_time_core(times_list: List[np.ndarray]) -> np.ndarray:
    """
    Numba-accelerated core implementation of refresh time algorithm.
    
    Args:
        times_list: List of time arrays for each asset
        
    Returns:
        Array of refresh times
    """
    n_assets = len(times_list)
    
    # Find the maximum starting time across all assets
    max_start = 0.0
    for i in range(n_assets):
        if times_list[i][0] > max_start:
            max_start = times_list[i][0]
    
    # Initialize refresh times with the maximum starting time
    refresh_times = [max_start]
    
    # Find indices of the first observation after max_start for each asset
    indices = np.zeros(n_assets, dtype=np.int64)
    for i in range(n_assets):
        times = times_list[i]
        idx = 0
        while idx < len(times) and times[idx] < max_start:
            idx += 1
        indices[i] = idx
    
    # Main refresh time algorithm
    while True:
        # Check if any asset has reached the end of its time series
        end_reached = False
        for i in range(n_assets):
            if indices[i] >= len(times_list[i]):
                end_reached = True
                break
        
        if end_reached:
            break
        
        # Find the maximum time among current indices
        max_time = 0.0
        for i in range(n_assets):
            time_i = times_list[i][indices[i]]
            if time_i > max_time:
                max_time = time_i
        
        # Add the maximum time to refresh times
        refresh_times.append(max_time)
        
        # Update indices to the next observation after max_time
        for i in range(n_assets):
            times = times_list[i]
            idx = indices[i]
            while idx < len(times) and times[idx] <= max_time:
                idx += 1
            indices[i] = idx
    
    return np.array(refresh_times)


# ============================================================================
# Core Hayashi-Yoshida Functions
# ============================================================================

@njit(cache=True)
def _hayashi_yoshida_core(returns1: np.ndarray, times1: np.ndarray, 
                         returns2: np.ndarray, times2: np.ndarray) -> float:
    """
    Numba-accelerated core implementation of Hayashi-Yoshida covariance estimator.
    
    Args:
        returns1: Array of returns for first asset
        times1: Array of times for first asset
        returns2: Array of returns for second asset
        times2: Array of times for second asset
        
    Returns:
        Hayashi-Yoshida covariance estimate
    """
    n1 = len(returns1)
    n2 = len(returns2)
    
    # Compute end times for each return interval
    end_times1 = times1[1:]
    end_times2 = times2[1:]
    
    # Start times are the original times except the last one
    start_times1 = times1[:-1]
    start_times2 = times2[:-1]
    
    # Compute Hayashi-Yoshida estimator
    hy_cov = 0.0
    
    for i in range(n1 - 1):
        for j in range(n2 - 1):
            # Check if intervals overlap
            if (start_times1[i] < end_times2[j] and end_times1[i] > start_times2[j]):
                hy_cov += returns1[i] * returns2[j]
    
    return hy_cov


# ============================================================================
# Module Initialization
# ============================================================================

def _register_numba_functions() -> None:
    """
    Register Numba JIT-compiled functions for realized volatility computations.
    
    This function is called during module initialization to register
    performance-critical functions for JIT compilation if Numba is available.
    """
    if HAS_NUMBA:
        # The functions are already decorated with @jit/@njit, so no additional
        # registration is needed here. This function is kept for consistency
        # with the module structure and potential future enhancements.
        logger.debug("Realized volatility Numba JIT functions registered")
    else:
        logger.warning("Numba not available. Realized volatility computations will use pure NumPy implementations.")


# Initialize the module
_register_numba_functions()
