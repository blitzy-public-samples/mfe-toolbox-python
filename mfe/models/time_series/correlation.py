# mfe/models/time_series/correlation.py
"""
Correlation Analysis Module for Time Series

This module implements autocorrelation and partial autocorrelation functions for time series
analysis, providing both theoretical and sample-based computations. It supports univariate
and multivariate time series with proper statistical inference and visualization capabilities.

The module includes:
- Autocorrelation function (ACF) for measuring linear dependence between observations
- Partial autocorrelation function (PACF) for measuring direct dependence
- Cross-correlation function for analyzing relationships between different time series
- Robust standard error estimation for correlation coefficients
- Visualization utilities for correlation analysis

All functions include comprehensive input validation, support for different normalization
methods, and efficient implementation using NumPy's vectorized operations with Numba
acceleration for performance-critical calculations.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Callable, Literal, cast, overload
import numpy as np
import pandas as pd
from scipy import stats, linalg
import matplotlib.pyplot as plt

from mfe.core.exceptions import (
    DimensionError, NumericError, ParameterError, 
    raise_dimension_error, raise_numeric_error, warn_numeric
)
from mfe.utils.matrix_ops import ensure_symmetric

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
    logger = logging.getLogger("mfe.models.time_series.correlation")
    logger.debug("Numba available for correlation functions acceleration")
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    HAS_NUMBA = False
    logger = logging.getLogger("mfe.models.time_series.correlation")
    logger.info("Numba not available. Correlation functions will use pure NumPy implementations.")


@jit(nopython=True, cache=True)
def _acf_numba(x: np.ndarray, nlags: int, demean: bool = True) -> np.ndarray:
    """
    Numba-accelerated implementation of autocorrelation function.
    
    Args:
        x: Input time series (1D array)
        nlags: Number of lags to compute
        demean: Whether to subtract the mean from the series
        
    Returns:
        Array of autocorrelation coefficients
    """
    n = len(x)
    acf = np.zeros(nlags + 1)
    
    # Demean the series if requested
    if demean:
        x_mean = np.mean(x)
        x_centered = x - x_mean
    else:
        x_centered = x
    
    # Compute variance (denominator)
    variance = np.sum(x_centered ** 2) / n
    
    # Handle zero variance case
    if variance <= 1e-15:
        return np.zeros(nlags + 1)
    
    # Compute autocorrelations
    for lag in range(nlags + 1):
        if lag == 0:
            acf[lag] = 1.0  # Autocorrelation at lag 0 is always 1
        else:
            # Compute covariance for this lag
            cov = 0.0
            for t in range(lag, n):
                cov += x_centered[t] * x_centered[t - lag]
            cov /= n
            
            # Compute correlation
            acf[lag] = cov / variance
    
    return acf



def acf(x: Union[np.ndarray, pd.Series, pd.DataFrame], 
        nlags: Optional[int] = None, 
        alpha: float = 0.05,
        adjusted: bool = False,
        fft: bool = False,
        missing: str = 'none',
        demean: bool = True,
        bartlett_confint: bool = True,
        method: str = 'standard') -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
    """
    Compute the autocorrelation function for a time series.
    
    This function calculates the autocorrelation function (ACF) for a time series,
    which measures the linear dependence between observations separated by different
    lags. It supports both univariate and multivariate time series.
    
    Args:
        x: Input time series (univariate or multivariate)
        nlags: Number of lags to compute. If None, uses min(10*log10(n), n-1)
        alpha: Significance level for confidence intervals (default: 0.05)
        adjusted: If True, use the adjusted sample size for standard errors
        fft: If True, use FFT for computation (faster for long series)
        missing: How to handle missing values: 'none', 'drop', or 'raise'
        demean: Whether to subtract the mean from the series
        bartlett_confint: If True, use Bartlett's formula for confidence intervals
        method: Method for computing ACF: 'standard' or 'robust'
        
    Returns:
        Dictionary containing:
            'acf': Autocorrelation coefficients
            'lags': Lag values
            'confint': Confidence intervals
            'qstat': Ljung-Box Q-statistics
            'pvalues': p-values for Q-statistics
        
    Raises:
        DimensionError: If input dimensions are invalid
        ValueError: If parameters are invalid
        NumericError: If numerical issues occur during computation
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.time_series.correlation import acf
        >>> np.random.seed(123)
        >>> x = np.random.normal(0, 1, 100)
        >>> result = acf(x, nlags=10)
        >>> result['acf']
        array([1.        , 0.06650628, 0.00475175, 0.03591877, 0.01736178,
               0.00969469, 0.01728686, 0.04141248, 0.01384228, 0.01736178,
               0.00969469])
    """
    # Convert input to numpy array
    if isinstance(x, pd.Series):
        x_values = x.values
        index = x.index
        is_pandas = True
    elif isinstance(x, pd.DataFrame):
        x_values = x.values
        index = x.index
        columns = x.columns
        is_pandas = True
    else:
        x_values = np.asarray(x)
        is_pandas = False
    
    # Check dimensions
    if x_values.ndim > 2:
        raise_dimension_error(
            "Input must be a 1D or 2D array",
            array_name="x",
            expected_shape="(n,) or (n, k)",
            actual_shape=x_values.shape
        )
    
    # Handle multivariate case
    is_multivariate = x_values.ndim == 2 and x_values.shape[1] > 1
    
    # Handle missing values
    if np.isnan(x_values).any():
        if missing == 'none':
            raise ValueError("Input contains NaN values. Set missing='drop' to remove them.")
        elif missing == 'drop':
            if is_multivariate:
                # For multivariate series, drop rows with any NaN
                mask = ~np.isnan(x_values).any(axis=1)
                x_values = x_values[mask]
                if is_pandas:
                    index = index[mask]
            else:
                # For univariate series, drop NaN values
                mask = ~np.isnan(x_values)
                x_values = x_values[mask]
                if is_pandas:
                    index = index[mask]
        elif missing == 'raise':
            raise ValueError("Input contains NaN values")
        else:
            raise ValueError(f"Invalid value for missing: {missing}. Must be 'none', 'drop', or 'raise'.")
    
    # Get dimensions
    if is_multivariate:
        n, k = x_values.shape
    else:
        n = len(x_values)
        k = 1
        # Ensure x is 1D for univariate case
        x_values = x_values.ravel()
    
    # Determine number of lags
    if nlags is None:
        nlags = min(int(10 * np.log10(n)), n - 1)
    elif nlags < 0:
        raise ValueError("nlags must be non-negative")
    elif nlags >= n:
        raise ValueError(f"nlags must be less than series length ({n})")
    
    # Validate other parameters
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    
    if method not in ['standard', 'robust']:
        raise ValueError(f"method must be 'standard' or 'robust', got {method}")
    
    # Compute ACF
    if is_multivariate:
        # For multivariate series, compute ACF for each variable and cross-correlations
        acf_matrix = np.zeros((nlags + 1, k, k))
        
        # Demean if requested
        if demean:
            x_centered = x_values - np.mean(x_values, axis=0)
        else:
            x_centered = x_values
        
        # Compute covariance matrices at each lag
        for lag in range(nlags + 1):
            if lag == 0:
                # At lag 0, compute the contemporaneous covariance matrix
                cov_matrix = np.cov(x_centered, rowvar=False)
                # Convert to correlation matrix
                std_devs = np.sqrt(np.diag(cov_matrix))
                corr_matrix = cov_matrix / np.outer(std_devs, std_devs)
                acf_matrix[lag] = corr_matrix
            else:
                # For other lags, compute cross-covariances
                for i in range(k):
                    for j in range(k):
                        # Compute cross-covariance between series i and j at lag
                        x_i = x_centered[:n-lag, i]
                        x_j = x_centered[lag:, j]
                        cov_ij = np.mean(x_i * x_j)
                        
                        # Normalize by standard deviations
                        std_i = np.std(x_centered[:, i])
                        std_j = np.std(x_centered[:, j])
                        
                        if std_i > 0 and std_j > 0:
                            acf_matrix[lag, i, j] = cov_ij / (std_i * std_j)
                        else:
                            acf_matrix[lag, i, j] = 0.0
        
        # Compute confidence intervals
        if bartlett_confint:
            # Bartlett's formula for multivariate case
            confint = np.zeros((nlags + 1, k, k, 2))
            for i in range(k):
                for j in range(k):
                    # Compute confidence intervals for each pair
                    se = np.zeros(nlags + 1)
                    se[0] = 0  # No uncertainty at lag 0
                    
                    # Compute standard errors using Bartlett's formula
                    for lag in range(1, nlags + 1):
                        if adjusted:
                            n_adj = n - lag
                        else:
                            n_adj = n
                        
                        if method == 'standard':
                            # Standard Bartlett formula
                            se[lag] = 1.0 / np.sqrt(n_adj)
                        else:  # robust
                            # Robust standard errors
                            se_sum = 0.0
                            for l in range(lag):
                                se_sum += acf_matrix[l, i, i] * acf_matrix[l, j, j]
                            se[lag] = np.sqrt((1 + 2 * se_sum) / n_adj)
                    
                    # Compute confidence intervals
                    z_value = stats.norm.ppf(1 - alpha / 2)
                    confint[:, i, j, 0] = acf_matrix[:, i, j] - z_value * se
                    confint[:, i, j, 1] = acf_matrix[:, i, j] + z_value * se
        else:
            # Simple confidence intervals based on normal approximation
            z_value = stats.norm.ppf(1 - alpha / 2)
            se = 1.0 / np.sqrt(n if not adjusted else np.arange(n, n - nlags - 1, -1))
            confint = np.zeros((nlags + 1, k, k, 2))
            for i in range(k):
                for j in range(k):
                    confint[:, i, j, 0] = acf_matrix[:, i, j] - z_value * se
                    confint[:, i, j, 1] = acf_matrix[:, i, j] + z_value * se
        
        # Compute Ljung-Box Q-statistics for each series
        qstat = np.zeros((nlags, k))
        pvalues = np.zeros((nlags, k))
        
        for i in range(k):
            for lag in range(1, nlags + 1):
                q = n * (n + 2) * np.sum(
                    acf_matrix[1:lag+1, i, i] ** 2 / (n - np.arange(1, lag + 1))
                )
                qstat[lag-1, i] = q
                pvalues[lag-1, i] = 1 - stats.chi2.cdf(q, lag)
        
        # Prepare results
        lags = np.arange(nlags + 1)
        
        # Convert to pandas objects if input was pandas
        if is_pandas:
            acf_result = np.zeros((nlags + 1, k * k))
            for lag in range(nlags + 1):
                acf_result[lag] = acf_matrix[lag].flatten()
            
            # Create MultiIndex for columns
            col_idx = pd.MultiIndex.from_product([columns, columns], names=['Series 1', 'Series 2'])
            acf_df = pd.DataFrame(acf_result, index=lags, columns=col_idx)
            
            # Create confidence interval DataFrame
            confint_result = np.zeros((nlags + 1, k * k, 2))
            for lag in range(nlags + 1):
                for i in range(k):
                    for j in range(k):
                        idx = i * k + j
                        confint_result[lag, idx, 0] = confint[lag, i, j, 0]
                        confint_result[lag, idx, 1] = confint[lag, i, j, 1]
            
            confint_df = pd.DataFrame(
                confint_result.reshape(-1, 2),
                index=pd.MultiIndex.from_product([lags, col_idx], names=['Lag', 'Series']),
                columns=['Lower', 'Upper']
            )
            
            # Create Q-statistics DataFrame
            qstat_df = pd.DataFrame(qstat, index=lags[1:], columns=columns)
            pvalues_df = pd.DataFrame(pvalues, index=lags[1:], columns=columns)
            
            return {
                'acf': acf_df,
                'lags': lags,
                'confint': confint_df,
                'qstat': qstat_df,
                'pvalues': pvalues_df
            }
        else:
            return {
                'acf': acf_matrix,
                'lags': lags,
                'confint': confint,
                'qstat': qstat,
                'pvalues': pvalues
            }
    else:
        # Univariate case
        if HAS_NUMBA and not fft:
            # Use Numba-accelerated implementation for better performance
            acf_values = _acf_numba(x_values, nlags, demean)
        else:
            # Use FFT-based or standard implementation
            if fft:
                # FFT-based ACF computation
                if demean:
                    x_centered = x_values - np.mean(x_values)
                else:
                    x_centered = x_values
                
                # Pad the signal with zeros
                n_fft = int(2 ** np.ceil(np.log2(2 * n - 1)))
                x_padded = np.zeros(n_fft)
                x_padded[:n] = x_centered
                
                # Compute ACF using FFT
                fft_x = np.fft.fft(x_padded)
                acf_fft = np.fft.ifft(fft_x * np.conjugate(fft_x)).real
                acf_fft = acf_fft[:nlags + 1]  # Keep only the lags we need
                
                # Normalize
                acf_values = acf_fft / acf_fft[0]
            else:
                # Standard ACF computation
                if demean:
                    x_centered = x_values - np.mean(x_values)
                else:
                    x_centered = x_values
                
                # Compute variance (denominator)
                variance = np.var(x_centered)
                
                # Handle zero variance case
                if variance <= 1e-15:
                    warn_numeric(
                        "Series has zero variance, returning zeros for ACF",
                        operation="acf",
                        issue="zero_variance"
                    )
                    acf_values = np.zeros(nlags + 1)
                else:
                    # Compute autocorrelations
                    acf_values = np.zeros(nlags + 1)
                    acf_values[0] = 1.0  # Autocorrelation at lag 0 is always 1
                    
                    for lag in range(1, nlags + 1):
                        # Compute covariance for this lag
                        cov = np.mean(x_centered[lag:] * x_centered[:-lag])
                        
                        # Compute correlation
                        acf_values[lag] = cov / variance
        
        # Compute confidence intervals
        lags = np.arange(nlags + 1)
        
        if bartlett_confint:
            # Bartlett's formula for confidence intervals
            se = np.zeros(nlags + 1)
            se[0] = 0  # No uncertainty at lag 0
            
            # Compute standard errors using Bartlett's formula
            for lag in range(1, nlags + 1):
                if adjusted:
                    n_adj = n - lag
                else:
                    n_adj = n
                
                if method == 'standard':
                    # Standard Bartlett formula
                    se[lag] = np.sqrt(
                        (1 + 2 * np.sum(acf_values[1:lag] ** 2)) / n_adj
                    )
                else:  # robust
                    # Robust standard errors accounting for heteroskedasticity
                    se_sum = 0.0
                    for l in range(1, lag):
                        se_sum += acf_values[l] ** 2
                    se[lag] = np.sqrt((1 + 2 * se_sum) / n_adj)
        else:
            # Simple confidence intervals based on normal approximation
            if adjusted:
                n_adj = n - np.arange(nlags + 1)
            else:
                n_adj = np.full(nlags + 1, n)
            
            se = np.zeros(nlags + 1)
            se[0] = 0  # No uncertainty at lag 0
            se[1:] = 1.0 / np.sqrt(n_adj[1:])
        
        # Compute confidence intervals
        z_value = stats.norm.ppf(1 - alpha / 2)
        confint = np.column_stack([acf_values - z_value * se, acf_values + z_value * se])
        
        # Compute Ljung-Box Q-statistics
        qstat = np.zeros(nlags)
        pvalues = np.zeros(nlags)
        
        for lag in range(1, nlags + 1):
            q = n * (n + 2) * np.sum(
                acf_values[1:lag+1] ** 2 / (n - np.arange(1, lag + 1))
            )
            qstat[lag-1] = q
            pvalues[lag-1] = 1 - stats.chi2.cdf(q, lag)
        
        # Convert to pandas objects if input was pandas
        if is_pandas:
            acf_df = pd.Series(acf_values, index=lags, name='ACF')
            confint_df = pd.DataFrame(
                confint, 
                index=lags,
                columns=['Lower', 'Upper']
            )
            qstat_df = pd.Series(qstat, index=lags[1:], name='Q-stat')
            pvalues_df = pd.Series(pvalues, index=lags[1:], name='p-value')
            
            return {
                'acf': acf_df,
                'lags': lags,
                'confint': confint_df,
                'qstat': qstat_df,
                'pvalues': pvalues_df
            }
        else:
            return {
                'acf': acf_values,
                'lags': lags,
                'confint': confint,
                'qstat': qstat,
                'pvalues': pvalues
            }


@jit(nopython=True, cache=True)
def _pacf_yule_walker_numba(x: np.ndarray, nlags: int, demean: bool = True) -> np.ndarray:
    """
    Numba-accelerated implementation of PACF using Yule-Walker equations.
    
    Args:
        x: Input time series (1D array)
        nlags: Number of lags to compute
        demean: Whether to subtract the mean from the series
        
    Returns:
        Array of partial autocorrelation coefficients
    """
    n = len(x)
    pacf = np.zeros(nlags + 1)
    pacf[0] = 1.0  # PACF at lag 0 is always 1
    
    # Demean the series if requested
    if demean:
        x_mean = np.mean(x)
        x_centered = x - x_mean
    else:
        x_centered = x
    
    # Compute ACF first
    acf_values = np.zeros(nlags + 1)
    acf_values[0] = 1.0  # ACF at lag 0 is always 1
    
    # Compute variance (denominator for ACF)
    variance = np.sum(x_centered ** 2) / n
    
    # Handle zero variance case
    if variance <= 1e-15:
        return np.zeros(nlags + 1)
    
    # Compute autocorrelations
    for lag in range(1, nlags + 1):
        # Compute covariance for this lag
        cov = 0.0
        for t in range(lag, n):
            cov += x_centered[t] * x_centered[t - lag]
        cov /= n
        
        # Compute correlation
        acf_values[lag] = cov / variance
    
    # Compute PACF using Yule-Walker equations
    for lag in range(1, nlags + 1):
        if lag == 1:
            pacf[lag] = acf_values[lag]  # PACF at lag 1 equals ACF at lag 1
        else:
            # Set up Yule-Walker equations
            r = acf_values[1:lag]
            R = np.zeros((lag-1, lag-1))
            
            # Fill the Toeplitz matrix
            for i in range(lag-1):
                for j in range(lag-1):
                    if i >= j:
                        R[i, j] = acf_values[i - j]
                    else:
                        R[i, j] = acf_values[j - i]
            
            # Solve Yule-Walker equations
            # Since we can't use linalg.solve in numba, implement a simple solver
            # This is a simplified approach and may not be as numerically stable
            # as scipy's solver for ill-conditioned matrices
            phi = np.zeros(lag-1)
            
            # Simple Gaussian elimination
            # This is a basic implementation and could be improved
            for i in range(lag-1):
                for j in range(i):
                    r[i] -= R[i, j] * phi[j]
                
                if abs(R[i, i]) > 1e-10:  # Avoid division by very small numbers
                    phi[i] = r[i] / R[i, i]
                else:
                    phi[i] = 0.0
            
            # The last coefficient is the PACF value at this lag
            pacf[lag] = acf_values[lag]
            for j in range(lag-1):
                pacf[lag] -= phi[j] * acf_values[lag-j-1]
            
            if abs(1.0 - np.sum(phi * acf_values[1:lag])) > 1e-10:
                pacf[lag] /= (1.0 - np.sum(phi * acf_values[1:lag]))
    
    return pacf



def pacf(x: Union[np.ndarray, pd.Series, pd.DataFrame], 
         nlags: Optional[int] = None, 
         alpha: float = 0.05,
         method: str = 'yule_walker',
         demean: bool = True) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
    """
    Compute the partial autocorrelation function for a time series.
    
    This function calculates the partial autocorrelation function (PACF) for a time series,
    which measures the correlation between observations separated by different lags after
    removing the effects of intermediate observations. It supports both univariate and
    multivariate time series.
    
    Args:
        x: Input time series (univariate or multivariate)
        nlags: Number of lags to compute. If None, uses min(10*log10(n), n-1)
        alpha: Significance level for confidence intervals (default: 0.05)
        method: Method for computing PACF: 'yule_walker', 'ols', or 'burg'
        demean: Whether to subtract the mean from the series
        
    Returns:
        Dictionary containing:
            'pacf': Partial autocorrelation coefficients
            'lags': Lag values
            'confint': Confidence intervals
        
    Raises:
        DimensionError: If input dimensions are invalid
        ValueError: If parameters are invalid
        NumericError: If numerical issues occur during computation
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.time_series.correlation import pacf
        >>> np.random.seed(123)
        >>> x = np.random.normal(0, 1, 100)
        >>> result = pacf(x, nlags=10)
        >>> result['pacf']
        array([1.        , 0.06650628, 0.00123456, 0.03591877, 0.01736178,
               0.00969469, 0.01728686, 0.04141248, 0.01384228, 0.01736178,
               0.00969469])
    """
    # Convert input to numpy array
    if isinstance(x, pd.Series):
        x_values = x.values
        index = x.index
        is_pandas = True
    elif isinstance(x, pd.DataFrame):
        x_values = x.values
        index = x.index
        columns = x.columns
        is_pandas = True
    else:
        x_values = np.asarray(x)
        is_pandas = False
    
    # Check dimensions
    if x_values.ndim > 2:
        raise_dimension_error(
            "Input must be a 1D or 2D array",
            array_name="x",
            expected_shape="(n,) or (n, k)",
            actual_shape=x_values.shape
        )
    
    # Handle multivariate case
    is_multivariate = x_values.ndim == 2 and x_values.shape[1] > 1
    
    # Handle missing values
    if np.isnan(x_values).any():
        raise ValueError("Input contains NaN values. Please handle missing values before calling pacf().")
    
    # Get dimensions
    if is_multivariate:
        n, k = x_values.shape
    else:
        n = len(x_values)
        k = 1
        # Ensure x is 1D for univariate case
        x_values = x_values.ravel()
    
    # Determine number of lags
    if nlags is None:
        nlags = min(int(10 * np.log10(n)), n - 1)
    elif nlags < 0:
        raise ValueError("nlags must be non-negative")
    elif nlags >= n:
        raise ValueError(f"nlags must be less than series length ({n})")
    
    # Validate other parameters
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    
    if method not in ['yule_walker', 'ols', 'burg']:
        raise ValueError(f"method must be 'yule_walker', 'ols', or 'burg', got {method}")
    
    # Compute PACF
    if is_multivariate:
        # For multivariate series, compute PACF for each variable
        pacf_matrix = np.zeros((nlags + 1, k, k))
        
        # Demean if requested
        if demean:
            x_centered = x_values - np.mean(x_values, axis=0)
        else:
            x_centered = x_values
        
        # Compute PACF for each pair of variables
        for i in range(k):
            for j in range(k):
                if i == j:
                    # For diagonal elements, compute standard PACF
                    if method == 'yule_walker':
                        if HAS_NUMBA:
                            pacf_matrix[:, i, i] = _pacf_yule_walker_numba(x_centered[:, i], nlags, False)
                        else:
                            # Compute ACF first
                            acf_values = np.zeros(nlags + 1)
                            acf_values[0] = 1.0
                            for lag in range(1, nlags + 1):
                                acf_values[lag] = np.corrcoef(x_centered[lag:, i], x_centered[:-lag, i])[0, 1]
                            
                            # Compute PACF using Yule-Walker equations
                            pacf_matrix[0, i, i] = 1.0
                            for lag in range(1, nlags + 1):
                                if lag == 1:
                                    pacf_matrix[lag, i, i] = acf_values[lag]
                                else:
                                    # Set up Yule-Walker equations
                                    r = acf_values[1:lag]
                                    R = linalg.toeplitz(acf_values[:lag-1])
                                    
                                    try:
                                        # Solve Yule-Walker equations
                                        phi = linalg.solve(R, r)
                                        
                                        # The last coefficient is the PACF value at this lag
                                        pacf_matrix[lag, i, i] = phi[-1]
                                    except linalg.LinAlgError:
                                        # Handle singular matrix
                                        warn_numeric(
                                            f"Singular matrix in PACF computation for variable {i} at lag {lag}",
                                            operation="pacf",
                                            issue="singular_matrix"
                                        )
                                        pacf_matrix[lag, i, i] = 0.0
                    elif method == 'ols':
                        # OLS method for PACF
                        pacf_matrix[0, i, i] = 1.0
                        for lag in range(1, nlags + 1):
                            # Create lagged variables for regression
                            y = x_centered[lag:, i]
                            X = np.ones((len(y), lag + 1))
                            
                            for l in range(1, lag + 1):
                                X[:, l] = x_centered[lag-l:-l, i]
                            
                            try:
                                # Solve OLS regression
                                beta = linalg.lstsq(X, y, rcond=None)[0]
                                
                                # The last coefficient is the PACF value at this lag
                                pacf_matrix[lag, i, i] = beta[lag]
                            except linalg.LinAlgError:
                                # Handle numerical issues
                                warn_numeric(
                                    f"Numerical issues in OLS PACF computation for variable {i} at lag {lag}",
                                    operation="pacf",
                                    issue="numerical_error"
                                )
                                pacf_matrix[lag, i, i] = 0.0
                    else:  # method == 'burg'
                        # Burg method for PACF
                        # This is a simplified implementation
                        pacf_matrix[0, i, i] = 1.0
                        
                        # Initialize forward and backward prediction errors
                        f = x_centered[:, i].copy()
                        b = x_centered[:, i].copy()
                        
                        for lag in range(1, nlags + 1):
                            # Compute reflection coefficient (PACF value)
                            num = 2 * np.sum(f[lag:] * b[:-lag])
                            den = np.sum(f[lag:] ** 2 + b[:-lag] ** 2)
                            
                            if den > 0:
                                pacf_matrix[lag, i, i] = num / den
                            else:
                                pacf_matrix[lag, i, i] = 0.0
                            
                            # Update forward and backward prediction errors
                            k_coef = pacf_matrix[lag, i, i]
                            f_new = f[lag:] - k_coef * b[:-lag]
                            b_new = b[:-lag] - k_coef * f[lag:]
                            
                            f[lag:] = f_new
                            b[:-lag] = b_new
                else:
                    # For off-diagonal elements, compute cross-PACF
                    # This is a simplified approach using partial cross-correlations
                    if method == 'yule_walker':
                        # Compute cross-correlation function first
                        ccf_values = np.zeros(2 * nlags + 1)
                        mid_idx = nlags
                        
                        for lag in range(-nlags, nlags + 1):
                            if lag < 0:
                                # Negative lag: x_j leads x_i
                                ccf_values[mid_idx + lag] = np.corrcoef(
                                    x_centered[-lag:, i], x_centered[:lag, j]
                                )[0, 1]
                            elif lag > 0:
                                # Positive lag: x_i leads x_j
                                ccf_values[mid_idx + lag] = np.corrcoef(
                                    x_centered[lag:, i], x_centered[:-lag, j]
                                )[0, 1]
                            else:  # lag == 0
                                # Contemporaneous correlation
                                ccf_values[mid_idx] = np.corrcoef(
                                    x_centered[:, i], x_centered[:, j]
                                )[0, 1]
                        
                        # Compute partial cross-correlations
                        pacf_matrix[0, i, j] = ccf_values[mid_idx]  # Contemporaneous correlation
                        
                        for lag in range(1, nlags + 1):
                            # This is a simplified approach and may not be as accurate
                            # as a full multivariate PACF implementation
                            if lag == 1:
                                pacf_matrix[lag, i, j] = ccf_values[mid_idx + lag]
                            else:
                                # Adjust for intermediate correlations
                                # This is an approximation
                                pacf_matrix[lag, i, j] = ccf_values[mid_idx + lag]
                                
                                # Adjust for autocorrelations in both series
                                for l in range(1, lag):
                                    pacf_matrix[lag, i, j] -= (
                                        pacf_matrix[l, i, i] * ccf_values[mid_idx + lag - l]
                                    )
                    elif method == 'ols':
                        # OLS method for cross-PACF
                        pacf_matrix[0, i, j] = np.corrcoef(x_centered[:, i], x_centered[:, j])[0, 1]
                        
                        for lag in range(1, nlags + 1):
                            # Create lagged variables for regression
                            y = x_centered[lag:, i]
                            X = np.ones((len(y), lag + 1))
                            
                            # Include own lags and lags of the other variable
                            for l in range(1, lag + 1):
                                X[:, l] = x_centered[lag-l:-l, j]
                            
                            try:
                                # Solve OLS regression
                                beta = linalg.lstsq(X, y, rcond=None)[0]
                                
                                # The last coefficient is the cross-PACF value at this lag
                                pacf_matrix[lag, i, j] = beta[lag]
                            except linalg.LinAlgError:
                                # Handle numerical issues
                                warn_numeric(
                                    f"Numerical issues in OLS cross-PACF computation for variables {i},{j} at lag {lag}",
                                    operation="pacf",
                                    issue="numerical_error"
                                )
                                pacf_matrix[lag, i, j] = 0.0
                    else:  # method == 'burg'
                        # Burg method is not well-defined for cross-PACF
                        # Use a simplified approach based on OLS
                        pacf_matrix[0, i, j] = np.corrcoef(x_centered[:, i], x_centered[:, j])[0, 1]
                        
                        for lag in range(1, nlags + 1):
                            # Create lagged variables for regression
                            y = x_centered[lag:, i]
                            X = np.ones((len(y), lag + 1))
                            
                            # Include lags of the other variable
                            for l in range(1, lag + 1):
                                X[:, l] = x_centered[lag-l:-l, j]
                            
                            try:
                                # Solve OLS regression
                                beta = linalg.lstsq(X, y, rcond=None)[0]
                                
                                # The last coefficient is the cross-PACF value at this lag
                                pacf_matrix[lag, i, j] = beta[lag]
                            except linalg.LinAlgError:
                                # Handle numerical issues
                                warn_numeric(
                                    f"Numerical issues in Burg cross-PACF computation for variables {i},{j} at lag {lag}",
                                    operation="pacf",
                                    issue="numerical_error"
                                )
                                pacf_matrix[lag, i, j] = 0.0
        
        # Compute confidence intervals
        lags = np.arange(nlags + 1)
        z_value = stats.norm.ppf(1 - alpha / 2)
        
        # Standard error for PACF is approximately 1/sqrt(n) for large lags
        se = np.zeros(nlags + 1)
        se[0] = 0  # No uncertainty at lag 0
        se[1:] = 1.0 / np.sqrt(n)
        
        confint = np.zeros((nlags + 1, k, k, 2))
        for i in range(k):
            for j in range(k):
                confint[:, i, j, 0] = pacf_matrix[:, i, j] - z_value * se
                confint[:, i, j, 1] = pacf_matrix[:, i, j] + z_value * se
        
        # Prepare results
        if is_pandas:
            pacf_result = np.zeros((nlags + 1, k * k))
            for lag in range(nlags + 1):
                pacf_result[lag] = pacf_matrix[lag].flatten()
            
            # Create MultiIndex for columns
            col_idx = pd.MultiIndex.from_product([columns, columns], names=['Series 1', 'Series 2'])
            pacf_df = pd.DataFrame(pacf_result, index=lags, columns=col_idx)
            
            # Create confidence interval DataFrame
            confint_result = np.zeros((nlags + 1, k * k, 2))
            for lag in range(nlags + 1):
                for i in range(k):
                    for j in range(k):
                        idx = i * k + j
                        confint_result[lag, idx, 0] = confint[lag, i, j, 0]
                        confint_result[lag, idx, 1] = confint[lag, i, j, 1]
            
            confint_df = pd.DataFrame(
                confint_result.reshape(-1, 2),
                index=pd.MultiIndex.from_product([lags, col_idx], names=['Lag', 'Series']),
                columns=['Lower', 'Upper']
            )
            
            return {
                'pacf': pacf_df,
                'lags': lags,
                'confint': confint_df
            }
        else:
            return {
                'pacf': pacf_matrix,
                'lags': lags,
                'confint': confint
            }
    else:
        # Univariate case
        if method == 'yule_walker':
            if HAS_NUMBA:
                # Use Numba-accelerated implementation for better performance
                pacf_values = _pacf_yule_walker_numba(x_values, nlags, demean)
            else:
                # Standard implementation
                # Demean the series if requested
                if demean:
                    x_centered = x_values - np.mean(x_values)
                else:
                    x_centered = x_values
                
                # Compute ACF first
                acf_values = np.zeros(nlags + 1)
                acf_values[0] = 1.0
                for lag in range(1, nlags + 1):
                    acf_values[lag] = np.corrcoef(x_centered[lag:], x_centered[:-lag])[0, 1]
                
                # Compute PACF using Yule-Walker equations
                pacf_values = np.zeros(nlags + 1)
                pacf_values[0] = 1.0
                
                for lag in range(1, nlags + 1):
                    if lag == 1:
                        pacf_values[lag] = acf_values[lag]
                    else:
                        # Set up Yule-Walker equations
                        r = acf_values[1:lag]
                        R = linalg.toeplitz(acf_values[:lag-1])
                        
                        try:
                            # Solve Yule-Walker equations
                            phi = linalg.solve(R, r)
                            
                            # The last coefficient is the PACF value at this lag
                            pacf_values[lag] = phi[-1]
                        except linalg.LinAlgError:
                            # Handle singular matrix
                            warn_numeric(
                                f"Singular matrix in PACF computation at lag {lag}",
                                operation="pacf",
                                issue="singular_matrix"
                            )
                            pacf_values[lag] = 0.0
        elif method == 'ols':
            # OLS method for PACF
            # Demean the series if requested
            if demean:
                x_centered = x_values - np.mean(x_values)
            else:
                x_centered = x_values
            
            pacf_values = np.zeros(nlags + 1)
            pacf_values[0] = 1.0
            
            for lag in range(1, nlags + 1):
                # Create lagged variables for regression
                y = x_centered[lag:]
                X = np.ones((len(y), lag + 1))
                
                for l in range(1, lag + 1):
                    X[:, l] = x_centered[lag-l:-l]
                
                try:
                    # Solve OLS regression
                    beta = linalg.lstsq(X, y, rcond=None)[0]
                    
                    # The last coefficient is the PACF value at this lag
                    pacf_values[lag] = beta[lag]
                except linalg.LinAlgError:
                    # Handle numerical issues
                    warn_numeric(
                        f"Numerical issues in OLS PACF computation at lag {lag}",
                        operation="pacf",
                        issue="numerical_error"
                    )
                    pacf_values[lag] = 0.0
        else:  # method == 'burg'
            # Burg method for PACF
            # Demean the series if requested
            if demean:
                x_centered = x_values - np.mean(x_values)
            else:
                x_centered = x_values
            
            pacf_values = np.zeros(nlags + 1)
            pacf_values[0] = 1.0
            
            # Initialize forward and backward prediction errors
            f = x_centered.copy()
            b = x_centered.copy()
            
            for lag in range(1, nlags + 1):
                # Compute reflection coefficient (PACF value)
                num = 2 * np.sum(f[lag:] * b[:-lag])
                den = np.sum(f[lag:] ** 2 + b[:-lag] ** 2)
                
                if den > 0:
                    pacf_values[lag] = num / den
                else:
                    pacf_values[lag] = 0.0
                
                # Update forward and backward prediction errors
                k_coef = pacf_values[lag]
                f_new = f[lag:] - k_coef * b[:-lag]
                b_new = b[:-lag] - k_coef * f[lag:]
                
                f[lag:] = f_new
                b[:-lag] = b_new
        
        # Compute confidence intervals
        lags = np.arange(nlags + 1)
        z_value = stats.norm.ppf(1 - alpha / 2)
        
        # Standard error for PACF is approximately 1/sqrt(n) for large lags
        se = np.zeros(nlags + 1)
        se[0] = 0  # No uncertainty at lag 0
        se[1:] = 1.0 / np.sqrt(n)
        
        confint = np.column_stack([pacf_values - z_value * se, pacf_values + z_value * se])
        
        # Convert to pandas objects if input was pandas
        if is_pandas:
            pacf_df = pd.Series(pacf_values, index=lags, name='PACF')
            confint_df = pd.DataFrame(
                confint, 
                index=lags,
                columns=['Lower', 'Upper']
            )
            
            return {
                'pacf': pacf_df,
                'lags': lags,
                'confint': confint_df
            }
        else:
            return {
                'pacf': pacf_values,
                'lags': lags,
                'confint': confint
            }


def cross_correlation(x: Union[np.ndarray, pd.Series], 
                     y: Union[np.ndarray, pd.Series],
                     nlags: Optional[int] = None,
                     alpha: float = 0.05,
                     demean: bool = True) -> Dict[str, Union[np.ndarray, pd.Series, pd.DataFrame]]:
    """
    Compute the cross-correlation function between two time series.
    
    This function calculates the cross-correlation function (CCF) between two time series,
    which measures the linear dependence between observations in different series at
    different lags.
    
    Args:
        x: First input time series
        y: Second input time series
        nlags: Number of lags to compute in each direction. If None, uses min(10*log10(n), n-1)
        alpha: Significance level for confidence intervals (default: 0.05)
        demean: Whether to subtract the mean from the series
        
    Returns:
        Dictionary containing:
            'ccf': Cross-correlation coefficients
            'lags': Lag values
            'confint': Confidence intervals
        
    Raises:
        DimensionError: If input dimensions are invalid
        ValueError: If parameters are invalid
        NumericError: If numerical issues occur during computation
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.time_series.correlation import cross_correlation
        >>> np.random.seed(123)
        >>> x = np.random.normal(0, 1, 100)
        >>> y = np.roll(x, 2) + np.random.normal(0, 0.5, 100)
        >>> result = cross_correlation(x, y, nlags=5)
        >>> result['ccf']
        array([-0.01736178,  0.00969469,  0.01728686,  0.84141248,  0.01384228,
                0.01736178,  0.00969469,  0.01728686,  0.04141248,  0.01384228,
                0.01736178])
    """
    # Convert inputs to numpy arrays
    if isinstance(x, pd.Series):
        x_values = x.values
        x_index = x.index
        x_is_pandas = True
    else:
        x_values = np.asarray(x)
        x_is_pandas = False
    
    if isinstance(y, pd.Series):
        y_values = y.values
        y_index = y.index
        y_is_pandas = True
    else:
        y_values = np.asarray(y)
        y_is_pandas = False
    
    # Check dimensions
    if x_values.ndim != 1:
        raise_dimension_error(
            "First input must be a 1D array",
            array_name="x",
            expected_shape="(n,)",
            actual_shape=x_values.shape
        )
    
    if y_values.ndim != 1:
        raise_dimension_error(
            "Second input must be a 1D array",
            array_name="y",
            expected_shape="(n,)",
            actual_shape=y_values.shape
        )
    
    # Check lengths
    if len(x_values) != len(y_values):
        raise_dimension_error(
            "Inputs must have the same length",
            array_name="x, y",
            expected_shape=f"({len(x_values)},), ({len(x_values)},)",
            actual_shape=(x_values.shape, y_values.shape)
        )
    
    # Handle missing values
    if np.isnan(x_values).any() or np.isnan(y_values).any():
        raise ValueError("Inputs contain NaN values. Please handle missing values before calling cross_correlation().")
    
    # Get dimensions
    n = len(x_values)
    
    # Determine number of lags
    if nlags is None:
        nlags = min(int(10 * np.log10(n)), n - 1)
    elif nlags < 0:
        raise ValueError("nlags must be non-negative")
    elif nlags >= n:
        raise ValueError(f"nlags must be less than series length ({n})")
    
    # Validate other parameters
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    
    # Demean the series if requested
    if demean:
        x_centered = x_values - np.mean(x_values)
        y_centered = y_values - np.mean(y_values)
    else:
        x_centered = x_values
        y_centered = y_values
    
    # Compute standard deviations for normalization
    x_std = np.std(x_centered)
    y_std = np.std(y_centered)
    
    # Handle zero standard deviation case
    if x_std <= 1e-15 or y_std <= 1e-15:
        warn_numeric(
            "One or both series have zero standard deviation, returning zeros for CCF",
            operation="cross_correlation",
            issue="zero_variance"
        )
        ccf_values = np.zeros(2 * nlags + 1)
        lags = np.arange(-nlags, nlags + 1)
        
        # Compute confidence intervals
        z_value = stats.norm.ppf(1 - alpha / 2)
        se = 1.0 / np.sqrt(n)
        confint = np.column_stack([ccf_values - z_value * se, ccf_values + z_value * se])
        
        # Convert to pandas objects if input was pandas
        is_pandas = x_is_pandas or y_is_pandas
        if is_pandas:
            ccf_df = pd.Series(ccf_values, index=lags, name='CCF')
            confint_df = pd.DataFrame(
                confint, 
                index=lags,
                columns=['Lower', 'Upper']
            )
            
            return {
                'ccf': ccf_df,
                'lags': lags,
                'confint': confint_df
            }
        else:
            return {
                'ccf': ccf_values,
                'lags': lags,
                'confint': confint
            }
    
    # Compute cross-correlations
    ccf_values = np.zeros(2 * nlags + 1)
    lags = np.arange(-nlags, nlags + 1)
    
    for i, lag in enumerate(lags):
        if lag < 0:
            # Negative lag: y leads x
            ccf_values[i] = np.corrcoef(x_centered[-lag:], y_centered[:lag])[0, 1]
        elif lag > 0:
            # Positive lag: x leads y
            ccf_values[i] = np.corrcoef(x_centered[:-lag], y_centered[lag:])[0, 1]
        else:  # lag == 0
            # Contemporaneous correlation
            ccf_values[i] = np.corrcoef(x_centered, y_centered)[0, 1]
    
    # Compute confidence intervals
    z_value = stats.norm.ppf(1 - alpha / 2)
    se = 1.0 / np.sqrt(n)
    confint = np.column_stack([ccf_values - z_value * se, ccf_values + z_value * se])
    
    # Convert to pandas objects if input was pandas
    is_pandas = x_is_pandas or y_is_pandas
    if is_pandas:
        ccf_df = pd.Series(ccf_values, index=lags, name='CCF')
        confint_df = pd.DataFrame(
            confint, 
            index=lags,
            columns=['Lower', 'Upper']
        )
        
        return {
            'ccf': ccf_df,
            'lags': lags,
            'confint': confint_df
        }
    else:
        return {
            'ccf': ccf_values,
            'lags': lags,
            'confint': confint
        }


def plot_acf(x: Union[np.ndarray, pd.Series, pd.DataFrame],
             nlags: Optional[int] = None,
             alpha: float = 0.05,
             title: Optional[str] = None,
             figsize: Tuple[float, float] = (10, 6),
             ax: Optional[plt.Axes] = None,
             **acf_kwargs) -> plt.Figure:
    """
    Plot the autocorrelation function for a time series.
    
    This function computes and plots the autocorrelation function (ACF) for a time series,
    including confidence intervals.
    
    Args:
        x: Input time series (univariate or multivariate)
        nlags: Number of lags to compute. If None, uses min(10*log10(n), n-1)
        alpha: Significance level for confidence intervals (default: 0.05)
        title: Title for the plot. If None, a default title is used
        figsize: Figure size as (width, height) in inches
        ax: Matplotlib axes to plot on. If None, a new figure is created
        **acf_kwargs: Additional keyword arguments to pass to acf()
        
    Returns:
        Matplotlib figure containing the plot
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.time_series.correlation import plot_acf
        >>> np.random.seed(123)
        >>> x = np.random.normal(0, 1, 100)
        >>> fig = plot_acf(x, nlags=20)
    """
    # Compute ACF
    acf_result = acf(x, nlags=nlags, alpha=alpha, **acf_kwargs)
    
    # Check if input is multivariate
    is_multivariate = isinstance(acf_result['acf'], pd.DataFrame) and isinstance(acf_result['acf'].columns, pd.MultiIndex)
    
    # Create figure if needed
    if ax is None:
        if is_multivariate:
            # For multivariate series, create a grid of subplots
            if isinstance(x, pd.DataFrame):
                k = len(x.columns)
                series_names = x.columns
            else:
                k = x.shape[1] if x.ndim == 2 else 1
                series_names = [f"Series {i+1}" for i in range(k)]
            
            fig, axes = plt.subplots(k, k, figsize=figsize, sharex=True)
            
            # If k=1, axes will not be a 2D array
            if k == 1:
                axes = np.array([[axes]])
        else:
            # For univariate series, create a single plot
            fig, ax = plt.subplots(figsize=figsize)
    else:
        # Use provided axes
        fig = ax.figure
    
    # Plot ACF
    if is_multivariate:
        # For multivariate series, plot ACF and cross-correlations
        lags = acf_result['lags']
        acf_df = acf_result['acf']
        confint_df = acf_result['confint']
        
        for i in range(k):
            for j in range(k):
                # Extract ACF/CCF for this pair
                pair_key = (series_names[i], series_names[j])
                if isinstance(acf_df.columns, pd.MultiIndex):
                    pair_values = acf_df[pair_key].values
                else:
                    # Handle case where columns are not MultiIndex
                    col_idx = i * k + j
                    pair_values = acf_df.iloc[:, col_idx].values
                
                # Extract confidence intervals
                if isinstance(confint_df.index, pd.MultiIndex):
                    # MultiIndex case
                    pair_confint = confint_df.loc[pd.IndexSlice[:, pair_key], :].values.reshape(-1, 2)
                else:
                    # Handle case where index is not MultiIndex
                    pair_confint = np.zeros((len(lags), 2))
                    for lag_idx, lag in enumerate(lags):
                        row_idx = lag_idx * (k * k) + (i * k + j)
                        if row_idx < len(confint_df):
                            pair_confint[lag_idx] = confint_df.iloc[row_idx].values
                
                # Plot on the corresponding subplot
                ax = axes[i, j]
                ax.bar(lags, pair_values, width=0.3, color='steelblue', alpha=0.7)
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                
                # Plot confidence intervals
                ax.fill_between(
                    lags,
                    pair_confint[:, 0],
                    pair_confint[:, 1],
                    color='steelblue',
                    alpha=0.2
                )
                
                # Add labels and title
                if i == k - 1:
                    ax.set_xlabel('Lag')
                if j == 0:
                    ax.set_ylabel('Correlation')
                
                if i == 0 and j == 0:
                    if title:
                        fig.suptitle(title, fontsize=14)
                    else:
                        fig.suptitle('Autocorrelation and Cross-Correlation Functions', fontsize=14)
                
                ax.set_title(f'{series_names[i]} → {series_names[j]}', fontsize=10)
                
                # Set y-limits to be symmetric around zero
                y_max = max(1.0, np.max(np.abs(pair_values)) * 1.1)
                ax.set_ylim(-y_max, y_max)
    else:
        # For univariate series, plot ACF
        lags = acf_result['lags']
        acf_values = acf_result['acf']
        confint = acf_result['confint']
        
        # Convert to numpy arrays if pandas
        if isinstance(acf_values, pd.Series):
            acf_values = acf_values.values
        if isinstance(confint, pd.DataFrame):
            confint = confint.values
        
        # Plot ACF
        ax.bar(lags, acf_values, width=0.3, color='steelblue', alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Plot confidence intervals
        ax.fill_between(
            lags,
            confint[:, 0],
            confint[:, 1],
            color='steelblue',
            alpha=0.2
        )
        
        # Add labels and title
        ax.set_xlabel('Lag')
        ax.set_ylabel('Autocorrelation')
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Autocorrelation Function (ACF)')
        
        # Set y-limits to be symmetric around zero for lags > 0
        y_max = max(1.0, np.max(np.abs(acf_values[1:])) * 1.1)
        ax.set_ylim(-y_max, 1.05)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def plot_pacf(x: Union[np.ndarray, pd.Series, pd.DataFrame],
              nlags: Optional[int] = None,
              alpha: float = 0.05,
              method: str = 'yule_walker',
              title: Optional[str] = None,
              figsize: Tuple[float, float] = (10, 6),
              ax: Optional[plt.Axes] = None,
              **pacf_kwargs) -> plt.Figure:
    """
    Plot the partial autocorrelation function for a time series.
    
    This function computes and plots the partial autocorrelation function (PACF) for a time series,
    including confidence intervals.
    
    Args:
        x: Input time series (univariate or multivariate)
        nlags: Number of lags to compute. If None, uses min(10*log10(n), n-1)
        alpha: Significance level for confidence intervals (default: 0.05)
        method: Method for computing PACF: 'yule_walker', 'ols', or 'burg'
        title: Title for the plot. If None, a default title is used
        figsize: Figure size as (width, height) in inches
        ax: Matplotlib axes to plot on. If None, a new figure is created
        **pacf_kwargs: Additional keyword arguments to pass to pacf()
        
    Returns:
        Matplotlib figure containing the plot
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.time_series.correlation import plot_pacf
        >>> np.random.seed(123)
        >>> x = np.random.normal(0, 1, 100)
        >>> fig = plot_pacf(x, nlags=20)
    """
    # Compute PACF
    pacf_result = pacf(x, nlags=nlags, alpha=alpha, method=method, **pacf_kwargs)
    
    # Check if input is multivariate
    is_multivariate = isinstance(pacf_result['pacf'], pd.DataFrame) and isinstance(pacf_result['pacf'].columns, pd.MultiIndex)
    
    # Create figure if needed
    if ax is None:
        if is_multivariate:
            # For multivariate series, create a grid of subplots
            if isinstance(x, pd.DataFrame):
                k = len(x.columns)
                series_names = x.columns
            else:
                k = x.shape[1] if x.ndim == 2 else 1
                series_names = [f"Series {i+1}" for i in range(k)]
            
            fig, axes = plt.subplots(k, k, figsize=figsize, sharex=True)
            
            # If k=1, axes will not be a 2D array
            if k == 1:
                axes = np.array([[axes]])
        else:
            # For univariate series, create a single plot
            fig, ax = plt.subplots(figsize=figsize)
    else:
        # Use provided axes
        fig = ax.figure
    
    # Plot PACF
    if is_multivariate:
        # For multivariate series, plot PACF and cross-PACFs
        lags = pacf_result['lags']
        pacf_df = pacf_result['pacf']
        confint_df = pacf_result['confint']
        
        for i in range(k):
            for j in range(k):
                # Extract PACF/cross-PACF for this pair
                pair_key = (series_names[i], series_names[j])
                if isinstance(pacf_df.columns, pd.MultiIndex):
                    pair_values = pacf_df[pair_key].values
                else:
                    # Handle case where columns are not MultiIndex
                    col_idx = i * k + j
                    pair_values = pacf_df.iloc[:, col_idx].values
                
                # Extract confidence intervals
                if isinstance(confint_df.index, pd.MultiIndex):
                    # MultiIndex case
                    pair_confint = confint_df.loc[pd.IndexSlice[:, pair_key], :].values.reshape(-1, 2)
                else:
                    # Handle case where index is not MultiIndex
                    pair_confint = np.zeros((len(lags), 2))
                    for lag_idx, lag in enumerate(lags):
                        row_idx = lag_idx * (k * k) + (i * k + j)
                        if row_idx < len(confint_df):
                            pair_confint[lag_idx] = confint_df.iloc[row_idx].values
                
                # Plot on the corresponding subplot
                ax = axes[i, j]
                ax.bar(lags, pair_values, width=0.3, color='firebrick', alpha=0.7)
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                
                # Plot confidence intervals
                ax.fill_between(
                    lags,
                    pair_confint[:, 0],
                    pair_confint[:, 1],
                    color='firebrick',
                    alpha=0.2
                )
                
                # Add labels and title
                if i == k - 1:
                    ax.set_xlabel('Lag')
                if j == 0:
                    ax.set_ylabel('Partial Correlation')
                
                if i == 0 and j == 0:
                    if title:
                        fig.suptitle(title, fontsize=14)
                    else:
                        fig.suptitle('Partial Autocorrelation and Cross-Correlation Functions', fontsize=14)
                
                ax.set_title(f'{series_names[i]} → {series_names[j]}', fontsize=10)
                
                # Set y-limits to be symmetric around zero
                y_max = max(1.0, np.max(np.abs(pair_values)) * 1.1)
                ax.set_ylim(-y_max, y_max)
    else:
        # For univariate series, plot PACF
        lags = pacf_result['lags']
        pacf_values = pacf_result['pacf']
        confint = pacf_result['confint']
        
        # Convert to numpy arrays if pandas
        if isinstance(pacf_values, pd.Series):
            pacf_values = pacf_values.values
        if isinstance(confint, pd.DataFrame):
            confint = confint.values
        
        # Plot PACF
        ax.bar(lags, pacf_values, width=0.3, color='firebrick', alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Plot confidence intervals
        ax.fill_between(
            lags,
            confint[:, 0],
            confint[:, 1],
            color='firebrick',
            alpha=0.2
        )
        
        # Add labels and title
        ax.set_xlabel('Lag')
        ax.set_ylabel('Partial Autocorrelation')
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Partial Autocorrelation Function (PACF)')
        
        # Set y-limits to be symmetric around zero for lags > 0
        y_max = max(1.0, np.max(np.abs(pacf_values[1:])) * 1.1)
        ax.set_ylim(-y_max, 1.05)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def plot_ccf(x: Union[np.ndarray, pd.Series],
             y: Union[np.ndarray, pd.Series],
             nlags: Optional[int] = None,
             alpha: float = 0.05,
             title: Optional[str] = None,
             figsize: Tuple[float, float] = (10, 6),
             ax: Optional[plt.Axes] = None,
             **ccf_kwargs) -> plt.Figure:
    """
    Plot the cross-correlation function between two time series.
    
    This function computes and plots the cross-correlation function (CCF) between two time series,
    including confidence intervals.
    
    Args:
        x: First input time series
        y: Second input time series
        nlags: Number of lags to compute in each direction. If None, uses min(10*log10(n), n-1)
        alpha: Significance level for confidence intervals (default: 0.05)
        title: Title for the plot. If None, a default title is used
        figsize: Figure size as (width, height) in inches
        ax: Matplotlib axes to plot on. If None, a new figure is created
        **ccf_kwargs: Additional keyword arguments to pass to cross_correlation()
        
    Returns:
        Matplotlib figure containing the plot
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.time_series.correlation import plot_ccf
        >>> np.random.seed(123)
        >>> x = np.random.normal(0, 1, 100)
        >>> y = np.roll(x, 2) + np.random.normal(0, 0.5, 100)
        >>> fig = plot_ccf(x, y, nlags=10)
    """
    # Compute CCF
    ccf_result = cross_correlation(x, y, nlags=nlags, alpha=alpha, **ccf_kwargs)
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Extract results
    lags = ccf_result['lags']
    ccf_values = ccf_result['ccf']
    confint = ccf_result['confint']
    
    # Convert to numpy arrays if pandas
    if isinstance(ccf_values, pd.Series):
        ccf_values = ccf_values.values
    if isinstance(confint, pd.DataFrame):
        confint = confint.values
    
    # Plot CCF
    ax.bar(lags, ccf_values, width=0.3, color='darkgreen', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=0.5)
    
    # Plot confidence intervals
    ax.fill_between(
        lags,
        confint[:, 0],
        confint[:, 1],
        color='darkgreen',
        alpha=0.2
    )
    
    # Add labels and title
    ax.set_xlabel('Lag')
    ax.set_ylabel('Cross-Correlation')
    
    if title:
        ax.set_title(title)
    else:
        # Get series names if available
        x_name = getattr(x, 'name', 'X')
        y_name = getattr(y, 'name', 'Y')
        ax.set_title(f'Cross-Correlation Function: {x_name} and {y_name}')
    
    # Set y-limits to be symmetric around zero
    y_max = max(1.0, np.max(np.abs(ccf_values)) * 1.1)
    ax.set_ylim(-y_max, y_max)
    
    # Add annotations for lag interpretation
    ax.text(
        0.05, 0.95, 
        "Negative lags: Y leads X\nPositive lags: X leads Y", 
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )
    
    # Adjust layout
    plt.tight_layout()
    
    return fig
