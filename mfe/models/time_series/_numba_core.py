"""
Numba-accelerated core functions for time series analysis.

This module provides high-performance implementations of computationally intensive
operations used in time series analysis. These functions are accelerated using
Numba's just-in-time (JIT) compilation to achieve near-C performance while
maintaining the readability and maintainability of Python code.

The module includes optimized implementations for:
- ARMA/ARMAX model recursions and forecasting
- Autocorrelation and partial autocorrelation calculations
- Simulation and bootstrap procedures
- Likelihood evaluation for various time series models

These functions replace the MEX C implementations from the MATLAB version with
cross-platform Numba acceleration, providing significant performance improvements
for large datasets and complex models.
"""

import logging
import warnings
from typing import Tuple, Optional, Union, List, Dict, Any

import numpy as np
from numba import jit, prange, float64, int64, boolean, void

# Set up module-level logger
logger = logging.getLogger("mfe.models.time_series._numba_core")

# ============================================================================
# ARMA/ARMAX Model Core Functions
# ============================================================================

@jit(nopython=True, cache=True)
def arma_recursion(data: np.ndarray,
                  ar_params: np.ndarray,
                  ma_params: np.ndarray,
                  constant: float,
                  ar_order: int,
                  ma_order: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ARMA recursion for residuals and fitted values.
    
    This function implements the core ARMA recursion to compute residuals
    and fitted values given the model parameters and data. It is accelerated
    using Numba's JIT compilation for improved performance.
    
    Args:
        data: Input time series data
        ar_params: Autoregressive parameters
        ma_params: Moving average parameters
        constant: Constant term
        ar_order: Order of the autoregressive component
        ma_order: Order of the moving average component
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Residuals and fitted values
    """
    n = len(data)
    residuals = np.zeros(n)
    fitted = np.zeros(n)
    
    # Initialize with zeros
    for t in range(max(ar_order, ma_order)):
        fitted[t] = data[t]
        residuals[t] = 0.0
    
    # Main recursion
    for t in range(max(ar_order, ma_order), n):
        # Add constant term
        fitted[t] = constant
        
        # Add AR terms
        for i in range(ar_order):
            if t - i - 1 >= 0:
                fitted[t] += ar_params[i] * data[t - i - 1]
        
        # Add MA terms
        for j in range(ma_order):
            if t - j - 1 >= 0:
                fitted[t] -= ma_params[j] * residuals[t - j - 1]
        
        # Compute residual
        residuals[t] = data[t] - fitted[t]
    
    return residuals, fitted


@jit(nopython=True, cache=True)
def armax_recursion(data: np.ndarray,
                   exog: np.ndarray,
                   ar_params: np.ndarray,
                   ma_params: np.ndarray,
                   exog_params: np.ndarray,
                   constant: float,
                   ar_order: int,
                   ma_order: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ARMAX recursion for residuals and fitted values.
    
    This function implements the core ARMAX recursion to compute residuals
    and fitted values given the model parameters, data, and exogenous variables.
    
    Args:
        data: Input time series data
        exog: Exogenous variables
        ar_params: Autoregressive parameters
        ma_params: Moving average parameters
        exog_params: Parameters for exogenous variables
        constant: Constant term
        ar_order: Order of the autoregressive component
        ma_order: Order of the moving average component
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Residuals and fitted values
    """
    n = len(data)
    residuals = np.zeros(n)
    fitted = np.zeros(n)
    
    # Initialize with zeros
    for t in range(max(ar_order, ma_order)):
        fitted[t] = data[t]
        residuals[t] = 0.0
    
    # Main recursion
    for t in range(max(ar_order, ma_order), n):
        # Add constant term
        fitted[t] = constant
        
        # Add AR terms
        for i in range(ar_order):
            if t - i - 1 >= 0:
                fitted[t] += ar_params[i] * data[t - i - 1]
        
        # Add MA terms
        for j in range(ma_order):
            if t - j - 1 >= 0:
                fitted[t] -= ma_params[j] * residuals[t - j - 1]
        
        # Add exogenous variables
        for k in range(exog.shape[1]):
            fitted[t] += exog_params[k] * exog[t, k]
        
        # Compute residual
        residuals[t] = data[t] - fitted[t]
    
    return residuals, fitted


@jit(nopython=True, cache=True)
def arma_forecast(data: np.ndarray,
                 residuals: np.ndarray,
                 ar_params: np.ndarray,
                 ma_params: np.ndarray,
                 constant: float,
                 ar_order: int,
                 ma_order: int,
                 steps: int) -> np.ndarray:
    """
    Generate forecasts from an ARMA model.
    
    This function implements the core ARMA forecasting algorithm to generate
    point forecasts given the model parameters, data, and residuals.
    
    Args:
        data: Input time series data
        residuals: Model residuals
        ar_params: Autoregressive parameters
        ma_params: Moving average parameters
        constant: Constant term
        ar_order: Order of the autoregressive component
        ma_order: Order of the moving average component
        steps: Number of steps to forecast
    
    Returns:
        np.ndarray: Point forecasts
    """
    n = len(data)
    forecasts = np.zeros(steps)
    
    # Create extended data and residuals arrays
    extended_data = np.zeros(n + steps)
    extended_residuals = np.zeros(n + steps)
    
    # Fill with actual data and residuals
    extended_data[:n] = data
    extended_residuals[:n] = residuals
    
    # Generate forecasts
    for h in range(steps):
        t = n + h
        
        # Add constant term
        forecasts[h] = constant
        
        # Add AR terms
        for i in range(ar_order):
            if t - i - 1 >= 0:
                if t - i - 1 < n:
                    # Use actual data
                    forecasts[h] += ar_params[i] * data[t - i - 1]
                else:
                    # Use forecasted data
                    forecasts[h] += ar_params[i] * extended_data[t - i - 1]
        
        # Add MA terms
        for j in range(ma_order):
            if t - j - 1 >= 0 and t - j - 1 < n:
                # Only use known residuals
                forecasts[h] -= ma_params[j] * residuals[t - j - 1]
        
        # Store forecast in extended data
        extended_data[t] = forecasts[h]
    
    return forecasts


@jit(nopython=True, cache=True)
def armax_forecast(data: np.ndarray,
                  residuals: np.ndarray,
                  exog_future: np.ndarray,
                  ar_params: np.ndarray,
                  ma_params: np.ndarray,
                  exog_params: np.ndarray,
                  constant: float,
                  ar_order: int,
                  ma_order: int,
                  steps: int) -> np.ndarray:
    """
    Generate forecasts from an ARMAX model.
    
    This function implements the core ARMAX forecasting algorithm to generate
    point forecasts given the model parameters, data, residuals, and future
    exogenous variables.
    
    Args:
        data: Input time series data
        residuals: Model residuals
        exog_future: Future exogenous variables
        ar_params: Autoregressive parameters
        ma_params: Moving average parameters
        exog_params: Parameters for exogenous variables
        constant: Constant term
        ar_order: Order of the autoregressive component
        ma_order: Order of the moving average component
        steps: Number of steps to forecast
    
    Returns:
        np.ndarray: Point forecasts
    """
    n = len(data)
    forecasts = np.zeros(steps)
    
    # Create extended data and residuals arrays
    extended_data = np.zeros(n + steps)
    extended_residuals = np.zeros(n + steps)
    
    # Fill with actual data and residuals
    extended_data[:n] = data
    extended_residuals[:n] = residuals
    
    # Generate forecasts
    for h in range(steps):
        t = n + h
        
        # Add constant term
        forecasts[h] = constant
        
        # Add AR terms
        for i in range(ar_order):
            if t - i - 1 >= 0:
                if t - i - 1 < n:
                    # Use actual data
                    forecasts[h] += ar_params[i] * data[t - i - 1]
                else:
                    # Use forecasted data
                    forecasts[h] += ar_params[i] * extended_data[t - i - 1]
        
        # Add MA terms
        for j in range(ma_order):
            if t - j - 1 >= 0 and t - j - 1 < n:
                # Only use known residuals
                forecasts[h] -= ma_params[j] * residuals[t - j - 1]
        
        # Add exogenous variables
        for k in range(exog_future.shape[1]):
            forecasts[h] += exog_params[k] * exog_future[h, k]
        
        # Store forecast in extended data
        extended_data[t] = forecasts[h]
    
    return forecasts


@jit(nopython=True, cache=True)
def arma_simulate(ar_params: np.ndarray,
                 ma_params: np.ndarray,
                 constant: float,
                 sigma2: float,
                 ar_order: int,
                 ma_order: int,
                 n_periods: int,
                 burn: int,
                 initial_values: np.ndarray,
                 innovations: np.ndarray) -> np.ndarray:
    """
    Simulate data from an ARMA model.
    
    This function implements the core ARMA simulation algorithm to generate
    simulated data given the model parameters.
    
    Args:
        ar_params: Autoregressive parameters
        ma_params: Moving average parameters
        constant: Constant term
        sigma2: Innovation variance
        ar_order: Order of the autoregressive component
        ma_order: Order of the moving average component
        n_periods: Number of periods to simulate
        burn: Number of initial observations to discard
        initial_values: Initial values for the simulation
        innovations: Random innovations for the simulation
    
    Returns:
        np.ndarray: Simulated data
    """
    max_lag = max(ar_order, ma_order)
    total_periods = n_periods + burn
    
    # Initialize arrays
    simulated = np.zeros(total_periods + max_lag)
    errors = np.zeros(total_periods + max_lag)
    
    # Set initial values
    if len(initial_values) > 0:
        simulated[:min(len(initial_values), max_lag)] = initial_values[:min(len(initial_values), max_lag)]
    
    # Set innovations
    errors[max_lag:] = innovations
    
    # Generate simulated data
    for t in range(max_lag, total_periods + max_lag):
        # Add constant term
        simulated[t] = constant
        
        # Add AR terms
        for i in range(ar_order):
            if t - i - 1 >= 0:
                simulated[t] += ar_params[i] * simulated[t - i - 1]
        
        # Add MA terms and current error
        simulated[t] += errors[t]
        for i in range(ma_order):
            if t - i - 1 >= 0:
                simulated[t] += ma_params[i] * errors[t - i - 1]
    
    # Return simulated data (excluding burn-in)
    return simulated[max_lag + burn:]


@jit(nopython=True, cache=True)
def armax_simulate(ar_params: np.ndarray,
                  ma_params: np.ndarray,
                  exog_params: np.ndarray,
                  exog: np.ndarray,
                  constant: float,
                  sigma2: float,
                  ar_order: int,
                  ma_order: int,
                  n_periods: int,
                  burn: int,
                  initial_values: np.ndarray,
                  innovations: np.ndarray) -> np.ndarray:
    """
    Simulate data from an ARMAX model.
    
    This function implements the core ARMAX simulation algorithm to generate
    simulated data given the model parameters and exogenous variables.
    
    Args:
        ar_params: Autoregressive parameters
        ma_params: Moving average parameters
        exog_params: Parameters for exogenous variables
        exog: Exogenous variables
        constant: Constant term
        sigma2: Innovation variance
        ar_order: Order of the autoregressive component
        ma_order: Order of the moving average component
        n_periods: Number of periods to simulate
        burn: Number of initial observations to discard
        initial_values: Initial values for the simulation
        innovations: Random innovations for the simulation
    
    Returns:
        np.ndarray: Simulated data
    """
    max_lag = max(ar_order, ma_order)
    total_periods = n_periods + burn
    
    # Initialize arrays
    simulated = np.zeros(total_periods + max_lag)
    errors = np.zeros(total_periods + max_lag)
    
    # Set initial values
    if len(initial_values) > 0:
        simulated[:min(len(initial_values), max_lag)] = initial_values[:min(len(initial_values), max_lag)]
    
    # Set innovations
    errors[max_lag:] = innovations
    
    # Generate simulated data
    for t in range(max_lag, total_periods + max_lag):
        # Add constant term
        simulated[t] = constant
        
        # Add AR terms
        for i in range(ar_order):
            if t - i - 1 >= 0:
                simulated[t] += ar_params[i] * simulated[t - i - 1]
        
        # Add MA terms and current error
        simulated[t] += errors[t]
        for i in range(ma_order):
            if t - i - 1 >= 0:
                simulated[t] += ma_params[i] * errors[t - i - 1]
        
        # Add exogenous variables
        idx = t - max_lag
        if idx < exog.shape[0]:
            for k in range(exog.shape[1]):
                simulated[t] += exog_params[k] * exog[idx, k]
    
    # Return simulated data (excluding burn-in)
    return simulated[max_lag + burn:]


@jit(nopython=True, cache=True)
def arma_loglikelihood(data: np.ndarray,
                      ar_params: np.ndarray,
                      ma_params: np.ndarray,
                      constant: float,
                      sigma2: float,
                      ar_order: int,
                      ma_order: int) -> Tuple[float, np.ndarray]:
    """
    Compute the log-likelihood of an ARMA model.
    
    This function computes the log-likelihood of the data given the ARMA model
    parameters, which is used for parameter estimation and model comparison.
    
    Args:
        data: Input time series data
        ar_params: Autoregressive parameters
        ma_params: Moving average parameters
        constant: Constant term
        sigma2: Innovation variance
        ar_order: Order of the autoregressive component
        ma_order: Order of the moving average component
    
    Returns:
        Tuple[float, np.ndarray]: Log-likelihood value and residuals
    """
    n = len(data)
    residuals, _ = arma_recursion(data, ar_params, ma_params, constant, ar_order, ma_order)
    
    # Compute log-likelihood
    loglik = -0.5 * n * np.log(2 * np.pi * sigma2)
    loglik -= 0.5 * np.sum(residuals**2) / sigma2
    
    return loglik, residuals


@jit(nopython=True, cache=True)
def armax_loglikelihood(data: np.ndarray,
                       exog: np.ndarray,
                       ar_params: np.ndarray,
                       ma_params: np.ndarray,
                       exog_params: np.ndarray,
                       constant: float,
                       sigma2: float,
                       ar_order: int,
                       ma_order: int) -> Tuple[float, np.ndarray]:
    """
    Compute the log-likelihood of an ARMAX model.
    
    This function computes the log-likelihood of the data given the ARMAX model
    parameters, which is used for parameter estimation and model comparison.
    
    Args:
        data: Input time series data
        exog: Exogenous variables
        ar_params: Autoregressive parameters
        ma_params: Moving average parameters
        exog_params: Parameters for exogenous variables
        constant: Constant term
        sigma2: Innovation variance
        ar_order: Order of the autoregressive component
        ma_order: Order of the moving average component
    
    Returns:
        Tuple[float, np.ndarray]: Log-likelihood value and residuals
    """
    n = len(data)
    residuals, _ = armax_recursion(data, exog, ar_params, ma_params, exog_params, 
                                  constant, ar_order, ma_order)
    
    # Compute log-likelihood
    loglik = -0.5 * n * np.log(2 * np.pi * sigma2)
    loglik -= 0.5 * np.sum(residuals**2) / sigma2
    
    return loglik, residuals


# ============================================================================
# Correlation Analysis Functions
# ============================================================================

@jit(nopython=True, cache=True)
def acf_numba(x: np.ndarray, nlags: int, demean: bool = True) -> np.ndarray:
    """
    Compute autocorrelation function (ACF) for a time series.
    
    This function calculates the autocorrelation function for a time series,
    which measures the linear dependence between observations separated by
    different lags.
    
    Args:
        x: Input time series (1D array)
        nlags: Number of lags to compute
        demean: Whether to subtract the mean from the series
        
    Returns:
        np.ndarray: Array of autocorrelation coefficients
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


@jit(nopython=True, cache=True)
def pacf_yule_walker_numba(x: np.ndarray, nlags: int, demean: bool = True) -> np.ndarray:
    """
    Compute partial autocorrelation function (PACF) using Yule-Walker equations.
    
    This function calculates the partial autocorrelation function for a time series,
    which measures the correlation between observations separated by different lags
    after removing the effects of intermediate observations.
    
    Args:
        x: Input time series (1D array)
        nlags: Number of lags to compute
        demean: Whether to subtract the mean from the series
        
    Returns:
        np.ndarray: Array of partial autocorrelation coefficients
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


@jit(nopython=True, cache=True)
def ccf_numba(x: np.ndarray, y: np.ndarray, nlags: int, demean: bool = True) -> np.ndarray:
    """
    Compute cross-correlation function (CCF) between two time series.
    
    This function calculates the cross-correlation function between two time series,
    which measures the linear dependence between observations in different series
    at different lags.
    
    Args:
        x: First input time series (1D array)
        y: Second input time series (1D array)
        nlags: Number of lags to compute in each direction
        demean: Whether to subtract the mean from the series
        
    Returns:
        np.ndarray: Array of cross-correlation coefficients
    """
    n = len(x)
    ccf = np.zeros(2 * nlags + 1)
    
    # Demean the series if requested
    if demean:
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        x_centered = x - x_mean
        y_centered = y - y_mean
    else:
        x_centered = x
        y_centered = y
    
    # Compute standard deviations for normalization
    x_std = np.sqrt(np.sum(x_centered ** 2) / n)
    y_std = np.sqrt(np.sum(y_centered ** 2) / n)
    
    # Handle zero standard deviation case
    if x_std <= 1e-15 or y_std <= 1e-15:
        return np.zeros(2 * nlags + 1)
    
    # Compute cross-correlations
    for i, lag in enumerate(range(-nlags, nlags + 1)):
        if lag < 0:
            # Negative lag: y leads x
            cov = 0.0
            for t in range(-lag, n):
                cov += x_centered[t] * y_centered[t + lag]
            cov /= n
            ccf[i] = cov / (x_std * y_std)
        elif lag > 0:
            # Positive lag: x leads y
            cov = 0.0
            for t in range(lag, n):
                cov += x_centered[t - lag] * y_centered[t]
            cov /= n
            ccf[i] = cov / (x_std * y_std)
        else:  # lag == 0
            # Contemporaneous correlation
            cov = 0.0
            for t in range(n):
                cov += x_centered[t] * y_centered[t]
            cov /= n
            ccf[i] = cov / (x_std * y_std)
    
    return ccf


# ============================================================================
# Unit Root Test Functions
# ============================================================================

@jit(nopython=True, cache=True)
def adf_regression(y: np.ndarray, lags: int, trend: str) -> Tuple[float, float, float, np.ndarray]:
    """
    Perform Augmented Dickey-Fuller regression.
    
    This function implements the core regression for the Augmented Dickey-Fuller
    unit root test, which tests for the presence of a unit root in a time series.
    
    Args:
        y: Input time series
        lags: Number of lagged difference terms
        trend: Trend specification ('n', 'c', 'ct', or 'ctt')
    
    Returns:
        Tuple[float, float, float, np.ndarray]: ADF statistic, p-value, 
                                               regression coefficient, and residuals
    """
    n = len(y)
    
    # Create dependent variable (first difference)
    dy = np.zeros(n - 1)
    for t in range(1, n):
        dy[t - 1] = y[t] - y[t - 1]
    
    # Create lagged level
    y_lag = y[:-1]
    
    # Create lagged differences
    if lags > 0:
        X = np.zeros((n - lags - 1, lags + 1))
        X[:, 0] = y_lag[lags:]
        
        for i in range(1, lags + 1):
            for t in range(lags, n - 1):
                X[t - lags, i] = dy[t - i]
        
        # Add trend components
        if trend != 'n':  # 'c', 'ct', or 'ctt'
            if trend == 'c':
                # Constant only
                X_trend = np.ones((n - lags - 1, 1))
            elif trend == 'ct':
                # Constant and trend
                X_trend = np.zeros((n - lags - 1, 2))
                X_trend[:, 0] = 1.0
                X_trend[:, 1] = np.arange(lags + 1, n)
            else:  # 'ctt'
                # Constant, trend, and trend^2
                X_trend = np.zeros((n - lags - 1, 3))
                X_trend[:, 0] = 1.0
                X_trend[:, 1] = np.arange(lags + 1, n)
                X_trend[:, 2] = np.arange(lags + 1, n) ** 2
            
            # Combine regressors
            X_full = np.zeros((n - lags - 1, X.shape[1] + X_trend.shape[1]))
            X_full[:, :X_trend.shape[1]] = X_trend
            X_full[:, X_trend.shape[1]:] = X
            
            X = X_full
        
        # Dependent variable
        y_reg = dy[lags:]
        
        # Perform OLS regression
        # X'X
        XTX = np.zeros((X.shape[1], X.shape[1]))
        for i in range(X.shape[1]):
            for j in range(X.shape[1]):
                XTX[i, j] = np.sum(X[:, i] * X[:, j])
        
        # X'y
        XTy = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            XTy[i] = np.sum(X[:, i] * y_reg)
        
        # Solve for beta: (X'X)^(-1) X'y
        # Simple Gaussian elimination for small systems
        beta = np.zeros(X.shape[1])
        
        # Copy XTX to avoid modifying it
        A = XTX.copy()
        b = XTy.copy()
        
        # Gaussian elimination
        n_params = X.shape[1]
        for i in range(n_params):
            # Find pivot
            max_idx = i
            max_val = abs(A[i, i])
            for j in range(i + 1, n_params):
                if abs(A[j, i]) > max_val:
                    max_idx = j
                    max_val = abs(A[j, i])
            
            # Swap rows if necessary
            if max_idx != i:
                for j in range(i, n_params):
                    A[i, j], A[max_idx, j] = A[max_idx, j], A[i, j]
                b[i], b[max_idx] = b[max_idx], b[i]
            
            # Eliminate below
            for j in range(i + 1, n_params):
                factor = A[j, i] / A[i, i]
                for k in range(i, n_params):
                    A[j, k] -= factor * A[i, k]
                b[j] -= factor * b[i]
        
        # Back substitution
        for i in range(n_params - 1, -1, -1):
            beta[i] = b[i]
            for j in range(i + 1, n_params):
                beta[i] -= A[i, j] * beta[j]
            beta[i] /= A[i, i]
        
        # Compute residuals
        residuals = y_reg - np.dot(X, beta)
        
        # Compute standard error of the coefficient
        ssr = np.sum(residuals ** 2)
        sigma2 = ssr / (len(y_reg) - n_params)
        
        # Compute (X'X)^(-1)
        # This is a simplified approach and may not be as numerically stable
        # as scipy's solver for ill-conditioned matrices
        XTX_inv = np.zeros((n_params, n_params))
        
        # Identity matrix
        I = np.zeros((n_params, n_params))
        for i in range(n_params):
            I[i, i] = 1.0
        
        # Solve for each column of the inverse
        for i in range(n_params):
            # Solve A * x = e_i
            x = np.zeros(n_params)
            e_i = I[:, i].copy()
            
            # Forward substitution
            for j in range(n_params):
                x[j] = e_i[j]
                for k in range(j):
                    x[j] -= A[j, k] * x[k]
                x[j] /= A[j, j]
            
            # Backward substitution
            for j in range(n_params - 1, -1, -1):
                for k in range(j + 1, n_params):
                    x[j] -= A[j, k] * x[k]
                x[j] /= A[j, j]
            
            # Store the solution in the corresponding column of the inverse
            XTX_inv[:, i] = x
        
        # Standard error of the coefficient
        se_beta = np.sqrt(sigma2 * XTX_inv[0, 0])
        
        # Compute ADF statistic
        adf_stat = beta[0] / se_beta
        
        # Return results
        return adf_stat, beta[0], se_beta, residuals
    else:
        # No lags case
        if trend != 'n':  # 'c', 'ct', or 'ctt'
            if trend == 'c':
                # Constant only
                X = np.zeros((n - 1, 2))
                X[:, 0] = 1.0
                X[:, 1] = y_lag
            elif trend == 'ct':
                # Constant and trend
                X = np.zeros((n - 1, 3))
                X[:, 0] = 1.0
                X[:, 1] = np.arange(1, n)
                X[:, 2] = y_lag
            else:  # 'ctt'
                # Constant, trend, and trend^2
                X = np.zeros((n - 1, 4))
                X[:, 0] = 1.0
                X[:, 1] = np.arange(1, n)
                X[:, 2] = np.arange(1, n) ** 2
                X[:, 3] = y_lag
        else:
            # No constant, no trend
            X = np.zeros((n - 1, 1))
            X[:, 0] = y_lag
        
        # Dependent variable
        y_reg = dy
        
        # Perform OLS regression
        # X'X
        XTX = np.zeros((X.shape[1], X.shape[1]))
        for i in range(X.shape[1]):
            for j in range(X.shape[1]):
                XTX[i, j] = np.sum(X[:, i] * X[:, j])
        
        # X'y
        XTy = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            XTy[i] = np.sum(X[:, i] * y_reg)
        
        # Solve for beta: (X'X)^(-1) X'y
        # Simple Gaussian elimination for small systems
        beta = np.zeros(X.shape[1])
        
        # Copy XTX to avoid modifying it
        A = XTX.copy()
        b = XTy.copy()
        
        # Gaussian elimination
        n_params = X.shape[1]
        for i in range(n_params):
            # Find pivot
            max_idx = i
            max_val = abs(A[i, i])
            for j in range(i + 1, n_params):
                if abs(A[j, i]) > max_val:
                    max_idx = j
                    max_val = abs(A[j, i])
            
            # Swap rows if necessary
            if max_idx != i:
                for j in range(i, n_params):
                    A[i, j], A[max_idx, j] = A[max_idx, j], A[i, j]
                b[i], b[max_idx] = b[max_idx], b[i]
            
            # Eliminate below
            for j in range(i + 1, n_params):
                factor = A[j, i] / A[i, i]
                for k in range(i, n_params):
                    A[j, k] -= factor * A[i, k]
                b[j] -= factor * b[i]
        
        # Back substitution
        for i in range(n_params - 1, -1, -1):
            beta[i] = b[i]
            for j in range(i + 1, n_params):
                beta[i] -= A[i, j] * beta[j]
            beta[i] /= A[i, i]
        
        # Compute residuals
        residuals = y_reg - np.dot(X, beta)
        
        # Compute standard error of the coefficient
        ssr = np.sum(residuals ** 2)
        sigma2 = ssr / (len(y_reg) - n_params)
        
        # Compute (X'X)^(-1)
        # This is a simplified approach and may not be as numerically stable
        # as scipy's solver for ill-conditioned matrices
        XTX_inv = np.zeros((n_params, n_params))
        
        # Identity matrix
        I = np.zeros((n_params, n_params))
        for i in range(n_params):
            I[i, i] = 1.0
        
        # Solve for each column of the inverse
        for i in range(n_params):
            # Solve A * x = e_i
            x = np.zeros(n_params)
            e_i = I[:, i].copy()
            
            # Forward substitution
            for j in range(n_params):
                x[j] = e_i[j]
                for k in range(j):
                    x[j] -= A[j, k] * x[k]
                x[j] /= A[j, j]
            
            # Backward substitution
            for j in range(n_params - 1, -1, -1):
                for k in range(j + 1, n_params):
                    x[j] -= A[j, k] * x[k]
                x[j] /= A[j, j]
            
            # Store the solution in the corresponding column of the inverse
            XTX_inv[:, i] = x
        
        # Standard error of the coefficient
        if trend == 'n':
            se_beta = np.sqrt(sigma2 * XTX_inv[0, 0])
            adf_stat = beta[0] / se_beta
            return adf_stat, beta[0], se_beta, residuals
        else:
            # The coefficient of interest is the last one (for y_lag)
            coef_idx = X.shape[1] - 1
            se_beta = np.sqrt(sigma2 * XTX_inv[coef_idx, coef_idx])
            adf_stat = beta[coef_idx] / se_beta
            return adf_stat, beta[coef_idx], se_beta, residuals


# ============================================================================
# Simulation and Bootstrap Functions
# ============================================================================

@jit(nopython=True, cache=True)
def block_bootstrap_indices(n: int, block_length: int, num_blocks: int, 
                           random_state: np.ndarray) -> np.ndarray:
    """
    Generate indices for block bootstrap.
    
    This function generates indices for the block bootstrap method, which
    resamples blocks of consecutive observations to preserve the dependence
    structure in the data.
    
    Args:
        n: Length of the original time series
        block_length: Length of each block
        num_blocks: Number of blocks to generate
        random_state: Random number generator state
    
    Returns:
        np.ndarray: Bootstrap indices
    """
    # Calculate the number of possible starting positions for blocks
    num_possible_blocks = n - block_length + 1
    
    # Generate random starting positions for blocks
    block_starts = np.floor(random_state[:num_blocks] * num_possible_blocks).astype(np.int64)
    
    # Generate indices for each block
    indices = np.zeros(num_blocks * block_length, dtype=np.int64)
    
    for i in range(num_blocks):
        start = block_starts[i]
        for j in range(block_length):
            indices[i * block_length + j] = start + j
    
    # Trim to the original length if necessary
    if len(indices) > n:
        indices = indices[:n]
    
    return indices


@jit(nopython=True, cache=True)
def stationary_bootstrap_indices(n: int, mean_block_length: float, 
                                random_state: np.ndarray) -> np.ndarray:
    """
    Generate indices for stationary bootstrap.
    
    This function generates indices for the stationary bootstrap method, which
    resamples blocks of random length to preserve the dependence structure in
    the data while ensuring stationarity.
    
    Args:
        n: Length of the original time series
        mean_block_length: Mean length of blocks
        random_state: Random number generator state (should be of length 2*n)
    
    Returns:
        np.ndarray: Bootstrap indices
    """
    # Probability of starting a new block
    p = 1.0 / mean_block_length
    
    # Initialize indices
    indices = np.zeros(n, dtype=np.int64)
    
    # Generate the first index randomly
    indices[0] = np.floor(random_state[0] * n).astype(np.int64)
    
    # Generate the rest of the indices
    for i in range(1, n):
        # Decide whether to start a new block or continue the current one
        if random_state[i] < p:
            # Start a new block
            indices[i] = np.floor(random_state[i + n] * n).astype(np.int64)
        else:
            # Continue the current block
            if indices[i - 1] == n - 1:
                # Wrap around if at the end of the series
                indices[i] = 0
            else:
                indices[i] = indices[i - 1] + 1
    
    return indices


@jit(nopython=True, cache=True)
def moving_block_bootstrap(data: np.ndarray, block_length: int, 
                          num_bootstraps: int, random_state: np.ndarray) -> np.ndarray:
    """
    Perform moving block bootstrap for a time series.
    
    This function implements the moving block bootstrap method, which resamples
    blocks of consecutive observations to preserve the dependence structure in
    the data.
    
    Args:
        data: Input time series
        block_length: Length of each block
        num_bootstraps: Number of bootstrap samples to generate
        random_state: Random number generator state
    
    Returns:
        np.ndarray: Bootstrap samples (num_bootstraps x n)
    """
    n = len(data)
    
    # Calculate the number of blocks needed
    num_blocks = int(np.ceil(n / block_length))
    
    # Initialize bootstrap samples
    bootstrap_samples = np.zeros((num_bootstraps, n))
    
    for b in range(num_bootstraps):
        # Generate indices for this bootstrap sample
        indices = block_bootstrap_indices(
            n, block_length, num_blocks, 
            random_state[b * num_blocks:(b + 1) * num_blocks]
        )
        
        # Create bootstrap sample
        bootstrap_samples[b, :] = data[indices]
    
    return bootstrap_samples


@jit(nopython=True, cache=True)
def stationary_bootstrap(data: np.ndarray, mean_block_length: float, 
                        num_bootstraps: int, random_state: np.ndarray) -> np.ndarray:
    """
    Perform stationary bootstrap for a time series.
    
    This function implements the stationary bootstrap method, which resamples
    blocks of random length to preserve the dependence structure in the data
    while ensuring stationarity.
    
    Args:
        data: Input time series
        mean_block_length: Mean length of blocks
        num_bootstraps: Number of bootstrap samples to generate
        random_state: Random number generator state
    
    Returns:
        np.ndarray: Bootstrap samples (num_bootstraps x n)
    """
    n = len(data)
    
    # Initialize bootstrap samples
    bootstrap_samples = np.zeros((num_bootstraps, n))
    
    for b in range(num_bootstraps):
        # Generate indices for this bootstrap sample
        indices = stationary_bootstrap_indices(
            n, mean_block_length, 
            random_state[b * 2 * n:(b + 1) * 2 * n]
        )
        
        # Create bootstrap sample
        bootstrap_samples[b, :] = data[indices]
    
    return bootstrap_samples


# ============================================================================
# Utility Functions
# ============================================================================

@jit(nopython=True, cache=True)
def lag_matrix(x: np.ndarray, lags: int, include_original: bool = True) -> np.ndarray:
    """
    Create a matrix of lagged values from a time series.
    
    This function creates a matrix where each column is a lagged version of the
    input time series, which is useful for regression models with lagged variables.
    
    Args:
        x: Input time series
        lags: Number of lags to include
        include_original: Whether to include the original series as the first column
    
    Returns:
        np.ndarray: Matrix of lagged values
    """
    n = len(x)
    
    if include_original:
        # Include the original series as the first column
        result = np.zeros((n - lags, lags + 1))
        result[:, 0] = x[lags:]
        
        for i in range(1, lags + 1):
            result[:, i] = x[lags - i:-i]
    else:
        # Only include lagged values
        result = np.zeros((n - lags, lags))
        
        for i in range(lags):
            result[:, i] = x[lags - i - 1:-i - 1]
    
    return result


@jit(nopython=True, cache=True)
def difference(x: np.ndarray, d: int = 1) -> np.ndarray:
    """
    Compute differences of a time series.
    
    This function computes the d-th differences of a time series, which is
    useful for making a series stationary.
    
    Args:
        x: Input time series
        d: Order of differencing
    
    Returns:
        np.ndarray: Differenced time series
    """
    result = x.copy()
    
    for _ in range(d):
        temp = np.zeros(len(result) - 1)
        for i in range(len(temp)):
            temp[i] = result[i + 1] - result[i]
        result = temp
    
    return result


@jit(nopython=True, cache=True)
def seasonal_difference(x: np.ndarray, period: int, d: int = 1) -> np.ndarray:
    """
    Compute seasonal differences of a time series.
    
    This function computes the d-th seasonal differences of a time series,
    which is useful for removing seasonal patterns.
    
    Args:
        x: Input time series
        period: Seasonal period
        d: Order of seasonal differencing
    
    Returns:
        np.ndarray: Seasonally differenced time series
    """
    result = x.copy()
    
    for _ in range(d):
        temp = np.zeros(len(result) - period)
        for i in range(len(temp)):
            temp[i] = result[i + period] - result[i]
        result = temp
    
    return result


@jit(nopython=True, cache=True)
def ljung_box_test(residuals: np.ndarray, lags: int, df: int = 0) -> Tuple[float, float]:
    """
    Compute the Ljung-Box test for autocorrelation in residuals.
    
    This function implements the Ljung-Box test, which tests for the presence
    of autocorrelation in a time series.
    
    Args:
        residuals: Residuals from a time series model
        lags: Number of lags to include in the test
        df: Degrees of freedom (number of parameters in the model)
    
    Returns:
        Tuple[float, float]: Test statistic and p-value
    """
    n = len(residuals)
    
    # Compute autocorrelations
    acf_values = acf_numba(residuals, lags, True)
    
    # Compute test statistic
    q_stat = 0.0
    for lag in range(1, lags + 1):
        q_stat += (acf_values[lag] ** 2) / (n - lag)
    
    q_stat = n * (n + 2) * q_stat
    
    # Compute p-value (approximated using chi-squared distribution)
    # Since we can't use scipy.stats in numba, we'll return the test statistic
    # and let the calling function compute the p-value
    
    return q_stat, lags - df
