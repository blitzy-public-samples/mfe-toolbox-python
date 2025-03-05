import numpy as np
from numba import jit, float64, int64, boolean, void


"""
Performance-critical core implementations for univariate volatility models accelerated with Numba.

This module contains JIT-compiled functions for computationally intensive recursions used by
various volatility models. These functions are optimized for performance using Numba's
just-in-time compilation capabilities, replacing the previous MATLAB MEX implementations
with platform-independent Python code that achieves similar performance.

The module is not intended to be used directly by end users but is instead imported by
the various volatility model classes that need high-performance implementations of their
core recursions.

All functions in this module are decorated with Numba's @jit decorator to enable
just-in-time compilation to machine code, providing significant performance improvements
(typically 10-100x) over pure Python implementations.
"""


# GARCH model core functions
@jit(float64[:](float64[:], float64, float64, float64, float64[:], float64), 
     nopython=True, cache=True)
def garch_recursion(data, omega, alpha, beta, sigma2, backcast):
    """Compute GARCH(1,1) conditional variances using Numba acceleration.
    
    This function implements the core recursion for the GARCH(1,1) model:
    σ²_t = ω + α * r²_{t-1} + β * σ²_{t-1}
    
    Args:
        data: Input data (typically residuals)
        omega: Constant term in variance equation
        alpha: ARCH parameter
        beta: GARCH parameter
        sigma2: Pre-allocated array for conditional variances
        backcast: Value to use for initializing the variance process
    
    Returns:
        np.ndarray: Conditional variances
    """
    T = len(data)
    
    # Initialize first variance with backcast value
    sigma2[0] = backcast
    
    # Compute conditional variances recursively
    for t in range(1, T):
        sigma2[t] = omega + alpha * data[t-1]**2 + beta * sigma2[t-1]
    
    return sigma2


@jit(float64[:](float64[:], float64, float64[:], float64[:], float64[:], float64), 
     nopython=True, cache=True)
def garch_p_q_recursion(data, omega, alpha, beta, sigma2, backcast):
    """Compute GARCH(p,q) conditional variances using Numba acceleration.
    
    This function implements the core recursion for the GARCH(p,q) model:
    σ²_t = ω + Σ(α_i * r²_{t-i}) + Σ(β_j * σ²_{t-j})
    
    Args:
        data: Input data (typically residuals)
        omega: Constant term in variance equation
        alpha: ARCH parameters (array of length p)
        beta: GARCH parameters (array of length q)
        sigma2: Pre-allocated array for conditional variances
        backcast: Value to use for initializing the variance process
    
    Returns:
        np.ndarray: Conditional variances
    """
    T = len(data)
    p = len(alpha)
    q = len(beta)
    
    # Initialize first max(p,q) variances with backcast value
    max_lag = max(p, q)
    for t in range(max_lag):
        sigma2[t] = backcast
    
    # Compute conditional variances recursively
    for t in range(max_lag, T):
        # Add constant term
        sigma2[t] = omega
        
        # Add ARCH terms
        for i in range(p):
            if t - i - 1 >= 0:  # Ensure we don't go out of bounds
                sigma2[t] += alpha[i] * data[t-i-1]**2
        
        # Add GARCH terms
        for j in range(q):
            if t - j - 1 >= 0:  # Ensure we don't go out of bounds
                sigma2[t] += beta[j] * sigma2[t-j-1]
    
    return sigma2


@jit(float64[:](float64, float64, float64, float64, int64), 
     nopython=True, cache=True)
def garch_forecast(omega, alpha, beta, last_variance, steps):
    """Generate analytic forecasts for GARCH(1,1) model using Numba acceleration.
    
    This function computes multi-step ahead forecasts for the GARCH(1,1) model.
    
    Args:
        omega: Constant term in variance equation
        alpha: ARCH parameter
        beta: GARCH parameter
        last_variance: Last observed conditional variance
        steps: Number of steps to forecast
    
    Returns:
        np.ndarray: Forecasted conditional variances
    """
    forecasts = np.zeros(steps)
    persistence = alpha + beta
    
    # Compute unconditional variance
    unconditional_variance = omega / (1 - persistence) if persistence < 1 else last_variance
    
    # First step forecast
    forecasts[0] = omega + alpha * last_variance + beta * last_variance
    
    # Multi-step forecasts
    for h in range(1, steps):
        forecasts[h] = omega + persistence * forecasts[h-1]
        
        # For long horizons, approach the unconditional variance
        if persistence < 1 and h > 100:
            forecasts[h] = unconditional_variance - (unconditional_variance - forecasts[h]) * 0.1
    
    return forecasts


@jit(float64[:](float64, float64[:], float64[:], float64[:], float64[:], int64), 
     nopython=True, cache=True)
def garch_p_q_forecast(omega, alpha, beta, last_variances, last_squared_returns, steps):
    """Generate analytic forecasts for GARCH(p,q) model using Numba acceleration.
    
    This function computes multi-step ahead forecasts for the GARCH(p,q) model.
    
    Args:
        omega: Constant term in variance equation
        alpha: ARCH parameters (array of length p)
        beta: GARCH parameters (array of length q)
        last_variances: Last q observed conditional variances (most recent first)
        last_squared_returns: Last p squared returns (most recent first)
        steps: Number of steps to forecast
    
    Returns:
        np.ndarray: Forecasted conditional variances
    """
    forecasts = np.zeros(steps)
    p = len(alpha)
    q = len(beta)
    
    # Compute persistence
    persistence = np.sum(alpha) + np.sum(beta)
    
    # Compute unconditional variance
    unconditional_variance = omega / (1 - persistence) if persistence < 1 else np.mean(last_variances)
    
    # Initialize arrays for multi-step forecasting
    future_variances = np.zeros(max(p, q) + steps)
    future_squared_returns = np.zeros(p + steps)
    
    # Fill in known values
    for i in range(q):
        if i < len(last_variances):
            future_variances[i] = last_variances[i]
        else:
            future_variances[i] = unconditional_variance
    
    for i in range(p):
        if i < len(last_squared_returns):
            future_squared_returns[i] = last_squared_returns[i]
        else:
            future_squared_returns[i] = unconditional_variance
    
    # Generate forecasts
    for h in range(steps):
        # Add constant term
        forecasts[h] = omega
        
        # Add ARCH terms (for h=1, these are known; for h>1, use unconditional variance)
        for i in range(p):
            if h + i < p:
                forecasts[h] += alpha[i] * future_squared_returns[h+i]
            else:
                forecasts[h] += alpha[i] * unconditional_variance
        
        # Add GARCH terms
        for j in range(q):
            if h + j < q:
                forecasts[h] += beta[j] * future_variances[h+j]
            else:
                idx = h + j - q
                if idx < len(forecasts):
                    forecasts[h] += beta[j] * forecasts[idx]
                else:
                    forecasts[h] += beta[j] * unconditional_variance
        
        # Update future values for next iteration
        future_variances[q + h] = forecasts[h]
        future_squared_returns[p + h] = forecasts[h]  # E[r²] = σ²
        
        # For long horizons, approach the unconditional variance
        if persistence < 1 and h > 100:
            forecasts[h] = unconditional_variance - (unconditional_variance - forecasts[h]) * 0.1
    
    return forecasts


# EGARCH model core functions
@jit(float64[:](float64[:], float64, float64, float64, float64, float64[:], float64), 
     nopython=True, cache=True)
def egarch_recursion(data, omega, alpha, gamma, beta, log_sigma2, backcast):
    """Compute EGARCH(1,1) conditional log-variances using Numba acceleration.
    
    This function implements the core recursion for the EGARCH(1,1) model:
    log(σ²_t) = ω + α * (|z_{t-1}| - E[|z|]) + γ * z_{t-1} + β * log(σ²_{t-1})
    
    where z_t = r_t/σ_t are standardized residuals, and E[|z|] = sqrt(2/π) for standard normal.
    
    Args:
        data: Input data (typically residuals)
        omega: Constant term in log-variance equation
        alpha: ARCH parameter
        gamma: Asymmetry parameter
        beta: GARCH parameter
        log_sigma2: Pre-allocated array for conditional log-variances
        backcast: Value to use for initializing the log-variance process
    
    Returns:
        np.ndarray: Conditional log-variances
    """
    T = len(data)
    
    # Initialize first log-variance with backcast value
    log_sigma2[0] = np.log(backcast)
    
    # Expected absolute value of standard normal
    expected_abs = np.sqrt(2.0 / np.pi)
    
    # Compute conditional log-variances recursively
    for t in range(1, T):
        # Standardized residual
        if log_sigma2[t-1] > -30:  # Avoid numerical underflow
            std_resid = data[t-1] / np.exp(0.5 * log_sigma2[t-1])
        else:
            std_resid = data[t-1] * 1000  # Large value for very small variance
        
        # Asymmetric term: |z_t| - E[|z_t|] + gamma * z_t
        abs_std_resid = np.abs(std_resid)
        asym_term = alpha * (abs_std_resid - expected_abs) + gamma * std_resid
        
        # EGARCH recursion
        log_sigma2[t] = omega + asym_term + beta * log_sigma2[t-1]
    
    return log_sigma2


@jit(float64[:](float64[:], float64, float64[:], float64[:], float64[:], float64[:], float64), 
     nopython=True, cache=True)
def egarch_p_q_recursion(data, omega, alpha, gamma, beta, log_sigma2, backcast):
    """Compute EGARCH(p,q) conditional log-variances using Numba acceleration.
    
    This function implements the core recursion for the EGARCH(p,q) model:
    log(σ²_t) = ω + Σ(α_i * (|z_{t-i}| - E[|z|]) + γ_i * z_{t-i}) + Σ(β_j * log(σ²_{t-j}))
    
    Args:
        data: Input data (typically residuals)
        omega: Constant term in log-variance equation
        alpha: ARCH parameters (array of length p)
        gamma: Asymmetry parameters (array of length p)
        beta: GARCH parameters (array of length q)
        log_sigma2: Pre-allocated array for conditional log-variances
        backcast: Value to use for initializing the log-variance process
    
    Returns:
        np.ndarray: Conditional log-variances
    """
    T = len(data)
    p = len(alpha)
    q = len(beta)
    
    # Initialize first max(p,q) log-variances with backcast value
    max_lag = max(p, q)
    for t in range(max_lag):
        log_sigma2[t] = np.log(backcast)
    
    # Expected absolute value of standard normal
    expected_abs = np.sqrt(2.0 / np.pi)
    
    # Compute conditional log-variances recursively
    for t in range(max_lag, T):
        # Add constant term
        log_sigma2[t] = omega
        
        # Add ARCH and asymmetry terms
        for i in range(p):
            if t - i - 1 >= 0:  # Ensure we don't go out of bounds
                # Standardized residual
                if log_sigma2[t-i-1] > -30:  # Avoid numerical underflow
                    std_resid = data[t-i-1] / np.exp(0.5 * log_sigma2[t-i-1])
                else:
                    std_resid = data[t-i-1] * 1000  # Large value for very small variance
                
                # Asymmetric term: |z_t| - E[|z_t|] + gamma * z_t
                abs_std_resid = np.abs(std_resid)
                asym_term = alpha[i] * (abs_std_resid - expected_abs) + gamma[i] * std_resid
                log_sigma2[t] += asym_term
        
        # Add GARCH terms
        for j in range(q):
            if t - j - 1 >= 0:  # Ensure we don't go out of bounds
                log_sigma2[t] += beta[j] * log_sigma2[t-j-1]
    
    return log_sigma2


@jit(float64[:](float64, float64, float64, float64, float64, int64), 
     nopython=True, cache=True)
def egarch_forecast(omega, alpha, gamma, beta, last_log_variance, steps):
    """Generate analytic forecasts for EGARCH(1,1) model using Numba acceleration.
    
    This function computes multi-step ahead forecasts for the EGARCH(1,1) model.
    
    Args:
        omega: Constant term in log-variance equation
        alpha: ARCH parameter
        gamma: Asymmetry parameter
        beta: GARCH parameter
        last_log_variance: Last observed conditional log-variance
        steps: Number of steps to forecast
    
    Returns:
        np.ndarray: Forecasted conditional variances (not log-variances)
    """
    forecasts = np.zeros(steps)
    log_forecasts = np.zeros(steps)
    
    # For EGARCH, E[|z_t| - E[|z_t|] + gamma * z_t] = 0 for future periods
    # So the forecast simplifies to omega + beta * log_sigma2[t-1]
    
    # First step forecast
    log_forecasts[0] = omega + beta * last_log_variance
    
    # Multi-step forecasts
    for h in range(1, steps):
        log_forecasts[h] = omega + beta * log_forecasts[h-1]
    
    # Convert log-variances to variances
    for h in range(steps):
        forecasts[h] = np.exp(log_forecasts[h])
    
    return forecasts


@jit(float64[:](float64, float64[:], float64[:], float64[:], float64[:], int64), 
     nopython=True, cache=True)
def egarch_p_q_forecast(omega, alpha, gamma, beta, last_log_variances, steps):
    """Generate analytic forecasts for EGARCH(p,q) model using Numba acceleration.
    
    This function computes multi-step ahead forecasts for the EGARCH(p,q) model.
    
    Args:
        omega: Constant term in log-variance equation
        alpha: ARCH parameters (array of length p)
        gamma: Asymmetry parameters (array of length p)
        beta: GARCH parameters (array of length q)
        last_log_variances: Last q observed conditional log-variances (most recent first)
        steps: Number of steps to forecast
    
    Returns:
        np.ndarray: Forecasted conditional variances (not log-variances)
    """
    forecasts = np.zeros(steps)
    log_forecasts = np.zeros(steps)
    q = len(beta)
    
    # Initialize arrays for multi-step forecasting
    future_log_variances = np.zeros(q + steps)
    
    # Fill in known values
    for i in range(q):
        if i < len(last_log_variances):
            future_log_variances[i] = last_log_variances[i]
    
    # Generate forecasts
    for h in range(steps):
        # Add constant term
        log_forecasts[h] = omega
        
        # For EGARCH, E[|z_t| - E[|z_t|] + gamma * z_t] = 0 for future periods
        # So we only need to add the GARCH terms
        
        # Add GARCH terms
        for j in range(q):
            if h + j < q:
                log_forecasts[h] += beta[j] * future_log_variances[h+j]
            else:
                idx = h + j - q
                if idx < len(log_forecasts):
                    log_forecasts[h] += beta[j] * log_forecasts[idx]
        
        # Update future values for next iteration
        future_log_variances[q + h] = log_forecasts[h]
        
        # Convert log-variance to variance
        forecasts[h] = np.exp(log_forecasts[h])
    
    return forecasts


# TARCH model core functions
@jit(float64[:](float64[:], float64, float64, float64, float64, float64[:], float64, int64), 
     nopython=True, cache=True)
def tarch_recursion(data, omega, alpha, gamma, beta, sigma2, backcast, tarch_type=1):
    """Compute TARCH(1,1) conditional variances using Numba acceleration.
    
    This function implements the core recursion for the TARCH(1,1) model:
    σ²_t = ω + α * r²_{t-1} + γ * I_{t-1} * r²_{t-1} + β * σ²_{t-1}
    
    where I_{t-1} is an indicator function that equals 1 if r_{t-1} < 0 and 0 otherwise.
    
    Args:
        data: Input data (typically residuals)
        omega: Constant term in variance equation
        alpha: ARCH parameter
        gamma: Asymmetry parameter
        beta: GARCH parameter
        sigma2: Pre-allocated array for conditional variances
        backcast: Value to use for initializing the variance process
        tarch_type: Type of TARCH model (1 for squared innovations, 2 for absolute innovations)
    
    Returns:
        np.ndarray: Conditional variances
    """
    T = len(data)
    
    # Initialize first variance with backcast value
    sigma2[0] = backcast
    
    # Compute conditional variances recursively
    for t in range(1, T):
        # Determine if previous return was negative
        neg_indicator = 1.0 if data[t-1] < 0 else 0.0
        
        if tarch_type == 1:  # Squared innovations (GJR-GARCH)
            # GJR-GARCH formulation with squared innovations
            sigma2[t] = omega + alpha * data[t-1]**2 + gamma * neg_indicator * data[t-1]**2 + beta * sigma2[t-1]
        else:  # Absolute innovations (Zakoian's TARCH)
            # Original TARCH formulation with absolute innovations
            # Note: This returns conditional standard deviation, not variance
            sigma = np.sqrt(sigma2[t-1])
            sigma2[t] = (omega + alpha * abs(data[t-1]) + gamma * neg_indicator * abs(data[t-1]) + beta * sigma)**2
    
    return sigma2


@jit(float64[:](float64, float64, float64, float64, float64, int64, int64), 
     nopython=True, cache=True)
def tarch_forecast(omega, alpha, gamma, beta, last_variance, steps, tarch_type=1):
    """Generate analytic forecasts for TARCH(1,1) model using Numba acceleration.
    
    This function computes multi-step ahead forecasts for the TARCH(1,1) model.
    
    Args:
        omega: Constant term in variance equation
        alpha: ARCH parameter
        gamma: Asymmetry parameter
        beta: GARCH parameter
        last_variance: Last observed conditional variance
        steps: Number of steps to forecast
        tarch_type: Type of TARCH model (1 for squared innovations, 2 for absolute innovations)
    
    Returns:
        np.ndarray: Forecasted conditional variances
    """
    forecasts = np.zeros(steps)
    
    # For GJR-GARCH (tarch_type=1), the expected value of the asymmetry term is gamma/2
    # since negative returns occur approximately half the time
    if tarch_type == 1:
        # Compute effective persistence
        persistence = alpha + beta + gamma / 2
        
        # Compute unconditional variance
        unconditional_variance = omega / (1 - persistence) if persistence < 1 else last_variance
        
        # First step forecast
        forecasts[0] = omega + (alpha + gamma / 2) * last_variance + beta * last_variance
        
        # Multi-step forecasts
        for h in range(1, steps):
            forecasts[h] = omega + persistence * forecasts[h-1]
            
            # For long horizons, approach the unconditional variance
            if persistence < 1 and h > 100:
                forecasts[h] = unconditional_variance - (unconditional_variance - forecasts[h]) * 0.1
    else:
        # For Zakoian's TARCH (tarch_type=2), the forecasting is more complex
        # This is a simplified approximation
        # Convert last_variance to standard deviation
        last_std = np.sqrt(last_variance)
        
        # Expected value of |z| for standard normal is sqrt(2.0 / np.pi)
        expected_abs_z = np.sqrt(2.0 / np.pi)
        
        # Compute effective persistence
        persistence = beta + (alpha + gamma / 2) * expected_abs_z
        
        # Compute unconditional standard deviation
        unconditional_std = omega / (1 - persistence) if persistence < 1 else last_std
        unconditional_variance = unconditional_std**2
        
        # First step forecast (standard deviation)
        next_std = omega + (alpha + gamma / 2) * expected_abs_z * last_std + beta * last_std
        forecasts[0] = next_std**2  # Convert to variance
        
        # Multi-step forecasts
        for h in range(1, steps):
            next_std = omega + persistence * np.sqrt(forecasts[h-1])
            forecasts[h] = next_std**2  # Convert to variance
            
            # For long horizons, approach the unconditional variance
            if persistence < 1 and h > 100:
                forecasts[h] = unconditional_variance - (unconditional_variance - forecasts[h]) * 0.1
    
    return forecasts


# APARCH model core functions
@jit(float64[:](float64[:], float64, float64, float64, float64, float64, float64[:], float64), 
     nopython=True, cache=True)
def aparch_recursion(data, omega, alpha, gamma, beta, delta, sigma_delta, backcast):
    """Compute APARCH(1,1) conditional power variances using Numba acceleration.
    
    This function implements the core recursion for the APARCH(1,1) model:
    σ^δ_t = ω + α * (|r_{t-1}| - γ * r_{t-1})^δ + β * σ^δ_{t-1}
    
    Args:
        data: Input data (typically residuals)
        omega: Constant term in power variance equation
        alpha: ARCH parameter
        gamma: Asymmetry parameter (-1 < gamma < 1)
        beta: GARCH parameter
        delta: Power parameter (delta > 0)
        sigma_delta: Pre-allocated array for conditional power variances
        backcast: Value to use for initializing the power variance process
    
    Returns:
        np.ndarray: Conditional power variances (σ^δ)
    """
    T = len(data)
    
    # Initialize first power variance with backcast value
    sigma_delta[0] = backcast
    
    # Compute conditional power variances recursively
    for t in range(1, T):
        # Asymmetric absolute return term: (|r_{t-1}| - γ * r_{t-1})^δ
        abs_return = abs(data[t-1])
        asym_term = (abs_return - gamma * data[t-1])**delta
        
        # APARCH recursion
        sigma_delta[t] = omega + alpha * asym_term + beta * sigma_delta[t-1]
    
    return sigma_delta


@jit(float64[:](float64, float64, float64, float64, float64, float64, int64), 
     nopython=True, cache=True)
def aparch_forecast(omega, alpha, gamma, beta, delta, last_power_variance, steps):
    """Generate analytic forecasts for APARCH(1,1) model using Numba acceleration.
    
    This function computes multi-step ahead forecasts for the APARCH(1,1) model.
    
    Args:
        omega: Constant term in power variance equation
        alpha: ARCH parameter
        gamma: Asymmetry parameter (-1 < gamma < 1)
        beta: GARCH parameter
        delta: Power parameter (delta > 0)
        last_power_variance: Last observed conditional power variance
        steps: Number of steps to forecast
    
    Returns:
        np.ndarray: Forecasted conditional variances (not power variances)
    """
    power_forecasts = np.zeros(steps)
    forecasts = np.zeros(steps)
    
    # For APARCH, we need to compute E[(|z| - γ*z)^δ]
    # For standard normal z, this expectation can be approximated
    # For γ = 0, it's just E[|z|^δ]
    # For general γ, it's more complex
    
    # Approximate the expected value of (|z| - γ*z)^δ for standard normal z
    # This is a simplified approximation
    expected_term = 1.0
    if abs(gamma) > 0:
        # More accurate approximation would require numerical integration
        # This is a rough approximation that works reasonably well for typical values
        expected_term = (1 + gamma**2) ** (delta/2)
    
    # Compute effective persistence
    persistence = alpha * expected_term + beta
    
    # Compute unconditional power variance
    unconditional_power_var = omega / (1 - persistence) if persistence < 1 else last_power_variance
    
    # First step forecast (power variance)
    power_forecasts[0] = omega + alpha * expected_term * last_power_variance + beta * last_power_variance
    
    # Multi-step forecasts (power variance)
    for h in range(1, steps):
        power_forecasts[h] = omega + persistence * power_forecasts[h-1]
        
        # For long horizons, approach the unconditional power variance
        if persistence < 1 and h > 100:
            power_forecasts[h] = unconditional_power_var - (unconditional_power_var - power_forecasts[h]) * 0.1
    
    # Convert power variances to regular variances
    for h in range(steps):
        forecasts[h] = power_forecasts[h] ** (2.0 / delta)
    
    return forecasts


# FIGARCH model core functions
@jit(float64[:](float64[:], float64, float64, float64, float64, float64[:], float64, int64), 
     nopython=True, cache=True)
def figarch_recursion(data, omega, phi, d, beta, sigma2, backcast, truncation):
    """Compute FIGARCH(1,d,1) conditional variances using Numba acceleration.
    
    This function implements the core recursion for the FIGARCH(1,d,1) model:
    σ²_t = ω + [1 - (1-βL)^(-1) * (1-φL) * (1-L)^d] * r²_t + β * σ²_{t-1}
    
    where L is the lag operator, d is the fractional differencing parameter,
    and the infinite ARCH representation is truncated at 'truncation' lags.
    
    Args:
        data: Input data (typically residuals)
        omega: Constant term in variance equation
        phi: ARCH parameter
        d: Fractional differencing parameter (0 < d < 1)
        beta: GARCH parameter
        sigma2: Pre-allocated array for conditional variances
        backcast: Value to use for initializing the variance process
        truncation: Number of lags to use in the truncated ARCH representation
    
    Returns:
        np.ndarray: Conditional variances
    """
    T = len(data)
    
    # Compute the weights for the truncated ARCH representation
    # λ_k = [Γ(k-d) / (Γ(k+1) * Γ(-d))] * φ - [Γ(k-d+1) / (Γ(k+1) * Γ(-d+1))] * β
    # where Γ is the gamma function
    
    # For numerical stability, we compute the weights recursively
    lambda_weights = np.zeros(truncation + 1)
    
    # First weight
    lambda_weights[0] = phi - beta
    
    # Recursive computation of weights
    for k in range(1, truncation + 1):
        lambda_weights[k] = ((k - 1 - d) / k) * lambda_weights[k-1]
        if k == 1:
            lambda_weights[k] += d
    
    # Initialize first variance with backcast value
    sigma2[0] = backcast
    
    # Compute conditional variances recursively
    for t in range(1, T):
        # Add constant term
        sigma2[t] = omega
        
        # Add ARCH terms with fractional weights
        for k in range(1, min(t + 1, truncation + 1)):
            if t - k >= 0:  # Ensure we don't go out of bounds
                sigma2[t] += lambda_weights[k] * data[t-k]**2
        
        # Add GARCH term
        sigma2[t] += beta * sigma2[t-1]
    
    return sigma2


@jit(float64[:](float64, float64, float64, float64, float64, int64, int64), 
     nopython=True, cache=True)
def figarch_forecast(omega, phi, d, beta, last_variance, steps, truncation):
    """Generate analytic forecasts for FIGARCH(1,d,1) model using Numba acceleration.
    
    This function computes multi-step ahead forecasts for the FIGARCH(1,d,1) model.
    
    Args:
        omega: Constant term in variance equation
        phi: ARCH parameter
        d: Fractional differencing parameter (0 < d < 1)
        beta: GARCH parameter
        last_variance: Last observed conditional variance
        steps: Number of steps to forecast
        truncation: Number of lags to use in the truncated ARCH representation
    
    Returns:
        np.ndarray: Forecasted conditional variances
    """
    forecasts = np.zeros(steps)
    
    # Compute the weights for the truncated ARCH representation
    lambda_weights = np.zeros(truncation + 1)
    
    # First weight
    lambda_weights[0] = phi - beta
    
    # Recursive computation of weights
    for k in range(1, truncation + 1):
        lambda_weights[k] = ((k - 1 - d) / k) * lambda_weights[k-1]
        if k == 1:
            lambda_weights[k] += d
    
    # Compute the sum of weights (needed for long-term forecasting)
    sum_weights = np.sum(lambda_weights[1:])
    
    # For FIGARCH, the unconditional variance is not well-defined
    # We use a large value as a proxy for the long-term forecast
    long_term_var = omega / (1 - sum_weights - beta) if sum_weights + beta < 1 else last_variance
    
    # First step forecast
    forecasts[0] = omega + sum_weights * last_variance + beta * last_variance
    
    # Multi-step forecasts
    for h in range(1, steps):
        forecasts[h] = omega + (sum_weights + beta) * forecasts[h-1]
        
        # For long horizons, approach the long-term variance
        if sum_weights + beta < 1 and h > 100:
            forecasts[h] = long_term_var - (long_term_var - forecasts[h]) * 0.1
    
    return forecasts


# Utility functions for volatility models
@jit(float64[:](float64[:], float64[:], float64[:], float64), 
     nopython=True, cache=True)
def backcast_weighted(data_squared, weights, sigma2, default_value):
    """Compute weighted backcast value for initializing variance processes.
    
    This function computes a weighted average of past squared returns to use
    as an initial value for the variance process.
    
    Args:
        data_squared: Squared returns
        weights: Weights for the weighted average
        sigma2: Pre-allocated array for conditional variances
        default_value: Default value to use if computation fails
    
    Returns:
        np.ndarray: Conditional variances with initialized first value
    """
    T = len(data_squared)
    n_weights = len(weights)
    
    if T < n_weights:
        # Not enough data for weighted backcast, use default
        sigma2[0] = default_value
    else:
        # Compute weighted average of past squared returns
        backcast = 0.0
        for i in range(n_weights):
            backcast += weights[i] * data_squared[i]
        
        sigma2[0] = backcast
    
    return sigma2


@jit(float64[:](float64, int64, boolean), 
     nopython=True, cache=True)
def compute_exponential_weights(decay_factor, n_weights, normalize=True):
    """Compute exponentially decaying weights for backcast calculation.
    
    This function computes weights that decay exponentially, which can be
    used for initializing variance processes with more weight on recent observations.
    
    Args:
        decay_factor: Rate of decay (0 < decay_factor < 1)
        n_weights: Number of weights to compute
        normalize: Whether to normalize weights to sum to 1
    
    Returns:
        np.ndarray: Array of exponentially decaying weights
    """
    weights = np.zeros(n_weights)
    
    # Compute exponentially decaying weights
    for i in range(n_weights):
        weights[i] = decay_factor ** i
    
    # Normalize weights to sum to 1 if requested
    if normalize and np.sum(weights) > 0:
        weights = weights / np.sum(weights)
    
    return weights


@jit(float64[:](float64[:], float64, float64, float64, float64[:], float64), 
     nopython=True, cache=True)
def simulate_garch_path(innovations, omega, alpha, beta, sigma2, initial_variance):
    """Simulate a path from a GARCH(p,q) model.
    
    This function simulates a path from a GARCH(p,q) model given innovations
    and model parameters.
    
    Args:
        innovations: Random innovations (e.g., from normal distribution)
        omega: Constant term in variance equation
        alpha: ARCH parameters (array of length p)
        beta: GARCH parameters (array of length q)
        sigma2: Pre-allocated array for conditional variances
        initial_variance: Initial variance value
    
    Returns:
        np.ndarray: Simulated conditional variances
    """
    T = len(innovations)
    
    # Initialize first variance with initial value
    sigma2[0] = initial_variance
    
    # Generate path
    for t in range(1, T):
        # GARCH recursion
        sigma2[t] = omega + alpha * innovations[t-1]**2 + beta * sigma2[t-1]
    
    return sigma2


@jit(float64[:](float64[:], float64, float64, float64, float64, float64[:], float64), 
     nopython=True, cache=True)
def simulate_tarch_path(innovations, omega, alpha, gamma, beta, sigma2, initial_variance):
    """Simulate a path from a TARCH(1,1) model.
    
    This function simulates a path from a TARCH(1,1) model given innovations
    and model parameters.
    
    Args:
        innovations: Random innovations (e.g., from normal distribution)
        omega: Constant term in variance equation
        alpha: ARCH parameter
        gamma: Asymmetry parameter
        beta: GARCH parameter
        sigma2: Pre-allocated array for conditional variances
        initial_variance: Initial variance value
    
    Returns:
        np.ndarray: Simulated conditional variances
    """
    T = len(innovations)
    
    # Initialize first variance with initial value
    sigma2[0] = initial_variance
    
    # Generate path
    for t in range(1, T):
        # Determine if previous innovation was negative
        neg_indicator = 1.0 if innovations[t-1] < 0 else 0.0
        
        # TARCH recursion
        sigma2[t] = omega + alpha * innovations[t-1]**2 + gamma * neg_indicator * innovations[t-1]**2 + beta * sigma2[t-1]
    
    return sigma2


@jit(float64[:](float64[:], float64, float64, float64, float64, float64[:], float64), 
     nopython=True, cache=True)
def simulate_egarch_path(innovations, omega, alpha, gamma, beta, log_sigma2, initial_log_variance):
    """Simulate a path from an EGARCH(1,1) model.
    
    This function simulates a path from an EGARCH(1,1) model given innovations
    and model parameters.
    
    Args:
        innovations: Random innovations (e.g., from normal distribution)
        omega: Constant term in log-variance equation
        alpha: ARCH parameter
        gamma: Asymmetry parameter
        beta: GARCH parameter
        log_sigma2: Pre-allocated array for conditional log-variances
        initial_log_variance: Initial log-variance value
    
    Returns:
        np.ndarray: Simulated conditional log-variances
    """
    T = len(innovations)
    
    # Initialize first log-variance with initial value
    log_sigma2[0] = initial_log_variance
    
    # Expected absolute value of standard normal
    expected_abs = np.sqrt(2.0 / np.pi)
    
    # Generate path
    for t in range(1, T):
        # Standardized innovation
        std_innov = innovations[t-1] / np.exp(0.5 * log_sigma2[t-1])
        
        # Asymmetric term
        abs_std_innov = np.abs(std_innov)
        asym_term = alpha * (abs_std_innov - expected_abs) + gamma * std_innov
        
        # EGARCH recursion
        log_sigma2[t] = omega + asym_term + beta * log_sigma2[t-1]
    
    return log_sigma2
