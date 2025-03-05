"""
EGARCH (Exponential GARCH) model implementation.

This module provides a comprehensive implementation of the EGARCH(p,q) model for
modeling time-varying volatility in financial time series. The EGARCH model uses
log-variance dynamics to ensure positive variance without parameter constraints,
and can capture asymmetric effects in volatility response to positive and negative
shocks.

The implementation includes parameter estimation via maximum likelihood,
simulation, forecasting, and diagnostic tools. Performance-critical operations
are accelerated using Numba's just-in-time compilation.

References:
    Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns: A new
    approach. Econometrica, 59(2), 347-370.
"""

import asyncio
import warnings
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union, cast, overload
)
import numpy as np
import pandas as pd
from scipy import optimize, stats
from numba import jit

from mfe.core.base import ModelBase, VolatilityModelBase
from mfe.core.parameters import (
    EGARCHParameters, UnivariateVolatilityParameters, ParameterError,
    validate_positive, validate_non_negative, validate_probability, validate_range
)
from mfe.core.exceptions import (
    MFEError, ConvergenceError, NumericError, EstimationError, SimulationError,
    NotFittedError, raise_convergence_error, raise_numeric_error, raise_not_fitted_error,
    warn_convergence, warn_numeric, warn_model
)
from mfe.models.univariate.base import VolatilityModel


# Numba-accelerated core functions for EGARCH model
@jit(nopython=True, cache=True)
def _egarch_recursion(data: np.ndarray, 
                     omega: float, 
                     alpha: float, 
                     gamma: float,
                     beta: float, 
                     log_sigma2: np.ndarray,
                     backcast: float) -> np.ndarray:
    """Compute EGARCH(1,1) conditional log-variances using Numba acceleration.
    
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
    
    # Compute conditional log-variances recursively
    for t in range(1, T):
        # Standardized residual
        if log_sigma2[t-1] > -30:  # Avoid numerical underflow
            std_resid = data[t-1] / np.exp(0.5 * log_sigma2[t-1])
        else:
            std_resid = data[t-1] * 1000  # Large value for very small variance
        
        # Asymmetric term: |z_t| - E[|z_t|] + gamma * z_t
        # For standard normal, E[|z_t|] = sqrt(2/pi) ≈ 0.7979
        abs_std_resid = np.abs(std_resid)
        expected_abs = np.sqrt(2.0 / np.pi)
        asym_term = alpha * (abs_std_resid - expected_abs) + gamma * std_resid
        
        # EGARCH recursion
        log_sigma2[t] = omega + asym_term + beta * log_sigma2[t-1]
    
    return log_sigma2


@jit(nopython=True, cache=True)
def _egarch_p_q_recursion(data: np.ndarray,
                          omega: float,
                          alpha: np.ndarray,
                          gamma: np.ndarray,
                          beta: np.ndarray,
                          log_sigma2: np.ndarray,
                          backcast: float) -> np.ndarray:
    """Compute EGARCH(p,q) conditional log-variances using Numba acceleration.
    
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


@jit(nopython=True, cache=True)
def _egarch_forecast(omega: float, 
                    alpha: float, 
                    gamma: float,
                    beta: float, 
                    last_log_variance: float, 
                    steps: int) -> np.ndarray:
    """Generate analytic forecasts for EGARCH(1,1) model using Numba acceleration.
    
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
    
    # Expected absolute value of standard normal
    expected_abs = np.sqrt(2.0 / np.pi)
    
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


@jit(nopython=True, cache=True)
def _egarch_p_q_forecast(omega: float,
                         alpha: np.ndarray,
                         gamma: np.ndarray,
                         beta: np.ndarray,
                         last_log_variances: np.ndarray,
                         steps: int) -> np.ndarray:
    """Generate analytic forecasts for EGARCH(p,q) model using Numba acceleration.
    
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
    p = len(alpha)
    q = len(beta)
    
    # Expected absolute value of standard normal
    expected_abs = np.sqrt(2.0 / np.pi)
    
    # Initialize arrays for multi-step forecasting
    future_log_variances = np.zeros(max(p, q) + steps)
    
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


@dataclass
class EGARCHParams(EGARCHParameters):
    """Parameters for EGARCH(p,q) model.
    
    This class extends the base EGARCHParameters class to support higher-order
    EGARCH models with multiple ARCH, asymmetry, and GARCH terms.
    
    Attributes:
        omega: Constant term in log-variance equation
        alpha: ARCH parameters
        gamma: Asymmetry parameters
        beta: GARCH parameters
    """
    
    omega: float
    alpha: Union[float, np.ndarray]
    gamma: Union[float, np.ndarray]
    beta: Union[float, np.ndarray]
    
    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        # Convert alpha, gamma, and beta to arrays if they are scalars
        if isinstance(self.alpha, (int, float)):
            self.alpha = np.array([self.alpha])
        elif not isinstance(self.alpha, np.ndarray):
            self.alpha = np.array(self.alpha)
            
        if isinstance(self.gamma, (int, float)):
            self.gamma = np.array([self.gamma])
        elif not isinstance(self.gamma, np.ndarray):
            self.gamma = np.array(self.gamma)
            
        if isinstance(self.beta, (int, float)):
            self.beta = np.array([self.beta])
        elif not isinstance(self.beta, np.ndarray):
            self.beta = np.array(self.beta)
        
        self.validate()
    
    def validate(self) -> None:
        """Validate EGARCH parameter constraints.
        
        Raises:
            ParameterError: If parameter constraints are violated
        """
        # EGARCH has fewer constraints than GARCH
        # The key constraint is |beta| < 1 for stationarity
        for i, b in enumerate(self.beta):
            if abs(b) >= 1:
                raise ParameterError(
                    f"EGARCH stationarity constraint violated: |beta[{i}]| = {abs(b)} >= 1"
                )
    
    def to_array(self) -> np.ndarray:
        """Convert parameters to a NumPy array.
        
        Returns:
            np.ndarray: Array representation of parameters
        """
        return np.concatenate([
            np.array([self.omega]),
            self.alpha,
            self.gamma,
            self.beta
        ])
    
    @classmethod
    def from_array(cls, array: np.ndarray, p: int = 1, q: int = 1, **kwargs: Any) -> 'EGARCHParams':
        """Create parameters from a NumPy array.
        
        Args:
            array: Array representation of parameters
            p: Order of the ARCH component
            q: Order of the GARCH component
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            EGARCHParams: Parameter object
        
        Raises:
            ValueError: If the array length doesn't match the expected length
        """
        expected_length = 1 + p + p + q  # omega, alpha, gamma, beta
        if len(array) != expected_length:
            raise ValueError(
                f"Array length ({len(array)}) doesn't match expected length ({expected_length})"
            )
        
        # Extract parameters
        omega = array[0]
        alpha = array[1:1+p]
        gamma = array[1+p:1+2*p]
        beta = array[1+2*p:]
        
        return cls(omega=omega, alpha=alpha, gamma=gamma, beta=beta)
    
    def transform(self) -> np.ndarray:
        """Transform parameters to unconstrained space for optimization.
        
        Returns:
            np.ndarray: Transformed parameters in unconstrained space
        """
        from mfe.core.parameters import transform_correlation
        
        # omega, alpha, and gamma have no constraints, so they remain unchanged
        transformed_omega = self.omega
        transformed_alpha = self.alpha.copy()
        transformed_gamma = self.gamma.copy()
        
        # Transform beta to ensure |beta| < 1 using Fisher's z-transformation
        transformed_beta = np.zeros_like(self.beta)
        for i, b in enumerate(self.beta):
            transformed_beta[i] = transform_correlation(b)
        
        # Combine all transformed parameters
        return np.concatenate([
            np.array([transformed_omega]),
            transformed_alpha,
            transformed_gamma,
            transformed_beta
        ])
    
    @classmethod
    def inverse_transform(cls, array: np.ndarray, p: int = 1, q: int = 1, **kwargs: Any) -> 'EGARCHParams':
        """Transform parameters from unconstrained space back to constrained space.
        
        Args:
            array: Parameters in unconstrained space
            p: Order of the ARCH component
            q: Order of the GARCH component
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            EGARCHParams: Parameter object with constrained parameters
        
        Raises:
            ValueError: If the array length doesn't match the expected length
        """
        from mfe.core.parameters import inverse_transform_correlation
        
        expected_length = 1 + p + p + q  # omega, alpha, gamma, beta
        if len(array) != expected_length:
            raise ValueError(
                f"Array length ({len(array)}) doesn't match expected length ({expected_length})"
            )
        
        # Extract transformed parameters
        transformed_omega = array[0]
        transformed_alpha = array[1:1+p]
        transformed_gamma = array[1+p:1+2*p]
        transformed_beta = array[1+2*p:]
        
        # omega, alpha, and gamma have no constraints
        omega = transformed_omega
        alpha = transformed_alpha
        gamma = transformed_gamma
        
        # Inverse transform beta using inverse of Fisher's z-transformation
        beta = np.zeros_like(transformed_beta)
        for i, b in enumerate(transformed_beta):
            beta[i] = inverse_transform_correlation(b)
        
        return cls(omega=omega, alpha=alpha, gamma=gamma, beta=beta)


class EGARCH(VolatilityModel):
    """EGARCH (Exponential GARCH) model for volatility modeling.
    
    This class implements the EGARCH(p,q) model for modeling time-varying volatility
    in financial time series. The model is defined as:
    
    log(σ²_t) = ω + Σ(α_i * (|z_{t-i}| - E[|z|]) + γ_i * z_{t-i}) + Σ(β_j * log(σ²_{t-j}))
    
    where σ²_t is the conditional variance at time t, z_t = r_t/σ_t are standardized
    residuals, and E[|z|] = sqrt(2/π) for standard normal innovations.
    
    The EGARCH model uses log-variance dynamics to ensure positive variance without
    parameter constraints, and can capture asymmetric effects in volatility response
    to positive and negative shocks.
    """
    
    def __init__(self, 
                 p: int = 1, 
                 q: int = 1, 
                 parameters: Optional[EGARCHParams] = None, 
                 name: str = "EGARCH") -> None:
        """Initialize the EGARCH model.
        
        Args:
            p: Order of the ARCH component
            q: Order of the GARCH component
            parameters: Pre-specified model parameters if available
            name: A descriptive name for the model
        
        Raises:
            ValueError: If p or q is not a positive integer
        """
        if not isinstance(p, int) or p <= 0:
            raise ValueError(f"p must be a positive integer, got {p}")
        if not isinstance(q, int) or q <= 0:
            raise ValueError(f"q must be a positive integer, got {q}")
        
        self.p = p
        self.q = q
        
        # Set model name based on p and q
        if name == "EGARCH":
            name = f"EGARCH({p},{q})"
        
        super().__init__(parameters=parameters, name=name)
        
        # Initialize additional attributes
        self._conditional_variances = None
        self._log_conditional_variances = None
    
    def parameter_class(self) -> Type[EGARCHParams]:
        """Get the parameter class for this model.
        
        Returns:
            Type[EGARCHParams]: The parameter class for this model
        """
        return EGARCHParams
    
    def compute_variance(self, 
                         parameters: EGARCHParams, 
                         data: np.ndarray, 
                         sigma2: Optional[np.ndarray] = None,
                         backcast: Optional[float] = None) -> np.ndarray:
        """Compute conditional variances for the given parameters and data.
        
        Args:
            parameters: Model parameters
            data: Input data (typically residuals)
            sigma2: Pre-allocated array for conditional variances
            backcast: Value to use for initializing the variance process
        
        Returns:
            np.ndarray: Conditional variances
        
        Raises:
            ValueError: If the data is invalid
            NumericError: If numerical issues occur during computation
        """
        # Validate input data
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        
        if data.ndim != 1:
            raise ValueError(f"data must be a 1-dimensional array, got shape {data.shape}")
        
        T = len(data)
        
        # Allocate array for conditional variances if not provided
        if sigma2 is None:
            sigma2 = np.zeros(T)
        elif len(sigma2) != T:
            raise ValueError(f"sigma2 must have length {T}, got {len(sigma2)}")
        
        # Allocate array for log-variances
        log_sigma2 = np.zeros(T)
        
        # Compute backcast value if not provided
        if backcast is None:
            backcast = np.mean(data**2)
        
        try:
            # Use optimized Numba implementation
            if self.p == 1 and self.q == 1:
                # EGARCH(1,1) case
                omega = parameters.omega
                alpha = parameters.alpha[0]
                gamma = parameters.gamma[0]
                beta = parameters.beta[0]
                
                log_sigma2 = _egarch_recursion(data, omega, alpha, gamma, beta, log_sigma2, backcast)
            else:
                # EGARCH(p,q) case
                omega = parameters.omega
                alpha = parameters.alpha
                gamma = parameters.gamma
                beta = parameters.beta
                
                log_sigma2 = _egarch_p_q_recursion(data, omega, alpha, gamma, beta, log_sigma2, backcast)
            
            # Convert log-variances to variances
            for t in range(T):
                sigma2[t] = np.exp(log_sigma2[t])
            
            # Store log-variances for later use
            self._log_conditional_variances = log_sigma2
            
            # Check for numerical issues
            if np.any(~np.isfinite(sigma2)) or np.any(sigma2 <= 0):
                raise_numeric_error(
                    "Numerical issues detected in variance computation",
                    operation="EGARCH variance computation",
                    error_type="Invalid variance values",
                    details="Non-finite or non-positive variance values detected"
                )
            
            return sigma2
            
        except NumericError as e:
            # Re-raise numeric errors
            raise e
        except Exception as e:
            # Wrap other exceptions in NumericError
            raise_numeric_error(
                f"Error during variance computation: {str(e)}",
                operation="EGARCH variance computation",
                error_type="Computation error",
                details=str(e)
            )
    
    def _generate_starting_values(self, 
                                 data: np.ndarray, 
                                 variance_targeting: bool = False,
                                 backcast: Optional[float] = None) -> EGARCHParams:
        """Generate starting values for parameter estimation.
        
        Args:
            data: Input data (typically residuals)
            variance_targeting: Whether to use variance targeting
            backcast: Value to use for initializing the variance process
        
        Returns:
            EGARCHParams: Starting parameter values
        """
        # Compute sample variance
        sample_variance = np.var(data)
        
        if backcast is None:
            backcast = sample_variance
        
        # Generate starting values based on model order
        if self.p == 1 and self.q == 1:
            # EGARCH(1,1) case
            # For EGARCH, omega is related to the unconditional variance differently
            # log(σ²) = ω / (1 - β) in the stationary case
            omega = 0.0  # Start with zero
            alpha = 0.1  # Modest ARCH effect
            gamma = -0.1  # Slight leverage effect (negative shocks increase volatility more)
            beta = 0.9   # High persistence
            
            return EGARCHParams(omega=omega, alpha=alpha, gamma=gamma, beta=beta)
        else:
            # EGARCH(p,q) case
            # Distribute alpha, gamma, and beta values
            alpha_total = 0.1
            gamma_total = -0.1
            beta_total = 0.9
            
            alpha = np.zeros(self.p)
            gamma = np.zeros(self.p)
            beta = np.zeros(self.q)
            
            # Distribute alpha values with exponential decay
            if self.p > 0:
                alpha_weights = np.exp(-np.arange(self.p))
                alpha_weights = alpha_weights / np.sum(alpha_weights)
                alpha = alpha_weights * alpha_total
            
            # Distribute gamma values with exponential decay
            if self.p > 0:
                gamma_weights = np.exp(-np.arange(self.p))
                gamma_weights = gamma_weights / np.sum(gamma_weights)
                gamma = gamma_weights * gamma_total
            
            # Distribute beta values with exponential decay
            if self.q > 0:
                beta_weights = np.exp(-np.arange(self.q))
                beta_weights = beta_weights / np.sum(beta_weights)
                beta = beta_weights * beta_total
            
            # For EGARCH, omega is not directly related to unconditional variance
            omega = 0.0
            
            return EGARCHParams(omega=omega, alpha=alpha, gamma=gamma, beta=beta)
    
    def _compute_unconditional_variance(self) -> float:
        """Compute the unconditional variance of the process.
        
        Returns:
            float: Unconditional variance
        
        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self._fitted or self._parameters is None:
            raise_not_fitted_error(
                "Model must be fitted before computing unconditional variance.",
                model_type=self.name,
                operation="compute_unconditional_variance"
            )
        
        # For EGARCH model, unconditional variance is exp(omega / (1 - sum(beta)))
        omega = self._parameters.omega
        beta_sum = np.sum(self._parameters.beta)
        
        if abs(beta_sum) >= 1:
            # If the process is not stationary, return the last observed variance
            if self._conditional_variances is not None:
                return self._conditional_variances[-1]
            else:
                # Fallback to a reasonable value
                return np.exp(omega / 0.1)
        
        # Compute the unconditional log-variance
        unconditional_log_var = omega / (1 - beta_sum)
        
        # Convert to variance
        return np.exp(unconditional_log_var)
    
    def _simulate_variance(self, 
                          t: int, 
                          innovations: np.ndarray, 
                          sigma2: np.ndarray) -> float:
        """Simulate the conditional variance for time t.
        
        Args:
            t: Time index
            innovations: Past innovations up to t-1
            sigma2: Past conditional variances up to t-1
        
        Returns:
            float: Conditional variance for time t
        
        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self._fitted or self._parameters is None:
            raise_not_fitted_error(
                "Model must be fitted before simulation.",
                model_type=self.name,
                operation="simulate_variance"
            )
        
        # Extract parameters
        omega = self._parameters.omega
        alpha = self._parameters.alpha
        gamma = self._parameters.gamma
        beta = self._parameters.beta
        
        # Expected absolute value of standard normal
        expected_abs = np.sqrt(2.0 / np.pi)
        
        # Compute log-variance for time t
        log_variance = omega
        
        # Add ARCH and asymmetry terms
        for i in range(len(alpha)):
            if t - i - 1 >= 0:  # Ensure we don't go out of bounds
                # Standardized residual
                std_resid = innovations[t-i-1] / np.sqrt(sigma2[t-i-1])
                
                # Asymmetric term: |z_t| - E[|z_t|] + gamma * z_t
                abs_std_resid = np.abs(std_resid)
                asym_term = alpha[i] * (abs_std_resid - expected_abs) + gamma[i] * std_resid
                log_variance += asym_term
        
        # Add GARCH terms
        for j in range(len(beta)):
            if t - j - 1 >= 0:  # Ensure we don't go out of bounds
                # For EGARCH, we need log-variances, but we store variances
                # So we need to take the log of the past variance
                log_variance += beta[j] * np.log(sigma2[t-j-1])
        
        # Convert log-variance to variance
        return np.exp(log_variance)
    
    def _forecast_analytic(self, 
                          steps: int, 
                          last_variance: float) -> np.ndarray:
        """Generate analytic volatility forecasts.
        
        Args:
            steps: Number of steps to forecast
            last_variance: Last observed conditional variance
        
        Returns:
            np.ndarray: Volatility forecasts
        
        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self._fitted or self._parameters is None:
            raise_not_fitted_error(
                "Model must be fitted before forecasting.",
                model_type=self.name,
                operation="forecast_analytic"
            )
        
        # Extract parameters
        omega = self._parameters.omega
        alpha = self._parameters.alpha
        gamma = self._parameters.gamma
        beta = self._parameters.beta
        
        # For EGARCH, we need the last log-variance
        last_log_variance = np.log(last_variance)
        
        # Use optimized Numba implementation
        if len(alpha) == 1 and len(beta) == 1:
            # EGARCH(1,1) case
            return _egarch_forecast(omega, alpha[0], gamma[0], beta[0], last_log_variance, steps)
        else:
            # EGARCH(p,q) case
            # We need the last q log-variances (most recent first)
            if self._log_conditional_variances is None:
                # If log-variances are not available, compute them from variances
                if self._conditional_variances is None:
                    raise RuntimeError("Conditional variances not available for forecasting")
                
                last_log_variances = np.zeros(len(beta))
                for i in range(min(len(beta), len(self._conditional_variances))):
                    last_log_variances[i] = np.log(self._conditional_variances[-(i+1)])
            else:
                # Use stored log-variances
                last_log_variances = np.zeros(len(beta))
                for i in range(min(len(beta), len(self._log_conditional_variances))):
                    last_log_variances[i] = self._log_conditional_variances[-(i+1)]
            
            return _egarch_p_q_forecast(omega, alpha, gamma, beta, last_log_variances, steps)
    
    def fit(self, 
            data: np.ndarray, 
            starting_values: Optional[Union[np.ndarray, EGARCHParams]] = None,
            distribution: Literal["normal", "t", "skewed-t", "ged"] = "normal",
            variance_targeting: bool = False,
            backcast: Optional[float] = None,
            method: str = "SLSQP",
            options: Optional[Dict[str, Any]] = None,
            **kwargs: Any) -> 'UnivariateVolatilityResult':
        """Fit the EGARCH model to the provided data.
        
        Args:
            data: The data to fit the model to (typically residuals)
            starting_values: Initial parameter values for optimization
            distribution: Error distribution assumption
            variance_targeting: Whether to use variance targeting
            backcast: Value to use for initializing the variance process
            method: Optimization method to use
            options: Additional options for the optimizer
            **kwargs: Additional keyword arguments for model fitting
        
        Returns:
            UnivariateVolatilityResult: The model estimation results
        
        Raises:
            ValueError: If the data is invalid
            ConvergenceError: If the optimization fails to converge
            EstimationError: If there are other issues with model estimation
        """
        # Call the parent class implementation
        return super().fit(
            data, 
            starting_values, 
            distribution, 
            variance_targeting, 
            backcast, 
            method, 
            options, 
            **kwargs
        )
    
    async def fit_async(self, 
                       data: np.ndarray, 
                       starting_values: Optional[Union[np.ndarray, EGARCHParams]] = None,
                       distribution: Literal["normal", "t", "skewed-t", "ged"] = "normal",
                       variance_targeting: bool = False,
                       backcast: Optional[float] = None,
                       method: str = "SLSQP",
                       options: Optional[Dict[str, Any]] = None,
                       progress_callback: Optional[Callable[[float, str], None]] = None,
                       **kwargs: Any) -> 'UnivariateVolatilityResult':
        """Asynchronously fit the EGARCH model to the provided data.
        
        This method provides an asynchronous interface to the fit method,
        allowing for non-blocking model estimation in UI contexts.
        
        Args:
            data: The data to fit the model to (typically residuals)
            starting_values: Initial parameter values for optimization
            distribution: Error distribution assumption
            variance_targeting: Whether to use variance targeting
            backcast: Value to use for initializing the variance process
            method: Optimization method to use
            options: Additional options for the optimizer
            progress_callback: Optional callback function for progress updates
            **kwargs: Additional keyword arguments for model fitting
        
        Returns:
            UnivariateVolatilityResult: The model estimation results
        
        Raises:
            ValueError: If the data is invalid
            ConvergenceError: If the optimization fails to converge
            EstimationError: If there are other issues with model estimation
        """
        # Call the parent class implementation
        return await super().fit_async(
            data, 
            starting_values, 
            distribution, 
            variance_targeting, 
            backcast, 
            method, 
            options, 
            progress_callback, 
            **kwargs
        )
    
    def simulate(self, 
                n_periods: int, 
                burn: int = 500, 
                initial_values: Optional[np.ndarray] = None,
                random_state: Optional[Union[int, np.random.Generator]] = None,
                distribution: Literal["normal", "t", "skewed-t", "ged"] = "normal",
                distribution_params: Optional[Dict[str, Any]] = None,
                **kwargs: Any) -> np.ndarray:
        """Simulate data from the EGARCH model.
        
        Args:
            n_periods: Number of periods to simulate
            burn: Number of initial observations to discard
            initial_values: Initial values for the simulation
            random_state: Random number generator or seed
            distribution: Error distribution to use for simulation
            distribution_params: Additional parameters for the error distribution
            **kwargs: Additional keyword arguments for simulation
        
        Returns:
            np.ndarray: Simulated data
        
        Raises:
            RuntimeError: If the model has not been fitted
            ValueError: If the simulation parameters are invalid
            SimulationError: If there are issues during simulation
        """
        # Call the parent class implementation
        return super().simulate(
            n_periods, 
            burn, 
            initial_values, 
            random_state, 
            distribution, 
            distribution_params, 
            **kwargs
        )
    
    async def simulate_async(self, 
                           n_periods: int, 
                           burn: int = 500, 
                           initial_values: Optional[np.ndarray] = None,
                           random_state: Optional[Union[int, np.random.Generator]] = None,
                           distribution: Literal["normal", "t", "skewed-t", "ged"] = "normal",
                           distribution_params: Optional[Dict[str, Any]] = None,
                           progress_callback: Optional[Callable[[float, str], None]] = None,
                           **kwargs: Any) -> np.ndarray:
        """Asynchronously simulate data from the EGARCH model.
        
        This method provides an asynchronous interface to the simulate method,
        allowing for non-blocking simulation in UI contexts.
        
        Args:
            n_periods: Number of periods to simulate
            burn: Number of initial observations to discard
            initial_values: Initial values for the simulation
            random_state: Random number generator or seed
            distribution: Error distribution to use for simulation
            distribution_params: Additional parameters for the error distribution
            progress_callback: Optional callback function for progress updates
            **kwargs: Additional keyword arguments for simulation
        
        Returns:
            np.ndarray: Simulated data
        
        Raises:
            RuntimeError: If the model has not been fitted
            ValueError: If the simulation parameters are invalid
            SimulationError: If there are issues during simulation
        """
        # Call the parent class implementation
        return await super().simulate_async(
            n_periods, 
            burn, 
            initial_values, 
            random_state, 
            distribution, 
            distribution_params, 
            progress_callback, 
            **kwargs
        )
    
    def forecast(self, 
                steps: int, 
                data: Optional[np.ndarray] = None,
                method: Literal["analytic", "simulation"] = "analytic",
                n_simulations: int = 1000,
                random_state: Optional[Union[int, np.random.Generator]] = None,
                distribution: Literal["normal", "t", "skewed-t", "ged"] = "normal",
                distribution_params: Optional[Dict[str, Any]] = None,
                **kwargs: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate volatility forecasts from the fitted EGARCH model.
        
        Args:
            steps: Number of steps to forecast
            data: Historical data to condition the forecast on (if different from fitting data)
            method: Forecasting method ('analytic' or 'simulation')
            n_simulations: Number of simulations for simulation-based forecasting
            random_state: Random number generator or seed for simulation
            distribution: Error distribution to use for simulation
            distribution_params: Additional parameters for the error distribution
            **kwargs: Additional keyword arguments for forecasting
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Point forecasts, lower bounds, and upper bounds
        
        Raises:
            RuntimeError: If the model has not been fitted
            ValueError: If the forecast parameters are invalid
        """
        # Call the parent class implementation
        return super().forecast(
            steps, 
            data, 
            method, 
            n_simulations, 
            random_state, 
            distribution, 
            distribution_params, 
            **kwargs
        )
    
    async def forecast_async(self, 
                           steps: int, 
                           data: Optional[np.ndarray] = None,
                           method: Literal["analytic", "simulation"] = "analytic",
                           n_simulations: int = 1000,
                           random_state: Optional[Union[int, np.random.Generator]] = None,
                           distribution: Literal["normal", "t", "skewed-t", "ged"] = "normal",
                           distribution_params: Optional[Dict[str, Any]] = None,
                           progress_callback: Optional[Callable[[float, str], None]] = None,
                           **kwargs: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Asynchronously generate volatility forecasts from the fitted EGARCH model.
        
        This method provides an asynchronous interface to the forecast method,
        allowing for non-blocking forecasting in UI contexts.
        
        Args:
            steps: Number of steps to forecast
            data: Historical data to condition the forecast on (if different from fitting data)
            method: Forecasting method ('analytic' or 'simulation')
            n_simulations: Number of simulations for simulation-based forecasting
            random_state: Random number generator or seed for simulation
            distribution: Error distribution to use for simulation
            distribution_params: Additional parameters for the error distribution
            progress_callback: Optional callback function for progress updates
            **kwargs: Additional keyword arguments for forecasting
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Point forecasts, lower bounds, and upper bounds
        
        Raises:
            RuntimeError: If the model has not been fitted
            ValueError: If the forecast parameters are invalid
        """
        # Call the parent class implementation
        return await super().forecast_async(
            steps, 
            data, 
            method, 
            n_simulations, 
            random_state, 
            distribution, 
            distribution_params, 
            progress_callback, 
            **kwargs
        )
    
    def validate_data(self, data: np.ndarray) -> None:
        """Validate input data for the EGARCH model.
        
        Args:
            data: Input data to validate
        
        Raises:
            ValueError: If the data is invalid
        """
        # Convert to NumPy array if needed
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        
        # Check dimensions
        if data.ndim != 1:
            raise ValueError(f"data must be a 1-dimensional array, got shape {data.shape}")
        
        # Check for sufficient observations
        if len(data) < max(self.p, self.q) + 1:
            raise ValueError(
                f"data must have at least {max(self.p, self.q) + 1} observations, "
                f"got {len(data)}"
            )
        
        # Check for NaN or infinite values
        if np.any(~np.isfinite(data)):
            raise ValueError("data contains NaN or infinite values")
    
    def plot_diagnostics(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """Plot diagnostic plots for the fitted EGARCH model.
        
        Args:
            figsize: Figure size for the plots
        
        Raises:
            RuntimeError: If the model has not been fitted
            ImportError: If matplotlib is not installed
        """
        if not self._fitted or self._parameters is None or self._conditional_variances is None:
            raise_not_fitted_error(
                "Model must be fitted before plotting diagnostics.",
                model_type=self.name,
                operation="plot_diagnostics"
            )
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Matplotlib is required for plotting. "
                "Please install it with 'pip install matplotlib'."
            )
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        
        # Plot conditional volatility
        axs[0, 0].plot(np.sqrt(self._conditional_variances))
        axs[0, 0].set_title('Conditional Volatility')
        axs[0, 0].set_xlabel('Time')
        axs[0, 0].set_ylabel('Volatility')
        
        # Plot standardized residuals
        if hasattr(self, '_data') and self._data is not None:
            std_residuals = self._data / np.sqrt(self._conditional_variances)
            axs[0, 1].plot(std_residuals)
            axs[0, 1].set_title('Standardized Residuals')
            axs[0, 1].set_xlabel('Time')
            axs[0, 1].set_ylabel('Standardized Residuals')
            
            # Plot histogram of standardized residuals
            axs[1, 0].hist(std_residuals, bins=50, density=True, alpha=0.6)
            
            # Add normal distribution for comparison
            x = np.linspace(min(std_residuals), max(std_residuals), 100)
            axs[1, 0].plot(x, stats.norm.pdf(x, 0, 1), 'r-', lw=2)
            
            axs[1, 0].set_title('Histogram of Standardized Residuals')
            axs[1, 0].set_xlabel('Standardized Residuals')
            axs[1, 0].set_ylabel('Density')
            
            # Plot QQ plot of standardized residuals
            from scipy import stats
            stats.probplot(std_residuals, dist="norm", plot=axs[1, 1])
            axs[1, 1].set_title('QQ Plot of Standardized Residuals')
        
        # Adjust layout and show plot
        plt.tight_layout()
        plt.show()
    
    def __str__(self) -> str:
        """Get a string representation of the EGARCH model.
        
        Returns:
            str: String representation of the model
        """
        if not self._fitted or self._parameters is None:
            return f"{self.name} model (not fitted)"
        
        # Extract parameters
        omega = self._parameters.omega
        alpha = self._parameters.alpha
        gamma = self._parameters.gamma
        beta = self._parameters.beta
        
        # Format parameters
        alpha_str = ", ".join([f"{a:.4f}" for a in alpha])
        gamma_str = ", ".join([f"{g:.4f}" for g in gamma])
        beta_str = ", ".join([f"{b:.4f}" for b in beta])
        
        # Compute persistence
        beta_sum = np.sum(beta)
        
        # Create string representation
        model_str = f"{self.name} model\n"
        model_str += f"omega: {omega:.6f}\n"
        model_str += f"alpha: [{alpha_str}]\n"
        model_str += f"gamma: [{gamma_str}]\n"
        model_str += f"beta: [{beta_str}]\n"
        model_str += f"beta persistence: {beta_sum:.6f}\n"
        
        # Add unconditional variance if model is stationary
        if abs(beta_sum) < 1:
            uncond_var = self._compute_unconditional_variance()
            model_str += f"unconditional variance: {uncond_var:.6f}\n"
        
        return model_str
    
    def __repr__(self) -> str:
        """Get a string representation of the EGARCH model.
        
        Returns:
            str: String representation of the model
        """
        return self.__str__()
