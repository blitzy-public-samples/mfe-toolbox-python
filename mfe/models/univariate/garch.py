'''
GARCH (Generalized Autoregressive Conditional Heteroskedasticity) model implementation.

This module provides a comprehensive implementation of the GARCH(p,q) model for
modeling time-varying volatility in financial time series. The GARCH model is
a generalization of the ARCH model that allows for both autoregressive and
moving average components in the conditional variance equation.

The implementation includes parameter estimation via maximum likelihood,
simulation, forecasting, and diagnostic tools. Performance-critical operations
are accelerated using Numba's just-in-time compilation.

References:
    Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity.
    Journal of Econometrics, 31(3), 307-327.
'''

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
    GARCHParameters, UnivariateVolatilityParameters, ParameterError,
    validate_positive, validate_non_negative, validate_probability, validate_range
)
from mfe.core.exceptions import (
    MFEError, ConvergenceError, NumericError, EstimationError, SimulationError,
    NotFittedError, raise_convergence_error, raise_numeric_error, raise_not_fitted_error,
    warn_convergence, warn_numeric, warn_model
)
from mfe.models.univariate.base import VolatilityModel


# Numba-accelerated core functions for GARCH model
@jit(nopython=True, cache=True)
def _garch_recursion(data: np.ndarray, 
                    omega: float, 
                    alpha: float, 
                    beta: float, 
                    sigma2: np.ndarray,
                    backcast: float) -> np.ndarray:
    """Compute GARCH(1,1) conditional variances using Numba acceleration.
    
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


@jit(nopython=True, cache=True)
def _garch_p_q_recursion(data: np.ndarray,
                         omega: float,
                         alpha: np.ndarray,
                         beta: np.ndarray,
                         sigma2: np.ndarray,
                         backcast: float) -> np.ndarray:
    """Compute GARCH(p,q) conditional variances using Numba acceleration.
    
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


@jit(nopython=True, cache=True)
def _garch_forecast(omega: float, 
                   alpha: float, 
                   beta: float, 
                   last_variance: float, 
                   steps: int) -> np.ndarray:
    """Generate analytic forecasts for GARCH(1,1) model using Numba acceleration.
    
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


@jit(nopython=True, cache=True)
def _garch_p_q_forecast(omega: float,
                        alpha: np.ndarray,
                        beta: np.ndarray,
                        last_variances: np.ndarray,
                        last_squared_returns: np.ndarray,
                        steps: int) -> np.ndarray:
    """Generate analytic forecasts for GARCH(p,q) model using Numba acceleration.
    
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


@dataclass
class GARCHParams(GARCHParameters):
    """Parameters for GARCH(p,q) model.
    
    This class extends the base GARCHParameters class to support higher-order
    GARCH models with multiple ARCH and GARCH terms.
    
    Attributes:
        omega: Constant term in variance equation (must be positive)
        alpha: ARCH parameters (must be non-negative)
        beta: GARCH parameters (must be non-negative)
    """
    
    omega: float
    alpha: Union[float, np.ndarray]
    beta: Union[float, np.ndarray]
    
    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        # Convert alpha and beta to arrays if they are scalars
        if isinstance(self.alpha, (int, float)):
            self.alpha = np.array([self.alpha])
        elif not isinstance(self.alpha, np.ndarray):
            self.alpha = np.array(self.alpha)
            
        if isinstance(self.beta, (int, float)):
            self.beta = np.array([self.beta])
        elif not isinstance(self.beta, np.ndarray):
            self.beta = np.array(self.beta)
        
        self.validate()
    
    def validate(self) -> None:
        """Validate GARCH parameter constraints.
        
        Raises:
            ParameterError: If parameter constraints are violated
        """
        # Validate omega
        validate_positive(self.omega, "omega")
        
        # Validate alpha and beta
        for i, a in enumerate(self.alpha):
            validate_non_negative(a, f"alpha[{i}]")
        
        for i, b in enumerate(self.beta):
            validate_non_negative(b, f"beta[{i}]")
        
        # Validate stationarity constraint
        persistence = np.sum(self.alpha) + np.sum(self.beta)
        if persistence >= 1:
            raise ParameterError(
                f"GARCH stationarity constraint violated: sum(alpha) + sum(beta) = {persistence} >= 1"
            )
    
    def to_array(self) -> np.ndarray:
        """Convert parameters to a NumPy array.
        
        Returns:
            np.ndarray: Array representation of parameters
        """
        return np.concatenate([
            np.array([self.omega]),
            self.alpha,
            self.beta
        ])
    
    @classmethod
    def from_array(cls, array: np.ndarray, p: int = 1, q: int = 1, **kwargs: Any) -> 'GARCHParams':
        """Create parameters from a NumPy array.
        
        Args:
            array: Array representation of parameters
            p: Order of the ARCH component
            q: Order of the GARCH component
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            GARCHParams: Parameter object
        
        Raises:
            ValueError: If the array length doesn't match the expected length
        """
        expected_length = 1 + p + q  # omega, alpha, beta
        if len(array) != expected_length:
            raise ValueError(
                f"Array length ({len(array)}) doesn't match expected length ({expected_length})"
            )
        
        # Extract parameters
        omega = array[0]
        alpha = array[1:1+p]
        beta = array[1+p:]
        
        return cls(omega=omega, alpha=alpha, beta=beta)
    
    def transform(self) -> np.ndarray:
        """Transform parameters to unconstrained space for optimization.
        
        Returns:
            np.ndarray: Transformed parameters in unconstrained space
        """
        from mfe.core.parameters import transform_positive, transform_probability
        
        # Transform omega to unconstrained space (log)
        transformed_omega = transform_positive(self.omega)
        
        # Transform alpha and beta to unconstrained space
        # We use a special transformation to ensure sum(alpha) + sum(beta) < 1
        persistence = np.sum(self.alpha) + np.sum(self.beta)
        if persistence >= 1:
            # If constraint is violated, adjust parameters slightly
            factor = 0.99 / persistence
            self.alpha = self.alpha * factor
            self.beta = self.beta * factor
            persistence = 0.99
        
        # Use logit-like transformation for persistence
        transformed_persistence = transform_probability(persistence)
        
        # Transform relative weights of alpha and beta
        alpha_sum = np.sum(self.alpha)
        beta_sum = np.sum(self.beta)
        
        # Compute the proportion of persistence due to alpha
        alpha_proportion = alpha_sum / persistence if persistence > 0 else 0.5
        transformed_alpha_proportion = transform_probability(alpha_proportion)
        
        # Compute relative weights within alpha and beta
        alpha_weights = self.alpha / alpha_sum if alpha_sum > 0 else np.ones(len(self.alpha)) / len(self.alpha)
        beta_weights = self.beta / beta_sum if beta_sum > 0 else np.ones(len(self.beta)) / len(self.beta)
        
        # Transform weights to unconstrained space
        # We use a simplex transformation for the weights
        transformed_alpha_weights = np.zeros(len(alpha_weights) - 1) if len(alpha_weights) > 1 else np.array([])
        transformed_beta_weights = np.zeros(len(beta_weights) - 1) if len(beta_weights) > 1 else np.array([])
        
        if len(alpha_weights) > 1:
            remaining = 1.0
            for i in range(len(alpha_weights) - 1):
                if remaining > 0:
                    proportion = alpha_weights[i] / remaining
                    transformed_alpha_weights[i] = transform_probability(proportion)
                    remaining -= alpha_weights[i]
        
        if len(beta_weights) > 1:
            remaining = 1.0
            for i in range(len(beta_weights) - 1):
                if remaining > 0:
                    proportion = beta_weights[i] / remaining
                    transformed_beta_weights[i] = transform_probability(proportion)
                    remaining -= beta_weights[i]
        
        # Combine all transformed parameters
        return np.concatenate([
            np.array([transformed_omega, transformed_persistence, transformed_alpha_proportion]),
            transformed_alpha_weights,
            transformed_beta_weights
        ])
    
    @classmethod
    def inverse_transform(cls, array: np.ndarray, p: int = 1, q: int = 1, **kwargs: Any) -> 'GARCHParams':
        """Transform parameters from unconstrained space back to constrained space.
        
        Args:
            array: Parameters in unconstrained space
            p: Order of the ARCH component
            q: Order of the GARCH component
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            GARCHParams: Parameter object with constrained parameters
        
        Raises:
            ValueError: If the array length doesn't match the expected length
        """
        from mfe.core.parameters import inverse_transform_positive, inverse_transform_probability
        
        expected_length = 3 + (p - 1) + (q - 1)  # omega, persistence, alpha_proportion, alpha_weights, beta_weights
        if len(array) != expected_length:
            raise ValueError(
                f"Array length ({len(array)}) doesn't match expected length ({expected_length})"
            )
        
        # Extract transformed parameters
        transformed_omega = array[0]
        transformed_persistence = array[1]
        transformed_alpha_proportion = array[2]
        
        # Extract transformed weights
        transformed_alpha_weights = array[3:3+(p-1)] if p > 1 else np.array([])
        transformed_beta_weights = array[3+(p-1):] if q > 1 else np.array([])
        
        # Inverse transform omega
        omega = inverse_transform_positive(transformed_omega)
        
        # Inverse transform persistence
        persistence = inverse_transform_probability(transformed_persistence)
        
        # Inverse transform alpha proportion
        alpha_proportion = inverse_transform_probability(transformed_alpha_proportion)
        
        # Compute alpha and beta sums
        alpha_sum = persistence * alpha_proportion
        beta_sum = persistence * (1 - alpha_proportion)
        
        # Inverse transform weights to get alpha and beta
        alpha = np.zeros(p)
        beta = np.zeros(q)
        
        if p == 1:
            alpha[0] = alpha_sum
        else:
            # Reconstruct alpha weights from transformed values
            alpha_weights = np.zeros(p)
            remaining = 1.0
            for i in range(p - 1):
                proportion = inverse_transform_probability(transformed_alpha_weights[i])
                alpha_weights[i] = proportion * remaining
                remaining -= alpha_weights[i]
            alpha_weights[p - 1] = remaining
            
            # Scale weights by alpha_sum
            alpha = alpha_weights * alpha_sum
        
        if q == 1:
            beta[0] = beta_sum
        else:
            # Reconstruct beta weights from transformed values
            beta_weights = np.zeros(q)
            remaining = 1.0
            for i in range(q - 1):
                proportion = inverse_transform_probability(transformed_beta_weights[i])
                beta_weights[i] = proportion * remaining
                remaining -= beta_weights[i]
            beta_weights[q - 1] = remaining
            
            # Scale weights by beta_sum
            beta = beta_weights * beta_sum
        
        return cls(omega=omega, alpha=alpha, beta=beta)


class GARCH(VolatilityModel):
    """GARCH (Generalized Autoregressive Conditional Heteroskedasticity) model.
    
    This class implements the GARCH(p,q) model for modeling time-varying volatility
    in financial time series. The model is defined as:
    
    σ²_t = ω + Σ(α_i * r²_{t-i}) + Σ(β_j * σ²_{t-j})
    
    where σ²_t is the conditional variance at time t, r²_{t-i} are past squared
    returns, and σ²_{t-j} are past conditional variances.
    """
    
    def __init__(self, 
                 p: int = 1, 
                 q: int = 1, 
                 parameters: Optional[GARCHParams] = None, 
                 name: str = "GARCH") -> None:
        """Initialize the GARCH model.
        
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
        if name == "GARCH":
            name = f"GARCH({p},{q})"
        
        super().__init__(parameters=parameters, name=name)
        
        # Initialize additional attributes
        self._conditional_variances = None
    
    def parameter_class(self) -> Type[GARCHParams]:
        """Get the parameter class for this model.
        
        Returns:
            Type[GARCHParams]: The parameter class for this model
        """
        return GARCHParams
    
    def compute_variance(self, 
                         parameters: GARCHParams, 
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
        
        # Compute backcast value if not provided
        if backcast is None:
            backcast = np.mean(data**2)
        
        try:
            # Use optimized Numba implementation
            if self.p == 1 and self.q == 1:
                # GARCH(1,1) case
                omega = parameters.omega
                alpha = parameters.alpha[0]
                beta = parameters.beta[0]
                
                sigma2 = _garch_recursion(data, omega, alpha, beta, sigma2, backcast)
            else:
                # GARCH(p,q) case
                omega = parameters.omega
                alpha = parameters.alpha
                beta = parameters.beta
                
                sigma2 = _garch_p_q_recursion(data, omega, alpha, beta, sigma2, backcast)
            
            # Check for numerical issues
            if np.any(~np.isfinite(sigma2)) or np.any(sigma2 <= 0):
                raise_numeric_error(
                    "Numerical issues detected in variance computation",
                    operation="GARCH variance computation",
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
                operation="GARCH variance computation",
                error_type="Computation error",
                details=str(e)
            )
    
    def _generate_starting_values(self, 
                                 data: np.ndarray, 
                                 variance_targeting: bool = False,
                                 backcast: Optional[float] = None) -> GARCHParams:
        """Generate starting values for parameter estimation.
        
        Args:
            data: Input data (typically residuals)
            variance_targeting: Whether to use variance targeting
            backcast: Value to use for initializing the variance process
        
        Returns:
            GARCHParams: Starting parameter values
        """
        # Compute sample variance
        sample_variance = np.var(data)
        
        if backcast is None:
            backcast = sample_variance
        
        # Generate starting values based on model order
        if self.p == 1 and self.q == 1:
            # GARCH(1,1) case
            if variance_targeting:
                # With variance targeting, omega is determined by the other parameters
                alpha = 0.05
                beta = 0.90
                omega = sample_variance * (1 - alpha - beta)
            else:
                # Without variance targeting, use reasonable defaults
                omega = 0.05 * sample_variance
                alpha = 0.05
                beta = 0.90
            
            return GARCHParams(omega=omega, alpha=alpha, beta=beta)
        else:
            # GARCH(p,q) case
            # Distribute alpha and beta values
            alpha_total = 0.05
            beta_total = 0.90
            
            alpha = np.zeros(self.p)
            beta = np.zeros(self.q)
            
            # Distribute alpha values with exponential decay
            if self.p > 0:
                alpha_weights = np.exp(-np.arange(self.p))
                alpha_weights = alpha_weights / np.sum(alpha_weights)
                alpha = alpha_weights * alpha_total
            
            # Distribute beta values with exponential decay
            if self.q > 0:
                beta_weights = np.exp(-np.arange(self.q))
                beta_weights = beta_weights / np.sum(beta_weights)
                beta = beta_weights * beta_total
            
            if variance_targeting:
                # With variance targeting, omega is determined by the other parameters
                omega = sample_variance * (1 - np.sum(alpha) - np.sum(beta))
            else:
                # Without variance targeting, use reasonable defaults
                omega = 0.05 * sample_variance
            
            return GARCHParams(omega=omega, alpha=alpha, beta=beta)
    
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
        
        # For GARCH model, unconditional variance is omega / (1 - sum(alpha) - sum(beta))
        omega = self._parameters.omega
        persistence = np.sum(self._parameters.alpha) + np.sum(self._parameters.beta)
        
        if persistence >= 1:
            # If the process is not stationary, return the last observed variance
            if self._conditional_variances is not None:
                return self._conditional_variances[-1]
            else:
                # Fallback to a reasonable value
                return omega / 0.05
        
        return omega / (1 - persistence)
    
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
        beta = self._parameters.beta
        
        # Compute variance for time t
        variance = omega
        
        # Add ARCH terms
        for i in range(len(alpha)):
            if t - i - 1 >= 0:  # Ensure we don't go out of bounds
                variance += alpha[i] * innovations[t-i-1]**2
        
        # Add GARCH terms
        for j in range(len(beta)):
            if t - j - 1 >= 0:  # Ensure we don't go out of bounds
                variance += beta[j] * sigma2[t-j-1]
        
        return variance
    
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
        beta = self._parameters.beta
        
        # Use optimized Numba implementation
        if len(alpha) == 1 and len(beta) == 1:
            # GARCH(1,1) case
            return _garch_forecast(omega, alpha[0], beta[0], last_variance, steps)
        else:
            # GARCH(p,q) case
            # We need the last p squared returns and last q variances
            if self._conditional_variances is None:
                raise RuntimeError("Conditional variances not available for forecasting")
            
            # Get the last q variances (most recent first)
            last_variances = np.zeros(len(beta))
            for i in range(min(len(beta), len(self._conditional_variances))):
                last_variances[i] = self._conditional_variances[-(i+1)]
            
            # For squared returns, we use the last p conditional variances as a proxy
            # This is because E[r²] = σ² for zero-mean returns
            last_squared_returns = np.zeros(len(alpha))
            for i in range(min(len(alpha), len(self._conditional_variances))):
                last_squared_returns[i] = self._conditional_variances[-(i+1)]
            
            return _garch_p_q_forecast(omega, alpha, beta, last_variances, last_squared_returns, steps)
    
    def fit(self, 
            data: np.ndarray, 
            starting_values: Optional[Union[np.ndarray, GARCHParams]] = None,
            distribution: Literal["normal", "t", "skewed-t", "ged"] = "normal",
            variance_targeting: bool = False,
            backcast: Optional[float] = None,
            method: str = "SLSQP",
            options: Optional[Dict[str, Any]] = None,
            **kwargs: Any) -> 'UnivariateVolatilityResult':
        """Fit the GARCH model to the provided data.
        
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
                       starting_values: Optional[Union[np.ndarray, GARCHParams]] = None,
                       distribution: Literal["normal", "t", "skewed-t", "ged"] = "normal",
                       variance_targeting: bool = False,
                       backcast: Optional[float] = None,
                       method: str = "SLSQP",
                       options: Optional[Dict[str, Any]] = None,
                       progress_callback: Optional[Callable[[float, str], None]] = None,
                       **kwargs: Any) -> 'UnivariateVolatilityResult':
        """Asynchronously fit the GARCH model to the provided data.
        
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
        """Simulate data from the GARCH model.
        
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
        """Asynchronously simulate data from the GARCH model.
        
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
        """Generate volatility forecasts from the fitted GARCH model.
        
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
        """Asynchronously generate volatility forecasts from the fitted GARCH model.
        
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
        """Validate input data for the GARCH model.
        
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
        """Plot diagnostic plots for the fitted GARCH model.
        
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
        """Get a string representation of the GARCH model.
        
        Returns:
            str: String representation of the model
        """
        if not self._fitted or self._parameters is None:
            return f"{self.name} model (not fitted)"
        
        # Extract parameters
        omega = self._parameters.omega
        alpha = self._parameters.alpha
        beta = self._parameters.beta
        
        # Format parameters
        alpha_str = ", ".join([f"{a:.4f}" for a in alpha])
        beta_str = ", ".join([f"{b:.4f}" for b in beta])
        
        # Compute persistence
        persistence = np.sum(alpha) + np.sum(beta)
        
        # Create string representation
        model_str = f"{self.name} model\n"
        model_str += f"omega: {omega:.6f}\n"
        model_str += f"alpha: [{alpha_str}]\n"
        model_str += f"beta: [{beta_str}]\n"
        model_str += f"persistence: {persistence:.6f}\n"
        
        # Add unconditional variance if model is stationary
        if persistence < 1:
            uncond_var = self._compute_unconditional_variance()
            model_str += f"unconditional variance: {uncond_var:.6f}\n"
        
        return model_str
    
    def __repr__(self) -> str:
        """Get a string representation of the GARCH model.
        
        Returns:
            str: String representation of the model
        """
        return self.__str__()
