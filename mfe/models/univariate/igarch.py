'''
Integrated GARCH (IGARCH) model implementation.

This module provides a comprehensive implementation of the IGARCH(p,q) model for
modeling time-varying volatility in financial time series with a unit root in the
volatility process. The IGARCH model is a special case of the GARCH model where
the sum of the ARCH and GARCH parameters equals 1, creating a non-stationary
variance process where shocks to volatility persist indefinitely.

The implementation includes parameter estimation via maximum likelihood,
simulation, forecasting, and diagnostic tools. Performance-critical operations
are accelerated using Numba's just-in-time compilation.

References:
    Engle, R. F., & Bollerslev, T. (1986). Modelling the persistence of conditional
    variances. Econometric Reviews, 5(1), 1-50.
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
    UnivariateVolatilityParameters, ParameterError,
    validate_positive, validate_non_negative, validate_probability, validate_range,
    transform_positive, inverse_transform_positive, transform_probability, inverse_transform_probability
)
from mfe.core.exceptions import (
    MFEError, ConvergenceError, NumericError, EstimationError, SimulationError,
    NotFittedError, raise_convergence_error, raise_numeric_error, raise_not_fitted_error,
    warn_convergence, warn_numeric, warn_model
)
from mfe.models.univariate.base import VolatilityModel


# Numba-accelerated core functions for IGARCH model
@jit(nopython=True, cache=True)
def _igarch_recursion(data: np.ndarray, 
                     omega: float, 
                     alpha: float, 
                     sigma2: np.ndarray,
                     backcast: float) -> np.ndarray:
    """Compute IGARCH(1,1) conditional variances using Numba acceleration.
    
    Args:
        data: Input data (typically residuals)
        omega: Constant term in variance equation
        alpha: ARCH parameter
        sigma2: Pre-allocated array for conditional variances
        backcast: Value to use for initializing the variance process
    
    Returns:
        np.ndarray: Conditional variances
    """
    T = len(data)
    
    # Initialize first variance with backcast value
    sigma2[0] = backcast
    
    # Compute conditional variances recursively
    # For IGARCH(1,1), beta = 1 - alpha
    for t in range(1, T):
        sigma2[t] = omega + alpha * data[t-1]**2 + (1 - alpha) * sigma2[t-1]
    
    return sigma2


@jit(nopython=True, cache=True)
def _igarch_p_q_recursion(data: np.ndarray,
                          omega: float,
                          alpha: np.ndarray,
                          beta: np.ndarray,
                          sigma2: np.ndarray,
                          backcast: float) -> np.ndarray:
    """Compute IGARCH(p,q) conditional variances using Numba acceleration.
    
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
def _igarch_forecast(omega: float, 
                    alpha: float, 
                    last_variance: float, 
                    steps: int) -> np.ndarray:
    """Generate analytic forecasts for IGARCH(1,1) model using Numba acceleration.
    
    Args:
        omega: Constant term in variance equation
        alpha: ARCH parameter
        last_variance: Last observed conditional variance
        steps: Number of steps to forecast
    
    Returns:
        np.ndarray: Forecasted conditional variances
    """
    forecasts = np.zeros(steps)
    
    # First step forecast
    forecasts[0] = omega + alpha * last_variance + (1 - alpha) * last_variance
    
    # Multi-step forecasts
    # For IGARCH, the forecasts increase linearly with the horizon
    for h in range(1, steps):
        forecasts[h] = forecasts[h-1] + omega
    
    return forecasts


@jit(nopython=True, cache=True)
def _igarch_p_q_forecast(omega: float,
                         alpha: np.ndarray,
                         beta: np.ndarray,
                         last_variances: np.ndarray,
                         last_squared_returns: np.ndarray,
                         steps: int) -> np.ndarray:
    """Generate analytic forecasts for IGARCH(p,q) model using Numba acceleration.
    
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
    
    # Initialize arrays for multi-step forecasting
    future_variances = np.zeros(max(p, q) + steps)
    future_squared_returns = np.zeros(p + steps)
    
    # Fill in known values
    for i in range(q):
        if i < len(last_variances):
            future_variances[i] = last_variances[i]
        else:
            future_variances[i] = np.mean(last_variances)
    
    for i in range(p):
        if i < len(last_squared_returns):
            future_squared_returns[i] = last_squared_returns[i]
        else:
            future_squared_returns[i] = np.mean(last_squared_returns)
    
    # Generate forecasts
    for h in range(steps):
        # Add constant term
        forecasts[h] = omega
        
        # Add ARCH terms (for h=1, these are known; for h>1, use conditional variance)
        for i in range(p):
            if h + i < p:
                forecasts[h] += alpha[i] * future_squared_returns[h+i]
            else:
                forecasts[h] += alpha[i] * forecasts[h+i-p]
        
        # Add GARCH terms
        for j in range(q):
            if h + j < q:
                forecasts[h] += beta[j] * future_variances[h+j]
            else:
                idx = h + j - q
                if idx < len(forecasts):
                    forecasts[h] += beta[j] * forecasts[idx]
                else:
                    forecasts[h] += beta[j] * forecasts[-1]
        
        # Update future values for next iteration
        future_variances[q + h] = forecasts[h]
        future_squared_returns[p + h] = forecasts[h]  # E[r²] = σ²
    
    return forecasts


@dataclass
class IGARCHParams(UnivariateVolatilityParameters):
    """Parameters for IGARCH(p,q) model.
    
    This class implements parameters for the IGARCH model, which is a special case
    of the GARCH model where the sum of ARCH and GARCH parameters equals 1.
    
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
        
        # Enforce the unit root constraint: sum(alpha) + sum(beta) = 1
        # Adjust beta to ensure the constraint is satisfied
        alpha_sum = np.sum(self.alpha)
        beta_sum = np.sum(self.beta)
        
        if abs(alpha_sum + beta_sum - 1.0) > 1e-6:
            # Rescale beta to enforce the constraint
            if len(self.beta) == 1:
                self.beta[0] = 1.0 - alpha_sum
            else:
                # Preserve relative proportions in beta
                beta_weights = self.beta / beta_sum if beta_sum > 0 else np.ones(len(self.beta)) / len(self.beta)
                self.beta = beta_weights * (1.0 - alpha_sum)
        
        self.validate()
    
    def validate(self) -> None:
        """Validate IGARCH parameter constraints.
        
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
        
        # Validate unit root constraint
        alpha_sum = np.sum(self.alpha)
        beta_sum = np.sum(self.beta)
        
        if abs(alpha_sum + beta_sum - 1.0) > 1e-6:
            raise ParameterError(
                f"IGARCH unit root constraint violated: sum(alpha) + sum(beta) = {alpha_sum + beta_sum} ≠ 1"
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
    def from_array(cls, array: np.ndarray, p: int = 1, q: int = 1, **kwargs: Any) -> 'IGARCHParams':
        """Create parameters from a NumPy array.
        
        Args:
            array: Array representation of parameters
            p: Order of the ARCH component
            q: Order of the GARCH component
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            IGARCHParams: Parameter object
        
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
        
        For IGARCH, we only need to transform omega and the relative weights of alpha,
        since the sum of alpha and beta is constrained to be 1.
        
        Returns:
            np.ndarray: Transformed parameters in unconstrained space
        """
        # Transform omega to unconstrained space (log)
        transformed_omega = transform_positive(self.omega)
        
        # For IGARCH, we only need to transform the relative weights of alpha
        # since the sum of alpha and beta is constrained to be 1
        alpha_sum = np.sum(self.alpha)
        
        # Transform alpha_sum to unconstrained space
        transformed_alpha_sum = transform_probability(alpha_sum)
        
        # Compute relative weights within alpha
        if len(self.alpha) > 1:
            alpha_weights = self.alpha / alpha_sum if alpha_sum > 0 else np.ones(len(self.alpha)) / len(self.alpha)
            
            # Transform weights to unconstrained space
            transformed_alpha_weights = np.zeros(len(alpha_weights) - 1)
            
            remaining = 1.0
            for i in range(len(alpha_weights) - 1):
                if remaining > 0:
                    proportion = alpha_weights[i] / remaining
                    transformed_alpha_weights[i] = transform_probability(proportion)
                    remaining -= alpha_weights[i]
            
            # Combine all transformed parameters
            return np.concatenate([
                np.array([transformed_omega, transformed_alpha_sum]),
                transformed_alpha_weights
            ])
        else:
            # If there's only one alpha, we just need to transform alpha_sum
            return np.array([transformed_omega, transformed_alpha_sum])
    
    @classmethod
    def inverse_transform(cls, array: np.ndarray, p: int = 1, q: int = 1, **kwargs: Any) -> 'IGARCHParams':
        """Transform parameters from unconstrained space back to constrained space.
        
        Args:
            array: Parameters in unconstrained space
            p: Order of the ARCH component
            q: Order of the GARCH component
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            IGARCHParams: Parameter object with constrained parameters
        
        Raises:
            ValueError: If the array length doesn't match the expected length
        """
        expected_length = 2 + (p - 1)  # omega, alpha_sum, alpha_weights
        if len(array) != expected_length:
            raise ValueError(
                f"Array length ({len(array)}) doesn't match expected length ({expected_length})"
            )
        
        # Extract transformed parameters
        transformed_omega = array[0]
        transformed_alpha_sum = array[1]
        
        # Inverse transform omega
        omega = inverse_transform_positive(transformed_omega)
        
        # Inverse transform alpha_sum
        alpha_sum = inverse_transform_probability(transformed_alpha_sum)
        
        # Compute alpha and beta
        alpha = np.zeros(p)
        beta = np.zeros(q)
        
        if p == 1:
            # If there's only one alpha, it's just alpha_sum
            alpha[0] = alpha_sum
        else:
            # Extract transformed alpha weights
            transformed_alpha_weights = array[2:]
            
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
        
        # For IGARCH, beta is determined by the constraint sum(alpha) + sum(beta) = 1
        beta_sum = 1.0 - alpha_sum
        
        if q == 1:
            # If there's only one beta, it's just beta_sum
            beta[0] = beta_sum
        else:
            # Distribute beta_sum equally among beta parameters
            # This is a simplification; in practice, we might want a more sophisticated approach
            beta = np.ones(q) * beta_sum / q
        
        return cls(omega=omega, alpha=alpha, beta=beta)


class IGARCH(VolatilityModel):
    """IGARCH (Integrated Generalized Autoregressive Conditional Heteroskedasticity) model.
    
    This class implements the IGARCH(p,q) model for modeling time-varying volatility
    in financial time series with a unit root in the volatility process. The model is defined as:
    
    σ²_t = ω + Σ(α_i * r²_{t-i}) + Σ(β_j * σ²_{t-j})
    
    with the constraint that Σα_i + Σβ_j = 1, creating a unit root in the variance process.
    
    This non-stationary variance process means that shocks to volatility persist indefinitely,
    making it suitable for modeling highly persistent volatility series.
    """
    
    def __init__(self, 
                 p: int = 1, 
                 q: int = 1, 
                 parameters: Optional[IGARCHParams] = None, 
                 name: str = "IGARCH") -> None:
        """Initialize the IGARCH model.
        
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
        if name == "IGARCH":
            name = f"IGARCH({p},{q})"
        
        super().__init__(parameters=parameters, name=name)
        
        # Initialize additional attributes
        self._conditional_variances = None
        self._data = None
    
    def parameter_class(self) -> Type[IGARCHParams]:
        """Get the parameter class for this model.
        
        Returns:
            Type[IGARCHParams]: The parameter class for this model
        """
        return IGARCHParams
    
    def compute_variance(self, 
                         parameters: IGARCHParams, 
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
                # IGARCH(1,1) case
                omega = parameters.omega
                alpha = parameters.alpha[0]
                
                sigma2 = _igarch_recursion(data, omega, alpha, sigma2, backcast)
            else:
                # IGARCH(p,q) case
                omega = parameters.omega
                alpha = parameters.alpha
                beta = parameters.beta
                
                sigma2 = _igarch_p_q_recursion(data, omega, alpha, beta, sigma2, backcast)
            
            # Check for numerical issues
            if np.any(~np.isfinite(sigma2)) or np.any(sigma2 <= 0):
                raise_numeric_error(
                    "Numerical issues detected in variance computation",
                    operation="IGARCH variance computation",
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
                operation="IGARCH variance computation",
                error_type="Computation error",
                details=str(e)
            )
    
    def _generate_starting_values(self, 
                                 data: np.ndarray, 
                                 variance_targeting: bool = False,
                                 backcast: Optional[float] = None) -> IGARCHParams:
        """Generate starting values for parameter estimation.
        
        Args:
            data: Input data (typically residuals)
            variance_targeting: Whether to use variance targeting
            backcast: Value to use for initializing the variance process
        
        Returns:
            IGARCHParams: Starting parameter values
        """
        # Compute sample variance
        sample_variance = np.var(data)
        
        if backcast is None:
            backcast = sample_variance
        
        # Generate starting values based on model order
        if self.p == 1 and self.q == 1:
            # IGARCH(1,1) case
            # For IGARCH, we need alpha + beta = 1
            alpha = 0.1  # Start with a small alpha
            beta = 0.9   # Beta is determined by the constraint
            
            # For IGARCH, omega determines the rate of growth of the unconditional variance
            # A small value is typically appropriate
            omega = 0.01 * sample_variance
            
            return IGARCHParams(omega=omega, alpha=alpha, beta=beta)
        else:
            # IGARCH(p,q) case
            # Distribute alpha and beta values
            alpha_total = 0.1
            beta_total = 0.9  # alpha_total + beta_total = 1 for IGARCH
            
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
            
            # For IGARCH, omega determines the rate of growth of the unconditional variance
            omega = 0.01 * sample_variance
            
            return IGARCHParams(omega=omega, alpha=alpha, beta=beta)
    
    def _compute_unconditional_variance(self) -> float:
        """Compute the unconditional variance of the process.
        
        For IGARCH, the unconditional variance is not defined (infinite).
        This method returns a reasonable proxy based on the last observed variance.
        
        Returns:
            float: Proxy for unconditional variance
        
        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self._fitted or self._parameters is None:
            raise_not_fitted_error(
                "Model must be fitted before computing unconditional variance.",
                model_type=self.name,
                operation="compute_unconditional_variance"
            )
        
        # For IGARCH, the unconditional variance is not defined (infinite)
        # Return the last observed variance as a proxy
        if self._conditional_variances is not None:
            return self._conditional_variances[-1]
        else:
            # Fallback to a reasonable value based on omega
            # This is just a heuristic
            return self._parameters.omega * 100
    
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
            # IGARCH(1,1) case
            return _igarch_forecast(omega, alpha[0], last_variance, steps)
        else:
            # IGARCH(p,q) case
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
            
            return _igarch_p_q_forecast(omega, alpha, beta, last_variances, last_squared_returns, steps)
    
    def fit(self, 
            data: np.ndarray, 
            starting_values: Optional[Union[np.ndarray, IGARCHParams]] = None,
            distribution: Literal["normal", "t", "skewed-t", "ged"] = "normal",
            variance_targeting: bool = False,
            backcast: Optional[float] = None,
            method: str = "SLSQP",
            options: Optional[Dict[str, Any]] = None,
            **kwargs: Any) -> 'UnivariateVolatilityResult':
        """Fit the IGARCH model to the provided data.
        
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
        # Store data for later use
        self._data = data
        
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
                       starting_values: Optional[Union[np.ndarray, IGARCHParams]] = None,
                       distribution: Literal["normal", "t", "skewed-t", "ged"] = "normal",
                       variance_targeting: bool = False,
                       backcast: Optional[float] = None,
                       method: str = "SLSQP",
                       options: Optional[Dict[str, Any]] = None,
                       progress_callback: Optional[Callable[[float, str], None]] = None,
                       **kwargs: Any) -> 'UnivariateVolatilityResult':
        """Asynchronously fit the IGARCH model to the provided data.
        
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
        # Store data for later use
        self._data = data
        
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
        """Simulate data from the IGARCH model.
        
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
        """Asynchronously simulate data from the IGARCH model.
        
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
        """Generate volatility forecasts from the fitted IGARCH model.
        
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
        """Asynchronously generate volatility forecasts from the fitted IGARCH model.
        
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
        """Validate input data for the IGARCH model.
        
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
        """Plot diagnostic plots for the fitted IGARCH model.
        
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
        if self._data is not None:
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
        """Get a string representation of the IGARCH model.
        
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
        model_str += "Note: IGARCH has a unit root in the variance process (infinite unconditional variance)\n"
        
        return model_str
    
    def __repr__(self) -> str:
        """Get a string representation of the IGARCH model.
        
        Returns:
            str: String representation of the model
        """
        return self.__str__()
