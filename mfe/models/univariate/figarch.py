'''
Fractionally Integrated GARCH (FIGARCH) model implementation.

This module provides a comprehensive implementation of the FIGARCH(p,d,q) model for
modeling long memory in volatility. The FIGARCH model extends the standard GARCH
model by introducing a fractional differencing parameter that allows for long-range
dependence in the conditional variance process.

The implementation includes parameter estimation via maximum likelihood,
simulation, forecasting, and diagnostic tools. Performance-critical operations
are accelerated using Numba's just-in-time compilation.

References:
    Baillie, R. T., Bollerslev, T., & Mikkelsen, H. O. (1996). Fractionally integrated
    generalized autoregressive conditional heteroskedasticity. Journal of Econometrics,
    74(1), 3-30.
'''

import asyncio
import warnings
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union, cast, overload
)
import numpy as np
import pandas as pd
from scipy import optimize, stats, special
from numba import jit

from mfe.core.base import ModelBase, VolatilityModelBase
from mfe.core.parameters import (
    UnivariateVolatilityParameters, ParameterError,
    validate_positive, validate_non_negative, validate_probability, validate_range
)
from mfe.core.exceptions import (
    MFEError, ConvergenceError, NumericError, EstimationError, SimulationError,
    NotFittedError, raise_convergence_error, raise_numeric_error, raise_not_fitted_error,
    warn_convergence, warn_numeric, warn_model
)
from mfe.models.univariate.base import VolatilityModel


# Numba-accelerated core functions for FIGARCH model
@jit(nopython=True, cache=True)
def _figarch_weights(d: float, phi: np.ndarray, beta: np.ndarray, trunc_lag: int) -> np.ndarray:
    """Compute FIGARCH weights (lambda coefficients) using Numba acceleration.
    
    This function computes the weights for the ARCH(∞) representation of the FIGARCH model.
    
    Args:
        d: Fractional differencing parameter
        phi: ARCH parameters (array of length p)
        beta: GARCH parameters (array of length q)
        trunc_lag: Truncation lag for the infinite ARCH representation
    
    Returns:
        np.ndarray: FIGARCH weights (lambda coefficients)
    """
    # Initialize weights
    weights = np.zeros(trunc_lag + 1)
    
    # First weight is always 1
    weights[0] = 1.0
    
    # Compute fractional differencing coefficients
    frac_diff = np.zeros(trunc_lag + 1)
    frac_diff[0] = 1.0
    for i in range(1, trunc_lag + 1):
        frac_diff[i] = frac_diff[i-1] * (i - 1 - d) / i
    
    # Compute FIGARCH weights recursively
    p = len(phi)
    q = len(beta)
    
    for i in range(1, trunc_lag + 1):
        # Start with fractional differencing term
        weights[i] = frac_diff[i]
        
        # Add ARCH terms
        for j in range(1, min(i, p) + 1):
            weights[i] += phi[j-1] * weights[i-j]
        
        # Subtract GARCH terms
        for j in range(1, min(i, q) + 1):
            weights[i] -= beta[j-1] * weights[i-j]
    
    return weights


@jit(nopython=True, cache=True)
def _figarch_recursion(data: np.ndarray, 
                      omega: float, 
                      d: float,
                      phi: np.ndarray, 
                      beta: np.ndarray, 
                      sigma2: np.ndarray,
                      backcast: float,
                      trunc_lag: int) -> np.ndarray:
    """Compute FIGARCH conditional variances using Numba acceleration.
    
    Args:
        data: Input data (typically residuals)
        omega: Constant term in variance equation
        d: Fractional differencing parameter
        phi: ARCH parameters (array of length p)
        beta: GARCH parameters (array of length q)
        sigma2: Pre-allocated array for conditional variances
        backcast: Value to use for initializing the variance process
        trunc_lag: Truncation lag for the infinite ARCH representation
    
    Returns:
        np.ndarray: Conditional variances
    """
    T = len(data)
    
    # Compute FIGARCH weights
    weights = _figarch_weights(d, phi, beta, trunc_lag)
    
    # Initialize first variance with backcast value
    sigma2[0] = backcast
    
    # Compute conditional variances recursively
    for t in range(1, T):
        # Start with constant term
        sigma2[t] = omega
        
        # Add weighted squared residuals
        for i in range(1, min(t, trunc_lag) + 1):
            sigma2[t] += weights[i] * data[t-i]**2
        
        # For lags beyond available data, use backcast value
        if t < trunc_lag:
            remaining_weight = np.sum(weights[t+1:trunc_lag+1])
            sigma2[t] += remaining_weight * backcast
    
    return sigma2


@jit(nopython=True, cache=True)
def _figarch_forecast(omega: float, 
                     d: float,
                     phi: np.ndarray, 
                     beta: np.ndarray, 
                     last_data: np.ndarray,
                     last_variance: float,
                     steps: int,
                     trunc_lag: int) -> np.ndarray:
    """Generate analytic forecasts for FIGARCH model using Numba acceleration.
    
    Args:
        omega: Constant term in variance equation
        d: Fractional differencing parameter
        phi: ARCH parameters (array of length p)
        beta: GARCH parameters (array of length q)
        last_data: Last trunc_lag squared returns (most recent first)
        last_variance: Last observed conditional variance
        steps: Number of steps to forecast
        trunc_lag: Truncation lag for the infinite ARCH representation
    
    Returns:
        np.ndarray: Forecasted conditional variances
    """
    # Compute FIGARCH weights
    weights = _figarch_weights(d, phi, beta, trunc_lag)
    
    # Compute unconditional variance
    unconditional_variance = omega / (1.0 - np.sum(weights[1:]))
    
    # Initialize forecasts
    forecasts = np.zeros(steps)
    
    # Create array for future squared returns
    future_data = np.zeros(trunc_lag + steps)
    
    # Fill in known values
    for i in range(min(trunc_lag, len(last_data))):
        future_data[i] = last_data[i]
    
    # For lags beyond available data, use unconditional variance
    for i in range(len(last_data), trunc_lag):
        future_data[i] = unconditional_variance
    
    # Generate forecasts
    for h in range(steps):
        # Start with constant term
        forecasts[h] = omega
        
        # Add weighted squared residuals/forecasts
        for i in range(1, trunc_lag + 1):
            if h + i <= trunc_lag:
                # Use known data
                forecasts[h] += weights[i] * future_data[h+i-1]
            else:
                # Use unconditional variance for future innovations
                forecasts[h] += weights[i] * unconditional_variance
        
        # Update future data for next iteration
        future_data[trunc_lag + h] = forecasts[h]
    
    return forecasts


from dataclasses import dataclass

@dataclass
class FIGARCHParams(UnivariateVolatilityParameters):
    """Parameters for FIGARCH(p,d,q) model.
    
    This class defines the parameters for the FIGARCH model, including
    validation of parameter constraints.
    
    Attributes:
        omega: Constant term in variance equation (must be positive)
        d: Fractional differencing parameter (must be between 0 and 1)
        phi: ARCH parameters (must satisfy stationarity constraints)
        beta: GARCH parameters (must satisfy stationarity constraints)
    """
    
    omega: float
    d: float
    phi: Union[float, np.ndarray]
    beta: Union[float, np.ndarray]
    
    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        # Convert phi and beta to arrays if they are scalars
        if isinstance(self.phi, (int, float)):
            self.phi = np.array([self.phi])
        elif not isinstance(self.phi, np.ndarray):
            self.phi = np.array(self.phi)
            
        if isinstance(self.beta, (int, float)):
            self.beta = np.array([self.beta])
        elif not isinstance(self.beta, np.ndarray):
            self.beta = np.array(self.beta)
        
        self.validate()
    
    def validate(self) -> None:
        """Validate FIGARCH parameter constraints.
        
        Raises:
            ParameterError: If parameter constraints are violated
        """
        # Validate omega
        validate_positive(self.omega, "omega")
        
        # Validate fractional differencing parameter
        validate_range(self.d, "d", 0.0, 1.0)
        
        # Validate phi and beta
        for i, p in enumerate(self.phi):
            validate_non_negative(p, f"phi[{i}]")
        
        for i, b in enumerate(self.beta):
            validate_non_negative(b, f"beta[{i}]")
        
        # Validate stationarity constraints
        # For FIGARCH, the constraints are more complex
        # We check that the roots of the characteristic polynomials are outside the unit circle
        
        # Check phi(L) = 1 - phi_1*L - phi_2*L^2 - ... - phi_p*L^p
        if len(self.phi) > 0:
            phi_poly = np.concatenate([np.array([1.0]), -self.phi])
            phi_roots = np.roots(phi_poly)
            if np.any(np.abs(phi_roots) <= 1.0):
                raise ParameterError(
                    "FIGARCH stationarity constraint violated: "
                    "roots of phi(L) polynomial must be outside the unit circle"
                )
        
        # Check beta(L) = 1 - beta_1*L - beta_2*L^2 - ... - beta_q*L^q
        if len(self.beta) > 0:
            beta_poly = np.concatenate([np.array([1.0]), -self.beta])
            beta_roots = np.roots(beta_poly)
            if np.any(np.abs(beta_roots) <= 1.0):
                raise ParameterError(
                    "FIGARCH stationarity constraint violated: "
                    "roots of beta(L) polynomial must be outside the unit circle"
                )
        
        # Check additional FIGARCH constraints
        # The sum of the ARCH(∞) coefficients should be less than 1
        # This is difficult to check directly, so we'll use a simplified check
        if len(self.phi) > 0 and len(self.beta) > 0:
            if np.sum(self.phi) >= np.sum(self.beta) + self.d:
                raise ParameterError(
                    "FIGARCH stationarity constraint violated: "
                    "sum(phi) must be less than sum(beta) + d"
                )
    
    def to_array(self) -> np.ndarray:
        """Convert parameters to a NumPy array.
        
        Returns:
            np.ndarray: Array representation of parameters
        """
        return np.concatenate([
            np.array([self.omega, self.d]),
            self.phi,
            self.beta
        ])
    
    @classmethod
    def from_array(cls, array: np.ndarray, p: int = 1, q: int = 1, **kwargs: Any) -> 'FIGARCHParams':
        """Create parameters from a NumPy array.
        
        Args:
            array: Array representation of parameters
            p: Order of the ARCH component
            q: Order of the GARCH component
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            FIGARCHParams: Parameter object
        
        Raises:
            ValueError: If the array length doesn't match the expected length
        """
        expected_length = 2 + p + q  # omega, d, phi, beta
        if len(array) != expected_length:
            raise ValueError(
                f"Array length ({len(array)}) doesn't match expected length ({expected_length})"
            )
        
        # Extract parameters
        omega = array[0]
        d = array[1]
        phi = array[2:2+p]
        beta = array[2+p:]
        
        return cls(omega=omega, d=d, phi=phi, beta=beta)
    
    def transform(self) -> np.ndarray:
        """Transform parameters to unconstrained space for optimization.
        
        Returns:
            np.ndarray: Transformed parameters in unconstrained space
        """
        from mfe.core.parameters import transform_positive, transform_probability
        
        # Transform omega to unconstrained space (log)
        transformed_omega = transform_positive(self.omega)
        
        # Transform d to unconstrained space (logit)
        transformed_d = transform_probability(self.d)
        
        # Transform phi and beta
        # We use a transformation that ensures stationarity constraints
        # This is a simplification; a more complex transformation might be needed
        
        # For phi, we ensure sum(phi) < sum(beta) + d
        phi_sum = np.sum(self.phi)
        beta_sum = np.sum(self.beta)
        max_phi_sum = beta_sum + self.d
        
        if phi_sum >= max_phi_sum:
            # If constraint is violated, adjust parameters slightly
            factor = 0.99 * max_phi_sum / phi_sum
            phi_adjusted = self.phi * factor
        else:
            phi_adjusted = self.phi.copy()
        
        # Transform phi weights to unconstrained space
        phi_weights = phi_adjusted / max_phi_sum if max_phi_sum > 0 else np.ones(len(phi_adjusted)) / len(phi_adjusted)
        transformed_phi_weights = np.zeros(len(phi_weights) - 1) if len(phi_weights) > 1 else np.array([])
        
        if len(phi_weights) > 1:
            remaining = 1.0
            for i in range(len(phi_weights) - 1):
                if remaining > 0:
                    proportion = phi_weights[i] / remaining
                    transformed_phi_weights[i] = transform_probability(proportion)
                    remaining -= phi_weights[i]
        
        # Transform beta weights to unconstrained space
        beta_weights = self.beta / np.sum(self.beta) if np.sum(self.beta) > 0 else np.ones(len(self.beta)) / len(self.beta)
        transformed_beta_weights = np.zeros(len(beta_weights) - 1) if len(beta_weights) > 1 else np.array([])
        
        if len(beta_weights) > 1:
            remaining = 1.0
            for i in range(len(beta_weights) - 1):
                if remaining > 0:
                    proportion = beta_weights[i] / remaining
                    transformed_beta_weights[i] = transform_probability(proportion)
                    remaining -= beta_weights[i]
        
        # Transform phi_sum and beta_sum
        transformed_phi_sum = transform_probability(phi_sum / max_phi_sum) if max_phi_sum > 0 else 0.0
        transformed_beta_sum = transform_positive(beta_sum)
        
        # Combine all transformed parameters
        return np.concatenate([
            np.array([transformed_omega, transformed_d, transformed_phi_sum, transformed_beta_sum]),
            transformed_phi_weights,
            transformed_beta_weights
        ])
    
    @classmethod
    def inverse_transform(cls, array: np.ndarray, p: int = 1, q: int = 1, **kwargs: Any) -> 'FIGARCHParams':
        """Transform parameters from unconstrained space back to constrained space.
        
        Args:
            array: Parameters in unconstrained space
            p: Order of the ARCH component
            q: Order of the GARCH component
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            FIGARCHParams: Parameter object with constrained parameters
        
        Raises:
            ValueError: If the array length doesn't match the expected length
        """
        from mfe.core.parameters import inverse_transform_positive, inverse_transform_probability
        
        expected_length = 4 + (p - 1) + (q - 1)  # omega, d, phi_sum, beta_sum, phi_weights, beta_weights
        if len(array) != expected_length:
            raise ValueError(
                f"Array length ({len(array)}) doesn't match expected length ({expected_length})"
            )
        
        # Extract transformed parameters
        transformed_omega = array[0]
        transformed_d = array[1]
        transformed_phi_sum = array[2]
        transformed_beta_sum = array[3]
        
        # Extract transformed weights
        transformed_phi_weights = array[4:4+(p-1)] if p > 1 else np.array([])
        transformed_beta_weights = array[4+(p-1):] if q > 1 else np.array([])
        
        # Inverse transform omega and d
        omega = inverse_transform_positive(transformed_omega)
        d = inverse_transform_probability(transformed_d)
        
        # Inverse transform beta_sum
        beta_sum = inverse_transform_positive(transformed_beta_sum)
        
        # Compute max_phi_sum
        max_phi_sum = beta_sum + d
        
        # Inverse transform phi_sum
        phi_sum = max_phi_sum * inverse_transform_probability(transformed_phi_sum)
        
        # Inverse transform weights to get phi and beta
        phi = np.zeros(p)
        beta = np.zeros(q)
        
        if p == 1:
            phi[0] = phi_sum
        else:
            # Reconstruct phi weights from transformed values
            phi_weights = np.zeros(p)
            remaining = 1.0
            for i in range(p - 1):
                proportion = inverse_transform_probability(transformed_phi_weights[i])
                phi_weights[i] = proportion * remaining
                remaining -= phi_weights[i]
            phi_weights[p - 1] = remaining
            
            # Scale weights by phi_sum
            phi = phi_weights * phi_sum
        
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
        
        return cls(omega=omega, d=d, phi=phi, beta=beta)


class FIGARCH(VolatilityModel):
    """FIGARCH (Fractionally Integrated GARCH) model.
    
    This class implements the FIGARCH(p,d,q) model for modeling long memory in
    volatility. The model is defined as:
    
    (1 - β(L))σ²_t = ω + (1 - β(L) - φ(L)(1-L)^d)ε²_t
    
    where σ²_t is the conditional variance at time t, ε²_t are squared
    residuals, L is the lag operator, d is the fractional differencing parameter,
    φ(L) = φ_1*L + φ_2*L^2 + ... + φ_p*L^p, and β(L) = β_1*L + β_2*L^2 + ... + β_q*L^q.
    
    The FIGARCH model can be rewritten as an ARCH(∞) process:
    
    σ²_t = ω/(1-β(1)) + λ(L)ε²_t
    
    where λ(L) = λ_1*L + λ_2*L^2 + ... is an infinite lag polynomial.
    """
    
    def __init__(self, 
                 p: int = 1, 
                 d: Optional[float] = None,
                 q: int = 1, 
                 trunc_lag: int = 1000,
                 parameters: Optional[FIGARCHParams] = None, 
                 name: str = "FIGARCH") -> None:
        """Initialize the FIGARCH model.
        
        Args:
            p: Order of the ARCH component
            d: Fractional differencing parameter (if None, estimated during fitting)
            q: Order of the GARCH component
            trunc_lag: Truncation lag for the infinite ARCH representation
            parameters: Pre-specified model parameters if available
            name: A descriptive name for the model
        
        Raises:
            ValueError: If p or q is not a positive integer, or if d is not between 0 and 1
        """
        if not isinstance(p, int) or p <= 0:
            raise ValueError(f"p must be a positive integer, got {p}")
        if not isinstance(q, int) or q <= 0:
            raise ValueError(f"q must be a positive integer, got {q}")
        if d is not None and (d <= 0 or d >= 1):
            raise ValueError(f"d must be between 0 and 1, got {d}")
        if not isinstance(trunc_lag, int) or trunc_lag <= 0:
            raise ValueError(f"trunc_lag must be a positive integer, got {trunc_lag}")
        
        self.p = p
        self.q = q
        self.d = d
        self.trunc_lag = trunc_lag
        
        # Set model name based on p, d, and q
        if name == "FIGARCH":
            name = f"FIGARCH({p},{d if d is not None else 'd'},{q})"
        
        super().__init__(parameters=parameters, name=name)
        
        # Initialize additional attributes
        self._conditional_variances = None
        self._weights = None
    
    def parameter_class(self) -> Type[FIGARCHParams]:
        """Get the parameter class for this model.
        
        Returns:
            Type[FIGARCHParams]: The parameter class for this model
        """
        return FIGARCHParams
    
    def compute_variance(self, 
                         parameters: FIGARCHParams, 
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
            omega = parameters.omega
            d = parameters.d
            phi = parameters.phi
            beta = parameters.beta
            
            sigma2 = _figarch_recursion(data, omega, d, phi, beta, sigma2, backcast, self.trunc_lag)
            
            # Store the weights for later use
            self._weights = _figarch_weights(d, phi, beta, self.trunc_lag)
            
            # Check for numerical issues
            if np.any(~np.isfinite(sigma2)) or np.any(sigma2 <= 0):
                raise_numeric_error(
                    "Numerical issues detected in variance computation",
                    operation="FIGARCH variance computation",
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
                operation="FIGARCH variance computation",
                error_type="Computation error",
                details=str(e)
            )
    
    def _generate_starting_values(self, 
                                 data: np.ndarray, 
                                 variance_targeting: bool = False,
                                 backcast: Optional[float] = None) -> FIGARCHParams:
        """Generate starting values for parameter estimation.
        
        Args:
            data: Input data (typically residuals)
            variance_targeting: Whether to use variance targeting
            backcast: Value to use for initializing the variance process
        
        Returns:
            FIGARCHParams: Starting parameter values
        """
        # Compute sample variance
        sample_variance = np.var(data)
        
        if backcast is None:
            backcast = sample_variance
        
        # Generate starting values based on model order
        # For FIGARCH, we need to be careful with the parameter constraints
        
        # Start with reasonable defaults
        d_start = 0.3 if self.d is None else self.d
        
        # Generate phi and beta with exponential decay
        phi = np.zeros(self.p)
        beta = np.zeros(self.q)
        
        # Set phi and beta values
        phi_sum = 0.2
        beta_sum = 0.6
        
        # Distribute phi values with exponential decay
        if self.p > 0:
            phi_weights = np.exp(-np.arange(self.p))
            phi_weights = phi_weights / np.sum(phi_weights)
            phi = phi_weights * phi_sum
        
        # Distribute beta values with exponential decay
        if self.q > 0:
            beta_weights = np.exp(-np.arange(self.q))
            beta_weights = beta_weights / np.sum(beta_weights)
            beta = beta_weights * beta_sum
        
        # Compute omega based on variance targeting or use a reasonable default
        if variance_targeting:
            # For FIGARCH, variance targeting is more complex
            # We use the formula: omega = sample_variance * (1 - sum(lambda))
            # where lambda are the ARCH(∞) weights
            
            # Compute approximate weights
            approx_weights = _figarch_weights(d_start, phi, beta, self.trunc_lag)
            weight_sum = np.sum(approx_weights[1:])
            
            # Ensure weight_sum < 1 for stationarity
            if weight_sum >= 1:
                weight_sum = 0.99
            
            omega = sample_variance * (1 - weight_sum)
        else:
            # Without variance targeting, use a reasonable default
            omega = 0.05 * sample_variance
        
        # Ensure omega is positive
        omega = max(omega, 1e-6)
        
        return FIGARCHParams(omega=omega, d=d_start, phi=phi, beta=beta)
    
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
        
        # For FIGARCH model, unconditional variance is omega / (1 - sum(lambda))
        # where lambda are the ARCH(∞) weights
        
        # If weights have been computed, use them
        if self._weights is not None:
            weight_sum = np.sum(self._weights[1:])
            
            # Ensure weight_sum < 1 for stationarity
            if weight_sum >= 1:
                # If not stationary, return the last observed variance
                if self._conditional_variances is not None:
                    return self._conditional_variances[-1]
                else:
                    # Fallback to a reasonable value
                    return self._parameters.omega / 0.05
            
            return self._parameters.omega / (1 - weight_sum)
        else:
            # If weights haven't been computed, compute them now
            omega = self._parameters.omega
            d = self._parameters.d
            phi = self._parameters.phi
            beta = self._parameters.beta
            
            weights = _figarch_weights(d, phi, beta, self.trunc_lag)
            weight_sum = np.sum(weights[1:])
            
            # Ensure weight_sum < 1 for stationarity
            if weight_sum >= 1:
                # If not stationary, return the last observed variance
                if self._conditional_variances is not None:
                    return self._conditional_variances[-1]
                else:
                    # Fallback to a reasonable value
                    return omega / 0.05
            
            return omega / (1 - weight_sum)
    
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
        d = self._parameters.d
        phi = self._parameters.phi
        beta = self._parameters.beta
        
        # Compute variance for time t using the ARCH(∞) representation
        # If weights haven't been computed, compute them now
        if self._weights is None:
            self._weights = _figarch_weights(d, phi, beta, self.trunc_lag)
        
        # Start with constant term
        variance = omega
        
        # Add weighted squared residuals
        for i in range(1, min(t, self.trunc_lag) + 1):
            variance += self._weights[i] * innovations[t-i]**2
        
        # For lags beyond available data, use unconditional variance
        if t < self.trunc_lag:
            unconditional_variance = self._compute_unconditional_variance()
            remaining_weight = np.sum(self._weights[t+1:self.trunc_lag+1])
            variance += remaining_weight * unconditional_variance
        
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
        d = self._parameters.d
        phi = self._parameters.phi
        beta = self._parameters.beta
        
        # For FIGARCH forecasting, we need the last trunc_lag squared returns
        # We'll use the conditional variances as a proxy for squared returns
        if self._conditional_variances is None:
            raise RuntimeError("Conditional variances not available for forecasting")
        
        # Get the last trunc_lag squared returns (most recent first)
        last_data = np.zeros(min(self.trunc_lag, len(self._conditional_variances)))
        for i in range(len(last_data)):
            last_data[i] = self._conditional_variances[-(i+1)]
        
        # Use optimized Numba implementation for forecasting
        return _figarch_forecast(omega, d, phi, beta, last_data, last_variance, steps, self.trunc_lag)
    
    def fit(self, 
            data: np.ndarray, 
            starting_values: Optional[Union[np.ndarray, FIGARCHParams]] = None,
            distribution: Literal["normal", "t", "skewed-t", "ged"] = "normal",
            variance_targeting: bool = False,
            backcast: Optional[float] = None,
            method: str = "SLSQP",
            options: Optional[Dict[str, Any]] = None,
            **kwargs: Any) -> 'UnivariateVolatilityResult':
        """Fit the FIGARCH model to the provided data.
        
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
        # Store the data for later use in forecasting
        self._data = data
        
        # Call the parent class implementation
        result = super().fit(
            data, 
            starting_values, 
            distribution, 
            variance_targeting, 
            backcast, 
            method, 
            options, 
            **kwargs
        )
        
        # Update the model name with the estimated d parameter
        if self.d is None and self._parameters is not None:
            self.name = f"FIGARCH({self.p},{self._parameters.d:.3f},{self.q})"
        
        return result
    
    async def fit_async(self, 
                       data: np.ndarray, 
                       starting_values: Optional[Union[np.ndarray, FIGARCHParams]] = None,
                       distribution: Literal["normal", "t", "skewed-t", "ged"] = "normal",
                       variance_targeting: bool = False,
                       backcast: Optional[float] = None,
                       method: str = "SLSQP",
                       options: Optional[Dict[str, Any]] = None,
                       progress_callback: Optional[Callable[[float, str], None]] = None,
                       **kwargs: Any) -> 'UnivariateVolatilityResult':
        """Asynchronously fit the FIGARCH model to the provided data.
        
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
        # Store the data for later use in forecasting
        self._data = data
        
        # Call the parent class implementation
        result = await super().fit_async(
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
        
        # Update the model name with the estimated d parameter
        if self.d is None and self._parameters is not None:
            self.name = f"FIGARCH({self.p},{self._parameters.d:.3f},{self.q})"
        
        return result
    
    def simulate(self, 
                n_periods: int, 
                burn: int = 500, 
                initial_values: Optional[np.ndarray] = None,
                random_state: Optional[Union[int, np.random.Generator]] = None,
                distribution: Literal["normal", "t", "skewed-t", "ged"] = "normal",
                distribution_params: Optional[Dict[str, Any]] = None,
                **kwargs: Any) -> np.ndarray:
        """Simulate data from the FIGARCH model.
        
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
        """Asynchronously simulate data from the FIGARCH model.
        
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
        """Generate volatility forecasts from the fitted FIGARCH model.
        
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
        """Asynchronously generate volatility forecasts from the fitted FIGARCH model.
        
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
        """Validate input data for the FIGARCH model.
        
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
    
    def plot_diagnostics(self, figsize: Tuple[int, int] = (12, 10)) -> None:
        """Plot diagnostic plots for the fitted FIGARCH model.
        
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
        fig, axs = plt.subplots(3, 2, figsize=figsize)
        
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
        
        # Plot FIGARCH weights
        if self._weights is not None:
            # Plot first 100 weights
            plot_lags = min(100, len(self._weights) - 1)
            axs[2, 0].bar(range(1, plot_lags + 1), self._weights[1:plot_lags+1])
            axs[2, 0].set_title(f'FIGARCH Weights (first {plot_lags} lags)')
            axs[2, 0].set_xlabel('Lag')
            axs[2, 0].set_ylabel('Weight')
            
            # Plot cumulative sum of weights
            cum_weights = np.cumsum(self._weights[1:])
            axs[2, 1].plot(range(1, len(cum_weights) + 1), cum_weights)
            axs[2, 1].axhline(y=1.0, color='r', linestyle='--')
            axs[2, 1].set_title('Cumulative Sum of FIGARCH Weights')
            axs[2, 1].set_xlabel('Lag')
            axs[2, 1].set_ylabel('Cumulative Sum')
            
            # Use log scale for x-axis to better see the long memory property
            axs[2, 1].set_xscale('log')
        
        # Adjust layout and show plot
        plt.tight_layout()
        plt.show()
    
    def __str__(self) -> str:
        """Get a string representation of the FIGARCH model.
        
        Returns:
            str: String representation of the model
        """
        if not self._fitted or self._parameters is None:
            return f"{self.name} model (not fitted)"
        
        # Extract parameters
        omega = self._parameters.omega
        d = self._parameters.d
        phi = self._parameters.phi
        beta = self._parameters.beta
        
        # Format parameters
        phi_str = ", ".join([f"{p:.4f}" for p in phi])
        beta_str = ", ".join([f"{b:.4f}" for b in beta])
        
        # Create string representation
        model_str = f"{self.name} model\n"
        model_str += f"omega: {omega:.6f}\n"
        model_str += f"d: {d:.6f}\n"
        model_str += f"phi: [{phi_str}]\n"
        model_str += f"beta: [{beta_str}]\n"
        
        # Add truncation lag
        model_str += f"truncation lag: {self.trunc_lag}\n"
        
        # Add unconditional variance
        try:
            uncond_var = self._compute_unconditional_variance()
            model_str += f"unconditional variance: {uncond_var:.6f}\n"
        except Exception:
            # If unconditional variance can't be computed, skip it
            pass
        
        return model_str
    
    def __repr__(self) -> str:
        """Get a string representation of the FIGARCH model.
        
        Returns:
            str: String representation of the model
        """
        return self.__str__()
