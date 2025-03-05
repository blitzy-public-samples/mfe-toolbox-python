# mfe/models/univariate/heavy.py
"""
High-frEquency-bAsed VolatilitY (HEAVY) model implementation.

This module provides a comprehensive implementation of the HEAVY model for
incorporating realized measures in volatility modeling. The HEAVY model extends
traditional GARCH-type models by allowing high-frequency realized volatility
measures to directly influence the conditional variance process.

The implementation includes parameter estimation via maximum likelihood,
simulation, forecasting, and diagnostic tools. Performance-critical operations
are accelerated using Numba's just-in-time compilation.

References:
    Shephard, N., & Sheppard, K. (2010). Realising the future: forecasting with
    high-frequency-based volatility (HEAVY) models. Journal of Applied Econometrics,
    25(2), 197-231.
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
    UnivariateVolatilityParameters, ParameterError,
    validate_positive, validate_non_negative, validate_probability, validate_range,
    transform_positive, inverse_transform_positive, transform_probability, 
    inverse_transform_probability
)
from mfe.core.exceptions import (
    MFEError, ConvergenceError, NumericError, EstimationError, SimulationError,
    NotFittedError, raise_convergence_error, raise_numeric_error, raise_not_fitted_error,
    warn_convergence, warn_numeric, warn_model
)
from mfe.models.univariate.base import VolatilityModel


# Numba-accelerated core functions for HEAVY model
@jit(nopython=True, cache=True)
def _heavy_recursion(returns: np.ndarray, 
                    realized: np.ndarray,
                    omega_r: float, 
                    alpha_r: float, 
                    beta_r: float,
                    omega_h: float,
                    alpha_h: float,
                    beta_h: float,
                    sigma2_r: np.ndarray,
                    sigma2_h: np.ndarray,
                    backcast_r: float,
                    backcast_h: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute HEAVY model conditional variances using Numba acceleration.
    
    Args:
        returns: Return data
        realized: Realized volatility measures
        omega_r: Constant term in realized equation
        alpha_r: Impact of lagged realized measure in realized equation
        beta_r: Persistence parameter in realized equation
        omega_h: Constant term in return equation
        alpha_h: Impact of lagged realized measure in return equation
        beta_h: Persistence parameter in return equation
        sigma2_r: Pre-allocated array for realized conditional variances
        sigma2_h: Pre-allocated array for return conditional variances
        backcast_r: Value to use for initializing the realized variance process
        backcast_h: Value to use for initializing the return variance process
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Realized and return conditional variances
    """
    T = len(returns)
    
    # Initialize first variance with backcast values
    sigma2_r[0] = backcast_r
    sigma2_h[0] = backcast_h
    
    # Compute conditional variances recursively
    for t in range(1, T):
        # Realized equation
        sigma2_r[t] = omega_r + alpha_r * realized[t-1] + beta_r * sigma2_r[t-1]
        
        # Return equation
        sigma2_h[t] = omega_h + alpha_h * realized[t-1] + beta_h * sigma2_h[t-1]
    
    return sigma2_r, sigma2_h


@jit(nopython=True, cache=True)
def _heavy_forecast(omega_r: float, 
                   alpha_r: float, 
                   beta_r: float,
                   omega_h: float,
                   alpha_h: float,
                   beta_h: float,
                   last_realized: float,
                   last_variance_r: float,
                   last_variance_h: float,
                   steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate analytic forecasts for HEAVY model using Numba acceleration.
    
    Args:
        omega_r: Constant term in realized equation
        alpha_r: Impact of lagged realized measure in realized equation
        beta_r: Persistence parameter in realized equation
        omega_h: Constant term in return equation
        alpha_h: Impact of lagged realized measure in return equation
        beta_h: Persistence parameter in return equation
        last_realized: Last observed realized measure
        last_variance_r: Last observed realized conditional variance
        last_variance_h: Last observed return conditional variance
        steps: Number of steps to forecast
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Forecasted realized and return conditional variances
    """
    forecasts_r = np.zeros(steps)
    forecasts_h = np.zeros(steps)
    
    # Compute persistence
    persistence_r = alpha_r + beta_r
    persistence_h = beta_h
    
    # Compute unconditional variances
    unconditional_variance_r = omega_r / (1 - persistence_r) if persistence_r < 1 else last_variance_r
    unconditional_variance_h = omega_h / (1 - persistence_h) if persistence_h < 1 else last_variance_h
    
    # First step forecast
    forecasts_r[0] = omega_r + alpha_r * last_realized + beta_r * last_variance_r
    forecasts_h[0] = omega_h + alpha_h * last_realized + beta_h * last_variance_h
    
    # Multi-step forecasts
    for h in range(1, steps):
        # For realized equation, we use the previous forecast
        forecasts_r[h] = omega_r + alpha_r * forecasts_r[h-1] + beta_r * forecasts_r[h-1]
        
        # For return equation, we use the previous forecast of realized measure
        forecasts_h[h] = omega_h + alpha_h * forecasts_r[h-1] + beta_h * forecasts_h[h-1]
        
        # For long horizons, approach the unconditional variance
        if persistence_r < 1 and h > 100:
            forecasts_r[h] = unconditional_variance_r - (unconditional_variance_r - forecasts_r[h]) * 0.1
        
        if persistence_h < 1 and h > 100:
            forecasts_h[h] = unconditional_variance_h - (unconditional_variance_h - forecasts_h[h]) * 0.1
    
    return forecasts_r, forecasts_h


@dataclass
class HEAVYParams(UnivariateVolatilityParameters):
    """Parameters for HEAVY model.
    
    This class defines the parameters for the HEAVY model, which incorporates
    realized volatility measures into the conditional variance process.
    
    Attributes:
        omega_r: Constant term in realized equation (must be positive)
        alpha_r: Impact of lagged realized measure in realized equation (must be non-negative)
        beta_r: Persistence parameter in realized equation (must be non-negative)
        omega_h: Constant term in return equation (must be positive)
        alpha_h: Impact of lagged realized measure in return equation (must be non-negative)
        beta_h: Persistence parameter in return equation (must be non-negative)
    """
    
    omega_r: float
    alpha_r: float
    beta_r: float
    omega_h: float
    alpha_h: float
    beta_h: float
    
    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate HEAVY parameter constraints.
        
        Raises:
            ParameterError: If parameter constraints are violated
        """
        super().validate()
        
        # Validate individual parameters
        validate_positive(self.omega_r, "omega_r")
        validate_non_negative(self.alpha_r, "alpha_r")
        validate_non_negative(self.beta_r, "beta_r")
        validate_positive(self.omega_h, "omega_h")
        validate_non_negative(self.alpha_h, "alpha_h")
        validate_non_negative(self.beta_h, "beta_h")
        
        # Validate stationarity constraints
        if self.alpha_r + self.beta_r >= 1:
            raise ParameterError(
                f"HEAVY realized equation stationarity constraint violated: "
                f"alpha_r + beta_r = {self.alpha_r + self.beta_r} >= 1"
            )
        
        if self.beta_h >= 1:
            raise ParameterError(
                f"HEAVY return equation stationarity constraint violated: "
                f"beta_h = {self.beta_h} >= 1"
            )
    
    def to_array(self) -> np.ndarray:
        """Convert parameters to a NumPy array.
        
        Returns:
            np.ndarray: Array representation of parameters
        """
        return np.array([
            self.omega_r, self.alpha_r, self.beta_r,
            self.omega_h, self.alpha_h, self.beta_h
        ])
    
    @classmethod
    def from_array(cls, array: np.ndarray, **kwargs: Any) -> 'HEAVYParams':
        """Create parameters from a NumPy array.
        
        Args:
            array: Array representation of parameters
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            HEAVYParams: Parameter object
        
        Raises:
            ValueError: If the array length is not 6
        """
        if len(array) != 6:
            raise ValueError(f"Array length must be 6, got {len(array)}")
        
        return cls(
            omega_r=array[0],
            alpha_r=array[1],
            beta_r=array[2],
            omega_h=array[3],
            alpha_h=array[4],
            beta_h=array[5]
        )
    
    def transform(self) -> np.ndarray:
        """Transform parameters to unconstrained space for optimization.
        
        Returns:
            np.ndarray: Transformed parameters in unconstrained space
        """
        # Transform omega parameters to unconstrained space (log)
        transformed_omega_r = transform_positive(self.omega_r)
        transformed_omega_h = transform_positive(self.omega_h)
        
        # Transform alpha and beta parameters for realized equation
        # We use a special transformation to ensure alpha_r + beta_r < 1
        if self.alpha_r + self.beta_r >= 1:
            # If constraint is violated, adjust parameters slightly
            sum_ab = self.alpha_r + self.beta_r
            self.alpha_r = self.alpha_r / (sum_ab + 0.01)
            self.beta_r = self.beta_r / (sum_ab + 0.01)
        
        # Use logit-like transformation for alpha_r and beta_r
        gamma_r = self.alpha_r + self.beta_r
        delta_r = self.alpha_r / gamma_r if gamma_r > 0 else 0.5
        
        transformed_gamma_r = transform_probability(gamma_r)
        transformed_delta_r = transform_probability(delta_r)
        
        # Transform alpha_h and beta_h
        # beta_h must be < 1, alpha_h must be >= 0
        transformed_beta_h = transform_probability(self.beta_h)
        transformed_alpha_h = transform_positive(self.alpha_h + 1e-6) # Add small constant to ensure positivity
        
        return np.array([
            transformed_omega_r,
            transformed_gamma_r,
            transformed_delta_r,
            transformed_omega_h,
            transformed_alpha_h,
            transformed_beta_h
        ])
    
    @classmethod
    def inverse_transform(cls, array: np.ndarray, **kwargs: Any) -> 'HEAVYParams':
        """Transform parameters from unconstrained space back to constrained space.
        
        Args:
            array: Parameters in unconstrained space
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            HEAVYParams: Parameter object with constrained parameters
        
        Raises:
            ValueError: If the array length is not 6
        """
        if len(array) != 6:
            raise ValueError(f"Array length must be 6, got {len(array)}")
        
        # Extract transformed parameters
        transformed_omega_r = array[0]
        transformed_gamma_r = array[1]
        transformed_delta_r = array[2]
        transformed_omega_h = array[3]
        transformed_alpha_h = array[4]
        transformed_beta_h = array[5]
        
        # Inverse transform omega parameters
        omega_r = inverse_transform_positive(transformed_omega_r)
        omega_h = inverse_transform_positive(transformed_omega_h)
        
        # Inverse transform gamma_r and delta_r
        gamma_r = inverse_transform_probability(transformed_gamma_r)
        delta_r = inverse_transform_probability(transformed_delta_r)
        
        # Compute alpha_r and beta_r
        alpha_r = gamma_r * delta_r
        beta_r = gamma_r * (1 - delta_r)
        
        # Inverse transform alpha_h and beta_h
        alpha_h = inverse_transform_positive(transformed_alpha_h) - 1e-6  # Subtract the small constant added during transform
        beta_h = inverse_transform_probability(transformed_beta_h)
        
        return cls(
            omega_r=omega_r,
            alpha_r=alpha_r,
            beta_r=beta_r,
            omega_h=omega_h,
            alpha_h=alpha_h,
            beta_h=beta_h
        )


class HEAVYModel(VolatilityModel):
    """HEAVY (High-frEquency-bAsed VolatilitY) model.
    
    This class implements the HEAVY model for incorporating realized volatility
    measures into the conditional variance process. The model consists of two
    equations:
    
    Realized equation:
    σ²_r,t = ω_r + α_r * RM_{t-1} + β_r * σ²_r,{t-1}
    
    Return equation:
    σ²_h,t = ω_h + α_h * RM_{t-1} + β_h * σ²_h,{t-1}
    
    where σ²_r,t is the conditional variance of the realized measure,
    σ²_h,t is the conditional variance of returns, and RM_t is the
    realized measure at time t.
    
    Attributes:
        name: Model name
        _parameters: Model parameters if set
        _results: Estimation results if the model has been fitted
        _conditional_variances_r: Conditional variances for realized measures
        _conditional_variances_h: Conditional variances for returns
    """
    
    def __init__(self, 
                 parameters: Optional[HEAVYParams] = None, 
                 name: str = "HEAVY") -> None:
        """Initialize the HEAVY model.
        
        Args:
            parameters: Pre-specified model parameters if available
            name: A descriptive name for the model
        """
        super().__init__(parameters=parameters, name=name)
        
        # Initialize additional attributes
        self._conditional_variances_r = None
        self._conditional_variances_h = None
        self._realized_data = None
        self._return_data = None
    
    def parameter_class(self) -> Type[HEAVYParams]:
        """Get the parameter class for this model.
        
        Returns:
            Type[HEAVYParams]: The parameter class for this model
        """
        return HEAVYParams
    
    def compute_variance(self, 
                         parameters: HEAVYParams, 
                         data: np.ndarray, 
                         sigma2: Optional[np.ndarray] = None,
                         backcast: Optional[float] = None) -> np.ndarray:
        """Compute conditional variances for the given parameters and data.
        
        For the HEAVY model, this method computes the conditional variances
        for the return equation only. To compute both realized and return
        conditional variances, use compute_variances() instead.
        
        Args:
            parameters: Model parameters
            data: Input data (typically returns)
            sigma2: Pre-allocated array for conditional variances
            backcast: Value to use for initializing the variance process
        
        Returns:
            np.ndarray: Conditional variances for returns
        
        Raises:
            ValueError: If the data is invalid
            NumericError: If numerical issues occur during computation
        """
        # Check if realized data is available
        if self._realized_data is None:
            raise ValueError(
                "Realized data is required for HEAVY model. "
                "Use fit() with both returns and realized data."
            )
        
        # Validate input data
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        
        if data.ndim != 1:
            raise ValueError(f"data must be a 1-dimensional array, got shape {data.shape}")
        
        # Ensure realized data and returns have the same length
        if len(self._realized_data) != len(data):
            raise ValueError(
                f"Realized data length ({len(self._realized_data)}) "
                f"must match returns length ({len(data)})"
            )
        
        T = len(data)
        
        # Allocate arrays for conditional variances if not provided
        if sigma2 is None:
            sigma2_h = np.zeros(T)
        elif len(sigma2) != T:
            raise ValueError(f"sigma2 must have length {T}, got {len(sigma2)}")
        else:
            sigma2_h = sigma2
        
        sigma2_r = np.zeros(T)
        
        # Compute backcast values if not provided
        if backcast is None:
            backcast_r = np.mean(self._realized_data)
            backcast_h = np.mean(data**2)
        else:
            backcast_r = backcast
            backcast_h = backcast
        
        try:
            # Use optimized Numba implementation
            sigma2_r, sigma2_h = _heavy_recursion(
                data, self._realized_data,
                parameters.omega_r, parameters.alpha_r, parameters.beta_r,
                parameters.omega_h, parameters.alpha_h, parameters.beta_h,
                sigma2_r, sigma2_h, backcast_r, backcast_h
            )
            
            # Check for numerical issues
            if (np.any(~np.isfinite(sigma2_r)) or np.any(sigma2_r <= 0) or
                np.any(~np.isfinite(sigma2_h)) or np.any(sigma2_h <= 0)):
                raise_numeric_error(
                    "Numerical issues detected in variance computation",
                    operation="HEAVY variance computation",
                    error_type="Invalid variance values",
                    details="Non-finite or non-positive variance values detected"
                )
            
            # Store realized conditional variances
            self._conditional_variances_r = sigma2_r
            
            return sigma2_h
            
        except NumericError as e:
            # Re-raise numeric errors
            raise e
        except Exception as e:
            # Wrap other exceptions in NumericError
            raise_numeric_error(
                f"Error during variance computation: {str(e)}",
                operation="HEAVY variance computation",
                error_type="Computation error",
                details=str(e)
            )
    
    def compute_variances(self, 
                         parameters: HEAVYParams, 
                         returns: np.ndarray,
                         realized: np.ndarray,
                         sigma2_r: Optional[np.ndarray] = None,
                         sigma2_h: Optional[np.ndarray] = None,
                         backcast_r: Optional[float] = None,
                         backcast_h: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute both realized and return conditional variances.
        
        Args:
            parameters: Model parameters
            returns: Return data
            realized: Realized volatility measures
            sigma2_r: Pre-allocated array for realized conditional variances
            sigma2_h: Pre-allocated array for return conditional variances
            backcast_r: Value to use for initializing the realized variance process
            backcast_h: Value to use for initializing the return variance process
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Realized and return conditional variances
        
        Raises:
            ValueError: If the data is invalid
            NumericError: If numerical issues occur during computation
        """
        # Validate input data
        if not isinstance(returns, np.ndarray):
            returns = np.asarray(returns)
        
        if not isinstance(realized, np.ndarray):
            realized = np.asarray(realized)
        
        if returns.ndim != 1:
            raise ValueError(f"returns must be a 1-dimensional array, got shape {returns.shape}")
        
        if realized.ndim != 1:
            raise ValueError(f"realized must be a 1-dimensional array, got shape {realized.shape}")
        
        # Ensure realized data and returns have the same length
        if len(realized) != len(returns):
            raise ValueError(
                f"Realized data length ({len(realized)}) "
                f"must match returns length ({len(returns)})"
            )
        
        T = len(returns)
        
        # Allocate arrays for conditional variances if not provided
        if sigma2_r is None:
            sigma2_r = np.zeros(T)
        elif len(sigma2_r) != T:
            raise ValueError(f"sigma2_r must have length {T}, got {len(sigma2_r)}")
        
        if sigma2_h is None:
            sigma2_h = np.zeros(T)
        elif len(sigma2_h) != T:
            raise ValueError(f"sigma2_h must have length {T}, got {len(sigma2_h)}")
        
        # Compute backcast values if not provided
        if backcast_r is None:
            backcast_r = np.mean(realized)
        
        if backcast_h is None:
            backcast_h = np.mean(returns**2)
        
        try:
            # Use optimized Numba implementation
            sigma2_r, sigma2_h = _heavy_recursion(
                returns, realized,
                parameters.omega_r, parameters.alpha_r, parameters.beta_r,
                parameters.omega_h, parameters.alpha_h, parameters.beta_h,
                sigma2_r, sigma2_h, backcast_r, backcast_h
            )
            
            # Check for numerical issues
            if (np.any(~np.isfinite(sigma2_r)) or np.any(sigma2_r <= 0) or
                np.any(~np.isfinite(sigma2_h)) or np.any(sigma2_h <= 0)):
                raise_numeric_error(
                    "Numerical issues detected in variance computation",
                    operation="HEAVY variance computation",
                    error_type="Invalid variance values",
                    details="Non-finite or non-positive variance values detected"
                )
            
            return sigma2_r, sigma2_h
            
        except NumericError as e:
            # Re-raise numeric errors
            raise e
        except Exception as e:
            # Wrap other exceptions in NumericError
            raise_numeric_error(
                f"Error during variance computation: {str(e)}",
                operation="HEAVY variance computation",
                error_type="Computation error",
                details=str(e)
            )
    
    def _generate_starting_values(self, 
                                 data: np.ndarray, 
                                 variance_targeting: bool = False,
                                 backcast: Optional[float] = None) -> HEAVYParams:
        """Generate starting values for parameter estimation.
        
        Args:
            data: Input data (typically returns)
            variance_targeting: Whether to use variance targeting
            backcast: Value to use for initializing the variance process
        
        Returns:
            HEAVYParams: Starting parameter values
        
        Raises:
            ValueError: If realized data is not available
        """
        # Check if realized data is available
        if self._realized_data is None:
            raise ValueError(
                "Realized data is required for HEAVY model. "
                "Use fit() with both returns and realized data."
            )
        
        # Compute sample variances
        return_variance = np.var(data)
        realized_variance = np.mean(self._realized_data)
        
        if backcast is None:
            backcast_r = realized_variance
            backcast_h = return_variance
        else:
            backcast_r = backcast
            backcast_h = backcast
        
        # Generate starting values
        if variance_targeting:
            # With variance targeting, omega is determined by the other parameters
            alpha_r = 0.4
            beta_r = 0.5
            omega_r = realized_variance * (1 - alpha_r - beta_r)
            
            alpha_h = 0.2
            beta_h = 0.7
            omega_h = return_variance * (1 - beta_h) - alpha_h * realized_variance
        else:
            # Without variance targeting, use reasonable defaults
            omega_r = 0.1 * realized_variance
            alpha_r = 0.4
            beta_r = 0.5
            
            omega_h = 0.1 * return_variance
            alpha_h = 0.2
            beta_h = 0.7
        
        # Ensure omega_h is positive
        omega_h = max(omega_h, 1e-6)
        
        return HEAVYParams(
            omega_r=omega_r,
            alpha_r=alpha_r,
            beta_r=beta_r,
            omega_h=omega_h,
            alpha_h=alpha_h,
            beta_h=beta_h
        )
    
    def _compute_unconditional_variance(self) -> Tuple[float, float]:
        """Compute the unconditional variance of the process.
        
        Returns:
            Tuple[float, float]: Unconditional variances for realized and return equations
        
        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self._fitted or self._parameters is None:
            raise_not_fitted_error(
                "Model must be fitted before computing unconditional variance.",
                model_type=self.name,
                operation="compute_unconditional_variance"
            )
        
        # For HEAVY model, unconditional variances are:
        # Realized: omega_r / (1 - alpha_r - beta_r)
        # Return: (omega_h + alpha_h * unconditional_realized) / (1 - beta_h)
        
        omega_r = self._parameters.omega_r
        alpha_r = self._parameters.alpha_r
        beta_r = self._parameters.beta_r
        
        omega_h = self._parameters.omega_h
        alpha_h = self._parameters.alpha_h
        beta_h = self._parameters.beta_h
        
        persistence_r = alpha_r + beta_r
        
        if persistence_r >= 1:
            # If the realized process is not stationary, use the last observed variance
            if self._conditional_variances_r is not None:
                unconditional_r = self._conditional_variances_r[-1]
            else:
                # Fallback to a reasonable value
                unconditional_r = omega_r / 0.05
        else:
            unconditional_r = omega_r / (1 - persistence_r)
        
        if beta_h >= 1:
            # If the return process is not stationary, use the last observed variance
            if self._conditional_variances_h is not None:
                unconditional_h = self._conditional_variances_h[-1]
            else:
                # Fallback to a reasonable value
                unconditional_h = omega_h / 0.05
        else:
            unconditional_h = (omega_h + alpha_h * unconditional_r) / (1 - beta_h)
        
        return unconditional_r, unconditional_h
    
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
            NotImplementedError: HEAVY model simulation requires both realized and return processes
        """
        raise NotImplementedError(
            "HEAVY model simulation requires both realized and return processes. "
            "Use simulate_variances() instead."
        )
    
    def simulate_variances(self, 
                          t: int, 
                          realized: np.ndarray,
                          innovations: np.ndarray, 
                          sigma2_r: np.ndarray,
                          sigma2_h: np.ndarray) -> Tuple[float, float]:
        """Simulate the conditional variances for time t.
        
        Args:
            t: Time index
            realized: Past realized measures up to t-1
            innovations: Past return innovations up to t-1
            sigma2_r: Past realized conditional variances up to t-1
            sigma2_h: Past return conditional variances up to t-1
        
        Returns:
            Tuple[float, float]: Conditional variances for realized and return processes at time t
        
        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self._fitted or self._parameters is None:
            raise_not_fitted_error(
                "Model must be fitted before simulation.",
                model_type=self.name,
                operation="simulate_variances"
            )
        
        # Extract parameters
        omega_r = self._parameters.omega_r
        alpha_r = self._parameters.alpha_r
        beta_r = self._parameters.beta_r
        
        omega_h = self._parameters.omega_h
        alpha_h = self._parameters.alpha_h
        beta_h = self._parameters.beta_h
        
        # Compute variance for time t
        # Realized equation
        variance_r = omega_r
        if t - 1 >= 0:  # Ensure we don't go out of bounds
            variance_r += alpha_r * realized[t-1]
            variance_r += beta_r * sigma2_r[t-1]
        
        # Return equation
        variance_h = omega_h
        if t - 1 >= 0:  # Ensure we don't go out of bounds
            variance_h += alpha_h * realized[t-1]
            variance_h += beta_h * sigma2_h[t-1]
        
        return variance_r, variance_h
    
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
            NotImplementedError: HEAVY model forecasting requires both realized and return processes
        """
        raise NotImplementedError(
            "HEAVY model forecasting requires both realized and return processes. "
            "Use forecast_variances() instead."
        )
    
    def forecast_variances(self, 
                          steps: int, 
                          last_realized: Optional[float] = None,
                          last_variance_r: Optional[float] = None,
                          last_variance_h: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate analytic forecasts for both realized and return variances.
        
        Args:
            steps: Number of steps to forecast
            last_realized: Last observed realized measure
            last_variance_r: Last observed realized conditional variance
            last_variance_h: Last observed return conditional variance
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Forecasted realized and return conditional variances
        
        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self._fitted or self._parameters is None:
            raise_not_fitted_error(
                "Model must be fitted before forecasting.",
                model_type=self.name,
                operation="forecast_variances"
            )
        
        # Extract parameters
        omega_r = self._parameters.omega_r
        alpha_r = self._parameters.alpha_r
        beta_r = self._parameters.beta_r
        
        omega_h = self._parameters.omega_h
        alpha_h = self._parameters.alpha_h
        beta_h = self._parameters.beta_h
        
        # Use last observed values if not provided
        if last_realized is None:
            if self._realized_data is not None:
                last_realized = self._realized_data[-1]
            else:
                # Fallback to unconditional variance
                unconditional_r, _ = self._compute_unconditional_variance()
                last_realized = unconditional_r
        
        if last_variance_r is None:
            if self._conditional_variances_r is not None:
                last_variance_r = self._conditional_variances_r[-1]
            else:
                # Fallback to unconditional variance
                unconditional_r, _ = self._compute_unconditional_variance()
                last_variance_r = unconditional_r
        
        if last_variance_h is None:
            if self._conditional_variances_h is not None:
                last_variance_h = self._conditional_variances_h[-1]
            else:
                # Fallback to unconditional variance
                _, unconditional_h = self._compute_unconditional_variance()
                last_variance_h = unconditional_h
        
        # Use optimized Numba implementation
        return _heavy_forecast(
            omega_r, alpha_r, beta_r,
            omega_h, alpha_h, beta_h,
            last_realized, last_variance_r, last_variance_h,
            steps
        )
    
    def fit(self, 
            returns: np.ndarray,
            realized: np.ndarray,
            starting_values: Optional[Union[np.ndarray, HEAVYParams]] = None,
            distribution: Literal["normal", "t", "skewed-t", "ged"] = "normal",
            variance_targeting: bool = False,
            backcast_r: Optional[float] = None,
            backcast_h: Optional[float] = None,
            method: str = "SLSQP",
            options: Optional[Dict[str, Any]] = None,
            **kwargs: Any) -> 'UnivariateVolatilityResult':
        """Fit the HEAVY model to the provided data.
        
        Args:
            returns: Return data
            realized: Realized volatility measures
            starting_values: Initial parameter values for optimization
            distribution: Error distribution assumption
            variance_targeting: Whether to use variance targeting
            backcast_r: Value to use for initializing the realized variance process
            backcast_h: Value to use for initializing the return variance process
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
        # Store realized data for later use
        self._realized_data = realized
        self._return_data = returns
        
        # Call the parent class implementation with returns data
        # The compute_variance method will use both returns and realized data
        return super().fit(
            returns, 
            starting_values, 
            distribution, 
            variance_targeting, 
            backcast_h,  # Use backcast_h as the backcast parameter for the parent method
            method, 
            options, 
            **kwargs
        )
    
    async def fit_async(self, 
                       returns: np.ndarray,
                       realized: np.ndarray,
                       starting_values: Optional[Union[np.ndarray, HEAVYParams]] = None,
                       distribution: Literal["normal", "t", "skewed-t", "ged"] = "normal",
                       variance_targeting: bool = False,
                       backcast_r: Optional[float] = None,
                       backcast_h: Optional[float] = None,
                       method: str = "SLSQP",
                       options: Optional[Dict[str, Any]] = None,
                       progress_callback: Optional[Callable[[float, str], None]] = None,
                       **kwargs: Any) -> 'UnivariateVolatilityResult':
        """Asynchronously fit the HEAVY model to the provided data.
        
        This method provides an asynchronous interface to the fit method,
        allowing for non-blocking model estimation in UI contexts.
        
        Args:
            returns: Return data
            realized: Realized volatility measures
            starting_values: Initial parameter values for optimization
            distribution: Error distribution assumption
            variance_targeting: Whether to use variance targeting
            backcast_r: Value to use for initializing the realized variance process
            backcast_h: Value to use for initializing the return variance process
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
        # Report initial progress
        if progress_callback:
            await progress_callback(0.0, "Starting HEAVY model estimation...")
        
        # Store realized data for later use
        self._realized_data = realized
        self._return_data = returns
        
        # Run the fit method in a separate thread to avoid blocking
        loop = asyncio.get_event_loop()
        
        # Define a wrapper function that reports progress
        def fit_with_progress():
            result = self.fit(
                returns, 
                realized,
                starting_values, 
                distribution, 
                variance_targeting, 
                backcast_r,
                backcast_h,
                method, 
                options, 
                **kwargs
            )
            return result
        
        # Run the fit method in a separate thread
        try:
            result = await loop.run_in_executor(None, fit_with_progress)
            
            # Report completion
            if progress_callback:
                await progress_callback(1.0, "HEAVY model estimation complete.")
            
            return result
        except Exception as e:
            # Report error
            if progress_callback:
                await progress_callback(1.0, f"Error: {str(e)}")
            raise
    
    def simulate(self, 
                n_periods: int, 
                burn: int = 500, 
                initial_values: Optional[Dict[str, Any]] = None,
                random_state: Optional[Union[int, np.random.Generator]] = None,
                distribution: Literal["normal", "t", "skewed-t", "ged"] = "normal",
                distribution_params: Optional[Dict[str, Any]] = None,
                **kwargs: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate data from the HEAVY model.
        
        Args:
            n_periods: Number of periods to simulate
            burn: Number of initial observations to discard
            initial_values: Initial values for the simulation
                (dict with 'realized', 'returns', 'sigma2_r', 'sigma2_h' keys)
            random_state: Random number generator or seed
            distribution: Error distribution to use for simulation
            distribution_params: Additional parameters for the error distribution
            **kwargs: Additional keyword arguments for simulation
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Simulated returns, realized measures, and return conditional variances
        
        Raises:
            RuntimeError: If the model has not been fitted
            ValueError: If the simulation parameters are invalid
            SimulationError: If there are issues during simulation
        """
        if not self._fitted or self._parameters is None:
            raise_not_fitted_error(
                "Model must be fitted before simulation.",
                model_type=self.name,
                operation="simulate"
            )
        
        # Set up random number generator
        if isinstance(random_state, np.random.Generator):
            rng = random_state
        else:
            rng = np.random.default_rng(random_state)
        
        # Normalize distribution name
        distribution = distribution.lower()
        if distribution not in ["normal", "t", "skewed-t", "ged"]:
            raise ValueError(
                f"Unknown distribution: {distribution}. "
                f"Supported distributions are 'normal', 't', 'skewed-t', and 'ged'."
            )
        
        # Set up distribution parameters
        if distribution_params is None:
            distribution_params = {}
        
        # Total number of periods to simulate (including burn-in)
        total_periods = n_periods + burn
        
        try:
            # Initialize arrays for simulation
            returns = np.zeros(total_periods)
            realized = np.zeros(total_periods)
            sigma2_r = np.zeros(total_periods)
            sigma2_h = np.zeros(total_periods)
            
            # Set initial values
            if initial_values is None:
                initial_values = {}
            
            # Get unconditional variances
            unconditional_r, unconditional_h = self._compute_unconditional_variance()
            
            # Set initial variances
            sigma2_r[0] = initial_values.get('sigma2_r', unconditional_r)
            sigma2_h[0] = initial_values.get('sigma2_h', unconditional_h)
            
            # Set initial returns and realized measures
            returns[0] = initial_values.get('returns', 0.0)
            realized[0] = initial_values.get('realized', sigma2_r[0])
            
            # Generate random innovations based on distribution
            if distribution == "normal":
                std_innovations = rng.standard_normal(total_periods)
            elif distribution == "t":
                df = distribution_params.get("df", 5.0)
                std_innovations = rng.standard_t(df, total_periods)
                # Scale to have unit variance
                std_innovations = std_innovations * np.sqrt((df - 2) / df)
            elif distribution == "skewed-t":
                # TODO: Implement skewed t-distribution random generation
                raise NotImplementedError("skewed t-distribution not yet implemented")
            elif distribution == "ged":
                # TODO: Implement GED random generation
                raise NotImplementedError("GED not yet implemented")
            else:
                raise ValueError(f"Unknown distribution: {distribution}")
            
            # Extract parameters
            omega_r = self._parameters.omega_r
            alpha_r = self._parameters.omega_r  # NOTE: original code uses parameters.omega_r, parameters.alpha_r, etc.
            alpha_r = self._parameters.alpha_r
            beta_r = self._parameters.beta_r
            
            omega_h = self._parameters.omega_h
            alpha_h = self._parameters.alpha_h
            beta_h = self._parameters.beta_h
            
            # Simulate the process
            for t in range(1, total_periods):
                # Compute conditional variances for time t
                sigma2_r[t] = omega_r + alpha_r * realized[t-1] + beta_r * sigma2_r[t-1]
                sigma2_h[t] = omega_h + alpha_h * realized[t-1] + beta_h * sigma2_h[t-1]
                
                # Generate return for time t
                returns[t] = np.sqrt(sigma2_h[t]) * std_innovations[t]
                
                # Generate realized measure for time t
                # In a simple model, realized measure is return variance plus noise
                realized[t] = returns[t]**2 + rng.normal(0, 0.1 * sigma2_r[t])
                realized[t] = max(realized[t], 1e-6)  # Ensure positivity
            
            # Discard burn-in period
            return returns[burn:], realized[burn:], sigma2_h[burn:]
            
        except Exception as e:
            # Wrap exceptions in SimulationError
            raise SimulationError(
                f"Error during simulation: {str(e)}",
                model_type=self.name,
                n_periods=n_periods,
                issue=str(e)
            ) from e
    
    async def simulate_async(self, 
                           n_periods: int, 
                           burn: int = 500, 
                           initial_values: Optional[Dict[str, Any]] = None,
                           random_state: Optional[Union[int, np.random.Generator]] = None,
                           distribution: Literal["normal", "t", "skewed-t", "ged"] = "normal",
                           distribution_params: Optional[Dict[str, Any]] = None,
                           progress_callback: Optional[Callable[[float, str], None]] = None,
                           **kwargs: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Asynchronously simulate data from the HEAVY model.
        
        This method provides an asynchronous interface to the simulate method,
        allowing for non-blocking simulation in UI contexts.
        
        Args:
            n_periods: Number of periods to simulate
            burn: Number of initial observations to discard
            initial_values: Initial values for the simulation
                (dict with 'realized', 'returns', 'sigma2_r', 'sigma2_h' keys)
            random_state: Random number generator or seed
            distribution: Error distribution to use for simulation
            distribution_params: Additional parameters for the error distribution
            progress_callback: Optional callback function for progress updates
            **kwargs: Additional keyword arguments for simulation
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Simulated returns, realized measures, and return conditional variances
        
        Raises:
            RuntimeError: If the model has not been fitted
            ValueError: If the simulation parameters are invalid
            SimulationError: If there are issues during simulation
        """
        # Report initial progress
        if progress_callback:
            await progress_callback(0.0, "Starting HEAVY model simulation...")
        
        # Run the simulate method in a separate thread to avoid blocking
        loop = asyncio.get_event_loop()
        
        # Define a wrapper function that reports progress
        def simulate_with_progress():
            result = self.simulate(
                n_periods, 
                burn, 
                initial_values, 
                random_state, 
                distribution, 
                distribution_params, 
                **kwargs
            )
            return result
        
        # Run the simulate method in a separate thread
        try:
            result = await loop.run_in_executor(None, simulate_with_progress)
            
            # Report completion
            if progress_callback:
                await progress_callback(1.0, "HEAVY model simulation complete.")
            
            return result
        except Exception as e:
            # Report error
            if progress_callback:
                await progress_callback(1.0, f"Error: {str(e)}")
            raise
    
    def forecast(self, 
                steps: int, 
                data: Optional[Dict[str, np.ndarray]] = None,
                method: Literal["analytic", "simulation"] = "analytic",
                n_simulations: int = 1000,
                random_state: Optional[Union[int, np.random.Generator]] = None,
                distribution: Literal["normal", "t", "skewed-t", "ged"] = "normal",
                distribution_params: Optional[Dict[str, Any]] = None,
                **kwargs: Any) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Generate volatility forecasts from the fitted HEAVY model.
        
        Args:
            steps: Number of steps to forecast
            data: Dictionary containing 'returns' and 'realized' data to condition the forecast on
                (if different from fitting data)
            method: Forecasting method ('analytic' or 'simulation')
            n_simulations: Number of simulations for simulation-based forecasting
            random_state: Random number generator or seed for simulation
            distribution: Error distribution to use for simulation
            distribution_params: Additional parameters for the error distribution
            **kwargs: Additional keyword arguments for forecasting
        
        Returns:
            Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]: Dictionary with 'returns' and 'realized' keys,
                each containing (point forecasts, lower bounds, upper bounds)
        
        Raises:
            RuntimeError: If the model has not been fitted
            ValueError: If the forecast parameters are invalid
        """
        if not self._fitted or self._parameters is None:
            raise_not_fitted_error(
                "Model must be fitted before forecasting.",
                model_type=self.name,
                operation="forecast"
            )
        
        # If data is provided, compute conditional variances for the data
        if data is not None:
            returns = data.get('returns')
            realized = data.get('realized')
            
            if returns is None or realized is None:
                raise ValueError(
                    "Both 'returns' and 'realized' must be provided in the data dictionary."
                )
            
            self.validate_data(returns)
            
            if len(returns) != len(realized):
                raise ValueError(
                    f"Returns length ({len(returns)}) must match realized length ({len(realized)})"
                )
            
            # Compute conditional variances
            sigma2_r = np.zeros(len(returns))
            sigma2_h = np.zeros(len(returns))
            
            sigma2_r, sigma2_h = self.compute_variances(
                self._parameters, returns, realized, sigma2_r, sigma2_h
            )
            
            last_realized = realized[-1]
            last_variance_r = sigma2_r[-1]
            last_variance_h = sigma2_h[-1]
        elif self._conditional_variances_r is not None and self._conditional_variances_h is not None:
            # Use the last conditional variances from the fitted model
            last_realized = self._realized_data[-1]
            last_variance_r = self._conditional_variances_r[-1]
            last_variance_h = self._conditional_variances_h[-1]
        else:
            # Use unconditional variances as fallback
            unconditional_r, unconditional_h = self._compute_unconditional_variance()
            last_realized = self._realized_data[-1] if self._realized_data is not None else unconditional_r
            last_variance_r = unconditional_r
            last_variance_h = unconditional_h
        
        if method == "analytic":
            # Analytic forecasting
            forecasts_r, forecasts_h = self.forecast_variances(
                steps, last_realized, last_variance_r, last_variance_h
            )
            
            # Compute forecast intervals (assuming normal distribution)
            # For volatility, we use log-normal confidence intervals
            alpha = 0.05  # 95% confidence interval
            z_value = stats.norm.ppf(1 - alpha / 2)
            
            # Approximate standard errors for volatility forecasts
            # This is a simplification; proper intervals require model-specific derivation
            std_errors_r = np.sqrt(forecasts_r) * 0.5  # Approximate standard error
            std_errors_h = np.sqrt(forecasts_h) * 0.5  # Approximate standard error
            
            lower_bounds_r = forecasts_r * np.exp(-z_value * std_errors_r / forecasts_r)
            upper_bounds_r = forecasts_r * np.exp(z_value * std_errors_r / forecasts_r)
            
            lower_bounds_h = forecasts_h * np.exp(-z_value * std_errors_h / forecasts_h)
            upper_bounds_h = forecasts_h * np.exp(z_value * std_errors_h / forecasts_h)
            
            return {
                'realized': (forecasts_r, lower_bounds_r, upper_bounds_r),
                'returns': (forecasts_h, lower_bounds_h, upper_bounds_h)
            }
            
        elif method == "simulation":
            # Simulation-based forecasting
            # Set up random number generator
            if isinstance(random_state, np.random.Generator):
                rng = random_state
            else:
                rng = np.random.default_rng(random_state)
            
            # Initialize arrays for simulation
            sim_forecasts_r = np.zeros((n_simulations, steps))
            sim_forecasts_h = np.zeros((n_simulations, steps))
            
            # Initial values for simulation
            initial_values = {
                'realized': last_realized,
                'returns': 0.0,  # Not used for forecasting
                'sigma2_r': last_variance_r,
                'sigma2_h': last_variance_h
            }
            
            # Run simulations
            for i in range(n_simulations):
                # Simulate future paths
                _, sim_realized, sim_sigma2_h = self.simulate(
                    steps, 
                    burn=0, 
                    initial_values=initial_values, 
                    random_state=rng, 
                    distribution=distribution, 
                    distribution_params=distribution_params
                )
                
                # Compute conditional variances for the simulated path
                sim_sigma2_r = np.zeros(steps)
                sim_sigma2_r[0] = last_variance_r
                
                # Compute realized conditional variances
                for t in range(1, steps):
                    sim_sigma2_r[t] = (self._parameters.omega_r + 
                                      self._parameters.alpha_r * sim_realized[t-1] + 
                                      self._parameters.beta_r * sim_sigma2_r[t-1])
                
                # Store the forecasted variances
                sim_forecasts_r[i, :] = sim_sigma2_r
                sim_forecasts_h[i, :] = sim_sigma2_h
            
            # Compute point forecasts and intervals
            forecasts_r = np.mean(sim_forecasts_r, axis=0)
            lower_bounds_r = np.percentile(sim_forecasts_r, 2.5, axis=0)
            upper_bounds_r = np.percentile(sim_forecasts_r, 97.5, axis=0)
            
            forecasts_h = np.mean(sim_forecasts_h, axis=0)
            lower_bounds_h = np.percentile(sim_forecasts_h, 2.5, axis=0)
            upper_bounds_h = np.percentile(sim_forecasts_h, 97.5, axis=0)
            
            return {
                'realized': (forecasts_r, lower_bounds_r, upper_bounds_r),
                'returns': (forecasts_h, lower_bounds_h, upper_bounds_h)
            }
            
        else:
            raise ValueError(
                f"Unknown forecasting method: {method}. "
                f"Supported methods are 'analytic' and 'simulation'."
            )
    
    async def forecast_async(self, 
                           steps: int, 
                           data: Optional[Dict[str, np.ndarray]] = None,
                           method: Literal["analytic", "simulation"] = "analytic",
                           n_simulations: int = 1000,
                           random_state: Optional[Union[int, np.random.Generator]] = None,
                           distribution: Literal["normal", "t", "skewed-t", "ged"] = "normal",
                           distribution_params: Optional[Dict[str, Any]] = None,
                           progress_callback: Optional[Callable[[float, str], None]] = None,
                           **kwargs: Any) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Asynchronously generate volatility forecasts from the fitted HEAVY model.
        
        This method provides an asynchronous interface to the forecast method,
        allowing for non-blocking forecasting in UI contexts.
        
        Args:
            steps: Number of steps to forecast
            data: Dictionary containing 'returns' and 'realized' data to condition the forecast on
                (if different from fitting data)
            method: Forecasting method ('analytic' or 'simulation')
            n_simulations: Number of simulations for simulation-based forecasting
            random_state: Random number generator or seed for simulation
            distribution: Error distribution to use for simulation
            distribution_params: Additional parameters for the error distribution
            progress_callback: Optional callback function for progress updates
            **kwargs: Additional keyword arguments for forecasting
        
        Returns:
            Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]: Dictionary with 'returns' and 'realized' keys,
                each containing (point forecasts, lower bounds, upper bounds)
        
        Raises:
            RuntimeError: If the model has not been fitted
            ValueError: If the forecast parameters are invalid
        """
        # Report initial progress
        if progress_callback:
            await progress_callback(0.0, "Starting HEAVY model forecasting...")
        
        # Run the forecast method in a separate thread to avoid blocking
        loop = asyncio.get_event_loop()
        
        # Define a wrapper function that reports progress
        def forecast_with_progress():
            result = self.forecast(
                steps, 
                data, 
                method, 
                n_simulations, 
                random_state, 