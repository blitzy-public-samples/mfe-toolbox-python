# mfe/models/univariate/agarch.py

"""
Asymmetric GARCH (AGARCH) model implementation for capturing leverage effects in volatility.

This module provides a comprehensive implementation of the AGARCH model for modeling
asymmetric volatility responses to positive and negative shocks in financial time series.
Two variants are supported:

1. Standard AGARCH: σ²_t = ω + α(ε_{t-1} - γ)² + βσ²_{t-1}
2. NAGARCH (Nonlinear AGARCH): σ²_t = ω + α(ε_{t-1}/σ_{t-1} - γ)² σ²_{t-1} + βσ²_{t-1}

The implementation includes parameter estimation via maximum likelihood, simulation,
forecasting, and diagnostic tools. Performance-critical operations are accelerated
using Numba's just-in-time compilation.

References:
    Engle, R.F. and Ng, V.K. (1993). Measuring and Testing the Impact of News on Volatility.
    Journal of Finance, 48(5), 1749-1778.

    Higgins, M.L. and Bera, A.K. (1992). A Class of Nonlinear ARCH Models.
    International Economic Review, 33(1), 137-158.
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
    validate_positive, validate_non_negative, validate_range,
    transform_positive, transform_correlation, inverse_transform_positive, inverse_transform_correlation
)
from mfe.core.exceptions import (
    MFEError, ConvergenceError, NumericError, EstimationError, SimulationError,
    NotFittedError, raise_convergence_error, raise_numeric_error, raise_not_fitted_error,
    warn_convergence, warn_numeric, warn_model
)
from mfe.models.univariate.base import VolatilityModel


# Numba-accelerated core functions for AGARCH model
@jit(nopython=True, cache=True)
def _agarch_recursion(data: np.ndarray,
                      omega: float,
                      alpha: float,
                      gamma: float,
                      beta: float,
                      sigma2: np.ndarray,
                      backcast: float) -> np.ndarray:
    """Compute AGARCH conditional variances using Numba acceleration.

    Args:
        data: Input data (typically residuals)
        omega: Constant term in variance equation
        alpha: ARCH parameter
        gamma: Asymmetry parameter
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
        # Standard AGARCH: σ²_t = ω + α(ε_{t-1} - γ)² + βσ²_{t-1}
        sigma2[t] = omega + alpha * (data[t-1] - gamma)**2 + beta * sigma2[t-1]

    return sigma2


@jit(nopython=True, cache=True)
def _nagarch_recursion(data: np.ndarray,
                       omega: float,
                       alpha: float,
                       gamma: float,
                       beta: float,
                       sigma2: np.ndarray,
                       backcast: float) -> np.ndarray:
    """Compute NAGARCH (Nonlinear AGARCH) conditional variances using Numba acceleration.

    Args:
        data: Input data (typically residuals)
        omega: Constant term in variance equation
        alpha: ARCH parameter
        gamma: Asymmetry parameter
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
        # NAGARCH: σ²_t = ω + α(ε_{t-1}/σ_{t-1} - γ)² σ²_{t-1} + βσ²_{t-1}
        std_resid = data[t-1] / np.sqrt(sigma2[t-1])
        sigma2[t] = omega + alpha * ((std_resid - gamma)**2) * sigma2[t-1] + beta * sigma2[t-1]

    return sigma2


@jit(nopython=True, cache=True)
def _agarch_forecast(omega: float,
                     alpha: float,
                     gamma: float,
                     beta: float,
                     last_variance: float,
                     last_residual: float,
                     steps: int) -> np.ndarray:
    """Generate analytic forecasts for AGARCH model using Numba acceleration.

    Args:
        omega: Constant term in variance equation
        alpha: ARCH parameter
        gamma: Asymmetry parameter
        beta: GARCH parameter
        last_variance: Last observed conditional variance
        last_residual: Last observed residual
        steps: Number of steps to forecast

    Returns:
        np.ndarray: Forecasted conditional variances
    """
    forecasts = np.zeros(steps)

    # First step forecast uses the last residual
    forecasts[0] = omega + alpha * (last_residual - gamma)**2 + beta * last_variance

    # For subsequent steps, we use E[(ε_t - γ)²] = E[ε_t²] - 2γE[ε_t] + γ²
    # For zero-mean residuals, E[ε_t] = 0 and E[ε_t²] = σ²_t
    # So E[(ε_t - γ)²] = σ²_t + γ²
    expected_term = gamma**2  # E[ε_t²] - 2γE[ε_t] + γ² = 0 - 0 + γ²

    # Multi-step forecasts
    for h in range(1, steps):
        forecasts[h] = omega + alpha * (forecasts[h-1] + expected_term) + beta * forecasts[h-1]

    return forecasts


@jit(nopython=True, cache=True)
def _nagarch_forecast(omega: float,
                      alpha: float,
                      gamma: float,
                      beta: float,
                      last_variance: float,
                      steps: int) -> np.ndarray:
    """Generate analytic forecasts for NAGARCH model using Numba acceleration.

    Args:
        omega: Constant term in variance equation
        alpha: ARCH parameter
        gamma: Asymmetry parameter
        beta: GARCH parameter
        last_variance: Last observed conditional variance
        steps: Number of steps to forecast

    Returns:
        np.ndarray: Forecasted conditional variances
    """
    forecasts = np.zeros(steps)

    # For NAGARCH, E[(ε_t/σ_t - γ)²] = 1 - 2γE[ε_t/σ_t] + γ²
    # For standardized residuals, E[ε_t/σ_t] = 0 and E[(ε_t/σ_t)²] = 1
    # So E[(ε_t/σ_t - γ)²] = 1 + γ²
    expected_term = 1 + gamma**2

    # First step forecast
    forecasts[0] = omega + (alpha * expected_term + beta) * last_variance

    # Multi-step forecasts
    for h in range(1, steps):
        forecasts[h] = omega + (alpha * expected_term + beta) * forecasts[h-1]

    return forecasts


@dataclass
class AGARCHParams(UnivariateVolatilityParameters):
    """Parameters for AGARCH model.

    This class defines the parameters for the AGARCH model, including
    validation to ensure parameter constraints are satisfied.

    Attributes:
        omega: Constant term in variance equation (must be positive)
        alpha: ARCH parameter (must be non-negative)
        gamma: Asymmetry parameter
        beta: GARCH parameter (must be non-negative)
    """

    omega: float
    alpha: float
    gamma: float
    beta: float

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate AGARCH parameter constraints.

        Raises:
            ParameterError: If parameter constraints are violated
        """
        # Validate individual parameters
        validate_positive(self.omega, "omega")
        validate_non_negative(self.alpha, "alpha")
        validate_non_negative(self.beta, "beta")

        # Validate stationarity constraint
        # For AGARCH, the constraint is alpha + beta < 1
        if self.alpha + self.beta >= 1:
            raise ParameterError(
                f"AGARCH stationarity constraint violated: alpha + beta = {self.alpha + self.beta} >= 1"
            )

    def to_array(self) -> np.ndarray:
        """Convert parameters to a NumPy array.

        Returns:
            np.ndarray: Array representation of parameters
        """
        return np.array([self.omega, self.alpha, self.gamma, self.beta])

    @classmethod
    def from_array(cls, array: np.ndarray, **kwargs: Any) -> 'AGARCHParams':
        """Create parameters from a NumPy array.

        Args:
            array: Array representation of parameters
            **kwargs: Additional keyword arguments for parameter creation

        Returns:
            AGARCHParams: Parameter object

        Raises:
            ValueError: If the array length is not 4
        """
        if len(array) != 4:
            raise ValueError(f"Array length must be 4, got {len(array)}")

        return cls(omega=array[0], alpha=array[1], gamma=array[2], beta=array[3])

    def transform(self) -> np.ndarray:
        """Transform parameters to unconstrained space for optimization.

        Returns:
            np.ndarray: Transformed parameters in unconstrained space
        """
        # Transform omega to unconstrained space (log)
        transformed_omega = transform_positive(self.omega)

        # Transform alpha and beta to unconstrained space
        # We use a special transformation to ensure alpha + beta < 1
        if self.alpha + self.beta >= 1:
            # If constraint is violated, adjust parameters slightly
            sum_ab = self.alpha + self.beta
            self.alpha = self.alpha / (sum_ab + 0.01) * 0.99
            self.beta = self.beta / (sum_ab + 0.01) * 0.99

        # Use logit-like transformation for alpha and beta
        from mfe.core.parameters import transform_probability

        gamma = self.alpha + self.beta
        delta = self.alpha / gamma if gamma > 0 else 0.5

        transformed_gamma = transform_probability(gamma)
        transformed_delta = transform_probability(delta)

        # No constraints on asymmetry parameter, so it remains unchanged
        transformed_asymmetry = self.gamma

        return np.array([transformed_omega, transformed_gamma, transformed_delta, transformed_asymmetry])

    @classmethod
    def inverse_transform(cls, array: np.ndarray, **kwargs: Any) -> 'AGARCHParams':
        """Transform parameters from unconstrained space back to constrained space.

        Args:
            array: Parameters in unconstrained space
            **kwargs: Additional keyword arguments for parameter creation

        Returns:
            AGARCHParams: Parameter object with constrained parameters

        Raises:
            ValueError: If the array length is not 4
        """
        if len(array) != 4:
            raise ValueError(f"Array length must be 4, got {len(array)}")

        # Extract transformed parameters
        transformed_omega, transformed_gamma, transformed_delta, transformed_asymmetry = array

        # Inverse transform omega
        omega = inverse_transform_positive(transformed_omega)

        # Inverse transform gamma (alpha + beta) and delta (alpha / (alpha + beta))
        from mfe.core.parameters import inverse_transform_probability

        gamma = inverse_transform_probability(transformed_gamma)
        delta = inverse_transform_probability(transformed_delta)

        # Compute alpha and beta
        alpha = gamma * delta
        beta = gamma * (1 - delta)

        # Asymmetry parameter has no constraints
        asymmetry = transformed_asymmetry

        return cls(omega=omega, alpha=alpha, gamma=asymmetry, beta=beta)


class AGARCHModel(VolatilityModel):
    """Asymmetric GARCH (AGARCH) model for volatility modeling with leverage effects.

    This class implements the AGARCH model for modeling asymmetric volatility responses
    to positive and negative shocks in financial time series. Two variants are supported:

    1. Standard AGARCH: σ²_t = ω + α(ε_{t-1} - γ)² + βσ²_{t-1}
    2. NAGARCH (Nonlinear AGARCH): σ²_t = ω + α(ε_{t-1}/σ_{t-1} - γ)² σ²_{t-1} + βσ²_{t-1}

    The asymmetry parameter γ allows the model to capture leverage effects, where
    negative returns typically increase volatility more than positive returns of
    the same magnitude.

    Attributes:
        model_type: Type of AGARCH model ('standard' or 'nonlinear')
        name: Model name
        parameters: Model parameters if set
        _conditional_variances: Conditional variances if the model has been fitted
        _data: Input data used for fitting if the model has been fitted
    """

    def __init__(self,
                 model_type: Literal['standard', 'nonlinear'] = 'standard',
                 parameters: Optional[AGARCHParams] = None,
                 name: Optional[str] = None) -> None:
        """Initialize the AGARCH model.

        Args:
            model_type: Type of AGARCH model ('standard' or 'nonlinear')
            parameters: Pre-specified model parameters if available
            name: A descriptive name for the model

        Raises:
            ValueError: If model_type is not 'standard' or 'nonlinear'
        """
        if model_type not in ['standard', 'nonlinear']:
            raise ValueError(
                f"model_type must be 'standard' or 'nonlinear', got {model_type}"
            )

        self.model_type = model_type

        # Set model name based on type
        if name is None:
            name = "AGARCH" if model_type == 'standard' else "NAGARCH"

        super().__init__(parameters=parameters, name=name)

        # Initialize additional attributes
        self._conditional_variances = None
        self._data = None

    def parameter_class(self) -> Type[AGARCHParams]:
        """Get the parameter class for this model.

        Returns:
            Type[AGARCHParams]: The parameter class for this model
        """
        return AGARCHParams

    def compute_variance(self,
                         parameters: AGARCHParams,
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
            # Extract parameters
            omega = parameters.omega
            alpha = parameters.alpha
            gamma = parameters.gamma
            beta = parameters.beta

            # Use optimized Numba implementation based on model type
            if self.model_type == 'standard':
                sigma2 = _agarch_recursion(data, omega, alpha, gamma, beta, sigma2, backcast)
            else:  # nonlinear
                sigma2 = _nagarch_recursion(data, omega, alpha, gamma, beta, sigma2, backcast)

            # Check for numerical issues
            if np.any(~np.isfinite(sigma2)) or np.any(sigma2 <= 0):
                raise_numeric_error(
                    "Numerical issues detected in variance computation",
                    operation=f"{self.name} variance computation",
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
                operation=f"{self.name} variance computation",
                error_type="Computation error",
                details=str(e)
            )

    def _generate_starting_values(self,
                                  data: np.ndarray,
                                  variance_targeting: bool = False,
                                  backcast: Optional[float] = None) -> AGARCHParams:
        """Generate starting values for parameter estimation.

        Args:
            data: Input data (typically residuals)
            variance_targeting: Whether to use variance targeting
            backcast: Value to use for initializing the variance process

        Returns:
            AGARCHParams: Starting parameter values
        """
        # Compute sample variance
        sample_variance = np.var(data)

        if backcast is None:
            backcast = sample_variance

        # Generate reasonable starting values
        if variance_targeting:
            # With variance targeting, omega is determined by the other parameters
            alpha = 0.05
            gamma = 0.0  # Start with symmetric model
            beta = 0.90
            omega = sample_variance * (1 - alpha - beta)
        else:
            # Without variance targeting, use reasonable defaults
            omega = 0.05 * sample_variance
            alpha = 0.05
            gamma = 0.0  # Start with symmetric model
            beta = 0.90

        # For NAGARCH, adjust gamma to a reasonable value
        if self.model_type == 'nonlinear':
            # Typical value for NAGARCH gamma is around 0.1-0.3
            gamma = 0.2

        return AGARCHParams(omega=omega, alpha=alpha, gamma=gamma, beta=beta)

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

        # Extract parameters
        omega = self._parameters.omega
        alpha = self._parameters.alpha
        gamma = self._parameters.gamma
        beta = self._parameters.beta

        # For standard AGARCH, unconditional variance is:
        # omega / (1 - alpha - beta) + alpha * gamma^2 / (1 - alpha - beta)
        if self.model_type == 'standard':
            persistence = alpha + beta
            if persistence >= 1:
                # If the process is not stationary, return the last observed variance
                if self._conditional_variances is not None:
                    return self._conditional_variances[-1]
                else:
                    # Fallback to a reasonable value
                    return omega / 0.05

            return omega / (1 - persistence) + alpha * gamma**2 / (1 - persistence)

        # For NAGARCH, unconditional variance is:
        # omega / (1 - beta - alpha * (1 + gamma^2))
        else:
            persistence = beta + alpha * (1 + gamma**2)
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
        gamma = self._parameters.gamma
        beta = self._parameters.beta

        # Compute variance for time t based on model type
        if self.model_type == 'standard':
            # Standard AGARCH: σ²_t = ω + α(ε_{t-1} - γ)² + βσ²_{t-1}
            variance = omega + alpha * (innovations[t-1] - gamma)**2 + beta * sigma2[t-1]
        else:
            # NAGARCH: σ²_t = ω + α(ε_{t-1}/σ_{t-1} - γ)² σ²_{t-1} + βσ²_{t-1}
            std_resid = innovations[t-1] / np.sqrt(sigma2[t-1])
            variance = omega + alpha * ((std_resid - gamma)**2) * sigma2[t-1] + beta * sigma2[t-1]

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
        gamma = self._parameters.gamma
        beta = self._parameters.beta

        # Use optimized Numba implementation based on model type
        if self.model_type == 'standard':
            # For standard AGARCH, we need the last residual
            if self._data is None or self._conditional_variances is None:
                raise RuntimeError("Data and conditional variances not available for forecasting")

            last_residual = self._data[-1]
            return _agarch_forecast(omega, alpha, gamma, beta, last_variance, last_residual, steps)
        else:
            # For NAGARCH, we don't need the last residual
            return _nagarch_forecast(omega, alpha, gamma, beta, last_variance, steps)

    def fit(self,
            data: np.ndarray,
            starting_values: Optional[Union[np.ndarray, AGARCHParams]] = None,
            distribution: Literal["normal", "t", "skewed-t", "ged"] = "normal",
            variance_targeting: bool = False,
            backcast: Optional[float] = None,
            method: str = "SLSQP",
            options: Optional[Dict[str, Any]] = None,
            **kwargs: Any) -> 'UnivariateVolatilityResult':
        """Fit the AGARCH model to the provided data.

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
        # Store data for later use in forecasting
        self._data = np.asarray(data)

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
                        starting_values: Optional[Union[np.ndarray, AGARCHParams]] = None,
                        distribution: Literal["normal", "t", "skewed-t", "ged"] = "normal",
                        variance_targeting: bool = False,
                        backcast: Optional[float] = None,
                        method: str = "SLSQP",
                        options: Optional[Dict[str, Any]] = None,
                        progress_callback: Optional[Callable[[float, str], None]] = None,
                        **kwargs: Any) -> 'UnivariateVolatilityResult':
        """Asynchronously fit the AGARCH model to the provided data.

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
        # Store data for later use in forecasting
        self._data = np.asarray(data)

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
        """Simulate data from the AGARCH model.

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
        """Asynchronously simulate data from the AGARCH model.

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
        """Generate volatility forecasts from the fitted AGARCH model.

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
        # If new data is provided, update the stored data
        if data is not None:
            self._data = np.asarray(data)

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
        """Asynchronously generate volatility forecasts from the fitted AGARCH model.

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
        # If new data is provided, update the stored data
        if data is not None:
            self._data = np.asarray(data)

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
        """Validate input data for the AGARCH model.

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
        if len(data) < 2:  # Need at least 2 observations for AGARCH
            raise ValueError(
                f"data must have at least 2 observations, got {len(data)}"
            )

        # Check for NaN or infinite values
        if np.any(~np.isfinite(data)):
            raise ValueError("data contains NaN or infinite values")

    def plot_diagnostics(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """Plot diagnostic plots for the fitted AGARCH model.

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

    def plot_news_impact(self,
                         range_multiple: float = 3.0,
                         num_points: int = 100,
                         figsize: Tuple[int, int] = (10, 6)) -> None:
        """Plot the news impact curve for the fitted AGARCH model.

        The news impact curve shows how past shocks affect current volatility,
        highlighting the asymmetric response to positive and negative shocks.

        Args:
            range_multiple: Multiple of standard deviation to use for x-axis range
            num_points: Number of points to plot
            figsize: Figure size for the plot

        Raises:
            RuntimeError: If the model has not been fitted
            ImportError: If matplotlib is not installed
        """
        if not self._fitted or self._parameters is None:
            raise_not_fitted_error(
                "Model must be fitted before plotting news impact curve.",
                model_type=self.name,
                operation="plot_news_impact"
            )

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Matplotlib is required for plotting. "
                "Please install it with 'pip install matplotlib'."
            )

        # Extract parameters
        omega = self._parameters.omega
        alpha = self._parameters.alpha
        gamma = self._parameters.gamma
        beta = self._parameters.beta

        # Compute unconditional variance
        uncond_var = self._compute_unconditional_variance()

        # Create range of shocks
        if self._data is not None:
            std_dev = np.std(self._data)
        else:
            std_dev = np.sqrt(uncond_var)

        shock_range = np.linspace(-range_multiple * std_dev, range_multiple * std_dev, num_points)

        # Compute news impact for each shock
        impact = np.zeros_like(shock_range)

        if self.model_type == 'standard':
            # Standard AGARCH: News impact is α(ε - γ)² + β*σ²
            # We set σ² to unconditional variance for a baseline
            for i, shock in enumerate(shock_range):
                impact[i] = omega + alpha * (shock - gamma)**2 + beta * uncond_var
        else:
            # NAGARCH: News impact is ω + α(ε/σ - γ)²σ² + βσ²
            # We set σ to sqrt(unconditional variance) for a baseline
            sigma = np.sqrt(uncond_var)
            for i, shock in enumerate(shock_range):
                std_shock = shock / sigma
                impact[i] = omega + alpha * ((std_shock - gamma)**2) * uncond_var + beta * uncond_var

        # Create plot
        plt.figure(figsize=figsize)
        plt.plot(shock_range, impact)
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=uncond_var, color='g', linestyle='--', alpha=0.5)
        plt.title(f'News Impact Curve for {self.name}')
        plt.xlabel('Shock (ε)')
        plt.ylabel('Impact on Variance (σ²)')
        plt.grid(True, alpha=0.3)

        # Add annotations
        plt.annotate('Unconditional Variance',
                     xy=(shock_range[-1], uncond_var),
                     xytext=(shock_range[-1] * 0.7, uncond_var * 1.1),
                     arrowprops=dict(facecolor='green', shrink=0.05, alpha=0.5))

        # Show plot
        plt.tight_layout()
        plt.show()

    def __str__(self) -> str:
        """Get a string representation of the AGARCH model.

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

        # Compute persistence
        persistence = alpha + beta

        # Create string representation
        model_str = f"{self.name} model\n"
        model_str += f"Model type: {self.model_type}\n"
        model_str += f"omega: {omega:.6f}\n"
        model_str += f"alpha: {alpha:.6f}\n"
        model_str += f"gamma: {gamma:.6f}\n"
        model_str += f"beta: {beta:.6f}\n"
        model_str += f"persistence: {persistence:.6f}\n"

        # Add unconditional variance if model is stationary
        if persistence < 1:
            uncond_var = self._compute_unconditional_variance()
            model_str += f"unconditional variance: {uncond_var:.6f}\n"

        return model_str

    def __repr__(self) -> str:
        """Get a string representation of the AGARCH model.

        Returns:
            str: String representation of the model
        """
        return self.__str__()
