# mfe/models/univariate/aparch.py

"""
Asymmetric Power ARCH (APARCH) model implementation.

This module provides a comprehensive implementation of the APARCH(1,1) model for
modeling time-varying volatility in financial time series. The APARCH model extends
the GARCH model by introducing a power parameter (delta) and an asymmetry parameter
that allows different impacts of positive and negative shocks on volatility.

The implementation includes parameter estimation via maximum likelihood,
simulation, forecasting, and diagnostic tools. Performance-critical operations
are accelerated using Numba's just-in-time compilation.

References:
    Ding, Z., Granger, C. W., & Engle, R. F. (1993). A long memory property of
    stock market returns and a new model. Journal of Empirical Finance, 1(1), 83-106.
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
    APARCHParameters, UnivariateVolatilityParameters, ParameterError,
    validate_positive, validate_non_negative, validate_probability, validate_range
)
from mfe.core.exceptions import (
    MFEError, ConvergenceError, NumericError, EstimationError, SimulationError,
    NotFittedError, raise_convergence_error, raise_numeric_error, raise_not_fitted_error,
    warn_convergence, warn_numeric, warn_model
)
from mfe.models.univariate.base import VolatilityModel


# Numba-accelerated core functions for APARCH model
@jit(nopython=True, cache=True)
def _aparch_recursion(data: np.ndarray,
                      omega: float,
                      alpha: float,
                      gamma: float,
                      beta: float,
                      delta: float,
                      sigma_delta: np.ndarray,
                      backcast: float) -> np.ndarray:
    """Compute APARCH(1,1) conditional power-transformed variances using Numba acceleration.

    Args:
        data: Input data (typically residuals)
        omega: Constant term in power variance equation
        alpha: ARCH parameter
        gamma: Asymmetry parameter
        beta: GARCH parameter
        delta: Power parameter
        sigma_delta: Pre-allocated array for conditional power-transformed variances
        backcast: Value to use for initializing the variance process

    Returns:
        np.ndarray: Conditional power-transformed variances
    """
    T = len(data)

    # Initialize first power-transformed variance with backcast value
    sigma_delta[0] = backcast

    # Compute conditional power-transformed variances recursively
    for t in range(1, T):
        # Compute |r_{t-1}| - γ*r_{t-1} term
        abs_r = abs(data[t-1])
        asymm_term = abs_r - gamma * data[t-1]

        # Compute power-transformed variance
        sigma_delta[t] = omega + alpha * (abs_r - gamma * data[t-1])**delta + beta * sigma_delta[t-1]

    return sigma_delta


@jit(nopython=True, cache=True)
def _aparch_forecast(omega: float,
                     alpha: float,
                     gamma: float,
                     beta: float,
                     delta: float,
                     last_variance_delta: float,
                     steps: int) -> np.ndarray:
    """Generate analytic forecasts for APARCH(1,1) model using Numba acceleration.

    Args:
        omega: Constant term in power variance equation
        alpha: ARCH parameter
        gamma: Asymmetry parameter
        beta: GARCH parameter
        delta: Power parameter
        last_variance_delta: Last observed conditional power-transformed variance
        steps: Number of steps to forecast

    Returns:
        np.ndarray: Forecasted conditional power-transformed variances
    """
    forecasts = np.zeros(steps)

    # Compute E[|z| - γ*z]^δ term for standard normal z
    # This is a constant for a given gamma and delta
    # For standard normal, E[|z|] = sqrt(2/π)
    # We use numerical approximation for the expectation
    # This could be pre-computed for efficiency
    expected_asymm_term = 0.0
    n_points = 1000
    z_values = np.linspace(-4.0, 4.0, n_points)
    pdf_values = np.exp(-0.5 * z_values**2) / np.sqrt(2.0 * np.pi)
    for i in range(n_points):
        z = z_values[i]
        pdf = pdf_values[i]
        expected_asymm_term += (abs(z) - gamma * z)**delta * pdf
    expected_asymm_term *= 8.0 / n_points  # Scale for the range [-4, 4]

    # Compute persistence
    persistence = beta + alpha * expected_asymm_term

    # Compute unconditional power-transformed variance
    if persistence < 1:
        unconditional_variance_delta = omega / (1 - persistence)
    else:
        unconditional_variance_delta = last_variance_delta

    # First step forecast
    forecasts[0] = omega + alpha * expected_asymm_term * last_variance_delta + beta * last_variance_delta

    # Multi-step forecasts
    for h in range(1, steps):
        forecasts[h] = omega + persistence * forecasts[h-1]

        # For long horizons, approach the unconditional variance
        if persistence < 1 and h > 100:
            forecasts[h] = unconditional_variance_delta - (unconditional_variance_delta - forecasts[h]) * 0.1

    return forecasts


@dataclass
class APARCHParams(APARCHParameters):
    """Parameters for APARCH(1,1) model.

    Attributes:
        omega: Constant term in power variance equation (must be positive)
        alpha: ARCH parameter (must be non-negative)
        gamma: Asymmetry parameter (must be between -1 and 1)
        beta: GARCH parameter (must be non-negative)
        delta: Power parameter (must be positive)
    """

    omega: float
    alpha: float
    gamma: float
    beta: float
    delta: float

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate APARCH parameter constraints.

        Raises:
            ParameterError: If parameter constraints are violated
        """
        super().validate()

        # Validate individual parameters
        validate_positive(self.omega, "omega")
        validate_non_negative(self.alpha, "alpha")
        validate_range(self.gamma, "gamma", -1, 1)
        validate_non_negative(self.beta, "beta")
        validate_positive(self.delta, "delta")

        # Validate stationarity constraint
        if self.alpha + self.beta >= 1:
            raise ParameterError(
                f"APARCH stationarity constraint violated: alpha + beta = {self.alpha + self.beta} >= 1"
            )

    def to_array(self) -> np.ndarray:
        """Convert parameters to a NumPy array.

        Returns:
            np.ndarray: Array representation of parameters
        """
        return np.array([self.omega, self.alpha, self.gamma, self.beta, self.delta])

    @classmethod
    def from_array(cls, array: np.ndarray, **kwargs: Any) -> 'APARCHParams':
        """Create parameters from a NumPy array.

        Args:
            array: Array representation of parameters
            **kwargs: Additional keyword arguments for parameter creation

        Returns:
            APARCHParams: Parameter object

        Raises:
            ValueError: If the array length is not 5
        """
        if len(array) != 5:
            raise ValueError(f"Array length must be 5, got {len(array)}")

        return cls(
            omega=array[0],
            alpha=array[1],
            gamma=array[2],
            beta=array[3],
            delta=array[4]
        )

    def transform(self) -> np.ndarray:
        """Transform parameters to unconstrained space for optimization.

        Returns:
            np.ndarray: Transformed parameters in unconstrained space
        """
        from mfe.core.parameters import (
            transform_positive, transform_probability, transform_correlation
        )

        # Transform omega and delta to unconstrained space (log)
        transformed_omega = transform_positive(self.omega)
        transformed_delta = transform_positive(self.delta)

        # Transform gamma to unconstrained space (arctanh)
        transformed_gamma = transform_correlation(self.gamma)

        # Transform alpha and beta to unconstrained space
        # We use a special transformation to ensure alpha + beta < 1
        if self.alpha + self.beta >= 1:
            # If constraint is violated, adjust parameters slightly
            sum_ab = self.alpha + self.beta
            self.alpha = self.alpha / (sum_ab + 0.01)
            self.beta = self.beta / (sum_ab + 0.01)

        # Use logit-like transformation for alpha and beta
        lambda_param = self.alpha + self.beta
        delta_param = self.alpha / lambda_param if lambda_param > 0 else 0.5

        transformed_lambda = transform_probability(lambda_param)
        transformed_delta_param = transform_probability(delta_param)

        return np.array([
            transformed_omega,
            transformed_lambda,
            transformed_delta_param,
            transformed_gamma,
            transformed_delta
        ])

    @classmethod
    def inverse_transform(cls, array: np.ndarray, **kwargs: Any) -> 'APARCHParams':
        """Transform parameters from unconstrained space back to constrained space.

        Args:
            array: Parameters in unconstrained space
            **kwargs: Additional keyword arguments for parameter creation

        Returns:
            APARCHParams: Parameter object with constrained parameters

        Raises:
            ValueError: If the array length is not 5
        """
        from mfe.core.parameters import (
            inverse_transform_positive, inverse_transform_probability, inverse_transform_correlation
        )

        if len(array) != 5:
            raise ValueError(f"Array length must be 5, got {len(array)}")

        # Extract transformed parameters
        (transformed_omega, transformed_lambda, transformed_delta_param,
         transformed_gamma, transformed_delta) = array

        # Inverse transform omega and delta
        omega = inverse_transform_positive(transformed_omega)
        delta = inverse_transform_positive(transformed_delta)

        # Inverse transform gamma
        gamma = inverse_transform_correlation(transformed_gamma)

        # Inverse transform lambda and delta_param
        lambda_param = inverse_transform_probability(transformed_lambda)
        delta_param = inverse_transform_probability(transformed_delta_param)

        # Compute alpha and beta
        alpha = lambda_param * delta_param
        beta = lambda_param * (1 - delta_param)

        return cls(omega=omega, alpha=alpha, gamma=gamma, beta=beta, delta=delta)


class APARCHModel(VolatilityModel):
    """APARCH (Asymmetric Power ARCH) model.

    This class implements the APARCH(1,1) model for modeling time-varying volatility
    in financial time series. The model is defined as:

    σ^δ_t = ω + α * (|r_{t-1}| - γ*r_{t-1})^δ + β * σ^δ_{t-1}

    where σ^δ_t is the conditional power-transformed variance at time t, r_{t-1} is the
    previous return, and γ is the asymmetry parameter that allows different impacts of
    positive and negative shocks on volatility. The parameter δ is the power parameter
    that allows modeling volatility at different powers than the standard squared returns.
    """

    def __init__(self,
                 parameters: Optional[APARCHParams] = None,
                 name: str = "APARCH(1,1)") -> None:
        """Initialize the APARCH model.

        Args:
            parameters: Pre-specified model parameters if available
            name: A descriptive name for the model
        """
        super().__init__(parameters=parameters, name=name)

        # Initialize additional attributes
        self._conditional_variances = None
        self._conditional_power_variances = None

    def parameter_class(self) -> Type[APARCHParams]:
        """Get the parameter class for this model.

        Returns:
            Type[APARCHParams]: The parameter class for this model
        """
        return APARCHParams

    def compute_variance(self,
                         parameters: APARCHParams,
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

        # Allocate array for conditional power-transformed variances
        sigma_delta = np.zeros(T)

        # Compute backcast value if not provided
        if backcast is None:
            # For APARCH, we need to backcast the power-transformed variance
            # We use the mean of |r|^δ as a simple approximation
            delta = parameters.delta
            backcast = np.mean(np.abs(data)**delta)

        try:
            # Extract parameters
            omega = parameters.omega
            alpha = parameters.alpha
            gamma = parameters.gamma
            beta = parameters.beta
            delta = parameters.delta

            # Use optimized Numba implementation to compute power-transformed variances
            sigma_delta = _aparch_recursion(data, omega, alpha, gamma, beta, delta, sigma_delta, backcast)

            # Convert power-transformed variances to standard variances
            sigma2 = sigma_delta**(2.0/delta)

            # Store power-transformed variances for later use
            self._conditional_power_variances = sigma_delta

            # Check for numerical issues
            if np.any(~np.isfinite(sigma2)) or np.any(sigma2 <= 0):
                raise_numeric_error(
                    "Numerical issues detected in variance computation",
                    operation="APARCH variance computation",
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
                operation="APARCH variance computation",
                error_type="Computation error",
                details=str(e)
            )

    def _generate_starting_values(self,
                                  data: np.ndarray,
                                  variance_targeting: bool = False,
                                  backcast: Optional[float] = None) -> APARCHParams:
        """Generate starting values for parameter estimation.

        Args:
            data: Input data (typically residuals)
            variance_targeting: Whether to use variance targeting
            backcast: Value to use for initializing the variance process

        Returns:
            APARCHParams: Starting parameter values
        """
        # Compute sample variance
        sample_variance = np.var(data)

        # For APARCH, we need reasonable starting values
        if variance_targeting:
            # With variance targeting, omega is determined by the other parameters
            alpha = 0.05
            gamma = 0.0  # Start with symmetric model
            beta = 0.90
            delta = 2.0  # Start with standard GARCH power

            # Compute omega based on variance targeting
            # For delta=2, this is similar to GARCH
            omega = sample_variance * (1 - alpha - beta)
        else:
            # Without variance targeting, use reasonable defaults
            omega = 0.05 * sample_variance
            alpha = 0.05
            gamma = 0.0  # Start with symmetric model
            beta = 0.90
            delta = 2.0  # Start with standard GARCH power

        return APARCHParams(omega=omega, alpha=alpha, gamma=gamma, beta=beta, delta=delta)

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
        delta = self._parameters.delta

        # For APARCH, the unconditional variance is more complex due to the asymmetry and power
        # We need to compute E[|z| - γ*z]^δ for standard normal z
        # This is a constant for a given gamma and delta

        # Numerical approximation for the expectation
        expected_asymm_term = 0.0
        n_points = 1000
        z_values = np.linspace(-4.0, 4.0, n_points)
        pdf_values = np.exp(-0.5 * z_values**2) / np.sqrt(2.0 * np.pi)
        for i in range(n_points):
            z = z_values[i]
            pdf = pdf_values[i]
            expected_asymm_term += (abs(z) - gamma * z)**delta * pdf
        expected_asymm_term *= 8.0 / n_points  # Scale for the range [-4, 4]

        # Compute persistence
        persistence = beta + alpha * expected_asymm_term

        if persistence >= 1:
            # If the process is not stationary, return the last observed variance
            if self._conditional_variances is not None:
                return self._conditional_variances[-1]
            else:
                # Fallback to a reasonable value
                return omega / 0.05

        # Compute unconditional power-transformed variance
        unconditional_power_variance = omega / (1 - persistence)

        # Convert to standard variance
        unconditional_variance = unconditional_power_variance**(2.0/delta)

        return unconditional_variance

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
        delta = self._parameters.delta

        # For simulation, we need to compute the power-transformed variance
        # and then convert back to standard variance

        # Compute power-transformed variance for time t
        if t == 0:
            # For t=0, use unconditional variance
            power_variance = self._compute_unconditional_variance()**(delta/2.0)
        else:
            # For t>0, use the APARCH recursion
            # First, convert previous variance to power-transformed variance
            prev_power_variance = sigma2[t-1]**(delta/2.0)

            # Compute |r_{t-1}| - γ*r_{t-1} term
            r_prev = innovations[t-1]
            asymm_term = abs(r_prev) - gamma * r_prev

            # Compute power-transformed variance
            power_variance = omega + alpha * asymm_term**delta + beta * prev_power_variance

        # Convert power-transformed variance to standard variance
        variance = power_variance**(2.0/delta)

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
        delta = self._parameters.delta

        # For APARCH, we need to forecast the power-transformed variance
        # and then convert back to standard variance

        # Convert last variance to power-transformed variance
        last_power_variance = last_variance**(delta/2.0)

        # Use optimized Numba implementation to forecast power-transformed variances
        power_forecasts = _aparch_forecast(
            omega, alpha, gamma, beta, delta, last_power_variance, steps
        )

        # Convert power-transformed forecasts to standard variance forecasts
        forecasts = power_forecasts**(2.0/delta)

        return forecasts

    @property
    def conditional_variances(self) -> Optional[np.ndarray]:
        """Get the conditional variances from the fitted model.

        Returns:
            Optional[np.ndarray]: Conditional variances if the model has been fitted, None otherwise
        """
        return self._conditional_variances

    @property
    def conditional_power_variances(self) -> Optional[np.ndarray]:
        """Get the conditional power-transformed variances from the fitted model.

        Returns:
            Optional[np.ndarray]: Conditional power-transformed variances if the model has been fitted, None otherwise
        """
        return self._conditional_power_variances

    @property
    def conditional_volatilities(self) -> Optional[np.ndarray]:
        """Get the conditional volatilities from the fitted model.

        Returns:
            Optional[np.ndarray]: Conditional volatilities if the model has been fitted, None otherwise
        """
        if self._conditional_variances is not None:
            return np.sqrt(self._conditional_variances)
        return None

    @property
    def power_parameter(self) -> Optional[float]:
        """Get the power parameter (delta) from the fitted model.

        Returns:
            Optional[float]: Power parameter if the model has been fitted, None otherwise
        """
        if self._parameters is not None:
            return self._parameters.delta
        return None

    @property
    def asymmetry_parameter(self) -> Optional[float]:
        """Get the asymmetry parameter (gamma) from the fitted model.

        Returns:
            Optional[float]: Asymmetry parameter if the model has been fitted, None otherwise
        """
        if self._parameters is not None:
            return self._parameters.gamma
        return None

    def fit(self,
            data: np.ndarray,
            starting_values: Optional[Union[np.ndarray, APARCHParams]] = None,
            distribution: Literal["normal", "t", "skewed-t", "ged"] = "normal",
            variance_targeting: bool = False,
            backcast: Optional[float] = None,
            method: str = "SLSQP",
            options: Optional[Dict[str, Any]] = None,
            **kwargs: Any) -> 'UnivariateVolatilityResult':
        """Fit the APARCH model to the provided data.

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
        # Store the data for later use
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

        return result

    async def fit_async(self,
                        data: np.ndarray,
                        starting_values: Optional[Union[np.ndarray, APARCHParams]] = None,
                        distribution: Literal["normal", "t", "skewed-t", "ged"] = "normal",
                        variance_targeting: bool = False,
                        backcast: Optional[float] = None,
                        method: str = "SLSQP",
                        options: Optional[Dict[str, Any]] = None,
                        progress_callback: Optional[Callable[[float, str], None]] = None,
                        **kwargs: Any) -> 'UnivariateVolatilityResult':
        """Asynchronously fit the APARCH model to the provided data.

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
        # Store the data for later use
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
        """Simulate data from the APARCH model.

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
        """Asynchronously simulate data from the APARCH model.

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
        """Generate volatility forecasts from the fitted APARCH model.

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
        """Asynchronously generate volatility forecasts from the fitted APARCH model.

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
        """Validate input data for the APARCH model.

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
        if len(data) < 2:  # APARCH(1,1) needs at least 2 observations
            raise ValueError(
                f"data must have at least 2 observations, got {len(data)}"
            )

        # Check for NaN or infinite values
        if np.any(~np.isfinite(data)):
            raise ValueError("data contains NaN or infinite values")

    def plot_diagnostics(self, figsize: Tuple[int, int] = (12, 10)) -> None:
        """Plot diagnostic plots for the fitted APARCH model.

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

        # Plot conditional power-transformed volatility
        if self._conditional_power_variances is not None:
            delta = self._parameters.delta
            axs[0, 1].plot(self._conditional_power_variances**(1.0/delta))
            axs[0, 1].set_title(f'Conditional Power-Transformed Volatility (δ={delta:.2f})')
            axs[0, 1].set_xlabel('Time')
            axs[0, 1].set_ylabel(f'σ^{delta:.2f}')

        # Plot standardized residuals
        if hasattr(self, '_data') and self._data is not None:
            std_residuals = self._data / np.sqrt(self._conditional_variances)
            axs[1, 0].plot(std_residuals)
            axs[1, 0].set_title('Standardized Residuals')
            axs[1, 0].set_xlabel('Time')
            axs[1, 0].set_ylabel('Standardized Residuals')

            # Plot histogram of standardized residuals
            axs[1, 1].hist(std_residuals, bins=50, density=True, alpha=0.6)

            # Add normal distribution for comparison
            x = np.linspace(min(std_residuals), max(std_residuals), 100)
            axs[1, 1].plot(x, stats.norm.pdf(x, 0, 1), 'r-', lw=2)

            axs[1, 1].set_title('Histogram of Standardized Residuals')
            axs[1, 1].set_xlabel('Standardized Residuals')
            axs[1, 1].set_ylabel('Density')

            # Plot QQ plot of standardized residuals
            from scipy import stats
            stats.probplot(std_residuals, dist="norm", plot=axs[2, 0])
            axs[2, 0].set_title('QQ Plot of Standardized Residuals')

            # Plot asymmetry effect
            if self._parameters is not None:
                gamma = self._parameters.gamma
                delta = self._parameters.delta

                # Create a range of values for demonstration
                x_vals = np.linspace(-3, 3, 100)
                y_vals = (np.abs(x_vals) - gamma * x_vals)**delta

                axs[2, 1].plot(x_vals, y_vals)
                axs[2, 1].set_title(f'Asymmetry Effect: (|x| - γx)^δ\nγ={gamma:.2f}, δ={delta:.2f}')
                axs[2, 1].set_xlabel('x')
                axs[2, 1].set_ylabel('(|x| - γx)^δ')
                axs[2, 1].axvline(x=0, color='k', linestyle='--', alpha=0.5)
                axs[2, 1].grid(True, alpha=0.3)

        # Adjust layout and show plot
        plt.tight_layout()
        plt.show()

    def __str__(self) -> str:
        """Get a string representation of the APARCH model.

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
        delta = self._parameters.delta

        # Compute persistence (simplified approximation)
        persistence = alpha + beta

        # Create string representation
        model_str = f"{self.name} model\n"
        model_str += f"omega: {omega:.6f}\n"
        model_str += f"alpha: {alpha:.6f}\n"
        model_str += f"gamma: {gamma:.6f}\n"
        model_str += f"beta: {beta:.6f}\n"
        model_str += f"delta: {delta:.6f}\n"
        model_str += f"persistence: {persistence:.6f}\n"

        # Add unconditional variance if model is stationary
        if persistence < 1:
            uncond_var = self._compute_unconditional_variance()
            model_str += f"unconditional variance: {uncond_var:.6f}\n"

        return model_str

    def __repr__(self) -> str:
        """Get a string representation of the APARCH model.

        Returns:
            str: String representation of the model
        """
        return self.__str__()
