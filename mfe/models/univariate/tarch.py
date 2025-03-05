# mfe/models/univariate/tarch.py
"""
Threshold ARCH (TARCH) model implementation.

This module provides a comprehensive implementation of the TARCH(1,1) model for
modeling asymmetric volatility in financial time series. The TARCH model extends
the GARCH model by allowing different responses to positive and negative returns,
capturing the leverage effect commonly observed in financial markets.

The implementation includes parameter estimation via maximum likelihood,
simulation, forecasting, and diagnostic tools. Performance-critical operations
are accelerated using Numba's just-in-time compilation.

References:
    Zakoian, J.M. (1994). Threshold heteroskedastic models.
    Journal of Economic Dynamics and Control, 18(5), 931-955.
    
    Glosten, L.R., Jagannathan, R., & Runkle, D.E. (1993).
    On the relation between the expected value and the volatility of the nominal
    excess return on stocks. The Journal of Finance, 48(5), 1779-1801.
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
    TARCHParameters, UnivariateVolatilityParameters, ParameterError,
    validate_positive, validate_non_negative, validate_probability, validate_range
)
from mfe.core.exceptions import (
    MFEError, ConvergenceError, NumericError, EstimationError, SimulationError,
    NotFittedError, raise_convergence_error, raise_numeric_error, raise_not_fitted_error,
    warn_convergence, warn_numeric, warn_model
)
from mfe.models.univariate.base import VolatilityModel


# Numba-accelerated core functions for TARCH model
@jit(nopython=True, cache=True)
def _tarch_recursion(data: np.ndarray,
                     omega: float,
                     alpha: float,
                     gamma: float,
                     beta: float,
                     sigma2: np.ndarray,
                     backcast: float,
                     tarch_type: int = 1) -> np.ndarray:
    """Compute TARCH(1,1) conditional variances using Numba acceleration.

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


@jit(nopython=True, cache=True)
def _tarch_forecast(omega: float,
                    alpha: float,
                    gamma: float,
                    beta: float,
                    last_variance: float,
                    steps: int,
                    tarch_type: int = 1) -> np.ndarray:
    """Generate analytic forecasts for TARCH(1,1) model using Numba acceleration.

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


@dataclass
class TARCHParams(TARCHParameters):
    """Parameters for TARCH(1,1) model.

    This class extends the base TARCHParameters class to provide
    parameter validation and transformation methods specific to the TARCH model.

    Attributes:
        omega: Constant term in variance equation (must be positive)
        alpha: ARCH parameter (must be non-negative)
        gamma: Asymmetry parameter (must be non-negative)
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
        """Validate TARCH parameter constraints.

        Raises:
            ParameterError: If parameter constraints are violated
        """
        super().validate()

        # Validate individual parameters
        validate_positive(self.omega, "omega")
        validate_non_negative(self.alpha, "alpha")
        validate_non_negative(self.gamma, "gamma")
        validate_non_negative(self.beta, "beta")

        # Validate stationarity constraint
        # For TARCH, the constraint is alpha + beta + 0.5*gamma < 1
        if self.alpha + self.beta + 0.5 * self.gamma >= 1:
            raise ParameterError(
                f"TARCH stationarity constraint violated: "
                f"alpha + beta + 0.5*gamma = {self.alpha + self.beta + 0.5 * self.gamma} >= 1"
            )

    def to_array(self) -> np.ndarray:
        """Convert parameters to a NumPy array.

        Returns:
            np.ndarray: Array representation of parameters
        """
        return np.array([self.omega, self.alpha, self.gamma, self.beta])

    @classmethod
    def from_array(cls, array: np.ndarray, **kwargs: Any) -> 'TARCHParams':
        """Create parameters from a NumPy array.

        Args:
            array: Array representation of parameters
            **kwargs: Additional keyword arguments for parameter creation

        Returns:
            TARCHParams: Parameter object

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
        from mfe.core.parameters import transform_positive, transform_probability

        # Transform omega to unconstrained space (log)
        transformed_omega = transform_positive(self.omega)

        # Transform alpha, gamma, and beta to unconstrained space
        # We use a special transformation to ensure alpha + beta + 0.5*gamma < 1
        sum_params = self.alpha + self.beta + 0.5 * self.gamma
        if sum_params >= 1:
            # If constraint is violated, adjust parameters slightly
            factor = 0.99 / sum_params
            self.alpha *= factor
            self.beta *= factor
            self.gamma *= factor

        # Use a transformation that preserves the constraint
        # We parameterize in terms of:
        # lambda = alpha + beta + 0.5*gamma (must be < 1)
        # delta1 = alpha / lambda
        # delta2 = gamma / (2 * lambda)
        # This ensures beta = lambda * (1 - delta1 - delta2)

        lambda_param = self.alpha + self.beta + 0.5 * self.gamma
        delta1 = self.alpha / lambda_param if lambda_param > 0 else 0.33
        delta2 = 0.5 * self.gamma / lambda_param if lambda_param > 0 else 0.33

        transformed_lambda = transform_probability(lambda_param)
        transformed_delta1 = transform_probability(delta1)
        transformed_delta2 = transform_probability(delta2)

        return np.array([
            transformed_omega,
            transformed_lambda,
            transformed_delta1,
            transformed_delta2
        ])

    @classmethod
    def inverse_transform(cls, array: np.ndarray, **kwargs: Any) -> 'TARCHParams':
        """Transform parameters from unconstrained space back to constrained space.

        Args:
            array: Parameters in unconstrained space [omega*, lambda*, delta1*, delta2*]
            **kwargs: Additional keyword arguments for parameter creation

        Returns:
            TARCHParams: Parameter object with constrained parameters

        Raises:
            ValueError: If the array length is not 4
        """
        from mfe.core.parameters import inverse_transform_positive, inverse_transform_probability

        if len(array) != 4:
            raise ValueError(f"Array length must be 4, got {len(array)}")

        # Extract transformed parameters
        transformed_omega, transformed_lambda, transformed_delta1, transformed_delta2 = array

        # Inverse transform omega
        omega = inverse_transform_positive(transformed_omega)

        # Inverse transform lambda, delta1, and delta2
        lambda_param = inverse_transform_probability(transformed_lambda)
        delta1 = inverse_transform_probability(transformed_delta1)
        delta2 = inverse_transform_probability(transformed_delta2)

        # Ensure delta1 + delta2 <= 1 (for numerical stability)
        if delta1 + delta2 > 1:
            sum_deltas = delta1 + delta2
            delta1 = delta1 / sum_deltas * 0.99
            delta2 = delta2 / sum_deltas * 0.99

        # Compute alpha, gamma, and beta
        alpha = lambda_param * delta1
        gamma = lambda_param * delta2 * 2
        beta = lambda_param * (1 - delta1 - delta2)

        return cls(omega=omega, alpha=alpha, gamma=gamma, beta=beta)


class TARCH(VolatilityModel):
    """TARCH (Threshold ARCH) model for asymmetric volatility.

    This class implements the TARCH(1,1) model for modeling asymmetric volatility
    in financial time series. The model allows different responses to positive and
    negative returns, capturing the leverage effect commonly observed in financial markets.

    The model is defined as:

    σ²_t = ω + α * r²_{t-1} + γ * I_{t-1} * r²_{t-1} + β * σ²_{t-1}

    where I_{t-1} is an indicator function that equals 1 if r_{t-1} < 0 and 0 otherwise.

    Attributes:
        tarch_type: Type of TARCH model (1 for squared innovations, 2 for absolute innovations)
        name: Model name
        _parameters: Model parameters if set
        _results: Estimation results if the model has been fitted
        _conditional_variances: Conditional variances if the model has been fitted
    """

    def __init__(self,
                 tarch_type: int = 1,
                 parameters: Optional[TARCHParams] = None,
                 name: str = "TARCH") -> None:
        """Initialize the TARCH model.

        Args:
            tarch_type: Type of TARCH model (1 for squared innovations, 2 for absolute innovations)
            parameters: Pre-specified model parameters if available
            name: A descriptive name for the model

        Raises:
            ValueError: If tarch_type is not 1 or 2
        """
        if tarch_type not in [1, 2]:
            raise ValueError(f"tarch_type must be 1 or 2, got {tarch_type}")

        self.tarch_type = tarch_type

        # Set model name based on tarch_type
        if name == "TARCH":
            if tarch_type == 1:
                name = "GJR-GARCH(1,1)"  # Squared innovations (GJR-GARCH)
            else:
                name = "TARCH(1,1)"  # Absolute innovations (Zakoian's TARCH)

        super().__init__(parameters=parameters, name=name)

        # Initialize additional attributes
        self._conditional_variances = None
        self._data = None

    def parameter_class(self) -> Type[TARCHParams]:
        """Get the parameter class for this model.

        Returns:
            Type[TARCHParams]: The parameter class for this model
        """
        return TARCHParams

    def compute_variance(self,
                         parameters: TARCHParams,
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
            alpha = parameters.alpha
            gamma = parameters.gamma
            beta = parameters.beta

            sigma2 = _tarch_recursion(data, omega, alpha, gamma, beta, sigma2, backcast, self.tarch_type)

            # Check for numerical issues
            if np.any(~np.isfinite(sigma2)) or np.any(sigma2 <= 0):
                raise_numeric_error(
                    "Numerical issues detected in variance computation",
                    operation="TARCH variance computation",
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
                operation="TARCH variance computation",
                error_type="Computation error",
                details=str(e)
            )

    def _generate_starting_values(self,
                                  data: np.ndarray,
                                  variance_targeting: bool = False,
                                  backcast: Optional[float] = None) -> TARCHParams:
        """Generate starting values for parameter estimation.

        Args:
            data: Input data (typically residuals)
            variance_targeting: Whether to use variance targeting
            backcast: Value to use for initializing the variance process

        Returns:
            TARCHParams: Starting parameter values
        """
        # Compute sample variance
        sample_variance = np.var(data)

        if backcast is None:
            backcast = sample_variance

        # Generate reasonable starting values
        if variance_targeting:
            # With variance targeting, omega is determined by the other parameters
            alpha = 0.05
            gamma = 0.10  # Positive gamma to capture leverage effect
            beta = 0.80
            omega = sample_variance * (1 - alpha - beta - 0.5 * gamma)
        else:
            # Without variance targeting, use reasonable defaults
            omega = 0.05 * sample_variance
            alpha = 0.05
            gamma = 0.10  # Positive gamma to capture leverage effect
            beta = 0.80

        # Ensure omega is positive
        omega = max(omega, 1e-6)

        return TARCHParams(omega=omega, alpha=alpha, gamma=gamma, beta=beta)

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

        # For TARCH model, unconditional variance is omega / (1 - alpha - beta - 0.5*gamma)
        omega = self._parameters.omega
        alpha = self._parameters.alpha
        gamma = self._parameters.gamma
        beta = self._parameters.beta

        # Compute persistence
        persistence = alpha + beta + 0.5 * gamma

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

        # Compute variance for time t
        if t > 0:
            # Determine if previous return was negative
            neg_indicator = 1.0 if innovations[t-1] < 0 else 0.0

            if self.tarch_type == 1:  # Squared innovations (GJR-GARCH)
                # GJR-GARCH formulation with squared innovations
                variance = omega + alpha * innovations[t-1]**2 + gamma * \
                    neg_indicator * innovations[t-1]**2 + beta * sigma2[t-1]
            else:  # Absolute innovations (Zakoian's TARCH)
                # Original TARCH formulation with absolute innovations
                # Note: This returns conditional standard deviation, not variance
                sigma = np.sqrt(sigma2[t-1])
                variance = (omega + alpha * abs(innovations[t-1]) + gamma *
                            neg_indicator * abs(innovations[t-1]) + beta * sigma)**2
        else:
            # For t=0, use unconditional variance
            variance = self._compute_unconditional_variance()

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

        # Use optimized Numba implementation
        return _tarch_forecast(omega, alpha, gamma, beta, last_variance, steps, self.tarch_type)

    def fit(self,
            data: np.ndarray,
            starting_values: Optional[Union[np.ndarray, TARCHParams]] = None,
            distribution: Literal["normal", "t", "skewed-t", "ged"] = "normal",
            variance_targeting: bool = False,
            backcast: Optional[float] = None,
            method: str = "SLSQP",
            options: Optional[Dict[str, Any]] = None,
            **kwargs: Any) -> 'UnivariateVolatilityResult':
        """Fit the TARCH model to the provided data.

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
                        starting_values: Optional[Union[np.ndarray, TARCHParams]] = None,
                        distribution: Literal["normal", "t", "skewed-t", "ged"] = "normal",
                        variance_targeting: bool = False,
                        backcast: Optional[float] = None,
                        method: str = "SLSQP",
                        options: Optional[Dict[str, Any]] = None,
                        progress_callback: Optional[Callable[[float, str], None]] = None,
                        **kwargs: Any) -> 'UnivariateVolatilityResult':
        """Asynchronously fit the TARCH model to the provided data.

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
        """Simulate data from the TARCH model.

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
        """Asynchronously simulate data from the TARCH model.

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
        """Generate volatility forecasts from the fitted TARCH model.

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
        """Asynchronously generate volatility forecasts from the fitted TARCH model.

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
        """Validate input data for the TARCH model.

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
        if len(data) < 2:  # TARCH(1,1) needs at least 2 observations
            raise ValueError(
                f"data must have at least 2 observations, got {len(data)}"
            )

        # Check for NaN or infinite values
        if np.any(~np.isfinite(data)):
            raise ValueError("data contains NaN or infinite values")

    def plot_news_impact(self, figsize: Tuple[int, int] = (10, 6)) -> None:
        """Plot the news impact curve for the fitted TARCH model.

        The news impact curve shows how past returns affect future volatility,
        highlighting the asymmetric response to positive and negative returns.

        Args:
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

        # Create a range of return values
        if self._data is not None:
            # Use the range of the data
            data_std = np.std(self._data)
            x_range = np.linspace(-4 * data_std, 4 * data_std, 1000)
        else:
            # Use a standard range
            x_range = np.linspace(-4, 4, 1000)

        # Compute the news impact
        if self.tarch_type == 1:  # Squared innovations (GJR-GARCH)
            # For GJR-GARCH, the news impact is:
            # omega + alpha * r^2 + gamma * I(r<0) * r^2
            # where I(r<0) is the indicator function

            # Assume a baseline volatility level (e.g., unconditional variance)
            baseline = self._compute_unconditional_variance()

            # Compute the news impact (excluding the beta * sigma^2 term)
            news_impact = np.zeros_like(x_range)
            for i, r in enumerate(x_range):
                if r < 0:
                    news_impact[i] = omega + (alpha + gamma) * r**2 + beta * baseline
                else:
                    news_impact[i] = omega + alpha * r**2 + beta * baseline
        else:  # Absolute innovations (Zakoian's TARCH)
            # For Zakoian's TARCH, the news impact is:
            # (omega + alpha * |r| + gamma * I(r<0) * |r| + beta * sigma)^2

            # Assume a baseline volatility level (e.g., unconditional standard deviation)
            baseline = np.sqrt(self._compute_unconditional_variance())

            # Compute the news impact
            news_impact = np.zeros_like(x_range)
            for i, r in enumerate(x_range):
                if r < 0:
                    news_impact[i] = (omega + (alpha + gamma) * abs(r) + beta * baseline)**2
                else:
                    news_impact[i] = (omega + alpha * abs(r) + beta * baseline)**2

        # Create the plot
        plt.figure(figsize=figsize)
        plt.plot(x_range, news_impact)
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        plt.title(f'News Impact Curve for {self.name}')
        plt.xlabel('Return (r)')
        plt.ylabel('Conditional Variance (σ²)')
        plt.grid(True, alpha=0.3)

        # Add a legend explaining the asymmetry
        if gamma > 0:
            plt.legend(['News Impact Curve', 'Zero Return'],
                       title=f'Asymmetry (γ): {gamma:.4f}')
        else:
            plt.legend(['News Impact Curve', 'Zero Return'])

        plt.tight_layout()
        plt.show()

    def plot_diagnostics(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """Plot diagnostic plots for the fitted TARCH model.

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
        """Get a string representation of the TARCH model.

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
        persistence = alpha + beta + 0.5 * gamma

        # Create string representation
        model_str = f"{self.name} model\n"
        model_str += f"omega: {omega:.6f}\n"
        model_str += f"alpha: {alpha:.6f}\n"
        model_str += f"gamma: {gamma:.6f}\n"
        model_str += f"beta: {beta:.6f}\n"
        model_str += f"persistence: {persistence:.6f}\n"

        # Add unconditional variance if model is stationary
        if persistence < 1:
            uncond_var = self._compute_unconditional_variance()
            model_str += f"unconditional variance: {uncond_var:.6f}\n"

        # Add model type information
        if self.tarch_type == 1:
            model_str += "type: GJR-GARCH (squared innovations)\n"
        else:
            model_str += "type: Zakoian's TARCH (absolute innovations)\n"

        return model_str

    def __repr__(self) -> str:
        """Get a string representation of the TARCH model.

        Returns:
            str: String representation of the model
        """
        return self.__str__()
