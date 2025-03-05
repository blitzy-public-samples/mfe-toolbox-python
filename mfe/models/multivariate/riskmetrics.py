# mfe/models/multivariate/riskmetrics.py

"""
RiskMetrics and RiskMetrics2006 multivariate volatility models.

This module implements the RiskMetrics and RiskMetrics2006 models for multivariate
volatility modeling. These models provide simple exponentially weighted moving average
approaches to covariance estimation that are computationally efficient and widely
used in industry applications for risk measurement.

The RiskMetrics model uses a single decay factor (lambda) to update covariance
estimates, while the RiskMetrics2006 model employs a more sophisticated approach
with multiple decay factors to capture both short-term and long-term volatility
dynamics.

References:
    J.P. Morgan/Reuters. (1996). RiskMetrics Technical Document (4th ed.).
    Zumbach, G. (2007). The RiskMetrics 2006 methodology. RiskMetrics Group.
"""

import logging
import warnings
import asyncio
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union, cast, overload
)

import numpy as np
import pandas as pd
from scipy import optimize, stats

from mfe.core.base import ModelBase
from mfe.core.parameters import (
    MultivariateVolatilityParameters, ParameterBase, ParameterError,
    validate_positive_definite, validate_probability, validate_range
)
from mfe.core.results import MultivariateVolatilityResult
from mfe.core.types import (
    AsyncProgressCallback, CovarianceMatrix, CorrelationMatrix, Matrix,
    MultivariateVolatilityType, ProgressCallback, Vector
)
from mfe.models.multivariate.base import CovarianceModelBase
from mfe.utils.matrix_ops import cov2corr, vech, ivech, ensure_symmetric

# Set up module-level logger
logger = logging.getLogger("mfe.models.multivariate.riskmetrics")

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
    logger.debug("Numba available for RiskMetrics model acceleration")
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    HAS_NUMBA = False
    logger.info("Numba not available. RiskMetrics model will use pure NumPy implementations.")


@dataclass
class RiskMetricsParameters(MultivariateVolatilityParameters):
    """Parameters for the RiskMetrics model.

    This class holds the decay factor (lambda) for the RiskMetrics model.

    Attributes:
        lambda_: Decay factor (must be between 0 and 1)
    """

    lambda_: float = 0.94  # Default value from RiskMetrics Technical Document

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate RiskMetrics parameter constraints.

        Raises:
            ParameterError: If parameter constraints are violated
        """
        super().validate()

        # Validate lambda (must be between 0 and 1)
        validate_probability(self.lambda_, "lambda_")

    def to_array(self) -> np.ndarray:
        """Convert parameters to a NumPy array.

        Returns:
            np.ndarray: Array representation of parameters
        """
        return np.array([self.lambda_])

    @classmethod
    def from_array(cls, array: np.ndarray, **kwargs: Any) -> 'RiskMetricsParameters':
        """Create parameters from a NumPy array.

        Args:
            array: Array representation of parameters
            **kwargs: Additional keyword arguments for parameter creation

        Returns:
            RiskMetricsParameters: Parameter object

        Raises:
            ValueError: If the array length is not 1
        """
        if len(array) != 1:
            raise ValueError(f"Array length must be 1, got {len(array)}")

        return cls(lambda_=array[0])

    def transform(self) -> np.ndarray:
        """Transform parameters to unconstrained space for optimization.

        Returns:
            np.ndarray: Transformed parameters in unconstrained space
        """
        # Transform lambda to unconstrained space using logit transformation
        # logit(x) = log(x / (1 - x))
        eps = np.finfo(float).eps
        lambda_clipped = np.clip(self.lambda_, eps, 1 - eps)
        transformed_lambda = np.log(lambda_clipped / (1 - lambda_clipped))

        return np.array([transformed_lambda])

    @classmethod
    def inverse_transform(cls, array: np.ndarray, **kwargs: Any) -> 'RiskMetricsParameters':
        """Transform parameters from unconstrained space back to constrained space.

        Args:
            array: Parameters in unconstrained space
            **kwargs: Additional keyword arguments for parameter creation

        Returns:
            RiskMetricsParameters: Parameter object with constrained parameters

        Raises:
            ValueError: If the array length is not 1
        """
        if len(array) != 1:
            raise ValueError(f"Array length must be 1, got {len(array)}")

        # Inverse transform lambda using sigmoid function
        # sigmoid(x) = 1 / (1 + exp(-x))
        lambda_ = 1.0 / (1.0 + np.exp(-array[0]))

        return cls(lambda_=lambda_)


@dataclass
class RiskMetrics2006Parameters(MultivariateVolatilityParameters):
    """Parameters for the RiskMetrics2006 model.

    This class holds the parameters for the RiskMetrics2006 model, which uses
    multiple decay factors to capture both short-term and long-term volatility.

    Attributes:
        decay_factors: Array of decay factors (must be between 0 and 1)
        weights: Array of weights for each decay factor (must sum to 1)
    """

    decay_factors: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        # Ensure arrays are NumPy arrays
        if not isinstance(self.decay_factors, np.ndarray):
            self.decay_factors = np.array(self.decay_factors)
        if not isinstance(self.weights, np.ndarray):
            self.weights = np.array(self.weights)

        self.validate()

    def validate(self) -> None:
        """Validate RiskMetrics2006 parameter constraints.

        Raises:
            ParameterError: If parameter constraints are violated
        """
        super().validate()

        # Validate decay factors (must be between 0 and 1)
        for i, factor in enumerate(self.decay_factors):
            validate_probability(factor, f"decay_factors[{i}]")

        # Validate weights (must be non-negative and sum to 1)
        for i, weight in enumerate(self.weights):
            validate_range(weight, f"weights[{i}]", 0.0, 1.0)

        if not np.isclose(np.sum(self.weights), 1.0):
            raise ParameterError(
                f"Weights must sum to 1, got {np.sum(self.weights)}"
            )

        # Validate that decay_factors and weights have the same length
        if len(self.decay_factors) != len(self.weights):
            raise ParameterError(
                f"decay_factors and weights must have the same length, "
                f"got {len(self.decay_factors)} and {len(self.weights)}"
            )

    def to_array(self) -> np.ndarray:
        """Convert parameters to a NumPy array.

        Returns:
            np.ndarray: Array representation of parameters
        """
        # Concatenate decay factors and weights (excluding the last weight)
        # The last weight is determined by the constraint that weights sum to 1
        return np.concatenate([
            self.decay_factors,
            self.weights[:-1]
        ])

    @classmethod
    def from_array(cls, array: np.ndarray, n_factors: Optional[int] = None, **kwargs: Any) -> 'RiskMetrics2006Parameters':
        """Create parameters from a NumPy array.

        Args:
            array: Array representation of parameters
            n_factors: Number of decay factors (if None, inferred from array length)
            **kwargs: Additional keyword arguments for parameter creation

        Returns:
            RiskMetrics2006Parameters: Parameter object

        Raises:
            ValueError: If the array length is invalid
        """
        if n_factors is None:
            # Infer n_factors from array length
            # array contains decay_factors and weights[:-1]
            # so length should be 2*n_factors - 1
            n_factors = (len(array) + 1) // 2

            if len(array) != 2 * n_factors - 1:
                raise ValueError(
                    f"Array length ({len(array)}) is not valid for any number of factors"
                )
        else:
            # Validate array length
            if len(array) != 2 * n_factors - 1:
                raise ValueError(
                    f"Array length must be {2 * n_factors - 1} for {n_factors} factors, "
                    f"got {len(array)}"
                )

        # Extract decay factors and weights
        decay_factors = array[:n_factors]
        weights_partial = array[n_factors:]

        # Compute the last weight to ensure sum is 1
        last_weight = 1.0 - np.sum(weights_partial)
        weights = np.concatenate([weights_partial, [last_weight]])

        return cls(decay_factors=decay_factors, weights=weights)

    def transform(self) -> np.ndarray:
        """Transform parameters to unconstrained space for optimization.

        Returns:
            np.ndarray: Transformed parameters in unconstrained space
        """
        n_factors = len(self.decay_factors)
        transformed_params = np.zeros(2 * n_factors - 1)

        # Transform decay factors using logit transformation
        eps = np.finfo(float).eps
        for i, factor in enumerate(self.decay_factors):
            factor_clipped = np.clip(factor, eps, 1 - eps)
            transformed_params[i] = np.log(factor_clipped / (1 - factor_clipped))

        # Transform weights using a multinomial logit transformation
        # We only need to transform n_factors - 1 weights, as the last one
        # is determined by the constraint that weights sum to 1
        if n_factors > 1:
            # Use softmax-based parameterization
            # We transform to log(w_i / w_n) for i=1,...,n-1
            last_weight = self.weights[-1]
            for i in range(n_factors - 1):
                weight_i = self.weights[i]
                # Avoid numerical issues
                if last_weight < eps or weight_i < eps:
                    transformed_params[n_factors + i] = -20.0 if last_weight > weight_i else 20.0
                else:
                    transformed_params[n_factors + i] = np.log(weight_i / last_weight)

        return transformed_params

    @classmethod
    def inverse_transform(cls, array: np.ndarray, n_factors: Optional[int] = None, **kwargs: Any) -> 'RiskMetrics2006Parameters':
        """Transform parameters from unconstrained space back to constrained space.

        Args:
            array: Parameters in unconstrained space
            n_factors: Number of decay factors (if None, inferred from array length)
            **kwargs: Additional keyword arguments for parameter creation

        Returns:
            RiskMetrics2006Parameters: Parameter object with constrained parameters

        Raises:
            ValueError: If the array length is invalid
        """
        if n_factors is None:
            # Infer n_factors from array length
            n_factors = (len(array) + 1) // 2

            if len(array) != 2 * n_factors - 1:
                raise ValueError(
                    f"Array length ({len(array)}) is not valid for any number of factors"
                )
        else:
            # Validate array length
            if len(array) != 2 * n_factors - 1:
                raise ValueError(
                    f"Array length must be {2 * n_factors - 1} for {n_factors} factors, "
                    f"got {len(array)}"
                )

        # Inverse transform decay factors using sigmoid function
        decay_factors = np.zeros(n_factors)
        for i in range(n_factors):
            decay_factors[i] = 1.0 / (1.0 + np.exp(-array[i]))

        # Inverse transform weights using softmax function
        weights = np.zeros(n_factors)
        if n_factors > 1:
            # Extract transformed weights
            transformed_weights = array[n_factors:]

            # Apply softmax transformation
            exp_weights = np.zeros(n_factors)
            exp_weights[-1] = 1.0  # Reference weight
            for i in range(n_factors - 1):
                exp_weights[i] = np.exp(transformed_weights[i])

            # Normalize to sum to 1
            weights = exp_weights / np.sum(exp_weights)
        else:
            # If there's only one factor, the weight is 1
            weights[0] = 1.0

        return cls(decay_factors=decay_factors, weights=weights)


@jit(nopython=True, cache=True)
def _compute_riskmetrics_covariance_numba(
    data: np.ndarray,
    lambda_: float,
    sigma: np.ndarray,
    backcast: np.ndarray
) -> np.ndarray:
    """
    Numba-accelerated implementation of RiskMetrics covariance computation.

    Args:
        data: Input data array with shape (T, n_assets)
        lambda_: Decay factor
        sigma: Pre-allocated array for conditional covariances with shape (n_assets, n_assets, T)
        backcast: Initial covariance matrix with shape (n_assets, n_assets)

    Returns:
        np.ndarray: Conditional covariance matrices with shape (n_assets, n_assets, T)
    """
    T, n_assets = data.shape

    # Initialize first covariance matrix with backcast
    for i in range(n_assets):
        for j in range(n_assets):
            sigma[i, j, 0] = backcast[i, j]

    # Compute covariance matrices recursively
    for t in range(1, T):
        for i in range(n_assets):
            for j in range(n_assets):
                # RiskMetrics update: sigma_t = lambda * sigma_{t-1} + (1 - lambda) * r_{t-1} * r_{t-1}'
                sigma[i, j, t] = lambda_ * sigma[i, j, t-1] + (1 - lambda_) * data[t-1, i] * data[t-1, j]

    return sigma


@jit(nopython=True, cache=True)
def _compute_riskmetrics2006_covariance_numba(
    data: np.ndarray,
    decay_factors: np.ndarray,
    weights: np.ndarray,
    sigma: np.ndarray,
    component_sigmas: np.ndarray,
    backcast: np.ndarray
) -> np.ndarray:
    """
    Numba-accelerated implementation of RiskMetrics2006 covariance computation.

    Args:
        data: Input data array with shape (T, n_assets)
        decay_factors: Array of decay factors with shape (n_factors,)
        weights: Array of weights with shape (n_factors,)
        sigma: Pre-allocated array for conditional covariances with shape (n_assets, n_assets, T)
        component_sigmas: Pre-allocated array for component covariances with shape (n_factors, n_assets, n_assets, T)
        backcast: Initial covariance matrix with shape (n_assets, n_assets)

    Returns:
        np.ndarray: Conditional covariance matrices with shape (n_assets, n_assets, T)
    """
    T, n_assets = data.shape
    n_factors = len(decay_factors)

    # Initialize first covariance matrix for each component with backcast
    for k in range(n_factors):
        for i in range(n_assets):
            for j in range(n_assets):
                component_sigmas[k, i, j, 0] = backcast[i, j]

    # Compute component covariance matrices recursively
    for t in range(1, T):
        # Update each component
        for k in range(n_factors):
            lambda_k = decay_factors[k]
            for i in range(n_assets):
                for j in range(n_assets):
                    # RiskMetrics update for each component
                    component_sigmas[k, i, j, t] = (
                        lambda_k * component_sigmas[k, i, j, t-1] +
                        (1 - lambda_k) * data[t-1, i] * data[t-1, j]
                    )

        # Combine components using weights
        for i in range(n_assets):
            for j in range(n_assets):
                sigma[i, j, t] = 0.0
                for k in range(n_factors):
                    sigma[i, j, t] += weights[k] * component_sigmas[k, i, j, t]

    return sigma


class RiskMetricsModel(CovarianceModelBase):
    """RiskMetrics multivariate volatility model.

    This class implements the RiskMetrics model, which uses an exponentially
    weighted moving average approach to estimate covariance matrices.

    Attributes:
        name: Model name
        n_assets: Number of assets
        parameters: Model parameters if fitted
        _conditional_covariances: Conditional covariance matrices if fitted
        _residuals: Residual data used for fitting
    """

    def __init__(
        self,
        n_assets: Optional[int] = None,
        lambda_: float = 0.94,
        name: str = "RiskMetrics"
    ):
        """Initialize the RiskMetrics model.

        Args:
            n_assets: Number of assets (if None, determined from data)
            lambda_: Decay factor (default: 0.94, from RiskMetrics Technical Document)
            name: Model name
        """
        super().__init__(name=name, n_assets=n_assets)

        # Initialize parameters
        self._parameters = RiskMetricsParameters(lambda_=lambda_)

    def _array_to_parameters(self, array: np.ndarray) -> RiskMetricsParameters:
        """Convert a parameter array to a parameter object.

        Args:
            array: Parameter array

        Returns:
            RiskMetricsParameters: Parameter object
        """
        return RiskMetricsParameters.from_array(array)

    def _parameters_to_array(self, parameters: RiskMetricsParameters) -> np.ndarray:
        """Convert a parameter object to a parameter array.

        Args:
            parameters: Parameter object

        Returns:
            np.ndarray: Parameter array
        """
        return parameters.to_array()

    def compute_covariance(
        self,
        parameters: RiskMetricsParameters,
        data: np.ndarray,
        sigma: Optional[np.ndarray] = None,
        backcast: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute conditional covariance matrices for the given parameters and data.

        Args:
            parameters: Model parameters
            data: Input data (typically residuals)
            sigma: Pre-allocated array for conditional covariances
            backcast: Matrix to use for initializing the covariance process

        Returns:
            np.ndarray: Conditional covariance matrices (n_assets x n_assets x T)
        """
        if self._n_assets is None:
            self._n_assets = data.shape[1]

        n_assets = self._n_assets
        T = data.shape[0]

        # Allocate covariance matrices if not provided
        if sigma is None:
            sigma = np.zeros((n_assets, n_assets, T))

        # Compute backcast if not provided
        if backcast is None:
            backcast = self.compute_backcast(data)

        # Extract lambda parameter
        lambda_ = parameters.lambda_

        # Use Numba-accelerated implementation if available
        if HAS_NUMBA:
            return _compute_riskmetrics_covariance_numba(data, lambda_, sigma, backcast)

        # Pure NumPy implementation
        # Initialize first covariance matrix with backcast
        sigma[:, :, 0] = backcast

        # Compute covariance matrices recursively
        for t in range(1, T):
            # RiskMetrics update: sigma_t = lambda * sigma_{t-1} + (1 - lambda) * r_{t-1} * r_{t-1}'
            outer_product = np.outer(data[t-1], data[t-1])
            sigma[:, :, t] = lambda_ * sigma[:, :, t-1] + (1 - lambda_) * outer_product

        return sigma

    def fit(
        self,
        data: np.ndarray,
        starting_values: Optional[Union[np.ndarray, RiskMetricsParameters]] = None,
        backcast: Optional[np.ndarray] = None,
        method: str = "SLSQP",
        options: Optional[Dict[str, Any]] = None,
        constraints: Optional[List[Dict[str, Any]]] = None,
        callback: Optional[ProgressCallback] = None,
        optimize_lambda: bool = False,
        **kwargs: Any
    ) -> MultivariateVolatilityResult:
        """Fit the RiskMetrics model to the provided data.

        Args:
            data: Residual data for model fitting (T x n_assets)
            starting_values: Initial parameter values for optimization
            backcast: Value to use for initializing the covariance process
            method: Optimization method to use
            options: Additional options for the optimizer
            constraints: Constraints for the optimizer
            callback: Callback function for reporting optimization progress
            optimize_lambda: Whether to optimize the decay factor (default: False)
            **kwargs: Additional keyword arguments for model fitting

        Returns:
            MultivariateVolatilityResult: The model estimation results

        Raises:
            ValueError: If the data is invalid
            RuntimeError: If the model fails to converge
        """
        # Validate data
        self.validate_data(data)
        T, n_assets = data.shape
        self._n_assets = n_assets
        self._residuals = data.copy()

        # Use provided starting values if available
        if starting_values is not None:
            if isinstance(starting_values, RiskMetricsParameters):
                self._parameters = starting_values
            else:
                self._parameters = self._array_to_parameters(starting_values)

        # Compute backcast if not provided
        if backcast is None:
            backcast = self.compute_backcast(data)

        # If optimize_lambda is True, estimate lambda via maximum likelihood
        if optimize_lambda:
            logger.info("Optimizing lambda parameter")
            if callback:
                callback(0.1, "Optimizing lambda parameter")

            # Define objective function (negative log-likelihood)
            def objective(params: np.ndarray) -> float:
                try:
                    param_obj = self._array_to_parameters(params)
                    return -self.loglikelihood(param_obj, data, backcast=backcast)
                except (ValueError, ParameterError, np.linalg.LinAlgError) as e:
                    logger.warning(f"Error in objective function: {str(e)}")
                    return 1e10

            # Set up optimization
            initial_params = self._parameters_to_array(self._parameters)
            bounds = [(0.01, 0.9999)]  # Bounds for lambda

            if constraints is None:
                constraints = []

            if options is None:
                options = {'disp': False}

            # Run optimization
            opt_result = optimize.minimize(
                objective,
                initial_params,
                method=method,
                bounds=bounds,
                constraints=constraints,
                options=options,
                callback=lambda x: callback(0.5, f"Optimizing lambda: {x[0]:.4f}") if callback else None
            )

            # Check convergence
            if not opt_result.success:
                warnings.warn(
                    f"Optimization did not converge: {opt_result.message}",
                    UserWarning
                )

            # Update parameters
            self._parameters = self._array_to_parameters(opt_result.x)
            iterations = opt_result.nit if hasattr(opt_result, 'nit') else 0
            convergence = opt_result.success
            optimization_message = opt_result.message
        else:
            # Use default or provided lambda
            logger.info(f"Using fixed lambda = {self._parameters.lambda_}")
            if callback:
                callback(0.1, f"Using fixed lambda = {self._parameters.lambda_}")

            iterations = 0
            convergence = True
            optimization_message = "Fixed lambda parameter"

        # Compute conditional covariance matrices
        logger.info("Computing conditional covariance matrices")
        if callback:
            callback(0.7, "Computing conditional covariance matrices")

        self._conditional_covariances = self.compute_covariance(
            self._parameters, data, backcast=backcast
        )

        # Compute conditional correlation matrices
        self._conditional_correlations = np.zeros_like(self._conditional_covariances)
        for t in range(T):
            self._conditional_correlations[:, :, t] = cov2corr(self._conditional_covariances[:, :, t])

        # Compute log-likelihood
        logger.info("Computing log-likelihood")
        if callback:
            callback(0.9, "Computing log-likelihood")

        loglik = self.loglikelihood(self._parameters, data, backcast=backcast)

        # Compute standard errors if lambda was optimized
        if optimize_lambda:
            std_errors = self._compute_std_errors(
                self._parameters_to_array(self._parameters),
                data,
                backcast=backcast
            )

            # Compute t-statistics and p-values
            param_array = self._parameters_to_array(self._parameters)
            t_stats = param_array / std_errors
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), T - len(param_array)))
        else:
            # No standard errors if lambda was not optimized
            std_errors = np.array([np.nan])
            t_stats = np.array([np.nan])
            p_values = np.array([np.nan])

        # Create result object
        self._fitted = True

        result = MultivariateVolatilityResult(
            model_name=self.name,
            parameters=self._parameters,
            convergence=convergence,
            iterations=iterations,
            log_likelihood=loglik,
            aic=-2 * loglik + 2 * (1 if optimize_lambda else 0),
            bic=-2 * loglik + (1 if optimize_lambda else 0) * np.log(T),
            hqic=-2 * loglik + 2 * (1 if optimize_lambda else 0) * np.log(np.log(T)),
            std_errors=std_errors,
            t_stats=t_stats,
            p_values=p_values,
            covariance_matrix=None,
            optimization_message=optimization_message,
            conditional_covariances=self._conditional_covariances,
            conditional_correlations=self._conditional_correlations,
            standardized_residuals=self.standardized_residuals(data),
            n_assets=n_assets,
            persistence=self._parameters.lambda_,
            half_life=np.log(0.5) / np.log(self._parameters.lambda_) if self._parameters.lambda_ < 1 else np.inf,
            unconditional_covariance=None,
            residuals=data
        )

        if callback:
            callback(1.0, "Estimation complete")

        return result

    async def fit_async(
        self,
        data: np.ndarray,
        starting_values: Optional[Union[np.ndarray, RiskMetricsParameters]] = None,
        backcast: Optional[np.ndarray] = None,
        method: str = "SLSQP",
        options: Optional[Dict[str, Any]] = None,
        constraints: Optional[List[Dict[str, Any]]] = None,
        callback: Optional[AsyncProgressCallback] = None,
        optimize_lambda: bool = False,
        **kwargs: Any
    ) -> MultivariateVolatilityResult:
        """Asynchronously fit the RiskMetrics model to the provided data.

        This method provides an asynchronous interface to the fit method,
        allowing for non-blocking model estimation in UI contexts.

        Args:
            data: Residual data for model fitting (T x n_assets)
            starting_values: Initial parameter values for optimization
            backcast: Value to use for initializing the covariance process
            method: Optimization method to use
            options: Additional options for the optimizer
            constraints: Constraints for the optimizer
            callback: Async callback function for reporting optimization progress
            optimize_lambda: Whether to optimize the decay factor (default: False)
            **kwargs: Additional keyword arguments for model fitting

        Returns:
            MultivariateVolatilityResult: The model estimation results

        Raises:
            ValueError: If the data is invalid
            RuntimeError: If the model fails to converge
        """
        # Create a synchronous callback that calls the async callback
        sync_callback = None
        if callback is not None:
            def sync_callback_wrapper(progress: float, message: str) -> None:
                asyncio.create_task(callback(progress, message))
            sync_callback = sync_callback_wrapper

        # Run the synchronous fit method in an executor to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.fit(
                data=data,
                starting_values=starting_values,
                backcast=backcast,
                method=method,
                options=options,
                constraints=constraints,
                callback=sync_callback,
                optimize_lambda=optimize_lambda,
                **kwargs
            )
        )

        return result

    def simulate(
        self,
        n_periods: int,
        burn: int = 0,
        initial_value: Optional[np.ndarray] = None,
        random_state: Optional[Union[int, np.random.Generator]] = None,
        return_covariances: bool = False,
        **kwargs: Any
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Simulate data from the RiskMetrics model.

        Args:
            n_periods: Number of periods to simulate
            burn: Number of initial observations to discard
            initial_value: Initial covariance matrix for the simulation
            random_state: Random number generator or seed
            return_covariances: Whether to return conditional covariances
            **kwargs: Additional keyword arguments for simulation

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: 
                Simulated data and optionally conditional covariances

        Raises:
            RuntimeError: If the model has not been fitted
            ValueError: If the simulation parameters are invalid
        """
        if not self._fitted or self._parameters is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        if self._n_assets is None:
            raise RuntimeError("Number of assets is not set")

        n_assets = self._n_assets

        # Set up random number generator
        if isinstance(random_state, np.random.Generator):
            rng = random_state
        else:
            rng = np.random.default_rng(random_state)

        # Total number of periods to simulate (including burn-in)
        total_periods = n_periods + burn

        # Initialize covariance matrix
        if initial_value is None:
            if self._conditional_covariances is not None:
                # Use the last estimated covariance matrix
                initial_cov = self._conditional_covariances[:, :, -1].copy()
            else:
                # Use identity matrix
                initial_cov = np.eye(n_assets)
        else:
            initial_cov = initial_value.copy()

        # Allocate arrays for simulated data and covariances
        simulated_data = np.zeros((total_periods, n_assets))
        covariances = np.zeros((n_assets, n_assets, total_periods))

        # Extract lambda parameter
        lambda_ = self._parameters.lambda_

        # Initialize first covariance matrix
        covariances[:, :, 0] = initial_cov

        # Generate first observation
        try:
            # Try Cholesky decomposition for numerical stability
            chol = np.linalg.cholesky(initial_cov)
            simulated_data[0, :] = rng.standard_normal(n_assets) @ chol.T
        except np.linalg.LinAlgError:
            # If Cholesky decomposition fails, use eigenvalue decomposition
            eigvals, eigvecs = np.linalg.eigh(initial_cov)
            # Ensure all eigenvalues are positive
            eigvals = np.maximum(eigvals, 1e-8)
            # Compute square root of covariance matrix
            sqrt_cov = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            simulated_data[0, :] = rng.standard_normal(n_assets) @ sqrt_cov

        # Simulate data
        for t in range(1, total_periods):
            # Update covariance matrix
            outer_product = np.outer(simulated_data[t-1], simulated_data[t-1])
            covariances[:, :, t] = lambda_ * covariances[:, :, t-1] + (1 - lambda_) * outer_product

            # Generate observation
            try:
                # Try Cholesky decomposition for numerical stability
                chol = np.linalg.cholesky(covariances[:, :, t])
                simulated_data[t, :] = rng.standard_normal(n_assets) @ chol.T
            except np.linalg.LinAlgError:
                # If Cholesky decomposition fails, use eigenvalue decomposition
                eigvals, eigvecs = np.linalg.eigh(covariances[:, :, t])
                # Ensure all eigenvalues are positive
                eigvals = np.maximum(eigvals, 1e-8)
                # Compute square root of covariance matrix
                sqrt_cov = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                simulated_data[t, :] = rng.standard_normal(n_assets) @ sqrt_cov

        # Discard burn-in periods
        if burn > 0:
            simulated_data = simulated_data[burn:]
            covariances = covariances[:, :, burn:]

        if return_covariances:
            return simulated_data, covariances
        else:
            return simulated_data

    def forecast(
        self,
        steps: int,
        initial_value: Optional[np.ndarray] = None,
        **kwargs: Any
    ) -> np.ndarray:
        """Forecast conditional covariance matrices.

        Args:
            steps: Number of steps to forecast
            initial_value: Initial covariance matrix for forecasting
            **kwargs: Additional keyword arguments for forecasting

        Returns:
            np.ndarray: Forecasted conditional covariance matrices (n_assets x n_assets x steps)

        Raises:
            RuntimeError: If the model has not been fitted
            ValueError: If the forecast parameters are invalid
        """
        if not self._fitted or self._parameters is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        if self._n_assets is None:
            raise RuntimeError("Number of assets is not set")

        n_assets = self._n_assets

        # Initialize covariance matrix
        if initial_value is None:
            if self._conditional_covariances is not None:
                # Use the last estimated covariance matrix
                initial_cov = self._conditional_covariances[:, :, -1].copy()
            else:
                # Use identity matrix
                initial_cov = np.eye(n_assets)
        else:
            initial_cov = initial_value.copy()

        # Allocate array for forecasted covariances
        forecasted_covariances = np.zeros((n_assets, n_assets, steps))

        # Extract lambda parameter
        lambda_ = self._parameters.lambda_

        # For RiskMetrics, the forecast is simply the last covariance matrix
        # repeated for all future periods, with exponential decay
        for h in range(steps):
            forecasted_covariances[:, :, h] = initial_cov

        return forecasted_covariances

    def _compute_persistence(self) -> Optional[float]:
        """Compute the persistence of the model.

        Returns:
            Optional[float]: Persistence value (lambda)
        """
        if not self._fitted or self._parameters is None:
            return None

        return self._parameters.lambda_

    def _compute_half_life(self) -> Optional[float]:
        """Compute the half-life of shocks in the model.

        Returns:
            Optional[float]: Half-life value
        """
        persistence = self._compute_persistence()
        if persistence is not None and persistence < 1:
            return np.log(0.5) / np.log(persistence)
        return None

    def __str__(self) -> str:
        """Return a string representation of the model.

        Returns:
            str: String representation
        """
        if not self._fitted:
            return f"{self.name} model (not fitted)"

        n_assets = self._n_assets if self._n_assets is not None else 0
        lambda_ = self._parameters.lambda_ if self._parameters is not None else 0.0

        return (
            f"{self.name} model with {n_assets} assets\n"
            f"Lambda: {lambda_:.4f}\n"
            f"Half-life: {self._compute_half_life():.2f} periods\n"
            f"Fitted: {self._fitted}"
        )

    def __repr__(self) -> str:
        """Return a string representation of the model.

        Returns:
            str: String representation
        """
        return self.__str__()


class RiskMetrics2006Model(CovarianceModelBase):
    """RiskMetrics2006 multivariate volatility model.

    This class implements the RiskMetrics2006 model, which uses multiple
    exponentially weighted moving averages with different decay factors
    to estimate covariance matrices.

    Attributes:
        name: Model name
        n_assets: Number of assets
        parameters: Model parameters if fitted
        _conditional_covariances: Conditional covariance matrices if fitted
        _component_covariances: Component covariance matrices if fitted
        _residuals: Residual data used for fitting
    """

    def __init__(
        self,
        n_assets: Optional[int] = None,
        decay_factors: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        name: str = "RiskMetrics2006"
    ):
        """Initialize the RiskMetrics2006 model.

        Args:
            n_assets: Number of assets (if None, determined from data)
            decay_factors: Array of decay factors (if None, uses default values)
            weights: Array of weights (if None, uses default values)
            name: Model name
        """
        super().__init__(name=name, n_assets=n_assets)

        # Initialize parameters with default values if not provided
        if decay_factors is None:
            # Default decay factors from RiskMetrics2006 methodology
            decay_factors = np.array([0.94, 0.96, 0.98])

        if weights is None:
            # Default weights from RiskMetrics2006 methodology
            # Equal weights for simplicity
            n_factors = len(decay_factors)
            weights = np.ones(n_factors) / n_factors

        # Initialize parameters
        self._parameters = RiskMetrics2006Parameters(
            decay_factors=decay_factors,
            weights=weights
        )

        self._component_covariances: Optional[np.ndarray] = None

    @property
    def component_covariances(self) -> Optional[np.ndarray]:
        """Get the component covariance matrices.

        Returns:
            Optional[np.ndarray]: Component covariance matrices if the model has been fitted,
                                 None otherwise

        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self._component_covariances

    def _array_to_parameters(self, array: np.ndarray) -> RiskMetrics2006Parameters:
        """Convert a parameter array to a parameter object.

        Args:
            array: Parameter array

        Returns:
            RiskMetrics2006Parameters: Parameter object
        """
        n_factors = len(self._parameters.decay_factors) if self._parameters is not None else None
        return RiskMetrics2006Parameters.from_array(array, n_factors=n_factors)

    def _parameters_to_array(self, parameters: RiskMetrics2006Parameters) -> np.ndarray:
        """Convert a parameter object to a parameter array.

        Args:
            parameters: Parameter object

        Returns:
            np.ndarray: Parameter array
        """
        return parameters.to_array()

    def compute_covariance(
        self,
        parameters: RiskMetrics2006Parameters,
        data: np.ndarray,
        sigma: Optional[np.ndarray] = None,
        backcast: Optional[np.ndarray] = None,
        return_components: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Compute conditional covariance matrices for the given parameters and data.

        Args:
            parameters: Model parameters
            data: Input data (typically residuals)
            sigma: Pre-allocated array for conditional covariances
            backcast: Matrix to use for initializing the covariance process
            return_components: Whether to return component covariance matrices

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: 
                Conditional covariance matrices and optionally component covariance matrices
        """
        if self._n_assets is None:
            self._n_assets = data.shape[1]

        n_assets = self._n_assets
        T = data.shape[0]
        n_factors = len(parameters.decay_factors)

        # Allocate covariance matrices if not provided
        if sigma is None:
            sigma = np.zeros((n_assets, n_assets, T))

        # Allocate component covariance matrices
        component_sigmas = np.zeros((n_factors, n_assets, n_assets, T))

        # Compute backcast if not provided
        if backcast is None:
            backcast = self.compute_backcast(data)

        # Extract parameters
        decay_factors = parameters.decay_factors
        weights = parameters.weights

        # Use Numba-accelerated implementation if available
        if HAS_NUMBA:
            sigma = _compute_riskmetrics2006_covariance_numba(
                data, decay_factors, weights, sigma, component_sigmas, backcast
            )
        else:
            # Pure NumPy implementation
            # Initialize first covariance matrix for each component with backcast
            for k in range(n_factors):
                component_sigmas[k, :, :, 0] = backcast

            # Compute component covariance matrices recursively
            for t in range(1, T):
                # Update each component
                for k in range(n_factors):
                    lambda_k = decay_factors[k]
                    # RiskMetrics update for each component
                    outer_product = np.outer(data[t-1], data[t-1])
                    component_sigmas[k, :, :, t] = (
                        lambda_k * component_sigmas[k, :, :, t-1] +
                        (1 - lambda_k) * outer_product
                    )

                # Combine components using weights
                sigma[:, :, t] = np.sum(
                    weights[:, np.newaxis, np.newaxis] * component_sigmas[:, :, :, t],
                    axis=0
                )

        if return_components:
            return sigma, component_sigmas
        else:
            return sigma

    def fit(
        self,
        data: np.ndarray,
        starting_values: Optional[Union[np.ndarray, RiskMetrics2006Parameters]] = None,
        backcast: Optional[np.ndarray] = None,
        method: str = "SLSQP",
        options: Optional[Dict[str, Any]] = None,
        constraints: Optional[List[Dict[str, Any]]] = None,
        callback: Optional[ProgressCallback] = None,
        optimize_parameters: bool = False,
        **kwargs: Any
    ) -> MultivariateVolatilityResult:
        """Fit the RiskMetrics2006 model to the provided data.

        Args:
            data: Residual data for model fitting (T x n_assets)
            starting_values: Initial parameter values for optimization
            backcast: Value to use for initializing the covariance process
            method: Optimization method to use
            options: Additional options for the optimizer
            constraints: Constraints for the optimizer
            callback: Callback function for reporting optimization progress
            optimize_parameters: Whether to optimize the decay factors and weights (default: False)
            **kwargs: Additional keyword arguments for model fitting

        Returns:
            MultivariateVolatilityResult: The model estimation results

        Raises:
            ValueError: If the data is invalid
            RuntimeError: If the model fails to converge
        """
        # Validate data
        self.validate_data(data)
        T, n_assets = data.shape
        self._n_assets = n_assets
        self._residuals = data.copy()

        # Use provided starting values if available
        if starting_values is not None:
            if isinstance(starting_values, RiskMetrics2006Parameters):
                self._parameters = starting_values
            else:
                self._parameters = self._array_to_parameters(starting_values)

        # Compute backcast if not provided
        if backcast is None:
            backcast = self.compute_backcast(data)

        # If optimize_parameters is True, estimate decay factors and weights via maximum likelihood
        if optimize_parameters:
            logger.info("Optimizing decay factors and weights")
            if callback:
                callback(0.1, "Optimizing decay factors and weights")

            # Define objective function (negative log-likelihood)
            def objective(params: np.ndarray) -> float:
                try:
                    param_obj = self._array_to_parameters(params)
                    return -self.loglikelihood(param_obj, data, backcast=backcast)
                except (ValueError, ParameterError, np.linalg.LinAlgError) as e:
                    logger.warning(f"Error in objective function: {str(e)}")
                    return 1e10

            # Set up optimization
            initial_params = self._parameters_to_array(self._parameters)
            n_factors = len(self._parameters.decay_factors)

            # Bounds for decay factors and weights
            bounds = [(0.01, 0.9999)] * n_factors  # Bounds for decay factors
            bounds.extend([(0.0, 1.0)] * (n_factors - 1))  # Bounds for weights (except last one)

            # Constraint: weights sum to 1 (implicitly handled by parameterization)
            if constraints is None:
                constraints = []

            if options is None:
                options = {'disp': False}

            # Run optimization
            opt_result = optimize.minimize(
                objective,
                initial_params,
                method=method,
                bounds=bounds,
                constraints=constraints,
                options=options,
                callback=lambda x: callback(
                    0.5, f"Optimizing parameters: iteration {opt_result.nit if hasattr(opt_result, 'nit') else 0}") if callback else None
            )

            # Check convergence
            if not opt_result.success:
                warnings.warn(
                    f"Optimization did not converge: {opt_result.message}",
                    UserWarning
                )

            # Update parameters
            self._parameters = self._array_to_parameters(opt_result.x)
            iterations = opt_result.nit if hasattr(opt_result, 'nit') else 0
            convergence = opt_result.success
            optimization_message = opt_result.message
        else:
            # Use default or provided parameters
            logger.info(
                f"Using fixed parameters: decay_factors={self._parameters.decay_factors}, weights={self._parameters.weights}")
            if callback:
                callback(0.1, "Using fixed parameters")

            iterations = 0
            convergence = True
            optimization_message = "Fixed parameters"

        # Compute conditional covariance matrices
        logger.info("Computing conditional covariance matrices")
        if callback:
            callback(0.7, "Computing conditional covariance matrices")

        result = self.compute_covariance(
            self._parameters, data, backcast=backcast, return_components=True
        )

        if isinstance(result, tuple):
            self._conditional_covariances, self._component_covariances = result
        else:
            self._conditional_covariances = result
            self._component_covariances = None

        # Compute conditional correlation matrices
        self._conditional_correlations = np.zeros_like(self._conditional_covariances)
        for t in range(T):
            self._conditional_correlations[:, :, t] = cov2corr(self._conditional_covariances[:, :, t])

        # Compute log-likelihood
        logger.info("Computing log-likelihood")
        if callback:
            callback(0.9, "Computing log-likelihood")

        loglik = self.loglikelihood(self._parameters, data, backcast=backcast)

        # Compute standard errors if parameters were optimized
        if optimize_parameters:
            std_errors = self._compute_std_errors(
                self._parameters_to_array(self._parameters),
                data,
                backcast=backcast
            )

            # Compute t-statistics and p-values
            param_array = self._parameters_to_array(self._parameters)
            t_stats = param_array / std_errors
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), T - len(param_array)))
        else:
            # No standard errors if parameters were not optimized
            n_params = len(self._parameters_to_array(self._parameters))
            std_errors = np.full(n_params, np.nan)
            t_stats = np.full(n_params, np.nan)
            p_values = np.full(n_params, np.nan)

        # Create result object
        self._fitted = True

        # Compute persistence as weighted average of decay factors
        persistence = np.sum(self._parameters.weights * self._parameters.decay_factors)

        # Compute half-life
        half_life = np.log(0.5) / np.log(persistence) if persistence < 1 else np.inf

        result = MultivariateVolatilityResult(
            model_name=self.name,
            parameters=self._parameters,
            convergence=convergence,
            iterations=iterations,
            log_likelihood=loglik,
            aic=-2 * loglik + 2 * (n_params if optimize_parameters else 0),
            bic=-2 * loglik + (n_params if optimize_parameters else 0) * np.log(T),
            hqic=-2 * loglik + 2 * (n_params if optimize_parameters else 0) * np.log(np.log(T)),
            std_errors=std_errors,
            t_stats=t_stats,
            p_values=p_values,
            covariance_matrix=None,
            optimization_message=optimization_message,
            conditional_covariances=self._conditional_covariances,
            conditional_correlations=self._conditional_correlations,
            standardized_residuals=self.standardized_residuals(data),
            n_assets=n_assets,
            persistence=persistence,
            half_life=half_life,
            unconditional_covariance=None,
            residuals=data
        )

        if callback:
            callback(1.0, "Estimation complete")

        return result

    async def fit_async(
        self,
        data: np.ndarray,
        starting_values: Optional[Union[np.ndarray, RiskMetrics2006Parameters]] = None,
        backcast: Optional[np.ndarray] = None,
        method: str = "SLSQP",
        options: Optional[Dict[str, Any]] = None,
        constraints: Optional[List[Dict[str, Any]]] = None,
        callback: Optional[AsyncProgressCallback] = None,
        optimize_parameters: bool = False,
        **kwargs: Any
    ) -> MultivariateVolatilityResult:
        """Asynchronously fit the RiskMetrics2006 model to the provided data.

        This method provides an asynchronous interface to the fit method,
        allowing for non-blocking model estimation in UI contexts.

        Args:
            data: Residual data for model fitting (T x n_assets)
            starting_values: Initial parameter values for optimization
            backcast: Value to use for initializing the covariance process
            method: Optimization method to use
            options: Additional options for the optimizer
            constraints: Constraints for the optimizer
            callback: Async callback function for reporting optimization progress
            optimize_parameters: Whether to optimize the decay factors and weights (default: False)
            **kwargs: Additional keyword arguments for model fitting

        Returns:
            MultivariateVolatilityResult: The model estimation results

        Raises:
            ValueError: If the data is invalid
            RuntimeError: If the model fails to converge
        """
        # Create a synchronous callback that calls the async callback
        sync_callback = None
        if callback is not None:
            def sync_callback_wrapper(progress: float, message: str) -> None:
                asyncio.create_task(callback(progress, message))
            sync_callback = sync_callback_wrapper

        # Run the synchronous fit method in an executor to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.fit(
                data=data,
                starting_values=starting_values,
                backcast=backcast,
                method=method,
                options=options,
                constraints=constraints,
                callback=sync_callback,
                optimize_parameters=optimize_parameters,
                **kwargs
            )
        )

        return result

    def simulate(
        self,
        n_periods: int,
        burn: int = 0,
        initial_value: Optional[np.ndarray] = None,
        initial_components: Optional[np.ndarray] = None,
        random_state: Optional[Union[int, np.random.Generator]] = None,
        return_covariances: bool = False,
        **kwargs: Any
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Simulate data from the RiskMetrics2006 model.

        Args:
            n_periods: Number of periods to simulate
            burn: Number of initial observations to discard
            initial_value: Initial covariance matrix for the simulation
            initial_components: Initial component covariance matrices
            random_state: Random number generator or seed
            return_covariances: Whether to return conditional covariances
            **kwargs: Additional keyword arguments for simulation

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: 
                Simulated data and optionally conditional covariances

        Raises:
            RuntimeError: If the model has not been fitted
            ValueError: If the simulation parameters are invalid
        """
        if not self._fitted or self._parameters is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        if self._n_assets is None:
            raise RuntimeError("Number of assets is not set")

        n_assets = self._n_assets
        n_factors = len(self._parameters.decay_factors)

        # Set up random number generator
        if isinstance(random_state, np.random.Generator):
            rng = random_state
        else:
            rng = np.random.default_rng(random_state)

        # Total number of periods to simulate (including burn-in)
        total_periods = n_periods + burn

        # Initialize covariance matrix
        if initial_value is None:
            if self._conditional_covariances is not None:
                # Use the last estimated covariance matrix
                initial_cov = self._conditional_covariances[:, :, -1].copy()
            else:
                # Use identity matrix
                initial_cov = np.eye(n_assets)
        else:
            initial_cov = initial_value.copy()

        # Initialize component covariance matrices
        if initial_components is None:
            if self._component_covariances is not None:
                # Use the last estimated component covariance matrices
                initial_components = self._component_covariances[:, :, :, -1].copy()
            else:
                # Use the initial covariance matrix for all components
                initial_components = np.zeros((n_factors, n_assets, n_assets))
                for k in range(n_factors):
                    initial_components[k] = initial_cov.copy()

        # Allocate arrays for simulated data and covariances
        simulated_data = np.zeros((total_periods, n_assets))
        covariances = np.zeros((n_assets, n_assets, total_periods))
        component_covs = np.zeros((n_factors, n_assets, n_assets, total_periods))

        # Extract parameters
        decay_factors = self._parameters.decay_factors
        weights = self._parameters.weights

        # Initialize first covariance matrix and components
        covariances[:, :, 0] = initial_cov
        for k in range(n_factors):
            component_covs[k, :, :, 0] = initial_components[k]

        # Generate first observation
        try:
            # Try Cholesky decomposition for numerical stability
            chol = np.linalg.cholesky(initial_cov)
            simulated_data[0, :] = rng.standard_normal(n_assets) @ chol.T
        except np.linalg.LinAlgError:
            # If Cholesky decomposition fails, use eigenvalue decomposition
            eigvals, eigvecs = np.linalg.eigh(initial_cov)
            # Ensure all eigenvalues are positive
            eigvals = np.maximum(eigvals, 1e-8)
            # Compute square root of covariance matrix
            sqrt_cov = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            simulated_data[0, :] = rng.standard_normal(n_assets) @ sqrt_cov

        # Simulate data
        for t in range(1, total_periods):
            # Update component covariance matrices
            outer_product = np.outer(simulated_data[t-1], simulated_data[t-1])
            for k in range(n_factors):
                lambda_k = decay_factors[k]
                component_covs[k, :, :, t] = (
                    lambda_k * component_covs[k, :, :, t-1] +
                    (1 - lambda_k) * outer_product
                )

            # Combine components using weights
            covariances[:, :, t] = np.sum(
                weights[:, np.newaxis, np.newaxis] * component_covs[:, :, :, t],
                axis=0
            )

            # Generate observation
            try:
                # Try Cholesky decomposition for numerical stability
                chol = np.linalg.cholesky(covariances[:, :, t])
                simulated_data[t, :] = rng.standard_normal(n_assets) @ chol.T
            except np.linalg.LinAlgError:
                # If Cholesky decomposition fails, use eigenvalue decomposition
                eigvals, eigvecs = np.linalg.eigh(covariances[:, :, t])
                # Ensure all eigenvalues are positive
                eigvals = np.maximum(eigvals, 1e-8)
                # Compute square root of covariance matrix
                sqrt_cov = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                simulated_data[t, :] = rng.standard_normal(n_assets) @ sqrt_cov

        # Discard burn-in periods
        if burn > 0:
            simulated_data = simulated_data[burn:]
            covariances = covariances[:, :, burn:]

        if return_covariances:
            return simulated_data, covariances
        else:
            return simulated_data

    def forecast(
        self,
        steps: int,
        initial_value: Optional[np.ndarray] = None,
        initial_components: Optional[np.ndarray] = None,
        **kwargs: Any
    ) -> np.ndarray:
        """Forecast conditional covariance matrices.

        Args:
            steps: Number of steps to forecast
            initial_value: Initial covariance matrix for forecasting
            initial_components: Initial component covariance matrices
            **kwargs: Additional keyword arguments for forecasting

        Returns:
            np.ndarray: Forecasted conditional covariance matrices (n_assets x n_assets x steps)

        Raises:
            RuntimeError: If the model has not been fitted
            ValueError: If the forecast parameters are invalid
        """
        if not self._fitted or self._parameters is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        if self._n_assets is None:
            raise RuntimeError("Number of assets is not set")

        n_assets = self._n_assets
        n_factors = len(self._parameters.decay_factors)

        # Initialize covariance matrix
        if initial_value is None:
            if self._conditional_covariances is not None:
                # Use the last estimated covariance matrix
                initial_cov = self._conditional_covariances[:, :, -1].copy()
            else:
                # Use identity matrix
                initial_cov = np.eye(n_assets)
        else:
            initial_cov = initial_value.copy()

        # Initialize component covariance matrices
        if initial_components is None:
            if self._component_covariances is not None:
                # Use the last estimated component covariance matrices
                initial_components = self._component_covariances[:, :, :, -1].copy()
            else:
                # Use the initial covariance matrix for all components
                initial_components = np.zeros((n_factors, n_assets, n_assets))
                for k in range(n_factors):
                    initial_components[k] = initial_cov.copy()

        # Allocate array for forecasted covariances
        forecasted_covariances = np.zeros((n_assets, n_assets, steps))

        # Extract parameters
        decay_factors = self._parameters.decay_factors
        weights = self._parameters.weights

        # For RiskMetrics2006, the forecast is simply the weighted average of component matrices
        # repeated for all future periods
        for h in range(steps):
            forecasted_covariances[:, :, h] = np.sum(
                weights[:, np.newaxis, np.newaxis] * initial_components,
                axis=0
            )

        return forecasted_covariances

    def _compute_persistence(self) -> Optional[float]:
        """Compute the persistence of the model.

        Returns:
            Optional[float]: Persistence value (weighted average of decay factors)
        """
        if not self._fitted or self._parameters is None:
            return None

        # Compute weighted average of decay factors
        return np.sum(self._parameters.weights * self._parameters.decay_factors)

    def _compute_half_life(self) -> Optional[float]:
        """Compute the half-life of shocks in the model.

        Returns:
            Optional[float]: Half-life value
        """
        persistence = self._compute_persistence()
        if persistence is not None and persistence < 1:
            return np.log(0.5) / np.log(persistence)
        return None

    def __str__(self) -> str:
        """Return a string representation of the model.

        Returns:
            str: String representation
        """
        if not self._fitted:
            return f"{self.name} model (not fitted)"

        n_assets = self._n_assets if self._n_assets is not None else 0
        n_factors = len(self._parameters.decay_factors) if self._parameters is not None else 0

        decay_factors_str = ", ".join(f"{x:.4f}" for x in self._parameters.decay_factors)
        weights_str = ", ".join(f"{x:.4f}" for x in self._parameters.weights)

        return (
            f"{self.name} model with {n_assets} assets\n"
            f"Number of factors: {n_factors}\n"
            f"Decay factors: [{decay_factors_str}]\n"
            f"Weights: [{weights_str}]\n"
            f"Persistence: {self._compute_persistence():.4f}\n"
            f"Half-life: {self._compute_half_life():.2f} periods\n"
            f"Fitted: {self._fitted}"
        )

    def __repr__(self) -> str:
        """Return a string representation of the model.

        Returns:
            str: String representation
        """
        return self.__str__()


# Register Numba-accelerated functions if available

def _register_numba_functions() -> None:
    """
    Register Numba JIT-compiled functions for RiskMetrics models.

    This function is called during module initialization to register
    performance-critical functions for JIT compilation if Numba is available.
    """
    if HAS_NUMBA:
        # The functions are already decorated with @jit, so no additional
        # registration is needed here. This function is kept for consistency
        # with the module structure and potential future enhancements.
        logger.debug("RiskMetrics model Numba JIT functions registered")
    else:
        logger.info("Numba not available. RiskMetrics models will use pure NumPy implementations.")


# Initialize the module
_register_numba_functions()
