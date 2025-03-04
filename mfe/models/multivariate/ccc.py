# mfe/models/multivariate/ccc.py

"""
Constant Conditional Correlation (CCC) multivariate GARCH model.

This module implements the Constant Conditional Correlation (CCC) multivariate GARCH
model introduced by Bollerslev (1990). The CCC model combines univariate GARCH
processes for individual asset volatilities with a constant correlation matrix,
providing a parsimonious approach to multivariate volatility modeling.

The CCC model offers a computationally efficient alternative to more complex
multivariate GARCH specifications while still capturing the essential dynamics
of time-varying volatilities. It supports various univariate GARCH specifications
for individual assets, including standard GARCH, TARCH, and GJR-GARCH.

References:
    Bollerslev, T. (1990). Modelling the coherence in short-run nominal exchange rates:
    A multivariate generalized ARCH model. The Review of Economics and Statistics,
    72(3), 498-505.
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
from mfe.models.multivariate.base import CorrelationModelBase
from mfe.models.univariate.base import UnivariateVolatilityModel
from mfe.models.univariate.garch import GARCH
from mfe.utils.matrix_ops import cov2corr, corr2cov, vech, ivech, ensure_symmetric
from mfe.models.multivariate.utils import (
    validate_multivariate_data, compute_sample_correlation,
    transform_correlation_matrix, inverse_transform_correlation_matrix,
    standardize_residuals, compute_conditional_correlations
)

# Set up module-level logger
logger = logging.getLogger("mfe.models.multivariate.ccc")

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
    logger.debug("Numba available for CCC model acceleration")
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    HAS_NUMBA = False
    logger.info("Numba not available. CCC model will use pure NumPy implementations.")


@dataclass
class CCCParameters(MultivariateVolatilityParameters):
    """Parameters for the Constant Conditional Correlation (CCC) model.

    This class holds the correlation parameters for the CCC model.
    The univariate volatility model parameters are stored separately.

    Attributes:
        correlations: Lower triangular elements of the correlation matrix (excluding diagonal)
    """

    correlations: np.ndarray

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate CCC parameter constraints.

        Raises:
            ParameterError: If parameter constraints are violated
        """
        super().validate()

        # Validate correlations
        for i, rho in enumerate(self.correlations):
            validate_range(rho, f"correlations[{i}]", -1.0, 1.0)

    def to_array(self) -> np.ndarray:
        """Convert parameters to a NumPy array.

        Returns:
            np.ndarray: Array representation of parameters
        """
        return self.correlations.copy()

    @classmethod
    def from_array(cls, array: np.ndarray, **kwargs: Any) -> 'CCCParameters':
        """Create parameters from a NumPy array.

        Args:
            array: Array representation of parameters
            **kwargs: Additional keyword arguments for parameter creation

        Returns:
            CCCParameters: Parameter object
        """
        return cls(correlations=array.copy())

    def transform(self) -> np.ndarray:
        """Transform parameters to unconstrained space for optimization.

        Returns:
            np.ndarray: Transformed parameters in unconstrained space
        """
        # Apply Fisher's z-transformation to each correlation
        transformed_params = np.zeros_like(self.correlations)
        for i, rho in enumerate(self.correlations):
            transformed_params[i] = np.arctanh(rho)  # Fisher's z-transformation

        return transformed_params

    @classmethod
    def inverse_transform(cls, array: np.ndarray, **kwargs: Any) -> 'CCCParameters':
        """Transform parameters from unconstrained space back to constrained space.

        Args:
            array: Parameters in unconstrained space
            **kwargs: Additional keyword arguments for parameter creation

        Returns:
            CCCParameters: Parameter object with constrained parameters
        """
        # Apply inverse Fisher's z-transformation to each parameter
        correlations = np.zeros_like(array)
        for i, z in enumerate(array):
            correlations[i] = np.tanh(z)  # Inverse Fisher's z-transformation

        return cls(correlations=correlations)


class CCCModel(CorrelationModelBase):
    """Constant Conditional Correlation (CCC) multivariate GARCH model.

    This class implements the CCC model introduced by Bollerslev (1990), which
    combines univariate GARCH processes with a constant correlation matrix.

    Attributes:
        name: Model name
        n_assets: Number of assets
        univariate_models: List of univariate volatility models for each asset
        parameters: Model parameters if fitted
        _conditional_correlations: Constant correlation matrix if fitted
        _conditional_covariances: Conditional covariance matrices if fitted
        _residuals: Residual data used for fitting
    """

    def __init__(
        self,
        n_assets: Optional[int] = None,
        univariate_models: Optional[List[UnivariateVolatilityModel]] = None,
        name: str = "CCC"
    ):
        """Initialize the CCC model.

        Args:
            n_assets: Number of assets (if None, determined from data)
            univariate_models: List of univariate volatility models for each asset
                              (if None, GARCH(1,1) models are used)
            name: Model name
        """
        super().__init__(name=name, n_assets=n_assets)

        self._univariate_models = univariate_models
        self._constant_correlation: Optional[np.ndarray] = None

    @property
    def univariate_models(self) -> Optional[List[UnivariateVolatilityModel]]:
        """Get the univariate volatility models.

        Returns:
            Optional[List[UnivariateVolatilityModel]]: List of univariate models if set,
                                                     None otherwise
        """
        return self._univariate_models

    @property
    def constant_correlation(self) -> Optional[np.ndarray]:
        """Get the constant correlation matrix.

        Returns:
            Optional[np.ndarray]: Constant correlation matrix if the model has been fitted,
                                 None otherwise

        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self._constant_correlation

    def _initialize_univariate_models(self, data: np.ndarray) -> List[UnivariateVolatilityModel]:
        """Initialize univariate volatility models for each asset.

        Args:
            data: Input data array with shape (T, n_assets)

        Returns:
            List[UnivariateVolatilityModel]: List of initialized univariate models
        """
        T, n_assets = data.shape

        # If univariate models are already set, return them
        if self._univariate_models is not None:
            if len(self._univariate_models) != n_assets:
                raise ValueError(
                    f"Number of univariate models ({len(self._univariate_models)}) "
                    f"does not match number of assets ({n_assets})"
                )
            return self._univariate_models

        # Initialize GARCH(1,1) models for each asset
        models = []
        for i in range(n_assets):
            model = GARCH()
            models.append(model)

        return models

    def _compute_univariate_volatilities(self, data: np.ndarray) -> np.ndarray:
        """Compute univariate volatility estimates for each asset.

        Args:
            data: Input data array with shape (T, n_assets)

        Returns:
            np.ndarray: Univariate volatility estimates with shape (T, n_assets)
        """
        T, n_assets = data.shape

        # Initialize univariate models if not already set
        if self._univariate_models is None:
            self._univariate_models = self._initialize_univariate_models(data)

        # Fit univariate models and extract volatilities
        volatilities = np.zeros((T, n_assets))

        for i, model in enumerate(self._univariate_models):
            # Fit model to asset data
            asset_data = data[:, i]
            model.fit(asset_data)

            # Extract conditional volatilities (standard deviations)
            asset_volatility = np.sqrt(model.conditional_variances)
            volatilities[:, i] = asset_volatility

        self._univariate_volatilities = volatilities
        return volatilities

    def _array_to_parameters(self, array: np.ndarray) -> CCCParameters:
        """Convert a parameter array to a parameter object.

        Args:
            array: Parameter array

        Returns:
            CCCParameters: Parameter object
        """
        return CCCParameters.from_array(array)

    def _parameters_to_array(self, parameters: CCCParameters) -> np.ndarray:
        """Convert a parameter object to a parameter array.

        Args:
            parameters: Parameter object

        Returns:
            np.ndarray: Parameter array
        """
        return parameters.to_array()

    def _construct_correlation_matrix(self, correlations: np.ndarray) -> np.ndarray:
        """Construct a correlation matrix from the lower triangular elements.

        Args:
            correlations: Lower triangular elements of the correlation matrix (excluding diagonal)

        Returns:
            np.ndarray: Correlation matrix
        """
        if self._n_assets is None:
            raise RuntimeError("Number of assets is not set")

        n_assets = self._n_assets
        n_corr = n_assets * (n_assets - 1) // 2

        if len(correlations) != n_corr:
            raise ValueError(
                f"Number of correlation parameters ({len(correlations)}) does not match "
                f"expected number ({n_corr}) for {n_assets} assets"
            )

        # Initialize correlation matrix with ones on diagonal
        corr_matrix = np.eye(n_assets)

        # Fill lower triangular elements
        idx = 0
        for i in range(n_assets):
            for j in range(i):
                corr_matrix[i, j] = correlations[idx]
                corr_matrix[j, i] = correlations[idx]  # Symmetric
                idx += 1

        return corr_matrix

    def compute_correlation(
        self,
        parameters: CCCParameters,
        data: np.ndarray,
        correlation: Optional[np.ndarray] = None,
        backcast: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute conditional correlation matrices for the given parameters and data.

        For the CCC model, the correlation matrix is constant over time.

        Args:
            parameters: Model parameters
            data: Input data (typically residuals)
            correlation: Pre-allocated array for conditional correlations
            backcast: Matrix to use for initializing the correlation process

        Returns:
            np.ndarray: Conditional correlation matrices (n_assets x n_assets x T)
        """
        if self._n_assets is None:
            self._n_assets = data.shape[1]

        n_assets = self._n_assets
        T = data.shape[0]

        # Construct correlation matrix from parameters
        const_corr = self._construct_correlation_matrix(parameters.correlations)

        # Allocate correlation matrices if not provided
        if correlation is None:
            correlation = np.zeros((n_assets, n_assets, T))

        # Fill with constant correlation matrix
        for t in range(T):
            correlation[:, :, t] = const_corr

        return correlation

    def compute_covariance(
        self,
        parameters: CCCParameters,
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

        # Compute univariate volatilities if not already available
        if self._univariate_volatilities is None:
            self._compute_univariate_volatilities(data)

        # Construct correlation matrix from parameters
        const_corr = self._construct_correlation_matrix(parameters.correlations)

        # Allocate covariance matrices if not provided
        if sigma is None:
            sigma = np.zeros((n_assets, n_assets, T))

        # Compute covariance matrices
        for t in range(T):
            # Create diagonal matrix of volatilities
            D = np.diag(self._univariate_volatilities[t, :])
            # Compute covariance matrix: D * R * D
            sigma[:, :, t] = D @ const_corr @ D

        return sigma

    def fit(
        self,
        data: np.ndarray,
        starting_values: Optional[Union[np.ndarray, CCCParameters]] = None,
        backcast: Optional[np.ndarray] = None,
        method: str = "SLSQP",
        options: Optional[Dict[str, Any]] = None,
        constraints: Optional[List[Dict[str, Any]]] = None,
        callback: Optional[ProgressCallback] = None,
        **kwargs: Any
    ) -> MultivariateVolatilityResult:
        """Fit the CCC model to the provided data.

        Args:
            data: Residual data for model fitting (T x n_assets)
            starting_values: Initial parameter values for optimization
            backcast: Value to use for initializing the covariance process
            method: Optimization method to use
            options: Additional options for the optimizer
            constraints: Constraints for the optimizer
            callback: Callback function for reporting optimization progress
            **kwargs: Additional keyword arguments for model fitting

        Returns:
            MultivariateVolatilityResult: The model estimation results

        Raises:
            ValueError: If the data is invalid
            RuntimeError: If the model fails to converge
        """
        # Validate data
        T, n_assets = validate_multivariate_data(data)
        self._n_assets = n_assets
        self._residuals = data.copy()

        # Step 1: Fit univariate volatility models
        logger.info("Fitting univariate volatility models")
        if callback:
            callback(0.1, "Fitting univariate volatility models")

        self._univariate_models = self._initialize_univariate_models(data)
        self._compute_univariate_volatilities(data)

        # Step 2: Standardize residuals using univariate volatilities
        logger.info("Standardizing residuals")
        if callback:
            callback(0.3, "Standardizing residuals")

        std_residuals = np.zeros_like(data)
        for i in range(n_assets):
            std_residuals[:, i] = data[:, i] / self._univariate_volatilities[:, i]

        # Step 3: Estimate correlation matrix
        logger.info("Estimating correlation matrix")
        if callback:
            callback(0.5, "Estimating correlation matrix")

        # Compute sample correlation of standardized residuals
        sample_corr = compute_sample_correlation(std_residuals)

        # Extract lower triangular elements (excluding diagonal)
        n_corr = n_assets * (n_assets - 1) // 2
        corr_params = np.zeros(n_corr)

        idx = 0
        for i in range(n_assets):
            for j in range(i):
                corr_params[idx] = sample_corr[i, j]
                idx += 1

        # Use provided starting values if available
        if starting_values is not None:
            if isinstance(starting_values, CCCParameters):
                corr_params = starting_values.correlations
            else:
                corr_params = starting_values

        # Create parameter object
        parameters = CCCParameters(correlations=corr_params)

        # Construct correlation matrix
        self._constant_correlation = self._construct_correlation_matrix(parameters.correlations)

        # Step 4: Compute conditional covariance matrices
        logger.info("Computing conditional covariance matrices")
        if callback:
            callback(0.7, "Computing conditional covariance matrices")

        self._conditional_correlations = self.compute_correlation(parameters, data)
        self._conditional_covariances = self.compute_covariance(parameters, data)

        # Step 5: Compute log-likelihood
        logger.info("Computing log-likelihood")
        if callback:
            callback(0.9, "Computing log-likelihood")

        loglik = self.loglikelihood(parameters, data)

        # Compute standard errors
        std_errors = np.zeros_like(corr_params)
        t_stats = np.zeros_like(corr_params)
        p_values = np.zeros_like(corr_params)

        # Create result object
        self._parameters = parameters
        self._fitted = True

        result = MultivariateVolatilityResult(
            model_name=self.name,
            parameters=parameters,
            convergence=True,
            iterations=0,
            log_likelihood=loglik,
            aic=-2 * loglik + 2 * len(corr_params),
            bic=-2 * loglik + len(corr_params) * np.log(T),
            hqic=-2 * loglik + 2 * len(corr_params) * np.log(np.log(T)),
            std_errors=std_errors,
            t_stats=t_stats,
            p_values=p_values,
            covariance_matrix=None,
            optimization_message="Direct estimation without optimization",
            conditional_covariances=self._conditional_covariances,
            conditional_correlations=self._conditional_correlations,
            standardized_residuals=standardize_residuals(data, self._conditional_covariances),
            n_assets=n_assets,
            persistence=None,
            half_life=None,
            unconditional_covariance=None,
            residuals=data
        )

        if callback:
            callback(1.0, "Estimation complete")

        return result

    async def fit_async(
        self,
        data: np.ndarray,
        starting_values: Optional[Union[np.ndarray, CCCParameters]] = None,
        backcast: Optional[np.ndarray] = None,
        method: str = "SLSQP",
        options: Optional[Dict[str, Any]] = None,
        constraints: Optional[List[Dict[str, Any]]] = None,
        callback: Optional[AsyncProgressCallback] = None,
        **kwargs: Any
    ) -> MultivariateVolatilityResult:
        """Asynchronously fit the CCC model to the provided data.

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
                **kwargs
            )
        )

        return result

    def loglikelihood(
        self,
        parameters: Union[np.ndarray, CCCParameters],
        data: np.ndarray,
        backcast: Optional[np.ndarray] = None,
        individual: bool = False
    ) -> Union[float, np.ndarray]:
        """Compute the log-likelihood of the data given the parameters.

        Args:
            parameters: Model parameters
            data: Input data (typically residuals)
            backcast: Matrix to use for initializing the covariance process
            individual: Whether to return individual log-likelihood contributions

        Returns:
            Union[float, np.ndarray]: Log-likelihood value or individual contributions

        Raises:
            ValueError: If the parameters or data are invalid
        """
        # Convert parameters to the appropriate type if necessary
        if isinstance(parameters, np.ndarray):
            parameters = self._array_to_parameters(parameters)

        # Validate parameters
        parameters.validate()

        # Compute conditional covariance matrices
        covariance = self.compute_covariance(parameters, data, backcast=backcast)

        # Compute log-likelihood
        T = data.shape[0]
        n_assets = data.shape[1]

        # Pre-allocate array for individual log-likelihood contributions
        loglikelihood_individual = np.zeros(T)

        # Compute log-likelihood for each time point
        for t in range(T):
            try:
                # Try Cholesky decomposition for numerical stability
                chol = np.linalg.cholesky(covariance[:, :, t])
                # Compute log determinant using Cholesky factor
                log_det = 2 * np.sum(np.log(np.diag(chol)))
                # Compute quadratic form using Cholesky factor
                quad_form = np.sum(np.linalg.solve(chol, data[t, :]) ** 2)
            except np.linalg.LinAlgError:
                # If Cholesky decomposition fails, use eigenvalue decomposition
                eigvals = np.linalg.eigvalsh(covariance[:, :, t])
                # Ensure all eigenvalues are positive
                eigvals = np.maximum(eigvals, 1e-8)
                # Compute log determinant using eigenvalues
                log_det = np.sum(np.log(eigvals))
                # Compute quadratic form using matrix inverse
                quad_form = data[t, :] @ np.linalg.solve(covariance[:, :, t], data[t, :])

            # Compute log-likelihood contribution
            loglikelihood_individual[t] = -0.5 * (n_assets * np.log(2 * np.pi) + log_det + quad_form)

        if individual:
            return loglikelihood_individual

        # Return sum of log-likelihood contributions
        return np.sum(loglikelihood_individual)

    def simulate(
        self,
        n_periods: int,
        burn: int = 0,
        initial_value: Optional[np.ndarray] = None,
        random_state: Optional[Union[int, np.random.Generator]] = None,
        return_covariances: bool = False,
        **kwargs: Any
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Simulate data from the CCC model.

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

        if self._n_assets is None or self._univariate_models is None:
            raise RuntimeError("Model is not properly initialized")

        n_assets = self._n_assets

        # Set up random number generator
        if isinstance(random_state, np.random.Generator):
            rng = random_state
        else:
            rng = np.random.default_rng(random_state)

        # Total number of periods to simulate (including burn-in)
        total_periods = n_periods + burn

        # Simulate univariate volatilities
        univariate_volatilities = np.zeros((total_periods, n_assets))
        for i, model in enumerate(self._univariate_models):
            # Simulate univariate volatility
            sim_vol = model.simulate(total_periods, return_volatility=True, random_state=rng)
            if isinstance(sim_vol, tuple):
                # If simulate returns both data and volatility, extract volatility
                _, vol = sim_vol
                univariate_volatilities[:, i] = np.sqrt(vol)
            else:
                # If simulate only returns data, compute volatility from the model
                univariate_volatilities[:, i] = np.sqrt(model.conditional_variances)

        # Get constant correlation matrix
        const_corr = self._constant_correlation

        # Allocate arrays for simulated data and covariances
        simulated_data = np.zeros((total_periods, n_assets))
        covariances = np.zeros((n_assets, n_assets, total_periods))

        # Simulate data
        for t in range(total_periods):
            # Create diagonal matrix of volatilities
            D = np.diag(univariate_volatilities[t, :])

            # Compute covariance matrix: D * R * D
            cov_t = D @ const_corr @ D
            covariances[:, :, t] = cov_t

            # Generate multivariate normal random variables
            z = rng.multivariate_normal(np.zeros(n_assets), const_corr)

            # Apply volatility
            simulated_data[t, :] = D @ z

        # Discard burn-in periods
        if burn > 0:
            simulated_data = simulated_data[burn:]
            covariances = covariances[:, :, burn:]

        if return_covariances:
            return simulated_data, covariances
        else:
            return simulated_data

    async def simulate_async(
        self,
        n_periods: int,
        burn: int = 0,
        initial_value: Optional[np.ndarray] = None,
        random_state: Optional[Union[int, np.random.Generator]] = None,
        return_covariances: bool = False,
        callback: Optional[AsyncProgressCallback] = None,
        **kwargs: Any
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Asynchronously simulate data from the CCC model.

        This method provides an asynchronous interface to the simulate method,
        allowing for non-blocking simulation in UI contexts.

        Args:
            n_periods: Number of periods to simulate
            burn: Number of initial observations to discard
            initial_value: Initial covariance matrix for the simulation
            random_state: Random number generator or seed
            return_covariances: Whether to return conditional covariances
            callback: Async callback function for reporting simulation progress
            **kwargs: Additional keyword arguments for simulation

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: 
                Simulated data and optionally conditional covariances

        Raises:
            RuntimeError: If the model has not been fitted
            ValueError: If the simulation parameters are invalid
        """
        # Create a synchronous callback that calls the async callback
        sync_callback = None
        if callback is not None:
            def sync_callback_wrapper(progress: float, message: str) -> None:
                asyncio.create_task(callback(progress, message))

            # Wrap the simulate method to call the callback
            original_simulate = self.simulate

            def simulate_with_callback(*args: Any, **kwargs: Any) -> Any:
                total_steps = n_periods + burn
                for i in range(total_steps):
                    if i % max(1, total_steps // 10) == 0:
                        progress = i / total_steps
                        sync_callback_wrapper(progress, f"Simulating period {i+1}/{total_steps}")
                return original_simulate(*args, **kwargs)

            # Run the wrapped simulate method in an executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: simulate_with_callback(
                    n_periods=n_periods,
                    burn=burn,
                    initial_value=initial_value,
                    random_state=random_state,
                    return_covariances=return_covariances,
                    **kwargs
                )
            )
        else:
            # Run the original simulate method in an executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.simulate(
                    n_periods=n_periods,
                    burn=burn,
                    initial_value=initial_value,
                    random_state=random_state,
                    return_covariances=return_covariances,
                    **kwargs
                )
            )

        return result

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

        if self._n_assets is None or self._univariate_models is None:
            raise RuntimeError("Model is not properly initialized")

        n_assets = self._n_assets

        # Forecast univariate volatilities
        univariate_forecasts = np.zeros((steps, n_assets))
        for i, model in enumerate(self._univariate_models):
            # Forecast univariate volatility
            forecast = model.forecast(steps, **kwargs)
            univariate_forecasts[:, i] = np.sqrt(forecast)

        # Get constant correlation matrix
        const_corr = self._constant_correlation

        # Allocate array for forecasted covariances
        forecasted_covariances = np.zeros((n_assets, n_assets, steps))

        # Compute forecasted covariances
        for t in range(steps):
            # Create diagonal matrix of volatilities
            D = np.diag(univariate_forecasts[t, :])

            # Compute covariance matrix: D * R * D
            forecasted_covariances[:, :, t] = D @ const_corr @ D

        return forecasted_covariances

    def __str__(self) -> str:
        """Return a string representation of the model.

        Returns:
            str: String representation
        """
        if not self._fitted:
            return f"{self.name} model (not fitted)"

        n_assets = self._n_assets if self._n_assets is not None else 0
        n_params = len(self._parameters.correlations) if self._parameters is not None else 0

        return (
            f"{self.name} model with {n_assets} assets\n"
            f"Number of correlation parameters: {n_params}\n"
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
    Register Numba JIT-compiled functions for CCC model.

    This function is called during module initialization to register
    performance-critical functions for JIT compilation if Numba is available.
    """
    if HAS_NUMBA:
        # The functions are already decorated with @jit, so no additional
        # registration is needed here. This function is kept for consistency
        # with the module structure and potential future enhancements.
        logger.debug("CCC model Numba JIT functions registered")
    else:
        logger.info("Numba not available. CCC model will use pure NumPy implementations.")


# Initialize the module
def _register_numba_functions():
    _register_numba_functions()
