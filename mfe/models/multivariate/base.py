"""
Abstract base classes for multivariate volatility models.

This module defines the abstract base classes that establish the contract for all
multivariate volatility model implementations in the MFE Toolbox. These classes
provide a consistent interface for initialization, parameter validation, model
estimation, simulation, forecasting, and diagnostics across different multivariate
volatility model types.

The class hierarchy is designed to ensure that all multivariate volatility models
follow the same patterns while allowing for model-specific implementations of core
functionality such as covariance matrix computation and parameter transformation.
"""

import abc
import asyncio
import warnings
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union, cast, overload
)

import numpy as np
import pandas as pd
from scipy import optimize, stats

from mfe.core.base import ModelBase, MultivariateVolatilityModelBase
from mfe.core.parameters import (
    MultivariateVolatilityParameters, ParameterBase, ParameterError,
    validate_positive_definite
)
from mfe.core.results import MultivariateVolatilityResult
from mfe.core.types import (
    AsyncProgressCallback, CovarianceMatrix, CorrelationMatrix, Matrix,
    MultivariateVolatilityType, ProgressCallback, Vector
)
from mfe.utils.matrix_ops import cov2corr, vech, ivech


class AbstractMultivariateVolatilityModel(MultivariateVolatilityModelBase):
    """Abstract base class for all multivariate volatility models.
    
    This class defines the common interface that all multivariate volatility model
    implementations must follow, establishing a consistent API across the entire
    multivariate volatility module.
    
    Attributes:
        name: A descriptive name for the model
        n_assets: Number of assets in the model
        parameters: Model parameters if fitted
        _conditional_covariances: Conditional covariance matrices if fitted
        _conditional_correlations: Conditional correlation matrices if fitted
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        n_assets: Optional[int] = None
    ):
        """Initialize the multivariate volatility model.
        
        Args:
            name: A descriptive name for the model (if None, uses class name)
            n_assets: Number of assets in the model (if None, determined from data)
        """
        if name is None:
            name = self.__class__.__name__
        super().__init__(name=name)
        
        self._n_assets = n_assets
        self._parameters: Optional[MultivariateVolatilityParameters] = None
        self._conditional_correlations: Optional[np.ndarray] = None
        self._residuals: Optional[np.ndarray] = None
        self._backcast: Optional[np.ndarray] = None
    
    @property
    def parameters(self) -> Optional[MultivariateVolatilityParameters]:
        """Get the model parameters.
        
        Returns:
            Optional[MultivariateVolatilityParameters]: The model parameters if fitted,
                                                      None otherwise
        """
        return self._parameters
    
    @property
    def conditional_correlations(self) -> Optional[np.ndarray]:
        """Get the conditional correlation matrices.
        
        Returns:
            Optional[np.ndarray]: The conditional correlation matrices if the model
                                 has been fitted, None otherwise
        
        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self._conditional_correlations
    
    @property
    def residuals(self) -> Optional[np.ndarray]:
        """Get the residuals used for model fitting.
        
        Returns:
            Optional[np.ndarray]: The residuals if the model has been fitted,
                                 None otherwise
        
        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self._residuals
    
    @abc.abstractmethod
    def fit(
        self,
        data: np.ndarray,
        starting_values: Optional[Union[np.ndarray, MultivariateVolatilityParameters]] = None,
        backcast: Optional[np.ndarray] = None,
        method: str = "SLSQP",
        options: Optional[Dict[str, Any]] = None,
        constraints: Optional[List[Dict[str, Any]]] = None,
        callback: Optional[ProgressCallback] = None,
        **kwargs: Any
    ) -> MultivariateVolatilityResult:
        """Fit the model to the provided data.
        
        This method must be implemented by all subclasses to estimate model
        parameters from the provided data.
        
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
        pass
    
    async def fit_async(
        self,
        data: np.ndarray,
        starting_values: Optional[Union[np.ndarray, MultivariateVolatilityParameters]] = None,
        backcast: Optional[np.ndarray] = None,
        method: str = "SLSQP",
        options: Optional[Dict[str, Any]] = None,
        constraints: Optional[List[Dict[str, Any]]] = None,
        callback: Optional[AsyncProgressCallback] = None,
        **kwargs: Any
    ) -> MultivariateVolatilityResult:
        """Asynchronously fit the model to the provided data.
        
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
    
    @abc.abstractmethod
    def simulate(
        self,
        n_periods: int,
        burn: int = 0,
        initial_value: Optional[np.ndarray] = None,
        random_state: Optional[Union[int, np.random.Generator]] = None,
        return_covariances: bool = False,
        **kwargs: Any
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Simulate data from the model.
        
        This method must be implemented by all subclasses to generate simulated
        data based on the model parameters.
        
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
        pass
    
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
        """Asynchronously simulate data from the model.
        
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
    
    @abc.abstractmethod
    def forecast(
        self,
        steps: int,
        initial_value: Optional[np.ndarray] = None,
        **kwargs: Any
    ) -> np.ndarray:
        """Forecast conditional covariance matrices.
        
        This method must be implemented by all subclasses to generate forecasts
        of conditional covariance matrices.
        
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
        pass
    
    async def forecast_async(
        self,
        steps: int,
        initial_value: Optional[np.ndarray] = None,
        callback: Optional[AsyncProgressCallback] = None,
        **kwargs: Any
    ) -> np.ndarray:
        """Asynchronously forecast conditional covariance matrices.
        
        This method provides an asynchronous interface to the forecast method,
        allowing for non-blocking forecasting in UI contexts.
        
        Args:
            steps: Number of steps to forecast
            initial_value: Initial covariance matrix for forecasting
            callback: Async callback function for reporting forecast progress
            **kwargs: Additional keyword arguments for forecasting
        
        Returns:
            np.ndarray: Forecasted conditional covariance matrices (n_assets x n_assets x steps)
        
        Raises:
            RuntimeError: If the model has not been fitted
            ValueError: If the forecast parameters are invalid
        """
        # Create a synchronous callback that calls the async callback
        sync_callback = None
        if callback is not None:
            def sync_callback_wrapper(progress: float, message: str) -> None:
                asyncio.create_task(callback(progress, message))
            
            # Wrap the forecast method to call the callback
            original_forecast = self.forecast
            
            def forecast_with_callback(*args: Any, **kwargs: Any) -> Any:
                for i in range(steps):
                    if i % max(1, steps // 10) == 0:
                        progress = i / steps
                        sync_callback_wrapper(progress, f"Forecasting step {i+1}/{steps}")
                return original_forecast(*args, **kwargs)
            
            # Run the wrapped forecast method in an executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: forecast_with_callback(
                    steps=steps,
                    initial_value=initial_value,
                    **kwargs
                )
            )
        else:
            # Run the original forecast method in an executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.forecast(
                    steps=steps,
                    initial_value=initial_value,
                    **kwargs
                )
            )
        
        return result
    
    @abc.abstractmethod
    def compute_covariance(
        self,
        parameters: MultivariateVolatilityParameters,
        data: np.ndarray,
        sigma: Optional[np.ndarray] = None,
        backcast: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute conditional covariance matrices for the given parameters and data.
        
        This method must be implemented by all subclasses to compute the
        conditional covariance matrices based on the model parameters and data.
        
        Args:
            parameters: Model parameters
            data: Input data (typically residuals)
            sigma: Pre-allocated array for conditional covariances
            backcast: Matrix to use for initializing the covariance process
        
        Returns:
            np.ndarray: Conditional covariance matrices (n_assets x n_assets x T)
        """
        pass
    
    def compute_correlation(
        self,
        parameters: MultivariateVolatilityParameters,
        data: np.ndarray,
        correlation: Optional[np.ndarray] = None,
        backcast: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute conditional correlation matrices for the given parameters and data.
        
        This method computes conditional correlation matrices by first computing
        covariance matrices and then converting them to correlation matrices.
        Subclasses may override this method for more efficient implementations.
        
        Args:
            parameters: Model parameters
            data: Input data (typically residuals)
            correlation: Pre-allocated array for conditional correlations
            backcast: Matrix to use for initializing the correlation process
        
        Returns:
            np.ndarray: Conditional correlation matrices (n_assets x n_assets x T)
        """
        # Compute covariance matrices
        covariance = self.compute_covariance(parameters, data, backcast=backcast)
        
        # Convert to correlation matrices
        n_assets = covariance.shape[0]
        T = covariance.shape[2]
        
        if correlation is None:
            correlation = np.zeros((n_assets, n_assets, T))
        
        # Convert each covariance matrix to a correlation matrix
        for t in range(T):
            correlation[:, :, t] = cov2corr(covariance[:, :, t])
        
        return correlation
    
    def validate_data(self, data: np.ndarray) -> None:
        """Validate the input data for multivariate volatility model fitting.
        
        Args:
            data: The data to validate
        
        Raises:
            ValueError: If the data is invalid
            TypeError: If the data has an incorrect type
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a NumPy array")
        
        if data.ndim != 2:
            raise ValueError(f"Data must be 2-dimensional, got {data.ndim} dimensions")
        
        if data.shape[0] < 10:  # Arbitrary minimum length
            raise ValueError(f"Data length must be at least 10, got {data.shape[0]}")
        
        if data.shape[1] < 2:
            raise ValueError(f"Data must have at least 2 variables, got {data.shape[1]}")
        
        if np.isnan(data).any():
            raise ValueError("Data contains NaN values")
        
        if np.isinf(data).any():
            raise ValueError("Data contains infinite values")
        
        # Update n_assets if not already set
        if self._n_assets is None:
            self._n_assets = data.shape[1]
        elif self._n_assets != data.shape[1]:
            raise ValueError(
                f"Data has {data.shape[1]} variables, but model was initialized with {self._n_assets} assets"
            )
    
    def compute_backcast(self, data: np.ndarray) -> np.ndarray:
        """Compute a backcast value for initializing the covariance process.
        
        This method computes a backcast value based on the unconditional covariance
        of the input data. Subclasses may override this method for model-specific
        backcast computation.
        
        Args:
            data: Input data (typically residuals)
        
        Returns:
            np.ndarray: Backcast covariance matrix
        """
        # Compute sample covariance matrix
        backcast = np.cov(data, rowvar=False)
        
        # Ensure positive definiteness
        try:
            validate_positive_definite(backcast, "backcast")
        except ParameterError:
            # If not positive definite, add a small diagonal component
            min_eigenvalue = np.min(np.linalg.eigvalsh(backcast))
            if min_eigenvalue <= 0:
                backcast += np.eye(backcast.shape[0]) * (abs(min_eigenvalue) + 1e-6)
        
        return backcast
    
    def standardized_residuals(self, data: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute standardized residuals from the fitted model.
        
        Args:
            data: The data to compute standardized residuals for (if None, uses the data from fitting)
        
        Returns:
            np.ndarray: Standardized residuals
        
        Raises:
            RuntimeError: If the model has not been fitted
            ValueError: If the data dimensions don't match the conditional covariances
        """
        if not self._fitted or self._conditional_covariances is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        # Use the data from fitting if not provided
        if data is None:
            if self._residuals is None:
                raise RuntimeError("No residuals available from fitting")
            data = self._residuals
        
        if data.shape[0] != self._conditional_covariances.shape[2]:
            raise ValueError(
                f"Data length ({data.shape[0]}) must match conditional covariances "
                f"length ({self._conditional_covariances.shape[2]})"
            )
        
        if data.shape[1] != self._conditional_covariances.shape[0]:
            raise ValueError(
                f"Data width ({data.shape[1]}) must match conditional covariances "
                f"width ({self._conditional_covariances.shape[0]})"
            )
        
        std_resid = np.zeros_like(data)
        for t in range(data.shape[0]):
            # Compute Cholesky decomposition of covariance matrix
            try:
                chol = np.linalg.cholesky(self._conditional_covariances[:, :, t])
                # Standardize using the Cholesky factor
                std_resid[t, :] = np.linalg.solve(chol, data[t, :])
            except np.linalg.LinAlgError:
                # If Cholesky decomposition fails, use eigenvalue decomposition
                eigvals, eigvecs = np.linalg.eigh(self._conditional_covariances[:, :, t])
                # Ensure all eigenvalues are positive
                eigvals = np.maximum(eigvals, 1e-8)
                # Compute inverse square root of covariance matrix
                inv_sqrt_cov = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                # Standardize using the inverse square root
                std_resid[t, :] = inv_sqrt_cov @ data[t, :]
        
        return std_resid
    
    def loglikelihood(
        self,
        parameters: Union[np.ndarray, MultivariateVolatilityParameters],
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
    
    def _array_to_parameters(self, array: np.ndarray) -> MultivariateVolatilityParameters:
        """Convert a parameter array to a parameter object.
        
        This method must be implemented by subclasses to convert a NumPy array
        of parameters to a parameter object specific to the model.
        
        Args:
            array: Parameter array
        
        Returns:
            MultivariateVolatilityParameters: Parameter object
        
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("_array_to_parameters must be implemented by subclass")
    
    def _parameters_to_array(self, parameters: MultivariateVolatilityParameters) -> np.ndarray:
        """Convert a parameter object to a parameter array.
        
        This method must be implemented by subclasses to convert a parameter object
        specific to the model to a NumPy array of parameters.
        
        Args:
            parameters: Parameter object
        
        Returns:
            np.ndarray: Parameter array
        
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("_parameters_to_array must be implemented by subclass")
    
    def _objective_function(
        self,
        parameters: np.ndarray,
        data: np.ndarray,
        backcast: Optional[np.ndarray] = None
    ) -> float:
        """Objective function for optimization (negative log-likelihood).
        
        Args:
            parameters: Parameter array
            data: Input data (typically residuals)
            backcast: Matrix to use for initializing the covariance process
        
        Returns:
            float: Negative log-likelihood value
        """
        try:
            # Convert parameters to parameter object
            param_obj = self._array_to_parameters(parameters)
            
            # Compute log-likelihood
            loglik = self.loglikelihood(param_obj, data, backcast=backcast)
            
            # Return negative log-likelihood for minimization
            return -loglik
        except (ValueError, ParameterError, np.linalg.LinAlgError) as e:
            # Return a large value if parameters are invalid
            return 1e10
    
    def _create_constraints(self) -> List[Dict[str, Any]]:
        """Create constraints for optimization.
        
        This method should be implemented by subclasses to create constraints
        specific to the model.
        
        Returns:
            List[Dict[str, Any]]: List of constraints for optimization
        """
        return []
    
    def _create_bounds(self) -> List[Tuple[float, float]]:
        """Create bounds for optimization parameters.
        
        This method should be implemented by subclasses to create bounds
        specific to the model.
        
        Returns:
            List[Tuple[float, float]]: List of (lower, upper) bounds for each parameter
        """
        return []
    
    def _create_starting_values(self, data: np.ndarray) -> np.ndarray:
        """Create starting values for optimization.
        
        This method should be implemented by subclasses to create starting values
        specific to the model.
        
        Args:
            data: Input data (typically residuals)
        
        Returns:
            np.ndarray: Starting values for optimization
        """
        raise NotImplementedError("_create_starting_values must be implemented by subclass")
    
    def _compute_std_errors(
        self,
        parameters: np.ndarray,
        data: np.ndarray,
        backcast: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute standard errors for parameter estimates.
        
        Args:
            parameters: Parameter array
            data: Input data (typically residuals)
            backcast: Matrix to use for initializing the covariance process
        
        Returns:
            np.ndarray: Standard errors for parameter estimates
        """
        # Compute Hessian using finite differences
        def objective(params: np.ndarray) -> float:
            return self._objective_function(params, data, backcast)
        
        try:
            # Compute Hessian using finite differences
            hessian = optimize.approx_fprime(
                parameters,
                lambda params: optimize.approx_fprime(
                    params,
                    objective,
                    epsilon=1e-5
                ),
                epsilon=1e-5
            )
            
            # Compute covariance matrix as inverse of Hessian
            try:
                cov_matrix = np.linalg.inv(hessian)
                # Extract standard errors as square root of diagonal elements
                std_errors = np.sqrt(np.diag(cov_matrix))
                return std_errors
            except np.linalg.LinAlgError:
                # If Hessian is singular, use pseudo-inverse
                cov_matrix = np.linalg.pinv(hessian)
                std_errors = np.sqrt(np.diag(cov_matrix))
                warnings.warn(
                    "Hessian matrix is singular. Standard errors may be unreliable.",
                    UserWarning
                )
                return std_errors
        except Exception as e:
            # If Hessian computation fails, return NaN values
            warnings.warn(
                f"Failed to compute standard errors: {str(e)}",
                UserWarning
            )
            return np.full(len(parameters), np.nan)
    
    def _compute_persistence(self) -> Optional[float]:
        """Compute the persistence of the model.
        
        This method should be implemented by subclasses to compute the persistence
        specific to the model.
        
        Returns:
            Optional[float]: Persistence value
        """
        return None
    
    def _compute_half_life(self) -> Optional[float]:
        """Compute the half-life of shocks in the model.
        
        Returns:
            Optional[float]: Half-life value
        """
        persistence = self._compute_persistence()
        if persistence is not None and persistence < 1:
            return np.log(0.5) / np.log(persistence)
        return None
    
    def _compute_unconditional_covariance(self) -> Optional[np.ndarray]:
        """Compute the unconditional covariance matrix implied by the model.
        
        This method should be implemented by subclasses to compute the unconditional
        covariance matrix specific to the model.
        
        Returns:
            Optional[np.ndarray]: Unconditional covariance matrix
        """
        return None
    
    def _create_result_object(
        self,
        parameters: MultivariateVolatilityParameters,
        data: np.ndarray,
        covariance: np.ndarray,
        correlation: Optional[np.ndarray],
        loglikelihood: float,
        std_errors: np.ndarray,
        iterations: int,
        convergence: bool,
        optimization_message: Optional[str] = None,
        **kwargs: Any
    ) -> MultivariateVolatilityResult:
        """Create a result object for the model.
        
        Args:
            parameters: Estimated model parameters
            data: Input data (typically residuals)
            covariance: Conditional covariance matrices
            correlation: Conditional correlation matrices
            loglikelihood: Log-likelihood value
            std_errors: Standard errors for parameter estimates
            iterations: Number of iterations used in optimization
            convergence: Whether the optimization converged
            optimization_message: Message from the optimizer
            **kwargs: Additional keyword arguments for the result object
        
        Returns:
            MultivariateVolatilityResult: Model estimation results
        """
        # Compute t-statistics and p-values
        t_stats = np.full_like(std_errors, np.nan)
        p_values = np.full_like(std_errors, np.nan)
        
        param_array = self._parameters_to_array(parameters)
        
        # Compute t-statistics and p-values where standard errors are available
        mask = ~np.isnan(std_errors) & (std_errors > 0)
        if np.any(mask):
            t_stats[mask] = param_array[mask] / std_errors[mask]
            p_values[mask] = 2 * (1 - stats.t.cdf(np.abs(t_stats[mask]), data.shape[0] - len(param_array)))
        
        # Compute information criteria
        n_params = len(param_array)
        T = data.shape[0]
        aic = -2 * loglikelihood + 2 * n_params
        bic = -2 * loglikelihood + n_params * np.log(T)
        hqic = -2 * loglikelihood + 2 * n_params * np.log(np.log(T))
        
        # Compute persistence and half-life
        persistence = self._compute_persistence()
        half_life = self._compute_half_life()
        
        # Compute unconditional covariance
        unconditional_covariance = self._compute_unconditional_covariance()
        
        # Create result object
        result = MultivariateVolatilityResult(
            model_name=self.name,
            parameters=parameters,
            convergence=convergence,
            iterations=iterations,
            log_likelihood=loglikelihood,
            aic=aic,
            bic=bic,
            hqic=hqic,
            std_errors=std_errors,
            t_stats=t_stats,
            p_values=p_values,
            covariance_matrix=None,  # Covariance matrix of parameter estimates
            optimization_message=optimization_message,
            conditional_covariances=covariance,
            conditional_correlations=correlation,
            standardized_residuals=self.standardized_residuals(data),
            n_assets=self._n_assets,
            persistence=persistence,
            half_life=half_life,
            unconditional_covariance=unconditional_covariance,
            residuals=data,
            **kwargs
        )
        
        return result


class CorrelationModelBase(AbstractMultivariateVolatilityModel):
    """Base class for correlation-based multivariate volatility models.
    
    This class extends AbstractMultivariateVolatilityModel to provide specialized
    functionality for correlation-based models like DCC and CCC, which separate
    the estimation of univariate volatility models and correlation dynamics.
    
    Attributes:
        name: A descriptive name for the model
        n_assets: Number of assets in the model
        univariate_volatilities: Univariate volatility estimates
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        n_assets: Optional[int] = None
    ):
        """Initialize the correlation-based multivariate volatility model.
        
        Args:
            name: A descriptive name for the model (if None, uses class name)
            n_assets: Number of assets in the model (if None, determined from data)
        """
        if name is None:
            name = self.__class__.__name__
        super().__init__(name=name, n_assets=n_assets)
        
        self._univariate_volatilities: Optional[np.ndarray] = None
    
    @property
    def univariate_volatilities(self) -> Optional[np.ndarray]:
        """Get the univariate volatility estimates.
        
        Returns:
            Optional[np.ndarray]: The univariate volatility estimates if the model
                                 has been fitted, None otherwise
        
        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self._univariate_volatilities
    
    def compute_covariance(
        self,
        parameters: MultivariateVolatilityParameters,
        data: np.ndarray,
        sigma: Optional[np.ndarray] = None,
        backcast: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute conditional covariance matrices for the given parameters and data.
        
        For correlation-based models, this method computes covariance matrices by
        combining univariate volatilities and correlation matrices.
        
        Args:
            parameters: Model parameters
            data: Input data (typically residuals)
            sigma: Pre-allocated array for conditional covariances
            backcast: Matrix to use for initializing the covariance process
        
        Returns:
            np.ndarray: Conditional covariance matrices (n_assets x n_assets x T)
        """
        # Compute univariate volatilities if not already available
        if self._univariate_volatilities is None:
            self._compute_univariate_volatilities(data)
        
        # Compute correlation matrices
        correlation = self.compute_correlation(parameters, data, backcast=backcast)
        
        # Combine volatilities and correlations to form covariance matrices
        n_assets = data.shape[1]
        T = data.shape[0]
        
        if sigma is None:
            sigma = np.zeros((n_assets, n_assets, T))
        
        for t in range(T):
            # Create diagonal matrix of volatilities
            D = np.diag(self._univariate_volatilities[t, :])
            # Compute covariance matrix: D * R * D
            sigma[:, :, t] = D @ correlation[:, :, t] @ D
        
        return sigma
    
    def _compute_univariate_volatilities(self, data: np.ndarray) -> np.ndarray:
        """Compute univariate volatility estimates for each asset.
        
        This method should be implemented by subclasses to compute univariate
        volatility estimates specific to the model.
        
        Args:
            data: Input data (typically residuals)
        
        Returns:
            np.ndarray: Univariate volatility estimates (T x n_assets)
        
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("_compute_univariate_volatilities must be implemented by subclass")


class CovarianceModelBase(AbstractMultivariateVolatilityModel):
    """Base class for covariance-based multivariate volatility models.
    
    This class extends AbstractMultivariateVolatilityModel to provide specialized
    functionality for covariance-based models like BEKK and RARCH, which directly
    model the covariance matrix dynamics.
    
    Attributes:
        name: A descriptive name for the model
        n_assets: Number of assets in the model
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        n_assets: Optional[int] = None
    ):
        """Initialize the covariance-based multivariate volatility model.
        
        Args:
            name: A descriptive name for the model (if None, uses class name)
            n_assets: Number of assets in the model (if None, determined from data)
        """
        if name is None:
            name = self.__class__.__name__
        super().__init__(name=name, n_assets=n_assets)
    
    def compute_correlation(
        self,
        parameters: MultivariateVolatilityParameters,
        data: np.ndarray,
        correlation: Optional[np.ndarray] = None,
        backcast: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute conditional correlation matrices for the given parameters and data.
        
        For covariance-based models, this method computes correlation matrices by
        first computing covariance matrices and then converting them to correlation matrices.
        
        Args:
            parameters: Model parameters
            data: Input data (typically residuals)
            correlation: Pre-allocated array for conditional correlations
            backcast: Matrix to use for initializing the correlation process
        
        Returns:
            np.ndarray: Conditional correlation matrices (n_assets x n_assets x T)
        """
        # Compute covariance matrices
        covariance = self.compute_covariance(parameters, data, backcast=backcast)
        
        # Convert to correlation matrices
        n_assets = covariance.shape[0]
        T = covariance.shape[2]
        
        if correlation is None:
            correlation = np.zeros((n_assets, n_assets, T))
        
        # Convert each covariance matrix to a correlation matrix
        for t in range(T):
            correlation[:, :, t] = cov2corr(covariance[:, :, t])
        
        return correlation


class FactorModelBase(AbstractMultivariateVolatilityModel):
    """Base class for factor-based multivariate volatility models.
    
    This class extends AbstractMultivariateVolatilityModel to provide specialized
    functionality for factor-based models like O-GARCH and GO-GARCH, which use
    factor decompositions to model covariance dynamics.
    
    Attributes:
        name: A descriptive name for the model
        n_assets: Number of assets in the model
        n_factors: Number of factors in the model
        factor_loadings: Factor loading matrix
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        n_assets: Optional[int] = None,
        n_factors: Optional[int] = None
    ):
        """Initialize the factor-based multivariate volatility model.
        
        Args:
            name: A descriptive name for the model (if None, uses class name)
            n_assets: Number of assets in the model (if None, determined from data)
            n_factors: Number of factors in the model (if None, determined from data)
        """
        if name is None:
            name = self.__class__.__name__
        super().__init__(name=name, n_assets=n_assets)
        
        self._n_factors = n_factors
        self._factor_loadings: Optional[np.ndarray] = None
        self._factor_volatilities: Optional[np.ndarray] = None
    
    @property
    def n_factors(self) -> Optional[int]:
        """Get the number of factors in the model.
        
        Returns:
            Optional[int]: The number of factors if the model has been fitted,
                          None otherwise
        """
        return self._n_factors
    
    @property
    def factor_loadings(self) -> Optional[np.ndarray]:
        """Get the factor loading matrix.
        
        Returns:
            Optional[np.ndarray]: The factor loading matrix if the model has been fitted,
                                 None otherwise
        
        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self._factor_loadings
    
    @property
    def factor_volatilities(self) -> Optional[np.ndarray]:
        """Get the factor volatility estimates.
        
        Returns:
            Optional[np.ndarray]: The factor volatility estimates if the model
                                 has been fitted, None otherwise
        
        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self._factor_volatilities
    
    def compute_covariance(
        self,
        parameters: MultivariateVolatilityParameters,
        data: np.ndarray,
        sigma: Optional[np.ndarray] = None,
        backcast: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute conditional covariance matrices for the given parameters and data.
        
        For factor-based models, this method computes covariance matrices by
        combining factor loadings and factor volatilities.
        
        Args:
            parameters: Model parameters
            data: Input data (typically residuals)
            sigma: Pre-allocated array for conditional covariances
            backcast: Matrix to use for initializing the covariance process
        
        Returns:
            np.ndarray: Conditional covariance matrices (n_assets x n_assets x T)
        """
        # Compute factor loadings if not already available
        if self._factor_loadings is None:
            self._compute_factor_loadings(data)
        
        # Compute factor volatilities
        factor_volatilities = self._compute_factor_volatilities(parameters, data, backcast=backcast)
        
        # Combine factor loadings and volatilities to form covariance matrices
        n_assets = data.shape[1]
        T = data.shape[0]
        
        if sigma is None:
            sigma = np.zeros((n_assets, n_assets, T))
        
        for t in range(T):
            # Create diagonal matrix of factor volatilities
            D = np.diag(factor_volatilities[t, :])
            # Compute covariance matrix: A * D * A'
            sigma[:, :, t] = self._factor_loadings @ D @ self._factor_loadings.T
        
        return sigma
    
    def _compute_factor_loadings(self, data: np.ndarray) -> np.ndarray:
        """Compute factor loadings from the data.
        
        This method should be implemented by subclasses to compute factor loadings
        specific to the model.
        
        Args:
            data: Input data (typically residuals)
        
        Returns:
            np.ndarray: Factor loading matrix (n_assets x n_factors)
        
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("_compute_factor_loadings must be implemented by subclass")
    
    def _compute_factor_volatilities(
        self,
        parameters: MultivariateVolatilityParameters,
        data: np.ndarray,
        backcast: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute factor volatility estimates.
        
        This method should be implemented by subclasses to compute factor volatility
        estimates specific to the model.
        
        Args:
            parameters: Model parameters
            data: Input data (typically residuals)
            backcast: Matrix to use for initializing the volatility process
        
        Returns:
            np.ndarray: Factor volatility estimates (T x n_factors)
        
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("_compute_factor_volatilities must be implemented by subclass")
