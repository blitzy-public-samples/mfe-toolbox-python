# mfe/models/multivariate/matrix_garch.py

"""
Matrix GARCH model implementation.

This module implements the Matrix GARCH model, a direct multivariate extension of the
univariate GARCH that models covariance matrices directly. This approach allows for
complex dynamics in the conditional covariance structure without imposing restrictive
assumptions about correlation patterns.

The Matrix GARCH model follows the recursion:
    vech(H_t) = vech(C) + sum_{i=1}^p A_i vech(ε_{t-i}ε_{t-i}') + sum_{j=1}^q B_j vech(H_{t-j})

where:
    - H_t is the conditional covariance matrix at time t
    - ε_t is the innovation vector at time t
    - vech() is the half-vectorization operator that stacks the lower triangular
      portion of a matrix into a vector
    - C is a constant matrix
    - A_i and B_j are coefficient matrices for the ARCH and GARCH terms

The model provides a flexible framework for modeling time-varying covariance matrices
with direct parameter interpretation and straightforward implementation.
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats

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
from mfe.models.multivariate.base import CovarianceModelBase
from mfe.models.multivariate.utils import (
    compute_sample_covariance, compute_sample_correlation, ensure_positive_definite,
    initialize_parameters, format_multivariate_results, compute_persistence,
    compute_half_life, standardize_residuals, compute_robust_covariance,
    check_stationarity, check_positive_definiteness, compute_eigenvalues,
    compute_conditional_correlations, compute_conditional_variances,
    compute_unconditional_covariance, validate_multivariate_data
)
from mfe.utils.matrix_ops import vech, ivech, ensure_symmetric
from numba import jit

# Set up module-level logger
logger = logging.getLogger("mfe.models.multivariate.matrix_garch")


@dataclass
class MatrixGARCHParameters(MultivariateVolatilityParameters):
    """
    Parameters for the Matrix GARCH model.

    This class encapsulates the parameters of a Matrix GARCH model, including
    the constant term, ARCH coefficients, and GARCH coefficients.

    Attributes:
        C: Constant matrix in vech form
        A: ARCH coefficient matrices in vech form (list of length p)
        B: GARCH coefficient matrices in vech form (list of length q)
        p: ARCH order
        q: GARCH order
        n_assets: Number of assets
    """

    C: np.ndarray
    A: List[np.ndarray]
    B: List[np.ndarray]
    p: int
    q: int
    n_assets: int

    def __post_init__(self) -> None:
        """
        Validate the parameters after initialization.

        Raises:
            ParameterError: If parameters are invalid
        """
        # Check dimensions
        n_vech = self.n_assets * (self.n_assets + 1) // 2

        if self.C.shape != (n_vech,):
            raise ParameterError(
                f"C must have shape ({n_vech},), got {self.C.shape}"
            )

        if len(self.A) != self.p:
            raise ParameterError(
                f"A must have length p={self.p}, got {len(self.A)}"
            )

        if len(self.B) != self.q:
            raise ParameterError(
                f"B must have length q={self.q}, got {len(self.B)}"
            )

        for i, A_i in enumerate(self.A):
            if A_i.shape != (n_vech,):
                raise ParameterError(
                    f"A[{i}] must have shape ({n_vech},), got {A_i.shape}"
                )

        for i, B_i in enumerate(self.B):
            if B_i.shape != (n_vech,):
                raise ParameterError(
                    f"B[{i}] must have shape ({n_vech},), got {B_i.shape}"
                )

        # Check stationarity
        persistence = np.sum([np.sum(A_i) for A_i in self.A]) + np.sum([np.sum(B_i) for B_i in self.B])
        if persistence >= 1:
            warnings.warn(
                f"Model may not be stationary: sum of A and B coefficients = {persistence} >= 1",
                UserWarning
            )

    def to_array(self) -> np.ndarray:
        """
        Convert parameters to a flat array.

        Returns:
            np.ndarray: Flattened parameter array
        """
        # Concatenate all parameters into a single array
        params = [self.C]
        params.extend(self.A)
        params.extend(self.B)

        return np.concatenate(params)

    @classmethod
    def from_array(cls, array: np.ndarray, p: int, q: int, n_assets: int) -> "MatrixGARCHParameters":
        """
        Create parameters from a flat array.

        Args:
            array: Flattened parameter array
            p: ARCH order
            q: GARCH order
            n_assets: Number of assets

        Returns:
            MatrixGARCHParameters: Parameter object

        Raises:
            ParameterError: If array length is incorrect
        """
        n_vech = n_assets * (n_assets + 1) // 2
        expected_length = n_vech * (1 + p + q)

        if len(array) != expected_length:
            raise ParameterError(
                f"Parameter array length ({len(array)}) doesn't match expected length "
                f"n_vech*(1+p+q) = {expected_length} for n_assets={n_assets}, p={p}, q={q}"
            )

        # Extract parameters
        C = array[:n_vech]

        A = []
        for i in range(p):
            start = n_vech * (1 + i)
            end = n_vech * (2 + i)
            A.append(array[start:end])

        B = []
        for i in range(q):
            start = n_vech * (1 + p + i)
            end = n_vech * (2 + p + i)
            B.append(array[start:end])

        return cls(C=C, A=A, B=B, p=p, q=q, n_assets=n_assets)


@jit(nopython=True, cache=True)
def matrix_garch_recursion(data: np.ndarray, C: np.ndarray, A: np.ndarray, B: np.ndarray,
                           H_t: np.ndarray, T: int, n_assets: int, p: int, q: int) -> np.ndarray:
    """
    Numba-accelerated Matrix GARCH recursion.

    This function computes the conditional covariance matrices for the Matrix GARCH model
    using the recursion:
    vech(H_t) = vech(C) + sum_{i=1}^p A_i vech(ε_{t-i}ε_{t-i}') + sum_{j=1}^q B_j vech(H_{t-j})

    Args:
        data: Residual data matrix (T x n_assets)
        C: Constant vector in vech form (n_vech)
        A: ARCH coefficient matrices in vech form, flattened (p * n_vech)
        B: GARCH coefficient matrices in vech form, flattened (q * n_vech)
        H_t: Pre-allocated array for conditional covariances (n_assets x n_assets x T)
        T: Number of time periods
        n_assets: Number of assets
        p: ARCH order
        q: GARCH order

    Returns:
        Conditional covariance matrices (n_assets x n_assets x T)
    """
    n_vech = n_assets * (n_assets + 1) // 2

    # Pre-allocate array for vech(H_t)
    vech_H = np.zeros((T, n_vech))

    # Initialize with unconditional covariance (backcast)
    # H_0 is already set in H_t[0] from the input

    # Convert H_t[0] to vech form
    idx = 0
    for i in range(n_assets):
        for j in range(i + 1):
            vech_H[0, idx] = H_t[i, j, 0]
            idx += 1

    # Pre-compute vech(ε_t * ε_t') for all t
    vech_eps_eps = np.zeros((T, n_vech))
    for t in range(T):
        idx = 0
        for i in range(n_assets):
            for j in range(i + 1):
                vech_eps_eps[t, idx] = data[t, i] * data[t, j]
                idx += 1

    # Main recursion
    for t in range(1, T):
        # Start with constant term
        for i in range(n_vech):
            vech_H[t, i] = C[i]

        # Add ARCH terms
        for i in range(min(t, p)):
            for j in range(n_vech):
                vech_H[t, j] += A[i * n_vech + j] * vech_eps_eps[t-i-1, j]

        # Add GARCH terms
        for i in range(min(t, q)):
            for j in range(n_vech):
                vech_H[t, j] += B[i * n_vech + j] * vech_H[t-i-1, j]

        # Convert vech_H[t] to full matrix H_t[:,:,t]
        idx = 0
        for i in range(n_assets):
            for j in range(i + 1):
                H_t[i, j, t] = vech_H[t, idx]
                H_t[j, i, t] = vech_H[t, idx]  # Ensure symmetry
                idx += 1

    return H_t


class MatrixGARCHModel(CovarianceModelBase):
    """
    Matrix GARCH model for multivariate volatility.

    This class implements the Matrix GARCH model, a direct multivariate extension
    of the univariate GARCH that models covariance matrices directly. The model
    follows the recursion:

    vech(H_t) = vech(C) + sum_{i=1}^p A_i vech(ε_{t-i}ε_{t-i}') + sum_{j=1}^q B_j vech(H_{t-j})

    Attributes:
        name: Model name
        n_assets: Number of assets
        p: ARCH order
        q: GARCH order
        parameters: Model parameters if fitted
        _conditional_covariances: Conditional covariance matrices if fitted
        _conditional_correlations: Conditional correlation matrices if fitted
        _residuals: Residuals used for model fitting
        _backcast: Matrix used for initializing the covariance process
    """

    def __init__(
        self,
        n_assets: Optional[int] = None,
        p: int = 1,
        q: int = 1,
        name: str = "Matrix-GARCH"
    ):
        """
        Initialize the Matrix GARCH model.

        Args:
            n_assets: Number of assets (if None, determined from data)
            p: ARCH order (default: 1)
            q: GARCH order (default: 1)
            name: Model name (default: "Matrix-GARCH")
        """
        super().__init__(name=name, n_assets=n_assets)

        self.p = p
        self.q = q

        self._parameters: Optional[MatrixGARCHParameters] = None
        self._conditional_covariances: Optional[np.ndarray] = None
        self._conditional_correlations: Optional[np.ndarray] = None
        self._residuals: Optional[np.ndarray] = None
        self._backcast: Optional[np.ndarray] = None
        self._fitted = False

    def fit(
        self,
        data: np.ndarray,
        starting_values: Optional[Union[np.ndarray, MatrixGARCHParameters]] = None,
        backcast: Optional[np.ndarray] = None,
        method: str = "SLSQP",
        options: Optional[Dict[str, Any]] = None,
        constraints: Optional[List[Dict[str, Any]]] = None,
        callback: Optional[ProgressCallback] = None,
        **kwargs: Any
    ) -> MultivariateVolatilityResult:
        """
        Fit the Matrix GARCH model to the provided data.

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
        self.validate_data(data)
        T, n_assets = data.shape

        # Update n_assets if not already set
        if self._n_assets is None:
            self._n_assets = n_assets

        # Compute backcast if not provided
        if backcast is None:
            backcast = self.compute_backcast(data)

        self._backcast = backcast

        # Generate starting values if not provided
        if starting_values is None:
            starting_values = initialize_parameters(
                data,
                model_type="MATRIX-GARCH",
                p=self.p,
                q=self.q,
                n_assets=n_assets
            )

        # Convert starting_values to array if it's a parameter object
        if isinstance(starting_values, MatrixGARCHParameters):
            starting_values = starting_values.to_array()

        # Set up optimization options
        if options is None:
            options = {}

        # Set up constraints if not provided
        if constraints is None:
            constraints = self._create_constraints()

        # Set up bounds
        bounds = self._create_bounds()

        # Define objective function for optimization
        def objective(params: np.ndarray) -> float:
            return self._objective_function(params, data, backcast)

        # Run optimization
        optimization_result = optimize.minimize(
            objective,
            starting_values,
            method=method,
            bounds=bounds,
            constraints=constraints,
            options=options,
            callback=callback
        )

        # Check if optimization converged
        if not optimization_result.success:
            warnings.warn(
                f"Optimization did not converge: {optimization_result.message}",
                UserWarning
            )

        # Extract optimized parameters
        optimized_params = optimization_result.x

        # Convert to parameter object
        self._parameters = MatrixGARCHParameters.from_array(
            optimized_params,
            p=self.p,
            q=self.q,
            n_assets=n_assets
        )

        # Compute conditional covariances
        self._conditional_covariances = self.compute_covariance(
            self._parameters,
            data,
            backcast=backcast
        )

        # Compute conditional correlations
        self._conditional_correlations = self.compute_correlation(
            self._parameters,
            data,
            backcast=backcast
        )

        # Store residuals
        self._residuals = data

        # Compute log-likelihood
        loglik = -optimization_result.fun

        # Compute standard errors
        std_errors = self._compute_std_errors(
            optimized_params,
            data,
            backcast=backcast
        )

        # Mark as fitted
        self._fitted = True

        # Create result object
        result = self._create_result_object(
            parameters=self._parameters,
            data=data,
            covariance=self._conditional_covariances,
            correlation=self._conditional_correlations,
            loglikelihood=loglik,
            std_errors=std_errors,
            iterations=optimization_result.nit,
            convergence=optimization_result.success,
            optimization_message=optimization_result.message
        )

        return result

    async def fit_async(
        self,
        data: np.ndarray,
        starting_values: Optional[Union[np.ndarray, MatrixGARCHParameters]] = None,
        backcast: Optional[np.ndarray] = None,
        method: str = "SLSQP",
        options: Optional[Dict[str, Any]] = None,
        constraints: Optional[List[Dict[str, Any]]] = None,
        callback: Optional[AsyncProgressCallback] = None,
        **kwargs: Any
    ) -> MultivariateVolatilityResult:
        """
        Asynchronously fit the Matrix GARCH model to the provided data.

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
        # Use the base class implementation which handles the async logic
        return await super().fit_async(
            data=data,
            starting_values=starting_values,
            backcast=backcast,
            method=method,
            options=options,
            constraints=constraints,
            callback=callback,
            **kwargs
        )

    def compute_covariance(
        self,
        parameters: MatrixGARCHParameters,
        data: np.ndarray,
        sigma: Optional[np.ndarray] = None,
        backcast: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute conditional covariance matrices for the given parameters and data.

        Args:
            parameters: Model parameters
            data: Input data (typically residuals)
            sigma: Pre-allocated array for conditional covariances
            backcast: Matrix to use for initializing the covariance process

        Returns:
            np.ndarray: Conditional covariance matrices (n_assets x n_assets x T)
        """
        # Get dimensions
        T, n_assets = data.shape
        n_vech = n_assets * (n_assets + 1) // 2

        # Compute backcast if not provided
        if backcast is None:
            if self._backcast is not None:
                backcast = self._backcast
            else:
                backcast = self.compute_backcast(data)

        # Pre-allocate array for conditional covariances if not provided
        if sigma is None:
            sigma = np.zeros((n_assets, n_assets, T))

        # Set initial covariance matrix
        sigma[:, :, 0] = backcast

        # Flatten A and B parameters for Numba
        A_flat = np.concatenate(parameters.A)
        B_flat = np.concatenate(parameters.B)

        # Use Numba-accelerated recursion
        sigma = matrix_garch_recursion(
            data,
            parameters.C,
            A_flat,
            B_flat,
            sigma,
            T,
            n_assets,
            parameters.p,
            parameters.q
        )

        return sigma

    def simulate(
        self,
        n_periods: int,
        burn: int = 0,
        initial_value: Optional[np.ndarray] = None,
        random_state: Optional[Union[int, np.random.Generator]] = None,
        return_covariances: bool = False,
        **kwargs: Any
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Simulate data from the Matrix GARCH model.

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

        # Set up random number generator
        if isinstance(random_state, np.random.Generator):
            rng = random_state
        else:
            rng = np.random.default_rng(random_state)

        # Get dimensions
        n_assets = self._n_assets
        if n_assets is None:
            raise RuntimeError("Number of assets is not set.")

        # Set initial covariance matrix
        if initial_value is None:
            if self._conditional_covariances is not None:
                initial_value = self._conditional_covariances[:, :, -1]
            elif self._backcast is not None:
                initial_value = self._backcast
            else:
                # Use identity matrix as fallback
                initial_value = np.eye(n_assets)

        # Ensure initial_value is positive definite
        initial_value = ensure_positive_definite(initial_value)

        # Pre-allocate arrays
        total_periods = n_periods + burn
        simulated_data = np.zeros((total_periods, n_assets))
        simulated_covariances = np.zeros((n_assets, n_assets, total_periods))

        # Set initial covariance matrix
        simulated_covariances[:, :, 0] = initial_value

        # Generate initial data
        chol = np.linalg.cholesky(initial_value)
        simulated_data[0, :] = chol @ rng.standard_normal(n_assets)

        # Simulate data
        for t in range(1, total_periods):
            # Compute conditional covariance matrix
            vech_H_t = np.copy(self._parameters.C)

            # Add ARCH terms
            for i in range(min(t, self.p)):
                eps_eps_t_i = np.outer(simulated_data[t-i-1, :], simulated_data[t-i-1, :])
                vech_eps_eps_t_i = vech(eps_eps_t_i)
                vech_H_t += self._parameters.A[i] * vech_eps_eps_t_i

            # Add GARCH terms
            for j in range(min(t, self.q)):
                vech_H_t_j = vech(simulated_covariances[:, :, t-j-1])
                vech_H_t += self._parameters.B[j] * vech_H_t_j

            # Convert vech_H_t to full matrix
            H_t = ivech(vech_H_t)

            # Ensure H_t is symmetric
            H_t = ensure_symmetric(H_t)

            # Ensure H_t is positive definite
            H_t = ensure_positive_definite(H_t)

            # Store covariance matrix
            simulated_covariances[:, :, t] = H_t

            # Generate data
            chol = np.linalg.cholesky(H_t)
            simulated_data[t, :] = chol @ rng.standard_normal(n_assets)

        # Discard burn-in period
        if burn > 0:
            simulated_data = simulated_data[burn:]
            simulated_covariances = simulated_covariances[:, :, burn:]

        if return_covariances:
            return simulated_data, simulated_covariances
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
        """
        Asynchronously simulate data from the Matrix GARCH model.

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
        # Use the base class implementation which handles the async logic
        return await super().simulate_async(
            n_periods=n_periods,
            burn=burn,
            initial_value=initial_value,
            random_state=random_state,
            return_covariances=return_covariances,
            callback=callback,
            **kwargs
        )

    def forecast(
        self,
        steps: int,
        initial_value: Optional[np.ndarray] = None,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Forecast conditional covariance matrices.

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

        # Get dimensions
        n_assets = self._n_assets
        if n_assets is None:
            raise RuntimeError("Number of assets is not set.")

        # Set initial values
        if initial_value is None:
            if self._conditional_covariances is not None:
                initial_value = self._conditional_covariances[:, :, -1]
            elif self._backcast is not None:
                initial_value = self._backcast
            else:
                # Use identity matrix as fallback
                initial_value = np.eye(n_assets)

        # Ensure initial_value is positive definite
        initial_value = ensure_positive_definite(initial_value)

        # Pre-allocate array for forecasted covariances
        forecasted_covariances = np.zeros((n_assets, n_assets, steps))

        # Set initial covariance matrix
        forecasted_covariances[:, :, 0] = initial_value

        # If we have residuals, use them for the ARCH terms
        if self._residuals is not None:
            residuals = self._residuals
            T = residuals.shape[0]
        else:
            # If no residuals available, use zeros
            T = 0
            residuals = np.zeros((0, n_assets))

        # Compute forecasts
        for t in range(1, steps):
            # Start with constant term
            vech_H_t = np.copy(self._parameters.C)

            # Add ARCH terms
            for i in range(self.p):
                if t-i-1 < 0 and T+t-i-1 >= 0:
                    # Use historical residuals
                    eps_t_i = residuals[T+t-i-1, :]
                    eps_eps_t_i = np.outer(eps_t_i, eps_t_i)
                    vech_eps_eps_t_i = vech(eps_eps_t_i)
                    vech_H_t += self._parameters.A[i] * vech_eps_eps_t_i
                else:
                    # For future periods, use unconditional expectation
                    # E[ε_t ε_t'] = H_t
                    if t-i-1 >= 0:
                        vech_H_t_i = vech(forecasted_covariances[:, :, t-i-1])
                        vech_H_t += self._parameters.A[i] * vech_H_t_i

            # Add GARCH terms
            for j in range(self.q):
                if t-j-1 >= 0:
                    vech_H_t_j = vech(forecasted_covariances[:, :, t-j-1])
                    vech_H_t += self._parameters.B[j] * vech_H_t_j
                else:
                    # If we don't have enough forecasted values yet,
                    # use the initial value
                    vech_H_0 = vech(initial_value)
                    vech_H_t += self._parameters.B[j] * vech_H_0

            # Convert vech_H_t to full matrix
            H_t = ivech(vech_H_t)

            # Ensure H_t is symmetric
            H_t = ensure_symmetric(H_t)

            # Ensure H_t is positive definite
            H_t = ensure_positive_definite(H_t)

            # Store forecasted covariance matrix
            forecasted_covariances[:, :, t] = H_t

        return forecasted_covariances

    async def forecast_async(
        self,
        steps: int,
        initial_value: Optional[np.ndarray] = None,
        callback: Optional[AsyncProgressCallback] = None,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Asynchronously forecast conditional covariance matrices.

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
        # Use the base class implementation which handles the async logic
        return await super().forecast_async(
            steps=steps,
            initial_value=initial_value,
            callback=callback,
            **kwargs
        )

    def _array_to_parameters(self, array: np.ndarray) -> MatrixGARCHParameters:
        """
        Convert a parameter array to a parameter object.

        Args:
            array: Parameter array

        Returns:
            MatrixGARCHParameters: Parameter object
        """
        n_assets = self._n_assets
        if n_assets is None:
            raise RuntimeError("Number of assets is not set.")

        return MatrixGARCHParameters.from_array(
            array,
            p=self.p,
            q=self.q,
            n_assets=n_assets
        )

    def _parameters_to_array(self, parameters: MatrixGARCHParameters) -> np.ndarray:
        """
        Convert a parameter object to a parameter array.

        Args:
            parameters: Parameter object

        Returns:
            np.ndarray: Parameter array
        """
        return parameters.to_array()

    def _create_constraints(self) -> List[Dict[str, Any]]:
        """
        Create constraints for optimization.

        Returns:
            List[Dict[str, Any]]: List of constraints for optimization
        """
        # For Matrix GARCH, we need to ensure stationarity
        # This means the sum of A and B coefficients should be less than 1

        n_assets = self._n_assets
        if n_assets is None:
            raise RuntimeError("Number of assets is not set.")

        n_vech = n_assets * (n_assets + 1) // 2
        n_params = n_vech * (1 + self.p + self.q)

        # Create constraint function
        def constraint_func(params: np.ndarray) -> float:
            # Extract A and B parameters
            A_start = n_vech
            A_end = n_vech * (1 + self.p)
            B_start = A_end
            B_end = n_vech * (1 + self.p + self.q)

            A_params = params[A_start:A_end]
            B_params = params[B_start:B_end]

            # Compute sum of A and B coefficients
            sum_A_B = np.sum(A_params) + np.sum(B_params)

            # Constraint: sum_A_B < 1
            return 1 - sum_A_B - 1e-6  # Small buffer for numerical stability

        # Create constraint dictionary
        constraint = {
            'type': 'ineq',
            'fun': constraint_func
        }

        return [constraint]

    def _create_bounds(self) -> List[Tuple[float, float]]:
        """
        Create bounds for optimization parameters.

        Returns:
            List[Tuple[float, float]]: List of (lower, upper) bounds for each parameter
        """
        n_assets = self._n_assets
        if n_assets is None:
            raise RuntimeError("Number of assets is not set.")

        n_vech = n_assets * (n_assets + 1) // 2
        n_params = n_vech * (1 + self.p + self.q)

        # Create bounds
        bounds = []

        # Bounds for C (constant term)
        for i in range(n_vech):
            bounds.append((1e-8, None))  # Positive but small lower bound

        # Bounds for A (ARCH terms)
        for i in range(self.p * n_vech):
            bounds.append((0, 0.5))  # Between 0 and 0.5

        # Bounds for B (GARCH terms)
        for i in range(self.q * n_vech):
            bounds.append((0, 0.999))  # Between 0 and 0.999

        return bounds

    def _create_starting_values(self, data: np.ndarray) -> np.ndarray:
        """
        Create starting values for optimization.

        Args:
            data: Input data (typically residuals)

        Returns:
            np.ndarray: Starting values for optimization
        """
        return initialize_parameters(
            data,
            model_type="MATRIX-GARCH",
            p=self.p,
            q=self.q
        )

    def _compute_persistence(self) -> float:
        """
        Compute the persistence of the model.

        Returns:
            float: Persistence value

        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self._fitted or self._parameters is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        # Compute persistence as sum of A and B coefficients
        persistence = np.sum([np.sum(A_i) for A_i in self._parameters.A]) + \
            np.sum([np.sum(B_i) for B_i in self._parameters.B])

        return persistence

    def _compute_unconditional_covariance(self) -> Optional[np.ndarray]:
        """
        Compute the unconditional covariance matrix implied by the model.

        Returns:
            Optional[np.ndarray]: Unconditional covariance matrix

        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self._fitted or self._parameters is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        # Compute persistence
        persistence = self._compute_persistence()

        # Check if model is stationary
        if persistence >= 1:
            warnings.warn(
                f"Model is not stationary (persistence = {persistence}). "
                "Unconditional covariance is not defined.",
                UserWarning
            )
            return None

        # Compute unconditional covariance
        # For Matrix GARCH, unconditional covariance is:
        # vech(H) = (I - sum(A_i) - sum(B_j))^(-1) * vech(C)

        n_assets = self._n_assets
        if n_assets is None:
            raise RuntimeError("Number of assets is not set.")

        n_vech = n_assets * (n_assets + 1) // 2

        # Compute sum of A and B coefficients
        sum_A_B = np.zeros(n_vech)
        for A_i in self._parameters.A:
            sum_A_B += A_i
        for B_j in self._parameters.B:
            sum_A_B += B_j

        # Compute (I - sum(A_i) - sum(B_j))^(-1) * vech(C)
        I = np.eye(n_vech)
        try:
            vech_H = np.linalg.solve(I - np.diag(sum_A_B), self._parameters.C)

            # Convert vech_H to full matrix
            H = ivech(vech_H)

            # Ensure H is symmetric
            H = ensure_symmetric(H)

            # Ensure H is positive definite
            H = ensure_positive_definite(H)

            return H
        except np.linalg.LinAlgError:
            warnings.warn(
                "Failed to compute unconditional covariance matrix. "
                "The model may be close to non-stationarity.",
                UserWarning
            )
            return None

    def __str__(self) -> str:
        """
        Return a string representation of the model.

        Returns:
            str: String representation
        """
        if not self._fitted or self._parameters is None:
            return f"Matrix GARCH({self.p},{self.q}) model (not fitted)"

        n_assets = self._n_assets
        if n_assets is None:
            return f"Matrix GARCH({self.p},{self.q}) model (fitted, unknown dimensions)"

        persistence = self._compute_persistence()

        return (
            f"Matrix GARCH({self.p},{self.q}) model for {n_assets} assets\n"
            f"Persistence: {persistence:.4f}\n"
            f"Number of parameters: {len(self._parameters.to_array())}"
        )

    def __repr__(self) -> str:
        """
        Return a string representation of the model.

        Returns:
            str: String representation
        """
        return self.__str__()

    def summary(self) -> str:
        """
        Generate a summary of the model.

        Returns:
            str: Summary string

        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self._fitted or self._parameters is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        n_assets = self._n_assets
        if n_assets is None:
            raise RuntimeError("Number of assets is not set.")

        # Create summary string
        summary_str = f"Matrix GARCH({self.p},{self.q}) Model Summary\n"
        summary_str += "=" * 50 + "\n\n"

        # Add model information
        summary_str += f"Number of assets: {n_assets}\n"
        summary_str += f"ARCH order (p): {self.p}\n"
        summary_str += f"GARCH order (q): {self.q}\n"

        if self._residuals is not None:
            summary_str += f"Number of observations: {self._residuals.shape[0]}\n"

        # Add persistence information
        persistence = self._compute_persistence()
        summary_str += f"Persistence: {persistence:.4f}\n"

        # Add half-life information if model is stationary
        if persistence < 1:
            half_life = compute_half_life(persistence)
            summary_str += f"Half-life of shocks: {half_life:.2f} periods\n"
        else:
            summary_str += "Half-life of shocks: Infinite (non-stationary model)\n"

        summary_str += "\n"

        # Add parameter information
        summary_str += "Parameter Estimates:\n"
        summary_str += "-" * 50 + "\n"

        # Format parameters
        param_array = self._parameters.to_array()
        n_vech = n_assets * (n_assets + 1) // 2

        # Extract C parameters
        C = param_array[:n_vech]

        # Extract A parameters
        A = []
        for i in range(self.p):
            start = n_vech * (1 + i)
            end = n_vech * (2 + i)
            A.append(param_array[start:end])

        # Extract B parameters
        B = []
        for i in range(self.q):
            start = n_vech * (1 + self.p + i)
            end = n_vech * (2 + self.p + i)
            B.append(param_array[start:end])

        # Format C parameters
        summary_str += "Constant (C) parameters:\n"
        idx = 0
        for i in range(n_assets):
            for j in range(i + 1):
                summary_str += f"  C[{i+1},{j+1}] = {C[idx]:.6f}\n"
                idx += 1

        summary_str += "\n"

        # Format A parameters
        for k in range(self.p):
            summary_str += f"ARCH (A{k+1}) parameters:\n"
            idx = 0
            for i in range(n_assets):
                for j in range(i + 1):
                    summary_str += f"  A{k+1}[{i+1},{j+1}] = {A[k][idx]:.6f}\n"
                    idx += 1
            summary_str += "\n"

        # Format B parameters
        for k in range(self.q):
            summary_str += f"GARCH (B{k+1}) parameters:\n"
            idx = 0
            for i in range(n_assets):
                for j in range(i + 1):
                    summary_str += f"  B{k+1}[{i+1},{j+1}] = {B[k][idx]:.6f}\n"
                    idx += 1
            summary_str += "\n"

        return summary_str
