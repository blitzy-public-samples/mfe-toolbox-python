# mfe/models/multivariate/bekk.py

"""
BEKK (Baba-Engle-Kraft-Kroner) multivariate GARCH model implementation.

This module implements the BEKK multivariate GARCH model, which provides a positive
definite parameterization for multivariate volatility modeling. The BEKK model is
essential for ensuring valid covariance matrices in portfolio risk assessment and
cross-asset analysis.

The implementation supports both symmetric and asymmetric BEKK specifications, with
comprehensive parameter validation, optimization, and simulation capabilities. The
model is implemented using NumPy for efficient matrix operations with Numba
acceleration for performance-critical calculations.

References:
    Engle, R. F., & Kroner, K. F. (1995). Multivariate simultaneous generalized ARCH.
    Econometric Theory, 11(1), 122-150.
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import (
    Any, Dict, List, Literal, Optional, Tuple, Union, cast
)

import numpy as np
from scipy import optimize

from mfe.core.base import ModelBase
from mfe.core.parameters import (
    MultivariateVolatilityParameters, ParameterBase, ParameterError,
    validate_positive_definite
)
from mfe.core.results import MultivariateVolatilityResult
from mfe.core.types import (
    AsyncProgressCallback, CovarianceMatrix, Matrix, ProgressCallback, Vector
)
from mfe.models.multivariate.base import CovarianceModelBase
from mfe.models.multivariate._numba_core import (
    bekk_recursion, bekk_recursion_asymmetric, bekk_simulate, bekk_simulate_asymmetric
)
from mfe.utils.matrix_ops import vech, ivech


logger = logging.getLogger(__name__)


@dataclass
class BEKKParameters(MultivariateVolatilityParameters):
    """Parameters for the BEKK multivariate GARCH model.

    This class contains the parameters for the BEKK model, including the constant
    term (C), ARCH coefficient matrix (A), and GARCH coefficient matrix (B).
    For the asymmetric BEKK model, it also includes the asymmetry coefficient
    matrix (G).

    Attributes:
        C: Constant term matrix (lower triangular)
        A: ARCH coefficient matrix
        B: GARCH coefficient matrix
        G: Asymmetry coefficient matrix (for asymmetric BEKK)
        asymmetric: Whether the model is asymmetric
    """

    C: np.ndarray
    A: np.ndarray
    B: np.ndarray
    G: Optional[np.ndarray] = None
    asymmetric: bool = False

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        # Ensure arrays are NumPy arrays
        self.C = np.asarray(self.C)
        self.A = np.asarray(self.A)
        self.B = np.asarray(self.B)

        if self.G is not None:
            self.G = np.asarray(self.G)
            self.asymmetric = True
        elif self.asymmetric:
            raise ParameterError("G must be provided for asymmetric BEKK model")

        # Validate parameters
        self.validate()

    def validate(self) -> None:
        """Validate BEKK parameter constraints.

        Raises:
            ParameterError: If parameter constraints are violated
        """
        # Check dimensions
        if self.C.ndim != 2:
            raise ParameterError(f"C must be a 2D array, got {self.C.ndim}D")

        if self.A.ndim != 2:
            raise ParameterError(f"A must be a 2D array, got {self.A.ndim}D")

        if self.B.ndim != 2:
            raise ParameterError(f"B must be a 2D array, got {self.B.ndim}D")

        # Check square matrices
        n = self.C.shape[0]

        if self.C.shape[1] != n:
            raise ParameterError(
                f"C must be square, got shape {self.C.shape}"
            )

        if self.A.shape != (n, n):
            raise ParameterError(
                f"A must have shape ({n}, {n}), got {self.A.shape}"
            )

        if self.B.shape != (n, n):
            raise ParameterError(
                f"B must have shape ({n}, {n}), got {self.B.shape}"
            )

        # Check if C is lower triangular
        if not np.allclose(self.C, np.tril(self.C)):
            raise ParameterError("C must be lower triangular")

        # For asymmetric model, check G
        if self.asymmetric:
            if self.G is None:
                raise ParameterError("G must be provided for asymmetric BEKK model")

            if self.G.ndim != 2:
                raise ParameterError(f"G must be a 2D array, got {self.G.ndim}D")

            if self.G.shape != (n, n):
                raise ParameterError(
                    f"G must have shape ({n}, {n}), got {self.G.shape}"
                )

        # Check stationarity condition
        # For BEKK, this is complex and involves eigenvalues of a larger matrix
        # We'll implement a simplified check based on the sum of squared elements
        a_norm = np.sum(self.A ** 2)
        b_norm = np.sum(self.B ** 2)
        g_norm = np.sum(self.G ** 2) if self.G is not None else 0

        if a_norm + b_norm + 0.5 * g_norm >= 1:
            warnings.warn(
                "BEKK stationarity condition may be violated. "
                f"Sum of squared elements: A={a_norm:.4f}, B={b_norm:.4f}, "
                f"G={g_norm:.4f}, total={a_norm + b_norm + 0.5 * g_norm:.4f}",
                UserWarning
            )

    def to_array(self) -> np.ndarray:
        """Convert parameters to a NumPy array.

        Returns:
            np.ndarray: Array representation of parameters
        """
        # For C, we only need the lower triangular part
        c_vech = vech(self.C)

        # Flatten A and B
        a_flat = self.A.flatten()
        b_flat = self.B.flatten()

        if self.asymmetric and self.G is not None:
            # Flatten G
            g_flat = self.G.flatten()
            return np.concatenate([c_vech, a_flat, b_flat, g_flat])

        return np.concatenate([c_vech, a_flat, b_flat])

    @classmethod
    def from_array(cls, array: np.ndarray, n_assets: int,
                   asymmetric: bool = False, **kwargs: Any) -> 'BEKKParameters':
        """Create parameters from a NumPy array.

        Args:
            array: Array representation of parameters
            n_assets: Number of assets
            asymmetric: Whether the model is asymmetric
            **kwargs: Additional keyword arguments for parameter creation

        Returns:
            BEKKParameters: Parameter object

        Raises:
            ValueError: If the array length doesn't match the expected length
        """
        # Calculate expected array length
        c_len = n_assets * (n_assets + 1) // 2  # Length of vech(C)
        a_len = n_assets * n_assets  # Length of A.flatten()
        b_len = n_assets * n_assets  # Length of B.flatten()
        g_len = n_assets * n_assets if asymmetric else 0  # Length of G.flatten()

        expected_len = c_len + a_len + b_len + g_len

        if len(array) != expected_len:
            raise ValueError(
                f"Array length ({len(array)}) doesn't match expected length ({expected_len})"
            )

        # Extract parameters
        c_vech = array[:c_len]
        a_flat = array[c_len:c_len + a_len]
        b_flat = array[c_len + a_len:c_len + a_len + b_len]

        # Reconstruct matrices
        C = ivech(c_vech)
        A = a_flat.reshape(n_assets, n_assets)
        B = b_flat.reshape(n_assets, n_assets)

        if asymmetric:
            g_flat = array[c_len + a_len + b_len:]
            G = g_flat.reshape(n_assets, n_assets)
            return cls(C=C, A=A, B=B, G=G, asymmetric=True)

        return cls(C=C, A=A, B=B, asymmetric=False)

    def transform(self) -> np.ndarray:
        """Transform parameters to unconstrained space for optimization.

        Returns:
            np.ndarray: Transformed parameters in unconstrained space
        """
        # For BEKK, we don't need complex transformations
        # We just need to ensure C is lower triangular, which is handled in from_array
        return self.to_array()

    @classmethod
    def inverse_transform(cls, array: np.ndarray, n_assets: int,
                          asymmetric: bool = False, **kwargs: Any) -> 'BEKKParameters':
        """Transform parameters from unconstrained space back to constrained space.

        Args:
            array: Parameters in unconstrained space
            n_assets: Number of assets
            asymmetric: Whether the model is asymmetric
            **kwargs: Additional keyword arguments for parameter creation

        Returns:
            BEKKParameters: Parameter object with constrained parameters
        """
        # For BEKK, the transformation is the same as from_array
        return cls.from_array(array, n_assets, asymmetric, **kwargs)


class BEKKModel(CovarianceModelBase):
    """BEKK multivariate GARCH model implementation.

    The BEKK model provides a positive definite parameterization for multivariate
    volatility modeling, ensuring valid covariance matrices for portfolio risk
    assessment and cross-asset analysis.

    Attributes:
        name: Name of the model
        n_assets: Number of assets in the model
        asymmetric: Whether the model is asymmetric
        parameters: Model parameters if fitted
        _conditional_covariances: Conditional covariance matrices if fitted
        _conditional_correlations: Conditional correlation matrices if fitted
        _residuals: Residuals used for model fitting
        _backcast: Matrix used for initializing the covariance process
    """

    def __init__(
        self,
        n_assets: Optional[int] = None,
        asymmetric: bool = False,
        name: Optional[str] = None
    ):
        """Initialize the BEKK model.

        Args:
            n_assets: Number of assets in the model (if None, determined from data)
            asymmetric: Whether to use the asymmetric BEKK specification
            name: Name of the model (if None, uses default name)
        """
        if name is None:
            name = f"{'Asymmetric ' if asymmetric else ''}BEKK"

        super().__init__(name=name, n_assets=n_assets)

        self.asymmetric = asymmetric
        self._parameters: Optional[BEKKParameters] = None
        self._conditional_covariances: Optional[np.ndarray] = None
        self._conditional_correlations: Optional[np.ndarray] = None
        self._residuals: Optional[np.ndarray] = None
        self._backcast: Optional[np.ndarray] = None

    def fit(
        self,
        data: np.ndarray,
        starting_values: Optional[Union[np.ndarray, BEKKParameters]] = None,
        backcast: Optional[np.ndarray] = None,
        method: str = "SLSQP",
        options: Optional[Dict[str, Any]] = None,
        constraints: Optional[List[Dict[str, Any]]] = None,
        callback: Optional[ProgressCallback] = None,
        **kwargs: Any
    ) -> MultivariateVolatilityResult:
        """Fit the BEKK model to the provided data.

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

        # Store residuals
        self._residuals = data

        # Compute backcast if not provided
        if backcast is None:
            self._backcast = self.compute_backcast(data)
        else:
            self._backcast = backcast

            # Validate backcast dimensions
            if self._backcast.shape != (self._n_assets, self._n_assets):
                raise ValueError(
                    f"Backcast shape ({self._backcast.shape}) doesn't match "
                    f"expected shape ({self._n_assets}, {self._n_assets})"
                )

        # Create starting values if not provided
        if starting_values is None:
            param_array = self._create_starting_values(data)
        elif isinstance(starting_values, BEKKParameters):
            param_array = starting_values.to_array()
        else:
            param_array = starting_values

        # Create optimization options
        if options is None:
            options = {}

        # Default options for SLSQP
        if method == "SLSQP" and "maxiter" not in options:
            options["maxiter"] = 1000

        # Create constraints if not provided
        if constraints is None:
            constraints = self._create_constraints()

        # Create bounds
        bounds = self._create_bounds()

        # Define progress tracking
        iteration = [0]

        def callback_wrapper(xk: np.ndarray) -> None:
            """Wrapper for the callback function to track iterations."""
            iteration[0] += 1
            if callback is not None:
                progress = min(iteration[0] / 1000, 0.99)  # Cap at 99%
                callback(progress, f"Iteration {iteration[0]}")

        # Run optimization
        try:
            result = optimize.minimize(
                self._objective_function,
                param_array,
                args=(data, self._backcast),
                method=method,
                bounds=bounds,
                constraints=constraints,
                options=options,
                callback=callback_wrapper if callback else None
            )

            # Check convergence
            if not result.success:
                warnings.warn(
                    f"Optimization did not converge: {result.message}",
                    UserWarning
                )

            # Convert parameters to parameter object
            self._parameters = BEKKParameters.from_array(
                result.x, self._n_assets, self.asymmetric
            )

            # Compute conditional covariances
            self._conditional_covariances = self.compute_covariance(
                self._parameters, data, backcast=self._backcast
            )

            # Compute conditional correlations
            self._conditional_correlations = self.compute_correlation(
                self._parameters, data, backcast=self._backcast
            )

            # Compute standard errors
            std_errors = self._compute_std_errors(result.x, data, self._backcast)

            # Mark as fitted
            self._fitted = True

            # Report final progress
            if callback is not None:
                callback(1.0, "Estimation complete")

            # Create result object
            return self._create_result_object(
                parameters=self._parameters,
                data=data,
                covariance=self._conditional_covariances,
                correlation=self._conditional_correlations,
                loglikelihood=-result.fun,
                std_errors=std_errors,
                iterations=result.nit,
                convergence=result.success,
                optimization_message=result.message
            )

        except Exception as e:
            # Log the error
            logger.error(f"Error during BEKK model estimation: {str(e)}")
            raise RuntimeError(f"BEKK model estimation failed: {str(e)}") from e

    async def fit_async(
        self,
        data: np.ndarray,
        starting_values: Optional[Union[np.ndarray, BEKKParameters]] = None,
        backcast: Optional[np.ndarray] = None,
        method: str = "SLSQP",
        options: Optional[Dict[str, Any]] = None,
        constraints: Optional[List[Dict[str, Any]]] = None,
        callback: Optional[AsyncProgressCallback] = None,
        **kwargs: Any
    ) -> MultivariateVolatilityResult:
        """Asynchronously fit the BEKK model to the provided data.

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
        # Use the base class implementation which handles the async pattern
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

    def simulate(
        self,
        n_periods: int,
        burn: int = 0,
        initial_value: Optional[np.ndarray] = None,
        random_state: Optional[Union[int, np.random.Generator]] = None,
        return_covariances: bool = False,
        **kwargs: Any
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Simulate data from the BEKK model.

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

        # Set up initial covariance matrix
        if initial_value is None:
            if self._backcast is not None:
                initial_value = self._backcast
            else:
                # Use unconditional covariance
                unconditional_cov = self._compute_unconditional_covariance()
                if unconditional_cov is not None:
                    initial_value = unconditional_cov
                else:
                    # Use identity matrix as fallback
                    initial_value = np.eye(self._n_assets)

        # Validate initial value dimensions
        if initial_value.shape != (self._n_assets, self._n_assets):
            raise ValueError(
                f"Initial value shape ({initial_value.shape}) doesn't match "
                f"expected shape ({self._n_assets}, {self._n_assets})"
            )

        # Ensure initial value is positive definite
        try:
            np.linalg.cholesky(initial_value)
        except np.linalg.LinAlgError:
            # If not positive definite, add a small diagonal component
            min_eigenvalue = np.min(np.linalg.eigvalsh(initial_value))
            if min_eigenvalue <= 0:
                initial_value += np.eye(self._n_assets) * (abs(min_eigenvalue) + 1e-6)

        # Extract parameters
        C = self._parameters.C
        A = self._parameters.A
        B = self._parameters.B
        G = self._parameters.G if self._parameters.asymmetric else None

        # Simulate data
        total_periods = n_periods + burn

        if self.asymmetric and G is not None:
            # Use asymmetric simulation
            simulated_data, simulated_covs = bekk_simulate_asymmetric(
                total_periods, C, A, B, G, initial_value, rng
            )
        else:
            # Use symmetric simulation
            simulated_data, simulated_covs = bekk_simulate(
                total_periods, C, A, B, initial_value, rng
            )

        # Discard burn-in periods
        if burn > 0:
            simulated_data = simulated_data[burn:]
            simulated_covs = simulated_covs[:, :, burn:]

        if return_covariances:
            return simulated_data, simulated_covs

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
        """Asynchronously simulate data from the BEKK model.

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
        # Use the base class implementation which handles the async pattern
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

        # Set up initial covariance matrix
        if initial_value is None:
            if self._conditional_covariances is not None:
                # Use the last estimated covariance matrix
                initial_value = self._conditional_covariances[:, :, -1]
            elif self._backcast is not None:
                initial_value = self._backcast
            else:
                # Use unconditional covariance
                unconditional_cov = self._compute_unconditional_covariance()
                if unconditional_cov is not None:
                    initial_value = unconditional_cov
                else:
                    # Use identity matrix as fallback
                    initial_value = np.eye(self._n_assets)

        # Validate initial value dimensions
        if initial_value.shape != (self._n_assets, self._n_assets):
            raise ValueError(
                f"Initial value shape ({initial_value.shape}) doesn't match "
                f"expected shape ({self._n_assets}, {self._n_assets})"
            )

        # Extract parameters
        C = self._parameters.C
        A = self._parameters.A
        B = self._parameters.B
        G = self._parameters.G if self._parameters.asymmetric else None

        # Initialize forecast array
        forecasts = np.zeros((self._n_assets, self._n_assets, steps))

        # Set initial forecast
        current_cov = initial_value.copy()

        # Compute forecasts
        for t in range(steps):
            # For BEKK, the forecast is:
            # H_{t+1} = CC' + A ε_t ε_t' A' + B H_t B' + G η_t η_t' G' (if asymmetric)
            # where η_t = ε_t * I(ε_t < 0)
            # For forecasting, we use E[ε_t ε_t'] = H_t and E[η_t η_t'] = 0.5 * H_t

            # Compute CC'
            CC = C @ C.T

            # Compute B H_t B'
            BHB = B @ current_cov @ B.T

            # For forecasting, E[ε_t ε_t'] = H_t
            AeA = A @ current_cov @ A.T

            if self.asymmetric and G is not None:
                # For asymmetric term, we use E[η_t η_t'] = 0.5 * H_t
                # This is an approximation assuming returns are symmetric
                GnG = 0.5 * (G @ current_cov @ G.T)
                current_cov = CC + AeA + BHB + GnG
            else:
                current_cov = CC + AeA + BHB

            # Store forecast
            forecasts[:, :, t] = current_cov

        return forecasts

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
        # Use the base class implementation which handles the async pattern
        return await super().forecast_async(
            steps=steps,
            initial_value=initial_value,
            callback=callback,
            **kwargs
        )

    def compute_covariance(
        self,
        parameters: BEKKParameters,
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
        # Validate parameters
        if not isinstance(parameters, BEKKParameters):
            raise TypeError(
                f"parameters must be BEKKParameters, got {type(parameters)}"
            )

        # Validate data dimensions
        if data.ndim != 2:
            raise ValueError(f"data must be 2D, got {data.ndim}D")

        n_assets = data.shape[1]
        T = data.shape[0]

        # Check parameter dimensions
        if parameters.C.shape != (n_assets, n_assets):
            raise ValueError(
                f"C shape ({parameters.C.shape}) doesn't match "
                f"expected shape ({n_assets}, {n_assets})"
            )

        if parameters.A.shape != (n_assets, n_assets):
            raise ValueError(
                f"A shape ({parameters.A.shape}) doesn't match "
                f"expected shape ({n_assets}, {n_assets})"
            )

        if parameters.B.shape != (n_assets, n_assets):
            raise ValueError(
                f"B shape ({parameters.B.shape}) doesn't match "
                f"expected shape ({n_assets}, {n_assets})"
            )

        if parameters.asymmetric and parameters.G is not None:
            if parameters.G.shape != (n_assets, n_assets):
                raise ValueError(
                    f"G shape ({parameters.G.shape}) doesn't match "
                    f"expected shape ({n_assets}, {n_assets})"
                )

        # Initialize sigma if not provided
        if sigma is None:
            sigma = np.zeros((n_assets, n_assets, T))

        # Use backcast if provided, otherwise compute it
        if backcast is None:
            backcast = self.compute_backcast(data)

        # Extract parameters
        C = parameters.C
        A = parameters.A
        B = parameters.B
        G = parameters.G if parameters.asymmetric else None

        # Compute conditional covariances
        if parameters.asymmetric and G is not None:
            # Use asymmetric recursion
            bekk_recursion_asymmetric(data, C, A, B, G, backcast, sigma)
        else:
            # Use symmetric recursion
            bekk_recursion(data, C, A, B, backcast, sigma)

        return sigma

    def _array_to_parameters(self, array: np.ndarray) -> BEKKParameters:
        """Convert a parameter array to a parameter object.

        Args:
            array: Parameter array

        Returns:
            BEKKParameters: Parameter object
        """
        return BEKKParameters.from_array(array, self._n_assets, self.asymmetric)

    def _parameters_to_array(self, parameters: BEKKParameters) -> np.ndarray:
        """Convert a parameter object to a parameter array.

        Args:
            parameters: Parameter object

        Returns:
            np.ndarray: Parameter array
        """
        return parameters.to_array()

    def _create_starting_values(self, data: np.ndarray) -> np.ndarray:
        """Create starting values for optimization.

        Args:
            data: Input data (typically residuals)

        Returns:
            np.ndarray: Starting values for optimization
        """
        n_assets = data.shape[1]

        # Compute sample covariance
        sample_cov = np.cov(data, rowvar=False)

        # Ensure positive definiteness
        try:
            chol = np.linalg.cholesky(sample_cov)
        except np.linalg.LinAlgError:
            # If not positive definite, add a small diagonal component
            min_eigenvalue = np.min(np.linalg.eigvalsh(sample_cov))
            if min_eigenvalue <= 0:
                sample_cov += np.eye(n_assets) * (abs(min_eigenvalue) + 1e-6)
            chol = np.linalg.cholesky(sample_cov)

        # Create C as lower triangular Cholesky factor scaled down
        C = chol * 0.5

        # Create A and B with small values
        # For stability, we want A and B to be small enough that A^2 + B^2 < 1
        A = np.eye(n_assets) * 0.2
        B = np.eye(n_assets) * 0.7

        # Create G if asymmetric
        if self.asymmetric:
            G = np.eye(n_assets) * 0.1

            # Create parameter object
            params = BEKKParameters(C=C, A=A, B=B, G=G, asymmetric=True)
        else:
            # Create parameter object
            params = BEKKParameters(C=C, A=A, B=B, asymmetric=False)

        # Convert to array
        return params.to_array()

    def _create_constraints(self) -> List[Dict[str, Any]]:
        """Create constraints for optimization.

        Returns:
            List[Dict[str, Any]]: List of constraints for optimization
        """
        # For BEKK, we don't need explicit constraints
        # The parameterization ensures positive definiteness
        return []

    def _create_bounds(self) -> List[Tuple[float, float]]:
        """Create bounds for optimization parameters.

        Returns:
            List[Tuple[float, float]]: List of (lower, upper) bounds for each parameter
        """
        n_assets = self._n_assets

        # Calculate parameter counts
        c_len = n_assets * (n_assets + 1) // 2  # Length of vech(C)
        a_len = n_assets * n_assets  # Length of A.flatten()
        b_len = n_assets * n_assets  # Length of B.flatten()
        g_len = n_assets * n_assets if self.asymmetric else 0  # Length of G.flatten()

        # Create bounds
        # For C, we don't need strict bounds
        c_bounds = [(None, None) for _ in range(c_len)]

        # For A, B, and G, we use bounds to help with stability
        # We use loose bounds to allow flexibility
        a_bounds = [(-2.0, 2.0) for _ in range(a_len)]
        b_bounds = [(-2.0, 2.0) for _ in range(b_len)]
        g_bounds = [(-2.0, 2.0) for _ in range(g_len)] if self.asymmetric else []

        return c_bounds + a_bounds + b_bounds + g_bounds

    def _compute_persistence(self) -> Optional[float]:
        """Compute the persistence of the model.

        Returns:
            Optional[float]: Persistence value
        """
        if not self._fitted or self._parameters is None:
            return None

        # For BEKK, persistence is complex and involves eigenvalues
        # We'll use a simplified measure based on squared elements
        a_norm = np.sum(self._parameters.A ** 2)
        b_norm = np.sum(self._parameters.B ** 2)
        g_norm = np.sum(self._parameters.G ** 2) if self._parameters.G is not None else 0

        return a_norm + b_norm + 0.5 * g_norm

    def _compute_unconditional_covariance(self) -> Optional[np.ndarray]:
        """Compute the unconditional covariance matrix implied by the model.

        Returns:
            Optional[np.ndarray]: Unconditional covariance matrix
        """
        if not self._fitted or self._parameters is None:
            return None

        # For BEKK, the unconditional covariance is complex to compute analytically
        # We'll use a numerical approximation by simulating a long series
        try:
            # Simulate a long series and take the average of the conditional covariances
            _, covs = self.simulate(
                n_periods=1000,
                burn=500,
                return_covariances=True,
                random_state=42
            )

            # Average over time
            return np.mean(covs, axis=2)
        except Exception as e:
            logger.warning(f"Failed to compute unconditional covariance: {str(e)}")
            return None
