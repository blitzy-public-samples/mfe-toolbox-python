# mfe/models/multivariate/gogarch.py

"""
Generalized Orthogonal GARCH (GOGARCH) and Orthogonal GARCH (OGARCH) Models

This module implements the GOGARCH and OGARCH models for multivariate volatility
modeling. These factor-based approaches reduce dimensionality through orthogonal
transformations, making them computationally efficient for large dimensions while
maintaining covariance structure.

Both models use principal component analysis (PCA) to transform the data into
orthogonal factors, which are then modeled with univariate GARCH processes.
OGARCH uses standard PCA, while GOGARCH extends this with an additional rotation
matrix to capture more complex dependency structures.

The implementation leverages NumPy for efficient matrix operations and Numba for
performance-critical calculations. Both models support asynchronous processing
for computationally intensive operations.

Classes:
    OGARCHParameters: Parameter container for OGARCH model
    GOGARCHParameters: Parameter container for GOGARCH model
    OGARCHModel: Implementation of the Orthogonal GARCH model
    GOGARCHModel: Implementation of the Generalized Orthogonal GARCH model
"""

import asyncio
import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
from scipy import linalg, optimize, stats

from mfe.core.base import ModelBase
from mfe.core.parameters import (
    MultivariateVolatilityParameters, ParameterBase, ParameterError,
    validate_positive_definite, validate_range
)
from mfe.core.results import MultivariateVolatilityResult
from mfe.core.types import (
    AsyncProgressCallback, CovarianceMatrix, CorrelationMatrix, Matrix,
    MultivariateVolatilityType, ProgressCallback, Vector
)
from mfe.models.multivariate.base import FactorModelBase
from mfe.models.univariate.garch import GARCH, GARCHParameters
from mfe.utils.matrix_ops import cov2corr, vech, ivech

# Set up module-level logger
logger = logging.getLogger("mfe.models.multivariate.gogarch")

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
    logger.debug("Numba available for GOGARCH/OGARCH acceleration")
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    HAS_NUMBA = False
    logger.info("Numba not available. GOGARCH/OGARCH will use pure NumPy implementations.")


@dataclass
class OGARCHParameters(MultivariateVolatilityParameters):
    """Parameters for OGARCH model.

    This class contains the parameters for the OGARCH model, which include
    the number of factors to use and the univariate GARCH parameters for each factor.

    Attributes:
        n_factors: Number of factors to use (must be positive and <= n_assets)
        garch_parameters: List of GARCHParameters for each factor
    """

    n_factors: int
    garch_parameters: List[GARCHParameters] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate OGARCH parameter constraints.

        Raises:
            ParameterError: If parameter constraints are violated
        """
        super().validate()

        # Validate n_factors
        if self.n_factors <= 0:
            raise ParameterError(f"n_factors must be positive, got {self.n_factors}")

        # Validate garch_parameters if provided
        if self.garch_parameters:
            if len(self.garch_parameters) != self.n_factors:
                raise ParameterError(
                    f"Length of garch_parameters ({len(self.garch_parameters)}) "
                    f"must match n_factors ({self.n_factors})"
                )

            # Validate each GARCH parameter set
            for i, params in enumerate(self.garch_parameters):
                try:
                    params.validate()
                except ParameterError as e:
                    raise ParameterError(f"Invalid GARCH parameters for factor {i}: {str(e)}")

    def to_array(self) -> np.ndarray:
        """Convert parameters to a NumPy array.

        Returns:
            np.ndarray: Array representation of parameters
        """
        # Convert each GARCH parameter set to an array and concatenate
        if not self.garch_parameters:
            return np.array([self.n_factors])

        garch_arrays = [params.to_array() for params in self.garch_parameters]
        return np.concatenate([np.array([self.n_factors]), *garch_arrays])

    @classmethod
    def from_array(cls, array: np.ndarray, **kwargs: Any) -> 'OGARCHParameters':
        """Create parameters from a NumPy array.

        Args:
            array: Array representation of parameters
            **kwargs: Additional keyword arguments for parameter creation

        Returns:
            OGARCHParameters: Parameter object

        Raises:
            ValueError: If the array length is invalid
        """
        if len(array) < 1:
            raise ValueError("Array must have at least one element (n_factors)")

        n_factors = int(array[0])

        # Check if array contains GARCH parameters
        if len(array) > 1:
            # Each GARCH parameter set has 3 parameters (omega, alpha, beta)
            if (len(array) - 1) % 3 != 0:
                raise ValueError(
                    f"Array length ({len(array)}) must be 1 + 3*n_factors"
                )

            # Extract GARCH parameters
            garch_params = []
            for i in range(n_factors):
                start_idx = 1 + i * 3
                omega = array[start_idx]
                alpha = array[start_idx + 1]
                beta = array[start_idx + 2]
                garch_params.append(GARCHParameters(omega=omega, alpha=alpha, beta=beta))
        else:
            garch_params = []

        return cls(n_factors=n_factors, garch_parameters=garch_params)

    def transform(self) -> np.ndarray:
        """Transform parameters to unconstrained space for optimization.

        Returns:
            np.ndarray: Transformed parameters in unconstrained space
        """
        # n_factors is discrete and not transformed
        transformed = [float(self.n_factors)]

        # Transform each GARCH parameter set
        for params in self.garch_parameters:
            transformed.extend(params.transform())

        return np.array(transformed)

    @classmethod
    def inverse_transform(cls, array: np.ndarray, **kwargs: Any) -> 'OGARCHParameters':
        """Transform parameters from unconstrained space back to constrained space.

        Args:
            array: Parameters in unconstrained space
            **kwargs: Additional keyword arguments for parameter creation

        Returns:
            OGARCHParameters: Parameter object with constrained parameters

        Raises:
            ValueError: If the array length is invalid
        """
        if len(array) < 1:
            raise ValueError("Array must have at least one element (n_factors)")

        n_factors = int(array[0])

        # Check if array contains transformed GARCH parameters
        if len(array) > 1:
            # Each transformed GARCH parameter set has 3 parameters
            if (len(array) - 1) % 3 != 0:
                raise ValueError(
                    f"Array length ({len(array)}) must be 1 + 3*n_factors"
                )

            # Extract and inverse transform GARCH parameters
            garch_params = []
            for i in range(n_factors):
                start_idx = 1 + i * 3
                transformed_garch = array[start_idx:start_idx + 3]
                garch_params.append(GARCHParameters.inverse_transform(transformed_garch))
        else:
            garch_params = []

        return cls(n_factors=n_factors, garch_parameters=garch_params)


@dataclass
class GOGARCHParameters(OGARCHParameters):
    """Parameters for GOGARCH model.

    This class extends OGARCHParameters to include the rotation matrix U
    that defines the non-orthogonal factors in GOGARCH.

    Attributes:
        n_factors: Number of factors to use (must be positive and <= n_assets)
        garch_parameters: List of GARCHParameters for each factor
        rotation_angles: Angles for the rotation matrix (in radians)
    """

    rotation_angles: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        super().__post_init__()

        # Initialize rotation_angles if not provided
        if self.rotation_angles is None and self.n_factors > 1:
            # For n factors, we need n(n-1)/2 angles
            n_angles = self.n_factors * (self.n_factors - 1) // 2
            self.rotation_angles = np.zeros(n_angles)

    def validate(self) -> None:
        """Validate GOGARCH parameter constraints.

        Raises:
            ParameterError: If parameter constraints are violated
        """
        super().validate()

        # Validate rotation_angles if provided
        if self.rotation_angles is not None:
            if self.n_factors <= 1:
                # No rotation needed for 1 factor
                if len(self.rotation_angles) > 0:
                    warnings.warn(
                        f"rotation_angles provided for n_factors={self.n_factors}, "
                        "but no rotation is needed for n_factors <= 1"
                    )
            else:
                # For n factors, we need n(n-1)/2 angles
                n_angles = self.n_factors * (self.n_factors - 1) // 2
                if len(self.rotation_angles) != n_angles:
                    raise ParameterError(
                        f"Length of rotation_angles ({len(self.rotation_angles)}) "
                        f"must be n_factors*(n_factors-1)/2 = {n_angles}"
                    )

    def to_array(self) -> np.ndarray:
        """Convert parameters to a NumPy array.

        Returns:
            np.ndarray: Array representation of parameters
        """
        # Start with OGARCHParameters array
        array = super().to_array()

        # Add rotation angles if available
        if self.rotation_angles is not None and len(self.rotation_angles) > 0:
            array = np.concatenate([array, self.rotation_angles])

        return array

    @classmethod
    def from_array(cls, array: np.ndarray, **kwargs: Any) -> 'GOGARCHParameters':
        """Create parameters from a NumPy array.

        Args:
            array: Array representation of parameters
            **kwargs: Additional keyword arguments for parameter creation

        Returns:
            GOGARCHParameters: Parameter object

        Raises:
            ValueError: If the array length is invalid
        """
        if len(array) < 1:
            raise ValueError("Array must have at least one element (n_factors)")

        n_factors = int(array[0])

        # Calculate expected array length
        garch_params_length = 3 * n_factors  # Each GARCH has 3 parameters
        rotation_angles_length = n_factors * (n_factors - 1) // 2 if n_factors > 1 else 0
        expected_length = 1 + garch_params_length + rotation_angles_length

        if len(array) != expected_length and len(array) != 1 + garch_params_length:
            raise ValueError(
                f"Array length ({len(array)}) must be 1 + 3*n_factors "
                f"or 1 + 3*n_factors + n_factors*(n_factors-1)/2"
            )

        # Extract GARCH parameters
        garch_params = []
        if len(array) > 1:
            for i in range(n_factors):
                start_idx = 1 + i * 3
                omega = array[start_idx]
                alpha = array[start_idx + 1]
                beta = array[start_idx + 2]
                garch_params.append(GARCHParameters(omega=omega, alpha=alpha, beta=beta))

        # Extract rotation angles if available
        rotation_angles = None
        if len(array) == expected_length and rotation_angles_length > 0:
            start_idx = 1 + garch_params_length
            rotation_angles = array[start_idx:]

        return cls(
            n_factors=n_factors,
            garch_parameters=garch_params,
            rotation_angles=rotation_angles
        )

    def transform(self) -> np.ndarray:
        """Transform parameters to unconstrained space for optimization.

        Returns:
            np.ndarray: Transformed parameters in unconstrained space
        """
        # Start with OGARCHParameters transformed array
        transformed = super().transform()

        # Add rotation angles if available (no transformation needed)
        if self.rotation_angles is not None and len(self.rotation_angles) > 0:
            transformed = np.concatenate([transformed, self.rotation_angles])

        return transformed

    @classmethod
    def inverse_transform(cls, array: np.ndarray, **kwargs: Any) -> 'GOGARCHParameters':
        """Transform parameters from unconstrained space back to constrained space.

        Args:
            array: Parameters in unconstrained space
            **kwargs: Additional keyword arguments for parameter creation

        Returns:
            GOGARCHParameters: Parameter object with constrained parameters

        Raises:
            ValueError: If the array length is invalid
        """
        if len(array) < 1:
            raise ValueError("Array must have at least one element (n_factors)")

        n_factors = int(array[0])

        # Calculate expected array length
        garch_params_length = 3 * n_factors  # Each GARCH has 3 parameters
        rotation_angles_length = n_factors * (n_factors - 1) // 2 if n_factors > 1 else 0
        expected_length = 1 + garch_params_length + rotation_angles_length

        if len(array) != expected_length and len(array) != 1 + garch_params_length:
            raise ValueError(
                f"Array length ({len(array)}) must be 1 + 3*n_factors "
                f"or 1 + 3*n_factors + n_factors*(n_factors-1)/2"
            )

        # Extract and inverse transform GARCH parameters
        garch_params = []
        if len(array) > 1:
            for i in range(n_factors):
                start_idx = 1 + i * 3
                transformed_garch = array[start_idx:start_idx + 3]
                garch_params.append(GARCHParameters.inverse_transform(transformed_garch))

        # Extract rotation angles if available
        rotation_angles = None
        if len(array) == expected_length and rotation_angles_length > 0:
            start_idx = 1 + garch_params_length
            rotation_angles = array[start_idx:]

        return cls(
            n_factors=n_factors,
            garch_parameters=garch_params,
            rotation_angles=rotation_angles
        )


@jit(nopython=True, cache=True)
def _compute_rotation_matrix_numba(angles: np.ndarray, n_factors: int) -> np.ndarray:
    """
    Numba-accelerated implementation to compute the rotation matrix from angles.

    Args:
        angles: Array of rotation angles (in radians)
        n_factors: Number of factors

    Returns:
        Rotation matrix U
    """
    # Initialize with identity matrix
    U = np.eye(n_factors)

    # Apply Givens rotations
    angle_idx = 0
    for i in range(n_factors - 1):
        for j in range(i + 1, n_factors):
            # Create Givens rotation matrix
            G = np.eye(n_factors)
            c = np.cos(angles[angle_idx])
            s = np.sin(angles[angle_idx])
            G[i, i] = c
            G[j, j] = c
            G[i, j] = -s
            G[j, i] = s

            # Apply rotation
            U = G @ U
            angle_idx += 1

    return U


def compute_rotation_matrix(angles: np.ndarray, n_factors: int) -> np.ndarray:
    """
    Compute the rotation matrix from angles using Givens rotations.

    Args:
        angles: Array of rotation angles (in radians)
        n_factors: Number of factors

    Returns:
        Rotation matrix U

    Raises:
        ValueError: If the length of angles doesn't match n_factors*(n_factors-1)/2
    """
    # Check if angles has the correct length
    expected_length = n_factors * (n_factors - 1) // 2
    if len(angles) != expected_length:
        raise ValueError(
            f"Length of angles ({len(angles)}) must be "
            f"n_factors*(n_factors-1)/2 = {expected_length}"
        )

    # Use Numba-accelerated implementation if available
    if HAS_NUMBA:
        return _compute_rotation_matrix_numba(angles, n_factors)

    # Pure NumPy implementation
    # Initialize with identity matrix
    U = np.eye(n_factors)

    # Apply Givens rotations
    angle_idx = 0
    for i in range(n_factors - 1):
        for j in range(i + 1, n_factors):
            # Create Givens rotation matrix
            G = np.eye(n_factors)
            c = np.cos(angles[angle_idx])
            s = np.sin(angles[angle_idx])
            G[i, i] = c
            G[j, j] = c
            G[i, j] = -s
            G[j, i] = s

            # Apply rotation
            U = G @ U
            angle_idx += 1

    return U


class OGARCHModel(FactorModelBase):
    """Orthogonal GARCH (OGARCH) model implementation.

    The OGARCH model uses principal component analysis (PCA) to transform the data
    into orthogonal factors, which are then modeled with univariate GARCH processes.
    This approach is computationally efficient for high-dimensional problems.

    Attributes:
        name: Name of the model
        n_assets: Number of assets
        n_factors: Number of factors used
        factor_loadings: PCA loading matrix (eigenvectors)
        factor_volatilities: Conditional volatilities of the factors
        parameters: Model parameters if fitted
    """

    def __init__(
        self,
        n_factors: Optional[int] = None,
        n_assets: Optional[int] = None,
        name: str = "OGARCH"
    ):
        """Initialize the OGARCH model.

        Args:
            n_factors: Number of factors to use (if None, determined from data)
            n_assets: Number of assets (if None, determined from data)
            name: Name of the model
        """
        super().__init__(name=name, n_assets=n_assets, n_factors=n_factors)

        self._eigenvalues: Optional[np.ndarray] = None
        self._factor_models: List[GARCH] = []
        self._factor_data: Optional[np.ndarray] = None

    def fit(
        self,
        data: np.ndarray,
        starting_values: Optional[Union[np.ndarray, OGARCHParameters]] = None,
        backcast: Optional[np.ndarray] = None,
        method: str = "SLSQP",
        options: Optional[Dict[str, Any]] = None,
        constraints: Optional[List[Dict[str, Any]]] = None,
        callback: Optional[ProgressCallback] = None,
        **kwargs: Any
    ) -> MultivariateVolatilityResult:
        """Fit the OGARCH model to the provided data.

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

        # Set n_assets if not already set
        if self._n_assets is None:
            self._n_assets = data.shape[1]

        # Compute backcast if not provided
        if backcast is None:
            backcast = self.compute_backcast(data)

        # Compute factor loadings (PCA)
        self._compute_factor_loadings(data)

        # Determine n_factors if not specified
        if self._n_factors is None:
            # Default to using all factors
            self._n_factors = self._n_assets

        # Extract factors
        self._factor_data = data @ self._factor_loadings[:, :self._n_factors]

        # Initialize GARCH models for each factor
        self._factor_models = []
        garch_parameters = []

        # Report progress
        if callback:
            callback(0.1, "Computing factor loadings")

        # Fit GARCH models to each factor
        for i in range(self._n_factors):
            if callback:
                progress = 0.1 + 0.8 * (i / self._n_factors)
                callback(progress, f"Fitting GARCH model for factor {i+1}/{self._n_factors}")

            # Extract factor data
            factor_data = self._factor_data[:, i]

            # Create and fit GARCH model
            garch_model = GARCH()
            garch_result = garch_model.fit(factor_data)

            # Store model and parameters
            self._factor_models.append(garch_model)
            garch_parameters.append(garch_model.parameters)

        # Create parameter object
        self._parameters = OGARCHParameters(
            n_factors=self._n_factors,
            garch_parameters=garch_parameters
        )

        # Compute conditional covariances
        if callback:
            callback(0.9, "Computing conditional covariances")

        covariance = self.compute_covariance(self._parameters, data, backcast=backcast)
        correlation = self.compute_correlation(self._parameters, data, backcast=backcast)

        # Compute log-likelihood
        loglik = self.loglikelihood(self._parameters, data, backcast=backcast)

        # Compute standard errors (not implemented for OGARCH)
        std_errors = np.array([])

        # Mark as fitted
        self._fitted = True
        self._residuals = data
        self._conditional_covariances = covariance
        self._conditional_correlations = correlation

        # Report completion
        if callback:
            callback(1.0, "Model estimation complete")

        # Create result object
        result = self._create_result_object(
            parameters=self._parameters,
            data=data,
            covariance=covariance,
            correlation=correlation,
            loglikelihood=loglik,
            std_errors=std_errors,
            iterations=0,  # No iterations for OGARCH
            convergence=True,
            optimization_message="OGARCH estimation complete",
            eigenvalues=self._eigenvalues,
            factor_loadings=self._factor_loadings,
            factor_models=self._factor_models
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
        """Simulate data from the model.

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

        # Set random state
        if isinstance(random_state, int):
            rng = np.random.Generator(np.random.PCG64(random_state))
        elif random_state is None:
            rng = np.random.default_rng()
        else:
            rng = random_state

        # Simulate factors
        factor_data = np.zeros((n_periods + burn, self._n_factors))
        factor_variances = np.zeros((n_periods + burn, self._n_factors))

        # Simulate each factor
        for i in range(self._n_factors):
            # Simulate from GARCH model
            factor_sim = self._factor_models[i].simulate(
                n_periods + burn,
                random_state=rng
            )

            # Extract simulated data and variances
            if isinstance(factor_sim, tuple):
                factor_data[:, i] = factor_sim[0]
                factor_variances[:, i] = factor_sim[1]
            else:
                factor_data[:, i] = factor_sim
                # Compute variances if not returned
                params = self._factor_models[i].parameters
                if params is not None:
                    omega, alpha, beta = params.omega, params.alpha, params.beta
                    variance = np.zeros(n_periods + burn)
                    variance[0] = omega / (1 - alpha - beta)  # Unconditional variance
                    for t in range(1, n_periods + burn):
                        variance[t] = omega + alpha * factor_data[t-1, i]**2 + beta * variance[t-1]
                    factor_variances[:, i] = variance

        # Transform factors back to original space
        simulated_data = factor_data @ self._factor_loadings[:, :self._n_factors].T

        # Compute conditional covariances if requested
        if return_covariances:
            covariances = np.zeros((self._n_assets, self._n_assets, n_periods + burn))
            for t in range(n_periods + burn):
                # Create diagonal matrix of factor variances
                D = np.diag(factor_variances[t, :])
                # Compute covariance matrix: A * D * A'
                covariances[:, :, t] = (
                    self._factor_loadings[:, :self._n_factors] @
                    D @
                    self._factor_loadings[:, :self._n_factors].T
                )

            # Discard burn-in period
            if burn > 0:
                simulated_data = simulated_data[burn:]
                covariances = covariances[:, :, burn:]

            return simulated_data, covariances

        # Discard burn-in period
        if burn > 0:
            simulated_data = simulated_data[burn:]

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

        # Initialize forecasted covariances
        forecasted_covariances = np.zeros((self._n_assets, self._n_assets, steps))

        # Forecast each factor's variance
        factor_variances = np.zeros((steps, self._n_factors))
        for i in range(self._n_factors):
            # Get GARCH model for this factor
            garch_model = self._factor_models[i]

            # Forecast variance
            factor_forecast = garch_model.forecast(steps)

            # Extract forecasted variances
            if isinstance(factor_forecast, tuple):
                factor_variances[:, i] = factor_forecast[0]
            else:
                factor_variances[:, i] = factor_forecast

        # Compute forecasted covariances
        for t in range(steps):
            # Create diagonal matrix of factor variances
            D = np.diag(factor_variances[t, :])
            # Compute covariance matrix: A * D * A'
            forecasted_covariances[:, :, t] = (
                self._factor_loadings[:, :self._n_factors] @
                D @
                self._factor_loadings[:, :self._n_factors].T
            )

        return forecasted_covariances

    def compute_covariance(
        self,
        parameters: MultivariateVolatilityParameters,
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

        Raises:
            ValueError: If parameters is not an OGARCHParameters object
        """
        if not isinstance(parameters, OGARCHParameters):
            raise ValueError("parameters must be an OGARCHParameters object")

        # Validate parameters
        parameters.validate()

        # Extract parameters
        n_factors = parameters.n_factors
        garch_parameters = parameters.garch_parameters

        # Check if factor loadings are available
        if self._factor_loadings is None:
            self._compute_factor_loadings(data)

        # Extract factors if not already done
        if self._factor_data is None or self._factor_data.shape[1] != n_factors:
            self._factor_data = data @ self._factor_loadings[:, :n_factors]

        # Initialize output array
        T = data.shape[0]
        n_assets = data.shape[1]

        if sigma is None:
            sigma = np.zeros((n_assets, n_assets, T))

        # Compute factor volatilities
        factor_volatilities = self._compute_factor_volatilities(parameters, data, backcast)

        # Compute conditional covariances
        for t in range(T):
            # Create diagonal matrix of factor volatilities
            D = np.diag(factor_volatilities[t, :])
            # Compute covariance matrix: A * D * A'
            sigma[:, :, t] = (
                self._factor_loadings[:, :n_factors] @
                D @
                self._factor_loadings[:, :n_factors].T
            )

        return sigma

    def _compute_factor_loadings(self, data: np.ndarray) -> np.ndarray:
        """Compute factor loadings from the data using PCA.

        Args:
            data: Input data (typically residuals)

        Returns:
            np.ndarray: Factor loading matrix (eigenvectors)
        """
        # Compute sample covariance matrix
        cov_matrix = np.cov(data, rowvar=False)

        # Compute eigendecomposition
        eigenvalues, eigenvectors = linalg.eigh(cov_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Store eigenvalues and eigenvectors
        self._eigenvalues = eigenvalues
        self._factor_loadings = eigenvectors

        return eigenvectors

    def _compute_factor_volatilities(
        self,
        parameters: OGARCHParameters,
        data: np.ndarray,
        backcast: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute factor volatility estimates.

        Args:
            parameters: Model parameters
            data: Input data (typically residuals)
            backcast: Matrix to use for initializing the volatility process

        Returns:
            np.ndarray: Factor volatility estimates (T x n_factors)
        """
        # Extract parameters
        n_factors = parameters.n_factors
        garch_parameters = parameters.garch_parameters

        # Check if factor loadings are available
        if self._factor_loadings is None:
            self._compute_factor_loadings(data)

        # Extract factors if not already done
        if self._factor_data is None or self._factor_data.shape[1] != n_factors:
            self._factor_data = data @ self._factor_loadings[:, :n_factors]

        # Initialize output array
        T = data.shape[0]
        factor_volatilities = np.zeros((T, n_factors))

        # Compute volatilities for each factor
        for i in range(n_factors):
            # Extract factor data
            factor_data = self._factor_data[:, i]

            # Check if GARCH parameters are available
            if garch_parameters and i < len(garch_parameters):
                # Create GARCH model with provided parameters
                garch_model = GARCH(parameters=garch_parameters[i])
            elif self._factor_models and i < len(self._factor_models):
                # Use existing GARCH model
                garch_model = self._factor_models[i]
            else:
                # Create and fit new GARCH model
                garch_model = GARCH()
                garch_model.fit(factor_data)

                # Store model if not already stored
                if i >= len(self._factor_models):
                    self._factor_models.append(garch_model)

            # Compute conditional variances
            variances = garch_model.compute_variance(factor_data)

            # Store volatilities (standard deviations)
            factor_volatilities[:, i] = np.sqrt(variances)

        return factor_volatilities

    def _array_to_parameters(self, array: np.ndarray) -> OGARCHParameters:
        """Convert a parameter array to a parameter object.

        Args:
            array: Parameter array

        Returns:
            OGARCHParameters: Parameter object
        """
        return OGARCHParameters.from_array(array)

    def _parameters_to_array(self, parameters: MultivariateVolatilityParameters) -> np.ndarray:
        """Convert a parameter object to a parameter array.

        Args:
            parameters: Parameter object

        Returns:
            np.ndarray: Parameter array

        Raises:
            ValueError: If parameters is not an OGARCHParameters object
        """
        if not isinstance(parameters, OGARCHParameters):
            raise ValueError("parameters must be an OGARCHParameters object")

        return parameters.to_array()

    def _create_starting_values(self, data: np.ndarray) -> np.ndarray:
        """Create starting values for optimization.

        Args:
            data: Input data (typically residuals)

        Returns:
            np.ndarray: Starting values for optimization
        """
        # For OGARCH, we don't need starting values for optimization
        # as we use PCA and individual GARCH fits
        return np.array([self._n_factors or data.shape[1]])

    def _compute_persistence(self) -> Optional[float]:
        """Compute the persistence of the model.

        Returns:
            Optional[float]: Persistence value (maximum of factor persistences)
        """
        if not self._fitted or not self._factor_models:
            return None

        # Compute persistence for each factor
        persistences = []
        for model in self._factor_models:
            if model.parameters is not None:
                persistences.append(model.parameters.alpha + model.parameters.beta)

        # Return maximum persistence
        return max(persistences) if persistences else None

    def _compute_half_life(self) -> Optional[float]:
        """Compute the half-life of shocks in the model.

        Returns:
            Optional[float]: Half-life value (based on maximum persistence)
        """
        persistence = self._compute_persistence()
        if persistence is not None and persistence < 1:
            return np.log(0.5) / np.log(persistence)
        return None

    def _compute_unconditional_covariance(self) -> Optional[np.ndarray]:
        """Compute the unconditional covariance matrix implied by the model.

        Returns:
            Optional[np.ndarray]: Unconditional covariance matrix
        """
        if not self._fitted or self._factor_loadings is None or not self._factor_models:
            return None

        # Compute unconditional variance for each factor
        factor_variances = np.zeros(self._n_factors)
        for i, model in enumerate(self._factor_models):
            if i >= self._n_factors:
                break

            if model.parameters is not None:
                omega, alpha, beta = model.parameters.omega, model.parameters.alpha, model.parameters.beta
                if alpha + beta < 1:
                    factor_variances[i] = omega / (1 - alpha - beta)
                else:
                    # Non-stationary model
                    return None

        # Compute unconditional covariance matrix
        D = np.diag(factor_variances)
        uncond_cov = (
            self._factor_loadings[:, :self._n_factors] @
            D @
            self._factor_loadings[:, :self._n_factors].T
        )

        return uncond_cov


class GOGARCHModel(OGARCHModel):
    """Generalized Orthogonal GARCH (GOGARCH) model implementation.

    The GOGARCH model extends OGARCH by introducing a rotation matrix that allows
    for non-orthogonal factors, providing more flexibility in capturing complex
    dependency structures.

    Attributes:
        name: Name of the model
        n_assets: Number of assets
        n_factors: Number of factors used
        factor_loadings: PCA loading matrix (eigenvectors)
        factor_volatilities: Conditional volatilities of the factors
        parameters: Model parameters if fitted
        rotation_matrix: Rotation matrix for non-orthogonal factors
    """

    def __init__(
        self,
        n_factors: Optional[int] = None,
        n_assets: Optional[int] = None,
        name: str = "GOGARCH"
    ):
        """Initialize the GOGARCH model.

        Args:
            n_factors: Number of factors to use (if None, determined from data)
            n_assets: Number of assets (if None, determined from data)
            name: Name of the model
        """
        super().__init__(n_factors=n_factors, n_assets=n_assets, name=name)

        self._rotation_matrix: Optional[np.ndarray] = None

    def fit(
        self,
        data: np.ndarray,
        starting_values: Optional[Union[np.ndarray, GOGARCHParameters]] = None,
        backcast: Optional[np.ndarray] = None,
        method: str = "SLSQP",
        options: Optional[Dict[str, Any]] = None,
        constraints: Optional[List[Dict[str, Any]]] = None,
        callback: Optional[ProgressCallback] = None,
        **kwargs: Any
    ) -> MultivariateVolatilityResult:
        """Fit the GOGARCH model to the provided data.

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

        # Set n_assets if not already set
        if self._n_assets is None:
            self._n_assets = data.shape[1]

        # Compute backcast if not provided
        if backcast is None:
            backcast = self.compute_backcast(data)

        # Compute factor loadings (PCA)
        self._compute_factor_loadings(data)

        # Determine n_factors if not specified
        if self._n_factors is None:
            # Default to using all factors
            self._n_factors = self._n_assets

        # Report progress
        if callback:
            callback(0.1, "Computing factor loadings")

        # Initialize rotation matrix
        if self._n_factors > 1:
            # For n factors, we need n(n-1)/2 angles
            n_angles = self._n_factors * (self._n_factors - 1) // 2

            # Check if starting values are provided
            if starting_values is not None:
                if isinstance(starting_values, GOGARCHParameters):
                    # Use provided rotation angles
                    rotation_angles = starting_values.rotation_angles
                    if rotation_angles is None or len(rotation_angles) != n_angles:
                        # Initialize with zeros if not provided or invalid
                        rotation_angles = np.zeros(n_angles)
                elif isinstance(starting_values, np.ndarray) and len(starting_values) >= 1 + n_angles:
                    # Extract rotation angles from array
                    rotation_angles = starting_values[1:1+n_angles]
                else:
                    # Initialize with zeros
                    rotation_angles = np.zeros(n_angles)
            else:
                # Initialize with zeros
                rotation_angles = np.zeros(n_angles)

            # Optimize rotation matrix
            if callback:
                callback(0.2, "Optimizing rotation matrix")

            # Define objective function for rotation optimization
            def objective(angles: np.ndarray) -> float:
                # Compute rotation matrix
                U = compute_rotation_matrix(angles, self._n_factors)

                # Extract factors with rotation
                rotated_loadings = self._factor_loadings[:, :self._n_factors] @ U
                factors = data @ rotated_loadings

                # Compute negentropy (approximation of non-Gaussianity)
                negentropy = 0.0
                for i in range(self._n_factors):
                    # Standardize factor
                    factor = factors[:, i]
                    factor = (factor - np.mean(factor)) / np.std(factor)

                    # Compute kurtosis
                    kurt = np.mean(factor**4) - 3

                    # Add to negentropy (we want to maximize non-Gaussianity)
                    negentropy += kurt**2

                # Return negative negentropy for minimization
                return -negentropy

            # Set up optimization
            if options is None:
                options = {'maxiter': 1000}

            # Run optimization
            result = optimize.minimize(
                objective,
                rotation_angles,
                method=method,
                options=options
            )

            # Extract optimized angles
            rotation_angles = result.x

            # Compute rotation matrix
            self._rotation_matrix = compute_rotation_matrix(rotation_angles, self._n_factors)
        else:
            # No rotation needed for 1 factor
            self._rotation_matrix = np.eye(self._n_factors)
            rotation_angles = np.array([])

        # Extract factors with rotation
        rotated_loadings = self._factor_loadings[:, :self._n_factors] @ self._rotation_matrix
        self._factor_data = data @ rotated_loadings

        # Report progress
        if callback:
            callback(0.3, "Fitting GARCH models to factors")

        # Initialize GARCH models for each factor
        self._factor_models = []
        garch_parameters = []

        # Fit GARCH models to each factor
        for i in range(self._n_factors):
            if callback:
                progress = 0.3 + 0.6 * (i / self._n_factors)
                callback(progress, f"Fitting GARCH model for factor {i+1}/{self._n_factors}")

            # Extract factor data
            factor_data = self._factor_data[:, i]

            # Create and fit GARCH model
            garch_model = GARCH()
            garch_result = garch_model.fit(factor_data)

            # Store model and parameters
            self._factor_models.append(garch_model)
            garch_parameters.append(garch_model.parameters)

        # Create parameter object
        self._parameters = GOGARCHParameters(
            n_factors=self._n_factors,
            garch_parameters=garch_parameters,
            rotation_angles=rotation_angles
        )

        # Compute conditional covariances
        if callback:
            callback(0.9, "Computing conditional covariances")

        covariance = self.compute_covariance(self._parameters, data, backcast=backcast)
        correlation = self.compute_correlation(self._parameters, data, backcast=backcast)

        # Compute log-likelihood
        loglik = self.loglikelihood(self._parameters, data, backcast=backcast)

        # Compute standard errors (not implemented for GOGARCH)
        std_errors = np.array([])

        # Mark as fitted
        self._fitted = True
        self._residuals = data
        self._conditional_covariances = covariance
        self._conditional_correlations = correlation

        # Report completion
        if callback:
            callback(1.0, "Model estimation complete")

        # Create result object
        result = self._create_result_object(
            parameters=self._parameters,
            data=data,
            covariance=covariance,
            correlation=correlation,
            loglikelihood=loglik,
            std_errors=std_errors,
            iterations=result.nit if hasattr(result, 'nit') else 0,
            convergence=result.success if hasattr(result, 'success') else True,
            optimization_message=result.message if hasattr(result, 'message') else "GOGARCH estimation complete",
            eigenvalues=self._eigenvalues,
            factor_loadings=self._factor_loadings,
            factor_models=self._factor_models,
            rotation_matrix=self._rotation_matrix
        )

        return result

    def compute_covariance(
        self,
        parameters: MultivariateVolatilityParameters,
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

        Raises:
            ValueError: If parameters is not a GOGARCHParameters object
        """
        if not isinstance(parameters, GOGARCHParameters):
            raise ValueError("parameters must be a GOGARCHParameters object")

        # Validate parameters
        parameters.validate()

        # Extract parameters
        n_factors = parameters.n_factors
        garch_parameters = parameters.garch_parameters
        rotation_angles = parameters.rotation_angles

        # Check if factor loadings are available
        if self._factor_loadings is None:
            self._compute_factor_loadings(data)

        # Compute rotation matrix if needed
        if rotation_angles is not None and len(rotation_angles) > 0:
            self._rotation_matrix = compute_rotation_matrix(rotation_angles, n_factors)
        elif self._rotation_matrix is None or self._rotation_matrix.shape != (n_factors, n_factors):
            # Default to identity matrix
            self._rotation_matrix = np.eye(n_factors)

        # Compute rotated loadings
        rotated_loadings = self._factor_loadings[:, :n_factors] @ self._rotation_matrix

        # Extract factors if not already done
        if self._factor_data is None or self._factor_data.shape[1] != n_factors:
            self._factor_data = data @ rotated_loadings

        # Initialize output array
        T = data.shape[0]
        n_assets = data.shape[1]

        if sigma is None:
            sigma = np.zeros((n_assets, n_assets, T))

        # Compute factor volatilities
        factor_volatilities = self._compute_factor_volatilities(parameters, data, backcast)

        # Compute conditional covariances
        for t in range(T):
            # Create diagonal matrix of factor volatilities
            D = np.diag(factor_volatilities[t, :])
            # Compute covariance matrix: A * D * A'
            sigma[:, :, t] = rotated_loadings @ D @ rotated_loadings.T

        return sigma

    def _compute_factor_volatilities(
        self,
        parameters: GOGARCHParameters,
        data: np.ndarray,
        backcast: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute factor volatility estimates.

        Args:
            parameters: Model parameters
            data: Input data (typically residuals)
            backcast: Matrix to use for initializing the volatility process

        Returns:
            np.ndarray: Factor volatility estimates (T x n_factors)
        """
        # Extract parameters
        n_factors = parameters.n_factors
        garch_parameters = parameters.garch_parameters
        rotation_angles = parameters.rotation_angles

        # Check if factor loadings are available
        if self._factor_loadings is None:
            self._compute_factor_loadings(data)

        # Compute rotation matrix if needed
        if rotation_angles is not None and len(rotation_angles) > 0:
            self._rotation_matrix = compute_rotation_matrix(rotation_angles, n_factors)
        elif self._rotation_matrix is None or self._rotation_matrix.shape != (n_factors, n_factors):
            # Default to identity matrix
            self._rotation_matrix = np.eye(n_factors)

        # Compute rotated loadings
        rotated_loadings = self._factor_loadings[:, :n_factors] @ self._rotation_matrix

        # Extract factors if not already done
        if self._factor_data is None or self._factor_data.shape[1] != n_factors:
            self._factor_data = data @ rotated_loadings

        # Initialize output array
        T = data.shape[0]
        factor_volatilities = np.zeros((T, n_factors))

        # Compute volatilities for each factor
        for i in range(n_factors):
            # Extract factor data
            factor_data = self._factor_data[:, i]

            # Check if GARCH parameters are available
            if garch_parameters and i < len(garch_parameters):
                # Create GARCH model with provided parameters
                garch_model = GARCH(parameters=garch_parameters[i])
            elif self._factor_models and i < len(self._factor_models):
                # Use existing GARCH model
                garch_model = self._factor_models[i]
            else:
                # Create and fit new GARCH model
                garch_model = GARCH()
                garch_model.fit(factor_data)

                # Store model if not already stored
                if i >= len(self._factor_models):
                    self._factor_models.append(garch_model)

            # Compute conditional variances
            variances = garch_model.compute_variance(factor_data)

            # Store volatilities (standard deviations)
            factor_volatilities[:, i] = np.sqrt(variances)

        return factor_volatilities

    def _array_to_parameters(self, array: np.ndarray) -> GOGARCHParameters:
        """Convert a parameter array to a parameter object.

        Args:
            array: Parameter array

        Returns:
            GOGARCHParameters: Parameter object
        """
        return GOGARCHParameters.from_array(array)

    def _parameters_to_array(self, parameters: MultivariateVolatilityParameters) -> np.ndarray:
        """Convert a parameter object to a parameter array.

        Args:
            parameters: Parameter object

        Returns:
            np.ndarray: Parameter array

        Raises:
            ValueError: If parameters is not a GOGARCHParameters object
        """
        if not isinstance(parameters, GOGARCHParameters):
            raise ValueError("parameters must be a GOGARCHParameters object")

        return parameters.to_array()

    def _create_starting_values(self, data: np.ndarray) -> np.ndarray:
        """Create starting values for optimization.

        Args:
            data: Input data (typically residuals)

        Returns:
            np.ndarray: Starting values for optimization
        """
        # For GOGARCH, we need starting values for the rotation angles
        n_factors = self._n_factors or data.shape[1]

        # For n factors, we need n(n-1)/2 angles
        n_angles = n_factors * (n_factors - 1) // 2 if n_factors > 1 else 0

        # Initialize with zeros
        starting_values = np.zeros(1 + n_angles)
        starting_values[0] = n_factors

        return starting_values
