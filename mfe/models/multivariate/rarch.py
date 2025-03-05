'''
Rotated ARCH (RARCH) Multivariate Volatility Model

This module implements the Rotated ARCH (RARCH) multivariate volatility model,
which uses rotation techniques for variance and correlation targeting. The RARCH
model provides a flexible approach to multivariate volatility modeling while
ensuring positive definiteness of conditional covariance matrices.

The model supports scalar, diagonal, and full parameterizations, with efficient
implementation using NumPy's array operations and Numba acceleration for
performance-critical calculations.

Classes:
    RARCHParameters: Parameter container for RARCH model
    RARCHModel: Implementation of the RARCH model
'''

import logging
import warnings
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union, cast
)

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
from mfe.models.multivariate.base import CovarianceModelBase
from mfe.models.multivariate.utils import (
    compute_sample_covariance, compute_sample_correlation,
    ensure_positive_definite, initialize_parameters,
    transform_correlation_matrix, inverse_transform_correlation_matrix,
    validate_multivariate_data
)
from mfe.utils.matrix_ops import (
    cov2corr, corr2cov, vech, ivech, ensure_symmetric, 
    is_positive_definite, nearest_positive_definite
)

# Set up module-level logger
logger = logging.getLogger("mfe.models.multivariate.rarch")

# Try to import numba for JIT compilation
try:
    from numba import jit
    from mfe.models.multivariate._numba_core import (
        _compute_rarch_covariance_numba
    )
    HAS_NUMBA = True
    logger.debug("Numba available for RARCH model acceleration")
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    HAS_NUMBA = False
    logger.info("Numba not available. RARCH model will use pure NumPy implementations.")


@dataclass
class RARCHParameters(MultivariateVolatilityParameters):
    """Parameters for the RARCH model.
    
    This class contains the parameters for the Rotated ARCH (RARCH) model,
    which consists of rotation angles for the orthogonal rotation matrix.
    
    Attributes:
        rotation_angles: Rotation angles for the orthogonal rotation matrix
        n_assets: Number of assets in the model
    """
    
    rotation_angles: np.ndarray
    n_assets: int
    
    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        # Ensure rotation_angles is a NumPy array
        if not isinstance(self.rotation_angles, np.ndarray):
            self.rotation_angles = np.array(self.rotation_angles)
        
        self.validate()
    
    def validate(self) -> None:
        """Validate RARCH parameter constraints.
        
        Raises:
            ParameterError: If parameter constraints are violated
        """
        super().validate()
        
        # Check number of rotation angles
        expected_angles = self.n_assets * (self.n_assets - 1) // 2
        if len(self.rotation_angles) != expected_angles:
            raise ParameterError(
                f"Number of rotation angles ({len(self.rotation_angles)}) does not match "
                f"expected number ({expected_angles}) for {self.n_assets} assets"
            )
        
        # Rotation angles don't have specific constraints, but we can check for NaN or inf
        if np.any(np.isnan(self.rotation_angles)) or np.any(np.isinf(self.rotation_angles)):
            raise ParameterError("Rotation angles contain NaN or infinite values")
    
    def to_array(self) -> np.ndarray:
        """Convert parameters to a NumPy array.
        
        Returns:
            np.ndarray: Array representation of parameters
        """
        return self.rotation_angles.copy()
    
    @classmethod
    def from_array(cls, array: np.ndarray, n_assets: int, **kwargs: Any) -> 'RARCHParameters':
        """Create parameters from a NumPy array.
        
        Args:
            array: Array representation of parameters
            n_assets: Number of assets in the model
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            RARCHParameters: Parameter object
        
        Raises:
            ValueError: If the array length doesn't match the expected number of parameters
        """
        expected_angles = n_assets * (n_assets - 1) // 2
        if len(array) != expected_angles:
            raise ValueError(
                f"Array length ({len(array)}) doesn't match expected number of "
                f"rotation angles ({expected_angles}) for {n_assets} assets"
            )
        
        return cls(rotation_angles=array, n_assets=n_assets)
    
    def transform(self) -> np.ndarray:
        """Transform parameters to unconstrained space for optimization.
        
        For RARCH, the rotation angles are already unconstrained, so we just return them.
        
        Returns:
            np.ndarray: Transformed parameters in unconstrained space
        """
        # Rotation angles are already unconstrained
        return self.rotation_angles.copy()
    
    @classmethod
    def inverse_transform(cls, array: np.ndarray, n_assets: int, **kwargs: Any) -> 'RARCHParameters':
        """Transform parameters from unconstrained space back to constrained space.
        
        For RARCH, the rotation angles are already unconstrained, so we just use them directly.
        
        Args:
            array: Parameters in unconstrained space
            n_assets: Number of assets in the model
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            RARCHParameters: Parameter object with constrained parameters
        """
        return cls.from_array(array, n_assets=n_assets)


class RARCHModel(CovarianceModelBase):
    """Rotated ARCH (RARCH) multivariate volatility model.
    
    This class implements the Rotated ARCH (RARCH) model, which uses rotation
    techniques for variance and correlation targeting. The model provides a
    flexible approach to multivariate volatility modeling while ensuring
    positive definiteness of conditional covariance matrices.
    
    Attributes:
        name: Name of the model
        n_assets: Number of assets in the model
        parameters: Model parameters if fitted
        _conditional_covariances: Conditional covariance matrices if fitted
        _conditional_correlations: Conditional correlation matrices if fitted
        _residuals: Residuals used for model fitting
        _backcast: Backcast value for initializing the covariance process
        _rotation_matrix: Orthogonal rotation matrix
        _target_covariance: Target (unconditional) covariance matrix
    """
    
    def __init__(
        self,
        n_assets: Optional[int] = None,
        name: str = "RARCH"
    ):
        """Initialize the RARCH model.
        
        Args:
            n_assets: Number of assets in the model (if None, determined from data)
            name: Name of the model
        """
        super().__init__(name=name, n_assets=n_assets)
        
        # Additional attributes specific to RARCH
        self._rotation_matrix: Optional[np.ndarray] = None
        self._target_covariance: Optional[np.ndarray] = None
    
    def compute_rotation_matrix(self, angles: np.ndarray) -> np.ndarray:
        """Compute the orthogonal rotation matrix from rotation angles.
        
        This method constructs an orthogonal rotation matrix from a set of
        rotation angles using Givens rotations.
        
        Args:
            angles: Rotation angles
            
        Returns:
            np.ndarray: Orthogonal rotation matrix
        
        Raises:
            ValueError: If the number of angles doesn't match the expected number
        """
        n = self._n_assets
        expected_angles = n * (n - 1) // 2
        
        if len(angles) != expected_angles:
            raise ValueError(
                f"Number of angles ({len(angles)}) doesn't match expected number "
                f"({expected_angles}) for {n} assets"
            )
        
        # Initialize rotation matrix as identity
        rotation = np.eye(n)
        
        # Apply Givens rotations
        angle_idx = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Create Givens rotation matrix
                theta = angles[angle_idx]
                givens = np.eye(n)
                
                # Set the four elements that change
                c, s = np.cos(theta), np.sin(theta)
                givens[i, i] = c
                givens[i, j] = -s
                givens[j, i] = s
                givens[j, j] = c
                
                # Apply rotation
                rotation = rotation @ givens
                angle_idx += 1
        
        return rotation
    
    def compute_covariance(
        self,
        parameters: MultivariateVolatilityParameters,
        data: np.ndarray,
        sigma: Optional[np.ndarray] = None,
        backcast: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute conditional covariance matrices for the given parameters and data.
        
        This method computes the conditional covariance matrices for the RARCH model
        based on the provided parameters and data.
        
        Args:
            parameters: Model parameters
            data: Input data (typically residuals)
            sigma: Pre-allocated array for conditional covariances
            backcast: Matrix to use for initializing the covariance process
            
        Returns:
            np.ndarray: Conditional covariance matrices (n_assets x n_assets x T)
            
        Raises:
            ValueError: If parameters is not a RARCHParameters object
            ValueError: If data dimensions are invalid
        """
        # Validate parameters type
        if not isinstance(parameters, RARCHParameters):
            raise ValueError(f"Parameters must be RARCHParameters, got {type(parameters)}")
        
        # Validate data
        T, n_assets = validate_multivariate_data(data)
        
        if n_assets != self._n_assets:
            raise ValueError(
                f"Data has {n_assets} assets, but model was initialized with {self._n_assets} assets"
            )
        
        # Initialize conditional covariance matrices
        if sigma is None:
            sigma = np.zeros((n_assets, n_assets, T))
        
        # Get rotation angles
        angles = parameters.rotation_angles
        
        # Compute rotation matrix if not already computed
        if self._rotation_matrix is None or not np.array_equal(angles, parameters.rotation_angles):
            self._rotation_matrix = self.compute_rotation_matrix(angles)
        
        # Compute target covariance if not already computed
        if self._target_covariance is None:
            self._target_covariance = compute_sample_covariance(data)
        
        # Use backcast if provided, otherwise use target covariance
        if backcast is None:
            backcast = self._target_covariance.copy()
        
        # Ensure backcast is positive definite
        backcast = ensure_positive_definite(backcast, epsilon=1e-6)
        
        # Compute conditional covariances
        if HAS_NUMBA:
            # Use Numba-accelerated implementation
            try:
                _compute_rarch_covariance_numba(
                    data, self._rotation_matrix, self._target_covariance, 
                    backcast, sigma
                )
            except Exception as e:
                logger.warning(
                    f"Numba implementation failed: {str(e)}. Falling back to NumPy implementation."
                )
                self._compute_rarch_covariance_numpy(
                    data, self._rotation_matrix, self._target_covariance, 
                    backcast, sigma
                )
        else:
            # Use pure NumPy implementation
            self._compute_rarch_covariance_numpy(
                data, self._rotation_matrix, self._target_covariance, 
                backcast, sigma
            )
        
        return sigma
    
    def _compute_rarch_covariance_numpy(
        self,
        data: np.ndarray,
        rotation_matrix: np.ndarray,
        target_covariance: np.ndarray,
        backcast: np.ndarray,
        sigma: np.ndarray
    ) -> None:
        """
        Pure NumPy implementation of RARCH covariance computation.
        
        Args:
            data: Input data (typically residuals)
            rotation_matrix: Orthogonal rotation matrix
            target_covariance: Target (unconditional) covariance matrix
            backcast: Matrix to use for initializing the covariance process
            sigma: Pre-allocated array for conditional covariances
        """
        T, n_assets = data.shape
        
        # Rotate data
        rotated_data = data @ rotation_matrix
        
        # Compute rotated target covariance
        rotated_target = rotation_matrix.T @ target_covariance @ rotation_matrix
        
        # Initialize rotated conditional covariance
        rotated_sigma = np.zeros((n_assets, n_assets, T))
        
        # Set initial covariance
        rotated_sigma[:, :, 0] = rotation_matrix.T @ backcast @ rotation_matrix
        
        # Compute rotated conditional covariances
        for t in range(1, T):
            # Previous rotated covariance
            prev_cov = rotated_sigma[:, :, t-1]
            
            # Current rotated covariance (diagonal elements only)
            curr_cov = np.diag(np.diag(prev_cov))
            
            # Add off-diagonal elements from target
            off_diag_mask = ~np.eye(n_assets, dtype=bool)
            curr_cov[off_diag_mask] = rotated_target[off_diag_mask]
            
            # Update with current observation
            outer_prod = np.outer(rotated_data[t-1], rotated_data[t-1])
            diag_indices = np.diag_indices(n_assets)
            curr_cov[diag_indices] = 0.05 * outer_prod[diag_indices] + 0.95 * prev_cov[diag_indices]
            
            # Ensure symmetry
            curr_cov = ensure_symmetric(curr_cov)
            
            # Store rotated covariance
            rotated_sigma[:, :, t] = curr_cov
        
        # Rotate back to original space
        for t in range(T):
            sigma[:, :, t] = rotation_matrix @ rotated_sigma[:, :, t] @ rotation_matrix.T
            
            # Ensure positive definiteness
            if not is_positive_definite(sigma[:, :, t]):
                sigma[:, :, t] = nearest_positive_definite(sigma[:, :, t])
    
    def fit(
        self,
        data: np.ndarray,
        starting_values: Optional[Union[np.ndarray, RARCHParameters]] = None,
        backcast: Optional[np.ndarray] = None,
        method: str = "SLSQP",
        options: Optional[Dict[str, Any]] = None,
        constraints: Optional[List[Dict[str, Any]]] = None,
        callback: Optional[ProgressCallback] = None,
        **kwargs: Any
    ) -> MultivariateVolatilityResult:
        """Fit the RARCH model to the provided data.
        
        This method estimates the parameters of the RARCH model from the provided data.
        
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
        
        # Store data dimensions
        T, n_assets = data.shape
        
        # Update n_assets if not already set
        if self._n_assets is None:
            self._n_assets = n_assets
        
        # Compute target covariance
        self._target_covariance = compute_sample_covariance(data)
        
        # Compute backcast if not provided
        if backcast is None:
            backcast = self.compute_backcast(data)
        
        self._backcast = backcast
        
        # Create starting values if not provided
        if starting_values is None:
            starting_values_array = initialize_parameters(
                data, model_type="RARCH", n_assets=n_assets
            )
        elif isinstance(starting_values, RARCHParameters):
            starting_values_array = starting_values.to_array()
        else:
            starting_values_array = starting_values
        
        # Create parameter object from starting values
        starting_params = RARCHParameters.from_array(
            starting_values_array, n_assets=n_assets
        )
        
        # Transform parameters to unconstrained space for optimization
        unconstrained_params = starting_params.transform()
        
        # Set up optimization options
        if options is None:
            options = {}
        
        default_options = {
            'maxiter': 1000,
            'ftol': 1e-8,
            'disp': False
        }
        
        for key, value in default_options.items():
            if key not in options:
                options[key] = value
        
        # Set up optimization constraints
        if constraints is None:
            constraints = self._create_constraints()
        
        # Create callback function for progress reporting
        iteration = [0]
        max_iterations = options.get('maxiter', 1000)
        
        def opt_callback(params: np.ndarray) -> None:
            """Callback function for optimization progress reporting."""
            iteration[0] += 1
            if callback is not None:
                progress = min(iteration[0] / max_iterations, 0.99)
                callback(progress, f"Iteration {iteration[0]}/{max_iterations}")
        
        # Run optimization
        try:
            result = optimize.minimize(
                self._objective_function,
                unconstrained_params,
                args=(data, backcast),
                method=method,
                options=options,
                constraints=constraints,
                callback=opt_callback if callback else None
            )
            
            # Check convergence
            if not result.success:
                warnings.warn(
                    f"Optimization did not converge: {result.message}",
                    RuntimeWarning
                )
            
            # Transform parameters back to constrained space
            params = RARCHParameters.inverse_transform(
                result.x, n_assets=n_assets
            )
            
            # Compute conditional covariances
            covariance = self.compute_covariance(params, data, backcast=backcast)
            
            # Compute conditional correlations
            correlation = self.compute_correlation(params, data, backcast=backcast)
            
            # Compute log-likelihood
            loglik = -result.fun
            
            # Compute standard errors
            std_errors = self._compute_std_errors(result.x, data, backcast)
            
            # Create result object
            model_result = self._create_result_object(
                parameters=params,
                data=data,
                covariance=covariance,
                correlation=correlation,
                loglikelihood=loglik,
                std_errors=std_errors,
                iterations=result.nit,
                convergence=result.success,
                optimization_message=result.message,
                rotation_matrix=self._rotation_matrix
            )
            
            # Update model state
            self._parameters = params
            self._conditional_covariances = covariance
            self._conditional_correlations = correlation
            self._residuals = data
            self._fitted = True
            
            # Report completion
            if callback is not None:
                callback(1.0, "Model estimation complete")
            
            return model_result
            
        except Exception as e:
            # Handle optimization failure
            raise RuntimeError(f"Model estimation failed: {str(e)}") from e
    
    def simulate(
        self,
        n_periods: int,
        burn: int = 0,
        initial_value: Optional[np.ndarray] = None,
        random_state: Optional[Union[int, np.random.Generator]] = None,
        return_covariances: bool = False,
        **kwargs: Any
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Simulate data from the RARCH model.
        
        This method generates simulated data based on the estimated RARCH model.
        
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
        
        # Get model parameters
        params = self._parameters
        n_assets = self._n_assets
        
        # Set up initial covariance
        if initial_value is None:
            if self._conditional_covariances is not None:
                # Use last fitted covariance
                initial_value = self._conditional_covariances[:, :, -1].copy()
            else:
                # Use target covariance
                initial_value = self._target_covariance.copy()
        
        # Ensure initial value is positive definite
        initial_value = ensure_positive_definite(initial_value)
        
        # Total number of periods to simulate (including burn-in)
        total_periods = n_periods + burn
        
        # Initialize arrays for simulated data and covariances
        simulated_data = np.zeros((total_periods, n_assets))
        simulated_covariances = np.zeros((n_assets, n_assets, total_periods))
        
        # Set initial covariance
        simulated_covariances[:, :, 0] = initial_value
        
        # Generate first observation
        chol = np.linalg.cholesky(initial_value)
        simulated_data[0] = rng.standard_normal(n_assets) @ chol.T
        
        # Get rotation matrix
        if self._rotation_matrix is None:
            self._rotation_matrix = self.compute_rotation_matrix(params.rotation_angles)
        
        rotation_matrix = self._rotation_matrix
        
        # Get target covariance
        if self._target_covariance is None:
            raise RuntimeError("Target covariance matrix is not available")
        
        target_covariance = self._target_covariance
        
        # Compute rotated target covariance
        rotated_target = rotation_matrix.T @ target_covariance @ rotation_matrix
        
        # Simulate data
        for t in range(1, total_periods):
            # Rotate previous observation
            rotated_data = simulated_data[t-1] @ rotation_matrix
            
            # Previous rotated covariance
            prev_cov = rotation_matrix.T @ simulated_covariances[:, :, t-1] @ rotation_matrix
            
            # Current rotated covariance (diagonal elements only)
            curr_cov = np.diag(np.diag(prev_cov))
            
            # Add off-diagonal elements from target
            off_diag_mask = ~np.eye(n_assets, dtype=bool)
            curr_cov[off_diag_mask] = rotated_target[off_diag_mask]
            
            # Update with current observation
            outer_prod = np.outer(rotated_data, rotated_data)
            diag_indices = np.diag_indices(n_assets)
            curr_cov[diag_indices] = 0.05 * outer_prod[diag_indices] + 0.95 * prev_cov[diag_indices]
            
            # Ensure symmetry
            curr_cov = ensure_symmetric(curr_cov)
            
            # Rotate back to original space
            curr_cov_original = rotation_matrix @ curr_cov @ rotation_matrix.T
            
            # Ensure positive definiteness
            if not is_positive_definite(curr_cov_original):
                curr_cov_original = nearest_positive_definite(curr_cov_original)
            
            # Store covariance
            simulated_covariances[:, :, t] = curr_cov_original
            
            # Generate observation
            chol = np.linalg.cholesky(curr_cov_original)
            simulated_data[t] = rng.standard_normal(n_assets) @ chol.T
        
        # Discard burn-in periods
        if burn > 0:
            simulated_data = simulated_data[burn:]
            simulated_covariances = simulated_covariances[:, :, burn:]
        
        # Return results
        if return_covariances:
            return simulated_data, simulated_covariances
        else:
            return simulated_data
    
    def forecast(
        self,
        steps: int,
        initial_value: Optional[np.ndarray] = None,
        **kwargs: Any
    ) -> np.ndarray:
        """Forecast conditional covariance matrices.
        
        This method generates forecasts of conditional covariance matrices
        based on the estimated RARCH model.
        
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
        
        # Get model parameters
        params = self._parameters
        n_assets = self._n_assets
        
        # Set up initial covariance
        if initial_value is None:
            if self._conditional_covariances is not None:
                # Use last fitted covariance
                initial_value = self._conditional_covariances[:, :, -1].copy()
            else:
                # Use target covariance
                initial_value = self._target_covariance.copy()
        
        # Ensure initial value is positive definite
        initial_value = ensure_positive_definite(initial_value)
        
        # Initialize array for forecasted covariances
        forecasted_covariances = np.zeros((n_assets, n_assets, steps))
        
        # Set initial covariance
        forecasted_covariances[:, :, 0] = initial_value
        
        # Get rotation matrix
        if self._rotation_matrix is None:
            self._rotation_matrix = self.compute_rotation_matrix(params.rotation_angles)
        
        rotation_matrix = self._rotation_matrix
        
        # Get target covariance
        if self._target_covariance is None:
            raise RuntimeError("Target covariance matrix is not available")
        
        target_covariance = self._target_covariance
        
        # Compute rotated target covariance
        rotated_target = rotation_matrix.T @ target_covariance @ rotation_matrix
        
        # Generate forecasts
        for t in range(1, steps):
            # Previous rotated covariance
            prev_cov = rotation_matrix.T @ forecasted_covariances[:, :, t-1] @ rotation_matrix
            
            # Current rotated covariance (diagonal elements only)
            curr_cov = np.diag(np.diag(prev_cov))
            
            # Add off-diagonal elements from target
            off_diag_mask = ~np.eye(n_assets, dtype=bool)
            curr_cov[off_diag_mask] = rotated_target[off_diag_mask]
            
            # Update diagonal elements
            diag_indices = np.diag_indices(n_assets)
            curr_cov[diag_indices] = 0.95 * prev_cov[diag_indices] + 0.05 * rotated_target[diag_indices]
            
            # Ensure symmetry
            curr_cov = ensure_symmetric(curr_cov)
            
            # Rotate back to original space
            curr_cov_original = rotation_matrix @ curr_cov @ rotation_matrix.T
            
            # Ensure positive definiteness
            if not is_positive_definite(curr_cov_original):
                curr_cov_original = nearest_positive_definite(curr_cov_original)
            
            # Store covariance
            forecasted_covariances[:, :, t] = curr_cov_original
        
        return forecasted_covariances
    
    def _array_to_parameters(self, array: np.ndarray) -> RARCHParameters:
        """Convert a parameter array to a parameter object.
        
        Args:
            array: Parameter array
            
        Returns:
            RARCHParameters: Parameter object
        """
        return RARCHParameters.from_array(array, n_assets=self._n_assets)
    
    def _parameters_to_array(self, parameters: MultivariateVolatilityParameters) -> np.ndarray:
        """Convert a parameter object to a parameter array.
        
        Args:
            parameters: Parameter object
            
        Returns:
            np.ndarray: Parameter array
            
        Raises:
            ValueError: If parameters is not a RARCHParameters object
        """
        if not isinstance(parameters, RARCHParameters):
            raise ValueError(f"Parameters must be RARCHParameters, got {type(parameters)}")
        
        return parameters.to_array()
    
    def _create_constraints(self) -> List[Dict[str, Any]]:
        """Create constraints for optimization.
        
        Returns:
            List[Dict[str, Any]]: List of constraints for optimization
        """
        # RARCH doesn't have specific constraints on the rotation angles
        return []
    
    def _create_bounds(self) -> List[Tuple[float, float]]:
        """Create bounds for optimization parameters.
        
        Returns:
            List[Tuple[float, float]]: List of (lower, upper) bounds for each parameter
        """
        # RARCH doesn't have specific bounds on the rotation angles
        n_params = self._n_assets * (self._n_assets - 1) // 2
        return [(-np.inf, np.inf)] * n_params
    
    def _create_starting_values(self, data: np.ndarray) -> np.ndarray:
        """Create starting values for optimization.
        
        Args:
            data: Input data (typically residuals)
            
        Returns:
            np.ndarray: Starting values for optimization
        """
        return initialize_parameters(data, model_type="RARCH")
    
    def _compute_persistence(self) -> Optional[float]:
        """Compute the persistence of the model.
        
        Returns:
            Optional[float]: Persistence value
        """
        # RARCH doesn't have a simple persistence measure
        return None
    
    def _compute_unconditional_covariance(self) -> Optional[np.ndarray]:
        """Compute the unconditional covariance matrix implied by the model.
        
        Returns:
            Optional[np.ndarray]: Unconditional covariance matrix
        """
        return self._target_covariance
    
    def _create_result_object(
        self,
        parameters: RARCHParameters,
        data: np.ndarray,
        covariance: np.ndarray,
        correlation: Optional[np.ndarray],
        loglikelihood: float,
        std_errors: np.ndarray,
        iterations: int,
        convergence: bool,
        optimization_message: Optional[str] = None,
        rotation_matrix: Optional[np.ndarray] = None,
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
            rotation_matrix: Orthogonal rotation matrix
            **kwargs: Additional keyword arguments for the result object
        
        Returns:
            MultivariateVolatilityResult: Model estimation results
        """
        # Compute t-statistics and p-values
        param_array = self._parameters_to_array(parameters)
        t_stats = np.full_like(std_errors, np.nan)
        p_values = np.full_like(std_errors, np.nan)
        
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
        half_life = None
        if persistence is not None and persistence < 1:
            half_life = np.log(0.5) / np.log(persistence)
        
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
            rotation_matrix=rotation_matrix,
            **kwargs
        )
        
        return result
