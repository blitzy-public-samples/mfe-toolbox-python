'''
Rotated Conditional Correlation (RCC) Multivariate Volatility Model

This module implements the Rotated Conditional Correlation (RCC) multivariate
volatility model, which combines univariate GARCH processes with a dynamic
correlation structure based on the RARCH methodology. The RCC model provides
a flexible approach to multivariate volatility modeling while ensuring
positive definiteness of conditional covariance matrices.

The model uses orthogonal rotation techniques to transform the data into a space
where correlation dynamics can be more efficiently modeled, combining the strengths
of both RARCH and DCC approaches. This implementation supports both synchronous
and asynchronous processing for computationally intensive operations.

Classes:
    RCCParameters: Parameter container for RCC model
    RCCModel: Implementation of the RCC model
'''

import logging
import warnings
import asyncio
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union, cast, overload
)

import numpy as np
from scipy import linalg, optimize, stats

from mfe.core.base import ModelBase
from mfe.core.parameters import (
    MultivariateVolatilityParameters, ParameterBase, ParameterError,
    validate_positive_definite, validate_probability, validate_range,
    validate_non_negative, DCCParameters
)
from mfe.core.results import MultivariateVolatilityResult
from mfe.core.types import (
    AsyncProgressCallback, CovarianceMatrix, CorrelationMatrix, Matrix,
    MultivariateVolatilityType, ProgressCallback, Vector
)
from mfe.models.multivariate.base import CorrelationModelBase
from mfe.models.univariate.base import UnivariateVolatilityModel
from mfe.models.univariate.garch import GARCH
from mfe.utils.matrix_ops import (
    cov2corr, corr2cov, vech, ivech, ensure_symmetric, 
    is_positive_definite, nearest_positive_definite
)
from mfe.models.multivariate.utils import (
    validate_multivariate_data, compute_sample_correlation, compute_sample_covariance,
    transform_correlation_matrix, inverse_transform_correlation_matrix,
    standardize_residuals, compute_conditional_correlations, ensure_positive_definite,
    initialize_parameters
)

# Set up module-level logger
logger = logging.getLogger("mfe.models.multivariate.rcc")

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
    logger.debug("Numba available for RCC model acceleration")
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    HAS_NUMBA = False
    logger.info("Numba not available. RCC model will use pure NumPy implementations.")


@dataclass
class RCCParameters(MultivariateVolatilityParameters):
    """Parameters for the RCC model.
    
    This class contains the parameters for the Rotated Conditional Correlation (RCC) model,
    which combines rotation angles for the orthogonal rotation matrix with DCC-style
    correlation dynamics parameters.
    
    Attributes:
        rotation_angles: Rotation angles for the orthogonal rotation matrix
        a: News parameter for correlation dynamics (must be non-negative)
        b: Decay parameter for correlation dynamics (must be non-negative)
        n_assets: Number of assets in the model
    """
    
    rotation_angles: np.ndarray
    a: float
    b: float
    n_assets: int
    
    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        # Ensure rotation_angles is a NumPy array
        if not isinstance(self.rotation_angles, np.ndarray):
            self.rotation_angles = np.array(self.rotation_angles)
        
        self.validate()
    
    def validate(self) -> None:
        """Validate RCC parameter constraints.
        
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
        
        # Validate DCC parameters
        validate_non_negative(self.a, "a")
        validate_non_negative(self.b, "b")
        
        # Validate stationarity constraint
        if self.a + self.b >= 1:
            raise ParameterError(
                f"RCC stationarity constraint violated: a + b = {self.a + self.b} >= 1"
            )
    
    def to_array(self) -> np.ndarray:
        """Convert parameters to a NumPy array.
        
        Returns:
            np.ndarray: Array representation of parameters
        """
        # Combine rotation angles with a and b parameters
        return np.concatenate([self.rotation_angles, np.array([self.a, self.b])])
    
    @classmethod
    def from_array(cls, array: np.ndarray, n_assets: int, **kwargs: Any) -> 'RCCParameters':
        """Create parameters from a NumPy array.
        
        Args:
            array: Array representation of parameters
            n_assets: Number of assets in the model
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            RCCParameters: Parameter object
        
        Raises:
            ValueError: If the array length doesn't match the expected number of parameters
        """
        expected_angles = n_assets * (n_assets - 1) // 2
        expected_params = expected_angles + 2  # rotation angles + a + b
        
        if len(array) != expected_params:
            raise ValueError(
                f"Array length ({len(array)}) doesn't match expected number of "
                f"parameters ({expected_params}) for {n_assets} assets"
            )
        
        # Extract rotation angles and DCC parameters
        rotation_angles = array[:expected_angles]
        a = array[-2]
        b = array[-1]
        
        return cls(rotation_angles=rotation_angles, a=a, b=b, n_assets=n_assets)
    
    def transform(self) -> np.ndarray:
        """Transform parameters to unconstrained space for optimization.
        
        Returns:
            np.ndarray: Transformed parameters in unconstrained space
        """
        # Rotation angles are already unconstrained
        rotation_angles = self.rotation_angles.copy()
        
        # Transform a and b parameters using logit-like transformation
        # Similar to DCC transformation
        if self.a + self.b >= 1:
            # If constraint is violated, adjust parameters slightly
            sum_ab = self.a + self.b
            a = self.a / (sum_ab + 0.01)
            b = self.b / (sum_ab + 0.01)
        else:
            a = self.a
            b = self.b
        
        lambda_param = a + b
        delta_param = a / lambda_param if lambda_param > 0 else 0.5
        
        transformed_lambda = np.log(lambda_param / (1 - lambda_param))  # logit
        transformed_delta = np.log(delta_param / (1 - delta_param))  # logit
        
        # Combine transformed parameters
        return np.concatenate([rotation_angles, np.array([transformed_lambda, transformed_delta])])
    
    @classmethod
    def inverse_transform(cls, array: np.ndarray, n_assets: int, **kwargs: Any) -> 'RCCParameters':
        """Transform parameters from unconstrained space back to constrained space.
        
        Args:
            array: Parameters in unconstrained space
            n_assets: Number of assets in the model
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            RCCParameters: Parameter object with constrained parameters
        
        Raises:
            ValueError: If the array length doesn't match the expected number of parameters
        """
        expected_angles = n_assets * (n_assets - 1) // 2
        expected_params = expected_angles + 2  # rotation angles + transformed lambda + transformed delta
        
        if len(array) != expected_params:
            raise ValueError(
                f"Array length ({len(array)}) doesn't match expected number of "
                f"parameters ({expected_params}) for {n_assets} assets"
            )
        
        # Extract rotation angles and transformed DCC parameters
        rotation_angles = array[:expected_angles]
        transformed_lambda = array[-2]
        transformed_delta = array[-1]
        
        # Inverse transform lambda and delta
        lambda_param = 1.0 / (1.0 + np.exp(-transformed_lambda))
        delta_param = 1.0 / (1.0 + np.exp(-transformed_delta))
        
        # Compute a and b
        a = lambda_param * delta_param
        b = lambda_param * (1.0 - delta_param)
        
        return cls(rotation_angles=rotation_angles, a=a, b=b, n_assets=n_assets)


class RCCModel(CorrelationModelBase):
    """Rotated Conditional Correlation (RCC) multivariate volatility model.
    
    This class implements the RCC model, which combines univariate GARCH processes
    with a dynamic correlation structure based on the RARCH methodology. The model
    uses orthogonal rotation techniques to transform the data into a space where
    correlation dynamics can be more efficiently modeled.
    
    Attributes:
        name: Model name
        n_assets: Number of assets
        univariate_models: List of univariate volatility models for each asset
        parameters: Model parameters if fitted
        _conditional_correlations: Dynamic correlation matrices if fitted
        _conditional_covariances: Conditional covariance matrices if fitted
        _residuals: Residual data used for fitting
        _standardized_residuals: Standardized residuals used for correlation estimation
        _unconditional_correlation: Unconditional correlation matrix
        _rotation_matrix: Orthogonal rotation matrix
    """
    
    def __init__(
        self,
        n_assets: Optional[int] = None,
        univariate_models: Optional[List[UnivariateVolatilityModel]] = None,
        name: str = "RCC"
    ):
        """Initialize the RCC model.
        
        Args:
            n_assets: Number of assets (if None, determined from data)
            univariate_models: List of univariate volatility models for each asset
                              (if None, GARCH(1,1) models are used)
            name: Model name
        """
        super().__init__(name=name, n_assets=n_assets)
        
        self._univariate_models = univariate_models
        self._standardized_residuals: Optional[np.ndarray] = None
        self._unconditional_correlation: Optional[np.ndarray] = None
        self._rotation_matrix: Optional[np.ndarray] = None
        self._rotated_residuals: Optional[np.ndarray] = None
        self._rotated_correlation: Optional[np.ndarray] = None
    
    @property
    def univariate_models(self) -> Optional[List[UnivariateVolatilityModel]]:
        """Get the univariate volatility models.
        
        Returns:
            Optional[List[UnivariateVolatilityModel]]: List of univariate models if set,
                                                     None otherwise
        """
        return self._univariate_models
    
    @property
    def unconditional_correlation(self) -> Optional[np.ndarray]:
        """Get the unconditional correlation matrix.
        
        Returns:
            Optional[np.ndarray]: Unconditional correlation matrix if the model has been fitted,
                                 None otherwise
        
        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self._unconditional_correlation
    
    @property
    def rotation_matrix(self) -> Optional[np.ndarray]:
        """Get the orthogonal rotation matrix.
        
        Returns:
            Optional[np.ndarray]: Rotation matrix if the model has been fitted,
                                 None otherwise
        
        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self._rotation_matrix
    
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
    
    def _compute_standardized_residuals(self, data: np.ndarray) -> np.ndarray:
        """Compute standardized residuals using univariate volatilities.
        
        Args:
            data: Input data array with shape (T, n_assets)
        
        Returns:
            np.ndarray: Standardized residuals with shape (T, n_assets)
        """
        T, n_assets = data.shape
        
        # Compute univariate volatilities if not already available
        if self._univariate_volatilities is None:
            self._compute_univariate_volatilities(data)
        
        # Standardize residuals
        std_residuals = np.zeros_like(data)
        for i in range(n_assets):
            std_residuals[:, i] = data[:, i] / self._univariate_volatilities[:, i]
        
        self._standardized_residuals = std_residuals
        return std_residuals
    
    def _compute_rotated_residuals(self, std_residuals: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
        """Compute rotated standardized residuals.
        
        Args:
            std_residuals: Standardized residuals with shape (T, n_assets)
            rotation_matrix: Orthogonal rotation matrix with shape (n_assets, n_assets)
        
        Returns:
            np.ndarray: Rotated standardized residuals with shape (T, n_assets)
        """
        # Apply rotation to standardized residuals
        rotated_residuals = std_residuals @ rotation_matrix
        
        self._rotated_residuals = rotated_residuals
        return rotated_residuals
    
    @jit(nopython=True, cache=True)
    def _compute_rcc_correlations_numba(
        self,
        rotated_residuals: np.ndarray,
        a: float,
        b: float,
        unconditional_corr: np.ndarray
    ) -> np.ndarray:
        """Compute RCC correlation matrices using Numba acceleration.
        
        Args:
            rotated_residuals: Rotated standardized residuals with shape (T, n_assets)
            a: News parameter
            b: Decay parameter
            unconditional_corr: Unconditional correlation matrix of rotated residuals
        
        Returns:
            np.ndarray: Correlation matrices with shape (n_assets, n_assets, T)
        """
        T, n_assets = rotated_residuals.shape
        
        # Initialize correlation matrices
        correlations = np.zeros((n_assets, n_assets, T))
        
        # Initialize Q with unconditional correlation
        Qt = unconditional_corr.copy()
        
        # Compute correlation matrices
        for t in range(T):
            # Update Qt
            epsilon = rotated_residuals[t, :]
            outer_prod = np.outer(epsilon, epsilon)
            
            Qt = (1 - a - b) * unconditional_corr + a * outer_prod + b * Qt
            
            # Compute correlation matrix from Qt
            q_diag = np.sqrt(np.diag(Qt))
            q_diag_inv = np.zeros_like(q_diag)
            
            # Handle potential zeros in diagonal
            for i in range(n_assets):
                if q_diag[i] > 1e-8:
                    q_diag_inv[i] = 1.0 / q_diag[i]
            
            # Compute correlation matrix: D^(-1/2) * Q * D^(-1/2)
            D_inv_sqrt = np.diag(q_diag_inv)
            correlations[:, :, t] = D_inv_sqrt @ Qt @ D_inv_sqrt
            
            # Ensure diagonal is exactly 1
            for i in range(n_assets):
                correlations[i, i, t] = 1.0
        
        return correlations
    
    def _compute_rcc_correlations_numpy(
        self,
        rotated_residuals: np.ndarray,
        a: float,
        b: float,
        unconditional_corr: np.ndarray
    ) -> np.ndarray:
        """Compute RCC correlation matrices using NumPy.
        
        Args:
            rotated_residuals: Rotated standardized residuals with shape (T, n_assets)
            a: News parameter
            b: Decay parameter
            unconditional_corr: Unconditional correlation matrix of rotated residuals
        
        Returns:
            np.ndarray: Correlation matrices with shape (n_assets, n_assets, T)
        """
        T, n_assets = rotated_residuals.shape
        
        # Initialize correlation matrices
        correlations = np.zeros((n_assets, n_assets, T))
        
        # Initialize Q with unconditional correlation
        Qt = unconditional_corr.copy()
        
        # Compute correlation matrices
        for t in range(T):
            # Update Qt
            epsilon = rotated_residuals[t, :].reshape(-1, 1)
            outer_prod = epsilon @ epsilon.T
            
            Qt = (1 - a - b) * unconditional_corr + a * outer_prod + b * Qt
            
            # Compute correlation matrix from Qt
            q_diag = np.sqrt(np.diag(Qt))
            q_diag_inv = np.zeros_like(q_diag)
            
            # Handle potential zeros in diagonal
            mask = q_diag > 1e-8
            q_diag_inv[mask] = 1.0 / q_diag[mask]
            
            # Compute correlation matrix: D^(-1/2) * Q * D^(-1/2)
            D_inv_sqrt = np.diag(q_diag_inv)
            correlations[:, :, t] = D_inv_sqrt @ Qt @ D_inv_sqrt
            
            # Ensure diagonal is exactly 1
            np.fill_diagonal(correlations[:, :, t], 1.0)
        
        return correlations
    
    def compute_correlation(
        self,
        parameters: RCCParameters,
        data: np.ndarray,
        correlation: Optional[np.ndarray] = None,
        backcast: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute conditional correlation matrices for the given parameters and data.
        
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
        
        # Compute standardized residuals if not already available
        if self._standardized_residuals is None:
            self._compute_standardized_residuals(data)
        
        std_residuals = self._standardized_residuals
        
        # Extract rotation angles and compute rotation matrix
        rotation_angles = parameters.rotation_angles
        if self._rotation_matrix is None or not np.array_equal(rotation_angles, parameters.rotation_angles):
            self._rotation_matrix = self.compute_rotation_matrix(rotation_angles)
        
        rotation_matrix = self._rotation_matrix
        
        # Compute rotated standardized residuals
        rotated_residuals = self._compute_rotated_residuals(std_residuals, rotation_matrix)
        
        # Compute unconditional correlation of rotated residuals if not already available
        if self._rotated_correlation is None:
            self._rotated_correlation = compute_sample_correlation(rotated_residuals)
        
        rotated_unconditional_corr = self._rotated_correlation
        
        # Allocate correlation matrices if not provided
        if correlation is None:
            correlation = np.zeros((n_assets, n_assets, T))
        
        # Extract DCC parameters
        a = parameters.a
        b = parameters.b
        
        # Compute rotated correlation matrices
        if HAS_NUMBA:
            rotated_correlation = self._compute_rcc_correlations_numba(
                rotated_residuals, a, b, rotated_unconditional_corr
            )
        else:
            rotated_correlation = self._compute_rcc_correlations_numpy(
                rotated_residuals, a, b, rotated_unconditional_corr
            )
        
        # Rotate correlation matrices back to original space
        for t in range(T):
            correlation[:, :, t] = rotation_matrix @ rotated_correlation[:, :, t] @ rotation_matrix.T
            
            # Ensure the result is a valid correlation matrix
            np.fill_diagonal(correlation[:, :, t], 1.0)
            
            # Ensure symmetry
            correlation[:, :, t] = ensure_symmetric(correlation[:, :, t])
            
            # Ensure positive definiteness
            if not is_positive_definite(correlation[:, :, t]):
                correlation[:, :, t] = nearest_positive_definite(correlation[:, :, t])
        
        return correlation
    
    def _array_to_parameters(self, array: np.ndarray) -> RCCParameters:
        """Convert a parameter array to a parameter object.
        
        Args:
            array: Parameter array
        
        Returns:
            RCCParameters: Parameter object
        """
        return RCCParameters.from_array(array, n_assets=self._n_assets)
    
    def _parameters_to_array(self, parameters: MultivariateVolatilityParameters) -> np.ndarray:
        """Convert a parameter object to a parameter array.
        
        Args:
            parameters: Parameter object
        
        Returns:
            np.ndarray: Parameter array
        
        Raises:
            ValueError: If parameters is not a RCCParameters object
        """
        if not isinstance(parameters, RCCParameters):
            raise ValueError(f"Parameters must be RCCParameters, got {type(parameters)}")
        
        return parameters.to_array()
    
    def _objective_function(
        self,
        parameters: np.ndarray,
        std_residuals: np.ndarray
    ) -> float:
        """Objective function for RCC parameter optimization (negative log-likelihood).
        
        Args:
            parameters: Parameter array
            std_residuals: Standardized residuals
        
        Returns:
            float: Negative log-likelihood value
        """
        try:
            # Convert parameters to parameter object
            param_obj = self._array_to_parameters(parameters)
            
            # Compute correlation matrices
            n_assets = self._n_assets
            T = std_residuals.shape[0]
            
            # Extract rotation angles and compute rotation matrix
            rotation_angles = param_obj.rotation_angles
            rotation_matrix = self.compute_rotation_matrix(rotation_angles)
            
            # Compute rotated standardized residuals
            rotated_residuals = std_residuals @ rotation_matrix
            
            # Compute unconditional correlation of rotated residuals
            rotated_unconditional_corr = compute_sample_correlation(rotated_residuals)
            
            # Extract DCC parameters
            a = param_obj.a
            b = param_obj.b
            
            # Compute rotated correlation matrices
            if HAS_NUMBA:
                rotated_correlation = self._compute_rcc_correlations_numba(
                    rotated_residuals, a, b, rotated_unconditional_corr
                )
            else:
                rotated_correlation = self._compute_rcc_correlations_numpy(
                    rotated_residuals, a, b, rotated_unconditional_corr
                )
            
            # Compute log-likelihood
            loglik = 0.0
            
            for t in range(T):
                # Rotate correlation matrix back to original space
                Rt = rotation_matrix @ rotated_correlation[:, :, t] @ rotation_matrix.T
                
                # Ensure the result is a valid correlation matrix
                np.fill_diagonal(Rt, 1.0)
                Rt = ensure_symmetric(Rt)
                
                # Ensure positive definiteness
                if not is_positive_definite(Rt):
                    Rt = nearest_positive_definite(Rt)
                
                try:
                    # Try Cholesky decomposition for numerical stability
                    chol = np.linalg.cholesky(Rt)
                    # Compute log determinant using Cholesky factor
                    log_det = 2 * np.sum(np.log(np.diag(chol)))
                    # Compute quadratic form using Cholesky factor
                    epsilon = std_residuals[t, :]
                    quad_form = np.sum(np.linalg.solve(chol, epsilon) ** 2)
                except np.linalg.LinAlgError:
                    # If Cholesky decomposition fails, use eigenvalue decomposition
                    eigvals = np.linalg.eigvalsh(Rt)
                    # Ensure all eigenvalues are positive
                    eigvals = np.maximum(eigvals, 1e-8)
                    # Compute log determinant using eigenvalues
                    log_det = np.sum(np.log(eigvals))
                    # Compute quadratic form using matrix inverse
                    epsilon = std_residuals[t, :]
                    quad_form = epsilon @ np.linalg.solve(Rt, epsilon)
                
                # Compute log-likelihood contribution
                loglik += -0.5 * (n_assets * np.log(2 * np.pi) + log_det + quad_form)
            
            # Return negative log-likelihood for minimization
            return -loglik
        except (ValueError, ParameterError, np.linalg.LinAlgError) as e:
            # Return a large value if parameters are invalid
            logger.warning(f"Error in objective function: {str(e)}")
            return 1e10
    
    def _create_starting_values(self, data: np.ndarray) -> np.ndarray:
        """Create starting values for RCC parameter optimization.
        
        Args:
            data: Input data array with shape (T, n_assets)
        
        Returns:
            np.ndarray: Starting values for optimization
        """
        return initialize_parameters(data, model_type="RCC")
    
    def _create_bounds(self) -> List[Tuple[float, float]]:
        """Create bounds for RCC parameter optimization.
        
        Returns:
            List[Tuple[float, float]]: List of (lower, upper) bounds for each parameter
        """
        n_assets = self._n_assets
        n_rot_params = n_assets * (n_assets - 1) // 2
        
        # Bounds for rotation angles (unconstrained)
        rot_bounds = [(-np.pi, np.pi)] * n_rot_params
        
        # Bounds for a and b parameters
        dcc_bounds = [(0.0, 0.3), (0.5, 0.999)]
        
        return rot_bounds + dcc_bounds
    
    def _create_constraints(self) -> List[Dict[str, Any]]:
        """Create constraints for RCC parameter optimization.
        
        Returns:
            List[Dict[str, Any]]: List of constraints for optimization
        """
        n_assets = self._n_assets
        n_rot_params = n_assets * (n_assets - 1) // 2
        
        # Constraint: a + b < 1
        def constraint_func(x: np.ndarray) -> float:
            a = x[-2]
            b = x[-1]
            return 1 - a - b
        
        return [
            {'type': 'ineq', 'fun': constraint_func}
        ]
    
    def _compute_std_errors(
        self,
        parameters: np.ndarray,
        std_residuals: np.ndarray
    ) -> np.ndarray:
        """Compute standard errors for RCC parameter estimates.
        
        Args:
            parameters: Parameter array
            std_residuals: Standardized residuals
        
        Returns:
            np.ndarray: Standard errors for parameter estimates
        """
        # Compute Hessian using finite differences
        def objective(params: np.ndarray) -> float:
            return self._objective_function(params, std_residuals)
        
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
    
    def _compute_persistence(self) -> float:
        """Compute the persistence of the RCC model.
        
        Returns:
            float: Persistence value
        """
        if not self._fitted or self._parameters is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        if isinstance(self._parameters, RCCParameters):
            return self._parameters.a + self._parameters.b
        else:
            raise TypeError("Parameters must be RCCParameters")
    
    def fit(
        self,
        data: np.ndarray,
        starting_values: Optional[Union[np.ndarray, RCCParameters]] = None,
        backcast: Optional[np.ndarray] = None,
        method: str = "SLSQP",
        options: Optional[Dict[str, Any]] = None,
        constraints: Optional[List[Dict[str, Any]]] = None,
        callback: Optional[ProgressCallback] = None,
        **kwargs: Any
    ) -> MultivariateVolatilityResult:
        """Fit the RCC model to the provided data.
        
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
        
        self._compute_standardized_residuals(data)
        std_residuals = self._standardized_residuals
        
        # Step 3: Compute unconditional correlation
        logger.info("Computing unconditional correlation")
        if callback:
            callback(0.4, "Computing unconditional correlation")
        
        self._unconditional_correlation = compute_sample_correlation(std_residuals)
        
        # Step 4: Estimate RCC parameters
        logger.info("Estimating RCC parameters")
        if callback:
            callback(0.5, "Estimating RCC parameters")
        
        # Use provided starting values if available
        if starting_values is not None:
            if isinstance(starting_values, RCCParameters):
                start_params = self._parameters_to_array(starting_values)
            else:
                start_params = starting_values
        else:
            start_params = self._create_starting_values(data)
        
        # Set up optimization options
        if options is None:
            options = {'maxiter': 1000, 'disp': False}
        
        # Set up constraints
        if constraints is None:
            constraints = self._create_constraints()
        
        # Set up bounds
        bounds = self._create_bounds()
        
        # Define callback function for optimization progress
        opt_callback = None
        if callback:
            def opt_callback(xk: np.ndarray) -> None:
                callback(0.5 + 0.3 * np.random.random(), "Optimizing RCC parameters...")
        
        # Run optimization
        try:
            result = optimize.minimize(
                self._objective_function,
                start_params,
                args=(std_residuals,),
                method=method,
                bounds=bounds,
                constraints=constraints,
                options=options,
                callback=opt_callback
            )
            
            if not result.success:
                warnings.warn(
                    f"RCC parameter optimization did not converge: {result.message}",
                    UserWarning
                )
        except Exception as e:
            raise RuntimeError(f"Error during RCC parameter optimization: {str(e)}")
        
        # Extract optimized parameters
        opt_params = result.x
        
        # Create parameter object
        parameters = self._array_to_parameters(opt_params)
        
        # Step 5: Compute conditional correlation and covariance matrices
        logger.info("Computing conditional correlation and covariance matrices")
        if callback:
            callback(0.8, "Computing conditional correlation and covariance matrices")
        
        self._conditional_correlations = self.compute_correlation(parameters, data)
        self._conditional_covariances = self.compute_covariance(parameters, data)
        
        # Step 6: Compute standard errors and other statistics
        logger.info("Computing standard errors and statistics")
        if callback:
            callback(0.9, "Computing standard errors and statistics")
        
        std_errors = self._compute_std_errors(opt_params, std_residuals)
        
        # Compute t-statistics and p-values
        t_stats = np.full_like(std_errors, np.nan)
        p_values = np.full_like(std_errors, np.nan)
        
        # Compute t-statistics and p-values where standard errors are available
        mask = ~np.isnan(std_errors) & (std_errors > 0)
        if np.any(mask):
            t_stats[mask] = opt_params[mask] / std_errors[mask]
            p_values[mask] = 2 * (1 - stats.t.cdf(np.abs(t_stats[mask]), T - len(opt_params)))
        
        # Compute log-likelihood
        loglik = -result.fun
        
        # Compute information criteria
        n_params = len(opt_params)
        aic = -2 * loglik + 2 * n_params
        bic = -2 * loglik + n_params * np.log(T)
        hqic = -2 * loglik + 2 * n_params * np.log(np.log(T))
        
        # Compute persistence and half-life
        persistence = self._compute_persistence()
        half_life = np.log(0.5) / np.log(persistence) if persistence < 1 else np.inf
        
        # Create result object
        self._parameters = parameters
        self._fitted = True
        
        result = MultivariateVolatilityResult(
            model_name=self.name,
            parameters=parameters,
            convergence=result.success,
            iterations=result.nit,
            log_likelihood=loglik,
            aic=aic,
            bic=bic,
            hqic=hqic,
            std_errors=std_errors,
            t_stats=t_stats,
            p_values=p_values,
            covariance_matrix=None,  # Covariance matrix of parameter estimates
            optimization_message=result.message,
            conditional_covariances=self._conditional_covariances,
            conditional_correlations=self._conditional_correlations,
            standardized_residuals=std_residuals,
            n_assets=n_assets,
            persistence=persistence,
            half_life=half_life,
            unconditional_covariance=None,
            residuals=data,
            rotation_matrix=self._rotation_matrix
        )
        
        if callback:
            callback(1.0, "Estimation complete")
        
        return result
    
    async def fit_async(
        self,
        data: np.ndarray,
        starting_values: Optional[Union[np.ndarray, RCCParameters]] = None,
        backcast: Optional[np.ndarray] = None,
        method: str = "SLSQP",
        options: Optional[Dict[str, Any]] = None,
        constraints: Optional[List[Dict[str, Any]]] = None,
        callback: Optional[AsyncProgressCallback] = None,
        **kwargs: Any
    ) -> MultivariateVolatilityResult:
        """Asynchronously fit the RCC model to the provided data.
        
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
    
    def simulate(
        self,
        n_periods: int,
        burn: int = 0,
        initial_value: Optional[np.ndarray] = None,
        random_state: Optional[Union[int, np.random.Generator]] = None,
        return_covariances: bool = False,
        **kwargs: Any
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Simulate data from the RCC model.
        
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
        
        # Extract parameters
        parameters = self._parameters
        if not isinstance(parameters, RCCParameters):
            raise TypeError("Parameters must be RCCParameters")
        
        rotation_angles = parameters.rotation_angles
        a = parameters.a
        b = parameters.b
        
        # Get rotation matrix
        rotation_matrix = self.compute_rotation_matrix(rotation_angles)
        
        # Get unconditional correlation of rotated residuals
        if self._rotated_correlation is None:
            # If not available, use identity matrix as fallback
            rotated_unconditional_corr = np.eye(n_assets)
        else:
            rotated_unconditional_corr = self._rotated_correlation
        
        # Allocate arrays for simulated data, correlations, and covariances
        simulated_data = np.zeros((total_periods, n_assets))
        correlations = np.zeros((n_assets, n_assets, total_periods))
        covariances = np.zeros((n_assets, n_assets, total_periods))
        
        # Initialize Q with unconditional correlation
        Qt = rotated_unconditional_corr.copy()
        
        # Simulate data
        for t in range(total_periods):
            # Generate standard normal random variables
            z = rng.standard_normal(n_assets)
            
            # Update Qt
            if t > 0:
                # Rotate standardized residuals
                epsilon = simulated_data[t-1, :] / univariate_volatilities[t-1, :]
                rotated_epsilon = epsilon @ rotation_matrix
                
                # Update Qt
                outer_prod = np.outer(rotated_epsilon, rotated_epsilon)
                Qt = (1 - a - b) * rotated_unconditional_corr + a * outer_prod + b * Qt
            
            # Compute correlation matrix from Qt
            q_diag = np.sqrt(np.diag(Qt))
            q_diag_inv = np.zeros_like(q_diag)
            
            # Handle potential zeros in diagonal
            mask = q_diag > 1e-8
            q_diag_inv[mask] = 1.0 / q_diag[mask]
            
            # Compute correlation matrix: D^(-1/2) * Q * D^(-1/2)
            D_inv_sqrt = np.diag(q_diag_inv)
            rotated_Rt = D_inv_sqrt @ Qt @ D_inv_sqrt
            
            # Ensure diagonal is exactly 1
            np.fill_diagonal(rotated_Rt, 1.0)
            
            # Rotate correlation matrix back to original space
            Rt = rotation_matrix @ rotated_Rt @ rotation_matrix.T
            
            # Ensure the result is a valid correlation matrix
            np.fill_diagonal(Rt, 1.0)
            Rt = ensure_symmetric(Rt)
            
            # Ensure positive definiteness
            if not is_positive_definite(Rt):
                Rt = nearest_positive_definite(Rt)
            
            # Store correlation matrix
            correlations[:, :, t] = Rt
            
            # Create diagonal matrix of volatilities
            D = np.diag(univariate_volatilities[t, :])
            
            # Compute covariance matrix: D * R * D
            cov_t = D @ Rt @ D
            covariances[:, :, t] = cov_t
            
            # Generate correlated random variables
            # Compute Cholesky decomposition of correlation matrix
            try:
                chol = np.linalg.cholesky(Rt)
                # Apply correlation structure
                corr_z = chol @ z
            except np.linalg.LinAlgError:
                # If Cholesky decomposition fails, use eigenvalue decomposition
                eigvals, eigvecs = np.linalg.eigh(Rt)
                # Ensure all eigenvalues are positive
                eigvals = np.maximum(eigvals, 1e-8)
                # Compute square root of correlation matrix
                sqrt_corr = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                # Apply correlation structure
                corr_z = sqrt_corr @ z
            
            # Apply volatility
            simulated_data[t, :] = univariate_volatilities[t, :] * corr_z
        
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
        """Asynchronously simulate data from the RCC model.
        
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
        
        # Extract parameters
        parameters = self._parameters
        if not isinstance(parameters, RCCParameters):
            raise TypeError("Parameters must be RCCParameters")
        
        rotation_angles = parameters.rotation_angles
        a = parameters.a
        b = parameters.b
        
        # Get rotation matrix
        rotation_matrix = self.compute_rotation_matrix(rotation_angles)
        
        # Get unconditional correlation of rotated residuals
        if self._rotated_correlation is None:
            # If not available, use identity matrix as fallback
            rotated_unconditional_corr = np.eye(n_assets)
        else:
            rotated_unconditional_corr = self._rotated_correlation
        
        # Get standardized residuals
        if self._standardized_residuals is None:
            raise RuntimeError("Standardized residuals not available")
        
        std_residuals = self._standardized_residuals
        
        # Get rotated standardized residuals
        if self._rotated_residuals is None:
            rotated_residuals = std_residuals @ rotation_matrix
        else:
            rotated_residuals = self._rotated_residuals
        
        # Get last Q matrix from the fitted model
        T = std_residuals.shape[0]
        
        # Initialize Q with the last Q matrix from the fitted model
        # For simplicity, we'll recompute it from the last observation
        last_rotated_epsilon = rotated_residuals[T-1, :]
        last_outer_prod = np.outer(last_rotated_epsilon, last_rotated_epsilon)
        Qt = (1 - a - b) * rotated_unconditional_corr + a * last_outer_prod + b * rotated_unconditional_corr
        
        # Allocate arrays for forecasted correlations and covariances
        forecasted_correlations = np.zeros((n_assets, n_assets, steps))
        forecasted_covariances = np.zeros((n_assets, n_assets, steps))
        
        # Forecast correlation matrices
        for t in range(steps):
            # Update Qt
            if t == 0:
                # Use last observation for the first forecast
                pass  # Qt is already initialized with the last observation
            else:
                # For subsequent forecasts, use the unconditional expectation of outer product
                Qt = (1 - a - b) * rotated_unconditional_corr + a * rotated_unconditional_corr + b * Qt
            
            # Compute correlation matrix from Qt
            q_diag = np.sqrt(np.diag(Qt))
            q_diag_inv = np.zeros_like(q_diag)
            
            # Handle potential zeros in diagonal
            mask = q_diag > 1e-8
            q_diag_inv[mask] = 1.0 / q_diag[mask]
            
            # Compute correlation matrix: D^(-1/2) * Q * D^(-1/2)
            D_inv_sqrt = np.diag(q_diag_inv)
            rotated_Rt = D_inv_sqrt @ Qt @ D_inv_sqrt
            
            # Ensure diagonal is exactly 1
            np.fill_diagonal(rotated_Rt, 1.0)
            
            # Rotate correlation matrix back to original space
            Rt = rotation_matrix @ rotated_Rt @ rotation_matrix.T
            
            # Ensure the result is a valid correlation matrix
            np.fill_diagonal(Rt, 1.0)
            Rt = ensure_symmetric(Rt)
            
            # Ensure positive definiteness
            if not is_positive_definite(Rt):
                Rt = nearest_positive_definite(Rt)
            
            # Store correlation matrix
            forecasted_correlations[:, :, t] = Rt
            
            # Create diagonal matrix of volatilities
            D = np.diag(univariate_forecasts[t, :])
            
            # Compute covariance matrix: D * R * D
            forecasted_covariances[:, :, t] = D @ Rt @ D
        
        return forecasted_covariances
    
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
    
    def __str__(self) -> str:
        """Return a string representation of the model.
        
        Returns:
            str: String representation
        """
        if not self._fitted:
            return f"{self.name} model (not fitted)"
        
        n_assets = self._n_assets if self._n_assets is not None else 0
        
        if isinstance(self._parameters, RCCParameters):
            param_str = (
                f"a = {self._parameters.a:.4f}, "
                f"b = {self._parameters.b:.4f}, "
                f"rotation angles: {len(self._parameters.rotation_angles)} parameters"
            )
        else:
            param_str = "Unknown parameter type"
        
        persistence = self._compute_persistence()
        
        return (
            f"{self.name} model with {n_assets} assets\n"
            f"Parameters: {param_str}\n"
            f"Persistence: {persistence:.4f}\n"
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
    Register Numba JIT-compiled functions for RCC model.
    
    This function is called during module initialization to register
    performance-critical functions for JIT compilation if Numba is available.
    """
    if HAS_NUMBA:
        # The functions are already decorated with @jit, so no additional
        # registration is needed here. This function is kept for consistency
        # with the module structure and potential future enhancements.
        logger.debug("RCC model Numba JIT functions registered")
    else:
        logger.info("Numba not available. RCC model will use pure NumPy implementations.")


# Initialize the module
_register_numba_functions()
