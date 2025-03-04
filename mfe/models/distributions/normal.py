# mfe/models/distributions/normal.py
"""
Normal distribution implementations for the MFE Toolbox.

This module provides implementations of the normal (Gaussian) distribution,
including both univariate and multivariate variants. These distributions are
fundamental in financial econometrics, serving as the basis for many models
and statistical tests.

The module includes Numba-accelerated implementations of probability density
functions, cumulative distribution functions, quantile functions, random
number generation, and log-likelihood evaluation for both univariate and
multivariate normal distributions.

Key features:
- Univariate normal distribution with mean and standard deviation parameters
- Multivariate normal distribution with mean vector and covariance matrix
- Numba-accelerated core functions for performance-critical operations
- Integration with SciPy's statistical distributions for core functionality
- Comprehensive parameter validation with dataclasses
- Support for both scalar and vectorized operations
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import (
    Any, Dict, List, Optional, Tuple, Union, cast, overload
)

import numpy as np
from scipy import stats
from numba import jit

from mfe.core.base import DistributionBase
from mfe.core.parameters import (
    ParameterBase, ParameterError, validate_positive, validate_positive_definite,
    transform_positive, inverse_transform_positive
)
from mfe.core.exceptions import (
    DistributionError, NumericError, raise_parameter_error, warn_numeric
)
from mfe.core.types import (
    Vector, Matrix, DistributionType, DistributionLike, ParameterVector,
    PDFFunction, CDFFunction, PPFFunction, RVSFunction
)
from mfe.models.distributions.base import (
    BaseDistribution, ContinuousDistribution, NumbaDistribution
)


@dataclass
class NormalParams(ParameterBase):
    """Parameters for the univariate normal distribution.
    
    Attributes:
        mu: Mean parameter
        sigma: Standard deviation parameter (must be positive)
    """
    
    mu: float = 0.0
    sigma: float = 1.0
    
    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate normal distribution parameter constraints.
        
        Raises:
            ParameterError: If parameter constraints are violated
        """
        # Validate sigma (must be positive)
        validate_positive(self.sigma, "sigma")
    
    def to_array(self) -> np.ndarray:
        """Convert parameters to a NumPy array.
        
        Returns:
            np.ndarray: Array representation of parameters [mu, sigma]
        """
        return np.array([self.mu, self.sigma])
    
    @classmethod
    def from_array(cls, array: np.ndarray, **kwargs: Any) -> 'NormalParams':
        """Create parameters from a NumPy array.
        
        Args:
            array: Array representation of parameters [mu, sigma]
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            NormalParams: Parameter object
            
        Raises:
            ValueError: If the array length is not 2
        """
        if len(array) != 2:
            raise ValueError(f"Array length must be 2, got {len(array)}")
        
        return cls(mu=array[0], sigma=array[1])
    
    def transform(self) -> np.ndarray:
        """Transform parameters to unconstrained space for optimization.
        
        Returns:
            np.ndarray: Transformed parameters in unconstrained space [mu, log(sigma)]
        """
        # mu is already unconstrained
        # Transform sigma to unconstrained space (log)
        transformed_sigma = transform_positive(self.sigma)
        
        return np.array([self.mu, transformed_sigma])
    
    @classmethod
    def inverse_transform(cls, array: np.ndarray, **kwargs: Any) -> 'NormalParams':
        """Transform parameters from unconstrained space back to constrained space.
        
        Args:
            array: Parameters in unconstrained space [mu, log(sigma)]
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            NormalParams: Parameter object with constrained parameters
            
        Raises:
            ValueError: If the array length is not 2
        """
        if len(array) != 2:
            raise ValueError(f"Array length must be 2, got {len(array)}")
        
        # Extract transformed parameters
        mu = array[0]  # mu is already unconstrained
        transformed_sigma = array[1]
        
        # Inverse transform sigma
        sigma = inverse_transform_positive(transformed_sigma)
        
        return cls(mu=mu, sigma=sigma)


@dataclass
class MultivariateNormalParams(ParameterBase):
    """Parameters for the multivariate normal distribution.
    
    Attributes:
        mu: Mean vector
        sigma: Covariance matrix (must be positive definite)
    """
    
    mu: np.ndarray
    sigma: np.ndarray
    
    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        # Ensure mu and sigma are NumPy arrays
        if not isinstance(self.mu, np.ndarray):
            self.mu = np.array(self.mu, dtype=np.float64)
        if not isinstance(self.sigma, np.ndarray):
            self.sigma = np.array(self.sigma, dtype=np.float64)
        
        self.validate()
    
    def validate(self) -> None:
        """Validate multivariate normal distribution parameter constraints.
        
        Raises:
            ParameterError: If parameter constraints are violated
        """
        # Validate mu (must be a vector)
        if self.mu.ndim != 1:
            raise ParameterError(
                f"Mean vector must be 1-dimensional, got {self.mu.ndim} dimensions",
                param_name="mu",
                param_value=self.mu.shape
            )
        
        # Validate sigma (must be a square matrix)
        if self.sigma.ndim != 2:
            raise ParameterError(
                f"Covariance matrix must be 2-dimensional, got {self.sigma.ndim} dimensions",
                param_name="sigma",
                param_value=self.sigma.shape
            )
        
        if self.sigma.shape[0] != self.sigma.shape[1]:
            raise ParameterError(
                f"Covariance matrix must be square, got shape {self.sigma.shape}",
                param_name="sigma",
                param_value=self.sigma.shape
            )
        
        # Validate dimensions match
        if self.mu.shape[0] != self.sigma.shape[0]:
            raise ParameterError(
                f"Mean vector length ({self.mu.shape[0]}) must match covariance matrix dimension ({self.sigma.shape[0]})",
                param_name="mu, sigma",
                param_value=(self.mu.shape, self.sigma.shape)
            )
        
        # Validate sigma is positive definite
        try:
            validate_positive_definite(self.sigma, "sigma")
        except ParameterError as e:
            # Add more context to the error
            raise ParameterError(
                "Covariance matrix must be positive definite",
                param_name="sigma",
                param_value=self.sigma,
                details=str(e)
            )
    
    def to_array(self) -> np.ndarray:
        """Convert parameters to a NumPy array.
        
        This flattens the parameters into a 1D array for optimization.
        
        Returns:
            np.ndarray: Array representation of parameters [mu_1, mu_2, ..., sigma_11, sigma_12, ...]
        """
        # Flatten the mean vector
        mu_flat = self.mu.flatten()
        
        # Flatten the covariance matrix (only lower triangular part due to symmetry)
        n = self.sigma.shape[0]
        sigma_flat = np.zeros(n * (n + 1) // 2)
        idx = 0
        for i in range(n):
            for j in range(i + 1):
                sigma_flat[idx] = self.sigma[i, j]
                idx += 1
        
        # Concatenate flattened parameters
        return np.concatenate([mu_flat, sigma_flat])
    
    @classmethod
    def from_array(cls, array: np.ndarray, dim: int, **kwargs: Any) -> 'MultivariateNormalParams':
        """Create parameters from a NumPy array.
        
        Args:
            array: Array representation of parameters [mu_1, mu_2, ..., sigma_11, sigma_12, ...]
            dim: Dimension of the multivariate normal distribution
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            MultivariateNormalParams: Parameter object
            
        Raises:
            ValueError: If the array length doesn't match the expected length
        """
        expected_length = dim + dim * (dim + 1) // 2
        if len(array) != expected_length:
            raise ValueError(
                f"Array length ({len(array)}) doesn't match expected length ({expected_length}) for dimension {dim}"
            )
        
        # Extract mean vector
        mu = array[:dim]
        
        # Extract covariance matrix (lower triangular part)
        sigma = np.zeros((dim, dim))
        idx = dim
        for i in range(dim):
            for j in range(i + 1):
                sigma[i, j] = array[idx]
                if i != j:
                    sigma[j, i] = array[idx]  # Fill upper triangular part (symmetry)
                idx += 1
        
        return cls(mu=mu, sigma=sigma)
    
    def transform(self) -> np.ndarray:
        """Transform parameters to unconstrained space for optimization.
        
        This transformation ensures that the covariance matrix remains positive definite.
        
        Returns:
            np.ndarray: Transformed parameters in unconstrained space
        """
        # mu is already unconstrained
        mu_transformed = self.mu.copy()
        
        # Transform sigma to unconstrained space using Cholesky decomposition
        try:
            L = np.linalg.cholesky(self.sigma)
            
            # Extract lower triangular elements
            n = L.shape[0]
            sigma_transformed = np.zeros(n * (n + 1) // 2)
            idx = 0
            for i in range(n):
                for j in range(i + 1):
                    if i == j:
                        # Log transform diagonal elements to ensure positivity
                        sigma_transformed[idx] = np.log(L[i, j])
                    else:
                        # Off-diagonal elements are unconstrained
                        sigma_transformed[idx] = L[i, j]
                    idx += 1
        except np.linalg.LinAlgError:
            # If Cholesky decomposition fails, use a different approach
            # This is a fallback for numerical stability
            warn_numeric(
                "Cholesky decomposition failed in parameter transformation. Using eigenvalue decomposition instead.",
                operation="transform",
                issue="Cholesky decomposition failure"
            )
            
            # Use eigenvalue decomposition
            eigvals, eigvecs = np.linalg.eigh(self.sigma)
            
            # Ensure eigenvalues are positive
            eigvals = np.maximum(eigvals, 1e-8)
            
            # Reconstruct sigma with positive eigenvalues
            sigma_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Try Cholesky again with the positive definite matrix
            L = np.linalg.cholesky(sigma_psd)
            
            # Extract lower triangular elements
            n = L.shape[0]
            sigma_transformed = np.zeros(n * (n + 1) // 2)
            idx = 0
            for i in range(n):
                for j in range(i + 1):
                    if i == j:
                        # Log transform diagonal elements to ensure positivity
                        sigma_transformed[idx] = np.log(L[i, j])
                    else:
                        # Off-diagonal elements are unconstrained
                        sigma_transformed[idx] = L[i, j]
                    idx += 1
        
        # Concatenate transformed parameters
        return np.concatenate([mu_transformed, sigma_transformed])
    
    @classmethod
    def inverse_transform(cls, array: np.ndarray, dim: int, **kwargs: Any) -> 'MultivariateNormalParams':
        """Transform parameters from unconstrained space back to constrained space.
        
        Args:
            array: Parameters in unconstrained space
            dim: Dimension of the multivariate normal distribution
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            MultivariateNormalParams: Parameter object with constrained parameters
            
        Raises:
            ValueError: If the array length doesn't match the expected length
        """
        expected_length = dim + dim * (dim + 1) // 2
        if len(array) != expected_length:
            raise ValueError(
                f"Array length ({len(array)}) doesn't match expected length ({expected_length}) for dimension {dim}"
            )
        
        # Extract mean vector (already unconstrained)
        mu = array[:dim]
        
        # Extract transformed Cholesky factors
        L = np.zeros((dim, dim))
        idx = dim
        for i in range(dim):
            for j in range(i + 1):
                if i == j:
                    # Exponentiate diagonal elements to ensure positivity
                    L[i, j] = np.exp(array[idx])
                else:
                    # Off-diagonal elements are unconstrained
                    L[i, j] = array[idx]
                idx += 1
        
        # Reconstruct covariance matrix from Cholesky factors
        sigma = L @ L.T
        
        return cls(mu=mu, sigma=sigma)


# Initialize Numba JIT-compiled functions for normal distribution

@jit(nopython=True, cache=True)
def _normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Numba-accelerated PDF for normal distribution.
    
    Args:
        x: Values to compute the PDF for
        mu: Mean parameter
        sigma: Standard deviation parameter
    
    Returns:
        np.ndarray: PDF values
    """
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))


@jit(nopython=True, cache=True)
def _normal_loglikelihood(x: np.ndarray, mu: float, sigma: float) -> float:
    """Numba-accelerated log-likelihood for normal distribution.
    
    Args:
        x: Data to compute the log-likelihood for
        mu: Mean parameter
        sigma: Standard deviation parameter
    
    Returns:
        float: Log-likelihood value
    """
    n = len(x)
    z = (x - mu) / sigma
    return -0.5 * n * np.log(2.0 * np.pi) - n * np.log(sigma) - 0.5 * np.sum(z * z)


@jit(nopython=True, cache=True)
def _mvnormal_pdf(x: np.ndarray, mu: np.ndarray, sigma_inv: np.ndarray, log_det_sigma: float) -> np.ndarray:
    """Numba-accelerated PDF for multivariate normal distribution.
    
    Args:
        x: Values to compute the PDF for (n_samples, n_dim)
        mu: Mean vector (n_dim,)
        sigma_inv: Inverse of covariance matrix (n_dim, n_dim)
        log_det_sigma: Log determinant of covariance matrix
    
    Returns:
        np.ndarray: PDF values (n_samples,)
    """
    n_dim = len(mu)
    n_samples = x.shape[0]
    pdf_values = np.zeros(n_samples)
    
    # Compute PDF for each sample
    for i in range(n_samples):
        # Compute (x - mu)
        diff = x[i] - mu
        
        # Compute (x - mu)' * sigma_inv * (x - mu)
        mahalanobis = 0.0
        for j in range(n_dim):
            temp = 0.0
            for k in range(n_dim):
                temp += diff[k] * sigma_inv[k, j]
            mahalanobis += temp * diff[j]
        
        # Compute PDF
        pdf_values[i] = np.exp(-0.5 * mahalanobis) / np.sqrt((2.0 * np.pi) ** n_dim * np.exp(log_det_sigma))
    
    return pdf_values


@jit(nopython=True, cache=True)
def _mvnormal_loglikelihood(x: np.ndarray, mu: np.ndarray, sigma_inv: np.ndarray, log_det_sigma: float) -> float:
    """Numba-accelerated log-likelihood for multivariate normal distribution.
    
    Args:
        x: Data to compute the log-likelihood for (n_samples, n_dim)
        mu: Mean vector (n_dim,)
        sigma_inv: Inverse of covariance matrix (n_dim, n_dim)
        log_det_sigma: Log determinant of covariance matrix
    
    Returns:
        float: Log-likelihood value
    """
    n_dim = len(mu)
    n_samples = x.shape[0]
    
    # Constant term
    const_term = -0.5 * n_samples * (n_dim * np.log(2.0 * np.pi) + log_det_sigma)
    
    # Sum of Mahalanobis distances
    sum_mahalanobis = 0.0
    for i in range(n_samples):
        # Compute (x - mu)
        diff = x[i] - mu
        
        # Compute (x - mu)' * sigma_inv * (x - mu)
        mahalanobis = 0.0
        for j in range(n_dim):
            temp = 0.0
            for k in range(n_dim):
                temp += diff[k] * sigma_inv[k, j]
            mahalanobis += temp * diff[j]
        
        sum_mahalanobis += mahalanobis
    
    return const_term - 0.5 * sum_mahalanobis


class Normal(NumbaDistribution[NormalParams]):
    """Normal distribution implementation with Numba acceleration.
    
    This class implements the normal (Gaussian) distribution with parameters
    mu (mean) and sigma (standard deviation). It provides Numba-accelerated
    implementations of the PDF, CDF, PPF, and log-likelihood functions.
    
    Attributes:
        name: A descriptive name for the distribution
        params: Distribution parameters (mu, sigma)
    """
    
    # Class variables for JIT-compiled functions
    _jit_pdf = staticmethod(_normal_pdf)
    _jit_loglikelihood = staticmethod(_normal_loglikelihood)
    
    def __init__(self, 
                name: str = "Normal", 
                params: Optional[NormalParams] = None):
        """Initialize the normal distribution.
        
        Args:
            name: A descriptive name for the distribution
            params: Distribution parameters (mu, sigma)
        """
        if params is None:
            params = NormalParams()
        
        super().__init__(name=name, params=params)
    
    def _params_to_tuple(self) -> Tuple[float, float]:
        """Convert parameters to a tuple for JIT functions.
        
        Returns:
            Tuple[float, float]: Parameter tuple (mu, sigma)
            
        Raises:
            DistributionError: If parameters are not set
        """
        if self._params is None:
            raise DistributionError(
                "Parameters are not set",
                distribution_type=self.name
            )
        
        return (self._params.mu, self._params.sigma)
    
    def _vector_to_params(self, vector: np.ndarray) -> NormalParams:
        """Convert a parameter vector to a parameter object.
        
        Args:
            vector: Parameter vector [mu, log(sigma)]
        
        Returns:
            NormalParams: Parameter object
            
        Raises:
            ValueError: If the vector has incorrect length
        """
        return NormalParams.inverse_transform(vector)
    
    def _get_initial_params(self, data: np.ndarray) -> np.ndarray:
        """Get initial parameter values for optimization.
        
        Args:
            data: Data to estimate initial parameters from
        
        Returns:
            np.ndarray: Initial parameter vector [mu, log(sigma)]
        """
        # Estimate mu and sigma from data
        mu_est = np.mean(data)
        sigma_est = np.std(data, ddof=1)
        
        # Ensure sigma is positive
        if sigma_est <= 0:
            sigma_est = 0.1
        
        # Transform to unconstrained space
        return np.array([mu_est, np.log(sigma_est)])
    
    def cdf(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute the cumulative distribution function.
        
        Args:
            x: Values to compute the CDF for
            **kwargs: Additional keyword arguments for the CDF
        
        Returns:
            np.ndarray: CDF values
            
        Raises:
            DistributionError: If the parameters are not set
            ValueError: If x contains invalid values
        """
        if self._params is None:
            raise DistributionError(
                "Parameters are not set",
                distribution_type=self.name
            )
        
        # Convert input to numpy array if needed
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        
        # Check for invalid values
        if np.isnan(x).any() or np.isinf(x).any():
            raise ValueError("Input contains NaN or infinite values")
        
        # Use scipy.stats.norm for CDF computation
        z = (x - self._params.mu) / self._params.sigma
        return stats.norm.cdf(z)
    
    def ppf(self, q: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute the percent point function (inverse of CDF).
        
        Args:
            q: Probabilities to compute the PPF for
            **kwargs: Additional keyword arguments for the PPF
        
        Returns:
            np.ndarray: PPF values
            
        Raises:
            DistributionError: If the parameters are not set
            ValueError: If q contains values outside [0, 1]
        """
        if self._params is None:
            raise DistributionError(
                "Parameters are not set",
                distribution_type=self.name
            )
        
        # Convert input to numpy array if needed
        if not isinstance(q, np.ndarray):
            q = np.asarray(q)
        
        # Check for invalid values
        if np.isnan(q).any() or np.isinf(q).any():
            raise ValueError("Input contains NaN or infinite values")
        
        if np.any((q < 0) | (q > 1)):
            raise ValueError("Probabilities must be between 0 and 1")
        
        # Use scipy.stats.norm for PPF computation
        return self._params.mu + self._params.sigma * stats.norm.ppf(q)
    
    def rvs(self, 
           size: Union[int, Tuple[int, ...]], 
           random_state: Optional[Union[int, np.random.Generator]] = None,
           **kwargs: Any) -> np.ndarray:
        """Generate random variates from the distribution.
        
        Args:
            size: Number of random variates to generate
            random_state: Random number generator or seed
            **kwargs: Additional keyword arguments for random variate generation
        
        Returns:
            np.ndarray: Random variates
            
        Raises:
            DistributionError: If the parameters are not set
            ValueError: If size is invalid
        """
        if self._params is None:
            raise DistributionError(
                "Parameters are not set",
                distribution_type=self.name
            )
        
        # Handle random state
        if random_state is None:
            rng = np.random.default_rng()
        elif isinstance(random_state, int):
            rng = np.random.default_rng(random_state)
        else:
            rng = random_state
        
        # Generate standard normal random variates
        z = rng.standard_normal(size=size)
        
        # Transform to desired mean and standard deviation
        return self._params.mu + self._params.sigma * z


class MultivariateNormal(NumbaDistribution[MultivariateNormalParams]):
    """Multivariate normal distribution implementation with Numba acceleration.
    
    This class implements the multivariate normal distribution with parameters
    mu (mean vector) and sigma (covariance matrix). It provides Numba-accelerated
    implementations of the PDF and log-likelihood functions.
    
    Attributes:
        name: A descriptive name for the distribution
        params: Distribution parameters (mu, sigma)
    """
    
    def __init__(self, 
                name: str = "MultivariateNormal", 
                params: Optional[MultivariateNormalParams] = None):
        """Initialize the multivariate normal distribution.
        
        Args:
            name: A descriptive name for the distribution
            params: Distribution parameters (mu, sigma)
        """
        super().__init__(name=name, params=params)
        
        # Cache for expensive computations
        self._sigma_inv: Optional[np.ndarray] = None
        self._log_det_sigma: Optional[float] = None
        self._cholesky: Optional[np.ndarray] = None
        
        # Update cache if parameters are provided
        if params is not None:
            self._update_cache()
    
    def _update_cache(self) -> None:
        """Update cached computations based on current parameters.
        
        This method computes and caches the inverse covariance matrix,
        log determinant, and Cholesky decomposition of the covariance matrix.
        
        Raises:
            DistributionError: If parameters are not set
            NumericError: If matrix inversion or Cholesky decomposition fails
        """
        if self._params is None:
            raise DistributionError(
                "Parameters are not set",
                distribution_type=self.name
            )
        
        try:
            # Compute Cholesky decomposition
            self._cholesky = np.linalg.cholesky(self._params.sigma)
            
            # Compute log determinant using Cholesky decomposition
            # log(det(Sigma)) = 2 * sum(log(diag(L)))
            self._log_det_sigma = 2.0 * np.sum(np.log(np.diag(self._cholesky)))
            
            # Compute inverse covariance matrix
            self._sigma_inv = np.linalg.inv(self._params.sigma)
        except np.linalg.LinAlgError as e:
            # Handle numerical issues
            raise NumericError(
                "Failed to compute matrix operations for multivariate normal distribution",
                operation="matrix_operations",
                values=self._params.sigma,
                error_type="linear algebra error",
                details=str(e)
            )
    
    def _params_to_tuple(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Convert parameters to a tuple for JIT functions.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, float]: Parameter tuple (mu, sigma_inv, log_det_sigma)
            
        Raises:
            DistributionError: If parameters are not set
        """
        if self._params is None:
            raise DistributionError(
                "Parameters are not set",
                distribution_type=self.name
            )
        
        # Ensure cache is updated
        if self._sigma_inv is None or self._log_det_sigma is None:
            self._update_cache()
        
        return (self._params.mu, self._sigma_inv, self._log_det_sigma)
    
    def _vector_to_params(self, vector: np.ndarray) -> MultivariateNormalParams:
        """Convert a parameter vector to a parameter object.
        
        Args:
            vector: Parameter vector
        
        Returns:
            MultivariateNormalParams: Parameter object
            
        Raises:
            ValueError: If the vector has incorrect length
        """
        # Determine dimension from vector length
        # vector length = dim + dim*(dim+1)/2
        # Solve quadratic equation: dim^2 + dim - 2*vector_length = 0
        # dim = (-1 + sqrt(1 + 8*vector_length))/2
        vector_length = len(vector)
        dim = int((-1 + np.sqrt(1 + 8 * vector_length)) / 2)
        
        # Verify dimension is correct
        expected_length = dim + dim * (dim + 1) // 2
        if expected_length != vector_length:
            raise ValueError(
                f"Vector length ({vector_length}) doesn't match expected length ({expected_length}) for dimension {dim}"
            )
        
        return MultivariateNormalParams.inverse_transform(vector, dim=dim)
    
    def _get_initial_params(self, data: np.ndarray) -> np.ndarray:
        """Get initial parameter values for optimization.
        
        Args:
            data: Data to estimate initial parameters from
        
        Returns:
            np.ndarray: Initial parameter vector
        """
        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Estimate mu and sigma from data
        mu_est = np.mean(data, axis=0)
        sigma_est = np.cov(data, rowvar=False)
        
        # Ensure sigma is positive definite
        n = sigma_est.shape[0]
        min_eig = np.min(np.linalg.eigvalsh(sigma_est))
        if min_eig <= 0:
            # Add small positive value to diagonal
            sigma_est += np.eye(n) * (abs(min_eig) + 0.01)
        
        # Create parameters
        params = MultivariateNormalParams(mu=mu_est, sigma=sigma_est)
        
        # Transform to unconstrained space
        return params.transform()
    
    def pdf(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute the probability density function.
        
        Args:
            x: Values to compute the PDF for
            **kwargs: Additional keyword arguments for the PDF
        
        Returns:
            np.ndarray: PDF values
            
        Raises:
            DistributionError: If the parameters are not set
            ValueError: If x contains invalid values
        """
        if self._params is None:
            raise DistributionError(
                "Parameters are not set",
                distribution_type=self.name
            )
        
        # Ensure cache is updated
        if self._sigma_inv is None or self._log_det_sigma is None:
            self._update_cache()
        
        # Convert input to numpy array if needed
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        
        # Check for invalid values
        if np.isnan(x).any() or np.isinf(x).any():
            raise ValueError("Input contains NaN or infinite values")
        
        # Ensure x is 2D
        if x.ndim == 1:
            # Single observation
            if x.shape[0] != self._params.mu.shape[0]:
                raise ValueError(
                    f"Input dimension ({x.shape[0]}) doesn't match distribution dimension ({self._params.mu.shape[0]})"
                )
            x = x.reshape(1, -1)
        else:
            # Multiple observations
            if x.shape[1] != self._params.mu.shape[0]:
                raise ValueError(
                    f"Input dimension ({x.shape[1]}) doesn't match distribution dimension ({self._params.mu.shape[0]})"
                )
        
        # Compute PDF using Numba-accelerated function
        return _mvnormal_pdf(x, self._params.mu, self._sigma_inv, self._log_det_sigma)
    
    def cdf(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute the cumulative distribution function.
        
        Note: The multivariate normal CDF does not have a closed-form expression.
        This implementation uses SciPy's multivariate normal CDF approximation.
        
        Args:
            x: Values to compute the CDF for
            **kwargs: Additional keyword arguments for the CDF
        
        Returns:
            np.ndarray: CDF values
            
        Raises:
            DistributionError: If the parameters are not set
            ValueError: If x contains invalid values
        """
        if self._params is None:
            raise DistributionError(
                "Parameters are not set",
                distribution_type=self.name
            )
        
        # Convert input to numpy array if needed
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        
        # Check for invalid values
        if np.isnan(x).any() or np.isinf(x).any():
            raise ValueError("Input contains NaN or infinite values")
        
        # Ensure x is 2D
        if x.ndim == 1:
            # Single observation
            if x.shape[0] != self._params.mu.shape[0]:
                raise ValueError(
                    f"Input dimension ({x.shape[0]}) doesn't match distribution dimension ({self._params.mu.shape[0]})"
                )
            x = x.reshape(1, -1)
        else:
            # Multiple observations
            if x.shape[1] != self._params.mu.shape[0]:
                raise ValueError(
                    f"Input dimension ({x.shape[1]}) doesn't match distribution dimension ({self._params.mu.shape[0]})"
                )
        
        # Use SciPy's multivariate normal CDF
        n_samples = x.shape[0]
        cdf_values = np.zeros(n_samples)
        
        for i in range(n_samples):
            cdf_values[i] = stats.multivariate_normal.cdf(
                x[i], mean=self._params.mu, cov=self._params.sigma
            )
        
        return cdf_values
    
    def ppf(self, q: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute the percent point function (inverse of CDF).
        
        Note: The multivariate normal PPF does not have a closed-form expression.
        This method is not implemented for the multivariate case.
        
        Args:
            q: Probabilities to compute the PPF for
            **kwargs: Additional keyword arguments for the PPF
        
        Raises:
            NotImplementedError: This method is not implemented for multivariate normal
        """
        raise NotImplementedError(
            "Percent point function (PPF) is not implemented for multivariate normal distribution"
        )
    
    def rvs(self, 
           size: Union[int, Tuple[int, ...]], 
           random_state: Optional[Union[int, np.random.Generator]] = None,
           **kwargs: Any) -> np.ndarray:
        """Generate random variates from the distribution.
        
        Args:
            size: Number of random variates to generate
            random_state: Random number generator or seed
            **kwargs: Additional keyword arguments for random variate generation
        
        Returns:
            np.ndarray: Random variates
            
        Raises:
            DistributionError: If the parameters are not set
            ValueError: If size is invalid
        """
        if self._params is None:
            raise DistributionError(
                "Parameters are not set",
                distribution_type=self.name
            )
        
        # Ensure Cholesky decomposition is available
        if self._cholesky is None:
            self._update_cache()
        
        # Handle random state
        if random_state is None:
            rng = np.random.default_rng()
        elif isinstance(random_state, int):
            rng = np.random.default_rng(random_state)
        else:
            rng = random_state
        
        # Get dimension
        dim = self._params.mu.shape[0]
        
        # Handle different size specifications
        if isinstance(size, int):
            n_samples = size
            output_shape = (size, dim)
        else:
            n_samples = np.prod(size)
            output_shape = size + (dim,)
        
        # Generate standard normal random variates
        z = rng.standard_normal(size=(n_samples, dim))
        
        # Transform to desired mean and covariance
        # x = mu + L*z where L is the Cholesky factor of sigma
        x = self._params.mu + np.dot(z, self._cholesky.T)
        
        # Reshape to desired output shape
        return x.reshape(output_shape)
    
    def loglikelihood(self, x: np.ndarray, **kwargs: Any) -> float:
        """Compute the log-likelihood of the data under the distribution.
        
        Args:
            x: Data to compute the log-likelihood for
            **kwargs: Additional keyword arguments for the log-likelihood
        
        Returns:
            float: Log-likelihood value
            
        Raises:
            DistributionError: If the parameters are not set
            ValueError: If x contains invalid values
        """
        if self._params is None:
            raise DistributionError(
                "Parameters are not set",
                distribution_type=self.name
            )
        
        # Ensure cache is updated
        if self._sigma_inv is None or self._log_det_sigma is None:
            self._update_cache()
        
        # Convert input to numpy array if needed
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        
        # Check for invalid values
        if np.isnan(x).any() or np.isinf(x).any():
            raise ValueError("Input contains NaN or infinite values")
        
        # Ensure x is 2D
        if x.ndim == 1:
            # Single observation
            if x.shape[0] != self._params.mu.shape[0]:
                raise ValueError(
                    f"Input dimension ({x.shape[0]}) doesn't match distribution dimension ({self._params.mu.shape[0]})"
                )
            x = x.reshape(1, -1)
        else:
            # Multiple observations
            if x.shape[1] != self._params.mu.shape[0]:
                raise ValueError(
                    f"Input dimension ({x.shape[1]}) doesn't match distribution dimension ({self._params.mu.shape[0]})"
                )
        
        # Compute log-likelihood using Numba-accelerated function
        return _mvnormal_loglikelihood(x, self._params.mu, self._sigma_inv, self._log_det_sigma)


# Aliases for backward compatibility
NormalDistribution = Normal
MultivariateNormalDistribution = MultivariateNormal


# Function to create a normal distribution from parameters
def create_normal(mu: float = 0.0, sigma: float = 1.0) -> Normal:
    """Create a normal distribution with the specified parameters.
    
    Args:
        mu: Mean parameter
        sigma: Standard deviation parameter
    
    Returns:
        Normal: Normal distribution object
    
    Raises:
        ParameterError: If parameter constraints are violated
    """
    params = NormalParams(mu=mu, sigma=sigma)
    return Normal(params=params)


# Function to create a multivariate normal distribution from parameters
def create_multivariate_normal(mu: np.ndarray, sigma: np.ndarray) -> MultivariateNormal:
    """Create a multivariate normal distribution with the specified parameters.
    
    Args:
        mu: Mean vector
        sigma: Covariance matrix
    
    Returns:
        MultivariateNormal: Multivariate normal distribution object
    
    Raises:
        ParameterError: If parameter constraints are violated
    """
    params = MultivariateNormalParams(mu=mu, sigma=sigma)
    return MultivariateNormal(params=params)


# Function to compute normal log-likelihood (for backward compatibility)
def normloglik(x: np.ndarray, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Compute the log-likelihood of data under a normal distribution.
    
    Args:
        x: Data to compute the log-likelihood for
        mu: Mean parameter
        sigma: Standard deviation parameter
    
    Returns:
        float: Log-likelihood value
    
    Raises:
        ValueError: If x contains invalid values or sigma is not positive
    """
    # Create normal distribution
    dist = create_normal(mu=mu, sigma=sigma)
    
    # Compute log-likelihood
    return dist.loglikelihood(x)


# Function to compute multivariate normal log-likelihood (for backward compatibility)
def mvnormloglik(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Compute the log-likelihood of data under a multivariate normal distribution.
    
    Args:
        x: Data to compute the log-likelihood for
        mu: Mean vector
        sigma: Covariance matrix
    
    Returns:
        float: Log-likelihood value
    
    Raises:
        ValueError: If x contains invalid values or sigma is not positive definite
    """
    # Create multivariate normal distribution
    dist = create_multivariate_normal(mu=mu, sigma=sigma)
    
    # Compute log-likelihood
    return dist.loglikelihood(x)


# mfe/models/distributions/normal.py
"""
Normal distribution implementations for the MFE Toolbox.

This module provides implementations of the normal (Gaussian) distribution,
including both univariate and multivariate variants. These distributions are
fundamental in financial econometrics, serving as the basis for many models
and statistical tests.

The module includes Numba-accelerated implementations of probability density
functions, cumulative distribution functions, quantile functions, random
number generation, and log-likelihood evaluation for both univariate and
multivariate normal distributions.

Key features:
- Univariate normal distribution with mean and standard deviation parameters
- Multivariate normal distribution with mean vector and covariance matrix
- Numba-accelerated core functions for performance-critical operations
- Integration with SciPy's statistical distributions for core functionality
- Comprehensive parameter validation with dataclasses
- Support for both scalar and vectorized operations
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import (
    Any, Dict, List, Optional, Tuple, Union, cast, overload
)

import numpy as np
from scipy import stats
from numba import jit

from mfe.core.base import DistributionBase
from mfe.core.parameters import (
    ParameterBase, ParameterError, validate_positive, validate_positive_definite,
    transform_positive, inverse_transform_positive
)
from mfe.core.exceptions import (
    DistributionError, NumericError, raise_parameter_error, warn_numeric
)
from mfe.core.types import (
    Vector, Matrix, DistributionType, DistributionLike, ParameterVector,
    PDFFunction, CDFFunction, PPFFunction, RVSFunction
)
from mfe.models.distributions.base import (
    BaseDistribution, ContinuousDistribution, NumbaDistribution
)


@dataclass
class NormalParams(ParameterBase):
    """Parameters for the univariate normal distribution.
    
    Attributes:
        mu: Mean parameter
        sigma: Standard deviation parameter (must be positive)
    """
    
    mu: float = 0.0
    sigma: float = 1.0
    
    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate normal distribution parameter constraints.
        
        Raises:
            ParameterError: If parameter constraints are violated
        """
        # Validate sigma (must be positive)
        validate_positive(self.sigma, "sigma")
    
    def to_array(self) -> np.ndarray:
        """Convert parameters to a NumPy array.
        
        Returns:
            np.ndarray: Array representation of parameters [mu, sigma]
        """
        return np.array([self.mu, self.sigma])
    
    @classmethod
    def from_array(cls, array: np.ndarray, **kwargs: Any) -> 'NormalParams':
        """Create parameters from a NumPy array.
        
        Args:
            array: Array representation of parameters [mu, sigma]
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            NormalParams: Parameter object
            
        Raises:
            ValueError: If the array length is not 2
        """
        if len(array) != 2:
            raise ValueError(f"Array length must be 2, got {len(array)}")
        
        return cls(mu=array[0], sigma=array[1])
    
    def transform(self) -> np.ndarray:
        """Transform parameters to unconstrained space for optimization.
        
        Returns:
            np.ndarray: Transformed parameters in unconstrained space [mu, log(sigma)]
        """
        # mu is already unconstrained
        # Transform sigma to unconstrained space (log)
        transformed_sigma = transform_positive(self.sigma)
        
        return np.array([self.mu, transformed_sigma])
    
    @classmethod
    def inverse_transform(cls, array: np.ndarray, **kwargs: Any) -> 'NormalParams':
        """Transform parameters from unconstrained space back to constrained space.
        
        Args:
            array: Parameters in unconstrained space [mu, log(sigma)]
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            NormalParams: Parameter object with constrained parameters
            
        Raises:
            ValueError: If the array length is not 2
        """
        if len(array) != 2:
            raise ValueError(f"Array length must be 2, got {len(array)}")
        
        # Extract transformed parameters
        mu = array[0]  # mu is already unconstrained
        transformed_sigma = array[1]
        
        # Inverse transform sigma
        sigma = inverse_transform_positive(transformed_sigma)
        
        return cls(mu=mu, sigma=sigma)


@dataclass
class MultivariateNormalParams(ParameterBase):
    """Parameters for the multivariate normal distribution.
    
    Attributes:
        mu: Mean vector
        sigma: Covariance matrix (must be positive definite)
    """
    
    mu: np.ndarray
    sigma: np.ndarray
    
    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        # Ensure mu and sigma are NumPy arrays
        if not isinstance(self.mu, np.ndarray):
            self.mu = np.array(self.mu, dtype=np.float64)
        if not isinstance(self.sigma, np.ndarray):
            self.sigma = np.array(self.sigma, dtype=np.float64)
        
        self.validate()
    
    def validate(self) -> None:
        """Validate multivariate normal distribution parameter constraints.
        
        Raises:
            ParameterError: If parameter constraints are violated
        """
        # Validate mu (must be a vector)
        if self.mu.ndim != 1:
            raise ParameterError(
                f"Mean vector must be 1-dimensional, got {self.mu.ndim} dimensions",
                param_name="mu",
                param_value=self.mu.shape
            )
        
        # Validate sigma (must be a square matrix)
        if self.sigma.ndim != 2:
            raise ParameterError(
                f"Covariance matrix must be 2-dimensional, got {self.sigma.ndim} dimensions",
                param_name="sigma",
                param_value=self.sigma.shape
            )
        
        if self.sigma.shape[0] != self.sigma.shape[1]:
            raise ParameterError(
                f"Covariance matrix must be square, got shape {self.sigma.shape}",
                param_name="sigma",
                param_value=self.sigma.shape
            )
        
        # Validate dimensions match
        if self.mu.shape[0] != self.sigma.shape[0]:
            raise ParameterError(
                f"Mean vector length ({self.mu.shape[0]}) must match covariance matrix dimension ({self.sigma.shape[0]})",
                param_name="mu, sigma",
                param_value=(self.mu.shape, self.sigma.shape)
            )
        
        # Validate sigma is positive definite
        try:
            validate_positive_definite(self.sigma, "sigma")
        except ParameterError as e:
            # Add more context to the error
            raise ParameterError(
                "Covariance matrix must be positive definite",
                param_name="sigma",
                param_value=self.sigma,
                details=str(e)
            )
    
    def to_array(self) -> np.ndarray:
        """Convert parameters to a NumPy array.
        
        This flattens the parameters into a 1D array for optimization.
        
        Returns:
            np.ndarray: Array representation of parameters [mu_1, mu_2, ..., sigma_11, sigma_12, ...]
        """
        # Flatten the mean vector
        mu_flat = self.mu.flatten()
        
        # Flatten the covariance matrix (only lower triangular part due to symmetry)
        n = self.sigma.shape[0]
        sigma_flat = np.zeros(n * (n + 1) // 2)
        idx = 0
        for i in range(n):
            for j in range(i + 1):
                sigma_flat[idx] = self.sigma[i, j]
                idx += 1
        
        # Concatenate flattened parameters
        return np.concatenate([mu_flat, sigma_flat])
    
    @classmethod
    def from_array(cls, array: np.ndarray, dim: int, **kwargs: Any) -> 'MultivariateNormalParams':
        """Create parameters from a NumPy array.
        
        Args:
            array: Array representation of parameters [mu_1, mu_2, ..., sigma_11, sigma_12, ...]
            dim: Dimension of the multivariate normal distribution
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            MultivariateNormalParams: Parameter object
            
        Raises:
            ValueError: If the array length doesn't match the expected length
        """
        expected_length = dim + dim * (dim + 1) // 2
        if len(array) != expected_length:
            raise ValueError(
                f"Array length ({len(array)}) doesn't match expected length ({expected_length}) for dimension {dim}"
            )
        
        # Extract mean vector
        mu = array[:dim]
        
        # Extract covariance matrix (lower triangular part)
        sigma = np.zeros((dim, dim))
        idx = dim
        for i in range(dim):
            for j in range(i + 1):
                sigma[i, j] = array[idx]
                if i != j:
                    sigma[j, i] = array[idx]  # Fill upper triangular part (symmetry)
                idx += 1
        
        return cls(mu=mu, sigma=sigma)
    
    def transform(self) -> np.ndarray:
        """Transform parameters to unconstrained space for optimization.
        
        This transformation ensures that the covariance matrix remains positive definite.
        
        Returns:
            np.ndarray: Transformed parameters in unconstrained space
        """
        # mu is already unconstrained
        mu_transformed = self.mu.copy()
        
        # Transform sigma to unconstrained space using Cholesky decomposition
        try:
            L = np.linalg.cholesky(self.sigma)
            
            # Extract lower triangular elements
            n = L.shape[0]
            sigma_transformed = np.zeros(n * (n + 1) // 2)
            idx = 0
            for i in range(n):
                for j in range(i + 1):
                    if i == j:
                        # Log transform diagonal elements to ensure positivity
                        sigma_transformed[idx] = np.log(L[i, j])
                    else:
                        # Off-diagonal elements are unconstrained
                        sigma_transformed[idx] = L[i, j]
                    idx += 1
        except np.linalg.LinAlgError:
            # If Cholesky decomposition fails, use a different approach
            # This is a fallback for numerical stability
            warn_numeric(
                "Cholesky decomposition failed in parameter transformation. Using eigenvalue decomposition instead.",
                operation="transform",
                issue="Cholesky decomposition failure"
            )
            
            # Use eigenvalue decomposition
            eigvals, eigvecs = np.linalg.eigh(self.sigma)
            
            # Ensure eigenvalues are positive
            eigvals = np.maximum(eigvals, 1e-8)
            
            # Reconstruct sigma with positive eigenvalues
            sigma_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Try Cholesky again with the positive definite matrix
            L = np.linalg.cholesky(sigma_psd)
            
            # Extract lower triangular elements
            n = L.shape[0]
            sigma_transformed = np.zeros(n * (n + 1) // 2)
            idx = 0
            for i in range(n):
                for j in range(i + 1):
                    if i == j:
                        # Log transform diagonal elements to ensure positivity
                        sigma_transformed[idx] = np.log(L[i, j])
                    else:
                        # Off-diagonal elements are unconstrained
                        sigma_transformed[idx] = L[i, j]
                    idx += 1
        
        # Concatenate transformed parameters
        return np.concatenate([mu_transformed, sigma_transformed])
    
    @classmethod
    def inverse_transform(cls, array: np.ndarray, dim: int, **kwargs: Any) -> 'MultivariateNormalParams':
        """Transform parameters from unconstrained space back to constrained space.
        
        Args:
            array: Parameters in unconstrained space
            dim: Dimension of the multivariate normal distribution
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            MultivariateNormalParams: Parameter object with constrained parameters
            
        Raises:
            ValueError: If the array length doesn't match the expected length
        """
        expected_length = dim + dim * (dim + 1) // 2
        if len(array) != expected_length:
            raise ValueError(
                f"Array length ({len(array)}) doesn't match expected length ({expected_length}) for dimension {dim}"
            )
        
        # Extract mean vector (already unconstrained)
        mu = array[:dim]
        
        # Extract transformed Cholesky factors
        L = np.zeros((dim, dim))
        idx = dim
        for i in range(dim):
            for j in range(i + 1):
                if i == j:
                    # Exponentiate diagonal elements to ensure positivity
                    L[i, j] = np.exp(array[idx])
                else:
                    # Off-diagonal elements are unconstrained
                    L[i, j] = array[idx]
                idx += 1
        
        # Reconstruct covariance matrix from Cholesky factors
        sigma = L @ L.T
        
        return cls(mu=mu, sigma=sigma)


# Initialize Numba JIT-compiled functions for normal distribution

@jit(nopython=True, cache=True)
def _normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Numba-accelerated PDF for normal distribution.
    
    Args:
        x: Values to compute the PDF for
        mu: Mean parameter
        sigma: Standard deviation parameter
    
    Returns:
        np.ndarray: PDF values
    """
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))


@jit(nopython=True, cache=True)
def _normal_loglikelihood(x: np.ndarray, mu: float, sigma: float) -> float:
    """Numba-accelerated log-likelihood for normal distribution.
    
    Args:
        x: Data to compute the log-likelihood for
        mu: Mean parameter
        sigma: Standard deviation parameter
    
    Returns:
        float: Log-likelihood value
    """
    n = len(x)
    z = (x - mu) / sigma
    return -0.5 * n * np.log(2.0 * np.pi) - n * np.log(sigma) - 0.5 * np.sum(z * z)


@jit(nopython=True, cache=True)
def _mvnormal_pdf(x: np.ndarray, mu: np.ndarray, sigma_inv: np.ndarray, log_det_sigma: float) -> np.ndarray:
    """Numba-accelerated PDF for multivariate normal distribution.
    
    Args:
        x: Values to compute the PDF for (n_samples, n_dim)
        mu: Mean vector (n_dim,)
        sigma_inv: Inverse of covariance matrix (n_dim, n_dim)
        log_det_sigma: Log determinant of covariance matrix
    
    Returns:
        np.ndarray: PDF values (n_samples,)
    """
    n_dim = len(mu)
    n_samples = x.shape[0]
    pdf_values = np.zeros(n_samples)
    
    # Compute PDF for each sample
    for i in range(n_samples):
        # Compute (x - mu)
        diff = x[i] - mu
        
        # Compute (x - mu)' * sigma_inv * (x - mu)
        mahalanobis = 0.0
        for j in range(n_dim):
            temp = 0.0
            for k in range(n_dim):
                temp += diff[k] * sigma_inv[k, j]
            mahalanobis += temp * diff[j]
        
        # Compute PDF
        pdf_values[i] = np.exp(-0.5 * mahalanobis) / np.sqrt((2.0 * np.pi) ** n_dim * np.exp(log_det_sigma))
    
    return pdf_values


@jit(nopython=True, cache=True)
def _mvnormal_loglikelihood(x: np.ndarray, mu: np.ndarray, sigma_inv: np.ndarray, log_det_sigma: float) -> float:
    """Numba-accelerated log-likelihood for multivariate normal distribution.
    
    Args:
        x: Data to compute the log-likelihood for (n_samples, n_dim)
        mu: Mean vector (n_dim,)
        sigma_inv: Inverse of covariance matrix (n_dim, n_dim)
        log_det_sigma: Log determinant of covariance matrix
    
    Returns:
        float: Log-likelihood value
    """
    n_dim = len(mu)
    n_samples = x.shape[0]
    
    # Constant term
    const_term = -0.5 * n_samples * (n_dim * np.log(2.0 * np.pi) + log_det_sigma)
    
    # Sum of Mahalanobis distances
    sum_mahalanobis = 0.0
    for i in range(n_samples):
        # Compute (x - mu)
        diff = x[i] - mu
        
        # Compute (x - mu)' * sigma_inv * (x - mu)
        mahalanobis = 0.0
        for j in range(n_dim):
            temp = 0.0
            for k in range(n_dim):
                temp += diff[k] * sigma_inv[k, j]
            mahalanobis += temp * diff[j]
        
        sum_mahalanobis += mahalanobis
    
    return const_term - 0.5 * sum_mahalanobis


class Normal(NumbaDistribution[NormalParams]):
    """Normal distribution implementation with Numba acceleration.
    
    This class implements the normal (Gaussian) distribution with parameters
    mu (mean) and sigma (standard deviation). It provides Numba-accelerated
    implementations of the PDF, CDF, PPF, and log-likelihood functions.
    
    Attributes:
        name: A descriptive name for the distribution
        params: Distribution parameters (mu, sigma)
    """
    
    # Class variables for JIT-compiled functions
    _jit_pdf = staticmethod(_normal_pdf)
    _jit_loglikelihood = staticmethod(_normal_loglikelihood)
    
    def __init__(self, 
                name: str = "Normal", 
                params: Optional[NormalParams] = None):
        """Initialize the normal distribution.
        
        Args:
            name: A descriptive name for the distribution
            params: Distribution parameters (mu, sigma)
        """
        if params is None:
            params = NormalParams()
        
        super().__init__(name=name, params=params)
    
    def _params_to_tuple(self) -> Tuple[float, float]:
        """Convert parameters to a tuple for JIT functions.
        
        Returns:
            Tuple[float, float]: Parameter tuple (mu, sigma)
            
        Raises:
            DistributionError: If parameters are not set
        """
        if self._params is None:
            raise DistributionError(
                "Parameters are not set",
                distribution_type=self.name
            )
        
        return (self._params.mu, self._params.sigma)
    
    def _vector_to_params(self, vector: np.ndarray) -> NormalParams:
        """Convert a parameter vector to a parameter object.
        
        Args:
            vector: Parameter vector [mu, log(sigma)]
        
        Returns:
            NormalParams: Parameter object
            
        Raises:
            ValueError: If the vector has incorrect length
        """
        return NormalParams.inverse_transform(vector)
    
    def _get_initial_params(self, data: np.ndarray) -> np.ndarray:
        """Get initial parameter values for optimization.
        
        Args:
            data: Data to estimate initial parameters from
        
        Returns:
            np.ndarray: Initial parameter vector [mu, log(sigma)]
        """
        # Estimate mu and sigma from data
        mu_est = np.mean(data)
        sigma_est = np.std(data, ddof=1)
        
        # Ensure sigma is positive
        if sigma_est <= 0:
            sigma_est = 0.1
        
        # Transform to unconstrained space
        return np.array([mu_est, np.log(sigma_est)])
    
    def cdf(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute the cumulative distribution function.
        
        Args:
            x: Values to compute the CDF for
            **kwargs: Additional keyword arguments for the CDF
        
        Returns:
            np.ndarray: CDF values
            
        Raises:
            DistributionError: If the parameters are not set
            ValueError: If x contains invalid values
        """
        if self._params is None:
            raise DistributionError(
                "Parameters are not set",
                distribution_type=self.name
            )
        
        # Convert input to numpy array if needed
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        
        # Check for invalid values
        if np.isnan(x).any() or np.isinf(x).any():
            raise ValueError("Input contains NaN or infinite values")
        
        # Use scipy.stats.norm for CDF computation
        z = (x - self._params.mu) / self._params.sigma
        return stats.norm.cdf(z)
    
    def ppf(self, q: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute the percent point function (inverse of CDF).
        
        Args:
            q: Probabilities to compute the PPF for
            **kwargs: Additional keyword arguments for the PPF
        
        Returns:
            np.ndarray: PPF values
            
        Raises:
            DistributionError: If the parameters are not set
            ValueError: If q contains values outside [0, 1]
        """
        if self._params is None:
            raise DistributionError(
                "Parameters are not set",
                distribution_type=self.name
            )
        
        # Convert input to numpy array if needed
        if not isinstance(q, np.ndarray):
            q = np.asarray(q)
        
        # Check for invalid values
        if np.isnan(q).any() or np.isinf(q).any():
            raise ValueError("Input contains NaN or infinite values")
        
        if np.any((q < 0) | (q > 1)):
            raise ValueError("Probabilities must be between 0 and 1")
        
        # Use scipy.stats.norm for PPF computation
        return self._params.mu + self._params.sigma * stats.norm.ppf(q)
    
    def rvs(self, 
           size: Union[int, Tuple[int, ...]], 
           random_state: Optional[Union[int, np.random.Generator]] = None,
           **kwargs: Any) -> np.ndarray:
        """Generate random variates from the distribution.
        
        Args:
            size: Number of random variates to generate
            random_state: Random number generator or seed
            **kwargs: Additional keyword arguments for random variate generation
        
        Returns:
            np.ndarray: Random variates
            
        Raises:
            DistributionError: If the parameters are not set
            ValueError: If size is invalid
        """
        if self._params is None:
            raise DistributionError(
                "Parameters are not set",
                distribution_type=self.name
            )
        
        # Handle random state
        if random_state is None:
            rng = np.random.default_rng()
        elif isinstance(random_state, int):
            rng = np.random.default_rng(random_state)
        else:
            rng = random_state
        
        # Generate standard normal random variates
        z = rng.standard_normal(size=size)
        
        # Transform to desired mean and standard deviation
        return self._params.mu + self._params.sigma * z


class MultivariateNormal(NumbaDistribution[MultivariateNormalParams]):
    """Multivariate normal distribution implementation with Numba acceleration.
    
    This class implements the multivariate normal distribution with parameters
    mu (mean vector) and sigma (covariance matrix). It provides Numba-accelerated
    implementations of the PDF and log-likelihood functions.
    
    Attributes:
        name: A descriptive name for the distribution
        params: Distribution parameters (mu, sigma)
    """
    
    def __init__(self, 
                name: str = "MultivariateNormal", 
                params: Optional[MultivariateNormalParams] = None):
        """Initialize the multivariate normal distribution.
        
        Args:
            name: A descriptive name for the distribution
            params: Distribution parameters (mu, sigma)
        """
        super().__init__(name=name, params=params)
        
        # Cache for expensive computations
        self._sigma_inv: Optional[np.ndarray] = None
        self._log_det_sigma: Optional[float] = None
        self._cholesky: Optional[np.ndarray] = None
        
        # Update cache if parameters are provided
        if params is not None:
            self._update_cache()
    
    def _update_cache(self) -> None:
        """Update cached computations based on current parameters.
        
        This method computes and caches the inverse covariance matrix,
        log determinant, and Cholesky decomposition of the covariance matrix.
        
        Raises:
            DistributionError: If parameters are not set
            NumericError: If matrix inversion or Cholesky decomposition fails
        """
        if self._params is None:
            raise DistributionError(
                "Parameters are not set",
                distribution_type=self.name
            )
        
        try:
            # Compute Cholesky decomposition
            self._cholesky = np.linalg.cholesky(self._params.sigma)
            
            # Compute log determinant using Cholesky decomposition
            # log(det(Sigma)) = 2 * sum(log(diag(L)))
            self._log_det_sigma = 2.0 * np.sum(np.log(np.diag(self._cholesky)))
            
            # Compute inverse covariance matrix
            self._sigma_inv = np.linalg.inv(self._params.sigma)
        except np.linalg.LinAlgError as e:
            # Handle numerical issues
            raise NumericError(
                "Failed to compute matrix operations for multivariate normal distribution",
                operation="matrix_operations",
                values=self._params.sigma,
                error_type="linear algebra error",
                details=str(e)
            )
    
    def _params_to_tuple(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Convert parameters to a tuple for JIT functions.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, float]: Parameter tuple (mu, sigma_inv, log_det_sigma)
            
        Raises:
            DistributionError: If parameters are not set
        """
        if self._params is None:
            raise DistributionError(
                "Parameters are not set",
                distribution_type=self.name
            )
        
        # Ensure cache is updated
        if self._sigma_inv is None or self._log_det_sigma is None:
            self._update_cache()
        
        return (self._params.mu, self._sigma_inv, self._log_det_sigma)
    
    def _vector_to_params(self, vector: np.ndarray) -> MultivariateNormalParams:
        """Convert a parameter vector to a parameter object.
        
        Args:
            vector: Parameter vector
        
        Returns:
            MultivariateNormalParams: Parameter object
            
        Raises:
            ValueError: If the vector has incorrect length
        """
        # Determine dimension from vector length
        # vector length = dim + dim*(dim+1)/2
        # Solve quadratic equation: dim^2 + dim - 2*vector_length = 0
        # dim = (-1 + sqrt(1 + 8*vector_length))/2
        vector_length = len(vector)
        dim = int((-1 + np.sqrt(1 + 8 * vector_length)) / 2)
        
        # Verify dimension is correct
        expected_length = dim + dim * (dim + 1) // 2
        if expected_length != vector_length:
            raise ValueError(
                f"Vector length ({vector_length}) doesn't match expected length ({expected_length}) for dimension {dim}"
            )
        
        return MultivariateNormalParams.inverse_transform(vector, dim=dim)
    
    def _get_initial_params(self, data: np.ndarray) -> np.ndarray:
        """Get initial parameter values for optimization.
        
        Args:
            data: Data to estimate initial parameters from
        
        Returns:
            np.ndarray: Initial parameter vector
        """
        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Estimate mu and sigma from data
        mu_est = np.mean(data, axis=0)
        sigma_est = np.cov(data, rowvar=False)
        
        # Ensure sigma is positive definite
        n = sigma_est.shape[0]
        min_eig = np.min(np.linalg.eigvalsh(sigma_est))
        if min_eig <= 0:
            # Add small positive value to diagonal
            sigma_est += np.eye(n) * (abs(min_eig) + 0.01)
        
        # Create parameters
        params = MultivariateNormalParams(mu=mu_est, sigma=sigma_est)
        
        # Transform to unconstrained space
        return params.transform()
    
    def pdf(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute the probability density function.
        
        Args:
            x: Values to compute the PDF for
            **kwargs: Additional keyword arguments for the PDF
        
        Returns:
            np.ndarray: PDF values
            
        Raises:
            DistributionError: If the parameters are not set
            ValueError: If x contains invalid values
        """
        if self._params is None:
            raise DistributionError(
                "Parameters are not set",
                distribution_type=self.name
            )
        
        # Ensure cache is updated
        if self._sigma_inv is None or self._log_det_sigma is None:
            self._update_cache()
        
        # Convert input to numpy array if needed
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        
        # Check for invalid values
        if np.isnan(x).any() or np.isinf(x).any():
            raise ValueError("Input contains NaN or infinite values")
        
        # Ensure x is 2D
        if x.ndim == 1:
            # Single observation
            if x.shape[0] != self._params.mu.shape[0]:
                raise ValueError(
                    f"Input dimension ({x.shape[0]}) doesn't match distribution dimension ({self._params.mu.shape[0]})"
                )
            x = x.reshape(1, -1)
        else:
            # Multiple observations
            if x.shape[1] != self._params.mu.shape[0]:
                raise ValueError(
                    f"Input dimension ({x.shape[1]}) doesn't match distribution dimension ({self._params.mu.shape[0]})"
                )
        
        # Compute PDF using Numba-accelerated function
        return _mvnormal_pdf(x, self._params.mu, self._sigma_inv, self._log_det_sigma)
    
    def cdf(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute the cumulative distribution function.
        
        Note: The multivariate normal CDF does not have a closed-form expression.
        This implementation uses SciPy's multivariate normal CDF approximation.
        
        Args:
            x: Values to compute the CDF for
            **kwargs: Additional keyword arguments for the CDF
        
        Returns:
            np.ndarray: CDF values
            
        Raises:
            DistributionError: If the parameters are not set
            ValueError: If x contains invalid values
        """
        if self._params is None:
            raise DistributionError(
                "Parameters are not set",
                distribution_type=self.name
            )
        
        # Convert input to numpy array if needed
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        
        # Check for invalid values
        if np.isnan(x).any() or np.isinf(x).any():
            raise ValueError("Input contains NaN or infinite values")
        
        # Ensure x is 2D
        if x.ndim == 1:
            # Single observation
            if x.shape[0] != self._params.mu.shape[0]:
                raise ValueError(
                    f"Input dimension ({x.shape[0]}) doesn't match distribution dimension ({self._params.mu.shape[0]})"
                )
            x = x.reshape(1, -1)
        else:
            # Multiple observations
            if x.shape[1] != self._params.mu.shape[0]:
                raise ValueError(
                    f"Input dimension ({x.shape[1]}) doesn't match distribution dimension ({self._params.mu.shape[0]})"
                )
        
        # Use SciPy's multivariate normal CDF
        n_samples = x.shape[0]
        cdf_values = np.zeros(n_samples)
        
        for i in range(n_samples):
            cdf_values[i] = stats.multivariate_normal.cdf(
                x[i], mean=self._params.mu, cov=self._params.sigma
            )
        
        return cdf_values
    
    def ppf(self, q: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute the percent point function (inverse of CDF).
        
        Note: The multivariate normal PPF does not have a closed-form expression.
        This method is not implemented for the multivariate case.
        
        Args:
            q: Probabilities to compute the PPF for
            **kwargs: Additional keyword arguments for the PPF
        
        Raises:
            NotImplementedError: This method is not implemented for multivariate normal
        """
        raise NotImplementedError(
            "Percent point function (PPF) is not implemented for multivariate normal distribution"
        )
    
    def rvs(self, 
           size: Union[int, Tuple[int, ...]], 
           random_state: Optional[Union[int, np.random.Generator]] = None,
           **kwargs: Any) -> np.ndarray:
        """Generate random variates from the distribution.
        
        Args:
            size: Number of random variates to generate
            random_state: Random number generator or seed
            **kwargs: Additional keyword arguments for random variate generation
        
        Returns:
            np.ndarray: Random variates
            
        Raises:
            DistributionError: If the parameters are not set
            ValueError: If size is invalid
        """
        if self._params is None:
            raise DistributionError(
                "Parameters are not set",
                distribution_type=self.name
            )
        
        # Ensure Cholesky decomposition is available
        if self._cholesky is None:
            self._update_cache()
        
        # Handle random state
        if random_state is None:
            rng = np.random.default_rng()
        elif isinstance(random_state, int):
            rng = np.random.default_rng(random_state)
        else:
            rng = random_state
        
        # Get dimension
        dim = self._params.mu.shape[0]
        
        # Handle different size specifications
        if isinstance(size, int):
            n_samples = size
            output_shape = (size, dim)
        else:
            n_samples = np.prod(size)
            output_shape = size + (dim,)
        
        # Generate standard normal random variates
        z = rng.standard_normal(size=(n_samples, dim))
        
        # Transform to desired mean and covariance
        # x = mu + L*z where L is the Cholesky factor of sigma
        x = self._params.mu + np.dot(z, self._cholesky.T)
        
        # Reshape to desired output shape
        return x.reshape(output_shape)
    
    def loglikelihood(self, x: np.ndarray, **kwargs: Any) -> float:
        """Compute the log-likelihood of the data under the distribution.
        
        Args:
            x: Data to compute the log-likelihood for
            **kwargs: Additional keyword arguments for the log-likelihood
        
        Returns:
            float: Log-likelihood value
            
        Raises:
            DistributionError: If the parameters are not set
            ValueError: If x contains invalid values
        """
        if self._params is None:
            raise DistributionError(
                "Parameters are not set",
                distribution_type=self.name
            )
        
        # Ensure cache is updated
        if self._sigma_inv is None or self._log_det_sigma is None:
            self._update_cache()
        
        # Convert input to numpy array if needed
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        
        # Check for invalid values
        if np.isnan(x).any() or np.isinf(x).any():
            raise ValueError("Input contains NaN or infinite values")
        
        # Ensure x is 2D
        if x.ndim == 1:
            # Single observation
            if x.shape[0] != self._params.mu.shape[0]:
                raise ValueError(
                    f"Input dimension ({x.shape[0]}) doesn't match distribution dimension ({self._params.mu.shape[0]})"
                )
            x = x.reshape(1, -1)
        else:
            # Multiple observations
            if x.shape[1] != self._params.mu.shape[0]:
                raise ValueError(
                    f"Input dimension ({x.shape[1]}) doesn't match distribution dimension ({self._params.mu.shape[0]})"
                )
        
        # Compute log-likelihood using Numba-accelerated function
        return _mvnormal_loglikelihood(x, self._params.mu, self._sigma_inv, self._log_det_sigma)


# Aliases for backward compatibility
NormalDistribution = Normal
MultivariateNormalDistribution = MultivariateNormal


# Function to create a normal distribution from parameters
def create_normal(mu: float = 0.0, sigma: float = 1.0) -> Normal:
    """Create a normal distribution with the specified parameters.
    
    Args:
        mu: Mean parameter
        sigma: Standard deviation parameter
    
    Returns:
        Normal: Normal distribution object
    
    Raises:
        ParameterError: If parameter constraints are violated
    """
    params = NormalParams(mu=mu, sigma=sigma)
    return Normal(params=params)


# Function to create a multivariate normal distribution from parameters
def create_multivariate_normal(mu: np.ndarray, sigma: np.ndarray) -> MultivariateNormal:
    """Create a multivariate normal distribution with the specified parameters.
    
    Args:
        mu: Mean vector
        sigma: Covariance matrix
    
    Returns:
        MultivariateNormal: Multivariate normal distribution object
    
    Raises:
        ParameterError: If parameter constraints are violated
    """
    params = MultivariateNormalParams(mu=mu, sigma=sigma)
    return MultivariateNormal(params=params)


# Function to compute normal log-likelihood (for backward compatibility)
def normloglik(x: np.ndarray, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Compute the log-likelihood of data under a normal distribution.
    
    Args:
        x: Data to compute the log-likelihood for
        mu: Mean parameter
        sigma: Standard deviation parameter
    
    Returns:
        float: Log-likelihood value
    
    Raises:
        ValueError: If x contains invalid values or sigma is not positive
    """
    # Create normal distribution
    dist = create_normal(mu=mu, sigma=sigma)
    
    # Compute log-likelihood
    return dist.loglikelihood(x)


# Function to compute multivariate normal log-likelihood (for backward compatibility)
def mvnormloglik(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Compute the log-likelihood of data under a multivariate normal distribution.
    
    Args:
        x: Data to compute the log-likelihood for
        mu: Mean vector
        sigma: Covariance matrix
    
    Returns:
        float: Log-likelihood value
    
    Raises:
        ValueError: If x contains invalid values or sigma is not positive definite
    """
    # Create multivariate normal distribution
    dist = create_multivariate_normal(mu=mu, sigma=sigma)
    
    # Compute log-likelihood
    return dist.loglikelihood(x)