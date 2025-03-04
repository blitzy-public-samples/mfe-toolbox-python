# mfe/models/distributions/composite_likelihood.py
"""
Composite likelihood functions for multivariate models.

This module implements composite likelihood functions for multivariate models,
enabling efficient estimation when full likelihood computation is infeasible.
Composite likelihood methods are particularly useful for high-dimensional
multivariate GARCH models where traditional likelihood approaches become
computationally prohibitive.

The module provides both object-oriented interfaces through the CompositeLikelihood
class and functional APIs for flexibility. Performance-critical sections are
accelerated using Numba's just-in-time compilation for optimal performance.

References:
    Engle, R. F., Shephard, N., & Sheppard, K. (2008). Fitting vast dimensional
time-varying covariance models.
    
    Pakel, C., Shephard, N., Sheppard, K., & Engle, R. F. (2014). Fitting vast
time-varying covariance models.
"""

import warnings
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast
)

import numpy as np
from numba import jit
from scipy import stats

from mfe.core.base import DistributionBase
from mfe.core.parameters import (
    ParameterBase, ParameterError, validate_positive, validate_range,
    validate_probability
)
from mfe.core.exceptions import (
    DistributionError, NumericError, raise_parameter_error, warn_numeric,
    warn_performance
)
from mfe.core.types import (
    Vector, Matrix, ParameterVector, CovarianceMatrix, CorrelationMatrix
)


@dataclass
class CompositeLikelihoodParams(ParameterBase):
    """Parameters for composite likelihood computation.
    
    Attributes:
        block_size: Size of blocks for composite likelihood (must be positive)
        overlap: Overlap between blocks (must be between 0 and 1)
        weights: Optional weights for different blocks (must sum to 1)
    """
    
    block_size: int = 2
    overlap: float = 0.0
    weights: Optional[np.ndarray] = None
    
    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate composite likelihood parameter constraints.
        
        Raises:
            ParameterError: If parameter constraints are violated
        """
        # Validate block_size (must be positive)
        if self.block_size < 1:
            raise ParameterError(
                "Block size must be positive",
                param_name="block_size",
                param_value=self.block_size,
                constraint="block_size >= 1"
            )
        
        # Validate overlap (must be between 0 and 1)
        validate_range(self.overlap, "overlap", 0, 1)
        
        # Validate weights if provided
        if self.weights is not None:
            if not isinstance(self.weights, np.ndarray):
                raise ParameterError(
                    "Weights must be a NumPy array",
                    param_name="weights",
                    param_value=type(self.weights),
                    constraint="weights must be np.ndarray"
                )
            
            if np.any(self.weights < 0):
                raise ParameterError(
                    "Weights must be non-negative",
                    param_name="weights",
                    param_value="contains negative values",
                    constraint="all weights >= 0"
                )
            
            if not np.isclose(np.sum(self.weights), 1.0):
                raise ParameterError(
                    "Weights must sum to 1",
                    param_name="weights",
                    param_value=f"sum = {np.sum(self.weights)}",
                    constraint="sum(weights) = 1"
                )
    
    def to_array(self) -> np.ndarray:
        """Convert parameters to a NumPy array.
        
        Returns:
            np.ndarray: Array representation of parameters [block_size, overlap]
        """
        return np.array([self.block_size, self.overlap])
    
    @classmethod
    def from_array(cls, array: np.ndarray, **kwargs: Any) -> 'CompositeLikelihoodParams':
        """Create parameters from a NumPy array.
        
        Args:
            array: Array representation of parameters [block_size, overlap]
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            CompositeLikelihoodParams: Parameter object
            
        Raises:
            ValueError: If the array length is not 2
        """
        if len(array) != 2:
            raise ValueError(f"Array length must be 2, got {len(array)}")
        
        # Extract parameters
        block_size = int(array[0])
        overlap = array[1]
        
        # Get weights from kwargs if provided
        weights = kwargs.get("weights", None)
        
        return cls(block_size=block_size, overlap=overlap, weights=weights)
    
    def transform(self) -> np.ndarray:
        """Transform parameters to unconstrained space for optimization.
        
        Returns:
            np.ndarray: Transformed parameters in unconstrained space
        """
        # block_size is discrete, so we don't transform it
        # Transform overlap to unconstrained space using logit
        transformed_overlap = np.log(self.overlap / (1 - self.overlap)) if 0 < self.overlap < 1 else 0
        
        return np.array([float(self.block_size), transformed_overlap])
    
    @classmethod
    def inverse_transform(cls, array: np.ndarray, **kwargs: Any) -> 'CompositeLikelihoodParams':
        """Transform parameters from unconstrained space back to constrained space.
        
        Args:
            array: Parameters in unconstrained space
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            CompositeLikelihoodParams: Parameter object with constrained parameters
            
        Raises:
            ValueError: If the array length is not 2
        """
        if len(array) != 2:
            raise ValueError(f"Array length must be 2, got {len(array)}")
        
        # Extract transformed parameters
        transformed_block_size, transformed_overlap = array
        
        # Inverse transform parameters
        block_size = max(1, int(round(transformed_block_size)))
        overlap = 1.0 / (1.0 + np.exp(-transformed_overlap))
        
        # Get weights from kwargs if provided
        weights = kwargs.get("weights", None)
        
        return cls(block_size=block_size, overlap=overlap, weights=weights)


class CompositeLikelihood(DistributionBase):
    """Composite likelihood implementation for multivariate models.
    
    This class implements composite likelihood methods for multivariate models,
    enabling efficient estimation when full likelihood computation is infeasible.
    It provides methods for computing composite log-likelihood values and gradients,
    as well as generating block indices for composite likelihood computation.
    
    Attributes:
        name: A descriptive name for the distribution
        params: Composite likelihood parameters
    """
    
    def __init__(self, 
                name: str = "CompositeLikelihood", 
                params: Optional[CompositeLikelihoodParams] = None) -> None:
        """Initialize the composite likelihood.
        
        Args:
            name: A descriptive name for the distribution
            params: Composite likelihood parameters
        """
        super().__init__(name=name)
        self._params = params or CompositeLikelihoodParams()
        self._validate_params()
    
    @property
    def params(self) -> CompositeLikelihoodParams:
        """Get the composite likelihood parameters.
        
        Returns:
            CompositeLikelihoodParams: The composite likelihood parameters
        """
        return self._params
    
    @params.setter
    def params(self, value: CompositeLikelihoodParams) -> None:
        """Set the composite likelihood parameters.
        
        Args:
            value: The composite likelihood parameters to set
            
        Raises:
            ParameterError: If the parameters are invalid
        """
        self._params = value
        self._validate_params()
    
    def _validate_params(self) -> None:
        """Validate the composite likelihood parameters.
        
        Raises:
            ParameterError: If the parameters are invalid
        """
        if self._params is not None:
            self._params.validate()
    
    def generate_blocks(self, dimension: int) -> List[np.ndarray]:
        """Generate block indices for composite likelihood computation.
        
        This method generates the indices for blocks used in composite likelihood
        computation, taking into account the block size and overlap parameters.
        
        Args:
            dimension: The dimension of the multivariate distribution
            
        Returns:
            List[np.ndarray]: List of index arrays for each block
            
        Raises:
            ValueError: If dimension is less than block_size
        """
        block_size = self._params.block_size
        overlap = self._params.overlap
        
        if dimension < block_size:
            raise ValueError(
                f"Dimension ({dimension}) must be at least block_size ({block_size})"
            )
        
        # For block_size = dimension, return a single block with all indices
        if block_size >= dimension:
            return [np.arange(dimension)]
        
        # Calculate step size based on overlap
        step = max(1, int(block_size * (1 - overlap)))
        
        # Generate blocks
        blocks = []
        start = 0
        while start < dimension:
            end = min(start + block_size, dimension)
            blocks.append(np.arange(start, end))
            start += step
        
        # If the last block is too small, merge it with the previous block
        if len(blocks) > 1 and len(blocks[-1]) < block_size / 2:
            blocks[-2] = np.arange(blocks[-2][0], blocks[-1][-1] + 1)
            blocks.pop()
        
        return blocks
    
    def compute_weights(self, blocks: List[np.ndarray], dimension: int) -> np.ndarray:
        """Compute weights for composite likelihood blocks.
        
        This method computes weights for each block in the composite likelihood
        computation. If weights are provided in the parameters, they are used;
        otherwise, weights are computed based on block sizes.
        
        Args:
            blocks: List of index arrays for each block
            dimension: The dimension of the multivariate distribution
            
        Returns:
            np.ndarray: Array of weights for each block
        """
        if self._params.weights is not None:
            # Use provided weights if they match the number of blocks
            if len(self._params.weights) == len(blocks):
                return self._params.weights
            else:
                warnings.warn(
                    f"Number of provided weights ({len(self._params.weights)}) "
                    f"does not match number of blocks ({len(blocks)}). "
                    "Using default weights based on block sizes."
                )
        
        # Compute weights based on block sizes
        # Each variable should have equal total weight
        variable_counts = np.zeros(dimension)
        for block in blocks:
            for idx in block:
                variable_counts[idx] += 1
        
        weights = np.zeros(len(blocks))
        for i, block in enumerate(blocks):
            # Weight is proportional to sum of inverse counts for variables in block
            block_weight = 0
            for idx in block:
                block_weight += 1 / variable_counts[idx]
            weights[i] = block_weight
        
        # Normalize weights to sum to 1
        weights /= np.sum(weights)
        
        return weights
    
    def loglikelihood(self, 
                     data: np.ndarray, 
                     covariance_fn: Callable[[np.ndarray], np.ndarray],
                     **kwargs: Any) -> float:
        """Compute the composite log-likelihood of the data.
        
        This method computes the composite log-likelihood of the data using
        the provided covariance function. The covariance function should take
        a subset of indices and return the corresponding covariance matrix.
        
        Args:
            data: Data matrix (T×K) where T is the number of observations and K is the dimension
            covariance_fn: Function that takes indices and returns covariance matrix
            **kwargs: Additional keyword arguments for the covariance function
            
        Returns:
            float: Composite log-likelihood value
            
        Raises:
            ValueError: If data dimensions are invalid
            NumericError: If numerical issues occur during computation
        """
        # Validate data
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        
        if data.ndim != 2:
            raise ValueError(f"Data must be 2-dimensional, got {data.ndim} dimensions")
        
        T, K = data.shape
        
        # Generate blocks
        blocks = self.generate_blocks(K)
        
        # Compute weights
        weights = self.compute_weights(blocks, K)
        
        # Compute composite log-likelihood
        cl = 0.0
        for i, block in enumerate(blocks):
            # Extract block data
            block_data = data[:, block]
            
            # Get covariance matrices for this block
            try:
                block_covs = covariance_fn(block)
            except Exception as e:
                raise ValueError(f"Error computing covariance for block {i}: {str(e)}")
            
            # Validate covariance matrices
            if not isinstance(block_covs, np.ndarray):
                raise ValueError(f"Covariance function must return NumPy array, got {type(block_covs)}")
            
            if block_covs.ndim not in (2, 3):
                raise ValueError(
                    f"Covariance function must return 2D or 3D array, got {block_covs.ndim} dimensions"
                )
            
            # Handle both constant and time-varying covariance matrices
            if block_covs.ndim == 2:
                # Constant covariance matrix
                block_size = len(block)
                if block_covs.shape != (block_size, block_size):
                    raise ValueError(
                        f"Covariance matrix shape {block_covs.shape} does not match "
                        f"block size {block_size}"
                    )
                
                # Compute log-likelihood for constant covariance
                block_ll = _compute_mvn_loglikelihood_const(block_data, block_covs)
            else:
                # Time-varying covariance matrices
                block_size = len(block)
                if block_covs.shape[1:] != (block_size, block_size) or block_covs.shape[0] != T:
                    raise ValueError(
                        f"Covariance matrices shape {block_covs.shape} does not match "
                        f"data shape {block_data.shape}"
                    )
                
                # Compute log-likelihood for time-varying covariance
                block_ll = _compute_mvn_loglikelihood_tv(block_data, block_covs)
            
            # Add weighted block log-likelihood to total
            cl += weights[i] * block_ll
        
        return cl
    
    async def loglikelihood_async(self, 
                                 data: np.ndarray, 
                                 covariance_fn: Callable[[np.ndarray], np.ndarray],
                                 progress_callback: Optional[Callable[[float, str], None]] = None,
                                 **kwargs: Any) -> float:
        """Asynchronously compute the composite log-likelihood of the data.
        
        This method provides an asynchronous interface to the loglikelihood method,
        allowing for non-blocking computation and progress reporting.
        
        Args:
            data: Data matrix (T×K) where T is the number of observations and K is the dimension
            covariance_fn: Function that takes indices and returns covariance matrix
            progress_callback: Optional callback function for reporting progress
            **kwargs: Additional keyword arguments for the covariance function
            
        Returns:
            float: Composite log-likelihood value
            
        Raises:
            ValueError: If data dimensions are invalid
            NumericError: If numerical issues occur during computation
        """
        # Validate data
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        
        if data.ndim != 2:
            raise ValueError(f"Data must be 2-dimensional, got {data.ndim} dimensions")
        
        T, K = data.shape
        
        # Generate blocks
        blocks = self.generate_blocks(K)
        
        # Compute weights
        weights = self.compute_weights(blocks, K)
        
        # Compute composite log-likelihood
        cl = 0.0
        for i, block in enumerate(blocks):
            # Report progress if callback is provided
            if progress_callback:
                progress = (i / len(blocks)) * 100
                progress_callback(progress, f"Processing block {i+1}/{len(blocks)}")
            
            # Extract block data
            block_data = data[:, block]
            
            # Get covariance matrices for this block
            try:
                block_covs = covariance_fn(block)
            except Exception as e:
                raise ValueError(f"Error computing covariance for block {i}: {str(e)}")
            
            # Validate covariance matrices
            if not isinstance(block_covs, np.ndarray):
                raise ValueError(f"Covariance function must return NumPy array, got {type(block_covs)}")
            
            if block_covs.ndim not in (2, 3):
                raise ValueError(
                    f"Covariance function must return 2D or 3D array, got {block_covs.ndim} dimensions"
                )
            
            # Handle both constant and time-varying covariance matrices
            if block_covs.ndim == 2:
                # Constant covariance matrix
                block_size = len(block)
                if block_covs.shape != (block_size, block_size):
                    raise ValueError(
                        f"Covariance matrix shape {block_covs.shape} does not match "
                        f"block size {block_size}"
                    )
                
                # Compute log-likelihood for constant covariance
                block_ll = _compute_mvn_loglikelihood_const(block_data, block_covs)
            else:
                # Time-varying covariance matrices
                block_size = len(block)
                if block_covs.shape[1:] != (block_size, block_size) or block_covs.shape[0] != T:
                    raise ValueError(
                        f"Covariance matrices shape {block_covs.shape} does not match "
                        f"data shape {block_data.shape}"
                    )
                
                # Compute log-likelihood for time-varying covariance
                block_ll = _compute_mvn_loglikelihood_tv(block_data, block_covs)
            
            # Add weighted block log-likelihood to total
            cl += weights[i] * block_ll
            
            # Allow for cooperative multitasking
            import asyncio
            await asyncio.sleep(0)
        
        # Final progress report
        if progress_callback:
            progress_callback(100.0, "Composite likelihood computation complete")
        
        return cl
    
    def optimal_block_size(self, 
                          dimension: int, 
                          max_block_size: Optional[int] = None) -> int:
        """Determine the optimal block size for composite likelihood.
        
        This method computes the optimal block size for composite likelihood
        based on the dimension and computational constraints.
        
        Args:
            dimension: The dimension of the multivariate distribution
            max_block_size: Maximum allowed block size (default: None)
            
        Returns:
            int: Optimal block size
        """
        # If dimension is small, use full likelihood
        if dimension <= 10:
            return dimension
        
        # Default maximum block size
        if max_block_size is None:
            max_block_size = min(20, dimension)
        
        # Compute optimal block size based on computational complexity
        # This is a heuristic that balances accuracy and computational cost
        if dimension <= 50:
            optimal_size = min(10, dimension)
        elif dimension <= 100:
            optimal_size = min(5, dimension)
        else:
            optimal_size = min(3, dimension)
        
        # Ensure block size is within bounds
        optimal_size = min(optimal_size, max_block_size)
        
        return optimal_size
    
    def __str__(self) -> str:
        """Generate a string representation of the composite likelihood.
        
        Returns:
            str: A string representation of the composite likelihood
        """
        return (f"{self.name} with block_size={self._params.block_size}, "
                f"overlap={self._params.overlap}")
    
    def __repr__(self) -> str:
        """Generate a detailed string representation of the composite likelihood.
        
        Returns:
            str: A detailed string representation of the composite likelihood
        """
        return (f"CompositeLikelihood(name='{self.name}', "
                f"params=CompositeLikelihoodParams(block_size={self._params.block_size}, "
                f"overlap={self._params.overlap}, weights={self._params.weights}))")


# Numba-accelerated core functions for multivariate normal log-likelihood

@jit(nopython=True, cache=True)
def _compute_mvn_loglikelihood_const(data: np.ndarray, cov: np.ndarray) -> float:
    """Compute multivariate normal log-likelihood with constant covariance.
    
    This function computes the log-likelihood of data under a multivariate normal
    distribution with constant covariance matrix. It is optimized using Numba's
    just-in-time compilation for maximum performance.
    
    Args:
        data: Data matrix (T×K) where T is the number of observations and K is the dimension
        cov: Covariance matrix (K×K)
        
    Returns:
        float: Log-likelihood value
    """
    T, K = data.shape
    
    # Compute log-determinant of covariance matrix
    # Use Cholesky decomposition for numerical stability
    try:
        L = np.linalg.cholesky(cov)
        log_det = 2 * np.sum(np.log(np.diag(L)))
    except:
        # Fallback to eigenvalue decomposition if Cholesky fails
        eigvals = np.linalg.eigvalsh(cov)
        if np.any(eigvals <= 0):
            # Handle non-positive definite matrix
            eigvals = np.maximum(eigvals, 1e-8)
        log_det = np.sum(np.log(eigvals))
    
    # Compute quadratic form (x'Σ^(-1)x) for each observation
    # Use Cholesky decomposition for numerical stability
    quad_form_sum = 0.0
    try:
        # Solve L*L'*x = b using forward and backward substitution
        for t in range(T):
            x = data[t]
            # Solve L*y = x
            y = np.linalg.solve(L, x)
            # Compute y'*y = x'*Σ^(-1)*x
            quad_form_sum += np.sum(y * y)
    except:
        # Fallback to direct inversion if Cholesky approach fails
        try:
            inv_cov = np.linalg.inv(cov)
            for t in range(T):
                x = data[t]
                quad_form_sum += x.dot(inv_cov).dot(x)
        except:
            # If inversion fails, use pseudoinverse
            inv_cov = np.linalg.pinv(cov)
            for t in range(T):
                x = data[t]
                quad_form_sum += x.dot(inv_cov).dot(x)
    
    # Compute log-likelihood
    ll = -0.5 * T * (K * np.log(2 * np.pi) + log_det) - 0.5 * quad_form_sum
    
    return ll


@jit(nopython=True, cache=True)
def _compute_mvn_loglikelihood_tv(data: np.ndarray, covs: np.ndarray) -> float:
    """Compute multivariate normal log-likelihood with time-varying covariance.
    
    This function computes the log-likelihood of data under a multivariate normal
    distribution with time-varying covariance matrices. It is optimized using
    Numba's just-in-time compilation for maximum performance.
    
    Args:
        data: Data matrix (T×K) where T is the number of observations and K is the dimension
        covs: Time-varying covariance matrices (T×K×K)
        
    Returns:
        float: Log-likelihood value
    """
    T, K = data.shape
    ll = 0.0
    
    # Process each time point separately
    for t in range(T):
        x = data[t]
        cov = covs[t]
        
        # Compute log-determinant of covariance matrix
        # Use Cholesky decomposition for numerical stability
        try:
            L = np.linalg.cholesky(cov)
            log_det = 2 * np.sum(np.log(np.diag(L)))
            
            # Solve L*y = x
            y = np.linalg.solve(L, x)
            # Compute y'*y = x'*Σ^(-1)*x
            quad_form = np.sum(y * y)
        except:
            # Fallback to eigenvalue decomposition if Cholesky fails
            eigvals = np.linalg.eigvalsh(cov)
            if np.any(eigvals <= 0):
                # Handle non-positive definite matrix
                eigvals = np.maximum(eigvals, 1e-8)
            log_det = np.sum(np.log(eigvals))
            
            # Use pseudoinverse for quadratic form
            inv_cov = np.linalg.pinv(cov)
            quad_form = x.dot(inv_cov).dot(x)
        
        # Add to log-likelihood
        ll += -0.5 * (K * np.log(2 * np.pi) + log_det + quad_form)
    
    return ll


# Functional API for composite likelihood computation

def composite_loglikelihood(data: np.ndarray, 
                           covariance_fn: Callable[[np.ndarray], np.ndarray],
                           block_size: int = 2,
                           overlap: float = 0.0,
                           weights: Optional[np.ndarray] = None,
                           **kwargs: Any) -> float:
    """Compute composite log-likelihood for multivariate data.
    
    This function provides a convenient functional interface to compute
    composite log-likelihood for multivariate data.
    
    Args:
        data: Data matrix (T×K) where T is the number of observations and K is the dimension
        covariance_fn: Function that takes indices and returns covariance matrix
        block_size: Size of blocks for composite likelihood (default: 2)
        overlap: Overlap between blocks (default: 0.0)
        weights: Optional weights for different blocks
        **kwargs: Additional keyword arguments for the covariance function
        
    Returns:
        float: Composite log-likelihood value
        
    Raises:
        ValueError: If parameters or data dimensions are invalid
        NumericError: If numerical issues occur during computation
    """
    # Create CompositeLikelihood object
    params = CompositeLikelihoodParams(block_size=block_size, overlap=overlap, weights=weights)
    cl = CompositeLikelihood(params=params)
    
    # Compute log-likelihood
    return cl.loglikelihood(data, covariance_fn, **kwargs)


def composite_loglikelihood_async(data: np.ndarray, 
                                      covariance_fn: Callable[[np.ndarray], np.ndarray],
                                      block_size: int = 2,
                                      overlap: float = 0.0,
                                      weights: Optional[np.ndarray] = None,
                                      progress_callback: Optional[Callable[[float, str], None]] = None,
                                      **kwargs: Any) -> float:
    """Asynchronously compute composite log-likelihood for multivariate data.
    
    This function provides a convenient functional interface to asynchronously
    compute composite log-likelihood for multivariate data.
    
    Args:
        data: Data matrix (T×K) where T is the number of observations and K is the dimension
        covariance_fn: Function that takes indices and returns covariance matrix
        block_size: Size of blocks for composite likelihood (default: 2)
        overlap: Overlap between blocks (default: 0.0)
        weights: Optional weights for different blocks
        progress_callback: Optional callback function for reporting progress
        **kwargs: Additional keyword arguments for the covariance function
        
    Returns:
        float: Composite log-likelihood value
        
    Raises:
        ValueError: If parameters or data dimensions are invalid
        NumericError: If numerical issues occur during computation
    """
    # Create CompositeLikelihood object
    params = CompositeLikelihoodParams(block_size=block_size, overlap=overlap, weights=weights)
    cl = CompositeLikelihood(params=params)
    
    # Compute log-likelihood asynchronously
    return await cl.loglikelihood_async(data, covariance_fn, progress_callback, **kwargs)


def optimal_block_size(dimension: int, max_block_size: Optional[int] = None) -> int:
    """Determine the optimal block size for composite likelihood.
    
    This function computes the optimal block size for composite likelihood
    based on the dimension and computational constraints.
    
    Args:
        dimension: The dimension of the multivariate distribution
        max_block_size: Maximum allowed block size (default: None)
        
    Returns:
        int: Optimal block size
    """
    cl = CompositeLikelihood()
    return cl.optimal_block_size(dimension, max_block_size)


def generate_blocks(dimension: int, 
                   block_size: int = 2, 
                   overlap: float = 0.0) -> List[np.ndarray]:
    """Generate block indices for composite likelihood computation.
    
    This function generates the indices for blocks used in composite likelihood
    computation, taking into account the block size and overlap parameters.
    
    Args:
        dimension: The dimension of the multivariate distribution
        block_size: Size of blocks for composite likelihood (default: 2)
        overlap: Overlap between blocks (default: 0.0)
        
    Returns:
        List[np.ndarray]: List of index arrays for each block
        
    Raises:
        ValueError: If dimension is less than block_size or parameters are invalid
    """
    # Create CompositeLikelihood object
    params = CompositeLikelihoodParams(block_size=block_size, overlap=overlap)
    cl = CompositeLikelihood(params=params)
    
    # Generate blocks
    return cl.generate_blocks(dimension)


def compute_weights(blocks: List[np.ndarray], dimension: int) -> np.ndarray:
    """Compute weights for composite likelihood blocks.
    
    This function computes weights for each block in the composite likelihood
    computation based on block sizes.
    
    Args:
        blocks: List of index arrays for each block
        dimension: The dimension of the multivariate distribution
        
    Returns:
        np.ndarray: Array of weights for each block
    """
    cl = CompositeLikelihood()
    return cl.compute_weights(blocks, dimension)


# Utility functions for multivariate normal log-likelihood

def mvn_loglikelihood(data: np.ndarray, 
                     cov: Union[np.ndarray, Callable[[int], np.ndarray]],
                     **kwargs: Any) -> float:
    """Compute multivariate normal log-likelihood.
    
    This function computes the log-likelihood of data under a multivariate normal
    distribution with either constant or time-varying covariance.
    
    Args:
        data: Data matrix (T×K) where T is the number of observations and K is the dimension
        cov: Covariance matrix (K×K) or time-varying covariance matrices (T×K×K)
             or a function that returns covariance matrices
        **kwargs: Additional keyword arguments for the covariance function
        
    Returns:
        float: Log-likelihood value
        
    Raises:
        ValueError: If data dimensions are invalid
        NumericError: If numerical issues occur during computation
    """
    # Validate data
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    
    if data.ndim != 2:
        raise ValueError(f"Data must be 2-dimensional, got {data.ndim} dimensions")
    
    T, K = data.shape
    
    # Handle different covariance inputs
    if callable(cov):
        # Covariance function
        try:
            cov_matrix = cov(np.arange(K))
        except Exception as e:
            raise ValueError(f"Error computing covariance: {str(e)}")
    else:
        # Covariance matrix or matrices
        cov_matrix = cov
    
    # Validate covariance
    if not isinstance(cov_matrix, np.ndarray):
        raise ValueError(f"Covariance must be NumPy array, got {type(cov_matrix)}")
    
    # Compute log-likelihood based on covariance type
    if cov_matrix.ndim == 2:
        # Constant covariance matrix
        if cov_matrix.shape != (K, K):
            raise ValueError(
                f"Covariance matrix shape {cov_matrix.shape} does not match "
                f"data dimension {K}"
            )
        
        return _compute_mvn_loglikelihood_const(data, cov_matrix)
    elif cov_matrix.ndim == 3:
        # Time-varying covariance matrices
        if cov_matrix.shape[1:] != (K, K) or cov_matrix.shape[0] != T:
            raise ValueError(
                f"Covariance matrices shape {cov_matrix.shape} does not match "
                f"data shape {data.shape}"
            )
        
        return _compute_mvn_loglikelihood_tv(data, cov_matrix)
    else:
        raise ValueError(f"Covariance must be 2D or 3D array, got {cov_matrix.ndim} dimensions")


def mvn_loglikelihood_with_mean(data: np.ndarray, 
                               mean: Union[np.ndarray, Callable[[int], np.ndarray]],
                               cov: Union[np.ndarray, Callable[[int], np.ndarray]],
                               **kwargs: Any) -> float:
    """Compute multivariate normal log-likelihood with non-zero mean.
    
    This function computes the log-likelihood of data under a multivariate normal
    distribution with non-zero mean and either constant or time-varying covariance.
    
    Args:
        data: Data matrix (T×K) where T is the number of observations and K is the dimension
        mean: Mean vector (K) or time-varying mean vectors (T×K)
              or a function that returns mean vectors
        cov: Covariance matrix (K×K) or time-varying covariance matrices (T×K×K)
             or a function that returns covariance matrices
        **kwargs: Additional keyword arguments for the mean and covariance functions
        
    Returns:
        float: Log-likelihood value
        
    Raises:
        ValueError: If data dimensions are invalid
        NumericError: If numerical issues occur during computation
    """
    # Validate data
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    
    if data.ndim != 2:
        raise ValueError(f"Data must be 2-dimensional, got {data.ndim} dimensions")
    
    T, K = data.shape
    
    # Handle different mean inputs
    if callable(mean):
        # Mean function
        try:
            mean_vector = mean(np.arange(K))
        except Exception as e:
            raise ValueError(f"Error computing mean: {str(e)}")
    else:
        # Mean vector or vectors
        mean_vector = mean
    
    # Validate mean
    if not isinstance(mean_vector, np.ndarray):
        raise ValueError(f"Mean must be NumPy array, got {type(mean_vector)}")
    
    # Prepare centered data
    if mean_vector.ndim == 1:
        # Constant mean vector
        if mean_vector.shape != (K,):
            raise ValueError(
                f"Mean vector shape {mean_vector.shape} does not match "
                f"data dimension {K}"
            )
        
        # Center data
        centered_data = data - mean_vector
    elif mean_vector.ndim == 2:
        # Time-varying mean vectors
        if mean_vector.shape != (T, K):
            raise ValueError(
                f"Mean vectors shape {mean_vector.shape} does not match "
                f"data shape {data.shape}"
            )
        
        # Center data
        centered_data = data - mean_vector
    else:
        raise ValueError(f"Mean must be 1D or 2D array, got {mean_vector.ndim} dimensions")
    
    # Compute log-likelihood using centered data
    return mvn_loglikelihood(centered_data, cov, **kwargs)
