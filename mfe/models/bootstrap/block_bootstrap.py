
"""
Block Bootstrap implementation for the MFE Toolbox.

This module implements the circular block bootstrap algorithm for stationary, 
dependent time series data. The block bootstrap generates bootstrap replicates 
by resampling contiguous blocks of a specified length with circular wrapping 
for indices that exceed the original data length.

The implementation leverages NumPy's efficient array operations with 
performance-critical sections accelerated using Numba's @jit decorators.
This approach provides significant performance improvements while maintaining
the flexibility and readability of Python code.
"""

from typing import Optional, Union, Tuple, List, Dict, Any, Callable
import numpy as np
from numba import jit

from mfe.core.exceptions import ParameterError, BootstrapError
from mfe.core.types import (
    BootstrapIndices, BootstrapSamples, BootstrapStatistic, 
    BootstrapResult, ProgressCallback
)
from mfe.models.bootstrap.base import BootstrapBase, BootstrapParameters


@jit(nopython=True, cache=True)
def _generate_block_bootstrap_indices(
    data_length: int,
    n_bootstraps: int,
    block_length: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate block bootstrap indices using Numba acceleration.
    
    This function implements the core algorithm for generating block bootstrap
    indices with circular wrapping. It is accelerated using Numba's @jit decorator
    for improved performance on large datasets and many bootstrap replications.
    
    Args:
        data_length: Length of the original data
        n_bootstraps: Number of bootstrap samples to generate
        block_length: Length of each block
        seed: Random seed for reproducibility
        
    Returns:
        np.ndarray: Bootstrap indices with shape (n_bootstraps, data_length)
    """
    # Initialize random number generator
    if seed is not None:
        np.random.seed(seed)
    
    # Calculate number of blocks needed to cover the data length
    n_blocks = int(np.ceil(data_length / block_length))
    
    # Initialize indices array
    indices = np.zeros((n_bootstraps, data_length), dtype=np.int64)
    
    # Generate bootstrap indices for each bootstrap sample
    for i in range(n_bootstraps):
        # Generate random starting positions for blocks
        block_starts = np.random.randint(0, data_length, size=n_blocks)
        
        # Fill the indices array with blocks
        idx = 0
        for start in block_starts:
            # Add indices from the current block
            for j in range(block_length):
                if idx >= data_length:
                    break
                # Use modulo to implement circular wrapping
                indices[i, idx] = (start + j) % data_length
                idx += 1
    
    return indices


class BlockBootstrap(BootstrapBase):
    """
    Block Bootstrap implementation for dependent time series data.
    
    This class implements the circular block bootstrap algorithm, which generates
    bootstrap replicates by resampling contiguous blocks of a specified length.
    The algorithm uses circular wrapping for indices that exceed the original
    data length, ensuring that all observations have an equal probability of
    being included in the bootstrap samples.
    
    The implementation leverages NumPy's efficient array operations with
    performance-critical sections accelerated using Numba's @jit decorators.
    
    Attributes:
        params: Bootstrap parameters including n_bootstraps and block_length
        _bootstrap_indices: Generated bootstrap indices
        _bootstrap_samples: Generated bootstrap samples
        _bootstrap_statistics: Computed bootstrap statistics
        _original_statistic: Statistic computed on the original data
        _confidence_intervals: Computed confidence intervals
        _p_value: Computed bootstrap p-value
    """
    
    def __init__(
        self,
        block_length: int,
        n_bootstraps: int = 1000,
        random_state: Optional[Union[int, np.random.Generator]] = None,
        name: str = "Block Bootstrap"
    ) -> None:
        """
        Initialize the Block Bootstrap.
        
        Args:
            block_length: Length of each block
            n_bootstraps: Number of bootstrap samples to generate
            random_state: Random number generator seed for reproducibility
            name: Name of the bootstrap method
            
        Raises:
            ParameterError: If parameters violate constraints
        """
        super().__init__(
            n_bootstraps=n_bootstraps,
            block_length=block_length,
            random_state=random_state,
            name=name
        )
        
        # Validate block_length specifically for block bootstrap
        if not isinstance(block_length, int):
            raise ParameterError(
                "block_length must be an integer for BlockBootstrap",
                param_name="block_length",
                param_value=block_length
            )
        
        if block_length <= 0:
            raise ParameterError(
                "block_length must be positive",
                param_name="block_length",
                param_value=block_length
            )
    
    def generate_indices(
        self,
        data_length: int,
        n_bootstraps: int,
        random_state: Optional[Union[int, np.random.Generator]] = None
    ) -> np.ndarray:
        """
        Generate bootstrap indices for block bootstrap.
        
        This method implements the circular block bootstrap algorithm, which
        generates bootstrap replicates by resampling contiguous blocks of a
        specified length with circular wrapping for indices that exceed the
        original data length.
        
        Args:
            data_length: Length of the original data
            n_bootstraps: Number of bootstrap samples to generate
            random_state: Random number generator seed for reproducibility
            
        Returns:
            np.ndarray: Bootstrap indices with shape (n_bootstraps, data_length)
            
        Raises:
            BootstrapError: If parameters are invalid or generation fails
        """
        # Validate inputs
        if data_length <= 0:
            raise BootstrapError(
                "data_length must be positive",
                bootstrap_type=self._name,
                n_bootstraps=n_bootstraps,
                issue="Invalid data length"
            )
        
        if n_bootstraps <= 0:
            raise BootstrapError(
                "n_bootstraps must be positive",
                bootstrap_type=self._name,
                n_bootstraps=n_bootstraps,
                issue="Invalid number of bootstraps"
            )
        
        # Get block length from parameters
        block_length = int(self.params.block_length)
        
        # Validate block length relative to data length
        if block_length > data_length:
            raise BootstrapError(
                f"Block length ({block_length}) cannot exceed data length ({data_length})",
                bootstrap_type=self._name,
                n_bootstraps=n_bootstraps,
                issue="Block length too large"
            )
        
        # Handle random state
        seed = None
        if random_state is not None:
            if isinstance(random_state, np.random.Generator):
                # If a Generator is provided, we need to extract a seed for Numba
                # This is a limitation of Numba's compatibility with numpy.random
                seed = int(random_state.integers(0, 2**31 - 1))
            else:
                seed = random_state
        elif self.params.random_state is not None:
            if isinstance(self.params.random_state, np.random.Generator):
                seed = int(self.params.random_state.integers(0, 2**31 - 1))
            else:
                seed = self.params.random_state
        
        try:
            # Generate indices using Numba-accelerated function
            indices = _generate_block_bootstrap_indices(
                data_length=data_length,
                n_bootstraps=n_bootstraps,
                block_length=block_length,
                seed=seed
            )
            return indices
        except Exception as e:
            # Handle any errors during index generation
            raise BootstrapError(
                f"Failed to generate block bootstrap indices: {str(e)}",
                bootstrap_type=self._name,
                n_bootstraps=n_bootstraps,
                issue="Index generation failed",
                details=str(e)
            ) from e
    
    async def fit_async(
        self,
        data: np.ndarray,
        statistic_func: Callable[[np.ndarray], Union[float, np.ndarray]],
        confidence_level: float = 0.95,
        compute_p_value: bool = False,
        alternative: str = "two-sided",
        progress_callback: Optional[ProgressCallback] = None,
        **kwargs: Any
    ) -> BootstrapResult:
        """
        Run the block bootstrap procedure asynchronously.
        
        This method provides an asynchronous interface to the fit method,
        allowing for non-blocking bootstrap execution with progress reporting.
        
        Args:
            data: Data to bootstrap
            statistic_func: Function to compute the statistic of interest
            confidence_level: Confidence level for intervals (between 0 and 1)
            compute_p_value: Whether to compute bootstrap p-value
            alternative: Alternative hypothesis for p-value ('two-sided', 'greater', or 'less')
            progress_callback: Callback function for reporting progress
            **kwargs: Additional keyword arguments
            
        Returns:
            BootstrapResult: Bootstrap results
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Call the parent class implementation
        return await super().fit_async(
            data=data,
            statistic_func=statistic_func,
            confidence_level=confidence_level,
            compute_p_value=compute_p_value,
            alternative=alternative,
            progress_callback=progress_callback,
            **kwargs
        )
    
    def __str__(self) -> str:
        """Return a string representation of the BlockBootstrap instance."""
        if self._fitted:
            return (
                f"BlockBootstrap(block_length={self.params.block_length}, "
                f"n_bootstraps={self.params.n_bootstraps}, fitted=True)"
            )
        else:
            return (
                f"BlockBootstrap(block_length={self.params.block_length}, "
                f"n_bootstraps={self.params.n_bootstraps}, fitted=False)"
            )
    
    def __repr__(self) -> str:
        """Return a detailed string representation of the BlockBootstrap instance."""
        return (
            f"BlockBootstrap(block_length={self.params.block_length}, "
            f"n_bootstraps={self.params.n_bootstraps}, "
            f"random_state={self.params.random_state}, "
            f"fitted={self._fitted})"
        )
