"""
Stationary Bootstrap implementation for the MFE Toolbox.

This module implements the stationary bootstrap algorithm for dependent time series data,
as proposed by Politis and Romano (1994). The stationary bootstrap generates bootstrap
replicates by resampling blocks of random length, where the block length follows a
geometric distribution with a specified expected value.

Unlike the block bootstrap which uses fixed-length blocks, the stationary bootstrap
uses random block lengths, which ensures that the resampled series is stationary.
This property makes the stationary bootstrap particularly useful for time series
with complex dependence structures.

The implementation leverages NumPy's efficient array operations with performance-critical
sections accelerated using Numba's @jit decorators. This approach provides significant
performance improvements while maintaining the flexibility and readability of Python code.

References:
    Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap.
    Journal of the American Statistical Association, 89(428), 1303-1313.
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
def _generate_stationary_bootstrap_indices(
    data_length: int,
    n_bootstraps: int,
    expected_block_length: float,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate stationary bootstrap indices using Numba acceleration.
    
    This function implements the core algorithm for generating stationary bootstrap
    indices with random block lengths following a geometric distribution. It is
    accelerated using Numba's @jit decorator for improved performance on large
    datasets and many bootstrap replications.
    
    Args:
        data_length: Length of the original data
        n_bootstraps: Number of bootstrap samples to generate
        expected_block_length: Expected length of each block (parameter for geometric distribution)
        seed: Random seed for reproducibility
        
    Returns:
        np.ndarray: Bootstrap indices with shape (n_bootstraps, data_length)
    """
    # Initialize random number generator
    if seed is not None:
        np.random.seed(seed)
    
    # Calculate probability parameter for geometric distribution
    # p is the probability of starting a new block
    p = 1.0 / expected_block_length
    
    # Initialize indices array
    indices = np.zeros((n_bootstraps, data_length), dtype=np.int64)
    
    # Generate bootstrap indices for each bootstrap sample
    for i in range(n_bootstraps):
        # Initialize the first index randomly
        idx = 0
        indices[i, 0] = np.random.randint(0, data_length)
        
        # Generate the rest of the indices
        while idx < data_length - 1:
            # Decide whether to start a new block or continue the current one
            if np.random.random() < p:
                # Start a new block with a random index
                indices[i, idx + 1] = np.random.randint(0, data_length)
            else:
                # Continue the current block (with circular wrapping)
                indices[i, idx + 1] = (indices[i, idx] + 1) % data_length
            
            idx += 1
    
    return indices


class StationaryBootstrap(BootstrapBase):
    """
    Stationary Bootstrap implementation for dependent time series data.
    
    This class implements the stationary bootstrap algorithm proposed by Politis and
    Romano (1994), which generates bootstrap replicates by resampling blocks of random
    length. The block length follows a geometric distribution with a specified expected
    value, ensuring that the resampled series is stationary.
    
    The implementation leverages NumPy's efficient array operations with
    performance-critical sections accelerated using Numba's @jit decorators.
    
    Attributes:
        params: Bootstrap parameters including n_bootstraps and expected_block_length
        _bootstrap_indices: Generated bootstrap indices
        _bootstrap_samples: Generated bootstrap samples
        _bootstrap_statistics: Computed bootstrap statistics
        _original_statistic: Statistic computed on the original data
        _confidence_intervals: Computed confidence intervals
        _p_value: Computed bootstrap p-value
    """
    
    def __init__(
        self,
        expected_block_length: float,
        n_bootstraps: int = 1000,
        random_state: Optional[Union[int, np.random.Generator]] = None,
        name: str = "Stationary Bootstrap"
    ) -> None:
        """
        Initialize the Stationary Bootstrap.
        
        Args:
            expected_block_length: Expected length of each block (parameter for geometric distribution)
            n_bootstraps: Number of bootstrap samples to generate
            random_state: Random number generator seed for reproducibility
            name: Name of the bootstrap method
            
        Raises:
            ParameterError: If parameters violate constraints
        """
        super().__init__(
            n_bootstraps=n_bootstraps,
            block_length=expected_block_length,  # Store expected_block_length in block_length parameter
            random_state=random_state,
            name=name
        )
        
        # Validate expected_block_length specifically for stationary bootstrap
        if not isinstance(expected_block_length, (int, float)):
            raise ParameterError(
                "expected_block_length must be a number",
                param_name="expected_block_length",
                param_value=expected_block_length
            )
        
        if expected_block_length <= 0:
            raise ParameterError(
                "expected_block_length must be positive",
                param_name="expected_block_length",
                param_value=expected_block_length
            )
    
    @property
    def expected_block_length(self) -> float:
        """
        Get the expected block length.
        
        Returns:
            float: Expected block length
        """
        return float(self.params.block_length)
    
    def generate_indices(
        self,
        data_length: int,
        n_bootstraps: int,
        random_state: Optional[Union[int, np.random.Generator]] = None
    ) -> np.ndarray:
        """
        Generate bootstrap indices for stationary bootstrap.
        
        This method implements the stationary bootstrap algorithm, which generates
        bootstrap replicates by resampling blocks of random length. The block length
        follows a geometric distribution with a specified expected value.
        
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
        
        # Get expected block length from parameters
        expected_block_length = float(self.params.block_length)
        
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
            indices = _generate_stationary_bootstrap_indices(
                data_length=data_length,
                n_bootstraps=n_bootstraps,
                expected_block_length=expected_block_length,
                seed=seed
            )
            return indices
        except Exception as e:
            # Handle any errors during index generation
            raise BootstrapError(
                f"Failed to generate stationary bootstrap indices: {str(e)}",
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
        Run the stationary bootstrap procedure asynchronously.
        
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
        """Return a string representation of the StationaryBootstrap instance."""
        if self._fitted:
            return (
                f"StationaryBootstrap(expected_block_length={self.expected_block_length}, "
                f"n_bootstraps={self.params.n_bootstraps}, fitted=True)"
            )
        else:
            return (
                f"StationaryBootstrap(expected_block_length={self.expected_block_length}, "
                f"n_bootstraps={self.params.n_bootstraps}, fitted=False)"
            )
    
    def __repr__(self) -> str:
        """Return a detailed string representation of the StationaryBootstrap instance."""
        return (
            f"StationaryBootstrap(expected_block_length={self.expected_block_length}, "
            f"n_bootstraps={self.params.n_bootstraps}, "
            f"random_state={self.params.random_state}, "
            f"fitted={self._fitted})"
        )