# mfe/models/bootstrap/base.py

"""
Abstract base class for bootstrap methods in the MFE Toolbox.

This module defines the abstract base class for bootstrap methods, establishing
the common interface and functionality for all bootstrap implementations. It provides
the foundation for the object-oriented design of bootstrap methods, ensuring consistent
behavior and API across different bootstrap techniques.

The base class implements common validation logic, parameter handling, and defines
the contract that all bootstrap implementations must follow. This design enables
code reuse and consistent error handling across different bootstrap methods.
"""

import abc
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, Union, cast, overload
)
import numpy as np

from mfe.core.base import ModelBase
from mfe.core.exceptions import (
    BootstrapError, ParameterError, DimensionError, DataError, 
    raise_parameter_error, raise_dimension_error, raise_data_error
)
from mfe.core.types import (
    BootstrapIndices, BootstrapSamples, BootstrapStatistic, 
    BootstrapResult, BootstrapType, ProgressCallback
)


@dataclass
class BootstrapParameters:
    """Parameters for bootstrap methods.
    
    This dataclass encapsulates the common parameters used across different
    bootstrap methods, providing type validation and consistent parameter handling.
    
    Attributes:
        n_bootstraps: Number of bootstrap samples to generate
        block_length: Block length for block bootstrap methods
        random_state: Random number generator seed for reproducibility
    """
    
    n_bootstraps: int = 1000
    block_length: Optional[Union[int, float]] = None
    random_state: Optional[Union[int, np.random.Generator]] = None
    
    def __post_init__(self) -> None:
        """Validate parameters after initialization.
        
        Raises:
            ParameterError: If parameters violate constraints
        """
        # Validate n_bootstraps
        if not isinstance(self.n_bootstraps, int):
            raise ParameterError(
                "n_bootstraps must be an integer",
                param_name="n_bootstraps",
                param_value=self.n_bootstraps
            )
        if self.n_bootstraps <= 0:
            raise ParameterError(
                "n_bootstraps must be positive",
                param_name="n_bootstraps",
                param_value=self.n_bootstraps
            )
        
        # Validate block_length if provided
        if self.block_length is not None:
            if not isinstance(self.block_length, (int, float)):
                raise ParameterError(
                    "block_length must be a number",
                    param_name="block_length",
                    param_value=self.block_length
                )
            if self.block_length <= 0:
                raise ParameterError(
                    "block_length must be positive",
                    param_name="block_length",
                    param_value=self.block_length
                )
        
        # Validate random_state if provided
        if self.random_state is not None:
            if not isinstance(self.random_state, (int, np.random.Generator)):
                raise ParameterError(
                    "random_state must be an integer or numpy.random.Generator",
                    param_name="random_state",
                    param_value=type(self.random_state)
                )


@dataclass
class BootstrapResult:
    """Result container for bootstrap methods.
    
    This dataclass encapsulates the results of bootstrap operations, providing
    a consistent structure for accessing bootstrap statistics and diagnostics.
    
    Attributes:
        bootstrap_type: Type of bootstrap method used
        original_statistic: Statistic computed on the original data
        bootstrap_statistics: Statistics computed on bootstrap samples
        confidence_intervals: Confidence intervals for the statistic
        p_value: Bootstrap p-value if applicable
        n_bootstraps: Number of bootstrap samples used
        block_length: Block length used (for block bootstrap methods)
        indices: Bootstrap indices used to generate samples
    """
    
    bootstrap_type: str
    original_statistic: Union[float, np.ndarray]
    bootstrap_statistics: np.ndarray
    confidence_intervals: Optional[np.ndarray] = None
    p_value: Optional[float] = None
    n_bootstraps: int = 0
    block_length: Optional[Union[int, float]] = None
    indices: Optional[np.ndarray] = None
    
    def __post_init__(self) -> None:
        """Validate result object after initialization."""
        if self.n_bootstraps == 0:
            self.n_bootstraps = len(self.bootstrap_statistics)


class BootstrapBase(ModelBase):
    """Abstract base class for bootstrap methods.
    
    This class defines the common interface and functionality for all bootstrap
    implementations in the MFE Toolbox. It provides methods for generating bootstrap
    samples, computing bootstrap statistics, and validating inputs.
    
    Subclasses must implement the generate_indices method to define the specific
    resampling strategy for each bootstrap method.
    """
    
    def __init__(
        self,
        n_bootstraps: int = 1000,
        block_length: Optional[Union[int, float]] = None,
        random_state: Optional[Union[int, np.random.Generator]] = None,
        name: str = "Bootstrap"
    ) -> None:
        """Initialize the bootstrap method.
        
        Args:
            n_bootstraps: Number of bootstrap samples to generate
            block_length: Block length for block bootstrap methods
            random_state: Random number generator seed for reproducibility
            name: Name of the bootstrap method
            
        Raises:
            ParameterError: If parameters violate constraints
        """
        super().__init__(name=name)
        
        # Create and validate parameters
        self.params = BootstrapParameters(
            n_bootstraps=n_bootstraps,
            block_length=block_length,
            random_state=random_state
        )
        
        # Initialize result attributes
        self._bootstrap_indices: Optional[np.ndarray] = None
        self._bootstrap_samples: Optional[np.ndarray] = None
        self._bootstrap_statistics: Optional[np.ndarray] = None
        self._original_statistic: Optional[Union[float, np.ndarray]] = None
        self._confidence_intervals: Optional[np.ndarray] = None
        self._p_value: Optional[float] = None
    
    @property
    def bootstrap_indices(self) -> Optional[np.ndarray]:
        """Get the bootstrap indices.
        
        Returns:
            Optional[np.ndarray]: Bootstrap indices if available, None otherwise
            
        Raises:
            RuntimeError: If bootstrap has not been run
        """
        if not self._fitted:
            raise RuntimeError("Bootstrap has not been run. Call fit() first.")
        return self._bootstrap_indices
    
    @property
    def bootstrap_samples(self) -> Optional[np.ndarray]:
        """Get the bootstrap samples.
        
        Returns:
            Optional[np.ndarray]: Bootstrap samples if available, None otherwise
            
        Raises:
            RuntimeError: If bootstrap has not been run
        """
        if not self._fitted:
            raise RuntimeError("Bootstrap has not been run. Call fit() first.")
        return self._bootstrap_samples
    
    @property
    def bootstrap_statistics(self) -> Optional[np.ndarray]:
        """Get the bootstrap statistics.
        
        Returns:
            Optional[np.ndarray]: Bootstrap statistics if available, None otherwise
            
        Raises:
            RuntimeError: If bootstrap has not been run
        """
        if not self._fitted:
            raise RuntimeError("Bootstrap has not been run. Call fit() first.")
        return self._bootstrap_statistics
    
    @property
    def original_statistic(self) -> Optional[Union[float, np.ndarray]]:
        """Get the original statistic.
        
        Returns:
            Optional[Union[float, np.ndarray]]: Original statistic if available, None otherwise
            
        Raises:
            RuntimeError: If bootstrap has not been run
        """
        if not self._fitted:
            raise RuntimeError("Bootstrap has not been run. Call fit() first.")
        return self._original_statistic
    
    @property
    def confidence_intervals(self) -> Optional[np.ndarray]:
        """Get the confidence intervals.
        
        Returns:
            Optional[np.ndarray]: Confidence intervals if available, None otherwise
            
        Raises:
            RuntimeError: If bootstrap has not been run
        """
        if not self._fitted:
            raise RuntimeError("Bootstrap has not been run. Call fit() first.")
        return self._confidence_intervals
    
    @property
    def p_value(self) -> Optional[float]:
        """Get the bootstrap p-value.
        
        Returns:
            Optional[float]: Bootstrap p-value if available, None otherwise
            
        Raises:
            RuntimeError: If bootstrap has not been run
        """
        if not self._fitted:
            raise RuntimeError("Bootstrap has not been run. Call fit() first.")
        return self._p_value
    
    @abc.abstractmethod
    def generate_indices(
        self,
        data_length: int,
        n_bootstraps: int,
        random_state: Optional[Union[int, np.random.Generator]] = None
    ) -> np.ndarray:
        """Generate bootstrap indices.
        
        This abstract method must be implemented by subclasses to define the
        specific resampling strategy for each bootstrap method.
        
        Args:
            data_length: Length of the original data
            n_bootstraps: Number of bootstrap samples to generate
            random_state: Random number generator seed for reproducibility
            
        Returns:
            np.ndarray: Bootstrap indices with shape (n_bootstraps, data_length)
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement generate_indices")
    
    def sample(
        self,
        data: np.ndarray,
        indices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Generate bootstrap samples from data using indices.
        
        Args:
            data: Original data to resample
            indices: Bootstrap indices to use for resampling. If None, generates new indices.
            
        Returns:
            np.ndarray: Bootstrap samples
            
        Raises:
            ValueError: If indices shape is incompatible with data
        """
        # Validate data
        self.validate_data(data)
        
        # Generate indices if not provided
        if indices is None:
            indices = self.generate_indices(
                data_length=len(data),
                n_bootstraps=self.params.n_bootstraps,
                random_state=self.params.random_state
            )
        
        # Validate indices
        if indices.ndim != 2:
            raise DimensionError(
                "Bootstrap indices must be 2-dimensional",
                array_name="indices",
                expected_shape="(n_bootstraps, data_length)",
                actual_shape=indices.shape
            )
        
        if indices.shape[1] != len(data):
            raise DimensionError(
                "Bootstrap indices second dimension must match data length",
                array_name="indices",
                expected_shape=f"(n_bootstraps, {len(data)})",
                actual_shape=indices.shape
            )
        
        # Generate bootstrap samples
        n_bootstraps = indices.shape[0]
        
        # Handle different data dimensions
        if data.ndim == 1:
            # For 1D data, use simple indexing
            bootstrap_samples = np.array([data[idx] for idx in indices])
        elif data.ndim == 2:
            # For 2D data, index along the first dimension
            bootstrap_samples = np.array([data[idx, :] for idx in indices])
        else:
            raise DimensionError(
                "Data must be 1 or 2-dimensional",
                array_name="data",
                expected_shape="(n_observations,) or (n_observations, n_variables)",
                actual_shape=data.shape
            )
        
        return bootstrap_samples
    
    def compute_statistic(
        self,
        data: np.ndarray,
        indices: np.ndarray,
        statistic_func: Callable[[np.ndarray], Union[float, np.ndarray]]
    ) -> np.ndarray:
        """Compute bootstrap statistics.
        
        Args:
            data: Original data
            indices: Bootstrap indices
            statistic_func: Function to compute the statistic of interest
            
        Returns:
            np.ndarray: Bootstrap statistics
        """
        # Generate bootstrap samples
        bootstrap_samples = self.sample(data, indices)
        
        # Compute statistics for each bootstrap sample
        bootstrap_statistics = np.array([
            statistic_func(sample) for sample in bootstrap_samples
        ])
        
        return bootstrap_statistics
    
    def compute_confidence_interval(
        self,
        bootstrap_statistics: np.ndarray,
        confidence_level: float = 0.95
    ) -> np.ndarray:
        """Compute confidence intervals from bootstrap statistics.
        
        Args:
            bootstrap_statistics: Bootstrap statistics
            confidence_level: Confidence level (between 0 and 1)
            
        Returns:
            np.ndarray: Confidence intervals [lower, upper]
            
        Raises:
            ParameterError: If confidence_level is not between 0 and 1
        """
        if not 0 < confidence_level < 1:
            raise ParameterError(
                "confidence_level must be between 0 and 1",
                param_name="confidence_level",
                param_value=confidence_level
            )
        
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        # Handle different dimensions of bootstrap_statistics
        if bootstrap_statistics.ndim == 1:
            # For 1D statistics (scalar statistic)
            lower = np.percentile(bootstrap_statistics, lower_percentile)
            upper = np.percentile(bootstrap_statistics, upper_percentile)
            return np.array([lower, upper])
        else:
            # For multi-dimensional statistics
            lower = np.percentile(bootstrap_statistics, lower_percentile, axis=0)
            upper = np.percentile(bootstrap_statistics, upper_percentile, axis=0)
            return np.array([lower, upper])
    
    def compute_p_value(
        self,
        bootstrap_statistics: np.ndarray,
        original_statistic: Union[float, np.ndarray],
        alternative: str = "two-sided"
    ) -> float:
        """Compute bootstrap p-value.
        
        Args:
            bootstrap_statistics: Bootstrap statistics
            original_statistic: Statistic computed on the original data
            alternative: Alternative hypothesis ('two-sided', 'greater', or 'less')
            
        Returns:
            float: Bootstrap p-value
            
        Raises:
            ValueError: If alternative is not one of 'two-sided', 'greater', or 'less'
        """
        if alternative not in ["two-sided", "greater", "less"]:
            raise ValueError(
                "alternative must be one of 'two-sided', 'greater', or 'less'"
            )
        
        # Compute p-value based on alternative hypothesis
        if alternative == "two-sided":
            p_value = np.mean(np.abs(bootstrap_statistics) >= np.abs(original_statistic))
        elif alternative == "greater":
            p_value = np.mean(bootstrap_statistics >= original_statistic)
        else:  # alternative == "less"
            p_value = np.mean(bootstrap_statistics <= original_statistic)
        
        return float(p_value)
    
    def fit(
        self,
        data: np.ndarray,
        statistic_func: Callable[[np.ndarray], Union[float, np.ndarray]],
        confidence_level: float = 0.95,
        compute_p_value: bool = False,
        alternative: str = "two-sided",
        **kwargs: Any
    ) -> BootstrapResult:
        """Run the bootstrap procedure.
        
        Args:
            data: Data to bootstrap
            statistic_func: Function to compute the statistic of interest
            confidence_level: Confidence level for intervals (between 0 and 1)
            compute_p_value: Whether to compute bootstrap p-value
            alternative: Alternative hypothesis for p-value ('two-sided', 'greater', or 'less')
            **kwargs: Additional keyword arguments for specific bootstrap methods
            
        Returns:
            BootstrapResult: Bootstrap results
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Validate data
        self.validate_data(data)
        
        # Compute original statistic
        original_statistic = statistic_func(data)
        self._original_statistic = original_statistic
        
        # Generate bootstrap indices
        indices = self.generate_indices(
            data_length=len(data),
            n_bootstraps=self.params.n_bootstraps,
            random_state=self.params.random_state
        )
        self._bootstrap_indices = indices
        
        # Compute bootstrap statistics
        bootstrap_statistics = self.compute_statistic(
            data=data,
            indices=indices,
            statistic_func=statistic_func
        )
        self._bootstrap_statistics = bootstrap_statistics
        
        # Compute confidence intervals
        confidence_intervals = self.compute_confidence_interval(
            bootstrap_statistics=bootstrap_statistics,
            confidence_level=confidence_level
        )
        self._confidence_intervals = confidence_intervals
        
        # Compute p-value if requested
        p_value = None
        if compute_p_value:
            p_value = self.compute_p_value(
                bootstrap_statistics=bootstrap_statistics,
                original_statistic=original_statistic,
                alternative=alternative
            )
            self._p_value = p_value
        
        # Mark as fitted
        self._fitted = True
        
        # Create and return result object
        result = BootstrapResult(
            bootstrap_type=self._name,
            original_statistic=original_statistic,
            bootstrap_statistics=bootstrap_statistics,
            confidence_intervals=confidence_intervals,
            p_value=p_value,
            n_bootstraps=self.params.n_bootstraps,
            block_length=self.params.block_length,
            indices=indices
        )
        
        self._results = result
        return result
    
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
        """Run the bootstrap procedure asynchronously.
        
        This method provides an asynchronous interface to the fit method,
        allowing for non-blocking bootstrap execution with progress reporting.
        
        Args:
            data: Data to bootstrap
            statistic_func: Function to compute the statistic of interest
            confidence_level: Confidence level for intervals (between 0 and 1)
            compute_p_value: Whether to compute bootstrap p-value
            alternative: Alternative hypothesis for p-value ('two-sided', 'greater', or 'less')
            progress_callback: Callback function for reporting progress
            **kwargs: Additional keyword arguments for specific bootstrap methods
            
        Returns:
            BootstrapResult: Bootstrap results
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Validate data
        self.validate_data(data)
        
        # Report progress
        if progress_callback:
            progress_callback(0.0, "Starting bootstrap procedure")
        
        # Compute original statistic
        original_statistic = statistic_func(data)
        self._original_statistic = original_statistic
        
        if progress_callback:
            progress_callback(0.1, "Computed original statistic")
        
        # Generate bootstrap indices
        indices = self.generate_indices(
            data_length=len(data),
            n_bootstraps=self.params.n_bootstraps,
            random_state=self.params.random_state
        )
        self._bootstrap_indices = indices
        
        if progress_callback:
            progress_callback(0.2, "Generated bootstrap indices")
        
        # Compute bootstrap statistics with progress reporting
        bootstrap_statistics = []
        n_bootstraps = indices.shape[0]
        
        # Generate bootstrap samples
        bootstrap_samples = self.sample(data, indices)
        
        if progress_callback:
            progress_callback(0.4, "Generated bootstrap samples")
        
        # Compute statistics for each bootstrap sample
        for i, sample in enumerate(bootstrap_samples):
            stat = statistic_func(sample)
            bootstrap_statistics.append(stat)
            
            if progress_callback and i % max(1, n_bootstraps // 10) == 0:
                progress = 0.4 + 0.4 * (i / n_bootstraps)
                progress_callback(
                    progress, 
                    f"Computing bootstrap statistics ({i}/{n_bootstraps})"
                )
        
        bootstrap_statistics = np.array(bootstrap_statistics)
        self._bootstrap_statistics = bootstrap_statistics
        
        if progress_callback:
            progress_callback(0.8, "Computing confidence intervals")
        
        # Compute confidence intervals
        confidence_intervals = self.compute_confidence_interval(
            bootstrap_statistics=bootstrap_statistics,
            confidence_level=confidence_level
        )
        self._confidence_intervals = confidence_intervals
        
        # Compute p-value if requested
        p_value = None
        if compute_p_value:
            if progress_callback:
                progress_callback(0.9, "Computing p-value")
            
            p_value = self.compute_p_value(
                bootstrap_statistics=bootstrap_statistics,
                original_statistic=original_statistic,
                alternative=alternative
            )
            self._p_value = p_value
        
        # Mark as fitted
        self._fitted = True
        
        if progress_callback:
            progress_callback(1.0, "Bootstrap procedure complete")
        
        # Create and return result object
        result = BootstrapResult(
            bootstrap_type=self._name,
            original_statistic=original_statistic,
            bootstrap_statistics=bootstrap_statistics,
            confidence_intervals=confidence_intervals,
            p_value=p_value,
            n_bootstraps=self.params.n_bootstraps,
            block_length=self.params.block_length,
            indices=indices
        )
        
        self._results = result
        return result
    
    def simulate(
        self,
        n_periods: int,
        burn: int = 0,
        initial_values: Optional[np.ndarray] = None,
        random_state: Optional[Union[int, np.random.Generator]] = None,
        **kwargs: Any
    ) -> np.ndarray:
        """Simulate data using the bootstrap method.
        
        This method is implemented to satisfy the ModelBase interface but
        is not typically used for bootstrap methods. It generates simulated
        data by resampling from the original data.
        
        Args:
            n_periods: Number of periods to simulate
            burn: Number of initial observations to discard
            initial_values: Initial values for the simulation
            random_state: Random number generator seed for reproducibility
            **kwargs: Additional keyword arguments
            
        Returns:
            np.ndarray: Simulated data
            
        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self._fitted or self._bootstrap_samples is None:
            raise RuntimeError("Bootstrap has not been run. Call fit() first.")
        
        # Use the random state if provided, otherwise use the one from params
        rng = random_state if random_state is not None else self.params.random_state
        if isinstance(rng, int):
            rng = np.random.default_rng(rng)
        elif rng is None:
            rng = np.random.default_rng()
        
        # Randomly select one of the bootstrap samples
        sample_idx = rng.integers(0, len(self._bootstrap_samples))
        sample = self._bootstrap_samples[sample_idx]
        
        # If the sample is shorter than n_periods, resample with replacement
        if len(sample) < n_periods + burn:
            indices = rng.integers(0, len(sample), size=n_periods + burn)
            simulated_data = sample[indices]
        else:
            # Otherwise, take a random subset
            start_idx = rng.integers(0, len(sample) - (n_periods + burn) + 1)
            simulated_data = sample[start_idx:start_idx + n_periods + burn]
        
        # Discard burn-in period
        if burn > 0:
            simulated_data = simulated_data[burn:]
        
        return simulated_data
    
    def validate_data(self, data: np.ndarray) -> None:
        """Validate input data for bootstrap methods.
        
        Args:
            data: Data to validate
            
        Raises:
            TypeError: If data is not a NumPy array
            ValueError: If data is empty or contains invalid values
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a NumPy array")
        
        if data.size == 0:
            raise ValueError("Data cannot be empty")
        
        if np.isnan(data).any():
            raise DataError(
                "Data contains NaN values",
                data_name="data",
                issue="contains NaN values"
            )
        
        if np.isinf(data).any():
            raise DataError(
                "Data contains infinite values",
                data_name="data",
                issue="contains infinite values"
            )
