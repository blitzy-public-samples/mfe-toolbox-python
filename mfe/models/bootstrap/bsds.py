'''
Bootstrap Data Snooping (BSDS) test implementation for the MFE Toolbox.

This module implements the Bootstrap Data Snooping (BSDS) test, which is used to
compare the performance of multiple models against a benchmark model. The test
computes bootstrap-based p-values for the null hypothesis that the benchmark model
is not inferior to any of the alternative models.

The implementation leverages NumPy's efficient array operations with performance-critical
sections accelerated using Numba's @jit decorators. This approach provides significant
performance improvements while maintaining the flexibility and readability of Python code.

The BSDS test supports both block bootstrap and stationary bootstrap methods, and
implements consistent, upper, and lower p-value calculations. It also provides
visualization capabilities for p-value analysis and supports asynchronous processing
for large-scale comparisons.

References:
    White, H. (2000). A reality check for data snooping.
    Econometrica, 68(5), 1097-1126.

    Hansen, P. R. (2005). A test for superior predictive ability.
    Journal of Business & Economic Statistics, 23(4), 365-380.
''' 

from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Literal, Optional, Tuple, Union, cast, overload
)
import numpy as np
import matplotlib.pyplot as plt
from numba import jit

from mfe.core.exceptions import (
    BootstrapError, ParameterError, DimensionError, DataError, 
    raise_parameter_error, raise_dimension_error, raise_data_error
)
from mfe.core.types import (
    BootstrapIndices, BootstrapSamples, BootstrapStatistic, 
    BootstrapResult, ProgressCallback
)
from mfe.models.bootstrap.base import BootstrapBase, BootstrapParameters
from mfe.models.bootstrap.block_bootstrap import BlockBootstrap
from mfe.models.bootstrap.stationary_bootstrap import StationaryBootstrap


@dataclass
class BSDSParameters(BootstrapParameters):
    """Parameters for Bootstrap Data Snooping test.
    
    This dataclass extends the base BootstrapParameters with additional parameters
    specific to the BSDS test.
    
    Attributes:
        n_bootstraps: Number of bootstrap samples to generate
        block_length: Block length for block bootstrap methods
        bootstrap_method: Method for bootstrap sampling ('block' or 'stationary')
        p_value_type: Type of p-value to compute ('consistent', 'upper', 'lower')
        random_state: Random number generator seed for reproducibility
    """
    
    bootstrap_method: Literal['block', 'stationary'] = 'block'
    p_value_type: Literal['consistent', 'upper', 'lower'] = 'consistent'
    
    def __post_init__(self) -> None:
        """Validate parameters after initialization.
        
        Raises:
            ParameterError: If parameters violate constraints
        """
        # Call parent validation
        super().__post_init__()
        
        # Validate bootstrap_method
        if self.bootstrap_method not in ['block', 'stationary']:
            raise ParameterError(
                "bootstrap_method must be 'block' or 'stationary'",
                param_name="bootstrap_method",
                param_value=self.bootstrap_method
            )
        
        # Validate p_value_type
        if self.p_value_type not in ['consistent', 'upper', 'lower']:
            raise ParameterError(
                "p_value_type must be 'consistent', 'upper', or 'lower'",
                param_name="p_value_type",
                param_value=self.p_value_type
            )


@dataclass
class BSDSResult(BootstrapResult):
    """Result container for Bootstrap Data Snooping test.
    
    This dataclass extends the base BootstrapResult with additional fields
    specific to the BSDS test.
    
    Attributes:
        bootstrap_type: Type of bootstrap method used
        original_statistic: Statistic computed on the original data
        bootstrap_statistics: Statistics computed on bootstrap samples
        p_values: Bootstrap p-values for each model
        confidence_intervals: Confidence intervals for the statistic
        n_bootstraps: Number of bootstrap samples used
        block_length: Block length used (for block bootstrap methods)
        indices: Bootstrap indices used to generate samples
        p_value_type: Type of p-value computed ('consistent', 'upper', 'lower')
        benchmark_losses: Loss values for the benchmark model
        model_losses: Loss values for the alternative models
    """
    
    p_values: np.ndarray
    p_value_type: str = 'consistent'
    benchmark_losses: Optional[np.ndarray] = None
    model_losses: Optional[np.ndarray] = None


@jit(nopython=True, cache=True)
def _compute_bsds_statistics(
    benchmark_losses: np.ndarray,
    model_losses: np.ndarray
) -> np.ndarray:
    """
    Compute BSDS test statistics using Numba acceleration.
    
    This function computes the loss differentials between the benchmark model
    and each alternative model, and returns the maximum loss differential.
    
    Args:
        benchmark_losses: Loss values for the benchmark model
        model_losses: Loss values for the alternative models
        
    Returns:
        np.ndarray: BSDS test statistics
    """
    # Get dimensions
    n_obs = benchmark_losses.shape[0]
    n_models = model_losses.shape[1]
    
    # Compute loss differentials
    loss_diffs = np.zeros((n_obs, n_models))
    for i in range(n_models):
        loss_diffs[:, i] = benchmark_losses - model_losses[:, i]
    
    # Compute mean loss differentials
    mean_loss_diffs = np.mean(loss_diffs, axis=0)
    
    return mean_loss_diffs


@jit(nopython=True, cache=True)
def _compute_bsds_p_values(
    original_stats: np.ndarray,
    bootstrap_stats: np.ndarray,
    p_value_type: str
) -> np.ndarray:
    """
    Compute BSDS p-values using Numba acceleration.
    
    This function computes the p-values for the BSDS test based on the original
    statistics and bootstrap statistics.
    
    Args:
        original_stats: Original test statistics
        bootstrap_stats: Bootstrap test statistics
        p_value_type: Type of p-value to compute ('consistent', 'upper', 'lower')
        
    Returns:
        np.ndarray: BSDS p-values
    """
    n_bootstraps = bootstrap_stats.shape[0]
    n_models = bootstrap_stats.shape[1]
    
    # Initialize p-values
    p_values = np.zeros(n_models)
    
    # Compute p-values based on the specified type
    if p_value_type == 'consistent':
        # Consistent p-values (Hansen, 2005)
        for i in range(n_models):
            p_values[i] = np.mean(bootstrap_stats[:, i] >= original_stats[i])
    elif p_value_type == 'upper':
        # Upper p-values (White, 2000)
        bootstrap_max = np.max(bootstrap_stats, axis=1)
        for i in range(n_models):
            p_values[i] = np.mean(bootstrap_max >= original_stats[i])
    else:  # p_value_type == 'lower'
        # Lower p-values
        for i in range(n_models):
            p_values[i] = np.mean(bootstrap_stats[:, i] >= original_stats[i])
    
    return p_values


class BSDS(BootstrapBase):
    """
    Bootstrap Data Snooping (BSDS) test implementation.
    
    This class implements the Bootstrap Data Snooping test, which is used to
    compare the performance of multiple models against a benchmark model. The test
    computes bootstrap-based p-values for the null hypothesis that the benchmark model
    is not inferior to any of the alternative models.
    
    The implementation supports both block bootstrap and stationary bootstrap methods,
    and implements consistent, upper, and lower p-value calculations. It also provides
    visualization capabilities for p-value analysis and supports asynchronous processing
    for large-scale comparisons.
    
    Attributes:
        params: BSDS test parameters
        _bootstrap_indices: Generated bootstrap indices
        _bootstrap_samples: Generated bootstrap samples
        _bootstrap_statistics: Computed bootstrap statistics
        _original_statistic: Statistic computed on the original data
        _p_values: Computed p-values
        _benchmark_losses: Loss values for the benchmark model
        _model_losses: Loss values for the alternative models
    """
    
    def __init__(
        self,
        block_length: Union[int, float],
        bootstrap_method: Literal['block', 'stationary'] = 'block',
        p_value_type: Literal['consistent', 'upper', 'lower'] = 'consistent',
        n_bootstraps: int = 1000,
        random_state: Optional[Union[int, np.random.Generator]] = None,
        name: str = "Bootstrap Data Snooping"
    ) -> None:
        """
        Initialize the Bootstrap Data Snooping test.
        
        Args:
            block_length: Block length for block bootstrap methods
            bootstrap_method: Method for bootstrap sampling ('block' or 'stationary')
            p_value_type: Type of p-value to compute ('consistent', 'upper', 'lower')
            n_bootstraps: Number of bootstrap samples to generate
            random_state: Random number generator seed for reproducibility
            name: Name of the bootstrap method
            
        Raises:
            ParameterError: If parameters violate constraints
        """
        # Initialize with base parameters
        super().__init__(
            n_bootstraps=n_bootstraps,
            block_length=block_length,
            random_state=random_state,
            name=name
        )
        
        # Create and validate BSDS-specific parameters
        self.params = BSDSParameters(
            n_bootstraps=n_bootstraps,
            block_length=block_length,
            bootstrap_method=bootstrap_method,
            p_value_type=p_value_type,
            random_state=random_state
        )
        
        # Initialize BSDS-specific attributes
        self._p_values: Optional[np.ndarray] = None
        self._benchmark_losses: Optional[np.ndarray] = None
        self._model_losses: Optional[np.ndarray] = None
        
        # Create the appropriate bootstrap method
        if bootstrap_method == 'block':
            self._bootstrap = BlockBootstrap(
                block_length=int(block_length),
                n_bootstraps=n_bootstraps,
                random_state=random_state
            )
        else:  # bootstrap_method == 'stationary'
            self._bootstrap = StationaryBootstrap(
                expected_block_length=float(block_length),
                n_bootstraps=n_bootstraps,
                random_state=random_state
            )
    
    @property
    def p_values(self) -> Optional[np.ndarray]:
        """Get the p-values.
        
        Returns:
            Optional[np.ndarray]: p-values if available, None otherwise
            
        Raises:
            RuntimeError: If BSDS test has not been run
        """
        if not self._fitted:
            raise RuntimeError("BSDS test has not been run. Call fit() first.")
        return self._p_values
    
    @property
    def benchmark_losses(self) -> Optional[np.ndarray]:
        """Get the benchmark losses.
        
        Returns:
            Optional[np.ndarray]: Benchmark losses if available, None otherwise
            
        Raises:
            RuntimeError: If BSDS test has not been run
        """
        if not self._fitted:
            raise RuntimeError("BSDS test has not been run. Call fit() first.")
        return self._benchmark_losses
    
    @property
    def model_losses(self) -> Optional[np.ndarray]:
        """Get the model losses.
        
        Returns:
            Optional[np.ndarray]: Model losses if available, None otherwise
            
        Raises:
            RuntimeError: If BSDS test has not been run
        """
        if not self._fitted:
            raise RuntimeError("BSDS test has not been run. Call fit() first.")
        return self._model_losses
    
    def generate_indices(
        self,
        data_length: int,
        n_bootstraps: int,
        random_state: Optional[Union[int, np.random.Generator]] = None
    ) -> np.ndarray:
        """
        Generate bootstrap indices for BSDS test.
        
        This method delegates the index generation to the appropriate bootstrap method
        (block or stationary) based on the configuration.
        
        Args:
            data_length: Length of the original data
            n_bootstraps: Number of bootstrap samples to generate
            random_state: Random number generator seed for reproducibility
            
        Returns:
            np.ndarray: Bootstrap indices with shape (n_bootstraps, data_length)
            
        Raises:
            BootstrapError: If parameters are invalid or generation fails
        """
        # Delegate to the appropriate bootstrap method
        return self._bootstrap.generate_indices(
            data_length=data_length,
            n_bootstraps=n_bootstraps,
            random_state=random_state
        )
    
    def fit(
        self,
        benchmark_losses: np.ndarray,
        model_losses: np.ndarray,
        confidence_level: float = 0.95,
        **kwargs: Any
    ) -> BSDSResult:
        """
        Run the Bootstrap Data Snooping test.
        
        This method computes the BSDS test statistics and p-values for comparing
        the benchmark model against alternative models.
        
        Args:
            benchmark_losses: Loss values for the benchmark model (shape: n_observations)
            model_losses: Loss values for the alternative models (shape: n_observations x n_models)
            confidence_level: Confidence level for intervals (between 0 and 1)
            **kwargs: Additional keyword arguments
            
        Returns:
            BSDSResult: BSDS test results
            
        Raises:
            ValueError: If inputs are invalid
            DimensionError: If input dimensions are incompatible
        """
        # Validate inputs
        self._validate_inputs(benchmark_losses, model_losses)
        
        # Store the loss values
        self._benchmark_losses = benchmark_losses
        self._model_losses = model_losses
        
        # Compute original test statistics
        original_stats = _compute_bsds_statistics(benchmark_losses, model_losses)
        self._original_statistic = original_stats
        
        # Generate bootstrap indices
        indices = self.generate_indices(
            data_length=len(benchmark_losses),
            n_bootstraps=self.params.n_bootstraps,
            random_state=self.params.random_state
        )
        self._bootstrap_indices = indices
        
        # Compute bootstrap statistics
        bootstrap_stats = self._compute_bootstrap_statistics(
            benchmark_losses=benchmark_losses,
            model_losses=model_losses,
            indices=indices
        )
        self._bootstrap_statistics = bootstrap_stats
        
        # Compute p-values
        p_values = _compute_bsds_p_values(
            original_stats=original_stats,
            bootstrap_stats=bootstrap_stats,
            p_value_type=self.params.p_value_type
        )
        self._p_values = p_values
        
        # Compute confidence intervals
        confidence_intervals = self.compute_confidence_interval(
            bootstrap_statistics=bootstrap_stats,
            confidence_level=confidence_level
        )
        self._confidence_intervals = confidence_intervals
        
        # Mark as fitted
        self._fitted = True
        
        # Create and return result object
        result = BSDSResult(
            bootstrap_type=self._name,
            original_statistic=original_stats,
            bootstrap_statistics=bootstrap_stats,
            p_values=p_values,
            confidence_intervals=confidence_intervals,
            n_bootstraps=self.params.n_bootstraps,
            block_length=self.params.block_length,
            indices=indices,
            p_value_type=self.params.p_value_type,
            benchmark_losses=benchmark_losses,
            model_losses=model_losses
        )
        
        self._results = result
        return result
    
    async def fit_async(
        self,
        benchmark_losses: np.ndarray,
        model_losses: np.ndarray,
        confidence_level: float = 0.95,
        progress_callback: Optional[ProgressCallback] = None,
        **kwargs: Any
    ) -> BSDSResult:
        """
        Run the Bootstrap Data Snooping test asynchronously.
        
        This method provides an asynchronous interface to the fit method,
        allowing for non-blocking bootstrap execution with progress reporting.
        
        Args:
            benchmark_losses: Loss values for the benchmark model (shape: n_observations)
            model_losses: Loss values for the alternative models (shape: n_observations x n_models)
            confidence_level: Confidence level for intervals (between 0 and 1)
            progress_callback: Callback function for reporting progress
            **kwargs: Additional keyword arguments
            
        Returns:
            BSDSResult: BSDS test results
            
        Raises:
            ValueError: If inputs are invalid
            DimensionError: If input dimensions are incompatible
        """
        # Validate inputs
        self._validate_inputs(benchmark_losses, model_losses)
        
        # Report progress
        if progress_callback:
            progress_callback(0.0, "Starting BSDS test")
        
        # Store the loss values
        self._benchmark_losses = benchmark_losses
        self._model_losses = model_losses
        
        # Compute original test statistics
        if progress_callback:
            progress_callback(0.1, "Computing original test statistics")
        
        original_stats = _compute_bsds_statistics(benchmark_losses, model_losses)
        self._original_statistic = original_stats
        
        # Generate bootstrap indices
        if progress_callback:
            progress_callback(0.2, "Generating bootstrap indices")
        
        indices = self.generate_indices(
            data_length=len(benchmark_losses),
            n_bootstraps=self.params.n_bootstraps,
            random_state=self.params.random_state
        )
        self._bootstrap_indices = indices
        
        # Compute bootstrap statistics
        if progress_callback:
            progress_callback(0.3, "Computing bootstrap statistics")
        
        bootstrap_stats = self._compute_bootstrap_statistics(
            benchmark_losses=benchmark_losses,
            model_losses=model_losses,
            indices=indices
        )
        self._bootstrap_statistics = bootstrap_stats
        
        # Compute p-values
        if progress_callback:
            progress_callback(0.8, "Computing p-values")
        
        p_values = _compute_bsds_p_values(
            original_stats=original_stats,
            bootstrap_stats=bootstrap_stats,
            p_value_type=self.params.p_value_type
        )
        self._p_values = p_values
        
        # Compute confidence intervals
        if progress_callback:
            progress_callback(0.9, "Computing confidence intervals")
        
        confidence_intervals = self.compute_confidence_interval(
            bootstrap_statistics=bootstrap_stats,
            confidence_level=confidence_level
        )
        self._confidence_intervals = confidence_intervals
        
        # Mark as fitted
        self._fitted = True
        
        if progress_callback:
            progress_callback(1.0, "BSDS test complete")
        
        # Create and return result object
        result = BSDSResult(
            bootstrap_type=self._name,
            original_statistic=original_stats,
            bootstrap_statistics=bootstrap_stats,
            p_values=p_values,
            confidence_intervals=confidence_intervals,
            n_bootstraps=self.params.n_bootstraps,
            block_length=self.params.block_length,
            indices=indices,
            p_value_type=self.params.p_value_type,
            benchmark_losses=benchmark_losses,
            model_losses=model_losses
        )
        
        self._results = result
        return result
    
    def _compute_bootstrap_statistics(
        self,
        benchmark_losses: np.ndarray,
        model_losses: np.ndarray,
        indices: np.ndarray
    ) -> np.ndarray:
        """
        Compute bootstrap statistics for BSDS test.
        
        This method computes the bootstrap statistics by resampling the loss values
        using the provided bootstrap indices.
        
        Args:
            benchmark_losses: Loss values for the benchmark model
            model_losses: Loss values for the alternative models
            indices: Bootstrap indices
            
        Returns:
            np.ndarray: Bootstrap statistics
        """
        n_bootstraps = indices.shape[0]
        n_models = model_losses.shape[1]
        
        # Initialize bootstrap statistics
        bootstrap_stats = np.zeros((n_bootstraps, n_models))
        
        # Compute bootstrap statistics for each bootstrap sample
        for i in range(n_bootstraps):
            # Resample benchmark losses
            benchmark_sample = benchmark_losses[indices[i]]
            
            # Resample model losses
            model_sample = model_losses[indices[i]]
            
            # Compute statistics for this bootstrap sample
            bootstrap_stats[i] = _compute_bsds_statistics(benchmark_sample, model_sample)
        
        return bootstrap_stats
    
    def _validate_inputs(
        self,
        benchmark_losses: np.ndarray,
        model_losses: np.ndarray
    ) -> None:
        """
        Validate inputs for BSDS test.
        
        Args:
            benchmark_losses: Loss values for the benchmark model
            model_losses: Loss values for the alternative models
            
        Raises:
            TypeError: If inputs are not NumPy arrays
            DimensionError: If input dimensions are incompatible
            ValueError: If inputs contain invalid values
        """
        # Check types
        if not isinstance(benchmark_losses, np.ndarray):
            raise TypeError("benchmark_losses must be a NumPy array")
        
        if not isinstance(model_losses, np.ndarray):
            raise TypeError("model_losses must be a NumPy array")
        
        # Check dimensions
        if benchmark_losses.ndim != 1:
            raise DimensionError(
                "benchmark_losses must be a 1-dimensional array",
                array_name="benchmark_losses",
                expected_shape="(n_observations,)",
                actual_shape=benchmark_losses.shape
            )
        
        if model_losses.ndim != 2:
            raise DimensionError(
                "model_losses must be a 2-dimensional array",
                array_name="model_losses",
                expected_shape="(n_observations, n_models)",
                actual_shape=model_losses.shape
            )
        
        # Check compatibility
        if len(benchmark_losses) != model_losses.shape[0]:
            raise DimensionError(
                "benchmark_losses and model_losses must have the same number of observations",
                array_name="model_losses",
                expected_shape=f"({len(benchmark_losses)}, n_models)",
                actual_shape=model_losses.shape
            )
        
        # Check for invalid values
        if np.isnan(benchmark_losses).any():
            raise DataError(
                "benchmark_losses contains NaN values",
                data_name="benchmark_losses",
                issue="contains NaN values"
            )
        
        if np.isnan(model_losses).any():
            raise DataError(
                "model_losses contains NaN values",
                data_name="model_losses",
                issue="contains NaN values"
            )
        
        if np.isinf(benchmark_losses).any():
            raise DataError(
                "benchmark_losses contains infinite values",
                data_name="benchmark_losses",
                issue="contains infinite values"
            )
        
        if np.isinf(model_losses).any():
            raise DataError(
                "model_losses contains infinite values",
                data_name="model_losses",
                issue="contains infinite values"
            )
    
    def plot_p_values(
        self,
        model_names: Optional[List[str]] = None,
        significance_level: float = 0.05,
        figsize: Tuple[int, int] = (10, 6),
        title: Optional[str] = None,
        sort: bool = True
    ) -> plt.Figure:
        """
        Plot p-values from the BSDS test.
        
        This method creates a bar plot of the p-values from the BSDS test,
        with a horizontal line indicating the significance level.
        
        Args:
            model_names: Names of the models (if None, uses indices)
            significance_level: Significance level for the horizontal line
            figsize: Figure size (width, height)
            title: Plot title (if None, uses default)
            sort: Whether to sort p-values in ascending order
            
        Returns:
            plt.Figure: Matplotlib figure object
            
        Raises:
            RuntimeError: If BSDS test has not been run
        """
        if not self._fitted or self._p_values is None:
            raise RuntimeError("BSDS test has not been run. Call fit() first.")
        
        # Get p-values
        p_values = self._p_values
        n_models = len(p_values)
        
        # Create model names if not provided
        if model_names is None:
            model_names = [f"Model {i+1}" for i in range(n_models)]
        
        # Sort p-values if requested
        if sort:
            # Sort p-values and model names
            sorted_indices = np.argsort(p_values)
            p_values = p_values[sorted_indices]
            model_names = [model_names[i] for i in sorted_indices]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot p-values
        bars = ax.bar(model_names, p_values, color='skyblue')
        
        # Add horizontal line for significance level
        ax.axhline(y=significance_level, color='red', linestyle='--', 
                   label=f'Significance Level ({significance_level})')
        
        # Highlight significant models
        for i, bar in enumerate(bars):
            if p_values[i] < significance_level:
                bar.set_color('green')
        
        # Add labels and title
        ax.set_xlabel('Models')
        ax.set_ylabel('p-values')
        if title is None:
            title = f'BSDS Test p-values ({self.params.p_value_type})'
        ax.set_title(title)
        
        # Add legend
        ax.legend()
        
        # Rotate x-axis labels if there are many models
        if n_models > 5:
            plt.xticks(rotation=45, ha='right')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def summary(self) -> str:
        """
        Generate a summary of the BSDS test results.
        
        Returns:
            str: Summary of the BSDS test results
            
        Raises:
            RuntimeError: If BSDS test has not been run
        """
        if not self._fitted or self._p_values is None:
            raise RuntimeError("BSDS test has not been run. Call fit() first.")
        
        # Get results
        p_values = self._p_values
        n_models = len(p_values)
        
        # Create summary
        summary = [
            "Bootstrap Data Snooping (BSDS) Test Results",
            "=" * 50,
            f"Bootstrap Method: {self.params.bootstrap_method}",
            f"Block Length: {self.params.block_length}",
            f"Number of Bootstraps: {self.params.n_bootstraps}",
            f"p-value Type: {self.params.p_value_type}",
            "-" * 50,
            "p-values:",
        ]
        
        # Add p-values
        for i in range(n_models):
            summary.append(f"  Model {i+1}: {p_values[i]:.4f}")
        
        # Add significance indicators
        summary.append("-" * 50)
        summary.append("Significance at 10%, 5%, and 1% levels:")
        
        for i in range(n_models):
            sig_indicators = []
            if p_values[i] < 0.1:
                sig_indicators.append("*")
            if p_values[i] < 0.05:
                sig_indicators.append("*")
            if p_values[i] < 0.01:
                sig_indicators.append("*")
            
            sig_str = "".join(sig_indicators)
            summary.append(f"  Model {i+1}: {sig_str}")
        
        summary.append("=" * 50)
        summary.append("* p < 0.1, ** p < 0.05, *** p < 0.01")
        
        return "\n".join(summary)
    
    def __str__(self) -> str:
        """Return a string representation of the BSDS instance."""
        if self._fitted:
            return (
                f"BSDS(bootstrap_method={self.params.bootstrap_method}, "
                f"block_length={self.params.block_length}, "
                f"p_value_type={self.params.p_value_type}, "
                f"n_bootstraps={self.params.n_bootstraps}, fitted=True)"
            )
        else:
            return (
                f"BSDS(bootstrap_method={self.params.bootstrap_method}, "
                f"block_length={self.params.block_length}, "
                f"p_value_type={self.params.p_value_type}, "
                f"n_bootstraps={self.params.n_bootstraps}, fitted=False)"
            )
    
    def __repr__(self) -> str:
        """Return a detailed string representation of the BSDS instance."""
        return (
            f"BSDS(bootstrap_method={self.params.bootstrap_method}, "
            f"block_length={self.params.block_length}, "
            f"p_value_type={self.params.p_value_type}, "
            f"n_bootstraps={self.params.n_bootstraps}, "
            f"random_state={self.params.random_state}, "
            f"fitted={self._fitted})"
        )
