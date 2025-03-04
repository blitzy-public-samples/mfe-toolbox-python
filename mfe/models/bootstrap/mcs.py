'''
Model Confidence Set (MCS) implementation for the MFE Toolbox.

This module implements the Model Confidence Set (MCS) procedure proposed by
Hansen, Lunde, and Nason (2011) for selecting statistically superior models.
The MCS procedure identifies a set of models that contains all models that are
statistically indistinguishable from the best model at a specified confidence level.

The implementation performs iterative testing to eliminate inferior models and
construct a set of models that contains all superior models with a specified
confidence level. It supports different bootstrap methods (block and stationary)
and test statistics (T_max and T_R).

The implementation leverages NumPy's efficient array operations with
performance-critical sections accelerated using Numba's @jit decorators.
This approach provides significant performance improvements while maintaining
the flexibility and readability of Python code.

References:
    Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence set.
    Econometrica, 79(2), 453-497.
'''

from dataclasses import dataclass, field
from typing import (
    Dict, List, Optional, Tuple, Union, Any, Callable, Set, cast, 
    Literal, ClassVar, TypeVar, overload
)
import logging
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import asyncio

from mfe.core.base import ModelBase
from mfe.core.exceptions import (
    ParameterError, DimensionError, BootstrapError, 
    raise_parameter_error, raise_dimension_error
)
from mfe.core.types import ProgressCallback
from mfe.models.bootstrap.base import BootstrapBase
from mfe.models.bootstrap.block_bootstrap import BlockBootstrap
from mfe.models.bootstrap.stationary_bootstrap import StationaryBootstrap

# Set up module-level logger
logger = logging.getLogger("mfe.models.bootstrap.mcs")


@dataclass
class MCSParameters:
    """Parameters for the Model Confidence Set procedure.
    
    This dataclass encapsulates the parameters used in the Model Confidence Set
    procedure, providing type validation and consistent parameter handling.
    
    Attributes:
        alpha: Significance level for the MCS (default: 0.05)
        bootstrap_method: Method for bootstrap ('block' or 'stationary')
        block_length: Block length for block bootstrap or expected block length for stationary bootstrap
        n_bootstraps: Number of bootstrap replications (default: 1000)
        test_statistic: Test statistic to use ('T_max' or 'T_R')
        random_state: Random number generator seed for reproducibility
    """
    
    alpha: float = 0.05
    bootstrap_method: Literal["block", "stationary"] = "stationary"
    block_length: Optional[float] = None
    n_bootstraps: int = 1000
    test_statistic: Literal["T_max", "T_R"] = "T_max"
    random_state: Optional[Union[int, np.random.Generator]] = None
    
    def __post_init__(self) -> None:
        """Validate parameters after initialization.
        
        Raises:
            ParameterError: If parameters violate constraints
        """
        # Validate alpha
        if not isinstance(self.alpha, (int, float)):
            raise ParameterError(
                "alpha must be a number",
                param_name="alpha",
                param_value=self.alpha
            )
        if not 0 < self.alpha < 1:
            raise ParameterError(
                "alpha must be between 0 and 1",
                param_name="alpha",
                param_value=self.alpha
            )
        
        # Validate bootstrap_method
        if self.bootstrap_method not in ["block", "stationary"]:
            raise ParameterError(
                "bootstrap_method must be 'block' or 'stationary'",
                param_name="bootstrap_method",
                param_value=self.bootstrap_method
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
        
        # Validate test_statistic
        if self.test_statistic not in ["T_max", "T_R"]:
            raise ParameterError(
                "test_statistic must be 'T_max' or 'T_R'",
                param_name="test_statistic",
                param_value=self.test_statistic
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
class MCSResult:
    """Result container for the Model Confidence Set procedure.
    
    This dataclass encapsulates the results of the Model Confidence Set procedure,
    providing a consistent structure for accessing MCS results.
    
    Attributes:
        included_models: Indices of models in the MCS
        pvalues: MCS p-values for each model
        eliminated_order: Order in which models were eliminated
        test_statistics: Test statistics for each elimination step
        alpha: Significance level used for the MCS
        bootstrap_method: Bootstrap method used
        n_bootstraps: Number of bootstrap replications used
        block_length: Block length used (for block bootstrap methods)
    """
    
    included_models: np.ndarray
    pvalues: np.ndarray
    eliminated_order: np.ndarray
    test_statistics: np.ndarray
    alpha: float
    bootstrap_method: str
    n_bootstraps: int
    block_length: Optional[float] = None
    
    def __post_init__(self) -> None:
        """Validate result object after initialization."""
        # Ensure arrays are numpy arrays
        self.included_models = np.asarray(self.included_models)
        self.pvalues = np.asarray(self.pvalues)
        self.eliminated_order = np.asarray(self.eliminated_order)
        self.test_statistics = np.asarray(self.test_statistics)


@jit(nopython=True, cache=True)
def _compute_loss_differentials(losses: np.ndarray) -> np.ndarray:
    """
    Compute pairwise loss differentials for all models.
    
    This function computes the pairwise loss differentials between all models
    for each time period. It is accelerated using Numba's @jit decorator for
    improved performance on large datasets.
    
    Args:
        losses: Loss matrix of shape (T, M) where T is the number of time periods
               and M is the number of models
        
    Returns:
        np.ndarray: Loss differential matrix of shape (T, M, M)
    """
    T, M = losses.shape
    loss_diffs = np.zeros((T, M, M))
    
    for i in range(M):
        for j in range(M):
            if i != j:
                loss_diffs[:, i, j] = losses[:, i] - losses[:, j]
    
    return loss_diffs


@jit(nopython=True, cache=True)
def _compute_t_max_statistic(loss_diffs: np.ndarray, model_indices: np.ndarray) -> Tuple[float, int, np.ndarray]:
    """
    Compute the T_max test statistic for the MCS procedure.
    
    This function computes the T_max test statistic, which is the maximum t-statistic
    across all models in the current set. It is accelerated using Numba's @jit decorator
    for improved performance.
    
    Args:
        loss_diffs: Loss differential matrix of shape (T, M, M)
        model_indices: Indices of models in the current set
        
    Returns:
        Tuple containing:
        - float: The T_max test statistic
        - int: Index of the model with the worst performance
        - np.ndarray: Array of t-statistics for each model
    """
    T = loss_diffs.shape[0]
    n_models = len(model_indices)
    
    # Initialize arrays for mean loss differentials and variances
    mean_diffs = np.zeros(n_models)
    t_stats = np.zeros(n_models)
    
    # Compute mean loss differential for each model
    for i in range(n_models):
        model_idx = model_indices[i]
        # Compute mean loss differential against all other models in the set
        diff_sum = 0.0
        count = 0
        for j in range(n_models):
            if i != j:
                model_j_idx = model_indices[j]
                diff_sum += np.mean(loss_diffs[:, model_idx, model_j_idx])
                count += 1
        
        if count > 0:
            mean_diffs[i] = diff_sum / count
        
        # Compute variance of loss differentials
        var_sum = 0.0
        count = 0
        for j in range(n_models):
            if i != j:
                model_j_idx = model_indices[j]
                diffs = loss_diffs[:, model_idx, model_j_idx]
                var_sum += np.var(diffs) * (T - 1) / T  # Unbiased estimator
                count += 1
        
        # Compute t-statistic
        if count > 0 and var_sum > 0:
            t_stats[i] = mean_diffs[i] / np.sqrt(var_sum / (count * T))
        else:
            t_stats[i] = 0.0
    
    # Find the maximum t-statistic and corresponding model index
    max_t_idx = np.argmax(t_stats)
    max_t = t_stats[max_t_idx]
    worst_model_idx = model_indices[max_t_idx]
    
    return max_t, worst_model_idx, t_stats


@jit(nopython=True, cache=True)
def _compute_t_r_statistic(loss_diffs: np.ndarray, model_indices: np.ndarray) -> Tuple[float, int, np.ndarray]:
    """
    Compute the T_R test statistic for the MCS procedure.
    
    This function computes the T_R test statistic, which is the sum of squared
    t-statistics across all models in the current set. It is accelerated using
    Numba's @jit decorator for improved performance.
    
    Args:
        loss_diffs: Loss differential matrix of shape (T, M, M)
        model_indices: Indices of models in the current set
        
    Returns:
        Tuple containing:
        - float: The T_R test statistic
        - int: Index of the model with the worst performance
        - np.ndarray: Array of t-statistics for each model
    """
    T = loss_diffs.shape[0]
    n_models = len(model_indices)
    
    # Initialize arrays for mean loss differentials and variances
    mean_diffs = np.zeros(n_models)
    t_stats = np.zeros(n_models)
    
    # Compute mean loss differential for each model
    for i in range(n_models):
        model_idx = model_indices[i]
        # Compute mean loss differential against all other models in the set
        diff_sum = 0.0
        count = 0
        for j in range(n_models):
            if i != j:
                model_j_idx = model_indices[j]
                diff_sum += np.mean(loss_diffs[:, model_idx, model_j_idx])
                count += 1
        
        if count > 0:
            mean_diffs[i] = diff_sum / count
        
        # Compute variance of loss differentials
        var_sum = 0.0
        count = 0
        for j in range(n_models):
            if i != j:
                model_j_idx = model_indices[j]
                diffs = loss_diffs[:, model_idx, model_j_idx]
                var_sum += np.var(diffs) * (T - 1) / T  # Unbiased estimator
                count += 1
        
        # Compute t-statistic
        if count > 0 and var_sum > 0:
            t_stats[i] = mean_diffs[i] / np.sqrt(var_sum / (count * T))
        else:
            t_stats[i] = 0.0
    
    # Compute T_R statistic (sum of squared t-statistics)
    t_r = np.sum(t_stats**2)
    
    # Find the model with the highest individual t-statistic
    max_t_idx = np.argmax(t_stats)
    worst_model_idx = model_indices[max_t_idx]
    
    return t_r, worst_model_idx, t_stats


@jit(nopython=True, cache=True)
def _bootstrap_test_distribution(
    loss_diffs: np.ndarray,
    model_indices: np.ndarray,
    bootstrap_indices: np.ndarray,
    use_t_max: bool = True
) -> np.ndarray:
    """
    Compute the bootstrap distribution of the test statistic.
    
    This function computes the bootstrap distribution of either the T_max or T_R
    test statistic. It is accelerated using Numba's @jit decorator for improved
    performance.
    
    Args:
        loss_diffs: Loss differential matrix of shape (T, M, M)
        model_indices: Indices of models in the current set
        bootstrap_indices: Bootstrap indices of shape (n_bootstraps, T)
        use_t_max: Whether to use the T_max statistic (True) or T_R statistic (False)
        
    Returns:
        np.ndarray: Bootstrap distribution of the test statistic
    """
    T = loss_diffs.shape[0]
    n_bootstraps = bootstrap_indices.shape[0]
    n_models = len(model_indices)
    
    # Initialize array for bootstrap test statistics
    bootstrap_stats = np.zeros(n_bootstraps)
    
    # Compute bootstrap test statistics
    for b in range(n_bootstraps):
        # Get bootstrap indices for this replication
        indices = bootstrap_indices[b]
        
        # Initialize arrays for mean loss differentials and variances
        mean_diffs = np.zeros(n_models)
        t_stats = np.zeros(n_models)
        
        # Compute mean loss differential for each model
        for i in range(n_models):
            model_idx = model_indices[i]
            # Compute mean loss differential against all other models in the set
            diff_sum = 0.0
            count = 0
            for j in range(n_models):
                if i != j:
                    model_j_idx = model_indices[j]
                    # Use bootstrap indices to compute mean
                    diff_sum += np.mean(loss_diffs[indices, model_idx, model_j_idx])
                    count += 1
            
            if count > 0:
                mean_diffs[i] = diff_sum / count
            
            # Compute variance of loss differentials
            var_sum = 0.0
            count = 0
            for j in range(n_models):
                if i != j:
                    model_j_idx = model_indices[j]
                    # Use bootstrap indices to compute variance
                    diffs = loss_diffs[indices, model_idx, model_j_idx]
                    var_sum += np.var(diffs) * (T - 1) / T  # Unbiased estimator
                    count += 1
            
            # Compute t-statistic
            if count > 0 and var_sum > 0:
                t_stats[i] = mean_diffs[i] / np.sqrt(var_sum / (count * T))
            else:
                t_stats[i] = 0.0
        
        # Compute test statistic based on the specified type
        if use_t_max:
            # T_max is the maximum t-statistic
            bootstrap_stats[b] = np.max(t_stats)
        else:
            # T_R is the sum of squared t-statistics
            bootstrap_stats[b] = np.sum(t_stats**2)
    
    return bootstrap_stats


class ModelConfidenceSet(ModelBase):
    """
    Model Confidence Set (MCS) implementation.
    
    This class implements the Model Confidence Set procedure proposed by Hansen,
    Lunde, and Nason (2011) for selecting statistically superior models. The MCS
    procedure identifies a set of models that contains all models that are
    statistically indistinguishable from the best model at a specified confidence level.
    
    The implementation performs iterative testing to eliminate inferior models and
    construct a set of models that contains all superior models with a specified
    confidence level. It supports different bootstrap methods (block and stationary)
    and test statistics (T_max and T_R).
    """
    
    def __init__(
        self,
        losses: np.ndarray,
        alpha: float = 0.05,
        bootstrap_method: Literal["block", "stationary"] = "stationary",
        block_length: Optional[float] = None,
        n_bootstraps: int = 1000,
        test_statistic: Literal["T_max", "T_R"] = "T_max",
        random_state: Optional[Union[int, np.random.Generator]] = None,
        name: str = "Model Confidence Set"
    ) -> None:
        """
        Initialize the Model Confidence Set.
        
        Args:
            losses: Loss matrix of shape (T, M) where T is the number of time periods
                   and M is the number of models
            alpha: Significance level for the MCS (default: 0.05)
            bootstrap_method: Method for bootstrap ('block' or 'stationary')
            block_length: Block length for block bootstrap or expected block length for stationary bootstrap
            n_bootstraps: Number of bootstrap replications (default: 1000)
            test_statistic: Test statistic to use ('T_max' or 'T_R')
            random_state: Random number generator seed for reproducibility
            name: Name of the model
            
        Raises:
            ParameterError: If parameters violate constraints
            DimensionError: If losses has invalid dimensions
        """
        super().__init__(name=name)
        
        # Create and validate parameters
        self.params = MCSParameters(
            alpha=alpha,
            bootstrap_method=bootstrap_method,
            block_length=block_length,
            n_bootstraps=n_bootstraps,
            test_statistic=test_statistic,
            random_state=random_state
        )
        
        # Validate losses
        self._validate_losses(losses)
        
        # Store losses and compute loss differentials
        self._losses = losses
        self._loss_diffs = _compute_loss_differentials(losses)
        
        # Initialize bootstrap object
        self._bootstrap = self._create_bootstrap()
        
        # Initialize result attributes
        self._included_models: Optional[np.ndarray] = None
        self._pvalues: Optional[np.ndarray] = None
        self._eliminated_order: Optional[np.ndarray] = None
        self._test_statistics: Optional[np.ndarray] = None
    
    def _validate_losses(self, losses: np.ndarray) -> None:
        """
        Validate the loss matrix.
        
        Args:
            losses: Loss matrix to validate
            
        Raises:
            TypeError: If losses is not a NumPy array
            DimensionError: If losses has invalid dimensions
            ValueError: If losses contains invalid values
        """
        if not isinstance(losses, np.ndarray):
            raise TypeError("losses must be a NumPy array")
        
        if losses.ndim != 2:
            raise DimensionError(
                "losses must be a 2-dimensional array",
                array_name="losses",
                expected_shape="(T, M)",
                actual_shape=losses.shape
            )
        
        if losses.shape[0] < 2:
            raise DimensionError(
                "losses must have at least 2 time periods",
                array_name="losses",
                expected_shape="(T>=2, M)",
                actual_shape=losses.shape
            )
        
        if losses.shape[1] < 2:
            raise DimensionError(
                "losses must have at least 2 models",
                array_name="losses",
                expected_shape="(T, M>=2)",
                actual_shape=losses.shape
            )
        
        if np.isnan(losses).any():
            raise ValueError("losses contains NaN values")
        
        if np.isinf(losses).any():
            raise ValueError("losses contains infinite values")
    
    def _create_bootstrap(self) -> BootstrapBase:
        """
        Create a bootstrap object based on the specified method.
        
        Returns:
            BootstrapBase: Bootstrap object for generating bootstrap samples
            
        Raises:
            ParameterError: If bootstrap_method is invalid
        """
        # Determine block length if not provided
        if self.params.block_length is None:
            # Implement automatic block length selection
            # A common rule of thumb is to use n^(1/3) where n is the sample size
            T = self._losses.shape[0]
            block_length = int(T ** (1/3))
            logger.info(f"Automatically selected block length: {block_length}")
        else:
            block_length = self.params.block_length
        
        # Create the appropriate bootstrap instance
        if self.params.bootstrap_method == "block":
            return BlockBootstrap(
                block_length=int(block_length),
                n_bootstraps=self.params.n_bootstraps,
                random_state=self.params.random_state
            )
        elif self.params.bootstrap_method == "stationary":
            return StationaryBootstrap(
                expected_block_length=float(block_length),
                n_bootstraps=self.params.n_bootstraps,
                random_state=self.params.random_state
            )
        else:
            raise ParameterError(
                f"Invalid bootstrap_method: {self.params.bootstrap_method}. "
                f"Must be one of: 'block' or 'stationary'.",
                param_name="bootstrap_method",
                param_value=self.params.bootstrap_method
            )
    
    def compute(self) -> MCSResult:
        """
        Compute the Model Confidence Set.
        
        This method performs the iterative testing procedure to construct the
        Model Confidence Set. It eliminates inferior models one by one until
        all remaining models are statistically indistinguishable from the best model.
        
        Returns:
            MCSResult: Results of the MCS procedure
        
        Raises:
            BootstrapError: If bootstrap procedure fails
        """
        T, M = self._losses.shape
        
        # Generate bootstrap indices
        bootstrap_indices = self._bootstrap.generate_indices(
            data_length=T,
            n_bootstraps=self.params.n_bootstraps,
            random_state=self.params.random_state
        )
        
        # Initialize arrays for results
        pvalues = np.ones(M)
        eliminated_order = np.zeros(M, dtype=int)
        test_statistics = np.zeros(M)
        
        # Initialize set of models
        current_models = np.arange(M)
        
        # Iteratively eliminate models
        for i in range(M - 1):
            n_current = len(current_models)
            
            # Compute test statistic
            if self.params.test_statistic == "T_max":
                test_stat, worst_model, t_stats = _compute_t_max_statistic(
                    self._loss_diffs, current_models
                )
                use_t_max = True
            else:  # T_R
                test_stat, worst_model, t_stats = _compute_t_r_statistic(
                    self._loss_diffs, current_models
                )
                use_t_max = False
            
            # Compute bootstrap distribution of the test statistic
            bootstrap_stats = _bootstrap_test_distribution(
                self._loss_diffs, current_models, bootstrap_indices, use_t_max
            )
            
            # Compute p-value
            pvalue = np.mean(bootstrap_stats >= test_stat)
            
            # Store results
            pvalues[worst_model] = pvalue
            eliminated_order[i] = worst_model
            test_statistics[i] = test_stat
            
            # Check if we should stop elimination
            if pvalue >= self.params.alpha:
                # All remaining models are in the MCS
                remaining_models = np.setdiff1d(current_models, [worst_model])
                eliminated_order[i+1:] = remaining_models
                break
            
            # Remove worst model from current set
            current_models = np.setdiff1d(current_models, [worst_model])
            
            # If only one model remains, it's automatically in the MCS
            if len(current_models) == 1:
                eliminated_order[i+1] = current_models[0]
                break
        
        # Determine included models (those with p-value >= alpha)
        included_models = np.where(pvalues >= self.params.alpha)[0]
        
        # Store results
        self._included_models = included_models
        self._pvalues = pvalues
        self._eliminated_order = eliminated_order
        self._test_statistics = test_statistics
        
        # Mark as fitted
        self._fitted = True
        
        # Create and return result object
        result = MCSResult(
            included_models=included_models,
            pvalues=pvalues,
            eliminated_order=eliminated_order,
            test_statistics=test_statistics,
            alpha=self.params.alpha,
            bootstrap_method=self.params.bootstrap_method,
            n_bootstraps=self.params.n_bootstraps,
            block_length=self.params.block_length
        )
        
        self._results = result
        return result
    
    async def compute_async(
        self,
        progress_callback: Optional[ProgressCallback] = None
    ) -> MCSResult:
        """
        Compute the Model Confidence Set asynchronously.
        
        This method provides an asynchronous interface to the compute method,
        allowing for non-blocking MCS computation with progress reporting.
        
        Args:
            progress_callback: Callback function for reporting progress
        
        Returns:
            MCSResult: Results of the MCS procedure
        
        Raises:
            BootstrapError: If bootstrap procedure fails
        """
        T, M = self._losses.shape
        
        # Report progress
        if progress_callback:
            progress_callback(0.0, "Starting Model Confidence Set procedure")
        
        # Generate bootstrap indices
        bootstrap_indices = self._bootstrap.generate_indices(
            data_length=T,
            n_bootstraps=self.params.n_bootstraps,
            random_state=self.params.random_state
        )
        
        if progress_callback:
            progress_callback(0.1, "Generated bootstrap indices")
        
        # Initialize arrays for results
        pvalues = np.ones(M)
        eliminated_order = np.zeros(M, dtype=int)
        test_statistics = np.zeros(M)
        
        # Initialize set of models
        current_models = np.arange(M)
        
        # Iteratively eliminate models
        for i in range(M - 1):
            if progress_callback:
                progress = 0.1 + 0.8 * (i / (M - 1))
                progress_callback(
                    progress, 
                    f"Elimination step {i+1}/{M-1}: {len(current_models)} models remaining"
                )
            
            n_current = len(current_models)
            
            # Compute test statistic
            if self.params.test_statistic == "T_max":
                test_stat, worst_model, t_stats = _compute_t_max_statistic(
                    self._loss_diffs, current_models
                )
                use_t_max = True
            else:  # T_R
                test_stat, worst_model, t_stats = _compute_t_r_statistic(
                    self._loss_diffs, current_models
                )
                use_t_max = False
            
            # Compute bootstrap distribution of the test statistic
            bootstrap_stats = _bootstrap_test_distribution(
                self._loss_diffs, current_models, bootstrap_indices, use_t_max
            )
            
            # Compute p-value
            pvalue = np.mean(bootstrap_stats >= test_stat)
            
            # Store results
            pvalues[worst_model] = pvalue
            eliminated_order[i] = worst_model
            test_statistics[i] = test_stat
            
            # Check if we should stop elimination
            if pvalue >= self.params.alpha:
                # All remaining models are in the MCS
                remaining_models = np.setdiff1d(current_models, [worst_model])
                eliminated_order[i+1:] = remaining_models
                break
            
            # Remove worst model from current set
            current_models = np.setdiff1d(current_models, [worst_model])
            
            # If only one model remains, it's automatically in the MCS
            if len(current_models) == 1:
                eliminated_order[i+1] = current_models[0]
                break
            
            # Allow other tasks to run
            await asyncio.sleep(0)
        
        # Determine included models (those with p-value >= alpha)
        included_models = np.where(pvalues >= self.params.alpha)[0]
        
        if progress_callback:
            progress_callback(0.9, f"Identified {len(included_models)} models in the MCS")
        
        # Store results
        self._included_models = included_models
        self._pvalues = pvalues
        self._eliminated_order = eliminated_order
        self._test_statistics = test_statistics
        
        # Mark as fitted
        self._fitted = True
        
        if progress_callback:
            progress_callback(1.0, "Model Confidence Set procedure complete")
        
        # Create and return result object
        result = MCSResult(
            included_models=included_models,
            pvalues=pvalues,
            eliminated_order=eliminated_order,
            test_statistics=test_statistics,
            alpha=self.params.alpha,
            bootstrap_method=self.params.bootstrap_method,
            n_bootstraps=self.params.n_bootstraps,
            block_length=self.params.block_length
        )
        
        self._results = result
        return result
    
    @property
    def included_models(self) -> np.ndarray:
        """
        Get the indices of models in the MCS.
        
        Returns:
            np.ndarray: Indices of models in the MCS
            
        Raises:
            RuntimeError: If MCS has not been computed
        """
        if not self._fitted or self._included_models is None:
            raise RuntimeError("MCS has not been computed. Call compute() first.")
        return self._included_models
    
    @property
    def pvalues(self) -> np.ndarray:
        """
        Get the MCS p-values for each model.
        
        Returns:
            np.ndarray: MCS p-values
            
        Raises:
            RuntimeError: If MCS has not been computed
        """
        if not self._fitted or self._pvalues is None:
            raise RuntimeError("MCS has not been computed. Call compute() first.")
        return self._pvalues
    
    @property
    def eliminated_order(self) -> np.ndarray:
        """
        Get the order in which models were eliminated.
        
        Returns:
            np.ndarray: Order of model elimination
            
        Raises:
            RuntimeError: If MCS has not been computed
        """
        if not self._fitted or self._eliminated_order is None:
            raise RuntimeError("MCS has not been computed. Call compute() first.")
        return self._eliminated_order
    
    @property
    def test_statistics(self) -> np.ndarray:
        """
        Get the test statistics for each elimination step.
        
        Returns:
            np.ndarray: Test statistics
            
        Raises:
            RuntimeError: If MCS has not been computed
        """
        if not self._fitted or self._test_statistics is None:
            raise RuntimeError("MCS has not been computed. Call compute() first.")
        return self._test_statistics
    
    def plot_pvalues(
        self,
        model_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (10, 6),
        title: str = "Model Confidence Set p-values",
        show_threshold: bool = True,
        sort_values: bool = True,
        **kwargs: Any
    ) -> plt.Figure:
        """
        Plot the MCS p-values for each model.
        
        Args:
            model_names: Names of the models (if None, uses model indices)
            figsize: Figure size (width, height)
            title: Plot title
            show_threshold: Whether to show the significance threshold
            sort_values: Whether to sort models by p-value
            **kwargs: Additional keyword arguments for matplotlib
        
        Returns:
            plt.Figure: Matplotlib figure object
        
        Raises:
            RuntimeError: If MCS has not been computed
        """
        if not self._fitted or self._pvalues is None:
            raise RuntimeError("MCS has not been computed. Call compute() first.")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get model names or indices
        if model_names is None:
            model_names = [f"Model {i}" for i in range(len(self._pvalues))]
        
        # Sort models by p-value if requested
        if sort_values:
            sorted_indices = np.argsort(self._pvalues)[::-1]  # Descending order
            pvalues = self._pvalues[sorted_indices]
            names = [model_names[i] for i in sorted_indices]
        else:
            pvalues = self._pvalues
            names = model_names
        
        # Plot p-values
        bars = ax.bar(names, pvalues, **kwargs)
        
        # Highlight models in the MCS
        for i, bar in enumerate(bars):
            if pvalues[i] >= self.params.alpha:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        # Show significance threshold if requested
        if show_threshold:
            ax.axhline(y=self.params.alpha, color='black', linestyle='--', 
                      label=f'α = {self.params.alpha}')
            ax.legend()
        
        # Set labels and title
        ax.set_xlabel('Models')
        ax.set_ylabel('p-value')
        ax.set_title(title)
        
        # Rotate x-axis labels if there are many models
        if len(names) > 5:
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    def summary(self) -> str:
        """
        Generate a summary of the MCS results.
        
        Returns:
            str: Summary of the MCS results
        
        Raises:
            RuntimeError: If MCS has not been computed
        """
        if not self._fitted:
            return "Model Confidence Set has not been computed. Call compute() first."
        
        summary_lines = [
            "Model Confidence Set (MCS) Results",
            "=" * 40,
            f"Significance level (α): {self.params.alpha}",
            f"Bootstrap method: {self.params.bootstrap_method}",
            f"Number of bootstrap replications: {self.params.n_bootstraps}",
            f"Block length: {self.params.block_length}",
            f"Test statistic: {self.params.test_statistic}",
            f"Number of models: {self._losses.shape[1]}",
            f"Number of time periods: {self._losses.shape[0]}",
            f"Number of models in MCS: {len(self._included_models)}",
            "",
            "Models in MCS:",
            "-" * 20
        ]
        
        # Add models in MCS with their p-values
        for i, model_idx in enumerate(self._included_models):
            summary_lines.append(f"Model {model_idx}: p-value = {self._pvalues[model_idx]:.4f}")
        
        # Add elimination order
        summary_lines.extend([
            "",
            "Elimination Order:",
            "-" * 20
        ])
        
        for i, model_idx in enumerate(self._eliminated_order):
            if i < len(self._test_statistics):
                summary_lines.append(
                    f"Step {i+1}: Model {model_idx} eliminated "
                    f"(test statistic = {self._test_statistics[i]:.4f}, "
                    f"p-value = {self._pvalues[model_idx]:.4f})"
                )
            else:
                summary_lines.append(f"Step {i+1}: Model {model_idx}")
        
        return "\n".join(summary_lines)
    
    def __str__(self) -> str:
        """Return a string representation of the ModelConfidenceSet instance."""
        if self._fitted:
            return (
                f"ModelConfidenceSet(alpha={self.params.alpha}, "
                f"bootstrap_method='{self.params.bootstrap_method}', "
                f"n_bootstraps={self.params.n_bootstraps}, "
                f"test_statistic='{self.params.test_statistic}', "
                f"fitted=True, "
                f"models_in_mcs={len(self._included_models)})"
            )
        else:
            return (
                f"ModelConfidenceSet(alpha={self.params.alpha}, "
                f"bootstrap_method='{self.params.bootstrap_method}', "
                f"n_bootstraps={self.params.n_bootstraps}, "
                f"test_statistic='{self.params.test_statistic}', "
                f"fitted=False)"
            )
    
    def __repr__(self) -> str:
        """Return a detailed string representation of the ModelConfidenceSet instance."""
        return (
            f"ModelConfidenceSet(alpha={self.params.alpha}, "
            f"bootstrap_method='{self.params.bootstrap_method}', "
            f"block_length={self.params.block_length}, "
            f"n_bootstraps={self.params.n_bootstraps}, "
            f"test_statistic='{self.params.test_statistic}', "
            f"random_state={self.params.random_state}, "
            f"fitted={self._fitted})"
        )
