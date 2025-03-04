"""
MFE Toolbox Bootstrap Module

This module provides implementations of bootstrap methods for dependent data,
including block bootstrap, stationary bootstrap, and model confidence set procedures.
These methods are essential for robust statistical inference in financial time series
with non-standard distributions and dependencies.

The bootstrap module enables users to:
- Construct confidence intervals for parameter estimates
- Perform hypothesis tests without relying on parametric assumptions
- Compare models using the Model Confidence Set procedure
- Implement the Bootstrap Reality Check and Superior Predictive Ability tests
- Generate bootstrap samples that preserve time series dependencies

All implementations leverage NumPy's efficient array operations with performance-critical
sections accelerated using Numba's @jit decorators.
"""

import logging
from typing import List, Dict, Union, Optional, Tuple, Callable, Any, TypeVar, cast

# Set up module-level logger
logger = logging.getLogger("mfe.models.bootstrap")

# Import core bootstrap classes and functions
from .base import Bootstrap
from .block_bootstrap import BlockBootstrap
from .stationary_bootstrap import StationaryBootstrap
from .mcs import ModelConfidenceSet
from .bsds import BSDS, RealityCheck, SPA
from .utils import (
    bootstrap_indices,
    stationary_bootstrap_indices,
    moving_block_bootstrap_indices,
    circular_block_bootstrap_indices
)

# Import utility functions that may be useful for users
from .utils import (
    compute_statistic,
    compute_variance,
    compute_confidence_interval,
    compute_pvalue
)

# Type variable for generic bootstrap methods
T = TypeVar('T', bound=Bootstrap)


def create_bootstrap(
    data: Union[List, "numpy.ndarray", "pandas.Series", "pandas.DataFrame"],
    method: str = "stationary",
    block_length: Optional[float] = None,
    num_bootstraps: int = 1000,
    seed: Optional[int] = None
) -> Bootstrap:
    """
    Create a bootstrap instance with the specified method.
    
    This is a convenience function to create bootstrap objects with appropriate
    default parameters based on the data characteristics.
    
    Args:
        data: Time series data to bootstrap
        method: Bootstrap method to use, one of:
               - "stationary": Stationary bootstrap with random block lengths
               - "block": Circular block bootstrap with fixed block length
               - "moving": Moving block bootstrap with fixed block length
        block_length: Block length parameter (if None, will be automatically determined)
        num_bootstraps: Number of bootstrap replications to generate
        seed: Random seed for reproducibility
        
    Returns:
        Bootstrap instance of the requested type
        
    Raises:
        ValueError: If an invalid bootstrap method is specified
    """
    import numpy as np
    
    # Convert data to numpy array if needed
    if not isinstance(data, np.ndarray):
        try:
            import pandas as pd
            if isinstance(data, (pd.Series, pd.DataFrame)):
                data = data.values
            else:
                data = np.asarray(data)
        except ImportError:
            data = np.asarray(data)
    
    # Determine block length if not provided
    if block_length is None:
        # Implement automatic block length selection
        # A common rule of thumb is to use n^(1/3) where n is the sample size
        n = len(data)
        block_length = int(n ** (1/3))
        logger.info(f"Automatically selected block length: {block_length}")
    
    # Create the appropriate bootstrap instance
    if method.lower() == "stationary":
        return StationaryBootstrap(
            block_length=block_length,
            data=data,
            num_bootstraps=num_bootstraps,
            seed=seed
        )
    elif method.lower() == "block":
        return BlockBootstrap(
            block_length=block_length,
            data=data,
            num_bootstraps=num_bootstraps,
            seed=seed,
            wrap=True  # Circular block bootstrap
        )
    elif method.lower() == "moving":
        return BlockBootstrap(
            block_length=block_length,
            data=data,
            num_bootstraps=num_bootstraps,
            seed=seed,
            wrap=False  # Moving block bootstrap
        )
    else:
        raise ValueError(
            f"Invalid bootstrap method: {method}. "
            f"Must be one of: 'stationary', 'block', or 'moving'."
        )



def model_confidence_set(
    losses: "numpy.ndarray",
    alpha: float = 0.05,
    block_length: Optional[float] = None,
    num_bootstraps: int = 1000,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compute the Model Confidence Set (MCS) for a collection of models.
    
    This is a convenience function to perform the MCS procedure directly
    from a matrix of loss values.
    
    Args:
        losses: Loss matrix of shape (T, M) where T is the number of time periods
               and M is the number of models
        alpha: Significance level for the MCS
        block_length: Block length for bootstrap (if None, will be automatically determined)
        num_bootstraps: Number of bootstrap replications
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing:
        - 'included': Indices of models in the MCS
        - 'pvalues': MCS p-values for each model
        - 'eliminated': Order in which models were eliminated
        - 'statistics': Test statistics for each elimination step
    """
    mcs = ModelConfidenceSet(
        losses=losses,
        alpha=alpha,
        block_length=block_length,
        num_bootstraps=num_bootstraps,
        seed=seed
    )
    mcs.compute()
    return {
        'included': mcs.included_models,
        'pvalues': mcs.pvalues,
        'eliminated': mcs.eliminated_order,
        'statistics': mcs.test_statistics
    }



def bootstrap_test(
    data: "numpy.ndarray",
    statistic_func: Callable["numpy.ndarray", float],
    null_value: float = 0.0,
    method: str = "stationary",
    block_length: Optional[float] = None,
    num_bootstraps: int = 1000,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform a bootstrap hypothesis test for a given statistic.
    
    Args:
        data: Time series data
        statistic_func: Function that computes the test statistic from data
        null_value: Value of the statistic under the null hypothesis
        method: Bootstrap method to use ('stationary', 'block', or 'moving')
        block_length: Block length parameter (if None, will be automatically determined)
        num_bootstraps: Number of bootstrap replications
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing:
        - 'statistic': Original test statistic
        - 'pvalue': Bootstrap p-value
        - 'bootstrap_statistics': Array of bootstrap statistics
        - 'confidence_interval': 95% confidence interval
    """
    # Create bootstrap instance
    bootstrap = create_bootstrap(
        data=data,
        method=method,
        block_length=block_length,
        num_bootstraps=num_bootstraps,
        seed=seed
    )
    
    # Compute original statistic
    original_stat = statistic_func(data)
    
    # Generate bootstrap samples and compute statistics
    bootstrap_stats = []
    for i in range(num_bootstraps):
        bootstrap_sample = bootstrap.generate_sample()
        bootstrap_stats.append(statistic_func(bootstrap_sample))
    
    import numpy as np
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Compute p-value (two-sided test)
    p_value = np.mean(np.abs(bootstrap_stats - null_value) >= np.abs(original_stat - null_value))
    
    # Compute confidence interval
    alpha = 0.05  # 95% confidence interval
    lower = np.percentile(bootstrap_stats, alpha/2 * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha/2) * 100)
    
    return {
        'statistic': original_stat,
        'pvalue': p_value,
        'bootstrap_statistics': bootstrap_stats,
        'confidence_interval': (lower, upper)
    }


# Define what's available when using "from mfe.models.bootstrap import *"
__all__ = [
    # Core bootstrap classes
    'Bootstrap',
    'BlockBootstrap',
    'StationaryBootstrap',
    'ModelConfidenceSet',
    'BSDS',
    'RealityCheck',
    'SPA',
    
    # Utility functions
    'bootstrap_indices',
    'stationary_bootstrap_indices',
    'moving_block_bootstrap_indices',
    'circular_block_bootstrap_indices',
    'compute_statistic',
    'compute_variance',
    'compute_confidence_interval',
    'compute_pvalue',
    
    # Convenience functions
    'create_bootstrap',
    'model_confidence_set',
    'bootstrap_test'
]

logger.debug("MFE Bootstrap module initialized successfully")