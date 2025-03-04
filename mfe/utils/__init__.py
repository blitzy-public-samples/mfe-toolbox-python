"""
MFE Toolbox Utilities Module

This module provides essential utility functions used throughout the MFE Toolbox.
It includes matrix operations, covariance estimators, numerical differentiation,
data transformations, date utilities, and miscellaneous helper functions.

These utilities form the foundation for the econometric models and statistical
methods implemented in the toolbox, providing consistent and efficient implementations
of common operations.

Key components:
- Matrix operations (vech, ivech, vec2chol, chol2vec, etc.)
- Covariance estimators (robust covariance, Newey-West, etc.)
- Numerical differentiation (gradient, Hessian)
- Data transformations (standardization, demeaning, etc.)
- Date utilities for time series handling
- Miscellaneous helper functions
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Set, Callable

# Set up module-level logger
logger = logging.getLogger("mfe.utils")

# Import matrix operations
from .matrix_ops import (
    vech,
    ivech,
    vec2chol,
    chol2vec,
    cov2corr,
    corr2cov,
    ensure_symmetric,
    is_positive_definite,
    nearest_positive_definite,
    block_diagonal,
    commutation_matrix,
    duplication_matrix,
    elimination_matrix
)

# Import covariance estimators
from .covariance import (
    covnw,
    covvar,
    robustvcv,
    kernel_weight,
    kernel_optimal_bandwidth
)

# Import numerical differentiation utilities
from .differentiation import (
    gradient_2sided,
    hessian_2sided,
    jacobian,
    numerical_derivative,
    numerical_hessian
)

# Import data transformation utilities
from .data_transformations import (
    standardize,
    mvstandardize,
    demean,
    lag_matrix,
    lag_series,
    rolling_window,
    rolling_mean,
    rolling_variance,
    rolling_skewness,
    rolling_kurtosis
)

# Import date utilities
from .date_utils import (
    date_to_index,
    index_to_date,
    date_range,
    business_day_count,
    is_business_day,
    next_business_day,
    previous_business_day,
    align_time_series
)

# Import miscellaneous utilities
from .misc import (
    r2z,
    z2r,
    phi2r,
    r2phi,
    ensure_array,
    ensure_dataframe,
    ensure_series,
    progress_bar,
    format_time
)

# Register Numba-accelerated functions if available
def _register_numba_functions() -> None:
    """
    Register Numba JIT-compiled functions for utility operations.
    
    This function attempts to import Numba and register performance-critical
    utility functions for JIT compilation. If Numba is not available, it falls
    back to pure Python implementations.
    """
    try:
        import numba
        logger.debug(f"Numba version {numba.__version__} detected")
        
        # Import Numba-accelerated versions of critical functions
        # These would typically be defined in the respective modules with @jit decorators
        
        logger.debug("Utility Numba JIT functions registered")
    except ImportError:
        logger.info("Numba not available. Utility functions will use pure Python implementations.")
    except Exception as e:
        logger.warning(f"Failed to register utility Numba functions: {e}")


# Initialize the utils module
def _initialize_utils() -> None:
    """
    Initialize the utils module.
    
    This function:
    1. Sets up logging for the utils module
    2. Registers Numba-accelerated functions if available
    """
    # Configure logging
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    
    # Set default log level (can be overridden by configuration)
    log_level = logging.INFO
    logger.setLevel(log_level)
    
    # Register Numba functions
    _register_numba_functions()
    
    logger.debug("MFE Utils module initialized")


# Initialize the module
_initialize_utils()

# Define what's available when using "from mfe.utils import *"
__all__ = [
    # Matrix operations
    'vech',
    'ivech',
    'vec2chol',
    'chol2vec',
    'cov2corr',
    'corr2cov',
    'ensure_symmetric',
    'is_positive_definite',
    'nearest_positive_definite',
    'block_diagonal',
    'commutation_matrix',
    'duplication_matrix',
    'elimination_matrix',
    
    # Covariance estimators
    'covnw',
    'covvar',
    'robustvcv',
    'kernel_weight',
    'kernel_optimal_bandwidth',
    
    # Numerical differentiation
    'gradient_2sided',
    'hessian_2sided',
    'jacobian',
    'numerical_derivative',
    'numerical_hessian',
    
    # Data transformations
    'standardize',
    'mvstandardize',
    'demean',
    'lag_matrix',
    'lag_series',
    'rolling_window',
    'rolling_mean',
    'rolling_variance',
    'rolling_skewness',
    'rolling_kurtosis',
    
    # Date utilities
    'date_to_index',
    'index_to_date',
    'date_range',
    'business_day_count',
    'is_business_day',
    'next_business_day',
    'previous_business_day',
    'align_time_series',
    
    # Miscellaneous utilities
    'r2z',
    'z2r',
    'phi2r',
    'r2phi',
    'ensure_array',
    'ensure_dataframe',
    'ensure_series',
    'progress_bar',
    'format_time'
]

logger.debug("MFE Utils module import complete")
