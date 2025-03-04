# mfe/models/distributions/__init__.py
"""
MFE Toolbox Statistical Distributions Module

This module provides implementations of statistical distributions commonly used in
financial econometrics, including normal, Student's t, generalized error distribution (GED),
and Hansen's skewed t-distribution. These distributions are used for likelihood evaluation,
random number generation, and statistical inference throughout the MFE Toolbox.

The distributions module serves as a foundation for volatility modeling, time series analysis,
and other statistical procedures that require flexible distributional assumptions beyond
standard normality.

Key components:
- Normal distribution with standard methods
- Standardized Student's t-distribution with degrees of freedom parameter
- Generalized Error Distribution (GED) with shape parameter
- Hansen's Skewed t-distribution with asymmetry parameter
- Composite likelihood functions for multivariate models
- Utility functions for distribution-related operations
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Type

# Set up module-level logger
logger = logging.getLogger("mfe.models.distributions")

# Import distribution classes
from .base import Distribution
from .normal import Normal, NormalDistribution
from .student_t import StudentT, StudentTDistribution
from .generalized_error import GED, GEDDistribution
from .skewed_t import SkewedT, SkewedTDistribution
from .composite_likelihood import (
    CompositeLikelihood,
    PairwiseCompositeLikelihood,
    IndependentCompositeLikelihood
)
from .utils import (
    distribution_from_name,
    get_available_distributions,
    validate_distribution_parameters
)

# Module version (follows main package version)
__version__ = "4.0.0"

# Initialize the distributions module
def _initialize_distributions() -> None:
    """
    Initialize the distributions module.
    
    This function:
    1. Sets up logging for the distributions module
    2. Registers distribution classes
    3. Verifies distribution implementations
    """
    # Configure logging
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    
    # Set default log level (can be overridden by configuration)
    log_level = logging.INFO
    logger.setLevel(log_level)
    
    # Verify distribution implementations
    distributions = [Normal, StudentT, GED, SkewedT]
    for dist_class in distributions:
        try:
            # Verify that each distribution class has required methods
            required_methods = ["pdf", "cdf", "ppf", "loglikelihood", "random"]
            missing_methods = [method for method in required_methods if not hasattr(dist_class, method)]
            
            if missing_methods:
                logger.warning(f"Distribution {dist_class.__name__} is missing required methods: {missing_methods}")
        except Exception as e:
            logger.warning(f"Error verifying distribution {dist_class.__name__}: {e}")
    
    logger.debug(f"MFE Distributions module v{__version__} initialized")

# Register Numba-accelerated functions if available
def _register_numba_functions() -> None:
    """
    Register Numba JIT-compiled functions for distribution operations.
    
    This function attempts to import Numba and register performance-critical
    distribution functions for JIT compilation. If Numba is not available,
    it falls back to pure Python implementations.
    """
    try:
        import numba
        logger.debug(f"Numba version {numba.__version__} detected")
        
        # Distribution functions are registered in their respective modules
        
        logger.debug("Distribution Numba JIT functions registered")
    except ImportError:
        logger.info("Numba not available. Distribution functions will use pure Python implementations.")
    except Exception as e:
        logger.warning(f"Failed to register distribution Numba functions: {e}")

# Initialize the distributions module
_initialize_distributions()
_register_numba_functions()

# Define what's available when using "from mfe.models.distributions import *"
__all__ = [
    # Base classes
    'Distribution',
    
    # Distribution classes
    'Normal',
    'NormalDistribution',
    'StudentT',
    'StudentTDistribution',
    'GED',
    'GEDDistribution',
    'SkewedT',
    'SkewedTDistribution',
    
    # Composite likelihood
    'CompositeLikelihood',
    'PairwiseCompositeLikelihood',
    'IndependentCompositeLikelihood',
    
    # Utility functions
    'distribution_from_name',
    'get_available_distributions',
    'validate_distribution_parameters'
]

logger.debug("MFE Distributions module import complete")
