"""
MFE Toolbox Core Module

This module provides the core functionality and base classes for the MFE Toolbox.
It defines fundamental abstractions, parameter containers, result objects, type
definitions, validation utilities, and exception classes used throughout the toolbox.

The core module serves as the foundation for all other components, ensuring
consistent behavior, robust error handling, and type safety across the entire
toolbox.

Key components:
- Base classes for models and estimators
- Parameter containers with validation
- Result objects for storing estimation outputs
- Custom type definitions and annotations
- Exception hierarchy for error handling
- Validation utilities for input checking
- Configuration management
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Set

# Set up module-level logger
logger = logging.getLogger("mfe.core")

# Core module version (follows main package version)
__version__ = "4.0.0"

# Import core components to make them available at the package level
from .base import (
    BaseModel,
    BaseEstimator,
    BaseResult,
    ModelConfig,
    EstimationResult,
    has_numba,
    requires_numba
)

from .parameters import (
    Parameter,
    ParameterSet,
    ConstrainedParameter,
    TransformedParameter,
    validate_parameters
)

from .results import (
    ResultContainer,
    ModelResult,
    DiagnosticResult,
    ForecastResult,
    SimulationResult
)

from .types import (
    ArrayLike,
    FloatArray,
    IntArray,
    TimeSeriesData,
    MatrixLike,
    OptimizeResult,
    StrEnum,
    ModelType
)

from .exceptions import (
    MFEError,
    ParameterError,
    DimensionError,
    ConvergenceError,
    NumericError,
    ValidationError,
    ConfigurationError
)

from .validation import (
    validate_array,
    validate_dimensions,
    validate_type,
    validate_positive,
    validate_nonnegative,
    validate_in_range,
    validate_shape,
    check_stationarity,
    check_invertibility
)

from .config import (
    get_config,
    set_config,
    reset_config,
    ConfigManager,
    CoreConfig
)

# Initialize core module
def _initialize_core() -> None:
    """
    Initialize the core module.
    
    This function:
    1. Sets up logging for the core module
    2. Registers custom types
    3. Initializes configuration
    """
    # Configure logging
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    
    # Set default log level (can be overridden by configuration)
    log_level = logging.INFO
    logger.setLevel(log_level)
    
    # Initialize configuration
    try:
        from .config import initialize_config
        initialize_config()
        logger.debug("Core configuration initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize core configuration: {e}")
        logger.warning("Using default settings")
    
    logger.debug(f"MFE Core module v{__version__} initialized")


# Register Numba-accelerated functions if available
def _register_numba_functions() -> None:
    """
    Register Numba JIT-compiled functions for core operations.
    
    This function attempts to import Numba and register performance-critical
    functions for JIT compilation. If Numba is not available, it falls back
    to pure Python implementations.
    """
    try:
        import numba
        logger.debug(f"Numba version {numba.__version__} detected")
        
        # No core JIT functions to register yet, but the infrastructure is in place
        
        logger.debug("Core Numba JIT functions registered")
    except ImportError:
        logger.info("Numba not available. Core functions will use pure Python implementations.")
    except Exception as e:
        logger.warning(f"Failed to register core Numba functions: {e}")


# Initialize the core module
_initialize_core()
_register_numba_functions()

# Define what's available when using "from mfe.core import *"
__all__ = [
    # Base classes
    'BaseModel',
    'BaseEstimator',
    'BaseResult',
    'ModelConfig',
    'EstimationResult',
    'has_numba',
    'requires_numba',
    
    # Parameters
    'Parameter',
    'ParameterSet',
    'ConstrainedParameter',
    'TransformedParameter',
    'validate_parameters',
    
    # Results
    'ResultContainer',
    'ModelResult',
    'DiagnosticResult',
    'ForecastResult',
    'SimulationResult',
    
    # Types
    'ArrayLike',
    'FloatArray',
    'IntArray',
    'TimeSeriesData',
    'MatrixLike',
    'OptimizeResult',
    'StrEnum',
    'ModelType',
    
    # Exceptions
    'MFEError',
    'ParameterError',
    'DimensionError',
    'ConvergenceError',
    'NumericError',
    'ValidationError',
    'ConfigurationError',
    
    # Validation
    'validate_array',
    'validate_dimensions',
    'validate_type',
    'validate_positive',
    'validate_nonnegative',
    'validate_in_range',
    'validate_shape',
    'check_stationarity',
    'check_invertibility',
    
    # Configuration
    'get_config',
    'set_config',
    'reset_config',
    'ConfigManager',
    'CoreConfig',
]

logger.debug("MFE Core module import complete")
