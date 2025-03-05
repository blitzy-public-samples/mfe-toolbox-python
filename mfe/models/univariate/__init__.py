'''
MFE Toolbox - Univariate Volatility Models

This module provides a comprehensive collection of univariate volatility models
for financial time series analysis. These models capture time-varying volatility
patterns in financial returns and other time series data.

The module includes implementations of:
- GARCH (Generalized Autoregressive Conditional Heteroskedasticity)
- EGARCH (Exponential GARCH)
- TARCH (Threshold ARCH)
- AGARCH (Asymmetric GARCH)
- APARCH (Asymmetric Power ARCH)
- FIGARCH (Fractionally Integrated GARCH)
- HEAVY (High-frEquency-bAsed VolatilitY)
- IGARCH (Integrated GARCH)

All models are implemented using a class-based architecture with inheritance
from a common base class, providing consistent interfaces for estimation,
forecasting, and simulation.
'''  

import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Set
import warnings

# Set up module-level logger
logger = logging.getLogger("mfe.models.univariate")

# Import base classes and utilities
from .base import (
    VolatilityModel,
    VolatilityModelResult,
    VolatilityForecast,
    VolatilityParameters
)

# Import model implementations
try:
    from .garch import GARCH
    from .egarch import EGARCH
    from .tarch import TARCH
    from .agarch import AGARCH
    from .aparch import APARCH
    from .figarch import FIGARCH
    from .heavy import HEAVY
    from .igarch import IGARCH
except ImportError as e:
    logger.error(f"Error importing univariate volatility models: {e}")
    raise ImportError(
        "Failed to import univariate volatility models. Please ensure the package "
        "is correctly installed. You can install it using: "
        "pip install mfe-toolbox"
    ) from e

# Import utility functions
try:
    from .utils import (
        volatility_process_to_variance,
        variance_to_volatility_process,
        backcast_variance,
        simulate_univariate_volatility,
        forecast_volatility
    )
except ImportError as e:
    logger.warning(f"Some utility functions could not be imported: {e}")

# Import Numba-accelerated core functions
try:
    from ._core import (
        garch_recursion,
        egarch_recursion,
        tarch_recursion,
        aparch_recursion,
        figarch_recursion,
        igarch_recursion,
        agarch_recursion,
        heavy_recursion
    )
except ImportError as e:
    logger.warning(f"Numba-accelerated core functions could not be imported: {e}")
    logger.warning("Performance will be reduced for computationally intensive operations")

# Import error classes specific to volatility modeling
from mfe.core.exceptions import (
    MFEError,
    ParameterError,
    ConvergenceError,
    NumericError,
    EstimationError,
    ForecastError,
    SimulationError
)

# Define custom volatility model exceptions
def VolatilityModelError(MFEError):
    """Exception raised for errors specific to volatility models.
    
    This exception is used when volatility model-specific errors occur that
    are not covered by more general exception types.
    
    Attributes:
        model_type: The type of volatility model
        issue: Description of the issue with the model
    """
    
    def __init__(self, 
                 message: str, 
                 model_type: Optional[str] = None,
                 issue: Optional[str] = None,
                 details: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the VolatilityModelError.
        
        Args:
            message: The primary error message
            model_type: The type of volatility model
            issue: Description of the issue with the model
            details: Additional details about the error
            context: Dictionary containing contextual information about the error
        """
        self.model_type = model_type
        self.issue = issue
        
        # Add model information to context
        context_dict = context or {}
        if model_type:
            context_dict["Model Type"] = model_type
        if issue:
            context_dict["Issue"] = issue
        
        super().__init__(message, details, context_dict)


class StationarityError(ParameterError):
    """Exception raised when volatility model parameters violate stationarity conditions.
    
    This exception is used when model parameters would result in a non-stationary
    volatility process.
    
    Attributes:
        model_type: The type of volatility model
        condition: The stationarity condition that was violated
        param_values: The parameter values that violated the condition
    """
    
    def __init__(self, 
                 message: str, 
                 model_type: Optional[str] = None,
                 condition: Optional[str] = None,
                 param_values: Optional[Dict[str, float]] = None,
                 details: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the StationarityError.
        
        Args:
            message: The primary error message
            model_type: The type of volatility model
            condition: The stationarity condition that was violated
            param_values: The parameter values that violated the condition
            details: Additional details about the error
            context: Dictionary containing contextual information about the error
        """
        self.model_type = model_type
        self.condition = condition
        self.param_values = param_values
        
        # Add stationarity information to context
        context_dict = context or {}
        if model_type:
            context_dict["Model Type"] = model_type
        if condition:
            context_dict["Condition"] = condition
        if param_values:
            context_dict["Parameter Values"] = param_values
        
        super().__init__(message, details=details, context=context_dict)


def list_volatility_models() -> List[str]:
    """
    List all available univariate volatility models.
    
    Returns:
        List of available model class names
    """
    return [
        "GARCH",
        "EGARCH",
        "TARCH",
        "AGARCH",
        "APARCH",
        "FIGARCH",
        "HEAVY",
        "IGARCH"
    ]


def get_volatility_model(model_name: str) -> type:
    """
    Get a volatility model class by name.
    
    Args:
        model_name: Name of the volatility model class
        
    Returns:
        The requested volatility model class
        
    Raises:
        ValueError: If the requested model is not available
    """
    models = {
        "GARCH": GARCH,
        "EGARCH": EGARCH,
        "TARCH": TARCH,
        "AGARCH": AGARCH,
        "APARCH": APARCH,
        "FIGARCH": FIGARCH,
        "HEAVY": HEAVY,
        "IGARCH": IGARCH
    }
    
    if model_name not in models:
        available = ", ".join(models.keys())
        raise ValueError(
            f"Unknown volatility model: {model_name}. "
            f"Available models are: {available}"
        )
    
    return models[model_name]


# Initialize the module
logger.debug("Univariate volatility models module initialized")

# Define what's available when using "from mfe.models.univariate import *"
__all__ = [
    # Base classes
    'VolatilityModel',
    'VolatilityModelResult',
    'VolatilityForecast',
    'VolatilityParameters',
    
    # Model classes
    'GARCH',
    'EGARCH',
    'TARCH',
    'AGARCH',
    'APARCH',
    'FIGARCH',
    'HEAVY',
    'IGARCH',
    
    # Utility functions
    'volatility_process_to_variance',
    'variance_to_volatility_process',
    'backcast_variance',
    'simulate_univariate_volatility',
    'forecast_volatility',
    
    # Core functions
    'garch_recursion',
    'egarch_recursion',
    'tarch_recursion',
    'aparch_recursion',
    'figarch_recursion',
    'igarch_recursion',
    'agarch_recursion',
    'heavy_recursion',
    
    # Exception classes
    'VolatilityModelError',
    'StationarityError',
    
    # Module functions
    'list_volatility_models',
    'get_volatility_model'
]
