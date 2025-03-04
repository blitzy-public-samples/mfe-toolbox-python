"""
MFE Toolbox UI Models Package

This package contains model classes for the MFE Toolbox UI components, following the
Model part of the Model-View-Controller (MVC) pattern. These models encapsulate the
 data and business logic for the UI components, separate from their visual representation
and controller logic.

The models in this package handle:
- Data storage and manipulation for UI components
- Business logic for UI operations
- State management for UI components
- Data validation and transformation

Each model class corresponds to a specific UI component and provides a clean API
for controllers to interact with the underlying data.
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Set, Callable, Awaitable

# Import version information
from mfe.version import __version__

# Set up module-level logger
logger = logging.getLogger("mfe.ui.models")

# Import all model classes to make them available when importing the package
from .armax_model import ARMAXModel
from .about_dialog_model import AboutDialogModel
from .close_dialog_model import CloseDialogModel
from .model_viewer_model import ModelViewerModel

# Initialize models package
def _initialize_models() -> None:
    """
    Initialize the UI models package.
    
    This function:
    1. Sets up logging for the models package
    2. Performs any necessary initialization for model classes
    """
    # Set default log level (can be overridden by configuration)
    log_level = logging.INFO
    logger.setLevel(log_level)
    
    logger.debug(f"MFE UI models package v{__version__} initialized")

# Initialize the models package
_initialize_models()

# Define what's available when using "from mfe.ui.models import *"
__all__ = [
    # Model classes
    'ARMAXModel',
    'AboutDialogModel',
    'CloseDialogModel',
    'ModelViewerModel',
]

logger.debug("MFE UI models package import complete")
