# mfe/ui/views/__init__.py
"""
MFE Toolbox UI Views Module

This module provides the view components for the MFE Toolbox's graphical user interface,
implementing the View layer in the Model-View-Controller (MVC) architecture. The views
are responsible for rendering the user interface and capturing user input, but delegate
business logic to their respective controllers.

The views are built using PyQt6 and follow modern UI design principles with responsive
layouts, asynchronous processing support, and matplotlib integration for visualization.
Each view class is designed to be reusable and maintainable, with clear separation from
business logic.

Key components:
- ARMAXView: Main application window for ARMAX modeling
- AboutDialogView: Information dialog showing application details
- CloseDialogView: Confirmation dialog for application exit
- ModelViewerView: Detailed view for displaying model results
- BaseView: Abstract base class with common functionality for all views
- Components: Reusable UI components shared across views
- Styles: Consistent styling definitions for UI elements
"""

import logging
from typing import List, Dict, Optional, Union, Any, Tuple, Type

# Set up module-level logger
logger = logging.getLogger("mfe.ui.views")

# Import all view classes to make them available at the module level
from .armax_view import ARMAXView
from .about_dialog_view import AboutDialogView
from .close_dialog_view import CloseDialogView
from .model_viewer_view import ModelViewerView
from .base_view import BaseView

# Import utility modules
from .components import (
    FigureCanvas,
    ParameterTable,
    ResultsTable,
    EquationLabel,
    DiagnosticPlot,
    ProgressIndicator
)
from .styles import (
    get_default_style,
    apply_style_to_widget,
    get_theme_colors,
    get_font_metrics
)

# Define what's available when using "from mfe.ui.views import *"
__all__: List[str] = [
    # View classes
    'ARMAXView',
    'AboutDialogView',
    'CloseDialogView',
    'ModelViewerView',
    'BaseView',
    
    # Component classes
    'FigureCanvas',
    'ParameterTable',
    'ResultsTable',
    'EquationLabel',
    'DiagnosticPlot',
    'ProgressIndicator',
    
    # Style functions
    'get_default_style',
    'apply_style_to_widget',
    'get_theme_colors',
    'get_font_metrics'
]

# Initialize the views module
def _initialize_views() -> None:
    """
    Initialize the views module.
    
    This function:
    1. Sets up logging for the views module
    2. Checks for PyQt6 availability
    3. Configures default styles
    """
    # Configure logging
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    
    # Set default log level (can be overridden by configuration)
    logger.setLevel(logging.INFO)
    
    # Check for PyQt6 availability
    try:
        import PyQt6
        logger.debug(f"PyQt6 version {PyQt6.QtCore.PYQT_VERSION_STR} detected for views")
    except ImportError:
        logger.warning(
            "PyQt6 not found. UI views will not be available. "
            "Install PyQt6 with: pip install PyQt6"
        )
        return
    
    # Check for matplotlib availability (needed for embedded plots)
    try:
        import matplotlib
        logger.debug(f"Matplotlib version {matplotlib.__version__} detected for views")
    except ImportError:
        logger.warning(
            "Matplotlib not found. Visualization components will be limited. "
            "Install matplotlib with: pip install matplotlib"
        )
    
    logger.debug("MFE UI views module initialized")


# Check if PyQt6 is available for views
def has_pyqt6() -> bool:
    """
    Check if PyQt6 is available for views.
    
    Returns:
        bool: True if PyQt6 is available, False otherwise
    """
    try:
        import PyQt6
        return True
    except ImportError:
        return False


# Initialize the views module
_initialize_views()

logger.debug("MFE UI views module import complete")
