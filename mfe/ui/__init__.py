# mfe/ui/__init__.py
"""
MFE Toolbox UI Module

This module provides graphical user interface components for the MFE Toolbox,
specifically focused on the ARMAX (AutoRegressive Moving Average with eXogenous inputs)
modeling interface. It implements a modern PyQt6-based application that enables
interactive time series analysis, estimation, and diagnostics.

The UI module leverages Python's asynchronous capabilities (async/await) to maintain
responsiveness during computationally intensive operations, and follows the
Model-View-Controller (MVC) architectural pattern for clean separation of concerns.

Key components:
- ARMAX modeling application with interactive parameter configuration
- Model viewer for displaying estimation results and diagnostics
- About and confirmation dialogs
- Visualization components for time series and diagnostic plots
"""

import logging
import sys
from typing import Dict, List, Optional, Union, Any, Tuple, Set, Callable, Awaitable

# Add compatibility layer for PyQt6
try:
    from PyQt6.QtWidgets import QAction
except ImportError:
    try:
        from PyQt6.QtGui import QAction
        import PyQt6.QtWidgets
        # Add QAction to QtWidgets namespace for backward compatibility
        PyQt6.QtWidgets.QAction = QAction
    except ImportError:
        logging.warning("PyQt6 not available. UI components will not function.")

# Set up module-level logger
logger = logging.getLogger("mfe.ui")

# UI module version (follows main package version)
__version__ = "4.0.0"

# Import main application class to make it available at the package level
try:
    from .armax_app import ARMAXApplication as ARMAXApp
except ImportError:
    # Create a placeholder for ARMAXApp if it's not available
    logger = logging.getLogger("mfe.ui")
    logger.warning("ARMAXApp not available. UI functionality will be limited.")
    
    class ARMAXApp:
        """Placeholder for ARMAXApp when PyQt6 is not available."""
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("ARMAXApp is not available. Please install PyQt6.")

# Import dialog components
from .about_dialog import AboutDialog
from .close_dialog import CloseDialog
from .model_viewer import ModelViewer

# Import utility functions
from .utils import (
    create_figure_canvas,
    embed_matplotlib_figure,
    format_equation,
    create_parameter_table,
    create_results_table
)

# Import launch function
# from .launch import launch_armax_app  # Commented out as this module doesn't exist

# Initialize UI module
def _initialize_ui() -> None:
    """
    Initialize the UI module.
    
    This function:
    1. Sets up logging for the UI module
    2. Checks for PyQt6 availability
    3. Configures matplotlib for embedding in PyQt6
    """
    # Configure logging
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    
    # Set default log level (can be overridden by configuration)
    log_level = logging.INFO
    logger.setLevel(log_level)
    
    # Check for PyQt6 availability
    try:
        import PyQt6
        logger.debug(f"PyQt6 version {PyQt6.QtCore.PYQT_VERSION_STR} detected")
    except ImportError:
        logger.warning(
            "PyQt6 not found. UI components will not be available. "
            "Install PyQt6 with: pip install PyQt6"
        )
        return
    
    # Configure matplotlib for PyQt6 integration
    try:
        import matplotlib
        matplotlib.use('Qt5Agg')  # Use Qt5Agg backend for matplotlib
        logger.debug(f"Matplotlib configured for PyQt6 integration")
    except ImportError:
        logger.warning(
            "Matplotlib not found. Visualization components will be limited. "
            "Install matplotlib with: pip install matplotlib"
        )
    except Exception as e:
        logger.warning(f"Failed to configure matplotlib for PyQt6: {e}")
    
    logger.debug(f"MFE UI module v{__version__} initialized")


# Check if PyQt6 is available
def has_pyqt6() -> bool:
    """
    Check if PyQt6 is available.
    
    Returns:
        bool: True if PyQt6 is available, False otherwise
    """
    try:
        import PyQt6
        return True
    except ImportError:
        return False


# Define the launch_armax_app function directly in this module
async def launch_armax_app(data=None):
    """
    Launch the ARMAX modeling application asynchronously.
    
    Args:
        data: Optional time series data to load into the application
        
    Returns:
        Optional[ARMAXApp]: The ARMAX application instance if successful, None otherwise
        
    Raises:
        ImportError: If PyQt6 is not available
        RuntimeError: If the application fails to start
    """
    # Check if PyQt6 is available
    if not has_pyqt6():
        raise ImportError("PyQt6 is required to run the ARMAX application")
    
    try:
        # Initialize the UI module
        _initialize_ui()
        
        # Create and launch the application
        app = ARMAXApp()
        await app.initialize()
        
        # Load data if provided
        if data is not None:
            await app.load_data(data)
        
        # Show the application window
        app.show()
        
        return app
    
    except NotImplementedError:
        logger = logging.getLogger("mfe.ui")
        logger.error("ARMAXApp is not available in this installation.")
        return None
    except Exception as e:
        logger = logging.getLogger("mfe.ui")
        logger.error(f"Failed to launch ARMAX application: {e}")
        return None


# Initialize the UI module
_initialize_ui()

# Define what's available when using "from mfe.ui import *"
__all__ = [
    # Main application
    'ARMAXApp',
    
    # Dialog components
    'AboutDialog',
    'CloseDialog',
    'ModelViewer',
    
    # Utility functions
    'create_figure_canvas',
    'embed_matplotlib_figure',
    'format_equation',
    'create_parameter_table',
    'create_results_table',
    
    # Module functions
    'has_pyqt6',
    'launch_armax_app',
]

logger.debug("MFE UI module import complete")