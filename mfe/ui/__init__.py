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
from typing import Dict, List, Optional, Union, Any, Tuple, Set, Callable, Awaitable

# Set up module-level logger
logger = logging.getLogger("mfe.ui")

# UI module version (follows main package version)
__version__ = "4.0.0"

# Import main application class to make it available at the package level
from .armax_app import ARMAXApp

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


# Launch the ARMAX application
async def launch_armax_app(data: Optional[Any] = None) -> Optional[ARMAXApp]:
    """
    Launch the ARMAX modeling application asynchronously.
    
    Args:
        data: Optional time series data to load into the application
              (NumPy array or Pandas Series/DataFrame)
    
    Returns:
        ARMAXApp instance if successful, None if PyQt6 is not available
        or if the application fails to launch
    
    This function launches the ARMAX modeling application asynchronously,
    allowing the UI to remain responsive during initialization and data loading.
    """
    if not has_pyqt6():
        logger.error("Cannot launch ARMAX app: PyQt6 is not available")
        return None
    
    try:
        from PyQt6.QtWidgets import QApplication
        import sys
        
        # Create QApplication instance if one doesn't exist
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        # Create and show the ARMAX application
        armax_app = ARMAXApp()
        
        # Load data if provided
        if data is not None:
            await armax_app.load_data(data)
        
        armax_app.show()
        
        # Return the application instance
        return armax_app
    except Exception as e:
        logger.error(f"Failed to launch ARMAX app: {e}")
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