#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main application module for the ARMAX modeling interface.

This module serves as the entry point for the ARMAX GUI application, creating
the main window and setting up the MVC architecture. It initializes the UI
components, connects signals to slots for event handling, and configures the
plotting canvas with matplotlib integration.

The application follows a Model-View-Controller (MVC) architecture for better
separation of concerns, with asynchronous processing capabilities for long-running
calculations to maintain UI responsiveness.
"""

import sys
import asyncio
import logging
import traceback
from typing import Optional, List, Dict, Any, Tuple, Union, cast
import numpy as np
import pandas as pd
from pathlib import Path

# PyQt6 imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QMessageBox, QSplashScreen,
    QFileDialog, QProgressDialog
)
from PyQt6.QtCore import Qt, QTimer, QSize, QSettings, pyqtSlot, QEvent
from PyQt6.QtGui import QIcon, QPixmap, QCloseEvent, QFontMetrics

# Matplotlib integration
import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt5Agg backend for matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# MFE Toolbox imports
from mfe.ui.views.armax_view import ARMAXView
from mfe.ui.models.armax_model import ARMAXModel
from mfe.ui.controllers.armax_controller import ARMAXController
from mfe.ui.resources.resource_loader import get_resource_path
from mfe.core.exceptions import (
    MFEError, UIError, AsyncError, ModelSpecificationError,
    EstimationError, DataError
)

# Configure logging
logger = logging.getLogger(__name__)


class ARMAXApplication:
    """
    Main application class for the ARMAX modeling interface.
    
    This class serves as the entry point for the ARMAX GUI application,
    initializing the MVC components and managing the application lifecycle.
    It follows the Singleton pattern to ensure only one instance of the
    application exists.
    
    Attributes:
        app: The QApplication instance
        model: The ARMAX model instance
        view: The ARMAX view instance
        controller: The ARMAX controller instance
        settings: Application settings
    """
    
    _instance: Optional['ARMAXApplication'] = None
    
    def __new__(cls, *args, **kwargs):
        """Implement the Singleton pattern."""
        if cls._instance is None:
            cls._instance = super(ARMAXApplication, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the ARMAX application if not already initialized."""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self.app = None
            self.model = None
            self.view = None
            self.controller = None
            self.settings = None
            self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Configure logging for the application."""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(Path.home() / '.mfe' / 'armax_gui.log')
            ]
        )
        logger.info("ARMAX GUI logging initialized")
    
    def initialize(self, args: List[str]) -> None:
        """
        Initialize the application with command line arguments.
        
        Args:
            args: Command line arguments
        """
        # Create QApplication instance if it doesn't exist
        if QApplication.instance() is None:
            self.app = QApplication(args)
        else:
            self.app = QApplication.instance()
        
        # Set application metadata
        self.app.setApplicationName("MFE ARMAX Modeler")
        self.app.setApplicationVersion("4.0.0")
        self.app.setOrganizationName("MFE Toolbox")
        self.app.setOrganizationDomain("mfe-toolbox.org")
        
        # Load application settings
        self.settings = QSettings()
        
        # Create splash screen
        splash_pixmap = QPixmap(str(get_resource_path("oxford_logo.png")))
        splash = QSplashScreen(splash_pixmap)
        splash.show()
        self.app.processEvents()
        
        # Initialize MVC components
        splash.showMessage("Initializing model...", Qt.AlignmentFlag.AlignBottom)
        self.model = ARMAXModel()
        
        splash.showMessage("Initializing view...", Qt.AlignmentFlag.AlignBottom)
        self.view = ARMAXView()
        
        splash.showMessage("Initializing controller...", Qt.AlignmentFlag.AlignBottom)
        self.controller = ARMAXController(self.model, self.view)
        
        # Restore application state
        self._restore_state()
        
        # Show the main window
        splash.showMessage("Starting application...", Qt.AlignmentFlag.AlignBottom)
        self.view.show()
        splash.finish(self.view)
        
        logger.info("ARMAX GUI application initialized")
    
    def _restore_state(self) -> None:
        """Restore application state from settings."""
        # Restore window geometry
        geometry = self.settings.value("geometry")
        if geometry:
            self.view.restoreGeometry(geometry)
        
        # Restore window state
        state = self.settings.value("windowState")
        if state:
            self.view.restoreState(state)
        
        # Restore other settings
        ar_order = self.settings.value("ar_order", 0, int)
        ma_order = self.settings.value("ma_order", 0, int)
        include_constant = self.settings.value("include_constant", True, bool)
        
        # Update view with restored settings
        self.view.set_ar_order(ar_order)
        self.view.set_ma_order(ma_order)
        self.view.set_include_constant(include_constant)
        
        logger.debug("Application state restored")
    
    def _save_state(self) -> None:
        """Save application state to settings."""
        # Save window geometry
        self.settings.setValue("geometry", self.view.saveGeometry())
        
        # Save window state
        self.settings.setValue("windowState", self.view.saveState())
        
        # Save other settings
        self.settings.setValue("ar_order", self.view.get_ar_order())
        self.settings.setValue("ma_order", self.view.get_ma_order())
        self.settings.setValue("include_constant", self.view.get_include_constant())
        
        logger.debug("Application state saved")
    
    def run(self) -> int:
        """
        Run the application main loop.
        
        Returns:
            int: Application exit code
        """
        if self.app is None:
            raise RuntimeError("Application not initialized. Call initialize() first.")
        
        # Install exception hook for uncaught exceptions
        sys._excepthook = sys.excepthook
        
        def exception_hook(exctype, value, traceback_obj):
            """Handle uncaught exceptions."""
            error_msg = ''.join(traceback.format_exception(exctype, value, traceback_obj))
            logger.critical(f"Uncaught exception: {error_msg}")
            
            # Show error dialog
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Icon.Critical)
            error_dialog.setWindowTitle("Error")
            error_dialog.setText("An unexpected error occurred.")
            error_dialog.setInformativeText(str(value))
            error_dialog.setDetailedText(error_msg)
            error_dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
            error_dialog.exec()
            
            # Call the original exception hook
            sys._excepthook(exctype, value, traceback_obj)
        
        sys.excepthook = exception_hook
        
        # Run the application
        logger.info("Starting ARMAX GUI application main loop")
        exit_code = self.app.exec()
        
        # Save application state before exiting
        self._save_state()
        
        logger.info(f"ARMAX GUI application exited with code {exit_code}")
        return exit_code
    
    def shutdown(self) -> None:
        """Perform cleanup operations before application shutdown."""
        # Save application state
        self._save_state()
        
        # Clean up resources
        if self.controller:
            self.controller.cleanup()
        
        logger.info("ARMAX GUI application shutdown complete")


class AsyncHelper:
    """
    Helper class for running asynchronous tasks in a Qt application.
    
    This class provides utilities for running asynchronous coroutines
    within the Qt event loop, enabling non-blocking operations for
    long-running tasks like model estimation and forecasting.
    """
    
    @staticmethod
    def run_async(coro, callback=None, error_callback=None):
        """
        Run an asynchronous coroutine in the Qt event loop.
        
        Args:
            coro: The coroutine to run
            callback: Function to call with the result when the coroutine completes
            error_callback: Function to call if the coroutine raises an exception
        """
        future = asyncio.ensure_future(coro)
        
        if callback:
            future.add_done_callback(
                lambda fut: callback(fut.result())
            )
        
        if error_callback:
            future.add_done_callback(
                lambda fut: error_callback(fut.exception()) if fut.exception() else None
            )
        
        return future
    
    @staticmethod
    async def run_with_progress(coro, parent, title, message, cancellable=True):
        """
        Run an asynchronous coroutine with a progress dialog.
        
        Args:
            coro: The coroutine to run
            parent: Parent widget for the progress dialog
            title: Title for the progress dialog
            message: Message to display in the progress dialog
            cancellable: Whether the operation can be cancelled
            
        Returns:
            The result of the coroutine
            
        Raises:
            AsyncError: If the operation is cancelled or fails
        """
        # Create progress dialog
        progress = QProgressDialog(message, "Cancel" if cancellable else None, 0, 0, parent)
        progress.setWindowTitle(title)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(500)  # Show after 500ms
        progress.setValue(0)
        progress.setMaximum(0)  # Indeterminate progress
        
        # Create a future for the coroutine
        future = asyncio.ensure_future(coro)
        
        # Set up cancellation if supported
        if cancellable:
            progress.canceled.connect(lambda: future.cancel())
        
        # Process events while waiting for the coroutine to complete
        while not future.done():
            QApplication.processEvents()
            await asyncio.sleep(0.1)
        
        # Close the progress dialog
        progress.close()
        
        # Handle cancellation
        if future.cancelled():
            raise AsyncError(
                "Operation cancelled by user",
                operation=title,
                issue="User cancelled the operation"
            )
        
        # Handle exceptions
        if future.exception():
            raise AsyncError(
                f"Operation failed: {str(future.exception())}",
                operation=title,
                issue="Exception during asynchronous operation",
                details=str(future.exception())
            ) from future.exception()
        
        # Return the result
        return future.result()


def main():
    """
    Main entry point for the ARMAX GUI application.
    
    This function initializes and runs the ARMAX application.
    """
    # Create and initialize the application
    app = ARMAXApplication()
    app.initialize(sys.argv)
    
    # Run the application
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
