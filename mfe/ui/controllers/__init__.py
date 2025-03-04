#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MFE Toolbox UI Controllers Package

This package contains controller classes that implement the Controller component
of the Model-View-Controller (MVC) pattern for the MFE Toolbox user interface.
Controllers handle the interaction between models (data) and views (presentation),
managing user inputs, coordinating model operations, and updating the UI accordingly.

The controllers leverage PyQt6's signal-slot mechanism for event handling and
Python's async/await pattern for long-running operations, ensuring the UI remains
responsive during computationally intensive tasks.

Available Controllers:
    ARMAXController: Controller for the ARMAX modeling interface
    AboutDialogController: Controller for the About dialog
    CloseDialogController: Controller for the close confirmation dialog
    ModelViewerController: Controller for the model results viewer

Functions:
    show_about_dialog: Convenience function to show the About dialog
"""

import logging
from typing import Optional

# Import controller classes
from mfe.ui.controllers.armax_controller import ARMAXController
from mfe.ui.controllers.about_dialog_controller import AboutDialogController, show_about_dialog
from mfe.ui.controllers.close_dialog_controller import CloseDialogController
from mfe.ui.controllers.model_viewer_controller import ModelViewerController

# Set up package-level logger
logger = logging.getLogger("mfe.ui.controllers")

# Version information
__version__ = "4.0.0"

# Export public API
__all__ = [
    'ARMAXController',
    'AboutDialogController',
    'CloseDialogController',
    'ModelViewerController',
    'show_about_dialog',
]
