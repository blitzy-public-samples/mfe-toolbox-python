#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARMAX GUI view implementation using PyQt6.

This module implements the main view for the ARMAX modeling interface,
providing a graphical user interface for time series analysis, model
specification, estimation, and visualization. It follows the Model-View-Controller
pattern, separating the presentation logic from business logic.

The view includes controls for setting model parameters, buttons for actions,
and visualization areas for time series data and model diagnostics. It uses
matplotlib for plotting embedded within PyQt6 widgets.
"""

import logging
from typing import Optional, List, Dict, Any, Tuple, Union, Callable, cast
import numpy as np
import pandas as pd
from pathlib import Path

# PyQt6 imports
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QCheckBox, QComboBox,
    QTabWidget, QGroupBox, QSplitter, QFrame, QScrollArea, QSizePolicy,
    QFileDialog, QMessageBox, QProgressBar, QStatusBar, QToolBar, QToolButton
)
from PyQt6.QtCore import (
    Qt, QSize, QTimer, pyqtSignal, pyqtSlot, QSettings, QEvent
)
from PyQt6.QtGui import (
    QIcon, QPixmap, QFont, QFontMetrics, QAction, QCloseEvent
)

# Matplotlib integration
import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt5Agg backend for matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Set up module-level logger
logger = logging.getLogger("mfe.ui.views.armax_view")


class MatplotlibCanvas(FigureCanvas):
    """
    Canvas for embedding matplotlib plots in PyQt6 widgets.
    
    This class provides a custom canvas for matplotlib figures that can be
    embedded in PyQt6 layouts, with proper sizing and event handling.
    
    Attributes:
        figure: The matplotlib Figure instance
    """
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        """
        Initialize the matplotlib canvas.
        
        Args:
            parent: Parent widget
            width: Figure width in inches
            height: Figure height in inches
            dpi: Dots per inch
        """
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.figure)
        self.setParent(parent)
        
        # Set up figure with tight layout
        self.figure.set_tight_layout(True)
        
        # Configure size policies
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.updateGeometry()


class ARMAXView(QMainWindow):
    """
    Main view for the ARMAX modeling interface.
    
    This class implements the main window and UI components for the ARMAX
    modeling interface, providing controls for model specification, estimation,
    and visualization. It emits signals when user actions occur, which are
    connected to controller methods for handling.
    
    Signals:
        ar_order_changed: Emitted when the AR order is changed
        ma_order_changed: Emitted when the MA order is changed
        constant_toggled: Emitted when the constant inclusion is toggled
        exog_variables_changed: Emitted when exogenous variables are changed
        estimate_clicked: Emitted when the estimate button is clicked
        reset_clicked: Emitted when the reset button is clicked
        about_clicked: Emitted when the about button is clicked
        load_data_clicked: Emitted when the load data button is clicked
        save_results_clicked: Emitted when the save results button is clicked
        forecast_clicked: Emitted when the forecast button is clicked
        cancel_clicked: Emitted when the cancel button is clicked
    """
    
    # Define signals for user interactions
    ar_order_changed = pyqtSignal(int)
    ma_order_changed = pyqtSignal(int)
    constant_toggled = pyqtSignal(bool)
    exog_variables_changed = pyqtSignal(list)
    estimate_clicked = pyqtSignal()
    reset_clicked = pyqtSignal()
    about_clicked = pyqtSignal()
    load_data_clicked = pyqtSignal()
    save_results_clicked = pyqtSignal()
    forecast_clicked = pyqtSignal()
    cancel_clicked = pyqtSignal()
    
    def __init__(self):
        """Initialize the ARMAX view."""
        super().__init__()
        
        # Set up window properties
        self.setWindowTitle("ARMAX Modeler - MFE Toolbox")
        self.setMinimumSize(800, 600)
        
        # Initialize UI components
        self._init_ui()
        
        # Connect internal signals
        self._connect_signals()
        
        logger.debug("ARMAX view initialized")
    
    def _init_ui(self):
        """Initialize the user interface components."""
        # Create central widget and main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        
        # Create left panel for controls
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create right panel for plots and results
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        self.right_layout.setContentsMargins(10, 10, 10, 10)
        
        # Add panels to main layout with splitter
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.left_panel)
        self.splitter.addWidget(self.right_panel)
        self.splitter.setStretchFactor(0, 1)  # Left panel gets 1/3
        self.splitter.setStretchFactor(1, 2)  # Right panel gets 2/3
        self.main_layout.addWidget(self.splitter)
        
        # Set up model specification group
        self._init_model_specification()
        
        # Set up action buttons
        self._init_action_buttons()
        
        # Set up plot areas
        self._init_plot_areas()
        
        # Set up tab widget for results
        self._init_results_tabs()
        
        # Set up status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Set up toolbar
        self._init_toolbar()
        
        # Set up progress bar (initially hidden)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        logger.debug("UI components initialized")
    
    def _init_model_specification(self):
        """Initialize the model specification group."""
        # Create model specification group
        self.model_spec_group = QGroupBox("Model Specification")
        self.model_spec_layout = QGridLayout(self.model_spec_group)
        
        # AR order controls
        self.ar_order_label = QLabel("AR Order:")
        self.ar_order_spin = QSpinBox()
        self.ar_order_spin.setRange(0, 20)
        self.ar_order_spin.setValue(0)
        self.ar_order_spin.setToolTip("Order of the autoregressive component")
        self.model_spec_layout.addWidget(self.ar_order_label, 0, 0)
        self.model_spec_layout.addWidget(self.ar_order_spin, 0, 1)
        
        # MA order controls
        self.ma_order_label = QLabel("MA Order:")
        self.ma_order_spin = QSpinBox()
        self.ma_order_spin.setRange(0, 20)
        self.ma_order_spin.setValue(0)
        self.ma_order_spin.setToolTip("Order of the moving average component")
        self.model_spec_layout.addWidget(self.ma_order_label, 1, 0)
        self.model_spec_layout.addWidget(self.ma_order_spin, 1, 1)
        
        # Include constant checkbox
        self.constant_check = QCheckBox("Include Constant")
        self.constant_check.setChecked(True)
        self.constant_check.setToolTip("Include a constant term in the model")
        self.model_spec_layout.addWidget(self.constant_check, 2, 0, 1, 2)
        
        # Exogenous variables controls
        self.exog_label = QLabel("Exogenous Variables:")
        self.exog_combo = QComboBox()
        self.exog_combo.setToolTip("Select exogenous variables to include")
        self.exog_combo.setEnabled(False)  # Disabled until data is loaded
        self.model_spec_layout.addWidget(self.exog_label, 3, 0)
        self.model_spec_layout.addWidget(self.exog_combo, 3, 1)
        
        # Estimation method controls
        self.method_label = QLabel("Estimation Method:")
        self.method_combo = QComboBox()
        self.method_combo.addItems(["CSS-MLE", "MLE", "CSS"])
        self.method_combo.setToolTip("Method for parameter estimation")
        self.model_spec_layout.addWidget(self.method_label, 4, 0)
        self.model_spec_layout.addWidget(self.method_combo, 4, 1)
        
        # Add model specification group to left panel
        self.left_layout.addWidget(self.model_spec_group)
        
        logger.debug("Model specification controls initialized")
    
    def _init_action_buttons(self):
        """Initialize the action buttons."""
        # Create action buttons group
        self.action_group = QGroupBox("Actions")
        self.action_layout = QVBoxLayout(self.action_group)
        
        # Estimate button
        self.estimate_button = QPushButton("Estimate Model")
        self.estimate_button.setToolTip("Estimate the specified model")
        self.action_layout.addWidget(self.estimate_button)
        
        # Reset button
        self.reset_button = QPushButton("Reset Parameters")
        self.reset_button.setToolTip("Reset model parameters to defaults")
        self.action_layout.addWidget(self.reset_button)
        
        # Load data button
        self.load_data_button = QPushButton("Load Data")
        self.load_data_button.setToolTip("Load time series data from file")
        self.action_layout.addWidget(self.load_data_button)
        
        # Save results button
        self.save_results_button = QPushButton("Save Results")
        self.save_results_button.setToolTip("Save model results to file")
        self.save_results_button.setEnabled(False)  # Disabled until model is estimated
        self.action_layout.addWidget(self.save_results_button)
        
        # Forecast button
        self.forecast_button = QPushButton("Generate Forecast")
        self.forecast_button.setToolTip("Generate forecasts from the estimated model")
        self.forecast_button.setEnabled(False)  # Disabled until model is estimated
        self.action_layout.addWidget(self.forecast_button)
        
        # Cancel button (initially hidden)
        self.cancel_button = QPushButton("Cancel Operation")
        self.cancel_button.setToolTip("Cancel the current operation")
        self.cancel_button.setVisible(False)
        self.action_layout.addWidget(self.cancel_button)
        
        # About button
        self.about_button = QPushButton("About")
        self.about_button.setToolTip("Show information about the application")
        self.action_layout.addWidget(self.about_button)
        
        # Add action group to left panel
        self.left_layout.addWidget(self.action_group)
        
        # Add stretch to push buttons to the top
        self.left_layout.addStretch(1)
        
        logger.debug("Action buttons initialized")
    
    def _init_plot_areas(self):
        """Initialize the plot areas."""
        # Create plot group
        self.plot_group = QGroupBox("Data Visualization")
        self.plot_layout = QVBoxLayout(self.plot_group)
        
        # Original data plot
        self.original_plot_label = QLabel("Original Data")
        self.original_plot_canvas = MatplotlibCanvas(width=5, height=3)
        self.original_plot = self.original_plot_canvas.figure.add_subplot(111)
        self.original_plot.set_title("Time Series Data")
        self.original_plot.set_xlabel("Time")
        self.original_plot.set_ylabel("Value")
        self.original_plot.grid(True)
        
        # Add toolbar for original plot
        self.original_plot_toolbar = NavigationToolbar(self.original_plot_canvas, self)
        
        # Add original plot components to layout
        self.plot_layout.addWidget(self.original_plot_label)
        self.plot_layout.addWidget(self.original_plot_canvas)
        self.plot_layout.addWidget(self.original_plot_toolbar)
        
        # Residual plot
        self.residual_plot_label = QLabel("Model Residuals")
        self.residual_plot_canvas = MatplotlibCanvas(width=5, height=3)
        self.residual_plot = self.residual_plot_canvas.figure.add_subplot(111)
        self.residual_plot.set_title("Model Residuals")
        self.residual_plot.set_xlabel("Time")
        self.residual_plot.set_ylabel("Residual")
        self.residual_plot.grid(True)
        
        # Add toolbar for residual plot
        self.residual_plot_toolbar = NavigationToolbar(self.residual_plot_canvas, self)
        
        # Add residual plot components to layout
        self.plot_layout.addWidget(self.residual_plot_label)
        self.plot_layout.addWidget(self.residual_plot_canvas)
        self.plot_layout.addWidget(self.residual_plot_toolbar)
        
        # Add plot group to right panel
        self.right_layout.addWidget(self.plot_group)
        
        logger.debug("Plot areas initialized")
    
    def _init_results_tabs(self):
        """Initialize the results tab widget."""
        # Create tab widget
        self.results_tabs = QTabWidget()
        
        # Model results tab
        self.model_results_tab = QWidget()
        self.model_results_layout = QVBoxLayout(self.model_results_tab)
        
        # Model equation display
        self.equation_group = QGroupBox("Model Equation")
        self.equation_layout = QVBoxLayout(self.equation_group)
        self.equation_label = QLabel("No model estimated yet")
        self.equation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.equation_label.setWordWrap(True)
        self.equation_layout.addWidget(self.equation_label)
        self.model_results_layout.addWidget(self.equation_group)
        
        # Parameter estimates display
        self.params_group = QGroupBox("Parameter Estimates")
        self.params_layout = QVBoxLayout(self.params_group)
        self.params_label = QLabel("No parameters estimated yet")
        self.params_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.params_layout.addWidget(self.params_label)
        self.model_results_layout.addWidget(self.params_group)
        
        # Add model results tab
        self.results_tabs.addTab(self.model_results_tab, "Model Results")
        
        # Diagnostics tab
        self.diagnostics_tab = QWidget()
        self.diagnostics_layout = QVBoxLayout(self.diagnostics_tab)
        
        # ACF/PACF plots
        self.acf_pacf_canvas = MatplotlibCanvas(width=5, height=6)
        self.acf_plot = self.acf_pacf_canvas.figure.add_subplot(211)
        self.acf_plot.set_title("Autocorrelation Function (ACF)")
        self.acf_plot.set_xlabel("Lag")
        self.acf_plot.set_ylabel("ACF")
        self.acf_plot.grid(True)
        
        self.pacf_plot = self.acf_pacf_canvas.figure.add_subplot(212)
        self.pacf_plot.set_title("Partial Autocorrelation Function (PACF)")
        self.pacf_plot.set_xlabel("Lag")
        self.pacf_plot.set_ylabel("PACF")
        self.pacf_plot.grid(True)
        
        # Add ACF/PACF toolbar
        self.acf_pacf_toolbar = NavigationToolbar(self.acf_pacf_canvas, self)
        
        # Add ACF/PACF components to layout
        self.diagnostics_layout.addWidget(self.acf_pacf_canvas)
        self.diagnostics_layout.addWidget(self.acf_pacf_toolbar)
        
        # Add diagnostics tab
        self.results_tabs.addTab(self.diagnostics_tab, "Diagnostics")
        
        # Statistics tab
        self.stats_tab = QWidget()
        self.stats_layout = QVBoxLayout(self.stats_tab)
        
        # Model statistics display
        self.stats_group = QGroupBox("Model Statistics")
        self.stats_layout_inner = QVBoxLayout(self.stats_group)
        self.stats_label = QLabel("No model statistics available")
        self.stats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stats_layout_inner.addWidget(self.stats_label)
        self.stats_layout.addWidget(self.stats_group)
        
        # Diagnostic tests display
        self.tests_group = QGroupBox("Diagnostic Tests")
        self.tests_layout = QVBoxLayout(self.tests_group)
        self.tests_label = QLabel("No diagnostic tests available")
        self.tests_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tests_layout.addWidget(self.tests_label)
        self.stats_layout.addWidget(self.tests_group)
        
        # Add statistics tab
        self.results_tabs.addTab(self.stats_tab, "Statistics")
        
        # Forecast tab
        self.forecast_tab = QWidget()
        self.forecast_layout = QVBoxLayout(self.forecast_tab)
        
        # Forecast plot
        self.forecast_canvas = MatplotlibCanvas(width=5, height=4)
        self.forecast_plot = self.forecast_canvas.figure.add_subplot(111)
        self.forecast_plot.set_title("Forecasts")
        self.forecast_plot.set_xlabel("Time")
        self.forecast_plot.set_ylabel("Value")
        self.forecast_plot.grid(True)
        
        # Add forecast toolbar
        self.forecast_toolbar = NavigationToolbar(self.forecast_canvas, self)
        
        # Add forecast components to layout
        self.forecast_layout.addWidget(self.forecast_canvas)
        self.forecast_layout.addWidget(self.forecast_toolbar)
        
        # Add forecast tab
        self.results_tabs.addTab(self.forecast_tab, "Forecasts")
        
        # Add results tabs to right panel
        self.right_layout.addWidget(self.results_tabs)
        
        logger.debug("Results tabs initialized")
    
    def _init_toolbar(self):
        """Initialize the toolbar."""
        # Create toolbar
        self.toolbar = QToolBar("Main Toolbar")
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolbar)
        
        # Load data action
        self.load_data_action = QAction(QIcon.fromTheme("document-open"), "Load Data", self)
        self.load_data_action.setStatusTip("Load time series data from file")
        self.toolbar.addAction(self.load_data_action)
        
        # Save results action
        self.save_results_action = QAction(QIcon.fromTheme("document-save"), "Save Results", self)
        self.save_results_action.setStatusTip("Save model results to file")
        self.save_results_action.setEnabled(False)  # Disabled until model is estimated
        self.toolbar.addAction(self.save_results_action)
        
        # Add separator
        self.toolbar.addSeparator()
        
        # Estimate action
        self.estimate_action = QAction(QIcon.fromTheme("system-run"), "Estimate Model", self)
        self.estimate_action.setStatusTip("Estimate the specified model")
        self.toolbar.addAction(self.estimate_action)
        
        # Forecast action
        self.forecast_action = QAction(QIcon.fromTheme("go-next"), "Generate Forecast", self)
        self.forecast_action.setStatusTip("Generate forecasts from the estimated model")
        self.forecast_action.setEnabled(False)  # Disabled until model is estimated
        self.toolbar.addAction(self.forecast_action)
        
        # Add separator
        self.toolbar.addSeparator()
        
        # Reset action
        self.reset_action = QAction(QIcon.fromTheme("edit-clear"), "Reset Parameters", self)
        self.reset_action.setStatusTip("Reset model parameters to defaults")
        self.toolbar.addAction(self.reset_action)
        
        # Add separator
        self.toolbar.addSeparator()
        
        # About action
        self.about_action = QAction(QIcon.fromTheme("help-about"), "About", self)
        self.about_action.setStatusTip("Show information about the application")
        self.toolbar.addAction(self.about_action)
        
        logger.debug("Toolbar initialized")
    
    def _connect_signals(self):
        """Connect internal signals to slots."""
        # Connect model specification controls
        self.ar_order_spin.valueChanged.connect(self._on_ar_order_changed)
        self.ma_order_spin.valueChanged.connect(self._on_ma_order_changed)
        self.constant_check.toggled.connect(self._on_constant_toggled)
        self.exog_combo.currentTextChanged.connect(self._on_exog_variables_changed)
        
        # Connect action buttons
        self.estimate_button.clicked.connect(self._on_estimate_clicked)
        self.reset_button.clicked.connect(self._on_reset_clicked)
        self.about_button.clicked.connect(self._on_about_clicked)
        self.load_data_button.clicked.connect(self._on_load_data_clicked)
        self.save_results_button.clicked.connect(self._on_save_results_clicked)
        self.forecast_button.clicked.connect(self._on_forecast_clicked)
        self.cancel_button.clicked.connect(self._on_cancel_clicked)
        
        # Connect toolbar actions
        self.load_data_action.triggered.connect(self._on_load_data_clicked)
        self.save_results_action.triggered.connect(self._on_save_results_clicked)
        self.estimate_action.triggered.connect(self._on_estimate_clicked)
        self.forecast_action.triggered.connect(self._on_forecast_clicked)
        self.reset_action.triggered.connect(self._on_reset_clicked)
        self.about_action.triggered.connect(self._on_about_clicked)
        
        logger.debug("Internal signals connected")
    
    def closeEvent(self, event: QCloseEvent):
        """
        Handle window close event.
        
        Args:
            event: Close event
        """
        # Show confirmation dialog
        reply = QMessageBox.question(
            self, "Confirm Close",
            "Are you sure you want to close the ARMAX GUI?\n\nAll unsaved changes will be lost.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Accept the close event
            event.accept()
            logger.info("ARMAX view closed by user")
        else:
            # Reject the close event
            event.ignore()
            logger.debug("Close event rejected by user")
    
    # Signal handlers (slots)
    @pyqtSlot(int)
    def _on_ar_order_changed(self, value: int):
        """
        Handle AR order change.
        
        Args:
            value: New AR order
        """
        self.ar_order_changed.emit(value)
        logger.debug(f"AR order changed to {value}")
    
    @pyqtSlot(int)
    def _on_ma_order_changed(self, value: int):
        """
        Handle MA order change.
        
        Args:
            value: New MA order
        """
        self.ma_order_changed.emit(value)
        logger.debug(f"MA order changed to {value}")
    
    @pyqtSlot(bool)
    def _on_constant_toggled(self, checked: bool):
        """
        Handle constant toggle.
        
        Args:
            checked: Whether constant is included
        """
        self.constant_toggled.emit(checked)
        logger.debug(f"Constant toggled to {checked}")
    
    @pyqtSlot(str)
    def _on_exog_variables_changed(self, text: str):
        """
        Handle exogenous variables change.
        
        Args:
            text: Selected exogenous variable
        """
        # Get all selected exogenous variables
        selected_vars = []
        for i in range(self.exog_combo.count()):
            if self.exog_combo.itemData(i, Qt.ItemDataRole.CheckStateRole) == Qt.CheckState.Checked:
                selected_vars.append(self.exog_combo.itemText(i))
        
        self.exog_variables_changed.emit(selected_vars)
        logger.debug(f"Exogenous variables changed to {selected_vars}")
    
    @pyqtSlot()
    def _on_estimate_clicked(self):
        """Handle estimate button click."""
        self.estimate_clicked.emit()
        logger.debug("Estimate button clicked")
    
    @pyqtSlot()
    def _on_reset_clicked(self):
        """Handle reset button click."""
        self.reset_clicked.emit()
        logger.debug("Reset button clicked")
    
    @pyqtSlot()
    def _on_about_clicked(self):
        """Handle about button click."""
        self.about_clicked.emit()
        logger.debug("About button clicked")
    
    @pyqtSlot()
    def _on_load_data_clicked(self):
        """Handle load data button click."""
        self.load_data_clicked.emit()
        logger.debug("Load data button clicked")
    
    @pyqtSlot()
    def _on_save_results_clicked(self):
        """Handle save results button click."""
        self.save_results_clicked.emit()
        logger.debug("Save results button clicked")
    
    @pyqtSlot()
    def _on_forecast_clicked(self):
        """Handle forecast button click."""
        self.forecast_clicked.emit()
        logger.debug("Forecast button clicked")
    
    @pyqtSlot()
    def _on_cancel_clicked(self):
        """Handle cancel button click."""
        self.cancel_clicked.emit()
        logger.debug("Cancel button clicked")
    
    # Public methods for controller to call
    def set_ar_order(self, value: int):
        """
        Set the AR order.
        
        Args:
            value: AR order
        """
        self.ar_order_spin.setValue(value)
    
    def get_ar_order(self) -> int:
        """
        Get the AR order.
        
        Returns:
            int: AR order
        """
        return self.ar_order_spin.value()
    
    def set_ma_order(self, value: int):
        """
        Set the MA order.
        
        Args:
            value: MA order
        """
        self.ma_order_spin.setValue(value)
    
    def get_ma_order(self) -> int:
        """
        Get the MA order.
        
        Returns:
            int: MA order
        """
        return self.ma_order_spin.value()
    
    def set_include_constant(self, value: bool):
        """
        Set whether to include a constant term.
        
        Args:
            value: Whether to include a constant term
        """
        self.constant_check.setChecked(value)
    
    def get_include_constant(self) -> bool:
        """
        Get whether to include a constant term.
        
        Returns:
            bool: Whether to include a constant term
        """
        return self.constant_check.isChecked()
    
    def set_exog_variables(self, variables: List[str], selected: List[str] = None):
        """
        Set the available exogenous variables.
        
        Args:
            variables: List of available exogenous variables
            selected: List of selected exogenous variables
        """
        # Clear existing items
        self.exog_combo.clear()
        
        if not variables:
            self.exog_combo.setEnabled(False)
            return
        
        # Add variables as checkable items
        for var in variables:
            self.exog_combo.addItem(var)
            index = self.exog_combo.count() - 1
            self.exog_combo.setItemData(
                index,
                Qt.CheckState.Checked if selected and var in selected else Qt.CheckState.Unchecked,
                Qt.ItemDataRole.CheckStateRole
            )
        
        self.exog_combo.setEnabled(True)
    
    def get_exog_variables(self) -> List[str]:
        """
        Get the selected exogenous variables.
        
        Returns:
            List[str]: Selected exogenous variables
        """
        selected_vars = []
        for i in range(self.exog_combo.count()):
            if self.exog_combo.itemData(i, Qt.ItemDataRole.CheckStateRole) == Qt.CheckState.Checked:
                selected_vars.append(self.exog_combo.itemText(i))
        
        return selected_vars
    
    def set_estimation_method(self, method: str):
        """
        Set the estimation method.
        
        Args:
            method: Estimation method
        """
        index = self.method_combo.findText(method.upper())
        if index >= 0:
            self.method_combo.setCurrentIndex(index)
    
    def get_estimation_method(self) -> str:
        """
        Get the estimation method.
        
        Returns:
            str: Estimation method
        """
        return self.method_combo.currentText().lower()
    
    def plot_time_series(self, data: np.ndarray, title: str = "Time Series Data", 
                        xlabel: str = "Time", ylabel: str = "Value"):
        """
        Plot time series data.
        
        Args:
            data: Time series data
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
        """
        # Clear the plot
        self.original_plot.clear()
        
        # Plot the data
        x = np.arange(len(data))
        self.original_plot.plot(x, data)
        
        # Set labels and title
        self.original_plot.set_title(title)
        self.original_plot.set_xlabel(xlabel)
        self.original_plot.set_ylabel(ylabel)
        self.original_plot.grid(True)
        
        # Redraw the canvas
        self.original_plot_canvas.draw()
    
    def plot_residuals(self, residuals: np.ndarray, title: str = "Model Residuals", 
                      xlabel: str = "Time", ylabel: str = "Residual"):
        """
        Plot model residuals.
        
        Args:
            residuals: Model residuals
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
        """
        # Clear the plot
        self.residual_plot.clear()
        
        # Plot the residuals
        x = np.arange(len(residuals))
        self.residual_plot.plot(x, residuals)
        
        # Add a horizontal line at y=0
        self.residual_plot.axhline(y=0, color='r', linestyle='-')
        
        # Set labels and title
        self.residual_plot.set_title(title)
        self.residual_plot.set_xlabel(xlabel)
        self.residual_plot.set_ylabel(ylabel)
        self.residual_plot.grid(True)
        
        # Redraw the canvas
        self.residual_plot_canvas.draw()
    
    def plot_acf_pacf(self, acf: np.ndarray, pacf: np.ndarray, lags: np.ndarray = None):
        """
        Plot autocorrelation and partial autocorrelation functions.
        
        Args:
            acf: Autocorrelation function values
            pacf: Partial autocorrelation function values
            lags: Lag values
        """
        # Clear the plots
        self.acf_plot.clear()
        self.pacf_plot.clear()
        
        # Create lag values if not provided
        if lags is None:
            lags = np.arange(len(acf))
        
        # Plot ACF
        self.acf_plot.bar(lags, acf, width=0.3)
        self.acf_plot.axhline(y=0, color='k', linestyle='-')
        
        # Add confidence intervals (±1.96/sqrt(n))
        if len(acf) > 1:
            ci = 1.96 / np.sqrt(len(acf))
            self.acf_plot.axhline(y=ci, color='r', linestyle='--')
            self.acf_plot.axhline(y=-ci, color='r', linestyle='--')
        
        # Set ACF labels and title
        self.acf_plot.set_title("Autocorrelation Function (ACF)")
        self.acf_plot.set_xlabel("Lag")
        self.acf_plot.set_ylabel("ACF")
        self.acf_plot.grid(True)
        
        # Plot PACF
        self.pacf_plot.bar(lags, pacf, width=0.3)
        self.pacf_plot.axhline(y=0, color='k', linestyle='-')
        
        # Add confidence intervals (±1.96/sqrt(n))
        if len(pacf) > 1:
            ci = 1.96 / np.sqrt(len(pacf))
            self.pacf_plot.axhline(y=ci, color='r', linestyle='--')
            self.pacf_plot.axhline(y=-ci, color='r', linestyle='--')
        
        # Set PACF labels and title
        self.pacf_plot.set_title("Partial Autocorrelation Function (PACF)")
        self.pacf_plot.set_xlabel("Lag")
        self.pacf_plot.set_ylabel("PACF")
        self.pacf_plot.grid(True)
        
        # Adjust layout
        self.acf_pacf_canvas.figure.tight_layout()
        
        # Redraw the canvas
        self.acf_pacf_canvas.draw()
        
        # Switch to diagnostics tab
        self.results_tabs.setCurrentIndex(1)  # Index 1 is the diagnostics tab
    
    def plot_forecast(self, data: np.ndarray, forecasts: np.ndarray, 
                     lower_bounds: np.ndarray = None, upper_bounds: np.ndarray = None,
                     title: str = "Forecasts", xlabel: str = "Time", ylabel: str = "Value"):
        """
        Plot forecasts with confidence intervals.
        
        Args:
            data: Original time series data
            forecasts: Forecast values
            lower_bounds: Lower confidence bounds
            upper_bounds: Upper confidence bounds
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
        """
        # Clear the plot
        self.forecast_plot.clear()
        
        # Create time indices
        x_data = np.arange(len(data))
        x_forecast = np.arange(len(data), len(data) + len(forecasts))
        
        # Plot original data
        self.forecast_plot.plot(x_data, data, 'b-', label='Original Data')
        
        # Plot forecasts
        self.forecast_plot.plot(x_forecast, forecasts, 'r-', label='Forecast')
        
        # Plot confidence intervals if provided
        if lower_bounds is not None and upper_bounds is not None:
            self.forecast_plot.fill_between(
                x_forecast, lower_bounds, upper_bounds,
                color='r', alpha=0.2, label='95% Confidence Interval'
            )
        
        # Set labels and title
        self.forecast_plot.set_title(title)
        self.forecast_plot.set_xlabel(xlabel)
        self.forecast_plot.set_ylabel(ylabel)
        self.forecast_plot.grid(True)
        self.forecast_plot.legend()
        
        # Redraw the canvas
        self.forecast_canvas.draw()
        
        # Switch to forecast tab
        self.results_tabs.setCurrentIndex(3)  # Index 3 is the forecast tab
    
    def display_model_equation(self, equation: str):
        """
        Display the model equation.
        
        Args:
            equation: Model equation string
        """
        self.equation_label.setText(equation)
        
        # Switch to model results tab
        self.results_tabs.setCurrentIndex(0)  # Index 0 is the model results tab
    
    def display_parameter_estimates(self, params_df: pd.DataFrame):
        """
        Display parameter estimates.
        
        Args:
            params_df: DataFrame containing parameter estimates
        """
        if params_df is None or params_df.empty:
            self.params_label.setText("No parameters estimated yet")
            return
        
        # Convert DataFrame to HTML table
        html = "<table border='1' cellpadding='3' cellspacing='0'>"
        
        # Add header row
        html += "<tr>"
        for col in params_df.columns:
            html += f"<th>{col}</th>"
        html += "</tr>"
        
        # Add data rows
        for _, row in params_df.iterrows():
            html += "<tr>"
            for i, col in enumerate(params_df.columns):
                value = row[col]
                
                # Format p-values with asterisks for significance
                if col == 'p-Value':
                    if value < 0.01:
                        html += f"<td>{value:.4f} ***</td>"
                    elif value < 0.05:
                        html += f"<td>{value:.4f} **</td>"
                    elif value < 0.1:
                        html += f"<td>{value:.4f} *</td>"
                    else:
                        html += f"<td>{value:.4f}</td>"
                # Format numeric values
                elif isinstance(value, (int, float)):
                    html += f"<td>{value:.4f}</td>"
                else:
                    html += f"<td>{value}</td>"
            html += "</tr>"
        
        html += "</table>"
        
        # Add significance legend
        html += "<p><small>Significance levels: *** p<0.01, ** p<0.05, * p<0.1</small></p>"
        
        self.params_label.setText(html)
        
        # Switch to model results tab
        self.results_tabs.setCurrentIndex(0)  # Index 0 is the model results tab
    
    def display_model_statistics(self, stats: Dict[str, float]):
        """
        Display model statistics.
        
        Args:
            stats: Dictionary of model statistics
        """
        if not stats:
            self.stats_label.setText("No model statistics available")
            return
        
        # Convert dictionary to HTML table
        html = "<table border='1' cellpadding='3' cellspacing='0'>"
        
        # Add header row
        html += "<tr><th>Statistic</th><th>Value</th></tr>"
        
        # Add data rows
        for stat, value in stats.items():
            if isinstance(value, (int, float)):
                html += f"<tr><td>{stat}</td><td>{value:.6f}</td></tr>"
            else:
                html += f"<tr><td>{stat}</td><td>{value}</td></tr>"
        
        html += "</table>"
        
        self.stats_label.setText(html)
        
        # Switch to statistics tab
        self.results_tabs.setCurrentIndex(2)  # Index 2 is the statistics tab
    
    def display_diagnostic_tests(self, tests: Dict[str, Any]):
        """
        Display diagnostic test results.
        
        Args:
            tests: Dictionary of diagnostic test results
        """
        if not tests:
            self.tests_label.setText("No diagnostic tests available")
            return
        
        # Convert dictionary to HTML
        html = ""
        
        for test_name, test_result in tests.items():
            html += f"<h3>{test_name}</h3>"
            
            if isinstance(test_result, dict):
                # Create table for dictionary results
                html += "<table border='1' cellpadding='3' cellspacing='0'>"
                
                for key, value in test_result.items():
                    if isinstance(value, list):
                        # Format list values
                        values_str = ", ".join([f"{v:.4f}" if isinstance(v, (int, float)) else str(v) for v in value])
                        html += f"<tr><td>{key}</td><td>{values_str}</td></tr>"
                    elif isinstance(value, (int, float)):
                        # Format numeric values
                        html += f"<tr><td>{key}</td><td>{value:.6f}</td></tr>"
                    else:
                        # Format other values
                        html += f"<tr><td>{key}</td><td>{value}</td></tr>"
                
                html += "</table>"
            else:
                # Display single value
                if isinstance(test_result, (int, float)):
                    html += f"<p>{test_result:.6f}</p>"
                else:
                    html += f"<p>{test_result}</p>"
            
            html += "<br>"
        
        self.tests_label.setText(html)
        
        # Switch to statistics tab
        self.results_tabs.setCurrentIndex(2)  # Index 2 is the statistics tab
    
    def show_progress(self, value: int, message: str):
        """
        Show progress in the status bar.
        
        Args:
            value: Progress value (0-100)
            message: Progress message
        """
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(value)
        
        # Show message in status bar
        self.status_bar.showMessage(message)
        
        # Show cancel button
        self.cancel_button.setVisible(True)
        
        # Disable estimate button during operation
        self.estimate_button.setEnabled(False)
        self.estimate_action.setEnabled(False)
        
        # Process events to update UI
        QApplication.processEvents()
    
    def hide_progress(self):
        """Hide the progress bar and reset status."""
        # Hide progress bar
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        
        # Reset status bar
        self.status_bar.showMessage("Ready")
        
        # Hide cancel button
        self.cancel_button.setVisible(False)
        
        # Enable estimate button
        self.estimate_button.setEnabled(True)
        self.estimate_action.setEnabled(True)
        
        # Process events to update UI
        QApplication.processEvents()
    
    def enable_results_actions(self, enabled: bool = True):
        """
        Enable or disable results-related actions.
        
        Args:
            enabled: Whether to enable the actions
        """
        self.save_results_button.setEnabled(enabled)
        self.save_results_action.setEnabled(enabled)
        self.forecast_button.setEnabled(enabled)
        self.forecast_action.setEnabled(enabled)
    
    def show_error_message(self, title: str, message: str, details: str = None):
        """
        Show an error message dialog.
        
        Args:
            title: Dialog title
            message: Error message
            details: Detailed error information
        """
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        
        if details:
            msg_box.setDetailedText(details)
        
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg_box.exec()
    
    def show_info_message(self, title: str, message: str):
        """
        Show an information message dialog.
        
        Args:
            title: Dialog title
            message: Information message
        """
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Icon.Information)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg_box.exec()
    
    def show_file_dialog(self, mode: str = "open", title: str = None, 
                        filter_str: str = None, default_dir: str = None) -> Optional[str]:
        """
        Show a file dialog for opening or saving files.
        
        Args:
            mode: Dialog mode ("open" or "save")
            title: Dialog title
            filter_str: File filter string
            default_dir: Default directory
            
        Returns:
            str: Selected file path or None if cancelled
        """
        if default_dir is None:
            default_dir = str(Path.home())
        
        if mode.lower() == "open":
            if title is None:
                title = "Open File"
            
            if filter_str is None:
                filter_str = "CSV Files (*.csv);;Text Files (*.txt);;All Files (*.*)"
            
            file_path, _ = QFileDialog.getOpenFileName(
                self, title, default_dir, filter_str
            )
        elif mode.lower() == "save":
            if title is None:
                title = "Save File"
            
            if filter_str is None:
                filter_str = "CSV Files (*.csv);;Text Files (*.txt);;All Files (*.*)"
            
            file_path, _ = QFileDialog.getSaveFileName(
                self, title, default_dir, filter_str
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        return file_path if file_path else None
