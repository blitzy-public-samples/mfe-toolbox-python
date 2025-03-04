#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARMAX model results viewer view using PyQt6.

This module implements the view component for displaying ARMAX model results,
including model equations, parameter estimates, t-statistics, p-values, and
other model diagnostics in a structured layout. It supports pagination for
large parameter tables and LaTeX rendering for equations.

The ModelViewerView class follows the Model-View-Controller pattern, providing
the presentation layer for model results while delegating business logic to
the controller.
"""

import logging
from typing import Optional, List, Dict, Any, Tuple, Union, Callable
import numpy as np
import pandas as pd
from pathlib import Path

# PyQt6 imports
from PyQt6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QGroupBox, QFrame, QScrollArea, QSizePolicy, QSpacerItem,
    QTabWidget, QApplication
)
from PyQt6.QtCore import (
    Qt, QSize, pyqtSignal, pyqtSlot
)
from PyQt6.QtGui import (
    QIcon, QFont, QCloseEvent, QColor
)

# Matplotlib integration for LaTeX rendering
import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt5Agg backend for matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import utility functions
from mfe.ui.utils import (
    format_latex_equation, create_equation_figure, create_parameter_table,
    create_statistics_table, center_widget_on_parent, show_error_message,
    format_model_equation
)

# Set up module-level logger
logger = logging.getLogger("mfe.ui.views.model_viewer_view")


class ModelViewerView(QDialog):
    """
    Dialog for displaying ARMAX model results.
    
    This class implements a dialog for displaying detailed model results,
    including model equations, parameter estimates, t-statistics, p-values,
    and other model diagnostics in a structured layout. It supports pagination
    for large parameter tables and LaTeX rendering for equations.
    
    Signals:
        close_clicked: Emitted when the close button is clicked
        next_page_clicked: Emitted when the next page button is clicked
        prev_page_clicked: Emitted when the previous page button is clicked
        copy_to_clipboard_clicked: Emitted when the copy to clipboard button is clicked
    """
    
    # Define signals for user interactions
    close_clicked = pyqtSignal()
    next_page_clicked = pyqtSignal()
    prev_page_clicked = pyqtSignal()
    copy_to_clipboard_clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        """
        Initialize the model viewer dialog.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Set up window properties
        self.setWindowTitle("ARMAX Model Results")
        self.setMinimumSize(700, 500)
        self.resize(800, 600)
        
        # Initialize UI components
        self._init_ui()
        
        # Connect internal signals
        self._connect_signals()
        
        # Center the dialog on the parent
        if parent:
            center_widget_on_parent(self, parent)
        
        logger.debug("ModelViewerView initialized")
    
    def _init_ui(self):
        """Initialize the user interface components."""
        # Create main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)
        
        # Create tab widget for different result views
        self.tab_widget = QTabWidget()
        
        # Create overview tab
        self.overview_tab = QWidget()
        self.overview_layout = QVBoxLayout(self.overview_tab)
        self.overview_layout.setContentsMargins(10, 10, 10, 10)
        self.overview_layout.setSpacing(10)
        
        # Create equation group
        self.equation_group = QGroupBox("Model Equation")
        self.equation_layout = QVBoxLayout(self.equation_group)
        
        # Create equation canvas for LaTeX rendering
        self.equation_canvas_container = QWidget()
        self.equation_canvas_layout = QVBoxLayout(self.equation_canvas_container)
        self.equation_canvas_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create a placeholder for the equation canvas
        self.equation_canvas = None
        self.equation_placeholder = QLabel("No model equation available")
        self.equation_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.equation_canvas_layout.addWidget(self.equation_placeholder)
        
        self.equation_layout.addWidget(self.equation_canvas_container)
        self.overview_layout.addWidget(self.equation_group)
        
        # Create parameters group
        self.params_group = QGroupBox("Parameter Estimates")
        self.params_layout = QVBoxLayout(self.params_group)
        
        # Create parameter table
        self.params_table = QTableWidget()
        self.params_table.setColumnCount(5)
        self.params_table.setHorizontalHeaderLabels(
            ["Parameter", "Estimate", "Std. Error", "t-Stat", "p-Value"]
        )
        self.params_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.params_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.params_table.setAlternatingRowColors(True)
        self.params_layout.addWidget(self.params_table)
        
        # Create pagination controls
        self.pagination_widget = QWidget()
        self.pagination_layout = QHBoxLayout(self.pagination_widget)
        self.pagination_layout.setContentsMargins(0, 0, 0, 0)
        
        self.prev_page_button = QPushButton("< Previous")
        self.prev_page_button.setEnabled(False)
        self.pagination_layout.addWidget(self.prev_page_button)
        
        self.page_label = QLabel("Page 1 of 1")
        self.page_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pagination_layout.addWidget(self.page_label)
        
        self.next_page_button = QPushButton("Next >")
        self.next_page_button.setEnabled(False)
        self.pagination_layout.addWidget(self.next_page_button)
        
        self.params_layout.addWidget(self.pagination_widget)
        
        # Add significance legend
        self.significance_label = QLabel(
            "<small>Significance levels: [***] p<0.01, [**] p<0.05, [*] p<0.1</small>"
        )
        self.significance_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.params_layout.addWidget(self.significance_label)
        
        self.overview_layout.addWidget(self.params_group)
        
        # Create statistics tab
        self.stats_tab = QWidget()
        self.stats_layout = QVBoxLayout(self.stats_tab)
        self.stats_layout.setContentsMargins(10, 10, 10, 10)
        self.stats_layout.setSpacing(10)
        
        # Create model statistics group
        self.stats_group = QGroupBox("Model Statistics")
        self.stats_group_layout = QVBoxLayout(self.stats_group)
        
        # Create statistics table
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(3)
        self.stats_table.setHorizontalHeaderLabels(
            ["Statistic", "Value", "Notes"]
        )
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.stats_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.stats_table.setAlternatingRowColors(True)
        self.stats_group_layout.addWidget(self.stats_table)
        
        self.stats_layout.addWidget(self.stats_group)
        
        # Create diagnostic tests group
        self.tests_group = QGroupBox("Diagnostic Tests")
        self.tests_group_layout = QVBoxLayout(self.tests_group)
        
        # Create diagnostic tests table
        self.tests_table = QTableWidget()
        self.tests_table.setColumnCount(3)
        self.tests_table.setHorizontalHeaderLabels(
            ["Test", "Value", "p-Value"]
        )
        self.tests_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tests_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.tests_table.setAlternatingRowColors(True)
        self.tests_group_layout.addWidget(self.tests_table)
        
        self.stats_layout.addWidget(self.tests_group)
        
        # Add tabs to tab widget
        self.tab_widget.addTab(self.overview_tab, "Overview")
        self.tab_widget.addTab(self.stats_tab, "Statistics")
        
        # Add tab widget to main layout
        self.main_layout.addWidget(self.tab_widget)
        
        # Create button layout
        self.button_layout = QHBoxLayout()
        
        # Add copy to clipboard button
        self.copy_button = QPushButton("Copy to Clipboard")
        self.copy_button.setToolTip("Copy results to clipboard")
        self.button_layout.addWidget(self.copy_button)
        
        # Add spacer
        self.button_layout.addStretch(1)
        
        # Add close button
        self.close_button = QPushButton("Close")
        self.close_button.setToolTip("Close the model viewer")
        self.button_layout.addWidget(self.close_button)
        
        # Add button layout to main layout
        self.main_layout.addLayout(self.button_layout)
        
        logger.debug("ModelViewerView UI components initialized")
    
    def _connect_signals(self):
        """Connect internal signals to slots."""
        # Connect button signals
        self.close_button.clicked.connect(self._on_close_clicked)
        self.prev_page_button.clicked.connect(self._on_prev_page_clicked)
        self.next_page_button.clicked.connect(self._on_next_page_clicked)
        self.copy_button.clicked.connect(self._on_copy_to_clipboard_clicked)
        
        logger.debug("ModelViewerView internal signals connected")
    
    def closeEvent(self, event: QCloseEvent):
        """
        Handle window close event.
        
        Args:
            event: Close event
        """
        # Accept the close event
        event.accept()
        logger.debug("ModelViewerView closed")
    
    # Signal handlers (slots)
    @pyqtSlot()
    def _on_close_clicked(self):
        """Handle close button click."""
        self.close_clicked.emit()
        self.close()
        logger.debug("Close button clicked")
    
    @pyqtSlot()
    def _on_prev_page_clicked(self):
        """Handle previous page button click."""
        self.prev_page_clicked.emit()
        logger.debug("Previous page button clicked")
    
    @pyqtSlot()
    def _on_next_page_clicked(self):
        """Handle next page button click."""
        self.next_page_clicked.emit()
        logger.debug("Next page button clicked")
    
    @pyqtSlot()
    def _on_copy_to_clipboard_clicked(self):
        """Handle copy to clipboard button click."""
        self.copy_to_clipboard_clicked.emit()
        logger.debug("Copy to clipboard button clicked")
    
    # Public methods for controller to call
    def display_model_equation(self, equation: str):
        """
        Display the model equation using LaTeX rendering.
        
        Args:
            equation: LaTeX equation string
        """
        try:
            # Clear existing canvas if any
            if self.equation_canvas is not None:
                self.equation_canvas_layout.removeWidget(self.equation_canvas)
                self.equation_canvas.deleteLater()
                self.equation_canvas = None
            
            # Hide the placeholder
            self.equation_placeholder.setVisible(False)
            
            # Create a new figure with the equation
            fig = create_equation_figure(equation, figsize=(6, 1.5), dpi=100, fontsize=14)
            
            # Create a canvas for the figure
            self.equation_canvas = FigureCanvas(fig)
            self.equation_canvas.setMinimumHeight(100)
            self.equation_canvas.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
            )
            
            # Add the canvas to the layout
            self.equation_canvas_layout.addWidget(self.equation_canvas)
            
            logger.debug("Model equation displayed")
        except Exception as e:
            logger.error(f"Error displaying model equation: {e}")
            self.equation_placeholder.setText(f"Error displaying equation: {str(e)}")
            self.equation_placeholder.setVisible(True)
    
    def display_parameters(self, parameters: List[Dict[str, Any]], 
                          current_page: int = 1, total_pages: int = 1):
        """
        Display parameter estimates in the table with pagination.
        
        Args:
            parameters: List of parameter dictionaries
            current_page: Current page number (1-based)
            total_pages: Total number of pages
        """
        try:
            # Update the parameter table
            current_page, total_pages = create_parameter_table(
                parameters, self.params_table, current_page, 10
            )
            
            # Update pagination controls
            self.page_label.setText(f"Page {current_page} of {total_pages}")
            self.prev_page_button.setEnabled(current_page > 1)
            self.next_page_button.setEnabled(current_page < total_pages)
            
            logger.debug(f"Parameters displayed (page {current_page}/{total_pages})")
        except Exception as e:
            logger.error(f"Error displaying parameters: {e}")
            self.params_table.setRowCount(1)
            self.params_table.setColumnCount(1)
            self.params_table.setHorizontalHeaderLabels(["Error"])
            error_item = QTableWidgetItem(f"Error loading parameters: {str(e)}")
            error_item.setForeground(QColor(255, 0, 0))
            self.params_table.setItem(0, 0, error_item)
    
    def display_statistics(self, statistics: Dict[str, Any]):
        """
        Display model statistics in the statistics table.
        
        Args:
            statistics: Dictionary of model statistics
        """
        try:
            # Update the statistics table
            create_statistics_table(statistics, self.stats_table)
            
            logger.debug("Statistics displayed")
        except Exception as e:
            logger.error(f"Error displaying statistics: {e}")
            self.stats_table.setRowCount(1)
            self.stats_table.setColumnCount(1)
            self.stats_table.setHorizontalHeaderLabels(["Error"])
            error_item = QTableWidgetItem(f"Error loading statistics: {str(e)}")
            error_item.setForeground(QColor(255, 0, 0))
            self.stats_table.setItem(0, 0, error_item)
    
    def display_diagnostic_tests(self, tests: Dict[str, Any]):
        """
        Display diagnostic test results in the tests table.
        
        Args:
            tests: Dictionary of diagnostic test results
        """
        try:
            # Clear the table
            self.tests_table.setRowCount(0)
            
            # Add tests to the table
            row = 0
            for test_name, test_result in tests.items():
                if isinstance(test_result, dict):
                    # For tests with multiple statistics
                    for key, value in test_result.items():
                        if key != "Lags":  # Skip lag information
                            if isinstance(value, list):
                                # For lists of values (e.g., Q-statistics at different lags)
                                for i, val in enumerate(value):
                                    lag = test_result.get("Lags", [])[i] if "Lags" in test_result and i < len(test_result["Lags"]) else i + 1
                                    
                                    self.tests_table.insertRow(row)
                                    self.tests_table.setItem(row, 0, QTableWidgetItem(f"{test_name} ({key}, lag={lag})"))
                                    
                                    # Value
                                    value_item = QTableWidgetItem(f"{val:.4f}" if isinstance(val, (int, float)) else str(val))
                                    value_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                                    self.tests_table.setItem(row, 1, value_item)
                                    
                                    # p-value if available
                                    p_value = test_result.get(f"{key}_pvalue", [None])[i] if f"{key}_pvalue" in test_result else None
                                    if p_value is not None:
                                        p_value_item = QTableWidgetItem(f"{p_value:.4f}" if isinstance(p_value, (int, float)) else str(p_value))
                                        p_value_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                                        self.tests_table.setItem(row, 2, p_value_item)
                                    
                                    row += 1
                            else:
                                # For single values
                                self.tests_table.insertRow(row)
                                self.tests_table.setItem(row, 0, QTableWidgetItem(f"{test_name} ({key})"))
                                
                                # Value
                                value_item = QTableWidgetItem(f"{value:.4f}" if isinstance(value, (int, float)) else str(value))
                                value_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                                self.tests_table.setItem(row, 1, value_item)
                                
                                # p-value if available
                                p_value = test_result.get(f"{key}_pvalue")
                                if p_value is not None:
                                    p_value_item = QTableWidgetItem(f"{p_value:.4f}" if isinstance(p_value, (int, float)) else str(p_value))
                                    p_value_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                                    self.tests_table.setItem(row, 2, p_value_item)
                                
                                row += 1
                else:
                    # For tests with a single statistic
                    self.tests_table.insertRow(row)
                    self.tests_table.setItem(row, 0, QTableWidgetItem(test_name))
                    
                    # Value
                    value_item = QTableWidgetItem(f"{test_result:.4f}" if isinstance(test_result, (int, float)) else str(test_result))
                    value_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    self.tests_table.setItem(row, 1, value_item)
                    
                    row += 1
            
            # Resize rows to content
            self.tests_table.resizeRowsToContents()
            
            logger.debug("Diagnostic tests displayed")
        except Exception as e:
            logger.error(f"Error displaying diagnostic tests: {e}")
            self.tests_table.setRowCount(1)
            self.tests_table.setColumnCount(1)
            self.tests_table.setHorizontalHeaderLabels(["Error"])
            error_item = QTableWidgetItem(f"Error loading diagnostic tests: {str(e)}")
            error_item.setForeground(QColor(255, 0, 0))
            self.tests_table.setItem(0, 0, error_item)
    
    def update_pagination(self, current_page: int, total_pages: int):
        """
        Update pagination controls.
        
        Args:
            current_page: Current page number (1-based)
            total_pages: Total number of pages
        """
        self.page_label.setText(f"Page {current_page} of {total_pages}")
        self.prev_page_button.setEnabled(current_page > 1)
        self.next_page_button.setEnabled(current_page < total_pages)
        
        logger.debug(f"Pagination updated (page {current_page}/{total_pages})")
    
    def copy_results_to_clipboard(self):
        """Copy the current results to the clipboard."""
        try:
            # Get the current tab
            current_tab = self.tab_widget.currentWidget()
            
            # Create a string to hold the results
            results_text = "ARMAX Model Results\n\n"
            
            # Add model equation if available
            if self.equation_canvas is not None:
                results_text += "Model Equation:\n"
                # We can't get the exact equation from the canvas, so we'll add a placeholder
                results_text += "See model viewer for equation\n\n"
            
            # Add parameter estimates if on overview tab
            if current_tab == self.overview_tab:
                results_text += "Parameter Estimates:\n"
                results_text += "Parameter\tEstimate\tStd. Error\tt-Stat\tp-Value\n"
                
                for row in range(self.params_table.rowCount()):
                    param_name = self.params_table.item(row, 0).text()
                    estimate = self.params_table.item(row, 1).text()
                    std_error = self.params_table.item(row, 2).text() if self.params_table.item(row, 2) else ""
                    t_stat = self.params_table.item(row, 3).text() if self.params_table.item(row, 3) else ""
                    p_value = self.params_table.item(row, 4).text() if self.params_table.item(row, 4) else ""
                    
                    results_text += f"{param_name}\t{estimate}\t{std_error}\t{t_stat}\t{p_value}\n"
                
                results_text += "\nSignificance levels: [***] p<0.01, [**] p<0.05, [*] p<0.1\n\n"
            
            # Add statistics if on statistics tab
            if current_tab == self.stats_tab:
                results_text += "Model Statistics:\n"
                results_text += "Statistic\tValue\tNotes\n"
                
                for row in range(self.stats_table.rowCount()):
                    stat_name = self.stats_table.item(row, 0).text()
                    value = self.stats_table.item(row, 1).text()
                    notes = self.stats_table.item(row, 2).text() if self.stats_table.item(row, 2) else ""
                    
                    results_text += f"{stat_name}\t{value}\t{notes}\n"
                
                results_text += "\nDiagnostic Tests:\n"
                results_text += "Test\tValue\tp-Value\n"
                
                for row in range(self.tests_table.rowCount()):
                    test_name = self.tests_table.item(row, 0).text()
                    value = self.tests_table.item(row, 1).text() if self.tests_table.item(row, 1) else ""
                    p_value = self.tests_table.item(row, 2).text() if self.tests_table.item(row, 2) else ""
                    
                    results_text += f"{test_name}\t{value}\t{p_value}\n"
            
            # Copy to clipboard
            clipboard = QApplication.clipboard()
            clipboard.setText(results_text)
            
            logger.debug("Results copied to clipboard")
        except Exception as e:
            logger.error(f"Error copying results to clipboard: {e}")
            show_error_message(
                self, "Copy Error", 
                "Failed to copy results to clipboard", 
                str(e)
            )
    
    def show_error(self, title: str, message: str, details: str = None):
        """
        Show an error message dialog.
        
        Args:
            title: Dialog title
            message: Error message
            details: Detailed error information
        """
        show_error_message(self, title, message, details)
