#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARMAX model viewer interface for the MFE Toolbox.

This module implements a detailed viewer for ARMAX model results using PyQt6,
providing visualization of model equations, parameter estimates, and statistics.
It uses matplotlib for LaTeX-formatted equation rendering and implements pagination
for navigating through large parameter sets.

The viewer follows the Model-View-Controller (MVC) pattern with asynchronous
loading of model results to maintain UI responsiveness.
"""

import asyncio
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, cast

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox,
    QSizePolicy, QSpacerItem, QFrame, QWidget, QTabWidget,
    QScrollArea, QGridLayout
)
from PyQt6.QtCore import Qt, QSize, QEvent, pyqtSlot, pyqtSignal
from PyQt6.QtGui import QFont, QKeyEvent, QColor, QPixmap

from mfe.core.exceptions import UIError

# Configure logging
logger = logging.getLogger(__name__)


class ModelViewer(QDialog):
    """
    ARMAX model viewer dialog for displaying estimation results.
    
    This dialog provides a detailed view of ARMAX model results, including
    LaTeX-formatted equations, parameter estimates with standard errors,
    and model statistics. It supports pagination for navigating through
    large parameter sets and implements asynchronous loading for responsive UI.
    
    Attributes:
        parent: The parent widget
        model_results: Dictionary containing model estimation results
        current_page: Current page in the parameter table pagination
        total_pages: Total number of pages in the parameter table
        equation_canvas: Matplotlib canvas for rendering model equations
        parameter_table: Table widget for displaying parameter estimates
        statistics_table: Table widget for displaying model statistics
    """
    
    # Signal emitted when the model results are loaded
    results_loaded = pyqtSignal()
    
    def __init__(self, parent=None, model_results: Optional[Dict[str, Any]] = None):
        """
        Initialize the model viewer dialog.
        
        Args:
            parent: The parent widget
            model_results: Dictionary containing model estimation results
        """
        super().__init__(parent)
        self.parent = parent
        self.model_results = model_results
        self.current_page = 1
        self.total_pages = 1
        self.equation_canvas = None
        self.parameter_table = None
        self.statistics_table = None
        self.parameters_per_page = 10  # Number of parameters to display per page
        
        # Set up dialog properties
        self.setWindowTitle("ARMAX Model Viewer")
        self.setMinimumWidth(700)
        self.setMinimumHeight(500)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowCloseButtonHint)
        self.setModal(True)
        
        # Initialize UI components
        self._init_ui()
        
        # Connect signals
        self.results_loaded.connect(self._on_results_loaded)
        
        # Load model results if provided
        if model_results:
            asyncio.create_task(self._load_model_results_async(model_results))
        
        logger.debug("Model viewer dialog initialized")
    
    def _init_ui(self) -> None:
        """Initialize the UI components."""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Create a scroll area for the content
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        
        # Create a widget to hold the scrollable content
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(15)
        content_layout.setContentsMargins(0, 0, 0, 0)
        
        # Model equation section
        equation_group = QGroupBox("Model Equation", self)
        equation_layout = QVBoxLayout(equation_group)
        
        # Create a figure for the equation
        equation_figure = Figure(figsize=(6, 1.5), dpi=100)
        equation_figure.patch.set_facecolor('white')
        self.equation_canvas = FigureCanvas(equation_figure)
        self.equation_canvas.setMinimumHeight(100)
        equation_layout.addWidget(self.equation_canvas)
        
        content_layout.addWidget(equation_group)
        
        # Parameter estimates section
        parameter_group = QGroupBox("Parameter Estimates", self)
        parameter_layout = QVBoxLayout(parameter_group)
        
        # Create parameter table
        self.parameter_table = QTableWidget(0, 5, self)
        self.parameter_table.setHorizontalHeaderLabels(
            ["Parameter", "Estimate", "Std. Error", "t-Stat", "p-Value"]
        )
        self.parameter_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.parameter_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.parameter_table.setAlternatingRowColors(True)
        parameter_layout.addWidget(self.parameter_table)
        
        # Pagination controls
        pagination_layout = QHBoxLayout()
        pagination_layout.addStretch()
        
        self.prev_button = QPushButton("< Previous", self)
        self.prev_button.clicked.connect(self._on_prev_page)
        self.prev_button.setEnabled(False)
        pagination_layout.addWidget(self.prev_button)
        
        self.page_label = QLabel("Page 1 of 1", self)
        self.page_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pagination_layout.addWidget(self.page_label)
        
        self.next_button = QPushButton("Next >", self)
        self.next_button.clicked.connect(self._on_next_page)
        self.next_button.setEnabled(False)
        pagination_layout.addWidget(self.next_button)
        
        pagination_layout.addStretch()
        parameter_layout.addLayout(pagination_layout)
        
        content_layout.addWidget(parameter_group)
        
        # Model statistics section
        statistics_group = QGroupBox("Model Statistics", self)
        statistics_layout = QVBoxLayout(statistics_group)
        
        # Create statistics table
        self.statistics_table = QTableWidget(0, 3, self)
        self.statistics_table.setHorizontalHeaderLabels(
            ["Statistic", "Value", "Notes"]
        )
        self.statistics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.statistics_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.statistics_table.setAlternatingRowColors(True)
        statistics_layout.addWidget(self.statistics_table)
        
        content_layout.addWidget(statistics_group)
        
        # Set the content widget for the scroll area
        scroll_area.setWidget(content_widget)
        layout.addWidget(scroll_area)
        
        # Close button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        close_button = QPushButton("Close", self)
        close_button.setDefault(True)
        close_button.clicked.connect(self.accept)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
        
        # Set the layout
        self.setLayout(layout)
    
    async def _load_model_results_async(self, model_results: Dict[str, Any]) -> None:
        """
        Load model results asynchronously to keep the UI responsive.
        
        Args:
            model_results: Dictionary containing model estimation results
        """
        try:
            # Store the model results
            self.model_results = model_results
            
            # Small delay to allow the UI to render before processing
            await asyncio.sleep(0.1)
            
            # Emit signal to update the UI
            self.results_loaded.emit()
            
            logger.debug("Model results loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model results: {e}")
            raise UIError(
                f"Failed to load model results: {str(e)}",
                component="ModelViewer",
                issue="Error loading model results",
                details=str(e)
            ) from e
    
    @pyqtSlot()
    def _on_results_loaded(self) -> None:
        """Handle the results_loaded signal."""
        if not self.model_results:
            logger.warning("No model results to display")
            return
        
        # Update the model equation
        self._update_equation()
        
        # Update the parameter table
        self._update_parameter_table()
        
        # Update the statistics table
        self._update_statistics_table()
        
        logger.debug("UI updated with model results")
    
    def _update_equation(self) -> None:
        """Update the model equation display."""
        if not self.model_results or not self.equation_canvas:
            return
        
        try:
            # Get the equation string from the model results
            equation_str = self.model_results.get('equation', r'y_t = \beta_0 + \sum_{i=1}^{p} \beta_i y_{t-i} + \varepsilon_t + \sum_{j=1}^{q} \theta_j \varepsilon_{t-j}')
            
            # Clear the figure
            self.equation_canvas.figure.clear()
            
            # Create an axis for the equation
            ax = self.equation_canvas.figure.add_subplot(111)
            ax.set_axis_off()
            
            # Render the equation using LaTeX
            ax.text(
                0.5, 0.5, f"${{equation_str}}$",
                fontsize=12, ha='center', va='center',
                transform=ax.transAxes
            )
            
            # Adjust the figure layout
            self.equation_canvas.figure.tight_layout()
            
            # Redraw the canvas
            self.equation_canvas.draw()
            
            logger.debug("Model equation updated")
        except Exception as e:
            logger.error(f"Error updating equation: {e}")
            # Display error message in the equation area
            if self.equation_canvas:
                self.equation_canvas.figure.clear()
                ax = self.equation_canvas.figure.add_subplot(111)
                ax.set_axis_off()
                ax.text(
                    0.5, 0.5, "Error rendering equation",
                    fontsize=12, ha='center', va='center', color='red',
                    transform=ax.transAxes
                )
                self.equation_canvas.figure.tight_layout()
                self.equation_canvas.draw()
    
    def _update_parameter_table(self) -> None:
        """Update the parameter table with model results."""
        if not self.model_results or not self.parameter_table:
            return
        
        try:
            # Get the parameter estimates from the model results
            parameters = self.model_results.get('parameters', [])
            
            if not parameters:
                # No parameters to display
                self.parameter_table.setRowCount(0)
                self.page_label.setText("Page 0 of 0")
                self.prev_button.setEnabled(False)
                self.next_button.setEnabled(False)
                return
            
            # Calculate pagination
            self.total_pages = math.ceil(len(parameters) / self.parameters_per_page)
            self.current_page = min(self.current_page, self.total_pages)
            
            # Update pagination controls
            self.page_label.setText(f"Page {self.current_page} of {self.total_pages}")
            self.prev_button.setEnabled(self.current_page > 1)
            self.next_button.setEnabled(self.current_page < self.total_pages)
            
            # Calculate the range of parameters to display
            start_idx = (self.current_page - 1) * self.parameters_per_page
            end_idx = min(start_idx + self.parameters_per_page, len(parameters))
            
            # Clear the table
            self.parameter_table.setRowCount(0)
            
            # Add parameters to the table
            for i, param in enumerate(parameters[start_idx:end_idx]):
                self.parameter_table.insertRow(i)
                
                # Parameter name
                name_item = QTableWidgetItem(param.get('name', f"Parameter {i+1}"))
                self.parameter_table.setItem(i, 0, name_item)
                
                # Estimate
                estimate = param.get('estimate', 0.0)
                estimate_item = QTableWidgetItem(f"{estimate:.4f}")
                estimate_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.parameter_table.setItem(i, 1, estimate_item)
                
                # Standard error
                std_error = param.get('std_error', 0.0)
                std_error_item = QTableWidgetItem(f"{std_error:.4f}")
                std_error_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.parameter_table.setItem(i, 2, std_error_item)
                
                # t-statistic
                t_stat = param.get('t_stat', 0.0)
                t_stat_item = QTableWidgetItem(f"{t_stat:.4f}")
                t_stat_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.parameter_table.setItem(i, 3, t_stat_item)
                
                # p-value with significance indicator
                p_value = param.get('p_value', 1.0)
                p_value_text = f"{p_value:.4f}"
                
                # Add significance indicators
                if p_value < 0.01:
                    p_value_text += " [***]"
                elif p_value < 0.05:
                    p_value_text += " [**]"
                elif p_value < 0.1:
                    p_value_text += " [*]"
                
                p_value_item = QTableWidgetItem(p_value_text)
                p_value_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.parameter_table.setItem(i, 4, p_value_item)
            
            # Resize rows to content
            self.parameter_table.resizeRowsToContents()
            
            logger.debug(f"Parameter table updated (page {self.current_page} of {self.total_pages})")
        except Exception as e:
            logger.error(f"Error updating parameter table: {e}")
            # Clear the table and show error
            self.parameter_table.setRowCount(1)
            self.parameter_table.setColumnCount(1)
            self.parameter_table.setHorizontalHeaderLabels(["Error"])
            error_item = QTableWidgetItem(f"Error loading parameters: {str(e)}")
            error_item.setForeground(QColor(255, 0, 0))
            self.parameter_table.setItem(0, 0, error_item)
    
    def _update_statistics_table(self) -> None:
        """Update the statistics table with model results."""
        if not self.model_results or not self.statistics_table:
            return
        
        try:
            # Get the model statistics from the model results
            statistics = self.model_results.get('statistics', {})
            
            if not statistics:
                # No statistics to display
                self.statistics_table.setRowCount(0)
                return
            
            # Clear the table
            self.statistics_table.setRowCount(0)
            
            # Add statistics to the table
            row = 0
            
            # Log-likelihood
            if 'log_likelihood' in statistics:
                self.statistics_table.insertRow(row)
                self.statistics_table.setItem(row, 0, QTableWidgetItem("Log-likelihood"))
                
                log_likelihood = statistics['log_likelihood']
                log_likelihood_item = QTableWidgetItem(f"{log_likelihood:.4f}")
                log_likelihood_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.statistics_table.setItem(row, 1, log_likelihood_item)
                
                self.statistics_table.setItem(row, 2, QTableWidgetItem(""))
                row += 1
            
            # AIC
            if 'aic' in statistics:
                self.statistics_table.insertRow(row)
                self.statistics_table.setItem(row, 0, QTableWidgetItem("AIC"))
                
                aic = statistics['aic']
                aic_item = QTableWidgetItem(f"{aic:.4f}")
                aic_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.statistics_table.setItem(row, 1, aic_item)
                
                self.statistics_table.setItem(row, 2, QTableWidgetItem("Lower values indicate better fit"))
                row += 1
            
            # BIC
            if 'bic' in statistics:
                self.statistics_table.insertRow(row)
                self.statistics_table.setItem(row, 0, QTableWidgetItem("BIC"))
                
                bic = statistics['bic']
                bic_item = QTableWidgetItem(f"{bic:.4f}")
                bic_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.statistics_table.setItem(row, 1, bic_item)
                
                self.statistics_table.setItem(row, 2, QTableWidgetItem("Lower values indicate better fit"))
                row += 1
            
            # RMSE
            if 'rmse' in statistics:
                self.statistics_table.insertRow(row)
                self.statistics_table.setItem(row, 0, QTableWidgetItem("RMSE"))
                
                rmse = statistics['rmse']
                rmse_item = QTableWidgetItem(f"{rmse:.4f}")
                rmse_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.statistics_table.setItem(row, 1, rmse_item)
                
                self.statistics_table.setItem(row, 2, QTableWidgetItem("Root Mean Square Error"))
                row += 1
            
            # R-squared
            if 'r_squared' in statistics:
                self.statistics_table.insertRow(row)
                self.statistics_table.setItem(row, 0, QTableWidgetItem("R²"))
                
                r_squared = statistics['r_squared']
                r_squared_item = QTableWidgetItem(f"{r_squared:.4f}")
                r_squared_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.statistics_table.setItem(row, 1, r_squared_item)
                
                self.statistics_table.setItem(row, 2, QTableWidgetItem("Coefficient of determination"))
                row += 1
            
            # Adjusted R-squared
            if 'adj_r_squared' in statistics:
                self.statistics_table.insertRow(row)
                self.statistics_table.setItem(row, 0, QTableWidgetItem("Adjusted R²"))
                
                adj_r_squared = statistics['adj_r_squared']
                adj_r_squared_item = QTableWidgetItem(f"{adj_r_squared:.4f}")
                adj_r_squared_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.statistics_table.setItem(row, 1, adj_r_squared_item)
                
                self.statistics_table.setItem(row, 2, QTableWidgetItem("Adjusted for number of parameters"))
                row += 1
            
            # Number of observations
            if 'n_obs' in statistics:
                self.statistics_table.insertRow(row)
                self.statistics_table.setItem(row, 0, QTableWidgetItem("Observations"))
                
                n_obs = statistics['n_obs']
                n_obs_item = QTableWidgetItem(f"{n_obs}")
                n_obs_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.statistics_table.setItem(row, 1, n_obs_item)
                
                self.statistics_table.setItem(row, 2, QTableWidgetItem("Number of observations"))
                row += 1
            
            # Degrees of freedom
            if 'df' in statistics:
                self.statistics_table.insertRow(row)
                self.statistics_table.setItem(row, 0, QTableWidgetItem("Degrees of Freedom"))
                
                df = statistics['df']
                df_item = QTableWidgetItem(f"{df}")
                df_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.statistics_table.setItem(row, 1, df_item)
                
                self.statistics_table.setItem(row, 2, QTableWidgetItem(""))
                row += 1
            
            # Add any other statistics
            for key, value in statistics.items():
                if key not in ['log_likelihood', 'aic', 'bic', 'rmse', 'r_squared', 'adj_r_squared', 'n_obs', 'df']:
                    self.statistics_table.insertRow(row)
                    
                    # Format the key for display (convert snake_case to Title Case)
                    display_key = ' '.join(word.capitalize() for word in key.split('_'))
                    self.statistics_table.setItem(row, 0, QTableWidgetItem(display_key))
                    
                    # Format the value based on its type
                    if isinstance(value, (int, float)):
                        value_item = QTableWidgetItem(f"{value:.4f}" if isinstance(value, float) else f"{value}")
                        value_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    else:
                        value_item = QTableWidgetItem(str(value))
                    
                    self.statistics_table.setItem(row, 1, value_item)
                    self.statistics_table.setItem(row, 2, QTableWidgetItem(""))
                    row += 1
            
            # Resize rows to content
            self.statistics_table.resizeRowsToContents()
            
            logger.debug("Statistics table updated")
        except Exception as e:
            logger.error(f"Error updating statistics table: {e}")
            # Clear the table and show error
            self.statistics_table.setRowCount(1)
            self.statistics_table.setColumnCount(1)
            self.statistics_table.setHorizontalHeaderLabels(["Error"])
            error_item = QTableWidgetItem(f"Error loading statistics: {str(e)}")
            error_item.setForeground(QColor(255, 0, 0))
            self.statistics_table.setItem(0, 0, error_item)
    
    @pyqtSlot()
    def _on_prev_page(self) -> None:
        """Handle the previous page button click."""
        if self.current_page > 1:
            self.current_page -= 1
            self._update_parameter_table()
    
    @pyqtSlot()
    def _on_next_page(self) -> None:
        """Handle the next page button click."""
        if self.current_page < self.total_pages:
            self.current_page += 1
            self._update_parameter_table()
    
    def keyPressEvent(self, event: QKeyEvent) -> None:
        """
        Handle key press events.
        
        Args:
            event: The key event
        """
        # Close dialog on Escape key
        if event.key() == Qt.Key.Key_Escape:
            self.reject()
        # Accept dialog on Return/Enter key
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.accept()
        # Previous page on Left key
        elif event.key() == Qt.Key.Key_Left and self.prev_button.isEnabled():
            self._on_prev_page()
        # Next page on Right key
        elif event.key() == Qt.Key.Key_Right and self.next_button.isEnabled():
            self._on_next_page()
        else:
            super().keyPressEvent(event)
    
    def showEvent(self, event: QEvent) -> None:
        """
        Handle show events.
        
        This method centers the dialog on the parent window when shown.
        
        Args:
            event: The show event
        """
        super().showEvent(event)
        
        # Center the dialog on the parent window
        if self.parent:
            parent_geo = self.parent.geometry()
            size = self.geometry().size()
            x = parent_geo.x() + (parent_geo.width() - size.width()) // 2
            y = parent_geo.y() + (parent_geo.height() - size.height()) // 2
            self.move(x, y)
        
        logger.debug("Model viewer dialog shown")
    
    @classmethod
    async def show_dialog_async(cls, parent=None, model_results: Optional[Dict[str, Any]] = None) -> int:
        """
        Show the model viewer dialog asynchronously.
        
        Args:
            parent: The parent widget
            model_results: Dictionary containing model estimation results
            
        Returns:
            int: Dialog result code (QDialog.Accepted or QDialog.Rejected)
        """
        dialog = cls(parent, model_results)
        result = dialog.exec()
        return result
    
    @classmethod
    def show_dialog(cls, parent=None, model_results: Optional[Dict[str, Any]] = None) -> int:
        """
        Show the model viewer dialog.
        
        This is a synchronous convenience method for creating and showing the dialog.
        
        Args:
            parent: The parent widget
            model_results: Dictionary containing model estimation results
            
        Returns:
            int: Dialog result code (QDialog.Accepted or QDialog.Rejected)
        """
        dialog = cls(parent, model_results)
        result = dialog.exec()
        return result



def show_model_results(parent=None, model_results: Optional[Dict[str, Any]] = None) -> int:
    """
    Show the model viewer dialog with the given model results.
    
    This is a convenience function for showing the model viewer dialog.
    
    Args:
        parent: The parent widget
        model_results: Dictionary containing model estimation results
        
    Returns:
        int: Dialog result code (QDialog.Accepted or QDialog.Rejected)
    """
    return ModelViewer.show_dialog(parent, model_results)


if __name__ == "__main__":
    # Test the dialog if run directly
    import sys
    from PyQt6.QtWidgets import QApplication
    
    # Create sample model results
    sample_results = {
        'equation': r'y_t = 0.452 + 0.721 y_{t-1} - 0.342 y_{t-2} + \varepsilon_t - 0.256 \varepsilon_{t-1}',
        'parameters': [
            {'name': 'Constant', 'estimate': 0.452, 'std_error': 0.124, 't_stat': 3.645, 'p_value': 0.0003},
            {'name': 'AR{1}', 'estimate': 0.721, 'std_error': 0.095, 't_stat': 7.589, 'p_value': 0.0001},
            {'name': 'AR{2}', 'estimate': -0.342, 'std_error': 0.098, 't_stat': -3.490, 'p_value': 0.0005},
            {'name': 'MA{1}', 'estimate': -0.256, 'std_error': 0.102, 't_stat': -2.509, 'p_value': 0.0122},
        ],
        'statistics': {
            'log_likelihood': 67.89,
            'aic': -123.45,
            'bic': -120.67,
            'rmse': 0.342,
            'r_squared': 0.876,
            'adj_r_squared': 0.872,
            'n_obs': 200,
            'df': 196
        }
    }
    
    app = QApplication(sys.argv)
    result = show_model_results(model_results=sample_results)
    sys.exit(0)
