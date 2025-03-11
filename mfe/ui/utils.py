#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UI utility functions for the MFE Toolbox.

This module provides utility functions for the UI components of the MFE Toolbox,
handling tasks like LaTeX equation formatting, table generation, asynchronous
operations, and screen size detection. These utilities support the main UI
components by providing common functionality in a reusable form.

The module is designed to work with PyQt6 and integrates with matplotlib for
equation rendering and visualization. It includes utilities for asynchronous
processing using Python's async/await pattern to maintain UI responsiveness
during long-running operations.
"""

import asyncio
import logging
import math
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TypeVar, cast
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt6.QtWidgets import (
    QApplication, QWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QProgressDialog, QMessageBox, QLabel, QSizePolicy
)
from PyQt6.QtCore import Qt, QSize, QTimer, QObject, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont, QFontMetrics, QColor, QScreen, QGuiApplication

from mfe.core.exceptions import UIError, AsyncError

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for generic function typing
T = TypeVar('T')


def format_latex_equation(equation: str) -> str:
    """
    Format a LaTeX equation for display in matplotlib.
    
    This function formats a LaTeX equation string for proper display in matplotlib
    by ensuring it has the correct delimiters and escaping special characters.
    
    Args:
        equation: LaTeX equation string
    
    Returns:
        str: Formatted LaTeX equation string
    
    Examples:
        >>> format_latex_equation("y = \\beta_0 + \\beta_1 x")
        '$y = \\beta_0 + \\beta_1 x$'
    """
    # Remove any existing $ delimiters
    equation = equation.strip()
    if equation.startswith('$') and equation.endswith('$'):
        equation = equation[1:-1]
    
    # Add $ delimiters
    formatted_equation = f"${equation}$"
    
    return formatted_equation


# Alias for backward compatibility
format_equation = format_latex_equation


def create_equation_figure(equation: str, 
                          font_size: int = 12, 
                          dpi: int = 100, 
                          dark_mode: bool = False) -> Figure:
    """
    Create a matplotlib figure with a LaTeX equation.
    
    This function creates a matplotlib figure with a LaTeX equation rendered
    using matplotlib's math text rendering. The figure is sized to fit the
    equation and can be customized with different font sizes and DPI settings.
    
    Args:
        equation: LaTeX equation string
        font_size: Font size for the equation
        dpi: DPI for the figure
        dark_mode: Whether to use dark mode colors
    
    Returns:
        Figure: Matplotlib figure with the rendered equation
    
    Raises:
        ValueError: If the equation is empty or invalid
    """
    if not equation:
        raise ValueError("Equation cannot be empty")
    
    # Format the equation for LaTeX
    formatted_equation = format_latex_equation(equation)
    
    # Create a figure with the right size
    fig = Figure(figsize=(6, 1.5), dpi=dpi)
    
    # Set dark mode if requested
    if dark_mode:
        fig.patch.set_facecolor("#2D2D30")
        text_color = "white"
    else:
        fig.patch.set_facecolor("white")
        text_color = "black"
    
    # Add a single axes to the figure
    ax = fig.add_subplot(111)
    
    # Remove axes and ticks
    ax.axis("off")
    
    # Set the equation as the title
    ax.set_title(formatted_equation, fontsize=font_size, color=text_color)
    
    # Adjust the layout to fit the equation
    fig.tight_layout(pad=0.5)
    
    return fig


def create_figure_canvas(figsize: Tuple[float, float] = (6, 4), 
                        dpi: int = 100, 
                        tight_layout: bool = True) -> Tuple[Figure, FigureCanvas]:
    """
    Create a matplotlib figure and canvas for embedding in Qt widgets.
    
    This function creates a matplotlib figure and a FigureCanvas that can be
    embedded in Qt widgets. The figure and canvas are configured with the
    specified size and DPI settings.
    
    Args:
        figsize: Size of the figure in inches (width, height)
        dpi: DPI for the figure
        tight_layout: Whether to use tight layout for the figure
    
    Returns:
        Tuple[Figure, FigureCanvas]: Matplotlib figure and canvas
    
    Examples:
        >>> fig, canvas = create_figure_canvas(figsize=(8, 6), dpi=100)
        >>> ax = fig.add_subplot(111)
        >>> ax.plot([1, 2, 3], [4, 5, 6])
        >>> canvas.draw()
        >>> layout.addWidget(canvas)
    """
    # Create a figure with the specified size and DPI
    fig = Figure(figsize=figsize, dpi=dpi)
    
    # Create a canvas for the figure
    canvas = FigureCanvas(fig)
    
    # Set the canvas size policy
    canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    
    # Use tight layout if requested
    if tight_layout:
        fig.tight_layout()
    
    return fig, canvas


def embed_matplotlib_figure(fig: Figure, parent: Optional[QWidget] = None) -> FigureCanvas:
    """
    Embed a matplotlib figure in a Qt widget.
    
    This function creates a FigureCanvas for a matplotlib figure and configures
    it for embedding in a Qt widget. The canvas is set to expand to fill the
    available space.
    
    Args:
        fig: Matplotlib figure to embed
        parent: Parent widget for the canvas
    
    Returns:
        FigureCanvas: Canvas widget containing the figure
    
    Examples:
        >>> fig = Figure(figsize=(8, 6))
        >>> ax = fig.add_subplot(111)
        >>> ax.plot([1, 2, 3], [4, 5, 6])
        >>> canvas = embed_matplotlib_figure(fig)
        >>> layout.addWidget(canvas)
    """
    # Create a canvas for the figure
    canvas = FigureCanvas(fig)
    
    # Set the parent if provided
    if parent is not None:
        canvas.setParent(parent)
    
    # Set the canvas size policy
    canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    
    # Update the figure
    canvas.draw()
    
    return canvas


def create_parameter_table(parameters: List[Dict[str, Any]], 
                          table: QTableWidget,
                          page: int = 1,
                          items_per_page: int = 10) -> Tuple[int, int]:
    """
    Create a table of model parameters with pagination.
    
    This function populates a QTableWidget with model parameters, including
    parameter names, values, standard errors, t-statistics, and p-values.
    It supports pagination for large parameter sets.
    
    Args:
        parameters: List of parameter dictionaries
        table: QTableWidget to populate
        page: Current page number (1-based)
        items_per_page: Number of items to display per page
    
    Returns:
        Tuple[int, int]: Current page and total pages
    
    Raises:
        ValueError: If parameters is empty or table is None
    """
    if not parameters:
        raise ValueError("Parameters list cannot be empty")
    
    if table is None:
        raise ValueError("Table widget cannot be None")
    
    try:
        # Calculate pagination
        total_pages = math.ceil(len(parameters) / items_per_page)
        current_page = min(max(1, page), max(1, total_pages))
        
        # Calculate the range of parameters to display
        start_idx = (current_page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, len(parameters))
        
        # Clear the table
        table.setRowCount(0)
        
        # Set up the table headers if not already set
        if table.columnCount() != 5:
            table.setColumnCount(5)
            table.setHorizontalHeaderLabels(
                ["Parameter", "Estimate", "Std. Error", "t-Stat", "p-Value"]
            )
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        
        # Add parameters to the table
        for i, param in enumerate(parameters[start_idx:end_idx]):
            table.insertRow(i)
            
            # Parameter name
            name_item = QTableWidgetItem(param.get('name', f"Parameter {i+1}"))
            table.setItem(i, 0, name_item)
            
            # Estimate
            estimate = param.get('estimate', 0.0)
            estimate_item = QTableWidgetItem(f"{estimate:.4f}")
            estimate_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            table.setItem(i, 1, estimate_item)
            
            # Standard error
            std_error = param.get('std_error', 0.0)
            std_error_item = QTableWidgetItem(f"{std_error:.4f}")
            std_error_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            table.setItem(i, 2, std_error_item)
            
            # t-statistic
            t_stat = param.get('t_stat', 0.0)
            t_stat_item = QTableWidgetItem(f"{t_stat:.4f}")
            t_stat_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            table.setItem(i, 3, t_stat_item)
            
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
            table.setItem(i, 4, p_value_item)
        
        # Resize rows to content
        table.resizeRowsToContents()
        
        return current_page, total_pages
    except Exception as e:
        logger.error(f"Error creating parameter table: {e}")
        # Clear the table and show error
        table.setRowCount(1)
        table.setColumnCount(1)
        table.setHorizontalHeaderLabels(["Error"])
        error_item = QTableWidgetItem(f"Error loading parameters: {str(e)}")
        error_item.setForeground(QColor(255, 0, 0))
        table.setItem(0, 0, error_item)
        return 1, 1


def create_results_table(results: Dict[str, Any], 
                        table: QTableWidget,
                        show_significance: bool = True,
                        decimal_places: int = 4) -> None:
    """
    Create a table of model estimation results.
    
    This function populates a QTableWidget with model estimation results,
    including parameter estimates, standard errors, t-statistics, and p-values.
    It supports formatting of values and highlighting of significant parameters.
    
    Args:
        results: Dictionary of model results
        table: QTableWidget to populate
        show_significance: Whether to show significance stars
        decimal_places: Number of decimal places to display
    
    Raises:
        ValueError: If results is empty or table is None
    """
    if not results:
        raise ValueError("Results dictionary cannot be empty")
    
    if table is None:
        raise ValueError("Table widget cannot be None")
    
    # Clear the table
    table.clearContents()
    table.setRowCount(0)
    
    # Set up the table headers
    table.setColumnCount(5)
    table.setHorizontalHeaderLabels(["Parameter", "Value", "Std. Error", "t-Statistic", "p-Value"])
    
    # Get the parameters from the results
    parameters = results.get("parameters", {})
    std_errors = results.get("std_errors", {})
    t_stats = results.get("t_statistics", {})
    p_values = results.get("p_values", {})
    
    # Define significance levels and symbols
    significance_levels = {
        0.01: "***",
        0.05: "**",
        0.10: "*"
    }
    
    # Add the parameters to the table
    row = 0
    for param_name, param_value in parameters.items():
        table.insertRow(row)
        
        # Parameter name
        table.setItem(row, 0, QTableWidgetItem(param_name))
        
        # Parameter value
        value_str = f"{param_value:.{decimal_places}f}"
        table.setItem(row, 1, QTableWidgetItem(value_str))
        
        # Standard error
        std_err = std_errors.get(param_name, float('nan'))
        std_err_str = f"{std_err:.{decimal_places}f}" if not math.isnan(std_err) else ""
        table.setItem(row, 2, QTableWidgetItem(std_err_str))
        
        # t-statistic
        t_stat = t_stats.get(param_name, float('nan'))
        t_stat_str = f"{t_stat:.{decimal_places}f}" if not math.isnan(t_stat) else ""
        table.setItem(row, 3, QTableWidgetItem(t_stat_str))
        
        # p-value and significance
        p_val = p_values.get(param_name, float('nan'))
        if not math.isnan(p_val):
            # Add significance stars if requested
            sig_stars = ""
            if show_significance:
                for level, symbol in significance_levels.items():
                    if p_val < level:
                        sig_stars = symbol
                        break
            
            p_val_str = f"{p_val:.{decimal_places}f}{sig_stars}"
            item = QTableWidgetItem(p_val_str)
            
            # Highlight significant p-values
            if p_val < 0.05:
                item.setForeground(QColor(0, 0, 255))  # Blue for significant
            elif p_val < 0.10:
                item.setForeground(QColor(0, 0, 128))  # Dark blue for marginally significant
            
            table.setItem(row, 4, item)
        else:
            table.setItem(row, 4, QTableWidgetItem(""))
        
        row += 1
    
    # Resize columns to content
    table.resizeColumnsToContents()
    
    # Set the table to stretch the last section
    header = table.horizontalHeader()
    header.setStretchLastSection(True)


def create_statistics_table(statistics: Dict[str, Any], table: QTableWidget) -> None:
    """
    Populate a QTableWidget with model statistics data.
    
    This function populates a QTableWidget with model statistics from a
    dictionary of statistics values.
    
    Args:
        statistics: Dictionary of statistics with statistic names as keys
        table: QTableWidget to populate
        
    Examples:
        >>> stats = {'log_likelihood': 67.89, 'aic': -123.45, 'bic': -120.67}
        >>> create_statistics_table(stats, table_widget)
    """
    try:
        # Clear the table
        table.setRowCount(0)
        
        # Set up the table headers if not already set
        if table.columnCount() != 3:
            table.setColumnCount(3)
            table.setHorizontalHeaderLabels(
                ["Statistic", "Value", "Notes"]
            )
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        
        # Add statistics to the table
        row = 0
        
        # Common statistics with predefined notes
        common_stats = {
            'log_likelihood': ("Log-likelihood", ""),
            'aic': ("AIC", "Lower values indicate better fit"),
            'bic': ("BIC", "Lower values indicate better fit"),
            'rmse': ("RMSE", "Root Mean Square Error"),
            'r_squared': ("R²", "Coefficient of determination"),
            'adj_r_squared': ("Adjusted R²", "Adjusted for number of parameters"),
            'n_obs': ("Observations", "Number of observations"),
            'df': ("Degrees of Freedom", "")
        }
        
        # Add common statistics first
        for key, (display_name, note) in common_stats.items():
            if key in statistics:
                table.insertRow(row)
                table.setItem(row, 0, QTableWidgetItem(display_name))
                
                value = statistics[key]
                if isinstance(value, (int, float)):
                    value_item = QTableWidgetItem(f"{value:.4f}" if isinstance(value, float) else f"{value}")
                    value_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                else:
                    value_item = QTableWidgetItem(str(value))
                
                table.setItem(row, 1, value_item)
                table.setItem(row, 2, QTableWidgetItem(note))
                row += 1
        
        # Add any other statistics
        for key, value in statistics.items():
            if key not in common_stats:
                table.insertRow(row)
                
                # Format the key for display (convert snake_case to Title Case)
                display_key = ' '.join(word.capitalize() for word in key.split('_'))
                table.setItem(row, 0, QTableWidgetItem(display_key))
                
                # Format the value based on its type
                if isinstance(value, (int, float)):
                    value_item = QTableWidgetItem(f"{value:.4f}" if isinstance(value, float) else f"{value}")
                    value_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                else:
                    value_item = QTableWidgetItem(str(value))
                
                table.setItem(row, 1, value_item)
                table.setItem(row, 2, QTableWidgetItem(""))
                row += 1
        
        # Resize rows to content
        table.resizeRowsToContents()
    except Exception as e:
        logger.error(f"Error creating statistics table: {e}")
        # Clear the table and show error
        table.setRowCount(1)
        table.setColumnCount(1)
        table.setHorizontalHeaderLabels(["Error"])
        error_item = QTableWidgetItem(f"Error loading statistics: {str(e)}")
        error_item.setForeground(QColor(255, 0, 0))
        table.setItem(0, 0, error_item)


class AsyncHelper:
    """
    Helper class for running asynchronous tasks in a PyQt6 application.
    
    This class provides utilities for running asynchronous coroutines
    within the Qt event loop, enabling non-blocking operations for
    long-running tasks like model estimation and forecasting.
    
    Examples:
        >>> async def long_task():
        ...     await asyncio.sleep(2)
        ...     return "Result"
        >>> 
        >>> def on_result(result):
        ...     print(f"Task completed with result: {result}")
        >>> 
        >>> AsyncHelper.run_async(long_task(), on_result)
    """
    
    @staticmethod
    def run_async(coro, callback=None, error_callback=None):
        """
        Run an asynchronous coroutine in the Qt event loop.
        
        Args:
            coro: The coroutine to run
            callback: Function to call with the result when the coroutine completes
            error_callback: Function to call if the coroutine raises an exception
            
        Returns:
            asyncio.Future: The future representing the coroutine execution
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
    
    @staticmethod
    async def run_with_progress_updates(coro, parent, title, message, 
                                       progress_callback=None, cancellable=True):
        """
        Run an asynchronous coroutine with a progress dialog that shows updates.
        
        Args:
            coro: The coroutine to run
            parent: Parent widget for the progress dialog
            title: Title for the progress dialog
            message: Initial message to display in the progress dialog
            progress_callback: Function to call with progress updates (0-100)
            cancellable: Whether the operation can be cancelled
            
        Returns:
            The result of the coroutine
            
        Raises:
            AsyncError: If the operation is cancelled or fails
        """
        # Create progress dialog with determinate progress
        progress = QProgressDialog(message, "Cancel" if cancellable else None, 0, 100, parent)
        progress.setWindowTitle(title)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(500)  # Show after 500ms
        progress.setValue(0)
        
        # Create a future for the coroutine
        future = asyncio.ensure_future(coro)
        
        # Set up cancellation if supported
        if cancellable:
            progress.canceled.connect(lambda: future.cancel())
        
        # Create a progress update function
        def update_progress(percent, message_update=None):
            progress.setValue(int(percent))
            if message_update:
                progress.setLabelText(message_update)
        
        # Call the original progress callback if provided
        original_callback = progress_callback
        
        # Process events while waiting for the coroutine to complete
        while not future.done():
            QApplication.processEvents()
            if original_callback:
                original_callback(update_progress)
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


class ProgressReporter:
    """
    Helper class for reporting progress from asynchronous operations.
    
    This class provides a standardized way to report progress from
    asynchronous operations to UI components.
    
    Examples:
        >>> reporter = ProgressReporter()
        >>> 
        >>> async def long_task(reporter):
        ...     for i in range(10):
        ...         await asyncio.sleep(0.5)
        ...         reporter.update(i * 10, f"Processing step {i+1}/10")
        ...     return "Result"
        >>> 
        >>> def on_progress(percent, message):
        ...     print(f"{percent}% complete: {message}")
        >>> 
        >>> reporter.set_callback(on_progress)
        >>> AsyncHelper.run_async(long_task(reporter))
    """
    
    def __init__(self):
        """Initialize the progress reporter."""
        self._callback = None
        self._start_time = None
        self._last_update_time = None
    
    def set_callback(self, callback: Callable[[float, Optional[str]], None]) -> None:
        """
        Set the progress callback function.
        
        Args:
            callback: Function to call with progress updates (percent, message)
        """
        self._callback = callback
        self._start_time = time.time()
        self._last_update_time = self._start_time
    
    def update(self, percent: float, message: Optional[str] = None) -> None:
        """
        Report progress update.
        
        Args:
            percent: Progress percentage (0-100)
            message: Optional progress message
        """
        if self._callback:
            current_time = time.time()
            # Limit update rate to avoid UI flooding (max 10 updates per second)
            if current_time - self._last_update_time >= 0.1:
                self._callback(percent, message)
                self._last_update_time = current_time
    
    def estimate_remaining(self, percent: float) -> float:
        """
        Estimate remaining time based on progress percentage.
        
        Args:
            percent: Progress percentage (0-100)
            
        Returns:
            Estimated remaining time in seconds, or -1 if unknown
        """
        if self._start_time is None or percent <= 0:
            return -1
        
        elapsed = time.time() - self._start_time
        if percent >= 100:
            return 0
        
        return elapsed * (100 - percent) / percent


def get_screen_info() -> Dict[str, Any]:
    """
    Get information about the current screen configuration.
    
    This function retrieves information about the current screen configuration,
    including screen size, resolution, and DPI.
    
    Returns:
        Dictionary containing screen information
        
    Examples:
        >>> screen_info = get_screen_info()
        >>> print(f"Screen size: {screen_info['width']}x{screen_info['height']}")
        >>> print(f"DPI: {screen_info['dpi']}")
    """
    screen_info = {}
    
    # Get the primary screen
    app = QGuiApplication.instance()
    if app is None:
        # Create a temporary application if none exists
        app = QGuiApplication([])
    
    screen = QGuiApplication.primaryScreen()
    if screen:
        # Get screen geometry
        geometry = screen.geometry()
        screen_info['width'] = geometry.width()
        screen_info['height'] = geometry.height()
        
        # Get screen physical size
        physical_size = screen.physicalSize()
        screen_info['physical_width_mm'] = physical_size.width()
        screen_info['physical_height_mm'] = physical_size.height()
        
        # Get screen DPI
        dpi = screen.logicalDotsPerInch()
        screen_info['dpi'] = dpi
        
        # Get screen scale factor
        scale_factor = screen.devicePixelRatio()
        screen_info['scale_factor'] = scale_factor
        
        # Calculate effective resolution
        screen_info['effective_width'] = int(geometry.width() * scale_factor)
        screen_info['effective_height'] = int(geometry.height() * scale_factor)
    
    return screen_info


def center_widget_on_screen(widget: QWidget) -> None:
    """
    Center a widget on the screen.
    
    This function centers a widget on the primary screen.
    
    Args:
        widget: The widget to center
        
    Examples:
        >>> center_widget_on_screen(my_dialog)
    """
    screen = QGuiApplication.primaryScreen()
    if screen:
        center = screen.availableGeometry().center()
        geo = widget.frameGeometry()
        geo.moveCenter(center)
        widget.move(geo.topLeft())


def center_widget_on_parent(widget: QWidget, parent: QWidget) -> None:
    """
    Center a widget on its parent widget.
    
    This function centers a widget on its parent widget.
    
    Args:
        widget: The widget to center
        parent: The parent widget
        
    Examples:
        >>> center_widget_on_parent(my_dialog, main_window)
    """
    if parent:
        parent_geo = parent.geometry()
        size = widget.geometry().size()
        x = parent_geo.x() + (parent_geo.width() - size.width()) // 2
        y = parent_geo.y() + (parent_geo.height() - size.height()) // 2
        widget.move(x, y)


def calculate_optimal_font_size(text: str, width: int, height: int, 
                               font_family: str = "Arial", 
                               min_size: int = 8, 
                               max_size: int = 24) -> int:
    """
    Calculate the optimal font size to fit text within given dimensions.
    
    This function calculates the optimal font size to fit text within
    given width and height constraints.
    
    Args:
        text: The text to fit
        width: Available width in pixels
        height: Available height in pixels
        font_family: Font family to use
        min_size: Minimum font size
        max_size: Maximum font size
        
    Returns:
        Optimal font size
        
    Examples:
        >>> size = calculate_optimal_font_size("Long equation text", 400, 100)
        >>> label.setFont(QFont("Arial", size))
    """
    # Binary search for optimal font size
    low = min_size
    high = max_size
    
    while low <= high:
        mid = (low + high) // 2
        font = QFont(font_family, mid)
        metrics = QFontMetrics(font)
        
        text_width = metrics.horizontalAdvance(text)
        text_height = metrics.height()
        
        if text_width <= width and text_height <= height:
            low = mid + 1
        else:
            high = mid - 1
    
    # Return the largest size that fits
    return max(min_size, high)


def create_responsive_label(text: str, parent: Optional[QWidget] = None) -> QLabel:
    """
    Create a QLabel with responsive font sizing.
    
    This function creates a QLabel that automatically adjusts its font size
    to fit its container when resized.
    
    Args:
        text: The text for the label
        parent: The parent widget
        
    Returns:
        QLabel with responsive font sizing
        
    Examples:
        >>> label = create_responsive_label("Responsive text", parent_widget)
        >>> layout.addWidget(label)
    """
    label = QLabel(text, parent)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    
    # Create a resize event handler
    def resize_event(event):
        # Calculate optimal font size
        width = label.width()
        height = label.height()
        font_size = calculate_optimal_font_size(text, width, height)
        
        # Set the font
        font = label.font()
        font.setPointSize(font_size)
        label.setFont(font)
    
    # Connect the resize event
    label.resizeEvent = resize_event
    
    return label


def format_model_equation(model_results: Dict[str, Any]) -> str:
    """
    Format a model equation string from model results.
    
    This function formats a LaTeX equation string for an ARMAX model
    based on the model results dictionary.
    
    Args:
        model_results: Dictionary containing model estimation results
        
    Returns:
        Formatted LaTeX equation string
        
    Examples:
        >>> equation = format_model_equation(model_results)
        >>> fig = create_equation_figure(equation)
    """
    try:
        # Extract model parameters
        parameters = model_results.get('parameters', [])
        
        # If an equation is already provided, use it
        if 'equation' in model_results:
            return model_results['equation']
        
        # Otherwise, build the equation from parameters
        ar_terms = []
        ma_terms = []
        constant_term = None
        exog_terms = []
        
        for param in parameters:
            name = param.get('name', '')
            estimate = param.get('estimate', 0.0)
            
            # Skip parameters with zero estimates
            if abs(estimate) < 1e-10:
                continue
            
            # Format the coefficient with sign
            coef = f"{abs(estimate):.3f}"
            sign = "+" if estimate > 0 else "-"
            
            # Identify parameter type
            if name.lower() == 'constant':
                constant_term = f"{sign} {coef}"
            elif name.lower().startswith('ar'):
                # Extract lag from AR{lag}
                lag_match = re.search(r'\{(\d+)\}', name)
                lag = lag_match.group(1) if lag_match else "1"
                ar_terms.append(f"{sign} {coef} y_{{t-{lag}}}")
            elif name.lower().startswith('ma'):
                # Extract lag from MA{lag}
                lag_match = re.search(r'\{(\d+)\}', name)
                lag = lag_match.group(1) if lag_match else "1"
                ma_terms.append(f"{sign} {coef} \varepsilon_{{t-{lag}}}")
            else:
                # Assume exogenous variable
                exog_terms.append(f"{sign} {coef} {name}_t")
        
        # Build the equation
        equation_parts = []
        
        # Start with y_t =
        equation_parts.append("y_t = ")
        
        # Add constant term if present
        if constant_term:
            # Remove leading + sign if it's the first term
            if constant_term.startswith('+'):
                constant_term = constant_term[2:]
            equation_parts.append(constant_term)
        
        # Add AR terms
        for term in ar_terms:
            equation_parts.append(term)
        
        # Add exogenous terms
        for term in exog_terms:
            equation_parts.append(term)
        
        # Add error term
        equation_parts.append("+ \varepsilon_t")
        
        # Add MA terms
        for term in ma_terms:
            equation_parts.append(term)
        
        # Join the equation parts
        equation = " ".join(equation_parts)
        
        return equation
    except Exception as e:
        logger.error(f"Error formatting model equation: {e}")
        return r"y_t = \beta_0 + \sum_{i=1}^{p} \beta_i y_{t-i} + \varepsilon_t + \sum_{j=1}^{q} \theta_j \varepsilon_{t-j}"


def show_error_message(parent: QWidget, title: str, message: str, 
                      details: Optional[str] = None) -> None:
    """
    Show an error message dialog.
    
    This function displays an error message dialog with optional details.
    
    Args:
        parent: The parent widget
        title: The dialog title
        message: The error message
        details: Optional detailed error information
        
    Examples:
        >>> show_error_message(
        ...     self, "Estimation Error", 
        ...     "Failed to estimate model parameters",
        ...     "ValueError: Invalid parameter values"
        ... )
    """
    error_dialog = QMessageBox(parent)
    error_dialog.setIcon(QMessageBox.Icon.Critical)
    error_dialog.setWindowTitle(title)
    error_dialog.setText(message)
    
    if details:
        error_dialog.setDetailedText(details)
    
    error_dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
    error_dialog.exec()


def show_warning_message(parent: QWidget, title: str, message: str, 
                        details: Optional[str] = None) -> None:
    """
    Show a warning message dialog.
    
    This function displays a warning message dialog with optional details.
    
    Args:
        parent: The parent widget
        title: The dialog title
        message: The warning message
        details: Optional detailed warning information
        
    Examples:
        >>> show_warning_message(
        ...     self, "Convergence Warning", 
        ...     "Model estimation may not have fully converged",
        ...     "Reached maximum iterations (100) with gradient norm 1e-4"
        ... )
    """
    warning_dialog = QMessageBox(parent)
    warning_dialog.setIcon(QMessageBox.Icon.Warning)
    warning_dialog.setWindowTitle(title)
    warning_dialog.setText(message)
    
    if details:
        warning_dialog.setDetailedText(details)
    
    warning_dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
    warning_dialog.exec()


def show_confirmation_dialog(parent: QWidget, title: str, message: str) -> bool:
    """
    Show a confirmation dialog and return the user's choice.
    
    This function displays a confirmation dialog and returns True if
    the user confirms, or False if they cancel.
    
    Args:
        parent: The parent widget
        title: The dialog title
        message: The confirmation message
        
    Returns:
        True if confirmed, False if cancelled
        
    Examples:
        >>> if show_confirmation_dialog(
        ...     self, "Confirm Close", 
        ...     "Are you sure you want to close? Unsaved changes will be lost."
        ... ):
        ...     self.close()
    """
    confirm_dialog = QMessageBox(parent)
    confirm_dialog.setIcon(QMessageBox.Icon.Question)
    confirm_dialog.setWindowTitle(title)
    confirm_dialog.setText(message)
    confirm_dialog.setStandardButtons(
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
    )
    confirm_dialog.setDefaultButton(QMessageBox.StandardButton.No)
    
    result = confirm_dialog.exec()
    return result == QMessageBox.StandardButton.Yes


def format_time_delta(seconds: float) -> str:
    """
    Format a time delta in seconds to a human-readable string.
    
    This function formats a time delta in seconds to a human-readable
    string (e.g., "2 minutes 30 seconds").
    
    Args:
        seconds: Time delta in seconds
        
    Returns:
        Formatted time string
        
    Examples:
        >>> print(format_time_delta(150))
        '2 minutes 30 seconds'
    """
    if seconds < 0:
        return "unknown"
    
    if seconds < 1:
        return "less than a second"
    
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    
    parts = []
    
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    
    if seconds > 0 and hours == 0:  # Only show seconds if less than an hour
        parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")
    
    return " ".join(parts)


class DebounceTimer(QObject):
    """
    Timer for debouncing UI events.
    
    This class provides a timer for debouncing UI events, such as text input
    or resize events, to avoid excessive processing.
    
    Attributes:
        timeout: Signal emitted when the debounce timer expires
        
    Examples:
        >>> debouncer = DebounceTimer()
        >>> debouncer.timeout.connect(self.process_input)
        >>> 
        >>> def on_text_changed(text):
        ...     debouncer.debounce(300)  # Wait 300ms after typing stops
    """
    
    timeout = pyqtSignal()
    
    def __init__(self, parent=None):
        """Initialize the debounce timer."""
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.timeout)
    
    def debounce(self, msec: int = 300) -> None:
        """
        Start or restart the debounce timer.
        
        Args:
            msec: Debounce timeout in milliseconds
        """
        self._timer.stop()
        self._timer.start(msec)
    
    def cancel(self) -> None:
        """Cancel the debounce timer."""
        self._timer.stop()


def create_color_scale(n: int, cmap_name: str = 'viridis') -> List[str]:
    """
    Create a list of color hex codes from a matplotlib colormap.
    
    This function creates a list of color hex codes from a matplotlib
    colormap, useful for creating color-coded UI elements.
    
    Args:
        n: Number of colors to generate
        cmap_name: Name of the matplotlib colormap
        
    Returns:
        List of color hex codes
        
    Examples:
        >>> colors = create_color_scale(5, 'plasma')
        >>> for i, color in enumerate(colors):
        ...     button = QPushButton(f"Button {i+1}")
        ...     button.setStyleSheet(f"background-color: {color}")
    """
    cmap = plt.get_cmap(cmap_name)
    colors = []
    
    for i in range(n):
        rgba = cmap(i / max(1, n - 1))
        # Convert to hex color code
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        )
        colors.append(hex_color)
    
    return colors


def dataframe_to_table(df: pd.DataFrame, table: QTableWidget) -> None:
    """
    Populate a QTableWidget with data from a pandas DataFrame.
    
    This function populates a QTableWidget with data from a pandas DataFrame,
    preserving column names and data types.
    
    Args:
        df: pandas DataFrame
        table: QTableWidget to populate
        
    Examples:
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
        >>> dataframe_to_table(df, table_widget)
    """
    try:
        # Clear the table
        table.setRowCount(0)
        table.setColumnCount(0)
        
        # Set column headers
        table.setColumnCount(len(df.columns))
        table.setHorizontalHeaderLabels(df.columns.astype(str).tolist())
        
        # Set row count
        table.setRowCount(len(df))
        
        # Populate the table
        for row in range(len(df)):
            for col, column in enumerate(df.columns):
                value = df.iloc[row, col]
                
                # Format the value based on its type
                if pd.isna(value):
                    item = QTableWidgetItem("")
                elif isinstance(value, (int, float, np.number)):
                    item = QTableWidgetItem(f"{value}")
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                else:
                    item = QTableWidgetItem(str(value))
                
                table.setItem(row, col, item)
        
        # Resize columns to content
        table.resizeColumnsToContents()
    except Exception as e:
        logger.error(f"Error converting DataFrame to table: {e}")
        # Clear the table and show error
        table.setRowCount(1)
        table.setColumnCount(1)
        table.setHorizontalHeaderLabels(["Error"])
        error_item = QTableWidgetItem(f"Error loading data: {str(e)}")
        error_item.setForeground(QColor(255, 0, 0))
        table.setItem(0, 0, error_item)


def array_to_table(arr: np.ndarray, table: QTableWidget, 
                  row_labels: Optional[List[str]] = None,
                  col_labels: Optional[List[str]] = None) -> None:
    """
    Populate a QTableWidget with data from a NumPy array.
    
    This function populates a QTableWidget with data from a NumPy array,
    with optional row and column labels.
    
    Args:
        arr: NumPy array
        table: QTableWidget to populate
        row_labels: Optional list of row labels
        col_labels: Optional list of column labels
        
    Examples:
        >>> arr = np.array([[1, 2, 3], [4, 5, 6]])
        >>> array_to_table(arr, table_widget, ['Row 1', 'Row 2'], ['A', 'B', 'C'])
    """
    try:
        # Clear the table
        table.setRowCount(0)
        table.setColumnCount(0)
        
        # Ensure the array is 2D
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        
        # Set row and column counts
        rows, cols = arr.shape
        table.setRowCount(rows)
        table.setColumnCount(cols)
        
        # Set column headers if provided
        if col_labels:
            if len(col_labels) != cols:
                logger.warning(f"Column labels length ({len(col_labels)}) doesn't match array columns ({cols})")
                col_labels = col_labels[:cols] if len(col_labels) > cols else col_labels + [f"Col {i+1}" for i in range(len(col_labels), cols)]
            table.setHorizontalHeaderLabels(col_labels)
        
        # Set row headers if provided
        if row_labels:
            if len(row_labels) != rows:
                logger.warning(f"Row labels length ({len(row_labels)}) doesn't match array rows ({rows})")
                row_labels = row_labels[:rows] if len(row_labels) > rows else row_labels + [f"Row {i+1}" for i in range(len(row_labels), rows)]
            table.setVerticalHeaderLabels(row_labels)
        
        # Populate the table
        for row in range(rows):
            for col in range(cols):
                value = arr[row, col]
                
                # Format the value based on its type
                if np.isnan(value):
                    item = QTableWidgetItem("")
                elif isinstance(value, (int, float, np.number)):
                    item = QTableWidgetItem(f"{value}")
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                else:
                    item = QTableWidgetItem(str(value))
                
                table.setItem(row, col, item)
        
        # Resize columns to content
        table.resizeColumnsToContents()
    except Exception as e:
        logger.error(f"Error converting array to table: {e}")
        # Clear the table and show error
        table.setRowCount(1)
        table.setColumnCount(1)
        table.setHorizontalHeaderLabels(["Error"])
        error_item = QTableWidgetItem(f"Error loading data: {str(e)}")
        error_item.setForeground(QColor(255, 0, 0))
        table.setItem(0, 0, error_item)
