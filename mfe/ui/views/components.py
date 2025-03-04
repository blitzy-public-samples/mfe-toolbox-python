'''
Reusable UI components for the MFE Toolbox.

This module provides a collection of specialized PyQt6 widget classes that encapsulate
common UI patterns used across the MFE Toolbox interface. These components ensure
consistency in appearance and behavior while reducing code duplication in view classes.

The components include parameter input widgets with validation, plot containers for
embedding matplotlib visualizations, parameter tables with formatting, equation displays
with LaTeX rendering, statistics panels, and progress indicators with asynchronous support.
'''

import asyncio
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast, TypeVar

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt5Agg backend for matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from PyQt6.QtCore import (
    Qt, QSize, QTimer, QObject, QEvent, QRegularExpression, pyqtSignal, pyqtSlot, QPoint
)
from PyQt6.QtGui import (
    QColor, QFont, QFontMetrics, QPalette, QPixmap, QRegularExpressionValidator
)
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox,
    QTabWidget, QGroupBox, QSplitter, QFrame, QScrollArea, QSizePolicy, QTableWidget,
    QTableWidgetItem, QHeaderView, QProgressBar, QToolTip, QSlider, QAbstractSpinBox,
    QStackedWidget, QTextEdit, QToolButton, QMenu, QAction
)

from mfe.core.exceptions import UIError
from mfe.core.types import UICallback, UIEventHandler, UIUpdateFunction
from mfe.ui.utils import (
    format_latex_equation, create_equation_figure, create_parameter_table,
    create_statistics_table, AsyncHelper, ProgressReporter, calculate_optimal_font_size
)

# Set up module-level logger
logger = logging.getLogger("mfe.ui.views.components")

# Type variable for generic function typing
T = TypeVar('T')


class NumericInputField(QLineEdit):
    """
    Input field for numeric values with validation and formatting.
    
    This widget extends QLineEdit to provide specialized handling for numeric
    input with validation, formatting, and unit display. It supports different
    numeric types (integer, float) and can display units alongside the value.
    
    Attributes:
        valueChanged: Signal emitted when the numeric value changes
        validationChanged: Signal emitted when validation state changes
    """
    
    valueChanged = pyqtSignal(object)  # Emits the numeric value (int or float)
    validationChanged = pyqtSignal(bool)  # Emits whether the input is valid
    
    def __init__(
        self, 
        parent: Optional[QWidget] = None,
        value_type: str = "float",
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        decimals: int = 4,
        unit: str = "",
        placeholder: str = "",
        tooltip: str = ""
    ):
        """
        Initialize the numeric input field.
        
        Args:
            parent: Parent widget
            value_type: Type of numeric value ("int" or "float")
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            decimals: Number of decimal places for float values
            unit: Unit to display after the value
            placeholder: Placeholder text
            tooltip: Tooltip text
        """
        super().__init__(parent)
        
        self._value_type = value_type.lower()
        self._min_value = min_value
        self._max_value = max_value
        self._decimals = max(0, decimals)
        self._unit = unit
        self._is_valid = False
        self._current_value = None
        
        # Set up the input field
        self.setPlaceholderText(placeholder)
        if tooltip:
            self.setToolTip(tooltip)
        
        # Set up validator based on value type
        if self._value_type == "int":
            # Integer validator
            pattern = r"^[-+]?\d+$"
            if min_value is not None and min_value >= 0:
                pattern = r"^\+?\d+$"
            validator = QRegularExpressionValidator(QRegularExpression(pattern), self)
            self.setValidator(validator)
        else:
            # Float validator
            pattern = r"^[-+]?\d*\.?\d+([eE][-+]?\d+)?$"
            if min_value is not None and min_value >= 0:
                pattern = r"^\+?\d*\.?\d+([eE][-+]?\d+)?$"
            validator = QRegularExpressionValidator(QRegularExpression(pattern), self)
            self.setValidator(validator)
        
        # Connect signals
        self.textChanged.connect(self._on_text_changed)
        self.editingFinished.connect(self._on_editing_finished)
        
        # Set initial style
        self._update_style()
    
    def _on_text_changed(self, text: str) -> None:
        """
        Handle text changes in the input field.
        
        Args:
            text: Current text in the input field
        """
        # Check if the text is empty
        if not text:
            self._is_valid = False
            self._current_value = None
            self._update_style()
            self.validationChanged.emit(False)
            return
        
        # Parse the value
        try:
            if self._value_type == "int":
                value = int(text)
            else:
                value = float(text)
            
            # Check bounds
            if (self._min_value is not None and value < self._min_value) or \
               (self._max_value is not None and value > self._max_value):
                self._is_valid = False
                self._current_value = value  # Store the value even if invalid
            else:
                self._is_valid = True
                self._current_value = value
                self.valueChanged.emit(value)
        except ValueError:
            self._is_valid = False
            self._current_value = None
        
        # Update the style
        self._update_style()
        self.validationChanged.emit(self._is_valid)
    
    def _on_editing_finished(self) -> None:
        """Handle completion of editing."""
        # Format the value if valid
        if self._is_valid and self._current_value is not None:
            self._format_value()
    
    def _format_value(self) -> None:
        """Format the displayed value according to settings."""
        if self._current_value is None:
            return
        
        # Format the value based on type
        if self._value_type == "int":
            formatted = str(self._current_value)
        else:
            # Format float with specified decimals
            formatted = f"{self._current_value:.{self._decimals}f}"
        
        # Add unit if specified
        if self._unit:
            formatted = f"{formatted} {self._unit}"
        
        # Update the text without triggering textChanged
        self.blockSignals(True)
        self.setText(formatted)
        self.blockSignals(False)
    
    def _update_style(self) -> None:
        """Update the widget style based on validation state."""
        if not self.text():
            # Empty field - neutral style
            self.setStyleSheet("")
        elif self._is_valid:
            # Valid input - green border
            self.setStyleSheet("border: 1px solid green;")
        else:
            # Invalid input - red border
            self.setStyleSheet("border: 1px solid red;")
    
    def setValue(self, value: Optional[Union[int, float]]) -> None:
        """
        Set the numeric value of the input field.
        
        Args:
            value: Numeric value to set, or None to clear
        """
        if value is None:
            self.clear()
            self._is_valid = False
            self._current_value = None
        else:
            # Convert to the correct type
            if self._value_type == "int":
                value = int(value)
            else:
                value = float(value)
            
            # Check bounds
            if (self._min_value is not None and value < self._min_value) or \
               (self._max_value is not None and value > self._max_value):
                logger.warning(f"Value {value} is outside allowed range [{self._min_value}, {self._max_value}]")
            
            self._current_value = value
            self._is_valid = True
            self._format_value()
        
        # Update style
        self._update_style()
        self.validationChanged.emit(self._is_valid)
    
    def value(self) -> Optional[Union[int, float]]:
        """
        Get the current numeric value.
        
        Returns:
            Current numeric value, or None if invalid
        """
        return self._current_value if self._is_valid else None
    
    def isValid(self) -> bool:
        """
        Check if the current input is valid.
        
        Returns:
            True if the input is valid, False otherwise
        """
        return self._is_valid
    
    def setMinimum(self, min_value: Optional[Union[int, float]]) -> None:
        """
        Set the minimum allowed value.
        
        Args:
            min_value: Minimum allowed value, or None for no minimum
        """
        self._min_value = min_value
        self._on_text_changed(self.text())  # Re-validate
    
    def setMaximum(self, max_value: Optional[Union[int, float]]) -> None:
        """
        Set the maximum allowed value.
        
        Args:
            max_value: Maximum allowed value, or None for no maximum
        """
        self._max_value = max_value
        self._on_text_changed(self.text())  # Re-validate
    
    def setDecimals(self, decimals: int) -> None:
        """
        Set the number of decimal places for float values.
        
        Args:
            decimals: Number of decimal places
        """
        self._decimals = max(0, decimals)
        if self._is_valid and self._current_value is not None:
            self._format_value()
    
    def setUnit(self, unit: str) -> None:
        """
        Set the unit to display after the value.
        
        Args:
            unit: Unit string
        """
        self._unit = unit
        if self._is_valid and self._current_value is not None:
            self._format_value()


class PlotContainer(QWidget):
    """
    Container widget for embedding matplotlib plots in PyQt6 interfaces.
    
    This widget provides a container for matplotlib plots with an integrated
    navigation toolbar and optional title. It handles the creation and
    management of the matplotlib figure, canvas, and toolbar.
    
    Attributes:
        figure: The matplotlib Figure instance
        canvas: The FigureCanvas instance
        toolbar: The NavigationToolbar instance
    """
    
    def __init__(
        self, 
        parent: Optional[QWidget] = None,
        title: str = "",
        figsize: Tuple[float, float] = (6, 4),
        dpi: int = 100,
        show_toolbar: bool = True
    ):
        """
        Initialize the plot container.
        
        Args:
            parent: Parent widget
            title: Optional title for the plot container
            figsize: Figure size in inches (width, height)
            dpi: Dots per inch for the figure
            show_toolbar: Whether to show the navigation toolbar
        """
        super().__init__(parent)
        
        # Create layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        # Create title label if specified
        if title:
            self.title_label = QLabel(title)
            self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.title_label.setStyleSheet("font-weight: bold;")
            self.layout.addWidget(self.title_label)
        else:
            self.title_label = None
        
        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=figsize, dpi=dpi)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(self)
        
        # Configure canvas size policy
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Add canvas to layout
        self.layout.addWidget(self.canvas)
        
        # Create navigation toolbar if requested
        if show_toolbar:
            self.toolbar = NavigationToolbar(self.canvas, self)
            self.layout.addWidget(self.toolbar)
        else:
            self.toolbar = None
        
        # Set up figure with tight layout
        self.figure.set_tight_layout(True)
    
    def add_subplot(self, *args, **kwargs) -> plt.Axes:
        """
        Add a subplot to the figure.
        
        Args:
            *args: Positional arguments for add_subplot
            **kwargs: Keyword arguments for add_subplot
            
        Returns:
            The created subplot axes
        """
        return self.figure.add_subplot(*args, **kwargs)
    
    def clear(self) -> None:
        """Clear all subplots in the figure."""
        self.figure.clear()
        self.canvas.draw()
    
    def draw(self) -> None:
        """Redraw the canvas."""
        self.canvas.draw()
    
    def save_figure(self, filename: str, **kwargs) -> None:
        """
        Save the figure to a file.
        
        Args:
            filename: File path to save to
            **kwargs: Additional arguments for savefig
        """
        self.figure.savefig(filename, **kwargs)
    
    def set_title(self, title: str) -> None:
        """
        Set the container title.
        
        Args:
            title: Title text
        """
        if self.title_label is None:
            # Create title label if it doesn't exist
            self.title_label = QLabel(title)
            self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.title_label.setStyleSheet("font-weight: bold;")
            self.layout.insertWidget(0, self.title_label)
        else:
            # Update existing title
            self.title_label.setText(title)


class ParameterTable(QWidget):
    """
    Widget for displaying model parameters with formatting and pagination.
    
    This widget displays model parameters in a table format with proper
    formatting, significance indicators, and pagination for large parameter sets.
    
    Attributes:
        pageChanged: Signal emitted when the current page changes
    """
    
    pageChanged = pyqtSignal(int)
    
    def __init__(
        self, 
        parent: Optional[QWidget] = None,
        title: str = "Parameter Estimates",
        items_per_page: int = 10
    ):
        """
        Initialize the parameter table.
        
        Args:
            parent: Parent widget
            title: Title for the parameter table
            items_per_page: Number of parameters to display per page
        """
        super().__init__(parent)
        
        # Store settings
        self._title = title
        self._items_per_page = items_per_page
        self._current_page = 1
        self._total_pages = 1
        self._parameters = []
        
        # Create layout
        self.layout = QVBoxLayout(self)
        
        # Create group box
        self.group_box = QGroupBox(title)
        self.group_box_layout = QVBoxLayout(self.group_box)
        
        # Create table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            ["Parameter", "Estimate", "Std. Error", "t-Stat", "p-Value"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.group_box_layout.addWidget(self.table)
        
        # Create pagination controls
        self.pagination_layout = QHBoxLayout()
        
        # Previous page button
        self.prev_button = QPushButton("< Previous")
        self.prev_button.setEnabled(False)
        self.prev_button.clicked.connect(self._on_prev_page)
        self.pagination_layout.addWidget(self.prev_button)
        
        # Page indicator
        self.page_label = QLabel("Page 1 of 1")
        self.page_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pagination_layout.addWidget(self.page_label, 1)
        
        # Next page button
        self.next_button = QPushButton("Next >")
        self.next_button.setEnabled(False)
        self.next_button.clicked.connect(self._on_next_page)
        self.pagination_layout.addWidget(self.next_button)
        
        # Add pagination controls to group box
        self.group_box_layout.addLayout(self.pagination_layout)
        
        # Add significance legend
        self.legend_label = QLabel(
            "<small>Significance levels: *** p<0.01, ** p<0.05, * p<0.1</small>"
        )
        self.legend_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.group_box_layout.addWidget(self.legend_label)
        
        # Add group box to main layout
        self.layout.addWidget(self.group_box)
    
    def _on_prev_page(self) -> None:
        """Handle previous page button click."""
        if self._current_page > 1:
            self._current_page -= 1
            self._update_table()
            self.pageChanged.emit(self._current_page)
    
    def _on_next_page(self) -> None:
        """Handle next page button click."""
        if self._current_page < self._total_pages:
            self._current_page += 1
            self._update_table()
            self.pageChanged.emit(self._current_page)
    
    def _update_table(self) -> None:
        """Update the table with current page data."""
        # Update the table
        self._current_page, self._total_pages = create_parameter_table(
            self._parameters, 
            self.table, 
            self._current_page, 
            self._items_per_page
        )
        
        # Update pagination controls
        self.page_label.setText(f"Page {self._current_page} of {self._total_pages}")
        self.prev_button.setEnabled(self._current_page > 1)
        self.next_button.setEnabled(self._current_page < self._total_pages)
    
    def setParameters(self, parameters: List[Dict[str, Any]]) -> None:
        """
        Set the parameters to display.
        
        Args:
            parameters: List of parameter dictionaries with keys 'name', 'estimate',
                       'std_error', 't_stat', and 'p_value'
        """
        self._parameters = parameters
        self._current_page = 1
        self._update_table()
    
    def setTitle(self, title: str) -> None:
        """
        Set the table title.
        
        Args:
            title: Title text
        """
        self._title = title
        self.group_box.setTitle(title)
    
    def setItemsPerPage(self, items_per_page: int) -> None:
        """
        Set the number of items to display per page.
        
        Args:
            items_per_page: Number of items per page
        """
        self._items_per_page = max(1, items_per_page)
        self._update_table()
    
    def currentPage(self) -> int:
        """
        Get the current page number.
        
        Returns:
            Current page number (1-based)
        """
        return self._current_page
    
    def totalPages(self) -> int:
        """
        Get the total number of pages.
        
        Returns:
            Total number of pages
        """
        return self._total_pages
    
    def goToPage(self, page: int) -> None:
        """
        Go to a specific page.
        
        Args:
            page: Page number (1-based)
        """
        page = max(1, min(page, self._total_pages))
        if page != self._current_page:
            self._current_page = page
            self._update_table()
            self.pageChanged.emit(self._current_page)


class EquationDisplay(QWidget):
    """
    Widget for displaying LaTeX-rendered mathematical equations.
    
    This widget displays mathematical equations rendered using matplotlib's
    LaTeX support. It provides a clean, professional display for model equations
    and mathematical formulas.
    """
    
    def __init__(
        self, 
        parent: Optional[QWidget] = None,
        equation: str = "",
        title: str = "Model Equation",
        figsize: Tuple[float, float] = (6, 1.5),
        dpi: int = 100,
        fontsize: int = 12
    ):
        """
        Initialize the equation display.
        
        Args:
            parent: Parent widget
            equation: LaTeX equation string
            title: Title for the equation display
            figsize: Figure size in inches (width, height)
            dpi: Dots per inch for the figure
            fontsize: Font size for the equation
        """
        super().__init__(parent)
        
        # Store settings
        self._equation = equation
        self._title = title
        self._figsize = figsize
        self._dpi = dpi
        self._fontsize = fontsize
        
        # Create layout
        self.layout = QVBoxLayout(self)
        
        # Create group box
        self.group_box = QGroupBox(title)
        self.group_box_layout = QVBoxLayout(self.group_box)
        
        # Create equation display
        if equation:
            self._create_equation_figure()
        else:
            # Create placeholder label
            self.placeholder_label = QLabel("No equation available")
            self.placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.group_box_layout.addWidget(self.placeholder_label)
            self.canvas = None
        
        # Add group box to main layout
        self.layout.addWidget(self.group_box)
    
    def _create_equation_figure(self) -> None:
        """Create the matplotlib figure for the equation."""
        try:
            # Create equation figure
            fig = create_equation_figure(
                self._equation,
                figsize=self._figsize,
                dpi=self._dpi,
                fontsize=self._fontsize
            )
            
            # Create canvas
            self.canvas = FigureCanvas(fig)
            self.canvas.setParent(self)
            
            # Configure canvas size policy
            self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.canvas.setMinimumHeight(int(self._figsize[1] * self._dpi * 0.8))
            
            # Clear existing widgets from group box layout
            while self.group_box_layout.count():
                item = self.group_box_layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
            
            # Add canvas to group box layout
            self.group_box_layout.addWidget(self.canvas)
        except Exception as e:
            logger.error(f"Error creating equation figure: {e}")
            # Create error label
            error_label = QLabel(f"Error rendering equation: {str(e)}")
            error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            error_label.setStyleSheet("color: red;")
            
            # Clear existing widgets from group box layout
            while self.group_box_layout.count():
                item = self.group_box_layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
            
            # Add error label to group box layout
            self.group_box_layout.addWidget(error_label)
            self.canvas = None
    
    def setEquation(self, equation: str) -> None:
        """
        Set the equation to display.
        
        Args:
            equation: LaTeX equation string
        """
        self._equation = equation
        if equation:
            self._create_equation_figure()
        else:
            # Create placeholder label
            placeholder_label = QLabel("No equation available")
            placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            # Clear existing widgets from group box layout
            while self.group_box_layout.count():
                item = self.group_box_layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
            
            # Add placeholder label to group box layout
            self.group_box_layout.addWidget(placeholder_label)
            self.canvas = None
    
    def setTitle(self, title: str) -> None:
        """
        Set the display title.
        
        Args:
            title: Title text
        """
        self._title = title
        self.group_box.setTitle(title)
    
    def setFigureSize(self, figsize: Tuple[float, float]) -> None:
        """
        Set the figure size.
        
        Args:
            figsize: Figure size in inches (width, height)
        """
        self._figsize = figsize
        if self._equation:
            self._create_equation_figure()
    
    def setFontSize(self, fontsize: int) -> None:
        """
        Set the equation font size.
        
        Args:
            fontsize: Font size for the equation
        """
        self._fontsize = fontsize
        if self._equation:
            self._create_equation_figure()


class ModelStatisticsPanel(QWidget):
    """
    Widget for displaying model statistics in a formatted table.
    
    This widget displays model statistics in a table format with proper
    formatting and organization. It groups common statistics together
    and provides explanatory notes.
    """
    
    def __init__(
        self, 
        parent: Optional[QWidget] = None,
        title: str = "Model Statistics",
        statistics: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the model statistics panel.
        
        Args:
            parent: Parent widget
            title: Title for the statistics panel
            statistics: Dictionary of statistics with statistic names as keys
        """
        super().__init__(parent)
        
        # Store settings
        self._title = title
        self._statistics = statistics or {}
        
        # Create layout
        self.layout = QVBoxLayout(self)
        
        # Create group box
        self.group_box = QGroupBox(title)
        self.group_box_layout = QVBoxLayout(self.group_box)
        
        # Create table
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(
            ["Statistic", "Value", "Notes"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.group_box_layout.addWidget(self.table)
        
        # Add group box to main layout
        self.layout.addWidget(self.group_box)
        
        # Populate table if statistics are provided
        if statistics:
            self._update_table()
    
    def _update_table(self) -> None:
        """Update the table with current statistics."""
        create_statistics_table(self._statistics, self.table)
    
    def setStatistics(self, statistics: Dict[str, Any]) -> None:
        """
        Set the statistics to display.
        
        Args:
            statistics: Dictionary of statistics with statistic names as keys
        """
        self._statistics = statistics
        self._update_table()
    
    def setTitle(self, title: str) -> None:
        """
        Set the panel title.
        
        Args:
            title: Title text
        """
        self._title = title
        self.group_box.setTitle(title)
    
    def addStatistic(self, name: str, value: Any, update: bool = True) -> None:
        """
        Add a single statistic to the display.
        
        Args:
            name: Statistic name
            value: Statistic value
            update: Whether to update the table immediately
        """
        self._statistics[name] = value
        if update:
            self._update_table()
    
    def removeStatistic(self, name: str, update: bool = True) -> None:
        """
        Remove a statistic from the display.
        
        Args:
            name: Statistic name
            update: Whether to update the table immediately
        """
        if name in self._statistics:
            del self._statistics[name]
            if update:
                self._update_table()
    
    def clearStatistics(self) -> None:
        """Clear all statistics from the display."""
        self._statistics = {}
        self._update_table()


class AsyncProgressIndicator(QWidget):
    """
    Progress indicator for asynchronous operations with cancellation support.
    
    This widget provides a progress bar and status message for tracking
    asynchronous operations, with support for cancellation and detailed
    progress reporting.
    
    Attributes:
        canceled: Signal emitted when the user cancels the operation
        progressUpdated: Signal emitted when progress is updated
    """
    
    canceled = pyqtSignal()
    progressUpdated = pyqtSignal(float, str)
    
    def __init__(
        self, 
        parent: Optional[QWidget] = None,
        title: str = "Operation Progress",
        cancellable: bool = True,
        show_percent: bool = True,
        show_time_estimate: bool = False
    ):
        """
        Initialize the progress indicator.
        
        Args:
            parent: Parent widget
            title: Title for the progress indicator
            cancellable: Whether the operation can be cancelled
            show_percent: Whether to show percentage in the progress bar
            show_time_estimate: Whether to show time estimates
        """
        super().__init__(parent)
        
        # Store settings
        self._title = title
        self._cancellable = cancellable
        self._show_percent = show_percent
        self._show_time_estimate = show_time_estimate
        self._start_time = None
        self._reporter = ProgressReporter()
        
        # Create layout
        self.layout = QVBoxLayout(self)
        
        # Create group box
        self.group_box = QGroupBox(title)
        self.group_box_layout = QVBoxLayout(self.group_box)
        
        # Create status label
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.group_box_layout.addWidget(self.status_label)
        
        # Create progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(show_percent)
        self.group_box_layout.addWidget(self.progress_bar)
        
        # Create time estimate label if requested
        if show_time_estimate:
            self.time_label = QLabel("Estimated time remaining: --")
            self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.group_box_layout.addWidget(self.time_label)
        else:
            self.time_label = None
        
        # Create cancel button if cancellable
        if cancellable:
            self.cancel_button = QPushButton("Cancel")
            self.cancel_button.clicked.connect(self._on_cancel_clicked)
            self.group_box_layout.addWidget(self.cancel_button)
        else:
            self.cancel_button = None
        
        # Add group box to main layout
        self.layout.addWidget(self.group_box)
        
        # Set up progress reporter
        self._reporter.set_callback(self._on_progress_update)
        
        # Hide initially
        self.setVisible(False)
    
    def _on_cancel_clicked(self) -> None:
        """Handle cancel button click."""
        self.canceled.emit()
    
    def _on_progress_update(self, percent: float, message: Optional[str]) -> None:
        """
        Handle progress update from reporter.
        
        Args:
            percent: Progress percentage (0-100)
            message: Progress message
        """
        # Update progress bar
        self.progress_bar.setValue(int(percent))
        
        # Update status message if provided
        if message:
            self.status_label.setText(message)
        
        # Update time estimate if enabled
        if self._show_time_estimate and self.time_label and self._start_time:
            remaining = self._reporter.estimate_remaining(percent)
            if remaining >= 0:
                minutes, seconds = divmod(int(remaining), 60)
                if minutes > 0:
                    time_text = f"{minutes}m {seconds}s"
                else:
                    time_text = f"{seconds}s"
                self.time_label.setText(f"Estimated time remaining: {time_text}")
        
        # Emit progress updated signal
        self.progressUpdated.emit(percent, message or self.status_label.text())
        
        # Process events to update UI
        QApplication.processEvents()
    
    def start(self, message: str = "Starting operation...") -> None:
        """
        Start tracking a new operation.
        
        Args:
            message: Initial status message
        """
        # Reset progress
        self.progress_bar.setValue(0)
        self.status_label.setText(message)
        
        # Reset time tracking
        self._start_time = asyncio.get_event_loop().time()
        if self._show_time_estimate and self.time_label:
            self.time_label.setText("Estimated time remaining: calculating...")
        
        # Show the widget
        self.setVisible(True)
        
        # Process events to update UI
        QApplication.processEvents()
    
    def update(self, percent: float, message: Optional[str] = None) -> None:
        """
        Update progress status.
        
        Args:
            percent: Progress percentage (0-100)
            message: Progress message
        """
        self._on_progress_update(percent, message)
    
    def finish(self, message: str = "Operation completed") -> None:
        """
        Finish the operation.
        
        Args:
            message: Completion message
        """
        # Update progress to 100%
        self.progress_bar.setValue(100)
        self.status_label.setText(message)
        
        # Clear time estimate
        if self._show_time_estimate and self.time_label:
            self.time_label.setText("Completed")
        
        # Process events to update UI
        QApplication.processEvents()
        
        # Hide after a short delay
        QTimer.singleShot(1000, self.hide)
    
    def error(self, message: str = "Operation failed") -> None:
        """
        Show an error status.
        
        Args:
            message: Error message
        """
        # Update status
        self.status_label.setText(message)
        self.status_label.setStyleSheet("color: red;")
        
        # Clear time estimate
        if self._show_time_estimate and self.time_label:
            self.time_label.setText("Failed")
        
        # Process events to update UI
        QApplication.processEvents()
        
        # Hide after a short delay
        QTimer.singleShot(2000, self.hide)
        QTimer.singleShot(2000, self._reset_style)
    
    def _reset_style(self) -> None:
        """Reset widget styles after error display."""
        self.status_label.setStyleSheet("")
    
    def getReporter(self) -> ProgressReporter:
        """
        Get the progress reporter for this indicator.
        
        Returns:
            ProgressReporter instance
        """
        return self._reporter
    
    def isCancellable(self) -> bool:
        """
        Check if the progress indicator supports cancellation.
        
        Returns:
            True if cancellable, False otherwise
        """
        return self._cancellable and self.cancel_button is not None
    
    def setCancellable(self, cancellable: bool) -> None:
        """
        Set whether the operation can be cancelled.
        
        Args:
            cancellable: Whether the operation can be cancelled
        """
        self._cancellable = cancellable
        if self.cancel_button:
            self.cancel_button.setVisible(cancellable)


class HelpTooltip(QWidget):
    """
    Widget for displaying help tooltips with rich text and icons.
    
    This widget provides enhanced tooltips with rich text formatting,
    icons, and optional hover effects. It can be attached to any widget
    to provide contextual help information.
    """
    
    def __init__(
        self, 
        parent: QWidget,
        text: str,
        icon: Optional[QPixmap] = None,
        max_width: int = 300,
        show_on_hover: bool = True,
        persistent: bool = False
    ):
        """
        Initialize the help tooltip.
        
        Args:
            parent: Parent widget to attach the tooltip to
            text: Tooltip text (can include HTML formatting)
            icon: Optional icon to display
            max_width: Maximum width for the tooltip
            show_on_hover: Whether to show the tooltip on hover
            persistent: Whether the tooltip should remain visible until clicked
        """
        super().__init__(parent.window())
        
        # Store settings
        self._parent = parent
        self._text = text
        self._icon = icon
        self._max_width = max_width
        self._show_on_hover = show_on_hover
        self._persistent = persistent
        
        # Set up widget properties
        self.setWindowFlags(Qt.WindowType.ToolTip)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setVisible(False)
        
        # Create layout
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        
        # Add icon if provided
        if icon:
            self.icon_label = QLabel()
            self.icon_label.setPixmap(icon.scaled(
                24, 24, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            ))
            self.layout.addWidget(self.icon_label)
        
        # Add text label
        self.text_label = QLabel(text)
        self.text_label.setWordWrap(True)
        self.text_label.setMaximumWidth(max_width)
        self.text_label.setTextFormat(Qt.TextFormat.RichText)
        self.layout.addWidget(self.text_label)
        
        # Style the tooltip
        self.setStyleSheet("""
            background-color: #FFFFD0;
            border: 1px solid #E0E0A0;
            border-radius: 5px;
        """)
        
        # Install event filter on parent if showing on hover
        if show_on_hover:
            parent.installEventFilter(self)
        
        # Create help button if not showing on hover
        if not show_on_hover:
            # Create help button
            self.help_button = QToolButton(parent)
            self.help_button.setText("?")
            self.help_button.setToolTip("Click for help")
            self.help_button.setStyleSheet("""
                QToolButton {
                    background-color: #F0F0F0;
                    border: 1px solid #C0C0C0;
                    border-radius: 10px;
                    padding: 2px;
                    font-weight: bold;
                    color: #505050;
                }
                QToolButton:hover {
                    background-color: #E0E0E0;
                }
            """)
            
            # Position the help button
            parent_layout = parent.layout()
            if isinstance(parent_layout, QHBoxLayout):
                parent_layout.addWidget(self.help_button)
            elif isinstance(parent_layout, QGridLayout):
                # Try to find a good position in the grid
                parent_layout.addWidget(self.help_button, 0, parent_layout.columnCount())
            else:
                # Position in the corner of the parent
                self.help_button.setParent(parent)
                self.help_button.move(parent.width() - 25, 5)
                parent.installEventFilter(self)  # For repositioning on resize
            
            # Connect button click
            self.help_button.clicked.connect(self.toggle)
    
    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        """
        Filter events from the parent widget.
        
        Args:
            obj: Object that received the event
            event: The event
            
        Returns:
            True if the event was handled, False otherwise
        """
        if obj == self._parent:
            if self._show_on_hover:
                # Handle hover events
                if event.type() == QEvent.Type.Enter:
                    self.show_tooltip()
                    return False
                elif event.type() == QEvent.Type.Leave:
                    if not self._persistent:
                        self.hide()
                    return False
            
            # Handle resize events for help button repositioning
            if event.type() == QEvent.Type.Resize and hasattr(self, 'help_button'):
                self.help_button.move(self._parent.width() - 25, 5)
                return False
        
        return super().eventFilter(obj, event)
    
    def show_tooltip(self) -> None:
        """Show the tooltip at the appropriate position."""
        if not self.isVisible():
            # Calculate position
            parent_pos = self._parent.mapToGlobal(self._parent.pos())
            tooltip_x = parent_pos.x() + self._parent.width() // 2 - self.width() // 2
            tooltip_y = parent_pos.y() + self._parent.height() + 5
            
            # Ensure tooltip is visible on screen
            screen = QApplication.primaryScreen()
            if screen:
                screen_geometry = screen.availableGeometry()
                if tooltip_x + self.width() > screen_geometry.right():
                    tooltip_x = screen_geometry.right() - self.width()
                if tooltip_y + self.height() > screen_geometry.bottom():
                    tooltip_y = parent_pos.y() - self.height() - 5
            
            # Position and show the tooltip
            self.move(tooltip_x, tooltip_y)
            self.show()
    
    def toggle(self) -> None:
        """Toggle the visibility of the tooltip."""
        if self.isVisible():
            self.hide()
        else:
            self.show_tooltip()
    
    def setText(self, text: str) -> None:
        """
        Set the tooltip text.
        
        Args:
            text: Tooltip text (can include HTML formatting)
        """
        self._text = text
        self.text_label.setText(text)
    
    def setIcon(self, icon: QPixmap) -> None:
        """
        Set the tooltip icon.
        
        Args:
            icon: Icon to display
        """
        self._icon = icon
        if hasattr(self, 'icon_label'):
            self.icon_label.setPixmap(icon.scaled(
                24, 24, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            ))
        else:
            # Create icon label if it doesn't exist
            self.icon_label = QLabel()
            self.icon_label.setPixmap(icon.scaled(
                24, 24, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            ))
            self.layout.insertWidget(0, self.icon_label)
    
    def setPersistent(self, persistent: bool) -> None:
        """
        Set whether the tooltip should remain visible until clicked.
        
        Args:
            persistent: Whether the tooltip should be persistent
        """
        self._persistent = persistent


class CollapsibleSection(QWidget):
    """
    Widget for creating collapsible sections with toggle functionality.
    
    This widget provides a collapsible section with a header that can be
    clicked to expand or collapse the content. It's useful for organizing
    complex interfaces with optional or advanced settings.
    
    Attributes:
        toggled: Signal emitted when the section is expanded or collapsed
    """
    
    toggled = pyqtSignal(bool)  # Emits True when expanded, False when collapsed
    
    def __init__(
        self, 
        parent: Optional[QWidget] = None,
        title: str = "Section",
        expanded: bool = False
    ):
        """
        Initialize the collapsible section.
        
        Args:
            parent: Parent widget
            title: Section title
            expanded: Whether the section is initially expanded
        """
        super().__init__(parent)
        
        # Store settings
        self._title = title
        self._expanded = expanded
        
        # Create layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        # Create header
        self.header = QFrame()
        self.header.setFrameShape(QFrame.Shape.StyledPanel)
        self.header.setFrameShadow(QFrame.Shadow.Raised)
        self.header.setCursor(Qt.CursorShape.PointingHandCursor)
        self.header.mousePressEvent = self._on_header_clicked
        
        # Create header layout
        self.header_layout = QHBoxLayout(self.header)
        self.header_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create toggle indicator
        self.toggle_label = QLabel("▶" if not expanded else "▼")
        self.header_layout.addWidget(self.toggle_label)
        
        # Create title label
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-weight: bold;")
        self.header_layout.addWidget(self.title_label)
        
        # Add stretch to push title to the left
        self.header_layout.addStretch(1)
        
        # Add header to main layout
        self.layout.addWidget(self.header)
        
        # Create content container
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(10, 10, 10, 10)
        
        # Add content to main layout
        self.layout.addWidget(self.content)
        
        # Set initial state
        self.setExpanded(expanded)
    
    def _on_header_clicked(self, event: QEvent) -> None:
        """
        Handle header click events.
        
        Args:
            event: Mouse press event
        """
        self.toggle()
        event.accept()
    
    def toggle(self) -> None:
        """Toggle the expanded state of the section."""
        self.setExpanded(not self._expanded)
    
    def setExpanded(self, expanded: bool) -> None:
        """
        Set the expanded state of the section.
        
        Args:
            expanded: Whether the section should be expanded
        """
        self._expanded = expanded
        self.content.setVisible(expanded)
        self.toggle_label.setText("▼" if expanded else "▶")
        self.toggled.emit(expanded)
    
    def isExpanded(self) -> bool:
        """
        Check if the section is expanded.
        
        Returns:
            True if expanded, False if collapsed
        """
        return self._expanded
    
    def setTitle(self, title: str) -> None:
        """
        Set the section title.
        
        Args:
            title: Section title
        """
        self._title = title
        self.title_label.setText(title)
    
    def addWidget(self, widget: QWidget) -> None:
        """
        Add a widget to the section content.
        
        Args:
            widget: Widget to add
        """
        self.content_layout.addWidget(widget)
    
    def addLayout(self, layout: QLayout) -> None:
        """
        Add a layout to the section content.
        
        Args:
            layout: Layout to add
        """
        self.content_layout.addLayout(layout)


class FormField(QWidget):
    """
    Widget for creating form fields with labels, inputs, and help tooltips.
    
    This widget provides a standardized form field with a label, input widget,
    and optional help tooltip. It ensures consistent layout and styling for
    form fields throughout the application.
    """
    
    def __init__(
        self, 
        parent: Optional[QWidget] = None,
        label: str = "",
        input_widget: Optional[QWidget] = None,
        help_text: str = "",
        required: bool = False
    ):
        """
        Initialize the form field.
        
        Args:
            parent: Parent widget
            label: Field label
            input_widget: Input widget (e.g., QLineEdit, QComboBox)
            help_text: Help text for tooltip
            required: Whether the field is required
        """
        super().__init__(parent)
        
        # Store settings
        self._label_text = label
        self._help_text = help_text
        self._required = required
        self._input_widget = input_widget
        
        # Create layout
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Create label
        self.label = QLabel(label)
        if required:
            self.label.setText(f"{label} *")
            self.label.setStyleSheet("font-weight: bold;")
        
        # Set fixed width for label to align fields
        font_metrics = QFontMetrics(self.label.font())
        self.label.setMinimumWidth(font_metrics.horizontalAdvance("X" * 15))
        
        self.layout.addWidget(self.label)
        
        # Add input widget if provided
        if input_widget:
            self.layout.addWidget(input_widget, 1)
        else:
            # Add placeholder for input widget
            self.placeholder = QWidget()
            self.placeholder.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            self.layout.addWidget(self.placeholder, 1)
        
        # Add help tooltip if provided
        if help_text:
            self.help_button = QToolButton()
            self.help_button.setText("?")
            self.help_button.setToolTip("Click for help")
            self.help_button.setStyleSheet("""
                QToolButton {
                    background-color: #F0F0F0;
                    border: 1px solid #C0C0C0;
                    border-radius: 10px;
                    padding: 2px;
                    font-weight: bold;
                    color: #505050;
                }
                QToolButton:hover {
                    background-color: #E0E0E0;
                }
            """)
            self.layout.addWidget(self.help_button)
            
            # Connect help button to show tooltip
            self.help_button.clicked.connect(self._show_help_tooltip)
    
    def _show_help_tooltip(self) -> None:
        """Show the help tooltip."""
        QToolTip.showText(
            self.help_button.mapToGlobal(QPoint(0, self.help_button.height())),
            self._help_text,
            self.help_button
        )
    
    def setInputWidget(self, widget: QWidget) -> None:
        """
        Set the input widget.
        
        Args:
            widget: Input widget
        """
        # Remove existing input widget or placeholder
        if self._input_widget:
            self.layout.removeWidget(self._input_widget)
            self._input_widget.setParent(None)
        elif hasattr(self, 'placeholder'):
            self.layout.removeWidget(self.placeholder)
            self.placeholder.setParent(None)
        
        # Add new input widget
        self._input_widget = widget
        self.layout.insertWidget(1, widget, 1)
    
    def inputWidget(self) -> Optional[QWidget]:
        """
        Get the input widget.
        
        Returns:
            Input widget, or None if not set
        """
        return self._input_widget
    
    def setLabel(self, label: str) -> None:
        """
        Set the field label.
        
        Args:
            label: Field label
        """
        self._label_text = label
        if self._required:
            self.label.setText(f"{label} *")
        else:
            self.label.setText(label)
    
    def setRequired(self, required: bool) -> None:
        """
        Set whether the field is required.
        
        Args:
            required: Whether the field is required
        """
        self._required = required
        if required:
            self.label.setText(f"{self._label_text} *")
            self.label.setStyleSheet("font-weight: bold;")
        else:
            self.label.setText(self._label_text)
            self.label.setStyleSheet("")
    
    def setHelpText(self, help_text: str) -> None:
        """
        Set the help text.
        
        Args:
            help_text: Help text for tooltip
        """
        self._help_text = help_text
        
        # Add help button if not already present
        if help_text and not hasattr(self, 'help_button'):
            self.help_button = QToolButton()
            self.help_button.setText("?")
            self.help_button.setToolTip("Click for help")
            self.help_button.setStyleSheet("""
                QToolButton {
                    background-color: #F0F0F0;
                    border: 1px solid #C0C0C0;
                    border-radius: 10px;
                    padding: 2px;
                    font-weight: bold;
                    color: #505050;
                }
                QToolButton:hover {
                    background-color: #E0E0E0;
                }
            """)
            self.layout.addWidget(self.help_button)
            
            # Connect help button to show tooltip
            self.help_button.clicked.connect(self._show_help_tooltip)
        elif not help_text and hasattr(self, 'help_button'):
            # Remove help button if help text is empty
            self.layout.removeWidget(self.help_button)
            self.help_button.setParent(None)
            delattr(self, 'help_button')


class TabularDataView(QWidget):
    """
    Widget for displaying tabular data with sorting and filtering.
    
    This widget provides a table view for displaying tabular data with
    support for sorting, filtering, and pagination. It can display data
    from NumPy arrays, Pandas DataFrames, or lists of dictionaries.
    
    Attributes:
        selectionChanged: Signal emitted when the selection changes
        sortChanged: Signal emitted when the sort order changes
        pageChanged: Signal emitted when the current page changes
    """
    
    selectionChanged = pyqtSignal(list)  # Emits list of selected row indices
    sortChanged = pyqtSignal(int, Qt.SortOrder)  # Emits column index and sort order
    pageChanged = pyqtSignal(int)  # Emits current page number
    
    def __init__(
        self, 
        parent: Optional[QWidget] = None,
        title: str = "Data View",
        data: Optional[Union[np.ndarray, pd.DataFrame, List[Dict[str, Any]]]] = None,
        headers: Optional[List[str]] = None,
        items_per_page: int = 20,
        sortable: bool = True,
        filterable: bool = True,
        selectable: bool = True
    ):
        """
        Initialize the tabular data view.
        
        Args:
            parent: Parent widget
            title: View title
            data: Data to display
            headers: Column headers
            items_per_page: Number of items to display per page
            sortable: Whether the data can be sorted
            filterable: Whether the data can be filtered
            selectable: Whether rows can be selected
        """
        super().__init__(parent)
        
        # Store settings
        self._title = title
        self._headers = headers or []
        self._items_per_page = items_per_page
        self._sortable = sortable
        self._filterable = filterable
        self._selectable = selectable
        self._current_page = 1
        self._total_pages = 1
        self._raw_data = None
        self._filtered_data = None
        self._displayed_data = None
        self._filter_text = ""
        self._sort_column = -1
        self._sort_order = Qt.SortOrder.AscendingOrder
        
        # Create layout
        self.layout = QVBoxLayout(self)
        
        # Create group box
        self.group_box = QGroupBox(title)
        self.group_box_layout = QVBoxLayout(self.group_box)
        
        # Create filter controls if filterable
        if filterable:
            self.filter_layout = QHBoxLayout()
            
            # Filter label
            self.filter_label = QLabel("Filter:")
            self.filter_layout.addWidget(self.filter_label)
            
            # Filter input
            self.filter_input = QLineEdit()
            self.filter_input.setPlaceholderText("Enter filter text...")
            self.filter_input.textChanged.connect(self._on_filter_changed)
            self.filter_layout.addWidget(self.filter_input, 1)
            
            # Clear filter button
            self.clear_filter_button = QPushButton("Clear")
            self.clear_filter_button.clicked.connect(self._on_clear_filter)
            self.filter_layout.addWidget(self.clear_filter_button)
            
            # Add filter controls to layout
            self.group_box_layout.addLayout(self.filter_layout)
        
        # Create table
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows if selectable else QTableWidget.SelectionBehavior.SelectItems
        )
        self.table.setSelectionMode(
            QTableWidget.SelectionMode.ExtendedSelection if selectable else QTableWidget.SelectionMode.NoSelection
        )
        self.table.setSortingEnabled(sortable)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        # Connect table signals
        if sortable:
            self.table.horizontalHeader().sectionClicked.connect(self._on_header_clicked)
        if selectable:
            self.table.itemSelectionChanged.connect(self._on_selection_changed)
        
        # Add table to layout
        self.group_box_layout.addWidget(self.table)
        
        # Create pagination controls
        self.pagination_layout = QHBoxLayout()
        
        # Previous page button
        self.prev_button = QPushButton("< Previous")
        self.prev_button.setEnabled(False)
        self.prev_button.clicked.connect(self._on_prev_page)
        self.pagination_layout.addWidget(self.prev_button)
        
        # Page indicator
        self.page_label = QLabel("Page 1 of 1")
        self.page_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pagination_layout.addWidget(self.page_label, 1)
        
        # Next page button
        self.next_button = QPushButton("Next >")
        self.next_button.setEnabled(False)
        self.next_button.clicked.connect(self._on_next_page)
        self.pagination_layout.addWidget(self.next_button)
        
        # Add pagination controls to layout
        self.group_box_layout.addLayout(self.pagination_layout)
        
        # Add group box to main layout
        self.layout.addWidget(self.group_box)
        
        # Set data if provided
        if data is not None:
            self.setData(data, headers)
    
    def _on_filter_changed(self, text: str) -> None:
        """
        Handle filter text changes.
        
        Args:
            text: Filter text
        """
        self._filter_text = text
        self._apply_filter()
        self._update_table()
    
    def _on_clear_filter(self) -> None:
        """Handle clear filter button click."""
        if hasattr(self, 'filter_input'):
            self.filter_input.clear()
    
    def _on_header_clicked(self, column: int) -> None:
        """
        Handle header click for sorting.
        
        Args:
            column: Column index
        """
        if self._sort_column == column:
            # Toggle sort order
            self._sort_order = (
                Qt.SortOrder.DescendingOrder 
                if self._sort_order == Qt.SortOrder.AscendingOrder 
                else Qt.SortOrder.AscendingOrder
            )
        else:
            # Sort by new column
            self._sort_column = column
            self._sort_order = Qt.SortOrder.AscendingOrder
        
        # Apply sorting
        self._apply_sort()
        self._update_table()
        
        # Emit sort changed signal
        self.sortChanged.emit(self._sort_column, self._sort_order)
    
    def _on_selection_changed(self) -> None:
        """Handle selection changes."""
        selected_rows = sorted(set(index.row() for index in self.table.selectedIndexes()))
        
        # Convert to indices in the original data
        if self._displayed_data is not None and selected_rows:
            start_idx = (self._current_page - 1) * self._items_per_page
            original_indices = [start_idx + row for row in selected_rows]
            self.selectionChanged.emit(original_indices)
        else:
            self.selectionChanged.emit([])
    
    def _on_prev_page(self) -> None:
        """Handle previous page button click."""
        if self._current_page > 1:
            self._current_page -= 1
            self._update_table()
            self.pageChanged.emit(self._current_page)
    
    def _on_next_page(self) -> None:
        """Handle next page button click."""
        if self._current_page < self._total_pages:
            self._current_page += 1
            self._update_table()
            self.pageChanged.emit(self._current_page)
    
    def _apply_filter(self) -> None:
        """Apply the current filter to the data."""
        if not self._raw_data:
            self._filtered_data = None
            return
        
        if not self._filter_text:
            # No filter, use all data
            self._filtered_data = self._raw_data
            return
        
        # Convert data to list of dictionaries for consistent filtering
        if isinstance(self._raw_data, np.ndarray):
            # Convert NumPy array to list of dictionaries
            data_list = []
            for row in self._raw_data:
                row_dict = {}
                for i, value in enumerate(row):
                    header = self._headers[i] if i < len(self._headers) else f"Column {i+1}"
                    row_dict[header] = value
                data_list.append(row_dict)
        elif isinstance(self._raw_data, pd.DataFrame):
            # Convert DataFrame to list of dictionaries
            data_list = self._raw_data.to_dict('records')
        else:
            # Already a list of dictionaries
            data_list = self._raw_data
        
        # Apply filter
        filter_text = self._filter_text.lower()
        filtered_list = []
        
        for row_dict in data_list:
            # Check if any value contains the filter text
            if any(
                str(value).lower().find(filter_text) >= 0
                for value in row_dict.values()
            ):
                filtered_list.append(row_dict)
        
        # Store filtered data
        self._filtered_data = filtered_list
    
    def _apply_sort(self) -> None:
        """Apply the current sort to the filtered data."""
        if not self._filtered_data or self._sort_column < 0:
            return
        
        # Convert data to list of dictionaries for consistent sorting
        if isinstance(self._filtered_data, np.ndarray):
            # Sort NumPy array by column
            sort_idx = self._sort_column
            self._filtered_data = np.array(
                sorted(
                    self._filtered_data, 
                    key=lambda x: x[sort_idx] if sort_idx < len(x) else 0,
                    reverse=(self._sort_order == Qt.SortOrder.DescendingOrder)
                )
            )
        elif isinstance(self._filtered_data, pd.DataFrame):
            # Sort DataFrame by column
            if self._sort_column < len(self._filtered_data.columns):
                column_name = self._filtered_data.columns[self._sort_column]
                self._filtered_data = self._filtered_data.sort_values(
                    by=column_name,
                    ascending=(self._sort_order == Qt.SortOrder.AscendingOrder)
                )
        else:
            # Sort list of dictionaries
            if self._sort_column < len(self._headers):
                sort_key = self._headers[self._sort_column]
                self._filtered_data = sorted(
                    self._filtered_data,
                    key=lambda x: x.get(sort_key, ""),
                    reverse=(self._sort_order == Qt.SortOrder.DescendingOrder)
                )
    
    def _update_table(self) -> None:
        """Update the table with current page data."""
        if not self._filtered_data:
            # Clear the table
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            self._current_page = 1
            self._total_pages = 1
            self.page_label.setText("Page 1 of 1")
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
            return
        
        # Calculate pagination
        if isinstance(self._filtered_data, list):
            total_items = len(self._filtered_data)
        elif isinstance(self._filtered_data, np.ndarray):
            total_items = self._filtered_data.shape[0]
        elif isinstance(self._filtered_data, pd.DataFrame):
            total_items = len(self._filtered_data)
        else:
            total_items = 0
        
        self._total_pages = max(1, (total_items + self._items_per_page - 1) // self._items_per_page)
        self._current_page = min(max(1, self._current_page), self._total_pages)
        
        # Calculate the range of items to display
        start_idx = (self._current_page - 1) * self._items_per_page
        end_idx = min(start_idx + self._items_per_page, total_items)
        
        # Get the data for the current page
        if isinstance(self._filtered_data, list):
            self._displayed_data = self._filtered_data[start_idx:end_idx]
        elif isinstance(self._filtered_data, np.ndarray):
            self._displayed_data = self._filtered_data[start_idx:end_idx]
        elif isinstance(self._filtered_data, pd.DataFrame):
            self._displayed_data = self._filtered_data.iloc[start_idx:end_idx]
        else:
            self._displayed_data = None
        
        # Update the table
        self._populate_table()
        
        # Update pagination controls
        self.page_label.setText(f"Page {self._current_page} of {self._total_pages}")
        self.prev_button.setEnabled(self._current_page > 1)
        self.next_button.setEnabled(self._current_page < self._total_pages)
    
    def _populate_table(self) -> None:
        """Populate the table with the current displayed data."""
        if self._displayed_data is None:
            return
        
        # Disconnect signals temporarily to prevent triggering events
        if self._sortable:
            self.table.horizontalHeader().sectionClicked.disconnect(self._on_header_clicked)
        if self._selectable:
            self.table.itemSelectionChanged.disconnect(self._on_selection_changed)
        
        # Clear the table
        self.table.setRowCount(0)
        
        # Set up columns
        if isinstance(self._displayed_data, list) and self._displayed_data:
            # Get headers from first dictionary
            if not self._headers:
                self._headers = list(self._displayed_data[0].keys())
            
            # Set column count and headers
            self.table.setColumnCount(len(self._headers))
            self.table.setHorizontalHeaderLabels(self._headers)
            
            # Add rows
            for row_idx, row_dict in enumerate(self._displayed_data):
                self.table.insertRow(row_idx)
                
                # Add cells
                for col_idx, header in enumerate(self._headers):
                    value = row_dict.get(header, "")
                    item = QTableWidgetItem(str(value))
                    
                    # Align numeric values to the right
                    if isinstance(value, (int, float, np.number)):
                        item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    
                    self.table.setItem(row_idx, col_idx, item)
        elif isinstance(self._displayed_data, np.ndarray):
            # Set column count and headers
            if self._displayed_data.ndim == 1:
                # 1D array
                self.table.setColumnCount(1)
                self.table.setHorizontalHeaderLabels(self._headers[:1] or ["Value"])
                
                # Add rows
                for row_idx, value in enumerate(self._displayed_data):
                    self.table.insertRow(row_idx)
                    item = QTableWidgetItem(str(value))
                    
                    # Align numeric values to the right
                    if isinstance(value, (int, float, np.number)):
                        item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    
                    self.table.setItem(row_idx, 0, item)
            else:
                # 2D array
                self.table.setColumnCount(self._displayed_data.shape[1])
                self.table.setHorizontalHeaderLabels(
                    self._headers[:self._displayed_data.shape[1]] or 
                    [f"Column {i+1}" for i in range(self._displayed_data.shape[1])]
                )
                
                # Add rows
                for row_idx, row in enumerate(self._displayed_data):
                    self.table.insertRow(row_idx)
                    
                    # Add cells
                    for col_idx, value in enumerate(row):
                        item = QTableWidgetItem(str(value))
                        
                        # Align numeric values to the right
                        if isinstance(value, (int, float, np.number)):
                            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                        
                        self.table.setItem(row_idx, col_idx, item)
        elif isinstance(self._displayed_data, pd.DataFrame):
            # Set column count and headers
            self.table.setColumnCount(len(self._displayed_data.columns))
            self.table.setHorizontalHeaderLabels(self._displayed_data.columns.astype(str).tolist())
            
            # Add rows
            for row_idx, (_, row) in enumerate(self._displayed_data.iterrows()):
                self.table.insertRow(row_idx)
                
                # Add cells
                for col_idx, (column, value) in enumerate(row.items()):
                    if pd.isna(value):
                        item = QTableWidgetItem("")
                    else:
                        item = QTableWidgetItem(str(value))
                        
                        # Align numeric values to the right
                        if isinstance(value, (int, float, np.number)):
                            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    
                    self.table.setItem(row_idx, col_idx, item)
        
        # Resize columns to content
        self.table.resizeColumnsToContents()
        
        # Reconnect signals
        if self._sortable:
            self.table.horizontalHeader().sectionClicked.connect(self._on_header_clicked)
        if self._selectable:
            self.table.itemSelectionChanged.connect(self._on_selection_changed)
    
    def setData(self, data: Union[np.ndarray, pd.DataFrame, List[Dict[str, Any]]], 
               headers: Optional[List[str]] = None) -> None:
        """
        Set the data to display.
        
        Args:
            data: Data to display
            headers: Column headers
        """
        self._raw_data = data
        if headers:
            self._headers = headers
        elif isinstance(data, pd.DataFrame):
            self._headers = data.columns.astype(str).tolist()
        
        # Reset state
        self._current_page = 1
        self._filter_text = ""
        self._sort_column = -1
        self._sort_order = Qt.SortOrder.AscendingOrder
        
        # Clear filter input if present
        if hasattr(self, 'filter_input'):
            self.filter_input.clear()
        
        # Apply filter (which will use all data since filter_text is empty)
        self._apply_filter()
        
        # Update the table
        self._update_table()
    
    def setTitle(self, title: str) -> None:
        """
        Set the view title.
        
        Args:
            title: View title
        """
        self._title = title
        self.group_box.setTitle(title)
    
    def setItemsPerPage(self, items_per_page: int) -> None:
        """
        Set the number of items to display per page.
        
        Args:
            items_per_page: Number of items per page
        """
        self._items_per_page = max(1, items_per_page)
        self._update_table()
    
    def setHeaders(self, headers: List[str]) -> None:
        """
        Set the column headers.
        
        Args:
            headers: Column headers
        """
        self._headers = headers
        self._update_table()
    
    def setSortable(self, sortable: bool) -> None:
        """
        Set whether the data can be sorted.
        
        Args:
            sortable: Whether the data can be sorted
        """
        self._sortable = sortable
        self.table.setSortingEnabled(sortable)
    
    def setFilterable(self, filterable: bool) -> None:
        """
        Set whether the data can be filtered.
        
        Args:
            filterable: Whether the data can be filtered
        """
        self._filterable = filterable
        
        # Show/hide filter controls
        if hasattr(self, 'filter_label'):
            self.filter_label.setVisible(filterable)
            self.filter_input.setVisible(filterable)
            self.clear_filter_button.setVisible(filterable)
    
    def setSelectable(self, selectable: bool) -> None:
        """
        Set whether rows can be selected.
        
        Args:
            selectable: Whether rows can be selected
        """
        self._selectable = selectable
        self.table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows if selectable else QTableWidget.SelectionBehavior.SelectItems
        )
        self.table.setSelectionMode(
            QTableWidget.SelectionMode.ExtendedSelection if selectable else QTableWidget.SelectionMode.NoSelection
        )
    
    def currentPage(self) -> int:
        """
        Get the current page number.
        
        Returns:
            Current page number (1-based)
        """
        return self._current_page
    
    def totalPages(self) -> int:
        """
        Get the total number of pages.
        
        Returns:
            Total number of pages
        """
        return self._total_pages
    
    def goToPage(self, page: int) -> None:
        """
        Go to a specific page.
        
        Args:
            page: Page number (1-based)
        """
        page = max(1, min(page, self._total_pages))
        if page != self._current_page:
            self._current_page = page
            self._update_table()
            self.pageChanged.emit(self._current_page)
    
    def selectedRows(self) -> List[int]:
        """
        Get the indices of selected rows in the original data.
        
        Returns:
            List of selected row indices
        """
        selected_rows = sorted(set(index.row() for index in self.table.selectedIndexes()))
        
        # Convert to indices in the original data
        if self._displayed_data is not None and selected_rows:
            start_idx = (self._current_page - 1) * self._items_per_page
            return [start_idx + row for row in selected_rows]
        
        return []
    
    def clearSelection(self) -> None:
        """Clear the current selection."""
        self.table.clearSelection()
    
    def selectRow(self, row: int) -> None:
        """
        Select a specific row.
        
        Args:
            row: Row index in the original data
        """
        if not self._selectable or self._displayed_data is None:
            return
        
        # Calculate page and row index
        page = (row // self._items_per_page) + 1
        row_in_page = row % self._items_per_page
        
        # Go to the page if needed
        if page != self._current_page:
            self.goToPage(page)
        
        # Select the row if it's on the current page
        if row_in_page < self.table.rowCount():
            self.table.selectRow(row_in_page)