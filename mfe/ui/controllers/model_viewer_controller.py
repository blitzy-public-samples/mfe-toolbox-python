# mfe/ui/controllers/model_viewer_controller.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARMAX model results viewer controller using PyQt6.

This module implements the controller component for displaying ARMAX model results,
managing the interaction between the model viewer model and view. It handles
pagination, display formatting, and user interactions with the result viewer
while keeping business logic separate from presentation.

The ModelViewerController class follows the Model-View-Controller pattern,
providing the business logic layer for model results display while delegating
presentation to the view and data management to the model.
"""

import logging
from typing import Optional, List, Dict, Any, Tuple, Union, cast
import numpy as np
import pandas as pd

# Import model and view components
from mfe.ui.models.model_viewer_model import ModelViewerModel, ModelParameter, ModelStatistic
from mfe.ui.views.model_viewer_view import ModelViewerView
from mfe.ui.models.armax_model import ARMAXModel, ARMAXModelResults

# Import utility functions
from mfe.ui.utils import (
    format_latex_equation, create_equation_figure, create_parameter_table,
    create_statistics_table, center_widget_on_parent, show_error_message,
    format_model_equation
)

# Import core exceptions
from mfe.core.exceptions import MFEError, UIError

# Set up module-level logger
logger = logging.getLogger("mfe.ui.controllers.model_viewer_controller")


class ModelViewerController:
    """
    Controller for the ARMAX model results viewer.
    
    This class manages the interaction between the model viewer model and view,
    handling pagination, display formatting, and user interactions with the
    result viewer while keeping business logic separate from presentation.
    
    Attributes:
        model: The model viewer data model
        view: The model viewer view
        current_page: Current page of parameter display
        items_per_page: Number of parameters to display per page
    """
    
    def __init__(self, parent=None):
        """
        Initialize the model viewer controller.
        
        Args:
            parent: Parent widget for the view
        """
        self.model = ModelViewerModel()
        self.view = ModelViewerView(parent)
        self.current_page = 1
        self.items_per_page = 10
        
        # Connect view signals to controller methods
        self._connect_signals()
        
        logger.debug("ModelViewerController initialized")
    
    def _connect_signals(self):
        """Connect view signals to controller methods."""
        self.view.close_clicked.connect(self._on_close_clicked)
        self.view.next_page_clicked.connect(self._on_next_page_clicked)
        self.view.prev_page_clicked.connect(self._on_prev_page_clicked)
        self.view.copy_to_clipboard_clicked.connect(self._on_copy_to_clipboard_clicked)
        
        logger.debug("View signals connected to controller methods")
    
    def show(self):
        """Show the model viewer dialog."""
        self.view.show()
        logger.debug("Model viewer dialog shown")
    
    def show_model_results(self, armax_model: ARMAXModel):
        """
        Show results for an ARMAX model.
        
        Args:
            armax_model: The ARMAX model to display results for
            
        Raises:
            UIError: If the model is not fitted or has no results
        """
        try:
            # Check if the model is fitted
            if not armax_model.is_fitted or armax_model.results is None:
                raise UIError(
                    "Cannot display results for an unfitted model",
                    component="ModelViewer",
                    operation="show_model_results",
                    issue="Model not fitted"
                )
            
            # Initialize the model with the ARMAX model
            self.model.initialize_from_model(armax_model)
            
            # Reset pagination
            self.current_page = 1
            
            # Update the view with model data
            self._update_view()
            
            # Show the view
            self.show()
            
            logger.info("Model results displayed successfully")
        except Exception as e:
            logger.error(f"Error showing model results: {e}")
            if isinstance(e, UIError):
                raise
            else:
                raise UIError(
                    f"Failed to display model results: {str(e)}",
                    component="ModelViewer",
                    operation="show_model_results",
                    issue=str(e),
                    details=str(e)
                ) from e
    
    def _update_view(self):
        """Update the view with current model data."""
        try:
            # Display model equation
            equation = self.model.get_model_equation()
            if equation:
                self.view.display_model_equation(equation)
            
            # Display parameters with pagination
            parameters = self._format_parameters_for_display()
            self.view.display_parameters(
                parameters, 
                self.current_page, 
                self._calculate_total_pages()
            )
            
            # Display statistics
            statistics = self._format_statistics_for_display()
            self.view.display_statistics(statistics)
            
            # Display diagnostic tests if available
            if self.model.armax_model and self.model.armax_model.results:
                diagnostic_tests = self.model.armax_model.results.diagnostic_tests
                if diagnostic_tests:
                    self.view.display_diagnostic_tests(diagnostic_tests)
            
            logger.debug("View updated with model data")
        except Exception as e:
            logger.error(f"Error updating view: {e}")
            self.view.show_error(
                "Display Error", 
                "Failed to update display with model results", 
                str(e)
            )
    
    def _format_parameters_for_display(self) -> List[Dict[str, Any]]:
        """
        Format parameters for display in the view.
        
        Returns:
            List of parameter dictionaries for display
        """
        parameters = []
        
        # Get all parameters from the model
        model_parameters = self.model.get_parameters()
        
        # Convert to dictionaries for the view
        for param in model_parameters:
            param_dict = {
                'name': param.name,
                'estimate': param.value,
                'std_error': param.std_error if param.std_error is not None else np.nan,
                't_stat': param.t_stat if param.t_stat is not None else np.nan,
                'p_value': param.p_value if param.p_value is not None else np.nan
            }
            parameters.append(param_dict)
        
        return parameters
    
    def _format_statistics_for_display(self) -> Dict[str, Any]:
        """
        Format statistics for display in the view.
        
        Returns:
            Dictionary of statistics for display
        """
        statistics = {}
        
        # Get all statistics from the model
        model_statistics = self.model.get_statistics()
        
        # Convert to dictionary for the view
        for stat in model_statistics:
            statistics[stat.name] = {
                'value': stat.value,
                'description': stat.description if stat.description else ""
            }
        
        return statistics
    
    def _calculate_total_pages(self) -> int:
        """
        Calculate the total number of pages for parameter display.
        
        Returns:
            Total number of pages
        """
        total_parameters = len(self.model.get_parameters())
        return max(1, (total_parameters + self.items_per_page - 1) // self.items_per_page)
    
    def _on_close_clicked(self):
        """Handle close button click."""
        self.view.close()
        logger.debug("Close button clicked")
    
    def _on_next_page_clicked(self):
        """Handle next page button click."""
        total_pages = self._calculate_total_pages()
        if self.current_page < total_pages:
            self.current_page += 1
            self._update_pagination()
            logger.debug(f"Next page clicked, now showing page {self.current_page}/{total_pages}")
    
    def _on_prev_page_clicked(self):
        """Handle previous page button click."""
        if self.current_page > 1:
            self.current_page -= 1
            self._update_pagination()
            logger.debug(f"Previous page clicked, now showing page {self.current_page}/{total_pages}")
    
    def _update_pagination(self):
        """Update the parameter display with current pagination."""
        parameters = self._format_parameters_for_display()
        self.view.display_parameters(
            parameters, 
            self.current_page, 
            self._calculate_total_pages()
        )
        
        # Update pagination controls
        self.view.update_pagination(
            self.current_page, 
            self._calculate_total_pages()
        )
    
    def _on_copy_to_clipboard_clicked(self):
        """Handle copy to clipboard button click."""
        self.view.copy_results_to_clipboard()
        logger.debug("Results copied to clipboard")
    
    def format_significance_indicator(self, p_value: float) -> str:
        """
        Format a significance indicator based on p-value.
        
        Args:
            p_value: p-value to format
            
        Returns:
            Significance indicator string
        """
        if p_value < 0.01:
            return "[***]"
        elif p_value < 0.05:
            return "[**]"
        elif p_value < 0.1:
            return "[*]"
        else:
            return ""
    
    def format_parameter_value(self, value: float, precision: int = 4) -> str:
        """
        Format a parameter value with appropriate precision.
        
        Args:
            value: Parameter value to format
            precision: Number of decimal places
            
        Returns:
            Formatted parameter value string
        """
        return f"{value:.{precision}f}"
    
    def format_p_value(self, p_value: float, precision: int = 4) -> str:
        """
        Format a p-value with appropriate precision and significance indicator.
        
        Args:
            p_value: p-value to format
            precision: Number of decimal places
            
        Returns:
            Formatted p-value string with significance indicator
        """
        p_value_str = f"{p_value:.{precision}f}"
        significance = self.format_significance_indicator(p_value)
        if significance:
            p_value_str += f" {significance}"
        return p_value_str
    
    def close(self):
        """Close the model viewer dialog."""
        self.view.close()
        logger.debug("Model viewer dialog closed")

# mfe/ui/controllers/model_viewer_controller.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARMAX model results viewer controller using PyQt6.

This module implements the controller component for displaying ARMAX model results,
managing the interaction between the model viewer model and view. It handles
pagination, display formatting, and user interactions with the result viewer
while keeping business logic separate from presentation.

The ModelViewerController class follows the Model-View-Controller pattern,
providing the business logic layer for model results display while delegating
presentation to the view and data management to the model.
"""

import logging
from typing import Optional, List, Dict, Any, Tuple, Union, cast
import numpy as np
import pandas as pd

# Import model and view components
from mfe.ui.models.model_viewer_model import ModelViewerModel, ModelParameter, ModelStatistic
from mfe.ui.views.model_viewer_view import ModelViewerView
from mfe.ui.models.armax_model import ARMAXModel, ARMAXModelResults

# Import utility functions
from mfe.ui.utils import (
    format_latex_equation, create_equation_figure, create_parameter_table,
    create_statistics_table, center_widget_on_parent, show_error_message,
    format_model_equation
)

# Import core exceptions
from mfe.core.exceptions import MFEError, UIError

# Set up module-level logger
logger = logging.getLogger("mfe.ui.controllers.model_viewer_controller")


class ModelViewerController:
    """
    Controller for the ARMAX model results viewer.
    
    This class manages the interaction between the model viewer model and view,
    handling pagination, display formatting, and user interactions with the
    result viewer while keeping business logic separate from presentation.
    
    Attributes:
        model: The model viewer data model
        view: The model viewer view
        current_page: Current page of parameter display
        items_per_page: Number of parameters to display per page
    """
    
    def __init__(self, parent=None):
        """
        Initialize the model viewer controller.
        
        Args:
            parent: Parent widget for the view
        """
        self.model = ModelViewerModel()
        self.view = ModelViewerView(parent)
        self.current_page = 1
        self.items_per_page = 10
        
        # Connect view signals to controller methods
        self._connect_signals()
        
        logger.debug("ModelViewerController initialized")
    
    def _connect_signals(self):
        """Connect view signals to controller methods."""
        self.view.close_clicked.connect(self._on_close_clicked)
        self.view.next_page_clicked.connect(self._on_next_page_clicked)
        self.view.prev_page_clicked.connect(self._on_prev_page_clicked)
        self.view.copy_to_clipboard_clicked.connect(self._on_copy_to_clipboard_clicked)
        
        logger.debug("View signals connected to controller methods")
    
    def show(self):
        """Show the model viewer dialog."""
        self.view.show()
        logger.debug("Model viewer dialog shown")
    
    def show_model_results(self, armax_model: ARMAXModel):
        """
        Show results for an ARMAX model.
        
        Args:
            armax_model: The ARMAX model to display results for
            
        Raises:
            UIError: If the model is not fitted or has no results
        """
        try:
            # Check if the model is fitted
            if not armax_model.is_fitted or armax_model.results is None:
                raise UIError(
                    "Cannot display results for an unfitted model",
                    component="ModelViewer",
                    operation="show_model_results",
                    issue="Model not fitted"
                )
            
            # Initialize the model with the ARMAX model
            self.model.initialize_from_model(armax_model)
            
            # Reset pagination
            self.current_page = 1
            
            # Update the view with model data
            self._update_view()
            
            # Show the view
            self.show()
            
            logger.info("Model results displayed successfully")
        except Exception as e:
            logger.error(f"Error showing model results: {e}")
            if isinstance(e, UIError):
                raise
            else:
                raise UIError(
                    f"Failed to display model results: {str(e)}",
                    component="ModelViewer",
                    operation="show_model_results",
                    issue=str(e),
                    details=str(e)
                ) from e
    
    def _update_view(self):
        """Update the view with current model data."""
        try:
            # Display model equation
            equation = self.model.get_model_equation()
            if equation:
                self.view.display_model_equation(equation)
            
            # Display parameters with pagination
            parameters = self._format_parameters_for_display()
            self.view.display_parameters(
                parameters, 
                self.current_page, 
                self._calculate_total_pages()
            )
            
            # Display statistics
            statistics = self._format_statistics_for_display()
            self.view.display_statistics(statistics)
            
            # Display diagnostic tests if available
            if self.model.armax_model and self.model.armax_model.results:
                diagnostic_tests = self.model.armax_model.results.diagnostic_tests
                if diagnostic_tests:
                    self.view.display_diagnostic_tests(diagnostic_tests)
            
            logger.debug("View updated with model data")
        except Exception as e:
            logger.error(f"Error updating view: {e}")
            self.view.show_error(
                "Display Error", 
                "Failed to update display with model results", 
                str(e)
            )
    
    def _format_parameters_for_display(self) -> List[Dict[str, Any]]:
        """
        Format parameters for display in the view.
        
        Returns:
            List of parameter dictionaries for display
        """
        parameters = []
        
        # Get all parameters from the model
        model_parameters = self.model.get_parameters()
        
        # Convert to dictionaries for the view
        for param in model_parameters:
            param_dict = {
                'name': param.name,
                'estimate': param.value,
                'std_error': param.std_error if param.std_error is not None else np.nan,
                't_stat': param.t_stat if param.t_stat is not None else np.nan,
                'p_value': param.p_value if param.p_value is not None else np.nan
            }
            parameters.append(param_dict)
        
        return parameters
    
    def _format_statistics_for_display(self) -> Dict[str, Any]:
        """
        Format statistics for display in the view.
        
        Returns:
            Dictionary of statistics for display
        """
        statistics = {}
        
        # Get all statistics from the model
        model_statistics = self.model.get_statistics()
        
        # Convert to dictionary for the view
        for stat in model_statistics:
            statistics[stat.name] = {
                'value': stat.value,
                'description': stat.description if stat.description else ""
            }
        
        return statistics
    
    def _calculate_total_pages(self) -> int:
        """
        Calculate the total number of pages for parameter display.
        
        Returns:
            Total number of pages
        """
        total_parameters = len(self.model.get_parameters())
        return max(1, (total_parameters + self.items_per_page - 1) // self.items_per_page)
    
    def _on_close_clicked(self):
        """Handle close button click."""
        self.view.close()
        logger.debug("Close button clicked")
    
    def _on_next_page_clicked(self):
        """Handle next page button click."""
        total_pages = self._calculate_total_pages()
        if self.current_page < total_pages:
            self.current_page += 1
            self._update_pagination()
            logger.debug(f"Next page clicked, now showing page {self.current_page}/{total_pages}")
    
    def _on_prev_page_clicked(self):
        """Handle previous page button click."""
        if self.current_page > 1:
            self.current_page -= 1
            self._update_pagination()
            logger.debug(f"Previous page clicked, now showing page {self.current_page}/{total_pages}")
    
    def _update_pagination(self):
        """Update the parameter display with current pagination."""
        parameters = self._format_parameters_for_display()
        self.view.display_parameters(
            parameters, 
            self.current_page, 
            self._calculate_total_pages()
        )
        
        # Update pagination controls
        self.view.update_pagination(
            self.current_page, 
            self._calculate_total_pages()
        )
    
    def _on_copy_to_clipboard_clicked(self):
        """Handle copy to clipboard button click."""
        self.view.copy_results_to_clipboard()
        logger.debug("Results copied to clipboard")
    
    def format_significance_indicator(self, p_value: float) -> str:
        """
        Format a significance indicator based on p-value.
        
        Args:
            p_value: p-value to format
            
        Returns:
            Significance indicator string
        """
        if p_value < 0.01:
            return "[***]"
        elif p_value < 0.05:
            return "[**]"
        elif p_value < 0.1:
            return "[*]"
        else:
            return ""
    
    def format_parameter_value(self, value: float, precision: int = 4) -> str:
        """
        Format a parameter value with appropriate precision.
        
        Args:
            value: Parameter value to format
            precision: Number of decimal places
            
        Returns:
            Formatted parameter value string
        """
        return f"{value:.{precision}f}"
    
    def format_p_value(self, p_value: float, precision: int = 4) -> str:
        """
        Format a p-value with appropriate precision and significance indicator.
        
        Args:
            p_value: p-value to format
            precision: Number of decimal places
            
        Returns:
            Formatted p-value string with significance indicator
        """
        p_value_str = f"{p_value:.{precision}f}"
        significance = self.format_significance_indicator(p_value)
        if significance:
            p_value_str += f" {significance}"
        return p_value_str
    
    def close(self):
        """Close the model viewer dialog."""
        self.view.close()
        logger.debug("Model viewer dialog closed")
