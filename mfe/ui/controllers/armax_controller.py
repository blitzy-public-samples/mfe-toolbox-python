#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARMAX Controller

This module implements the controller for the ARMAX modeling interface,
following the Model-View-Controller pattern. It handles the interaction
between the ARMAX model and view, manages user inputs, coordinates model
estimation, and updates the UI accordingly.

The controller leverages PyQt6's signal-slot mechanism for event handling
and Python's async/await pattern for long-running operations, ensuring
the UI remains responsive during computationally intensive tasks.
"""

import logging
import asyncio
import os
import sys
import traceback
from typing import Optional, List, Dict, Any, Tuple, Union, cast
from pathlib import Path
import numpy as np
import pandas as pd

# PyQt6 imports
from PyQt6.QtCore import QObject, pyqtSlot, QTimer

# Import models and views
from mfe.ui.models.armax_model import ARMAXModel, ARMAXModelResults
from mfe.ui.views.armax_view import ARMAXView
from mfe.ui.models.about_dialog_model import AboutDialogModel
from mfe.ui.views.about_dialog_view import AboutDialogView
from mfe.ui.views.model_viewer_view import ModelViewerView

# Import core exceptions
from mfe.core.exceptions import (
    MFEError, ModelSpecificationError, EstimationError, DataError,
    ForecastError, NotFittedError
)

# Set up module-level logger
logger = logging.getLogger("mfe.ui.controllers.armax_controller")


class ARMAXController(QObject):
    """
    Controller for the ARMAX modeling interface.
    
    This class implements the Controller part of the MVC pattern for the ARMAX
    modeling interface. It handles the interaction between the model and view,
    manages user inputs, coordinates model estimation, and updates the UI accordingly.
    
    The controller leverages PyQt6's signal-slot mechanism for event handling
    and Python's async/await pattern for long-running operations, ensuring
    the UI remains responsive during computationally intensive tasks.
    
    Attributes:
        model: The ARMAX model instance
        view: The ARMAX view instance
        about_dialog: The About dialog view instance
        about_model: The About dialog model instance
        model_viewer: The Model Viewer view instance
        _estimation_task: The current estimation task
        _forecast_task: The current forecast task
    """
    
    def __init__(self, model: Optional[ARMAXModel] = None, view: Optional[ARMAXView] = None):
        """
        Initialize the ARMAX controller.
        
        Args:
            model: The ARMAX model instance (created if None)
            view: The ARMAX view instance (created if None)
        """
        super().__init__()
        
        # Initialize model and view
        self.model = model if model is not None else ARMAXModel()
        self.view = view if view is not None else ARMAXView()
        
        # Initialize dialogs
        self.about_model = AboutDialogModel()
        self.about_dialog: Optional[AboutDialogView] = None
        self.model_viewer: Optional[ModelViewerView] = None
        
        # Initialize task tracking
        self._estimation_task: Optional[asyncio.Task] = None
        self._forecast_task: Optional[asyncio.Task] = None
        
        # Connect signals to slots
        self._connect_signals()
        
        # Initialize view with model state
        self._initialize_view()
        
        logger.debug("ARMAXController initialized")
    
    def _connect_signals(self) -> None:
        """Connect view signals to controller slots."""
        # Connect model specification signals
        self.view.ar_order_changed.connect(self._on_ar_order_changed)
        self.view.ma_order_changed.connect(self._on_ma_order_changed)
        self.view.constant_toggled.connect(self._on_constant_toggled)
        self.view.exog_variables_changed.connect(self._on_exog_variables_changed)
        
        # Connect action signals
        self.view.estimate_clicked.connect(self._on_estimate_clicked)
        self.view.reset_clicked.connect(self._on_reset_clicked)
        self.view.about_clicked.connect(self._on_about_clicked)
        self.view.load_data_clicked.connect(self._on_load_data_clicked)
        self.view.save_results_clicked.connect(self._on_save_results_clicked)
        self.view.forecast_clicked.connect(self._on_forecast_clicked)
        self.view.cancel_clicked.connect(self._on_cancel_clicked)
        
        logger.debug("View signals connected to controller slots")
    
    def _initialize_view(self) -> None:
        """Initialize the view with the current model state."""
        # Set model parameters in view
        self.view.set_ar_order(self.model.ar_order)
        self.view.set_ma_order(self.model.ma_order)
        self.view.set_include_constant(self.model.include_constant)
        self.view.set_estimation_method(self.model.parameters.estimation_method)
        
        # Set exogenous variables if available
        if self.model.exog_names:
            self.view.set_exog_variables(self.model.exog_names, self.model.exog_variables)
        
        # Update UI based on model state
        self.view.enable_results_actions(self.model.is_fitted)
        
        # Plot data if available
        if self.model.data is not None:
            self.view.plot_time_series(
                self.model.data,
                title=f"Time Series: {self.model.data_name}" if self.model.data_name else "Time Series Data"
            )
        
        # Plot residuals if available
        if self.model.is_fitted and self.model.results and self.model.results.residuals is not None:
            self.view.plot_residuals(self.model.results.residuals)
        
        # Display model results if available
        if self.model.is_fitted and self.model.results:
            self._display_model_results()
        
        logger.debug("View initialized with model state")
    
    def show(self) -> None:
        """Show the main view."""
        self.view.show()
        logger.debug("ARMAX view shown")
    
    # Signal handlers (slots)
    @pyqtSlot(int)
    def _on_ar_order_changed(self, value: int) -> None:
        """
        Handle AR order change.
        
        Args:
            value: New AR order
        """
        try:
            self.model.ar_order = value
            logger.debug(f"AR order set to {value}")
        except ModelSpecificationError as e:
            self.view.show_error_message("Invalid AR Order", str(e))
            # Reset view to model's current value
            self.view.set_ar_order(self.model.ar_order)
    
    @pyqtSlot(int)
    def _on_ma_order_changed(self, value: int) -> None:
        """
        Handle MA order change.
        
        Args:
            value: New MA order
        """
        try:
            self.model.ma_order = value
            logger.debug(f"MA order set to {value}")
        except ModelSpecificationError as e:
            self.view.show_error_message("Invalid MA Order", str(e))
            # Reset view to model's current value
            self.view.set_ma_order(self.model.ma_order)
    
    @pyqtSlot(bool)
    def _on_constant_toggled(self, checked: bool) -> None:
        """
        Handle constant toggle.
        
        Args:
            checked: Whether constant is included
        """
        self.model.include_constant = checked
        logger.debug(f"Include constant set to {checked}")
    
    @pyqtSlot(list)
    def _on_exog_variables_changed(self, variables: List[str]) -> None:
        """
        Handle exogenous variables change.
        
        Args:
            variables: Selected exogenous variables
        """
        self.model.exog_variables = variables
        logger.debug(f"Exogenous variables set to {variables}")
    
    @pyqtSlot()
    def _on_estimate_clicked(self) -> None:
        """Handle estimate button click."""
        # Check if data is available
        if self.model.data is None:
            self.view.show_error_message(
                "No Data",
                "No time series data available for estimation.\n\nPlease load data first."
            )
            return
        
        # Start asynchronous estimation
        self._start_estimation()
    
    @pyqtSlot()
    def _on_reset_clicked(self) -> None:
        """Handle reset button click."""
        # Reset model
        self.model.reset()
        
        # Update view
        self.view.set_ar_order(self.model.ar_order)
        self.view.set_ma_order(self.model.ma_order)
        self.view.set_include_constant(self.model.include_constant)
        self.view.set_estimation_method(self.model.parameters.estimation_method)
        
        # Clear plots
        if self.model.data is not None:
            self.view.plot_time_series(
                self.model.data,
                title=f"Time Series: {self.model.data_name}" if self.model.data_name else "Time Series Data"
            )
        
        # Clear results
        self.view.display_model_equation("No model estimated yet")
        self.view.display_parameter_estimates(None)
        self.view.display_model_statistics({})
        self.view.display_diagnostic_tests({})
        
        # Disable results actions
        self.view.enable_results_actions(False)
        
        logger.debug("Model and view reset")
    
    @pyqtSlot()
    def _on_about_clicked(self) -> None:
        """Handle about button click."""
        # Create about dialog if it doesn't exist
        if self.about_dialog is None:
            self.about_dialog = AboutDialogView(self.view)
            
            # Connect closed signal
            self.about_dialog.closed.connect(self._on_about_dialog_closed)
        
        # Set dialog content
        self.about_dialog.set_content(self.about_model.get_dialog_content())
        
        # Load logo asynchronously
        asyncio.create_task(self._load_about_dialog_logo())
        
        # Show dialog
        self.about_dialog.show()
        logger.debug("About dialog shown")
    
    @pyqtSlot()
    def _on_about_dialog_closed(self) -> None:
        """Handle about dialog closed."""
        logger.debug("About dialog closed")
    
    @pyqtSlot()
    def _on_load_data_clicked(self) -> None:
        """Handle load data button click."""
        # Show file dialog
        file_path = self.view.show_file_dialog(
            mode="open",
            title="Load Time Series Data",
            filter_str="CSV Files (*.csv);;Text Files (*.txt);;All Files (*.*)"
        )
        
        if not file_path:
            logger.debug("Load data cancelled")
            return
        
        try:
            # Load data from file
            data = self._load_data_from_file(file_path)
            
            if data is None:
                return
            
            # Set data in model
            if isinstance(data, pd.DataFrame):
                # If multiple columns, ask user which to use as the main series
                if data.shape[1] > 1:
                    # For now, just use the first column as the main series
                    # and the rest as exogenous variables
                    main_series = data.iloc[:, 0]
                    exog_data = data.iloc[:, 1:]
                    
                    # Set data in model
                    self.model.set_data(main_series)
                    self.model.set_exog_data(exog_data, exog_data.columns.tolist())
                    
                    # Update exogenous variables in view
                    self.view.set_exog_variables(exog_data.columns.tolist())
                else:
                    # Single column DataFrame
                    self.model.set_data(data)
            else:
                # Series or array
                self.model.set_data(data)
            
            # Plot data
            self.view.plot_time_series(
                self.model.data,
                title=f"Time Series: {self.model.data_name}" if self.model.data_name else "Time Series Data"
            )
            
            # Compute and plot ACF/PACF
            acf, pacf = self.model.compute_acf_pacf()
            self.view.plot_acf_pacf(acf, pacf)
            
            logger.debug(f"Data loaded from {file_path}")
            self.view.show_info_message(
                "Data Loaded",
                f"Successfully loaded data from {Path(file_path).name}"
            )
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self.view.show_error_message(
                "Data Loading Error",
                f"Failed to load data from {file_path}",
                details=str(e)
            )
    
    @pyqtSlot()
    def _on_save_results_clicked(self) -> None:
        """Handle save results button click."""
        # Check if model is fitted
        if not self.model.is_fitted or self.model.results is None:
            self.view.show_error_message(
                "No Results",
                "No model results available to save.\n\nPlease estimate a model first."
            )
            return
        
        # Show file dialog
        file_path = self.view.show_file_dialog(
            mode="save",
            title="Save Model Results",
            filter_str="CSV Files (*.csv);;Text Files (*.txt);;JSON Files (*.json);;All Files (*.*)"
        )
        
        if not file_path:
            logger.debug("Save results cancelled")
            return
        
        try:
            # Save results to file
            self._save_results_to_file(file_path)
            
            logger.debug(f"Results saved to {file_path}")
            self.view.show_info_message(
                "Results Saved",
                f"Successfully saved results to {Path(file_path).name}"
            )
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            self.view.show_error_message(
                "Save Error",
                f"Failed to save results to {file_path}",
                details=str(e)
            )
    
    @pyqtSlot()
    def _on_forecast_clicked(self) -> None:
        """Handle forecast button click."""
        # Check if model is fitted
        if not self.model.is_fitted or self.model.results is None:
            self.view.show_error_message(
                "No Model",
                "No fitted model available for forecasting.\n\nPlease estimate a model first."
            )
            return
        
        # TODO: Show forecast dialog to get forecast horizon and other parameters
        # For now, use default values
        steps = 10
        confidence_level = 0.95
        
        # Start asynchronous forecasting
        self._start_forecasting(steps, confidence_level)
    
    @pyqtSlot()
    def _on_cancel_clicked(self) -> None:
        """Handle cancel button click."""
        # Cancel current tasks
        if self._estimation_task is not None and not self._estimation_task.done():
            logger.debug("Cancelling estimation task")
            self.model.cancel_operation()
            # Note: The task will complete with an EstimationError
        
        if self._forecast_task is not None and not self._forecast_task.done():
            logger.debug("Cancelling forecast task")
            self.model.cancel_operation()
            # Note: The task will complete with a ForecastError
        
        logger.debug("Operation cancelled by user")
    
    # Helper methods
    def _start_estimation(self) -> None:
        """Start asynchronous model estimation."""
        # Create and start the estimation task
        self._estimation_task = asyncio.create_task(self._estimate_model_async())
        logger.debug("Estimation task started")
    
    async def _estimate_model_async(self) -> None:
        """
        Estimate the model asynchronously.
        
        This method runs the model estimation in an asynchronous task,
        updating the UI with progress and handling errors.
        """
        # Show progress in view
        self.view.show_progress(0, "Preparing model estimation...")
        
        try:
            # Estimate the model with progress reporting
            results = await self.model.estimate_async(
                progress_callback=self._report_estimation_progress
            )
            
            # Update view with results
            self._display_model_results()
            
            # Enable results actions
            self.view.enable_results_actions(True)
            
            # Show success message
            self.view.show_info_message(
                "Estimation Complete",
                "Model estimation completed successfully."
            )
            
            logger.info("Model estimation completed successfully")
            
        except MFEError as e:
            # Handle known errors
            logger.error(f"Estimation error: {e}")
            self.view.show_error_message(
                "Estimation Error",
                str(e),
                details=getattr(e, "details", None)
            )
            
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error during estimation: {e}")
            self.view.show_error_message(
                "Unexpected Error",
                f"An unexpected error occurred during estimation: {e}",
                details=traceback.format_exc()
            )
            
        finally:
            # Hide progress
            self.view.hide_progress()
    
    async def _report_estimation_progress(self, percent: float, message: str) -> None:
        """
        Report estimation progress to the view.
        
        Args:
            percent: Progress percentage (0-1)
            message: Progress message
        """
        # Convert to integer percentage (0-100)
        percent_int = int(percent * 100)
        
        # Update view
        self.view.show_progress(percent_int, message)
        
        # Allow UI to update
        await asyncio.sleep(0)
    
    def _start_forecasting(self, steps: int, confidence_level: float) -> None:
        """
        Start asynchronous forecasting.
        
        Args:
            steps: Number of steps to forecast
            confidence_level: Confidence level for prediction intervals
        """
        # Create and start the forecast task
        self._forecast_task = asyncio.create_task(
            self._forecast_async(steps, confidence_level)
        )
        logger.debug(f"Forecast task started for {steps} steps")
    
    async def _forecast_async(self, steps: int, confidence_level: float) -> None:
        """
        Generate forecasts asynchronously.
        
        Args:
            steps: Number of steps to forecast
            confidence_level: Confidence level for prediction intervals
        """
        # Show progress in view
        self.view.show_progress(0, "Preparing forecasting...")
        
        try:
            # Generate forecasts with progress reporting
            forecasts, lower_bounds, upper_bounds = await self.model.forecast_async(
                steps=steps,
                confidence_level=confidence_level,
                progress_callback=self._report_forecast_progress
            )
            
            # Plot forecasts
            self.view.plot_forecast(
                self.model.data,
                forecasts,
                lower_bounds,
                upper_bounds,
                title=f"Forecasts for {self.model.data_name}" if self.model.data_name else "Forecasts"
            )
            
            logger.info(f"Generated {steps}-step forecasts successfully")
            
        except MFEError as e:
            # Handle known errors
            logger.error(f"Forecasting error: {e}")
            self.view.show_error_message(
                "Forecasting Error",
                str(e),
                details=getattr(e, "details", None)
            )
            
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error during forecasting: {e}")
            self.view.show_error_message(
                "Unexpected Error",
                f"An unexpected error occurred during forecasting: {e}",
                details=traceback.format_exc()
            )
            
        finally:
            # Hide progress
            self.view.hide_progress()
    
    async def _report_forecast_progress(self, percent: float, message: str) -> None:
        """
        Report forecast progress to the view.
        
        Args:
            percent: Progress percentage (0-1)
            message: Progress message
        """
        # Convert to integer percentage (0-100)
        percent_int = int(percent * 100)
        
        # Update view
        self.view.show_progress(percent_int, message)
        
        # Allow UI to update
        await asyncio.sleep(0)
    
    def _display_model_results(self) -> None:
        """Display model results in the view."""
        if not self.model.is_fitted or self.model.results is None:
            logger.warning("Cannot display results: model not fitted")
            return
        
        results = self.model.results
        
        # Display model equation
        self.view.display_model_equation(results.model_equation)
        
        # Display parameter estimates
        self.view.display_parameter_estimates(results.parameter_table)
        
        # Display model statistics
        stats = {
            "Log-Likelihood": results.log_likelihood,
            "AIC": results.aic,
            "BIC": results.bic,
            "HQIC": results.hqic,
            "Residual Variance": results.sigma2,
            "Iterations": results.iterations,
            "Convergence": "Yes" if results.convergence else "No"
        }
        self.view.display_model_statistics(stats)
        
        # Display diagnostic tests
        self.view.display_diagnostic_tests(results.diagnostic_tests)
        
        # Plot residuals
        if results.residuals is not None:
            self.view.plot_residuals(results.residuals)
        
        # Compute and plot residual ACF/PACF
        try:
            acf, pacf = self.model.compute_residual_acf_pacf()
            self.view.plot_acf_pacf(acf, pacf)
        except Exception as e:
            logger.error(f"Error computing residual ACF/PACF: {e}")
        
        logger.debug("Model results displayed in view")
    
    def _load_data_from_file(self, file_path: str) -> Optional[Union[pd.DataFrame, pd.Series, np.ndarray]]:
        """
        Load data from a file.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Loaded data or None if loading fails
        """
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext == '.csv':
                # Try to load as CSV
                data = pd.read_csv(file_path)
                
                # Check if the first column might be a date/time index
                first_col = data.columns[0]
                try:
                    # Try to parse as datetime
                    date_index = pd.to_datetime(data[first_col])
                    # If successful, set as index and drop the column
                    data.index = date_index
                    data = data.drop(columns=[first_col])
                except:
                    # Not a datetime column, continue with numeric index
                    pass
                
                return data
                
            elif file_ext == '.txt':
                # Try to load as whitespace-delimited text
                data = pd.read_csv(file_path, delim_whitespace=True)
                return data
                
            else:
                # Try to infer format
                data = pd.read_csv(file_path)
                return data
                
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            self.view.show_error_message(
                "Data Loading Error",
                f"Failed to load data from {file_path}",
                details=str(e)
            )
            return None
    
    def _save_results_to_file(self, file_path: str) -> None:
        """
        Save model results to a file.
        
        Args:
            file_path: Path to save the results
        
        Raises:
            Exception: If saving fails
        """
        if not self.model.is_fitted or self.model.results is None:
            raise ValueError("No model results available to save")
        
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.csv':
            # Save parameter table to CSV
            if self.model.results.parameter_table is not None:
                self.model.results.parameter_table.to_csv(file_path)
            else:
                raise ValueError("No parameter table available to save")
                
        elif file_ext == '.json':
            # Save full results as JSON
            import json
            with open(file_path, 'w') as f:
                json.dump(self.model.to_dict(), f, indent=2)
                
        elif file_ext == '.txt':
            # Save summary as text
            with open(file_path, 'w') as f:
                if self.model.results.model_equation:
                    f.write(f"Model Equation:\n{self.model.results.model_equation}\n\n")
                
                f.write("Parameter Estimates:\n")
                if self.model.results.parameter_table is not None:
                    f.write(self.model.results.parameter_table.to_string())
                    f.write("\n\n")
                
                f.write("Model Statistics:\n")
                stats = {
                    "Log-Likelihood": self.model.results.log_likelihood,
                    "AIC": self.model.results.aic,
                    "BIC": self.model.results.bic,
                    "HQIC": self.model.results.hqic,
                    "Residual Variance": self.model.results.sigma2,
                    "Iterations": self.model.results.iterations,
                    "Convergence": "Yes" if self.model.results.convergence else "No"
                }
                for key, value in stats.items():
                    if value is not None:
                        if isinstance(value, (int, float)):
                            f.write(f"{key}: {value:.6f}\n")
                        else:
                            f.write(f"{key}: {value}\n")
                
                f.write("\nDiagnostic Tests:\n")
                for test_name, test_result in self.model.results.diagnostic_tests.items():
                    f.write(f"{test_name}:\n")
                    if isinstance(test_result, dict):
                        for key, value in test_result.items():
                            if isinstance(value, list):
                                f.write(f"  {key}: {', '.join(map(str, value))}\n")
                            else:
                                f.write(f"  {key}: {value}\n")
                    else:
                        f.write(f"  {test_result}\n")
        else:
            # Default to CSV
            if self.model.results.parameter_table is not None:
                self.model.results.parameter_table.to_csv(file_path)
            else:
                raise ValueError("No parameter table available to save")
    
    async def _load_about_dialog_logo(self) -> None:
        """Load the logo for the About dialog asynchronously."""
        if self.about_dialog is None:
            return
        
        try:
            # Get logo path from model
            logo_path = self.about_model.logo_path
            
            # Load logo asynchronously
            await self.about_dialog.load_logo_async(logo_path)
            
        except Exception as e:
            logger.error(f"Error loading About dialog logo: {e}")