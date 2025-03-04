'''
Custom exception classes for the MFE Toolbox.

This module defines a comprehensive hierarchy of exception classes used throughout
the MFE Toolbox. These specialized exceptions provide clear, context-specific error
messages and additional diagnostic information to help users identify and resolve
issues. Each exception type is designed for a specific category of errors, making
error handling more precise and informative.

The exception hierarchy follows a logical structure, with base classes for general
categories and specialized subclasses for specific error conditions. This approach
enables both precise error handling and graceful error recovery where appropriate.
'''

from typing import Any, Dict, List, Optional, Tuple, Type, Union
import inspect
import numpy as np
from pathlib import Path


class MFEError(Exception):
    """Base exception class for all MFE Toolbox errors.
    
    This class serves as the foundation for all custom exceptions in the MFE Toolbox,
    providing consistent error formatting and additional context information.
    
    Attributes:
        message: The error message
        details: Additional details about the error
        context: Dictionary containing contextual information about the error
    """
    
    def __init__(self, 
                 message: str, 
                 details: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the MFEError.
        
        Args:
            message: The primary error message
            details: Additional details about the error
            context: Dictionary containing contextual information about the error
        """
        self.message = message
        self.details = details
        self.context = context or {}
        
        # Build the full error message
        full_message = message
        if details:
            full_message += f"\n\nDetails: {details}"
        
        if context:
            context_str = "\n".join(f"  {k}: {v}" for k, v in context.items())
            full_message += f"\n\nContext:\n{context_str}"
        
        # Add caller information for better debugging
        frame = inspect.currentframe()
        if frame:
            try:
                frame = frame.f_back  # Get the caller's frame
                if frame:
                    caller_info = inspect.getframeinfo(frame)
                    full_message += f"\n\nLocation: {Path(caller_info.filename).name}:{caller_info.lineno}"
            finally:
                del frame  # Avoid reference cycles
        
        super().__init__(full_message)


class ParameterError(MFEError):
    """Exception raised for errors related to model parameters.
    
    This exception is used when model parameters violate constraints or are otherwise invalid.
    
    Attributes:
        param_name: The name of the parameter that caused the error
        param_value: The invalid parameter value
        constraint: Description of the constraint that was violated
    """
    
    def __init__(self, 
                 message: str, 
                 param_name: Optional[str] = None,
                 param_value: Optional[Any] = None,
                 constraint: Optional[str] = None,
                 details: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the ParameterError.
        
        Args:
            message: The primary error message
            param_name: The name of the parameter that caused the error
            param_value: The invalid parameter value
            constraint: Description of the constraint that was violated
            details: Additional details about the error
            context: Dictionary containing contextual information about the error
        """
        self.param_name = param_name
        self.param_value = param_value
        self.constraint = constraint
        
        # Add parameter information to context
        context_dict = context or {}
        if param_name:
            context_dict["Parameter"] = param_name
        if param_value is not None:
            context_dict["Value"] = param_value
        if constraint:
            context_dict["Constraint"] = constraint
        
        super().__init__(message, details, context_dict)


class DimensionError(MFEError):
    """Exception raised for errors related to array dimensions.
    
    This exception is used when array dimensions are incompatible with the expected dimensions.
    
    Attributes:
        array_name: The name of the array that caused the error
        expected_shape: The expected shape of the array
        actual_shape: The actual shape of the array
    """
    
    def __init__(self, 
                 message: str, 
                 array_name: Optional[str] = None,
                 expected_shape: Optional[Union[Tuple[int, ...], str]] = None,
                 actual_shape: Optional[Tuple[int, ...]] = None,
                 details: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the DimensionError.
        
        Args:
            message: The primary error message
            array_name: The name of the array that caused the error
            expected_shape: The expected shape of the array
            actual_shape: The actual shape of the array
            details: Additional details about the error
            context: Dictionary containing contextual information about the error
        """
        self.array_name = array_name
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape
        
        # Add dimension information to context
        context_dict = context or {}
        if array_name:
            context_dict["Array"] = array_name
        if expected_shape is not None:
            context_dict["Expected Shape"] = expected_shape
        if actual_shape is not None:
            context_dict["Actual Shape"] = actual_shape
        
        super().__init__(message, details, context_dict)


class ConvergenceError(MFEError):
    """Exception raised when an optimization algorithm fails to converge.
    
    This exception is used when model estimation or other optimization processes
    fail to converge to a solution.
    
    Attributes:
        iterations: The number of iterations performed before failure
        tolerance: The convergence tolerance that was used
        final_value: The final objective function value
        gradient_norm: The norm of the final gradient
    """
    
    def __init__(self, 
                 message: str, 
                 iterations: Optional[int] = None,
                 tolerance: Optional[float] = None,
                 final_value: Optional[float] = None,
                 gradient_norm: Optional[float] = None,
                 details: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the ConvergenceError.
        
        Args:
            message: The primary error message
            iterations: The number of iterations performed before failure
            tolerance: The convergence tolerance that was used
            final_value: The final objective function value
            gradient_norm: The norm of the final gradient
            details: Additional details about the error
            context: Dictionary containing contextual information about the error
        """
        self.iterations = iterations
        self.tolerance = tolerance
        self.final_value = final_value
        self.gradient_norm = gradient_norm
        
        # Add convergence information to context
        context_dict = context or {}
        if iterations is not None:
            context_dict["Iterations"] = iterations
        if tolerance is not None:
            context_dict["Tolerance"] = tolerance
        if final_value is not None:
            context_dict["Final Value"] = final_value
        if gradient_norm is not None:
            context_dict["Gradient Norm"] = gradient_norm
        
        super().__init__(message, details, context_dict)


class NumericError(MFEError):
    """Exception raised for numerical computation errors.
    
    This exception is used when numerical issues such as overflow, underflow,
    or division by zero occur during computation.
    
    Attributes:
        operation: The operation that caused the error
        values: The values that caused the error
        error_type: The type of numerical error (e.g., "overflow", "underflow")
    """
    
    def __init__(self, 
                 message: str, 
                 operation: Optional[str] = None,
                 values: Optional[Any] = None,
                 error_type: Optional[str] = None,
                 details: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the NumericError.
        
        Args:
            message: The primary error message
            operation: The operation that caused the error
            values: The values that caused the error
            error_type: The type of numerical error (e.g., "overflow", "underflow")
            details: Additional details about the error
            context: Dictionary containing contextual information about the error
        """
        self.operation = operation
        self.values = values
        self.error_type = error_type
        
        # Add numerical information to context
        context_dict = context or {}
        if operation:
            context_dict["Operation"] = operation
        if values is not None:
            if isinstance(values, np.ndarray) and values.size > 10:
                # Truncate large arrays for readability
                context_dict["Values"] = f"Array with shape {values.shape}"
            else:
                context_dict["Values"] = values
        if error_type:
            context_dict["Error Type"] = error_type
        
        super().__init__(message, details, context_dict)


class DataError(MFEError):
    """Exception raised for errors related to input data.
    
    This exception is used when input data is invalid, contains missing values,
    or is otherwise unsuitable for the requested operation.
    
    Attributes:
        data_name: The name of the data that caused the error
        issue: Description of the issue with the data
        index: The index or location where the issue was detected
    """
    
    def __init__(self, 
                 message: str, 
                 data_name: Optional[str] = None,
                 issue: Optional[str] = None,
                 index: Optional[Union[int, Tuple[int, ...], str]] = None,
                 details: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the DataError.
        
        Args:
            message: The primary error message
            data_name: The name of the data that caused the error
            issue: Description of the issue with the data
            index: The index or location where the issue was detected
            details: Additional details about the error
            context: Dictionary containing contextual information about the error
        """
        self.data_name = data_name
        self.issue = issue
        self.index = index
        
        # Add data information to context
        context_dict = context or {}
        if data_name:
            context_dict["Data"] = data_name
        if issue:
            context_dict["Issue"] = issue
        if index is not None:
            context_dict["Index"] = index
        
        super().__init__(message, details, context_dict)


class ModelSpecificationError(MFEError):
    """Exception raised for errors in model specification.
    
    This exception is used when a model is incorrectly specified, such as
    invalid model orders or incompatible model components.
    
    Attributes:
        model_type: The type of model being specified
        parameter: The parameter or component that is incorrectly specified
        valid_options: List of valid options for the parameter
    """
    
    def __init__(self, 
                 message: str, 
                 model_type: Optional[str] = None,
                 parameter: Optional[str] = None,
                 valid_options: Optional[List[Any]] = None,
                 details: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the ModelSpecificationError.
        
        Args:
            message: The primary error message
            model_type: The type of model being specified
            parameter: The parameter or component that is incorrectly specified
            valid_options: List of valid options for the parameter
            details: Additional details about the error
            context: Dictionary containing contextual information about the error
        """
        self.model_type = model_type
        self.parameter = parameter
        self.valid_options = valid_options
        
        # Add model specification information to context
        context_dict = context or {}
        if model_type:
            context_dict["Model Type"] = model_type
        if parameter:
            context_dict["Parameter"] = parameter
        if valid_options:
            context_dict["Valid Options"] = valid_options
        
        super().__init__(message, details, context_dict)


class EstimationError(MFEError):
    """Exception raised for errors during model estimation.
    
    This exception is used when model estimation fails for reasons other than
    convergence issues, such as singular matrices or invalid likelihood values.
    
    Attributes:
        model_type: The type of model being estimated
        estimation_method: The estimation method being used
        issue: Description of the issue that occurred during estimation
    """
    
    def __init__(self, 
                 message: str, 
                 model_type: Optional[str] = None,
                 estimation_method: Optional[str] = None,
                 issue: Optional[str] = None,
                 details: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the EstimationError.
        
        Args:
            message: The primary error message
            model_type: The type of model being estimated
            estimation_method: The estimation method being used
            issue: Description of the issue that occurred during estimation
            details: Additional details about the error
            context: Dictionary containing contextual information about the error
        """
        self.model_type = model_type
        self.estimation_method = estimation_method
        self.issue = issue
        
        # Add estimation information to context
        context_dict = context or {}
        if model_type:
            context_dict["Model Type"] = model_type
        if estimation_method:
            context_dict["Estimation Method"] = estimation_method
        if issue:
            context_dict["Issue"] = issue
        
        super().__init__(message, details, context_dict)


class ForecastError(MFEError):
    """Exception raised for errors during forecasting.
    
    This exception is used when forecasting fails, such as due to invalid
    forecast horizons or missing required inputs.
    
    Attributes:
        model_type: The type of model being used for forecasting
        horizon: The forecast horizon that was requested
        issue: Description of the issue that occurred during forecasting
    """
    
    def __init__(self, 
                 message: str, 
                 model_type: Optional[str] = None,
                 horizon: Optional[int] = None,
                 issue: Optional[str] = None,
                 details: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the ForecastError.
        
        Args:
            message: The primary error message
            model_type: The type of model being used for forecasting
            horizon: The forecast horizon that was requested
            issue: Description of the issue that occurred during forecasting
            details: Additional details about the error
            context: Dictionary containing contextual information about the error
        """
        self.model_type = model_type
        self.horizon = horizon
        self.issue = issue
        
        # Add forecasting information to context
        context_dict = context or {}
        if model_type:
            context_dict["Model Type"] = model_type
        if horizon is not None:
            context_dict["Horizon"] = horizon
        if issue:
            context_dict["Issue"] = issue
        
        super().__init__(message, details, context_dict)


class SimulationError(MFEError):
    """Exception raised for errors during simulation.
    
    This exception is used when simulation fails, such as due to invalid
    simulation parameters or issues with random number generation.
    
    Attributes:
        model_type: The type of model being used for simulation
        n_periods: The number of periods to simulate
        issue: Description of the issue that occurred during simulation
    """
    
    def __init__(self, 
                 message: str, 
                 model_type: Optional[str] = None,
                 n_periods: Optional[int] = None,
                 issue: Optional[str] = None,
                 details: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the SimulationError.
        
        Args:
            message: The primary error message
            model_type: The type of model being used for simulation
            n_periods: The number of periods to simulate
            issue: Description of the issue that occurred during simulation
            details: Additional details about the error
            context: Dictionary containing contextual information about the error
        """
        self.model_type = model_type
        self.n_periods = n_periods
        self.issue = issue
        
        # Add simulation information to context
        context_dict = context or {}
        if model_type:
            context_dict["Model Type"] = model_type
        if n_periods is not None:
            context_dict["Periods"] = n_periods
        if issue:
            context_dict["Issue"] = issue
        
        super().__init__(message, details, context_dict)


class BootstrapError(MFEError):
    """Exception raised for errors during bootstrap procedures.
    
    This exception is used when bootstrap procedures fail, such as due to
    invalid block lengths or issues with resampling.
    
    Attributes:
        bootstrap_type: The type of bootstrap being used
        n_bootstraps: The number of bootstrap replications
        issue: Description of the issue that occurred during bootstrap
    """
    
    def __init__(self, 
                 message: str, 
                 bootstrap_type: Optional[str] = None,
                 n_bootstraps: Optional[int] = None,
                 issue: Optional[str] = None,
                 details: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the BootstrapError.
        
        Args:
            message: The primary error message
            bootstrap_type: The type of bootstrap being used
            n_bootstraps: The number of bootstrap replications
            issue: Description of the issue that occurred during bootstrap
            details: Additional details about the error
            context: Dictionary containing contextual information about the error
        """
        self.bootstrap_type = bootstrap_type
        self.n_bootstraps = n_bootstraps
        self.issue = issue
        
        # Add bootstrap information to context
        context_dict = context or {}
        if bootstrap_type:
            context_dict["Bootstrap Type"] = bootstrap_type
        if n_bootstraps is not None:
            context_dict["Replications"] = n_bootstraps
        if issue:
            context_dict["Issue"] = issue
        
        super().__init__(message, details, context_dict)


class DistributionError(MFEError):
    """Exception raised for errors related to probability distributions.
    
    This exception is used when there are issues with probability distributions,
    such as invalid parameters or unsupported distribution types.
    
    Attributes:
        distribution_type: The type of distribution
        parameter: The parameter that caused the error
        value: The invalid parameter value
        issue: Description of the issue with the distribution
    """
    
    def __init__(self, 
                 message: str, 
                 distribution_type: Optional[str] = None,
                 parameter: Optional[str] = None,
                 value: Optional[Any] = None,
                 issue: Optional[str] = None,
                 details: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the DistributionError.
        
        Args:
            message: The primary error message
            distribution_type: The type of distribution
            parameter: The parameter that caused the error
            value: The invalid parameter value
            issue: Description of the issue with the distribution
            details: Additional details about the error
            context: Dictionary containing contextual information about the error
        """
        self.distribution_type = distribution_type
        self.parameter = parameter
        self.value = value
        self.issue = issue
        
        # Add distribution information to context
        context_dict = context or {}
        if distribution_type:
            context_dict["Distribution"] = distribution_type
        if parameter:
            context_dict["Parameter"] = parameter
        if value is not None:
            context_dict["Value"] = value
        if issue:
            context_dict["Issue"] = issue
        
        super().__init__(message, details, context_dict)


class TestError(MFEError):
    """Exception raised for errors during statistical tests.
    
    This exception is used when statistical tests fail, such as due to
    invalid test parameters or issues with test statistic computation.
    
    Attributes:
        test_type: The type of statistical test
        parameter: The parameter that caused the error
        issue: Description of the issue with the test
    """
    
    def __init__(self, 
                 message: str, 
                 test_type: Optional[str] = None,
                 parameter: Optional[str] = None,
                 issue: Optional[str] = None,
                 details: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the TestError.
        
        Args:
            message: The primary error message
            test_type: The type of statistical test
            parameter: The parameter that caused the error
            issue: Description of the issue with the test
            details: Additional details about the error
            context: Dictionary containing contextual information about the error
        """
        self.test_type = test_type
        self.parameter = parameter
        self.issue = issue
        
        # Add test information to context
        context_dict = context or {}
        if test_type:
            context_dict["Test Type"] = test_type
        if parameter:
            context_dict["Parameter"] = parameter
        if issue:
            context_dict["Issue"] = issue
        
        super().__init__(message, details, context_dict)


class RealizedVolatilityError(MFEError):
    """Exception raised for errors in realized volatility estimation.
    
    This exception is used when realized volatility estimation fails, such as due to
    invalid price data or issues with sampling schemes.
    
    Attributes:
        estimator_type: The type of realized volatility estimator
        sampling_scheme: The sampling scheme being used
        issue: Description of the issue with the estimation
    """
    
    def __init__(self, 
                 message: str, 
                 estimator_type: Optional[str] = None,
                 sampling_scheme: Optional[str] = None,
                 issue: Optional[str] = None,
                 details: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the RealizedVolatilityError.
        
        Args:
            message: The primary error message
            estimator_type: The type of realized volatility estimator
            sampling_scheme: The sampling scheme being used
            issue: Description of the issue with the estimation
            details: Additional details about the error
            context: Dictionary containing contextual information about the error
        """
        self.estimator_type = estimator_type
        self.sampling_scheme = sampling_scheme
        self.issue = issue
        
        # Add realized volatility information to context
        context_dict = context or {}
        if estimator_type:
            context_dict["Estimator"] = estimator_type
        if sampling_scheme:
            context_dict["Sampling Scheme"] = sampling_scheme
        if issue:
            context_dict["Issue"] = issue
        
        super().__init__(message, details, context_dict)


class OptimizationError(MFEError):
    """Exception raised for errors during optimization.
    
    This exception is used when optimization fails for reasons other than
    convergence issues, such as invalid objective functions or constraints.
    
    Attributes:
        algorithm: The optimization algorithm being used
        objective: Description of the objective function
        issue: Description of the issue with the optimization
    """
    
    def __init__(self, 
                 message: str, 
                 algorithm: Optional[str] = None,
                 objective: Optional[str] = None,
                 issue: Optional[str] = None,
                 details: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the OptimizationError.
        
        Args:
            message: The primary error message
            algorithm: The optimization algorithm being used
            objective: Description of the objective function
            issue: Description of the issue with the optimization
            details: Additional details about the error
            context: Dictionary containing contextual information about the error
        """
        self.algorithm = algorithm
        self.objective = objective
        self.issue = issue
        
        # Add optimization information to context
        context_dict = context or {}
        if algorithm:
            context_dict["Algorithm"] = algorithm
        if objective:
            context_dict["Objective"] = objective
        if issue:
            context_dict["Issue"] = issue
        
        super().__init__(message, details, context_dict)


class ConfigurationError(MFEError):
    """Exception raised for errors in configuration.
    
    This exception is used when there are issues with configuration settings,
    such as invalid options or missing required settings.
    
    Attributes:
        config_file: The configuration file path
        setting: The setting that caused the error
        value: The invalid setting value
        issue: Description of the issue with the configuration
    """
    
    def __init__(self, 
                 message: str, 
                 config_file: Optional[Union[str, Path]] = None,
                 setting: Optional[str] = None,
                 value: Optional[Any] = None,
                 issue: Optional[str] = None,
                 details: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the ConfigurationError.
        
        Args:
            message: The primary error message
            config_file: The configuration file path
            setting: The setting that caused the error
            value: The invalid setting value
            issue: Description of the issue with the configuration
            details: Additional details about the error
            context: Dictionary containing contextual information about the error
        """
        self.config_file = config_file
        self.setting = setting
        self.value = value
        self.issue = issue
        
        # Add configuration information to context
        context_dict = context or {}
        if config_file:
            context_dict["Config File"] = str(config_file)
        if setting:
            context_dict["Setting"] = setting
        if value is not None:
            context_dict["Value"] = value
        if issue:
            context_dict["Issue"] = issue
        
        super().__init__(message, details, context_dict)


class UIError(MFEError):
    """Exception raised for errors in the user interface.
    
    This exception is used when there are issues with the user interface,
    such as invalid inputs or UI component failures.
    
    Attributes:
        component: The UI component that caused the error
        action: The action that was being performed
        issue: Description of the issue with the UI
    """
    
    def __init__(self, 
                 message: str, 
                 component: Optional[str] = None,
                 action: Optional[str] = None,
                 issue: Optional[str] = None,
                 details: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the UIError.
        
        Args:
            message: The primary error message
            component: The UI component that caused the error
            action: The action that was being performed
            issue: Description of the issue with the UI
            details: Additional details about the error
            context: Dictionary containing contextual information about the error
        """
        self.component = component
        self.action = action
        self.issue = issue
        
        # Add UI information to context
        context_dict = context or {}
        if component:
            context_dict["Component"] = component
        if action:
            context_dict["Action"] = action
        if issue:
            context_dict["Issue"] = issue
        
        super().__init__(message, details, context_dict)


class AsyncError(MFEError):
    """Exception raised for errors in asynchronous operations.
    
    This exception is used when asynchronous operations fail, such as due to
    cancellation or timeout.
    
    Attributes:
        operation: The asynchronous operation that failed
        duration: The duration of the operation before failure
        issue: Description of the issue with the asynchronous operation
    """
    
    def __init__(self, 
                 message: str, 
                 operation: Optional[str] = None,
                 duration: Optional[float] = None,
                 issue: Optional[str] = None,
                 details: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the AsyncError.
        
        Args:
            message: The primary error message
            operation: The asynchronous operation that failed
            duration: The duration of the operation before failure
            issue: Description of the issue with the asynchronous operation
            details: Additional details about the error
            context: Dictionary containing contextual information about the error
        """
        self.operation = operation
        self.duration = duration
        self.issue = issue
        
        # Add asynchronous operation information to context
        context_dict = context or {}
        if operation:
            context_dict["Operation"] = operation
        if duration is not None:
            context_dict["Duration"] = f"{duration:.2f} seconds"
        if issue:
            context_dict["Issue"] = issue
        
        super().__init__(message, details, context_dict)


class NotFittedError(MFEError):
    """Exception raised when an operation requires a fitted model.
    
    This exception is used when an operation is attempted on a model that has not
    been fitted, such as forecasting or simulation.
    
    Attributes:
        model_type: The type of model
        operation: The operation that requires a fitted model
    """
    
    def __init__(self, 
                 message: str, 
                 model_type: Optional[str] = None,
                 operation: Optional[str] = None,
                 details: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the NotFittedError.
        
        Args:
            message: The primary error message
            model_type: The type of model
            operation: The operation that requires a fitted model
            details: Additional details about the error
            context: Dictionary containing contextual information about the error
        """
        self.model_type = model_type
        self.operation = operation
        
        # Add model information to context
        context_dict = context or {}
        if model_type:
            context_dict["Model Type"] = model_type
        if operation:
            context_dict["Operation"] = operation
        
        super().__init__(message, details, context_dict)


class MFEWarning(Warning):
    """Base warning class for all MFE Toolbox warnings.
    
    This class serves as the foundation for all custom warnings in the MFE Toolbox,
    providing consistent warning formatting and additional context information.
    
    Attributes:
        message: The warning message
        details: Additional details about the warning
        context: Dictionary containing contextual information about the warning
    """
    
    def __init__(self, 
                 message: str, 
                 details: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the MFEWarning.
        
        Args:
            message: The primary warning message
            details: Additional details about the warning
            context: Dictionary containing contextual information about the warning
        """
        self.message = message
        self.details = details
        self.context = context or {}
        
        # Build the full warning message
        full_message = message
        if details:
            full_message += f"\n\nDetails: {details}"
        
        if context:
            context_str = "\n".join(f"  {k}: {v}" for k, v in context.items())
            full_message += f"\n\nContext:\n{context_str}"
        
        # Add caller information for better debugging
        frame = inspect.currentframe()
        if frame:
            try:
                frame = frame.f_back  # Get the caller's frame
                if frame:
                    caller_info = inspect.getframeinfo(frame)
                    full_message += f"\n\nLocation: {Path(caller_info.filename).name}:{caller_info.lineno}"
            finally:
                del frame  # Avoid reference cycles
        
        super().__init__(full_message)


class ConvergenceWarning(MFEWarning):
    """Warning for potential convergence issues.
    
    This warning is used when convergence is achieved but may be questionable,
    such as when the optimization terminates at the maximum number of iterations.
    
    Attributes:
        iterations: The number of iterations performed
        tolerance: The convergence tolerance that was used
        gradient_norm: The norm of the final gradient
    """
    
    def __init__(self, 
                 message: str, 
                 iterations: Optional[int] = None,
                 tolerance: Optional[float] = None,
                 gradient_norm: Optional[float] = None,
                 details: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the ConvergenceWarning.
        
        Args:
            message: The primary warning message
            iterations: The number of iterations performed
            tolerance: The convergence tolerance that was used
            gradient_norm: The norm of the final gradient
            details: Additional details about the warning
            context: Dictionary containing contextual information about the warning
        """
        self.iterations = iterations
        self.tolerance = tolerance
        self.gradient_norm = gradient_norm
        
        # Add convergence information to context
        context_dict = context or {}
        if iterations is not None:
            context_dict["Iterations"] = iterations
        if tolerance is not None:
            context_dict["Tolerance"] = tolerance
        if gradient_norm is not None:
            context_dict["Gradient Norm"] = gradient_norm
        
        super().__init__(message, details, context_dict)


class NumericWarning(MFEWarning):
    """Warning for potential numerical issues.
    
    This warning is used when numerical issues are detected but do not prevent
    computation, such as near-singular matrices or large condition numbers.
    
    Attributes:
        operation: The operation where the issue was detected
        issue: Description of the numerical issue
        value: The value that may cause numerical issues
    """
    
    def __init__(self, 
                 message: str, 
                 operation: Optional[str] = None,
                 issue: Optional[str] = None,
                 value: Optional[Any] = None,
                 details: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the NumericWarning.
        
        Args:
            message: The primary warning message
            operation: The operation where the issue was detected
            issue: Description of the numerical issue
            value: The value that may cause numerical issues
            details: Additional details about the warning
            context: Dictionary containing contextual information about the warning
        """
        self.operation = operation
        self.issue = issue
        self.value = value
        
        # Add numerical information to context
        context_dict = context or {}
        if operation:
            context_dict["Operation"] = operation
        if issue:
            context_dict["Issue"] = issue
        if value is not None:
            if isinstance(value, np.ndarray) and value.size > 10:
                # Truncate large arrays for readability
                context_dict["Value"] = f"Array with shape {value.shape}"
            else:
                context_dict["Value"] = value
        
        super().__init__(message, details, context_dict)


class ModelWarning(MFEWarning):
    """Warning for potential model issues.
    
    This warning is used when model issues are detected but do not prevent
    estimation or usage, such as near-nonstationary models or unusual parameter values.
    
    Attributes:
        model_type: The type of model
        issue: Description of the model issue
        parameter: The parameter that may cause issues
        value: The parameter value that may cause issues
    """
    
    def __init__(self, 
                 message: str, 
                 model_type: Optional[str] = None,
                 issue: Optional[str] = None,
                 parameter: Optional[str] = None,
                 value: Optional[Any] = None,
                 details: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the ModelWarning.
        
        Args:
            message: The primary warning message
            model_type: The type of model
            issue: Description of the model issue
            parameter: The parameter that may cause issues
            value: The parameter value that may cause issues
            details: Additional details about the warning
            context: Dictionary containing contextual information about the warning
        """
        self.model_type = model_type
        self.issue = issue
        self.parameter = parameter
        self.value = value
        
        # Add model information to context
        context_dict = context or {}
        if model_type:
            context_dict["Model Type"] = model_type
        if issue:
            context_dict["Issue"] = issue
        if parameter:
            context_dict["Parameter"] = parameter
        if value is not None:
            context_dict["Value"] = value
        
        super().__init__(message, details, context_dict)


class PerformanceWarning(MFEWarning):
    """Warning for potential performance issues.
    
    This warning is used when performance issues are detected, such as
    inefficient algorithms or large data structures.
    
    Attributes:
        operation: The operation where the performance issue was detected
        issue: Description of the performance issue
        suggestion: Suggested action to improve performance
    """
    
    def __init__(self, 
                 message: str, 
                 operation: Optional[str] = None,
                 issue: Optional[str] = None,
                 suggestion: Optional[str] = None,
                 details: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the PerformanceWarning.
        
        Args:
            message: The primary warning message
            operation: The operation where the performance issue was detected
            issue: Description of the performance issue
            suggestion: Suggested action to improve performance
            details: Additional details about the warning
            context: Dictionary containing contextual information about the warning
        """
        self.operation = operation
        self.issue = issue
        self.suggestion = suggestion
        
        # Add performance information to context
        context_dict = context or {}
        if operation:
            context_dict["Operation"] = operation
        if issue:
            context_dict["Issue"] = issue
        if suggestion:
            context_dict["Suggestion"] = suggestion
        
        super().__init__(message, details, context_dict)


class DeprecationWarning(MFEWarning):
    """Warning for deprecated features.
    
    This warning is used when a deprecated feature is used, providing information
    about the deprecation and suggested alternatives.
    
    Attributes:
        feature: The deprecated feature
        alternative: The suggested alternative
        removal_version: The version when the feature will be removed
    """
    
    def __init__(self, 
                 message: str, 
                 feature: Optional[str] = None,
                 alternative: Optional[str] = None,
                 removal_version: Optional[str] = None,
                 details: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the DeprecationWarning.
        
        Args:
            message: The primary warning message
            feature: The deprecated feature
            alternative: The suggested alternative
            removal_version: The version when the feature will be removed
            details: Additional details about the warning
            context: Dictionary containing contextual information about the warning
        """
        self.feature = feature
        self.alternative = alternative
        self.removal_version = removal_version
        
        # Add deprecation information to context
        context_dict = context or {}
        if feature:
            context_dict["Feature"] = feature
        if alternative:
            context_dict["Alternative"] = alternative
        if removal_version:
            context_dict["Removal Version"] = removal_version
        
        super().__init__(message, details, context_dict)


# Helper functions for raising exceptions with consistent formatting

def raise_parameter_error(message: str, 
                         param_name: Optional[str] = None,
                         param_value: Optional[Any] = None,
                         constraint: Optional[str] = None,
                         details: Optional[str] = None, 
                         context: Optional[Dict[str, Any]] = None) -> None:
    """Raise a ParameterError with consistent formatting.
    
    Args:
        message: The primary error message
        param_name: The name of the parameter that caused the error
        param_value: The invalid parameter value
        constraint: Description of the constraint that was violated
        details: Additional details about the error
        context: Dictionary containing contextual information about the error
        
    Raises:
        ParameterError: The formatted parameter error
    """
    raise ParameterError(message, param_name, param_value, constraint, details, context)


def raise_dimension_error(message: str, 
                         array_name: Optional[str] = None,
                         expected_shape: Optional[Union[Tuple[int, ...], str]] = None,
                         actual_shape: Optional[Tuple[int, ...]] = None,
                         details: Optional[str] = None, 
                         context: Optional[Dict[str, Any]] = None) -> None:
    """Raise a DimensionError with consistent formatting.
    
    Args:
        message: The primary error message
        array_name: The name of the array that caused the error
        expected_shape: The expected shape of the array
        actual_shape: The actual shape of the array
        details: Additional details about the error
        context: Dictionary containing contextual information about the error
        
    Raises:
        DimensionError: The formatted dimension error
    """
    raise DimensionError(message, array_name, expected_shape, actual_shape, details, context)


def raise_convergence_error(message: str, 
                           iterations: Optional[int] = None,
                           tolerance: Optional[float] = None,
                           final_value: Optional[float] = None,
                           gradient_norm: Optional[float] = None,
                           details: Optional[str] = None, 
                           context: Optional[Dict[str, Any]] = None) -> None:
    """Raise a ConvergenceError with consistent formatting.
    
    Args:
        message: The primary error message
        iterations: The number of iterations performed before failure
        tolerance: The convergence tolerance that was used
        final_value: The final objective function value
        gradient_norm: The norm of the final gradient
        details: Additional details about the error
        context: Dictionary containing contextual information about the error
        
    Raises:
        ConvergenceError: The formatted convergence error
    """
    raise ConvergenceError(message, iterations, tolerance, final_value, gradient_norm, details, context)


def raise_numeric_error(message: str, 
                       operation: Optional[str] = None,
                       values: Optional[Any] = None,
                       error_type: Optional[str] = None,
                       details: Optional[str] = None, 
                       context: Optional[Dict[str, Any]] = None) -> None:
    """Raise a NumericError with consistent formatting.
    
    Args:
        message: The primary error message
        operation: The operation that caused the error
        values: The values that caused the error
        error_type: The type of numerical error (e.g., "overflow", "underflow")
        details: Additional details about the error
        context: Dictionary containing contextual information about the error
        
    Raises:
        NumericError: The formatted numeric error
    """
    raise NumericError(message, operation, values, error_type, details, context)


def raise_data_error(message: str, 
                    data_name: Optional[str] = None,
                    issue: Optional[str] = None,
                    index: Optional[Union[int, Tuple[int, ...], str]] = None,
                    details: Optional[str] = None, 
                    context: Optional[Dict[str, Any]] = None) -> None:
    """Raise a DataError with consistent formatting.
    
    Args:
        message: The primary error message
        data_name: The name of the data that caused the error
        issue: Description of the issue with the data
        index: The index or location where the issue was detected
        details: Additional details about the error
        context: Dictionary containing contextual information about the error
        
    Raises:
        DataError: The formatted data error
    """
    raise DataError(message, data_name, issue, index, details, context)


def raise_not_fitted_error(message: str, 
                          model_type: Optional[str] = None,
                          operation: Optional[str] = None,
                          details: Optional[str] = None, 
                          context: Optional[Dict[str, Any]] = None) -> None:
    """Raise a NotFittedError with consistent formatting.
    
    Args:
        message: The primary error message
        model_type: The type of model
        operation: The operation that requires a fitted model
        details: Additional details about the error
        context: Dictionary containing contextual information about the error
        
    Raises:
        NotFittedError: The formatted not fitted error
    """
    raise NotFittedError(message, model_type, operation, details, context)


# Helper functions for issuing warnings with consistent formatting

def warn_convergence(message: str, 
                    iterations: Optional[int] = None,
                    tolerance: Optional[float] = None,
                    gradient_norm: Optional[float] = None,
                    details: Optional[str] = None, 
                    context: Optional[Dict[str, Any]] = None) -> None:
    """Issue a ConvergenceWarning with consistent formatting.
    
    Args:
        message: The primary warning message
        iterations: The number of iterations performed
        tolerance: The convergence tolerance that was used
        gradient_norm: The norm of the final gradient
        details: Additional details about the warning
        context: Dictionary containing contextual information about the warning
    """
    import warnings
    warnings.warn(
        ConvergenceWarning(message, iterations, tolerance, gradient_norm, details, context)
    )


def warn_numeric(message: str, 
                operation: Optional[str] = None,
                issue: Optional[str] = None,
                value: Optional[Any] = None,
                details: Optional[str] = None, 
                context: Optional[Dict[str, Any]] = None) -> None:
    """Issue a NumericWarning with consistent formatting.
    
    Args:
        message: The primary warning message
        operation: The operation where the issue was detected
        issue: Description of the numerical issue
        value: The value that may cause numerical issues
        details: Additional details about the warning
        context: Dictionary containing contextual information about the warning
    """
    import warnings
    warnings.warn(
        NumericWarning(message, operation, issue, value, details, context)
    )


def warn_model(message: str, 
              model_type: Optional[str] = None,
              issue: Optional[str] = None,
              parameter: Optional[str] = None,
              value: Optional[Any] = None,
              details: Optional[str] = None, 
              context: Optional[Dict[str, Any]] = None) -> None:
    """Issue a ModelWarning with consistent formatting.
    
    Args:
        message: The primary warning message
        model_type: The type of model
        issue: Description of the model issue
        parameter: The parameter that may cause issues
        value: The parameter value that may cause issues
        details: Additional details about the warning
        context: Dictionary containing contextual information about the warning
    """
    import warnings
    warnings.warn(
        ModelWarning(message, model_type, issue, parameter, value, details, context)
    )


def warn_performance(message: str, 
                    operation: Optional[str] = None,
                    issue: Optional[str] = None,
                    suggestion: Optional[str] = None,
                    details: Optional[str] = None, 
                    context: Optional[Dict[str, Any]] = None) -> None:
    """Issue a PerformanceWarning with consistent formatting.
    
    Args:
        message: The primary warning message
        operation: The operation where the performance issue was detected
        issue: Description of the performance issue
        suggestion: Suggested action to improve performance
        details: Additional details about the warning
        context: Dictionary containing contextual information about the warning
    """
    import warnings
    warnings.warn(
        PerformanceWarning(message, operation, issue, suggestion, details, context)
    )


def warn_deprecation(message: str, 
                    feature: Optional[str] = None,
                    alternative: Optional[str] = None,
                    removal_version: Optional[str] = None,
                    details: Optional[str] = None, 
                    context: Optional[Dict[str, Any]] = None) -> None:
    """Issue a DeprecationWarning with consistent formatting.
    
    Args:
        message: The primary warning message
        feature: The deprecated feature
        alternative: The suggested alternative
        removal_version: The version when the feature will be removed
        details: Additional details about the warning
        context: Dictionary containing contextual information about the warning
    """
    import warnings
    warnings.warn(
        DeprecationWarning(message, feature, alternative, removal_version, details, context)
    )
