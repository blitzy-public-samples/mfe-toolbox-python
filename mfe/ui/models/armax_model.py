import logging
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Tuple, Callable, Awaitable, cast
import numpy as np
import pandas as pd
from pathlib import Path

# Import core components
from mfe.core.base import ModelBase, TimeSeriesModelBase
from mfe.core.parameters import ARMAParameters, TimeSeriesParameters
from mfe.core.exceptions import (
    MFEError, ModelSpecificationError, EstimationError, DataError,
    ForecastError, NotFittedError
)

# Set up module-level logger
logger = logging.getLogger("mfe.ui.models.armax_model")


@dataclass
class ARMAXModelParameters:
    """Parameter container for ARMAX model configuration.
    
    This dataclass encapsulates all parameters needed to configure an ARMAX model,
    providing type validation and a clean interface for parameter management.
    
    Attributes:
        ar_order: Order of the autoregressive component
        ma_order: Order of the moving average component
        include_constant: Whether to include a constant term in the model
        exog_variables: Names of exogenous variables to include
        estimation_method: Method to use for parameter estimation
        max_iterations: Maximum number of iterations for estimation
        convergence_tolerance: Tolerance for convergence criteria
        display_diagnostics: Whether to display diagnostic statistics
    """
    
    ar_order: int = 0
    ma_order: int = 0
    include_constant: bool = True
    exog_variables: List[str] = field(default_factory=list)
    estimation_method: str = "css-mle"
    max_iterations: int = 500
    convergence_tolerance: float = 1e-8
    display_diagnostics: bool = True
    
    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate parameter constraints.
        
        Raises:
            ModelSpecificationError: If parameters are invalid
        """
        # Validate AR and MA orders
        if self.ar_order < 0:
            raise ModelSpecificationError(
                "AR order must be non-negative",
                model_type="ARMAX",
                parameter="ar_order",
                valid_options="Non-negative integers"
            )
        
        if self.ma_order < 0:
            raise ModelSpecificationError(
                "MA order must be non-negative",
                model_type="ARMAX",
                parameter="ma_order",
                valid_options="Non-negative integers"
            )
        
        # Validate estimation method
        valid_methods = ["css", "mle", "css-mle"]
        if self.estimation_method not in valid_methods:
            raise ModelSpecificationError(
                f"Invalid estimation method: {self.estimation_method}",
                model_type="ARMAX",
                parameter="estimation_method",
                valid_options=valid_methods
            )
        
        # Validate iteration and tolerance parameters
        if self.max_iterations <= 0:
            raise ModelSpecificationError(
                "Maximum iterations must be positive",
                model_type="ARMAX",
                parameter="max_iterations",
                valid_options="Positive integers"
            )
        
        if self.convergence_tolerance <= 0:
            raise ModelSpecificationError(
                "Convergence tolerance must be positive",
                model_type="ARMAX",
                parameter="convergence_tolerance",
                valid_options="Positive values"
            )


@dataclass
class ARMAXModelResults:
    """Container for ARMAX model estimation results.
    
    This dataclass encapsulates all results from ARMAX model estimation,
    providing a structured container for accessing and displaying results.
    
    Attributes:
        ar_params: Estimated autoregressive parameters
        ma_params: Estimated moving average parameters
        exog_params: Estimated exogenous variable parameters
        constant: Estimated constant term
        sigma2: Estimated innovation variance
        log_likelihood: Log-likelihood of the fitted model
        aic: Akaike Information Criterion
        bic: Bayesian Information Criterion
        hqic: Hannan-Quinn Information Criterion
        fitted_values: Fitted values from the model
        residuals: Model residuals
        standardized_residuals: Standardized residuals
        convergence: Whether the estimation converged
        iterations: Number of iterations performed
        model_equation: String representation of the model equation
        parameter_table: DataFrame containing parameter estimates and statistics
        diagnostic_tests: Dictionary of diagnostic test results
    """
    
    ar_params: Optional[np.ndarray] = None
    ma_params: Optional[np.ndarray] = None
    exog_params: Optional[np.ndarray] = None
    constant: Optional[float] = None
    sigma2: Optional[float] = None
    log_likelihood: Optional[float] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    hqic: Optional[float] = None
    fitted_values: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None
    standardized_residuals: Optional[np.ndarray] = None
    convergence: bool = False
    iterations: int = 0
    model_equation: str = ""
    parameter_table: Optional[pd.DataFrame] = None
    diagnostic_tests: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of results
        """
        result_dict = {}
        for field_name, field_value in self.__dict__.items():
            # Handle NumPy arrays and DataFrames
            if isinstance(field_value, np.ndarray):
                result_dict[field_name] = field_value.tolist()
            elif isinstance(field_value, pd.DataFrame):
                result_dict[field_name] = field_value.to_dict()
            else:
                result_dict[field_name] = field_value
        
        return result_dict
    
    def summary(self) -> str:
        """Generate a text summary of the model results.
        
        Returns:
            str: Formatted summary of model results
        """
        if not self.convergence:
            return "Model estimation did not converge."
        
        header = "ARMAX Model Results\n"
        header += "=" * len(header) + "\n\n"
        
        # Model equation
        equation = f"Model Equation: {self.model_equation}\n\n"
        
        # Model fit statistics
        fit_stats = "Model Statistics:\n"
        if self.log_likelihood is not None:
            fit_stats += f"  Log-Likelihood: {self.log_likelihood:.4f}\n"
        if self.aic is not None:
            fit_stats += f"  AIC: {self.aic:.4f}\n"
        if self.bic is not None:
            fit_stats += f"  BIC: {self.bic:.4f}\n"
        if self.hqic is not None:
            fit_stats += f"  HQIC: {self.hqic:.4f}\n"
        if self.sigma2 is not None:
            fit_stats += f"  Residual Variance: {self.sigma2:.6f}\n"
        fit_stats += f"  Iterations: {self.iterations}\n\n"
        
        # Parameter estimates
        params = "Parameter Estimates:\n"
        if self.parameter_table is not None:
            params += self.parameter_table.to_string() + "\n\n"
        else:
            # Manually format parameters if table is not available
            if self.constant is not None:
                params += f"  Constant: {self.constant:.6f}\n"
            
            if self.ar_params is not None and len(self.ar_params) > 0:
                for i, param in enumerate(self.ar_params):
                    params += f"  AR{i+1}: {param:.6f}\n"
            
            if self.ma_params is not None and len(self.ma_params) > 0:
                for i, param in enumerate(self.ma_params):
                    params += f"  MA{i+1}: {param:.6f}\n"
            
            if self.exog_params is not None and len(self.exog_params) > 0:
                for i, param in enumerate(self.exog_params):
                    params += f"  Exog{i+1}: {param:.6f}\n"
            
            params += "\n"
        
        # Diagnostic tests
        diagnostics = "Diagnostic Tests:\n"
        if self.diagnostic_tests:
            for test_name, test_result in self.diagnostic_tests.items():
                if isinstance(test_result, dict):
                    diagnostics += f"  {test_name}:\n"
                    for key, value in test_result.items():
                        diagnostics += f"    {key}: {value}\n"
                else:
                    diagnostics += f"  {test_name}: {test_result}\n"
        else:
            diagnostics += "  No diagnostic tests performed.\n"
        
        return header + equation + fit_stats + params + diagnostics


class ARMAXModel:
    """Data model for the ARMAX GUI application.
    
    This class encapsulates the state and data structures needed for the ARMAX
    model estimation interface, providing a clean separation between UI and data logic.
    It manages model parameters, estimation results, data transformations, and state
    persistence for the ARMAX time series modeling component.
    
    Attributes:
        parameters: Model configuration parameters
        results: Model estimation results
        data: Time series data for modeling
        exog_data: Exogenous variables data
        is_fitted: Whether the model has been fitted
    """
    
    def __init__(self) -> None:
        """Initialize the ARMAX model with default parameters."""
        self.parameters = ARMAXModelParameters()
        self.results: Optional[ARMAXModelResults] = None
        self.data: Optional[np.ndarray] = None
        self.exog_data: Optional[np.ndarray] = None
        self.data_name: Optional[str] = None
        self.exog_names: List[str] = []
        self.is_fitted: bool = False
        self._cancel_requested: bool = False
        
        logger.debug("ARMAXModel initialized with default parameters")
    
    @property
    def ar_order(self) -> int:
        """Get the AR order.
        
        Returns:
            int: AR order
        """
        return self.parameters.ar_order
    
    @ar_order.setter
    def ar_order(self, value: int) -> None:
        """Set the AR order.
        
        Args:
            value: New AR order
            
        Raises:
            ModelSpecificationError: If the value is invalid
        """
        if value < 0:
            raise ModelSpecificationError(
                "AR order must be non-negative",
                model_type="ARMAX",
                parameter="ar_order",
                valid_options="Non-negative integers"
            )
        
        self.parameters.ar_order = value
        self.is_fitted = False  # Reset fitted state
        logger.debug(f"AR order set to {value}")
    
    @property
    def ma_order(self) -> int:
        """Get the MA order.
        
        Returns:
            int: MA order
        """
        return self.parameters.ma_order
    
    @ma_order.setter
    def ma_order(self, value: int) -> None:
        """Set the MA order.
        
        Args:
            value: New MA order
            
        Raises:
            ModelSpecificationError: If the value is invalid
        """
        if value < 0:
            raise ModelSpecificationError(
                "MA order must be non-negative",
                model_type="ARMAX",
                parameter="ma_order",
                valid_options="Non-negative integers"
            )
        
        self.parameters.ma_order = value
        self.is_fitted = False  # Reset fitted state
        logger.debug(f"MA order set to {value}")
    
    @property
    def include_constant(self) -> bool:
        """Get whether to include a constant term.
        
        Returns:
            bool: Whether to include a constant term
        """
        return self.parameters.include_constant
    
    @include_constant.setter
    def include_constant(self, value: bool) -> None:
        """Set whether to include a constant term.
        
        Args:
            value: Whether to include a constant term
        """
        self.parameters.include_constant = value
        self.is_fitted = False  # Reset fitted state
        logger.debug(f"Include constant set to {value}")
    
    @property
    def exog_variables(self) -> List[str]:
        """Get the list of exogenous variables.
        
        Returns:
            List[str]: Names of exogenous variables
        """
        return self.parameters.exog_variables
    
    @exog_variables.setter
    def exog_variables(self, value: List[str]) -> None:
        """Set the list of exogenous variables.
        
        Args:
            value: Names of exogenous variables
        """
        self.parameters.exog_variables = value
        self.is_fitted = False  # Reset fitted state
        logger.debug(f"Exogenous variables set to {value}")
    
    @property
    def estimation_method(self) -> str:
        """Get the estimation method.
        
        Returns:
            str: Estimation method
        """
        return self.parameters.estimation_method
    
    @estimation_method.setter
    def estimation_method(self, value: str) -> None:
        """Set the estimation method.
        
        Args:
            value: Estimation method
            
        Raises:
            ModelSpecificationError: If the value is invalid
        """
        valid_methods = ["css", "mle", "css-mle"]
        if value not in valid_methods:
            raise ModelSpecificationError(
                f"Invalid estimation method: {value}",
                model_type="ARMAX",
                parameter="estimation_method",
                valid_options=valid_methods
            )
        
        self.parameters.estimation_method = value
        self.is_fitted = False  # Reset fitted state
        logger.debug(f"Estimation method set to {value}")
    
    def set_data(self, data: Union[np.ndarray, pd.Series, pd.DataFrame], 
                data_name: Optional[str] = None) -> None:
        """Set the time series data for modeling.
        
        Args:
            data: Time series data
            data_name: Name of the data series
            
        Raises:
            DataError: If the data is invalid
        """
        # Convert data to NumPy array
        if isinstance(data, pd.Series):
            data_array = data.values
            if data_name is None:
                data_name = data.name
        elif isinstance(data, pd.DataFrame):
            if data.shape[1] != 1:
                raise DataError(
                    "DataFrame must have exactly one column for time series data",
                    data_name="data",
                    issue="Multiple columns in DataFrame"
                )
            data_array = data.iloc[:, 0].values
            if data_name is None:
                data_name = data.columns[0]
        elif isinstance(data, np.ndarray):
            if data.ndim != 1:
                raise DataError(
                    "NumPy array must be 1-dimensional for time series data",
                    data_name="data",
                    issue=f"Array has {data.ndim} dimensions"
                )
            data_array = data
        else:
            raise DataError(
                "Data must be a NumPy array, Pandas Series, or single-column DataFrame",
                data_name="data",
                issue=f"Invalid data type: {type(data)}"
            )
        
        # Validate data
        if len(data_array) < 3:
            raise DataError(
                "Time series data must have at least 3 observations",
                data_name="data",
                issue="Insufficient data points"
            )
        
        if np.isnan(data_array).any():
            raise DataError(
                "Time series data contains NaN values",
                data_name="data",
                issue="NaN values present"
            )
        
        if np.isinf(data_array).any():
            raise DataError(
                "Time series data contains infinite values",
                data_name="data",
                issue="Infinite values present"
            )
        
        self.data = data_array
        self.data_name = data_name if data_name is not None else "Y"
        self.is_fitted = False  # Reset fitted state
        logger.debug(f"Time series data set with {len(data_array)} observations")
    
    def set_exog_data(self, exog_data: Union[np.ndarray, pd.DataFrame], 
                     exog_names: Optional[List[str]] = None) -> None:
        """Set the exogenous variables data.
        
        Args:
            exog_data: Exogenous variables data
            exog_names: Names of exogenous variables
            
        Raises:
            DataError: If the data is invalid
        """
        # Convert exog_data to NumPy array
        if isinstance(exog_data, pd.DataFrame):
            exog_array = exog_data.values
            if exog_names is None:
                exog_names = exog_data.columns.tolist()
        elif isinstance(exog_data, np.ndarray):
            if exog_data.ndim != 2:
                raise DataError(
                    "Exogenous data must be 2-dimensional",
                    data_name="exog_data",
                    issue=f"Array has {exog_data.ndim} dimensions"
                )
            exog_array = exog_data
            if exog_names is None:
                # Generate default names
                exog_names = [f"X{i+1}" for i in range(exog_data.shape[1])]
        else:
            raise DataError(
                "Exogenous data must be a NumPy array or Pandas DataFrame",
                data_name="exog_data",
                issue=f"Invalid data type: {type(exog_data)}"
            )
        
        # Validate exog_data
        if self.data is not None and len(exog_array) != len(self.data):
            raise DataError(
                "Exogenous data length must match time series data length",
                data_name="exog_data",
                issue=f"Length mismatch: {len(exog_array)} vs {len(self.data)}"
            )
        
        if np.isnan(exog_array).any():
            raise DataError(
                "Exogenous data contains NaN values",
                data_name="exog_data",
                issue="NaN values present"
            )
        
        if np.isinf(exog_array).any():
            raise DataError(
                "Exogenous data contains infinite values",
                data_name="exog_data",
                issue="Infinite values present"
            )
        
        self.exog_data = exog_array
        self.exog_names = exog_names
        self.is_fitted = False  # Reset fitted state
        logger.debug(f"Exogenous data set with {exog_array.shape[1]} variables")
    
    def validate_model_specification(self) -> None:
        """Validate the model specification.
        
        Raises:
            ModelSpecificationError: If the model specification is invalid
            DataError: If required data is missing
        """
        # Check if data is available
        if self.data is None:
            raise DataError(
                "No time series data available for estimation",
                data_name="data",
                issue="Missing data"
            )
        
        # Check if model is specified
        if self.ar_order == 0 and self.ma_order == 0 and not self.include_constant and not self.exog_variables:
            raise ModelSpecificationError(
                "Model is not specified: no AR terms, MA terms, constant, or exogenous variables",
                model_type="ARMAX",
                issue="Empty model specification"
            )
        
        # Check if data is sufficient for the specified model
        min_obs = max(self.ar_order, self.ma_order) + 1
        if len(self.data) <= min_obs:
            raise DataError(
                f"Insufficient data for the specified model: need at least {min_obs+1} observations",
                data_name="data",
                issue="Insufficient data points"
            )
        
        # Check if exogenous variables are available if specified
        if self.exog_variables and self.exog_data is None:
            raise DataError(
                "Exogenous variables specified but no exogenous data provided",
                data_name="exog_data",
                issue="Missing exogenous data"
            )
        
        # Check if specified exogenous variables exist in the data
        if self.exog_variables and self.exog_data is not None:
            for var in self.exog_variables:
                if var not in self.exog_names:
                    raise ModelSpecificationError(
                        f"Specified exogenous variable '{var}' not found in exogenous data",
                        model_type="ARMAX",
                        parameter="exog_variables",
                        valid_options=self.exog_names
                    )
        
        logger.debug("Model specification validated successfully")
    
    def estimate(self) -> ARMAXModelResults:
        """Estimate the ARMAX model.
        
        Returns:
            ARMAXModelResults: Estimation results
            
        Raises:
            ModelSpecificationError: If the model specification is invalid
            DataError: If required data is missing
            EstimationError: If estimation fails
        """
        # This is a synchronous wrapper around the asynchronous estimation method
        try:
            # Create an event loop if one doesn't exist
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop in current thread, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the asynchronous estimation method
        return loop.run_until_complete(self.estimate_async())
    
    async def estimate_async(self, 
                           progress_callback: Optional[Callable[[float, str], Awaitable[None]]] = None) -> ARMAXModelResults:
        """Estimate the ARMAX model asynchronously.
        
        Args:
            progress_callback: Optional callback function for reporting progress
            
        Returns:
            ARMAXModelResults: Estimation results
            
        Raises:
            ModelSpecificationError: If the model specification is invalid
            DataError: If required data is missing
            EstimationError: If estimation fails
        """
        # Reset cancellation flag
        self._cancel_requested = False
        
        # Validate model specification
        self.validate_model_specification()
        
        # Report initial progress
        if progress_callback:
            await progress_callback(0.0, "Preparing model estimation...")
        
        try:
            # Import statsmodels here to avoid circular imports
            import statsmodels.api as sm
            
            # Prepare data
            endog = self.data
            
            # Prepare exogenous variables if specified
            exog = None
            if self.exog_variables and self.exog_data is not None:
                # Get indices of specified exogenous variables
                exog_indices = [self.exog_names.index(var) for var in self.exog_variables]
                exog = self.exog_data[:, exog_indices]
            
            # Report progress
            if progress_callback:
                await progress_callback(0.1, "Creating ARIMA model...")
            
            # Check for cancellation
            if self._cancel_requested:
                logger.debug("Estimation cancelled by user")
                raise EstimationError(
                    "Estimation cancelled by user",
                    model_type="ARMAX",
                    issue="User cancellation"
                )
            
            # Create and fit the model
            model = sm.tsa.ARIMA(
                endog=endog,
                exog=exog,
                order=(self.ar_order, 0, self.ma_order),  # (p, d, q)
                trend='c' if self.include_constant else 'n'
            )
            
            # Report progress
            if progress_callback:
                await progress_callback(0.2, "Estimating model parameters...")
            
            # Fit the model
            fit_result = model.fit(
                method=self.parameters.estimation_method,
                maxiter=self.parameters.max_iterations,
                disp=False
            )
            
            # Check for cancellation
            if self._cancel_requested:
                logger.debug("Estimation cancelled by user")
                raise EstimationError(
                    "Estimation cancelled by user",
                    model_type="ARMAX",
                    issue="User cancellation"
                )
            
            # Report progress
            if progress_callback:
                await progress_callback(0.7, "Computing diagnostics...")
            
            # Create results object
            results = ARMAXModelResults(
                convergence=True,
                iterations=fit_result.mle_retvals.get('iterations', 0)
            )
            
            # Extract parameters
            params = fit_result.params
            
            # Extract constant if included
            if self.include_constant:
                results.constant = params[0]
                params = params[1:]
            
            # Extract AR parameters
            if self.ar_order > 0:
                results.ar_params = params[:self.ar_order]
                params = params[self.ar_order:]
            else:
                results.ar_params = np.array([])
            
            # Extract MA parameters
            if self.ma_order > 0:
                results.ma_params = params[:self.ma_order]
                params = params[self.ma_order:]
            else:
                results.ma_params = np.array([])
            
            # Extract exogenous parameters
            if self.exog_variables:
                results.exog_params = params
            
            # Extract model statistics
            results.sigma2 = fit_result.sigma2
            results.log_likelihood = fit_result.llf
            results.aic = fit_result.aic
            results.bic = fit_result.bic
            results.hqic = fit_result.hqic
            
            # Extract fitted values and residuals
            results.fitted_values = fit_result.fittedvalues
            results.residuals = fit_result.resid
            
            # Compute standardized residuals
            if results.residuals is not None and results.sigma2 is not None:
                results.standardized_residuals = results.residuals / np.sqrt(results.sigma2)
            
            # Check for cancellation
            if self._cancel_requested:
                logger.debug("Estimation cancelled by user")
                raise EstimationError(
                    "Estimation cancelled by user",
                    model_type="ARMAX",
                    issue="User cancellation"
                )
            
            # Report progress
            if progress_callback:
                await progress_callback(0.8, "Creating parameter table...")
            
            # Create parameter table
            param_names = []
            param_values = []
            param_std_errs = []
            param_t_stats = []
            param_p_values = []
            
            # Add constant if included
            if self.include_constant:
                param_names.append("Constant")
                param_values.append(results.constant)
                
                # Get standard error, t-stat, and p-value from summary table
                summary_table = fit_result.summary().tables[1]
                const_row = summary_table.data[1]  # First row after header
                param_std_errs.append(float(const_row[1]))
                param_t_stats.append(float(const_row[2]))
                param_p_values.append(float(const_row[3]))
            
            # Add AR parameters
            if self.ar_order > 0:
                for i, param in enumerate(results.ar_params):
                    param_names.append(f"AR({i+1})")
                    param_values.append(param)
                    
                    # Get standard error, t-stat, and p-value from summary table
                    summary_table = fit_result.summary().tables[1]
                    row_idx = 1 + (1 if self.include_constant else 0) + i
                    row = summary_table.data[row_idx]
                    param_std_errs.append(float(row[1]))
                    param_t_stats.append(float(row[2]))
                    param_p_values.append(float(row[3]))
            
            # Add MA parameters
            if self.ma_order > 0:
                for i, param in enumerate(results.ma_params):
                    param_names.append(f"MA({i+1})")
                    param_values.append(param)
                    
                    # Get standard error, t-stat, and p-value from summary table
                    summary_table = fit_result.summary().tables[1]
                    row_idx = 1 + (1 if self.include_constant else 0) + self.ar_order + i
                    row = summary_table.data[row_idx]
                    param_std_errs.append(float(row[1]))
                    param_t_stats.append(float(row[2]))
                    param_p_values.append(float(row[3]))
            
            # Add exogenous parameters
            if self.exog_variables and results.exog_params is not None:
                for i, (var, param) in enumerate(zip(self.exog_variables, results.exog_params)):
                    param_names.append(var)
                    param_values.append(param)
                    
                    # Get standard error, t-stat, and p-value from summary table
                    summary_table = fit_result.summary().tables[1]
                    row_idx = 1 + (1 if self.include_constant else 0) + self.ar_order + self.ma_order + i
                    row = summary_table.data[row_idx]
                    param_std_errs.append(float(row[1]))
                    param_t_stats.append(float(row[2]))
                    param_p_values.append(float(row[3]))
            
            # Create parameter table as DataFrame
            results.parameter_table = pd.DataFrame({
                'Parameter': param_names,
                'Estimate': param_values,
                'Std. Error': param_std_errs,
                't-Statistic': param_t_stats,
                'p-Value': param_p_values
            })
            
            # Check for cancellation
            if self._cancel_requested:
                logger.debug("Estimation cancelled by user")
                raise EstimationError(
                    "Estimation cancelled by user",
                    model_type="ARMAX",
                    issue="User cancellation"
                )
            
            # Report progress
            if progress_callback:
                await progress_callback(0.9, "Computing diagnostic tests...")
            
            # Compute diagnostic tests
            results.diagnostic_tests = {}
            
            # Ljung-Box test for autocorrelation
            from scipy import stats
            max_lag = min(10, len(self.data) // 5)
            lb_test = sm.stats.acorr_ljungbox(results.residuals, lags=max_lag)
            results.diagnostic_tests['Ljung-Box Q Test'] = {
                'Q-Statistics': lb_test.iloc[:, 0].tolist(),
                'p-Values': lb_test.iloc[:, 1].tolist(),
                'Lags': list(range(1, max_lag + 1))
            }
            
            # Jarque-Bera test for normality
            jb_test = stats.jarque_bera(results.residuals)
            results.diagnostic_tests['Jarque-Bera Test'] = {
                'Statistic': jb_test[0],
                'p-Value': jb_test[1]
            }
            
            # ARCH LM test for heteroskedasticity
            arch_test = sm.stats.diagnostic.het_arch(results.residuals, nlags=5)
            results.diagnostic_tests['ARCH LM Test'] = {
                'Statistic': arch_test[0],
                'p-Value': arch_test[1],
                'F-Statistic': arch_test[2],
                'F p-Value': arch_test[3]
            }
            
            # Create model equation string
            results.model_equation = self._create_model_equation(results)
            
            # Report progress
            if progress_callback:
                await progress_callback(1.0, "Model estimation complete")
            
            # Store results and update fitted state
            self.results = results
            self.is_fitted = True
            
            logger.info("ARMAX model estimation completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error during ARMAX model estimation: {str(e)}")
            if isinstance(e, (ModelSpecificationError, DataError, EstimationError)):
                raise
            else:
                raise EstimationError(
                    f"Failed to estimate ARMAX model: {str(e)}",
                    model_type="ARMAX",
                    estimation_method=self.parameters.estimation_method,
                    issue=str(e),
                    details=str(e)
                ) from e
    
    def _create_model_equation(self, results: ARMAXModelResults) -> str:
        """Create a string representation of the model equation.
        
        Args:
            results: Model estimation results
            
        Returns:
            str: Model equation string
        """
        # Start with the dependent variable
        equation = f"{self.data_name}_t = "
        
        # Add constant term if included
        if self.include_constant and results.constant is not None:
            if results.constant >= 0:
                equation += f"{results.constant:.4f} "
            else:
                equation += f"({results.constant:.4f}) "
        
        # Add AR terms
        if self.ar_order > 0 and results.ar_params is not None:
            for i, param in enumerate(results.ar_params):
                if i > 0 or self.include_constant:
                    if param >= 0:
                        equation += f"+ {param:.4f} {self.data_name}_{{t-{i+1}}} "
                    else:
                        equation += f"- {abs(param):.4f} {self.data_name}_{{t-{i+1}}} "
                else:
                    if param >= 0:
                        equation += f"{param:.4f} {self.data_name}_{{t-{i+1}}} "
                    else:
                        equation += f"({param:.4f}) {self.data_name}_{{t-{i+1}}} "
        
        # Add exogenous variables
        if self.exog_variables and results.exog_params is not None:
            for i, (var, param) in enumerate(zip(self.exog_variables, results.exog_params)):
                if i > 0 or self.ar_order > 0 or self.include_constant:
                    if param >= 0:
                        equation += f"+ {param:.4f} {var}_t "
                    else:
                        equation += f"- {abs(param):.4f} {var}_t "
                else:
                    if param >= 0:
                        equation += f"{param:.4f} {var}_t "
                    else:
                        equation += f"({param:.4f}) {var}_t "
        
        # Add error term
        equation += "+ ε_t "
        
        # Add MA terms
        if self.ma_order > 0 and results.ma_params is not None:
            for i, param in enumerate(results.ma_params):
                if param >= 0:
                    equation += f"+ {param:.4f} ε_{{t-{i+1}}} "
                else:
                    equation += f"- {abs(param):.4f} ε_{{t-{i+1}}} "
        
        return equation.strip()
    
    def forecast(self, steps: int, 
                exog_forecast: Optional[np.ndarray] = None,
                confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate forecasts from the fitted model.
        
        Args:
            steps: Number of steps to forecast
            exog_forecast: Exogenous variables for the forecast period
            confidence_level: Confidence level for prediction intervals
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Forecasts, lower bounds, and upper bounds
            
        Raises:
            NotFittedError: If the model has not been fitted
            ForecastError: If forecasting fails
        """
        # This is a synchronous wrapper around the asynchronous forecasting method
        try:
            # Create an event loop if one doesn't exist
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop in current thread, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the asynchronous forecasting method
        return loop.run_until_complete(
            self.forecast_async(steps, exog_forecast, confidence_level)
        )
    
    async def forecast_async(self, steps: int, 
                           exog_forecast: Optional[np.ndarray] = None,
                           confidence_level: float = 0.95,
                           progress_callback: Optional[Callable[[float, str], Awaitable[None]]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate forecasts from the fitted model asynchronously.
        
        Args:
            steps: Number of steps to forecast
            exog_forecast: Exogenous variables for the forecast period
            confidence_level: Confidence level for prediction intervals
            progress_callback: Optional callback function for reporting progress
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Forecasts, lower bounds, and upper bounds
            
        Raises:
            NotFittedError: If the model has not been fitted
            ForecastError: If forecasting fails
        """
        # Reset cancellation flag
        self._cancel_requested = False
        
        # Check if model is fitted
        if not self.is_fitted or self.results is None:
            raise NotFittedError(
                "Model must be fitted before forecasting",
                model_type="ARMAX",
                operation="forecast"
            )
        
        # Report initial progress
        if progress_callback:
            await progress_callback(0.0, "Preparing forecasting...")
        
        try:
            # Import statsmodels here to avoid circular imports
            import statsmodels.api as sm
            
            # Validate steps
            if steps <= 0:
                raise ForecastError(
                    "Forecast horizon must be positive",
                    model_type="ARMAX",
                    horizon=steps,
                    issue="Invalid forecast horizon"
                )
            
            # Validate confidence level
            if confidence_level <= 0 or confidence_level >= 1:
                raise ForecastError(
                    "Confidence level must be between 0 and 1",
                    model_type="ARMAX",
                    issue="Invalid confidence level"
                )
            
            # Validate exogenous forecast data if needed
            if self.exog_variables and exog_forecast is None:
                raise ForecastError(
                    "Exogenous variables are used in the model but no forecast values provided",
                    model_type="ARMAX",
                    issue="Missing exogenous forecast data"
                )
            
            if exog_forecast is not None:
                if isinstance(exog_forecast, pd.DataFrame):
                    exog_forecast_array = exog_forecast.values
                elif isinstance(exog_forecast, np.ndarray):
                    exog_forecast_array = exog_forecast
                else:
                    raise ForecastError(
                        "Exogenous forecast data must be a NumPy array or Pandas DataFrame",
                        model_type="ARMAX",
                        issue=f"Invalid data type: {type(exog_forecast)}"
                    )
                
                # Check dimensions
                if exog_forecast_array.ndim != 2:
                    raise ForecastError(
                        "Exogenous forecast data must be 2-dimensional",
                        model_type="ARMAX",
                        issue=f"Array has {exog_forecast_array.ndim} dimensions"
                    )
                
                if exog_forecast_array.shape[0] != steps:
                    raise ForecastError(
                        f"Exogenous forecast data length ({exog_forecast_array.shape[0]}) "
                        f"must match forecast horizon ({steps})",
                        model_type="ARMAX",
                        issue="Length mismatch"
                    )
                
                if self.exog_data is not None and exog_forecast_array.shape[1] != self.exog_data.shape[1]:
                    raise ForecastError(
                        f"Exogenous forecast data width ({exog_forecast_array.shape[1]}) "
                        f"must match original exogenous data width ({self.exog_data.shape[1]})",
                        model_type="ARMAX",
                        issue="Width mismatch"
                    )
                
                # Filter exogenous forecast data if needed
                if self.exog_variables and self.exog_data is not None:
                    # Get indices of specified exogenous variables
                    exog_indices = [self.exog_names.index(var) for var in self.exog_variables]
                    exog_forecast_array = exog_forecast_array[:, exog_indices]
            
            # Report progress
            if progress_callback:
                await progress_callback(0.2, "Recreating ARIMA model...")
            
            # Check for cancellation
            if self._cancel_requested:
                logger.debug("Forecasting cancelled by user")
                raise ForecastError(
                    "Forecasting cancelled by user",
                    model_type="ARMAX",
                    issue="User cancellation"
                )
            
            # Recreate the model
            model = sm.tsa.ARIMA(
                endog=self.data,
                exog=self.exog_data[:, [self.exog_names.index(var) for var in self.exog_variables]] 
                      if self.exog_variables and self.exog_data is not None else None,
                order=(self.ar_order, 0, self.ma_order),  # (p, d, q)
                trend='c' if self.include_constant else 'n'
            )
            
            # Report progress
            if progress_callback:
                await progress_callback(0.4, "Fitting model for forecasting...")
            
            # Fit the model
            fit_result = model.fit(
                method=self.parameters.estimation_method,
                maxiter=self.parameters.max_iterations,
                disp=False
            )
            
            # Check for cancellation
            if self._cancel_requested:
                logger.debug("Forecasting cancelled by user")
                raise ForecastError(
                    "Forecasting cancelled by user",
                    model_type="ARMAX",
                    issue="User cancellation"
                )
            
            # Report progress
            if progress_callback:
                await progress_callback(0.7, "Generating forecasts...")
            
            # Generate forecasts
            alpha = 1 - confidence_level
            forecast_result = fit_result.get_forecast(
                steps=steps,
                exog=exog_forecast_array if self.exog_variables and exog_forecast is not None else None,
                alpha=alpha
            )
            
            # Extract forecast results
            forecasts = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int(alpha=alpha)
            lower_bounds = conf_int.iloc[:, 0].values
            upper_bounds = conf_int.iloc[:, 1].values
            
            # Report progress
            if progress_callback:
                await progress_callback(1.0, "Forecasting complete")
            
            logger.info(f"Generated {steps}-step forecasts successfully")
            return forecasts.values, lower_bounds, upper_bounds
            
        except Exception as e:
            logger.error(f"Error during ARMAX model forecasting: {str(e)}")
            if isinstance(e, (NotFittedError, ForecastError)):
                raise
            else:
                raise ForecastError(
                    f"Failed to generate forecasts: {str(e)}",
                    model_type="ARMAX",
                    horizon=steps,
                    issue=str(e),
                    details=str(e)
                ) from e
    
    def compute_acf_pacf(self, max_lag: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Compute autocorrelation and partial autocorrelation functions.
        
        Args:
            max_lag: Maximum lag to compute
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: ACF and PACF values
            
        Raises:
            DataError: If no data is available
        """
        if self.data is None:
            raise DataError(
                "No time series data available",
                data_name="data",
                issue="Missing data"
            )
        
        # Import statsmodels here to avoid circular imports
        import statsmodels.api as sm
        
        # Compute ACF and PACF
        acf = sm.tsa.acf(self.data, nlags=max_lag, fft=True)
        pacf = sm.tsa.pacf(self.data, nlags=max_lag)
        
        return acf, pacf
    
    def compute_residual_acf_pacf(self, max_lag: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Compute autocorrelation and partial autocorrelation functions of residuals.
        
        Args:
            max_lag: Maximum lag to compute
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: ACF and PACF values of residuals
            
        Raises:
            NotFittedError: If the model has not been fitted
        """
        if not self.is_fitted or self.results is None or self.results.residuals is None:
            raise NotFittedError(
                "Model must be fitted before computing residual diagnostics",
                model_type="ARMAX",
                operation="residual_acf_pacf"
            )
        
        # Import statsmodels here to avoid circular imports
        import statsmodels.api as sm
        
        # Compute ACF and PACF of residuals
        acf = sm.tsa.acf(self.results.residuals, nlags=max_lag, fft=True)
        pacf = sm.tsa.pacf(self.results.residuals, nlags=max_lag)
        
        return acf, pacf
    
    def cancel_operation(self) -> None:
        """Cancel the current asynchronous operation."""
        self._cancel_requested = True
        logger.debug("Operation cancellation requested")
    
    def reset(self) -> None:
        """Reset the model to its initial state."""
        self.parameters = ARMAXModelParameters()
        self.results = None
        self.data = None
        self.exog_data = None
        self.data_name = None
        self.exog_names = []
        self.is_fitted = False
        self._cancel_requested = False
        
        logger.debug("Model reset to initial state")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the model
        """
        model_dict = {
            "parameters": {
                "ar_order": self.parameters.ar_order,
                "ma_order": self.parameters.ma_order,
                "include_constant": self.parameters.include_constant,
                "exog_variables": self.parameters.exog_variables,
                "estimation_method": self.parameters.estimation_method,
                "max_iterations": self.parameters.max_iterations,
                "convergence_tolerance": self.parameters.convergence_tolerance,
                "display_diagnostics": self.parameters.display_diagnostics
            },
            "is_fitted": self.is_fitted,
            "data_name": self.data_name,
            "exog_names": self.exog_names
        }
        
        # Add results if available
        if self.results is not None:
            model_dict["results"] = self.results.to_dict()
        
        # Add data if available (convert to lists for JSON serialization)
        if self.data is not None:
            model_dict["data"] = self.data.tolist()
        
        if self.exog_data is not None:
            model_dict["exog_data"] = self.exog_data.tolist()
        
        return model_dict
    
    @classmethod
    def from_dict(cls, model_dict: Dict[str, Any]) -> 'ARMAXModel':
        """Create a model from a dictionary representation.
        
        Args:
            model_dict: Dictionary representation of the model
            
        Returns:
            ARMAXModel: Reconstructed model
        """
        model = cls()
        
        # Set parameters
        if "parameters" in model_dict:
            params = model_dict["parameters"]
            model.parameters = ARMAXModelParameters(
                ar_order=params.get("ar_order", 0),
                ma_order=params.get("ma_order", 0),
                include_constant=params.get("include_constant", True),
                exog_variables=params.get("exog_variables", []),
                estimation_method=params.get("estimation_method", "css-mle"),
                max_iterations=params.get("max_iterations", 500),
                convergence_tolerance=params.get("convergence_tolerance", 1e-8),
                display_diagnostics=params.get("display_diagnostics", True)
            )
        
        # Set data if available
        if "data" in model_dict:
            model.data = np.array(model_dict["data"])
        
        if "exog_data" in model_dict:
            model.exog_data = np.array(model_dict["exog_data"])
        
        # Set metadata
        model.data_name = model_dict.get("data_name")
        model.exog_names = model_dict.get("exog_names", [])
        model.is_fitted = model_dict.get("is_fitted", False)
        
        # Set results if available
        if "results" in model_dict and model_dict["is_fitted"]:
            results_dict = model_dict["results"]
            
            # Convert list representations back to NumPy arrays
            for key in ["ar_params", "ma_params", "exog_params", "fitted_values", 
                       "residuals", "standardized_residuals"]:
                if key in results_dict and results_dict[key] is not None:
                    results_dict[key] = np.array(results_dict[key])
            
            # Convert parameter_table back to DataFrame if available
            if "parameter_table" in results_dict and results_dict["parameter_table"] is not None:
                results_dict["parameter_table"] = pd.DataFrame.from_dict(results_dict["parameter_table"])
            
            # Create results object
            model.results = ARMAXModelResults(**results_dict)
        
        return model
