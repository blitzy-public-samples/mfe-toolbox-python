# mfe/models/time_series/base.py
"""
Abstract base classes for time series models in the MFE Toolbox.

This module defines the core abstract base classes that establish the contract
for all time series model implementations in the MFE Toolbox. These classes provide
the foundation for consistent interfaces across different time series model types,
ensuring that all implementations follow the same patterns for initialization,
fitting, forecasting, and result presentation.

The base classes implement shared functionality like parameter validation,
transformation between constrained and unconstrained parameter spaces, and
asynchronous processing for long-running operations.
"""

import abc
import asyncio
import logging
import warnings
from dataclasses import dataclass, field, asdict
from typing import (
    Any, Callable, Dict, Generic, List, Literal, Optional, Protocol, 
    Sequence, Tuple, Type, TypeVar, Union, cast, overload
)
import numpy as np
import pandas as pd
from scipy import stats, optimize
import statsmodels.api as sm

from mfe.core.base import ModelBase
from mfe.core.parameters import (
    ParameterBase, TimeSeriesParameters, ARMAParameters,
    validate_positive, validate_non_negative, validate_range,
    transform_positive, inverse_transform_positive
)
from mfe.core.exceptions import (
    ParameterError, DimensionError, ConvergenceError, NumericError,
    EstimationError, ForecastError, SimulationError, NotFittedError,
    warn_convergence, warn_numeric, warn_model
)

# Set up module-level logger
logger = logging.getLogger("mfe.models.time_series.base")

# Type variables for generic base classes
T = TypeVar('T', bound=TimeSeriesParameters)  # Type for time series parameters
D = TypeVar('D', bound=Union[np.ndarray, pd.Series, pd.DataFrame])  # Type for data


@dataclass
class TimeSeriesConfig:
    """Configuration options for time series models.
    
    This class provides a standardized way to configure time series models,
    including estimation methods, optimization settings, and other options.
    
    Attributes:
        method: Estimation method to use
        solver: Optimization solver to use
        max_iter: Maximum number of iterations for optimization
        tol: Convergence tolerance for optimization
        cov_type: Type of covariance matrix to compute
        use_numba: Whether to use Numba acceleration if available
        display_progress: Whether to display progress during estimation
    """
    
    method: str = "css"  # Conditional sum of squares
    solver: str = "BFGS"  # Optimization solver
    max_iter: int = 1000  # Maximum iterations
    tol: float = 1e-8  # Convergence tolerance
    cov_type: str = "robust"  # Covariance matrix type
    use_numba: bool = True  # Use Numba acceleration if available
    display_progress: bool = False  # Display progress during estimation
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Validate method
        valid_methods = ["css", "mle", "ols"]
        if self.method not in valid_methods:
            raise ParameterError(
                f"Invalid estimation method: {self.method}",
                param_name="method",
                param_value=self.method,
                constraint=f"Must be one of {valid_methods}"
            )
        
        # Validate solver
        valid_solvers = ["BFGS", "L-BFGS-B", "CG", "Newton-CG", "Powell", "TNC"]
        if self.solver not in valid_solvers:
            raise ParameterError(
                f"Invalid solver: {self.solver}",
                param_name="solver",
                param_value=self.solver,
                constraint=f"Must be one of {valid_solvers}"
            )
        
        # Validate max_iter
        if self.max_iter <= 0:
            raise ParameterError(
                f"Invalid max_iter: {self.max_iter}",
                param_name="max_iter",
                param_value=self.max_iter,
                constraint="Must be positive"
            )
        
        # Validate tol
        if self.tol <= 0:
            raise ParameterError(
                f"Invalid tol: {self.tol}",
                param_name="tol",
                param_value=self.tol,
                constraint="Must be positive"
            )
        
        # Validate cov_type
        valid_cov_types = ["robust", "standard", "hac", "none"]
        if self.cov_type not in valid_cov_types:
            raise ParameterError(
                f"Invalid cov_type: {self.cov_type}",
                param_name="cov_type",
                param_value=self.cov_type,
                constraint=f"Must be one of {valid_cov_types}"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of configuration
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TimeSeriesConfig':
        """Create configuration from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values
        
        Returns:
            TimeSeriesConfig: Configuration object
        """
        # Filter out unknown keys
        valid_keys = {f.name for f in field(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        
        return cls(**filtered_dict)


@dataclass
class TimeSeriesResult:
    """Base class for time series model estimation results.
    
    This class provides a standardized container for time series model results,
    including parameter estimates, standard errors, diagnostics, and fitted values.
    
    Attributes:
        model_name: Name of the model
        params: Estimated parameters
        std_errors: Standard errors of parameter estimates
        t_stats: t-statistics for parameter estimates
        p_values: p-values for parameter estimates
        log_likelihood: Log-likelihood of the model
        aic: Akaike Information Criterion
        bic: Bayesian Information Criterion
        hqic: Hannan-Quinn Information Criterion
        residuals: Model residuals
        fitted_values: Fitted values from the model
        convergence: Whether the optimization converged
        iterations: Number of iterations performed during optimization
        cov_type: Type of covariance matrix used
        cov_params: Covariance matrix of parameter estimates
        nobs: Number of observations
        df_model: Degrees of freedom used by the model
        df_resid: Residual degrees of freedom
    """
    
    model_name: str
    params: Dict[str, float]
    std_errors: Dict[str, float]
    t_stats: Dict[str, float]
    p_values: Dict[str, float]
    log_likelihood: float
    aic: float
    bic: float
    hqic: float
    residuals: np.ndarray
    fitted_values: np.ndarray
    convergence: bool = True
    iterations: int = 0
    cov_type: str = "robust"
    cov_params: Optional[np.ndarray] = None
    nobs: Optional[int] = None
    df_model: Optional[int] = None
    df_resid: Optional[int] = None
    
    def __post_init__(self) -> None:
        """Validate result object after initialization."""
        if not self.convergence:
            warn_convergence(
                f"Model {self.model_name} did not converge after {self.iterations} iterations.",
                iterations=self.iterations
            )
    
    def summary(self) -> str:
        """Generate a text summary of the model results.
        
        Returns:
            str: A formatted string containing the model results summary.
        """
        header = f"Model: {self.model_name}\n"
        header += "=" * (len(header) - 1) + "\n\n"
        
        # Add convergence information
        convergence_info = f"Convergence: {'Yes' if self.convergence else 'No'}\n"
        convergence_info += f"Iterations: {self.iterations}\n\n"
        
        # Add parameter table
        param_table = "Parameter Estimates:\n"
        param_table += "-" * 80 + "\n"
        param_table += f"{'Parameter':<15} {'Estimate':>12} {'Std. Error':>12} "
        param_table += f"{'t-stat':>12} {'p-value':>12} {'Significance':>10}\n"
        param_table += "-" * 80 + "\n"
        
        for param_name in self.params:
            estimate = self.params[param_name]
            std_err = self.std_errors.get(param_name, np.nan)
            t_stat = self.t_stats.get(param_name, np.nan)
            p_value = self.p_values.get(param_name, np.nan)
            
            # Determine significance
            if p_value < 0.01:
                sig = "***"
            elif p_value < 0.05:
                sig = "**"
            elif p_value < 0.1:
                sig = "*"
            else:
                sig = ""
            
            param_table += f"{param_name:<15} {estimate:>12.6f} {std_err:>12.6f} "
            param_table += f"{t_stat:>12.6f} {p_value:>12.6f} {sig:>10}\n"
        
        param_table += "-" * 80 + "\n"
        param_table += "Significance codes: *** 0.01, ** 0.05, * 0.1\n\n"
        
        # Add fit statistics
        fit_stats = "Model Statistics:\n"
        fit_stats += "-" * 40 + "\n"
        fit_stats += f"Log-Likelihood: {self.log_likelihood:.6f}\n"
        fit_stats += f"AIC: {self.aic:.6f}\n"
        fit_stats += f"BIC: {self.bic:.6f}\n"
        fit_stats += f"HQIC: {self.hqic:.6f}\n"
        
        if self.nobs is not None:
            fit_stats += f"Number of observations: {self.nobs}\n"
        if self.df_model is not None:
            fit_stats += f"Degrees of freedom (model): {self.df_model}\n"
        if self.df_resid is not None:
            fit_stats += f"Degrees of freedom (residuals): {self.df_resid}\n"
        
        fit_stats += "-" * 40 + "\n"
        
        return header + convergence_info + param_table + fit_stats
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result object to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the result object.
        """
        result_dict = asdict(self)
        
        # Convert NumPy arrays to lists for better serialization
        for key, value in result_dict.items():
            if isinstance(value, np.ndarray):
                result_dict[key] = value.tolist()
        
        return result_dict
    
    @classmethod
    def from_dict(cls, result_dict: Dict[str, Any]) -> 'TimeSeriesResult':
        """Create a result object from a dictionary.
        
        Args:
            result_dict: Dictionary containing result values
        
        Returns:
            TimeSeriesResult: Result object
        """
        # Convert lists back to NumPy arrays
        for key in ['residuals', 'fitted_values', 'cov_params']:
            if key in result_dict and result_dict[key] is not None:
                result_dict[key] = np.array(result_dict[key])
        
        return cls(**result_dict)


class TimeSeriesModel(ModelBase[T, TimeSeriesResult, Union[np.ndarray, pd.Series]]):
    """Abstract base class for time series models.
    
    This class defines the common interface that all time series model
    implementations must follow, establishing a consistent API across
    the entire time series module.
    
    Type Parameters:
        T: The parameter type for this model
    """
    
    def __init__(self, name: str = "TimeSeriesModel"):
        """Initialize the time series model.
        
        Args:
            name: A descriptive name for the model
        """
        super().__init__(name=name)
        self._config = TimeSeriesConfig()
        self._data: Optional[np.ndarray] = None
        self._index: Optional[Union[pd.DatetimeIndex, pd.Index]] = None
        self._residuals: Optional[np.ndarray] = None
        self._fitted_values: Optional[np.ndarray] = None
        self._params: Optional[T] = None
        self._cov_params: Optional[np.ndarray] = None
    
    @property
    def config(self) -> TimeSeriesConfig:
        """Get the model configuration.
        
        Returns:
            TimeSeriesConfig: The model configuration
        """
        return self._config
    
    @config.setter
    def config(self, config: TimeSeriesConfig) -> None:
        """Set the model configuration.
        
        Args:
            config: The model configuration
        """
        self._config = config
    
    @property
    def data(self) -> Optional[np.ndarray]:
        """Get the model data.
        
        Returns:
            Optional[np.ndarray]: The model data if set, None otherwise
        """
        return self._data
    
    @property
    def index(self) -> Optional[Union[pd.DatetimeIndex, pd.Index]]:
        """Get the data index.
        
        Returns:
            Optional[Union[pd.DatetimeIndex, pd.Index]]: The data index if available, None otherwise
        """
        return self._index
    
    @property
    def residuals(self) -> Optional[np.ndarray]:
        """Get the model residuals.
        
        Returns:
            Optional[np.ndarray]: The model residuals if fitted, None otherwise
        
        Raises:
            NotFittedError: If the model has not been fitted
        """
        if not self._fitted:
            raise NotFittedError(
                "Model has not been fitted. Call fit() first.",
                model_type=self._name,
                operation="residuals"
            )
        return self._residuals
    
    @property
    def fitted_values(self) -> Optional[np.ndarray]:
        """Get the fitted values.
        
        Returns:
            Optional[np.ndarray]: The fitted values if the model has been fitted,
                                 None otherwise
        
        Raises:
            NotFittedError: If the model has not been fitted
        """
        if not self._fitted:
            raise NotFittedError(
                "Model has not been fitted. Call fit() first.",
                model_type=self._name,
                operation="fitted_values"
            )
        return self._fitted_values
    
    @property
    def params(self) -> Optional[T]:
        """Get the model parameters.
        
        Returns:
            Optional[T]: The model parameters if fitted, None otherwise
        
        Raises:
            NotFittedError: If the model has not been fitted
        """
        if not self._fitted:
            raise NotFittedError(
                "Model has not been fitted. Call fit() first.",
                model_type=self._name,
                operation="params"
            )
        return self._params
    
    @property
    def cov_params(self) -> Optional[np.ndarray]:
        """Get the parameter covariance matrix.
        
        Returns:
            Optional[np.ndarray]: The parameter covariance matrix if fitted, None otherwise
        
        Raises:
            NotFittedError: If the model has not been fitted
        """
        if not self._fitted:
            raise NotFittedError(
                "Model has not been fitted. Call fit() first.",
                model_type=self._name,
                operation="cov_params"
            )
        return self._cov_params
    
    def validate_data(self, data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Validate the input data for time series model fitting.
        
        Args:
            data: The data to validate
        
        Returns:
            np.ndarray: The validated data as a NumPy array
        
        Raises:
            TypeError: If the data has an incorrect type
            ValueError: If the data is invalid
        """
        # Store the index if available
        if isinstance(data, pd.Series):
            self._index = data.index
            data_array = data.values
        elif isinstance(data, np.ndarray):
            data_array = data
            self._index = None
        else:
            raise TypeError(
                f"Data must be a NumPy array or Pandas Series, got {type(data).__name__}"
            )
        
        # Check dimensions
        if data_array.ndim != 1:
            raise DimensionError(
                f"Data must be 1-dimensional, got {data_array.ndim} dimensions",
                array_name="data",
                expected_shape="(n,)",
                actual_shape=data_array.shape
            )
        
        # Check length
        if len(data_array) < 10:  # Arbitrary minimum length
            raise ValueError(
                f"Data length must be at least 10, got {len(data_array)}"
            )
        
        # Check for NaN and Inf values
        if np.isnan(data_array).any():
            raise ValueError("Data contains NaN values")
        
        if np.isinf(data_array).any():
            raise ValueError("Data contains infinite values")
        
        return data_array
    
    @abc.abstractmethod
    def fit(self, 
            data: Union[np.ndarray, pd.Series], 
            **kwargs: Any) -> TimeSeriesResult:
        """Fit the model to the provided data.
        
        This method must be implemented by all subclasses to estimate model
        parameters from the provided data.
        
        Args:
            data: The data to fit the model to
            **kwargs: Additional keyword arguments for model fitting
        
        Returns:
            TimeSeriesResult: The model estimation results
        
        Raises:
            ValueError: If the data is invalid
            EstimationError: If the model estimation fails
        """
        pass
    
    async def fit_async(self, 
                       data: Union[np.ndarray, pd.Series], 
                       **kwargs: Any) -> TimeSeriesResult:
        """Asynchronously fit the model to the provided data.
        
        This method provides an asynchronous interface to the fit method,
        allowing for non-blocking model estimation in UI contexts.
        
        Args:
            data: The data to fit the model to
            **kwargs: Additional keyword arguments for model fitting
        
        Returns:
            TimeSeriesResult: The model estimation results
        
        Raises:
            ValueError: If the data is invalid
            EstimationError: If the model estimation fails
        """
        # Create a coroutine that runs the synchronous fit method in a thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: self.fit(data, **kwargs)
        )
        return result
    
    @abc.abstractmethod
    def forecast(self, 
                steps: int, 
                exog: Optional[np.ndarray] = None,
                confidence_level: float = 0.95,
                **kwargs: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate forecasts from the fitted model.
        
        This method must be implemented by all subclasses to generate forecasts
        based on the fitted model parameters.
        
        Args:
            steps: Number of steps to forecast
            exog: Exogenous variables for the forecast period
            confidence_level: Confidence level for prediction intervals
            **kwargs: Additional keyword arguments for forecasting
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Forecasts, lower bounds, and upper bounds
        
        Raises:
            NotFittedError: If the model has not been fitted
            ValueError: If the forecast parameters are invalid
            ForecastError: If the forecasting fails
        """
        pass
    
    async def forecast_async(self, 
                           steps: int, 
                           exog: Optional[np.ndarray] = None,
                           confidence_level: float = 0.95,
                           **kwargs: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Asynchronously generate forecasts from the fitted model.
        
        This method provides an asynchronous interface to the forecast method,
        allowing for non-blocking forecasting in UI contexts.
        
        Args:
            steps: Number of steps to forecast
            exog: Exogenous variables for the forecast period
            confidence_level: Confidence level for prediction intervals
            **kwargs: Additional keyword arguments for forecasting
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Forecasts, lower bounds, and upper bounds
        
        Raises:
            NotFittedError: If the model has not been fitted
            ValueError: If the forecast parameters are invalid
            ForecastError: If the forecasting fails
        """
        # Create a coroutine that runs the synchronous forecast method in a thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: self.forecast(steps, exog, confidence_level, **kwargs)
        )
        return result
    
    @abc.abstractmethod
    def simulate(self, 
                n_periods: int, 
                burn: int = 0, 
                initial_values: Optional[np.ndarray] = None,
                random_state: Optional[Union[int, np.random.Generator]] = None,
                **kwargs: Any) -> np.ndarray:
        """Simulate data from the model.
        
        This method must be implemented by all subclasses to generate simulated
        data based on the model parameters.
        
        Args:
            n_periods: Number of periods to simulate
            burn: Number of initial observations to discard
            initial_values: Initial values for the simulation
            random_state: Random number generator or seed
            **kwargs: Additional keyword arguments for simulation
        
        Returns:
            np.ndarray: Simulated data
        
        Raises:
            NotFittedError: If the model has not been fitted
            ValueError: If the simulation parameters are invalid
            SimulationError: If the simulation fails
        """
        pass
    
    async def simulate_async(self, 
                           n_periods: int, 
                           burn: int = 0, 
                           initial_values: Optional[np.ndarray] = None,
                           random_state: Optional[Union[int, np.random.Generator]] = None,
                           **kwargs: Any) -> np.ndarray:
        """Asynchronously simulate data from the model.
        
        This method provides an asynchronous interface to the simulate method,
        allowing for non-blocking simulation in UI contexts.
        
        Args:
            n_periods: Number of periods to simulate
            burn: Number of initial observations to discard
            initial_values: Initial values for the simulation
            random_state: Random number generator or seed
            **kwargs: Additional keyword arguments for simulation
        
        Returns:
            np.ndarray: Simulated data
        
        Raises:
            NotFittedError: If the model has not been fitted
            ValueError: If the simulation parameters are invalid
            SimulationError: If the simulation fails
        """
        # Create a coroutine that runs the synchronous simulate method in a thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: self.simulate(n_periods, burn, initial_values, random_state, **kwargs)
        )
        return result
    
    @abc.abstractmethod
    def loglikelihood(self, 
                     params: T, 
                     data: np.ndarray, 
                     **kwargs: Any) -> float:
        """Compute the log-likelihood of the model.
        
        This method must be implemented by all subclasses to compute the
        log-likelihood of the model given parameters and data.
        
        Args:
            params: Model parameters
            data: Input data
            **kwargs: Additional keyword arguments for log-likelihood computation
        
        Returns:
            float: Log-likelihood value
        
        Raises:
            ValueError: If the parameters or data are invalid
            NumericError: If the log-likelihood computation fails
        """
        pass
    
    def information_criteria(self, 
                            loglikelihood: float, 
                            nobs: int, 
                            k: int) -> Tuple[float, float, float]:
        """Compute information criteria for model selection.
        
        Args:
            loglikelihood: Log-likelihood of the model
            nobs: Number of observations
            k: Number of parameters
        
        Returns:
            Tuple[float, float, float]: AIC, BIC, and HQIC values
        """
        aic = -2 * loglikelihood + 2 * k
        bic = -2 * loglikelihood + k * np.log(nobs)
        hqic = -2 * loglikelihood + 2 * k * np.log(np.log(nobs))
        
        return aic, bic, hqic
    
    def _compute_standard_errors(self, 
                               params: T, 
                               data: np.ndarray, 
                               hessian: Optional[np.ndarray] = None,
                               **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Compute standard errors for parameter estimates.
        
        Args:
            params: Model parameters
            data: Input data
            hessian: Hessian matrix of the log-likelihood function
            **kwargs: Additional keyword arguments for standard error computation
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Standard errors and covariance matrix
        
        Raises:
            NumericError: If the standard error computation fails
        """
        if hessian is None:
            # Compute Hessian numerically
            param_array = params.to_array()
            
            def neg_loglikelihood(p: np.ndarray) -> float:
                try:
                    param_obj = params.__class__.from_array(p)
                    return -self.loglikelihood(param_obj, data, **kwargs)
                except Exception as e:
                    logger.warning(f"Error in negative log-likelihood: {e}")
                    return np.inf
            
            try:
                # Compute Hessian using finite differences
                hessian = optimize.approx_fprime(
                    param_array, 
                    lambda p: optimize.approx_fprime(
                        p, neg_loglikelihood, 1e-6
                    ),
                    1e-6
                )
            except Exception as e:
                raise NumericError(
                    f"Failed to compute Hessian matrix: {e}",
                    operation="Hessian computation",
                    error_type="numerical differentiation"
                )
        
        try:
            # Compute covariance matrix as inverse of Hessian
            cov_params = np.linalg.inv(hessian)
            
            # Check for positive definiteness
            if np.any(np.diag(cov_params) <= 0):
                warn_numeric(
                    "Covariance matrix has non-positive diagonal elements",
                    operation="Standard error computation",
                    issue="Non-positive definite covariance matrix"
                )
                # Use absolute values for standard errors
                std_errors = np.sqrt(np.abs(np.diag(cov_params)))
            else:
                std_errors = np.sqrt(np.diag(cov_params))
            
            return std_errors, cov_params
        except np.linalg.LinAlgError as e:
            raise NumericError(
                f"Failed to invert Hessian matrix: {e}",
                operation="Standard error computation",
                error_type="matrix inversion"
            )
    
    def _compute_robust_standard_errors(self, 
                                      params: T, 
                                      data: np.ndarray, 
                                      hessian: Optional[np.ndarray] = None,
                                      **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Compute robust standard errors for parameter estimates.
        
        Args:
            params: Model parameters
            data: Input data
            hessian: Hessian matrix of the log-likelihood function
            **kwargs: Additional keyword arguments for standard error computation
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Robust standard errors and covariance matrix
        
        Raises:
            NumericError: If the standard error computation fails
        """
        # First compute standard errors and covariance matrix
        std_errors, cov_params = self._compute_standard_errors(
            params, data, hessian, **kwargs
        )
        
        # For robust standard errors, we need to compute the outer product
        # of the gradient (OPG) and sandwich it with the inverse Hessian
        param_array = params.to_array()
        n_params = len(param_array)
        nobs = len(data)
        
        try:
            # Compute scores for each observation
            scores = np.zeros((nobs, n_params))
            
            # This is a simplified approach - in practice, you would compute
            # the gradient of the log-likelihood for each observation
            # Here we're using numerical differentiation as a placeholder
            for i in range(nobs):
                obs_data = data[i:i+1]  # Single observation
                
                def neg_loglikelihood(p: np.ndarray) -> float:
                    try:
                        param_obj = params.__class__.from_array(p)
                        return -self.loglikelihood(param_obj, obs_data, **kwargs)
                    except Exception as e:
                        logger.warning(f"Error in negative log-likelihood: {e}")
                        return np.inf
                
                scores[i, :] = optimize.approx_fprime(
                    param_array, neg_loglikelihood, 1e-6
                )
            
            # Compute outer product of gradient
            opg = np.zeros((n_params, n_params))
            for i in range(nobs):
                opg += np.outer(scores[i, :], scores[i, :])
            
            # Compute sandwich estimator: H^(-1) * OPG * H^(-1)
            inv_hessian = np.linalg.inv(hessian)
            robust_cov = inv_hessian @ opg @ inv_hessian
            
            # Compute robust standard errors
            robust_std_errors = np.sqrt(np.diag(robust_cov))
            
            return robust_std_errors, robust_cov
        except Exception as e:
            warn_numeric(
                f"Failed to compute robust standard errors: {e}",
                operation="Robust standard error computation",
                issue="Falling back to standard errors"
            )
            return std_errors, cov_params
    
    def _create_result_object(self, 
                            params: T, 
                            data: np.ndarray, 
                            residuals: np.ndarray,
                            fitted_values: np.ndarray,
                            loglikelihood: float,
                            iterations: int,
                            convergence: bool,
                            cov_params: np.ndarray,
                            std_errors: np.ndarray,
                            **kwargs: Any) -> TimeSeriesResult:
        """Create a result object from model estimation results.
        
        Args:
            params: Estimated parameters
            data: Input data
            residuals: Model residuals
            fitted_values: Fitted values
            loglikelihood: Log-likelihood of the model
            iterations: Number of iterations performed during optimization
            convergence: Whether the optimization converged
            cov_params: Covariance matrix of parameter estimates
            std_errors: Standard errors of parameter estimates
            **kwargs: Additional keyword arguments for result creation
        
        Returns:
            TimeSeriesResult: Model estimation results
        """
        # Convert parameters to dictionary
        param_dict = params.to_dict()
        
        # Create parameter-related dictionaries
        param_names = list(param_dict.keys())
        param_values = list(param_dict.values())
        
        params_dict = {name: value for name, value in zip(param_names, param_values)}
        std_errors_dict = {name: value for name, value in zip(param_names, std_errors)}
        
        # Compute t-statistics and p-values
        t_stats_dict = {}
        p_values_dict = {}
        
        for name, value in params_dict.items():
            std_err = std_errors_dict.get(name, np.nan)
            if std_err > 0:
                t_stat = value / std_err
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(data) - len(param_names)))
            else:
                t_stat = np.nan
                p_value = np.nan
            
            t_stats_dict[name] = t_stat
            p_values_dict[name] = p_value
        
        # Compute information criteria
        nobs = len(data)
        k = len(param_names)
        aic, bic, hqic = self.information_criteria(loglikelihood, nobs, k)
        
        # Create result object
        result = TimeSeriesResult(
            model_name=self._name,
            params=params_dict,
            std_errors=std_errors_dict,
            t_stats=t_stats_dict,
            p_values=p_values_dict,
            log_likelihood=loglikelihood,
            aic=aic,
            bic=bic,
            hqic=hqic,
            residuals=residuals,
            fitted_values=fitted_values,
            convergence=convergence,
            iterations=iterations,
            cov_type=self._config.cov_type,
            cov_params=cov_params,
            nobs=nobs,
            df_model=k,
            df_resid=nobs - k
        )
        
        return result


class ARMAModel(TimeSeriesModel[ARMAParameters]):
    """Base class for ARMA (AutoRegressive Moving Average) models.
    
    This class provides the foundation for ARMA and ARMAX models, implementing
    common functionality for parameter validation, estimation, forecasting,
    and simulation.
    
    Attributes:
        ar_order: Order of the autoregressive component
        ma_order: Order of the moving average component
        include_constant: Whether to include a constant term in the model
    """
    
    def __init__(self, 
                ar_order: int = 0, 
                ma_order: int = 0, 
                include_constant: bool = True,
                name: str = "ARMA"):
        """Initialize the ARMA model.
        
        Args:
            ar_order: Order of the autoregressive component
            ma_order: Order of the moving average component
            include_constant: Whether to include a constant term in the model
            name: A descriptive name for the model
        """
        super().__init__(name=name)
        self.ar_order = ar_order
        self.ma_order = ma_order
        self.include_constant = include_constant
        
        # Validate model orders
        self._validate_model_orders()
    
    def _validate_model_orders(self) -> None:
        """Validate the AR and MA orders.
        
        Raises:
            ParameterError: If the orders are invalid
        """
        if not isinstance(self.ar_order, int) or self.ar_order < 0:
            raise ParameterError(
                f"AR order must be a non-negative integer, got {self.ar_order}",
                param_name="ar_order",
                param_value=self.ar_order,
                constraint="Must be a non-negative integer"
            )
        
        if not isinstance(self.ma_order, int) or self.ma_order < 0:
            raise ParameterError(
                f"MA order must be a non-negative integer, got {self.ma_order}",
                param_name="ma_order",
                param_value=self.ma_order,
                constraint="Must be a non-negative integer"
            )
        
        if self.ar_order == 0 and self.ma_order == 0:
            raise ParameterError(
                "At least one of AR or MA order must be positive",
                param_name="ar_order, ma_order",
                constraint="At least one must be positive"
            )
    
    def fit(self, 
           data: Union[np.ndarray, pd.Series], 
           **kwargs: Any) -> TimeSeriesResult:
        """Fit the ARMA model to the provided data.
        
        Args:
            data: The data to fit the model to
            **kwargs: Additional keyword arguments for model fitting
        
        Returns:
            TimeSeriesResult: The model estimation results
        
        Raises:
            ValueError: If the data is invalid
            EstimationError: If the model estimation fails
        """
        # Validate and prepare data
        data_array = self.validate_data(data)
        self._data = data_array
        
        # Update config with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
        
        # Use statsmodels for ARMA estimation
        try:
            # Create statsmodels ARIMA model
            sm_model = sm.tsa.ARIMA(
                data_array,
                order=(self.ar_order, 0, self.ma_order),
                trend='c' if self.include_constant else 'n'
            )
            
            # Fit the model
            sm_result = sm_model.fit(
                method=self._config.method,
                maxiter=self._config.max_iter,
                disp=self._config.display_progress
            )
            
            # Extract results
            params_dict = sm_result.params.to_dict()
            
            # Create ARMAParameters object
            ar_params = np.zeros(self.ar_order)
            ma_params = np.zeros(self.ma_order)
            constant = 0.0
            sigma2 = sm_result.sigma2
            
            # Extract AR parameters
            for i in range(self.ar_order):
                param_name = f'ar.L{i+1}'
                if param_name in params_dict:
                    ar_params[i] = params_dict[param_name]
            
            # Extract MA parameters
            for i in range(self.ma_order):
                param_name = f'ma.L{i+1}'
                if param_name in params_dict:
                    ma_params[i] = params_dict[param_name]
            
            # Extract constant
            if self.include_constant and 'const' in params_dict:
                constant = params_dict['const']
            
            # Create parameter object
            params = ARMAParameters(
                ar_params=ar_params,
                ma_params=ma_params,
                sigma2=sigma2,
                constant=constant
            )
            
            # Store model attributes
            self._params = params
            self._residuals = sm_result.resid
            self._fitted_values = data_array - sm_result.resid
            self._cov_params = sm_result.cov_params()
            self._fitted = True
            
            # Create result object
            std_errors = np.sqrt(np.diag(self._cov_params))
            
            # Map standard errors to parameter names
            param_names = []
            if self.include_constant:
                param_names.append('constant')
            for i in range(self.ar_order):
                param_names.append(f'ar{i+1}')
            for i in range(self.ma_order):
                param_names.append(f'ma{i+1}')
            param_names.append('sigma2')
            
            # Create result object
            result = self._create_result_object(
                params=params,
                data=data_array,
                residuals=self._residuals,
                fitted_values=self._fitted_values,
                loglikelihood=sm_result.llf,
                iterations=sm_result.mle_retvals.get('iterations', 0),
                convergence=sm_result.mle_retvals.get('success', True),
                cov_params=self._cov_params,
                std_errors=std_errors
            )
            
            self._results = result
            return result
            
        except Exception as e:
            raise EstimationError(
                f"ARMA model estimation failed: {e}",
                model_type=self._name,
                estimation_method=self._config.method,
                details=str(e)
            )
    
    def forecast(self, 
                steps: int, 
                exog: Optional[np.ndarray] = None,
                confidence_level: float = 0.95,
                **kwargs: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate forecasts from the fitted ARMA model.
        
        Args:
            steps: Number of steps to forecast
            exog: Exogenous variables for the forecast period (not used in ARMA)
            confidence_level: Confidence level for prediction intervals
            **kwargs: Additional keyword arguments for forecasting
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Forecasts, lower bounds, and upper bounds
        
        Raises:
            NotFittedError: If the model has not been fitted
            ValueError: If the forecast parameters are invalid
            ForecastError: If the forecasting fails
        """
        if not self._fitted:
            raise NotFittedError(
                "Model has not been fitted. Call fit() first.",
                model_type=self._name,
                operation="forecast"
            )
        
        if steps <= 0:
            raise ValueError(f"Steps must be positive, got {steps}")
        
        if not 0 < confidence_level < 1:
            raise ValueError(
                f"Confidence level must be between 0 and 1, got {confidence_level}"
            )
        
        try:
            # Use statsmodels for forecasting
            sm_model = sm.tsa.ARIMA(
                self._data,
                order=(self.ar_order, 0, self.ma_order),
                trend='c' if self.include_constant else 'n'
            )
            
            # Refit with the same parameters
            sm_result = sm_model.filter(self._params.to_array())
            
            # Generate forecasts
            forecast_result = sm_result.get_forecast(steps=steps)
            
            # Extract forecasts and confidence intervals
            forecasts = forecast_result.predicted_mean
            
            # Compute confidence intervals
            alpha = 1 - confidence_level
            ci = forecast_result.conf_int(alpha=alpha)
            lower_bounds = ci.iloc[:, 0].values
            upper_bounds = ci.iloc[:, 1].values
            
            return forecasts.values, lower_bounds, upper_bounds
            
        except Exception as e:
            raise ForecastError(
                f"ARMA forecasting failed: {e}",
                model_type=self._name,
                horizon=steps,
                details=str(e)
            )
    
    def simulate(self, 
                n_periods: int, 
                burn: int = 0, 
                initial_values: Optional[np.ndarray] = None,
                random_state: Optional[Union[int, np.random.Generator]] = None,
                **kwargs: Any) -> np.ndarray:
        """Simulate data from the ARMA model.
        
        Args:
            n_periods: Number of periods to simulate
            burn: Number of initial observations to discard
            initial_values: Initial values for the simulation
            random_state: Random number generator or seed
            **kwargs: Additional keyword arguments for simulation
        
        Returns:
            np.ndarray: Simulated data
        
        Raises:
            NotFittedError: If the model has not been fitted
            ValueError: If the simulation parameters are invalid
            SimulationError: If the simulation fails
        """
        if not self._fitted:
            raise NotFittedError(
                "Model has not been fitted. Call fit() first.",
                model_type=self._name,
                operation="simulate"
            )
        
        if n_periods <= 0:
            raise ValueError(f"n_periods must be positive, got {n_periods}")
        
        if burn < 0:
            raise ValueError(f"burn must be non-negative, got {burn}")
        
        # Set up random number generator
        if isinstance(random_state, np.random.Generator):
            rng = random_state
        else:
            rng = np.random.default_rng(random_state)
        
        try:
            # Extract parameters
            ar_params = self._params.ar_params
            ma_params = self._params.ma_params
            sigma2 = self._params.sigma2
            constant = self._params.constant
            
            # Total periods to simulate (including burn-in)
            total_periods = n_periods + burn
            
            # Maximum lag order
            max_lag = max(self.ar_order, self.ma_order)
            
            # Initialize arrays
            simulated = np.zeros(total_periods + max_lag)
            errors = np.zeros(total_periods + max_lag)
            
            # Set initial values if provided
            if initial_values is not None:
                if len(initial_values) < max_lag:
                    raise ValueError(
                        f"initial_values must have length at least {max_lag}, "
                        f"got {len(initial_values)}"
                    )
                simulated[:max_lag] = initial_values[:max_lag]
            
            # Generate random errors
            errors[max_lag:] = rng.normal(0, np.sqrt(sigma2), total_periods)
            
            # Generate simulated data
            for t in range(max_lag, total_periods + max_lag):
                # Add constant term
                simulated[t] = constant
                
                # Add AR terms
                for i in range(self.ar_order):
                    if t - i - 1 >= 0:
                        simulated[t] += ar_params[i] * simulated[t - i - 1]
                
                # Add MA terms and current error
                simulated[t] += errors[t]
                for i in range(self.ma_order):
                    if t - i - 1 >= 0:
                        simulated[t] += ma_params[i] * errors[t - i - 1]
            
            # Return simulated data (excluding burn-in)
            return simulated[max_lag + burn:]
            
        except Exception as e:
            raise SimulationError(
                f"ARMA simulation failed: {e}",
                model_type=self._name,
                n_periods=n_periods,
                details=str(e)
            )
    
    def loglikelihood(self, 
                     params: ARMAParameters, 
                     data: np.ndarray, 
                     **kwargs: Any) -> float:
        """Compute the log-likelihood of the ARMA model.
        
        Args:
            params: Model parameters
            data: Input data
            **kwargs: Additional keyword arguments for log-likelihood computation
        
        Returns:
            float: Log-likelihood value
        
        Raises:
            ValueError: If the parameters or data are invalid
            NumericError: If the log-likelihood computation fails
        """
        try:
            # Validate parameters
            params.validate()
            
            # Extract parameters
            ar_params = params.ar_params
            ma_params = params.ma_params
            sigma2 = params.sigma2
            constant = params.constant
            
            # Use statsmodels for log-likelihood computation
            sm_model = sm.tsa.ARIMA(
                data,
                order=(self.ar_order, 0, self.ma_order),
                trend='c' if self.include_constant else 'n'
            )
            
            # Convert parameters to statsmodels format
            sm_params = []
            if self.include_constant:
                sm_params.append(constant)
            sm_params.extend(ar_params)
            sm_params.extend(ma_params)
            sm_params.append(np.sqrt(sigma2))  # statsmodels uses standard deviation
            
            # Compute log-likelihood
            llf = sm_model.loglike(sm_params)
            
            return llf
            
        except Exception as e:
            raise NumericError(
                f"ARMA log-likelihood computation failed: {e}",
                operation="log-likelihood",
                error_type="computation",
                details=str(e)
            )
    
    def summary(self) -> str:
        """Generate a text summary of the ARMA model.
        
        Returns:
            str: A formatted string containing the model summary
        
        Raises:
            NotFittedError: If the model has not been fitted
        """
        if not self._fitted:
            return f"ARMA({self.ar_order}, {self.ma_order}) Model (not fitted)"
        
        if self._results is None:
            return f"ARMA({self.ar_order}, {self.ma_order}) Model (fitted, but no results available)"
        
        # Use the result object's summary method
        return self._results.summary()


class ARMAXModel(ARMAModel):
    """ARMAX (AutoRegressive Moving Average with eXogenous variables) model.
    
    This class extends the ARMA model to include exogenous variables.
    
    Attributes:
        ar_order: Order of the autoregressive component
        ma_order: Order of the moving average component
        include_constant: Whether to include a constant term in the model
        exog: Exogenous variables
    """
    
    def __init__(self, 
                ar_order: int = 0, 
                ma_order: int = 0, 
                include_constant: bool = True,
                name: str = "ARMAX"):
        """Initialize the ARMAX model.
        
        Args:
            ar_order: Order of the autoregressive component
            ma_order: Order of the moving average component
            include_constant: Whether to include a constant term in the model
            name: A descriptive name for the model
        """
        super().__init__(
            ar_order=ar_order,
            ma_order=ma_order,
            include_constant=include_constant,
            name=name
        )
        self._exog: Optional[np.ndarray] = None
        self._exog_names: Optional[List[str]] = None
    
    def fit(self, 
           data: Union[np.ndarray, pd.Series], 
           exog: Optional[Union[np.ndarray, pd.DataFrame]] = None,
           exog_names: Optional[List[str]] = None,
           **kwargs: Any) -> TimeSeriesResult:
        """Fit the ARMAX model to the provided data.
        
        Args:
            data: The data to fit the model to
            exog: Exogenous variables
            exog_names: Names of exogenous variables
            **kwargs: Additional keyword arguments for model fitting
        
        Returns:
            TimeSeriesResult: The model estimation results
        
        Raises:
            ValueError: If the data is invalid
            EstimationError: If the model estimation fails
        """
        # Validate and prepare data
        data_array = self.validate_data(data)
        self._data = data_array
        
        # Validate and prepare exogenous variables
        if exog is not None:
            if isinstance(exog, pd.DataFrame):
                self._exog_names = exog.columns.tolist() if exog_names is None else exog_names
                exog_array = exog.values
            elif isinstance(exog, np.ndarray):
                if exog.ndim == 1:
                    exog_array = exog.reshape(-1, 1)
                else:
                    exog_array = exog
                
                if exog_names is None:
                    self._exog_names = [f"exog{i+1}" for i in range(exog_array.shape[1])]
                else:
                    if len(exog_names) != exog_array.shape[1]:
                        raise ValueError(
                            f"Length of exog_names ({len(exog_names)}) must match "
                            f"number of exogenous variables ({exog_array.shape[1]})"
                        )
                    self._exog_names = exog_names
            else:
                raise TypeError(
                    f"exog must be a NumPy array or Pandas DataFrame, got {type(exog).__name__}"
                )
            
            # Check dimensions
            if len(data_array) != len(exog_array):
                raise DimensionError(
                    f"Length of data ({len(data_array)}) must match "
                    f"length of exog ({len(exog_array)})",
                    array_name="exog",
                    expected_shape=f"({len(data_array)}, k)",
                    actual_shape=exog_array.shape
                )
            
            # Check for NaN and Inf values
            if np.isnan(exog_array).any():
                raise ValueError("exog contains NaN values")
            
            if np.isinf(exog_array).any():
                raise ValueError("exog contains infinite values")
            
            self._exog = exog_array
        else:
            self._exog = None
            self._exog_names = None
        
        # Update config with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
        
        # Use statsmodels for ARMAX estimation
        try:
            # Create statsmodels ARIMA model
            sm_model = sm.tsa.ARIMA(
                data_array,
                exog=self._exog,
                order=(self.ar_order, 0, self.ma_order),
                trend='c' if self.include_constant else 'n'
            )
            
            # Fit the model
            sm_result = sm_model.fit(
                method=self._config.method,
                maxiter=self._config.max_iter,
                disp=self._config.display_progress
            )
            
            # Extract results
            params_dict = sm_result.params.to_dict()
            
            # Create ARMAParameters object
            ar_params = np.zeros(self.ar_order)
            ma_params = np.zeros(self.ma_order)
            constant = 0.0
            sigma2 = sm_result.sigma2
            
            # Extract AR parameters
            for i in range(self.ar_order):
                param_name = f'ar.L{i+1}'
                if param_name in params_dict:
                    ar_params[i] = params_dict[param_name]
            
            # Extract MA parameters
            for i in range(self.ma_order):
                param_name = f'ma.L{i+1}'
                if param_name in params_dict:
                    ma_params[i] = params_dict[param_name]
            
            # Extract constant
            if self.include_constant and 'const' in params_dict:
                constant = params_dict['const']
            
            # Create parameter object
            params = ARMAParameters(
                ar_params=ar_params,
                ma_params=ma_params,
                sigma2=sigma2,
                constant=constant
            )
            
            # Store model attributes
            self._params = params
            self._residuals = sm_result.resid
            self._fitted_values = data_array - sm_result.resid
            self._cov_params = sm_result.cov_params()
            self._fitted = True
            
            # Create result object
            std_errors = np.sqrt(np.diag(self._cov_params))
            
            # Map standard errors to parameter names
            param_names = []
            if self.include_constant:
                param_names.append('constant')
            for i in range(self.ar_order):
                param_names.append(f'ar{i+1}')
            for i in range(self.ma_order):
                param_names.append(f'ma{i+1}')
            if self._exog is not None and self._exog_names is not None:
                param_names.extend(self._exog_names)
            param_names.append('sigma2')
            
            # Create result object
            result = self._create_result_object(
                params=params,
                data=data_array,
                residuals=self._residuals,
                fitted_values=self._fitted_values,
                loglikelihood=sm_result.llf,
                iterations=sm_result.mle_retvals.get('iterations', 0),
                convergence=sm_result.mle_retvals.get('success', True),
                cov_params=self._cov_params,
                std_errors=std_errors
            )
            
            self._results = result
            return result
            
        except Exception as e:
            raise EstimationError(
                f"ARMAX model estimation failed: {e}",
                model_type=self._name,
                estimation_method=self._config.method,
                details=str(e)
            )
    
    def forecast(self, 
                steps: int, 
                exog: Optional[np.ndarray] = None,
                confidence_level: float = 0.95,
                **kwargs: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate forecasts from the fitted ARMAX model.
        
        Args:
            steps: Number of steps to forecast
            exog: Exogenous variables for the forecast period
            confidence_level: Confidence level for prediction intervals
            **kwargs: Additional keyword arguments for forecasting
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Forecasts, lower bounds, and upper bounds
        
        Raises:
            NotFittedError: If the model has not been fitted
            ValueError: If the forecast parameters are invalid
            ForecastError: If the forecasting fails
        """
        if not self._fitted:
            raise NotFittedError(
                "Model has not been fitted. Call fit() first.",
                model_type=self._name,
                operation="forecast"
            )
        
        if steps <= 0:
            raise ValueError(f"Steps must be positive, got {steps}")
        
        if not 0 < confidence_level < 1:
            raise ValueError(
                f"Confidence level must be between 0 and 1, got {confidence_level}"
            )
        
        # Check if model was fitted with exogenous variables
        if self._exog is not None and exog is None:
            raise ValueError(
                "Model was fitted with exogenous variables, but none were provided for forecasting"
            )
        
        # Validate exogenous variables for forecasting
        if exog is not None:
            if isinstance(exog, pd.DataFrame):
                exog_array = exog.values
            elif isinstance(exog, np.ndarray):
                if exog.ndim == 1:
                    exog_array = exog.reshape(-1, 1)
                else:
                    exog_array = exog
            else:
                raise TypeError(
                    f"exog must be a NumPy array or Pandas DataFrame, got {type(exog).__name__}"
                )
            
            # Check dimensions
            if len(exog_array) < steps:
                raise ValueError(
                    f"Length of exog ({len(exog_array)}) must be at least steps ({steps})"
                )
            
            if self._exog is not None and exog_array.shape[1] != self._exog.shape[1]:
                raise DimensionError(
                    f"Number of exogenous variables ({exog_array.shape[1]}) must match "
                    f"number of exogenous variables used in fitting ({self._exog.shape[1]})",
                    array_name="exog",
                    expected_shape=f"(steps, {self._exog.shape[1]})",
                    actual_shape=exog_array.shape
                )
            
            # Check for NaN and Inf values
            if np.isnan(exog_array).any():
                raise ValueError("exog contains NaN values")
            
            if np.isinf(exog_array).any():
                raise ValueError("exog contains infinite values")
            
            # Use only the first 'steps' rows
            exog_array = exog_array[:steps]
        else:
            exog_array = None
        
        try:
            # Use statsmodels for forecasting
            sm_model = sm.tsa.ARIMA(
                self._data,
                exog=self._exog,
                order=(self.ar_order, 0, self.ma_order),
                trend='c' if self.include_constant else 'n'
            )
            
            # Refit with the same parameters
            sm_result = sm_model.filter(self._params.to_array())
            
            # Generate forecasts
            forecast_result = sm_result.get_forecast(steps=steps, exog=exog_array)
            
            # Extract forecasts and confidence intervals
            forecasts = forecast_result.predicted_mean
            
            # Compute confidence intervals
            alpha = 1 - confidence_level
            ci = forecast_result.conf_int(alpha=alpha)
            lower_bounds = ci.iloc[:, 0].values
            upper_bounds = ci.iloc[:, 1].values
            
            return forecasts.values, lower_bounds, upper_bounds
            
        except Exception as e:
            raise ForecastError(
                f"ARMAX forecasting failed: {e}",
                model_type=self._name,
                horizon=steps,
                details=str(e)
            )
    
    def simulate(self, 
                n_periods: int, 
                burn: int = 0, 
                initial_values: Optional[np.ndarray] = None,
                exog: Optional[np.ndarray] = None,
                random_state: Optional[Union[int, np.random.Generator]] = None,
                **kwargs: Any) -> np.ndarray:
        """Simulate data from the ARMAX model.
        
        Args:
            n_periods: Number of periods to simulate
            burn: Number of initial observations to discard
            initial_values: Initial values for the simulation
            exog: Exogenous variables for the simulation period
            random_state: Random number generator or seed
            **kwargs: Additional keyword arguments for simulation
        
        Returns:
            np.ndarray: Simulated data
        
        Raises:
            NotFittedError: If the model has not been fitted
            ValueError: If the simulation parameters are invalid
            SimulationError: If the simulation fails
        """
        if not self._fitted:
            raise NotFittedError(
                "Model has not been fitted. Call fit() first.",
                model_type=self._name,
                operation="simulate"
            )
        
        if n_periods <= 0:
            raise ValueError(f"n_periods must be positive, got {n_periods}")
        
        if burn < 0:
            raise ValueError(f"burn must be non-negative, got {burn}")
        
        # Check if model was fitted with exogenous variables
        if self._exog is not None and exog is None:
            raise ValueError(
                "Model was fitted with exogenous variables, but none were provided for simulation"
            )
        
        # Validate exogenous variables for simulation
        if exog is not None:
            if isinstance(exog, pd.DataFrame):
                exog_array = exog.values
            elif isinstance(exog, np.ndarray):
                if exog.ndim == 1:
                    exog_array = exog.reshape(-1, 1)
                else:
                    exog_array = exog
            else:
                raise TypeError(
                    f"exog must be a NumPy array or Pandas DataFrame, got {type(exog).__name__}"
                )
            
            # Check dimensions
            total_periods = n_periods + burn
            if len(exog_array) < total_periods:
                raise ValueError(
                    f"Length of exog ({len(exog_array)}) must be at least "
                    f"n_periods + burn ({total_periods})"
                )
            
            if self._exog is not None and exog_array.shape[1] != self._exog.shape[1]:
                raise DimensionError(
                    f"Number of exogenous variables ({exog_array.shape[1]}) must match "
                    f"number of exogenous variables used in fitting ({self._exog.shape[1]})",
                    array_name="exog",
                    expected_shape=f"(n_periods + burn, {self._exog.shape[1]})",
                    actual_shape=exog_array.shape
                )
            
            # Check for NaN and Inf values
            if np.isnan(exog_array).any():
                raise ValueError("exog contains NaN values")
            
            if np.isinf(exog_array).any():
                raise ValueError("exog contains infinite values")
            
            exog_array = exog_array
        else:
            exog_array = None
        
        # Set up random number generator
        if isinstance(random_state, np.random.Generator):
            rng = random_state
        else:
            rng = np.random.default_rng(random_state)
        
        try:
            # Extract parameters
            ar_params = self._params.ar_params
            ma_params = self._params.ma_params
            sigma2 = self._params.sigma2
            constant = self._params.constant
            
            # Total periods to simulate (including burn-in)
            total_periods = n_periods + burn
            
            # Maximum lag order
            max_lag = max(self.ar_order, self.ma_order)
            
            # Initialize arrays
            simulated = np.zeros(total_periods + max_lag)
            errors = np.zeros(total_periods + max_lag)
            
            # Set initial values if provided
            if initial_values is not None:
                if len(initial_values) < max_lag:
                    raise ValueError(
                        f"initial_values must have length at least {max_lag}, "
                        f"got {len(initial_values)}"
                    )
                simulated[:max_lag] = initial_values[:max_lag]
            
            # Generate random errors
            errors[max_lag:] = rng.normal(0, np.sqrt(sigma2), total_periods)
            
            # Generate simulated data
            for t in range(max_lag, total_periods + max_lag):
                # Add constant term
                simulated[t] = constant
                
                # Add exogenous variables if provided
                if exog_array is not None:
                    # Get exogenous variables for this time period
                    exog_t = exog_array[t - max_lag]
                    
                    # Add exogenous effects
                    # Note: This is a simplified approach - in practice, you would
                    # need to extract the exogenous coefficients from the model
                    # For now, we'll assume they're stored somewhere in the model
                    # and accessible as self._exog_coefs
                    if hasattr(self, '_exog_coefs'):
                        simulated[t] += np.dot(exog_t, self._exog_coefs)
                
                # Add AR terms
                for i in range(self.ar_order):
                    if t - i - 1 >= 0:
                        simulated[t] += ar_params[i] * simulated[t - i - 1]
                
                # Add MA terms and current error
                simulated[t] += errors[t]
                for i in range(self.ma_order):
                    if t - i - 1 >= 0:
                        simulated[t] += ma_params[i] * errors[t - i - 1]
            
            # Return simulated data (excluding burn-in)
            return simulated[max_lag + burn:]
            
        except Exception as e:
            raise SimulationError(
                f"ARMAX simulation failed: {e}",
                model_type=self._name,
                n_periods=n_periods,
                details=str(e)
            )
    
    def loglikelihood(self, 
                     params: ARMAParameters, 
                     data: np.ndarray, 
                     exog: Optional[np.ndarray] = None,
                     **kwargs: Any) -> float:
        """Compute the log-likelihood of the ARMAX model.
        
        Args:
            params: Model parameters
            data: Input data
            exog: Exogenous variables
            **kwargs: Additional keyword arguments for log-likelihood computation
        
        Returns:
            float: Log-likelihood value
        
        Raises:
            ValueError: If the parameters or data are invalid
            NumericError: If the log-likelihood computation fails
        """
        try:
            # Validate parameters
            params.validate()
            
            # Use statsmodels for log-likelihood computation
            sm_model = sm.tsa.ARIMA(
                data,
                exog=exog,
                order=(self.ar_order, 0, self.ma_order),
                trend='c' if self.include_constant else 'n'
            )
            
            # Convert parameters to statsmodels format
            sm_params = []
            if self.include_constant:
                sm_params.append(params.constant)
            sm_params.extend(params.ar_params)
            sm_params.extend(params.ma_params)
            
            # Add exogenous coefficients if available
            if exog is not None and hasattr(self, '_exog_coefs'):
                sm_params.extend(self._exog_coefs)
            
            sm_params.append(np.sqrt(params.sigma2))  # statsmodels uses standard deviation
            
            # Compute log-likelihood
            llf = sm_model.loglike(sm_params)
            
            return llf
            
        except Exception as e:
            raise NumericError(
                f"ARMAX log-likelihood computation failed: {e}",
                operation="log-likelihood",
                error_type="computation",
                details=str(e)
            )
    
    def summary(self) -> str:
        """Generate a text summary of the ARMAX model.
        
        Returns:
            str: A formatted string containing the model summary
        
        Raises:
            NotFittedError: If the model has not been fitted
        """
        if not self._fitted:
            return f"ARMAX({self.ar_order}, {self.ma_order}) Model (not fitted)"
        
        if self._results is None:
            return f"ARMAX({self.ar_order}, {self.ma_order}) Model (fitted, but no results available)"
        
        # Use the result object's summary method
        return self._results.summary()