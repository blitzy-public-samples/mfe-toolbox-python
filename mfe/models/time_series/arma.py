# mfe/models/time_series/arma.py
"""
ARMA and ARMAX time series models for financial time series analysis.

This module implements AutoRegressive Moving Average (ARMA) and AutoRegressive
Moving Average with eXogenous variables (ARMAX) models for financial time series
analysis. It provides comprehensive support for model specification, estimation,
diagnostics, and forecasting through a class-based design that integrates with
Statsmodels while extending it with financial-specific features.

The implementation supports custom lag specifications, various error distributions,
and optimized parameter estimation. It leverages Numba's JIT compilation for
performance-critical sections and provides both synchronous and asynchronous
interfaces for long-running operations.
"""

import asyncio
import logging
import warnings
from dataclasses import dataclass, field, asdict
from typing import (
    Any, Callable, Dict, List, Literal, Optional, Protocol, Sequence, 
    Tuple, Type, TypeVar, Union, cast, overload
)

import numpy as np
import pandas as pd
from scipy import stats, optimize
import statsmodels.api as sm
from numba import jit

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
from mfe.models.distributions.base import BaseDistribution
from mfe.models.distributions.normal import Normal, NormalParams
from mfe.models.distributions.student_t import StudentT, StudentTParams
from mfe.models.distributions.generalized_error import GED, GEDParams
from mfe.models.distributions.skewed_t import SkewedT, SkewedTParams
from mfe.models.time_series.base import (
    TimeSeriesModel, TimeSeriesConfig, TimeSeriesResult
)

# Set up module-level logger
logger = logging.getLogger("mfe.models.time_series.arma")


# Numba-accelerated core functions for ARMA models
@jit(nopython=True, cache=True)
def _arma_recursion(
    data: np.ndarray,
    ar_params: np.ndarray,
    ma_params: np.ndarray,
    constant: float,
    sigma2: float,
    ar_order: int,
    ma_order: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ARMA recursion for residuals and fitted values.
    
    This function implements the core ARMA recursion to compute residuals
    and fitted values given the model parameters and data. It is accelerated
    using Numba's JIT compilation for improved performance.
    
    Args:
        data: Input time series data
        ar_params: Autoregressive parameters
        ma_params: Moving average parameters
        constant: Constant term
        sigma2: Innovation variance
        ar_order: Order of the autoregressive component
        ma_order: Order of the moving average component
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Residuals and fitted values
    """
    n = len(data)
    residuals = np.zeros(n)
    fitted = np.zeros(n)
    
    # Initialize with zeros
    for t in range(max(ar_order, ma_order)):
        fitted[t] = data[t]
        residuals[t] = 0.0
    
    # Main recursion
    for t in range(max(ar_order, ma_order), n):
        # Add constant term
        fitted[t] = constant
        
        # Add AR terms
        for i in range(ar_order):
            if t - i - 1 >= 0:
                fitted[t] += ar_params[i] * data[t - i - 1]
        
        # Add MA terms
        for j in range(ma_order):
            if t - j - 1 >= 0:
                fitted[t] -= ma_params[j] * residuals[t - j - 1]
        
        # Compute residual
        residuals[t] = data[t] - fitted[t]
    
    return residuals, fitted


@jit(nopython=True, cache=True)
def _arma_forecast(
    data: np.ndarray,
    residuals: np.ndarray,
    ar_params: np.ndarray,
    ma_params: np.ndarray,
    constant: float,
    sigma2: float,
    ar_order: int,
    ma_order: int,
    steps: int
) -> np.ndarray:
    """Generate forecasts from an ARMA model.
    
    This function implements the core ARMA forecasting algorithm to generate
    point forecasts given the model parameters, data, and residuals. It is
    accelerated using Numba's JIT compilation for improved performance.
    
    Args:
        data: Input time series data
        residuals: Model residuals
        ar_params: Autoregressive parameters
        ma_params: Moving average parameters
        constant: Constant term
        sigma2: Innovation variance
        ar_order: Order of the autoregressive component
        ma_order: Order of the moving average component
        steps: Number of steps to forecast
    
    Returns:
        np.ndarray: Point forecasts
    """
    n = len(data)
    forecasts = np.zeros(steps)
    
    # Create extended data and residuals arrays
    extended_data = np.zeros(n + steps)
    extended_residuals = np.zeros(n + steps)
    
    # Fill with actual data and residuals
    extended_data[:n] = data
    extended_residuals[:n] = residuals
    
    # Generate forecasts
    for h in range(steps):
        t = n + h
        
        # Add constant term
        forecasts[h] = constant
        
        # Add AR terms
        for i in range(ar_order):
            if t - i - 1 >= 0:
                if t - i - 1 < n:
                    # Use actual data
                    forecasts[h] += ar_params[i] * data[t - i - 1]
                else:
                    # Use forecasted data
                    forecasts[h] += ar_params[i] * extended_data[t - i - 1]
        
        # Add MA terms
        for j in range(ma_order):
            if t - j - 1 >= 0 and t - j - 1 < n:
                # Only use known residuals
                forecasts[h] -= ma_params[j] * residuals[t - j - 1]
        
        # Store forecast in extended data
        extended_data[t] = forecasts[h]
    
    return forecasts


@jit(nopython=True, cache=True)
def _arma_simulate(
    ar_params: np.ndarray,
    ma_params: np.ndarray,
    constant: float,
    sigma2: float,
    ar_order: int,
    ma_order: int,
    n_periods: int,
    burn: int,
    initial_values: np.ndarray,
    innovations: np.ndarray
) -> np.ndarray:
    """Simulate data from an ARMA model.
    
    This function implements the core ARMA simulation algorithm to generate
    simulated data given the model parameters. It is accelerated using Numba's
    JIT compilation for improved performance.
    
    Args:
        ar_params: Autoregressive parameters
        ma_params: Moving average parameters
        constant: Constant term
        sigma2: Innovation variance
        ar_order: Order of the autoregressive component
        ma_order: Order of the moving average component
        n_periods: Number of periods to simulate
        burn: Number of initial observations to discard
        initial_values: Initial values for the simulation
        innovations: Random innovations for the simulation
    
    Returns:
        np.ndarray: Simulated data
    """
    max_lag = max(ar_order, ma_order)
    total_periods = n_periods + burn
    
    # Initialize arrays
    simulated = np.zeros(total_periods + max_lag)
    errors = np.zeros(total_periods + max_lag)
    
    # Set initial values
    if len(initial_values) > 0:
        simulated[:min(len(initial_values), max_lag)] = initial_values[:min(len(initial_values), max_lag)]
    
    # Set innovations
    errors[max_lag:] = innovations
    
    # Generate simulated data
    for t in range(max_lag, total_periods + max_lag):
        # Add constant term
        simulated[t] = constant
        
        # Add AR terms
        for i in range(ar_order):
            if t - i - 1 >= 0:
                simulated[t] += ar_params[i] * simulated[t - i - 1]
        
        # Add MA terms and current error
        simulated[t] += errors[t]
        for i in range(ma_order):
            if t - i - 1 >= 0:
                simulated[t] += ma_params[i] * errors[t - i - 1]
    
    # Return simulated data (excluding burn-in)
    return simulated[max_lag + burn:]


@dataclass
class ARMAXConfig(TimeSeriesConfig):
    """Configuration options for ARMA/ARMAX models.
    
    This class extends TimeSeriesConfig with additional options specific to
    ARMA/ARMAX models, including distribution type and optimization settings.
    
    Attributes:
        method: Estimation method to use
        solver: Optimization solver to use
        max_iter: Maximum number of iterations for optimization
        tol: Convergence tolerance for optimization
        cov_type: Type of covariance matrix to compute
        use_numba: Whether to use Numba acceleration if available
        display_progress: Whether to display progress during estimation
        distribution: Error distribution type
        distribution_params: Additional parameters for the error distribution
    """
    
    distribution: Literal["normal", "t", "ged", "skewed_t"] = "normal"
    distribution_params: Optional[Dict[str, Any]] = None
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        super().__post_init__()
        
        # Validate distribution
        valid_distributions = ["normal", "t", "ged", "skewed_t"]
        if self.distribution not in valid_distributions:
            raise ParameterError(
                f"Invalid distribution: {self.distribution}",
                param_name="distribution",
                param_value=self.distribution,
                constraint=f"Must be one of {valid_distributions}"
            )
        
        # Initialize distribution_params if None
        if self.distribution_params is None:
            self.distribution_params = {}


@dataclass
class ARMAXResult(TimeSeriesResult):
    """Results from ARMA/ARMAX model estimation.
    
    This class extends TimeSeriesResult with additional attributes specific to
    ARMA/ARMAX models, including distribution information and model orders.
    
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
        ar_order: Order of the autoregressive component
        ma_order: Order of the moving average component
        include_constant: Whether a constant term is included
        distribution: Error distribution type
        distribution_params: Parameters of the error distribution
        exog_names: Names of exogenous variables (if any)
    """
    
    ar_order: int = 0
    ma_order: int = 0
    include_constant: bool = True
    distribution: str = "normal"
    distribution_params: Dict[str, Any] = field(default_factory=dict)
    exog_names: Optional[List[str]] = None
    
    def summary(self) -> str:
        """Generate a text summary of the model results.
        
        Returns:
            str: A formatted string containing the model results summary.
        """
        # Get the base summary from the parent class
        base_summary = super().summary()
        
        # Add ARMA/ARMAX specific information
        armax_info = f"\nModel Specification:\n"
        armax_info += "-" * 40 + "\n"
        armax_info += f"AR order: {self.ar_order}\n"
        armax_info += f"MA order: {self.ma_order}\n"
        armax_info += f"Include constant: {self.include_constant}\n"
        armax_info += f"Distribution: {self.distribution}\n"
        
        if self.distribution_params:
            armax_info += "Distribution parameters:\n"
            for param, value in self.distribution_params.items():
                armax_info += f"  {param}: {value}\n"
        
        if self.exog_names:
            armax_info += "Exogenous variables:\n"
            for name in self.exog_names:
                armax_info += f"  {name}\n"
        
        armax_info += "-" * 40 + "\n"
        
        # Insert ARMA/ARMAX info after the header but before the parameter table
        header_end = base_summary.find("Parameter Estimates:")
        if header_end > 0:
            return base_summary[:header_end] + armax_info + base_summary[header_end:]
        else:
            return base_summary + armax_info


class ARMAModel(TimeSeriesModel[ARMAParameters]):
    """ARMA (AutoRegressive Moving Average) model implementation.
    
    This class implements the ARMA model for time series analysis, providing
    methods for model specification, estimation, diagnostics, and forecasting.
    It integrates with Statsmodels while extending it with additional features
    and optimizations.
    
    Attributes:
        ar_order: Order of the autoregressive component
        ma_order: Order of the moving average component
        include_constant: Whether to include a constant term in the model
        config: Model configuration options
    """
    
    def __init__(self, 
                ar_order: int = 1, 
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
        self._config = ARMAXConfig()
        self._distribution: Optional[BaseDistribution] = None
        
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
    
    def _initialize_distribution(self) -> None:
        """Initialize the error distribution based on configuration.
        
        This method creates the appropriate distribution object based on the
        configuration settings.
        
        Raises:
            ParameterError: If the distribution type is invalid
        """
        dist_type = self._config.distribution
        dist_params = self._config.distribution_params or {}
        
        if dist_type == "normal":
            mu = dist_params.get("mu", 0.0)
            sigma = dist_params.get("sigma", 1.0)
            self._distribution = Normal(params=NormalParams(mu=mu, sigma=sigma))
        
        elif dist_type == "t":
            df = dist_params.get("df", 5.0)
            self._distribution = StudentT(params=StudentTParams(df=df))
        
        elif dist_type == "ged":
            nu = dist_params.get("nu", 1.5)
            self._distribution = GED(params=GEDParams(nu=nu))
        
        elif dist_type == "skewed_t":
            df = dist_params.get("df", 5.0)
            lambda_ = dist_params.get("lambda_", 0.0)
            self._distribution = SkewedT(params=SkewedTParams(df=df, lambda_=lambda_))
        
        else:
            raise ParameterError(
                f"Invalid distribution type: {dist_type}",
                param_name="distribution",
                param_value=dist_type,
                constraint="Must be one of ['normal', 't', 'ged', 'skewed_t']"
            )
    
    def fit(self, 
           data: Union[np.ndarray, pd.Series], 
           **kwargs: Any) -> ARMAXResult:
        """Fit the ARMA model to the provided data.
        
        This method estimates the ARMA model parameters from the provided data
        using the specified method and configuration options.
        
        Args:
            data: The data to fit the model to
            **kwargs: Additional keyword arguments for model fitting
        
        Returns:
            ARMAXResult: The model estimation results
        
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
        
        # Initialize distribution
        self._initialize_distribution()
        
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
            
            # Compute residuals and fitted values using Numba-accelerated function
            if self._config.use_numba:
                residuals, fitted_values = _arma_recursion(
                    data_array,
                    ar_params,
                    ma_params,
                    constant,
                    sigma2,
                    self.ar_order,
                    self.ma_order
                )
            else:
                # Use statsmodels residuals and fitted values
                residuals = sm_result.resid
                fitted_values = data_array - residuals
            
            # Store model attributes
            self._params = params
            self._residuals = residuals
            self._fitted_values = fitted_values
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
            
            # Create parameter dictionaries
            params_dict = {}
            std_errors_dict = {}
            t_stats_dict = {}
            p_values_dict = {}
            
            # Add constant
            if self.include_constant:
                params_dict['constant'] = constant
                std_errors_dict['constant'] = std_errors[0]
                t_stats_dict['constant'] = constant / std_errors[0]
                p_values_dict['constant'] = 2 * (1 - stats.t.cdf(abs(t_stats_dict['constant']), len(data_array) - len(param_names)))
                offset = 1
            else:
                offset = 0
            
            # Add AR parameters
            for i in range(self.ar_order):
                param_name = f'ar{i+1}'
                params_dict[param_name] = ar_params[i]
                std_errors_dict[param_name] = std_errors[i + offset]
                t_stats_dict[param_name] = ar_params[i] / std_errors[i + offset]
                p_values_dict[param_name] = 2 * (1 - stats.t.cdf(abs(t_stats_dict[param_name]), len(data_array) - len(param_names)))
            
            # Add MA parameters
            for i in range(self.ma_order):
                param_name = f'ma{i+1}'
                params_dict[param_name] = ma_params[i]
                std_errors_dict[param_name] = std_errors[i + offset + self.ar_order]
                t_stats_dict[param_name] = ma_params[i] / std_errors[i + offset + self.ar_order]
                p_values_dict[param_name] = 2 * (1 - stats.t.cdf(abs(t_stats_dict[param_name]), len(data_array) - len(param_names)))
            
            # Add sigma2
            params_dict['sigma2'] = sigma2
            std_errors_dict['sigma2'] = std_errors[-1]
            t_stats_dict['sigma2'] = sigma2 / std_errors[-1]
            p_values_dict['sigma2'] = 2 * (1 - stats.t.cdf(abs(t_stats_dict['sigma2']), len(data_array) - len(param_names)))
            
            # Extract distribution parameters
            dist_params = {}
            if self._distribution is not None and self._distribution.params is not None:
                dist_params = self._distribution.params.to_dict()
            
            # Create result object
            result = ARMAXResult(
                model_name=self._name,
                params=params_dict,
                std_errors=std_errors_dict,
                t_stats=t_stats_dict,
                p_values=p_values_dict,
                log_likelihood=sm_result.llf,
                aic=sm_result.aic,
                bic=sm_result.bic,
                hqic=sm_result.hqic,
                residuals=residuals,
                fitted_values=fitted_values,
                convergence=sm_result.mle_retvals.get('success', True),
                iterations=sm_result.mle_retvals.get('iterations', 0),
                cov_type=self._config.cov_type,
                cov_params=self._cov_params,
                nobs=len(data_array),
                df_model=len(param_names),
                df_resid=len(data_array) - len(param_names),
                ar_order=self.ar_order,
                ma_order=self.ma_order,
                include_constant=self.include_constant,
                distribution=self._config.distribution,
                distribution_params=dist_params
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
    
    async def fit_async(self, 
                       data: Union[np.ndarray, pd.Series], 
                       progress_callback: Optional[Callable[[float, str], None]] = None,
                       **kwargs: Any) -> ARMAXResult:
        """Asynchronously fit the ARMA model to the provided data.
        
        This method provides an asynchronous interface to the fit method,
        allowing for non-blocking model estimation in UI contexts.
        
        Args:
            data: The data to fit the model to
            progress_callback: Optional callback function for reporting progress
            **kwargs: Additional keyword arguments for model fitting
        
        Returns:
            ARMAXResult: The model estimation results
        
        Raises:
            ValueError: If the data is invalid
            EstimationError: If the model estimation fails
        """
        # Create a coroutine that runs the synchronous fit method in a thread pool
        loop = asyncio.get_event_loop()
        
        # Report initial progress
        if progress_callback:
            await loop.run_in_executor(
                None, progress_callback, 0.0, "Starting ARMA model estimation..."
            )
        
        # Define a wrapper function that reports progress
        def fit_with_progress():
            # Perform the actual fit
            result = self.fit(data, **kwargs)
            
            # Report completion
            if progress_callback:
                progress_callback(1.0, "ARMA model estimation complete")
            
            return result
        
        # Run the fit operation in a thread pool
        result = await loop.run_in_executor(None, fit_with_progress)
        
        return result
    
    def forecast(self, 
                steps: int, 
                exog: Optional[np.ndarray] = None,
                confidence_level: float = 0.95,
                **kwargs: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate forecasts from the fitted ARMA model.
        
        This method generates point forecasts and prediction intervals for the
        specified number of steps ahead.
        
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
            # Extract parameters
            ar_params = self._params.ar_params
            ma_params = self._params.ma_params
            constant = self._params.constant
            sigma2 = self._params.sigma2
            
            # Generate forecasts using Numba-accelerated function if enabled
            if self._config.use_numba:
                forecasts = _arma_forecast(
                    self._data,
                    self._residuals,
                    ar_params,
                    ma_params,
                    constant,
                    sigma2,
                    self.ar_order,
                    self.ma_order,
                    steps
                )
            else:
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
                forecasts = forecast_result.predicted_mean.values
            
            # Compute prediction intervals
            alpha = 1 - confidence_level
            z_value = stats.norm.ppf(1 - alpha / 2)
            
            # Compute forecast standard errors
            # This is a simplified approach - in practice, forecast errors
            # accumulate over the forecast horizon
            forecast_std = np.sqrt(sigma2) * np.ones(steps)
            for h in range(1, steps):
                # Increase uncertainty for longer horizons
                forecast_std[h] = forecast_std[h-1] * np.sqrt(1 + 0.05 * h)
            
            # Compute prediction intervals
            lower_bounds = forecasts - z_value * forecast_std
            upper_bounds = forecasts + z_value * forecast_std
            
            return forecasts, lower_bounds, upper_bounds
            
        except Exception as e:
            raise ForecastError(
                f"ARMA forecasting failed: {e}",
                model_type=self._name,
                horizon=steps,
                details=str(e)
            )
    
    async def forecast_async(self, 
                           steps: int, 
                           exog: Optional[np.ndarray] = None,
                           confidence_level: float = 0.95,
                           progress_callback: Optional[Callable[[float, str], None]] = None,
                           **kwargs: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Asynchronously generate forecasts from the fitted ARMA model.
        
        This method provides an asynchronous interface to the forecast method,
        allowing for non-blocking forecasting in UI contexts.
        
        Args:
            steps: Number of steps to forecast
            exog: Exogenous variables for the forecast period (not used in ARMA)
            confidence_level: Confidence level for prediction intervals
            progress_callback: Optional callback function for reporting progress
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
        
        # Report initial progress
        if progress_callback:
            await loop.run_in_executor(
                None, progress_callback, 0.0, "Starting ARMA forecast..."
            )
        
        # Define a wrapper function that reports progress
        def forecast_with_progress():
            # Perform the actual forecast
            result = self.forecast(steps, exog, confidence_level, **kwargs)
            
            # Report completion
            if progress_callback:
                progress_callback(1.0, "ARMA forecast complete")
            
            return result
        
        # Run the forecast operation in a thread pool
        result = await loop.run_in_executor(None, forecast_with_progress)
        
        return result
    
    def simulate(self, 
                n_periods: int, 
                burn: int = 0, 
                initial_values: Optional[np.ndarray] = None,
                random_state: Optional[Union[int, np.random.Generator]] = None,
                **kwargs: Any) -> np.ndarray:
        """Simulate data from the ARMA model.
        
        This method generates simulated data from the fitted ARMA model for the
        specified number of periods.
        
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
            constant = self._params.constant
            sigma2 = self._params.sigma2
            
            # Maximum lag order
            max_lag = max(self.ar_order, self.ma_order)
            
            # Prepare initial values
            if initial_values is None:
                initial_values = np.zeros(max_lag)
            elif len(initial_values) < max_lag:
                # Pad with zeros if not enough initial values
                padded = np.zeros(max_lag)
                padded[:len(initial_values)] = initial_values
                initial_values = padded
            
            # Generate random innovations
            total_periods = n_periods + burn
            innovations = rng.normal(0, np.sqrt(sigma2), total_periods)
            
            # Generate simulated data using Numba-accelerated function if enabled
            if self._config.use_numba:
                simulated = _arma_simulate(
                    ar_params,
                    ma_params,
                    constant,
                    sigma2,
                    self.ar_order,
                    self.ma_order,
                    n_periods,
                    burn,
                    initial_values,
                    innovations
                )
            else:
                # Initialize arrays
                simulated = np.zeros(total_periods + max_lag)
                errors = np.zeros(total_periods + max_lag)
                
                # Set initial values
                simulated[:max_lag] = initial_values[:max_lag]
                
                # Set innovations
                errors[max_lag:] = innovations
                
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
                simulated = simulated[max_lag + burn:]
            
            return simulated
            
        except Exception as e:
            raise SimulationError(
                f"ARMA simulation failed: {e}",
                model_type=self._name,
                n_periods=n_periods,
                details=str(e)
            )
    
    async def simulate_async(self, 
                           n_periods: int, 
                           burn: int = 0, 
                           initial_values: Optional[np.ndarray] = None,
                           random_state: Optional[Union[int, np.random.Generator]] = None,
                           progress_callback: Optional[Callable[[float, str], None]] = None,
                           **kwargs: Any) -> np.ndarray:
        """Asynchronously simulate data from the ARMA model.
        
        This method provides an asynchronous interface to the simulate method,
        allowing for non-blocking simulation in UI contexts.
        
        Args:
            n_periods: Number of periods to simulate
            burn: Number of initial observations to discard
            initial_values: Initial values for the simulation
            random_state: Random number generator or seed
            progress_callback: Optional callback function for reporting progress
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
        
        # Report initial progress
        if progress_callback:
            await loop.run_in_executor(
                None, progress_callback, 0.0, "Starting ARMA simulation..."
            )
        
        # Define a wrapper function that reports progress
        def simulate_with_progress():
            # Perform the actual simulation
            result = self.simulate(n_periods, burn, initial_values, random_state, **kwargs)
            
            # Report completion
            if progress_callback:
                progress_callback(1.0, "ARMA simulation complete")
            
            return result
        
        # Run the simulation operation in a thread pool
        result = await loop.run_in_executor(None, simulate_with_progress)
        
        return result
    
    def loglikelihood(self, 
                     params: ARMAParameters, 
                     data: np.ndarray, 
                     **kwargs: Any) -> float:
        """Compute the log-likelihood of the ARMA model.
        
        This method computes the log-likelihood of the data given the model
        parameters, which is used for parameter estimation and model comparison.
        
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
    
    This class extends the ARMA model to include exogenous variables, providing
    methods for model specification, estimation, diagnostics, and forecasting
    with exogenous inputs.
    
    Attributes:
        ar_order: Order of the autoregressive component
        ma_order: Order of the moving average component
        include_constant: Whether to include a constant term in the model
        config: Model configuration options
    """
    
    def __init__(self, 
                ar_order: int = 1, 
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
        self._exog_params: Optional[np.ndarray] = None
    
    def fit(self, 
           data: Union[np.ndarray, pd.Series], 
           exog: Optional[Union[np.ndarray, pd.DataFrame]] = None,
           exog_names: Optional[List[str]] = None,
           **kwargs: Any) -> ARMAXResult:
        """Fit the ARMAX model to the provided data.
        
        This method estimates the ARMAX model parameters from the provided data
        and exogenous variables using the specified method and configuration options.
        
        Args:
            data: The data to fit the model to
            exog: Exogenous variables
            exog_names: Names of exogenous variables
            **kwargs: Additional keyword arguments for model fitting
        
        Returns:
            ARMAXResult: The model estimation results
        
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
        
        # Initialize distribution
        self._initialize_distribution()
        
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
            
            # Extract exogenous parameters
            if self._exog is not None and self._exog_names is not None:
                self._exog_params = np.zeros(len(self._exog_names))
                for i, name in enumerate(self._exog_names):
                    if name in params_dict:
                        self._exog_params[i] = params_dict[name]
            
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
            
            # Create parameter dictionaries
            params_dict = {}
            std_errors_dict = {}
            t_stats_dict = {}
            p_values_dict = {}
            
            # Add constant
            if self.include_constant:
                params_dict['constant'] = constant
                std_errors_dict['constant'] = std_errors[0]
                t_stats_dict['constant'] = constant / std_errors[0]
                p_values_dict['constant'] = 2 * (1 - stats.t.cdf(abs(t_stats_dict['constant']), len(data_array) - len(param_names)))
                offset = 1
            else:
                offset = 0
            
            # Add AR parameters
            for i in range(self.ar_order):
                param_name = f'ar{i+1}'
                params_dict[param_name] = ar_params[i]
                std_errors_dict[param_name] = std_errors[i + offset]
                t_stats_dict[param_name] = ar_params[i] / std_errors[i + offset]
                p_values_dict[param_name] = 2 * (1 - stats.t.cdf(abs(t_stats_dict[param_name]), len(data_array) - len(param_names)))
            
            # Add MA parameters
            for i in range(self.ma_order):
                param_name = f'ma{i+1}'
                params_dict[param_name] = ma_params[i]
                std_errors_dict[param_name] = std_errors[i + offset + self.ar_order]
                t_stats_dict[param_name] = ma_params[i] / std_errors[i + offset + self.ar_order]
                p_values_dict[param_name] = 2 * (1 - stats.t.cdf(abs(t_stats_dict[param_name]), len(data_array) - len(param_names)))
            
            # Add exogenous parameters
            if self._exog is not None and self._exog_names is not None and self._exog_params is not None:
                for i, name in enumerate(self._exog_names):
                    params_dict[name] = self._exog_params[i]
                    idx = i + offset + self.ar_order + self.ma_order
                    std_errors_dict[name] = std_errors[idx]
                    t_stats_dict[name] = self._exog_params[i] / std_errors[idx]
                    p_values_dict[name] = 2 * (1 - stats.t.cdf(abs(t_stats_dict[name]), len(data_array) - len(param_names)))
            
            # Add sigma2
            params_dict['sigma2'] = sigma2
            std_errors_dict['sigma2'] = std_errors[-1]
            t_stats_dict['sigma2'] = sigma2 / std_errors[-1]
            p_values_dict['sigma2'] = 2 * (1 - stats.t.cdf(abs(t_stats_dict['sigma2']), len(data_array) - len(param_names)))
            
            # Extract distribution parameters
            dist_params = {}
            if self._distribution is not None and self._distribution.params is not None:
                dist_params = self._distribution.params.to_dict()
            
            # Create result object
            result = ARMAXResult(
                model_name=self._name,
                params=params_dict,
                std_errors=std_errors_dict,
                t_stats=t_stats_dict,
                p_values=p_values_dict,
                log_likelihood=sm_result.llf,
                aic=sm_result.aic,
                bic=sm_result.bic,
                hqic=sm_result.hqic,
                residuals=self._residuals,
                fitted_values=self._fitted_values,
                convergence=sm_result.mle_retvals.get('success', True),
                iterations=sm_result.mle_retvals.get('iterations', 0),
                cov_type=self._config.cov_type,
                cov_params=self._cov_params,
                nobs=len(data_array),
                df_model=len(param_names),
                df_resid=len(data_array) - len(param_names),
                ar_order=self.ar_order,
                ma_order=self.ma_order,
                include_constant=self.include_constant,
                distribution=self._config.distribution,
                distribution_params=dist_params,
                exog_names=self._exog_names
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
        
        This method generates point forecasts and prediction intervals for the
        specified number of steps ahead, using exogenous variables if provided.
        
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
        
        This method generates simulated data from the fitted ARMAX model for the
        specified number of periods, using exogenous variables if provided.
        
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
            
            exog_array = exog_array[:total_periods]
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
                if exog_array is not None and self._exog_params is not None:
                    # Get exogenous variables for this time period
                    exog_t = exog_array[t - max_lag]
                    
                    # Add exogenous effects
                    simulated[t] += np.dot(exog_t, self._exog_params)
                
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
        
        This method computes the log-likelihood of the data given the model
        parameters and exogenous variables, which is used for parameter estimation
        and model comparison.
        
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
            if exog is not None and self._exog_params is not None:
                sm_params.extend(self._exog_params)
            
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
