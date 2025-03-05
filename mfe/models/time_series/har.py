'''
Heterogeneous Autoregression (HAR) models for time series analysis.

This module implements HAR (Heterogeneous Autoregression) models, which are
particularly useful for modeling and forecasting realized volatility. HAR models
capture the long-memory properties of volatility through a parsimonious
specification using different time horizons (daily, weekly, monthly).

The implementation provides various HAR model specifications, including the
standard HAR model, HAR with leverage effects, HAR with jumps, and customizable
lag aggregation periods. The module supports efficient estimation, forecasting,
and simulation with proper diagnostics and inference.

References:
    Corsi, F. (2009). A simple approximate long-memory model of realized
    volatility. Journal of Financial Econometrics, 7(2), 174-196.
    
    Andersen, T. G., Bollerslev, T., & Diebold, F. X. (2007). Roughing it up:
    Including jump components in the measurement, modeling, and forecasting of
    return volatility. The Review of Economics and Statistics, 89(4), 701-720.
'''

import logging
import warnings
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union, cast
)
import numpy as np
import pandas as pd
from scipy import stats, optimize
import statsmodels.api as sm

from numba import jit

from mfe.core.base import ModelBase
from mfe.core.parameters import (
    ParameterBase, TimeSeriesParameters, validate_positive, validate_non_negative,
    validate_range, transform_positive, inverse_transform_positive
)
from mfe.core.exceptions import (
    ParameterError, DimensionError, ConvergenceError, NumericError,
    EstimationError, ForecastError, SimulationError, NotFittedError,
    warn_convergence, warn_numeric, warn_model
)
from mfe.models.time_series.base import TimeSeriesModel, TimeSeriesResult, TimeSeriesConfig

# Set up module-level logger
logger = logging.getLogger("mfe.models.time_series.har")


@dataclass
class HARParameters(TimeSeriesParameters):
    """Parameters for HAR (Heterogeneous Autoregression) model.
    
    This class defines the parameters for HAR models, including the constant term
    and coefficients for different time horizons (daily, weekly, monthly, etc.).
    
    Attributes:
        constant: Constant term in the model
        beta_d: Coefficient for daily component
        beta_w: Coefficient for weekly component
        beta_m: Coefficient for monthly component
        beta_q: Coefficient for quarterly component (optional)
        beta_y: Coefficient for yearly component (optional)
        beta_j: Coefficient for jump component (optional)
        beta_r: Coefficient for signed return (leverage) component (optional)
        sigma2: Innovation variance (must be positive)
    """
    
    constant: float
    beta_d: float
    beta_w: float
    beta_m: float
    beta_q: Optional[float] = None
    beta_y: Optional[float] = None
    beta_j: Optional[float] = None
    beta_r: Optional[float] = None
    sigma2: float = field(default=1.0)
    
    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate HAR parameter constraints.
        
        Raises:
            ParameterError: If parameter constraints are violated
        """
        super().validate()
        
        # Validate sigma2
        validate_positive(self.sigma2, "sigma2")
        
        # No strict constraints on beta parameters, but we can check for stationarity
        # by ensuring the sum of coefficients is less than 1
        beta_sum = self.beta_d + self.beta_w + self.beta_m
        if self.beta_q is not None:
            beta_sum += self.beta_q
        if self.beta_y is not None:
            beta_sum += self.beta_y
            
        if beta_sum >= 1:
            warnings.warn(
                f"Sum of beta coefficients ({beta_sum:.4f}) is >= 1, which may indicate non-stationarity",
                UserWarning
            )
    
    def to_array(self) -> np.ndarray:
        """Convert parameters to a NumPy array.
        
        Returns:
            np.ndarray: Array representation of parameters
        """
        # Start with required parameters
        params = [self.constant, self.beta_d, self.beta_w, self.beta_m]
        
        # Add optional parameters if they exist
        if self.beta_q is not None:
            params.append(self.beta_q)
        if self.beta_y is not None:
            params.append(self.beta_y)
        if self.beta_j is not None:
            params.append(self.beta_j)
        if self.beta_r is not None:
            params.append(self.beta_r)
            
        # Add sigma2
        params.append(self.sigma2)
        
        return np.array(params)
    
    @classmethod
    def from_array(cls, array: np.ndarray, 
                  include_quarterly: bool = False,
                  include_yearly: bool = False,
                  include_jump: bool = False,
                  include_leverage: bool = False,
                  **kwargs: Any) -> 'HARParameters':
        """Create parameters from a NumPy array.
        
        Args:
            array: Array representation of parameters
            include_quarterly: Whether to include quarterly component
            include_yearly: Whether to include yearly component
            include_jump: Whether to include jump component
            include_leverage: Whether to include leverage component
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            HARParameters: Parameter object
        
        Raises:
            ValueError: If the array length doesn't match the expected length
        """
        # Calculate expected array length
        expected_length = 4  # constant, beta_d, beta_w, beta_m
        if include_quarterly:
            expected_length += 1
        if include_yearly:
            expected_length += 1
        if include_jump:
            expected_length += 1
        if include_leverage:
            expected_length += 1
        expected_length += 1  # sigma2
        
        if len(array) != expected_length:
            raise ValueError(
                f"Array length ({len(array)}) doesn't match expected length ({expected_length})"
            )
        
        # Extract parameters
        params = {}
        params['constant'] = array[0]
        params['beta_d'] = array[1]
        params['beta_w'] = array[2]
        params['beta_m'] = array[3]
        
        # Track current index
        idx = 4
        
        # Extract optional parameters
        if include_quarterly:
            params['beta_q'] = array[idx]
            idx += 1
        if include_yearly:
            params['beta_y'] = array[idx]
            idx += 1
        if include_jump:
            params['beta_j'] = array[idx]
            idx += 1
        if include_leverage:
            params['beta_r'] = array[idx]
            idx += 1
            
        # Extract sigma2
        params['sigma2'] = array[idx]
        
        # Update with any additional kwargs
        params.update(kwargs)
        
        return cls(**params)
    
    def transform(self) -> np.ndarray:
        """Transform parameters to unconstrained space for optimization.
        
        Returns:
            np.ndarray: Transformed parameters in unconstrained space
        """
        # Transform sigma2 to unconstrained space (log)
        transformed_sigma2 = transform_positive(self.sigma2)
        
        # Start with required parameters (no transformation needed)
        params = [self.constant, self.beta_d, self.beta_w, self.beta_m]
        
        # Add optional parameters if they exist
        if self.beta_q is not None:
            params.append(self.beta_q)
        if self.beta_y is not None:
            params.append(self.beta_y)
        if self.beta_j is not None:
            params.append(self.beta_j)
        if self.beta_r is not None:
            params.append(self.beta_r)
            
        # Add transformed sigma2
        params.append(transformed_sigma2)
        
        return np.array(params)
    
    @classmethod
    def inverse_transform(cls, array: np.ndarray,
                         include_quarterly: bool = False,
                         include_yearly: bool = False,
                         include_jump: bool = False,
                         include_leverage: bool = False,
                         **kwargs: Any) -> 'HARParameters':
        """Transform parameters from unconstrained space back to constrained space.
        
        Args:
            array: Parameters in unconstrained space
            include_quarterly: Whether to include quarterly component
            include_yearly: Whether to include yearly component
            include_jump: Whether to include jump component
            include_leverage: Whether to include leverage component
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            HARParameters: Parameter object with constrained parameters
        
        Raises:
            ValueError: If the array length doesn't match the expected length
        """
        # Calculate expected array length
        expected_length = 4  # constant, beta_d, beta_w, beta_m
        if include_quarterly:
            expected_length += 1
        if include_yearly:
            expected_length += 1
        if include_jump:
            expected_length += 1
        if include_leverage:
            expected_length += 1
        expected_length += 1  # sigma2
        
        if len(array) != expected_length:
            raise ValueError(
                f"Array length ({len(array)}) doesn't match expected length ({expected_length})"
            )
        
        # Extract parameters
        params = {}
        params['constant'] = array[0]
        params['beta_d'] = array[1]
        params['beta_w'] = array[2]
        params['beta_m'] = array[3]
        
        # Track current index
        idx = 4
        
        # Extract optional parameters
        if include_quarterly:
            params['beta_q'] = array[idx]
            idx += 1
        if include_yearly:
            params['beta_y'] = array[idx]
            idx += 1
        if include_jump:
            params['beta_j'] = array[idx]
            idx += 1
        if include_leverage:
            params['beta_r'] = array[idx]
            idx += 1
            
        # Extract and inverse transform sigma2
        params['sigma2'] = inverse_transform_positive(array[idx])
        
        # Update with any additional kwargs
        params.update(kwargs)
        
        return cls(**params)


@jit(nopython=True, cache=True)
def _compute_har_aggregates(data: np.ndarray, 
                           daily_lags: int = 1,
                           weekly_lags: int = 5, 
                           monthly_lags: int = 22,
                           quarterly_lags: Optional[int] = None,
                           yearly_lags: Optional[int] = None) -> Tuple[np.ndarray, ...]:
    """Compute HAR model aggregates using Numba acceleration.
    
    This function computes the aggregated time series for different time horizons
    used in HAR models. It is accelerated using Numba's JIT compilation for
    improved performance.
    
    Args:
        data: Input time series data
        daily_lags: Number of lags for daily component (default: 1)
        weekly_lags: Number of lags for weekly component (default: 5)
        monthly_lags: Number of lags for monthly component (default: 22)
        quarterly_lags: Number of lags for quarterly component (optional)
        yearly_lags: Number of lags for yearly component (optional)
    
    Returns:
        Tuple of NumPy arrays containing the aggregated time series
    """
    n = len(data)
    
    # Determine the maximum lag needed
    max_lag = monthly_lags
    if quarterly_lags is not None and quarterly_lags > max_lag:
        max_lag = quarterly_lags
    if yearly_lags is not None and yearly_lags > max_lag:
        max_lag = yearly_lags
    
    # Initialize arrays for aggregated series
    # We'll compute these for the full data length and then slice later
    daily = np.zeros(n)
    weekly = np.zeros(n)
    monthly = np.zeros(n)
    
    # Optional components
    if quarterly_lags is not None:
        quarterly = np.zeros(n)
    if yearly_lags is not None:
        yearly = np.zeros(n)
    
    # Compute daily component (lagged values)
    for i in range(daily_lags, n):
        daily[i] = data[i - daily_lags]
    
    # Compute weekly component (average of last weekly_lags values)
    for i in range(weekly_lags, n):
        weekly[i] = np.mean(data[i - weekly_lags:i])
    
    # Compute monthly component (average of last monthly_lags values)
    for i in range(monthly_lags, n):
        monthly[i] = np.mean(data[i - monthly_lags:i])
    
    # Compute quarterly component if requested
    if quarterly_lags is not None:
        for i in range(quarterly_lags, n):
            quarterly[i] = np.mean(data[i - quarterly_lags:i])
    
    # Compute yearly component if requested
    if yearly_lags is not None:
        for i in range(yearly_lags, n):
            yearly[i] = np.mean(data[i - yearly_lags:i])
    
    # Return the aggregated series
    if quarterly_lags is not None and yearly_lags is not None:
        return daily, weekly, monthly, quarterly, yearly
    elif quarterly_lags is not None:
        return daily, weekly, monthly, quarterly
    elif yearly_lags is not None:
        return daily, weekly, monthly, yearly
    else:
        return daily, weekly, monthly


@jit(nopython=True, cache=True)
def _har_forecast(params: np.ndarray, 
                 aggregates: Tuple[np.ndarray, ...],
                 steps: int,
                 include_quarterly: bool = False,
                 include_yearly: bool = False,
                 include_jump: bool = False,
                 include_leverage: bool = False,
                 jump_series: Optional[np.ndarray] = None,
                 return_series: Optional[np.ndarray] = None,
                 daily_lags: int = 1,
                 weekly_lags: int = 5,
                 monthly_lags: int = 22,
                 quarterly_lags: Optional[int] = None,
                 yearly_lags: Optional[int] = None) -> np.ndarray:
    """Generate forecasts from HAR model parameters using Numba acceleration.
    
    This function generates forecasts from HAR model parameters for a specified
    number of steps ahead. It is accelerated using Numba's JIT compilation for
    improved performance.
    
    Args:
        params: Model parameters as a NumPy array
        aggregates: Tuple of aggregated time series
        steps: Number of steps to forecast
        include_quarterly: Whether to include quarterly component
        include_yearly: Whether to include yearly component
        include_jump: Whether to include jump component
        include_leverage: Whether to include leverage component
        jump_series: Jump component time series (required if include_jump is True)
        return_series: Return series for leverage effect (required if include_leverage is True)
        daily_lags: Number of lags for daily component
        weekly_lags: Number of lags for weekly component
        monthly_lags: Number of lags for monthly component
        quarterly_lags: Number of lags for quarterly component
        yearly_lags: Number of lags for yearly component
    
    Returns:
        np.ndarray: Forecasted values
    """
    # Extract parameters
    constant = params[0]
    beta_d = params[1]
    beta_w = params[2]
    beta_m = params[3]
    
    # Track current index
    idx = 4
    
    # Extract optional parameters
    beta_q = 0.0
    if include_quarterly:
        beta_q = params[idx]
        idx += 1
        
    beta_y = 0.0
    if include_yearly:
        beta_y = params[idx]
        idx += 1
        
    beta_j = 0.0
    if include_jump:
        beta_j = params[idx]
        idx += 1
        
    beta_r = 0.0
    if include_leverage:
        beta_r = params[idx]
        idx += 1
    
    # Extract aggregated series
    daily = aggregates[0]
    weekly = aggregates[1]
    monthly = aggregates[2]
    
    agg_idx = 3
    quarterly = None
    if include_quarterly:
        quarterly = aggregates[agg_idx]
        agg_idx += 1
        
    yearly = None
    if include_yearly:
        yearly = aggregates[agg_idx]
    
    # Get the last observed values
    n = len(daily)
    
    # Create array to store forecasts
    forecasts = np.zeros(steps)
    
    # Create extended series for recursive forecasting
    extended_data = np.zeros(n + steps)
    extended_data[:n] = daily  # Use the daily series as the base data
    
    extended_daily = np.zeros(n + steps)
    extended_daily[:n] = daily
    
    extended_weekly = np.zeros(n + steps)
    extended_weekly[:n] = weekly
    
    extended_monthly = np.zeros(n + steps)
    extended_monthly[:n] = monthly
    
    if include_quarterly and quarterly is not None:
        extended_quarterly = np.zeros(n + steps)
        extended_quarterly[:n] = quarterly
        
    if include_yearly and yearly is not None:
        extended_yearly = np.zeros(n + steps)
        extended_yearly[:n] = yearly
    
    # Generate forecasts
    for h in range(steps):
        # Current position in extended series
        t = n + h
        
        # Forecast for step h
        forecast = constant + beta_d * extended_daily[t-1]
        
        # Add weekly component
        if t >= weekly_lags:
            # If we have enough data, compute the average
            extended_weekly[t] = np.mean(extended_data[t-weekly_lags:t])
            forecast += beta_w * extended_weekly[t]
        else:
            # Otherwise, use the last available weekly value
            forecast += beta_w * extended_weekly[n-1]
        
        # Add monthly component
        if t >= monthly_lags:
            # If we have enough data, compute the average
            extended_monthly[t] = np.mean(extended_data[t-monthly_lags:t])
            forecast += beta_m * extended_monthly[t]
        else:
            # Otherwise, use the last available monthly value
            forecast += beta_m * extended_monthly[n-1]
        
        # Add quarterly component if included
        if include_quarterly and quarterly is not None:
            if t >= quarterly_lags:
                # If we have enough data, compute the average
                extended_quarterly[t] = np.mean(extended_data[t-quarterly_lags:t])
                forecast += beta_q * extended_quarterly[t]
            else:
                # Otherwise, use the last available quarterly value
                forecast += beta_q * extended_quarterly[n-1]
        
        # Add yearly component if included
        if include_yearly and yearly is not None:
            if t >= yearly_lags:
                # If we have enough data, compute the average
                extended_yearly[t] = np.mean(extended_data[t-yearly_lags:t])
                forecast += beta_y * extended_yearly[t]
            else:
                # Otherwise, use the last available yearly value
                forecast += beta_y * extended_yearly[n-1]
        
        # Add jump component if included
        if include_jump and jump_series is not None:
            # For simplicity, we'll use the last observed jump value
            # In practice, you might want to model the jump process separately
            forecast += beta_j * jump_series[n-1]
        
        # Add leverage component if included
        if include_leverage and return_series is not None:
            # For simplicity, we'll use the last observed return value
            # In practice, you might want to model the return process separately
            forecast += beta_r * return_series[n-1]
        
        # Store forecast
        forecasts[h] = forecast
        
        # Update extended data for next step
        extended_data[t] = forecast
        extended_daily[t] = forecast
    
    return forecasts


class HARModel(TimeSeriesModel):
    """Heterogeneous Autoregression (HAR) model for time series analysis.
    
    This class implements the HAR model introduced by Corsi (2009), which is
    particularly useful for modeling and forecasting realized volatility. The
    model captures the long-memory properties of volatility through a parsimonious
    specification using different time horizons (daily, weekly, monthly).
    
    The implementation supports various HAR model specifications, including the
    standard HAR model, HAR with leverage effects, HAR with jumps, and customizable
    lag aggregation periods.
    
    Attributes:
        daily_lags: Number of lags for daily component (default: 1)
        weekly_lags: Number of lags for weekly component (default: 5)
        monthly_lags: Number of lags for monthly component (default: 22)
        quarterly_lags: Number of lags for quarterly component (optional)
        yearly_lags: Number of lags for yearly component (optional)
        include_jump: Whether to include jump component (default: False)
        include_leverage: Whether to include leverage effect (default: False)
    """
    
    def __init__(self, 
                daily_lags: int = 1,
                weekly_lags: int = 5, 
                monthly_lags: int = 22,
                quarterly_lags: Optional[int] = None,
                yearly_lags: Optional[int] = None,
                include_jump: bool = False,
                include_leverage: bool = False,
                name: str = "HAR"):
        """Initialize the HAR model.
        
        Args:
            daily_lags: Number of lags for daily component (default: 1)
            weekly_lags: Number of lags for weekly component (default: 5)
            monthly_lags: Number of lags for monthly component (default: 22)
            quarterly_lags: Number of lags for quarterly component (optional)
            yearly_lags: Number of lags for yearly component (optional)
            include_jump: Whether to include jump component (default: False)
            include_leverage: Whether to include leverage effect (default: False)
            name: A descriptive name for the model
        
        Raises:
            ParameterError: If the lag parameters are invalid
        """
        super().__init__(name=name)
        
        # Validate lag parameters
        if daily_lags <= 0:
            raise ParameterError(
                "daily_lags must be positive",
                param_name="daily_lags",
                param_value=daily_lags,
                constraint="Must be positive"
            )
        
        if weekly_lags <= 0:
            raise ParameterError(
                "weekly_lags must be positive",
                param_name="weekly_lags",
                param_value=weekly_lags,
                constraint="Must be positive"
            )
        
        if monthly_lags <= 0:
            raise ParameterError(
                "monthly_lags must be positive",
                param_name="monthly_lags",
                param_value=monthly_lags,
                constraint="Must be positive"
            )
        
        if quarterly_lags is not None and quarterly_lags <= 0:
            raise ParameterError(
                "quarterly_lags must be positive",
                param_name="quarterly_lags",
                param_value=quarterly_lags,
                constraint="Must be positive"
            )
        
        if yearly_lags is not None and yearly_lags <= 0:
            raise ParameterError(
                "yearly_lags must be positive",
                param_name="yearly_lags",
                param_value=yearly_lags,
                constraint="Must be positive"
            )
        
        # Check that lags are in ascending order
        if weekly_lags <= daily_lags:
            raise ParameterError(
                "weekly_lags must be greater than daily_lags",
                param_name="weekly_lags",
                param_value=weekly_lags,
                constraint=f"Must be greater than daily_lags ({daily_lags})"
            )
        
        if monthly_lags <= weekly_lags:
            raise ParameterError(
                "monthly_lags must be greater than weekly_lags",
                param_name="monthly_lags",
                param_value=monthly_lags,
                constraint=f"Must be greater than weekly_lags ({weekly_lags})"
            )
        
        if quarterly_lags is not None and quarterly_lags <= monthly_lags:
            raise ParameterError(
                "quarterly_lags must be greater than monthly_lags",
                param_name="quarterly_lags",
                param_value=quarterly_lags,
                constraint=f"Must be greater than monthly_lags ({monthly_lags})"
            )
        
        if yearly_lags is not None and yearly_lags <= (quarterly_lags or monthly_lags):
            raise ParameterError(
                "yearly_lags must be greater than quarterly_lags or monthly_lags",
                param_name="yearly_lags",
                param_value=yearly_lags,
                constraint=f"Must be greater than quarterly_lags or monthly_lags"
            )
        
        # Store model parameters
        self.daily_lags = daily_lags
        self.weekly_lags = weekly_lags
        self.monthly_lags = monthly_lags
        self.quarterly_lags = quarterly_lags
        self.yearly_lags = yearly_lags
        self.include_jump = include_jump
        self.include_leverage = include_leverage
        
        # Initialize additional attributes
        self._jump_series: Optional[np.ndarray] = None
        self._return_series: Optional[np.ndarray] = None
        self._aggregates: Optional[Tuple[np.ndarray, ...]] = None
    
    def _prepare_regressors(self, 
                           data: np.ndarray,
                           jump_series: Optional[np.ndarray] = None,
                           return_series: Optional[np.ndarray] = None) -> Tuple[np.ndarray, int]:
        """Prepare regressors for HAR model estimation.
        
        This method computes the aggregated time series for different time horizons
        and prepares the regressor matrix for model estimation.
        
        Args:
            data: Input time series data
            jump_series: Jump component time series (required if include_jump is True)
            return_series: Return series for leverage effect (required if include_leverage is True)
        
        Returns:
            Tuple[np.ndarray, int]: Regressor matrix and effective sample size
        
        Raises:
            ValueError: If required series are missing or have invalid dimensions
        """
        # Compute aggregated series
        aggregates = _compute_har_aggregates(
            data, 
            self.daily_lags, 
            self.weekly_lags, 
            self.monthly_lags,
            self.quarterly_lags,
            self.yearly_lags
        )
        
        # Store aggregates for later use
        self._aggregates = aggregates
        
        # Determine the maximum lag needed
        max_lag = self.monthly_lags
        if self.quarterly_lags is not None and self.quarterly_lags > max_lag:
            max_lag = self.quarterly_lags
        if self.yearly_lags is not None and self.yearly_lags > max_lag:
            max_lag = self.yearly_lags
        
        # Extract aggregated series
        daily = aggregates[0]
        weekly = aggregates[1]
        monthly = aggregates[2]
        
        agg_idx = 3
        quarterly = None
        if self.quarterly_lags is not None:
            quarterly = aggregates[agg_idx]
            agg_idx += 1
            
        yearly = None
        if self.yearly_lags is not None:
            yearly = aggregates[agg_idx]
        
        # Determine effective sample size
        n = len(data)
        effective_sample = n - max_lag
        
        # Prepare regressor matrix
        X = np.ones((effective_sample, 1))  # Start with constant term
        
        # Add daily component
        X = np.column_stack((X, daily[max_lag:]))
        
        # Add weekly component
        X = np.column_stack((X, weekly[max_lag:]))
        
        # Add monthly component
        X = np.column_stack((X, monthly[max_lag:]))
        
        # Add quarterly component if requested
        if self.quarterly_lags is not None and quarterly is not None:
            X = np.column_stack((X, quarterly[max_lag:]))
        
        # Add yearly component if requested
        if self.yearly_lags is not None and yearly is not None:
            X = np.column_stack((X, yearly[max_lag:]))
        
        # Add jump component if requested
        if self.include_jump:
            if jump_series is None:
                raise ValueError(
                    "jump_series must be provided when include_jump is True"
                )
            if len(jump_series) != n:
                raise DimensionError(
                    f"jump_series length ({len(jump_series)}) must match data length ({n})",
                    array_name="jump_series",
                    expected_shape=f"({n},)",
                    actual_shape=jump_series.shape
                )
            X = np.column_stack((X, jump_series[max_lag:]))
            self._jump_series = jump_series
        
        # Add leverage component if requested
        if self.include_leverage:
            if return_series is None:
                raise ValueError(
                    "return_series must be provided when include_leverage is True"
                )
            if len(return_series) != n:
                raise DimensionError(
                    f"return_series length ({len(return_series)}) must match data length ({n})",
                    array_name="return_series",
                    expected_shape=f"({n},)",
                    actual_shape=return_series.shape
                )
            X = np.column_stack((X, return_series[max_lag:]))
            self._return_series = return_series
        
        return X, effective_sample
    
    def fit(self, 
           data: Union[np.ndarray, pd.Series], 
           jump_series: Optional[Union[np.ndarray, pd.Series]] = None,
           return_series: Optional[Union[np.ndarray, pd.Series]] = None,
           **kwargs: Any) -> TimeSeriesResult:
        """Fit the HAR model to the provided data.
        
        This method estimates the HAR model parameters using OLS regression.
        
        Args:
            data: The data to fit the model to (typically realized volatility)
            jump_series: Jump component time series (required if include_jump is True)
            return_series: Return series for leverage effect (required if include_leverage is True)
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
        
        # Convert jump_series and return_series to NumPy arrays if provided
        jump_array = None
        if jump_series is not None:
            if isinstance(jump_series, pd.Series):
                jump_array = jump_series.values
            else:
                jump_array = jump_series
        
        return_array = None
        if return_series is not None:
            if isinstance(return_series, pd.Series):
                return_array = return_series.values
            else:
                return_array = return_series
        
        # Update config with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
        
        try:
            # Prepare regressors
            X, effective_sample = self._prepare_regressors(
                data_array, jump_array, return_array
            )
            
            # Prepare dependent variable (target)
            y = data_array[self.monthly_lags:]
            
            # Estimate model using OLS
            model = sm.OLS(y, X)
            results = model.fit()
            
            # Extract parameter estimates
            params_dict = {
                'constant': results.params[0],
                'beta_d': results.params[1],
                'beta_w': results.params[2],
                'beta_m': results.params[3]
            }
            
            # Track current index
            idx = 4
            
            # Extract optional parameters
            if self.quarterly_lags is not None:
                params_dict['beta_q'] = results.params[idx]
                idx += 1
                
            if self.yearly_lags is not None:
                params_dict['beta_y'] = results.params[idx]
                idx += 1
                
            if self.include_jump:
                params_dict['beta_j'] = results.params[idx]
                idx += 1
                
            if self.include_leverage:
                params_dict['beta_r'] = results.params[idx]
                idx += 1
            
            # Add sigma2 (residual variance)
            params_dict['sigma2'] = results.mse_resid
            
            # Create parameter object
            params = HARParameters(**params_dict)
            
            # Store model attributes
            self._params = params
            self._residuals = results.resid
            self._fitted_values = results.fittedvalues
            self._cov_params = results.cov_params()
            self._fitted = True
            
            # Create result object
            std_errors = results.bse
            
            # Create parameter dictionaries for result object
            param_names = list(params_dict.keys())
            param_values = list(params_dict.values())
            
            params_dict_result = {name: value for name, value in zip(param_names, param_values)}
            std_errors_dict = {name: std_errors[i] for i, name in enumerate(param_names[:-1])}
            std_errors_dict['sigma2'] = np.sqrt(2 * params_dict['sigma2']**2 / effective_sample)
            
            # Compute t-statistics and p-values
            t_stats_dict = {}
            p_values_dict = {}
            
            for name, value in params_dict_result.items():
                if name != 'sigma2':  # Skip sigma2 for t-stats and p-values
                    std_err = std_errors_dict.get(name, np.nan)
                    if std_err > 0:
                        t_stat = value / std_err
                        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), effective_sample - len(param_names)))
                    else:
                        t_stat = np.nan
                        p_value = np.nan
                    
                    t_stats_dict[name] = t_stat
                    p_values_dict[name] = p_value
            
            # Add sigma2 t-stat and p-value
            t_stats_dict['sigma2'] = np.nan  # Not applicable for sigma2
            p_values_dict['sigma2'] = np.nan  # Not applicable for sigma2
            
            # Create result object
            result = TimeSeriesResult(
                model_name=self._name,
                params=params_dict_result,
                std_errors=std_errors_dict,
                t_stats=t_stats_dict,
                p_values=p_values_dict,
                log_likelihood=-0.5 * effective_sample * (np.log(2 * np.pi) + np.log(params_dict['sigma2']) + 1),
                aic=effective_sample * (np.log(2 * np.pi) + np.log(params_dict['sigma2']) + 1) + 2 * len(param_names),
                bic=effective_sample * (np.log(2 * np.pi) + np.log(params_dict['sigma2']) + 1) + np.log(effective_sample) * len(param_names),
                hqic=effective_sample * (np.log(2 * np.pi) + np.log(params_dict['sigma2']) + 1) + 2 * np.log(np.log(effective_sample)) * len(param_names),
                residuals=self._residuals,
                fitted_values=self._fitted_values,
                convergence=True,
                iterations=1,  # OLS is a direct solution, not iterative
                cov_type=self._config.cov_type,
                cov_params=self._cov_params,
                nobs=effective_sample,
                df_model=len(param_names) - 1,  # Exclude sigma2
                df_resid=effective_sample - (len(param_names) - 1)
            )
            
            self._results = result
            return result
            
        except Exception as e:
            raise EstimationError(
                f"HAR model estimation failed: {e}",
                model_type=self._name,
                estimation_method="OLS",
                details=str(e)
            )
    
    def forecast(self, 
                steps: int, 
                exog: Optional[np.ndarray] = None,
                confidence_level: float = 0.95,
                **kwargs: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate forecasts from the fitted HAR model.
        
        This method generates forecasts from the fitted HAR model for a specified
        number of steps ahead.
        
        Args:
            steps: Number of steps to forecast
            exog: Exogenous variables for the forecast period (not used in HAR)
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
            # Check if aggregates are available
            if self._aggregates is None:
                raise ValueError("Aggregates not available. Model may not be properly fitted.")
            
            # Extract parameters
            params = self._params.to_array()
            
            # Generate forecasts
            forecasts = _har_forecast(
                params,
                self._aggregates,
                steps,
                include_quarterly=self.quarterly_lags is not None,
                include_yearly=self.yearly_lags is not None,
                include_jump=self.include_jump,
                include_leverage=self.include_leverage,
                jump_series=self._jump_series,
                return_series=self._return_series,
                daily_lags=self.daily_lags,
                weekly_lags=self.weekly_lags,
                monthly_lags=self.monthly_lags,
                quarterly_lags=self.quarterly_lags,
                yearly_lags=self.yearly_lags
            )
            
            # Compute prediction intervals
            alpha = 1 - confidence_level
            z_value = stats.norm.ppf(1 - alpha / 2)
            
            # For HAR models, prediction variance increases with horizon
            # This is a simplified approach - in practice, you might want to use
            # a more sophisticated method for computing prediction intervals
            prediction_std = np.sqrt(self._params.sigma2 * np.arange(1, steps + 1))
            
            lower_bounds = forecasts - z_value * prediction_std
            upper_bounds = forecasts + z_value * prediction_std
            
            return forecasts, lower_bounds, upper_bounds
            
        except Exception as e:
            raise ForecastError(
                f"HAR forecasting failed: {e}",
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
        """Simulate data from the HAR model.
        
        This method generates simulated data from the HAR model based on the
        estimated parameters.
        
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
            # Determine the maximum lag needed
            max_lag = self.monthly_lags
            if self.quarterly_lags is not None and self.quarterly_lags > max_lag:
                max_lag = self.quarterly_lags
            if self.yearly_lags is not None and self.yearly_lags > max_lag:
                max_lag = self.yearly_lags
            
            # Total periods to simulate (including burn-in)
            total_periods = n_periods + burn
            
            # Initialize simulated data
            simulated = np.zeros(total_periods + max_lag)
            
            # Set initial values if provided
            if initial_values is not None:
                if len(initial_values) < max_lag:
                    raise ValueError(
                        f"initial_values must have length at least {max_lag}, "
                        f"got {len(initial_values)}"
                    )
                simulated[:max_lag] = initial_values[:max_lag]
            else:
                # Use random values for initialization
                simulated[:max_lag] = rng.normal(0, 1, max_lag) * np.sqrt(self._params.sigma2)
            
            # Extract parameters
            constant = self._params.constant
            beta_d = self._params.beta_d
            beta_w = self._params.beta_w
            beta_m = self._params.beta_m
            beta_q = self._params.beta_q
            beta_y = self._params.beta_y
            sigma2 = self._params.sigma2
            
            # Generate simulated data
            for t in range(max_lag, total_periods + max_lag):
                # Compute aggregated components
                daily_component = simulated[t - self.daily_lags]
                weekly_component = np.mean(simulated[t - self.weekly_lags:t])
                monthly_component = np.mean(simulated[t - self.monthly_lags:t])
                
                # Compute expected value
                expected = constant + beta_d * daily_component + beta_w * weekly_component + beta_m * monthly_component
                
                # Add quarterly component if included
                if self.quarterly_lags is not None and beta_q is not None:
                    quarterly_component = np.mean(simulated[t - self.quarterly_lags:t])
                    expected += beta_q * quarterly_component
                
                # Add yearly component if included
                if self.yearly_lags is not None and beta_y is not None:
                    yearly_component = np.mean(simulated[t - self.yearly_lags:t])
                    expected += beta_y * yearly_component
                
                # Add random innovation
                simulated[t] = expected + rng.normal(0, 1) * np.sqrt(sigma2)
            
            # Return simulated data (excluding burn-in and initialization)
            return simulated[max_lag + burn:]
            
        except Exception as e:
            raise SimulationError(
                f"HAR simulation failed: {e}",
                model_type=self._name,
                n_periods=n_periods,
                details=str(e)
            )
    
    def loglikelihood(self, 
                     params: HARParameters, 
                     data: np.ndarray, 
                     **kwargs: Any) -> float:
        """Compute the log-likelihood of the HAR model.
        
        This method computes the log-likelihood of the HAR model given parameters
        and data, assuming Gaussian innovations.
        
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
            
            # Prepare regressors
            jump_series = kwargs.get('jump_series', self._jump_series)
            return_series = kwargs.get('return_series', self._return_series)
            
            X, effective_sample = self._prepare_regressors(
                data, jump_series, return_series
            )
            
            # Prepare dependent variable (target)
            y = data[self.monthly_lags:]
            
            # Extract parameters
            beta = np.array([
                params.constant,
                params.beta_d,
                params.beta_w,
                params.beta_m
            ])
            
            # Add optional parameters
            if self.quarterly_lags is not None and params.beta_q is not None:
                beta = np.append(beta, params.beta_q)
                
            if self.yearly_lags is not None and params.beta_y is not None:
                beta = np.append(beta, params.beta_y)
                
            if self.include_jump and params.beta_j is not None:
                beta = np.append(beta, params.beta_j)
                
            if self.include_leverage and params.beta_r is not None:
                beta = np.append(beta, params.beta_r)
            
            # Compute fitted values
            fitted = X @ beta
            
            # Compute residuals
            residuals = y - fitted
            
            # Compute log-likelihood (assuming Gaussian innovations)
            n = len(residuals)
            sigma2 = params.sigma2
            
            loglik = -0.5 * n * (np.log(2 * np.pi) + np.log(sigma2)) - 0.5 * np.sum(residuals**2) / sigma2
            
            return loglik
            
        except Exception as e:
            raise NumericError(
                f"HAR log-likelihood computation failed: {e}",
                operation="log-likelihood",
                error_type="computation",
                details=str(e)
            )
    
    def summary(self) -> str:
        """Generate a text summary of the HAR model.
        
        Returns:
            str: A formatted string containing the model summary
        
        Raises:
            NotFittedError: If the model has not been fitted
        """
        if not self._fitted:
            model_spec = f"HAR Model (not fitted)"
            model_spec += f"\n  Daily Lags: {self.daily_lags}"
            model_spec += f"\n  Weekly Lags: {self.weekly_lags}"
            model_spec += f"\n  Monthly Lags: {self.monthly_lags}"
            
            if self.quarterly_lags is not None:
                model_spec += f"\n  Quarterly Lags: {self.quarterly_lags}"
                
            if self.yearly_lags is not None:
                model_spec += f"\n  Yearly Lags: {self.yearly_lags}"
                
            model_spec += f"\n  Include Jump: {self.include_jump}"
            model_spec += f"\n  Include Leverage: {self.include_leverage}"
            
            return model_spec
        
        if self._results is None:
            return f"HAR Model (fitted, but no results available)"
        
        # Use the result object's summary method
        return self._results.summary()


class HARJModel(HARModel):
    """HAR-J model with jump component for realized volatility.
    
    This class implements the HAR-J model, which extends the standard HAR model
    by including a jump component. The model is particularly useful for modeling
    and forecasting realized volatility when jumps are present in the data.
    
    Attributes:
        daily_lags: Number of lags for daily component (default: 1)
        weekly_lags: Number of lags for weekly component (default: 5)
        monthly_lags: Number of lags for monthly component (default: 22)
        quarterly_lags: Number of lags for quarterly component (optional)
        yearly_lags: Number of lags for yearly component (optional)
    """
    
    def __init__(self, 
                daily_lags: int = 1,
                weekly_lags: int = 5, 
                monthly_lags: int = 22,
                quarterly_lags: Optional[int] = None,
                yearly_lags: Optional[int] = None,
                name: str = "HAR-J"):
        """Initialize the HAR-J model.
        
        Args:
            daily_lags: Number of lags for daily component (default: 1)
            weekly_lags: Number of lags for weekly component (default: 5)
            monthly_lags: Number of lags for monthly component (default: 22)
            quarterly_lags: Number of lags for quarterly component (optional)
            yearly_lags: Number of lags for yearly component (optional)
            name: A descriptive name for the model
        """
        super().__init__(
            daily_lags=daily_lags,
            weekly_lags=weekly_lags,
            monthly_lags=monthly_lags,
            quarterly_lags=quarterly_lags,
            yearly_lags=yearly_lags,
            include_jump=True,
            include_leverage=False,
            name=name
        )
    
    def fit(self, 
           data: Union[np.ndarray, pd.Series], 
           jump_series: Union[np.ndarray, pd.Series],
           **kwargs: Any) -> TimeSeriesResult:
        """Fit the HAR-J model to the provided data.
        
        This method estimates the HAR-J model parameters using OLS regression.
        
        Args:
            data: The data to fit the model to (typically realized volatility)
            jump_series: Jump component time series
            **kwargs: Additional keyword arguments for model fitting
        
        Returns:
            TimeSeriesResult: The model estimation results
        
        Raises:
            ValueError: If the data is invalid
            EstimationError: If the model estimation fails
        """
        return super().fit(data, jump_series=jump_series, **kwargs)


class HARLModel(HARModel):
    """HAR-L model with leverage effect for realized volatility.
    
    This class implements the HAR-L model, which extends the standard HAR model
    by including a leverage effect component. The model is particularly useful for
    modeling and forecasting realized volatility when asymmetric effects are present.
    
    Attributes:
        daily_lags: Number of lags for daily component (default: 1)
        weekly_lags: Number of lags for weekly component (default: 5)
        monthly_lags: Number of lags for monthly component (default: 22)
        quarterly_lags: Number of lags for quarterly component (optional)
        yearly_lags: Number of lags for yearly component (optional)
    """
    
    def __init__(self, 
                daily_lags: int = 1,
                weekly_lags: int = 5, 
                monthly_lags: int = 22,
                quarterly_lags: Optional[int] = None,
                yearly_lags: Optional[int] = None,
                name: str = "HAR-L"):
        """Initialize the HAR-L model.
        
        Args:
            daily_lags: Number of lags for daily component (default: 1)
            weekly_lags: Number of lags for weekly component (default: 5)
            monthly_lags: Number of lags for monthly component (default: 22)
            quarterly_lags: Number of lags for quarterly component (optional)
            yearly_lags: Number of lags for yearly component (optional)
            name: A descriptive name for the model
        """
        super().__init__(
            daily_lags=daily_lags,
            weekly_lags=weekly_lags,
            monthly_lags=monthly_lags,
            quarterly_lags=quarterly_lags,
            yearly_lags=yearly_lags,
            include_jump=False,
            include_leverage=True,
            name=name
        )
    
    def fit(self, 
           data: Union[np.ndarray, pd.Series], 
           return_series: Union[np.ndarray, pd.Series],
           **kwargs: Any) -> TimeSeriesResult:
        """Fit the HAR-L model to the provided data.
        
        This method estimates the HAR-L model parameters using OLS regression.
        
        Args:
            data: The data to fit the model to (typically realized volatility)
            return_series: Return series for leverage effect
            **kwargs: Additional keyword arguments for model fitting
        
        Returns:
            TimeSeriesResult: The model estimation results
        
        Raises:
            ValueError: If the data is invalid
            EstimationError: If the model estimation fails
        """
        return super().fit(data, return_series=return_series, **kwargs)


class HARJLModel(HARModel):
    """HAR-JL model with jump and leverage effects for realized volatility.
    
    This class implements the HAR-JL model, which extends the standard HAR model
    by including both jump and leverage effect components. The model is particularly
    useful for modeling and forecasting realized volatility when both jumps and
    asymmetric effects are present.
    
    Attributes:
        daily_lags: Number of lags for daily component (default: 1)
        weekly_lags: Number of lags for weekly component (default: 5)
        monthly_lags: Number of lags for monthly component (default: 22)
        quarterly_lags: Number of lags for quarterly component (optional)
        yearly_lags: Number of lags for yearly component (optional)
    """
    
    def __init__(self, 
                daily_lags: int = 1,
                weekly_lags: int = 5, 
                monthly_lags: int = 22,
                quarterly_lags: Optional[int] = None,
                yearly_lags: Optional[int] = None,
                name: str = "HAR-JL"):
        """Initialize the HAR-JL model.
        
        Args:
            daily_lags: Number of lags for daily component (default: 1)
            weekly_lags: Number of lags for weekly component (default: 5)
            monthly_lags: Number of lags for monthly component (default: 22)
            quarterly_lags: Number of lags for quarterly component (optional)
            yearly_lags: Number of lags for yearly component (optional)
            name: A descriptive name for the model
        """
        super().__init__(
            daily_lags=daily_lags,
            weekly_lags=weekly_lags,
            monthly_lags=monthly_lags,
            quarterly_lags=quarterly_lags,
            yearly_lags=yearly_lags,
            include_jump=True,
            include_leverage=True,
            name=name
        )
    
    def fit(self, 
           data: Union[np.ndarray, pd.Series], 
           jump_series: Union[np.ndarray, pd.Series],
           return_series: Union[np.ndarray, pd.Series],
           **kwargs: Any) -> TimeSeriesResult:
        """Fit the HAR-JL model to the provided data.
        
        This method estimates the HAR-JL model parameters using OLS regression.
        
        Args:
            data: The data to fit the model to (typically realized volatility)
            jump_series: Jump component time series
            return_series: Return series for leverage effect
            **kwargs: Additional keyword arguments for model fitting
        
        Returns:
            TimeSeriesResult: The model estimation results
        
        Raises:
            ValueError: If the data is invalid
            EstimationError: If the model estimation fails
        """
        return super().fit(data, jump_series=jump_series, return_series=return_series, **kwargs)


class HARQModel(HARModel):
    """HAR-Q model with quarterly component for realized volatility.
    
    This class implements the HAR-Q model, which extends the standard HAR model
    by including a quarterly component. The model is particularly useful for
    modeling and forecasting realized volatility with longer-term dependencies.
    
    Attributes:
        daily_lags: Number of lags for daily component (default: 1)
        weekly_lags: Number of lags for weekly component (default: 5)
        monthly_lags: Number of lags for monthly component (default: 22)
        quarterly_lags: Number of lags for quarterly component (default: 66)
    """
    
    def __init__(self, 
                daily_lags: int = 1,
                weekly_lags: int = 5, 
                monthly_lags: int = 22,
                quarterly_lags: int = 66,
                name: str = "HAR-Q"):
        """Initialize the HAR-Q model.
        
        Args:
            daily_lags: Number of lags for daily component (default: 1)
            weekly_lags: Number of lags for weekly component (default: 5)
            monthly_lags: Number of lags for monthly component (default: 22)
            quarterly_lags: Number of lags for quarterly component (default: 66)
            name: A descriptive name for the model
        """
        super().__init__(
            daily_lags=daily_lags,
            weekly_lags=weekly_lags,
            monthly_lags=monthly_lags,
            quarterly_lags=quarterly_lags,
            yearly_lags=None,
            include_jump=False,
            include_leverage=False,
            name=name
        )


class HARYModel(HARModel):
    """HAR-Y model with yearly component for realized volatility.
    
    This class implements the HAR-Y model, which extends the standard HAR model
    by including a yearly component. The model is particularly useful for
    modeling and forecasting realized volatility with very long-term dependencies.
    
    Attributes:
        daily_lags: Number of lags for daily component (default: 1)
        weekly_lags: Number of lags for weekly component (default: 5)
        monthly_lags: Number of lags for monthly component (default: 22)
        yearly_lags: Number of lags for yearly component (default: 252)
    """
    
    def __init__(self, 
                daily_lags: int = 1,
                weekly_lags: int = 5, 
                monthly_lags: int = 22,
                yearly_lags: int = 252,
                name: str = "HAR-Y"):
        """Initialize the HAR-Y model.
        
        Args:
            daily_lags: Number of lags for daily component (default: 1)
            weekly_lags: Number of lags for weekly component (default: 5)
            monthly_lags: Number of lags for monthly component (default: 22)
            yearly_lags: Number of lags for yearly component (default: 252)
            name: A descriptive name for the model
        """
        super().__init__(
            daily_lags=daily_lags,
            weekly_lags=weekly_lags,
            monthly_lags=monthly_lags,
            quarterly_lags=None,
            yearly_lags=yearly_lags,
            include_jump=False,
            include_leverage=False,
            name=name
        )


class HARQYModel(HARModel):
    """HAR-QY model with quarterly and yearly components for realized volatility.
    
    This class implements the HAR-QY model, which extends the standard HAR model
    by including both quarterly and yearly components. The model is particularly
    useful for modeling and forecasting realized volatility with both medium and
    long-term dependencies.
    
    Attributes:
        daily_lags: Number of lags for daily component (default: 1)
        weekly_lags: Number of lags for weekly component (default: 5)
        monthly_lags: Number of lags for monthly component (default: 22)
        quarterly_lags: Number of lags for quarterly component (default: 66)
        yearly_lags: Number of lags for yearly component (default: 252)
    """
    
    def __init__(self, 
                daily_lags: int = 1,
                weekly_lags: int = 5, 
                monthly_lags: int = 22,
                quarterly_lags: int = 66,
                yearly_lags: int = 252,
                name: str = "HAR-QY"):
        """Initialize the HAR-QY model.
        
        Args:
            daily_lags: Number of lags for daily component (default: 1)
            weekly_lags: Number of lags for weekly component (default: 5)
            monthly_lags: Number of lags for monthly component (default: 22)
            quarterly_lags: Number of lags for quarterly component (default: 66)
            yearly_lags: Number of lags for yearly component (default: 252)
            name: A descriptive name for the model
        """
        super().__init__(
            daily_lags=daily_lags,
            weekly_lags=weekly_lags,
            monthly_lags=monthly_lags,
            quarterly_lags=quarterly_lags,
            yearly_lags=yearly_lags,
            include_jump=False,
            include_leverage=False,
            name=name
        )
