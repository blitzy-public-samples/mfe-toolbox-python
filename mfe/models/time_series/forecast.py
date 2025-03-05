# mfe/models/time_series/forecast.py
"""
Forecasting functionality for time series models in the MFE Toolbox.

This module provides comprehensive forecasting capabilities for time series models,
including point forecasts, forecast error bands, and simulation-based forecasting.
It supports both deterministic and stochastic forecasting methods, with options for
different types of confidence intervals and prediction bands.

The module implements asynchronous forecasting for long-horizon forecasts, progress
reporting for time-consuming operations, and proper handling of exogenous variables
in forecast generation. It leverages Numba acceleration for computationally intensive
operations to ensure efficient performance even with large datasets or long forecast
horizons.

Key features include:
- Multi-step ahead forecasting with various error band calculation methods
- Monte Carlo simulation for distributional forecasts
- Bootstrap-based prediction intervals
- Scenario analysis through forecast path simulation
- Asynchronous processing for responsive user interfaces
- Progress reporting for long-running forecast operations
- Proper handling of exogenous variables in forecasts
- Numba acceleration for performance-critical calculations
"""

import asyncio
import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any, Callable, Dict, List, Literal, Optional, Protocol, Sequence,
    Tuple, Type, TypeVar, Union, cast, overload
)

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from numba import jit

from mfe.core.base import ModelBase
from mfe.core.parameters import (
    ParameterBase, TimeSeriesParameters, ARMAParameters,
    validate_positive, validate_non_negative, validate_range,
    transform_positive, inverse_transform_positive
)
from mfe.core.exceptions import (
    ForecastError, NotFittedError, NumericError, ParameterError, DimensionError,
    warn_numeric, warn_model
)

# Set up module-level logger
logger = logging.getLogger("mfe.models.time_series.forecast")


class ForecastBandMethod(str, Enum):
    """Enumeration of methods for calculating forecast error bands.

    This enum defines the available methods for calculating forecast error bands,
    including analytical, simulation-based, and bootstrap-based approaches.
    """

    ANALYTICAL = "analytical"  # Analytical error bands based on model properties
    SIMULATION = "simulation"  # Simulation-based error bands using Monte Carlo
    BOOTSTRAP = "bootstrap"    # Bootstrap-based error bands using historical residuals
    ASYMPTOTIC = "asymptotic"  # Asymptotic error bands based on parameter uncertainty

    @classmethod
    def from_string(cls, method: str) -> 'ForecastBandMethod':
        """Convert a string to a ForecastBandMethod enum value.

        Args:
            method: String representation of the method

        Returns:
            ForecastBandMethod: The corresponding enum value

        Raises:
            ValueError: If the string does not match any enum value
        """
        try:
            return cls(method.lower())
        except ValueError:
            valid_methods = [m.value for m in cls]
            raise ValueError(
                f"Invalid forecast band method: {method}. "
                f"Valid methods are: {', '.join(valid_methods)}"
            )


@dataclass
class ForecastResult:
    """Container for time series forecast results.

    This class provides a standardized container for forecast results,
    including point forecasts, prediction intervals, and metadata.

    Attributes:
        point_forecast: Point forecasts for each forecast horizon
        lower_bound: Lower bounds of prediction intervals
        upper_bound: Upper bounds of prediction intervals
        forecast_index: Index for the forecast period (e.g., dates)
        confidence_level: Confidence level used for prediction intervals
        method: Method used to calculate prediction intervals
        simulated_paths: Simulated forecast paths (if simulation method used)
        exog_forecast: Exogenous variables used in the forecast (if any)
        model_name: Name of the model used for forecasting
    """

    point_forecast: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    confidence_level: float
    method: ForecastBandMethod
    forecast_index: Optional[Union[pd.DatetimeIndex, pd.Index]] = None
    simulated_paths: Optional[np.ndarray] = None
    exog_forecast: Optional[np.ndarray] = None
    model_name: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate forecast result after initialization."""
        # Ensure arrays have the same length
        if len(self.point_forecast) != len(self.lower_bound) or len(self.point_forecast) != len(self.upper_bound):
            raise DimensionError(
                "Point forecast and bounds must have the same length",
                array_name="point_forecast, lower_bound, upper_bound",
                expected_shape=f"({len(self.point_forecast)},)",
                actual_shape=(len(self.point_forecast), len(self.lower_bound), len(self.upper_bound))
            )

        # Validate confidence level
        if not 0 < self.confidence_level < 1:
            raise ParameterError(
                f"Confidence level must be between 0 and 1, got {self.confidence_level}",
                param_name="confidence_level",
                param_value=self.confidence_level,
                constraint="0 < confidence_level < 1"
            )

        # Validate forecast index if provided
        if self.forecast_index is not None and len(self.forecast_index) != len(self.point_forecast):
            raise DimensionError(
                "Forecast index length must match forecast length",
                array_name="forecast_index",
                expected_shape=f"({len(self.point_forecast)},)",
                actual_shape=(len(self.forecast_index),)
            )

        # Validate simulated paths if provided
        if self.simulated_paths is not None:
            if self.simulated_paths.shape[1] != len(self.point_forecast):
                raise DimensionError(
                    "Simulated paths must have the same number of time points as the forecast",
                    array_name="simulated_paths",
                    expected_shape=f"(n_paths, {len(self.point_forecast)})",
                    actual_shape=self.simulated_paths.shape
                )

        # Validate exogenous forecast if provided
        if self.exog_forecast is not None:
            if len(self.exog_forecast) != len(self.point_forecast):
                raise DimensionError(
                    "Exogenous forecast length must match forecast length",
                    array_name="exog_forecast",
                    expected_shape=f"({len(self.point_forecast)}, k)",
                    actual_shape=self.exog_forecast.shape
                )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert forecast results to a Pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing forecast results
        """
        # Create DataFrame with point forecast and bounds
        data = {
            "forecast": self.point_forecast,
            f"lower_{int(self.confidence_level * 100)}": self.lower_bound,
            f"upper_{int(self.confidence_level * 100)}": self.upper_bound
        }

        # Use forecast index if available, otherwise create a simple index
        if self.forecast_index is not None:
            df = pd.DataFrame(data, index=self.forecast_index)
        else:
            df = pd.DataFrame(data)

        # Add metadata as attributes
        df.attrs["confidence_level"] = self.confidence_level
        df.attrs["method"] = self.method.value
        if self.model_name:
            df.attrs["model_name"] = self.model_name

        return df

    def plot(self, ax=None, figsize=(10, 6), title=None, **kwargs):
        """Plot forecast results with prediction intervals.

        Args:
            ax: Matplotlib axis to plot on (optional)
            figsize: Figure size if creating a new figure
            title: Plot title (optional)
            **kwargs: Additional keyword arguments for plotting

        Returns:
            matplotlib.axes.Axes: The plot axes
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        # Create x-axis values
        if self.forecast_index is not None:
            x = self.forecast_index
        else:
            x = np.arange(len(self.point_forecast))

        # Plot point forecast
        ax.plot(x, self.point_forecast, 'b-', label='Forecast')

        # Plot prediction intervals
        ax.fill_between(
            x, self.lower_bound, self.upper_bound,
            color='b', alpha=0.2,
            label=f'{int(self.confidence_level * 100)}% Prediction Interval'
        )

        # Add title and labels
        if title:
            ax.set_title(title)
        else:
            model_name = self.model_name or "Time Series"
            ax.set_title(f"{model_name} Forecast")

        ax.set_xlabel('Forecast Horizon')
        ax.set_ylabel('Value')
        ax.legend()

        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)

        return ax


@dataclass
class ForecastConfig:
    """Configuration options for time series forecasting.

    This class provides a standardized way to configure forecasting operations,
    including methods for calculating prediction intervals, simulation parameters,
    and other forecast-related settings.

    Attributes:
        method: Method for calculating forecast error bands
        confidence_level: Confidence level for prediction intervals
        n_simulations: Number of simulations for simulation-based methods
        n_bootstraps: Number of bootstrap replications for bootstrap-based methods
        random_state: Random seed for reproducibility
        use_numba: Whether to use Numba acceleration if available
        display_progress: Whether to display progress during long operations
    """

    method: Union[str, ForecastBandMethod] = ForecastBandMethod.ANALYTICAL
    confidence_level: float = 0.95
    n_simulations: int = 1000
    n_bootstraps: int = 1000
    random_state: Optional[Union[int, np.random.Generator]] = None
    use_numba: bool = True
    display_progress: bool = False

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Convert string method to enum if needed
        if isinstance(self.method, str):
            self.method = ForecastBandMethod.from_string(self.method)

        # Validate confidence level
        if not 0 < self.confidence_level < 1:
            raise ParameterError(
                f"Confidence level must be between 0 and 1, got {self.confidence_level}",
                param_name="confidence_level",
                param_value=self.confidence_level,
                constraint="0 < confidence_level < 1"
            )

        # Validate simulation parameters
        if self.n_simulations <= 0:
            raise ParameterError(
                f"Number of simulations must be positive, got {self.n_simulations}",
                param_name="n_simulations",
                param_value=self.n_simulations,
                constraint="n_simulations > 0"
            )

        # Validate bootstrap parameters
        if self.n_bootstraps <= 0:
            raise ParameterError(
                f"Number of bootstrap replications must be positive, got {self.n_bootstraps}",
                param_name="n_bootstraps",
                param_value=self.n_bootstraps,
                constraint="n_bootstraps > 0"
            )


class ProgressCallback(Protocol):
    """Protocol for progress callback functions.

    This protocol defines the interface for callback functions that report
    progress during long-running forecast operations.
    """

    def __call__(self, progress: float, message: str) -> None:
        """Report progress of a long-running operation.

        Args:
            progress: Progress as a fraction between 0 and 1
            message: Description of the current progress state
        """
        ...


@jit(nopython=True, cache=True)
def _forecast_arma_numba(ar_params, ma_params, constant, data, steps, sigma2):
    """Numba-accelerated ARMA forecasting function.

    This function implements multi-step ahead forecasting for ARMA models
    using Numba's JIT compilation for improved performance.

    Args:
        ar_params: Autoregressive parameters
        ma_params: Moving average parameters
        constant: Constant term
        data: Historical data
        steps: Number of steps to forecast
        sigma2: Innovation variance

    Returns:
        np.ndarray: Point forecasts
    """
    ar_order = len(ar_params)
    ma_order = len(ma_params)

    # Initialize forecast array
    forecasts = np.zeros(steps)

    # Compute residuals for MA terms
    residuals = np.zeros(len(data))
    for t in range(max(ar_order, ma_order), len(data)):
        residuals[t] = data[t] - constant
        for i in range(ar_order):
            if t - i - 1 >= 0:
                residuals[t] -= ar_params[i] * data[t - i - 1]

    # Generate forecasts
    for h in range(steps):
        # Initialize with constant term
        forecasts[h] = constant

        # Add AR terms
        for i in range(ar_order):
            if h - i - 1 >= 0:
                # Use previous forecasts
                forecasts[h] += ar_params[i] * forecasts[h - i - 1]
            else:
                # Use historical data
                idx = len(data) - 1 - i + h
                if idx >= 0:
                    forecasts[h] += ar_params[i] * data[idx]

        # Add MA terms (only for h <= ma_order)
        if h < ma_order:
            for j in range(h, ma_order):
                idx = len(data) - 1 - (j - h)
                if idx >= 0:
                    forecasts[h] += ma_params[j] * residuals[idx]

    return forecasts


@jit(nopython=True, cache=True)
def _simulate_forecast_paths_numba(ar_params, ma_params, constant, data, steps, n_paths, sigma2, seed):
    """Numba-accelerated simulation of forecast paths for ARMA models.

    This function generates multiple simulated forecast paths for ARMA models
    using Numba's JIT compilation for improved performance.

    Args:
        ar_params: Autoregressive parameters
        ma_params: Moving average parameters
        constant: Constant term
        data: Historical data
        steps: Number of steps to forecast
        n_paths: Number of simulation paths
        sigma2: Innovation variance
        seed: Random seed for reproducibility

    Returns:
        np.ndarray: Simulated forecast paths with shape (n_paths, steps)
    """
    ar_order = len(ar_params)
    ma_order = len(ma_params)

    # Initialize random number generator
    np.random.seed(seed)

    # Initialize forecast paths array
    paths = np.zeros((n_paths, steps))

    # Compute historical residuals for MA terms
    residuals = np.zeros(len(data))
    for t in range(max(ar_order, ma_order), len(data)):
        residuals[t] = data[t] - constant
        for i in range(ar_order):
            if t - i - 1 >= 0:
                residuals[t] -= ar_params[i] * data[t - i - 1]

    # Generate forecast paths
    for p in range(n_paths):
        # Generate random innovations
        innovations = np.random.normal(0, np.sqrt(sigma2), steps)

        # Initialize path-specific residuals
        path_residuals = np.zeros(steps)

        # Generate forecasts for this path
        for h in range(steps):
            # Initialize with constant term and innovation
            paths[p, h] = constant + innovations[h]

            # Add AR terms
            for i in range(ar_order):
                if h - i - 1 >= 0:
                    # Use previous forecasts
                    paths[p, h] += ar_params[i] * paths[p, h - i - 1]
                else:
                    # Use historical data
                    idx = len(data) - 1 - i + h
                    if idx >= 0:
                        paths[p, h] += ar_params[i] * data[idx]

            # Add MA terms
            for j in range(ma_order):
                if h - j - 1 >= 0:
                    # Use previous innovations
                    paths[p, h] += ma_params[j] * path_residuals[h - j - 1]
                else:
                    # Use historical residuals
                    idx = len(residuals) - 1 - j + h
                    if idx >= 0:
                        paths[p, h] += ma_params[j] * residuals[idx]

            # Store residual for this step
            path_residuals[h] = innovations[h]

    return paths


def forecast_arma(
    ar_params: np.ndarray,
    ma_params: np.ndarray,
    constant: float,
    data: np.ndarray,
    steps: int,
    sigma2: float,
    exog: Optional[np.ndarray] = None,
    exog_params: Optional[np.ndarray] = None,
    use_numba: bool = True
) -> np.ndarray:
    """Generate point forecasts for an ARMA model.

    This function computes multi-step ahead forecasts for an ARMA model
    based on the provided parameters and historical data.

    Args:
        ar_params: Autoregressive parameters
        ma_params: Moving average parameters
        constant: Constant term
        data: Historical data
        steps: Number of steps to forecast
        sigma2: Innovation variance
        exog: Exogenous variables for the forecast period
        exog_params: Parameters for exogenous variables
        use_numba: Whether to use Numba acceleration

    Returns:
        np.ndarray: Point forecasts

    Raises:
        ValueError: If the input parameters are invalid
        NumericError: If numerical issues occur during forecasting
    """
    # Validate inputs
    if steps <= 0:
        raise ValueError(f"Steps must be positive, got {steps}")

    if len(data) < max(len(ar_params), len(ma_params)):
        raise ValueError(
            f"Data length ({len(data)}) must be at least as large as "
            f"max(AR order, MA order) = {max(len(ar_params), len(ma_params))}"
        )

    # Check for exogenous variables
    if exog is not None:
        if exog_params is None:
            raise ValueError("exog_params must be provided when exog is provided")

        if len(exog) != steps:
            raise ValueError(
                f"Length of exog ({len(exog)}) must match steps ({steps})"
            )

        if exog.ndim == 1:
            if len(exog_params) != 1:
                raise ValueError(
                    f"Length of exog_params ({len(exog_params)}) must match "
                    f"number of exogenous variables (1)"
                )
        else:
            if len(exog_params) != exog.shape[1]:
                raise ValueError(
                    f"Length of exog_params ({len(exog_params)}) must match "
                    f"number of exogenous variables ({exog.shape[1]})"
                )

    try:
        # Use Numba-accelerated function if available and requested
        if use_numba:
            try:
                forecasts = _forecast_arma_numba(
                    ar_params, ma_params, constant, data, steps, sigma2
                )

                # Add exogenous effects if provided
                if exog is not None and exog_params is not None:
                    if exog.ndim == 1:
                        forecasts += exog * exog_params[0]
                    else:
                        forecasts += np.dot(exog, exog_params)

                return forecasts
            except Exception as e:
                logger.warning(f"Numba acceleration failed: {e}. Falling back to pure Python implementation.")

        # Pure Python implementation as fallback
        ar_order = len(ar_params)
        ma_order = len(ma_params)

        # Initialize forecast array
        forecasts = np.zeros(steps)

        # Compute residuals for MA terms
        residuals = np.zeros(len(data))
        for t in range(max(ar_order, ma_order), len(data)):
            residuals[t] = data[t] - constant
            for i in range(ar_order):
                if t - i - 1 >= 0:
                    residuals[t] -= ar_params[i] * data[t - i - 1]

        # Generate forecasts
        for h in range(steps):
            # Initialize with constant term
            forecasts[h] = constant

            # Add AR terms
            for i in range(ar_order):
                if h - i - 1 >= 0:
                    # Use previous forecasts
                    forecasts[h] += ar_params[i] * forecasts[h - i - 1]
                else:
                    # Use historical data
                    idx = len(data) - 1 - i + h
                    if idx >= 0:
                        forecasts[h] += ar_params[i] * data[idx]

            # Add MA terms (only for h <= ma_order)
            if h < ma_order:
                for j in range(h, ma_order):
                    idx = len(data) - 1 - (j - h)
                    if idx >= 0:
                        forecasts[h] += ma_params[j] * residuals[idx]

            # Add exogenous effects if provided
            if exog is not None and exog_params is not None:
                if exog.ndim == 1:
                    forecasts[h] += exog[h] * exog_params[0]
                else:
                    forecasts[h] += np.dot(exog[h], exog_params)

        return forecasts

    except Exception as e:
        raise NumericError(
            f"Error in ARMA forecasting: {e}",
            operation="ARMA forecasting",
            error_type="computation",
            details=str(e)
        )


def forecast_error_variance_arma(
    ar_params: np.ndarray,
    ma_params: np.ndarray,
    sigma2: float,
    steps: int
) -> np.ndarray:
    """Compute forecast error variances for an ARMA model.

    This function calculates the analytical forecast error variances
    for an ARMA model at different forecast horizons.

    Args:
        ar_params: Autoregressive parameters
        ma_params: Moving average parameters
        sigma2: Innovation variance
        steps: Number of steps to forecast

    Returns:
        np.ndarray: Forecast error variances for each forecast horizon

    Raises:
        ValueError: If the input parameters are invalid
        NumericError: If numerical issues occur during computation
    """
    try:
        # Initialize variance array
        variances = np.zeros(steps)

        # Compute MA representation coefficients (psi weights)
        # For an ARMA model, we need to compute the infinite MA representation
        # We'll approximate it with a sufficiently large number of terms
        max_lag = max(100, steps + 50)  # Ensure enough terms for accuracy
        psi_weights = np.zeros(max_lag)

        # Initialize with MA coefficients
        psi_weights[:len(ma_params)] = ma_params.copy()
        psi_weights[0] = 1.0  # First coefficient is always 1

        # Compute remaining psi weights recursively
        for i in range(1, max_lag):
            for j in range(min(i, len(ar_params))):
                if i - j - 1 >= 0:
                    psi_weights[i] += ar_params[j] * psi_weights[i - j - 1]

        # Compute forecast error variances
        for h in range(steps):
            # For h-step ahead forecast, variance is sigma^2 * sum(psi_j^2) for j=0 to h-1
            variances[h] = sigma2 * np.sum(psi_weights[:h+1]**2)

        return variances

    except Exception as e:
        raise NumericError(
            f"Error in computing ARMA forecast error variances: {e}",
            operation="ARMA forecast error variance",
            error_type="computation",
            details=str(e)
        )


def simulate_forecast_paths(
    ar_params: np.ndarray,
    ma_params: np.ndarray,
    constant: float,
    data: np.ndarray,
    steps: int,
    sigma2: float,
    n_paths: int = 1000,
    exog: Optional[np.ndarray] = None,
    exog_params: Optional[np.ndarray] = None,
    random_state: Optional[Union[int, np.random.Generator]] = None,
    use_numba: bool = True,
    progress_callback: Optional[ProgressCallback] = None
) -> np.ndarray:
    """Simulate multiple forecast paths for an ARMA model.

    This function generates multiple simulated forecast paths for an ARMA model
    using Monte Carlo simulation, which can be used for constructing prediction
    intervals and analyzing forecast uncertainty.

    Args:
        ar_params: Autoregressive parameters
        ma_params: Moving average parameters
        constant: Constant term
        data: Historical data
        steps: Number of steps to forecast
        sigma2: Innovation variance
        n_paths: Number of simulation paths
        exog: Exogenous variables for the forecast period
        exog_params: Parameters for exogenous variables
        random_state: Random seed for reproducibility
        use_numba: Whether to use Numba acceleration
        progress_callback: Callback function for reporting progress

    Returns:
        np.ndarray: Simulated forecast paths with shape (n_paths, steps)

    Raises:
        ValueError: If the input parameters are invalid
        NumericError: If numerical issues occur during simulation
    """
    # Validate inputs
    if steps <= 0:
        raise ValueError(f"Steps must be positive, got {steps}")

    if n_paths <= 0:
        raise ValueError(f"Number of paths must be positive, got {n_paths}")

    if len(data) < max(len(ar_params), len(ma_params)):
        raise ValueError(
            f"Data length ({len(data)}) must be at least as large as "
            f"max(AR order, MA order) = {max(len(ar_params), len(ma_params))}"
        )

    # Check for exogenous variables
    if exog is not None:
        if exog_params is None:
            raise ValueError("exog_params must be provided when exog is provided")

        if len(exog) != steps:
            raise ValueError(
                f"Length of exog ({len(exog)}) must match steps ({steps})"
            )

        if exog.ndim == 1:
            if len(exog_params) != 1:
                raise ValueError(
                    f"Length of exog_params ({len(exog_params)}) must match "
                    f"number of exogenous variables (1)"
                )
        else:
            if len(exog_params) != exog.shape[1]:
                raise ValueError(
                    f"Length of exog_params ({len(exog_params)}) must match "
                    f"number of exogenous variables ({exog.shape[1]})"
                )

    # Set up random number generator
    if isinstance(random_state, np.random.Generator):
        rng = random_state
        seed = None  # For Numba, we'll need to extract a seed
        try:
            # Try to get a seed from the RNG state
            seed = int(rng.bit_generator.state['state']['state'])
        except (AttributeError, KeyError, TypeError):
            # If that fails, generate a random seed
            seed = np.random.randint(0, 2**31 - 1)
    else:
        seed = random_state if random_state is not None else np.random.randint(0, 2**31 - 1)
        rng = np.random.default_rng(seed)

    try:
        # Use Numba-accelerated function if available and requested
        if use_numba and exog is None:  # Numba version doesn't support exog yet
            try:
                paths = _simulate_forecast_paths_numba(
                    ar_params, ma_params, constant, data, steps, n_paths, sigma2, seed
                )
                return paths
            except Exception as e:
                logger.warning(f"Numba acceleration failed: {e}. Falling back to pure Python implementation.")

        # Pure Python implementation as fallback
        ar_order = len(ar_params)
        ma_order = len(ma_params)

        # Initialize forecast paths array
        paths = np.zeros((n_paths, steps))

        # Compute historical residuals for MA terms
        residuals = np.zeros(len(data))
        for t in range(max(ar_order, ma_order), len(data)):
            residuals[t] = data[t] - constant
            for i in range(ar_order):
                if t - i - 1 >= 0:
                    residuals[t] -= ar_params[i] * data[t - i - 1]

        # Generate forecast paths with progress reporting
        for p in range(n_paths):
            # Report progress if callback is provided
            if progress_callback is not None and p % max(1, n_paths // 100) == 0:
                progress_callback(p / n_paths, f"Simulating path {p+1}/{n_paths}")

            # Generate random innovations
            innovations = rng.normal(0, np.sqrt(sigma2), steps)

            # Initialize path-specific residuals
            path_residuals = np.zeros(steps)

            # Generate forecasts for this path
            for h in range(steps):
                # Initialize with constant term and innovation
                paths[p, h] = constant + innovations[h]

                # Add AR terms
                for i in range(ar_order):
                    if h - i - 1 >= 0:
                        # Use previous forecasts
                        paths[p, h] += ar_params[i] * paths[p, h - i - 1]
                    else:
                        # Use historical data
                        idx = len(data) - 1 - i + h
                        if idx >= 0:
                            paths[p, h] += ar_params[i] * data[idx]

                # Add MA terms
                for j in range(ma_order):
                    if h - j - 1 >= 0:
                        # Use previous innovations
                        paths[p, h] += ma_params[j] * path_residuals[h - j - 1]
                    else:
                        # Use historical residuals
                        idx = len(residuals) - 1 - j + h
                        if idx >= 0:
                            paths[p, h] += ma_params[j] * residuals[idx]

                # Add exogenous effects if provided
                if exog is not None and exog_params is not None:
                    if exog.ndim == 1:
                        paths[p, h] += exog[h] * exog_params[0]
                    else:
                        paths[p, h] += np.dot(exog[h], exog_params)

                # Store residual for this step
                path_residuals[h] = innovations[h]

        # Final progress report
        if progress_callback is not None:
            progress_callback(1.0, f"Completed {n_paths} simulation paths")

        return paths

    except Exception as e:
        raise NumericError(
            f"Error in simulating forecast paths: {e}",
            operation="Forecast path simulation",
            error_type="computation",
            details=str(e)
        )


def bootstrap_forecast_paths(
    ar_params: np.ndarray,
    ma_params: np.ndarray,
    constant: float,
    data: np.ndarray,
    residuals: np.ndarray,
    steps: int,
    n_paths: int = 1000,
    exog: Optional[np.ndarray] = None,
    exog_params: Optional[np.ndarray] = None,
    random_state: Optional[Union[int, np.random.Generator]] = None,
    progress_callback: Optional[ProgressCallback] = None
) -> np.ndarray:
    """Generate bootstrap forecast paths for an ARMA model.

    This function generates multiple forecast paths using bootstrap resampling
    of historical residuals, which can be used for constructing prediction
    intervals that don't rely on distributional assumptions.

    Args:
        ar_params: Autoregressive parameters
        ma_params: Moving average parameters
        constant: Constant term
        data: Historical data
        residuals: Historical residuals
        steps: Number of steps to forecast
        n_paths: Number of bootstrap paths
        exog: Exogenous variables for the forecast period
        exog_params: Parameters for exogenous variables
        random_state: Random seed for reproducibility
        progress_callback: Callback function for reporting progress

    Returns:
        np.ndarray: Bootstrapped forecast paths with shape (n_paths, steps)

    Raises:
        ValueError: If the input parameters are invalid
        NumericError: If numerical issues occur during bootstrap
    """
    # Validate inputs
    if steps <= 0:
        raise ValueError(f"Steps must be positive, got {steps}")

    if n_paths <= 0:
        raise ValueError(f"Number of paths must be positive, got {n_paths}")

    if len(data) < max(len(ar_params), len(ma_params)):
        raise ValueError(
            f"Data length ({len(data)}) must be at least as large as "
            f"max(AR order, MA order) = {max(len(ar_params), len(ma_params))}"
        )

    if len(residuals) != len(data):
        raise ValueError(
            f"Residuals length ({len(residuals)}) must match data length ({len(data)})"
        )

    # Check for exogenous variables
    if exog is not None:
        if exog_params is None:
            raise ValueError("exog_params must be provided when exog is provided")

        if len(exog) != steps:
            raise ValueError(
                f"Length of exog ({len(exog)}) must match steps ({steps})"
            )

        if exog.ndim == 1:
            if len(exog_params) != 1:
                raise ValueError(
                    f"Length of exog_params ({len(exog_params)}) must match "
                    f"number of exogenous variables (1)"
                )
        else:
            if len(exog_params) != exog.shape[1]:
                raise ValueError(
                    f"Length of exog_params ({len(exog_params)}) must match "
                    f"number of exogenous variables ({exog.shape[1]})"
                )

    # Set up random number generator
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    try:
        ar_order = len(ar_params)
        ma_order = len(ma_params)

        # Initialize forecast paths array
        paths = np.zeros((n_paths, steps))

        # Filter residuals to remove initial values that might be zeros
        valid_residuals = residuals[max(ar_order, ma_order):]

        # Generate bootstrap forecast paths
        for p in range(n_paths):
            # Report progress if callback is provided
            if progress_callback is not None and p % max(1, n_paths // 100) == 0:
                progress_callback(p / n_paths, f"Bootstrapping path {p+1}/{n_paths}")

            # Generate bootstrapped residuals by resampling historical residuals
            bootstrap_residuals = rng.choice(valid_residuals, size=steps, replace=True)

            # Generate forecasts for this path
            for h in range(steps):
                # Initialize with constant term and bootstrapped residual
                paths[p, h] = constant + bootstrap_residuals[h]

                # Add AR terms
                for i in range(ar_order):
                    if h - i - 1 >= 0:
                        # Use previous forecasts
                        paths[p, h] += ar_params[i] * paths[p, h - i - 1]
                    else:
                        # Use historical data
                        idx = len(data) - 1 - i + h
                        if idx >= 0:
                            paths[p, h] += ar_params[i] * data[idx]

                # Add MA terms
                for j in range(ma_order):
                    if h - j - 1 >= 0:
                        # Use previous bootstrapped residuals
                        paths[p, h] += ma_params[j] * bootstrap_residuals[h - j - 1]
                    else:
                        # Use historical residuals
                        idx = len(residuals) - 1 - j + h
                        if idx >= 0:
                            paths[p, h] += ma_params[j] * residuals[idx]

                # Add exogenous effects if provided
                if exog is not None and exog_params is not None:
                    if exog.ndim == 1:
                        paths[p, h] += exog[h] * exog_params[0]
                    else:
                        paths[p, h] += np.dot(exog[h], exog_params)

        # Final progress report
        if progress_callback is not None:
            progress_callback(1.0, f"Completed {n_paths} bootstrap paths")

        return paths

    except Exception as e:
        raise NumericError(
            f"Error in bootstrapping forecast paths: {e}",
            operation="Bootstrap forecast",
            error_type="computation",
            details=str(e)
        )


def generate_forecast_bands(
    point_forecast: np.ndarray,
    paths: Optional[np.ndarray] = None,
    error_variance: Optional[np.ndarray] = None,
    confidence_level: float = 0.95,
    method: ForecastBandMethod = ForecastBandMethod.ANALYTICAL
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate forecast error bands based on the specified method.

    This function computes lower and upper bounds for forecast prediction
    intervals using various methods, including analytical formulas,
    simulation-based quantiles, or bootstrap-based quantiles.

    Args:
        point_forecast: Point forecasts for each forecast horizon
        paths: Simulated or bootstrapped forecast paths (required for simulation/bootstrap methods)
        error_variance: Forecast error variances (required for analytical method)
        confidence_level: Confidence level for prediction intervals
        method: Method for calculating prediction intervals

    Returns:
        Tuple[np.ndarray, np.ndarray]: Lower and upper bounds of prediction intervals

    Raises:
        ValueError: If the inputs are invalid or incompatible with the chosen method
        NumericError: If numerical issues occur during computation
    """
    # Validate inputs
    if not 0 < confidence_level < 1:
        raise ValueError(
            f"Confidence level must be between 0 and 1, got {confidence_level}"
        )

    # Validate method-specific requirements
    if method in [ForecastBandMethod.SIMULATION, ForecastBandMethod.BOOTSTRAP]:
        if paths is None:
            raise ValueError(
                f"Simulated paths must be provided for {method.value} method"
            )
        if paths.shape[1] != len(point_forecast):
            raise ValueError(
                f"Number of time points in paths ({paths.shape[1]}) must match "
                f"length of point forecast ({len(point_forecast)})"
            )

    if method == ForecastBandMethod.ANALYTICAL:
        if error_variance is None:
            raise ValueError(
                "Error variance must be provided for analytical method"
            )
        if len(error_variance) != len(point_forecast):
            raise ValueError(
                f"Length of error variance ({len(error_variance)}) must match "
                f"length of point forecast ({len(point_forecast)})"
            )

    try:
        # Compute prediction intervals based on the specified method
        if method in [ForecastBandMethod.SIMULATION, ForecastBandMethod.BOOTSTRAP]:
            # Compute quantiles from simulated/bootstrapped paths
            alpha = (1 - confidence_level) / 2
            lower_bound = np.quantile(paths, alpha, axis=0)
            upper_bound = np.quantile(paths, 1 - alpha, axis=0)

        elif method == ForecastBandMethod.ANALYTICAL:
            # Compute analytical prediction intervals
            z_value = stats.norm.ppf(1 - (1 - confidence_level) / 2)
            std_error = np.sqrt(error_variance)
            lower_bound = point_forecast - z_value * std_error
            upper_bound = point_forecast + z_value * std_error

        elif method == ForecastBandMethod.ASYMPTOTIC:
            # This would require parameter covariance matrix and forecast Jacobian
            # For simplicity, we'll fall back to analytical method with a warning
            if error_variance is None:
                raise ValueError(
                    "Error variance must be provided for asymptotic method"
                )

            warn_numeric(
                "Asymptotic method not fully implemented, falling back to analytical method",
                operation="Forecast bands",
                issue="Method not implemented"
            )

            z_value = stats.norm.ppf(1 - (1 - confidence_level) / 2)
            std_error = np.sqrt(error_variance)
            lower_bound = point_forecast - z_value * std_error
            upper_bound = point_forecast + z_value * std_error

        else:
            raise ValueError(f"Unknown method: {method}")

        return lower_bound, upper_bound

    except Exception as e:
        raise NumericError(
            f"Error in generating forecast bands: {e}",
            operation="Forecast bands",
            error_type="computation",
            details=str(e)
        )


def generate_forecast_scenarios(
    point_forecast: np.ndarray,
    paths: np.ndarray,
    n_scenarios: int = 5,
    method: str = "quantile",
    random_state: Optional[Union[int, np.random.Generator]] = None
) -> Tuple[List[np.ndarray], List[str]]:
    """Generate representative forecast scenarios from simulated paths.

    This function selects a small number of representative forecast paths
    from a larger set of simulated paths, which can be used for scenario
    analysis and visualization of possible future trajectories.

    Args:
        point_forecast: Point forecasts for each forecast horizon
        paths: Simulated forecast paths
        n_scenarios: Number of scenarios to generate
        method: Method for selecting scenarios ("quantile", "cluster", or "random")
        random_state: Random seed for reproducibility

    Returns:
        Tuple[List[np.ndarray], List[str]]: Selected scenario paths and their descriptions

    Raises:
        ValueError: If the inputs are invalid
        NumericError: If numerical issues occur during computation
    """
    # Validate inputs
    if n_scenarios <= 0:
        raise ValueError(f"Number of scenarios must be positive, got {n_scenarios}")

    if n_scenarios > paths.shape[0]:
        raise ValueError(
            f"Number of scenarios ({n_scenarios}) cannot exceed "
            f"number of paths ({paths.shape[0]})"
        )

    if paths.shape[1] != len(point_forecast):
        raise ValueError(
            f"Number of time points in paths ({paths.shape[1]}) must match "
            f"length of point forecast ({len(point_forecast)})"
        )

    # Set up random number generator
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    try:
        # Select scenarios based on the specified method
        if method.lower() == "quantile":
            # Select paths based on quantiles of the final value
            final_values = paths[:, -1]
            quantiles = np.linspace(0, 1, n_scenarios)
            scenario_indices = [
                np.argmin(np.abs(final_values - np.quantile(final_values, q)))
                for q in quantiles
            ]

            # Create scenario descriptions
            descriptions = [
                f"Quantile {int(q * 100)}%" for q in quantiles
            ]

        elif method.lower() == "cluster":
            # Use k-means clustering to find representative paths
            try:
                from sklearn.cluster import KMeans

                # Normalize paths for clustering
                normalized_paths = (paths - np.mean(paths, axis=0)) / np.std(paths, axis=0)

                # Apply k-means clustering
                kmeans = KMeans(n_clusters=n_scenarios, random_state=random_state)
                clusters = kmeans.fit_predict(normalized_paths)

                # Select the path closest to each cluster center
                scenario_indices = []
                for i in range(n_scenarios):
                    cluster_paths = np.where(clusters == i)[0]
                    if len(cluster_paths) > 0:
                        # Find path closest to cluster center
                        distances = np.sum(
                            (normalized_paths[cluster_paths] - kmeans.cluster_centers_[i])**2,
                            axis=1
                        )
                        closest_idx = cluster_paths[np.argmin(distances)]
                        scenario_indices.append(closest_idx)

                # If we don't have enough scenarios, add random ones
                while len(scenario_indices) < n_scenarios:
                    idx = rng.integers(0, paths.shape[0])
                    if idx not in scenario_indices:
                        scenario_indices.append(idx)

                # Create scenario descriptions
                descriptions = [
                    f"Cluster {i+1}" for i in range(n_scenarios)
                ]

            except ImportError:
                # Fall back to random selection if sklearn is not available
                logger.warning(
                    "sklearn not available for clustering. Falling back to random selection."
                )
                scenario_indices = rng.choice(
                    paths.shape[0], size=n_scenarios, replace=False
                )
                descriptions = [
                    f"Random Scenario {i+1}" for i in range(n_scenarios)
                ]

        elif method.lower() == "random":
            # Randomly select paths
            scenario_indices = rng.choice(
                paths.shape[0], size=n_scenarios, replace=False
            )

            # Create scenario descriptions
            descriptions = [
                f"Random Scenario {i+1}" for i in range(n_scenarios)
            ]

        else:
            raise ValueError(f"Unknown method: {method}")

        # Extract selected scenarios
        scenarios = [paths[idx] for idx in scenario_indices]

        return scenarios, descriptions

    except Exception as e:
        raise NumericError(
            f"Error in generating forecast scenarios: {e}",
            operation="Forecast scenarios",
            error_type="computation",
            details=str(e)
        )


async def forecast_model_async(
    model: Any,
    steps: int,
    exog: Optional[np.ndarray] = None,
    config: Optional[ForecastConfig] = None,
    progress_callback: Optional[ProgressCallback] = None
) -> ForecastResult:
    """Generate forecasts from a time series model asynchronously.

    This function provides an asynchronous interface for generating forecasts
    from a time series model, which is useful for maintaining UI responsiveness
    during long-running forecast operations.

    Args:
        model: Time series model object with forecast method
        steps: Number of steps to forecast
        exog: Exogenous variables for the forecast period
        config: Forecast configuration options
        progress_callback: Callback function for reporting progress

    Returns:
        ForecastResult: Forecast results including point forecasts and prediction intervals

    Raises:
        NotFittedError: If the model has not been fitted
        ValueError: If the forecast parameters are invalid
        ForecastError: If the forecasting fails
    """
    # Validate model
    if not hasattr(model, 'forecast'):
        raise ValueError(f"Model must have a forecast method")

    if not hasattr(model, '_fitted') or not model._fitted:
        raise NotFittedError(
            "Model has not been fitted. Call fit() first.",
            model_type=getattr(model, '_name', type(model).__name__),
            operation="forecast"
        )

    # Use default config if not provided
    if config is None:
        config = ForecastConfig()

    try:
        # Create a coroutine that runs the forecast method in a thread pool
        loop = asyncio.get_event_loop()

        # Report initial progress
        if progress_callback is not None:
            progress_callback(0.0, "Starting forecast computation")

        # Run the forecast method in a thread pool
        result = await loop.run_in_executor(
            None,
            lambda: model.forecast(
                steps=steps,
                exog=exog,
                confidence_level=config.confidence_level,
                method=config.method.value,
                n_simulations=config.n_simulations,
                n_bootstraps=config.n_bootstraps,
                random_state=config.random_state,
                use_numba=config.use_numba,
                progress_callback=progress_callback
            )
        )

        # Report final progress
        if progress_callback is not None:
            progress_callback(1.0, "Forecast computation complete")

        return result

    except Exception as e:
        raise ForecastError(
            f"Asynchronous forecasting failed: {e}",
            model_type=getattr(model, '_name', type(model).__name__),
            horizon=steps,
            details=str(e)
        )


def extend_forecast_index(
    original_index: Union[pd.DatetimeIndex, pd.Index],
    steps: int,
    freq: Optional[str] = None
) -> Union[pd.DatetimeIndex, pd.Index]:
    """Extend a time series index for forecasting.

    This function extends a time series index to include future periods
    for forecasting, handling both date-based and integer-based indices.

    Args:
        original_index: Original time series index
        steps: Number of steps to extend
        freq: Frequency string for DatetimeIndex (optional)

    Returns:
        Union[pd.DatetimeIndex, pd.Index]: Extended index for forecast period

    Raises:
        ValueError: If the index cannot be extended
    """
    if len(original_index) == 0:
        raise ValueError("Original index is empty")

    try:
        if isinstance(original_index, pd.DatetimeIndex):
            # For DatetimeIndex, infer frequency if not provided
            if freq is None:
                freq = original_index.freq

                # If frequency is still None, try to infer it
                if freq is None:
                    freq = pd.infer_freq(original_index)

                # If still None, use the difference between the last two points
                if freq is None and len(original_index) >= 2:
                    freq = original_index[-1] - original_index[-2]

                # If still None, raise an error
                if freq is None:
                    raise ValueError(
                        "Could not infer frequency from index. "
                        "Please provide a frequency using the freq parameter."
                    )

            # Create forecast index
            last_date = original_index[-1]
            forecast_index = pd.date_range(
                start=last_date + freq,
                periods=steps,
                freq=freq
            )

            return forecast_index

        else:
            # For other index types, create a simple range
            last_idx = original_index[-1]

            # Check if index is numeric
            if pd.api.types.is_numeric_dtype(original_index):
                # For numeric index, increment by the average step size
                if len(original_index) >= 2:
                    step_size = (original_index[-1] - original_index[0]) / (len(original_index) - 1)
                else:
                    step_size = 1

                forecast_index = pd.Index([
                    last_idx + (i + 1) * step_size for i in range(steps)
                ])
            else:
                # For non-numeric index, just use integers
                forecast_index = pd.Index([
                    f"Forecast {i+1}" for i in range(steps)
                ])

            return forecast_index

    except Exception as e:
        raise ValueError(f"Error extending index: {e}")


def combine_history_and_forecast(
    historical_data: Union[np.ndarray, pd.Series, pd.DataFrame],
    forecast_result: ForecastResult
) -> Union[pd.Series, pd.DataFrame]:
    """Combine historical data and forecasts into a single series or DataFrame.

    This function combines historical data with forecast results, creating
    a single time series that includes both historical values and forecasts
    with prediction intervals.

    Args:
        historical_data: Historical time series data
        forecast_result: Forecast results

    Returns:
        Union[pd.Series, pd.DataFrame]: Combined historical and forecast data

    Raises:
        ValueError: If the inputs are incompatible
    """
    # Convert historical data to pandas if it's a NumPy array
    if isinstance(historical_data, np.ndarray):
        if historical_data.ndim == 1:
            historical_data = pd.Series(historical_data)
        else:
            historical_data = pd.DataFrame(historical_data)

    # Create forecast DataFrame
    forecast_df = forecast_result.to_dataframe()

    try:
        if isinstance(historical_data, pd.Series):
            # For Series, create a DataFrame with historical values and forecasts
            result = pd.DataFrame({
                'historical': historical_data,
                'forecast': pd.Series(
                    forecast_result.point_forecast,
                    index=forecast_result.forecast_index
                ),
                f'lower_{int(forecast_result.confidence_level * 100)}': pd.Series(
                    forecast_result.lower_bound,
                    index=forecast_result.forecast_index
                ),
                f'upper_{int(forecast_result.confidence_level * 100)}': pd.Series(
                    forecast_result.upper_bound,
                    index=forecast_result.forecast_index
                )
            })

        elif isinstance(historical_data, pd.DataFrame):
            # For DataFrame, add forecast columns
            if historical_data.shape[1] == 1:
                # Single column DataFrame
                col_name = historical_data.columns[0]
                result = pd.DataFrame({
                    col_name: historical_data[col_name],
                    f'{col_name}_forecast': pd.Series(
                        forecast_result.point_forecast,
                        index=forecast_result.forecast_index
                    ),
                    f'{col_name}_lower_{int(forecast_result.confidence_level * 100)}': pd.Series(
                        forecast_result.lower_bound,
                        index=forecast_result.forecast_index
                    ),
                    f'{col_name}_upper_{int(forecast_result.confidence_level * 100)}': pd.Series(
                        forecast_result.upper_bound,
                        index=forecast_result.forecast_index
                    )
                })
            else:
                # Multi-column DataFrame - assume forecast is for the first column
                col_name = historical_data.columns[0]
                result = historical_data.copy()
                result[f'{col_name}_forecast'] = pd.Series(
                    forecast_result.point_forecast,
                    index=forecast_result.forecast_index
                )
                result[f'{col_name}_lower_{int(forecast_result.confidence_level * 100)}'] = pd.Series(
                    forecast_result.lower_bound,
                    index=forecast_result.forecast_index
                )
                result[f'{col_name}_upper_{int(forecast_result.confidence_level * 100)}'] = pd.Series(
                    forecast_result.upper_bound,
                    index=forecast_result.forecast_index
                )
        else:
            raise TypeError(
                f"Unsupported historical data type: {type(historical_data)}"
            )

        # Add metadata
        result.attrs["forecast_method"] = forecast_result.method.value
        result.attrs["confidence_level"] = forecast_result.confidence_level
        if forecast_result.model_name:
            result.attrs["model_name"] = forecast_result.model_name

        return result

    except Exception as e:
        raise ValueError(f"Error combining historical data and forecast: {e}")


def plot_forecast(
    historical_data: Union[np.ndarray, pd.Series, pd.DataFrame],
    forecast_result: ForecastResult,
    ax=None,
    figsize=(12, 6),
    title=None,
    include_history=True,
    **kwargs
):
    """Plot historical data and forecasts with prediction intervals.

    This function creates a visualization of historical data and forecasts,
    including prediction intervals, which is useful for presenting forecast
    results in a clear and informative way.

    Args:
        historical_data: Historical time series data
        forecast_result: Forecast results
        ax: Matplotlib axis to plot on (optional)
        figsize: Figure size if creating a new figure
        title: Plot title (optional)
        include_history: Whether to include historical data in the plot
        **kwargs: Additional keyword arguments for plotting

    Returns:
        matplotlib.axes.Axes: The plot axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Convert historical data to pandas if it's a NumPy array
    if isinstance(historical_data, np.ndarray):
        if historical_data.ndim == 1:
            historical_data = pd.Series(historical_data)
        else:
            historical_data = pd.DataFrame(historical_data)

    # Plot historical data if requested
    if include_history:
        if isinstance(historical_data, pd.Series):
            historical_data.plot(ax=ax, label='Historical Data')
        elif isinstance(historical_data, pd.DataFrame):
            if historical_data.shape[1] == 1:
                historical_data.iloc[:, 0].plot(ax=ax, label='Historical Data')
            else:
                # Plot first column if multiple columns
                historical_data.iloc[:, 0].plot(ax=ax, label=f'Historical {historical_data.columns[0]}')

    # Plot forecast
    forecast_result.plot(ax=ax)

    # Add title
    if title:
        ax.set_title(title)
    else:
        model_name = forecast_result.model_name or "Time Series"
        ax.set_title(f"{model_name} Forecast")

    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)

    return ax
