# tests/test_time_series.py

"""
Tests for time series analysis functionality in the MFE Toolbox.

This module contains comprehensive tests for the time series analysis components,
including ARMA/ARMAX model estimation, forecasting, diagnostics, and unit root testing.
Tests verify correct parameter estimation, model specification, transformation between
different representations, and proper handling of various time series data types.

The test suite includes:
- Tests for basic ARMA model estimation and forecasting
- Tests for ARMAX models with exogenous variables
- Tests for seasonal ARIMA (SARIMA) models
- Tests for diagnostic tests (Ljung-Box, LM tests, etc.)
- Tests for asynchronous implementations of time-consuming operations
- Tests for proper error handling and parameter validation
- Tests for integration with Pandas Series/DataFrames
- Tests for Statsmodels integration and extension points
"""

import asyncio
import numpy as np
import pandas as pd
import pytest
from scipy import stats
import statsmodels.api as sm
from typing import List, Tuple, Dict, Optional, Union, Any

from mfe.models.time_series.arma import ARMAModel, ARMAXModel
from mfe.models.time_series.base import TimeSeriesModel, TimeSeriesConfig, TimeSeriesResult
from mfe.models.time_series.diagnostics import (
    ljung_box, jarque_bera, durbin_watson, breusch_godfrey, arch_test,
    information_criteria, LjungBoxResult, JarqueBeraResult
)
from mfe.core.parameters import ARMAParameters
from mfe.core.exceptions import (
    ParameterError, DimensionError, EstimationError, ForecastError,
    SimulationError, NotFittedError
)


class TestARMAModel:
    """Tests for ARMA model estimation, forecasting, and diagnostics."""

    @pytest.fixture
    def ar1_data(self) -> np.ndarray:
        """Generate AR(1) data for testing."""
        np.random.seed(42)
        n = 200
        phi = 0.7
        sigma = 1.0

        # Generate AR(1) process: y_t = phi * y_{t-1} + e_t
        y = np.zeros(n)
        e = np.random.normal(0, sigma, n)

        for t in range(1, n):
            y[t] = phi * y[t-1] + e[t]

        return y

    @pytest.fixture
    def ar1_pandas_data(self, ar1_data) -> pd.Series:
        """Generate AR(1) data as Pandas Series with DatetimeIndex."""
        dates = pd.date_range(start='2020-01-01', periods=len(ar1_data), freq='D')
        return pd.Series(ar1_data, index=dates)

    @pytest.fixture
    def ma1_data(self) -> np.ndarray:
        """Generate MA(1) data for testing."""
        np.random.seed(43)
        n = 200
        theta = -0.5
        sigma = 1.0

        # Generate MA(1) process: y_t = e_t + theta * e_{t-1}
        e = np.random.normal(0, sigma, n+1)
        y = np.zeros(n)

        for t in range(n):
            y[t] = e[t+1] + theta * e[t]

        return y

    @pytest.fixture
    def arma11_data(self) -> np.ndarray:
        """Generate ARMA(1,1) data for testing."""
        np.random.seed(44)
        n = 200
        phi = 0.7
        theta = -0.3
        sigma = 1.0

        # Generate ARMA(1,1) process: y_t = phi * y_{t-1} + e_t + theta * e_{t-1}
        y = np.zeros(n)
        e = np.random.normal(0, sigma, n+1)

        for t in range(1, n):
            y[t] = phi * y[t-1] + e[t] + theta * e[t-1]

        return y

    @pytest.fixture
    def seasonal_data(self) -> np.ndarray:
        """Generate seasonal ARIMA data for testing."""
        np.random.seed(45)
        n = 200
        phi = 0.7
        Phi = 0.5  # Seasonal AR parameter
        s = 4      # Seasonal period
        sigma = 1.0

        # Generate seasonal AR process: y_t = phi * y_{t-1} + Phi * y_{t-s} + e_t
        y = np.zeros(n)
        e = np.random.normal(0, sigma, n)

        for t in range(s+1, n):
            y[t] = phi * y[t-1] + Phi * y[t-s] + e[t]

        return y

    def test_arma_model_initialization(self):
        """Test ARMA model initialization with various parameters."""
        # Test valid initialization
        model = ARMAModel(ar_order=1, ma_order=1, include_constant=True)
        assert model.ar_order == 1
        assert model.ma_order == 1
        assert model.include_constant is True

        # Test initialization with only AR component
        model = ARMAModel(ar_order=2, ma_order=0)
        assert model.ar_order == 2
        assert model.ma_order == 0

        # Test initialization with only MA component
        model = ARMAModel(ar_order=0, ma_order=2)
        assert model.ar_order == 0
        assert model.ma_order == 2

        # Test initialization without constant
        model = ARMAModel(ar_order=1, ma_order=1, include_constant=False)
        assert model.include_constant is False

        # Test initialization with custom name
        model = ARMAModel(ar_order=1, ma_order=1, name="Custom ARMA")
        assert model._name == "Custom ARMA"

    def test_arma_model_invalid_initialization(self):
        """Test ARMA model initialization with invalid parameters."""
        # Test negative AR order
        with pytest.raises(ParameterError):
            ARMAModel(ar_order=-1, ma_order=1)

        # Test negative MA order
        with pytest.raises(ParameterError):
            ARMAModel(ar_order=1, ma_order=-1)

        # Test both orders zero
        with pytest.raises(ParameterError):
            ARMAModel(ar_order=0, ma_order=0)

        # Test non-integer orders
        with pytest.raises(ParameterError):
            ARMAModel(ar_order=1.5, ma_order=1)

        with pytest.raises(ParameterError):
            ARMAModel(ar_order=1, ma_order=1.5)

    def test_ar1_model_estimation(self, ar1_data):
        """Test AR(1) model estimation."""
        model = ARMAModel(ar_order=1, ma_order=0)
        result = model.fit(ar1_data)

        # Check that the model was fitted
        assert model._fitted is True

        # Check that parameters were estimated
        assert model._params is not None
        assert len(model._params.ar_params) == 1

        # Check that the AR parameter is close to the true value (0.7)
        assert 0.6 < model._params.ar_params[0] < 0.8

        # Check that residuals and fitted values were computed
        assert model._residuals is not None
        assert model._fitted_values is not None
        assert len(model._residuals) == len(ar1_data)
        assert len(model._fitted_values) == len(ar1_data)

        # Check result object
        assert isinstance(result, TimeSeriesResult)
        assert 'ar1' in result.params
        assert 'ar1' in result.std_errors
        assert 'ar1' in result.t_stats
        assert 'ar1' in result.p_values

        # Check information criteria
        assert result.aic is not None
        assert result.bic is not None
        assert result.hqic is not None
        assert result.log_likelihood is not None

    def test_ma1_model_estimation(self, ma1_data):
        """Test MA(1) model estimation."""
        model = ARMAModel(ar_order=0, ma_order=1)
        result = model.fit(ma1_data)

        # Check that the model was fitted
        assert model._fitted is True

        # Check that parameters were estimated
        assert model._params is not None
        assert len(model._params.ma_params) == 1

        # Check that the MA parameter is close to the true value (-0.5)
        assert -0.6 < model._params.ma_params[0] < -0.4

        # Check result object
        assert isinstance(result, TimeSeriesResult)
        assert 'ma1' in result.params
        assert 'ma1' in result.std_errors
        assert 'ma1' in result.t_stats
        assert 'ma1' in result.p_values

    def test_arma11_model_estimation(self, arma11_data):
        """Test ARMA(1,1) model estimation."""
        model = ARMAModel(ar_order=1, ma_order=1)
        result = model.fit(arma11_data)

        # Check that the model was fitted
        assert model._fitted is True

        # Check that parameters were estimated
        assert model._params is not None
        assert len(model._params.ar_params) == 1
        assert len(model._params.ma_params) == 1

        # Check that parameters are close to true values (phi=0.7, theta=-0.3)
        assert 0.6 < model._params.ar_params[0] < 0.8
        assert -0.4 < model._params.ma_params[0] < -0.2

        # Check result object
        assert isinstance(result, TimeSeriesResult)
        assert 'ar1' in result.params
        assert 'ma1' in result.params

    def test_arma_model_with_pandas_series(self, ar1_pandas_data):
        """Test ARMA model estimation with Pandas Series input."""
        model = ARMAModel(ar_order=1, ma_order=0)
        result = model.fit(ar1_pandas_data)

        # Check that the model was fitted
        assert model._fitted is True

        # Check that parameters were estimated
        assert model._params is not None
        assert len(model._params.ar_params) == 1

        # Check that the AR parameter is close to the true value (0.7)
        assert 0.6 < model._params.ar_params[0] < 0.8

        # Check that the index was preserved
        assert model._index is not None
        assert isinstance(model._index, pd.DatetimeIndex)
        assert len(model._index) == len(ar1_pandas_data)

    def test_arma_model_forecasting(self, ar1_data):
        """Test ARMA model forecasting."""
        model = ARMAModel(ar_order=1, ma_order=0)
        model.fit(ar1_data)

        # Forecast 10 steps ahead
        steps = 10
        forecasts, lower_bounds, upper_bounds = model.forecast(steps)

        # Check forecast dimensions
        assert len(forecasts) == steps
        assert len(lower_bounds) == steps
        assert len(upper_bounds) == steps

        # Check that lower bounds are less than forecasts
        assert np.all(lower_bounds < forecasts)

        # Check that upper bounds are greater than forecasts
        assert np.all(upper_bounds > forecasts)

        # Check that forecasts are finite
        assert np.all(np.isfinite(forecasts))
        assert np.all(np.isfinite(lower_bounds))
        assert np.all(np.isfinite(upper_bounds))

    def test_arma_model_forecasting_errors(self, ar1_data):
        """Test ARMA model forecasting error handling."""
        model = ARMAModel(ar_order=1, ma_order=0)

        # Test forecasting before fitting
        with pytest.raises(NotFittedError):
            model.forecast(10)

        # Fit the model
        model.fit(ar1_data)

        # Test invalid steps
        with pytest.raises(ValueError):
            model.forecast(0)

        with pytest.raises(ValueError):
            model.forecast(-1)

        # Test invalid confidence level
        with pytest.raises(ValueError):
            model.forecast(10, confidence_level=0)

        with pytest.raises(ValueError):
            model.forecast(10, confidence_level=1)

    def test_arma_model_simulation(self, ar1_data):
        """Test ARMA model simulation."""
        model = ARMAModel(ar_order=1, ma_order=0)
        model.fit(ar1_data)

        # Simulate 100 observations
        n_periods = 100
        simulated = model.simulate(n_periods)

        # Check simulation dimensions
        assert len(simulated) == n_periods

        # Check that simulated data is finite
        assert np.all(np.isfinite(simulated))

        # Check that simulated data has similar properties to original data
        assert abs(np.mean(simulated)) < 0.5  # Should be close to zero
        assert 0.8 < np.std(simulated) < 1.2  # Should be close to 1.0

    def test_arma_model_simulation_errors(self, ar1_data):
        """Test ARMA model simulation error handling."""
        model = ARMAModel(ar_order=1, ma_order=0)

        # Test simulation before fitting
        with pytest.raises(NotFittedError):
            model.simulate(100)

        # Fit the model
        model.fit(ar1_data)

        # Test invalid n_periods
        with pytest.raises(ValueError):
            model.simulate(0)

        with pytest.raises(ValueError):
            model.simulate(-1)

        # Test invalid burn
        with pytest.raises(ValueError):
            model.simulate(100, burn=-1)

    def test_arma_model_loglikelihood(self, ar1_data):
        """Test ARMA model log-likelihood computation."""
        model = ARMAModel(ar_order=1, ma_order=0)
        model.fit(ar1_data)

        # Create parameters similar to estimated ones
        params = ARMAParameters(
            ar_params=np.array([0.7]),
            ma_params=np.array([]),
            constant=0.0,
            sigma2=1.0
        )

        # Compute log-likelihood
        llf = model.loglikelihood(params, ar1_data)

        # Check that log-likelihood is finite
        assert np.isfinite(llf)

        # Check that log-likelihood is negative (for this type of model)
        assert llf < 0

    def test_arma_model_summary(self, ar1_data):
        """Test ARMA model summary generation."""
        model = ARMAModel(ar_order=1, ma_order=0)

        # Test summary before fitting
        summary = model.summary()
        assert "not fitted" in summary

        # Fit the model
        model.fit(ar1_data)

        # Test summary after fitting
        summary = model.summary()
        assert "not fitted" not in summary
        assert "AR" in summary
        assert "Log-Likelihood" in summary
        assert "AIC" in summary
        assert "BIC" in summary

    @pytest.mark.asyncio
    async def test_arma_model_async_fit(self, ar1_data):
        """Test asynchronous ARMA model fitting."""
        model = ARMAModel(ar_order=1, ma_order=0)

        # Define a progress callback
        progress_updates = []

        async def progress_callback(percent, message):
            progress_updates.append((percent, message))

        # Fit the model asynchronously
        result = await model.fit_async(ar1_data, progress_callback=progress_callback)

        # Check that the model was fitted
        assert model._fitted is True

        # Check that parameters were estimated
        assert model._params is not None
        assert len(model._params.ar_params) == 1

        # Check that the AR parameter is close to the true value (0.7)
        assert 0.6 < model._params.ar_params[0] < 0.8

        # Check that progress callback was called
        assert len(progress_updates) > 0
        assert progress_updates[-1][0] == 1.0  # Final progress should be 100%

    @pytest.mark.asyncio
    async def test_arma_model_async_forecast(self, ar1_data):
        """Test asynchronous ARMA model forecasting."""
        model = ARMAModel(ar_order=1, ma_order=0)
        model.fit(ar1_data)

        # Define a progress callback
        progress_updates = []

        async def progress_callback(percent, message):
            progress_updates.append((percent, message))

        # Forecast asynchronously
        steps = 10
        forecasts, lower_bounds, upper_bounds = await model.forecast_async(
            steps, progress_callback=progress_callback
        )

        # Check forecast dimensions
        assert len(forecasts) == steps
        assert len(lower_bounds) == steps
        assert len(upper_bounds) == steps

        # Check that progress callback was called
        assert len(progress_updates) > 0
        assert progress_updates[-1][0] == 1.0  # Final progress should be 100%

    @pytest.mark.asyncio
    async def test_arma_model_async_simulate(self, ar1_data):
        """Test asynchronous ARMA model simulation."""
        model = ARMAModel(ar_order=1, ma_order=0)
        model.fit(ar1_data)

        # Define a progress callback
        progress_updates = []

        async def progress_callback(percent, message):
            progress_updates.append((percent, message))

        # Simulate asynchronously
        n_periods = 100
        simulated = await model.simulate_async(
            n_periods, progress_callback=progress_callback
        )

        # Check simulation dimensions
        assert len(simulated) == n_periods

        # Check that progress callback was called
        assert len(progress_updates) > 0
        assert progress_updates[-1][0] == 1.0  # Final progress should be 100%

    def test_arma_model_with_config(self, ar1_data):
        """Test ARMA model with custom configuration."""
        model = ARMAModel(ar_order=1, ma_order=0)

        # Create custom configuration
        config = TimeSeriesConfig(
            method="css",
            solver="BFGS",
            max_iter=500,
            tol=1e-6,
            cov_type="robust",
            use_numba=True,
            display_progress=False
        )

        # Set configuration
        model.config = config

        # Fit the model
        result = model.fit(ar1_data)

        # Check that the model was fitted
        assert model._fitted is True

        # Check that parameters were estimated
        assert model._params is not None
        assert len(model._params.ar_params) == 1

        # Check that the AR parameter is close to the true value (0.7)
        assert 0.6 < model._params.ar_params[0] < 0.8

    def test_arma_model_with_invalid_config(self):
        """Test ARMA model with invalid configuration."""
        model = ARMAModel(ar_order=1, ma_order=0)

        # Test invalid method
        config = TimeSeriesConfig()
        config.method = "invalid"
        with pytest.raises(ParameterError):
            model.config = config

        # Test invalid solver
        config = TimeSeriesConfig()
        config.solver = "invalid"
        with pytest.raises(ParameterError):
            model.config = config

        # Test invalid max_iter
        config = TimeSeriesConfig()
        config.max_iter = 0
        with pytest.raises(ParameterError):
            model.config = config

        # Test invalid tol
        config = TimeSeriesConfig()
        config.tol = 0
        with pytest.raises(ParameterError):
            model.config = config

        # Test invalid cov_type
        config = TimeSeriesConfig()
        config.cov_type = "invalid"
        with pytest.raises(ParameterError):
            model.config = config

    def test_arma_model_with_invalid_data(self):
        """Test ARMA model with invalid data."""
        model = ARMAModel(ar_order=1, ma_order=0)

        # Test empty data
        with pytest.raises(ValueError):
            model.fit(np.array([]))

        # Test data with NaN
        with pytest.raises(ValueError):
            model.fit(np.array([1.0, 2.0, np.nan, 4.0]))

        # Test data with Inf
        with pytest.raises(ValueError):
            model.fit(np.array([1.0, 2.0, np.inf, 4.0]))

        # Test data with wrong dimensions
        with pytest.raises(DimensionError):
            model.fit(np.array([[1.0, 2.0], [3.0, 4.0]]))

        # Test data with wrong type
        with pytest.raises(TypeError):
            model.fit([1.0, 2.0, 3.0, 4.0])  # List instead of array or Series

    def test_seasonal_arima_model(self, seasonal_data):
        """Test seasonal ARIMA model estimation using statsmodels."""
        # Use statsmodels directly for seasonal ARIMA
        sm_model = sm.tsa.SARIMAX(
            seasonal_data,
            order=(1, 0, 0),
            seasonal_order=(1, 0, 0, 4)
        )
        sm_result = sm_model.fit()

        # Check that parameters were estimated
        assert sm_result.params is not None

        # Extract parameters
        ar_param = sm_result.params['ar.L1']
        seasonal_ar_param = sm_result.params['ar.S.L4']

        # Check that parameters are close to true values (phi=0.7, Phi=0.5)
        assert 0.6 < ar_param < 0.8
        assert 0.4 < seasonal_ar_param < 0.6

        # Check information criteria
        assert sm_result.aic is not None
        assert sm_result.bic is not None
        assert sm_result.hqic is not None
        assert sm_result.llf is not None

    def test_arma_to_sarima_transformation(self, seasonal_data):
        """Test transformation between ARMA and SARIMA representations."""
        # First, fit a seasonal ARIMA model using statsmodels
        sm_model = sm.tsa.SARIMAX(
            seasonal_data,
            order=(1, 0, 0),
            seasonal_order=(1, 0, 0, 4)
        )
        sm_result = sm_model.fit()

        # Extract parameters
        ar_param = sm_result.params['ar.L1']
        seasonal_ar_param = sm_result.params['ar.S.L4']

        # Now, fit an equivalent ARMA(5,0) model
        # The AR polynomial is (1 - phi*L)(1 - Phi*L^4) = 1 - phi*L - Phi*L^4 + phi*Phi*L^5
        # So we need AR parameters at lags 1, 4, and 5
        arma_model = ARMAModel(ar_order=5, ma_order=0)
        arma_result = arma_model.fit(seasonal_data)

        # Check that the ARMA model was fitted
        assert arma_model._fitted is True

        # Check that parameters were estimated
        assert arma_model._params is not None
        assert len(arma_model._params.ar_params) == 5

        # Check that the AR parameters match the expected values
        # AR(1) should be close to phi
        assert abs(arma_model._params.ar_params[0] - ar_param) < 0.1

        # AR(4) should be close to Phi
        assert abs(arma_model._params.ar_params[3] - seasonal_ar_param) < 0.1

        # AR(5) should be close to -phi*Phi
        expected_ar5 = -ar_param * seasonal_ar_param
        assert abs(arma_model._params.ar_params[4] - expected_ar5) < 0.1

    def test_differencing_for_nonstationary_data(self):
        """Test differencing for nonstationary data."""
        # Generate nonstationary data (random walk)
        np.random.seed(46)
        n = 200
        e = np.random.normal(0, 1, n)
        y = np.cumsum(e)  # Random walk: y_t = y_{t-1} + e_t

        # Difference the data
        dy = np.diff(y)

        # Fit ARMA model to differenced data
        model = ARMAModel(ar_order=1, ma_order=0)
        result = model.fit(dy)

        # Check that the model was fitted
        assert model._fitted is True

        # Check that parameters were estimated
        assert model._params is not None
        assert len(model._params.ar_params) == 1

        # Check that the AR parameter is close to 0 (differenced random walk is white noise)
        assert abs(model._params.ar_params[0]) < 0.2

    def test_arma_model_with_numba_acceleration(self, ar1_data):
        """Test ARMA model with Numba acceleration."""
        # Create model with Numba acceleration enabled
        model = ARMAModel(ar_order=1, ma_order=0)
        model.config.use_numba = True

        # Fit the model
        result = model.fit(ar1_data)

        # Check that the model was fitted
        assert model._fitted is True

        # Check that parameters were estimated
        assert model._params is not None
        assert len(model._params.ar_params) == 1

        # Check that the AR parameter is close to the true value (0.7)
        assert 0.6 < model._params.ar_params[0] < 0.8

        # Create model with Numba acceleration disabled
        model_no_numba = ARMAModel(ar_order=1, ma_order=0)
        model_no_numba.config.use_numba = False

        # Fit the model
        result_no_numba = model_no_numba.fit(ar1_data)

        # Check that the model was fitted
        assert model_no_numba._fitted is True

        # Check that parameters were estimated
        assert model_no_numba._params is not None
        assert len(model_no_numba._params.ar_params) == 1

        # Check that the AR parameter is close to the true value (0.7)
        assert 0.6 < model_no_numba._params.ar_params[0] < 0.8

        # Check that both models give similar results
        assert abs(model._params.ar_params[0] - model_no_numba._params.ar_params[0]) < 0.1


class TestARMAXModel:
    """Tests for ARMAX model estimation, forecasting, and diagnostics."""

    @pytest.fixture
    def armax_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate ARMAX data for testing."""
        np.random.seed(47)
        n = 200
        phi = 0.7
        beta = 1.5
        sigma = 1.0

        # Generate exogenous variable
        x = np.random.normal(0, 1, n)

        # Generate ARMAX process: y_t = phi * y_{t-1} + beta * x_t + e_t
        y = np.zeros(n)
        e = np.random.normal(0, sigma, n)

        for t in range(1, n):
            y[t] = phi * y[t-1] + beta * x[t] + e[t]

        return y, x

    @pytest.fixture
    def armax_pandas_data(self, armax_data) -> Tuple[pd.Series, pd.DataFrame]:
        """Generate ARMAX data as Pandas Series/DataFrame with DatetimeIndex."""
        y, x = armax_data
        dates = pd.date_range(start='2020-01-01', periods=len(y), freq='D')
        y_series = pd.Series(y, index=dates)
        x_df = pd.DataFrame({'x': x}, index=dates)
        return y_series, x_df

    def test_armax_model_initialization(self):
        """Test ARMAX model initialization with various parameters."""
        # Test valid initialization
        model = ARMAXModel(ar_order=1, ma_order=1, include_constant=True)
        assert model.ar_order == 1
        assert model.ma_order == 1
        assert model.include_constant is True

        # Test initialization with only AR component
        model = ARMAXModel(ar_order=2, ma_order=0)
        assert model.ar_order == 2
        assert model.ma_order == 0

        # Test initialization with only MA component
        model = ARMAXModel(ar_order=0, ma_order=2)
        assert model.ar_order == 0
        assert model.ma_order == 2

        # Test initialization without constant
        model = ARMAXModel(ar_order=1, ma_order=1, include_constant=False)
        assert model.include_constant is False

        # Test initialization with custom name
        model = ARMAXModel(ar_order=1, ma_order=1, name="Custom ARMAX")
        assert model._name == "Custom ARMAX"

    def test_armax_model_estimation(self, armax_data):
        """Test ARMAX model estimation."""
        y, x = armax_data

        model = ARMAXModel(ar_order=1, ma_order=0)
        result = model.fit(y, exog=x.reshape(-1, 1), exog_names=['x1'])

        # Check that the model was fitted
        assert model._fitted is True

        # Check that parameters were estimated
        assert model._params is not None
        assert len(model._params.ar_params) == 1

        # Check that the AR parameter is close to the true value (0.7)
        assert 0.6 < model._params.ar_params[0] < 0.8

        # Check that exogenous variables were stored
        assert model._exog is not None
        assert model._exog_names is not None
        assert model._exog_names == ['x1']

        # Check result object
        assert isinstance(result, TimeSeriesResult)
        assert 'ar1' in result.params
        assert 'x1' in result.params

        # Check that the exogenous parameter is close to the true value (1.5)
        assert 1.3 < result.params['x1'] < 1.7

    def test_armax_model_with_pandas_data(self, armax_pandas_data):
        """Test ARMAX model estimation with Pandas Series/DataFrame input."""
        y_series, x_df = armax_pandas_data

        model = ARMAXModel(ar_order=1, ma_order=0)
        result = model.fit(y_series, exog=x_df)

        # Check that the model was fitted
        assert model._fitted is True

        # Check that parameters were estimated
        assert model._params is not None
        assert len(model._params.ar_params) == 1

        # Check that the AR parameter is close to the true value (0.7)
        assert 0.6 < model._params.ar_params[0] < 0.8

        # Check that exogenous variables were stored
        assert model._exog is not None
        assert model._exog_names is not None
        assert model._exog_names == ['x']

        # Check that the index was preserved
        assert model._index is not None
        assert isinstance(model._index, pd.DatetimeIndex)
        assert len(model._index) == len(y_series)

        # Check result object
        assert isinstance(result, TimeSeriesResult)
        assert 'ar1' in result.params
        assert 'x' in result.params

        # Check that the exogenous parameter is close to the true value (1.5)
        assert 1.3 < result.params['x'] < 1.7

    def test_armax_model_forecasting(self, armax_data):
        """Test ARMAX model forecasting."""
        y, x = armax_data

        model = ARMAXModel(ar_order=1, ma_order=0)
        model.fit(y, exog=x.reshape(-1, 1), exog_names=['x1'])

        # Create future exogenous data
        steps = 10
        future_x = np.random.normal(0, 1, steps).reshape(-1, 1)

        # Forecast with exogenous data
        forecasts, lower_bounds, upper_bounds = model.forecast(steps, exog=future_x)

        # Check forecast dimensions
        assert len(forecasts) == steps
        assert len(lower_bounds) == steps
        assert len(upper_bounds) == steps

        # Check that lower bounds are less than forecasts
        assert np.all(lower_bounds < forecasts)

        # Check that upper bounds are greater than forecasts
        assert np.all(upper_bounds > forecasts)

        # Check that forecasts are finite
        assert np.all(np.isfinite(forecasts))
        assert np.all(np.isfinite(lower_bounds))
        assert np.all(np.isfinite(upper_bounds))

    def test_armax_model_forecasting_errors(self, armax_data):
        """Test ARMAX model forecasting error handling."""
        y, x = armax_data

        model = ARMAXModel(ar_order=1, ma_order=0)

        # Test forecasting before fitting
        with pytest.raises(NotFittedError):
            model.forecast(10)

        # Fit the model with exogenous variables
        model.fit(y, exog=x.reshape(-1, 1), exog_names=['x1'])

        # Test forecasting without exogenous data
        with pytest.raises(ValueError):
            model.forecast(10)

        # Test forecasting with wrong exogenous dimensions
        with pytest.raises(DimensionError):
            model.forecast(10, exog=np.random.normal(0, 1, (10, 2)))

        # Test forecasting with insufficient exogenous data
        with pytest.raises(ValueError):
            model.forecast(10, exog=np.random.normal(0, 1, 5).reshape(-1, 1))

    def test_armax_model_simulation(self, armax_data):
        """Test ARMAX model simulation."""
        y, x = armax_data

        model = ARMAXModel(ar_order=1, ma_order=0)
        model.fit(y, exog=x.reshape(-1, 1), exog_names=['x1'])

        # Create simulation exogenous data
        n_periods = 100
        sim_x = np.random.normal(0, 1, n_periods).reshape(-1, 1)

        # Simulate with exogenous data
        simulated = model.simulate(n_periods, exog=sim_x)

        # Check simulation dimensions
        assert len(simulated) == n_periods

        # Check that simulated data is finite
        assert np.all(np.isfinite(simulated))

    def test_armax_model_simulation_errors(self, armax_data):
        """Test ARMAX model simulation error handling."""
        y, x = armax_data

        model = ARMAXModel(ar_order=1, ma_order=0)

        # Test simulation before fitting
        with pytest.raises(NotFittedError):
            model.simulate(100)

        # Fit the model with exogenous variables
        model.fit(y, exog=x.reshape(-1, 1), exog_names=['x1'])

        # Test simulation without exogenous data
        with pytest.raises(ValueError):
            model.simulate(100)

        # Test simulation with wrong exogenous dimensions
        with pytest.raises(DimensionError):
            model.simulate(100, exog=np.random.normal(0, 1, (100, 2)))

        # Test simulation with insufficient exogenous data
        with pytest.raises(ValueError):
            model.simulate(100, exog=np.random.normal(0, 1, 50).reshape(-1, 1))

    @pytest.mark.asyncio
    async def test_armax_model_async_fit(self, armax_data):
        """Test asynchronous ARMAX model fitting."""
        y, x = armax_data

        model = ARMAXModel(ar_order=1, ma_order=0)

        # Define a progress callback
        progress_updates = []

        async def progress_callback(percent, message):
            progress_updates.append((percent, message))

        # Fit the model asynchronously
        result = await model.fit_async(
            y, exog=x.reshape(-1, 1), exog_names=['x1'],
            progress_callback=progress_callback
        )

        # Check that the model was fitted
        assert model._fitted is True

        # Check that parameters were estimated
        assert model._params is not None
        assert len(model._params.ar_params) == 1

        # Check that the AR parameter is close to the true value (0.7)
        assert 0.6 < model._params.ar_params[0] < 0.8

        # Check that progress callback was called
        assert len(progress_updates) > 0
        assert progress_updates[-1][0] == 1.0  # Final progress should be 100%

    @pytest.mark.asyncio
    async def test_armax_model_async_forecast(self, armax_data):
        """Test asynchronous ARMAX model forecasting."""
        y, x = armax_data

        model = ARMAXModel(ar_order=1, ma_order=0)
        model.fit(y, exog=x.reshape(-1, 1), exog_names=['x1'])

        # Create future exogenous data
        steps = 10
        future_x = np.random.normal(0, 1, steps).reshape(-1, 1)

        # Define a progress callback
        progress_updates = []

        async def progress_callback(percent, message):
            progress_updates.append((percent, message))

        # Forecast asynchronously
        forecasts, lower_bounds, upper_bounds = await model.forecast_async(
            steps, exog=future_x, progress_callback=progress_callback
        )

        # Check forecast dimensions
        assert len(forecasts) == steps
        assert len(lower_bounds) == steps
        assert len(upper_bounds) == steps

        # Check that progress callback was called
        assert len(progress_updates) > 0
        assert progress_updates[-1][0] == 1.0  # Final progress should be 100%

    def test_armax_model_with_multiple_exog(self, armax_data):
        """Test ARMAX model with multiple exogenous variables."""
        y, x1 = armax_data

        # Create a second exogenous variable
        np.random.seed(48)
        x2 = np.random.normal(0, 1, len(y))

        # Combine exogenous variables
        X = np.column_stack((x1, x2))

        model = ARMAXModel(ar_order=1, ma_order=0)
        result = model.fit(y, exog=X, exog_names=['x1', 'x2'])

        # Check that the model was fitted
        assert model._fitted is True

        # Check that parameters were estimated
        assert model._params is not None
        assert len(model._params.ar_params) == 1

        # Check that the AR parameter is close to the true value (0.7)
        assert 0.6 < model._params.ar_params[0] < 0.8

        # Check that exogenous variables were stored
        assert model._exog is not None
        assert model._exog_names is not None
        assert model._exog_names == ['x1', 'x2']

        # Check result object
        assert isinstance(result, TimeSeriesResult)
        assert 'ar1' in result.params
        assert 'x1' in result.params
        assert 'x2' in result.params

        # Check that the first exogenous parameter is close to the true value (1.5)
        assert 1.3 < result.params['x1'] < 1.7

    def test_armax_model_with_invalid_exog(self, armax_data):
        """Test ARMAX model with invalid exogenous variables."""
        y, x = armax_data

        model = ARMAXModel(ar_order=1, ma_order=0)

        # Test exog with wrong length
        with pytest.raises(DimensionError):
            model.fit(y, exog=x[:-10].reshape(-1, 1))

        # Test exog with NaN
        x_nan = x.copy()
        x_nan[10] = np.nan
        with pytest.raises(ValueError):
            model.fit(y, exog=x_nan.reshape(-1, 1))

        # Test exog with Inf
        x_inf = x.copy()
        x_inf[10] = np.inf
        with pytest.raises(ValueError):
            model.fit(y, exog=x_inf.reshape(-1, 1))

        # Test exog_names with wrong length
        with pytest.raises(ValueError):
            model.fit(y, exog=np.column_stack((x, x)), exog_names=['x1'])

        # Test exog with wrong type
        with pytest.raises(TypeError):
            model.fit(y, exog=[1.0, 2.0, 3.0])  # List instead of array or DataFrame


class TestDiagnosticTests:
    """Tests for time series diagnostic tests."""

    @pytest.fixture
    def white_noise(self) -> np.ndarray:
        """Generate white noise data for testing."""
        np.random.seed(49)
        return np.random.normal(0, 1, 100)

    @pytest.fixture
    def autocorrelated_data(self) -> np.ndarray:
        """Generate autocorrelated data for testing."""
        np.random.seed(50)
        n = 100
        phi = 0.7

        # Generate AR(1) process: y_t = phi * y_{t-1} + e_t
        y = np.zeros(n)
        e = np.random.normal(0, 1, n)

        for t in range(1, n):
            y[t] = phi * y[t-1] + e[t]

        return y

    @pytest.fixture
    def non_normal_data(self) -> np.ndarray:
        """Generate non-normal data for testing."""
        np.random.seed(51)
        # Generate chi-squared data with 3 degrees of freedom (skewed)
        return np.random.chisquare(3, 100)

    def test_ljung_box_test_white_noise(self, white_noise):
        """Test Ljung-Box test with white noise data."""
        result = ljung_box(white_noise, lags=10)

        # Check result type
        assert isinstance(result, LjungBoxResult)

        # Check test attributes
        assert result.test_name == "Ljung-Box"
        assert result.lags == 10
        assert result.df == 0

        # Check test statistics
        assert np.isfinite(result.test_statistic)
        assert np.isfinite(result.p_value)

        # White noise should not show autocorrelation
        assert result.p_value > 0.05
        assert "fail to reject" in result.conclusion

    def test_ljung_box_test_autocorrelated(self, autocorrelated_data):
        """Test Ljung-Box test with autocorrelated data."""
        result = ljung_box(autocorrelated_data, lags=10)

        # Check result type
        assert isinstance(result, LjungBoxResult)

        # Check test attributes
        assert result.test_name == "Ljung-Box"
        assert result.lags == 10
        assert result.df == 0

        # Check test statistics
        assert np.isfinite(result.test_statistic)
        assert np.isfinite(result.p_value)

        # Autocorrelated data should show autocorrelation
        assert result.p_value < 0.05
        assert "reject" in result.conclusion

    def test_ljung_box_test_with_df(self, autocorrelated_data):
        """Test Ljung-Box test with degrees of freedom adjustment."""
        # Fit an AR(1) model
        model = ARMAModel(ar_order=1, ma_order=0)
        model.fit(autocorrelated_data)

        # Get residuals
        residuals = model.residuals

        # Test residuals with df=1 (for AR(1) model)
        result = ljung_box(residuals, lags=10, df=1)

        # Check result type
        assert isinstance(result, LjungBoxResult)

        # Check test attributes
        assert result.test_name == "Ljung-Box"
        assert result.lags == 10
        assert result.df == 1

        # Check test statistics
        assert np.isfinite(result.test_statistic)
        assert np.isfinite(result.p_value)

        # Residuals from a well-specified model should not show autocorrelation
        assert result.p_value > 0.05
        assert "fail to reject" in result.conclusion

    def test_ljung_box_test_invalid_inputs(self, white_noise):
        """Test Ljung-Box test with invalid inputs."""
        # Test with NaN
        data_nan = white_noise.copy()
        data_nan[10] = np.nan
        with pytest.raises(ValueError):
            ljung_box(data_nan)

        # Test with Inf
        data_inf = white_noise.copy()
        data_inf[10] = np.inf
        with pytest.raises(ValueError):
            ljung_box(data_inf)

        # Test with negative lags
        with pytest.raises(ValueError):
            ljung_box(white_noise, lags=-1)

        # Test with lags >= nobs
        with pytest.raises(ValueError):
            ljung_box(white_noise, lags=len(white_noise))

        # Test with negative df
        with pytest.raises(ValueError):
            ljung_box(white_noise, df=-1)

        # Test with df >= lags
        with pytest.raises(ValueError):
            ljung_box(white_noise, lags=10, df=10)

    def test_jarque_bera_test_normal(self, white_noise):
        """Test Jarque-Bera test with normal data."""
        result = jarque_bera(white_noise)

        # Check result type
        assert isinstance(result, JarqueBeraResult)

        # Check test attributes
        assert result.test_name == "Jarque-Bera"

        # Check test statistics
        assert np.isfinite(result.test_statistic)
        assert np.isfinite(result.p_value)
        assert np.isfinite(result.skewness)
        assert np.isfinite(result.kurtosis)

        # Normal data should not reject normality
        assert result.p_value > 0.05
        assert "fail to reject" in result.conclusion

    def test_jarque_bera_test_non_normal(self, non_normal_data):
        """Test Jarque-Bera test with non-normal data."""
        result = jarque_bera(non_normal_data)

        # Check result type
        assert isinstance(result, JarqueBeraResult)

        # Check test attributes
        assert result.test_name == "Jarque-Bera"

        # Check test statistics
        assert np.isfinite(result.test_statistic)
        assert np.isfinite(result.p_value)
        assert np.isfinite(result.skewness)
        assert np.isfinite(result.kurtosis)

        # Non-normal data should reject normality
        assert result.p_value < 0.05
        assert "reject" in result.conclusion

        # Chi-squared distribution is right-skewed
        assert result.skewness > 0

    def test_jarque_bera_test_invalid_inputs(self, white_noise):
        """Test Jarque-Bera test with invalid inputs."""
        # Test with NaN
        data_nan = white_noise.copy()
        data_nan[10] = np.nan
        with pytest.raises(ValueError):
            jarque_bera(data_nan)

        # Test with Inf
        data_inf = white_noise.copy()
        data_inf[10] = np.inf
        with pytest.raises(ValueError):
            jarque_bera(data_inf)

    def test_durbin_watson_test(self, white_noise, autocorrelated_data):
        """Test Durbin-Watson test."""
        # Test with white noise
        result_wn = durbin_watson(white_noise)

        # Check result attributes
        assert result_wn.test_name == "Durbin-Watson"
        assert np.isfinite(result_wn.test_statistic)

        # White noise should have DW statistic close to 2
        assert 1.5 < result_wn.test_statistic < 2.5

        # Test with autocorrelated data
        result_ac = durbin_watson(autocorrelated_data)

        # Check result attributes
        assert result_ac.test_name == "Durbin-Watson"
        assert np.isfinite(result_ac.test_statistic)

        # Positively autocorrelated data should have DW statistic < 2
        assert result_ac.test_statistic < 1.5

    def test_durbin_watson_test_invalid_inputs(self, white_noise):
        """Test Durbin-Watson test with invalid inputs."""
        # Test with NaN
        data_nan = white_noise.copy()
        data_nan[10] = np.nan
        with pytest.raises(ValueError):
            durbin_watson(data_nan)

        # Test with Inf
        data_inf = white_noise.copy()
        data_inf[10] = np.inf
        with pytest.raises(ValueError):
            durbin_watson(data_inf)

    def test_breusch_godfrey_test(self, white_noise, autocorrelated_data):
        """Test Breusch-Godfrey test."""
        # Test with white noise
        result_wn = breusch_godfrey(white_noise, lags=1)

        # Check result attributes
        assert result_wn.test_name == "Breusch-Godfrey"
        assert np.isfinite(result_wn.test_statistic)
        assert np.isfinite(result_wn.p_value)

        # White noise should not show serial correlation
        assert result_wn.p_value > 0.05
        assert "fail to reject" in result_wn.conclusion

        # Test with autocorrelated data
        result_ac = breusch_godfrey(autocorrelated_data, lags=1)

        # Check result attributes
        assert result_ac.test_name == "Breusch-Godfrey"
        assert np.isfinite(result_ac.test_statistic)
        assert np.isfinite(result_ac.p_value)

        # Autocorrelated data should show serial correlation
        assert result_ac.p_value < 0.05
        assert "reject" in result_ac.conclusion

    def test_breusch_godfrey_test_with_design_matrix(self, autocorrelated_data):
        """Test Breusch-Godfrey test with design matrix."""
        # Create a design matrix with a constant and a trend
        n = len(autocorrelated_data)
        X = np.column_stack((np.ones(n), np.arange(n)))

        # Test with design matrix
        result = breusch_godfrey(autocorrelated_data, X=X, lags=1)

        # Check result attributes
        assert result.test_name == "Breusch-Godfrey"
        assert np.isfinite(result.test_statistic)
        assert np.isfinite(result.p_value)

    def test_breusch_godfrey_test_invalid_inputs(self, white_noise):
        """Test Breusch-Godfrey test with invalid inputs."""
        # Test with NaN
        data_nan = white_noise.copy()
        data_nan[10] = np.nan
        with pytest.raises(ValueError):
            breusch_godfrey(data_nan)

        # Test with Inf
        data_inf = white_noise.copy()
        data_inf[10] = np.inf
        with pytest.raises(ValueError):
            breusch_godfrey(data_inf)

        # Test with negative lags
        with pytest.raises(ValueError):
            breusch_godfrey(white_noise, lags=-1)

        # Test with lags >= nobs
        with pytest.raises(ValueError):
            breusch_godfrey(white_noise, lags=len(white_noise))

        # Test with X having wrong number of rows
        with pytest.raises(ValueError):
            breusch_godfrey(white_noise, X=np.ones((len(white_noise) - 1, 1)))

    def test_arch_test(self, white_noise):
        """Test ARCH test."""
        # Generate ARCH data
        np.random.seed(52)
        n = 200
        alpha = 0.7

        # Generate ARCH(1) process: y_t = e_t * sqrt(omega + alpha * y_{t-1}^2)
        omega = 0.2
        y = np.zeros(n)
        e = np.random.normal(0, 1, n)

        for t in range(1, n):
            y[t] = e[t] * np.sqrt(omega + alpha * y[t-1]**2)

        # Test with white noise
        result_wn = arch_test(white_noise, lags=1)

        # Check result attributes
        assert result_wn.test_name == "ARCH"
        assert np.isfinite(result_wn.test_statistic)
        assert np.isfinite(result_wn.p_value)

        # White noise should not show ARCH effects
        assert result_wn.p_value > 0.05
        assert "fail to reject" in result_wn.conclusion

        # Test with ARCH data
        result_arch = arch_test(y, lags=1)

        # Check result attributes
        assert result_arch.test_name == "ARCH"
        assert np.isfinite(result_arch.test_statistic)
        assert np.isfinite(result_arch.p_value)

        # ARCH data should show ARCH effects
        assert result_arch.p_value < 0.05
        assert "reject" in result_arch.conclusion

    def test_arch_test_invalid_inputs(self, white_noise):
        """Test ARCH test with invalid inputs."""
        # Test with NaN
        data_nan = white_noise.copy()
        data_nan[10] = np.nan
        with pytest.raises(ValueError):
            arch_test(data_nan)

        # Test with Inf
        data_inf = white_noise.copy()
        data_inf[10] = np.inf
        with pytest.raises(ValueError):
            arch_test(data_inf)

        # Test with negative lags
        with pytest.raises(ValueError):
            arch_test(white_noise, lags=-1)

        # Test with lags >= nobs
        with pytest.raises(ValueError):
            arch_test(white_noise, lags=len(white_noise))

    def test_information_criteria(self):
        """Test information criteria calculation."""
        # Create sample values
        loglikelihood = -100.0
        nobs = 100
        nparams = 5
        model_name = "Test Model"

        # Calculate information criteria
        result = information_criteria(loglikelihood, nobs, nparams, model_name)

        # Check result attributes
        assert result.aic == -2 * loglikelihood + 2 * nparams
        assert result.bic == -2 * loglikelihood + nparams * np.log(nobs)
        assert result.hqic == -2 * loglikelihood + 2 * nparams * np.log(np.log(nobs))
        assert result.loglikelihood == loglikelihood
        assert result.nobs == nobs
        assert result.nparams == nparams
        assert result.model_name == model_name

        # Check string representation
        str_result = str(result)
        assert model_name in str_result
        assert "AIC" in str_result
        assert "BIC" in str_result
        assert "HQIC" in str_result

    def test_information_criteria_invalid_inputs(self):
        """Test information criteria with invalid inputs."""
        # Test with non-finite loglikelihood
        with pytest.raises(ValueError):
            information_criteria(np.nan, 100, 5)

        with pytest.raises(ValueError):
            information_criteria(np.inf, 100, 5)

        # Test with non-positive nobs
        with pytest.raises(ValueError):
            information_criteria(-100.0, 0, 5)

        with pytest.raises(ValueError):
            information_criteria(-100.0, -1, 5)

        # Test with non-positive nparams
        with pytest.raises(ValueError):
            information_criteria(-100.0, 100, 0)

        with pytest.raises(ValueError):
            information_criteria(-100.0, 100, -1)

    def test_model_diagnostics_integration(self, ar1_data):
        """Test integration of diagnostic tests with ARMA model."""
        # Fit an AR(1) model
        model = ARMAModel(ar_order=1, ma_order=0)
        result = model.fit(ar1_data)

        # Get residuals
        residuals = model.residuals

        # Perform Ljung-Box test on residuals
        lb_result = ljung_box(residuals, lags=10, df=1)

        # Check that the test was performed
        assert isinstance(lb_result, LjungBoxResult)
        assert np.isfinite(lb_result.test_statistic)
        assert np.isfinite(lb_result.p_value)

        # Residuals from a well-specified model should not show autocorrelation
        assert lb_result.p_value > 0.05
        assert "fail to reject" in lb_result.conclusion

        # Perform Jarque-Bera test on residuals
        jb_result = jarque_bera(residuals)

        # Check that the test was performed
        assert isinstance(jb_result, JarqueBeraResult)
        assert np.isfinite(jb_result.test_statistic)
        assert np.isfinite(jb_result.p_value)

        # Residuals from a well-specified model should be approximately normal
        assert jb_result.p_value > 0.05
        assert "fail to reject" in jb_result.conclusion


# Add fixtures for test data
@pytest.fixture
def ar1_data() -> np.ndarray:
    """Generate AR(1) data for testing."""
    np.random.seed(42)
