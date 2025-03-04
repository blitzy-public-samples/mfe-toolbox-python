# tests/test_time_series.py
'''
Tests for time series analysis functionality in the MFE Toolbox.

This module contains comprehensive tests for time series analysis functionality
including ARMA/ARMAX model estimation, forecasting, diagnostics, and seasonal
differencing. It verifies correct parameter estimation, model specification,
and diagnostic statistics for time series models.
'''
import asyncio
import math
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays
from scipy import stats

from mfe.models.time_series import ARMA, SARIMA
from mfe.models.time_series.diagnostics import ljung_box, lm_test
from mfe.models.time_series.unit_root import adf_test
from mfe.models.time_series.forecast import forecast_arma


class TestARMAModel:
    """Tests for ARMA model estimation and forecasting."""

    def test_ar1_estimation(self, ar1_process, assert_array_equal):
        """Test that AR(1) model correctly estimates parameters."""
        # Fit AR(1) model
        model = ARMA().fit(ar1_process, ar_order=1, ma_order=0, include_constant=True)

        # Check that parameters are close to expected values
        # AR(1) process fixture uses ar_coef=0.7
        assert 0.65 <= model.params['ar.1'] <= 0.75, "AR(1) coefficient not correctly estimated"

        # Check that model attributes are correctly set
        assert model.ar_order == 1, "AR order not correctly set"
        assert model.ma_order == 0, "MA order not correctly set"
        assert model.include_constant == True, "Constant inclusion flag not correctly set"

        # Check that residuals have expected properties
        assert len(model.residuals) == len(ar1_process), "Residual length doesn't match data length"

        # Check that fitted values + residuals = original data
        reconstructed_data = model.fitted_values + model.residuals
        assert_array_equal(reconstructed_data, ar1_process,
                           err_msg="Fitted values + residuals doesn't equal original data")

        # Check that log-likelihood, AIC, and BIC are computed
        assert isinstance(model.log_likelihood, float), "Log-likelihood not computed"
        assert isinstance(model.aic, float), "AIC not computed"
        assert isinstance(model.bic, float), "BIC not computed"

    def test_ma1_estimation(self, ma1_process, assert_array_equal):
        """Test that MA(1) model correctly estimates parameters."""
        # Fit MA(1) model
        model = ARMA().fit(ma1_process, ar_order=0, ma_order=1, include_constant=True)

        # Check that parameters are close to expected values
        # MA(1) process fixture uses ma_coef=0.7
        assert 0.65 <= model.params['ma.1'] <= 0.75, "MA(1) coefficient not correctly estimated"

        # Check that model attributes are correctly set
        assert model.ar_order == 0, "AR order not correctly set"
        assert model.ma_order == 1, "MA order not correctly set"
        assert model.include_constant == True, "Constant inclusion flag not correctly set"

        # Check that residuals have expected properties
        assert len(model.residuals) == len(ma1_process), "Residual length doesn't match data length"

        # Check that fitted values + residuals = original data
        reconstructed_data = model.fitted_values + model.residuals
        assert_array_equal(reconstructed_data, ma1_process,
                           err_msg="Fitted values + residuals doesn't equal original data")

    def test_arma11_estimation(self, arma11_process, assert_array_equal):
        """Test that ARMA(1,1) model correctly estimates parameters."""
        # Fit ARMA(1,1) model
        model = ARMA().fit(arma11_process, ar_order=1, ma_order=1, include_constant=True)

        # Check that parameters are close to expected values
        # ARMA(1,1) process fixture uses ar_coef=0.7, ma_coef=0.3
        assert 0.65 <= model.params['ar.1'] <= 0.75, "AR(1) coefficient not correctly estimated"
        assert 0.25 <= model.params['ma.1'] <= 0.35, "MA(1) coefficient not correctly estimated"

        # Check that model attributes are correctly set
        assert model.ar_order == 1, "AR order not correctly set"
        assert model.ma_order == 1, "MA order not correctly set"
        assert model.include_constant == True, "Constant inclusion flag not correctly set"

        # Check that residuals have expected properties
        assert len(model.residuals) == len(arma11_process), "Residual length doesn't match data length"

        # Check that fitted values + residuals = original data
        reconstructed_data = model.fitted_values + model.residuals
        assert_array_equal(reconstructed_data, arma11_process,
                           err_msg="Fitted values + residuals doesn't equal original data")

    def test_armax_estimation(self, ar1_process, rng, assert_array_equal):
        """Test that ARMAX model correctly estimates parameters with exogenous variables."""
        # Generate exogenous variables
        n = len(ar1_process)
        exog = rng.standard_normal((n, 2))

        # Add exogenous effect to the process
        beta = np.array([0.5, -0.3])
        y = ar1_process + exog @ beta

        # Fit ARMAX model
        model = ARMA().fit(y, ar_order=1, ma_order=0, include_constant=True, exog=exog)

        # Check that parameters are close to expected values
        assert 0.65 <= model.params['ar.1'] <= 0.75, "AR(1) coefficient not correctly estimated"
        assert 0.45 <= model.params['exog.1'] <= 0.55, "First exogenous coefficient not correctly estimated"
        assert -0.35 <= model.params['exog.2'] <= -0.25, "Second exogenous coefficient not correctly estimated"

        # Check that model attributes are correctly set
        assert model.ar_order == 1, "AR order not correctly set"
        assert model.ma_order == 0, "MA order not correctly set"
        assert model.include_constant == True, "Constant inclusion flag not correctly set"
        assert model.exog is not None, "Exogenous variables not stored"

        # Check that residuals have expected properties
        assert len(model.residuals) == len(y), "Residual length doesn't match data length"

        # Check that fitted values + residuals = original data
        reconstructed_data = model.fitted_values + model.residuals
        assert_array_equal(reconstructed_data, y,
                           err_msg="Fitted values + residuals doesn't equal original data")

    def test_pandas_input(self, ar1_process, assert_series_equal):
        """Test that ARMA model works correctly with pandas Series input."""
        # Convert to pandas Series with DatetimeIndex
        dates = pd.date_range(start='2020-01-01', periods=len(ar1_process), freq='D')
        series = pd.Series(ar1_process, index=dates)

        # Fit AR(1) model
        model = ARMA().fit(series, ar_order=1, ma_order=0, include_constant=True)

        # Check that parameters are close to expected values
        assert 0.65 <= model.params['ar.1'] <= 0.75, "AR(1) coefficient not correctly estimated"

        # Check that output preserves index
        assert isinstance(model.fitted_values, pd.Series), "Fitted values not returned as Series"
        assert isinstance(model.residuals, pd.Series), "Residuals not returned as Series"
        assert_series_equal(model.fitted_values.index, series.index,
                            err_msg="Fitted values index doesn't match input index")
        assert_series_equal(model.residuals.index, series.index,
                            err_msg="Residuals index doesn't match input index")

        # Check that fitted values + residuals = original data
        reconstructed_data = model.fitted_values + model.residuals
        assert_series_equal(reconstructed_data, series,
                            err_msg="Fitted values + residuals doesn't equal original data")

    def test_forecast(self, arma11_process):
        """Test ARMA model forecasting functionality."""
        # Fit ARMA(1,1) model
        model = ARMA().fit(arma11_process, ar_order=1, ma_order=1, include_constant=True)

        # Generate forecasts
        horizon = 10
        forecasts, forecast_errors = model.forecast(horizon=horizon)

        # Check forecast dimensions
        assert len(forecasts) == horizon, "Forecast length doesn't match horizon"
        assert len(forecast_errors) == horizon, "Forecast error length doesn't match horizon"

        # Check that forecast errors increase with horizon
        assert np.all(np.diff(forecast_errors) >= 0), "Forecast errors don't increase with horizon"

        # Test with pandas Series input
        dates = pd.date_range(start='2020-01-01', periods=len(arma11_process), freq='D')
        series = pd.Series(arma11_process, index=dates)

        model_pd = ARMA().fit(series, ar_order=1, ma_order=1, include_constant=True)
        forecasts_pd, forecast_errors_pd = model_pd.forecast(horizon=horizon)

        # Check that forecasts are returned as Series with extended index
        assert isinstance(forecasts_pd, pd.Series), "Forecasts not returned as Series"
        assert isinstance(forecast_errors_pd, pd.Series), "Forecast errors not returned as Series"
        assert len(forecasts_pd.index) == horizon, "Forecast index length doesn't match horizon"
        assert forecasts_pd.index[0] == dates[-1] + pd.Timedelta(days=1), "Forecast index doesn't continue from data"

    def test_forecast_with_exog(self, ar1_process, rng):
        """Test ARMA model forecasting with exogenous variables."""
        # Generate exogenous variables
        n = len(ar1_process)
        exog = rng.standard_normal((n, 2))

        # Add exogenous effect to the process
        beta = np.array([0.5, -0.3])
        y = ar1_process + exog @ beta

        # Fit ARMAX model
        model = ARMA().fit(y, ar_order=1, ma_order=0, include_constant=True, exog=exog)

        # Generate future exogenous variables
        horizon = 10
        future_exog = rng.standard_normal((horizon, 2))

        # Generate forecasts
        forecasts, forecast_errors = model.forecast(horizon=horizon, exog=future_exog)

        # Check forecast dimensions
        assert len(forecasts) == horizon, "Forecast length doesn't match horizon"
        assert len(forecast_errors) == horizon, "Forecast error length doesn't match horizon"

        # Check that forecast errors increase with horizon
        assert np.all(np.diff(forecast_errors) >= 0), "Forecast errors don't increase with horizon"

        # Test with missing future exogenous variables
        with pytest.raises(ValueError, match="Exogenous variables must be provided for forecasting"):
            model.forecast(horizon=horizon)

    @pytest.mark.asyncio
    async def test_async_fit(self, ar1_process):
        """Test asynchronous fitting of ARMA model."""
        # Define progress callback
        progress_updates = []

        async def progress_callback(percent, message):
            progress_updates.append((percent, message))

        # Fit AR(1) model asynchronously
        model = await ARMA().fit_async(ar1_process, ar_order=1, ma_order=0,
                                       include_constant=True, progress_callback=progress_callback)

        # Check that parameters are close to expected values
        assert 0.65 <= model.params['ar.1'] <= 0.75, "AR(1) coefficient not correctly estimated"

        # Check that progress callback was called
        assert len(progress_updates) > 0, "Progress callback was not called"
        assert progress_updates[-1][0] == 100.0, "Final progress update not 100%"

    @pytest.mark.asyncio
    async def test_async_forecast(self, arma11_process):
        """Test asynchronous forecasting with ARMA model."""
        # Fit ARMA(1,1) model
        model = ARMA().fit(arma11_process, ar_order=1, ma_order=1, include_constant=True)

        # Define progress callback
        progress_updates = []

        async def progress_callback(percent, message):
            progress_updates.append((percent, message))

        # Generate forecasts asynchronously
        horizon = 10
        forecasts, forecast_errors = await model.forecast_async(
            horizon=horizon, progress_callback=progress_callback)

        # Check forecast dimensions
        assert len(forecasts) == horizon, "Forecast length doesn't match horizon"
        assert len(forecast_errors) == horizon, "Forecast error length doesn't match horizon"

        # Check that progress callback was called
        assert len(progress_updates) > 0, "Progress callback was not called"
        assert progress_updates[-1][0] == 100.0, "Final progress update not 100%"

    def test_model_selection(self, arma11_process):
        """Test automatic model selection for ARMA models."""
        # Use model selection to find best order
        model = ARMA().select_order(arma11_process, max_ar=2, max_ma=2, include_constant=True)

        # Check that selected orders are correct
        assert model.ar_order == 1, "Selected AR order is incorrect"
        assert model.ma_order == 1, "Selected MA order is incorrect"

        # Check that parameters are close to expected values
        assert 0.65 <= model.params['ar.1'] <= 0.75, "AR(1) coefficient not correctly estimated"
        assert 0.25 <= model.params['ma.1'] <= 0.35, "MA(1) coefficient not correctly estimated"

    def test_invalid_inputs(self):
        """Test that ARMA model properly validates inputs."""
        # Create model instance
        model = ARMA()

        # Test with empty array
        with pytest.raises(ValueError, match="Input array must contain data"):
            model.fit(np.array([]), ar_order=1, ma_order=0)

        # Test with NaN values
        with pytest.raises(ValueError, match="Input array contains NaN values"):
            model.fit(np.array([1.0, 2.0, np.nan, 4.0]), ar_order=1, ma_order=0)

        # Test with infinite values
        with pytest.raises(ValueError, match="Input array contains infinite values"):
            model.fit(np.array([1.0, 2.0, np.inf, 4.0]), ar_order=1, ma_order=0)

        # Test with negative AR order
        with pytest.raises(ValueError, match="AR order must be non-negative"):
            model.fit(np.array([1.0, 2.0, 3.0]), ar_order=-1, ma_order=0)

        # Test with negative MA order
        with pytest.raises(ValueError, match="MA order must be non-negative"):
            model.fit(np.array([1.0, 2.0, 3.0]), ar_order=0, ma_order=-1)

        # Test with AR+MA order too large for sample size
        with pytest.raises(ValueError, match="Sample size too small for specified model order"):
            model.fit(np.array([1.0, 2.0, 3.0]), ar_order=2, ma_order=2)

    @given(arrays(dtype=np.float64, shape=st.integers(20, 100),
                  elements=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)))
    def test_property_based(self, data):
        """Property-based test for ARMA model using hypothesis."""
        # Fit AR(1) model
        try:
            model = ARMA().fit(data, ar_order=1, ma_order=0, include_constant=True)

            # Basic properties that should always hold
            assert len(model.residuals) == len(data), "Residual length doesn't match data length"
            assert len(model.fitted_values) == len(data), "Fitted values length doesn't match data length"
            assert np.isclose(model.fitted_values + model.residuals, data).all(), \
                "Fitted values + residuals doesn't equal original data"
            assert model.aic <= model.bic, "AIC should be less than or equal to BIC"

            # Test forecasting
            horizon = 5
            forecasts, forecast_errors = model.forecast(horizon=horizon)
            assert len(forecasts) == horizon, "Forecast length doesn't match horizon"
            assert len(forecast_errors) == horizon, "Forecast error length doesn't match horizon"
            assert np.all(forecast_errors >= 0), "Forecast errors should be non-negative"
        except (ValueError, RuntimeError):
            # Some random arrays might cause numerical issues, which is acceptable
            # in property-based testing
            pass

    def test_type_hints_and_dataclass_validation(self, ar1_process):
        """Test that type hints and dataclass validation work correctly."""
        # Fit AR(1) model
        model = ARMA().fit(ar1_process, ar_order=1, ma_order=0, include_constant=True)

        # Check that model parameters are stored in a dataclass
        assert hasattr(model, 'params'), "Model should have params attribute"
        assert isinstance(model.params, dict), "Model params should be a dictionary"

        # Check that model results include proper type annotations
        assert hasattr(model, 'residuals'), "Model should have residuals attribute"
        assert hasattr(model, 'fitted_values'), "Model should have fitted_values attribute"
        assert hasattr(model, 'log_likelihood'), "Model should have log_likelihood attribute"
        assert hasattr(model, 'aic'), "Model should have aic attribute"
        assert hasattr(model, 'bic'), "Model should have bic attribute"


class TestSARIMAModel:
    """Tests for SARIMA model estimation and forecasting."""

    def test_sarima_estimation(self, rng):
        """Test that SARIMA model correctly estimates parameters."""
        # Generate seasonal data
        n = 200
        seasonal_period = 12
        t = np.arange(n)
        seasonal_component = 2 * np.sin(2 * np.pi * t / seasonal_period)
        ar_component = np.zeros(n)
        ar_component[0] = rng.standard_normal()
        for i in range(1, n):
            ar_component[i] = 0.7 * ar_component[i-1] + rng.standard_normal()

        data = ar_component + seasonal_component

        # Fit SARIMA model
        model = SARIMA().fit(data, ar_order=1, ma_order=0, seasonal_ar_order=1,
                             seasonal_ma_order=0, seasonal_period=seasonal_period,
                             include_constant=True)

        # Check that parameters are close to expected values
        assert 0.65 <= model.params['ar.1'] <= 0.75, "AR(1) coefficient not correctly estimated"
        assert 0.8 <= model.params['seasonal_ar.1'] <= 1.0, "Seasonal AR coefficient not correctly estimated"

        # Check that model attributes are correctly set
        assert model.ar_order == 1, "AR order not correctly set"
        assert model.ma_order == 0, "MA order not correctly set"
        assert model.seasonal_ar_order == 1, "Seasonal AR order not correctly set"
        assert model.seasonal_ma_order == 0, "Seasonal MA order not correctly set"
        assert model.seasonal_period == seasonal_period, "Seasonal period not correctly set"
        assert model.include_constant == True, "Constant inclusion flag not correctly set"

        # Check that residuals have expected properties
        assert len(model.residuals) == len(data), "Residual length doesn't match data length"

        # Check that fitted values + residuals = original data
        reconstructed_data = model.fitted_values + model.residuals
        np.testing.assert_allclose(reconstructed_data, data, rtol=1e-10,
                                   err_msg="Fitted values + residuals doesn't equal original data")

    def test_sarima_with_differencing(self, rng):
        """Test SARIMA model with differencing."""
        # Generate non-stationary seasonal data
        n = 200
        seasonal_period = 12
        t = np.arange(n)
        trend = 0.01 * t  # Linear trend
        seasonal_component = 2 * np.sin(2 * np.pi * t / seasonal_period)
        ar_component = np.zeros(n)
        ar_component[0] = rng.standard_normal()
        for i in range(1, n):
            ar_component[i] = 0.7 * ar_component[i-1] + rng.standard_normal()

        data = trend + ar_component + seasonal_component

        # Fit SARIMA model with differencing
        model = SARIMA().fit(data, ar_order=1, ma_order=0, seasonal_ar_order=1,
                             seasonal_ma_order=0, seasonal_period=seasonal_period,
                             d=1, seasonal_d=1, include_constant=True)

        # Check that model attributes are correctly set
        assert model.d == 1, "Regular differencing order not correctly set"
        assert model.seasonal_d == 1, "Seasonal differencing order not correctly set"

        # Check that residuals have expected properties
        assert len(model.residuals) == len(data) - seasonal_period - 1, \
            "Residual length incorrect after differencing"

        # Check forecasting with differencing
        horizon = 24
        forecasts, forecast_errors = model.forecast(horizon=horizon)

        # Check forecast dimensions
        assert len(forecasts) == horizon, "Forecast length doesn't match horizon"
        assert len(forecast_errors) == horizon, "Forecast error length doesn't match horizon"

        # Check that forecast errors increase with horizon
        assert np.all(np.diff(forecast_errors) >= 0), "Forecast errors don't increase with horizon"

    def test_sarima_to_arma_conversion(self, rng):
        """Test conversion between SARIMA and ARMA representations."""
        # Generate seasonal data
        n = 200
        seasonal_period = 4  # Small period for easier testing
        t = np.arange(n)
        seasonal_component = 2 * np.sin(2 * np.pi * t / seasonal_period)
        ar_component = np.zeros(n)
        ar_component[0] = rng.standard_normal()
        for i in range(1, n):
            ar_component[i] = 0.7 * ar_component[i-1] + rng.standard_normal()

        data = ar_component + seasonal_component

        # Fit SARIMA model
        sarima_model = SARIMA().fit(data, ar_order=1, ma_order=0, seasonal_ar_order=1,
                                    seasonal_ma_order=0, seasonal_period=seasonal_period,
                                    include_constant=True)

        # Convert to ARMA representation
        arma_params = sarima_model.to_arma_representation()

        # Check that ARMA representation has correct orders
        assert len(arma_params['ar']) == 1 + seasonal_period, \
            "AR parameters in ARMA representation have incorrect length"

        # Fit equivalent ARMA model
        arma_model = ARMA().fit(data, ar_order=1 + seasonal_period, ma_order=0, include_constant=True)

        # Compare forecasts from both models
        horizon = 12
        sarima_forecasts, _ = sarima_model.forecast(horizon=horizon)
        arma_forecasts, _ = arma_model.forecast(horizon=horizon)

        # Forecasts should be similar (not exactly equal due to different estimation methods)
        np.testing.assert_allclose(sarima_forecasts, arma_forecasts, rtol=0.1,
                                   err_msg="SARIMA and equivalent ARMA forecasts differ significantly")

    def test_seasonal_differencing(self, rng):
        """Test seasonal differencing operation."""
        # Generate seasonal data
        n = 100
        seasonal_period = 12
        t = np.arange(n)
        seasonal_component = 2 * np.sin(2 * np.pi * t / seasonal_period)
        data = seasonal_component + rng.standard_normal(n)

        # Apply seasonal differencing
        from mfe.models.time_series import seasonal_difference
        diff_data = seasonal_difference(data, seasonal_period=seasonal_period)

        # Check that seasonal pattern is reduced
        from scipy import stats

        # Calculate autocorrelation at seasonal lag before differencing
        acf_before = np.corrcoef(data[seasonal_period:], data[:-seasonal_period])[0, 1]

        # Calculate autocorrelation at seasonal lag after differencing
        acf_after = np.corrcoef(diff_data[seasonal_period:], diff_data[:-seasonal_period])[0, 1]

        # Seasonal differencing should reduce autocorrelation at seasonal lag
        assert abs(acf_after) < abs(acf_before), \
            "Seasonal differencing didn't reduce autocorrelation at seasonal lag"

        # Test with pandas Series
        dates = pd.date_range(start='2020-01-01', periods=n, freq='M')
        series = pd.Series(data, index=dates)

        diff_series = seasonal_difference(series, seasonal_period=seasonal_period)

        # Check that output is a Series with correct index
        assert isinstance(diff_series, pd.Series), "Output not returned as Series"
        assert len(diff_series) == n - seasonal_period, "Output length incorrect"
        assert diff_series.index[0] == dates[seasonal_period], "Output index incorrect"

    @pytest.mark.asyncio
    async def test_async_sarima_fit(self, rng):
        """Test asynchronous fitting of SARIMA model."""
        # Generate seasonal data
        n = 200
        seasonal_period = 12
        t = np.arange(n)
        seasonal_component = 2 * np.sin(2 * np.pi * t / seasonal_period)
        ar_component = np.zeros(n)
        ar_component[0] = rng.standard_normal()
        for i in range(1, n):
            ar_component[i] = 0.7 * ar_component[i-1] + rng.standard_normal()

        data = ar_component + seasonal_component

        # Define progress callback
        progress_updates = []

        async def progress_callback(percent, message):
            progress_updates.append((percent, message))

        # Fit SARIMA model asynchronously
        model = await SARIMA().fit_async(data, ar_order=1, ma_order=0, seasonal_ar_order=1,
                                         seasonal_ma_order=0, seasonal_period=seasonal_period,
                                         include_constant=True, progress_callback=progress_callback)

        # Check that parameters are close to expected values
        assert 0.65 <= model.params['ar.1'] <= 0.75, "AR(1) coefficient not correctly estimated"
        assert 0.8 <= model.params['seasonal_ar.1'] <= 1.0, "Seasonal AR coefficient not correctly estimated"

        # Check that progress callback was called
        assert len(progress_updates) > 0, "Progress callback was not called"
        assert progress_updates[-1][0] == 100.0, "Final progress update not 100%"

    def test_invalid_inputs(self):
        """Test that SARIMA model properly validates inputs."""
        # Create model instance
        model = SARIMA()

        # Test with negative seasonal period
        with pytest.raises(ValueError, match="Seasonal period must be positive"):
            model.fit(np.array([1.0, 2.0, 3.0]), ar_order=0, ma_order=0,
                      seasonal_ar_order=1, seasonal_ma_order=0, seasonal_period=-1)

        # Test with seasonal period of 1
        with pytest.raises(ValueError, match="Seasonal period must be greater than 1"):
            model.fit(np.array([1.0, 2.0, 3.0]), ar_order=0, ma_order=0,
                      seasonal_ar_order=1, seasonal_ma_order=0, seasonal_period=1)

        # Test with negative differencing order
        with pytest.raises(ValueError, match="Differencing order must be non-negative"):
            model.fit(np.array([1.0, 2.0, 3.0]), ar_order=0, ma_order=0,
                      seasonal_ar_order=0, seasonal_ma_order=0, seasonal_period=2, d=-1)

        # Test with negative seasonal differencing order
        with pytest.raises(ValueError, match="Seasonal differencing order must be non-negative"):
            model.fit(np.array([1.0, 2.0, 3.0]), ar_order=0, ma_order=0,
                      seasonal_ar_order=0, seasonal_ma_order=0, seasonal_period=2, seasonal_d=-1)

        # Test with sample size too small after differencing
        with pytest.raises(ValueError, match="Sample size too small after differencing"):
            model.fit(np.array([1.0, 2.0, 3.0]), ar_order=0, ma_order=0,
                      seasonal_ar_order=0, seasonal_ma_order=0, seasonal_period=2, d=1, seasonal_d=1)


class TestDiagnosticTests:
    """Tests for time series diagnostic tests."""

    def test_ljung_box_test(self, ar1_process, ma1_process, random_univariate_series):
        """Test Ljung-Box test for serial correlation."""
        # Test on white noise (should not reject)
        lb_stat, p_value = ljung_box(random_univariate_series, lags=10)
        assert p_value > 0.05, "Ljung-Box incorrectly rejected white noise"

        # Test on AR(1) process (should reject)
        lb_stat, p_value = ljung_box(ar1_process, lags=10)
        assert p_value < 0.05, "Ljung-Box failed to reject AR(1) process"

        # Test on MA(1) process (should reject)
        lb_stat, p_value = ljung_box(ma1_process, lags=10)
        assert p_value < 0.05, "Ljung-Box failed to reject MA(1) process"

        # Test on ARMA model residuals (should not reject)
        model = ARMA().fit(ar1_process, ar_order=1, ma_order=0, include_constant=True)
        lb_stat, p_value = ljung_box(model.residuals, lags=10)
        assert p_value > 0.01, "Ljung-Box incorrectly rejected ARMA model residuals"

        # Test with pandas Series
        series = pd.Series(random_univariate_series)
        lb_stat_series, p_value_series = ljung_box(series, lags=10)

        # Results should be the same regardless of input type
        np.testing.assert_allclose(lb_stat, lb_stat_series, rtol=1e-10,
                                   err_msg="LB statistic differs between array and Series")
        np.testing.assert_allclose(p_value, p_value_series, rtol=1e-10,
                                   err_msg="P-value differs between array and Series")

    def test_lm_test(self, random_univariate_series, garch_process):
        """Test LM test for ARCH effects."""
        # Test on white noise (should not reject)
        lm_stat, p_value = lm_test(random_univariate_series, lags=10)
        assert p_value > 0.05, "LM test incorrectly rejected white noise"

        # Test on GARCH process (should reject)
        returns, _ = garch_process
        lm_stat, p_value = lm_test(returns, lags=10)
        assert p_value < 0.05, "LM test failed to reject GARCH process"

        # Test with pandas Series
        series = pd.Series(random_univariate_series)
        lm_stat_series, p_value_series = lm_test(series, lags=10)

        # Results should be the same regardless of input type
        np.testing.assert_allclose(lm_stat, lm_stat_series, rtol=1e-10,
                                   err_msg="LM statistic differs between array and Series")
        np.testing.assert_allclose(p_value, p_value_series, rtol=1e-10,
                                   err_msg="P-value differs between array and Series")

    def test_adf_test(self, rng):
        """Test Augmented Dickey-Fuller test for unit roots."""
        # Generate stationary data
        stationary_data = rng.standard_normal(100)

        # Generate non-stationary data (random walk)
        non_stationary_data = np.cumsum(rng.standard_normal(100))

        # Test on stationary data (should reject unit root)
        adf_stat, p_value, critical_values, lags = adf_test(stationary_data)
        assert p_value < 0.05, "ADF test failed to reject unit root for stationary data"

        # Test on non-stationary data (should not reject unit root)
        adf_stat, p_value, critical_values, lags = adf_test(non_stationary_data)
        assert p_value > 0.05, "ADF test incorrectly rejected unit root for non-stationary data"

        # Test with pandas Series
        series = pd.Series(stationary_data)
        adf_stat_series, p_value_series, critical_values_series, lags_series = adf_test(series)

        # Results should be the same regardless of input type
        np.testing.assert_allclose(adf_stat, adf_stat_series, rtol=1e-10,
                                   err_msg="ADF statistic differs between array and Series")
        np.testing.assert_allclose(p_value, p_value_series, rtol=1e-10,
                                   err_msg="P-value differs between array and Series")

    @pytest.mark.asyncio
    async def test_async_diagnostic_tests(self, random_univariate_series, ar1_process):
        """Test asynchronous versions of diagnostic tests."""
        # Test async Ljung-Box
        from mfe.models.time_series.diagnostics import ljung_box_async
        lb_stat, p_value = await ljung_box_async(random_univariate_series, lags=10)
        assert isinstance(lb_stat, float), "LB statistic not returned as float"
        assert isinstance(p_value, float), "P-value not returned as float"

        # Test async LM test
        from mfe.models.time_series.diagnostics import lm_test_async
        lm_stat, p_value = await lm_test_async(random_univariate_series, lags=10)
        assert isinstance(lm_stat, float), "LM statistic not returned as float"
        assert isinstance(p_value, float), "P-value not returned as float"

        # Test async ADF test
        from mfe.models.time_series.unit_root import adf_test_async
        adf_stat, p_value, critical_values, lags = await adf_test_async(ar1_process)
        assert isinstance(adf_stat, float), "ADF statistic not returned as float"
        assert isinstance(p_value, float), "P-value not returned as float"
        assert isinstance(critical_values, dict), "Critical values not returned as dict"
        assert isinstance(lags, int), "Lags not returned as int"


class TestForecastingFunctions:
    """Tests for standalone forecasting functions."""

    def test_forecast_arma(self, arma11_process):
        """Test standalone ARMA forecasting function."""
        # Fit ARMA(1,1) model
        model = ARMA().fit(arma11_process, ar_order=1, ma_order=1, include_constant=True)

        # Extract parameters
        ar_params = np.array([model.params.get(f'ar.{i+1}', 0) for i in range(1)])
        ma_params = np.array([model.params.get(f'ma.{i+1}', 0) for i in range(1)])
        constant = model.params.get('const', 0)

        # Generate forecasts using standalone function
        horizon = 10
        last_values = arma11_process[-1:]
        last_errors = model.residuals[-1:]

        forecasts, forecast_errors = forecast_arma(
            last_values, last_errors, ar_params, ma_params, constant, horizon)

        # Check forecast dimensions
        assert len(forecasts) == horizon, "Forecast length doesn't match horizon"
        assert len(forecast_errors) == horizon, "Forecast error length doesn't match horizon"

        # Compare with model's forecast method
        model_forecasts, model_forecast_errors = model.forecast(horizon=horizon)

        # Forecasts should be identical
        np.testing.assert_allclose(forecasts, model_forecasts, rtol=1e-10,
                                   err_msg="Standalone forecasts differ from model forecasts")
        np.testing.assert_allclose(forecast_errors, model_forecast_errors, rtol=1e-10,
                                   err_msg="Standalone forecast errors differ from model forecast errors")

    def test_forecast_arma_with_exog(self, ar1_process, rng):
        """Test standalone ARMA forecasting function with exogenous variables."""
        # Generate exogenous variables
        n = len(ar1_process)
        exog = rng.standard_normal((n, 2))

        # Add exogenous effect to the process
        beta = np.array([0.5, -0.3])
        y = ar1_process + exog @ beta

        # Fit ARMAX model
        model = ARMA().fit(y, ar_order=1, ma_order=0, include_constant=True, exog=exog)

        # Extract parameters
        ar_params = np.array([model.params.get(f'ar.{i+1}', 0) for i in range(1)])
        ma_params = np.array([])
        constant = model.params.get('const', 0)
        exog_params = np.array([model.params.get(f'exog.{i+1}', 0) for i in range(2)])

        # Generate future exogenous variables
        horizon = 10
        future_exog = rng.standard_normal((horizon, 2))

        # Generate forecasts using standalone function
        from mfe.models.time_series.forecast import forecast_armax
        last_values = y[-1:]
        last_errors = model.residuals[-1:]

        forecasts, forecast_errors = forecast_armax(
            last_values, last_errors, ar_params, ma_params, constant,
            exog_params, future_exog, horizon)

        # Check forecast dimensions
        assert len(forecasts) == horizon, "Forecast length doesn't match horizon"
        assert len(forecast_errors) == horizon, "Forecast error length doesn't match horizon"

        # Compare with model's forecast method
        model_forecasts, model_forecast_errors = model.forecast(horizon=horizon, exog=future_exog)

        # Forecasts should be identical
        np.testing.assert_allclose(forecasts, model_forecasts, rtol=1e-10,
                                   err_msg="Standalone forecasts differ from model forecasts")
        np.testing.assert_allclose(forecast_errors, model_forecast_errors, rtol=1e-10,
                                   err_msg="Standalone forecast errors differ from model forecast errors")


class TestIntegrationTests:
    """Integration tests for time series functionality."""

    def test_model_diagnostics_workflow(self, ar1_process):
        """Test complete workflow from model estimation to diagnostics."""
        # Fit AR(1) model
        model = ARMA().fit(ar1_process, ar_order=1, ma_order=0, include_constant=True)

        # Check residuals for autocorrelation
        lb_stat, p_value = ljung_box(model.residuals, lags=10)
        assert p_value > 0.01, "Ljung-Box incorrectly detected autocorrelation in residuals"

        # Check residuals for ARCH effects
        lm_stat, p_value = lm_test(model.residuals, lags=10)
        assert p_value > 0.01, "LM test incorrectly detected ARCH effects in residuals"

        # Check residuals for normality
        from scipy import stats
        _, p_value = stats.jarque_bera(model.residuals)
        assert p_value > 0.01, "Jarque-Bera incorrectly rejected normality of residuals"

        # Generate forecasts
        horizon = 10
        forecasts, forecast_errors = model.forecast(horizon=horizon)

        # Check forecast properties
        assert len(forecasts) == horizon, "Forecast length doesn't match horizon"
        assert np.all(forecast_errors > 0), "Forecast errors should be positive"
        assert np.all(np.diff(forecast_errors) >= 0), "Forecast errors should increase with horizon"

    def test_statsmodels_integration(self, ar1_process):
        """Test integration with statsmodels."""
        try:
            import statsmodels.api as sm

            # Fit AR(1) model using MFE Toolbox
            mfe_model = ARMA().fit(ar1_process, ar_order=1, ma_order=0, include_constant=True)

            # Fit AR(1) model using statsmodels
            sm_model = sm.tsa.ARIMA(ar1_process, order=(1, 0, 0), trend='c').fit()

            # Compare parameter estimates
            assert abs(mfe_model.params['ar.1'] - sm_model.arparams[0]) < 0.05, \
                "AR parameter estimates differ significantly between MFE and statsmodels"
            assert abs(mfe_model.params['const'] - sm_model.params[0]) < 0.05, \
                "Constant estimates differ significantly between MFE and statsmodels"

            # Compare log-likelihood
            assert abs(mfe_model.log_likelihood - sm_model.llf) / abs(sm_model.llf) < 0.05, \
                "Log-likelihood differs significantly between MFE and statsmodels"

            # Compare forecasts
            horizon = 10
            mfe_forecasts, _ = mfe_model.forecast(horizon=horizon)
            sm_forecasts = sm_model.forecast(steps=horizon)

            # Forecasts should be similar (not exactly equal due to different implementations)
            np.testing.assert_allclose(mfe_forecasts, sm_forecasts, rtol=0.1,
                                       err_msg="Forecasts differ significantly between MFE and statsmodels")
        except ImportError:
            pytest.skip("statsmodels not available for comparison")

    def test_pandas_integration(self, ar1_process):
        """Test integration with pandas time series functionality."""
        # Create pandas Series with DatetimeIndex
        dates = pd.date_range(start='2020-01-01', periods=len(ar1_process), freq='D')
        series = pd.Series(ar1_process, index=dates, name='data')

        # Fit AR(1) model
        model = ARMA().fit(series, ar_order=1, ma_order=0, include_constant=True)

        # Check that output preserves index and name
        assert isinstance(model.fitted_values, pd.Series), "Fitted values not returned as Series"
        assert isinstance(model.residuals, pd.Series), "Residuals not returned as Series"
        assert model.fitted_values.index.equals(series.index), "Fitted values index doesn't match input index"
        assert model.residuals.index.equals(series.index), "Residuals index doesn't match input index"
        assert model.fitted_values.name == 'data', "Fitted values name not preserved"
        assert model.residuals.name == 'data', "Residuals name not preserved"

        # Generate forecasts
        horizon = 10
        forecasts, forecast_errors = model.forecast(horizon=horizon)

        # Check that forecasts are returned as Series with extended index
        assert isinstance(forecasts, pd.Series), "Forecasts not returned as Series"
        assert isinstance(forecast_errors, pd.Series), "Forecast errors not returned as Series"
        assert len(forecasts.index) == horizon, "Forecast index length doesn't match horizon"
        assert forecasts.index[0] == dates[-1] + pd.Timedelta(days=1), "Forecast index doesn't continue from data"
        assert forecasts.name == 'data', "Forecast name not preserved"
        assert forecast_errors.name == 'data', "Forecast error name not preserved"
