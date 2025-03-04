'''
Common pytest fixtures and configuration for the MFE Toolbox test suite.

This module provides shared fixtures and utilities for testing the MFE Toolbox,
including data generators, mock objects, and parameterized test utilities.
'''
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Callable, Any, Generator

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

# Constants for test data generation
DEFAULT_SEED = 12345
DEFAULT_SAMPLE_SIZE = 1000
DEFAULT_BURN = 500
DEFAULT_VOLATILITY = 0.2
DEFAULT_MEAN = 0.0


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """
    Provide a numpy random number generator with fixed seed for reproducible tests.
    
    Returns:
        np.random.Generator: Seeded random number generator
    """
    return np.random.default_rng(DEFAULT_SEED)


@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    """
    Create a temporary directory for test file operations.
    
    Yields:
        Path: Path to temporary directory that is cleaned up after the test
    """
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path)


@pytest.fixture(scope="function")
def temp_file() -> Generator[Path, None, None]:
    """
    Create a temporary file for test operations.
    
    Yields:
        Path: Path to temporary file that is cleaned up after the test
    """
    fd, path = tempfile.mkstemp()
    os.close(fd)
    temp_path = Path(path)
    try:
        yield temp_path
    finally:
        if temp_path.exists():
            temp_path.unlink()


@pytest.fixture
def random_univariate_series(rng: np.random.Generator) -> np.ndarray:
    """
    Generate a random univariate time series with standard normal distribution.
    
    Args:
        rng: Random number generator
        
    Returns:
        np.ndarray: Random univariate time series of shape (DEFAULT_SAMPLE_SIZE,)
    """
    return rng.standard_normal(DEFAULT_SAMPLE_SIZE)


@pytest.fixture
def random_multivariate_series(rng: np.random.Generator, n_assets: int = 5) -> np.ndarray:
    """
    Generate a random multivariate time series.
    
    Args:
        rng: Random number generator
        n_assets: Number of assets (columns) in the series
        
    Returns:
        np.ndarray: Random multivariate time series of shape (DEFAULT_SAMPLE_SIZE, n_assets)
    """
    return rng.standard_normal((DEFAULT_SAMPLE_SIZE, n_assets))


@pytest.fixture
def garch_process(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a GARCH(1,1) process with known parameters.
    
    Args:
        rng: Random number generator
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Returns (returns, volatility) arrays
    """
    # GARCH(1,1) parameters
    omega, alpha, beta = 0.05, 0.1, 0.85
    
    # Initialize arrays
    n = DEFAULT_SAMPLE_SIZE + DEFAULT_BURN
    returns = np.zeros(n)
    sigma2 = np.zeros(n)
    
    # Initial variance
    sigma2[0] = omega / (1 - alpha - beta)
    
    # Generate GARCH process
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
        returns[t] = rng.standard_normal() * np.sqrt(sigma2[t])
    
    # Discard burn-in period
    returns = returns[DEFAULT_BURN:]
    sigma2 = sigma2[DEFAULT_BURN:]
    
    return returns, np.sqrt(sigma2)


@pytest.fixture
def ar1_process(rng: np.random.Generator, ar_coef: float = 0.7) -> np.ndarray:
    """
    Generate an AR(1) process with specified coefficient.
    
    Args:
        rng: Random number generator
        ar_coef: AR(1) coefficient
        
    Returns:
        np.ndarray: AR(1) process time series
    """
    n = DEFAULT_SAMPLE_SIZE + DEFAULT_BURN
    series = np.zeros(n)
    
    # Generate AR(1) process
    for t in range(1, n):
        series[t] = ar_coef * series[t-1] + rng.standard_normal()
    
    # Discard burn-in period
    return series[DEFAULT_BURN:]


@pytest.fixture
def ma1_process(rng: np.random.Generator, ma_coef: float = 0.7) -> np.ndarray:
    """
    Generate an MA(1) process with specified coefficient.
    
    Args:
        rng: Random number generator
        ma_coef: MA(1) coefficient
        
    Returns:
        np.ndarray: MA(1) process time series
    """
    n = DEFAULT_SAMPLE_SIZE + DEFAULT_BURN
    innovations = rng.standard_normal(n)
    series = np.zeros(n)
    
    # Generate MA(1) process
    for t in range(1, n):
        series[t] = innovations[t] + ma_coef * innovations[t-1]
    
    # Discard burn-in period
    return series[DEFAULT_BURN:]


@pytest.fixture
def arma11_process(rng: np.random.Generator, ar_coef: float = 0.7, ma_coef: float = 0.3) -> np.ndarray:
    """
    Generate an ARMA(1,1) process with specified coefficients.
    
    Args:
        rng: Random number generator
        ar_coef: AR(1) coefficient
        ma_coef: MA(1) coefficient
        
    Returns:
        np.ndarray: ARMA(1,1) process time series
    """
    n = DEFAULT_SAMPLE_SIZE + DEFAULT_BURN
    innovations = rng.standard_normal(n)
    series = np.zeros(n)
    
    # Generate ARMA(1,1) process
    for t in range(1, n):
        series[t] = ar_coef * series[t-1] + innovations[t] + ma_coef * innovations[t-1]
    
    # Discard burn-in period
    return series[DEFAULT_BURN:]


@pytest.fixture
def heavy_tailed_series(rng: np.random.Generator, df: float = 5.0) -> np.ndarray:
    """
    Generate a heavy-tailed time series using Student's t-distribution.
    
    Args:
        rng: Random number generator
        df: Degrees of freedom for t-distribution
        
    Returns:
        np.ndarray: Heavy-tailed time series
    """
    return stats.t.rvs(df=df, size=DEFAULT_SAMPLE_SIZE, random_state=rng)


@pytest.fixture
def skewed_series(rng: np.random.Generator, skew: float = 0.5) -> np.ndarray:
    """
    Generate a skewed time series using skewed normal distribution.
    
    Args:
        rng: Random number generator
        skew: Skewness parameter
        
    Returns:
        np.ndarray: Skewed time series
    """
    return stats.skewnorm.rvs(a=skew, size=DEFAULT_SAMPLE_SIZE, random_state=rng)


@pytest.fixture
def high_frequency_data(rng: np.random.Generator, n_obs: int = 5000) -> pd.DataFrame:
    """
    Generate simulated high-frequency price data with timestamps.
    
    Args:
        rng: Random number generator
        n_obs: Number of observations
        
    Returns:
        pd.DataFrame: DataFrame with 'time' and 'price' columns
    """
    # Generate random timestamps over a trading day (6.5 hours)
    seconds_in_day = 6.5 * 60 * 60
    timestamps = np.sort(rng.uniform(0, seconds_in_day, n_obs))
    
    # Convert to pandas timestamps
    base_date = pd.Timestamp('2023-01-01 09:30:00')
    times = [base_date + pd.Timedelta(seconds=t) for t in timestamps]
    
    # Generate price path (random walk with drift)
    log_price = np.zeros(n_obs)
    log_price[0] = np.log(100.0)  # Start at price 100
    
    # Parameters
    drift = 0.0001  # Small positive drift
    vol = 0.001     # Small volatility for high-frequency data
    
    # Generate log price path
    for i in range(1, n_obs):
        time_diff = (timestamps[i] - timestamps[i-1]) / seconds_in_day
        log_price[i] = log_price[i-1] + drift * time_diff + vol * np.sqrt(time_diff) * rng.standard_normal()
    
    # Convert to price
    price = np.exp(log_price)
    
    # Add microstructure noise
    noise_level = 0.0001
    noisy_price = price * (1 + noise_level * rng.standard_normal(n_obs))
    
    # Create DataFrame
    return pd.DataFrame({'time': times, 'price': noisy_price})


@pytest.fixture
def multivariate_correlated_series(rng: np.random.Generator, n_assets: int = 5, correlation: float = 0.5) -> np.ndarray:
    """
    Generate multivariate time series with specified correlation structure.
    
    Args:
        rng: Random number generator
        n_assets: Number of assets
        correlation: Base correlation between assets
        
    Returns:
        np.ndarray: Correlated multivariate time series
    """
    # Create correlation matrix with specified correlation
    corr_matrix = np.ones((n_assets, n_assets)) * correlation
    np.fill_diagonal(corr_matrix, 1.0)
    
    # Ensure positive definite
    min_eig = np.min(np.linalg.eigvals(corr_matrix))
    if min_eig < 0:
        corr_matrix += np.eye(n_assets) * (abs(min_eig) + 0.01)
    
    # Generate correlated random variables
    chol = np.linalg.cholesky(corr_matrix)
    uncorrelated = rng.standard_normal((DEFAULT_SAMPLE_SIZE, n_assets))
    correlated = uncorrelated @ chol.T
    
    return correlated


@pytest.fixture
def dcc_process(rng: np.random.Generator, n_assets: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a DCC-GARCH process with known parameters.
    
    Args:
        rng: Random number generator
        n_assets: Number of assets
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Returns (returns, volatilities, correlations)
    """
    # DCC parameters
    a, b = 0.05, 0.93
    
    # GARCH parameters for each series
    garch_params = [(0.05, 0.1, 0.85) for _ in range(n_assets)]
    
    # Initialize arrays
    n = DEFAULT_SAMPLE_SIZE + DEFAULT_BURN
    returns = np.zeros((n, n_assets))
    volatilities = np.zeros((n, n_assets))
    correlations = np.zeros((n, n_assets, n_assets))
    
    # Initial unconditional correlation matrix (identity for simplicity)
    Q_bar = np.eye(n_assets)
    Q_t = Q_bar.copy()
    
    # Initial volatilities
    for i in range(n_assets):
        omega, alpha, beta = garch_params[i]
        volatilities[0, i] = np.sqrt(omega / (1 - alpha - beta))
    
    # Initial correlation
    correlations[0] = np.eye(n_assets)
    
    # Generate standardized innovations
    z = rng.standard_normal((n, n_assets))
    
    # Generate DCC process
    for t in range(1, n):
        # Update volatilities for each series
        for i in range(n_assets):
            omega, alpha, beta = garch_params[i]
            volatilities[t, i] = np.sqrt(
                omega + alpha * returns[t-1, i]**2 + beta * volatilities[t-1, i]**2
            )
        
        # Update Q_t
        z_lagged = z[t-1].reshape(-1, 1)
        Q_t = (1 - a - b) * Q_bar + a * (z_lagged @ z_lagged.T) + b * Q_t
        
        # Compute correlation matrix
        Q_diag = np.diag(1.0 / np.sqrt(np.diag(Q_t)))
        R_t = Q_diag @ Q_t @ Q_diag
        correlations[t] = R_t
        
        # Generate correlated standardized innovations
        chol = np.linalg.cholesky(R_t)
        correlated_z = z[t].reshape(-1, 1).T @ chol
        
        # Generate returns
        returns[t] = correlated_z.flatten() * volatilities[t]
    
    # Discard burn-in period
    returns = returns[DEFAULT_BURN:]
    volatilities = volatilities[DEFAULT_BURN:]
    correlations = correlations[DEFAULT_BURN:]
    
    return returns, volatilities, correlations


@pytest.fixture
def mock_garch_model():
    """
    Create a mock GARCH model for testing interfaces without full computation.
    
    Returns:
        Mock: A mock GARCH model object
    """
    class MockGARCHModel:
        def __init__(self):
            self.fitted = False
            self.params = {'omega': 0.05, 'alpha': 0.1, 'beta': 0.85}
            self.std_errors = {'omega': 0.01, 'alpha': 0.02, 'beta': 0.03}
            self.log_likelihood = -1500.0
            self.aic = 3006.0
            self.bic = 3020.0
            self.residuals = None
            self.conditional_volatility = None
            self.data = None
        
        def fit(self, data, starting_values=None):
            self.data = data
            self.fitted = True
            n = len(data)
            self.residuals = np.random.standard_normal(n)
            self.conditional_volatility = np.ones(n) * 0.2
            return self
        
        async def fit_async(self, data, starting_values=None, progress_callback=None):
            if progress_callback:
                await progress_callback(0.5, "Halfway through estimation")
            return self.fit(data, starting_values)
        
        def forecast(self, horizon=10, num_simulations=1000):
            if not self.fitted:
                raise RuntimeError("Model must be fitted before forecasting")
            point_forecasts = np.ones(horizon) * 0.2
            forecast_paths = np.ones((num_simulations, horizon)) * 0.2
            return point_forecasts, forecast_paths
        
        def simulate(self, num_simulations=1, horizon=1000):
            return np.random.standard_normal((num_simulations, horizon))
    
    return MockGARCHModel()


@pytest.fixture
def mock_arma_model():
    """
    Create a mock ARMA model for testing interfaces without full computation.
    
    Returns:
        Mock: A mock ARMA model object
    """
    class MockARMAModel:
        def __init__(self):
            self.fitted = False
            self.params = {'const': 0.0, 'ar.1': 0.7, 'ma.1': 0.3}
            self.std_errors = {'const': 0.01, 'ar.1': 0.05, 'ma.1': 0.04}
            self.log_likelihood = -1400.0
            self.aic = 2806.0
            self.bic = 2820.0
            self.residuals = None
            self.fitted_values = None
            self.data = None
            self.ar_order = 1
            self.ma_order = 1
            self.include_constant = True
        
        def fit(self, data, ar_order=1, ma_order=1, include_constant=True):
            self.data = data
            self.fitted = True
            self.ar_order = ar_order
            self.ma_order = ma_order
            self.include_constant = include_constant
            n = len(data)
            self.residuals = np.random.standard_normal(n)
            self.fitted_values = data - self.residuals
            return self
        
        async def fit_async(self, data, ar_order=1, ma_order=1, include_constant=True, progress_callback=None):
            if progress_callback:
                await progress_callback(0.5, "Halfway through estimation")
            return self.fit(data, ar_order, ma_order, include_constant)
        
        def forecast(self, horizon=10, exog=None):
            if not self.fitted:
                raise RuntimeError("Model must be fitted before forecasting")
            point_forecasts = np.ones(horizon) * 0.1
            forecast_std_errors = np.ones(horizon) * 0.2
            return point_forecasts, forecast_std_errors
        
        def simulate(self, num_simulations=1, horizon=1000):
            return np.random.standard_normal((num_simulations, horizon))
    
    return MockARMAModel()


# Hypothesis strategies for generating test data
@pytest.fixture
def array_strategy():
    """
    Provide a strategy for generating NumPy arrays with various shapes and dtypes.
    
    Returns:
        callable: A function that returns a hypothesis strategy for arrays
    """
    def _array_strategy(min_dims=1, max_dims=2, min_side=2, max_side=100, dtype=np.float64):
        return arrays(
            dtype=dtype,
            shape=st.tuples(*(
                st.integers(min_side, max_side) 
                for _ in range(st.integers(min_dims, max_dims).example())
            )),
            elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)
        )
    
    return _array_strategy


@pytest.fixture
def time_series_strategy():
    """
    Provide a strategy for generating time series data with various properties.
    
    Returns:
        callable: A function that returns a hypothesis strategy for time series
    """
    def _time_series_strategy(min_length=10, max_length=1000, with_trend=False, with_seasonality=False):
        base_series = arrays(
            dtype=np.float64,
            shape=st.integers(min_length, max_length),
            elements=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)
        )
        
        if not with_trend and not with_seasonality:
            return base_series
        
        @st.composite
        def series_with_components(draw):
            series = draw(base_series)
            n = len(series)
            
            if with_trend:
                trend_coef = draw(st.floats(min_value=-0.1, max_value=0.1))
                trend = np.arange(n) * trend_coef
                series = series + trend
            
            if with_seasonality:
                period = draw(st.integers(min_value=2, max_value=min(50, n // 2)))
                amplitude = draw(st.floats(min_value=0.1, max_value=2.0))
                seasonal = amplitude * np.sin(2 * np.pi * np.arange(n) / period)
                series = series + seasonal
            
            return series
        
        return series_with_components()
    
    return _time_series_strategy


@pytest.fixture
def parametrize_models():
    """
    Provide a utility for parametrizing tests across multiple model types.
    
    Returns:
        callable: A decorator function for parametrizing tests
    """
    def _parametrize_models(model_types):
        """
        Parametrize a test function to run with multiple model types.
        
        Args:
            model_types: List of model types to test
        
        Returns:
            decorator: pytest parametrize decorator
        """
        return pytest.mark.parametrize("model_type", model_types)
    
    return _parametrize_models


@pytest.fixture
def parametrize_distributions():
    """
    Provide a utility for parametrizing tests across multiple distribution types.
    
    Returns:
        callable: A decorator function for parametrizing tests
    """
    def _parametrize_distributions(distribution_types):
        """
        Parametrize a test function to run with multiple distribution types.
        
        Args:
            distribution_types: List of distribution types to test
        
        Returns:
            decorator: pytest parametrize decorator
        """
        return pytest.mark.parametrize("distribution_type", distribution_types)
    
    return _parametrize_distributions


@pytest.fixture
def assert_array_equal():
    """
    Provide a utility for asserting array equality with tolerance.
    
    Returns:
        callable: A function for asserting array equality
    """
    def _assert_array_equal(actual, expected, rtol=1e-7, atol=1e-10, err_msg=""):
        """
        Assert that two arrays are equal within a tolerance.
        
        Args:
            actual: Actual array
            expected: Expected array
            rtol: Relative tolerance
            atol: Absolute tolerance
            err_msg: Error message
        
        Raises:
            AssertionError: If arrays are not equal within tolerance
        """
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=err_msg)
    
    return _assert_array_equal


@pytest.fixture
def assert_series_equal():
    """
    Provide a utility for asserting pandas Series equality.
    
    Returns:
        callable: A function for asserting Series equality
    """
    def _assert_series_equal(actual, expected, rtol=1e-7, atol=1e-10, check_index=True, err_msg=""):
        """
        Assert that two pandas Series are equal within a tolerance.
        
        Args:
            actual: Actual Series
            expected: Expected Series
            rtol: Relative tolerance
            atol: Absolute tolerance
            check_index: Whether to check index equality
            err_msg: Error message
        
        Raises:
            AssertionError: If Series are not equal within tolerance
        """
        pd.testing.assert_series_equal(
            actual, expected, 
            rtol=rtol, atol=atol, 
            check_index=check_index,
            check_names=check_index,
            obj=err_msg
        )
    
    return _assert_series_equal


@pytest.fixture
def assert_frame_equal():
    """
    Provide a utility for asserting pandas DataFrame equality.
    
    Returns:
        callable: A function for asserting DataFrame equality
    """
    def _assert_frame_equal(actual, expected, rtol=1e-7, atol=1e-10, check_index=True, check_column_order=True, err_msg=""):
        """
        Assert that two pandas DataFrames are equal within a tolerance.
        
        Args:
            actual: Actual DataFrame
            expected: Expected DataFrame
            rtol: Relative tolerance
            atol: Absolute tolerance
            check_index: Whether to check index equality
            check_column_order: Whether to check column order
            err_msg: Error message
        
        Raises:
            AssertionError: If DataFrames are not equal within tolerance
        """
        pd.testing.assert_frame_equal(
            actual, expected, 
            rtol=rtol, atol=atol, 
            check_index=check_index,
            check_column_order=check_column_order,
            obj=err_msg
        )
    
    return _assert_frame_equal
