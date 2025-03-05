'''
Pytest configuration and fixtures for the MFE Toolbox test suite.

This module provides common fixtures and utilities used across the test suite,
including data generators, mock objects, and parameterized test utilities.
It ensures consistent test environment setup and streamlines test creation.
'''

import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from hypothesis import strategies as st

from mfe.core.exceptions import (
    MFEError, ParameterError, DimensionError, ConvergenceError, NumericError
)
from mfe.core.base import (
    ModelBase, VolatilityModelBase, MultivariateVolatilityModelBase,
    TimeSeriesModelBase, BootstrapModelBase, RealizedVolatilityModelBase,
    CrossSectionalModelBase
)
from mfe.core.parameters import (
    ParameterBase, GARCHParameters, EGARCHParameters, TARCHParameters,
    APARCHParameters, DCCParameters, ARMAParameters, StudentTParameters,
    SkewedTParameters, GEDParameters
)


# ---- Basic Data Generation Fixtures ----

@pytest.fixture

def rng() -> np.random.Generator:
    """Provide a seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture

def sample_size() -> int:
    """Default sample size for test data."""
    return 1000


@pytest.fixture

def univariate_normal_data(rng: np.random.Generator, sample_size: int) -> np.ndarray:
    """Generate univariate normal random data for testing."""
    return rng.standard_normal(sample_size)


@pytest.fixture

def univariate_t_data(rng: np.random.Generator, sample_size: int) -> np.ndarray:
    """Generate univariate Student's t random data for testing."""
    df = 5.0  # Degrees of freedom
    return rng.standard_t(df, size=sample_size)


@pytest.fixture

def univariate_skewed_t_data(rng: np.random.Generator, sample_size: int) -> np.ndarray:
    """Generate univariate skewed Student's t random data for testing."""
    df = 5.0  # Degrees of freedom
    skew = 0.5  # Skewness parameter
    
    # Generate uniform random variables
    u = rng.uniform(0, 1, size=sample_size)
    
    # Transform to skewed t using inverse CDF method
    # This is a simplified approximation
    z = np.zeros(sample_size)
    mask = u < 0.5
    
    # For u < 0.5, use left tail
    z[mask] = -stats.t.ppf(2 * u[mask], df)
    
    # For u >= 0.5, use right tail with skewness
    z[~mask] = stats.t.ppf(2 * (u[~mask] - 0.5) / (1 + skew) + 0.5, df)
    
    return z


@pytest.fixture

def multivariate_normal_data(rng: np.random.Generator, sample_size: int) -> np.ndarray:
    """Generate multivariate normal random data for testing."""
    n_assets = 3
    mean = np.zeros(n_assets)
    cov = np.array([
        [1.0, 0.5, 0.3],
        [0.5, 1.0, 0.2],
        [0.3, 0.2, 1.0]
    ])
    return rng.multivariate_normal(mean, cov, size=sample_size)


@pytest.fixture

def time_series_with_trend(rng: np.random.Generator, sample_size: int) -> np.ndarray:
    """Generate time series data with a linear trend for testing."""
    trend = np.linspace(0, 2, sample_size)
    noise = rng.standard_normal(sample_size) * 0.5
    return trend + noise


@pytest.fixture

def time_series_with_seasonality(rng: np.random.Generator, sample_size: int) -> np.ndarray:
    """Generate time series data with seasonality for testing."""
    t = np.arange(sample_size)
    seasonality = np.sin(2 * np.pi * t / 50)  # Period of 50
    noise = rng.standard_normal(sample_size) * 0.3
    return seasonality + noise


@pytest.fixture

def ar1_process(rng: np.random.Generator, sample_size: int) -> np.ndarray:
    """Generate AR(1) process for testing."""
    phi = 0.7  # AR parameter
    sigma = 1.0  # Innovation standard deviation
    
    # Initialize the series
    y = np.zeros(sample_size)
    y[0] = rng.standard_normal()
    
    # Generate the AR(1) process
    for t in range(1, sample_size):
        y[t] = phi * y[t-1] + sigma * rng.standard_normal()
    
    return y


@pytest.fixture

def ma1_process(rng: np.random.Generator, sample_size: int) -> np.ndarray:
    """Generate MA(1) process for testing."""
    theta = 0.5  # MA parameter
    sigma = 1.0  # Innovation standard deviation
    
    # Generate innovations
    e = sigma * rng.standard_normal(sample_size + 1)
    
    # Generate the MA(1) process
    y = np.zeros(sample_size)
    for t in range(sample_size):
        y[t] = e[t+1] + theta * e[t]
    
    return y


@pytest.fixture

def arma11_process(rng: np.random.Generator, sample_size: int) -> np.ndarray:
    """Generate ARMA(1,1) process for testing."""
    phi = 0.7  # AR parameter
    theta = 0.5  # MA parameter
    sigma = 1.0  # Innovation standard deviation
    
    # Generate innovations
    e = sigma * rng.standard_normal(sample_size + 1)
    
    # Initialize the series
    y = np.zeros(sample_size)
    y[0] = e[1] + theta * e[0]
    
    # Generate the ARMA(1,1) process
    for t in range(1, sample_size):
        y[t] = phi * y[t-1] + e[t+1] + theta * e[t]
    
    return y


@pytest.fixture

def garch11_process(rng: np.random.Generator, sample_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate GARCH(1,1) process for testing.
    
    Returns:
        Tuple containing (returns, conditional_variances)
    """
    omega = 0.05  # Constant
    alpha = 0.1   # ARCH parameter
    beta = 0.8    # GARCH parameter
    
    # Initialize arrays
    returns = np.zeros(sample_size)
    variances = np.zeros(sample_size)
    
    # Set initial variance
    variances[0] = omega / (1 - alpha - beta)
    returns[0] = np.sqrt(variances[0]) * rng.standard_normal()
    
    # Generate the GARCH(1,1) process
    for t in range(1, sample_size):
        variances[t] = omega + alpha * returns[t-1]**2 + beta * variances[t-1]
        returns[t] = np.sqrt(variances[t]) * rng.standard_normal()
    
    return returns, variances


@pytest.fixture

def egarch11_process(rng: np.random.Generator, sample_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate EGARCH(1,1) process for testing.
    
    Returns:
        Tuple containing (returns, conditional_variances)
    """
    omega = -0.1   # Constant
    alpha = 0.2    # ARCH parameter
    gamma = -0.1   # Asymmetry parameter
    beta = 0.95    # GARCH parameter
    
    # Initialize arrays
    returns = np.zeros(sample_size)
    log_variances = np.zeros(sample_size)
    variances = np.zeros(sample_size)
    
    # Set initial log variance
    log_variances[0] = omega / (1 - beta)
    variances[0] = np.exp(log_variances[0])
    returns[0] = np.sqrt(variances[0]) * rng.standard_normal()
    
    # Generate the EGARCH(1,1) process
    for t in range(1, sample_size):
        z_t_1 = returns[t-1] / np.sqrt(variances[t-1])
        log_variances[t] = omega + beta * log_variances[t-1] + alpha * (np.abs(z_t_1) - np.sqrt(2/np.pi)) + gamma * z_t_1
        variances[t] = np.exp(log_variances[t])
        returns[t] = np.sqrt(variances[t]) * rng.standard_normal()
    
    return returns, variances


@pytest.fixture

def tarch11_process(rng: np.random.Generator, sample_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate TARCH(1,1) process for testing.
    
    Returns:
        Tuple containing (returns, conditional_variances)
    """
    omega = 0.05  # Constant
    alpha = 0.05  # ARCH parameter
    gamma = 0.1   # Asymmetry parameter
    beta = 0.8    # GARCH parameter
    
    # Initialize arrays
    returns = np.zeros(sample_size)
    variances = np.zeros(sample_size)
    
    # Set initial variance
    variances[0] = omega / (1 - alpha - beta - 0.5 * gamma)
    returns[0] = np.sqrt(variances[0]) * rng.standard_normal()
    
    # Generate the TARCH(1,1) process
    for t in range(1, sample_size):
        neg_indicator = returns[t-1] < 0
        variances[t] = omega + alpha * returns[t-1]**2 + gamma * returns[t-1]**2 * neg_indicator + beta * variances[t-1]
        returns[t] = np.sqrt(variances[t]) * rng.standard_normal()
    
    return returns, variances


@pytest.fixture

def dcc_process(rng: np.random.Generator, sample_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate DCC process for testing.
    
    Returns:
        Tuple containing (returns, conditional_correlations)
    """
    n_assets = 3
    
    # GARCH parameters for each asset
    omega = np.array([0.05, 0.03, 0.04])
    alpha = np.array([0.1, 0.08, 0.12])
    beta = np.array([0.8, 0.85, 0.78])
    
    # DCC parameters
    a = 0.05
    b = 0.9
    
    # Initialize arrays
    returns = np.zeros((sample_size, n_assets))
    asset_variances = np.zeros((sample_size, n_assets))
    correlations = np.zeros((sample_size, n_assets, n_assets))
    
    # Unconditional correlation matrix
    R_bar = np.array([
        [1.0, 0.5, 0.3],
        [0.5, 1.0, 0.2],
        [0.3, 0.2, 1.0]
    ])
    
    # Set initial values
    for i in range(n_assets):
        asset_variances[0, i] = omega[i] / (1 - alpha[i] - beta[i])
    
    correlations[0] = R_bar
    
    # Generate standardized innovations
    z = rng.standard_normal((sample_size, n_assets))
    
    # Generate the DCC process
    for t in range(1, sample_size):
        # Update asset variances (univariate GARCH)
        for i in range(n_assets):
            asset_variances[t, i] = omega[i] + alpha[i] * returns[t-1, i]**2 + beta[i] * asset_variances[t-1, i]
        
        # Compute standardized residuals
        std_resid = returns[t-1] / np.sqrt(asset_variances[t-1])
        
        # Update correlation matrix (DCC)
        Q_t = (1 - a - b) * R_bar + a * np.outer(std_resid, std_resid) + b * ((1 - a - b) * R_bar + a * np.outer(std_resid, std_resid))
        
        # Normalize to get correlation matrix
        Q_diag_inv_sqrt = np.diag(1 / np.sqrt(np.diag(Q_t)))
        R_t = Q_diag_inv_sqrt @ Q_t @ Q_diag_inv_sqrt
        
        # Store correlation matrix
        correlations[t] = R_t
        
        # Generate returns
        H_t = np.diag(np.sqrt(asset_variances[t])) @ R_t @ np.diag(np.sqrt(asset_variances[t]))
        returns[t] = rng.multivariate_normal(np.zeros(n_assets), H_t)
    
    return returns, correlations


@pytest.fixture

def high_frequency_price_data(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Generate high-frequency price and time data for testing.
    
    Returns:
        Tuple containing (prices, times)
    """
    # Generate approximately 1000 intraday observations (about 1 trading day)
    n_obs = 1000
    
    # Generate times (in seconds from market open)
    times = np.sort(rng.uniform(0, 23400, n_obs))  # 6.5 hours = 23400 seconds
    
    # Generate log price process with realistic properties
    # Start with a random walk
    log_returns = 0.0001 * rng.standard_normal(n_obs)
    
    # Add some microstructure noise
    noise = 0.00005 * rng.standard_normal(n_obs)
    log_returns = log_returns + noise
    
    # Cumulate to get log prices
    log_prices = np.cumsum(log_returns)
    
    # Convert to prices
    initial_price = 100.0
    prices = initial_price * np.exp(log_prices)
    
    return prices, times


@pytest.fixture

def cross_sectional_data(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Generate cross-sectional data for testing.
    
    Returns:
        Tuple containing (y, X)
    """
    n_obs = 100
    n_vars = 3
    
    # Generate predictors
    X = rng.standard_normal((n_obs, n_vars))
    
    # True coefficients
    beta = np.array([1.0, -0.5, 0.2])
    
    # Generate dependent variable with noise
    y = X @ beta + 0.5 * rng.standard_normal(n_obs)
    
    return y, X


# ---- Model Parameter Fixtures ----

@pytest.fixture

def garch_params() -> GARCHParameters:
    """Create GARCH(1,1) parameters for testing."""
    return GARCHParameters(omega=0.05, alpha=0.1, beta=0.8)


@pytest.fixture

def egarch_params() -> EGARCHParameters:
    """Create EGARCH(1,1) parameters for testing."""
    return EGARCHParameters(omega=-0.1, alpha=0.2, gamma=-0.1, beta=0.95)


@pytest.fixture

def tarch_params() -> TARCHParameters:
    """Create TARCH(1,1) parameters for testing."""
    return TARCHParameters(omega=0.05, alpha=0.05, gamma=0.1, beta=0.8)


@pytest.fixture

def aparch_params() -> APARCHParameters:
    """Create APARCH(1,1) parameters for testing."""
    return APARCHParameters(omega=0.05, alpha=0.1, gamma=0.1, beta=0.8, delta=1.5)


@pytest.fixture

def dcc_params() -> DCCParameters:
    """Create DCC parameters for testing."""
    return DCCParameters(a=0.05, b=0.9)


@pytest.fixture

def arma_params() -> ARMAParameters:
    """Create ARMA(1,1) parameters for testing."""
    return ARMAParameters(
        ar_params=np.array([0.7]),
        ma_params=np.array([0.5]),
        sigma2=1.0,
        constant=0.0
    )


@pytest.fixture

def student_t_params() -> StudentTParameters:
    """Create Student's t distribution parameters for testing."""
    return StudentTParameters(df=5.0)


@pytest.fixture

def skewed_t_params() -> SkewedTParameters:
    """Create skewed Student's t distribution parameters for testing."""
    return SkewedTParameters(df=5.0, lambda_=0.5)


@pytest.fixture

def ged_params() -> GEDParameters:
    """Create GED distribution parameters for testing."""
    return GEDParameters(nu=1.5)


# ---- Mock Objects ----

class MockVolatilityModel(VolatilityModelBase):
    """Mock volatility model for testing."""
    
    def __init__(self, name="MockVolatilityModel"):
        super().__init__(name=name)
        self._params = None
    
    def fit(self, data, **kwargs):
        """Mock implementation of fit method."""
        self.validate_data(data)
        self._fitted = True
        self._params = kwargs.get("params", GARCHParameters(omega=0.05, alpha=0.1, beta=0.8))
        self._conditional_variances = np.ones_like(data)
        
        # Create a simple result object
        from dataclasses import dataclass
        
        @dataclass
        class MockResult:
            model_name: str
            convergence: bool = True
            iterations: int = 10
            log_likelihood: float = -1000.0
            aic: float = 2010.0
            bic: float = 2030.0
            
        self._results = MockResult(model_name=self.name)
        return self._results
    
    def simulate(self, n_periods, burn=0, initial_values=None, random_state=None, **kwargs):
        """Mock implementation of simulate method."""
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        rng = np.random.default_rng(random_state)
        return rng.standard_normal(n_periods)
    
    def compute_variance(self, parameters, data, sigma2=None, backcast=None):
        """Mock implementation of compute_variance method."""
        if sigma2 is None:
            sigma2 = np.ones_like(data)
        return sigma2


class MockMultivariateVolatilityModel(MultivariateVolatilityModelBase):
    """Mock multivariate volatility model for testing."""
    
    def __init__(self, name="MockMultivariateVolatilityModel"):
        super().__init__(name=name)
        self._params = None
    
    def fit(self, data, **kwargs):
        """Mock implementation of fit method."""
        self.validate_data(data)
        self._fitted = True
        self._n_assets = data.shape[1]
        self._params = kwargs.get("params", DCCParameters(a=0.05, b=0.9))
        
        # Create mock conditional covariances
        T = data.shape[0]
        self._conditional_covariances = np.zeros((self._n_assets, self._n_assets, T))
        for t in range(T):
            self._conditional_covariances[:, :, t] = np.eye(self._n_assets)
        
        # Create a simple result object
        from dataclasses import dataclass
        
        @dataclass
        class MockResult:
            model_name: str
            convergence: bool = True
            iterations: int = 10
            log_likelihood: float = -1000.0
            aic: float = 2010.0
            bic: float = 2030.0
            
        self._results = MockResult(model_name=self.name)
        return self._results
    
    def simulate(self, n_periods, burn=0, initial_values=None, random_state=None, **kwargs):
        """Mock implementation of simulate method."""
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        rng = np.random.default_rng(random_state)
        return rng.standard_normal((n_periods, self._n_assets))
    
    def compute_covariance(self, parameters, data, sigma=None, backcast=None):
        """Mock implementation of compute_covariance method."""
        T = data.shape[0]
        n_assets = data.shape[1]
        
        if sigma is None:
            sigma = np.zeros((n_assets, n_assets, T))
            for t in range(T):
                sigma[:, :, t] = np.eye(n_assets)
        
        return sigma


class MockTimeSeriesModel(TimeSeriesModelBase):
    """Mock time series model for testing."""
    
    def __init__(self, name="MockTimeSeriesModel"):
        super().__init__(name=name)
        self._params = None
    
    def fit(self, data, **kwargs):
        """Mock implementation of fit method."""
        data_array = self.validate_data(data)
        self._fitted = True
        self._params = kwargs.get("params", ARMAParameters(
            ar_params=np.array([0.7]),
            ma_params=np.array([0.5]),
            sigma2=1.0,
            constant=0.0
        ))
        
        # Create mock fitted values and residuals
        self._fitted_values = np.zeros_like(data_array)
        self._residuals = data_array - self._fitted_values
        
        # Create a simple result object
        from dataclasses import dataclass
        
        @dataclass
        class MockResult:
            model_name: str
            convergence: bool = True
            iterations: int = 10
            log_likelihood: float = -1000.0
            aic: float = 2010.0
            bic: float = 2030.0
            
        self._results = MockResult(model_name=self.name)
        return self._results
    
    def simulate(self, n_periods, burn=0, initial_values=None, random_state=None, **kwargs):
        """Mock implementation of simulate method."""
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        rng = np.random.default_rng(random_state)
        return rng.standard_normal(n_periods)
    
    def forecast(self, steps, exog=None, confidence_level=0.95, **kwargs):
        """Mock implementation of forecast method."""
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        # Generate mock forecasts
        forecasts = np.zeros(steps)
        lower_bounds = forecasts - 1.96
        upper_bounds = forecasts + 1.96
        
        return forecasts, lower_bounds, upper_bounds


@pytest.fixture

def mock_volatility_model() -> MockVolatilityModel:
    """Create a mock volatility model for testing."""
    return MockVolatilityModel()


@pytest.fixture

def mock_multivariate_volatility_model() -> MockMultivariateVolatilityModel:
    """Create a mock multivariate volatility model for testing."""
    return MockMultivariateVolatilityModel()


@pytest.fixture

def mock_time_series_model() -> MockTimeSeriesModel:
    """Create a mock time series model for testing."""
    return MockTimeSeriesModel()


# ---- Temporary File and Directory Fixtures ----

@pytest.fixture

def temp_dir() -> str:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture

def temp_file() -> str:
    """Create a temporary file for testing."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_name = tmp_file.name
    
    yield tmp_name
    
    # Clean up the file after the test
    try:
        os.unlink(tmp_name)
    except:
        pass


# ---- Hypothesis Strategies for Property-Based Testing ----

@pytest.fixture

def garch_param_strategy() -> st.SearchStrategy:
    """Strategy for generating valid GARCH parameters."""
    return st.fixed_dictionaries({
        'omega': st.floats(min_value=0.001, max_value=0.1),
        'alpha': st.floats(min_value=0.01, max_value=0.3),
        'beta': st.floats(min_value=0.5, max_value=0.85)
    }).filter(lambda params: params['alpha'] + params['beta'] < 0.99)


@pytest.fixture

def egarch_param_strategy() -> st.SearchStrategy:
    """Strategy for generating valid EGARCH parameters."""
    return st.fixed_dictionaries({
        'omega': st.floats(min_value=-0.5, max_value=0.5),
        'alpha': st.floats(min_value=0.01, max_value=0.5),
        'gamma': st.floats(min_value=-0.3, max_value=0.3),
        'beta': st.floats(min_value=0.5, max_value=0.98)
    }).filter(lambda params: abs(params['beta']) < 0.99)


@pytest.fixture

def tarch_param_strategy() -> st.SearchStrategy:
    """Strategy for generating valid TARCH parameters."""
    return st.fixed_dictionaries({
        'omega': st.floats(min_value=0.001, max_value=0.1),
        'alpha': st.floats(min_value=0.01, max_value=0.2),
        'gamma': st.floats(min_value=0.01, max_value=0.2),
        'beta': st.floats(min_value=0.5, max_value=0.85)
    }).filter(lambda params: params['alpha'] + params['beta'] + 0.5 * params['gamma'] < 0.99)


@pytest.fixture

def arma_param_strategy() -> st.SearchStrategy:
    """Strategy for generating valid ARMA parameters."""
    return st.fixed_dictionaries({
        'ar_params': st.lists(st.floats(min_value=-0.9, max_value=0.9), min_size=1, max_size=3),
        'ma_params': st.lists(st.floats(min_value=-0.9, max_value=0.9), min_size=1, max_size=3),
        'sigma2': st.floats(min_value=0.1, max_value=2.0),
        'constant': st.floats(min_value=-1.0, max_value=1.0)
    })


@pytest.fixture

def distribution_param_strategy() -> st.SearchStrategy:
    """Strategy for generating valid distribution parameters."""
    return st.one_of(
        st.fixed_dictionaries({
            'distribution': st.just('normal')
        }),
        st.fixed_dictionaries({
            'distribution': st.just('t'),
            'df': st.floats(min_value=2.1, max_value=30.0)
        }),
        st.fixed_dictionaries({
            'distribution': st.just('skewed_t'),
            'df': st.floats(min_value=2.1, max_value=30.0),
            'lambda_': st.floats(min_value=-0.9, max_value=0.9)
        }),
        st.fixed_dictionaries({
            'distribution': st.just('ged'),
            'nu': st.floats(min_value=0.5, max_value=5.0)
        })
    )


# ---- Parameterization Utilities ----

def pytest_generate_tests(metafunc):
    """Custom test generation for parameterized tests."""
    # Example of custom parameterization
    if "distribution_type" in metafunc.fixturenames:
        metafunc.parametrize(
            "distribution_type",
            ["normal", "t", "skewed_t", "ged"]
        )
    
    if "volatility_model_type" in metafunc.fixturenames:
        metafunc.parametrize(
            "volatility_model_type",
            ["GARCH", "EGARCH", "TARCH", "APARCH"]
        )
    
    if "multivariate_model_type" in metafunc.fixturenames:
        metafunc.parametrize(
            "multivariate_model_type",
            ["DCC", "BEKK", "CCC", "RARCH"]
        )


# ---- Utility Functions for Tests ----

def assert_array_equal(actual, expected, rtol=1e-7, atol=1e-7, err_msg=""):
    """Assert that two arrays are equal within tolerance."""
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=err_msg)


def assert_series_equal(actual, expected, rtol=1e-7, atol=1e-7, err_msg=""):
    """Assert that two pandas Series are equal within tolerance."""
    pd.testing.assert_series_equal(
        actual, expected, 
        rtol=rtol, atol=atol, 
        check_dtype=False, 
        check_index_type=False,
        check_names=False,
        obj=err_msg
    )


def assert_frame_equal(actual, expected, rtol=1e-7, atol=1e-7, err_msg=""):
    """Assert that two pandas DataFrames are equal within tolerance."""
    pd.testing.assert_frame_equal(
        actual, expected, 
        rtol=rtol, atol=atol, 
        check_dtype=False, 
        check_index_type=False,
        check_column_type=False,
        check_names=False,
        obj=err_msg
    )


def assert_parameters_equal(actual, expected, rtol=1e-7, atol=1e-7, err_msg=""):
    """Assert that two parameter objects are equal within tolerance."""
    assert type(actual) == type(expected), f"Parameter types do not match: {type(actual)} != {type(expected)}"
    
    # Convert to dictionaries
    actual_dict = actual.to_dict()
    expected_dict = expected.to_dict()
    
    # Check that keys match
    assert set(actual_dict.keys()) == set(expected_dict.keys()), f"Parameter keys do not match: {set(actual_dict.keys())} != {set(expected_dict.keys())}"
    
    # Check each parameter
    for key in actual_dict:
        actual_val = actual_dict[key]
        expected_val = expected_dict[key]
        
        if isinstance(actual_val, np.ndarray) and isinstance(expected_val, np.ndarray):
            assert_array_equal(actual_val, expected_val, rtol=rtol, atol=atol, err_msg=f"{err_msg} Parameter '{key}' does not match")
        else:
            assert abs(actual_val - expected_val) <= atol + rtol * abs(expected_val), f"{err_msg} Parameter '{key}' does not match: {actual_val} != {expected_val}"


def assert_model_results_equal(actual, expected, rtol=1e-7, atol=1e-7, err_msg=""):
    """Assert that two model result objects are equal within tolerance."""
    # Check common attributes
    for attr in ['model_name', 'convergence', 'iterations']:
        assert hasattr(actual, attr), f"Actual result missing attribute: {attr}"
        assert hasattr(expected, attr), f"Expected result missing attribute: {attr}"
        assert getattr(actual, attr) == getattr(expected, attr), f"{err_msg} Attribute '{attr}' does not match: {getattr(actual, attr)} != {getattr(expected, attr)}"
    
    # Check numeric attributes with tolerance
    for attr in ['log_likelihood', 'aic', 'bic']:
        if hasattr(actual, attr) and hasattr(expected, attr):
            actual_val = getattr(actual, attr)
            expected_val = getattr(expected, attr)
            
            # Handle None values
            if actual_val is None and expected_val is None:
                continue
            
            assert actual_val is not None, f"{err_msg} Attribute '{attr}' is None in actual but not in expected"
            assert expected_val is not None, f"{err_msg} Attribute '{attr}' is None in expected but not in actual"
            
            assert abs(actual_val - expected_val) <= atol + rtol * abs(expected_val), f"{err_msg} Attribute '{attr}' does not match: {actual_val} != {expected_val}"
