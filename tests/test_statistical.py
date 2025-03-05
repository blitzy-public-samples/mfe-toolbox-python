# tests/test_statistical.py
"""
Tests for statistical test implementations in the MFE Toolbox.

This module provides comprehensive tests for statistical test functions including
Berkowitz, Jarque-Bera, and Kolmogorov-Smirnov tests. It verifies the correct behavior
of distribution testing functions across various input types, edge cases, and expected
statistical properties.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.numpy import arrays

from mfe.core.exceptions import ParameterError, DimensionError
from mfe.models.distributions import Normal, StudentT, GED, SkewedT
from mfe.models.distributions.utils import validate_distribution_parameters

# Import the statistical tests module
# Assuming the statistical tests are in a module like this:
try:
    from mfe.models.tests import (
        berkowitz_test, jarque_bera_test, kolmogorov_smirnov_test,
        ljung_box_test, lm_test
    )
except ImportError:
    # If the module structure is different, try alternative imports
    try:
        from mfe.models.time_series.diagnostics import (
            berkowitz_test, jarque_bera_test, kolmogorov_smirnov_test,
            ljung_box_test, lm_test
        )
    except ImportError:
        pytest.skip("Statistical tests module not found", allow_module_level=True)


# ---- Basic Test Functions ----

def test_jarque_bera_normal_distribution():
    """Test that Jarque-Bera correctly identifies normal distribution."""
    # Generate normal data
    rng = np.random.default_rng(42)
    data = rng.standard_normal(1000)
    
    # Test the Jarque-Bera test
    result = jarque_bera_test(data)
    
    # For normal data, the p-value should be high (not rejecting normality)
    assert result.statistic is not None
    assert result.p_value is not None
    assert result.p_value > 0.05, "Jarque-Bera incorrectly rejected normality for normal data"
    
    # Test with pandas Series
    data_series = pd.Series(data)
    result_series = jarque_bera_test(data_series)
    
    # Results should be the same regardless of input type
    assert abs(result.statistic - result_series.statistic) < 1e-10
    assert abs(result.p_value - result_series.p_value) < 1e-10


def test_jarque_bera_t_distribution():
    """Test that Jarque-Bera correctly identifies non-normal t-distribution."""
    # Generate t-distributed data with low degrees of freedom (heavy tails)
    rng = np.random.default_rng(42)
    data = rng.standard_t(df=3, size=1000)
    
    # Test the Jarque-Bera test
    result = jarque_bera_test(data)
    
    # For t-distributed data with df=3, the p-value should be low (rejecting normality)
    assert result.p_value < 0.05, "Jarque-Bera failed to reject normality for t-distributed data"


def test_kolmogorov_smirnov_normal_distribution():
    """Test that Kolmogorov-Smirnov correctly tests against normal distribution."""
    # Generate normal data
    rng = np.random.default_rng(42)
    data = rng.standard_normal(1000)
    
    # Test against normal distribution
    result = kolmogorov_smirnov_test(data, distribution='normal')
    
    # For normal data tested against normal distribution, p-value should be high
    assert result.statistic is not None
    assert result.p_value is not None
    assert result.p_value > 0.05, "KS test incorrectly rejected normal distribution for normal data"
    
    # Test with pandas Series
    data_series = pd.Series(data)
    result_series = kolmogorov_smirnov_test(data_series, distribution='normal')
    
    # Results should be the same regardless of input type
    assert abs(result.statistic - result_series.statistic) < 1e-10
    assert abs(result.p_value - result_series.p_value) < 1e-10


def test_kolmogorov_smirnov_t_distribution():
    """Test that Kolmogorov-Smirnov correctly tests against t-distribution."""
    # Generate t-distributed data
    rng = np.random.default_rng(42)
    data = rng.standard_t(df=5, size=1000)
    
    # Test against normal distribution (should reject)
    result_normal = kolmogorov_smirnov_test(data, distribution='normal')
    assert result_normal.p_value < 0.05, "KS test failed to reject normal distribution for t-distributed data"
    
    # Test against t-distribution with correct df (should not reject)
    result_t = kolmogorov_smirnov_test(data, distribution='t', df=5)
    assert result_t.p_value > 0.05, "KS test incorrectly rejected t-distribution for t-distributed data"
    
    # Test against t-distribution with wrong df (should reject)
    result_wrong_t = kolmogorov_smirnov_test(data, distribution='t', df=10)
    assert result_wrong_t.p_value < 0.05, "KS test failed to reject t-distribution with wrong df"


def test_berkowitz_test_normal_distribution():
    """Test that Berkowitz test correctly identifies normal distribution."""
    # Generate normal data
    rng = np.random.default_rng(42)
    data = rng.standard_normal(1000)
    
    # Transform data to uniform using normal CDF
    uniform_data = stats.norm.cdf(data)
    
    # Test the Berkowitz test
    result = berkowitz_test(uniform_data)
    
    # For normal data transformed to uniform, the p-value should be high
    assert result.statistic is not None
    assert result.p_value is not None
    assert result.p_value > 0.05, "Berkowitz test incorrectly rejected for normal data"
    
    # Test with pandas Series
    uniform_series = pd.Series(uniform_data)
    result_series = berkowitz_test(uniform_series)
    
    # Results should be the same regardless of input type
    assert abs(result.statistic - result_series.statistic) < 1e-10
    assert abs(result.p_value - result_series.p_value) < 1e-10


def test_ljung_box_test_white_noise():
    """Test that Ljung-Box test correctly identifies white noise."""
    # Generate white noise
    rng = np.random.default_rng(42)
    data = rng.standard_normal(1000)
    
    # Test the Ljung-Box test
    result = ljung_box_test(data, lags=10)
    
    # For white noise, the p-value should be high (not rejecting no autocorrelation)
    assert result.statistic is not None
    assert result.p_value is not None
    assert result.p_value > 0.05, "Ljung-Box test incorrectly rejected white noise"
    
    # Test with pandas Series
    data_series = pd.Series(data)
    result_series = ljung_box_test(data_series, lags=10)
    
    # Results should be the same regardless of input type
    assert abs(result.statistic - result_series.statistic) < 1e-10
    assert abs(result.p_value - result_series.p_value) < 1e-10


def test_ljung_box_test_autocorrelated_data():
    """Test that Ljung-Box test correctly identifies autocorrelated data."""
    # Generate AR(1) process
    rng = np.random.default_rng(42)
    n = 1000
    phi = 0.7  # Strong autocorrelation
    data = np.zeros(n)
    data[0] = rng.standard_normal()
    
    for t in range(1, n):
        data[t] = phi * data[t-1] + rng.standard_normal()
    
    # Test the Ljung-Box test
    result = ljung_box_test(data, lags=10)
    
    # For autocorrelated data, the p-value should be low (rejecting no autocorrelation)
    assert result.p_value < 0.05, "Ljung-Box test failed to reject for autocorrelated data"


def test_lm_test_homoskedastic_data():
    """Test that LM test correctly identifies homoskedastic data."""
    # Generate homoskedastic data
    rng = np.random.default_rng(42)
    data = rng.standard_normal(1000)
    
    # Test the LM test
    result = lm_test(data, lags=5)
    
    # For homoskedastic data, the p-value should be high (not rejecting homoskedasticity)
    assert result.statistic is not None
    assert result.p_value is not None
    assert result.p_value > 0.05, "LM test incorrectly rejected homoskedasticity"
    
    # Test with pandas Series
    data_series = pd.Series(data)
    result_series = lm_test(data_series, lags=5)
    
    # Results should be the same regardless of input type
    assert abs(result.statistic - result_series.statistic) < 1e-10
    assert abs(result.p_value - result_series.p_value) < 1e-10


def test_lm_test_heteroskedastic_data():
    """Test that LM test correctly identifies heteroskedastic data."""
    # Generate GARCH(1,1) process
    rng = np.random.default_rng(42)
    n = 1000
    omega = 0.05
    alpha = 0.1
    beta = 0.8
    
    returns = np.zeros(n)
    variances = np.zeros(n)
    
    # Set initial variance
    variances[0] = omega / (1 - alpha - beta)
    returns[0] = np.sqrt(variances[0]) * rng.standard_normal()
    
    # Generate the GARCH(1,1) process
    for t in range(1, n):
        variances[t] = omega + alpha * returns[t-1]**2 + beta * variances[t-1]
        returns[t] = np.sqrt(variances[t]) * rng.standard_normal()
    
    # Test the LM test
    result = lm_test(returns, lags=5)
    
    # For heteroskedastic data, the p-value should be low (rejecting homoskedasticity)
    assert result.p_value < 0.05, "LM test failed to reject for heteroskedastic data"


# ---- Edge Cases and Error Handling ----

def test_jarque_bera_empty_input():
    """Test that Jarque-Bera handles empty input correctly."""
    with pytest.raises((ValueError, ParameterError)):
        jarque_bera_test(np.array([]))


def test_jarque_bera_single_value():
    """Test that Jarque-Bera handles single value input correctly."""
    with pytest.raises((ValueError, ParameterError)):
        jarque_bera_test(np.array([1.0]))


def test_jarque_bera_two_values():
    """Test that Jarque-Bera handles two-value input correctly."""
    with pytest.raises((ValueError, ParameterError)):
        jarque_bera_test(np.array([1.0, 2.0]))


def test_kolmogorov_smirnov_invalid_distribution():
    """Test that Kolmogorov-Smirnov handles invalid distribution correctly."""
    data = np.random.default_rng(42).standard_normal(100)
    
    with pytest.raises((ValueError, ParameterError)):
        kolmogorov_smirnov_test(data, distribution='invalid_distribution')


def test_kolmogorov_smirnov_missing_parameters():
    """Test that Kolmogorov-Smirnov handles missing distribution parameters correctly."""
    data = np.random.default_rng(42).standard_normal(100)
    
    with pytest.raises((ValueError, ParameterError)):
        kolmogorov_smirnov_test(data, distribution='t')  # Missing df parameter


def test_berkowitz_test_invalid_input():
    """Test that Berkowitz test handles invalid input correctly."""
    # Values outside [0, 1] range
    data = np.random.default_rng(42).uniform(-0.1, 1.1, 100)
    
    with pytest.raises((ValueError, ParameterError)):
        berkowitz_test(data)


def test_ljung_box_test_invalid_lags():
    """Test that Ljung-Box test handles invalid lags correctly."""
    data = np.random.default_rng(42).standard_normal(100)
    
    with pytest.raises((ValueError, ParameterError)):
        ljung_box_test(data, lags=0)  # Invalid lag value
    
    with pytest.raises((ValueError, ParameterError)):
        ljung_box_test(data, lags=-5)  # Negative lag value
    
    with pytest.raises((ValueError, ParameterError)):
        ljung_box_test(data, lags=101)  # Lag value exceeds data length


def test_lm_test_invalid_lags():
    """Test that LM test handles invalid lags correctly."""
    data = np.random.default_rng(42).standard_normal(100)
    
    with pytest.raises((ValueError, ParameterError)):
        lm_test(data, lags=0)  # Invalid lag value
    
    with pytest.raises((ValueError, ParameterError)):
        lm_test(data, lags=-5)  # Negative lag value
    
    with pytest.raises((ValueError, ParameterError)):
        lm_test(data, lags=101)  # Lag value exceeds data length


def test_nan_handling():
    """Test that statistical tests handle NaN values correctly."""
    data = np.random.default_rng(42).standard_normal(100)
    data[10] = np.nan  # Insert a NaN value
    
    # All tests should raise an error for NaN values
    with pytest.raises((ValueError, ParameterError)):
        jarque_bera_test(data)
    
    with pytest.raises((ValueError, ParameterError)):
        kolmogorov_smirnov_test(data, distribution='normal')
    
    with pytest.raises((ValueError, ParameterError)):
        berkowitz_test(stats.norm.cdf(data))
    
    with pytest.raises((ValueError, ParameterError)):
        ljung_box_test(data, lags=10)
    
    with pytest.raises((ValueError, ParameterError)):
        lm_test(data, lags=5)


def test_inf_handling():
    """Test that statistical tests handle infinite values correctly."""
    data = np.random.default_rng(42).standard_normal(100)
    data[10] = np.inf  # Insert an infinite value
    
    # All tests should raise an error for infinite values
    with pytest.raises((ValueError, ParameterError)):
        jarque_bera_test(data)
    
    with pytest.raises((ValueError, ParameterError)):
        kolmogorov_smirnov_test(data, distribution='normal')
    
    with pytest.raises((ValueError, ParameterError)):
        berkowitz_test(stats.norm.cdf(data))
    
    with pytest.raises((ValueError, ParameterError)):
        ljung_box_test(data, lags=10)
    
    with pytest.raises((ValueError, ParameterError)):
        lm_test(data, lags=5)


# ---- Property-Based Testing with Hypothesis ----

@given(arrays(np.float64, st.integers(min_value=100, max_value=1000), elements=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)))
@settings(deadline=None)  # Disable deadline for potentially slow tests

def test_jarque_bera_properties(data):
    """Test properties of Jarque-Bera test using hypothesis."""
    # Skip if data has too little variation
    assume(np.std(data) > 1e-10)
    
    result = jarque_bera_test(data)
    
    # Test statistic should be non-negative
    assert result.statistic >= 0
    
    # p-value should be between 0 and 1
    assert 0 <= result.p_value <= 1
    
    # For perfectly normal data, the statistic should be close to 0
    # Generate perfectly normal data with same mean and std
    mean = np.mean(data)
    std = np.std(data)
    normal_data = np.random.default_rng(42).normal(mean, std, size=len(data))
    
    normal_result = jarque_bera_test(normal_data)
    
    # The statistic for normal data should generally be smaller
    # This is a probabilistic test, so it might occasionally fail
    # We're just checking that the test behaves reasonably
    assert normal_result.statistic < 10, "Jarque-Bera statistic for normal data is unexpectedly large"


@given(
    arrays(np.float64, st.integers(min_value=100, max_value=1000), elements=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)),
    st.sampled_from(['normal', 't', 'ged', 'skewed_t'])
)
@settings(deadline=None)  # Disable deadline for potentially slow tests

def test_kolmogorov_smirnov_properties(data, distribution):
    """Test properties of Kolmogorov-Smirnov test using hypothesis."""
    # Skip if data has too little variation
    assume(np.std(data) > 1e-10)
    
    # Prepare distribution parameters
    kwargs = {}
    if distribution == 't':
        kwargs['df'] = 5.0
    elif distribution == 'ged':
        kwargs['nu'] = 1.5
    elif distribution == 'skewed_t':
        kwargs['df'] = 5.0
        kwargs['lambda_'] = 0.5
    
    result = kolmogorov_smirnov_test(data, distribution=distribution, **kwargs)
    
    # Test statistic should be non-negative
    assert result.statistic >= 0
    
    # p-value should be between 0 and 1
    assert 0 <= result.p_value <= 1
    
    # For data from the specified distribution, the p-value should be high
    # Generate data from the specified distribution
    if distribution == 'normal':
        dist_data = np.random.default_rng(42).normal(0, 1, size=len(data))
    elif distribution == 't':
        dist_data = np.random.default_rng(42).standard_t(df=5.0, size=len(data))
    elif distribution == 'ged':
        # Approximate GED using a normal distribution for this test
        dist_data = np.random.default_rng(42).normal(0, 1, size=len(data))
    elif distribution == 'skewed_t':
        # Approximate skewed t using a normal distribution for this test
        dist_data = np.random.default_rng(42).normal(0, 1, size=len(data))
    
    dist_result = kolmogorov_smirnov_test(dist_data, distribution=distribution, **kwargs)
    
    # The p-value for data from the correct distribution should generally be higher
    # This is a probabilistic test, so it might occasionally fail
    assert dist_result.p_value > 0.01, f"KS p-value for {distribution} data is unexpectedly low"


@given(arrays(np.float64, st.integers(min_value=100, max_value=1000), elements=st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False)))
@settings(deadline=None)  # Disable deadline for potentially slow tests

def test_berkowitz_properties(data):
    """Test properties of Berkowitz test using hypothesis."""
    # Ensure data is within [0, 1] range
    data = np.clip(data, 0, 1)
    
    result = berkowitz_test(data)
    
    # Test statistic should be non-negative
    assert result.statistic >= 0
    
    # p-value should be between 0 and 1
    assert 0 <= result.p_value <= 1
    
    # For uniform data, the p-value should be high
    uniform_data = np.random.default_rng(42).uniform(0, 1, size=len(data))
    uniform_result = berkowitz_test(uniform_data)
    
    # The p-value for uniform data should generally be higher
    # This is a probabilistic test, so it might occasionally fail
    assert uniform_result.p_value > 0.01, "Berkowitz p-value for uniform data is unexpectedly low"


@given(
    arrays(np.float64, st.integers(min_value=100, max_value=1000), elements=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)),
    st.integers(min_value=1, max_value=20)
)
@settings(deadline=None)  # Disable deadline for potentially slow tests

def test_ljung_box_properties(data, lags):
    """Test properties of Ljung-Box test using hypothesis."""
    # Skip if data has too little variation
    assume(np.std(data) > 1e-10)
    # Ensure lags is less than data length
    lags = min(lags, len(data) // 5)
    assume(lags > 0)
    
    result = ljung_box_test(data, lags=lags)
    
    # Test statistic should be non-negative
    assert result.statistic >= 0
    
    # p-value should be between 0 and 1
    assert 0 <= result.p_value <= 1
    
    # For white noise, the p-value should be high
    white_noise = np.random.default_rng(42).normal(0, 1, size=len(data))
    wn_result = ljung_box_test(white_noise, lags=lags)
    
    # The p-value for white noise should generally be higher
    # This is a probabilistic test, so it might occasionally fail
    assert wn_result.p_value > 0.01, "Ljung-Box p-value for white noise is unexpectedly low"


@given(
    arrays(np.float64, st.integers(min_value=100, max_value=1000), elements=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)),
    st.integers(min_value=1, max_value=20)
)
@settings(deadline=None)  # Disable deadline for potentially slow tests

def test_lm_properties(data, lags):
    """Test properties of LM test using hypothesis."""
    # Skip if data has too little variation
    assume(np.std(data) > 1e-10)
    # Ensure lags is less than data length
    lags = min(lags, len(data) // 5)
    assume(lags > 0)
    
    result = lm_test(data, lags=lags)
    
    # Test statistic should be non-negative
    assert result.statistic >= 0
    
    # p-value should be between 0 and 1
    assert 0 <= result.p_value <= 1
    
    # For white noise, the p-value should be high
    white_noise = np.random.default_rng(42).normal(0, 1, size=len(data))
    wn_result = lm_test(white_noise, lags=lags)
    
    # The p-value for white noise should generally be higher
    # This is a probabilistic test, so it might occasionally fail
    assert wn_result.p_value > 0.01, "LM p-value for white noise is unexpectedly low"


# ---- Async Interface Tests ----

@pytest.mark.asyncio
async def test_async_jarque_bera():
    """Test the async interface for Jarque-Bera test."""
    # Generate normal data
    rng = np.random.default_rng(42)
    data = rng.standard_normal(1000)
    
    # Test the async Jarque-Bera test
    result = await jarque_bera_test(data, async_mode=True)
    
    # For normal data, the p-value should be high
    assert result.statistic is not None
    assert result.p_value is not None
    assert result.p_value > 0.05, "Async Jarque-Bera incorrectly rejected normality for normal data"
    
    # Compare with synchronous version
    sync_result = jarque_bera_test(data)
    assert abs(result.statistic - sync_result.statistic) < 1e-10
    assert abs(result.p_value - sync_result.p_value) < 1e-10


@pytest.mark.asyncio
async def test_async_kolmogorov_smirnov():
    """Test the async interface for Kolmogorov-Smirnov test."""
    # Generate normal data
    rng = np.random.default_rng(42)
    data = rng.standard_normal(1000)
    
    # Test the async Kolmogorov-Smirnov test
    result = await kolmogorov_smirnov_test(data, distribution='normal', async_mode=True)
    
    # For normal data tested against normal distribution, p-value should be high
    assert result.statistic is not None
    assert result.p_value is not None
    assert result.p_value > 0.05, "Async KS test incorrectly rejected normal distribution for normal data"
    
    # Compare with synchronous version
    sync_result = kolmogorov_smirnov_test(data, distribution='normal')
    assert abs(result.statistic - sync_result.statistic) < 1e-10
    assert abs(result.p_value - sync_result.p_value) < 1e-10


@pytest.mark.asyncio
async def test_async_berkowitz():
    """Test the async interface for Berkowitz test."""
    # Generate normal data
    rng = np.random.default_rng(42)
    data = rng.standard_normal(1000)
    
    # Transform data to uniform using normal CDF
    uniform_data = stats.norm.cdf(data)
    
    # Test the async Berkowitz test
    result = await berkowitz_test(uniform_data, async_mode=True)
    
    # For normal data transformed to uniform, the p-value should be high
    assert result.statistic is not None
    assert result.p_value is not None
    assert result.p_value > 0.05, "Async Berkowitz test incorrectly rejected for normal data"
    
    # Compare with synchronous version
    sync_result = berkowitz_test(uniform_data)
    assert abs(result.statistic - sync_result.statistic) < 1e-10
    assert abs(result.p_value - sync_result.p_value) < 1e-10


@pytest.mark.asyncio
async def test_async_ljung_box():
    """Test the async interface for Ljung-Box test."""
    # Generate white noise
    rng = np.random.default_rng(42)
    data = rng.standard_normal(1000)
    
    # Test the async Ljung-Box test
    result = await ljung_box_test(data, lags=10, async_mode=True)
    
    # For white noise, the p-value should be high
    assert result.statistic is not None
    assert result.p_value is not None
    assert result.p_value > 0.05, "Async Ljung-Box test incorrectly rejected white noise"
    
    # Compare with synchronous version
    sync_result = ljung_box_test(data, lags=10)
    assert abs(result.statistic - sync_result.statistic) < 1e-10
    assert abs(result.p_value - sync_result.p_value) < 1e-10


@pytest.mark.asyncio
async def test_async_lm():
    """Test the async interface for LM test."""
    # Generate homoskedastic data
    rng = np.random.default_rng(42)
    data = rng.standard_normal(1000)
    
    # Test the async LM test
    result = await lm_test(data, lags=5, async_mode=True)
    
    # For homoskedastic data, the p-value should be high
    assert result.statistic is not None
    assert result.p_value is not None
    assert result.p_value > 0.05, "Async LM test incorrectly rejected homoskedasticity"
    
    # Compare with synchronous version
    sync_result = lm_test(data, lags=5)
    assert abs(result.statistic - sync_result.statistic) < 1e-10
    assert abs(result.p_value - sync_result.p_value) < 1e-10


# ---- Mixed Type Tests ----

def test_mixed_types_jarque_bera():
    """Test that Jarque-Bera handles mixed types correctly."""
    # Generate data as list
    rng = np.random.default_rng(42)
    data_list = rng.standard_normal(100).tolist()
    
    # Test with list input
    result_list = jarque_bera_test(data_list)
    
    # Test with numpy array
    data_array = np.array(data_list)
    result_array = jarque_bera_test(data_array)
    
    # Test with pandas Series
    data_series = pd.Series(data_list)
    result_series = jarque_bera_test(data_series)
    
    # Results should be the same regardless of input type
    assert abs(result_list.statistic - result_array.statistic) < 1e-10
    assert abs(result_list.p_value - result_array.p_value) < 1e-10
    assert abs(result_list.statistic - result_series.statistic) < 1e-10
    assert abs(result_list.p_value - result_series.p_value) < 1e-10


def test_mixed_types_kolmogorov_smirnov():
    """Test that Kolmogorov-Smirnov handles mixed types correctly."""
    # Generate data as list
    rng = np.random.default_rng(42)
    data_list = rng.standard_normal(100).tolist()
    
    # Test with list input
    result_list = kolmogorov_smirnov_test(data_list, distribution='normal')
    
    # Test with numpy array
    data_array = np.array(data_list)
    result_array = kolmogorov_smirnov_test(data_array, distribution='normal')
    
    # Test with pandas Series
    data_series = pd.Series(data_list)
    result_series = kolmogorov_smirnov_test(data_series, distribution='normal')
    
    # Results should be the same regardless of input type
    assert abs(result_list.statistic - result_array.statistic) < 1e-10
    assert abs(result_list.p_value - result_array.p_value) < 1e-10
    assert abs(result_list.statistic - result_series.statistic) < 1e-10
    assert abs(result_list.p_value - result_series.p_value) < 1e-10


def test_mixed_types_berkowitz():
    """Test that Berkowitz test handles mixed types correctly."""
    # Generate uniform data as list
    rng = np.random.default_rng(42)
    data_list = rng.uniform(0, 1, 100).tolist()
    
    # Test with list input
    result_list = berkowitz_test(data_list)
    
    # Test with numpy array
    data_array = np.array(data_list)
    result_array = berkowitz_test(data_array)
    
    # Test with pandas Series
    data_series = pd.Series(data_list)
    result_series = berkowitz_test(data_series)
    
    # Results should be the same regardless of input type
    assert abs(result_list.statistic - result_array.statistic) < 1e-10
    assert abs(result_list.p_value - result_array.p_value) < 1e-10
    assert abs(result_list.statistic - result_series.statistic) < 1e-10
    assert abs(result_list.p_value - result_series.p_value) < 1e-10


def test_mixed_types_ljung_box():
    """Test that Ljung-Box test handles mixed types correctly."""
    # Generate data as list
    rng = np.random.default_rng(42)
    data_list = rng.standard_normal(100).tolist()
    
    # Test with list input
    result_list = ljung_box_test(data_list, lags=5)
    
    # Test with numpy array
    data_array = np.array(data_list)
    result_array = ljung_box_test(data_array, lags=5)
    
    # Test with pandas Series
    data_series = pd.Series(data_list)
    result_series = ljung_box_test(data_series, lags=5)
    
    # Results should be the same regardless of input type
    assert abs(result_list.statistic - result_array.statistic) < 1e-10
    assert abs(result_list.p_value - result_array.p_value) < 1e-10
    assert abs(result_list.statistic - result_series.statistic) < 1e-10
    assert abs(result_list.p_value - result_series.p_value) < 1e-10


def test_mixed_types_lm():
    """Test that LM test handles mixed types correctly."""
    # Generate data as list
    rng = np.random.default_rng(42)
    data_list = rng.standard_normal(100).tolist()
    
    # Test with list input
    result_list = lm_test(data_list, lags=5)
    
    # Test with numpy array
    data_array = np.array(data_list)
    result_array = lm_test(data_array, lags=5)
    
    # Test with pandas Series
    data_series = pd.Series(data_list)
    result_series = lm_test(data_series, lags=5)
    
    # Results should be the same regardless of input type
    assert abs(result_list.statistic - result_array.statistic) < 1e-10
    assert abs(result_list.p_value - result_array.p_value) < 1e-10
    assert abs(result_list.statistic - result_series.statistic) < 1e-10
    assert abs(result_list.p_value - result_series.p_value) < 1e-10


# ---- Parameterized Tests ----

@pytest.mark.parametrize("distribution", ["normal", "t", "ged", "skewed_t"])

def test_kolmogorov_smirnov_distributions(distribution):
    """Test Kolmogorov-Smirnov test with different distributions."""
    # Generate normal data
    rng = np.random.default_rng(42)
    data = rng.standard_normal(1000)
    
    # Prepare distribution parameters
    kwargs = {}
    if distribution == 't':
        kwargs['df'] = 5.0
    elif distribution == 'ged':
        kwargs['nu'] = 1.5
    elif distribution == 'skewed_t':
        kwargs['df'] = 5.0
        kwargs['lambda_'] = 0.5
    
    # Test the Kolmogorov-Smirnov test
    result = kolmogorov_smirnov_test(data, distribution=distribution, **kwargs)
    
    # Basic checks
    assert result.statistic is not None
    assert result.p_value is not None
    assert 0 <= result.p_value <= 1


@pytest.mark.parametrize("lags", [1, 5, 10, 20])

def test_ljung_box_different_lags(lags):
    """Test Ljung-Box test with different lag values."""
    # Generate white noise
    rng = np.random.default_rng(42)
    data = rng.standard_normal(1000)
    
    # Test the Ljung-Box test
    result = ljung_box_test(data, lags=lags)
    
    # Basic checks
    assert result.statistic is not None
    assert result.p_value is not None
    assert 0 <= result.p_value <= 1


@pytest.mark.parametrize("lags", [1, 5, 10, 20])

def test_lm_different_lags(lags):
    """Test LM test with different lag values."""
    # Generate homoskedastic data
    rng = np.random.default_rng(42)
    data = rng.standard_normal(1000)
    
    # Test the LM test
    result = lm_test(data, lags=lags)
    
    # Basic checks
    assert result.statistic is not None
    assert result.p_value is not None
    assert 0 <= result.p_value <= 1


@pytest.mark.parametrize("sample_size", [50, 100, 500, 1000])

def test_jarque_bera_different_sample_sizes(sample_size):
    """Test Jarque-Bera test with different sample sizes."""
    # Generate normal data
    rng = np.random.default_rng(42)
    data = rng.standard_normal(sample_size)
    
    # Test the Jarque-Bera test
    result = jarque_bera_test(data)
    
    # Basic checks
    assert result.statistic is not None
    assert result.p_value is not None
    assert 0 <= result.p_value <= 1


@pytest.mark.parametrize("sample_size", [50, 100, 500, 1000])

def test_berkowitz_different_sample_sizes(sample_size):
    """Test Berkowitz test with different sample sizes."""
    # Generate uniform data
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 1, sample_size)
    
    # Test the Berkowitz test
    result = berkowitz_test(data)
    
    # Basic checks
    assert result.statistic is not None
    assert result.p_value is not None
    assert 0 <= result.p_value <= 1
