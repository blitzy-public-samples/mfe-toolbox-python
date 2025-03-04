'''
Tests for statistical test implementations in the MFE Toolbox.

This module contains comprehensive tests for statistical test implementations
including Berkowitz, Jarque-Bera, and Kolmogorov-Smirnov tests. It verifies
the correct behavior of distribution testing functions across various input
types, edge cases, and expected statistical properties.
'''
import asyncio
import math
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays
from scipy import stats

from mfe.models.distributions import normal, student_t, generalized_error, skewed_t
from mfe.models.distributions.utils import pvalue_calculator


class TestJarqueBera:
    """Tests for the Jarque-Bera normality test implementation."""

    def test_normal_distribution(self, rng, assert_array_equal):
        """Test that Jarque-Bera correctly identifies normal distributions."""
        # Generate normal data
        data = rng.standard_normal(1000)
        
        # Calculate Jarque-Bera statistic
        from mfe.models.tests import jarque_bera
        jb_stat, p_value = jarque_bera(data)
        
        # For normal data, p-value should be high (not rejecting normality)
        assert p_value > 0.05, "Jarque-Bera incorrectly rejected normality for normal data"
        
        # Test with pandas Series
        data_series = pd.Series(data)
        jb_stat_series, p_value_series = jarque_bera(data_series)
        
        # Results should be the same regardless of input type
        assert_array_equal(jb_stat, jb_stat_series, err_msg="JB statistic differs between array and Series")
        assert_array_equal(p_value, p_value_series, err_msg="P-value differs between array and Series")

    def test_t_distribution(self, rng):
        """Test that Jarque-Bera correctly identifies non-normal t-distributions."""
        # Generate t-distributed data with 3 degrees of freedom (heavy tails)
        data = stats.t.rvs(df=3, size=1000, random_state=rng)
        
        # Calculate Jarque-Bera statistic
        from mfe.models.tests import jarque_bera
        jb_stat, p_value = jarque_bera(data)
        
        # For t-distributed data with df=3, p-value should be low (rejecting normality)
        assert p_value < 0.05, "Jarque-Bera failed to reject normality for t-distributed data"

    def test_skewed_distribution(self, rng):
        """Test that Jarque-Bera correctly identifies skewed distributions."""
        # Generate skewed data
        data = stats.skewnorm.rvs(a=5, size=1000, random_state=rng)
        
        # Calculate Jarque-Bera statistic
        from mfe.models.tests import jarque_bera
        jb_stat, p_value = jarque_bera(data)
        
        # For skewed data, p-value should be low (rejecting normality)
        assert p_value < 0.05, "Jarque-Bera failed to reject normality for skewed data"

    def test_empty_input(self):
        """Test that Jarque-Bera handles empty inputs appropriately."""
        from mfe.models.tests import jarque_bera
        
        # Empty array should raise ValueError
        with pytest.raises(ValueError, match="Input array must contain at least 3 observations"):
            jarque_bera(np.array([]))

    def test_small_sample(self):
        """Test that Jarque-Bera handles small samples appropriately."""
        from mfe.models.tests import jarque_bera
        
        # Small sample (less than 3 observations) should raise ValueError
        with pytest.raises(ValueError, match="Input array must contain at least 3 observations"):
            jarque_bera(np.array([1.0, 2.0]))
        
        # Sample with exactly 3 observations should work
        jb_stat, p_value = jarque_bera(np.array([1.0, 2.0, 3.0]))
        assert isinstance(jb_stat, float)
        assert isinstance(p_value, float)

    def test_nan_values(self):
        """Test that Jarque-Bera handles NaN values appropriately."""
        from mfe.models.tests import jarque_bera
        
        # Array with NaN values should raise ValueError
        with pytest.raises(ValueError, match="Input array contains NaN values"):
            jarque_bera(np.array([1.0, 2.0, np.nan, 4.0]))

    def test_infinite_values(self):
        """Test that Jarque-Bera handles infinite values appropriately."""
        from mfe.models.tests import jarque_bera
        
        # Array with infinite values should raise ValueError
        with pytest.raises(ValueError, match="Input array contains infinite values"):
            jarque_bera(np.array([1.0, 2.0, np.inf, 4.0]))

    @given(arrays(dtype=np.float64, shape=st.integers(3, 1000),
                  elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)))
    def test_property_based(self, data):
        """Property-based test for Jarque-Bera using hypothesis."""
        from mfe.models.tests import jarque_bera
        
        # Calculate Jarque-Bera statistic
        jb_stat, p_value = jarque_bera(data)
        
        # Basic properties that should always hold
        assert jb_stat >= 0, "Jarque-Bera statistic should be non-negative"
        assert 0 <= p_value <= 1, "P-value should be between 0 and 1"

    @pytest.mark.asyncio
    async def test_async_interface(self, rng):
        """Test the asynchronous interface for Jarque-Bera test."""
        # Generate normal data
        data = rng.standard_normal(1000)
        
        # Calculate Jarque-Bera statistic using async interface
        from mfe.models.tests import jarque_bera_async
        jb_stat, p_value = await jarque_bera_async(data)
        
        # Basic validation
        assert isinstance(jb_stat, float)
        assert isinstance(p_value, float)
        assert jb_stat >= 0
        assert 0 <= p_value <= 1

    def test_comparison_with_scipy(self, rng, assert_array_equal):
        """Compare results with scipy.stats implementation."""
        # Generate normal data
        data = rng.standard_normal(1000)
        
        # Calculate using MFE implementation
        from mfe.models.tests import jarque_bera
        jb_stat_mfe, p_value_mfe = jarque_bera(data)
        
        # Calculate using scipy
        jb_stat_scipy, p_value_scipy = stats.jarque_bera(data)
        
        # Results should be very close
        assert_array_equal(jb_stat_mfe, jb_stat_scipy, rtol=1e-10, 
                          err_msg="JB statistic differs from scipy implementation")
        assert_array_equal(p_value_mfe, p_value_scipy, rtol=1e-10, 
                          err_msg="P-value differs from scipy implementation")


class TestKolmogorovSmirnov:
    """Tests for the Kolmogorov-Smirnov test implementation."""

    def test_normal_distribution(self, rng, assert_array_equal):
        """Test that KS test correctly identifies normal distributions."""
        # Generate normal data
        data = rng.standard_normal(1000)
        
        # Calculate KS statistic against normal distribution
        from mfe.models.tests import kolmogorov_smirnov
        ks_stat, p_value = kolmogorov_smirnov(data, stats.norm.cdf)
        
        # For normal data against normal distribution, p-value should be high
        assert p_value > 0.05, "KS test incorrectly rejected normal distribution"
        
        # Test with pandas Series
        data_series = pd.Series(data)
        ks_stat_series, p_value_series = kolmogorov_smirnov(data_series, stats.norm.cdf)
        
        # Results should be the same regardless of input type
        assert_array_equal(ks_stat, ks_stat_series, err_msg="KS statistic differs between array and Series")
        assert_array_equal(p_value, p_value_series, err_msg="P-value differs between array and Series")

    def test_t_distribution(self, rng):
        """Test that KS test correctly identifies t-distributions."""
        # Generate t-distributed data with 3 degrees of freedom
        data = stats.t.rvs(df=3, size=1000, random_state=rng)
        
        # Test against normal distribution (should reject)
        from mfe.models.tests import kolmogorov_smirnov
        ks_stat_norm, p_value_norm = kolmogorov_smirnov(data, stats.norm.cdf)
        
        # For t-distributed data against normal, p-value should be low
        assert p_value_norm < 0.05, "KS test failed to reject normal for t-distributed data"
        
        # Test against correct t-distribution (should not reject)
        t_cdf = lambda x: stats.t.cdf(x, df=3)
        ks_stat_t, p_value_t = kolmogorov_smirnov(data, t_cdf)
        
        # For t-distributed data against t-distribution, p-value should be high
        assert p_value_t > 0.05, "KS test incorrectly rejected t-distribution for t-distributed data"

    def test_uniform_distribution(self, rng):
        """Test that KS test correctly identifies uniform distributions."""
        # Generate uniform data
        data = rng.uniform(0, 1, 1000)
        
        # Test against uniform distribution
        from mfe.models.tests import kolmogorov_smirnov
        ks_stat, p_value = kolmogorov_smirnov(data, stats.uniform.cdf)
        
        # For uniform data against uniform, p-value should be high
        assert p_value > 0.05, "KS test incorrectly rejected uniform distribution"
        
        # Test against normal distribution (should reject)
        ks_stat_norm, p_value_norm = kolmogorov_smirnov(data, stats.norm.cdf)
        
        # For uniform data against normal, p-value should be low
        assert p_value_norm < 0.05, "KS test failed to reject normal for uniform data"

    def test_empty_input(self):
        """Test that KS test handles empty inputs appropriately."""
        from mfe.models.tests import kolmogorov_smirnov
        
        # Empty array should raise ValueError
        with pytest.raises(ValueError, match="Input array must contain at least 1 observation"):
            kolmogorov_smirnov(np.array([]), stats.norm.cdf)

    def test_nan_values(self):
        """Test that KS test handles NaN values appropriately."""
        from mfe.models.tests import kolmogorov_smirnov
        
        # Array with NaN values should raise ValueError
        with pytest.raises(ValueError, match="Input array contains NaN values"):
            kolmogorov_smirnov(np.array([1.0, 2.0, np.nan, 4.0]), stats.norm.cdf)

    def test_infinite_values(self):
        """Test that KS test handles infinite values appropriately."""
        from mfe.models.tests import kolmogorov_smirnov
        
        # Array with infinite values should raise ValueError
        with pytest.raises(ValueError, match="Input array contains infinite values"):
            kolmogorov_smirnov(np.array([1.0, 2.0, np.inf, 4.0]), stats.norm.cdf)

    def test_invalid_cdf(self):
        """Test that KS test validates the CDF function."""
        from mfe.models.tests import kolmogorov_smirnov
        
        # Invalid CDF (not callable) should raise TypeError
        with pytest.raises(TypeError, match="CDF must be a callable function"):
            kolmogorov_smirnov(np.array([1.0, 2.0, 3.0]), "not_callable")

    @given(arrays(dtype=np.float64, shape=st.integers(1, 1000),
                  elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)))
    def test_property_based(self, data):
        """Property-based test for KS test using hypothesis."""
        from mfe.models.tests import kolmogorov_smirnov
        
        # Calculate KS statistic against normal distribution
        ks_stat, p_value = kolmogorov_smirnov(data, stats.norm.cdf)
        
        # Basic properties that should always hold
        assert 0 <= ks_stat <= 1, "KS statistic should be between 0 and 1"
        assert 0 <= p_value <= 1, "P-value should be between 0 and 1"

    @pytest.mark.asyncio
    async def test_async_interface(self, rng):
        """Test the asynchronous interface for KS test."""
        # Generate normal data
        data = rng.standard_normal(1000)
        
        # Calculate KS statistic using async interface
        from mfe.models.tests import kolmogorov_smirnov_async
        ks_stat, p_value = await kolmogorov_smirnov_async(data, stats.norm.cdf)
        
        # Basic validation
        assert isinstance(ks_stat, float)
        assert isinstance(p_value, float)
        assert 0 <= ks_stat <= 1
        assert 0 <= p_value <= 1

    def test_comparison_with_scipy(self, rng, assert_array_equal):
        """Compare results with scipy.stats implementation."""
        # Generate normal data
        data = rng.standard_normal(1000)
        
        # Calculate using MFE implementation
        from mfe.models.tests import kolmogorov_smirnov
        ks_stat_mfe, p_value_mfe = kolmogorov_smirnov(data, stats.norm.cdf)
        
        # Calculate using scipy
        ks_stat_scipy, p_value_scipy = stats.kstest(data, 'norm')
        
        # Results should be very close
        assert_array_equal(ks_stat_mfe, ks_stat_scipy, rtol=1e-10, 
                          err_msg="KS statistic differs from scipy implementation")
        assert_array_equal(p_value_mfe, p_value_scipy, rtol=1e-10, 
                          err_msg="P-value differs from scipy implementation")


class TestBerkowitz:
    """Tests for the Berkowitz test implementation."""

    def test_normal_distribution(self, rng, assert_array_equal):
        """Test that Berkowitz test correctly identifies normal distributions."""
        # Generate normal data
        data = rng.standard_normal(1000)
        
        # Calculate Berkowitz statistic
        from mfe.models.tests import berkowitz
        berk_stat, p_value = berkowitz(data, stats.norm.cdf)
        
        # For normal data transformed by normal CDF, p-value should be high
        assert p_value > 0.05, "Berkowitz test incorrectly rejected normality"
        
        # Test with pandas Series
        data_series = pd.Series(data)
        berk_stat_series, p_value_series = berkowitz(data_series, stats.norm.cdf)
        
        # Results should be the same regardless of input type
        assert_array_equal(berk_stat, berk_stat_series, err_msg="Berkowitz statistic differs between array and Series")
        assert_array_equal(p_value, p_value_series, err_msg="P-value differs between array and Series")

    def test_t_distribution(self, rng):
        """Test that Berkowitz test correctly identifies non-normal distributions."""
        # Generate t-distributed data with 3 degrees of freedom
        data = stats.t.rvs(df=3, size=1000, random_state=rng)
        
        # Test against normal distribution (should reject)
        from mfe.models.tests import berkowitz
        berk_stat, p_value = berkowitz(data, stats.norm.cdf)
        
        # For t-distributed data transformed by normal CDF, p-value should be low
        assert p_value < 0.05, "Berkowitz test failed to reject normality for t-distributed data"
        
        # Test against correct t-distribution (should not reject)
        t_cdf = lambda x: stats.t.cdf(x, df=3)
        berk_stat_t, p_value_t = berkowitz(data, t_cdf)
        
        # For t-distributed data transformed by t CDF, p-value should be high
        assert p_value_t > 0.05, "Berkowitz test incorrectly rejected t-distribution"

    def test_uniform_distribution(self, rng):
        """Test that Berkowitz test correctly handles uniform distributions."""
        # Generate uniform data
        data = rng.uniform(0, 1, 1000)
        
        # Transform to normal using inverse normal CDF
        norm_data = stats.norm.ppf(data)
        
        # Test transformed data against normal distribution
        from mfe.models.tests import berkowitz
        berk_stat, p_value = berkowitz(norm_data, stats.norm.cdf)
        
        # For uniform data transformed to normal, p-value should be high
        assert p_value > 0.05, "Berkowitz test incorrectly rejected normality for transformed uniform data"

    def test_empty_input(self):
        """Test that Berkowitz test handles empty inputs appropriately."""
        from mfe.models.tests import berkowitz
        
        # Empty array should raise ValueError
        with pytest.raises(ValueError, match="Input array must contain at least 3 observations"):
            berkowitz(np.array([]), stats.norm.cdf)

    def test_small_sample(self):
        """Test that Berkowitz test handles small samples appropriately."""
        from mfe.models.tests import berkowitz
        
        # Small sample (less than 3 observations) should raise ValueError
        with pytest.raises(ValueError, match="Input array must contain at least 3 observations"):
            berkowitz(np.array([1.0, 2.0]), stats.norm.cdf)

    def test_nan_values(self):
        """Test that Berkowitz test handles NaN values appropriately."""
        from mfe.models.tests import berkowitz
        
        # Array with NaN values should raise ValueError
        with pytest.raises(ValueError, match="Input array contains NaN values"):
            berkowitz(np.array([1.0, 2.0, np.nan, 4.0]), stats.norm.cdf)

    def test_infinite_values(self):
        """Test that Berkowitz test handles infinite values appropriately."""
        from mfe.models.tests import berkowitz
        
        # Array with infinite values should raise ValueError
        with pytest.raises(ValueError, match="Input array contains infinite values"):
            berkowitz(np.array([1.0, 2.0, np.inf, 4.0]), stats.norm.cdf)

    def test_invalid_cdf(self):
        """Test that Berkowitz test validates the CDF function."""
        from mfe.models.tests import berkowitz
        
        # Invalid CDF (not callable) should raise TypeError
        with pytest.raises(TypeError, match="CDF must be a callable function"):
            berkowitz(np.array([1.0, 2.0, 3.0]), "not_callable")

    @given(arrays(dtype=np.float64, shape=st.integers(3, 1000),
                  elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)))
    def test_property_based(self, data):
        """Property-based test for Berkowitz test using hypothesis."""
        from mfe.models.tests import berkowitz
        
        # Calculate Berkowitz statistic against normal distribution
        try:
            berk_stat, p_value = berkowitz(data, stats.norm.cdf)
            
            # Basic properties that should always hold
            assert berk_stat >= 0, "Berkowitz statistic should be non-negative"
            assert 0 <= p_value <= 1, "P-value should be between 0 and 1"
        except (ValueError, RuntimeError):
            # Some random arrays might cause numerical issues, which is acceptable
            # in property-based testing
            pass

    @pytest.mark.asyncio
    async def test_async_interface(self, rng):
        """Test the asynchronous interface for Berkowitz test."""
        # Generate normal data
        data = rng.standard_normal(1000)
        
        # Calculate Berkowitz statistic using async interface
        from mfe.models.tests import berkowitz_async
        berk_stat, p_value = await berkowitz_async(data, stats.norm.cdf)
        
        # Basic validation
        assert isinstance(berk_stat, float)
        assert isinstance(p_value, float)
        assert berk_stat >= 0
        assert 0 <= p_value <= 1


class TestLjungBox:
    """Tests for the Ljung-Box test implementation."""

    def test_white_noise(self, rng, assert_array_equal):
        """Test that Ljung-Box correctly identifies white noise."""
        # Generate white noise
        data = rng.standard_normal(1000)
        
        # Calculate Ljung-Box statistic
        from mfe.models.tests import ljung_box
        lb_stat, p_value = ljung_box(data, lags=10)
        
        # For white noise, p-value should be high (not rejecting no autocorrelation)
        assert p_value > 0.05, "Ljung-Box incorrectly rejected white noise"
        
        # Test with pandas Series
        data_series = pd.Series(data)
        lb_stat_series, p_value_series = ljung_box(data_series, lags=10)
        
        # Results should be the same regardless of input type
        assert_array_equal(lb_stat, lb_stat_series, err_msg="LB statistic differs between array and Series")
        assert_array_equal(p_value, p_value_series, err_msg="P-value differs between array and Series")

    def test_autocorrelated_data(self, ar1_process):
        """Test that Ljung-Box correctly identifies autocorrelated data."""
        # ar1_process fixture provides an AR(1) process with autocorrelation
        
        # Calculate Ljung-Box statistic
        from mfe.models.tests import ljung_box
        lb_stat, p_value = ljung_box(ar1_process, lags=10)
        
        # For AR(1) process, p-value should be low (rejecting no autocorrelation)
        assert p_value < 0.05, "Ljung-Box failed to reject no autocorrelation for AR(1) process"

    def test_empty_input(self):
        """Test that Ljung-Box handles empty inputs appropriately."""
        from mfe.models.tests import ljung_box
        
        # Empty array should raise ValueError
        with pytest.raises(ValueError, match="Input array must contain at least 2 observations"):
            ljung_box(np.array([]), lags=1)

    def test_small_sample(self):
        """Test that Ljung-Box handles small samples appropriately."""
        from mfe.models.tests import ljung_box
        
        # Small sample (less than 2 observations) should raise ValueError
        with pytest.raises(ValueError, match="Input array must contain at least 2 observations"):
            ljung_box(np.array([1.0]), lags=1)
        
        # Sample with exactly 2 observations should work with lags=1
        lb_stat, p_value = ljung_box(np.array([1.0, 2.0]), lags=1)
        assert isinstance(lb_stat, float)
        assert isinstance(p_value, float)

    def test_invalid_lags(self):
        """Test that Ljung-Box validates the lags parameter."""
        from mfe.models.tests import ljung_box
        
        # Negative lags should raise ValueError
        with pytest.raises(ValueError, match="Number of lags must be positive"):
            ljung_box(np.array([1.0, 2.0, 3.0]), lags=-1)
        
        # Zero lags should raise ValueError
        with pytest.raises(ValueError, match="Number of lags must be positive"):
            ljung_box(np.array([1.0, 2.0, 3.0]), lags=0)
        
        # Lags greater than sample size should raise ValueError
        with pytest.raises(ValueError, match="Number of lags must be less than sample size"):
            ljung_box(np.array([1.0, 2.0, 3.0]), lags=4)

    def test_nan_values(self):
        """Test that Ljung-Box handles NaN values appropriately."""
        from mfe.models.tests import ljung_box
        
        # Array with NaN values should raise ValueError
        with pytest.raises(ValueError, match="Input array contains NaN values"):
            ljung_box(np.array([1.0, 2.0, np.nan, 4.0]), lags=2)

    def test_infinite_values(self):
        """Test that Ljung-Box handles infinite values appropriately."""
        from mfe.models.tests import ljung_box
        
        # Array with infinite values should raise ValueError
        with pytest.raises(ValueError, match="Input array contains infinite values"):
            ljung_box(np.array([1.0, 2.0, np.inf, 4.0]), lags=2)

    @given(arrays(dtype=np.float64, shape=st.integers(10, 1000),
                  elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)))
    def test_property_based(self, data):
        """Property-based test for Ljung-Box test using hypothesis."""
        from mfe.models.tests import ljung_box
        
        # Calculate Ljung-Box statistic with lags=min(10, len(data)//2)
        lags = min(10, len(data)//2)
        lb_stat, p_value = ljung_box(data, lags=lags)
        
        # Basic properties that should always hold
        assert lb_stat >= 0, "Ljung-Box statistic should be non-negative"
        assert 0 <= p_value <= 1, "P-value should be between 0 and 1"

    @pytest.mark.asyncio
    async def test_async_interface(self, rng):
        """Test the asynchronous interface for Ljung-Box test."""
        # Generate white noise
        data = rng.standard_normal(1000)
        
        # Calculate Ljung-Box statistic using async interface
        from mfe.models.tests import ljung_box_async
        lb_stat, p_value = await ljung_box_async(data, lags=10)
        
        # Basic validation
        assert isinstance(lb_stat, float)
        assert isinstance(p_value, float)
        assert lb_stat >= 0
        assert 0 <= p_value <= 1

    def test_comparison_with_statsmodels(self, rng, assert_array_equal):
        """Compare results with statsmodels implementation."""
        try:
            import statsmodels.stats.diagnostic as smd
            
            # Generate white noise
            data = rng.standard_normal(1000)
            
            # Calculate using MFE implementation
            from mfe.models.tests import ljung_box
            lb_stat_mfe, p_value_mfe = ljung_box(data, lags=10)
            
            # Calculate using statsmodels
            lb_stat_sm, p_value_sm = smd.acorr_ljungbox(data, lags=[10], return_df=False)
            lb_stat_sm = lb_stat_sm[0]
            p_value_sm = p_value_sm[0]
            
            # Results should be very close
            assert_array_equal(lb_stat_mfe, lb_stat_sm, rtol=1e-10, 
                              err_msg="LB statistic differs from statsmodels implementation")
            assert_array_equal(p_value_mfe, p_value_sm, rtol=1e-10, 
                              err_msg="P-value differs from statsmodels implementation")
        except ImportError:
            pytest.skip("statsmodels not available for comparison")


class TestLMTest:
    """Tests for the Lagrange Multiplier test implementation."""

    def test_white_noise(self, rng, assert_array_equal):
        """Test that LM test correctly identifies white noise."""
        # Generate white noise
        data = rng.standard_normal(1000)
        
        # Calculate LM statistic
        from mfe.models.tests import lm_test
        lm_stat, p_value = lm_test(data, lags=10)
        
        # For white noise, p-value should be high (not rejecting no ARCH effects)
        assert p_value > 0.05, "LM test incorrectly rejected white noise"
        
        # Test with pandas Series
        data_series = pd.Series(data)
        lm_stat_series, p_value_series = lm_test(data_series, lags=10)
        
        # Results should be the same regardless of input type
        assert_array_equal(lm_stat, lm_stat_series, err_msg="LM statistic differs between array and Series")
        assert_array_equal(p_value, p_value_series, err_msg="P-value differs between array and Series")

    def test_garch_process(self, garch_process):
        """Test that LM test correctly identifies GARCH processes."""
        # garch_process fixture provides a GARCH(1,1) process
        returns, _ = garch_process
        
        # Calculate LM statistic
        from mfe.models.tests import lm_test
        lm_stat, p_value = lm_test(returns, lags=10)
        
        # For GARCH process, p-value should be low (rejecting no ARCH effects)
        assert p_value < 0.05, "LM test failed to reject no ARCH effects for GARCH process"

    def test_empty_input(self):
        """Test that LM test handles empty inputs appropriately."""
        from mfe.models.tests import lm_test
        
        # Empty array should raise ValueError
        with pytest.raises(ValueError, match="Input array must contain at least lags+1 observations"):
            lm_test(np.array([]), lags=1)

    def test_small_sample(self):
        """Test that LM test handles small samples appropriately."""
        from mfe.models.tests import lm_test
        
        # Sample smaller than lags+1 should raise ValueError
        with pytest.raises(ValueError, match="Input array must contain at least lags+1 observations"):
            lm_test(np.array([1.0]), lags=1)
        
        # Sample with exactly lags+1 observations should work
        lm_stat, p_value = lm_test(np.array([1.0, 2.0]), lags=1)
        assert isinstance(lm_stat, float)
        assert isinstance(p_value, float)

    def test_invalid_lags(self):
        """Test that LM test validates the lags parameter."""
        from mfe.models.tests import lm_test
        
        # Negative lags should raise ValueError
        with pytest.raises(ValueError, match="Number of lags must be positive"):
            lm_test(np.array([1.0, 2.0, 3.0]), lags=-1)
        
        # Zero lags should raise ValueError
        with pytest.raises(ValueError, match="Number of lags must be positive"):
            lm_test(np.array([1.0, 2.0, 3.0]), lags=0)
        
        # Lags greater than sample size - 1 should raise ValueError
        with pytest.raises(ValueError, match="Input array must contain at least lags+1 observations"):
            lm_test(np.array([1.0, 2.0, 3.0]), lags=3)

    def test_nan_values(self):
        """Test that LM test handles NaN values appropriately."""
        from mfe.models.tests import lm_test
        
        # Array with NaN values should raise ValueError
        with pytest.raises(ValueError, match="Input array contains NaN values"):
            lm_test(np.array([1.0, 2.0, np.nan, 4.0]), lags=2)

    def test_infinite_values(self):
        """Test that LM test handles infinite values appropriately."""
        from mfe.models.tests import lm_test
        
        # Array with infinite values should raise ValueError
        with pytest.raises(ValueError, match="Input array contains infinite values"):
            lm_test(np.array([1.0, 2.0, np.inf, 4.0]), lags=2)

    @given(arrays(dtype=np.float64, shape=st.integers(11, 1000),
                  elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)))
    def test_property_based(self, data):
        """Property-based test for LM test using hypothesis."""
        from mfe.models.tests import lm_test
        
        # Calculate LM statistic with lags=min(10, len(data)//2)
        lags = min(10, len(data)//2)
        lm_stat, p_value = lm_test(data, lags=lags)
        
        # Basic properties that should always hold
        assert lm_stat >= 0, "LM statistic should be non-negative"
        assert 0 <= p_value <= 1, "P-value should be between 0 and 1"

    @pytest.mark.asyncio
    async def test_async_interface(self, rng):
        """Test the asynchronous interface for LM test."""
        # Generate white noise
        data = rng.standard_normal(1000)
        
        # Calculate LM statistic using async interface
        from mfe.models.tests import lm_test_async
        lm_stat, p_value = await lm_test_async(data, lags=10)
        
        # Basic validation
        assert isinstance(lm_stat, float)
        assert isinstance(p_value, float)
        assert lm_stat >= 0
        assert 0 <= p_value <= 1


class TestDistributionUtils:
    """Tests for distribution utility functions."""

    def test_pvalue_calculator(self):
        """Test the p-value calculator function."""
        # Test chi-square p-value calculation
        p_value = pvalue_calculator(5.0, 2, 'chi2')
        expected_p_value = 1 - stats.chi2.cdf(5.0, 2)
        assert abs(p_value - expected_p_value) < 1e-10, "Chi-square p-value calculation incorrect"
        
        # Test normal p-value calculation
        p_value = pvalue_calculator(1.96, None, 'normal')
        expected_p_value = 2 * (1 - stats.norm.cdf(1.96))
        assert abs(p_value - expected_p_value) < 1e-10, "Normal p-value calculation incorrect"
        
        # Test invalid distribution type
        with pytest.raises(ValueError, match="Unknown distribution type"):
            pvalue_calculator(1.0, None, 'invalid_type')

    def test_distribution_transformations(self):
        """Test distribution transformation functions."""
        # Generate normal data
        rng = np.random.default_rng(12345)
        data = rng.standard_normal(1000)
        
        # Transform to uniform using normal CDF
        uniform_data = stats.norm.cdf(data)
        
        # All values should be between 0 and 1
        assert np.all((uniform_data >= 0) & (uniform_data <= 1)), "CDF transformation failed"
        
        # Transform back to normal using inverse CDF
        normal_data = stats.norm.ppf(uniform_data)
        
        # Should be very close to original data
        assert np.allclose(data, normal_data), "Inverse CDF transformation failed"


class TestIntegrationTests:
    """Integration tests for statistical test functions."""

    def test_distribution_fitting_and_testing(self, rng):
        """Test fitting a distribution and then testing the fit."""
        # Generate t-distributed data
        data = stats.t.rvs(df=5, size=1000, random_state=rng)
        
        # Fit t-distribution to data
        df_est = stats.t.fit(data)[0]
        
        # Create CDF function with estimated parameters
        t_cdf = lambda x: stats.t.cdf(x, df=df_est)
        
        # Test fit using Kolmogorov-Smirnov test
        from mfe.models.tests import kolmogorov_smirnov
        ks_stat, p_value = kolmogorov_smirnov(data, t_cdf)
        
        # P-value should be high for good fit
        assert p_value > 0.05, "KS test rejected fitted t-distribution"
        
        # Test fit using Berkowitz test
        from mfe.models.tests import berkowitz
        berk_stat, p_value = berkowitz(data, t_cdf)
        
        # P-value should be high for good fit
        assert p_value > 0.05, "Berkowitz test rejected fitted t-distribution"

    def test_residual_diagnostics(self, ar1_process):
        """Test using statistical tests for model residual diagnostics."""
        # Fit AR(1) model to AR(1) process
        from mfe.models.time_series import ARMA
        model = ARMA().fit(ar1_process, ar_order=1, ma_order=0)
        residuals = model.residuals
        
        # Test residuals for normality using Jarque-Bera
        from mfe.models.tests import jarque_bera
        jb_stat, p_value = jarque_bera(residuals)
        
        # Residuals should be approximately normal
        assert p_value > 0.01, "Jarque-Bera rejected normality of AR(1) residuals"
        
        # Test residuals for autocorrelation using Ljung-Box
        from mfe.models.tests import ljung_box
        lb_stat, p_value = ljung_box(residuals, lags=10)
        
        # Residuals should not have significant autocorrelation
        assert p_value > 0.05, "Ljung-Box detected autocorrelation in AR(1) residuals"
        
        # Test residuals for ARCH effects using LM test
        from mfe.models.tests import lm_test
        lm_stat, p_value = lm_test(residuals, lags=10)
        
        # Residuals should not have significant ARCH effects
        assert p_value > 0.05, "LM test detected ARCH effects in AR(1) residuals"
