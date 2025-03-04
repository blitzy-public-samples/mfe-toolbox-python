# tests/test_realized.py
'''
Tests for realized volatility estimators and high-frequency financial econometrics.

This module contains comprehensive tests for realized volatility estimators and
high-frequency financial econometrics functionality, including realized variance,
bipower variation, kernel estimators, and noise-robust techniques. It verifies
correct handling of irregularly spaced data, microstructure noise, and various
sampling schemes.
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

from mfe.models.realized import (
    variance, bipower_variation, kernel, semivariance, quarticity,
    multiscale_variance, twoscale_variance, qmle_variance, range as rv_range,
    threshold_variance, preaveraged_variance, preaveraged_bipower_variation,
    threshold_multipower_variation, covariance, multivariate_kernel,
    seconds2unit, unit2seconds, seconds2wall, wall2seconds, unit2wall, wall2unit,
    price_filter, return_filter, subsample, refresh_time, refresh_time_bivariate,
    variance_optimal_sampling, kernel_bandwidth, kernel_weights, kernel_jitter_lag_length
)
from mfe.models.realized._numba_core import (
    realized_variance_core, realized_bipower_variation_core, realized_quarticity_core,
    realized_semivariance_core, realized_threshold_variance_core
)


class TestRealizedVariance:
    """Tests for realized variance estimators."""

    def test_realized_variance_basic(self, high_frequency_data, assert_array_equal):
        """Test basic realized variance calculation."""
        # Extract returns from high-frequency data
        prices = high_frequency_data['price'].values
        returns = np.diff(np.log(prices))

        # Calculate realized variance
        rv = variance(returns)

        # Basic checks
        assert isinstance(rv, float), "Realized variance should be a float"
        assert rv > 0, "Realized variance should be positive"

        # Calculate manually for comparison
        manual_rv = np.sum(returns**2)
        assert_array_equal(rv, manual_rv, rtol=1e-10,
                           err_msg="Realized variance doesn't match manual calculation")

        # Test with pandas Series
        returns_series = pd.Series(returns)
        rv_series = variance(returns_series)
        assert isinstance(rv_series, float), "Realized variance should be a float even with Series input"
        assert_array_equal(rv, rv_series, rtol=1e-10,
                           err_msg="Realized variance differs between ndarray and Series")

    def test_realized_variance_with_timestamps(self, high_frequency_data, assert_array_equal):
        """Test realized variance with timestamp information."""
        # Extract prices and times from high-frequency data
        prices = high_frequency_data['price'].values
        times = high_frequency_data['time'].values

        # Calculate realized variance with timestamps
        rv = variance(prices, times=times, use_prices=True)

        # Basic checks
        assert isinstance(rv, float), "Realized variance should be a float"
        assert rv > 0, "Realized variance should be positive"

        # Calculate manually for comparison
        log_prices = np.log(prices)
        returns = np.diff(log_prices)
        manual_rv = np.sum(returns**2)
        assert_array_equal(rv, manual_rv, rtol=1e-10,
                           err_msg="Realized variance with timestamps doesn't match manual calculation")

        # Test with pandas DataFrame
        rv_df = variance(high_frequency_data['price'],
                         times=high_frequency_data['time'],
                         use_prices=True)
        assert isinstance(rv_df, float), "Realized variance should be a float with DataFrame input"
        assert_array_equal(rv, rv_df, rtol=1e-10,
                           err_msg="Realized variance differs between arrays and DataFrame")

    def test_realized_variance_annualization(self, high_frequency_data):
        """Test realized variance annualization."""
        # Extract returns from high-frequency data
        prices = high_frequency_data['price'].values
        returns = np.diff(np.log(prices))

        # Calculate realized variance without annualization
        rv = variance(returns, annualize=False)

        # Calculate realized variance with annualization
        rv_annual = variance(returns, annualize=True, annualization_factor=252)

        # Check that annualized value is scaled correctly
        assert np.isclose(rv_annual, rv * 252), "Annualization scaling incorrect"

        # Test with custom annualization factor
        rv_custom = variance(returns, annualize=True, annualization_factor=100)
        assert np.isclose(rv_custom, rv * 100), "Custom annualization scaling incorrect"

    def test_realized_variance_sampling(self, high_frequency_data):
        """Test realized variance with different sampling schemes."""
        # Extract prices and times from high-frequency data
        prices = high_frequency_data['price']
        times = high_frequency_data['time']

        # Calculate realized variance with calendar time sampling
        rv_calendar = variance(prices, times=times, use_prices=True,
                               sampling_type='calendar', sampling_interval=300)  # 5-minute intervals

        # Calculate realized variance with business time sampling
        rv_business = variance(prices, times=times, use_prices=True,
                               sampling_type='business', sampling_points=48)  # 48 points per day

        # Calculate realized variance with fixed interval sampling
        rv_fixed = variance(prices, times=times, use_prices=True,
                            sampling_type='fixed', sampling_interval=100)  # Every 100 observations

        # Basic checks
        assert rv_calendar > 0, "Calendar time realized variance should be positive"
        assert rv_business > 0, "Business time realized variance should be positive"
        assert rv_fixed > 0, "Fixed interval realized variance should be positive"

        # Different sampling methods should give different results
        assert not np.isclose(rv_calendar, rv_business), "Calendar and business sampling gave same result"
        assert not np.isclose(rv_calendar, rv_fixed), "Calendar and fixed sampling gave same result"
        assert not np.isclose(rv_business, rv_fixed), "Business and fixed sampling gave same result"

    def test_realized_variance_pandas_integration(self, high_frequency_data):
        """Test realized variance with pandas DatetimeIndex."""
        # Create pandas DataFrame with DatetimeIndex
        df = high_frequency_data.copy()
        df.set_index('time', inplace=True)

        # Calculate realized variance with DatetimeIndex
        rv = variance(df['price'], use_prices=True, sampling_type='calendar', sampling_interval=300)

        # Basic checks
        assert isinstance(rv, float), "Realized variance should be a float"
        assert rv > 0, "Realized variance should be positive"

        # Test with pandas Series with DatetimeIndex
        series = df['price']
        rv_series = variance(series, use_prices=True, sampling_type='calendar', sampling_interval=300)
        assert np.isclose(rv, rv_series), "Realized variance differs between DataFrame and Series"

    def test_realized_variance_numba_acceleration(self, high_frequency_data):
        """Test Numba-accelerated realized variance calculation."""
        # Extract returns from high-frequency data
        prices = high_frequency_data['price'].values
        returns = np.diff(np.log(prices))

        # Calculate realized variance using the main function
        rv = variance(returns)

        # Calculate realized variance using the Numba-accelerated core function
        rv_numba = realized_variance_core(returns)

        # Results should be identical
        assert np.isclose(rv, rv_numba), "Numba-accelerated calculation differs from main function"

        # Test performance (should be fast even with large arrays)
        import time

        # Create a large array of returns
        large_returns = np.random.normal(0, 0.01, 100000)

        # Measure time for Numba-accelerated calculation
        start = time.time()
        _ = realized_variance_core(large_returns)
        numba_time = time.time() - start

        # Should complete quickly (typically < 0.01 seconds)
        assert numba_time < 0.1, "Numba-accelerated calculation is slower than expected"

    def test_realized_variance_error_handling(self):
        """Test error handling in realized variance calculation."""
        # Test with empty array
        with pytest.raises(ValueError, match="Input array must contain data"):
            variance(np.array([]))

        # Test with NaN values
        with pytest.raises(ValueError, match="Input array contains NaN values"):
            variance(np.array([1.0, 2.0, np.nan, 4.0]))

        # Test with infinite values
        with pytest.raises(ValueError, match="Input array contains infinite values"):
            variance(np.array([1.0, 2.0, np.inf, 4.0]))

        # Test with invalid sampling type
        with pytest.raises(ValueError, match="Unknown sampling type"):
            variance(np.array([1.0, 2.0, 3.0]), sampling_type='invalid')

        # Test with times but no sampling interval
        with pytest.raises(ValueError, match="Sampling interval must be provided"):
            variance(np.array([1.0, 2.0, 3.0]), times=np.array([1, 2, 3]),
                     sampling_type='calendar', sampling_interval=None)

        # Test with inconsistent array lengths
        with pytest.raises(ValueError, match="Prices and times must have the same length"):
            variance(np.array([1.0, 2.0, 3.0]), times=np.array([1, 2]), use_prices=True)

    @given(arrays(dtype=np.float64, shape=st.integers(10, 1000),
                  elements=st.floats(min_value=0.0001, max_value=1000, allow_nan=False, allow_infinity=False)))
    def test_property_based_realized_variance(self, prices):
        """Property-based test for realized variance using hypothesis."""
        # Calculate log returns
        log_prices = np.log(prices)
        returns = np.diff(log_prices)

        # Calculate realized variance
        rv = variance(returns)

        # Properties that should always hold
        assert rv >= 0, "Realized variance should be non-negative"
        assert np.isclose(rv, np.sum(returns**2)), "Realized variance should equal sum of squared returns"

        # Test with prices directly
        rv_prices = variance(prices, use_prices=True)
        assert np.isclose(rv, rv_prices), "Realized variance from prices should match returns calculation"

    @pytest.mark.asyncio
    async def test_async_realized_variance(self, high_frequency_data):
        """Test asynchronous realized variance calculation."""
        # Extract returns from high-frequency data
        prices = high_frequency_data['price'].values
        returns = np.diff(np.log(prices))

        # Define progress callback
        progress_updates = []

        async def progress_callback(percent, message):
            progress_updates.append((percent, message))

        # Calculate realized variance asynchronously
        rv = await variance(returns, return_async=True, progress_callback=progress_callback)

        # Basic checks
        assert isinstance(rv, float), "Realized variance should be a float"
        assert rv > 0, "Realized variance should be positive"

        # Check that progress callback was called
        assert len(progress_updates) > 0, "Progress callback was not called"
        assert progress_updates[-1][0] == 100.0, "Final progress update not 100%"


class TestBipowerVariation:
    """Tests for bipower variation estimators."""

    def test_bipower_variation_basic(self, high_frequency_data, assert_array_equal):
        """Test basic bipower variation calculation."""
        # Extract returns from high-frequency data
        prices = high_frequency_data['price'].values
        returns = np.diff(np.log(prices))

        # Calculate bipower variation
        bpv = bipower_variation(returns)

        # Basic checks
        assert isinstance(bpv, float), "Bipower variation should be a float"
        assert bpv > 0, "Bipower variation should be positive"

        # Calculate manually for comparison
        abs_returns = np.abs(returns)
        manual_bpv = (np.pi/2) * np.sum(abs_returns[1:] * abs_returns[:-1]) / (len(returns) - 1)
        assert_array_equal(bpv, manual_bpv, rtol=1e-10,
                           err_msg="Bipower variation doesn't match manual calculation")

        # Test with pandas Series
        returns_series = pd.Series(returns)
        bpv_series = bipower_variation(returns_series)
        assert isinstance(bpv_series, float), "Bipower variation should be a float even with Series input"
        assert_array_equal(bpv, bpv_series, rtol=1e-10,
                           err_msg="Bipower variation differs between ndarray and Series")

    def test_bipower_variation_with_timestamps(self, high_frequency_data, assert_array_equal):
        """Test bipower variation with timestamp information."""
        # Extract prices and times from high-frequency data
        prices = high_frequency_data['price'].values
        times = high_frequency_data['time'].values

        # Calculate bipower variation with timestamps
        bpv = bipower_variation(prices, times=times, use_prices=True)

        # Basic checks
        assert isinstance(bpv, float), "Bipower variation should be a float"
        assert bpv > 0, "Bipower variation should be positive"

        # Test with pandas DataFrame
        bpv_df = bipower_variation(high_frequency_data['price'],
                                   times=high_frequency_data['time'],
                                   use_prices=True)
        assert isinstance(bpv_df, float), "Bipower variation should be a float with DataFrame input"
        assert_array_equal(bpv, bpv_df, rtol=1e-10,
                           err_msg="Bipower variation differs between arrays and DataFrame")

    def test_bipower_variation_sampling(self, high_frequency_data):
        """Test bipower variation with different sampling schemes."""
        # Extract prices and times from high-frequency data
        prices = high_frequency_data['price']
        times = high_frequency_data['time']

        # Calculate bipower variation with calendar time sampling
        bpv_calendar = bipower_variation(prices, times=times, use_prices=True,
                                         sampling_type='calendar', sampling_interval=300)  # 5-minute intervals

        # Calculate bipower variation with business time sampling
        bpv_business = bipower_variation(prices, times=times, use_prices=True,
                                         sampling_type='business', sampling_points=48)  # 48 points per day

        # Calculate bipower variation with fixed interval sampling
        bpv_fixed = bipower_variation(prices, times=times, use_prices=True,
                                      sampling_type='fixed', sampling_interval=100)  # Every 100 observations

        # Basic checks
        assert bpv_calendar > 0, "Calendar time bipower variation should be positive"
        assert bpv_business > 0, "Business time bipower variation should be positive"
        assert bpv_fixed > 0, "Fixed interval bipower variation should be positive"

        # Different sampling methods should give different results
        assert not np.isclose(bpv_calendar, bpv_business), "Calendar and business sampling gave same result"
        assert not np.isclose(bpv_calendar, bpv_fixed), "Calendar and fixed sampling gave same result"
        assert not np.isclose(bpv_business, bpv_fixed), "Business and fixed sampling gave same result"

    def test_bipower_variation_numba_acceleration(self, high_frequency_data):
        """Test Numba-accelerated bipower variation calculation."""
        # Extract returns from high-frequency data
        prices = high_frequency_data['price'].values
        returns = np.diff(np.log(prices))

        # Calculate bipower variation using the main function
        bpv = bipower_variation(returns)

        # Calculate bipower variation using the Numba-accelerated core function
        bpv_numba = realized_bipower_variation_core(returns)

        # Results should be identical
        assert np.isclose(bpv, bpv_numba), "Numba-accelerated calculation differs from main function"

        # Test performance (should be fast even with large arrays)
        import time

        # Create a large array of returns
        large_returns = np.random.normal(0, 0.01, 100000)

        # Measure time for Numba-accelerated calculation
        start = time.time()
        _ = realized_bipower_variation_core(large_returns)
        numba_time = time.time() - start

        # Should complete quickly (typically < 0.01 seconds)
        assert numba_time < 0.1, "Numba-accelerated calculation is slower than expected"

    def test_bipower_variation_jump_robustness(self, high_frequency_data):
        """Test bipower variation robustness to jumps."""
        # Extract returns from high-frequency data
        prices = high_frequency_data['price'].values
        returns = np.diff(np.log(prices))

        # Calculate realized variance and bipower variation
        rv = variance(returns)
        bpv = bipower_variation(returns)

        # Add a single large jump
        jump_returns = returns.copy()
        jump_returns[len(jump_returns) // 2] += 0.05  # Add a 5% jump

        # Calculate realized variance and bipower variation with jump
        rv_jump = variance(jump_returns)
        bpv_jump = bipower_variation(jump_returns)

        # Realized variance should increase significantly with jump
        assert rv_jump > 1.5 * rv, "Realized variance not sensitive enough to jump"

        # Bipower variation should be less affected by jump
        assert bpv_jump < 1.5 * bpv, "Bipower variation too sensitive to jump"

        # Jump ratio should be significantly above 1
        jump_ratio = rv_jump / bpv_jump
        assert jump_ratio > 1.2, "Jump ratio not high enough for data with jump"

    @given(arrays(dtype=np.float64, shape=st.integers(10, 1000),
                  elements=st.floats(min_value=0.0001, max_value=1000, allow_nan=False, allow_infinity=False)))
    def test_property_based_bipower_variation(self, prices):
        """Property-based test for bipower variation using hypothesis."""
        # Calculate log returns
        log_prices = np.log(prices)
        returns = np.diff(log_prices)

        # Calculate bipower variation
        bpv = bipower_variation(returns)

        # Properties that should always hold
        assert bpv >= 0, "Bipower variation should be non-negative"

        # Calculate realized variance for comparison
        rv = variance(returns)

        # For normal returns without jumps, BPV should be close to RV
        # but generally slightly smaller
        assert bpv <= rv * 1.1, "Bipower variation should not be much larger than realized variance"

    @pytest.mark.asyncio
    async def test_async_bipower_variation(self, high_frequency_data):
        """Test asynchronous bipower variation calculation."""
        # Extract returns from high-frequency data
        prices = high_frequency_data['price'].values
        returns = np.diff(np.log(prices))

        # Define progress callback
        progress_updates = []

        async def progress_callback(percent, message):
            progress_updates.append((percent, message))

        # Calculate bipower variation asynchronously
        bpv = await bipower_variation(returns, return_async=True, progress_callback=progress_callback)

        # Basic checks
        assert isinstance(bpv, float), "Bipower variation should be a float"
        assert bpv > 0, "Bipower variation should be positive"

        # Check that progress callback was called
        assert len(progress_updates) > 0, "Progress callback was not called"
        assert progress_updates[-1][0] == 100.0, "Final progress update not 100%"


class TestRealizedKernel:
    """Tests for realized kernel estimators."""

    def test_realized_kernel_basic(self, high_frequency_data, assert_array_equal):
        """Test basic realized kernel calculation."""
        # Extract returns from high-frequency data
        prices = high_frequency_data['price'].values
        returns = np.diff(np.log(prices))

        # Calculate realized kernel
        rk = kernel(returns)

        # Basic checks
        assert isinstance(rk, float), "Realized kernel should be a float"
        assert rk > 0, "Realized kernel should be positive"

        # Test with pandas Series
        returns_series = pd.Series(returns)
        rk_series = kernel(returns_series)
        assert isinstance(rk_series, float), "Realized kernel should be a float even with Series input"
        assert_array_equal(rk, rk_series, rtol=1e-10,
                           err_msg="Realized kernel differs between ndarray and Series")

    def test_realized_kernel_with_timestamps(self, high_frequency_data, assert_array_equal):
        """Test realized kernel with timestamp information."""
        # Extract prices and times from high-frequency data
        prices = high_frequency_data['price'].values
        times = high_frequency_data['time'].values

        # Calculate realized kernel with timestamps
        rk = kernel(prices, times=times, use_prices=True)

        # Basic checks
        assert isinstance(rk, float), "Realized kernel should be a float"
        assert rk > 0, "Realized kernel should be positive"

        # Test with pandas DataFrame
        rk_df = kernel(high_frequency_data['price'],
                       times=high_frequency_data['time'],
                       use_prices=True)
        assert isinstance(rk_df, float), "Realized kernel should be a float with DataFrame input"
        assert_array_equal(rk, rk_df, rtol=1e-10,
                           err_msg="Realized kernel differs between arrays and DataFrame")

    def test_realized_kernel_bandwidth(self, high_frequency_data):
        """Test realized kernel with different bandwidth selection methods."""
        # Extract returns from high-frequency data
        prices = high_frequency_data['price'].values
        returns = np.diff(np.log(prices))

        # Calculate realized kernel with different bandwidth methods
        rk_default = kernel(returns)
        rk_optimal = kernel(returns, bandwidth_method='optimal')
        rk_custom = kernel(returns, bandwidth=20)

        # Basic checks
        assert rk_default > 0, "Default bandwidth realized kernel should be positive"
        assert rk_optimal > 0, "Optimal bandwidth realized kernel should be positive"
        assert rk_custom > 0, "Custom bandwidth realized kernel should be positive"

        # Different bandwidth methods should give different results
        assert not np.isclose(rk_default, rk_custom), "Default and custom bandwidth gave same result"

    def test_realized_kernel_weights(self, high_frequency_data):
        """Test realized kernel with different weight functions."""
        # Extract returns from high-frequency data
        prices = high_frequency_data['price'].values
        returns = np.diff(np.log(prices))

        # Calculate realized kernel with different weight functions
        rk_parzen = kernel(returns, kernel_type='parzen')
        rk_bartlett = kernel(returns, kernel_type='bartlett')
        rk_fejer = kernel(returns, kernel_type='fejer')
        rk_qs = kernel(returns, kernel_type='qs')
        rk_tukey_hanning = kernel(returns, kernel_type='tukey-hanning')

        # Basic checks
        assert rk_parzen > 0, "Parzen kernel should be positive"
        assert rk_bartlett > 0, "Bartlett kernel should be positive"
        assert rk_fejer > 0, "Fejer kernel should be positive"
        assert rk_qs > 0, "QS kernel should be positive"
        assert rk_tukey_hanning > 0, "Tukey-Hanning kernel should be positive"

        # Different kernel types should give different results
        kernels = [rk_parzen, rk_bartlett, rk_fejer, rk_qs, rk_tukey_hanning]
        for i in range(len(kernels)):
            for j in range(i+1, len(kernels)):
                assert not np.isclose(kernels[i], kernels[j]), f"Kernel types {i} and {j} gave same result"

    def test_realized_kernel_jitter(self, high_frequency_data):
        """Test realized kernel with jittering."""
        # Extract returns from high-frequency data
        prices = high_frequency_data['price'].values
        returns = np.diff(np.log(prices))

        # Calculate realized kernel with and without jittering
        rk_no_jitter = kernel(returns, jitter=False)
        rk_jitter = kernel(returns, jitter=True)

        # Basic checks
        assert rk_no_jitter > 0, "Non-jittered kernel should be positive"
        assert rk_jitter > 0, "Jittered kernel should be positive"

        # Jittering should affect the result
        assert not np.isclose(rk_no_jitter, rk_jitter), "Jittering had no effect on kernel"

    def test_realized_kernel_noise_robustness(self, high_frequency_data):
        """Test realized kernel robustness to microstructure noise."""
        # Extract returns from high-frequency data
        prices = high_frequency_data['price'].values
        returns = np.diff(np.log(prices))

        # Calculate realized variance and realized kernel
        rv = variance(returns)
        rk = kernel(returns)

        # Add microstructure noise
        noise_level = 0.0005  # 5 basis points
        noise = np.random.normal(0, noise_level, len(prices))
        noisy_prices = prices * np.exp(noise)
        noisy_returns = np.diff(np.log(noisy_prices))

        # Calculate realized variance and realized kernel with noise
        rv_noise = variance(noisy_returns)
        rk_noise = kernel(noisy_returns)

        # Realized variance should increase significantly with noise
        assert rv_noise > rv * 1.2, "Realized variance not sensitive enough to noise"

        # Realized kernel should be less affected by noise
        assert rk_noise < rv_noise * 0.9, "Realized kernel not robust enough to noise"

        # Noise ratio should be significantly above 1
        noise_ratio = rv_noise / rk_noise
        assert noise_ratio > 1.1, "Noise ratio not high enough for data with microstructure noise"

    @pytest.mark.asyncio
    async def test_async_realized_kernel(self, high_frequency_data):
        """Test asynchronous realized kernel calculation."""
        # Extract returns from high-frequency data
        prices = high_frequency_data['price'].values
        returns = np.diff(np.log(prices))

        # Define progress callback
        progress_updates = []

        async def progress_callback(percent, message):
            progress_updates.append((percent, message))

        # Calculate realized kernel asynchronously
        rk = await kernel(returns, return_async=True, progress_callback=progress_callback)

        # Basic checks
        assert isinstance(rk, float), "Realized kernel should be a float"
        assert rk > 0, "Realized kernel should be positive"

        # Check that progress callback was called
        assert len(progress_updates) > 0, "Progress callback was not called"
        assert progress_updates[-1][0] == 100.0, "Final progress update not 100%"


class TestMultiscaleEstimators:
    """Tests for multiscale and two-scale realized variance estimators."""

    def test_twoscale_variance_basic(self, high_frequency_data, assert_array_equal):
        """Test basic two-scale realized variance calculation."""
        # Extract prices from high-frequency data
        prices = high_frequency_data['price'].values

        # Calculate two-scale realized variance
        tsrv = twoscale_variance(prices)

        # Basic checks
        assert isinstance(tsrv, float), "Two-scale realized variance should be a float"
        assert tsrv > 0, "Two-scale realized variance should be positive"

        # Test with pandas Series
        prices_series = pd.Series(prices)
        tsrv_series = twoscale_variance(prices_series)
        assert isinstance(tsrv_series, float), "Two-scale realized variance should be a float even with Series input"
        assert_array_equal(tsrv, tsrv_series, rtol=1e-10,
                           err_msg="Two-scale realized variance differs between ndarray and Series")

    def test_multiscale_variance_basic(self, high_frequency_data, assert_array_equal):
        """Test basic multiscale realized variance calculation."""
        # Extract prices from high-frequency data
        prices = high_frequency_data['price'].values

        # Calculate multiscale realized variance
        msrv = multiscale_variance(prices)

        # Basic checks
        assert isinstance(msrv, float), "Multiscale realized variance should be a float"
        assert msrv > 0, "Multiscale realized variance should be positive"

        # Test with pandas Series
        prices_series = pd.Series(prices)
        msrv_series = multiscale_variance(prices_series)
        assert isinstance(msrv_series, float), "Multiscale realized variance should be a float even with Series input"
        assert_array_equal(msrv, msrv_series, rtol=1e-10,
                           err_msg="Multiscale realized variance differs between ndarray and Series")

    def test_multiscale_variance_scales(self, high_frequency_data):
        """Test multiscale realized variance with different scales."""
        # Extract prices from high-frequency data
        prices = high_frequency_data['price'].values

        # Calculate multiscale realized variance with different scales
        msrv_default = multiscale_variance(prices)
        msrv_custom = multiscale_variance(prices, scales=[1, 5, 10, 20, 50])

        # Basic checks
        assert msrv_default > 0, "Default scales multiscale realized variance should be positive"
        assert msrv_custom > 0, "Custom scales multiscale realized variance should be positive"

        # Different scales should give different results
        assert not np.isclose(msrv_default, msrv_custom), "Default and custom scales gave same result"

    def test_noise_robustness_comparison(self, high_frequency_data):
        """Test noise robustness of different estimators."""
        # Extract prices from high-frequency data
        prices = high_frequency_data['price'].values
        returns = np.diff(np.log(prices))

        # Calculate different estimators
        rv = variance(returns)
        rk = kernel(returns)
        tsrv = twoscale_variance(prices)
        msrv = multiscale_variance(prices)

        # Add microstructure noise
        noise_level = 0.0005  # 5 basis points
        noise = np.random.normal(0, noise_level, len(prices))
        noisy_prices = prices * np.exp(noise)
        noisy_returns = np.diff(np.log(noisy_prices))

        # Calculate estimators with noise
        rv_noise = variance(noisy_returns)
        rk_noise = kernel(noisy_returns)
        tsrv_noise = twoscale_variance(noisy_prices)
        msrv_noise = multiscale_variance(noisy_prices)

        # Calculate noise ratios
        rv_ratio = rv_noise / rv
        rk_ratio = rk_noise / rk
        tsrv_ratio = tsrv_noise / tsrv
        msrv_ratio = msrv_noise / msrv

        # Noise-robust estimators should have ratios closer to 1
        assert rv_ratio > rk_ratio, "Realized kernel not more robust than realized variance"
        assert rv_ratio > tsrv_ratio, "Two-scale not more robust than realized variance"
        assert rv_ratio > msrv_ratio, "Multiscale not more robust than realized variance"

    @pytest.mark.asyncio
    async def test_async_multiscale_variance(self, high_frequency_data):
        """Test asynchronous multiscale realized variance calculation."""
        # Extract prices from high-frequency data
        prices = high_frequency_data['price'].values

        # Define progress callback
        progress_updates = []

        async def progress_callback(percent, message):
            progress_updates.append((percent, message))

        # Calculate multiscale realized variance asynchronously
        msrv = await multiscale_variance(prices, return_async=True, progress_callback=progress_callback)

        # Basic checks
        assert isinstance(msrv, float), "Multiscale realized variance should be a float"
        assert msrv > 0, "Multiscale realized variance should be positive"

        # Check that progress callback was called
        assert len(progress_updates) > 0, "Progress callback was not called"
        assert progress_updates[-1][0] == 100.0, "Final progress update not 100%"


class TestQMLEVariance:
    """Tests for QMLE variance estimator."""

    def test_qmle_variance_basic(self, high_frequency_data, assert_array_equal):
        """Test basic QMLE variance calculation."""
        # Extract prices from high-frequency data
        prices = high_frequency_data['price'].values

        # Calculate QMLE variance
        qmle = qmle_variance(prices)

        # Basic checks
        assert isinstance(qmle, float), "QMLE variance should be a float"
        assert qmle > 0, "QMLE variance should be positive"

        # Test with pandas Series
        prices_series = pd.Series(prices)
        qmle_series = qmle_variance(prices_series)
        assert isinstance(qmle_series, float), "QMLE variance should be a float even with Series input"
        assert_array_equal(qmle, qmle_series, rtol=1e-10,
                           err_msg="QMLE variance differs between ndarray and Series")

    def test_qmle_variance_iterations(self, high_frequency_data):
        """Test QMLE variance with different iteration settings."""
        # Extract prices from high-frequency data
        prices = high_frequency_data['price'].values

        # Calculate QMLE variance with different iteration settings
        qmle_default = qmle_variance(prices)
        qmle_more_iter = qmle_variance(prices, max_iterations=20, tolerance=1e-8)

        # Basic checks
        assert qmle_default > 0, "Default QMLE variance should be positive"
        assert qmle_more_iter > 0, "QMLE variance with more iterations should be positive"

        # Results should be similar but not identical
        assert np.isclose(qmle_default, qmle_more_iter,
                          rtol=0.1), "QMLE results differ too much with different iterations"
        assert qmle_default != qmle_more_iter, "QMLE results identical with different iterations"

    def test_qmle_variance_noise_robustness(self, high_frequency_data):
        """Test QMLE variance robustness to microstructure noise."""
        # Extract prices from high-frequency data
        prices = high_frequency_data['price'].values
        returns = np.diff(np.log(prices))

        # Calculate realized variance and QMLE variance
        rv = variance(returns)
        qmle = qmle_variance(prices)

        # Add microstructure noise
        noise_level = 0.0005  # 5 basis points
        noise = np.random.normal(0, noise_level, len(prices))
        noisy_prices = prices * np.exp(noise)
        noisy_returns = np.diff(np.log(noisy_prices))

        # Calculate realized variance and QMLE variance with noise
        rv_noise = variance(noisy_returns)
        qmle_noise = qmle_variance(noisy_prices)

        # Realized variance should increase significantly with noise
        assert rv_noise > rv * 1.2, "Realized variance not sensitive enough to noise"

        # QMLE variance should be less affected by noise
        assert qmle_noise < rv_noise * 0.9, "QMLE variance not robust enough to noise"

        # Noise ratio should be significantly above 1
        noise_ratio = rv_noise / qmle_noise
        assert noise_ratio > 1.1, "Noise ratio not high enough for data with microstructure noise"

    @pytest.mark.asyncio
    async def test_async_qmle_variance(self, high_frequency_data):
        """Test asynchronous QMLE variance calculation."""
        # Extract prices from high-frequency data
        prices = high_frequency_data['price'].values

        # Define progress callback
        progress_updates = []

        async def progress_callback(percent, message):
            progress_updates.append((percent, message))

        # Calculate QMLE variance asynchronously
        qmle = await qmle_variance(prices, return_async=True, progress_callback=progress_callback)

        # Basic checks
        assert isinstance(qmle, float), "QMLE variance should be a float"
        assert qmle > 0, "QMLE variance should be positive"

        # Check that progress callback was called
        assert len(progress_updates) > 0, "Progress callback was not called"
        assert progress_updates[-1][0] == 100.0, "Final progress update not 100%"


class TestPreaveragedEstimators:
    """Tests for preaveraged variance and bipower variation estimators."""

    def test_preaveraged_variance_basic(self, high_frequency_data, assert_array_equal):
        """Test basic preaveraged variance calculation."""
        # Extract prices from high-frequency data
        prices = high_frequency_data['price'].values

        # Calculate preaveraged variance
        pav = preaveraged_variance(prices)

        # Basic checks
        assert isinstance(pav, float), "Preaveraged variance should be a float"
        assert pav > 0, "Preaveraged variance should be positive"

        # Test with pandas Series
        prices_series = pd.Series(prices)
        pav_series = preaveraged_variance(prices_series)
        assert isinstance(pav_series, float), "Preaveraged variance should be a float even with Series input"
        assert_array_equal(pav, pav_series, rtol=1e-10,
                           err_msg="Preaveraged variance differs between ndarray and Series")

    def test_preaveraged_bipower_variation_basic(self, high_frequency_data, assert_array_equal):
        """Test basic preaveraged bipower variation calculation."""
        # Extract prices from high-frequency data
        prices = high_frequency_data['price'].values

        # Calculate preaveraged bipower variation
        pbv = preaveraged_bipower_variation(prices)

        # Basic checks
        assert isinstance(pbv, float), "Preaveraged bipower variation should be a float"
        assert pbv > 0, "Preaveraged bipower variation should be positive"

        # Test with pandas Series
        prices_series = pd.Series(prices)
        pbv_series = preaveraged_bipower_variation(prices_series)
        assert isinstance(pbv_series, float), "Preaveraged bipower variation should be a float even with Series input"
        assert_array_equal(pbv, pbv_series, rtol=1e-10,
                           err_msg="Preaveraged bipower variation differs between ndarray and Series")

    def test_preaveraged_window_size(self, high_frequency_data):
        """Test preaveraged estimators with different window sizes."""
        # Extract prices from high-frequency data
        prices = high_frequency_data['price'].values

        # Calculate preaveraged variance with different window sizes
        pav_default = preaveraged_variance(prices)
        pav_small = preaveraged_variance(prices, window_size=10)
        pav_large = preaveraged_variance(prices, window_size=50)

        # Basic checks
        assert pav_default > 0, "Default window preaveraged variance should be positive"
        assert pav_small > 0, "Small window preaveraged variance should be positive"
        assert pav_large > 0, "Large window preaveraged variance should be positive"

        # Different window sizes should give different results
        assert not np.isclose(pav_small, pav_large), "Small and large windows gave same result"

    def test_preaveraged_noise_robustness(self, high_frequency_data):
        """Test preaveraged estimators robustness to microstructure noise."""
        # Extract prices from high-frequency data
        prices = high_frequency_data['price'].values
        returns = np.diff(np.log(prices))

        # Calculate realized variance and preaveraged variance
        rv = variance(returns)
        pav = preaveraged_variance(prices)

        # Add microstructure noise
        noise_level = 0.0005  # 5 basis points
        noise = np.random.normal(0, noise_level, len(prices))
        noisy_prices = prices * np.exp(noise)
        noisy_returns = np.diff(np.log(noisy_prices))

        # Calculate realized variance and preaveraged variance with noise
        rv_noise = variance(noisy_returns)
        pav_noise = preaveraged_variance(noisy_prices)

        # Realized variance should increase significantly with noise
        assert rv_noise > rv * 1.2, "Realized variance not sensitive enough to noise"

        # Preaveraged variance should be less affected by noise
        assert pav_noise < rv_noise * 0.9, "Preaveraged variance not robust enough to noise"

        # Noise ratio should be significantly above 1
        noise_ratio = rv_noise / pav_noise
        assert noise_ratio > 1.1, "Noise ratio not high enough for data with microstructure noise"

    @pytest.mark.asyncio
    async def test_async_preaveraged_variance(self, high_frequency_data):
        """Test asynchronous preaveraged variance calculation."""
        # Extract prices from high-frequency data
        prices = high_frequency_data['price'].values

        # Define progress callback
        progress_updates = []

        async def progress_callback(percent, message):
            progress_updates.append((percent, message))

        # Calculate preaveraged variance asynchronously
        pav = await preaveraged_variance(prices, return_async=True, progress_callback=progress_callback)

        # Basic checks
        assert isinstance(pav, float), "Preaveraged variance should be a float"
        assert pav > 0, "Preaveraged variance should be positive"

        # Check that progress callback was called
        assert len(progress_updates) > 0, "Progress callback was not called"
        assert progress_updates[-1][0] == 100.0, "Final progress update not 100%"


class TestThresholdEstimators:
    """Tests for threshold variance and multipower variation estimators."""

    def test_threshold_variance_basic(self, high_frequency_data, assert_array_equal):
        """Test basic threshold variance calculation."""
        # Extract returns from high-frequency data
        prices = high_frequency_data['price'].values
        returns = np.diff(np.log(prices))

        # Calculate threshold variance
        tv = threshold_variance(returns)

        # Basic checks
        assert isinstance(tv, float), "Threshold variance should be a float"
        assert tv > 0, "Threshold variance should be positive"

        # Test with pandas Series
        returns_series = pd.Series(returns)
        tv_series = threshold_variance(returns_series)
        assert isinstance(tv_series, float), "Threshold variance should be a float even with Series input"
        assert_array_equal(tv, tv_series, rtol=1e-10,
                           err_msg="Threshold variance differs between ndarray and Series")

    def test_threshold_multipower_variation_basic(self, high_frequency_data, assert_array_equal):
        """Test basic threshold multipower variation calculation."""
        # Extract returns from high-frequency data
        prices = high_frequency_data['price'].values
        returns = np.diff(np.log(prices))

        # Calculate threshold multipower variation
        tmpv = threshold_multipower_variation(returns)

        # Basic checks
        assert isinstance(tmpv, float), "Threshold multipower variation should be a float"
        assert tmpv > 0, "Threshold multipower variation should be positive"

        # Test with pandas Series
        returns_series = pd.Series(returns)
        tmpv_series = threshold_multipower_variation(returns_series)
        assert isinstance(tmpv_series, float), "Threshold multipower variation should be a float even with Series input"
        assert_array_equal(tmpv, tmpv_series, rtol=1e-10,
                           err_msg="Threshold multipower variation differs between ndarray and Series")

    def test_threshold_parameter(self, high_frequency_data):
        """Test threshold estimators with different threshold parameters."""
        # Extract returns from high-frequency data
        prices = high_frequency_data['price'].values
        returns = np.diff(np.log(prices))

        # Calculate threshold variance with different threshold parameters
        tv_default = threshold_variance(returns)
        tv_small = threshold_variance(returns, c=2.0)
        tv_large = threshold_variance(returns, c=5.0)

        # Basic checks
        assert tv_default > 0, "Default threshold variance should be positive"
        assert tv_small > 0, "Small threshold variance should be positive"
        assert tv_large > 0, "Large threshold variance should be positive"

        # Different threshold parameters should give different results
        assert not np.isclose(tv_small, tv_large), "Small and large thresholds gave same result"

    def test_threshold_jump_robustness(self, high_frequency_data):
        """Test threshold estimators robustness to jumps."""
        # Extract returns from high-frequency data
        prices = high_frequency_data['price'].values
        returns = np.diff(np.log(prices))

        # Calculate realized variance and threshold variance
        rv = variance(returns)
        tv = threshold_variance(returns)

        # Add a single large jump
        jump_returns = returns.copy()
        jump_returns[len(jump_returns) // 2] += 0.05  # Add a 5% jump

        # Calculate realized variance and threshold variance with jump
        rv_jump = variance(jump_returns)
        tv_jump = threshold_variance(jump_returns)

        # Realized variance should increase significantly with jump
        assert rv_jump > 1.5 * rv, "Realized variance not sensitive enough to jump"

        # Threshold variance should be less affected by jump
        assert tv_jump < 1.5 * tv, "Threshold variance too sensitive to jump"

        # Jump ratio should be significantly above 1
        jump_ratio = rv_jump / tv_jump
        assert jump_ratio > 1.2, "Jump ratio not high enough for data with jump"

    def test_threshold_numba_acceleration(self, high_frequency_data):
        """Test Numba-accelerated threshold variance calculation."""
        # Extract returns from high-frequency data
        prices = high_frequency_data['price'].values
        returns = np.diff(np.log(prices))

        # Calculate threshold variance using the main function
        tv = threshold_variance(returns)

        # Calculate threshold variance using the Numba-accelerated core function
        c = 3.0  # Default threshold parameter
        local_variance = np.median(returns**2) * 1.5  # Approximation of local variance
        threshold = c * np.sqrt(local_variance)
        tv_numba = realized_threshold_variance_core(returns, threshold)

        # Results should be close (not identical due to local variance estimation)
        assert np.isclose(tv, tv_numba, rtol=0.1), "Numba-accelerated calculation differs too much from main function"

    @pytest.mark.asyncio
    async def test_async_threshold_variance(self, high_frequency_data):
        """Test asynchronous threshold variance calculation."""
        # Extract returns from high-frequency data
        prices = high_frequency_data['price'].values
        returns = np.diff(np.log(prices))

        # Define progress callback
        progress_updates = []

        async def progress_callback(percent, message):
            progress_updates.append((percent, message))

        # Calculate threshold variance asynchronously
        tv = await threshold_variance(returns, return_async=True, progress_callback=progress_callback)

        # Basic checks
        assert isinstance(tv, float), "Threshold variance should be a float"
        assert tv > 0, "Threshold variance should be positive"

        # Check that progress callback was called
        assert len(progress_updates) > 0, "Progress callback was not called"
        assert progress_updates[-1][0] == 100.0, "Final progress update not 100%"


class TestRealizedRange:
    """Tests for realized range estimators."""

    def test_realized_range_basic(self, high_frequency_data, assert_array_equal):
        """Test basic realized range calculation."""
        # Create high/low/close data
        n = len(high_frequency_data)
        sampling_points = 48  # 48 points per day (30-minute intervals)
        samples_per_point = n // sampling_points

        high = np.zeros(sampling_points)
        low = np.zeros(sampling_points)
        close = np.zeros(sampling_points)

        for i in range(sampling_points):
            start_idx = i * samples_per_point
            end_idx = min((i + 1) * samples_per_point, n)
            if end_idx > start_idx:
                high[i] = np.max(high_frequency_data['price'].values[start_idx:end_idx])
                low[i] = np.min(high_frequency_data['price'].values[start_idx:end_idx])
                close[i] = high_frequency_data['price'].values[end_idx - 1]

        # Calculate realized range
        rr = rv_range(high, low)

        # Basic checks
        assert isinstance(rr, float), "Realized range should be a float"
        assert rr > 0, "Realized range should be positive"

        # Test with pandas Series
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        rr_series = rv_range(high_series, low_series)
        assert isinstance(rr_series, float), "Realized range should be a float even with Series input"
        assert_array_equal(rr, rr_series, rtol=1e-10,
                           err_msg="Realized range differs between ndarray and Series")

    def test_realized_range_with_close(self, high_frequency_data, assert_array_equal):
        """Test realized range calculation with close prices."""
        # Create high/low/close data
        n = len(high_frequency_data)
        sampling_points = 48  # 48 points per day (30-minute intervals)
        samples_per_point = n // sampling_points

        high = np.zeros(sampling_points)
        low = np.zeros(sampling_points)
        close = np.zeros(sampling_points)

        for i in range(sampling_points):
            start_idx = i * samples_per_point
            end_idx = min((i + 1) * samples_per_point, n)
            if end_idx > start_idx:
                high[i] = np.max(high_frequency_data['price'].values[start_idx:end_idx])
                low[i] = np.min(high_frequency_data['price'].values[start_idx:end_idx])
                close[i] = high_frequency_data['price'].values[end_idx - 1]

        # Calculate realized range with close prices
        rr = rv_range(high, low, close=close)

        # Basic checks
        assert isinstance(rr, float), "Realized range should be a float"
        assert rr > 0, "Realized range should be positive"

        # Test with pandas Series
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        close_series = pd.Series(close)
        rr_series = rv_range(high_series, low_series, close=close_series)
        assert isinstance(rr_series, float), "Realized range should be a float even with Series input"
        assert_array_equal(rr, rr_series, rtol=1e-10,
                           err_msg="Realized range differs between ndarray and Series")

    def test_realized_range_scaling(self, high_frequency_data):
        """Test realized range with different scaling methods."""
        # Create high/low/close data
        n = len(high_frequency_data)
        sampling_points = 48  # 48 points per day (30-minute intervals)
        samples_per_point = n // sampling_points

        high = np.zeros(sampling_points)
        low = np.zeros(sampling_points)

        for i in range(sampling_points):
            start_idx = i * samples_per_point
            end_idx = min((i + 1) * samples_per_point, n)
            if end_idx > start_idx:
                high[i] = np.max(high_frequency_data['price'].values[start_idx:end_idx])
                low[i] = np.min(high_frequency_data['price'].values[start_idx:end_idx])

        # Calculate realized range with different scaling methods
        rr_default = rv_range(high, low)
        rr_parkinson = rv_range(high, low, scaling='parkinson')
        rr_rogers_satchell = rv_range(high, low, scaling='rogers-satchell',
                                      close=high_frequency_data['price'].values[-sampling_points:])

        # Basic checks
        assert rr_default > 0, "Default scaling realized range should be positive"
        assert rr_parkinson > 0, "Parkinson scaling realized range should be positive"
        assert rr_rogers_satchell > 0, "Rogers-Satchell scaling realized range should be positive"

        # Different scaling methods should give different results
        assert not np.isclose(rr_default, rr_parkinson), "Default and Parkinson scaling gave same result"
        assert not np.isclose(rr_default, rr_rogers_satchell), "Default and Rogers-Satchell scaling gave same result"
        assert not np.isclose(
            rr_parkinson, rr_rogers_satchell), "Parkinson and Rogers-Satchell scaling gave same result"

    @pytest.mark.asyncio
    async def test_async_realized_range(self, high_frequency_data):
        """Test asynchronous realized range calculation."""
        # Create high/low/close data
        n = len(high_frequency_data)
        sampling_points = 48  # 48 points per day (30-minute intervals)
        samples_per_point = n // sampling_points

        high = np.zeros(sampling_points)
        low = np.zeros(sampling_points)

        for i in range(sampling_points):
            start_idx = i * samples_per_point
            end_idx = min((i + 1) * samples_per_point, n)
            if end_idx > start_idx:
                high[i] = np.max(high_frequency_data['price'].values[start_idx:end_idx])
                low[i] = np.min(high_frequency_data['price'].values[start_idx:end_idx])

        # Define progress callback
        progress_updates = []

        async def progress_callback(percent, message):
            progress_updates.append((percent, message))

        # Calculate realized range asynchronously
        rr = await rv_range(high, low, return_async=True, progress_callback=progress_callback)

        # Basic checks
        assert isinstance(rr, float), "Realized range should be a float"
        assert rr > 0, "Realized range should be positive"

        # Check that progress callback was called
        assert len(progress_updates) > 0, "Progress callback was not called"
        assert progress_updates[-1][0] == 100.0, "Final progress update not 100%"


class TestMultivariateEstimators:
    """Tests for multivariate realized volatility estimators."""

    def test_realized_covariance_basic(self, multivariate_correlated_series, assert_array_equal):
        """Test basic realized covariance calculation."""
        # Extract returns from multivariate series
        returns = np.diff(multivariate_correlated_series, axis=0)

        # Calculate realized covariance
        rcov = covariance(returns)

        # Basic checks
        assert isinstance(rcov, np.ndarray), "Realized covariance should be a numpy array"
        assert rcov.shape == (returns.shape[1], returns.shape[1]), "Realized covariance has incorrect shape"
        assert np.all(np.diag(rcov) > 0), "Diagonal elements of realized covariance should be positive"

        # Check that covariance matrix is symmetric
        assert np.allclose(rcov, rcov.T), "Realized covariance matrix should be symmetric"

        # Calculate manually for comparison
        manual_rcov = returns.T @ returns / len(returns)
        assert_array_equal(rcov, manual_rcov, rtol=1e-10,
                           err_msg="Realized covariance doesn't match manual calculation")

        # Test with pandas DataFrame
        returns_df = pd.DataFrame(returns)
        rcov_df = covariance(returns_df)
        assert isinstance(rcov_df, np.ndarray), "Realized covariance should be a numpy array even with DataFrame input"
        assert_array_equal(rcov, rcov_df, rtol=1e-10,
                           err_msg="Realized covariance differs between ndarray and DataFrame")

    def test_multivariate_kernel_basic(self, multivariate_correlated_series, assert_array_equal):
        """Test basic multivariate realized kernel calculation."""
        # Extract returns from multivariate series
        returns = np.diff(multivariate_correlated_series, axis=0)

        # Calculate multivariate realized kernel
        mrk = multivariate_kernel(returns)

        # Basic checks
        assert isinstance(mrk, np.ndarray), "Multivariate realized kernel should be a numpy array"
        assert mrk.shape == (returns.shape[1], returns.shape[1]), "Multivariate realized kernel has incorrect shape"
        assert np.all(np.diag(mrk) > 0), "Diagonal elements of multivariate realized kernel should be positive"

        # Check that kernel matrix is symmetric
        assert np.allclose(mrk, mrk.T), "Multivariate realized kernel matrix should be symmetric"

        # Test with pandas DataFrame
        returns_df = pd.DataFrame(returns)
        mrk_df = multivariate_kernel(returns_df)
        assert isinstance(
            mrk_df, np.ndarray), "Multivariate realized kernel should be a numpy array even with DataFrame input"
        assert_array_equal(mrk, mrk_df, rtol=1e-10,
                           err_msg="Multivariate realized kernel differs between ndarray and DataFrame")

    def test_multivariate_kernel_bandwidth(self, multivariate_correlated_series):
        """Test multivariate realized kernel with different bandwidth selection methods."""
        # Extract returns from multivariate series
        returns = np.diff(multivariate_correlated_series, axis=0)

        # Calculate multivariate realized kernel with different bandwidth methods
        mrk_default = multivariate_kernel(returns)
        mrk_optimal = multivariate_kernel(returns, bandwidth_method='optimal')
        mrk_custom = multivariate_kernel(returns, bandwidth=20)

        # Basic checks
        assert np.all(np.diag(mrk_default) >
                      0), "Default bandwidth multivariate realized kernel should have positive diagonal"
        assert np.all(np.diag(mrk_optimal) >
                      0), "Optimal bandwidth multivariate realized kernel should have positive diagonal"
        assert np.all(np.diag(mrk_custom) > 0), "Custom bandwidth multivariate realized kernel should have positive diagonal"

        # Different bandwidth methods should give different results
        assert not np.allclose(mrk_default, mrk_custom), "Default and custom bandwidth gave same result"

    def test_multivariate_kernel_weights(self, multivariate_correlated_series):
        """Test multivariate realized kernel with different weight functions."""
        # Extract returns from multivariate series
        returns = np.diff(multivariate_correlated_series, axis=0)

        # Calculate multivariate realized kernel with different weight functions
        mrk_parzen = multivariate_kernel(returns, kernel_type='parzen')
        mrk_bartlett = multivariate_kernel(returns, kernel_type='bartlett')
        mrk_qs = multivariate_kernel(returns, kernel_type='qs')

        # Basic checks
        assert np.all(np.diag(mrk_parzen) > 0), "Parzen kernel should have positive diagonal"
        assert np.all(np.diag(mrk_bartlett) > 0), "Bartlett kernel should have positive diagonal"
        assert np.all(np.diag(mrk_qs) > 0), "QS kernel should have positive diagonal"

        # Different kernel types should give different results
        kernels = [mrk_parzen, mrk_bartlett, mrk_qs]
        for i in range(len(kernels)):
            for j in range(i+1, len(kernels)):
                assert not np.allclose(kernels[i], kernels[j]), f"Kernel types {i} and {j} gave same result"

    def test_multivariate_noise_robustness(self, multivariate_correlated_series):
        """Test multivariate estimators robustness to microstructure noise."""
        # Extract returns from multivariate series
        returns = np.diff(multivariate_correlated_series, axis=0)

        # Calculate realized covariance and multivariate realized kernel
        rcov = covariance(returns)
        mrk = multivariate_kernel(returns)

        # Add microstructure noise
        noise_level = 0.0005  # 5 basis points
        noise = np.random.normal(0, noise_level, multivariate_correlated_series.shape)
        noisy_series = multivariate_correlated_series + noise
        noisy_returns = np.diff(noisy_series, axis=0)

        # Calculate realized covariance and multivariate realized kernel with noise
        rcov_noise = covariance(noisy_returns)
        mrk_noise = multivariate_kernel(noisy_returns)

        # Realized covariance should be more affected by noise than multivariate realized kernel
        # Compare trace of matrices
        rcov_trace = np.trace(rcov)
        rcov_noise_trace = np.trace(rcov_noise)
        mrk_trace = np.trace(mrk)
        mrk_noise_trace = np.trace(mrk_noise)

        rcov_ratio = rcov_noise_trace / rcov_trace
        mrk_ratio = mrk_noise_trace / mrk_trace

        assert rcov_ratio > mrk_ratio, "Multivariate realized kernel not more robust than realized covariance"

    @pytest.mark.asyncio
    async def test_async_multivariate_kernel(self, multivariate_correlated_series):
        """Test asynchronous multivariate realized kernel calculation."""
        # Extract returns from multivariate series
        returns = np.diff(multivariate_correlated_series, axis=0)

        # Define progress callback
        progress_updates = []

        async def progress_callback(percent, message):
            progress_updates.append((percent, message))

        # Calculate multivariate realized kernel asynchronously
        mrk = await multivariate_kernel(returns, return_async=True, progress_callback=progress_callback)

        # Basic checks
        assert isinstance(mrk, np.ndarray), "Multivariate realized kernel should be a numpy array"
        assert mrk.shape == (returns.shape[1], returns.shape[1]), "Multivariate realized kernel has incorrect shape"
        assert np.all(np.diag(mrk) > 0), "Diagonal elements of multivariate realized kernel should be positive"

        # Check that progress callback was called
        assert len(progress_updates) > 0, "Progress callback was not called"
        assert progress_updates[-1][0] == 100.0, "Final progress update not 100%"


class TestTimeConversionFunctions:
    """Tests for time conversion utility functions."""

    def test_seconds2unit_unit2seconds(self, assert_array_equal):
        """Test conversion between seconds and unit time."""
        # Create seconds array
        seconds = np.array([0, 3600, 7200, 10800, 14400])

        # Convert seconds to unit time
        unit_time = seconds2unit(seconds)

        # Basic checks
        assert isinstance(unit_time, np.ndarray), "Unit time should be a numpy array"
        assert unit_time.shape == seconds.shape, "Unit time has incorrect shape"
        assert np.all(unit_time >= 0) and np.all(unit_time <= 1), "Unit time should be between 0 and 1"

        # Convert back to seconds
        seconds_back = unit2seconds(unit_time)

        # Check round-trip conversion
        assert_array_equal(seconds, seconds_back, rtol=1e-10,
                           err_msg="Round-trip conversion between seconds and unit time failed")

        # Test with pandas Series
        seconds_series = pd.Series(seconds)
        unit_time_series = seconds2unit(seconds_series)
        assert isinstance(unit_time_series, np.ndarray), "Unit time should be a numpy array even with Series input"
        assert_array_equal(unit_time, unit_time_series, rtol=1e-10,
                           err_msg="seconds2unit differs between ndarray and Series")

    def test_seconds2wall_wall2seconds(self, assert_array_equal):
        """Test conversion between seconds and wall time."""
        # Create seconds array
        seconds = np.array([0, 3600, 7200, 10800, 14400])

        # Convert seconds to wall time
        wall_time = seconds2wall(seconds)

        # Basic checks
        assert isinstance(wall_time, np.ndarray), "Wall time should be a numpy array"
        assert wall_time.shape == seconds.shape, "Wall time has incorrect shape"

        # Convert back to seconds
        seconds_back = wall2seconds(wall_time)

        # Check round-trip conversion
        assert_array_equal(seconds, seconds_back, rtol=1e-10,
                           err_msg="Round-trip conversion between seconds and wall time failed")

        # Test with pandas Series
        seconds_series = pd.Series(seconds)
        wall_time_series = seconds2wall(seconds_series)
        assert isinstance(wall_time_series, np.ndarray), "Wall time should be a numpy array even with Series input"
        assert_array_equal(wall_time, wall_time_series, rtol=1e-10,
                           err_msg="seconds2wall differs between ndarray and Series")

    def test_unit2wall_wall2unit(self, assert_array_equal):
        """Test conversion between unit time and wall time."""
        # Create unit time array
        unit_time = np.array([0, 0.25, 0.5, 0.75, 1.0])

        # Convert unit time to wall time
        wall_time = unit2wall(unit_time)

        # Basic checks
        assert isinstance(wall_time, np.ndarray), "Wall time should be a numpy array"
        assert wall_time.shape == unit_time.shape, "Wall time has incorrect shape"

        # Convert back to unit time
        unit_time_back = wall2unit(wall_time)

        # Check round-trip conversion
        assert_array_equal(unit_time, unit_time_back, rtol=1e-10,
                           err_msg="Round-trip conversion between unit time and wall time failed")

        # Test with pandas Series
        unit_time_series = pd.Series(unit_time)
        wall_time_series = unit2wall(unit_time_series)
        assert isinstance(wall_time_series, np.ndarray), "Wall time should be a numpy array even with Series input"
        assert_array_equal(wall_time, wall_time_series, rtol=1e-10,
                           err_msg="unit2wall differs between ndarray and Series")

    def test_time_conversion_with_pandas_datetime(self, assert_array_equal):
        """Test time conversion with pandas datetime objects."""
        # Create pandas datetime array
        dates = pd.date_range(start='2023-01-01 09:30:00', periods=5, freq='H')

        # Convert to seconds since midnight
        seconds = np.array([(d.hour * 3600 + d.minute * 60 + d.second) for d in dates])

        # Convert seconds to unit time
        unit_time = seconds2unit(seconds)

        # Basic checks
        assert isinstance(unit_time, np.ndarray), "Unit time should be a numpy array"
        assert unit_time.shape == seconds.shape, "Unit time has incorrect shape"
        assert np.all(unit_time >= 0) and np.all(unit_time <= 1), "Unit time should be between 0 and 1"

        # Convert back to seconds
        seconds_back = unit2seconds(unit_time)

        # Check round-trip conversion
        assert_array_equal(seconds, seconds_back, rtol=1e-10,
                           err_msg="Round-trip conversion with pandas datetime failed")


class TestFilteringFunctions:
    """Tests for price and return filtering functions."""

    def test_price_filter_basic(self, high_frequency_data, assert_array_equal):
        """Test basic price filtering."""
        # Extract prices and times from high-frequency data
        prices = high_frequency_data['price'].values
        times = high_frequency_data['time'].values

        # Apply price filter
        filtered_prices, filtered_times = price_filter(prices, times)

        # Basic checks
        assert isinstance(filtered_prices, np.ndarray), "Filtered prices should be a numpy array"
        assert isinstance(filtered_times, np.ndarray), "Filtered times should be a numpy array"
        assert filtered_prices.shape == filtered_times.shape, "Filtered prices and times should have the same shape"
        assert len(filtered_prices) <= len(prices), "Filtered prices should not be longer than original prices"

        # Test with pandas Series
        prices_series = pd.Series(prices)
        times_series = pd.Series(times)
        filtered_prices_series, filtered_times_series = price_filter(prices_series, times_series)
        assert isinstance(filtered_prices_series,
                          np.ndarray), "Filtered prices should be a numpy array even with Series input"
        assert_array_equal(filtered_prices, filtered_prices_series, rtol=1e-10,
                           err_msg="price_filter differs between ndarray and Series")

    def test_price_filter_methods(self, high_frequency_data):
        """Test price filtering with different methods."""
        # Extract prices and times from high-frequency data
        prices = high_frequency_data['price'].values
        times = high_frequency_data['time'].values

        # Apply price filter with different methods
        filtered_prices_median, filtered_times_median = price_filter(prices, times, method='median')
        filtered_prices_mean, filtered_times_mean = price_filter(prices, times, method='mean')
        filtered_prices_percentile, filtered_times_percentile = price_filter(prices, times, method='percentile')

        # Basic checks
        assert len(filtered_prices_median) <= len(
            prices), "Median filtered prices should not be longer than original prices"
        assert len(filtered_prices_mean) <= len(
            prices), "Mean filtered prices should not be longer than original prices"
        assert len(filtered_prices_percentile) <= len(
            prices), "Percentile filtered prices should not be longer than original prices"

        # Different methods should give different results
        assert not np.array_equal(filtered_prices_median,
                                  filtered_prices_mean), "Median and mean filtering gave same result"
        assert not np.array_equal(filtered_prices_median,
                                  filtered_prices_percentile), "Median and percentile filtering gave same result"
        assert not np.array_equal(filtered_prices_mean,
                                  filtered_prices_percentile), "Mean and percentile filtering gave same result"

    def test_return_filter_basic(self, high_frequency_data, assert_array_equal):
        """Test basic return filtering."""
        # Extract prices and times from high-frequency data
        prices = high_frequency_data['price'].values
        times = high_frequency_data['time'].values

        # Calculate returns
        returns = np.diff(np.log(prices))
        return_times = times[1:]

        # Apply return filter
        filtered_returns, filtered_times = return_filter(returns, return_times)

        # Basic checks
        assert isinstance(filtered_returns, np.ndarray), "Filtered returns should be a numpy array"
        assert isinstance(filtered_times, np.ndarray), "Filtered times should be a numpy array"
        assert filtered_returns.shape == filtered_times.shape, "Filtered returns and times should have the same shape"
        assert len(filtered_returns) <= len(returns), "Filtered returns should not be longer than original returns"

        # Test with pandas Series
        returns_series = pd.Series(returns)
        times_series = pd.Series(return_times)
        filtered_returns_series, filtered_times_series = return_filter(returns_series, times_series)
        assert isinstance(filtered_returns_series,
                          np.ndarray), "Filtered returns should be a numpy array even with Series input"
        assert_array_equal(filtered_returns, filtered_returns_series, rtol=1e-10,
                           err_msg="return_filter differs between ndarray and Series")

    def test_return_filter_methods(self, high_frequency_data):
        """Test return filtering with different methods."""
        # Extract prices and times from high-frequency data
        prices = high_frequency_data['price'].values
        times = high_frequency_data['time'].values

        # Calculate returns
        returns = np.diff(np.log(prices))
        return_times = times[1:]

        # Apply return filter with different methods
        filtered_returns_median, filtered_times_median = return_filter(returns, return_times, method='median')
        filtered_returns_mean, filtered_times_mean = return_filter(returns, return_times, method='mean')
        filtered_returns_percentile, filtered_times_percentile = return_filter(
            returns, return_times, method='percentile')

        # Basic checks
        assert len(filtered_returns_median) <= len(
            returns), "Median filtered returns should not be longer than original returns"
        assert len(filtered_returns_mean) <= len(
            returns), "Mean filtered returns should not be longer than original returns"
        assert len(filtered_returns_percentile) <= len(
            returns), "Percentile filtered returns should not be longer than original returns"

        # Different methods should give different results
        assert not np.array_equal(filtered_returns_median,
                                  filtered_returns_mean), "Median and mean filtering gave same result"
        assert not np.array_equal(filtered_returns_median,
                                  filtered_returns_percentile), "Median and percentile filtering gave same result"
        assert not np.array_equal(filtered_returns_mean,
                                  filtered_returns_percentile), "Mean and percentile filtering gave same result"

    @pytest.mark.asyncio
    async def test_async_price_filter(self, high_frequency_data):
        """Test asynchronous price filtering."""
        # Extract prices and times from high-frequency data
        prices = high_frequency_data['price'].values
        times = high_frequency_data['time'].values

        # Define progress callback
        progress_updates = []

        async def progress_callback(percent, message):
            progress_updates.append((percent, message))

        # Apply price filter asynchronously
        filtered_prices, filtered_times = await price_filter(prices, times, return_async=True,
                                                             progress_callback=progress_callback)

        # Basic checks
        assert isinstance(filtered_prices, np.ndarray), "Filtered prices should be a numpy array"
        assert isinstance(filtered_times, np.ndarray), "Filtered times should be a numpy array"
        assert filtered_prices.shape == filtered_times.shape, "Filtered prices and times should have the same shape"
        assert len(filtered_prices) <= len(prices), "Filtered prices should not be longer than original prices"

        # Check that progress callback was called
        assert len(progress_updates) > 0, "Progress callback was not called"
        assert progress_updates[-1][0] == 100.0, "Final progress update not 100%"


class TestSamplingFunctions:
    """Tests for sampling functions."""

    def test_subsample_basic(self, high_frequency_data, assert_array_equal):
        """Test basic subsampling."""
        # Extract prices and times from high-frequency data
        prices = high_frequency_data['price'].values
        times = high_frequency_data['time'].values

        # Apply subsampling
        subsampled_prices, subsampled_times = subsample(prices, times, sampling_type='fixed', sampling_interval=10)

        # Basic checks
        assert isinstance(subsampled_prices, np.ndarray), "Subsampled prices should be a numpy array"
        assert isinstance(subsampled_times, np.ndarray), "Subsampled times should be a numpy array"
        assert subsampled_prices.shape == subsampled_times.shape, "Subsampled prices and times should have the same shape"
        assert len(subsampled_prices) < len(prices), "Subsampled prices should be shorter than original prices"

        # Test with pandas Series
        prices_series = pd.Series(prices)
        times_series = pd.Series(times)
        subsampled_prices_series, subsampled_times_series = subsample(prices_series, times_series,
                                                                      sampling_type='fixed', sampling_interval=10)
        assert isinstance(subsampled_prices_series,
                          np.ndarray), "Subsampled prices should be a numpy array even with Series input"
        assert_array_equal(subsampled_prices, subsampled_prices_series, rtol=1e-10,
                           err_msg="subsample differs between ndarray and Series")

    def test_subsample_methods(self, high_frequency_data):
        """Test subsampling with different methods."""
        # Extract prices and times from high-frequency data
        prices = high_frequency_data['price'].values
        times = high_frequency_data['time'].values

        # Apply subsampling with different methods
        subsampled_prices_fixed, subsampled_times_fixed = subsample(prices, times,
                                                                    sampling_type='fixed', sampling_interval=10)
        subsampled_prices_calendar, subsampled_times_calendar = subsample(prices, times,
                                                                          sampling_type='calendar', sampling_interval=300)
        subsampled_prices_business, subsampled_times_business = subsample(prices, times,
                                                                          sampling_type='business', sampling_points=48)

        # Basic checks
        assert len(subsampled_prices_fixed) < len(
            prices), "Fixed subsampled prices should be shorter than original prices"
        assert len(subsampled_prices_calendar) < len(
            prices), "Calendar subsampled prices should be shorter than original prices"
        assert len(subsampled_prices_business) < len(
            prices), "Business subsampled prices should be shorter than original prices"

        # Different methods should give different results
        assert not np.array_equal(subsampled_prices_fixed,
                                  subsampled_prices_calendar), "Fixed and calendar subsampling gave same result"
        assert not np.array_equal(subsampled_prices_fixed,
                                  subsampled_prices_business), "Fixed and business subsampling gave same result"
        assert not np.array_equal(subsampled_prices_calendar,
                                  subsampled_prices_business), "Calendar and business subsampling gave same result"

    def test_refresh_time_basic(self, high_frequency_data, assert_array_equal):
        """Test basic refresh time sampling."""
        # Create two price series with different timestamps
        n = len(high_frequency_data)
        half_n = n // 2

        prices1 = high_frequency_data['price'].values[:half_n]
        times1 = high_frequency_data['time'].values[:half_n]

        prices2 = high_frequency_data['price'].values[half_n:n]
        times2 = high_frequency_data['time'].values[half_n:n]

        # Apply refresh time sampling
        refreshed_prices, refreshed_times = refresh_time([prices1, prices2], [times1, times2])

        # Basic checks
        assert isinstance(refreshed_prices, list), "Refreshed prices should be a list"
        assert isinstance(refreshed_times, np.ndarray), "Refreshed times should be a numpy array"
        assert len(refreshed_prices) == 2, "Refreshed prices should have 2 elements"
        assert len(refreshed_prices[0]) == len(refreshed_prices[1]
                                               ), "Refreshed price series should have the same length"
        assert len(refreshed_prices[0]) == len(
            refreshed_times), "Refreshed prices and times should have the same length"
        assert len(refreshed_prices[0]) <= min(len(prices1), len(prices2)
                                               ), "Refreshed prices should not be longer than the shortest original series"

        # Test with pandas Series
        prices1_series = pd.Series(prices1)
        times1_series = pd.Series(times1)
        prices2_series = pd.Series(prices2)
        times2_series = pd.Series(times2)

        refreshed_prices_series, refreshed_times_series = refresh_time([prices1_series, prices2_series],
                                                                       [times1_series, times2_series])
        assert isinstance(refreshed_prices_series, list), "Refreshed prices should be a list even with Series input"
        assert_array_equal(refreshed_prices[0], refreshed_prices_series[0], rtol=1e-10,
                           err_msg="refresh_time differs between ndarray and Series")

    def test_refresh_time_bivariate_basic(self, high_frequency_data, assert_array_equal):
        """Test basic bivariate refresh time sampling."""
        # Create two price series with different timestamps
        n = len(high_frequency_data)
        half_n = n // 2

        prices1 = high_frequency_data['price'].values[:half_n]
        times1 = high_frequency_data['time'].values[:half_n]

        prices2 = high_frequency_data['price'].values[half_n:n]
        times2 = high_frequency_data['time'].values[half_n:n]

        # Apply bivariate refresh time sampling
        refreshed_prices1, refreshed_prices2, refreshed_times = refresh_time_bivariate(prices1, prices2, times1, times2)

        # Basic checks
        assert isinstance(refreshed_prices1, np.ndarray), "Refreshed prices1 should be a numpy array"
        assert isinstance(refreshed_prices2, np.ndarray), "Refreshed prices2 should be a numpy array"
        assert isinstance(refreshed_times, np.ndarray), "Refreshed times should be a numpy array"
        assert len(refreshed_prices1) == len(refreshed_prices2), "Refreshed price series should have the same length"
        assert len(refreshed_prices1) == len(refreshed_times), "Refreshed prices and times should have the same length"
        assert len(refreshed_prices1) <= min(len(prices1), len(prices2)
                                             ), "Refreshed prices should not be longer than the shortest original series"

        # Test with pandas Series
        prices1_series = pd.Series(prices1)
        times1_series = pd.Series(times1)
        prices2_series = pd.Series(prices2)
        times2_series = pd.Series(times2)

        refreshed_prices1_series, refreshed_prices2_series, refreshed_times_series = refresh_time_bivariate(
            prices1_series, prices2_series, times1_series, times2_series)
        assert isinstance(refreshed_prices1_series,
                          np.ndarray), "Refreshed prices1 should be a numpy array even with Series input"
        assert_array_equal(refreshed_prices1, refreshed_prices1_series, rtol=1e-10,
                           err_msg="refresh_time_bivariate differs between ndarray and Series")

    def test_variance_optimal_sampling_basic(self, high_frequency_data, assert_array_equal):
        """Test basic variance optimal sampling."""
        # Extract prices and times from high-frequency data
        prices = high_frequency_data['price'].values
        times = high_frequency_data['time'].values

        # Apply variance optimal sampling
        sampled_prices, sampled_times = variance_optimal_sampling(prices, times, target_observations=100)

        # Basic checks
        assert isinstance(sampled_prices, np.ndarray), "Sampled prices should be a numpy array"
        assert isinstance(sampled_times, np.ndarray), "Sampled times should be a numpy array"
        assert sampled_prices.shape == sampled_times.shape, "Sampled prices and times should have the same shape"
        assert len(sampled_prices) < len(prices), "Sampled prices should be shorter than original prices"
        assert len(sampled_prices) <= 100, "Sampled prices should not exceed target observations"

        # Test with pandas Series
        prices_series = pd.Series(prices)
        times_series = pd.Series(times)
        sampled_prices_series, sampled_times_series = variance_optimal_sampling(prices_series, times_series,
                                                                                target_observations=100)
        assert isinstance(sampled_prices_series,
                          np.ndarray), "Sampled prices should be a numpy array even with Series input"
        assert_array_equal(sampled_prices, sampled_prices_series, rtol=1e-10,
                           err_msg="variance_optimal_sampling differs between ndarray and Series")

    @pytest.mark.asyncio
    async def test_async_subsample(self, high_frequency_data):
        """Test asynchronous subsampling."""
        # Extract prices and times from high-frequency data
        prices = high_frequency_data['price'].values
        times = high_frequency_data['time'].values

        # Define progress callback
        progress_updates = []

        async def progress_callback(percent, message):
            progress_updates.append((percent, message))

        # Apply subsampling asynchronously
        subsampled_prices, subsampled_times = await subsample(prices, times, sampling_type='fixed',
                                                              sampling_interval=10, return_async=True,
                                                              progress_callback=progress_callback)

        # Basic checks
        assert isinstance(subsampled_prices, np.ndarray), "Subsampled prices should be a numpy array"
        assert isinstance(subsampled_times, np.ndarray), "Subsampled times should be a numpy array"
        assert subsampled_prices.shape == subsampled_times.shape, "Subsampled prices and times should have the same shape"
        assert len(subsampled_prices) < len(prices), "Subsampled prices should be shorter than original prices"

        # Check that progress callback was called
        assert len(progress_updates) > 0, "Progress callback was not called"
        assert progress_updates[-1][0] == 100.0, "Final progress update not 100%"


class TestKernelUtilityFunctions:
    """Tests for kernel utility functions."""

    def test_kernel_bandwidth_basic(self, high_frequency_data, assert_array_equal):
        """Test basic kernel bandwidth calculation."""
        # Extract returns from high-frequency data
        prices = high_frequency_data['price'].values
        returns = np.diff(np.log(prices))

        # Calculate kernel bandwidth
        bandwidth = kernel_bandwidth(returns)

        # Basic checks
        assert isinstance(bandwidth, float), "Kernel bandwidth should be a float"
        assert bandwidth > 0, "Kernel bandwidth should be positive"

        # Test with pandas Series
        returns_series = pd.Series(returns)
        bandwidth_series = kernel_bandwidth(returns_series)
        assert isinstance(bandwidth_series, float), "Kernel bandwidth should be a float even with Series input"
        assert_array_equal(bandwidth, bandwidth_series, rtol=1e-10,
                           err_msg="kernel_bandwidth differs between ndarray and Series")

    def test_kernel_bandwidth_methods(self, high_frequency_data):
        """Test kernel bandwidth calculation with different methods."""
        # Extract returns from high-frequency data
        prices = high_frequency_data['price'].values
        returns = np.diff(np.log(prices))

        # Calculate kernel bandwidth with different methods
        bandwidth_default = kernel_bandwidth(returns)
        bandwidth_optimal = kernel_bandwidth(returns, method='optimal')

        # Basic checks
        assert bandwidth_default > 0, "Default method kernel bandwidth should be positive"
        assert bandwidth_optimal > 0, "Optimal method kernel bandwidth should be positive"

        # Different methods should give different results
        assert not np.isclose(bandwidth_default, bandwidth_optimal), "Default and optimal methods gave same result"

    def test_kernel_weights_basic(self, assert_array_equal):
        """Test basic kernel weight calculation."""
        # Create x values
        x = np.linspace(0, 1, 11)

        # Calculate kernel weights
        weights = kernel_weights(x, kernel_type='parzen')

        # Basic checks
        assert isinstance(weights, np.ndarray), "Kernel weights should be a numpy array"
        assert weights.shape == x.shape, "Kernel weights should have the same shape as input"
        assert np.all(weights >= 0), "Kernel weights should be non-negative"
        assert np.isclose(weights[0], 1.0), "Weight at x=0 should be 1.0"
        assert np.isclose(weights[-1], 0.0), "Weight at x=1 should be 0.0"

        # Test with pandas Series
        x_series = pd.Series(x)
        weights_series = kernel_weights(x_series, kernel_type='parzen')
        assert isinstance(weights_series, np.ndarray), "Kernel weights should be a numpy array even with Series input"
        assert_array_equal(weights, weights_series, rtol=1e-10,
                           err_msg="kernel_weights differs between ndarray and Series")

    def test_kernel_weights_types(self, assert_array_equal):
        """Test kernel weight calculation with different kernel types."""
        # Create x values
        x = np.linspace(0, 1, 11)

        # Calculate kernel weights with different kernel types
        weights_parzen = kernel_weights(x, kernel_type='parzen')
        weights_bartlett = kernel_weights(x, kernel_type='bartlett')
        weights_fejer = kernel_weights(x, kernel_type='fejer')
        weights_qs = kernel_weights(x, kernel_type='qs')
        weights_tukey_hanning = kernel_weights(x, kernel_type='tukey-hanning')

        # Basic checks
        assert np.all(weights_parzen >= 0), "Parzen kernel weights should be non-negative"
        assert np.all(weights_bartlett >= 0), "Bartlett kernel weights should be non-negative"
        assert np.all(weights_fejer >= 0), "Fejer kernel weights should be non-negative"
        assert np.all(weights_qs >= 0), "QS kernel weights should be non-negative"
        assert np.all(weights_tukey_hanning >= 0), "Tukey-Hanning kernel weights should be non-negative"

        # All kernel weights should be 1.0 at x=0
        assert np.isclose(weights_parzen[0], 1.0), "Parzen weight at x=0 should be 1.0"
        assert np.isclose(weights_bartlett[0], 1.0), "Bartlett weight at x=0 should be 1.0"
        assert np.isclose(weights_fejer[0], 1.0), "Fejer weight at x=0 should be 1.0"
        assert np.isclose(weights_qs[0], 1.0), "QS weight at x=0 should be 1.0"
        assert np.isclose(weights_tukey_hanning[0], 1.0), "Tukey-Hanning weight at x=0 should be 1.0"

        # All kernel weights should be 0.0 at x=1
        assert np.isclose(weights_parzen[-1], 0.0), "Parzen weight at x=1 should be 0.0"
        assert np.isclose(weights_bartlett[-1], 0.0), "Bartlett weight at x=1 should be 0.0"
        assert np.isclose(weights_fejer[-1], 0.0), "Fejer weight at x=1 should be 0.0"
        assert np.isclose(weights_qs[-1], 0.0), "QS weight at x=1 should be 0.0"
        assert np.isclose(weights_tukey_hanning[-1], 0.0), "Tukey-Hanning weight at x=1 should be 0.0"

        # Different kernel types should give different results
        kernels = [weights_parzen, weights_bartlett, weights_fejer, weights_qs, weights_tukey_hanning]
        for i in range(len(kernels)):
            for j in range(i+1, len(kernels)):
                assert not np.allclose(kernels[i], kernels[j]), f"Kernel types {i} and {j} gave same result"

    def test_kernel_jitter_lag_length_basic(self, high_frequency_data, assert_array_equal):
        """Test basic kernel jitter lag length calculation."""
        # Extract returns from high-frequency data
        prices = high_frequency_data['price'].values
        returns = np.diff(np.log(prices))

        # Calculate kernel jitter lag length
        lag_length = kernel_jitter_lag_length(returns)

        # Basic checks
        assert isinstance(lag_length, int), "Kernel jitter lag length should be an integer"
        assert lag_length > 0, "Kernel jitter lag length should be positive"

        # Test with pandas Series
        returns_series = pd.Series(returns)
        lag_length_series = kernel_jitter_lag_length(returns_series)
        assert isinstance(lag_length_series, int), "Kernel jitter lag length should be an integer even with Series input"
        assert_array_equal(lag_length, lag_length_series, rtol=1e-10,
                           err_msg="kernel_jitter_lag_length differs between ndarray and Series")


class TestIntegrationTests:
    """Integration tests for realized volatility estimators."""

    def test_estimator_comparison(self, high_frequency_data):
        """Test comparison of different realized volatility estimators."""
        # Extract prices and times from high-frequency data
        prices = high_frequency_data['price'].values
        times = high_frequency_data['time'].values
        returns = np.diff(np.log(prices))

        # Calculate different estimators
        rv = variance(returns)
        bpv = bipower_variation(returns)
        rk = kernel(returns)
        tsrv = twoscale_variance(prices)
        msrv = multiscale_variance(prices)
        qmle = qmle_variance(prices)
        pav = preaveraged_variance(prices)
        tv = threshold_variance(returns)

        # Basic checks
        assert rv > 0, "Realized variance should be positive"
        assert bpv > 0, "Bipower variation should be positive"
        assert rk > 0, "Realized kernel should be positive"
        assert tsrv > 0, "Two-scale realized variance should be positive"
        assert msrv > 0, "Multiscale realized variance should be positive"
        assert qmle > 0, "QMLE variance should be positive"
        assert pav > 0, "Preaveraged variance should be positive"
        assert tv > 0, "Threshold variance should be positive"

        # Add microstructure noise
        noise_level = 0.0005  # 5 basis points
        noise = np.random.normal(0, noise_level, len(prices))
        noisy_prices = prices * np.exp(noise)
        noisy_returns = np.diff(np.log(noisy_prices))

        # Calculate estimators with noise
        rv_noise = variance(noisy_returns)
        bpv_noise = bipower_variation(noisy_returns)
        rk_noise = kernel(noisy_returns)
        tsrv_noise = twoscale_variance(noisy_prices)
        msrv_noise = multiscale_variance(noisy_prices)
        qmle_noise = qmle_variance(noisy_prices)
        pav_noise = preaveraged_variance(noisy_prices)
        tv_noise = threshold_variance(noisy_returns)

        # Calculate noise ratios
        rv_ratio = rv_noise / rv
        bpv_ratio = bpv_noise / bpv
        rk_ratio = rk_noise / rk
        tsrv_ratio = tsrv_noise / tsrv
        msrv_ratio = msrv_noise / msrv
        qmle_ratio = qmle_noise / qmle
        pav_ratio = pav_noise / pav
        tv_ratio = tv_noise / tv

        # Realized variance should be most affected by noise
        assert rv_ratio > bpv_ratio, "Bipower variation not more robust than realized variance"
        assert rv_ratio > rk_ratio, "Realized kernel not more robust than realized variance"
        assert rv_ratio > tsrv_ratio, "Two-scale not more robust than realized variance"
        assert rv_ratio > msrv_ratio, "Multiscale not more robust than realized variance"
        assert rv_ratio > qmle_ratio, "QMLE not more robust than realized variance"
        assert rv_ratio > pav_ratio, "Preaveraged variance not more robust than realized variance"
        assert rv_ratio > tv_ratio, "Threshold variance not more robust than realized variance"

    def test_sampling_scheme_comparison(self, high_frequency_data):
        """Test comparison of different sampling schemes."""
        # Extract prices and times from high-frequency data
        prices = high_frequency_data['price'].values
        times = high_frequency_data['time'].values

        # Apply different sampling schemes
        prices_fixed, times_fixed = subsample(prices, times, sampling_type='fixed', sampling_interval=10)
        prices_calendar, times_calendar = subsample(prices, times, sampling_type='calendar', sampling_interval=300)
        prices_business, times_business = subsample(prices, times, sampling_type='business', sampling_points=48)
        prices_optimal, times_optimal = variance_optimal_sampling(prices, times, target_observations=100)

        # Calculate returns
        returns_fixed = np.diff(np.log(prices_fixed))
        returns_calendar = np.diff(np.log(prices_calendar))
        returns_business = np.diff(np.log(prices_business))
        returns_optimal = np.diff(np.log(prices_optimal))

        # Calculate realized variance with different sampling schemes
        rv_fixed = variance(returns_fixed)
        rv_calendar = variance(returns_calendar)
        rv_business = variance(returns_business)
        rv_optimal = variance(returns_optimal)

        # Basic checks
        assert rv_fixed > 0, "Fixed sampling realized variance should be positive"
        assert rv_calendar > 0, "Calendar sampling realized variance should be positive"
        assert rv_business > 0, "Business sampling realized variance should be positive"
        assert rv_optimal > 0, "Optimal sampling realized variance should be positive"

        # Different sampling schemes should give different results
        assert not np.isclose(rv_fixed, rv_calendar), "Fixed and calendar sampling gave same result"
        assert not np.isclose(rv_fixed, rv_business), "Fixed and business sampling gave same result"
        assert not np.isclose(rv_fixed, rv_optimal), "Fixed and optimal sampling gave same result"
        assert not np.isclose(rv_calendar, rv_business), "Calendar and business sampling gave same result"
        assert not np.isclose(rv_calendar, rv_optimal), "Calendar and optimal sampling gave same result"
        assert not np.isclose(rv_business, rv_optimal), "Business and optimal sampling gave same result"

    def test_jump_detection_workflow(self, high_frequency_data):
        """Test jump detection workflow."""
        # Extract returns from high-frequency data
        prices = high_frequency_data['price'].values
        returns = np.diff(np.log(prices))

        # Add a single large jump
        jump_returns = returns.copy()
        jump_returns[len(jump_returns) // 2] += 0.05  # Add a 5% jump

        # Calculate realized variance and bipower variation
        rv = variance(jump_returns)
        bpv = bipower_variation(jump_returns)

        # Calculate jump test statistic
        jump_ratio = rv / bpv
        jump_test = (rv - bpv) / np.sqrt((np.pi**2 / 4 + np.pi - 5) *
                                         max(1, quarticity(jump_returns)) / len(jump_returns))

        # Calculate p-value
        p_value = 1 - stats.norm.cdf(jump_test)

        # Check for jump
        alpha = 0.05  # 5% significance level
        has_jump = p_value < alpha

        # Basic checks
        assert jump_ratio > 1, "Jump ratio should be greater than 1 for data with jump"
        assert has_jump, "Jump test should detect jump in data with jump"

        # Calculate threshold variance
        tv = threshold_variance(jump_returns)

        # Calculate jump component
        jump_component = rv - tv

        # Basic checks
        assert jump_component > 0, "Jump component should be positive for data with jump"
        assert jump_component / rv > 0.1, "Jump component should be significant portion of total variance"

    def test_noise_robust_estimation_workflow(self, high_frequency_data):
        """Test noise-robust estimation workflow."""
        # Extract prices from high-frequency data
        prices = high_frequency_data['price'].values
        returns = np.diff(np.log(prices))

        # Add microstructure noise
        noise_level = 0.0005  # 5 basis points
        noise = np.random.normal(0, noise_level, len(prices))
        noisy_prices = prices * np.exp(noise)
        noisy_returns = np.diff(np.log(noisy_prices))

        # Calculate realized variance
        rv = variance(noisy_returns)

        # Calculate noise-robust estimators
        rk = kernel(noisy_returns)
        tsrv = twoscale_variance(noisy_prices)
        msrv = multiscale_variance(noisy_prices)
        qmle = qmle_variance(noisy_prices)
        pav = preaveraged_variance(noisy_prices)

        # Calculate noise variance estimate
        noise_var_estimate = (rv - rk) / 2

        # Basic checks
        assert noise_var_estimate > 0, "Noise variance estimate should be positive"
        assert noise_var_estimate < rv, "Noise variance estimate should be less than realized variance"

        # Compare noise-robust estimators
        estimators = [rk, tsrv, msrv, qmle, pav]
        estimator_names = ["Realized Kernel", "Two-Scale RV", "Multiscale RV", "QMLE", "Preaveraged RV"]

        # All noise-robust estimators should be less than realized variance
        for i, estimator in enumerate(estimators):
            assert estimator < rv, f"{estimator_names[i]} should be less than realized variance"

        # Calculate signal-to-noise ratios
        snr = [estimator / noise_var_estimate for estimator in estimators]

        # All signal-to-noise ratios should be positive
        for i, ratio in enumerate(snr):
            assert ratio > 0, f"{estimator_names[i]} signal-to-noise ratio should be positive"

    @pytest.mark.asyncio
    async def test_async_estimation_workflow(self, high_frequency_data):
        """Test asynchronous estimation workflow."""
        # Extract prices and times from high-frequency data
        prices = high_frequency_data['price'].values
        times = high_frequency_data['time'].values

        # Define progress callback
        progress_updates = []

        async def progress_callback(percent, message):
            progress_updates.append((percent, message))

        # Step 1: Filter prices asynchronously
        filtered_prices, filtered_times = await price_filter(prices, times, return_async=True,
                                                             progress_callback=progress_callback)

        # Step 2: Subsample filtered prices asynchronously
        subsampled_prices, subsampled_times = await subsample(filtered_prices, filtered_times,
                                                              sampling_type='calendar', sampling_interval=300,
                                                              return_async=True, progress_callback=progress_callback)

        # Step 3: Calculate returns
        returns = np.diff(np.log(subsampled_prices))

        # Step 4: Calculate realized variance asynchronously
        rv = await variance(returns, return_async=True, progress_callback=progress_callback)

        # Step 5: Calculate realized kernel asynchronously
        rk = await kernel(returns, return_async=True, progress_callback=progress_callback)

        # Basic checks
        assert isinstance(rv, float), "Realized variance should be a float"
        assert isinstance(rk, float), "Realized kernel should be a float"
        assert rv > 0, "Realized variance should be positive"
        assert rk > 0, "Realized kernel should be positive"

        # Check that progress callback was called multiple times
        assert len(progress_updates) > 0, "Progress callback was not called"
        assert progress_updates[-1][0] == 100.0, "Final progress update not 100%"

    def test_pandas_datetime_integration(self, high_frequency_data):
        """Test integration with pandas datetime objects."""
        # Create pandas DataFrame with DatetimeIndex
        df = high_frequency_data.copy()
        df.set_index('time', inplace=True)

        # Calculate returns
        returns = df['price'].pct_change().dropna()

        # Calculate realized variance
        rv = variance(returns)

        # Calculate realized kernel
        rk = kernel(returns)

        # Basic checks
        assert isinstance(rv, float), "Realized variance should be a float"
        assert isinstance(rk, float), "Realized kernel should be a float"
        assert rv > 0, "Realized variance should be positive"
        assert rk > 0, "Realized kernel should be positive"

        # Calculate realized variance with calendar time sampling
        rv_calendar = variance(df['price'], use_prices=True, sampling_type='calendar', sampling_interval=300)

        # Basic checks
        assert isinstance(rv_calendar, float), "Calendar time realized variance should be a float"
        assert rv_calendar > 0, "Calendar time realized variance should be positive"


class TestIntegrationTests:  # (Duplicate class name in original file, kept as is)
    """Integration tests for realized volatility estimators."""

    def test_estimator_comparison(self, high_frequency_data):
        """Test comparison of different realized volatility estimators."""
        # Extract prices and times from high-frequency data
        prices = high_frequency_data['price'].values
        times = high_frequency_data['time'].values
        returns = np.diff(np.log(prices))

        # Calculate different estimators
        rv = variance(returns)
        bpv = bipower_variation(returns)
        rk = kernel(returns)
        tsrv = twoscale_variance(prices)
        msrv = multiscale_variance(prices)
        qmle = qmle_variance(prices)
        pav = preaveraged_variance(prices)
        tv = threshold_variance(returns)

        # Basic checks
        assert rv > 0, "Realized variance should be positive"
        assert bpv > 0, "Bipower variation should be positive"
        assert rk > 0, "Realized kernel should be positive"
        assert tsrv > 0, "Two-scale realized variance should be positive"
        assert msrv > 0, "Multiscale realized variance should be positive"
        assert qmle > 0, "QMLE variance should be positive"
        assert pav > 0, "Preaveraged variance should be positive"
        assert tv > 0, "Threshold variance should be positive"

        # Add microstructure noise
        noise_level = 0.0005  # 5 basis points
        noise = np.random.normal(0, noise_level, len(prices))
        noisy_prices = prices * np.exp(noise)
        noisy_returns = np.diff(np.log(noisy_prices))

        # Calculate estimators with noise
        rv_noise = variance(noisy_returns)
        bpv_noise = bipower_variation(noisy_returns)
        rk_noise = kernel(noisy_returns)
        tsrv_noise = twoscale_variance(noisy_prices)
        msrv_noise = multiscale_variance(noisy_prices)
        qmle_noise = qmle_variance(noisy_prices)
        pav_noise = preaveraged_variance(noisy_prices)
        tv_noise = threshold_variance(noisy_returns)

        # Calculate noise ratios
        rv_ratio = rv_noise / rv
        bpv_ratio = bpv_noise / bpv
        rk_ratio = rk_noise / rk
        tsrv_ratio = tsrv_noise / tsrv
        msrv_ratio = msrv_noise / msrv
        qmle_ratio = qmle_noise / qmle
        pav_ratio = pav_noise / pav
        tv_ratio = tv_noise / tv

        # Realized variance should be most affected by noise
        assert rv_ratio > bpv_ratio, "Bipower variation not more robust than realized variance"
        assert rv_ratio > rk_ratio, "Realized kernel not more robust than realized variance"
        assert rv_ratio > tsrv_ratio, "Two-scale not more robust than realized variance"
        assert rv_ratio > msrv_ratio, "Multiscale not more robust than realized variance"
        assert rv_ratio > qmle_ratio, "QMLE not more robust than realized variance"
        assert rv_ratio > pav_ratio, "Preaveraged variance not more robust than realized variance"
        assert rv_ratio > tv_ratio, "Threshold variance not more robust than realized variance"

    def test_sampling_scheme_comparison(self, high_frequency_data):
        """Test comparison of different sampling schemes."""
        # Extract prices and times from high-frequency data
        prices = high_frequency_data['price'].values
        times = high_frequency_data['time'].values

        # Apply different sampling schemes
        prices_fixed, times_fixed = subsample(prices, times, sampling_type='fixed', sampling_interval=10)
        prices_calendar, times_calendar = subsample(prices, times, sampling_type='calendar', sampling_interval=300)
        prices_business, times_business = subsample(prices, times, sampling_type='business', sampling_points=48)
        prices_optimal, times_optimal = variance_optimal_sampling(prices, times, target_observations=100)

        # Calculate returns
        returns_fixed = np.diff(np.log(prices_fixed))
        returns_calendar = np.diff(np.log(prices_calendar))
        returns_business = np.diff(np.log(prices_business))
        returns_optimal = np.diff(np.log(prices_optimal))

        # Calculate realized variance with different sampling schemes
        rv_fixed = variance(returns_fixed)
        rv_calendar = variance(returns_calendar)
        rv_business = variance(returns_business)
        rv_optimal = variance(returns_optimal)

        # Basic checks
        assert rv_fixed > 0, "Fixed sampling realized variance should be positive"
        assert rv_calendar > 0, "Calendar sampling realized variance should be positive"
        assert rv_business > 0, "Business sampling realized variance should be positive"
        assert rv_optimal > 0, "Optimal sampling realized variance should be positive"

        # Different sampling schemes should give different results
        assert not np.isclose(rv_fixed, rv_calendar), "Fixed and calendar sampling gave same result"
        assert not np.isclose(rv_fixed, rv_business), "Fixed and business sampling gave same result"
        assert not np.isclose(rv_fixed, rv_optimal), "Fixed and optimal sampling gave same result"
        assert not np.isclose(rv_calendar, rv_business), "Calendar and business sampling gave same result"
        assert not np.isclose(rv_calendar, rv_optimal), "Calendar and optimal sampling gave same result"
        assert not np.isclose(rv_business, rv_optimal), "Business and optimal sampling gave same result"

    def test_jump_detection_workflow(self, high_frequency_data):
        """Test jump detection workflow."""
        # Extract returns from high-frequency data
        prices = high_frequency_data['price'].values
        returns = np.diff(np.log(prices))

        # Add a single large jump
        jump_returns = returns.copy()
        jump_returns[len(jump_returns) // 2] += 0.05  # Add a 5% jump

        # Calculate realized variance and bipower variation
        rv = variance(jump_returns)
        bpv = bipower_variation(jump_returns)

        # Calculate jump test statistic
        jump_ratio = rv / bpv
        jump_test = (rv - bpv) / np.sqrt((np.pi**2 / 4 + np.pi - 5) *
                                         max(1, quarticity(jump_returns)) / len(jump_returns))

        # Calculate p-value
        p_value = 1 - stats.norm.cdf(jump_test)

        # Check for jump
        alpha = 0.05  # 5% significance level
        has_jump = p_value < alpha

        # Basic checks
        assert jump_ratio > 1, "Jump ratio should be greater than 1 for data with jump"
        assert has_jump, "Jump test should detect jump in data with jump"

        # Calculate threshold variance
        tv = threshold_variance(jump_returns)

        # Calculate jump component
        jump_component = rv - tv

        # Basic checks
        assert jump_component > 0, "Jump component should be positive for data with jump"
        assert jump_component / rv > 0.1, "Jump component should be significant portion of total variance"

    def test_noise_robust_estimation_workflow(self, high_frequency_data):
        """Test noise-robust estimation workflow."""
        # Extract prices from high-frequency data
        prices = high_frequency_data['price'].values
        returns = np.diff(np.log(prices))

        # Add microstructure noise
        noise_level = 0.0005  # 5 basis points
        noise = np.random.normal(0, noise_level, len(prices))
        noisy_prices = prices * np.exp(noise)
        noisy_returns = np.diff(np.log(noisy_prices))

        # Calculate realized variance
        rv = variance(noisy_returns)

        # Calculate noise-robust estimators
        rk = kernel(noisy_returns)
        tsrv = twoscale_variance(noisy_prices)
        msrv = multiscale_variance(noisy_prices)
        qmle = qmle_variance(noisy_prices)
        pav = preaveraged_variance(noisy_prices)

        # Calculate noise variance estimate
        noise_var_estimate = (rv - rk) / 2

        # Basic checks
        assert noise_var_estimate > 0, "Noise variance estimate should be positive"
        assert noise_var_estimate < rv, "Noise variance estimate should be less than realized variance"

        # Compare noise-robust estimators
        estimators = [rk, tsrv, msrv, qmle, pav]
        estimator_names = ["Realized Kernel", "Two-Scale RV", "Multiscale RV", "QMLE", "Preaveraged RV"]

        # All noise-robust estimators should be less than realized variance
        for i, estimator in enumerate(estimators):
            assert estimator < rv, f"{estimator_names[i]} should be less than realized variance"

        # Calculate signal-to-noise ratios
        snr = [estimator / noise_var_estimate for estimator in estimators]

        # All signal-to-noise ratios should be positive
        for i, ratio in enumerate(snr):
            assert ratio > 0, f"{estimator_names[i]} signal-to-noise ratio should be positive"

    @pytest.mark.asyncio
    async def test_async_estimation_workflow(self, high_frequency_data):
        """Test asynchronous estimation workflow."""
        # Extract prices and times from high-frequency data
        prices = high_frequency_data['price'].values
        times = high_frequency_data['time'].values

        # Define progress callback
        progress_updates = []

        async def progress_callback(percent, message):
            progress_updates.append((percent, message))

        # Step 1: Filter prices asynchronously
        filtered_prices, filtered_times = await price_filter(prices, times, return_async=True,
                                                             progress_callback=progress_callback)

        # Step 2: Subsample filtered prices asynchronously
        subsampled_prices, subsampled_times = await subsample(filtered_prices, filtered_times,
                                                              sampling_type='calendar', sampling_interval=300,
                                                              return_async=True, progress_callback=progress_callback)

        # Step 3: Calculate returns
        returns = np.diff(np.log(subsampled_prices))

        # Step 4: Calculate realized variance asynchronously
        rv = await variance(returns, return_async=True, progress_callback=progress_callback)

        # Step 5: Calculate realized kernel asynchronously
        rk = await kernel(returns, return_async=True, progress_callback=progress_callback)

        # Basic checks
        assert isinstance(rv, float), "Realized variance should be a float"
        assert isinstance(rk, float), "Realized kernel should be a float"
        assert rv > 0, "Realized variance should be positive"
        assert rk > 0, "Realized kernel should be positive"

        # Check that progress callback was called multiple times
        assert len(progress_updates) > 0, "Progress callback was not called"
        assert progress_updates[-1][0] == 100.0, "Final progress update not 100%"

    def test_pandas_datetime_integration(self, high_frequency_data):
        """Test integration with pandas datetime objects."""
        # Create pandas DataFrame with DatetimeIndex
        df = high_frequency_data.copy()
        df.set_index('time', inplace=True)

        # Calculate returns
        returns = df['price'].pct_change().dropna()

        # Calculate realized variance
        rv = variance(returns)

        # Calculate realized kernel
        rk = kernel(returns)

        # Basic checks
        assert isinstance(rv, float), "Realized variance should be a float"
        assert isinstance(rk, float), "Realized kernel should be a float"
        assert rv > 0, "Realized variance should be positive"
        assert rk > 0, "Realized kernel should be positive"

        # Calculate realized variance with calendar time sampling
        rv_calendar = variance(df['price'], use_prices=True, sampling_type='calendar', sampling_interval=300)

        # Basic checks
        assert isinstance(rv_calendar, float), "Calendar time realized variance should be a float"
        assert rv_calendar > 0, "Calendar time realized variance should be positive"
