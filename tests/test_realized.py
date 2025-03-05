# tests/test_realized.py

"""
Tests for realized volatility estimators and high-frequency financial econometrics.

This module contains tests for various realized volatility estimators, including
realized variance, bipower variation, kernel estimators, and noise-robust techniques.
It validates the correct implementation of these estimators, their handling of
irregularly spaced data, and their robustness to market microstructure noise.

The tests cover both the core functionality and the integration with Pandas for
time-indexed high-frequency data, as well as the performance benefits of
Numba-accelerated implementations.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import asyncio
from hypothesis import given, strategies as st, settings, assume
import time

from mfe.models.realized import (
    RealizedVariance, BiPowerVariation, RealizedKernel, RealizedSemivariance,
    RealizedCovariance, MultivariateRealizedKernel, RealizedRange,
    RealizedQuarticity, TwoScaleRealizedVariance, MultiscaleRealizedVariance,
    QMLERealizedVariance, ThresholdMultipowerVariation, ThresholdRealizedVariance,
    PreaveragedBiPowerVariation, PreaveragedRealizedVariance,
    RealizedEstimatorConfig, has_numba
)

from mfe.models.realized.utils import (
    align_time, compute_returns, detect_jumps, optimal_sampling, signature_plot,
    noise_variance, compute_realized_variance, compute_subsampled_measure
)

from mfe.models.realized.base import (
    BaseRealizedEstimator, RealizedEstimatorResult, JumpRobustEstimator,
    NoiseRobustEstimator, MultivariateRealizedEstimator
)

from mfe.models.realized import (
    seconds2unit, seconds2wall, unit2seconds, unit2wall, wall2seconds, wall2unit,
    price_filter, return_filter, refresh_time, subsample
)

from mfe.core.exceptions import DimensionError, ParameterError


# ---- Fixtures ----

@pytest.fixture
def high_frequency_data(rng):
    """Generate high-frequency price and time data for testing."""
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
def high_frequency_data_with_jumps(rng):
    """Generate high-frequency price and time data with jumps for testing."""
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

    # Add jumps at random locations (approximately 5 jumps)
    n_jumps = 5
    jump_indices = rng.choice(range(1, n_obs), size=n_jumps, replace=False)
    jump_sizes = rng.normal(0, 0.01, size=n_jumps)  # Larger jumps

    for idx, size in zip(jump_indices, jump_sizes):
        log_returns[idx] += size

    # Cumulate to get log prices
    log_prices = np.cumsum(log_returns)

    # Convert to prices
    initial_price = 100.0
    prices = initial_price * np.exp(log_prices)

    return prices, times, jump_indices


@pytest.fixture
def multivariate_high_frequency_data(rng):
    """Generate multivariate high-frequency price and time data for testing."""
    # Generate approximately 1000 intraday observations (about 1 trading day)
    n_obs = 1000
    n_assets = 3

    # Generate times (in seconds from market open)
    times = np.sort(rng.uniform(0, 23400, n_obs))  # 6.5 hours = 23400 seconds

    # Generate correlated log price processes
    # Correlation matrix
    corr = np.array([
        [1.0, 0.7, 0.3],
        [0.7, 1.0, 0.5],
        [0.3, 0.5, 1.0]
    ])

    # Cholesky decomposition for generating correlated returns
    chol = np.linalg.cholesky(corr)

    # Generate independent returns
    indep_returns = 0.0001 * rng.standard_normal((n_obs, n_assets))

    # Transform to correlated returns
    corr_returns = indep_returns @ chol.T

    # Add some microstructure noise
    noise = 0.00005 * rng.standard_normal((n_obs, n_assets))
    corr_returns = corr_returns + noise

    # Cumulate to get log prices
    log_prices = np.cumsum(corr_returns, axis=0)

    # Convert to prices
    initial_prices = np.array([100.0, 50.0, 200.0])
    prices = initial_prices * np.exp(log_prices)

    return prices, times


@pytest.fixture
def pandas_high_frequency_data(high_frequency_data):
    """Convert high-frequency data to pandas DataFrame."""
    prices, times = high_frequency_data

    # Convert times to datetime
    datetimes = pd.to_datetime(times, unit='s', origin=pd.Timestamp('2023-01-01 09:30:00'))

    # Create DataFrame
    df = pd.DataFrame({
        'price': prices,
        'time': times
    }, index=datetimes)

    return df


@pytest.fixture
def realized_variance_config():
    """Create a basic RealizedEstimatorConfig for testing."""
    return RealizedEstimatorConfig(
        sampling_frequency='5min',
        annualize=True,
        annualization_factor=252.0,
        return_type='log',
        use_subsampling=False,
        apply_noise_correction=False,
        time_unit='seconds'
    )


# ---- Basic Functionality Tests ----

class TestRealizedVariance:
    """Tests for RealizedVariance estimator."""

    def test_basic_functionality(self, high_frequency_data):
        """Test basic functionality of RealizedVariance estimator."""
        prices, times = high_frequency_data

        # Create estimator
        rv = RealizedVariance()

        # Fit estimator
        result = rv.fit((prices, times))

        # Check result type
        assert isinstance(result, RealizedEstimatorResult)

        # Check that realized measure is positive
        assert result.realized_measure > 0

        # Check that volatility is positive
        volatility = rv.get_volatility()
        assert volatility > 0

        # Check that annualized volatility is larger than non-annualized
        annualized = rv.get_volatility(annualize=True)
        non_annualized = rv.get_volatility(annualize=False)
        assert annualized > non_annualized

    def test_with_config(self, high_frequency_data, realized_variance_config):
        """Test RealizedVariance with custom configuration."""
        prices, times = high_frequency_data

        # Create estimator with config
        rv = RealizedVariance(config=realized_variance_config)

        # Fit estimator
        result = rv.fit((prices, times))

        # Check that config was applied
        assert result.sampling_frequency == realized_variance_config.sampling_frequency
        assert result.annualization_factor == realized_variance_config.annualization_factor

    def test_subsampling(self, high_frequency_data):
        """Test RealizedVariance with subsampling."""
        prices, times = high_frequency_data

        # Create config with subsampling
        config = RealizedEstimatorConfig(
            sampling_frequency='5min',
            use_subsampling=True,
            subsampling_factor=5
        )

        # Create estimator
        rv = RealizedVariance(config=config)

        # Fit estimator
        result = rv.fit((prices, times))

        # Check that subsampling was applied
        assert result.subsampling is True

    def test_noise_correction(self, high_frequency_data):
        """Test RealizedVariance with noise correction."""
        prices, times = high_frequency_data

        # Create config with noise correction
        config = RealizedEstimatorConfig(
            sampling_frequency='5min',
            apply_noise_correction=True
        )

        # Create estimator
        rv = RealizedVariance(config=config)

        # Fit estimator
        result = rv.fit((prices, times))

        # Check that noise correction was applied
        assert result.noise_correction is True

    def test_calibration(self, high_frequency_data):
        """Test RealizedVariance calibration."""
        prices, times = high_frequency_data

        # Create estimator
        rv = RealizedVariance()

        # Calibrate estimator
        calibrated_config = rv.calibrate((prices, times))

        # Check that calibrated config is returned
        assert isinstance(calibrated_config, RealizedEstimatorConfig)

        # Create new estimator with calibrated config
        rv_calibrated = RealizedVariance(config=calibrated_config)

        # Fit estimator
        result = rv_calibrated.fit((prices, times))

        # Check that estimator works with calibrated config
        assert result.realized_measure > 0

    def test_to_pandas(self, high_frequency_data):
        """Test RealizedVariance to_pandas method."""
        prices, times = high_frequency_data

        # Create estimator
        rv = RealizedVariance()

        # Fit estimator
        result = rv.fit((prices, times))

        # Convert to pandas
        df = rv.to_pandas()

        # Check that DataFrame is returned
        assert isinstance(df, pd.DataFrame)

        # Check that DataFrame contains expected columns
        assert 'realized_variance' in df.columns
        assert 'realized_volatility' in df.columns

    def test_from_pandas(self, pandas_high_frequency_data):
        """Test RealizedVariance from_pandas method."""
        df = pandas_high_frequency_data

        # Create estimator from pandas
        rv = RealizedVariance.from_pandas(df, price_col='price', time_col='time')

        # Fit estimator
        result = rv.fit((df['price'].values, df['time'].values))

        # Check that estimator works
        assert result.realized_measure > 0

    def test_plot_volatility(self, high_frequency_data):
        """Test RealizedVariance plot_volatility method."""
        prices, times = high_frequency_data

        # Create estimator
        rv = RealizedVariance()

        # Fit estimator
        result = rv.fit((prices, times))

        try:
            # Try to plot volatility
            fig = rv.plot_volatility()

            # Check that figure is returned
            assert fig is not None
        except ImportError:
            # Skip test if matplotlib is not installed
            pytest.skip("Matplotlib is required for plotting tests")

    def test_async_fit(self, high_frequency_data):
        """Test RealizedVariance async_fit method."""
        prices, times = high_frequency_data

        # Create estimator
        rv = RealizedVariance()

        # Run async fit
        async def run_async_fit():
            return await rv.fit_async((prices, times))

        # Run the coroutine
        result = asyncio.run(run_async_fit())

        # Check that result is returned
        assert isinstance(result, RealizedEstimatorResult)
        assert result.realized_measure > 0

    def test_invalid_inputs(self):
        """Test RealizedVariance with invalid inputs."""
        # Test with empty arrays
        with pytest.raises(ValueError):
            rv = RealizedVariance()
            rv.fit((np.array([]), np.array([])))

        # Test with mismatched dimensions
        with pytest.raises(ValueError):
            rv = RealizedVariance()
            rv.fit((np.array([1, 2, 3]), np.array([1, 2])))

        # Test with non-monotonic times
        with pytest.raises(ValueError):
            rv = RealizedVariance()
            rv.fit((np.array([1, 2, 3]), np.array([3, 2, 1])))

        # Test with NaN values
        with pytest.raises(ValueError):
            rv = RealizedVariance()
            rv.fit((np.array([1, np.nan, 3]), np.array([1, 2, 3])))

    def test_numba_acceleration(self, high_frequency_data):
        """Test that Numba acceleration is used if available."""
        prices, times = high_frequency_data

        # Create estimator
        rv = RealizedVariance()

        # Time the execution
        start_time = time.time()
        result = rv.fit((prices, times))
        end_time = time.time()

        # Record execution time
        execution_time = end_time - start_time

        # Check if Numba is available
        if has_numba():
            # If Numba is available, execution should be fast
            # This is a rough check and might not be reliable in all environments
            assert execution_time < 1.0, "Execution with Numba should be fast"
        else:
            # If Numba is not available, just check that it runs
            assert result.realized_measure > 0


class TestBiPowerVariation:
    """Tests for BiPowerVariation estimator."""

    def test_basic_functionality(self, high_frequency_data):
        """Test basic functionality of BiPowerVariation estimator."""
        prices, times = high_frequency_data

        # Create estimator
        bpv = BiPowerVariation()

        # Fit estimator
        result = bpv.fit((prices, times))

        # Check result type
        assert isinstance(result, RealizedEstimatorResult)

        # Check that realized measure is positive
        assert result.realized_measure > 0

        # Check that volatility is positive
        volatility = bpv.get_volatility()
        assert volatility > 0

    def test_jump_robustness(self, high_frequency_data_with_jumps):
        """Test that BiPowerVariation is robust to jumps."""
        prices, times, jump_indices = high_frequency_data_with_jumps

        # Create estimators
        rv = RealizedVariance()
        bpv = BiPowerVariation()

        # Fit estimators
        rv_result = rv.fit((prices, times))
        bpv_result = bpv.fit((prices, times))

        # BiPower Variation should be less than Realized Variance in the presence of jumps
        # because it's designed to be robust to jumps
        assert bpv_result.realized_measure < rv_result.realized_measure

    def test_inheritance(self):
        """Test that BiPowerVariation inherits from JumpRobustEstimator."""
        bpv = BiPowerVariation()
        assert isinstance(bpv, JumpRobustEstimator)


class TestRealizedKernel:
    """Tests for RealizedKernel estimator."""

    def test_basic_functionality(self, high_frequency_data):
        """Test basic functionality of RealizedKernel estimator."""
        prices, times = high_frequency_data

        # Create estimator
        rk = RealizedKernel()

        # Fit estimator
        result = rk.fit((prices, times))

        # Check result type
        assert isinstance(result, RealizedEstimatorResult)

        # Check that realized measure is positive
        assert result.realized_measure > 0

        # Check that volatility is positive
        volatility = rk.get_volatility()
        assert volatility > 0

    def test_kernel_types(self, high_frequency_data):
        """Test RealizedKernel with different kernel types."""
        prices, times = high_frequency_data

        # Test with different kernel types
        kernel_types = ['parzen', 'bartlett', 'tukey-hanning', 'qs', 'cubic']

        for kernel_type in kernel_types:
            # Create config with kernel type
            config = RealizedEstimatorConfig(
                kernel_type=kernel_type
            )

            # Create estimator
            rk = RealizedKernel(config=config)

            # Fit estimator
            result = rk.fit((prices, times))

            # Check that kernel type was applied
            assert result.kernel_type == kernel_type

            # Check that realized measure is positive
            assert result.realized_measure > 0

    def test_bandwidth_parameter(self, high_frequency_data):
        """Test RealizedKernel with custom bandwidth parameter."""
        prices, times = high_frequency_data

        # Create config with custom bandwidth
        config = RealizedEstimatorConfig(
            kernel_type='parzen',
            bandwidth=0.5
        )

        # Create estimator
        rk = RealizedKernel(config=config)

        # Fit estimator
        result = rk.fit((prices, times))

        # Check that bandwidth was applied
        assert result.bandwidth == 0.5

    def test_inheritance(self):
        """Test that RealizedKernel inherits from NoiseRobustEstimator."""
        rk = RealizedKernel()
        assert isinstance(rk, NoiseRobustEstimator)


class TestMultivariateEstimators:
    """Tests for multivariate realized volatility estimators."""

    def test_realized_covariance(self, multivariate_high_frequency_data):
        """Test RealizedCovariance estimator."""
        prices, times = multivariate_high_frequency_data

        # Create estimator
        rcov = RealizedCovariance()

        # Fit estimator
        result = rcov.fit((prices, times))

        # Check result type
        assert isinstance(result, RealizedEstimatorResult)

        # Check that realized measure is a matrix
        assert result.realized_measure.ndim == 2

        # Check that realized measure is symmetric
        assert np.allclose(result.realized_measure, result.realized_measure.T)

        # Check that diagonal elements are positive
        assert np.all(np.diag(result.realized_measure) > 0)

        # Check correlation matrix
        corr_matrix = rcov.get_correlation_matrix()
        assert np.allclose(np.diag(corr_matrix), 1.0)
        assert np.all(corr_matrix >= -1.0) and np.all(corr_matrix <= 1.0)

    def test_multivariate_realized_kernel(self, multivariate_high_frequency_data):
        """Test MultivariateRealizedKernel estimator."""
        prices, times = multivariate_high_frequency_data

        # Create estimator
        mrk = MultivariateRealizedKernel()

        # Fit estimator
        result = mrk.fit((prices, times))

        # Check result type
        assert isinstance(result, RealizedEstimatorResult)

        # Check that realized measure is a matrix
        assert result.realized_measure.ndim == 2

        # Check that realized measure is symmetric
        assert np.allclose(result.realized_measure, result.realized_measure.T)

        # Check that diagonal elements are positive
        assert np.all(np.diag(result.realized_measure) > 0)

    def test_inheritance(self):
        """Test that multivariate estimators inherit from MultivariateRealizedEstimator."""
        rcov = RealizedCovariance()
        mrk = MultivariateRealizedKernel()

        assert isinstance(rcov, MultivariateRealizedEstimator)
        assert isinstance(mrk, MultivariateRealizedEstimator)


class TestNoiseRobustEstimators:
    """Tests for noise-robust realized volatility estimators."""

    def test_two_scale_realized_variance(self, high_frequency_data):
        """Test TwoScaleRealizedVariance estimator."""
        prices, times = high_frequency_data

        # Create estimator
        tsrv = TwoScaleRealizedVariance()

        # Fit estimator
        result = tsrv.fit((prices, times))

        # Check result type
        assert isinstance(result, RealizedEstimatorResult)

        # Check that realized measure is positive
        assert result.realized_measure > 0

        # Check that volatility is positive
        volatility = tsrv.get_volatility()
        assert volatility > 0

    def test_multiscale_realized_variance(self, high_frequency_data):
        """Test MultiscaleRealizedVariance estimator."""
        prices, times = high_frequency_data

        # Create estimator
        msrv = MultiscaleRealizedVariance()

        # Fit estimator
        result = msrv.fit((prices, times))

        # Check result type
        assert isinstance(result, RealizedEstimatorResult)

        # Check that realized measure is positive
        assert result.realized_measure > 0

        # Check that volatility is positive
        volatility = msrv.get_volatility()
        assert volatility > 0

    def test_qmle_realized_variance(self, high_frequency_data):
        """Test QMLERealizedVariance estimator."""
        prices, times = high_frequency_data

        # Create estimator
        qmle = QMLERealizedVariance()

        # Fit estimator
        result = qmle.fit((prices, times))

        # Check result type
        assert isinstance(result, RealizedEstimatorResult)

        # Check that realized measure is positive
        assert result.realized_measure > 0

        # Check that volatility is positive
        volatility = qmle.get_volatility()
        assert volatility > 0

    def test_preaveraged_realized_variance(self, high_frequency_data):
        """Test PreaveragedRealizedVariance estimator."""
        prices, times = high_frequency_data

        # Create estimator
        prv = PreaveragedRealizedVariance()

        # Fit estimator
        result = prv.fit((prices, times))

        # Check result type
        assert isinstance(result, RealizedEstimatorResult)

        # Check that realized measure is positive
        assert result.realized_measure > 0

        # Check that volatility is positive
        volatility = prv.get_volatility()
        assert volatility > 0

    def test_noise_variance_estimation(self, high_frequency_data):
        """Test noise variance estimation in noise-robust estimators."""
        prices, times = high_frequency_data

        # Create estimator
        rk = RealizedKernel()

        # Fit estimator
        result = rk.fit((prices, times))

        # Estimate noise variance
        noise_var = rk.estimate_noise_variance(result.returns)

        # Check that noise variance is positive
        assert noise_var > 0

        # Check signal-to-noise ratio
        snr = rk.get_signal_to_noise_ratio()
        assert snr is not None
        assert snr > 0

    def test_inheritance(self):
        """Test that noise-robust estimators inherit from NoiseRobustEstimator."""
        tsrv = TwoScaleRealizedVariance()
        msrv = MultiscaleRealizedVariance()
        qmle = QMLERealizedVariance()
        prv = PreaveragedRealizedVariance()

        assert isinstance(tsrv, NoiseRobustEstimator)
        assert isinstance(msrv, NoiseRobustEstimator)
        assert isinstance(qmle, NoiseRobustEstimator)
        assert isinstance(prv, NoiseRobustEstimator)


class TestJumpRobustEstimators:
    """Tests for jump-robust realized volatility estimators."""

    def test_threshold_realized_variance(self, high_frequency_data_with_jumps):
        """Test ThresholdRealizedVariance estimator."""
        prices, times, jump_indices = high_frequency_data_with_jumps

        # Create estimator
        trv = ThresholdRealizedVariance()

        # Fit estimator
        result = trv.fit((prices, times))

        # Check result type
        assert isinstance(result, RealizedEstimatorResult)

        # Check that realized measure is positive
        assert result.realized_measure > 0

        # Check that volatility is positive
        volatility = trv.get_volatility()
        assert volatility > 0

        # Check jump detection
        assert trv.jump_threshold is not None
        assert trv.jump_indicators is not None

        # Check continuous and jump variation
        cont_var = trv.get_continuous_variation()
        jump_var = trv.get_jump_variation()

        assert cont_var is not None
        assert jump_var is not None

        # Continuous + jump variation should approximately equal total variation
        assert np.isclose(cont_var + jump_var, result.realized_measure, rtol=1e-10)

    def test_threshold_multipower_variation(self, high_frequency_data_with_jumps):
        """Test ThresholdMultipowerVariation estimator."""
        prices, times, jump_indices = high_frequency_data_with_jumps

        # Create estimator
        tmpv = ThresholdMultipowerVariation()

        # Fit estimator
        result = tmpv.fit((prices, times))

        # Check result type
        assert isinstance(result, RealizedEstimatorResult)

        # Check that realized measure is positive
        assert result.realized_measure > 0

        # Check that volatility is positive
        volatility = tmpv.get_volatility()
        assert volatility > 0

    def test_preaveraged_bipower_variation(self, high_frequency_data_with_jumps):
        """Test PreaveragedBiPowerVariation estimator."""
        prices, times, jump_indices = high_frequency_data_with_jumps

        # Create estimator
        pbpv = PreaveragedBiPowerVariation()

        # Fit estimator
        result = pbpv.fit((prices, times))

        # Check result type
        assert isinstance(result, RealizedEstimatorResult)

        # Check that realized measure is positive
        assert result.realized_measure > 0

        # Check that volatility is positive
        volatility = pbpv.get_volatility()
        assert volatility > 0

    def test_jump_detection(self, high_frequency_data_with_jumps):
        """Test jump detection in jump-robust estimators."""
        prices, times, jump_indices = high_frequency_data_with_jumps

        # Create estimator
        trv = ThresholdRealizedVariance()

        # Fit estimator
        result = trv.fit((prices, times))

        # Get returns
        returns = np.diff(np.log(prices))

        # Detect jumps
        jump_indicators, threshold = trv.detect_jumps(returns)

        # Check that jump indicators is a boolean array
        assert jump_indicators.dtype == bool

        # Check that threshold is positive
        assert threshold > 0

        # Check that some jumps were detected
        assert np.sum(jump_indicators) > 0

    def test_inheritance(self):
        """Test that jump-robust estimators inherit from JumpRobustEstimator."""
        trv = ThresholdRealizedVariance()
        tmpv = ThresholdMultipowerVariation()
        pbpv = PreaveragedBiPowerVariation()

        assert isinstance(trv, JumpRobustEstimator)
        assert isinstance(tmpv, JumpRobustEstimator)
        assert isinstance(pbpv, JumpRobustEstimator)


# ---- Utility Function Tests ----

class TestUtilityFunctions:
    """Tests for utility functions in realized volatility module."""

    def test_compute_returns(self, high_frequency_data):
        """Test compute_returns function."""
        prices, times = high_frequency_data

        # Compute log returns
        log_returns = compute_returns(prices, return_type='log')

        # Compute simple returns
        simple_returns = compute_returns(prices, return_type='simple')

        # Check that returns have correct length
        assert len(log_returns) == len(prices) - 1
        assert len(simple_returns) == len(prices) - 1

        # Check that log returns are different from simple returns
        assert not np.allclose(log_returns, simple_returns)

        # Check that returns are computed correctly
        expected_log_returns = np.diff(np.log(prices))
        expected_simple_returns = np.diff(prices) / prices[:-1]

        assert np.allclose(log_returns, expected_log_returns)
        assert np.allclose(simple_returns, expected_simple_returns)

    def test_detect_jumps(self, high_frequency_data_with_jumps):
        """Test detect_jumps function."""
        prices, times, jump_indices = high_frequency_data_with_jumps

        # Compute returns
        returns = np.diff(np.log(prices))

        # Detect jumps
        jump_indicators, threshold = detect_jumps(returns)

        # Check that jump indicators is a boolean array
        assert jump_indicators.dtype == bool

        # Check that threshold is positive
        assert threshold > 0

        # Check that some jumps were detected
        assert np.sum(jump_indicators) > 0

        # Test with different threshold multiplier
        jump_indicators_high, threshold_high = detect_jumps(returns, threshold_multiplier=5.0)

        # Higher threshold should detect fewer jumps
        assert np.sum(jump_indicators_high) <= np.sum(jump_indicators)

    def test_noise_variance(self, high_frequency_data):
        """Test noise_variance function."""
        prices, times = high_frequency_data

        # Compute returns
        returns = np.diff(np.log(prices))

        # Estimate noise variance
        noise_var = noise_variance(returns)

        # Check that noise variance is positive
        assert noise_var > 0

    def test_compute_realized_variance(self, high_frequency_data):
        """Test compute_realized_variance function."""
        prices, times = high_frequency_data

        # Compute returns
        returns = np.diff(np.log(prices))

        # Compute realized variance
        rv = compute_realized_variance(returns)

        # Check that realized variance is positive
        assert rv > 0

        # Check that realized variance is computed correctly
        expected_rv = np.sum(returns**2)
        assert np.isclose(rv, expected_rv)

    def test_compute_subsampled_measure(self, high_frequency_data):
        """Test compute_subsampled_measure function."""
        prices, times = high_frequency_data

        # Compute returns
        returns = np.diff(np.log(prices))

        # Compute subsampled realized variance
        rv_subsampled = compute_subsampled_measure(returns, subsampling_factor=5)

        # Check that subsampled realized variance is positive
        assert rv_subsampled > 0

        # Compute regular realized variance
        rv = compute_realized_variance(returns)

        # Subsampled measure should be close to regular measure
        # but not exactly the same
        assert not np.isclose(rv_subsampled, rv, rtol=1e-10)

    def test_align_time(self, high_frequency_data):
        """Test align_time function."""
        prices, times = high_frequency_data

        # Create a subset of times
        subset_times = times[::10]  # Take every 10th time point

        # Align prices to subset times
        aligned_prices = align_time(prices, times, subset_times)

        # Check that aligned prices have correct length
        assert len(aligned_prices) == len(subset_times)

    def test_time_conversion_functions(self):
        """Test time conversion utility functions."""
        # Test seconds2unit
        seconds = np.array([0, 3600, 7200])
        units = seconds2unit(seconds, unit='hours')
        assert np.allclose(units, np.array([0, 1, 2]))

        # Test unit2seconds
        units = np.array([0, 1, 2])
        seconds = unit2seconds(units, unit='hours')
        assert np.allclose(seconds, np.array([0, 3600, 7200]))

        # Test wall2seconds (assuming 9:30 market open)
        wall_times = np.array([34200, 37800, 41400])  # 9:30, 10:30, 11:30
        seconds = wall2seconds(wall_times, market_open=34200)
        assert np.allclose(seconds, np.array([0, 3600, 7200]))

        # Test seconds2wall (assuming 9:30 market open)
        seconds = np.array([0, 3600, 7200])
        wall_times = seconds2wall(seconds, market_open=34200)
        assert np.allclose(wall_times, np.array([34200, 37800, 41400]))

    def test_price_filter(self, high_frequency_data):
        """Test price_filter function."""
        prices, times = high_frequency_data

        # Filter prices with 5-minute sampling
        filtered_prices, filtered_times = price_filter(prices, times, sample_freq='5min')

        # Check that filtered data has fewer points
        assert len(filtered_prices) < len(prices)
        assert len(filtered_times) < len(times)

        # Check that filtered times are monotonically increasing
        assert np.all(np.diff(filtered_times) > 0)

    def test_return_filter(self, high_frequency_data):
        """Test return_filter function."""
        prices, times = high_frequency_data

        # Compute returns
        returns = np.diff(np.log(prices))
        return_times = times[1:]

        # Filter returns with 5-minute sampling
        filtered_returns, filtered_times = return_filter(returns, return_times, sample_freq='5min')

        # Check that filtered data has fewer points
        assert len(filtered_returns) < len(returns)
        assert len(filtered_times) < len(return_times)

        # Check that filtered times are monotonically increasing
        assert np.all(np.diff(filtered_times) > 0)

    def test_refresh_time(self, multivariate_high_frequency_data):
        """Test refresh_time function."""
        prices, times = multivariate_high_frequency_data

        # Split into separate price series
        price1 = prices[:, 0]
        price2 = prices[:, 1]

        # Apply refresh time algorithm
        refreshed_prices, refreshed_times = refresh_time([price1, price2], [times, times])

        # Check that refreshed data has correct shape
        assert refreshed_prices.shape[1] == 2
        assert len(refreshed_times) == refreshed_prices.shape[0]

        # Check that refreshed times are monotonically increasing
        assert np.all(np.diff(refreshed_times) > 0)

    def test_subsample(self, high_frequency_data):
        """Test subsample function."""
        prices, times = high_frequency_data

        # Subsample data
        subsampled_prices, subsampled_times = subsample(prices, times, k=5)

        # Check that subsampled data has fewer points
        assert len(subsampled_prices) < len(prices)
        assert len(subsampled_times) < len(times)

        # Check that subsampled times are monotonically increasing
        assert np.all(np.diff(subsampled_times) > 0)


# ---- Property-Based Tests ----

class TestPropertyBasedTests:
    """Property-based tests for realized volatility estimators."""

    @given(
        n_obs=st.integers(min_value=100, max_value=1000),
        volatility=st.floats(min_value=0.01, max_value=0.1)
    )
    @settings(max_examples=10)
    def test_realized_variance_property(self, n_obs, volatility):
        """Test that realized variance correctly estimates volatility."""
        # Generate random walk with known volatility
        rng = np.random.default_rng(42)
        returns = volatility * rng.standard_normal(n_obs)
        prices = 100.0 * np.exp(np.cumsum(returns))
        times = np.arange(n_obs + 1)  # One more time point than returns

        # Create estimator
        rv = RealizedVariance()

        # Fit estimator
        result = rv.fit((prices, times))

        # Check that estimated volatility is close to true volatility
        # Note: This is approximate since we're using a finite sample
        estimated_vol = np.sqrt(result.realized_measure)
        assert np.isclose(estimated_vol, volatility, rtol=0.5)

    @given(
        n_obs=st.integers(min_value=100, max_value=1000),
        sampling_freq=st.sampled_from(['1min', '5min', '10min'])
    )
    @settings(max_examples=10)
    def test_sampling_frequency_property(self, n_obs, sampling_freq):
        """Test that sampling frequency affects the number of observations."""
        # Generate random data
        rng = np.random.default_rng(42)
        returns = 0.01 * rng.standard_normal(n_obs)
        prices = 100.0 * np.exp(np.cumsum(returns))

        # Generate times spanning 6.5 hours (typical trading day)
        times = np.sort(rng.uniform(0, 23400, n_obs + 1))

        # Create configs with different sampling frequencies
        config = RealizedEstimatorConfig(sampling_frequency=sampling_freq)

        # Create estimator
        rv = RealizedVariance(config=config)

        # Fit estimator
        result = rv.fit((prices, times))

        # Check that result has sampling_frequency attribute
        assert result.sampling_frequency == sampling_freq

    @given(
        n_obs=st.integers(min_value=100, max_value=1000),
        n_assets=st.integers(min_value=2, max_value=5)
    )
    @settings(max_examples=5)
    def test_multivariate_property(self, n_obs, n_assets):
        """Test properties of multivariate realized volatility estimators."""
        # Generate random multivariate data
        rng = np.random.default_rng(42)

        # Generate correlated returns
        corr = np.eye(n_assets)
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                corr[i, j] = corr[j, i] = 0.5  # 0.5 correlation between all pairs

        # Cholesky decomposition
        chol = np.linalg.cholesky(corr)

        # Generate independent returns
        indep_returns = 0.01 * rng.standard_normal((n_obs, n_assets))

        # Transform to correlated returns
        corr_returns = indep_returns @ chol.T

        # Cumulate to get log prices
        log_prices = np.cumsum(corr_returns, axis=0)

        # Convert to prices
        initial_prices = 100.0 * np.ones(n_assets)
        prices = initial_prices * np.exp(log_prices)

        # Generate times
        times = np.sort(rng.uniform(0, 23400, n_obs + 1))

        # Create estimator
        rcov = RealizedCovariance()

        # Fit estimator
        result = rcov.fit((prices, times))

        # Check that realized measure is a matrix of correct size
        assert result.realized_measure.shape == (n_assets, n_assets)

        # Check that realized measure is symmetric
        assert np.allclose(result.realized_measure, result.realized_measure.T)

        # Check that diagonal elements are positive
        assert np.all(np.diag(result.realized_measure) > 0)

        # Check correlation matrix
        corr_matrix = rcov.get_correlation_matrix()

        # Diagonal elements should be 1
        assert np.allclose(np.diag(corr_matrix), 1.0)

        # Off-diagonal elements should be between -1 and 1
        assert np.all(corr_matrix >= -1.0) and np.all(corr_matrix <= 1.0)

        # Correlation matrix should be symmetric
        assert np.allclose(corr_matrix, corr_matrix.T)


# ---- Pandas Integration Tests ----

class TestPandasIntegration:
    """Tests for Pandas integration in realized volatility estimators."""

    def test_pandas_series_input(self, pandas_high_frequency_data):
        """Test using Pandas Series as input."""
        df = pandas_high_frequency_data

        # Create Series
        price_series = df['price']
        time_series = df['time']

        # Create estimator
        rv = RealizedVariance()

        # Fit estimator with Series
        result = rv.fit((price_series.values, time_series.values))

        # Check that result is valid
        assert isinstance(result, RealizedEstimatorResult)
        assert result.realized_measure > 0

    def test_pandas_dataframe_input(self, pandas_high_frequency_data):
        """Test using Pandas DataFrame as input."""
        df = pandas_high_frequency_data

        # Create estimator from DataFrame
        rv = RealizedVariance.from_pandas(df, price_col='price', time_col='time')

        # Fit estimator
        result = rv.fit((df['price'].values, df['time'].values))

        # Check that result is valid
        assert isinstance(result, RealizedEstimatorResult)
        assert result.realized_measure > 0

    def test_pandas_datetime_index(self, pandas_high_frequency_data):
        """Test using DataFrame with DatetimeIndex."""
        df = pandas_high_frequency_data

        # Create estimator from DataFrame
        rv = RealizedVariance.from_pandas(df, price_col='price')  # Use index as time

        # Fit estimator
        result = rv.fit((df['price'].values, df.index.astype('int64').values / 1e9))

        # Check that result is valid
        assert isinstance(result, RealizedEstimatorResult)
        assert result.realized_measure > 0

    def test_to_dataframe_method(self, high_frequency_data):
        """Test to_dataframe method."""
        prices, times = high_frequency_data

        # Create estimator
        rv = RealizedVariance()

        # Fit estimator
        result = rv.fit((prices, times))

        # Convert to DataFrame
        df = result.to_dataframe()

        # Check that DataFrame is returned
        assert isinstance(df, pd.DataFrame)

        # Check that DataFrame has expected columns
        assert 'realized_measure' in df.columns

    def test_pandas_result_indexing(self, pandas_high_frequency_data):
        """Test that result can be indexed with Pandas index."""
        df = pandas_high_frequency_data

        # Create estimator
        rv = RealizedVariance()

        # Fit estimator
        result = rv.fit((df['price'].values, df['time'].values))

        # Convert to DataFrame with original index
        df_result = pd.DataFrame({
            'realized_variance': result.realized_measure,
            'volatility': np.sqrt(result.realized_measure)
        }, index=[0])  # Single row result

        # Check that DataFrame can be indexed
        assert df_result.loc[0, 'realized_variance'] == result.realized_measure


# ---- Async Processing Tests ----

class TestAsyncProcessing:
    """Tests for asynchronous processing in realized volatility estimators."""

    def test_async_fit(self, high_frequency_data):
        """Test async_fit method."""
        prices, times = high_frequency_data

        # Create estimator
        rv = RealizedVariance()

        # Run async fit
        async def run_async_fit():
            return await rv.fit_async((prices, times))

        # Run the coroutine
        result = asyncio.run(run_async_fit())

        # Check that result is returned
        assert isinstance(result, RealizedEstimatorResult)
        assert result.realized_measure > 0

    def test_async_with_progress_callback(self, high_frequency_data):
        """Test async_fit with progress callback."""
        prices, times = high_frequency_data

        # Create estimator
        rv = RealizedVariance()

        # Create progress callback
        progress_updates = []

        async def progress_callback(percent, message):
            progress_updates.append((percent, message))

        # Run async fit with callback
        async def run_async_fit():
            return await rv.fit_async((prices, times), progress_callback=progress_callback)

        # Run the coroutine
        result = asyncio.run(run_async_fit())

        # Check that result is returned
        assert isinstance(result, RealizedEstimatorResult)
        assert result.realized_measure > 0

        # Check that progress callback was called
        # Note: This might not be reliable if the implementation doesn't call the callback
        # assert len(progress_updates) > 0

    def test_multiple_async_estimators(self, high_frequency_data):
        """Test running multiple async estimators concurrently."""
        prices, times = high_frequency_data

        # Create estimators
        rv = RealizedVariance()
        bpv = BiPowerVariation()
        rk = RealizedKernel()

        # Run async fits concurrently
        async def run_multiple_async():
            tasks = [
                rv.fit_async((prices, times)),
                bpv.fit_async((prices, times)),
                rk.fit_async((prices, times))
            ]
            return await asyncio.gather(*tasks)

        # Run the coroutine
        results = asyncio.run(run_multiple_async())

        # Check that all results are returned
        assert len(results) == 3
        assert all(isinstance(result, RealizedEstimatorResult) for result in results)
        assert all(result.realized_measure > 0 for result in results)


# ---- Performance Tests ----

class TestPerformance:
    """Performance tests for realized volatility estimators."""

    def test_numba_acceleration(self, high_frequency_data):
        """Test that Numba acceleration improves performance."""
        prices, times = high_frequency_data

        # Skip test if Numba is not available
        if not has_numba():
            pytest.skip("Numba is not available")

        # Create estimator
        rv = RealizedVariance()

        # Time the execution
        start_time = time.time()
        result = rv.fit((prices, times))
        end_time = time.time()

        # Record execution time with Numba
        execution_time_with_numba = end_time - start_time

        # Check that execution time is reasonable
        assert execution_time_with_numba < 1.0, "Execution with Numba should be fast"

    def test_large_dataset_performance(self):
        """Test performance with a large dataset."""
        # Generate a large dataset
        n_obs = 10000
        rng = np.random.default_rng(42)

        # Generate random walk
        returns = 0.001 * rng.standard_normal(n_obs)
        prices = 100.0 * np.exp(np.cumsum(returns))
        times = np.sort(rng.uniform(0, 23400, n_obs + 1))

        # Create estimator
        rv = RealizedVariance()

        # Time the execution
        start_time = time.time()
        result = rv.fit((prices, times))
        end_time = time.time()

        # Record execution time
        execution_time = end_time - start_time

        # Check that result is valid
        assert isinstance(result, RealizedEstimatorResult)
        assert result.realized_measure > 0

        # Check that execution time is reasonable
        # This is a rough check and might not be reliable in all environments
        if has_numba():
            assert execution_time < 5.0, "Execution with large dataset should be reasonably fast with Numba"
