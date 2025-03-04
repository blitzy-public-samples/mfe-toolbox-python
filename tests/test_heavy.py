'''
Tests for heavy-tailed distribution simulation and parameter estimation.

This module contains comprehensive tests for heavy-tailed distribution implementations
including Student's t, skewed t, and generalized error distributions. It validates
parameter estimation accuracy, simulation properties, likelihood computation, and
numerical stability during parameter estimation.
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
from scipy import stats, optimize

from mfe.models.distributions import normal, student_t, generalized_error, skewed_t
from mfe.models.distributions.utils import pvalue_calculator


class TestStudentTDistribution:
    """Tests for the Student's t-distribution implementation."""

    def test_pdf_computation(self, rng, assert_array_equal):
        """Test that PDF computation matches scipy implementation."""
        # Generate random data
        data = rng.standard_normal(1000)

        # Create MFE Student's t distribution with df=5
        t_dist = student_t.StudentT(df=5.0)

        # Compute PDF using MFE implementation
        pdf_mfe = t_dist.pdf(data)

        # Compute PDF using scipy
        pdf_scipy = stats.t.pdf(data, df=5.0)

        # Results should match
        assert_array_equal(pdf_mfe, pdf_scipy, rtol=1e-10,
                           err_msg="Student's t PDF differs from scipy implementation")

        # Test with pandas Series
        data_series = pd.Series(data)
        pdf_mfe_series = t_dist.pdf(data_series)

        # Results should be the same regardless of input type
        assert_array_equal(pdf_mfe, pdf_mfe_series, err_msg="PDF differs between array and Series")

    def test_cdf_computation(self, rng, assert_array_equal):
        """Test that CDF computation matches scipy implementation."""
        # Generate random data
        data = rng.standard_normal(1000)

        # Create MFE Student's t distribution with df=5
        t_dist = student_t.StudentT(df=5.0)

        # Compute CDF using MFE implementation
        cdf_mfe = t_dist.cdf(data)

        # Compute CDF using scipy
        cdf_scipy = stats.t.cdf(data, df=5.0)

        # Results should match
        assert_array_equal(cdf_mfe, cdf_scipy, rtol=1e-10,
                           err_msg="Student's t CDF differs from scipy implementation")

    def test_ppf_computation(self, rng, assert_array_equal):
        """Test that PPF (quantile) computation matches scipy implementation."""
        # Generate uniform random data between 0 and 1
        data = rng.uniform(0.01, 0.99, 1000)  # Avoid 0 and 1 for numerical stability

        # Create MFE Student's t distribution with df=5
        t_dist = student_t.StudentT(df=5.0)

        # Compute PPF using MFE implementation
        ppf_mfe = t_dist.ppf(data)

        # Compute PPF using scipy
        ppf_scipy = stats.t.ppf(data, df=5.0)

        # Results should match
        assert_array_equal(ppf_mfe, ppf_scipy, rtol=1e-10,
                           err_msg="Student's t PPF differs from scipy implementation")

    def test_random_generation(self, rng):
        """Test random number generation from Student's t distribution."""
        # Create MFE Student's t distribution with df=5
        t_dist = student_t.StudentT(df=5.0)

        # Generate random samples
        n_samples = 10000
        samples = t_dist.rvs(n_samples, random_state=rng)

        # Check sample size
        assert len(samples) == n_samples, "Random sample size doesn't match requested size"

        # Check basic statistical properties
        # For t(5), mean=0, variance=5/(5-2)=5/3
        assert abs(np.mean(samples)) < 0.1, "Sample mean too far from expected 0"
        assert abs(np.var(samples) - 5/3) < 0.2, "Sample variance too far from expected 5/3"

        # Test with size parameter as tuple
        samples_2d = t_dist.rvs(size=(100, 50), random_state=rng)
        assert samples_2d.shape == (100, 50), "Random sample shape doesn't match requested shape"

    def test_log_likelihood(self, rng, assert_array_equal):
        """Test log-likelihood computation for Student's t distribution."""
        # Generate t-distributed data
        data = stats.t.rvs(df=5.0, size=1000, random_state=rng)

        # Create MFE Student's t distribution with df=5
        t_dist = student_t.StudentT(df=5.0)

        # Compute log-likelihood using MFE implementation
        ll_mfe = t_dist.loglikelihood(data)

        # Compute log-likelihood manually using scipy
        ll_scipy = np.sum(stats.t.logpdf(data, df=5.0))

        # Results should match
        assert_array_equal(ll_mfe, ll_scipy, rtol=1e-10,
                           err_msg="Student's t log-likelihood differs from manual calculation")

    def test_parameter_estimation(self, rng):
        """Test parameter estimation for Student's t distribution."""
        # Generate t-distributed data with known df
        true_df = 5.0
        data = stats.t.rvs(df=true_df, size=5000, random_state=rng)

        # Define negative log-likelihood function for optimization
        def neg_ll(params):
            df = params[0]
            if df <= 2.0:  # Ensure df > 2 for finite variance
                return 1e10
            t_dist = student_t.StudentT(df=df)
            return -t_dist.loglikelihood(data)

        # Estimate parameters using SciPy's optimization
        result = optimize.minimize(neg_ll, x0=[8.0], method='L-BFGS-B', bounds=[(2.1, 20.0)])
        estimated_df = result.x[0]

        # Check if estimated df is close to true df
        assert abs(estimated_df - true_df) < 0.5, f"Estimated df={estimated_df} too far from true df={true_df}"

    def test_parameter_constraints(self):
        """Test that parameter constraints are enforced."""
        # df must be > 0
        with pytest.raises(ValueError, match="degrees of freedom must be positive"):
            student_t.StudentT(df=0.0)

        with pytest.raises(ValueError, match="degrees of freedom must be positive"):
            student_t.StudentT(df=-1.0)

    def test_edge_cases(self):
        """Test behavior at edge cases."""
        # Create distribution with very high df (approaches normal)
        high_df = 1000.0
        t_dist_high = student_t.StudentT(df=high_df)
        norm_dist = normal.Normal()

        # Test points
        x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])

        # PDF should be close to normal for high df
        pdf_t = t_dist_high.pdf(x)
        pdf_norm = norm_dist.pdf(x)
        assert np.allclose(pdf_t, pdf_norm, rtol=1e-2), "High df t-distribution not close to normal"

        # Create distribution with df just above 2 (finite but high variance)
        low_df = 2.1
        t_dist_low = student_t.StudentT(df=low_df)

        # PDF should be defined and positive
        pdf_low = t_dist_low.pdf(x)
        assert np.all(pdf_low > 0), "PDF not positive for low df"
        assert np.all(np.isfinite(pdf_low)), "PDF not finite for low df"

    @given(arrays(dtype=np.float64, shape=st.integers(10, 1000),
                  elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)))
    def test_property_based(self, data):
        """Property-based test for Student's t distribution using hypothesis."""
        # Create distribution with moderate df
        t_dist = student_t.StudentT(df=5.0)

        # PDF should be positive
        pdf = t_dist.pdf(data)
        assert np.all(pdf >= 0), "PDF returned negative values"

        # CDF should be between 0 and 1
        cdf = t_dist.cdf(data)
        assert np.all((cdf >= 0) & (cdf <= 1)), "CDF returned values outside [0,1]"

        # CDF should be monotonically increasing
        sorted_data = np.sort(data)
        sorted_cdf = t_dist.cdf(sorted_data)
        assert np.all(np.diff(sorted_cdf) >= 0), "CDF is not monotonically increasing"

    @pytest.mark.asyncio
    async def test_async_interface(self, rng):
        """Test the asynchronous interface for Student's t distribution."""
        # Create distribution
        t_dist = student_t.StudentT(df=5.0)

        # Generate data
        data = rng.standard_normal(1000)

        # Test async loglikelihood
        ll = await t_dist.loglikelihood_async(data)
        assert isinstance(ll, float), "Async loglikelihood did not return a float"

        # Test async random generation
        samples = await t_dist.rvs_async(1000, random_state=rng)
        assert len(samples) == 1000, "Async rvs did not return correct number of samples"

        # Test with progress callback
        progress_values = []

        async def progress_callback(percent, message):
            progress_values.append(percent)

        samples = await t_dist.rvs_async(1000, random_state=rng, progress_callback=progress_callback)
        assert len(samples) == 1000, "Async rvs with callback did not return correct number of samples"
        assert len(progress_values) > 0, "Progress callback was not called"

    def test_numba_acceleration(self, rng, assert_array_equal):
        """Test that Numba-accelerated functions produce correct results."""
        # Generate data
        data = rng.standard_normal(1000)

        # Create distribution
        t_dist = student_t.StudentT(df=5.0)

        # Get PDF using standard method
        pdf_standard = t_dist.pdf(data)

        # Get PDF using Numba-accelerated method (if available)
        # This assumes the implementation has a _pdf_numba method
        if hasattr(t_dist, '_pdf_numba'):
            pdf_numba = t_dist._pdf_numba(data)
            assert_array_equal(pdf_standard, pdf_numba, rtol=1e-10,
                               err_msg="Numba-accelerated PDF differs from standard implementation")


class TestSkewedTDistribution:
    """Tests for the skewed t-distribution implementation."""

    def test_pdf_computation(self, rng, assert_array_equal):
        """Test PDF computation for skewed t-distribution."""
        # Generate random data
        data = rng.standard_normal(1000)

        # Create MFE skewed t-distribution with df=5, skew=0.5
        skew_t_dist = skewed_t.SkewedT(df=5.0, skew=0.5)

        # Compute PDF
        pdf = skew_t_dist.pdf(data)

        # PDF should be positive
        assert np.all(pdf >= 0), "Skewed t PDF returned negative values"

        # Test with pandas Series
        data_series = pd.Series(data)
        pdf_series = skew_t_dist.pdf(data_series)

        # Results should be the same regardless of input type
        assert_array_equal(pdf, pdf_series, err_msg="PDF differs between array and Series")

        # Test that skew=0 gives symmetric t-distribution
        sym_t_dist = skewed_t.SkewedT(df=5.0, skew=0.0)
        std_t_dist = student_t.StudentT(df=5.0)

        pdf_skew0 = sym_t_dist.pdf(data)
        pdf_std_t = std_t_dist.pdf(data)

        assert_array_equal(pdf_skew0, pdf_std_t, rtol=1e-10,
                           err_msg="Skewed t with skew=0 differs from standard t")

    def test_cdf_computation(self, rng):
        """Test CDF computation for skewed t-distribution."""
        # Generate random data
        data = rng.standard_normal(1000)

        # Create MFE skewed t-distribution with df=5, skew=0.5
        skew_t_dist = skewed_t.SkewedT(df=5.0, skew=0.5)

        # Compute CDF
        cdf = skew_t_dist.cdf(data)

        # CDF should be between 0 and 1
        assert np.all((cdf >= 0) & (cdf <= 1)), "Skewed t CDF returned values outside [0,1]"

        # CDF should be monotonically increasing
        sorted_data = np.sort(data)
        sorted_cdf = skew_t_dist.cdf(sorted_data)
        assert np.all(np.diff(sorted_cdf) >= 0), "CDF is not monotonically increasing"

    def test_ppf_computation(self, rng):
        """Test PPF (quantile) computation for skewed t-distribution."""
        # Generate uniform random data between 0 and 1
        data = rng.uniform(0.01, 0.99, 1000)  # Avoid 0 and 1 for numerical stability

        # Create MFE skewed t-distribution with df=5, skew=0.5
        skew_t_dist = skewed_t.SkewedT(df=5.0, skew=0.5)

        # Compute PPF
        ppf = skew_t_dist.ppf(data)

        # PPF should be finite
        assert np.all(np.isfinite(ppf)), "Skewed t PPF returned non-finite values"

        # PPF should be the inverse of CDF
        cdf_of_ppf = skew_t_dist.cdf(ppf)
        assert np.allclose(cdf_of_ppf, data, rtol=1e-5, atol=1e-5), "PPF is not inverse of CDF"

    def test_random_generation(self, rng):
        """Test random number generation from skewed t-distribution."""
        # Create MFE skewed t-distribution with df=5, skew=0.5
        skew_t_dist = skewed_t.SkewedT(df=5.0, skew=0.5)

        # Generate random samples
        n_samples = 10000
        samples = skew_t_dist.rvs(n_samples, random_state=rng)

        # Check sample size
        assert len(samples) == n_samples, "Random sample size doesn't match requested size"

        # Check skewness
        sample_skewness = stats.skew(samples)
        assert sample_skewness > 0, "Sample skewness should be positive for skew=0.5"

        # Test with size parameter as tuple
        samples_2d = skew_t_dist.rvs(size=(100, 50), random_state=rng)
        assert samples_2d.shape == (100, 50), "Random sample shape doesn't match requested shape"

        # Test that skew=0 gives symmetric distribution
        sym_t_dist = skewed_t.SkewedT(df=5.0, skew=0.0)
        sym_samples = sym_t_dist.rvs(n_samples, random_state=rng)
        sym_skewness = stats.skew(sym_samples)
        assert abs(sym_skewness) < 0.1, "Sample skewness should be close to 0 for skew=0"

    def test_log_likelihood(self, rng):
        """Test log-likelihood computation for skewed t-distribution."""
        # Create MFE skewed t-distribution with df=5, skew=0.5
        skew_t_dist = skewed_t.SkewedT(df=5.0, skew=0.5)

        # Generate random samples from the distribution
        samples = skew_t_dist.rvs(1000, random_state=rng)

        # Compute log-likelihood
        ll = skew_t_dist.loglikelihood(samples)

        # Log-likelihood should be finite
        assert np.isfinite(ll), "Log-likelihood is not finite"

        # Log-likelihood should be higher for correct parameters than for incorrect ones
        wrong_dist = skewed_t.SkewedT(df=10.0, skew=-0.5)
        wrong_ll = wrong_dist.loglikelihood(samples)

        assert ll > wrong_ll, "Log-likelihood not higher for correct parameters"

    def test_parameter_estimation(self, rng):
        """Test parameter estimation for skewed t-distribution."""
        # True parameters
        true_df = 5.0
        true_skew = 0.5

        # Generate skewed t-distributed data
        skew_t_dist = skewed_t.SkewedT(df=true_df, skew=true_skew)
        data = skew_t_dist.rvs(5000, random_state=rng)

        # Define negative log-likelihood function for optimization
        def neg_ll(params):
            df, skew = params
            if df <= 2.0:  # Ensure df > 2 for finite variance
                return 1e10
            dist = skewed_t.SkewedT(df=df, skew=skew)
            return -dist.loglikelihood(data)

        # Estimate parameters using SciPy's optimization
        result = optimize.minimize(neg_ll, x0=[8.0, 0.0], method='L-BFGS-B',
                                   bounds=[(2.1, 20.0), (-0.99, 0.99)])
        estimated_df, estimated_skew = result.x

        # Check if estimated parameters are close to true parameters
        assert abs(estimated_df - true_df) < 1.0, f"Estimated df={estimated_df} too far from true df={true_df}"
        assert abs(estimated_skew -
                   true_skew) < 0.2, f"Estimated skew={estimated_skew} too far from true skew={true_skew}"

    def test_parameter_constraints(self):
        """Test that parameter constraints are enforced."""
        # df must be > 0
        with pytest.raises(ValueError, match="degrees of freedom must be positive"):
            skewed_t.SkewedT(df=0.0, skew=0.5)

        with pytest.raises(ValueError, match="degrees of freedom must be positive"):
            skewed_t.SkewedT(df=-1.0, skew=0.5)

        # skew must be between -1 and 1
        with pytest.raises(ValueError, match="skewness parameter must be between -1 and 1"):
            skewed_t.SkewedT(df=5.0, skew=-1.5)

        with pytest.raises(ValueError, match="skewness parameter must be between -1 and 1"):
            skewed_t.SkewedT(df=5.0, skew=1.5)

    def test_edge_cases(self):
        """Test behavior at edge cases."""
        # Create distribution with very high df (approaches skewed normal)
        high_df = 1000.0
        skew_t_dist = skewed_t.SkewedT(df=high_df, skew=0.5)

        # Test points
        x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])

        # PDF should be defined and positive
        pdf = skew_t_dist.pdf(x)
        assert np.all(pdf > 0), "PDF not positive for high df"
        assert np.all(np.isfinite(pdf)), "PDF not finite for high df"

        # Create distribution with skew close to limits
        near_limit_skew = 0.99
        skew_t_near_limit = skewed_t.SkewedT(df=5.0, skew=near_limit_skew)

        # PDF should be defined and positive
        pdf_limit = skew_t_near_limit.pdf(x)
        assert np.all(pdf_limit > 0), "PDF not positive for near-limit skew"
        assert np.all(np.isfinite(pdf_limit)), "PDF not finite for near-limit skew"

    @given(arrays(dtype=np.float64, shape=st.integers(10, 1000),
                  elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)))
    def test_property_based(self, data):
        """Property-based test for skewed t-distribution using hypothesis."""
        # Create distribution with moderate parameters
        skew_t_dist = skewed_t.SkewedT(df=5.0, skew=0.5)

        # PDF should be positive
        pdf = skew_t_dist.pdf(data)
        assert np.all(pdf >= 0), "PDF returned negative values"

        # CDF should be between 0 and 1
        cdf = skew_t_dist.cdf(data)
        assert np.all((cdf >= 0) & (cdf <= 1)), "CDF returned values outside [0,1]"

        # CDF should be monotonically increasing
        sorted_data = np.sort(data)
        sorted_cdf = skew_t_dist.cdf(sorted_data)
        assert np.all(np.diff(sorted_cdf) >= 0), "CDF is not monotonically increasing"

    @pytest.mark.asyncio
    async def test_async_interface(self, rng):
        """Test the asynchronous interface for skewed t-distribution."""
        # Create distribution
        skew_t_dist = skewed_t.SkewedT(df=5.0, skew=0.5)

        # Generate data
        data = rng.standard_normal(1000)

        # Test async loglikelihood
        ll = await skew_t_dist.loglikelihood_async(data)
        assert isinstance(ll, float), "Async loglikelihood did not return a float"

        # Test async random generation
        samples = await skew_t_dist.rvs_async(1000, random_state=rng)
        assert len(samples) == 1000, "Async rvs did not return correct number of samples"

        # Test with progress callback
        progress_values = []

        async def progress_callback(percent, message):
            progress_values.append(percent)

        samples = await skew_t_dist.rvs_async(1000, random_state=rng, progress_callback=progress_callback)
        assert len(samples) == 1000, "Async rvs with callback did not return correct number of samples"
        assert len(progress_values) > 0, "Progress callback was not called"

    def test_numba_acceleration(self, rng, assert_array_equal):
        """Test that Numba-accelerated functions produce correct results."""
        # Generate data
        data = rng.standard_normal(1000)

        # Create distribution
        skew_t_dist = skewed_t.SkewedT(df=5.0, skew=0.5)

        # Get PDF using standard method
        pdf_standard = skew_t_dist.pdf(data)

        # Get PDF using Numba-accelerated method (if available)
        if hasattr(skew_t_dist, '_pdf_numba'):
            pdf_numba = skew_t_dist._pdf_numba(data)
            assert_array_equal(pdf_standard, pdf_numba, rtol=1e-10,
                               err_msg="Numba-accelerated PDF differs from standard implementation")


class TestGeneralizedErrorDistribution:
    """Tests for the Generalized Error Distribution (GED) implementation."""

    def test_pdf_computation(self, rng, assert_array_equal):
        """Test PDF computation for GED."""
        # Generate random data
        data = rng.standard_normal(1000)

        # Create MFE GED with nu=1.5
        ged_dist = generalized_error.GeneralizedError(nu=1.5)

        # Compute PDF
        pdf = ged_dist.pdf(data)

        # PDF should be positive
        assert np.all(pdf >= 0), "GED PDF returned negative values"

        # Test with pandas Series
        data_series = pd.Series(data)
        pdf_series = ged_dist.pdf(data_series)

        # Results should be the same regardless of input type
        assert_array_equal(pdf, pdf_series, err_msg="PDF differs between array and Series")

        # Test that nu=2 gives normal distribution
        normal_ged = generalized_error.GeneralizedError(nu=2.0)
        norm_dist = normal.Normal()

        pdf_nu2 = normal_ged.pdf(data)
        pdf_norm = norm_dist.pdf(data)

        assert_array_equal(pdf_nu2, pdf_norm, rtol=1e-10,
                           err_msg="GED with nu=2 differs from normal distribution")

    def test_cdf_computation(self, rng):
        """Test CDF computation for GED."""
        # Generate random data
        data = rng.standard_normal(1000)

        # Create MFE GED with nu=1.5
        ged_dist = generalized_error.GeneralizedError(nu=1.5)

        # Compute CDF
        cdf = ged_dist.cdf(data)

        # CDF should be between 0 and 1
        assert np.all((cdf >= 0) & (cdf <= 1)), "GED CDF returned values outside [0,1]"

        # CDF should be monotonically increasing
        sorted_data = np.sort(data)
        sorted_cdf = ged_dist.cdf(sorted_data)
        assert np.all(np.diff(sorted_cdf) >= 0), "CDF is not monotonically increasing"

        # Test that nu=2 gives normal distribution
        normal_ged = generalized_error.GeneralizedError(nu=2.0)
        norm_dist = normal.Normal()

        cdf_nu2 = normal_ged.cdf(data)
        cdf_norm = norm_dist.cdf(data)

        assert np.allclose(cdf_nu2, cdf_norm, rtol=1e-10), "GED with nu=2 CDF differs from normal distribution"

    def test_ppf_computation(self, rng):
        """Test PPF (quantile) computation for GED."""
        # Generate uniform random data between 0 and 1
        data = rng.uniform(0.01, 0.99, 1000)  # Avoid 0 and 1 for numerical stability

        # Create MFE GED with nu=1.5
        ged_dist = generalized_error.GeneralizedError(nu=1.5)

        # Compute PPF
        ppf = ged_dist.ppf(data)

        # PPF should be finite
        assert np.all(np.isfinite(ppf)), "GED PPF returned non-finite values"

        # PPF should be the inverse of CDF
        cdf_of_ppf = ged_dist.cdf(ppf)
        assert np.allclose(cdf_of_ppf, data, rtol=1e-5, atol=1e-5), "PPF is not inverse of CDF"

        # Test that nu=2 gives normal distribution
        normal_ged = generalized_error.GeneralizedError(nu=2.0)
        norm_dist = normal.Normal()

        ppf_nu2 = normal_ged.ppf(data)
        ppf_norm = norm_dist.ppf(data)

        assert np.allclose(ppf_nu2, ppf_norm, rtol=1e-10), "GED with nu=2 PPF differs from normal distribution"

    def test_random_generation(self, rng):
        """Test random number generation from GED."""
        # Create MFE GED with nu=1.5
        ged_dist = generalized_error.GeneralizedError(nu=1.5)

        # Generate random samples
        n_samples = 10000
        samples = ged_dist.rvs(n_samples, random_state=rng)

        # Check sample size
        assert len(samples) == n_samples, "Random sample size doesn't match requested size"

        # Check basic statistical properties
        # For GED, mean=0, variance=1
        assert abs(np.mean(samples)) < 0.1, "Sample mean too far from expected 0"
        assert abs(np.var(samples) - 1.0) < 0.1, "Sample variance too far from expected 1"

        # Test with size parameter as tuple
        samples_2d = ged_dist.rvs(size=(100, 50), random_state=rng)
        assert samples_2d.shape == (100, 50), "Random sample shape doesn't match requested shape"

        # Test kurtosis for different nu values
        # nu < 2: leptokurtic (heavy tails)
        # nu = 2: mesokurtic (normal)
        # nu > 2: platykurtic (light tails)
        ged_heavy = generalized_error.GeneralizedError(nu=1.0)  # Laplace distribution
        ged_normal = generalized_error.GeneralizedError(nu=2.0)  # Normal distribution
        ged_light = generalized_error.GeneralizedError(nu=5.0)   # Light-tailed distribution

        samples_heavy = ged_heavy.rvs(n_samples, random_state=rng)
        samples_normal = ged_normal.rvs(n_samples, random_state=rng)
        samples_light = ged_light.rvs(n_samples, random_state=rng)

        kurt_heavy = stats.kurtosis(samples_heavy, fisher=True)  # Excess kurtosis
        kurt_normal = stats.kurtosis(samples_normal, fisher=True)
        kurt_light = stats.kurtosis(samples_light, fisher=True)

        assert kurt_heavy > 1.0, "Heavy-tailed GED (nu=1) should have positive excess kurtosis"
        assert abs(kurt_normal) < 0.5, "Normal GED (nu=2) should have near-zero excess kurtosis"
        assert kurt_light < -0.5, "Light-tailed GED (nu=5) should have negative excess kurtosis"

    def test_log_likelihood(self, rng):
        """Test log-likelihood computation for GED."""
        # Create MFE GED with nu=1.5
        ged_dist = generalized_error.GeneralizedError(nu=1.5)

        # Generate random samples from the distribution
        samples = ged_dist.rvs(1000, random_state=rng)

        # Compute log-likelihood
        ll = ged_dist.loglikelihood(samples)

        # Log-likelihood should be finite
        assert np.isfinite(ll), "Log-likelihood is not finite"

        # Log-likelihood should be higher for correct parameters than for incorrect ones
        wrong_dist = generalized_error.GeneralizedError(nu=3.0)
        wrong_ll = wrong_dist.loglikelihood(samples)

        assert ll > wrong_ll, "Log-likelihood not higher for correct parameters"

    def test_parameter_estimation(self, rng):
        """Test parameter estimation for GED."""
        # True parameter
        true_nu = 1.5

        # Generate GED data
        ged_dist = generalized_error.GeneralizedError(nu=true_nu)
        data = ged_dist.rvs(5000, random_state=rng)

        # Define negative log-likelihood function for optimization
        def neg_ll(params):
            nu = params[0]
            if nu <= 0.0:  # Ensure nu > 0
                return 1e10
            dist = generalized_error.GeneralizedError(nu=nu)
            return -dist.loglikelihood(data)

        # Estimate parameters using SciPy's optimization
        result = optimize.minimize(neg_ll, x0=[2.0], method='L-BFGS-B', bounds=[(0.1, 10.0)])
        estimated_nu = result.x[0]

        # Check if estimated nu is close to true nu
        assert abs(estimated_nu - true_nu) < 0.2, f"Estimated nu={estimated_nu} too far from true nu={true_nu}"

    def test_parameter_constraints(self):
        """Test that parameter constraints are enforced."""
        # nu must be > 0
        with pytest.raises(ValueError, match="shape parameter must be positive"):
            generalized_error.GeneralizedError(nu=0.0)

        with pytest.raises(ValueError, match="shape parameter must be positive"):
            generalized_error.GeneralizedError(nu=-1.0)

    def test_edge_cases(self):
        """Test behavior at edge cases."""
        # Create distribution with nu close to 0 (very heavy tails)
        low_nu = 0.1
        ged_low = generalized_error.GeneralizedError(nu=low_nu)

        # Test points
        x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])

        # PDF should be defined and positive
        pdf_low = ged_low.pdf(x)
        assert np.all(pdf_low > 0), "PDF not positive for low nu"
        assert np.all(np.isfinite(pdf_low)), "PDF not finite for low nu"

        # Create distribution with high nu (very light tails)
        high_nu = 10.0
        ged_high = generalized_error.GeneralizedError(nu=high_nu)

        # PDF should be defined and positive
        pdf_high = ged_high.pdf(x)
        assert np.all(pdf_high > 0), "PDF not positive for high nu"
        assert np.all(np.isfinite(pdf_high)), "PDF not finite for high nu"

    @given(arrays(dtype=np.float64, shape=st.integers(10, 1000),
                  elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)))
    def test_property_based(self, data):
        """Property-based test for GED using hypothesis."""
        # Create distribution with moderate parameters
        ged_dist = generalized_error.GeneralizedError(nu=1.5)

        # PDF should be positive
        pdf = ged_dist.pdf(data)
        assert np.all(pdf >= 0), "PDF returned negative values"

        # CDF should be between 0 and 1
        cdf = ged_dist.cdf(data)
        assert np.all((cdf >= 0) & (cdf <= 1)), "CDF returned values outside [0,1]"

        # CDF should be monotonically increasing
        sorted_data = np.sort(data)
        sorted_cdf = ged_dist.cdf(sorted_data)
        assert np.all(np.diff(sorted_cdf) >= 0), "CDF is not monotonically increasing"

    @pytest.mark.asyncio
    async def test_async_interface(self, rng):
        """Test the asynchronous interface for GED."""
        # Create distribution
        ged_dist = generalized_error.GeneralizedError(nu=1.5)

        # Generate data
        data = rng.standard_normal(1000)

        # Test async loglikelihood
        ll = await ged_dist.loglikelihood_async(data)
        assert isinstance(ll, float), "Async loglikelihood did not return a float"

        # Test async random generation
        samples = await ged_dist.rvs_async(1000, random_state=rng)
        assert len(samples) == 1000, "Async rvs did not return correct number of samples"

        # Test with progress callback
        progress_values = []

        async def progress_callback(percent, message):
            progress_values.append(percent)

        samples = await ged_dist.rvs_async(1000, random_state=rng, progress_callback=progress_callback)
        assert len(samples) == 1000, "Async rvs with callback did not return correct number of samples"
        assert len(progress_values) > 0, "Progress callback was not called"

    def test_numba_acceleration(self, rng, assert_array_equal):
        """Test that Numba-accelerated functions produce correct results."""
        # Generate data
        data = rng.standard_normal(1000)

        # Create distribution
        ged_dist = generalized_error.GeneralizedError(nu=1.5)

        # Get PDF using standard method
        pdf_standard = ged_dist.pdf(data)

        # Get PDF using Numba-accelerated method (if available)
        if hasattr(ged_dist, '_pdf_numba'):
            pdf_numba = ged_dist._pdf_numba(data)
            assert_array_equal(pdf_standard, pdf_numba, rtol=1e-10,
                               err_msg="Numba-accelerated PDF differs from standard implementation")


class TestDistributionIntegration:
    """Integration tests for heavy-tailed distributions."""

    def test_distribution_fitting_workflow(self, rng):
        """Test the complete workflow of fitting a heavy-tailed distribution to data."""
        # Generate t-distributed data
        true_df = 5.0
        data = stats.t.rvs(df=true_df, size=5000, random_state=rng)

        # Convert to pandas Series to test DataFrame integration
        data_series = pd.Series(data, name='returns')

        # Define distributions to test
        distributions = [
            student_t.StudentT(df=4.0),  # Initial guess
            skewed_t.SkewedT(df=4.0, skew=0.0),  # Initial guess
            generalized_error.GeneralizedError(nu=1.5),  # Initial guess
            normal.Normal()  # For comparison
        ]

        # Compute log-likelihood for each distribution
        log_likelihoods = []
        for dist in distributions:
            ll = dist.loglikelihood(data_series)
            log_likelihoods.append(ll)

        # Student's t should have higher likelihood than normal for t-distributed data
        assert log_likelihoods[0] > log_likelihoods[3], "Student's t should fit t-distributed data better than normal"

        # Fit Student's t distribution
        def neg_ll_t(params):
            df = params[0]
            if df <= 2.0:  # Ensure df > 2 for finite variance
                return 1e10
            dist = student_t.StudentT(df=df)
            return -dist.loglikelihood(data_series)

        result_t = optimize.minimize(neg_ll_t, x0=[4.0], method='L-BFGS-B', bounds=[(2.1, 20.0)])
        estimated_df = result_t.x[0]

        # Check if estimated df is close to true df
        assert abs(estimated_df - true_df) < 0.5, f"Estimated df={estimated_df} too far from true df={true_df}"

        # Compute fitted log-likelihood
        fitted_t_dist = student_t.StudentT(df=estimated_df)
        fitted_ll = fitted_t_dist.loglikelihood(data_series)

        # Fitted distribution should have higher likelihood than initial guess
        assert fitted_ll > log_likelihoods[0], "Fitted distribution should have higher likelihood than initial guess"

    @pytest.mark.asyncio
    async def test_async_parameter_estimation(self, rng):
        """Test asynchronous parameter estimation for heavy-tailed distributions."""
        # Generate t-distributed data
        true_df = 5.0
        data = stats.t.rvs(df=true_df, size=5000, random_state=rng)

        # Define async negative log-likelihood function
        async def neg_ll_async(params):
            df = params[0]
            if df <= 2.0:  # Ensure df > 2 for finite variance
                return 1e10
            dist = student_t.StudentT(df=df)
            ll = await dist.loglikelihood_async(data)
            return -ll

        # Track optimization progress
        progress_values = []

        async def progress_callback(percent, message):
            progress_values.append(percent)

        # Define wrapper for SciPy's minimize to work with async functions
        async def minimize_async(func, x0, bounds):
            # Simple grid search as an example of async optimization
            best_x = x0
            best_val = await func(x0)

            # Try 10 points in the parameter space
            for i in range(10):
                # Generate point within bounds
                df = bounds[0][0] + (bounds[0][1] - bounds[0][0]) * (i / 9)
                val = await func([df])

                # Report progress
                if progress_callback:
                    await progress_callback((i + 1) / 10 * 100, f"Evaluating df={df:.2f}")

                if val < best_val:
                    best_val = val
                    best_x = [df]

            # Return result similar to SciPy's minimize
            class Result:
                def __init__(self, x, fun):
                    self.x = x
                    self.fun = fun

            return Result(best_x, best_val)

        # Perform async parameter estimation
        result = await minimize_async(neg_ll_async, [4.0], [(2.1, 20.0)])
        estimated_df = result.x[0]

        # Check if estimated df is reasonable (less precise due to simple grid search)
        assert 3.0 < estimated_df < 8.0, f"Estimated df={estimated_df} too far from true df={true_df}"

        # Check that progress callback was called
        assert len(progress_values) > 0, "Progress callback was not called during async estimation"

    def test_pandas_integration(self, rng):
        """Test integration with pandas DataFrames and Series."""
        # Generate data
        n = 1000
        data = rng.standard_normal(n)

        # Create pandas Series and DataFrame
        series = pd.Series(data, name='returns')
        dates = pd.date_range('2020-01-01', periods=n)
        df = pd.DataFrame({'returns': data}, index=dates)

        # Create distributions
        t_dist = student_t.StudentT(df=5.0)
        skew_t_dist = skewed_t.SkewedT(df=5.0, skew=0.5)
        ged_dist = generalized_error.GeneralizedError(nu=1.5)

        # Test with Series
        pdf_series_t = t_dist.pdf(series)
        pdf_series_skew_t = skew_t_dist.pdf(series)
        pdf_series_ged = ged_dist.pdf(series)

        assert isinstance(pdf_series_t, pd.Series), "PDF with Series input should return Series"
        assert isinstance(pdf_series_skew_t, pd.Series), "PDF with Series input should return Series"
        assert isinstance(pdf_series_ged, pd.Series), "PDF with Series input should return Series"

        assert pdf_series_t.name == 'returns', "Series name should be preserved"
        assert len(pdf_series_t) == n, "Series length should be preserved"

        # Test with DataFrame column
        pdf_df_t = t_dist.pdf(df['returns'])
        pdf_df_skew_t = skew_t_dist.pdf(df['returns'])
        pdf_df_ged = ged_dist.pdf(df['returns'])

        assert isinstance(pdf_df_t, pd.Series), "PDF with DataFrame column should return Series"
        assert isinstance(pdf_df_skew_t, pd.Series), "PDF with DataFrame column should return Series"
        assert isinstance(pdf_df_ged, pd.Series), "PDF with DataFrame column should return Series"

        assert pdf_df_t.name == 'returns', "Series name should be preserved"
        assert len(pdf_df_t) == n, "Series length should be preserved"

        # Test that index is preserved
        assert np.all(pdf_df_t.index == dates), "Series index should be preserved"

    def test_numba_performance(self, rng):
        """Test performance improvement from Numba acceleration."""
        # Skip if Numba is not available or functions don't have Numba versions
        t_dist = student_t.StudentT(df=5.0)
        if not hasattr(t_dist, '_pdf_numba'):
            pytest.skip("Numba-accelerated functions not available")

        # Generate large dataset for performance testing
        data = rng.standard_normal(100000)

        # Time standard implementation
        import time

        start_standard = time.time()
        pdf_standard = t_dist.pdf(data)
        end_standard = time.time()
        standard_time = end_standard - start_standard

        # Time Numba implementation
        start_numba = time.time()
        pdf_numba = t_dist._pdf_numba(data)
        end_numba = time.time()
        numba_time = end_numba - start_numba

        # Numba should be faster (allowing for JIT compilation overhead on first run)
        # This is a soft assertion as performance can vary by environment
        if numba_time > standard_time:
            warnings.warn(f"Numba implementation ({numba_time:.4f}s) not faster than standard ({standard_time:.4f}s)")

        # Results should be identical
        assert np.allclose(pdf_standard, pdf_numba), "Numba and standard implementations give different results"

    def test_model_confidence_set(self, rng):
        """Test Model Confidence Set procedure with heavy-tailed distributions."""
        # Generate t-distributed data
        true_df = 5.0
        data = stats.t.rvs(df=true_df, size=1000, random_state=rng)

        # Create distributions with different parameters
        distributions = [
            student_t.StudentT(df=3.0),
            student_t.StudentT(df=5.0),  # True model
            student_t.StudentT(df=7.0),
            skewed_t.SkewedT(df=5.0, skew=0.3),
            generalized_error.GeneralizedError(nu=1.5),
            normal.Normal()
        ]

        # Compute negative log-likelihoods (loss function)
        losses = np.zeros((len(distributions), len(data)))
        for i, dist in enumerate(distributions):
            pdf = dist.pdf(data)
            # Individual observation negative log-likelihoods
            losses[i, :] = -np.log(np.maximum(pdf, 1e-10))  # Avoid log(0)

        # Compute pairwise loss differences
        n_models = len(distributions)
        loss_diffs = np.zeros((n_models, n_models, len(data)))
        for i in range(n_models):
            for j in range(n_models):
                if i != j:
                    loss_diffs[i, j, :] = losses[i, :] - losses[j, :]

        # Compute t-statistics for each pair
        t_stats = np.zeros((n_models, n_models))
        for i in range(n_models):
            for j in range(n_models):
                if i != j:
                    mean_diff = np.mean(loss_diffs[i, j, :])
                    std_diff = np.std(loss_diffs[i, j, :], ddof=1)
                    t_stats[i, j] = mean_diff / (std_diff / np.sqrt(len(data)))

        # Compute p-values
        p_values = np.zeros((n_models, n_models))
        for i in range(n_models):
            for j in range(n_models):
                if i != j:
                    p_values[i, j] = 2 * (1 - stats.t.cdf(abs(t_stats[i, j]), df=len(data)-1))

        # True model (index 1) should have low p-values against misspecified models
        assert p_values[5, 1] < 0.05, "Normal should be rejected against true t-distribution"

        # Models close to true model should have higher p-values
        assert p_values[2, 1] > 0.01, "t(df=7) should not be strongly rejected against t(df=5)"


class TestDistributionProperties:
    """Tests for theoretical properties of heavy-tailed distributions."""

    @given(st.floats(min_value=2.1, max_value=100.0))  # df > 2 for finite variance
    def test_student_t_properties(self, df):
        """Test theoretical properties of Student's t-distribution."""
        # Create distribution
        t_dist = student_t.StudentT(df=df)

        # Symmetry: PDF(-x) = PDF(x)
        x_values = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
        pdf_neg = t_dist.pdf(-x_values)
        pdf_pos = t_dist.pdf(x_values)
        assert np.allclose(pdf_neg, pdf_pos[::-1]), "Student's t PDF is not symmetric"

        # CDF properties: CDF(-x) = 1 - CDF(x)
        cdf_neg = t_dist.cdf(-x_values)
        cdf_pos = t_dist.cdf(x_values)
        assert np.allclose(cdf_neg, 1 - cdf_pos[::-1]), "Student's t CDF doesn't satisfy CDF(-x) = 1 - CDF(x)"

        # Median is 0
        assert abs(t_dist.ppf(0.5)) < 1e-10, "Student's t median is not 0"

        # Variance is df/(df-2) for df > 2
        # Test by generating samples and checking variance
        samples = t_dist.rvs(10000, random_state=np.random.default_rng(12345))
        expected_var = df / (df - 2)
        sample_var = np.var(samples)
        assert abs(sample_var - expected_var) / expected_var < 0.1, "Sample variance doesn't match theoretical variance"

    @given(st.floats(min_value=2.1, max_value=100.0),  # df > 2 for finite variance
           st.floats(min_value=-0.9, max_value=0.9))   # skew in (-1, 1)
    def test_skewed_t_properties(self, df, skew):
        """Test theoretical properties of skewed t-distribution."""
        # Create distribution
        skew_t_dist = skewed_t.SkewedT(df=df, skew=skew)

        # CDF properties: CDF should be between 0 and 1
        x_values = np.array([-10.0, -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0])
        cdf_values = skew_t_dist.cdf(x_values)
        assert np.all((cdf_values >= 0) & (cdf_values <= 1)), "Skewed t CDF outside [0,1]"

        # CDF should be monotonically increasing
        assert np.all(np.diff(cdf_values) > 0), "Skewed t CDF is not monotonically increasing"

        # When skew=0, should reduce to Student's t
        if abs(skew) < 1e-10:
            t_dist = student_t.StudentT(df=df)
            pdf_skew_t = skew_t_dist.pdf(x_values)
            pdf_t = t_dist.pdf(x_values)
            assert np.allclose(pdf_skew_t, pdf_t), "Skewed t with skew=0 doesn't match Student's t"

        # Test skewness direction
        # For positive skew, median should be negative
        # For negative skew, median should be positive
        if abs(skew) > 0.1:  # Only test if skew is not too close to 0
            median = skew_t_dist.ppf(0.5)
            if skew > 0:
                assert median < 0, "Median should be negative for positive skew"
            else:
                assert median > 0, "Median should be positive for negative skew"

    @given(st.floats(min_value=0.5, max_value=10.0))  # nu > 0
    def test_ged_properties(self, nu):
        """Test theoretical properties of Generalized Error Distribution."""
        # Create distribution
        ged_dist = generalized_error.GeneralizedError(nu=nu)

        # Symmetry: PDF(-x) = PDF(x)
        x_values = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
        pdf_neg = ged_dist.pdf(-x_values)
        pdf_pos = ged_dist.pdf(x_values)
        assert np.allclose(pdf_neg, pdf_pos[::-1]), "GED PDF is not symmetric"

        # CDF properties: CDF(-x) = 1 - CDF(x)
        cdf_neg = ged_dist.cdf(-x_values)
        cdf_pos = ged_dist.cdf(x_values)
        assert np.allclose(cdf_neg, 1 - cdf_pos[::-1]), "GED CDF doesn't satisfy CDF(-x) = 1 - CDF(x)"

        # Median is 0
        assert abs(ged_dist.ppf(0.5)) < 1e-10, "GED median is not 0"

        # Variance should be 1 (standardized GED)
        samples = ged_dist.rvs(10000, random_state=np.random.default_rng(12345))
        sample_var = np.var(samples)
        assert abs(sample_var - 1.0) < 0.1, "Sample variance doesn't match theoretical variance of 1"

        # When nu=2, should reduce to standard normal
        if abs(nu - 2.0) < 1e-10:
            norm_dist = normal.Normal()
            pdf_ged = ged_dist.pdf(x_values)
            pdf_norm = norm_dist.pdf(x_values)
            assert np.allclose(pdf_ged, pdf_norm), "GED with nu=2 doesn't match normal distribution"


if __name__ == "__main__":
    pytest.main()
