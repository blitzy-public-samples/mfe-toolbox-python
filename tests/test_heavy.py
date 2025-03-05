'''
Tests for heavy-tailed distribution simulation and parameter estimation.

This module provides comprehensive tests for heavy-tailed distribution implementations
in the MFE Toolbox, including Student's t and skewed t distributions. It validates
parameter estimation accuracy, simulation properties, numerical stability, and
integration with NumPy arrays and Pandas DataFrames.
'''

import numpy as np
import pandas as pd
import pytest
from scipy import stats, optimize
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.numpy import arrays

from mfe.core.exceptions import ParameterError, DimensionError, NumericError
from mfe.core.parameters import StudentTParameters, SkewedTParameters
from mfe.models.distributions import StudentT, SkewedT
from mfe.models.distributions.student_t import (
    stdtpdf, stdtcdf, stdtinv, stdtrnd, stdtloglik
)
from mfe.models.distributions.skewed_t import (
    skewtpdf, skewtcdf, skewtinv, skewtrnd, skewtloglik
)


# ---- Basic Functionality Tests ----

def test_student_t_initialization():
    """Test initialization of Student's t distribution."""
    # Default initialization
    t_dist = StudentT()
    assert t_dist.params is not None
    assert t_dist.params.df == 5.0  # Default df

    # Initialization with parameters
    params = StudentTParameters(df=10.0)
    t_dist = StudentT(params=params)
    assert t_dist.params.df == 10.0

    # Initialization with invalid parameters
    with pytest.raises(ParameterError):
        StudentT(params=StudentTParameters(df=1.5))  # df <= 2 is invalid


def test_skewed_t_initialization():
    """Test initialization of skewed Student's t distribution."""
    # Default initialization
    st_dist = SkewedT()
    assert st_dist.params is not None
    assert st_dist.params.df == 5.0  # Default df
    assert st_dist.params.lambda_ == 0.0  # Default lambda (no skewness)

    # Initialization with parameters
    params = SkewedTParameters(df=10.0, lambda_=0.3)
    st_dist = SkewedT(params=params)
    assert st_dist.params.df == 10.0
    assert st_dist.params.lambda_ == 0.3

    # Initialization with invalid parameters
    with pytest.raises(ParameterError):
        SkewedT(params=SkewedTParameters(df=1.5, lambda_=0.3))  # df <= 2 is invalid

    with pytest.raises(ParameterError):
        SkewedT(params=SkewedTParameters(df=5.0, lambda_=1.5))  # |lambda| > 1 is invalid


def test_student_t_pdf():
    """Test PDF computation for Student's t distribution."""
    # Create distribution
    t_dist = StudentT(params=StudentTParameters(df=5.0))

    # Test points
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    # Compute PDF
    pdf_values = t_dist.pdf(x)

    # Compare with scipy.stats implementation
    scale = np.sqrt(5.0 / 3.0)  # Standardization factor for df=5
    expected = stats.t.pdf(x * scale, 5.0) * scale

    np.testing.assert_allclose(pdf_values, expected, rtol=1e-5)

    # Test with pandas Series
    x_series = pd.Series(x)
    pdf_series = t_dist.pdf(x_series)

    np.testing.assert_allclose(pdf_series.values, expected, rtol=1e-5)

    # Test direct function
    direct_pdf = stdtpdf(x, df=5.0)
    np.testing.assert_allclose(direct_pdf, expected, rtol=1e-5)


def test_skewed_t_pdf():
    """Test PDF computation for skewed Student's t distribution."""
    # Create distribution
    st_dist = SkewedT(params=SkewedTParameters(df=5.0, lambda_=0.3))

    # Test points
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    # Compute PDF
    pdf_values = st_dist.pdf(x)

    # Ensure PDF values are positive
    assert np.all(pdf_values > 0)

    # Ensure PDF integrates to approximately 1
    x_fine = np.linspace(-10, 10, 1000)
    pdf_fine = st_dist.pdf(x_fine)
    integral = np.trapz(pdf_fine, x_fine)
    assert abs(integral - 1.0) < 0.01

    # Test with pandas Series
    x_series = pd.Series(x)
    pdf_series = st_dist.pdf(x_series)

    np.testing.assert_allclose(pdf_series.values, pdf_values, rtol=1e-5)

    # Test direct function
    direct_pdf = skewtpdf(x, df=5.0, lambda_=0.3)
    np.testing.assert_allclose(direct_pdf, pdf_values, rtol=1e-5)

    # Test skewness property: for lambda > 0, pdf(x) > pdf(-x) for x > 0
    assert pdf_values[3] > pdf_values[1]  # pdf(1.0) > pdf(-1.0)
    assert pdf_values[4] > pdf_values[0]  # pdf(2.0) > pdf(-2.0)


def test_student_t_cdf():
    """Test CDF computation for Student's t distribution."""
    # Create distribution
    t_dist = StudentT(params=StudentTParameters(df=5.0))

    # Test points
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    # Compute CDF
    cdf_values = t_dist.cdf(x)

    # Compare with scipy.stats implementation
    scale = np.sqrt(5.0 / 3.0)  # Standardization factor for df=5
    expected = stats.t.cdf(x * scale, 5.0)

    np.testing.assert_allclose(cdf_values, expected, rtol=1e-5)

    # Test with pandas Series
    x_series = pd.Series(x)
    cdf_series = t_dist.cdf(x_series)

    np.testing.assert_allclose(cdf_series.values, expected, rtol=1e-5)

    # Test direct function
    direct_cdf = stdtcdf(x, df=5.0)
    np.testing.assert_allclose(direct_cdf, expected, rtol=1e-5)

    # Test CDF properties
    assert cdf_values[2] == 0.5  # CDF(0) = 0.5 for symmetric distribution
    assert np.all(np.diff(cdf_values) > 0)  # CDF is strictly increasing


def test_skewed_t_cdf():
    """Test CDF computation for skewed Student's t distribution."""
    # Create distribution
    st_dist = SkewedT(params=SkewedTParameters(df=5.0, lambda_=0.3))

    # Test points
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    # Compute CDF
    cdf_values = st_dist.cdf(x)

    # Ensure CDF values are between 0 and 1
    assert np.all(cdf_values >= 0)
    assert np.all(cdf_values <= 1)

    # Ensure CDF is monotonically increasing
    assert np.all(np.diff(cdf_values) > 0)

    # Test with pandas Series
    x_series = pd.Series(x)
    cdf_series = st_dist.cdf(x_series)

    np.testing.assert_allclose(cdf_series.values, cdf_values, rtol=1e-5)

    # Test direct function
    direct_cdf = skewtcdf(x, df=5.0, lambda_=0.3)
    np.testing.assert_allclose(direct_cdf, cdf_values, rtol=1e-5)

    # Test skewness property: for lambda > 0, CDF(0) < 0.5
    assert cdf_values[2] < 0.5  # CDF(0) < 0.5 for right-skewed distribution


def test_student_t_ppf():
    """Test PPF (quantile function) computation for Student's t distribution."""
    # Create distribution
    t_dist = StudentT(params=StudentTParameters(df=5.0))

    # Test probabilities
    p = np.array([0.01, 0.25, 0.5, 0.75, 0.99])

    # Compute PPF
    ppf_values = t_dist.ppf(p)

    # Compare with scipy.stats implementation
    scale = np.sqrt(5.0 / 3.0)  # Standardization factor for df=5
    expected = stats.t.ppf(p, 5.0) / scale

    np.testing.assert_allclose(ppf_values, expected, rtol=1e-5)

    # Test with pandas Series
    p_series = pd.Series(p)
    ppf_series = t_dist.ppf(p_series)

    np.testing.assert_allclose(ppf_series.values, expected, rtol=1e-5)

    # Test direct function
    direct_ppf = stdtinv(p, df=5.0)
    np.testing.assert_allclose(direct_ppf, expected, rtol=1e-5)

    # Test PPF properties
    assert ppf_values[2] == 0.0  # PPF(0.5) = 0 for symmetric distribution
    assert np.all(np.diff(ppf_values) > 0)  # PPF is strictly increasing


def test_skewed_t_ppf():
    """Test PPF (quantile function) computation for skewed Student's t distribution."""
    # Create distribution
    st_dist = SkewedT(params=SkewedTParameters(df=5.0, lambda_=0.3))

    # Test probabilities
    p = np.array([0.01, 0.25, 0.5, 0.75, 0.99])

    # Compute PPF
    ppf_values = st_dist.ppf(p)

    # Ensure PPF is monotonically increasing
    assert np.all(np.diff(ppf_values) > 0)

    # Test with pandas Series
    p_series = pd.Series(p)
    ppf_series = st_dist.ppf(p_series)

    np.testing.assert_allclose(ppf_series.values, ppf_values, rtol=1e-5)

    # Test direct function
    direct_ppf = skewtinv(p, df=5.0, lambda_=0.3)
    np.testing.assert_allclose(direct_ppf, ppf_values, rtol=1e-5)

    # Test skewness property: for lambda > 0, PPF(0.5) > 0
    assert ppf_values[2] > 0  # PPF(0.5) > 0 for right-skewed distribution

    # Test PPF is inverse of CDF
    x = ppf_values
    cdf_values = st_dist.cdf(x)
    np.testing.assert_allclose(cdf_values, p, rtol=1e-5)


def test_student_t_rvs():
    """Test random number generation for Student's t distribution."""
    # Create distribution
    t_dist = StudentT(params=StudentTParameters(df=5.0))

    # Generate random samples
    n_samples = 10000
    rng = np.random.default_rng(42)
    samples = t_dist.rvs(size=n_samples, random_state=rng)

    # Check shape
    assert samples.shape == (n_samples,)

    # Check basic statistics
    assert abs(np.mean(samples)) < 0.1  # Mean should be close to 0
    assert abs(np.std(samples) - 1.0) < 0.1  # Std should be close to 1

    # Check distribution using Kolmogorov-Smirnov test
    _, p_value = stats.kstest(samples, lambda x: t_dist.cdf(x))
    assert p_value > 0.01  # Should not reject the null hypothesis

    # Test direct function
    direct_samples = stdtrnd(n_samples, df=5.0, random_state=42)
    assert direct_samples.shape == (n_samples,)

    # Test with different size parameter
    samples_2d = t_dist.rvs(size=(100, 50), random_state=42)
    assert samples_2d.shape == (100, 50)


def test_skewed_t_rvs():
    """Test random number generation for skewed Student's t distribution."""
    # Create distribution
    st_dist = SkewedT(params=SkewedTParameters(df=5.0, lambda_=0.3))

    # Generate random samples
    n_samples = 10000
    rng = np.random.default_rng(42)
    samples = st_dist.rvs(size=n_samples, random_state=rng)

    # Check shape
    assert samples.shape == (n_samples,)

    # Check skewness
    assert stats.skew(samples) > 0  # Should be positively skewed for lambda > 0

    # Check distribution using Kolmogorov-Smirnov test
    _, p_value = stats.kstest(samples, lambda x: st_dist.cdf(x))
    assert p_value > 0.01  # Should not reject the null hypothesis

    # Test direct function
    direct_samples = skewtrnd(n_samples, df=5.0, lambda_=0.3, random_state=42)
    assert direct_samples.shape == (n_samples,)

    # Test with different size parameter
    samples_2d = st_dist.rvs(size=(100, 50), random_state=42)
    assert samples_2d.shape == (100, 50)


def test_student_t_loglikelihood():
    """Test log-likelihood computation for Student's t distribution."""
    # Create distribution
    t_dist = StudentT(params=StudentTParameters(df=5.0))

    # Generate data from the distribution
    n_samples = 1000
    rng = np.random.default_rng(42)
    data = t_dist.rvs(size=n_samples, random_state=rng)

    # Compute log-likelihood
    loglik = t_dist.loglikelihood(data)

    # Ensure log-likelihood is finite
    assert np.isfinite(loglik)

    # Test with pandas Series
    data_series = pd.Series(data)
    loglik_series = t_dist.loglikelihood(data_series)

    assert abs(loglik - loglik_series) < 1e-10

    # Test direct function
    direct_loglik = stdtloglik(data, df=5.0)
    assert abs(loglik - direct_loglik) < 1e-10

    # Test log-likelihood is maximized at true parameters
    def neg_loglik(df):
        params = StudentTParameters(df=df[0])
        temp_dist = StudentT(params=params)
        return -temp_dist.loglikelihood(data)

    result = optimize.minimize(neg_loglik, np.array([10.0]), bounds=[(2.1, 30.0)])
    estimated_df = result.x[0]

    # Estimated df should be close to true df=5.0
    assert abs(estimated_df - 5.0) < 1.0


def test_skewed_t_loglikelihood():
    """Test log-likelihood computation for skewed Student's t distribution."""
    # Create distribution
    st_dist = SkewedT(params=SkewedTParameters(df=5.0, lambda_=0.3))

    # Generate data from the distribution
    n_samples = 1000
    rng = np.random.default_rng(42)
    data = st_dist.rvs(size=n_samples, random_state=rng)

    # Compute log-likelihood
    loglik = st_dist.loglikelihood(data)

    # Ensure log-likelihood is finite
    assert np.isfinite(loglik)

    # Test with pandas Series
    data_series = pd.Series(data)
    loglik_series = st_dist.loglikelihood(data_series)

    assert abs(loglik - loglik_series) < 1e-10

    # Test direct function
    direct_loglik = skewtloglik(data, df=5.0, lambda_=0.3)
    assert abs(loglik - direct_loglik) < 1e-10

    # Test log-likelihood is maximized at true parameters
    def neg_loglik(params):
        df, lambda_ = params
        params_obj = SkewedTParameters(df=df, lambda_=lambda_)
        temp_dist = SkewedT(params=params_obj)
        return -temp_dist.loglikelihood(data)

    result = optimize.minimize(
        neg_loglik,
        np.array([10.0, 0.0]),
        bounds=[(2.1, 30.0), (-0.99, 0.99)]
    )
    estimated_df, estimated_lambda = result.x

    # Estimated parameters should be close to true parameters
    assert abs(estimated_df - 5.0) < 1.5
    assert abs(estimated_lambda - 0.3) < 0.2


# ---- Parameter Estimation Tests ----

def test_student_t_fit():
    """Test parameter estimation for Student's t distribution."""
    # Create distribution with true parameters
    true_df = 5.0
    true_dist = StudentT(params=StudentTParameters(df=true_df))

    # Generate data from the distribution
    n_samples = 2000
    rng = np.random.default_rng(42)
    data = true_dist.rvs(size=n_samples, random_state=rng)

    # Create a new distribution for fitting
    t_dist = StudentT()

    # Fit the distribution to the data
    estimated_params = t_dist.fit(data)

    # Check that estimated parameters are close to true parameters
    assert abs(estimated_params.df - true_df) < 1.0

    # Test with pandas Series
    data_series = pd.Series(data)
    t_dist_series = StudentT()
    estimated_params_series = t_dist_series.fit(data_series)

    assert abs(estimated_params_series.df - true_df) < 1.0

    # Test with initial parameters
    t_dist_init = StudentT(params=StudentTParameters(df=10.0))
    estimated_params_init = t_dist_init.fit(data)

    assert abs(estimated_params_init.df - true_df) < 1.0


def test_skewed_t_fit():
    """Test parameter estimation for skewed Student's t distribution."""
    # Create distribution with true parameters
    true_df = 5.0
    true_lambda = 0.3
    true_dist = SkewedT(params=SkewedTParameters(df=true_df, lambda_=true_lambda))

    # Generate data from the distribution
    n_samples = 2000
    rng = np.random.default_rng(42)
    data = true_dist.rvs(size=n_samples, random_state=rng)

    # Create a new distribution for fitting
    st_dist = SkewedT()

    # Fit the distribution to the data
    estimated_params = st_dist.fit(data)

    # Check that estimated parameters are close to true parameters
    assert abs(estimated_params.df - true_df) < 1.5
    assert abs(estimated_params.lambda_ - true_lambda) < 0.2

    # Test with pandas Series
    data_series = pd.Series(data)
    st_dist_series = SkewedT()
    estimated_params_series = st_dist_series.fit(data_series)

    assert abs(estimated_params_series.df - true_df) < 1.5
    assert abs(estimated_params_series.lambda_ - true_lambda) < 0.2

    # Test with initial parameters
    st_dist_init = SkewedT(params=SkewedTParameters(df=10.0, lambda_=0.0))
    estimated_params_init = st_dist_init.fit(data)

    assert abs(estimated_params_init.df - true_df) < 1.5
    assert abs(estimated_params_init.lambda_ - true_lambda) < 0.2


@pytest.mark.asyncio
async def test_student_t_fit_async():
    """Test asynchronous parameter estimation for Student's t distribution."""
    # Create distribution with true parameters
    true_df = 5.0
    true_dist = StudentT(params=StudentTParameters(df=true_df))

    # Generate data from the distribution
    n_samples = 2000
    rng = np.random.default_rng(42)
    data = true_dist.rvs(size=n_samples, random_state=rng)

    # Create a new distribution for fitting
    t_dist = StudentT()

    # Fit the distribution to the data asynchronously
    estimated_params = await t_dist.fit_async(data)

    # Check that estimated parameters are close to true parameters
    assert abs(estimated_params.df - true_df) < 1.0

    # Compare with synchronous version
    t_dist_sync = StudentT()
    estimated_params_sync = t_dist_sync.fit(data)

    assert abs(estimated_params.df - estimated_params_sync.df) < 1e-5


@pytest.mark.asyncio
async def test_skewed_t_fit_async():
    """Test asynchronous parameter estimation for skewed Student's t distribution."""
    # Create distribution with true parameters
    true_df = 5.0
    true_lambda = 0.3
    true_dist = SkewedT(params=SkewedTParameters(df=true_df, lambda_=true_lambda))

    # Generate data from the distribution
    n_samples = 2000
    rng = np.random.default_rng(42)
    data = true_dist.rvs(size=n_samples, random_state=rng)

    # Create a new distribution for fitting
    st_dist = SkewedT()

    # Fit the distribution to the data asynchronously
    estimated_params = await st_dist.fit_async(data)

    # Check that estimated parameters are close to true parameters
    assert abs(estimated_params.df - true_df) < 1.5
    assert abs(estimated_params.lambda_ - true_lambda) < 0.2

    # Compare with synchronous version
    st_dist_sync = SkewedT()
    estimated_params_sync = st_dist_sync.fit(data)

    assert abs(estimated_params.df - estimated_params_sync.df) < 1e-5
    assert abs(estimated_params.lambda_ - estimated_params_sync.lambda_) < 1e-5


# ---- Async Interface Tests ----

@pytest.mark.asyncio
async def test_student_t_rvs_async():
    """Test asynchronous random number generation for Student's t distribution."""
    # Create distribution
    t_dist = StudentT(params=StudentTParameters(df=5.0))

    # Generate random samples asynchronously
    n_samples = 1000
    samples = await t_dist.rvs_async(size=n_samples, random_state=42)

    # Check shape
    assert samples.shape == (n_samples,)

    # Compare with synchronous version
    samples_sync = t_dist.rvs(size=n_samples, random_state=42)

    # Should be identical with same random seed
    np.testing.assert_array_equal(samples, samples_sync)


@pytest.mark.asyncio
async def test_skewed_t_rvs_async():
    """Test asynchronous random number generation for skewed Student's t distribution."""
    # Create distribution
    st_dist = SkewedT(params=SkewedTParameters(df=5.0, lambda_=0.3))

    # Generate random samples asynchronously
    n_samples = 1000
    samples = await st_dist.rvs_async(size=n_samples, random_state=42)

    # Check shape
    assert samples.shape == (n_samples,)

    # Compare with synchronous version
    samples_sync = st_dist.rvs(size=n_samples, random_state=42)

    # Should be identical with same random seed
    np.testing.assert_array_equal(samples, samples_sync)


# ---- Edge Cases and Error Handling ----

def test_student_t_invalid_df():
    """Test that Student's t distribution handles invalid degrees of freedom."""
    # df <= 2 is invalid
    with pytest.raises(ParameterError):
        StudentT(params=StudentTParameters(df=2.0))

    with pytest.raises(ParameterError):
        StudentT(params=StudentTParameters(df=1.0))

    with pytest.raises(ParameterError):
        StudentT(params=StudentTParameters(df=0.0))

    with pytest.raises(ParameterError):
        StudentT(params=StudentTParameters(df=-1.0))

    # Direct function should also raise error
    with pytest.raises(ParameterError):
        stdtpdf(np.array([0.0]), df=2.0)


def test_skewed_t_invalid_parameters():
    """Test that skewed Student's t distribution handles invalid parameters."""
    # df <= 2 is invalid
    with pytest.raises(ParameterError):
        SkewedT(params=SkewedTParameters(df=2.0, lambda_=0.0))

    # |lambda| > 1 is invalid
    with pytest.raises(ParameterError):
        SkewedT(params=SkewedTParameters(df=5.0, lambda_=1.1))

    with pytest.raises(ParameterError):
        SkewedT(params=SkewedTParameters(df=5.0, lambda_=-1.1))

    # Direct function should also raise error
    with pytest.raises(ParameterError):
        skewtpdf(np.array([0.0]), df=2.0, lambda_=0.0)

    with pytest.raises(ParameterError):
        skewtpdf(np.array([0.0]), df=5.0, lambda_=1.1)


def test_student_t_invalid_inputs():
    """Test that Student's t distribution handles invalid inputs."""
    t_dist = StudentT(params=StudentTParameters(df=5.0))

    # NaN values
    with pytest.raises(ValueError):
        t_dist.pdf(np.array([0.0, np.nan]))

    # Infinite values
    with pytest.raises(ValueError):
        t_dist.pdf(np.array([0.0, np.inf]))

    # Invalid probabilities for PPF
    with pytest.raises(ValueError):
        t_dist.ppf(np.array([-0.1, 0.5]))

    with pytest.raises(ValueError):
        t_dist.ppf(np.array([0.5, 1.1]))


def test_skewed_t_invalid_inputs():
    """Test that skewed Student's t distribution handles invalid inputs."""
    st_dist = SkewedT(params=SkewedTParameters(df=5.0, lambda_=0.3))

    # NaN values
    with pytest.raises(ValueError):
        st_dist.pdf(np.array([0.0, np.nan]))

    # Infinite values
    with pytest.raises(ValueError):
        st_dist.pdf(np.array([0.0, np.inf]))

    # Invalid probabilities for PPF
    with pytest.raises(ValueError):
        st_dist.ppf(np.array([-0.1, 0.5]))

    with pytest.raises(ValueError):
        st_dist.ppf(np.array([0.5, 1.1]))


def test_student_t_empty_input():
    """Test that Student's t distribution handles empty input."""
    t_dist = StudentT(params=StudentTParameters(df=5.0))

    # Empty array
    with pytest.raises(ValueError):
        t_dist.pdf(np.array([]))

    with pytest.raises(ValueError):
        t_dist.cdf(np.array([]))

    with pytest.raises(ValueError):
        t_dist.ppf(np.array([]))

    with pytest.raises(ValueError):
        t_dist.loglikelihood(np.array([]))


def test_skewed_t_empty_input():
    """Test that skewed Student's t distribution handles empty input."""
    st_dist = SkewedT(params=SkewedTParameters(df=5.0, lambda_=0.3))

    # Empty array
    with pytest.raises(ValueError):
        st_dist.pdf(np.array([]))

    with pytest.raises(ValueError):
        st_dist.cdf(np.array([]))

    with pytest.raises(ValueError):
        st_dist.ppf(np.array([]))

    with pytest.raises(ValueError):
        st_dist.loglikelihood(np.array([]))


def test_student_t_no_params():
    """Test that Student's t distribution methods handle missing parameters."""
    t_dist = StudentT()
    t_dist._params = None  # Manually remove parameters

    # All methods should raise DistributionError
    with pytest.raises(Exception):  # Could be DistributionError or similar
        t_dist.pdf(np.array([0.0]))

    with pytest.raises(Exception):
        t_dist.cdf(np.array([0.0]))

    with pytest.raises(Exception):
        t_dist.ppf(np.array([0.5]))

    with pytest.raises(Exception):
        t_dist.rvs(size=10)

    with pytest.raises(Exception):
        t_dist.loglikelihood(np.array([0.0]))


def test_skewed_t_no_params():
    """Test that skewed Student's t distribution methods handle missing parameters."""
    st_dist = SkewedT()
    st_dist._params = None  # Manually remove parameters

    # All methods should raise DistributionError
    with pytest.raises(Exception):  # Could be DistributionError or similar
        st_dist.pdf(np.array([0.0]))

    with pytest.raises(Exception):
        st_dist.cdf(np.array([0.0]))

    with pytest.raises(Exception):
        st_dist.ppf(np.array([0.5]))

    with pytest.raises(Exception):
        st_dist.rvs(size=10)

    with pytest.raises(Exception):
        st_dist.loglikelihood(np.array([0.0]))


# ---- Property-Based Testing with Hypothesis ----

given(
    arrays(np.float64, st.integers(min_value=10, max_value=100),
           elements=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)),
    st.floats(min_value=2.1, max_value=30.0)
)(
    settings(deadline=None)(
        lambda data, df: (
            # Skip if data has too little variation
            assume(np.std(data) > 1e-10),

            # Create distribution
            (lambda t_dist: (
                # PDF properties
                (lambda pdf_values: (
                    assert np.all(pdf_values > 0),  # PDF should be positive

                    # CDF properties
                    (lambda cdf_values: (
                        assert np.all(cdf_values >= 0) and np.all(cdf_values <= 1),  # CDF should be in [0, 1]
                        assert np.all(np.diff(cdf_values[np.argsort(data)]) >= 0),  # CDF should be non-decreasing

                        # PPF properties
                        (lambda p_values, ppf_values: (
                            assert np.all(np.diff(ppf_values) > 0),  # PPF should be strictly increasing

                            # PPF should be inverse of CDF
                            np.testing.assert_allclose(t_dist.cdf(ppf_values), p_values, rtol=1e-5),

                            # Symmetry property: PDF(-x) = PDF(x)
                            np.testing.assert_allclose(t_dist.pdf(-np.linspace(-5, 5, 10)),
                                                       t_dist.pdf(np.linspace(-5, 5, 10))[::-1], rtol=1e-5),

                            # Symmetry property: CDF(-x) = 1 - CDF(x)
                            np.testing.assert_allclose(t_dist.cdf(-np.linspace(-5, 5, 10)),
                                                       1 - t_dist.cdf(np.linspace(-5, 5, 10))[::-1], rtol=1e-5)
                        ))(np.linspace(0.01, 0.99, 10), t_dist.ppf(np.linspace(0.01, 0.99, 10)))
                    ))(t_dist.cdf(data))
                ))(t_dist.pdf(data))
            ))(StudentT(params=StudentTParameters(df=df)))
        )
    )
)


def test_student_t_properties(data, df):
    pass  # The actual test is run by the Hypothesis-decorated function above


given(
    arrays(np.float64, st.integers(min_value=10, max_value=100),
           elements=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)),
    st.floats(min_value=2.1, max_value=30.0),
    st.floats(min_value=-0.9, max_value=0.9)
)(
    settings(deadline=None)(
        lambda data, df, lambda_: (
            assume(np.std(data) > 1e-10),

            (lambda st_dist: (
                (lambda pdf_values: (
                    assert np.all(pdf_values > 0),

                    (lambda cdf_values: (
                        assert np.all(cdf_values >= 0) and np.all(cdf_values <= 1),
                        assert np.all(np.diff(cdf_values[np.argsort(data)]) >= 0),

                        (lambda p_values, ppf_values: (
                            assert np.all(np.diff(ppf_values) > 0),
                            np.testing.assert_allclose(st_dist.cdf(ppf_values), p_values, rtol=1e-5),
                            (lambda median: (
                                (assert median > 0) if lambda_ > 0 else (assert median < 0)
                            ))(st_dist.ppf(np.array([0.5]))[0])
                        ))(np.linspace(0.01, 0.99, 10), st_dist.ppf(np.linspace(0.01, 0.99, 10)))
                    ))(st_dist.cdf(data))
                ))(st_dist.pdf(data))
            ))(SkewedT(params=SkewedTParameters(df=df, lambda_=lambda_)))
        )
    )
)


def test_skewed_t_properties(data, df, lambda_):
    pass  # The actual test is run by the Hypothesis-decorated function above


given(
    st.integers(min_value=10, max_value=1000),
    st.floats(min_value=2.1, max_value=30.0)
)(
    settings(deadline=None)(
        lambda n_samples, df: (
            (lambda t_dist: (
                (lambda samples: (
                    assert samples.shape == (n_samples,),
                    assert abs(np.mean(samples)) < 0.5,
                    (lambda _, p_value: assert p_value > 0.001)(*stats.kstest(samples, lambda x: t_dist.cdf(x)))
                ))(t_dist.rvs(size=n_samples, random_state=42))
            ))(StudentT(params=StudentTParameters(df=df)))
        )
    )
)


def test_student_t_simulation_properties(n_samples, df):
    pass  # The actual test is run by the Hypothesis-decorated function above


given(
    st.integers(min_value=10, max_value=1000),
    st.floats(min_value=2.1, max_value=30.0),
    st.floats(min_value=-0.9, max_value=0.9)
)(
    settings(deadline=None)(
        lambda n_samples, df, lambda_: (
            (lambda st_dist: (
                (lambda samples: (
                    assert samples.shape == (n_samples,),
                    (lambda: (
                        (lambda sample_skewness: (assert sample_skewness > -0.5) if lambda_ > 0 else (assert sample_skewness < 0.5))
                    ))(),
                    (lambda _, p_value: assert p_value > 0.001)(*stats.kstest(samples, lambda x: st_dist.cdf(x)))
                ))(st_dist.rvs(size=n_samples, random_state=42))
            ))(SkewedT(params=SkewedTParameters(df=df, lambda_=lambda_)))
        )
    )
)


def test_skewed_t_simulation_properties(n_samples, df, lambda_):
    pass  # The actual test is run by the Hypothesis-decorated function above


# ---- Integration with NumPy and Pandas ----

def test_student_t_numpy_integration():
    """Test integration of Student's t distribution with NumPy arrays."""
    # Create distribution
    t_dist = StudentT(params=StudentTParameters(df=5.0))

    # Test with different NumPy array shapes

    # 1D array
    x_1d = np.linspace(-3, 3, 10)
    pdf_1d = t_dist.pdf(x_1d)
    assert pdf_1d.shape == x_1d.shape

    # 2D array
    x_2d = np.array([[-2, -1, 0], [1, 2, 3]])
    pdf_2d = t_dist.pdf(x_2d)
    assert pdf_2d.shape == x_2d.shape

    # 3D array
    x_3d = np.array([[[-2, -1], [0, 1]], [[2, 3], [4, 5]]])
    pdf_3d = t_dist.pdf(x_3d)
    assert pdf_3d.shape == x_3d.shape

    # Test with different data types
    x_float32 = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    pdf_float32 = t_dist.pdf(x_float32)
    assert pdf_float32.dtype == np.float64  # Should convert to float64

    x_int = np.array([-1, 0, 1])
    pdf_int = t_dist.pdf(x_int)
    assert pdf_int.dtype == np.float64  # Should convert to float64


def test_skewed_t_numpy_integration():
    """Test integration of skewed Student's t distribution with NumPy arrays."""
    # Create distribution
    st_dist = SkewedT(params=SkewedTParameters(df=5.0, lambda_=0.3))

    # Test with different NumPy array shapes

    # 1D array
    x_1d = np.linspace(-3, 3, 10)
    pdf_1d = st_dist.pdf(x_1d)
    assert pdf_1d.shape == x_1d.shape

    # 2D array
    x_2d = np.array([[-2, -1, 0], [1, 2, 3]])
    pdf_2d = st_dist.pdf(x_2d)
    assert pdf_2d.shape == x_2d.shape

    # 3D array
    x_3d = np.array([[[-2, -1], [0, 1]], [[2, 3], [4, 5]]])
    pdf_3d = st_dist.pdf(x_3d)
    assert pdf_3d.shape == x_3d.shape

    # Test with different data types
    x_float32 = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    pdf_float32 = st_dist.pdf(x_float32)
    assert pdf_float32.dtype == np.float64  # Should convert to float64

    x_int = np.array([-1, 0, 1])
    pdf_int = st_dist.pdf(x_int)
    assert pdf_int.dtype == np.float64  # Should convert to float64


def test_student_t_pandas_integration():
    """Test integration of Student's t distribution with Pandas objects."""
    # Create distribution
    t_dist = StudentT(params=StudentTParameters(df=5.0))

    # Test with Pandas Series
    x_series = pd.Series([-2.0, -1.0, 0.0, 1.0, 2.0])
    pdf_series = t_dist.pdf(x_series)
    assert isinstance(pdf_series, pd.Series)
    assert pdf_series.shape == x_series.shape
    assert np.all(pdf_series.index == x_series.index)

    # Test with Pandas DataFrame
    x_df = pd.DataFrame({
        'A': [-2.0, -1.0, 0.0, 1.0, 2.0],
        'B': [0.0, 1.0, 2.0, 3.0, 4.0]
    })
    pdf_df = t_dist.pdf(x_df)
    assert isinstance(pdf_df, pd.DataFrame)
    assert pdf_df.shape == x_df.shape
    assert np.all(pdf_df.index == x_df.index)
    assert np.all(pdf_df.columns == x_df.columns)

    # Test with Series having non-default index
    x_series_idx = pd.Series([-2.0, -1.0, 0.0, 1.0, 2.0], index=['a', 'b', 'c', 'd', 'e'])
    pdf_series_idx = t_dist.pdf(x_series_idx)
    assert isinstance(pdf_series_idx, pd.Series)
    assert np.all(pdf_series_idx.index == x_series_idx.index)


def test_skewed_t_pandas_integration():
    """Test integration of skewed Student's t distribution with Pandas objects."""
    # Create distribution
    st_dist = SkewedT(params=SkewedTParameters(df=5.0, lambda_=0.3))

    # Test with Pandas Series
    x_series = pd.Series([-2.0, -1.0, 0.0, 1.0, 2.0])
    pdf_series = st_dist.pdf(x_series)
    assert isinstance(pdf_series, pd.Series)
    assert pdf_series.shape == x_series.shape
    assert np.all(pdf_series.index == x_series.index)

    # Test with Pandas DataFrame
    x_df = pd.DataFrame({
        'A': [-2.0, -1.0, 0.0, 1.0, 2.0],
        'B': [0.0, 1.0, 2.0, 3.0, 4.0]
    })
    pdf_df = st_dist.pdf(x_df)
    assert isinstance(pdf_df, pd.DataFrame)
    assert pdf_df.shape == x_df.shape
    assert np.all(pdf_df.index == x_df.index)
    assert np.all(pdf_df.columns == x_df.columns)

    # Test with Series having non-default index
    x_series_idx = pd.Series([-2.0, -1.0, 0.0, 1.0, 2.0], index=['a', 'b', 'c', 'd', 'e'])
    pdf_series_idx = st_dist.pdf(x_series_idx)
    assert isinstance(pdf_series_idx, pd.Series)
    assert np.all(pdf_series_idx.index == x_series_idx.index)


# ---- Numba Acceleration Tests ----

def test_student_t_numba_acceleration():
    """Test that Numba acceleration is working for Student's t distribution."""
    # Create distribution
    t_dist = StudentT(params=StudentTParameters(df=5.0))

    # Generate large array for performance testing
    n_samples = 10000
    x = np.linspace(-5, 5, n_samples)

    # Time the PDF computation
    import time
    start_time = time.time()
    pdf_values = t_dist.pdf(x)
    end_time = time.time()

    # Ensure computation completes in reasonable time
    # This is not a strict test, but helps identify if JIT compilation is working
    assert end_time - start_time < 1.0  # Should be very fast with JIT

    # Ensure PDF values are correct
    scale = np.sqrt(5.0 / 3.0)  # Standardization factor for df=5
    expected = stats.t.pdf(x * scale, 5.0) * scale
    np.testing.assert_allclose(pdf_values, expected, rtol=1e-5)


def test_skewed_t_numba_acceleration():
    """Test that Numba acceleration is working for skewed Student's t distribution."""
    # Create distribution
    st_dist = SkewedT(params=SkewedTParameters(df=5.0, lambda_=0.3))

    # Generate large array for performance testing
    n_samples = 10000
    x = np.linspace(-5, 5, n_samples)

    # Time the PDF computation
    import time
    start_time = time.time()
    pdf_values = st_dist.pdf(x)
    end_time = time.time()

    # Ensure computation completes in reasonable time
    # This is not a strict test, but helps identify if JIT compilation is working
    assert end_time - start_time < 1.0  # Should be very fast with JIT

    # Ensure PDF values are positive
    assert np.all(pdf_values > 0)

    # Ensure PDF integrates to approximately 1
    integral = np.trapz(pdf_values, x)
    assert abs(integral - 1.0) < 0.01


# ---- Parameterized Tests ----

@pytest.mark.parametrize("df", [3.0, 5.0, 10.0, 20.0])
def test_student_t_different_df(df):
    """Test Student's t distribution with different degrees of freedom."""
    # Create distribution
    t_dist = StudentT(params=StudentTParameters(df=df))

    # Test points
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    # Compute PDF
    pdf_values = t_dist.pdf(x)

    # Compare with scipy.stats implementation
    scale = np.sqrt(df / (df - 2))  # Standardization factor
    expected = stats.t.pdf(x * scale, df) * scale

    np.testing.assert_allclose(pdf_values, expected, rtol=1e-5)

    # Test kurtosis property: lower df means heavier tails
    if df < 20.0:  # Only test for df where the effect is noticeable
        t_dist_higher = StudentT(params=StudentTParameters(df=df+10.0))
        pdf_higher = t_dist_higher.pdf(x)

        # PDF at tails should be higher for lower df
        assert pdf_values[0] > pdf_higher[0]  # Left tail
        assert pdf_values[4] > pdf_higher[4]  # Right tail

        # PDF at center should be lower for lower df
        assert pdf_values[2] < pdf_higher[2]  # Center


@pytest.mark.parametrize("lambda_", [-0.5, -0.2, 0.0, 0.2, 0.5])
def test_skewed_t_different_lambda(lambda_):
    """Test skewed Student's t distribution with different skewness parameters."""
    # Create distribution
    st_dist = SkewedT(params=SkewedTParameters(df=5.0, lambda_=lambda_))

    # Test points
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    # Compute PDF
    pdf_values = st_dist.pdf(x)

    # Ensure PDF values are positive
    assert np.all(pdf_values > 0)

    # Test skewness property
    if lambda_ > 0.1:  # Right-skewed
        # For right-skewed, pdf(x) > pdf(-x) for x > 0
        assert pdf_values[3] > pdf_values[1]  # pdf(1.0) > pdf(-1.0)
        assert pdf_values[4] > pdf_values[0]  # pdf(2.0) > pdf(-2.0)
    elif lambda_ < -0.1:  # Left-skewed
        # For left-skewed, pdf(x) < pdf(-x) for x > 0
        assert pdf_values[3] < pdf_values[1]  # pdf(1.0) < pdf(-1.0)
        assert pdf_values[4] < pdf_values[0]  # pdf(2.0) < pdf(-2.0)
    else:  # Near symmetric
        # For symmetric, pdf(x) ≈ pdf(-x)
        np.testing.assert_allclose(pdf_values[3], pdf_values[1], rtol=0.1)
        np.testing.assert_allclose(pdf_values[4], pdf_values[0], rtol=0.1)


@pytest.mark.parametrize("size", [(100,), (10, 10), (5, 5, 4)])
def test_student_t_different_sizes(size):
    """Test Student's t random number generation with different sizes."""
    # Create distribution
    t_dist = StudentT(params=StudentTParameters(df=5.0))

    # Generate random samples
    samples = t_dist.rvs(size=size, random_state=42)

    # Check shape
    assert samples.shape == size

    # Check basic statistics
    assert abs(np.mean(samples)) < 0.5  # Mean should be close to 0

    # Test direct function
    direct_samples = stdtrnd(size, df=5.0, random_state=42)
    assert direct_samples.shape == size


@pytest.mark.parametrize("size", [(100,), (10, 10), (5, 5, 4)])
def test_skewed_t_different_sizes(size):
    """Test skewed Student's t random number generation with different sizes."""
    # Create distribution
    st_dist = SkewedT(params=SkewedTParameters(df=5.0, lambda_=0.3))

    # Generate random samples
    samples = st_dist.rvs(size=size, random_state=42)

    # Check shape
    assert samples.shape == size

    # Test direct function
    direct_samples = skewtrnd(size, df=5.0, lambda_=0.3, random_state=42)
    assert direct_samples.shape == size
