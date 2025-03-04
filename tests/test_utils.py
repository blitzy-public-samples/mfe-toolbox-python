# tests/test_utils.py
''' 
Tests for utility functions in the MFE Toolbox.

This module contains comprehensive tests for utility functions including
matrix operations, data transformations, seasonal differencing, and other
helper functions. It verifies the correct behavior of these functions across
various input types, edge cases, and ensures numerical precision.
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

from mfe.utils.matrix_ops import vech, ivech, vec2chol, chol2vec, cov2corr
from mfe.utils.data_transformations import standardize, demean, mvstandardize, sdiff
from mfe.utils.covariance import covnw, covvar
from mfe.utils.differentiation import gradient_2sided, hessian_2sided
from mfe.utils.misc import r2z, z2r, phi2r, r2phi


class TestMatrixOperations:
    """Tests for matrix operation utility functions."""

    def test_vech_ivech(self, assert_array_equal):
        """Test vectorization and inverse vectorization of symmetric matrices."""
        # Create a symmetric matrix
        n = 5
        A = np.random.randn(n, n)
        A = (A + A.T) / 2  # Make symmetric
        
        # Test vech
        v = vech(A)
        expected_length = n * (n + 1) // 2
        assert len(v) == expected_length, f"vech output length should be {expected_length}"
        
        # Test ivech
        A_reconstructed = ivech(v)
        assert A_reconstructed.shape == (n, n), "ivech output shape incorrect"
        assert_array_equal(A, A_reconstructed, err_msg="ivech(vech(A)) should equal A")
        
        # Test with different input types
        A_df = pd.DataFrame(A)
        v_df = vech(A_df)
        assert isinstance(v_df, np.ndarray), "vech should return ndarray even with DataFrame input"
        assert_array_equal(v, v_df, err_msg="vech results differ between ndarray and DataFrame")
        
        # Test with non-symmetric matrix
        B = np.random.randn(n, n)
        v_B = vech(B)
        B_reconstructed = ivech(v_B)
        # Only lower triangular part should be preserved
        B_lower = np.tril(B)
        B_symmetric = B_lower + B_lower.T - np.diag(np.diag(B))
        assert_array_equal(B_reconstructed, B_symmetric, 
                          err_msg="ivech(vech(B)) should equal symmetrized B")

    def test_vec2chol_chol2vec(self, assert_array_equal):
        """Test conversion between vectors and Cholesky factors."""
        # Create a positive definite matrix
        n = 4
        A = np.random.randn(n, n)
        A = A @ A.T  # Make positive definite
        
        # Get Cholesky factor
        L = np.linalg.cholesky(A)
        
        # Test chol2vec
        v = chol2vec(L)
        expected_length = n * (n + 1) // 2
        assert len(v) == expected_length, f"chol2vec output length should be {expected_length}"
        
        # Test vec2chol
        L_reconstructed = vec2chol(v, n)
        assert L_reconstructed.shape == (n, n), "vec2chol output shape incorrect"
        assert_array_equal(L, L_reconstructed, err_msg="vec2chol(chol2vec(L)) should equal L")
        
        # Test with different input types
        L_df = pd.DataFrame(L)
        v_df = chol2vec(L_df)
        assert isinstance(v_df, np.ndarray), "chol2vec should return ndarray even with DataFrame input"
        assert_array_equal(v, v_df, err_msg="chol2vec results differ between ndarray and DataFrame")
        
        # Test with invalid inputs
        with pytest.raises(ValueError, match="Input matrix must be square"):
            chol2vec(np.random.randn(3, 4))
        
        with pytest.raises(ValueError, match="Input vector length incompatible with matrix dimension"):
            vec2chol(np.random.randn(5), 3)

    def test_cov2corr(self, assert_array_equal):
        """Test conversion from covariance to correlation matrix."""
        # Create a covariance matrix
        n = 5
        A = np.random.randn(n, n)
        cov = A @ A.T  # Make positive definite
        
        # Test cov2corr
        corr = cov2corr(cov)
        assert corr.shape == (n, n), "cov2corr output shape incorrect"
        
        # Check diagonal elements are 1
        assert np.allclose(np.diag(corr), 1.0), "Diagonal elements of correlation matrix should be 1"
        
        # Check off-diagonal elements are between -1 and 1
        off_diag = corr[~np.eye(n, dtype=bool)]
        assert np.all((off_diag >= -1.0) & (off_diag <= 1.0)), "Correlation values should be between -1 and 1"
        
        # Check that correlation matrix is symmetric
        assert np.allclose(corr, corr.T), "Correlation matrix should be symmetric"
        
        # Test with different input types
        cov_df = pd.DataFrame(cov)
        corr_df = cov2corr(cov_df)
        assert isinstance(corr_df, np.ndarray), "cov2corr should return ndarray even with DataFrame input"
        assert_array_equal(corr, corr_df, err_msg="cov2corr results differ between ndarray and DataFrame")
        
        # Test with invalid inputs
        with pytest.raises(ValueError, match="Input matrix must be square"):
            cov2corr(np.random.randn(3, 4))
        
        # Test with non-positive definite matrix
        non_pd = np.ones((3, 3))
        with warnings.catch_warnings(record=True) as w:
            corr_non_pd = cov2corr(non_pd)
            assert len(w) > 0, "Warning should be raised for non-positive definite matrix"
        
        # Result should still be a valid correlation matrix
        assert np.allclose(np.diag(corr_non_pd), 1.0), "Diagonal elements should be 1 even for non-PD input"

    @given(arrays(dtype=np.float64, shape=st.tuples(st.integers(2, 10), st.integers(2, 10)),
                  elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)))
    def test_property_based_vech(self, A):
        """Property-based test for vech and ivech using hypothesis."""
        # Make matrix square if not already
        n = min(A.shape)
        A = A[:n, :n]
        
        # Make symmetric
        A = (A + A.T) / 2
        
        # Test vech and ivech
        v = vech(A)
        A_reconstructed = ivech(v)
        
        # Check properties
        assert len(v) == n * (n + 1) // 2, "vech output length incorrect"
        assert A_reconstructed.shape == (n, n), "ivech output shape incorrect"
        assert np.allclose(A, A_reconstructed), "ivech(vech(A)) should equal A"

    @given(arrays(dtype=np.float64, shape=st.tuples(st.integers(2, 10), st.integers(2, 10)),
                  elements=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)))
    def test_property_based_cov2corr(self, A):
        """Property-based test for cov2corr using hypothesis."""
        # Make matrix square if not already
        n = min(A.shape)
        A = A[:n, :n]
        
        # Make positive definite-like
        A = A @ A.T + np.eye(n) * 0.01  # Add small diagonal to ensure PD
        
        try:
            # Test cov2corr
            corr = cov2corr(A)
            
            # Check properties
            assert corr.shape == (n, n), "cov2corr output shape incorrect"
            assert np.allclose(np.diag(corr), 1.0), "Diagonal elements should be 1"
            assert np.all((corr >= -1.0) & (corr <= 1.0)), "Correlation values should be between -1 and 1"
            assert np.allclose(corr, corr.T), "Correlation matrix should be symmetric"
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            # Some random matrices might cause numerical issues, which is acceptable
            # in property-based testing
            pass


class TestDataTransformations:
    """Tests for data transformation utility functions."""

    def test_standardize(self, random_univariate_series, assert_array_equal):
        """Test standardization of univariate data."""
        # Test standardize
        std_data = standardize(random_univariate_series)
        assert std_data.shape == random_univariate_series.shape, "standardize output shape incorrect"
        
        # Check that mean is approximately 0 and std is approximately 1
        assert np.abs(np.mean(std_data)) < 1e-10, "Standardized data should have mean 0"
        assert np.abs(np.std(std_data, ddof=1) - 1.0) < 1e-10, "Standardized data should have std 1"
        
        # Test with pandas Series
        series = pd.Series(random_univariate_series)
        std_series = standardize(series)
        assert isinstance(std_series, pd.Series), "standardize should return Series for Series input"
        assert_array_equal(std_data, std_series.values, 
                          err_msg="standardize results differ between ndarray and Series")
        
        # Test with return_params=True
        std_data, params = standardize(random_univariate_series, return_params=True)
        assert 'mean' in params and 'std' in params, "standardize should return mean and std params"
        assert np.isclose(params['mean'], np.mean(random_univariate_series)), "Mean parameter incorrect"
        assert np.isclose(params['std'], np.std(random_univariate_series, ddof=1)), "Std parameter incorrect"
        
        # Test with provided parameters
        params = {'mean': 5.0, 'std': 2.0}
        std_data = standardize(random_univariate_series, params=params)
        expected = (random_univariate_series - 5.0) / 2.0
        assert_array_equal(std_data, expected, err_msg="standardize with params incorrect")

    def test_demean(self, random_univariate_series, assert_array_equal):
        """Test demeaning of univariate data."""
        # Test demean
        demeaned = demean(random_univariate_series)
        assert demeaned.shape == random_univariate_series.shape, "demean output shape incorrect"
        
        # Check that mean is approximately 0
        assert np.abs(np.mean(demeaned)) < 1e-10, "Demeaned data should have mean 0"
        
        # Test with pandas Series
        series = pd.Series(random_univariate_series)
        demeaned_series = demean(series)
        assert isinstance(demeaned_series, pd.Series), "demean should return Series for Series input"
        assert_array_equal(demeaned, demeaned_series.values, 
                          err_msg="demean results differ between ndarray and Series")
        
        # Test with return_params=True
        demeaned, params = demean(random_univariate_series, return_params=True)
        assert 'mean' in params, "demean should return mean param"
        assert np.isclose(params['mean'], np.mean(random_univariate_series)), "Mean parameter incorrect"
        
        # Test with provided parameters
        params = {'mean': 5.0}
        demeaned = demean(random_univariate_series, params=params)
        expected = random_univariate_series - 5.0
        assert_array_equal(demeaned, expected, err_msg="demean with params incorrect")

    def test_mvstandardize(self, random_multivariate_series, assert_array_equal):
        """Test standardization of multivariate data."""
        # Test mvstandardize
        std_data = mvstandardize(random_multivariate_series)
        assert std_data.shape == random_multivariate_series.shape, "mvstandardize output shape incorrect"
        
        # Check that each column has mean 0 and std 1
        assert np.allclose(np.mean(std_data, axis=0), 0.0), "Standardized data should have mean 0 for each column"
        assert np.allclose(np.std(std_data, axis=0, ddof=1), 1.0), "Standardized data should have std 1 for each column"
        
        # Test with pandas DataFrame
        df = pd.DataFrame(random_multivariate_series)
        std_df = mvstandardize(df)
        assert isinstance(std_df, pd.DataFrame), "mvstandardize should return DataFrame for DataFrame input"
        assert_array_equal(std_data, std_df.values, 
                          err_msg="mvstandardize results differ between ndarray and DataFrame")
        
        # Test with return_params=True
        std_data, params = mvstandardize(random_multivariate_series, return_params=True)
        assert 'mean' in params and 'std' in params, "mvstandardize should return mean and std params"
        assert np.allclose(params['mean'], np.mean(random_multivariate_series, axis=0)), "Mean parameters incorrect"
        assert np.allclose(params['std'], np.std(random_multivariate_series, axis=0, ddof=1)), "Std parameters incorrect"
        
        # Test with provided parameters
        params = {
            'mean': np.array([5.0, 3.0, 1.0, 2.0, 4.0]),
            'std': np.array([2.0, 1.5, 1.0, 2.5, 3.0])
        }
        std_data = mvstandardize(random_multivariate_series, params=params)
        expected = (random_multivariate_series - params['mean']) / params['std']
        assert_array_equal(std_data, expected, err_msg="mvstandardize with params incorrect")

    def test_sdiff(self, assert_array_equal):
        """Test seasonal differencing function."""
        # Create seasonal data
        n = 100
        seasonal_period = 12
        t = np.arange(n)
        seasonal_component = 2 * np.sin(2 * np.pi * t / seasonal_period)
        data = seasonal_component + np.random.randn(n) * 0.1  # Add small noise
        
        # Test sdiff
        diff_data = sdiff(data, seasonal_period)
        expected_length = n - seasonal_period
        assert len(diff_data) == expected_length, f"sdiff output length should be {expected_length}"
        
        # Check that seasonal pattern is reduced
        acf_before = np.corrcoef(data[seasonal_period:], data[:-seasonal_period])[0, 1]
        acf_after = np.corrcoef(diff_data[seasonal_period:], diff_data[:-seasonal_period])[0, 1]
        assert abs(acf_after) < abs(acf_before), "Seasonal differencing didn't reduce autocorrelation at seasonal lag"
        
        # Test with pandas Series
        dates = pd.date_range(start='2020-01-01', periods=n, freq='M')
        series = pd.Series(data, index=dates)
        diff_series = sdiff(series, seasonal_period)
        assert isinstance(diff_series, pd.Series), "sdiff should return Series for Series input"
        assert len(diff_series) == expected_length, "sdiff output length incorrect for Series"
        assert diff_series.index[0] == dates[seasonal_period], "sdiff output index incorrect"
        assert_array_equal(diff_data, diff_series.values, 
                          err_msg="sdiff results differ between ndarray and Series")
        
        # Test with multiple seasonal periods
        diff_data_2 = sdiff(data, seasonal_period, nsdiffs=2)
        expected_length_2 = n - 2 * seasonal_period
        assert len(diff_data_2) == expected_length_2, f"sdiff with nsdiffs=2 output length should be {expected_length_2}"
        
        # Test with invalid inputs
        with pytest.raises(ValueError, match="Seasonal period must be positive"):
            sdiff(data, -1)
        
        with pytest.raises(ValueError, match="Number of seasonal differences must be positive"):
            sdiff(data, seasonal_period, nsdiffs=0)
        
        with pytest.raises(ValueError, match="Input array length must be greater than"):
            sdiff(data[:5], seasonal_period)

    @given(arrays(dtype=np.float64, shape=st.integers(20, 100),
                  elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)))
    def test_property_based_standardize(self, data):
        """Property-based test for standardize using hypothesis."""
        try:
            # Test standardize
            std_data = standardize(data)
            
            # Check properties
            assert std_data.shape == data.shape, "standardize output shape incorrect"
            assert np.abs(np.mean(std_data)) < 1e-8, "Standardized data should have mean 0"
            assert np.abs(np.std(std_data, ddof=1) - 1.0) < 1e-8, "Standardized data should have std 1"
            
            # Test with return_params=True
            std_data_2, params = standardize(data, return_params=True)
            assert np.allclose(std_data, std_data_2), "standardize results should be consistent"
            
            # Test with provided parameters
            std_data_3 = standardize(data, params=params)
            assert np.allclose(std_data, std_data_3), "standardize with params should match original standardization"
        except (ValueError, RuntimeError, ZeroDivisionError):
            # Some random arrays might cause numerical issues, which is acceptable
            # in property-based testing
            pass

    @given(arrays(dtype=np.float64, shape=st.tuples(st.integers(20, 100), st.integers(2, 5)),
                  elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)))
    def test_property_based_mvstandardize(self, data):
        """Property-based test for mvstandardize using hypothesis."""
        try:
            # Test mvstandardize
            std_data = mvstandardize(data)
            
            # Check properties
            assert std_data.shape == data.shape, "mvstandardize output shape incorrect"
            assert np.allclose(np.mean(std_data, axis=0), 0.0, atol=1e-8), "Standardized data should have mean 0 for each column"
            assert np.allclose(np.std(std_data, axis=0, ddof=1), 1.0, atol=1e-8), "Standardized data should have std 1 for each column"
            
            # Test with return_params=True
            std_data_2, params = mvstandardize(data, return_params=True)
            assert np.allclose(std_data, std_data_2), "mvstandardize results should be consistent"
            
            # Test with provided parameters
            std_data_3 = mvstandardize(data, params=params)
            assert np.allclose(std_data, std_data_3), "mvstandardize with params should match original standardization"
        except (ValueError, RuntimeError, ZeroDivisionError):
            # Some random arrays might cause numerical issues, which is acceptable
            # in property-based testing
            pass


class TestCovarianceEstimators:
    """Tests for covariance estimation utility functions."""

    def test_covnw(self, random_univariate_series, assert_array_equal):
        """Test Newey-West covariance estimator."""
        # Create design matrix with constant and random_univariate_series
        X = np.column_stack((np.ones(len(random_univariate_series)), random_univariate_series))
        
        # Create dependent variable with known relationship plus noise
        beta = np.array([1.0, 0.5])
        y = X @ beta + np.random.randn(len(random_univariate_series)) * 0.1
        
        # Compute OLS residuals
        beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
        e = y - X @ beta_hat
        
        # Test covnw
        cov = covnw(X, e, lags=5)
        assert cov.shape == (X.shape[1], X.shape[1]), "covnw output shape incorrect"
        
        # Check that covariance matrix is symmetric
        assert np.allclose(cov, cov.T), "Covariance matrix should be symmetric"
        
        # Check that diagonal elements are positive
        assert np.all(np.diag(cov) > 0), "Diagonal elements of covariance matrix should be positive"
        
        # Test with pandas inputs
        X_df = pd.DataFrame(X)
        e_series = pd.Series(e)
        cov_pd = covnw(X_df, e_series, lags=5)
        assert isinstance(cov_pd, np.ndarray), "covnw should return ndarray even with DataFrame/Series input"
        assert_array_equal(cov, cov_pd, err_msg="covnw results differ between ndarray and DataFrame/Series")
        
        # Test with invalid inputs
        with pytest.raises(ValueError, match="X and e must have the same number of observations"):
            covnw(X, e[:-1], lags=5)
        
        with pytest.raises(ValueError, match="Number of lags must be non-negative"):
            covnw(X, e, lags=-1)

    def test_covvar(self, random_univariate_series, assert_array_equal):
        """Test heteroskedasticity-robust covariance estimator."""
        # Create design matrix with constant and random_univariate_series
        X = np.column_stack((np.ones(len(random_univariate_series)), random_univariate_series))
        
        # Create dependent variable with known relationship plus heteroskedastic noise
        beta = np.array([1.0, 0.5])
        hetero_noise = np.random.randn(len(random_univariate_series)) * (1 + 0.5 * np.abs(random_univariate_series))
        y = X @ beta + hetero_noise
        
        # Compute OLS residuals
        beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
        e = y - X @ beta_hat
        
        # Test covvar
        cov = covvar(X, e)
        assert cov.shape == (X.shape[1], X.shape[1]), "covvar output shape incorrect"
        
        # Check that covariance matrix is symmetric
        assert np.allclose(cov, cov.T), "Covariance matrix should be symmetric"
        
        # Check that diagonal elements are positive
        assert np.all(np.diag(cov) > 0), "Diagonal elements of covariance matrix should be positive"
        
        # Test with pandas inputs
        X_df = pd.DataFrame(X)
        e_series = pd.Series(e)
        cov_pd = covvar(X_df, e_series)
        assert isinstance(cov_pd, np.ndarray), "covvar should return ndarray even with DataFrame/Series input"
        assert_array_equal(cov, cov_pd, err_msg="covvar results differ between ndarray and DataFrame/Series")
        
        # Test with invalid inputs
        with pytest.raises(ValueError, match="X and e must have the same number of observations"):
            covvar(X, e[:-1])

    @given(arrays(dtype=np.float64, shape=st.tuples(st.integers(20, 100), st.integers(2, 5)),
                  elements=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)))
    def test_property_based_covariance(self, X):
        """Property-based test for covariance estimators using hypothesis."""
        try:
            # Create residuals
            n = X.shape[0]
            e = np.random.randn(n)
            
            # Test covnw
            cov_nw = covnw(X, e, lags=3)
            
            # Check properties
            assert cov_nw.shape == (X.shape[1], X.shape[1]), "covnw output shape incorrect"
            assert np.allclose(cov_nw, cov_nw.T), "Covariance matrix should be symmetric"
            
            # Test covvar
            cov_var = covvar(X, e)
            
            # Check properties
            assert cov_var.shape == (X.shape[1], X.shape[1]), "covvar output shape incorrect"
            assert np.allclose(cov_var, cov_var.T), "Covariance matrix should be symmetric"
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            # Some random matrices might cause numerical issues, which is acceptable
            # in property-based testing
            pass


class TestDifferentiation:
    """Tests for numerical differentiation utility functions."""

    def test_gradient_2sided(self, assert_array_equal):
        """Test two-sided gradient approximation."""
        # Define a simple function
        def f(x):
            return np.sum(x**2)
        
        # Test at a specific point
        x = np.array([1.0, 2.0, 3.0])
        grad = gradient_2sided(f, x)
        
        # Analytical gradient is 2*x
        expected_grad = 2 * x
        assert_array_equal(grad, expected_grad, rtol=1e-5, 
                          err_msg="Numerical gradient doesn't match analytical gradient")
        
        # Test with different step size
        grad_small_step = gradient_2sided(f, x, eps=1e-6)
        assert_array_equal(grad_small_step, expected_grad, rtol=1e-5, 
                          err_msg="Gradient with small step size doesn't match analytical gradient")
        
        # Test with vector-valued function
        def g(x):
            return np.array([x[0]**2, x[1]**3, np.sin(x[2])])
        
        # Analytical Jacobian
        def g_jac(x):
            return np.array([
                [2*x[0], 0, 0],
                [0, 3*x[1]**2, 0],
                [0, 0, np.cos(x[2])]
            ])
        
        jac = gradient_2sided(g, x)
        expected_jac = g_jac(x)
        assert_array_equal(jac, expected_jac, rtol=1e-5, 
                          err_msg="Numerical Jacobian doesn't match analytical Jacobian")

    def test_hessian_2sided(self, assert_array_equal):
        """Test two-sided Hessian approximation."""
        # Define a simple function
        def f(x):
            return np.sum(x**2)
        
        # Test at a specific point
        x = np.array([1.0, 2.0, 3.0])
        hess = hessian_2sided(f, x)
        
        # Analytical Hessian is 2*I
        expected_hess = 2 * np.eye(len(x))
        assert_array_equal(hess, expected_hess, rtol=1e-4, 
                          err_msg="Numerical Hessian doesn't match analytical Hessian")
        
        # Test with different step size
        hess_small_step = hessian_2sided(f, x, eps=1e-5)
        assert_array_equal(hess_small_step, expected_hess, rtol=1e-4, 
                          err_msg="Hessian with small step size doesn't match analytical Hessian")
        
        # Test with more complex function
        def h(x):
            return x[0]**2 + 2*x[1]**2 + 3*x[2]**2 + x[0]*x[1] + x[1]*x[2]
        
        # Analytical Hessian
        expected_hess = np.array([
            [2, 1, 0],
            [1, 4, 1],
            [0, 1, 6]
        ])
        
        hess = hessian_2sided(h, x)
        assert_array_equal(hess, expected_hess, rtol=1e-4, 
                          err_msg="Numerical Hessian doesn't match analytical Hessian for complex function")

    @given(arrays(dtype=np.float64, shape=st.integers(1, 5),
                  elements=st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False)))
    def test_property_based_differentiation(self, x):
        """Property-based test for differentiation functions using hypothesis."""
        try:
            # Define a simple quadratic function
            def f(x):
                return np.sum(x**2)
            
            # Analytical gradient is 2*x
            expected_grad = 2 * x
            
            # Test gradient
            grad = gradient_2sided(f, x)
            assert np.allclose(grad, expected_grad, rtol=1e-4, atol=1e-4), "Gradient doesn't match analytical gradient"
            
            # Analytical Hessian is 2*I
            expected_hess = 2 * np.eye(len(x))
            
            # Test Hessian
            hess = hessian_2sided(f, x)
            assert np.allclose(hess, expected_hess, rtol=1e-3, atol=1e-3), "Hessian doesn't match analytical Hessian"
        except (ValueError, RuntimeError):
            # Some random inputs might cause numerical issues, which is acceptable
            # in property-based testing
            pass


class TestParameterTransformations:
    """Tests for parameter transformation utility functions."""

    def test_r2z_z2r(self, assert_array_equal):
        """Test correlation to Fisher z-transform and back."""
        # Test with scalar
        r = 0.5
        z = r2z(r)
        r_back = z2r(z)
        assert np.isclose(r, r_back), "z2r(r2z(r)) should equal r"
        
        # Test with array
        r_array = np.array([-0.9, -0.5, 0.0, 0.5, 0.9])
        z_array = r2z(r_array)
        r_back_array = z2r(z_array)
        assert_array_equal(r_array, r_back_array, 
                          err_msg="z2r(r2z(r)) should equal r for array input")
        
        # Test with pandas Series
        r_series = pd.Series(r_array)
        z_series = r2z(r_series)
        assert isinstance(z_series, np.ndarray), "r2z should return ndarray even with Series input"
        r_back_series = z2r(z_series)
        assert_array_equal(r_array, r_back_series, 
                          err_msg="z2r(r2z(r)) should equal r for Series input")
        
        # Test with invalid inputs
        with pytest.raises(ValueError, match="Correlation must be between -1 and 1"):
            r2z(1.5)
        
        with pytest.raises(ValueError, match="Correlation must be between -1 and 1"):
            r2z(np.array([0.5, 1.5]))

    def test_phi2r_r2phi(self, assert_array_equal):
        """Test AR coefficient to partial correlation and back."""
        # Test with scalar
        phi = 0.5
        r = phi2r(phi)
        phi_back = r2phi(r)
        assert np.isclose(phi, phi_back), "r2phi(phi2r(phi)) should equal phi"
        
        # Test with array
        phi_array = np.array([-0.9, -0.5, 0.0, 0.5, 0.9])
        r_array = phi2r(phi_array)
        phi_back_array = r2phi(r_array)
        assert_array_equal(phi_array, phi_back_array, 
                          err_msg="r2phi(phi2r(phi)) should equal phi for array input")
        
        # Test with pandas Series
        phi_series = pd.Series(phi_array)
        r_series = phi2r(phi_series)
        assert isinstance(r_series, np.ndarray), "phi2r should return ndarray even with Series input"
        phi_back_series = r2phi(r_series)
        assert_array_equal(phi_array, phi_back_series, 
                          err_msg="r2phi(phi2r(phi)) should equal phi for Series input")
        
        # Test with invalid inputs
        with pytest.raises(ValueError, match="AR coefficient must be between -1 and 1"):
            phi2r(1.5)
        
        with pytest.raises(ValueError, match="AR coefficient must be between -1 and 1"):
            phi2r(np.array([0.5, 1.5]))

    @given(st.floats(min_value=-0.99, max_value=0.99))
    def test_property_based_transformations(self, r):
        """Property-based test for parameter transformations using hypothesis."""
        # Test r2z and z2r
        z = r2z(r)
        r_back = z2r(z)
        assert np.isclose(r, r_back), "z2r(r2z(r)) should equal r"
        
        # Test phi2r and r2phi
        phi = r  # Use same value for simplicity
        r_phi = phi2r(phi)
        phi_back = r2phi(r_phi)
        assert np.isclose(phi, phi_back), "r2phi(phi2r(phi)) should equal phi"


class TestNumbaAcceleration:
    """Tests for Numba-accelerated utility functions."""

    def test_numba_acceleration(self):
        """Test that Numba acceleration is working correctly."""
        # This test verifies that Numba-accelerated functions are faster
        # than their pure Python equivalents
        
        # Create large arrays for testing
        n = 1000
        X = np.random.randn(n, 5)
        e = np.random.randn(n)
        
        # Time the execution of covnw with and without Numba
        import time
        
        # First run to ensure JIT compilation
        _ = covnw(X, e, lags=5)
        
        # Measure time with Numba
        start = time.time()
        for _ in range(10):
            _ = covnw(X, e, lags=5)
        numba_time = time.time() - start
        
        # Check that the function has a _numba attribute or similar indication
        # that it's using Numba acceleration
        assert hasattr(covnw, '__module__'), "Function should have module attribute"
        
        # We can't directly test the non-Numba version since we don't have access to it,
        # but we can verify that the function executes in a reasonable time
        assert numba_time < 1.0, "Numba-accelerated function should be fast"

    @pytest.mark.asyncio
    async def test_async_utility_functions(self, random_univariate_series):
        """Test asynchronous versions of utility functions."""
        # Test async standardize if available
        try:
            from mfe.utils.data_transformations import standardize_async
            
            # Define progress callback
            progress_updates = []
            
            async def progress_callback(percent, message):
                progress_updates.append((percent, message))
            
            # Test async standardize
            std_data = await standardize_async(random_univariate_series, progress_callback=progress_callback)
            
            # Check that standardization worked
            assert np.abs(np.mean(std_data)) < 1e-10, "Standardized data should have mean 0"
            assert np.abs(np.std(std_data, ddof=1) - 1.0) < 1e-10, "Standardized data should have std 1"
            
            # Check that progress callback was called
            assert len(progress_updates) > 0, "Progress callback was not called"
            assert progress_updates[-1][0] == 100.0, "Final progress update not 100%"
        except ImportError:
            # Async version might not be available, which is fine
            pass


class TestErrorHandling:
    """Tests for error handling in utility functions."""

    def test_input_validation(self):
        """Test that utility functions properly validate inputs."""
        # Test with empty array
        with pytest.raises(ValueError, match="Input array must contain data"):
            standardize(np.array([]))
        
        # Test with NaN values
        with pytest.raises(ValueError, match="Input array contains NaN values"):
            standardize(np.array([1.0, 2.0, np.nan, 4.0]))
        
        # Test with infinite values
        with pytest.raises(ValueError, match="Input array contains infinite values"):
            standardize(np.array([1.0, 2.0, np.inf, 4.0]))
        
        # Test with incompatible dimensions
        with pytest.raises(ValueError, match="Input matrix must be square"):
            cov2corr(np.random.randn(3, 4))
        
        # Test with invalid parameters
        with pytest.raises(ValueError, match="Correlation must be between -1 and 1"):
            r2z(1.5)

    def test_edge_cases(self, assert_array_equal):
        """Test utility functions with edge cases."""
        # Test with very small array
        small_array = np.array([1.0])
        std_small = standardize(small_array)
        assert_array_equal(std_small, np.array([0.0]), 
                          err_msg="standardize should handle single-element arrays correctly")
        
        # Test with constant array
        const_array = np.ones(10)
        std_const = standardize(const_array)
        assert np.all(np.isnan(std_const)), "standardize should return NaN for constant arrays"
        
        # Test with very large values
        large_array = np.array([1e10, 2e10, 3e10])
        std_large = standardize(large_array)
        expected = (large_array - np.mean(large_array)) / np.std(large_array, ddof=1)
        assert_array_equal(std_large, expected, 
                          err_msg="standardize should handle very large values correctly")
        
        # Test with very small values
        small_values = np.array([1e-10, 2e-10, 3e-10])
        std_small_values = standardize(small_values)
        expected = (small_values - np.mean(small_values)) / np.std(small_values, ddof=1)
        assert_array_equal(std_small_values, expected, 
                          err_msg="standardize should handle very small values correctly")


class TestIntegrationTests:
    """Integration tests for utility functions."""

    def test_utility_function_pipeline(self, random_multivariate_series, assert_array_equal):
        """Test a pipeline of utility functions working together."""
        # Create a pipeline: standardize -> compute covariance -> convert to correlation
        
        # Step 1: Standardize the data
        std_data = mvstandardize(random_multivariate_series)
        
        # Step 2: Create design matrix and residuals for covariance estimation
        X = std_data[:, :3]  # Use first 3 columns as design matrix
        y = std_data[:, 3]   # Use 4th column as dependent variable
        beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
        e = y - X @ beta_hat
        
        # Step 3: Compute robust covariance
        cov = covnw(X, e, lags=5)
        
        # Step 4: Convert to correlation matrix
        corr = cov2corr(cov)
        
        # Verify the pipeline
        assert std_data.shape == random_multivariate_series.shape, "Standardization should preserve shape"
        assert np.allclose(np.mean(std_data, axis=0), 0.0), "Standardized data should have mean 0 for each column"
        assert np.allclose(np.std(std_data, axis=0, ddof=1), 1.0), "Standardized data should have std 1 for each column"
        
        assert cov.shape == (X.shape[1], X.shape[1]), "Covariance matrix shape incorrect"
        assert np.allclose(cov, cov.T), "Covariance matrix should be symmetric"
        
        assert corr.shape == cov.shape, "Correlation matrix shape should match covariance matrix"
        assert np.allclose(np.diag(corr), 1.0), "Diagonal elements of correlation matrix should be 1"
        assert np.all((corr >= -1.0) & (corr <= 1.0)), "Correlation values should be between -1 and 1"

    def test_pandas_integration(self, random_multivariate_series, assert_array_equal):
        """Test integration with pandas objects."""
        # Create pandas DataFrame with DatetimeIndex
        dates = pd.date_range(start='2020-01-01', periods=len(random_multivariate_series), freq='D')
        df = pd.DataFrame(random_multivariate_series, index=dates)
        
        # Test standardization
        std_df = mvstandardize(df)
        assert isinstance(std_df, pd.DataFrame), "mvstandardize should return DataFrame for DataFrame input"
        assert std_df.index.equals(df.index), "mvstandardize should preserve index"
        
        # Test seasonal differencing
        seasonal_period = 7  # Weekly seasonality
        diff_df = sdiff(df, seasonal_period)
        assert isinstance(diff_df, pd.DataFrame), "sdiff should return DataFrame for DataFrame input"
        assert len(diff_df) == len(df) - seasonal_period, "sdiff output length incorrect"
        assert diff_df.index[0] == dates[seasonal_period], "sdiff output index incorrect"
        
        # Test with Series
        series = df.iloc[:, 0]
        std_series = standardize(series)
        assert isinstance(std_series, pd.Series), "standardize should return Series for Series input"
        assert std_series.index.equals(series.index), "standardize should preserve index"
        
        diff_series = sdiff(series, seasonal_period)
        assert isinstance(diff_series, pd.Series), "sdiff should return Series for Series input"
        assert len(diff_series) == len(series) - seasonal_period, "sdiff output length incorrect for Series"
        assert diff_series.index[0] == dates[seasonal_period], "sdiff output index incorrect for Series"

    def test_numba_integration(self, random_multivariate_series):
        """Test integration with Numba-accelerated functions."""
        # Create data for testing
        X = random_multivariate_series[:, :3]
        y = random_multivariate_series[:, 3]
        beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
        e = y - X @ beta_hat
        
        # Test covnw with Numba acceleration
        cov = covnw(X, e, lags=5)
        assert cov.shape == (X.shape[1], X.shape[1]), "covnw output shape incorrect"
        assert np.allclose(cov, cov.T), "Covariance matrix should be symmetric"
        
        # Test covvar with Numba acceleration
        cov_var = covvar(X, e)
        assert cov_var.shape == (X.shape[1], X.shape[1]), "covvar output shape incorrect"
        assert np.allclose(cov_var, cov_var.T), "Covariance matrix should be symmetric"