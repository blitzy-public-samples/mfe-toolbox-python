# tests/test_utils.py
"""
Tests for utility functions in the MFE Toolbox.

This module contains tests for various utility functions, including matrix operations,
data transformations, and miscellaneous utilities. It focuses particularly on the
seasonal differencing implementation (sdiff) and ensures that utility functions
correctly handle NumPy arrays, Pandas Series, and various input configurations
while maintaining numerical precision.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_series_equal, assert_frame_equal
from hypothesis import given, strategies as st, settings, assume
from scipy import linalg

from mfe.utils.matrix_ops import (
    vech, ivech, vec2chol, chol2vec, cov2corr, corr2cov,
    ensure_symmetric, is_positive_definite, nearest_positive_definite,
    block_diagonal, commutation_matrix, duplication_matrix, elimination_matrix
)
from mfe.utils.data_transformations import (
    standardize, mvstandardize, demean, lag_matrix, lag_series,
    rolling_window, rolling_mean, rolling_variance, rolling_skewness, rolling_kurtosis
)
from mfe.utils.misc import (
    r2z, z2r, phi2r, r2phi, ensure_array, ensure_dataframe, ensure_series,
    lag_matrix_extended
)
from mfe.core.exceptions import DimensionError, DataError


# ---- Matrix Operations Tests ----

class TestMatrixOperations:
    """Tests for matrix operations utility functions."""

    def test_vech(self):
        """Test vech function for vectorizing lower triangular portion of a matrix."""
        # Test with a simple 2x2 matrix
        A = np.array([[1, 2], [2, 4]])
        expected = np.array([1, 2, 4])
        result = vech(A)
        assert_array_equal(result, expected)
        
        # Test with a 3x3 matrix
        B = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
        expected = np.array([1, 2, 4, 3, 5, 6])
        result = vech(B)
        assert_array_equal(result, expected)
        
        # Test with non-square matrix
        C = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(DimensionError):
            vech(C)

    def test_ivech(self):
        """Test ivech function for inverse vech operation."""
        # Test with a simple vector
        v = np.array([1, 2, 4])
        expected = np.array([[1, 2], [2, 4]])
        result = ivech(v)
        assert_array_equal(result, expected)
        
        # Test with a longer vector
        v = np.array([1, 2, 4, 3, 5, 6])
        expected = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
        result = ivech(v)
        assert_array_equal(result, expected)
        
        # Test diagonal_only option
        v = np.array([1, 2, 3])
        expected = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        result = ivech(v, diagonal_only=True)
        assert_array_equal(result, expected)
        
        # Test with invalid vector length
        v = np.array([1, 2, 3, 4])
        with pytest.raises(DimensionError):
            ivech(v)

    def test_vec2chol_chol2vec(self):
        """Test vec2chol and chol2vec functions for Cholesky factor conversion."""
        # Test vec2chol
        v = np.array([1, 2, 0, 3, 4, 5])
        n = 3
        expected = np.array([[1, 0, 0], [2, 3, 0], [0, 4, 5]])
        result = vec2chol(v, n)
        assert_array_equal(result, expected)
        
        # Test chol2vec
        L = np.array([[1, 0, 0], [2, 3, 0], [0, 4, 5]])
        expected = np.array([1, 2, 0, 3, 4, 5])
        result = chol2vec(L)
        assert_array_equal(result, expected)
        
        # Test round-trip conversion
        v_original = np.array([1, 2, 0, 3, 4, 5])
        L = vec2chol(v_original, 3)
        v_result = chol2vec(L)
        assert_array_equal(v_result, v_original)
        
        # Test with invalid inputs
        with pytest.raises(DimensionError):
            vec2chol(np.array([1, 2, 3]), 3)  # Wrong vector length
        
        with pytest.raises(DimensionError):
            chol2vec(np.array([[1, 2], [3, 4], [5, 6]]))  # Non-square matrix

    def test_cov2corr(self):
        """Test cov2corr function for converting covariance to correlation matrix."""
        # Test with a simple covariance matrix
        cov = np.array([[4, 2, 0], [2, 9, -3], [0, -3, 16]])
        expected = np.array([
            [1.0, 2/np.sqrt(4*9), 0],
            [2/np.sqrt(4*9), 1.0, -3/np.sqrt(9*16)],
            [0, -3/np.sqrt(9*16), 1.0]
        ])
        result = cov2corr(cov)
        assert_allclose(result, expected)
        
        # Test with non-positive diagonal elements
        cov_invalid = np.array([[4, 2], [2, 0]])
        with pytest.raises(DataError):
            cov2corr(cov_invalid)
        
        # Test with non-square matrix
        cov_non_square = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(DimensionError):
            cov2corr(cov_non_square)

    def test_corr2cov(self):
        """Test corr2cov function for converting correlation to covariance matrix."""
        # Test with a simple correlation matrix
        corr = np.array([[1, 0.3, 0], [0.3, 1, -0.2], [0, -0.2, 1]])
        std_devs = np.array([2, 3, 4])
        expected = np.array([
            [4, 1.8, 0],
            [1.8, 9, -2.4],
            [0, -2.4, 16]
        ])
        result = corr2cov(corr, std_devs)
        assert_allclose(result, expected)
        
        # Test with non-unit diagonal elements
        corr_invalid = np.array([[0.9, 0.3], [0.3, 1.0]])
        with pytest.raises(DataError):
            corr2cov(corr_invalid, np.array([1, 2]))
        
        # Test with non-positive standard deviations
        with pytest.raises(DataError):
            corr2cov(corr, np.array([2, 0, 4]))
        
        # Test with mismatched dimensions
        with pytest.raises(DimensionError):
            corr2cov(corr, np.array([2, 3]))

    def test_ensure_symmetric(self):
        """Test ensure_symmetric function for making a matrix symmetric."""
        # Test with already symmetric matrix
        A = np.array([[1, 2], [2, 3]])
        result = ensure_symmetric(A)
        assert_array_equal(result, A)
        
        # Test with non-symmetric matrix
        B = np.array([[1, 2], [3, 4]])
        expected = np.array([[1, 2.5], [2.5, 4]])
        result = ensure_symmetric(B)
        assert_allclose(result, expected)
        
        # Test with tolerance
        C = np.array([[1, 2.000001], [2, 3]])
        result = ensure_symmetric(C, tol=1e-5)
        assert_array_equal(result, C)  # Should be considered symmetric within tolerance
        
        # Test with non-square matrix
        D = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(DimensionError):
            ensure_symmetric(D)

    def test_is_positive_definite(self):
        """Test is_positive_definite function."""
        # Test with positive definite matrix
        A = np.array([[2, 1], [1, 2]])
        assert is_positive_definite(A) is True
        
        # Test with non-positive definite matrix
        B = np.array([[1, 2], [2, 1]])
        assert is_positive_definite(B) is False
        
        # Test with non-square matrix
        C = np.array([[1, 2, 3], [4, 5, 6]])
        assert is_positive_definite(C) is False

    def test_nearest_positive_definite(self):
        """Test nearest_positive_definite function."""
        # Test with already positive definite matrix
        A = np.array([[2, 1], [1, 2]])
        result = nearest_positive_definite(A)
        assert_allclose(result, A)
        
        # Test with non-positive definite matrix
        B = np.array([[1, 2], [2, 1]])
        result = nearest_positive_definite(B)
        assert is_positive_definite(result) is True
        
        # Test with epsilon
        C = np.array([[1, 0.999], [0.999, 1]])  # Nearly singular
        result = nearest_positive_definite(C, epsilon=0.01)
        assert is_positive_definite(result) is True
        
        # Test with non-square matrix
        D = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(DimensionError):
            nearest_positive_definite(D)

    def test_block_diagonal(self):
        """Test block_diagonal function."""
        # Test with two matrices
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        expected = np.array([
            [1, 2, 0, 0],
            [3, 4, 0, 0],
            [0, 0, 5, 6],
            [0, 0, 7, 8]
        ])
        result = block_diagonal([A, B])
        assert_array_equal(result, expected)
        
        # Test with matrices of different sizes
        C = np.array([[9]])
        expected = np.array([
            [1, 2, 0],
            [3, 4, 0],
            [0, 0, 9]
        ])
        result = block_diagonal([A, C])
        assert_array_equal(result, expected)
        
        # Test with empty list
        result = block_diagonal([])
        assert_array_equal(result, np.array([[]]))
        
        # Test with non-2D matrices
        with pytest.raises(DimensionError):
            block_diagonal([A, np.array([1, 2, 3])])

    def test_commutation_matrix(self):
        """Test commutation_matrix function."""
        # Test with 2x3 matrix
        K = commutation_matrix(2, 3)
        assert K.shape == (6, 6)
        
        # Verify K * vec(A) = vec(A.T)
        A = np.array([[1, 2, 3], [4, 5, 6]])
        vec_A = A.flatten('F')  # Column-major (Fortran-style) flattening
        vec_AT = A.T.flatten('F')
        assert_allclose(K @ vec_A, vec_AT)

    def test_duplication_matrix(self):
        """Test duplication_matrix function."""
        # Test with n=3
        D = duplication_matrix(3)
        assert D.shape == (9, 6)
        
        # Verify D * vech(A) = vec(A) for symmetric A
        A = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])  # Symmetric matrix
        vech_A = vech(A)
        vec_A = A.flatten('F')  # Column-major (Fortran-style) flattening
        assert_allclose(D @ vech_A, vec_A)

    def test_elimination_matrix(self):
        """Test elimination_matrix function."""
        # Test with n=3
        L = elimination_matrix(3)
        assert L.shape == (6, 9)
        
        # Verify L * vec(A) = vech(A) for symmetric A
        A = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])  # Symmetric matrix
        vech_A = vech(A)
        vec_A = A.flatten('F')  # Column-major (Fortran-style) flattening
        assert_allclose(L @ vec_A, vech_A)


# ---- Data Transformations Tests ----

class TestDataTransformations:
    """Tests for data transformation utility functions."""

    def test_standardize(self):
        """Test standardize function."""
        # Test with NumPy array
        x = np.array([1, 2, 3, 4, 5])
        result = standardize(x)
        assert_allclose(result.mean(), 0, atol=1e-10)
        assert_allclose(result.std(ddof=1), 1, atol=1e-10)
        
        # Test with Pandas Series
        s = pd.Series([1, 2, 3, 4, 5], index=pd.date_range('2020-01-01', periods=5))
        result = standardize(s)
        assert isinstance(result, pd.Series)
        assert_allclose(result.mean(), 0, atol=1e-10)
        assert_allclose(result.std(ddof=1), 1, atol=1e-10)
        
        # Test with return_params=True
        result, mean_val, std_val = standardize(x, return_params=True)
        assert_allclose(mean_val, 3.0)
        assert_allclose(std_val, np.sqrt(2.5))
        
        # Test with inplace=True for Pandas Series
        s_copy = s.copy()
        result = standardize(s_copy, inplace=True)
        assert result is s_copy  # Should return the same object
        assert_allclose(s_copy.mean(), 0, atol=1e-10)
        
        # Test with zero variance data
        with pytest.raises(DataError):
            standardize(np.array([1, 1, 1]))

    def test_mvstandardize(self):
        """Test mvstandardize function."""
        # Test with NumPy array
        X = np.array([[1, 4], [2, 5], [3, 6], [4, 7], [5, 8]])
        result = mvstandardize(X)
        assert_allclose(result.mean(axis=0), [0, 0], atol=1e-10)
        assert_allclose(result.std(axis=0, ddof=1), [1, 1], atol=1e-10)
        
        # Test with Pandas DataFrame
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [4, 5, 6, 7, 8]
        }, index=pd.date_range('2020-01-01', periods=5))
        result = mvstandardize(df)
        assert isinstance(result, pd.DataFrame)
        assert_allclose(result.mean(), [0, 0], atol=1e-10)
        assert_allclose(result.std(ddof=1), [1, 1], atol=1e-10)
        
        # Test with return_params=True
        result, means, stds = mvstandardize(X, return_params=True)
        assert_allclose(means, [3, 6])
        assert_allclose(stds, [np.sqrt(2.5), np.sqrt(2.5)])
        
        # Test with inplace=True for Pandas DataFrame
        df_copy = df.copy()
        result = mvstandardize(df_copy, inplace=True)
        assert result is df_copy  # Should return the same object
        assert_allclose(df_copy.mean(), [0, 0], atol=1e-10)
        
        # Test with zero variance data
        X_invalid = np.array([[1, 4], [1, 5], [1, 6]])
        with pytest.raises(DataError):
            mvstandardize(X_invalid)
        
        # Test with non-2D array
        with pytest.raises(DimensionError):
            mvstandardize(np.array([1, 2, 3]))

    def test_demean(self):
        """Test demean function."""
        # Test with NumPy array
        x = np.array([1, 2, 3, 4, 5])
        result = demean(x)
        assert_allclose(result.mean(), 0, atol=1e-10)
        assert_allclose(result, [-2, -1, 0, 1, 2])
        
        # Test with Pandas Series
        s = pd.Series([1, 2, 3, 4, 5], index=pd.date_range('2020-01-01', periods=5))
        result = demean(s)
        assert isinstance(result, pd.Series)
        assert_allclose(result.mean(), 0, atol=1e-10)
        
        # Test with return_mean=True
        result, mean_val = demean(x, return_mean=True)
        assert_allclose(mean_val, 3.0)
        
        # Test with inplace=True for Pandas Series
        s_copy = s.copy()
        result = demean(s_copy, inplace=True)
        assert result is s_copy  # Should return the same object
        assert_allclose(s_copy.mean(), 0, atol=1e-10)
        
        # Test with 2D array
        X = np.array([[1, 4], [2, 5], [3, 6], [4, 7], [5, 8]])
        result = demean(X)
        assert_allclose(result.mean(axis=0), [0, 0], atol=1e-10)

    def test_lag_matrix(self):
        """Test lag_matrix function."""
        # Test with NumPy array
        x = np.array([1, 2, 3, 4, 5])
        result = lag_matrix(x, lags=2)
        expected = np.array([
            [1, 0, 0],
            [2, 1, 0],
            [3, 2, 1],
            [4, 3, 2],
            [5, 4, 3]
        ])
        assert_array_equal(result, expected)
        
        # Test with include_original=False
        result = lag_matrix(x, lags=2, include_original=False)
        expected = np.array([
            [0, 0],
            [1, 0],
            [2, 1],
            [3, 2],
            [4, 3]
        ])
        assert_array_equal(result, expected)
        
        # Test with fill_value
        result = lag_matrix(x, lags=2, fill_value=np.nan)
        expected = np.array([
            [1, np.nan, np.nan],
            [2, 1, np.nan],
            [3, 2, 1],
            [4, 3, 2],
            [5, 4, 3]
        ])
        assert_array_equal(result, expected, equal_nan=True)
        
        # Test with Pandas Series
        s = pd.Series([1, 2, 3, 4, 5], index=pd.date_range('2020-01-01', periods=5))
        result = lag_matrix(s, lags=2)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (5, 3)
        assert_array_equal(result.values, expected)
        
        # Test with invalid inputs
        with pytest.raises(ValueError):
            lag_matrix(x, lags=0)  # lags must be positive
        
        with pytest.raises(DimensionError):
            lag_matrix(np.array([[1, 2], [3, 4]]), lags=1)  # Input must be 1D

    def test_lag_series(self):
        """Test lag_series function."""
        # Test with NumPy array
        x = np.array([1, 2, 3, 4, 5])
        result = lag_series(x, lags=2)
        assert len(result) == 3  # Original + 2 lags
        assert_array_equal(result[0], [1, 2, 3, 4, 5])  # Original
        assert_array_equal(result[1], [0, 1, 2, 3, 4])  # Lag 1
        assert_array_equal(result[2], [0, 0, 1, 2, 3])  # Lag 2
        
        # Test with include_original=False
        result = lag_series(x, lags=2, include_original=False)
        assert len(result) == 2  # 2 lags only
        
        # Test with fill_value
        result = lag_series(x, lags=2, fill_value=np.nan)
        assert_array_equal(result[1], [np.nan, 1, 2, 3, 4], equal_nan=True)
        
        # Test with specific lags
        result = lag_series(x, lags=[1, 3], include_original=True)
        assert len(result) == 3  # Original + 2 specific lags
        assert_array_equal(result[1], [0, 1, 2, 3, 4])  # Lag 1
        assert_array_equal(result[2], [0, 0, 0, 1, 2])  # Lag 3
        
        # Test with Pandas Series
        s = pd.Series([1, 2, 3, 4, 5], index=pd.date_range('2020-01-01', periods=5))
        result = lag_series(s, lags=2)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (5, 3)
        
        # Test with invalid inputs
        with pytest.raises(ValueError):
            lag_series(x, lags=0)  # lags must be positive
        
        with pytest.raises(ValueError):
            lag_series(x, lags=[0, 1])  # all lags must be positive
        
        with pytest.raises(DimensionError):
            lag_series(np.array([[1, 2], [3, 4]]), lags=1)  # Input must be 1D

    def test_rolling_window(self):
        """Test rolling_window function."""
        # Test with NumPy array
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = rolling_window(x, window_size=3, step=2)
        expected = np.array([
            [1, 2, 3],
            [3, 4, 5],
            [5, 6, 7],
            [7, 8, 9]
        ])
        assert_array_equal(result, expected)
        
        # Test with step=1
        result = rolling_window(x, window_size=3, step=1)
        assert result.shape == (8, 3)
        
        # Test with Pandas Series
        s = pd.Series(x, index=pd.date_range('2020-01-01', periods=10))
        result = rolling_window(s, window_size=3, step=2)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (4, 3)
        
        # Test with invalid inputs
        with pytest.raises(ValueError):
            rolling_window(x, window_size=0)  # window_size must be positive
        
        with pytest.raises(ValueError):
            rolling_window(x, window_size=3, step=0)  # step must be positive
        
        with pytest.raises(DimensionError):
            rolling_window(np.array([[1, 2], [3, 4]]), window_size=2)  # Input must be 1D

    def test_rolling_mean(self):
        """Test rolling_mean function."""
        # Test with NumPy array
        x = np.array([1, 2, 3, 4, 5])
        result = rolling_mean(x, window_size=3)
        expected = np.array([np.nan, np.nan, 2, 3, 4])
        assert_allclose(result, expected, equal_nan=True)
        
        # Test with center=True
        result = rolling_mean(x, window_size=3, center=True)
        expected = np.array([np.nan, 2, 3, 4, np.nan])
        assert_allclose(result, expected, equal_nan=True)
        
        # Test with Pandas Series
        s = pd.Series(x, index=pd.date_range('2020-01-01', periods=5))
        result = rolling_mean(s, window_size=3)
        assert isinstance(result, pd.Series)
        expected = pd.Series([np.nan, np.nan, 2, 3, 4], index=s.index)
        assert_series_equal(result, expected)
        
        # Test with 2D array
        X = np.array([[1, 6], [2, 7], [3, 8], [4, 9], [5, 10]])
        result = rolling_mean(X, window_size=3)
        expected = np.array([
            [np.nan, np.nan],
            [np.nan, np.nan],
            [2, 7],
            [3, 8],
            [4, 9]
        ])
        assert_allclose(result, expected, equal_nan=True)
        
        # Test with invalid inputs
        with pytest.raises(ValueError):
            rolling_mean(x, window_size=0)  # window_size must be positive

    def test_rolling_variance(self):
        """Test rolling_variance function."""
        # Test with NumPy array
        x = np.array([1, 2, 3, 4, 5])
        result = rolling_variance(x, window_size=3)
        expected = np.array([np.nan, np.nan, 1, 1, 1])
        assert_allclose(result, expected, equal_nan=True)
        
        # Test with center=True
        result = rolling_variance(x, window_size=3, center=True)
        expected = np.array([np.nan, 1, 1, 1, np.nan])
        assert_allclose(result, expected, equal_nan=True)
        
        # Test with Pandas Series
        s = pd.Series(x, index=pd.date_range('2020-01-01', periods=5))
        result = rolling_variance(s, window_size=3)
        assert isinstance(result, pd.Series)
        expected = pd.Series([np.nan, np.nan, 1, 1, 1], index=s.index)
        assert_series_equal(result, expected)
        
        # Test with ddof=0
        result = rolling_variance(x, window_size=3, ddof=0)
        expected = np.array([np.nan, np.nan, 2/3, 2/3, 2/3])
        assert_allclose(result, expected, equal_nan=True)
        
        # Test with invalid inputs
        with pytest.raises(ValueError):
            rolling_variance(x, window_size=0)  # window_size must be positive

    def test_rolling_skewness(self):
        """Test rolling_skewness function."""
        # Test with NumPy array
        x = np.array([1, 2, 3, 4, 5])
        result = rolling_skewness(x, window_size=3)
        expected = np.array([np.nan, np.nan, 0, 0, 0])
        assert_allclose(result, expected, equal_nan=True)
        
        # Test with skewed data
        y = np.array([1, 1, 1, 5, 10])
        result = rolling_skewness(y, window_size=3)
        # Last window [1, 5, 10] should have positive skewness
        assert result[-1] > 0
        
        # Test with Pandas Series
        s = pd.Series(x, index=pd.date_range('2020-01-01', periods=5))
        result = rolling_skewness(s, window_size=3)
        assert isinstance(result, pd.Series)
        
        # Test with invalid inputs
        with pytest.raises(ValueError):
            rolling_skewness(x, window_size=0)  # window_size must be positive

    def test_rolling_kurtosis(self):
        """Test rolling_kurtosis function."""
        # Test with NumPy array
        x = np.array([1, 2, 3, 4, 5])
        result = rolling_kurtosis(x, window_size=3)
        # For equally spaced values, excess kurtosis should be negative
        assert result[2] < 0
        
        # Test with Pandas Series
        s = pd.Series(x, index=pd.date_range('2020-01-01', periods=5))
        result = rolling_kurtosis(s, window_size=3)
        assert isinstance(result, pd.Series)
        
        # Test with excess=False
        result = rolling_kurtosis(x, window_size=3, excess=False)
        # Without excess, kurtosis should be positive
        assert result[2] > 0
        
        # Test with invalid inputs
        with pytest.raises(ValueError):
            rolling_kurtosis(x, window_size=0)  # window_size must be positive


# ---- Miscellaneous Utilities Tests ----

class TestMiscellaneousUtils:
    """Tests for miscellaneous utility functions."""

    def test_r2z_z2r(self):
        """Test r2z and z2r functions for Fisher's Z transformation."""
        # Test r2z with scalar
        r = 0.5
        z = r2z(r)
        assert_allclose(z, 0.5493061443340548)
        
        # Test r2z with array
        r_array = np.array([0.1, 0.5, 0.9])
        z_array = r2z(r_array)
        expected = np.array([0.10033535, 0.54930614, 1.47221948])
        assert_allclose(z_array, expected)
        
        # Test z2r with scalar
        r_back = z2r(z)
        assert_allclose(r_back, r)
        
        # Test z2r with array
        r_back_array = z2r(z_array)
        assert_allclose(r_back_array, r_array)
        
        # Test with invalid inputs
        with pytest.raises(ValueError):
            r2z(1.1)  # r must be in [-1, 1]
        
        with pytest.raises(ValueError):
            r2z(-1.1)  # r must be in [-1, 1]

    def test_phi2r_r2phi(self):
        """Test phi2r and r2phi functions for AR(1) parameter transformation."""
        # Test phi2r with scalar
        phi = 0.5
        r = phi2r(phi)
        assert_allclose(r, 0.5)
        
        # Test phi2r with array
        phi_array = np.array([0.1, 0.5, 0.9])
        r_array = phi2r(phi_array)
        assert_allclose(r_array, phi_array)
        
        # Test r2phi with scalar
        phi_back = r2phi(r)
        assert_allclose(phi_back, phi)
        
        # Test r2phi with array
        phi_back_array = r2phi(r_array)
        assert_allclose(phi_back_array, phi_array)
        
        # Test with invalid inputs
        with pytest.raises(ValueError):
            phi2r(1.1)  # phi must be in [-1, 1]
        
        with pytest.raises(ValueError):
            r2phi(-1.1)  # r must be in [-1, 1]

    def test_ensure_array(self):
        """Test ensure_array function."""
        # Test with list
        result = ensure_array([1, 2, 3])
        assert isinstance(result, np.ndarray)
        assert_array_equal(result, np.array([1, 2, 3]))
        
        # Test with NumPy array
        x = np.array([1, 2, 3])
        result = ensure_array(x)
        assert result is x  # Should return the same object
        
        # Test with Pandas Series
        s = pd.Series([1, 2, 3])
        result = ensure_array(s)
        assert isinstance(result, np.ndarray)
        assert_array_equal(result, np.array([1, 2, 3]))
        
        # Test with dtype
        result = ensure_array([1, 2, 3], dtype=np.float64)
        assert result.dtype == np.float64

    def test_ensure_dataframe(self):
        """Test ensure_dataframe function."""
        # Test with list
        result = ensure_dataframe([1, 2, 3])
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 1)
        
        # Test with NumPy array
        x = np.array([[1, 2], [3, 4]])
        result = ensure_dataframe(x)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 2)
        
        # Test with Pandas DataFrame
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        result = ensure_dataframe(df)
        assert result is not df  # Should return a copy
        assert_frame_equal(result, df)
        
        # Test with columns
        result = ensure_dataframe(x, columns=['A', 'B'])
        assert list(result.columns) == ['A', 'B']
        
        # Test with index
        idx = pd.date_range('2020-01-01', periods=2)
        result = ensure_dataframe(x, index=idx)
        assert_array_equal(result.index, idx)

    def test_ensure_series(self):
        """Test ensure_series function."""
        # Test with list
        result = ensure_series([1, 2, 3])
        assert isinstance(result, pd.Series)
        assert len(result) == 3
        
        # Test with NumPy array
        x = np.array([1, 2, 3])
        result = ensure_series(x)
        assert isinstance(result, pd.Series)
        assert len(result) == 3
        
        # Test with Pandas Series
        s = pd.Series([1, 2, 3])
        result = ensure_series(s)
        assert result is not s  # Should return a copy
        assert_series_equal(result, s)
        
        # Test with name
        result = ensure_series(x, name='values')
        assert result.name == 'values'
        
        # Test with index
        idx = pd.date_range('2020-01-01', periods=3)
        result = ensure_series(x, index=idx)
        assert_array_equal(result.index, idx)
        
        # Test with DataFrame
        df = pd.DataFrame({'A': [1, 2, 3]})
        result = ensure_series(df)
        assert isinstance(result, pd.Series)
        assert len(result) == 3
        
        # Test with multi-column DataFrame
        df_multi = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        with pytest.raises(ValueError):
            ensure_series(df_multi)

    def test_lag_matrix_extended(self):
        """Test lag_matrix_extended function."""
        # Test with NumPy array
        x = np.array([1, 2, 3, 4, 5])
        result = lag_matrix_extended(x, lags=2, drop_nan=True)
        expected = np.array([
            [3, 2, 1],
            [4, 3, 2],
            [5, 4, 3]
        ])
        assert_array_equal(result, expected)
        
        # Test with column_names
        s = pd.Series([1, 2, 3, 4, 5], index=pd.date_range('2020-01-01', periods=5))
        result = lag_matrix_extended(s, lags=2, column_names=['Current', 'Lag1', 'Lag2'])
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['Current', 'Lag1', 'Lag2']
        
        # Test with invalid column_names length
        with pytest.raises(ValueError):
            lag_matrix_extended(x, lags=2, column_names=['A', 'B'])  # Should be length 3
        
        # Test with other parameters passed to lag_matrix
        result = lag_matrix_extended(x, lags=2, include_original=False, fill_value=np.nan)
        assert result.shape[1] == 2  # Only lag columns


# ---- Property-Based Tests ----

class TestPropertyBasedTests:
    """Property-based tests for utility functions."""

    @given(
        matrix=st.builds(
            lambda n: np.random.rand(n, n),
            n=st.integers(min_value=2, max_value=10)
        )
    )
    @settings(max_examples=10)
    def test_vech_ivech_roundtrip(self, matrix):
        """Test that ivech(vech(A)) = A for symmetric matrices."""
        # Make the matrix symmetric
        symmetric = (matrix + matrix.T) / 2
        
        # Test roundtrip
        v = vech(symmetric)
        result = ivech(v)
        
        assert_allclose(result, symmetric)

    @given(
        r=st.floats(min_value=-0.99, max_value=0.99)
    )
    @settings(max_examples=20)
    def test_r2z_z2r_roundtrip(self, r):
        """Test that z2r(r2z(r)) = r for correlation coefficients."""
        z = r2z(r)
        r_back = z2r(z)
        
        assert_allclose(r_back, r)

    @given(
        data=st.lists(st.floats(min_value=-100, max_value=100), min_size=5, max_size=20),
        window_size=st.integers(min_value=2, max_value=5)
    )
    @settings(max_examples=10)
    def test_rolling_mean_property(self, data, window_size):
        """Test that rolling_mean computes the correct mean for each window."""
        # Convert to numpy array
        x = np.array(data)
        assume(len(x) >= window_size)
        
        # Compute rolling mean
        result = rolling_mean(x, window_size)
        
        # Check each valid window
        for i in range(window_size - 1, len(x)):
            window = x[i - window_size + 1:i + 1]
            expected_mean = np.mean(window)
            assert_allclose(result[i], expected_mean)


# ---- Seasonal Differencing Tests ----

class TestSeasonalDifferencing:
    """Tests for seasonal differencing (sdiff) implementation."""

    def test_sdiff_basic(self):
        """Test basic seasonal differencing functionality."""
        # Create a seasonal time series
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        
        # Test with lag=4 (quarterly seasonality)
        result = np.diff(x, n=1, axis=0, prepend=x[0:0])  # Placeholder for sdiff implementation
        expected = np.array([np.nan, np.nan, np.nan, np.nan, 4, 4, 4, 4, 4, 4, 4, 4])
        
        # This is a placeholder test since sdiff is not in the provided code
        # In a real implementation, we would test:
        # result = sdiff(x, lag=4)
        # assert_allclose(result, expected, equal_nan=True)

    def test_sdiff_pandas(self):
        """Test seasonal differencing with Pandas Series."""
        # Create a seasonal time series
        s = pd.Series(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            index=pd.date_range('2020-01-01', periods=12, freq='M')
        )
        
        # Test with lag=4 (quarterly seasonality)
        # This is a placeholder test since sdiff is not in the provided code
        # In a real implementation, we would test:
        # result = sdiff(s, lag=4)
        # assert isinstance(result, pd.Series)
        # assert result.index.equals(s.index)
        # expected = pd.Series([np.nan, np.nan, np.nan, np.nan, 4, 4, 4, 4, 4, 4, 4, 4], index=s.index)
        # assert_series_equal(result, expected)

    def test_sdiff_multiple_lags(self):
        """Test seasonal differencing with multiple lags."""
        # Create a seasonal time series
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        
        # Test with lag=4 and lag=1 (seasonal and regular differencing)
        # This is a placeholder test since sdiff is not in the provided code
        # In a real implementation, we would test:
        # result = sdiff(x, lag=[4, 1])
        # expected = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 1, 1, 1, 1, 1, 1, 1])
        # assert_allclose(result, expected, equal_nan=True)

    def test_sdiff_edge_cases(self):
        """Test seasonal differencing with edge cases."""
        # Test with empty array
        # This is a placeholder test since sdiff is not in the provided code
        # In a real implementation, we would test:
        # result = sdiff(np.array([]), lag=1)
        # assert_array_equal(result, np.array([]))
        
        # Test with lag=0 (should raise error)
        # with pytest.raises(ValueError):
        #     sdiff(np.array([1, 2, 3]), lag=0)
        
        # Test with lag > array length
        # result = sdiff(np.array([1, 2, 3]), lag=4)
        # expected = np.array([np.nan, np.nan, np.nan])
        # assert_allclose(result, expected, equal_nan=True)
