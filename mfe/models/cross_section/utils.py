# mfe/models/cross_section/utils.py
"""
Cross-Sectional Analysis Utilities Module

This module provides specialized utility functions for cross-sectional analysis,
including robust error estimation, data transformation, diagnostic tests, and
matrix manipulation helpers. These functions support both OLS regression and
Principal Component Analysis implementations in the MFE Toolbox.

The module implements optimized versions of cross-sectional utilities using NumPy's
efficient array operations and Numba's JIT compilation for performance-critical
functions. All functions include comprehensive type hints and input validation
to ensure reliability and proper error handling.

Functions:
    add_constant: Add a constant term to a design matrix
    remove_constant: Remove a constant term from a design matrix
    standardize_data: Standardize data by removing mean and scaling by std dev
    compute_vif: Compute variance inflation factors for multicollinearity detection
    compute_condition_number: Compute condition number of a design matrix
    compute_white_se: Compute White's heteroskedasticity-robust standard errors
    compute_hc_se: Compute HC0-HC3 heteroskedasticity-consistent standard errors
    compute_jarque_bera: Compute Jarque-Bera test for normality
    compute_breusch_pagan: Compute Breusch-Pagan test for heteroskedasticity
    compute_durbin_watson: Compute Durbin-Watson statistic for autocorrelation
    cross_validate: Perform k-fold cross-validation for regression models
    compute_press: Compute PRESS statistic (prediction sum of squares)
    compute_leverage: Compute leverage (hat) values for observations
    compute_influence: Compute influence measures (Cook's distance, DFFITS)
    compute_partial_corr: Compute partial correlation matrix
    compute_eigenvalue_bootstrap: Bootstrap confidence intervals for eigenvalues
    compute_loading_bootstrap: Bootstrap confidence intervals for PCA loadings
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Literal, Optional, Tuple, Union, 
    cast, overload, TypeVar, Protocol
)

import numpy as np
import pandas as pd
from scipy import stats, linalg
from scipy.optimize import minimize

from mfe.core.exceptions import (
    DimensionError, NumericError, ParameterError, DataError,
    raise_dimension_error, raise_numeric_error, raise_parameter_error, 
    raise_data_error, warn_numeric
)
from mfe.core.types import Matrix, Vector
from mfe.utils.matrix_ops import ensure_symmetric, is_positive_definite, nearest_positive_definite

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    HAS_NUMBA = False

# Set up module-level logger
logger = logging.getLogger("mfe.models.cross_section.utils")


def add_constant(X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
    """
    Add a constant term (column of ones) to a design matrix.
    
    This function adds a column of ones to the left side of a design matrix
    for regression analysis. If the input is a pandas DataFrame, the constant
    column will be named 'const'.
    
    Args:
        X: Design matrix (n_samples x n_features)
        
    Returns:
        Design matrix with constant term added (n_samples x (n_features+1))
        
    Raises:
        DimensionError: If X is not a 2D array or DataFrame
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.cross_section.utils import add_constant
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> add_constant(X)
        array([[1., 1., 2.],
               [1., 3., 4.],
               [1., 5., 6.]])
    """
    # Check if X is a pandas DataFrame
    if isinstance(X, pd.DataFrame):
        if 'const' in X.columns:
            # Check if the constant column is already all ones
            if X['const'].equals(pd.Series(np.ones(len(X)), index=X.index)):
                return X
            else:
                warnings.warn("DataFrame already contains a 'const' column that is not all ones. "
                             "Renaming existing column to 'const_0' and adding new constant.")
                X = X.rename(columns={'const': 'const_0'})
        
        # Add constant column
        result = pd.DataFrame({'const': np.ones(len(X))}, index=X.index)
        result = pd.concat([result, X], axis=1)
        return result
    
    # Handle numpy array
    X_array = np.asarray(X)
    
    # Check if X is 2D
    if X_array.ndim != 2:
        raise_dimension_error(
            "Input must be a 2D array",
            array_name="X",
            expected_shape="(n_samples, n_features)",
            actual_shape=X_array.shape
        )
    
    # Check if the first column is already all ones
    n_samples = X_array.shape[0]
    if X_array.shape[1] > 0 and np.allclose(X_array[:, 0], np.ones(n_samples)):
        return X_array
    
    # Add constant column
    X_with_const = np.column_stack((np.ones(n_samples), X_array))
    
    return X_with_const


def remove_constant(X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
    """
    Remove a constant term (column of ones) from a design matrix.
    
    This function removes a column of ones from the left side of a design matrix.
    If the input is a pandas DataFrame, it will remove a column named 'const'.
    
    Args:
        X: Design matrix with constant term (n_samples x n_features)
        
    Returns:
        Design matrix without constant term (n_samples x (n_features-1))
        
    Raises:
        DimensionError: If X is not a 2D array or DataFrame
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.cross_section.utils import remove_constant
        >>> X = np.array([[1, 1, 2], [1, 3, 4], [1, 5, 6]])
        >>> remove_constant(X)
        array([[1, 2],
               [3, 4],
               [5, 6]])
    """
    # Check if X is a pandas DataFrame
    if isinstance(X, pd.DataFrame):
        if 'const' in X.columns:
            return X.drop('const', axis=1)
        else:
            return X
    
    # Handle numpy array
    X_array = np.asarray(X)
    
    # Check if X is 2D
    if X_array.ndim != 2:
        raise_dimension_error(
            "Input must be a 2D array",
            array_name="X",
            expected_shape="(n_samples, n_features)",
            actual_shape=X_array.shape
        )
    
    # Check if the first column is all ones
    n_samples = X_array.shape[0]
    if X_array.shape[1] > 0 and np.allclose(X_array[:, 0], np.ones(n_samples)):
        return X_array[:, 1:]
    else:
        return X_array


def standardize_data(X: Union[np.ndarray, pd.DataFrame], 
                    center: bool = True, 
                    scale: bool = True,
                    skip_constant: bool = True) -> Tuple[Union[np.ndarray, pd.DataFrame], np.ndarray, np.ndarray]:
    """
    Standardize data by removing mean and scaling by standard deviation.
    
    This function standardizes a design matrix by centering (subtracting the mean)
    and scaling (dividing by the standard deviation). It can optionally skip
    standardizing a constant column.
    
    Args:
        X: Design matrix (n_samples x n_features)
        center: Whether to center the data (subtract mean)
        scale: Whether to scale the data (divide by std)
        skip_constant: Whether to skip standardizing a constant column
        
    Returns:
        Tuple containing:
            - Standardized design matrix
            - Mean of each column (or zeros if center=False)
            - Standard deviation of each column (or ones if scale=False)
        
    Raises:
        DimensionError: If X is not a 2D array or DataFrame
        NumericError: If any column has zero standard deviation
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.cross_section.utils import standardize_data
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> X_std, mean, std = standardize_data(X)
        >>> X_std
        array([[-1.22474487, -1.22474487],
               [ 0.        ,  0.        ],
               [ 1.22474487,  1.22474487]])
        >>> mean
        array([3., 4.])
        >>> std
        array([1.63299316, 1.63299316])
    """
    # Check if X is a pandas DataFrame
    is_dataframe = isinstance(X, pd.DataFrame)
    
    # Convert to numpy array for processing
    X_array = X.values if is_dataframe else np.asarray(X)
    
    # Check if X is 2D
    if X_array.ndim != 2:
        raise_dimension_error(
            "Input must be a 2D array",
            array_name="X",
            expected_shape="(n_samples, n_features)",
            actual_shape=X_array.shape
        )
    
    n_samples, n_features = X_array.shape
    
    # Initialize mean and std arrays
    mean = np.zeros(n_features)
    std = np.ones(n_features)
    
    # Identify constant column if needed
    const_col = None
    if skip_constant and n_features > 0:
        # Check if the first column is all ones
        if np.allclose(X_array[:, 0], np.ones(n_samples)):
            const_col = 0
        # If it's a DataFrame, check for 'const' column
        elif is_dataframe and 'const' in X.columns:
            const_col = list(X.columns).index('const')
    
    # Compute mean if centering
    if center:
        mean = np.mean(X_array, axis=0)
        if const_col is not None:
            mean[const_col] = 0  # Don't center constant column
    
    # Compute std if scaling
    if scale:
        std = np.std(X_array, axis=0, ddof=1)
        if const_col is not None:
            std[const_col] = 1  # Don't scale constant column
        
        # Check for zero standard deviation
        zero_std_cols = np.where(std == 0)[0]
        if len(zero_std_cols) > 0:
            raise_numeric_error(
                f"Columns {zero_std_cols} have zero standard deviation, cannot scale",
                operation="standardize_data",
                values=std,
                error_type="zero_std_dev"
            )
    
    # Standardize the data
    X_std = X_array.copy()
    if center:
        X_std -= mean
    if scale:
        X_std /= std
    
    # Convert back to DataFrame if input was DataFrame
    if is_dataframe:
        X_std = pd.DataFrame(X_std, index=X.index, columns=X.columns)
    
    return X_std, mean, std


@jit(nopython=True, cache=True)
def _compute_vif_numba(X: np.ndarray, skip_constant: bool = True) -> np.ndarray:
    """
    Numba-accelerated implementation of variance inflation factor computation.
    
    Args:
        X: Design matrix (n_samples x n_features)
        skip_constant: Whether to skip computing VIF for a constant column
        
    Returns:
        Array of VIF values for each column
    """
    n_samples, n_features = X.shape
    vif = np.ones(n_features)
    
    start_idx = 1 if skip_constant else 0
    
    for i in range(start_idx, n_features):
        # Create X matrix without the current variable
        X_without_i = np.zeros((n_samples, n_features - 1))
        col_idx = 0
        for j in range(n_features):
            if j != i:
                X_without_i[:, col_idx] = X[:, j]
                col_idx += 1
        
        # Compute X'X and X'y for the regression
        XtX = np.zeros((n_features - 1, n_features - 1))
        Xty = np.zeros(n_features - 1)
        
        for j in range(n_features - 1):
            Xty[j] = np.sum(X_without_i[:, j] * X[:, i])
            for k in range(n_features - 1):
                XtX[j, k] = np.sum(X_without_i[:, j] * X_without_i[:, k])
        
        # Solve for beta: (X'X)^(-1) X'y
        # Note: In a real implementation, we would use a more robust method
        # This is a simplified version for demonstration
        # In practice, we would use np.linalg.solve, but that's not supported in numba
        
        # Compute inverse of X'X using Gaussian elimination
        # This is a simplified implementation and may not be numerically stable
        # for all matrices, but works for demonstration purposes
        n = n_features - 1
        L = np.eye(n)
        U = XtX.copy()
        
        for k in range(n-1):
            for i in range(k+1, n):
                L[i, k] = U[i, k] / U[k, k]
                for j in range(k, n):
                    U[i, j] = U[i, j] - L[i, k] * U[k, j]
        
        # Solve Ly = b
        y = np.zeros(n)
        for i in range(n):
            y[i] = Xty[i]
            for j in range(i):
                y[i] -= L[i, j] * y[j]
        
        # Solve Ux = y
        beta = np.zeros(n)
        for i in range(n-1, -1, -1):
            beta[i] = y[i]
            for j in range(i+1, n):
                beta[i] -= U[i, j] * beta[j]
            beta[i] /= U[i, i]
        
        # Compute R-squared
        y_pred = np.zeros(n_samples)
        for j in range(n_features - 1):
            y_pred += X_without_i[:, j] * beta[j]
        
        y_mean = np.mean(X[:, i])
        ss_total = np.sum((X[:, i] - y_mean) ** 2)
        ss_residual = np.sum((X[:, i] - y_pred) ** 2)
        r_squared = 1 - (ss_residual / ss_total)
        
        # Compute VIF
        if r_squared < 1.0:  # Avoid division by zero
            vif[i] = 1.0 / (1.0 - r_squared)
        else:
            vif[i] = float('inf')
    
    return vif


def compute_vif(X: Union[np.ndarray, pd.DataFrame], 
               skip_constant: bool = True) -> Union[np.ndarray, Dict[str, float]]:
    """
    Compute variance inflation factors for multicollinearity detection.
    
    This function computes the variance inflation factor (VIF) for each variable
    in a design matrix. VIF measures how much the variance of a regression coefficient
    is inflated due to multicollinearity with other predictors.
    
    Args:
        X: Design matrix (n_samples x n_features)
        skip_constant: Whether to skip computing VIF for a constant column
        
    Returns:
        If X is a numpy array: Array of VIF values for each column
        If X is a pandas DataFrame: Dictionary mapping column names to VIF values
        
    Raises:
        DimensionError: If X is not a 2D array or DataFrame
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.cross_section.utils import compute_vif
        >>> X = np.array([[1, 1, 2], [1, 3, 4], [1, 5, 6]])
        >>> compute_vif(X)
        array([1.        , 1.        , 1.        ])
    """
    # Check if X is a pandas DataFrame
    is_dataframe = isinstance(X, pd.DataFrame)
    
    # Convert to numpy array for processing
    X_array = X.values if is_dataframe else np.asarray(X)
    
    # Check if X is a 2D array
    if X_array.ndim != 2:
        raise_dimension_error(
            "Input must be a 2D array",
            array_name="X",
            expected_shape="(n_samples, n_features)",
            actual_shape=X_array.shape
        )
    
    n_samples, n_features = X_array.shape
    
    # Check if we have enough samples
    if n_samples <= n_features:
        warnings.warn("Number of samples must be greater than number of features "
                     "for reliable VIF computation. Results may be unreliable.")
    
    # Use Numba-accelerated implementation if available
    if HAS_NUMBA:
        vif_values = _compute_vif_numba(X_array, skip_constant)
    else:
        # Pure NumPy/SciPy implementation
        vif_values = np.ones(n_features)
        
        start_idx = 1 if skip_constant else 0
        
        for i in range(start_idx, n_features):
            # Create X matrix without the current variable
            mask = np.ones(n_features, dtype=bool)
            mask[i] = False
            X_without_i = X_array[:, mask]
            
            # Regress the current variable on all other variables
            try:
                # Use OLS to compute R-squared
                # (X'X)^(-1) X'y
                XtX = X_without_i.T @ X_without_i
                Xty = X_without_i.T @ X_array[:, i]
                beta = np.linalg.solve(XtX, Xty)
                
                # Compute R-squared
                y_pred = X_without_i @ beta
                y_mean = np.mean(X_array[:, i])
                ss_total = np.sum((X_array[:, i] - y_mean) ** 2)
                ss_residual = np.sum((X_array[:, i] - y_pred) ** 2)
                r_squared = 1 - (ss_residual / ss_total)
                
                # Compute VIF
                if r_squared < 1.0:  # Avoid division by zero
                    vif_values[i] = 1.0 / (1.0 - r_squared)
                else:
                    vif_values[i] = float('inf')
            
            except np.linalg.LinAlgError:
                # If the matrix is singular, set VIF to infinity
                vif_values[i] = float('inf')
    
    # Return as dictionary if input was DataFrame
    if is_dataframe:
        column_names = X.columns
        return {col: vif_values[i] for i, col in enumerate(column_names)}
    else:
        return vif_values


def compute_condition_number(X: Union[np.ndarray, pd.DataFrame]) -> float:
    """
    Compute the condition number of a design matrix.
    
    This function computes the condition number of a design matrix, which is
    the ratio of the largest to smallest singular value. A high condition number
    indicates potential multicollinearity issues.
    
    Args:
        X: Design matrix (n_samples x n_features)
        
    Returns:
        Condition number of the design matrix
        
    Raises:
        DimensionError: If X is not a 2D array or DataFrame
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.cross_section.utils import compute_condition_number
        >>> X = np.array([[1, 1, 2], [1, 3, 4], [1, 5, 6]])
        >>> compute_condition_number(X)
        17.92
    """
    # Convert to numpy array if needed
    X_array = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    
    # Check if X is a 2D array
    if X_array.ndim != 2:
        raise_dimension_error(
            "Input must be a 2D array",
            array_name="X",
            expected_shape="(n_samples, n_features)",
            actual_shape=X_array.shape
        )
    
    # Compute singular values
    s = np.linalg.svd(X_array, compute_uv=False)
    
    # Compute condition number
    condition_number = s[0] / s[-1]
    
    return condition_number


def compute_white_se(X: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """
    Compute White's heteroskedasticity-robust standard errors.
    
    This function computes heteroskedasticity-robust standard errors for OLS
    regression coefficients using White's method.
    
    Args:
        X: Design matrix (n_samples x n_features)
        residuals: Residuals from OLS regression (n_samples)
        
    Returns:
        Array of robust standard errors for each coefficient
        
    Raises:
        DimensionError: If X is not a 2D array or residuals is not a 1D array
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.cross_section.utils import compute_white_se
        >>> X = np.array([[1, 1], [1, 2], [1, 3]])
        >>> residuals = np.array([0.1, -0.2, 0.1])
        >>> compute_white_se(X, residuals)
        array([0.11547005, 0.08164966])
    """
    # Convert to numpy arrays if needed
    X = np.asarray(X)
    residuals = np.asarray(residuals)
    
    # Check if X is a 2D array
    if X.ndim != 2:
        raise_dimension_error(
            "X must be a 2D array",
            array_name="X",
            expected_shape="(n_samples, n_features)",
            actual_shape=X.shape
        )
    
    # Check if residuals is a 1D array
    if residuals.ndim != 1:
        raise_dimension_error(
            "residuals must be a 1D array",
            array_name="residuals",
            expected_shape="(n_samples,)",
            actual_shape=residuals.shape
        )
    
    # Check if dimensions match
    n_samples, n_features = X.shape
    if residuals.shape[0] != n_samples:
        raise_dimension_error(
            "Number of residuals must match number of samples in X",
            array_name="residuals",
            expected_shape=f"({n_samples},)",
            actual_shape=residuals.shape
        )
    
    # Compute (X'X)^(-1)
    XtX_inv = np.linalg.inv(X.T @ X)
    
    # Compute X'diag(e^2)X
    squared_residuals = residuals ** 2
    X_weighted = X * squared_residuals[:, np.newaxis]
    middle_term = X.T @ X_weighted
    
    # Compute the robust covariance matrix
    cov_matrix = XtX_inv @ middle_term @ XtX_inv
    
    # Extract the diagonal elements and take the square root
    robust_se = np.sqrt(np.diag(cov_matrix))
    
    return robust_se


def compute_hc_se(X: np.ndarray, residuals: np.ndarray, 
                 cov_type: Literal['hc0', 'hc1', 'hc2', 'hc3'] = 'hc1') -> np.ndarray:
    """
    Compute heteroskedasticity-consistent standard errors.
    
    This function computes heteroskedasticity-consistent (HC) standard errors
    for OLS regression coefficients using various methods (HC0-HC3).
    
    Args:
        X: Design matrix (n_samples x n_features)
        residuals: Residuals from OLS regression (n_samples)
        cov_type: Type of HC standard errors to compute:
                 'hc0': White's original estimator
                 'hc1': HC0 with small sample correction (n/(n-k))
                 'hc2': HC0 with leverage correction
                 'hc3': HC0 with more aggressive leverage correction
        
    Returns:
        Array of robust standard errors for each coefficient
        
    Raises:
        DimensionError: If X is not a 2D array or residuals is not a 1D array
        ValueError: If cov_type is not one of the supported types
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.cross_section.utils import compute_hc_se
        >>> X = np.array([[1, 1], [1, 2], [1, 3]])
        >>> residuals = np.array([0.1, -0.2, 0.1])
        >>> compute_hc_se(X, residuals, cov_type='hc1')
        array([0.14142136, 0.1])
    """
    # Convert to numpy arrays if needed
    X = np.asarray(X)
    residuals = np.asarray(residuals)
    
    # Check if X is a 2D array
    if X.ndim != 2:
        raise_dimension_error(
            "X must be a 2D array",
            array_name="X",
            expected_shape="(n_samples, n_features)",
            actual_shape=X.shape
        )
    
    # Check if residuals is a 1D array
    if residuals.ndim != 1:
        raise_dimension_error(
            "residuals must be a 1D array",
            array_name="residuals",
            expected_shape="(n_samples,)",
            actual_shape=residuals.shape
        )
    
    # Check if dimensions match
    n_samples, n_features = X.shape
    if residuals.shape[0] != n_samples:
        raise_dimension_error(
            "Number of residuals must match number of samples in X",
            array_name="residuals",
            expected_shape=f"({n_samples},)",
            actual_shape=residuals.shape
        )
    
    # Check if cov_type is valid
    valid_cov_types = ['hc0', 'hc1', 'hc2', 'hc3']
    if cov_type.lower() not in valid_cov_types:
        raise ValueError(f"cov_type must be one of {valid_cov_types}, got {cov_type}")
    
    # Compute (X'X)^(-1)
    XtX_inv = np.linalg.inv(X.T @ X)
    
    # Compute hat matrix diagonal (leverage values)
    # H = X(X'X)^(-1)X'
    # We only need the diagonal elements: h_ii = X_i(X'X)^(-1)X_i'
    leverage = np.zeros(n_samples)
    for i in range(n_samples):
        leverage[i] = X[i] @ XtX_inv @ X[i].T
    
    # Adjust residuals based on the specified covariance type
    if cov_type.lower() == 'hc0':
        # White's original estimator
        adjusted_residuals = residuals ** 2
    elif cov_type.lower() == 'hc1':
        # HC0 with small sample correction
        correction = n_samples / (n_samples - n_features)
        adjusted_residuals = correction * (residuals ** 2)
    elif cov_type.lower() == 'hc2':
        # HC0 with leverage correction
        adjusted_residuals = (residuals ** 2) / (1 - leverage)
    elif cov_type.lower() == 'hc3':
        # HC0 with more aggressive leverage correction
        adjusted_residuals = (residuals ** 2) / ((1 - leverage) ** 2)
    
    # Compute X'diag(adjusted_residuals)X
    X_weighted = X * adjusted_residuals[:, np.newaxis]
    middle_term = X.T @ X_weighted
    
    # Compute the robust covariance matrix
    cov_matrix = XtX_inv @ middle_term @ XtX_inv
    
    # Extract the diagonal elements and take the square root
    robust_se = np.sqrt(np.diag(cov_matrix))
    
    return robust_se


def compute_jarque_bera(residuals: np.ndarray) -> Tuple[float, float]:
    """
    Compute Jarque-Bera test for normality of residuals.
    
    This function computes the Jarque-Bera test statistic and p-value for
    testing whether residuals follow a normal distribution.
    
    Args:
        residuals: Residuals from regression (n_samples)
        
    Returns:
        Tuple containing:
            - Jarque-Bera test statistic
            - p-value
        
    Raises:
        DimensionError: If residuals is not a 1D array
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.cross_section.utils import compute_jarque_bera
        >>> residuals = np.array([0.1, -0.2, 0.1, 0.3, -0.1, 0.2])
        >>> jb_stat, p_value = compute_jarque_bera(residuals)
    """
    # Convert to numpy array if needed
    residuals = np.asarray(residuals)
    
    # Check if residuals is a 1D array
    if residuals.ndim != 1:
        raise_dimension_error(
            "residuals must be a 1D array",
            array_name="residuals",
            expected_shape="(n_samples,)",
            actual_shape=residuals.shape
        )
    
    # Compute sample size
    n = len(residuals)
    
    # Compute skewness and kurtosis
    skewness = stats.skew(residuals)
    kurtosis = stats.kurtosis(residuals, fisher=True)  # Fisher's definition (excess kurtosis)
    
    # Compute Jarque-Bera statistic
    jb_stat = n / 6 * (skewness ** 2 + (kurtosis ** 2) / 4)
    
    # Compute p-value (JB follows a chi-squared distribution with 2 degrees of freedom)
    p_value = 1 - stats.chi2.cdf(jb_stat, 2)
    
    return jb_stat, p_value


def compute_breusch_pagan(residuals: np.ndarray, X: np.ndarray) -> Tuple[float, float]:
    """
    Compute Breusch-Pagan test for heteroskedasticity.
    
    This function computes the Breusch-Pagan test statistic and p-value for
    testing whether residuals exhibit heteroskedasticity.
    
    Args:
        residuals: Residuals from regression (n_samples)
        X: Design matrix (n_samples x n_features)
        
    Returns:
        Tuple containing:
            - Breusch-Pagan test statistic
            - p-value
        
    Raises:
        DimensionError: If residuals is not a 1D array or X is not a 2D array
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.cross_section.utils import compute_breusch_pagan
        >>> X = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6]])
        >>> residuals = np.array([0.1, -0.2, 0.1, 0.3, -0.1, 0.2])
        >>> bp_stat, p_value = compute_breusch_pagan(residuals, X)
    """
    # Convert to numpy arrays if needed
    residuals = np.asarray(residuals)
    X = np.asarray(X)
    
    # Check if residuals is a 1D array
    if residuals.ndim != 1:
        raise_dimension_error(
            "residuals must be a 1D array",
            array_name="residuals",
            expected_shape="(n_samples,)",
            actual_shape=residuals.shape
        )
    
    # Check if X is a 2D array
    if X.ndim != 2:
        raise_dimension_error(
            "X must be a 2D array",
            array_name="X",
            expected_shape="(n_samples, n_features)",
            actual_shape=X.shape
        )
    
    # Check if dimensions match
    n_samples, n_features = X.shape
    if residuals.shape[0] != n_samples:
        raise_dimension_error(
            "Number of residuals must match number of samples in X",
            array_name="residuals",
            expected_shape=f"({n_samples},)",
            actual_shape=residuals.shape
        )
    
    # Compute squared residuals
    residuals_squared = residuals ** 2
    
    # Compute mean of squared residuals
    residuals_squared_mean = np.mean(residuals_squared)
    
    # Normalize squared residuals
    u = residuals_squared / residuals_squared_mean - 1
    
    # Regress u on X
    try:
        # Compute OLS coefficients
        beta = np.linalg.solve(X.T @ X, X.T @ u)
        
        # Compute fitted values
        u_hat = X @ beta
        
        # Compute explained sum of squares
        ess = np.sum(u_hat ** 2)
        
        # Compute Breusch-Pagan statistic
        bp_stat = n_samples * ess / 2
        
        # Compute p-value (BP follows a chi-squared distribution with k-1 degrees of freedom)
        p_value = 1 - stats.chi2.cdf(bp_stat, n_features - 1)
        
        return bp_stat, p_value
    
    except np.linalg.LinAlgError:
        # If the matrix is singular, return NaN
        warnings.warn("Singular matrix encountered in Breusch-Pagan test. "
                     "Results may be unreliable.")
        return np.nan, np.nan


def compute_durbin_watson(residuals: np.ndarray) -> float:
    """
    Compute Durbin-Watson statistic for autocorrelation.
    
    This function computes the Durbin-Watson statistic for testing whether
    residuals exhibit first-order autocorrelation.
    
    Args:
        residuals: Residuals from regression (n_samples)
        
    Returns:
        Durbin-Watson statistic
        
    Raises:
        DimensionError: If residuals is not a 1D array
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.cross_section.utils import compute_durbin_watson
        >>> residuals = np.array([0.1, -0.2, 0.1, 0.3, -0.1, 0.2])
        >>> dw_stat = compute_durbin_watson(residuals)
    """
    # Convert to numpy array if needed
    residuals = np.asarray(residuals)
    
    # Check if residuals is a 1D array
    if residuals.ndim != 1:
        raise_dimension_error(
            "residuals must be a 1D array",
            array_name="residuals",
            expectedShape="(n_samples,)",
            actualShape=residuals.shape
        )
    
    # Compute Durbin-Watson statistic
    n = len(residuals)
    diff = np.diff(residuals)
    dw_stat = np.sum(diff ** 2) / np.sum(residuals ** 2)
    
    return dw_stat


def cross_validate(X: np.ndarray, y: np.ndarray, 
                  k_folds: int = 5, 
                  shuffle: bool = True,
                  random_state: Optional[int] = None) -> Dict[str, float]:
    """
    Perform k-fold cross-validation for regression models.
    
    This function performs k-fold cross-validation for OLS regression,
    computing various performance metrics.
    
    Args:
        X: Design matrix (n_samples x n_features)
        y: Target variable (n_samples)
        k_folds: Number of folds for cross-validation
        shuffle: Whether to shuffle the data before splitting
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing cross-validation metrics:
            - 'r2_mean': Mean R-squared across folds
            - 'r2_std': Standard deviation of R-squared across folds
            - 'mse_mean': Mean mean squared error across folds
            - 'mse_std': Standard deviation of mean squared error across folds
            - 'mae_mean': Mean mean absolute error across folds
            - 'mae_std': Standard deviation of mean absolute error across folds
        
    Raises:
        DimensionError: If X is not a 2D array or y is not a 1D array
        ValueError: If k_folds is less than 2
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.cross_section.utils import cross_validate
        >>> X = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6]])
        >>> y = np.array([2, 3, 4, 5, 6, 7])
        >>> cv_results = cross_validate(X, y, k_folds=3)
    """
    # Convert to numpy arrays if needed
    X = np.asarray(X)
    y = np.asarray(y)
    
    # Check if X is a 2D array
    if X.ndim != 2:
        raise_dimension_error(
            "X must be a 2D array",
            array_name="X",
            expectedShape="(n_samples, n_features)",
            actualShape=X.shape
        )
    
    # Check if y is a 1D array
    if y.ndim != 1:
        raise_dimension_error(
            "y must be a 1D array",
            array_name="y",
            expectedShape="(n_samples,)",
            actualShape=y.shape
        )
    
    # Check if dimensions match
    n_samples, n_features = X.shape
    if y.shape[0] != n_samples:
        raise_dimension_error(
            "Number of samples in y must match number of samples in X",
            array_name="y",
            expectedShape=f"({n_samples},)",
            actualShape=y.shape
        )
    
    # Check if k_folds is valid
    if k_folds < 2:
        raise ValueError("k_folds must be at least 2")
    
    # Ensure k_folds is not larger than the number of samples
    k_folds = min(k_folds, n_samples)
    
    # Initialize random number generator
    rng = np.random.RandomState(random_state)
    
    # Create indices for cross-validation
    indices = np.arange(n_samples)
    if shuffle:
        rng.shuffle(indices)
    
    # Split indices into k folds
    fold_sizes = np.full(k_folds, n_samples // k_folds, dtype=int)
    fold_sizes[:n_samples % k_folds] += 1
    current = 0
    fold_indices = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        fold_indices.append(indices[start:stop])
        current = stop
    
    # Initialize arrays to store metrics
    r2_scores = np.zeros(k_folds)
    mse_scores = np.zeros(k_folds)
    mae_scores = np.zeros(k_folds)
    
    # Perform cross-validation
    for i, test_idx in enumerate(fold_indices):
        # Create train/test split
        train_idx = np.concatenate([fold_indices[j] for j in range(k_folds) if j != i])
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit OLS model
        try:
            # Compute OLS coefficients
            beta = np.linalg.solve(X_train.T @ X_train, X_train.T @ y_train)
            
            # Make predictions
            y_pred = X_test @ beta
            
            # Compute metrics
            # R-squared
            y_test_mean = np.mean(y_test)
            ss_total = np.sum((y_test - y_test_mean) ** 2)
            ss_residual = np.sum((y_test - y_pred) ** 2)
            r2 = 1 - (ss_residual / ss_total)
            r2_scores[i] = r2
            
            # Mean squared error
            mse = np.mean((y_test - y_pred) ** 2)
            mse_scores[i] = mse
            
            # Mean absolute error
            mae = np.mean(np.abs(y_test - y_pred))
            mae_scores[i] = mae
        
        except np.linalg.LinAlgError:
            # If the matrix is singular, set metrics to NaN
            r2_scores[i] = np.nan
            mse_scores[i] = np.nan
            mae_scores[i] = np.nan
    
    # Compute mean and standard deviation of metrics
    r2_mean = np.nanmean(r2_scores)
    r2_std = np.nanstd(r2_scores)
    mse_mean = np.nanmean(mse_scores)
    mse_std = np.nanstd(mse_scores)
    mae_mean = np.nanmean(mae_scores)
    mae_std = np.nanstd(mae_scores)
    
    # Return results as dictionary
    return {
        'r2_mean': r2_mean,
        'r2_std': r2_std,
        'mse_mean': mse_mean,
        'mse_std': mse_std,
        'mae_mean': mae_mean,
        'mae_std': mae_std
    }


def compute_press(X: np.ndarray, y: np.ndarray) -> float:
    """
    Compute PRESS statistic (prediction sum of squares).
    
    This function computes the PRESS statistic, which is the sum of squared
    prediction errors when each observation is left out one at a time.
    
    Args:
        X: Design matrix (n_samples x n_features)
        y: Target variable (n_samples)
        
    Returns:
        PRESS statistic
        
    Raises:
        DimensionError: If X is not a 2D array or y is not a 1D array
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.cross_section.utils import compute_press
        >>> X = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
        >>> y = np.array([2, 3, 4, 5])
        >>> press = compute_press(X, y)
    """
    # Convert to numpy arrays if needed
    X = np.asarray(X)
    y = np.asarray(y)
    
    # Check if X is a 2D array
    if X.ndim != 2:
        raise_dimension_error(
            "X must be a 2D array",
            array_name="X",
            expectedShape="(n_samples, n_features)",
            actualShape=X.shape
        )
    
    # Check if y is a 1D array
    if y.ndim != 1:
        raise_dimension_error(
            "y must be a 1D array",
            array_name="y",
            expectedShape="(n_samples,)",
            actualShape=y.shape
        )
    
    # Check if dimensions match
    n_samples, n_features = X.shape
    if y.shape[0] != n_samples:
        raise_dimension_error(
            "Number of samples in y must match number of samples in X",
            array_name="y",
            expectedShape=f"({n_samples},)",
            actualShape=y.shape
        )
    
    # Compute OLS coefficients for the full model
    try:
        beta_full = np.linalg.solve(X.T @ X, X.T @ y)
        
        # Compute hat matrix diagonal (leverage values)
        # H = X(X'X)^(-1)X'
        # We only need the diagonal elements: h_ii = X_i(X'X)^(-1)X_i'
        XtX_inv = np.linalg.inv(X.T @ X)
        leverage = np.zeros(n_samples)
        for i in range(n_samples):
            leverage[i] = X[i] @ XtX_inv @ X[i].T
        
        # Compute residuals for the full model
        residuals = y - X @ beta_full
        
        # Compute PRESS residuals
        press_residuals = residuals / (1 - leverage)
        
        # Compute PRESS statistic
        press = np.sum(press_residuals ** 2)
        
        return press
    
    except np.linalg.LinAlgError:
        # If the matrix is singular, return NaN
        warnings.warn("Singular matrix encountered in PRESS computation. "
                     "Results may be unreliable.")
        return np.nan


def compute_leverage(X: np.ndarray) -> np.ndarray:
    """
    Compute leverage (hat) values for observations.
    
    This function computes the leverage (hat) values for each observation,
    which are the diagonal elements of the hat matrix H = X(X'X)^(-1)X'.
    
    Args:
        X: Design matrix (n_samples x n_features)
        
    Returns:
        Array of leverage values for each observation
        
    Raises:
        DimensionError: If X is not a 2D array
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.cross_section.utils import compute_leverage
        >>> X = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
        >>> leverage = compute_leverage(X)
    """
    # Convert to numpy arrays if needed
    X = np.asarray(X)
    
    # Check if X is a 2D array
    if X.ndim != 2:
        raise_dimension_error(
            "X must be a 2D array",
            array_name="X",
            expectedShape="(n_samples, n_features)",
            actualShape=X.shape
        )
    
    # Compute (X'X)^(-1)
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        
        # Compute leverage values
        n_samples = X.shape[0]
        leverage = np.zeros(n_samples)
        for i in range(n_samples):
            leverage[i] = X[i] @ XtX_inv @ X[i].T
        
        return leverage
    
    except np.linalg.LinAlgError:
        # If the matrix is singular, return NaN
        warnings.warn("Singular matrix encountered in leverage computation. "
                     "Results may be unreliable.")
        return np.full(X.shape[0], np.nan)


def compute_influence(X: np.ndarray, y: np.ndarray, 
                     residuals: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute influence measures for observations.
    
    This function computes various influence measures for each observation,
    including Cook's distance and DFFITS.
    
    Args:
        X: Design matrix (n_samples x n_features)
        y: Target variable (n_samples)
        residuals: Residuals from regression (n_samples)
        
    Returns:
        Dictionary containing influence measures:
            - 'cooks_distance': Cook's distance for each observation
            - 'dffits': DFFITS for each observation
            - 'leverage': Leverage (hat) values for each observation
            - 'studentized_residuals': Studentized residuals
        
    Raises:
        DimensionError: If X is not a 2D array or y/residuals are not a 1D array
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.cross_section.utils import compute_influence
        >>> X = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
        >>> y = np.array([2, 3, 4, 5])
        >>> residuals = np.array([0.1, -0.1, 0.1, -0.1])
        >>> influence = compute_influence(X, y, residuals)
    """
    # Convert to numpy arrays if needed
    X = np.asarray(X)
    y = np.asarray(y)
    residuals = np.asarray(residuals)
    
    # Check if X is a 2D array
    if X.ndim != 2:
        raise_dimension_error(
            "X must be a 2D array",
            array_name="X",
            expectedShape="(n_samples, n_features)",
            actualShape=X.shape
        )
    
    # Check if y is a 1D array
    if y.ndim != 1:
        raise_dimension_error(
            "y must be a 1D array",
            array_name="y",
            expectedShape="(n_samples,)",
            actualShape=y.shape
        )
    
    # Check if residuals is a 1D array
    if residuals.ndim != 1:
        raise_dimension_error(
            "residuals must be a 1D array",
            array_name="residuals",
            expectedShape="(n_samples,)",
            actualShape=residuals.shape
        )
    
    # Check if dimensions match
    n_samples, n_features = X.shape
    if y.shape[0] != n_samples:
        raise_dimension_error(
            "Number of samples in y must match number of samples in X",
            array_name="y",
            expectedShape=f"({n_samples},)",
            actualShape=y.shape
        )
    
    if residuals.shape[0] != n_samples:
        raise_dimension_error(
            "Number of residuals must match number of samples in X",
            array_name="residuals",
            expectedShape=f"({n_samples},)",
            actualShape=residuals.shape
        )
    
    try:
        # Compute leverage values
        leverage = compute_leverage(X)
        
        # Compute MSE
        mse = np.sum(residuals ** 2) / (n_samples - n_features)
        
        # Compute studentized residuals
        studentized_residuals = residuals / np.sqrt(mse * (1 - leverage))
        
        # Compute Cook's distance
        cooks_distance = (studentized_residuals ** 2 / n_features) * (leverage / (1 - leverage))
        
        # Compute DFFITS
        dffits = studentized_residuals * np.sqrt(leverage / (1 - leverage))
        
        return {
            'cooks_distance': cooks_distance,
            'dffits': dffits,
            'leverage': leverage,
            'studentized_residuals': studentized_residuals
        }
    
    except np.linalg.LinAlgError:
        # If the matrix is singular, return NaN
        warnings.warn("Singular matrix encountered in influence computation. "
                     "Results may be unreliable.")
        return {
            'cooks_distance': np.full(n_samples, np.nan),
            'dffits': np.full(n_samples, np.nan),
            'leverage': np.full(n_samples, np.nan),
            'studentized_residuals': np.full(n_samples, np.nan)
        }


def compute_partial_corr(X: np.ndarray) -> np.ndarray:
    """
    Compute partial correlation matrix.
    
    This function computes the partial correlation matrix for a set of variables,
    which measures the correlation between pairs of variables while controlling
    for the effects of all other variables.
    
    Args:
        X: Data matrix (n_samples x n_features)
        
    Returns:
        Partial correlation matrix (n_features x n_features)
        
    Raises:
        DimensionError: If X is not a 2D array
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.cross_section.utils import compute_partial_corr
        >>> X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        >>> partial_corr = compute_partial_corr(X)
    """
    # Convert to numpy array if needed
    X = np.asarray(X)
    
    # Check if X is a 2D array
    if X.ndim != 2:
        raise_dimension_error(
            "X must be a 2D array",
            array_name="X",
            expectedShape="(n_samples, n_features)",
            actualShape=X.shape
        )
    
    # Compute correlation matrix
    corr = np.corrcoef(X, rowvar=False)
    
    # Compute partial correlation matrix
    try:
        # Compute precision matrix (inverse of correlation matrix)
        precision = np.linalg.inv(corr)
        
        # Compute partial correlation matrix
        n_features = X.shape[1]
        partial_corr = np.zeros((n_features, n_features))
        for i in range(n_features):
            for j in range(n_features):
                if i == j:
                    partial_corr[i, j] = 1.0
                else:
                    partial_corr[i, j] = -precision[i, j] / np.sqrt(precision[i, i] * precision[j, j])
        
        return partial_corr
    
    except np.linalg.LinAlgError:
        # If the matrix is singular, return NaN
        warnings.warn("Singular matrix encountered in partial correlation computation. "
                     "Results may be unreliable.")
        return np.full((X.shape[1], X.shape[1]), np.nan)


async def compute_eigenvalue_bootstrap(X: np.ndarray, 
                                     n_bootstrap: int = 1000,
                                     confidence_level: float = 0.95,
                                     random_state: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Bootstrap confidence intervals for eigenvalues in PCA.
    
    This function computes bootstrap confidence intervals for eigenvalues
    in Principal Component Analysis (PCA).
    
    Args:
        X: Data matrix (n_samples x n_features)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals (between 0 and 1)
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing:
            - 'eigenvalues': Original eigenvalues
            - 'lower_bound': Lower bounds of confidence intervals
            - 'upper_bound': Upper bounds of confidence intervals
        
    Raises:
        DimensionError: If X is not a 2D array
        ValueError: If confidence_level is not between 0 and 1
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.cross_section.utils import compute_eigenvalue_bootstrap
        >>> X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        >>> result = await compute_eigenvalue_bootstrap(X, n_bootstrap=100)
    """
    # Convert to numpy array if needed
    X = np.asarray(X)
    
    # Check if X is a 2D array
    if X.ndim != 2:
        raise_dimension_error(
            "X must be a 2D array",
            array_name="X",
            expectedShape="(n_samples, n_features)",
            actualShape=X.shape
        )
    
    # Check if confidence_level is valid
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    
    # Initialize random number generator
    rng = np.random.RandomState(random_state)
    
    # Get dimensions
    n_samples, n_features = X.shape
    
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov = X_centered.T @ X_centered / (n_samples - 1)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, _ = np.linalg.eigh(cov)
    
    # Sort in descending order
    eigenvalues = eigenvalues[::-1]
    
    # Initialize array to store bootstrap eigenvalues
    bootstrap_eigenvalues = np.zeros((n_bootstrap, n_features))
    
    # Perform bootstrap
    for i in range(n_bootstrap):
        # Generate bootstrap sample
        indices = rng.randint(0, n_samples, size=n_samples)
        X_boot = X_centered[indices]
        
        # Compute covariance matrix
        cov_boot = X_boot.T @ X_boot / (n_samples - 1)
        
        # Compute eigenvalues
        eigenvalues_boot, _ = np.linalg.eigh(cov_boot)
        
        # Sort in descending order
        eigenvalues_boot = eigenvalues_boot[::-1]
        
        # Store bootstrap eigenvalues
        bootstrap_eigenvalues[i] = eigenvalues_boot
        
        # Yield control to allow other tasks to run
        await asyncio.sleep(0)
    
    # Compute confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_eigenvalues, lower_percentile, axis=0)
    upper_bound = np.percentile(bootstrap_eigenvalues, upper_percentile, axis=0)
    
    return {
        'eigenvalues': eigenvalues,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }


async def compute_loading_bootstrap(X: np.ndarray, 
                                  n_components: int,
                                  n_bootstrap: int = 1000,
                                  confidence_level: float = 0.95,
                                  random_state: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Bootstrap confidence intervals for PCA loadings.
    
    This function computes bootstrap confidence intervals for loadings
    (eigenvectors) in Principal Component Analysis (PCA).
    
    Args:
        X: Data matrix (n_samples x n_features)
        n_components: Number of components to compute intervals for
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals (between 0 and 1)
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing:
            - 'loadings': Original loadings (n_features x n_components)
            - 'lower_bound': Lower bounds of confidence intervals
            - 'upper_bound': Upper bounds of confidence intervals
        
    Raises:
        DimensionError: If X is not a 2D array
        ValueError: If confidence_level is not between 0 and 1
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.cross_section.utils import compute_loading_bootstrap
        >>> X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        >>> result = await compute_loading_bootstrap(X, n_components=2, n_bootstrap=100)
    """
    # Convert to numpy array if needed
    X = np.asarray(X)
    
    # Check if X is a 2D array
    if X.ndim != 2:
        raise_dimension_error(
            "X must be a 2D array",
            array_name="X",
            expectedShape="(n_samples, n_features)",
            actualShape=X.shape
        )
    
    # Check if confidence_level is valid
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    
    # Check if n_components is valid
    n_samples, n_features = X.shape
    if not 1 <= n_components <= n_features:
        raise ValueError(f"n_components must be between 1 and {n_features}")
    
    # Initialize random number generator
    rng = np.random.RandomState(random_state)
    
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov = X_centered.T @ X_centered / (n_samples - 1)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select the first n_components eigenvectors
    loadings = eigenvectors[:, :n_components]
    
    # Initialize array to store bootstrap loadings
    bootstrap_loadings = np.zeros((n_bootstrap, n_features, n_components))
    
    # Perform bootstrap
    for i in range(n_bootstrap):
        # Generate bootstrap sample
        indices = rng.randint(0, n_samples, size=n_samples)
        X_boot = X_centered[indices]
        
        # Compute covariance matrix
        cov_boot = X_boot.T @ X_boot / (n_samples - 1)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues_boot, eigenvectors_boot = np.linalg.eigh(cov_boot)
        
        # Sort in descending order
        idx_boot = np.argsort(eigenvalues_boot)[::-1]
        eigenvectors_boot = eigenvectors_boot[:, idx_boot]
        
        # Select the first n_components eigenvectors
        loadings_boot = eigenvectors_boot[:, :n_components]
        
        # Ensure the sign of the loadings is consistent with the original
        for j in range(n_components):
            if np.sum(loadings[:, j] * loadings_boot[:, j]) < 0:
                loadings_boot[:, j] = -loadings_boot[:, j]
        
        # Store bootstrap loadings
        bootstrap_loadings[i] = loadings_boot
        
        # Yield control to allow other tasks to run
        await asyncio.sleep(0)
    
    # Compute confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_loadings, lower_percentile, axis=0)
    upper_bound = np.percentile(bootstrap_loadings, upper_percentile, axis=0)
    
    return {
        'loadings': loadings,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }


# Import asyncio for asynchronous functions
import asyncio

# Register Numba-accelerated functions if available
def _register_numba_functions() -> None:
    """
    Register Numba JIT-compiled functions for cross-sectional analysis utilities.
    
    This function is called during module initialization to register
    performance-critical functions for JIT compilation if Numba is available.
    """
    if HAS_NUMBA:
        logger.debug("Cross-sectional analysis Numba JIT functions registered")
    else:
        logger.info("Numba not available. Cross-sectional analysis utilities will use pure NumPy implementations.")


# Initialize the module
_register_numba_functions()

# mfe/models/cross_section/utils.py
"""
Cross-Sectional Analysis Utilities Module

This module provides specialized utility functions for cross-sectional analysis,
including robust error estimation, data transformation, diagnostic tests, and
matrix manipulation helpers. These functions support both OLS regression and
Principal Component Analysis implementations in the MFE Toolbox.

The module implements optimized versions of cross-sectional utilities using NumPy's
efficient array operations and Numba's JIT compilation for performance-critical
functions. All functions include comprehensive type hints and input validation
to ensure reliability and proper error handling.

Functions:
    add_constant: Add a constant term to a design matrix
    remove_constant: Remove a constant term from a design matrix
    standardize_data: Standardize data by removing mean and scaling by std dev
    compute_vif: Compute variance inflation factors for multicollinearity detection
    compute_condition_number: Compute condition number of a design matrix
    compute_white_se: Compute White's heteroskedasticity-robust standard errors
    compute_hc_se: Compute HC0-HC3 heteroskedasticity-consistent standard errors
    compute_jarque_bera: Compute Jarque-Bera test for normality
    compute_breusch_pagan: Compute Breusch-Pagan test for heteroskedasticity
    compute_durbin_watson: Compute Durbin-Watson statistic for autocorrelation
    cross_validate: Perform k-fold cross-validation for regression models
    compute_press: Compute PRESS statistic (prediction sum of squares)
    compute_leverage: Compute leverage (hat) values for observations
    compute_influence: Compute influence measures (Cook's distance, DFFITS)
    compute_partial_corr: Compute partial correlation matrix
    compute_eigenvalue_bootstrap: Bootstrap confidence intervals for eigenvalues
    compute_loading_bootstrap: Bootstrap confidence intervals for PCA loadings
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Literal, Optional, Tuple, Union, 
    cast, overload, TypeVar, Protocol
)

import numpy as np
import pandas as pd
from scipy import stats, linalg
from scipy.optimize import minimize

from mfe.core.exceptions import (
    DimensionError, NumericError, ParameterError, DataError,
    raise_dimension_error, raise_numeric_error, raise_parameter_error, 
    raise_data_error, warn_numeric
)
from mfe.core.types import Matrix, Vector
from mfe.utils.matrix_ops import ensure_symmetric, is_positive_definite, nearest_positive_definite

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    HAS_NUMBA = False

# Set up module-level logger
logger = logging.getLogger("mfe.models.cross_section.utils")


def add_constant(X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
    """
    Add a constant term (column of ones) to a design matrix.
    
    This function adds a column of ones to the left side of a design matrix
    for regression analysis. If the input is a pandas DataFrame, the constant
    column will be named 'const'.
    
    Args:
        X: Design matrix (n_samples x n_features)
        
    Returns:
        Design matrix with constant term added (n_samples x (n_features+1))
        
    Raises:
        DimensionError: If X is not a 2D array or DataFrame
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.cross_section.utils import add_constant
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> add_constant(X)
        array([[1., 1., 2.],
               [1., 3., 4.],
               [1., 5., 6.]])
    """
    # Check if X is a pandas DataFrame
    if isinstance(X, pd.DataFrame):
        if 'const' in X.columns:
            # Check if the constant column is already all ones
            if X['const'].equals(pd.Series(np.ones(len(X)), index=X.index)):
                return X
            else:
                warnings.warn("DataFrame already contains a 'const' column that is not all ones. "
                             "Renaming existing column to 'const_0' and adding new constant.")
                X = X.rename(columns={'const': 'const_0'})
        
        # Add constant column
        result = pd.DataFrame({'const': np.ones(len(X))}, index=X.index)
        result = pd.concat([result, X], axis=1)
        return result
    
    # Handle numpy array
    X_array = np.asarray(X)
    
    # Check if X is a 2D array
    if X_array.ndim != 2:
        raise_dimension_error(
            "Input must be a 2D array",
            array_name="X",
            expected_shape="(n_samples, n_features)",
            actual_shape=X_array.shape
        )
    
    # Check if the first column is already all ones
    n_samples = X_array.shape[0]
    if X_array.shape[1] > 0 and np.allclose(X_array[:, 0], np.ones(n_samples)):
        return X_array
    
    # Add constant column
    X_with_const = np.column_stack((np.ones(n_samples), X_array))
    
    return X_with_const


def remove_constant(X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
    """
    Remove a constant term (column of ones) from a design matrix.
    
    This function removes a column of ones from the left side of a design matrix.
    If the input is a pandas DataFrame, it will remove a column named 'const'.
    
    Args:
        X: Design matrix with constant term (n_samples x n_features)
        
    Returns:
        Design matrix without constant term (n_samples x (n_features-1))
        
    Raises:
        DimensionError: If X is not a 2D array or DataFrame
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.cross_section.utils import remove_constant
        >>> X = np.array([[1, 1, 2], [1, 3, 4], [1, 5, 6]])
        >>> remove_constant(X)
        array([[1, 2],
               [3, 4],
               [5, 6]])
    """
    # Check if X is a pandas DataFrame
    if isinstance(X, pd.DataFrame):
        if 'const' in X.columns:
            return X.drop('const', axis=1)
        else:
            return X
    
    # Handle numpy array
    X_array = np.asarray(X)
    
    # Check if X is a 2D array
    if X_array.ndim != 2:
        raise_dimension_error(
            "Input must be a 2D array",
            array_name="X",
            expected_shape="(n_samples, n_features)",
            actual_shape=X_array.shape
        )
    
    # Check if the first column is all ones
    n_samples = X_array.shape[0]
    if X_array.shape[1] > 0 and np.allclose(X_array[:, 0], np.ones(n_samples)):
        return X_array[:, 1:]
    else:
        return X_array


def standardize_data(X: Union[np.ndarray, pd.DataFrame], 
                    center: bool = True, 
                    scale: bool = True,
                    skip_constant: bool = True) -> Tuple[Union[np.ndarray, pd.DataFrame], np.ndarray, np.ndarray]:
    """
    Standardize data by removing mean and scaling by standard deviation.
    
    This function standardizes a design matrix by centering (subtracting the mean)
    and scaling (dividing by the standard deviation). It can optionally skip
    standardizing a constant column.
    
    Args:
        X: Design matrix (n_samples x n_features)
        center: Whether to center the data (subtract mean)
        scale: Whether to scale the data (divide by std)
        skip_constant: Whether to skip standardizing a constant column
        
    Returns:
        Tuple containing:
            - Standardized design matrix
            - Mean of each column (or zeros if center=False)
            - Standard deviation of each column (or ones if scale=False)
        
    Raises:
        DimensionError: If X is not a 2D array or DataFrame
        NumericError: If any column has zero standard deviation
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.cross_section.utils import standardize_data
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> X_std, mean, std = standardize_data(X)
        >>> X_std
        array([[-1.22474487, -1.22474487],
               [ 0.        ,  0.        ],
               [ 1.22474487,  1.22474487]])
        >>> mean
        array([3., 4.])
        >>> std
        array([1.63299316, 1.63299316])
    """
    # Check if X is a pandas DataFrame
    is_dataframe = isinstance(X, pd.DataFrame)
    
    # Convert to numpy array for processing
    X_array = X.values if is_dataframe else np.asarray(X)
    
    # Check if X is a 2D array
    if X_array.ndim != 2:
        raise_dimension_error(
            "Input must be a 2D array",
            array_name="X",
            expected_shape="(n_samples, n_features)",
            actual_shape=X_array.shape
        )
    
    n_samples, n_features = X_array.shape
    
    # Initialize mean and std arrays
    mean = np.zeros(n_features)
    std = np.ones(n_features)
    
    # Identify constant column if needed
    const_col = None
    if skip_constant and n_features > 0:
        # Check if the first column is all ones
        if np.allclose(X_array[:, 0], np.ones(n_samples)):
            const_col = 0
        # If it's a DataFrame, check for 'const' column
        elif is_dataframe and 'const' in X.columns:
            const_col = list(X.columns).index('const')
    
    # Compute mean if centering
    if center:
        mean = np.mean(X_array, axis=0)
        if const_col is not None:
            mean[const_col] = 0  # Don't center constant column
    
    # Compute std if scaling
    if scale:
        std = np.std(X_array, axis=0, ddof=1)
        if const_col is not None:
            std[const_col] = 1  # Don't scale constant column
        
        # Check for zero standard deviation
        zero_std_cols = np.where(std == 0)[0]
        if len(zero_std_cols) > 0:
            raise_numeric_error(
                f"Columns {zero_std_cols} have zero standard deviation, cannot scale",
                operation="standardize_data",
                values=std,
                error_type="zero_std_dev"
            )
    
    # Standardize the data
    X_std = X_array.copy()
    if center:
        X_std -= mean
    if scale:
        X_std /= std
    
    # Convert back to DataFrame if input was DataFrame
    if is_dataframe:
        X_std = pd.DataFrame(X_std, index=X.index, columns=X.columns)
    
    return X_std, mean, std


@jit(nopython=True, cache=True)
def _compute_vif_numba(X: np.ndarray, skip_constant: bool = True) -> np.ndarray:
    """
    Numba-accelerated implementation of variance inflation factor computation.
    
    Args:
        X: Design matrix (n_samples x n_features)
        skip_constant: Whether to skip computing VIF for a constant column
        
    Returns:
        Array of VIF values for each column
    """
    n_samples, n_features = X.shape
    vif = np.ones(n_features)
    
    start_idx = 1 if skip_constant else 0
    
    for i in range(start_idx, n_features):
        # Create X matrix without the current variable
        X_without_i = np.zeros((n_samples, n_features - 1))
        col_idx = 0
        for j in range(n_features):
            if j != i:
                X_without_i[:, col_idx] = X[:, j]
                col_idx += 1
        
        # Compute X'X and X'y for the regression
        XtX = np.zeros((n_features - 1, n_features - 1))
        Xty = np.zeros(n_features - 1)
        
        for j in range(n_features - 1):
            Xty[j] = np.sum(X_without_i[:, j] * X[:, i])
            for k in range(n_features - 1):
                XtX[j, k] = np.sum(X_without_i[:, j] * X_without_i[:, k])
        
        # Solve for beta: (X'X)^(-1) X'y
        # Note: In a real implementation, we would use a more robust method
        # This is a simplified version for demonstration
        # In practice, we would use np.linalg.solve, but that's not supported in numba
        
        # Compute inverse of X'X using Gaussian elimination
        # This is a simplified implementation and may not be numerically stable
        # for all matrices, but works for demonstration purposes
        n = n_features - 1
        L = np.eye(n)
        U = XtX.copy()
        
        for k in range(n-1):
            for i in range(k+1, n):
                L[i, k] = U[i, k] / U[k, k]
                for j in range(k, n):
                    U[i, j] = U[i, j] - L[i, k] * U[k, j]
        
        # Solve Ly = b
        y = np.zeros(n)
        for i in range(n):
            y[i] = Xty[i]
            for j in range(i):
                y[i] -= L[i, j] * y[j]
        
        # Solve Ux = y
        beta = np.zeros(n)
        for i in range(n-1, -1, -1):
            beta[i] = y[i]
            for j in range(i+1, n):
                beta[i] -= U[i, j] * beta[j]
            beta[i] /= U[i, i]
        
        # Compute R-squared
        y_pred = np.zeros(n_samples)
        for j in range(n_features - 1):
            y_pred += X_without_i[:, j] * beta[j]
        
        y_mean = np.mean(X[:, i])
        ss_total = np.sum((X[:, i] - y_mean) ** 2)
        ss_residual = np.sum((X[:, i] - y_pred) ** 2)
        r_squared = 1 - (ss_residual / ss_total)
        
        # Compute VIF
        if r_squared < 1.0:  # Avoid division by zero
            vif[i] = 1.0 / (1.0 - r_squared)
        else:
            vif[i] = float('inf')
    
    return vif


def compute_vif(X: Union[np.ndarray, pd.DataFrame], 
               skip_constant: bool = True) -> Union[np.ndarray, Dict[str, float]]:
    """
    Compute variance inflation factors for multicollinearity detection.
    
    This function computes the variance inflation factor (VIF) for each variable
    in a design matrix. VIF measures how much the variance of a regression coefficient
    is inflated due to multicollinearity with other predictors.
    
    Args:
        X: Design matrix (n_samples x n_features)
        skip_constant: Whether to skip computing VIF for a constant column
        
    Returns:
        If X is a numpy array: Array of VIF values for each column
        If X is a pandas DataFrame: Dictionary mapping column names to VIF values
        
    Raises:
        DimensionError: If X is not a 2D array or DataFrame
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.cross_section.utils import compute_vif
        >>> X = np.array([[1, 1, 2], [1, 3, 4], [1, 5, 6]])
        >>> compute_vif(X)
        array([1.        , 1.        , 1.        ])
    """
    # Check if X is a pandas DataFrame
    is_dataframe = isinstance(X, pd.DataFrame)
    
    # Convert to numpy array for processing
    X_array = X.values if is_dataframe else np.asarray(X)
    
    # Check if X is a 2D array
    if X_array.ndim != 2:
        raise_dimension_error(
            "Input must be a 2D array",
            array_name="X",
            expected_shape="(n_samples, n_features)",
            actual_shape=X_array.shape
        )
    
    n_samples, n_features = X_array.shape
    
    # Check if we have enough samples
    if n_samples <= n_features:
        warnings.warn("Number of samples must be greater than number of features "
                     "for reliable VIF computation. Results may be unreliable.")
    
    # Use Numba-accelerated implementation if available
    if HAS_NUMBA:
        vif_values = _compute_vif_numba(X_array, skip_constant)
    else:
        # Pure NumPy/SciPy implementation
        vif_values = np.ones(n_features)
        
        start_idx = 1 if skip_constant else 0
        
        for i in range(start_idx, n_features):
            # Create X matrix without the current variable
            mask = np.ones(n_features, dtype=bool)
            mask[i] = False
            X_without_i = X_array[:, mask]
            
            # Regress the current variable on all other variables
            try:
                # Use OLS to compute R-squared
                # (X'X)^(-1) X'y
                XtX = X_without_i.T @ X_without_i
                Xty = X_without_i.T @ X_array[:, i]
                beta = np.linalg.solve(XtX, Xty)
                
                # Compute R-squared
                y_pred = X_without_i @ beta
                y_mean = np.mean(X_array[:, i])
                ss_total = np.sum((X_array[:, i] - y_mean) ** 2)
                ss_residual = np.sum((X_array[:, i] - y_pred) ** 2)
                r_squared = 1 - (ss_residual / ss_total)
                
                # Compute VIF
                if r_squared < 1.0:  # Avoid division by zero
                    vif_values[i] = 1.0 / (1.0 - r_squared)
                else:
                    vif_values[i] = float('inf')
            
            except np.linalg.LinAlgError:
                # If the matrix is singular, set VIF to infinity
                vif_values[i] = float('inf')
    
    # Return as dictionary if input was DataFrame
    if is_dataframe:
        column_names = X.columns
        return {col: vif_values[i] for i, col in enumerate(column_names)}
    else:
        return vif_values


def compute_condition_number(X: Union[np.ndarray, pd.DataFrame]) -> float:
    """
    Compute the condition number of a design matrix.
    
    This function computes the condition number of a design matrix, which is
    the ratio of the largest to smallest singular value. A high condition number
    indicates potential multicollinearity issues.
    
    Args:
        X: Design matrix (n_samples x n_features)
        
    Returns:
        Condition number of the design matrix
        
    Raises:
        DimensionError: If X is not a 2D array or DataFrame
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.cross_section.utils import compute_condition_number
        >>> X = np.array([[1, 1, 2], [1, 3, 4], [1, 5, 6]])
        >>> compute_condition_number(X)
        17.92
    """
    # Convert to numpy array if needed
    X_array = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    
    # Check if X is a 2D array
    if X_array.ndim != 2:
        raise_dimension_error(
            "Input must be a 2D array",
            array_name="X",
            expected_shape="(n_samples, n_features)",
            actual_shape=X_array.shape
        )
    
    # Compute singular values
    s = np.linalg.svd(X_array, compute_uv=False)
    
    # Compute condition number
    condition_number = s[0] / s[-1]
    
    return condition_number


def compute_white_se(X: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """
    Compute White's heteroskedasticity-robust standard errors.
    
    This function computes heteroskedasticity-robust standard errors for OLS
    regression coefficients using White's method.
    
    Args:
        X: Design matrix (n_samples x n_features)
        residuals: Residuals from OLS regression (n_samples)
        
    Returns:
        Array of robust standard errors for each coefficient
        
    Raises:
        DimensionError: If X is not a 2D array or residuals is not a 1D array
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.cross_section.utils import compute_white_se
        >>> X = np.array([[1, 1], [1, 2], [1, 3]])
        >>> residuals = np.array([0.1, -0.2, 0.1])
        >>> compute_white_se(X, residuals)
        array([0.11547005, 0.08164966])
    """
    # Convert to numpy arrays if needed
    X = np.asarray(X)
    residuals = np.asarray(residuals)
    
    # Check if X is a 2D array
    if X.ndim != 2:
        raise_dimension_error(
            "X must be a 2D array",
            array_name="X",
            expected_shape="(n_samples, n_features)",
            actual_shape=X.shape
        )
    
    # Check if residuals is a 1D array
    if residuals.ndim != 1:
        raise_dimension_error(
            "residuals must be a 1D array",
            array_name="residuals",
            expected_shape="(n_samples,)",
            actual_shape=residuals.shape
        )
    
    # Check if dimensions match
    n_samples, n_features = X.shape
    if residuals.shape[0] != n_samples:
        raise_dimension_error(
            "Number of residuals must match number of samples in X",
            array_name="residuals",
            expected_shape=f"({n_samples},)",
            actual_shape=residuals.shape
        )
    
    # Compute (X'X)^(-1)
    XtX_inv = np.linalg.inv(X.T @ X)
    
    # Compute X'diag(e^2)X
    squared_residuals = residuals ** 2
    X_weighted = X * squared_residuals[:, np.newaxis]
    middle_term = X.T @ X_weighted
    
    # Compute the robust covariance matrix
    cov_matrix = XtX_inv @ middle_term @ XtX_inv
    
    # Extract the diagonal elements and take the square root
    robust_se = np.sqrt(np.diag(cov_matrix))
    
    return robust_se


def compute_hc_se(X: np.ndarray, residuals: np.ndarray, 
                 cov_type: Literal['hc0', 'hc1', 'hc2', 'hc3'] = 'hc1') -> np.ndarray:
    """
    Compute heteroskedasticity-consistent standard errors.
    
    This function computes heteroskedasticity-consistent (HC) standard errors
    for OLS regression coefficients using various methods (HC0-HC3).
    
    Args:
        X: Design matrix (n_samples x n_features)
        residuals: Residuals from OLS regression (n_samples)
        cov_type: Type of HC standard errors to compute:
                 'hc0': White's original estimator
                 'hc1': HC0 with small sample correction (n/(n-k))
                 'hc2': HC0 with leverage correction
                 'hc3': HC0 with more aggressive leverage correction
        
    Returns:
        Array of robust standard errors for each coefficient
        
    Raises:
        DimensionError: If X is not a 2D array or residuals is not a 1D array
        ValueError: If cov_type is not one of the supported types
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.cross_section.utils import compute_hc_se
        >>> X = np.array([[1, 1], [1, 2], [1, 3]])
        >>> residuals = np.array([0.1, -0.2, 0.1])
        >>> compute_hc_se(X, residuals, cov_type='hc1')
        array([0.14142136, 0.1])
    """
    # Convert to numpy arrays if needed
    X = np.asarray(X)
    residuals = np.asarray(residuals)
    
    # Check if X is a 2D array
    if X.ndim != 2:
        raise_dimension_error(
            "X must be a 2D array",
            array_name="X",
            expected_shape="(n_samples, n_features)",
            actual_shape=X.shape
        )
    
    # Check if residuals is a 1D array
    if residuals.ndim != 1:
        raise_dimension_error(
            "residuals must be a 1D array",
            array_name="residuals",
            expected_shape="(n_samples,)",
            actual_shape=residuals.shape
        )
    
    # Check if dimensions match
    n_samples, n_features = X.shape
    if residuals.shape[0] != n_samples:
        raise_dimension_error(
            "Number of residuals must match number of samples in X",
            array_name="residuals",
            expected_shape=f"({n_samples},)",
            actual_shape=residuals.shape
        )
    
    # Check if cov_type is valid
    valid_cov_types = ['hc0', 'hc1', 'hc2', 'hc3']
    if cov_type.lower() not in valid_cov_types:
        raise ValueError(f"cov_type must be one of {valid_cov_types}, got {cov_type}")
    
    # Compute (X'X)^(-1)
    XtX_inv = np.linalg.inv(X.T @ X)
    
    # Compute hat matrix diagonal (leverage values)
    # H = X(X'X)^(-1)X'
    # We only need the diagonal elements: h_ii = X_i(X'X)^(-1)X_i'
    leverage = np.zeros(n_samples)
    for i in range(n_samples):
        leverage[i] = X[i] @ XtX_inv @ X[i].T
    
    # Adjust residuals based on the specified covariance type
    if cov_type.lower() == 'hc0':
        # White's original estimator
        adjusted_residuals = residuals ** 2
    elif cov_type.lower() == 'hc1':
        # HC0 with small sample correction
        correction = n_samples / (n_samples - n_features)
        adjusted_residuals = correction * (residuals ** 2)
    elif cov_type.lower() == 'hc2':
        # HC0 with leverage correction
        adjusted_residuals = (residuals ** 2) / (1 - leverage)
    elif cov_type.lower() == 'hc3':
        # HC0 with more aggressive leverage correction
        adjusted_residuals = (residuals ** 2) / ((1 - leverage) ** 2)
    
    # Compute X'diag(adjusted_residuals)X
    X_weighted = X * adjusted_residuals[:, np.newaxis]
    middle_term = X.T @ X_weighted
    
    # Compute the robust covariance matrix
    cov_matrix = XtX_inv @ middle_term @ XtX_inv
    
    # Extract the diagonal elements and take the square root
    robust_se = np.sqrt(np.diag(cov_matrix))
    
    return robust_se


def compute_jarque_bera(residuals: np.ndarray) -> Tuple[float, float]:
    """
    Compute Jarque-Bera test for normality of residuals.
    
    This function computes the Jarque-Bera test statistic and p-value for
    testing whether residuals follow a normal distribution.
    
    Args:
        residuals: Residuals from regression (n_samples)
        
    Returns:
        Tuple containing:
            - Jarque-Bera test statistic
            - p-value
        
    Raises:
        DimensionError: If residuals is not a 1D array
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.cross_section.utils import compute_jarque_bera
        >>> residuals = np.array([0.1, -0.2, 0.1, 0.3, -0.1, 0.2])
        >>> jb_stat, p_value = compute_jarque_bera(residuals)
    """
    # Convert to numpy array if needed
    residuals = np.asarray(residuals)
    
    # Check if residuals is a 1D array
    if residuals.ndim != 1:
        raise_dimension_error(
            "residuals must be a 1D array",
            array_name="residuals",
            expected_shape="(n_samples,)",
            actual_shape=residuals.shape
        )
    
    # Compute sample size
    n = len(residuals)
    
    # Compute skewness and kurtosis
    skewness = stats.skew(residuals)
    kurtosis = stats.kurtosis(residuals, fisher=True)  # Fisher's definition (excess kurtosis)
    
    # Compute Jarque-Bera statistic
    jb_stat = n / 6 * (skewness ** 2 + (kurtosis ** 2) / 4)
    
    # Compute p-value (JB follows a chi-squared distribution with 2 degrees of freedom)
    p_value = 1 - stats.chi2.cdf(jb_stat, 2)
    
    return jb_stat, p_value


def compute_breusch_pagan(residuals: np.ndarray, X: np.ndarray) -> Tuple[float, float]:
    """
    Compute Breusch-Pagan test for heteroskedasticity.
    
    This function computes the Breusch-Pagan test statistic and p-value for
    testing whether residuals exhibit heteroskedasticity.
    
    Args:
        residuals: Residuals from regression (n_samples)
        X: Design matrix (n_samples x n_features)
        
    Returns:
        Tuple containing:
            - Breusch-Pagan test statistic
            - p-value
        
    Raises:
        DimensionError: If residuals is not a 1D array or X is not a 2D array
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.cross_section.utils import compute_breusch_pagan
        >>> X = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6]])
        >>> residuals = np.array([0.1, -0.2, 0.1, 0.3, -0.1, 0.2])
        >>> bp_stat, p_value = compute_breusch_pagan(residuals, X)
    """
    # Convert to numpy arrays if needed
    residuals = np.asarray(residuals)
    X = np.asarray(X)
    
    # Check if residuals is a 1D array
    if residuals.ndim != 1:
        raise_dimension_error(
            "residuals must be a 1D array",
            array_name="residuals",
            expected_shape="(n_samples,)",
            actual_shape=residuals.shape
        )
    
    # Check if X is a 2D array
    if X.ndim != 2:
        raise_dimension_error(
            "X must be a 2D array",
            array_name="X",
            expected_shape="(n_samples, n_features)",
            actual_shape=X.shape
        )
    
    # Check if dimensions match
    n_samples, n_features = X.shape
    if residuals.shape[0] != n_samples:
        raise_dimension_error(
            "Number of residuals must match number of samples in X",
            array_name="residuals",
            expected_shape=f"({n_samples},)",
            actual_shape=residuals.shape
        )
    
    # Compute squared residuals
    residuals_squared = residuals ** 2
    
    # Compute mean of squared residuals
    residuals_squared_mean = np.mean(residuals_squared)
    
    # Normalize squared residuals
    u = residuals_squared / residuals_squared_mean - 1
    
    # Regress u on X
    try:
        # Compute OLS coefficients
        beta = np.linalg.solve(X.T @ X, X.T @ u)
        
        # Compute fitted values
        u_hat = X @ beta
        
        # Compute explained sum of squares
        ess = np.sum(u_hat ** 2)
        
        # Compute Breusch-Pagan statistic
        bp_stat = n_samples * ess / 2
        
        # Compute p-value (BP follows a chi-squared distribution with k-1 degrees of freedom)
        p_value = 1 - stats.chi2.cdf(bp_stat, n_features - 1)
        
        return bp_stat, p_value
    
    except np.linalg.LinAlgError:
        # If the matrix is singular, return NaN
        warnings.warn("Singular matrix encountered in Breusch-Pagan test. "
                     "Results may be unreliable.")
        return np.nan, np.nan


def compute_durbin_watson(residuals: np.ndarray) -> float:
    """
    Compute Durbin-Watson statistic for autocorrelation.
    
    This function computes the Durbin-Watson statistic for testing whether
    residuals exhibit first-order autocorrelation.
    
    Args:
        residuals: Residuals from regression (n_samples)
        
    Returns:
        Durbin-Watson statistic
        
    Raises:
        DimensionError: If residuals is not a 1D array
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.cross_section.utils import compute_durbin_watson
        >>> residuals = np.array([0.1, -0.2, 0.1, 0.3, -0.1, 0.2])
        >>> dw_stat = compute_durbin_watson(residuals)
    """
    # Convert to numpy arrays if needed
    residuals = np.asarray(residuals)
    
    # Check if residuals is a 1D array
    if residuals.ndim != 1:
        raise_dimension_error(
            "residuals must be a 1D array",
            array_name="residuals",
            expectedShape="(n_samples,)",
            actualShape=residuals.shape
        )
    
    # Compute Durbin-Watson statistic
    n = len(residuals)
    diff = np.diff(residuals)
    dw_stat = np.sum(diff ** 2) / np.sum(residuals ** 2)
    
    return dw_stat


def cross_validate(X: np.ndarray, y: np.ndarray, 
                  k_folds: int = 5, 
                  shuffle: bool = True,
                  random_state: Optional[int] = None) -> Dict[str, float]:
    """
    Perform k-fold cross-validation for regression models.
    
    This function performs k-fold cross-validation for OLS regression,
    computing various performance metrics.
    
    Args:
        X: Design matrix (n_samples x n_features)
        y: Target variable (n_samples)
        k_folds: Number of folds for cross-validation
        shuffle: Whether to shuffle the data before splitting
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing cross-validation metrics:
            - 'r2_mean': Mean R-squared across folds
            - 'r2_std': Standard deviation of R-squared across folds
            - 'mse_mean': Mean mean squared error across folds
            - 'mse_std': Standard deviation of mean squared error across folds
            - 'mae_mean': Mean mean absolute error across folds
            - 'mae_std': Standard deviation of mean absolute error across folds
        
    Raises:
        DimensionError: If X is not a 2D array or y is not a 1D array
        ValueError: If k_folds is less than 2
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.cross_section.utils import cross_validate
        >>> X = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6]])
        >>> y = np.array([2, 3, 4, 5, 6, 7])
        >>> cv_results = cross_validate(X, y, k_folds=3)
    """
    # Convert to numpy arrays if needed
    X = np.asarray(X)
    y = np.asarray(y)
    
    # Check if X is a 2D array
    if X.ndim != 2:
        raise_dimension_error(
            "X must be a 2D array",
            array_name="X",
            expectedShape="(n_samples, n_features)",
            actualShape=X.shape
        )
    
    # Check if y is a 1D array
    if y.ndim != 1:
        raise_dimension_error(
            "y must be a 1D array",
            array_name="y",
            expectedShape="(n_samples,)",
            actualShape=y.shape
        )
    
    # Check if dimensions match
    n_samples, n_features = X.shape
    if y.shape[0] != n_samples:
        raise_dimension_error(
            "Number of samples in y must match number of samples in X",
            array_name="y",
            expectedShape=f"({n_samples},)",
            actualShape=y.shape
        )
    
    # Check if k_folds is valid
    if k_folds < 2:
        raise ValueError("k_folds must be at least 2")
    
    # Ensure k_folds is not larger than the number of samples
    k_folds = min(k_folds, n_samples)
    
    # Initialize random number generator
    rng = np.random.RandomState(random_state)
    
    # Create indices for cross-validation
    indices = np.arange(n_samples)
    if shuffle:
        rng.shuffle(indices)
    
    # Split indices into k folds
    fold_sizes = np.full(k_folds, n_samples // k_folds, dtype=int)
    fold_sizes[:n_samples % k_folds] += 1
    current = 0
    fold_indices = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        fold_indices.append(indices[start:stop])
        current = stop
    
    # Initialize arrays to store metrics
    r2_scores = np.zeros(k_folds)
    mse_scores = np.zeros(k_folds)
    mae_scores = np.zeros(k_folds)
    
    # Perform cross-validation
    for i, test_idx in enumerate(fold_indices):
        # Create train/test split
        train_idx = np.concatenate([fold_indices[j] for j in range(k_folds) if j != i])
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit OLS model
        try:
            # Compute OLS coefficients
            beta = np.linalg.solve(X_train.T @ X_train, X_train.T @ y_train)
            
            # Make predictions
            y_pred = X_test @ beta
            
            # Compute metrics
            # R-squared
            y_test_mean = np.mean(y_test)
            ss_total = np.sum((y_test - y_test_mean) ** 2)
            ss_residual = np.sum((y_test - y_pred) ** 2)
            r2 = 1 - (ss_residual / ss_total)
            r2_scores[i] = r2
            
            # Mean squared error
            mse = np.mean((y_test - y_pred) ** 2)
            mse_scores[i] = mse
            
            # Mean absolute error
            mae = np.mean(np.abs(y_test - y_pred))
            mae_scores[i] = mae
        
        except np.linalg.LinAlgError:
            # If the matrix is singular, set metrics to NaN
            r2_scores[i] = np.nan
            mse_scores[i] = np.nan
            mae_scores[i] = np.nan
    
    # Compute mean and standard deviation of metrics
    r2_mean = np.nanmean(r2_scores)
    r2_std = np.nanstd(r2_scores)
    mse_mean = np.nanmean(mse_scores)
    mse_std = np.nanstd(mse_scores)
    mae_mean = np.nanmean(mae_scores)
    mae_std = np.nanstd(mae_scores)
    
    # Return results as dictionary
    return {
        'r2_mean': r2_mean,
        'r2_std': r2_std,
        'mse_mean': mse_mean,
        'mse_std': mse_std,
        'mae_mean': mae_mean,
        'mae_std': mae_std
    }


def compute_press(X: np.ndarray, y: np.ndarray) -> float:
    """
    Compute PRESS statistic (prediction sum of squares).
    
    This function computes the PRESS statistic, which is the sum of squared
    prediction errors when each observation is left out one at a time.
    
    Args:
        X: Design matrix (n_samples x n_features)
        y: Target variable (n_samples)
        
    Returns:
        PRESS statistic
        
    Raises:
        DimensionError: If X is not a 2D array or y is not a 1D array
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.cross_section.utils import compute_press
        >>> X = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
        >>> y = np.array([2, 3, 4, 5])
        >>> press = compute_press(X, y)
    """
    # Convert to numpy arrays if needed
    X = np.asarray(X)
    y = np.asarray(y)
    
    # Check if X is a 2D array
    if X.ndim != 2:
        raise_dimension_error(
            "X must be a 2D array",
            array_name="X",
            expectedShape="(n_samples, n_features)",
            actualShape=X.shape
        )
    
    # Check if y is a 1D array
    if y.ndim != 1:
        raise_dimension_error(
            "y must be a 1D array",
            array_name="y",
            expectedShape="(n_samples,)",
            actualShape=y.shape
        )
    
    # Check if dimensions match
    n_samples, n_features = X.shape
    if y.shape[0] != n_samples:
        raise_dimension_error(
            "Number of samples in y must match number of samples in X",
            array_name="y",
            expectedShape=f"({n_samples},)",
            actualShape=y.shape
        )
    
    # Compute OLS coefficients for the full model
    try:
        beta_full = np.linalg.solve(X.T @ X, X.T @ y)
        
        # Compute hat matrix diagonal (leverage values)
        # H = X(X'X)^(-1)X'
        # We only need the diagonal elements: h_ii = X_i(X'X)^(-1)X_i'
        XtX_inv = np.linalg.inv(X.T @ X)
        leverage = np.zeros(n_samples)
        for i in range(n_samples):
            leverage[i] = X[i] @ XtX_inv @ X[i].T
        
        # Compute residuals for the full model
        residuals = y - X @ beta_full
        
        # Compute PRESS residuals
        press_residuals = residuals / (1 - leverage)
        
        # Compute PRESS statistic
        press = np.sum(press_residuals ** 2)
        
        return press
    
    except np.linalg.LinAlgError:
        # If the matrix is singular, return NaN
        warnings.warn("Singular matrix encountered in PRESS computation. "
                     "Results may be unreliable.")
        return np.nan


def compute_leverage(X: np.ndarray) -> np.ndarray:
    """
    Compute leverage (hat) values for observations.
    
    This function computes the leverage (hat) values for each observation,
    which are the diagonal elements of the hat matrix H = X(X'X)^(-1)X'.
    
    Args:
        X: Design matrix (n_samples x n_features)
        
    Returns:
        Array of leverage values for each observation
        
    Raises:
        DimensionError: If X is not a 2D array
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.cross_section.utils import compute_leverage
        >>> X = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
        >>> leverage = compute_leverage(X)
    """
    # Convert to numpy arrays if needed
    X = np.asarray(X)
    
    # Check if X is a 2D array
    if X.ndim != 2:
        raise_dimension_error(
            "X must be a 2D array",
            array_name="X",
            expectedShape="(n_samples, n_features)",
            actualShape=X.shape
        )
    
    # Compute (X'X)^(-1)
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        
        # Compute leverage values
        n_samples = X.shape[0]
        leverage = np.zeros(n_samples)
        for i in range(n_samples):
            leverage[i] = X[i] @ XtX_inv @ X[i].T
        
        return leverage
    
    except np.linalg.LinAlgError:
        # If the matrix is singular, return NaN
        warnings.warn("Singular matrix encountered in leverage computation. "
                     "Results may be unreliable.")
        return np.full(X.shape[0], np.nan)


def compute_influence(X: np.ndarray, y: np.ndarray, 
                     residuals: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute influence measures for observations.
    
    This function computes various influence measures for each observation,
    including Cook's distance and DFFITS.
    
    Args:
        X: Design matrix (n_samples x n_features)
        y: Target variable (n_samples)
        residuals: Residuals from regression (n_samples)
        
    Returns:
        Dictionary containing influence measures:
            - 'cooks_distance': Cook's distance for each observation
            - 'dffits': DFFITS for each observation
            - 'leverage': Leverage (hat) values for each observation
            - 'studentized_residuals': Studentized residuals
        
    Raises:
        DimensionError: If X is not a 2D array or y/residuals are not a 1D array
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.cross_section.utils import compute_influence
        >>> X = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
            >>> y = np.array([2, 3, 4, 5])
            >>> residuals = np.array([0.1, -0.1, 0.1, -0.1])
            >>> influence = compute_influence(X, y, residuals)
    """
    # Convert to numpy arrays if needed
    X = np.asarray(X)
    y = np.asarray(y)
    residuals = np.asarray(residuals)
    
    # Check if X is a 2D array
    if X.ndim != 2:
        raise_dimension_error(
            "X must be a 2D array",
            array_name="X",
            expectedShape="(n_samples, n_features)",
            actualShape=X.shape
        )
    
    # Check if y is a 1D array
    if y.ndim != 1:
        raise_dimension_error(
            "y must be a 1D array",
            array_name="y",
            expectedShape="(n_samples,)",
            actualShape=y.shape
        )
    
    # Check if residuals is a 1D array
    if residuals.ndim != 1:
        raise_dimension_error(
            "residuals must be a 1D array",
            array_name="residuals",
            expectedShape="(n_samples,)",
            actualShape=residuals.shape
        )
    
    # Check if dimensions match
    n_samples, n_features = X.shape
    if y.shape[0] != n_samples:
        raise_dimension_error(
            "Number of samples in y must match number of samples in X",
            array_name="y",
            expectedShape=f"({n_samples},)",
            actualShape=y.shape
        )
    
    if residuals.shape[0] != n_samples:
        raise_dimension_error(
            "Number of residuals must match number of samples in X",
            array_name="residuals",
            expectedShape=f"({n_samples},)",
            actualShape=residuals.shape
        )
    
    try:
        # Compute leverage values
        leverage = compute_leverage(X)
        
        # Compute MSE
        mse = np.sum(residuals ** 2) / (n_samples - n_features)
        
        # Compute studentized residuals
        studentized_residuals = residuals / np.sqrt(mse * (1 - leverage))
        
        # Compute Cook's distance
        cooks_distance = (studentized_residuals ** 2 / n_features) * (leverage / (1 - leverage))
        
        # Compute DFFITS
        dffits = studentized_residuals * np.sqrt(leverage / (1 - leverage))
        
        return {
            'cooks_distance': cooks_distance,
            'dffits': dffits,
            'leverage': leverage,
            'studentized_residuals': studentized_residuals
        }
    
    except np.linalg.LinAlgError:
        # If the matrix is singular, return NaN
        warnings.warn("Singular matrix encountered in influence computation. "
                     "Results may be unreliable.")
        return {
            'cooks_distance': np.full(n_samples, np.nan),
            'dffits': np.full(n_samples, np.nan),
            'leverage': np.full(n_samples, np.nan),
            'studentized_residuals': np.full(n_samples, np.nan)
        }


def compute_partial_corr(X: np.ndarray) -> np.ndarray:
    """
    Compute partial correlation matrix.
    
    This function computes the partial correlation matrix for a set of variables,
    which measures the correlation between pairs of variables while controlling
    for the effects of all other variables.
    
    Args:
        X: Data matrix (n_samples x n_features)
        
    Returns:
        Partial correlation matrix (n_features x n_features)
        
    Raises:
        DimensionError: If X is not a 2D array
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.cross_section.utils import compute_partial_corr
        >>> X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        >>> partial_corr = compute_partial_corr(X)
    """
    # Convert to numpy array if needed
    X = np.asarray(X)
    
    # Check if X is a 2D array
    if X.ndim != 2:
        raise_dimension_error(
            "X must be a 2D array",
            array_name="X",
            expectedShape="(n_samples, n_features)",
            actualShape=X.shape
        )
    
    # Compute correlation matrix
    corr = np.corrcoef(X, rowvar=False)
    
    # Compute partial correlation matrix
    try:
        # Compute precision matrix (inverse of correlation matrix)
        precision = np.linalg.inv(corr)
        
        # Compute partial correlation matrix
        n_features = X.shape[1]
        partial_corr = np.zeros((n_features, n_features))
        for i in range(n_features):
            for j in range(n_features):
                if i == j:
                    partial_corr[i, j] = 1.0
                else:
                    partial_corr[i, j] = -precision[i, j] / np.sqrt(precision[i, i] * precision[j, j])
        
        return partial_corr
    
    except np.linalg.LinAlgError:
        # If the matrix is singular, return NaN
        warnings.warn("Singular matrix encountered in partial correlation computation. "
                     "Results may be unreliable.")
        return np.full((X.shape[1], X.shape[1]), np.nan)


async def compute_eigenvalue_bootstrap(X: np.ndarray, 
                                     n_bootstrap: int = 1000,
                                     confidence_level: float = 0.95,
                                     random_state: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Bootstrap confidence intervals for eigenvalues in PCA.
    
    This function computes bootstrap confidence intervals for eigenvalues
    in Principal Component Analysis (PCA).
    
    Args:
        X: Data matrix (n_samples x n_features)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals (between 0 and 1)
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing:
            - 'eigenvalues': Original eigenvalues
            - 'lower_bound': Lower bounds of confidence intervals
            - 'upper_bound': Upper bounds of confidence intervals
        
    Raises:
        DimensionError: If X is not a 2D array
        ValueError: If confidence_level is not between 0 and 1
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.cross_section.utils import compute_eigenvalue_bootstrap
        >>> X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        >>> result = await compute_eigenvalue_bootstrap(X, n_bootstrap=100)
    """
    # Convert to numpy array if needed
    X = np.asarray(X)
    
    # Check if X is a 2D array
    if X.ndim != 2:
        raise_dimension_error(
            "X must be a 2D array",
            array_name="X",
            expectedShape="(n_samples, n_features)",
            actualShape=X.shape
        )
    
    # Check if confidence_level is valid
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    
    # Initialize random number generator
    rng = np.random.RandomState(random_state)
    
    # Get dimensions
    n_samples, n_features = X.shape
    
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov = X_centered.T @ X_centered / (n_samples - 1)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, _ = np.linalg.eigh(cov)
    
    # Sort in descending order
    eigenvalues = eigenvalues[::-1]
    
    # Initialize array to store bootstrap eigenvalues
    bootstrap_eigenvalues = np.zeros((n_bootstrap, n_features))
    
    # Perform bootstrap
    for i in range(n_bootstrap):
        # Generate bootstrap sample
        indices = rng.randint(0, n_samples, size=n_samples)
        X_boot = X_centered[indices]
        
        # Compute covariance matrix
        cov_boot = X_boot.T @ X_boot / (n_samples - 1)
        
        # Compute eigenvalues
        eigenvalues_boot, _ = np.linalg.eigh(cov_boot)
        
        # Sort in descending order
        eigenvalues_boot = eigenvalues_boot[::-1]
        
        # Store bootstrap eigenvalues
        bootstrap_eigenvalues[i] = eigenvalues_boot
        
        # Yield control to allow other tasks to run
        await asyncio.sleep(0)
    
    # Compute confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_eigenvalues, lower_percentile, axis=0)
    upper_bound = np.percentile(bootstrap_eigenvalues, upper_percentile, axis=0)
    
    return {
        'eigenvalues': eigenvalues,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }


async def compute_loading_bootstrap(X: np.ndarray, 
                                  n_components: int,
                                  n_bootstrap: int = 1000,
                                  confidence_level: float = 0.95,
                                  random_state: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Bootstrap confidence intervals for PCA loadings.
    
    This function computes bootstrap confidence intervals for loadings
    (eigenvectors) in Principal Component Analysis (PCA).
    
    Args:
        X: Data matrix (n_samples x n_features)
        n_components: Number of components to compute intervals for
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals (between 0 and 1)
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing:
            - 'loadings': Original loadings (n_features x n_components)
            - 'lower_bound': Lower bounds of confidence intervals
            - 'upper_bound': Upper bounds of confidence intervals
        
    Raises:
        DimensionError: If X is not a 2D array
        ValueError: If confidence_level is not between 0 and 1
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.cross_section.utils import compute_loading_bootstrap
        >>> X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        >>> result = await compute_loading_bootstrap(X, n_components=2, n_bootstrap=100)
    """
    # Convert to numpy array if needed
    X = np.asarray(X)
    
    # Check if X is a 2D array
    if X.ndim != 2:
        raise_dimension_error(
            "X must be a 2D array",
            array_name="X",
            expectedShape="(n_samples, n_features)",
            actualShape=X.shape
        )
    
    # Check if confidence_level is valid
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    
    # Check if n_components is valid
    n_samples, n_features = X.shape
    if not 1 <= n_components <= n_features:
        raise ValueError(f"n_components must be between 1 and {n_features}")
    
    # Initialize random number generator
    rng = np.random.RandomState(random_state)
    
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov = X_centered.T @ X_centered / (n_samples - 1)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select the first n_components eigenvectors
    loadings = eigenvectors[:, :n_components]
    
    # Initialize array to store bootstrap loadings
    bootstrap_loadings = np.zeros((n_bootstrap, n_features, n_components))
    
    # Perform bootstrap
    for i in range(n_bootstrap):
        # Generate bootstrap sample
        indices = rng.randint(0, n_samples, size=n_samples)
        X_boot = X_centered[indices]
        
        # Compute covariance matrix
        cov_boot = X_boot.T @ X_boot / (n_samples - 1)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues_boot, eigenvectors_boot = np.linalg.eigh(cov_boot)
        
        # Sort in descending order
        idx_boot = np.argsort(eigenvalues_boot)[::-1]
        eigenvectors_boot = eigenvectors_boot[:, idx_boot]
        
        # Select the first n_components eigenvectors
        loadings_boot = eigenvectors_boot[:, :n_components]
        
        # Ensure the sign of the loadings is consistent with the original
        for j in range(n_components):
            if np.sum(loadings[:, j] * loadings_boot[:, j]) < 0:
                loadings_boot[:, j] = -loadings_boot[:, j]
        
        # Store bootstrap loadings
        bootstrap_loadings[i] = loadings_boot
        
        # Yield control to allow other tasks to run
        await asyncio.sleep(0)
    
    # Compute confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_loadings, lower_percentile, axis=0)
    upper_bound = np.percentile(bootstrap_loadings, upper_percentile, axis=0)
    
    return {
        'loadings': loadings,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }


# Import asyncio for asynchronous functions
import asyncio

# Register Numba-accelerated functions if available
def _register_numba_functions() -> None:
    """
    Register Numba JIT-compiled functions for cross-sectional analysis utilities.
    
    This function is called during module initialization to register
    performance-critical functions for JIT compilation if Numba is available.
    """
    if HAS_NUMBA:
        logger.debug("Cross-sectional analysis Numba JIT functions registered")
    else:
        logger.info("Numba not available. Cross-sectional analysis utilities will use pure NumPy implementations.")


# Initialize the module
_register_numba_functions()
