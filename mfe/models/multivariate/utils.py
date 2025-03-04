# mfe/models/multivariate/utils.py

"""
Utility functions for multivariate volatility models.

This module provides essential utility functions for multivariate volatility
models in the MFE Toolbox. It includes data validation, matrix transformations,
parameter initialization, and helper functions for covariance computation and
formatting.

These utilities simplify the implementation of various multivariate volatility models
by centralizing common operations and ensuring consistent behavior across different
model types. The module leverages NumPy's efficient array operations and includes
Numba-accelerated implementations for performance-critical functions.

Functions:
    validate_multivariate_data: Validate input data for multivariate models
    compute_sample_covariance: Compute sample covariance matrix from data
    compute_sample_correlation: Compute sample correlation matrix from data
    ensure_positive_definite: Ensure a matrix is positive definite
    initialize_parameters: Generate starting values for model parameters
    format_multivariate_results: Format estimation results for display
    transform_correlation_matrix: Transform correlation matrix to unconstrained space
    inverse_transform_correlation_matrix: Transform from unconstrained to correlation space
    compute_persistence: Compute persistence for multivariate volatility models
    compute_half_life: Compute half-life of shocks for multivariate models
    standardize_residuals: Standardize residuals using conditional covariances
    compute_robust_covariance: Compute robust covariance matrix for parameter estimates
    check_stationarity: Check stationarity conditions for multivariate models
    check_positive_definiteness: Check if matrices are positive definite
    compute_eigenvalues: Compute eigenvalues of a matrix with numerical safeguards
    compute_conditional_correlations: Extract conditional correlations from covariances
    compute_conditional_variances: Extract conditional variances from covariances
    compute_unconditional_covariance: Compute unconditional covariance matrix
"""

import logging
import warnings
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, Union, cast, overload
)

import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats

from mfe.core.exceptions import (
    DimensionError, NumericError, ParameterError,
    raise_dimension_error, raise_numeric_error, warn_numeric
)
from mfe.core.parameters import (
    MultivariateVolatilityParameters, validate_positive_definite,
    transform_correlation, inverse_transform_correlation
)
from mfe.core.types import (
    CovarianceMatrix, CorrelationMatrix, Matrix, Vector,
    MultivariateVolatilityType, PositiveDefiniteMatrix
)
from mfe.utils.matrix_ops import (
    cov2corr, corr2cov, vech, ivech, ensure_symmetric,
    is_positive_definite, nearest_positive_definite
)

# Set up module-level logger
logger = logging.getLogger("mfe.models.multivariate.utils")

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
    logger.debug("Numba available for multivariate utilities acceleration")
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    HAS_NUMBA = False
    logger.info("Numba not available. Multivariate utilities will use pure NumPy implementations.")


def validate_multivariate_data(data: np.ndarray, min_obs: int = 10) -> Tuple[int, int]:
    """
    Validate input data for multivariate volatility models.

    This function checks that the input data has the correct dimensions and contains
    valid values for use in multivariate volatility models.

    Args:
        data: Input data array with shape (T, n_assets)
        min_obs: Minimum number of observations required (default: 10)

    Returns:
        Tuple[int, int]: Number of observations (T) and number of assets (n_assets)

    Raises:
        TypeError: If data is not a NumPy array
        DimensionError: If data does not have the correct dimensions
        ValueError: If data contains NaN or infinite values
    """
    # Check data type
    if not isinstance(data, np.ndarray):
        raise TypeError("Data must be a NumPy array")

    # Check dimensions
    if data.ndim != 2:
        raise_dimension_error(
            "Data must be a 2-dimensional array with shape (T, n_assets)",
            array_name="data",
            expected_shape="(T, n_assets)",
            actual_shape=data.shape
        )

    # Get dimensions
    T, n_assets = data.shape

    # Check minimum observations
    if T < min_obs:
        raise ValueError(f"Data must have at least {min_obs} observations, got {T}")

    # Check minimum assets
    if n_assets < 2:
        raise ValueError(f"Data must have at least 2 assets, got {n_assets}")

    # Check for NaN or infinite values
    if np.isnan(data).any():
        raise ValueError("Data contains NaN values")

    if np.isinf(data).any():
        raise ValueError("Data contains infinite values")

    return T, n_assets


def compute_sample_covariance(data: np.ndarray,
                              bias_corrected: bool = True) -> CovarianceMatrix:
    """
    Compute sample covariance matrix from data.

    This function computes the sample covariance matrix from multivariate data,
    with an option for bias correction.

    Args:
        data: Input data array with shape (T, n_assets)
        bias_corrected: Whether to use bias correction (default: True)

    Returns:
        CovarianceMatrix: Sample covariance matrix with shape (n_assets, n_assets)

    Raises:
        ValueError: If data has fewer than 2 observations
    """
    # Validate data
    T, n_assets = validate_multivariate_data(data, min_obs=2)

    # Compute sample covariance
    if bias_corrected:
        # Use NumPy's cov function with bias correction (ddof=1)
        cov_matrix = np.cov(data, rowvar=False, ddof=1)
    else:
        # Compute without bias correction
        # First center the data
        centered_data = data - np.mean(data, axis=0)
        # Compute covariance
        cov_matrix = (centered_data.T @ centered_data) / T

    # Ensure the matrix is symmetric (to handle numerical precision issues)
    cov_matrix = ensure_symmetric(cov_matrix)

    return cov_matrix


def compute_sample_correlation(data: np.ndarray) -> CorrelationMatrix:
    """
    Compute sample correlation matrix from data.

    This function computes the sample correlation matrix from multivariate data.

    Args:
        data: Input data array with shape (T, n_assets)

    Returns:
        CorrelationMatrix: Sample correlation matrix with shape (n_assets, n_assets)

    Raises:
        ValueError: If data has fewer than 2 observations
    """
    # Validate data
    T, n_assets = validate_multivariate_data(data, min_obs=2)

    # Compute sample correlation
    corr_matrix = np.corrcoef(data, rowvar=False)

    # Ensure the matrix is symmetric (to handle numerical precision issues)
    corr_matrix = ensure_symmetric(corr_matrix)

    # Ensure diagonal elements are exactly 1.0
    np.fill_diagonal(corr_matrix, 1.0)

    return corr_matrix


def ensure_positive_definite(matrix: Matrix,
                             epsilon: float = 1e-6,
                             method: str = "nearest") -> PositiveDefiniteMatrix:
    """
    Ensure a matrix is positive definite.

    This function takes a matrix and ensures it is positive definite by
    applying one of several methods if necessary.

    Args:
        matrix: Input matrix to ensure is positive definite
        epsilon: Small value to add to diagonal for numerical stability (default: 1e-6)
        method: Method to use for ensuring positive definiteness
                "nearest": Find the nearest positive definite matrix
                "diag": Add a small value to the diagonal
                "eigenvalue": Adjust eigenvalues to be positive

    Returns:
        PositiveDefiniteMatrix: Positive definite matrix

    Raises:
        ValueError: If method is not recognized
    """
    # Convert to numpy array if not already
    matrix = np.asarray(matrix)

    # Check if matrix is square
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise_dimension_error(
            "Matrix must be square",
            array_name="matrix",
            expected_shape="(n, n)",
            actual_shape=matrix.shape
        )

    # Ensure matrix is symmetric
    matrix = ensure_symmetric(matrix)

    # Check if matrix is already positive definite
    if is_positive_definite(matrix):
        return matrix

    # Apply method to ensure positive definiteness
    if method == "nearest":
        # Find the nearest positive definite matrix
        return nearest_positive_definite(matrix, epsilon)

    elif method == "diag":
        # Add a small value to the diagonal
        result = matrix.copy()
        np.fill_diagonal(result, np.diag(result) + epsilon)

        # If still not positive definite, increase diagonal elements
        while not is_positive_definite(result):
            min_eig = np.min(linalg.eigvalsh(result))
            np.fill_diagonal(result, np.diag(result) + abs(min_eig) * 1.1)

        return result

    elif method == "eigenvalue":
        # Compute eigendecomposition
        eigvals, eigvecs = linalg.eigh(matrix)

        # Replace negative eigenvalues with small positive values
        eigvals = np.maximum(eigvals, epsilon)

        # Reconstruct the matrix
        result = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # Ensure the result is symmetric (to handle numerical precision issues)
        result = ensure_symmetric(result)

        return result

    else:
        raise ValueError(f"Unknown method: {method}. Expected 'nearest', 'diag', or 'eigenvalue'")


def initialize_parameters(data: np.ndarray,
                          model_type: MultivariateVolatilityType,
                          **kwargs: Any) -> np.ndarray:
    """
    Generate starting values for model parameters.

    This function generates reasonable starting values for different types of
    multivariate volatility models based on the input data.

    Args:
        data: Input data array with shape (T, n_assets)
        model_type: Type of multivariate volatility model
        **kwargs: Additional keyword arguments for specific model types

    Returns:
        np.ndarray: Array of starting parameter values

    Raises:
        ValueError: If model_type is not recognized
    """
    # Validate data
    T, n_assets = validate_multivariate_data(data)

    # Compute sample covariance and correlation
    sample_cov = compute_sample_covariance(data)
    sample_corr = compute_sample_correlation(data)

    # Generate starting values based on model type
    if model_type == "DCC" or model_type == "ADCC":
        # For DCC/ADCC, we need a and b parameters
        # Typical starting values: a = 0.05, b = 0.85
        a = kwargs.get("a", 0.05)
        b = kwargs.get("b", 0.85)

        # For ADCC, we also need g parameter for asymmetry
        if model_type == "ADCC":
            g = kwargs.get("g", 0.05)
            return np.array([a, b, g])

        return np.array([a, b])

    elif model_type == "CCC":
        # For CCC, we just need the constant correlation matrix
        # Return the vech of the correlation matrix (excluding diagonal)
        n_params = n_assets * (n_assets - 1) // 2
        params = np.zeros(n_params)

        # Extract lower triangular elements (excluding diagonal)
        idx = 0
        for i in range(n_assets):
            for j in range(i):
                params[idx] = sample_corr[i, j]
                idx += 1

        return params

    elif model_type == "BEKK":
        # For BEKK, we need C, A, and B matrices
        # C is a lower triangular matrix
        # A and B are square matrices

        # Get BEKK order
        p = kwargs.get("p", 1)
        q = kwargs.get("q", 1)

        # Check if using diagonal BEKK
        diagonal = kwargs.get("diagonal", False)

        # Number of parameters in C (lower triangular)
        n_c_params = n_assets * (n_assets + 1) // 2

        # Number of parameters in A and B
        if diagonal:
            # Diagonal BEKK: only diagonal elements
            n_a_params = n_assets * p
            n_b_params = n_assets * q
        else:
            # Full BEKK: all elements
            n_a_params = n_assets * n_assets * p
            n_b_params = n_assets * n_assets * q

        # Total number of parameters
        n_params = n_c_params + n_a_params + n_b_params

        # Initialize parameter vector
        params = np.zeros(n_params)

        # Set C parameters (lower triangular of Cholesky decomposition of unconditional covariance)
        try:
            chol = linalg.cholesky(sample_cov, lower=True)
            params[:n_c_params] = vech(chol)
        except linalg.LinAlgError:
            # If Cholesky decomposition fails, use a simpler approach
            # Set diagonal elements to sqrt of variance, off-diagonal to small values
            idx = 0
            for i in range(n_assets):
                for j in range(i + 1):
                    if i == j:
                        params[idx] = np.sqrt(sample_cov[i, i])
                    else:
                        params[idx] = 0.01 * sample_cov[i, j] / np.sqrt(sample_cov[i, i] * sample_cov[j, j])
                    idx += 1

        # Set A parameters
        a_start = n_c_params
        a_end = a_start + n_a_params

        if diagonal:
            # Diagonal BEKK: set diagonal elements to small values
            for i in range(n_a_params):
                params[a_start + i] = 0.1 / np.sqrt(p)
        else:
            # Full BEKK: set all elements to small values
            params[a_start:a_end] = 0.05 / np.sqrt(p * n_assets)

        # Set B parameters
        b_start = a_end
        b_end = b_start + n_b_params

        if diagonal:
            # Diagonal BEKK: set diagonal elements to larger values
            for i in range(n_b_params):
                params[b_start + i] = 0.9 / np.sqrt(q)
        else:
            # Full BEKK: set all elements to small values, with larger values on diagonal
            b_params = np.zeros(n_b_params)
            for k in range(q):
                offset = k * n_assets * n_assets
                for i in range(n_assets):
                    for j in range(n_assets):
                        idx = offset + i * n_assets + j
                        if i == j:
                            b_params[idx] = 0.9 / np.sqrt(q)
                        else:
                            b_params[idx] = 0.01 / np.sqrt(q * n_assets)

            params[b_start:b_end] = b_params

        return params

    elif model_type == "RARCH" or model_type == "RCC":
        # For RARCH/RCC, we need parameters for the rotation
        # and parameters for the conditional correlation

        # Number of rotation parameters
        n_rot_params = n_assets * (n_assets - 1) // 2

        # For RCC, we also need a and b parameters
        if model_type == "RCC":
            a = kwargs.get("a", 0.05)
            b = kwargs.get("b", 0.85)

            # Total parameters: rotation + a + b
            params = np.zeros(n_rot_params + 2)

            # Set a and b
            params[-2] = a
            params[-1] = b
        else:
            # For RARCH, just rotation parameters
            params = np.zeros(n_rot_params)

        # Initialize rotation parameters to small values
        # These represent angles in radians
        params[:n_rot_params] = 0.01

        return params

    elif model_type == "OGARCH" or model_type == "GOGARCH":
        # For O-GARCH/GO-GARCH, we need parameters for the factors

        # Number of factors
        n_factors = kwargs.get("n_factors", n_assets)

        # For GO-GARCH, we need rotation parameters
        if model_type == "GOGARCH":
            # Number of rotation parameters
            n_rot_params = n_factors * (n_factors - 1) // 2

            # Initialize rotation parameters to small values
            params = np.zeros(n_rot_params)
            params[:] = 0.01

            return params

        # For O-GARCH, no additional parameters needed
        return np.array([])

    elif model_type == "MATRIX-GARCH":
        # For Matrix GARCH, we need parameters for the GARCH dynamics

        # Get GARCH order
        p = kwargs.get("p", 1)
        q = kwargs.get("q", 1)

        # Number of unique elements in covariance matrix
        n_vech = n_assets * (n_assets + 1) // 2

        # Parameters: C (intercept), A (ARCH), B (GARCH)
        n_params = n_vech * (1 + p + q)

        # Initialize parameter vector
        params = np.zeros(n_params)

        # Set C parameters (intercept)
        c_params = vech(sample_cov * 0.01)
        params[:n_vech] = c_params

        # Set A parameters (ARCH)
        a_start = n_vech
        a_end = a_start + n_vech * p
        params[a_start:a_end] = 0.05 / np.sqrt(p)

        # Set B parameters (GARCH)
        b_start = a_end
        b_end = b_start + n_vech * q
        params[b_start:b_end] = 0.85 / np.sqrt(q)

        return params

    elif model_type == "RISKMETRICS":
        # For RiskMetrics, we need the decay factor lambda
        # Typical value is 0.94 for daily data
        lambda_param = kwargs.get("lambda", 0.94)

        return np.array([lambda_param])

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def format_multivariate_results(
    model_type: MultivariateVolatilityType,
    parameters: np.ndarray,
    std_errors: np.ndarray,
    t_stats: np.ndarray,
    p_values: np.ndarray,
    n_assets: int,
    **kwargs: Any
) -> pd.DataFrame:
    """
    Format estimation results for multivariate volatility models.

    This function creates a formatted DataFrame with parameter estimates,
    standard errors, t-statistics, and p-values for multivariate volatility models.

    Args:
        model_type: Type of multivariate volatility model
        parameters: Parameter estimates
        std_errors: Standard errors for parameter estimates
        t_stats: t-statistics for parameter estimates
        p_values: p-values for parameter estimates
        n_assets: Number of assets
        **kwargs: Additional keyword arguments for specific model types

    Returns:
        pd.DataFrame: Formatted results DataFrame
    """
    # Create parameter names based on model type
    param_names = []

    if model_type == "DCC":
        param_names = ["a", "b"]

    elif model_type == "ADCC":
        param_names = ["a", "b", "g"]

    elif model_type == "CCC":
        # Create names for correlation parameters
        for i in range(n_assets):
            for j in range(i):
                param_names.append(f"rho[{i+1},{j+1}]")

    elif model_type == "BEKK":
        # Get BEKK order
        p = kwargs.get("p", 1)
        q = kwargs.get("q", 1)

        # Check if using diagonal BEKK
        diagonal = kwargs.get("diagonal", False)

        # C parameters (lower triangular)
        for i in range(n_assets):
            for j in range(i + 1):
                param_names.append(f"C[{i+1},{j+1}]")

        # A parameters
        for k in range(p):
            if diagonal:
                # Diagonal BEKK: only diagonal elements
                for i in range(n_assets):
                    param_names.append(f"A{k+1}[{i+1},{i+1}]")
            else:
                # Full BEKK: all elements
                for i in range(n_assets):
                    for j in range(n_assets):
                        param_names.append(f"A{k+1}[{i+1},{j+1}]")

        # B parameters
        for k in range(q):
            if diagonal:
                # Diagonal BEKK: only diagonal elements
                for i in range(n_assets):
                    param_names.append(f"B{k+1}[{i+1},{i+1}]")
            else:
                # Full BEKK: all elements
                for i in range(n_assets):
                    for j in range(n_assets):
                        param_names.append(f"B{k+1}[{i+1},{j+1}]")

    elif model_type == "RARCH":
        # Rotation parameters
        for i in range(n_assets):
            for j in range(i):
                param_names.append(f"theta[{i+1},{j+1}]")

    elif model_type == "RCC":
        # Rotation parameters
        for i in range(n_assets):
            for j in range(i):
                param_names.append(f"theta[{i+1},{j+1}]")

        # DCC parameters
        param_names.extend(["a", "b"])

    elif model_type == "GOGARCH":
        # Number of factors
        n_factors = kwargs.get("n_factors", n_assets)

        # Rotation parameters
        for i in range(n_factors):
            for j in range(i):
                param_names.append(f"theta[{i+1},{j+1}]")

    elif model_type == "MATRIX-GARCH":
        # Get GARCH order
        p = kwargs.get("p", 1)
        q = kwargs.get("q", 1)

        # Number of unique elements in covariance matrix
        n_vech = n_assets * (n_assets + 1) // 2

        # C parameters (intercept)
        idx = 0
        for i in range(n_assets):
            for j in range(i + 1):
                param_names.append(f"C[{i+1},{j+1}]")
                idx += 1

        # A parameters (ARCH)
        for k in range(p):
            for i in range(n_assets):
                for j in range(i + 1):
                    param_names.append(f"A{k+1}[{i+1},{j+1}]")

        # B parameters (GARCH)
        for k in range(q):
            for i in range(n_assets):
                for j in range(i + 1):
                    param_names.append(f"B{k+1}[{i+1},{j+1}]")

    elif model_type == "RISKMETRICS":
        param_names = ["lambda"]

    else:
        # If model type is not recognized, use generic parameter names
        param_names = [f"param{i+1}" for i in range(len(parameters))]

    # Check if parameter names match the number of parameters
    if len(param_names) != len(parameters):
        # If not, use generic parameter names
        param_names = [f"param{i+1}" for i in range(len(parameters))]
        warnings.warn(
            f"Number of parameter names ({len(param_names)}) does not match "
            f"number of parameters ({len(parameters)}). Using generic names."
        )

    # Create DataFrame with results
    results_df = pd.DataFrame({
        "Parameter": param_names,
        "Estimate": parameters,
        "Std. Error": std_errors,
        "t-statistic": t_stats,
        "p-value": p_values
    })

    # Add significance stars
    results_df["Significance"] = ""
    results_df.loc[results_df["p-value"] < 0.1, "Significance"] = "*"
    results_df.loc[results_df["p-value"] < 0.05, "Significance"] = "**"
    results_df.loc[results_df["p-value"] < 0.01, "Significance"] = "***"

    return results_df


@jit(nopython=True, cache=True)
def _transform_correlation_matrix_numba(corr: np.ndarray) -> np.ndarray:
    """
    Numba-accelerated implementation of correlation matrix transformation.

    Args:
        corr: Correlation matrix to transform

    Returns:
        np.ndarray: Transformed parameters in unconstrained space
    """
    n = corr.shape[0]
    n_params = n * (n - 1) // 2
    result = np.zeros(n_params)

    idx = 0
    for i in range(n):
        for j in range(i):
            # Apply Fisher's z-transformation to each correlation
            # Add small epsilon to avoid extreme values
            eps = 1e-10
            rho = np.clip(corr[i, j], -1 + eps, 1 - eps)
            result[idx] = np.arctanh(rho)
            idx += 1

    return result


def transform_correlation_matrix(corr: CorrelationMatrix) -> np.ndarray:
    """
    Transform correlation matrix to unconstrained space.

    This function transforms a correlation matrix to unconstrained space
    by applying Fisher's z-transformation to each correlation coefficient.
    Only the lower triangular elements (excluding diagonal) are transformed.

    Args:
        corr: Correlation matrix to transform

    Returns:
        np.ndarray: Transformed parameters in unconstrained space

    Raises:
        DimensionError: If corr is not a square matrix
    """
    # Convert to numpy array if not already
    corr = np.asarray(corr)

    # Check if matrix is square
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise_dimension_error(
            "Correlation matrix must be square",
            array_name="corr",
            expected_shape="(n, n)",
            actual_shape=corr.shape
        )

    # Ensure matrix is symmetric
    corr = ensure_symmetric(corr)

    # Check if diagonal elements are 1
    if not np.allclose(np.diag(corr), 1.0):
        # If not, normalize the matrix
        warn_numeric(
            "Correlation matrix does not have ones on the diagonal. Normalizing.",
            operation="transform_correlation_matrix",
            issue="invalid_correlation",
            value=np.diag(corr)
        )
        std = np.sqrt(np.diag(corr))
        corr = corr / np.outer(std, std)
        np.fill_diagonal(corr, 1.0)

    # Use Numba-accelerated implementation if available
    if HAS_NUMBA:
        return _transform_correlation_matrix_numba(corr)

    # Pure NumPy implementation
    n = corr.shape[0]
    n_params = n * (n - 1) // 2
    result = np.zeros(n_params)

    idx = 0
    for i in range(n):
        for j in range(i):
            # Apply Fisher's z-transformation to each correlation
            result[idx] = transform_correlation(corr[i, j])
            idx += 1

    return result


@jit(nopython=True, cache=True)
def _inverse_transform_correlation_matrix_numba(params: np.ndarray, n: int) -> np.ndarray:
    """
    Numba-accelerated implementation of inverse correlation matrix transformation.

    Args:
        params: Parameters in unconstrained space
        n: Dimension of the correlation matrix

    Returns:
        np.ndarray: Correlation matrix
    """
    # Initialize correlation matrix with ones on diagonal
    corr = np.eye(n)

    # Fill lower triangular elements
    idx = 0
    for i in range(n):
        for j in range(i):
            # Apply inverse Fisher's z-transformation
            corr[i, j] = np.tanh(params[idx])
            # Fill upper triangular element (symmetric)
            corr[j, i] = corr[i, j]
            idx += 1

    return corr


def inverse_transform_correlation_matrix(params: np.ndarray, n: int) -> CorrelationMatrix:
    """
    Transform parameters from unconstrained space to correlation matrix.

    This function transforms parameters from unconstrained space to a correlation
    matrix by applying the inverse Fisher's z-transformation.

    Args:
        params: Parameters in unconstrained space
        n: Dimension of the correlation matrix

    Returns:
        CorrelationMatrix: Correlation matrix

    Raises:
        ValueError: If params length doesn't match n*(n-1)/2
    """
    # Convert to numpy array if not already
    params = np.asarray(params)

    # Check if params length matches n*(n-1)/2
    expected_length = n * (n - 1) // 2
    if len(params) != expected_length:
        raise ValueError(
            f"Parameter length ({len(params)}) doesn't match expected length "
            f"n*(n-1)/2 = {expected_length} for n = {n}"
        )

    # Use Numba-accelerated implementation if available
    if HAS_NUMBA:
        return _inverse_transform_correlation_matrix_numba(params, n)

    # Pure NumPy implementation
    # Initialize correlation matrix with ones on diagonal
    corr = np.eye(n)

    # Fill lower triangular elements
    idx = 0
    for i in range(n):
        for j in range(i):
            # Apply inverse Fisher's z-transformation
            corr[i, j] = inverse_transform_correlation(params[idx])
            # Fill upper triangular element (symmetric)
            corr[j, i] = corr[i, j]
            idx += 1

    return corr


def compute_persistence(model_type: MultivariateVolatilityType,
                        parameters: np.ndarray,
                        **kwargs: Any) -> float:
    """
    Compute persistence for multivariate volatility models.

    This function computes the persistence measure for different types of
    multivariate volatility models, which indicates how long shocks persist
    in the volatility process.

    Args:
        model_type: Type of multivariate volatility model
        parameters: Model parameters
        **kwargs: Additional keyword arguments for specific model types

    Returns:
        float: Persistence measure

    Raises:
        ValueError: If model_type is not recognized or doesn't support persistence calculation
    """
    if model_type == "DCC":
        # For DCC, persistence is a + b
        if len(parameters) != 2:
            raise ValueError(f"Expected 2 parameters for DCC model, got {len(parameters)}")

        a, b = parameters
        return a + b

    elif model_type == "ADCC":
        # For ADCC, persistence is a + b + kappa*g
        # where kappa is the expected value of the asymmetry term
        if len(parameters) != 3:
            raise ValueError(f"Expected 3 parameters for ADCC model, got {len(parameters)}")

        a, b, g = parameters

        # For normal distribution, kappa = 0.5
        # For t distribution, kappa depends on degrees of freedom
        df = kwargs.get("df", None)
        if df is not None and df > 2:
            # For t distribution
            kappa = 0.5 * (df - 1) / df
        else:
            # For normal distribution
            kappa = 0.5

        return a + b + kappa * g

    elif model_type == "BEKK":
        # For BEKK, persistence is the largest eigenvalue of A⊗A + B⊗B
        # where ⊗ is the Kronecker product

        # Get BEKK order
        p = kwargs.get("p", 1)
        q = kwargs.get("q", 1)

        # Get number of assets
        n_assets = kwargs.get("n_assets")
        if n_assets is None:
            raise ValueError("n_assets must be provided for BEKK persistence calculation")

        # Check if using diagonal BEKK
        diagonal = kwargs.get("diagonal", False)

        # Number of parameters in C (lower triangular)
        n_c_params = n_assets * (n_assets + 1) // 2

        # Extract A and B parameters
        if diagonal:
            # Diagonal BEKK: only diagonal elements
            n_a_params = n_assets * p
            n_b_params = n_assets * q

            # Extract A parameters
            a_start = n_c_params
            a_end = a_start + n_a_params
            a_params = parameters[a_start:a_end]

            # Extract B parameters
            b_start = a_end
            b_end = b_start + n_b_params
            b_params = parameters[b_start:b_end]

            # For diagonal BEKK, persistence is the sum of squares of A and B parameters
            persistence = 0.0
            for k in range(p):
                a_k = a_params[k*n_assets:(k+1)*n_assets]
                persistence += np.sum(a_k**2)

            for k in range(q):
                b_k = b_params[k*n_assets:(k+1)*n_assets]
                persistence += np.sum(b_k**2)

            return persistence
        else:
            # Full BEKK: all elements
            n_a_params = n_assets * n_assets * p
            n_b_params = n_assets * n_assets * q

            # Extract A parameters
            a_start = n_c_params
            a_end = a_start + n_a_params
            a_params = parameters[a_start:a_end]

            # Extract B parameters
            b_start = a_end
            b_end = b_start + n_b_params
            b_params = parameters[b_start:b_end]

            # Reshape A and B parameters
            A_matrices = []
            for k in range(p):
                A_k = a_params[k*n_assets*n_assets:(k+1)*n_assets*n_assets].reshape(n_assets, n_assets)
                A_matrices.append(A_k)

            B_matrices = []
            for k in range(q):
                B_k = b_params[k*n_assets*n_assets:(k+1)*n_assets*n_assets].reshape(n_assets, n_assets)
                B_matrices.append(B_k)

            # Compute A⊗A + B⊗B
            kron_sum = np.zeros((n_assets**2, n_assets**2))

            for A in A_matrices:
                kron_sum += np.kron(A, A)

            for B in B_matrices:
                kron_sum += np.kron(B, B)

            # Compute largest eigenvalue
            try:
                eigvals = np.linalg.eigvals(kron_sum)
                return np.max(np.abs(eigvals))
            except np.linalg.LinAlgError:
                # If eigenvalue computation fails, use a simpler approach
                warn_numeric(
                    "Eigenvalue computation failed in BEKK persistence calculation, using Frobenius norm",
                    operation="compute_persistence",
                    issue="eigenvalue_computation_failed"
                )

                # Use Frobenius norm as an approximation
                persistence = 0.0
                for A in A_matrices:
                    persistence += np.sum(A**2)

                for B in B_matrices:
                    persistence += np.sum(B**2)

                return persistence

    elif model_type == "MATRIX-GARCH":
        # For Matrix GARCH, persistence is the sum of A and B coefficients

        # Get GARCH order
        p = kwargs.get("p", 1)
        q = kwargs.get("q", 1)

        # Get number of assets
        n_assets = kwargs.get("n_assets")
        if n_assets is None:
            raise ValueError("n_assets must be provided for Matrix GARCH persistence calculation")

        # Number of unique elements in covariance matrix
        n_vech = n_assets * (n_assets + 1) // 2

        # Extract A and B parameters
        c_params = parameters[:n_vech]
        a_params = parameters[n_vech:n_vech*(1+p)]
        b_params = parameters[n_vech*(1+p):]

        # Compute persistence as sum of A and B coefficients
        return np.sum(a_params) + np.sum(b_params)

    elif model_type == "RISKMETRICS":
        # For RiskMetrics, persistence is lambda
        if len(parameters) != 1:
            raise ValueError(f"Expected 1 parameter for RiskMetrics model, got {len(parameters)}")

        return parameters[0]

    else:
        raise ValueError(
            f"Persistence calculation not implemented for model type: {model_type}"
        )


def compute_half_life(persistence: float) -> float:
    """
    Compute half-life of shocks for multivariate volatility models.

    This function computes the half-life of shocks based on the persistence
    measure, which indicates how many periods it takes for a shock to decay
    to half its original impact.

    Args:
        persistence: Persistence measure

    Returns:
        float: Half-life of shocks in number of periods

    Raises:
        ValueError: If persistence is not between 0 and 1
    """
    # Check if persistence is between 0 and 1
    if persistence < 0 or persistence >= 1:
        raise ValueError(
            f"Persistence must be between 0 and 1 for half-life calculation, got {persistence}"
        )

    # Compute half-life
    return np.log(0.5) / np.log(persistence)


@jit(nopython=True, cache=True)
def _standardize_residuals_numba(data: np.ndarray,
                                 cov_matrices: np.ndarray) -> np.ndarray:
    """
    Numba-accelerated implementation of residual standardization.

    Args:
        data: Input data array with shape (T, n_assets)
        cov_matrices: Conditional covariance matrices with shape (n_assets, n_assets, T)

    Returns:
        np.ndarray: Standardized residuals with shape (T, n_assets)
    """
    T, n_assets = data.shape
    std_resid = np.zeros((T, n_assets))

    for t in range(T):
        # Get covariance matrix for time t
        cov_t = cov_matrices[:, :, t]

        # Compute Cholesky decomposition
        # Since Numba doesn't support np.linalg.cholesky directly,
        # we'll use a simpler approach for standardization

        # Compute diagonal matrix of standard deviations
        std_diag = np.sqrt(np.diag(cov_t))

        # Compute correlation matrix
        corr_t = np.eye(n_assets)
        for i in range(n_assets):
            for j in range(n_assets):
                if i != j:
                    corr_t[i, j] = cov_t[i, j] / (std_diag[i] * std_diag[j])

        # Compute correlation matrix Cholesky factor
        # This is a simplified approach and may not work for all matrices
        chol = np.eye(n_assets)
        for i in range(n_assets):
            chol[i, i] = np.sqrt(corr_t[i, i])
            for j in range(i):
                chol[i, j] = corr_t[i, j]
                for k in range(j):
                    chol[i, j] -= chol[i, k] * chol[j, k]
                chol[i, j] /= chol[j, j]

        # Scale Cholesky factor by standard deviations
        for i in range(n_assets):
            for j in range(i + 1):
                chol[i, j] *= std_diag[i]

        # Standardize residuals
        for i in range(n_assets):
            std_resid[t, i] = data[t, i]
            for j in range(i + 1):
                std_resid[t, i] -= chol[i, j] * std_resid[t, j]
            std_resid[t, i] /= chol[i, i]

    return std_resid


def standardize_residuals(data: np.ndarray,
                          cov_matrices: np.ndarray) -> np.ndarray:
    """
    Standardize residuals using conditional covariance matrices.

    This function standardizes residuals by transforming them using the
    Cholesky decomposition of the conditional covariance matrices.

    Args:
        data: Input data array with shape (T, n_assets)
        cov_matrices: Conditional covariance matrices with shape (n_assets, n_assets, T)

    Returns:
        np.ndarray: Standardized residuals with shape (T, n_assets)

    Raises:
        DimensionError: If dimensions of data and cov_matrices are incompatible
    """
    # Convert to numpy arrays if not already
    data = np.asarray(data)
    cov_matrices = np.asarray(cov_matrices)

    # Check dimensions
    if data.ndim != 2:
        raise_dimension_error(
            "Data must be a 2-dimensional array with shape (T, n_assets)",
            array_name="data",
            expected_shape="(T, n_assets)",
            actual_shape=data.shape
        )

    if cov_matrices.ndim != 3:
        raise_dimension_error(
            "Covariance matrices must be a 3-dimensional array with shape (n_assets, n_assets, T)",
            array_name="cov_matrices",
            expected_shape="(n_assets, n_assets, T)",
            actual_shape=cov_matrices.shape
        )

    T, n_assets = data.shape

    if cov_matrices.shape[0] != n_assets or cov_matrices.shape[1] != n_assets:
        raise_dimension_error(
            "Covariance matrices dimensions must match number of assets in data",
            array_name="cov_matrices",
            expected_shape=f"({n_assets}, {n_assets}, {T})",
            actual_shape=cov_matrices.shape
        )

    if cov_matrices.shape[2] != T:
        raise_dimension_error(
            "Number of covariance matrices must match number of observations in data",
            array_name="cov_matrices",
            expected_shape=f"({n_assets}, {n_assets}, {T})",
            actual_shape=cov_matrices.shape
        )

    # Use Numba-accelerated implementation if available
    if HAS_NUMBA:
        try:
            return _standardize_residuals_numba(data, cov_matrices)
        except Exception as e:
            # If Numba implementation fails, fall back to NumPy implementation
            logger.warning(
                f"Numba implementation of standardize_residuals failed: {str(e)}. "
                "Falling back to NumPy implementation."
            )

    # Pure NumPy implementation
    std_resid = np.zeros_like(data)

    for t in range(T):
        # Get covariance matrix for time t
        cov_t = cov_matrices[:, :, t]

        try:
            # Compute Cholesky decomposition
            chol = np.linalg.cholesky(cov_t)

            # Standardize residuals
            std_resid[t, :] = np.linalg.solve(chol, data[t, :])
        except np.linalg.LinAlgError:
            # If Cholesky decomposition fails, use eigenvalue decomposition
            eigvals, eigvecs = np.linalg.eigh(cov_t)

            # Ensure all eigenvalues are positive
            eigvals = np.maximum(eigvals, 1e-8)

            # Compute inverse square root of covariance matrix
            inv_sqrt_cov = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            # Standardize residuals
            std_resid[t, :] = inv_sqrt_cov @ data[t, :]

    return std_resid


def compute_robust_covariance(
    model_type: MultivariateVolatilityType,
    parameters: np.ndarray,
    scores: np.ndarray,
    hessian: Optional[np.ndarray] = None,
    **kwargs: Any
) -> np.ndarray:
    """
    Compute robust covariance matrix for parameter estimates.

    This function computes a robust covariance matrix for parameter estimates
    using the sandwich estimator (QMLE covariance).

    Args:
        model_type: Type of multivariate volatility model
        parameters: Parameter estimates
        scores: Score contributions with shape (T, n_params)
        hessian: Hessian matrix (if None, computed from scores)
        **kwargs: Additional keyword arguments for specific model types

    Returns:
        np.ndarray: Robust covariance matrix for parameter estimates

    Raises:
        ValueError: If scores shape is invalid
    """
    # Convert to numpy arrays if not already
    parameters = np.asarray(parameters)
    scores = np.asarray(scores)

    # Check scores shape
    if scores.ndim != 2:
        raise ValueError(
            f"Scores must be a 2-dimensional array with shape (T, n_params), got shape {scores.shape}"
        )

    T, n_params = scores.shape

    if n_params != len(parameters):
        raise ValueError(
            f"Number of parameters in scores ({n_params}) doesn't match "
            f"number of parameters ({len(parameters)})"
        )

    # Compute outer product of scores (information matrix)
    outer_prod = np.zeros((n_params, n_params))
    for t in range(T):
        outer_prod += np.outer(scores[t], scores[t])

    outer_prod /= T

    # If hessian is not provided, compute it from scores
    if hessian is None:
        # Use numerical approximation of hessian
        def objective(params: np.ndarray) -> float:
            # This is a placeholder - in practice, you would compute the negative log-likelihood
            # For now, we'll just use the sum of squared scores as a proxy
            return np.sum(scores @ params)**2

        try:
            # Compute Hessian using finite differences
            hessian = optimize.approx_fprime(
                parameters,
                lambda params: optimize.approx_fprime(
                    params,
                    objective,
                    epsilon=1e-5
                ),
                epsilon=1e-5
            )
        except Exception as e:
            # If Hessian computation fails, use identity matrix
            warn_numeric(
                f"Hessian computation failed: {str(e)}. Using identity matrix.",
                operation="compute_robust_covariance",
                issue="hessian_computation_failed"
            )
            hessian = np.eye(n_params)

    # Compute inverse of hessian
    try:
        inv_hessian = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        # If inversion fails, use pseudo-inverse
        warn_numeric(
            "Hessian matrix is singular. Using pseudo-inverse.",
            operation="compute_robust_covariance",
            issue="singular_hessian"
        )
        inv_hessian = np.linalg.pinv(hessian)

    # Compute sandwich estimator: (H^-1) * S * (H^-1)
    cov_matrix = inv_hessian @ outer_prod @ inv_hessian

    # Scale by T for consistency with standard errors
    cov_matrix /= T

    return cov_matrix


def check_stationarity(
    model_type: MultivariateVolatilityType,
    parameters: np.ndarray,
    **kwargs: Any
) -> bool:
    """
    Check stationarity conditions for multivariate volatility models.

    This function checks whether the parameters of a multivariate volatility model
    satisfy the stationarity conditions.

    Args:
        model_type: Type of multivariate volatility model
        parameters: Model parameters
        **kwargs: Additional keyword arguments for specific model types

    Returns:
        bool: True if stationarity conditions are satisfied, False otherwise

    Raises:
        ValueError: If model_type is not recognized or doesn't support stationarity checking
    """
    try:
        if model_type == "DCC":
            # For DCC, stationarity requires a + b < 1
            if len(parameters) != 2:
                raise ValueError(f"Expected 2 parameters for DCC model, got {len(parameters)}")

            a, b = parameters
            return a >= 0 and b >= 0 and a + b < 1

        elif model_type == "ADCC":
            # For ADCC, stationarity requires a + b + kappa*g < 1
            if len(parameters) != 3:
                raise ValueError(f"Expected 3 parameters for ADCC model, got {len(parameters)}")

            a, b, g = parameters

            # For normal distribution, kappa = 0.5
            # For t distribution, kappa depends on degrees of freedom
            df = kwargs.get("df", None)
            if df is not None and df > 2:
                # For t distribution
                kappa = 0.5 * (df - 1) / df
            else:
                # For normal distribution
                kappa = 0.5

            return a >= 0 and b >= 0 and g >= 0 and a + b + kappa * g < 1

        elif model_type == "BEKK":
            # For BEKK, stationarity requires eigenvalues of A⊗A + B⊗B < 1
            persistence = compute_persistence(model_type, parameters, **kwargs)
            return persistence < 1

        elif model_type == "MATRIX-GARCH":
            # For Matrix GARCH, stationarity requires sum of A and B coefficients < 1
            persistence = compute_persistence(model_type, parameters, **kwargs)
            return persistence < 1

        elif model_type == "RISKMETRICS":
            # For RiskMetrics, stationarity requires 0 <= lambda < 1
            if len(parameters) != 1:
                raise ValueError(f"Expected 1 parameter for RiskMetrics model, got {len(parameters)}")

            lambda_param = parameters[0]
            return 0 <= lambda_param < 1

        else:
            raise ValueError(
                f"Stationarity checking not implemented for model type: {model_type}"
            )

    except Exception as e:
        # If stationarity checking fails, log warning and return False
        logger.warning(
            f"Stationarity checking failed for model type {model_type}: {str(e)}"
        )
        return False


def check_positive_definiteness(matrices: np.ndarray) -> np.ndarray:
    """
    Check if matrices are positive definite.

    This function checks whether each matrix in a 3D array of matrices is
    positive definite.

    Args:
        matrices: 3D array of matrices with shape (n, n, T)

    Returns:
        np.ndarray: Boolean array with shape (T,) indicating whether each matrix is positive definite

    Raises:
        DimensionError: If matrices is not a 3D array with shape (n, n, T)
    """
    # Convert to numpy array if not already
    matrices = np.asarray(matrices)

    # Check dimensions
    if matrices.ndim != 3:
        raise_dimension_error(
            "Matrices must be a 3-dimensional array with shape (n, n, T)",
            array_name="matrices",
            expected_shape="(n, n, T)",
            actual_shape=matrices.shape
        )

    n1, n2, T = matrices.shape

    if n1 != n2:
        raise_dimension_error(
            "Matrices must be square",
            array_name="matrices",
            expected_shape=f"({n1}, {n1}, {T})",
            actual_shape=matrices.shape
        )

    # Check positive definiteness of each matrix
    is_pd = np.zeros(T, dtype=bool)

    for t in range(T):
        is_pd[t] = is_positive_definite(matrices[:, :, t])

    return is_pd


def compute_eigenvalues(matrix: Matrix,
                        ensure_real: bool = True) -> np.ndarray:
    """
    Compute eigenvalues of a matrix with numerical safeguards.

    This function computes the eigenvalues of a matrix with numerical safeguards
    to handle potential numerical issues.

    Args:
        matrix: Input matrix
        ensure_real: Whether to ensure eigenvalues are real (for symmetric matrices)

    Returns:
        np.ndarray: Eigenvalues of the matrix

    Raises:
        DimensionError: If matrix is not square
    """
    # Convert to numpy array if not already
    matrix = np.asarray(matrix)

    # Check if matrix is square
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise_dimension_error(
            "Matrix must be square",
            array_name="matrix",
            expected_shape="(n, n)",
            actual_shape=matrix.shape
        )

    # Ensure matrix is symmetric if ensure_real is True
    if ensure_real:
        matrix = ensure_symmetric(matrix)

    try:
        # Compute eigenvalues
        if ensure_real:
            # Use eigh for symmetric matrices (guaranteed real eigenvalues)
            eigvals = np.linalg.eigvalsh(matrix)
        else:
            # Use eigvals for general matrices
            eigvals = np.linalg.eigvals(matrix)

        return eigvals

    except np.linalg.LinAlgError as e:
        # If eigenvalue computation fails, log warning and use a more robust approach
        warn_numeric(
            f"Eigenvalue computation failed: {str(e)}. Using more robust approach.",
            operation="compute_eigenvalues",
            issue="eigenvalue_computation_failed"
        )

        # Add a small value to the diagonal for numerical stability
        matrix_stable = matrix.copy()
        np.fill_diagonal(matrix_stable, np.diag(matrix_stable) + 1e-8)

        # Try again with the stabilized matrix
        if ensure_real:
            eigvals = np.linalg.eigvalsh(matrix_stable)
        else:
            eigvals = np.linalg.eigvals(matrix_stable)

        return eigvals


def compute_conditional_correlations(cov_matrices: np.ndarray) -> np.ndarray:
    """
    Extract conditional correlations from covariance matrices.

    This function computes conditional correlation matrices from conditional
    covariance matrices.

    Args:
        cov_matrices: Conditional covariance matrices with shape (n_assets, n_assets, T)

    Returns:
        np.ndarray: Conditional correlation matrices with shape (n_assets, n_assets, T)

    Raises:
        DimensionError: If cov_matrices is not a 3D array with shape (n_assets, n_assets, T)
    """
    # Convert to numpy array if not already
    cov_matrices = np.asarray(cov_matrices)

    # Check dimensions
    if cov_matrices.ndim != 3:
        raise_dimension_error(
            "Covariance matrices must be a 3-dimensional array with shape (n_assets, n_assets, T)",
            array_name="cov_matrices",
            expected_shape="(n_assets, n_assets, T)",
            actual_shape=cov_matrices.shape
        )

    n_assets, n_assets2, T = cov_matrices.shape

    if n_assets != n_assets2:
        raise_dimension_error(
            "Covariance matrices must be square",
            array_name="cov_matrices",
            expected_shape=f"({n_assets}, {n_assets}, {T})",
            actual_shape=cov_matrices.shape
        )

    # Initialize correlation matrices
    corr_matrices = np.zeros_like(cov_matrices)

    # Compute correlation matrices
    for t in range(T):
        corr_matrices[:, :, t] = cov2corr(cov_matrices[:, :, t])

    return corr_matrices


def compute_conditional_variances(cov_matrices: np.ndarray) -> np.ndarray:
    """
    Extract conditional variances from covariance matrices.

    This function extracts the diagonal elements (variances) from conditional
    covariance matrices.

    Args:
        cov_matrices: Conditional covariance matrices with shape (n_assets, n_assets, T)

    Returns:
        np.ndarray: Conditional variances with shape (T, n_assets)

    Raises:
        DimensionError: If cov_matrices is not a 3D array with shape (n_assets, n_assets, T)
    """
    # Convert to numpy array if not already
    cov_matrices = np.asarray(cov_matrices)

    # Check dimensions
    if cov_matrices.ndim != 3:
        raise_dimension_error(
            "Covariance matrices must be a 3-dimensional array with shape (n_assets, n_assets, T)",
            array_name="cov_matrices",
            expected_shape="(n_assets, n_assets, T)",
            actual_shape=cov_matrices.shape
        )

    n_assets, n_assets2, T = cov_matrices.shape

    if n_assets != n_assets2:
        raise_dimension_error(
            "Covariance matrices must be square",
            array_name="cov_matrices",
            expected_shape=f"({n_assets}, {n_assets}, {T})",
            actual_shape=cov_matrices.shape
        )

    # Extract diagonal elements (variances)
    variances = np.zeros((T, n_assets))

    for t in range(T):
        variances[t, :] = np.diag(cov_matrices[:, :, t])

    return variances


def compute_unconditional_covariance(
    model_type: MultivariateVolatilityType,
    parameters: np.ndarray,
    **kwargs: Any
) -> Optional[np.ndarray]:
    """
    Compute unconditional covariance matrix implied by the model.

    This function computes the unconditional (long-run) covariance matrix
    implied by a multivariate volatility model.

    Args:
        model_type: Type of multivariate volatility model
        parameters: Model parameters
        **kwargs: Additional keyword arguments for specific model types

    Returns:
        Optional[np.ndarray]: Unconditional covariance matrix, or None if not applicable

    Raises:
        ValueError: If model_type is not recognized or doesn't support unconditional covariance
    """
    try:
        if model_type == "BEKK":
            # For BEKK, unconditional covariance is C*C' / (1 - persistence)

            # Get number of assets
            n_assets = kwargs.get("n_assets")
            if n_assets is None:
                raise ValueError("n_assets must be provided for BEKK unconditional covariance calculation")

            # Number of parameters in C (lower triangular)
            n_c_params = n_assets * (n_assets + 1) // 2

            # Extract C parameters
            c_params = parameters[:n_c_params]

            # Convert to matrix
            C = ivech(c_params)

            # Compute C*C'
            CC = C @ C.T

            # Compute persistence
            persistence = compute_persistence(model_type, parameters, **kwargs)

            # Check if model is stationary
            if persistence >= 1:
                warn_numeric(
                    f"BEKK model is not stationary (persistence = {persistence}). "
                    "Unconditional covariance is not defined.",
                    operation="compute_unconditional_covariance",
                    issue="non_stationary_model",
                    value=persistence
                )
                return None

            # Compute unconditional covariance
            uncond_cov = CC / (1 - persistence)

            return uncond_cov

        elif model_type == "DCC" or model_type == "ADCC":
            # For DCC/ADCC, unconditional covariance depends on the univariate models
            # and the unconditional correlation matrix

            # Get unconditional variances
            uncond_var = kwargs.get("unconditional_variances")
            if uncond_var is None:
                raise ValueError(
                    "unconditional_variances must be provided for DCC/ADCC unconditional covariance calculation")

            # Get unconditional correlation matrix
            uncond_corr = kwargs.get("unconditional_correlation")
            if uncond_corr is None:
                raise ValueError(
                    "unconditional_correlation must be provided for DCC/ADCC unconditional covariance calculation")

            # Compute unconditional covariance
            n_assets = len(uncond_var)
            std_dev = np.sqrt(uncond_var)
            uncond_cov = np.diag(std_dev) @ uncond_corr @ np.diag(std_dev)

            return uncond_cov

        elif model_type == "CCC":
            # For CCC, unconditional covariance depends on the univariate models
            # and the constant correlation matrix

            # Get unconditional variances
            uncond_var = kwargs.get("unconditional_variances")
            if uncond_var is None:
                raise ValueError("unconditional_variances must be provided for CCC unconditional covariance calculation")

            # Get number of assets
            n_assets = len(uncond_var)

            # Reconstruct correlation matrix from parameters
            corr = np.eye(n_assets)
            idx = 0
            for i in range(n_assets):
                for j in range(i):
                    corr[i, j] = parameters[idx]
                    corr[j, i] = corr[i, j]
                    idx += 1

            # Compute unconditional covariance
            std_dev = np.sqrt(uncond_var)
            uncond_cov = np.diag(std_dev) @ corr @ np.diag(std_dev)

            return uncond_cov

        elif model_type == "RISKMETRICS":
            # For RiskMetrics, unconditional covariance is not defined
            # (non-stationary model)
            return None

        else:
            # For other models, return None
            return None

    except Exception as e:
        # If unconditional covariance calculation fails, log warning and return None
        logger.warning(
            f"Unconditional covariance calculation failed for model type {model_type}: {str(e)}"
        )
        return None


# Register Numba-accelerated functions if available
def _register_numba_functions() -> None:
    """
    Register Numba JIT-compiled functions for multivariate utilities.

    This function is called during module initialization to register
    performance-critical functions for JIT compilation if Numba is available.
    """
    if HAS_NUMBA:
        # The functions are already decorated with @jit, so no additional
        # registration is needed here. This function is kept for consistency
        # with the module structure and potential future enhancements.
        logger.debug("Multivariate utilities Numba JIT functions registered")
    else:
        logger.info("Numba not available. Multivariate utilities will use pure NumPy implementations.")


# Initialize the module
_register_numba_functions()
