'''
Numba-Accelerated Core Functions for Cross-Sectional Analysis

This module provides performance-critical functions for cross-sectional analysis
accelerated with Numba's Just-In-Time (JIT) compilation. These optimized implementations
significantly improve performance for computationally intensive operations in OLS
regression and Principal Component Analysis.

The module leverages Numba's @jit decorator to compile Python functions to optimized
machine code at runtime, providing near-native performance for numerical operations
while maintaining the flexibility and readability of Python code. All functions
include proper handling of zero-based indexing and are optimized for NumPy's row-major
memory layout.

Functions:
    _ols_fit_core: Accelerated OLS parameter estimation
    _ols_predict_core: Accelerated OLS prediction
    _ols_residuals_core: Accelerated OLS residual computation
    _ols_robust_errors_core: Accelerated robust standard error computation
    _pca_outer_product_core: Accelerated PCA outer product computation
    _pca_covariance_core: Accelerated PCA covariance matrix computation
    _pca_correlation_core: Accelerated PCA correlation matrix computation
    _pca_transform_core: Accelerated PCA data transformation
    _pca_inverse_transform_core: Accelerated PCA inverse transformation
'''

import logging
import warnings
from typing import Optional, Tuple, Union, Dict, Any

import numpy as np

# Set up module-level logger
logger = logging.getLogger("mfe.models.cross_section._numba_core")

# Try to import numba for JIT compilation
try:
    from numba import jit, prange
    HAS_NUMBA = True
    logger.debug("Numba available for cross-sectional analysis acceleration")
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    
    # Create a no-op parallel range function
    prange = range
    
    HAS_NUMBA = False
    logger.info("Numba not available. Cross-sectional analysis will use pure NumPy implementations.")


@jit(nopython=True, cache=True)
def _ols_fit_core(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-accelerated core implementation of OLS parameter estimation.
    
    This function computes OLS parameter estimates, fitted values, and residuals
    using optimized matrix operations. It is designed for maximum performance
    with large datasets.
    
    Args:
        X: Design matrix (n x k)
        y: Dependent variable (n x 1)
        
    Returns:
        Tuple containing:
            - beta: Parameter estimates (k x 1)
            - fitted: Fitted values (n x 1)
            - residuals: Residuals (n x 1)
            - XpXinv: (X'X)^(-1) matrix for inference (k x k)
    """
    # Compute X'X
    XpX = X.T @ X
    
    # Compute X'y
    Xpy = X.T @ y
    
    # Compute (X'X)^(-1)
    XpXinv = np.linalg.inv(XpX)
    
    # Compute beta = (X'X)^(-1)X'y
    beta = XpXinv @ Xpy
    
    # Compute fitted values
    fitted = X @ beta
    
    # Compute residuals
    residuals = y - fitted
    
    return beta, fitted, residuals, XpXinv


@jit(nopython=True, cache=True)
def _ols_predict_core(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Numba-accelerated core implementation of OLS prediction.
    
    This function computes predictions from OLS parameter estimates using
    optimized matrix operations.
    
    Args:
        X: Design matrix for prediction (n x k)
        beta: Parameter estimates (k x 1)
        
    Returns:
        Predicted values (n x 1)
    """
    # Compute predictions
    predictions = X @ beta
    
    return predictions


@jit(nopython=True, cache=True)
def _ols_residuals_core(y: np.ndarray, fitted: np.ndarray) -> np.ndarray:
    """
    Numba-accelerated core implementation of OLS residual computation.
    
    This function computes residuals from observed and fitted values using
    optimized operations.
    
    Args:
        y: Observed dependent variable (n x 1)
        fitted: Fitted values (n x 1)
        
    Returns:
        Residuals (n x 1)
    """
    # Compute residuals
    residuals = y - fitted
    
    return residuals


@jit(nopython=True, cache=True)
def _ols_robust_errors_core(X: np.ndarray, residuals: np.ndarray, XpXinv: np.ndarray, 
                           hc_type: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba-accelerated core implementation of robust standard error computation.
    
    This function computes heteroskedasticity-robust standard errors for OLS
    regression using various HC estimators (HC0, HC1, HC2, HC3).
    
    Args:
        X: Design matrix (n x k)
        residuals: Residuals (n x 1)
        XpXinv: (X'X)^(-1) matrix (k x k)
        hc_type: Type of HC estimator:
                 0 = HC0 (White)
                 1 = HC1 (HC0 with small sample correction)
                 2 = HC2 (leverage adjusted)
                 3 = HC3 (jackknife)
        
    Returns:
        Tuple containing:
            - robust_cov: Robust covariance matrix (k x k)
            - robust_se: Robust standard errors (k x 1)
    """
    n, k = X.shape
    
    # Initialize matrices
    robust_cov = np.zeros((k, k))
    
    if hc_type == 0:  # HC0
        # White's heteroskedasticity-consistent estimator
        for i in range(n):
            xi = X[i, :]
            ei = residuals[i]
            for j in range(k):
                for l in range(k):
                    robust_cov[j, l] += xi[j] * xi[l] * ei * ei
    
    elif hc_type == 1:  # HC1
        # HC0 with small sample correction
        for i in range(n):
            xi = X[i, :]
            ei = residuals[i]
            for j in range(k):
                for l in range(k):
                    robust_cov[j, l] += xi[j] * xi[l] * ei * ei
        
        # Apply small sample correction
        robust_cov *= n / (n - k)
    
    elif hc_type == 2:  # HC2
        # Leverage adjusted
        h_diag = np.zeros(n)
        
        # Compute diagonal elements of hat matrix H = X(X'X)^(-1)X'
        for i in range(n):
            xi = X[i, :]
            h_diag[i] = 0.0
            for j in range(k):
                for l in range(k):
                    h_diag[i] += xi[j] * XpXinv[j, l] * xi[l]
        
        # Compute HC2 estimator
        for i in range(n):
            xi = X[i, :]
            ei = residuals[i]
            hi = h_diag[i]
            
            # Avoid division by zero
            if hi >= 1.0:
                hi = 0.99
            
            weight = 1.0 / (1.0 - hi)
            
            for j in range(k):
                for l in range(k):
                    robust_cov[j, l] += xi[j] * xi[l] * ei * ei * weight
    
    elif hc_type == 3:  # HC3
        # Jackknife estimator
        h_diag = np.zeros(n)
        
        # Compute diagonal elements of hat matrix H = X(X'X)^(-1)X'
        for i in range(n):
            xi = X[i, :]
            h_diag[i] = 0.0
            for j in range(k):
                for l in range(k):
                    h_diag[i] += xi[j] * XpXinv[j, l] * xi[l]
        
        # Compute HC3 estimator
        for i in range(n):
            xi = X[i, :]
            ei = residuals[i]
            hi = h_diag[i]
            
            # Avoid division by zero
            if hi >= 1.0:
                hi = 0.99
            
            weight = 1.0 / ((1.0 - hi) * (1.0 - hi))
            
            for j in range(k):
                for l in range(k):
                    robust_cov[j, l] += xi[j] * xi[l] * ei * ei * weight
    
    # Compute robust covariance matrix
    robust_cov = XpXinv @ robust_cov @ XpXinv
    
    # Compute robust standard errors
    robust_se = np.sqrt(np.diag(robust_cov))
    
    return robust_cov, robust_se


@jit(nopython=True, cache=True)
def _ols_newey_west_core(X: np.ndarray, residuals: np.ndarray, XpXinv: np.ndarray, 
                         lags: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba-accelerated core implementation of Newey-West HAC standard errors.
    
    This function computes Newey-West heteroskedasticity and autocorrelation
    consistent (HAC) standard errors for OLS regression.
    
    Args:
        X: Design matrix (n x k)
        residuals: Residuals (n x 1)
        XpXinv: (X'X)^(-1) matrix (k x k)
        lags: Number of lags to include
        
    Returns:
        Tuple containing:
            - nw_cov: Newey-West covariance matrix (k x k)
            - nw_se: Newey-West standard errors (k x 1)
    """
    n, k = X.shape
    
    # Compute X*u (score matrix)
    Xu = np.zeros((n, k))
    for i in range(n):
        for j in range(k):
            Xu[i, j] = X[i, j] * residuals[i]
    
    # Compute S0 (variance component)
    S0 = np.zeros((k, k))
    for i in range(n):
        for j in range(k):
            for l in range(k):
                S0[j, l] += Xu[i, j] * Xu[i, l]
    
    # Add autocorrelation components
    for lag in range(1, lags + 1):
        # Bartlett kernel weight
        w = 1.0 - lag / (lags + 1.0)
        
        # Compute autocovariance for this lag
        Sl = np.zeros((k, k))
        for t in range(lag, n):
            for j in range(k):
                for l in range(k):
                    Sl[j, l] += Xu[t, j] * Xu[t-lag, l] + Xu[t-lag, j] * Xu[t, l]
        
        # Add weighted autocovariance to S0
        for j in range(k):
            for l in range(k):
                S0[j, l] += w * Sl[j, l]
    
    # Scale by sample size
    S0 /= n
    
    # Compute Newey-West covariance matrix
    nw_cov = XpXinv @ S0 @ XpXinv
    
    # Compute Newey-West standard errors
    nw_se = np.sqrt(np.diag(nw_cov))
    
    return nw_cov, nw_se


@jit(nopython=True, cache=True)
def _ols_diagnostic_core(X: np.ndarray, residuals: np.ndarray, 
                         beta: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Numba-accelerated core implementation of OLS diagnostic statistics.
    
    This function computes various diagnostic statistics for OLS regression,
    including condition number, Durbin-Watson statistic, and components for
    heteroskedasticity tests.
    
    Args:
        X: Design matrix (n x k)
        residuals: Residuals (n x 1)
        beta: Parameter estimates (k x 1)
        
    Returns:
        Tuple containing:
            - condition_number: Condition number of X'X
            - durbin_watson: Durbin-Watson statistic
            - bp_stat: Breusch-Pagan test statistic
            - bp_p: Breusch-Pagan p-value
    """
    n, k = X.shape
    
    # Compute condition number using SVD
    u, s, vh = np.linalg.svd(X)
    condition_number = s[0] / s[-1]
    
    # Compute Durbin-Watson statistic
    dw_num = 0.0
    for i in range(1, n):
        dw_num += (residuals[i] - residuals[i-1])**2
    
    dw_den = np.sum(residuals**2)
    durbin_watson = dw_num / dw_den if dw_den > 0 else np.nan
    
    # Compute Breusch-Pagan test
    # Regress squared residuals on X
    e_squared = residuals**2
    e_squared_mean = np.mean(e_squared)
    
    # Normalize squared residuals
    z = e_squared / e_squared_mean
    
    # Compute explained sum of squares
    z_hat = X @ np.linalg.inv(X.T @ X) @ X.T @ z
    ess = np.sum((z_hat - np.mean(z))**2)
    
    # BP test statistic is 0.5 * ESS
    bp_stat = 0.5 * ess
    
    # p-value from chi-squared distribution with k-1 degrees of freedom
    # Note: We can't use scipy.stats in numba, so we return the statistic
    # and compute the p-value outside this function
    bp_p = 0.0  # Placeholder, will be computed outside
    
    return condition_number, durbin_watson, bp_stat, k - 1  # Return df instead of p-value


@jit(nopython=True, cache=True)
def _pca_outer_product_core(data: np.ndarray, 
                           center: bool, scale: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-accelerated core implementation of PCA using the outer product method.
    
    This function computes PCA using the outer product method (X'X), which is
    more efficient when n_samples > n_features.
    
    Args:
        data: Data matrix (n_samples x n_features)
        center: Whether to center the data (subtract mean)
        scale: Whether to scale the data (divide by std. dev.)
        
    Returns:
        Tuple containing:
            - eigenvalues: Eigenvalues of the decomposition
            - eigenvectors: Eigenvectors (loadings) of the decomposition
            - components: Principal components (scores)
            - explained_variance_ratio: Proportion of variance explained by each component
    """
    n_samples, n_features = data.shape
    
    # Preprocess data
    processed_data = data.copy()
    mean = np.zeros(n_features)
    std = np.ones(n_features)
    
    if center:
        # Compute mean
        for j in range(n_features):
            mean[j] = 0.0
            for i in range(n_samples):
                mean[j] += data[i, j]
            mean[j] /= n_samples
        
        # Center data
        for i in range(n_samples):
            for j in range(n_features):
                processed_data[i, j] -= mean[j]
    
    if scale:
        # Compute standard deviation
        for j in range(n_features):
            std[j] = 0.0
            for i in range(n_samples):
                std[j] += (processed_data[i, j] ** 2)
            std[j] = np.sqrt(std[j] / (n_samples - 1))
            
            # Avoid division by zero
            if std[j] < 1e-10:
                std[j] = 1.0
        
        # Scale data
        for i in range(n_samples):
            for j in range(n_features):
                processed_data[i, j] /= std[j]
    
    # Compute outer product (X'X)
    outer_product = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(n_features):
            for k in range(n_samples):
                outer_product[i, j] += processed_data[k, i] * processed_data[k, j]
    
    # Ensure symmetry (for numerical stability)
    for i in range(n_features):
        for j in range(i+1, n_features):
            avg = (outer_product[i, j] + outer_product[j, i]) / 2
            outer_product[i, j] = avg
            outer_product[j, i] = avg
    
    # Compute eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(outer_product)
    
    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Scale eigenvalues by 1/n_samples to get variance
    eigenvalues = eigenvalues / n_samples
    
    # Compute principal components (scores)
    components = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        for j in range(n_features):
            for k in range(n_features):
                components[i, j] += processed_data[i, k] * eigenvectors[k, j]
    
    # Compute explained variance ratio
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance if total_variance > 0 else np.zeros_like(eigenvalues)
    
    return eigenvalues, eigenvectors, components, explained_variance_ratio


@jit(nopython=True, cache=True)
def _pca_transform_core(data: np.ndarray, mean: np.ndarray, std: Optional[np.ndarray], 
                       eigenvectors: np.ndarray, center: bool, scale: bool) -> np.ndarray:
    """
    Numba-accelerated core implementation of PCA data transformation.
    
    This function transforms data using a fitted PCA model, projecting it onto
    the principal component space.
    
    Args:
        data: Data matrix (n_samples x n_features)
        mean: Mean of the original data (n_features)
        std: Standard deviation of the original data (n_features), or None if not scaled
        eigenvectors: Eigenvectors (loadings) of the PCA model (n_features x n_components)
        center: Whether to center the data (subtract mean)
        scale: Whether to scale the data (divide by std. dev.)
        
    Returns:
        Transformed data (n_samples x n_components)
    """
    n_samples, n_features = data.shape
    n_components = eigenvectors.shape[1]
    
    # Preprocess data
    processed_data = data.copy()
    
    if center:
        # Center data
        for i in range(n_samples):
            for j in range(n_features):
                processed_data[i, j] -= mean[j]
    
    if scale and std is not None:
        # Scale data
        for i in range(n_samples):
            for j in range(n_features):
                # Avoid division by zero
                if std[j] > 1e-10:
                    processed_data[i, j] /= std[j]
    
    # Transform data
    transformed_data = np.zeros((n_samples, n_components))
    for i in range(n_samples):
        for j in range(n_components):
            for k in range(n_features):
                transformed_data[i, j] += processed_data[i, k] * eigenvectors[k, j]
    
    return transformed_data


@jit(nopython=True, cache=True)
def _pca_inverse_transform_core(components: np.ndarray, mean: np.ndarray, std: Optional[np.ndarray], 
                               eigenvectors: np.ndarray, center: bool, scale: bool) -> np.ndarray:
    """
    Numba-accelerated core implementation of PCA inverse transformation.
    
    This function performs the inverse transformation of PCA, projecting data
    from the principal component space back to the original feature space.
    
    Args:
        components: Principal components (n_samples x n_components)
        mean: Mean of the original data (n_features)
        std: Standard deviation of the original data (n_features), or None if not scaled
        eigenvectors: Eigenvectors (loadings) of the PCA model (n_features x n_components)
        center: Whether to center the data (subtract mean)
        scale: Whether to scale the data (divide by std. dev.)
        
    Returns:
        Reconstructed data in original space (n_samples x n_features)
    """
    n_samples, n_components = components.shape
    n_features = eigenvectors.shape[0]
    
    # Inverse transform
    reconstructed_data = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        for j in range(n_features):
            for k in range(n_components):
                reconstructed_data[i, j] += components[i, k] * eigenvectors[j, k]
    
    # Undo preprocessing
    if scale and std is not None:
        # Unscale data
        for i in range(n_samples):
            for j in range(n_features):
                reconstructed_data[i, j] *= std[j]
    
    if center:
        # Uncenter data
        for i in range(n_samples):
            for j in range(n_features):
                reconstructed_data[i, j] += mean[j]
    
    return reconstructed_data


@jit(nopython=True, cache=True)
def _compute_vif_core(X: np.ndarray) -> np.ndarray:
    """
    Numba-accelerated core implementation of Variance Inflation Factor computation.
    
    This function computes the Variance Inflation Factor (VIF) for each variable
    in a design matrix, which is used to detect multicollinearity.
    
    Args:
        X: Design matrix (n x k)
        
    Returns:
        VIF values for each variable (k x 1)
    """
    n, k = X.shape
    vif = np.ones(k)
    
    # Skip if only one variable (plus constant)
    if k <= 1:
        return vif
    
    # Compute VIF for each variable
    for i in range(k):
        # Create X matrix without the current variable
        X_without_i = np.zeros((n, k-1))
        col_idx = 0
        for j in range(k):
            if j != i:
                X_without_i[:, col_idx] = X[:, j]
                col_idx += 1
        
        # Compute X'X and its inverse
        XpX = X_without_i.T @ X_without_i
        XpXinv = np.linalg.inv(XpX)
        
        # Compute X'y where y is the current variable
        Xpy = X_without_i.T @ X[:, i]
        
        # Compute beta = (X'X)^(-1)X'y
        beta = XpXinv @ Xpy
        
        # Compute fitted values
        fitted = X_without_i @ beta
        
        # Compute residuals
        residuals = X[:, i] - fitted
        
        # Compute R-squared
        tss = np.sum((X[:, i] - np.mean(X[:, i]))**2)
        rss = np.sum(residuals**2)
        r_squared = 1.0 - (rss / tss) if tss > 0 else 0.0
        
        # Compute VIF = 1 / (1 - R^2)
        vif[i] = 1.0 / (1.0 - r_squared) if r_squared < 1.0 else float('inf')
    
    return vif


# Register Numba-accelerated functions if available
def _register_numba_functions() -> None:
    """
    Register Numba JIT-compiled functions for cross-sectional analysis.
    
    This function is called during module initialization to register
    performance-critical functions for JIT compilation if Numba is available.
    """
    if HAS_NUMBA:
        # The functions are already decorated with @jit, so no additional
        # registration is needed here. This function is kept for consistency
        # with the module structure and potential future enhancements.
        logger.debug("Cross-sectional analysis Numba JIT functions registered")
    else:
        logger.info("Numba not available. Cross-sectional analysis will use pure NumPy implementations.")


# Initialize the module
_register_numba_functions()
