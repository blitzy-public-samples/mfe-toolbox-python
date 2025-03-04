# mfe/models/multivariate/_numba_core.py

"""
Numba-accelerated core functions for multivariate volatility models.

This module provides JIT-compiled implementations of performance-critical operations
for multivariate volatility models in the MFE Toolbox. These functions replace the
previous MEX implementations with platform-independent optimized Python code,
providing significant performance improvements for likelihood evaluations and
recursive computations.

The module leverages Numba's just-in-time compilation capabilities to generate
efficient machine code at runtime, achieving near-native performance while
maintaining the flexibility and maintainability of Python code. All functions
include explicit type signatures to maximize Numba compilation efficiency.

Functions in this module are not typically called directly by users but are
instead used internally by the multivariate volatility model classes.
"""

import logging
import warnings
from typing import Optional, Tuple, Union

import numpy as np
from numba import jit, float64, int64, boolean, void

from mfe.core.exceptions import NumericError, warn_numeric

# Set up module-level logger
logger = logging.getLogger("mfe.models.multivariate._numba_core")

# Check if Numba is available with full functionality
try:
    # Test Numba's functionality with a simple function
    @jit(nopython=True)
    def _test_numba(x):
        return x + 1

    _test_numba(1)  # This will trigger compilation
    HAS_NUMBA = True
    logger.debug("Numba available for multivariate volatility acceleration")
except Exception as e:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator

    HAS_NUMBA = False
    logger.warning(f"Numba not available or failed: {str(e)}. Using pure NumPy implementations.")


@jit(float64[:, :, :](float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64[:, :, :], int64),
     nopython=True, cache=True)
def bekk_recursion(data: np.ndarray, C: np.ndarray, A: np.ndarray, B: np.ndarray,
                   H_t: np.ndarray, T: int) -> np.ndarray:
    """
    Numba-accelerated BEKK GARCH recursion.

    This function computes the conditional covariance matrices for the BEKK GARCH model
    using the recursion:
    H_t = C*C' + A*ε_{t-1}*ε_{t-1}'*A' + B*H_{t-1}*B'

    Args:
        data: Residual data matrix (T x n_assets)
        C: Constant matrix (n_assets x n_assets)
        A: ARCH coefficient matrix (n_assets x n_assets)
        B: GARCH coefficient matrix (n_assets x n_assets)
        H_t: Pre-allocated array for conditional covariances (n_assets x n_assets x T)
        T: Number of time periods

    Returns:
        Conditional covariance matrices (n_assets x n_assets x T)
    """
    n_assets = data.shape[1]

    # Initialize with unconditional covariance (backcast)
    # H_0 is already set in H_t[0] from the input

    # Compute C*C' once
    CC = np.zeros((n_assets, n_assets), dtype=np.float64)
    for i in range(n_assets):
        for j in range(n_assets):
            for k in range(n_assets):
                CC[i, j] += C[i, k] * C[j, k]

    # Main recursion
    for t in range(1, T):
        # Compute ε_{t-1}*ε_{t-1}'
        eps_eps = np.zeros((n_assets, n_assets), dtype=np.float64)
        for i in range(n_assets):
            for j in range(n_assets):
                eps_eps[i, j] = data[t-1, i] * data[t-1, j]

        # Compute A*ε_{t-1}*ε_{t-1}'*A'
        A_eps_eps_A = np.zeros((n_assets, n_assets), dtype=np.float64)
        for i in range(n_assets):
            for j in range(n_assets):
                for k in range(n_assets):
                    for l in range(n_assets):
                        A_eps_eps_A[i, j] += A[i, k] * eps_eps[k, l] * A[j, l]

        # Compute B*H_{t-1}*B'
        B_H_B = np.zeros((n_assets, n_assets), dtype=np.float64)
        for i in range(n_assets):
            for j in range(n_assets):
                for k in range(n_assets):
                    for l in range(n_assets):
                        B_H_B[i, j] += B[i, k] * H_t[:, :, t-1][k, l] * B[j, l]

        # Compute H_t = C*C' + A*ε_{t-1}*ε_{t-1}'*A' + B*H_{t-1}*B'
        for i in range(n_assets):
            for j in range(n_assets):
                H_t[i, j, t] = CC[i, j] + A_eps_eps_A[i, j] + B_H_B[i, j]

    return H_t


@jit(float64[:, :, :](float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64[:, :, :], int64),
     nopython=True, cache=True)
def bekk_asymmetric_recursion(data: np.ndarray, C: np.ndarray, A: np.ndarray, B: np.ndarray,
                              G: np.ndarray, H_t: np.ndarray, T: int) -> np.ndarray:
    """
    Numba-accelerated asymmetric BEKK GARCH recursion.

    This function computes the conditional covariance matrices for the asymmetric BEKK GARCH model
    using the recursion:
    H_t = C*C' + A*ε_{t-1}*ε_{t-1}'*A' + B*H_{t-1}*B' + G*η_{t-1}*η_{t-1}'*G'
    where η_{t-1} = ε_{t-1} * I(ε_{t-1} < 0) (element-wise product with indicator function)

    Args:
        data: Residual data matrix (T x n_assets)
        C: Constant matrix (n_assets x n_assets)
        A: ARCH coefficient matrix (n_assets x n_assets)
        B: GARCH coefficient matrix (n_assets x n_assets)
        G: Asymmetry coefficient matrix (n_assets x n_assets)
        H_t: Pre-allocated array for conditional covariances (n_assets x n_assets x T)
        T: Number of time periods

    Returns:
        Conditional covariance matrices (n_assets x n_assets x T)
    """
    n_assets = data.shape[1]

    # Initialize with unconditional covariance (backcast)
    # H_0 is already set in H_t[0] from the input

    # Compute C*C' once
    CC = np.zeros((n_assets, n_assets), dtype=np.float64)
    for i in range(n_assets):
        for j in range(n_assets):
            for k in range(n_assets):
                CC[i, j] += C[i, k] * C[j, k]

    # Main recursion
    for t in range(1, T):
        # Compute ε_{t-1}*ε_{t-1}'
        eps_eps = np.zeros((n_assets, n_assets), dtype=np.float64)
        for i in range(n_assets):
            for j in range(n_assets):
                eps_eps[i, j] = data[t-1, i] * data[t-1, j]

        # Compute η_{t-1} = ε_{t-1} * I(ε_{t-1} < 0)
        eta = np.zeros(n_assets, dtype=np.float64)
        for i in range(n_assets):
            if data[t-1, i] < 0:
                eta[i] = data[t-1, i]

        # Compute η_{t-1}*η_{t-1}'
        eta_eta = np.zeros((n_assets, n_assets), dtype=np.float64)
        for i in range(n_assets):
            for j in range(n_assets):
                eta_eta[i, j] = eta[i] * eta[j]

        # Compute A*ε_{t-1}*ε_{t-1}'*A'
        A_eps_eps_A = np.zeros((n_assets, n_assets), dtype=np.float64)
        for i in range(n_assets):
            for j in range(n_assets):
                for k in range(n_assets):
                    for l in range(n_assets):
                        A_eps_eps_A[i, j] += A[i, k] * eps_eps[k, l] * A[j, l]

        # Compute B*H_{t-1}*B'
        B_H_B = np.zeros((n_assets, n_assets), dtype=np.float64)
        for i in range(n_assets):
            for j in range(n_assets):
                for k in range(n_assets):
                    for l in range(n_assets):
                        B_H_B[i, j] += B[i, k] * H_t[:, :, t-1][k, l] * B[j, l]

        # Compute G*η_{t-1}*η_{t-1}'*G'
        G_eta_eta_G = np.zeros((n_assets, n_assets), dtype=np.float64)
        for i in range(n_assets):
            for j in range(n_assets):
                for k in range(n_assets):
                    for l in range(n_assets):
                        G_eta_eta_G[i, j] += G[i, k] * eta_eta[k, l] * G[j, l]

        # Compute H_t = C*C' + A*ε_{t-1}*ε_{t-1}'*A' + B*H_{t-1}*B' + G*η_{t-1}*η_{t-1}'*G'
        for i in range(n_assets):
            for j in range(n_assets):
                H_t[i, j, t] = CC[i, j] + A_eps_eps_A[i, j] + B_H_B[i, j] + G_eta_eta_G[i, j]

    return H_t


@jit(float64[:, :, :](float64[:, :], float64[:], float64[:], float64[:, :], float64[:, :, :], int64),
     nopython=True, cache=True)
def dcc_recursion(std_residuals: np.ndarray, a: np.ndarray, b: np.ndarray,
                  R_bar: np.ndarray, R_t: np.ndarray, T: int) -> np.ndarray:
    """
    Numba-accelerated DCC GARCH recursion.

    This function computes the conditional correlation matrices for the DCC GARCH model
    using the recursion:
    Q_t = (1 - sum(a) - sum(b))*R_bar + sum(a_i*z_{t-i}*z_{t-i}') + sum(b_j*Q_{t-j})
    R_t = diag(Q_t)^(-1/2) * Q_t * diag(Q_t)^(-1/2)

    Args:
        std_residuals: Standardized residual data matrix (T x n_assets)
        a: DCC ARCH coefficient vector (typically scalar or vector of length 1)
        b: DCC GARCH coefficient vector (typically scalar or vector of length 1)
        R_bar: Unconditional correlation matrix (n_assets x n_assets)
        R_t: Pre-allocated array for conditional correlations (n_assets x n_assets x T)
        T: Number of time periods

    Returns:
        Conditional correlation matrices (n_assets x n_assets x T)
    """
    n_assets = std_residuals.shape[1]

    # Initialize Q_t with R_bar
    Q_t = np.zeros((n_assets, n_assets), dtype=np.float64)
    for i in range(n_assets):
        for j in range(n_assets):
            Q_t[i, j] = R_bar[i, j]

    # Compute (1 - sum(a) - sum(b))*R_bar once
    sum_a = 0.0
    for i in range(len(a)):
        sum_a += a[i]

    sum_b = 0.0
    for i in range(len(b)):
        sum_b += b[i]

    scale = 1.0 - sum_a - sum_b
    scaled_R_bar = np.zeros((n_assets, n_assets), dtype=np.float64)
    for i in range(n_assets):
        for j in range(n_assets):
            scaled_R_bar[i, j] = scale * R_bar[i, j]

    # Set initial correlation matrix
    for i in range(n_assets):
        for j in range(n_assets):
            R_t[i, j, 0] = R_bar[i, j]

    # Main recursion
    for t in range(1, T):
        # Compute z_{t-1}*z_{t-1}'
        z_z = np.zeros((n_assets, n_assets), dtype=np.float64)
        for i in range(n_assets):
            for j in range(n_assets):
                z_z[i, j] = std_residuals[t-1, i] * std_residuals[t-1, j]

        # Compute Q_t = (1-sum(a)-sum(b))*R_bar + a*z_{t-1}*z_{t-1}' + b*Q_{t-1}
        for i in range(n_assets):
            for j in range(n_assets):
                Q_t[i, j] = scaled_R_bar[i, j] + a[0] * z_z[i, j] + b[0] * Q_t[i, j]

        # Compute diag(Q_t)^(-1/2)
        Q_diag_inv_sqrt = np.zeros(n_assets, dtype=np.float64)
        for i in range(n_assets):
            if Q_t[i, i] > 0:
                Q_diag_inv_sqrt[i] = 1.0 / np.sqrt(Q_t[i, i])
            else:
                # Handle numerical issues
                Q_diag_inv_sqrt[i] = 1.0 / np.sqrt(1e-8)

        # Compute R_t = diag(Q_t)^(-1/2) * Q_t * diag(Q_t)^(-1/2)
        for i in range(n_assets):
            for j in range(n_assets):
                R_t[i, j, t] = Q_t[i, j] * Q_diag_inv_sqrt[i] * Q_diag_inv_sqrt[j]

                # Ensure correlation bounds
                if i != j:
                    if R_t[i, j, t] > 0.9999:
                        R_t[i, j, t] = 0.9999
                    elif R_t[i, j, t] < -0.9999:
                        R_t[i, j, t] = -0.9999
                else:
                    R_t[i, j, t] = 1.0

    return R_t


@jit(float64[:, :, :](float64[:, :], float64[:], float64[:], float64[:], float64[:, :], float64[:, :, :], int64),
     nopython=True, cache=True)
def adcc_recursion(std_residuals: np.ndarray, a: np.ndarray, b: np.ndarray, g: np.ndarray,
                   R_bar: np.ndarray, R_t: np.ndarray, T: int) -> np.ndarray:
    """
    Numba-accelerated Asymmetric DCC GARCH recursion.

    This function computes the conditional correlation matrices for the Asymmetric DCC GARCH model
    using the recursion:
    Q_t = (1-sum(a)-sum(b)-kappa*sum(g))*R_bar + sum(a_i*z_{t-i}*z_{t-i}') + sum(b_j*Q_{t-j}) + sum(g_k*n_{t-k}*n_{t-k}')
    R_t = diag(Q_t)^(-1/2) * Q_t * diag(Q_t)^(-1/2)
    where n_t = I(z_t < 0) ⊙ z_t (element-wise product with indicator function)

    Args:
        std_residuals: Standardized residual data matrix (T x n_assets)
        a: DCC ARCH coefficient vector (typically scalar or vector of length 1)
        b: DCC GARCH coefficient vector (typically scalar or vector of length 1)
        g: DCC asymmetry coefficient vector (typically scalar or vector of length 1)
        R_bar: Unconditional correlation matrix (n_assets x n_assets)
        R_t: Pre-allocated array for conditional correlations (n_assets x n_assets x T)
        T: Number of time periods

    Returns:
        Conditional correlation matrices (n_assets x n_assets x T)
    """
    n_assets = std_residuals.shape[1]

    # Initialize Q_t with R_bar
    Q_t = np.zeros((n_assets, n_assets), dtype=np.float64)
    for i in range(n_assets):
        for j in range(n_assets):
            Q_t[i, j] = R_bar[i, j]

    # Compute E[n_t*n_t'] for scaling
    n_bar = np.zeros((n_assets, n_assets), dtype=np.float64)
    count = 0
    for t in range(T):
        n_t = np.zeros(n_assets, dtype=np.float64)
        for i in range(n_assets):
            if std_residuals[t, i] < 0:
                n_t[i] = std_residuals[t, i]

        for i in range(n_assets):
            for j in range(n_assets):
                n_bar[i, j] += n_t[i] * n_t[j]
        count += 1

    if count > 0:
        for i in range(n_assets):
            for j in range(n_assets):
                n_bar[i, j] /= count

    # Compute kappa = E[n_t*n_t']/R_bar
    kappa = np.zeros((n_assets, n_assets), dtype=np.float64)
    for i in range(n_assets):
        for j in range(n_assets):
            if R_bar[i, j] != 0:
                kappa[i, j] = n_bar[i, j] / R_bar[i, j]
            else:
                kappa[i, j] = 1.0

    # Use average kappa for simplicity
    kappa_avg = 0.0
    count = 0
    for i in range(n_assets):
        for j in range(n_assets):
            if i != j:  # Exclude diagonal elements
                kappa_avg += kappa[i, j]
                count += 1

    if count > 0:
        kappa_avg /= count
    else:
        kappa_avg = 0.25  # Default value if no off-diagonal elements

    # Compute (1 - sum(a) - sum(b) - kappa*sum(g))*R_bar once
    sum_a = 0.0
    for i in range(len(a)):
        sum_a += a[i]

    sum_b = 0.0
    for i in range(len(b)):
        sum_b += b[i]

    sum_g = 0.0
    for i in range(len(g)):
        sum_g += g[i]

    scale = 1.0 - sum_a - sum_b - kappa_avg * sum_g
    scaled_R_bar = np.zeros((n_assets, n_assets), dtype=np.float64)
    for i in range(n_assets):
        for j in range(n_assets):
            scaled_R_bar[i, j] = scale * R_bar[i, j]

    # Set initial correlation matrix
    for i in range(n_assets):
        for j in range(n_assets):
            R_t[i, j, 0] = R_bar[i, j]

    # Main recursion
    for t in range(1, T):
        # Compute z_{t-1}*z_{t-1}'
        z_z = np.zeros((n_assets, n_assets), dtype=np.float64)
        for i in range(n_assets):
            for j in range(n_assets):
                z_z[i, j] = std_residuals[t-1, i] * std_residuals[t-1, j]

        # Compute n_{t-1} = I(z_{t-1} < 0) ⊙ z_{t-1}
        n_t = np.zeros(n_assets, dtype=np.float64)
        for i in range(n_assets):
            if std_residuals[t-1, i] < 0:
                n_t[i] = std_residuals[t-1, i]

        # Compute n_{t-1}*n_{t-1}'
        n_n = np.zeros((n_assets, n_assets), dtype=np.float64)
        for i in range(n_assets):
            for j in range(n_assets):
                n_n[i, j] = n_t[i] * n_t[j]

        # Compute Q_t = (1-sum(a)-sum(b)-kappa*sum(g))*R_bar + a*z_{t-1}*z_{t-1}' + b*Q_{t-1} + g*n_{t-1}*n_{t-1}'
        for i in range(n_assets):
            for j in range(n_assets):
                Q_t[i, j] = scaled_R_bar[i, j] + a[0] * z_z[i, j] + b[0] * Q_t[i, j] + g[0] * n_n[i, j]

        # Compute diag(Q_t)^(-1/2)
        Q_diag_inv_sqrt = np.zeros(n_assets, dtype=np.float64)
        for i in range(n_assets):
            if Q_t[i, i] > 0:
                Q_diag_inv_sqrt[i] = 1.0 / np.sqrt(Q_t[i, i])
            else:
                # Handle numerical issues
                Q_diag_inv_sqrt[i] = 1.0 / np.sqrt(1e-8)

        # Compute R_t = diag(Q_t)^(-1/2) * Q_t * diag(Q_t)^(-1/2)
        for i in range(n_assets):
            for j in range(n_assets):
                R_t[i, j, t] = Q_t[i, j] * Q_diag_inv_sqrt[i] * Q_diag_inv_sqrt[j]

                # Ensure correlation bounds
                if i != j:
                    if R_t[i, j, t] > 0.9999:
                        R_t[i, j, t] = 0.9999
                    elif R_t[i, j, t] < -0.9999:
                        R_t[i, j, t] = -0.9999
                else:
                    R_t[i, j, t] = 1.0

    return R_t


@jit(float64[:, :, :](float64[:, :], float64, float64[:, :, :], int64),
     nopython=True, cache=True)
def ccc_recursion(std_residuals: np.ndarray, R: float, R_t: np.ndarray, T: int) -> np.ndarray:
    """
    Numba-accelerated CCC GARCH recursion.

    This function computes the conditional correlation matrices for the CCC GARCH model,
    which are constant over time.

    Args:
        std_residuals: Standardized residual data matrix (T x n_assets)
        R: Constant correlation value (for constant correlation model)
        R_t: Pre-allocated array for conditional correlations (n_assets x n_assets x T)
        T: Number of time periods

    Returns:
        Conditional correlation matrices (n_assets x n_assets x T)
    """
    n_assets = std_residuals.shape[1]

    # Set constant correlation matrix for all time periods
    for t in range(T):
        for i in range(n_assets):
            for j in range(n_assets):
                if i == j:
                    R_t[i, j, t] = 1.0
                else:
                    R_t[i, j, t] = R

    return R_t


@jit(float64[:, :, :](float64[:, :], float64, float64[:, :], float64[:, :, :], int64),
     nopython=True, cache=True)
def riskmetrics_recursion(data: np.ndarray, lambda_param: float,
                          H_0: np.ndarray, H_t: np.ndarray, T: int) -> np.ndarray:
    """
    Numba-accelerated RiskMetrics EWMA recursion.

    This function computes the conditional covariance matrices for the RiskMetrics model
    using the recursion:
    H_t = (1-lambda)*ε_{t-1}*ε_{t-1}' + lambda*H_{t-1}

    Args:
        data: Residual data matrix (T x n_assets)
        lambda_param: Smoothing parameter (typically 0.94 for daily data)
        H_0: Initial covariance matrix (n_assets x n_assets)
        H_t: Pre-allocated array for conditional covariances (n_assets x n_assets x T)
        T: Number of time periods

    Returns:
        Conditional covariance matrices (n_assets x n_assets x T)
    """
    n_assets = data.shape[1]

    # Set initial covariance matrix
    for i in range(n_assets):
        for j in range(n_assets):
            H_t[i, j, 0] = H_0[i, j]

    # Compute (1-lambda) once
    one_minus_lambda = 1.0 - lambda_param

    # Main recursion
    for t in range(1, T):
        # Compute ε_{t-1}*ε_{t-1}'
        eps_eps = np.zeros((n_assets, n_assets), dtype=np.float64)
        for i in range(n_assets):
            for j in range(n_assets):
                eps_eps[i, j] = data[t-1, i] * data[t-1, j]

        # Compute H_t = (1-lambda)*ε_{t-1}*ε_{t-1}' + lambda*H_{t-1}
        for i in range(n_assets):
            for j in range(n_assets):
                H_t[i, j, t] = one_minus_lambda * eps_eps[i, j] + lambda_param * H_t[i, j, t-1]

    return H_t


@jit(float64[:, :, :](float64[:, :], float64[:, :], float64[:], float64[:], float64[:, :, :], int64),
     nopython=True, cache=True)
def scalar_vt_vech_recursion(data: np.ndarray, C: np.ndarray, a: np.ndarray, b: np.ndarray,
                             H_t: np.ndarray, T: int) -> np.ndarray:
    """
    Numba-accelerated scalar VT-VECH GARCH recursion.

    This function computes the conditional covariance matrices for the scalar VT-VECH GARCH model
    using the recursion:
    vech(H_t) = vech(C) + a*vech(ε_{t-1}*ε_{t-1}') + b*vech(H_{t-1})

    Args:
        data: Residual data matrix (T x n_assets)
        C: Constant matrix (n_assets x n_assets)
        a: ARCH coefficient vector (scalar or vector of length 1)
        b: GARCH coefficient vector (scalar or vector of length 1)
        H_t: Pre-allocated array for conditional covariances (n_assets x n_assets x T)
        T: Number of time periods

    Returns:
        Conditional covariance matrices (n_assets x n_assets x T)
    """
    n_assets = data.shape[1]

    # Set initial covariance matrix
    # H_0 is already set in H_t[0] from the input

    # Main recursion
    for t in range(1, T):
        # Compute ε_{t-1}*ε_{t-1}'
        eps_eps = np.zeros((n_assets, n_assets), dtype=np.float64)
        for i in range(n_assets):
            for j in range(n_assets):
                eps_eps[i, j] = data[t-1, i] * data[t-1, j]

        # Compute H_t = C + a*ε_{t-1}*ε_{t-1}' + b*H_{t-1}
        for i in range(n_assets):
            for j in range(n_assets):
                H_t[i, j, t] = C[i, j] + a[0] * eps_eps[i, j] + b[0] * H_t[i, j, t-1]

    return H_t


@jit(float64[:, :, :](float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64[:, :, :], int64),
     nopython=True, cache=True)
def rarch_recursion(data: np.ndarray, C: np.ndarray, A: np.ndarray, B: np.ndarray,
                    H_t: np.ndarray, T: int) -> np.ndarray:
    """
    Numba-accelerated Rotated ARCH (RARCH) recursion.

    This function computes the conditional covariance matrices for the RARCH model
    using the recursion:
    H_t = C + A ⊙ (ε_{t-1}*ε_{t-1}') + B ⊙ H_{t-1}
    where ⊙ denotes the Hadamard (element-wise) product.

    Args:
        data: Residual data matrix (T x n_assets)
        C: Constant matrix (n_assets x n_assets)
        A: ARCH coefficient matrix (n_assets x n_assets)
        B: GARCH coefficient matrix (n_assets x n_assets)
        H_t: Pre-allocated array for conditional covariances (n_assets x n_assets x T)
        T: Number of time periods

    Returns:
        Conditional covariance matrices (n_assets x n_assets x T)
    """
    n_assets = data.shape[1]

    # Set initial covariance matrix
    # H_0 is already set in H_t[0] from the input

    # Main recursion
    for t in range(1, T):
        # Compute ε_{t-1}*ε_{t-1}'
        eps_eps = np.zeros((n_assets, n_assets), dtype=np.float64)
        for i in range(n_assets):
            for j in range(n_assets):
                eps_eps[i, j] = data[t-1, i] * data[t-1, j]

        # Compute H_t = C + A ⊙ (ε_{t-1}*ε_{t-1}') + B ⊙ H_{t-1}
        for i in range(n_assets):
            for j in range(n_assets):
                H_t[i, j, t] = C[i, j] + A[i, j] * eps_eps[i, j] + B[i, j] * H_t[i, j, t-1]

    return H_t


@jit(float64[:, :, :](float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64[:, :, :], int64),
     nopython=True, cache=True)
def rarch_asymmetric_recursion(data: np.ndarray, C: np.ndarray, A: np.ndarray, B: np.ndarray,
                               G: np.ndarray, H_t: np.ndarray, T: int) -> np.ndarray:
    """
    Numba-accelerated asymmetric Rotated ARCH (RARCH) recursion.

    This function computes the conditional covariance matrices for the asymmetric RARCH model
    using the recursion:
    H_t = C + A ⊙ (ε_{t-1}*ε_{t-1}') + B ⊙ H_{t-1} + G ⊙ (η_{t-1}*η_{t-1}')
    where ⊙ denotes the Hadamard (element-wise) product and
    η_{t-1} = ε_{t-1} * I(ε_{t-1} < 0) (element-wise product with indicator function)

    Args:
        data: Residual data matrix (T x n_assets)
        C: Constant matrix (n_assets x n_assets)
        A: ARCH coefficient matrix (n_assets x n_assets)
        B: GARCH coefficient matrix (n_assets x n_assets)
        G: Asymmetry coefficient matrix (n_assets x n_assets)
        H_t: Pre-allocated array for conditional covariances (n_assets x n_assets x T)
        T: Number of time periods

    Returns:
        Conditional covariance matrices (n_assets x n_assets x T)
    """
    n_assets = data.shape[1]

    # Set initial covariance matrix
    # H_0 is already set in H_t[0] from the input

    # Main recursion
    for t in range(1, T):
        # Compute ε_{t-1}*ε_{t-1}'
        eps_eps = np.zeros((n_assets, n_assets), dtype=np.float64)
        for i in range(n_assets):
            for j in range(n_assets):
                eps_eps[i, j] = data[t-1, i] * data[t-1, j]

        # Compute η_{t-1} = ε_{t-1} * I(ε_{t-1} < 0)
        eta = np.zeros(n_assets, dtype=np.float64)
        for i in range(n_assets):
            if data[t-1, i] < 0:
                eta[i] = data[t-1, i]

        # Compute η_{t-1}*η_{t-1}'
        eta_eta = np.zeros((n_assets, n_assets), dtype=np.float64)
        for i in range(n_assets):
            for j in range(n_assets):
                eta_eta[i, j] = eta[i] * eta[j]

        # Compute H_t = C + A ⊙ (ε_{t-1}*ε_{t-1}') + B ⊙ H_{t-1} + G ⊙ (η_{t-1}*η_{t-1}')
        for i in range(n_assets):
            for j in range(n_assets):
                H_t[i, j, t] = C[i, j] + A[i, j] * eps_eps[i, j] + B[i, j] * H_t[i, j, t-1] + G[i, j] * eta_eta[i, j]

    return H_t


@jit(float64[:, :, :](float64[:, :], float64[:, :], float64[:, :], float64[:, :, :], int64),
     nopython=True, cache=True)
def rcc_recursion(std_residuals: np.ndarray, R_bar: np.ndarray, P: np.ndarray,
                  R_t: np.ndarray, T: int) -> np.ndarray:
    """
    Numba-accelerated Rotated Conditional Correlation (RCC) recursion.

    This function computes the conditional correlation matrices for the RCC model
    using the recursion:
    R_t = R_bar ⊙ (ιι' - P) + P ⊙ (z_{t-1}*z_{t-1}')
    where ⊙ denotes the Hadamard (element-wise) product and ι is a vector of ones.

    Args:
        std_residuals: Standardized residual data matrix (T x n_assets)
        R_bar: Unconditional correlation matrix (n_assets x n_assets)
        P: Rotation parameter matrix (n_assets x n_assets)
        R_t: Pre-allocated array for conditional correlations (n_assets x n_assets x T)
        T: Number of time periods

    Returns:
        Conditional correlation matrices (n_assets x n_assets x T)
    """
    n_assets = std_residuals.shape[1]

    # Compute ιι' - P once
    ones_ones_minus_P = np.zeros((n_assets, n_assets), dtype=np.float64)
    for i in range(n_assets):
        for j in range(n_assets):
            ones_ones_minus_P[i, j] = 1.0 - P[i, j]

    # Set initial correlation matrix
    for i in range(n_assets):
        for j in range(n_assets):
            R_t[i, j, 0] = R_bar[i, j]

    # Main recursion
    for t in range(1, T):
        # Compute z_{t-1}*z_{t-1}'
        z_z = np.zeros((n_assets, n_assets), dtype=np.float64)
        for i in range(n_assets):
            for j in range(n_assets):
                z_z[i, j] = std_residuals[t-1, i] * std_residuals[t-1, j]

        # Compute R_t = R_bar ⊙ (ιι' - P) + P ⊙ (z_{t-1}*z_{t-1}')
        for i in range(n_assets):
            for j in range(n_assets):
                R_t[i, j, t] = R_bar[i, j] * ones_ones_minus_P[i, j] + P[i, j] * z_z[i, j]

                # Ensure correlation bounds
                if i != j:
                    if R_t[i, j, t] > 0.9999:
                        R_t[i, j, t] = 0.9999
                    elif R_t[i, j, t] < -0.9999:
                        R_t[i, j, t] = -0.9999
                else:
                    R_t[i, j, t] = 1.0

    return R_t


@jit(float64[:, :](float64[:, :], float64[:, :], float64[:, :]),
     nopython=True, cache=True)
def matrix_multiply(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Numba-accelerated matrix multiplication C = A @ B.

    This function performs matrix multiplication and stores the result in a pre-allocated array.

    Args:
        A: First matrix (m x n)
        B: Second matrix (n x p)
        C: Pre-allocated result matrix (m x p)

    Returns:
        Result of matrix multiplication (m x p)
    """
    m, n = A.shape
    p = B.shape[1]

    for i in range(m):
        for j in range(p):
            C[i, j] = 0.0
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]

    return C


@jit(float64[:, :](float64[:, :], float64[:, :], float64[:, :]),
     nopython=True, cache=True)
def matrix_multiply_transpose(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Numba-accelerated matrix multiplication C = A @ B.T.

    This function performs matrix multiplication with the transpose of the second matrix
    and stores the result in a pre-allocated array.

    Args:
        A: First matrix (m x n)
        B: Second matrix (p x n) to be transposed
        C: Pre-allocated result matrix (m x p)

    Returns:
        Result of matrix multiplication (m x p)
    """
    m, n = A.shape
    p = B.shape[0]

    for i in range(m):
        for j in range(p):
            C[i, j] = 0.0
            for k in range(n):
                C[i, j] += A[i, k] * B[j, k]

    return C


@jit(float64[:, :](float64[:, :], float64[:, :], float64[:, :]),
     nopython=True, cache=True)
def transpose_matrix_multiply(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Numba-accelerated matrix multiplication C = A.T @ B.

    This function performs matrix multiplication with the transpose of the first matrix
    and stores the result in a pre-allocated array.

    Args:
        A: First matrix (n x m) to be transposed
        B: Second matrix (n x p)
        C: Pre-allocated result matrix (m x p)

    Returns:
        Result of matrix multiplication (m x p)
    """
    n, m = A.shape
    p = B.shape[1]

    for i in range(m):
        for j in range(p):
            C[i, j] = 0.0
            for k in range(n):
                C[i, j] += A[k, i] * B[k, j]

    return C


@jit(float64[:, :](float64[:, :], float64[:, :], float64[:, :]),
     nopython=True, cache=True)
def hadamard_product(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Numba-accelerated Hadamard (element-wise) product C = A ⊙ B.

    This function performs element-wise multiplication and stores the result in a pre-allocated array.

    Args:
        A: First matrix (m x n)
        B: Second matrix (m x n)
        C: Pre-allocated result matrix (m x n)

    Returns:
        Result of element-wise multiplication (m x n)
    """
    m, n = A.shape

    for i in range(m):
        for j in range(n):
            C[i, j] = A[i, j] * B[i, j]

    return C


@jit(float64[:, :](float64[:, :], float64[:, :]),
     nopython=True, cache=True)
def ensure_positive_definite(H: np.ndarray, result: np.ndarray) -> np.ndarray:
    """
    Numba-accelerated function to ensure a matrix is positive definite.

    This function checks if a matrix is positive definite and, if not, adjusts it
    by adding a small value to the diagonal until it becomes positive definite.

    Args:
        H: Input matrix to check (n x n)
        result: Pre-allocated result matrix (n x n)

    Returns:
        Positive definite matrix (n x n)
    """
    n = H.shape[0]

    # Copy input matrix to result
    for i in range(n):
        for j in range(n):
            result[i, j] = H[i, j]

    # Check if matrix is symmetric
    is_symmetric = True
    for i in range(n):
        for j in range(i+1, n):
            if abs(result[i, j] - result[j, i]) > 1e-8:
                is_symmetric = False
                # Make symmetric
                avg = (result[i, j] + result[j, i]) / 2.0
                result[i, j] = avg
                result[j, i] = avg

    # Try to compute Cholesky decomposition
    L = np.zeros((n, n), dtype=np.float64)
    is_positive_definite = True

    for i in range(n):
        for j in range(i+1):
            s = 0.0
            for k in range(j):
                s += L[i, k] * L[j, k]

            if i == j:
                # Diagonal element
                val = result[i, i] - s
                if val <= 0.0:
                    is_positive_definite = False
                    break
                L[i, j] = np.sqrt(val)
            else:
                # Off-diagonal element
                if L[j, j] > 0:
                    L[i, j] = (result[i, j] - s) / L[j, j]
                else:
                    is_positive_definite = False
                    break

    # If not positive definite, adjust the matrix
    if not is_positive_definite:
        # Add a small value to the diagonal
        epsilon = 1e-6
        max_attempts = 100
        attempts = 0

        while not is_positive_definite and attempts < max_attempts:
            attempts += 1

            # Increase epsilon exponentially
            if attempts > 1:
                epsilon *= 10.0

            # Add epsilon to diagonal
            for i in range(n):
                result[i, i] += epsilon

            # Try Cholesky decomposition again
            is_positive_definite = True
            for i in range(n):
                for j in range(i+1):
                    s = 0.0
                    for k in range(j):
                        s += L[i, k] * L[j, k]

                    if i == j:
                        # Diagonal element
                        val = result[i, i] - s
                        if val <= 0.0:
                            is_positive_definite = False
                            break
                        L[i, j] = np.sqrt(val)
                    else:
                        # Off-diagonal element
                        if L[j, j] > 0:
                            L[i, j] = (result[i, j] - s) / L[j, j]
                        else:
                            is_positive_definite = False
                            break

                if not is_positive_definite:
                    break

    return result


@jit(float64[:, :](float64[:, :], float64[:, :]),
     nopython=True, cache=True)
def cov2corr(cov: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """
    Numba-accelerated conversion from covariance matrix to correlation matrix.

    This function converts a covariance matrix to a correlation matrix and
    stores the result in a pre-allocated array.

    Args:
        cov: Covariance matrix (n x n)
        corr: Pre-allocated correlation matrix (n x n)

    Returns:
        Correlation matrix (n x n)
    """
    n = cov.shape[0]

    # Extract standard deviations from diagonal
    std_devs = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if cov[i, i] > 0:
            std_devs[i] = np.sqrt(cov[i, i])
        else:
            # Handle numerical issues
            std_devs[i] = np.sqrt(1e-8)

    # Compute correlation matrix
    for i in range(n):
        for j in range(n):
            corr[i, j] = cov[i, j] / (std_devs[i] * std_devs[j])

            # Ensure correlation bounds
            if i != j:
                if corr[i, j] > 0.9999:
                    corr[i, j] = 0.9999
                elif corr[i, j] < -0.9999:
                    corr[i, j] = -0.9999
            else:
                corr[i, j] = 1.0

    return corr


@jit(float64[:, :](float64[:, :], float64[:], float64[:, :]),
     nopython=True, cache=True)
def corr2cov(corr: np.ndarray, std_devs: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    Numba-accelerated conversion from correlation matrix to covariance matrix.

    This function converts a correlation matrix to a covariance matrix using
    a vector of standard deviations and stores the result in a pre-allocated array.

    Args:
        corr: Correlation matrix (n x n)
        std_devs: Standard deviations (n)
        cov: Pre-allocated covariance matrix (n x n)

    Returns:
        Covariance matrix (n x n)
    """
    n = corr.shape[0]

    # Compute covariance matrix
    for i in range(n):
        for j in range(n):
            cov[i, j] = corr[i, j] * std_devs[i] * std_devs[j]

    return cov


def register_numba_functions() -> None:
    """
    Register Numba JIT-compiled functions for multivariate volatility models.

    This function is called during module initialization to register
    performance-critical functions for JIT compilation if Numba is available.
    """
    if HAS_NUMBA:
        # The functions are already decorated with @jit, so no additional
        # registration is needed here. This function is kept for consistency
        # with the module structure and potential future enhancements.
        logger.debug("Multivariate volatility Numba JIT functions registered")
    else:
        logger.warning("Numba not available. Multivariate volatility models will use pure NumPy implementations.")


# Initialize the module
register_numba_functions()
