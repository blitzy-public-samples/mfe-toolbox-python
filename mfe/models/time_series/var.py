'''
Vector Autoregression (VAR) Model Implementation

This module implements Vector Autoregression (VAR) models for multivariate time series
analysis, providing comprehensive functionality for model specification, estimation,
inference, and forecasting. The implementation supports both regular and structural
VAR models with various estimation methods, lag selection criteria, and diagnostic tools.

The module leverages NumPy for efficient matrix operations, SciPy for optimization and
statistical functions, Pandas for time series handling, and Statsmodels for core VAR
functionality. Performance-critical operations are accelerated using Numba's JIT
compilation for optimal performance.

Classes:
    VARParameters: Parameter container for VAR models
    VARResults: Results container for VAR model estimation
    VARModel: Base class for VAR models
    StructuralVARModel: Class for structural VAR models with identification restrictions

Functions:
    lag_matrix: Create matrix of lagged variables for VAR estimation
    information_criteria: Compute information criteria for lag selection
    granger_causality: Test for Granger causality between variables
'''

import asyncio
import logging
import warnings
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union, cast
)

import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats
import statsmodels.api as sm
from statsmodels.tsa.api import VAR as StatsmodelsVAR

from mfe.core.base import ModelBase
from mfe.core.parameters import (
    ParameterBase, TimeSeriesParameters, validate_positive, validate_non_negative,
    validate_range, transform_positive, inverse_transform_positive
)
from mfe.core.exceptions import (
    ParameterError, DimensionError, ConvergenceError, NumericError,
    EstimationError, ForecastError, SimulationError, NotFittedError,
    warn_convergence, warn_numeric, warn_model
)
from mfe.utils.matrix_ops import (
    vech, ivech, ensure_symmetric, is_positive_definite, nearest_positive_definite
)

# Set up module-level logger
logger = logging.getLogger("mfe.models.time_series.var")

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
    logger.debug("Numba available for VAR acceleration")
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    HAS_NUMBA = False
    logger.info("Numba not available. VAR will use pure NumPy implementations.")


@dataclass
class VARParameters(TimeSeriesParameters):
    """
    Parameters for Vector Autoregression (VAR) models.
    
    This class provides a container for VAR model parameters, including coefficient
    matrices for each lag and the constant term. It includes validation methods to
    ensure parameter constraints are satisfied.
    
    Attributes:
        coef_matrices: List of coefficient matrices for each lag [A₁, A₂, ..., Aₚ]
        constant: Constant term vector (intercept)
        sigma: Covariance matrix of residuals
    """
    
    coef_matrices: List[np.ndarray]
    constant: np.ndarray
    sigma: np.ndarray
    
    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        # Ensure arrays are NumPy arrays
        if not isinstance(self.coef_matrices, list):
            raise TypeError("coef_matrices must be a list of NumPy arrays")
        
        self.coef_matrices = [np.asarray(mat) for mat in self.coef_matrices]
        self.constant = np.asarray(self.constant)
        self.sigma = np.asarray(self.sigma)
        
        # Validate parameters
        self.validate()
    
    def validate(self) -> None:
        """
        Validate VAR parameter constraints.
        
        Raises:
            ParameterError: If parameter constraints are violated
            DimensionError: If parameter dimensions are inconsistent
        """
        super().validate()
        
        # Check if there are any coefficient matrices
        if not self.coef_matrices:
            raise ParameterError(
                "At least one coefficient matrix must be provided",
                param_name="coef_matrices",
                constraint="Non-empty list"
            )
        
        # Get dimensions from first coefficient matrix
        k = self.coef_matrices[0].shape[0]
        
        # Check dimensions of coefficient matrices
        for i, mat in enumerate(self.coef_matrices):
            if mat.ndim != 2 or mat.shape[0] != k or mat.shape[1] != k:
                raise DimensionError(
                    f"Coefficient matrix {i+1} has incorrect dimensions",
                    array_name=f"coef_matrices[{i}]",
                    expected_shape=f"({k}, {k})",
                    actual_shape=mat.shape
                )
        
        # Check dimensions of constant term
        if self.constant.ndim != 1 or self.constant.shape[0] != k:
            raise DimensionError(
                "Constant term has incorrect dimensions",
                array_name="constant",
                expected_shape=f"({k},)",
                actual_shape=self.constant.shape
            )
        
        # Check dimensions of sigma
        if self.sigma.ndim != 2 or self.sigma.shape[0] != k or self.sigma.shape[1] != k:
            raise DimensionError(
                "Sigma has incorrect dimensions",
                array_name="sigma",
                expected_shape=f"({k}, {k})",
                actual_shape=self.sigma.shape
            )
        
        # Check if sigma is symmetric and positive definite
        if not np.allclose(self.sigma, self.sigma.T):
            raise ParameterError(
                "Sigma must be symmetric",
                param_name="sigma",
                constraint="Symmetric matrix"
            )
        
        if not is_positive_definite(self.sigma):
            raise ParameterError(
                "Sigma must be positive definite",
                param_name="sigma",
                constraint="Positive definite matrix"
            )
        
        # Check stability of VAR model
        self._check_stability()
    
    def _check_stability(self) -> None:
        """
        Check stability of the VAR model.
        
        A VAR model is stable if all eigenvalues of the companion matrix have
        modulus less than 1.
        
        Raises:
            ParameterError: If the VAR model is not stable
        """
        # Get dimensions
        k = self.coef_matrices[0].shape[0]
        p = len(self.coef_matrices)
        
        # Construct companion matrix
        companion = np.zeros((k * p, k * p))
        
        # Fill first block row with coefficient matrices
        for i in range(p):
            companion[:k, i*k:(i+1)*k] = self.coef_matrices[i]
        
        # Fill lower block diagonal with identity matrices
        for i in range(1, p):
            companion[i*k:(i+1)*k, (i-1)*k:i*k] = np.eye(k)
        
        # Compute eigenvalues
        eigvals = np.linalg.eigvals(companion)
        max_modulus = np.max(np.abs(eigvals))
        
        # Check stability
        if max_modulus >= 1:
            warn_model(
                "VAR model may not be stable",
                model_type="VAR",
                issue="stability",
                value=max_modulus,
                details="Maximum eigenvalue modulus is greater than or equal to 1"
            )
    
    def to_array(self) -> np.ndarray:
        """
        Convert parameters to a NumPy array.
        
        Returns:
            np.ndarray: Array representation of parameters
        """
        # Flatten coefficient matrices
        coef_flat = np.concatenate([mat.flatten() for mat in self.coef_matrices])
        
        # Concatenate with constant and vech(sigma)
        return np.concatenate([coef_flat, self.constant, vech(self.sigma)])
    
    @classmethod
    def from_array(cls, array: np.ndarray, k: int, p: int) -> 'VARParameters':
        """
        Create parameters from a NumPy array.
        
        Args:
            array: Array representation of parameters
            k: Number of variables in the VAR model
            p: Number of lags in the VAR model
            
        Returns:
            VARParameters: Parameter object
            
        Raises:
            ValueError: If the array length doesn't match the expected length
        """
        # Calculate expected array length
        expected_length = k * k * p + k + k * (k + 1) // 2
        
        if len(array) != expected_length:
            raise ValueError(
                f"Array length ({len(array)}) doesn't match expected length ({expected_length})"
            )
        
        # Extract coefficient matrices
        coef_matrices = []
        idx = 0
        for _ in range(p):
            coef_mat = array[idx:idx + k*k].reshape(k, k)
            coef_matrices.append(coef_mat)
            idx += k * k
        
        # Extract constant
        constant = array[idx:idx + k]
        idx += k
        
        # Extract sigma
        sigma_vech = array[idx:]
        sigma = ivech(sigma_vech)
        
        return cls(coef_matrices=coef_matrices, constant=constant, sigma=sigma)
    
    def transform(self) -> np.ndarray:
        """
        Transform parameters to unconstrained space for optimization.
        
        Returns:
            np.ndarray: Transformed parameters in unconstrained space
        """
        # Coefficient matrices and constant don't need transformation
        coef_flat = np.concatenate([mat.flatten() for mat in self.coef_matrices])
        
        # Transform sigma to ensure positive definiteness
        # We use Cholesky decomposition
        L = np.linalg.cholesky(self.sigma)
        # Extract lower triangular elements
        L_vech = vech(L)
        
        # Concatenate all parameters
        return np.concatenate([coef_flat, self.constant, L_vech])
    
    @classmethod
    def inverse_transform(cls, array: np.ndarray, k: int, p: int) -> 'VARParameters':
        """
        Transform parameters from unconstrained space back to constrained space.
        
        Args:
            array: Parameters in unconstrained space
            k: Number of variables in the VAR model
            p: Number of lags in the VAR model
            
        Returns:
            VARParameters: Parameter object with constrained parameters
            
        Raises:
            ValueError: If the array length doesn't match the expected length
        """
        # Calculate expected array length
        expected_length = k * k * p + k + k * (k + 1) // 2
        
        if len(array) != expected_length:
            raise ValueError(
                f"Array length ({len(array)}) doesn't match expected length ({expected_length})"
            )
        
        # Extract coefficient matrices
        coef_matrices = []
        idx = 0
        for _ in range(p):
            coef_mat = array[idx:idx + k*k].reshape(k, k)
            coef_matrices.append(coef_mat)
            idx += k * k
        
        # Extract constant
        constant = array[idx:idx + k]
        idx += k
        
        # Extract Cholesky factor and reconstruct sigma
        L_vech = array[idx:]
        L = ivech(L_vech)
        sigma = L @ L.T
        
        return cls(coef_matrices=coef_matrices, constant=constant, sigma=sigma)


@dataclass
class VARResults:
    """
    Results container for VAR model estimation.
    
    This class provides a container for VAR model estimation results, including
    parameter estimates, standard errors, diagnostics, and fitted values.
    
    Attributes:
        model_name: Name of the model
        params: Estimated parameters
        residuals: Model residuals
        fitted_values: Fitted values from the model
        log_likelihood: Log-likelihood of the model
        aic: Akaike Information Criterion
        bic: Bayesian Information Criterion
        hqic: Hannan-Quinn Information Criterion
        fpe: Final Prediction Error
        coef_matrices: List of coefficient matrices for each lag
        constant: Constant term vector (intercept)
        sigma: Covariance matrix of residuals
        std_errors: Standard errors of parameter estimates
        t_stats: t-statistics for parameter estimates
        p_values: p-values for parameter estimates
        convergence: Whether the optimization converged
        iterations: Number of iterations performed during optimization
        cov_type: Type of covariance matrix used
        cov_params: Covariance matrix of parameter estimates
        nobs: Number of observations
        k: Number of variables
        p: Number of lags
        df_model: Degrees of freedom used by the model
        df_resid: Residual degrees of freedom
        eigenvalues: Eigenvalues of the companion matrix
        is_stable: Whether the VAR model is stable
    """
    
    model_name: str
    params: VARParameters
    residuals: np.ndarray
    fitted_values: np.ndarray
    log_likelihood: float
    aic: float
    bic: float
    hqic: float
    fpe: float
    coef_matrices: List[np.ndarray] = field(init=False)
    constant: np.ndarray = field(init=False)
    sigma: np.ndarray = field(init=False)
    std_errors: Dict[str, np.ndarray] = field(default_factory=dict)
    t_stats: Dict[str, np.ndarray] = field(default_factory=dict)
    p_values: Dict[str, np.ndarray] = field(default_factory=dict)
    convergence: bool = True
    iterations: int = 0
    cov_type: str = "robust"
    cov_params: Optional[np.ndarray] = None
    nobs: Optional[int] = None
    k: Optional[int] = None
    p: Optional[int] = None
    df_model: Optional[int] = None
    df_resid: Optional[int] = None
    eigenvalues: Optional[np.ndarray] = None
    is_stable: bool = True
    
    def __post_init__(self) -> None:
        """Initialize derived attributes after initialization."""
        # Extract parameters from params object
        self.coef_matrices = self.params.coef_matrices
        self.constant = self.params.constant
        self.sigma = self.params.sigma
        
        # Set dimensions if not provided
        if self.k is None:
            self.k = self.constant.shape[0]
        
        if self.p is None:
            self.p = len(self.coef_matrices)
        
        if self.nobs is None and self.residuals is not None:
            self.nobs = self.residuals.shape[0]
        
        if self.df_model is None and self.k is not None and self.p is not None:
            self.df_model = self.k * self.p + 1  # +1 for constant
        
        if self.df_resid is None and self.nobs is not None and self.df_model is not None:
            self.df_resid = self.nobs - self.df_model
        
        # Check stability if eigenvalues are provided
        if self.eigenvalues is not None:
            self.is_stable = np.max(np.abs(self.eigenvalues)) < 1
        else:
            # Compute eigenvalues of companion matrix
            self._compute_eigenvalues()
    
    def _compute_eigenvalues(self) -> None:
        """Compute eigenvalues of the companion matrix."""
        # Get dimensions
        k = self.k if self.k is not None else self.constant.shape[0]
        p = self.p if self.p is not None else len(self.coef_matrices)
        
        # Construct companion matrix
        companion = np.zeros((k * p, k * p))
        
        # Fill first block row with coefficient matrices
        for i in range(p):
            companion[:k, i*k:(i+1)*k] = self.coef_matrices[i]
        
        # Fill lower block diagonal with identity matrices
        for i in range(1, p):
            companion[i*k:(i+1)*k, (i-1)*k:i*k] = np.eye(k)
        
        # Compute eigenvalues
        self.eigenvalues = np.linalg.eigvals(companion)
        self.is_stable = np.max(np.abs(self.eigenvalues)) < 1
    
    def summary(self) -> str:
        """
        Generate a text summary of the VAR model results.
        
        Returns:
            str: A formatted string containing the model results summary.
        """
        k = self.k if self.k is not None else self.constant.shape[0]
        p = self.p if self.p is not None else len(self.coef_matrices)
        
        header = f"Model: {self.model_name}\n"
        header += "=" * (len(header) - 1) + "\n\n"
        
        # Add model information
        info = f"Number of variables (k): {k}\n"
        info += f"Number of lags (p): {p}\n"
        info += f"Number of observations: {self.nobs}\n"
        info += f"Log-likelihood: {self.log_likelihood:.6f}\n"
        info += f"AIC: {self.aic:.6f}\n"
        info += f"BIC: {self.bic:.6f}\n"
        info += f"HQIC: {self.hqic:.6f}\n"
        info += f"FPE: {self.fpe:.6f}\n"
        info += f"Stability: {'Yes' if self.is_stable else 'No'}\n\n"
        
        # Add coefficient tables for each equation
        coef_tables = ""
        for i in range(k):
            coef_tables += f"Equation {i+1}:\n"
            coef_tables += "-" * 80 + "\n"
            coef_tables += f"{'Parameter':<15} {'Estimate':>12} {'Std. Error':>12} "
            coef_tables += f"{'t-stat':>12} {'p-value':>12} {'Significance':>10}\n"
            coef_tables += "-" * 80 + "\n"
            
            # Add constant term
            const_val = self.constant[i]
            const_se = self.std_errors.get('constant', np.zeros(k))[i]
            const_t = self.t_stats.get('constant', np.zeros(k))[i]
            const_p = self.p_values.get('constant', np.zeros(k))[i]
            
            # Determine significance
            if const_p < 0.01:
                sig = "***"
            elif const_p < 0.05:
                sig = "**"
            elif const_p < 0.1:
                sig = "*"
            else:
                sig = ""
            
            coef_tables += f"{'constant':<15} {const_val:>12.6f} {const_se:>12.6f} "
            coef_tables += f"{const_t:>12.6f} {const_p:>12.6f} {sig:>10}\n"
            
            # Add coefficients for each lag and variable
            for lag in range(p):
                for j in range(k):
                    param_name = f"L{lag+1}.y{j+1}"
                    param_val = self.coef_matrices[lag][i, j]
                    param_se = self.std_errors.get(f'A{lag+1}', np.zeros((k, k)))[i, j]
                    param_t = self.t_stats.get(f'A{lag+1}', np.zeros((k, k)))[i, j]
                    param_p = self.p_values.get(f'A{lag+1}', np.zeros((k, k)))[i, j]
                    
                    # Determine significance
                    if param_p < 0.01:
                        sig = "***"
                    elif param_p < 0.05:
                        sig = "**"
                    elif param_p < 0.1:
                        sig = "*"
                    else:
                        sig = ""
                    
                    coef_tables += f"{param_name:<15} {param_val:>12.6f} {param_se:>12.6f} "
                    coef_tables += f"{param_t:>12.6f} {param_p:>12.6f} {sig:>10}\n"
            
            coef_tables += "-" * 80 + "\n\n"
        
        # Add residual covariance matrix
        cov_matrix = "Residual Covariance Matrix:\n"
        cov_matrix += "-" * 80 + "\n"
        for i in range(k):
            row = " ".join(f"{self.sigma[i, j]:>12.6f}" for j in range(k))
            cov_matrix += row + "\n"
        cov_matrix += "-" * 80 + "\n\n"
        
        # Add significance codes
        sig_codes = "Significance codes: *** 0.01, ** 0.05, * 0.1\n\n"
        
        return header + info + coef_tables + cov_matrix + sig_codes
    
    def to_pandas(self) -> Dict[str, pd.DataFrame]:
        """
        Convert results to Pandas DataFrames.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of DataFrames containing results
        """
        k = self.k if self.k is not None else self.constant.shape[0]
        p = self.p if self.p is not None else len(self.coef_matrices)
        
        # Create variable names
        var_names = [f"y{i+1}" for i in range(k)]
        
        # Create DataFrames for each result component
        results = {}
        
        # Coefficients DataFrame
        coef_index = ['const'] + [f"L{lag+1}.{var}" for lag in range(p) for var in var_names]
        coef_data = np.vstack([
            self.constant.reshape(1, -1),
            np.vstack([mat.T for mat in self.coef_matrices])
        ])
        results['coefficients'] = pd.DataFrame(
            coef_data, index=coef_index, columns=var_names
        )
        
        # Standard errors DataFrame
        if 'constant' in self.std_errors and all(f'A{i+1}' in self.std_errors for i in range(p)):
            se_data = np.vstack([
                self.std_errors['constant'].reshape(1, -1),
                np.vstack([self.std_errors[f'A{lag+1}'].T for lag in range(p)])
            ])
            results['std_errors'] = pd.DataFrame(
                se_data, index=coef_index, columns=var_names
            )
        
        # t-statistics DataFrame
        if 'constant' in self.t_stats and all(f'A{i+1}' in self.t_stats for i in range(p)):
            t_data = np.vstack([
                self.t_stats['constant'].reshape(1, -1),
                np.vstack([self.t_stats[f'A{lag+1}'].T for lag in range(p)])
            ])
            results['t_stats'] = pd.DataFrame(
                t_data, index=coef_index, columns=var_names
            )
        
        # p-values DataFrame
        if 'constant' in self.p_values and all(f'A{i+1}' in self.p_values for i in range(p)):
            p_data = np.vstack([
                self.p_values['constant'].reshape(1, -1),
                np.vstack([self.p_values[f'A{lag+1}'].T for lag in range(p)])
            ])
            results['p_values'] = pd.DataFrame(
                p_data, index=coef_index, columns=var_names
            )
        
        # Residual covariance matrix
        results['residual_cov'] = pd.DataFrame(
            self.sigma, index=var_names, columns=var_names
        )
        
        # Residuals
        if isinstance(self.residuals, np.ndarray) and self.residuals.ndim == 2:
            results['residuals'] = pd.DataFrame(
                self.residuals, columns=var_names
            )
        
        # Fitted values
        if isinstance(self.fitted_values, np.ndarray) and self.fitted_values.ndim == 2:
            results['fitted_values'] = pd.DataFrame(
                self.fitted_values, columns=var_names
            )
        
        # Information criteria
        results['info_criteria'] = pd.DataFrame({
            'AIC': [self.aic],
            'BIC': [self.bic],
            'HQIC': [self.hqic],
            'FPE': [self.fpe],
            'Log-likelihood': [self.log_likelihood]
        })
        
        return results


@jit(nopython=True, cache=True)
def _lag_matrix_numba(y: np.ndarray, lags: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba-accelerated implementation of lag matrix creation.
    
    Args:
        y: Time series data matrix (T x k)
        lags: Number of lags to include
        
    Returns:
        Tuple containing:
            - Lagged data matrix (T-lags x k*lags)
            - Trimmed original data (T-lags x k)
    """
    T, k = y.shape
    # Create lagged data matrix
    X = np.zeros((T - lags, k * lags))
    
    # Fill the lagged data matrix
    for t in range(lags, T):
        row_idx = t - lags
        for lag in range(1, lags + 1):
            X[row_idx, (lag-1)*k:lag*k] = y[t - lag]
    
    # Return lagged data matrix and trimmed original data
    return X, y[lags:]


def lag_matrix(y: Union[np.ndarray, pd.DataFrame], lags: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create matrix of lagged variables for VAR estimation.
    
    Args:
        y: Time series data matrix (T x k) or DataFrame
        lags: Number of lags to include
    
    Returns:
        Tuple containing:
            - Lagged data matrix (T-lags x k*lags)
            - Trimmed original data (T-lags x k)
            
    Raises:
        ValueError: If lags is not positive or if y has insufficient observations
    """
    # Convert to numpy array if DataFrame
    if isinstance(y, pd.DataFrame):
        y_array = y.values
    else:
        y_array = np.asarray(y)
    
    # Check inputs
    if lags <= 0:
        raise ValueError("Number of lags must be positive")
    
    if y_array.ndim != 2:
        raise ValueError("Input data must be a 2D array or DataFrame")
    
    T, k = y_array.shape
    
    if T <= lags:
        raise ValueError(f"Number of observations ({T}) must be greater than lags ({lags})")
    
    # Use Numba-accelerated implementation if available
    if HAS_NUMBA:
        return _lag_matrix_numba(y_array, lags)
    
    # Pure NumPy implementation
    # Create lagged data matrix
    X = np.zeros((T - lags, k * lags))
    
    # Fill the lagged data matrix
    for t in range(lags, T):
        row_idx = t - lags
        for lag in range(1, lags + 1):
            X[row_idx, (lag-1)*k:lag*k] = y_array[t - lag]
    
    # Return lagged data matrix and trimmed original data
    return X, y_array[lags:]


def information_criteria(
    log_likelihood: float, 
    nobs: int, 
    k: int, 
    p: int
) -> Tuple[float, float, float, float]:
    """
    Compute information criteria for VAR model selection.
    
    Args:
        log_likelihood: Log-likelihood of the model
        nobs: Number of observations
        k: Number of variables
        p: Number of lags
        
    Returns:
        Tuple containing AIC, BIC, HQIC, and FPE values
    """
    # Number of parameters: k*k*p (coefficients) + k (constant) + k*(k+1)/2 (covariance)
    n_params = k * k * p + k + k * (k + 1) // 2
    
    # Compute information criteria
    aic = -2 * log_likelihood + 2 * n_params
    bic = -2 * log_likelihood + n_params * np.log(nobs)
    hqic = -2 * log_likelihood + 2 * n_params * np.log(np.log(nobs))
    
    # Compute Final Prediction Error (FPE)
    fpe = ((nobs + n_params) / (nobs - n_params)) ** k * np.exp(aic / nobs)
    
    return aic, bic, hqic, fpe


@dataclass
class VARConfig:
    """
    Configuration options for VAR models.
    
    This class provides a standardized way to configure VAR models,
    including estimation methods, optimization settings, and other options.
    
    Attributes:
        method: Estimation method to use ('ols', 'mle')
        trend: Trend specification ('n' for none, 'c' for constant, 't' for trend, 'ct' for both)
        cov_type: Type of covariance matrix to compute ('standard', 'robust', 'hac')
        use_numba: Whether to use Numba acceleration if available
        display_progress: Whether to display progress during estimation
        ic: Information criterion for lag selection ('aic', 'bic', 'hqic', 'fpe')
        max_iter: Maximum number of iterations for optimization
        tol: Convergence tolerance for optimization
    """
    
    method: str = "ols"
    trend: str = "c"
    cov_type: str = "robust"
    use_numba: bool = True
    display_progress: bool = False
    ic: str = "aic"
    max_iter: int = 1000
    tol: float = 1e-8
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Validate method
        valid_methods = ["ols", "mle"]
        if self.method not in valid_methods:
            raise ParameterError(
                f"Invalid estimation method: {self.method}",
                param_name="method",
                param_value=self.method,
                constraint=f"Must be one of {valid_methods}"
            )
        
        # Validate trend
        valid_trends = ["n", "c", "t", "ct"]
        if self.trend not in valid_trends:
            raise ParameterError(
                f"Invalid trend specification: {self.trend}",
                param_name="trend",
                param_value=self.trend,
                constraint=f"Must be one of {valid_trends}"
            )
        
        # Validate cov_type
        valid_cov_types = ["standard", "robust", "hac", "none"]
        if self.cov_type not in valid_cov_types:
            raise ParameterError(
                f"Invalid cov_type: {self.cov_type}",
                param_name="cov_type",
                param_value=self.cov_type,
                constraint=f"Must be one of {valid_cov_types}"
            )
        
        # Validate ic
        valid_ics = ["aic", "bic", "hqic", "fpe"]
        if self.ic not in valid_ics:
            raise ParameterError(
                f"Invalid information criterion: {self.ic}",
                param_name="ic",
                param_value=self.ic,
                constraint=f"Must be one of {valid_ics}"
            )
        
        # Validate max_iter
        if self.max_iter <= 0:
            raise ParameterError(
                f"Invalid max_iter: {self.max_iter}",
                param_name="max_iter",
                param_value=self.max_iter,
                constraint="Must be positive"
            )
        
        # Validate tol
        if self.tol <= 0:
            raise ParameterError(
                f"Invalid tol: {self.tol}",
                param_name="tol",
                param_value=self.tol,
                constraint="Must be positive"
            )


class VARModel(ModelBase):
    """
    Vector Autoregression (VAR) Model.
    
    This class implements Vector Autoregression (VAR) models for multivariate time
    series analysis. It provides methods for model specification, estimation,
    inference, forecasting, and impulse response analysis.
    
    Attributes:
        lags: Number of lags in the VAR model
        config: Configuration options for the VAR model
    """
    
    def __init__(self, lags: int = 1, config: Optional[VARConfig] = None, name: str = "VAR"):
        """
        Initialize the VAR model.
        
        Args:
            lags: Number of lags in the VAR model
            config: Configuration options for the VAR model
            name: A descriptive name for the model
        """
        super().__init__(name=name)
        
        # Validate lags
        if not isinstance(lags, int) or lags <= 0:
            raise ParameterError(
                f"lags must be a positive integer, got {lags}",
                param_name="lags",
                param_value=lags,
                constraint="Must be a positive integer"
            )
        
        self.lags = lags
        self.config = config or VARConfig()
        
        # Initialize model attributes
        self._data = None
        self._index = None
        self._var_names = None
        self._params = None
        self._results = None
        self._statsmodels_var = None
        self._statsmodels_results = None
    
    @property
    def data(self) -> Optional[np.ndarray]:
        """Get the model data."""
        return self._data
    
    @property
    def index(self) -> Optional[Union[pd.DatetimeIndex, pd.Index]]:
        """Get the data index."""
        return self._index
    
    @property
    def var_names(self) -> Optional[List[str]]:
        """Get the variable names."""
        return self._var_names
    
    @property
    def params(self) -> Optional[VARParameters]:
        """Get the model parameters."""
        if not self._fitted:
            raise NotFittedError(
                "Model has not been fitted. Call fit() first.",
                model_type=self._name,
                operation="params"
            )
        return self._params
    
    @property
    def results(self) -> Optional[VARResults]:
        """Get the model results."""
        if not self._fitted:
            raise NotFittedError(
                "Model has not been fitted. Call fit() first.",
                model_type=self._name,
                operation="results"
            )
        return self._results
    
    def validate_data(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Validate the input data for VAR model fitting.
        
        Args:
            data: The data to validate
            
        Returns:
            np.ndarray: The validated data as a NumPy array
            
        Raises:
            TypeError: If the data has an incorrect type
            ValueError: If the data is invalid
        """
        # Store the index and variable names if available
        if isinstance(data, pd.DataFrame):
            self._index = data.index
            self._var_names = data.columns.tolist()
            data_array = data.values
        elif isinstance(data, np.ndarray):
            data_array = data
            self._index = None
            self._var_names = None
        else:
            raise TypeError(
                f"Data must be a NumPy array or Pandas DataFrame, got {type(data).__name__}"
            )
        
        # Check dimensions
        if data_array.ndim != 2:
            raise DimensionError(
                f"Data must be 2-dimensional, got {data_array.ndim} dimensions",
                array_name="data",
                expected_shape="(T, k)",
                actual_shape=data_array.shape
            )
        
        # Check length
        if len(data_array) <= self.lags:
            raise ValueError(
                f"Data length must be greater than lags ({self.lags}), got {len(data_array)}"
            )
        
        # Check for NaN and Inf values
        if np.isnan(data_array).any():
            raise ValueError("Data contains NaN values")
        
        if np.isinf(data_array).any():
            raise ValueError("Data contains infinite values")
        
        return data_array
    
    def fit(self, data: Union[np.ndarray, pd.DataFrame], **kwargs: Any) -> VARResults:
        """
        Fit the VAR model to the provided data.
        
        Args:
            data: The data to fit the model to
            **kwargs: Additional keyword arguments for model fitting
                - method: Estimation method ('ols', 'mle')
                - trend: Trend specification ('n', 'c', 't', 'ct')
                - cov_type: Covariance matrix type ('standard', 'robust', 'hac')
                
        Returns:
            VARResults: The model estimation results
            
        Raises:
            ValueError: If the data is invalid
            EstimationError: If the model estimation fails
        """
        # Validate and prepare data
        data_array = self.validate_data(data)
        self._data = data_array
        
        # Update config with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Create variable names if not available
        if self._var_names is None:
            self._var_names = [f"y{i+1}" for i in range(data_array.shape[1])]
        
        # Use statsmodels for VAR estimation
        try:
            # Create statsmodels VAR model
            if isinstance(data, pd.DataFrame):
                sm_model = StatsmodelsVAR(data)
            else:
                sm_model = StatsmodelsVAR(data_array)
            
            self._statsmodels_var = sm_model
            
            # Fit the model
            sm_result = sm_model.fit(
                maxlags=self.lags,
                method=self.config.method,
                trend=self.config.trend,
                ic=self.config.ic
            )
            
            self._statsmodels_results = sm_result
            
            # Extract results
            k = data_array.shape[1]
            p = self.lags
            
            # Extract coefficient matrices
            coef_matrices = []
            for lag in range(p):
                coef_mat = sm_result.coefs[lag]
                coef_matrices.append(coef_mat)
            
            # Extract constant
            if self.config.trend in ['c', 'ct']:
                constant = sm_result.intercept
            else:
                constant = np.zeros(k)
            
            # Extract sigma
            sigma = sm_result.sigma_u
            
            # Create parameter object
            params = VARParameters(
                coef_matrices=coef_matrices,
                constant=constant,
                sigma=sigma
            )
            
            # Store model attributes
            self._params = params
            self._fitted = True
            
            # Extract residuals and fitted values
            residuals = sm_result.resid
            fitted_values = data_array[p:] - residuals
            
            # Compute log-likelihood
            log_likelihood = sm_result.llf
            
            # Compute information criteria
            nobs = len(residuals)
            aic, bic, hqic, fpe = information_criteria(log_likelihood, nobs, k, p)
            
            # Extract standard errors, t-stats, and p-values
            std_errors = {}
            t_stats = {}
            p_values = {}
            
            # Extract standard errors for constant
            if self.config.trend in ['c', 'ct']:
                std_errors['constant'] = np.array([sm_result.stderr_intercept])
                t_stats['constant'] = np.array([sm_result.tvalues_intercept])
                p_values['constant'] = np.array([sm_result.pvalues_intercept])
            
            # Extract standard errors for coefficient matrices
            for lag in range(p):
                std_errors[f'A{lag+1}'] = sm_result.stderr_endog_lagged[lag]
                t_stats[f'A{lag+1}'] = sm_result.tvalues_endog_lagged[lag]
                p_values[f'A{lag+1}'] = sm_result.pvalues_endog_lagged[lag]
            
            # Create result object
            result = VARResults(
                model_name=self._name,
                params=params,
                residuals=residuals,
                fitted_values=fitted_values,
                log_likelihood=log_likelihood,
                aic=aic,
                bic=bic,
                hqic=hqic,
                fpe=fpe,
                std_errors=std_errors,
                t_stats=t_stats,
                p_values=p_values,
                convergence=True,
                iterations=0,
                cov_type=self.config.cov_type,
                nobs=nobs,
                k=k,
                p=p,
                df_model=k * p + (1 if self.config.trend in ['c', 'ct'] else 0),
                df_resid=nobs - (k * p + (1 if self.config.trend in ['c', 'ct'] else 0))
            )
            
            self._results = result
            return result
            
        except Exception as e:
            raise EstimationError(
                f"VAR model estimation failed: {e}",
                model_type=self._name,
                estimation_method=self.config.method,
                details=str(e)
            )
    
    async def fit_async(self, data: Union[np.ndarray, pd.DataFrame], **kwargs: Any) -> VARResults:
        """
        Asynchronously fit the VAR model to the provided data.
        
        Args:
            data: The data to fit the model to
            **kwargs: Additional keyword arguments for model fitting
                
        Returns:
            VARResults: The model estimation results
            
        Raises:
            ValueError: If the data is invalid
            EstimationError: If the model estimation fails
        """
        # Create a coroutine that runs the synchronous fit method in a thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: self.fit(data, **kwargs)
        )
        return result
    
    def forecast(
        self, 
        steps: int, 
        exog: Optional[np.ndarray] = None,
        confidence_level: float = 0.95,
        **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate forecasts from the fitted VAR model.
        
        Args:
            steps: Number of steps to forecast
            exog: Exogenous variables for the forecast period (not used in VAR)
            confidence_level: Confidence level for prediction intervals
            **kwargs: Additional keyword arguments for forecasting
                
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Forecasts, lower bounds, and upper bounds
            
        Raises:
            NotFittedError: If the model has not been fitted
            ValueError: If the forecast parameters are invalid
            ForecastError: If the forecasting fails
        """
        if not self._fitted:
            raise NotFittedError(
                "Model has not been fitted. Call fit() first.",
                model_type=self._name,
                operation="forecast"
            )
        
        if steps <= 0:
            raise ValueError(f"Steps must be positive, got {steps}")
        
        if not 0 < confidence_level < 1:
            raise ValueError(
                f"Confidence level must be between 0 and 1, got {confidence_level}"
            )
        
        try:
            # Use statsmodels for forecasting
            if self._statsmodels_results is not None:
                # Generate forecasts
                forecast_result = self._statsmodels_results.forecast(
                    y=self._data[-self.lags:], steps=steps
                )
                
                # Compute confidence intervals
                alpha = 1 - confidence_level
                forecast_error = self._statsmodels_results.forecast_interval(
                    y=self._data[-self.lags:], steps=steps, alpha=alpha
                )
                
                lower_bounds = forecast_error[0]
                upper_bounds = forecast_error[1]
                
                return forecast_result, lower_bounds, upper_bounds
            else:
                # Manual forecasting if statsmodels results not available
                k = self._data.shape[1]
                p = self.lags
                
                # Get the last p observations
                y_last = self._data[-p:]
                
                # Initialize forecasts
                forecasts = np.zeros((steps, k))
                
                # Generate forecasts
                for h in range(steps):
                    # For each forecast step
                    for lag in range(p):
                        if lag < h:
                            # Use previously forecasted values
                            lag_data = forecasts[h - lag - 1]
                        else:
                            # Use actual data
                            lag_data = y_last[p - lag - 1]
                        
                        # Add contribution from this lag
                        forecasts[h] += self._params.coef_matrices[lag] @ lag_data
                    
                    # Add constant term
                    forecasts[h] += self._params.constant
                
                # Compute forecast error covariance matrices
                forecast_var = np.zeros((steps, k, k))
                forecast_var[0] = self._params.sigma
                
                for h in range(1, steps):
                    for i in range(h):
                        term = np.zeros((k, k))
                        for j in range(p):
                            if i - j >= 0 and i - j < h:
                                term += self._params.coef_matrices[j] @ forecast_var[i - j - 1] @ self._params.coef_matrices[j].T
                        forecast_var[h] += term
                    
                    forecast_var[h] += self._params.sigma
                
                # Compute confidence intervals
                z_value = stats.norm.ppf(1 - alpha / 2)
                lower_bounds = np.zeros((steps, k))
                upper_bounds = np.zeros((steps, k))
                
                for h in range(steps):
                    std_errors = np.sqrt(np.diag(forecast_var[h]))
                    lower_bounds[h] = forecasts[h] - z_value * std_errors
                    upper_bounds[h] = forecasts[h] + z_value * std_errors
                
                return forecasts, lower_bounds, upper_bounds
            
        except Exception as e:
            raise ForecastError(
                f"VAR forecasting failed: {e}",
                model_type=self._name,
                horizon=steps,
                details=str(e)
            )
    
    async def forecast_async(
        self, 
        steps: int, 
        exog: Optional[np.ndarray] = None,
        confidence_level: float = 0.95,
        **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Asynchronously generate forecasts from the fitted VAR model.
        
        Args:
            steps: Number of steps to forecast
            exog: Exogenous variables for the forecast period (not used in VAR)
            confidence_level: Confidence level for prediction intervals
            **kwargs: Additional keyword arguments for forecasting
                
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Forecasts, lower bounds, and upper bounds
            
        Raises:
            NotFittedError: If the model has not been fitted
            ValueError: If the forecast parameters are invalid
            ForecastError: If the forecasting fails
        """
        # Create a coroutine that runs the synchronous forecast method in a thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: self.forecast(steps, exog, confidence_level, **kwargs)
        )
        return result
    
    def simulate(
        self, 
        n_periods: int, 
        burn: int = 0, 
        initial_values: Optional[np.ndarray] = None,
        random_state: Optional[Union[int, np.random.Generator]] = None,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Simulate data from the VAR model.
        
        Args:
            n_periods: Number of periods to simulate
            burn: Number of initial observations to discard
            initial_values: Initial values for the simulation
            random_state: Random number generator or seed
            **kwargs: Additional keyword arguments for simulation
                
        Returns:
            np.ndarray: Simulated data
            
        Raises:
            NotFittedError: If the model has not been fitted
            ValueError: If the simulation parameters are invalid
            SimulationError: If the simulation fails
        """
        if not self._fitted:
            raise NotFittedError(
                "Model has not been fitted. Call fit() first.",
                model_type=self._name,
                operation="simulate"
            )
        
        if n_periods <= 0:
            raise ValueError(f"n_periods must be positive, got {n_periods}")
        
        if burn < 0:
            raise ValueError(f"burn must be non-negative, got {burn}")
        
        # Set up random number generator
        if isinstance(random_state, np.random.Generator):
            rng = random_state
        else:
            rng = np.random.default_rng(random_state)
        
        try:
            # Extract parameters
            k = self._params.constant.shape[0]
            p = len(self._params.coef_matrices)
            
            # Check initial values
            if initial_values is not None:
                if initial_values.ndim != 2:
                    raise ValueError("initial_values must be a 2D array")
                
                if initial_values.shape[1] != k:
                    raise ValueError(
                        f"initial_values must have {k} columns, got {initial_values.shape[1]}"
                    )
                
                if initial_values.shape[0] < p:
                    raise ValueError(
                        f"initial_values must have at least {p} rows, got {initial_values.shape[0]}"
                    )
                
                # Use the last p observations
                initial_values = initial_values[-p:]
            else:
                # Use zeros as initial values
                initial_values = np.zeros((p, k))
            
            # Total periods to simulate (including burn-in)
            total_periods = n_periods + burn
            
            # Initialize simulated data
            simulated = np.zeros((total_periods + p, k))
            
            # Set initial values
            simulated[:p] = initial_values
            
            # Cholesky decomposition of sigma for generating correlated errors
            chol_sigma = np.linalg.cholesky(self._params.sigma)
            
            # Generate simulated data
            for t in range(p, total_periods + p):
                # Add constant term
                simulated[t] = self._params.constant.copy()
                
                # Add AR terms
                for lag in range(p):
                    simulated[t] += self._params.coef_matrices[lag] @ simulated[t - lag - 1]
                
                # Add random error
                error = rng.standard_normal(k)
                simulated[t] += chol_sigma @ error
            
            # Return simulated data (excluding burn-in and initial values)
            return simulated[p + burn:]
            
        except Exception as e:
            raise SimulationError(
                f"VAR simulation failed: {e}",
                model_type=self._name,
                n_periods=n_periods,
                details=str(e)
            )
    
    async def simulate_async(
        self, 
        n_periods: int, 
        burn: int = 0, 
        initial_values: Optional[np.ndarray] = None,
        random_state: Optional[Union[int, np.random.Generator]] = None,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Asynchronously simulate data from the VAR model.
        
        Args:
            n_periods: Number of periods to simulate
            burn: Number of initial observations to discard
            initial_values: Initial values for the simulation
            random_state: Random number generator or seed
            **kwargs: Additional keyword arguments for simulation
                
        Returns:
            np.ndarray: Simulated data
            
        Raises:
            NotFittedError: If the model has not been fitted
            ValueError: If the simulation parameters are invalid
            SimulationError: If the simulation fails
        """
        # Create a coroutine that runs the synchronous simulate method in a thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: self.simulate(n_periods, burn, initial_values, random_state, **kwargs)
        )
        return result
    
    def impulse_response(
        self, 
        periods: int = 10, 
        method: str = "orthogonalized",
        identification: Optional[Union[str, np.ndarray]] = None,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Compute impulse response functions for the VAR model.
        
        Args:
            periods: Number of periods for impulse response
            method: Method for impulse response calculation
                - "orthogonalized": Orthogonalized impulse responses (Cholesky decomposition)
                - "generalized": Generalized impulse responses (Pesaran and Shin)
                - "structural": Structural impulse responses (requires identification)
            identification: Identification method for structural VAR
                - "short": Short-run restrictions (Cholesky decomposition)
                - "long": Long-run restrictions
                - numpy.ndarray: Custom identification matrix
            **kwargs: Additional keyword arguments for impulse response calculation
                
        Returns:
            np.ndarray: Impulse response functions (periods x k x k)
                
        Raises:
            NotFittedError: If the model has not been fitted
            ValueError: If the parameters are invalid
        """
        if not self._fitted:
            raise NotFittedError(
                "Model has not been fitted. Call fit() first.",
                model_type=self._name,
                operation="impulse_response"
            )
        
        if periods <= 0:
            raise ValueError(f"periods must be positive, got {periods}")
        
        valid_methods = ["orthogonalized", "generalized", "structural"]
        if method not in valid_methods:
            raise ValueError(
                f"method must be one of {valid_methods}, got {method}"
            )
        
        if method == "structural" and identification is None:
            raise ValueError(
                "identification must be provided for structural impulse responses"
            )
        
        try:
            # Use statsmodels for impulse response calculation
            if self._statsmodels_results is not None:
                if method == "orthogonalized":
                    irf = self._statsmodels_results.irf(periods=periods)
                    return irf.orth_irfs
                elif method == "generalized":
                    # Statsmodels doesn't directly support generalized IRFs
                    # Implement manually
                    k = self._params.constant.shape[0]
                    p = len(self._params.coef_matrices)
                    
                    # Compute MA coefficient matrices
                    ma_coefs = self._compute_ma_coefficients(periods)
                    
                    # Compute generalized IRFs
                    sigma = self._params.sigma
                    sigma_diag = np.diag(np.diag(sigma))
                    sigma_diag_sqrt_inv = np.diag(1.0 / np.sqrt(np.diag(sigma)))
                    
                    girf = np.zeros((periods, k, k))
                    for i in range(k):
                        # Shock to variable i
                        e_i = np.zeros(k)
                        e_i[i] = 1.0
                        
                        # Generalized IRF formula
                        sigma_i = sigma[:, i]
                        scaling = 1.0 / np.sqrt(sigma[i, i])
                        
                        for h in range(periods):
                            girf[h, :, i] = scaling * ma_coefs[h] @ sigma_i
                    
                    return girf
                elif method == "structural":
                    # Handle structural IRFs
                    if identification == "short":
                        # Short-run restrictions (Cholesky)
                        A_inv = np.linalg.cholesky(self._params.sigma)
                    elif identification == "long":
                        # Long-run restrictions
                        # Compute long-run impact matrix
                        k = self._params.constant.shape[0]
                        p = len(self._params.coef_matrices)
                        
                        # Compute companion matrix
                        companion = np.zeros((k * p, k * p))
                        
                        # Fill first block row with coefficient matrices
                        for i in range(p):
                            companion[:k, i*k:(i+1)*k] = self._params.coef_matrices[i]
                        
                        # Fill lower block diagonal with identity matrices
                        for i in range(1, p):
                            companion[i*k:(i+1)*k, (i-1)*k:i*k] = np.eye(k)
                        
                        # Compute (I - A)^(-1)
                        I_k = np.eye(k)
                        A_sum = np.zeros((k, k))
                        for i in range(p):
                            A_sum += self._params.coef_matrices[i]
                        
                        long_run = np.linalg.inv(I_k - A_sum)
                        
                        # Compute long-run impact matrix
                        long_run_cov = long_run @ self._params.sigma @ long_run.T
                        
                        # Cholesky decomposition of long-run covariance
                        A_inv = np.linalg.cholesky(long_run_cov)
                        A_inv = long_run @ np.linalg.inv(A_inv)
                    elif isinstance(identification, np.ndarray):
                        # Custom identification matrix
                        if identification.shape != (k, k):
                            raise ValueError(
                                f"identification matrix must have shape ({k}, {k}), "
                                f"got {identification.shape}"
                            )
                        
                        A_inv = identification
                    else:
                        raise ValueError(
                            f"Invalid identification: {identification}"
                        )
                    
                    # Compute MA coefficient matrices
                    ma_coefs = self._compute_ma_coefficients(periods)
                    
                    # Compute structural IRFs
                    sirf = np.zeros((periods, k, k))
                    for h in range(periods):
                        sirf[h] = ma_coefs[h] @ A_inv
                    
                    return sirf
            else:
                # Manual impulse response calculation
                k = self._params.constant.shape[0]
                p = len(self._params.coef_matrices)
                
                # Compute MA coefficient matrices
                ma_coefs = self._compute_ma_coefficients(periods)
                
                if method == "orthogonalized":
                    # Orthogonalized IRFs using Cholesky decomposition
                    chol = np.linalg.cholesky(self._params.sigma)
                    
                    irf = np.zeros((periods, k, k))
                    for h in range(periods):
                        irf[h] = ma_coefs[h] @ chol
                    
                    return irf
                elif method == "generalized":
                    # Generalized IRFs
                    sigma = self._params.sigma
                    
                    girf = np.zeros((periods, k, k))
                    for i in range(k):
                        # Shock to variable i
                        sigma_i = sigma[:, i]
                        scaling = 1.0 / np.sqrt(sigma[i, i])
                        
                        for h in range(periods):
                            girf[h, :, i] = scaling * ma_coefs[h] @ sigma_i
                    
                    return girf
                elif method == "structural":
                    # Handle structural IRFs
                    if identification == "short":
                        # Short-run restrictions (Cholesky)
                        A_inv = np.linalg.cholesky(self._params.sigma)
                    elif identification == "long":
                        # Long-run restrictions
                        # Compute long-run impact matrix
                        I_k = np.eye(k)
                        A_sum = np.zeros((k, k))
                        for i in range(p):
                            A_sum += self._params.coef_matrices[i]
                        
                        long_run = np.linalg.inv(I_k - A_sum)
                        
                        # Compute long-run impact matrix
                        long_run_cov = long_run @ self._params.sigma @ long_run.T
                        
                        # Cholesky decomposition of long-run covariance
                        A_inv = np.linalg.cholesky(long_run_cov)
                        A_inv = long_run @ np.linalg.inv(A_inv)
                    elif isinstance(identification, np.ndarray):
                        # Custom identification matrix
                        if identification.shape != (k, k):
                            raise ValueError(
                                f"identification matrix must have shape ({k}, {k}), "
                                f"got {identification.shape}"
                            )
                        
                        A_inv = identification
                    else:
                        raise ValueError(
                            f"Invalid identification: {identification}"
                        )
                    
                    # Compute structural IRFs
                    sirf = np.zeros((periods, k, k))
                    for h in range(periods):
                        sirf[h] = ma_coefs[h] @ A_inv
                    
                    return sirf
            
        except Exception as e:
            raise ValueError(f"Impulse response calculation failed: {e}")
    
    def _compute_ma_coefficients(self, periods: int) -> List[np.ndarray]:
        """
        Compute Moving Average coefficient matrices for the VAR model.
        
        Args:
            periods: Number of periods
            
        Returns:
            List[np.ndarray]: List of MA coefficient matrices
        """
        k = self._params.constant.shape[0]
        p = len(self._params.coef_matrices)
        
        # Initialize MA coefficient matrices
        ma_coefs = [np.eye(k)]  # Psi_0 = I
        
        # Compute MA coefficients recursively
        for h in range(1, periods):
            psi_h = np.zeros((k, k))
            for j in range(min(h, p)):
                psi_h += self._params.coef_matrices[j] @ ma_coefs[h - j - 1]
            
            ma_coefs.append(psi_h)
        
        return ma_coefs
    
    def forecast_error_variance_decomposition(
        self, 
        periods: int = 10, 
        method: str = "orthogonalized",
        identification: Optional[Union[str, np.ndarray]] = None,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Compute forecast error variance decomposition for the VAR model.
        
        Args:
            periods: Number of periods for variance decomposition
            method: Method for variance decomposition
                - "orthogonalized": Orthogonalized decomposition (Cholesky)
                - "generalized": Generalized decomposition (Pesaran and Shin)
                - "structural": Structural decomposition (requires identification)
            identification: Identification method for structural VAR
                - "short": Short-run restrictions (Cholesky decomposition)
                - "long": Long-run restrictions
                - numpy.ndarray: Custom identification matrix
            **kwargs: Additional keyword arguments for variance decomposition
                
        Returns:
            np.ndarray: Forecast error variance decomposition (periods x k x k)
                
        Raises:
            NotFittedError: If the model has not been fitted
            ValueError: If the parameters are invalid
        """
        if not self._fitted:
            raise NotFittedError(
                "Model has not been fitted. Call fit() first.",
                model_type=self._name,
                operation="forecast_error_variance_decomposition"
            )
        
        if periods <= 0:
            raise ValueError(f"periods must be positive, got {periods}")
        
        valid_methods = ["orthogonalized", "generalized", "structural"]
        if method not in valid_methods:
            raise ValueError(
                f"method must be one of {valid_methods}, got {method}"
            )
        
        if method == "structural" and identification is None:
            raise ValueError(
                "identification must be provided for structural variance decomposition"
            )
        
        try:
            # Compute impulse responses
            irf = self.impulse_response(
                periods=periods, 
                method=method, 
                identification=identification,
                **kwargs
            )
            
            # Compute forecast error variance decomposition
            k = self._params.constant.shape[0]
            fevd = np.zeros((periods, k, k))
            
            if method == "orthogonalized" or method == "structural":
                # Orthogonalized or structural FEVD
                for h in range(periods):
                    # Compute forecast error variance at horizon h
                    for i in range(k):
                        # Variance of variable i
                        var_i = 0.0
                        for s in range(h + 1):
                            for j in range(k):
                                var_i += irf[s, i, j] ** 2
                        
                        # Contribution of each shock
                        for j in range(k):
                            shock_contrib = 0.0
                            for s in range(h + 1):
                                shock_contrib += irf[s, i, j] ** 2
                            
                            fevd[h, i, j] = shock_contrib / var_i if var_i > 0 else 0.0
            elif method == "generalized":
                # Generalized FEVD
                sigma = self._params.sigma
                
                for h in range(periods):
                    # Compute forecast error variance at horizon h
                    for i in range(k):
                        # Variance of variable i
                        var_i = 0.0
                        for s in range(h + 1):
                            for j in range(k):
                                for l in range(k):
                                    var_i += irf[s, i, j] * sigma[j, l] * irf[s, i, l]
                        
                        # Contribution of each shock
                        for j in range(k):
                            shock_contrib = 0.0
                            for s in range(h + 1):
                                shock_contrib += (irf[s, i, j] ** 2) * sigma[j, j]
                            
                            fevd[h, i, j] = shock_contrib / var_i if var_i > 0 else 0.0
                
                # Normalize to sum to 1
                for h in range(periods):
                    for i in range(k):
                        fevd[h, i, :] = fevd[h, i, :] / np.sum(fevd[h, i, :])
            
            return fevd
            
        except Exception as e:
            raise ValueError(f"Variance decomposition calculation failed: {e}")
    
    def granger_causality(
        self, 
        caused: Union[int, str, List[Union[int, str]]],
        causing: Union[int, str, List[Union[int, str]]],
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Test for Granger causality between variables.
        
        Args:
            caused: Index or name of the variable(s) being caused
            causing: Index or name of the variable(s) causing
            **kwargs: Additional keyword arguments for Granger causality test
                
        Returns:
            Dict[str, Any]: Dictionary containing test results
                - test_statistic: Test statistic
                - p_value: p-value of the test
                - df: Degrees of freedom
                - conclusion: Verbal conclusion of the test
                
        Raises:
            NotFittedError: If the model has not been fitted
            ValueError: If the parameters are invalid
        """
        if not self._fitted:
            raise NotFittedError(
                "Model has not been fitted. Call fit() first.",
                model_type=self._name,
                operation="granger_causality"
            )
        
        # Get variable names or indices
        k = self._params.constant.shape[0]
        var_names = self._var_names if self._var_names is not None else [f"y{i+1}" for i in range(k)]
        
        # Convert variable names to indices
        def _get_indices(variables: Union[int, str, List[Union[int, str]]]) -> List[int]:
            if isinstance(variables, (int, str)):
                variables = [variables]
            
            indices = []
            for var in variables:
                if isinstance(var, int):
                    if var < 0 or var >= k:
                        raise ValueError(f"Variable index {var} out of range [0, {k-1}]")
                    indices.append(var)
                elif isinstance(var, str):
                    if var not in var_names:
                        raise ValueError(f"Variable name '{var}' not found in {var_names}")
                    indices.append(var_names.index(var))
                else:
                    raise TypeError(f"Variable must be int or str, got {type(var)}")
            
            return indices
        
        caused_idx = _get_indices(caused)
        causing_idx = _get_indices(causing)
        
        # Check for overlap
        if set(caused_idx).intersection(set(causing_idx)):
            raise ValueError("caused and causing variables must be disjoint")
        
        try:
            # Use statsmodels for Granger causality test
            if self._statsmodels_results is not None:
                # Convert indices to variable names for statsmodels
                causing_names = [var_names[i] for i in causing_idx]
                
                # Perform Granger causality test
                gc_result = self._statsmodels_results.test_causality(
                    caused=caused_idx, causing=causing_idx
                )
                
                # Extract results
                test_statistic = gc_result.test_statistic
                p_value = gc_result.pvalue
                df = gc_result.df
                
                # Determine conclusion
                alpha = kwargs.get('alpha', 0.05)
                if p_value < alpha:
                    conclusion = f"Reject the null hypothesis: {causing_names} Granger-cause the equation variable(s)"
                else:
                    conclusion = f"Fail to reject the null hypothesis: {causing_names} do not Granger-cause the equation variable(s)"
                
                return {
                    'test_statistic': test_statistic,
                    'p_value': p_value,
                    'df': df,
                    'conclusion': conclusion
                }
            else:
                # Manual Granger causality test
                # This is a simplified implementation
                
                # Create restricted model (without causing variables)
                X, y = lag_matrix(self._data, self.lags)
                k = self._data.shape[1]
                
                # Create masks for restricted model
                restricted_mask = np.ones((k, k * self.lags), dtype=bool)
                for i in caused_idx:
                    for j in causing_idx:
                        for lag in range(self.lags):
                            restricted_mask[i, j + lag * k] = False
                
                # Estimate restricted model
                restricted_ssr = np.zeros(k)
                for i in range(k):
                    if i in caused_idx:
                        # For caused variables, use restricted model
                        X_restricted = X[:, restricted_mask[i]]
                        beta_restricted = np.linalg.lstsq(X_restricted, y[:, i], rcond=None)[0]
                        residuals = y[:, i] - X_restricted @ beta_restricted
                        restricted_ssr[i] = np.sum(residuals ** 2)
                    else:
                        # For other variables, use full model
                        beta = np.linalg.lstsq(X, y[:, i], rcond=None)[0]
                        residuals = y[:, i] - X @ beta
                        restricted_ssr[i] = np.sum(residuals ** 2)
                
                # Unrestricted SSR (from full model)
                unrestricted_ssr = np.zeros(k)
                for i in range(k):
                    beta = np.linalg.lstsq(X, y[:, i], rcond=None)[0]
                    residuals = y[:, i] - X @ beta
                    unrestricted_ssr[i] = np.sum(residuals ** 2)
                
                # Compute test statistic
                n = len(y)
                q = len(causing_idx) * self.lags  # Number of restrictions
                p = k * self.lags + 1  # Number of parameters in unrestricted model
                
                # F-statistic for each equation
                f_stats = np.zeros(len(caused_idx))
                p_values = np.zeros(len(caused_idx))
                
                for i, idx in enumerate(caused_idx):
                    f_stats[i] = ((restricted_ssr[idx] - unrestricted_ssr[idx]) / q) / (unrestricted_ssr[idx] / (n - p))
                    p_values[i] = 1 - stats.f.cdf(f_stats[i], q, n - p)
                
                # Combine results (using average for multiple equations)
                test_statistic = np.mean(f_stats)
                p_value = np.mean(p_values)
                df = (q, n - p)
                
                # Determine conclusion
                alpha = kwargs.get('alpha', 0.05)
                causing_names = [var_names[i] for i in causing_idx]
                caused_names = [var_names[i] for i in caused_idx]
                
                if p_value < alpha:
                    conclusion = f"Reject the null hypothesis: {causing_names} Granger-cause {caused_names}"
                else:
                    conclusion = f"Fail to reject the null hypothesis: {causing_names} do not Granger-cause {caused_names}"
                
                return {
                    'test_statistic': test_statistic,
                    'p_value': p_value,
                    'df': df,
                    'conclusion': conclusion
                }
            
        except Exception as e:
            raise ValueError(f"Granger causality test failed: {e}")
    
    def select_order(
        self, 
        data: Union[np.ndarray, pd.DataFrame],
        maxlags: int,
        ic: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Select the optimal lag order for the VAR model.
        
        Args:
            data: The data to use for lag selection
            maxlags: Maximum number of lags to consider
            ic: Information criterion to use ('aic', 'bic', 'hqic', 'fpe')
            **kwargs: Additional keyword arguments for lag selection
                
        Returns:
            Dict[str, Any]: Dictionary containing lag selection results
                - selected_order: Selected lag order
                - aic: AIC values for each lag
                - bic: BIC values for each lag
                - hqic: HQIC values for each lag
                - fpe: FPE values for each lag
                
        Raises:
            ValueError: If the parameters are invalid
        """
        # Validate data
        data_array = self.validate_data(data)
        
        # Use information criterion from config if not provided
        if ic is None:
            ic = self.config.ic
        
        # Validate ic
        valid_ics = ["aic", "bic", "hqic", "fpe"]
        if ic not in valid_ics:
            raise ValueError(
                f"ic must be one of {valid_ics}, got {ic}"
            )
        
        # Validate maxlags
        if maxlags <= 0:
            raise ValueError(f"maxlags must be positive, got {maxlags}")
        
        if maxlags >= len(data_array):
            raise ValueError(
                f"maxlags ({maxlags}) must be less than the number of observations ({len(data_array)})"
            )
        
        try:
            # Use statsmodels for lag selection
            if isinstance(data, pd.DataFrame):
                sm_model = StatsmodelsVAR(data)
            else:
                sm_model = StatsmodelsVAR(data_array)
            
            # Perform lag selection
            lag_results = sm_model.select_order(maxlags=maxlags, trend=self.config.trend)
            
            # Extract results
            selected_orders = {
                'aic': lag_results.aic,
                'bic': lag_results.bic,
                'hqic': lag_results.hqic,
                'fpe': lag_results.fpe
            }
            
            # Get selected order based on specified criterion
            selected_order = selected_orders[ic]
            
            # Extract information criteria for each lag
            aic_values = lag_results.aic_results
            bic_values = lag_results.bic_results
            hqic_values = lag_results.hqic_results
            fpe_values = lag_results.fpe_results
            
            return {
                'selected_order': selected_order,
                'aic': aic_values,
                'bic': bic_values,
                'hqic': hqic_values,
                'fpe': fpe_values
            }
            
        except Exception as e:
            raise ValueError(f"Lag selection failed: {e}")
    
    def summary(self) -> str:
        """
        Generate a text summary of the VAR model.
        
        Returns:
            str: A formatted string containing the model summary
            
        Raises:
            NotFittedError: If the model has not been fitted
        """
        if not self._fitted:
            return f"{self._name}({self.lags}) Model (not fitted)"
        
        if self._results is None:
            return f"{self._name}({self.lags}) Model (fitted, but no results available)"
        
        # Use the result object's summary method
        return self._results.summary()
    
    def to_pandas(self) -> Dict[str, pd.DataFrame]:
        """
        Convert model results to Pandas DataFrames.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of DataFrames containing results
            
        Raises:
            NotFittedError: If the model has not been fitted
        """
        if not self._fitted:
            raise NotFittedError(
                "Model has not been fitted. Call fit() first.",
                model_type=self._name,
                operation="to_pandas"
            )
        
        if self._results is None:
            raise ValueError("Model results not available")
        
        return self._results.to_pandas()



class StructuralVARModel(VARModel):
    """
    Structural Vector Autoregression (SVAR) Model.
    
    This class extends the VAR model to implement Structural VAR models with
    various identification restrictions. It provides methods for model estimation,
    impulse response analysis, and forecast error variance decomposition.
    
    Attributes:
        lags: Number of lags in the VAR model
        config: Configuration options for the VAR model
        identification: Identification method for the SVAR model
        restrictions: Restriction matrix for the SVAR model
    """
    
    def __init__(
        self, 
        lags: int = 1, 
        identification: str = "short",
        restrictions: Optional[np.ndarray] = None,
        config: Optional[VARConfig] = None, 
        name: str = "SVAR"
    ):
        """
        Initialize the Structural VAR model.
        
        Args:
            lags: Number of lags in the VAR model
            identification: Identification method for the SVAR model
                - "short": Short-run restrictions (Cholesky decomposition)
                - "long": Long-run restrictions
                - "blanchard-quah": Blanchard-Quah identification
                - "custom": Custom restrictions (requires restrictions matrix)
            restrictions: Restriction matrix for the SVAR model
                - For "short" and "long": Lower triangular matrix of 0s and 1s
                - For "custom": Matrix of 0s and 1s where 1 indicates a free parameter
            config: Configuration options for the VAR model
            name: A descriptive name for the model
        """
        super().__init__(lags=lags, config=config, name=name)
        
        # Validate identification
        valid_identifications = ["short", "long", "blanchard-quah", "custom"]
        if identification not in valid_identifications:
            raise ParameterError(
                f"Invalid identification method: {identification}",
                param_name="identification",
                param_value=identification,
                constraint=f"Must be one of {valid_identifications}"
            )
        
        self.identification = identification
        
        # Validate restrictions
        if identification == "custom" and restrictions is None:
            raise ParameterError(
                "Restriction matrix must be provided for custom identification",
                param_name="restrictions",
                constraint="Non-None for custom identification"
            )
        
        self.restrictions = restrictions
        
        # Initialize additional attributes
        self._structural_params = None
        self._structural_results = None
    
    @property
    def structural_params(self) -> Optional[Dict[str, np.ndarray]]:
        """Get the structural parameters."""
        if not self._fitted:
            raise NotFittedError(
                "Model has not been fitted. Call fit() first.",
                model_type=self._name,
                operation="structural_params"
            )
        return self._structural_params
    
    def fit(self, data: Union[np.ndarray, pd.DataFrame], **kwargs: Any) -> VARResults:
        """
        Fit the Structural VAR model to the provided data.
        
        Args:
            data: The data to fit the model to
            **kwargs: Additional keyword arguments for model fitting
                
        Returns:
            VARResults: The model estimation results
            
        Raises:
            ValueError: If the data is invalid
            EstimationError: If the model estimation fails
        """
        # First fit the reduced-form VAR model
        results = super().fit(data, **kwargs)
        
        # Now estimate the structural parameters
        try:
            k = self._data.shape[1]
            
            # Validate restrictions matrix if provided
            if self.restrictions is not None:
                if self.restrictions.shape != (k, k):
                    raise ValueError(
                        f"Restriction matrix must have shape ({k}, {k}), "
                        f"got {self.restrictions.shape}"
                    )
                
                if not np.all((self.restrictions == 0) | (self.restrictions == 1)):
                    raise ValueError(
                        "Restriction matrix must contain only 0s and 1s"
                    )
            
            # Estimate structural parameters based on identification method
            if self.identification == "short":
                # Short-run restrictions (Cholesky decomposition)
                A_inv = np.linalg.cholesky(self._params.sigma)
                A = np.linalg.inv(A_inv)
                B = np.eye(k)
                
                # Apply restrictions if provided
                if self.restrictions is not None:
                    # Create mask for restricted elements
                    mask = np.tril(np.ones((k, k))) * self.restrictions
                    
                    # Apply mask to A_inv
                    A_inv_restricted = A_inv * mask
                    
                    # Re-estimate A_inv to satisfy restrictions
                    # This is a simplified approach
                    A_inv = A_inv_restricted
                    A = np.linalg.inv(A_inv)
            
            elif self.identification == "long":
                # Long-run restrictions
                p = self.lags
                
                # Compute long-run impact matrix
                I_k = np.eye(k)
                A_sum = np.zeros((k, k))
                for i in range(p):
                    A_sum += self._params.coef_matrices[i]
                
                long_run = np.linalg.inv(I_k - A_sum)
                
                # Compute long-run impact matrix
                long_run_cov = long_run @ self._params.sigma @ long_run.T
                
                # Cholesky decomposition of long-run covariance
                C = np.linalg.cholesky(long_run_cov)
                
                # Compute A and B matrices
                A_inv = long_run @ np.linalg.inv(C)
                A = np.linalg.inv(A_inv)
                B = np.eye(k)
                
                # Apply restrictions if provided
                if self.restrictions is not None:
                    # Create mask for restricted elements
                    mask = np.tril(np.ones((k, k))) * self.restrictions
                    
                    # Apply mask to C
                    C_restricted = C * mask
                    
                    # Re-estimate A_inv to satisfy restrictions
                    A_inv = long_run @ np.linalg.inv(C_restricted)
                    A = np.linalg.inv(A_inv)
            
            elif self.identification == "blanchard-quah":
                # Blanchard-Quah identification
                # Assumes first variable is non-stationary (e.g., output)
                # and second variable is stationary (e.g., unemployment)
                p = self.lags
                
                # Compute long-run impact matrix
                I_k = np.eye(k)
                A_sum = np.zeros((k, k))
                for i in range(p):
                    A_sum += self._params.coef_matrices[i]
                
                long_run = np.linalg.inv(I_k - A_sum)
                
                # Compute long-run impact matrix
                long_run_cov = long_run @ self._params.sigma @ long_run.T
                
                # Cholesky decomposition of long-run covariance
                C = np.linalg.cholesky(long_run_cov)
                
                # Compute A and B matrices
                A_inv = long_run @ np.linalg.inv(C)
                A = np.linalg.inv(A_inv)
                B = np.eye(k)
                
                # Apply Blanchard-Quah restriction: long-run effect of second shock on first variable is zero
                if k >= 2:
                    # Create restriction matrix
                    restrictions = np.ones((k, k))
                    restrictions[0, 1:] = 0  # Zero long-run effect on first variable
                    
                    # Apply restrictions
                    C_restricted = C * np.tril(restrictions)
                    
                    # Re-estimate A_inv to satisfy restrictions
                    A_inv = long_run @ np.linalg.inv(C_restricted)
                    A = np.linalg.inv(A_inv)
            
            elif self.identification == "custom":
                # Custom identification with provided restrictions
                if self.restrictions is None:
                    raise ValueError(
                        "Restriction matrix must be provided for custom identification"
                    )
                
                # Estimate A and B matrices using maximum likelihood
                # This is a simplified approach
                # In practice, this would involve numerical optimization
                
                # Initialize A and B matrices
                A = np.eye(k)
                B = np.eye(k)
                
                # Count free parameters
                n_free_params = np.sum(self.restrictions)
                
                # Check identification
                if n_free_params > k * (k + 1) / 2:
                    raise ValueError(
                        f"Model is not identified: {n_free_params} free parameters, "
                        f"but only {k * (k + 1) / 2} can be identified"
                    )
                
                # Estimate free parameters
                # This is a placeholder for a more complex estimation procedure
                # In practice, this would involve numerical optimization
                
                # For now, use a simple approach based on Cholesky decomposition
                A_inv = np.linalg.cholesky(self._params.sigma)
                A = np.linalg.inv(A_inv)
                
                # Apply restrictions
                A = A * self.restrictions
            
            # Store structural parameters
            self._structural_params = {
                'A': A,
                'B': B,
                'A_inv': np.linalg.inv(A) if np.linalg.det(A) != 0 else None,
                'B_inv': np.linalg.inv(B) if np.linalg.det(B) != 0 else None,
                'identification': self.identification,
                'restrictions': self.restrictions
            }
            
            return results
            
        except Exception as e:
            raise EstimationError(
                f"Structural VAR estimation failed: {e}",
                model_type=self._name,
                estimation_method=self.config.method,
                details=str(e)
            )
    
    def impulse_response(
        self, 
        periods: int = 10, 
        method: str = "structural",
        **kwargs: Any
    ) -> np.ndarray:
        """
        Compute impulse response functions for the Structural VAR model.
        
        Args:
            periods: Number of periods for impulse response
            method: Method for impulse response calculation
                - "structural": Structural impulse responses (default)
                - "orthogonalized": Orthogonalized impulse responses (Cholesky)
                - "generalized": Generalized impulse responses (Pesaran and Shin)
            **kwargs: Additional keyword arguments for impulse response calculation
                
        Returns:
            np.ndarray: Impulse response functions (periods x k x k)
                
        Raises:
            NotFittedError: If the model has not been fitted
            ValueError: If the parameters are invalid
        """
        if not self._fitted:
            raise NotFittedError(
                "Model has not been fitted. Call fit() first.",
                model_type=self._name,
                operation="impulse_response"
            )
        
        # For structural method, use the structural parameters
        if method == "structural":
            if self._structural_params is None:
                raise ValueError("Structural parameters not available")
            
            # Use the structural parameters for identification
            return super().impulse_response(
                periods=periods,
                method="structural",
                identification=self._structural_params['A_inv'],
                **kwargs
            )
        else:
            # For other methods, use the parent class implementation
            return super().impulse_response(
                periods=periods,
                method=method,
                **kwargs
            )
    
    def forecast_error_variance_decomposition(
        self, 
        periods: int = 10, 
        method: str = "structural",
        **kwargs: Any
    ) -> np.ndarray:
        """
        Compute forecast error variance decomposition for the Structural VAR model.
        
        Args:
            periods: Number of periods for variance decomposition
            method: Method for variance decomposition
                - "structural": Structural decomposition (default)
                - "orthogonalized": Orthogonalized decomposition (Cholesky)
                - "generalized": Generalized decomposition (Pesaran and Shin)
            **kwargs: Additional keyword arguments for variance decomposition
                
        Returns:
            np.ndarray: Forecast error variance decomposition (periods x k x k)
                
        Raises:
            NotFittedError: If the model has not been fitted
            ValueError: If the parameters are invalid
        """
        if not self._fitted:
            raise NotFittedError(
                "Model has not been fitted. Call fit() first.",
                model_type=self._name,
                operation="forecast_error_variance_decomposition"
            )
        
        # For structural method, use the structural parameters
        if method == "structural":
            if self._structural_params is None:
                raise ValueError("Structural parameters not available")
            
            # Use the structural parameters for identification
            return super().forecast_error_variance_decomposition(
                periods=periods,
                method="structural",
                identification=self._structural_params['A_inv'],
                **kwargs
            )
        else:
            # For other methods, use the parent class implementation
            return super().forecast_error_variance_decomposition(
                periods=periods,
                method=method,
                **kwargs
            )
    
    def historical_decomposition(
        self, 
        periods: Optional[int] = None,
        **kwargs: Any
    ) -> Dict[str, np.ndarray]:
        """
        Compute historical decomposition of the data.
        
        Args:
            periods: Number of periods for historical decomposition (default: all available)
            **kwargs: Additional keyword arguments for historical decomposition
                
        Returns:
            Dict[str, np.ndarray]: Dictionary containing historical decomposition
                - baseline: Baseline forecast
                - shock_contributions: Contribution of each shock to each variable
                - actual: Actual data
                
        Raises:
            NotFittedError: If the model has not been fitted
            ValueError: If the parameters are invalid
        """
        if not self._fitted:
            raise NotFittedError(
                "Model has not been fitted. Call fit() first.",
                model_type=self._name,
                operation="historical_decomposition"
            )
        
        if self._structural_params is None:
            raise ValueError("Structural parameters not available")
        
        try:
            # Get dimensions
            k = self._data.shape[1]
            p = self.lags
            T = self._data.shape[0]
            
            # Use all available periods if not specified
            if periods is None:
                periods = T
            else:
                periods = min(periods, T)
            
            # Extract structural parameters
            A_inv = self._structural_params['A_inv']
            
            # Compute MA coefficient matrices
            ma_coefs = self._compute_ma_coefficients(periods)
            
            # Compute structural shocks
            structural_shocks = np.zeros((T, k))
            
            # Get residuals
            residuals = self._results.residuals
            
            # Compute structural shocks
            for t in range(len(residuals)):
                structural_shocks[t] = np.linalg.inv(A_inv) @ residuals[t]
            
            # Compute historical decomposition
            # Initialize arrays
            baseline = np.zeros((periods, k))
            shock_contributions = np.zeros((periods, k, k))
            
            # Compute baseline forecast (unconditional mean)
            if self.config.trend in ['c', 'ct']:
                baseline_mean = np.linalg.inv(np.eye(k) - np.sum(self._params.coef_matrices, axis=0)) @ self._params.constant
                baseline = np.tile(baseline_mean, (periods, 1))
            
            # Compute shock contributions
            for t in range(periods):
                for j in range(k):  # For each shock
                    for i in range(min(t + 1, len(ma_coefs))):  # For each lag up to t
                        if t - i >= 0 and t - i < len(structural_shocks):
                            # Contribution of shock j at time t-i
                            shock_j = np.zeros(k)
                            shock_j[j] = structural_shocks[t - i, j]
                            
                            # Add contribution to each variable
                            shock_contributions[t, :, j] += ma_coefs[i] @ A_inv @ shock_j
            
            # Get actual data for comparison
            actual = self._data[-periods:] if periods <= T else self._data
            
            return {
                'baseline': baseline,
                'shock_contributions': shock_contributions,
                'actual': actual
            }
            
        except Exception as e:
            raise ValueError(f"Historical decomposition failed: {e}")
    
    def summary(self) -> str:
        """
        Generate a text summary of the Structural VAR model.
        
        Returns:
            str: A formatted string containing the model summary
            
        Raises:
            NotFittedError: If the model has not been fitted
        """
        if not self._fitted:
            return f"{self._name}({self.lags}) Model (not fitted)"
        
        if self._results is None:
            return f"{self._name}({self.lags}) Model (fitted, but no results available)"
        
        # Start with the reduced-form VAR summary
        summary = self._results.summary()
        
        # Add structural information
        if self._structural_params is not None:
            k = self._params.constant.shape[0]
            
            structural_info = "\nStructural VAR Information:\n"
            structural_info += "-" * 80 + "\n"
            structural_info += f"Identification method: {self.identification}\n\n"
            
            # Add A matrix
            structural_info += "A matrix (contemporaneous effects):\n"
            for i in range(k):
                row = " ".join(f"{self._structural_params['A'][i, j]:>12.6f}" for j in range(k))
                structural_info += row + "\n"
            structural_info += "\n"
            
            # Add B matrix
            structural_info += "B matrix (shock impacts):\n"
            for i in range(k):
                row = " ".join(f"{self._structural_params['B'][i, j]:>12.6f}" for j in range(k))
                structural_info += row + "\n"
            structural_info += "\n"
            
            # Add restrictions if available
            if self.restrictions is not None:
                structural_info += "Restriction matrix (1 = unrestricted, 0 = restricted to zero):\n"
                for i in range(k):
                    row = " ".join(f"{self.restrictions[i, j]:>12d}" for j in range(k))
                    structural_info += row + "\n"
                structural_info += "\n"
            
            summary += structural_info
        
        return summary
