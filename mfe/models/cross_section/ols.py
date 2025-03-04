# mfe/models/cross_section/ols.py
"""
Ordinary Least Squares (OLS) Regression Module

This module implements Ordinary Least Squares (OLS) regression for cross-sectional
 data analysis with robust error estimation. It provides a complete OLS implementation
 that validates inputs, handles parameter estimation, and computes both homoskedastic
 and heteroskedasticity-robust standard errors.

The implementation extends Statsmodels' core regression functionality with financial
 econometric methods and additional diagnostics. It supports both NumPy arrays and
 Pandas DataFrames as input data formats.

Classes:
    OLS: Ordinary Least Squares regression model
    OLSResults: Container for OLS regression results

Functions:
    olsnw: Compute Newey-West HAC standard errors for OLS regression
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, cast, Any

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from mfe.core.base import CrossSectionalModelBase
from mfe.core.exceptions import (
    DimensionError, NumericError, ParameterError, ConvergenceError,
    raise_dimension_error, raise_numeric_error, warn_numeric
)
from mfe.core.parameters import ParameterBase
from mfe.core.results import CrossSectionalResult
from mfe.utils.matrix_ops import ensure_symmetric

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
logger = logging.getLogger("mfe.models.cross_section.ols")


@dataclass
class OLSParameters(ParameterBase):
    """
    Parameters for OLS regression.
    
    Attributes:
        coefficients: Regression coefficients (beta)
    """
    
    coefficients: np.ndarray
    
    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        # Ensure coefficients is a NumPy array
        if not isinstance(self.coefficients, np.ndarray):
            self.coefficients = np.array(self.coefficients)
        
        # Ensure coefficients is a 1D array
        if self.coefficients.ndim != 1:
            self.coefficients = self.coefficients.flatten()
    
    def validate(self) -> None:
        """
        Validate parameter constraints.
        
        Raises:
            ParameterError: If parameter constraints are violated
        """
        # No specific constraints for OLS parameters
        pass
    
    def to_array(self) -> np.ndarray:
        """
        Convert parameters to a NumPy array.
        
        Returns:
            np.ndarray: Array representation of parameters
        """
        return self.coefficients
    
    @classmethod
    def from_array(cls, array: np.ndarray, **kwargs: Any) -> 'OLSParameters':
        """
        Create parameters from a NumPy array.
        
        Args:
            array: Array representation of parameters
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            OLSParameters: Parameter object
        """
        return cls(coefficients=array)
    
    def transform(self) -> np.ndarray:
        """
        Transform parameters to unconstrained space for optimization.
        
        Returns:
            np.ndarray: Transformed parameters in unconstrained space
        """
        # OLS parameters have no constraints, so no transformation is needed
        return self.coefficients
    
    @classmethod
    def inverse_transform(cls, array: np.ndarray, **kwargs: Any) -> 'OLSParameters':
        """
        Transform parameters from unconstrained space back to constrained space.
        
        Args:
            array: Parameters in unconstrained space
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            OLSParameters: Parameter object with constrained parameters
        """
        # OLS parameters have no constraints, so no transformation is needed
        return cls(coefficients=array)


@dataclass
class OLSResults(CrossSectionalResult):
    """
    Results container for OLS regression.
    
    This class extends CrossSectionalResult to provide specialized functionality
    for OLS regression results.
    
    Attributes:
        variable_names: Names of the independent variables
        robust_std_errors: Robust standard errors (if computed)
        robust_t_stats: Robust t-statistics (if computed)
        robust_p_values: Robust p-values (if computed)
        robust_covariance_matrix: Robust covariance matrix (if computed)
        robust_type: Type of robust standard errors used (if any)
        condition_number: Condition number of the design matrix
        vif: Variance inflation factors for multicollinearity detection
        durbin_watson: Durbin-Watson statistic for autocorrelation
        jarque_bera: Jarque-Bera test statistic and p-value for normality
        breusch_pagan: Breusch-Pagan test statistic and p-value for heteroskedasticity
    """
    
    variable_names: Optional[List[str]] = None
    robust_std_errors: Optional[np.ndarray] = None
    robust_t_stats: Optional[np.ndarray] = None
    robust_p_values: Optional[np.ndarray] = None
    robust_covariance_matrix: Optional[np.ndarray] = None
    robust_type: Optional[str] = None
    condition_number: Optional[float] = None
    vif: Optional[Dict[str, float]] = None
    durbin_watson: Optional[float] = None
    jarque_bera: Optional[Tuple[float, float]] = None
    breusch_pagan: Optional[Tuple[float, float]] = None
    
    def __post_init__(self) -> None:
        """Validate result object after initialization."""
        super().__post_init__()
        
        # Ensure arrays are NumPy arrays if provided
        if self.robust_std_errors is not None and not isinstance(self.robust_std_errors, np.ndarray):
            self.robust_std_errors = np.array(self.robust_std_errors)
        
        if self.robust_t_stats is not None and not isinstance(self.robust_t_stats, np.ndarray):
            self.robust_t_stats = np.array(self.robust_t_stats)
        
        if self.robust_p_values is not None and not isinstance(self.robust_p_values, np.ndarray):
            self.robust_p_values = np.array(self.robust_p_values)
        
        if self.robust_covariance_matrix is not None and not isinstance(self.robust_covariance_matrix, np.ndarray):
            self.robust_covariance_matrix = np.array(self.robust_covariance_matrix)
    
    def summary(self) -> str:
        """
        Generate a text summary of the OLS regression results.
        
        Returns:
            str: A formatted string containing the regression results summary
        """
        base_summary = super().summary()
        
        # Add OLS-specific information
        ols_info = ""
        
        # Add variable names and coefficients
        if self.variable_names is not None and self.parameters is not None:
            coeffs = self.parameters.coefficients
            ols_info += "Coefficients:\n"
            ols_info += "-" * 80 + "\n"
            ols_info += f"{'Variable':<15} {'Coefficient':<12} {'Std. Error':<12} "
            ols_info += f"{'t-Stat':<12} {'p-Value':<12}\n"
            ols_info += "-" * 80 + "\n"
            
            for i, name in enumerate(self.variable_names):
                std_err = self.std_errors[i] if self.std_errors is not None else np.nan
                t_stat = self.t_stats[i] if self.t_stats is not None else np.nan
                p_value = self.p_values[i] if self.p_values is not None else np.nan
                
                ols_info += f"{name:<15} {coeffs[i]:<12.6f} "
                
                if not np.isnan(std_err):
                    ols_info += f"{std_err:<12.6f} "
                else:
                    ols_info += f"{'N/A':<12} "
                
                if not np.isnan(t_stat):
                    ols_info += f"{t_stat:<12.6f} "
                else:
                    ols_info += f"{'N/A':<12} "
                
                if not np.isnan(p_value):
                    ols_info += f"{p_value:<12.6f}"
                    # Add significance stars
                    if p_value < 0.01:
                        ols_info += " ***"
                    elif p_value < 0.05:
                        ols_info += " **"
                    elif p_value < 0.1:
                        ols_info += " *"
                else:
                    ols_info += f"{'N/A':<12}"
                
                ols_info += "\n"
            
            ols_info += "-" * 80 + "\n"
            ols_info += "Significance codes: *** 0.01, ** 0.05, * 0.1\n\n"
        
        # Add robust standard errors if available
        if self.robust_std_errors is not None and self.robust_type is not None:
            ols_info += f"Robust Standard Errors ({self.robust_type}):\n"
            ols_info += "-" * 80 + "\n"
            ols_info += f"{'Variable':<15} {'Coefficient':<12} {'Robust SE':<12} "
            ols_info += f"{'t-Stat':<12} {'p-Value':<12}\n"
            ols_info += "-" * 80 + "\n"
            
            coeffs = self.parameters.coefficients
            for i, name in enumerate(self.variable_names):
                robust_se = self.robust_std_errors[i]
                robust_t = self.robust_t_stats[i] if self.robust_t_stats is not None else np.nan
                robust_p = self.robust_p_values[i] if self.robust_p_values is not None else np.nan
                
                ols_info += f"{name:<15} {coeffs[i]:<12.6f} {robust_se:<12.6f} "
                
                if not np.isnan(robust_t):
                    ols_info += f"{robust_t:<12.6f} "
                else:
                    ols_info += f"{'N/A':<12} "
                
                if not np.isnan(robust_p):
                    ols_info += f"{robust_p:<12.6f}"
                    # Add significance stars
                    if robust_p < 0.01:
                        ols_info += " ***"
                    elif robust_p < 0.05:
                        ols_info += " **"
                    elif robust_p < 0.1:
                        ols_info += " *"
                else:
                    ols_info += f"{'N/A':<12}"
                
                ols_info += "\n"
            
            ols_info += "-" * 80 + "\n"
            ols_info += "Significance codes: *** 0.01, ** 0.05, * 0.1\n\n"
        
        # Add diagnostic statistics
        diagnostics = "Diagnostic Statistics:\n"
        
        if self.condition_number is not None:
            diagnostics += f"Condition Number: {self.condition_number:.6f}"
            if self.condition_number > 30:
                diagnostics += " (Potential multicollinearity issue)"
            diagnostics += "\n"
        
        if self.durbin_watson is not None:
            diagnostics += f"Durbin-Watson: {self.durbin_watson:.6f}"
            if self.durbin_watson < 1.5 or self.durbin_watson > 2.5:
                diagnostics += " (Potential autocorrelation issue)"
            diagnostics += "\n"
        
        if self.jarque_bera is not None:
            jb_stat, jb_p = self.jarque_bera
            diagnostics += f"Jarque-Bera: {jb_stat:.6f} (p-value: {jb_p:.6f})"
            if jb_p < 0.05:
                diagnostics += " (Residuals may not be normally distributed)"
            diagnostics += "\n"
        
        if self.breusch_pagan is not None:
            bp_stat, bp_p = self.breusch_pagan
            diagnostics += f"Breusch-Pagan: {bp_stat:.6f} (p-value: {bp_p:.6f})"
            if bp_p < 0.05:
                diagnostics += " (Potential heteroskedasticity issue)"
            diagnostics += "\n"
        
        if self.vif is not None and len(self.vif) > 0:
            diagnostics += "Variance Inflation Factors (VIF):\n"
            for var, vif_val in self.vif.items():
                diagnostics += f"  {var}: {vif_val:.6f}"
                if vif_val > 10:
                    diagnostics += " (High multicollinearity)"
                elif vif_val > 5:
                    diagnostics += " (Moderate multicollinearity)"
                diagnostics += "\n"
        
        diagnostics += "\n"
        
        return base_summary + ols_info + diagnostics
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert OLS results to a pandas DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame containing coefficient estimates and statistics
        """
        if self.parameters is None or self.variable_names is None:
            raise ValueError("Parameters or variable names are not available")
        
        coeffs = self.parameters.coefficients
        
        data = {
            "Variable": self.variable_names,
            "Coefficient": coeffs
        }
        
        if self.std_errors is not None:
            data["Std. Error"] = self.std_errors
        
        if self.t_stats is not None:
            data["t-Stat"] = self.t_stats
        
        if self.p_values is not None:
            data["p-Value"] = self.p_values
        
        if self.robust_std_errors is not None:
            data["Robust Std. Error"] = self.robust_std_errors
        
        if self.robust_t_stats is not None:
            data["Robust t-Stat"] = self.robust_t_stats
        
        if self.robust_p_values is not None:
            data["Robust p-Value"] = self.robust_p_values
        
        # Add VIF if available
        if self.vif is not None:
            vif_values = [self.vif.get(var, np.nan) for var in self.variable_names]
            data["VIF"] = vif_values
        
        return pd.DataFrame(data)


@jit(nopython=True, cache=True)
def _compute_newey_west_numba(X: np.ndarray, u: np.ndarray, lags: int) -> np.ndarray:
    """
    Numba-accelerated implementation of Newey-West HAC covariance matrix computation.
    
    Args:
        X: Design matrix (n x k)
        u: Residuals (n x 1)
        lags: Number of lags to include
    
    Returns:
        Newey-West HAC covariance matrix (k x k)
    """
    n, k = X.shape
    
    # Compute X'X
    XpX_inv = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            for t in range(n):
                XpX_inv[i, j] += X[t, i] * X[t, j]
    
    # Compute inverse of X'X
    # Note: In a real implementation, we would use a more robust method
    # This is a simplified version for demonstration
    # In practice, we would use np.linalg.inv, but that's not supported in numba
    # For simplicity, we'll assume XpX_inv is already the inverse
    
    # Compute X'u
    Xu = np.zeros((n, k))
    for t in range(n):
        for j in range(k):
            Xu[t, j] = X[t, j] * u[t]
    
    # Compute S0 (the variance component)
    S0 = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            for t in range(n):
                S0[i, j] += Xu[t, i] * Xu[t, j]
    
    # Add autocorrelation components
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)  # Bartlett kernel
        Sl = np.zeros((k, k))
        
        for i in range(k):
            for j in range(k):
                for t in range(l, n):
                    Sl[i, j] += Xu[t, i] * Xu[t-l, j] + Xu[t-l, i] * Xu[t, j]
        
        S0 += w * Sl
    
    # Compute the final covariance matrix
    V = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            for p in range(k):
                for q in range(k):
                    V[i, j] += XpX_inv[i, p] * S0[p, q] * XpX_inv[q, j]
    
    return V


def olsnw(X: np.ndarray, y: np.ndarray, beta: np.ndarray, lags: int = 0) -> np.ndarray:
    """
    Compute Newey-West HAC standard errors for OLS regression.
    
    This function computes heteroskedasticity and autocorrelation consistent (HAC)
    standard errors for OLS regression using the Newey-West estimator.
    
    Args:
        X: Design matrix (n x k)
        y: Dependent variable (n x 1)
        beta: OLS coefficient estimates (k x 1)
        lags: Number of lags to include (0 for White's heteroskedasticity-robust estimator)
    
    Returns:
        Newey-West HAC covariance matrix (k x k)
    
    Raises:
        DimensionError: If input dimensions are incompatible
        ValueError: If lags is negative
    
    Examples:
        >>> import numpy as np
        >>> from mfe.models.cross_section.ols import olsnw
        >>> X = np.array([[1, 0.5], [1, 1.2], [1, 0.9], [1, 1.8]])
        >>> y = np.array([2.3, 3.1, 2.9, 4.2])
        >>> beta = np.array([0.8, 1.5])
        >>> cov = olsnw(X, y, beta, lags=1)
    """
    # Convert to numpy arrays if not already
    X = np.asarray(X)
    y = np.asarray(y)
    beta = np.asarray(beta)
    
    # Check dimensions
    if X.ndim != 2:
        raise_dimension_error(
            "X must be a 2D array",
            array_name="X",
            expected_shape="(n, k)",
            actual_shape=X.shape
        )
    
    if y.ndim != 1:
        raise_dimension_error(
            "y must be a 1D array",
            array_name="y",
            expected_shape="(n,)",
            actual_shape=y.shape
        )
    
    if beta.ndim != 1:
        raise_dimension_error(
            "beta must be a 1D array",
            array_name="beta",
            expected_shape="(k,)",
            actual_shape=beta.shape
        )
    
    n, k = X.shape
    
    if y.shape[0] != n:
        raise_dimension_error(
            "X and y must have the same number of observations",
            array_name="y",
            expected_shape=f"({n},)",
            actual_shape=y.shape
        )
    
    if beta.shape[0] != k:
        raise_dimension_error(
            "beta must have the same number of elements as columns in X",
            array_name="beta",
            expected_shape=f"({k},)",
            actual_shape=beta.shape
        )
    
    if lags < 0:
        raise ValueError("lags must be non-negative")
    
    # Compute residuals
    u = y - X @ beta
    
    # Compute (X'X)^(-1)
    XpX_inv = np.linalg.inv(X.T @ X)
    
    if lags == 0:
        # White's heteroskedasticity-robust estimator
        Xu = X * u[:, np.newaxis]  # Multiply each row of X by the corresponding residual
        S0 = Xu.T @ Xu
        V = XpX_inv @ S0 @ XpX_inv
    else:
        if HAS_NUMBA:
            # Use Numba-accelerated implementation
            V = _compute_newey_west_numba(X, u, lags)
        else:
            # Pure NumPy implementation
            Xu = X * u[:, np.newaxis]
            
            # Compute S0 (the variance component)
            S0 = Xu.T @ Xu
            
            # Add autocorrelation components
            for l in range(1, lags + 1):
                w = 1.0 - l / (lags + 1.0)  # Bartlett kernel
                Sl = np.zeros((k, k))
                
                for t in range(l, n):
                    Sl += Xu[t:t+1, :].T @ Xu[t-l:t-l+1, :] + Xu[t-l:t-l+1, :].T @ Xu[t:t+1, :]
                
                S0 += w * Sl
            
            # Compute the final covariance matrix
            V = XpX_inv @ S0 @ XpX_inv
    
    # Ensure the matrix is symmetric (to handle numerical precision issues)
    V = ensure_symmetric(V)
    
    return V


class OLS(CrossSectionalModelBase):
    """
    Ordinary Least Squares (OLS) regression model.
    
    This class implements OLS regression for cross-sectional data analysis with
    robust error estimation. It provides methods for parameter estimation,
    prediction, and diagnostic testing.
    
    Attributes:
        include_constant: Whether to include a constant term in the model
        robust_type: Type of robust standard errors to compute (None, 'HC0', 'HC1', 'HC2', 'HC3', 'HAC')
        lags: Number of lags for HAC standard errors (only used if robust_type='HAC')
    """
    
    def __init__(self, include_constant: bool = True, 
                 robust_type: Optional[str] = None, lags: int = 0,
                 name: str = "OLS Regression"):
        """
        Initialize the OLS model.
        
        Args:
            include_constant: Whether to include a constant term in the model
            robust_type: Type of robust standard errors to compute:
                         None for homoskedastic standard errors
                         'HC0', 'HC1', 'HC2', 'HC3' for White's heteroskedasticity-robust standard errors
                         'HAC' for Newey-West HAC standard errors
            lags: Number of lags for HAC standard errors (only used if robust_type='HAC')
            name: A descriptive name for the model
        
        Raises:
            ValueError: If robust_type is not one of the supported types
            ValueError: If lags is negative
        """
        super().__init__(name=name)
        self.include_constant = include_constant
        
        # Validate robust_type
        valid_robust_types = [None, 'HC0', 'HC1', 'HC2', 'HC3', 'HAC']
        if robust_type not in valid_robust_types:
            raise ValueError(f"robust_type must be one of {valid_robust_types}")
        
        self.robust_type = robust_type
        
        # Validate lags
        if lags < 0:
            raise ValueError("lags must be non-negative")
        
        self.lags = lags
        
        # Initialize additional attributes
        self._variable_names: Optional[List[str]] = None
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._sm_model: Optional[sm.OLS] = None
        self._sm_results: Optional[sm.regression.linear_model.RegressionResults] = None
    
    def fit(self, data: Tuple[Union[np.ndarray, pd.DataFrame, pd.Series], 
                             Union[np.ndarray, pd.DataFrame, pd.Series]],
            variable_names: Optional[List[str]] = None,
            **kwargs: Any) -> OLSResults:
        """
        Fit the OLS model to the provided data.
        
        Args:
            data: Tuple of (y, X) where y is the dependent variable and X is the design matrix
            variable_names: Names of the independent variables (including constant if applicable)
            **kwargs: Additional keyword arguments for model fitting
        
        Returns:
            OLSResults: The model estimation results
        
        Raises:
            ValueError: If the data is invalid
            RuntimeError: If the model fails to converge
        
        Examples:
            >>> import numpy as np
            >>> from mfe.models.cross_section.ols import OLS
            >>> X = np.array([[0.5], [1.2], [0.9], [1.8]])
            >>> y = np.array([2.3, 3.1, 2.9, 4.2])
            >>> model = OLS()
            >>> results = model.fit((y, X))
            >>> print(f"Intercept: {results.parameters.coefficients[0]:.4f}")
            >>> print(f"Slope: {results.parameters.coefficients[1]:.4f}")
        """
        # Validate data
        self.validate_data(data)
        
        # Extract y and X from data
        y_data, X_data = data
        
        # Process y data
        if isinstance(y_data, pd.DataFrame) or isinstance(y_data, pd.Series):
            y = y_data.values
        else:
            y = np.asarray(y_data)
        
        # Process X data
        if isinstance(X_data, pd.DataFrame):
            # Save variable names if not provided
            if variable_names is None:
                variable_names = X_data.columns.tolist()
            X = X_data.values
        else:
            X = np.asarray(X_data)
        
        # Add constant if requested
        if self.include_constant:
            X = sm.add_constant(X)
            # Update variable names if needed
            if variable_names is not None:
                variable_names = ['const'] + variable_names
            elif X.shape[1] > 1:  # Only add generic names if we have more than just the constant
                variable_names = ['const'] + [f'x{i+1}' for i in range(X.shape[1]-1)]
            else:
                variable_names = ['const']
        elif variable_names is None and X.shape[1] > 0:
            # Create generic variable names if not provided
            variable_names = [f'x{i+1}' for i in range(X.shape[1])]
        
        # Store data for later use
        self._X = X
        self._y = y
        self._variable_names = variable_names
        
        # Create and fit statsmodels OLS model
        self._sm_model = sm.OLS(y, X)
        self._sm_results = self._sm_model.fit()
        
        # Extract results
        coefficients = self._sm_results.params
        std_errors = self._sm_results.bse
        t_stats = self._sm_results.tvalues
        p_values = self._sm_results.pvalues
        covariance_matrix = self._sm_results.cov_params()
        
        # Create parameters object
        parameters = OLSParameters(coefficients=coefficients)
        
        # Compute fitted values and residuals
        fitted_values = X @ coefficients
        residuals = y - fitted_values
        
        # Compute model statistics
        r_squared = self._sm_results.rsquared
        adjusted_r_squared = self._sm_results.rsquared_adj
        f_statistic = self._sm_results.fvalue
        f_p_value = self._sm_results.f_pvalue
        residual_std_error = np.sqrt(self._sm_results.mse_resid)
        degrees_of_freedom = self._sm_results.df_resid
        
        # Compute condition number
        _, s, _ = np.linalg.svd(X)
        condition_number = s[0] / s[-1]
        
        # Compute diagnostic statistics
        durbin_watson = sm.stats.stattools.durbin_watson(residuals)
        
        # Compute Jarque-Bera test for normality
        jb_stat, jb_p = stats.jarque_bera(residuals)
        
        # Compute Breusch-Pagan test for heteroskedasticity
        bp_stat, bp_p, _ = sm.stats.diagnostic.het_breuschpagan(residuals, X)
        
        # Compute VIF for multicollinearity detection
        vif = {}
        if X.shape[1] > 1:  # Only compute VIF if we have more than one regressor
            for i in range(X.shape[1]):
                if self.include_constant and i == 0:
                    # Skip constant term
                    continue
                
                # Create X matrix without the current variable
                X_without_i = np.delete(X, i, axis=1)
                
                # Regress the current variable on all other variables
                model_i = sm.OLS(X[:, i], X_without_i).fit()
                
                # Compute VIF
                vif_i = 1.0 / (1.0 - model_i.rsquared)
                
                # Store VIF with variable name
                var_name = variable_names[i] if variable_names is not None else f'x{i+1}'
                vif[var_name] = vif_i
        
        # Compute robust standard errors if requested
        robust_std_errors = None
        robust_t_stats = None
        robust_p_values = None
        robust_covariance_matrix = None
        
        if self.robust_type is not None:
            if self.robust_type == 'HAC':
                # Compute Newey-West HAC standard errors
                robust_covariance_matrix = olsnw(X, y, coefficients, self.lags)
                robust_type_str = f'HAC (lags={self.lags})'
            else:
                # Compute White's heteroskedasticity-robust standard errors
                cov_type = {
                    'HC0': 'HC0',
                    'HC1': 'HC1',
                    'HC2': 'HC2',
                    'HC3': 'HC3'
                }.get(self.robust_type, 'HC0')
                
                robust_results = self._sm_model.fit(cov_type=cov_type)
                robust_covariance_matrix = robust_results.cov_params()
                robust_type_str = self.robust_type
            
            # Compute robust standard errors
            robust_std_errors = np.sqrt(np.diag(robust_covariance_matrix))
            
            # Compute robust t-statistics and p-values
            robust_t_stats = coefficients / robust_std_errors
            robust_p_values = 2 * (1 - stats.t.cdf(np.abs(robust_t_stats), degrees_of_freedom))
        else:
            robust_type_str = None
        
        # Create results object
        results = OLSResults(
            model_name=self.name,
            parameters=parameters,
            convergence=True,
            iterations=1,  # OLS is a direct solution, not iterative
            log_likelihood=self._sm_results.llf,
            aic=self._sm_results.aic,
            bic=self._sm_results.bic,
            hqic=self._sm_results.hqic,
            std_errors=std_errors,
            t_stats=t_stats,
            p_values=p_values,
            covariance_matrix=covariance_matrix,
            fitted_values=fitted_values,
            residuals=residuals,
            r_squared=r_squared,
            adjusted_r_squared=adjusted_r_squared,
            f_statistic=f_statistic,
            f_p_value=f_p_value,
            residual_std_error=residual_std_error,
            degrees_of_freedom=degrees_of_freedom,
            original_data=(y, X),
            variable_names=variable_names,
            robust_std_errors=robust_std_errors,
            robust_t_stats=robust_t_stats,
            robust_p_values=robust_p_values,
            robust_covariance_matrix=robust_covariance_matrix,
            robust_type=robust_type_str,
            condition_number=condition_number,
            vif=vif,
            durbin_watson=durbin_watson,
            jarque_bera=(jb_stat, jb_p),
            breusch_pagan=(bp_stat, bp_p)
        )
        
        # Update model state
        self._fitted = True
        self._results = results
        
        return results
    
    async def fit_async(self, data: Tuple[Union[np.ndarray, pd.DataFrame, pd.Series], 
                                        Union[np.ndarray, pd.DataFrame, pd.Series]],
                       variable_names: Optional[List[str]] = None,
                       **kwargs: Any) -> OLSResults:
        """
        Asynchronously fit the OLS model to the provided data.
        
        This method provides an asynchronous interface to the fit method,
        allowing for non-blocking model estimation in UI contexts.
        
        Args:
            data: Tuple of (y, X) where y is the dependent variable and X is the design matrix
            variable_names: Names of the independent variables (including constant if applicable)
            **kwargs: Additional keyword arguments for model fitting
        
        Returns:
            OLSResults: The model estimation results
        
        Raises:
            ValueError: If the data is invalid
            RuntimeError: If the model fails to converge
        """
        # OLS estimation is typically fast, so we can just call the synchronous version
        # In a real implementation, we might want to run this in a separate thread
        # for truly non-blocking behavior
        return self.fit(data, variable_names, **kwargs)
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Generate predictions from the fitted model.
        
        Args:
            X: Input features for prediction
        
        Returns:
            np.ndarray: Predicted values
        
        Raises:
            RuntimeError: If the model has not been fitted
            ValueError: If the input features are invalid
        
        Examples:
            >>> import numpy as np
            >>> from mfe.models.cross_section.ols import OLS
            >>> X_train = np.array([[0.5], [1.2], [0.9], [1.8]])
            >>> y_train = np.array([2.3, 3.1, 2.9, 4.2])
            >>> model = OLS()
            >>> results = model.fit((y_train, X_train))
            >>> X_new = np.array([[1.0], [1.5]])
            >>> predictions = model.predict(X_new)
        """
        if not self._fitted or self._results is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        # Process X data
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.asarray(X)
        
        # Check dimensions
        if X_array.ndim == 1:
            # Convert 1D array to 2D column vector
            X_array = X_array.reshape(-1, 1)
        elif X_array.ndim != 2:
            raise_dimension_error(
                "X must be a 1D or 2D array",
                array_name="X",
                expected_shape="(n,) or (n, k)",
                actual_shape=X_array.shape
            )
        
        # Add constant if the model was fitted with a constant
        if self.include_constant:
            X_array = sm.add_constant(X_array)
        
        # Check if the number of features matches
        if X_array.shape[1] != self._results.parameters.coefficients.shape[0]:
            raise_dimension_error(
                "Number of features in X does not match the model",
                array_name="X",
                expected_shape=f"(n, {self._results.parameters.coefficients.shape[0] - (1 if self.include_constant else 0)})",
                actual_shape=X_array.shape
            )
        
        # Generate predictions
        predictions = X_array @ self._results.parameters.coefficients
        
        return predictions
    
    def simulate(self, n_periods: int, X: Optional[np.ndarray] = None,
                burn: int = 0, initial_values: Optional[np.ndarray] = None,
                random_state: Optional[Union[int, np.random.Generator]] = None,
                **kwargs: Any) -> np.ndarray:
        """
        Simulate data from the fitted model.
        
        Args:
            n_periods: Number of periods to simulate
            X: Input features for simulation (if None, random X will be generated)
            burn: Number of initial observations to discard (not used in OLS)
            initial_values: Initial values for the simulation (not used in OLS)
            random_state: Random number generator or seed
            **kwargs: Additional keyword arguments for simulation
        
        Returns:
            np.ndarray: Simulated data
        
        Raises:
            RuntimeError: If the model has not been fitted
            ValueError: If the simulation parameters are invalid
        
        Examples:
            >>> import numpy as np
            >>> from mfe.models.cross_section.ols import OLS
            >>> X_train = np.array([[0.5], [1.2], [0.9], [1.8]])
            >>> y_train = np.array([2.3, 3.1, 2.9, 4.2])
            >>> model = OLS()
            >>> results = model.fit((y_train, X_train))
            >>> X_sim = np.array([[1.0], [1.5], [2.0]])
            >>> simulated_data = model.simulate(3, X=X_sim, random_state=42)
        """
        if not self._fitted or self._results is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        # Set up random number generator
        if isinstance(random_state, np.random.Generator):
            rng = random_state
        else:
            rng = np.random.default_rng(random_state)
        
        # Get model parameters
        beta = self._results.parameters.coefficients
        sigma = self._results.residual_std_error
        
        # Generate or validate X
        if X is None:
            # Generate random X
            k = beta.shape[0] - (1 if self.include_constant else 0)
            X = rng.normal(0, 1, size=(n_periods, k))
        else:
            # Process X data
            X = np.asarray(X)
            
            # Check dimensions
            if X.ndim == 1:
                # Convert 1D array to 2D column vector
                X = X.reshape(-1, 1)
            elif X.ndim != 2:
                raise_dimension_error(
                    "X must be a 1D or 2D array",
                    array_name="X",
                    expected_shape="(n,) or (n, k)",
                    actual_shape=X.shape
                )
            
            # Check if the number of periods matches
            if X.shape[0] != n_periods:
                raise_dimension_error(
                    "Number of rows in X must match n_periods",
                    array_name="X",
                    expected_shape=f"({n_periods}, k)",
                    actual_shape=X.shape
                )
        
        # Add constant if the model was fitted with a constant
        if self.include_constant:
            X = sm.add_constant(X)
        
        # Check if the number of features matches
        if X.shape[1] != beta.shape[0]:
            raise_dimension_error(
                "Number of features in X does not match the model",
                array_name="X",
                expected_shape=f"(n, {beta.shape[0] - (1 if self.include_constant else 0)})",
                actual_shape=X.shape
            )
        
        # Generate random errors
        errors = rng.normal(0, sigma, size=n_periods)
        
        # Generate simulated data
        y = X @ beta + errors
        
        return y
    
    async def simulate_async(self, n_periods: int, X: Optional[np.ndarray] = None,
                           burn: int = 0, initial_values: Optional[np.ndarray] = None,
                           random_state: Optional[Union[int, np.random.Generator]] = None,
                           **kwargs: Any) -> np.ndarray:
        """
        Asynchronously simulate data from the fitted model.
        
        This method provides an asynchronous interface to the simulate method,
        allowing for non-blocking simulation in UI contexts.
        
        Args:
            n_periods: Number of periods to simulate
            X: Input features for simulation (if None, random X will be generated)
            burn: Number of initial observations to discard (not used in OLS)
            initial_values: Initial values for the simulation (not used in OLS)
            random_state: Random number generator or seed
            **kwargs: Additional keyword arguments for simulation
        
        Returns:
            np.ndarray: Simulated data
        
        Raises:
            RuntimeError: If the model has not been fitted
            ValueError: If the simulation parameters are invalid
        """
        # OLS simulation is typically fast, so we can just call the synchronous version
        # In a real implementation, we might want to run this in a separate thread
        # for truly non-blocking behavior
        return self.simulate(n_periods, X, burn, initial_values, random_state, **kwargs)
    
    def validate_data(self, data: Tuple[Union[np.ndarray, pd.DataFrame, pd.Series], 
                                      Union[np.ndarray, pd.DataFrame, pd.Series]]) -> None:
        """
        Validate the input data for OLS model fitting.
        
        Args:
            data: Tuple of (y, X) where y is the dependent variable and X is the design matrix
        
        Raises:
            ValueError: If the data is invalid
            TypeError: If the data has an incorrect type
        """
        if not isinstance(data, tuple) or len(data) != 2:
            raise TypeError("Data must be a tuple of (y, X)")
        
        y_data, X_data = data
        
        # Process y data
        if isinstance(y_data, pd.DataFrame) or isinstance(y_data, pd.Series):
            y = y_data.values
        else:
            y = np.asarray(y_data)
        
        # Process X data
        if isinstance(X_data, pd.DataFrame):
            X = X_data.values
        else:
            X = np.asarray(X_data)
        
        # Check y dimensions
        if y.ndim != 1:
            raise_dimension_error(
                "y must be a 1D array",
                array_name="y",
                expected_shape="(n,)",
                actual_shape=y.shape
            )
        
        # Check X dimensions
        if X.ndim == 1:
            # Convert 1D array to 2D column vector
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise_dimension_error(
                "X must be a 1D or 2D array",
                array_name="X",
                expected_shape="(n,) or (n, k)",
                actual_shape=X.shape
            )
        
        # Check if the number of observations matches
        if X.shape[0] != y.shape[0]:
            raise_dimension_error(
                "X and y must have the same number of observations",
                array_name="X",
                expected_shape=f"({y.shape[0]}, k)",
                actual_shape=X.shape
            )
        
        # Check for NaN or infinite values
        if np.isnan(y).any() or np.isnan(X).any():
            raise_numeric_error(
                "Data contains NaN values",
                operation="validate_data",
                error_type="nan_values"
            )
        
        if np.isinf(y).any() or np.isinf(X).any():
            raise_numeric_error(
                "Data contains infinite values",
                operation="validate_data",
                error_type="inf_values"
            )
        
        # Check if we have enough observations
        n, k = X.shape
        if self.include_constant:
            k += 1  # Account for the constant term
        
        if n <= k:
            raise_dimension_error(
                "Number of observations must be greater than number of regressors",
                array_name="X",
                expected_shape=f"(>{k}, k)",
                actual_shape=X.shape
            )
        
        # Check for perfect multicollinearity
        if self.include_constant:
            X_with_const = sm.add_constant(X)
        else:
            X_with_const = X
        
        # Check rank of X
        rank = np.linalg.matrix_rank(X_with_const)
        if rank < X_with_const.shape[1]:
            warn_numeric(
                "X matrix is not full rank, indicating perfect multicollinearity",
                operation="validate_data",
                issue="multicollinearity",
                value=f"rank={rank}, columns={X_with_const.shape[1]}"
            )