# mfe/models/cross_section/base.py
"""
Abstract base classes and interfaces for cross-sectional analysis tools.

This module defines the foundational classes and interfaces for cross-sectional
analysis in the MFE Toolbox, including OLS regression and Principal Component
Analysis (PCA). It provides common functionality for parameter validation,
result storage, and diagnostic metrics that are shared across different
cross-sectional analysis implementations.

The base classes establish a consistent API pattern and ensure proper type
safety through Python's abstract base classes and type annotations.
"""

import abc
import warnings
from dataclasses import dataclass, field
from typing import (
    Any, Dict, List, Literal, Optional, Protocol, Tuple, Type, TypeVar, 
    Union, cast, overload, Callable, AsyncIterator
)
import numpy as np
import pandas as pd
from scipy import stats, linalg

from mfe.core.base import CrossSectionalModelBase, ModelBase
from mfe.core.parameters import ParameterBase, ParameterError, validate_positive
from mfe.core.results import CrossSectionalResult, ModelResult

# Type variables for generic parameters
T = TypeVar('T', bound=ParameterBase)  # Parameter type
R = TypeVar('R', bound=CrossSectionalResult)  # Result type


@dataclass
class CrossSectionalParameters(ParameterBase):
    """Base class for cross-sectional model parameters.
    
    This class provides common functionality for cross-sectional model
    parameters, including validation and transformation methods.
    
    Attributes:
        include_constant: Whether to include a constant term in the model
    """
    
    include_constant: bool = True
    
    def validate(self) -> None:
        """Validate parameter constraints for cross-sectional models.
        
        Raises:
            ParameterError: If parameter constraints are violated
        """
        # Base class doesn't enforce specific constraints
        # Subclasses should implement model-specific constraints
        pass
    
    def to_array(self) -> np.ndarray:
        """Convert parameters to a NumPy array.
        
        Returns:
            np.ndarray: Array representation of parameters
        """
        return np.array([self.include_constant], dtype=bool)
    
    @classmethod
    def from_array(cls, array: np.ndarray, **kwargs: Any) -> 'CrossSectionalParameters':
        """Create parameters from a NumPy array.
        
        Args:
            array: Array representation of parameters
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            CrossSectionalParameters: Parameter object
        
        Raises:
            ValueError: If the array length doesn't match the expected length
        """
        if len(array) != 1:
            raise ValueError(f"Array length must be 1, got {len(array)}")
        
        return cls(include_constant=bool(array[0]))
    
    def transform(self) -> np.ndarray:
        """Transform parameters to unconstrained space for optimization.
        
        Returns:
            np.ndarray: Transformed parameters in unconstrained space
        """
        # Boolean parameters don't need transformation
        return self.to_array()
    
    @classmethod
    def inverse_transform(cls, array: np.ndarray, **kwargs: Any) -> 'CrossSectionalParameters':
        """Transform parameters from unconstrained space back to constrained space.
        
        Args:
            array: Parameters in unconstrained space
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            CrossSectionalParameters: Parameter object with constrained parameters
        """
        return cls.from_array(array, **kwargs)


class CrossSectionalModel(CrossSectionalModelBase[T, R]):
    """Abstract base class for cross-sectional models.
    
    This class extends CrossSectionalModelBase to provide specialized functionality
    for cross-sectional analysis, including methods for parameter estimation,
    prediction, and diagnostics.
    
    Type Parameters:
        T: The parameter type for this model
        R: The result type for this model
    """
    
    def __init__(self, name: str = "CrossSectionalModel"):
        """Initialize the cross-sectional model.
        
        Args:
            name: A descriptive name for the model
        """
        super().__init__(name=name)
        self._parameters: Optional[T] = None
    
    @property
    def parameters(self) -> Optional[T]:
        """Get the model parameters.
        
        Returns:
            Optional[T]: The model parameters if fitted, None otherwise
        
        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self._parameters
    
    @abc.abstractmethod
    def fit(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> R:
        """Fit the model to the provided data.
        
        This method must be implemented by all subclasses to estimate model
        parameters from the provided data.
        
        Args:
            data: The data to fit the model to, as a tuple of (y, X)
            **kwargs: Additional keyword arguments for model fitting
        
        Returns:
            R: The model estimation results
        
        Raises:
            ValueError: If the data is invalid
            RuntimeError: If the model fails to converge
        """
        pass
    
    async def fit_async(self, data: Tuple[np.ndarray, np.ndarray], 
                       progress_callback: Optional[Callable[[float, str], None]] = None,
                       **kwargs: Any) -> R:
        """Asynchronously fit the model to the provided data.
        
        This method provides an asynchronous interface to the fit method,
        allowing for non-blocking model estimation in UI contexts.
        
        Args:
            data: The data to fit the model to, as a tuple of (y, X)
            progress_callback: Optional callback function to report progress
            **kwargs: Additional keyword arguments for model fitting
        
        Returns:
            R: The model estimation results
        
        Raises:
            ValueError: If the data is invalid
            RuntimeError: If the model fails to converge
        """
        # Report initial progress
        if progress_callback:
            progress_callback(0.0, "Starting model estimation...")
        
        # Validate data
        self.validate_data(data)
        y, X = data
        
        # Report progress after validation
        if progress_callback:
            progress_callback(0.1, "Data validated, preparing for estimation...")
        
        # Default implementation calls the synchronous version
        # Subclasses can override with truly asynchronous implementations
        result = self.fit(data, **kwargs)
        
        # Report completion
        if progress_callback:
            progress_callback(1.0, "Model estimation completed.")
        
        return result
    
    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions from the fitted model.
        
        This method must be implemented by all subclasses to generate predictions
        based on the fitted model parameters.
        
        Args:
            X: Input features for prediction
        
        Returns:
            np.ndarray: Predicted values
        
        Raises:
            RuntimeError: If the model has not been fitted
            ValueError: If the input features are invalid
        """
        pass
    
    def validate_data(self, data: Tuple[np.ndarray, np.ndarray]) -> None:
        """Validate the input data for cross-sectional model fitting.
        
        Args:
            data: The data to validate, as a tuple of (y, X)
        
        Raises:
            ValueError: If the data is invalid
            TypeError: If the data has an incorrect type
        """
        super().validate_data(data)
        
        # Additional validation specific to cross-sectional models
        y, X = data
        
        # Check for NaN or infinite values in X
        if np.isnan(X).any():
            raise ValueError("X contains NaN values")
        
        if np.isinf(X).any():
            raise ValueError("X contains infinite values")
        
        # Check for NaN or infinite values in y
        if np.isnan(y).any():
            raise ValueError("y contains NaN values")
        
        if np.isinf(y).any():
            raise ValueError("y contains infinite values")
        
        # Check for sufficient observations
        if len(y) < X.shape[1] + 1:
            raise ValueError(
                f"Insufficient observations ({len(y)}) for the number of variables ({X.shape[1]})"
            )
    
    def compute_residuals(self, y: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Compute residuals from the fitted model.
        
        Args:
            y: Observed values
            X: Input features
        
        Returns:
            np.ndarray: Residuals (y - y_hat)
        
        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        y_hat = self.predict(X)
        return y - y_hat
    
    def compute_r_squared(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """Compute the coefficient of determination (R²).
        
        Args:
            y: Observed values
            y_hat: Predicted values
        
        Returns:
            float: R-squared value
        """
        # Total sum of squares
        y_mean = np.mean(y)
        ss_total = np.sum((y - y_mean) ** 2)
        
        # Residual sum of squares
        ss_residual = np.sum((y - y_hat) ** 2)
        
        # R-squared
        r_squared = 1 - (ss_residual / ss_total)
        
        return r_squared
    
    def compute_adjusted_r_squared(self, r_squared: float, n: int, k: int) -> float:
        """Compute the adjusted coefficient of determination.
        
        Args:
            r_squared: R-squared value
            n: Number of observations
            k: Number of predictors (including constant)
        
        Returns:
            float: Adjusted R-squared value
        """
        return 1 - ((1 - r_squared) * (n - 1) / (n - k))
    
    def compute_f_statistic(self, r_squared: float, n: int, k: int) -> Tuple[float, float]:
        """Compute the F-statistic and its p-value.
        
        Args:
            r_squared: R-squared value
            n: Number of observations
            k: Number of predictors (including constant)
        
        Returns:
            Tuple[float, float]: F-statistic and its p-value
        """
        # F-statistic
        f_stat = (r_squared / (k - 1)) / ((1 - r_squared) / (n - k))
        
        # p-value
        p_value = 1 - stats.f.cdf(f_stat, k - 1, n - k)
        
        return f_stat, p_value
    
    def compute_residual_std_error(self, residuals: np.ndarray, n: int, k: int) -> float:
        """Compute the residual standard error.
        
        Args:
            residuals: Model residuals
            n: Number of observations
            k: Number of predictors (including constant)
        
        Returns:
            float: Residual standard error
        """
        return np.sqrt(np.sum(residuals ** 2) / (n - k))
    
    def compute_diagnostic_tests(self, residuals: np.ndarray, X: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Compute diagnostic tests for the model.
        
        Args:
            residuals: Model residuals
            X: Input features
        
        Returns:
            Dict[str, Dict[str, float]]: Dictionary of test results
        """
        tests = {}
        
        # Jarque-Bera test for normality
        jb_stat = len(residuals) / 6 * (
            stats.skew(residuals) ** 2 + 
            (stats.kurtosis(residuals, fisher=True) ** 2) / 4
        )
        jb_p_value = 1 - stats.chi2.cdf(jb_stat, 2)
        tests["jarque_bera"] = {
            "statistic": jb_stat,
            "p_value": jb_p_value,
            "description": "Test for normality of residuals"
        }
        
        # Breusch-Pagan test for heteroskedasticity
        # Regress squared residuals on X
        n = len(residuals)
        residuals_squared = residuals ** 2
        residuals_squared_mean = np.mean(residuals_squared)
        
        # Normalize squared residuals
        u = residuals_squared / residuals_squared_mean - 1
        
        # Compute explained sum of squares
        if X.shape[1] > 0:  # Only if we have predictors
            # Add constant if not already present
            if np.all(X[:, 0] != 1):
                X_bp = np.column_stack((np.ones(n), X))
            else:
                X_bp = X
            
            # Compute OLS coefficients
            try:
                beta_bp = np.linalg.solve(X_bp.T @ X_bp, X_bp.T @ u)
                u_hat = X_bp @ beta_bp
                ess = np.sum(u_hat ** 2)
                
                # BP statistic
                bp_stat = n * ess / 2
                bp_p_value = 1 - stats.chi2.cdf(bp_stat, X_bp.shape[1] - 1)
                
                tests["breusch_pagan"] = {
                    "statistic": bp_stat,
                    "p_value": bp_p_value,
                    "description": "Test for heteroskedasticity"
                }
            except np.linalg.LinAlgError:
                # Skip if matrix is singular
                pass
        
        return tests
    
    def summary(self) -> str:
        """Generate a text summary of the model.
        
        Returns:
            str: A formatted string containing the model summary
        
        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self._fitted:
            return f"Model: {self._name} (not fitted)"
        
        if self._results is None:
            return f"Model: {self._name} (fitted, but no results available)"
        
        return self._results.summary()


class DimensionReductionModel(ModelBase[T, R, np.ndarray]):
    """Abstract base class for dimension reduction models.
    
    This class extends ModelBase to provide specialized functionality
    for dimension reduction techniques like PCA.
    
    Type Parameters:
        T: The parameter type for this model
        R: The result type for this model
    """
    
    def __init__(self, name: str = "DimensionReductionModel"):
        """Initialize the dimension reduction model.
        
        Args:
            name: A descriptive name for the model
        """
        super().__init__(name=name)
        self._components: Optional[np.ndarray] = None
        self._explained_variance: Optional[np.ndarray] = None
        self._explained_variance_ratio: Optional[np.ndarray] = None
    
    @property
    def components(self) -> Optional[np.ndarray]:
        """Get the principal components or loadings.
        
        Returns:
            Optional[np.ndarray]: The components if fitted, None otherwise
        
        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self._components
    
    @property
    def explained_variance(self) -> Optional[np.ndarray]:
        """Get the explained variance for each component.
        
        Returns:
            Optional[np.ndarray]: The explained variance if fitted, None otherwise
        
        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self._explained_variance
    
    @property
    def explained_variance_ratio(self) -> Optional[np.ndarray]:
        """Get the ratio of explained variance for each component.
        
        Returns:
            Optional[np.ndarray]: The explained variance ratio if fitted, None otherwise
        
        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self._explained_variance_ratio
    
    @abc.abstractmethod
    def fit(self, data: np.ndarray, **kwargs: Any) -> R:
        """Fit the model to the provided data.
        
        This method must be implemented by all subclasses to estimate model
        parameters from the provided data.
        
        Args:
            data: The data to fit the model to
            **kwargs: Additional keyword arguments for model fitting
        
        Returns:
            R: The model estimation results
        
        Raises:
            ValueError: If the data is invalid
        """
        pass
    
    async def fit_async(self, data: np.ndarray, 
                       progress_callback: Optional[Callable[[float, str], None]] = None,
                       **kwargs: Any) -> R:
        """Asynchronously fit the model to the provided data.
        
        This method provides an asynchronous interface to the fit method,
        allowing for non-blocking model estimation in UI contexts.
        
        Args:
            data: The data to fit the model to
            progress_callback: Optional callback function to report progress
            **kwargs: Additional keyword arguments for model fitting
        
        Returns:
            R: The model estimation results
        
        Raises:
            ValueError: If the data is invalid
        """
        # Report initial progress
        if progress_callback:
            progress_callback(0.0, "Starting dimension reduction...")
        
        # Validate data
        self.validate_data(data)
        
        # Report progress after validation
        if progress_callback:
            progress_callback(0.1, "Data validated, preparing for dimension reduction...")
        
        # Default implementation calls the synchronous version
        # Subclasses can override with truly asynchronous implementations
        result = self.fit(data, **kwargs)
        
        # Report completion
        if progress_callback:
            progress_callback(1.0, "Dimension reduction completed.")
        
        return result
    
    @abc.abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using the fitted model.
        
        This method must be implemented by all subclasses to transform
        data based on the fitted model parameters.
        
        Args:
            data: The data to transform
        
        Returns:
            np.ndarray: Transformed data
        
        Raises:
            RuntimeError: If the model has not been fitted
            ValueError: If the data is invalid
        """
        pass
    
    @abc.abstractmethod
    def inverse_transform(self, transformed_data: np.ndarray) -> np.ndarray:
        """Inverse transform data using the fitted model.
        
        This method must be implemented by all subclasses to inverse transform
        data based on the fitted model parameters.
        
        Args:
            transformed_data: The transformed data to inverse transform
        
        Returns:
            np.ndarray: Inverse transformed data
        
        Raises:
            RuntimeError: If the model has not been fitted
            ValueError: If the data is invalid
        """
        pass
    
    def validate_data(self, data: np.ndarray) -> None:
        """Validate the input data for dimension reduction.
        
        Args:
            data: The data to validate
        
        Raises:
            ValueError: If the data is invalid
            TypeError: If the data has an incorrect type
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a NumPy array")
        
        if data.ndim != 2:
            raise ValueError(f"Data must be 2-dimensional, got {data.ndim} dimensions")
        
        if data.shape[0] < 2:
            raise ValueError(f"Data must have at least 2 observations, got {data.shape[0]}")
        
        if data.shape[1] < 2:
            raise ValueError(f"Data must have at least 2 variables, got {data.shape[1]}")
        
        if np.isnan(data).any():
            raise ValueError("Data contains NaN values")
        
        if np.isinf(data).any():
            raise ValueError("Data contains infinite values")
    
    def compute_explained_variance_ratio(self, explained_variance: np.ndarray) -> np.ndarray:
        """Compute the ratio of explained variance for each component.
        
        Args:
            explained_variance: Explained variance for each component
        
        Returns:
            np.ndarray: Explained variance ratio
        """
        total_variance = np.sum(explained_variance)
        return explained_variance / total_variance
    
    def compute_cumulative_explained_variance(self, explained_variance_ratio: np.ndarray) -> np.ndarray:
        """Compute the cumulative explained variance ratio.
        
        Args:
            explained_variance_ratio: Explained variance ratio for each component
        
        Returns:
            np.ndarray: Cumulative explained variance ratio
        """
        return np.cumsum(explained_variance_ratio)
    
    def summary(self) -> str:
        """Generate a text summary of the model.
        
        Returns:
            str: A formatted string containing the model summary
        
        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self._fitted:
            return f"Model: {self._name} (not fitted)"
        
        if self._results is None:
            return f"Model: {self._name} (fitted, but no results available)"
        
        return self._results.summary()


@dataclass
class PCAParameters(ParameterBase):
    """Parameters for Principal Component Analysis (PCA).
    
    Attributes:
        n_components: Number of components to keep (if None, keep all components)
        standardize: Whether to standardize the data before PCA
        method: Method for computing PCA ('svd' or 'eig')
    """
    
    n_components: Optional[int] = None
    standardize: bool = True
    method: Literal['svd', 'eig'] = 'svd'
    
    def validate(self) -> None:
        """Validate parameter constraints for PCA.
        
        Raises:
            ParameterError: If parameter constraints are violated
        """
        if self.n_components is not None and self.n_components < 1:
            raise ParameterError(f"n_components must be positive, got {self.n_components}")
        
        if self.method not in ['svd', 'eig']:
            raise ParameterError(f"method must be 'svd' or 'eig', got {self.method}")
    
    def to_array(self) -> np.ndarray:
        """Convert parameters to a NumPy array.
        
        Returns:
            np.ndarray: Array representation of parameters
        """
        n_components_val = -1 if self.n_components is None else self.n_components
        method_val = 0 if self.method == 'svd' else 1
        
        return np.array([n_components_val, self.standardize, method_val])
    
    @classmethod
    def from_array(cls, array: np.ndarray, **kwargs: Any) -> 'PCAParameters':
        """Create parameters from a NumPy array.
        
        Args:
            array: Array representation of parameters
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            PCAParameters: Parameter object
        
        Raises:
            ValueError: If the array length doesn't match the expected length
        """
        if len(array) != 3:
            raise ValueError(f"Array length must be 3, got {len(array)}")
        
        n_components_val = None if array[0] < 0 else int(array[0])
        standardize = bool(array[1])
        method = 'svd' if array[2] == 0 else 'eig'
        
        return cls(
            n_components=n_components_val,
            standardize=standardize,
            method=method
        )
    
    def transform(self) -> np.ndarray:
        """Transform parameters to unconstrained space for optimization.
        
        Returns:
            np.ndarray: Transformed parameters in unconstrained space
        """
        # These parameters don't need transformation
        return self.to_array()
    
    @classmethod
    def inverse_transform(cls, array: np.ndarray, **kwargs: Any) -> 'PCAParameters':
        """Transform parameters from unconstrained space back to constrained space.
        
        Args:
            array: Parameters in unconstrained space
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            PCAParameters: Parameter object with constrained parameters
        """
        return cls.from_array(array, **kwargs)


@dataclass
class OLSParameters(CrossSectionalParameters):
    """Parameters for Ordinary Least Squares (OLS) regression.
    
    Attributes:
        include_constant: Whether to include a constant term in the model
        robust_se: Whether to compute robust standard errors
        cov_type: Type of covariance estimator for robust standard errors
    """
    
    robust_se: bool = False
    cov_type: Literal['hc0', 'hc1', 'hc2', 'hc3'] = 'hc1'
    
    def validate(self) -> None:
        """Validate parameter constraints for OLS.
        
        Raises:
            ParameterError: If parameter constraints are violated
        """
        super().validate()
        
        if self.cov_type not in ['hc0', 'hc1', 'hc2', 'hc3']:
            raise ParameterError(
                f"cov_type must be one of 'hc0', 'hc1', 'hc2', 'hc3', got {self.cov_type}"
            )
    
    def to_array(self) -> np.ndarray:
        """Convert parameters to a NumPy array.
        
        Returns:
            np.ndarray: Array representation of parameters
        """
        cov_type_val = {'hc0': 0, 'hc1': 1, 'hc2': 2, 'hc3': 3}[self.cov_type]
        
        return np.array([self.include_constant, self.robust_se, cov_type_val])
    
    @classmethod
    def from_array(cls, array: np.ndarray, **kwargs: Any) -> 'OLSParameters':
        """Create parameters from a NumPy array.
        
        Args:
            array: Array representation of parameters
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            OLSParameters: Parameter object
        
        Raises:
            ValueError: If the array length doesn't match the expected length
        """
        if len(array) != 3:
            raise ValueError(f"Array length must be 3, got {len(array)}")
        
        include_constant = bool(array[0])
        robust_se = bool(array[1])
        cov_type_val = int(array[2])
        cov_type = ['hc0', 'hc1', 'hc2', 'hc3'][cov_type_val]
        
        return cls(
            include_constant=include_constant,
            robust_se=robust_se,
            cov_type=cov_type
        )
    
    def transform(self) -> np.ndarray:
        """Transform parameters to unconstrained space for optimization.
        
        Returns:
            np.ndarray: Transformed parameters in unconstrained space
        """
        # These parameters don't need transformation
        return self.to_array()
    
    @classmethod
    def inverse_transform(cls, array: np.ndarray, **kwargs: Any) -> 'OLSParameters':
        """Transform parameters from unconstrained space back to constrained space.
        
        Args:
            array: Parameters in unconstrained space
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            OLSParameters: Parameter object with constrained parameters
        """
        return cls.from_array(array, **kwargs)


@dataclass
class CrossSectionalModelResult(CrossSectionalResult):
    """Result container for cross-sectional models.
    
    This class extends CrossSectionalResult to provide specialized functionality
    for cross-sectional model results.
    
    Attributes:
        coefficients: Estimated model coefficients
        variable_names: Names of the variables
        diagnostic_tests: Dictionary of diagnostic test results
    """
    
    coefficients: Optional[np.ndarray] = None
    variable_names: Optional[List[str]] = None
    diagnostic_tests: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate result object after initialization."""
        super().__post_init__()
        
        # Ensure coefficients is a NumPy array if provided
        if self.coefficients is not None and not isinstance(self.coefficients, np.ndarray):
            self.coefficients = np.array(self.coefficients)
        
        # Create default variable names if not provided
        if self.coefficients is not None and self.variable_names is None:
            self.variable_names = [f"X{i}" for i in range(len(self.coefficients))]
    
    def summary(self) -> str:
        """Generate a text summary of the cross-sectional model results.
        
        Returns:
            str: A formatted string containing the model results summary
        """
        base_summary = super().summary()
        
        # Add coefficient table
        coef_table = ""
        if self.coefficients is not None and self.variable_names is not None:
            coef_table = "Coefficient Estimates:\n"
            coef_table += "-" * 80 + "\n"
            coef_table += f"{'Variable':<20} {'Coefficient':<12} {'Std. Error':<12} "
            coef_table += f"{'t-Stat':<12} {'p-Value':<12}\n"
            coef_table += "-" * 80 + "\n"
            
            for i, (name, coef) in enumerate(zip(self.variable_names, self.coefficients)):
                std_err = self.std_errors[i] if self.std_errors is not None else np.nan
                t_stat = self.t_stats[i] if self.t_stats is not None else np.nan
                p_value = self.p_values[i] if self.p_values is not None else np.nan
                
                coef_table += f"{name:<20} {coef:<12.6f} "
                
                if not np.isnan(std_err):
                    coef_table += f"{std_err:<12.6f} "
                else:
                    coef_table += f"{'N/A':<12} "
                
                if not np.isnan(t_stat):
                    coef_table += f"{t_stat:<12.6f} "
                else:
                    coef_table += f"{'N/A':<12} "
                
                if not np.isnan(p_value):
                    coef_table += f"{p_value:<12.6f}"
                    # Add significance stars
                    if p_value < 0.01:
                        coef_table += " ***"
                    elif p_value < 0.05:
                        coef_table += " **"
                    elif p_value < 0.1:
                        coef_table += " *"
                else:
                    coef_table += f"{'N/A':<12}"
                
                coef_table += "\n"
            
            coef_table += "-" * 80 + "\n"
            coef_table += "Significance codes: *** 0.01, ** 0.05, * 0.1\n\n"
        
        # Add diagnostic test results
        diag_table = ""
        if self.diagnostic_tests:
            diag_table = "Diagnostic Tests:\n"
            diag_table += "-" * 60 + "\n"
            diag_table += f"{'Test':<30} {'Statistic':<12} {'p-Value':<12}\n"
            diag_table += "-" * 60 + "\n"
            
            for test_name, test_results in self.diagnostic_tests.items():
                test_stat = test_results.get("statistic", np.nan)
                p_value = test_results.get("p_value", np.nan)
                
                diag_table += f"{test_name:<30} "
                
                if not np.isnan(test_stat):
                    diag_table += f"{test_stat:<12.6f} "
                else:
                    diag_table += f"{'N/A':<12} "
                
                if not np.isnan(p_value):
                    diag_table += f"{p_value:<12.6f}"
                else:
                    diag_table += f"{'N/A':<12}"
                
                diag_table += "\n"
            
            diag_table += "-" * 60 + "\n\n"
        
        return base_summary + coef_table + diag_table
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert coefficient estimates to a pandas DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame containing coefficient estimates and statistics
        
        Raises:
            ValueError: If coefficients are not available
        """
        if self.coefficients is None or self.variable_names is None:
            raise ValueError("Coefficients or variable names are not available")
        
        data = {
            "Variable": self.variable_names,
            "Coefficient": self.coefficients
        }
        
        if self.std_errors is not None:
            data["Std. Error"] = self.std_errors
        
        if self.t_stats is not None:
            data["t-Stat"] = self.t_stats
        
        if self.p_values is not None:
            data["p-Value"] = self.p_values
        
        return pd.DataFrame(data)


@dataclass
class PCAResult(ModelResult):
    """Result container for Principal Component Analysis (PCA).
    
    This class extends ModelResult to provide specialized functionality
    for PCA results.
    
    Attributes:
        components: Principal components (loadings)
        explained_variance: Explained variance for each component
        explained_variance_ratio: Ratio of explained variance for each component
        cumulative_explained_variance: Cumulative explained variance ratio
        n_components: Number of components
        mean: Mean of the data (used for centering)
        std: Standard deviation of the data (used for scaling)
        variable_names: Names of the variables
    """
    
    components: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_explained_variance: np.ndarray
    n_components: int
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    variable_names: Optional[List[str]] = None
    
    def __post_init__(self) -> None:
        """Validate result object after initialization."""
        super().__post_init__()
        
        # Ensure arrays are NumPy arrays
        if not isinstance(self.components, np.ndarray):
            self.components = np.array(self.components)
        
        if not isinstance(self.explained_variance, np.ndarray):
            self.explained_variance = np.array(self.explained_variance)
        
        if not isinstance(self.explained_variance_ratio, np.ndarray):
            self.explained_variance_ratio = np.array(self.explained_variance_ratio)
        
        if not isinstance(self.cumulative_explained_variance, np.ndarray):
            self.cumulative_explained_variance = np.array(self.cumulative_explained_variance)
        
        if self.mean is not None and not isinstance(self.mean, np.ndarray):
            self.mean = np.array(self.mean)
        
        if self.std is not None and not isinstance(self.std, np.ndarray):
            self.std = np.array(self.std)
        
        # Create default variable names if not provided
        if self.variable_names is None:
            n_vars = self.components.shape[1] if self.components.ndim > 1 else 1
            self.variable_names = [f"Var{i+1}" for i in range(n_vars)]
    
    def summary(self) -> str:
        """Generate a text summary of the PCA results.
        
        Returns:
            str: A formatted string containing the PCA results summary
        """
        base_summary = super().summary()
        
        # Add component information
        comp_info = f"Number of Components: {self.n_components}\n\n"
        
        # Add explained variance table
        var_table = "Explained Variance:\n"
        var_table += "-" * 80 + "\n"
        var_table += f"{'Component':<12} {'Eigenvalue':<15} {'Variance %':<15} {'Cumulative %':<15}\n"
        var_table += "-" * 80 + "\n"
        
        for i in range(len(self.explained_variance)):
            var_table += f"PC{i+1:<11} {self.explained_variance[i]:<15.6f} "
            var_table += f"{self.explained_variance_ratio[i] * 100:<15.2f} "
            var_table += f"{self.cumulative_explained_variance[i] * 100:<15.2f}\n"
        
        var_table += "-" * 80 + "\n\n"
        
        # Add loadings table
        loadings_table = "Component Loadings:\n"
        loadings_table += "-" * 80 + "\n"
        
        # Create header
        header = "Variable"
        for i in range(min(5, self.n_components)):
            header += f"{'':<5}PC{i+1:<10}"
        loadings_table += f"{header}\n"
        loadings_table += "-" * 80 + "\n"
        
        # Add loadings for each variable
        for i, var_name in enumerate(self.variable_names):
            row = f"{var_name:<12}"
            for j in range(min(5, self.n_components)):
                row += f"{'':<5}{self.components[j, i]:<10.6f}"
            loadings_table += f"{row}\n"
        
        if self.n_components > 5:
            loadings_table += "... (showing only first 5 components)\n"
        
        loadings_table += "-" * 80 + "\n"
        
        return base_summary + comp_info + var_table + loadings_table
    
    def to_dataframe(self, what: Literal['loadings', 'variance'] = 'loadings') -> pd.DataFrame:
        """Convert PCA results to a pandas DataFrame.
        
        Args:
            what: What to return ('loadings' or 'variance')
        
        Returns:
            pd.DataFrame: DataFrame containing PCA results
        
        Raises:
            ValueError: If what is not 'loadings' or 'variance'
        """
        if what == 'loadings':
            # Create DataFrame with loadings
            component_names = [f"PC{i+1}" for i in range(self.n_components)]
            
            # Transpose components to get variables as rows and components as columns
            df = pd.DataFrame(
                self.components.T,
                index=self.variable_names,
                columns=component_names
            )
            
            return df
        
        elif what == 'variance':
            # Create DataFrame with variance information
            component_names = [f"PC{i+1}" for i in range(len(self.explained_variance))]
            
            df = pd.DataFrame({
                "Component": component_names,
                "Eigenvalue": self.explained_variance,
                "Variance_Ratio": self.explained_variance_ratio,
                "Cumulative_Variance": self.cumulative_explained_variance
            })
            
            return df
        
        else:
            raise ValueError(f"what must be 'loadings' or 'variance', got {what}")
    
    def plot_scree(self, **kwargs: Any) -> Any:
        """Plot the scree plot (explained variance).
        
        Args:
            **kwargs: Additional keyword arguments for plotting
        
        Returns:
            Any: Plot object
        
        Raises:
            ImportError: If matplotlib is not installed
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib is required for plotting")
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot eigenvalues
        ax1.plot(
            range(1, len(self.explained_variance) + 1),
            self.explained_variance,
            'bo-',
            linewidth=2,
            markersize=8,
            label='Eigenvalue'
        )
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Eigenvalue', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        # Create second y-axis for cumulative explained variance
        ax2 = ax1.twinx()
        ax2.plot(
            range(1, len(self.cumulative_explained_variance) + 1),
            self.cumulative_explained_variance * 100,
            'ro-',
            linewidth=2,
            markersize=8,
            label='Cumulative Explained Variance'
        )
        ax2.set_ylabel('Cumulative Explained Variance (%)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Add grid
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Add title
        plt.title('Scree Plot')
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
        
        plt.tight_layout()
        
        return fig
    
    def plot_loadings(self, components: Optional[List[int]] = None, **kwargs: Any) -> Any:
        """Plot the component loadings.
        
        Args:
            components: List of component indices to plot (0-based)
            **kwargs: Additional keyword arguments for plotting
        
        Returns:
            Any: Plot object
        
        Raises:
            ImportError: If matplotlib is not installed
            ValueError: If components contains invalid indices
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib is required for plotting")
        
        # Default to first two components
        if components is None:
            components = [0, 1]
        
        # Validate component indices
        for comp in components:
            if comp < 0 or comp >= self.n_components:
                raise ValueError(
                    f"Component index {comp} is out of range [0, {self.n_components - 1}]"
                )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot loadings
        if len(components) == 2:
            # 2D loading plot
            x_comp, y_comp = components
            
            # Plot points
            ax.scatter(
                self.components[x_comp, :],
                self.components[y_comp, :],
                s=100,
                alpha=0.7
            )
            
            # Add variable labels
            for i, var_name in enumerate(self.variable_names):
                ax.annotate(
                    var_name,
                    (self.components[x_comp, i], self.components[y_comp, i]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    ha='left',
                    va='bottom'
                )
            
            # Add axis lines
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            
            # Add labels
            ax.set_xlabel(f"PC{x_comp + 1} ({self.explained_variance_ratio[x_comp] * 100:.2f}%)")
            ax.set_ylabel(f"PC{y_comp + 1} ({self.explained_variance_ratio[y_comp] * 100:.2f}%)")
            
            # Add title
            ax.set_title('Component Loadings')
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
        else:
            # Bar plot for multiple components
            n_vars = len(self.variable_names)
            n_comps = len(components)
            
            # Set up bar positions
            bar_width = 0.8 / n_comps
            positions = np.arange(n_vars)
            
            # Plot bars for each component
            for i, comp in enumerate(components):
                ax.bar(
                    positions + i * bar_width - (n_comps - 1) * bar_width / 2,
                    self.components[comp, :],
                    width=bar_width,
                    label=f"PC{comp + 1}"
                )
            
            # Add labels
            ax.set_xlabel('Variable')
            ax.set_ylabel('Loading')
            ax.set_title('Component Loadings')
            ax.set_xticks(positions)
            ax.set_xticklabels(self.variable_names, rotation=45, ha='right')
            ax.legend()
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        plt.tight_layout()
        
        return fig
