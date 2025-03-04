# mfe/models/cross_section/pca.py
"""
Principal Component Analysis (PCA) Module

This module implements Principal Component Analysis (PCA) for cross-sectional data
with multiple computation modes. It provides a flexible PCA implementation that
handles data transformations, computes principal components, and returns detailed
results including eigenvalues, eigenvectors, and variance explained.

The implementation extends NumPy's linear algebra functionality with specialized
PCA methods for financial applications, supporting different computation modes:
- 'outer': Based on the outer product of the data matrix (X'X)
- 'cov': Based on the covariance matrix
- 'corr': Based on the correlation matrix

The module provides both synchronous and asynchronous interfaces for handling
large datasets efficiently, with comprehensive error handling and input validation.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union, cast, Any

import numpy as np
import pandas as pd
from scipy import linalg

from mfe.core.base import CrossSectionalModelBase, ModelResult
from mfe.core.exceptions import (
    DimensionError, ParameterError, NumericError, DataError,
    raise_dimension_error, raise_parameter_error, raise_numeric_error, raise_data_error
)
from mfe.core.types import Matrix, Vector, TimeSeriesDataFrame
from mfe.core.validation import (
    validate_matrix_shape, validate_input_type, validate_input_bounds,
    validate_input_matrix, validate_input_numeric_array
)
from mfe.utils.matrix_ops import ensure_symmetric

# Set up module-level logger
logger = logging.getLogger("mfe.models.cross_section.pca")

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
    logger.debug("Numba available for PCA acceleration")
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    HAS_NUMBA = False
    logger.info("Numba not available. PCA will use pure NumPy implementations.")


@dataclass
class PCAResult(ModelResult):
    """
    Result container for Principal Component Analysis.
    
    This class stores the results of a PCA computation, including eigenvalues,
    eigenvectors, principal components, and variance explained.
    
    Attributes:
        model_name: Name of the model ("PCA")
        n_components: Number of principal components retained
        eigenvalues: Eigenvalues of the decomposition
        eigenvectors: Eigenvectors (loadings) of the decomposition
        components: Principal components (scores)
        explained_variance: Variance explained by each component
        explained_variance_ratio: Proportion of variance explained by each component
        cumulative_explained_variance: Cumulative variance explained
        mean: Mean of the original data (used for centering)
        std: Standard deviation of the original data (used for scaling in 'corr' mode)
        computation_mode: Mode used for computation ('outer', 'cov', or 'corr')
        n_samples: Number of samples in the original data
        n_features: Number of features in the original data
    """
    
    n_components: int
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_explained_variance: np.ndarray
    mean: np.ndarray
    std: Optional[np.ndarray] = None
    computation_mode: str = "outer"
    n_samples: int = 0
    n_features: int = 0
    
    def __post_init__(self) -> None:
        """Validate the PCA result after initialization."""
        super().__post_init__()
        
        # Ensure eigenvalues are sorted in descending order
        if not np.all(np.diff(self.eigenvalues) <= 0):
            logger.warning("Eigenvalues are not sorted in descending order. Sorting now.")
            # Get sorting indices
            idx = np.argsort(self.eigenvalues)[::-1]
            # Sort eigenvalues and eigenvectors
            self.eigenvalues = self.eigenvalues[idx]
            self.eigenvectors = self.eigenvectors[:, idx]
            # Sort components if available
            if self.components is not None:
                self.components = self.components[:, idx]
            # Sort explained variance and ratio
            self.explained_variance = self.explained_variance[idx]
            self.explained_variance_ratio = self.explained_variance_ratio[idx]
            self.cumulative_explained_variance = np.cumsum(self.explained_variance_ratio)
    
    def summary(self) -> str:
        """
        Generate a text summary of the PCA results.
        
        Returns:
            str: A formatted string containing the PCA results summary
        """
        header = f"Principal Component Analysis ({self.computation_mode} mode)\n"
        header += "=" * len(header) + "\n\n"
        
        info = f"Number of samples: {self.n_samples}\n"
        info += f"Number of features: {self.n_features}\n"
        info += f"Number of components: {self.n_components}\n\n"
        
        variance_table = "Explained Variance:\n"
        variance_table += "-" * 80 + "\n"
        variance_table += "Component | Eigenvalue | Variance Explained | Ratio | Cumulative\n"
        variance_table += "-" * 80 + "\n"
        
        # Display information for each component
        for i in range(min(self.n_components, 10)):  # Show at most 10 components
            variance_table += f"{i+1:9d} | {self.eigenvalues[i]:10.4f} | "
            variance_table += f"{self.explained_variance[i]:17.4f} | "
            variance_table += f"{self.explained_variance_ratio[i]:5.2%} | "
            variance_table += f"{self.cumulative_explained_variance[i]:9.2%}\n"
        
        # If there are more than 10 components, add an ellipsis
        if self.n_components > 10:
            variance_table += "...\n"
        
        variance_table += "-" * 80 + "\n\n"
        
        # Add information about top loadings for first few components
        loadings_info = "Top Feature Loadings (absolute values):\n"
        loadings_info += "-" * 80 + "\n"
        
        n_display = min(self.n_components, 5)  # Show at most 5 components
        n_top_features = min(self.n_features, 5)  # Show at most 5 features per component
        
        for i in range(n_display):
            loadings = self.eigenvectors[:, i]
            # Get indices of top features by absolute loading value
            top_indices = np.argsort(np.abs(loadings))[::-1][:n_top_features]
            
            loadings_info += f"Component {i+1}:\n"
            for idx in top_indices:
                loadings_info += f"  Feature {idx+1}: {loadings[idx]:+.4f}\n"
            loadings_info += "\n"
        
        return header + info + variance_table + loadings_info
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the PCA result to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the PCA result
        """
        result = super().to_dict()
        result.update({
            "n_components": self.n_components,
            "eigenvalues": self.eigenvalues,
            "eigenvectors": self.eigenvectors,
            "components": self.components,
            "explained_variance": self.explained_variance,
            "explained_variance_ratio": self.explained_variance_ratio,
            "cumulative_explained_variance": self.cumulative_explained_variance,
            "mean": self.mean,
            "std": self.std,
            "computation_mode": self.computation_mode,
            "n_samples": self.n_samples,
            "n_features": self.n_features
        })
        return result


class PCA(CrossSectionalModelBase):
    """
    Principal Component Analysis implementation.
    
    This class implements Principal Component Analysis (PCA) with support for
    different computation modes and data transformations. It provides methods
    for fitting PCA models, transforming data, and computing inverse transforms.
    
    Attributes:
        n_components: Number of components to retain (default: None, keep all)
        computation_mode: Mode for computing PCA ('outer', 'cov', or 'corr')
        center: Whether to center the data (subtract mean)
        scale: Whether to scale the data (divide by std. dev.)
        tol: Tolerance for small eigenvalues
        results: PCA results after fitting
    """
    
    def __init__(
        self,
        n_components: Optional[int] = None,
        computation_mode: Literal["outer", "cov", "corr"] = "outer",
        center: bool = True,
        scale: bool = False,
        tol: float = 1e-10,
        name: str = "PCA"
    ) -> None:
        """
        Initialize the PCA model.
        
        Args:
            n_components: Number of components to retain (default: None, keep all)
            computation_mode: Mode for computing PCA:
                - 'outer': Based on the outer product (X'X)
                - 'cov': Based on the covariance matrix
                - 'corr': Based on the correlation matrix
            center: Whether to center the data (subtract mean)
            scale: Whether to scale the data (divide by std. dev.)
            tol: Tolerance for small eigenvalues
            name: Name of the model
            
        Raises:
            ParameterError: If parameters are invalid
        """
        super().__init__(name=name)
        
        # Validate computation mode
        if computation_mode not in ["outer", "cov", "corr"]:
            raise_parameter_error(
                f"Invalid computation mode: {computation_mode}. "
                "Must be one of 'outer', 'cov', or 'corr'.",
                param_name="computation_mode",
                param_value=computation_mode,
                constraint="one of 'outer', 'cov', or 'corr'"
            )
        
        # Validate tolerance
        if tol <= 0:
            raise_parameter_error(
                f"Tolerance must be positive, got {tol}",
                param_name="tol",
                param_value=tol,
                constraint="> 0"
            )
        
        # Set attributes
        self.n_components = n_components
        self.computation_mode = computation_mode
        self.center = center
        self.scale = scale
        self.tol = tol
        
        # If computation_mode is 'corr', force center and scale to True
        if self.computation_mode == "corr":
            if not center or not scale:
                logger.info(
                    "Setting center=True and scale=True for 'corr' computation mode"
                )
                self.center = True
                self.scale = True
        
        # Initialize results
        self._results = None
    
    def validate_data(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Validate the input data for PCA.
        
        Args:
            data: Input data matrix (samples × features)
            
        Returns:
            np.ndarray: Validated data as a NumPy array
            
        Raises:
            TypeError: If data is not a NumPy array or Pandas DataFrame
            DimensionError: If data is not 2-dimensional
            DataError: If data contains NaN or infinite values
        """
        # Convert to numpy array if needed
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        elif isinstance(data, np.ndarray):
            data_array = data
        else:
            raise TypeError(
                f"Data must be a NumPy array or Pandas DataFrame, got {type(data).__name__}"
            )
        
        # Check dimensions
        if data_array.ndim != 2:
            raise_dimension_error(
                f"Data must be 2-dimensional, got {data_array.ndim} dimensions",
                array_name="data",
                expected_shape="(n_samples, n_features)",
                actual_shape=data_array.shape
            )
        
        # Check for NaN or infinite values
        if np.isnan(data_array).any():
            raise_data_error(
                "Data contains NaN values",
                data_name="data",
                issue="contains NaN values"
            )
        
        if np.isinf(data_array).any():
            raise_data_error(
                "Data contains infinite values",
                data_name="data",
                issue="contains infinite values"
            )
        
        return data_array
    
    def _preprocess_data(
        self, 
        data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess the data by centering and/or scaling.
        
        Args:
            data: Input data matrix (samples × features)
            
        Returns:
            Tuple containing:
                - Preprocessed data
                - Mean of the original data (for centering)
                - Standard deviation of the original data (for scaling, or None if not scaled)
        """
        n_samples, n_features = data.shape
        
        # Compute mean if centering
        if self.center:
            mean = np.mean(data, axis=0)
        else:
            mean = np.zeros(n_features)
        
        # Compute standard deviation if scaling
        if self.scale:
            std = np.std(data, axis=0, ddof=1)
            # Check for zero standard deviation
            if np.any(std == 0):
                zero_std_features = np.where(std == 0)[0]
                raise_numeric_error(
                    f"Features {zero_std_features} have zero standard deviation, "
                    "cannot scale. Consider removing these features.",
                    operation="PCA preprocessing",
                    values=std,
                    error_type="zero_std_dev"
                )
        else:
            std = None
        
        # Apply preprocessing
        processed_data = data.copy()
        
        if self.center:
            processed_data -= mean
        
        if self.scale:
            processed_data /= std
        
        return processed_data, mean, std
    
    def _compute_pca_outer(
        self, 
        data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute PCA using the outer product method (X'X).
        
        This method is more efficient when n_samples > n_features.
        
        Args:
            data: Preprocessed data matrix (samples × features)
            
        Returns:
            Tuple containing:
                - Eigenvalues
                - Eigenvectors (loadings)
                - Principal components (scores)
        """
        n_samples, n_features = data.shape
        
        # Compute outer product (X'X)
        outer_product = data.T @ data
        
        # Ensure symmetry (for numerical stability)
        outer_product = ensure_symmetric(outer_product)
        
        # Compute eigendecomposition
        eigenvalues, eigenvectors = linalg.eigh(outer_product)
        
        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Scale eigenvalues by 1/n_samples to get variance
        eigenvalues = eigenvalues / n_samples
        
        # Compute principal components (scores)
        components = data @ eigenvectors
        
        return eigenvalues, eigenvectors, components
    
    def _compute_pca_cov(
        self, 
        data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute PCA using the covariance matrix method.
        
        Args:
            data: Preprocessed data matrix (samples × features)
            
        Returns:
            Tuple containing:
                - Eigenvalues
                - Eigenvectors (loadings)
                - Principal components (scores)
        """
        n_samples, n_features = data.shape
        
        # Compute covariance matrix
        cov_matrix = np.cov(data, rowvar=False, ddof=1)
        
        # Ensure symmetry (for numerical stability)
        cov_matrix = ensure_symmetric(cov_matrix)
        
        # Compute eigendecomposition
        eigenvalues, eigenvectors = linalg.eigh(cov_matrix)
        
        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Compute principal components (scores)
        components = data @ eigenvectors
        
        return eigenvalues, eigenvectors, components
    
    def _compute_pca_corr(
        self, 
        data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute PCA using the correlation matrix method.
        
        Args:
            data: Preprocessed data matrix (samples × features)
            
        Returns:
            Tuple containing:
                - Eigenvalues
                - Eigenvectors (loadings)
                - Principal components (scores)
        """
        n_samples, n_features = data.shape
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(data, rowvar=False)
        
        # Ensure symmetry (for numerical stability)
        corr_matrix = ensure_symmetric(corr_matrix)
        
        # Compute eigendecomposition
        eigenvalues, eigenvectors = linalg.eigh(corr_matrix)
        
        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Compute principal components (scores)
        components = data @ eigenvectors
        
        return eigenvalues, eigenvectors, components
    
    @validate_input_type(0, (np.ndarray, pd.DataFrame))
    def fit(
        self, 
        data: Union[np.ndarray, pd.DataFrame],
        **kwargs: Any
    ) -> PCAResult:
        """
        Fit the PCA model to the data.
        
        Args:
            data: Input data matrix (samples × features)
            **kwargs: Additional keyword arguments (not used)
            
        Returns:
            PCAResult: PCA results
            
        Raises:
            TypeError: If data is not a NumPy array or Pandas DataFrame
            DimensionError: If data is not 2-dimensional
            DataError: If data contains NaN or infinite values
            NumericError: If eigendecomposition fails
        """
        # Validate data
        data_array = self.validate_data(data)
        n_samples, n_features = data_array.shape
        
        # Validate n_components
        if self.n_components is not None:
            if not isinstance(self.n_components, int):
                raise_parameter_error(
                    f"n_components must be an integer, got {type(self.n_components).__name__}",
                    param_name="n_components",
                    param_value=self.n_components
                )
            if self.n_components < 1 or self.n_components > min(n_samples, n_features):
                raise_parameter_error(
                    f"n_components must be between 1 and min(n_samples, n_features) = "
                    f"{min(n_samples, n_features)}, got {self.n_components}",
                    param_name="n_components",
                    param_value=self.n_components,
                    constraint=f"between 1 and {min(n_samples, n_features)}"
                )
        
        # Preprocess data
        processed_data, mean, std = self._preprocess_data(data_array)
        
        # Compute PCA based on the selected mode
        try:
            if self.computation_mode == "outer":
                eigenvalues, eigenvectors, components = self._compute_pca_outer(processed_data)
            elif self.computation_mode == "cov":
                eigenvalues, eigenvectors, components = self._compute_pca_cov(processed_data)
            elif self.computation_mode == "corr":
                eigenvalues, eigenvectors, components = self._compute_pca_corr(processed_data)
            else:
                # This should never happen due to validation in __init__
                raise ValueError(f"Invalid computation mode: {self.computation_mode}")
        except Exception as e:
            raise_numeric_error(
                f"PCA computation failed: {str(e)}",
                operation=f"PCA {self.computation_mode}",
                error_type="eigendecomposition_failed"
            ) from e
        
        # Filter out small eigenvalues
        valid_eigenvalues = eigenvalues > self.tol * eigenvalues[0]
        eigenvalues = eigenvalues[valid_eigenvalues]
        eigenvectors = eigenvectors[:, valid_eigenvalues]
        components = components[:, valid_eigenvalues]
        
        # Determine number of components to keep
        max_components = len(eigenvalues)
        if self.n_components is None:
            n_components = max_components
        else:
            n_components = min(self.n_components, max_components)
        
        # Truncate to the desired number of components
        eigenvalues = eigenvalues[:n_components]
        eigenvectors = eigenvectors[:, :n_components]
        components = components[:, :n_components]
        
        # Compute explained variance and ratios
        explained_variance = eigenvalues
        explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
        cumulative_explained_variance = np.cumsum(explained_variance_ratio)
        
        # Create result object
        self._results = PCAResult(
            model_name=self.name,
            n_components=n_components,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            components=components,
            explained_variance=explained_variance,
            explained_variance_ratio=explained_variance_ratio,
            cumulative_explained_variance=cumulative_explained_variance,
            mean=mean,
            std=std,
            computation_mode=self.computation_mode,
            n_samples=n_samples,
            n_features=n_features,
            convergence=True,
            iterations=1
        )
        
        self._fitted = True
        
        return cast(PCAResult, self._results)
    
    async def fit_async(
        self, 
        data: Union[np.ndarray, pd.DataFrame],
        **kwargs: Any
    ) -> PCAResult:
        """
        Asynchronously fit the PCA model to the data.
        
        This method provides an asynchronous interface to the fit method,
        allowing for non-blocking PCA computation in UI contexts.
        
        Args:
            data: Input data matrix (samples × features)
            **kwargs: Additional keyword arguments (not used)
            
        Returns:
            PCAResult: PCA results
            
        Raises:
            TypeError: If data is not a NumPy array or Pandas DataFrame
            DimensionError: If data is not 2-dimensional
            DataError: If data contains NaN or infinite values
            NumericError: If eigendecomposition fails
        """
        # This is a simple implementation that just calls the synchronous version
        # In a real-world scenario, you might want to implement a truly asynchronous
        # version for very large datasets
        import asyncio
        
        # Run the synchronous fit method in a thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: self.fit(data, **kwargs))
        
        return result
    
    @validate_input_type(0, (np.ndarray, pd.DataFrame))
    def transform(
        self, 
        data: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """
        Transform data using the fitted PCA model.
        
        Args:
            data: Input data matrix (samples × features)
            
        Returns:
            np.ndarray: Transformed data (principal components)
            
        Raises:
            RuntimeError: If the model has not been fitted
            TypeError: If data is not a NumPy array or Pandas DataFrame
            DimensionError: If data does not have the correct number of features
        """
        if not self._fitted or self._results is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        # Convert to numpy array if needed
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = data
        
        # Check dimensions
        if data_array.ndim != 2:
            raise_dimension_error(
                f"Data must be 2-dimensional, got {data_array.ndim} dimensions",
                array_name="data",
                expected_shape="(n_samples, n_features)",
                actual_shape=data_array.shape
            )
        
        if data_array.shape[1] != self._results.n_features:
            raise_dimension_error(
                f"Data has {data_array.shape[1]} features, but the model was trained with "
                f"{self._results.n_features} features",
                array_name="data",
                expected_shape=f"(n_samples, {self._results.n_features})",
                actual_shape=data_array.shape
            )
        
        # Preprocess data
        processed_data = data_array.copy()
        
        if self.center:
            processed_data -= self._results.mean
        
        if self.scale and self._results.std is not None:
            processed_data /= self._results.std
        
        # Transform data
        transformed_data = processed_data @ self._results.eigenvectors
        
        return transformed_data
    
    @validate_input_type(0, np.ndarray)
    def inverse_transform(
        self, 
        components: np.ndarray
    ) -> np.ndarray:
        """
        Inverse transform principal components back to original space.
        
        Args:
            components: Principal components (samples × n_components)
            
        Returns:
            np.ndarray: Reconstructed data in original space
            
        Raises:
            RuntimeError: If the model has not been fitted
            TypeError: If components is not a NumPy array
            DimensionError: If components does not have the correct shape
        """
        if not self._fitted or self._results is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        # Check dimensions
        if components.ndim != 2:
            raise_dimension_error(
                f"Components must be 2-dimensional, got {components.ndim} dimensions",
                array_name="components",
                expected_shape=f"(n_samples, {self._results.n_components})",
                actual_shape=components.shape
            )
        
        if components.shape[1] > self._results.n_components:
            raise_dimension_error(
                f"Components has {components.shape[1]} columns, but the model has only "
                f"{self._results.n_components} components",
                array_name="components",
                expected_shape=f"(n_samples, <= {self._results.n_components})",
                actual_shape=components.shape
            )
        
        # Inverse transform
        reconstructed_data = components @ self._results.eigenvectors[:, :components.shape[1]].T
        
        # Undo preprocessing
        if self.scale and self._results.std is not None:
            reconstructed_data *= self._results.std
        
        if self.center:
            reconstructed_data += self._results.mean
        
        return reconstructed_data
    
    def get_components(
        self, 
        n_components: Optional[int] = None
    ) -> np.ndarray:
        """
        Get the principal components from the fitted model.
        
        Args:
            n_components: Number of components to return (default: all)
            
        Returns:
            np.ndarray: Principal components
            
        Raises:
            RuntimeError: If the model has not been fitted
            ParameterError: If n_components is invalid
        """
        if not self._fitted or self._results is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        if n_components is None:
            return self._results.components
        
        if not isinstance(n_components, int) or n_components < 1:
            raise_parameter_error(
                f"n_components must be a positive integer, got {n_components}",
                param_name="n_components",
                param_value=n_components,
                constraint="positive integer"
            )
        
        if n_components > self._results.n_components:
            raise_parameter_error(
                f"n_components must be <= {self._results.n_components}, got {n_components}",
                param_name="n_components",
                param_value=n_components,
                constraint=f"<= {self._results.n_components}"
            )
        
        return self._results.components[:, :n_components]
    
    def get_loadings(
        self, 
        n_components: Optional[int] = None
    ) -> np.ndarray:
        """
        Get the loadings (eigenvectors) from the fitted model.
        
        Args:
            n_components: Number of components to return (default: all)
            
        Returns:
            np.ndarray: Loadings (eigenvectors)
            
        Raises:
            RuntimeError: If the model has not been fitted
            ParameterError: If n_components is invalid
        """
        if not self._fitted or self._results is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        if n_components is None:
            return self._results.eigenvectors
        
        if not isinstance(n_components, int) or n_components < 1:
            raise_parameter_error(
                f"n_components must be a positive integer, got {n_components}",
                param_name="n_components",
                param_value=n_components,
                constraint="positive integer"
            )
        
        if n_components > self._results.n_components:
            raise_parameter_error(
                f"n_components must be <= {self._results.n_components}, got {n_components}",
                param_name="n_components",
                param_value=n_components,
                constraint=f"<= {self._results.n_components}"
            )
        
        return self._results.eigenvectors[:, :n_components]
    
    def get_explained_variance(
        self, 
        n_components: Optional[int] = None
    ) -> np.ndarray:
        """
        Get the explained variance from the fitted model.
        
        Args:
            n_components: Number of components to return (default: all)
            
        Returns:
            np.ndarray: Explained variance
            
        Raises:
            RuntimeError: If the model has not been fitted
            ParameterError: If n_components is invalid
        """
        if not self._fitted or self._results is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        if n_components is None:
            return self._results.explained_variance
        
        if not isinstance(n_components, int) or n_components < 1:
            raise_parameter_error(
                f"n_components must be a positive integer, got {n_components}",
                param_name="n_components",
                param_value=n_components,
                constraint="positive integer"
            )
        
        if n_components > self._results.n_components:
            raise_parameter_error(
                f"n_components must be <= {self._results.n_components}, got {n_components}",
                param_name="n_components",
                param_value=n_components,
                constraint=f"<= {self._results.n_components}"
            )
        
        return self._results.explained_variance[:n_components]
    
    def get_explained_variance_ratio(
        self, 
        n_components: Optional[int] = None
    ) -> np.ndarray:
        """
        Get the explained variance ratio from the fitted model.
        
        Args:
            n_components: Number of components to return (default: all)
            
        Returns:
            np.ndarray: Explained variance ratio
            
        Raises:
            RuntimeError: If the model has not been fitted
            ParameterError: If n_components is invalid
        """
        if not self._fitted or self._results is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        if n_components is None:
            return self._results.explained_variance_ratio
        
        if not isinstance(n_components, int) or n_components < 1:
            raise_parameter_error(
                f"n_components must be a positive integer, got {n_components}",
                param_name="n_components",
                param_value=n_components,
                constraint="positive integer"
            )
        
        if n_components > self._results.n_components:
            raise_parameter_error(
                f"n_components must be <= {self._results.n_components}, got {n_components}",
                param_name="n_components",
                param_value=n_components,
                constraint=f"<= {self._results.n_components}"
            )
        
        return self._results.explained_variance_ratio[:n_components]
    
    def get_cumulative_explained_variance(
        self, 
        n_components: Optional[int] = None
    ) -> np.ndarray:
        """
        Get the cumulative explained variance from the fitted model.
        
        Args:
            n_components: Number of components to return (default: all)
            
        Returns:
            np.ndarray: Cumulative explained variance
            
        Raises:
            RuntimeError: If the model has not been fitted
            ParameterError: If n_components is invalid
        """
        if not self._fitted or self._results is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        if n_components is None:
            return self._results.cumulative_explained_variance
        
        if not isinstance(n_components, int) or n_components < 1:
            raise_parameter_error(
                f"n_components must be a positive integer, got {n_components}",
                param_name="n_components",
                param_value=n_components,
                constraint="positive integer"
            )
        
        if n_components > self._results.n_components:
            raise_parameter_error(
                f"n_components must be <= {self._results.n_components}, got {n_components}",
                param_name="n_components",
                param_value=n_components,
                constraint=f"<= {self._results.n_components}"
            )
        
        return self._results.cumulative_explained_variance[:n_components]
    
    def find_n_components(
        self, 
        explained_variance_threshold: float = 0.95
    ) -> int:
        """
        Find the number of components needed to explain a certain amount of variance.
        
        Args:
            explained_variance_threshold: Threshold for cumulative explained variance
                                         (between 0 and 1)
            
        Returns:
            int: Number of components needed
            
        Raises:
            RuntimeError: If the model has not been fitted
            ParameterError: If explained_variance_threshold is invalid
        """
        if not self._fitted or self._results is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        if not 0 < explained_variance_threshold <= 1:
            raise_parameter_error(
                f"explained_variance_threshold must be between 0 and 1, got {explained_variance_threshold}",
                param_name="explained_variance_threshold",
                param_value=explained_variance_threshold,
                constraint="between 0 and 1"
            )
        
        # Find the first index where cumulative explained variance exceeds the threshold
        n_components = np.argmax(self._results.cumulative_explained_variance >= explained_variance_threshold) + 1
        
        return n_components
    
    def simulate(
        self, 
        n_periods: int, 
        burn: int = 0, 
        initial_values: Optional[np.ndarray] = None,
        random_state: Optional[Union[int, np.random.Generator]] = None,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Simulate data from the PCA model.
        
        This method generates random data in the principal component space and
        transforms it back to the original space.
        
        Args:
            n_periods: Number of periods to simulate
            burn: Number of initial observations to discard (not used)
            initial_values: Initial values (not used)
            random_state: Random number generator or seed
            **kwargs: Additional keyword arguments (not used)
            
        Returns:
            np.ndarray: Simulated data
            
        Raises:
            RuntimeError: If the model has not been fitted
            ParameterError: If n_periods is invalid
        """
        if not self._fitted or self._results is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        if not isinstance(n_periods, int) or n_periods < 1:
            raise_parameter_error(
                f"n_periods must be a positive integer, got {n_periods}",
                param_name="n_periods",
                param_value=n_periods,
                constraint="positive integer"
            )
        
        # Set random state
        if isinstance(random_state, np.random.Generator):
            rng = random_state
        else:
            rng = np.random.default_rng(random_state)
        
        # Generate random components with the correct variance structure
        random_components = rng.standard_normal((n_periods, self._results.n_components))
        random_components *= np.sqrt(self._results.explained_variance)
        
        # Transform back to original space
        simulated_data = self.inverse_transform(random_components)
        
        return simulated_data
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using the fitted PCA model (alias for transform).
        
        This method is provided for compatibility with the scikit-learn API.
        
        Args:
            X: Input data matrix (samples × features)
            
        Returns:
            np.ndarray: Transformed data (principal components)
            
        Raises:
            RuntimeError: If the model has not been fitted
        """
        return self.transform(X)

# mfe/models/cross_section/pca.py
"""
Principal Component Analysis (PCA) Module

This module implements Principal Component Analysis (PCA) for cross-sectional data
with multiple computation modes. It provides a flexible PCA implementation that
handles data transformations, computes principal components, and returns detailed
results including eigenvalues, eigenvectors, and variance explained.

The implementation extends NumPy's linear algebra functionality with specialized
PCA methods for financial applications, supporting different computation modes:
- 'outer': Based on the outer product of the data matrix (X'X)
- 'cov': Based on the covariance matrix
- 'corr': Based on the correlation matrix

The module provides both synchronous and asynchronous interfaces for handling
large datasets efficiently, with comprehensive error handling and input validation.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union, cast, Any

import numpy as np
import pandas as pd
from scipy import linalg

from mfe.core.base import CrossSectionalModelBase, ModelResult
from mfe.core.exceptions import (
    DimensionError, ParameterError, NumericError, DataError,
    raise_dimension_error, raise_parameter_error, raise_numeric_error, raise_data_error
)
from mfe.core.types import Matrix, Vector, TimeSeriesDataFrame
from mfe.core.validation import (
    validate_matrix_shape, validate_input_type, validate_input_bounds,
    validate_input_matrix, validate_input_numeric_array
)
from mfe.utils.matrix_ops import ensure_symmetric

# Set up module-level logger
logger = logging.getLogger("mfe.models.cross_section.pca")

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
    logger.debug("Numba available for PCA acceleration")
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    HAS_NUMBA = False
    logger.info("Numba not available. PCA will use pure NumPy implementations.")


@dataclass
class PCAResult(ModelResult):
    """
    Result container for Principal Component Analysis.
    
    This class stores the results of a PCA computation, including eigenvalues,
    eigenvectors, principal components, and variance explained.
    
    Attributes:
        model_name: Name of the model ("PCA")
        n_components: Number of principal components retained
        eigenvalues: Eigenvalues of the decomposition
        eigenvectors: Eigenvectors (loadings) of the decomposition
        components: Principal components (scores)
        explained_variance: Variance explained by each component
        explained_variance_ratio: Proportion of variance explained by each component
        cumulative_explained_variance: Cumulative variance explained
        mean: Mean of the original data (used for centering)
        std: Standard deviation of the original data (used for scaling in 'corr' mode)
        computation_mode: Mode used for computation ('outer', 'cov', or 'corr')
        n_samples: Number of samples in the original data
        n_features: Number of features in the original data
    """
    
    n_components: int
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_explained_variance: np.ndarray
    mean: np.ndarray
    std: Optional[np.ndarray] = None
    computation_mode: str = "outer"
    n_samples: int = 0
    n_features: int = 0
    
    def __post_init__(self) -> None:
        """Validate the PCA result after initialization."""
        super().__post_init__()
        
        # Ensure eigenvalues are sorted in descending order
        if not np.all(np.diff(self.eigenvalues) <= 0):
            logger.warning("Eigenvalues are not sorted in descending order. Sorting now.")
            # Get sorting indices
            idx = np.argsort(self.eigenvalues)[::-1]
            # Sort eigenvalues and eigenvectors
            self.eigenvalues = self.eigenvalues[idx]
            self.eigenvectors = self.eigenvectors[:, idx]
            # Sort components if available
            if self.components is not None:
                self.components = self.components[:, idx]
            # Sort explained variance and ratio
            self.explained_variance = self.explained_variance[idx]
            self.explained_variance_ratio = self.explained_variance_ratio[idx]
            self.cumulative_explained_variance = np.cumsum(self.explained_variance_ratio)
    
    def summary(self) -> str:
        """
        Generate a text summary of the PCA results.
        
        Returns:
            str: A formatted string containing the PCA results summary
        """
        header = f"Principal Component Analysis ({self.computation_mode} mode)\n"
        header += "=" * len(header) + "\n\n"
        
        info = f"Number of samples: {self.n_samples}\n"
        info += f"Number of features: {self.n_features}\n"
        info += f"Number of components: {self.n_components}\n\n"
        
        variance_table = "Explained Variance:\n"
        variance_table += "-" * 80 + "\n"
        variance_table += "Component | Eigenvalue | Variance Explained | Ratio | Cumulative\n"
        variance_table += "-" * 80 + "\n"
        
        # Display information for each component
        for i in range(min(self.n_components, 10)):
            variance_table += f"{i+1:9d} | {self.eigenvalues[i]:10.4f} | "
            variance_table += f"{self.explained_variance[i]:17.4f} | "
            variance_table += f"{self.explained_variance_ratio[i]:5.2%} | "
            variance_table += f"{self.cumulative_explained_variance[i]:9.2%}\n"
        
        if self.n_components > 10:
            variance_table += "...\n"
        
        variance_table += "-" * 80 + "\n\n"
        
        loadings_info = "Top Feature Loadings (absolute values):\n"
        loadings_info += "-" * 80 + "\n"
        
        n_display = min(self.n_components, 5)
        n_top_features = min(self.n_features, 5)
        
        for i in range(n_display):
            loadings = self.eigenvectors[:, i]
            top_indices = np.argsort(np.abs(loadings))[::-1][:n_top_features]
            
            loadings_info += f"Component {i+1}:\n"
            for idx in top_indices:
                loadings_info += f"  Feature {idx+1}: {loadings[idx]:+.4f}\n"
            loadings_info += "\n"
        
        return header + info + variance_table + loadings_info
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the PCA result to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the PCA result
        """
        result = super().to_dict()
        result.update({
            "n_components": self.n_components,
            "eigenvalues": self.eigenvalues,
            "eigenvectors": self.eigenvectors,
            "components": self.components,
            "explained_variance": self.explained_variance,
            "explained_variance_ratio": self.explained_variance_ratio,
            "cumulative_explained_variance": self.cumulative_explained_variance,
            "mean": self.mean,
            "std": self.std,
            "computation_mode": self.computation_mode,
            "n_samples": self.n_samples,
            "n_features": self.n_features
        })
        return result


class PCA(CrossSectionalModelBase):
    """
    Principal Component Analysis implementation.
    
    This class implements Principal Component Analysis (PCA) with support for
    different computation modes and data transformations. It provides methods
    for fitting PCA models, transforming data, and computing inverse transforms.
    
    Attributes:
        n_components: Number of components to retain (default: None, keep all)
        computation_mode: Mode for computing PCA ('outer', 'cov', or 'corr')
        center: Whether to center the data (subtract mean)
        scale: Whether to scale the data (divide by std. dev.)
        tol: Tolerance for small eigenvalues
        results: PCA results after fitting
    """
    
    def __init__(
        self,
        n_components: Optional[int] = None,
        computation_mode: Literal["outer", "cov", "corr"] = "outer",
        center: bool = True,
        scale: bool = False,
        tol: float = 1e-10,
        name: str = "PCA"
    ) -> None:
        """
        Initialize the PCA model.
        
        Args:
            n_components: Number of components to retain (default: None, keep all)
            computation_mode: Mode for computing PCA:
                - 'outer': Based on the outer product (X'X)
                - 'cov': Based on the covariance matrix
                - 'corr': Based on the correlation matrix
            center: Whether to center the data (subtract mean)
            scale: Whether to scale the data (divide by std. dev.)
            tol: Tolerance for small eigenvalues
            name: Name of the model
            
        Raises:
            ParameterError: If parameters are invalid
        """
        super().__init__(name=name)
        
        if computation_mode not in ["outer", "cov", "corr"]:
            raise_parameter_error(
                f"Invalid computation mode: {computation_mode}. "
                "Must be one of 'outer', 'cov', or 'corr'.",
                param_name="computation_mode",
                param_value=computation_mode,
                constraint="one of 'outer', 'cov', or 'corr'"
            )
        
        if tol <= 0:
            raise_parameter_error(
                f"Tolerance must be positive, got {tol}",
                param_name="tol",
                param_value=tol,
                constraint="> 0"
            )
        
        self.n_components = n_components
        self.computation_mode = computation_mode
        self.center = center
        self.scale = scale
        self.tol = tol
        
        if self.computation_mode == "corr":
            if not center or not scale:
                logger.info(
                    "Setting center=True and scale=True for 'corr' computation mode"
                )
                self.center = True
                self.scale = True
        
        self._results = None
    
    def validate_data(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        elif isinstance(data, np.ndarray):
            data_array = data
        else:
            raise TypeError(
                f"Data must be a NumPy array or Pandas DataFrame, got {type(data).__name__}"
            )
        
        if data_array.ndim != 2:
            raise_dimension_error(
                f"Data must be 2-dimensional, got {data_array.ndim} dimensions",
                array_name="data",
                expected_shape="(n_samples, n_features)",
                actual_shape=data_array.shape
            )
        
        if np.isnan(data_array).any():
            raise_data_error(
                "Data contains NaN values",
                data_name="data",
                issue="contains NaN values"
            )
        
        if np.isinf(data_array).any():
            raise_data_error(
                "Data contains infinite values",
                data_name="data",
                issue="contains infinite values"
            )
        
        return data_array
    
    def _preprocess_data(
        self, 
        data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        n_samples, n_features = data.shape
        
        if self.center:
            mean = np.mean(data, axis=0)
        else:
            mean = np.zeros(n_features)
        
        if self.scale:
            std = np.std(data, axis=0, ddof=1)
            if np.any(std == 0):
                zero_std_features = np.where(std == 0)[0]
                raise_numeric_error(
                    f"Features {zero_std_features} have zero standard deviation, "
                    "cannot scale. Consider removing these features.",
                    operation="PCA preprocessing",
                    values=std,
                    error_type="zero_std_dev"
                )
        else:
            std = None
        
        processed_data = data.copy()
        
        if self.center:
            processed_data -= mean
        
        if self.scale:
            processed_data /= std
        
        return processed_data, mean, std
    
    def _compute_pca_outer(
        self, 
        data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_samples, n_features = data.shape
        
        outer_product = data.T @ data
        
        outer_product = ensure_symmetric(outer_product)
        
        eigenvalues, eigenvectors = linalg.eigh(outer_product)
        
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        eigenvalues = eigenvalues / n_samples
        
        components = data @ eigenvectors
        
        return eigenvalues, eigenvectors, components
    
    def _compute_pca_cov(
        self, 
        data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_samples, n_features = data.shape
        
        cov_matrix = np.cov(data, rowvar=False, ddof=1)
        
        cov_matrix = ensure_symmetric(cov_matrix)
        
        eigenvalues, eigenvectors = linalg.eigh(cov_matrix)
        
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        components = data @ eigenvectors
        
        return eigenvalues, eigenvectors, components
    
    def _compute_pca_corr(
        self, 
        data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_samples, n_features = data.shape
        
        corr_matrix = np.corrcoef(data, rowvar=False)
        
        corr_matrix = ensure_symmetric(corr_matrix)
        
        eigenvalues, eigenvectors = linalg.eigh(corr_matrix)
        
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        components = data @ eigenvectors
        
        return eigenvalues, eigenvectors, components
    
    @validate_input_type(0, (np.ndarray, pd.DataFrame))
    def fit(
        self, 
        data: Union[np.ndarray, pd.DataFrame],
        **kwargs: Any
    ) -> PCAResult:
        data_array = self.validate_data(data)
        n_samples, n_features = data_array.shape
        
        if self.n_components is not None:
            if not isinstance(self.n_components, int):
                raise_parameter_error(
                    f"n_components must be an integer, got {type(self.n_components).__name__}",
                    param_name="n_components",
                    param_value=self.n_components
                )
            if self.n_components < 1 or self.n_components > min(n_samples, n_features):
                raise_parameter_error(
                    f"n_components must be between 1 and min(n_samples, n_features) = "
                    f"{min(n_samples, n_features)}, got {self.n_components}",
                    param_name="n_components",
                    param_value=self.n_components,
                    constraint=f"between 1 and {min(n_samples, n_features)}"
                )
        
        processed_data, mean, std = self._preprocess_data(data_array)
        
        try:
            if self.computation_mode == "outer":
                eigenvalues, eigenvectors, components = self._compute_pca_outer(processed_data)
            elif self.computation_mode == "cov":
                eigenvalues, eigenvectors, components = self._compute_pca_cov(processed_data)
            elif self.computation_mode == "corr":
                eigenvalues, eigenvectors, components = self._compute_pca_corr(processed_data)
            else:
                raise ValueError(f"Invalid computation mode: {self.computation_mode}")
        except Exception as e:
            raise_numeric_error(
                f"PCA computation failed: {str(e)}",
                operation=f"PCA {self.computation_mode}",
                error_type="eigendecomposition_failed"
            ) from e
        
        valid_eigenvalues = eigenvalues > self.tol * eigenvalues[0]
        eigenvalues = eigenvalues[valid_eigenvalues]
        eigenvectors = eigenvectors[:, valid_eigenvalues]
        components = components[:, valid_eigenvalues]
        
        max_components = len(eigenvalues)
        if self.n_components is None:
            n_components = max_components
        else:
            n_components = min(self.n_components, max_components)
        
        eigenvalues = eigenvalues[:n_components]
        eigenvectors = eigenvectors[:, :n_components]
        components = components[:, :n_components]
        
        explained_variance = eigenvalues
        explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
        cumulative_explained_variance = np.cumsum(explained_variance_ratio)
        
        self._results = PCAResult(
            model_name=self.name,
            n_components=n_components,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            components=components,
            explained_variance=explained_variance,
            explained_variance_ratio=explained_variance_ratio,
            cumulative_explained_variance=cumulative_explained_variance,
            mean=mean,
            std=std,
            computation_mode=self.computation_mode,
            n_samples=n_samples,
            n_features=n_features,
            convergence=True,
            iterations=1
        )
        
        self._fitted = True
        
        return cast(PCAResult, self._results)
    
    async def fit_async(
        self, 
        data: Union[np.ndarray, pd.DataFrame],
        **kwargs: Any
    ) -> PCAResult:
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: self.fit(data, **kwargs))
        return result
    
    @validate_input_type(0, (np.ndarray, pd.DataFrame))
    def transform(
        self, 
        data: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        if not self._fitted or self._results is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = data
        
        if data_array.ndim != 2:
            raise_dimension_error(
                f"Data must be 2-dimensional, got {data_array.ndim} dimensions",
                array_name="data",
                expected_shape="(n_samples, n_features)",
                actual_shape=data_array.shape
            )
        
        if data_array.shape[1] != self._results.n_features:
            raise_dimension_error(
                f"Data has {data_array.shape[1]} features, but the model was trained with "
                f"{self._results.n_features} features",
                array_name="data",
                expected_shape=f"(n_samples, {self._results.n_features})",
                actual_shape=data_array.shape
            )
        
        processed_data = data_array.copy()
        
        if self.center:
            processed_data -= self._results.mean
        
        if self.scale and self._results.std is not None:
            processed_data /= self._results.std
        
        transformed_data = processed_data @ self._results.eigenvectors
        
        return transformed_data
    
    @validate_input_type(0, np.ndarray)
    def inverse_transform(
        self, 
        components: np.ndarray
    ) -> np.ndarray:
        if not self._fitted or self._results is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        if components.ndim != 2:
            raise_dimension_error(
                f"Components must be 2-dimensional, got {components.ndim} dimensions",
                array_name="components",
                expected_shape=f"(n_samples, {self._results.n_components})",
                actual_shape=components.shape
            )
        
        if components.shape[1] > self._results.n_components:
            raise_dimension_error(
                f"Components has {components.shape[1]} columns, but the model has only "
                f"{self._results.n_components} components",
                array_name="components",
                expected_shape=f"(n_samples, <= {self._results.n_components})",
                actual_shape=components.shape
            )
        
        reconstructed_data = components @ self._results.eigenvectors[:, :components.shape[1]].T
        
        if self.scale and self._results.std is not None:
            reconstructed_data *= self._results.std
        
        if self.center:
            reconstructed_data += self._results.mean
        
        return reconstructed_data
    
    def get_components(
        self, 
        n_components: Optional[int] = None
    ) -> np.ndarray:
        if not self._fitted or self._results is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        if n_components is None:
            return self._results.components
        
        if not isinstance(n_components, int) or n_components < 1:
            raise_parameter_error(
                f"n_components must be a positive integer, got {n_components}",
                param_name="n_components",
                param_value=n_components,
                constraint="positive integer"
            )
        
        if n_components > self._results.n_components:
            raise_parameter_error(
                f"n_components must be <= {self._results.n_components}, got {n_components}",
                param_name="n_components",
                param_value=n_components,
                constraint=f"<= {self._results.n_components}"
            )
        
        return self._results.components[:, :n_components]
    
    def get_loadings(
        self, 
        n_components: Optional[int] = None
    ) -> np.ndarray:
        if not self._fitted or self._results is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        if n_components is None:
            return self._results.eigenvectors
        
        if not isinstance(n_components, int) or n_components < 1:
            raise_parameter_error(
                f"n_components must be a positive integer, got {n_components}",
                param_name="n_components",
                param_value=n_components,
                constraint="positive integer"
            )
        
        if n_components > self._results.n_components:
            raise_parameter_error(
                f"n_components must be <= {self._results.n_components}, got {n_components}",
                param_name="n_components",
                param_value=n_components,
                constraint=f"<= {self._results.n_components}"
            )
        
        return self._results.eigenvectors[:, :n_components]
    
    def get_explained_variance(
        self, 
        n_components: Optional[int] = None
    ) -> np.ndarray:
        if not self._fitted or self._results is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        if n_components is None:
            return self._results.explained_variance
        
        if not isinstance(n_components, int) or n_components < 1:
            raise_parameter_error(
                f"n_components must be a positive integer, got {n_components}",
                param_name="n_components",
                param_value=n_components,
                constraint="positive integer"
            )
        
        if n_components > self._results.n_components:
            raise_parameter_error(
                f"n_components must be <= {self._results.n_components}, got {n_components}",
                param_name="n_components",
                param_value=n_components,
                constraint=f"<= {self._results.n_components}"
            )
        
        return self._results.explained_variance[:n_components]
    
    def get_explained_variance_ratio(
        self, 
        n_components: Optional[int] = None
    ) -> np.ndarray:
        if not self._fitted or self._results is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        if n_components is None:
            return self._results.explained_variance_ratio
        
        if not isinstance(n_components, int) or n_components < 1:
            raise_parameter_error(
                f"n_components must be a positive integer, got {n_components}",
                param_name="n_components",
                param_value=n_components,
                constraint="positive integer"
            )
        
        if n_components > self._results.n_components:
            raise_parameter_error(
                f"n_components must be <= {self._results.n_components}, got {n_components}",
                param_name="n_components",
                param_value=n_components,
                constraint=f"<= {self._results.n_components}"
            )
        
        return self._results.explained_variance_ratio[:n_components]
    
    def get_cumulative_explained_variance(
        self, 
        n_components: Optional[int] = None
    ) -> np.ndarray:
        if not self._fitted or self._results is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        if n_components is None:
            return self._results.cumulative_explained_variance
        
        if not isinstance(n_components, int) or n_components < 1:
            raise_parameter_error(
                f"n_components must be a positive integer, got {n_components}",
                param_name="n_components",
                param_value=n_components,
                constraint="positive integer"
            )
        
        if n_components > self._results.n_components:
            raise_parameter_error(
                f"n_components must be <= {self._results.n_components}, got {n_components}",
                param_name="n_components",
                param_value=n_components,
                constraint=f"<= {self._results.n_components}"
            )
        
        return self._results.cumulative_explained_variance[:n_components]
    
    def find_n_components(
        self, 
        explained_variance_threshold: float = 0.95
    ) -> int:
        if not self._fitted or self._results is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        if not 0 < explained_variance_threshold <= 1:
            raise_parameter_error(
                f"explained_variance_threshold must be between 0 and 1, got {explained_variance_threshold}",
                param_name="explained_variance_threshold",
                param_value=explained_variance_threshold,
                constraint="between 0 and 1"
            )
        
        n_components = np.argmax(self._results.cumulative_explained_variance >= explained_variance_threshold) + 1
        
        return n_components
    
    def simulate(
        self, 
        n_periods: int, 
        burn: int = 0, 
        initial_values: Optional[np.ndarray] = None,
        random_state: Optional[Union[int, np.random.Generator]] = None,
        **kwargs: Any
    ) -> np.ndarray:
        if not self._fitted or self._results is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        if not isinstance(n_periods, int) or n_periods < 1:
            raise_parameter_error(
                f"n_periods must be a positive integer, got {n_periods}",
                param_name="n_periods",
                param_value=n_periods,
                constraint="positive integer"
            )
        
        if isinstance(random_state, np.random.Generator):
            rng = random_state
        else:
            rng = np.random.default_rng(random_state)
        
        random_components = rng.standard_normal((n_periods, self._results.n_components))
        random_components *= np.sqrt(self._results.explained_variance)
        
        simulated_data = self.inverse_transform(random_components)
        
        return simulated_data
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.transform(X)