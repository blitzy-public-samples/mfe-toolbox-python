# mfe/models/time_series/filters.py

"""
Time Series Filtering and Decomposition Module

This module provides a comprehensive set of tools for time series filtering and
decomposition, including Hodrick-Prescott filter, Baxter-King filter, and
Beveridge-Nelson decomposition. These methods are essential for extracting trend,
cycle, and noise components from economic time series data.

The module implements both univariate and multivariate versions of popular filters
with configurable parameters, proper validation, and visualization capabilities.
All implementations leverage NumPy and SciPy's efficient array operations with
Numba acceleration for performance-critical computations.

Classes:
    FilterBase: Abstract base class for all time series filters
    HPFilter: Hodrick-Prescott filter implementation
    BKFilter: Baxter-King band-pass filter implementation
    CFFilter: Christiano-Fitzgerald band-pass filter implementation
    BNDecomposition: Beveridge-Nelson decomposition implementation
    HamiltonFilter: Hamilton alternative to HP filter implementation
    UnobservedComponentsFilter: Unobserved components model filter

Functions:
    hp_filter: Convenience function for Hodrick-Prescott filtering
    bk_filter: Convenience function for Baxter-King filtering
    cf_filter: Convenience function for Christiano-Fitzgerald filtering
    bn_decomposition: Convenience function for Beveridge-Nelson decomposition
    hamilton_filter: Convenience function for Hamilton filtering
    uc_filter: Convenience function for unobserved components filtering
"""

import asyncio
import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Literal, Optional, Protocol, Tuple, 
    Type, TypeVar, Union, cast, overload
)

import numpy as np
import pandas as pd
from scipy import sparse, stats, signal, optimize, linalg
import matplotlib.pyplot as plt

from mfe.core.base import ModelBase
from mfe.core.parameters import (
    ParameterBase, validate_positive, validate_non_negative, validate_range,
    transform_positive, inverse_transform_positive
)
from mfe.core.exceptions import (
    ParameterError, DimensionError, NumericError, NotFittedError,
    warn_numeric, warn_model
)
from mfe.core.types import (
    Vector, Matrix, TimeSeriesData, TimeSeriesDataFrame, ProgressCallback
)
from mfe.utils.matrix_ops import (
    ensure_symmetric, is_positive_definite, nearest_positive_definite
)

# Set up module-level logger
logger = logging.getLogger("mfe.models.time_series.filters")

# Type variables for generic base classes
T = TypeVar('T', bound=ParameterBase)  # Type for filter parameters
D = TypeVar('D', bound=Union[np.ndarray, pd.Series, pd.DataFrame])  # Type for data


@dataclass
class FilterResult:
    """Result container for time series filtering operations.
    
    This class provides a standardized container for filter results,
    including the original series, trend component, cycle component,
    and any additional components specific to the filter.
    
    Attributes:
        original: Original time series data
        trend: Trend component
        cycle: Cycle component
        components: Dictionary of additional components
        parameters: Dictionary of filter parameters
        filter_name: Name of the filter used
        index: Time index from the original data (if available)
    """
    
    original: np.ndarray
    trend: np.ndarray
    cycle: np.ndarray
    components: Dict[str, np.ndarray] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    filter_name: str = "Unknown Filter"
    index: Optional[Union[pd.DatetimeIndex, pd.Index]] = None
    
    def __post_init__(self) -> None:
        """Validate result object after initialization."""
        # Ensure all arrays have the same length
        n = len(self.original)
        if len(self.trend) != n:
            raise DimensionError(
                "Trend component length must match original data length",
                array_name="trend",
                expected_shape=f"({n},)",
                actual_shape=self.trend.shape
            )
        if len(self.cycle) != n:
            raise DimensionError(
                "Cycle component length must match original data length",
                array_name="cycle",
                expected_shape=f"({n},)",
                actual_shape=self.cycle.shape
            )
        for name, component in self.components.items():
            if len(component) != n:
                raise DimensionError(
                    f"Component '{name}' length must match original data length",
                    array_name=f"components['{name}']",
                    expected_shape=f"({n},)",
                    actual_shape=component.shape
                )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result object to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the result object
        """
        result_dict = {
            "original": self.original,
            "trend": self.trend,
            "cycle": self.cycle,
            "components": self.components.copy(),
            "parameters": self.parameters.copy(),
            "filter_name": self.filter_name
        }
        if self.index is not None:
            result_dict["index"] = self.index
        
        return result_dict
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert the result object to a pandas DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame containing all components
        """
        data = {
            "original": self.original,
            "trend": self.trend,
            "cycle": self.cycle
        }
        # Add additional components
        data.update(self.components)
        
        # Create DataFrame with index if available
        if self.index is not None:
            return pd.DataFrame(data, index=self.index)
        else:
            return pd.DataFrame(data)
    
    def plot(self, figsize: Tuple[int, int] = (12, 8), 
             components: Optional[List[str]] = None) -> plt.Figure:
        """Plot the filter results.
        
        Args:
            figsize: Figure size (width, height) in inches
            components: List of components to plot (default: original, trend, cycle)
            
        Returns:
            plt.Figure: Matplotlib figure object
        """
        # Determine which components to plot
        if components is None:
            components = ["original", "trend", "cycle"]
        
        # Create figure and subplots
        fig, axes = plt.subplots(len(components), 1, figsize=figsize, sharex=True)
        if len(components) == 1:
            axes = [axes]  # Make sure axes is always a list
        
        # Create x-axis values (either index or range)
        x = self.index if self.index is not None else np.arange(len(self.original))
        
        # Plot each component
        for i, component in enumerate(components):
            if component == "original":
                axes[i].plot(x, self.original, label="Original")
                axes[i].set_title("Original Series")
            elif component == "trend":
                axes[i].plot(x, self.trend, label="Trend", color="red")
                axes[i].set_title("Trend Component")
            elif component == "cycle":
                axes[i].plot(x, self.cycle, label="Cycle", color="green")
                axes[i].set_title("Cycle Component")
            elif component in self.components:
                axes[i].plot(x, self.components[component], 
                             label=component.capitalize(), color="purple")
                axes[i].set_title(f"{component.capitalize()} Component")
            else:
                warnings.warn(f"Component '{component}' not found in filter results")
                continue
            
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # Add filter information to figure title
        fig.suptitle(f"{self.filter_name} Decomposition", fontsize=14)
        
        # Adjust layout
        fig.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig


@dataclass
class FilterParameters(ParameterBase):
    """Base class for filter parameters.
    
    This class provides a common structure for filter parameters,
    including validation and transformation methods.
    """
    
    def validate(self) -> None:
        """Validate parameter constraints for filters.
        
        Raises:
            ParameterError: If parameter constraints are violated
        """
        # Base class doesn't enforce specific constraints
        # Subclasses should implement filter-specific constraints
        pass


@dataclass
class HPFilterParameters(FilterParameters):
    """Parameters for Hodrick-Prescott filter.
    
    Attributes:
        lambda_: Smoothing parameter (must be positive)
    """
    
    lambda_: float
    
    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate HP filter parameter constraints.
        
        Raises:
            ParameterError: If parameter constraints are violated
        """
        super().validate()
        
        # Validate lambda
        validate_positive(self.lambda_, "lambda_")
    
    def to_array(self) -> np.ndarray:
        """Convert parameters to a NumPy array.
        
        Returns:
            np.ndarray: Array representation of parameters
        """
        return np.array([self.lambda_])
    
    @classmethod
    def from_array(cls, array: np.ndarray, **kwargs: Any) -> 'HPFilterParameters':
        """Create parameters from a NumPy array.
        
        Args:
            array: Array representation of parameters
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            HPFilterParameters: Parameter object
        
        Raises:
            ValueError: If the array length is not 1
        """
        if len(array) != 1:
            raise ValueError(f"Array length must be 1, got {len(array)}")
        
        return cls(lambda_=array[0])
    
    def transform(self) -> np.ndarray:
        """Transform parameters to unconstrained space for optimization.
        
        Returns:
            np.ndarray: Transformed parameters in unconstrained space
        """
        # Transform lambda to unconstrained space (log)
        transformed_lambda = transform_positive(self.lambda_)
        
        return np.array([transformed_lambda])
    
    @classmethod
    def inverse_transform(cls, array: np.ndarray, **kwargs: Any) -> 'HPFilterParameters':
        """Transform parameters from unconstrained space back to constrained space.
        
        Args:
            array: Parameters in unconstrained space
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            HPFilterParameters: Parameter object with constrained parameters
        
        Raises:
            ValueError: If the array length is not 1
        """
        if len(array) != 1:
            raise ValueError(f"Array length must be 1, got {len(array)}")
        
        # Inverse transform lambda
        lambda_ = inverse_transform_positive(array[0])
        
        return cls(lambda_=lambda_)


@dataclass
class BKFilterParameters(FilterParameters):
    """Parameters for Baxter-King band-pass filter.
    
    Attributes:
        low: Lower cutoff frequency in periods (must be positive)
        high: Upper cutoff frequency in periods (must be greater than low)
        K: Number of lags/leads (must be positive)
    """
    
    low: float
    high: float
    K: int
    
    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate BK filter parameter constraints.
        
        Raises:
            ParameterError: If parameter constraints are violated
        """
        super().validate()
        
        # Validate low and high
        validate_positive(self.low, "low")
        validate_positive(self.high, "high")
        
        # Validate high > low
        if self.high <= self.low:
            raise ParameterError(
                f"high must be greater than low, got high={self.high}, low={self.low}",
                param_name="high",
                param_value=self.high,
                constraint=f"Must be greater than low ({self.low})"
            )
        
        # Validate K
        if not isinstance(self.K, int) or self.K <= 0:
            raise ParameterError(
                f"K must be a positive integer, got {self.K}",
                param_name="K",
                param_value=self.K,
                constraint="Must be a positive integer"
            )
    
    def to_array(self) -> np.ndarray:
        """Convert parameters to a NumPy array.
        
        Returns:
            np.ndarray: Array representation of parameters
        """
        return np.array([self.low, self.high, self.K])
    
    @classmethod
    def from_array(cls, array: np.ndarray, **kwargs: Any) -> 'BKFilterParameters':
        """Create parameters from a NumPy array.
        
        Args:
            array: Array representation of parameters
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            BKFilterParameters: Parameter object
        
        Raises:
            ValueError: If the array length is not 3
        """
        if len(array) != 3:
            raise ValueError(f"Array length must be 3, got {len(array)}")
        
        return cls(low=array[0], high=array[1], K=int(array[2]))
    
    def transform(self) -> np.ndarray:
        """Transform parameters to unconstrained space for optimization.
        
        Returns:
            np.ndarray: Transformed parameters in unconstrained space
        """
        # Transform low to unconstrained space (log)
        transformed_low = transform_positive(self.low)
        
        # Transform high to ensure high > low
        # We parameterize as high = low + delta, where delta > 0
        delta = self.high - self.low
        transformed_delta = transform_positive(delta)
        
        # K is an integer, so we don't transform it
        
        return np.array([transformed_low, transformed_delta, self.K])
    
    @classmethod
    def inverse_transform(cls, array: np.ndarray, **kwargs: Any) -> 'BKFilterParameters':
        """Transform parameters from unconstrained space back to constrained space.
        
        Args:
            array: Parameters in unconstrained space
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            BKFilterParameters: Parameter object with constrained parameters
        
        Raises:
            ValueError: If the array length is not 3
        """
        if len(array) != 3:
            raise ValueError(f"Array length must be 3, got {len(array)}")
        
        # Inverse transform low
        low = inverse_transform_positive(array[0])
        
        # Inverse transform delta and compute high
        delta = inverse_transform_positive(array[1])
        high = low + delta
        
        # K is an integer, so we round it
        K = int(round(array[2]))
        
        return cls(low=low, high=high, K=K)


@dataclass
class CFFilterParameters(FilterParameters):
    """Parameters for Christiano-Fitzgerald band-pass filter.
    
    Attributes:
        low: Lower cutoff frequency in periods (must be positive)
        high: Upper cutoff frequency in periods (must be greater than low)
        drift: Whether to remove drift from the series
        symmetric: Whether to use symmetric filter weights
    """
    
    low: float
    high: float
    drift: bool = False
    symmetric: bool = False
    
    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate CF filter parameter constraints.
        
        Raises:
            ParameterError: If parameter constraints are violated
        """
        super().validate()
        
        # Validate low and high
        validate_positive(self.low, "low")
        validate_positive(self.high, "high")
        
        # Validate high > low
        if self.high <= self.low:
            raise ParameterError(
                f"high must be greater than low, got high={self.high}, low={self.low}",
                param_name="high",
                param_value=self.high,
                constraint=f"Must be greater than low ({self.low})"
            )
    
    def to_array(self) -> np.ndarray:
        """Convert parameters to a NumPy array.
        
        Returns:
            np.ndarray: Array representation of parameters
        """
        return np.array([self.low, self.high, float(self.drift), float(self.symmetric)])
    
    @classmethod
    def from_array(cls, array: np.ndarray, **kwargs: Any) -> 'CFFilterParameters':
        """Create parameters from a NumPy array.
        
        Args:
            array: Array representation of parameters
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            CFFilterParameters: Parameter object
        
        Raises:
            ValueError: If the array length is not 4
        """
        if len(array) != 4:
            raise ValueError(f"Array length must be 4, got {len(array)}")
        
        return cls(
            low=array[0], 
            high=array[1], 
            drift=bool(array[2]), 
            symmetric=bool(array[3])
        )
    
    def transform(self) -> np.ndarray:
        """Transform parameters to unconstrained space for optimization.
        
        Returns:
            np.ndarray: Transformed parameters in unconstrained space
        """
        # Transform low to unconstrained space (log)
        transformed_low = transform_positive(self.low)
        
        # Transform high to ensure high > low
        # We parameterize as high = low + delta, where delta > 0
        delta = self.high - self.low
        transformed_delta = transform_positive(delta)
        
        # drift and symmetric are booleans, so we don't transform them
        
        return np.array([
            transformed_low, 
            transformed_delta, 
            float(self.drift), 
            float(self.symmetric)
        ])
    
    @classmethod
    def inverse_transform(cls, array: np.ndarray, **kwargs: Any) -> 'CFFilterParameters':
        """Transform parameters from unconstrained space back to constrained space.
        
        Args:
            array: Parameters in unconstrained space
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            CFFilterParameters: Parameter object with constrained parameters
        
        Raises:
            ValueError: If the array length is not 4
        """
        if len(array) != 4:
            raise ValueError(f"Array length must be 4, got {len(array)}")
        
        # Inverse transform low
        low = inverse_transform_positive(array[0])
        
        # Inverse transform delta and compute high
        delta = inverse_transform_positive(array[1])
        high = low + delta
        
        # drift and symmetric are booleans
        drift = bool(array[2])
        symmetric = bool(array[3])
        
        return cls(low=low, high=high, drift=drift, symmetric=symmetric)


@dataclass
class BNDecompositionParameters(FilterParameters):
    """Parameters for Beveridge-Nelson decomposition.
    
    Attributes:
        ar_order: Order of the autoregressive component (must be non-negative)
        ma_order: Order of the moving average component (must be non-negative)
        forecast_horizon: Forecast horizon for long-run component (must be positive)
    """
    
    ar_order: int
    ma_order: int
    forecast_horizon: int = 40
    
    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate BN decomposition parameter constraints.
        
        Raises:
            ParameterError: If parameter constraints are violated
        """
        super().validate()
        
        # Validate ar_order and ma_order
        if not isinstance(self.ar_order, int) or self.ar_order < 0:
            raise ParameterError(
                f"ar_order must be a non-negative integer, got {self.ar_order}",
                param_name="ar_order",
                param_value=self.ar_order,
                constraint="Must be a non-negative integer"
            )
        
        if not isinstance(self.ma_order, int) or self.ma_order < 0:
            raise ParameterError(
                f"ma_order must be a non-negative integer, got {self.ma_order}",
                param_name="ma_order",
                param_value=self.ma_order,
                constraint="Must be a non-negative integer"
            )
        
        # At least one of ar_order or ma_order must be positive
        if self.ar_order == 0 and self.ma_order == 0:
            raise ParameterError(
                "At least one of ar_order or ma_order must be positive",
                param_name="ar_order, ma_order",
                constraint="At least one must be positive"
            )
        
        # Validate forecast_horizon
        if not isinstance(self.forecast_horizon, int) or self.forecast_horizon <= 0:
            raise ParameterError(
                f"forecast_horizon must be a positive integer, got {self.forecast_horizon}",
                param_name="forecast_horizon",
                param_value=self.forecast_horizon,
                constraint="Must be a positive integer"
            )
    
    def to_array(self) -> np.ndarray:
        """Convert parameters to a NumPy array.
        
        Returns:
            np.ndarray: Array representation of parameters
        """
        return np.array([self.ar_order, self.ma_order, self.forecast_horizon])
    
    @classmethod
    def from_array(cls, array: np.ndarray, **kwargs: Any) -> 'BNDecompositionParameters':
        """Create parameters from a NumPy array.
        
        Args:
            array: Array representation of parameters
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            BNDecompositionParameters: Parameter object
        
        Raises:
            ValueError: If the array length is not 3
        """
        if len(array) != 3:
            raise ValueError(f"Array length must be 3, got {len(array)}")
        
        return cls(
            ar_order=int(array[0]), 
            ma_order=int(array[1]), 
            forecast_horizon=int(array[2])
        )
    
    def transform(self) -> np.ndarray:
        """Transform parameters to unconstrained space for optimization.
        
        Returns:
            np.ndarray: Transformed parameters in unconstrained space
        """
        # These are all integers, so we don't transform them
        return np.array([self.ar_order, self.ma_order, self.forecast_horizon])
    
    @classmethod
    def inverse_transform(cls, array: np.ndarray, **kwargs: Any) -> 'BNDecompositionParameters':
        """Transform parameters from unconstrained space back to constrained space.
        
        Args:
            array: Parameters in unconstrained space
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            BNDecompositionParameters: Parameter object with constrained parameters
        
        Raises:
            ValueError: If the array length is not 3
        """
        if len(array) != 3:
            raise ValueError(f"Array length must be 3, got {len(array)}")
        
        # Round to integers
        ar_order = max(0, int(round(array[0])))
        ma_order = max(0, int(round(array[1])))
        forecast_horizon = max(1, int(round(array[2])))
        
        return cls(
            ar_order=ar_order, 
            ma_order=ma_order, 
            forecast_horizon=forecast_horizon
        )


@dataclass
class HamiltonFilterParameters(FilterParameters):
    """Parameters for Hamilton filter.
    
    Attributes:
        h: Forecast horizon (must be positive)
        p: Order of the autoregressive component (must be positive)
    """
    
    h: int
    p: int
    
    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate Hamilton filter parameter constraints.
        
        Raises:
            ParameterError: If parameter constraints are violated
        """
        super().validate()
        
        # Validate h
        if not isinstance(self.h, int) or self.h <= 0:
            raise ParameterError(
                f"h must be a positive integer, got {self.h}",
                param_name="h",
                param_value=self.h,
                constraint="Must be a positive integer"
            )
        
        # Validate p
        if not isinstance(self.p, int) or self.p <= 0:
            raise ParameterError(
                f"p must be a positive integer, got {self.p}",
                param_name="p",
                param_value=self.p,
                constraint="Must be a positive integer"
            )
    
    def to_array(self) -> np.ndarray:
        """Convert parameters to a NumPy array.
        
        Returns:
            np.ndarray: Array representation of parameters
        """
        return np.array([self.h, self.p])
    
    @classmethod
    def from_array(cls, array: np.ndarray, **kwargs: Any) -> 'HamiltonFilterParameters':
        """Create parameters from a NumPy array.
        
        Args:
            array: Array representation of parameters
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            HamiltonFilterParameters: Parameter object
        
        Raises:
            ValueError: If the array length is not 2
        """
        if len(array) != 2:
            raise ValueError(f"Array length must be 2, got {len(array)}")
        
        return cls(h=int(array[0]), p=int(array[1]))
    
    def transform(self) -> np.ndarray:
        """Transform parameters to unconstrained space for optimization.
        
        Returns:
            np.ndarray: Transformed parameters in unconstrained space
        """
        # These are all integers, so we don't transform them
        return np.array([self.h, self.p])
    
    @classmethod
    def inverse_transform(cls, array: np.ndarray, **kwargs: Any) -> 'HamiltonFilterParameters':
        """Transform parameters from unconstrained space back to constrained space.
        
        Args:
            array: Parameters in unconstrained space
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            HamiltonFilterParameters: Parameter object with constrained parameters
        
        Raises:
            ValueError: If the array length is not 2
        """
        if len(array) != 2:
            raise ValueError(f"Array length must be 2, got {len(array)}")
        
        # Round to integers and ensure they're positive
        h = max(1, int(round(array[0])))
        p = max(1, int(round(array[1])))
        
        return cls(h=h, p=p)


@dataclass
class UCFilterParameters(FilterParameters):
    """Parameters for Unobserved Components filter.
    
    Attributes:
        trend_order: Order of the trend component (0, 1, or 2)
        cycle_periods: Number of periods in the cycle (must be positive)
        damping_factor: Damping factor for the cycle (between 0 and 1)
        irregular_var: Variance of the irregular component (must be positive)
    """
    
    trend_order: int
    cycle_periods: float
    damping_factor: float
    irregular_var: float
    
    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate UC filter parameter constraints.
        
        Raises:
            ParameterError: If parameter constraints are violated
        """
        super().validate()
        
        # Validate trend_order
        if self.trend_order not in [0, 1, 2]:
            raise ParameterError(
                f"trend_order must be 0, 1, or 2, got {self.trend_order}",
                param_name="trend_order",
                param_value=self.trend_order,
                constraint="Must be 0, 1, or 2"
            )
        
        # Validate cycle_periods
        validate_positive(self.cycle_periods, "cycle_periods")
        
        # Validate damping_factor
        validate_range(self.damping_factor, "damping_factor", 0, 1)
        
        # Validate irregular_var
        validate_positive(self.irregular_var, "irregular_var")
    
    def to_array(self) -> np.ndarray:
        """Convert parameters to a NumPy array.
        
        Returns:
            np.ndarray: Array representation of parameters
        """
        return np.array([
            self.trend_order, 
            self.cycle_periods, 
            self.damping_factor, 
            self.irregular_var
        ])
    
    @classmethod
    def from_array(cls, array: np.ndarray, **kwargs: Any) -> 'UCFilterParameters':
        """Create parameters from a NumPy array.
        
        Args:
            array: Array representation of parameters
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            UCFilterParameters: Parameter object
        
        Raises:
            ValueError: If the array length is not 4
        """
        if len(array) != 4:
            raise ValueError(f"Array length must be 4, got {len(array)}")
        
        return cls(
            trend_order=int(array[0]),
            cycle_periods=array[1],
            damping_factor=array[2],
            irregular_var=array[3]
        )
    
    def transform(self) -> np.ndarray:
        """Transform parameters to unconstrained space for optimization.
        
        Returns:
            np.ndarray: Transformed parameters in unconstrained space
        """
        # trend_order is an integer with specific values, so we don't transform it
        
        # Transform cycle_periods to unconstrained space (log)
        transformed_cycle_periods = transform_positive(self.cycle_periods)
        
        # Transform damping_factor to unconstrained space (logit)
        from scipy.special import logit
        transformed_damping_factor = logit(self.damping_factor)
        
        # Transform irregular_var to unconstrained space (log)
        transformed_irregular_var = transform_positive(self.irregular_var)
        
        return np.array([
            self.trend_order,
            transformed_cycle_periods,
            transformed_damping_factor,
            transformed_irregular_var
        ])
    
    @classmethod
    def inverse_transform(cls, array: np.ndarray, **kwargs: Any) -> 'UCFilterParameters':
        """Transform parameters from unconstrained space back to constrained space.
        
        Args:
            array: Parameters in unconstrained space
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            UCFilterParameters: Parameter object with constrained parameters
        
        Raises:
            ValueError: If the array length is not 4
        """
        if len(array) != 4:
            raise ValueError(f"Array length must be 4, got {len(array)}")
        
        # Round trend_order to nearest valid value (0, 1, or 2)
        trend_order_raw = int(round(array[0]))
        trend_order = min(max(trend_order_raw, 0), 2)
        
        # Inverse transform cycle_periods
        cycle_periods = inverse_transform_positive(array[1])
        
        # Inverse transform damping_factor
        from scipy.special import expit  # logistic function, inverse of logit
        damping_factor = expit(array[2])
        
        # Inverse transform irregular_var
        irregular_var = inverse_transform_positive(array[3])
        
        return cls(
            trend_order=trend_order,
            cycle_periods=cycle_periods,
            damping_factor=damping_factor,
            irregular_var=irregular_var
        )


class FilterBase(ABC, ModelBase[T, FilterResult, TimeSeriesData]):
    """Abstract base class for time series filters.
    
    This class defines the common interface that all filter implementations
    must follow, establishing a consistent API across the entire module.
    
    Type Parameters:
        T: The parameter type for this filter
    """
    
    def __init__(self, name: str = "FilterBase"):
        """Initialize the filter.
        
        Args:
            name: A descriptive name for the filter
        """
        super().__init__(name=name)
        self._data: Optional[np.ndarray] = None
        self._index: Optional[Union[pd.DatetimeIndex, pd.Index]] = None
        self._trend: Optional[np.ndarray] = None
        self._cycle: Optional[np.ndarray] = None
        self._components: Dict[str, np.ndarray] = {}
    
    @property
    def data(self) -> Optional[np.ndarray]:
        """Get the filter data.
        
        Returns:
            Optional[np.ndarray]: The filter data if set, None otherwise
        """
        return self._data
    
    @property
    def index(self) -> Optional[Union[pd.DatetimeIndex, pd.Index]]:
        """Get the data index.
        
        Returns:
            Optional[Union[pd.DatetimeIndex, pd.Index]]: The data index if available, None otherwise
        """
        return self._index
    
    @property
    def trend(self) -> Optional[np.ndarray]:
        """Get the trend component.
        
        Returns:
            Optional[np.ndarray]: The trend component if filtered, None otherwise
        
        Raises:
            NotFittedError: If the filter has not been applied
        """
        if not self._fitted:
            raise NotFittedError(
                "Filter has not been applied. Call filter() first.",
                model_type=self._name,
                operation="trend"
            )
        return self._trend
    
    @property
    def cycle(self) -> Optional[np.ndarray]:
        """Get the cycle component.
        
        Returns:
            Optional[np.ndarray]: The cycle component if filtered, None otherwise
        
        Raises:
            NotFittedError: If the filter has not been applied
        """
        if not self._fitted:
            raise NotFittedError(
                "Filter has not been applied. Call filter() first.",
                model_type=self._name,
                operation="cycle"
            )
        return self._cycle
    
    @property
    def components(self) -> Dict[str, np.ndarray]:
        """Get additional components.
        
        Returns:
            Dict[str, np.ndarray]: Dictionary of additional components
        
        Raises:
            NotFittedError: If the filter has not been applied
        """
        if not self._fitted:
            raise NotFittedError(
                "Filter has not been applied. Call filter() first.",
                model_type=self._name,
                operation="components"
            )
        return self._components.copy()
    
    def validate_data(self, data: TimeSeriesData) -> np.ndarray:
        """Validate the input data for filtering.
        
        Args:
            data: The data to validate
        
        Returns:
            np.ndarray: The validated data as a NumPy array
        
        Raises:
            TypeError: If the data has an incorrect type
            ValueError: If the data is invalid
        """
        # Store the index if available
        if isinstance(data, pd.Series):
            self._index = data.index
            data_array = data.values
        elif isinstance(data, np.ndarray):
            data_array = data
            self._index = None
        else:
            raise TypeError(
                f"Data must be a NumPy array or Pandas Series, got {type(data).__name__}"
            )
        
        # Check dimensions
        if data_array.ndim != 1:
            raise DimensionError(
                f"Data must be 1-dimensional, got {data_array.ndim} dimensions",
                array_name="data",
                expected_shape="(n,)",
                actual_shape=data_array.shape
            )
        
        # Check length
        if len(data_array) < 4:  # Minimum length for most filters
            raise ValueError(
                f"Data length must be at least 4, got {len(data_array)}"
            )
        
        # Check for NaN and Inf values
        if np.isnan(data_array).any():
            raise ValueError("Data contains NaN values")
        
        if np.isinf(data_array).any():
            raise ValueError("Data contains infinite values")
        
        return data_array
    
    @abstractmethod
    def filter(self, 
              data: TimeSeriesData, 
              **kwargs: Any) -> FilterResult:
        """Apply the filter to the provided data.
        
        This method must be implemented by all subclasses to apply the
        specific filtering algorithm to the input data.
        
        Args:
            data: The data to filter
            **kwargs: Additional keyword arguments for filtering
        
        Returns:
            FilterResult: The filter results
        
        Raises:
            ValueError: If the data is invalid
            NumericError: If the filtering fails
        """
        pass
    
    async def filter_async(self, 
                         data: TimeSeriesData, 
                         **kwargs: Any) -> FilterResult:
        """Asynchronously apply the filter to the provided data.
        
        This method provides an asynchronous interface to the filter method,
        allowing for non-blocking filtering in UI contexts.
        
        Args:
            data: The data to filter
            **kwargs: Additional keyword arguments for filtering
        
        Returns:
            FilterResult: The filter results
        
        Raises:
            ValueError: If the data is invalid
            NumericError: If the filtering fails
        """
        # Create a coroutine that runs the synchronous filter method in a thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: self.filter(data, **kwargs)
        )
        return result
    
    def _create_result_object(self, 
                            data: np.ndarray, 
                            trend: np.ndarray, 
                            cycle: np.ndarray,
                            components: Optional[Dict[str, np.ndarray]] = None,
                            parameters: Optional[Dict[str, Any]] = None) -> FilterResult:
        """Create a result object from filter results.
        
        Args:
            data: Original data
            trend: Trend component
            cycle: Cycle component
            components: Additional components
            parameters: Filter parameters
        
        Returns:
            FilterResult: Filter results
        """
        # Create result object
        result = FilterResult(
            original=data,
            trend=trend,
            cycle=cycle,
            components=components or {},
            parameters=parameters or {},
            filter_name=self._name,
            index=self._index
        )
        
        self._results = result
        return result



class HPFilter(FilterBase[HPFilterParameters]):
    """Hodrick-Prescott filter implementation.
    
    The Hodrick-Prescott filter is a smoothing method that is widely used to
    obtain a smooth estimate of the long-term trend component of a series.
    The filter separates a time series into trend and cycle components.
    
    Attributes:
        lambda_: Smoothing parameter (higher values give smoother trends)
    """
    
    def __init__(self, lambda_: float = 1600.0, name: str = "Hodrick-Prescott Filter"):
        """Initialize the HP filter.
        
        Args:
            lambda_: Smoothing parameter (default: 1600.0 for quarterly data)
            name: A descriptive name for the filter
        """
        super().__init__(name=name)
        self.params = HPFilterParameters(lambda_=lambda_)
    
    @property
    def lambda_(self) -> float:
        """Get the smoothing parameter.
        
        Returns:
            float: The smoothing parameter
        """
        return self.params.lambda_
    
    @lambda_.setter
    def lambda_(self, value: float) -> None:
        """Set the smoothing parameter.
        
        Args:
            value: The new smoothing parameter
        
        Raises:
            ParameterError: If the parameter is invalid
        """
        self.params = HPFilterParameters(lambda_=value)
    
    def filter(self, 
              data: TimeSeriesData, 
              lambda_: Optional[float] = None,
              **kwargs: Any) -> FilterResult:
        """Apply the Hodrick-Prescott filter to the provided data.
        
        Args:
            data: The data to filter
            lambda_: Smoothing parameter (overrides the instance parameter if provided)
            **kwargs: Additional keyword arguments for filtering
        
        Returns:
            FilterResult: The filter results
        
        Raises:
            ValueError: If the data is invalid
            NumericError: If the filtering fails
        """
        # Validate and prepare data
        data_array = self.validate_data(data)
        self._data = data_array
        
        # Update lambda if provided
        if lambda_ is not None:
            self.lambda_ = lambda_
        
        try:
            # Get the data length
            n = len(data_array)
            
            # Create the second difference matrix
            # This is a sparse matrix with 1s on the main diagonal, -2s on the first
            # diagonal, and 1s on the second diagonal
            eye = sparse.eye(n, format='csc')
            D = eye[1:] - eye[:-1]  # First difference
            D = D[1:] - D[:-1]      # Second difference
            
            # Create the HP filter matrix
            # This is a sparse matrix that represents the minimization problem
            # min sum((y - tau)^2) + lambda * sum((D2 * tau)^2)
            I = sparse.eye(n, format='csc')
            A = I + self.lambda_ * D.T @ D
            
            # Solve the system A * trend = data
            trend = sparse.linalg.spsolve(A, data_array)
            
            # Compute the cycle as the difference between the data and the trend
            cycle = data_array - trend
            
            # Store the results
            self._trend = trend
            self._cycle = cycle
            self._fitted = True
            
            # Create result object
            parameters = {"lambda_": self.lambda_}
            result = self._create_result_object(
                data=data_array,
                trend=trend,
                cycle=cycle,
                parameters=parameters
            )
            
            return result
            
        except Exception as e:
            raise NumericError(
                f"HP filter failed: {e}",
                operation="HP filter",
                error_type="computation",
                details=str(e)
            )


class BKFilter(FilterBase[BKFilterParameters]):
    """Baxter-King band-pass filter implementation.
    
    The Baxter-King filter is a band-pass filter that isolates the cyclical
    component of a time series by removing high and low frequency components.
    
    Attributes:
        low: Lower cutoff frequency in periods
        high: Upper cutoff frequency in periods
        K: Number of lags/leads
    """
    
    def __init__(self, 
                low: float = 6.0, 
                high: float = 32.0, 
                K: int = 12,
                name: str = "Baxter-King Filter"):
        """Initialize the BK filter.
        
        Args:
            low: Lower cutoff frequency in periods (default: 6.0)
            high: Upper cutoff frequency in periods (default: 32.0)
            K: Number of lags/leads (default: 12)
            name: A descriptive name for the filter
        """
        super().__init__(name=name)
        self.params = BKFilterParameters(low=low, high=high, K=K)
    
    @property
    def low(self) -> float:
        """Get the lower cutoff frequency.
        
        Returns:
            float: The lower cutoff frequency
        """
        return self.params.low
    
    @low.setter
    def low(self, value: float) -> None:
        """Set the lower cutoff frequency.
        
        Args:
            value: The new lower cutoff frequency
        
        Raises:
            ParameterError: If the parameter is invalid
        """
        self.params = BKFilterParameters(
            low=value, 
            high=self.params.high, 
            K=self.params.K
        )
    
    @property
    def high(self) -> float:
        """Get the upper cutoff frequency.
        
        Returns:
            float: The upper cutoff frequency
        """
        return self.params.high
    
    @high.setter
    def high(self, value: float) -> None:
        """Set the upper cutoff frequency.
        
        Args:
            value: The new upper cutoff frequency
        
        Raises:
            ParameterError: If the parameter is invalid
        """
        self.params = BKFilterParameters(
            low=self.params.low, 
            high=value, 
            K=self.params.K
        )
    
    @property
    def K(self) -> int:
        """Get the number of lags/leads.
        
        Returns:
            int: The number of lags/leads
        """
        return self.params.K
    
    @K.setter
    def K(self, value: int) -> None:
        """Set the number of lags/leads.
        
        Args:
            value: The new number of lags/leads
        
        Raises:
            ParameterError: If the parameter is invalid
        """
        self.params = BKFilterParameters(
            low=self.params.low, 
            high=self.params.high, 
            K=value
        )
    
    def filter(self, 
              data: TimeSeriesData, 
              low: Optional[float] = None,
              high: Optional[float] = None,
              K: Optional[int] = None,
              **kwargs: Any) -> FilterResult:
        """Apply the Baxter-King filter to the provided data.
        
        Args:
            data: The data to filter
            low: Lower cutoff frequency in periods (overrides the instance parameter if provided)
            high: Upper cutoff frequency in periods (overrides the instance parameter if provided)
            K: Number of lags/leads (overrides the instance parameter if provided)
            **kwargs: Additional keyword arguments for filtering
        
        Returns:
            FilterResult: The filter results
        
        Raises:
            ValueError: If the data is invalid
            NumericError: If the filtering fails
        """
        # Validate and prepare data
        data_array = self.validate_data(data)
        self._data = data_array
        
        # Update parameters if provided
        if low is not None:
            self.low = low
        if high is not None:
            self.high = high
        if K is not None:
            self.K = K
        
        try:
            # Get the data length
            n = len(data_array)
            
            # Check if K is too large for the data
            if 2 * self.K >= n:
                raise ValueError(
                    f"K is too large for the data length. 2*K ({2*self.K}) must be less than n ({n})."
                )
            
            # Convert cutoff frequencies to angular frequencies
            omega_low = 2 * np.pi / self.high
            omega_high = 2 * np.pi / self.low
            
            # Compute filter weights
            # The ideal band-pass filter has weights:
            # b_j = (sin(omega_high*j) - sin(omega_low*j)) / (pi*j) for j != 0
            # b_0 = (omega_high - omega_low) / pi
            b = np.zeros(2 * self.K + 1)
            b[self.K] = (omega_high - omega_low) / np.pi
            j = np.arange(1, self.K + 1)
            weights = (np.sin(omega_high * j) - np.sin(omega_low * j)) / (np.pi * j)
            b[self.K + j] = weights
            b[self.K - j] = weights
            
            # Ensure the weights sum to zero (remove zero frequency)
            b -= np.mean(b)
            
            # Apply the filter using convolution
            # We lose K observations at the beginning and end
            cycle = np.convolve(data_array, b, mode='valid')
            
            # Create trend by removing the cycle
            # The trend is defined only for the same range as the cycle
            trend = np.zeros_like(cycle)
            trend_data = data_array[self.K:-self.K]
            trend = trend_data - cycle
            
            # Pad the cycle and trend to match the original data length
            padded_cycle = np.zeros_like(data_array)
            padded_cycle[self.K:-self.K] = cycle
            
            padded_trend = np.zeros_like(data_array)
            padded_trend[self.K:-self.K] = trend
            
            # Store the results
            self._trend = padded_trend
            self._cycle = padded_cycle
            self._components["weights"] = b
            self._fitted = True
            
            # Create result object
            parameters = {
                "low": self.low,
                "high": self.high,
                "K": self.K
            }
            result = self._create_result_object(
                data=data_array,
                trend=padded_trend,
                cycle=padded_cycle,
                components={"weights": b},
                parameters=parameters
            )
            
            return result
            
        except Exception as e:
            raise NumericError(
                f"BK filter failed: {e}",
                operation="BK filter",
                error_type="computation",
                details=str(e)
            )


class CFFilter(FilterBase[CFFilterParameters]):
    """Christiano-Fitzgerald band-pass filter implementation.
    
    The Christiano-Fitzgerald filter is an asymmetric band-pass filter that
    isolates the cyclical component of a time series by removing high and
    low frequency components. Unlike the Baxter-King filter, it uses the
    full sample and can handle non-stationary data.
    
    Attributes:
        low: Lower cutoff frequency in periods
        high: Upper cutoff frequency in periods
        drift: Whether to remove drift from the series
        symmetric: Whether to use symmetric filter weights
    """
    
    def __init__(self, 
                low: float = 6.0, 
                high: float = 32.0, 
                drift: bool = False,
                symmetric: bool = False,
                name: str = "Christiano-Fitzgerald Filter"):
        """Initialize the CF filter.
        
        Args:
            low: Lower cutoff frequency in periods (default: 6.0)
            high: Upper cutoff frequency in periods (default: 32.0)
            drift: Whether to remove drift from the series (default: False)
            symmetric: Whether to use symmetric filter weights (default: False)
            name: A descriptive name for the filter
        """
        super().__init__(name=name)
        self.params = CFFilterParameters(
            low=low, 
            high=high, 
            drift=drift,
            symmetric=symmetric
        )
    
    @property
    def low(self) -> float:
        """Get the lower cutoff frequency.
        
        Returns:
            float: The lower cutoff frequency
        """
        return self.params.low
    
    @low.setter
    def low(self, value: float) -> None:
        """Set the lower cutoff frequency.
        
        Args:
            value: The new lower cutoff frequency
        
        Raises:
            ParameterError: If the parameter is invalid
        """
        self.params = CFFilterParameters(
            low=value, 
            high=self.params.high, 
            drift=self.params.drift,
            symmetric=self.params.symmetric
        )
    
    @property
    def high(self) -> float:
        """Get the upper cutoff frequency.
        
        Returns:
            float: The upper cutoff frequency
        """
        return self.params.high
    
    @high.setter
    def high(self, value: float) -> None:
        """Set the upper cutoff frequency.
        
        Args:
            value: The new upper cutoff frequency
        
        Raises:
            ParameterError: If the parameter is invalid
        """
        self.params = CFFilterParameters(
            low=self.params.low, 
            high=value, 
            drift=self.params.drift,
            symmetric=self.params.symmetric
        )
    
    @property
    def drift(self) -> bool:
        """Get the drift flag.
        
        Returns:
            bool: Whether to remove drift from the series
        """
        return self.params.drift
    
    @drift.setter
    def drift(self, value: bool) -> None:
        """Set the drift flag.
        
        Args:
            value: The new drift flag
        """
        self.params = CFFilterParameters(
            low=self.params.low, 
            high=self.params.high, 
            drift=value,
            symmetric=self.params.symmetric
        )
    
    @property
    def symmetric(self) -> bool:
        """Get the symmetric flag.
        
        Returns:
            bool: Whether to use symmetric filter weights
        """
        return self.params.symmetric
    
    @symmetric.setter
    def symmetric(self, value: bool) -> None:
        """Set the symmetric flag.
        
        Args:
            value: The new symmetric flag
        """
        self.params = CFFilterParameters(
            low=self.params.low, 
            high=self.params.high, 
            drift=self.params.drift,
            symmetric=value
        )
    
    def filter(self, 
              data: TimeSeriesData, 
              low: Optional[float] = None,
              high: Optional[float] = None,
              drift: Optional[bool] = None,
              symmetric: Optional[bool] = None,
              **kwargs: Any) -> FilterResult:
        """Apply the Christiano-Fitzgerald filter to the provided data.
        
        Args:
            data: The data to filter
            low: Lower cutoff frequency in periods (overrides the instance parameter if provided)
            high: Upper cutoff frequency in periods (overrides the instance parameter if provided)
            drift: Whether to remove drift from the series (overrides the instance parameter if provided)
            symmetric: Whether to use symmetric filter weights (overrides the instance parameter if provided)
            **kwargs: Additional keyword arguments for filtering
        
        Returns:
            FilterResult: The filter results
        
        Raises:
            ValueError: If the data is invalid
            NumericError: If the filtering fails
        """
        # Validate and prepare data
        data_array = self.validate_data(data)
        self._data = data_array
        
        # Update parameters if provided
        if low is not None:
            self.low = low
        if high is not None:
            self.high = high
        if drift is not None:
            self.drift = drift
        if symmetric is not None:
            self.symmetric = symmetric
        
        try:
            # Get the data length
            n = len(data_array)
            
            # Convert cutoff frequencies to angular frequencies
            omega_low = 2 * np.pi / self.high
            omega_high = 2 * np.pi / self.low
            
            # Remove drift if requested
            if self.drift:
                # Estimate and remove linear trend
                t = np.arange(n)
                A = np.vstack([t, np.ones(n)]).T
                beta = np.linalg.lstsq(A, data_array, rcond=None)[0]
                trend_line = beta[0] * t + beta[1]
                detrended_data = data_array - trend_line
            else:
                detrended_data = data_array
            
            # Initialize cycle array
            cycle = np.zeros_like(data_array)
            
            # Compute filter weights and apply filter for each time point
            for t in range(n):
                if self.symmetric:
                    # Symmetric filter (same as BK but without truncation)
                    j_max = min(t, n - t - 1)
                    weights = np.zeros(2 * j_max + 1)
                    weights[j_max] = (omega_high - omega_low) / np.pi
                    j = np.arange(1, j_max + 1)
                    j_weights = (np.sin(omega_high * j) - np.sin(omega_low * j)) / (np.pi * j)
                    weights[j_max + j] = j_weights
                    weights[j_max - j] = j_weights
                    
                    # Ensure the weights sum to zero (remove zero frequency)
                    weights -= np.mean(weights)
                    
                    # Apply the filter
                    start = t - j_max
                    end = t + j_max + 1
                    cycle[t] = np.sum(weights * detrended_data[start:end])
                else:
                    # Asymmetric filter (CF)
                    # Compute weights for the specific time point
                    b0 = (omega_high - omega_low) / np.pi
                    
                    # Weights for future observations
                    if t < n - 1:
                        j = np.arange(1, n - t)
                        bj_future = (np.sin(omega_high * j) - np.sin(omega_low * j)) / (np.pi * j)
                    else:
                        bj_future = np.array([])
                    
                    # Weights for past observations
                    if t > 0:
                        j = np.arange(1, t + 1)
                        bj_past = (np.sin(omega_high * j) - np.sin(omega_low * j)) / (np.pi * j)
                    else:
                        bj_past = np.array([])
                    
                    # Combine weights
                    weights = np.concatenate([bj_past[::-1], [b0], bj_future])
                    
                    # Adjust weights to ensure they sum to zero
                    weights -= np.mean(weights)
                    
                    # Apply the filter
                    cycle[t] = np.sum(weights * detrended_data)
            
            # Add back the trend line if drift was removed
            if self.drift:
                cycle = cycle + trend_line
                trend = data_array - cycle
            else:
                trend = data_array - cycle
            
            # Store the results
            self._trend = trend
            self._cycle = cycle
            self._fitted = True
            
            # Create result object
            parameters = {
                "low": self.low,
                "high": self.high,
                "drift": self.drift,
                "symmetric": self.symmetric
            }
            result = self._create_result_object(
                data=data_array,
                trend=trend,
                cycle=cycle,
                parameters=parameters
            )
            
            return result
            
        except Exception as e:
            raise NumericError(
                f"CF filter failed: {e}",
                operation="CF filter",
                error_type="computation",
                details=str(e)
            )


class BNDecomposition(FilterBase[BNDecompositionParameters]):
    """Beveridge-Nelson decomposition implementation.
    
    The Beveridge-Nelson decomposition separates a time series into a permanent
    (trend) component and a transitory (cycle) component. The permanent component
    is a random walk with drift, while the transitory component is stationary.
    """
    
    def __init__(self, 
                ar_order: int = 1, 
                ma_order: int = 0, 
                forecast_horizon: int = 40,
                name: str = "Beveridge-Nelson Decomposition"):
        """Initialize the BN decomposition.
        
        Args:
            ar_order: Order of the autoregressive component (default: 1)
            ma_order: Order of the moving average component (default: 0)
            forecast_horizon: Forecast horizon for long-run component (default: 40)
            name: A descriptive name for the filter
        """
        super().__init__(name=name)
        self.params = BNDecompositionParameters(
            ar_order=ar_order, 
            ma_order=ma_order, 
            forecast_horizon=forecast_horizon
        )
    
    @property
    def ar_order(self) -> int:
        """Get the AR order.
        
        Returns:
            int: The AR order
        """
        return self.params.ar_order
    
    @ar_order.setter
    def ar_order(self, value: int) -> None:
        """Set the AR order.
        
        Args:
            value: The new AR order
        
        Raises:
            ParameterError: If the parameter is invalid
        """
        self.params = BNDecompositionParameters(
            ar_order=value, 
            ma_order=self.params.ma_order, 
            forecast_horizon=self.params.forecast_horizon
        )
    
    @property
    def ma_order(self) -> int:
        """Get the MA order.
        
        Returns:
            int: The MA order
        """
        return self.params.ma_order
    
    @ma_order.setter
    def ma_order(self, value: int) -> None:
        """Set the MA order.
        
        Args:
            value: The new MA order
        
        Raises:
            ParameterError: If the parameter is invalid
        """
        self.params = BNDecompositionParameters(
            ar_order=self.params.ar_order, 
            ma_order=value, 
            forecast_horizon=self.params.forecast_horizon
        )
    
    @property
    def forecast_horizon(self) -> int:
        """Get the forecast horizon.
        
        Returns:
            int: The forecast horizon
        """
        return self.params.forecast_horizon
    
    @forecast_horizon.setter
    def forecast_horizon(self, value: int) -> None:
        """Set the forecast horizon.
        
        Args:
            value: The new forecast horizon
        
        Raises:
            ParameterError: If the parameter is invalid
        """
        self.params = BNDecompositionParameters(
            ar_order=self.params.ar_order, 
            ma_order=self.params.ma_order, 
            forecast_horizon=value
        )
    
    def filter(self, 
              data: TimeSeriesData, 
              ar_order: Optional[int] = None,
              ma_order: Optional[int] = None,
              forecast_horizon: Optional[int] = None,
              **kwargs: Any) -> FilterResult:
        """Apply the Beveridge-Nelson decomposition to the provided data.
        
        Args:
            data: The data to filter
            ar_order: Order of the autoregressive component (overrides the instance parameter if provided)
            ma_order: Order of the moving average component (overrides the instance parameter if provided)
            forecast_horizon: Forecast horizon for long-run component (overrides the instance parameter if provided)
            **kwargs: Additional keyword arguments for filtering
        
        Returns:
            FilterResult: The filter results
        
        Raises:
            ValueError: If the data is invalid
            NumericError: If the filtering fails
        """
        # Validate and prepare data
        data_array = self.validate_data(data)
        self._data = data_array
        
        # Update parameters if provided
        if ar_order is not None:
            self.ar_order = ar_order
        if ma_order is not None:
            self.ma_order = ma_order
        if forecast_horizon is not None:
            self.forecast_horizon = forecast_horizon
        
        try:
            # Import statsmodels for ARIMA estimation
            import statsmodels.api as sm
            
            # Get the data length
            n = len(data_array)
            
            # Compute first differences
            diff_data = np.diff(data_array)
            
            # Fit ARIMA model to the differenced data
            arima_model = sm.tsa.ARIMA(
                diff_data, 
                order=(self.ar_order, 0, self.ma_order)
            )
            arima_result = arima_model.fit()
            
            # Get the model parameters
            ar_params = np.zeros(self.ar_order)
            ma_params = np.zeros(self.ma_order)
            
            # Extract AR parameters
            if self.ar_order > 0:
                for i in range(self.ar_order):
                    param_name = f'ar.L{i+1}'
                    if param_name in arima_result.params:
                        ar_params[i] = arima_result.params[param_name]
            
            # Extract MA parameters
            if self.ma_order > 0:
                for i in range(self.ma_order):
                    param_name = f'ma.L{i+1}'
                    if param_name in arima_result.params:
                        ma_params[i] = arima_result.params[param_name]
            
            # Get the constant term
            const = arima_result.params.get('const', 0.0)
            
            # Compute the long-run multiplier (sum of AR coefficients)
            ar_sum = np.sum(ar_params)
            long_run_multiplier = 1.0 / (1.0 - ar_sum) if ar_sum < 1.0 else 1.0
            
            # Compute the long-run forecast (permanent component)
            # The permanent component at time t is the current value plus
            # the expected sum of all future changes
            trend = np.zeros_like(data_array)
            trend[0] = data_array[0]
            
            # Generate forecasts for each time point
            for t in range(1, n):
                # Initialize forecast sum
                forecast_sum = 0.0
                
                # Generate forecasts for the specified horizon
                for h in range(1, self.forecast_horizon + 1):
                    # For h=1, use the model's one-step ahead forecast
                    if h == 1:
                        forecast = const
                        for i in range(min(t, self.ar_order)):
                            if t - i - 1 >= 0:
                                forecast += ar_params[i] * diff_data[t - i - 1]
                        for i in range(min(t, self.ma_order)):
                            if t - i - 1 >= 0:
                                residual = diff_data[t - i - 1] - arima_result.fittedvalues[t - i - 1]
                                forecast += ma_params[i] * residual
                    else:
                        # For h>1, only the constant and AR terms affect the forecast
                        forecast = const
                        for i in range(self.ar_order):
                            if h - i - 1 > 0:
                                forecast += ar_params[i] * forecast_sum
                    
                    # Add to the forecast sum
                    forecast_sum += forecast
                
                # Compute the long-run forecast (permanent component)
                # The permanent component is the current value plus
                # the expected sum of all future changes
                trend[t] = data_array[t] + long_run_multiplier * const * self.forecast_horizon
                
                # Adjust for the AR component
                if ar_sum < 1.0:
                    trend[t] += forecast_sum
            
            # Compute the cycle as the difference between the data and the trend
            cycle = data_array - trend
            
            # Store the results
            self._trend = trend
            self._cycle = cycle
            self._components["diff_data"] = np.concatenate([[0], diff_data])  # Pad with 0 for the first observation
            self._components["long_run_multiplier"] = np.full_like(data_array, long_run_multiplier)
            self._fitted = True
            
            # Create result object
            parameters = {
                "ar_order": self.ar_order,
                "ma_order": self.ma_order,
                "forecast_horizon": self.forecast_horizon,
                "ar_params": ar_params.tolist(),
                "ma_params": ma_params.tolist(),
                "const": const,
                "long_run_multiplier": long_run_multiplier
            }
            components = {
                "diff_data": np.concatenate([[0], diff_data]),
                "long_run_multiplier": np.full_like(data_array, long_run_multiplier)
            }
            result = self._create_result_object(
                data=data_array,
                trend=trend,
                cycle=cycle,
                components=components,
                parameters=parameters
            )
            
            return result
            
        except Exception as e:
            raise NumericError(
                f"Beveridge-Nelson decomposition failed: {e}",
                operation="BN decomposition",
                error_type="computation",
                details=str(e)
            )


class HamiltonFilter(FilterBase[HamiltonFilterParameters]):
    """Hamilton filter implementation.
    
    The Hamilton filter is an alternative to the Hodrick-Prescott filter for
    extracting the cyclical component of a time series. It is based on the
    regression of the series at time t+h on its lags at time t and earlier.
    
    Attributes:
        h: Forecast horizon
        p: Order of the autoregressive component
    """
    
    def __init__(self, 
                h: int = 8, 
                p: int = 4,
                name: str = "Hamilton Filter"):
        """Initialize the Hamilton filter.
        
        Args:
            h: Forecast horizon (default: 8)
            p: Order of the autoregressive component (default: 4)
            name: A descriptive name for the filter
        """
        super().__init__(name=name)
        self.params = HamiltonFilterParameters(h=h, p=p)
    
    @property
    def h(self) -> int:
        """Get the forecast horizon.
        
        Returns:
            int: The forecast horizon
        """
        return self.params.h
    
    @h.setter
    def h(self, value: int) -> None:
        """Set the forecast horizon.
        
        Args:
            value: The new forecast horizon
        
        Raises:
            ParameterError: If the parameter is invalid
        """
        self.params = HamiltonFilterParameters(h=value, p=self.params.p)
    
    @property
    def p(self) -> int:
        """Get the AR order.
        
        Returns:
            int: The AR order
        """
        return self.params.p
    
    @p.setter
    def p(self, value: int) -> None:
        """Set the AR order.
        
        Args:
            value: The new AR order
        
        Raises:
            ParameterError: If the parameter is invalid
        """
        self.params = HamiltonFilterParameters(h=self.params.h, p=value)
    
    def filter(self, 
              data: TimeSeriesData, 
              h: Optional[int] = None,
              p: Optional[int] = None,
              **kwargs: Any) -> FilterResult:
        """Apply the Hamilton filter to the provided data.
        
        Args:
            data: The data to filter
            h: Forecast horizon (overrides the instance parameter if provided)
            p: Order of the autoregressive component (overrides the instance parameter if provided)
            **kwargs: Additional keyword arguments for filtering
        
        Returns:
            FilterResult: The filter results
        
        Raises:
            ValueError: If the data is invalid
            NumericError: If the filtering fails
        """
        # Validate and prepare data
        data_array = self.validate_data(data)
        self._data = data_array
        
        # Update parameters if provided
        if h is not None:
            self.h = h
        if p is not None:
            self.p = p
        
        try:
            # Get the data length
            n = len(data_array)
            
            # Check if we have enough data
            if n <= self.h + self.p:
                raise ValueError(
                    f"Data length ({n}) must be greater than h + p ({self.h + self.p})"
                )
            
            # Create the regression variables
            y = data_array[self.h:]  # Target: y_{t+h}
            X = np.zeros((n - self.h, self.p))
            
            # Fill the X matrix with lags
            for i in range(self.p):
                X[:, i] = data_array[self.h - i - 1:n - i - 1]
            
            # Add a constant term
            X = np.column_stack([np.ones(n - self.h), X])
            
            # Estimate the regression coefficients
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            
            # Compute the fitted values
            y_hat = X @ beta
            
            # The cycle is the residual from the regression
            cycle_partial = y - y_hat
            
            # Create the full cycle array (padded with zeros for the first h observations)
            cycle = np.zeros_like(data_array)
            cycle[self.h:] = cycle_partial
            
            # The trend is the original data minus the cycle
            trend = data_array - cycle
            
            # Store the results
            self._trend = trend
            self._cycle = cycle
            self._components["coefficients"] = beta
            self._components["fitted_values"] = np.concatenate([np.zeros(self.h), y_hat])
            self._fitted = True
            
            # Create result object
            parameters = {
                "h": self.h,
                "p": self.p,
                "coefficients": beta.tolist()
            }
            components = {
                "coefficients": beta,
                "fitted_values": np.concatenate([np.zeros(self.h), y_hat])
            }
            result = self._create_result_object(
                data=data_array,
                trend=trend,
                cycle=cycle,
                components=components,
                parameters=parameters
            )
            
            return result
            
        except Exception as e:
            raise NumericError(
                f"Hamilton filter failed: {e}",
                operation="Hamilton filter",
                error_type="computation",
                details=str(e)
            )


class UCFilter(FilterBase[UCFilterParameters]):
    """Unobserved Components filter implementation.
    
    The Unobserved Components filter decomposes a time series into trend,
    cycle, and irregular components using a state-space model.
    
    Attributes:
        trend_order: Order of the trend component (0, 1, or 2)
        cycle_periods: Number of periods in the cycle
        damping_factor: Damping factor for the cycle
        irregular_var: Variance of the irregular component
    """
    
    def __init__(self, 
                trend_order: int = 1, 
                cycle_periods: float = 20.0,
                damping_factor: float = 0.9,
                irregular_var: float = 1.0,
                name: str = "Unobserved Components Filter"):
        """Initialize the UC filter.
        
        Args:
            trend_order: Order of the trend component (0, 1, or 2) (default: 1)
            cycle_periods: Number of periods in the cycle (default: 20.0)
            damping_factor: Damping factor for the cycle (default: 0.9)
            irregular_var: Variance of the irregular component (default: 1.0)
            name: A descriptive name for the filter
        """
        super().__init__(name=name)
        self.params = UCFilterParameters(
            trend_order=trend_order,
            cycle_periods=cycle_periods,
            damping_factor=damping_factor,
            irregular_var=irregular_var
        )
    
    @property
    def trend_order(self) -> int:
        """Get the trend order.
        
        Returns:
            int: The trend order
        """
        return self.params.trend_order
    
    @trend_order.setter
    def trend_order(self, value: int) -> None:
        """Set the trend order.
        
        Args:
            value: The new trend order
        
        Raises:
            ParameterError: If the parameter is invalid
        """
        self.params = UCFilterParameters(
            trend_order=value,
            cycle_periods=self.params.cycle_periods,
            damping_factor=self.params.damping_factor,
            irregular_var=self.params.irregular_var
        )
    
    @property
    def cycle_periods(self) -> float:
        """Get the number of periods in the cycle.
        
        Returns:
            float: The number of periods in the cycle
        """
        return self.params.cycle_periods
    
    @cycle_periods.setter
    def cycle_periods(self, value: float) -> None:
        """Set the number of periods in the cycle.
        
        Args:
            value: The new number of periods in the cycle
        
        Raises:
            ParameterError: If the parameter is invalid
        """
        self.params = UCFilterParameters(
            trend_order=self.params.trend_order,
            cycle_periods=value,
            damping_factor=self.params.damping_factor,
            irregular_var=self.params.irregular_var
        )
    
    @property
    def damping_factor(self) -> float:
        """Get the damping factor.
        
        Returns:
            float: The damping factor
        """
        return self.params.damping_factor
    
    @damping_factor.setter
    def damping_factor(self, value: float) -> None:
        """Set the damping factor.
        
        Args:
            value: The new damping factor
        
        Raises:
            ParameterError: If the parameter is invalid
        """
        self.params = UCFilterParameters(
            trend_order=self.params.trend_order,
            cycle_periods=self.params.cycle_periods,
            damping_factor=value,
            irregular_var=self.params.irregular_var
        )
    
    @property
    def irregular_var(self) -> float:
        """Get the irregular component variance.
        
        Returns:
            float: The irregular component variance
        """
        return self.params.irregular_var
    
    @irregular_var.setter
    def irregular_var(self, value: float) -> None:
        """Set the irregular component variance.
        
        Args:
            value: The new irregular component variance
        
        Raises:
            ParameterError: If the parameter is invalid
        """
        self.params = UCFilterParameters(
            trend_order=self.params.trend_order,
            cycle_periods=self.params.cycle_periods,
            damping_factor=self.params.damping_factor,
            irregular_var=value
        )
    
    def filter(self, 
              data: TimeSeriesData, 
              trend_order: Optional[int] = None,
              cycle_periods: Optional[float] = None,
              damping_factor: Optional[float] = None,
              irregular_var: Optional[float] = None,
              **kwargs: Any) -> FilterResult:
        """Apply the Unobserved Components filter to the provided data.
        
        Args:
            data: The data to filter
            trend_order: Order of the trend component (overrides the instance parameter if provided)
            cycle_periods: Number of periods in the cycle (overrides the instance parameter if provided)
            damping_factor: Damping factor for the cycle (overrides the instance parameter if provided)
            irregular_var: Variance of the irregular component (overrides the instance parameter if provided)
            **kwargs: Additional keyword arguments for filtering
        
        Returns:
            FilterResult: The filter results
        
        Raises:
            ValueError: If the data is invalid
            NumericError: If the filtering fails
        """
        # Validate and prepare data
        data_array = self.validate_data(data)
        self._data = data_array
        
        # Update parameters if provided
        if trend_order is not None:
            self.trend_order = trend_order
        if cycle_periods is not None:
            self.cycle_periods = cycle_periods
        if damping_factor is not None:
            self.damping_factor = damping_factor
        if irregular_var is not None:
            self.irregular_var = irregular_var
        
        try:
            # Import statsmodels for state space modeling
            import statsmodels.api as sm
            
            # Get the data length
            n = len(data_array)
            
            # Create the unobserved components model
            # The model specification depends on the trend order
            if self.trend_order == 0:
                # No trend (only cycle and irregular)
                model = sm.tsa.UnobservedComponents(
                    data_array,
                    level=False,
                    trend=False,
                    cycle=True,
                    damped_cycle=True,
                    stochastic_cycle=True
                )
            elif self.trend_order == 1:
                # Local level model (random walk trend)
                model = sm.tsa.UnobservedComponents(
                    data_array,
                    level=True,
                    trend=False,
                    cycle=True,
                    damped_cycle=True,
                    stochastic_cycle=True
                )
            elif self.trend_order == 2:
                # Local linear trend model
                model = sm.tsa.UnobservedComponents(
                    data_array,
                    level=True,
                    trend=True,
                    cycle=True,
                    damped_cycle=True,
                    stochastic_cycle=True
                )
            else:
                raise ValueError(f"Invalid trend_order: {self.trend_order}")
            
            # Set initial parameters
            # The parameters depend on the model specification
            if self.trend_order == 0:
                # No trend
                initial_params = [
                    self.irregular_var,  # irregular variance
                    2 * np.pi / self.cycle_periods,  # frequency
                    self.damping_factor,  # damping
                    0.1  # cycle variance
                ]
            elif self.trend_order == 1:
                # Local level model
                initial_params = [
                    self.irregular_var,  # irregular variance
                    0.1,  # level variance
                    2 * np.pi / self.cycle_periods,  # frequency
                    self.damping_factor,  # damping
                    0.1  # cycle variance
                ]
            else:
                # Local linear trend model
                initial_params = [
                    self.irregular_var,  # irregular variance
                    0.1,  # level variance
                    0.1,  # trend variance
                    2 * np.pi / self.cycle_periods,  # frequency
                    self.damping_factor,  # damping
                    0.1  # cycle variance
                ]
            
            # Fit the model
            result = model.fit(initial_params)
            
            # Extract the components
            components = result.estimate_separate_components()
            
            # Extract trend and cycle
            if self.trend_order == 0:
                # No trend, so the trend is zero
                trend = np.zeros_like(data_array)
                cycle = components['cycle']
            elif self.trend_order == 1:
                # Local level model
                trend = components['level']
                cycle = components['cycle']
            else:
                # Local linear trend model
                trend = components['level'] + components['trend']
                cycle = components['cycle']
            
            # Extract irregular component
            irregular = components['irregular']
            
            # Store the results
            self._trend = trend
            self._cycle = cycle
            self._components["irregular"] = irregular
            self._fitted = True
            
            # Create result object
            parameters = {
                "trend_order": self.trend_order,
                "cycle_periods": self.cycle_periods,
                "damping_factor": self.damping_factor,
                "irregular_var": self.irregular_var,
                "estimated_params": result.params.tolist()
            }
            components_dict = {
                "irregular": irregular
            }
            result = self._create_result_object(
                data=data_array,
                trend=trend,
                cycle=cycle,
                components=components_dict,
                parameters=parameters
            )
            
            return result
            
        except Exception as e:
            raise NumericError(
                f"Unobserved Components filter failed: {e}",
                operation="UC filter",
                error_type="computation",
                details=str(e)
            )


# Convenience functions for direct filtering

def hp_filter(data: TimeSeriesData, 
             lambda_: float = 1600.0,
             **kwargs: Any) -> FilterResult:
    """Apply the Hodrick-Prescott filter to the provided data.
    
    Args:
        data: The data to filter
        lambda_: Smoothing parameter (default: 1600.0 for quarterly data)
        **kwargs: Additional keyword arguments for filtering
    
    Returns:
        FilterResult: The filter results
    
    Raises:
        ValueError: If the data is invalid
        NumericError: If the filtering fails
    """
    filter_obj = HPFilter(lambda_=lambda_)
    return filter_obj.filter(data, **kwargs)

def bk_filter(data: TimeSeriesData, 
             low: float = 6.0, 
             high: float = 32.0, 
             K: int = 12,
             **kwargs: Any) -> FilterResult:
    """Apply the Baxter-King filter to the provided data.
    
    Args:
        data: The data to filter
        low: Lower cutoff frequency in periods (default: 6.0)
        high: Upper cutoff frequency in periods (default: 32.0)
        K: Number of lags/leads (default: 12)
        **kwargs: Additional keyword arguments for filtering
    
    Returns:
        FilterResult: The filter results
    
    Raises:
        ValueError: If the data is invalid
        NumericError: If the filtering fails
    """
    filter_obj = BKFilter(low=low, high=high, K=K)
    return filter_obj.filter(data, **kwargs)

def cf_filter(data: TimeSeriesData, 
             low: float = 6.0, 
             high: float = 32.0, 
             drift: bool = False,
             symmetric: bool = False,
             **kwargs: Any) -> FilterResult:
    """Apply the Christiano-Fitzgerald filter to the provided data.
    
    Args:
        data: The data to filter
        low: Lower cutoff frequency in periods (default: 6.0)
        high: Upper cutoff frequency in periods (default: 32.0)
        drift: Whether to remove drift from the series (default: False)
        symmetric: Whether to use symmetric filter weights (default: False)
        **kwargs: Additional keyword arguments for filtering
    
    Returns:
        FilterResult: The filter results
    
    Raises:
        ValueError: If the data is invalid
        NumericError: If the filtering fails
    """
    filter_obj = CFFilter(low=low, high=high, drift=drift, symmetric=symmetric)
    return filter_obj.filter(data, **kwargs)

def bn_decomposition(data: TimeSeriesData, 
                    ar_order: int = 1, 
                    ma_order: int = 0, 
                    forecast_horizon: int = 40,
                    **kwargs: Any) -> FilterResult:
    """Apply the Beveridge-Nelson decomposition to the provided data.
    
    Args:
        data: The data to filter
        ar_order: Order of the autoregressive component (default: 1)
        ma_order: Order of the moving average component (default: 0)
        forecast_horizon: Forecast horizon for long-run component (default: 40)
        **kwargs: Additional keyword arguments for filtering
    
    Returns:
        FilterResult: The filter results
    
    Raises:
        ValueError: If the data is invalid
        NumericError: If the filtering fails
    """
    filter_obj = BNDecomposition(ar_order=ar_order, ma_order=ma_order, 
                                forecast_horizon=forecast_horizon)
    return filter_obj.filter(data, **kwargs)

def hamilton_filter(data: TimeSeriesData, 
                   h: int = 8, 
                   p: int = 4,
                   **kwargs: Any) -> FilterResult:
    """Apply the Hamilton filter to the provided data.
    
    Args:
        data: The data to filter
        h: Forecast horizon (default: 8)
        p: Order of the autoregressive component (default: 4)
        **kwargs: Additional keyword arguments for filtering
    
    Returns:
        FilterResult: The filter results
    
    Raises:
        ValueError: If the data is invalid
        NumericError: If the filtering fails
    """
    filter_obj = HamiltonFilter(h=h, p=p)
    return filter_obj.filter(data, **kwargs)

def uc_filter(data: TimeSeriesData, 
             trend_order: int = 1, 
             cycle_periods: float = 20.0,
             damping_factor: float = 0.9,
             irregular_var: float = 1.0,
             **kwargs: Any) -> FilterResult:
    """Apply the Unobserved Components filter to the provided data.
    
    Args:
        data: The data to filter
        trend_order: Order of the trend component (0, 1, or 2) (default: 1)
        cycle_periods: Number of periods in the cycle (default: 20.0)
        damping_factor: Damping factor for the cycle (default: 0.9)
        irregular_var: Variance of the irregular component (default: 1.0)
        **kwargs: Additional keyword arguments for filtering
    
    Returns:
        FilterResult: The filter results
    
    Raises:
        ValueError: If the data is invalid
        NumericError: If the filtering fails
    """
    filter_obj = UCFilter(trend_order=trend_order, cycle_periods=cycle_periods,
                         damping_factor=damping_factor, irregular_var=irregular_var)
    return filter_obj.filter(data, **kwargs)
