# mfe/models/time_series/unit_root.py
"""
Unit root tests for time series stationarity analysis.

This module implements various unit root tests to determine whether a time series
is stationary or contains a unit root. The primary test implemented is the
Augmented Dickey-Fuller (ADF) test, which is widely used in econometrics to test
for stationarity.

The module provides a comprehensive implementation with support for:
- Different deterministic trend specifications (none, constant, trend)
- Automatic lag selection using information criteria
- Critical value calculation and p-value computation
- Visualization of test results
- Both synchronous and asynchronous execution

These tests are essential for proper time series modeling, as many time series
models require stationary data for valid inference.
"""

import asyncio
import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any, Callable, Dict, List, Literal, Optional, Tuple, Union, cast, overload
)

import numpy as np
import pandas as pd
from scipy import stats, optimize
import statsmodels.api as sm

from mfe.core.base import ModelBase
from mfe.core.parameters import ParameterBase, validate_range
from mfe.core.exceptions import (
    ParameterError, DimensionError, TestError, warn_numeric
)

# Set up module-level logger
logger = logging.getLogger("mfe.models.time_series.unit_root")


class TrendType(str, Enum):
    """Enumeration of trend specifications for unit root tests."""
    
    NONE = "none"
    """No deterministic components (no constant, no trend)."""
    
    CONSTANT = "constant"
    """Include a constant term."""
    
    TREND = "trend"
    """Include a constant and a linear time trend."""
    
    @classmethod
    def from_string(cls, value: str) -> 'TrendType':
        """Convert a string to a TrendType.
        
        Args:
            value: String representation of trend type
        
        Returns:
            TrendType: The corresponding TrendType
        
        Raises:
            ValueError: If the string is not a valid trend type
        """
        value = value.lower()
        if value in ("none", "n", "nc"):
            return cls.NONE
        elif value in ("constant", "c", "ct"):
            return cls.CONSTANT
        elif value in ("trend", "t", "ctt"):
            return cls.TREND
        else:
            raise ValueError(
                f"Invalid trend type: {value}. Must be one of: "
                f"'none', 'constant', or 'trend'."
            )


class LagSelectionMethod(str, Enum):
    """Enumeration of lag selection methods for unit root tests."""
    
    FIXED = "fixed"
    """Use a fixed lag order specified by the user."""
    
    AIC = "aic"
    """Select lag order using Akaike Information Criterion."""
    
    BIC = "bic"
    """Select lag order using Bayesian Information Criterion."""
    
    HQIC = "hqic"
    """Select lag order using Hannan-Quinn Information Criterion."""
    
    T_STAT = "t-stat"
    """Select lag order using t-statistic significance of the highest lag."""
    
    @classmethod
    def from_string(cls, value: str) -> 'LagSelectionMethod':
        """Convert a string to a LagSelectionMethod.
        
        Args:
            value: String representation of lag selection method
        
        Returns:
            LagSelectionMethod: The corresponding LagSelectionMethod
        
        Raises:
            ValueError: If the string is not a valid lag selection method
        """
        value = value.lower()
        if value in ("fixed", "f", "none"):
            return cls.FIXED
        elif value in ("aic", "a"):
            return cls.AIC
        elif value in ("bic", "b", "sic"):
            return cls.BIC
        elif value in ("hqic", "hq", "h"):
            return cls.HQIC
        elif value in ("t-stat", "t", "tstat"):
            return cls.T_STAT
        else:
            raise ValueError(
                f"Invalid lag selection method: {value}. Must be one of: "
                f"'fixed', 'aic', 'bic', 'hqic', or 't-stat'."
            )


@dataclass
class UnitRootTestResult:
    """Result container for unit root tests.
    
    This class provides a standardized container for unit root test results,
    including test statistics, critical values, p-values, and additional
    information about the test configuration.
    
    Attributes:
        test_name: Name of the unit root test
        test_statistic: Test statistic value
        pvalue: p-value of the test
        critical_values: Dictionary of critical values at different significance levels
        lags: Number of lags used in the test
        nobs: Number of observations used in the test
        trend_type: Type of deterministic trend included in the test
        regression_results: Optional regression results from the test
        method: Method used for lag selection
        null_hypothesis: Description of the null hypothesis
        alternative_hypothesis: Description of the alternative hypothesis
    """
    
    test_name: str
    test_statistic: float
    pvalue: float
    critical_values: Dict[str, float]
    lags: int
    nobs: int
    trend_type: TrendType
    regression_results: Optional[Any] = None
    method: Optional[LagSelectionMethod] = None
    null_hypothesis: str = "The process contains a unit root (non-stationary)"
    alternative_hypothesis: str = "The process is stationary"
    
    def __post_init__(self) -> None:
        """Validate result object after initialization."""
        # Ensure critical values are properly formatted
        for key in list(self.critical_values.keys()):
            if not key.endswith("%"):
                # Convert numeric keys to percentage strings
                try:
                    numeric_key = float(key)
                    new_key = f"{numeric_key}%"
                    self.critical_values[new_key] = self.critical_values[key]
                    del self.critical_values[key]
                except (ValueError, TypeError):
                    # If not convertible to float, leave as is
                    pass
    
    def is_stationary(self, significance_level: float = 0.05) -> bool:
        """Check if the series is stationary at the given significance level.
        
        Args:
            significance_level: Significance level for the test (default: 0.05)
        
        Returns:
            bool: True if the null hypothesis of a unit root is rejected
                 (i.e., the series is stationary), False otherwise
        """
        return self.pvalue < significance_level
    
    def summary(self) -> str:
        """Generate a text summary of the test results.
        
        Returns:
            str: A formatted string containing the test results summary
        """
        header = f"{self.test_name} Results\n"
        header += "=" * len(header) + "\n\n"
        
        # Test configuration
        config = f"Test Configuration:\n"
        config += f"  Trend specification: {self.trend_type.value}\n"
        config += f"  Lags: {self.lags}"
        if self.method:
            config += f" (selected using {self.method.value})\n"
        else:
            config += "\n"
        config += f"  Number of observations: {self.nobs}\n\n"
        
        # Test results
        results = f"Test Results:\n"
        results += f"  Test statistic: {self.test_statistic:.6f}\n"
        results += f"  p-value: {self.pvalue:.6f}\n\n"
        
        # Critical values
        cv = f"Critical Values:\n"
        for level, value in sorted(self.critical_values.items()):
            cv += f"  {level}: {value:.6f}\n"
        cv += "\n"
        
        # Interpretation
        interp = f"Hypothesis:\n"
        interp += f"  Null Hypothesis: {self.null_hypothesis}\n"
        interp += f"  Alternative Hypothesis: {self.alternative_hypothesis}\n\n"
        
        # Conclusion
        conclusion = "Conclusion:\n"
        if self.pvalue < 0.01:
            conclusion += "  Reject the null hypothesis at the 1% significance level.\n"
            conclusion += "  The series appears to be stationary.\n"
        elif self.pvalue < 0.05:
            conclusion += "  Reject the null hypothesis at the 5% significance level.\n"
            conclusion += "  The series appears to be stationary.\n"
        elif self.pvalue < 0.1:
            conclusion += "  Reject the null hypothesis at the 10% significance level.\n"
            conclusion += "  The series appears to be stationary.\n"
        else:
            conclusion += "  Fail to reject the null hypothesis.\n"
            conclusion += "  The series appears to contain a unit root (non-stationary).\n"
        
        return header + config + results + cv + interp + conclusion
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result object to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the result object
        """
        result_dict = {
            "test_name": self.test_name,
            "test_statistic": self.test_statistic,
            "pvalue": self.pvalue,
            "critical_values": self.critical_values.copy(),
            "lags": self.lags,
            "nobs": self.nobs,
            "trend_type": self.trend_type.value,
            "method": self.method.value if self.method else None,
            "null_hypothesis": self.null_hypothesis,
            "alternative_hypothesis": self.alternative_hypothesis
        }
        
        # Regression results are not included as they may not be serializable
        
        return result_dict
    
    @classmethod
    def from_dict(cls, result_dict: Dict[str, Any]) -> 'UnitRootTestResult':
        """Create a result object from a dictionary.
        
        Args:
            result_dict: Dictionary containing result values
        
        Returns:
            UnitRootTestResult: Result object
        """
        # Convert trend_type string to TrendType enum
        trend_type = TrendType.from_string(result_dict["trend_type"])
        
        # Convert method string to LagSelectionMethod enum if present
        method = None
        if result_dict.get("method"):
            method = LagSelectionMethod.from_string(result_dict["method"])
        
        # Create result object
        return cls(
            test_name=result_dict["test_name"],
            test_statistic=result_dict["test_statistic"],
            pvalue=result_dict["pvalue"],
            critical_values=result_dict["critical_values"],
            lags=result_dict["lags"],
            nobs=result_dict["nobs"],
            trend_type=trend_type,
            method=method,
            null_hypothesis=result_dict.get("null_hypothesis", 
                "The process contains a unit root (non-stationary)"),
            alternative_hypothesis=result_dict.get("alternative_hypothesis", 
                "The process is stationary")
        )


class UnitRootTest(ModelBase):
    """Base class for unit root tests.
    
    This abstract base class defines the common interface that all unit root
    test implementations must follow, establishing a consistent API across
    the entire unit root testing module.
    """
    
    def __init__(self, name: str = "UnitRootTest"):
        """Initialize the unit root test.
        
        Args:
            name: A descriptive name for the test
        """
        super().__init__(name=name)
        self._results: Optional[UnitRootTestResult] = None
    
    def test(self, 
            data: Union[np.ndarray, pd.Series], 
            lags: Optional[int] = None,
            trend: Union[str, TrendType] = TrendType.CONSTANT,
            method: Union[str, LagSelectionMethod] = LagSelectionMethod.AIC,
            max_lags: Optional[int] = None,
            **kwargs: Any) -> UnitRootTestResult:
        """Perform the unit root test on the provided data.
        
        This method must be implemented by all subclasses to perform the
        specific unit root test.
        
        Args:
            data: The data to test for stationarity
            lags: Number of lags to include in the test regression
                 If None, lags will be selected using the specified method
            trend: Type of deterministic trend to include in the test regression
                  Can be a string or TrendType enum
            method: Method for selecting the optimal lag order if lags is None
                   Can be a string or LagSelectionMethod enum
            max_lags: Maximum number of lags to consider when selecting optimal lag order
                     If None, a default value based on the sample size will be used
            **kwargs: Additional keyword arguments for the test
        
        Returns:
            UnitRootTestResult: The test results
        
        Raises:
            ValueError: If the data or parameters are invalid
            TestError: If the test fails
        """
        raise NotImplementedError("test method must be implemented by subclass")
    
    async def test_async(self, 
                       data: Union[np.ndarray, pd.Series], 
                       lags: Optional[int] = None,
                       trend: Union[str, TrendType] = TrendType.CONSTANT,
                       method: Union[str, LagSelectionMethod] = LagSelectionMethod.AIC,
                       max_lags: Optional[int] = None,
                       **kwargs: Any) -> UnitRootTestResult:
        """Asynchronously perform the unit root test on the provided data.
        
        This method provides an asynchronous interface to the test method,
        allowing for non-blocking test execution in UI contexts.
        
        Args:
            data: The data to test for stationarity
            lags: Number of lags to include in the test regression
                 If None, lags will be selected using the specified method
            trend: Type of deterministic trend to include in the test regression
                  Can be a string or TrendType enum
            method: Method for selecting the optimal lag order if lags is None
                   Can be a string or LagSelectionMethod enum
            max_lags: Maximum number of lags to consider when selecting optimal lag order
                     If None, a default value based on the sample size will be used
            **kwargs: Additional keyword arguments for the test
        
        Returns:
            UnitRootTestResult: The test results
        
        Raises:
            ValueError: If the data or parameters are invalid
            TestError: If the test fails
        """
        # Create a coroutine that runs the synchronous test method in a thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: self.test(data, lags, trend, method, max_lags, **kwargs)
        )
        return result
    
    def validate_data(self, data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Validate the input data for unit root testing.
        
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
        if len(data_array) < 10:  # Arbitrary minimum length
            raise ValueError(
                f"Data length must be at least 10, got {len(data_array)}"
            )
        
        # Check for NaN and Inf values
        if np.isnan(data_array).any():
            raise ValueError("Data contains NaN values")
        
        if np.isinf(data_array).any():
            raise ValueError("Data contains infinite values")
        
        return data_array
    
    def _get_max_lags(self, nobs: int) -> int:
        """Calculate the default maximum number of lags based on sample size.
        
        Args:
            nobs: Number of observations
        
        Returns:
            int: Default maximum number of lags
        """
        # Common rule of thumb: int(12 * (nobs / 100) ** (1/4))
        return int(12 * (nobs / 100) ** (1/4))
    
    def _select_optimal_lags(self, 
                           data: np.ndarray, 
                           max_lags: int,
                           trend: TrendType,
                           method: LagSelectionMethod) -> int:
        """Select the optimal lag order using the specified method.
        
        Args:
            data: The data for the test
            max_lags: Maximum number of lags to consider
            trend: Type of deterministic trend to include
            method: Method for selecting the optimal lag order
        
        Returns:
            int: The optimal lag order
        
        Raises:
            ValueError: If the method is invalid
            TestError: If the lag selection fails
        """
        if method == LagSelectionMethod.FIXED:
            raise ValueError(
                "Cannot select optimal lags with method='fixed'. "
                "Please specify a lag order explicitly."
            )
        
        # Prepare data for ADF regression
        y = np.diff(data)
        y_lagged = data[:-1]
        nobs = len(y)
        
        # Initialize arrays to store information criteria
        aic_values = np.zeros(max_lags + 1)
        bic_values = np.zeros(max_lags + 1)
        hqic_values = np.zeros(max_lags + 1)
        t_stats = np.zeros(max_lags + 1)
        
        # Loop through different lag orders
        for lag in range(max_lags + 1):
            # Create lagged differences
            if lag > 0:
                dy_lags = np.zeros((nobs, lag))
                for i in range(lag):
                    dy_lags[i:, i] = np.diff(data)[:(nobs - i)]
            else:
                dy_lags = None
            
            # Create design matrix
            if trend == TrendType.NONE:
                X = np.column_stack([y_lagged[:(nobs)]])
                if lag > 0:
                    X = np.column_stack([X, dy_lags])
            elif trend == TrendType.CONSTANT:
                X = np.column_stack([np.ones(nobs), y_lagged[:(nobs)]])
                if lag > 0:
                    X = np.column_stack([X, dy_lags])
            elif trend == TrendType.TREND:
                X = np.column_stack([
                    np.ones(nobs), 
                    np.arange(1, nobs + 1), 
                    y_lagged[:(nobs)]
                ])
                if lag > 0:
                    X = np.column_stack([X, dy_lags])
            
            # Fit regression
            try:
                model = sm.OLS(y, X)
                results = model.fit()
                
                # Store information criteria
                aic_values[lag] = results.aic
                bic_values[lag] = results.bic
                hqic_values[lag] = results.hqic
                
                # Store t-statistic of the highest lag
                if lag > 0:
                    t_stats[lag] = results.tvalues[-1]
                else:
                    t_stats[lag] = 0
            except Exception as e:
                logger.warning(f"Error fitting model with lag {lag}: {e}")
                # Set to infinity to ensure this lag is not selected
                aic_values[lag] = np.inf
                bic_values[lag] = np.inf
                hqic_values[lag] = np.inf
                t_stats[lag] = 0
        
        # Select optimal lag order based on the specified method
        if method == LagSelectionMethod.AIC:
            optimal_lag = np.argmin(aic_values)
        elif method == LagSelectionMethod.BIC:
            optimal_lag = np.argmin(bic_values)
        elif method == LagSelectionMethod.HQIC:
            optimal_lag = np.argmin(hqic_values)
        elif method == LagSelectionMethod.T_STAT:
            # Find the highest lag with a significant t-statistic
            significant_lags = np.where(np.abs(t_stats) > 1.96)[0]
            if len(significant_lags) > 0:
                optimal_lag = np.max(significant_lags)
            else:
                optimal_lag = 0
        else:
            raise ValueError(f"Invalid lag selection method: {method}")
        
        return optimal_lag
    
    def plot_results(self, 
                    data: Union[np.ndarray, pd.Series],
                    figsize: Tuple[int, int] = (12, 8),
                    **kwargs: Any) -> Any:
        """Plot the test results.
        
        Args:
            data: The data used for the test
            figsize: Figure size (width, height) in inches
            **kwargs: Additional keyword arguments for plotting
        
        Returns:
            matplotlib.figure.Figure: The figure object
        
        Raises:
            ImportError: If matplotlib is not installed
            ValueError: If the test has not been performed
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Matplotlib is required for plotting. "
                "Install it with: pip install matplotlib"
            )
        
        if self._results is None:
            raise ValueError(
                "No test results available. Call test() first."
            )
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Plot the time series
        if isinstance(data, pd.Series):
            data.plot(ax=axes[0], title="Time Series")
        else:
            axes[0].plot(data)
            axes[0].set_title("Time Series")
        
        # Plot the autocorrelation function
        from statsmodels.graphics.tsaplots import plot_acf
        plot_acf(data, ax=axes[1], title="Autocorrelation Function")
        
        # Add test results as text
        result_text = (
            f"{self._results.test_name}\n"
            f"Test statistic: {self._results.test_statistic:.4f}\n"
            f"p-value: {self._results.pvalue:.4f}\n"
            f"Lags: {self._results.lags}\n"
            f"Trend: {self._results.trend_type.value}\n"
        )
        
        # Add critical values
        cv_text = "Critical values:\n"
        for level, value in sorted(self._results.critical_values.items()):
            cv_text += f"  {level}: {value:.4f}\n"
        
        # Add conclusion
        if self._results.pvalue < 0.05:
            conclusion = "Conclusion: Reject the null hypothesis.\nThe series appears to be stationary."
        else:
            conclusion = "Conclusion: Fail to reject the null hypothesis.\nThe series appears to be non-stationary."
        
        # Add text to the figure
        fig.text(0.02, 0.02, result_text + cv_text + conclusion, 
                 verticalalignment='bottom', horizontalalignment='left',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.15, 1, 1])
        return fig


class ADF(UnitRootTest):
    """Augmented Dickey-Fuller test for unit roots.
    
    The Augmented Dickey-Fuller (ADF) test is a test for a unit root in a time
    series sample. The null hypothesis is that the series contains a unit root
    (i.e., it is non-stationary), against the alternative that it is stationary.
    
    The test includes lagged differences of the series to account for serial
    correlation in the errors. The number of lags can be specified directly or
    selected automatically using information criteria.
    
    Attributes:
        regression_type: Type of regression used for the test
    """
    
    def __init__(self, name: str = "Augmented Dickey-Fuller Test"):
        """Initialize the ADF test.
        
        Args:
            name: A descriptive name for the test
        """
        super().__init__(name=name)
        self.regression_type = "OLS"
    
    def test(self, 
            data: Union[np.ndarray, pd.Series], 
            lags: Optional[int] = None,
            trend: Union[str, TrendType] = TrendType.CONSTANT,
            method: Union[str, LagSelectionMethod] = LagSelectionMethod.AIC,
            max_lags: Optional[int] = None,
            **kwargs: Any) -> UnitRootTestResult:
        """Perform the ADF test on the provided data.
        
        Args:
            data: The data to test for stationarity
            lags: Number of lags to include in the test regression
                 If None, lags will be selected using the specified method
            trend: Type of deterministic trend to include in the test regression
                  Can be a string or TrendType enum
            method: Method for selecting the optimal lag order if lags is None
                   Can be a string or LagSelectionMethod enum
            max_lags: Maximum number of lags to consider when selecting optimal lag order
                     If None, a default value based on the sample size will be used
            **kwargs: Additional keyword arguments for the test
        
        Returns:
            UnitRootTestResult: The test results
        
        Raises:
            ValueError: If the data or parameters are invalid
            TestError: If the test fails
        """
        # Validate data
        data_array = self.validate_data(data)
        nobs = len(data_array)
        
        # Convert trend to TrendType enum if it's a string
        if isinstance(trend, str):
            trend = TrendType.from_string(trend)
        
        # Convert method to LagSelectionMethod enum if it's a string
        if isinstance(method, str):
            method = LagSelectionMethod.from_string(method)
        
        # Determine maximum lags if not specified
        if max_lags is None:
            max_lags = self._get_max_lags(nobs)
        
        # Select optimal lag order if lags is None
        if lags is None:
            if method == LagSelectionMethod.FIXED:
                raise ValueError(
                    "Cannot use method='fixed' when lags=None. "
                    "Please specify a lag order explicitly."
                )
            
            try:
                lags = self._select_optimal_lags(data_array, max_lags, trend, method)
                logger.info(f"Selected optimal lag order: {lags}")
            except Exception as e:
                raise TestError(
                    f"Failed to select optimal lag order: {e}",
                    test_type="ADF",
                    parameter="lags",
                    issue="lag selection failed",
                    details=str(e)
                )
        else:
            # If lags is specified, use FIXED method
            method = LagSelectionMethod.FIXED
        
        # Validate lag order
        if lags < 0:
            raise ValueError(f"Lag order must be non-negative, got {lags}")
        
        if lags >= nobs - 1:
            raise ValueError(
                f"Lag order ({lags}) must be less than the number of observations "
                f"minus 1 ({nobs - 1})"
            )
        
        # Prepare data for ADF regression
        y = np.diff(data_array)  # First differences
        y_lagged = data_array[:-1]  # Lagged levels
        nobs_eff = len(y)  # Effective number of observations
        
        # Create lagged differences
        if lags > 0:
            dy_lags = np.zeros((nobs_eff, lags))
            for i in range(lags):
                dy_lags[i:, i] = np.diff(data_array)[:(nobs_eff - i)]
        else:
            dy_lags = None
        
        # Create design matrix based on trend specification
        if trend == TrendType.NONE:
            X = np.column_stack([y_lagged[:(nobs_eff)]])
            if lags > 0:
                X = np.column_stack([X, dy_lags])
        elif trend == TrendType.CONSTANT:
            X = np.column_stack([np.ones(nobs_eff), y_lagged[:(nobs_eff)]])
            if lags > 0:
                X = np.column_stack([X, dy_lags])
        elif trend == TrendType.TREND:
            X = np.column_stack([
                np.ones(nobs_eff), 
                np.arange(1, nobs_eff + 1), 
                y_lagged[:(nobs_eff)]
            ])
            if lags > 0:
                X = np.column_stack([X, dy_lags])
        
        # Fit ADF regression
        try:
            model = sm.OLS(y, X)
            results = model.fit()
            
            # Extract test statistic (t-statistic on lagged level)
            if trend == TrendType.NONE:
                test_stat = results.tvalues[0]
            else:
                # For constant or trend, the lagged level is at index 1 or 2
                idx = 1 if trend == TrendType.CONSTANT else 2
                test_stat = results.tvalues[idx]
            
            # Compute critical values and p-value
            critical_values = self._get_critical_values(nobs_eff, trend)
            pvalue = self._get_pvalue(test_stat, nobs_eff, trend)
            
            # Create result object
            result = UnitRootTestResult(
                test_name=self._name,
                test_statistic=test_stat,
                pvalue=pvalue,
                critical_values=critical_values,
                lags=lags,
                nobs=nobs_eff,
                trend_type=trend,
                regression_results=results,
                method=method,
                null_hypothesis="The process contains a unit root (non-stationary)",
                alternative_hypothesis="The process is stationary"
            )
            
            self._results = result
            return result
            
        except Exception as e:
            raise TestError(
                f"ADF test failed: {e}",
                test_type="ADF",
                parameter="regression",
                issue="regression failed",
                details=str(e)
            )
    
    def _get_critical_values(self, 
                           nobs: int, 
                           trend: TrendType) -> Dict[str, float]:
        """Compute critical values for the ADF test.
        
        Args:
            nobs: Number of observations
            trend: Type of deterministic trend included in the test
        
        Returns:
            Dict[str, float]: Dictionary of critical values at different significance levels
        """
        # Critical values from MacKinnon (2010)
        # These are the asymptotic critical values for different trend specifications
        if trend == TrendType.NONE:
            # No constant, no trend
            tau_star = np.array([-2.56574, -1.94100, -1.61682])
            tau_coef = np.array([
                [-22.5, -50.7, -35.0],
                [0.0, 2.79, 0.0],
                [-0.2, 0.0, -0.2]
            ])
        elif trend == TrendType.CONSTANT:
            # Constant, no trend
            tau_star = np.array([-3.43035, -2.86154, -2.56677])
            tau_coef = np.array([
                [-27.2, -18.9, -13.8],
                [0.0, 0.0, 0.0],
                [-0.1, -0.1, -0.1]
            ])
        elif trend == TrendType.TREND:
            # Constant and trend
            tau_star = np.array([-3.95877, -3.41049, -3.12705])
            tau_coef = np.array([
                [-28.4, -21.8, -17.9],
                [0.0, 0.0, 0.0],
                [-0.1, -0.1, -0.1]
            ])
        
        # Compute finite-sample critical values
        cv = {}
        for i, p in enumerate([0.01, 0.05, 0.10]):
            cv_val = tau_star[i]
            for j in range(3):
                cv_val += tau_coef[j, i] / nobs ** (j + 1)
            cv[f"{int(p * 100)}%"] = cv_val
        
        return cv
    
    def _get_pvalue(self, 
                  stat: float, 
                  nobs: int, 
                  trend: TrendType) -> float:
        """Compute p-value for the ADF test.
        
        Args:
            stat: Test statistic
            nobs: Number of observations
            trend: Type of deterministic trend included in the test
        
        Returns:
            float: p-value of the test
        """
        # Compute p-value using MacKinnon's (1994, 2010) method
        # This is an approximation based on response surface regressions
        
        if trend == TrendType.NONE:
            # No constant, no trend
            c = np.array([-1.95, -0.31, 0.0, 0.0, -0.13, -0.07])
        elif trend == TrendType.CONSTANT:
            # Constant, no trend
            c = np.array([-2.86, -0.52, 0.0, 0.0, -0.14, -0.08])
        elif trend == TrendType.TREND:
            # Constant and trend
            c = np.array([-3.41, -0.60, 0.0, 0.0, -0.15, -0.09])
        
        # Compute p-value
        if stat <= c[0]:
            # For very small test statistics, use a quadratic approximation
            p = 1.0 - np.exp(c[1] + c[2] * stat + c[3] * stat**2)
        else:
            # For larger test statistics, use a different approximation
            p = np.exp(c[4] + c[5] * stat)
        
        # Ensure p-value is in [0, 1]
        return np.clip(p, 0.0, 1.0)


class KPSS(UnitRootTest):
    """KPSS test for stationarity.
    
    The Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test is a test for stationarity
    in a time series. Unlike the ADF test, the null hypothesis is that the series
    is stationary, against the alternative that it contains a unit root.
    
    The test includes a bandwidth parameter for the Newey-West estimator of the
    long-run variance, which can be specified directly or selected automatically.
    
    Attributes:
        regression_type: Type of regression used for the test
    """
    
    def __init__(self, name: str = "KPSS Test"):
        """Initialize the KPSS test.
        
        Args:
            name: A descriptive name for the test
        """
        super().__init__(name=name)
        self.regression_type = "OLS"
    
    def test(self, 
            data: Union[np.ndarray, pd.Series], 
            lags: Optional[int] = None,
            trend: Union[str, TrendType] = TrendType.CONSTANT,
            method: Union[str, LagSelectionMethod] = LagSelectionMethod.AIC,
            max_lags: Optional[int] = None,
            **kwargs: Any) -> UnitRootTestResult:
        """Perform the KPSS test on the provided data.
        
        Args:
            data: The data to test for stationarity
            lags: Number of lags to include in the Newey-West estimator
                 If None, lags will be selected automatically
            trend: Type of deterministic trend to include in the test regression
                  Can be a string or TrendType enum
                  Note: KPSS only supports 'constant' and 'trend'
            method: Method for selecting the optimal lag order if lags is None
                   Not used in KPSS test, included for API consistency
            max_lags: Maximum number of lags to consider
                     Not used in KPSS test, included for API consistency
            **kwargs: Additional keyword arguments for the test
        
        Returns:
            UnitRootTestResult: The test results
        
        Raises:
            ValueError: If the data or parameters are invalid
            TestError: If the test fails
        """
        # Validate data
        data_array = self.validate_data(data)
        nobs = len(data_array)
        
        # Convert trend to TrendType enum if it's a string
        if isinstance(trend, str):
            trend = TrendType.from_string(trend)
        
        # KPSS test only supports 'constant' and 'trend'
        if trend == TrendType.NONE:
            raise ValueError(
                "KPSS test requires a deterministic component. "
                "Use trend='constant' or trend='trend'."
            )
        
        # Determine lags if not specified
        if lags is None:
            # Default formula: int(np.ceil(12 * (nobs / 100) ** (1/4)))
            lags = int(np.ceil(12 * (nobs / 100) ** (1/4)))
        
        # Validate lag order
        if lags < 0:
            raise ValueError(f"Lag order must be non-negative, got {lags}")
        
        # Prepare data for KPSS test
        if trend == TrendType.CONSTANT:
            # Detrend data by removing mean
            resid = data_array - np.mean(data_array)
        elif trend == TrendType.TREND:
            # Detrend data by regressing on constant and trend
            X = np.column_stack([np.ones(nobs), np.arange(1, nobs + 1)])
            beta = np.linalg.lstsq(X, data_array, rcond=None)[0]
            resid = data_array - np.dot(X, beta)
        
        # Compute partial sum process
        partial_sum = np.cumsum(resid)
        
        # Compute long-run variance using Newey-West estimator
        s2 = 0.0
        for i in range(nobs):
            s2 += resid[i] ** 2
        s2 /= nobs
        
        if lags > 0:
            for l in range(1, lags + 1):
                w = 1 - l / (lags + 1)  # Bartlett kernel
                for i in range(l, nobs):
                    s2 += 2 * w * resid[i] * resid[i - l]
            s2 /= nobs
        
        # Compute test statistic
        test_stat = np.sum(partial_sum ** 2) / (nobs ** 2 * s2)
        
        # Compute critical values and p-value
        critical_values = self._get_critical_values(trend)
        pvalue = self._get_pvalue(test_stat, trend)
        
        # Create result object
        result = UnitRootTestResult(
            test_name=self._name,
            test_statistic=test_stat,
            pvalue=pvalue,
            critical_values=critical_values,
            lags=lags,
            nobs=nobs,
            trend_type=trend,
            regression_results=None,
            method=None,
            null_hypothesis="The process is stationary",
            alternative_hypothesis="The process contains a unit root (non-stationary)"
        )
        
        self._results = result
        return result
    
    def _get_critical_values(self, trend: TrendType) -> Dict[str, float]:
        """Get critical values for the KPSS test.
        
        Args:
            trend: Type of deterministic trend included in the test
        
        Returns:
            Dict[str, float]: Dictionary of critical values at different significance levels
        """
        # Critical values from Kwiatkowski et al. (1992)
        if trend == TrendType.CONSTANT:
            # Constant, no trend (mu)
            cv = {
                "1%": 0.739,
                "2.5%": 0.574,
                "5%": 0.463,
                "10%": 0.347
            }
        elif trend == TrendType.TREND:
            # Constant and trend (tau)
            cv = {
                "1%": 0.216,
                "2.5%": 0.176,
                "5%": 0.146,
                "10%": 0.119
            }
        
        return cv
    
    def _get_pvalue(self, stat: float, trend: TrendType) -> float:
        """Compute p-value for the KPSS test.
        
        Args:
            stat: Test statistic
            trend: Type of deterministic trend included in the test
        
        Returns:
            float: p-value of the test
        """
        # Approximate p-value using interpolation of critical values
        cv = self._get_critical_values(trend)
        
        # Extract critical values and corresponding significance levels
        crit_vals = np.array([cv["10%"], cv["5%"], cv["2.5%"], cv["1%"]])
        sig_levels = np.array([0.10, 0.05, 0.025, 0.01])
        
        # If test statistic is smaller than smallest critical value, p-value > 0.10
        if stat <= crit_vals[0]:
            return 0.10
        
        # If test statistic is larger than largest critical value, p-value < 0.01
        if stat >= crit_vals[-1]:
            return 0.01
        
        # Interpolate p-value
        for i in range(len(crit_vals) - 1):
            if crit_vals[i] <= stat < crit_vals[i + 1]:
                # Linear interpolation
                p = sig_levels[i] + (sig_levels[i + 1] - sig_levels[i]) * \
                    (stat - crit_vals[i]) / (crit_vals[i + 1] - crit_vals[i])
                return p
        
        # Fallback (should not reach here)
        return 0.05


# Convenience functions for unit root testing

def adf_test(data: Union[np.ndarray, pd.Series], 
            lags: Optional[int] = None,
            trend: Union[str, TrendType] = TrendType.CONSTANT,
            method: Union[str, LagSelectionMethod] = LagSelectionMethod.AIC,
            max_lags: Optional[int] = None,
            **kwargs: Any) -> UnitRootTestResult:
    """Perform the Augmented Dickey-Fuller test for unit roots.
    
    This is a convenience function that creates an ADF test object and
    performs the test in a single call.
    
    Args:
        data: The data to test for stationarity
        lags: Number of lags to include in the test regression
             If None, lags will be selected using the specified method
        trend: Type of deterministic trend to include in the test regression
              Can be a string or TrendType enum
        method: Method for selecting the optimal lag order if lags is None
               Can be a string or LagSelectionMethod enum
        max_lags: Maximum number of lags to consider when selecting optimal lag order
                 If None, a default value based on the sample size will be used
        **kwargs: Additional keyword arguments for the test
    
    Returns:
        UnitRootTestResult: The test results
    
    Raises:
        ValueError: If the data or parameters are invalid
        TestError: If the test fails
    """
    test = ADF()
    return test.test(data, lags, trend, method, max_lags, **kwargs)

def kpss_test(data: Union[np.ndarray, pd.Series], 
             lags: Optional[int] = None,
             trend: Union[str, TrendType] = TrendType.CONSTANT,
             **kwargs: Any) -> UnitRootTestResult:
    """Perform the KPSS test for stationarity.
    
    This is a convenience function that creates a KPSS test object and
    performs the test in a single call.
    
    Args:
        data: The data to test for stationarity
        lags: Number of lags to include in the Newey-West estimator
             If None, lags will be selected automatically
        trend: Type of deterministic trend to include in the test regression
              Can be a string or TrendType enum
              Note: KPSS only supports 'constant' and 'trend'
        **kwargs: Additional keyword arguments for the test
    
    Returns:
        UnitRootTestResult: The test results
    
    Raises:
        ValueError: If the data or parameters are invalid
        TestError: If the test fails
    """
    test = KPSS()
    return test.test(data, lags, trend, **kwargs)


async def adf_test_async(data: Union[np.ndarray, pd.Series], 
                       lags: Optional[int] = None,
                       trend: Union[str, TrendType] = TrendType.CONSTANT,
                       method: Union[str, LagSelectionMethod] = LagSelectionMethod.AIC,
                       max_lags: Optional[int] = None,
                       **kwargs: Any) -> UnitRootTestResult:
    """Asynchronously perform the Augmented Dickey-Fuller test for unit roots.
    
    This is a convenience function that creates an ADF test object and
    performs the test asynchronously in a single call.
    
    Args:
        data: The data to test for stationarity
        lags: Number of lags to include in the test regression
             If None, lags will be selected using the specified method
        trend: Type of deterministic trend to include in the test regression
              Can be a string or TrendType enum
        method: Method for selecting the optimal lag order if lags is None
               Can be a string or LagSelectionMethod enum
        max_lags: Maximum number of lags to consider when selecting optimal lag order
                 If None, a default value based on the sample size will be used
        **kwargs: Additional keyword arguments for the test
    
    Returns:
        UnitRootTestResult: The test results
    
    Raises:
        ValueError: If the data or parameters are invalid
        TestError: If the test fails
    """
    test = ADF()
    return await test.test_async(data, lags, trend, method, max_lags, **kwargs)


async def kpss_test_async(data: Union[np.ndarray, pd.Series], 
                        lags: Optional[int] = None,
                        trend: Union[str, TrendType] = TrendType.CONSTANT,
                        **kwargs: Any) -> UnitRootTestResult:
    """Asynchronously perform the KPSS test for stationarity.
    
    This is a convenience function that creates a KPSS test object and
    performs the test asynchronously in a single call.
    
    Args:
        data: The data to test for stationarity
        lags: Number of lags to include in the Newey-West estimator
             If None, lags will be selected automatically
        trend: Type of deterministic trend to include in the test regression
              Can be a string or TrendType enum
              Note: KPSS only supports 'constant' and 'trend'
        **kwargs: Additional keyword arguments for the test
    
    Returns:
        UnitRootTestResult: The test results
    
    Raises:
        ValueError: If the data or parameters are invalid
        TestError: If the test fails
    """
    test = KPSS()
    return await test.test_async(data, lags, trend, **kwargs)