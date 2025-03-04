'''
Abstract base classes for the MFE Toolbox.

This module defines the core abstract base classes that establish the contract
for all model implementations in the MFE Toolbox. These classes provide the
foundation for consistent interfaces across different model types, ensuring
that all implementations follow the same patterns for initialization, fitting,
simulation, forecasting, and result presentation.
'''

import abc
from dataclasses import dataclass, field
from typing import (
    Any, Dict, Generic, List, Literal, Optional, Protocol, Sequence, 
    Tuple, Type, TypeVar, Union, cast, overload
)
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# Type variables for generic base classes
T = TypeVar('T')  # Generic type for parameters
R = TypeVar('R')  # Generic type for results
D = TypeVar('D')  # Generic type for data


@dataclass
class ModelResult:
    """Base class for all model estimation results.
    
    This class serves as the foundation for all result objects returned by model
    estimation methods. It provides a consistent interface for accessing common
    result properties and generating summary information.
    """
    
    model_name: str
    convergence: bool = True
    iterations: int = 0
    log_likelihood: Optional[float] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    
    def __post_init__(self) -> None:
        """Validate result object after initialization."""
        if not self.convergence:
            warnings.warn(
                f"Model {self.model_name} did not converge after {self.iterations} iterations.",
                UserWarning
            )
    
    def summary(self) -> str:
        """Generate a text summary of the model results.
        
        Returns:
            str: A formatted string containing the model results summary.
        """
        header = f"Model: {self.model_name}\n"
        header += "=" * (len(header) - 1) + "\n\n"
        
        convergence_info = f"Convergence: {'Yes' if self.convergence else 'No'}\n"
        convergence_info += f"Iterations: {self.iterations}\n\n"
        
        fit_stats = ""
        if self.log_likelihood is not None:
            fit_stats += f"Log-Likelihood: {self.log_likelihood:.6f}\n"
        if self.aic is not None:
            fit_stats += f"AIC: {self.aic:.6f}\n"
        if self.bic is not None:
            fit_stats += f"BIC: {self.bic:.6f}\n"
        
        return header + convergence_info + fit_stats
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result object to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the result object.
        """
        return {
            "model_name": self.model_name,
            "convergence": self.convergence,
            "iterations": self.iterations,
            "log_likelihood": self.log_likelihood,
            "aic": self.aic,
            "bic": self.bic
        }


class ModelBase(abc.ABC, Generic[T, R, D]):
    """Abstract base class for all models in the MFE Toolbox.
    
    This class defines the common interface that all model implementations
    must follow, establishing a consistent API across the entire toolbox.
    
    Type Parameters:
        T: The parameter type for this model
        R: The result type for this model
        D: The data type this model accepts
    """
    
    def __init__(self, name: str = "Model"):
        """Initialize the model with a name.
        
        Args:
            name: A descriptive name for the model
        """
        self._name = name
        self._fitted = False
        self._results: Optional[R] = None
    
    @property
    def name(self) -> str:
        """Get the model name.
        
        Returns:
            str: The model name
        """
        return self._name
    
    @property
    def fitted(self) -> bool:
        """Check if the model has been fitted.
        
        Returns:
            bool: True if the model has been fitted, False otherwise
        """
        return self._fitted
    
    @property
    def results(self) -> Optional[R]:
        """Get the model estimation results.
        
        Returns:
            Optional[R]: The model results if fitted, None otherwise
        
        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self._results
    
    @abc.abstractmethod
    def fit(self, data: D, **kwargs: Any) -> R:
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
            RuntimeError: If the model fails to converge
        """
        pass
    
    async def fit_async(self, data: D, **kwargs: Any) -> R:
        """Asynchronously fit the model to the provided data.
        
        This method provides an asynchronous interface to the fit method,
        allowing for non-blocking model estimation in UI contexts.
        
        Args:
            data: The data to fit the model to
            **kwargs: Additional keyword arguments for model fitting
        
        Returns:
            R: The model estimation results
        
        Raises:
            ValueError: If the data is invalid
            RuntimeError: If the model fails to converge
        """
        # Default implementation calls the synchronous version
        # Subclasses can override with truly asynchronous implementations
        return self.fit(data, **kwargs)
    
    @abc.abstractmethod
    def simulate(self, 
                 n_periods: int, 
                 burn: int = 0, 
                 initial_values: Optional[np.ndarray] = None,
                 random_state: Optional[Union[int, np.random.Generator]] = None,
                 **kwargs: Any) -> np.ndarray:
        """Simulate data from the model.
        
        This method must be implemented by all subclasses to generate simulated
        data based on the model parameters.
        
        Args:
            n_periods: Number of periods to simulate
            burn: Number of initial observations to discard
            initial_values: Initial values for the simulation
            random_state: Random number generator or seed
            **kwargs: Additional keyword arguments for simulation
        
        Returns:
            np.ndarray: Simulated data
        
        Raises:
            RuntimeError: If the model has not been fitted
            ValueError: If the simulation parameters are invalid
        """
        pass
    
    async def simulate_async(self, 
                            n_periods: int, 
                            burn: int = 0, 
                            initial_values: Optional[np.ndarray] = None,
                            random_state: Optional[Union[int, np.random.Generator]] = None,
                            **kwargs: Any) -> np.ndarray:
        """Asynchronously simulate data from the model.
        
        This method provides an asynchronous interface to the simulate method,
        allowing for non-blocking simulation in UI contexts.
        
        Args:
            n_periods: Number of periods to simulate
            burn: Number of initial observations to discard
            initial_values: Initial values for the simulation
            random_state: Random number generator or seed
            **kwargs: Additional keyword arguments for simulation
        
        Returns:
            np.ndarray: Simulated data
        
        Raises:
            RuntimeError: If the model has not been fitted
            ValueError: If the simulation parameters are invalid
        """
        # Default implementation calls the synchronous version
        # Subclasses can override with truly asynchronous implementations
        return self.simulate(n_periods, burn, initial_values, random_state, **kwargs)
    
    @abc.abstractmethod
    def validate_data(self, data: D) -> None:
        """Validate the input data for model fitting.
        
        This method must be implemented by all subclasses to ensure that
        the provided data meets the requirements for model estimation.
        
        Args:
            data: The data to validate
        
        Raises:
            ValueError: If the data is invalid
            TypeError: If the data has an incorrect type
        """
        pass
    
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
        
        if hasattr(self._results, "summary") and callable(getattr(self._results, "summary")):
            return cast(Any, self._results).summary()
        
        # Fallback if results don't have a summary method
        return f"Model: {self._name} (fitted)"
    
    def __str__(self) -> str:
        """Generate a string representation of the model.
        
        Returns:
            str: A string representation of the model
        """
        return self.summary()
    
    def __repr__(self) -> str:
        """Generate a detailed string representation of the model.
        
        Returns:
            str: A detailed string representation of the model
        """
        return f"{self.__class__.__name__}(name='{self._name}', fitted={self._fitted})"


class VolatilityModelBase(ModelBase[T, R, np.ndarray]):
    """Abstract base class for volatility models.
    
    This class extends the ModelBase class to provide specialized functionality
    for volatility models, including methods for computing conditional variances
    and standardized residuals.
    """
    
    def __init__(self, name: str = "VolatilityModel"):
        """Initialize the volatility model.
        
        Args:
            name: A descriptive name for the model
        """
        super().__init__(name=name)
        self._conditional_variances: Optional[np.ndarray] = None
    
    @property
    def conditional_variances(self) -> Optional[np.ndarray]:
        """Get the conditional variances from the fitted model.
        
        Returns:
            Optional[np.ndarray]: The conditional variances if the model has been fitted,
                                 None otherwise
        
        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self._conditional_variances
    
    @abc.abstractmethod
    def compute_variance(self, 
                         parameters: T, 
                         data: np.ndarray, 
                         sigma2: Optional[np.ndarray] = None,
                         backcast: Optional[float] = None) -> np.ndarray:
        """Compute conditional variances for the given parameters and data.
        
        This method must be implemented by all subclasses to compute the
        conditional variances based on the model parameters and data.
        
        Args:
            parameters: Model parameters
            data: Input data (typically residuals)
            sigma2: Pre-allocated array for conditional variances
            backcast: Value to use for initializing the variance process
        
        Returns:
            np.ndarray: Conditional variances
        """
        pass
    
    def standardized_residuals(self, data: np.ndarray) -> np.ndarray:
        """Compute standardized residuals from the fitted model.
        
        Args:
            data: The data to compute standardized residuals for
        
        Returns:
            np.ndarray: Standardized residuals
        
        Raises:
            RuntimeError: If the model has not been fitted
            ValueError: If the data length doesn't match the conditional variances
        """
        if not self._fitted or self._conditional_variances is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        if len(data) != len(self._conditional_variances):
            raise ValueError(
                f"Data length ({len(data)}) must match conditional variances "
                f"length ({len(self._conditional_variances)})"
            )
        
        return data / np.sqrt(self._conditional_variances)
    
    def validate_data(self, data: np.ndarray) -> None:
        """Validate the input data for volatility model fitting.
        
        Args:
            data: The data to validate
        
        Raises:
            ValueError: If the data is invalid
            TypeError: If the data has an incorrect type
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a NumPy array")
        
        if data.ndim != 1:
            raise ValueError(f"Data must be 1-dimensional, got {data.ndim} dimensions")
        
        if len(data) < 10:  # Arbitrary minimum length
            raise ValueError(f"Data length must be at least 10, got {len(data)}")
        
        if np.isnan(data).any():
            raise ValueError("Data contains NaN values")
        
        if np.isinf(data).any():
            raise ValueError("Data contains infinite values")


class MultivariateVolatilityModelBase(ModelBase[T, R, np.ndarray]):
    """Abstract base class for multivariate volatility models.
    
    This class extends the ModelBase class to provide specialized functionality
    for multivariate volatility models, including methods for computing
    conditional covariance matrices.
    """
    
    def __init__(self, name: str = "MultivariateVolatilityModel"):
        """Initialize the multivariate volatility model.
        
        Args:
            name: A descriptive name for the model
        """
        super().__init__(name=name)
        self._conditional_covariances: Optional[np.ndarray] = None
        self._n_assets: Optional[int] = None
    
    @property
    def conditional_covariances(self) -> Optional[np.ndarray]:
        """Get the conditional covariance matrices from the fitted model.
        
        Returns:
            Optional[np.ndarray]: The conditional covariance matrices if the model
                                 has been fitted, None otherwise
        
        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self._conditional_covariances
    
    @property
    def n_assets(self) -> Optional[int]:
        """Get the number of assets in the model.
        
        Returns:
            Optional[int]: The number of assets if the model has been fitted,
                          None otherwise
        """
        return self._n_assets
    
    @abc.abstractmethod
    def compute_covariance(self, 
                          parameters: T, 
                          data: np.ndarray,
                          sigma: Optional[np.ndarray] = None,
                          backcast: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute conditional covariance matrices for the given parameters and data.
        
        This method must be implemented by all subclasses to compute the
        conditional covariance matrices based on the model parameters and data.
        
        Args:
            parameters: Model parameters
            data: Input data (typically residuals)
            sigma: Pre-allocated array for conditional covariances
            backcast: Matrix to use for initializing the covariance process
        
        Returns:
            np.ndarray: Conditional covariance matrices (k x k x t)
        """
        pass
    
    def standardized_residuals(self, data: np.ndarray) -> np.ndarray:
        """Compute standardized residuals from the fitted model.
        
        Args:
            data: The data to compute standardized residuals for
        
        Returns:
            np.ndarray: Standardized residuals
        
        Raises:
            RuntimeError: If the model has not been fitted
            ValueError: If the data dimensions don't match the conditional covariances
        """
        if not self._fitted or self._conditional_covariances is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        if data.shape[0] != self._conditional_covariances.shape[2]:
            raise ValueError(
                f"Data length ({data.shape[0]}) must match conditional covariances "
                f"length ({self._conditional_covariances.shape[2]})"
            )
        
        if data.shape[1] != self._conditional_covariances.shape[0]:
            raise ValueError(
                f"Data width ({data.shape[1]}) must match conditional covariances "
                f"width ({self._conditional_covariances.shape[0]})"
            )
        
        std_resid = np.zeros_like(data)
        for t in range(data.shape[0]):
            # Compute Cholesky decomposition of covariance matrix
            chol = np.linalg.cholesky(self._conditional_covariances[:, :, t])
            # Standardize using the Cholesky factor
            std_resid[t, :] = np.linalg.solve(chol, data[t, :])
        
        return std_resid
    
    def validate_data(self, data: np.ndarray) -> None:
        """Validate the input data for multivariate volatility model fitting.
        
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
        
        if data.shape[0] < 10:  # Arbitrary minimum length
            raise ValueError(f"Data length must be at least 10, got {data.shape[0]}")
        
        if data.shape[1] < 2:
            raise ValueError(f"Data must have at least 2 variables, got {data.shape[1]}")
        
        if np.isnan(data).any():
            raise ValueError("Data contains NaN values")
        
        if np.isinf(data).any():
            raise ValueError("Data contains infinite values")
        
        self._n_assets = data.shape[1]


class TimeSeriesModelBase(ModelBase[T, R, Union[np.ndarray, pd.Series]]):
    """Abstract base class for time series models.
    
    This class extends the ModelBase class to provide specialized functionality
    for time series models, including methods for forecasting and computing
    residuals.
    """
    
    def __init__(self, name: str = "TimeSeriesModel"):
        """Initialize the time series model.
        
        Args:
            name: A descriptive name for the model
        """
        super().__init__(name=name)
        self._residuals: Optional[np.ndarray] = None
        self._fitted_values: Optional[np.ndarray] = None
    
    @property
    def residuals(self) -> Optional[np.ndarray]:
        """Get the residuals from the fitted model.
        
        Returns:
            Optional[np.ndarray]: The residuals if the model has been fitted,
                                 None otherwise
        
        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self._residuals
    
    @property
    def fitted_values(self) -> Optional[np.ndarray]:
        """Get the fitted values from the fitted model.
        
        Returns:
            Optional[np.ndarray]: The fitted values if the model has been fitted,
                                 None otherwise
        
        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self._fitted_values
    
    @abc.abstractmethod
    def forecast(self, 
                steps: int, 
                exog: Optional[np.ndarray] = None,
                confidence_level: float = 0.95,
                **kwargs: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate forecasts from the fitted model.
        
        This method must be implemented by all subclasses to generate forecasts
        based on the fitted model parameters.
        
        Args:
            steps: Number of steps to forecast
            exog: Exogenous variables for the forecast period
            confidence_level: Confidence level for prediction intervals
            **kwargs: Additional keyword arguments for forecasting
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Forecasts, lower bounds, and upper bounds
        
        Raises:
            RuntimeError: If the model has not been fitted
            ValueError: If the forecast parameters are invalid
        """
        pass
    
    async def forecast_async(self, 
                           steps: int, 
                           exog: Optional[np.ndarray] = None,
                           confidence_level: float = 0.95,
                           **kwargs: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Asynchronously generate forecasts from the fitted model.
        
        This method provides an asynchronous interface to the forecast method,
        allowing for non-blocking forecasting in UI contexts.
        
        Args:
            steps: Number of steps to forecast
            exog: Exogenous variables for the forecast period
            confidence_level: Confidence level for prediction intervals
            **kwargs: Additional keyword arguments for forecasting
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Forecasts, lower bounds, and upper bounds
        
        Raises:
            RuntimeError: If the model has not been fitted
            ValueError: If the forecast parameters are invalid
        """
        # Default implementation calls the synchronous version
        # Subclasses can override with truly asynchronous implementations
        return self.forecast(steps, exog, confidence_level, **kwargs)
    
    def validate_data(self, data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Validate the input data for time series model fitting.
        
        Args:
            data: The data to validate
        
        Returns:
            np.ndarray: The validated data as a NumPy array
        
        Raises:
            ValueError: If the data is invalid
            TypeError: If the data has an incorrect type
        """
        if isinstance(data, pd.Series):
            data_array = data.values
        elif isinstance(data, np.ndarray):
            data_array = data
        else:
            raise TypeError("Data must be a NumPy array or Pandas Series")
        
        if data_array.ndim != 1:
            raise ValueError(f"Data must be 1-dimensional, got {data_array.ndim} dimensions")
        
        if len(data_array) < 10:  # Arbitrary minimum length
            raise ValueError(f"Data length must be at least 10, got {len(data_array)}")
        
        if np.isnan(data_array).any():
            raise ValueError("Data contains NaN values")
        
        if np.isinf(data_array).any():
            raise ValueError("Data contains infinite values")
        
        return data_array


class BootstrapModelBase(ModelBase[T, R, np.ndarray]):
    """Abstract base class for bootstrap methods.
    
    This class extends the ModelBase class to provide specialized functionality
    for bootstrap methods, including methods for generating bootstrap samples
    and computing bootstrap statistics.
    """
    
    def __init__(self, name: str = "BootstrapModel"):
        """Initialize the bootstrap model.
        
        Args:
            name: A descriptive name for the model
        """
        super().__init__(name=name)
        self._bootstrap_samples: Optional[np.ndarray] = None
        self._bootstrap_indices: Optional[np.ndarray] = None
    
    @property
    def bootstrap_samples(self) -> Optional[np.ndarray]:
        """Get the bootstrap samples from the fitted model.
        
        Returns:
            Optional[np.ndarray]: The bootstrap samples if the model has been fitted,
                                 None otherwise
        
        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self._bootstrap_samples
    
    @property
    def bootstrap_indices(self) -> Optional[np.ndarray]:
        """Get the bootstrap indices from the fitted model.
        
        Returns:
            Optional[np.ndarray]: The bootstrap indices if the model has been fitted,
                                 None otherwise
        
        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self._bootstrap_indices
    
    @abc.abstractmethod
    def generate_indices(self, 
                        data_length: int, 
                        n_bootstraps: int,
                        random_state: Optional[Union[int, np.random.Generator]] = None) -> np.ndarray:
        """Generate bootstrap indices.
        
        This method must be implemented by all subclasses to generate bootstrap
        indices based on the bootstrap method.
        
        Args:
            data_length: Length of the original data
            n_bootstraps: Number of bootstrap samples to generate
            random_state: Random number generator or seed
        
        Returns:
            np.ndarray: Bootstrap indices with shape (n_bootstraps, data_length)
        """
        pass
    
    @abc.abstractmethod
    def compute_statistic(self, 
                         data: np.ndarray, 
                         indices: np.ndarray,
                         statistic_func: callable) -> np.ndarray:
        """Compute bootstrap statistics.
        
        This method must be implemented by all subclasses to compute bootstrap
        statistics based on the bootstrap samples.
        
        Args:
            data: Original data
            indices: Bootstrap indices
            statistic_func: Function to compute the statistic of interest
        
        Returns:
            np.ndarray: Bootstrap statistics
        """
        pass
    
    def validate_data(self, data: np.ndarray) -> None:
        """Validate the input data for bootstrap methods.
        
        Args:
            data: The data to validate
        
        Raises:
            ValueError: If the data is invalid
            TypeError: If the data has an incorrect type
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a NumPy array")
        
        if data.ndim != 1 and data.ndim != 2:
            raise ValueError(f"Data must be 1 or 2-dimensional, got {data.ndim} dimensions")
        
        if len(data) < 10:  # Arbitrary minimum length
            raise ValueError(f"Data length must be at least 10, got {len(data)}")
        
        if np.isnan(data).any():
            raise ValueError("Data contains NaN values")
        
        if np.isinf(data).any():
            raise ValueError("Data contains infinite values")


class RealizedVolatilityModelBase(ModelBase[T, R, Tuple[np.ndarray, np.ndarray]]):
    """Abstract base class for realized volatility models.
    
    This class extends the ModelBase class to provide specialized functionality
    for realized volatility models, including methods for computing realized
    measures from high-frequency data.
    """
    
    def __init__(self, name: str = "RealizedVolatilityModel"):
        """Initialize the realized volatility model.
        
        Args:
            name: A descriptive name for the model
        """
        super().__init__(name=name)
        self._realized_measure: Optional[np.ndarray] = None
    
    @property
    def realized_measure(self) -> Optional[np.ndarray]:
        """Get the realized measure from the fitted model.
        
        Returns:
            Optional[np.ndarray]: The realized measure if the model has been fitted,
                                 None otherwise
        
        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self._realized_measure
    
    @abc.abstractmethod
    def compute_measure(self, 
                       prices: np.ndarray, 
                       times: np.ndarray,
                       **kwargs: Any) -> np.ndarray:
        """Compute the realized measure from high-frequency data.
        
        This method must be implemented by all subclasses to compute the
        realized measure based on the high-frequency price and time data.
        
        Args:
            prices: High-frequency price data
            times: Corresponding time points
            **kwargs: Additional keyword arguments for computation
        
        Returns:
            np.ndarray: Realized measure
        """
        pass
    
    def validate_data(self, data: Tuple[np.ndarray, np.ndarray]) -> None:
        """Validate the input data for realized volatility model fitting.
        
        Args:
            data: The data to validate, as a tuple of (prices, times)
        
        Raises:
            ValueError: If the data is invalid
            TypeError: If the data has an incorrect type
        """
        if not isinstance(data, tuple) or len(data) != 2:
            raise TypeError("Data must be a tuple of (prices, times)")
        
        prices, times = data
        
        if not isinstance(prices, np.ndarray) or not isinstance(times, np.ndarray):
            raise TypeError("Prices and times must be NumPy arrays")
        
        if prices.ndim != 1:
            raise ValueError(f"Prices must be 1-dimensional, got {prices.ndim} dimensions")
        
        if times.ndim != 1:
            raise ValueError(f"Times must be 1-dimensional, got {times.ndim} dimensions")
        
        if len(prices) != len(times):
            raise ValueError(
                f"Prices length ({len(prices)}) must match times length ({len(times)})"
            )
        
        if len(prices) < 10:  # Arbitrary minimum length
            raise ValueError(f"Data length must be at least 10, got {len(prices)}")
        
        if np.isnan(prices).any() or np.isnan(times).any():
            raise ValueError("Data contains NaN values")
        
        if np.isinf(prices).any() or np.isinf(times).any():
            raise ValueError("Data contains infinite values")
        
        # Check that times are monotonically increasing
        if not np.all(np.diff(times) >= 0):
            raise ValueError("Times must be monotonically increasing")


class CrossSectionalModelBase(ModelBase[T, R, Tuple[np.ndarray, np.ndarray]]):
    """Abstract base class for cross-sectional models.
    
    This class extends the ModelBase class to provide specialized functionality
    for cross-sectional models, including methods for parameter estimation and
    prediction.
    """
    
    def __init__(self, name: str = "CrossSectionalModel"):
        """Initialize the cross-sectional model.
        
        Args:
            name: A descriptive name for the model
        """
        super().__init__(name=name)
        self._coefficients: Optional[np.ndarray] = None
        self._residuals: Optional[np.ndarray] = None
        self._fitted_values: Optional[np.ndarray] = None
    
    @property
    def coefficients(self) -> Optional[np.ndarray]:
        """Get the coefficients from the fitted model.
        
        Returns:
            Optional[np.ndarray]: The coefficients if the model has been fitted,
                                 None otherwise
        
        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self._coefficients
    
    @property
    def residuals(self) -> Optional[np.ndarray]:
        """Get the residuals from the fitted model.
        
        Returns:
            Optional[np.ndarray]: The residuals if the model has been fitted,
                                 None otherwise
        
        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self._residuals
    
    @property
    def fitted_values(self) -> Optional[np.ndarray]:
        """Get the fitted values from the fitted model.
        
        Returns:
            Optional[np.ndarray]: The fitted values if the model has been fitted,
                                 None otherwise
        
        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self._fitted_values
    
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
        if not isinstance(data, tuple) or len(data) != 2:
            raise TypeError("Data must be a tuple of (y, X)")
        
        y, X = data
        
        if not isinstance(y, np.ndarray) or not isinstance(X, np.ndarray):
            raise TypeError("y and X must be NumPy arrays")
        
        if y.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got {y.ndim} dimensions")
        
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got {X.ndim} dimensions")
        
        if len(y) != X.shape[0]:
            raise ValueError(
                f"y length ({len(y)}) must match X rows ({X.shape[0]})"
            )
        
        if len(y) < 10:  # Arbitrary minimum length
            raise ValueError(f"Data length must be at least 10, got {len(y)}")
        
        if np.isnan(y).any() or np.isnan(X).any():
            raise ValueError("Data contains NaN values")
        
        if np.isinf(y).any() or np.isinf(X).any():
            raise ValueError("Data contains infinite values")


class StatisticalTestBase(abc.ABC):
    """Abstract base class for statistical tests.
    
    This class defines the common interface that all statistical test
    implementations must follow, establishing a consistent API across
    the entire toolbox.
    """
    
    def __init__(self, name: str = "StatisticalTest"):
        """Initialize the statistical test.
        
        Args:
            name: A descriptive name for the test
        """
        self._name = name
        self._test_statistic: Optional[float] = None
        self._p_value: Optional[float] = None
        self._critical_values: Optional[Dict[str, float]] = None
        self._null_hypothesis: Optional[str] = None
        self._alternative_hypothesis: Optional[str] = None
    
    @property
    def name(self) -> str:
        """Get the test name.
        
        Returns:
            str: The test name
        """
        return self._name
    
    @property
    def test_statistic(self) -> Optional[float]:
        """Get the test statistic.
        
        Returns:
            Optional[float]: The test statistic if the test has been run,
                            None otherwise
        """
        return self._test_statistic
    
    @property
    def p_value(self) -> Optional[float]:
        """Get the p-value.
        
        Returns:
            Optional[float]: The p-value if the test has been run,
                            None otherwise
        """
        return self._p_value
    
    @property
    def critical_values(self) -> Optional[Dict[str, float]]:
        """Get the critical values.
        
        Returns:
            Optional[Dict[str, float]]: The critical values if the test has been run,
                                       None otherwise
        """
        return self._critical_values
    
    @property
    def null_hypothesis(self) -> Optional[str]:
        """Get the null hypothesis.
        
        Returns:
            Optional[str]: The null hypothesis if set, None otherwise
        """
        return self._null_hypothesis
    
    @property
    def alternative_hypothesis(self) -> Optional[str]:
        """Get the alternative hypothesis.
        
        Returns:
            Optional[str]: The alternative hypothesis if set, None otherwise
        """
        return self._alternative_hypothesis
    
    @abc.abstractmethod
    def run(self, data: np.ndarray, **kwargs: Any) -> Dict[str, Any]:
        """Run the statistical test on the provided data.
        
        This method must be implemented by all subclasses to perform the
        statistical test on the provided data.
        
        Args:
            data: The data to test
            **kwargs: Additional keyword arguments for the test
        
        Returns:
            Dict[str, Any]: Test results including test statistic, p-value,
                           and critical values
        
        Raises:
            ValueError: If the data is invalid
        """
        pass
    
    def summary(self) -> str:
        """Generate a text summary of the test results.
        
        Returns:
            str: A formatted string containing the test results summary
        """
        if self._test_statistic is None or self._p_value is None:
            return f"Test: {self._name} (not run)"
        
        header = f"Test: {self._name}\n"
        header += "=" * (len(header) - 1) + "\n\n"
        
        if self._null_hypothesis is not None:
            header += f"Null Hypothesis: {self._null_hypothesis}\n"
        if self._alternative_hypothesis is not None:
            header += f"Alternative Hypothesis: {self._alternative_hypothesis}\n\n"
        
        results = f"Test Statistic: {self._test_statistic:.6f}\n"
        results += f"P-value: {self._p_value:.6f}\n\n"
        
        if self._critical_values is not None:
            results += "Critical Values:\n"
            for level, value in self._critical_values.items():
                results += f"  {level}: {value:.6f}\n"
        
        conclusion = "\nConclusion: "
        if self._p_value < 0.01:
            conclusion += "Reject the null hypothesis at the 1% significance level."
        elif self._p_value < 0.05:
            conclusion += "Reject the null hypothesis at the 5% significance level."
        elif self._p_value < 0.1:
            conclusion += "Reject the null hypothesis at the 10% significance level."
        else:
            conclusion += "Fail to reject the null hypothesis."
        
        return header + results + conclusion
    
    def __str__(self) -> str:
        """Generate a string representation of the test.
        
        Returns:
            str: A string representation of the test
        """
        return self.summary()
    
    def __repr__(self) -> str:
        """Generate a detailed string representation of the test.
        
        Returns:
            str: A detailed string representation of the test
        """
        return f"{self.__class__.__name__}(name='{self._name}')"


class DistributionBase(abc.ABC):
    """Abstract base class for probability distributions.
    
    This class defines the common interface that all probability distribution
    implementations must follow, establishing a consistent API across
    the entire toolbox.
    """
    
    def __init__(self, name: str = "Distribution"):
        """Initialize the probability distribution.
        
        Args:
            name: A descriptive name for the distribution
        """
        self._name = name
    
    @property
    def name(self) -> str:
        """Get the distribution name.
        
        Returns:
            str: The distribution name
        """
        return self._name
    
    @abc.abstractmethod
    def pdf(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute the probability density function.
        
        This method must be implemented by all subclasses to compute the
        probability density function for the given values.
        
        Args:
            x: Values to compute the PDF for
            **kwargs: Additional keyword arguments for the PDF
        
        Returns:
            np.ndarray: PDF values
        """
        pass
    
    @abc.abstractmethod
    def cdf(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute the cumulative distribution function.
        
        This method must be implemented by all subclasses to compute the
        cumulative distribution function for the given values.
        
        Args:
            x: Values to compute the CDF for
            **kwargs: Additional keyword arguments for the CDF
        
        Returns:
            np.ndarray: CDF values
        """
        pass
    
    @abc.abstractmethod
    def ppf(self, q: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute the percent point function (inverse of CDF).
        
        This method must be implemented by all subclasses to compute the
        percent point function for the given probabilities.
        
        Args:
            q: Probabilities to compute the PPF for
            **kwargs: Additional keyword arguments for the PPF
        
        Returns:
            np.ndarray: PPF values
        """
        pass
    
    @abc.abstractmethod
    def rvs(self, 
           size: Union[int, Tuple[int, ...]], 
           random_state: Optional[Union[int, np.random.Generator]] = None,
           **kwargs: Any) -> np.ndarray:
        """Generate random variates from the distribution.
        
        This method must be implemented by all subclasses to generate random
        variates from the distribution.
        
        Args:
            size: Number of random variates to generate
            random_state: Random number generator or seed
            **kwargs: Additional keyword arguments for random variate generation
        
        Returns:
            np.ndarray: Random variates
        """
        pass
    
    @abc.abstractmethod
    def loglikelihood(self, x: np.ndarray, **kwargs: Any) -> float:
        """Compute the log-likelihood of the data under the distribution.
        
        This method must be implemented by all subclasses to compute the
        log-likelihood of the data under the distribution.
        
        Args:
            x: Data to compute the log-likelihood for
            **kwargs: Additional keyword arguments for the log-likelihood
        
        Returns:
            float: Log-likelihood value
        """
        pass
    
    def __str__(self) -> str:
        """Generate a string representation of the distribution.
        
        Returns:
            str: A string representation of the distribution
        """
        return f"Distribution: {self._name}"
    
    def __repr__(self) -> str:
        """Generate a detailed string representation of the distribution.
        
        Returns:
            str: A detailed string representation of the distribution
        """
        return f"{self.__class__.__name__}(name='{self._name}')"