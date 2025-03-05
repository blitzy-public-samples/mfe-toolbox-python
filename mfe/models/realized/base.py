# mfe/models/realized/base.py
"""
Abstract base classes for realized volatility estimators.

This module defines the foundational abstract base classes that establish the contract
for all realized volatility estimator implementations in the MFE Toolbox. These classes
provide a consistent interface and shared functionality across different estimators,
ensuring uniform behavior for initialization, estimation, calibration, and result presentation.

The base classes implement the object-oriented architecture pattern with proper type hints,
parameter validation, and common preprocessing logic that would otherwise be duplicated
across individual estimator implementations.
"""

import abc
import logging
import warnings
from dataclasses import dataclass, field, asdict
from typing import (
    Any, Callable, Dict, List, Literal, Optional, Protocol, Sequence, 
    Tuple, Type, TypeVar, Union, cast, overload
)
import numpy as np
import pandas as pd
from scipy import stats

from ...core.base import ModelBase
from ...core.parameters import ParameterBase, ParameterError, validate_positive, validate_non_negative
from ...core.results import ModelResult, RealizedVolatilityResult

# Set up module-level logger
logger = logging.getLogger("mfe.models.realized.base")

# Type variables for generic classes
T = TypeVar('T')  # Generic type for parameters
R = TypeVar('R')  # Generic type for results
C = TypeVar('C')  # Generic type for configuration


@dataclass
class RealizedEstimatorConfig(ParameterBase):
    """Configuration parameters for realized volatility estimators.
    
    This class provides a standardized container for configuration parameters
    used by realized volatility estimators, with validation and serialization.
    
    Attributes:
        sampling_frequency: Sampling frequency for price data (e.g., '5min', 300)
        annualize: Whether to annualize the volatility estimate
        annualization_factor: Factor to use for annualization (e.g., 252 for daily data)
        return_type: Type of returns to compute ('log', 'simple')
        use_subsampling: Whether to use subsampling for noise reduction
        subsampling_factor: Factor for subsampling (number of subsamples)
        apply_noise_correction: Whether to apply microstructure noise correction
        kernel_type: Type of kernel for kernel-based estimators
        bandwidth: Bandwidth parameter for kernel-based estimators
        time_unit: Unit of time for high-frequency data ('seconds', 'minutes', etc.)
    """
    
    sampling_frequency: Optional[Union[str, float, int]] = None
    annualize: bool = False
    annualization_factor: float = 252.0
    return_type: Literal['log', 'simple'] = 'log'
    use_subsampling: bool = False
    subsampling_factor: int = 1
    apply_noise_correction: bool = False
    kernel_type: Optional[str] = None
    bandwidth: Optional[float] = None
    time_unit: str = 'seconds'
    
    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate configuration parameters.
        
        Raises:
            ParameterError: If any parameter constraints are violated
        """
        # Validate annualization_factor if annualize is True
        if self.annualize:
            validate_positive(self.annualization_factor, "annualization_factor")
        
        # Validate return_type
        if self.return_type not in ['log', 'simple']:
            raise ParameterError(f"return_type must be 'log' or 'simple', got {self.return_type}")
        
        # Validate subsampling_factor if use_subsampling is True
        if self.use_subsampling:
            if not isinstance(self.subsampling_factor, int):
                raise ParameterError(f"subsampling_factor must be an integer, got {type(self.subsampling_factor)}")
            validate_positive(self.subsampling_factor, "subsampling_factor")
        
        # Validate bandwidth if kernel_type is provided
        if self.kernel_type is not None and self.bandwidth is not None:
            validate_positive(self.bandwidth, "bandwidth")
        
        # Validate sampling_frequency if provided
        if self.sampling_frequency is not None:
            if isinstance(self.sampling_frequency, (int, float)):
                validate_positive(self.sampling_frequency, "sampling_frequency")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of configuration
        """
        return asdict(self)
    
    def copy(self) -> 'RealizedEstimatorConfig':
        """Create a copy of the configuration.
        
        Returns:
            RealizedEstimatorConfig: Copy of the configuration
        """
        return RealizedEstimatorConfig(**self.to_dict())


@dataclass
class RealizedEstimatorResult(RealizedVolatilityResult):
    """Result container for realized volatility estimators.
    
    This class extends RealizedVolatilityResult to provide specialized functionality
    for realized volatility estimator results, including additional metadata and
    diagnostic information specific to realized volatility estimation.
    
    Attributes:
        realized_measure: Computed realized measure (variance, bipower, kernel, etc.)
        prices: High-frequency price data used for computation
        times: Corresponding time points
        sampling_frequency: Sampling frequency used for computation
        kernel_type: Type of kernel used (for kernel-based estimators)
        bandwidth: Bandwidth parameter (for kernel-based estimators)
        subsampling: Whether subsampling was used
        noise_correction: Whether noise correction was applied
        annualization_factor: Factor used for annualization
        returns: Returns computed from prices (if available)
        noise_variance: Estimated noise variance (if noise correction was applied)
        jump_threshold: Threshold used for jump detection (if applicable)
        jump_indicators: Indicators of detected jumps (if applicable)
        computation_time: Time taken for computation (in seconds)
        config: Configuration used for estimation
    """
    
    returns: Optional[np.ndarray] = None
    noise_variance: Optional[float] = None
    jump_threshold: Optional[float] = None
    jump_indicators: Optional[np.ndarray] = None
    computation_time: Optional[float] = None
    config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self) -> None:
        """Validate result object after initialization."""
        super().__post_init__()
        
        # Ensure arrays are NumPy arrays if provided
        if self.returns is not None and not isinstance(self.returns, np.ndarray):
            self.returns = np.array(self.returns)
        
        if self.jump_indicators is not None and not isinstance(self.jump_indicators, np.ndarray):
            self.jump_indicators = np.array(self.jump_indicators)
    
    def summary(self) -> str:
        """Generate a text summary of the realized volatility results.
        
        Returns:
            str: A formatted string containing the realized volatility results summary
        """
        base_summary = super().summary()
        
        additional_info = ""
        if self.noise_variance is not None:
            additional_info += f"Estimated Noise Variance: {self.noise_variance:.6e}\n"
        
        if self.jump_threshold is not None:
            additional_info += f"Jump Detection Threshold: {self.jump_threshold:.6f}\n"
        
        if self.jump_indicators is not None:
            jump_count = np.sum(self.jump_indicators)
            additional_info += f"Detected Jumps: {jump_count} ({jump_count / len(self.jump_indicators) * 100:.2f}%)\n"
        
        if self.computation_time is not None:
            additional_info += f"Computation Time: {self.computation_time:.6f} seconds\n"
        
        if additional_info:
            additional_info = "Additional Information:\n" + additional_info + "\n"
        
        return base_summary + additional_info
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert realized measure to a pandas DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame containing realized measure and additional information
        """
        df = super().to_dataframe()
        
        # Add additional columns if available
        if self.returns is not None and len(self.returns) == len(df):
            df["returns"] = self.returns
        
        if self.jump_indicators is not None and len(self.jump_indicators) == len(df):
            df["jump_indicators"] = self.jump_indicators
        
        return df
    
    def get_annualized_measure(self) -> np.ndarray:
        """Get annualized realized measure.
        
        Returns:
            np.ndarray: Annualized realized measure
        
        Raises:
            ValueError: If annualization_factor is not available
        """
        if self.annualization_factor is None:
            raise ValueError("Annualization factor is not available")
        
        return self.realized_measure * self.annualization_factor
    
    def get_annualized_volatility(self) -> np.ndarray:
        """Get annualized realized volatility (square root of measure).
        
        Returns:
            np.ndarray: Annualized realized volatility
        
        Raises:
            ValueError: If annualization_factor is not available
        """
        return np.sqrt(self.get_annualized_measure())


class BaseRealizedEstimator(ModelBase[RealizedEstimatorConfig, RealizedEstimatorResult, Tuple[np.ndarray, np.ndarray]]):
    """Abstract base class for all realized volatility estimators.
    
    This class defines the common interface that all realized volatility estimator
    implementations must follow, establishing a consistent API across the entire
    realized volatility module.
    
    The class provides shared functionality for data validation, preprocessing,
    and result generation, while requiring subclasses to implement the specific
    estimation algorithms.
    """
    
    def __init__(self, config: Optional[RealizedEstimatorConfig] = None, name: str = "RealizedEstimator"):
        """Initialize the realized volatility estimator.
        
        Args:
            config: Configuration parameters for the estimator
            name: A descriptive name for the estimator
        """
        super().__init__(name=name)
        self._config = config if config is not None else RealizedEstimatorConfig()
        self._realized_measure: Optional[np.ndarray] = None
        self._prices: Optional[np.ndarray] = None
        self._times: Optional[np.ndarray] = None
        self._returns: Optional[np.ndarray] = None
    
    @property
    def config(self) -> RealizedEstimatorConfig:
        """Get the estimator configuration.
        
        Returns:
            RealizedEstimatorConfig: The estimator configuration
        """
        return self._config
    
    @config.setter
    def config(self, config: RealizedEstimatorConfig) -> None:
        """Set the estimator configuration.
        
        Args:
            config: New configuration parameters
        
        Raises:
            TypeError: If config is not a RealizedEstimatorConfig
        """
        if not isinstance(config, RealizedEstimatorConfig):
            raise TypeError(f"config must be a RealizedEstimatorConfig, got {type(config)}")
        
        self._config = config
        self._fitted = False  # Reset fitted state when configuration changes
    
    @property
    def realized_measure(self) -> Optional[np.ndarray]:
        """Get the realized measure from the fitted estimator.
        
        Returns:
            Optional[np.ndarray]: The realized measure if the estimator has been fitted,
                                 None otherwise
        
        Raises:
            RuntimeError: If the estimator has not been fitted
        """
        if not self._fitted:
            raise RuntimeError("Estimator has not been fitted. Call fit() first.")
        return self._realized_measure
    
    def validate_data(self, data: Tuple[np.ndarray, np.ndarray]) -> None:
        """Validate the input data for realized volatility estimation.
        
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
    
    def preprocess_data(self, prices: np.ndarray, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Preprocess the input data for realized volatility estimation.
        
        This method performs common preprocessing steps such as sampling,
        return calculation, and filtering based on the estimator configuration.
        
        Args:
            prices: High-frequency price data
            times: Corresponding time points
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Processed prices, times, and returns
        
        Raises:
            ValueError: If preprocessing fails
        """
        # Store original data
        self._prices = prices.copy()
        self._times = times.copy()
        
        # Apply sampling if sampling_frequency is provided
        if self._config.sampling_frequency is not None:
            try:
                from .price_filter import price_filter
                prices_sampled, times_sampled = price_filter(
                    prices, times, 
                    sample_freq=self._config.sampling_frequency,
                    time_unit=self._config.time_unit
                )
                
                # Check if sampling returned valid data
                if len(prices_sampled) < 10:
                    logger.warning(
                        f"Sampling with frequency {self._config.sampling_frequency} "
                        f"resulted in too few observations ({len(prices_sampled)}). "
                        f"Using original data."
                    )
                else:
                    prices = prices_sampled
                    times = times_sampled
            except Exception as e:
                logger.warning(f"Sampling failed: {str(e)}. Using original data.")
        
        # Compute returns
        if self._config.return_type == 'log':
            returns = np.diff(np.log(prices))
        else:  # 'simple'
            returns = np.diff(prices) / prices[:-1]
        
        # Adjust times to match returns (remove first observation)
        times = times[1:]
        
        # Store processed data
        self._returns = returns
        
        return prices[1:], times, returns
    
    def fit(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> RealizedEstimatorResult:
        """Fit the realized volatility estimator to the provided data.
        
        This method validates the input data, preprocesses it according to the
        estimator configuration, and then calls the _compute_realized_measure
        method to perform the actual estimation.
        
        Args:
            data: The data to fit the estimator to, as a tuple of (prices, times)
            **kwargs: Additional keyword arguments for estimation
        
        Returns:
            RealizedEstimatorResult: The estimation results
        
        Raises:
            ValueError: If the data is invalid
            RuntimeError: If the estimation fails
        """
        import time
        start_time = time.time()
        
        # Validate input data
        self.validate_data(data)
        prices, times = data
        
        # Preprocess data
        processed_prices, processed_times, returns = self.preprocess_data(prices, times)
        
        try:
            # Compute realized measure
            realized_measure = self._compute_realized_measure(
                processed_prices, processed_times, returns, **kwargs
            )
            
            # Update instance state
            self._realized_measure = realized_measure
            self._fitted = True
            
            # Create result object
            result = RealizedEstimatorResult(
                model_name=self._name,
                realized_measure=realized_measure,
                prices=prices,
                times=times,
                sampling_frequency=self._config.sampling_frequency,
                kernel_type=self._config.kernel_type,
                bandwidth=self._config.bandwidth,
                subsampling=self._config.use_subsampling,
                noise_correction=self._config.apply_noise_correction,
                annualization_factor=self._config.annualization_factor if self._config.annualize else None,
                returns=returns,
                computation_time=time.time() - start_time,
                config=self._config.to_dict()
            )
            
            # Store result
            self._results = result
            
            return result
        
        except Exception as e:
            logger.error(f"Estimation failed: {str(e)}")
            raise RuntimeError(f"Realized volatility estimation failed: {str(e)}") from e
    
    async def fit_async(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> RealizedEstimatorResult:
        """Asynchronously fit the realized volatility estimator to the provided data.
        
        This method provides an asynchronous interface to the fit method,
        allowing for non-blocking estimation in UI contexts.
        
        Args:
            data: The data to fit the estimator to, as a tuple of (prices, times)
            **kwargs: Additional keyword arguments for estimation
        
        Returns:
            RealizedEstimatorResult: The estimation results
        
        Raises:
            ValueError: If the data is invalid
            RuntimeError: If the estimation fails
        """
        # Default implementation calls the synchronous version
        # Subclasses can override with truly asynchronous implementations
        return self.fit(data, **kwargs)
    
    def simulate(self, 
                n_periods: int, 
                burn: int = 0, 
                initial_values: Optional[np.ndarray] = None,
                random_state: Optional[Union[int, np.random.Generator]] = None,
                **kwargs: Any) -> np.ndarray:
        """Simulate data from the realized volatility estimator.
        
        This method is not applicable for realized volatility estimators,
        which are non-parametric and do not have a data-generating process.
        
        Raises:
            NotImplementedError: Always, as simulation is not applicable
        """
        raise NotImplementedError(
            "Simulation is not applicable for realized volatility estimators, "
            "which are non-parametric and do not have a data-generating process."
        )
    
    @abc.abstractmethod
    def _compute_realized_measure(self, 
                                 prices: np.ndarray, 
                                 times: np.ndarray, 
                                 returns: np.ndarray,
                                 **kwargs: Any) -> np.ndarray:
        """Compute the realized measure from the preprocessed data.
        
        This method must be implemented by all subclasses to compute the
        specific realized measure based on the preprocessed data.
        
        Args:
            prices: Preprocessed price data
            times: Preprocessed time points
            returns: Returns computed from prices
            **kwargs: Additional keyword arguments for computation
        
        Returns:
            np.ndarray: Realized measure
        
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass
    
    def calibrate(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> RealizedEstimatorConfig:
        """Calibrate the estimator configuration based on the provided data.
        
        This method analyzes the input data and determines optimal configuration
        parameters for the estimator, such as bandwidth, sampling frequency, etc.
        
        Args:
            data: The data to calibrate the estimator with, as a tuple of (prices, times)
            **kwargs: Additional keyword arguments for calibration
        
        Returns:
            RealizedEstimatorConfig: Calibrated configuration
        
        Raises:
            NotImplementedError: If calibration is not implemented for this estimator
        """
        raise NotImplementedError(
            "Calibration is not implemented for this estimator. "
            "Subclasses should override this method if calibration is supported."
        )
    
    def get_volatility(self, annualize: Optional[bool] = None) -> np.ndarray:
        """Get the volatility (square root of realized measure).
        
        Args:
            annualize: Whether to annualize the volatility (overrides config)
        
        Returns:
            np.ndarray: Volatility estimate
        
        Raises:
            RuntimeError: If the estimator has not been fitted
        """
        if not self._fitted or self._realized_measure is None:
            raise RuntimeError("Estimator has not been fitted. Call fit() first.")
        
        # Determine whether to annualize
        do_annualize = self._config.annualize if annualize is None else annualize
        
        # Compute volatility
        if do_annualize:
            return np.sqrt(self._realized_measure * self._config.annualization_factor)
        else:
            return np.sqrt(self._realized_measure)
    
    def summary(self) -> str:
        """Generate a text summary of the estimator.
        
        Returns:
            str: A formatted string containing the estimator summary
        
        Raises:
            RuntimeError: If the estimator has not been fitted
        """
        if not self._fitted:
            return f"Estimator: {self._name} (not fitted)"
        
        if self._results is None:
            return f"Estimator: {self._name} (fitted, but no results available)"
        
        return self._results.summary()
    
    def __str__(self) -> str:
        """Generate a string representation of the estimator.
        
        Returns:
            str: A string representation of the estimator
        """
        return self.summary()
    
    def __repr__(self) -> str:
        """Generate a detailed string representation of the estimator.
        
        Returns:
            str: A detailed string representation of the estimator
        """
        return f"{self.__class__.__name__}(name='{self._name}', fitted={self._fitted})"


class MultivariateRealizedEstimator(BaseRealizedEstimator):
    """Abstract base class for multivariate realized volatility estimators.
    
    This class extends BaseRealizedEstimator to provide specialized functionality
    for multivariate realized volatility estimators, which compute covariance
    matrices from multiple price series.
    """
    
    def __init__(self, config: Optional[RealizedEstimatorConfig] = None, name: str = "MultivariateRealizedEstimator"):
        """Initialize the multivariate realized volatility estimator.
        
        Args:
            config: Configuration parameters for the estimator
            name: A descriptive name for the estimator
        """
        super().__init__(config=config, name=name)
        self._n_assets: Optional[int] = None
    
    @property
    def n_assets(self) -> Optional[int]:
        """Get the number of assets.
        
        Returns:
            Optional[int]: The number of assets if the estimator has been fitted,
                          None otherwise
        """
        return self._n_assets
    
    def validate_data(self, data: Tuple[np.ndarray, np.ndarray]) -> None:
        """Validate the input data for multivariate realized volatility estimation.
        
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
        
        if prices.ndim != 2:
            raise ValueError(f"Prices must be 2-dimensional (n_obs, n_assets), got {prices.ndim} dimensions")
        
        if times.ndim != 1:
            raise ValueError(f"Times must be 1-dimensional, got {times.ndim} dimensions")
        
        if len(times) != prices.shape[0]:
            raise ValueError(
                f"Times length ({len(times)}) must match prices rows ({prices.shape[0]})"
            )
        
        if prices.shape[1] < 2:
            raise ValueError(f"Prices must have at least 2 assets, got {prices.shape[1]}")
        
        if len(times) < 10:  # Arbitrary minimum length
            raise ValueError(f"Data length must be at least 10, got {len(times)}")
        
        if np.isnan(prices).any() or np.isnan(times).any():
            raise ValueError("Data contains NaN values")
        
        if np.isinf(prices).any() or np.isinf(times).any():
            raise ValueError("Data contains infinite values")
        
        # Check that times are monotonically increasing
        if not np.all(np.diff(times) >= 0):
            raise ValueError("Times must be monotonically increasing")
        
        # Store number of assets
        self._n_assets = prices.shape[1]
    
    def preprocess_data(self, prices: np.ndarray, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Preprocess the input data for multivariate realized volatility estimation.
        
        This method performs common preprocessing steps such as sampling,
        return calculation, and filtering based on the estimator configuration.
        
        Args:
            prices: High-frequency price data (n_obs, n_assets)
            times: Corresponding time points
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Processed prices, times, and returns
        
        Raises:
            ValueError: If preprocessing fails
        """
        # Store original data
        self._prices = prices.copy()
        self._times = times.copy()
        self._n_assets = prices.shape[1]
        
        # Apply sampling if sampling_frequency is provided
        if self._config.sampling_frequency is not None:
            try:
                from .price_filter import price_filter
                
                # Apply price filter to each asset
                prices_sampled_list = []
                times_sampled = None
                
                for i in range(self._n_assets):
                    asset_prices = prices[:, i]
                    asset_prices_sampled, asset_times_sampled = price_filter(
                        asset_prices, times, 
                        sample_freq=self._config.sampling_frequency,
                        time_unit=self._config.time_unit
                    )
                    
                    prices_sampled_list.append(asset_prices_sampled)
                    
                    if times_sampled is None:
                        times_sampled = asset_times_sampled
                    elif len(asset_times_sampled) < len(times_sampled):
                        times_sampled = asset_times_sampled
                
                # Align all assets to the same time points
                if times_sampled is not None and len(times_sampled) >= 10:
                    from .utils import align_time
                    
                    aligned_prices = np.zeros((len(times_sampled), self._n_assets))
                    
                    for i in range(self._n_assets):
                        asset_prices = prices[:, i]
                        aligned_asset_prices = align_time(asset_prices, times, times_sampled)
                        aligned_prices[:, i] = aligned_asset_prices
                    
                    prices = aligned_prices
                    times = times_sampled
                else:
                    logger.warning(
                        f"Sampling with frequency {self._config.sampling_frequency} "
                        f"resulted in too few observations. Using original data."
                    )
            except Exception as e:
                logger.warning(f"Sampling failed: {str(e)}. Using original data.")
        
        # Compute returns
        if self._config.return_type == 'log':
            returns = np.diff(np.log(prices), axis=0)
        else:  # 'simple'
            returns = np.diff(prices, axis=0) / prices[:-1, :]
        
        # Adjust times to match returns (remove first observation)
        times = times[1:]
        
        # Store processed data
        self._returns = returns
        
        return prices[1:], times, returns
    
    def get_correlation_matrix(self) -> np.ndarray:
        """Get the correlation matrix from the realized covariance matrix.
        
        Returns:
            np.ndarray: Correlation matrix
        
        Raises:
            RuntimeError: If the estimator has not been fitted
        """
        if not self._fitted or self._realized_measure is None:
            raise RuntimeError("Estimator has not been fitted. Call fit() first.")
        
        # Check if realized_measure is a covariance matrix
        if self._realized_measure.ndim != 2 or self._realized_measure.shape[0] != self._realized_measure.shape[1]:
            raise ValueError("Realized measure is not a covariance matrix")
        
        # Compute correlation matrix
        variances = np.diag(self._realized_measure)
        std_devs = np.sqrt(variances)
        
        # Avoid division by zero
        std_devs = np.maximum(std_devs, np.finfo(float).eps)
        
        # Compute correlation matrix
        correlation_matrix = np.zeros_like(self._realized_measure)
        
        for i in range(self._n_assets):
            for j in range(self._n_assets):
                correlation_matrix[i, j] = self._realized_measure[i, j] / (std_devs[i] * std_devs[j])
        
        # Ensure diagonal is exactly 1
        np.fill_diagonal(correlation_matrix, 1.0)
        
        return correlation_matrix


class JumpRobustEstimator(BaseRealizedEstimator):
    """Abstract base class for jump-robust realized volatility estimators.
    
    This class extends BaseRealizedEstimator to provide specialized functionality
    for jump-robust realized volatility estimators, which are designed to be
    robust to jumps in the price process.
    """
    
    def __init__(self, config: Optional[RealizedEstimatorConfig] = None, name: str = "JumpRobustEstimator"):
        """Initialize the jump-robust realized volatility estimator.
        
        Args:
            config: Configuration parameters for the estimator
            name: A descriptive name for the estimator
        """
        super().__init__(config=config, name=name)
        self._jump_threshold: Optional[float] = None
        self._jump_indicators: Optional[np.ndarray] = None
    
    @property
    def jump_threshold(self) -> Optional[float]:
        """Get the jump detection threshold.
        
        Returns:
            Optional[float]: The jump detection threshold if jumps were detected,
                            None otherwise
        """
        return self._jump_threshold
    
    @property
    def jump_indicators(self) -> Optional[np.ndarray]:
        """Get the jump indicators.
        
        Returns:
            Optional[np.ndarray]: The jump indicators if jumps were detected,
                                 None otherwise
        """
        return self._jump_indicators
    
    def detect_jumps(self, 
                    returns: np.ndarray, 
                    threshold_multiplier: float = 3.0,
                    **kwargs: Any) -> Tuple[np.ndarray, float]:
        """Detect jumps in the return series.
        
        This method implements a simple threshold-based jump detection algorithm,
        where returns exceeding a certain threshold (based on local volatility)
        are classified as jumps.
        
        Args:
            returns: Return series
            threshold_multiplier: Multiplier for the threshold (default: 3.0)
            **kwargs: Additional keyword arguments for jump detection
        
        Returns:
            Tuple[np.ndarray, float]: Jump indicators and threshold
        """
        try:
            from .utils import detect_jumps
            jump_indicators, threshold = detect_jumps(
                returns, threshold_multiplier=threshold_multiplier, **kwargs
            )
            
            self._jump_threshold = threshold
            self._jump_indicators = jump_indicators
            
            return jump_indicators, threshold
        except Exception as e:
            logger.warning(f"Jump detection failed: {str(e)}. No jumps will be removed.")
            # Return all zeros (no jumps)
            return np.zeros_like(returns, dtype=bool), 0.0
    
    def get_continuous_variation(self) -> Optional[float]:
        """Get the continuous variation (realized measure excluding jumps).
        
        Returns:
            Optional[float]: The continuous variation if jumps were detected,
                            None otherwise
        
        Raises:
            RuntimeError: If the estimator has not been fitted
        """
        if not self._fitted or self._realized_measure is None:
            raise RuntimeError("Estimator has not been fitted. Call fit() first.")
        
        if self._jump_indicators is None:
            return None
        
        # Compute continuous variation (realized measure excluding jumps)
        # This is a simplified approach and may not be accurate for all estimators
        if self._returns is not None:
            continuous_returns = self._returns.copy()
            continuous_returns[self._jump_indicators] = 0.0
            
            # Compute realized variance of continuous returns
            continuous_variation = np.sum(continuous_returns ** 2)
            
            return continuous_variation
        
        return None
    
    def get_jump_variation(self) -> Optional[float]:
        """Get the jump variation (realized measure of jumps only).
        
        Returns:
            Optional[float]: The jump variation if jumps were detected,
                            None otherwise
        
        Raises:
            RuntimeError: If the estimator has not been fitted
        """
        if not self._fitted or self._realized_measure is None:
            raise RuntimeError("Estimator has not been fitted. Call fit() first.")
        
        if self._jump_indicators is None or self._returns is None:
            return None
        
        # Compute jump variation (realized measure of jumps only)
        jump_returns = self._returns.copy()
        jump_returns[~self._jump_indicators] = 0.0
        
        # Compute realized variance of jump returns
        jump_variation = np.sum(jump_returns ** 2)
        
        return jump_variation


class NoiseRobustEstimator(BaseRealizedEstimator):
    """Abstract base class for noise-robust realized volatility estimators.
    
    This class extends BaseRealizedEstimator to provide specialized functionality
    for noise-robust realized volatility estimators, which are designed to be
    robust to microstructure noise in high-frequency data.
    """
    
    def __init__(self, config: Optional[RealizedEstimatorConfig] = None, name: str = "NoiseRobustEstimator"):
        """Initialize the noise-robust realized volatility estimator.
        
        Args:
            config: Configuration parameters for the estimator
            name: A descriptive name for the estimator
        """
        super().__init__(config=config, name=name)
        self._noise_variance: Optional[float] = None
    
    @property
    def noise_variance(self) -> Optional[float]:
        """Get the estimated noise variance.
        
        Returns:
            Optional[float]: The estimated noise variance if available,
                            None otherwise
        """
        return self._noise_variance
    
    def estimate_noise_variance(self, returns: np.ndarray, **kwargs: Any) -> float:
        """Estimate the variance of microstructure noise.
        
        This method implements a simple estimator for the variance of
        microstructure noise based on the autocovariance of returns.
        
        Args:
            returns: Return series
            **kwargs: Additional keyword arguments for noise estimation
        
        Returns:
            float: Estimated noise variance
        """
        try:
            from .noise_estimate import noise_variance
            noise_var = noise_variance(returns, **kwargs)
            
            self._noise_variance = noise_var
            
            return noise_var
        except Exception as e:
            logger.warning(f"Noise variance estimation failed: {str(e)}. Using default value.")
            # Return a small default value
            return 1e-6
    
    def get_signal_to_noise_ratio(self) -> Optional[float]:
        """Get the signal-to-noise ratio.
        
        Returns:
            Optional[float]: The signal-to-noise ratio if available,
                            None otherwise
        
        Raises:
            RuntimeError: If the estimator has not been fitted
        """
        if not self._fitted or self._realized_measure is None:
            raise RuntimeError("Estimator has not been fitted. Call fit() first.")
        
        if self._noise_variance is None or self._noise_variance <= 0:
            return None
        
        # Compute signal-to-noise ratio
        # Signal is the realized measure, noise is the noise variance
        signal_to_noise = self._realized_measure / self._noise_variance
        
        return signal_to_noise
