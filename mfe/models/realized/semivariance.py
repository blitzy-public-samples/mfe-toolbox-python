'''
Realized semivariance estimator for directional volatility measurement.

This module implements realized semivariance measures that separately estimate
upside and downside volatility components. These directional volatility measures
are important for asymmetric risk assessment, allowing analysts to distinguish
between positive and negative return contributions to overall volatility.

The implementation follows the object-oriented architecture of the MFE Toolbox,
leveraging NumPy's efficient array operations with Numba acceleration for performance-critical
calculations. The module supports both standard and threshold-based semivariance
variants, with comprehensive visualization capabilities for comparing upside and
downside volatility components.

References:
    Barndorff-Nielsen, O. E., Kinnebrock, S., & Shephard, N. (2010).
    Measuring downside risk-realised semivariance. In Volatility and Time
    Series Econometrics: Essays in Honor of Robert F. Engle.
''' 

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from ...core.exceptions import ParameterError, validate_positive
from ...core.parameters import ParameterBase
from .base import BaseRealizedEstimator, RealizedEstimatorConfig, RealizedEstimatorResult
from ._numba_core import _realized_semivariance_core
from .utils import compute_returns, detect_jumps

# Set up module-level logger
logger = logging.getLogger("mfe.models.realized.semivariance")

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
    logger.debug("Numba available for semivariance acceleration")
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    HAS_NUMBA = False
    logger.info("Numba not available. Semivariance will use pure NumPy implementation.")


@dataclass
class SemivarianceConfig(RealizedEstimatorConfig):
    """Configuration parameters for realized semivariance estimator.
    
    This class extends RealizedEstimatorConfig with parameters specific to
    realized semivariance estimation, including threshold options for
    jump detection and directional classification.
    
    Attributes:
        threshold_type: Method for determining the threshold ('zero', 'mean', 'median', 'custom')
        custom_threshold: Custom threshold value when threshold_type is 'custom'
        jump_robust: Whether to apply jump detection before computing semivariance
        jump_threshold_multiplier: Multiplier for jump detection threshold
        separate_jumps: Whether to compute separate semivariance for jumps
    """
    
    threshold_type: Literal['zero', 'mean', 'median', 'custom'] = 'zero'
    custom_threshold: Optional[float] = None
    jump_robust: bool = False
    jump_threshold_multiplier: float = 3.0
    separate_jumps: bool = False
    
    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        super().__post_init__()
        
        # Validate threshold_type
        valid_threshold_types = ['zero', 'mean', 'median', 'custom']
        if self.threshold_type not in valid_threshold_types:
            raise ParameterError(
                f"threshold_type must be one of {valid_threshold_types}, "
                f"got {self.threshold_type}"
            )
        
        # Validate custom_threshold if threshold_type is 'custom'
        if self.threshold_type == 'custom' and self.custom_threshold is None:
            raise ParameterError(
                "custom_threshold must be provided when threshold_type is 'custom'"
            )
        
        # Validate jump_threshold_multiplier if jump_robust is True
        if self.jump_robust:
            validate_positive(self.jump_threshold_multiplier, "jump_threshold_multiplier")


@dataclass
class SemivarianceResult(RealizedEstimatorResult):
    """Result container for realized semivariance estimator.
    
    This class extends RealizedEstimatorResult to provide specialized functionality
    for realized semivariance results, including separate storage for positive
    and negative semivariance components.
    
    Attributes:
        positive_semivariance: Realized semivariance for positive returns
        negative_semivariance: Realized semivariance for negative returns
        threshold: Threshold used for directional classification
        jump_positive_semivariance: Realized semivariance for positive jumps (if separate_jumps=True)
        jump_negative_semivariance: Realized semivariance for negative jumps (if separate_jumps=True)
        continuous_positive_semivariance: Realized semivariance for positive continuous returns (if separate_jumps=True)
        continuous_negative_semivariance: Realized semivariance for negative continuous returns (if separate_jumps=True)
    """
    
    positive_semivariance: float = 0.0
    negative_semivariance: float = 0.0
    threshold: float = 0.0
    jump_positive_semivariance: Optional[float] = None
    jump_negative_semivariance: Optional[float] = None
    continuous_positive_semivariance: Optional[float] = None
    continuous_negative_semivariance: Optional[float] = None
    
    def __post_init__(self) -> None:
        """Validate result object after initialization."""
        super().__post_init__()
        
        # Set realized_measure to the sum of positive and negative semivariance if not provided
        if self.realized_measure is None:
            self.realized_measure = self.positive_semivariance + self.negative_semivariance
    
    def summary(self) -> str:
        """Generate a text summary of the realized semivariance results.
        
        Returns:
            str: A formatted string containing the realized semivariance results summary
        """
        base_summary = super().summary()
        
        semivariance_info = (
            f"Realized Semivariance Results:\n"
            f"  Positive Semivariance: {self.positive_semivariance:.6e}\n"
            f"  Negative Semivariance: {self.negative_semivariance:.6e}\n"
            f"  Threshold: {self.threshold:.6f}\n"
            f"  Positive/Negative Ratio: {self.positive_semivariance / max(self.negative_semivariance, 1e-10):.4f}\n"
        )
        
        if self.jump_positive_semivariance is not None and self.jump_negative_semivariance is not None:
            semivariance_info += (
                f"\nJump Components:\n"
                f"  Jump Positive Semivariance: {self.jump_positive_semivariance:.6e}\n"
                f"  Jump Negative Semivariance: {self.jump_negative_semivariance:.6e}\n"
            )
        
        if self.continuous_positive_semivariance is not None and self.continuous_negative_semivariance is not None:
            semivariance_info += (
                f"\nContinuous Components:\n"
                f"  Continuous Positive Semivariance: {self.continuous_positive_semivariance:.6e}\n"
                f"  Continuous Negative Semivariance: {self.continuous_negative_semivariance:.6e}\n"
            )
        
        return base_summary + "\n" + semivariance_info
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert realized semivariance results to a pandas DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame containing realized semivariance results
        """
        df = super().to_dataframe()
        
        # Add semivariance components
        df["positive_semivariance"] = self.positive_semivariance
        df["negative_semivariance"] = self.negative_semivariance
        df["threshold"] = self.threshold
        
        # Add jump components if available
        if self.jump_positive_semivariance is not None:
            df["jump_positive_semivariance"] = self.jump_positive_semivariance
        
        if self.jump_negative_semivariance is not None:
            df["jump_negative_semivariance"] = self.jump_negative_semivariance
        
        # Add continuous components if available
        if self.continuous_positive_semivariance is not None:
            df["continuous_positive_semivariance"] = self.continuous_positive_semivariance
        
        if self.continuous_negative_semivariance is not None:
            df["continuous_negative_semivariance"] = self.continuous_negative_semivariance
        
        return df
    
    def plot_components(self, 
                       figsize: Tuple[float, float] = (10, 6),
                       title: str = "Realized Semivariance Components",
                       show_jumps: bool = True) -> plt.Figure:
        """Plot realized semivariance components.
        
        Args:
            figsize: Figure size (width, height) in inches
            title: Plot title
            show_jumps: Whether to show jump components if available
            
        Returns:
            plt.Figure: Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Basic components
        components = ["Positive", "Negative", "Total"]
        values = [self.positive_semivariance, self.negative_semivariance, self.realized_measure]
        colors = ["green", "red", "blue"]
        
        # Add jump components if available and requested
        if show_jumps and self.jump_positive_semivariance is not None and self.jump_negative_semivariance is not None:
            components.extend(["Jump Positive", "Jump Negative"])
            values.extend([self.jump_positive_semivariance, self.jump_negative_semivariance])
            colors.extend(["lightgreen", "lightcoral"])
        
        # Add continuous components if available and requested
        if show_jumps and self.continuous_positive_semivariance is not None and self.continuous_negative_semivariance is not None:
            components.extend(["Continuous Positive", "Continuous Negative"])
            values.extend([self.continuous_positive_semivariance, self.continuous_negative_semivariance])
            colors.extend(["darkgreen", "darkred"])
        
        # Create bar plot
        bars = ax.bar(components, values, color=colors, alpha=0.7)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.00001,
                    f'{height:.6f}', ha='center', va='bottom', rotation=0)
        
        # Add labels and title
        ax.set_ylabel("Realized Semivariance")
        ax.set_title(title)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def plot_returns_classification(self, 
                                  figsize: Tuple[float, float] = (12, 8),
                                  title: str = "Return Classification") -> plt.Figure:
        """Plot returns with classification (positive/negative, jumps if available).
        
        Args:
            figsize: Figure size (width, height) in inches
            title: Plot title
            
        Returns:
            plt.Figure: Matplotlib figure object
            
        Raises:
            ValueError: If returns are not available
        """
        if self.returns is None:
            raise ValueError("Returns are not available in the result object")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get returns and time points
        returns = self.returns
        times = np.arange(len(returns)) if self.times is None else self.times[1:]
        
        # Plot threshold line
        ax.axhline(y=self.threshold, color='black', linestyle='--', alpha=0.5, label=f'Threshold ({self.threshold:.6f})')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Classify returns
        positive_mask = returns > self.threshold
        negative_mask = returns <= self.threshold
        
        # Plot returns by classification
        ax.scatter(times[positive_mask], returns[positive_mask], color='green', alpha=0.6, label='Positive Returns')
        ax.scatter(times[negative_mask], returns[negative_mask], color='red', alpha=0.6, label='Negative Returns')
        
        # Add jump classification if available
        if self.jump_indicators is not None:
            jump_mask = self.jump_indicators
            ax.scatter(times[jump_mask], returns[jump_mask], color='purple', marker='x', s=100, alpha=0.8, label='Jumps')
        
        # Add labels and title
        ax.set_xlabel("Time")
        ax.set_ylabel("Returns")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig


class RealizedSemivariance(BaseRealizedEstimator):
    """Realized semivariance estimator for directional volatility measurement.
    
    This class implements realized semivariance measures that separately estimate
    upside and downside volatility components. It inherits from BaseRealizedEstimator
    and provides specialized functionality for computing and analyzing directional
    volatility components.
    
    The estimator supports various threshold types for classifying returns as
    positive or negative, as well as jump-robust estimation for separating
    continuous and jump components of volatility.
    """
    
    def __init__(self, config: Optional[SemivarianceConfig] = None, name: str = "RealizedSemivariance"):
        """Initialize the realized semivariance estimator.
        
        Args:
            config: Configuration parameters for the estimator
            name: A descriptive name for the estimator
        """
        super().__init__(config=config if config is not None else SemivarianceConfig(), name=name)
        self._positive_semivariance: Optional[float] = None
        self._negative_semivariance: Optional[float] = None
        self._threshold: Optional[float] = None
        self._jump_positive_semivariance: Optional[float] = None
        self._jump_negative_semivariance: Optional[float] = None
        self._continuous_positive_semivariance: Optional[float] = None
        self._continuous_negative_semivariance: Optional[float] = None
    
    @property
    def config(self) -> SemivarianceConfig:
        """Get the estimator configuration.
        
        Returns:
            SemivarianceConfig: The estimator configuration
        """
        return cast(SemivarianceConfig, self._config)
    
    @config.setter
    def config(self, config: SemivarianceConfig) -> None:
        """Set the estimator configuration.
        
        Args:
            config: New configuration parameters
            
        Raises:
            TypeError: If config is not a SemivarianceConfig
        """
        if not isinstance(config, SemivarianceConfig):
            raise TypeError(f"config must be a SemivarianceConfig, got {type(config)}")
        
        self._config = config
        self._fitted = False  # Reset fitted state when configuration changes
    
    @property
    def positive_semivariance(self) -> Optional[float]:
        """Get the positive semivariance.
        
        Returns:
            Optional[float]: The positive semivariance if the estimator has been fitted,
                           None otherwise
        """
        return self._positive_semivariance
    
    @property
    def negative_semivariance(self) -> Optional[float]:
        """Get the negative semivariance.
        
        Returns:
            Optional[float]: The negative semivariance if the estimator has been fitted,
                           None otherwise
        """
        return self._negative_semivariance
    
    @property
    def threshold(self) -> Optional[float]:
        """Get the threshold used for directional classification.
        
        Returns:
            Optional[float]: The threshold if the estimator has been fitted,
                           None otherwise
        """
        return self._threshold
    
    def _compute_threshold(self, returns: np.ndarray) -> float:
        """Compute the threshold for directional classification.
        
        Args:
            returns: Array of returns
        
        Returns:
            float: Threshold value
        """
        threshold_type = self.config.threshold_type
        
        if threshold_type == 'zero':
            return 0.0
        elif threshold_type == 'mean':
            return np.mean(returns)
        elif threshold_type == 'median':
            return np.median(returns)
        elif threshold_type == 'custom':
            if self.config.custom_threshold is None:
                raise ParameterError("custom_threshold must be provided when threshold_type is 'custom'")
            return self.config.custom_threshold
        else:
            # This should never happen due to validation in SemivarianceConfig.__post_init__
            raise ValueError(f"Unrecognized threshold_type: {threshold_type}")
    
    def _compute_realized_measure(self, 
                                 prices: np.ndarray, 
                                 times: np.ndarray, 
                                 returns: np.ndarray,
                                 **kwargs: Any) -> np.ndarray:
        """Compute the realized semivariance from the preprocessed data.
        
        Args:
            prices: Preprocessed price data
            times: Preprocessed time points
            returns: Returns computed from prices
            **kwargs: Additional keyword arguments for computation
            
        Returns:
            np.ndarray: Realized measure (sum of positive and negative semivariance)
        """
        # Compute threshold for directional classification
        threshold = self._compute_threshold(returns)
        self._threshold = threshold
        
        # Initialize jump indicators and jump returns
        jump_indicators = None
        jump_returns = None
        continuous_returns = None
        
        # Detect jumps if jump_robust is True
        if self.config.jump_robust:
            jump_indicators, jump_threshold = detect_jumps(
                returns, 
                threshold_multiplier=self.config.jump_threshold_multiplier
            )
            
            # Store jump indicators
            self._jump_indicators = jump_indicators
            
            # Separate jump and continuous returns if separate_jumps is True
            if self.config.separate_jumps:
                # Create arrays for jump and continuous returns
                jump_returns = np.zeros_like(returns)
                continuous_returns = np.zeros_like(returns)
                
                # Fill arrays
                jump_returns[jump_indicators] = returns[jump_indicators]
                continuous_returns[~jump_indicators] = returns[~jump_indicators]
        
        # Compute semivariance components
        if HAS_NUMBA:
            # Use Numba-accelerated implementation
            positive_sv, negative_sv = self._compute_semivariance_numba(returns, threshold)
        else:
            # Use pure NumPy implementation
            positive_sv, negative_sv = self._compute_semivariance_numpy(returns, threshold)
        
        # Store semivariance components
        self._positive_semivariance = positive_sv
        self._negative_semivariance = negative_sv
        
        # Compute jump and continuous components if separate_jumps is True
        if self.config.jump_robust and self.config.separate_jumps and jump_returns is not None and continuous_returns is not None:
            # Compute jump components
            if HAS_NUMBA:
                jump_positive_sv, jump_negative_sv = self._compute_semivariance_numba(jump_returns, threshold)
                continuous_positive_sv, continuous_negative_sv = self._compute_semivariance_numba(continuous_returns, threshold)
            else:
                jump_positive_sv, jump_negative_sv = self._compute_semivariance_numpy(jump_returns, threshold)
                continuous_positive_sv, continuous_negative_sv = self._compute_semivariance_numpy(continuous_returns, threshold)
            
            # Store jump and continuous components
            self._jump_positive_semivariance = jump_positive_sv
            self._jump_negative_semivariance = jump_negative_sv
            self._continuous_positive_semivariance = continuous_positive_sv
            self._continuous_negative_semivariance = continuous_negative_sv
        
        # Return total realized measure (sum of positive and negative semivariance)
        return np.array([positive_sv + negative_sv])
    
    @jit(nopython=True, cache=True)
    def _compute_semivariance_numba(self, returns: np.ndarray, threshold: float) -> Tuple[float, float]:
        """Numba-accelerated implementation of realized semivariance computation.
        
        Args:
            returns: Array of returns
            threshold: Threshold for directional classification
            
        Returns:
            Tuple[float, float]: Positive and negative semivariance
        """
        # This function is decorated with @jit in the class definition,
        # but Numba doesn't support JIT-compiling instance methods.
        # The actual implementation is in the _realized_semivariance_core function.
        # This method is kept for API consistency.
        pass
    
    def _compute_semivariance_numpy(self, returns: np.ndarray, threshold: float) -> Tuple[float, float]:
        """Pure NumPy implementation of realized semivariance computation.
        
        Args:
            returns: Array of returns
            threshold: Threshold for directional classification
            
        Returns:
            Tuple[float, float]: Positive and negative semivariance
        """
        # Compute positive and negative semivariance
        positive_mask = returns > threshold
        negative_mask = ~positive_mask
        
        positive_sv = np.sum(returns[positive_mask]**2)
        negative_sv = np.sum(returns[negative_mask]**2)
        
        return positive_sv, negative_sv
    
    def fit(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> SemivarianceResult:
        """Fit the realized semivariance estimator to the provided data.
        
        Args:
            data: The data to fit the estimator to, as a tuple of (prices, times)
            **kwargs: Additional keyword arguments for estimation
            
        Returns:
            SemivarianceResult: The estimation results
            
        Raises:
            ValueError: If the data is invalid
            RuntimeError: If the estimation fails
        """
        # Call the parent class fit method to perform common preprocessing and validation
        result = super().fit(data, **kwargs)
        
        # Create a SemivarianceResult object with additional semivariance-specific information
        semivariance_result = SemivarianceResult(
            model_name=result.model_name,
            realized_measure=result.realized_measure,
            prices=result.prices,
            times=result.times,
            sampling_frequency=result.sampling_frequency,
            kernel_type=result.kernel_type,
            bandwidth=result.bandwidth,
            subsampling=result.subsampling,
            noise_correction=result.noise_correction,
            annualization_factor=result.annualization_factor,
            returns=result.returns,
            noise_variance=result.noise_variance,
            jump_threshold=result.jump_threshold,
            jump_indicators=result.jump_indicators,
            computation_time=result.computation_time,
            config=result.config,
            positive_semivariance=self._positive_semivariance if self._positive_semivariance is not None else 0.0,
            negative_semivariance=self._negative_semivariance if self._negative_semivariance is not None else 0.0,
            threshold=self._threshold if self._threshold is not None else 0.0,
            jump_positive_semivariance=self._jump_positive_semivariance,
            jump_negative_semivariance=self._jump_negative_semivariance,
            continuous_positive_semivariance=self._continuous_positive_semivariance,
            continuous_negative_semivariance=self._continuous_negative_semivariance
        )
        
        # Store the result
        self._results = semivariance_result
        
        return semivariance_result
    
    async def fit_async(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> SemivarianceResult:
        """Asynchronously fit the realized semivariance estimator to the provided data.
        
        Args:
            data: The data to fit the estimator to, as a tuple of (prices, times)
            **kwargs: Additional keyword arguments for estimation
            
        Returns:
            SemivarianceResult: The estimation results
            
        Raises:
            ValueError: If the data is invalid
            RuntimeError: If the estimation fails
        """
        # Default implementation calls the synchronous version
        # This method is provided for API consistency with other estimators
        return self.fit(data, **kwargs)
    
    def get_asymmetry_ratio(self) -> Optional[float]:
        """Get the asymmetry ratio (positive/negative semivariance).
        
        Returns:
            Optional[float]: The asymmetry ratio if the estimator has been fitted,
                           None otherwise
                           
        Raises:
            RuntimeError: If the estimator has not been fitted
        """
        if not self._fitted:
            raise RuntimeError("Estimator has not been fitted. Call fit() first.")
        
        if self._positive_semivariance is None or self._negative_semivariance is None:
            return None
        
        # Avoid division by zero
        if self._negative_semivariance == 0:
            return float('inf')
        
        return self._positive_semivariance / self._negative_semivariance
    
    def plot_components(self, 
                       figsize: Tuple[float, float] = (10, 6),
                       title: str = "Realized Semivariance Components",
                       show_jumps: bool = True) -> plt.Figure:
        """Plot realized semivariance components.
        
        Args:
            figsize: Figure size (width, height) in inches
            title: Plot title
            show_jumps: Whether to show jump components if available
            
        Returns:
            plt.Figure: Matplotlib figure object
            
        Raises:
            RuntimeError: If the estimator has not been fitted
        """
        if not self._fitted or not isinstance(self._results, SemivarianceResult):
            raise RuntimeError("Estimator has not been fitted. Call fit() first.")
        
        return self._results.plot_components(figsize=figsize, title=title, show_jumps=show_jumps)
    
    def plot_returns_classification(self, 
                                  figsize: Tuple[float, float] = (12, 8),
                                  title: str = "Return Classification") -> plt.Figure:
        """Plot returns with classification (positive/negative, jumps if available).
        
        Args:
            figsize: Figure size (width, height) in inches
            title: Plot title
            
        Returns:
            plt.Figure: Matplotlib figure object
            
        Raises:
            RuntimeError: If the estimator has not been fitted
            ValueError: If returns are not available
        """
        if not self._fitted or not isinstance(self._results, SemivarianceResult):
            raise RuntimeError("Estimator has not been fitted. Call fit() first.")
        
        return self._results.plot_returns_classification(figsize=figsize, title=title)
    
    def summary(self) -> str:
        """Generate a text summary of the estimator.
        
        Returns:
            str: A formatted string containing the estimator summary
            
        Raises:
            RuntimeError: If the estimator has not been fitted
        """
        if not self._fitted:
            return f"Estimator: {self._name} (not fitted)"
        
        if not isinstance(self._results, SemivarianceResult):
            return f"Estimator: {self._name} (fitted, but results not available as SemivarianceResult)"
        
        return self._results.summary()
