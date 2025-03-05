# mfe/models/realized/noise_estimate.py
"""
Microstructure noise variance estimation for high-frequency financial data.

This module provides functions and classes for estimating the variance of microstructure
noise in high-frequency financial data. Accurate noise variance estimation is crucial
for calibrating noise-robust volatility estimators and understanding the quality of
high-frequency data.

The module implements multiple noise estimation methods, including:
1. Bandi-Russell debiased estimator
2. Oomen's AC(1) estimator
3. Signature plot-based estimators
4. First-order autocovariance estimators
5. Maximum likelihood estimators

All implementations leverage NumPy's efficient array operations with Numba acceleration
for performance-critical calculations. The module supports both raw NumPy arrays and
Pandas DataFrames with datetime indices for convenient time series analysis.

Classes:
    NoiseVarianceEstimator: Class for estimating microstructure noise variance

Functions:
    noise_variance: Estimate noise variance using various methods
    ac1_noise_variance: Estimate noise variance using first-order autocovariance
    bandi_russell_noise: Estimate noise variance using Bandi-Russell method
    signature_plot_noise: Estimate noise variance using signature plot method
    ml_noise_variance: Estimate noise variance using maximum likelihood
    plot_noise_signature: Visualize noise characteristics using signature plots
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Literal, Optional, Sequence, 
    Tuple, Type, TypeVar, Union, cast, overload
)

import numpy as np
import pandas as pd
from scipy import stats, optimize

from ...core.base import ModelBase
from ...core.parameters import ParameterBase, ParameterError, validate_positive, validate_non_negative
from ...core.exceptions import (
    DimensionError, NumericError, ParameterError, 
    raise_dimension_error, raise_numeric_error, warn_numeric
)
from .base import BaseRealizedEstimator, RealizedEstimatorConfig, RealizedEstimatorResult

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
logger = logging.getLogger("mfe.models.realized.noise_estimate")

# Try to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.info("Matplotlib not available. Plotting functions will be disabled.")


@dataclass
class NoiseVarianceConfig(RealizedEstimatorConfig):
    """Configuration parameters for noise variance estimation.
    
    This class extends RealizedEstimatorConfig to provide specialized configuration
    parameters for noise variance estimation methods.
    
    Attributes:
        method: Method for estimating noise variance
        max_lags: Maximum number of lags to consider for signature plot method
        min_lags: Minimum number of lags to consider for signature plot method
        bias_correction: Whether to apply bias correction
        robust: Whether to use robust estimation methods
        return_all: Whether to return all estimates or just the primary one
        plot: Whether to generate diagnostic plots
    """
    
    method: Literal['ac1', 'bandi-russell', 'signature', 'ml', 'auto'] = 'auto'
    max_lags: int = 20
    min_lags: int = 1
    bias_correction: bool = True
    robust: bool = False
    return_all: bool = False
    plot: bool = False
    
    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        super().__post_init__()
        
        # Validate method
        valid_methods = ['ac1', 'bandi-russell', 'signature', 'ml', 'auto']
        if self.method not in valid_methods:
            raise ParameterError(f"method must be one of {valid_methods}, got {self.method}")
        
        # Validate max_lags and min_lags
        validate_positive(self.max_lags, "max_lags")
        validate_positive(self.min_lags, "min_lags")
        
        if self.min_lags > self.max_lags:
            raise ParameterError(f"min_lags ({self.min_lags}) must be less than or equal to max_lags ({self.max_lags})")


@dataclass
class NoiseVarianceResult(RealizedEstimatorResult):
    """Result container for noise variance estimation.
    
    This class extends RealizedEstimatorResult to provide specialized functionality
    for noise variance estimation results, including additional metadata and
    diagnostic information specific to noise estimation.
    
    Attributes:
        noise_variance: Estimated noise variance
        method: Method used for estimation
        signature_plot: Signature plot data (if available)
        noise_to_signal: Noise-to-signal ratio
        bias_correction: Whether bias correction was applied
        all_estimates: Dictionary of all noise variance estimates (if return_all=True)
    """
    
    method: Optional[str] = None
    signature_plot: Optional[Dict[str, np.ndarray]] = None
    noise_to_signal: Optional[float] = None
    bias_correction: Optional[bool] = None
    all_estimates: Optional[Dict[str, float]] = None
    
    def __post_init__(self) -> None:
        """Validate result object after initialization."""
        super().__post_init__()
    
    def summary(self) -> str:
        """Generate a text summary of the noise variance estimation results.
        
        Returns:
            str: A formatted string containing the noise variance results summary
        """
        base_summary = super().summary()
        
        additional_info = f"Noise Variance Estimation Results:\n"
        additional_info += f"  Method: {self.method}\n"
        additional_info += f"  Estimated Noise Variance: {self.noise_variance:.6e}\n"
        
        if self.noise_to_signal is not None:
            additional_info += f"  Noise-to-Signal Ratio: {self.noise_to_signal:.6f}\n"
        
        if self.bias_correction is not None:
            additional_info += f"  Bias Correction Applied: {self.bias_correction}\n"
        
        if self.all_estimates is not None:
            additional_info += "  All Estimates:\n"
            for method, estimate in self.all_estimates.items():
                additional_info += f"    {method}: {estimate:.6e}\n"
        
        return base_summary + additional_info
    
    def plot(self, figsize: Tuple[int, int] = (10, 6)) -> Optional[Any]:
        """Plot noise variance estimation results.
        
        Args:
            figsize: Figure size (width, height) in inches
            
        Returns:
            Optional[Any]: Matplotlib figure if matplotlib is available, None otherwise
            
        Raises:
            ImportError: If matplotlib is not available
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib is required for plotting. Please install matplotlib.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot signature plot if available
        if self.signature_plot is not None:
            lags = self.signature_plot.get('lags')
            rv = self.signature_plot.get('rv')
            fit = self.signature_plot.get('fit')
            
            if lags is not None and rv is not None:
                ax.plot(lags, rv, 'o-', label='Realized Variance')
                
                if fit is not None:
                    ax.plot(lags, fit, 'r--', label='Linear Fit')
                
                ax.set_xlabel('Lag')
                ax.set_ylabel('Realized Variance')
                ax.set_title('Signature Plot for Noise Variance Estimation')
                ax.legend()
                
                # Add noise variance estimate as text
                if self.noise_variance is not None:
                    ax.text(0.05, 0.95, f'Noise Variance: {self.noise_variance:.6e}',
                            transform=ax.transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # If no signature plot, just show the returns
        else:
            if self.returns is not None:
                # Plot returns
                ax.plot(self.returns, 'b-', alpha=0.5)
                ax.set_title('Returns with Estimated Noise Variance')
                ax.set_xlabel('Observation')
                ax.set_ylabel('Return')
                
                # Add noise variance estimate as text
                if self.noise_variance is not None:
                    ax.text(0.05, 0.95, f'Noise Variance: {self.noise_variance:.6e}\nMethod: {self.method}',
                            transform=ax.transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig


class NoiseVarianceEstimator(BaseRealizedEstimator):
    """Estimator for microstructure noise variance in high-frequency financial data.
    
    This class implements various methods for estimating the variance of microstructure
    noise in high-frequency financial data, which is crucial for calibrating noise-robust
    volatility estimators and understanding the quality of high-frequency data.
    
    The class provides a unified interface to multiple noise estimation methods,
    including AC(1), Bandi-Russell, signature plot, and maximum likelihood approaches.
    
    Attributes:
        config: Configuration parameters for the estimator
        name: A descriptive name for the estimator
    """
    
    def __init__(self, config: Optional[NoiseVarianceConfig] = None, name: str = "NoiseVarianceEstimator"):
        """Initialize the noise variance estimator.
        
        Args:
            config: Configuration parameters for the estimator
            name: A descriptive name for the estimator
        """
        config_to_use = config if config is not None else NoiseVarianceConfig()
        super().__init__(config=config_to_use, name=name)
    
    @property
    def config(self) -> NoiseVarianceConfig:
        """Get the estimator configuration.
        
        Returns:
            NoiseVarianceConfig: The estimator configuration
        """
        return cast(NoiseVarianceConfig, self._config)
    
    @config.setter
    def config(self, config: NoiseVarianceConfig) -> None:
        """Set the estimator configuration.
        
        Args:
            config: New configuration parameters
            
        Raises:
            TypeError: If config is not a NoiseVarianceConfig
        """
        if not isinstance(config, NoiseVarianceConfig):
            raise TypeError(f"config must be a NoiseVarianceConfig, got {type(config)}")
        
        self._config = config
        self._fitted = False  # Reset fitted state when configuration changes
    
    def _compute_realized_measure(self, 
                                 prices: np.ndarray, 
                                 times: np.ndarray, 
                                 returns: np.ndarray,
                                 **kwargs: Any) -> np.ndarray:
        """Compute the noise variance from the preprocessed data.
        
        Args:
            prices: Preprocessed price data
            times: Preprocessed time points
            returns: Returns computed from prices
            **kwargs: Additional keyword arguments for computation
            
        Returns:
            np.ndarray: Noise variance estimate
            
        Raises:
            ValueError: If computation fails
        """
        # Estimate noise variance using the specified method
        if self.config.method == 'auto':
            # Try multiple methods and select the most reliable one
            estimates = {}
            
            try:
                estimates['ac1'] = ac1_noise_variance(returns, self.config.bias_correction)
            except Exception as e:
                logger.warning(f"AC(1) method failed: {str(e)}")
            
            try:
                estimates['bandi-russell'] = bandi_russell_noise(returns, self.config.bias_correction)
            except Exception as e:
                logger.warning(f"Bandi-Russell method failed: {str(e)}")
            
            try:
                signature_result = signature_plot_noise(
                    returns, 
                    min_lag=self.config.min_lags, 
                    max_lag=self.config.max_lags,
                    bias_correction=self.config.bias_correction,
                    return_plot_data=True
                )
                estimates['signature'] = signature_result[0]
                signature_plot_data = signature_result[1]
            except Exception as e:
                logger.warning(f"Signature plot method failed: {str(e)}")
                signature_plot_data = None
            
            try:
                estimates['ml'] = ml_noise_variance(returns, self.config.bias_correction)
            except Exception as e:
                logger.warning(f"Maximum likelihood method failed: {str(e)}")
            
            # Select the most reliable estimate
            if len(estimates) == 0:
                raise ValueError("All noise variance estimation methods failed")
            
            # Prefer methods in this order: signature, bandi-russell, ac1, ml
            method_preference = ['signature', 'bandi-russell', 'ac1', 'ml']
            selected_method = None
            
            for method in method_preference:
                if method in estimates:
                    selected_method = method
                    break
            
            if selected_method is None:
                # If none of the preferred methods worked, use the first available
                selected_method = list(estimates.keys())[0]
            
            noise_var = estimates[selected_method]
            method_used = selected_method
            
        else:
            # Use the specified method
            method_used = self.config.method
            signature_plot_data = None
            
            if method_used == 'ac1':
                noise_var = ac1_noise_variance(returns, self.config.bias_correction)
            
            elif method_used == 'bandi-russell':
                noise_var = bandi_russell_noise(returns, self.config.bias_correction)
            
            elif method_used == 'signature':
                signature_result = signature_plot_noise(
                    returns, 
                    min_lag=self.config.min_lags, 
                    max_lag=self.config.max_lags,
                    bias_correction=self.config.bias_correction,
                    return_plot_data=True
                )
                noise_var = signature_result[0]
                signature_plot_data = signature_result[1]
            
            elif method_used == 'ml':
                noise_var = ml_noise_variance(returns, self.config.bias_correction)
            
            else:
                raise ValueError(f"Unrecognized method: {method_used}")
        
        # Compute noise-to-signal ratio
        # Signal is the realized variance (sum of squared returns)
        realized_var = np.sum(returns**2)
        noise_to_signal = noise_var / (realized_var / len(returns))
        
        # Store additional results
        self._noise_variance = noise_var
        
        # Create result object with additional information
        result = NoiseVarianceResult(
            model_name=self._name,
            realized_measure=np.array([noise_var]),  # Store as array for consistency
            prices=prices,
            times=times,
            returns=returns,
            noise_variance=noise_var,
            method=method_used,
            signature_plot=signature_plot_data,
            noise_to_signal=noise_to_signal,
            bias_correction=self.config.bias_correction,
            all_estimates=estimates if self.config.return_all and self.config.method == 'auto' else None
        )
        
        # Store result
        self._results = result
        
        # Generate plot if requested
        if self.config.plot and HAS_MATPLOTLIB:
            result.plot()
        
        return np.array([noise_var])
    
    def fit(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> NoiseVarianceResult:
        """Fit the noise variance estimator to the provided data.
        
        Args:
            data: The data to fit the estimator to, as a tuple of (prices, times)
            **kwargs: Additional keyword arguments for estimation
            
        Returns:
            NoiseVarianceResult: The estimation results
            
        Raises:
            ValueError: If the data is invalid
            RuntimeError: If the estimation fails
        """
        result = super().fit(data, **kwargs)
        return cast(NoiseVarianceResult, result)
    
    async def fit_async(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> NoiseVarianceResult:
        """Asynchronously fit the noise variance estimator to the provided data.
        
        Args:
            data: The data to fit the estimator to, as a tuple of (prices, times)
            **kwargs: Additional keyword arguments for estimation
            
        Returns:
            NoiseVarianceResult: The estimation results
            
        Raises:
            ValueError: If the data is invalid
            RuntimeError: If the estimation fails
        """
        result = await super().fit_async(data, **kwargs)
        return cast(NoiseVarianceResult, result)
    
    def calibrate(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> NoiseVarianceConfig:
        """Calibrate the estimator configuration based on the provided data.
        
        This method analyzes the input data and determines optimal configuration
        parameters for the noise variance estimator.
        
        Args:
            data: The data to calibrate the estimator with, as a tuple of (prices, times)
            **kwargs: Additional keyword arguments for calibration
            
        Returns:
            NoiseVarianceConfig: Calibrated configuration
        """
        # Validate input data
        self.validate_data(data)
        prices, times = data
        
        # Preprocess data
        processed_prices, processed_times, returns = self.preprocess_data(prices, times)
        
        # Try different methods and select the most reliable one
        methods = ['ac1', 'bandi-russell', 'signature', 'ml']
        estimates = {}
        errors = {}
        
        for method in methods:
            try:
                # Create a temporary config with this method
                temp_config = NoiseVarianceConfig(method=method)
                
                # Create a temporary estimator
                temp_estimator = NoiseVarianceEstimator(config=temp_config)
                
                # Fit the estimator
                result = temp_estimator.fit(data)
                
                # Store the estimate
                estimates[method] = result.noise_variance
            except Exception as e:
                errors[method] = str(e)
        
        # Select the most reliable method
        if len(estimates) == 0:
            raise RuntimeError(f"All noise variance estimation methods failed: {errors}")
        
        # Compute the median of all estimates
        median_estimate = np.median(list(estimates.values()))
        
        # Select the method closest to the median
        best_method = min(estimates.keys(), key=lambda m: abs(estimates[m] - median_estimate))
        
        # Create calibrated configuration
        calibrated_config = NoiseVarianceConfig(
            method=best_method,
            max_lags=min(20, len(returns) // 10),  # Adaptive max_lags based on data length
            min_lags=1,
            bias_correction=True,
            robust=False,
            return_all=False,
            plot=False
        )
        
        return calibrated_config


@jit(nopython=True, cache=True)
def _ac1_noise_variance_numba(returns: np.ndarray, bias_correction: bool = True) -> float:
    """
    Numba-accelerated implementation of AC(1) noise variance estimation.
    
    Args:
        returns: Array of returns
        bias_correction: Whether to apply bias correction
        
    Returns:
        Estimated noise variance
    """
    n = len(returns)
    
    # Compute first-order autocovariance
    acov = 0.0
    for i in range(n - 1):
        acov += returns[i] * returns[i + 1]
    acov /= (n - 1)
    
    # Noise variance is -0.5 * first-order autocovariance
    noise_var = -0.5 * acov
    
    # Apply bias correction if requested
    if bias_correction:
        # Compute realized variance
        rv = 0.0
        for i in range(n):
            rv += returns[i]**2
        rv /= n
        
        # Adjust for small sample bias
        noise_var *= (1.0 + 1.0 / n)
    
    # If noise_var is negative, use alternative method
    if noise_var <= 0:
        # Use 0.5 * mean squared returns
        noise_var = 0.0
        for i in range(n):
            noise_var += returns[i]**2
        noise_var *= 0.5 / n
    
    return noise_var


def ac1_noise_variance(returns: np.ndarray, bias_correction: bool = True) -> float:
    """
    Estimate noise variance using first-order autocovariance (AC1) method.
    
    This method estimates the variance of microstructure noise based on the
    first-order autocovariance of returns. Under the assumption of i.i.d. noise,
    the noise variance is -0.5 times the first-order autocovariance.
    
    Args:
        returns: Array of returns
        bias_correction: Whether to apply bias correction
        
    Returns:
        Estimated noise variance
        
    Raises:
        ValueError: If returns is not a 1D array or contains invalid values
        
    References:
        Oomen, R. C. (2006). Properties of realized variance under alternative
        sampling schemes. Journal of Business & Economic Statistics, 24(2), 219-237.
    
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.noise_estimate import ac1_noise_variance
        >>> np.random.seed(42)
        >>> returns = np.random.normal(0, 0.01, 1000)
        >>> noise_var = ac1_noise_variance(returns)
        >>> noise_var
        0.00005...
    """
    # Convert to numpy array if not already
    returns = np.asarray(returns)
    
    # Validate inputs
    if returns.ndim != 1:
        raise ValueError("returns must be a 1D array")
    
    if np.isnan(returns).any() or np.isinf(returns).any():
        raise ValueError("returns contains NaN or infinite values")
    
    # Use Numba-accelerated implementation if available
    if HAS_NUMBA:
        return _ac1_noise_variance_numba(returns, bias_correction)
    
    # Pure NumPy implementation
    n = len(returns)
    
    # Compute first-order autocovariance
    acov = np.mean(returns[:-1] * returns[1:])
    
    # Noise variance is -0.5 * first-order autocovariance
    noise_var = -0.5 * acov
    
    # Apply bias correction if requested
    if bias_correction:
        # Compute realized variance
        rv = np.mean(returns**2)
        
        # Adjust for small sample bias
        noise_var *= (1.0 + 1.0 / n)
    
    # If noise_var is negative, use alternative method
    if noise_var <= 0:
        logger.warning("AC(1) method produced non-positive noise variance. "
                      "Falling back to 0.5 * mean squared returns.")
        noise_var = 0.5 * np.mean(returns**2)
    
    return noise_var


@jit(nopython=True, cache=True)
def _bandi_russell_noise_numba(returns: np.ndarray, bias_correction: bool = True) -> float:
    """
    Numba-accelerated implementation of Bandi-Russell noise variance estimation.
    
    Args:
        returns: Array of returns
        bias_correction: Whether to apply bias correction
        
    Returns:
        Estimated noise variance
    """
    n = len(returns)
    
    # Compute realized variance
    rv = 0.0
    for i in range(n):
        rv += returns[i]**2
    rv /= n
    
    # Compute first-order autocovariance
    acov = 0.0
    for i in range(n - 1):
        acov += returns[i] * returns[i + 1]
    acov /= (n - 1)
    
    # Compute second-order autocovariance
    acov2 = 0.0
    for i in range(n - 2):
        acov2 += returns[i] * returns[i + 2]
    acov2 /= (n - 2)
    
    # Bandi-Russell estimator
    noise_var = 0.5 * (rv + acov)
    
    # Apply bias correction if requested
    if bias_correction:
        # Adjust for small sample bias
        noise_var -= 0.25 * acov2
    
    # Ensure non-negativity
    if noise_var <= 0:
        # Use 0.5 * mean squared returns
        noise_var = 0.5 * rv
    
    return noise_var


def bandi_russell_noise(returns: np.ndarray, bias_correction: bool = True) -> float:
    """
    Estimate noise variance using Bandi-Russell method.
    
    This method estimates the variance of microstructure noise based on the
    realized variance and autocovariances of returns. It provides a more robust
    estimate than the simple AC(1) method, especially for data with serial
    correlation in the efficient price process.
    
    Args:
        returns: Array of returns
        bias_correction: Whether to apply bias correction
        
    Returns:
        Estimated noise variance
        
    Raises:
        ValueError: If returns is not a 1D array or contains invalid values
        
    References:
        Bandi, F. M., & Russell, J. R. (2008). Microstructure noise, realized
        variance, and optimal sampling. The Review of Economic Studies, 75(2), 339-369.
    
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.noise_estimate import bandi_russell_noise
        >>> np.random.seed(42)
        >>> returns = np.random.normal(0, 0.01, 1000)
        >>> noise_var = bandi_russell_noise(returns)
        >>> noise_var
        0.00005...
    """
    # Convert to numpy array if not already
    returns = np.asarray(returns)
    
    # Validate inputs
    if returns.ndim != 1:
        raise ValueError("returns must be a 1D array")
    
    if np.isnan(returns).any() or np.isinf(returns).any():
        raise ValueError("returns contains NaN or infinite values")
    
    # Use Numba-accelerated implementation if available
    if HAS_NUMBA:
        return _bandi_russell_noise_numba(returns, bias_correction)
    
    # Pure NumPy implementation
    n = len(returns)
    
    # Compute realized variance
    rv = np.mean(returns**2)
    
    # Compute first-order autocovariance
    acov = np.mean(returns[:-1] * returns[1:])
    
    # Compute second-order autocovariance
    acov2 = np.mean(returns[:-2] * returns[2:])
    
    # Bandi-Russell estimator
    noise_var = 0.5 * (rv + acov)
    
    # Apply bias correction if requested
    if bias_correction:
        # Adjust for small sample bias
        noise_var -= 0.25 * acov2
    
    # Ensure non-negativity
    if noise_var <= 0:
        logger.warning("Bandi-Russell method produced non-positive noise variance. "
                      "Falling back to 0.5 * mean squared returns.")
        noise_var = 0.5 * rv
    
    return noise_var


def signature_plot_noise(returns: np.ndarray, min_lag: int = 1, max_lag: int = 20,
                        bias_correction: bool = True, return_plot_data: bool = False) -> Union[float, Tuple[float, Dict[str, np.ndarray]]]:
    """
    Estimate noise variance using signature plot method.
    
    This method estimates the variance of microstructure noise by analyzing the
    behavior of realized variance at different sampling frequencies (lags).
    Under the presence of microstructure noise, the realized variance increases
    as the sampling frequency increases, and the slope of this relationship
    is related to the noise variance.
    
    Args:
        returns: Array of returns
        min_lag: Minimum lag to consider
        max_lag: Maximum lag to consider
        bias_correction: Whether to apply bias correction
        return_plot_data: Whether to return data for plotting
        
    Returns:
        If return_plot_data is False:
            Estimated noise variance
        If return_plot_data is True:
            Tuple of (estimated noise variance, plot data dictionary)
            
    Raises:
        ValueError: If returns is not a 1D array or contains invalid values
        
    References:
        Andersen, T. G., Bollerslev, T., Diebold, F. X., & Labys, P. (2000).
        Great realizations. Risk, 13(3), 105-108.
    
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.noise_estimate import signature_plot_noise
        >>> np.random.seed(42)
        >>> returns = np.random.normal(0, 0.01, 1000)
        >>> noise_var = signature_plot_noise(returns)
        >>> noise_var
        0.00005...
    """
    # Convert to numpy array if not already
    returns = np.asarray(returns)
    
    # Validate inputs
    if returns.ndim != 1:
        raise ValueError("returns must be a 1D array")
    
    if np.isnan(returns).any() or np.isinf(returns).any():
        raise ValueError("returns contains NaN or infinite values")
    
    if min_lag < 1:
        raise ValueError("min_lag must be at least 1")
    
    if max_lag < min_lag:
        raise ValueError("max_lag must be greater than or equal to min_lag")
    
    n = len(returns)
    max_lag = min(max_lag, n // 5)  # Ensure max_lag is not too large
    
    # Compute realized variance at different lags
    lags = np.arange(min_lag, max_lag + 1)
    rv = np.zeros(len(lags))
    
    for i, lag in enumerate(lags):
        # Skip every lag-th observation
        sampled_returns = returns[::lag]
        rv[i] = np.sum(sampled_returns**2) / len(sampled_returns)
    
    # Fit linear model: RV(h) = IV + 2*q*h
    # where h is the sampling interval, IV is integrated variance, q is noise variance
    X = np.column_stack((np.ones_like(lags), lags))
    y = rv
    
    # Compute linear regression coefficients
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    
    # Intercept is integrated variance, slope is 2*q
    integrated_var = beta[0]
    noise_var = beta[1] / 2.0
    
    # Apply bias correction if requested
    if bias_correction:
        # Adjust for small sample bias
        noise_var *= (1.0 + 1.0 / n)
    
    # Ensure non-negativity
    if noise_var <= 0:
        logger.warning("Signature plot method produced non-positive noise variance. "
                      "Falling back to 0.5 * mean squared returns.")
        noise_var = 0.5 * np.mean(returns**2)
    
    # Return results
    if return_plot_data:
        # Compute fitted values
        fitted = X @ beta
        
        # Create plot data dictionary
        plot_data = {
            'lags': lags,
            'rv': rv,
            'fit': fitted,
            'integrated_var': integrated_var,
            'noise_var': noise_var
        }
        
        return noise_var, plot_data
    else:
        return noise_var


def ml_noise_variance(returns: np.ndarray, bias_correction: bool = True) -> float:
    """
    Estimate noise variance using maximum likelihood method.
    
    This method estimates the variance of microstructure noise using a maximum
    likelihood approach, assuming that returns are the sum of an efficient price
    component and i.i.d. noise.
    
    Args:
        returns: Array of returns
        bias_correction: Whether to apply bias correction
        
    Returns:
        Estimated noise variance
        
    Raises:
        ValueError: If returns is not a 1D array or contains invalid values
        
    References:
        Ait-Sahalia, Y., Mykland, P. A., & Zhang, L. (2005). How often to sample
        a continuous-time process in the presence of market microstructure noise.
        The Review of Financial Studies, 18(2), 351-416.
    
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.noise_estimate import ml_noise_variance
        >>> np.random.seed(42)
        >>> returns = np.random.normal(0, 0.01, 1000)
        >>> noise_var = ml_noise_variance(returns)
        >>> noise_var
        0.00005...
    """
    # Convert to numpy array if not already
    returns = np.asarray(returns)
    
    # Validate inputs
    if returns.ndim != 1:
        raise ValueError("returns must be a 1D array")
    
    if np.isnan(returns).any() or np.isinf(returns).any():
        raise ValueError("returns contains NaN or infinite values")
    
    n = len(returns)
    
    # Define negative log-likelihood function
    def neg_log_likelihood(params):
        # params[0] is integrated variance, params[1] is noise variance
        iv, nv = params
        
        if iv <= 0 or nv <= 0:
            return 1e10  # Large value for invalid parameters
        
        # Compute autocovariance matrix
        acov_matrix = np.zeros((n, n))
        
        # Diagonal elements (variance)
        for i in range(n):
            acov_matrix[i, i] = iv + 2 * nv
        
        # Off-diagonal elements (first-order autocovariance)
        for i in range(n - 1):
            acov_matrix[i, i + 1] = -nv
            acov_matrix[i + 1, i] = -nv
        
        # Compute log-likelihood
        try:
            # Use Cholesky decomposition for numerical stability
            L = np.linalg.cholesky(acov_matrix)
            log_det = 2 * np.sum(np.log(np.diag(L)))
            
            # Solve linear system instead of computing inverse
            z = np.linalg.solve(L, returns)
            quad_form = np.sum(z**2)
            
            # Negative log-likelihood
            nll = 0.5 * (log_det + quad_form + n * np.log(2 * np.pi))
            
            return nll
        except np.linalg.LinAlgError:
            return 1e10  # Large value for numerical issues
    
    # Initial guess based on simple estimators
    rv = np.mean(returns**2)
    acov = np.mean(returns[:-1] * returns[1:])
    
    initial_noise_var = -0.5 * acov
    if initial_noise_var <= 0:
        initial_noise_var = 0.1 * rv
    
    initial_iv = rv - 2 * initial_noise_var
    if initial_iv <= 0:
        initial_iv = 0.5 * rv
    
    initial_params = [initial_iv, initial_noise_var]
    
    # Bounds for parameters (both must be positive)
    bounds = [(1e-10, None), (1e-10, None)]
    
    # Optimize negative log-likelihood
    try:
        result = optimize.minimize(
            neg_log_likelihood,
            initial_params,
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        # Extract estimated parameters
        iv_est, noise_var = result.x
        
        # Apply bias correction if requested
        if bias_correction:
            # Adjust for small sample bias
            noise_var *= (1.0 + 1.0 / n)
        
        # Ensure non-negativity (should be guaranteed by bounds)
        if noise_var <= 0:
            logger.warning("Maximum likelihood method produced non-positive noise variance. "
                          "Falling back to 0.5 * mean squared returns.")
            noise_var = 0.5 * rv
        
        return noise_var
    
    except Exception as e:
        logger.warning(f"Maximum likelihood optimization failed: {str(e)}. "
                      "Falling back to AC(1) method.")
        return ac1_noise_variance(returns, bias_correction)


def noise_variance(returns: np.ndarray, method: str = 'auto', 
                  bias_correction: bool = True, **kwargs: Any) -> float:
    """
    Estimate noise variance using various methods.
    
    This function provides a unified interface to multiple noise estimation methods,
    including AC(1), Bandi-Russell, signature plot, and maximum likelihood approaches.
    
    Args:
        returns: Array of returns
        method: Method for estimating noise variance
                ('ac1', 'bandi-russell', 'signature', 'ml', 'auto')
        bias_correction: Whether to apply bias correction
        **kwargs: Additional keyword arguments for specific methods
        
    Returns:
        Estimated noise variance
        
    Raises:
        ValueError: If returns is not a 1D array or contains invalid values
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.noise_estimate import noise_variance
        >>> np.random.seed(42)
        >>> returns = np.random.normal(0, 0.01, 1000)
        >>> noise_var = noise_variance(returns, method='auto')
        >>> noise_var
        0.00005...
    """
    # Convert to numpy array if not already
    returns = np.asarray(returns)
    
    # Validate inputs
    if returns.ndim != 1:
        raise ValueError("returns must be a 1D array")
    
    if np.isnan(returns).any() or np.isinf(returns).any():
        raise ValueError("returns contains NaN or infinite values")
    
    # Extract method-specific parameters from kwargs
    min_lag = kwargs.get('min_lag', 1)
    max_lag = kwargs.get('max_lag', 20)
    
    # Estimate noise variance using the specified method
    if method == 'auto':
        # Try multiple methods and select the most reliable one
        estimates = {}
        
        try:
            estimates['ac1'] = ac1_noise_variance(returns, bias_correction)
        except Exception as e:
            logger.warning(f"AC(1) method failed: {str(e)}")
        
        try:
            estimates['bandi-russell'] = bandi_russell_noise(returns, bias_correction)
        except Exception as e:
            logger.warning(f"Bandi-Russell method failed: {str(e)}")
        
        try:
            estimates['signature'] = signature_plot_noise(
                returns, 
                min_lag=min_lag, 
                max_lag=max_lag,
                bias_correction=bias_correction
            )
        except Exception as e:
            logger.warning(f"Signature plot method failed: {str(e)}")
        
        try:
            estimates['ml'] = ml_noise_variance(returns, bias_correction)
        except Exception as e:
            logger.warning(f"Maximum likelihood method failed: {str(e)}")
        
        # Select the most reliable estimate
        if len(estimates) == 0:
            raise ValueError("All noise variance estimation methods failed")
        
        # Prefer methods in this order: signature, bandi-russell, ac1, ml
        method_preference = ['signature', 'bandi-russell', 'ac1', 'ml']
        selected_method = None
        
        for m in method_preference:
            if m in estimates:
                selected_method = m
                break
        
        if selected_method is None:
            # If none of the preferred methods worked, use the first available
            selected_method = list(estimates.keys())[0]
        
        return estimates[selected_method]
    
    elif method == 'ac1':
        return ac1_noise_variance(returns, bias_correction)
    
    elif method == 'bandi-russell':
        return bandi_russell_noise(returns, bias_correction)
    
    elif method == 'signature':
        return signature_plot_noise(
            returns, 
            min_lag=min_lag, 
            max_lag=max_lag,
            bias_correction=bias_correction
        )
    
    elif method == 'ml':
        return ml_noise_variance(returns, bias_correction)
    
    else:
        raise ValueError(f"Unrecognized method: {method}. "
                         f"Supported methods are 'ac1', 'bandi-russell', 'signature', 'ml', 'auto'.")


def plot_noise_signature(returns: np.ndarray, min_lag: int = 1, max_lag: int = 20,
                        figsize: Tuple[int, int] = (10, 6)) -> Optional[Any]:
    """
    Visualize noise characteristics using signature plots.
    
    This function generates a signature plot showing the behavior of realized
    variance at different sampling frequencies (lags), which is useful for
    visualizing the presence and magnitude of microstructure noise.
    
    Args:
        returns: Array of returns
        min_lag: Minimum lag to consider
        max_lag: Maximum lag to consider
        figsize: Figure size (width, height) in inches
        
    Returns:
        Optional[Any]: Matplotlib figure if matplotlib is available, None otherwise
        
    Raises:
        ImportError: If matplotlib is not available
        ValueError: If returns is not a 1D array or contains invalid values
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.noise_estimate import plot_noise_signature
        >>> np.random.seed(42)
        >>> returns = np.random.normal(0, 0.01, 1000)
        >>> fig = plot_noise_signature(returns)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib is required for plotting. Please install matplotlib.")
    
    # Estimate noise variance and get plot data
    noise_var, plot_data = signature_plot_noise(
        returns, 
        min_lag=min_lag, 
        max_lag=max_lag,
        return_plot_data=True
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot signature plot
    lags = plot_data['lags']
    rv = plot_data['rv']
    fit = plot_data['fit']
    
    ax.plot(lags, rv, 'o-', label='Realized Variance')
    ax.plot(lags, fit, 'r--', label='Linear Fit')
    
    ax.set_xlabel('Lag')
    ax.set_ylabel('Realized Variance')
    ax.set_title('Signature Plot for Noise Variance Estimation')
    ax.legend()
    
    # Add noise variance estimate as text
    ax.text(0.05, 0.95, f'Noise Variance: {noise_var:.6e}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig


# Register Numba-accelerated functions if available
def _register_numba_functions() -> None:
    """
    Register Numba JIT-compiled functions for noise variance estimation.
    
    This function is called during module initialization to register
    performance-critical functions for JIT compilation if Numba is available.
    """
    if HAS_NUMBA:
        # The functions are already decorated with @jit, so no additional
        # registration is needed here. This function is kept for consistency
        # with the module structure and potential future enhancements.
        logger.debug("Noise variance estimation Numba JIT functions registered")
    else:
        logger.info("Numba not available. Noise variance estimation will use pure NumPy implementations.")


# Initialize the module
_register_numba_functions()

# mfe/models/realized/noise_estimate.py
"""
Microstructure noise variance estimation for high-frequency financial data.

This module provides functions and classes for estimating the variance of microstructure
noise in high-frequency financial data. Accurate noise variance estimation is crucial
for calibrating noise-robust volatility estimators and understanding the quality of
high-frequency data.

The module implements multiple noise estimation methods, including:
1. Bandi-Russell debiased estimator
2. Oomen's AC(1) estimator
3. Signature plot-based estimators
4. First-order autocovariance estimators
5. Maximum likelihood estimators

All implementations leverage NumPy's efficient array operations with Numba acceleration
for performance-critical calculations. The module supports both raw NumPy arrays and
Pandas DataFrames with datetime indices for convenient time series analysis.

Classes:
    NoiseVarianceEstimator: Class for estimating microstructure noise variance

Functions:
    noise_variance: Estimate noise variance using various methods
    ac1_noise_variance: Estimate noise variance using first-order autocovariance
    bandi_russell_noise: Estimate noise variance using Bandi-Russell method
    signature_plot_noise: Estimate noise variance using signature plot method
    ml_noise_variance: Estimate noise variance using maximum likelihood
    plot_noise_signature: Visualize noise characteristics using signature plots
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Literal, Optional, Sequence, 
    Tuple, Type, TypeVar, Union, cast, overload
)

import numpy as np
import pandas as pd
from scipy import stats, optimize

from ...core.base import ModelBase
from ...core.parameters import ParameterBase, ParameterError, validate_positive, validate_non_negative
from ...core.exceptions import (
    DimensionError, NumericError, ParameterError, 
    raise_dimension_error, raise_numeric_error, warn_numeric
)
from .base import BaseRealizedEstimator, RealizedEstimatorConfig, RealizedEstimatorResult

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
logger = logging.getLogger("mfe.models.realized.noise_estimate")

# Try to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.info("Matplotlib not available. Plotting functions will be disabled.")


@dataclass
class NoiseVarianceConfig(RealizedEstimatorConfig):
    """Configuration parameters for noise variance estimation.
    
    This class extends RealizedEstimatorConfig to provide specialized configuration
    parameters for noise variance estimation methods.
    
    Attributes:
        method: Method for estimating noise variance
        max_lags: Maximum number of lags to consider for signature plot method
        min_lags: Minimum number of lags to consider for signature plot method
        bias_correction: Whether to apply bias correction
        robust: Whether to use robust estimation methods
        return_all: Whether to return all estimates or just the primary one
        plot: Whether to generate diagnostic plots
    """
    
    method: Literal['ac1', 'bandi-russell', 'signature', 'ml', 'auto'] = 'auto'
    max_lags: int = 20
    min_lags: int = 1
    bias_correction: bool = True
    robust: bool = False
    return_all: bool = False
    plot: bool = False
    
    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        super().__post_init__()
        
        # Validate method
        valid_methods = ['ac1', 'bandi-russell', 'signature', 'ml', 'auto']
        if self.method not in valid_methods:
            raise ParameterError(f"method must be one of {valid_methods}, got {self.method}")
        
        # Validate max_lags and min_lags
        validate_positive(self.max_lags, "max_lags")
        validate_positive(self.min_lags, "min_lags")
        
        if self.min_lags > self.max_lags:
            raise ParameterError(f"min_lags ({self.min_lags}) must be less than or equal to max_lags ({self.max_lags})")


@dataclass
class NoiseVarianceResult(RealizedEstimatorResult):
    """Result container for noise variance estimation.
    
    This class extends RealizedEstimatorResult to provide specialized functionality
    for noise variance estimation results, including additional metadata and
    diagnostic information specific to noise estimation.
    
    Attributes:
        noise_variance: Estimated noise variance
        method: Method used for estimation
        signature_plot: Signature plot data (if available)
        noise_to_signal: Noise-to-signal ratio
        bias_correction: Whether bias correction was applied
        all_estimates: Dictionary of all noise variance estimates (if return_all=True)
    """
    
    method: Optional[str] = None
    signature_plot: Optional[Dict[str, np.ndarray]] = None
    noise_to_signal: Optional[float] = None
    bias_correction: Optional[bool] = None
    all_estimates: Optional[Dict[str, float]] = None
    
    def __post_init__(self) -> None:
        """Validate result object after initialization."""
        super().__post_init__()
    
    def summary(self) -> str:
        """Generate a text summary of the noise variance estimation results.
        
        Returns:
            str: A formatted string containing the noise variance results summary
        """
        base_summary = super().summary()
        
        additional_info = f"Noise Variance Estimation Results:\n"
        additional_info += f"  Method: {self.method}\n"
        additional_info += f"  Estimated Noise Variance: {self.noise_variance:.6e}\n"
        
        if self.noise_to_signal is not None:
            additional_info += f"  Noise-to-Signal Ratio: {self.noise_to_signal:.6f}\n"
        
        if self.bias_correction is not None:
            additional_info += f"  Bias Correction Applied: {self.bias_correction}\n"
        
        if self.all_estimates is not None:
            additional_info += "  All Estimates:\n"
            for method, estimate in self.all_estimates.items():
                additional_info += f"    {method}: {estimate:.6e}\n"
        
        return base_summary + additional_info
    
    def plot(self, figsize: Tuple[int, int] = (10, 6)) -> Optional[Any]:
        """Plot noise variance estimation results.
        
        Args:
            figsize: Figure size (width, height) in inches
            
        Returns:
            Optional[Any]: Matplotlib figure if matplotlib is available, None otherwise
            
        Raises:
            ImportError: If matplotlib is not available
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib is required for plotting. Please install matplotlib.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot signature plot if available
        if self.signature_plot is not None:
            lags = self.signature_plot.get('lags')
            rv = self.signature_plot.get('rv')
            fit = self.signature_plot.get('fit')
            
            if lags is not None and rv is not None:
                ax.plot(lags, rv, 'o-', label='Realized Variance')
                
                if fit is not None:
                    ax.plot(lags, fit, 'r--', label='Linear Fit')
                
                ax.set_xlabel('Lag')
                ax.set_ylabel('Realized Variance')
                ax.set_title('Signature Plot for Noise Variance Estimation')
                ax.legend()
                
                # Add noise variance estimate as text
                if self.noise_variance is not None:
                    ax.text(0.05, 0.95, f'Noise Variance: {self.noise_variance:.6e}',
                            transform=ax.transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # If no signature plot, just show the returns
        else:
            if self.returns is not None:
                # Plot returns
                ax.plot(self.returns, 'b-', alpha=0.5)
                ax.set_title('Returns with Estimated Noise Variance')
                ax.set_xlabel('Observation')
                ax.set_ylabel('Return')
                
                # Add noise variance estimate as text
                if self.noise_variance is not None:
                    ax.text(0.05, 0.95, f'Noise Variance: {self.noise_variance:.6e}\nMethod: {self.method}',
                            transform=ax.transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig


class NoiseVarianceEstimator(BaseRealizedEstimator):
    """Estimator for microstructure noise variance in high-frequency financial data.
    
    This class implements various methods for estimating the variance of microstructure
    noise in high-frequency financial data, which is crucial for calibrating noise-robust
    volatility estimators and understanding the quality of high-frequency data.
    
    The class provides a unified interface to multiple noise estimation methods,
    including AC(1), Bandi-Russell, signature plot, and maximum likelihood approaches.
    
    Attributes:
        config: Configuration parameters for the estimator
        name: A descriptive name for the estimator
    """
    
    def __init__(self, config: Optional[NoiseVarianceConfig] = None, name: str = "NoiseVarianceEstimator"):
        """Initialize the noise variance estimator.
        
        Args:
            config: Configuration parameters for the estimator
            name: A descriptive name for the estimator
        """
        config_to_use = config if config is not None else NoiseVarianceConfig()
        super().__init__(config=config_to_use, name=name)
    
    @property
    def config(self) -> NoiseVarianceConfig:
        """Get the estimator configuration.
        
        Returns:
            NoiseVarianceConfig: The estimator configuration
        """
        return cast(NoiseVarianceConfig, self._config)
    
    @config.setter
    def config(self, config: NoiseVarianceConfig) -> None:
        """Set the estimator configuration.
        
        Args:
            config: New configuration parameters
            
        Raises:
            TypeError: If config is not a NoiseVarianceConfig
        """
        if not isinstance(config, NoiseVarianceConfig):
            raise TypeError(f"config must be a NoiseVarianceConfig, got {type(config)}")
        
        self._config = config
        self._fitted = False  # Reset fitted state when configuration changes
    
    def _compute_realized_measure(self, 
                                 prices: np.ndarray, 
                                 times: np.ndarray, 
                                 returns: np.ndarray,
                                 **kwargs: Any) -> np.ndarray:
        """Compute the noise variance from the preprocessed data.
        
        Args:
            prices: Preprocessed price data
            times: Preprocessed time points
            returns: Returns computed from prices
            **kwargs: Additional keyword arguments for computation
            
        Returns:
            np.ndarray: Noise variance estimate
            
        Raises:
            ValueError: If computation fails
        """
        # Estimate noise variance using the specified method
        if self.config.method == 'auto':
            # Try multiple methods and select the most reliable one
            estimates = {}
            
            try:
                estimates['ac1'] = ac1_noise_variance(returns, self.config.bias_correction)
            except Exception as e:
                logger.warning(f"AC(1) method failed: {str(e)}")
            
            try:
                estimates['bandi-russell'] = bandi_russell_noise(returns, self.config.bias_correction)
            except Exception as e:
                logger.warning(f"Bandi-Russell method failed: {str(e)}")
            
            try:
                signature_result = signature_plot_noise(
                    returns, 
                    min_lag=self.config.min_lags, 
                    max_lag=self.config.max_lags,
                    bias_correction=self.config.bias_correction,
                    return_plot_data=True
                )
                estimates['signature'] = signature_result[0]
                signature_plot_data = signature_result[1]
            except Exception as e:
                logger.warning(f"Signature plot method failed: {str(e)}")
                signature_plot_data = None
            
            try:
                estimates['ml'] = ml_noise_variance(returns, self.config.bias_correction)
            except Exception as e:
                logger.warning(f"Maximum likelihood method failed: {str(e)}")
            
            # Select the most reliable estimate
            if len(estimates) == 0:
                raise ValueError("All noise variance estimation methods failed")
            
            # Prefer methods in this order: signature, bandi-russell, ac1, ml
            method_preference = ['signature', 'bandi-russell', 'ac1', 'ml']
            selected_method = None
            
            for method in method_preference:
                if method in estimates:
                    selected_method = method
                    break
            
            if selected_method is None:
                # If none of the preferred methods worked, use the first available
                selected_method = list(estimates.keys())[0]
            
            noise_var = estimates[selected_method]
            method_used = selected_method
            
        else:
            # Use the specified method
            method_used = self.config.method
            signature_plot_data = None
            
            if method_used == 'ac1':
                noise_var = ac1_noise_variance(returns, self.config.bias_correction)
            
            elif method_used == 'bandi-russell':
                noise_var = bandi_russell_noise(returns, self.config.bias_correction)
            
            elif method_used == 'signature':
                signature_result = signature_plot_noise(
                    returns, 
                    min_lag=self.config.min_lags, 
                    max_lag=self.config.max_lags,
                    bias_correction=self.config.bias_correction,
                    return_plot_data=True
                )
                noise_var = signature_result[0]
                signature_plot_data = signature_result[1]
            
            elif method_used == 'ml':
                noise_var = ml_noise_variance(returns, self.config.bias_correction)
            
            else:
                raise ValueError(f"Unrecognized method: {method_used}")
        
        # Compute noise-to-signal ratio
        # Signal is the realized variance (sum of squared returns)
        realized_var = np.sum(returns**2)
        noise_to_signal = noise_var / (realized_var / len(returns))
        
        # Store additional results
        self._noise_variance = noise_var
        
        # Create result object with additional information
        result = NoiseVarianceResult(
            model_name=self._name,
            realized_measure=np.array([noise_var]),  # Store as array for consistency
            prices=prices,
            times=times,
            returns=returns,
            noise_variance=noise_var,
            method=method_used,
            signature_plot=signature_plot_data,
            noise_to_signal=noise_to_signal,
            bias_correction=self.config.bias_correction,
            all_estimates=estimates if self.config.return_all and self.config.method == 'auto' else None
        )
        
        # Store result
        self._results = result
        
        # Generate plot if requested
        if self.config.plot and HAS_MATPLOTLIB:
            result.plot()
        
        return np.array([noise_var])
    
    def fit(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> NoiseVarianceResult:
        """Fit the noise variance estimator to the provided data.
        
        Args:
            data: The data to fit the estimator to, as a tuple of (prices, times)
            **kwargs: Additional keyword arguments for estimation
            
        Returns:
            NoiseVarianceResult: The estimation results
            
        Raises:
            ValueError: If the data is invalid
            RuntimeError: If the estimation fails
        """
        result = super().fit(data, **kwargs)
        return cast(NoiseVarianceResult, result)
    
    async def fit_async(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> NoiseVarianceResult:
        """Asynchronously fit the noise variance estimator to the provided data.
        
        Args:
            data: The data to fit the estimator to, as a tuple of (prices, times)
            **kwargs: Additional keyword arguments for estimation
            
        Returns:
            NoiseVarianceResult: The estimation results
            
        Raises:
            ValueError: If the data is invalid
            RuntimeError: If the estimation fails
        """
        result = await super().fit_async(data, **kwargs)
        return cast(NoiseVarianceResult, result)
    
    def calibrate(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> NoiseVarianceConfig:
        """Calibrate the estimator configuration based on the provided data.
        
        This method analyzes the input data and determines optimal configuration
        parameters for the noise variance estimator.
        
        Args:
            data: The data to calibrate the estimator with, as a tuple of (prices, times)
            **kwargs: Additional keyword arguments for calibration
            
        Returns:
            NoiseVarianceConfig: Calibrated configuration
        """
        # Validate input data
        self.validate_data(data)
        prices, times = data
        
        # Preprocess data
        processed_prices, processed_times, returns = self.preprocess_data(prices, times)
        
        # Try different methods and select the most reliable one
        methods = ['ac1', 'bandi-russell', 'signature', 'ml']
        estimates = {}
        errors = {}
        
        for method in methods:
            try:
                # Create a temporary config with this method
                temp_config = NoiseVarianceConfig(method=method)
                
                # Create a temporary estimator
                temp_estimator = NoiseVarianceEstimator(config=temp_config)
                
                # Fit the estimator
                result = temp_estimator.fit(data)
                
                # Store the estimate
                estimates[method] = result.noise_variance
            except Exception as e:
                errors[method] = str(e)
        
        # Select the most reliable method
        if len(estimates) == 0:
            raise RuntimeError(f"All noise variance estimation methods failed: {errors}")
        
        # Compute the median of all estimates
        median_estimate = np.median(list(estimates.values()))
        
        # Select the method closest to the median
        best_method = min(estimates.keys(), key=lambda m: abs(estimates[m] - median_estimate))
        
        # Create calibrated configuration
        calibrated_config = NoiseVarianceConfig(
            method=best_method,
            max_lags=min(20, len(returns) // 10),  # Adaptive max_lags based on data length
            min_lags=1,
            bias_correction=True,
            robust=False,
            return_all=False,
            plot=False
        )
        
        return calibrated_config


@jit(nopython=True, cache=True)
def _ac1_noise_variance_numba(returns: np.ndarray, bias_correction: bool = True) -> float:
    """
    Numba-accelerated implementation of AC(1) noise variance estimation.
    
    Args:
        returns: Array of returns
        bias_correction: Whether to apply bias correction
        
    Returns:
        Estimated noise variance
    """
    n = len(returns)
    
    # Compute first-order autocovariance
    acov = 0.0
    for i in range(n - 1):
        acov += returns[i] * returns[i + 1]
    acov /= (n - 1)
    
    # Noise variance is -0.5 * first-order autocovariance
    noise_var = -0.5 * acov
    
    # Apply bias correction if requested
    if bias_correction:
        # Compute realized variance
        rv = 0.0
        for i in range(n):
            rv += returns[i]**2
        rv /= n
        
        # Adjust for small sample bias
        noise_var *= (1.0 + 1.0 / n)
    
    # If noise_var is negative, use alternative method
    if noise_var <= 0:
        # Use 0.5 * mean squared returns
        noise_var = 0.0
        for i in range(n):
            noise_var += returns[i]**2
        noise_var *= 0.5 / n
    
    return noise_var


def ac1_noise_variance(returns: np.ndarray, bias_correction: bool = True) -> float:
    """
    Estimate noise variance using first-order autocovariance (AC1) method.
    
    This method estimates the variance of microstructure noise based on the
    first-order autocovariance of returns. Under the assumption of i.i.d. noise,
    the noise variance is -0.5 times the first-order autocovariance.
    
    Args:
        returns: Array of returns
        bias_correction: Whether to apply bias correction
        
    Returns:
        Estimated noise variance
        
    Raises:
        ValueError: If returns is not a 1D array or contains invalid values
        
    References:
        Oomen, R. C. (2006). Properties of realized variance under alternative
        sampling schemes. Journal of Business & Economic Statistics, 24(2), 219-237.
    
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.noise_estimate import ac1_noise_variance
        >>> np.random.seed(42)
        >>> returns = np.random.normal(0, 0.01, 1000)
        >>> noise_var = ac1_noise_variance(returns)
        >>> noise_var
        0.00005...
    """
    # Convert to numpy array if not already
    returns = np.asarray(returns)
    
    # Validate inputs
    if returns.ndim != 1:
        raise ValueError("returns must be a 1D array")
    
    if np.isnan(returns).any() or np.isinf(returns).any():
        raise ValueError("returns contains NaN or infinite values")
    
    # Use Numba-accelerated implementation if available
    if HAS_NUMBA:
        return _ac1_noise_variance_numba(returns, bias_correction)
    
    # Pure NumPy implementation
    n = len(returns)
    
    # Compute first-order autocovariance
    acov = np.mean(returns[:-1] * returns[1:])
    
    # Noise variance is -0.5 * first-order autocovariance
    noise_var = -0.5 * acov
    
    # Apply bias correction if requested
    if bias_correction:
        # Compute realized variance
        rv = np.mean(returns**2)
        
        # Adjust for small sample bias
        noise_var *= (1.0 + 1.0 / n)
    
    # If noise_var is negative, use alternative method
    if noise_var <= 0:
        logger.warning("AC(1) method produced non-positive noise variance. "
                      "Falling back to 0.5 * mean squared returns.")
        noise_var = 0.5 * np.mean(returns**2)
    
    return noise_var


@jit(nopython=True, cache=True)
def _bandi_russell_noise_numba(returns: np.ndarray, bias_correction: bool = True) -> float:
    """
    Numba-accelerated implementation of Bandi-Russell noise variance estimation.
    
    Args:
        returns: Array of returns
        bias_correction: Whether to apply bias correction
        
    Returns:
        Estimated noise variance
    """
    n = len(returns)
    
    # Compute realized variance
    rv = 0.0
    for i in range(n):
        rv += returns[i]**2
    rv /= n
    
    # Compute first-order autocovariance
    acov = 0.0
    for i in range(n - 1):
        acov += returns[i] * returns[i + 1]
    acov /= (n - 1)
    
    # Compute second-order autocovariance
    acov2 = 0.0
    for i in range(n - 2):
        acov2 += returns[i] * returns[i + 2]
    acov2 /= (n - 2)
    
    # Bandi-Russell estimator
    noise_var = 0.5 * (rv + acov)
    
    # Apply bias correction if requested
    if bias_correction:
        # Adjust for small sample bias
        noise_var -= 0.25 * acov2
    
    # Ensure non-negativity
    if noise_var <= 0:
        # Use 0.5 * mean squared returns
        noise_var = 0.5 * rv
    
    return noise_var


def bandi_russell_noise(returns: np.ndarray, bias_correction: bool = True) -> float:
    """
    Estimate noise variance using Bandi-Russell method.
    
    This method estimates the variance of microstructure noise based on the
    realized variance and autocovariances of returns. It provides a more robust
    estimate than the simple AC(1) method, especially for data with serial
    correlation in the efficient price process.
    
    Args:
        returns: Array of returns
        bias_correction: Whether to apply bias correction
        
    Returns:
        Estimated noise variance
        
    Raises:
        ValueError: If returns is not a 1D array or contains invalid values
        
    References:
        Bandi, F. M., & Russell, J. R. (2008). Microstructure noise, realized
        variance, and optimal sampling. The Review of Economic Studies, 75(2), 339-369.
    
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.noise_estimate import bandi_russell_noise
        >>> np.random.seed(42)
        >>> returns = np.random.normal(0, 0.01, 1000)
        >>> noise_var = bandi_russell_noise(returns)
        >>> noise_var
        0.00005...
    """
    # Convert to numpy array if not already
    returns = np.asarray(returns)
    
    # Validate inputs
    if returns.ndim != 1:
        raise ValueError("returns must be a 1D array")
    
    if np.isnan(returns).any() or np.isinf(returns).any():
        raise ValueError("returns contains NaN or infinite values")
    
    # Use Numba-accelerated implementation if available
    if HAS_NUMBA:
        return _bandi_russell_noise_numba(returns, bias_correction)
    
    # Pure NumPy implementation
    n = len(returns)
    
    # Compute realized variance
    rv = np.mean(returns**2)
    
    # Compute first-order autocovariance
    acov = np.mean(returns[:-1] * returns[1:])
    
    # Compute second-order autocovariance
    acov2 = np.mean(returns[:-2] * returns[2:])
    
    # Bandi-Russell estimator
    noise_var = 0.5 * (rv + acov)
    
    # Apply bias correction if requested
    if bias_correction:
        # Adjust for small sample bias
        noise_var -= 0.25 * acov2
    
    # Ensure non-negativity
    if noise_var <= 0:
        logger.warning("Bandi-Russell method produced non-positive noise variance. "
                      "Falling back to 0.5 * mean squared returns.")
        noise_var = 0.5 * rv
    
    return noise_var


def signature_plot_noise(returns: np.ndarray, min_lag: int = 1, max_lag: int = 20,
                        bias_correction: bool = True, return_plot_data: bool = False) -> Union[float, Tuple[float, Dict[str, np.ndarray]]]:
    """
    Estimate noise variance using signature plot method.
    
    This method estimates the variance of microstructure noise by analyzing the
    behavior of realized variance at different sampling frequencies (lags).
    Under the presence of microstructure noise, the realized variance increases
    as the sampling frequency increases, and the slope of this relationship
    is related to the noise variance.
    
    Args:
        returns: Array of returns
        min_lag: Minimum lag to consider
        max_lag: Maximum lag to consider
        bias_correction: Whether to apply bias correction
        return_plot_data: Whether to return data for plotting
        
    Returns:
        If return_plot_data is False:
            Estimated noise variance
        If return_plot_data is True:
            Tuple of (estimated noise variance, plot data dictionary)
            
    Raises:
        ValueError: If returns is not a 1D array or contains invalid values
        
    References:
        Andersen, T. G., Bollerslev, T., Diebold, F. X., & Labys, P. (2000).
        Great realizations. Risk, 13(3), 105-108.
    
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.noise_estimate import signature_plot_noise
        >>> np.random.seed(42)
        >>> returns = np.random.normal(0, 0.01, 1000)
        >>> noise_var = signature_plot_noise(returns)
        >>> noise_var
        0.00005...
    """
    # Convert to numpy array if not already
    returns = np.asarray(returns)
    
    # Validate inputs
    if returns.ndim != 1:
        raise ValueError("returns must be a 1D array")
    
    if np.isnan(returns).any() or np.isinf(returns).any():
        raise ValueError("returns contains NaN or infinite values")
    
    if min_lag < 1:
        raise ValueError("min_lag must be at least 1")
    
    if max_lag < min_lag:
        raise ValueError("max_lag must be greater than or equal to min_lag")
    
    n = len(returns)
    max_lag = min(max_lag, n // 5)  # Ensure max_lag is not too large
    
    # Compute realized variance at different lags
    lags = np.arange(min_lag, max_lag + 1)
    rv = np.zeros(len(lags))
    
    for i, lag in enumerate(lags):
        # Skip every lag-th observation
        sampled_returns = returns[::lag]
        rv[i] = np.sum(sampled_returns**2) / len(sampled_returns)
    
    # Fit linear model: RV(h) = IV + 2*q*h
    # where h is the sampling interval, IV is integrated variance, q is noise variance
    X = np.column_stack((np.ones_like(lags), lags))
    y = rv
    
    # Compute linear regression coefficients
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    
    # Intercept is integrated variance, slope is 2*q
    integrated_var = beta[0]
    noise_var = beta[1] / 2.0
    
    # Apply bias correction if requested
    if bias_correction:
        # Adjust for small sample bias
        noise_var *= (1.0 + 1.0 / n)
    
    # Ensure non-negativity
    if noise_var <= 0:
        logger.warning("Signature plot method produced non-positive noise variance. "
                      "Falling back to 0.5 * mean squared returns.")
        noise_var = 0.5 * np.mean(returns**2)
    
    # Return results
    if return_plot_data:
        # Compute fitted values
        fitted = X @ beta
        
        # Create plot data dictionary
        plot_data = {
            'lags': lags,
            'rv': rv,
            'fit': fitted,
            'integrated_var': integrated_var,
            'noise_var': noise_var
        }
        
        return noise_var, plot_data
    else:
        return noise_var


def ml_noise_variance(returns: np.ndarray, bias_correction: bool = True) -> float:
    """
    Estimate noise variance using maximum likelihood method.
    
    This method estimates the variance of microstructure noise using a maximum
    likelihood approach, assuming that returns are the sum of an efficient price
    component and i.i.d. noise.
    
    Args:
        returns: Array of returns
        bias_correction: Whether to apply bias correction
        
    Returns:
        Estimated noise variance
        
    Raises:
        ValueError: If returns is not a 1D array or contains invalid values
        
    References:
        Ait-Sahalia, Y., Mykland, P. A., & Zhang, L. (2005). How often to sample
        a continuous-time process in the presence of market microstructure noise.
        The Review of Financial Studies, 18(2), 351-416.
    
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.noise_estimate import ml_noise_variance
        >>> np.random.seed(42)
        >>> returns = np.random.normal(0, 0.01, 1000)
        >>> noise_var = ml_noise_variance(returns)
        >>> noise_var
        0.00005...
    """
    # Convert to numpy array if not already
    returns = np.asarray(returns)
    
    # Validate inputs
    if returns.ndim != 1:
        raise ValueError("returns must be a 1D array")
    
    if np.isnan(returns).any() or np.isinf(returns).any():
        raise ValueError("returns contains NaN or infinite values")
    
    n = len(returns)
    
    # Define negative log-likelihood function
    def neg_log_likelihood(params):
        # params[0] is integrated variance, params[1] is noise variance
        iv, nv = params
        
        if iv <= 0 or nv <= 0:
            return 1e10  # Large value for invalid parameters
        
        # Compute autocovariance matrix
        acov_matrix = np.zeros((n, n))
        
        # Diagonal elements (variance)
        for i in range(n):
            acov_matrix[i, i] = iv + 2 * nv
        
        # Off-diagonal elements (first-order autocovariance)
        for i in range(n - 1):
            acov_matrix[i, i + 1] = -nv
            acov_matrix[i + 1, i] = -nv
        
        # Compute log-likelihood
        try:
            # Use Cholesky decomposition for numerical stability
            L = np.linalg.cholesky(acov_matrix)
            log_det = 2 * np.sum(np.log(np.diag(L)))
            
            # Solve linear system instead of computing inverse
            z = np.linalg.solve(L, returns)
            quad_form = np.sum(z**2)
            
            # Negative log-likelihood
            nll = 0.5 * (log_det + quad_form + n * np.log(2 * np.pi))
            
            return nll
        except np.linalg.LinAlgError:
            return 1e10  # Large value for numerical issues
    
    # Initial guess based on simple estimators
    rv = np.mean(returns**2)
    acov = np.mean(returns[:-1] * returns[1:])
    
    initial_noise_var = -0.5 * acov
    if initial_noise_var <= 0:
        initial_noise_var = 0.1 * rv
    
    initial_iv = rv - 2 * initial_noise_var
    if initial_iv <= 0:
        initial_iv = 0.5 * rv
    
    initial_params = [initial_iv, initial_noise_var]
    
    # Bounds for parameters (both must be positive)
    bounds = [(1e-10, None), (1e-10, None)]
    
    # Optimize negative log-likelihood
    try:
        result = optimize.minimize(
            neg_log_likelihood,
            initial_params,
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        # Extract estimated parameters
        iv_est, noise_var = result.x
        
        # Apply bias correction if requested
        if bias_correction:
            # Adjust for small sample bias
            noise_var *= (1.0 + 1.0 / n)
        
        # Ensure non-negativity (should be guaranteed by bounds)
        if noise_var <= 0:
            logger.warning("Maximum likelihood method produced non-positive noise variance. "
                          "Falling back to 0.5 * mean squared returns.")
            noise_var = 0.5 * rv
        
        return noise_var
    
    except Exception as e:
        logger.warning(f"Maximum likelihood optimization failed: {str(e)}. "
                      "Falling back to AC(1) method.")
        return ac1_noise_variance(returns, bias_correction)


def noise_variance(returns: np.ndarray, method: str = 'auto', 
                  bias_correction: bool = True, **kwargs: Any) -> float:
    """
    Estimate noise variance using various methods.
    
    This function provides a unified interface to multiple noise estimation methods,
    including AC(1), Bandi-Russell, signature plot, and maximum likelihood approaches.
    
    Args:
        returns: Array of returns
        method: Method for estimating noise variance
                ('ac1', 'bandi-russell', 'signature', 'ml', 'auto')
        bias_correction: Whether to apply bias correction
        **kwargs: Additional keyword arguments for specific methods
        
    Returns:
        Estimated noise variance
        
    Raises:
        ValueError: If returns is not a 1D array or contains invalid values
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.noise_estimate import noise_variance
        >>> np.random.seed(42)
        >>> returns = np.random.normal(0, 0.01, 1000)
        >>> noise_var = noise_variance(returns, method='auto')
        >>> noise_var
        0.00005...
    """
    # Convert to numpy array if not already
    returns = np.asarray(returns)
    
    # Validate inputs
    if returns.ndim != 1:
        raise ValueError("returns must be a 1D array")
    
    if np.isnan(returns).any() or np.isinf(returns).any():
        raise ValueError("returns contains NaN or infinite values")
    
    # Extract method-specific parameters from kwargs
    min_lag = kwargs.get('min_lag', 1)
    max_lag = kwargs.get('max_lag', 20)
    
    # Estimate noise variance using the specified method
    if method == 'auto':
        # Try multiple methods and select the most reliable one
        estimates = {}
        
        try:
            estimates['ac1'] = ac1_noise_variance(returns, bias_correction)
        except Exception as e:
            logger.warning(f"AC(1) method failed: {str(e)}")
        
        try:
            estimates['bandi-russell'] = bandi_russell_noise(returns, bias_correction)
        except Exception as e:
            logger.warning(f"Bandi-Russell method failed: {str(e)}")
        
        try:
            estimates['signature'] = signature_plot_noise(
                returns, 
                min_lag=min_lag, 
                max_lag=max_lag,
                bias_correction=bias_correction
            )
        except Exception as e:
            logger.warning(f"Signature plot method failed: {str(e)}")
        
        try:
            estimates['ml'] = ml_noise_variance(returns, bias_correction)
        except Exception as e:
            logger.warning(f"Maximum likelihood method failed: {str(e)}")
        
        # Select the most reliable estimate
        if len(estimates) == 0:
            raise ValueError("All noise variance estimation methods failed")
        
        # Prefer methods in this order: signature, bandi-russell, ac1, ml
        method_preference = ['signature', 'bandi-russell', 'ac1', 'ml']
        selected_method = None
        
        for m in method_preference:
            if m in estimates:
                selected_method = m
                break
        
        if selected_method is None:
            # If none of the preferred methods worked, use the first available
            selected_method = list(estimates.keys())[0]
        
        return estimates[selected_method]
    
    elif method == 'ac1':
        return ac1_noise_variance(returns, bias_correction)
    
    elif method == 'bandi-russell':
        return bandi_russell_noise(returns, bias_correction)
    
    elif method == 'signature':
        return signature_plot_noise(
            returns, 
            min_lag=min_lag, 
            max_lag=max_lag,
            bias_correction=bias_correction
        )
    
    elif method == 'ml':
        return ml_noise_variance(returns, bias_correction)
    
    else:
        raise ValueError(f"Unrecognized method: {method}. "
                         f"Supported methods are 'ac1', 'bandi-russell', 'signature', 'ml', 'auto'.")


def plot_noise_signature(returns: np.ndarray, min_lag: int = 1, max_lag: int = 20,
                        figsize: Tuple[int, int] = (10, 6)) -> Optional[Any]:
    """
    Visualize noise characteristics using signature plots.
    
    This function generates a signature plot showing the behavior of realized
    variance at different sampling frequencies (lags), which is useful for
    visualizing the presence and magnitude of microstructure noise.
    
    Args:
        returns: Array of returns
        min_lag: Minimum lag to consider
        max_lag: Maximum lag to consider
        figsize: Figure size (width, height) in inches
        
    Returns:
        Optional[Any]: Matplotlib figure if matplotlib is available, None otherwise
        
    Raises:
        ImportError: If matplotlib is not available
        ValueError: If returns is not a 1D array or contains invalid values
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.noise_estimate import plot_noise_signature
        >>> np.random.seed(42)
        >>> returns = np.random.normal(0, 0.01, 1000)
        >>> fig = plot_noise_signature(returns)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib is required for plotting. Please install matplotlib.")
    
    # Estimate noise variance and get plot data
    noise_var, plot_data = signature_plot_noise(
        returns, 
        min_lag=min_lag, 
        max_lag=max_lag,
        return_plot_data=True
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot signature plot
    lags = plot_data['lags']
    rv = plot_data['rv']
    fit = plot_data['fit']
    
    ax.plot(lags, rv, 'o-', label='Realized Variance')
    ax.plot(lags, fit, 'r--', label='Linear Fit')
    
    ax.set_xlabel('Lag')
    ax.set_ylabel('Realized Variance')
    ax.set_title('Signature Plot for Noise Variance Estimation')
    ax.legend()
    
    # Add noise variance estimate as text
    ax.text(0.05, 0.95, f'Noise Variance: {noise_var:.6e}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig


# Register Numba-accelerated functions if available
def _register_numba_functions() -> None:
    """
    Register Numba JIT-compiled functions for noise variance estimation.
    
    This function is called during module initialization to register
    performance-critical functions for JIT compilation if Numba is available.
    """
    if HAS_NUMBA:
        # The functions are already decorated with @jit, so no additional
        # registration is needed here. This function is kept for consistency
        # with the module structure and potential future enhancements.
        logger.debug("Noise variance estimation Numba JIT functions registered")
    else:
        logger.info("Numba not available. Noise variance estimation will use pure NumPy implementations.")


# Initialize the module
_register_numba_functions()
