# mfe/models/realized/kernel_jitter_lag_length.py
"""
Optimal endpoint jitter lag length determination for kernel-based volatility estimators.

This module provides functionality for determining the optimal endpoint jitter lags
for kernel-based volatility estimators, which is critical for mitigating end-effects
in kernel estimation. The implementation balances microstructure noise and integrated
quarticity influences to optimize the jitter lag selection.

The module implements a robust search algorithm for optimal lag length with comprehensive
type hints, parameter validation, and Numba acceleration for performance-critical
calculations. It provides diagnostic information about the jitter lag selection process
to aid in understanding the estimator behavior.

Functions:
    compute_jitter_lag_length: Main function for computing optimal jitter lag length
    compute_jitter_lag_length_by_kernel: Compute jitter lag length for specific kernel type
    get_jitter_parameters: Get kernel-specific parameters for jitter lag computation
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Literal, Optional, Sequence, 
    Tuple, Type, TypeVar, Union, cast, overload
)
import numpy as np
from scipy import stats, optimize

from ...core.exceptions import ParameterError, NumericError
from ...core.parameters import validate_positive, validate_non_negative
from .utils import noise_variance
from .kernel_bandwidth import compute_integrated_quarticity

# Set up module-level logger
logger = logging.getLogger("mfe.models.realized.kernel_jitter_lag_length")

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
    logger.debug("Numba available for jitter lag length computation acceleration")
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    HAS_NUMBA = False
    logger.info("Numba not available. Jitter lag length computation will use pure NumPy implementations.")


@dataclass
class JitterLagResult:
    """Result container for jitter lag length computation.
    
    This class provides a structured container for jitter lag computation results,
    including the optimal lag length and diagnostic information about the selection process.
    
    Attributes:
        jitter_lag: Optimal jitter lag length
        kernel_type: Type of kernel function used
        bandwidth: Bandwidth parameter used
        noise_variance: Estimated noise variance
        integrated_quarticity: Estimated integrated quarticity
        search_values: Lag values considered during search
        objective_values: Objective function values for each lag
        method: Method used for lag selection
        diagnostic_info: Additional diagnostic information
    """
    
    jitter_lag: int
    kernel_type: str
    bandwidth: float
    noise_variance: float
    integrated_quarticity: float
    search_values: Optional[np.ndarray] = None
    objective_values: Optional[np.ndarray] = None
    method: str = "optimal"
    diagnostic_info: Optional[Dict[str, Any]] = None
    
    def __post_init__(self) -> None:
        """Validate result object after initialization."""
        # Ensure arrays are NumPy arrays if provided
        if self.search_values is not None and not isinstance(self.search_values, np.ndarray):
            self.search_values = np.array(self.search_values)
        
        if self.objective_values is not None and not isinstance(self.objective_values, np.ndarray):
            self.objective_values = np.array(self.objective_values)
    
    def summary(self) -> str:
        """Generate a text summary of the jitter lag results.
        
        Returns:
            str: A formatted string containing the jitter lag results summary
        """
        summary_text = f"Jitter Lag Length Results for {self.kernel_type.capitalize()} Kernel\n"
        summary_text += f"{'=' * 50}\n"
        summary_text += f"Optimal Jitter Lag: {self.jitter_lag}\n"
        summary_text += f"Bandwidth: {self.bandwidth:.2f}\n"
        summary_text += f"Noise Variance: {self.noise_variance:.6e}\n"
        summary_text += f"Integrated Quarticity: {self.integrated_quarticity:.6e}\n"
        summary_text += f"Method: {self.method}\n"
        
        if self.diagnostic_info:
            summary_text += "\nDiagnostic Information:\n"
            for key, value in self.diagnostic_info.items():
                summary_text += f"  {key}: {value}\n"
        
        return summary_text
    
    def plot(self, figsize: Tuple[int, int] = (10, 6)) -> Any:
        """Plot the objective function values for different lag lengths.
        
        Args:
            figsize: Figure size as (width, height) in inches
            
        Returns:
            matplotlib.figure.Figure: The generated figure
            
        Raises:
            ImportError: If matplotlib is not available
            ValueError: If search_values or objective_values are not available
        """
        if self.search_values is None or self.objective_values is None:
            raise ValueError("Search values and objective values are required for plotting")
        
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=figsize)
            
            ax.plot(self.search_values, self.objective_values, 'o-')
            ax.axvline(x=self.jitter_lag, color='r', linestyle='--', 
                      label=f'Optimal Lag: {self.jitter_lag}')
            
            ax.set_title(f"Jitter Lag Selection for {self.kernel_type.capitalize()} Kernel")
            ax.set_xlabel("Lag Length")
            ax.set_ylabel("Objective Function Value")
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            return fig
        except ImportError:
            raise ImportError("Matplotlib is required for plotting jitter lag results")


def get_jitter_parameters(kernel_type: str) -> Dict[str, Any]:
    """Get kernel-specific parameters for jitter lag computation.
    
    Different kernel types have different optimal jitter lag formulas based on their
    properties. This function returns the appropriate parameters for each kernel type.
    
    Args:
        kernel_type: Type of kernel function ('bartlett', 'parzen', 'tukey-hanning', 
                    'tukey', 'hanning', 'quadratic', 'flat-top')
        
    Returns:
        Dictionary of jitter parameters
        
    Raises:
        ValueError: If kernel_type is not recognized
    """
    # Normalize kernel type to lowercase
    kernel_type_lower = kernel_type.lower()
    
    # Define jitter parameters dictionary
    jitter_params = {
        'bartlett': {
            'jitter_factor': 0.5,
            'min_jitter': 1,
            'max_jitter_factor': 0.2,  # Max jitter as fraction of bandwidth
            'noise_weight': 0.6,       # Weight for noise influence
            'quarticity_weight': 0.4,  # Weight for quarticity influence
            'description': 'Linear kernel with moderate jitter needs'
        },
        'parzen': {
            'jitter_factor': 0.4,
            'min_jitter': 1,
            'max_jitter_factor': 0.15,
            'noise_weight': 0.7,
            'quarticity_weight': 0.3,
            'description': 'Smooth kernel with lower jitter needs'
        },
        'tukey-hanning': {
            'jitter_factor': 0.4,
            'min_jitter': 1,
            'max_jitter_factor': 0.15,
            'noise_weight': 0.7,
            'quarticity_weight': 0.3,
            'description': 'Cosine-based kernel with lower jitter needs'
        },
        'tukey': {
            'jitter_factor': 0.4,
            'min_jitter': 1,
            'max_jitter_factor': 0.15,
            'noise_weight': 0.7,
            'quarticity_weight': 0.3,
            'description': 'Alias for tukey-hanning'
        },
        'hanning': {
            'jitter_factor': 0.4,
            'min_jitter': 1,
            'max_jitter_factor': 0.15,
            'noise_weight': 0.7,
            'quarticity_weight': 0.3,
            'description': 'Alias for tukey-hanning'
        },
        'quadratic': {
            'jitter_factor': 0.35,
            'min_jitter': 1,
            'max_jitter_factor': 0.12,
            'noise_weight': 0.75,
            'quarticity_weight': 0.25,
            'description': 'Quadratic kernel with lower jitter needs'
        },
        'flat-top': {
            'jitter_factor': 0.6,
            'min_jitter': 2,
            'max_jitter_factor': 0.25,
            'noise_weight': 0.5,
            'quarticity_weight': 0.5,
            'description': 'Flat-top kernel with higher jitter needs'
        }
    }
    
    # Check if kernel type is recognized
    if kernel_type_lower not in jitter_params:
        valid_kernels = list(jitter_params.keys())
        raise ValueError(f"Unrecognized kernel type: {kernel_type}. "
                         f"Supported types are {valid_kernels}.")
    
    return jitter_params[kernel_type_lower]


@jit(nopython=True, cache=True)
def _compute_jitter_objective_numba(lag: int, 
                                   bandwidth: float, 
                                   noise_var: float, 
                                   quarticity: float,
                                   noise_weight: float,
                                   quarticity_weight: float) -> float:
    """
    Numba-accelerated computation of jitter lag objective function.
    
    This function evaluates the objective function for jitter lag selection,
    balancing the influence of microstructure noise and integrated quarticity.
    
    Args:
        lag: Jitter lag length to evaluate
        bandwidth: Bandwidth parameter
        noise_var: Noise variance
        quarticity: Integrated quarticity
        noise_weight: Weight for noise influence
        quarticity_weight: Weight for quarticity influence
        
    Returns:
        Objective function value (lower is better)
    """
    # Compute noise component (higher noise -> higher jitter needed)
    noise_component = noise_weight * (1.0 - np.exp(-noise_var * lag))
    
    # Compute quarticity component (higher quarticity -> lower jitter needed)
    quarticity_component = quarticity_weight * (np.exp(-quarticity * lag / bandwidth))
    
    # Compute objective (balance between noise and quarticity)
    # We want to maximize both components, so we minimize the negative sum
    objective = -(noise_component + quarticity_component)
    
    return objective


def compute_jitter_objective(lag: int, 
                           bandwidth: float, 
                           noise_var: float, 
                           quarticity: float,
                           noise_weight: float,
                           quarticity_weight: float) -> float:
    """
    Compute objective function for jitter lag selection.
    
    This function evaluates the objective function for jitter lag selection,
    balancing the influence of microstructure noise and integrated quarticity.
    
    Args:
        lag: Jitter lag length to evaluate
        bandwidth: Bandwidth parameter
        noise_var: Noise variance
        quarticity: Integrated quarticity
        noise_weight: Weight for noise influence
        quarticity_weight: Weight for quarticity influence
        
    Returns:
        Objective function value (lower is better)
        
    Raises:
        ValueError: If inputs have invalid values
    """
    # Validate inputs
    if lag < 0:
        raise ValueError("lag must be non-negative")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    if noise_var < 0:
        raise ValueError("noise_var must be non-negative")
    if quarticity < 0:
        raise ValueError("quarticity must be non-negative")
    
    # Use Numba-accelerated implementation if available
    if HAS_NUMBA:
        return _compute_jitter_objective_numba(
            lag, bandwidth, noise_var, quarticity, noise_weight, quarticity_weight
        )
    
    # Pure NumPy implementation
    # Compute noise component (higher noise -> higher jitter needed)
    noise_component = noise_weight * (1.0 - np.exp(-noise_var * lag))
    
    # Compute quarticity component (higher quarticity -> lower jitter needed)
    quarticity_component = quarticity_weight * (np.exp(-quarticity * lag / bandwidth))
    
    # Compute objective (balance between noise and quarticity)
    # We want to maximize both components, so we minimize the negative sum
    objective = -(noise_component + quarticity_component)
    
    return objective


def compute_jitter_lag_length_by_kernel(returns: np.ndarray, 
                                      kernel_type: str = 'bartlett',
                                      bandwidth: Optional[float] = None,
                                      noise_var: Optional[float] = None,
                                      quarticity: Optional[float] = None,
                                      method: str = 'optimal') -> JitterLagResult:
    """
    Compute optimal jitter lag length for a specific kernel type.
    
    This function determines the optimal jitter lag length for a given kernel type,
    taking into account the bandwidth, noise variance, and integrated quarticity.
    
    Args:
        returns: Array of returns
        kernel_type: Type of kernel function
        bandwidth: Bandwidth parameter (if None, defaults to sqrt(len(returns)))
        noise_var: Noise variance (if None, it will be estimated)
        quarticity: Integrated quarticity (if None, it will be estimated)
        method: Method for lag selection ('optimal', 'rule-of-thumb', 'fixed')
        
    Returns:
        JitterLagResult object containing the optimal jitter lag and diagnostic information
        
    Raises:
        ValueError: If returns has invalid dimensions, if kernel_type is not recognized,
                   or if method is not recognized
    """
    # Convert to numpy array if not already
    returns = np.asarray(returns)
    
    # Validate inputs
    if returns.ndim != 1:
        raise ValueError("returns must be a 1D array")
    
    # Get number of observations
    n = len(returns)
    
    # Set default bandwidth if not provided
    if bandwidth is None:
        bandwidth = np.sqrt(n)
    elif bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    
    # Estimate noise variance if not provided
    if noise_var is None:
        noise_var = noise_variance(returns)
    
    # Estimate integrated quarticity if not provided
    if quarticity is None:
        quarticity = compute_integrated_quarticity(returns)
    
    # Get jitter parameters for the specified kernel
    jitter_params = get_jitter_parameters(kernel_type)
    
    # Normalize method to lowercase
    method_lower = method.lower()
    
    # Compute jitter lag based on the specified method
    if method_lower == 'fixed':
        # Use fixed jitter lag based on minimum value
        jitter_lag = jitter_params['min_jitter']
        search_values = None
        objective_values = None
        diagnostic_info = {"method_description": "Fixed minimum jitter lag"}
        
    elif method_lower == 'rule-of-thumb':
        # Use rule-of-thumb formula based on bandwidth and jitter factor
        jitter_factor = jitter_params['jitter_factor']
        min_jitter = jitter_params['min_jitter']
        max_jitter = max(min_jitter, int(bandwidth * jitter_params['max_jitter_factor']))
        
        # Compute jitter lag as a fraction of bandwidth
        jitter_lag = max(min_jitter, int(jitter_factor * np.sqrt(bandwidth)))
        
        # Ensure jitter lag is within bounds
        jitter_lag = min(jitter_lag, max_jitter)
        
        search_values = None
        objective_values = None
        diagnostic_info = {
            "method_description": "Rule-of-thumb based on bandwidth",
            "jitter_factor": jitter_factor,
            "min_jitter": min_jitter,
            "max_jitter": max_jitter
        }
        
    elif method_lower == 'optimal':
        # Use optimization to find the best jitter lag
        min_jitter = jitter_params['min_jitter']
        max_jitter = max(min_jitter, int(bandwidth * jitter_params['max_jitter_factor']))
        noise_weight = jitter_params['noise_weight']
        quarticity_weight = jitter_params['quarticity_weight']
        
        # Define search range
        search_values = np.arange(min_jitter, max_jitter + 1)
        
        # Compute objective function for each lag value
        objective_values = np.array([
            compute_jitter_objective(
                lag, bandwidth, noise_var, quarticity, noise_weight, quarticity_weight
            )
            for lag in search_values
        ])
        
        # Find lag with minimum objective value
        optimal_idx = np.argmin(objective_values)
        jitter_lag = search_values[optimal_idx]
        
        diagnostic_info = {
            "method_description": "Optimal selection via objective function minimization",
            "min_jitter": min_jitter,
            "max_jitter": max_jitter,
            "noise_weight": noise_weight,
            "quarticity_weight": quarticity_weight,
            "optimal_objective": objective_values[optimal_idx]
        }
        
    else:
        raise ValueError(f"Unrecognized jitter lag selection method: {method}. "
                         f"Supported methods are 'optimal', 'rule-of-thumb', 'fixed'.")
    
    # Create and return result object
    result = JitterLagResult(
        jitter_lag=jitter_lag,
        kernel_type=kernel_type,
        bandwidth=bandwidth,
        noise_variance=noise_var,
        integrated_quarticity=quarticity,
        search_values=search_values,
        objective_values=objective_values,
        method=method_lower,
        diagnostic_info=diagnostic_info
    )
    
    return result


def compute_jitter_lag_length(returns: np.ndarray, 
                            kernel_type: str = 'bartlett',
                            bandwidth: Optional[float] = None,
                            noise_var: Optional[float] = None,
                            quarticity: Optional[float] = None,
                            method: str = 'optimal',
                            **kwargs: Any) -> Union[int, JitterLagResult]:
    """
    Compute optimal jitter lag length for kernel-based volatility estimators.
    
    This is the main function for jitter lag selection, which dispatches to the
    appropriate method based on the specified kernel type and approach.
    
    Args:
        returns: Array of returns
        kernel_type: Type of kernel function
        bandwidth: Bandwidth parameter (if None, defaults to sqrt(len(returns)))
        noise_var: Noise variance (if None, it will be estimated)
        quarticity: Integrated quarticity (if None, it will be estimated)
        method: Method for lag selection ('optimal', 'rule-of-thumb', 'fixed')
        **kwargs: Additional keyword arguments
            return_details: Whether to return detailed result object (default: False)
        
    Returns:
        If return_details is False (default): Optimal jitter lag length as integer
        If return_details is True: JitterLagResult object with full details
        
    Raises:
        ValueError: If returns has invalid dimensions, if kernel_type is not recognized,
                   or if method is not recognized
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.kernel_jitter_lag_length import compute_jitter_lag_length
        >>> np.random.seed(42)
        >>> returns = np.random.normal(0, 0.01, 1000)
        >>> compute_jitter_lag_length(returns, 'bartlett')
        2
        
        >>> result = compute_jitter_lag_length(returns, 'parzen', return_details=True)
        >>> result.jitter_lag
        1
    """
    # Extract additional keyword arguments
    return_details = kwargs.get('return_details', False)
    
    # Compute jitter lag length
    result = compute_jitter_lag_length_by_kernel(
        returns, kernel_type, bandwidth, noise_var, quarticity, method
    )
    
    # Return result based on return_details flag
    if return_details:
        return result
    else:
        return result.jitter_lag


# Register Numba-accelerated functions if available
def _register_numba_functions() -> None:
    """
    Register Numba JIT-compiled functions for jitter lag length computation.
    
    This function is called during module initialization to register
    performance-critical functions for JIT compilation if Numba is available.
    """
    if HAS_NUMBA:
        # The functions are already decorated with @jit, so no additional
        # registration is needed here. This function is kept for consistency
        # with the module structure and potential future enhancements.
        logger.debug("Jitter lag length Numba JIT functions registered")
    else:
        logger.info("Numba not available. Jitter lag length computation will use pure NumPy implementations.")


# Initialize the module
_register_numba_functions()
