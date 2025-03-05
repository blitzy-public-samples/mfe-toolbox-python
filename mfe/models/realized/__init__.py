"""
MFE Toolbox Realized Volatility Module

This module provides a comprehensive collection of realized volatility estimators
for high-frequency financial data analysis. These estimators enable non-parametric
volatility measurement using intraday price data, offering robust alternatives to
model-based volatility estimation.

Key components include:
- Basic realized variance estimators
- Noise-robust realized kernel estimators
- Jump-robust bipower variation estimators
- Directional semivariance estimators
- Multivariate realized covariance estimators
- Time conversion utilities for high-frequency data
- Price and return filtering functions
- Subsampling and averaging techniques

The realized volatility estimators are optimized for performance using Numba's
just-in-time compilation, enabling efficient processing of large high-frequency
datasets while maintaining the flexibility and readability of Python code.
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Set
import warnings

# Set up module-level logger
logger = logging.getLogger("mfe.models.realized")

# Import version information
from ...version import __version__

# Import base classes
from .base import (
    BaseRealizedEstimator,
    RealizedEstimatorResult,
    RealizedEstimatorConfig
)

# Import core estimators
from .variance import RealizedVariance
from .bipower_variation import BiPowerVariation
from .kernel import RealizedKernel
from .semivariance import RealizedSemivariance
from .covariance import RealizedCovariance
from .multivariate_kernel import MultivariateRealizedKernel
from .range import RealizedRange
from .quarticity import RealizedQuarticity
from .twoscale_variance import TwoScaleRealizedVariance
from .multiscale_variance import MultiscaleRealizedVariance
from .qmle_variance import QMLERealizedVariance
from .threshold_multipower_variation import ThresholdMultipowerVariation
from .threshold_variance import ThresholdRealizedVariance
from .preaveraged_bipower_variation import PreaveragedBiPowerVariation
from .preaveraged_variance import PreaveragedRealizedVariance

# Import kernel-related components
from .kernel_bandwidth import (
    optimal_bandwidth,
    asymptotic_optimal_bandwidth,
    improved_asymptotic_optimal_bandwidth
)
from .kernel_weights import (
    parzen_weights,
    bartlett_weights,
    tukey_hanning_weights,
    qs_weights,
    cubic_weights
)
from .kernel_jitter_lag_length import (
    jitter_lag_length,
    jitter_data_length
)

# Import time conversion utilities
from .seconds2unit import seconds2unit
from .seconds2wall import seconds2wall
from .unit2seconds import unit2seconds
from .unit2wall import unit2wall
from .wall2seconds import wall2seconds
from .wall2unit import wall2unit

# Import filtering and preprocessing utilities
from .price_filter import price_filter
from .return_filter import return_filter
from .refresh_time import refresh_time
from .refresh_time_bivariate import refresh_time_bivariate
from .subsample import subsample
from .noise_estimate import noise_variance

# Import utility functions
from .utils import (
    align_time,
    compute_returns,
    detect_jumps,
    optimal_sampling,
    signature_plot
)

# Import Numba-accelerated core functions
try:
    from ._numba_core import (
        _realized_variance_core,
        _realized_bipower_variation_core,
        _realized_quarticity_core,
        _realized_kernel_core,
        _realized_semivariance_core,
        _realized_covariance_core,
        _threshold_detection_core
    )
    _has_numba = True
except ImportError:
    logger.warning("Numba not available. Using slower pure Python implementations.")
    _has_numba = False

def has_numba() -> bool:
    """
    Check if Numba is available for accelerated computations.
    
    Returns:
        bool: True if Numba is available, False otherwise
    """
    return _has_numba

def list_estimators() -> Dict[str, List[str]]:
    """
    List all available realized volatility estimators.
    
    Returns:
        Dict mapping estimator categories to lists of available estimator classes
    """
    return {
        "basic": [
            "RealizedVariance",
            "RealizedRange",
            "RealizedQuarticity"
        ],
        "jump_robust": [
            "BiPowerVariation",
            "ThresholdRealizedVariance",
            "ThresholdMultipowerVariation"
        ],
        "noise_robust": [
            "RealizedKernel",
            "TwoScaleRealizedVariance",
            "MultiscaleRealizedVariance",
            "QMLERealizedVariance",
            "PreaveragedRealizedVariance",
            "PreaveragedBiPowerVariation"
        ],
        "directional": [
            "RealizedSemivariance"
        ],
        "multivariate": [
            "RealizedCovariance",
            "MultivariateRealizedKernel"
        ]
    }

# Define what's available when using "from mfe.models.realized import *"
__all__ = [
    # Base classes
    "BaseRealizedEstimator",
    "RealizedEstimatorResult",
    "RealizedEstimatorConfig",
    
    # Core estimators
    "RealizedVariance",
    "BiPowerVariation",
    "RealizedKernel",
    "RealizedSemivariance",
    "RealizedCovariance",
    "MultivariateRealizedKernel",
    "RealizedRange",
    "RealizedQuarticity",
    "TwoScaleRealizedVariance",
    "MultiscaleRealizedVariance",
    "QMLERealizedVariance",
    "ThresholdMultipowerVariation",
    "ThresholdRealizedVariance",
    "PreaveragedBiPowerVariation",
    "PreaveragedRealizedVariance",
    
    # Kernel components
    "optimal_bandwidth",
    "asymptotic_optimal_bandwidth",
    "improved_asymptotic_optimal_bandwidth",
    "parzen_weights",
    "bartlett_weights",
    "tukey_hanning_weights",
    "qs_weights",
    "cubic_weights",
    "jitter_lag_length",
    "jitter_data_length",
    
    # Time conversion utilities
    "seconds2unit",
    "seconds2wall",
    "unit2seconds",
    "unit2wall",
    "wall2seconds",
    "wall2unit",
    
    # Filtering and preprocessing
    "price_filter",
    "return_filter",
    "refresh_time",
    "refresh_time_bivariate",
    "subsample",
    "noise_variance",
    
    # Utility functions
    "align_time",
    "compute_returns",
    "detect_jumps",
    "optimal_sampling",
    "signature_plot",
    
    # Helper functions
    "has_numba",
    "list_estimators"
]

logger.debug("MFE Realized Volatility module initialized successfully")
