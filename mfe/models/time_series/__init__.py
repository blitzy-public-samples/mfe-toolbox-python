# mfe/models/time_series/__init__.py
"""
MFE Toolbox Time Series Module

This module provides comprehensive time series analysis tools including ARMA/ARMAX
modeling, forecasting, unit root testing, impulse response analysis, and time series
decomposition utilities.

The time series module implements state-of-the-art econometric methods for modeling
and forecasting economic and financial time series, with a focus on robust estimation,
comprehensive diagnostics, and accurate forecasting.

Key components:
- ARMA/ARMAX models for time series modeling with exogenous variables
- VAR (Vector Autoregression) for multivariate time series analysis
- Unit root tests for stationarity assessment
- Time series filters for trend-cycle decomposition
- Impulse response analysis for dynamic system analysis
- HAR (Heterogeneous Autoregressive) models for realized volatility
- Comprehensive diagnostic tools and visualization utilities
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Set
import importlib
import warnings

# Set up module-level logger
logger = logging.getLogger("mfe.models.time_series")

# Import core components
try:
    from .base import TimeSeriesModel, TimeSeriesResult, TimeSeriesConfig
    from .arma import ARMA, ARMAX
    from .var import VAR
    from .har import HAR
    from .unit_root import ADF, KPSS
    from .filters import HPFilter, BKFilter, BeveridgeNelsonDecomposition
    from .estimation import estimate_arma, estimate_armax, estimate_var
    from .forecast import forecast, forecast_intervals
    from .diagnostics import (
        ljung_box_test, 
        jarque_bera_test, 
        arch_test, 
        residual_diagnostics
    )
    from .correlation import acf, pacf, cross_correlation
    from .impulse_response import impulse_response, cumulative_impulse_response
    from .causality import granger_causality
    from .plots import (
        plot_acf, 
        plot_pacf, 
        plot_forecast, 
        plot_diagnostics, 
        plot_impulse_response
    )
except ImportError as e:
    logger.error(f"Error importing time series components: {e}")
    raise ImportError(
        "Failed to import time series components. Please ensure the package "
        "is correctly installed. You can install it using: "
        "pip install mfe-toolbox"
    ) from e

# Version information
__version__ = "4.0.0"
__author__ = "Kevin Sheppard"

def _check_dependencies() -> None:
    """
    Check for required dependencies and their versions for time series models.
    
    Warns if dependencies are missing or if versions are incompatible.
    """
    required_packages = {
        "numpy": "1.26.0",
        "scipy": "1.11.3",
        "pandas": "2.1.1",
        "statsmodels": "0.14.0"
    }
    
    optional_packages = {
        "numba": "0.58.0",
        "matplotlib": "3.8.0"
    }
    
    missing_optional = []
    outdated_packages = []
    
    # Check required packages
    for package, min_version in required_packages.items():
        try:
            imported = importlib.import_module(package)
            if not hasattr(imported, "__version__"):
                logger.warning(f"Cannot determine version for {package}")
                continue
            
            pkg_version = imported.__version__
            if pkg_version.split(".") < min_version.split(".")):
                outdated_packages.append((package, pkg_version, min_version))
        except ImportError:
            # This should never happen as the main package already checks required dependencies
            logger.error(f"Required package missing: {package}")
    
    # Check optional packages
    for package, min_version in optional_packages.items():
        try:
            imported = importlib.import_module(package)
            if not hasattr(imported, "__version__"):
                logger.warning(f"Cannot determine version for {package}")
                continue
            
            pkg_version = imported.__version__
            if pkg_version.split(".") < min_version.split(".")):
                outdated_packages.append((package, pkg_version, min_version))
        except ImportError:
            missing_optional.append(package)
    
    # Warn about missing optional packages
    if missing_optional:
        if "numba" in missing_optional:
            warnings.warn(
                "Numba not found. Performance will be significantly reduced for "
                "computationally intensive time series operations. Install Numba with: pip install numba",
                ImportWarning
            )
        if "matplotlib" in missing_optional:
            warnings.warn(
                "Matplotlib not found. Plotting functions will not be available. "
                "Install Matplotlib with: pip install matplotlib",
                ImportWarning
            )
    
    # Warn about outdated packages
    if outdated_packages:
        for package, current, required in outdated_packages:
            warnings.warn(
                f"{package} version {current} is older than the recommended "
                f"version {required}. This may cause compatibility issues with some time series models.",
                UserWarning
            )

def list_time_series_models() -> Dict[str, List[str]]:
    """
    List all available time series models in the MFE Toolbox.
    
    Returns:
        Dict mapping model categories to lists of available model classes
    """
    return {
        "univariate": [
            "ARMA", "ARMAX", "HAR"
        ],
        "multivariate": [
            "VAR"
        ],
        "unit_root_tests": [
            "ADF", "KPSS"
        ],
        "filters": [
            "HPFilter", "BKFilter", "BeveridgeNelsonDecomposition"
        ],
        "analysis_tools": [
            "acf", "pacf", "cross_correlation", "impulse_response", 
            "granger_causality", "forecast", "forecast_intervals"
        ]
    }

# Initialize the time series module
_check_dependencies()

# Define what's available when using "from mfe.models.time_series import *"
__all__ = [
    # Base classes
    'TimeSeriesModel',
    'TimeSeriesResult',
    'TimeSeriesConfig',
    
    # Model classes
    'ARMA',
    'ARMAX',
    'VAR',
    'HAR',
    
    # Unit root tests
    'ADF',
    'KPSS',
    
    # Filters
    'HPFilter',
    'BKFilter',
    'BeveridgeNelsonDecomposition',
    
    # Estimation functions
    'estimate_arma',
    'estimate_armax',
    'estimate_var',
    
    # Forecasting functions
    'forecast',
    'forecast_intervals',
    
    # Diagnostic functions
    'ljung_box_test',
    'jarque_bera_test',
    'arch_test',
    'residual_diagnostics',
    
    # Correlation analysis
    'acf',
    'pacf',
    'cross_correlation',
    
    # Impulse response analysis
    'impulse_response',
    'cumulative_impulse_response',
    
    # Causality analysis
    'granger_causality',
    
    # Plotting functions
    'plot_acf',
    'plot_pacf',
    'plot_forecast',
    'plot_diagnostics',
    'plot_impulse_response',
    
    # Utility functions
    'list_time_series_models'
]

logger.debug("MFE Time Series module initialized successfully")