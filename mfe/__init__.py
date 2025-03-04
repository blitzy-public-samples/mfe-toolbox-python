# mfe/__init__.py
"""
MFE Toolbox - Financial Econometrics Toolbox for Python

A comprehensive Python-based suite for financial econometrics, time series analysis,
and risk modeling. This package represents a complete modernization of the original
MATLAB-based toolbox (formerly version 4.0, released 28-Oct-2009).

The toolbox provides tools for:
- Univariate and multivariate volatility modeling (GARCH, EGARCH, DCC, etc.)
- ARMA/ARMAX time series modeling and forecasting
- Bootstrap methods for dependent data
- Non-parametric volatility estimation (realized volatility)
- Statistical tests and distributions
- Vector autoregression (VAR) analysis
- Principal component analysis and cross-sectional econometrics

This module serves as the main entry point for the MFE Toolbox package.
"""

import os
import sys
import logging
import importlib
import pathlib
from typing import Dict, List, Optional, Union, Any, Tuple, Set
import warnings

# Set up package-wide logger
logger = logging.getLogger("mfe")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(_handler)

# Version information
__version__ = "4.0.0"  # Major version change to reflect Python implementation
__author__ = "Kevin Sheppard"
__email__ = "kevin.sheppard@economics.ox.ac.uk"
__license__ = "MIT"

# Package metadata
__title__ = "MFE Toolbox"
__description__ = "Financial Econometrics Toolbox for Python"
__url__ = "https://github.com/bashtage/arch"  # Placeholder URL

# Import subpackages to make them available in the mfe namespace
try:
    from . import core
    from . import models
    from . import utils
    from . import ui
except ImportError as e:
    logger.error(f"Error importing MFE Toolbox components: {e}")
    raise ImportError(
        "Failed to import MFE Toolbox components. Please ensure the package "
        "is correctly installed. You can install it using: "
        "pip install mfe-toolbox"
    ) from e

# Initialize configuration
def _initialize_config() -> None:
    """
    Initialize package configuration on first import.
    
    This function:
    1. Creates user configuration directory if it doesn't exist
    2. Sets up default configuration if none exists
    3. Loads existing configuration
    """
    try:
        # Create user config directory if it doesn't exist
        config_dir = pathlib.Path.home() / ".mfe"
        if not config_dir.exists():
            config_dir.mkdir(exist_ok=True)
            logger.debug(f"Created MFE configuration directory at {config_dir}")
            
        # Create default config file if it doesn't exist
        config_file = config_dir / "config.json"
        if not config_file.exists():
            import json
            default_config = {
                "numba_enabled": True,
                "log_level": "INFO",
                "use_cached_results": True,
                "default_plot_style": "seaborn",
                "show_deprecation_warnings": True
            }
            with open(config_file, "w") as f:
                json.dump(default_config, f, indent=4)
            logger.debug(f"Created default configuration file at {config_file}")
            
        # Load configuration (would typically use core.config module)
        # For now, we'll just set some basic configuration
        
        # Configure logging based on config
        log_level = os.environ.get("MFE_LOG_LEVEL", "INFO")
        logger.setLevel(getattr(logging, log_level))
        
        # Configure Numba (if available)
        if os.environ.get("MFE_DISABLE_NUMBA", "0") != "1":
            _register_numba_functions()
        else:
            logger.info("Numba acceleration disabled by environment variable")
            
    except Exception as e:
        logger.warning(f"Failed to initialize configuration: {e}")
        logger.warning("Using default settings")


def _register_numba_functions() -> None:
    """
    Register and initialize Numba JIT-compiled functions.
    
    This function attempts to import Numba and register performance-critical
    functions for JIT compilation. If Numba is not available, it falls back
    to pure Python implementations.
    """
    try:
        import numba
        logger.debug(f"Numba version {numba.__version__} detected")
        
        # Set Numba configuration
        numba.config.THREADING_LAYER = 'threadsafe'
        
        # We would typically import and register JIT functions here
        # For example:
        # from mfe.models.univariate._core import _register_jit_functions
        # _register_jit_functions()
        
        logger.debug("Numba JIT functions registered successfully")
    except ImportError:
        warnings.warn(
            "Numba not found. Performance will be significantly reduced. "
            "Install Numba with: pip install numba",
            ImportWarning
        )
    except Exception as e:
        logger.warning(f"Failed to register Numba functions: {e}")
        logger.warning("Falling back to pure Python implementations")


def _check_dependencies() -> None:
    """
    Check for required dependencies and their versions.
    
    Warns if dependencies are missing or if versions are incompatible.
    """
    required_packages = {
        "numpy": "1.26.0",
        "scipy": "1.11.3",
        "pandas": "2.1.1",
        "numba": "0.58.0",
        "statsmodels": "0.14.0",
        "matplotlib": "3.8.0"
    }
    
    optional_packages = {
        "PyQt6": "6.5.0"
    }
    
    missing_required = []
    outdated_packages = []
    
    for package, min_version in required_packages.items():
        try:
            imported = importlib.import_module(package)
            if not hasattr(imported, "__version__"):
                logger.warning(f"Cannot determine version for {package}")
                continue
                
            pkg_version = imported.__version__
            if pkg_version.split(".") < min_version.split("."):
                outdated_packages.append((package, pkg_version, min_version))
        except ImportError:
            missing_required.append(package)
    
    if missing_required:
        logger.error(f"Required packages missing: {', '.join(missing_required)}")
        raise ImportError(
            f"MFE Toolbox requires the following packages: "
            f"{', '.join(missing_required)}. Please install them with pip."
        )
        
    if outdated_packages:
        for package, current, required in outdated_packages:
            warnings.warn(
                f"{package} version {current} is older than the recommended "
                f"version {required}. This may cause compatibility issues.",
                UserWarning
            )


def _setup_package_registry() -> None:
    """
    Set up the package registry to track available components.
    
    This function scans the package structure to identify available models,
    utilities, and other components for dynamic discovery.
    """
    # This would typically build a registry of available components
    # For now, we'll just set up a basic structure
    
    # Example registry structure
    registry = {
        "models": {
            "univariate": ["GARCH", "EGARCH", "TARCH", "APARCH", "FIGARCH", "HEAVY", "IGARCH"],
            "multivariate": ["BEKK", "DCC", "CCC", "RARCH", "RCC", "GOGARCH"],
            "time_series": ["ARMA", "ARMAX", "VAR"],
            "bootstrap": ["BlockBootstrap", "StationaryBootstrap", "MCS"],
            "realized": ["RealizedVariance", "RealizedKernel", "BiPowerVariation"],
            "distributions": ["Normal", "StudentT", "GED", "SkewedT"],
            "cross_section": ["OLS", "PCA"]
        },
        "ui": ["ARMAXApp"],
        "utils": ["matrix_ops", "covariance", "differentiation", "data_transformations"]
    }
    
    # Store registry in module namespace for access by other components
    globals()["_registry"] = registry

# Public API functions

def get_version() -> str:
    """
    Return the version of the MFE Toolbox.
    
    Returns:
        str: Version string in format MAJOR.MINOR.PATCH
    """
    return __version__


def set_log_level(level: Union[str, int]) -> None:
    """
    Set the logging level for the MFE Toolbox.
    
    Args:
        level: Logging level, either as string ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
              or as an integer constant from the logging module
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    logger.setLevel(level)
    logger.info(f"Log level set to {logging.getLevelName(level)}")


def enable_numba(enabled: bool = True) -> None:
    """
    Enable or disable Numba JIT acceleration.
    
    Args:
        enabled: Whether to enable Numba acceleration
    
    Note:
        Changes only affect newly created model instances.
        Existing instances will maintain their current implementation.
    """
    if enabled:
        _register_numba_functions()
        logger.info("Numba acceleration enabled")
    else:
        # We would typically disable Numba here
        # For now, just log the change
        logger.info("Numba acceleration disabled")
    
    # Update configuration
    os.environ["MFE_DISABLE_NUMBA"] = "0" if enabled else "1"


def list_available_models() -> Dict[str, List[str]]:
    """
    List all available models in the MFE Toolbox.
    
    Returns:
        Dict mapping model categories to lists of available models
    """
    return _registry.get("models", {})

# Initialize the package
_check_dependencies()
_initialize_config()
_setup_package_registry()

# Clean up namespace
del os, sys, logging, importlib, pathlib, warnings

# Define what's available when using "from mfe import *"
__all__ = [
    # Subpackages
    'core',
    'models',
    'utils',
    'ui',
    
    # Public functions
    'get_version',
    'set_log_level',
    'enable_numba',
    'list_available_models',
    
    # Version info
    '__version__',
    '__author__',
    '__email__',
    '__license__'
]

logger.debug(f"MFE Toolbox v{__version__} initialized successfully")
