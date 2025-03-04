"""
Multivariate Volatility Models

This module provides a comprehensive collection of multivariate volatility models
for analyzing and forecasting time-varying covariance matrices in financial time series.
These models are essential for portfolio risk management, asset allocation, and
understanding cross-asset dependencies.

The module implements state-of-the-art multivariate GARCH-family models including:
- BEKK: Baba-Engle-Kraft-Kroner model for direct covariance matrix modeling
- DCC: Dynamic Conditional Correlation for separating volatility and correlation dynamics
- CCC: Constant Conditional Correlation as a simpler alternative to DCC
- OGARCH/GOGARCH: (Generalized) Orthogonal GARCH for dimension reduction
- RARCH: Rotated ARCH for improved numerical stability
- RCC: Rotated Conditional Correlation combining rotation with correlation modeling
- Matrix GARCH: Direct matrix-based volatility specifications
- RiskMetrics: Industry-standard exponentially weighted moving average approach

All models follow a consistent object-oriented design with proper type hints,
 dataclass-based parameter containers, and Numba-accelerated core functions
for optimal performance.
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Set
import importlib
import warnings

# Set up module-level logger
logger = logging.getLogger("mfe.models.multivariate")

# Import version information
try:
    from ...version import __version__, get_version_info
except ImportError:
    __version__ = "4.0.0"
    logger.warning("Could not import version information")

# Import base classes and utilities
try:
    from .base import (
        MultivariateVolatilityModel,
        MultivariateVolatilityResult,
        covariance_to_correlation,
        correlation_to_covariance
    )
except ImportError as e:
    logger.error(f"Error importing multivariate base classes: {e}")
    raise ImportError(
        "Failed to import multivariate base classes. Please ensure the package "
        "is correctly installed. You can install it using: "
        "pip install mfe-toolbox"
    ) from e

# Import model implementations
try:
    from .bekk import BEKK
    from .dcc import DCC
    from .ccc import CCC
    from .gogarch import GOGARCH, OGARCH
    from .rarch import RARCH
    from .rcc import RCC
    from .matrix_garch import MatrixGARCH
    from .riskmetrics import RiskMetrics
    from .scalar_vt_vech import ScalarVTVECH
except ImportError as e:
    logger.warning(f"Some multivariate volatility models could not be imported: {e}")

# Import utility functions
try:
    from .utils import (
        estimate_constant_correlation,
        check_covariance_positive_definite,
        ensure_symmetric_matrix,
        compute_portfolio_variance
    )
except ImportError as e:
    logger.warning(f"Some utility functions could not be imported: {e}")

# Import Numba-accelerated core functions
try:
    from ._numba_core import (
        bekk_recursion,
        dcc_recursion,
        garch_recursion_matrix,
        compute_correlation_matrix
    )
except ImportError as e:
    logger.debug(f"Numba-accelerated functions could not be imported: {e}")
    logger.debug("Will use pure Python implementations")


def _check_dependencies() -> None:
    """
    Check for required dependencies and their versions for multivariate models.
    
    Warns if dependencies are missing or if versions are incompatible.
    """
    required_packages = {
        "numpy": "1.26.0",
        "scipy": "1.11.3",
        "pandas": "2.1.1"
    }
    
    optional_packages = {
        "numba": "0.58.0"
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
            if pkg_version.split(".") < min_version.split("."):
                outdated_packages.append((package, pkg_version, min_version))
        except ImportError:
            logger.error(f"Required package missing: {package}")
    
    # Check optional packages
    for package, min_version in optional_packages.items():
        try:
            imported = importlib.import_module(package)
            if not hasattr(imported, "__version__"):
                logger.warning(f"Cannot determine version for {package}")
                continue
                
            pkg_version = imported.__version__
            if pkg_version.split(".") < min_version.split("."):
                outdated_packages.append((package, pkg_version, min_version))
        except ImportError:
            missing_optional.append(package)
    
    # Warn about missing optional packages
    if missing_optional:
        if "numba" in missing_optional:
            warnings.warn(
                "Numba not found. Performance will be significantly reduced for "
                "multivariate volatility models. Install Numba with: pip install numba",
                ImportWarning
            )
    
    # Warn about outdated packages
    if outdated_packages:
        for package, current, required in outdated_packages:
            warnings.warn(
                f"{package} version {current} is older than the recommended "
                f"version {required}. This may cause compatibility issues with "
                f"multivariate volatility models.",
                UserWarning
            )


def list_models() -> Dict[str, str]:
    """
    List all available multivariate volatility models with descriptions.
    
    Returns:
        Dict mapping model names to brief descriptions
    """
    return {
        "BEKK": "Baba-Engle-Kraft-Kroner model for direct covariance matrix modeling",
        "DCC": "Dynamic Conditional Correlation for separating volatility and correlation dynamics",
        "CCC": "Constant Conditional Correlation as a simpler alternative to DCC",
        "GOGARCH": "Generalized Orthogonal GARCH for dimension reduction",
        "OGARCH": "Orthogonal GARCH for dimension reduction",
        "RARCH": "Rotated ARCH for improved numerical stability",
        "RCC": "Rotated Conditional Correlation combining rotation with correlation modeling",
        "MatrixGARCH": "Direct matrix-based volatility specifications",
        "RiskMetrics": "Industry-standard exponentially weighted moving average approach",
        "ScalarVTVECH": "Scalar Variance Targeting VECH model"
    }

# Initialize the multivariate models module
_check_dependencies()

# Module metadata
__author__ = "Kevin Sheppard"
__license__ = "MIT"
__copyright__ = "Copyright 2023 Kevin Sheppard"

# Define what's available when using "from mfe.models.multivariate import *"
__all__ = [
    # Base classes
    'MultivariateVolatilityModel',
    'MultivariateVolatilityResult',
    
    # Model implementations
    'BEKK',
    'DCC',
    'CCC',
    'GOGARCH',
    'OGARCH',
    'RARCH',
    'RCC',
    'MatrixGARCH',
    'RiskMetrics',
    'ScalarVTVECH',
    
    # Utility functions
    'covariance_to_correlation',
    'correlation_to_covariance',
    'estimate_constant_correlation',
    'check_covariance_positive_definite',
    'ensure_symmetric_matrix',
    'compute_portfolio_variance',
    
    # Module functions
    'list_models'
]

logger.debug("MFE Multivariate Volatility Models module initialized successfully")