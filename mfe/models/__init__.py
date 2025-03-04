# mfe/models/__init__.py
"""
MFE Toolbox Models Module

This module provides a comprehensive collection of financial econometric models
including univariate and multivariate volatility models, time series models,
bootstrap methods, realized volatility estimators, statistical distributions,
and cross-sectional analysis tools.

The models module serves as the central component of the MFE Toolbox, providing
implementations of state-of-the-art econometric methods for financial time series
analysis and risk modeling.
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Set
import importlib
import warnings

# Set up module-level logger
logger = logging.getLogger("mfe.models")

# Import submodules to make them available at the package level
try:
    from . import univariate
    from . import multivariate
    from . import time_series
    from . import bootstrap
    from . import realized
    from . import distributions
    from . import cross_section
except ImportError as e:
    logger.error(f"Error importing model components: {e}")
    raise ImportError(
        "Failed to import model components. Please ensure the package "
        "is correctly installed. You can install it using: "
        "pip install mfe-toolbox"
    ) from e

# Import key univariate volatility models
try:
    from .univariate.garch import GARCH
    from .univariate.egarch import EGARCH
    from .univariate.tarch import TARCH
    from .univariate.aparch import APARCH
    from .univariate.figarch import FIGARCH
    from .univariate.heavy import HEAVY
    from .univariate.igarch import IGARCH
    from .univariate.agarch import AGARCH
except ImportError as e:
    logger.warning(f"Some univariate volatility models could not be imported: {e}")

# Import key multivariate volatility models
try:
    from .multivariate.bekk import BEKK
    from .multivariate.dcc import DCC
    from .multivariate.ccc import CCC
    from .multivariate.rarch import RARCH
    from .multivariate.rcc import RCC
    from .multivariate.gogarch import GOGARCH
    from .multivariate.matrix_garch import MatrixGARCH
    from .multivariate.riskmetrics import RiskMetrics
except ImportError as e:
    logger.warning(f"Some multivariate volatility models could not be imported: {e}")

# Import key time series models
try:
    from .time_series.arma import ARMA, ARMAX
    from .time_series.var import VAR
    from .time_series.har import HAR
    from .time_series.unit_root import ADF, KPSS
    from .time_series.filters import HPFilter, BKFilter
except ImportError as e:
    logger.warning(f"Some time series models could not be imported: {e}")

# Import bootstrap methods
try:
    from .bootstrap.block_bootstrap import BlockBootstrap
    from .bootstrap.stationary_bootstrap import StationaryBootstrap
    from .bootstrap.mcs import ModelConfidenceSet
    from .bootstrap.bsds import BSDS
except ImportError as e:
    logger.warning(f"Some bootstrap methods could not be imported: {e}")

# Import realized volatility estimators
try:
    from .realized.variance import RealizedVariance
    from .realized.kernel import RealizedKernel
    from .realized.bipower_variation import BiPowerVariation
    from .realized.semivariance import RealizedSemivariance
    from .realized.covariance import RealizedCovariance
except ImportError as e:
    logger.warning(f"Some realized volatility estimators could not be imported: {e}")

# Import statistical distributions
try:
    from .distributions.normal import Normal
    from .distributions.student_t import StudentT
    from .distributions.generalized_error import GED
    from .distributions.skewed_t import SkewedT
except ImportError as e:
    logger.warning(f"Some statistical distributions could not be imported: {e}")

# Import cross-sectional analysis tools
try:
    from .cross_section.ols import OLS
    from .cross_section.pca import PCA
except ImportError as e:
    logger.warning(f"Some cross-sectional analysis tools could not be imported: {e}")

def _check_dependencies() -> None:
    """
    Check for required dependencies and their versions for models.
    
    Warns if dependencies are missing or if versions are incompatible.
    """
    required_packages = {
        "numpy": "1.26.0",
        "scipy": "1.11.3",
        "pandas": "2.1.1",
        "statsmodels": "0.14.0"
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
            if pkg_version.split(".") < min_version.split("."):
                outdated_packages.append((package, pkg_version, min_version))
        except ImportError:
            missing_optional.append(package)
    
    # Warn about missing optional packages
    if missing_optional:
        if "numba" in missing_optional:
            warnings.warn(
                "Numba not found. Performance will be significantly reduced for "
                "computationally intensive models. Install Numba with: pip install numba",
                ImportWarning
            )
    
    # Warn about outdated packages
    if outdated_packages:
        for package, current, required in outdated_packages:
            warnings.warn(
                f"{package} version {current} is older than the recommended "
                f"version {required}. This may cause compatibility issues with some models.",
                UserWarning
            )

def list_models() -> Dict[str, List[str]]:
    """
    List all available models in the MFE Toolbox.
    
    Returns:
        Dict mapping model categories to lists of available model classes
    """
    return {
        "univariate": [
            "GARCH", "EGARCH", "TARCH", "APARCH", "FIGARCH", "HEAVY", "IGARCH", "AGARCH"
        ],
        "multivariate": [
            "BEKK", "DCC", "CCC", "RARCH", "RCC", "GOGARCH", "MatrixGARCH", "RiskMetrics"
        ],
        "time_series": [
            "ARMA", "ARMAX", "VAR", "HAR", "ADF", "KPSS", "HPFilter", "BKFilter"
        ],
        "bootstrap": [
            "BlockBootstrap", "StationaryBootstrap", "ModelConfidenceSet", "BSDS"
        ],
        "realized": [
            "RealizedVariance", "RealizedKernel", "BiPowerVariation", 
            "RealizedSemivariance", "RealizedCovariance"
        ],
        "distributions": [
            "Normal", "StudentT", "GED", "SkewedT"
        ],
        "cross_section": [
            "OLS", "PCA"
        ]
    }

# Initialize the models module
_check_dependencies()

# Define what's available when using "from mfe.models import *"
__all__ = [
    # Submodules
    'univariate',
    'multivariate',
    'time_series',
    'bootstrap',
    'realized',
    'distributions',
    'cross_section',
    
    # Univariate volatility models
    'GARCH',
    'EGARCH',
    'TARCH',
    'APARCH',
    'FIGARCH',
    'HEAVY',
    'IGARCH',
    'AGARCH',
    
    # Multivariate volatility models
    'BEKK',
    'DCC',
    'CCC',
    'RARCH',
    'RCC',
    'GOGARCH',
    'MatrixGARCH',
    'RiskMetrics',
    
    # Time series models
    'ARMA',
    'ARMAX',
    'VAR',
    'HAR',
    'ADF',
    'KPSS',
    'HPFilter',
    'BKFilter',
    
    # Bootstrap methods
    'BlockBootstrap',
    'StationaryBootstrap',
    'ModelConfidenceSet',
    'BSDS',
    
    # Realized volatility estimators
    'RealizedVariance',
    'RealizedKernel',
    'BiPowerVariation',
    'RealizedSemivariance',
    'RealizedCovariance',
    
    # Statistical distributions
    'Normal',
    'StudentT',
    'GED',
    'SkewedT',
    
    # Cross-sectional analysis tools
    'OLS',
    'PCA',
    
    # Utility functions
    'list_models'
]

logger.debug("MFE Models module initialized successfully")
