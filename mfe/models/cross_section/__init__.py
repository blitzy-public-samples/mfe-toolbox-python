"""
MFE Toolbox Cross-Sectional Analysis Module

This module provides tools for cross-sectional econometric analysis, including
Ordinary Least Squares (OLS) regression and Principal Component Analysis (PCA).
These tools support fundamental economic and financial analyses across entities
at a point in time.

The cross-sectional analysis module integrates with the broader MFE Toolbox,
providing consistent interfaces and robust error handling while leveraging
NumPy's linear algebra functions and Statsmodels for core regression functionality.

Key components:
- OLS: Ordinary Least Squares regression with robust standard errors
- PCA: Principal Component Analysis with multiple modes and visualization
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple

# Set up module-level logger
logger = logging.getLogger("mfe.models.cross_section")

# Module version (follows main package version)
__version__ = "4.0.0"

# Import key classes to make them available at the module level
try:
    from .ols import OLS
    from .pca import PCA
    from .base import CrossSectionalModel
except ImportError as e:
    logger.error(f"Error importing cross-sectional analysis components: {e}")
    raise ImportError(
        "Failed to import cross-sectional analysis components. Please ensure the package "
        "is correctly installed. You can install it using: "
        "pip install mfe-toolbox"
    ) from e

# Import utility functions
try:
    from .utils import (
        compute_robust_errors,
        compute_white_errors,
        compute_newey_west_errors,
        standardize_data
    )
except ImportError as e:
    logger.warning(f"Some utility functions could not be imported: {e}")

# Define what's available when using "from mfe.models.cross_section import *"
__all__ = [
    # Main classes
    'OLS',
    'PCA',
    'CrossSectionalModel',
    
    # Utility functions
    'compute_robust_errors',
    'compute_white_errors',
    'compute_newey_west_errors',
    'standardize_data'
]

def list_models() -> Dict[str, str]:
    """
    List all available cross-sectional models with descriptions.
    
    Returns:
        Dict mapping model names to brief descriptions
    """
    return {
        "OLS": "Ordinary Least Squares regression with robust standard errors",
        "PCA": "Principal Component Analysis with multiple modes and visualization"
    }

logger.debug("MFE Cross-Sectional Analysis module initialized successfully")
