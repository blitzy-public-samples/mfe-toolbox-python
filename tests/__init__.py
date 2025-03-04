"""
MFE Toolbox Test Suite

This module initializes the test package for the MFE Toolbox, providing
common utilities, fixtures, and configuration for testing the Python-based
financial econometrics library.

The test suite uses pytest as the primary testing framework, with additional
support for property-based testing via hypothesis, and specialized fixtures
for generating financial time series data.
"""

import os
import sys
from pathlib import Path
import pytest

# Test package version
__version__ = "4.0.0"

# Add parent directory to path if running tests directly
parent_dir = Path(__file__).parent.parent
if parent_dir not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import common pytest markers for easier access in test modules
from pytest import mark

# Common test markers
slow = mark.slow
integration = mark.integration
gpu = mark.gpu
numba = mark.numba
async_test = mark.async_test
property_test = mark.property
benchmark = mark.benchmark

# Import commonly used fixtures for convenience
from tests.conftest import (
    rng,
    temp_dir,
    temp_file,
    random_univariate_series,
    random_multivariate_series,
    garch_process,
    ar1_process,
    ma1_process,
    arma11_process,
    heavy_tailed_series,
    skewed_series,
    high_frequency_data,
    multivariate_correlated_series,
    dcc_process,
    mock_garch_model,
    mock_arma_model,
    array_strategy,
    time_series_strategy,
    parametrize_models,
    parametrize_distributions,
    assert_array_equal,
    assert_series_equal,
    assert_frame_equal
)

# Environment detection for test configuration
def is_ci_environment() -> bool:
    """
    Detect if tests are running in a CI environment.
    
    Returns:
        bool: True if running in a CI environment, False otherwise
    """
    return any(os.environ.get(var) for var in [
        'CI', 'GITHUB_ACTIONS', 'TRAVIS', 'CIRCLECI', 'APPVEYOR', 'GITLAB_CI'
    ])

def has_numba_support() -> bool:
    """
    Check if Numba is available and working correctly.
    
    Returns:
        bool: True if Numba is available and working, False otherwise
    """
    try:
        import numba
        return True
    except ImportError:
        return False

def has_gpu_support() -> bool:
    """
    Check if GPU acceleration is available through Numba.
    
    Returns:
        bool: True if GPU acceleration is available, False otherwise
    """
    try:
        import numba.cuda
        return numba.cuda.is_available()
    except (ImportError, AttributeError):
        return False

def has_async_support() -> bool:
    """
    Check if Python version supports async/await syntax.
    
    Returns:
        bool: True if async/await is supported, False otherwise
    """
    return sys.version_info >= (3, 7)

# Configure test environment based on capabilities
TEST_CONFIG = {
    'ci_environment': is_ci_environment(),
    'numba_support': has_numba_support(),
    'gpu_support': has_gpu_support(),
    'async_support': has_async_support(),
    'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
}
