"""
MFE Toolbox Test Suite

This package contains tests for the MFE Toolbox, a comprehensive Python-based suite
for financial econometrics, time series analysis, and risk modeling.

The test suite verifies the functionality, accuracy, and performance of the toolbox's
components, ensuring reliable results for financial and economic analyses.
"""

import os
import sys
import pytest
from typing import Any, Dict, List, Optional, Tuple, Union

# Version information for the test package
__version__ = "1.0.0"

# Import commonly used test utilities and fixtures
from tests.conftest import (
    # Basic data generation fixtures
    rng,
    sample_size,
    univariate_normal_data,
    univariate_t_data,
    univariate_skewed_t_data,
    multivariate_normal_data,
    time_series_with_trend,
    time_series_with_seasonality,
    ar1_process,
    ma1_process,
    arma11_process,
    garch11_process,
    egarch11_process,
    tarch11_process,
    dcc_process,
    high_frequency_price_data,
    cross_sectional_data,
    
    # Model parameter fixtures
    garch_params,
    egarch_params,
    tarch_params,
    aparch_params,
    dcc_params,
    arma_params,
    student_t_params,
    skewed_t_params,
    ged_params,
    
    # Mock objects
    mock_volatility_model,
    mock_multivariate_volatility_model,
    mock_time_series_model,
    
    # Temporary file and directory fixtures
    temp_dir,
    temp_file,
    
    # Hypothesis strategies
    garch_param_strategy,
    egarch_param_strategy,
    tarch_param_strategy,
    arma_param_strategy,
    distribution_param_strategy,
    
    # Assertion utilities
    assert_array_equal,
    assert_series_equal,
    assert_frame_equal,
    assert_parameters_equal,
    assert_model_results_equal,
)

# Define custom pytest markers for test categorization
pytest.mark.univariate = pytest.mark.univariate
pytest.mark.multivariate = pytest.mark.multivariate
pytest.mark.timeseries = pytest.mark.timeseries
pytest.mark.bootstrap = pytest.mark.bootstrap
pytest.mark.realized = pytest.mark.realized
pytest.mark.distributions = pytest.mark.distributions
pytest.mark.tests = pytest.mark.tests
pytest.mark.cross_section = pytest.mark.cross_section
pytest.mark.numba = pytest.mark.numba
pytest.mark.slow = pytest.mark.slow
pytest.mark.gui = pytest.mark.gui

# Environment detection for test configuration
def is_ci_environment() -> bool:
    """Check if tests are running in a CI environment."""
    return os.environ.get("CI", "false").lower() == "true"

def is_numba_available() -> bool:
    """Check if Numba is available for JIT compilation."""
    try:
        import numba
        return True
    except ImportError:
        return False

def is_gui_testable() -> bool:
    """Check if GUI tests can be run in the current environment."""
    try:
        from PyQt6.QtWidgets import QApplication
        return not is_ci_environment() or os.environ.get("DISPLAY", "") != ""
    except ImportError:
        return False

# Test configuration based on environment
SKIP_SLOW_TESTS = os.environ.get("SKIP_SLOW_TESTS", "false").lower() == "true"
SKIP_GUI_TESTS = os.environ.get("SKIP_GUI_TESTS", "false").lower() == "true" or not is_gui_testable()
USE_NUMBA = os.environ.get("USE_NUMBA", "true").lower() == "true" and is_numba_available()