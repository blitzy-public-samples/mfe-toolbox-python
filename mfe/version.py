# mfe/version.py
"""
MFE Toolbox Version Information

This module contains version information, metadata, and release history for the MFE Toolbox.
It centralizes version tracking, making it accessible programmatically via mfe.__version__
and providing historical version information.

The MFE Toolbox follows semantic versioning (MAJOR.MINOR.PATCH):
- MAJOR: Incompatible API changes
- MINOR: Backwards-compatible functionality additions
- PATCH: Backwards-compatible bug fixes
"""

from typing import Dict, Tuple, Any, Optional, List

# Version components
VERSION_MAJOR = 4
VERSION_MINOR = 0
VERSION_PATCH = 0

# Full version string
__version__ = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_PATCH}"

# Package metadata
__title__ = "MFE Toolbox"
__description__ = "Financial Econometrics Toolbox for Python"
__author__ = "Kevin Sheppard"
__email__ = "kevin.sheppard@economics.ox.ac.uk"
__license__ = "MIT"
__copyright__ = "Copyright 2023 Kevin Sheppard"
__url__ = "https://github.com/bashtage/arch"  # Placeholder URL

# Python version requirements
__python_requires__ = ">=3.12"

# Package dependencies
__dependencies__ = {
    "numpy": ">=1.26.0",
    "scipy": ">=1.11.3",
    "pandas": ">=2.1.1",
    "numba": ">=0.58.0",
    "statsmodels": ">=0.14.0",
    "matplotlib": ">=3.8.0",
    "PyQt6": ">=6.5.0"
}

# Version history with release dates and major changes
VERSION_HISTORY = [
    {
        "version": "4.0.0",
        "release_date": "2023-11-15",
        "changes": [
            "Complete rewrite in Python 3.12 from original MATLAB codebase",
            "Implemented class-based architecture with inheritance hierarchies",
            "Added comprehensive type hints and dataclasses for parameter containers",
            "Replaced MEX files with Numba-accelerated functions using @jit decorators",
            "Integrated with Python scientific ecosystem (NumPy, SciPy, Pandas, Statsmodels)",
            "Added asynchronous processing support for long-duration computations",
            "Implemented PyQt6-based ARMAX modeling environment",
            "Modernized package structure following Python conventions"
        ]
    },
    {
        "version": "3.0.0",  # Historical MATLAB version
        "release_date": "2009-10-28",
        "changes": [
            "Added multivariate GARCH models (BEKK, DCC, RARCH)",
            "Expanded realized volatility estimators",
            "Added bootstrap methods for dependent data",
            "Improved MEX file performance for core functions",
            "Enhanced ARMAX modeling capabilities"
        ]
    },
    {
        "version": "2.0.0",  # Historical MATLAB version
        "release_date": "2008-07-15",
        "changes": [
            "Added univariate GARCH models",
            "Implemented ARMA/ARMAX modeling",
            "Added basic realized volatility estimators",
            "Introduced MEX file acceleration for performance-critical functions"
        ]
    },
    {
        "version": "1.0.0",  # Historical MATLAB version
        "release_date": "2007-05-10",
        "changes": [
            "Initial release of MFE Toolbox",
            "Basic time series analysis functions",
            "Statistical distributions and tests",
            "Utility functions for financial data analysis"
        ]
    }
]


def get_version_info() -> Dict[str, Any]:
    """
    Get detailed version information about the MFE Toolbox.
    
    Returns:
        Dict containing version information, including version string,
        version components, release date, and recent changes.
    """
    current_version = VERSION_HISTORY[0]
    
    return {
        "version": __version__,
        "major": VERSION_MAJOR,
        "minor": VERSION_MINOR,
        "patch": VERSION_PATCH,
        "release_date": current_version["release_date"],
        "changes": current_version["changes"],
        "python_requires": __python_requires__,
        "dependencies": __dependencies__,
        "author": __author__,
        "license": __license__
    }


def get_version_components() -> Tuple[int, int, int]:
    """
    Get the version components as a tuple.
    
    Returns:
        Tuple of (major, minor, patch) version components
    """
    return (VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH)


def is_compatible_with(version: str) -> bool:
    """
    Check if the current version is compatible with the specified version.
    
    Args:
        version: Version string to check compatibility with
    
    Returns:
        True if the current version is compatible with the specified version,
        False otherwise. Compatibility is determined by the semantic versioning
        rules (same major version, equal or higher minor/patch version).
    """
    try:
        # Parse version string
        parts = version.split(".")
        major = int(parts[0])
        minor = int(parts[1] if len(parts) > 1 else 0)
        patch = int(parts[2] if len(parts) > 2 else 0)
        
        # Check compatibility (same major version, equal or higher minor/patch)
        if VERSION_MAJOR != major:
            return False
        
        if VERSION_MINOR < minor:
            return False
            
        if VERSION_MINOR == minor and VERSION_PATCH < patch:
            return False
            
        return True
    except (ValueError, IndexError):
        # If version string is invalid, assume incompatible
        return False


def get_release_notes(version: Optional[str] = None) -> Dict[str, Any]:
    """
    Get release notes for a specific version.
    
    Args:
        version: Version string to get release notes for.
                If None, returns notes for the current version.
    
    Returns:
        Dict containing release notes for the specified version,
        including release date and changes.
    
    Raises:
        ValueError: If the specified version is not found in the version history.
    """
    if version is None:
        version = __version__
        
    for release in VERSION_HISTORY:
        if release["version"] == version:
            return {
                "version": release["version"],
                "release_date": release["release_date"],
                "changes": release["changes"]
            }
            
    raise ValueError(f"Version {version} not found in version history")


def list_all_versions() -> List[Dict[str, Any]]:
    """
    List all versions in the version history.
    
    Returns:
        List of dicts containing version, release date, and summary for each version
    """
    return [
        {
            "version": release["version"],
            "release_date": release["release_date"],
            "summary": release["changes"][0]  # First change as summary
        }
        for release in VERSION_HISTORY
    ]


# Make version components available at module level
major = VERSION_MAJOR
minor = VERSION_MINOR
patch = VERSION_PATCH
