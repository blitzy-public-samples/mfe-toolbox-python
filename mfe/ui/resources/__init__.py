#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MFE Toolbox UI Resources Package

This module makes the resources directory a proper Python package and provides
resource lookup functionality for UI components. It enables the Python import
system to recognize the resources directory as a package and provides utility
functions to access visual resources like icons and logos.

The module implements cross-platform resource path resolution and provides
a consistent interface for accessing resources regardless of the operating
system or installation method.
"""

import os
import sys
import logging
import pathlib
from typing import Optional, Union, Dict, List, Any

# Configure logging
logger = logging.getLogger(__name__)

# Resource file mapping
RESOURCE_FILES = {
    # UI icons
    "oxford_logo.png": "Oxford University logo",
    "dialog_icon_info.png": "Information dialog icon",
    "dialog_icon_warning.png": "Warning dialog icon",
    "dialog_icon_error.png": "Error dialog icon",
    "dialog_icon_question.png": "Question dialog icon"
}

def get_resource_path(resource_name: str) -> pathlib.Path:
    """
    Get the absolute path to a resource file.
    
    This function resolves the path to a resource file in a cross-platform manner,
    handling different installation methods and directory structures.
    
    Args:
        resource_name: The name of the resource file (e.g., "oxford_logo.png")
        
    Returns:
        Path object representing the absolute path to the resource
        
    Raises:
        FileNotFoundError: If the resource cannot be found
    """
    # Check if the resource name is valid
    if resource_name not in RESOURCE_FILES:
        logger.warning(f"Unknown resource requested: {resource_name}")
    
    # First, try to find the resource in the current directory
    current_dir = pathlib.Path(__file__).parent
    resource_path = current_dir / resource_name
    
    if resource_path.exists():
        logger.debug(f"Resource found at {resource_path}")
        return resource_path
    
    # If not found, try parent directories
    parent_dir = current_dir.parent
    resource_path = parent_dir / "resources" / resource_name
    
    if resource_path.exists():
        logger.debug(f"Resource found at {resource_path}")
        return resource_path
    
    # Try package data directory (for installed packages)
    try:
        import importlib.resources
        with importlib.resources.path("mfe.ui.resources", resource_name) as path:
            if path.exists():
                logger.debug(f"Resource found at {path}")
                return path
    except (ImportError, ModuleNotFoundError, FileNotFoundError):
        # importlib.resources not available or resource not found
        pass
    
    # If still not found, try some common locations
    common_locations = [
        # Development layout
        pathlib.Path(__file__).parent.parent.parent / "ui" / "resources",
        # Installed package layout
        pathlib.Path(sys.prefix) / "lib" / "python" / "site-packages" / "mfe" / "ui" / "resources",
        # User's home directory
        pathlib.Path.home() / ".mfe" / "resources"
    ]
    
    for location in common_locations:
        resource_path = location / resource_name
        if resource_path.exists():
            logger.debug(f"Resource found at {resource_path}")
            return resource_path
    
    # If we get here, the resource was not found
    logger.error(f"Resource not found: {resource_name}")
    raise FileNotFoundError(f"Resource '{resource_name}' not found")


def list_available_resources() -> Dict[str, str]:
    """
    List all available resources with their descriptions.
    
    Returns:
        Dictionary mapping resource filenames to their descriptions
    """
    return RESOURCE_FILES.copy()


def get_resource_info(resource_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific resource.
    
    Args:
        resource_name: The name of the resource file
        
    Returns:
        Dictionary containing resource metadata
        
    Raises:
        ValueError: If the resource name is not recognized
    """
    if resource_name not in RESOURCE_FILES:
        raise ValueError(f"Unknown resource: {resource_name}")
    
    try:
        resource_path = get_resource_path(resource_name)
        
        # Get file information
        stat_info = resource_path.stat()
        
        return {
            "name": resource_name,
            "description": RESOURCE_FILES[resource_name],
            "path": str(resource_path),
            "size": stat_info.st_size,
            "modified": stat_info.st_mtime,
            "exists": True
        }
    except FileNotFoundError:
        return {
            "name": resource_name,
            "description": RESOURCE_FILES[resource_name],
            "path": None,
            "size": None,
            "modified": None,
            "exists": False
        }


def verify_resources() -> Dict[str, bool]:
    """
    Verify that all required resources are available.
    
    Returns:
        Dictionary mapping resource names to boolean indicating availability
    """
    results = {}
    
    for resource_name in RESOURCE_FILES:
        try:
            get_resource_path(resource_name)
            results[resource_name] = True
        except FileNotFoundError:
            results[resource_name] = False
    
    # Log missing resources
    missing = [name for name, available in results.items() if not available]
    if missing:
        logger.warning(f"Missing resources: {', '.join(missing)}")
    
    return results


# Initialize the resources package
logger.debug("MFE UI resources package initialized")
