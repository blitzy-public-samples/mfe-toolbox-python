#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resource Loader for MFE Toolbox UI Components

This module provides functions to load and manage UI resources in Python. It handles
resource path resolution, scaling for different display densities, and caching for
performance. This module replaces the MATLAB's direct loading of .mat files with a
more flexible and platform-independent approach.

The module implements cross-platform resource path resolution and provides a consistent
interface for accessing resources regardless of the operating system or installation
method. It also supports high-DPI display scaling for visual resources.
"""

import os
import sys
import logging
import pathlib
from typing import Optional, Union, Dict, List, Any, Tuple, Set, Callable
import functools
import importlib.resources
import importlib.util
import time
import weakref

from PyQt6.QtCore import QSize, QStandardPaths, QDir, Qt
from PyQt6.QtGui import QPixmap, QImage, QIcon, QPainter, QColor, QFont
from PyQt6.QtWidgets import QApplication

# Configure logging
logger = logging.getLogger(__name__)

# Resource file mapping with descriptions
RESOURCE_FILES = {
    # UI icons
    "oxford_logo.png": "Oxford University logo",
    "dialog_icon_info.png": "Information dialog icon",
    "dialog_icon_warning.png": "Warning dialog icon",
    "dialog_icon_error.png": "Error dialog icon",
    "dialog_icon_question.png": "Question dialog icon"
}

# Cache for loaded resources to improve performance
_resource_cache: Dict[str, Dict[str, Any]] = {
    "pixmaps": {},
    "icons": {},
    "paths": {},
    "metadata": {}
}

# Cache expiration time in seconds (1 hour)
_CACHE_EXPIRATION = 3600

# Last cache cleanup time
_last_cache_cleanup = time.time()


def _cleanup_cache() -> None:
    """
    Clean up expired items from the resource cache.
    
    This function removes items from the cache that haven't been accessed
    for longer than the cache expiration time.
    """
    global _last_cache_cleanup
    
    # Only clean up once per hour
    current_time = time.time()
    if current_time - _last_cache_cleanup < 3600:
        return
    
    # Update cleanup time
    _last_cache_cleanup = current_time
    
    # Clean up each cache type
    for cache_type in _resource_cache:
        expired_keys = []
        for key, item in _resource_cache[cache_type].items():
            if isinstance(item, dict) and "last_access" in item:
                if current_time - item["last_access"] > _CACHE_EXPIRATION:
                    expired_keys.append(key)
        
        # Remove expired items
        for key in expired_keys:
            del _resource_cache[cache_type][key]
    
    logger.debug(f"Resource cache cleaned up, removed {len(expired_keys)} expired items")


def get_resource_path(resource_name: str) -> pathlib.Path:
    """
    Get the absolute path to a resource file.
    
    This function resolves the path to a resource file in a cross-platform manner,
    handling different installation methods and directory structures. It implements
    a sophisticated search algorithm that looks in multiple locations to find the
    requested resource.
    
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
    
    # Check if the path is already cached
    if resource_name in _resource_cache["paths"]:
        cache_entry = _resource_cache["paths"][resource_name]
        # Update last access time
        cache_entry["last_access"] = time.time()
        return cache_entry["path"]
    
    # First, try to find the resource in the current directory
    current_dir = pathlib.Path(__file__).parent.resolve()
    resource_path = current_dir / resource_name
    
    if resource_path.exists():
        logger.debug(f"Resource found at {resource_path}")
        # Cache the path
        _resource_cache["paths"][resource_name] = {
            "path": resource_path,
            "last_access": time.time()
        }
        return resource_path
    
    # If not found, try parent directories
    parent_dir = current_dir.parent
    resource_path = parent_dir / "resources" / resource_name
    
    if resource_path.exists():
        logger.debug(f"Resource found at {resource_path}")
        # Cache the path
        _resource_cache["paths"][resource_name] = {
            "path": resource_path,
            "last_access": time.time()
        }
        return resource_path
    
    # Try package data directory (for installed packages)
    try:
        with importlib.resources.path("mfe.ui.resources", resource_name) as path:
            if path.exists():
                logger.debug(f"Resource found at {path}")
                # Cache the path
                _resource_cache["paths"][resource_name] = {
                    "path": path,
                    "last_access": time.time()
                }
                return path
    except (ImportError, ModuleNotFoundError, FileNotFoundError):
        # importlib.resources not available or resource not found
        pass
    
    # Try some common locations
    common_locations = [
        # Development layout
        current_dir.parent.parent / "ui" / "resources",
        # Installed package layout
        pathlib.Path(sys.prefix) / "lib" / "python" / "site-packages" / "mfe" / "ui" / "resources",
        # User's home directory
        pathlib.Path.home() / ".mfe" / "resources",
        # Application data directory
        pathlib.Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)) / "mfe" / "resources",
        # Current working directory
        pathlib.Path.cwd() / "resources",
        # Executable directory (for frozen applications)
        pathlib.Path(sys.executable).parent / "resources"
    ]
    
    for location in common_locations:
        resource_path = location / resource_name
        if resource_path.exists():
            logger.debug(f"Resource found at {resource_path}")
            # Cache the path
            _resource_cache["paths"][resource_name] = {
                "path": resource_path,
                "last_access": time.time()
            }
            return resource_path
    
    # If we get here, the resource was not found
    logger.error(f"Resource not found: {resource_name}")
    raise FileNotFoundError(f"Resource '{resource_name}' not found")


def load_pixmap(resource_name: str, scale_factor: float = 1.0) -> QPixmap:
    """
    Load a pixmap resource with optional scaling.
    
    This function loads a pixmap resource from the resources directory and
    optionally scales it for high-DPI displays. It implements caching to
    improve performance for frequently accessed resources.
    
    Args:
        resource_name: The name of the resource file
        scale_factor: Optional scaling factor for high-DPI displays
        
    Returns:
        QPixmap object containing the loaded image
        
    Raises:
        FileNotFoundError: If the resource cannot be found
    """
    # Generate cache key based on resource name and scale factor
    cache_key = f"{resource_name}_{scale_factor}"
    
    # Check if the pixmap is already cached
    if cache_key in _resource_cache["pixmaps"]:
        cache_entry = _resource_cache["pixmaps"][cache_key]
        # Update last access time
        cache_entry["last_access"] = time.time()
        return cache_entry["pixmap"]
    
    # Get the resource path
    resource_path = get_resource_path(resource_name)
    
    # Load the pixmap
    pixmap = QPixmap(str(resource_path))
    
    if pixmap.isNull():
        logger.error(f"Failed to load pixmap: {resource_name}")
        # Create a placeholder pixmap
        pixmap = QPixmap(100, 100)
        pixmap.fill(QColor(200, 200, 200))
        
        # Draw an error indicator
        painter = QPainter(pixmap)
        painter.setPen(QColor(255, 0, 0))
        painter.setFont(QFont("Arial", 10))
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "Error")
        painter.end()
    
    # Scale the pixmap if needed
    if scale_factor != 1.0 and not pixmap.isNull():
        original_size = pixmap.size()
        scaled_size = QSize(
            int(original_size.width() * scale_factor),
            int(original_size.height() * scale_factor)
        )
        pixmap = pixmap.scaled(
            scaled_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
    
    # Cache the pixmap
    _resource_cache["pixmaps"][cache_key] = {
        "pixmap": pixmap,
        "last_access": time.time()
    }
    
    # Clean up cache if needed
    _cleanup_cache()
    
    return pixmap


def load_icon(resource_name: str) -> QIcon:
    """
    Load an icon resource.
    
    This function loads an icon resource from the resources directory.
    It implements caching to improve performance for frequently accessed resources.
    
    Args:
        resource_name: The name of the resource file
        
    Returns:
        QIcon object containing the loaded icon
        
    Raises:
        FileNotFoundError: If the resource cannot be found
    """
    # Check if the icon is already cached
    if resource_name in _resource_cache["icons"]:
        cache_entry = _resource_cache["icons"][resource_name]
        # Update last access time
        cache_entry["last_access"] = time.time()
        return cache_entry["icon"]
    
    # Load the pixmap
    pixmap = load_pixmap(resource_name)
    
    # Create the icon
    icon = QIcon(pixmap)
    
    # Cache the icon
    _resource_cache["icons"][resource_name] = {
        "icon": icon,
        "last_access": time.time()
    }
    
    return icon


def get_resource_info(resource_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific resource.
    
    This function retrieves metadata about a resource file, including its
    description, path, size, and modification time.
    
    Args:
        resource_name: The name of the resource file
        
    Returns:
        Dictionary containing resource metadata
        
    Raises:
        ValueError: If the resource name is not recognized
    """
    # Check if the resource name is valid
    if resource_name not in RESOURCE_FILES:
        raise ValueError(f"Unknown resource: {resource_name}")
    
    # Check if the metadata is already cached
    if resource_name in _resource_cache["metadata"]:
        cache_entry = _resource_cache["metadata"][resource_name]
        # Update last access time
        cache_entry["last_access"] = time.time()
        return cache_entry["info"].copy()
    
    try:
        # Get the resource path
        resource_path = get_resource_path(resource_name)
        
        # Get file information
        stat_info = resource_path.stat()
        
        # Create metadata dictionary
        info = {
            "name": resource_name,
            "description": RESOURCE_FILES[resource_name],
            "path": str(resource_path),
            "size": stat_info.st_size,
            "modified": stat_info.st_mtime,
            "exists": True
        }
        
        # Cache the metadata
        _resource_cache["metadata"][resource_name] = {
            "info": info.copy(),
            "last_access": time.time()
        }
        
        return info
    except FileNotFoundError:
        # Resource not found
        info = {
            "name": resource_name,
            "description": RESOURCE_FILES[resource_name],
            "path": None,
            "size": None,
            "modified": None,
            "exists": False
        }
        
        # Cache the metadata
        _resource_cache["metadata"][resource_name] = {
            "info": info.copy(),
            "last_access": time.time()
        }
        
        return info


def list_available_resources() -> Dict[str, str]:
    """
    List all available resources with their descriptions.
    
    Returns:
        Dictionary mapping resource filenames to their descriptions
    """
    return RESOURCE_FILES.copy()


def verify_resources() -> Dict[str, bool]:
    """
    Verify that all required resources are available.
    
    This function checks if all registered resources can be found in the
    resources directory or any of the alternative locations.
    
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


def get_optimal_scale_factor() -> float:
    """
    Get the optimal scale factor for the current display.
    
    This function determines the appropriate scaling factor for resources
    based on the device pixel ratio of the current display. It ensures that
    resources look crisp on high-DPI displays.
    
    Returns:
        Scaling factor as a float (1.0 for standard displays, higher for high-DPI)
    """
    # Get the application instance
    app = QApplication.instance()
    if app is None:
        # No application instance, assume standard DPI
        return 1.0
    
    # Get the primary screen
    screen = app.primaryScreen()
    if screen is None:
        # No screen available, assume standard DPI
        return 1.0
    
    # Get the device pixel ratio
    device_pixel_ratio = screen.devicePixelRatio()
    
    # Round to nearest 0.5 to avoid unnecessary scaling
    return round(device_pixel_ratio * 2) / 2


def create_colored_icon(
    base_icon_name: str,
    color: Union[QColor, str, Tuple[int, int, int], Tuple[int, int, int, int]],
    size: Optional[QSize] = None
) -> QIcon:
    """
    Create a colored version of an icon.
    
    This function loads an icon resource and applies a color tint to it,
    creating a new colored version. This is useful for creating icons in
    different colors from a single base icon.
    
    Args:
        base_icon_name: The name of the base icon resource
        color: The color to apply (QColor, color name, or RGB/RGBA tuple)
        size: Optional size for the icon
        
    Returns:
        QIcon object containing the colored icon
        
    Raises:
        FileNotFoundError: If the base icon cannot be found
    """
    # Convert color to QColor if needed
    if not isinstance(color, QColor):
        if isinstance(color, str):
            # Color name
            qcolor = QColor(color)
        elif isinstance(color, tuple):
            # RGB or RGBA tuple
            if len(color) == 3:
                qcolor = QColor(color[0], color[1], color[2])
            elif len(color) == 4:
                qcolor = QColor(color[0], color[1], color[2], color[3])
            else:
                raise ValueError("Color tuple must have 3 (RGB) or 4 (RGBA) components")
        else:
            raise TypeError("Color must be QColor, color name, or RGB/RGBA tuple")
    else:
        qcolor = color
    
    # Generate cache key
    cache_key = f"{base_icon_name}_{qcolor.name()}_{size.width() if size else 0}_{size.height() if size else 0}"
    
    # Check if the colored icon is already cached
    if cache_key in _resource_cache["icons"]:
        cache_entry = _resource_cache["icons"][cache_key]
        # Update last access time
        cache_entry["last_access"] = time.time()
        return cache_entry["icon"]
    
    # Load the base pixmap
    base_pixmap = load_pixmap(base_icon_name)
    
    # Create a new pixmap with the desired size
    if size is not None:
        pixmap = QPixmap(size)
    else:
        pixmap = QPixmap(base_pixmap.size())
    
    # Fill with transparent background
    pixmap.fill(Qt.GlobalColor.transparent)
    
    # Create painter
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    
    # Draw the base pixmap
    if size is not None:
        painter.drawPixmap(
            pixmap.rect(),
            base_pixmap,
            base_pixmap.rect()
        )
    else:
        painter.drawPixmap(0, 0, base_pixmap)
    
    # Apply color tint
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    painter.fillRect(pixmap.rect(), qcolor)
    
    # End painting
    painter.end()
    
    # Create icon from the colored pixmap
    icon = QIcon(pixmap)
    
    # Cache the colored icon
    _resource_cache["icons"][cache_key] = {
        "icon": icon,
        "last_access": time.time()
    }
    
    return icon


def clear_cache() -> None:
    """
    Clear the resource cache.
    
    This function removes all cached resources, forcing them to be reloaded
    the next time they are requested. This can be useful when resources have
    been updated or when memory usage needs to be reduced.
    """
    global _resource_cache
    
    # Clear each cache type
    for cache_type in _resource_cache:
        _resource_cache[cache_type].clear()
    
    logger.debug("Resource cache cleared")


def register_custom_resource(resource_name: str, description: str) -> None:
    """
    Register a custom resource.
    
    This function adds a custom resource to the resource registry, making it
    available for loading with the standard resource loading functions.
    
    Args:
        resource_name: The name of the resource file
        description: A description of the resource
    """
    RESOURCE_FILES[resource_name] = description
    logger.debug(f"Custom resource registered: {resource_name}")


def get_resource_dir() -> pathlib.Path:
    """
    Get the directory where resources are stored.
    
    This function returns the directory where resources are stored, which can
    be useful for operations that need to access multiple resources or scan
    the resources directory.
    
    Returns:
        Path object representing the resources directory
    """
    # Try to find the resources directory
    try:
        # Get the path to any known resource
        for resource_name in RESOURCE_FILES:
            try:
                resource_path = get_resource_path(resource_name)
                return resource_path.parent
            except FileNotFoundError:
                continue
        
        # If no resources were found, return the default location
        return pathlib.Path(__file__).parent.resolve()
    except Exception as e:
        logger.error(f"Error getting resource directory: {e}")
        return pathlib.Path(__file__).parent.resolve()


def load_pixmap_with_fallback(
    resource_name: str,
    fallback_resource: Optional[str] = None,
    scale_factor: float = 1.0
) -> QPixmap:
    """
    Load a pixmap resource with a fallback option.
    
    This function attempts to load a pixmap resource, and if that fails,
    it loads a fallback resource instead. This is useful for ensuring that
    a valid pixmap is always returned, even if the primary resource is missing.
    
    Args:
        resource_name: The name of the primary resource file
        fallback_resource: The name of the fallback resource file, or None to use a placeholder
        scale_factor: Optional scaling factor for high-DPI displays
        
    Returns:
        QPixmap object containing the loaded image
    """
    try:
        # Try to load the primary resource
        return load_pixmap(resource_name, scale_factor)
    except FileNotFoundError:
        # If the primary resource is not found, try the fallback
        if fallback_resource is not None:
            try:
                logger.warning(f"Resource {resource_name} not found, using fallback {fallback_resource}")
                return load_pixmap(fallback_resource, scale_factor)
            except FileNotFoundError:
                pass
        
        # If both resources are not found, create a placeholder
        logger.error(f"Resource {resource_name} and fallback not found, using placeholder")
        pixmap = QPixmap(100, 100)
        pixmap.fill(QColor(200, 200, 200))
        
        # Draw an error indicator
        painter = QPainter(pixmap)
        painter.setPen(QColor(255, 0, 0))
        painter.setFont(QFont("Arial", 10))
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "Missing")
        painter.end()
        
        return pixmap


# Initialize the module
logger.debug("MFE UI resource loader initialized")
