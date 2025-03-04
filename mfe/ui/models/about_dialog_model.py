"""
About Dialog Model

This module implements the data model for the About dialog in the MFE Toolbox UI.
It manages the static content displayed in the About dialog, including version
information, credits, and other metadata, providing a clean separation of UI
content from presentation logic.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import asyncio

# Import version information
from mfe.version import (
    __version__, __author__, __email__, __license__, __copyright__,
    __description__, get_version_info
)

# Set up module-level logger
logger = logging.getLogger("mfe.ui.models.about_dialog")


@dataclass
class AboutDialogModel:
    """
    Data model for the About dialog.
    
    This class manages the content displayed in the About dialog, including
    version information, credits, and other metadata. It provides a clean
    separation of UI content from presentation logic.
    
    Attributes:
        title: The title of the dialog
        version: The version of the MFE Toolbox
        description: A brief description of the MFE Toolbox
        author: The author of the MFE Toolbox
        email: The contact email for the MFE Toolbox
        license: The license under which the MFE Toolbox is distributed
        copyright: The copyright notice for the MFE Toolbox
        organization: The organization associated with the MFE Toolbox
        logo_path: Path to the logo image file
        acknowledgments: List of acknowledgments to display
        website: The website URL for the MFE Toolbox
        release_date: The release date of the current version
    """
    
    # Dialog title
    title: str = "About MFE Toolbox"
    
    # Version information
    version: str = __version__
    
    # Basic information
    description: str = __description__
    author: str = __author__
    email: str = __email__
    license: str = __license__
    copyright: str = __copyright__
    
    # Organization information
    organization: str = "University of Oxford"
    
    # Resources
    logo_path: str = field(default="oxford_logo.png")
    
    # Additional information
    acknowledgments: List[str] = field(default_factory=lambda: [
        "Original MATLAB MFE Toolbox by Kevin Sheppard",
        "Python implementation contributors",
        "NumPy, SciPy, Pandas, and Numba developers",
        "PyQt6 development team"
    ])
    
    # Links
    website: str = "https://github.com/bashtage/arch"  # Placeholder URL
    
    # Release information
    release_date: str = field(default_factory=lambda: get_version_info()["release_date"])
    
    # Private fields for internal use
    _version_details: Dict[str, Any] = field(default_factory=get_version_info, repr=False)
    
    def __post_init__(self) -> None:
        """
        Initialize the model after dataclass initialization.
        
        This method:
        1. Validates the model data
        2. Sets up any derived properties
        3. Logs initialization
        """
        logger.debug(f"Initializing AboutDialogModel with version {self.version}")
        
        # Ensure logo_path is properly formatted
        if not self.logo_path.startswith((":", "/", "./")):
            # Assume it's a relative path within the resources directory
            self.logo_path = str(Path(self.logo_path))
    
    def get_full_version_info(self) -> Dict[str, Any]:
        """
        Get detailed version information.
        
        Returns:
            Dict containing detailed version information including version components,
            release date, and recent changes.
        """
        return self._version_details
    
    def get_version_string(self) -> str:
        """
        Get a formatted version string.
        
        Returns:
            Formatted version string (e.g., "Version 4.0.0")
        """
        return f"Version {self.version}"
    
    def get_copyright_string(self) -> str:
        """
        Get a formatted copyright string.
        
        Returns:
            Formatted copyright string
        """
        return self.copyright
    
    def get_author_string(self) -> str:
        """
        Get a formatted author string.
        
        Returns:
            Formatted author string including email if available
        """
        if self.email:
            return f"{self.author} <{self.email}>"
        return self.author
    
    def get_acknowledgments_text(self) -> str:
        """
        Get formatted acknowledgments text.
        
        Returns:
            Acknowledgments as a multi-line string
        """
        return "\n".join(self.acknowledgments)
    
    async def load_logo_async(self) -> Optional[bytes]:
        """
        Asynchronously load the logo image data.
        
        This method demonstrates async capability for resource loading,
        though in practice logo loading is typically fast enough to be synchronous.
        
        Returns:
            Image data as bytes, or None if loading fails
        """
        try:
            # Simulate async loading (in a real implementation, this would load from disk or network)
            await asyncio.sleep(0.01)  # Minimal sleep to demonstrate async nature
            
            # In a real implementation, this would load the actual image data
            # For now, we just return None to indicate the path should be used directly
            logger.debug(f"Asynchronously loaded logo from {self.logo_path}")
            return None
        except Exception as e:
            logger.error(f"Error loading logo: {e}")
            return None
    
    def get_release_notes(self) -> List[str]:
        """
        Get release notes for the current version.
        
        Returns:
            List of changes in the current version
        """
        return self._version_details.get("changes", [])
    
    def get_dialog_content(self) -> Dict[str, Union[str, List[str]]]:
        """
        Get all dialog content as a dictionary.
        
        This method provides a convenient way to access all dialog content
        in a format that can be easily used by the view.
        
        Returns:
            Dictionary containing all dialog content
        """
        return {
            "title": self.title,
            "version": self.get_version_string(),
            "description": self.description,
            "author": self.get_author_string(),
            "organization": self.organization,
            "copyright": self.get_copyright_string(),
            "license": self.license,
            "logo_path": self.logo_path,
            "acknowledgments": self.acknowledgments,
            "website": self.website,
            "release_date": self.release_date,
            "release_notes": self.get_release_notes()
        }
