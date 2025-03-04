#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
About dialog for the ARMAX interface.

This module implements the About dialog for the ARMAX interface using PyQt6's QDialog.
It displays application information, version details, and credits. The dialog handles
modal dialog behavior, keyboard shortcuts, and proper window positioning.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QSpacerItem, QFrame
)
from PyQt6.QtCore import Qt, QSize, QEvent, pyqtSlot
from PyQt6.QtGui import QPixmap, QKeyEvent, QFont, QIcon

import mfe
from mfe.version import get_version_info

# Configure logging
logger = logging.getLogger(__name__)


class AboutDialog(QDialog):
    """
    About dialog for the ARMAX interface.
    
    This dialog displays application information, version details, and credits.
    It implements proper modal dialog behavior with PyQt6 signals and slots,
    keyboard shortcuts for Escape and Return keys, and screen-aware positioning.
    
    Attributes:
        parent: The parent widget
        logo_pixmap: The Oxford logo pixmap
        version_info: Dictionary containing version information
    """
    
    def __init__(self, parent=None):
        """
        Initialize the About dialog.
        
        Args:
            parent: The parent widget
        """
        super().__init__(parent)
        self.parent = parent
        self.logo_pixmap: Optional[QPixmap] = None
        self.version_info: Dict[str, Any] = get_version_info()
        
        # Set up dialog properties
        self.setWindowTitle("About ARMAX GUI")
        self.setMinimumWidth(400)
        self.setMinimumHeight(300)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowCloseButtonHint)
        self.setModal(True)
        
        # Initialize UI components
        self._init_ui()
        
        # Load resources asynchronously
        asyncio.create_task(self._load_resources_async())
        
        logger.debug("About dialog initialized")
    
    def _init_ui(self) -> None:
        """Initialize the UI components."""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Logo placeholder
        self.logo_label = QLabel(self)
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.logo_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.logo_label.setMinimumSize(QSize(200, 100))
        self.logo_label.setMaximumSize(QSize(200, 100))
        
        # Center the logo
        logo_layout = QHBoxLayout()
        logo_layout.addStretch()
        logo_layout.addWidget(self.logo_label)
        logo_layout.addStretch()
        layout.addLayout(logo_layout)
        
        # Add separator
        separator = QFrame(self)
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)
        
        # Application title
        title_label = QLabel("ARMAX GUI Interface", self)
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Version information
        version_label = QLabel(f"Version {self.version_info['version']}", self)
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version_label)
        
        # Part of MFE Toolbox
        toolbox_label = QLabel("Part of the MFE Toolbox", self)
        toolbox_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(toolbox_label)
        
        # Copyright information
        copyright_label = QLabel(f"© 2023 {self.version_info['author']}", self)
        copyright_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(copyright_label)
        
        # University affiliation
        affiliation_label = QLabel("University of Oxford", self)
        affiliation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(affiliation_label)
        
        # Add spacer
        layout.addItem(QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        
        # Close button
        close_button = QPushButton("Close", self)
        close_button.setDefault(True)
        close_button.clicked.connect(self.accept)
        
        # Center the button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(close_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Set the layout
        self.setLayout(layout)
    
    async def _load_resources_async(self) -> None:
        """
        Load resources asynchronously to keep the UI responsive.
        
        This method loads the Oxford logo from resources without blocking the UI.
        """
        try:
            # Simulate resource loading (in a real implementation, this would load from a file)
            # In a production environment, we would use the resource_loader module
            await asyncio.sleep(0.1)  # Small delay to demonstrate async loading
            
            # Try to load the logo from the resources directory
            resource_path = Path(__file__).parent / "resources" / "oxford_logo.png"
            
            if resource_path.exists():
                self.logo_pixmap = QPixmap(str(resource_path))
            else:
                # Fallback: Try to load from a different location
                alt_path = Path(__file__).parent.parent / "ui" / "resources" / "oxford_logo.png"
                if alt_path.exists():
                    self.logo_pixmap = QPixmap(str(alt_path))
                else:
                    logger.warning(f"Oxford logo not found at {resource_path} or {alt_path}")
                    # Create a placeholder pixmap
                    self.logo_pixmap = QPixmap(200, 100)
                    self.logo_pixmap.fill(Qt.GlobalColor.white)
            
            # Update the logo label
            if self.logo_pixmap and not self.logo_pixmap.isNull():
                self.logo_label.setPixmap(self.logo_pixmap.scaled(
                    200, 100, 
                    Qt.AspectRatioMode.KeepAspectRatio, 
                    Qt.TransformationMode.SmoothTransformation
                ))
            
            logger.debug("Resources loaded successfully")
        except Exception as e:
            logger.error(f"Error loading resources: {e}")
    
    def keyPressEvent(self, event: QKeyEvent) -> None:
        """
        Handle key press events.
        
        Args:
            event: The key event
        """
        # Close dialog on Escape key
        if event.key() == Qt.Key.Key_Escape:
            self.reject()
        # Accept dialog on Return/Enter key
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.accept()
        else:
            super().keyPressEvent(event)
    
    def showEvent(self, event: QEvent) -> None:
        """
        Handle show events.
        
        This method centers the dialog on the parent window when shown.
        
        Args:
            event: The show event
        """
        super().showEvent(event)
        
        # Center the dialog on the parent window
        if self.parent:
            parent_geo = self.parent.geometry()
            size = self.geometry().size()
            x = parent_geo.x() + (parent_geo.width() - size.width()) // 2
            y = parent_geo.y() + (parent_geo.height() - size.height()) // 2
            self.move(x, y)
        
        logger.debug("About dialog shown")
    
    @classmethod
    def show_dialog(cls, parent=None) -> int:
        """
        Create and show the About dialog.
        
        This is a convenience method for creating and showing the dialog.
        
        Args:
            parent: The parent widget
            
        Returns:
            int: Dialog result code (QDialog.Accepted or QDialog.Rejected)
        """
        dialog = cls(parent)
        result = dialog.exec()
        return result


if __name__ == "__main__":
    # Test the dialog if run directly
    import sys
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    result = AboutDialog.show_dialog()
    sys.exit(0)
