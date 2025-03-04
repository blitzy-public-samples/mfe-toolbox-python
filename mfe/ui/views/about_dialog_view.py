# mfe/ui/views/about_dialog_view.py
"""
About Dialog View

This module implements the About dialog view for the MFE Toolbox UI using PyQt6.
It displays application information, version details, and the Oxford University logo.
The dialog provides users with important information about the software's authorship
and version.
"""

import logging
from typing import Optional, Dict, Any, Union, List, Callable
import asyncio
from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QScrollArea, QWidget, QSizePolicy
)
from PyQt6.QtGui import QPixmap, QFont, QIcon
from PyQt6.QtCore import Qt, QSize, pyqtSignal, pyqtSlot

# Set up module-level logger
logger = logging.getLogger("mfe.ui.views.about_dialog")


class AboutDialogView(QDialog):
    """
    About dialog view for the MFE Toolbox.
    
    This dialog displays information about the MFE Toolbox, including version,
    author, and organization details. It uses a modern PyQt6 layout with the
    Oxford University logo and properly formatted text.
    
    Signals:
        closed: Emitted when the dialog is closed
    """
    
    # Signal emitted when the dialog is closed
    closed = pyqtSignal()
    
    def __init__(self, parent=None):
        """
        Initialize the About dialog view.
        
        Args:
            parent: Parent widget, if any
        """
        super().__init__(parent)
        
        # Initialize instance variables
        self._logo_label: Optional[QLabel] = None
        self._version_label: Optional[QLabel] = None
        self._description_label: Optional[QLabel] = None
        self._author_label: Optional[QLabel] = None
        self._organization_label: Optional[QLabel] = None
        self._copyright_label: Optional[QLabel] = None
        self._close_button: Optional[QPushButton] = None
        
        # Set up the UI
        self._setup_ui()
        
        logger.debug("AboutDialogView initialized")
    
    def _setup_ui(self) -> None:
        """
        Set up the user interface components.
        
        This method:
        1. Creates and configures the dialog window
        2. Sets up the layout with logo and information
        3. Adds the close button
        """
        # Configure dialog properties
        self.setWindowTitle("About ARMAX GUI")
        self.setMinimumWidth(400)
        self.setMinimumHeight(300)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowCloseButtonHint)
        self.setModal(True)
        
        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Create logo label
        self._logo_label = QLabel()
        self._logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._logo_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._logo_label.setMinimumSize(QSize(150, 150))
        self._logo_label.setMaximumSize(QSize(150, 150))
        self._logo_label.setScaledContents(True)
        
        # Center the logo
        logo_layout = QHBoxLayout()
        logo_layout.addStretch()
        logo_layout.addWidget(self._logo_label)
        logo_layout.addStretch()
        main_layout.addLayout(logo_layout)
        
        # Add application name with larger font
        app_name_label = QLabel("ARMAX GUI Interface")
        app_name_font = QFont()
        app_name_font.setPointSize(14)
        app_name_font.setBold(True)
        app_name_label.setFont(app_name_font)
        app_name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(app_name_label)
        
        # Add version information
        self._version_label = QLabel()
        self._version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self._version_label)
        
        # Add description
        self._description_label = QLabel()
        self._description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._description_label.setWordWrap(True)
        main_layout.addWidget(self._description_label)
        
        # Add separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        main_layout.addWidget(separator)
        
        # Add author information
        self._author_label = QLabel()
        self._author_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self._author_label)
        
        # Add organization information
        self._organization_label = QLabel()
        self._organization_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self._organization_label)
        
        # Add copyright information
        self._copyright_label = QLabel()
        self._copyright_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self._copyright_label)
        
        # Add spacer
        main_layout.addStretch()
        
        # Add close button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self._close_button = QPushButton("Close")
        self._close_button.setDefault(True)
        self._close_button.setAutoDefault(True)
        self._close_button.clicked.connect(self.accept)
        button_layout.addWidget(self._close_button)
        
        main_layout.addLayout(button_layout)
        
        # Set the layout
        self.setLayout(main_layout)
        
        logger.debug("AboutDialogView UI setup complete")
    
    def set_content(self, content: Dict[str, Union[str, List[str]]]) -> None:
        """
        Set the content of the dialog.
        
        Args:
            content: Dictionary containing dialog content including title, version,
                    description, author, organization, copyright, and logo_path
        """
        # Set window title
        if "title" in content:
            self.setWindowTitle(content["title"])
        
        # Set version
        if "version" in content:
            self._version_label.setText(content["version"])
        
        # Set description
        if "description" in content:
            self._description_label.setText(content["description"])
        
        # Set author
        if "author" in content:
            self._author_label.setText(content["author"])
        
        # Set organization
        if "organization" in content:
            self._organization_label.setText(content["organization"])
        
        # Set copyright
        if "copyright" in content:
            self._copyright_label.setText(content["copyright"])
        
        logger.debug("AboutDialogView content set")
    
    async def load_logo_async(self, logo_path: str) -> None:
        """
        Asynchronously load the logo image.
        
        Args:
            logo_path: Path to the logo image file
        """
        try:
            # Load the logo image asynchronously
            # In a real implementation, this would use a resource loader
            # For now, we'll just create a QPixmap directly
            pixmap = QPixmap(logo_path)
            
            if pixmap.isNull():
                logger.warning(f"Failed to load logo from {logo_path}")
                return
            
            # Set the logo image
            self._logo_label.setPixmap(pixmap)
            logger.debug(f"Logo loaded from {logo_path}")
            
        except Exception as e:
            logger.error(f"Error loading logo: {e}")
    
    def closeEvent(self, event):
        """
        Handle the dialog close event.
        
        Args:
            event: Close event
        """
        # Emit the closed signal
        self.closed.emit()
        
        # Accept the event to close the dialog
        event.accept()
        
        logger.debug("AboutDialogView closed")
    
    @pyqtSlot()
    def accept(self) -> None:
        """
        Handle the dialog accept event (e.g., when Close button is clicked).
        """
        # Emit the closed signal
        self.closed.emit()
        
        # Call the parent class implementation to close the dialog
        super().accept()
        
        logger.debug("AboutDialogView accepted")
    
    @pyqtSlot()
    def reject(self) -> None:
        """
        Handle the dialog reject event (e.g., when Escape key is pressed).
        """
        # Emit the closed signal
        self.closed.emit()
        
        # Call the parent class implementation to close the dialog
        super().reject()
        
        logger.debug("AboutDialogView rejected")
