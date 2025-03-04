#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Close confirmation dialog for the ARMAX interface.

This module implements a confirmation dialog that appears when the user attempts
to close the ARMAX application. It ensures users don't accidentally lose their work
by providing a modal confirmation prompt with options to continue or cancel the
close operation.
"""

import logging
from typing import Optional, Union, Tuple, cast

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QMessageBox, QSizePolicy, QSpacerItem
)
from PyQt6.QtCore import Qt, QSize, QEvent, pyqtSlot
from PyQt6.QtGui import QPixmap, QKeyEvent, QIcon

# Configure logging
logger = logging.getLogger(__name__)


class CloseDialog(QDialog):
    """
    Close confirmation dialog for the ARMAX interface.
    
    This dialog prompts the user to confirm before closing the ARMAX application,
    preventing accidental data loss. It implements proper modal dialog behavior
    with PyQt6 signals and slots, keyboard shortcuts for Escape and Return keys,
    and screen-aware positioning.
    
    Attributes:
        parent: The parent widget
    """
    
    def __init__(self, parent=None):
        """
        Initialize the close confirmation dialog.
        
        Args:
            parent: The parent widget
        """
        super().__init__(parent)
        self.parent = parent
        
        # Set up dialog properties
        self.setWindowTitle("Confirm Close")
        self.setMinimumWidth(350)
        self.setMinimumHeight(150)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowCloseButtonHint)
        self.setModal(True)
        
        # Initialize UI components
        self._init_ui()
        
        logger.debug("Close confirmation dialog initialized")
    
    def _init_ui(self) -> None:
        """Initialize the UI components."""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Create message layout with icon and text
        message_layout = QHBoxLayout()
        
        # Warning icon
        icon_label = QLabel(self)
        icon = QMessageBox.standardIcon(QMessageBox.Icon.Warning)
        icon_label.setPixmap(icon.pixmap(QSize(32, 32)))
        icon_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        message_layout.addWidget(icon_label)
        
        # Message text
        message_layout.addSpacing(10)
        message_label = QLabel(
            "Are you sure you want to close the ARMAX GUI?\n\n"
            "All unsaved changes will be lost.",
            self
        )
        message_label.setWordWrap(True)
        message_layout.addWidget(message_label)
        
        layout.addLayout(message_layout)
        
        # Add spacer
        layout.addItem(QSpacerItem(20, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        
        # Buttons layout
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        # Cancel button
        cancel_button = QPushButton("Cancel", self)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        # Close button
        close_button = QPushButton("Close", self)
        close_button.setDefault(True)
        close_button.clicked.connect(self.accept)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
        
        # Set the layout
        self.setLayout(layout)
    
    def keyPressEvent(self, event: QKeyEvent) -> None:
        """
        Handle key press events.
        
        Args:
            event: The key event
        """
        # Cancel on Escape key
        if event.key() == Qt.Key.Key_Escape:
            self.reject()
        # Confirm on Return/Enter key
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
        
        logger.debug("Close confirmation dialog shown")
    
    @classmethod
    async def confirm_close_async(cls, parent=None) -> bool:
        """
        Show the close confirmation dialog asynchronously.
        
        This method creates and shows the dialog, returning a boolean
        indicating whether the user confirmed the close operation.
        
        Args:
            parent: The parent widget
            
        Returns:
            bool: True if the user confirmed closing, False otherwise
        """
        dialog = cls(parent)
        result = dialog.exec()
        return result == QDialog.DialogCode.Accepted
    
    @classmethod
    def confirm_close(cls, parent=None) -> bool:
        """
        Show the close confirmation dialog.
        
        This is a synchronous convenience method for creating and showing the dialog.
        
        Args:
            parent: The parent widget
            
        Returns:
            bool: True if the user confirmed closing, False otherwise
        """
        dialog = cls(parent)
        result = dialog.exec()
        return result == QDialog.DialogCode.Accepted



def show_close_confirmation(parent=None) -> bool:
    """
    Show the close confirmation dialog.
    
    This is a convenience function for showing the close confirmation dialog.
    
    Args:
        parent: The parent widget
        
    Returns:
        bool: True if the user confirmed closing, False otherwise
    """
    return CloseDialog.confirm_close(parent)


if __name__ == "__main__":
    # Test the dialog if run directly
    import sys
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    result = show_close_confirmation()
    print(f"User {'confirmed' if result else 'cancelled'} closing")
    sys.exit(0)
