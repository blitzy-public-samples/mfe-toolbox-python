# mfe/ui/views/close_dialog_view.py
"""
Close Dialog View

This module implements the confirmation dialog view for the MFE Toolbox UI using PyQt6.
It presents a warning message asking users to confirm closing the application when
unsaved changes exist, preventing accidental data loss.
"""

import logging
from typing import Optional, Dict, Any, Union, Callable

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QMessageBox, QDialogButtonBox, QSizePolicy
)
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot

from mfe.ui.models.close_dialog_model import DialogIcon

# Set up module-level logger
logger = logging.getLogger("mfe.ui.views.close_dialog")


class CloseDialogView(QDialog):
    """
    Confirmation dialog view for the MFE Toolbox.
    
    This dialog asks users to confirm closing the application when unsaved changes exist.
    It uses PyQt6's QDialog with standard warning icons and properly positioned buttons.
    
    Signals:
        confirmed: Emitted when the user confirms closing
        canceled: Emitted when the user cancels closing
    """
    
    # Signals emitted based on user actions
    confirmed = pyqtSignal()
    canceled = pyqtSignal()
    
    def __init__(self, parent=None):
        """
        Initialize the close confirmation dialog view.
        
        Args:
            parent: Parent widget, if any
        """
        super().__init__(parent)
        
        # Initialize instance variables
        self._icon_label: Optional[QLabel] = None
        self._message_label: Optional[QLabel] = None
        self._detail_label: Optional[QLabel] = None
        self._button_box: Optional[QDialogButtonBox] = None
        self._confirm_button: Optional[QPushButton] = None
        self._cancel_button: Optional[QPushButton] = None
        
        # Set up the UI
        self._setup_ui()
        
        logger.debug("CloseDialogView initialized")
    
    def _setup_ui(self) -> None:
        """
        Set up the user interface components.
        
        This method:
        1. Creates and configures the dialog window
        2. Sets up the layout with warning icon and message
        3. Adds the confirmation and cancel buttons
        """
        # Configure dialog properties
        self.setWindowTitle("Confirm Close")
        self.setMinimumWidth(400)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowCloseButtonHint)
        self.setModal(True)
        
        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Create content layout (icon + message)
        content_layout = QHBoxLayout()
        
        # Create icon label
        self._icon_label = QLabel()
        self._icon_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._icon_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._icon_label.setMinimumSize(32, 32)
        self._icon_label.setMaximumSize(32, 32)
        content_layout.addWidget(self._icon_label)
        
        # Add spacing between icon and text
        content_layout.addSpacing(15)
        
        # Create message layout (vertical)
        message_layout = QVBoxLayout()
        
        # Create message label
        self._message_label = QLabel()
        self._message_label.setWordWrap(True)
        self._message_label.setTextFormat(Qt.TextFormat.RichText)
        message_layout.addWidget(self._message_label)
        
        # Create detail label
        self._detail_label = QLabel()
        self._detail_label.setWordWrap(True)
        self._detail_label.setTextFormat(Qt.TextFormat.RichText)
        message_layout.addWidget(self._detail_label)
        
        # Add message layout to content layout
        content_layout.addLayout(message_layout, 1)
        
        # Add content layout to main layout
        main_layout.addLayout(content_layout)
        
        # Add spacing before buttons
        main_layout.addSpacing(10)
        
        # Create button box
        self._button_box = QDialogButtonBox()
        
        # Create buttons
        self._cancel_button = QPushButton("Cancel")
        self._confirm_button = QPushButton("Close")
        
        # Set button roles
        self._button_box.addButton(self._cancel_button, QDialogButtonBox.ButtonRole.RejectRole)
        self._button_box.addButton(self._confirm_button, QDialogButtonBox.ButtonRole.AcceptRole)
        
        # Connect button signals
        self._button_box.accepted.connect(self.accept)
        self._button_box.rejected.connect(self.reject)
        
        # Add button box to main layout
        main_layout.addWidget(self._button_box)
        
        # Set the layout
        self.setLayout(main_layout)
        
        # Set default button (highlighted when Enter is pressed)
        self._cancel_button.setDefault(True)
        
        # Set focus to cancel button (safer default)
        self._cancel_button.setFocus()
        
        logger.debug("CloseDialogView UI setup complete")
    
    def set_content(self, content: Dict[str, Any]) -> None:
        """
        Set the content of the dialog.
        
        Args:
            content: Dictionary containing dialog content including title, message,
                    detail, icon_type, confirm_button_text, and cancel_button_text
        """
        # Set window title
        if "title" in content:
            self.setWindowTitle(content["title"])
        
        # Set message
        if "message" in content:
            self._message_label.setText(f"<b>{content['message']}</b>")
        
        # Set detail
        if "detail" in content:
            self._detail_label.setText(content["detail"])
            self._detail_label.setVisible(bool(content["detail"]))
        
        # Set icon
        if "icon_type" in content:
            self._set_icon(content["icon_type"])
        
        # Set button texts
        if "confirm_button_text" in content:
            self._confirm_button.setText(content["confirm_button_text"])
        
        if "cancel_button_text" in content:
            self._cancel_button.setText(content["cancel_button_text"])
        
        # Set modality
        if "is_modal" in content:
            self.setModal(content["is_modal"])
        
        logger.debug("CloseDialogView content set")
    
    def _set_icon(self, icon_type: DialogIcon) -> None:
        """
        Set the dialog icon based on the specified icon type.
        
        Args:
            icon_type: The type of icon to display
        """
        # Map DialogIcon enum to QMessageBox standard icon
        icon_map = {
            DialogIcon.NONE: None,
            DialogIcon.INFORMATION: QMessageBox.Icon.Information,
            DialogIcon.WARNING: QMessageBox.Icon.Warning,
            DialogIcon.QUESTION: QMessageBox.Icon.Question,
            DialogIcon.ERROR: QMessageBox.Icon.Critical
        }
        
        # Get the QMessageBox standard icon
        standard_icon = icon_map.get(icon_type)
        
        if standard_icon is None:
            # No icon
            self._icon_label.clear()
            self._icon_label.setVisible(False)
            return
        
        # Get the icon pixmap from QMessageBox
        pixmap = self.style().standardIcon(standard_icon).pixmap(32, 32)
        
        # Set the icon
        self._icon_label.setPixmap(pixmap)
        self._icon_label.setVisible(True)
        
        logger.debug(f"Dialog icon set to {icon_type.name}")
    
    @pyqtSlot()
    def accept(self) -> None:
        """
        Handle the dialog accept event (when Close button is clicked or Enter is pressed).
        """
        # Emit the confirmed signal
        self.confirmed.emit()
        
        # Call the parent class implementation to close the dialog
        super().accept()
        
        logger.debug("CloseDialogView accepted (close confirmed)")
    
    @pyqtSlot()
    def reject(self) -> None:
        """
        Handle the dialog reject event (when Cancel button is clicked or Escape is pressed).
        """
        # Emit the canceled signal
        self.canceled.emit()
        
        # Call the parent class implementation to close the dialog
        super().reject()
        
        logger.debug("CloseDialogView rejected (close canceled)")
    
    def keyPressEvent(self, event):
        """
        Handle key press events.
        
        This ensures proper handling of Enter and Escape keys.
        
        Args:
            event: Key press event
        """
        # Let the parent class handle the event
        super().keyPressEvent(event)
