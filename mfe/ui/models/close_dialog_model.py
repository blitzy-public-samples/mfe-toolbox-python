"""
Close Dialog Model

This module implements the data model for the close confirmation dialog in the MFE Toolbox UI.
It manages the dialog content, state, and result tracking for the confirmation dialog shown
when closing the application. It provides a clean separation between UI and data logic for
this modal dialog.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Union

# Set up module-level logger
logger = logging.getLogger("mfe.ui.models.close_dialog")


class DialogResult(Enum):
    """
    Enumeration of possible dialog results.
    
    This enum represents the possible outcomes when a user interacts with the close dialog.
    """
    NONE = auto()      # No result yet (dialog still open)
    CANCEL = auto()    # User canceled the close operation
    CONFIRM = auto()   # User confirmed the close operation


class DialogIcon(Enum):
    """
    Enumeration of possible dialog icons.
    
    This enum represents the available icon types for the close dialog.
    """
    NONE = auto()       # No icon
    INFORMATION = auto() # Information icon
    WARNING = auto()     # Warning icon
    QUESTION = auto()    # Question icon
    ERROR = auto()       # Error icon


@dataclass
class CloseDialogModel:
    """
    Data model for the close confirmation dialog.
    
    This class manages the content and state of the close confirmation dialog,
    providing a clean separation between UI content and presentation logic.
    
    Attributes:
        title: The title of the dialog
        message: The main message displayed in the dialog
        detail: Optional additional details to display
        icon_type: The type of icon to display in the dialog
        confirm_button_text: Text for the confirm button
        cancel_button_text: Text for the cancel button
        result: The current result of the dialog (NONE, CANCEL, or CONFIRM)
        is_modal: Whether the dialog should be modal (block interaction with parent window)
    """
    
    # Dialog title
    title: str = "Confirm Close"
    
    # Dialog content
    message: str = "Are you sure you want to close the ARMAX GUI?"
    detail: str = "All unsaved changes will be lost."
    
    # Dialog appearance
    icon_type: DialogIcon = DialogIcon.WARNING
    
    # Button labels
    confirm_button_text: str = "Close"
    cancel_button_text: str = "Cancel"
    
    # Dialog state
    result: DialogResult = DialogResult.NONE
    is_modal: bool = True
    
    # Private fields for internal use
    _custom_data: Dict[str, Any] = field(default_factory=dict, repr=False)
    
    def __post_init__(self) -> None:
        """
        Initialize the model after dataclass initialization.
        
        This method:
        1. Validates the model data
        2. Sets up any derived properties
        3. Logs initialization
        """
        logger.debug(f"Initializing CloseDialogModel with title '{self.title}'")
        self._validate_fields()
    
    def _validate_fields(self) -> None:
        """
        Validate the model fields to ensure they contain valid data.
        
        Raises:
            ValueError: If any field contains invalid data
        """
        if not self.title:
            raise ValueError("Dialog title cannot be empty")
        
        if not self.message:
            raise ValueError("Dialog message cannot be empty")
        
        if not isinstance(self.icon_type, DialogIcon):
            raise ValueError(f"Invalid icon type: {self.icon_type}")
        
        if not isinstance(self.result, DialogResult):
            raise ValueError(f"Invalid dialog result: {self.result}")
    
    def set_result(self, result: DialogResult) -> None:
        """
        Set the dialog result.
        
        Args:
            result: The dialog result (CANCEL or CONFIRM)
            
        Raises:
            ValueError: If the result is not a valid DialogResult
        """
        if not isinstance(result, DialogResult):
            raise ValueError(f"Invalid dialog result: {result}")
        
        logger.debug(f"Setting dialog result to {result.name}")
        self.result = result
    
    def set_confirm(self) -> None:
        """Set the dialog result to CONFIRM."""
        self.set_result(DialogResult.CONFIRM)
    
    def set_cancel(self) -> None:
        """Set the dialog result to CANCEL."""
        self.set_result(DialogResult.CANCEL)
    
    def reset(self) -> None:
        """Reset the dialog result to NONE."""
        logger.debug("Resetting dialog result")
        self.result = DialogResult.NONE
    
    def is_confirmed(self) -> bool:
        """
        Check if the dialog was confirmed.
        
        Returns:
            True if the dialog result is CONFIRM, False otherwise
        """
        return self.result == DialogResult.CONFIRM
    
    def is_canceled(self) -> bool:
        """
        Check if the dialog was canceled.
        
        Returns:
            True if the dialog result is CANCEL, False otherwise
        """
        return self.result == DialogResult.CANCEL
    
    def is_pending(self) -> bool:
        """
        Check if the dialog is still pending (no result yet).
        
        Returns:
            True if the dialog result is NONE, False otherwise
        """
        return self.result == DialogResult.NONE
    
    def set_custom_data(self, key: str, value: Any) -> None:
        """
        Set custom data for the dialog.
        
        This method allows storing additional data associated with the dialog
        that might be needed by the controller or view.
        
        Args:
            key: The key for the custom data
            value: The value to store
        """
        self._custom_data[key] = value
    
    def get_custom_data(self, key: str, default: Any = None) -> Any:
        """
        Get custom data from the dialog.
        
        Args:
            key: The key for the custom data
            default: The default value to return if the key is not found
            
        Returns:
            The stored value, or the default if the key is not found
        """
        return self._custom_data.get(key, default)
    
    def get_dialog_content(self) -> Dict[str, Any]:
        """
        Get all dialog content as a dictionary.
        
        This method provides a convenient way to access all dialog content
        in a format that can be easily used by the view.
        
        Returns:
            Dictionary containing all dialog content
        """
        return {
            "title": self.title,
            "message": self.message,
            "detail": self.detail,
            "icon_type": self.icon_type,
            "confirm_button_text": self.confirm_button_text,
            "cancel_button_text": self.cancel_button_text,
            "is_modal": self.is_modal
        }
    
    def update_content(self, 
                      title: Optional[str] = None,
                      message: Optional[str] = None,
                      detail: Optional[str] = None,
                      icon_type: Optional[DialogIcon] = None,
                      confirm_button_text: Optional[str] = None,
                      cancel_button_text: Optional[str] = None) -> None:
        """
        Update the dialog content.
        
        This method allows updating multiple content fields at once.
        Only the provided fields will be updated; others remain unchanged.
        
        Args:
            title: New dialog title
            message: New dialog message
            detail: New dialog detail
            icon_type: New dialog icon type
            confirm_button_text: New confirm button text
            cancel_button_text: New cancel button text
        """
        if title is not None:
            self.title = title
        
        if message is not None:
            self.message = message
        
        if detail is not None:
            self.detail = detail
        
        if icon_type is not None:
            self.icon_type = icon_type
        
        if confirm_button_text is not None:
            self.confirm_button_text = confirm_button_text
        
        if cancel_button_text is not None:
            self.cancel_button_text = cancel_button_text
        
        # Validate the updated fields
        self._validate_fields()
        logger.debug("Dialog content updated")