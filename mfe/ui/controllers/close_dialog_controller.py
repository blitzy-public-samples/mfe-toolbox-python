# mfe/ui/controllers/close_dialog_controller.py
"""
Close Dialog Controller

This module implements the controller for the close confirmation dialog in the MFE Toolbox UI.
It manages the interaction between the close dialog model and view, handling user responses
to the confirmation prompt when closing the application. The controller follows the MVC pattern
and leverages PyQt6's signal-slot mechanism for event handling.
"""

import logging
from typing import Optional, Callable, Any, Dict, Union

from PyQt6.QtCore import QObject, pyqtSlot

from mfe.ui.models.close_dialog_model import CloseDialogModel, DialogResult
from mfe.ui.views.close_dialog_view import CloseDialogView

# Set up module-level logger
logger = logging.getLogger("mfe.ui.controllers.close_dialog")


class CloseDialogController(QObject):
    """
    Controller for the close confirmation dialog.
    
    This class implements the Controller part of the MVC pattern for the close confirmation
    dialog. It handles the interaction between the dialog model and view, processes user
    responses, and communicates the result back to the parent controller.
    
    The controller leverages PyQt6's signal-slot mechanism for event handling and maintains
    a clean separation between the dialog's visual presentation and its underlying logic.
    
    Attributes:
        model: The close dialog model instance
        view: The close dialog view instance
        on_confirm_callback: Callback function to execute when dialog is confirmed
        on_cancel_callback: Callback function to execute when dialog is canceled
    """
    
    def __init__(
        self,
        model: Optional[CloseDialogModel] = None,
        view: Optional[CloseDialogView] = None,
        parent: Optional[QObject] = None
    ):
        """
        Initialize the close dialog controller.
        
        Args:
            model: The close dialog model instance (created if None)
            view: The close dialog view instance (created if None)
            parent: Parent QObject, if any
        """
        super().__init__(parent)
        
        # Initialize model and view
        self.model = model if model is not None else CloseDialogModel()
        self.view = view if view is not None else CloseDialogView()
        
        # Initialize callback functions
        self.on_confirm_callback: Optional[Callable[[], Any]] = None
        self.on_cancel_callback: Optional[Callable[[], Any]] = None
        
        # Connect signals to slots
        self._connect_signals()
        
        # Initialize view with model state
        self._initialize_view()
        
        logger.debug("CloseDialogController initialized")
    
    def _connect_signals(self) -> None:
        """Connect view signals to controller slots."""
        # Connect dialog result signals
        self.view.confirmed.connect(self._on_dialog_confirmed)
        self.view.canceled.connect(self._on_dialog_canceled)
        
        logger.debug("View signals connected to controller slots")
    
    def _initialize_view(self) -> None:
        """Initialize the view with the current model state."""
        # Set dialog content from model
        self.view.set_content(self.model.get_dialog_content())
        
        logger.debug("View initialized with model state")
    
    def show(self) -> None:
        """Show the dialog."""
        # Reset model result before showing
        self.model.reset()
        
        # Update view with current model state
        self._initialize_view()
        
        # Show the dialog
        self.view.show()
        
        logger.debug("Close dialog shown")
    
    def exec(self) -> DialogResult:
        """
        Execute the dialog modally and return the result.
        
        Returns:
            The dialog result (CONFIRM, CANCEL, or NONE)
        """
        # Reset model result before showing
        self.model.reset()
        
        # Update view with current model state
        self._initialize_view()
        
        # Execute the dialog modally
        result = self.view.exec()
        
        # Return the model result
        return self.model.result
    
    def set_callbacks(
        self,
        on_confirm: Optional[Callable[[], Any]] = None,
        on_cancel: Optional[Callable[[], Any]] = None
    ) -> None:
        """
        Set callback functions for dialog results.
        
        Args:
            on_confirm: Function to call when dialog is confirmed
            on_cancel: Function to call when dialog is canceled
        """
        self.on_confirm_callback = on_confirm
        self.on_cancel_callback = on_cancel
        
        logger.debug("Dialog callbacks set")
    
    def update_content(
        self,
        title: Optional[str] = None,
        message: Optional[str] = None,
        detail: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Update the dialog content.
        
        Args:
            title: New dialog title
            message: New dialog message
            detail: New dialog detail
            **kwargs: Additional content parameters to update
        """
        # Update model content
        self.model.update_content(
            title=title,
            message=message,
            detail=detail,
            **kwargs
        )
        
        # Update view with new content
        self.view.set_content(self.model.get_dialog_content())
        
        logger.debug("Dialog content updated")
    
    # Signal handlers (slots)
    @pyqtSlot()
    def _on_dialog_confirmed(self) -> None:
        """Handle dialog confirmed signal."""
        # Update model result
        self.model.set_confirm()
        
        logger.debug("Dialog confirmed")
        
        # Execute callback if set
        if self.on_confirm_callback is not None:
            self.on_confirm_callback()
    
    @pyqtSlot()
    def _on_dialog_canceled(self) -> None:
        """Handle dialog canceled signal."""
        # Update model result
        self.model.set_cancel()
        
        logger.debug("Dialog canceled")
        
        # Execute callback if set
        if self.on_cancel_callback is not None:
            self.on_cancel_callback()
    
    def is_confirmed(self) -> bool:
        """
        Check if the dialog was confirmed.
        
        Returns:
            True if the dialog result is CONFIRM, False otherwise
        """
        return self.model.is_confirmed()
    
    def is_canceled(self) -> bool:
        """
        Check if the dialog was canceled.
        
        Returns:
            True if the dialog result is CANCEL, False otherwise
        """
        return self.model.is_canceled()
    
    def is_pending(self) -> bool:
        """
        Check if the dialog is still pending (no result yet).
        
        Returns:
            True if the dialog result is NONE, False otherwise
        """
        return self.model.is_pending()
