# mfe/ui/views/base_view.py

"""
Base View Class for MFE Toolbox UI Components

This module defines the BaseView abstract class that serves as the foundation for all
view components in the MFE Toolbox UI. It provides common functionality such as event
handling setup, styling application, and standard dialog behaviors to ensure consistency
across all views and reduce code duplication.

The BaseView class implements the View component of the Model-View-Controller (MVC)
architecture, focusing on UI rendering and user interaction while delegating business
logic to controller classes.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

from PyQt6.QtCore import (
    QEvent, QObject, QPoint, QRect, QSize, Qt, pyqtSignal, pyqtSlot
)
from PyQt6.QtGui import (
    QColor, QFont, QIcon, QPainter, QPalette, QPixmap, QResizeEvent
)
from PyQt6.QtWidgets import (
    QApplication, QDialog, QFrame, QHBoxLayout, QLabel, QLayout, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QSizePolicy, QSpacerItem,
    QVBoxLayout, QWidget
)

from mfe.core.exceptions import UIError
from mfe.core.types import AsyncCallback, UICallback, UIEventHandler, UIUpdateFunction

# Set up module-level logger
logger = logging.getLogger("mfe.ui.views.base")


class BaseView(ABC):
    """
    Abstract base class for all view components in the MFE Toolbox UI.

    This class provides common functionality for all views, including setup methods
    for widgets and layouts, styling application, signal connections, and standard
    dialog behaviors. It serves as the foundation for the View layer in the MVC
    architecture.

    Attributes:
        widget: The main widget or window for this view
        is_initialized: Flag indicating whether the view has been initialized
        _progress_indicator: Optional progress indicator widget for async operations
    """

    def __init__(self) -> None:
        """
        Initialize the base view.

        This constructor sets up basic state but does not create any UI elements.
        Subclasses should call this constructor and then call setup_ui() to create
        their UI elements.
        """
        self.widget: Optional[QWidget] = None
        self.is_initialized: bool = False
        self._progress_indicator: Optional[QProgressBar] = None
        self._progress_label: Optional[QLabel] = None
        self._progress_frame: Optional[QFrame] = None
        self._is_showing_progress: bool = False

        # Initialize logger
        self.logger = logging.getLogger(f"mfe.ui.views.{self.__class__.__name__}")
        self.logger.debug(f"Initializing {self.__class__.__name__}")

    @abstractmethod
    def setup_ui(self) -> None:
        """
        Set up the user interface for this view.

        This abstract method must be implemented by all subclasses to create their
        specific UI elements. It should create widgets, set up layouts, and configure
        initial properties.
        """
        pass

    def setup_widget(self, widget_class: Type[QWidget], *args: Any, **kwargs: Any) -> QWidget:
        """
        Create and configure a widget with common setup.

        Args:
            widget_class: The class of widget to create
            *args: Positional arguments to pass to the widget constructor
            **kwargs: Keyword arguments to pass to the widget constructor

        Returns:
            The created and configured widget
        """
        widget = widget_class(*args, **kwargs)
        self.apply_style_to_widget(widget)
        return widget

    def setup_layout(self, layout_class: Type[QLayout], *args: Any, **kwargs: Any) -> QLayout:
        """
        Create and configure a layout with common setup.

        Args:
            layout_class: The class of layout to create
            *args: Positional arguments to pass to the layout constructor
            **kwargs: Keyword arguments to pass to the layout constructor

        Returns:
            The created and configured layout
        """
        layout = layout_class(*args, **kwargs)
        # Common layout settings
        if isinstance(layout, (QHBoxLayout, QVBoxLayout)):
            layout.setContentsMargins(10, 10, 10, 10)
            layout.setSpacing(6)
        return layout

    def apply_style_to_widget(self, widget: QWidget) -> None:
        """
        Apply the standard style to a widget.

        This method applies consistent styling to widgets based on their type.
        It can be overridden by subclasses to provide custom styling.

        Args:
            widget: The widget to style
        """
        # Import here to avoid circular imports
        from .styles import apply_style_to_widget
        apply_style_to_widget(widget)

    def apply_theme(self, theme_name: str = "default") -> None:
        """
        Apply a theme to the entire view.

        This method applies a consistent theme to all widgets in the view.

        Args:
            theme_name: The name of the theme to apply
        """
        if self.widget is None:
            self.logger.warning("Cannot apply theme: widget not initialized")
            return

        # Import here to avoid circular imports
        from .styles import get_theme_colors

        # Get theme colors
        colors = get_theme_colors(theme_name)

        # Apply to the main widget and propagate to children
        palette = self.widget.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(colors["background"]))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(colors["text"]))
        palette.setColor(QPalette.ColorRole.Base, QColor(colors["base"]))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(colors["alternate_base"]))
        palette.setColor(QPalette.ColorRole.Button, QColor(colors["button"]))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(colors["button_text"]))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(colors["highlight"]))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(colors["highlighted_text"]))

        self.widget.setPalette(palette)
        self.widget.setAutoFillBackground(True)

    def connect_signal(self,
                       sender: QObject,
                       signal_name: str,
                       slot: Callable[..., Any],
                       connection_type: Qt.ConnectionType = Qt.ConnectionType.AutoConnection) -> bool:
        """
        Connect a signal to a slot with error handling.

        This method provides a consistent way to connect signals to slots with
        proper error handling and logging.

        Args:
            sender: The object that emits the signal
            signal_name: The name of the signal to connect
            slot: The slot function to connect to the signal
            connection_type: The type of connection to use

        Returns:
            True if the connection was successful, False otherwise
        """
        try:
            # Get the signal by name
            signal = getattr(sender, signal_name)
            # Connect the signal to the slot
            signal.connect(slot, connection_type)
            self.logger.debug(f"Connected {signal_name} to {slot.__name__}")
            return True
        except (AttributeError, TypeError) as e:
            self.logger.error(f"Failed to connect signal {signal_name}: {str(e)}")
            return False

    def disconnect_signal(self,
                          sender: QObject,
                          signal_name: str,
                          slot: Optional[Callable[..., Any]] = None) -> bool:
        """
        Disconnect a signal from a slot with error handling.

        This method provides a consistent way to disconnect signals from slots with
        proper error handling and logging.

        Args:
            sender: The object that emits the signal
            signal_name: The name of the signal to disconnect
            slot: The slot function to disconnect, or None to disconnect all slots

        Returns:
            True if the disconnection was successful, False otherwise
        """
        try:
            # Get the signal by name
            signal = getattr(sender, signal_name)

            # Disconnect the signal from the slot
            if slot is None:
                signal.disconnect()
                self.logger.debug(f"Disconnected all slots from {signal_name}")
            else:
                signal.disconnect(slot)
                self.logger.debug(f"Disconnected {slot.__name__} from {signal_name}")
            return True
        except (AttributeError, TypeError) as e:
            self.logger.error(f"Failed to disconnect signal {signal_name}: {str(e)}")
            return False

    def center_widget(self, widget: Optional[QWidget] = None) -> None:
        """
        Center a widget on the screen.

        This method centers a widget on the screen, or centers the main widget
        if no widget is specified.

        Args:
            widget: The widget to center, or None to center the main widget
        """
        target_widget = widget or self.widget
        if target_widget is None:
            self.logger.warning("Cannot center widget: no widget specified")
            return

        # Get the screen geometry
        screen = QApplication.primaryScreen()
        if screen is None:
            self.logger.warning("Cannot center widget: no screen available")
            return

        screen_geometry = screen.availableGeometry()

        # Calculate the center position
        widget_geometry = target_widget.frameGeometry()
        center_point = screen_geometry.center()
        widget_geometry.moveCenter(center_point)

        # Move the widget to the center position
        target_widget.move(widget_geometry.topLeft())

    def show_message_box(self,
                         title: str,
                         message: str,
                         icon: QMessageBox.Icon = QMessageBox.Icon.Information,
                         buttons: QMessageBox.StandardButton = QMessageBox.StandardButton.Ok,
                         default_button: QMessageBox.StandardButton = QMessageBox.StandardButton.Ok) -> QMessageBox.StandardButton:
        """
        Show a message box with the specified parameters.

        This method provides a consistent way to show message boxes with proper
        styling and positioning.

        Args:
            title: The title of the message box
            message: The message to display
            icon: The icon to display in the message box
            buttons: The buttons to include in the message box
            default_button: The default button to select

        Returns:
            The button that was clicked
        """
        # Create the message box
        msg_box = QMessageBox()
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setIcon(icon)
        msg_box.setStandardButtons(buttons)
        msg_box.setDefaultButton(default_button)

        # Apply styling
        self.apply_style_to_widget(msg_box)

        # Center the message box
        self.center_widget(msg_box)

        # Show the message box and return the result
        return cast(QMessageBox.StandardButton, msg_box.exec())

    def show_error_message(self, title: str, message: str) -> None:
        """
        Show an error message box.

        This method provides a convenient way to show error message boxes with
        consistent styling and positioning.

        Args:
            title: The title of the error message box
            message: The error message to display
        """
        self.logger.error(f"Error: {message}")
        self.show_message_box(
            title,
            message,
            QMessageBox.Icon.Critical,
            QMessageBox.StandardButton.Ok
        )

    def show_warning_message(self, title: str, message: str) -> None:
        """
        Show a warning message box.

        This method provides a convenient way to show warning message boxes with
        consistent styling and positioning.

        Args:
            title: The title of the warning message box
            message: The warning message to display
        """
        self.logger.warning(f"Warning: {message}")
        self.show_message_box(
            title,
            message,
            QMessageBox.Icon.Warning,
            QMessageBox.StandardButton.Ok
        )

    def show_info_message(self, title: str, message: str) -> None:
        """
        Show an information message box.

        This method provides a convenient way to show information message boxes with
        consistent styling and positioning.

        Args:
            title: The title of the information message box
            message: The information message to display
        """
        self.logger.info(f"Info: {message}")
        self.show_message_box(
            title,
            message,
            QMessageBox.Icon.Information,
            QMessageBox.StandardButton.Ok
        )

    def show_confirmation_dialog(self,
                                 title: str,
                                 message: str,
                                 default_button: QMessageBox.StandardButton = QMessageBox.StandardButton.No) -> bool:
        """
        Show a confirmation dialog and return whether the user confirmed.

        This method provides a convenient way to show confirmation dialogs with
        consistent styling and positioning.

        Args:
            title: The title of the confirmation dialog
            message: The confirmation message to display
            default_button: The default button to select

        Returns:
            True if the user confirmed, False otherwise
        """
        result = self.show_message_box(
            title,
            message,
            QMessageBox.Icon.Question,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            default_button
        )
        return result == QMessageBox.StandardButton.Yes

    def create_progress_indicator(self) -> None:
        """
        Create a progress indicator for asynchronous operations.

        This method creates a progress indicator that can be shown during
        long-running asynchronous operations to provide feedback to the user.
        """
        if self.widget is None:
            self.logger.warning("Cannot create progress indicator: widget not initialized")
            return

        # Create the progress frame
        self._progress_frame = QFrame(self.widget)
        self._progress_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self._progress_frame.setFrameShadow(QFrame.Shadow.Raised)
        self._progress_frame.setAutoFillBackground(True)

        # Set up the layout
        layout = QVBoxLayout(self._progress_frame)

        # Create the progress label
        self._progress_label = QLabel("Processing...")
        self._progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._progress_label)

        # Create the progress bar
        self._progress_indicator = QProgressBar()
        self._progress_indicator.setRange(0, 100)
        self._progress_indicator.setValue(0)
        self._progress_indicator.setTextVisible(True)
        layout.addWidget(self._progress_indicator)

        # Create a cancel button
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.on_cancel_progress)
        layout.addWidget(cancel_button, 0, Qt.AlignmentFlag.AlignCenter)

        # Hide the progress frame initially
        self._progress_frame.hide()

        # Position the progress frame
        self._update_progress_frame_position()

    def _update_progress_frame_position(self) -> None:
        """
        Update the position of the progress frame.

        This method updates the position of the progress frame to ensure it is
        centered in the main widget.
        """
        if (self.widget is None or self._progress_frame is None or
                not self.widget.isVisible()):
            return

        # Set the size of the progress frame
        progress_width = min(400, self.widget.width() - 40)
        progress_height = 120
        self._progress_frame.setFixedSize(progress_width, progress_height)

        # Calculate the center position
        x = (self.widget.width() - progress_width) // 2
        y = (self.widget.height() - progress_height) // 2

        # Move the progress frame to the center position
        self._progress_frame.move(x, y)

    def show_progress(self, message: str, progress: float = 0.0) -> None:
        """
        Show the progress indicator with the specified message and progress.

        This method shows the progress indicator with the specified message and
        progress value to provide feedback during long-running operations.

        Args:
            message: The message to display in the progress indicator
            progress: The progress value (0.0 to 1.0)
        """
        if (self._progress_frame is None or self._progress_label is None or
                self._progress_indicator is None):
            self.create_progress_indicator()

        if (self._progress_frame is None or self._progress_label is None or
                self._progress_indicator is None):
            self.logger.warning("Cannot show progress: progress indicator not created")
            return

        # Update the progress label and value
        self._progress_label.setText(message)
        self._progress_indicator.setValue(int(progress * 100))

        # Show the progress frame if it's not already visible
        if not self._is_showing_progress:
            self._progress_frame.show()
            self._is_showing_progress = True

        # Update the position of the progress frame
        self._update_progress_frame_position()

        # Process events to update the UI
        QApplication.processEvents()

    def hide_progress(self) -> None:
        """
        Hide the progress indicator.

        This method hides the progress indicator when a long-running operation
        is complete.
        """
        if self._progress_frame is not None and self._is_showing_progress:
            self._progress_frame.hide()
            self._is_showing_progress = False

            # Process events to update the UI
            QApplication.processEvents()

    def update_progress(self, progress: float, message: Optional[str] = None) -> None:
        """
        Update the progress indicator with a new progress value and optional message.

        This method updates the progress indicator with a new progress value and
        optional message to provide feedback during long-running operations.

        Args:
            progress: The progress value (0.0 to 1.0)
            message: The message to display, or None to keep the current message
        """
        if (self._progress_indicator is None or not self._is_showing_progress):
            return

        # Update the progress value
        self._progress_indicator.setValue(int(progress * 100))

        # Update the message if provided
        if message is not None and self._progress_label is not None:
            self._progress_label.setText(message)

        # Process events to update the UI
        QApplication.processEvents()

    @pyqtSlot()
    def on_cancel_progress(self) -> None:
        """
        Handle cancellation of a progress operation.

        This method is called when the user clicks the cancel button in the
        progress indicator. Subclasses should override this method to implement
        cancellation behavior.
        """
        self.logger.debug("Progress operation cancelled by user")
        self.hide_progress()

    async def run_async_operation(self,
                                  operation: AsyncCallback,
                                  progress_callback: Optional[Callable[[float, str], None]] = None) -> Any:
        """
        Run an asynchronous operation with progress reporting.

        This method runs an asynchronous operation and shows a progress indicator
        during execution. It provides a consistent way to handle long-running
        operations with proper UI feedback.

        Args:
            operation: The asynchronous operation to run
            progress_callback: Optional callback for progress updates

        Returns:
            The result of the asynchronous operation
        """
        # Show the progress indicator
        self.show_progress("Starting operation...", 0.0)

        try:
            # Create a progress callback that updates the progress indicator
            def update_progress(progress: float, message: str) -> None:
                self.update_progress(progress, message)
                if progress_callback is not None:
                    progress_callback(progress, message)

            # Run the operation with the progress callback
            result = await operation()

            # Hide the progress indicator
            self.hide_progress()

            return result
        except Exception as e:
            # Hide the progress indicator
            self.hide_progress()

            # Log the error
            self.logger.error(f"Async operation failed: {str(e)}")

            # Show an error message
            self.show_error_message("Operation Failed", f"The operation failed: {str(e)}")

            # Re-raise the exception
            raise

    def handle_resize_event(self, event: QResizeEvent) -> None:
        """
        Handle resize events for the main widget.

        This method handles resize events for the main widget, updating the
        position of the progress frame if necessary.

        Args:
            event: The resize event
        """
        # Update the position of the progress frame
        self._update_progress_frame_position()

        # Accept the event
        event.accept()

    def handle_event(self, obj: QObject, event: QEvent) -> bool:
        """
        Handle events for the view.

        This method provides a central point for handling events for the view.
        Subclasses can override this method to handle specific events.

        Args:
            obj: The object that received the event
            event: The event

        Returns:
            True if the event was handled, False otherwise
        """
        # Handle resize events
        if event.type() == QEvent.Type.Resize and obj == self.widget:
            self.handle_resize_event(cast(QResizeEvent, event))
            return True

        # Let the default event handler handle other events
        return False

    def install_event_filter(self) -> None:
        """
        Install an event filter for the main widget.

        This method installs an event filter for the main widget to handle
        events like resize events.
        """
        if self.widget is None:
            self.logger.warning("Cannot install event filter: widget not initialized")
            return

        # Create an event filter
        class EventFilter(QObject):
            def __init__(self, view: BaseView):
                super().__init__()
                self.view = view

            def eventFilter(self, obj: QObject, event: QEvent) -> bool:
                return self.view.handle_event(obj, event)

        # Install the event filter
        event_filter = EventFilter(self)
        self.widget.installEventFilter(event_filter)

        # Store the event filter to prevent garbage collection
        self.widget.setProperty("event_filter", event_filter)

    def cleanup(self) -> None:
        """
        Clean up resources used by the view.

        This method cleans up resources used by the view, such as event filters
        and connections. Subclasses should call this method when the view is
        no longer needed.
        """
        self.logger.debug(f"Cleaning up {self.__class__.__name__}")

        # Hide the progress indicator
        self.hide_progress()

        # Clean up the progress indicator
        self._progress_indicator = None
        self._progress_label = None
        self._progress_frame = None

        # Clean up the widget
        if self.widget is not None:
            # Remove the event filter
            event_filter = self.widget.property("event_filter")
            if event_filter is not None:
                self.widget.removeEventFilter(event_filter)

            # Set the widget to None
            self.widget = None

        # Reset initialization flag
        self.is_initialized = False
