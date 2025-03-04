.. _api_ui:

=================
UI (mfe.ui)
=================

The ``mfe.ui`` package provides a modern graphical user interface for the MFE Toolbox, specifically focused on ARMAX (AutoRegressive Moving Average with eXogenous inputs) modeling. The interface is built using the PyQt6 framework and follows the Model-View-Controller (MVC) architectural pattern for clean separation of concerns.

The UI components enable interactive time series analysis, model configuration, parameter estimation, and diagnostic visualization through a responsive graphical interface that remains responsive during computationally intensive operations through asynchronous processing.

Architecture Overview
====================

The UI package implements a strict Model-View-Controller (MVC) architecture:

- **Models**: Data structures and business logic for the UI components
- **Views**: Visual presentation and user interaction components
- **Controllers**: Coordination between models and views, handling user actions

.. code-block:: python

   # Example of MVC pattern usage
   from mfe.ui.models.armax_model import ARMAXModel
   from mfe.ui.views.armax_view import ARMAXView
   from mfe.ui.controllers.armax_controller import ARMAXController
   
   # Create MVC components
   model = ARMAXModel()
   view = ARMAXView()
   controller = ARMAXController(model, view)
   
   # Launch the application
   view.show()

Main Application
===============

.. automodule:: mfe.ui.armax_app
   :members:
   :undoc-members:
   :show-inheritance:

The ``armax_app`` module provides the main entry point for the ARMAX modeling interface. It initializes the PyQt6 application, sets up the MVC components, and manages the application lifecycle.

.. code-block:: python

   from mfe.ui.armax_app import launch_armax_gui
   
   # Launch the ARMAX GUI
   launch_armax_gui(data=my_time_series)

Models
======

The models package contains data structures and business logic for the UI components.

ARMAX Model
-----------

.. automodule:: mfe.ui.models.armax_model
   :members:
   :undoc-members:
   :show-inheritance:

The ``armax_model`` module implements the data model for ARMAX time series analysis, handling parameter storage, model estimation, and results management.

.. code-block:: python

   from mfe.ui.models.armax_model import ARMAXModel
   
   # Create model with time series data
   model = ARMAXModel(data=my_time_series)
   
   # Configure model parameters
   model.set_ar_order(2)
   model.set_ma_order(1)
   model.set_include_constant(True)
   
   # Estimate model (returns awaitable coroutine)
   estimation_task = model.estimate_model()

About Dialog Model
-----------------

.. automodule:: mfe.ui.models.about_dialog_model
   :members:
   :undoc-members:
   :show-inheritance:

The ``about_dialog_model`` module provides data for the About dialog, including version information, credits, and application metadata.

Close Dialog Model
-----------------

.. automodule:: mfe.ui.models.close_dialog_model
   :members:
   :undoc-members:
   :show-inheritance:

The ``close_dialog_model`` module manages the state for the close confirmation dialog, tracking unsaved changes and confirmation status.

Model Viewer Model
-----------------

.. automodule:: mfe.ui.models.model_viewer_model
   :members:
   :undoc-members:
   :show-inheritance:

The ``model_viewer_model`` module handles the data representation for the detailed model results viewer, including parameter tables, statistics, and equation formatting.

Views
=====

The views package contains the visual presentation components of the UI.

Base View
---------

.. automodule:: mfe.ui.views.base_view
   :members:
   :undoc-members:
   :show-inheritance:

The ``base_view`` module provides a foundation class for all view components, implementing common functionality such as signal definitions, layout management, and styling.

ARMAX View
----------

.. automodule:: mfe.ui.views.armax_view
   :members:
   :undoc-members:
   :show-inheritance:

The ``armax_view`` module implements the main ARMAX modeling interface, including model configuration controls, visualization areas, and tabbed result displays.

.. code-block:: python

   from mfe.ui.views.armax_view import ARMAXView
   
   # Create the main view
   view = ARMAXView()
   
   # Connect signals to slots
   view.estimate_button.clicked.connect(on_estimate_clicked)
   
   # Show the view
   view.show()

About Dialog View
----------------

.. automodule:: mfe.ui.views.about_dialog_view
   :members:
   :undoc-members:
   :show-inheritance:

The ``about_dialog_view`` module implements the About dialog interface, displaying application information, version details, and credits.

Close Dialog View
----------------

.. automodule:: mfe.ui.views.close_dialog_view
   :members:
   :undoc-members:
   :show-inheritance:

The ``close_dialog_view`` module implements the close confirmation dialog, prompting users to confirm when closing with unsaved changes.

Model Viewer View
----------------

.. automodule:: mfe.ui.views.model_viewer_view
   :members:
   :undoc-members:
   :show-inheritance:

The ``model_viewer_view`` module implements the detailed model results viewer, displaying parameter estimates, standard errors, test statistics, and model equations.

UI Components
------------

.. automodule:: mfe.ui.views.components
   :members:
   :undoc-members:
   :show-inheritance:

The ``components`` module provides reusable UI components used throughout the interface, including specialized input controls, visualization widgets, and custom displays.

Styles
------

.. automodule:: mfe.ui.views.styles
   :members:
   :undoc-members:
   :show-inheritance:

The ``styles`` module defines consistent styling for UI components, ensuring a cohesive visual appearance throughout the application.

Controllers
==========

The controllers package contains the coordination logic between models and views.

ARMAX Controller
---------------

.. automodule:: mfe.ui.controllers.armax_controller
   :members:
   :undoc-members:
   :show-inheritance:

The ``armax_controller`` module coordinates between the ARMAX model and view, handling user interactions and updating the UI based on model changes.

.. code-block:: python

   from mfe.ui.models.armax_model import ARMAXModel
   from mfe.ui.views.armax_view import ARMAXView
   from mfe.ui.controllers.armax_controller import ARMAXController
   
   # Create MVC components
   model = ARMAXModel(data=my_time_series)
   view = ARMAXView()
   controller = ARMAXController(model, view)
   
   # Controller automatically connects signals and slots

About Dialog Controller
----------------------

.. automodule:: mfe.ui.controllers.about_dialog_controller
   :members:
   :undoc-members:
   :show-inheritance:

The ``about_dialog_controller`` module manages the About dialog interactions, handling display and dismissal.

Close Dialog Controller
----------------------

.. automodule:: mfe.ui.controllers.close_dialog_controller
   :members:
   :undoc-members:
   :show-inheritance:

The ``close_dialog_controller`` module manages the close confirmation dialog, handling user decisions and application exit logic.

Model Viewer Controller
----------------------

.. automodule:: mfe.ui.controllers.model_viewer_controller
   :members:
   :undoc-members:
   :show-inheritance:

The ``model_viewer_controller`` module coordinates the model results viewer, handling navigation, display formatting, and user interactions.

Dialogs
=======

About Dialog
-----------

.. automodule:: mfe.ui.about_dialog
   :members:
   :undoc-members:
   :show-inheritance:

The ``about_dialog`` module provides a modal dialog displaying application information, version details, and credits.

.. code-block:: python

   from mfe.ui.about_dialog import show_about_dialog
   
   # Display the about dialog
   show_about_dialog(parent_widget)

Close Dialog
-----------

.. automodule:: mfe.ui.close_dialog
   :members:
   :undoc-members:
   :show-inheritance:

The ``close_dialog`` module implements a confirmation dialog that appears when closing the application with unsaved changes.

Model Viewer
-----------

.. automodule:: mfe.ui.model_viewer
   :members:
   :undoc-members:
   :show-inheritance:

The ``model_viewer`` module provides a detailed view of model estimation results, including parameter estimates, standard errors, test statistics, and model equations.

Utilities
========

.. automodule:: mfe.ui.utils
   :members:
   :undoc-members:
   :show-inheritance:

The ``utils`` module provides helper functions for UI operations, including data formatting, visualization utilities, and PyQt6 integration helpers.

Resources
========

.. automodule:: mfe.ui.resources.resource_loader
   :members:
   :undoc-members:
   :show-inheritance:

The ``resource_loader`` module handles loading and management of UI resources such as icons, images, and other assets used in the interface.

Asynchronous Processing
======================

The UI components leverage Python's asynchronous programming capabilities to maintain responsiveness during computationally intensive operations. This is implemented through the async/await pattern, allowing long-running tasks to execute without blocking the user interface.

.. code-block:: python

   # Example of asynchronous model estimation
   async def estimate_model():
       # Update UI to show progress
       self.view.show_progress("Estimating model...")
       
       # Run computation asynchronously
       results = await self.model.estimate_async(
           progress_callback=self.update_progress
       )
       
       # Update UI with results
       self.view.update_results(results)
   
   # Launch asynchronous task without blocking UI
   asyncio.create_task(estimate_model())

The asynchronous design provides several benefits:

1. **Responsive UI**: The interface remains interactive during long computations
2. **Progress Reporting**: Real-time updates on computation progress
3. **Cancellation Support**: Users can cancel long-running operations
4. **Resource Efficiency**: Better utilization of system resources

PyQt6 Integration
================

The UI components are built using PyQt6, a comprehensive GUI framework that provides:

1. **Widget Toolkit**: Extensive set of UI controls and containers
2. **Signal-Slot Mechanism**: Event handling through connected signals and slots
3. **Layout Management**: Flexible arrangement of UI components
4. **Style Customization**: Consistent visual appearance
5. **Cross-Platform Support**: Works on Windows, macOS, and Linux

The integration with PyQt6 follows best practices for Python GUI development:

.. code-block:: python

   from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QPushButton
   from PyQt6.QtCore import pyqtSignal, pyqtSlot
   
   class ExampleView(QMainWindow):
       # Define custom signals
       data_changed = pyqtSignal(object)
       
       def __init__(self):
           super().__init__()
           self.setup_ui()
       
       def setup_ui(self):
           # Create and arrange widgets
           layout = QVBoxLayout()
           button = QPushButton("Process Data")
           layout.addWidget(button)
           
           # Connect signals to slots
           button.clicked.connect(self.on_button_clicked)
       
       @pyqtSlot()
       def on_button_clicked(self):
           # Handle button click
           result = process_data()
           self.data_changed.emit(result)

Matplotlib Integration
=====================

The UI components integrate matplotlib for visualization of time series data, model diagnostics, and statistical plots:

1. **Embedded Plots**: Matplotlib figures embedded in PyQt6 widgets
2. **Interactive Visualization**: Zoom, pan, and save capabilities
3. **LaTeX Rendering**: Mathematical equations displayed using matplotlib's LaTeX support
4. **Customized Styling**: Consistent visual appearance matching the application theme

This integration enables rich visualization capabilities while maintaining the native look and feel of the application.

Performance Considerations
=========================

The UI components are designed for optimal performance, even with large datasets and complex models:

1. **Asynchronous Processing**: Long-running computations execute without blocking the UI
2. **Lazy Loading**: Resources and components are loaded only when needed
3. **Efficient Data Handling**: Optimized data transfer between model and view
4. **Responsive Updates**: UI updates are batched for efficiency
5. **Memory Management**: Careful resource cleanup to prevent memory leaks

Type Safety
===========

The UI components make extensive use of Python's type hints to improve code reliability and development experience:

1. **Function Signatures**: Clear parameter and return type specifications
2. **Class Attributes**: Explicit typing for all class members
3. **Signal Definitions**: Type-safe signal and slot connections
4. **Generic Types**: Proper typing for collections and containers
5. **Optional Values**: Explicit handling of nullable parameters

These type hints serve as both documentation and runtime validation, helping prevent type-related errors and providing clear guidance on expected parameter types and return values.
