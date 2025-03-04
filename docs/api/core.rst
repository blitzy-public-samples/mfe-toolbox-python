.. _api_core:

=============
Core (mfe.core)
=============

The ``mfe.core`` package provides the foundational components upon which the rest of the MFE Toolbox is built. It contains base classes, parameter containers, type definitions, and other essential infrastructure that supports the various model implementations throughout the library.

Base Classes
===========

.. automodule:: mfe.core.base
   :members:
   :undoc-members:
   :show-inheritance:

The base module defines abstract base classes that establish common interfaces and functionality shared across different model types. These classes provide consistent patterns for parameter validation, model estimation, and result presentation.

.. code-block:: python

   from mfe.core.base import Model, VolatilityModel, TimeSeriesModel

   # Example of extending a base class
   class CustomModel(VolatilityModel):
       def __init__(self, parameters):
           super().__init__(parameters)
           # Custom initialization

       def fit(self, data):
           # Implementation of the fit method
           pass

Parameter Containers
==================

.. automodule:: mfe.core.parameters
   :members:
   :undoc-members:
   :show-inheritance:

The parameters module provides dataclass-based containers for model parameters with built-in validation. These classes ensure parameter constraints are enforced and provide clear error messages when invalid values are provided.

.. code-block:: python

   from mfe.core.parameters import GARCHParameters
   
   # Create parameter container with validation
   params = GARCHParameters(omega=0.1, alpha=0.1, beta=0.8)
   
   # Validation occurs automatically
   # This would raise an error since alpha + beta >= 1
   # invalid_params = GARCHParameters(omega=0.1, alpha=0.5, beta=0.5)

Results Containers
================

.. automodule:: mfe.core.results
   :members:
   :undoc-members:
   :show-inheritance:

The results module defines structured containers for model estimation results, including parameter estimates, standard errors, diagnostic statistics, and fitted values.

.. code-block:: python

   from mfe.core.results import ModelResults
   
   # Example of creating a results container
   results = ModelResults(
       parameters=estimated_params,
       standard_errors=std_errors,
       log_likelihood=log_lik,
       aic=aic_value,
       bic=bic_value
   )

Type Definitions
==============

.. automodule:: mfe.core.types
   :members:
   :undoc-members:
   :show-inheritance:

The types module provides type definitions and type aliases used throughout the MFE Toolbox. These definitions enhance code readability and enable static type checking with tools like mypy.

.. code-block:: python

   from mfe.core.types import ArrayLike, TimeSeriesData, OptionalFloat
   
   def process_data(data: TimeSeriesData) -> ArrayLike:
       # Function with type hints using core type definitions
       pass

Exceptions
=========

.. automodule:: mfe.core.exceptions
   :members:
   :undoc-members:
   :show-inheritance:

The exceptions module defines custom exception classes used throughout the MFE Toolbox. These specialized exceptions provide detailed context information to help diagnose and resolve issues.

.. code-block:: python

   from mfe.core.exceptions import ParameterError, ConvergenceError
   
   try:
       # Some operation that might fail
       pass
   except ParameterError as e:
       # Handle parameter validation error
       print(f"Parameter error: {e}")
   except ConvergenceError as e:
       # Handle convergence failure
       print(f"Convergence failed: {e}")

Validation Utilities
==================

.. automodule:: mfe.core.validation
   :members:
   :undoc-members:
   :show-inheritance:

The validation module provides utilities for validating inputs, parameters, and other data throughout the MFE Toolbox. These functions help ensure data consistency and prevent runtime errors.

.. code-block:: python

   from mfe.core.validation import validate_array, validate_positive

   # Validate input array
   data = validate_array(input_data, ndim=2)
   
   # Validate positive parameter
   omega = validate_positive(omega_value, "omega")

Configuration
===========

.. automodule:: mfe.core.config
   :members:
   :undoc-members:
   :show-inheritance:

The config module handles toolbox-wide configuration settings, including logging configuration, performance options, and user preferences.

.. code-block:: python

   from mfe.core.config import get_config, set_config
   
   # Get current configuration
   config = get_config()
   
   # Set configuration option
   set_config("logging.level", "INFO")
   set_config("performance.use_numba", True)

Performance Considerations
========================

Many core functions in the MFE Toolbox are designed to work efficiently with large datasets. Performance-critical operations are accelerated using Numba's @jit decorators, providing near-native performance while maintaining the flexibility of Python code.

The core package establishes patterns for asynchronous processing that are used throughout the toolbox, enabling long-running computations to execute without blocking the user interface or other operations.

.. code-block:: python

   # Example of asynchronous model estimation
   async def estimate_model_async():
       model = SomeModel(parameters)
       results = await model.fit_async(data, progress_callback=update_progress)
       return results

   # This can be awaited without blocking
   results = await estimate_model_async()

Type Safety
==========

The MFE Toolbox makes extensive use of Python's type hints to improve code reliability and development experience. All core components include comprehensive type annotations that enable static type checking with tools like mypy.

These type hints serve as both documentation and runtime validation, helping prevent type-related errors and providing clear guidance on expected parameter types and return values.

.. code-block:: python

   from typing import Optional, List, Dict, Union
   import numpy as np
   
   def example_function(
       data: np.ndarray,
       parameters: Optional[Dict[str, float]] = None,
       options: List[str] = []
   ) -> Union[np.ndarray, float]:
       # Function implementation with type safety
       pass
