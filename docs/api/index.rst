.. _api_reference:

=============
API Reference
=============

This section provides detailed documentation for all public modules, classes, and functions in the MFE Toolbox. The API reference is organized according to the package structure, making it easy to locate specific components.

The MFE Toolbox is designed with a modular architecture, where related functionality is grouped into specialized packages:

* **Core**: Base classes, parameter handling, and foundational components
* **Models**: Implementation of econometric models organized by type
* **UI**: User interface components for the ARMAX modeling environment
* **Utils**: Helper functions for data transformation and analysis
* **Distributions**: Statistical distribution functions
* **Tests**: Statistical test implementations

Each module is documented with detailed descriptions, parameter specifications, return value information, and usage examples. Performance-critical functions accelerated with Numba's @jit decorators are clearly marked.

.. note::
   Classes and functions marked with an asterisk (*) are considered part of the stable public API and follow semantic versioning for backward compatibility. Other components may change between minor versions.

Core Components
===============

.. toctree::
   :maxdepth: 2
   
   core

Models
======

The models package contains implementations of various econometric models organized by type:

.. toctree::
   :maxdepth: 2
   
   models/index
   models/bootstrap
   models/cross_section
   models/distributions/index
   models/multivariate
   models/realized
   models/time_series
   models/univariate

User Interface
=============

.. toctree::
   :maxdepth: 2
   
   ui

Utilities
=========

.. toctree::
   :maxdepth: 2
   
   utils

Distributions
============

.. toctree::
   :maxdepth: 2
   
   distributions

Statistical Tests
================

.. toctree::
   :maxdepth: 2
   
   tests

Development
==========

For developers extending the MFE Toolbox, the following guidelines are recommended:

* Follow the established class hierarchy when implementing new models
* Use dataclasses for parameter containers with appropriate validation
* Add comprehensive type hints to all functions and methods
* Accelerate performance-critical sections with Numba's @jit decorators
* Include detailed docstrings following the NumPy documentation style
* Write unit tests for all new functionality

The MFE Toolbox leverages modern Python features including:

* Type hints for improved code reliability and IDE support
* Dataclasses for structured parameter containers
* Asynchronous processing for long-running computations
* Numba acceleration for performance-critical functions
* Integration with NumPy, SciPy, Pandas, and Statsmodels

For more information on extending the toolbox, see the :ref:`developer_guide`.
