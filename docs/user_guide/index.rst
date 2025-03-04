====================
MFE Toolbox User Guide
====================

Welcome to the MFE Toolbox User Guide. This guide provides comprehensive documentation for the MFE Toolbox, a Python-based suite for financial econometrics, time series analysis, and risk modeling.

The MFE Toolbox represents a complete modernization of the original MATLAB-based toolbox, now fully implemented in Python 3.12. It leverages powerful Python libraries including NumPy for matrix operations, SciPy for optimization and statistical functions, Pandas for time series handling, and Statsmodels for econometric modeling. Performance optimization is achieved through Numba's just-in-time compilation (using @jit decorators), replacing the earlier C-based MEX implementations.

This user guide is organized into sections to help you quickly find the information you need:

Getting Started
==============

These sections will help you install the toolbox and begin using its basic functionality:

.. toctree::
   :maxdepth: 2
   
   installation
   getting_started

Core Functionality
================

Learn about the fundamental components and utilities that power the MFE Toolbox:

.. toctree::
   :maxdepth: 2
   
   statistical_distributions
   statistical_tests
   bootstrap_methods

Model Types
==========

Detailed guides for the various econometric models included in the toolbox:

.. toctree::
   :maxdepth: 2
   
   univariate_volatility_models
   multivariate_volatility_models
   time_series_analysis
   high_frequency_econometrics
   cross_sectional_analysis

Advanced Features
===============

Explore more sophisticated capabilities of the MFE Toolbox:

.. toctree::
   :maxdepth: 2
   
   gui_interface
   api_overview

Python Ecosystem Integration
==========================

The MFE Toolbox is designed to integrate seamlessly with the Python scientific ecosystem:

- **NumPy Integration**: Efficient array operations and linear algebra functions
- **Pandas Integration**: Time series handling with DatetimeIndex support
- **Statsmodels Integration**: Extended econometric modeling capabilities
- **Numba Acceleration**: Performance-critical functions optimized with @jit decorators
- **Type Safety**: Comprehensive type hints throughout the codebase
- **Asynchronous Processing**: Support for non-blocking execution of long-running operations

The toolbox follows modern Python programming paradigms such as explicit typing, dataclasses, and asynchronous processing to ensure maintainability and extensibility.

Example Workflow
==============

A typical workflow with the MFE Toolbox might look like:

.. code-block:: python

   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   from mfe.models.univariate import GARCH
   from mfe.models.distributions import StudentT
   
   # Load or create financial time series data
   data = pd.read_csv('returns.csv', index_col=0, parse_dates=True)
   returns = data['returns'].values
   
   # Create and fit a GARCH(1,1) model with Student's t distribution
   model = GARCH(p=1, q=1, error_dist=StudentT())
   results = model.fit(returns)
   
   # Print model summary
   print(results.summary())
   
   # Plot conditional volatility
   plt.figure(figsize=(10, 6))
   plt.plot(data.index, np.sqrt(results.conditional_variance))
   plt.title('GARCH(1,1) Conditional Volatility')
   plt.ylabel('Volatility')
   plt.show()
   
   # Generate forecasts
   forecasts = results.forecast(horizon=10)
   print(forecasts.summary())

For more detailed examples, see the specific model documentation and the example notebooks included with the toolbox.