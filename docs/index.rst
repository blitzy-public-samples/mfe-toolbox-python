.. MFE Toolbox documentation master file

======================
MFE Toolbox for Python
======================

**A comprehensive Python-based suite for financial econometrics, time series analysis, and risk modeling**

.. image:: _static/logo.png
   :align: center
   :alt: MFE Toolbox Logo

.. raw:: html

   <div class="badges">
     <a href="https://pypi.org/project/mfe-toolbox/">
       <img src="https://img.shields.io/pypi/v/mfe-toolbox.svg" alt="PyPI">
     </a>
     <a href="https://pypi.org/project/mfe-toolbox/">
       <img src="https://img.shields.io/pypi/pyversions/mfe-toolbox.svg" alt="Python Versions">
     </a>
     <a href="https://github.com/bashtage/arch/blob/master/LICENSE">
       <img src="https://img.shields.io/github/license/bashtage/arch.svg" alt="License">
     </a>
   </div>

Overview
========

The MFE Toolbox is a comprehensive Python-based suite for financial econometrics, time series analysis, and risk modeling, targeting Python 3.12. It represents a complete modernization of the original MATLAB-based toolbox (formerly version 4.0, released 28-Oct-2009).

The toolbox addresses the critical need for robust, reliable, and efficient econometric tools for modeling financial time series, including volatility modeling, risk assessment, and time series forecasting.

Key Features
-----------

* **Univariate volatility modeling**: GARCH, EGARCH, TARCH, and other variants
* **Multivariate volatility modeling**: BEKK, DCC, RARCH, and related models
* **ARMA/ARMAX time series modeling and forecasting**
* **Bootstrap methods for dependent data**
* **Non-parametric volatility estimation** (realized volatility)
* **Classical statistical tests and distributions**
* **Vector autoregression (VAR) analysis**
* **Principal component analysis and cross-sectional econometrics**

Modern Python Implementation
---------------------------

The toolbox leverages powerful Python libraries including:

* **NumPy** for matrix operations
* **SciPy** for optimization and statistical functions
* **Pandas** for time series handling
* **Statsmodels** for econometric modeling
* **Numba** for just-in-time compilation (using @jit decorators)

Performance optimization is achieved through Numba's just-in-time compilation, replacing the earlier C-based MEX implementations. The codebase employs modern Python programming paradigms such as explicit typing, dataclasses, and asynchronous processing to ensure maintainability and extensibility.

Quick Start
==========

Installation
-----------

Install the MFE Toolbox using pip:

.. code-block:: bash

   pip install mfe-toolbox

Basic Usage
----------

Here's a simple example of fitting a GARCH(1,1) model:

.. code-block:: python

   import numpy as np
   from mfe.models.univariate import GARCH
   
   # Generate some sample data
   np.random.seed(42)
   returns = np.random.normal(0, 1, 1000)
   
   # Create and fit a GARCH(1,1) model
   model = GARCH(p=1, q=1)
   results = model.fit(returns)
   
   # Print the results
   print(results.summary())
   
   # Plot conditional volatility
   results.plot_conditional_volatility()

Documentation
============

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   
   user_guide/index
   user_guide/installation
   user_guide/getting_started
   user_guide/univariate_volatility_models
   user_guide/multivariate_volatility_models
   user_guide/time_series_analysis
   user_guide/bootstrap_methods
   user_guide/high_frequency_econometrics
   user_guide/statistical_distributions
   user_guide/statistical_tests
   user_guide/cross_sectional_analysis
   user_guide/gui_interface
   user_guide/api_overview

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   
   api/index
   api/core
   api/models/index
   api/models/bootstrap
   api/models/cross_section
   api/models/distributions/index
   api/models/multivariate
   api/models/realized
   api/models/time_series
   api/models/univariate
   api/ui
   api/utils
   api/distributions
   api/tests

.. toctree::
   :maxdepth: 1
   :caption: Examples
   
   examples/getting_started.ipynb
   examples/univariate_volatility_models.ipynb
   examples/multivariate_volatility_models.ipynb
   examples/time_series_analysis.ipynb
   examples/bootstrap_methods.ipynb
   examples/realized_volatility.ipynb
   examples/statistical_distributions.ipynb
   examples/statistical_tests.ipynb
   examples/cross_sectional_analysis.ipynb
   examples/gui_interface.ipynb
   examples/real_world_applications.ipynb

Target Audience
===============

The MFE Toolbox serves:

* Financial analysts
* Econometricians
* Researchers
* Practitioners in financial economics
* Risk management professionals
* Macroeconomic analysts

Its primary value proposition lies in providing a unified, well-documented, and thoroughly-tested collection of Python modules that implement state-of-the-art econometric methods, saving development time and ensuring reliable results for complex financial and economic analyses.

As an open-source Python ecosystem, it facilitates easier integration with existing systems, continuous modernization, and long-term maintainability.

Architecture
===========

The MFE Toolbox is organized into specialized Python packages:

* **mfe/core**: Core functionality and base classes
* **mfe/models**: Implementation of econometric models
* **mfe/ui**: User interface components including PyQt6-based ARMAX modeling environment
* **mfe/utils**: Helper functions for data transformation and analysis

Performance-critical components are implemented using Numba's @jit decorators for optimization, replacing the earlier C-based MEX implementations.

Indices and Tables
=================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

License
=======

The MFE Toolbox is distributed under the MIT License. See the LICENSE file for more information.

Citation
========

If you use the MFE Toolbox in your research, please cite it as:

.. code-block:: text

   Sheppard, K. (2023). MFE Toolbox for Python [Computer software].
   https://github.com/bashtage/arch
