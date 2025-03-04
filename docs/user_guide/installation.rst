============
Installation
============

This guide provides detailed instructions for installing the MFE Toolbox, a Python-based suite for financial econometrics, time series analysis, and risk modeling.

System Requirements
==================

The MFE Toolbox requires:

* Python 3.12 or later
* Operating System: Windows, macOS, or Linux (any OS that supports Python 3.12)
* Recommended: 4GB+ RAM for large datasets and complex models

Standard Installation
====================

The simplest way to install the MFE Toolbox is using pip, Python's package installer:

.. code-block:: bash

    pip install mfe-toolbox

This command automatically installs the MFE Toolbox along with all required dependencies.

Installation with Virtual Environments
=====================================

For isolated installations that don't affect your system-wide Python setup, we recommend using virtual environments.

Using venv (Python's built-in virtual environment)
-------------------------------------------------

.. code-block:: bash

    # Create a virtual environment
    python -m venv mfe_env
    
    # Activate the environment (Windows)
    mfe_env\Scripts\activate
    
    # Activate the environment (macOS/Linux)
    source mfe_env/bin/activate
    
    # Install MFE Toolbox
    pip install mfe-toolbox
    
    # When finished, deactivate the environment
    deactivate

Using conda (with Anaconda or Miniconda)
---------------------------------------

.. code-block:: bash

    # Create a conda environment with Python 3.12
    conda create -n mfe_env python=3.12
    
    # Activate the environment
    conda activate mfe_env
    
    # Install MFE Toolbox using pip within conda
    pip install mfe-toolbox
    
    # When finished, deactivate the environment
    conda deactivate

Dependencies
===========

The MFE Toolbox automatically installs the following dependencies:

* **NumPy** (≥1.26.0): Efficient array operations and linear algebra
* **SciPy** (≥1.11.3): Optimization and statistical functions
* **Pandas** (≥2.1.1): Time series data handling
* **Numba** (≥0.58.0): Just-In-Time compilation for performance optimization
* **Statsmodels** (≥0.14.0): Econometric modeling and statistical analysis
* **Matplotlib** (≥3.8.0): Visualization capabilities
* **PyQt6** (≥6.5.0): GUI components (only required for the ARMAX GUI)

Verifying Your Installation
==========================

After installation, you can verify that the MFE Toolbox and its dependencies are correctly installed:

.. code-block:: python

    # Start Python
    import mfe
    
    # Print the version
    print(mfe.__version__)
    
    # Verify key components are available
    from mfe.models.univariate import GARCH
    from mfe.models.multivariate import DCC
    from mfe.models.time_series import ARMA
    from mfe.models.bootstrap import BlockBootstrap
    from mfe.models.realized import RealizedVariance
    
    print("MFE Toolbox successfully installed!")

Performance Optimization with Numba
==================================

The MFE Toolbox uses Numba's Just-In-Time (JIT) compilation to optimize performance-critical functions. Unlike the previous MATLAB implementation that required manual compilation of MEX files, Numba automatically compiles Python functions to optimized machine code at runtime.

Key benefits of this approach:

* No manual compilation steps required
* Cross-platform compatibility without platform-specific binaries
* Performance comparable to compiled C code (typically 10-100x faster than pure Python)
* Automatic optimization for your specific CPU architecture

When a JIT-decorated function is first called, you may notice a brief delay as Numba compiles it. Subsequent calls will be much faster as Numba reuses the compiled version.

Installation from Source
======================

For developers or users who want the latest development version, you can install directly from the source repository:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/bashtage/arch.git
    cd arch
    
    # Install in development mode
    pip install -e .
    
    # Install with development dependencies
    pip install -e ".[dev]"
    
    # Install with documentation dependencies
    pip install -e ".[docs]"
    
    # Install with testing dependencies
    pip install -e ".[test]"
    
    # Install with all optional dependencies
    pip install -e ".[dev,docs,test]"

Installing the development version allows you to modify the code and immediately see the effects without reinstalling.

Troubleshooting
==============

Common Installation Issues
-------------------------

1. **Python Version Error**:
   
   If you see an error about Python version requirements, ensure you're using Python 3.12 or later:
   
   .. code-block:: bash
   
       python --version
   
   If needed, download and install Python 3.12+ from https://www.python.org/downloads/

2. **Dependency Conflicts**:
   
   If you encounter dependency conflicts, try installing in a fresh virtual environment:
   
   .. code-block:: bash
   
       python -m venv fresh_env
       # Activate as shown earlier
       pip install mfe-toolbox

3. **Numba Installation Issues**:
   
   On some systems, Numba might require additional setup. See the Numba documentation for platform-specific instructions: https://numba.pydata.org/numba-doc/latest/user/installing.html

4. **PyQt6 Installation Issues**:
   
   If you encounter problems with PyQt6 installation but don't need the GUI components, you can install without PyQt6:
   
   .. code-block:: bash
   
       pip install mfe-toolbox
       # Then manually install other dependencies except PyQt6
       pip install numpy scipy pandas numba statsmodels matplotlib

5. **Import Errors After Installation**:
   
   If you see import errors, verify your Python environment is correctly activated and that the package is installed:
   
   .. code-block:: bash
   
       pip list | grep mfe-toolbox

Updating the MFE Toolbox
=======================

To update to the latest version:

.. code-block:: bash

    pip install --upgrade mfe-toolbox

This will install the newest version while preserving your existing configuration.

Uninstalling
===========

To remove the MFE Toolbox:

.. code-block:: bash

    pip uninstall mfe-toolbox

Note that this will not remove dependencies that were installed with the package. To remove unused dependencies, you can use a tool like pip-autoremove.

Next Steps
=========

Now that you have successfully installed the MFE Toolbox, you can proceed to the :doc:`getting_started` guide to learn how to use the package for financial econometric analysis.