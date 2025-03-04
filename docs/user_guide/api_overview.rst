Based on the repository structure and the requirements, I'll now create the API overview document.

# docs/user_guide/api_overview.rst

=======================
API Overview
=======================

This document provides an overview of the MFE Toolbox API structure, explaining how the package is organized and how to navigate its components. The MFE Toolbox follows modern Python package design principles with a clear hierarchical structure.

Package Organization
====================

The MFE Toolbox is organized as a Python package with the following main components:

.. code-block:: text

    mfe/                    # Main package
    ├── __init__.py         # Package initialization
    ├── core/               # Core functionality and base classes
    ├── models/             # Implementation of econometric models
    ├── ui/                 # User interface components
    └── utils/              # Helper functions and utilities

This modular organization allows you to import only the components you need for your specific analysis.

Import System
============

To use the MFE Toolbox, you first need to import it in your Python script or notebook. The package follows Python's standard import conventions:

.. code-block:: python

    # Import the entire package
    import mfe
    
    # Import specific modules
    from mfe import models
    from mfe.models import univariate
    
    # Import specific classes
    from mfe.models.univariate import GARCH
    from mfe.models.time_series import ARMA
    
    # Import utility functions
    from mfe.utils.matrix_ops import vech, ivech

For convenience, the most commonly used models are available directly from the main package namespace:

.. code-block:: python

    import mfe
    
    # These models are available directly
    model = mfe.GARCH(...)
    ts_model = mfe.ARMA(...)
    mv_model = mfe.DCC(...)
    bootstrap = mfe.BlockBootstrap(...)
    rv = mfe.RealizedVariance(...)

Core Module (mfe.core)
======================

The ``mfe.core`` module provides fundamental base classes and infrastructure used throughout the toolbox:

.. code-block:: text

    core/
    ├── __init__.py         # Module initialization
    ├── base.py             # Base classes for models
    ├── parameters.py       # Parameter container classes
    ├── results.py          # Result container classes
    ├── types.py            # Type definitions and aliases
    ├── exceptions.py       # Custom exception classes
    ├── validation.py       # Input validation utilities
    └── config.py           # Configuration management

Key components include:

* **Base Classes**: Abstract base classes that define common interfaces for all models
* **Parameter Containers**: Dataclasses for storing and validating model parameters
* **Result Containers**: Structured classes for model outputs and diagnostics
* **Type Definitions**: Common type aliases used throughout the codebase
* **Exception Classes**: Specialized exceptions for different error conditions
* **Validation Utilities**: Functions for validating inputs and parameters

Models Module (mfe.models)
=========================

The ``mfe.models`` module contains implementations of various econometric models organized into submodules by category:

.. code-block:: text

    models/
    ├── __init__.py             # Module initialization
    ├── bootstrap/              # Bootstrap methods for dependent data
    ├── cross_section/          # Cross-sectional analysis tools
    ├── distributions/          # Statistical distribution functions
    ├── multivariate/           # Multivariate volatility models
    ├── realized/               # High-frequency econometrics tools
    ├── time_series/            # Time series analysis toolkit
    └── univariate/             # Univariate volatility models

Each submodule follows a consistent pattern:

* A ``base.py`` file defining abstract base classes for that category
* Implementation files for specific models
* A ``_numba_core.py`` or ``_core.py`` file containing performance-critical functions
* Utility modules specific to that category

For example, the univariate volatility models are organized as:

.. code-block:: text

    univariate/
    ├── __init__.py         # Module initialization
    ├── base.py             # Base class for univariate models
    ├── garch.py            # GARCH model implementation
    ├── egarch.py           # EGARCH model implementation
    ├── tarch.py            # TARCH model implementation
    ├── ...                 # Other model implementations
    ├── _core.py            # Performance-critical functions
    └── utils.py            # Utility functions

UI Module (mfe.ui)
=================

The ``mfe.ui`` module provides graphical user interface components implemented using PyQt6:

.. code-block:: text

    ui/
    ├── __init__.py             # Module initialization
    ├── armax_app.py            # Main ARMAX application
    ├── about_dialog.py         # About dialog implementation
    ├── close_dialog.py         # Close confirmation dialog
    ├── model_viewer.py         # Model results viewer
    ├── utils.py                # UI utility functions
    ├── models/                 # UI model classes (MVC pattern)
    ├── views/                  # UI view classes (MVC pattern)
    ├── controllers/            # UI controller classes (MVC pattern)
    └── resources/              # UI resources (images, icons)

The UI module follows the Model-View-Controller (MVC) pattern:

* **Models**: Data structures and business logic
* **Views**: Visual components and layouts
* **Controllers**: Event handling and coordination

To launch the ARMAX GUI:

.. code-block:: python

    from mfe.ui.armax_app import launch_armax_gui
    
    launch_armax_gui()

Utils Module (mfe.utils)
=======================

The ``mfe.utils`` module provides helper functions and utilities used throughout the toolbox:

.. code-block:: text

    utils/
    ├── __init__.py             # Module initialization
    ├── matrix_ops.py           # Matrix operations (vech, ivech, etc.)
    ├── covariance.py           # Covariance estimation functions
    ├── differentiation.py      # Numerical differentiation utilities
    ├── data_transformations.py # Data transformation functions
    ├── date_utils.py           # Date handling utilities
    └── misc.py                 # Miscellaneous helper functions

These utility functions provide essential building blocks for implementing econometric models and statistical procedures.

Class Hierarchy and Object-Oriented Design
=========================================

The MFE Toolbox uses a class-based, object-oriented design with inheritance hierarchies to promote code reuse and consistent interfaces:

.. code-block:: text

    BaseModel (abstract)
    ├── UnivariateVolatilityModel (abstract)
    │   ├── GARCH
    │   ├── EGARCH
    │   ├── TARCH
    │   └── ...
    ├── MultivariateVolatilityModel (abstract)
    │   ├── BEKK
    │   ├── DCC
    │   └── ...
    ├── TimeSeriesModel (abstract)
    │   ├── ARMA
    │   ├── VAR
    │   └── ...
    └── ...

This hierarchical structure ensures that:

* Common functionality is implemented once in base classes
* All models provide a consistent interface
* Specialized behavior is encapsulated in subclasses
* Code is maintainable and extensible

Type Hinting System
==================

The MFE Toolbox makes extensive use of Python's type hinting system to improve code reliability and developer experience:

.. code-block:: python

    from typing import Optional, Union, List, Dict, Tuple, Any, Callable
    import numpy as np
    import pandas as pd
    
    def estimate_volatility(
        returns: Union[np.ndarray, pd.Series],
        p: int = 1,
        q: int = 1,
        power: float = 2.0,
        distribution: str = "normal"
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Estimate volatility using a GARCH model.
        
        Parameters
        ----------
        returns : array_like
            Return series to model
        p : int, optional
            GARCH lag order
        q : int, optional
            ARCH lag order
        power : float, optional
            Power transformation parameter
        distribution : str, optional
            Error distribution, one of: 'normal', 't', 'ged', 'skewed_t'
            
        Returns
        -------
        volatility : ndarray
            Estimated conditional volatility series
        results : dict
            Dictionary containing estimation results
        """
        # Implementation...

Type hints provide several benefits:

* **Documentation**: Clear indication of expected input and output types
* **IDE Support**: Better autocompletion and error detection in editors
* **Static Analysis**: Ability to catch type errors before runtime using tools like mypy
* **Runtime Validation**: Option to validate inputs against their expected types

Dataclasses for Parameter Management
===================================

The MFE Toolbox uses Python's dataclasses for parameter management, providing structured containers with validation:

.. code-block:: python

    from dataclasses import dataclass, field
    from typing import Optional, List
    
    @dataclass
    class GARCHParams:
        """Parameters for GARCH model."""
        omega: float
        alpha: List[float]
        beta: List[float]
        power: float = 2.0
        gamma: Optional[List[float]] = None
        
        def __post_init__(self):
            """Validate parameters after initialization."""
            if self.omega <= 0:
                raise ValueError("omega must be positive")
            
            if any(a < 0 for a in self.alpha):
                raise ValueError("alpha coefficients must be non-negative")
            
            if any(b < 0 for b in self.beta):
                raise ValueError("beta coefficients must be non-negative")
            
            if sum(self.alpha) + sum(self.beta) >= 1:
                raise ValueError("Model is not stationary (sum of alpha and beta >= 1)")

Dataclasses provide:

* **Automatic initialization**: No need to write boilerplate __init__ methods
* **Field validation**: Ability to validate parameters after initialization
* **Immutability option**: Can create frozen instances to prevent modification
* **Default values**: Specify default values for optional parameters
* **Representation**: Automatic __repr__ and __str__ methods

Asynchronous Processing
=====================

For long-running computations, the MFE Toolbox provides asynchronous processing capabilities using Python's async/await syntax:

.. code-block:: python

    import asyncio
    
    # Synchronous API
    model = mfe.GARCH(p=1, q=1)
    result = model.fit(returns)
    
    # Asynchronous API
    async def estimate_model():
        model = mfe.GARCH(p=1, q=1)
        result = await model.fit_async(returns, progress_callback=report_progress)
        return result
    
    async def report_progress(percent, message):
        print(f"{percent:.1f}% complete: {message}")
    
    # Run the async function
    result = asyncio.run(estimate_model())

Asynchronous processing provides:

* **Responsiveness**: UI remains responsive during long computations
* **Progress Reporting**: Ability to report progress during execution
* **Cancellation**: Option to cancel long-running operations
* **Concurrency**: Potential to run multiple operations concurrently

Numba Acceleration
================

Performance-critical functions in the MFE Toolbox are accelerated using Numba's just-in-time (JIT) compilation:

.. code-block:: python

    from numba import jit
    import numpy as np
    
    @jit(nopython=True, cache=True)
    def garch_recursion(parameters, residuals, sigma2, backcast):
        """
        Core GARCH variance recursion.
        
        This function is automatically compiled to optimized machine code
        the first time it's called, with results cached for subsequent calls.
        """
        T = len(residuals)
        omega, alpha, beta = parameters
        
        for t in range(1, T):
            sigma2[t] = omega + alpha * residuals[t-1]**2 + beta * sigma2[t-1]
        
        return sigma2

Numba acceleration provides:

* **Performance**: 10-100x speedup for computation-intensive functions
* **Cross-Platform**: Works consistently across operating systems
* **Simplicity**: No need for separate C/C++ implementations
* **Maintainability**: Single codebase in Python

Example Usage Patterns
====================

Here are some common usage patterns for the MFE Toolbox:

Univariate Volatility Modeling
-----------------------------

.. code-block:: python

    import numpy as np
    import pandas as pd
    from mfe.models.univariate import GARCH
    
    # Create and fit a GARCH model
    model = GARCH(p=1, q=1)
    result = model.fit(returns)
    
    # Access results
    print(f"Estimated parameters: {result.params}")
    print(f"Log-likelihood: {result.loglikelihood}")
    print(f"AIC: {result.aic}")
    
    # Get conditional volatility
    volatility = result.conditional_volatility
    
    # Forecast future volatility
    forecast = model.forecast(horizon=10)
    
    # Simulate from the model
    simulated_returns, simulated_volatility = model.simulate(T=1000)

Multivariate Volatility Modeling
------------------------------

.. code-block:: python

    import numpy as np
    import pandas as pd
    from mfe.models.multivariate import DCC
    
    # Create and fit a DCC model
    model = DCC(univariate_model="GARCH")
    result = model.fit(returns_matrix)
    
    # Access results
    print(f"Univariate parameters: {result.univariate_parameters}")
    print(f"Correlation parameters: {result.correlation_parameters}")
    
    # Get conditional covariance matrices
    covariance_matrices = result.conditional_covariance
    
    # Get conditional correlation matrices
    correlation_matrices = result.conditional_correlation
    
    # Forecast future covariance
    forecast = model.forecast(horizon=10)

Time Series Analysis
------------------

.. code-block:: python

    import numpy as np
    import pandas as pd
    from mfe.models.time_series import ARMA
    
    # Create and fit an ARMA model
    model = ARMA(ar_order=2, ma_order=1, include_constant=True)
    result = model.fit(time_series)
    
    # Access results
    print(f"AR parameters: {result.ar_parameters}")
    print(f"MA parameters: {result.ma_parameters}")
    print(f"Constant: {result.constant}")
    
    # Get fitted values and residuals
    fitted = result.fitted_values
    residuals = result.residuals
    
    # Forecast future values
    forecast = model.forecast(horizon=10, intervals=True, alpha=0.05)
    
    # Plot ACF and PACF
    from mfe.models.time_series.plots import plot_acf_pacf
    plot_acf_pacf(time_series, lags=20)

Bootstrap Analysis
----------------

.. code-block:: python

    import numpy as np
    import pandas as pd
    from mfe.models.bootstrap import BlockBootstrap
    
    # Create a block bootstrap instance
    bootstrap = BlockBootstrap(block_length=10)
    
    # Generate bootstrap samples
    bootstrap_samples = bootstrap.generate_samples(data, num_samples=1000)
    
    # Compute bootstrap statistics
    bootstrap_means = np.array([sample.mean() for sample in bootstrap_samples])
    
    # Compute confidence intervals
    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])
    
    # Run Model Confidence Set
    from mfe.models.bootstrap import MCS
    mcs = MCS()
    included_models = mcs.run(loss_matrix, alpha=0.05)

Realized Volatility Estimation
----------------------------

.. code-block:: python

    import numpy as np
    import pandas as pd
    from mfe.models.realized import RealizedVariance, RealizedKernel
    
    # Compute realized variance
    rv = RealizedVariance()
    variance = rv.compute(high_frequency_returns)
    
    # Compute realized kernel with optimal bandwidth
    rk = RealizedKernel(kernel_type="parzen", bandwidth="optimal")
    kernel_variance = rk.compute(high_frequency_returns)
    
    # Handle irregular timestamps
    from mfe.models.realized import seconds2unit, price_filter
    
    # Convert timestamps to normalized time
    times_unit = seconds2unit(timestamps_seconds)
    
    # Filter prices to regular grid
    filtered_prices = price_filter(prices, times_unit, grid_size=390)

Using the GUI
-----------

.. code-block:: python

    from mfe.ui.armax_app import launch_armax_gui
    
    # Launch the ARMAX GUI
    launch_armax_gui()

Conclusion
=========

The MFE Toolbox provides a comprehensive, well-organized API for financial econometrics and time series analysis. The package follows modern Python design principles with clear module organization, consistent class hierarchies, and extensive type hinting.

By leveraging Python's import system, you can easily access the specific components you need for your analysis. The object-oriented design ensures consistent interfaces across different model types, while the use of dataclasses and type hints improves code reliability and developer experience.

For performance-critical operations, the toolbox uses Numba's JIT compilation to achieve near-native performance without sacrificing the clarity and maintainability of Python code. Asynchronous processing capabilities ensure responsiveness during long-running computations, particularly important for the GUI components and bootstrap procedures.
