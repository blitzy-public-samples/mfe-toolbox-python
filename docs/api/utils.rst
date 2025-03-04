.. _api_utils:

==================
Utilities (mfe.utils)
==================

The ``mfe.utils`` package provides essential utility functions used throughout the MFE Toolbox. These utilities include matrix operations, covariance estimation, numerical differentiation, data transformations, date handling, and other helper functions that support the core functionality of the toolbox.

Matrix Operations
===============

.. automodule:: mfe.utils.matrix_ops
   :members:
   :undoc-members:
   :show-inheritance:

The matrix_ops module provides efficient matrix manipulation functions optimized for financial econometrics applications. These functions include vectorization operations (vech/ivech), matrix decompositions, and specialized transformations used in multivariate volatility models.

.. code-block:: python

   from mfe.utils.matrix_ops import vech, ivech, vec2chol, chol2vec
   import numpy as np
   
   # Create a symmetric matrix
   A = np.array([[1.0, 0.5, 0.3], 
                 [0.5, 2.0, 0.7],
                 [0.3, 0.7, 3.0]])
   
   # Convert to half-vector form
   v = vech(A)  # Returns [1.0, 0.5, 2.0, 0.3, 0.7, 3.0]
   
   # Convert back to symmetric matrix
   A_restored = ivech(v)  # Returns original matrix A
   
   # Convert Cholesky factor to vector representation
   L = np.linalg.cholesky(A)
   v_chol = vec2chol(L)
   
   # Convert vector back to Cholesky factor
   L_restored = chol2vec(v_chol, 3)

Many matrix operations are accelerated using Numba's @jit decorators for optimal performance with large matrices, while maintaining the flexibility and readability of Python code.

Covariance Estimation
===================

.. automodule:: mfe.utils.covariance
   :members:
   :undoc-members:
   :show-inheritance:

The covariance module implements robust covariance matrix estimators, including Newey-West and other heteroskedasticity and autocorrelation consistent (HAC) estimators. These functions are essential for accurate inference in time series and cross-sectional models.

.. code-block:: python

   from mfe.utils.covariance import covnw, covhac, robustvcv
   import numpy as np
   
   # Generate some example data
   T = 1000
   k = 3
   X = np.random.randn(T, k)
   e = np.random.randn(T, 1)
   
   # Compute Newey-West covariance matrix with automatic lag selection
   cov_nw = covnw(X, e)
   
   # Compute robust variance-covariance matrix for regression
   beta = np.array([0.5, 1.0, -0.5])
   scores = X * e  # Simplified example
   robust_vcv = robustvcv(scores, X)

These estimators are implemented using NumPy's efficient array operations and accelerated with Numba for performance-critical calculations, making them suitable for large datasets.

Numerical Differentiation
=======================

.. automodule:: mfe.utils.differentiation
   :members:
   :undoc-members:
   :show-inheritance:

The differentiation module provides numerical approximation methods for gradients, Jacobians, and Hessians. These functions are used extensively in optimization routines throughout the toolbox, particularly for maximum likelihood estimation.

.. code-block:: python

   from mfe.utils.differentiation import gradient_2sided, hessian_2sided
   import numpy as np
   
   # Define an objective function
   def objective(params):
       x, y = params
       return x**2 + y**2 + x*y
   
   # Compute gradient at a specific point
   params = np.array([1.0, 2.0])
   grad = gradient_2sided(objective, params)
   
   # Compute Hessian matrix
   hess = hessian_2sided(objective, params)

The module implements various finite difference methods with configurable step sizes and accuracy levels. Performance-critical operations are accelerated with Numba for efficient computation with high-dimensional parameter vectors.

Data Transformations
==================

.. automodule:: mfe.utils.data_transformations
   :members:
   :undoc-members:
   :show-inheritance:

The data_transformations module provides functions for common data preprocessing operations in financial econometrics, including standardization, demeaning, lag creation, and other transformations.

.. code-block:: python

   from mfe.utils.data_transformations import standardize, demean, create_lags
   import numpy as np
   import pandas as pd
   
   # Generate example data
   data = np.random.randn(100)
   
   # Standardize data (zero mean, unit variance)
   std_data = standardize(data)
   
   # Create time series with lags
   ts = pd.Series(data)
   lagged_data = create_lags(ts, lags=3)  # Creates DataFrame with original and 3 lags

These functions work seamlessly with both NumPy arrays and Pandas objects, providing flexible interfaces for different data structures while maintaining computational efficiency.

Date Utilities
============

.. automodule:: mfe.utils.date_utils
   :members:
   :undoc-members:
   :show-inheritance:

The date_utils module provides functions for handling dates and times in financial time series analysis. These utilities facilitate conversion between different date formats, business day calculations, and time series alignment.

.. code-block:: python

   from mfe.utils.date_utils import convert_dates, business_days_between
   import pandas as pd
   from datetime import datetime
   
   # Convert string dates to datetime objects
   dates = ["2023-01-01", "2023-01-15", "2023-02-01"]
   dt_objects = convert_dates(dates)
   
   # Calculate business days between dates
   start_date = datetime(2023, 1, 1)
   end_date = datetime(2023, 1, 31)
   days = business_days_between(start_date, end_date)

The module leverages Python's datetime standard library and Pandas' powerful datetime functionality, providing timezone-aware operations and support for various financial calendar conventions.

Miscellaneous Utilities
=====================

.. automodule:: mfe.utils.misc
   :members:
   :undoc-members:
   :show-inheritance:

The misc module contains various helper functions that don't fit into other categories but are useful throughout the toolbox. These include parameter transformation functions, special mathematical operations, and other utilities.

.. code-block:: python

   from mfe.utils.misc import r2z, z2r, phi2r, r2phi
   
   # Convert correlation to Fisher's Z
   rho = 0.7
   z = r2z(rho)
   
   # Convert back to correlation
   rho_restored = z2r(z)
   
   # Convert AR parameter to correlation
   phi = 0.8
   rho = phi2r(phi)
   
   # Convert correlation to AR parameter
   phi_restored = r2phi(rho)

These utilities implement various mathematical transformations and helper functions that are commonly used in financial econometrics, with a focus on numerical stability and computational efficiency.

Performance Optimization
======================

Many utility functions in the MFE Toolbox are performance-critical and have been optimized using Numba's just-in-time compilation. Functions decorated with ``@jit`` are automatically compiled to optimized machine code at runtime, providing significant performance improvements for computationally intensive operations.

.. code-block:: python

   from numba import jit
   import numpy as np
   
   @jit(nopython=True, cache=True)
   def optimized_function(x, y):
       # This function will be compiled to machine code
       result = np.zeros_like(x)
       for i in range(len(x)):
           result[i] = x[i] * y[i]
       return result

The utility modules make extensive use of NumPy's vectorized operations and broadcasting capabilities, further enhancing performance for large datasets. Where vectorization is not possible, Numba acceleration provides near-native performance while maintaining the readability and flexibility of Python code.

Type Safety
=========

All utility functions include comprehensive type hints that enable static type checking with tools like mypy. These type annotations serve as both documentation and runtime validation, helping prevent type-related errors and providing clear guidance on expected parameter types and return values.

.. code-block:: python

   from typing import Union, Optional, Tuple
   import numpy as np
   import pandas as pd
   
   def example_utility(
       data: Union[np.ndarray, pd.Series, pd.DataFrame],
       param: float,
       options: Optional[dict] = None
   ) -> Tuple[np.ndarray, float]:
       # Function implementation with type safety
       pass

The combination of explicit type hints and runtime validation ensures robust behavior across different input types and helps catch potential issues during development rather than at runtime.

Integration with Scientific Python Ecosystem
=========================================

The utility functions are designed to work seamlessly with the broader Python scientific ecosystem, particularly NumPy, SciPy, and Pandas. They accept and return standard data structures from these libraries, enabling easy integration with existing workflows and third-party packages.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mfe.utils.data_transformations import standardize
   
   # Works with NumPy arrays
   array_data = np.random.randn(100, 3)
   std_array = standardize(array_data)
   
   # Also works with Pandas objects
   df_data = pd.DataFrame(array_data, columns=['A', 'B', 'C'])
   std_df = standardize(df_data)  # Preserves DataFrame structure and index

This integration ensures that the MFE Toolbox can be easily incorporated into diverse analytical workflows while leveraging the strengths of the Python scientific computing ecosystem.
