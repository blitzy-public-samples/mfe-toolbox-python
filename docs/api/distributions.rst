.. _api_distributions:

==========================
Distributions (mfe.models.distributions)
==========================

The ``mfe.models.distributions`` package provides implementations of statistical distributions used throughout the MFE Toolbox. These distributions are essential for likelihood-based estimation in volatility models, time series analysis, and other econometric applications.

The package extends SciPy's distribution framework with specialized implementations optimized for financial econometrics, including Numba-accelerated density and quantile functions for performance-critical operations.

Base Distribution
================

.. automodule:: mfe.models.distributions.base
   :members:
   :undoc-members:
   :show-inheritance:

The base module defines the abstract base class that establishes the common interface for all distribution implementations. This ensures consistent behavior across different distribution types and provides a foundation for extending the distribution framework.

.. code-block:: python

   from mfe.models.distributions.base import Distribution
   
   # Example of extending the base distribution class
   class CustomDistribution(Distribution):
       def __init__(self, parameters):
           super().__init__(parameters)
           # Custom initialization
       
       def pdf(self, x):
           # Implementation of probability density function
           pass
       
       def cdf(self, x):
           # Implementation of cumulative distribution function
           pass
       
       def ppf(self, q):
           # Implementation of percent point function (inverse CDF)
           pass
       
       def loglikelihood(self, x):
           # Implementation of log-likelihood function
           pass
       
       def random(self, size):
           # Implementation of random number generation
           pass

Normal Distribution
=================

.. automodule:: mfe.models.distributions.normal
   :members:
   :undoc-members:
   :show-inheritance:

The normal distribution implementation provides standard normal and location-scale normal distributions with Numba-accelerated density and quantile functions. This distribution is commonly used as a baseline in financial modeling and as a component in more complex distributions.

.. code-block:: python

   from mfe.models.distributions.normal import Normal
   import numpy as np
   
   # Create a standard normal distribution
   norm = Normal()
   
   # Generate random samples
   samples = norm.random(1000)
   
   # Compute PDF values
   x = np.linspace(-3, 3, 100)
   pdf_values = norm.pdf(x)
   
   # Compute log-likelihood
   log_lik = norm.loglikelihood(samples)

Student's t Distribution
=====================

.. automodule:: mfe.models.distributions.student_t
   :members:
   :undoc-members:
   :show-inheritance:

The Student's t distribution implementation provides standardized t-distributions with configurable degrees of freedom. This distribution is widely used in financial econometrics to model heavy-tailed return distributions.

.. code-block:: python

   from mfe.models.distributions.student_t import StudentsT
   import numpy as np
   
   # Create a t-distribution with 5 degrees of freedom
   t_dist = StudentsT(nu=5.0)
   
   # Generate random samples
   samples = t_dist.random(1000)
   
   # Compute PDF values
   x = np.linspace(-4, 4, 100)
   pdf_values = t_dist.pdf(x)
   
   # Compute quantiles (inverse CDF)
   q = np.array([0.025, 0.5, 0.975])
   quantiles = t_dist.ppf(q)

Generalized Error Distribution
============================

.. automodule:: mfe.models.distributions.generalized_error
   :members:
   :undoc-members:
   :show-inheritance:

The Generalized Error Distribution (GED) implementation provides a flexible distribution family that includes the normal distribution as a special case. The shape parameter controls the tail thickness, allowing for both thinner and thicker tails than the normal distribution.

.. code-block:: python

   from mfe.models.distributions.generalized_error import GeneralizedError
   import numpy as np
   
   # Create a GED with shape parameter 1.5
   # (shape=2 corresponds to normal, shape<2 gives heavier tails)
   ged = GeneralizedError(nu=1.5)
   
   # Generate random samples
   samples = ged.random(1000)
   
   # Compute PDF values
   x = np.linspace(-4, 4, 100)
   pdf_values = ged.pdf(x)
   
   # Compute log-likelihood
   log_lik = ged.loglikelihood(samples)

Skewed t Distribution
==================

.. automodule:: mfe.models.distributions.skewed_t
   :members:
   :undoc-members:
   :show-inheritance:

The skewed t distribution implementation provides Hansen's skewed t-distribution, which extends the Student's t distribution with an additional skewness parameter. This distribution is particularly useful for modeling financial returns that exhibit both heavy tails and asymmetry.

.. code-block:: python

   from mfe.models.distributions.skewed_t import SkewedT
   import numpy as np
   
   # Create a skewed t-distribution with 5 degrees of freedom
   # and negative skewness (-0.2)
   skewed_t = SkewedT(nu=5.0, lambda_=-0.2)
   
   # Generate random samples
   samples = skewed_t.random(1000)
   
   # Compute PDF values
   x = np.linspace(-4, 4, 100)
   pdf_values = skewed_t.pdf(x)
   
   # Compute CDF values
   cdf_values = skewed_t.cdf(x)

Composite Likelihood
=================

.. automodule:: mfe.models.distributions.composite_likelihood
   :members:
   :undoc-members:
   :show-inheritance:

The composite likelihood module provides utilities for constructing and evaluating composite likelihood functions. These are particularly useful for high-dimensional problems where full likelihood evaluation is computationally intensive.

.. code-block:: python

   from mfe.models.distributions.composite_likelihood import pairwise_likelihood
   import numpy as np
   
   # Example of using pairwise likelihood with a multivariate dataset
   def pairwise_loglikelihood(parameters, data):
       # Compute pairwise loglikelihood for a custom model
       return pairwise_likelihood(parameters, data, custom_pair_loglik_function)
   
   # Optimize parameters using composite likelihood
   from scipy.optimize import minimize
   result = minimize(
       lambda params: -pairwise_loglikelihood(params, data),
       initial_parameters
   )

Utility Functions
================

.. automodule:: mfe.models.distributions.utils
   :members:
   :undoc-members:
   :show-inheritance:

The utils module provides helper functions for distribution-related operations, including parameter transformation, validation, and specialized numerical routines optimized for distribution computations.

.. code-block:: python

   from mfe.models.distributions.utils import transform_params, validate_distribution_params
   
   # Transform parameters from constrained to unconstrained space
   unconstrained_params = transform_params(constrained_params, "student_t")
   
   # Validate distribution parameters
   validate_distribution_params(params, distribution_type="ged")

Performance Considerations
========================

Many distribution functions in the MFE Toolbox are performance-critical, as they are called repeatedly during model estimation. To optimize performance, these functions are accelerated using Numba's @jit decorators, which compile Python code to optimized machine code at runtime.

.. code-block:: python

   # Example of a Numba-accelerated PDF function
   from numba import jit
   import numpy as np
   
   @jit(nopython=True, cache=True)
   def normal_pdf(x, mu=0.0, sigma=1.0):
       """Numba-accelerated normal PDF computation."""
       z = (x - mu) / sigma
       return np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))

The distribution implementations in the MFE Toolbox are designed to work efficiently with both scalar inputs and NumPy arrays, leveraging vectorized operations where possible for improved performance with large datasets.

Type Safety
===========

All distribution classes in the MFE Toolbox include comprehensive type hints that enable static type checking with tools like mypy. These type hints serve as both documentation and runtime validation, helping prevent type-related errors and providing clear guidance on expected parameter types and return values.

.. code-block:: python

   from typing import Union, Optional
   import numpy as np
   from dataclasses import dataclass
   
   @dataclass
   class DistributionParams:
       """Parameter container with validation."""
       nu: float
       
       def __post_init__(self) -> None:
           if self.nu <= 2.0:
               raise ValueError("Degrees of freedom must be greater than 2")
   
   class ExampleDistribution:
       def __init__(self, params: Optional[DistributionParams] = None) -> None:
           self.params = params or DistributionParams(nu=5.0)
       
       def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
           """Compute probability density function.
           
           Args:
               x: Input values, either scalar or array
               
           Returns:
               PDF values with same shape as input
           """
           # Implementation
           pass

Integration with SciPy
====================

The distribution implementations in the MFE Toolbox extend SciPy's distribution framework where possible, providing a familiar interface for users of SciPy's statistical functions while adding specialized capabilities for financial econometrics.

.. code-block:: python

   from scipy import stats
   from mfe.models.distributions.student_t import StudentsT
   
   # SciPy's t-distribution
   scipy_t = stats.t(df=5)
   
   # MFE Toolbox's t-distribution
   mfe_t = StudentsT(nu=5.0)
   
   # Both provide similar interfaces
   x = 1.5
   scipy_pdf = scipy_t.pdf(x)
   mfe_pdf = mfe_t.pdf(x)
