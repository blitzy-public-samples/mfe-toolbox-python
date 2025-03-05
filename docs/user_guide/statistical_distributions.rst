# docs/user_guide/statistical_distributions.rst
```rst
.. _statistical_distributions:

Statistical Distributions
========================

Introduction
-----------

The MFE Toolbox provides a comprehensive set of statistical distributions commonly used in financial econometrics. These distributions are implemented as Python classes that extend SciPy's distribution framework, providing consistent interfaces for density, distribution, quantile, and random number generation functions.

The distributions are particularly useful for:

* Modeling non-Gaussian error distributions in financial time series
* Maximum likelihood estimation in volatility models
* Value-at-Risk (VaR) and Expected Shortfall (ES) calculations
* Monte Carlo simulations with realistic return distributions

All distributions in the MFE Toolbox follow a consistent object-oriented design pattern with:

* Class-based implementation with inheritance from a common base class
* Type-validated parameters using Python dataclasses
* Numba-accelerated core functions for performance-critical operations
* Vectorized operations supporting NumPy arrays and Pandas Series
* Comprehensive error handling with descriptive messages

Available Distributions
----------------------

The MFE Toolbox includes the following distributions:

* :ref:`normal_distribution`: The standard normal (Gaussian) distribution
* :ref:`student_t_distribution`: The standardized Student's t-distribution for modeling heavy tails
* :ref:`ged_distribution`: The Generalized Error Distribution (GED) for flexible tail modeling
* :ref:`skewed_t_distribution`: Hansen's skewed t-distribution for asymmetric heavy-tailed data

Each distribution provides methods for:

* Probability density function (PDF)
* Cumulative distribution function (CDF)
* Quantile function (inverse CDF)
* Random number generation
* Log-likelihood calculation

.. _normal_distribution:

Normal Distribution
------------------

The normal distribution is implemented in the ``Normal`` class, which provides a standardized interface for working with Gaussian distributions.

.. code-block:: python

    from mfe.models.distributions import Normal
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Create a Normal distribution instance
    normal = Normal()
    
    # Generate sample data
    x = np.linspace(-4, 4, 1000)
    
    # Calculate PDF values (vectorized operation)
    pdf_values = normal.pdf(x)
    
    # Calculate CDF values (vectorized operation)
    cdf_values = normal.cdf(x)
    
    # Plot the results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(x, pdf_values)
    ax1.set_title('Normal PDF')
    ax1.set_xlabel('x')
    ax1.set_ylabel('Density')
    
    ax2.plot(x, cdf_values)
    ax2.set_title('Normal CDF')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Probability')
    
    plt.tight_layout()
    plt.show()

The ``Normal`` class also provides methods for generating random samples and calculating log-likelihood:

.. code-block:: python

    # Generate 1000 random samples
    samples = normal.rvs(size=1000)
    
    # Calculate log-likelihood for a data vector
    data = np.array([0.1, -0.2, 0.3, -0.1, 0.2])
    log_likelihood = normal.loglikelihood(data)
    print(f"Log-likelihood: {log_likelihood:.4f}")

.. _student_t_distribution:

Student's t-Distribution
-----------------------

The standardized Student's t-distribution is implemented in the ``StudentT`` class, which provides methods for working with heavy-tailed distributions.

.. code-block:: python

    from mfe.models.distributions import StudentT
    import numpy as np
    import matplotlib.pyplot as plt
    from dataclasses import dataclass
    
    # Create StudentT distribution instances with different degrees of freedom
    t_dist_3 = StudentT(nu=3)  # 3 degrees of freedom
    t_dist_5 = StudentT(nu=5)  # 5 degrees of freedom
    t_dist_10 = StudentT(nu=10)  # 10 degrees of freedom
    normal = Normal()  # For comparison
    
    # Generate sample data
    x = np.linspace(-4, 4, 1000)
    
    # Calculate PDF values
    pdf_t3 = t_dist_3.pdf(x)
    pdf_t5 = t_dist_5.pdf(x)
    pdf_t10 = t_dist_10.pdf(x)
    pdf_normal = normal.pdf(x)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(x, pdf_normal, 'k-', label='Normal')
    plt.plot(x, pdf_t3, 'r-', label='t(3)')
    plt.plot(x, pdf_t5, 'g-', label='t(5)')
    plt.plot(x, pdf_t10, 'b-', label='t(10)')
    plt.title('Standardized t-Distribution PDF')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

The ``StudentT`` class uses Python's dataclass for parameter validation:

.. code-block:: python

    # Parameter validation happens automatically
    try:
        invalid_t = StudentT(nu=1.5)  # nu must be > 2 for finite variance
    except ValueError as e:
        print(f"Validation error: {e}")
    
    # Generate random samples
    samples = t_dist_5.rvs(size=1000)
    
    # Calculate quantiles (Value-at-Risk)
    var_95 = t_dist_5.ppf(0.05)  # 5% VaR
    print(f"5% VaR with t(5): {var_95:.4f}")

.. _ged_distribution:

Generalized Error Distribution (GED)
-----------------------------------

The Generalized Error Distribution (GED) is implemented in the ``GeneralizedError`` class, providing a flexible distribution with adjustable tail thickness.

.. code-block:: python

    from mfe.models.distributions import GeneralizedError
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Create GED instances with different shape parameters
    ged_1 = GeneralizedError(nu=1.0)  # Laplace distribution
    ged_2 = GeneralizedError(nu=2.0)  # Normal distribution
    ged_5 = GeneralizedError(nu=5.0)  # Thinner tails than normal
    
    # Generate sample data
    x = np.linspace(-4, 4, 1000)
    
    # Calculate PDF values
    pdf_ged1 = ged_1.pdf(x)
    pdf_ged2 = ged_2.pdf(x)
    pdf_ged5 = ged_5.pdf(x)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(x, pdf_ged1, 'r-', label='GED(1.0) - Laplace')
    plt.plot(x, pdf_ged2, 'g-', label='GED(2.0) - Normal')
    plt.plot(x, pdf_ged5, 'b-', label='GED(5.0)')
    plt.title('Generalized Error Distribution PDF')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

The ``GeneralizedError`` class provides Numba-accelerated implementations for performance-critical functions:

.. code-block:: python

    # Performance comparison
    import time
    
    # Generate large data array
    large_data = np.random.randn(100000)
    
    # Time the PDF calculation
    start_time = time.time()
    pdf_values = ged_2.pdf(large_data)
    end_time = time.time()
    
    print(f"Time to compute 100,000 PDF values: {(end_time - start_time)*1000:.2f} ms")
    
    # Calculate log-likelihood for a data vector
    data = np.array([0.1, -0.2, 0.3, -0.1, 0.2])
    log_likelihood = ged_2.loglikelihood(data)
    print(f"Log-likelihood: {log_likelihood:.4f}")

.. _skewed_t_distribution:

Skewed t-Distribution
--------------------

Hansen's skewed t-distribution is implemented in the ``SkewedT`` class, providing a flexible distribution that can model both heavy tails and asymmetry.

.. code-block:: python

    from mfe.models.distributions import SkewedT
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Create SkewedT instances with different parameters
    # nu controls tail thickness, lambda controls asymmetry
    skewt_sym = SkewedT(nu=5, lambda_=0.0)  # Symmetric
    skewt_neg = SkewedT(nu=5, lambda_=-0.5)  # Negative skew
    skewt_pos = SkewedT(nu=5, lambda_=0.5)   # Positive skew
    
    # Generate sample data
    x = np.linspace(-4, 4, 1000)
    
    # Calculate PDF values
    pdf_sym = skewt_sym.pdf(x)
    pdf_neg = skewt_neg.pdf(x)
    pdf_pos = skewt_pos.pdf(x)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(x, pdf_sym, 'k-', label='Symmetric (λ=0)')
    plt.plot(x, pdf_neg, 'r-', label='Negative Skew (λ=-0.5)')
    plt.plot(x, pdf_pos, 'g-', label='Positive Skew (λ=0.5)')
    plt.title('Hansen\'s Skewed t-Distribution PDF')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

The ``SkewedT`` class uses dataclasses for parameter validation and provides comprehensive error handling:

.. code-block:: python

    # Parameter validation happens automatically
    try:
        invalid_skewt = SkewedT(nu=1.5, lambda_=0.0)  # nu must be > 2
    except ValueError as e:
        print(f"Validation error: {e}")
    
    try:
        invalid_skewt = SkewedT(nu=5, lambda_=1.5)  # lambda must be in [-1, 1]
    except ValueError as e:
        print(f"Validation error: {e}")
    
    # Generate random samples
    samples = skewt_pos.rvs(size=1000)
    
    # Calculate quantiles (Value-at-Risk)
    var_95 = skewt_pos.ppf(0.05)  # 5% VaR
    print(f"5% VaR with skewed-t: {var_95:.4f}")

Integration with Volatility Models
---------------------------------

The distribution classes are designed to integrate seamlessly with volatility models in the MFE Toolbox. Here\'s an example of using the Student\'s t-distribution with a GARCH model:

.. code-block:: python

    from mfe.models.univariate import GARCH
    from mfe.models.distributions import StudentT
    import numpy as np
    import pandas as pd
    
    # Load return data
    returns = pd.read_csv('example_returns.csv', index_col=0, parse_dates=True)['returns'].values
    
    # Create a GARCH model with Student\'s t errors
    t_dist = StudentT(nu=5)
    garch_model = GARCH(p=1, q=1, distribution=t_dist)
    
    # Fit the model
    result = garch_model.fit(returns)
    
    # Access the estimated distribution parameters
    estimated_nu = result.distribution_params.nu
    print(f"Estimated degrees of freedom: {estimated_nu:.4f}")
    
    # Generate forecasts with proper error distribution
    forecasts = garch_model.forecast(horizon=10, method='simulation', n_sims=1000)
    
    # Calculate VaR using the fitted distribution
    var_95 = result.conditional_volatility[-1] * t_dist.ppf(0.05)
    print(f"1-day ahead 5% VaR: {var_95:.4f}")

Advanced Usage
-------------

Composite Likelihood
~~~~~~~~~~~~~~~~~~~

For multivariate models, the MFE Toolbox provides a ``CompositeLikelihood`` class that combines multiple univariate distributions:

.. code-block:: python

    from mfe.models.distributions import CompositeLikelihood, StudentT
    import numpy as np
    
    # Create individual distributions
    t_dist1 = StudentT(nu=4)
    t_dist2 = StudentT(nu=6)
    t_dist3 = StudentT(nu=8)
    
    # Create a composite likelihood with three distributions
    composite = CompositeLikelihood([t_dist1, t_dist2, t_dist3])
    
    # Generate multivariate data (3 variables, 5 observations)
    data = np.random.randn(5, 3)
    
    # Calculate the composite log-likelihood
    log_likelihood = composite.loglikelihood(data)
    print(f"Composite log-likelihood: {log_likelihood:.4f}")

Custom Distribution Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The distribution classes use Python\'s dataclasses for parameter management, allowing for custom parameter validation:

.. code-block:: python

    from mfe.models.distributions import StudentT
    from dataclasses import dataclass
    
    @dataclass
    class CustomStudentTParams:
        nu: float
        
        def __post_init__(self):
            if self.nu <= 2:
                raise ValueError("Degrees of freedom must be > 2 for finite variance")
            if self.nu > 100:
                raise ValueError("Degrees of freedom too large, consider using Normal distribution")
    
    # Create a StudentT distribution with custom parameters
    params = CustomStudentTParams(nu=5)
    t_dist = StudentT(params=params)
    
    # Use the distribution
    x = np.linspace(-4, 4, 100)
    pdf_values = t_dist.pdf(x)

Asynchronous Processing
~~~~~~~~~~~~~~~~~~~~~~

For computationally intensive operations, the distribution classes support asynchronous processing:

.. code-block:: python

    import asyncio
    from mfe.models.distributions import StudentT
    import numpy as np
    
    async def calculate_large_loglikelihood():
        # Create a StudentT distribution
        t_dist = StudentT(nu=5)
        
        # Generate large data array
        large_data = np.random.randn(1000000)
        
        # Calculate log-likelihood asynchronously
        return await t_dist.loglikelihood_async(large_data)
    
    # Run the asynchronous function
    log_likelihood = asyncio.run(calculate_large_loglikelihood())
    print(f"Log-likelihood for 1,000,000 observations: {log_likelihood:.4f}")

API Reference
------------

For detailed information on all distribution classes and methods, see the :ref:`API reference <api_distributions>`.

.. seealso::
   
   - :ref:`univariate_volatility_models` - Using distributions with GARCH models
   - :ref:`multivariate_volatility_models` - Using distributions with multivariate models
   - :ref:`bootstrap_methods` - Resampling methods for statistical inference
```