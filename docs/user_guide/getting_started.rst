===============
Getting Started
===============

This guide provides an introduction to the MFE Toolbox, a comprehensive Python-based suite for financial econometrics, time series analysis, and risk modeling. It will help you understand the basic concepts, module structure, and how to use the most common features of the toolbox.

Introduction
===========

The MFE Toolbox is a Python library that provides tools for:

- Univariate volatility modeling (GARCH, EGARCH, TARCH, etc.)
- Multivariate volatility modeling (BEKK, DCC, RARCH, etc.)
- ARMA/ARMAX time series modeling and forecasting
- Bootstrap methods for dependent data
- Non-parametric volatility estimation (realized volatility)
- Classical statistical tests and distributions
- Vector autoregression (VAR) analysis
- Principal component analysis and cross-sectional econometrics

Package Structure
===============

The MFE Toolbox is organized into several modules:

- ``mfe.core``: Core functionality and base classes
- ``mfe.models``: Implementation of econometric models
  - ``mfe.models.bootstrap``: Bootstrap methods for dependent time series
  - ``mfe.models.cross_section``: Cross-sectional data analysis tools
  - ``mfe.models.distributions``: Statistical distribution functions
  - ``mfe.models.multivariate``: Multivariate volatility models
  - ``mfe.models.realized``: High-frequency financial econometrics tools
  - ``mfe.models.time_series``: Time series analysis toolkit
  - ``mfe.models.univariate``: Univariate volatility models
- ``mfe.ui``: User interface components including PyQt6-based ARMAX modeling environment
- ``mfe.utils``: Helper functions for data transformation and analysis

Importing the Toolbox
===================

To use the MFE Toolbox, you first need to import it in your Python script or notebook:

.. code-block:: python

    import mfe

You can check the installed version:

.. code-block:: python

    print(mfe.__version__)  # Outputs: 4.0.0

For specific components, you can import them directly:

.. code-block:: python

    # Import specific models
    from mfe.models.univariate import GARCH
    from mfe.models.time_series import ARMA
    from mfe.models.multivariate import DCC
    
    # Import utility functions
    from mfe.utils.matrix_ops import vech, ivech
    
    # Import statistical distributions
    from mfe.models.distributions import StudentT, Normal

Basic Usage Examples
==================

Here are some basic examples to get you started with the MFE Toolbox.

Working with Data
---------------

The MFE Toolbox works with NumPy arrays and Pandas DataFrames:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Create or load financial time series data
    # Option 1: Using NumPy arrays
    np.random.seed(42)
    returns = np.random.normal(0, 1, 1000) * 0.01  # 1000 daily returns
    
    # Option 2: Using Pandas DataFrames with dates
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='B')
    returns_df = pd.DataFrame({
        'returns': returns
    }, index=dates)
    
    # Plot the returns
    plt.figure(figsize=(10, 6))
    plt.plot(returns_df.index, returns_df['returns'])
    plt.title('Daily Returns')
    plt.ylabel('Return')
    plt.show()

Univariate Volatility Modeling
----------------------------

Estimate a GARCH(1,1) model:

.. code-block:: python

    from mfe.models.univariate import GARCH
    from mfe.models.distributions import StudentT
    
    # Create a GARCH(1,1) model with Student's t distribution
    model = GARCH(p=1, q=1, error_dist=StudentT())
    
    # Fit the model to the data
    result = model.fit(returns)
    
    # Print model summary
    print(result.summary())
    
    # Access model parameters
    omega = result.params.omega
    alpha = result.params.alpha
    beta = result.params.beta
    df = result.params.df  # Degrees of freedom for Student's t
    
    print(f"GARCH(1,1) Parameters: omega={omega:.6f}, alpha={alpha:.6f}, beta={beta:.6f}, df={df:.6f}")
    
    # Plot conditional volatility
    plt.figure(figsize=(10, 6))
    plt.plot(np.sqrt(result.conditional_variance))
    plt.title('GARCH(1,1) Conditional Volatility')
    plt.ylabel('Volatility')
    plt.show()

Time Series Analysis
-----------------

Estimate an ARMA(2,1) model:

.. code-block:: python

    from mfe.models.time_series import ARMA
    
    # Create an ARMA(2,1) model
    arma_model = ARMA(ar_order=2, ma_order=1, include_constant=True)
    
    # Fit the model to the data
    arma_result = arma_model.fit(returns)
    
    # Print model summary
    print(arma_result.summary())
    
    # Generate forecasts
    forecasts = arma_result.forecast(horizon=10)
    print("10-day ahead forecasts:")
    print(forecasts.mean)
    
    # Plot forecasts with confidence intervals
    forecasts.plot()
    plt.title('ARMA(2,1) Forecasts')
    plt.show()

Multivariate Volatility Modeling
-----------------------------

Estimate a DCC model for multiple assets:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from mfe.models.multivariate import DCC
    from mfe.models.univariate import GARCH
    
    # Generate multivariate returns data (2 assets)
    np.random.seed(42)
    n = 1000  # Number of observations
    returns_multi = np.random.multivariate_normal(
        mean=[0, 0],
        cov=[[1.0, 0.5], [0.5, 1.0]],
        size=n
    ) * 0.01
    
    # Create a DCC model with GARCH(1,1) for each series
    dcc_model = DCC(
        univariate_models=[GARCH(p=1, q=1), GARCH(p=1, q=1)],
        correlation_type='DCC'
    )
    
    # Fit the model
    dcc_result = dcc_model.fit(returns_multi)
    
    # Print model summary
    print(dcc_result.summary())
    
    # Access time-varying correlations
    correlations = dcc_result.conditional_correlations
    
    # Plot the time-varying correlation
    plt.figure(figsize=(10, 6))
    plt.plot(correlations[:, 0, 1])  # Correlation between asset 1 and 2
    plt.title('DCC Time-varying Correlation')
    plt.ylabel('Correlation')
    plt.show()

Bootstrap Methods
--------------

Implement block bootstrap for time series:

.. code-block:: python

    from mfe.models.bootstrap import BlockBootstrap
    
    # Create a block bootstrap with block size 50
    bootstrap = BlockBootstrap(block_size=50)
    
    # Generate 1000 bootstrap samples
    bootstrap_samples = bootstrap.generate(returns, num_samples=1000)
    
    # Compute bootstrap statistics (e.g., mean of each sample)
    bootstrap_means = np.array([sample.mean() for sample in bootstrap_samples])
    
    # Plot the bootstrap distribution
    plt.figure(figsize=(10, 6))
    plt.hist(bootstrap_means, bins=50)
    plt.axvline(returns.mean(), color='r', linestyle='--', label='Sample Mean')
    plt.title('Block Bootstrap Distribution of Mean')
    plt.xlabel('Mean')
    plt.legend()
    plt.show()
    
    # Compute bootstrap confidence interval
    conf_interval = np.percentile(bootstrap_means, [2.5, 97.5])
    print(f"95% Bootstrap Confidence Interval: [{conf_interval[0]:.6f}, {conf_interval[1]:.6f}]")

Realized Volatility Estimation
---------------------------

Estimate realized volatility from high-frequency data:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from mfe.models.realized import RealizedVariance
    
    # Generate simulated high-frequency price data
    np.random.seed(42)
    n_days = 10
    n_intraday = 100  # 100 observations per day
    
    # Create timestamps (10 days, 100 observations per day)
    timestamps = []
    for day in range(n_days):
        for i in range(n_intraday):
            # 9:30 AM to 4:00 PM (390 minutes = 6.5 hours)
            minute = 9*60 + 30 + i * (6.5*60 / n_intraday)
            hour = int(minute // 60)
            minute = int(minute % 60)
            timestamps.append(pd.Timestamp(f'2023-01-{day+1:02d} {hour:02d}:{minute:02d}:00'))
    
    # Generate random walk for prices
    log_prices = np.cumsum(np.random.normal(0, 0.001, n_days * n_intraday))
    prices = np.exp(log_prices)
    
    # Create DataFrame
    hf_data = pd.DataFrame({
        'price': prices
    }, index=timestamps)
    
    # Create a realized variance estimator
    rv_estimator = RealizedVariance()
    
    # Estimate daily realized variance
    rv = rv_estimator.compute(
        prices=hf_data['price'].values,
        timestamps=hf_data.index.values,
        sampling='5min'  # 5-minute sampling
    )
    
    # Print results
    print("Daily Realized Variance Estimates:")
    for day, variance in enumerate(rv):
        print(f"Day {day+1}: {variance:.8f} (Volatility: {np.sqrt(variance):.4f})")
    
    # Plot realized volatility
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, n_days+1), np.sqrt(rv))
    plt.title('Daily Realized Volatility')
    plt.xlabel('Day')
    plt.ylabel('Realized Volatility')
    plt.show()

Asynchronous Processing
--------------------

For long-running computations, the MFE Toolbox supports asynchronous processing:

.. code-block:: python

    import asyncio
    from mfe.models.bootstrap import BlockBootstrap
    
    async def run_bootstrap_async():
        # Create a block bootstrap with block size 50
        bootstrap = BlockBootstrap(block_size=50)
        
        # Generate 10,000 bootstrap samples asynchronously with progress reporting
        def progress_callback(percent, message):
            print(f"{percent:.1f}% complete: {message}")
        
        bootstrap_samples = await bootstrap.generate_async(
            returns, 
            num_samples=10000,
            progress_callback=progress_callback
        )
        
        # Compute bootstrap statistics
        bootstrap_means = np.array([sample.mean() for sample in bootstrap_samples])
        conf_interval = np.percentile(bootstrap_means, [2.5, 97.5])
        
        return conf_interval
    
    # Run the async function
    conf_interval = asyncio.run(run_bootstrap_async())
    print(f"95% Bootstrap Confidence Interval: [{conf_interval[0]:.6f}, {conf_interval[1]:.6f}]")

Using Type Hints for Better IDE Support
------------------------------------

The MFE Toolbox uses type hints throughout the codebase, which provides better IDE support and code completion:

.. code-block:: python

    from typing import List, Optional, Union, Tuple
    import numpy as np
    import pandas as pd
    from mfe.models.univariate import GARCH
    from mfe.models.distributions import Normal
    
    def estimate_garch(
        returns: Union[np.ndarray, pd.Series],
        p: int = 1,
        q: int = 1,
        include_mean: bool = False
    ) -> Tuple[GARCH, float, float, float]:
        """
        Estimate a GARCH(p,q) model and return key parameters.
        
        Args:
            returns: Array of return data
            p: ARCH order
            q: GARCH order
            include_mean: Whether to include a mean term
            
        Returns:
            Tuple containing (model_result, omega, alpha, beta)
        """
        # Convert pandas Series to numpy array if needed
        if isinstance(returns, pd.Series):
            returns_array = returns.values
        else:
            returns_array = returns
            
        # Create and fit the model
        model = GARCH(p=p, q=q, mean=include_mean, error_dist=Normal())
        result = model.fit(returns_array)
        
        # Extract parameters
        omega = result.params.omega
        alpha = result.params.alpha
        beta = result.params.beta
        
        return result, omega, alpha, beta
    
    # Using the function with type hints
    result, omega, alpha, beta = estimate_garch(
        returns=returns_df['returns'],
        p=1,
        q=1,
        include_mean=True
    )
    
    print(f"GARCH(1,1) Parameters: omega={omega:.6f}, alpha={alpha:.6f}, beta={beta:.6f}")

Next Steps
=========

Now that you're familiar with the basics of the MFE Toolbox, you can explore more advanced features:

- :doc:`univariate_volatility_models` - Learn about GARCH, EGARCH, TARCH, and other univariate volatility models
- :doc:`multivariate_volatility_models` - Explore BEKK, DCC, and other multivariate volatility models
- :doc:`time_series_analysis` - Discover ARMA/ARMAX modeling and forecasting
- :doc:`bootstrap_methods` - Learn about bootstrap techniques for dependent data
- :doc:`high_frequency_econometrics` - Explore realized volatility estimation
- :doc:`statistical_distributions` - Understand the available probability distributions
- :doc:`statistical_tests` - Learn about statistical tests for model validation
- :doc:`cross_sectional_analysis` - Discover OLS regression and PCA tools
- :doc:`gui_interface` - Explore the PyQt6-based ARMAX modeling interface
