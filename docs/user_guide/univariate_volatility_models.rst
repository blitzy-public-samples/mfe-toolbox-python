============================
Univariate Volatility Models
============================

This guide provides a comprehensive overview of the univariate volatility modeling capabilities in the MFE Toolbox. These models are essential for capturing time-varying volatility in financial time series, which is a key characteristic of financial returns.

Introduction to Volatility Modeling
==================================

Financial returns typically exhibit volatility clustering, where periods of high volatility tend to be followed by high volatility, and periods of low volatility tend to be followed by low volatility. Univariate volatility models capture this behavior by allowing the conditional variance of a time series to evolve over time based on past information.

The general form of a univariate volatility model can be expressed as:

.. math::

    r_t &= \mu_t + \varepsilon_t \\
    \varepsilon_t &= \sigma_t z_t \\
    z_t &\sim D(0,1)

where:

- :math:`r_t` is the return at time :math:`t`
- :math:`\mu_t` is the conditional mean at time :math:`t` (often assumed to be constant or modeled separately)
- :math:`\varepsilon_t` is the innovation at time :math:`t`
- :math:`\sigma_t` is the conditional standard deviation at time :math:`t`
- :math:`z_t` is a standardized random variable following distribution :math:`D` with mean 0 and variance 1

The MFE Toolbox implements various specifications for modeling the conditional variance :math:`\sigma_t^2`, each with different properties and capabilities.

Available Models
==============

The MFE Toolbox provides the following univariate volatility models:

- **GARCH**: Generalized Autoregressive Conditional Heteroskedasticity
- **EGARCH**: Exponential GARCH
- **TARCH**: Threshold ARCH (also known as GJR-GARCH)
- **AGARCH**: Asymmetric GARCH
- **APARCH**: Asymmetric Power ARCH
- **FIGARCH**: Fractionally Integrated GARCH
- **IGARCH**: Integrated GARCH
- **HEAVY**: High-frEquency-bAsed VolatilitY

All models are implemented as Python classes that inherit from a common base class, providing a consistent interface for model specification, estimation, and forecasting.

Model Specifications
==================

GARCH Model
----------

The GARCH(p,q) model is the most widely used volatility model, specified as:

.. math::

    \sigma_t^2 = \omega + \sum_{i=1}^p \alpha_i \varepsilon_{t-i}^2 + \sum_{j=1}^q \beta_j \sigma_{t-j}^2

where:

- :math:`\omega > 0` is the constant term
- :math:`\alpha_i \geq 0` are the ARCH parameters
- :math:`\beta_j \geq 0` are the GARCH parameters
- :math:`\sum_{i=1}^p \alpha_i + \sum_{j=1}^q \beta_j < 1` for stationarity

In Python, you can create and estimate a GARCH(1,1) model as follows:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mfe.models.univariate import GARCH
    from mfe.models.distributions import Normal

    # Create a GARCH(1,1) model with normal distribution
    model = GARCH(p=1, q=1, error_dist=Normal())
    
    # Fit the model to return data
    result = model.fit(returns)
    
    # Print model summary
    print(result.summary())
    
    # Access model parameters
    omega = result.params.omega
    alpha = result.params.alpha
    beta = result.params.beta
    
    # Plot conditional volatility
    plt.figure(figsize=(10, 6))
    plt.plot(np.sqrt(result.conditional_variance))
    plt.title('GARCH(1,1) Conditional Volatility')
    plt.ylabel('Volatility')
    plt.show()

EGARCH Model
-----------

The EGARCH(p,q) model captures asymmetric effects where negative shocks have a different impact on volatility than positive shocks:

.. math::

    \log(\sigma_t^2) = \omega + \sum_{i=1}^p \alpha_i g(z_{t-i}) + \sum_{j=1}^q \beta_j \log(\sigma_{t-j}^2)

where:

- :math:`g(z_t) = \theta z_t + \gamma [|z_t| - E(|z_t|)]`
- :math:`z_t = \varepsilon_t / \sigma_t`
- :math:`\theta` captures the asymmetric effect
- :math:`\gamma` captures the magnitude effect

Example usage:

.. code-block:: python

    from mfe.models.univariate import EGARCH
    from mfe.models.distributions import StudentT
    
    # Create an EGARCH(1,1) model with Student's t distribution
    model = EGARCH(p=1, q=1, error_dist=StudentT())
    
    # Fit the model
    result = model.fit(returns)
    
    # Print model summary
    print(result.summary())
    
    # Access asymmetry parameter
    theta = result.params.theta
    print(f"Asymmetry parameter: {theta:.4f}")
    
    # Check for leverage effect
    if theta < 0:
        print("Negative shocks increase volatility more than positive shocks (leverage effect)")
    else:
        print("No leverage effect detected")

TARCH Model
----------

The TARCH(p,q) model, also known as GJR-GARCH, explicitly models the asymmetric impact of positive and negative shocks:

.. math::

    \sigma_t^2 = \omega + \sum_{i=1}^p (\alpha_i + \gamma_i I_{t-i}) \varepsilon_{t-i}^2 + \sum_{j=1}^q \beta_j \sigma_{t-j}^2

where:

- :math:`I_{t-i} = 1` if :math:`\varepsilon_{t-i} < 0` and 0 otherwise
- :math:`\gamma_i` captures the additional impact of negative shocks

Example usage:

.. code-block:: python

    from mfe.models.univariate import TARCH
    
    # Create a TARCH(1,1) model
    model = TARCH(p=1, q=1)
    
    # Fit the model
    result = model.fit(returns)
    
    # Print model summary
    print(result.summary())
    
    # Access asymmetry parameter
    gamma = result.params.gamma
    print(f"Asymmetry parameter: {gamma:.4f}")
    
    # Interpret the asymmetry
    if gamma > 0:
        print("Negative shocks increase volatility more than positive shocks")
    else:
        print("No asymmetric effect detected")

APARCH Model
-----------

The APARCH(p,q) model introduces a power transformation of the conditional standard deviation:

.. math::

    \sigma_t^\delta = \omega + \sum_{i=1}^p \alpha_i (|\varepsilon_{t-i}| - \gamma_i \varepsilon_{t-i})^\delta + \sum_{j=1}^q \beta_j \sigma_{t-j}^\delta

where:

- :math:`\delta > 0` is the power parameter
- :math:`\gamma_i` captures the asymmetric effect with :math:`|\gamma_i| < 1`

Example usage:

.. code-block:: python

    from mfe.models.univariate import APARCH
    
    # Create an APARCH(1,1) model
    model = APARCH(p=1, q=1)
    
    # Fit the model
    result = model.fit(returns)
    
    # Print model summary
    print(result.summary())
    
    # Access power and asymmetry parameters
    delta = result.params.delta
    gamma = result.params.gamma
    print(f"Power parameter: {delta:.4f}")
    print(f"Asymmetry parameter: {gamma:.4f}")

FIGARCH Model
-----------

The FIGARCH(p,d,q) model incorporates long memory in volatility through fractional integration:

.. math::

    \sigma_t^2 = \omega + [1 - \beta(L) - \phi(L)(1-L)^d] \varepsilon_t^2 + \beta(L) \sigma_t^2

where:

- :math:`L` is the lag operator
- :math:`d` is the fractional integration parameter (0 < d < 1)
- :math:`\phi(L)` and :math:`\beta(L)` are lag polynomials

Example usage:

.. code-block:: python

    from mfe.models.univariate import FIGARCH
    
    # Create a FIGARCH(1,d,1) model
    model = FIGARCH(p=1, q=1)
    
    # Fit the model
    result = model.fit(returns)
    
    # Print model summary
    print(result.summary())
    
    # Access fractional integration parameter
    d = result.params.d
    print(f"Fractional integration parameter: {d:.4f}")
    
    # Interpret long memory
    if d > 0 and d < 0.5:
        print("Long memory in volatility detected")
    elif d >= 0.5:
        print("Strong persistence in volatility")
    else:
        print("No long memory detected")

IGARCH Model
----------

The IGARCH(p,q) model is a special case where the persistence parameters sum to exactly 1:

.. math::

    \sigma_t^2 = \omega + \sum_{i=1}^{p-1} \alpha_i \varepsilon_{t-i}^2 + (1 - \sum_{i=1}^{p-1} \alpha_i) \sigma_{t-1}^2

Example usage:

.. code-block:: python

    from mfe.models.univariate import IGARCH
    
    # Create an IGARCH(1,1) model
    model = IGARCH(p=1, q=1)
    
    # Fit the model
    result = model.fit(returns)
    
    # Print model summary
    print(result.summary())

HEAVY Model
---------

The HEAVY model incorporates realized measures of volatility:

.. math::

    \sigma_t^2 = \omega + \alpha RM_{t-1} + \beta \sigma_{t-1}^2

where :math:`RM_{t-1}` is a realized measure of volatility (e.g., realized variance).

Example usage:

.. code-block:: python

    from mfe.models.univariate import HEAVY
    from mfe.models.realized import RealizedVariance
    
    # Compute realized measures
    rv_estimator = RealizedVariance()
    realized_measures = rv_estimator.compute(
        prices=high_frequency_prices,
        timestamps=high_frequency_timestamps
    )
    
    # Create a HEAVY model
    model = HEAVY()
    
    # Fit the model with both returns and realized measures
    result = model.fit(returns, realized_measures=realized_measures)
    
    # Print model summary
    print(result.summary())

Model Estimation
==============

All univariate volatility models in the MFE Toolbox follow a consistent estimation approach using maximum likelihood estimation (MLE). The estimation process is optimized using Numba's just-in-time compilation for performance-critical operations.

Basic Estimation
--------------

The basic workflow for estimating a univariate volatility model is:

1. Create a model instance with desired parameters
2. Call the `fit()` method with return data
3. Examine the results

.. code-block:: python

    from mfe.models.univariate import GARCH
    from mfe.models.distributions import StudentT
    
    # Create a model
    model = GARCH(p=1, q=1, error_dist=StudentT())
    
    # Fit the model
    result = model.fit(returns)
    
    # Examine results
    print(result.summary())
    
    # Access specific components
    params = result.params
    std_errors = result.std_errors
    t_stats = result.t_stats
    p_values = result.p_values
    log_likelihood = result.log_likelihood
    aic = result.aic
    bic = result.bic
    conditional_variance = result.conditional_variance

Asynchronous Estimation
---------------------

For long-running estimations, the MFE Toolbox provides asynchronous versions of the estimation methods:

.. code-block:: python

    import asyncio
    from mfe.models.univariate import GARCH
    
    async def estimate_model_async():
        # Create a model
        model = GARCH(p=1, q=1)
        
        # Define a progress callback
        def progress_callback(percent, message):
            print(f"{percent:.1f}% complete: {message}")
        
        # Fit the model asynchronously
        result = await model.fit_async(
            returns, 
            progress_callback=progress_callback
        )
        
        return result
    
    # Run the async function
    result = asyncio.run(estimate_model_async())
    
    # Examine results
    print(result.summary())

Custom Starting Values
-------------------

You can provide custom starting values for the optimization:

.. code-block:: python

    from mfe.models.univariate import GARCH
    from mfe.core.parameters import GARCHParams
    
    # Create starting parameter values
    starting_params = GARCHParams(
        omega=0.00001,
        alpha=0.05,
        beta=0.90
    )
    
    # Create and fit the model with custom starting values
    model = GARCH(p=1, q=1)
    result = model.fit(returns, starting_values=starting_params)
    
    print(result.summary())

Error Distributions
----------------

The MFE Toolbox supports various error distributions for volatility models:

- **Normal**: Standard normal distribution
- **StudentT**: Student's t-distribution with estimated degrees of freedom
- **GED**: Generalized Error Distribution
- **SkewedT**: Hansen's skewed t-distribution

Example with Student's t-distribution:

.. code-block:: python

    from mfe.models.univariate import GARCH
    from mfe.models.distributions import StudentT
    
    # Create a GARCH model with Student's t errors
    model = GARCH(p=1, q=1, error_dist=StudentT())
    
    # Fit the model
    result = model.fit(returns)
    
    # Access distribution parameters
    df = result.params.df  # Degrees of freedom
    print(f"Estimated degrees of freedom: {df:.4f}")
    
    # Test for fat tails
    if df < 10:
        print("Evidence of fat tails in the return distribution")
    else:
        print("Return distribution close to normal")

Model Diagnostics
===============

After estimating a volatility model, it's important to check its adequacy through various diagnostic tests.

Standardized Residuals
--------------------

Examining the standardized residuals (:math:`z_t = \varepsilon_t / \sigma_t`) is a key diagnostic:

.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    
    # Get standardized residuals
    std_residuals = result.standardized_residuals
    
    # Plot standardized residuals
    plt.figure(figsize=(12, 8))
    
    # Time series plot
    plt.subplot(2, 2, 1)
    plt.plot(std_residuals)
    plt.title('Standardized Residuals')
    plt.axhline(y=0, color='r', linestyle='-')
    
    # Histogram with normal overlay
    plt.subplot(2, 2, 2)
    plt.hist(std_residuals, bins=50, density=True, alpha=0.6, color='g')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, 0, 1)
    plt.plot(x, p, 'k', linewidth=2)
    plt.title('Histogram of Standardized Residuals')
    
    # QQ plot
    plt.subplot(2, 2, 3)
    stats.probplot(std_residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    
    # ACF of squared residuals
    plt.subplot(2, 2, 4)
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(std_residuals**2, lags=20, title='ACF of Squared Standardized Residuals')
    
    plt.tight_layout()
    plt.show()

Statistical Tests
--------------

The MFE Toolbox provides various statistical tests for model diagnostics:

.. code-block:: python

    from mfe.models.tests import LjungBox, JarqueBera
    
    # Test for serial correlation in standardized residuals
    lb_test = LjungBox(lags=20)
    lb_result = lb_test.run(result.standardized_residuals)
    print("Ljung-Box Test for Serial Correlation:")
    print(f"Test statistic: {lb_result.statistic:.4f}")
    print(f"p-value: {lb_result.p_value:.4f}")
    if lb_result.p_value > 0.05:
        print("No evidence of serial correlation in standardized residuals")
    else:
        print("Evidence of serial correlation in standardized residuals")
    
    # Test for normality of standardized residuals
    jb_test = JarqueBera()
    jb_result = jb_test.run(result.standardized_residuals)
    print("\nJarque-Bera Test for Normality:")
    print(f"Test statistic: {jb_result.statistic:.4f}")
    print(f"p-value: {jb_result.p_value:.4f}")
    if jb_result.p_value > 0.05:
        print("Standardized residuals appear normally distributed")
    else:
        print("Standardized residuals are not normally distributed")

ARCH Effects Test
--------------

Test for remaining ARCH effects in the standardized residuals:

.. code-block:: python

    from mfe.models.tests import LMTest
    
    # Test for ARCH effects
    lm_test = LMTest(lags=10)
    lm_result = lm_test.run(result.standardized_residuals**2)
    print("LM Test for ARCH Effects:")
    print(f"Test statistic: {lm_result.statistic:.4f}")
    print(f"p-value: {lm_result.p_value:.4f}")
    if lm_result.p_value > 0.05:
        print("No evidence of remaining ARCH effects")
    else:
        print("Evidence of remaining ARCH effects")

Model Comparison
-------------

Compare different models using information criteria:

.. code-block:: python

    from mfe.models.univariate import GARCH, EGARCH, TARCH
    
    # Create and fit different models
    models = {
        'GARCH(1,1)': GARCH(p=1, q=1),
        'EGARCH(1,1)': EGARCH(p=1, q=1),
        'TARCH(1,1)': TARCH(p=1, q=1)
    }
    
    results = {}
    for name, model in models.items():
        results[name] = model.fit(returns)
    
    # Compare models using information criteria
    print("Model Comparison:")
    print(f"{'Model':<12} {'Log-Likelihood':<15} {'AIC':<10} {'BIC':<10}")
    print("-" * 47)
    for name, result in results.items():
        print(f"{name:<12} {result.log_likelihood:<15.4f} {result.aic:<10.4f} {result.bic:<10.4f}")
    
    # Find the best model according to AIC
    best_aic = min(results.items(), key=lambda x: x[1].aic)
    print(f"\nBest model according to AIC: {best_aic[0]}")
    
    # Find the best model according to BIC
    best_bic = min(results.items(), key=lambda x: x[1].bic)
    print(f"Best model according to BIC: {best_bic[0]}")

Forecasting
=========

Volatility forecasting is a key application of these models. The MFE Toolbox provides comprehensive forecasting capabilities.

Point Forecasts
-------------

Generate point forecasts for future volatility:

.. code-block:: python

    from mfe.models.univariate import GARCH
    
    # Create and fit a GARCH model
    model = GARCH(p=1, q=1)
    result = model.fit(returns)
    
    # Generate 10-day ahead volatility forecasts
    forecasts = result.forecast(horizon=10)
    
    # Print volatility forecasts
    print("Volatility Forecasts (Standard Deviations):")
    for h in range(10):
        print(f"h={h+1}: {np.sqrt(forecasts.variance[h]):.6f}")
    
    # Plot forecasts
    plt.figure(figsize=(10, 6))
    
    # Historical volatility
    plt.plot(np.sqrt(result.conditional_variance), label='In-sample Volatility')
    
    # Forecast volatility
    forecast_index = np.arange(len(result.conditional_variance), 
                              len(result.conditional_variance) + 10)
    plt.plot(forecast_index, np.sqrt(forecasts.variance), 'r--', label='Forecast Volatility')
    
    plt.title('GARCH(1,1) Volatility Forecast')
    plt.xlabel('Time')
    plt.ylabel('Volatility (Standard Deviation)')
    plt.legend()
    plt.show()

Simulation-Based Forecasts
------------------------

For more accurate forecasts, especially at longer horizons, simulation-based methods are recommended:

.. code-block:: python

    # Generate simulation-based forecasts
    sim_forecasts = result.forecast(horizon=10, method='simulation', num_simulations=10000)
    
    # Print mean forecasts
    print("Simulation-Based Volatility Forecasts (Standard Deviations):")
    for h in range(10):
        print(f"h={h+1}: {np.sqrt(sim_forecasts.variance[h]):.6f}")
    
    # Plot forecasts with confidence intervals
    plt.figure(figsize=(10, 6))
    
    # Historical volatility
    plt.plot(np.sqrt(result.conditional_variance), label='In-sample Volatility')
    
    # Forecast volatility
    forecast_index = np.arange(len(result.conditional_variance), 
                              len(result.conditional_variance) + 10)
    plt.plot(forecast_index, np.sqrt(sim_forecasts.variance), 'r--', 
             label='Forecast Volatility')
    
    # 95% confidence intervals
    plt.fill_between(forecast_index, 
                    np.sqrt(sim_forecasts.variance_lower), 
                    np.sqrt(sim_forecasts.variance_upper), 
                    color='r', alpha=0.2, label='95% Confidence Interval')
    
    plt.title('GARCH(1,1) Simulation-Based Volatility Forecast')
    plt.xlabel('Time')
    plt.ylabel('Volatility (Standard Deviation)')
    plt.legend()
    plt.show()

Asynchronous Forecasting
----------------------

For long-horizon forecasts or large simulation counts, asynchronous forecasting is available:

.. code-block:: python

    import asyncio
    
    async def generate_forecasts_async():
        # Create and fit a GARCH model
        model = GARCH(p=1, q=1)
        result = model.fit(returns)
        
        # Define a progress callback
        def progress_callback(percent, message):
            print(f"{percent:.1f}% complete: {message}")
        
        # Generate simulation-based forecasts asynchronously
        forecasts = await result.forecast_async(
            horizon=30, 
            method='simulation', 
            num_simulations=50000,
            progress_callback=progress_callback
        )
        
        return forecasts
    
    # Run the async function
    forecasts = asyncio.run(generate_forecasts_async())
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(np.sqrt(forecasts.variance), 'r-', label='Mean Forecast')
    plt.fill_between(range(len(forecasts.variance)), 
                    np.sqrt(forecasts.variance_lower), 
                    np.sqrt(forecasts.variance_upper), 
                    color='r', alpha=0.2, label='95% Confidence Interval')
    plt.title('Long-Horizon GARCH Volatility Forecast')
    plt.xlabel('Horizon')
    plt.ylabel('Volatility (Standard Deviation)')
    plt.legend()
    plt.show()

Value-at-Risk Forecasting
-----------------------

Volatility models are often used for Value-at-Risk (VaR) estimation:

.. code-block:: python

    from mfe.models.univariate import GARCH
    from mfe.models.distributions import StudentT
    from scipy import stats
    
    # Create and fit a GARCH model with Student's t distribution
    model = GARCH(p=1, q=1, error_dist=StudentT())
    result = model.fit(returns)
    
    # Generate 1-day ahead forecast
    forecast = result.forecast(horizon=1)
    
    # Calculate 1-day 99% VaR
    # For Student's t, we need the quantile from the t-distribution
    df = result.params.df  # Degrees of freedom
    t_quantile = stats.t.ppf(0.01, df)  # 1% quantile
    
    # VaR calculation (assuming zero mean)
    var_99 = t_quantile * np.sqrt(forecast.variance[0])
    
    print(f"1-day ahead 99% VaR: {var_99:.6f}")
    
    # For comparison, calculate VaR assuming normal distribution
    normal_quantile = stats.norm.ppf(0.01)  # 1% quantile
    var_99_normal = normal_quantile * np.sqrt(forecast.variance[0])
    
    print(f"1-day ahead 99% VaR (normal assumption): {var_99_normal:.6f}")
    
    # If the t-distribution VaR is more negative, it indicates fatter tails
    if var_99 < var_99_normal:
        print("The t-distribution VaR is more conservative due to fat tails")

Model Simulation
=============

The MFE Toolbox allows you to simulate data from estimated volatility models:

.. code-block:: python

    from mfe.models.univariate import GARCH
    
    # Create a GARCH model with specific parameters
    model = GARCH(p=1, q=1)
    
    # Set parameters manually
    from mfe.core.parameters import GARCHParams
    params = GARCHParams(omega=0.00001, alpha=0.05, beta=0.90)
    
    # Simulate 1000 observations
    simulated_data = model.simulate(
        params=params,
        num_obs=1000,
        burn=500,  # Burn-in period to remove initialization effects
        initial_value=None  # Use default initialization
    )
    
    # Plot simulated returns and volatility
    plt.figure(figsize=(12, 8))
    
    # Returns
    plt.subplot(2, 1, 1)
    plt.plot(simulated_data.returns)
    plt.title('Simulated Returns from GARCH(1,1)')
    plt.ylabel('Returns')
    
    # Volatility
    plt.subplot(2, 1, 2)
    plt.plot(np.sqrt(simulated_data.conditional_variance))
    plt.title('Simulated Volatility from GARCH(1,1)')
    plt.ylabel('Volatility')
    plt.xlabel('Time')
    
    plt.tight_layout()
    plt.show()

Asynchronous Simulation
--------------------

For large simulations, asynchronous processing is available:

.. code-block:: python

    import asyncio
    
    async def simulate_garch_async():
        # Create a GARCH model
        model = GARCH(p=1, q=1)
        
        # Set parameters
        params = GARCHParams(omega=0.00001, alpha=0.05, beta=0.90)
        
        # Define a progress callback
        def progress_callback(percent, message):
            print(f"{percent:.1f}% complete: {message}")
        
        # Simulate data asynchronously
        simulated_data = await model.simulate_async(
            params=params,
            num_obs=10000,
            burn=1000,
            progress_callback=progress_callback
        )
        
        return simulated_data
    
    # Run the async function
    simulated_data = asyncio.run(simulate_garch_async())
    
    # Plot a sample of the simulated data
    plt.figure(figsize=(12, 6))
    plt.plot(simulated_data.returns[:1000])
    plt.title('Sample of Simulated Returns from GARCH(1,1)')
    plt.xlabel('Time')
    plt.ylabel('Returns')
    plt.show()

Monte Carlo Analysis
-----------------

Perform Monte Carlo analysis to study model properties:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from mfe.models.univariate import GARCH
    from mfe.core.parameters import GARCHParams
    
    # Define true parameters
    true_params = GARCHParams(omega=0.00001, alpha=0.05, beta=0.90)
    
    # Create a GARCH model
    model = GARCH(p=1, q=1)
    
    # Number of Monte Carlo replications
    n_replications = 100
    
    # Store estimated parameters
    estimated_params = {
        'omega': np.zeros(n_replications),
        'alpha': np.zeros(n_replications),
        'beta': np.zeros(n_replications)
    }
    
    # Perform Monte Carlo simulation
    for i in range(n_replications):
        # Simulate data from the true model
        sim_data = model.simulate(
            params=true_params,
            num_obs=1000,
            burn=500
        )
        
        # Estimate the model on simulated data
        try:
            result = model.fit(sim_data.returns)
            
            # Store estimated parameters
            estimated_params['omega'][i] = result.params.omega
            estimated_params['alpha'][i] = result.params.alpha
            estimated_params['beta'][i] = result.params.beta
            
        except Exception as e:
            print(f"Estimation failed for replication {i}: {e}")
    
    # Plot the distribution of estimated parameters
    plt.figure(figsize=(15, 5))
    
    # Omega
    plt.subplot(1, 3, 1)
    plt.hist(estimated_params['omega'], bins=20)
    plt.axvline(true_params.omega, color='r', linestyle='--', label='True Value')
    plt.title('Distribution of Estimated Omega')
    plt.legend()
    
    # Alpha
    plt.subplot(1, 3, 2)
    plt.hist(estimated_params['alpha'], bins=20)
    plt.axvline(true_params.alpha, color='r', linestyle='--', label='True Value')
    plt.title('Distribution of Estimated Alpha')
    plt.legend()
    
    # Beta
    plt.subplot(1, 3, 3)
    plt.hist(estimated_params['beta'], bins=20)
    plt.axvline(true_params.beta, color='r', linestyle='--', label='True Value')
    plt.title('Distribution of Estimated Beta')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Calculate bias and RMSE
    print("Parameter Estimation Performance:")
    print(f"{'Parameter':<10} {'True Value':<12} {'Mean Estimate':<15} {'Bias':<10} {'RMSE':<10}")
    print("-" * 60)
    
    for param in ['omega', 'alpha', 'beta']:
        true_value = getattr(true_params, param)
        mean_estimate = np.mean(estimated_params[param])
        bias = mean_estimate - true_value
        rmse = np.sqrt(np.mean((estimated_params[param] - true_value)**2))
        
        print(f"{param:<10} {true_value:<12.6f} {mean_estimate:<15.6f} {bias:<10.6f} {rmse:<10.6f}")

Advanced Topics
=============

Model Selection
-------------

Automated model selection using information criteria:

.. code-block:: python

    from mfe.models.univariate import GARCH, EGARCH, TARCH
    import itertools
    
    # Define model types to consider
    model_classes = {
        'GARCH': GARCH,
        'EGARCH': EGARCH,
        'TARCH': TARCH
    }
    
    # Define orders to consider
    p_values = [1, 2]
    q_values = [1, 2]
    
    # Store results
    model_results = []
    
    # Estimate all model combinations
    for model_name, model_class in model_classes.items():
        for p, q in itertools.product(p_values, q_values):
            try:
                # Create and fit the model
                model = model_class(p=p, q=q)
                result = model.fit(returns)
                
                # Store results
                model_results.append({
                    'name': f"{model_name}({p},{q})",
                    'result': result,
                    'aic': result.aic,
                    'bic': result.bic,
                    'log_likelihood': result.log_likelihood
                })
                
                print(f"Estimated {model_name}({p},{q})")
                
            except Exception as e:
                print(f"Failed to estimate {model_name}({p},{q}): {e}")
    
    # Sort models by AIC
    model_results.sort(key=lambda x: x['aic'])
    
    # Print top 5 models by AIC
    print("\nTop 5 Models by AIC:")
    print(f"{'Rank':<6} {'Model':<15} {'AIC':<12} {'BIC':<12} {'Log-Likelihood':<15}")
    print("-" * 60)
    
    for i, model in enumerate(model_results[:5]):
        print(f"{i+1:<6} {model['name']:<15} {model['aic']:<12.4f} {model['bic']:<12.4f} {model['log_likelihood']:<15.4f}")
    
    # Sort models by BIC
    model_results.sort(key=lambda x: x['bic'])
    
    # Print top 5 models by BIC
    print("\nTop 5 Models by BIC:")
    print(f"{'Rank':<6} {'Model':<15} {'BIC':<12} {'AIC':<12} {'Log-Likelihood':<15}")
    print("-" * 60)
    
    for i, model in enumerate(model_results[:5]):
        print(f"{i+1:<6} {model['name']:<15} {model['bic']:<12.4f} {model['aic']:<12.4f} {model['log_likelihood']:<15.4f}")

Rolling Window Estimation
----------------------

Perform rolling window estimation to analyze parameter stability:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mfe.models.univariate import GARCH
    
    # Define window size and step
    window_size = 500
    step_size = 20
    
    # Prepare data
    data = returns
    n_obs = len(data)
    
    # Calculate number of windows
    n_windows = (n_obs - window_size) // step_size + 1
    
    # Store results
    rolling_params = {
        'omega': np.zeros(n_windows),
        'alpha': np.zeros(n_windows),
        'beta': np.zeros(n_windows),
        'persistence': np.zeros(n_windows),
        'end_date': np.zeros(n_windows, dtype='datetime64[D]')
    }
    
    # Create a GARCH model
    model = GARCH(p=1, q=1)
    
    # Perform rolling window estimation
    for i in range(n_windows):
        # Define window indices
        start_idx = i * step_size
        end_idx = start_idx + window_size
        
        # Extract window data
        window_data = data[start_idx:end_idx]
        
        try:
            # Fit the model
            result = model.fit(window_data)
            
            # Store parameters
            rolling_params['omega'][i] = result.params.omega
            rolling_params['alpha'][i] = result.params.alpha
            rolling_params['beta'][i] = result.params.beta
            rolling_params['persistence'][i] = result.params.alpha + result.params.beta
            
            # Store end date if data has a date index
            if isinstance(data, pd.Series) and isinstance(data.index, pd.DatetimeIndex):
                rolling_params['end_date'][i] = data.index[end_idx-1]
            else:
                rolling_params['end_date'][i] = i
            
            print(f"Completed window {i+1}/{n_windows}")
            
        except Exception as e:
            print(f"Estimation failed for window {i+1}: {e}")
    
    # Plot rolling parameters
    plt.figure(figsize=(12, 10))
    
    # Omega
    plt.subplot(3, 1, 1)
    plt.plot(rolling_params['end_date'], rolling_params['omega'])
    plt.title('Rolling Window GARCH(1,1) - Omega Parameter')
    plt.ylabel('Omega')
    
    # Alpha
    plt.subplot(3, 1, 2)
    plt.plot(rolling_params['end_date'], rolling_params['alpha'])
    plt.title('Rolling Window GARCH(1,1) - Alpha Parameter')
    plt.ylabel('Alpha')
    
    # Beta
    plt.subplot(3, 1, 3)
    plt.plot(rolling_params['end_date'], rolling_params['beta'])
    plt.title('Rolling Window GARCH(1,1) - Beta Parameter')
    plt.ylabel('Beta')
    plt.xlabel('End of Window Date')
    
    plt.tight_layout()
    plt.show()
    
    # Plot persistence
    plt.figure(figsize=(12, 6))
    plt.plot(rolling_params['end_date'], rolling_params['persistence'])
    plt.axhline(y=1, color='r', linestyle='--', label='Unit Persistence')
    plt.title('Rolling Window GARCH(1,1) - Persistence (Alpha + Beta)')
    plt.ylabel('Persistence')
    plt.xlabel('End of Window Date')
    plt.legend()
    plt.show()

Numba Acceleration
---------------

The MFE Toolbox uses Numba's just-in-time (JIT) compilation to accelerate performance-critical functions. This is handled automatically, but you can see the performance improvement:

.. code-block:: python

    import time
    import numpy as np
    from mfe.models.univariate import GARCH
    from mfe.models.univariate._core import garch_recursion
    from numba import jit
    
    # Generate test data
    np.random.seed(42)
    returns = np.random.normal(0, 1, 5000) * 0.01
    
    # Define parameters
    omega = 0.00001
    alpha = 0.05
    beta = 0.90
    
    # Create a pure Python version of the GARCH recursion
    def garch_recursion_python(parameters, residuals, sigma2, backcast):
        T = len(residuals)
        omega, alpha, beta = parameters
        
        for t in range(1, T):
            sigma2[t] = omega + alpha * residuals[t-1]**2 + beta * sigma2[t-1]
        
        return sigma2
    
    # Time the pure Python version
    residuals = returns
    T = len(residuals)
    sigma2 = np.zeros(T)
    sigma2[0] = np.mean(residuals**2)  # Initialize with sample variance
    
    start_time = time.time()
    garch_recursion_python([omega, alpha, beta], residuals, sigma2.copy(), sigma2[0])
    python_time = time.time() - start_time
    
    # Time the Numba-accelerated version
    start_time = time.time()
    garch_recursion(np.array([omega, alpha, beta]), residuals, sigma2.copy(), sigma2[0])
    numba_time = time.time() - start_time
    
    # Print results
    print(f"Pure Python time: {python_time:.6f} seconds")
    print(f"Numba-accelerated time: {numba_time:.6f} seconds")
    print(f"Speedup factor: {python_time / numba_time:.2f}x")

Conclusion
=========

The univariate volatility models in the MFE Toolbox provide a comprehensive suite of tools for modeling time-varying volatility in financial time series. These models are essential for risk management, option pricing, portfolio optimization, and other financial applications.

The Python implementation with Numba acceleration offers both ease of use and high performance, making it suitable for both research and practical applications. The consistent API across different model types simplifies the process of comparing and selecting the most appropriate model for a given dataset.

For more advanced applications, see the documentation on multivariate volatility models, realized volatility estimators, and bootstrap methods.