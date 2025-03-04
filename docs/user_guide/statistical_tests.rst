.. _statistical_tests:

Statistical Tests
================

Introduction
-----------

The MFE Toolbox provides a comprehensive suite of statistical tests for validating model assumptions, checking distributional properties, and performing hypothesis testing in financial time series analysis. These tests are implemented as Python classes with methods that follow a consistent interface pattern, making them easy to use and integrate into your analysis workflow.

This guide covers the main statistical tests available in the MFE Toolbox, their purpose, implementation details, and usage examples.

Key Features
~~~~~~~~~~~

* Class-based implementation with consistent interfaces
* Integration with SciPy's statistical testing framework
* Support for both NumPy arrays and Pandas Series/DataFrames
* Asynchronous interfaces for long-running tests
* Comprehensive input validation with type hints
* Detailed test result objects with statistics and p-values

Available Tests
--------------

Jarque-Bera Test
~~~~~~~~~~~~~~~

The Jarque-Bera test evaluates whether a dataset has the skewness and kurtosis matching a normal distribution.

**Purpose**: Test for normality of a distribution based on sample skewness and kurtosis.

**Implementation**:

.. code-block:: python

    from mfe.models.tests import jarque_bera
    
    # Test data for normality
    jb_stat, p_value = jarque_bera(data)
    
    # Interpret results
    if p_value > 0.05:
        print("Data appears to be normally distributed")
    else:
        print("Data does not appear to be normally distributed")
        print(f"Test statistic: {jb_stat:.4f}, p-value: {p_value:.4f}")

**Asynchronous Interface**:

.. code-block:: python

    import asyncio
    from mfe.models.tests import jarque_bera_async
    
    async def test_normality():
        # Test data for normality asynchronously
        jb_stat, p_value = await jarque_bera_async(data)
        return jb_stat, p_value
    
    # Run the async function
    jb_stat, p_value = asyncio.run(test_normality())

**Parameters**:

* ``data`` (Union[np.ndarray, pd.Series]): The data to test for normality

**Returns**:

* ``jb_stat`` (float): The Jarque-Bera test statistic
* ``p_value`` (float): The p-value for the test

**Notes**:

The Jarque-Bera test statistic is calculated as:

.. math::

    JB = \frac{n}{6} \left( S^2 + \frac{(K-3)^2}{4} \right)

where :math:`n` is the sample size, :math:`S` is the sample skewness, and :math:`K` is the sample kurtosis. The test statistic follows a chi-squared distribution with 2 degrees of freedom under the null hypothesis of normality.

Kolmogorov-Smirnov Test
~~~~~~~~~~~~~~~~~~~~~~

The Kolmogorov-Smirnov test compares a sample with a reference probability distribution.

**Purpose**: Test whether a sample comes from a specified distribution.

**Implementation**:

.. code-block:: python

    from mfe.models.tests import kolmogorov_smirnov
    from scipy import stats
    
    # Test if data follows a normal distribution
    ks_stat, p_value = kolmogorov_smirnov(data, stats.norm.cdf)
    
    # Test if data follows a t-distribution with 5 degrees of freedom
    t_cdf = lambda x: stats.t.cdf(x, df=5)
    ks_stat, p_value = kolmogorov_smirnov(data, t_cdf)
    
    # Interpret results
    if p_value > 0.05:
        print("Data appears to follow the specified distribution")
    else:
        print("Data does not appear to follow the specified distribution")
        print(f"Test statistic: {ks_stat:.4f}, p-value: {p_value:.4f}")

**Asynchronous Interface**:

.. code-block:: python

    import asyncio
    from mfe.models.tests import kolmogorov_smirnov_async
    
    async def test_distribution():
        # Test data against normal distribution asynchronously
        ks_stat, p_value = await kolmogorov_smirnov_async(data, stats.norm.cdf)
        return ks_stat, p_value
    
    # Run the async function
    ks_stat, p_value = asyncio.run(test_distribution())

**Parameters**:

* ``data`` (Union[np.ndarray, pd.Series]): The data to test
* ``cdf`` (Callable): The cumulative distribution function to test against

**Returns**:

* ``ks_stat`` (float): The Kolmogorov-Smirnov test statistic
* ``p_value`` (float): The p-value for the test

**Notes**:

The Kolmogorov-Smirnov test statistic is calculated as:

.. math::

    D = \sup_x |F_n(x) - F(x)|

where :math:`F_n(x)` is the empirical distribution function and :math:`F(x)` is the theoretical cumulative distribution function.

Berkowitz Test
~~~~~~~~~~~~~

The Berkowitz test is used to evaluate the adequacy of density forecasts, particularly useful in risk management.

**Purpose**: Test whether a series of probability integral transforms follows a standard normal distribution.

**Implementation**:

.. code-block:: python

    from mfe.models.tests import berkowitz
    from scipy import stats
    
    # Test if data transformed by normal CDF follows a normal distribution
    berk_stat, p_value = berkowitz(data, stats.norm.cdf)
    
    # Interpret results
    if p_value > 0.05:
        print("Transformed data appears to be normally distributed")
    else:
        print("Transformed data does not appear to be normally distributed")
        print(f"Test statistic: {berk_stat:.4f}, p-value: {p_value:.4f}")

**Asynchronous Interface**:

.. code-block:: python

    import asyncio
    from mfe.models.tests import berkowitz_async
    
    async def test_berkowitz():
        # Test data using Berkowitz test asynchronously
        berk_stat, p_value = await berkowitz_async(data, stats.norm.cdf)
        return berk_stat, p_value
    
    # Run the async function
    berk_stat, p_value = asyncio.run(test_berkowitz())

**Parameters**:

* ``data`` (Union[np.ndarray, pd.Series]): The data to test
* ``cdf`` (Callable): The cumulative distribution function to transform the data

**Returns**:

* ``berk_stat`` (float): The Berkowitz test statistic
* ``p_value`` (float): The p-value for the test

**Notes**:

The Berkowitz test applies the probability integral transform to the data using the specified CDF, then tests whether the transformed data follows a standard normal distribution using a likelihood ratio test.

Ljung-Box Test
~~~~~~~~~~~~~

The Ljung-Box test checks for autocorrelation in a time series.

**Purpose**: Test for the presence of autocorrelation in a time series up to a specified lag.

**Implementation**:

.. code-block:: python

    from mfe.models.tests import ljung_box
    
    # Test for autocorrelation up to lag 10
    lb_stat, p_value = ljung_box(residuals, lags=10)
    
    # Interpret results
    if p_value > 0.05:
        print("No significant autocorrelation detected")
    else:
        print("Significant autocorrelation detected")
        print(f"Test statistic: {lb_stat:.4f}, p-value: {p_value:.4f}")

**Asynchronous Interface**:

.. code-block:: python

    import asyncio
    from mfe.models.tests import ljung_box_async
    
    async def test_autocorrelation():
        # Test for autocorrelation asynchronously
        lb_stat, p_value = await ljung_box_async(residuals, lags=10)
        return lb_stat, p_value
    
    # Run the async function
    lb_stat, p_value = asyncio.run(test_autocorrelation())

**Parameters**:

* ``data`` (Union[np.ndarray, pd.Series]): The time series data to test
* ``lags`` (int): The number of lags to include in the test

**Returns**:

* ``lb_stat`` (float): The Ljung-Box Q-statistic
* ``p_value`` (float): The p-value for the test

**Notes**:

The Ljung-Box Q-statistic is calculated as:

.. math::

    Q = n(n+2) \sum_{k=1}^{h} \frac{\hat{\rho}_k^2}{n-k}

where :math:`n` is the sample size, :math:`h` is the number of lags, and :math:`\hat{\rho}_k` is the sample autocorrelation at lag :math:`k`. The test statistic follows a chi-squared distribution with :math:`h` degrees of freedom under the null hypothesis of no autocorrelation.

Lagrange Multiplier (LM) Test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Lagrange Multiplier test checks for ARCH effects in a time series.

**Purpose**: Test for the presence of autoregressive conditional heteroskedasticity (ARCH) effects in a time series.

**Implementation**:

.. code-block:: python

    from mfe.models.tests import lm_test
    
    # Test for ARCH effects up to lag 10
    lm_stat, p_value = lm_test(residuals, lags=10)
    
    # Interpret results
    if p_value > 0.05:
        print("No significant ARCH effects detected")
    else:
        print("Significant ARCH effects detected")
        print(f"Test statistic: {lm_stat:.4f}, p-value: {p_value:.4f}")

**Asynchronous Interface**:

.. code-block:: python

    import asyncio
    from mfe.models.tests import lm_test_async
    
    async def test_arch_effects():
        # Test for ARCH effects asynchronously
        lm_stat, p_value = await lm_test_async(residuals, lags=10)
        return lm_stat, p_value
    
    # Run the async function
    lm_stat, p_value = asyncio.run(test_arch_effects())

**Parameters**:

* ``data`` (Union[np.ndarray, pd.Series]): The time series data to test
* ``lags`` (int): The number of lags to include in the test

**Returns**:

* ``lm_stat`` (float): The Lagrange Multiplier test statistic
* ``p_value`` (float): The p-value for the test

**Notes**:

The LM test for ARCH effects is based on an auxiliary regression of squared residuals on their own lags. The test statistic follows a chi-squared distribution with degrees of freedom equal to the number of lags under the null hypothesis of no ARCH effects.

Practical Examples
-----------------

Testing Model Residuals
~~~~~~~~~~~~~~~~~~~~~

A common application of statistical tests is to validate the assumptions of a time series model by examining its residuals.

.. code-block:: python

    import numpy as np
    import pandas as pd
    from mfe.models.time_series import ARMA
    from mfe.models.tests import jarque_bera, ljung_box, lm_test
    import matplotlib.pyplot as plt
    
    # Fit an ARMA model to data
    model = ARMA().fit(data, ar_order=1, ma_order=1)
    residuals = model.residuals
    
    # Test residuals for normality
    jb_stat, jb_pvalue = jarque_bera(residuals)
    
    # Test residuals for autocorrelation
    lb_stat, lb_pvalue = ljung_box(residuals, lags=10)
    
    # Test residuals for ARCH effects
    lm_stat, lm_pvalue = lm_test(residuals, lags=10)
    
    # Print results
    print("Residual Diagnostics:")
    print(f"Jarque-Bera test: stat={jb_stat:.4f}, p-value={jb_pvalue:.4f}")
    print(f"Ljung-Box test: stat={lb_stat:.4f}, p-value={lb_pvalue:.4f}")
    print(f"LM test: stat={lm_stat:.4f}, p-value={lm_pvalue:.4f}")
    
    # Plot residuals
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(residuals)
    plt.title('Model Residuals')
    plt.subplot(2, 1, 2)
    plt.hist(residuals, bins=30, density=True)
    plt.title('Residual Distribution')
    plt.tight_layout()
    plt.show()

Distribution Testing
~~~~~~~~~~~~~~~~~~

Testing whether a dataset follows a specific distribution is useful for model selection and validation.

.. code-block:: python

    import numpy as np
    import pandas as pd
    from scipy import stats
    from mfe.models.tests import kolmogorov_smirnov, berkowitz
    import matplotlib.pyplot as plt
    
    # Generate some sample data
    np.random.seed(42)
    data = np.random.standard_t(df=5, size=1000)
    
    # Test against normal distribution
    ks_norm_stat, ks_norm_pvalue = kolmogorov_smirnov(data, stats.norm.cdf)
    
    # Test against t-distribution with 5 degrees of freedom
    t_cdf = lambda x: stats.t.cdf(x, df=5)
    ks_t_stat, ks_t_pvalue = kolmogorov_smirnov(data, t_cdf)
    
    # Test using Berkowitz test
    berk_norm_stat, berk_norm_pvalue = berkowitz(data, stats.norm.cdf)
    berk_t_stat, berk_t_pvalue = berkowitz(data, t_cdf)
    
    # Print results
    print("Distribution Tests:")
    print("Kolmogorov-Smirnov test against normal distribution:")
    print(f"  stat={ks_norm_stat:.4f}, p-value={ks_norm_pvalue:.4f}")
    print("Kolmogorov-Smirnov test against t-distribution (df=5):")
    print(f"  stat={ks_t_stat:.4f}, p-value={ks_t_pvalue:.4f}")
    print("Berkowitz test against normal distribution:")
    print(f"  stat={berk_norm_stat:.4f}, p-value={berk_norm_pvalue:.4f}")
    print("Berkowitz test against t-distribution (df=5):")
    print(f"  stat={berk_t_stat:.4f}, p-value={berk_t_pvalue:.4f}")
    
    # Plot data with fitted distributions
    x = np.linspace(-5, 5, 1000)
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=50, density=True, alpha=0.6, label='Data')
    plt.plot(x, stats.norm.pdf(x), 'r-', label='Normal')
    plt.plot(x, stats.t.pdf(x, df=5), 'g-', label='t (df=5)')
    plt.legend()
    plt.title('Data Distribution with Fitted Curves')
    plt.show()

Testing for Serial Correlation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Testing for serial correlation is important in time series analysis to validate model assumptions.

.. code-block:: python

    import numpy as np
    import pandas as pd
    from mfe.models.tests import ljung_box
    import matplotlib.pyplot as plt
    
    # Generate an AR(1) process
    np.random.seed(42)
    n = 1000
    phi = 0.7
    ar1_data = np.zeros(n)
    for t in range(1, n):
        ar1_data[t] = phi * ar1_data[t-1] + np.random.normal(0, 1)
    
    # Test for autocorrelation at different lag lengths
    lags = [5, 10, 15, 20]
    results = []
    
    for lag in lags:
        lb_stat, p_value = ljung_box(ar1_data, lags=lag)
        results.append((lag, lb_stat, p_value))
    
    # Print results
    print("Ljung-Box Test for Serial Correlation:")
    print("Lags | Q-Statistic | p-value")
    print("-" * 30)
    for lag, stat, pval in results:
        print(f"{lag:4d} | {stat:11.4f} | {pval:.4f}")
    
    # Plot autocorrelation function
    from matplotlib.pyplot import acorr
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(ar1_data)
    plt.title('AR(1) Process')
    plt.subplot(2, 1, 2)
    acorr(ar1_data, maxlags=20)
    plt.title('Autocorrelation Function')
    plt.tight_layout()
    plt.show()

Testing for ARCH Effects
~~~~~~~~~~~~~~~~~~~~~~

Testing for ARCH effects is crucial when modeling financial time series with volatility clustering.

.. code-block:: python

    import numpy as np
    import pandas as pd
    from mfe.models.tests import lm_test
    import matplotlib.pyplot as plt
    
    # Generate a GARCH(1,1) process
    np.random.seed(42)
    n = 1000
    omega, alpha, beta = 0.1, 0.2, 0.7
    
    # Initialize arrays
    returns = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)  # Unconditional variance
    
    # Generate GARCH process
    for t in range(1, n):
        returns[t] = np.sqrt(sigma2[t]) * np.random.normal(0, 1)
        sigma2[t+1] = omega + alpha * returns[t]**2 + beta * sigma2[t]
    
    # Test for ARCH effects at different lag lengths
    lags = [5, 10, 15, 20]
    results = []
    
    for lag in lags:
        lm_stat, p_value = lm_test(returns, lags=lag)
        results.append((lag, lm_stat, p_value))
    
    # Print results
    print("LM Test for ARCH Effects:")
    print("Lags | LM-Statistic | p-value")
    print("-" * 30)
    for lag, stat, pval in results:
        print(f"{lag:4d} | {stat:12.4f} | {pval:.4f}")
    
    # Plot returns and squared returns
    plt.figure(figsize=(10, 8))
    plt.subplot(3, 1, 1)
    plt.plot(returns)
    plt.title('GARCH(1,1) Returns')
    plt.subplot(3, 1, 2)
    plt.plot(returns**2)
    plt.title('Squared Returns')
    plt.subplot(3, 1, 3)
    plt.plot(np.sqrt(sigma2[:-1]))
    plt.title('Conditional Volatility')
    plt.tight_layout()
    plt.show()

Advanced Usage
-------------

Combining Multiple Tests
~~~~~~~~~~~~~~~~~~~~~~

For comprehensive model validation, it's often useful to combine multiple statistical tests.

.. code-block:: python

    import numpy as np
    import pandas as pd
    from mfe.models.time_series import ARMA
    from mfe.models.tests import jarque_bera, ljung_box, lm_test
    
    def run_diagnostic_tests(data, model=None):
        """Run a comprehensive set of diagnostic tests on data or model residuals."""
        if model is not None:
            # If a model is provided, use its residuals
            test_data = model.residuals
            print("Running tests on model residuals:")
        else:
            # Otherwise use the provided data
            test_data = data
            print("Running tests on raw data:")
        
        # Run tests
        jb_stat, jb_pvalue = jarque_bera(test_data)
        lb_stat, lb_pvalue = ljung_box(test_data, lags=10)
        lm_stat, lm_pvalue = lm_test(test_data, lags=10)
        
        # Compile results
        results = {
            "Jarque-Bera": {"statistic": jb_stat, "p-value": jb_pvalue, 
                           "conclusion": "Normal" if jb_pvalue > 0.05 else "Non-normal"},
            "Ljung-Box": {"statistic": lb_stat, "p-value": lb_pvalue,
                         "conclusion": "No autocorrelation" if lb_pvalue > 0.05 else "Autocorrelated"},
            "LM Test": {"statistic": lm_stat, "p-value": lm_pvalue,
                       "conclusion": "No ARCH effects" if lm_pvalue > 0.05 else "ARCH effects present"}
        }
        
        # Print formatted results
        print("\nTest Results:")
        print("-" * 60)
        print(f"{'Test':<15} | {'Statistic':>10} | {'p-value':>10} | {'Conclusion':<20}")
        print("-" * 60)
        for test, res in results.items():
            print(f"{test:<15} | {res['statistic']:10.4f} | {res['p-value']:10.4f} | {res['conclusion']:<20}")
        
        return results
    
    # Example usage:
    # 1. Test raw data
    results_raw = run_diagnostic_tests(data)
    
    # 2. Fit a model and test residuals
    model = ARMA().fit(data, ar_order=1, ma_order=1)
    results_model = run_diagnostic_tests(data, model)

Asynchronous Testing
~~~~~~~~~~~~~~~~~~

For large datasets or when running multiple tests, asynchronous execution can improve performance.

.. code-block:: python

    import asyncio
    import numpy as np
    import pandas as pd
    from mfe.models.tests import (
        jarque_bera_async, ljung_box_async, lm_test_async, 
        kolmogorov_smirnov_async, berkowitz_async
    )
    from scipy import stats
    
    async def run_all_tests(data):
        """Run all tests asynchronously."""
        # Create tasks for all tests
        jb_task = jarque_bera_async(data)
        lb_task = ljung_box_async(data, lags=10)
        lm_task = lm_test_async(data, lags=10)
        ks_norm_task = kolmogorov_smirnov_async(data, stats.norm.cdf)
        berk_task = berkowitz_async(data, stats.norm.cdf)
        
        # Run all tasks concurrently
        jb_result, lb_result, lm_result, ks_result, berk_result = await asyncio.gather(
            jb_task, lb_task, lm_task, ks_norm_task, berk_task
        )
        
        # Compile results
        results = {
            "Jarque-Bera": jb_result,
            "Ljung-Box": lb_result,
            "LM Test": lm_result,
            "Kolmogorov-Smirnov": ks_result,
            "Berkowitz": berk_result
        }
        
        return results
    
    # Run all tests asynchronously
    results = asyncio.run(run_all_tests(data))
    
    # Print results
    print("Asynchronous Test Results:")
    print("-" * 50)
    for test, (stat, pval) in results.items():
        print(f"{test:<20}: stat={stat:.4f}, p-value={pval:.4f}")

Custom Test Result Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Visualizing test results can help with interpretation and presentation.

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mfe.models.tests import ljung_box
    
    def plot_ljung_box_results(data, max_lag=20):
        """Plot Ljung-Box test results for multiple lags."""
        lags = range(1, max_lag + 1)
        stats = []
        pvalues = []
        
        for lag in lags:
            stat, pval = ljung_box(data, lags=lag)
            stats.append(stat)
            pvalues.append(pval)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot test statistics
        ax1.plot(lags, stats, 'o-', color='blue')
        ax1.axhline(y=stats[-1], linestyle='--', color='gray')
        ax1.set_xlabel('Lag')
        ax1.set_ylabel('Q-Statistic')
        ax1.set_title('Ljung-Box Q-Statistics')
        
        # Plot p-values
        ax2.plot(lags, pvalues, 'o-', color='red')
        ax2.axhline(y=0.05, linestyle='--', color='gray', label='5% Significance')
        ax2.set_xlabel('Lag')
        ax2.set_ylabel('p-value')
        ax2.set_title('Ljung-Box p-values')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Return results as DataFrame
        results = pd.DataFrame({
            'Lag': lags,
            'Q-Statistic': stats,
            'p-value': pvalues
        })
        
        return results

Integration with SciPy
~~~~~~~~~~~~~~~~~~~~

The MFE Toolbox tests are designed to work seamlessly with SciPy's statistical functions.

.. code-block:: python

    import numpy as np
    import pandas as pd
    from scipy import stats
    from mfe.models.tests import kolmogorov_smirnov
    
    # Generate data from a mixture of two normal distributions
    np.random.seed(42)
    n = 1000
    mixture = np.concatenate([
        np.random.normal(-2, 1, size=n//2),
        np.random.normal(2, 1, size=n//2)
    ])
    
    # Test against various distributions
    distributions = {
        'Normal': stats.norm,
        'T (df=5)': lambda: stats.t(df=5),
        'Cauchy': stats.cauchy,
        'Laplace': stats.laplace,
        'Logistic': stats.logistic
    }
    
    # Fit each distribution and test
    results = []
    
    for name, dist_func in distributions.items():
        # Get distribution object
        dist = dist_func()
        
        # Fit distribution to data
        params = dist.fit(mixture)
        
        # Create CDF function with fitted parameters
        if name == 'Normal':
            loc, scale = params
            cdf = lambda x: dist.cdf(x, loc=loc, scale=scale)
        elif name == 'T (df=5)':
            loc, scale = params
            cdf = lambda x: dist.cdf(x, df=5, loc=loc, scale=scale)
        else:
            cdf = lambda x: dist.cdf(x, *params)
        
        # Run Kolmogorov-Smirnov test
        ks_stat, p_value = kolmogorov_smirnov(mixture, cdf)
        
        # Store results
        results.append({
            'Distribution': name,
            'Parameters': params,
            'KS Statistic': ks_stat,
            'p-value': p_value
        })
    
    # Convert to DataFrame and sort by p-value (best fit first)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('p-value', ascending=False)
    
    # Print results
    print("Distribution Fitting Results:")
    print(results_df[['Distribution', 'KS Statistic', 'p-value']])

Conclusion
---------

The statistical tests provided by the MFE Toolbox offer a comprehensive suite of tools for validating model assumptions, checking distributional properties, and performing hypothesis testing in financial time series analysis. The consistent interface, integration with SciPy, and support for asynchronous execution make these tests easy to use and integrate into your analysis workflow.

For more detailed information on specific tests, refer to the API documentation in :ref:`api_tests`.

See Also
--------

* :ref:`time_series_analysis` - Time series modeling and forecasting
* :ref:`univariate_volatility_models` - Univariate volatility modeling
* :ref:`statistical_distributions` - Statistical distributions