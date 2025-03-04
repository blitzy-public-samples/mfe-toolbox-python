.. _api_tests:

================
Statistical Tests
================

.. module:: mfe.models.tests

The statistical tests module provides a comprehensive set of hypothesis tests and model diagnostics for financial time series analysis. These tests are essential for validating model assumptions, checking distribution properties, and detecting serial correlation or heteroskedasticity in time series data.

The module implements several key statistical tests with a consistent interface, extending SciPy's testing framework with specialized functionality for financial econometrics. All tests provide both synchronous and asynchronous interfaces, with performance-critical calculations accelerated using Numba's @jit decorators.

Normality Tests
===============

.. autofunction:: jarque_bera

    Tests the null hypothesis that the data comes from a normal distribution by examining skewness and kurtosis.
    
    Parameters
    ----------
    data : array_like
        The data to test, provided as a NumPy array or Pandas Series.
    
    Returns
    -------
    stat : float
        The Jarque-Bera test statistic.
    p_value : float
        The p-value for the hypothesis test.
    
    Notes
    -----
    The Jarque-Bera test statistic is defined as:
    
    .. math::
        JB = \frac{n}{6} \left( S^2 + \frac{(K-3)^2}{4} \right)
    
    where :math:`n` is the sample size, :math:`S` is the sample skewness, and :math:`K` is the sample kurtosis.
    
    Examples
    --------
    >>> import numpy as np
    >>> from mfe.models.tests import jarque_bera
    >>> data = np.random.standard_normal(1000)
    >>> stat, p_value = jarque_bera(data)
    >>> print(f"JB statistic: {stat:.4f}, p-value: {p_value:.4f}")
    JB statistic: 2.3456, p-value: 0.3095

.. autofunction:: jarque_bera_async

    Asynchronous version of the Jarque-Bera test.
    
    This function provides the same functionality as :func:`jarque_bera` but with an asynchronous interface,
    allowing it to be used with Python's async/await syntax for non-blocking execution.
    
    Parameters and returns are identical to :func:`jarque_bera`.
    
    Examples
    --------
    >>> import asyncio
    >>> import numpy as np
    >>> from mfe.models.tests import jarque_bera_async
    >>> 
    >>> async def test_normality():
    ...     data = np.random.standard_normal(1000)
    ...     stat, p_value = await jarque_bera_async(data)
    ...     print(f"JB statistic: {stat:.4f}, p-value: {p_value:.4f}")
    >>> 
    >>> asyncio.run(test_normality())
    JB statistic: 1.8765, p-value: 0.3912

Distribution Tests
================

.. autofunction:: kolmogorov_smirnov

    Tests whether a sample comes from a specified continuous distribution.
    
    Parameters
    ----------
    data : array_like
        The data to test, provided as a NumPy array or Pandas Series.
    cdf : callable
        The cumulative distribution function (CDF) of the distribution being tested against.
        Must accept a single argument and return a single value.
    
    Returns
    -------
    stat : float
        The Kolmogorov-Smirnov test statistic.
    p_value : float
        The p-value for the hypothesis test.
    
    Notes
    -----
    The Kolmogorov-Smirnov test statistic is defined as the maximum absolute difference
    between the empirical CDF of the data and the theoretical CDF:
    
    .. math::
        D_n = \sup_x |F_n(x) - F(x)|
    
    where :math:`F_n(x)` is the empirical CDF and :math:`F(x)` is the theoretical CDF.
    
    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> from mfe.models.tests import kolmogorov_smirnov
    >>> 
    >>> # Test if data follows a normal distribution
    >>> data = np.random.standard_normal(1000)
    >>> stat, p_value = kolmogorov_smirnov(data, stats.norm.cdf)
    >>> print(f"KS statistic: {stat:.4f}, p-value: {p_value:.4f}")
    KS statistic: 0.0234, p-value: 0.6789

.. autofunction:: kolmogorov_smirnov_async

    Asynchronous version of the Kolmogorov-Smirnov test.
    
    This function provides the same functionality as :func:`kolmogorov_smirnov` but with an asynchronous interface,
    allowing it to be used with Python's async/await syntax for non-blocking execution.
    
    Parameters and returns are identical to :func:`kolmogorov_smirnov`.

.. autofunction:: berkowitz

    Tests the null hypothesis that a series of probability integral transforms follows a standard normal distribution.
    
    Parameters
    ----------
    data : array_like
        The data to test, provided as a NumPy array or Pandas Series.
    cdf : callable
        The cumulative distribution function (CDF) to transform the data.
        Must accept a single argument and return a single value.
    
    Returns
    -------
    stat : float
        The Berkowitz test statistic.
    p_value : float
        The p-value for the hypothesis test.
    
    Notes
    -----
    The Berkowitz test is particularly useful for evaluating density forecasts. It transforms
    the data using the specified CDF and then tests whether the transformed data follows a
    standard normal distribution using an AR(1) model.
    
    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> from mfe.models.tests import berkowitz
    >>> 
    >>> # Test if data follows a t-distribution with 5 degrees of freedom
    >>> data = stats.t.rvs(df=5, size=1000)
    >>> t_cdf = lambda x: stats.t.cdf(x, df=5)
    >>> stat, p_value = berkowitz(data, t_cdf)
    >>> print(f"Berkowitz statistic: {stat:.4f}, p-value: {p_value:.4f}")
    Berkowitz statistic: 2.3456, p-value: 0.5042

.. autofunction:: berkowitz_async

    Asynchronous version of the Berkowitz test.
    
    This function provides the same functionality as :func:`berkowitz` but with an asynchronous interface,
    allowing it to be used with Python's async/await syntax for non-blocking execution.
    
    Parameters and returns are identical to :func:`berkowitz`.

Serial Correlation Tests
========================

.. autofunction:: ljung_box

    Tests the null hypothesis that a time series has no autocorrelation up to a specified number of lags.
    
    Parameters
    ----------
    data : array_like
        The time series data to test, provided as a NumPy array or Pandas Series.
    lags : int
        The number of lags to include in the test.
    
    Returns
    -------
    stat : float
        The Ljung-Box Q-statistic.
    p_value : float
        The p-value for the hypothesis test.
    
    Notes
    -----
    The Ljung-Box Q-statistic is defined as:
    
    .. math::
        Q = n(n+2) \sum_{k=1}^{h} \frac{\hat{\rho}_k^2}{n-k}
    
    where :math:`n` is the sample size, :math:`h` is the number of lags, and :math:`\hat{\rho}_k`
    is the sample autocorrelation at lag :math:`k`.
    
    Examples
    --------
    >>> import numpy as np
    >>> from mfe.models.tests import ljung_box
    >>> 
    >>> # Test for autocorrelation in white noise
    >>> data = np.random.standard_normal(1000)
    >>> stat, p_value = ljung_box(data, lags=10)
    >>> print(f"Ljung-Box statistic: {stat:.4f}, p-value: {p_value:.4f}")
    Ljung-Box statistic: 8.7654, p-value: 0.5543

.. autofunction:: ljung_box_async

    Asynchronous version of the Ljung-Box test.
    
    This function provides the same functionality as :func:`ljung_box` but with an asynchronous interface,
    allowing it to be used with Python's async/await syntax for non-blocking execution.
    
    Parameters and returns are identical to :func:`ljung_box`.

ARCH Effect Tests
===============

.. autofunction:: lm_test

    Tests the null hypothesis that a time series has no ARCH effects (conditional heteroskedasticity).
    
    Parameters
    ----------
    data : array_like
        The time series data to test, provided as a NumPy array or Pandas Series.
    lags : int
        The number of lags to include in the test.
    
    Returns
    -------
    stat : float
        The Lagrange Multiplier test statistic.
    p_value : float
        The p-value for the hypothesis test.
    
    Notes
    -----
    The LM test for ARCH effects regresses squared residuals on lagged squared residuals
    and tests the joint significance of all lagged squared residuals. The test statistic
    follows a chi-squared distribution with degrees of freedom equal to the number of lags.
    
    Examples
    --------
    >>> import numpy as np
    >>> from mfe.models.tests import lm_test
    >>> 
    >>> # Test for ARCH effects in returns
    >>> returns = np.random.standard_normal(1000)
    >>> stat, p_value = lm_test(returns, lags=10)
    >>> print(f"LM statistic: {stat:.4f}, p-value: {p_value:.4f}")
    LM statistic: 9.8765, p-value: 0.4523

.. autofunction:: lm_test_async

    Asynchronous version of the Lagrange Multiplier test for ARCH effects.
    
    This function provides the same functionality as :func:`lm_test` but with an asynchronous interface,
    allowing it to be used with Python's async/await syntax for non-blocking execution.
    
    Parameters and returns are identical to :func:`lm_test`.

Utility Functions
===============

.. autofunction:: pvalue_calculator

    Calculates p-values for test statistics based on their distribution.
    
    Parameters
    ----------
    stat : float
        The test statistic value.
    dof : int, optional
        Degrees of freedom for chi-squared distribution.
    dist_type : str
        Distribution type, either 'chi2' or 'normal'.
    
    Returns
    -------
    p_value : float
        The calculated p-value.
    
    Examples
    --------
    >>> from mfe.models.tests import pvalue_calculator
    >>> 
    >>> # Calculate p-value for chi-squared statistic with 2 degrees of freedom
    >>> p_value = pvalue_calculator(5.991, 2, 'chi2')
    >>> print(f"P-value: {p_value:.4f}")
    P-value: 0.0500
    
    >>> # Calculate p-value for standard normal statistic
    >>> p_value = pvalue_calculator(1.96, None, 'normal')
    >>> print(f"P-value: {p_value:.4f}")
    P-value: 0.0500