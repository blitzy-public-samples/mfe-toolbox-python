Now I'll create a comprehensive time series analysis guide for the MFE Toolbox based on the information gathered and the requirements.

# docs/user_guide/time_series_analysis.rst
====================
Time Series Analysis
====================

This guide provides a comprehensive overview of the time series analysis capabilities in the MFE Toolbox. The toolbox offers a robust set of tools for modeling, forecasting, and analyzing time series data with a focus on financial applications.

Introduction
===========

Time series analysis is a fundamental component of financial econometrics, enabling researchers and practitioners to model the dynamic behavior of economic and financial variables over time. The MFE Toolbox provides a comprehensive suite of time series modeling tools, including:

- ARMA/ARMAX modeling and forecasting
- Unit root testing and stationarity analysis
- Impulse response analysis
- Time series decomposition and filtering
- Vector autoregression (VAR) analysis
- Granger causality testing
- Heterogeneous autoregression (HAR) modeling

These tools are implemented using Python's scientific stack, leveraging NumPy for efficient array operations, Pandas for time series handling, Statsmodels for econometric modeling, and Numba for performance acceleration.

ARMA/ARMAX Modeling
==================

Autoregressive Moving Average (ARMA) models and their extension with exogenous variables (ARMAX) are fundamental tools for time series analysis. The MFE Toolbox provides a comprehensive implementation of these models.

Basic ARMA Model
--------------

An ARMA(p,q) model combines autoregressive (AR) and moving average (MA) components:

.. math::

    y_t = c + \sum_{i=1}^{p} \phi_i y_{t-i} + \sum_{j=1}^{q} \theta_j \varepsilon_{t-j} + \varepsilon_t

where:
- :math:`y_t` is the time series value at time t
- :math:`c` is a constant term
- :math:`\phi_i` are the AR coefficients
- :math:`\theta_j` are the MA coefficients
- :math:`\varepsilon_t` is white noise

Here's how to estimate an ARMA(2,1) model using the MFE Toolbox:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mfe.models.time_series import ARMA
    
    # Create or load time series data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=500, freq='B')
    data = pd.Series(np.random.normal(0, 1, 500), index=dates)
    
    # Create and fit an ARMA(2,1) model
    model = ARMA(ar_order=2, ma_order=1, include_constant=True)
    result = model.fit(data)
    
    # Print model summary
    print(result.summary())
    
    # Access model parameters
    constant = result.params.constant
    ar_params = result.params.ar  # Array of AR parameters
    ma_params = result.params.ma  # Array of MA parameters
    
    print(f"Constant: {constant:.4f}")
    print(f"AR parameters: {ar_params}")
    print(f"MA parameters: {ma_params}")
    
    # Plot original data and fitted values
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data, label='Original Data')
    plt.plot(data.index, result.fitted_values, label='Fitted Values', alpha=0.7)
    plt.legend()
    plt.title('ARMA(2,1) Model Fit')
    plt.show()

Note that Python uses 0-based indexing, so when accessing parameters, `ar_params[0]` corresponds to the coefficient for the first lag (AR(1)), unlike MATLAB where indexing starts at 1.

ARMAX Models with Exogenous Variables
----------------------------------

ARMAX models extend ARMA models by including exogenous variables:

.. math::

    y_t = c + \sum_{i=1}^{p} \phi_i y_{t-i} + \sum_{j=1}^{q} \theta_j \varepsilon_{t-j} + \sum_{k=1}^{r} \beta_k x_{k,t} + \varepsilon_t

where :math:`x_{k,t}` are exogenous variables and :math:`\beta_k` are their coefficients.

Here's how to estimate an ARMAX model:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from mfe.models.time_series import ARMA
    
    # Create time series data with exogenous variables
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=500, freq='B')
    
    # Create exogenous variables
    exog1 = np.sin(np.linspace(0, 10, 500))  # Seasonal component
    exog2 = np.random.normal(0, 1, 500)      # Random component
    
    # Create dependent variable with AR(1) process and exogenous effects
    y = np.zeros(500)
    e = np.random.normal(0, 0.5, 500)
    
    for t in range(1, 500):
        y[t] = 0.1 + 0.7 * y[t-1] + 0.5 * exog1[t] + 0.3 * exog2[t] + e[t]
    
    # Create DataFrame with all variables
    data = pd.DataFrame({
        'y': y,
        'exog1': exog1,
        'exog2': exog2
    }, index=dates)
    
    # Create and fit an ARMAX(1,0) model
    model = ARMA(
        ar_order=1, 
        ma_order=0, 
        include_constant=True,
        exog=data[['exog1', 'exog2']]  # Pass exogenous variables
    )
    
    result = model.fit(data['y'])
    
    # Print model summary
    print(result.summary())
    
    # Access model parameters
    constant = result.params.constant
    ar_params = result.params.ar
    exog_params = result.params.exog  # Array of exogenous variable coefficients
    
    print(f"Constant: {constant:.4f}")
    print(f"AR parameters: {ar_params}")
    print(f"Exogenous parameters: {exog_params}")

Model Selection
------------

The MFE Toolbox provides several criteria for model selection, including Akaike Information Criterion (AIC), Bayesian Information Criterion (BIC), and log-likelihood:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from mfe.models.time_series import ARMA
    
    # Load or create time series data
    np.random.seed(42)
    data = pd.Series(np.random.normal(0, 1, 500))
    
    # Define model orders to test
    ar_orders = range(0, 4)
    ma_orders = range(0, 4)
    
    # Store results
    results = []
    
    # Estimate models with different orders
    for p in ar_orders:
        for q in ma_orders:
            model = ARMA(ar_order=p, ma_order=q, include_constant=True)
            try:
                result = model.fit(data)
                results.append({
                    'ar_order': p,
                    'ma_order': q,
                    'aic': result.aic,
                    'bic': result.bic,
                    'loglikelihood': result.loglikelihood,
                    'result': result
                })
                print(f"ARMA({p},{q}): AIC={result.aic:.4f}, BIC={result.bic:.4f}")
            except Exception as e:
                print(f"ARMA({p},{q}) failed: {str(e)}")
    
    # Find the best model according to AIC
    best_aic = min(results, key=lambda x: x['aic'])
    print(f"\nBest model by AIC: ARMA({best_aic['ar_order']},{best_aic['ma_order']})")
    print(f"AIC: {best_aic['aic']:.4f}, BIC: {best_aic['bic']:.4f}")
    
    # Find the best model according to BIC
    best_bic = min(results, key=lambda x: x['bic'])
    print(f"Best model by BIC: ARMA({best_bic['ar_order']},{best_bic['ma_order']})")
    print(f"AIC: {best_bic['aic']:.4f}, BIC: {best_bic['bic']:.4f}")

Forecasting
=========

The MFE Toolbox provides comprehensive forecasting capabilities for time series models. Forecasts can be generated with confidence intervals and can be visualized easily.

Basic Forecasting
--------------

Here's how to generate forecasts from an ARMA model:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mfe.models.time_series import ARMA
    
    # Create or load time series data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=500, freq='B')
    data = pd.Series(np.random.normal(0, 1, 500), index=dates)
    
    # Create and fit an ARMA(1,1) model
    model = ARMA(ar_order=1, ma_order=1, include_constant=True)
    result = model.fit(data)
    
    # Generate forecasts for the next 20 periods
    forecast_horizon = 20
    forecasts = result.forecast(horizon=forecast_horizon)
    
    # Create forecast dates (business days after the last data point)
    forecast_dates = pd.date_range(
        start=dates[-1] + pd.Timedelta(days=1), 
        periods=forecast_horizon, 
        freq='B'
    )
    
    # Plot the data and forecasts
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data, label='Historical Data')
    plt.plot(forecast_dates, forecasts.mean, label='Point Forecast', color='red')
    
    # Add confidence intervals
    plt.fill_between(
        forecast_dates,
        forecasts.lower_ci,  # Lower confidence interval
        forecasts.upper_ci,  # Upper confidence interval
        color='red', alpha=0.2, label='95% Confidence Interval'
    )
    
    plt.legend()
    plt.title('ARMA(1,1) Forecasts')
    plt.show()
    
    # Print forecast values
    forecast_df = pd.DataFrame({
        'point_forecast': forecasts.mean,
        'lower_ci': forecasts.lower_ci,
        'upper_ci': forecasts.upper_ci
    }, index=forecast_dates)
    
    print(forecast_df)

Asynchronous Forecasting
---------------------

For long-horizon forecasts or when generating many simulation paths, the MFE Toolbox provides asynchronous forecasting capabilities:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import asyncio
    from mfe.models.time_series import ARMA
    
    async def generate_forecasts():
        # Create or load time series data
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=500, freq='B')
        data = pd.Series(np.random.normal(0, 1, 500), index=dates)
        
        # Create and fit an ARMA(1,1) model
        model = ARMA(ar_order=1, ma_order=1, include_constant=True)
        result = model.fit(data)
        
        # Define a progress callback function
        def progress_callback(percent, message):
            print(f"{percent:.1f}% complete: {message}")
        
        # Generate forecasts asynchronously with many simulation paths
        forecasts = await result.forecast_async(
            horizon=50,
            num_simulations=10000,  # Large number of simulation paths
            progress_callback=progress_callback
        )
        
        return data, forecasts
    
    # Run the asynchronous function
    data, forecasts = asyncio.run(generate_forecasts())
    
    # Create forecast dates
    forecast_dates = pd.date_range(
        start=data.index[-1] + pd.Timedelta(days=1), 
        periods=len(forecasts.mean), 
        freq='B'
    )
    
    # Plot the data and forecasts
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data, label='Historical Data')
    plt.plot(forecast_dates, forecasts.mean, label='Point Forecast', color='red')
    
    # Add confidence intervals
    plt.fill_between(
        forecast_dates,
        forecasts.lower_ci,
        forecasts.upper_ci,
        color='red', alpha=0.2, label='95% Confidence Interval'
    )
    
    plt.legend()
    plt.title('ARMA(1,1) Forecasts with 10,000 Simulation Paths')
    plt.show()

Forecast Evaluation
----------------

The MFE Toolbox provides tools for evaluating forecast accuracy:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from mfe.models.time_series import ARMA
    from mfe.models.time_series.diagnostics import forecast_evaluation
    
    # Create or load time series data
    np.random.seed(42)
    data = np.random.normal(0, 1, 600)  # 600 observations
    
    # Split into training and test sets
    train_data = data[:500]
    test_data = data[500:]
    
    # Create and fit an ARMA(1,1) model on training data
    model = ARMA(ar_order=1, ma_order=1, include_constant=True)
    result = model.fit(train_data)
    
    # Generate forecasts for the test period
    forecasts = result.forecast(horizon=len(test_data))
    
    # Evaluate forecasts
    evaluation = forecast_evaluation(
        actual=test_data,
        forecast=forecasts.mean,
        metrics=['mse', 'mae', 'rmse', 'mape', 'theil_u']
    )
    
    print("Forecast Evaluation Metrics:")
    for metric, value in evaluation.items():
        print(f"{metric.upper()}: {value:.6f}")

Diagnostic Checking
================

Model diagnostics are essential for validating time series models. The MFE Toolbox provides comprehensive diagnostic tools.

Residual Analysis
--------------

Analyzing model residuals helps verify model adequacy:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mfe.models.time_series import ARMA
    from mfe.models.time_series.diagnostics import residual_diagnostics
    
    # Create or load time series data
    np.random.seed(42)
    data = pd.Series(np.random.normal(0, 1, 500))
    
    # Create and fit an ARMA(1,1) model
    model = ARMA(ar_order=1, ma_order=1, include_constant=True)
    result = model.fit(data)
    
    # Get residuals
    residuals = result.residuals
    
    # Run diagnostic tests on residuals
    diagnostics = residual_diagnostics(residuals)
    
    print("Residual Diagnostics:")
    print(f"Jarque-Bera Test (Normality): p-value = {diagnostics['jarque_bera_p']:.6f}")
    print(f"Ljung-Box Test (Serial Correlation): p-value = {diagnostics['ljung_box_p']:.6f}")
    print(f"ARCH LM Test (Heteroskedasticity): p-value = {diagnostics['arch_lm_p']:.6f}")
    
    # Plot residuals
    plt.figure(figsize=(12, 8))
    
    # Residual time series
    plt.subplot(2, 2, 1)
    plt.plot(residuals)
    plt.title('Residuals')
    plt.axhline(y=0, color='r', linestyle='-')
    
    # Histogram of residuals
    plt.subplot(2, 2, 2)
    plt.hist(residuals, bins=30, density=True, alpha=0.7)
    plt.title('Residual Histogram')
    
    # ACF of residuals
    plt.subplot(2, 2, 3)
    from mfe.models.time_series.correlation import plot_acf
    plot_acf(residuals, lags=20, ax=plt.gca())
    plt.title('ACF of Residuals')
    
    # PACF of residuals
    plt.subplot(2, 2, 4)
    from mfe.models.time_series.correlation import plot_pacf
    plot_pacf(residuals, lags=20, ax=plt.gca())
    plt.title('PACF of Residuals')
    
    plt.tight_layout()
    plt.show()

ACF and PACF Analysis
------------------

Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) are essential tools for identifying appropriate ARMA model orders:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mfe.models.time_series.correlation import acf, pacf, plot_acf, plot_pacf
    
    # Create or load time series data
    np.random.seed(42)
    data = pd.Series(np.random.normal(0, 1, 500))
    
    # Generate an AR(2) process
    ar_data = np.zeros_like(data)
    for t in range(2, len(data)):
        ar_data[t] = 0.7 * ar_data[t-1] - 0.3 * ar_data[t-2] + data[t]
    
    # Calculate ACF and PACF
    acf_values = acf(ar_data, nlags=20)
    pacf_values = pacf(ar_data, nlags=20)
    
    # Print ACF and PACF values
    print("ACF values:")
    for i, val in enumerate(acf_values):
        print(f"Lag {i}: {val:.4f}")
    
    print("\nPACF values:")
    for i, val in enumerate(pacf_values):
        print(f"Lag {i}: {val:.4f}")
    
    # Plot ACF and PACF
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plot_acf(ar_data, lags=20, ax=plt.gca())
    plt.title('Autocorrelation Function')
    
    plt.subplot(1, 2, 2)
    plot_pacf(ar_data, lags=20, ax=plt.gca())
    plt.title('Partial Autocorrelation Function')
    
    plt.tight_layout()
    plt.show()

Unit Root Testing
==============

Unit root tests help determine whether a time series is stationary, which is a key assumption for many time series models.

Augmented Dickey-Fuller Test
-------------------------

The Augmented Dickey-Fuller (ADF) test is a common unit root test:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from mfe.models.time_series.unit_root import adf_test
    
    # Create or load time series data
    np.random.seed(42)
    
    # Create a stationary AR(1) process
    n = 500
    stationary_data = np.zeros(n)
    for t in range(1, n):
        stationary_data[t] = 0.7 * stationary_data[t-1] + np.random.normal(0, 1)
    
    # Create a non-stationary random walk
    nonstationary_data = np.cumsum(np.random.normal(0, 1, n))
    
    # Run ADF test on stationary data
    adf_stationary = adf_test(stationary_data)
    
    print("ADF Test for Stationary Series:")
    print(f"Test Statistic: {adf_stationary['adf_stat']:.4f}")
    print(f"p-value: {adf_stationary['p_value']:.4f}")
    print(f"Critical Values: 1%: {adf_stationary['critical_values']['1%']:.4f}, " +
          f"5%: {adf_stationary['critical_values']['5%']:.4f}, " +
          f"10%: {adf_stationary['critical_values']['10%']:.4f}")
    print(f"Is Stationary: {adf_stationary['is_stationary']}")
    
    # Run ADF test on non-stationary data
    adf_nonstationary = adf_test(nonstationary_data)
    
    print("\nADF Test for Non-Stationary Series:")
    print(f"Test Statistic: {adf_nonstationary['adf_stat']:.4f}")
    print(f"p-value: {adf_nonstationary['p_value']:.4f}")
    print(f"Critical Values: 1%: {adf_nonstationary['critical_values']['1%']:.4f}, " +
          f"5%: {adf_nonstationary['critical_values']['5%']:.4f}, " +
          f"10%: {adf_nonstationary['critical_values']['10%']:.4f}")
    print(f"Is Stationary: {adf_nonstationary['is_stationary']}")

KPSS Test
-------

The Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test is another popular unit root test:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from mfe.models.time_series.unit_root import kpss_test
    
    # Using the same data from the previous example
    
    # Run KPSS test on stationary data
    kpss_stationary = kpss_test(stationary_data)
    
    print("KPSS Test for Stationary Series:")
    print(f"Test Statistic: {kpss_stationary['kpss_stat']:.4f}")
    print(f"p-value: {kpss_stationary['p_value']:.4f}")
    print(f"Critical Values: 1%: {kpss_stationary['critical_values']['1%']:.4f}, " +
          f"5%: {kpss_stationary['critical_values']['5%']:.4f}, " +
          f"10%: {kpss_stationary['critical_values']['10%']:.4f}")
    print(f"Is Stationary: {kpss_stationary['is_stationary']}")
    
    # Run KPSS test on non-stationary data
    kpss_nonstationary = kpss_test(nonstationary_data)
    
    print("\nKPSS Test for Non-Stationary Series:")
    print(f"Test Statistic: {kpss_nonstationary['kpss_stat']:.4f}")
    print(f"p-value: {kpss_nonstationary['p_value']:.4f}")
    print(f"Critical Values: 1%: {kpss_nonstationary['critical_values']['1%']:.4f}, " +
          f"5%: {kpss_nonstationary['critical_values']['5%']:.4f}, " +
          f"10%: {kpss_nonstationary['critical_values']['10%']:.4f}")
    print(f"Is Stationary: {kpss_nonstationary['is_stationary']}")

Time Series Decomposition and Filtering
====================================

The MFE Toolbox provides tools for decomposing time series into trend, seasonal, and irregular components, as well as filtering techniques.

Hodrick-Prescott Filter
--------------------

The Hodrick-Prescott (HP) filter is a common method for extracting the trend component from a time series:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mfe.models.time_series.filters import hp_filter
    
    # Create or load time series data
    np.random.seed(42)
    n = 500
    
    # Create a trend component
    t = np.linspace(0, 5, n)
    trend = 0.01 * t**2
    
    # Create a cyclical component
    cycle = 2 * np.sin(2 * np.pi * t / 50)
    
    # Create a noise component
    noise = np.random.normal(0, 0.5, n)
    
    # Combine components
    data = trend + cycle + noise
    
    # Apply HP filter
    hp_result = hp_filter(data, lamb=1600)  # 1600 is standard for quarterly data
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(data, label='Original Data')
    plt.plot(hp_result['trend'], label='HP Trend', linewidth=2)
    plt.legend()
    plt.title('Original Data and HP Trend')
    
    plt.subplot(3, 1, 2)
    plt.plot(hp_result['trend'], label='HP Trend')
    plt.plot(trend, label='True Trend', linestyle='--')
    plt.legend()
    plt.title('HP Trend vs. True Trend')
    
    plt.subplot(3, 1, 3)
    plt.plot(hp_result['cycle'], label='HP Cycle')
    plt.plot(cycle, label='True Cycle', linestyle='--')
    plt.legend()
    plt.title('HP Cycle vs. True Cycle')
    
    plt.tight_layout()
    plt.show()

Baxter-King Filter
---------------

The Baxter-King (BK) filter is a band-pass filter that isolates cyclical components:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mfe.models.time_series.filters import bk_filter
    
    # Using the same data from the previous example
    
    # Apply BK filter
    bk_result = bk_filter(
        data,
        low=6,    # 6 periods for lower cutoff (e.g., 1.5 years for quarterly data)
        high=32,  # 32 periods for upper cutoff (e.g., 8 years for quarterly data)
        K=12      # Lead-lag length
    )
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(data, label='Original Data')
    plt.plot(bk_result['trend'], label='BK Trend', linewidth=2)
    plt.legend()
    plt.title('Original Data and BK Trend')
    
    plt.subplot(2, 1, 2)
    plt.plot(bk_result['cycle'], label='BK Cycle')
    plt.plot(cycle, label='True Cycle', linestyle='--')
    plt.legend()
    plt.title('BK Cycle vs. True Cycle')
    
    plt.tight_layout()
    plt.show()

Beveridge-Nelson Decomposition
---------------------------

The Beveridge-Nelson decomposition separates a time series into permanent and transitory components:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mfe.models.time_series.filters import beveridge_nelson_decomposition
    from mfe.models.time_series import ARMA
    
    # Create or load time series data
    np.random.seed(42)
    n = 500
    
    # Create a non-stationary series (random walk with drift)
    data = np.zeros(n)
    data[0] = 0
    for t in range(1, n):
        data[t] = 0.01 + data[t-1] + np.random.normal(0, 1)
    
    # Fit an ARMA model to the differenced data
    diff_data = np.diff(data)
    model = ARMA(ar_order=1, ma_order=1, include_constant=True)
    result = model.fit(diff_data)
    
    # Apply Beveridge-Nelson decomposition
    bn_result = beveridge_nelson_decomposition(
        data,
        ar_params=result.params.ar,
        ma_params=result.params.ma,
        drift=result.params.constant
    )
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(data, label='Original Data')
    plt.plot(bn_result['trend'], label='BN Trend', linewidth=2)
    plt.legend()
    plt.title('Original Data and Beveridge-Nelson Trend')
    
    plt.subplot(2, 1, 2)
    plt.plot(bn_result['cycle'], label='BN Cycle')
    plt.legend()
    plt.title('Beveridge-Nelson Cycle')
    
    plt.tight_layout()
    plt.show()

Vector Autoregression (VAR)
========================

Vector Autoregression (VAR) models extend univariate autoregressive models to multivariate time series.

Basic VAR Model
------------

Here's how to estimate a VAR model:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mfe.models.time_series.var import VAR
    
    # Create or load multivariate time series data
    np.random.seed(42)
    n = 500
    
    # Create a VAR(1) process
    # y_t = A y_{t-1} + e_t
    A = np.array([[0.5, 0.1], 
                  [0.2, 0.3]])  # 2x2 coefficient matrix
    
    # Initialize data
    y = np.zeros((n, 2))
    
    # Generate VAR(1) process
    for t in range(1, n):
        y[t] = A @ y[t-1] + np.random.multivariate_normal([0, 0], np.eye(2))
    
    # Create DataFrame with time index
    dates = pd.date_range(start='2020-01-01', periods=n, freq='B')
    data = pd.DataFrame(y, index=dates, columns=['y1', 'y2'])
    
    # Create and fit a VAR model
    model = VAR(lag_order=1)
    result = model.fit(data)
    
    # Print model summary
    print(result.summary())
    
    # Access coefficient matrices
    coef_matrix = result.coef_matrix
    print("\nCoefficient Matrix:")
    print(coef_matrix)
    
    # Generate forecasts
    forecasts = result.forecast(horizon=10)
    
    # Plot the data and forecasts
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data['y1'], label='y1 Data')
    plt.plot(
        pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=10, freq='B'),
        forecasts.mean[:, 0],
        label='y1 Forecast',
        color='red'
    )
    plt.legend()
    plt.title('VAR(1) Model: y1')
    
    plt.subplot(2, 1, 2)
    plt.plot(data.index, data['y2'], label='y2 Data')
    plt.plot(
        pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=10, freq='B'),
        forecasts.mean[:, 1],
        label='y2 Forecast',
        color='red'
    )
    plt.legend()
    plt.title('VAR(1) Model: y2')
    
    plt.tight_layout()
    plt.show()

Impulse Response Analysis
---------------------

Impulse response functions show how variables respond to shocks:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mfe.models.time_series.var import VAR
    from mfe.models.time_series.impulse_response import impulse_response
    
    # Using the VAR model from the previous example
    
    # Compute impulse responses
    irf = impulse_response(
        coef_matrices=result.coef_matrices,
        horizon=20,
        identification='cholesky'  # Cholesky decomposition for identification
    )
    
    # Plot impulse responses
    plt.figure(figsize=(12, 8))
    
    # Response of y1 to shocks
    plt.subplot(2, 2, 1)
    plt.plot(irf[:, 0, 0], label='Response of y1 to y1 shock')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.legend()
    plt.title('Response of y1 to y1 shock')
    
    plt.subplot(2, 2, 2)
    plt.plot(irf[:, 0, 1], label='Response of y1 to y2 shock')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.legend()
    plt.title('Response of y1 to y2 shock')
    
    # Response of y2 to shocks
    plt.subplot(2, 2, 3)
    plt.plot(irf[:, 1, 0], label='Response of y2 to y1 shock')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.legend()
    plt.title('Response of y2 to y1 shock')
    
    plt.subplot(2, 2, 4)
    plt.plot(irf[:, 1, 1], label='Response of y2 to y2 shock')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.legend()
    plt.title('Response of y2 to y2 shock')
    
    plt.tight_layout()
    plt.show()

Granger Causality
--------------

Granger causality tests help determine whether one time series is useful in forecasting another:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from mfe.models.time_series.causality import granger_causality
    
    # Using the VAR data from the previous example
    
    # Test Granger causality
    gc_y1_to_y2 = granger_causality(data['y1'], data['y2'], max_lag=5)
    gc_y2_to_y1 = granger_causality(data['y2'], data['y1'], max_lag=5)
    
    print("Granger Causality Tests:")
    print("\ny1 -> y2:")
    for lag, result in gc_y1_to_y2.items():
        print(f"Lag {lag}: F-stat = {result['f_stat']:.4f}, p-value = {result['p_value']:.4f}")
    
    print("\ny2 -> y1:")
    for lag, result in gc_y2_to_y1.items():
        print(f"Lag {lag}: F-stat = {result['f_stat']:.4f}, p-value = {result['p_value']:.4f}")

Heterogeneous Autoregression (HAR) Models
=====================================

Heterogeneous Autoregression (HAR) models are particularly useful for modeling realized volatility:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mfe.models.time_series.har import HAR
    
    # Create or load realized volatility data
    np.random.seed(42)
    n = 500
    
    # Create a persistent series similar to realized volatility
    rv = np.zeros(n)
    rv[0] = 0.1
    for t in range(1, n):
        rv[t] = 0.1 + 0.4 * rv[t-1] + 0.3 * np.mean(rv[max(0, t-5):t]) + 0.2 * np.mean(rv[max(0, t-22):t]) + np.random.normal(0, 0.05)
    
    # Create DataFrame with time index
    dates = pd.date_range(start='2020-01-01', periods=n, freq='B')
    rv_data = pd.Series(rv, index=dates)
    
    # Create and fit a HAR model
    model = HAR(lags=[1, 5, 22])  # Daily, weekly, monthly components
    result = model.fit(rv_data)
    
    # Print model summary
    print(result.summary())
    
    # Access model parameters
    constant = result.params.constant
    beta_d = result.params.beta[0]  # Daily coefficient
    beta_w = result.params.beta[1]  # Weekly coefficient
    beta_m = result.params.beta[2]  # Monthly coefficient
    
    print(f"Constant: {constant:.4f}")
    print(f"Daily coefficient: {beta_d:.4f}")
    print(f"Weekly coefficient: {beta_w:.4f}")
    print(f"Monthly coefficient: {beta_m:.4f}")
    
    # Generate forecasts
    forecasts = result.forecast(horizon=20)
    
    # Plot the data and forecasts
    plt.figure(figsize=(12, 6))
    plt.plot(rv_data.index, rv_data, label='Realized Volatility')
    plt.plot(
        pd.date_range(start=rv_data.index[-1] + pd.Timedelta(days=1), periods=20, freq='B'),
        forecasts.mean,
        label='HAR Forecast',
        color='red'
    )
    plt.legend()
    plt.title('HAR Model Forecast of Realized Volatility')
    plt.show()

Integration with Pandas Time Series Functionality
=============================================

The MFE Toolbox integrates seamlessly with Pandas time series functionality:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mfe.models.time_series import ARMA
    
    # Create time series data with Pandas DatetimeIndex
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=500, freq='B')
    data = pd.Series(np.random.normal(0, 1, 500), index=dates)
    
    # Add some seasonality
    data += np.sin(np.arange(len(data)) * 2 * np.pi / 20)  # 20-day cycle
    
    # Create and fit an ARMA model
    model = ARMA(ar_order=2, ma_order=1, include_constant=True)
    result = model.fit(data)
    
    # Generate forecasts
    forecasts = result.forecast(horizon=30)
    forecast_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=1), periods=30, freq='B')
    
    # Create a DataFrame with actual data and forecasts
    combined_data = pd.DataFrame({
        'actual': data,
        'fitted': pd.Series(result.fitted_values, index=dates)
    })
    
    # Add forecasts
    forecast_df = pd.DataFrame({
        'forecast': forecasts.mean,
        'lower_ci': forecasts.lower_ci,
        'upper_ci': forecasts.upper_ci
    }, index=forecast_dates)
    
    # Combine actual data and forecasts
    full_df = pd.concat([combined_data, forecast_df], axis=1)
    
    # Resample to monthly frequency
    monthly_data = full_df.resample('M').mean()
    
    # Plot monthly data
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_data.index, monthly_data['actual'], label='Actual (Monthly)')
    plt.plot(monthly_data.index, monthly_data['fitted'], label='Fitted (Monthly)')
    plt.plot(monthly_data.index, monthly_data['forecast'], label='Forecast (Monthly)', color='red')
    plt.fill_between(
        monthly_data.index,
        monthly_data['lower_ci'],
        monthly_data['upper_ci'],
        color='red', alpha=0.2, label='95% Confidence Interval'
    )
    plt.legend()
    plt.title('ARMA Model with Monthly Resampling')
    plt.show()
    
    # Perform time series operations with Pandas
    print("\nTime Series Statistics:")
    print(f"Rolling 20-day mean at end: {data.rolling(window=20).mean().iloc[-1]:.4f}")
    print(f"Expanding mean at end: {data.expanding().mean().iloc[-1]:.4f}")
    print(f"Year-to-date mean: {data[data.index.year == data.index[-1].year].mean():.4f}")
    
    # Seasonal decomposition
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # Create data with stronger seasonality
    seasonal_data = pd.Series(
        np.random.normal(0, 0.5, 500) + 2 * np.sin(np.arange(500) * 2 * np.pi / 20),
        index=dates
    )
    
    # Decompose the series
    decomposition = seasonal_decompose(seasonal_data, model='additive', period=20)
    
    # Plot the decomposition
    plt.figure(figsize=(12, 10))
    
    plt.subplot(4, 1, 1)
    plt.plot(decomposition.observed)
    plt.title('Original Data')
    
    plt.subplot(4, 1, 2)
    plt.plot(decomposition.trend)
    plt.title('Trend Component')
    
    plt.subplot(4, 1, 3)
    plt.plot(decomposition.seasonal)
    plt.title('Seasonal Component')
    
    plt.subplot(4, 1, 4)
    plt.plot(decomposition.resid)
    plt.title('Residual Component')
    
    plt.tight_layout()
    plt.show()

Conclusion
=========

The MFE Toolbox provides a comprehensive suite of time series analysis tools for financial econometrics. This guide has covered the basic usage of ARMA/ARMAX models, forecasting, diagnostic checking, unit root testing, time series decomposition, VAR models, and integration with Pandas time series functionality.

For more advanced topics, refer to the API documentation and examples in the following sections:

- :doc:`../api/models/time_series` - Detailed API documentation for time series models
- :doc:`../examples/time_series_analysis` - Additional examples and use cases