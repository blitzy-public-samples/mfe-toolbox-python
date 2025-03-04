============================
Multivariate Volatility Models
============================

This guide provides a comprehensive overview of the multivariate volatility modeling capabilities in the MFE Toolbox. These models are essential for analyzing volatility dynamics and correlations across multiple assets, making them crucial for portfolio risk management, asset allocation, and understanding cross-market dependencies.

Introduction to Multivariate Volatility Modeling
==============================================

Financial markets exhibit complex interdependencies where volatility and correlations between assets vary over time. Multivariate volatility models extend univariate volatility models to capture these dynamic relationships, allowing for more accurate risk assessment and portfolio optimization.

The general form of a multivariate volatility model can be expressed as:

.. math::

    \mathbf{r}_t &= \boldsymbol{\mu}_t + \boldsymbol{\varepsilon}_t \\
    \boldsymbol{\varepsilon}_t &= \mathbf{H}_t^{1/2} \mathbf{z}_t \\
    \mathbf{z}_t &\sim D(\mathbf{0}, \mathbf{I})

where:

- :math:`\mathbf{r}_t` is the :math:`k \times 1` vector of returns at time :math:`t`
- :math:`\boldsymbol{\mu}_t` is the :math:`k \times 1` vector of conditional means at time :math:`t` (often assumed to be constant or modeled separately)
- :math:`\boldsymbol{\varepsilon}_t` is the :math:`k \times 1` vector of innovations at time :math:`t`
- :math:`\mathbf{H}_t` is the :math:`k \times k` conditional covariance matrix at time :math:`t`
- :math:`\mathbf{H}_t^{1/2}` is a matrix square root of :math:`\mathbf{H}_t` such that :math:`\mathbf{H}_t^{1/2} (\mathbf{H}_t^{1/2})' = \mathbf{H}_t`
- :math:`\mathbf{z}_t` is a :math:`k \times 1` vector of standardized random variables following distribution :math:`D` with mean vector :math:`\mathbf{0}` and identity covariance matrix :math:`\mathbf{I}`

The MFE Toolbox implements various specifications for modeling the conditional covariance matrix :math:`\mathbf{H}_t`, each with different properties and capabilities.

Available Models
==============

The MFE Toolbox provides the following multivariate volatility models:

- **BEKK**: Baba-Engle-Kraft-Kroner model for direct covariance matrix modeling
- **CCC**: Constant Conditional Correlation model
- **DCC**: Dynamic Conditional Correlation model
- **ADCC**: Asymmetric Dynamic Conditional Correlation model
- **OGARCH/GOGARCH**: (Generalized) Orthogonal GARCH models
- **RARCH**: Rotated ARCH model
- **RCC**: Rotated Conditional Correlation model
- **Matrix GARCH**: Direct matrix generalization of GARCH
- **RiskMetrics**: Exponentially weighted moving average approach

All models are implemented as Python classes that inherit from a common base class, providing a consistent interface for model specification, estimation, and forecasting.

Model Specifications
==================

BEKK Model
---------

The BEKK model directly parameterizes the conditional covariance matrix:

.. math::

    \mathbf{H}_t = \mathbf{C}\mathbf{C}' + \sum_{i=1}^p \mathbf{A}_i \boldsymbol{\varepsilon}_{t-i}\boldsymbol{\varepsilon}_{t-i}' \mathbf{A}_i' + \sum_{j=1}^q \mathbf{B}_j \mathbf{H}_{t-j} \mathbf{B}_j'

where:

- :math:`\mathbf{C}` is a lower triangular :math:`k \times k` matrix
- :math:`\mathbf{A}_i` and :math:`\mathbf{B}_j` are :math:`k \times k` parameter matrices

In Python, you can create and estimate a BEKK(1,1) model as follows:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mfe.models.multivariate import BEKK
    from mfe.models.distributions import MultivariateNormal

    # Load multivariate return data (k assets)
    # returns should be a numpy array of shape (T, k) or pandas DataFrame
    returns = pd.read_csv('asset_returns.csv', index_col=0, parse_dates=True)
    
    # Create a BEKK(1,1) model with multivariate normal distribution
    model = BEKK(p=1, q=1, error_dist=MultivariateNormal())
    
    # Fit the model to return data
    result = model.fit(returns)
    
    # Print model summary
    print(result.summary())
    
    # Access model parameters
    C = result.params.C  # Constant matrix
    A = result.params.A  # ARCH matrix
    B = result.params.B  # GARCH matrix
    
    # Plot conditional volatilities (diagonal elements of H_t)
    volatilities = np.sqrt(np.array([H[i,i] for H in result.conditional_covariance for i in range(returns.shape[1])]))
    volatilities = volatilities.reshape(-1, returns.shape[1])
    
    plt.figure(figsize=(12, 6))
    for i in range(returns.shape[1]):
        plt.plot(returns.index, volatilities[:, i], label=f'Asset {i+1}')
    plt.title('BEKK(1,1) Conditional Volatilities')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend()
    plt.show()

The BEKK model can also include asymmetric effects:

.. code-block:: python

    # Create an asymmetric BEKK(1,1) model
    model = BEKK(p=1, q=1, asymmetric=True)
    
    # Fit the model
    result = model.fit(returns)
    
    # Access asymmetry parameter
    G = result.params.G  # Asymmetry matrix
    
    # Check for asymmetric effects
    print("Asymmetry matrix G:")
    print(G)
    
    # Test significance of asymmetric effects
    t_stats_G = result.t_stats.G
    print("t-statistics for asymmetry parameters:")
    print(t_stats_G)

DCC Model
--------

The Dynamic Conditional Correlation (DCC) model decomposes the conditional covariance matrix into conditional standard deviations and a conditional correlation matrix:

.. math::

    \mathbf{H}_t &= \mathbf{D}_t \mathbf{R}_t \mathbf{D}_t \\
    \mathbf{D}_t &= \text{diag}(\sigma_{1,t}, \sigma_{2,t}, \ldots, \sigma_{k,t}) \\
    \mathbf{R}_t &= \text{diag}(\mathbf{Q}_t)^{-1/2} \mathbf{Q}_t \text{diag}(\mathbf{Q}_t)^{-1/2} \\
    \mathbf{Q}_t &= (1 - \alpha - \beta) \bar{\mathbf{Q}} + \alpha \mathbf{z}_{t-1}\mathbf{z}_{t-1}' + \beta \mathbf{Q}_{t-1}

where:

- :math:`\sigma_{i,t}` is the conditional standard deviation of asset :math:`i` at time :math:`t`, typically from a univariate GARCH model
- :math:`\mathbf{R}_t` is the conditional correlation matrix at time :math:`t`
- :math:`\mathbf{Q}_t` is an auxiliary matrix used to ensure that :math:`\mathbf{R}_t` is a valid correlation matrix
- :math:`\bar{\mathbf{Q}}` is the unconditional covariance matrix of the standardized residuals :math:`\mathbf{z}_t = \mathbf{D}_t^{-1} \boldsymbol{\varepsilon}_t`
- :math:`\alpha` and :math:`\beta` are scalar parameters with :math:`\alpha, \beta \geq 0` and :math:`\alpha + \beta < 1`

Example usage:

.. code-block:: python

    from mfe.models.multivariate import DCC
    from mfe.models.univariate import GARCH
    
    # Create a DCC model with GARCH(1,1) for the univariate volatilities
    univariate_model = GARCH(p=1, q=1)
    model = DCC(univariate_model=univariate_model)
    
    # Fit the model using two-stage estimation
    result = model.fit(returns, method='two-stage')
    
    # Print model summary
    print(result.summary())
    
    # Access DCC parameters
    alpha = result.params.alpha
    beta = result.params.beta
    print(f"DCC parameters: alpha={alpha:.4f}, beta={beta:.4f}, persistence={alpha+beta:.4f}")
    
    # Plot conditional correlations
    correlations = np.array([R[0,1] for R in result.conditional_correlation])
    
    plt.figure(figsize=(12, 6))
    plt.plot(returns.index, correlations)
    plt.title('DCC(1,1) Conditional Correlation')
    plt.xlabel('Date')
    plt.ylabel('Correlation')
    plt.axhline(y=np.mean(correlations), color='r', linestyle='--', label='Average')
    plt.legend()
    plt.show()
    
    # Plot correlation heatmap at a specific time point
    import seaborn as sns
    
    # Get correlation matrix at the last time point
    last_corr = result.conditional_correlation[-1]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(last_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                xticklabels=returns.columns, yticklabels=returns.columns)
    plt.title(f'Conditional Correlation Matrix on {returns.index[-1].date()}')
    plt.tight_layout()
    plt.show()

The DCC model can also include asymmetric effects (ADCC):

.. code-block:: python

    # Create an asymmetric DCC model
    model = DCC(univariate_model=univariate_model, asymmetric=True)
    
    # Fit the model
    result = model.fit(returns)
    
    # Access asymmetry parameter
    gamma = result.params.gamma
    print(f"Asymmetry parameter: {gamma:.4f}")
    
    # Check for leverage effect
    if gamma > 0:
        print("Negative shocks increase correlation more than positive shocks")
    else:
        print("No asymmetric effect in correlations detected")

CCC Model
--------

The Constant Conditional Correlation (CCC) model is a simplified version of the DCC model where the correlation matrix is assumed to be constant over time:

.. math::

    \mathbf{H}_t &= \mathbf{D}_t \mathbf{R} \mathbf{D}_t 
    \mathbf{D}_t &= \text{diag}(\sigma_{1,t}, \sigma_{2,t}, \ldots, \sigma_{k,t})

where :math:`\mathbf{R}` is a constant correlation matrix.

Example usage:

.. code-block:: python

    from mfe.models.multivariate import CCC
    from mfe.models.univariate import GARCH
    
    # Create a CCC model with GARCH(1,1) for the univariate volatilities
    univariate_model = GARCH(p=1, q=1)
    model = CCC(univariate_model=univariate_model)
    
    # Fit the model
    result = model.fit(returns)
    
    # Print model summary
    print(result.summary())
    
    # Access constant correlation matrix
    R = result.params.R
    print("Constant Correlation Matrix:")
    print(R)
    
    # Visualize the correlation matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(R, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                xticklabels=returns.columns, yticklabels=returns.columns)
    plt.title('Constant Conditional Correlation Matrix')
    plt.tight_layout()
    plt.show()

OGARCH/GOGARCH Models
-------------------

The Orthogonal GARCH (OGARCH) model transforms the returns into orthogonal factors using principal component analysis (PCA) and then applies univariate GARCH models to these factors:

.. math::

    \mathbf{r}_t &= \mathbf{W} \mathbf{f}_t 
    \mathbf{f}_t &= \boldsymbol{\Lambda}^{1/2} \mathbf{u}_t 
    u_{i,t} &= \sigma_{i,t} z_{i,t}

where:

- :math:`\mathbf{W}` is the matrix of eigenvectors from the PCA
- :math:`\boldsymbol{\Lambda}` is the diagonal matrix of eigenvalues
- :math:`\mathbf{f}_t` are the orthogonal factors
- :math:`\mathbf{u}_t` are the standardized factors
- :math:`\sigma_{i,t}^2` follows a univariate GARCH process

Example usage:

.. code-block:: python

    from mfe.models.multivariate import OGARCH
    from mfe.models.univariate import GARCH
    
    # Create an OGARCH model with GARCH(1,1) for the factor volatilities
    univariate_model = GARCH(p=1, q=1)
    model = OGARCH(univariate_model=univariate_model, factors=None)  # Use all factors
    
    # Fit the model
    result = model.fit(returns)
    
    # Print model summary
    print(result.summary())
    
    # Access model components
    W = result.params.W  # Eigenvector matrix
    Lambda = result.params.Lambda  # Eigenvalue matrix
    
    # Calculate explained variance by each factor
    explained_variance = np.diag(Lambda) / np.sum(np.diag(Lambda))
    
    # Plot explained variance
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance)
    plt.title('Explained Variance by Principal Components')
    plt.xlabel('Principal Component')
    plt.ylabel('Proportion of Variance Explained')
    plt.xticks(range(1, len(explained_variance) + 1))
    plt.show()
    
    # Plot conditional volatilities of the first two factors
    factor_vols = np.array([np.sqrt(result.factor_variance[t][i]) 
                           for t in range(len(result.factor_variance)) 
                           for i in range(2)]).reshape(-1, 2)
    
    plt.figure(figsize=(12, 6))
    plt.plot(returns.index, factor_vols[:, 0], label='Factor 1')
    plt.plot(returns.index, factor_vols[:, 1], label='Factor 2')
    plt.title('OGARCH Factor Conditional Volatilities')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend()
    plt.show()

The Generalized Orthogonal GARCH (GOGARCH) model extends OGARCH by allowing for a more general linear transformation:

.. code-block:: python

    from mfe.models.multivariate import GOGARCH
    
    # Create a GOGARCH model
    model = GOGARCH(univariate_model=univariate_model)
    
    # Fit the model
    result = model.fit(returns)
    
    # Print model summary
    print(result.summary())

RARCH and RCC Models
------------------

The Rotated ARCH (RARCH) and Rotated Conditional Correlation (RCC) models use a data-driven approach to find an optimal rotation of the data:

.. code-block:: python

    from mfe.models.multivariate import RARCH, RCC
    
    # Create an RARCH model
    model = RARCH(univariate_model=univariate_model)
    
    # Fit the model
    result = model.fit(returns)
    
    # Print model summary
    print(result.summary())
    
    # Create an RCC model
    model_rcc = RCC(univariate_model=univariate_model)
    
    # Fit the model
    result_rcc = model_rcc.fit(returns)
    
    # Print model summary
    print(result_rcc.summary())

Matrix GARCH Model
---------------

The Matrix GARCH model directly parameterizes the conditional covariance matrix using a VECH representation:

.. code-block:: python

    from mfe.models.multivariate import MatrixGARCH
    
    # Create a Matrix GARCH model
    model = MatrixGARCH(p=1, q=1, form='diagonal')  # 'diagonal' or 'scalar'
    
    # Fit the model
    result = model.fit(returns)
    
    # Print model summary
    print(result.summary())

RiskMetrics Model
--------------

The RiskMetrics model uses an exponentially weighted moving average approach:

.. math::

    \mathbf{H}_t = (1 - \lambda) \boldsymbol{\varepsilon}_{t-1}\boldsymbol{\varepsilon}_{t-1}' + \lambda \mathbf{H}_{t-1}

where :math:`\lambda` is a decay factor (typically 0.94 for daily data).

.. code-block:: python

    from mfe.models.multivariate import RiskMetrics
    
    # Create a RiskMetrics model
    model = RiskMetrics(lambda_param=0.94)
    
    # Fit the model
    result = model.fit(returns)
    
    # Print model summary
    print(result.summary())
    
    # Access decay parameter
    lambda_param = result.params.lambda_param
    print(f"Decay parameter: {lambda_param:.4f}")

Model Estimation
==============

All multivariate volatility models in the MFE Toolbox follow a consistent estimation approach, typically using maximum likelihood estimation (MLE). The estimation process is optimized using Numba's just-in-time compilation for performance-critical operations.

Basic Estimation
--------------

The basic workflow for estimating a multivariate volatility model is:

1. Create a model instance with desired parameters
2. Call the `fit()` method with return data
3. Examine the results

.. code-block:: python

    from mfe.models.multivariate import DCC
    from mfe.models.univariate import GARCH
    from mfe.models.distributions import MultivariateStudentT
    
    # Create a model
    univariate_model = GARCH(p=1, q=1)
    model = DCC(univariate_model=univariate_model, 
                error_dist=MultivariateStudentT())
    
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
    conditional_covariance = result.conditional_covariance

Estimation Methods
---------------

For models like DCC, different estimation methods are available:

.. code-block:: python

    # Two-stage estimation (faster)
    result_two_stage = model.fit(returns, method='two-stage')
    
    # Joint estimation (more efficient but computationally intensive)
    result_joint = model.fit(returns, method='joint')
    
    # Compare log-likelihoods
    print(f"Two-stage log-likelihood: {result_two_stage.log_likelihood:.4f}")
    print(f"Joint log-likelihood: {result_joint.log_likelihood:.4f}")

Asynchronous Estimation
---------------------

For long-running estimations, the MFE Toolbox provides asynchronous versions of the estimation methods:

.. code-block:: python

    import asyncio
    from mfe.models.multivariate import BEKK
    
    async def estimate_model_async():
        # Create a model
        model = BEKK(p=1, q=1)
        
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

    from mfe.models.multivariate import DCC
    from mfe.core.parameters import DCCParams
    
    # Create starting parameter values
    starting_params = DCCParams(
        alpha=0.05,
        beta=0.90
    )
    
    # Create and fit the model with custom starting values
    model = DCC(univariate_model=univariate_model)
    result = model.fit(returns, starting_values=starting_params)
    
    print(result.summary())

Error Distributions
----------------

The MFE Toolbox supports various multivariate error distributions:

- **MultivariateNormal**: Multivariate normal distribution
- **MultivariateStudentT**: Multivariate Student's t-distribution with estimated degrees of freedom
- **MultivariateGED**: Multivariate Generalized Error Distribution

Example with Multivariate Student's t-distribution:

.. code-block:: python

    from mfe.models.multivariate import DCC
    from mfe.models.distributions import MultivariateStudentT
    
    # Create a DCC model with Multivariate Student's t errors
    model = DCC(univariate_model=univariate_model, 
                error_dist=MultivariateStudentT())
    
    # Fit the model
    result = model.fit(returns)
    
    # Access distribution parameters
    df = result.params.df  # Degrees of freedom
    print(f"Estimated degrees of freedom: {df:.4f}")
    
    # Test for fat tails
    if df < 10:
        print("Evidence of fat tails in the multivariate return distribution")
    else:
        print("Multivariate return distribution close to normal")

Composite Likelihood Estimation
----------------------------

For high-dimensional problems, composite likelihood estimation can be used to make estimation feasible:

.. code-block:: python

    from mfe.models.multivariate import DCC
    
    # Create a DCC model
    model = DCC(univariate_model=univariate_model)
    
    # Fit the model using composite likelihood
    result = model.fit(returns, method='composite', 
                       composite_method='pairwise')
    
    # Print model summary
    print(result.summary())

Model Diagnostics
===============

After estimating a multivariate volatility model, it's important to check its adequacy through various diagnostic tests.

Standardized Residuals
--------------------

Examining the standardized residuals is a key diagnostic:

.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    
    # Get standardized residuals
    std_residuals = result.standardized_residuals  # This is a T x k matrix
    
    # Plot standardized residuals for each series
    plt.figure(figsize=(15, 10))
    
    k = std_residuals.shape[1]  # Number of assets
    for i in range(k):
        plt.subplot(k, 2, 2*i+1)
        plt.plot(returns.index, std_residuals[:, i])
        plt.title(f'Standardized Residuals - Asset {i+1}')
        plt.axhline(y=0, color='r', linestyle='-')
        
        plt.subplot(k, 2, 2*i+2)
        plt.hist(std_residuals[:, i], bins=50, density=True, alpha=0.6, color='g')
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, 0, 1)
        plt.plot(x, p, 'k', linewidth=2)
        plt.title(f'Histogram - Asset {i+1}')
    
    plt.tight_layout()
    plt.show()

Multivariate Normality Tests
-------------------------

Test for multivariate normality of the standardized residuals:

.. code-block:: python

    from mfe.models.tests import MultivariateJarqueBera, MardiasTest
    
    # Multivariate Jarque-Bera test
    mjb_test = MultivariateJarqueBera()
    mjb_result = mjb_test.run(result.standardized_residuals)
    print("Multivariate Jarque-Bera Test:")
    print(f"Test statistic: {mjb_result.statistic:.4f}")
    print(f"p-value: {mjb_result.p_value:.4f}")
    if mjb_result.p_value > 0.05:
        print("Standardized residuals appear multivariate normal")
    else:
        print("Standardized residuals are not multivariate normal")
    
    # Mardia's test for multivariate normality
    mardia_test = MardiasTest()
    mardia_result = mardia_test.run(result.standardized_residuals)
    print("\nMardia's Test for Multivariate Normality:")
    print(f"Skewness statistic: {mardia_result.skewness_statistic:.4f}")
    print(f"Skewness p-value: {mardia_result.skewness_p_value:.4f}")
    print(f"Kurtosis statistic: {mardia_result.kurtosis_statistic:.4f}")
    print(f"Kurtosis p-value: {mardia_result.kurtosis_p_value:.4f}")

Correlation Tests
--------------

Test for constant correlation (useful for evaluating DCC vs. CCC):

.. code-block:: python

    from mfe.models.tests import ConstantCorrelationTest
    
    # Test for constant correlation
    cc_test = ConstantCorrelationTest()
    cc_result = cc_test.run(returns)
    print("Constant Correlation Test:")
    print(f"Test statistic: {cc_result.statistic:.4f}")
    print(f"p-value: {cc_result.p_value:.4f}")
    if cc_result.p_value > 0.05:
        print("No evidence against constant correlation")
    else:
        print("Evidence of time-varying correlation")

Model Comparison
-------------

Compare different models using information criteria:

.. code-block:: python

    from mfe.models.multivariate import DCC, BEKK, CCC
    from mfe.models.univariate import GARCH
    
    # Create and fit different models
    univariate_model = GARCH(p=1, q=1)
    
    models = {
        'DCC': DCC(univariate_model=univariate_model),
        'BEKK': BEKK(p=1, q=1),
        'CCC': CCC(univariate_model=univariate_model)
    }
    
    results = {}
    for name, model in models.items():
        print(f"Estimating {name}...")
        results[name] = model.fit(returns)
    
    # Compare models using information criteria
    print("\nModel Comparison:")
    print(f"{'Model':<10} {'Log-Likelihood':<15} {'AIC':<10} {'BIC':<10}")
    print("-" * 45)
    for name, result in results.items():
        print(f"{name:<10} {result.log_likelihood:<15.2f} {result.aic:<10.2f} {result.bic:<10.2f}")
    
    # Find the best model according to AIC
    best_aic = min(results.items(), key=lambda x: x[1].aic)
    print(f"\nBest model according to AIC: {best_aic[0]}")
    
    # Find the best model according to BIC
    best_bic = min(results.items(), key=lambda x: x[1].bic)
    print(f"Best model according to BIC: {best_bic[0]}")

Forecasting
=========

Multivariate volatility forecasting is essential for portfolio risk management and asset allocation.

Point Forecasts
-------------

Generate point forecasts for future covariance matrices:

.. code-block:: python

    from mfe.models.multivariate import DCC
    
    # Create and fit a DCC model
    univariate_model = GARCH(p=1, q=1)
    model = DCC(univariate_model=univariate_model)
    result = model.fit(returns)
    
    # Generate 10-day ahead covariance matrix forecasts
    forecasts = result.forecast(horizon=10)
    
    # Print volatility forecasts (diagonal elements of covariance matrices)
    print("Volatility Forecasts (Standard Deviations):")
    for h in range(10):
        volatilities = np.sqrt(np.diag(forecasts.covariance[h]))
        print(f"h={h+1}: {volatilities}")
    
    # Print correlation forecasts
    print("\nCorrelation Forecasts:")
    for h in range(10):
        # Convert covariance to correlation
        D = np.diag(np.sqrt(np.diag(forecasts.covariance[h])))
        D_inv = np.linalg.inv(D)
        corr = D_inv @ forecasts.covariance[h] @ D_inv
        print(f"h={h+1}:")
        print(corr)

Simulation-Based Forecasts
------------------------

For more accurate forecasts, especially at longer horizons, simulation-based methods are recommended:

.. code-block:: python

    # Generate simulation-based forecasts
    sim_forecasts = result.forecast(horizon=10, method='simulation', num_simulations=10000)
    
    # Print mean forecasts
    print("Simulation-Based Volatility Forecasts (Standard Deviations):")
    for h in range(10):
        volatilities = np.sqrt(np.diag(sim_forecasts.covariance[h]))
        print(f"h={h+1}: {volatilities}")
    
    # Plot forecasts for the first asset's volatility with confidence intervals
    plt.figure(figsize=(10, 6))
    
    # Historical volatility
    hist_vol = np.sqrt([H[0,0] for H in result.conditional_covariance])
    plt.plot(returns.index, hist_vol, label='In-sample Volatility')
    
    # Forecast volatility
    forecast_index = pd.date_range(
        start=returns.index[-1] + pd.Timedelta(days=1),
        periods=10,
        freq=pd.infer_freq(returns.index)
    )
    
    forecast_vol = np.sqrt([H[0,0] for H in sim_forecasts.covariance])
    forecast_vol_lower = np.sqrt([H[0,0] for H in sim_forecasts.covariance_lower])
    forecast_vol_upper = np.sqrt([H[0,0] for H in sim_forecasts.covariance_upper])
    
    plt.plot(forecast_index, forecast_vol, 'r--', label='Forecast Volatility')
    plt.fill_between(forecast_index, forecast_vol_lower, forecast_vol_upper, 
                     color='r', alpha=0.2, label='95% Confidence Interval')
    
    plt.title('DCC Simulation-Based Volatility Forecast - Asset 1')
    plt.xlabel('Date')
    plt.ylabel('Volatility (Standard Deviation)')
    plt.legend()
    plt.show()

Asynchronous Forecasting
----------------------

For long-horizon forecasts or large simulation counts, asynchronous forecasting is available:

.. code-block:: python

    import asyncio
    
    async def generate_forecasts_async():
        # Create and fit a DCC model
        univariate_model = GARCH(p=1, q=1)
        model = DCC(univariate_model=univariate_model)
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
    plt.plot(range(len(forecasts.covariance)), 
             np.sqrt([H[0,0] for H in forecasts.covariance]), 
             'r-', label='Mean Forecast')
    plt.fill_between(range(len(forecasts.covariance)), 
                    np.sqrt([H[0,0] for H in forecasts.covariance_lower]), 
                    np.sqrt([H[0,0] for H in forecasts.covariance_upper]), 
                    color='r', alpha=0.2, label='95% Confidence Interval')
    plt.title('Long-Horizon DCC Volatility Forecast - Asset 1')
    plt.xlabel('Horizon')
    plt.ylabel('Volatility (Standard Deviation)')
    plt.legend()
    plt.show()

Portfolio Value-at-Risk Forecasting
--------------------------------

Multivariate volatility models are particularly useful for portfolio risk assessment:

.. code-block:: python

    from mfe.models.multivariate import DCC
    from scipy import stats
    
    # Create and fit a DCC model
    univariate_model = GARCH(p=1, q=1)
    model = DCC(univariate_model=univariate_model, 
                error_dist=MultivariateStudentT())
    result = model.fit(returns)
    
    # Define portfolio weights
    weights = np.array([1/returns.shape[1]] * returns.shape[1])  # Equal weights
    
    # Generate 1-day ahead forecast
    forecast = result.forecast(horizon=1)
    
    # Calculate 1-day 99% portfolio VaR
    # Portfolio variance
    portfolio_var = weights @ forecast.covariance[0] @ weights
    portfolio_std = np.sqrt(portfolio_var)
    
    # For Student's t, we need the quantile from the t-distribution
    df = result.params.df  # Degrees of freedom
    t_quantile = stats.t.ppf(0.01, df)  # 1% quantile
    
    # VaR calculation (assuming zero mean)
    portfolio_var_99 = t_quantile * portfolio_std
    
    print(f"1-day ahead 99% Portfolio VaR: {portfolio_var_99:.6f}")
    
    # For comparison, calculate VaR assuming normal distribution
    normal_quantile = stats.norm.ppf(0.01)  # 1% quantile
    portfolio_var_99_normal = normal_quantile * portfolio_std
    
    print(f"1-day ahead 99% Portfolio VaR (normal assumption): {portfolio_var_99_normal:.6f}")

Model Simulation
=============

The MFE Toolbox allows you to simulate data from estimated multivariate volatility models:

.. code-block:: python

    from mfe.models.multivariate import DCC
    from mfe.core.parameters import DCCParams, GARCHParams
    
    # Create a DCC model
    univariate_model = GARCH(p=1, q=1)
    model = DCC(univariate_model=univariate_model)
    
    # Set univariate parameters
    univariate_params = [
        GARCHParams(omega=0.00001, alpha=0.05, beta=0.90),
        GARCHParams(omega=0.00002, alpha=0.07, beta=0.88)
    ]
    
    # Set DCC parameters
    dcc_params = DCCParams(alpha=0.05, beta=0.90)
    
    # Simulate 1000 observations
    simulated_data = model.simulate(
        univariate_params=univariate_params,
        correlation_params=dcc_params,
        num_obs=1000,
        burn=500,  # Burn-in period to remove initialization effects
        initial_value=None  # Use default initialization
    )
    
    # Plot simulated returns
    plt.figure(figsize=(12, 8))
    
    # Returns
    plt.subplot(2, 1, 1)
    plt.plot(simulated_data.returns[:, 0], label='Asset 1')
    plt.plot(simulated_data.returns[:, 1], label='Asset 2')
    plt.title('Simulated Returns from DCC(1,1)')
    plt.ylabel('Returns')
    plt.legend()
    
    # Correlation
    plt.subplot(2, 1, 2)
    plt.plot(simulated_data.conditional_correlation)
    plt.title('Simulated Conditional Correlation from DCC(1,1)')
    plt.ylabel('Correlation')
    plt.xlabel('Time')
    
    plt.tight_layout()
    plt.show()

Asynchronous Simulation
--------------------

For large simulations, asynchronous processing is available:

.. code-block:: python

    import asyncio
    
    async def simulate_dcc_async():
        # Create a DCC model
        univariate_model = GARCH(p=1, q=1)
        model = DCC(univariate_model=univariate_model)
        
        # Set parameters
        univariate_params = [
            GARCHParams(omega=0.00001, alpha=0.05, beta=0.90),
            GARCHParams(omega=0.00002, alpha=0.07, beta=0.88)
        ]
        dcc_params = DCCParams(alpha=0.05, beta=0.90)
        
        # Define a progress callback
        def progress_callback(percent, message):
            print(f"{percent:.1f}% complete: {message}")
        
        # Simulate data asynchronously
        simulated_data = await model.simulate_async(
            univariate_params=univariate_params,
            correlation_params=dcc_params,
            num_obs=10000,
            burn=1000,
            progress_callback=progress_callback
        )
        
        return simulated_data
    
    # Run the async function
    simulated_data = asyncio.run(simulate_dcc_async())
    
    # Plot a sample of the simulated data
    plt.figure(figsize=(12, 6))
    plt.plot(simulated_data.conditional_correlation[:1000])
    plt.title('Sample of Simulated Conditional Correlation from DCC(1,1)')
    plt.xlabel('Time')
    plt.ylabel('Correlation')
    plt.show()

Advanced Topics
=============

Portfolio Optimization
-------------------

Multivariate volatility models are particularly useful for portfolio optimization:

.. code-block:: python

    from mfe.models.multivariate import DCC
    import scipy.optimize as sco
    
    # Create and fit a DCC model
    univariate_model = GARCH(p=1, q=1)
    model = DCC(univariate_model=univariate_model)
    result = model.fit(returns)
    
    # Get the latest conditional covariance matrix
    latest_cov = result.conditional_covariance[-1]
    
    # Define the negative Sharpe ratio (to minimize)
    def neg_sharpe_ratio(weights, cov_matrix, returns):
        weights = np.array(weights)
        portfolio_std = np.sqrt(weights @ cov_matrix @ weights)
        mean_return = np.mean(returns, axis=0)
        portfolio_return = np.sum(mean_return * weights)
        return -portfolio_return / portfolio_std
    
    # Define constraints
    n_assets = returns.shape[1]
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights sum to 1
    bounds = tuple((0, 1) for _ in range(n_assets))  # Weights between 0 and 1
    
    # Initial guess (equal weights)
    initial_weights = np.array([1/n_assets] * n_assets)
    
    # Optimize portfolio
    result_opt = sco.minimize(
        neg_sharpe_ratio,
        initial_weights,
        args=(latest_cov, returns),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    # Get optimal weights
    optimal_weights = result_opt['x']
    
    # Print results
    print("Optimal Portfolio Weights:")
    for i, weight in enumerate(optimal_weights):
        print(f"Asset {i+1}: {weight:.4f}")
    
    # Calculate portfolio metrics
    portfolio_std = np.sqrt(optimal_weights @ latest_cov @ optimal_weights)
    mean_return = np.mean(returns, axis=0)
    portfolio_return = np.sum(mean_return * optimal_weights)
    sharpe_ratio = portfolio_return / portfolio_std
    
    print(f"\nPortfolio Expected Return: {portfolio_return:.6f}")
    print(f"Portfolio Volatility: {portfolio_std:.6f}")
    print(f"Portfolio Sharpe Ratio: {sharpe_ratio:.6f}")
    
    # Compare with equal-weighted portfolio
    equal_weights = np.array([1/n_assets] * n_assets)
    equal_std = np.sqrt(equal_weights @ latest_cov @ equal_weights)
    equal_return = np.sum(mean_return * equal_weights)
    equal_sharpe = equal_return / equal_std
    
    print("\nEqual-Weighted Portfolio:")
    print(f"Portfolio Expected Return: {equal_return:.6f}")
    print(f"Portfolio Volatility: {equal_std:.6f}")
    print(f"Portfolio Sharpe Ratio: {equal_sharpe:.6f}")

Efficient Frontier
---------------

Generate the efficient frontier using a multivariate volatility model:

.. code-block:: python

    from mfe.models.multivariate import DCC
    import scipy.optimize as sco
    
    # Create and fit a DCC model
    univariate_model = GARCH(p=1, q=1)
    model = DCC(univariate_model=univariate_model)
    result = model.fit(returns)
    
    # Get the latest conditional covariance matrix
    latest_cov = result.conditional_covariance[-1]
    
    # Define the portfolio variance function
    def portfolio_variance(weights, cov_matrix):
        weights = np.array(weights)
        return weights @ cov_matrix @ weights
    
    # Define the portfolio return function
    def portfolio_return(weights, mean_returns):
        weights = np.array(weights)
        return np.sum(mean_returns * weights)
    
    # Define constraints
    n_assets = returns.shape[1]
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights sum to 1
    bounds = tuple((0, 1) for _ in range(n_assets))  # Weights between 0 and 1
    
    # Calculate mean returns
    mean_returns = np.mean(returns, axis=0)
    
    # Generate efficient frontier
    target_returns = np.linspace(np.min(mean_returns), np.max(mean_returns), 50)
    efficient_risk = []
    efficient_weights = []
    
    for target in target_returns:
        # Define constraints including target return
        constraints_return = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
            {'type': 'eq', 'fun': lambda x: portfolio_return(x, mean_returns) - target}  # Target return
        )
        
        # Initial guess (equal weights)
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Minimize portfolio variance for the target return
        result_opt = sco.minimize(
            portfolio_variance,
            initial_weights,
            args=(latest_cov,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_return
        )
        
        if result_opt['success']:
            efficient_risk.append(np.sqrt(result_opt['fun']))
            efficient_weights.append(result_opt['x'])
    
    # Plot efficient frontier
    plt.figure(figsize=(12, 8))
    plt.plot(efficient_risk, target_returns, 'b-', linewidth=2)
    plt.scatter(efficient_risk, target_returns, c='b', marker='o')
    
    # Plot individual assets
    asset_std = np.sqrt(np.diag(latest_cov))
    plt.scatter(asset_std, mean_returns, c='r', marker='*', s=100, label='Individual Assets')
    
    # Calculate and plot the minimum variance portfolio
    min_var_result = sco.minimize(
        portfolio_variance,
        np.array([1/n_assets] * n_assets),
        args=(latest_cov,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    min_var_std = np.sqrt(min_var_result['fun'])
    min_var_return = portfolio_return(min_var_result['x'], mean_returns)
    plt.scatter(min_var_std, min_var_return, c='g', marker='D', s=100, label='Minimum Variance')
    
    # Calculate and plot the tangency portfolio (maximum Sharpe ratio)
    def neg_sharpe_ratio(weights, cov_matrix, returns):
        weights = np.array(weights)
        portfolio_std = np.sqrt(weights @ cov_matrix @ weights)
        portfolio_return = np.sum(returns * weights)
        return -portfolio_return / portfolio_std
    
    tangency_result = sco.minimize(
        neg_sharpe_ratio,
        np.array([1/n_assets] * n_assets),
        args=(latest_cov, mean_returns),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    tangency_std = np.sqrt(portfolio_variance(tangency_result['x'], latest_cov))
    tangency_return = portfolio_return(tangency_result['x'], mean_returns)
    plt.scatter(tangency_std, tangency_return, c='y', marker='P', s=100, label='Maximum Sharpe Ratio')
    
    plt.title('Efficient Frontier with DCC Conditional Covariance')
    plt.xlabel('Portfolio Risk (Standard Deviation)')
    plt.ylabel('Portfolio Expected Return')
    plt.legend()
    plt.grid(True)
    plt.show()

Numba Acceleration
---------------

The MFE Toolbox uses Numba's just-in-time (JIT) compilation to accelerate performance-critical functions. This is handled automatically, but you can see the performance improvement:

.. code-block:: python

    import time
    import numpy as np
    from mfe.models.multivariate._numba_core import dcc_recursion
    from numba import jit
    
    # Generate test data
    np.random.seed(42)
    T = 1000
    k = 5
    standardized_residuals = np.random.normal(0, 1, (T, k))
    
    # Define parameters
    alpha = 0.05
    beta = 0.90
    
    # Create a pure Python version of the DCC recursion
    def dcc_recursion_python(parameters, standardized_residuals, Q_bar):
        T, k = standardized_residuals.shape
        alpha, beta = parameters
        
        Q = np.zeros((T, k, k))
        R = np.zeros((T, k, k))
        
        # Initialize with unconditional correlation
        Q[0] = Q_bar.copy()
        
        # Compute Q_t recursively
        for t in range(1, T):
            z = standardized_residuals[t-1]
            Q[t] = (1 - alpha - beta) * Q_bar + alpha * np.outer(z, z) + beta * Q[t-1]
            
            # Compute correlation matrix
            Q_diag_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(Q[t])))
            R[t] = Q_diag_inv_sqrt @ Q[t] @ Q_diag_inv_sqrt
            
        return Q, R
    
    # Prepare inputs
    Q_bar = np.corrcoef(standardized_residuals, rowvar=False)
    
    # Time the pure Python version
    start_time = time.time()
    Q_py, R_py = dcc_recursion_python([alpha, beta], standardized_residuals, Q_bar)
    python_time = time.time() - start_time
    
    # Time the Numba-accelerated version
    start_time = time.time()
    Q_nb, R_nb = dcc_recursion(np.array([alpha, beta]), standardized_residuals, Q_bar)
    numba_time = time.time() - start_time
    
    # Print results
    print(f"Pure Python time: {python_time:.6f} seconds")
    print(f"Numba-accelerated time: {numba_time:.6f} seconds")
    print(f"Speedup factor: {python_time / numba_time:.2f}x")
    
    # Verify results are the same
    print(f"Maximum absolute difference in Q: {np.max(np.abs(Q_py - Q_nb)):.8f}")
    print(f"Maximum absolute difference in R: {np.max(np.abs(R_py - R_nb)):.8f}")

Conclusion
=========

The multivariate volatility models in the MFE Toolbox provide a comprehensive suite of tools for modeling time-varying covariance matrices and correlations between multiple assets. These models are essential for portfolio risk management, asset allocation, and understanding cross-market dependencies.

The Python implementation with Numba acceleration offers both ease of use and high performance, making it suitable for both research and practical applications. The consistent API across different model types simplifies the process of comparing and selecting the most appropriate model for a given dataset.

For more advanced applications, see the documentation on realized covariance estimators, bootstrap methods for multivariate time series, and portfolio optimization techniques.
