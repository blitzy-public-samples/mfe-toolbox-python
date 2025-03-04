.. _cross_sectional_analysis:

Cross-sectional Analysis
=======================

Introduction
-----------

The MFE Toolbox provides a comprehensive set of tools for cross-sectional data analysis, focusing on Ordinary Least Squares (OLS) regression and Principal Component Analysis (PCA). These tools are essential for financial and economic research, allowing users to analyze relationships between variables and reduce dimensionality in large datasets.

Unlike the previous MATLAB implementation, the Python version leverages NumPy's efficient linear algebra functions, Pandas for data handling, and Statsmodels for robust econometric analysis. The implementation also takes advantage of Python's modern features including dataclasses, type hints, and Numba acceleration for performance-critical operations.

Ordinary Least Squares (OLS)
---------------------------

The OLS implementation in the MFE Toolbox provides a comprehensive framework for linear regression analysis with robust standard errors, extensive diagnostics, and seamless integration with Pandas DataFrames.

Basic Usage
~~~~~~~~~~

.. code-block:: python

    import numpy as np
    import pandas as pd
    from mfe.models.cross_section import OLS

    # Create sample data
    np.random.seed(12345)
    n_obs = 100
    x1 = np.random.randn(n_obs)
    x2 = np.random.randn(n_obs)
    x3 = np.random.randn(n_obs)
    
    # True parameters
    beta = np.array([1.0, 0.5, -0.5, 2.0])
    
    # Generate dependent variable with noise
    X = np.column_stack((np.ones(n_obs), x1, x2, x3))
    y = X @ beta + np.random.randn(n_obs) * 0.5
    
    # Estimate OLS model
    model = OLS()
    results = model.fit(y, X)
    
    # Display results
    print(results)

Using with Pandas DataFrames
~~~~~~~~~~~~~~~~~~~~~~~~~~~

One of the advantages of the Python implementation is seamless integration with Pandas DataFrames, which allows for labeled data and more intuitive analysis:

.. code-block:: python

    import pandas as pd
    from mfe.models.cross_section import OLS
    
    # Create DataFrame with labeled data
    data = pd.DataFrame({
        'y': y,
        'x1': x1,
        'x2': x2,
        'x3': x3
    })
    
    # Estimate model with formula interface
    model = OLS()
    results = model.fit_from_formula('y ~ x1 + x2 + x3', data=data)
    
    # Access results with variable names
    print(f"Coefficient for x1: {results.params['x1']:.4f}")
    print(f"t-statistic for x1: {results.tvalues['x1']:.4f}")
    print(f"p-value for x1: {results.pvalues['x1']:.4f}")

Robust Standard Errors
~~~~~~~~~~~~~~~~~~~~

The OLS implementation supports various types of robust standard errors to account for heteroskedasticity and autocorrelation:

.. code-block:: python

    # Estimate with heteroskedasticity-robust standard errors (White)
    results_robust = model.fit(y, X, cov_type='HC0')
    
    # Estimate with Newey-West HAC standard errors
    results_hac = model.fit(y, X, cov_type='HAC', maxlags=10)
    
    # Compare standard errors
    print("Regular SE vs. Robust SE vs. HAC SE")
    for i, name in enumerate(['const', 'x1', 'x2', 'x3']):
        print(f"{name}: {results.std_errors[i]:.4f} vs. "
              f"{results_robust.std_errors[i]:.4f} vs. "
              f"{results_hac.std_errors[i]:.4f}")

Model Diagnostics
~~~~~~~~~~~~~~~

The OLS implementation provides comprehensive diagnostics for model evaluation:

.. code-block:: python

    # Model fit statistics
    print(f"R-squared: {results.rsquared:.4f}")
    print(f"Adjusted R-squared: {results.rsquared_adj:.4f}")
    print(f"F-statistic: {results.fvalue:.4f}")
    print(f"F p-value: {results.fpvalue:.4f}")
    
    # Residual diagnostics
    print(f"Durbin-Watson: {results.durbin_watson:.4f}")
    print(f"Jarque-Bera: {results.jarque_bera:.4f}, p-value: {results.jarque_bera_pvalue:.4f}")
    print(f"Breusch-Pagan: {results.breusch_pagan:.4f}, p-value: {results.breusch_pagan_pvalue:.4f}")

Visualization
~~~~~~~~~~~

The Python implementation makes it easy to visualize regression results using matplotlib and seaborn:

.. code-block:: python

    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Actual vs. Fitted values
    axes[0, 0].scatter(y, results.fitted_values)
    axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    axes[0, 0].set_xlabel('Actual')
    axes[0, 0].set_ylabel('Fitted')
    axes[0, 0].set_title('Actual vs. Fitted Values')
    
    # Residuals vs. Fitted values
    axes[0, 1].scatter(results.fitted_values, results.residuals)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Fitted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals vs. Fitted Values')
    
    # QQ plot of residuals
    from scipy import stats
    stats.probplot(results.residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Normal Q-Q Plot')
    
    # Residual histogram
    sns.histplot(results.residuals, kde=True, ax=axes[1, 1])
    axes[1, 1].set_title('Residual Distribution')
    
    plt.tight_layout()
    plt.show()

OLS Results Structure
~~~~~~~~~~~~~~~~~~

The OLS results are returned as a dataclass with comprehensive information:

.. code-block:: python

    from dataclasses import dataclass
    from typing import Dict, Optional, Union
    import numpy as np
    import pandas as pd
    
    @dataclass
    class OLSResults:
        """Results from OLS estimation."""
        params: Union[np.ndarray, pd.Series]
        std_errors: Union[np.ndarray, pd.Series]
        tvalues: Union[np.ndarray, pd.Series]
        pvalues: Union[np.ndarray, pd.Series]
        rsquared: float
        rsquared_adj: float
        fvalue: float
        fpvalue: float
        nobs: int
        df_model: int
        df_resid: int
        residuals: Union[np.ndarray, pd.Series]
        fitted_values: Union[np.ndarray, pd.Series]
        cov_params: Union[np.ndarray, pd.DataFrame]
        mse_model: float
        mse_resid: float
        mse_total: float
        ssr: float
        ess: float
        tss: float
        durbin_watson: float
        jarque_bera: float
        jarque_bera_pvalue: float
        breusch_pagan: float
        breusch_pagan_pvalue: float
        cov_type: str
        cov_kwds: Optional[Dict] = None
        
        def __str__(self):
            """String representation of OLS results."""
            # Implementation details...

Principal Component Analysis (PCA)
--------------------------------

The MFE Toolbox provides a comprehensive implementation of Principal Component Analysis (PCA) using NumPy's SVD and eigenvalue decomposition.

Basic Usage
~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from mfe.models.cross_section import PCA
    
    # Generate sample data
    np.random.seed(12345)
    n_obs = 100
    n_vars = 10
    
    # Create correlated data
    cov = np.eye(n_vars)
    # Add some correlation structure
    for i in range(n_vars):
        for j in range(n_vars):
            if i != j:
                cov[i, j] = 0.7 ** abs(i - j)
    
    X = np.random.multivariate_normal(np.zeros(n_vars), cov, size=n_obs)
    
    # Perform PCA
    pca = PCA()
    results = pca.fit(X)
    
    # Display results
    print(f"Number of components: {results.n_components}")
    print(f"Explained variance ratio: {results.explained_variance_ratio}")
    print(f"Cumulative explained variance: {results.explained_variance_ratio.cumsum()}")

PCA with Pandas DataFrames
~~~~~~~~~~~~~~~~~~~~~~~~

The PCA implementation works seamlessly with Pandas DataFrames:

.. code-block:: python

    import pandas as pd
    from mfe.models.cross_section import PCA
    
    # Create DataFrame with variable names
    columns = [f'var{i+1}' for i in range(n_vars)]
    df = pd.DataFrame(X, columns=columns)
    
    # Perform PCA
    pca = PCA()
    results = pca.fit(df)
    
    # Access loadings with variable names
    loadings_df = pd.DataFrame(
        results.loadings, 
        index=columns,
        columns=[f'PC{i+1}' for i in range(results.n_components)]
    )
    print("Component loadings:")
    print(loadings_df)

Selecting Number of Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PCA implementation provides several methods for selecting the number of components:

.. code-block:: python

    # PCA with automatic selection using explained variance threshold
    pca_var = PCA(n_components=0.9, svd_solver='full')  # Keep components explaining 90% of variance
    results_var = pca_var.fit(X)
    
    print(f"Selected {results_var.n_components} components explaining "
          f"{results_var.explained_variance_ratio.sum()*100:.2f}% of variance")
    
    # PCA with fixed number of components
    pca_fixed = PCA(n_components=3)
    results_fixed = pca_fixed.fit(X)
    
    print(f"Fixed 3 components explaining "
          f"{results_fixed.explained_variance_ratio.sum()*100:.2f}% of variance")

Visualization
~~~~~~~~~~

The Python implementation makes it easy to visualize PCA results:

.. code-block:: python

    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Scree plot
    axes[0, 0].plot(range(1, len(results.explained_variance_ratio) + 1), 
                   results.explained_variance_ratio, 'o-')
    axes[0, 0].set_xlabel('Principal Component')
    axes[0, 0].set_ylabel('Explained Variance Ratio')
    axes[0, 0].set_title('Scree Plot')
    
    # Cumulative explained variance
    axes[0, 1].plot(range(1, len(results.explained_variance_ratio) + 1), 
                   results.explained_variance_ratio.cumsum(), 'o-')
    axes[0, 1].axhline(y=0.9, color='r', linestyle='--', label='90% Threshold')
    axes[0, 1].set_xlabel('Number of Components')
    axes[0, 1].set_ylabel('Cumulative Explained Variance')
    axes[0, 1].set_title('Cumulative Explained Variance')
    axes[0, 1].legend()
    
    # Loadings heatmap
    sns.heatmap(loadings_df.iloc[:, :5], annot=True, cmap='coolwarm', ax=axes[1, 0])
    axes[1, 0].set_title('Component Loadings (First 5 PCs)')
    
    # Scatter plot of first two principal components
    transformed = results.transform(X)
    axes[1, 1].scatter(transformed[:, 0], transformed[:, 1], alpha=0.7)
    axes[1, 1].set_xlabel('PC1')
    axes[1, 1].set_ylabel('PC2')
    axes[1, 1].set_title('First Two Principal Components')
    
    plt.tight_layout()
    plt.show()

PCA Results Structure
~~~~~~~~~~~~~~~~~~

The PCA results are returned as a dataclass with comprehensive information:

.. code-block:: python

    from dataclasses import dataclass
    from typing import Optional, Union
    import numpy as np
    import pandas as pd
    
    @dataclass
    class PCAResults:
        """Results from PCA estimation."""
        components: np.ndarray
        explained_variance: np.ndarray
        explained_variance_ratio: np.ndarray
        singular_values: np.ndarray
        mean: np.ndarray
        n_components: int
        n_samples: int
        n_features: int
        noise_variance: float
        loadings: np.ndarray
        
        def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
            """Transform data to principal component space."""
            # Implementation details...
        
        def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
            """Transform data back to original space."""
            # Implementation details...
        
        def __str__(self):
            """String representation of PCA results."""
            # Implementation details...

Advanced PCA Features
~~~~~~~~~~~~~~~~~~

The PCA implementation includes several advanced features:

.. code-block:: python

    # PCA with different SVD solvers for performance optimization
    pca_full = PCA(svd_solver='full')  # Default, works for all cases
    pca_arpack = PCA(n_components=5, svd_solver='arpack')  # Efficient for large datasets, few components
    pca_randomized = PCA(n_components=5, svd_solver='randomized')  # Fast approximation
    
    # PCA with whitening
    pca_whitened = PCA(whiten=True)
    results_whitened = pca_whitened.fit(X)
    X_whitened = results_whitened.transform(X)
    
    # Verify whitened components have unit variance
    print("Variance of whitened components:")
    print(np.var(X_whitened, axis=0))

Integration with Statsmodels
--------------------------

The MFE Toolbox's cross-sectional analysis tools integrate with Statsmodels for additional functionality:

.. code-block:: python

    import statsmodels.api as sm
    from mfe.models.cross_section import OLS
    
    # MFE Toolbox OLS with Statsmodels integration
    X_with_const = sm.add_constant(X[:, 1:])  # Add constant using Statsmodels
    
    # Estimate using MFE Toolbox
    model = OLS()
    results = model.fit(y, X_with_const)
    
    # Compare with direct Statsmodels estimation
    sm_model = sm.OLS(y, X_with_const)
    sm_results = sm_model.fit()
    
    # Print comparison
    print("MFE Toolbox vs. Statsmodels:")
    print(f"R-squared: {results.rsquared:.6f} vs. {sm_results.rsquared:.6f}")
    
    # Access additional Statsmodels diagnostics
    from statsmodels.stats.diagnostic import het_breuschpagan
    bp_test = het_breuschpagan(results.residuals, X_with_const)
    print(f"Breusch-Pagan test: statistic={bp_test[0]:.4f}, p-value={bp_test[1]:.4f}")

Performance Optimization with Numba
---------------------------------

The cross-sectional analysis tools use Numba for performance-critical operations:

.. code-block:: python

    import time
    import numpy as np
    from mfe.models.cross_section import OLS
    
    # Generate large dataset
    np.random.seed(12345)
    n_obs = 10000
    n_vars = 20
    X = np.random.randn(n_obs, n_vars)
    X = np.column_stack((np.ones(n_obs), X))  # Add constant
    beta = np.random.randn(n_vars + 1)
    y = X @ beta + np.random.randn(n_obs) * 0.5
    
    # Time OLS estimation with Numba acceleration
    start_time = time.time()
    model = OLS()
    results = model.fit(y, X)
    numba_time = time.time() - start_time
    
    print(f"OLS estimation with Numba: {numba_time:.4f} seconds")
    
    # Note: The actual implementation uses Numba's @jit decorator for 
    # performance-critical operations like matrix calculations and 
    # standard error computations

Conclusion
---------

The cross-sectional analysis tools in the MFE Toolbox provide a comprehensive framework for OLS regression and PCA in Python. The implementation leverages NumPy's efficient linear algebra functions, Pandas for data handling, and Statsmodels for robust econometric analysis. The tools are designed to be intuitive, efficient, and compatible with the broader Python ecosystem.

For more detailed information on the API, please refer to the :ref:`API documentation <api_cross_section>`.