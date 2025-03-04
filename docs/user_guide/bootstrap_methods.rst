==================
Bootstrap Methods
==================

This guide provides a comprehensive overview of bootstrap methods for dependent data in the MFE Toolbox. Bootstrap techniques are essential for statistical inference with financial time series data where standard independence assumptions are invalid.

Introduction
===========

Bootstrap methods are resampling techniques that allow for statistical inference without making strong distributional assumptions. In financial econometrics, where data often exhibits serial dependence, specialized bootstrap methods are required to preserve the dependence structure.

The MFE Toolbox provides several bootstrap methods specifically designed for dependent data:

- **Block Bootstrap**: Resamples blocks of consecutive observations to preserve short-range dependence
- **Stationary Bootstrap**: Uses random block lengths for improved stationarity properties
- **Model Confidence Set (MCS)**: Identifies the set of models that are statistically indistinguishable from the best model
- **Bootstrap Reality Check and SPA Test**: Tests for superior predictive ability among competing forecasting models

All bootstrap implementations in the MFE Toolbox leverage NumPy for efficient array operations and Numba for performance acceleration of computationally intensive resampling algorithms.

Block Bootstrap
=============

The block bootstrap method resamples blocks of consecutive observations to preserve the dependence structure in the data.

Basic Usage
---------

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mfe.models.bootstrap import BlockBootstrap
    
    # Generate example time series data
    np.random.seed(42)
    n = 1000
    ar_coef = 0.7
    
    # Create an AR(1) process
    data = np.zeros(n)
    data[0] = np.random.normal(0, 1)
    for t in range(1, n):
        data[t] = ar_coef * data[t-1] + np.random.normal(0, 1)
    
    # Create a block bootstrap with block size 50
    bootstrap = BlockBootstrap(block_size=50)
    
    # Generate 1000 bootstrap samples
    bootstrap_samples = bootstrap.generate(data, num_samples=1000)
    
    # Compute bootstrap statistics (e.g., mean of each sample)
    bootstrap_means = np.array([sample.mean() for sample in bootstrap_samples])
    
    # Plot the bootstrap distribution
    plt.figure(figsize=(10, 6))
    plt.hist(bootstrap_means, bins=50, alpha=0.7)
    plt.axvline(data.mean(), color='r', linestyle='--', label='Sample Mean')
    plt.title('Block Bootstrap Distribution of Mean')
    plt.xlabel('Mean')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    
    # Compute bootstrap confidence interval
    conf_interval = np.percentile(bootstrap_means, [2.5, 97.5])
    print(f"95% Bootstrap Confidence Interval: [{conf_interval[0]:.6f}, {conf_interval[1]:.6f}]")

Advanced Configuration
-------------------

The ``BlockBootstrap`` class provides several configuration options:

.. code-block:: python

    from mfe.models.bootstrap import BlockBootstrap
    
    # Create a block bootstrap with custom configuration
    bootstrap = BlockBootstrap(
        block_size=50,           # Size of each block
        wrap=True,               # Whether to use circular blocks
        seed=42                  # Random seed for reproducibility
    )
    
    # Generate bootstrap samples with additional options
    bootstrap_samples = bootstrap.generate(
        data,                    # Input data
        num_samples=1000,        # Number of bootstrap samples
        sample_size=None,        # Size of each sample (default: same as input)
        rng=None                 # Custom random number generator
    )

Numba-Accelerated Implementation
-----------------------------

The block bootstrap implementation uses Numba's just-in-time compilation for performance-critical operations:

.. code-block:: python

    import time
    import numpy as np
    from mfe.models.bootstrap import BlockBootstrap
    
    # Generate large time series
    np.random.seed(42)
    n = 10000
    data = np.random.normal(0, 1, n)
    
    # Measure performance
    start_time = time.time()
    bootstrap = BlockBootstrap(block_size=100)
    samples = bootstrap.generate(data, num_samples=500)
    end_time = time.time()
    
    print(f"Generated 500 bootstrap samples from 10,000 observations in {end_time - start_time:.2f} seconds")

Stationary Bootstrap
==================

The stationary bootstrap improves upon the block bootstrap by using random block lengths, which enhances the stationarity properties of the resampled series.

Basic Usage
---------

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from mfe.models.bootstrap import StationaryBootstrap
    
    # Generate example time series data
    np.random.seed(42)
    n = 1000
    ar_coef = 0.7
    
    # Create an AR(1) process
    data = np.zeros(n)
    data[0] = np.random.normal(0, 1)
    for t in range(1, n):
        data[t] = ar_coef * data[t-1] + np.random.normal(0, 1)
    
    # Create a stationary bootstrap with expected block size 50
    bootstrap = StationaryBootstrap(expected_block_size=50)
    
    # Generate 1000 bootstrap samples
    bootstrap_samples = bootstrap.generate(data, num_samples=1000)
    
    # Compute bootstrap statistics (e.g., standard deviation of each sample)
    bootstrap_stds = np.array([sample.std() for sample in bootstrap_samples])
    
    # Plot the bootstrap distribution
    plt.figure(figsize=(10, 6))
    plt.hist(bootstrap_stds, bins=50, alpha=0.7)
    plt.axvline(data.std(), color='r', linestyle='--', label='Sample Std Dev')
    plt.title('Stationary Bootstrap Distribution of Standard Deviation')
    plt.xlabel('Standard Deviation')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    
    # Compute bootstrap confidence interval
    conf_interval = np.percentile(bootstrap_stds, [2.5, 97.5])
    print(f"95% Bootstrap Confidence Interval for Std Dev: [{conf_interval[0]:.6f}, {conf_interval[1]:.6f}]")

Advanced Configuration
-------------------

The ``StationaryBootstrap`` class provides several configuration options:

.. code-block:: python

    from mfe.models.bootstrap import StationaryBootstrap
    
    # Create a stationary bootstrap with custom configuration
    bootstrap = StationaryBootstrap(
        expected_block_size=50,  # Expected size of each block
        seed=42                  # Random seed for reproducibility
    )
    
    # Generate bootstrap samples with additional options
    bootstrap_samples = bootstrap.generate(
        data,                    # Input data
        num_samples=1000,        # Number of bootstrap samples
        sample_size=None,        # Size of each sample (default: same as input)
        rng=None                 # Custom random number generator
    )

Asynchronous Processing with Progress Tracking
-------------------------------------------

For large-scale bootstrap operations, the MFE Toolbox provides asynchronous processing with progress tracking:

.. code-block:: python

    import asyncio
    import numpy as np
    from mfe.models.bootstrap import StationaryBootstrap
    
    async def run_bootstrap_async():
        # Generate example data
        np.random.seed(42)
        n = 5000
        data = np.random.normal(0, 1, n)
        
        # Create a stationary bootstrap
        bootstrap = StationaryBootstrap(expected_block_size=50)
        
        # Define a progress callback function
        def progress_callback(percent, message):
            print(f"{percent:.1f}% complete: {message}")
        
        # Generate 10,000 bootstrap samples asynchronously with progress tracking
        bootstrap_samples = await bootstrap.generate_async(
            data, 
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

Model Confidence Set (MCS)
========================

The Model Confidence Set (MCS) procedure identifies the set of models that are statistically indistinguishable from the best model based on a user-defined loss function.

Basic Usage
---------

.. code-block:: python

    import numpy as np
    import pandas as pd
    from mfe.models.bootstrap import ModelConfidenceSet
    
    # Generate loss data for 5 different models over 100 time periods
    np.random.seed(42)
    n_models = 5
    n_periods = 100
    
    # Model 0 and 1 are the best, others are worse
    base_losses = np.random.normal(0, 1, (n_periods, n_models))
    base_losses[:, 2:] += 0.5  # Models 2-4 have higher loss
    
    # Create a Model Confidence Set
    mcs = ModelConfidenceSet(
        block_size=10,           # Block size for bootstrap
        num_bootstrap=1000,      # Number of bootstrap replications
        significance_level=0.05  # Significance level
    )
    
    # Run the MCS procedure
    mcs_result = mcs.run(base_losses)
    
    # Print results
    print("Model Confidence Set Results:")
    print(f"Included models: {mcs_result.included_models}")
    print(f"Excluded models: {mcs_result.excluded_models}")
    print("\nModel p-values:")
    for i, p_val in enumerate(mcs_result.pvalues):
        print(f"Model {i}: {p_val:.4f}")

Advanced Configuration
-------------------

The ``ModelConfidenceSet`` class provides several configuration options:

.. code-block:: python

    from mfe.models.bootstrap import ModelConfidenceSet
    
    # Create a Model Confidence Set with custom configuration
    mcs = ModelConfidenceSet(
        block_size=10,           # Block size for bootstrap
        num_bootstrap=1000,      # Number of bootstrap replications
        significance_level=0.05, # Significance level
        bootstrap_method='stationary',  # 'block' or 'stationary'
        test_statistic='t',      # 't' or 'range'
        seed=42                  # Random seed for reproducibility
    )
    
    # Run the MCS procedure with additional options
    mcs_result = mcs.run(
        losses,                  # Loss matrix (time x models)
        model_names=None,        # Optional model names
        rng=None                 # Custom random number generator
    )

Visualizing MCS Results
--------------------

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from mfe.models.bootstrap import ModelConfidenceSet
    
    # Generate loss data for 10 different models
    np.random.seed(42)
    n_models = 10
    n_periods = 200
    
    # First 3 models are the best, others progressively worse
    base_losses = np.random.normal(0, 1, (n_periods, n_models))
    for i in range(3, n_models):
        base_losses[:, i] += 0.2 * (i - 2)  # Increasing loss
    
    # Create model names
    model_names = [f"Model {i+1}" for i in range(n_models)]
    
    # Run MCS
    mcs = ModelConfidenceSet(block_size=10, num_bootstrap=1000)
    mcs_result = mcs.run(base_losses, model_names=model_names)
    
    # Plot p-values
    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_names, mcs_result.pvalues)
    
    # Color bars based on inclusion in MCS
    for i, model_idx in enumerate(range(n_models)):
        if model_idx in mcs_result.included_models:
            bars[i].set_color('green')
        else:
            bars[i].set_color('red')
    
    plt.axhline(mcs.significance_level, color='black', linestyle='--', 
                label=f'Significance Level ({mcs.significance_level})')
    plt.title('Model Confidence Set p-values')
    plt.ylabel('p-value')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

Bootstrap Reality Check and SPA Test
=================================

The Bootstrap Reality Check (BRC) and Superior Predictive Ability (SPA) tests evaluate whether any model in a set outperforms a benchmark model.

Basic Usage
---------

.. code-block:: python

    import numpy as np
    from mfe.models.bootstrap import BSDS
    
    # Generate loss data for benchmark and 5 competing models
    np.random.seed(42)
    n_periods = 100
    n_models = 5
    
    # Benchmark model losses
    benchmark_losses = np.random.normal(0, 1, n_periods)
    
    # Competing models' losses (model 0 is better, others are not)
    model_losses = np.random.normal(0, 1, (n_periods, n_models))
    model_losses[:, 0] -= 0.3  # Model 0 has lower loss
    
    # Create a BSDS test
    bsds = BSDS(
        block_size=10,           # Block size for bootstrap
        num_bootstrap=1000,      # Number of bootstrap replications
        seed=42                  # Random seed for reproducibility
    )
    
    # Run the test
    bsds_result = bsds.run(benchmark_losses, model_losses)
    
    # Print results
    print("BSDS Test Results:")
    print(f"Reality Check p-value: {bsds_result.rc_pvalue:.4f}")
    print(f"SPA p-value: {bsds_result.spa_pvalue:.4f}")
    print("\nIndividual model p-values:")
    for i, p_val in enumerate(bsds_result.model_pvalues):
        print(f"Model {i}: {p_val:.4f}")

Advanced Configuration
-------------------

The ``BSDS`` class provides several configuration options:

.. code-block:: python

    from mfe.models.bootstrap import BSDS
    
    # Create a BSDS test with custom configuration
    bsds = BSDS(
        block_size=10,           # Block size for bootstrap
        num_bootstrap=1000,      # Number of bootstrap replications
        bootstrap_method='stationary',  # 'block' or 'stationary'
        seed=42                  # Random seed for reproducibility
    )
    
    # Run the test with additional options
    bsds_result = bsds.run(
        benchmark_losses,        # Benchmark model losses
        model_losses,            # Competing models' losses
        model_names=None,        # Optional model names
        rng=None                 # Custom random number generator
    )

Visualizing BSDS Results
---------------------

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from mfe.models.bootstrap import BSDS
    
    # Generate loss data
    np.random.seed(42)
    n_periods = 200
    n_models = 8
    
    # Benchmark model losses
    benchmark_losses = np.random.normal(0, 1, n_periods)
    
    # Competing models' losses (first 2 are better, others are not)
    model_losses = np.random.normal(0, 1, (n_periods, n_models))
    model_losses[:, 0] -= 0.4  # Model 0 has lower loss
    model_losses[:, 1] -= 0.3  # Model 1 has lower loss
    
    # Create model names
    model_names = [f"Model {i+1}" for i in range(n_models)]
    
    # Run BSDS test
    bsds = BSDS(block_size=10, num_bootstrap=1000)
    bsds_result = bsds.run(benchmark_losses, model_losses, model_names=model_names)
    
    # Plot loss differences
    loss_diffs = np.mean(benchmark_losses - model_losses, axis=0)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_names, loss_diffs)
    
    # Color bars based on significance
    for i, p_val in enumerate(bsds_result.model_pvalues):
        if p_val < 0.05:
            bars[i].set_color('green')
        else:
            bars[i].set_color('red')
    
    plt.axhline(0, color='black', linestyle='--', label='Benchmark')
    plt.title('Average Loss Difference vs Benchmark')
    plt.ylabel('Benchmark Loss - Model Loss')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Print test results
    print(f"Reality Check p-value: {bsds_result.rc_pvalue:.4f}")
    print(f"SPA p-value: {bsds_result.spa_pvalue:.4f}")

Custom Bootstrap Functions
=======================

The MFE Toolbox allows you to create custom bootstrap functions for specialized applications:

.. code-block:: python

    import numpy as np
    from mfe.models.bootstrap import BlockBootstrap
    
    def bootstrap_sharpe_ratio(returns, block_size=20, num_samples=1000):
        """
        Compute bootstrap confidence interval for Sharpe ratio.
        
        Args:
            returns: Array of return data
            block_size: Size of bootstrap blocks
            num_samples: Number of bootstrap samples
            
        Returns:
            Tuple containing (sharpe_ratio, lower_bound, upper_bound)
        """
        # Calculate sample Sharpe ratio
        sample_sharpe = returns.mean() / returns.std()
        
        # Create bootstrap
        bootstrap = BlockBootstrap(block_size=block_size)
        bootstrap_samples = bootstrap.generate(returns, num_samples=num_samples)
        
        # Compute Sharpe ratio for each bootstrap sample
        bootstrap_sharpes = np.array([
            sample.mean() / sample.std() for sample in bootstrap_samples
        ])
        
        # Compute confidence interval
        conf_interval = np.percentile(bootstrap_sharpes, [2.5, 97.5])
        
        return sample_sharpe, conf_interval[0], conf_interval[1]
    
    # Example usage
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.01, 1000)  # Daily returns
    
    sharpe, lower, upper = bootstrap_sharpe_ratio(returns)
    print(f"Sharpe Ratio: {sharpe:.4f}")
    print(f"95% Confidence Interval: [{lower:.4f}, {upper:.4f}]")

Performance Considerations
=======================

The bootstrap methods in the MFE Toolbox are optimized for performance using Numba's just-in-time compilation. Here are some considerations for optimal performance:

1. **Block Size Selection**: Larger block sizes preserve more of the dependence structure but reduce the effective number of independent blocks.

2. **Number of Bootstrap Samples**: More samples provide better approximations but increase computation time. For most applications, 1,000-10,000 samples are sufficient.

3. **Asynchronous Processing**: For large datasets or many bootstrap samples, use the asynchronous methods (`generate_async`) to maintain responsiveness.

4. **Memory Usage**: Bootstrap samples can consume significant memory. For very large datasets, consider processing bootstrap statistics incrementally rather than storing all samples.

5. **Numba Acceleration**: The first call to a Numba-accelerated function includes compilation time. Subsequent calls will be much faster.

Example comparing performance with and without Numba acceleration:

.. code-block:: python

    import time
    import numpy as np
    from mfe.models.bootstrap import BlockBootstrap
    from mfe.models.bootstrap._numba_core import _generate_block_bootstrap_indices
    
    # Generate test data
    np.random.seed(42)
    data = np.random.normal(0, 1, 5000)
    
    # Define a Python implementation without Numba
    def generate_indices_python(n, b, k):
        """Generate bootstrap indices without Numba acceleration."""
        indices = np.zeros(n, dtype=np.int64)
        for i in range(0, n, b):
            block_start = np.random.randint(0, n - b + 1)
            indices[i:min(i+b, n)] = np.arange(block_start, block_start + min(b, n-i))
        return indices
    
    # Compare performance
    n = len(data)
    block_size = 50
    num_samples = 100
    
    # Time Numba-accelerated version
    start_time = time.time()
    for _ in range(num_samples):
        _ = _generate_block_bootstrap_indices(n, block_size, np.random.randint(0, 2**31-1))
    numba_time = time.time() - start_time
    
    # Time Python version
    start_time = time.time()
    for _ in range(num_samples):
        _ = generate_indices_python(n, block_size, None)
    python_time = time.time() - start_time
    
    print(f"Numba-accelerated version: {numba_time:.4f} seconds")
    print(f"Python version: {python_time:.4f} seconds")
    print(f"Speedup factor: {python_time / numba_time:.1f}x")

Conclusion
=========

The bootstrap methods in the MFE Toolbox provide powerful tools for statistical inference with dependent data. By leveraging NumPy's efficient array operations and Numba's performance acceleration, these methods enable robust analysis of financial time series data.

For more information on specific bootstrap applications, see the following resources:

- :doc:`univariate_volatility_models` - Bootstrap confidence intervals for GARCH parameters
- :doc:`multivariate_volatility_models` - Bootstrap tests for correlation dynamics
- :doc:`time_series_analysis` - Bootstrap prediction intervals for ARMA forecasts
- :doc:`high_frequency_econometrics` - Bootstrap inference for realized volatility estimators

References
=========

- Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. Journal of the American Statistical Association, 89(428), 1303-1313.
- Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. Annals of Statistics, 17(3), 1217-1241.
- Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence set. Econometrica, 79(2), 453-497.
- White, H. (2000). A reality check for data snooping. Econometrica, 68(5), 1097-1126.
- Hansen, P. R. (2005). A test for superior predictive ability. Journal of Business & Economic Statistics, 23(4), 365-380.