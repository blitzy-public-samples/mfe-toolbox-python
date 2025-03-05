# docs/user_guide/high_frequency_econometrics.rst
```rst
============================
High-Frequency Econometrics
============================

This guide provides a comprehensive overview of the high-frequency financial econometrics tools available in the MFE Toolbox. These tools enable the estimation of volatility and covariance from intraday price data, accounting for market microstructure noise and other challenges specific to high-frequency financial data.

Introduction to High-Frequency Econometrics
==========================================

High-frequency financial econometrics focuses on analyzing intraday price data to estimate volatility, covariance, and other financial metrics. Unlike traditional daily return-based methods, high-frequency approaches use tick-by-tick or regularly sampled intraday data to provide more precise estimates of volatility.

Key advantages of high-frequency methods include:

- More precise volatility estimation using intraday information
- Ability to separate continuous volatility from jumps
- Robust estimation in the presence of market microstructure noise
- Intraday patterns and dynamics analysis
- Real-time risk monitoring capabilities

The MFE Toolbox provides a comprehensive suite of realized volatility estimators and related tools, implemented in Python with Numba acceleration for optimal performance.

Data Preparation and Time Conversion
==================================

Working with High-Frequency Data
-------------------------------

High-frequency data typically consists of irregularly spaced observations of prices and corresponding timestamps. The MFE Toolbox works seamlessly with Pandas DataFrames for handling such data:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mfe.models.realized import RealizedVariance, price_filter

    # Load high-frequency data
    # Example with a CSV file containing timestamp and price columns
    hf_data = pd.read_csv('hf_data.csv')
    
    # Convert string timestamps to Pandas datetime objects
    hf_data['timestamp'] = pd.to_datetime(hf_data['timestamp'])
    
    # Set timestamp as index
    hf_data = hf_data.set_index('timestamp')
    
    # Display the first few rows
    print(hf_data.head())
    
    # Basic statistics
    print(hf_data.describe())
    
    # Plot the price series
    plt.figure(figsize=(12, 6))
    plt.plot(hf_data.index, hf_data['price'])
    plt.title('High-Frequency Price Data')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

Time Conversion Utilities
-----------------------

The MFE Toolbox provides utilities for converting between different time formats, leveraging Pandas' powerful datetime functionality:

.. code-block:: python

    import pandas as pd
    import numpy as np
    from mfe.models.realized import seconds2unit, unit2seconds, wall2unit, unit2wall

    # Create a trading day timeline
    trading_day = pd.date_range(
        start='2023-01-01 09:30:00',  # Market open
        end='2023-01-01 16:00:00',    # Market close
        freq='1min'                    # 1-minute intervals
    )
    
    # Convert wall clock time to unit time (normalized between 0 and 1)
    # Unit time represents the fraction of the trading day
    unit_times = np.array([wall2unit(t.time()) for t in trading_day])
    
    # Convert unit time back to wall clock time
    wall_times = np.array([unit2wall(u) for u in unit_times])
    
    # Convert unit time to seconds since midnight
    seconds = np.array([unit2seconds(u) for u in unit_times])
    
    # Convert seconds back to unit time
    unit_times_check = np.array([seconds2unit(s) for s in seconds])
    
    # Display conversions
    for i in range(5):  # Show first 5 conversions
        print(f"Wall time: {trading_day[i].time()}, "
              f"Unit time: {unit_times[i]:.6f}, "
              f"Seconds: {seconds[i]}, "
              f"Converted back to unit: {unit_times_check[i]:.6f}")

Working with Irregularly Spaced Data
----------------------------------

High-frequency data is often irregularly spaced. Pandas provides powerful tools for handling such data:

.. code-block:: python

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Create irregularly spaced timestamps
    np.random.seed(42)
    n_obs = 1000
    
    # Generate random intervals between observations (in seconds)
    intervals = np.random.exponential(scale=30, size=n_obs)
    
    # Create cumulative timestamps starting from market open
    market_open = pd.Timestamp('2023-01-01 09:30:00')
    timestamps = [market_open + pd.Timedelta(seconds=int(np.sum(intervals[:i]))) 
                 for i in range(n_obs)]
    
    # Generate random price path
    log_prices = np.cumsum(np.random.normal(0, 0.001, n_obs))
    prices = 100 * np.exp(log_prices)
    
    # Create DataFrame with irregular timestamps
    irregular_data = pd.DataFrame({
        'price': prices
    }, index=timestamps)
    
    # Resample to regular intervals (e.g., 5-minute bars)
    regular_data = irregular_data.resample('5min').last()
    
    # Fill missing values using forward fill
    regular_data = regular_data.fillna(method='ffill')
    
    # Plot both irregular and regular data
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(irregular_data.index, irregular_data['price'], 'o-', markersize=2)
    plt.title('Irregularly Spaced High-Frequency Data')
    plt.ylabel('Price')
    
    plt.subplot(2, 1, 2)
    plt.plot(regular_data.index, regular_data['price'], 'o-', markersize=3)
    plt.title('Regularly Sampled Data (5-minute intervals)')
    plt.ylabel('Price')
    
    plt.tight_layout()
    plt.show()

Price Filtering and Cleaning
--------------------------

High-frequency data often contains errors and outliers. The MFE Toolbox provides functions for filtering and cleaning price data:

.. code-block:: python

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from mfe.models.realized import price_filter
    
    # Generate sample data with outliers
    np.random.seed(42)
    n_obs = 1000
    
    # Create timestamps
    timestamps = pd.date_range(
        start='2023-01-01 09:30:00',
        periods=n_obs,
        freq='30s'
    )
    
    # Generate random walk with occasional outliers
    log_prices = np.cumsum(np.random.normal(0, 0.001, n_obs))
    
    # Add outliers (approximately 1% of observations)
    outlier_idx = np.random.choice(n_obs, size=int(n_obs * 0.01), replace=False)
    for idx in outlier_idx:
        log_prices[idx] += np.random.choice([-1, 1]) * np.random.uniform(0.05, 0.1)
    
    prices = 100 * np.exp(log_prices)
    
    # Create DataFrame
    data = pd.DataFrame({
        'price': prices
    }, index=timestamps)
    
    # Apply price filter
    filtered_prices = price_filter(
        prices=data['price'].values,
        timestamps=data.index.values,
        k=3.0  # Filter threshold (3 standard deviations)
    )
    
    # Create DataFrame with filtered prices
    filtered_data = pd.DataFrame({
        'price': filtered_prices
    }, index=timestamps)
    
    # Plot original and filtered prices
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data['price'])
    plt.title('Original High-Frequency Price Data with Outliers')
    plt.ylabel('Price')
    
    plt.subplot(2, 1, 2)
    plt.plot(filtered_data.index, filtered_data['price'])
    plt.title('Filtered High-Frequency Price Data')
    plt.ylabel('Price')
    
    plt.tight_layout()
    plt.show()
    
    # Identify outliers
    outliers = data[data['price'] != filtered_data['price']]
    print(f"Number of outliers detected: {len(outliers)}")
    print(f"Percentage of outliers: {len(outliers) / len(data) * 100:.2f}%")

Realized Volatility Estimators
============================

Basic Realized Variance
---------------------

The simplest realized volatility estimator is the realized variance, which is the sum of squared intraday returns:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mfe.models.realized import RealizedVariance
    
    # Generate simulated high-frequency data
    np.random.seed(42)
    n_days = 5
    n_intraday = 100  # 100 observations per day
    
    # Create timestamps (5 days, 100 observations per day)
    timestamps = []
    for day in range(n_days):
        day_date = pd.Timestamp(f'2023-01-{day+1:02d}')
        for i in range(n_intraday):
            # 9:30 AM to 4:00 PM (390 minutes = 6.5 hours)
            minute = 9*60 + 30 + i * (6.5*60 / n_intraday)
            hour = int(minute // 60)
            minute = int(minute % 60)
            timestamps.append(day_date + pd.Timedelta(hours=hour, minutes=minute))
    
    # Generate random walk for prices
    # Higher volatility on days 2 and 4
    volatility = np.ones(n_days * n_intraday) * 0.001
    volatility[n_intraday:2*n_intraday] *= 2  # Day 2
    volatility[3*n_intraday:4*n_intraday] *= 3  # Day 4
    
    returns = np.random.normal(0, volatility)
    log_prices = np.cumsum(returns)
    prices = np.exp(log_prices)
    
    # Create DataFrame
    hf_data = pd.DataFrame({
        'price': prices
    }, index=timestamps)
    
    # Create a realized variance estimator
    rv_estimator = RealizedVariance()
    
    # Estimate daily realized variance
    # Using 5-minute sampling
    rv = rv_estimator.compute(
        prices=hf_data['price'].values,
        timestamps=hf_data.index.values,
        sampling='5min'  # 5-minute sampling
    )
    
    # Convert to annualized volatility (standard deviation)
    # Assuming 252 trading days per year
    annualized_vol = np.sqrt(rv * 252)
    
    # Print results
    print("Daily Realized Variance and Annualized Volatility:")
    for day in range(n_days):
        print(f"Day {day+1}: RV = {rv[day]:.6f}, Annualized Vol = {annualized_vol[day]:.2f}%")
    
    # Plot realized volatility
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, n_days+1), annualized_vol)
    plt.title('Daily Realized Volatility (Annualized)')
    plt.xlabel('Day')
    plt.ylabel('Volatility (%)')
    plt.xticks(range(1, n_days+1))
    plt.show()

Bipower Variation
---------------

Bipower variation is robust to jumps in the price process:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mfe.models.realized import RealizedVariance, BipowerVariation
    
    # Generate simulated high-frequency data with jumps
    np.random.seed(42)
    n_days = 5
    n_intraday = 100  # 100 observations per day
    
    # Create timestamps
    timestamps = []
    for day in range(n_days):
        day_date = pd.Timestamp(f'2023-01-{day+1:02d}')
        for i in range(n_intraday):
            minute = 9*60 + 30 + i * (6.5*60 / n_intraday)
            hour = int(minute // 60)
            minute = int(minute % 60)
            timestamps.append(day_date + pd.Timedelta(hours=hour, minutes=minute))
    
    # Generate random walk with occasional jumps
    volatility = np.ones(n_days * n_intraday) * 0.001
    returns = np.random.normal(0, volatility)
    
    # Add jumps (one per day)
    for day in range(n_days):
        jump_idx = day * n_intraday + np.random.randint(0, n_intraday)
        returns[jump_idx] += np.random.choice([-1, 1]) * np.random.uniform(0.01, 0.02)
    
    log_prices = np.cumsum(returns)
    prices = np.exp(log_prices)
    
    # Create DataFrame
    hf_data = pd.DataFrame({
        'price': prices
    }, index=timestamps)
    
    # Create estimators
    rv_estimator = RealizedVariance()
    bv_estimator = BipowerVariation()
    
    # Estimate daily realized variance and bipower variation
    rv = rv_estimator.compute(
        prices=hf_data['price'].values,
        timestamps=hf_data.index.values,
        sampling='5min'
    )
    
    bv = bv_estimator.compute(
        prices=hf_data['price'].values,
        timestamps=hf_data.index.values,
        sampling='5min'
    )
    
    # Estimate jump component
    jump = np.maximum(0, rv - bv)
    
    # Convert to annualized volatility
    annualized_vol_rv = np.sqrt(rv * 252)
    annualized_vol_bv = np.sqrt(bv * 252)
    
    # Print results
    print("Comparison of Realized Variance and Bipower Variation:")
    for day in range(n_days):
        print(f"Day {day+1}:")
        print(f"  RV = {rv[day]:.6f}, Annualized Vol (RV) = {annualized_vol_rv[day]:.2f}%")
        print(f"  BV = {bv[day]:.6f}, Annualized Vol (BV) = {annualized_vol_bv[day]:.2f}%")
        print(f"  Jump Component = {jump[day]:.6f}")
        print(f"  Jump Ratio = {jump[day]/rv[day]*100:.2f}%")
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    x = np.arange(1, n_days+1)
    width = 0.35
    
    plt.bar(x - width/2, annualized_vol_rv, width, label='RV')
    plt.bar(x + width/2, annualized_vol_bv, width, label='BV')
    
    plt.title('Comparison of Realized Volatility Estimators')
    plt.xlabel('Day')
    plt.ylabel('Annualized Volatility (%)')
    plt.xticks(x)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Plot jump component
    plt.figure(figsize=(10, 6))
    plt.bar(x, jump/rv*100)
    plt.title('Jump Component as Percentage of Realized Variance')
    plt.xlabel('Day')
    plt.ylabel('Jump Component (%)')
    plt.xticks(x)
    plt.tight_layout()
    plt.show()

Realized Kernel Estimator
-----------------------

Realized kernel estimators are robust to market microstructure noise:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mfe.models.realized import RealizedVariance, RealizedKernel
    
    # Generate simulated high-frequency data with microstructure noise
    np.random.seed(42)
    n_days = 5
    n_intraday = 200  # 200 observations per day
    
    # Create timestamps
    timestamps = []
    for day in range(n_days):
        day_date = pd.Timestamp(f'2023-01-{day+1:02d}')
        for i in range(n_intraday):
            minute = 9*60 + 30 + i * (6.5*60 / n_intraday)
            hour = int(minute // 60)
            minute = int(minute % 60)
            timestamps.append(day_date + pd.Timedelta(hours=hour, minutes=minute))
    
    # Generate efficient price process
    volatility = np.ones(n_days * n_intraday) * 0.001
    efficient_returns = np.random.normal(0, volatility)
    efficient_log_prices = np.cumsum(efficient_returns)
    
    # Add microstructure noise
    noise_std = 0.0005  # Noise standard deviation
    noise = np.random.normal(0, noise_std, n_days * n_intraday)
    observed_log_prices = efficient_log_prices + noise
    
    # Convert to prices
    efficient_prices = np.exp(efficient_log_prices)
    observed_prices = np.exp(observed_log_prices)
    
    # Create DataFrames
    efficient_data = pd.DataFrame({
        'price': efficient_prices
    }, index=timestamps)
    
    observed_data = pd.DataFrame({
        'price': observed_prices
    }, index=timestamps)
    
    # Create estimators
    rv_estimator = RealizedVariance()
    rk_estimator = RealizedKernel(kernel_type='parzen')
    
    # Estimate daily realized variance and realized kernel
    rv = rv_estimator.compute(
        prices=observed_data['price'].values,
        timestamps=observed_data.index.values,
        sampling='5min'
    )
    
    rk = rk_estimator.compute(
        prices=observed_data['price'].values,
        timestamps=observed_data.index.values
    )
    
    # Compute true integrated variance (for comparison)
    true_iv = np.zeros(n_days)
    for day in range(n_days):
        day_returns = efficient_returns[day*n_intraday:(day+1)*n_intraday]
        true_iv[day] = np.sum(day_returns**2)
    
    # Convert to annualized volatility
    annualized_vol_rv = np.sqrt(rv * 252)
    annualized_vol_rk = np.sqrt(rk * 252)
    annualized_vol_true = np.sqrt(true_iv * 252)
    
    # Print results
    print("Comparison of Realized Variance and Realized Kernel:")
    for day in range(n_days):
        print(f"Day {day+1}:")
        print(f"  True IV = {true_iv[day]:.6f}, Annualized Vol (True) = {annualized_vol_true[day]:.2f}%")
        print(f"  RV = {rv[day]:.6f}, Annualized Vol (RV) = {annualized_vol_rv[day]:.2f}%")
        print(f"  RK = {rk[day]:.6f}, Annualized Vol (RK) = {annualized_vol_rk[day]:.2f}%")
        print(f"  RV Bias = {(rv[day]/true_iv[day]-1)*100:.2f}%")
        print(f"  RK Bias = {(rk[day]/true_iv[day]-1)*100:.2f}%")
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    x = np.arange(1, n_days+1)
    width = 0.25
    
    plt.bar(x - width, annualized_vol_true, width, label='True')
    plt.bar(x, annualized_vol_rv, width, label='RV')
    plt.bar(x + width, annualized_vol_rk, width, label='RK')
    
    plt.title('Comparison of Volatility Estimators with Microstructure Noise')
    plt.xlabel('Day')
    plt.ylabel('Annualized Volatility (%)')
    plt.xticks(x)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

Multiscale Realized Variance
--------------------------

Multiscale realized variance combines estimates at different sampling frequencies:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mfe.models.realized import RealizedVariance, MultiscaleVariance
    
    # Generate simulated high-frequency data with microstructure noise
    np.random.seed(42)
    n_days = 5
    n_intraday = 200  # 200 observations per day
    
    # Create timestamps
    timestamps = []
    for day in range(n_days):
        day_date = pd.Timestamp(f'2023-01-{day+1:02d}')
        for i in range(n_intraday):
            minute = 9*60 + 30 + i * (6.5*60 / n_intraday)
            hour = int(minute // 60)
            minute = int(minute % 60)
            timestamps.append(day_date + pd.Timedelta(hours=hour, minutes=minute))
    
    # Generate efficient price process
    volatility = np.ones(n_days * n_intraday) * 0.001
    efficient_returns = np.random.normal(0, volatility)
    efficient_log_prices = np.cumsum(efficient_returns)
    
    # Add microstructure noise
    noise_std = 0.0005  # Noise standard deviation
    noise = np.random.normal(0, noise_std, n_days * n_intraday)
    observed_log_prices = efficient_log_prices + noise
    
    # Convert to prices
    observed_prices = np.exp(observed_log_prices)
    
    # Create DataFrame
    observed_data = pd.DataFrame({
        'price': observed_prices
    }, index=timestamps)
    
    # Create estimators
    rv_estimator = RealizedVariance()
    msrv_estimator = MultiscaleVariance()
    
    # Estimate daily realized variance at different sampling frequencies
    rv_1min = rv_estimator.compute(
        prices=observed_data['price'].values,
        timestamps=observed_data.index.values,
        sampling='1min'
    )
    
    rv_5min = rv_estimator.compute(
        prices=observed_data['price'].values,
        timestamps=observed_data.index.values,
        sampling='5min'
    )
    
    rv_10min = rv_estimator.compute(
        prices=observed_data['price'].values,
        timestamps=observed_data.index.values,
        sampling='10min'
    )
    
    # Estimate multiscale realized variance
    msrv = msrv_estimator.compute(
        prices=observed_data['price'].values,
        timestamps=observed_data.index.values
    )
    
    # Convert to annualized volatility
    annualized_vol_1min = np.sqrt(rv_1min * 252)
    annualized_vol_5min = np.sqrt(rv_5min * 252)
    annualized_vol_10min = np.sqrt(rv_10min * 252)
    annualized_vol_msrv = np.sqrt(msrv * 252)
    
    # Print results
    print("Comparison of RV at Different Sampling Frequencies and MSRV:")
    for day in range(n_days):
        print(f"Day {day+1}:")
        print(f"  RV (1min) = {rv_1min[day]:.6f}, Annualized Vol = {annualized_vol_1min[day]:.2f}%")
        print(f"  RV (5min) = {rv_5min[day]:.6f}, Annualized Vol = {annualized_vol_5min[day]:.2f}%")
        print(f"  RV (10min) = {rv_10min[day]:.6f}, Annualized Vol = {annualized_vol_10min[day]:.2f}%")
        print(f"  MSRV = {msrv[day]:.6f}, Annualized Vol = {annualized_vol_msrv[day]:.2f}%")
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    x = np.arange(1, n_days+1)
    width = 0.2
    
    plt.bar(x - 1.5*width, annualized_vol_1min, width, label='RV (1min)')
    plt.bar(x - 0.5*width, annualized_vol_5min, width, label='RV (5min)')
    plt.bar(x + 0.5*width, annualized_vol_10min, width, label='RV (10min)')
    plt.bar(x + 1.5*width, annualized_vol_msrv, width, label='MSRV')
    
    plt.title('Comparison of Volatility Estimators at Different Sampling Frequencies')
    plt.xlabel('Day')
    plt.ylabel('Annualized Volatility (%)')
    plt.xticks(x)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

Realized Semivariance
-------------------

Realized semivariance separates upside and downside risk:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mfe.models.realized import RealizedVariance, RealizedSemivariance
    
    # Generate simulated high-frequency data
    np.random.seed(42)
    n_days = 5
    n_intraday = 100  # 100 observations per day
    
    # Create timestamps
    timestamps = []
    for day in range(n_days):
        day_date = pd.Timestamp(f'2023-01-{day+1:02d}')
        for i in range(n_intraday):
            minute = 9*60 + 30 + i * (6.5*60 / n_intraday)
            hour = int(minute // 60)
            minute = int(minute % 60)
            timestamps.append(day_date + pd.Timedelta(hours=hour, minutes=minute))
    
    # Generate random walk with asymmetric returns
    # Days 1, 3, 5: More negative jumps
    # Days 2, 4: More positive jumps
    volatility = np.ones(n_days * n_intraday) * 0.001
    returns = np.random.normal(0, volatility)
    
    # Add asymmetric jumps
    for day in range(n_days):
        n_jumps = 3  # Number of jumps per day
        jump_idx = day * n_intraday + np.random.choice(n_intraday, size=n_jumps, replace=False)
        
        if day % 2 == 0:  # Days 1, 3, 5: More negative jumps
            jump_sign = np.array([-1, -1, 1])
        else:  # Days 2, 4: More positive jumps
            jump_sign = np.array([1, 1, -1])
            
        for i, idx in enumerate(jump_idx):
            returns[idx] += jump_sign[i] * np.random.uniform(0.005, 0.01)
    
    log_prices = np.cumsum(returns)
    prices = np.exp(log_prices)
    
    # Create DataFrame
    hf_data = pd.DataFrame({
        'price': prices
    }, index=timestamps)
    
    # Create estimators
    rv_estimator = RealizedVariance()
    rsv_estimator = RealizedSemivariance()
    
    # Estimate daily realized variance
    rv = rv_estimator.compute(
        prices=hf_data['price'].values,
        timestamps=hf_data.index.values,
        sampling='5min'
    )
    
    # Estimate daily realized semivariance (positive and negative)
    rsv_pos = rsv_estimator.compute(
        prices=hf_data['price'].values,
        timestamps=hf_data.index.values,
        sampling='5min',
        type='positive'
    )
    
    rsv_neg = rsv_estimator.compute(
        prices=hf_data['price'].values,
        timestamps=hf_data.index.values,
        sampling='5min',
        type='negative'
    )
    
    # Convert to annualized volatility
    annualized_vol_rv = np.sqrt(rv * 252)
    annualized_vol_pos = np.sqrt(rsv_pos * 252)
    annualized_vol_neg = np.sqrt(rsv_neg * 252)
    
    # Print results
    print("Realized Variance and Semivariance:")
    for day in range(n_days):
        print(f"Day {day+1}:")
        print(f"  RV = {rv[day]:.6f}, Annualized Vol = {annualized_vol_rv[day]:.2f}%")
        print(f"  RSV+ = {rsv_pos[day]:.6f}, Annualized Vol+ = {annualized_vol_pos[day]:.2f}%")
        print(f"  RSV- = {rsv_neg[day]:.6f}, Annualized Vol- = {annualized_vol_neg[day]:.2f}%")
        print(f"  Asymmetry Ratio = {rsv_neg[day]/rsv_pos[day]:.2f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    x = np.arange(1, n_days+1)
    width = 0.3
    
    plt.bar(x - width, annualized_vol_pos, width, label='Upside Vol')
    plt.bar(x, annualized_vol_rv, width, label='Total Vol')
    plt.bar(x + width, annualized_vol_neg, width, label='Downside Vol')
    
    plt.title('Comparison of Realized Volatility Components')
    plt.xlabel('Day')
    plt.ylabel('Annualized Volatility (%)')
    plt.xticks(x)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Plot asymmetry ratio
    plt.figure(figsize=(10, 6))
    plt.bar(x, rsv_neg/rsv_pos)
    plt.axhline(y=1, color='r', linestyle='--', label='Symmetric')
    plt.title('Downside/Upside Volatility Ratio')
    plt.xlabel('Day')
    plt.ylabel('Ratio')
    plt.xticks(x)
    plt.legend()
    plt.tight_layout()
    plt.show()

Handling Microstructure Noise
===========================

Optimal Sampling Frequency
------------------------

Finding the optimal sampling frequency to balance bias and variance:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mfe.models.realized import RealizedVariance, variance_optimal_sampling
    
    # Generate simulated high-frequency data with microstructure noise
    np.random.seed(42)
    n_days = 1  # Focus on a single day
    n_intraday = 1000  # 1000 observations per day
    
    # Create timestamps
    timestamps = []
    for day in range(n_days):
        day_date = pd.Timestamp(f'2023-01-{day+1:02d}')
        for i in range(n_intraday):
            minute = 9*60 + 30 + i * (6.5*60 / n_intraday)
            hour = int(minute // 60)
            minute = int(minute % 60)
            timestamps.append(day_date + pd.Timedelta(hours=hour, minutes=minute))
    
    # Generate efficient price process
    volatility = 0.001
    efficient_returns = np.random.normal(0, volatility, n_intraday)
    efficient_log_prices = np.cumsum(efficient_returns)
    
    # Add microstructure noise
    noise_std = 0.0005  # Noise standard deviation
    noise = np.random.normal(0, noise_std, n_intraday)
    observed_log_prices = efficient_log_prices + noise
    
    # Convert to prices
    observed_prices = np.exp(observed_log_prices)
    
    # Create DataFrame
    observed_data = pd.DataFrame({
        'price': observed_prices
    }, index=timestamps)
    
    # Create estimator
    rv_estimator = RealizedVariance()
    
    # Compute true integrated variance
    true_iv = np.sum(efficient_returns**2)
    
    # Estimate realized variance at different sampling frequencies
    sampling_frequencies = [1, 2, 3, 5, 10, 15, 20, 30, 60]  # in minutes
    rv_estimates = []
    
    for freq in sampling_frequencies:
        rv = rv_estimator.compute(
            prices=observed_data['price'].values,
            timestamps=observed_data.index.values,
            sampling=f'{freq}min'
        )[0]  # Single day
        rv_estimates.append(rv)
    
    # Find optimal sampling frequency
    optimal_freq, optimal_rv = variance_optimal_sampling(
        prices=observed_data['price'].values,
        timestamps=observed_data.index.values,
        noise_estimate=None  # Automatically estimate noise
    )
    
    # Print results
    print(f"True Integrated Variance: {true_iv:.6f}")
    print(f"Optimal Sampling Frequency: {optimal_freq:.2f} minutes")
    print(f"Realized Variance at Optimal Frequency: {optimal_rv:.6f}")
    print(f"Bias: {(optimal_rv/true_iv-1)*100:.2f}%")
    
    # Print RV at different frequencies
    print("\nRealized Variance at Different Sampling Frequencies:")
    for i, freq in enumerate(sampling_frequencies):
        bias = (rv_estimates[i]/true_iv-1)*100
        print(f"  {freq} min: RV = {rv_estimates[i]:.6f}, Bias = {bias:.2f}%")
    
    # Plot RV vs. sampling frequency
    plt.figure(figsize=(10, 6))
    plt.plot(sampling_frequencies, rv_estimates, 'o-', label='RV Estimates')
    plt.axhline(y=true_iv, color='r', linestyle='--', label='True IV')
    plt.axvline(x=optimal_freq, color='g', linestyle='--', label=f'Optimal ({optimal_freq:.2f} min)')
    plt.title('Realized Variance vs. Sampling Frequency')
    plt.xlabel('Sampling Frequency (minutes)')
    plt.ylabel('Realized Variance')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

Noise-Robust Estimators
---------------------

Comparing different noise-robust estimators:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mfe.models.realized import (
        RealizedVariance, RealizedKernel, MultiscaleVariance,
        TwoScaleVariance, QMLEVariance
    )
    
    # Generate simulated high-frequency data with microstructure noise
    np.random.seed(42)
    n_days = 5
    n_intraday = 200  # 200 observations per day
    
    # Create timestamps
    timestamps = []
    for day in range(n_days):
        day_date = pd.Timestamp(f'2023-01-{day+1:02d}')
        for i in range(n_intraday):
            minute = 9*60 + 30 + i * (6.5*60 / n_intraday)
            hour = int(minute // 60)
            minute = int(minute % 60)
            timestamps.append(day_date + pd.Timedelta(hours=hour, minutes=minute))
    
    # Generate efficient price process
    volatility = np.ones(n_days * n_intraday) * 0.001
    efficient_returns = np.random.normal(0, volatility)
    efficient_log_prices = np.cumsum(efficient_returns)
    
    # Add microstructure noise
    noise_std = 0.0005  # Noise standard deviation
    noise = np.random.normal(0, noise_std, n_days * n_intraday)
    observed_log_prices = efficient_log_prices + noise
    
    # Convert to prices
    efficient_prices = np.exp(efficient_log_prices)
    observed_prices = np.exp(observed_log_prices)
    
    # Create DataFrames
    efficient_data = pd.DataFrame({
        'price': efficient_prices
    }, index=timestamps)
    
    observed_data = pd.DataFrame({
        'price': observed_prices
    }, index=timestamps)
    
    # Create estimators
    rv_estimator = RealizedVariance()
    rk_estimator = RealizedKernel(kernel_type='parzen')
    msrv_estimator = MultiscaleVariance()
    tsrv_estimator = TwoScaleVariance()
    qmle_estimator = QMLEVariance()
    
    # Compute true integrated variance
    true_iv = np.zeros(n_days)
    for day in range(n_days):
        day_returns = efficient_returns[day*n_intraday:(day+1)*n_intraday]
        true_iv[day] = np.sum(day_returns**2)
    
    # Estimate volatility using different estimators
    rv_5min = rv_estimator.compute(
        prices=observed_data['price'].values,
        timestamps=observed_data.index.values,
        sampling='5min'
    )
    
    rk = rk_estimator.compute(
        prices=observed_data['price'].values,
        timestamps=observed_data.index.values
    )
    
    msrv = msrv_estimator.compute(
        prices=observed_data['price'].values,
        timestamps=observed_data.index.values
    )
    
    tsrv = tsrv_estimator.compute(
        prices=observed_data['price'].values,
        timestamps=observed_data.index.values
    )
    
    qmle = qmle_estimator.compute(
        prices=observed_data['price'].values,
        timestamps=observed_data.index.values
    )
    
    # Convert to annualized volatility
    annualized_vol_true = np.sqrt(true_iv * 252)
    annualized_vol_rv = np.sqrt(rv_5min * 252)
    annualized_vol_rk = np.sqrt(rk * 252)
    annualized_vol_msrv = np.sqrt(msrv * 252)
    annualized_vol_tsrv = np.sqrt(tsrv * 252)
    annualized_vol_qmle = np.sqrt(qmle * 252)
    
    # Calculate mean absolute percentage error (MAPE)
    mape_rv = np.mean(np.abs(rv_5min/true_iv - 1)) * 100
    mape_rk = np.mean(np.abs(rk/true_iv - 1)) * 100
    mape_msrv = np.mean(np.abs(msrv/true_iv - 1)) * 100
    mape_tsrv = np.mean(np.abs(tsrv/true_iv - 1)) * 100
    mape_qmle = np.mean(np.abs(qmle/true_iv - 1)) * 100
    
    # Print results
    print("Comparison of Noise-Robust Estimators:")
    print(f"Mean Absolute Percentage Error (MAPE):")
    print(f"  RV (5min): {mape_rv:.2f}%")
    print(f"  Realized Kernel: {mape_rk:.2f}%")
    print(f"  Multiscale RV: {mape_msrv:.2f}%")
    print(f"  Two-Scale RV: {mape_tsrv:.2f}%")
    print(f"  QMLE: {mape_qmle:.2f}%")
    
    # Plot comparison
    plt.figure(figsize=(15, 6))
    
    x = np.arange(1, n_days+1)
    width = 0.15
    
    plt.bar(x - 2.5*width, annualized_vol_true, width, label='True')
    plt.bar(x - 1.5*width, annualized_vol_rv, width, label='RV (5min)')
    plt.bar(x - 0.5*width, annualized_vol_rk, width, label='RK')
    plt.bar(x + 0.5*width, annualized_vol_msrv, width, label='MSRV')
    plt.bar(x + 1.5*width, annualized_vol_tsrv, width, label='TSRV')
    plt.bar(x + 2.5*width, annualized_vol_qmle, width, label='QMLE')
    
    plt.title('Comparison of Noise-Robust Volatility Estimators')
    plt.xlabel('Day')
    plt.ylabel('Annualized Volatility (%)')
    plt.xticks(x)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Plot MAPE comparison
    plt.figure(figsize=(10, 6))
    estimators = ['RV (5min)', 'RK', 'MSRV', 'TSRV', 'QMLE']
    mapes = [mape_rv, mape_rk, mape_msrv, mape_tsrv, mape_qmle]
    
    plt.bar(estimators, mapes)
    plt.title('Mean Absolute Percentage Error (MAPE) of Volatility Estimators')
    plt.ylabel('MAPE (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

Multivariate Realized Volatility
==============================

Realized Covariance
-----------------

Estimating covariance between multiple assets:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mfe.models.realized import RealizedCovariance
    
    # Generate simulated high-frequency data for two assets
    np.random.seed(42)
    n_days = 5
    n_intraday = 100  # 100 observations per day
    
    # Create timestamps
    timestamps = []
    for day in range(n_days):
        day_date = pd.Timestamp(f'2023-01-{day+1:02d}')
        for i in range(n_intraday):
            minute = 9*60 + 30 + i * (6.5*60 / n_intraday)
            hour = int(minute // 60)
            minute = int(minute % 60)
            timestamps.append(day_date + pd.Timedelta(hours=hour, minutes=minute))
    
    # Generate correlated returns
    # Correlation varies by day
    correlations = [0.3, 0.5, 0.7, 0.2, 0.6]
    
    # Initialize price arrays
    n_total = n_days * n_intraday
    log_prices1 = np.zeros(n_total)
    log_prices2 = np.zeros(n_total)
    
    # Generate correlated returns for each day
    for day in range(n_days):
        # Correlation matrix for this day
        corr = correlations[day]
        cov_matrix = np.array([[1.0, corr], [corr, 1.0]]) * (0.001**2)
        
        # Generate correlated returns
        day_returns = np.random.multivariate_normal(
            mean=[0, 0],
            cov=cov_matrix,
            size=n_intraday
        )
        
        # Accumulate returns to log prices
        start_idx = day * n_intraday
        end_idx = (day + 1) * n_intraday
        
        if day == 0:
            log_prices1[start_idx:end_idx] = np.cumsum(day_returns[:, 0])
            log_prices2[start_idx:end_idx] = np.cumsum(day_returns[:, 1])
        else:
            log_prices1[start_idx:end_idx] = log_prices1[start_idx-1] + np.cumsum(day_returns[:, 0])
            log_prices2[start_idx:end_idx] = log_prices2[start_idx-1] + np.cumsum(day_returns[:, 1])
    
    # Convert to prices
    prices1 = np.exp(log_prices1)
    prices2 = np.exp(log_prices2)
    
    # Create DataFrame
    hf_data = pd.DataFrame({
        'price1': prices1,
        'price2': prices2
    }, index=timestamps)
    
    # Create realized covariance estimator
    rcov_estimator = RealizedCovariance()
    
    # Estimate daily realized covariance
    rcov = rcov_estimator.compute(
        prices=[hf_data['price1'].values, hf_data['price2'].values],
        timestamps=hf_data.index.values,
        sampling='5min'
    )
    
    # Extract variances and covariances
    var1 = rcov[:, 0, 0]  # Variance of asset 1
    var2 = rcov[:, 1, 1]  # Variance of asset 2
    cov12 = rcov[:, 0, 1]  # Covariance between assets 1 and 2
    
    # Calculate realized correlation
    rcorr = cov12 / np.sqrt(var1 * var2)
    
    # Print results
    print("Daily Realized Covariance and Correlation:")
    for day in range(n_days):
        print(f"Day {day+1}:")
        print(f"  True Correlation: {correlations[day]:.2f}")
        print(f"  Realized Correlation: {rcorr[day]:.2f}")
        print(f"  Realized Variance (Asset 1): {var1[day]:.6f}")
        print(f"  Realized Variance (Asset 2): {var2[day]:.6f}")
        print(f"  Realized Covariance: {cov12[day]:.6f}")
    
    # Plot realized correlation vs. true correlation
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_days+1), correlations, 'o-', label='True Correlation')
    plt.plot(range(1, n_days+1), rcorr, 's-', label='Realized Correlation')
    plt.title('True vs. Realized Correlation')
    plt.xlabel('Day')
    plt.ylabel('Correlation')
    plt.xticks(range(1, n_days+1))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Plot realized covariance matrix for day 3 (highest correlation)
    plt.figure(figsize=(8, 6))
    plt.imshow(rcov[2], cmap='coolwarm')
    plt.colorbar(label='Covariance')
    plt.title(f'Realized Covariance Matrix (Day 3)')
    plt.xticks([0, 1], ['Asset 1', 'Asset 2'])
    plt.yticks([0, 1], ['Asset 1', 'Asset 2'])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f'{rcov[2, i, j]:.6f}', 
                     ha='center', va='center', color='white')
    plt.tight_layout()
    plt.show()

Multivariate Realized Kernel
--------------------------

Noise-robust covariance estimation:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mfe.models.realized import RealizedCovariance, MultivariateRealizedKernel
    
    # Generate simulated high-frequency data for two assets with microstructure noise
    np.random.seed(42)
    n_days = 5
    n_intraday = 200  # 200 observations per day
    
    # Create timestamps
    timestamps = []
    for day in range(n_days):
        day_date = pd.Timestamp(f'2023-01-{day+1:02d}')
        for i in range(n_intraday):
            minute = 9*60 + 30 + i * (6.5*60 / n_intraday)
            hour = int(minute // 60)
            minute = int(minute % 60)
            timestamps.append(day_date + pd.Timedelta(hours=hour, minutes=minute))
    
    # Generate correlated returns
    # Correlation varies by day
    correlations = [0.3, 0.5, 0.7, 0.2, 0.6]
    
    # Initialize arrays
    n_total = n_days * n_intraday
    efficient_log_prices1 = np.zeros(n_total)
    efficient_log_prices2 = np.zeros(n_total)
    
    # Generate correlated returns for each day
    for day in range(n_days):
        # Correlation matrix for this day
        corr = correlations[day]
        cov_matrix = np.array([[1.0, corr], [corr, 1.0]]) * (0.001**2)
        
        # Generate correlated returns
        day_returns = np.random.multivariate_normal(
            mean=[0, 0],
            cov=cov_matrix,
            size=n_intraday
        )
        
        # Accumulate returns to log prices
        start_idx = day * n_intraday
        end_idx = (day + 1) * n_intraday
        
        if day == 0:
            efficient_log_prices1[start_idx:end_idx] = np.cumsum(day_returns[:, 0])
            efficient_log_prices2[start_idx:end_idx] = np.cumsum(day_returns[:, 1])
        else:
            efficient_log_prices1[start_idx:end_idx] = efficient_log_prices1[start_idx-1] + np.cumsum(day_returns[:, 0])
            efficient_log_prices2[start_idx:end_idx] = efficient_log_prices2[start_idx-1] + np.cumsum(day_returns[:, 1])
    
    # Add microstructure noise
    noise_std = 0.0005  # Noise standard deviation
    noise1 = np.random.normal(0, noise_std, n_total)
    noise2 = np.random.normal(0, noise_std, n_total)
    
    observed_log_prices1 = efficient_log_prices1 + noise1
    observed_log_prices2 = efficient_log_prices2 + noise2
    
    # Convert to prices
    observed_prices1 = np.exp(observed_log_prices1)
    observed_prices2 = np.exp(observed_log_prices2)
    
    # Create DataFrame
    hf_data = pd.DataFrame({
        'price1': observed_prices1,
        'price2': observed_prices2
    }, index=timestamps)
    
    # Create estimators
    rcov_estimator = RealizedCovariance()
    mrk_estimator = MultivariateRealizedKernel(kernel_type='parzen')
    
    # Estimate daily realized covariance
    rcov = rcov_estimator.compute(
        prices=[hf_data['price1'].values, hf_data['price2'].values],
        timestamps=hf_data.index.values,
        sampling='5min'
    )
    
    # Estimate daily multivariate realized kernel
    mrk = mrk_estimator.compute(
        prices=[hf_data['price1'].values, hf_data['price2'].values],
        timestamps=hf_data.index.values
    )
    
    # Extract correlations
    rcov_corr = rcov[:, 0, 1] / np.sqrt(rcov[:, 0, 0] * rcov[:, 1, 1])
    mrk_corr = mrk[:, 0, 1] / np.sqrt(mrk[:, 0, 0] * mrk[:, 1, 1])
    
    # Print results
    print("Comparison of Realized Covariance and Multivariate Realized Kernel:")
    for day in range(n_days):
        print(f"Day {day+1}:")
        print(f"  True Correlation: {correlations[day]:.2f}")
        print(f"  RC Correlation: {rcov_corr[day]:.2f}")
        print(f"  MRK Correlation: {mrk_corr[day]:.2f}")
        print(f"  RC Bias: {(rcov_corr[day]/correlations[day]-1)*100:.2f}%")
        print(f"  MRK Bias: {(mrk_corr[day]/correlations[day]-1)*100:.2f}%")
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    x = np.arange(1, n_days+1)
    width = 0.25
    
    plt.bar(x - width, correlations, width, label='True')
    plt.bar(x, rcov_corr, width, label='RC')
    plt.bar(x + width, mrk_corr, width, label='MRK')
    
    plt.title('Comparison of Correlation Estimators with Microstructure Noise')
    plt.xlabel('Day')
    plt.ylabel('Correlation')
    plt.xticks(x)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Calculate mean absolute percentage error (MAPE)
    mape_rcov = np.mean(np.abs(rcov_corr/correlations - 1)) * 100
    mape_mrk = np.mean(np.abs(mrk_corr/correlations - 1)) * 100
    
    # Plot MAPE comparison
    plt.figure(figsize=(8, 6))
    estimators = ['RC', 'MRK']
    mapes = [mape_rcov, mape_mrk]
    
    plt.bar(estimators, mapes)
    plt.title('Mean Absolute Percentage Error (MAPE) of Correlation Estimators')
    plt.ylabel('MAPE (%)')
    plt.tight_layout()
    plt.show()

Asynchronous Processing for Large Datasets
=======================================

The MFE Toolbox supports asynchronous processing for handling large high-frequency datasets efficiently:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import asyncio
    from mfe.models.realized import RealizedVariance, RealizedKernel
    
    # Generate a large high-frequency dataset
    np.random.seed(42)
    n_days = 20
    n_intraday = 1000  # 1000 observations per day
    
    # Create timestamps
    timestamps = []
    for day in range(n_days):
        day_date = pd.Timestamp(f'2023-01-{day+1:02d}')
        for i in range(n_intraday):
            minute = 9*60 + 30 + i * (6.5*60 / n_intraday)
            hour = int(minute // 60)
            minute = int(minute % 60)
            timestamps.append(day_date + pd.Timedelta(hours=hour, minutes=minute))
    
    # Generate price process with time-varying volatility
    n_total = n_days * n_intraday
    volatility = np.ones(n_total) * 0.001
    
    # Add volatility clusters
    for i in range(3):
        cluster_start = np.random.randint(0, n_total - n_intraday)
        cluster_length = np.random.randint(n_intraday, 3 * n_intraday)
        cluster_end = min(cluster_start + cluster_length, n_total)
        volatility[cluster_start:cluster_end] *= np.random.uniform(2, 4)
    
    returns = np.random.normal(0, volatility)
    log_prices = np.cumsum(returns)
    prices = np.exp(log_prices)
    
    # Create DataFrame
    hf_data = pd.DataFrame({
        'price': prices
    }, index=timestamps)
    
    # Create estimators
    rv_estimator = RealizedVariance()
    rk_estimator = RealizedKernel()
    
    # Define asynchronous function for realized variance estimation
    async def compute_rv_async():
        # Define progress callback
        def progress_callback(percent, message):
            print(f"{percent:.1f}% complete: {message}")
        
        # Compute realized variance asynchronously
        rv = await rv_estimator.compute_async(
            prices=hf_data['price'].values,
            timestamps=hf_data.index.values,
            sampling='5min',
            progress_callback=progress_callback
        )
        
        return rv
    
    # Define asynchronous function for realized kernel estimation
    async def compute_rk_async():
        # Define progress callback
        def progress_callback(percent, message):
            print(f"{percent:.1f}% complete: {message}")
        
        # Compute realized kernel asynchronously
        rk = await rk_estimator.compute_async(
            prices=hf_data['price'].values,
            timestamps=hf_data.index.values,
            progress_callback=progress_callback
        )
        
        return rk
    
    # Run asynchronous computations
    async def main():
        print("Computing Realized Variance...")
        rv = await compute_rv_async()
        
        print("\nComputing Realized Kernel...")
        rk = await compute_rk_async()
        
        return rv, rk
    
    # Execute the async function
    rv, rk = asyncio.run(main())
    
    # Convert to annualized volatility
    annualized_vol_rv = np.sqrt(rv * 252)
    annualized_vol_rk = np.sqrt(rk * 252)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(range(1, n_days+1), annualized_vol_rv, 'o-', label='RV')
    plt.plot(range(1, n_days+1), annualized_vol_rk, 's-', label='RK')
    plt.title('Realized Volatility Estimates')
    plt.ylabel('Annualized Volatility (%)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(range(1, n_days+1), annualized_vol_rk / annualized_vol_rv, 'o-')
    plt.axhline(y=1, color='r', linestyle='--')
    plt.title('Ratio of RK to RV')
    plt.xlabel('Day')
    plt.ylabel('Ratio')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

Advanced Applications
==================

Volatility Forecasting with Realized Measures
------------------------------------------

Using realized volatility for forecasting:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mfe.models.univariate import HEAVY
    from mfe.models.realized import RealizedVariance
    from mfe.models.time_series import HAR
    
    # Generate simulated daily returns and intraday data
    np.random.seed(42)
    n_days = 100
    n_intraday = 100  # 100 observations per day
    
    # Create timestamps
    timestamps = []
    for day in range(n_days):
        day_date = pd.Timestamp(f'2023-01-{day+1:02d}')
        for i in range(n_intraday):
            minute = 9*60 + 30 + i * (6.5*60 / n_intraday)
            hour = int(minute // 60)
            minute = int(minute % 60)
            timestamps.append(day_date + pd.Timedelta(hours=hour, minutes=minute))
    
    # Generate volatility process (persistent with occasional spikes)
    daily_vol = np.ones(n_days) * 0.01
    
    # Add AR(1) structure
    for i in range(1, n_days):
        daily_vol[i] = 0.0001 + 0.9 * daily_vol[i-1] + 0.0002 * np.random.normal()
    
    # Add occasional volatility spikes
    spike_days = np.random.choice(n_days, size=5, replace=False)
    for day in spike_days:
        daily_vol[day] *= np.random.uniform(2, 3)
    
    # Generate returns based on volatility
    daily_returns = np.random.normal(0, daily_vol)
    
    # Generate intraday returns
    intraday_returns = np.zeros(n_days * n_intraday)
    for day in range(n_days):
        day_vol = daily_vol[day] / np.sqrt(n_intraday)
        start_idx = day * n_intraday
        end_idx = (day + 1) * n_intraday
        intraday_returns[start_idx:end_idx] = np.random.normal(0, day_vol, n_intraday)
    
    # Convert to prices
    log_prices = np.cumsum(intraday_returns)
    prices = np.exp(log_prices)
    
    # Create DataFrames
    hf_data = pd.DataFrame({
        'price': prices
    }, index=timestamps)
    
    daily_data = pd.DataFrame({
        'returns': daily_returns,
        'volatility': daily_vol  # True volatility for comparison
    }, index=[timestamps[i*n_intraday] for i in range(n_days)])
    
    # Compute realized volatility
    rv_estimator = RealizedVariance()
    rv = rv_estimator.compute(
        prices=hf_data['price'].values,
        timestamps=hf_data.index.values,
        sampling='5min'
    )
    
    # Add realized volatility to daily data
    daily_data['rv'] = np.sqrt(rv)
    
    # Split data into training and testing sets
    train_size = int(0.7 * n_days)
    train_data = daily_data.iloc[:train_size]
    test_data = daily_data.iloc[train_size:]
    
    # Fit HAR model for realized volatility
    har_model = HAR(lags=[1, 5, 22])  # 1-day, 5-day, and 22-day lags
    har_result = har_model.fit(train_data['rv'].values)
    
    # Generate forecasts
    har_forecasts = har_result.forecast(
        horizon=len(test_data),
        x_t=train_data['rv'].values
    )
    
    # Fit HEAVY model
    heavy_model = HEAVY()
    heavy_result = heavy_model.fit(
        returns=train_data['returns'].values,
        realized_measures=train_data['rv'].values**2
    )
    
    # Generate HEAVY forecasts
    heavy_forecasts = np.zeros(len(test_data))
    
    # Initialize with last in-sample volatility
    sigma2 = heavy_result.conditional_variance[-1]
    rm = train_data['rv'].values[-1]**2
    
    for i in range(len(test_data)):
        # Update volatility forecast
        sigma2 = heavy_result.params.omega + heavy_result.params.alpha * rm + heavy_result.params.beta * sigma2
        heavy_forecasts[i] = np.sqrt(sigma2)
        
        # Update realized measure for next iteration (if available)
        if i < len(test_data) - 1:
            rm = test_data['rv'].values[i]**2
    
    # Evaluate forecasts
    har_mse = np.mean((har_forecasts - test_data['volatility'])**2)
    heavy_mse = np.mean((heavy_forecasts - test_data['volatility'])**2)
    
    print("Forecast Evaluation:")
    print(f"HAR Model MSE: {har_mse:.8f}")
    print(f"HEAVY Model MSE: {heavy_mse:.8f}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot volatility and realized volatility
    plt.subplot(2, 1, 1)
    plt.plot(daily_data.index, daily_data['volatility'], 'k-', label='True Volatility')
    plt.plot(daily_data.index, daily_data['rv'], 'b-', alpha=0.7, label='Realized Volatility')
    plt.axvline(x=train_data.index[-1], color='r', linestyle='--', label='Train/Test Split')
    plt.title('Volatility and Realized Volatility')
    plt.ylabel('Volatility')
    plt.legend()
    
    # Plot forecasts
    plt.subplot(2, 1, 2)
    plt.plot(test_data.index, test_data['volatility'], 'k-', label='True Volatility')
    plt.plot(test_data.index, har_forecasts, 'g-', label=f'HAR Forecast (MSE: {har_mse:.8f})')
    plt.plot(test_data.index, heavy_forecasts, 'r-', label=f'HEAVY Forecast (MSE: {heavy_mse:.8f})')
    plt.title('Volatility Forecasts')
    plt.ylabel('Volatility')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

Jump Detection
-----------

Detecting jumps in price processes:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mfe.models.realized import RealizedVariance, BipowerVariation
    
    # Generate simulated high-frequency data with jumps
    np.random.seed(42)
    n_days = 10
    n_intraday = 100  # 100 observations per day
    
    # Create timestamps
    timestamps = []
    for day in range(n_days):
        day_date = pd.Timestamp(f'2023-01-{day+1:02d}')
        for i in range(n_intraday):
            minute = 9*60 + 30 + i * (6.5*60 / n_intraday)
            hour = int(minute // 60)
            minute = int(minute % 60)
            timestamps.append(day_date + pd.Timedelta(hours=hour, minutes=minute))
    
    # Generate continuous price process
    volatility = np.ones(n_days * n_intraday) * 0.001
    continuous_returns = np.random.normal(0, volatility)
    
    # Add jumps
    jump_days = [2, 5, 8]  # Days with jumps
    jump_sizes = [0.02, -0.015, 0.025]  # Jump sizes
    
    # Copy continuous returns to create returns with jumps
    jump_returns = continuous_returns.copy()
    
    # Add jumps at specific times
    for i, day in enumerate(jump_days):
        jump_time = day * n_intraday + n_intraday // 2  # Middle of the day
        jump_returns[jump_time] += jump_sizes[i]
    
    # Convert to prices
    continuous_log_prices = np.cumsum(continuous_returns)
    jump_log_prices = np.cumsum(jump_returns)
    
    continuous_prices = np.exp(continuous_log_prices)
    jump_prices = np.exp(jump_log_prices)
    
    # Create DataFrames
    continuous_data = pd.DataFrame({
        'price': continuous_prices
    }, index=timestamps)
    
    jump_data = pd.DataFrame({
        'price': jump_prices
    }, index=timestamps)
    
    # Create estimators
    rv_estimator = RealizedVariance()
    bv_estimator = BipowerVariation()
    
    # Compute realized variance and bipower variation
    rv = rv_estimator.compute(
        prices=jump_data['price'].values,
        timestamps=jump_data.index.values,
        sampling='5min'
    )
    
    bv = bv_estimator.compute(
        prices=jump_data['price'].values,
        timestamps=jump_data.index.values,
        sampling='5min'
    )
    
    # Compute jump component
    jump_component = np.maximum(0, rv - bv)
    
    # Compute relative jump measure
    relative_jump = jump_component / rv
    
    # Compute z-statistic for jump detection
    # Under the null of no jumps, this follows a standard normal distribution
    n_obs_per_day = 78  # Approximate number of 5-minute intervals in a trading day
    z_statistic = (rv - bv) / np.sqrt((np.pi**2/4 + np.pi - 5) * (1/n_obs_per_day) * bv**2)
    
    # Critical value for 99% confidence
    critical_value = 2.576
    
    # Detect significant jumps
    significant_jumps = z_statistic > critical_value
    
    # Print results
    print("Jump Detection Results:")
    for day in range(n_days):
        print(f"Day {day+1}:")
        print(f"  RV = {rv[day]:.6f}")
        print(f"  BV = {bv[day]:.6f}")
        print(f"  Jump Component = {jump_component[day]:.6f}")
        print(f"  Relative Jump = {relative_jump[day]*100:.2f}%")
        print(f"  Z-statistic = {z_statistic[day]:.4f}")
        print(f"  Significant Jump: {'Yes' if significant_jumps[day] else 'No'}")
        print(f"  True Jump: {'Yes' if day+1 in jump_days else 'No'}")
    
    # Plot prices
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(jump_data.index, jump_data['price'])
    
    # Mark jump days
    for day in jump_days:
        jump_time = timestamps[day * n_intraday + n_intraday // 2]
        plt.axvline(x=jump_time, color='r', linestyle='--')
    
    plt.title('Price Process with Jumps')
    plt.ylabel('Price')
    
    # Plot jump measures
    plt.subplot(2, 1, 2)
    plt.bar(range(1, n_days+1), relative_jump*100)
    plt.axhline(y=0, color='k', linestyle='-')
    
    # Mark significant jumps
    for day in range(n_days):
        if significant_jumps[day]:
            plt.plot(day+1, relative_jump[day]*100, 'ro', markersize=10)
    
    plt.title('Relative Jump Measure (% of RV)')
    plt.xlabel('Day')
    plt.ylabel('Jump Component (%)')
    plt.xticks(range(1, n_days+1))
    
    plt.tight_layout()
    plt.show()
    
    # Plot z-statistics
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, n_days+1), z_statistic)
    plt.axhline(y=critical_value, color='r', linestyle='--', label='99% Critical Value')
    
    # Mark true jump days
    for day in jump_days:
        plt.plot(day, z_statistic[day-1], 'go', markersize=10)
    
    plt.title('Jump Test Z-Statistics')
    plt.xlabel('Day')
    plt.ylabel('Z-Statistic')
    plt.xticks(range(1, n_days+1))
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot z-statistics
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, n_days+1), z_statistic)
    plt.axhline(y=critical_value, color='r', linestyle='--', label='99% Critical Value')
    
    # Mark true jump days
    for day in jump_days:
        plt.plot(day, z_statistic[day-1], 'go', markersize=10)
    
    plt.title('Jump Test Z-Statistics')
    plt.xlabel('Day')
    plt.ylabel('Z-Statistic')
    plt.xticks(range(1, n_days+1))
    plt.legend()
    plt.tight_layout()
    plt.show()

Conclusion
=========

The high-frequency financial econometrics tools in the MFE Toolbox provide a comprehensive suite for analyzing intraday price data and estimating volatility and covariance. These tools are essential for modern risk management, market microstructure research, and high-frequency trading applications.

Key features include:

- Robust handling of irregularly spaced high-frequency data using Pandas time series capabilities
- Comprehensive set of realized volatility estimators with Numba acceleration for optimal performance
- Noise-robust estimators for handling market microstructure noise
- Multivariate covariance estimation tools
- Asynchronous processing support for large datasets
- Integration with other MFE Toolbox components for volatility forecasting and risk management

For more advanced applications, see the documentation on univariate and multivariate volatility models, time series analysis, and bootstrap methods.
```