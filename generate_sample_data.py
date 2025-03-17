import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate dates
start_date = datetime(2023, 1, 1)
dates = [start_date + timedelta(days=x) for x in range(365)]  # One year of daily data

# Generate main time series with trend and seasonality
t = np.arange(len(dates))
trend = 0.01 * t  # Upward trend
seasonal = 2 * np.sin(2 * np.pi * t / 30)  # Monthly seasonality
noise = np.random.normal(0, 0.5, len(dates))  # Random noise
main_series = trend + seasonal + noise

# Generate exogenous variables
exog1 = np.sin(2 * np.pi * t / 90)  # Quarterly cycle
exog2 = 0.5 * np.random.randn(len(dates))  # Random external factor
exog3 = 0.3 * trend + 0.2 * np.random.randn(len(dates))  # Related trend with noise

# Create DataFrame
df = pd.DataFrame({
    'Date': dates,
    'Price': main_series,  # Main time series
    'Market_Index': exog1,  # Exogenous variable 1
    'Interest_Rate': exog2,  # Exogenous variable 2
    'Economic_Indicator': exog3  # Exogenous variable 3
})

# Save to CSV
df.to_csv('sample_timeseries.csv', index=False)

print("Sample time series data has been generated and saved to 'sample_timeseries.csv'")
print("\nData structure:")
print(df.head())
print("\nSummary statistics:")
print(df.describe()) 