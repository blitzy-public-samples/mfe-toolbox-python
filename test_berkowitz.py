import numpy as np
import pandas as pd
import pytest
from scipy import stats
from mfe.models.distributions.utils import berkowitz_test

def test_berkowitz_normal_data():
    """Test Berkowitz test with normally distributed data."""
    # Generate normal data
    np.random.seed(42)
    n_samples = 1000
    data = np.random.standard_normal(n_samples)
    
    # Test against normal distribution
    result = berkowitz_test(data)
    
    # For normal data, we expect high p-value (> 0.05)
    assert result["p_value"] > 0.05, "Berkowitz test incorrectly rejected normal data"
    assert result["test_statistic"] >= 0, "Test statistic should be non-negative"
    assert "mu" in result and "rho" in result and "sigma2" in result, "Missing AR(1) parameters"

def test_berkowitz_student_t_data():
    """Test Berkowitz test with Student's t-distributed data against normal distribution."""
    # Generate t-distributed data
    np.random.seed(42)
    n_samples = 1000
    df = 2  # Very low degrees of freedom for heavy tails
    data = np.random.standard_t(df, size=n_samples)
    
    # Standardize the data
    data = (data - np.mean(data)) / np.std(data)
    
    # Transform data to uniform using t-distribution CDF
    uniform_data = stats.t.cdf(data, df=df)
    
    # Test the transformed data
    result = berkowitz_test(uniform_data)
    
    # For t-distributed data tested against normal, we expect low p-value (< 0.05)
    assert result["p_value"] < 0.05, "Berkowitz test failed to reject non-normal data"

def test_berkowitz_uniform_data():
    """Test Berkowitz test with uniform data transformed to normal."""
    # Generate uniform data
    np.random.seed(42)
    n_samples = 1000
    data = np.random.uniform(0, 1, n_samples)
    
    # Transform to normal using inverse normal CDF
    transformed_data = stats.norm.ppf(data)
    
    # Test transformed data
    result = berkowitz_test(transformed_data)
    
    # For transformed uniform data, we expect high p-value (> 0.05)
    assert result["p_value"] > 0.05, "Berkowitz test incorrectly rejected transformed uniform data"

def test_berkowitz_invalid_inputs():
    """Test Berkowitz test with invalid inputs."""
    # Test with NaN values
    data_with_nan = np.array([1.0, 2.0, np.nan, 4.0])
    with pytest.raises(ValueError, match="Data contains NaN or infinite values"):
        berkowitz_test(data_with_nan)
    
    # Test with infinite values
    data_with_inf = np.array([1.0, 2.0, np.inf, 4.0])
    with pytest.raises(ValueError, match="Data contains NaN or infinite values"):
        berkowitz_test(data_with_inf)
    
    # Test with empty array - should raise warning and return NaN values
    with pytest.warns(RuntimeWarning):
        result = berkowitz_test(np.array([]))
        assert np.isnan(result["test_statistic"])
        assert np.isnan(result["p_value"])

def test_berkowitz_different_input_types():
    """Test Berkowitz test with different input types."""
    np.random.seed(42)
    data = np.random.standard_normal(1000)
    
    # Test with numpy array
    result_array = berkowitz_test(data)
    
    # Test with list
    result_list = berkowitz_test(data.tolist())
    
    # Test with pandas Series
    result_series = berkowitz_test(pd.Series(data))
    
    # Results should be consistent across input types
    assert abs(result_array["test_statistic"] - result_list["test_statistic"]) < 1e-10
    assert abs(result_array["p_value"] - result_list["p_value"]) < 1e-10
    assert abs(result_array["test_statistic"] - result_series["test_statistic"]) < 1e-10
    assert abs(result_array["p_value"] - result_series["p_value"]) < 1e-10

def test_berkowitz_small_sample():
    """Test Berkowitz test with small sample size."""
    np.random.seed(42)
    data = np.random.standard_normal(30)  # Small sample size
    
    # Test should still run without errors
    result = berkowitz_test(data)
    
    # Basic checks
    assert result["test_statistic"] >= 0
    assert 0 <= result["p_value"] <= 1

def test_berkowitz_extreme_values():
    """Test Berkowitz test with extreme but valid values."""
    np.random.seed(42)
    # Generate mixture of normal distributions (one standard, one extreme)
    n_samples = 1000
    standard_data = np.random.standard_normal(n_samples // 2)
    extreme_data = np.random.normal(loc=10, scale=5, size=n_samples // 2)
    data = np.concatenate([standard_data, extreme_data])
    
    # Test should handle extreme values
    result = berkowitz_test(data)
    
    # For non-standard normal data, we expect low p-value
    assert result["p_value"] < 0.05, "Berkowitz test failed to reject extreme mixture distribution" 