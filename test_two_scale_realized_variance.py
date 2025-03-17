import numpy as np
from mfe.models.realized.twoscale_variance import TwoScaleVarianceEstimator, TwoScaleVarianceConfig, TwoScaleVarianceResult

def test_two_scale_realized_variance():
    """
    Test the Two-scale Realized Variance (TSRV) estimator with minimal configuration.
    This test simulates high-frequency price data and adds microstructure noise.
    """
    # Number of observations
    n = 1000
    print(f"Number of observations: {n}")

    # True daily variance
    sigma2 = 0.04
    print(f"True daily variance: {sigma2}")

    # Simulate efficient prices
    np.random.seed(42)
    dt = 1.0 / n
    dW = np.random.normal(0, np.sqrt(dt), n)
    sigma = np.sqrt(sigma2)
    efficient_returns = sigma * dW
    efficient_log_prices = np.cumsum(efficient_returns)
    efficient_prices = np.exp(efficient_log_prices)

    # Add microstructure noise
    noise_std = 0.001
    noise = np.random.normal(0, noise_std, n)
    observed_prices = efficient_prices * (1 + noise)
    times = np.linspace(0, 1, n)

    # Create configuration
    config = TwoScaleVarianceConfig(
        estimate_noise=True,
        noise_method='bandi-russell',
        auto_scale=True,
        bias_correction=True
    )

    # Initialize estimator
    estimator = TwoScaleVarianceEstimator(config=config)

    # Fit estimator
    result = estimator.fit((observed_prices, times))

    # Print results
    print(f"Estimated TSRV: {result.realized_measure[0]:.6e}")
    if result.noise_variance is not None:
        print(f"Estimated noise variance: {result.noise_variance:.6e}")
    else:
        print("Noise variance estimation failed")

    # Verify results
    assert result.realized_measure is not None, "TSRV estimation failed"
    assert result.realized_measure[0] > 0, "TSRV should be positive"
    assert estimator.slow_scale == 5, f"Expected slow_scale=5, got {estimator.slow_scale}"
    assert estimator.fast_scale == 1, f"Expected fast_scale=1, got {estimator.fast_scale}"
    
    # Verify TSRV is within reasonable bounds
    # The TSRV should be roughly close to the true variance (0.04)
    # but will be smaller due to noise robustness
    assert 0.0001 < result.realized_measure[0] < 0.04, \
        f"TSRV {result.realized_measure[0]} outside expected range"

if __name__ == "__main__":
    test_two_scale_realized_variance() 