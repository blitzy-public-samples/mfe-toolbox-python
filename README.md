# MFE Toolbox
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyPI version](https://badge.fury.io/py/mfe-toolbox.svg)](https://badge.fury.io/py/mfe-toolbox)
[![Build Status](https://github.com/bashtage/arch/workflows/Build/badge.svg)](https://github.com/bashtage/arch/actions)
[![Coverage Status](https://codecov.io/gh/bashtage/arch/branch/main/graph/badge.svg)](https://codecov.io/gh/bashtage/arch)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Python-based suite for financial econometrics, time series analysis, and risk modeling, targeting Python 3.12. This package represents a complete modernization of the original MATLAB-based toolbox (formerly version 4.0, released 28-Oct-2009).

## Features

The MFE Toolbox provides a wide range of capabilities for financial econometrics and time series analysis:

- **Univariate volatility modeling**: GARCH, EGARCH, TARCH, and other variants
- **Multivariate volatility modeling**: BEKK, DCC, RARCH, and related models
- **ARMA/ARMAX time series modeling and forecasting**
- **Bootstrap methods for dependent data**
- **Non-parametric volatility estimation** (realized volatility)
- **Classical statistical tests and distributions**
- **Vector autoregression (VAR) analysis**
- **Principal component analysis and cross-sectional econometrics**

## Installation

The MFE Toolbox requires Python 3.12 or later. You can install it using pip:

```bash
pip install mfe-toolbox
```

### Dependencies

The toolbox leverages powerful Python libraries:

- NumPy (≥1.26.0) for matrix operations
- SciPy (≥1.11.3) for optimization and statistical functions
- Pandas (≥2.1.1) for time series handling
- Statsmodels (≥0.14.0) for econometric modeling
- Numba (≥0.58.0) for performance optimization
- Matplotlib (≥3.8.0) for visualization
- PyQt6 (≥6.5.0) for GUI components (optional, only required for the ARMAX GUI)

## Quick Start

```python
import numpy as np
import pandas as pd
from mfe import GARCH, ARMA, DCC, BlockBootstrap, RealizedVariance

# Univariate GARCH example
returns = np.random.normal(0, 1, 1000)
garch_model = GARCH(p=1, q=1)
garch_result = garch_model.fit(returns)
print(garch_result.summary())

# ARMA modeling example
data = np.random.normal(0, 1, 1000)
arma_model = ARMA(p=1, q=1)
arma_result = arma_model.fit(data)
forecast = arma_result.forecast(horizon=10)
print(forecast)

# Bootstrap example
data = np.random.normal(0, 1, 1000)
bootstrap = BlockBootstrap(block_size=10)
bootstrap_samples = bootstrap.generate(data, num_samples=1000)
```

## Architecture

The MFE Toolbox is built with a modern Python architecture:

- **Class-based implementations** with strict type hints and dataclasses
- **Performance optimization** through Numba's just-in-time compilation (using @jit decorators)
- **Asynchronous processing** support for long-duration computations
- **Integration with Python's scientific ecosystem** (NumPy, SciPy, Pandas, Statsmodels)

## Documentation

Comprehensive documentation is available at [https://mfe-toolbox.readthedocs.io/](https://mfe-toolbox.readthedocs.io/) (placeholder URL).

## Examples

### Univariate GARCH Model

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mfe.models.univariate import GARCH

# Generate sample data
np.random.seed(42)
returns = np.random.normal(0, 1, 1000)

# Create and fit GARCH model
model = GARCH(p=1, q=1)
result = model.fit(returns)

# Print model summary
print(result.summary())

# Plot conditional volatility
plt.figure(figsize=(10, 6))
plt.plot(result.conditional_volatility)
plt.title('GARCH(1,1) Conditional Volatility')
plt.xlabel('Time')
plt.ylabel('Volatility')
plt.show()
```

### Multivariate DCC Model

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mfe.models.multivariate import DCC

# Generate sample multivariate data
np.random.seed(42)
returns = np.random.multivariate_normal(
    mean=[0, 0, 0], 
    cov=[[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]], 
    size=1000
)

# Create and fit DCC model
model = DCC(p=1, q=1)
result = model.fit(returns)

# Print model summary
print(result.summary())

# Plot time-varying correlations
plt.figure(figsize=(10, 6))
plt.plot(result.correlations[:, 0, 1], label='Asset 1-2')
plt.plot(result.correlations[:, 0, 2], label='Asset 1-3')
plt.plot(result.correlations[:, 1, 2], label='Asset 2-3')
plt.title('DCC Time-Varying Correlations')
plt.xlabel('Time')
plt.ylabel('Correlation')
plt.legend()
plt.show()
```

### ARMAX Modeling with GUI

```python
from mfe.ui import armax_app

# Launch the ARMAX GUI
app = armax_app.launch()
```

## Development

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/bashtage/arch.git
cd arch

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e " .[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=mfe

# Run specific test module
pytest tests/test_univariate.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code follows the project's style guidelines and includes appropriate tests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Kevin Sheppard, the original author of the MATLAB MFE Toolbox
- University of Oxford, Department of Economics

## Citation

If you use the MFE Toolbox in your research, please cite:

```
Sheppard, K. (2023). MFE Toolbox: A Python package for financial econometrics.
https://github.com/bashtage/arch
```
