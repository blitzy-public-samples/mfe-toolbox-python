Convert MATLAB Financial Econometrics Toolbox to Python 3.12:

Core Dependencies:
- NumPy for matrix operations (replacing MATLAB arrays)
- SciPy for optimization and statistical functions
- Pandas for time series handling
- Statsmodels for econometric models
- Numba for performance optimization (replacing MEX)
- PyQt6 for GUI (replacing MATLAB GUIDE)

Architecture Changes:
1. Replace MEX optimizations with Numba @jit decorators
2. Convert matrix operations to NumPy ndarray operations
3. Implement Class-based structure for models (GARCH, ARMA, etc.)
4. Use dataclasses for parameter containers
5. Implement async/await for long computations
6. Use typing for strict type hints

Package Structure:
```
mfe/
├── core/
│   ├── bootstrap.py
│   ├── distributions.py
│   └── statistics.py
├── models/
│   ├── arma.py
│   ├── garch.py
│   └── volatility.py
├── ui/
│   ├── armax_viewer.py
│   └── widgets.py
└── utils/
    ├── matrix.py
    └── optimization.py
```

Critical Conversions:
1. MATLAB's 1-based indexing → Python's 0-based indexing
2. MATLAB's column-major → NumPy's row-major operations
3. MATLAB's matrix syntax → NumPy array operations
4. MEX C code → Numba optimized Python
5. GUI forms → PyQt6 layouts

Testing Framework:
- pytest for unit testing
- hypothesis for property-based testing
- numba.testing for performance validation