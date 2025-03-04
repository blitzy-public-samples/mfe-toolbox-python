# mfe/core/types.py

"""
Core type annotations and custom types for the MFE Toolbox.

This module defines the type system for the MFE Toolbox, providing type aliases,
custom type definitions, and protocol classes that establish a consistent type
contract across the entire codebase. These type definitions enable static type
checking, improve IDE integration, and enhance code documentation.

The type system leverages Python's typing module to create clear type contracts
for arrays, time series data, parameters, and callback functions, ensuring
type safety throughout the toolbox.
"""

import asyncio
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any, Callable, Dict, Generic, List, Literal, NewType, Optional, Protocol,
    Sequence, Set, Tuple, Type, TypeVar, Union, cast, overload, runtime_checkable
)

import numpy as np
import pandas as pd
from scipy import stats

# Type variables for generic programming
T = TypeVar('T')  # Generic type
P = TypeVar('P')  # Parameter type
R = TypeVar('R')  # Result type
D = TypeVar('D')  # Data type
M = TypeVar('M')  # Model type
S = TypeVar('S')  # Statistic type
E = TypeVar('E', bound=Exception)  # Exception type

# NumPy array type aliases with shape and dtype constraints
# These provide more specific type information than np.ndarray alone
Vector = np.ndarray  # 1D array
Matrix = np.ndarray  # 2D array
Tensor3D = np.ndarray  # 3D array
Tensor4D = np.ndarray  # 4D array

# Specialized array types for specific use cases
TimeSeriesData = Union[np.ndarray, pd.Series]  # Single time series
TimeSeriesDataFrame = Union[np.ndarray, pd.DataFrame]  # Multiple time series
ParameterVector = np.ndarray  # Vector of model parameters
CovarianceMatrix = np.ndarray  # Covariance matrix (symmetric, positive definite)
CorrelationMatrix = np.ndarray  # Correlation matrix (symmetric with ones on diagonal)
TriangularMatrix = np.ndarray  # Upper or lower triangular matrix
DiagonalMatrix = np.ndarray  # Diagonal matrix
PositiveDefiniteMatrix = np.ndarray  # Positive definite matrix
OrthogonalMatrix = np.ndarray  # Orthogonal matrix (Q'Q = I)

# Time-related types
TimeStamp = Union[pd.Timestamp, np.datetime64, float]
TimeSeries = Union[pd.Series, pd.DataFrame]
TimeIndex = Union[pd.DatetimeIndex, np.ndarray, List[Union[pd.Timestamp, np.datetime64, float]]]

# Distribution types
DistributionType = Literal("normal", "t", "skewed_t", "ged")
DistributionLike = Union[stats.rv_continuous, Callable[[np.ndarray], np.ndarray]]

# Parameter types
ParameterDict = Dict[str, Any]
ParameterConstraint = Callable[[float], bool]
ParameterTransform = Callable[[float], float]

# Callback types for reporting progress and handling events
ProgressCallback = Callable[[float, str], None]
AsyncProgressCallback = Callable[[float, str], asyncio.coroutine]
EventCallback = Callable[[str, Dict[str, Any]], None]
AsyncEventCallback = Callable[[str, Dict[str, Any]], asyncio.coroutine]

# Model specification types
ARMAOrder = Tuple[int, int]  # (p, q) for AR and MA orders
GARCHOrder = Tuple[int, int]  # (p, q) for ARCH and GARCH orders
FIGARCHOrder = Tuple[int, int, int]  # (p, d, q) for FIGARCH orders

# Result container types
EstimationResult = Dict[str, Any]
ForecastResult = Dict[str, Any]
SimulationResult = Dict[str, Any]
TestResult = Dict[str, Any]
BootstrapResult = Dict[str, Any]

# File path types
FilePath = Union[str, Path]
DirectoryPath = Union[str, Path]

# Optimization types
OptimizationMethod = Literal("BFGS", "L-BFGS-B", "SLSQP", "Nelder-Mead", "Powell", "CG", "Newton-CG")
OptimizationResult = Dict[str, Any]
ObjectiveFunction = Callable[[np.ndarray], float]
GradientFunction = Callable[[np.ndarray], np.ndarray]
HessianFunction = Callable[[np.ndarray], np.ndarray]

# Volatility model types
VolatilityType = Literal[
    "GARCH", "EGARCH", "TARCH", "GJR-GARCH", "APARCH",
    "FIGARCH", "IGARCH", "HEAVY", "AGARCH"
]

# Multivariate volatility model types
MultivariateVolatilityType = Literal[
    "BEKK", "DCC", "ADCC", "CCC", "OGARCH", "GOGARCH",
    "RARCH", "RCC", "MATRIX-GARCH", "RISKMETRICS"
]

# Time series model types
TimeSeriesType = Literal["AR", "MA", "ARMA", "ARMAX", "ARIMA", "SARIMA", "VAR", "VECM"]

# Bootstrap method types
BootstrapType = Literal["block", "stationary", "moving-block", "circular-block"]

# Realized volatility estimator types
RealizedVolatilityType = Literal[
    "RV", "BPV", "MinRV", "MedRV", "RK", "TSRV", "MSRV",
    "QRV", "QMLE", "PAV", "PBPV", "TMPV", "TV"
]

# Statistical test types
TestType = Literal[
    "Jarque-Bera", "Ljung-Box", "Box-Pierce", "LM",
    "Kolmogorov-Smirnov", "Berkowitz", "ADF", "KPSS"
]

# Cross-sectional model types
CrossSectionalType = Literal["OLS", "WLS", "GLS", "PCA", "FA"]

# Enum classes for type-safe options


class VolatilityModel(Enum):
    """Enumeration of volatility model types."""
    GARCH = auto()
    EGARCH = auto()
    TARCH = auto()
    GJR_GARCH = auto()
    APARCH = auto()
    FIGARCH = auto()
    IGARCH = auto()
    HEAVY = auto()
    AGARCH = auto()


class MultivariateVolatilityModel(Enum):
    """Enumeration of multivariate volatility model types."""
    BEKK = auto()
    DCC = auto()
    ADCC = auto()
    CCC = auto()
    OGARCH = auto()
    GOGARCH = auto()
    RARCH = auto()
    RCC = auto()
    MATRIX_GARCH = auto()
    RISKMETRICS = auto()


class TimeSeriesModel(Enum):
    """Enumeration of time series model types."""
    AR = auto()
    MA = auto()
    ARMA = auto()
    ARMAX = auto()
    ARIMA = auto()
    SARIMA = auto()
    VAR = auto()
    VECM = auto()


class BootstrapMethod(Enum):
    """Enumeration of bootstrap method types."""
    BLOCK = auto()
    STATIONARY = auto()
    MOVING_BLOCK = auto()
    CIRCULAR_BLOCK = auto()


class RealizedVolatilityEstimator(Enum):
    """Enumeration of realized volatility estimator types."""
    RV = auto()  # Realized Variance
    BPV = auto()  # Bipower Variation
    MINRV = auto()  # Minimum Realized Variance
    MEDRV = auto()  # Median Realized Variance
    RK = auto()  # Realized Kernel
    TSRV = auto()  # Two-Scale Realized Variance
    MSRV = auto()  # Multi-Scale Realized Variance
    QRV = auto()  # Quantile Realized Variance
    QMLE = auto()  # Quasi-Maximum Likelihood Estimator
    PAV = auto()  # Pre-Averaged Variance
    PBPV = auto()  # Pre-Averaged Bipower Variation
    TMPV = auto()  # Threshold Multipower Variation
    TV = auto()  # Threshold Variance


class StatisticalTest(Enum):
    """Enumeration of statistical test types."""
    JARQUE_BERA = auto()
    LJUNG_BOX = auto()
    BOX_PIERCE = auto()
    LM = auto()
    KOLMOGOROV_SMIRNOV = auto()
    BERKOWITZ = auto()
    ADF = auto()
    KPSS = auto()


class CrossSectionalModel(Enum):
    """Enumeration of cross-sectional model types."""
    OLS = auto()
    WLS = auto()
    GLS = auto()
    PCA = auto()
    FA = auto()


class Distribution(Enum):
    """Enumeration of distribution types."""
    NORMAL = auto()
    STUDENT_T = auto()
    SKEWED_T = auto()
    GED = auto()


class OptimizationAlgorithm(Enum):
    """Enumeration of optimization algorithm types."""
    BFGS = auto()
    L_BFGS_B = auto()
    SLSQP = auto()
    NELDER_MEAD = auto()
    POWELL = auto()
    CG = auto()
    NEWTON_CG = auto()

# Protocol classes for structural typing


@runtime_checkable
class HasFit(Protocol[D, R]):
    """Protocol for objects that have a fit method."""

    def fit(self, data: D, **kwargs: Any) -> R:
        """Fit the model to data."""
        ...


@runtime_checkable
class HasFitAsync(Protocol[D, R]):
    """Protocol for objects that have an asynchronous fit method."""

    async def fit_async(self, data: D, **kwargs: Any) -> R:
        """Asynchronously fit the model to data."""
        ...


@runtime_checkable
class HasPredict(Protocol[D]):
    """Protocol for objects that have a predict method."""

    def predict(self, X: D, **kwargs: Any) -> np.ndarray:
        """Generate predictions."""
        ...


@runtime_checkable
class HasForecast(Protocol):
    """Protocol for objects that have a forecast method."""

    def forecast(self, steps: int, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate forecasts with confidence intervals."""
        ...


@runtime_checkable
class HasSimulate(Protocol):
    """Protocol for objects that have a simulate method."""

    def simulate(self, n_periods: int, **kwargs: Any) -> np.ndarray:
        """Simulate data from the model."""
        ...


@runtime_checkable
class HasLogLikelihood(Protocol[D]):
    """Protocol for objects that have a loglikelihood method."""

    def loglikelihood(self, data: D, **kwargs: Any) -> float:
        """Compute the log-likelihood of data."""
        ...


@runtime_checkable
class HasSummary(Protocol):
    """Protocol for objects that have a summary method."""

    def summary(self) -> str:
        """Generate a summary of the object."""
        ...


@runtime_checkable
class HasValidate(Protocol[D]):
    """Protocol for objects that have a validate method."""

    def validate(self, data: D) -> None:
        """Validate data for use with the object."""
        ...


@runtime_checkable
class HasTransform(Protocol[T]):
    """Protocol for objects that have transform and inverse_transform methods."""

    def transform(self, data: T) -> np.ndarray:
        """Transform data to a different representation."""
        ...

    @classmethod
    def inverse_transform(cls, data: np.ndarray) -> T:
        """Transform data back to the original representation."""
        ...


@runtime_checkable
class HasPDF(Protocol):
    """Protocol for objects that have probability density function methods."""

    def pdf(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute the probability density function."""
        ...

    def cdf(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute the cumulative distribution function."""
        ...

    def ppf(self, q: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute the percent point function (inverse of CDF)."""
        ...

    def rvs(self, size: Union[int, Tuple[int, ...]], **kwargs: Any) -> np.ndarray:
        """Generate random variates."""
        ...


# Type aliases for common function signatures
FitFunction = Callable[[D], R]
AsyncFitFunction = Callable[[D], asyncio.coroutine]
PredictFunction = Callable[[np.ndarray], np.ndarray]
ForecastFunction = Callable[[int], Tuple[np.ndarray, np.ndarray, np.ndarray]]
SimulateFunction = Callable[[int], np.ndarray]
LogLikelihoodFunction = Callable[[np.ndarray], float]
ValidationFunction = Callable[[Any], None]
TransformFunction = Callable[[Any], np.ndarray]
InverseTransformFunction = Callable[[np.ndarray], Any]
PDFFunction = Callable[[np.ndarray], np.ndarray]
CDFFunction = Callable[[np.ndarray], np.ndarray]
PPFFunction = Callable[[np.ndarray], np.ndarray]
RVSFunction = Callable[[Union[int, Tuple[int, ...]]], np.ndarray]

# Type aliases for common data structures
ModelParameters = Union[np.ndarray, Dict[str, Any], "ParameterBase"]
ModelResults = Union[Dict[str, Any], "ModelResult"]
DataContainer = Union[np.ndarray, pd.Series, pd.DataFrame, Tuple[np.ndarray, ...]]

# Type aliases for high-frequency data
HighFrequencyData = Tuple[np.ndarray, np.ndarray]  # (prices, times)
RealizedMeasure = np.ndarray  # Realized volatility measure
SamplingScheme = Literal("calendar", "business", "tick", "fixed")

# Type aliases for bootstrap methods
BootstrapIndices = np.ndarray  # Bootstrap indices (n_bootstraps x data_length)
BootstrapSamples = np.ndarray  # Bootstrap samples
BootstrapStatistic = Callable[[np.ndarray], Union[float, np.ndarray]]

# Type aliases for cross-sectional analysis
CrossSectionalData = Tuple[np.ndarray, np.ndarray]  # (y, X)
RegressionResults = Dict[str, Any]
PCAResults = Dict[str, Any]

# Type aliases for UI components
UICallback = Callable[[], None]
UIEventHandler = Callable[[Any], None]
UIUpdateFunction = Callable[[Any], None]

# Type aliases for asynchronous operations
AsyncTask = asyncio.Task
AsyncResult = asyncio.Future
AsyncCallback = Callable[[], asyncio.coroutine]
AsyncEventHandler = Callable[[Any], asyncio.coroutine]

# Forward references for circular dependencies
# These are used when type hints reference classes that are defined later
ParameterBase = Any  # Will be defined in parameters.py
ModelResult = Any  # Will be defined in results.py

# Type aliases for error handling
ErrorHandler = Callable[[Exception], None]
AsyncErrorHandler = Callable[[Exception], asyncio.coroutine]
ValidationError = Union[ValueError, TypeError]
OptimizationError = Exception
NumericalError = Exception

# Type aliases for file operations
FileReader = Callable[[FilePath], Any]
FileWriter = Callable[[Any, FilePath], None]
DataLoader = Callable[[FilePath], DataContainer]
DataSaver = Callable[[DataContainer, FilePath], None]

# Type aliases for configuration
ConfigDict = Dict[str, Any]
ConfigPath = Union[str, Path]
ConfigLoader = Callable[[ConfigPath], ConfigDict]
ConfigSaver = Callable[[ConfigDict, ConfigPath], None]

# Type aliases for logging
LogLevel = Literal("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
LogFunction = Callable[[str], None]
LogHandler = Callable[[str, LogLevel], None]

# Type aliases for testing
TestFunction = Callable[[np.ndarray], TestResult]
TestStatistic = Callable[[np.ndarray], float]
CriticalValueFunction = Callable[[float], float]
PValueFunction = Callable[[float], float]

# Type aliases for documentation
DocString = str
ExampleCode = str
ReferenceDoc = str

# Type aliases for performance monitoring
TimingFunction = Callable[[Callable], Callable]
MemoryUsageFunction = Callable[[Callable], Callable]
PerformanceMetrics = Dict[str, float]

# Type aliases for parallel processing
ParallelFunction = Callable[[Callable, List[Any]], List[Any]]
ParallelMap = Callable[[Callable, List[Any]], List[Any]]
ParallelReduce = Callable[[Callable, List[Any], Any], Any]

# Type aliases for model selection
ModelSelectionCriterion = Callable[[float, int, int], float]
ModelComparisonFunction = Callable[[List[ModelResults]], List[int]]
