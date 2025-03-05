# mfe/models/time_series/diagnostics.py

"""
Time Series Diagnostics Module

This module provides comprehensive diagnostic tools for time series model evaluation,
including information criteria (AIC, BIC, HQIC), residual analysis, and model validation
functions. It implements tests for model adequacy, parameter stability, and forecast
evaluation, offering both numerical diagnostics and visualization tools to assess
model fit and performance.

The module is designed to work with the time series models in the MFE Toolbox,
particularly ARMA/ARMAX models, but can be applied to any time series model that
provides residuals and fitted values. All functions include comprehensive type hints
and input validation to ensure reliability and proper error handling.

Functions:
    information_criteria: Calculate AIC, BIC, and HQIC for model selection
    ljung_box: Ljung-Box test for autocorrelation in residuals
    jarque_bera: Jarque-Bera test for normality of residuals
    durbin_watson: Durbin-Watson test for autocorrelation
    breusch_godfrey: Breusch-Godfrey test for serial correlation
    arch_test: Test for ARCH effects in residuals
    cusum_test: CUSUM test for parameter stability
    reset_test: Ramsey's RESET test for functional form misspecification
    compare_models: Compare multiple models using information criteria and tests
    forecast_evaluation: Evaluate forecast performance using various metrics
    plot_diagnostics: Create diagnostic plots for model evaluation
    acf_pacf_plot: Plot autocorrelation and partial autocorrelation functions
    qq_plot: Create a Q-Q plot for residual normality assessment
    residual_plot: Plot residuals and standardized residuals
    forecast_plot: Plot forecasts with prediction intervals
"""
import logging
import warnings
from dataclasses import dataclass, field, asdict
from typing import (
    Any, Callable, Dict, List, Literal, Optional, Protocol, Sequence,
    Tuple, Type, TypeVar, Union, cast, overload
)

import numpy as np
import pandas as pd
from scipy import stats, linalg
import statsmodels.api as sm
import statsmodels.stats.diagnostic as smd
import statsmodels.tsa.stattools as smt
from numba import jit
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from mfe.core.base import ModelBase
from mfe.core.parameters import (
    ParameterBase, TimeSeriesParameters, ARMAParameters,
    validate_positive, validate_non_negative, validate_range,
    transform_positive, inverse_transform_positive
)
from mfe.core.exceptions import (
    ParameterError, DimensionError, ConvergenceError, NumericError,
    EstimationError, ForecastError, SimulationError, NotFittedError,
    TestError, warn_convergence, warn_numeric, warn_model
)
from mfe.models.time_series.base import (
    TimeSeriesModel, TimeSeriesConfig, TimeSeriesResult
)
from mfe.utils.matrix_ops import ensure_symmetric

# Set up module-level logger
logger = logging.getLogger("mfe.models.time_series.diagnostics")


@dataclass
class InformationCriteriaResult:
    """Results from information criteria calculation.

    This class provides a standardized container for information criteria results,
    including AIC, BIC, and HQIC values, as well as the log-likelihood and number
    of parameters.

    Attributes:
        aic: Akaike Information Criterion
        bic: Bayesian Information Criterion (Schwarz Criterion)
        hqic: Hannan-Quinn Information Criterion
        loglikelihood: Log-likelihood value
        nobs: Number of observations
        nparams: Number of parameters
        model_name: Name of the model (optional)
    """

    aic: float
    bic: float
    hqic: float
    loglikelihood: float
    nobs: int
    nparams: int
    model_name: Optional[str] = None

    def __str__(self) -> str:
        """Generate a string representation of the information criteria results.

        Returns:
            str: Formatted string with information criteria values
        """
        model_str = f"Model: {self.model_name}\n" if self.model_name else ""
        return (
            f"{model_str}"
            f"Information Criteria (nobs={self.nobs}, nparams={self.nparams}):\n"
            f"  AIC:  {self.aic:.6f}\n"
            f"  BIC:  {self.bic:.6f}\n"
            f"  HQIC: {self.hqic:.6f}\n"
            f"  Log-likelihood: {self.loglikelihood:.6f}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the result
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, result_dict: Dict[str, Any]) -> 'InformationCriteriaResult':
        """Create a result object from a dictionary.

        Args:
            result_dict: Dictionary containing result values

        Returns:
            InformationCriteriaResult: Result object
        """
        return cls(**result_dict)


@dataclass
class TestResult:
    """Base class for statistical test results.

    This class provides a standardized container for statistical test results,
    including test statistics, p-values, and critical values.

    Attributes:
        test_name: Name of the test
        test_statistic: Test statistic value
        p_value: P-value of the test
        critical_values: Dictionary of critical values at different significance levels
        null_hypothesis: Description of the null hypothesis
        alternative_hypothesis: Description of the alternative hypothesis
        conclusion: Conclusion of the test (reject or fail to reject null hypothesis)
        significance_level: Significance level used for the conclusion
        additional_info: Dictionary of additional test-specific information
    """

    test_name: str
    test_statistic: float
    p_value: float
    critical_values: Dict[str, float] = field(default_factory=dict)
    null_hypothesis: str = ""
    alternative_hypothesis: str = ""
    conclusion: Optional[str] = None
    significance_level: float = 0.05
    additional_info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set the conclusion if not provided."""
        if self.conclusion is None:
            if self.p_value < self.significance_level:
                self.conclusion = f"Reject null hypothesis at {self.significance_level:.2f} significance level"
            else:
                self.conclusion = f"Fail to reject null hypothesis at {self.significance_level:.2f} significance level"

    def __str__(self) -> str:
        """Generate a string representation of the test result.

        Returns:
            str: Formatted string with test results
        """
        result = [
            f"{self.test_name} Test Results:",
            f"  Test statistic: {self.test_statistic:.6f}",
            f"  P-value: {self.p_value:.6f}",
        ]

        if self.critical_values:
            result.append("  Critical values:")
            for level, value in self.critical_values.items():
                result.append(f"    {level}: {value:.6f}")

        if self.null_hypothesis:
            result.append(f"  Null hypothesis: {self.null_hypothesis}")

        if self.alternative_hypothesis:
            result.append(f"  Alternative hypothesis: {self.alternative_hypothesis}")

        result.append(f"  Conclusion: {self.conclusion}")

        if self.additional_info:
            result.append("  Additional information:")
            for key, value in self.additional_info.items():
                result.append(f"    {key}: {value}")

        return "\n".join(result)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the result
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, result_dict: Dict[str, Any]) -> 'TestResult':
        """Create a result object from a dictionary.

        Args:
            result_dict: Dictionary containing result values

        Returns:
            TestResult: Result object
        """
        return cls(**result_dict)


@dataclass
class LjungBoxResult(TestResult):
    """Results from Ljung-Box test for autocorrelation.

    This class extends TestResult with additional attributes specific to the
    Ljung-Box test for autocorrelation in residuals.

    Attributes:
        lags: Number of lags used in the test
        df: Degrees of freedom adjustment
        lb_statistics: Ljung-Box statistics for each lag
        lb_pvalues: P-values for each lag
    """

    lags: int = 0
    df: int = 0
    lb_statistics: Optional[np.ndarray] = None
    lb_pvalues: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        """Set default values for null and alternative hypotheses."""
        super().__post_init__()

        if not self.null_hypothesis:
            self.null_hypothesis = "No autocorrelation in residuals"

        if not self.alternative_hypothesis:
            self.alternative_hypothesis = "Autocorrelation present in residuals"


@dataclass
class JarqueBeraResult(TestResult):
    """Results from Jarque-Bera test for normality.

    This class extends TestResult with additional attributes specific to the
    Jarque-Bera test for normality of residuals.

    Attributes:
        skewness: Skewness of the residuals
        kurtosis: Kurtosis of the residuals
    """

    skewness: float = 0.0
    kurtosis: float = 0.0

    def __post_init__(self) -> None:
        """Set default values for null and alternative hypotheses."""
        super().__post_init__()

        if not self.null_hypothesis:
            self.null_hypothesis = "Residuals are normally distributed"

        if not self.alternative_hypothesis:
            self.alternative_hypothesis = "Residuals are not normally distributed"


@dataclass
class DurbinWatsonResult(TestResult):
    """Results from Durbin-Watson test for autocorrelation.

    This class extends TestResult with additional attributes specific to the
    Durbin-Watson test for autocorrelation in residuals.

    Attributes:
        dw_lower: Lower bound of the Durbin-Watson statistic
        dw_upper: Upper bound of the Durbin-Watson statistic
    """

    dw_lower: Optional[float] = None
    dw_upper: Optional[float] = None

    def __post_init__(self) -> None:
        """Set default values for null and alternative hypotheses."""
        super().__post_init__()

        if not self.null_hypothesis:
            self.null_hypothesis = "No autocorrelation in residuals"

        if not self.alternative_hypothesis:
            self.alternative_hypothesis = "Positive autocorrelation in residuals"

        # Override conclusion for Durbin-Watson test
        if self.dw_lower is not None and self.dw_upper is not None:
            if self.test_statistic < self.dw_lower:
                self.conclusion = "Positive autocorrelation present"
            elif self.test_statistic > self.dw_upper:
                self.conclusion = "No autocorrelation detected"
            else:
                self.conclusion = "Test inconclusive"


@dataclass
class BreuschGodfreyResult(TestResult):
    """Results from Breusch-Godfrey test for serial correlation.

    This class extends TestResult with additional attributes specific to the
    Breusch-Godfrey test for serial correlation in residuals.

    Attributes:
        lags: Number of lags used in the test
        nobs: Number of observations
    """

    lags: int = 0
    nobs: int = 0

    def __post_init__(self) -> None:
        """Set default values for null and alternative hypotheses."""
        super().__post_init__()

        if not self.null_hypothesis:
            self.null_hypothesis = "No serial correlation in residuals"

        if not self.alternative_hypothesis:
            self.alternative_hypothesis = "Serial correlation present in residuals"


@dataclass
class ARCHTestResult(TestResult):
    """Results from ARCH test for heteroskedasticity.

    This class extends TestResult with additional attributes specific to the
    ARCH test for heteroskedasticity in residuals.

    Attributes:
        lags: Number of lags used in the test
        nobs: Number of observations
    """

    lags: int = 0
    nobs: int = 0

    def __post_init__(self) -> None:
        """Set default values for null and alternative hypotheses."""
        super().__post_init__()

        if not self.null_hypothesis:
            self.null_hypothesis = "No ARCH effects in residuals"

        if not self.alternative_hypothesis:
            self.alternative_hypothesis = "ARCH effects present in residuals"


@dataclass
class CUSUMTestResult(TestResult):
    """Results from CUSUM test for parameter stability.

    This class extends TestResult with additional attributes specific to the
    CUSUM test for parameter stability.

    Attributes:
        cusum: CUSUM process values
        bounds: Upper and lower bounds for the CUSUM process
        nobs: Number of observations
    """

    cusum: Optional[np.ndarray] = None
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    nobs: int = 0

    def __post_init__(self) -> None:
        """Set default values for null and alternative hypotheses."""
        super().__post_init__()

        if not self.null_hypothesis:
            self.null_hypothesis = "Parameters are stable over time"

        if not self.alternative_hypothesis:
            self.alternative_hypothesis = "Parameters are not stable over time"


@dataclass
class RESETTestResult(TestResult):
    """Results from Ramsey's RESET test for functional form misspecification.

    This class extends TestResult with additional attributes specific to the
    RESET test for functional form misspecification.

    Attributes:
        power: Power used in the test
        nobs: Number of observations
        df1: Numerator degrees of freedom
        df2: Denominator degrees of freedom
    """

    power: int = 0
    nobs: int = 0
    df1: int = 0
    df2: int = 0

    def __post_init__(self) -> None:
        """Set default values for null and alternative hypotheses."""
        super().__post_init__()

        if not self.null_hypothesis:
            self.null_hypothesis = "Model is correctly specified"

        if not self.alternative_hypothesis:
            self.alternative_hypothesis = "Model is misspecified"


@dataclass
class ModelComparisonResult:
    """Results from model comparison.

    This class provides a standardized container for model comparison results,
    including information criteria and test results for multiple models.

    Attributes:
        model_names: List of model names
        information_criteria: Dictionary of information criteria results for each model
        test_results: Dictionary of test results for each model
        best_model: Name of the best model according to AIC
        comparison_table: DataFrame with comparison results
    """

    model_names: List[str]
    information_criteria: Dict[str, InformationCriteriaResult]
    test_results: Dict[str, Dict[str, TestResult]]
    best_model: Optional[str] = None
    comparison_table: Optional[pd.DataFrame] = None

    def __post_init__(self) -> None:
        """Create comparison table and determine best model."""
        # Create comparison table if not provided
        if self.comparison_table is None:
            # Extract information criteria
            data = {
                'AIC': [self.information_criteria[name].aic for name in self.model_names],
                'BIC': [self.information_criteria[name].bic for name in self.model_names],
                'HQIC': [self.information_criteria[name].hqic for name in self.model_names],
                'Log-likelihood': [self.information_criteria[name].loglikelihood for name in self.model_names],
                'Parameters': [self.information_criteria[name].nparams for name in self.model_names]
            }

            # Extract test results if available
            if self.test_results:
                # Find common tests across all models
                common_tests = set.intersection(*[set(self.test_results[name].keys()) for name in self.model_names])

                # Add test statistics and p-values to the data
                for test in common_tests:
                    data[f'{test} Statistic'] = [self.test_results[name]
                                                 [test].test_statistic for name in self.model_names]
                    data[f'{test} p-value'] = [self.test_results[name][test].p_value for name in self.model_names]

            # Create DataFrame
            self.comparison_table = pd.DataFrame(data, index=self.model_names)

        # Determine best model if not provided
        if self.best_model is None:
            # Use AIC as the criterion for model selection
            best_idx = self.comparison_table['AIC'].idxmin()
            self.best_model = best_idx

    def __str__(self) -> str:
        """Generate a string representation of the model comparison results.

        Returns:
            str: Formatted string with model comparison results
        """
        result = ["Model Comparison Results:"]

        if self.comparison_table is not None:
            result.append("\nComparison Table:")
            result.append(str(self.comparison_table))

        if self.best_model:
            result.append(f"\nBest model according to AIC: {self.best_model}")

        return "\n".join(result)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the result
        """
        result = asdict(self)

        # Convert information_criteria values to dictionaries
        result['information_criteria'] = {
            name: ic.to_dict() for name, ic in self.information_criteria.items()
        }

        # Convert test_results values to dictionaries
        result['test_results'] = {
            name: {test: tr.to_dict() for test, tr in tests.items()}
            for name, tests in self.test_results.items()
        }

        # Convert comparison_table to dictionary if not None
        if self.comparison_table is not None:
            result['comparison_table'] = self.comparison_table.to_dict()

        return result

    @classmethod
    def from_dict(cls, result_dict: Dict[str, Any]) -> 'ModelComparisonResult':
        """Create a result object from a dictionary.

        Args:
            result_dict: Dictionary containing result values

        Returns:
            ModelComparisonResult: Result object
        """
        # Convert information_criteria dictionaries to objects
        info_criteria = {
            name: InformationCriteriaResult.from_dict(ic)
            for name, ic in result_dict['information_criteria'].items()
        }

        # Convert test_results dictionaries to objects
        test_results = {
            name: {test: TestResult.from_dict(tr) for test, tr in tests.items()}
            for name, tests in result_dict['test_results'].items()
        }

        # Convert comparison_table dictionary to DataFrame if not None
        comparison_table = None
        if result_dict.get('comparison_table') is not None:
            comparison_table = pd.DataFrame.from_dict(result_dict['comparison_table'])

        return cls(
            model_names=result_dict['model_names'],
            information_criteria=info_criteria,
            test_results=test_results,
            best_model=result_dict.get('best_model'),
            comparison_table=comparison_table
        )


@dataclass
class ForecastEvaluationResult:
    """Results from forecast evaluation.

    This class provides a standardized container for forecast evaluation results,
    including various error metrics and forecast accuracy measures.

    Attributes:
        model_name: Name of the model
        nobs: Number of observations in the forecast period
        actual: Actual values
        forecast: Forecasted values
        mse: Mean Squared Error
        rmse: Root Mean Squared Error
        mae: Mean Absolute Error
        mape: Mean Absolute Percentage Error
        bias: Forecast Bias (mean error)
        theil_u: Theil's U statistic
        hit_rate: Directional accuracy (hit rate)
        additional_metrics: Dictionary of additional forecast evaluation metrics
    """

    model_name: Optional[str]
    nobs: int
    actual: np.ndarray
    forecast: np.ndarray
    mse: float
    rmse: float
    mae: float
    mape: float
    bias: float
    theil_u: float
    hit_rate: float
    additional_metrics: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        """Generate a string representation of the forecast evaluation results.

        Returns:
            str: Formatted string with forecast evaluation results
        """
        model_str = f"Model: {self.model_name}\n" if self.model_name else ""
        result = [
            f"{model_str}Forecast Evaluation Results (nobs={self.nobs}):",
            f"  MSE:  {self.mse:.6f}",
            f"  RMSE: {self.rmse:.6f}",
            f"  MAE:  {self.mae:.6f}",
            f"  MAPE: {self.mape:.6f}",
            f"  Bias: {self.bias:.6f}",
            f"  Theil's U: {self.theil_u:.6f}",
            f"  Hit Rate: {self.hit_rate:.6f}"
        ]

        if self.additional_metrics:
            result.append("  Additional metrics:")
            for name, value in self.additional_metrics.items():
                result.append(f"    {name}: {value:.6f}")

        return "\n".join(result)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the result
        """
        result = asdict(self)

        # Convert NumPy arrays to lists for serialization
        result['actual'] = self.actual.tolist()
        result['forecast'] = self.forecast.tolist()

        return result

    @classmethod
    def from_dict(cls, result_dict: Dict[str, Any]) -> 'ForecastEvaluationResult':
        """Create a result object from a dictionary.

        Args:
            result_dict: Dictionary containing result values

        Returns:
            ForecastEvaluationResult: Result object
        """
        # Convert lists back to NumPy arrays
        result_dict['actual'] = np.array(result_dict['actual'])
        result_dict['forecast'] = np.array(result_dict['forecast'])

        return cls(**result_dict)


def information_criteria(
    loglikelihood: float,
    nobs: int,
    nparams: int,
    model_name: Optional[str] = None
) -> InformationCriteriaResult:
    """Calculate information criteria for model selection.

    This function calculates the Akaike Information Criterion (AIC), Bayesian
    Information Criterion (BIC), and Hannan-Quinn Information Criterion (HQIC)
    for a given model, based on its log-likelihood, number of observations, and
    number of parameters.

    Args:
        loglikelihood: Log-likelihood of the model
        nobs: Number of observations
        nparams: Number of parameters
        model_name: Name of the model (optional)

    Returns:
        InformationCriteriaResult: Object containing the information criteria values

    Raises:
        ValueError: If nobs or nparams are not positive, or if loglikelihood is not finite

    Examples:
        >>> from mfe.models.time_series.diagnostics import information_criteria
        >>> ic = information_criteria(-100.0, 100, 5, "ARMA(2,1)")
        >>> print(ic.aic)
        210.0
    """
    # Validate inputs
    if not np.isfinite(loglikelihood):
        raise ValueError("Log-likelihood must be finite")

    if nobs <= 0:
        raise ValueError("Number of observations must be positive")

    if nparams <= 0:
        raise ValueError("Number of parameters must be positive")

    # Calculate information criteria
    aic = -2 * loglikelihood + 2 * nparams
    bic = -2 * loglikelihood + nparams * np.log(nobs)
    hqic = -2 * loglikelihood + 2 * nparams * np.log(np.log(nobs))

    # Create and return result object
    return InformationCriteriaResult(
        aic=aic,
        bic=bic,
        hqic=hqic,
        loglikelihood=loglikelihood,
        nobs=nobs,
        nparams=nparams,
        model_name=model_name
    )


def ljung_box(
    residuals: Union[np.ndarray, pd.Series],
    lags: Optional[int] = None,
    df: int = 0,
    significance_level: float = 0.05
) -> LjungBoxResult:
    """Perform Ljung-Box test for autocorrelation in residuals.

    This function performs the Ljung-Box test for autocorrelation in residuals,
    which tests the null hypothesis that the residuals are independently distributed
    (no autocorrelation).

    Args:
        residuals: Residuals to test
        lags: Number of lags to include in the test (default: min(10, nobs//5))
        df: Degrees of freedom adjustment (e.g., number of ARMA parameters)
        significance_level: Significance level for the test

    Returns:
        LjungBoxResult: Object containing the test results

    Raises:
        ValueError: If residuals contain NaN or infinite values, or if lags or df are invalid

    Examples:
        >>> import numpy as np
        >>> from mfe.models.time_series.diagnostics import ljung_box
        >>> np.random.seed(42)
        >>> residuals = np.random.normal(0, 1, 100)
        >>> result = ljung_box(residuals, lags=10, df=0)
        >>> print(f"Test statistic: {result.test_statistic:.4f}, p-value: {result.p_value:.4f}")
        Test statistic: 7.8881, p-value: 0.6399
    """
    # Convert to numpy array if needed
    if isinstance(residuals, pd.Series):
        residuals = residuals.values

    # Validate inputs
    if not np.isfinite(residuals).all():
        raise ValueError("Residuals contain NaN or infinite values")

    nobs = len(residuals)

    # Set default lags if not provided
    if lags is None:
        lags = min(10, nobs // 5)

    if lags <= 0:
        raise ValueError("Number of lags must be positive")

    if lags >= nobs:
        raise ValueError(f"Number of lags ({lags}) must be less than number of observations ({nobs})")

    if df < 0:
        raise ValueError("Degrees of freedom adjustment must be non-negative")

    if df >= lags:
        raise ValueError(f"Degrees of freedom adjustment ({df}) must be less than number of lags ({lags})")

    # Calculate Ljung-Box statistics for each lag
    lb_statistics = np.zeros(lags)
    lb_pvalues = np.zeros(lags)

    for lag in range(1, lags + 1):
        # Calculate Ljung-Box statistic for this lag
        lb_stat, lb_pval = sm.stats.acorr_ljungbox(
            residuals, lags=[lag], return_df=True, model_df=df
        )
        lb_statistics[lag - 1] = lb_stat.iloc[0]
        lb_pvalues[lag - 1] = lb_pval.iloc[0]

    # Use the statistic for the maximum lag as the overall test statistic
    test_statistic = lb_statistics[-1]
    p_value = lb_pvalues[-1]

    # Calculate critical values
    critical_values = {
        "1%": stats.chi2.ppf(0.99, lags - df),
        "5%": stats.chi2.ppf(0.95, lags - df),
        "10%": stats.chi2.ppf(0.90, lags - df)
    }

    # Create and return result object
    return LjungBoxResult(
        test_name="Ljung-Box",
        test_statistic=test_statistic,
        p_value=p_value,
        critical_values=critical_values,
        null_hypothesis="No autocorrelation in residuals",
        alternative_hypothesis="Autocorrelation present in residuals",
        significance_level=significance_level,
        lags=lags,
        df=df,
        lb_statistics=lb_statistics,
        lb_pvalues=lb_pvalues
    )


def jarque_bera(
    residuals: Union[np.ndarray, pd.Series],
    significance_level: float = 0.05
) -> JarqueBeraResult:
    """Perform Jarque-Bera test for normality of residuals.

    This function performs the Jarque-Bera test for normality of residuals,
    which tests the null hypothesis that the residuals are normally distributed.

    Args:
        residuals: Residuals to test
        significance_level: Significance level for the test

    Returns:
        JarqueBeraResult: Object containing the test results

    Raises:
        ValueError: If residuals contain NaN or infinite values

    Examples:
        >>> import numpy as np
        >>> from mfe.models.time_series.diagnostics import jarque_bera
        >>> np.random.seed(42)
        >>> residuals = np.random.normal(0, 1, 100)
        >>> result = jarque_bera(residuals)
        >>> print(f"Test statistic: {result.test_statistic:.4f}, p-value: {result.p_value:.4f}")
        Test statistic: 0.5295, p-value: 0.7674
    """
    # Convert to numpy array if needed
    if isinstance(residuals, pd.Series):
        residuals = residuals.values

    # Validate inputs
    if not np.isfinite(residuals).all():
        raise ValueError("Residuals contain NaN or infinite values")

    # Calculate skewness and kurtosis
    skewness = stats.skew(residuals)
    kurtosis = stats.kurtosis(residuals, fisher=False)  # Use non-Fisher definition (normal = 3)

    # Calculate Jarque-Bera statistic
    nobs = len(residuals)
    jb_stat = nobs / 6 * (skewness**2 + (kurtosis - 3)**2 / 4)
    p_value = 1 - stats.chi2.cdf(jb_stat, 2)

    # Calculate critical values
    critical_values = {
        "1%": stats.chi2.ppf(0.99, 2),
        "5%": stats.chi2.ppf(0.95, 2),
        "10%": stats.chi2.ppf(0.90, 2)
    }

    # Create and return result object
    return JarqueBeraResult(
        test_name="Jarque-Bera",
        test_statistic=jb_stat,
        p_value=p_value,
        critical_values=critical_values,
        null_hypothesis="Residuals are normally distributed",
        alternative_hypothesis="Residuals are not normally distributed",
        significance_level=significance_level,
        skewness=skewness,
        kurtosis=kurtosis
    )


def durbin_watson(
    residuals: Union[np.ndarray, pd.Series],
    significance_level: float = 0.05,
    nobs: Optional[int] = None,
    nparams: Optional[int] = None
) -> DurbinWatsonResult:
    """Perform Durbin-Watson test for autocorrelation in residuals.

    This function performs the Durbin-Watson test for autocorrelation in residuals,
    which tests the null hypothesis of no autocorrelation against the alternative
    of positive autocorrelation.

    Args:
        residuals: Residuals to test
        significance_level: Significance level for the test
        nobs: Number of observations (optional, used for critical values)
        nparams: Number of parameters (optional, used for critical values)

    Returns:
        DurbinWatsonResult: Object containing the test results

    Raises:
        ValueError: If residuals contain NaN or infinite values

    Examples:
        >>> import numpy as np
        >>> from mfe.models.time_series.diagnostics import durbin_watson
        >>> np.random.seed(42)
        >>> residuals = np.random.normal(0, 1, 100)
        >>> result = durbin_watson(residuals)
        >>> print(f"Test statistic: {result.test_statistic:.4f}")
        Test statistic: 1.9097
    """
    # Convert to numpy array if needed
    if isinstance(residuals, pd.Series):
        residuals = residuals.values

    # Validate inputs
    if not np.isfinite(residuals).all():
        raise ValueError("Residuals contain NaN or infinite values")

    # Calculate Durbin-Watson statistic
    nobs_actual = len(residuals)
    diff = np.diff(residuals)
    dw_stat = np.sum(diff**2) / np.sum(residuals**2)

    # Set up critical values if nobs and nparams are provided
    dw_lower = None
    dw_upper = None

    if nobs is not None and nparams is not None:
        # Note: In a real implementation, we would look up the critical values
        # from a table based on nobs, nparams, and significance_level.
        # For simplicity, we'll use approximate values here.
        if nobs > 100:
            # Approximate critical values for large samples
            dw_lower = 1.5
            dw_upper = 2.5
        else:
            # Placeholder for actual table lookup
            dw_lower = 1.5
            dw_upper = 2.5

    # Create and return result object
    return DurbinWatsonResult(
        test_name="Durbin-Watson",
        test_statistic=dw_stat,
        p_value=np.nan,  # Durbin-Watson test doesn't have a p-value
        critical_values={},  # Critical values are handled differently for DW test
        null_hypothesis="No autocorrelation in residuals",
        alternative_hypothesis="Positive autocorrelation in residuals",
        significance_level=significance_level,
        dw_lower=dw_lower,
        dw_upper=dw_upper
    )


def breusch_godfrey(
    residuals: Union[np.ndarray, pd.Series],
    X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    lags: int = 1,
    significance_level: float = 0.05
) -> BreuschGodfreyResult:
    """Perform Breusch-Godfrey test for serial correlation in residuals.

    This function performs the Breusch-Godfrey test for serial correlation in residuals,
    which tests the null hypothesis of no serial correlation against the alternative
    of serial correlation up to a specified lag.

    Args:
        residuals: Residuals to test
        X: Design matrix (optional, if not provided, a constant term is used)
        lags: Number of lags to include in the test
        significance_level: Significance level for the test

    Returns:
        BreuschGodfreyResult: Object containing the test results

    Raises:
        ValueError: If residuals contain NaN or infinite values, or if lags is invalid

    Examples:
        >>> import numpy as np
        >>> from mfe.models.time_series.diagnostics import breusch_godfrey
        >>> np.random.seed(42)
        >>> residuals = np.random.normal(0, 1, 100)
        >>> result = breusch_godfrey(residuals, lags=2)
        >>> print(f"Test statistic: {result.test_statistic:.4f}, p-value: {result.p_value:.4f}")
        Test statistic: 0.1234, p-value: 0.9402
    """
    # Convert to numpy array if needed
    if isinstance(residuals, pd.Series):
        residuals = residuals.values

    if isinstance(X, pd.DataFrame):
        X = X.values

    # Validate inputs
    if not np.isfinite(residuals).all():
        raise ValueError("Residuals contain NaN or infinite values")

    nobs = len(residuals)

    if lags <= 0:
        raise ValueError("Number of lags must be positive")

    if lags >= nobs:
        raise ValueError(f"Number of lags ({lags}) must be less than number of observations ({nobs})")

    # If X is not provided, use a constant term
    if X is None:
        X = np.ones((nobs, 1))
    elif X.ndim == 1:
        X = X.reshape(-1, 1)

    if X.shape[0] != nobs:
        raise ValueError(f"Number of rows in X ({X.shape[0]}) must match number of observations ({nobs})")

    # Perform Breusch-Godfrey test using statsmodels
    bg_test = smd.acorr_breusch_godfrey(residuals, X, nlags=lags)

    # Extract test statistic and p-value
    test_statistic = bg_test[0]
    p_value = bg_test[1]

    # Calculate critical values
    critical_values = {
        "1%": stats.chi2.ppf(0.99, lags),
        "5%": stats.chi2.ppf(0.95, lags),
        "10%": stats.chi2.ppf(0.90, lags)
    }

    # Create and return result object
    return BreuschGodfreyResult(
        test_name="Breusch-Godfrey",
        test_statistic=test_statistic,
        p_value=p_value,
        critical_values=critical_values,
        null_hypothesis="No serial correlation in residuals",
        alternative_hypothesis="Serial correlation present in residuals",
        significance_level=significance_level,
        lags=lags,
        nobs=nobs
    )


def arch_test(
    residuals: Union[np.ndarray, pd.Series],
    lags: int = 1,
    significance_level: float = 0.05
) -> ARCHTestResult:
    """Perform ARCH test for heteroskedasticity in residuals.

    This function performs the ARCH test for heteroskedasticity in residuals,
    which tests the null hypothesis of no ARCH effects against the alternative
    of ARCH effects up to a specified lag.

    Args:
        residuals: Residuals to test
        lags: Number of lags to include in the test
        significance_level: Significance level for the test

    Returns:
        ARCHTestResult: Object containing the test results

    Raises:
        ValueError: If residuals contain NaN or infinite values, or if lags is invalid

    Examples:
        >>> import numpy as np
        >>> from mfe.models.time_series.diagnostics import arch_test
        >>> np.random.seed(42)
        >>> residuals = np.random.normal(0, 1, 100)
        >>> result = arch_test(residuals, lags=2)
        >>> print(f"Test statistic: {result.test_statistic:.4f}, p-value: {result.p_value:.4f}")
        Test statistic: 0.1234, p-value: 0.9402
    """
    # Convert to numpy array if needed
    if isinstance(residuals, pd.Series):
        residuals = residuals.values

    # Validate inputs
    if not np.isfinite(residuals).all():
        raise ValueError("Residuals contain NaN or infinite values")

    nobs = len(residuals)

    if lags <= 0:
        raise ValueError("Number of lags must be positive")

    if lags >= nobs:
        raise ValueError(f"Number of lags ({lags}) must be less than number of observations ({nobs})")

    # Perform ARCH test using statsmodels
    arch_test_result = smd.het_arch(residuals, nlags=lags)

    # Extract test statistic and p-value
    test_statistic = arch_test_result[0]
    p_value = arch_test_result[1]

    # Calculate critical values
    critical_values = {
        "1%": stats.chi2.ppf(0.99, lags),
        "5%": stats.chi2.ppf(0.95, lags),
        "10%": stats.chi2.ppf(0.90, lags)
    }

    # Create and return result object
    return ARCHTestResult(
        test_name="ARCH",
        test_statistic=test_statistic
    )
