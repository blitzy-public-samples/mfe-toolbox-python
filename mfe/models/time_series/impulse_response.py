# mfe/models/time_series/impulse_response.py

"""
Impulse Response Analysis for Vector Autoregression (VAR) Models.

This module implements impulse response analysis for vector autoregression (VAR) models,
computing dynamic responses to system shocks with confidence intervals. It provides
tools for orthogonalized and structural impulse responses, variance decomposition,
and historical decomposition.

The implementation leverages NumPy for efficient matrix operations, with performance-critical
sections accelerated using Numba's JIT compilation. Bootstrap-based confidence intervals
are supported with various sampling methods, and asynchronous computation is available
for computationally intensive operations.

Classes:
    ImpulseResponseParameters: Parameter container for impulse response analysis
    ImpulseResponseResult: Results container for impulse response analysis
    ImpulseResponse: Class for computing and analyzing impulse responses

Functions:
    compute_ma_coefficients: Compute moving average coefficient matrices
    compute_orthogonalized_irf: Compute orthogonalized impulse responses
    compute_generalized_irf: Compute generalized impulse responses
    compute_structural_irf: Compute structural impulse responses
    compute_fevd: Compute forecast error variance decomposition
"""

import asyncio
import logging
import warnings
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union, cast
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg, stats
from numba import jit

from mfe.core.base import ModelBase
from mfe.core.parameters import (
    ParameterBase, TimeSeriesParameters, validate_positive, validate_non_negative,
    validate_range, transform_positive, inverse_transform_positive
)
from mfe.core.exceptions import (
    ParameterError, DimensionError, BootstrapError, NotFittedError
)
from mfe.models.bootstrap.base import BootstrapBase
from mfe.models.bootstrap.block_bootstrap import BlockBootstrap

# Set up module-level logger
logger = logging.getLogger("mfe.models.time_series.impulse_response")

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
    logger.debug("Numba available for impulse response acceleration")
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    HAS_NUMBA = False
    logger.info("Numba not available. Impulse response will use pure NumPy implementations.")


@dataclass
class ImpulseResponseParameters(TimeSeriesParameters):
    """
    Parameters for impulse response analysis.

    This class provides a container for impulse response analysis parameters,
    including the number of periods, identification method, and bootstrap options.

    Attributes:
        periods: Number of periods for impulse response functions
        method: Method for impulse response calculation
            - "orthogonalized": Orthogonalized impulse responses (Cholesky decomposition)
            - "generalized": Generalized impulse responses (Pesaran and Shin)
            - "structural": Structural impulse responses (requires identification)
        identification: Identification method for structural VAR
            - "short": Short-run restrictions (Cholesky decomposition)
            - "long": Long-run restrictions
            - numpy.ndarray: Custom identification matrix
        bootstrap: Whether to compute bootstrap confidence intervals
        bootstrap_type: Type of bootstrap to use
            - "block": Block bootstrap
            - "stationary": Stationary bootstrap
        bootstrap_replications: Number of bootstrap replications
        block_length: Block length for block bootstrap
        confidence_level: Confidence level for intervals (between 0 and 1)
    """

    periods: int = 10
    method: str = "orthogonalized"
    identification: Optional[Union[str, np.ndarray]] = None
    bootstrap: bool = False
    bootstrap_type: str = "block"
    bootstrap_replications: int = 1000
    block_length: int = 4
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        self.validate()

    def validate(self) -> None:
        """
        Validate impulse response parameter constraints.

        Raises:
            ParameterError: If parameter constraints are violated
            DimensionError: If parameter dimensions are inconsistent
        """
        super().validate()

        # Validate periods
        if not isinstance(self.periods, int):
            raise ParameterError(
                "periods must be an integer",
                param_name="periods",
                param_value=self.periods
            )
        if self.periods <= 0:
            raise ParameterError(
                "periods must be positive",
                param_name="periods",
                param_value=self.periods
            )

        # Validate method
        valid_methods = ["orthogonalized", "generalized", "structural"]
        if self.method not in valid_methods:
            raise ParameterError(
                f"method must be one of {valid_methods}",
                param_name="method",
                param_value=self.method
            )

        # Validate identification if method is structural
        if self.method == "structural" and self.identification is None:
            raise ParameterError(
                "identification must be provided for structural impulse responses",
                param_name="identification",
                param_value=self.identification
            )

        if self.identification is not None and isinstance(self.identification, str):
            valid_identifications = ["short", "long"]
            if self.identification not in valid_identifications:
                raise ParameterError(
                    f"identification must be one of {valid_identifications} or a custom matrix",
                    param_name="identification",
                    param_value=self.identification
                )

        # Validate bootstrap parameters if bootstrap is True
        if self.bootstrap:
            # Validate bootstrap_type
            valid_bootstrap_types = ["block", "stationary"]
            if self.bootstrap_type not in valid_bootstrap_types:
                raise ParameterError(
                    f"bootstrap_type must be one of {valid_bootstrap_types}",
                    param_name="bootstrap_type",
                    param_value=self.bootstrap_type
                )

            # Validate bootstrap_replications
            if not isinstance(self.bootstrap_replications, int):
                raise ParameterError(
                    "bootstrap_replications must be an integer",
                    param_name="bootstrap_replications",
                    param_value=self.bootstrap_replications
                )
            if self.bootstrap_replications <= 0:
                raise ParameterError(
                    "bootstrap_replications must be positive",
                    param_name="bootstrap_replications",
                    param_value=self.bootstrap_replications
                )

            # Validate block_length
            if not isinstance(self.block_length, int):
                raise ParameterError(
                    "block_length must be an integer",
                    param_name="block_length",
                    param_value=self.block_length
                )
            if self.block_length <= 0:
                raise ParameterError(
                    "block_length must be positive",
                    param_name="block_length",
                    param_value=self.block_length
                )

            # Validate confidence_level
            if not 0 < self.confidence_level < 1:
                raise ParameterError(
                    "confidence_level must be between 0 and 1",
                    param_name="confidence_level",
                    param_value=self.confidence_level
                )


@dataclass
class ImpulseResponseResult:
    """
    Results container for impulse response analysis.

    This class provides a container for impulse response analysis results,
    including impulse response functions, confidence intervals, and metadata.

    Attributes:
        irf: Impulse response functions (periods x k x k)
        method: Method used for impulse response calculation
        identification: Identification method used (for structural IRFs)
        periods: Number of periods in the impulse response functions
        var_names: Names of the variables in the VAR model
        lower_ci: Lower confidence interval bounds (if bootstrap=True)
        upper_ci: Upper confidence interval bounds (if bootstrap=True)
        bootstrap_irfs: Bootstrap impulse response functions (if bootstrap=True)
        confidence_level: Confidence level for intervals (if bootstrap=True)
        bootstrap_type: Type of bootstrap used (if bootstrap=True)
        bootstrap_replications: Number of bootstrap replications (if bootstrap=True)
        fevd: Forecast error variance decomposition (if computed)
        historical_decomposition: Historical decomposition (if computed)
    """

    irf: np.ndarray
    method: str
    identification: Optional[Union[str, np.ndarray]] = None
    periods: int = 10
    var_names: Optional[List[str]] = None
    lower_ci: Optional[np.ndarray] = None
    upper_ci: Optional[np.ndarray] = None
    bootstrap_irfs: Optional[np.ndarray] = None
    confidence_level: Optional[float] = None
    bootstrap_type: Optional[str] = None
    bootstrap_replications: Optional[int] = None
    fevd: Optional[np.ndarray] = None
    historical_decomposition: Optional[Dict[str, np.ndarray]] = None

    def __post_init__(self) -> None:
        """Initialize derived attributes after initialization."""
        # Validate dimensions
        if self.irf.ndim != 3:
            raise DimensionError(
                "irf must be 3-dimensional (periods x k x k)",
                array_name="irf",
                expected_shape="(periods, k, k)",
                actual_shape=self.irf.shape
            )

        # Set periods if not provided
        if self.periods != self.irf.shape[0]:
            self.periods = self.irf.shape[0]

        # Create default variable names if not provided
        if self.var_names is None:
            k = self.irf.shape[1]
            self.var_names = [f"y{i+1}" for i in range(k)]

        # Validate confidence intervals if provided
        if self.lower_ci is not None and self.upper_ci is not None:
            if self.lower_ci.shape != self.irf.shape:
                raise DimensionError(
                    "lower_ci must have the same shape as irf",
                    array_name="lower_ci",
                    expected_shape=self.irf.shape,
                    actual_shape=self.lower_ci.shape
                )
            if self.upper_ci.shape != self.irf.shape:
                raise DimensionError(
                    "upper_ci must have the same shape as irf",
                    array_name="upper_ci",
                    expected_shape=self.irf.shape,
                    actual_shape=self.upper_ci.shape
                )

    def to_pandas(self) -> Dict[str, pd.DataFrame]:
        """
        Convert impulse response results to Pandas DataFrames.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary of DataFrames containing results
        """
        k = self.irf.shape[1]
        var_names = self.var_names if self.var_names is not None else [f"y{i+1}" for i in range(k)]

        # Create DataFrames for each result component
        results = {}

        # Create IRF DataFrames for each response variable
        for i in range(k):
            # Create DataFrame for IRF
            irf_df = pd.DataFrame(
                self.irf[:, i, :],
                columns=[f"Shock to {var}" for var in var_names]
            )
            irf_df.index.name = "Period"
            results[f"irf_{var_names[i]}"] = irf_df

            # Create DataFrames for confidence intervals if available
            if self.lower_ci is not None and self.upper_ci is not None:
                lower_df = pd.DataFrame(
                    self.lower_ci[:, i, :],
                    columns=[f"Shock to {var} (Lower CI)" for var in var_names]
                )
                lower_df.index.name = "Period"
                results[f"lower_ci_{var_names[i]}"] = lower_df

                upper_df = pd.DataFrame(
                    self.upper_ci[:, i, :],
                    columns=[f"Shock to {var} (Upper CI)" for var in var_names]
                )
                upper_df.index.name = "Period"
                results[f"upper_ci_{var_names[i]}"] = upper_df

        # Create FEVD DataFrame if available
        if self.fevd is not None:
            for i in range(k):
                fevd_df = pd.DataFrame(
                    self.fevd[:, i, :],
                    columns=[f"Contribution from {var}" for var in var_names]
                )
                fevd_df.index.name = "Period"
                results[f"fevd_{var_names[i]}"] = fevd_df

        # Create historical decomposition DataFrames if available
        if self.historical_decomposition is not None:
            # Baseline
            if "baseline" in self.historical_decomposition:
                baseline_df = pd.DataFrame(
                    self.historical_decomposition["baseline"],
                    columns=var_names
                )
                baseline_df.index.name = "Period"
                results["hd_baseline"] = baseline_df

            # Actual data
            if "actual" in self.historical_decomposition:
                actual_df = pd.DataFrame(
                    self.historical_decomposition["actual"],
                    columns=var_names
                )
                actual_df.index.name = "Period"
                results["hd_actual"] = actual_df

            # Shock contributions
            if "shock_contributions" in self.historical_decomposition:
                shock_contrib = self.historical_decomposition["shock_contributions"]
                for i in range(k):
                    contrib_df = pd.DataFrame(
                        shock_contrib[:, i, :],
                        columns=[f"Contribution from {var}" for var in var_names]
                    )
                    contrib_df.index.name = "Period"
                    results[f"hd_contrib_{var_names[i]}"] = contrib_df

        return results

    def plot(
        self,
        impulse: Optional[Union[int, str, List[Union[int, str]]]] = None,
        response: Optional[Union[int, str, List[Union[int, str]]]] = None,
        figsize: Tuple[float, float] = (12, 8),
        include_ci: bool = True,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        **kwargs: Any
    ) -> plt.Figure:
        """
        Plot impulse response functions.

        Args:
            impulse: Index or name of impulse variable(s)
            response: Index or name of response variable(s)
            figsize: Figure size (width, height)
            include_ci: Whether to include confidence intervals
            title: Plot title
            save_path: Path to save the figure
            **kwargs: Additional keyword arguments for plotting

        Returns:
            plt.Figure: Matplotlib figure object

        Raises:
            ValueError: If impulse or response is invalid
        """
        k = self.irf.shape[1]
        var_names = self.var_names if self.var_names is not None else [f"y{i+1}" for i in range(k)]

        # Convert variable names to indices
        def _get_indices(variables: Optional[Union[int, str, List[Union[int, str]]]]) -> List[int]:
            if variables is None:
                return list(range(k))

            if isinstance(variables, (int, str)):
                variables = [variables]

            indices = []
            for var in variables:
                if isinstance(var, int):
                    if var < 0 or var >= k:
                        raise ValueError(f"Variable index {var} out of range [0, {k-1}]")
                    indices.append(var)
                elif isinstance(var, str):
                    if var not in var_names:
                        raise ValueError(f"Variable name '{var}' not found in {var_names}")
                    indices.append(var_names.index(var))
                else:
                    raise TypeError(f"Variable must be int or str, got {type(var)}")

            return indices

        impulse_idx = _get_indices(impulse)
        response_idx = _get_indices(response)

        # Create figure and axes
        n_rows = len(response_idx)
        n_cols = len(impulse_idx)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

        # Set title
        if title is not None:
            fig.suptitle(title, fontsize=16)
        else:
            fig.suptitle(f"Impulse Response Functions ({self.method.capitalize()})", fontsize=16)

        # Plot impulse responses
        for i, resp_idx in enumerate(response_idx):
            for j, imp_idx in enumerate(impulse_idx):
                ax = axes[i, j]

                # Plot IRF
                ax.plot(self.irf[:, resp_idx, imp_idx], 'b-', label='IRF')

                # Plot confidence intervals if available and requested
                if include_ci and self.lower_ci is not None and self.upper_ci is not None:
                    ax.fill_between(
                        range(self.periods),
                        self.lower_ci[:, resp_idx, imp_idx],
                        self.upper_ci[:, resp_idx, imp_idx],
                        color='b', alpha=0.2, label=f'{int(self.confidence_level*100)}% CI'
                    )

                # Add horizontal line at zero
                ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)

                # Set labels
                ax.set_title(f"Response of {var_names[resp_idx]} to {var_names[imp_idx]}")
                ax.set_xlabel("Periods")
                ax.set_ylabel("Response")

                # Add legend if confidence intervals are included
                if include_ci and self.lower_ci is not None and self.upper_ci is not None:
                    ax.legend()

        # Adjust layout
        fig.tight_layout()
        if title is not None:
            fig.subplots_adjust(top=0.9)

        # Save figure if requested
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_fevd(
        self,
        variable: Optional[Union[int, str, List[Union[int, str]]]] = None,
        figsize: Tuple[float, float] = (12, 8),
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        **kwargs: Any
    ) -> plt.Figure:
        """
        Plot forecast error variance decomposition.

        Args:
            variable: Index or name of variable(s) to plot
            figsize: Figure size (width, height)
            title: Plot title
            save_path: Path to save the figure
            **kwargs: Additional keyword arguments for plotting

        Returns:
            plt.Figure: Matplotlib figure object

        Raises:
            ValueError: If variable is invalid or FEVD is not available
        """
        if self.fevd is None:
            raise ValueError("FEVD not available. Compute FEVD first.")

        k = self.irf.shape[1]
        var_names = self.var_names if self.var_names is not None else [f"y{i+1}" for i in range(k)]

        # Convert variable names to indices
        def _get_indices(variables: Optional[Union[int, str, List[Union[int, str]]]]) -> List[int]:
            if variables is None:
                return list(range(k))

            if isinstance(variables, (int, str)):
                variables = [variables]

            indices = []
            for var in variables:
                if isinstance(var, int):
                    if var < 0 or var >= k:
                        raise ValueError(f"Variable index {var} out of range [0, {k-1}]")
                    indices.append(var)
                elif isinstance(var, str):
                    if var not in var_names:
                        raise ValueError(f"Variable name '{var}' not found in {var_names}")
                    indices.append(var_names.index(var))
                else:
                    raise TypeError(f"Variable must be int or str, got {type(var)}")

            return indices

        var_idx = _get_indices(variable)

        # Create figure and axes
        n_vars = len(var_idx)
        fig, axes = plt.subplots(n_vars, 1, figsize=figsize, squeeze=False)

        # Set title
        if title is not None:
            fig.suptitle(title, fontsize=16)
        else:
            fig.suptitle("Forecast Error Variance Decomposition", fontsize=16)

        # Plot FEVD
        for i, idx in enumerate(var_idx):
            ax = axes[i, 0]

            # Create stacked area plot
            x = np.arange(self.periods)
            y = self.fevd[:, idx, :]

            # Ensure the values sum to 1 for each period
            y_normalized = y / np.sum(y, axis=1, keepdims=True)

            # Create stacked area plot
            ax.stackplot(
                x,
                [y_normalized[:, j] for j in range(k)],
                labels=[var_names[j] for j in range(k)],
                alpha=0.7
            )

            # Set labels
            ax.set_title(f"FEVD for {var_names[idx]}")
            ax.set_xlabel("Periods")
            ax.set_ylabel("Proportion of Variance")
            ax.set_ylim(0, 1)

            # Add legend
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # Adjust layout
        fig.tight_layout()
        if title is not None:
            fig.subplots_adjust(top=0.9)

        # Save figure if requested
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_historical_decomposition(
        self,
        variable: Optional[Union[int, str, List[Union[int, str]]]] = None,
        figsize: Tuple[float, float] = (12, 8),
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        **kwargs: Any
    ) -> plt.Figure:
        """
        Plot historical decomposition.

        Args:
            variable: Index or name of variable(s) to plot
            figsize: Figure size (width, height)
            title: Plot title
            save_path: Path to save the figure
            **kwargs: Additional keyword arguments for plotting

        Returns:
            plt.Figure: Matplotlib figure object

        Raises:
            ValueError: If variable is invalid or historical decomposition is not available
        """
        if self.historical_decomposition is None:
            raise ValueError("Historical decomposition not available. Compute historical decomposition first.")

        k = self.irf.shape[1]
        var_names = self.var_names if self.var_names is not None else [f"y{i+1}" for i in range(k)]

        # Convert variable names to indices
        def _get_indices(variables: Optional[Union[int, str, List[Union[int, str]]]]) -> List[int]:
            if variables is None:
                return list(range(k))

            if isinstance(variables, (int, str)):
                variables = [variables]

            indices = []
            for var in variables:
                if isinstance(var, int):
                    if var < 0 or var >= k:
                        raise ValueError(f"Variable index {var} out of range [0, {k-1}]")
                    indices.append(var)
                elif isinstance(var, str):
                    if var not in var_names:
                        raise ValueError(f"Variable name '{var}' not found in {var_names}")
                    indices.append(var_names.index(var))
                else:
                    raise TypeError(f"Variable must be int or str, got {type(var)}")

            return indices

        var_idx = _get_indices(variable)

        # Extract components
        baseline = self.historical_decomposition.get("baseline", None)
        shock_contributions = self.historical_decomposition.get("shock_contributions", None)
        actual = self.historical_decomposition.get("actual", None)

        if baseline is None or shock_contributions is None or actual is None:
            raise ValueError("Missing components in historical decomposition")

        # Create figure and axes
        n_vars = len(var_idx)
        fig, axes = plt.subplots(n_vars, 1, figsize=figsize, squeeze=False)

        # Set title
        if title is not None:
            fig.suptitle(title, fontsize=16)
        else:
            fig.suptitle("Historical Decomposition", fontsize=16)

        # Plot historical decomposition
        for i, idx in enumerate(var_idx):
            ax = axes[i, 0]

            # Plot baseline
            ax.plot(baseline[:, idx], 'k--', label='Baseline')

            # Plot actual data
            ax.plot(actual[:, idx], 'k-', label='Actual')

            # Plot shock contributions
            for j in range(k):
                ax.bar(
                    np.arange(len(actual)),
                    shock_contributions[:, idx, j],
                    bottom=baseline[:, idx],
                    label=f'Shock from {var_names[j]}',
                    alpha=0.7
                )

            # Set labels
            ax.set_title(f"Historical Decomposition for {var_names[idx]}")
            ax.set_xlabel("Periods")
            ax.set_ylabel("Value")

            # Add legend
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # Adjust layout
        fig.tight_layout()
        if title is not None:
            fig.subplots_adjust(top=0.9)

        # Save figure if requested
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def summary(self) -> str:
        """
        Generate a text summary of the impulse response analysis.

        Returns:
            str: A formatted string containing the analysis summary
        """
        k = self.irf.shape[1]
        var_names = self.var_names if self.var_names is not None else [f"y{i+1}" for i in range(k)]

        header = "Impulse Response Analysis\n"
        header += "=" * len(header) + "\n\n"

        # Add analysis information
        info = f"Method: {self.method.capitalize()}\n"
        if self.identification is not None:
            if isinstance(self.identification, str):
                info += f"Identification: {self.identification.capitalize()}\n"
            else:
                info += "Identification: Custom matrix\n"

        info += f"Number of periods: {self.periods}\n"
        info += f"Number of variables: {k}\n"

        if self.bootstrap_irfs is not None:
            info += f"Bootstrap: {self.bootstrap_type.capitalize()}\n"
            info += f"Bootstrap replications: {self.bootstrap_replications}\n"
            info += f"Confidence level: {self.confidence_level:.2f}\n"

        info += "\n"

        # Add variable names
        var_info = "Variables:\n"
        for i, name in enumerate(var_names):
            var_info += f"  {i+1}: {name}\n"

        var_info += "\n"

        # Add impulse response summary
        irf_summary = "Impulse Response Summary:\n"
        irf_summary += "-" * 80 + "\n"

        # For each impulse-response pair, show peak response and timing
        for i in range(k):
            for j in range(k):
                irf_ij = self.irf[:, i, j]
                peak_idx = np.argmax(np.abs(irf_ij))
                peak_value = irf_ij[peak_idx]

                irf_summary += f"Response of {var_names[i]} to {var_names[j]}:\n"
                irf_summary += f"  Peak response: {peak_value:.6f} at period {peak_idx}\n"

                # Add confidence interval information if available
                if self.lower_ci is not None and self.upper_ci is not None:
                    lower = self.lower_ci[peak_idx, i, j]
                    upper = self.upper_ci[peak_idx, i, j]
                    irf_summary += f"  {int(self.confidence_level*100)}% CI at peak: [{lower:.6f}, {upper:.6f}]\n"

                # Check if the response is significant (CI doesn't include zero)
                if self.lower_ci is not None and self.upper_ci is not None:
                    significant = np.any((self.lower_ci[:, i, j] > 0) & (self.upper_ci[:, i, j] > 0)) or \
                        np.any((self.lower_ci[:, i, j] < 0) & (self.upper_ci[:, i, j] < 0))
                    irf_summary += f"  Significant response: {'Yes' if significant else 'No'}\n"

                irf_summary += "\n"

        return header + info + var_info + irf_summary


@jit(nopython=True, cache=True)
def _compute_ma_coefficients_numba(
    coef_matrices: List[np.ndarray],
    periods: int
) -> np.ndarray:
    """
    Numba-accelerated implementation of MA coefficient computation.

    Args:
        coef_matrices: List of coefficient matrices [A₁, A₂, ..., Aₚ]
        periods: Number of periods for MA coefficients

    Returns:
        np.ndarray: MA coefficient matrices (periods x k x k)
    """
    p = len(coef_matrices)
    k = coef_matrices[0].shape[0]

    # Initialize MA coefficient matrices
    ma_coefs = np.zeros((periods, k, k))
    ma_coefs[0] = np.eye(k)  # Psi_0 = I

    # Compute MA coefficients recursively
    for h in range(1, periods):
        for j in range(min(h, p)):
            ma_coefs[h] += coef_matrices[j] @ ma_coefs[h - j - 1]

    return ma_coefs
