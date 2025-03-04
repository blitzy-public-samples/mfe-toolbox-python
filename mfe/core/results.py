'''
Standardized result containers for the MFE Toolbox.

This module provides dataclass-based result objects that store estimation outputs,
diagnostics, and metadata in a consistent format across the MFE Toolbox. These
containers ensure consistent access to model outputs and support features like
pretty printing, serialization, and export to various formats.

The result classes follow a hierarchical structure with base classes for different
types of results (estimation, simulation, forecasting) and specialized subclasses
for specific model families.
''' 

import json
import pickle
import warnings
from dataclasses import dataclass, field, asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import (
    Any, Dict, Generic, List, Literal, Optional, Sequence, 
    Tuple, Type, TypeVar, Union, cast, get_args, get_origin
)

import numpy as np
import pandas as pd
from scipy import stats

from mfe.core.parameters import ParameterBase

# Type variables for generic result classes
T = TypeVar('T')  # Generic type for parameters
D = TypeVar('D')  # Generic type for data


@dataclass
class ModelResult:
    """Base class for all model results.
    
    This class provides common functionality for all result objects,
    including serialization, comparison, and display methods.
    
    Attributes:
        model_name: Name of the model that generated the results
        creation_time: Timestamp when the result was created
        metadata: Additional metadata about the result
    """
    
    model_name: str
    creation_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate result object after initialization."""
        # Ensure metadata is a dictionary
        if not isinstance(self.metadata, dict):
            self.metadata = dict(self.metadata)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result object to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the result object
        """
        result_dict = asdict(self)
        
        # Convert numpy arrays to lists for JSON serialization
        for key, value in result_dict.items():
            if isinstance(value, np.ndarray):
                result_dict[key] = value.tolist()
            elif isinstance(value, pd.DataFrame):
                result_dict[key] = value.to_dict()
            elif isinstance(value, pd.Series):
                result_dict[key] = value.to_dict()
            elif isinstance(value, datetime):
                result_dict[key] = value.isoformat()
        
        return result_dict
    
    def to_json(self, path: Optional[Union[str, Path]] = None, **kwargs: Any) -> Optional[str]:
        """Convert the result object to JSON.
        
        Args:
            path: Path to save the JSON file (if None, returns the JSON string)
            **kwargs: Additional keyword arguments for json.dump/dumps
        
        Returns:
            Optional[str]: JSON string if path is None, None otherwise
        """
        result_dict = self.to_dict()
        
        if path is None:
            return json.dumps(result_dict, **kwargs)
        
        with open(path, 'w') as f:
            json.dump(result_dict, f, **kwargs)
        
        return None
    
    def to_pickle(self, path: Union[str, Path]) -> None:
        """Save the result object as a pickle file.
        
        Args:
            path: Path to save the pickle file
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def from_pickle(cls, path: Union[str, Path]) -> 'ModelResult':
        """Load a result object from a pickle file.
        
        Args:
            path: Path to the pickle file
        
        Returns:
            ModelResult: Loaded result object
        
        Raises:
            TypeError: If the loaded object is not a ModelResult
        """
        with open(path, 'rb') as f:
            result = pickle.load(f)
        
        if not isinstance(result, ModelResult):
            raise TypeError(f"Loaded object is not a ModelResult, got {type(result)}")
        
        return result
    
    def summary(self) -> str:
        """Generate a text summary of the model results.
        
        Returns:
            str: A formatted string containing the model results summary
        """
        header = f"Model: {self.model_name}\n"
        header += "=" * (len(header) - 1) + "\n\n"
        
        timestamp = f"Created: {self.creation_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        metadata_str = ""
        if self.metadata:
            metadata_str = "Metadata:\n"
            for key, value in self.metadata.items():
                metadata_str += f"  {key}: {value}\n"
            metadata_str += "\n"
        
        return header + timestamp + metadata_str
    
    def __str__(self) -> str:
        """Generate a string representation of the result object.
        
        Returns:
            str: A string representation of the result object
        """
        return self.summary()
    
    def __repr__(self) -> str:
        """Generate a detailed string representation of the result object.
        
        Returns:
            str: A detailed string representation of the result object
        """
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"
    
    def compare(self, other: 'ModelResult') -> Dict[str, Any]:
        """Compare this result object with another result object.
        
        Args:
            other: Another result object to compare with
        
        Returns:
            Dict[str, Any]: Comparison results
        
        Raises:
            TypeError: If other is not a ModelResult
        """
        if not isinstance(other, ModelResult):
            raise TypeError(f"other must be a ModelResult, got {type(other)}")
        
        comparison = {
            "model_names": (self.model_name, other.model_name),
            "creation_times": (self.creation_time, other.creation_time),
            "same_class": self.__class__ == other.__class__
        }
        
        return comparison


@dataclass
class EstimationResult(ModelResult):
    """Base class for model estimation results.
    
    This class extends ModelResult to provide common functionality for
    estimation results, including fit statistics and parameter estimates.
    
    Attributes:
        parameters: Estimated model parameters
        convergence: Whether the optimization converged
        iterations: Number of iterations used in optimization
        log_likelihood: Log-likelihood value at the optimum
        aic: Akaike Information Criterion
        bic: Bayesian Information Criterion
        hqic: Hannan-Quinn Information Criterion
        std_errors: Standard errors of parameter estimates
        t_stats: t-statistics for parameter estimates
        p_values: p-values for parameter estimates
        covariance_matrix: Covariance matrix of parameter estimates
        optimization_message: Message from the optimizer
    """
    
    parameters: Optional[ParameterBase] = None
    convergence: bool = True
    iterations: int = 0
    log_likelihood: Optional[float] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    hqic: Optional[float] = None
    std_errors: Optional[np.ndarray] = None
    t_stats: Optional[np.ndarray] = None
    p_values: Optional[np.ndarray] = None
    covariance_matrix: Optional[np.ndarray] = None
    optimization_message: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate result object after initialization."""
        super().__post_init__()
        
        if not self.convergence:
            warnings.warn(
                f"Model {self.model_name} did not converge after {self.iterations} iterations. "
                f"Results may not be reliable.",
                UserWarning
            )
    
    def summary(self) -> str:
        """Generate a text summary of the estimation results.
        
        Returns:
            str: A formatted string containing the estimation results summary
        """
        base_summary = super().summary()
        
        convergence_info = f"Convergence: {'Yes' if self.convergence else 'No'}\n"
        convergence_info += f"Iterations: {self.iterations}\n"
        if self.optimization_message:
            convergence_info += f"Optimizer message: {self.optimization_message}\n"
        convergence_info += "\n"
        
        fit_stats = ""
        if self.log_likelihood is not None:
            fit_stats += f"Log-Likelihood: {self.log_likelihood:.6f}\n"
        if self.aic is not None:
            fit_stats += f"AIC: {self.aic:.6f}\n"
        if self.bic is not None:
            fit_stats += f"BIC: {self.bic:.6f}\n"
        if self.hqic is not None:
            fit_stats += f"HQIC: {self.hqic:.6f}\n"
        fit_stats += "\n"
        
        param_table = ""
        if self.parameters is not None:
            param_dict = self.parameters.to_dict()
            param_table = "Parameter Estimates:\n"
            param_table += "-" * 80 + "\n"
            param_table += f"{'Parameter':<20} {'Estimate':<12} {'Std. Error':<12} "
            param_table += f"{'t-Stat':<12} {'p-Value':<12}\n"
            param_table += "-" * 80 + "\n"
            
            for i, (name, value) in enumerate(param_dict.items()):
                std_err = self.std_errors[i] if self.std_errors is not None else np.nan
                t_stat = self.t_stats[i] if self.t_stats is not None else np.nan
                p_value = self.p_values[i] if self.p_values is not None else np.nan
                
                param_table += f"{name:<20} {value:<12.6f} "
                
                if not np.isnan(std_err):
                    param_table += f"{std_err:<12.6f} "
                else:
                    param_table += f"{'N/A':<12} "
                
                if not np.isnan(t_stat):
                    param_table += f"{t_stat:<12.6f} "
                else:
                    param_table += f"{'N/A':<12} "
                
                if not np.isnan(p_value):
                    param_table += f"{p_value:<12.6f}"
                    # Add significance stars
                    if p_value < 0.01:
                        param_table += " ***"
                    elif p_value < 0.05:
                        param_table += " **"
                    elif p_value < 0.1:
                        param_table += " *"
                else:
                    param_table += f"{'N/A':<12}"
                
                param_table += "\n"
            
            param_table += "-" * 80 + "\n"
            param_table += "Significance codes: *** 0.01, ** 0.05, * 0.1\n\n"
        
        return base_summary + convergence_info + fit_stats + param_table
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert parameter estimates to a pandas DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame containing parameter estimates and statistics
        
        Raises:
            ValueError: If parameters are not available
        """
        if self.parameters is None:
            raise ValueError("Parameters are not available")
        
        param_dict = self.parameters.to_dict()
        param_names = list(param_dict.keys())
        param_values = list(param_dict.values())
        
        data = {
            "Parameter": param_names,
            "Estimate": param_values
        }
        
        if self.std_errors is not None:
            data["Std. Error"] = self.std_errors
        
        if self.t_stats is not None:
            data["t-Stat"] = self.t_stats
        
        if self.p_values is not None:
            data["p-Value"] = self.p_values
        
        return pd.DataFrame(data)
    
    def compare(self, other: 'EstimationResult') -> Dict[str, Any]:
        """Compare this estimation result with another estimation result.
        
        Args:
            other: Another estimation result to compare with
        
        Returns:
            Dict[str, Any]: Comparison results
        
        Raises:
            TypeError: If other is not an EstimationResult
        """
        if not isinstance(other, EstimationResult):
            raise TypeError(f"other must be an EstimationResult, got {type(other)}")
        
        # Get base comparison
        comparison = super().compare(other)
        
        # Add estimation-specific comparisons
        if self.log_likelihood is not None and other.log_likelihood is not None:
            comparison["log_likelihood_diff"] = self.log_likelihood - other.log_likelihood
        
        if self.aic is not None and other.aic is not None:
            comparison["aic_diff"] = self.aic - other.aic
        
        if self.bic is not None and other.bic is not None:
            comparison["bic_diff"] = self.bic - other.bic
        
        if self.hqic is not None and other.hqic is not None:
            comparison["hqic_diff"] = self.hqic - other.hqic
        
        # Determine which model is preferred based on information criteria
        if self.aic is not None and other.aic is not None:
            comparison["preferred_by_aic"] = self.model_name if self.aic < other.aic else other.model_name
        
        if self.bic is not None and other.bic is not None:
            comparison["preferred_by_bic"] = self.model_name if self.bic < other.bic else other.model_name
        
        if self.hqic is not None and other.hqic is not None:
            comparison["preferred_by_hqic"] = self.model_name if self.hqic < other.hqic else other.model_name
        
        return comparison


@dataclass
class SimulationResult(ModelResult):
    """Base class for model simulation results.
    
    This class extends ModelResult to provide common functionality for
    simulation results, including simulated data and simulation parameters.
    
    Attributes:
        simulated_data: Simulated data
        n_periods: Number of periods simulated
        burn: Number of initial observations discarded
        seed: Random seed used for simulation
        simulation_parameters: Parameters used for simulation
    """
    
    simulated_data: np.ndarray
    n_periods: int
    burn: int = 0
    seed: Optional[int] = None
    simulation_parameters: Optional[ParameterBase] = None
    
    def __post_init__(self) -> None:
        """Validate result object after initialization."""
        super().__post_init__()
        
        # Ensure simulated_data is a NumPy array
        if not isinstance(self.simulated_data, np.ndarray):
            self.simulated_data = np.array(self.simulated_data)
    
    def summary(self) -> str:
        """Generate a text summary of the simulation results.
        
        Returns:
            str: A formatted string containing the simulation results summary
        """
        base_summary = super().summary()
        
        sim_info = f"Simulation Periods: {self.n_periods}\n"
        sim_info += f"Burn-in Periods: {self.burn}\n"
        if self.seed is not None:
            sim_info += f"Random Seed: {self.seed}\n"
        sim_info += "\n"
        
        data_stats = "Simulated Data Statistics:\n"
        data_stats += f"  Shape: {self.simulated_data.shape}\n"
        data_stats += f"  Mean: {np.mean(self.simulated_data):.6f}\n"
        data_stats += f"  Std. Dev.: {np.std(self.simulated_data):.6f}\n"
        data_stats += f"  Min: {np.min(self.simulated_data):.6f}\n"
        data_stats += f"  Max: {np.max(self.simulated_data):.6f}\n"
        data_stats += "\n"
        
        param_info = ""
        if self.simulation_parameters is not None:
            param_info = "Simulation Parameters:\n"
            for name, value in self.simulation_parameters.to_dict().items():
                param_info += f"  {name}: {value}\n"
            param_info += "\n"
        
        return base_summary + sim_info + data_stats + param_info
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert simulated data to a pandas DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame containing simulated data
        """
        if self.simulated_data.ndim == 1:
            return pd.DataFrame({"simulated_data": self.simulated_data})
        
        # For multivariate data, create column names
        columns = [f"series_{i+1}" for i in range(self.simulated_data.shape[1])]
        return pd.DataFrame(self.simulated_data, columns=columns)
    
    def plot(self, **kwargs: Any) -> Any:
        """Plot the simulated data.
        
        Args:
            **kwargs: Additional keyword arguments for plotting
        
        Returns:
            Any: Plot object
        
        Note:
            This method requires matplotlib to be installed.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib is required for plotting")
        
        df = self.to_dataframe()
        ax = df.plot(**kwargs)
        plt.title(f"Simulated Data from {self.model_name}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        
        return ax


@dataclass
class ForecastResult(ModelResult):
    """Base class for model forecast results.
    
    This class extends ModelResult to provide common functionality for
    forecast results, including point forecasts and prediction intervals.
    
    Attributes:
        forecasts: Point forecasts
        lower_bounds: Lower bounds of prediction intervals
        upper_bounds: Upper bounds of prediction intervals
        confidence_level: Confidence level for prediction intervals
        forecast_horizon: Number of steps forecasted
        forecast_origin: Last observation used for forecasting
        forecast_parameters: Parameters used for forecasting
    """
    
    forecasts: np.ndarray
    lower_bounds: Optional[np.ndarray] = None
    upper_bounds: Optional[np.ndarray] = None
    confidence_level: float = 0.95
    forecast_horizon: int = 1
    forecast_origin: Optional[Union[int, str, datetime]] = None
    forecast_parameters: Optional[ParameterBase] = None
    
    def __post_init__(self) -> None:
        """Validate result object after initialization."""
        super().__post_init__()
        
        # Ensure forecasts is a NumPy array
        if not isinstance(self.forecasts, np.ndarray):
            self.forecasts = np.array(self.forecasts)
        
        # Ensure lower_bounds and upper_bounds are NumPy arrays if provided
        if self.lower_bounds is not None and not isinstance(self.lower_bounds, np.ndarray):
            self.lower_bounds = np.array(self.lower_bounds)
        
        if self.upper_bounds is not None and not isinstance(self.upper_bounds, np.ndarray):
            self.upper_bounds = np.array(self.upper_bounds)
    
    def summary(self) -> str:
        """Generate a text summary of the forecast results.
        
        Returns:
            str: A formatted string containing the forecast results summary
        """
        base_summary = super().summary()
        
        forecast_info = f"Forecast Horizon: {self.forecast_horizon}\n"
        if self.forecast_origin is not None:
            forecast_info += f"Forecast Origin: {self.forecast_origin}\n"
        forecast_info += f"Confidence Level: {self.confidence_level:.2f}\n"
        forecast_info += "\n"
        
        forecast_stats = "Forecast Statistics:\n"
        forecast_stats += f"  Shape: {self.forecasts.shape}\n"
        forecast_stats += f"  Mean: {np.mean(self.forecasts):.6f}\n"
        forecast_stats += f"  Min: {np.min(self.forecasts):.6f}\n"
        forecast_stats += f"  Max: {np.max(self.forecasts):.6f}\n"
        forecast_stats += "\n"
        
        param_info = ""
        if self.forecast_parameters is not None:
            param_info = "Forecast Parameters:\n"
            for name, value in self.forecast_parameters.to_dict().items():
                param_info += f"  {name}: {value}\n"
            param_info += "\n"
        
        return base_summary + forecast_info + forecast_stats + param_info
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert forecast results to a pandas DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame containing forecast results
        """
        # Create index based on forecast horizon
        if isinstance(self.forecast_origin, (datetime, str)):
            # If forecast_origin is a date, create a date index
            try:
                if isinstance(self.forecast_origin, str):
                    origin = pd.to_datetime(self.forecast_origin)
                else:
                    origin = self.forecast_origin
                
                index = pd.date_range(
                    start=origin, 
                    periods=self.forecast_horizon + 1, 
                    freq='D'
                )[1:]  # Skip the origin
            except:
                # If date conversion fails, use numeric index
                index = range(1, self.forecast_horizon + 1)
        else:
            # Use numeric index
            if self.forecast_origin is not None:
                start = self.forecast_origin + 1
            else:
                start = 1
            
            index = range(start, start + self.forecast_horizon)
        
        # Create DataFrame
        if self.forecasts.ndim == 1:
            df = pd.DataFrame(
                {"forecast": self.forecasts}, 
                index=index
            )
            
            if self.lower_bounds is not None and self.upper_bounds is not None:
                df[f"lower_{int(self.confidence_level * 100)}"] = self.lower_bounds
                df[f"upper_{int(self.confidence_level * 100)}"] = self.upper_bounds
        else:
            # For multivariate forecasts
            columns = [f"series_{i+1}" for i in range(self.forecasts.shape[1])]
            df = pd.DataFrame(self.forecasts, index=index, columns=columns)
            
            if self.lower_bounds is not None and self.upper_bounds is not None:
                for i, col in enumerate(columns):
                    df[f"{col}_lower_{int(self.confidence_level * 100)}"] = self.lower_bounds[:, i]
                    df[f"{col}_upper_{int(self.confidence_level * 100)}"] = self.upper_bounds[:, i]
        
        return df
    
    def plot(self, **kwargs: Any) -> Any:
        """Plot the forecast results.
        
        Args:
            **kwargs: Additional keyword arguments for plotting
        
        Returns:
            Any: Plot object
        
        Note:
            This method requires matplotlib to be installed.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib is required for plotting")
        
        df = self.to_dataframe()
        
        if self.forecasts.ndim == 1:
            ax = df["forecast"].plot(**kwargs)
            
            if "lower_" + str(int(self.confidence_level * 100)) in df.columns:
                lower_col = "lower_" + str(int(self.confidence_level * 100))
                upper_col = "upper_" + str(int(self.confidence_level * 100))
                
                plt.fill_between(
                    df.index, 
                    df[lower_col], 
                    df[upper_col], 
                    alpha=0.2, 
                    color='blue'
                )
        else:
            ax = df.plot(**kwargs)
        
        plt.title(f"Forecasts from {self.model_name}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        
        return ax


@dataclass
class DiagnosticResult(ModelResult):
    """Base class for model diagnostic results.
    
    This class extends ModelResult to provide common functionality for
    diagnostic results, including test statistics and p-values.
    
    Attributes:
        test_statistics: Dictionary of test statistics
        p_values: Dictionary of p-values
        critical_values: Dictionary of critical values
        residuals: Model residuals
        standardized_residuals: Standardized model residuals
        test_descriptions: Dictionary of test descriptions
    """
    
    test_statistics: Dict[str, float] = field(default_factory=dict)
    p_values: Dict[str, float] = field(default_factory=dict)
    critical_values: Dict[str, Dict[str, float]] = field(default_factory=dict)
    residuals: Optional[np.ndarray] = None
    standardized_residuals: Optional[np.ndarray] = None
    test_descriptions: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate result object after initialization."""
        super().__post_init__()
        
        # Ensure residuals and standardized_residuals are NumPy arrays if provided
        if self.residuals is not None and not isinstance(self.residuals, np.ndarray):
            self.residuals = np.array(self.residuals)
        
        if self.standardized_residuals is not None and not isinstance(self.standardized_residuals, np.ndarray):
            self.standardized_residuals = np.array(self.standardized_residuals)
    
    def summary(self) -> str:
        """Generate a text summary of the diagnostic results.
        
        Returns:
            str: A formatted string containing the diagnostic results summary
        """
        base_summary = super().summary()
        
        test_results = "Diagnostic Test Results:\n"
        test_results += "-" * 80 + "\n"
        test_results += f"{'Test':<30} {'Statistic':<12} {'p-Value':<12} {'Result':<20}\n"
        test_results += "-" * 80 + "\n"
        
        for test_name in sorted(self.test_statistics.keys()):
            test_stat = self.test_statistics.get(test_name, np.nan)
            p_value = self.p_values.get(test_name, np.nan)
            
            test_results += f"{test_name:<30} "
            
            if not np.isnan(test_stat):
                test_results += f"{test_stat:<12.6f} "
            else:
                test_results += f"{'N/A':<12} "
            
            if not np.isnan(p_value):
                test_results += f"{p_value:<12.6f} "
                
                # Add test result
                if p_value < 0.05:
                    test_results += f"{'Reject H0':<20}"
                else:
                    test_results += f"{'Fail to reject H0':<20}"
            else:
                test_results += f"{'N/A':<12} {'N/A':<20}"
            
            test_results += "\n"
        
        test_results += "-" * 80 + "\n\n"
        
        # Add test descriptions if available
        descriptions = ""
        if self.test_descriptions:
            descriptions = "Test Descriptions:\n"
            for test_name, description in self.test_descriptions.items():
                descriptions += f"  {test_name}: {description}\n"
            descriptions += "\n"
        
        # Add residual statistics if available
        residual_stats = ""
        if self.residuals is not None:
            residual_stats = "Residual Statistics:\n"
            residual_stats += f"  Mean: {np.mean(self.residuals):.6f}\n"
            residual_stats += f"  Std. Dev.: {np.std(self.residuals):.6f}\n"
            residual_stats += f"  Skewness: {stats.skew(self.residuals):.6f}\n"
            residual_stats += f"  Kurtosis: {stats.kurtosis(self.residuals, fisher=True) + 3:.6f}\n"
            residual_stats += "\n"
        
        return base_summary + test_results + descriptions + residual_stats
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert diagnostic results to a pandas DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame containing diagnostic test results
        """
        data = []
        
        for test_name in sorted(self.test_statistics.keys()):
            test_stat = self.test_statistics.get(test_name, np.nan)
            p_value = self.p_values.get(test_name, np.nan)
            description = self.test_descriptions.get(test_name, "")
            
            # Get critical values if available
            crit_values = self.critical_values.get(test_name, {})
            crit_1pct = crit_values.get("1%", np.nan)
            crit_5pct = crit_values.get("5%", np.nan)
            crit_10pct = crit_values.get("10%", np.nan)
            
            # Determine test result
            if not np.isnan(p_value):
                if p_value < 0.01:
                    result = "Reject H0 at 1%"
                elif p_value < 0.05:
                    result = "Reject H0 at 5%"
                elif p_value < 0.1:
                    result = "Reject H0 at 10%"
                else:
                    result = "Fail to reject H0"
            else:
                result = "N/A"
            
            data.append({
                "Test": test_name,
                "Statistic": test_stat,
                "p-Value": p_value,
                "Critical 1%": crit_1pct,
                "Critical 5%": crit_5pct,
                "Critical 10%": crit_10pct,
                "Result": result,
                "Description": description
            })
        
        return pd.DataFrame(data)
    
    def plot_residuals(self, **kwargs: Any) -> Any:
        """Plot the residuals.
        
        Args:
            **kwargs: Additional keyword arguments for plotting
        
        Returns:
            Any: Plot object
        
        Raises:
            ValueError: If residuals are not available
            ImportError: If matplotlib is not installed
        """
        if self.residuals is None:
            raise ValueError("Residuals are not available")
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib is required for plotting")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Time series plot of residuals
        axes[0, 0].plot(self.residuals)
        axes[0, 0].set_title("Residuals")
        axes[0, 0].set_xlabel("Time")
        axes[0, 0].set_ylabel("Value")
        
        # Histogram of residuals
        axes[0, 1].hist(self.residuals, bins=30, density=True, alpha=0.6)
        
        # Add normal distribution curve
        x = np.linspace(
            np.min(self.residuals), 
            np.max(self.residuals), 
            100
        )
        mean = np.mean(self.residuals)
        std = np.std(self.residuals)
        axes[0, 1].plot(
            x, 
            stats.norm.pdf(x, mean, std), 
            'r-', 
            lw=2
        )
        axes[0, 1].set_title("Histogram of Residuals")
        
        # Q-Q plot
        stats.probplot(self.residuals, plot=axes[1, 0])
        axes[1, 0].set_title("Q-Q Plot")
        
        # Autocorrelation plot
        lags = min(40, len(self.residuals) // 5)
        acf = np.array([1.0] + [np.corrcoef(self.residuals[:-i], self.residuals[i:])[0, 1] 
                               for i in range(1, lags + 1)])
        
        axes[1, 1].bar(range(len(acf)), acf)
        axes[1, 1].set_title("Autocorrelation Function")
        axes[1, 1].set_xlabel("Lag")
        axes[1, 1].set_ylabel("ACF")
        
        # Add confidence bands
        conf_level = 1.96 / np.sqrt(len(self.residuals))
        axes[1, 1].axhline(y=conf_level, linestyle='--', color='r')
        axes[1, 1].axhline(y=-conf_level, linestyle='--', color='r')
        
        plt.tight_layout()
        
        return fig


@dataclass
class UnivariateVolatilityResult(EstimationResult):
    """Result container for univariate volatility models.
    
    This class extends EstimationResult to provide specialized functionality
    for univariate volatility model results.
    
    Attributes:
        conditional_variances: Estimated conditional variances
        standardized_residuals: Standardized residuals
        persistence: Model persistence (e.g., alpha + beta for GARCH)
        half_life: Half-life of volatility shocks
        unconditional_variance: Unconditional variance implied by the model
        residuals: Model residuals (typically squared returns)
    """
    
    conditional_variances: Optional[np.ndarray] = None
    standardized_residuals: Optional[np.ndarray] = None
    persistence: Optional[float] = None
    half_life: Optional[float] = None
    unconditional_variance: Optional[float] = None
    residuals: Optional[np.ndarray] = None
    
    def __post_init__(self) -> None:
        """Validate result object after initialization."""
        super().__post_init__()
        
        # Ensure arrays are NumPy arrays if provided
        if self.conditional_variances is not None and not isinstance(self.conditional_variances, np.ndarray):
            self.conditional_variances = np.array(self.conditional_variances)
        
        if self.standardized_residuals is not None and not isinstance(self.standardized_residuals, np.ndarray):
            self.standardized_residuals = np.array(self.standardized_residuals)
        
        if self.residuals is not None and not isinstance(self.residuals, np.ndarray):
            self.residuals = np.array(self.residuals)
    
    def summary(self) -> str:
        """Generate a text summary of the univariate volatility model results.
        
        Returns:
            str: A formatted string containing the model results summary
        """
        base_summary = super().summary()
        
        volatility_info = ""
        if self.persistence is not None:
            volatility_info += f"Persistence: {self.persistence:.6f}\n"
        
        if self.half_life is not None:
            volatility_info += f"Half-Life: {self.half_life:.6f}\n"
        
        if self.unconditional_variance is not None:
            volatility_info += f"Unconditional Variance: {self.unconditional_variance:.6f}\n"
        
        if volatility_info:
            volatility_info = "Volatility Properties:\n" + volatility_info + "\n"
        
        return base_summary + volatility_info
    
    def to_dataframe(self, include_data: bool = True) -> pd.DataFrame:
        """Convert parameter estimates to a pandas DataFrame.
        
        Args:
            include_data: Whether to include conditional variances and residuals
        
        Returns:
            pd.DataFrame: DataFrame containing parameter estimates and statistics
        """
        # Get parameter DataFrame
        param_df = super().to_dataframe()
        
        if not include_data or (self.conditional_variances is None and self.residuals is None):
            return param_df
        
        # Create DataFrame with time series data
        data_dict = {}
        
        if self.residuals is not None:
            data_dict["residuals"] = self.residuals
        
        if self.conditional_variances is not None:
            data_dict["conditional_variance"] = self.conditional_variances
            data_dict["conditional_volatility"] = np.sqrt(self.conditional_variances)
        
        if self.standardized_residuals is not None:
            data_dict["standardized_residuals"] = self.standardized_residuals
        
        if data_dict:
            return pd.DataFrame(data_dict)
        
        return param_df
    
    def plot_volatility(self, **kwargs: Any) -> Any:
        """Plot the conditional volatility.
        
        Args:
            **kwargs: Additional keyword arguments for plotting
        
        Returns:
            Any: Plot object
        
        Raises:
            ValueError: If conditional variances are not available
            ImportError: If matplotlib is not installed
        """
        if self.conditional_variances is None:
            raise ValueError("Conditional variances are not available")
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib is required for plotting")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot conditional volatility
        ax.plot(np.sqrt(self.conditional_variances), label="Conditional Volatility")
        
        # Add unconditional volatility if available
        if self.unconditional_variance is not None:
            ax.axhline(
                y=np.sqrt(self.unconditional_variance), 
                linestyle='--', 
                color='r', 
                label="Unconditional Volatility"
            )
        
        ax.set_title(f"Conditional Volatility from {self.model_name}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Volatility")
        ax.legend()
        
        return fig
    
    def plot_standardized_residuals(self, **kwargs: Any) -> Any:
        """Plot the standardized residuals.
        
        Args:
            **kwargs: Additional keyword arguments for plotting
        
        Returns:
            Any: Plot object
        
        Raises:
            ValueError: If standardized residuals are not available
            ImportError: If matplotlib is not installed
        """
        if self.standardized_residuals is None:
            raise ValueError("Standardized residuals are not available")
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib is required for plotting")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Time series plot of standardized residuals
        axes[0, 0].plot(self.standardized_residuals)
        axes[0, 0].set_title("Standardized Residuals")
        axes[0, 0].set_xlabel("Time")
        axes[0, 0].set_ylabel("Value")
        
        # Histogram of standardized residuals
        axes[0, 1].hist(self.standardized_residuals, bins=30, density=True, alpha=0.6)
        
        # Add normal distribution curve
        x = np.linspace(
            np.min(self.standardized_residuals), 
            np.max(self.standardized_residuals), 
            100
        )
        axes[0, 1].plot(
            x, 
            stats.norm.pdf(x, 0, 1), 
            'r-', 
            lw=2
        )
        axes[0, 1].set_title("Histogram of Standardized Residuals")
        
        # Q-Q plot
        stats.probplot(self.standardized_residuals, plot=axes[1, 0])
        axes[1, 0].set_title("Q-Q Plot")
        
        # Autocorrelation plot of squared standardized residuals
        lags = min(40, len(self.standardized_residuals) // 5)
        squared = self.standardized_residuals ** 2
        acf = np.array([1.0] + [np.corrcoef(squared[:-i], squared[i:])[0, 1] 
                               for i in range(1, lags + 1)])
        
        axes[1, 1].bar(range(len(acf)), acf)
        axes[1, 1].set_title("ACF of Squared Standardized Residuals")
        axes[1, 1].set_xlabel("Lag")
        axes[1, 1].set_ylabel("ACF")
        
        # Add confidence bands
        conf_level = 1.96 / np.sqrt(len(self.standardized_residuals))
        axes[1, 1].axhline(y=conf_level, linestyle='--', color='r')
        axes[1, 1].axhline(y=-conf_level, linestyle='--', color='r')
        
        plt.tight_layout()
        
        return fig


@dataclass
class MultivariateVolatilityResult(EstimationResult):
    """Result container for multivariate volatility models.
    
    This class extends EstimationResult to provide specialized functionality
    for multivariate volatility model results.
    
    Attributes:
        conditional_covariances: Estimated conditional covariance matrices
        conditional_correlations: Estimated conditional correlation matrices
        standardized_residuals: Standardized residuals
        n_assets: Number of assets in the model
        persistence: Model persistence
        half_life: Half-life of volatility shocks
        unconditional_covariance: Unconditional covariance matrix implied by the model
        residuals: Model residuals
    """
    
    conditional_covariances: Optional[np.ndarray] = None
    conditional_correlations: Optional[np.ndarray] = None
    standardized_residuals: Optional[np.ndarray] = None
    n_assets: Optional[int] = None
    persistence: Optional[float] = None
    half_life: Optional[float] = None
    unconditional_covariance: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None
    
    def __post_init__(self) -> None:
        """Validate result object after initialization."""
        super().__post_init__()
        
        # Ensure arrays are NumPy arrays if provided
        if self.conditional_covariances is not None and not isinstance(self.conditional_covariances, np.ndarray):
            self.conditional_covariances = np.array(self.conditional_covariances)
        
        if self.conditional_correlations is not None and not isinstance(self.conditional_correlations, np.ndarray):
            self.conditional_correlations = np.array(self.conditional_correlations)
        
        if self.standardized_residuals is not None and not isinstance(self.standardized_residuals, np.ndarray):
            self.standardized_residuals = np.array(self.standardized_residuals)
        
        if self.unconditional_covariance is not None and not isinstance(self.unconditional_covariance, np.ndarray):
            self.unconditional_covariance = np.array(self.unconditional_covariance)
        
        if self.residuals is not None and not isinstance(self.residuals, np.ndarray):
            self.residuals = np.array(self.residuals)
        
        # Infer n_assets if not provided
        if self.n_assets is None:
            if self.conditional_covariances is not None:
                self.n_assets = self.conditional_covariances.shape[0]
            elif self.residuals is not None:
                self.n_assets = self.residuals.shape[1]
    
    def summary(self) -> str:
        """Generate a text summary of the multivariate volatility model results.
        
        Returns:
            str: A formatted string containing the model results summary
        """
        base_summary = super().summary()
        
        volatility_info = ""
        if self.n_assets is not None:
            volatility_info += f"Number of Assets: {self.n_assets}\n"
        
        if self.persistence is not None:
            volatility_info += f"Persistence: {self.persistence:.6f}\n"
        
        if self.half_life is not None:
            volatility_info += f"Half-Life: {self.half_life:.6f}\n"
        
        if volatility_info:
            volatility_info = "Volatility Properties:\n" + volatility_info + "\n"
        
        # Add unconditional covariance matrix if available
        uncond_cov_info = ""
        if self.unconditional_covariance is not None:
            uncond_cov_info = "Unconditional Covariance Matrix:\n"
            for i in range(self.unconditional_covariance.shape[0]):
                row = " ".join(f"{val:.6f}" for val in self.unconditional_covariance[i])
                uncond_cov_info += f"  {row}\n"
            uncond_cov_info += "\n"
        
        return base_summary + volatility_info + uncond_cov_info
    
    def to_dataframe(self, include_data: bool = True) -> pd.DataFrame:
        """Convert parameter estimates to a pandas DataFrame.
        
        Args:
            include_data: Whether to include conditional covariances and residuals
        
        Returns:
            pd.DataFrame: DataFrame containing parameter estimates and statistics
        """
        # Get parameter DataFrame
        param_df = super().to_dataframe()
        
        if not include_data or self.n_assets is None:
            return param_df
        
        # For multivariate models, we need to create multiple DataFrames
        # Return only the parameter DataFrame by default
        return param_df
    
    def get_conditional_variances(self) -> pd.DataFrame:
        """Get conditional variances as a pandas DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame containing conditional variances
        
        Raises:
            ValueError: If conditional covariances are not available
        """
        if self.conditional_covariances is None:
            raise ValueError("Conditional covariances are not available")
        
        # Extract diagonal elements (variances)
        variances = np.zeros((self.conditional_covariances.shape[2], self.n_assets))
        
        for t in range(self.conditional_covariances.shape[2]):
            for i in range(self.n_assets):
                variances[t, i] = self.conditional_covariances[i, i, t]
        
        # Create column names
        columns = [f"asset_{i+1}_variance" for i in range(self.n_assets)]
        
        return pd.DataFrame(variances, columns=columns)
    
    def get_conditional_correlations(self) -> Dict[str, pd.DataFrame]:
        """Get conditional correlations as a dictionary of pandas DataFrames.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of DataFrames containing conditional correlations
        
        Raises:
            ValueError: If conditional correlations are not available
        """
        if self.conditional_correlations is None:
            raise ValueError("Conditional correlations are not available")
        
        result = {}
        
        # Create DataFrames for each time point
        for t in range(self.conditional_correlations.shape[2]):
            columns = [f"asset_{i+1}" for i in range(self.n_assets)]
            df = pd.DataFrame(
                self.conditional_correlations[:, :, t],
                index=columns,
                columns=columns
            )
            result[f"t_{t}"] = df
        
        return result
    
    def plot_conditional_variances(self, **kwargs: Any) -> Any:
        """Plot the conditional variances.
        
        Args:
            **kwargs: Additional keyword arguments for plotting
        
        Returns:
            Any: Plot object
        
        Raises:
            ValueError: If conditional covariances are not available
            ImportError: If matplotlib is not installed
        """
        if self.conditional_covariances is None:
            raise ValueError("Conditional covariances are not available")
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib is required for plotting")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract diagonal elements (variances)
        for i in range(self.n_assets):
            variances = np.array([
                self.conditional_covariances[i, i, t] 
                for t in range(self.conditional_covariances.shape[2])
            ])
            
            ax.plot(np.sqrt(variances), label=f"Asset {i+1}")
        
        ax.set_title(f"Conditional Volatilities from {self.model_name}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Volatility")
        ax.legend()
        
        return fig
    
    def plot_conditional_correlations(self, time_points: Optional[List[int]] = None, **kwargs: Any) -> Any:
        """Plot the conditional correlations.
        
        Args:
            time_points: List of time points to plot (if None, plots all pairwise correlations over time)
            **kwargs: Additional keyword arguments for plotting
        
        Returns:
            Any: Plot object
        
        Raises:
            ValueError: If conditional correlations are not available
            ImportError: If matplotlib is not installed
        """
        if self.conditional_correlations is None:
            raise ValueError("Conditional correlations are not available")
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            raise ImportError("Matplotlib and seaborn are required for plotting")
        
        if time_points is not None:
            # Plot correlation matrices at specific time points
            n_plots = len(time_points)
            fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
            
            if n_plots == 1:
                axes = [axes]
            
            for i, t in enumerate(time_points):
                if t >= self.conditional_correlations.shape[2]:
                    raise ValueError(f"Time point {t} is out of range")
                
                sns.heatmap(
                    self.conditional_correlations[:, :, t],
                    annot=True,
                    cmap="coolwarm",
                    vmin=-1,
                    vmax=1,
                    ax=axes[i]
                )
                axes[i].set_title(f"Correlation Matrix at t={t}")
            
            plt.tight_layout()
        else:
            # Plot all pairwise correlations over time
            n_pairs = (self.n_assets * (self.n_assets - 1)) // 2
            fig, ax = plt.subplots(figsize=(10, 6))
            
            pair_idx = 0
            for i in range(self.n_assets):
                for j in range(i+1, self.n_assets):
                    correlations = np.array([
                        self.conditional_correlations[i, j, t] 
                        for t in range(self.conditional_correlations.shape[2])
                    ])
                    
                    ax.plot(correlations, label=f"Assets {i+1}-{j+1}")
            
            ax.set_title(f"Conditional Correlations from {self.model_name}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Correlation")
            ax.set_ylim(-1, 1)
            ax.legend()
        
        return fig


@dataclass
class TimeSeriesResult(EstimationResult):
    """Result container for time series models.
    
    This class extends EstimationResult to provide specialized functionality
    for time series model results.
    
    Attributes:
        fitted_values: Fitted values from the model
        residuals: Model residuals
        ar_order: Order of the autoregressive component
        ma_order: Order of the moving average component
        ar_roots: Roots of the AR characteristic polynomial
        ma_roots: Roots of the MA characteristic polynomial
        acf: Autocorrelation function of the residuals
        pacf: Partial autocorrelation function of the residuals
        original_data: Original data used for estimation
    """
    
    fitted_values: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None
    ar_order: Optional[int] = None
    ma_order: Optional[int] = None
    ar_roots: Optional[np.ndarray] = None
    ma_roots: Optional[np.ndarray] = None
    acf: Optional[np.ndarray] = None
    pacf: Optional[np.ndarray] = None
    original_data: Optional[np.ndarray] = None
    
    def __post_init__(self) -> None:
        """Validate result object after initialization."""
        super().__post_init__()
        
        # Ensure arrays are NumPy arrays if provided
        if self.fitted_values is not None and not isinstance(self.fitted_values, np.ndarray):
            self.fitted_values = np.array(self.fitted_values)
        
        if self.residuals is not None and not isinstance(self.residuals, np.ndarray):
            self.residuals = np.array(self.residuals)
        
        if self.ar_roots is not None and not isinstance(self.ar_roots, np.ndarray):
            self.ar_roots = np.array(self.ar_roots)
        
        if self.ma_roots is not None and not isinstance(self.ma_roots, np.ndarray):
            self.ma_roots = np.array(self.ma_roots)
        
        if self.acf is not None and not isinstance(self.acf, np.ndarray):
            self.acf = np.array(self.acf)
        
        if self.pacf is not None and not isinstance(self.pacf, np.ndarray):
            self.pacf = np.array(self.pacf)
        
        if self.original_data is not None and not isinstance(self.original_data, np.ndarray):
            self.original_data = np.array(self.original_data)
    
    def summary(self) -> str:
        """Generate a text summary of the time series model results.
        
        Returns:
            str: A formatted string containing the model results summary
        """
        base_summary = super().summary()
        
        model_info = ""
        if self.ar_order is not None:
            model_info += f"AR Order: {self.ar_order}\n"
        
        if self.ma_order is not None:
            model_info += f"MA Order: {self.ma_order}\n"
        
        if model_info:
            model_info = "Model Information:\n" + model_info + "\n"
        
        # Add information about roots
        roots_info = ""
        if self.ar_roots is not None and len(self.ar_roots) > 0:
            roots_info += "AR Roots:\n"
            for i, root in enumerate(self.ar_roots):
                roots_info += f"  Root {i+1}: {root}\n"
            roots_info += f"  All AR roots are {'outside' if np.all(np.abs(self.ar_roots) > 1) else 'inside'} the unit circle.\n"
        
        if self.ma_roots is not None and len(self.ma_roots) > 0:
            roots_info += "MA Roots:\n"
            for i, root in enumerate(self.ma_roots):
                roots_info += f"  Root {i+1}: {root}\n"
            roots_info += f"  All MA roots are {'outside' if np.all(np.abs(self.ma_roots) > 1) else 'inside'} the unit circle.\n"
        
        if roots_info:
            roots_info += "\n"
        
        # Add residual statistics
        residual_stats = ""
        if self.residuals is not None:
            residual_stats = "Residual Statistics:\n"
            residual_stats += f"  Mean: {np.mean(self.residuals):.6f}\n"
            residual_stats += f"  Std. Dev.: {np.std(self.residuals):.6f}\n"
            residual_stats += f"  Skewness: {stats.skew(self.residuals):.6f}\n"
            residual_stats += f"  Kurtosis: {stats.kurtosis(self.residuals, fisher=True) + 3:.6f}\n"
            residual_stats += "\n"
        
        return base_summary + model_info + roots_info + residual_stats
    
    def to_dataframe(self, include_data: bool = True) -> pd.DataFrame:
        """Convert time series results to a pandas DataFrame.
        
        Args:
            include_data: Whether to include fitted values and residuals
        
        Returns:
            pd.DataFrame: DataFrame containing time series results
        """
        # Get parameter DataFrame
        param_df = super().to_dataframe()
        
        if not include_data or (self.fitted_values is None and self.residuals is None):
            return param_df
        
        # Create DataFrame with time series data
        data_dict = {}
        
        if self.original_data is not None:
            data_dict["original_data"] = self.original_data
        
        if self.fitted_values is not None:
            data_dict["fitted_values"] = self.fitted_values
        
        if self.residuals is not None:
            data_dict["residuals"] = self.residuals
        
        if data_dict:
            return pd.DataFrame(data_dict)
        
        return param_df
    
    def plot_fit(self, **kwargs: Any) -> Any:
        """Plot the original data and fitted values.
        
        Args:
            **kwargs: Additional keyword arguments for plotting
        
        Returns:
            Any: Plot object
        
        Raises:
            ValueError: If fitted values are not available
            ImportError: If matplotlib is not installed
        """
        if self.fitted_values is None:
            raise ValueError("Fitted values are not available")
        
        if self.original_data is None:
            raise ValueError("Original data is not available")
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib is required for plotting")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(self.original_data, label="Original Data")
        ax.plot(self.fitted_values, label="Fitted Values", linestyle='--')
        
        ax.set_title(f"Model Fit for {self.model_name}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        
        return fig
    
    def plot_acf_pacf(self, lags: int = 40, **kwargs: Any) -> Any:
        """Plot the ACF and PACF of the residuals.
        
        Args:
            lags: Number of lags to include
            **kwargs: Additional keyword arguments for plotting
        
        Returns:
            Any: Plot object
        
        Raises:
            ValueError: If residuals are not available
            ImportError: If matplotlib is not installed
        """
        if self.residuals is None:
            raise ValueError("Residuals are not available")
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib is required for plotting")
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Compute ACF if not already available
        if self.acf is None:
            lags = min(lags, len(self.residuals) - 1)
            acf = np.array([1.0] + [np.corrcoef(self.residuals[:-i], self.residuals[i:])[0, 1] 
                                   for i in range(1, lags + 1)])
        else:
            acf = self.acf
            lags = min(lags, len(acf) - 1)
        
        # Plot ACF
        axes[0].bar(range(lags + 1), acf[:lags + 1])
        axes[0].set_title("Autocorrelation Function (ACF)")
        axes[0].set_xlabel("Lag")
        axes[0].set_ylabel("ACF")
        
        # Add confidence bands
        conf_level = 1.96 / np.sqrt(len(self.residuals))
        axes[0].axhline(y=conf_level, linestyle='--', color='r')
        axes[0].axhline(y=-conf_level, linestyle='--', color='r')
        
        # Compute PACF if not already available
        if self.pacf is None:
            # Simple implementation of PACF
            pacf = np.zeros(lags + 1)
            pacf[0] = 1.0
            
            # Compute PACF using Yule-Walker equations
            for i in range(1, lags + 1):
                r = acf[1:i+1]
                R = np.zeros((i, i))
                for j in range(i):
                    for k in range(i):
                        if j == k:
                            R[j, k] = 1.0
                        else:
                            R[j, k] = acf[abs(j - k)]
                
                try:
                    phi = np.linalg.solve(R, r)
                    pacf[i] = phi[-1]
                except np.linalg.LinAlgError:
                    # If matrix is singular, use pseudo-inverse
                    phi = np.linalg.lstsq(R, r, rcond=None)[0]
                    pacf[i] = phi[-1]
        else:
            pacf = self.pacf
            lags = min(lags, len(pacf) - 1)
        
        # Plot PACF
        axes[1].bar(range(lags + 1), pacf[:lags + 1])
        axes[1].set_title("Partial Autocorrelation Function (PACF)")
        axes[1].set_xlabel("Lag")
        axes[1].set_ylabel("PACF")
        
        # Add confidence bands
        axes[1].axhline(y=conf_level, linestyle='--', color='r')
        axes[1].axhline(y=-conf_level, linestyle='--', color='r')
        
        plt.tight_layout()
        
        return fig


@dataclass
class BootstrapResult(ModelResult):
    """Result container for bootstrap methods.
    
    This class extends ModelResult to provide specialized functionality
    for bootstrap results.
    
    Attributes:
        bootstrap_samples: Bootstrap samples
        bootstrap_statistics: Statistics computed from bootstrap samples
        original_statistic: Statistic computed from original data
        confidence_intervals: Confidence intervals for the statistic
        p_value: Bootstrap p-value
        n_bootstraps: Number of bootstrap replications
        block_length: Block length used for block bootstrap
        confidence_level: Confidence level for intervals
        bootstrap_method: Method used for bootstrap (e.g., 'block', 'stationary')
    """
    
    bootstrap_samples: Optional[np.ndarray] = None
    bootstrap_statistics: Optional[np.ndarray] = None
    original_statistic: Optional[Union[float, np.ndarray]] = None
    confidence_intervals: Optional[Dict[str, Union[float, np.ndarray]]] = None
    p_value: Optional[float] = None
    n_bootstraps: Optional[int] = None
    block_length: Optional[float] = None
    confidence_level: float = 0.95
    bootstrap_method: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate result object after initialization."""
        super().__post_init__()
        
        # Ensure arrays are NumPy arrays if provided
        if self.bootstrap_samples is not None and not isinstance(self.bootstrap_samples, np.ndarray):
            self.bootstrap_samples = np.array(self.bootstrap_samples)
        
        if self.bootstrap_statistics is not None and not isinstance(self.bootstrap_statistics, np.ndarray):
            self.bootstrap_statistics = np.array(self.bootstrap_statistics)
        
        if self.original_statistic is not None and isinstance(self.original_statistic, (list, tuple)):
            self.original_statistic = np.array(self.original_statistic)
    
    def summary(self) -> str:
        """Generate a text summary of the bootstrap results.
        
        Returns:
            str: A formatted string containing the bootstrap results summary
        """
        base_summary = super().summary()
        
        bootstrap_info = ""
        if self.bootstrap_method is not None:
            bootstrap_info += f"Bootstrap Method: {self.bootstrap_method}\n"
        
        if self.n_bootstraps is not None:
            bootstrap_info += f"Number of Bootstrap Replications: {self.n_bootstraps}\n"
        
        if self.block_length is not None:
            bootstrap_info += f"Block Length: {self.block_length}\n"
        
        bootstrap_info += f"Confidence Level: {self.confidence_level:.2f}\n"
        bootstrap_info += "\n"
        
        statistic_info = ""
        if self.original_statistic is not None:
            if isinstance(self.original_statistic, np.ndarray) and self.original_statistic.size > 1:
                statistic_info += "Original Statistics:\n"
                for i, stat in enumerate(self.original_statistic):
                    statistic_info += f"  Statistic {i+1}: {stat:.6f}\n"
            else:
                statistic_info += f"Original Statistic: {self.original_statistic:.6f}\n"
        
        if self.p_value is not None:
            statistic_info += f"Bootstrap p-value: {self.p_value:.6f}\n"
        
        if self.confidence_intervals is not None:
            statistic_info += "Confidence Intervals:\n"
            for name, interval in self.confidence_intervals.items():
                if isinstance(interval, np.ndarray) and interval.size > 1:
                    statistic_info += f"  {name}:\n"
                    for i, val in enumerate(interval):
                        statistic_info += f"    Statistic {i+1}: {val:.6f}\n"
                else:
                    statistic_info += f"  {name}: {interval:.6f}\n"
        
        if statistic_info:
            statistic_info += "\n"
        
        bootstrap_stats = ""
        if self.bootstrap_statistics is not None:
            bootstrap_stats = "Bootstrap Statistics Summary:\n"
            
            if self.bootstrap_statistics.ndim == 1:
                bootstrap_stats += f"  Mean: {np.mean(self.bootstrap_statistics):.6f}\n"
                bootstrap_stats += f"  Std. Dev.: {np.std(self.bootstrap_statistics):.6f}\n"
                bootstrap_stats += f"  Min: {np.min(self.bootstrap_statistics):.6f}\n"
                bootstrap_stats += f"  Max: {np.max(self.bootstrap_statistics):.6f}\n"
            else:
                for i in range(self.bootstrap_statistics.shape[1]):
                    bootstrap_stats += f"  Statistic {i+1}:\n"
                    bootstrap_stats += f"    Mean: {np.mean(self.bootstrap_statistics[:, i]):.6f}\n"
                    bootstrap_stats += f"    Std. Dev.: {np.std(self.bootstrap_statistics[:, i]):.6f}\n"
                    bootstrap_stats += f"    Min: {np.min(self.bootstrap_statistics[:, i]):.6f}\n"
                    bootstrap_stats += f"    Max: {np.max(self.bootstrap_statistics[:, i]):.6f}\n"
            
            bootstrap_stats += "\n"
        
        return base_summary + bootstrap_info + statistic_info + bootstrap_stats
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert bootstrap statistics to a pandas DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame containing bootstrap statistics
        
        Raises:
            ValueError: If bootstrap statistics are not available
        """
        if self.bootstrap_statistics is None:
            raise ValueError("Bootstrap statistics are not available")
        
        if self.bootstrap_statistics.ndim == 1:
            return pd.DataFrame({"bootstrap_statistic": self.bootstrap_statistics})
        
        # For multivariate statistics
        columns = [f"statistic_{i+1}" for i in range(self.bootstrap_statistics.shape[1])]
        return pd.DataFrame(self.bootstrap_statistics, columns=columns)
    
    def plot_histogram(self, **kwargs: Any) -> Any:
        """Plot a histogram of bootstrap statistics.
        
        Args:
            **kwargs: Additional keyword arguments for plotting
        
        Returns:
            Any: Plot object
        
        Raises:
            ValueError: If bootstrap statistics are not available
            ImportError: If matplotlib is not installed
        """
        if self.bootstrap_statistics is None:
            raise ValueError("Bootstrap statistics are not available")
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib is required for plotting")
        
        if self.bootstrap_statistics.ndim == 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.hist(self.bootstrap_statistics, bins=30, alpha=0.6, density=True)
            
            # Add vertical line for original statistic
            if self.original_statistic is not None:
                ax.axvline(
                    x=self.original_statistic, 
                    color='r', 
                    linestyle='--', 
                    label="Original Statistic"
                )
            
            # Add confidence interval
            if self.confidence_intervals is not None:
                lower = self.confidence_intervals.get("lower", None)
                upper = self.confidence_intervals.get("upper", None)
                
                if lower is not None and upper is not None:
                    ax.axvline(x=lower, color='g', linestyle='--', label="Lower CI")
                    ax.axvline(x=upper, color='g', linestyle='--', label="Upper CI")
            
            ax.set_title(f"Bootstrap Distribution for {self.model_name}")
            ax.set_xlabel("Statistic Value")
            ax.set_ylabel("Density")
            ax.legend()
        else:
            # For multivariate statistics, create multiple subplots
            n_stats = self.bootstrap_statistics.shape[1]
            fig, axes = plt.subplots(n_stats, 1, figsize=(10, 4 * n_stats))
            
            if n_stats == 1:
                axes = [axes]
            
            for i in range(n_stats):
                axes[i].hist(self.bootstrap_statistics[:, i], bins=30, alpha=0.6, density=True)
                
                # Add vertical line for original statistic
                if self.original_statistic is not None:
                    if isinstance(self.original_statistic, np.ndarray) and self.original_statistic.size > 1:
                        axes[i].axvline(
                            x=self.original_statistic[i], 
                            color='r', 
                            linestyle='--', 
                            label="Original Statistic"
                        )
                    else:
                        axes[i].axvline(
                            x=self.original_statistic, 
                            color='r', 
                            linestyle='--', 
                            label="Original Statistic"
                        )
                
                # Add confidence interval
                if self.confidence_intervals is not None:
                    lower = self.confidence_intervals.get("lower", None)
                    upper = self.confidence_intervals.get("upper", None)
                    
                    if lower is not None and upper is not None:
                        if isinstance(lower, np.ndarray) and lower.size > 1:
                            axes[i].axvline(x=lower[i], color='g', linestyle='--', label="Lower CI")
                            axes[i].axvline(x=upper[i], color='g', linestyle='--', label="Upper CI")
                        else:
                            axes[i].axvline(x=lower, color='g', linestyle='--', label="Lower CI")
                            axes[i].axvline(x=upper, color='g', linestyle='--', label="Upper CI")
                
                axes[i].set_title(f"Bootstrap Distribution for Statistic {i+1}")
                axes[i].set_xlabel("Statistic Value")
                axes[i].set_ylabel("Density")
                axes[i].legend()
            
            plt.tight_layout()
        
        return fig


@dataclass
class RealizedVolatilityResult(ModelResult):
    """Result container for realized volatility models.
    
    This class extends ModelResult to provide specialized functionality
    for realized volatility results.
    
    Attributes:
        realized_measure: Computed realized measure (e.g., variance, kernel)
        prices: High-frequency price data used for computation
        times: Corresponding time points
        sampling_frequency: Sampling frequency used for computation
        kernel_type: Type of kernel used (for kernel-based estimators)
        bandwidth: Bandwidth parameter (for kernel-based estimators)
        subsampling: Whether subsampling was used
        noise_correction: Whether noise correction was applied
        annualization_factor: Factor used for annualization
    """
    
    realized_measure: np.ndarray
    prices: Optional[np.ndarray] = None
    times: Optional[np.ndarray] = None
    sampling_frequency: Optional[Union[str, float]] = None
    kernel_type: Optional[str] = None
    bandwidth: Optional[float] = None
    subsampling: bool = False
    noise_correction: bool = False
    annualization_factor: Optional[float] = None
    
    def __post_init__(self) -> None:
        """Validate result object after initialization."""
        super().__post_init__()
        
        # Ensure arrays are NumPy arrays if provided
        if not isinstance(self.realized_measure, np.ndarray):
            self.realized_measure = np.array(self.realized_measure)
        
        if self.prices is not None and not isinstance(self.prices, np.ndarray):
            self.prices = np.array(self.prices)
        
        if self.times is not None and not isinstance(self.times, np.ndarray):
            self.times = np.array(self.times)
    
    def summary(self) -> str:
        """Generate a text summary of the realized volatility results.
        
        Returns:
            str: A formatted string containing the realized volatility results summary
        """
        base_summary = super().summary()
        
        method_info = ""
        if self.sampling_frequency is not None:
            method_info += f"Sampling Frequency: {self.sampling_frequency}\n"
        
        if self.kernel_type is not None:
            method_info += f"Kernel Type: {self.kernel_type}\n"
        
        if self.bandwidth is not None:
            method_info += f"Bandwidth: {self.bandwidth}\n"
        
        method_info += f"Subsampling: {'Yes' if self.subsampling else 'No'}\n"
        method_info += f"Noise Correction: {'Yes' if self.noise_correction else 'No'}\n"
        
        if self.annualization_factor is not None:
            method_info += f"Annualization Factor: {self.annualization_factor}\n"
        
        if method_info:
            method_info = "Estimation Method:\n" + method_info + "\n"
        
        measure_stats = "Realized Measure Statistics:\n"
        measure_stats += f"  Shape: {self.realized_measure.shape}\n"
        
        if self.realized_measure.ndim == 1:
            measure_stats += f"  Mean: {np.mean(self.realized_measure):.6f}\n"
            measure_stats += f"  Std. Dev.: {np.std(self.realized_measure):.6f}\n"
            measure_stats += f"  Min: {np.min(self.realized_measure):.6f}\n"
            measure_stats += f"  Max: {np.max(self.realized_measure):.6f}\n"
        else:
            # For multivariate measures, show diagonal elements
            for i in range(self.realized_measure.shape[0]):
                measure_stats += f"  Asset {i+1} Variance:\n"
                measure_stats += f"    Mean: {np.mean(self.realized_measure[i, i, :]):.6f}\n"
                measure_stats += f"    Std. Dev.: {np.std(self.realized_measure[i, i, :]):.6f}\n"
                measure_stats += f"    Min: {np.min(self.realized_measure[i, i, :]):.6f}\n"
                measure_stats += f"    Max: {np.max(self.realized_measure[i, i, :]):.6f}\n"
        
        measure_stats += "\n"
        
        return base_summary + method_info + measure_stats
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert realized measure to a pandas DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame containing realized measure
        """
        if self.realized_measure.ndim == 1:
            return pd.DataFrame({"realized_measure": self.realized_measure})
        
        # For multivariate measures
        if self.realized_measure.ndim == 3:
            # Extract diagonal elements (variances)
            n_assets = self.realized_measure.shape[0]
            variances = np.zeros((self.realized_measure.shape[2], n_assets))
            
            for t in range(self.realized_measure.shape[2]):
                for i in range(n_assets):
                    variances[t, i] = self.realized_measure[i, i, t]
            
            # Create column names
            columns = [f"asset_{i+1}_variance" for i in range(n_assets)]
            
            return pd.DataFrame(variances, columns=columns)
        
        # For other dimensions, return as is
        return pd.DataFrame(self.realized_measure)
    
    def plot(self, **kwargs: Any) -> Any:
        """Plot the realized measure.
        
        Args:
            **kwargs: Additional keyword arguments for plotting
        
        Returns:
            Any: Plot object
        
        Raises:
            ImportError: If matplotlib is not installed
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib is required for plotting")
        
        if self.realized_measure.ndim == 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot realized measure
            ax.plot(np.sqrt(self.realized_measure), label="Realized Volatility")
            
            ax.set_title(f"Realized Volatility from {self.model_name}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Volatility")
            ax.legend()
        else:
            # For multivariate measures
            if self.realized_measure.ndim == 3:
                n_assets = self.realized_measure.shape[0]
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Extract diagonal elements (variances)
                for i in range(n_assets):
                    variances = np.array([
                        self.realized_measure[i, i, t] 
                        for t in range(self.realized_measure.shape[2])
                    ])
                    
                    ax.plot(np.sqrt(variances), label=f"Asset {i+1}")
                
                ax.set_title(f"Realized Volatilities from {self.model_name}")
                ax.set_xlabel("Time")
                ax.set_ylabel("Volatility")
                ax.legend()
            else:
                # For other dimensions, create a heatmap
                fig, ax = plt.subplots(figsize=(10, 8))
                
                im = ax.imshow(self.realized_measure, cmap="viridis")
                plt.colorbar(im, ax=ax)
                
                ax.set_title(f"Realized Measure from {self.model_name}")
        
        return fig


@dataclass
class CrossSectionalResult(EstimationResult):
    """Result container for cross-sectional models.
    
    This class extends EstimationResult to provide specialized functionality
    for cross-sectional model results.
    
    Attributes:
        fitted_values: Fitted values from the model
        residuals: Model residuals
        r_squared: R-squared value
        adjusted_r_squared: Adjusted R-squared value
        f_statistic: F-statistic
        f_p_value: p-value for the F-statistic
        residual_std_error: Residual standard error
        degrees_of_freedom: Degrees of freedom
        original_data: Original data used for estimation (y, X)
    """
    
    fitted_values: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None
    r_squared: Optional[float] = None
    adjusted_r_squared: Optional[float] = None
    f_statistic: Optional[float] = None
    f_p_value: Optional[float] = None
    residual_std_error: Optional[float] = None
    degrees_of_freedom: Optional[int] = None
    original_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    
    def __post_init__(self) -> None:
        """Validate result object after initialization."""
        super().__post_init__()
        
        # Ensure arrays are NumPy arrays if provided
        if self.fitted_values is not None and not isinstance(self.fitted_values, np.ndarray):
            self.fitted_values = np.array(self.fitted_values)
        
        if self.residuals is not None and not isinstance(self.residuals, np.ndarray):
            self.residuals = np.array(self.residuals)
        
        if self.original_data is not None:
            y, X = self.original_data
            if not isinstance(y, np.ndarray):
                y = np.array(y)
            if not isinstance(X, np.ndarray):
                X = np.array(X)
            self.original_data = (y, X)
    
    def summary(self) -> str:
        """Generate a text summary of the cross-sectional model results.
        
        Returns:
            str: A formatted string containing the model results summary
        """
        base_summary = super().summary()
        
        fit_stats = ""
        if self.r_squared is not None:
            fit_stats += f"R-squared: {self.r_squared:.6f}\n"
        
        if self.adjusted_r_squared is not None:
            fit_stats += f"Adjusted R-squared: {self.adjusted_r_squared:.6f}\n"
        
        if self.f_statistic is not None:
            fit_stats += f"F-statistic: {self.f_statistic:.6f}"
            if self.f_p_value is not None:
                fit_stats += f" (p-value: {self.f_p_value:.6f})"
            fit_stats += "\n"
        
        if self.residual_std_error is not None:
            fit_stats += f"Residual Std. Error: {self.residual_std_error:.6f}"
            if self.degrees_of_freedom is not None:
                fit_stats += f" (df: {self.degrees_of_freedom})"
            fit_stats += "\n"
        
        if fit_stats:
            fit_stats = "Model Fit Statistics:\n" + fit_stats + "\n"
        
        # Add residual statistics
        residual_stats = ""
        if self.residuals is not None:
            residual_stats = "Residual Statistics:\n"
            residual_stats += f"  Mean: {np.mean(self.residuals):.6f}\n"
            residual_stats += f"  Std. Dev.: {np.std(self.residuals):.6f}\n"
            residual_stats += f"  Skewness: {stats.skew(self.residuals):.6f}\n"
            residual_stats += f"  Kurtosis: {stats.kurtosis(self.residuals, fisher=True) + 3:.6f}\n"
            residual_stats += "\n"
        
        return base_summary + fit_stats + residual_stats
    
    def to_dataframe(self, include_data: bool = True) -> pd.DataFrame:
        """Convert cross-sectional results to a pandas DataFrame.
        
        Args:
            include_data: Whether to include fitted values and residuals
        
        Returns:
            pd.DataFrame: DataFrame containing cross-sectional results
        """
        # Get parameter DataFrame
        param_df = super().to_dataframe()
        
        if not include_data or (self.fitted_values is None and self.residuals is None):
            return param_df
        
        # Create DataFrame with data
        data_dict = {}
        
        if self.original_data is not None:
            y, X = self.original_data
            data_dict["y"] = y
            
            # Add X columns
            n_vars = X.shape[1]
            for i in range(n_vars):
                data_dict[f"X_{i+1}"] = X[:, i]
        
        if self.fitted_values is not None:
            data_dict["fitted_values"] = self.fitted_values
        
        if self.residuals is not None:
            data_dict["residuals"] = self.residuals
        
        if data_dict:
            return pd.DataFrame(data_dict)
        
        return param_df
    
    def plot_fit(self, **kwargs: Any) -> Any:
        """Plot the original data and fitted values.
        
        Args:
            **kwargs: Additional keyword arguments for plotting
        
        Returns:
            Any: Plot object
        
        Raises:
            ValueError: If fitted values are not available
            ImportError: If matplotlib is not installed
        """
        if self.fitted_values is None:
            raise ValueError("Fitted values are not available")
        
        if self.original_data is None:
            raise ValueError("Original data is not available")
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib is required for plotting")
        
        y, X = self.original_data
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.scatter(y, self.fitted_values, alpha=0.6)
        
        # Add 45-degree line
        min_val = min(np.min(y), np.min(self.fitted_values))
        max_val = max(np.max(y), np.max(self.fitted_values))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        ax.set_title(f"Actual vs. Fitted Values for {self.model_name}")
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Fitted Values")
        
        return fig
    
    def plot_residuals(self, **kwargs: Any) -> Any:
        """Plot the residuals.
        
        Args:
            **kwargs: Additional keyword arguments for plotting
        
        Returns:
            Any: Plot object
        
        Raises:
            ValueError: If residuals are not available
            ImportError: If matplotlib is not installed
        """
        if self.residuals is None:
            raise ValueError("Residuals are not available")
        
        if self.fitted_values is None:
            raise ValueError("Fitted values are not available")
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib is required for plotting")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Residuals vs. fitted values
        axes[0, 0].scatter(self.fitted_values, self.residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, linestyle='--', color='r')
        axes[0, 0].set_title("Residuals vs. Fitted Values")
        axes[0, 0].set_xlabel("Fitted Values")
        axes[0, 0].set_ylabel("Residuals")
        
        # Histogram of residuals
        axes[0, 1].hist(self.residuals, bins=30, density=True, alpha=0.6)
        
        # Add normal distribution curve
        x = np.linspace(
            np.min(self.residuals), 
            np.max(self.residuals), 
            100
        )
        mean = np.mean(self.residuals)
        std = np.std(self.residuals)
        axes[0, 1].plot(
            x, 
            stats.norm.pdf(x, mean, std), 
            'r-', 
            lw=2
        )
        axes[0, 1].set_title("Histogram of Residuals")
        
        # Q-Q plot
        stats.probplot(self.residuals, plot=axes[1, 0])
        axes[1, 0].set_title("Q-Q Plot")
        
        # Scale-Location plot
        axes[1, 1].scatter(
            self.fitted_values, 
            np.sqrt(np.abs(self.residuals / np.std(self.residuals))), 
            alpha=0.6
        )
        axes[1, 1].set_title("Scale-Location Plot")
        axes[1, 1].set_xlabel("Fitted Values")
        axes[1, 1].set_ylabel("√|Standardized Residuals|")
        
        plt.tight_layout()
        
        return fig


# Helper functions for loading and saving results

def load_result(path: Union[str, Path]) -> ModelResult:
    """Load a result object from a file.
    
    Args:
        path: Path to the file
    
    Returns:
        ModelResult: Loaded result object
    
    Raises:
        ValueError: If the file format is not supported
        TypeError: If the loaded object is not a ModelResult
    """
    path_obj = Path(path)
    
    if path_obj.suffix == '.pkl':
        return ModelResult.from_pickle(path)
    
    raise ValueError(f"Unsupported file format: {path_obj.suffix}")


def save_result(result: ModelResult, path: Union[str, Path], format: str = 'pkl') -> None:
    """Save a result object to a file.
    
    Args:
        result: Result object to save
        path: Path to save the file
        format: File format ('pkl' or 'json')
    
    Raises:
        ValueError: If the format is not supported
        TypeError: If result is not a ModelResult
    """
    if not isinstance(result, ModelResult):
        raise TypeError(f"result must be a ModelResult, got {type(result)}")
    
    path_obj = Path(path)
    
    if format == 'pkl' or path_obj.suffix == '.pkl':
        result.to_pickle(path)
    elif format == 'json' or path_obj.suffix == '.json':
        result.to_json(path, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")


def compare_results(results: List[ModelResult]) -> pd.DataFrame:
    """Compare multiple result objects.
    
    Args:
        results: List of result objects to compare
    
    Returns:
        pd.DataFrame: DataFrame containing comparison results
    
    Raises:
        ValueError: If results is empty
        TypeError: If any element in results is not a ModelResult
    """
    if not results:
        raise ValueError("results cannot be empty")
    
    for result in results:
        if not isinstance(result, ModelResult):
            raise TypeError(f"All elements in results must be ModelResult objects, got {type(result)}")
    
    # Create comparison DataFrame
    comparison_data = []
    
    for i, result1 in enumerate(results):
        for j, result2 in enumerate(results[i+1:], i+1):
            comparison = result1.compare(result2)
            
            row = {
                "Model 1": result1.model_name,
                "Model 2": result2.model_name,
                "Same Class": comparison.get("same_class", False)
            }
            
            # Add additional comparison metrics if available
            if isinstance(result1, EstimationResult) and isinstance(result2, EstimationResult):
                if "log_likelihood_diff" in comparison:
                    row["Log-Likelihood Diff"] = comparison["log_likelihood_diff"]
                
                if "aic_diff" in comparison:
                    row["AIC Diff"] = comparison["aic_diff"]
                
                if "bic_diff" in comparison:
                    row["BIC Diff"] = comparison["bic_diff"]
                
                if "preferred_by_aic" in comparison:
                    row["Preferred by AIC"] = comparison["preferred_by_aic"]
                
                if "preferred_by_bic" in comparison:
                    row["Preferred by BIC"] = comparison["preferred_by_bic"]
            
            comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)