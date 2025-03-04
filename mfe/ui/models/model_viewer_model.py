'''
Model viewer model for displaying ARMAX model results.

This module implements the data model for the ARMAX model result viewer, which
manages the representation of estimated ARMAX models, their parameters, statistics,
and visual elements. It handles the conversion of model parameters to LaTeX equations,
statistical summaries, and structured result representation independent of the UI
presentation.

The ModelViewerModel class provides a clean separation between the data representation
and the UI components, following the Model-View-Controller (MVC) pattern.
'''

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Tuple, cast
import numpy as np
import pandas as pd

from mfe.core.exceptions import MFEError, ModelSpecificationError
from mfe.ui.models.armax_model import ARMAXModel, ARMAXModelResults

# Set up module-level logger
logger = logging.getLogger("mfe.ui.models.model_viewer_model")


@dataclass
class ModelParameter:
    """Container for a model parameter with its statistical properties.
    
    This dataclass encapsulates a model parameter along with its statistical
    properties such as standard error, t-statistic, and p-value.
    
    Attributes:
        name: Name of the parameter
        value: Estimated value of the parameter
        std_error: Standard error of the parameter estimate
        t_stat: t-statistic for the parameter
        p_value: p-value for the parameter
        significant: Whether the parameter is statistically significant
    """
    
    name: str
    value: float
    std_error: Optional[float] = None
    t_stat: Optional[float] = None
    p_value: Optional[float] = None
    significant: Optional[bool] = None
    
    def __post_init__(self) -> None:
        """Validate and compute derived properties after initialization."""
        # Compute significance if p-value is available
        if self.p_value is not None and self.significant is None:
            self.significant = self.p_value < 0.05
    
    def to_latex(self) -> str:
        """Convert the parameter to a LaTeX representation.
        
        Returns:
            str: LaTeX representation of the parameter
        """
        # Format the parameter value with appropriate sign
        if self.value >= 0:
            value_str = f"{self.value:.4f}"
        else:
            value_str = f"({self.value:.4f})"
        
        return value_str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the parameter to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the parameter
        """
        return {
            "name": self.name,
            "value": self.value,
            "std_error": self.std_error,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "significant": self.significant
        }


@dataclass
class ModelStatistic:
    """Container for a model statistic.
    
    This dataclass encapsulates a model statistic such as AIC, BIC, or log-likelihood.
    
    Attributes:
        name: Name of the statistic
        value: Value of the statistic
        description: Description of the statistic
    """
    
    name: str
    value: float
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the statistic to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the statistic
        """
        return {
            "name": self.name,
            "value": self.value,
            "description": self.description
        }


@dataclass
class ModelViewerPage:
    """Container for a page of model information in the viewer.
    
    This dataclass represents a page of information in the model viewer,
    which can contain parameters, statistics, or other model information.
    
    Attributes:
        title: Title of the page
        parameters: List of parameters on the page
        statistics: List of statistics on the page
        equation: LaTeX equation for the page
        description: Description of the page content
    """
    
    title: str
    parameters: List[ModelParameter] = field(default_factory=list)
    statistics: List[ModelStatistic] = field(default_factory=list)
    equation: Optional[str] = None
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the page to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the page
        """
        return {
            "title": self.title,
            "parameters": [param.to_dict() for param in self.parameters],
            "statistics": [stat.to_dict() for stat in self.statistics],
            "equation": self.equation,
            "description": self.description
        }


class ModelViewerModel:
    """Data model for the ARMAX model result viewer.
    
    This class manages the representation of estimated ARMAX models, their parameters,
    statistics, and visual elements. It handles the conversion of model parameters to
    LaTeX equations, statistical summaries, and structured result representation
    independent of the UI presentation.
    
    Attributes:
        armax_model: The ARMAX model being viewed
        pages: List of pages in the viewer
        current_page_index: Index of the currently displayed page
        model_equation: LaTeX representation of the model equation
        parameter_table: DataFrame containing parameter estimates and statistics
        statistics_table: DataFrame containing model statistics
    """
    
    def __init__(self, armax_model: Optional[ARMAXModel] = None) -> None:
        """Initialize the model viewer model.
        
        Args:
            armax_model: The ARMAX model to view (optional)
        """
        self.armax_model: Optional[ARMAXModel] = armax_model
        self.pages: List[ModelViewerPage] = []
        self.current_page_index: int = 0
        self.model_equation: Optional[str] = None
        self.parameter_table: Optional[pd.DataFrame] = None
        self.statistics_table: Optional[pd.DataFrame] = None
        
        # Initialize the model if provided
        if armax_model is not None and armax_model.is_fitted and armax_model.results is not None:
            self.initialize_from_model(armax_model)
        
        logger.debug("ModelViewerModel initialized")
    
    def initialize_from_model(self, armax_model: ARMAXModel) -> None:
        """Initialize the viewer model from an ARMAX model.
        
        Args:
            armax_model: The ARMAX model to view
            
        Raises:
            ModelSpecificationError: If the model is not fitted
        """
        if not armax_model.is_fitted or armax_model.results is None:
            raise ModelSpecificationError(
                "Model must be fitted before viewing results",
                model_type="ARMAX",
                operation="view_results"
            )
        
        self.armax_model = armax_model
        self.pages = []
        self.current_page_index = 0
        
        # Create model equation
        self.model_equation = self._create_model_equation(armax_model.results)
        
        # Create parameter table
        self.parameter_table = self._create_parameter_table(armax_model.results)
        
        # Create statistics table
        self.statistics_table = self._create_statistics_table(armax_model.results)
        
        # Create pages
        self._create_pages()
        
        logger.debug("ModelViewerModel initialized from ARMAX model")
    
    def _create_model_equation(self, results: ARMAXModelResults) -> str:
        """Create a LaTeX representation of the model equation.
        
        Args:
            results: ARMAX model results
            
        Returns:
            str: LaTeX representation of the model equation
        """
        if self.armax_model is None:
            return ""
        
        # Start with the dependent variable
        data_name = self.armax_model.data_name or "y"
        equation = f"{data_name}_t = "
        
        # Add constant term if included
        if self.armax_model.include_constant and results.constant is not None:
            if results.constant >= 0:
                equation += f"{results.constant:.4f} "
            else:
                equation += f"({results.constant:.4f}) "
        
        # Add AR terms
        if self.armax_model.ar_order > 0 and results.ar_params is not None:
            for i, param in enumerate(results.ar_params):
                if i > 0 or self.armax_model.include_constant:
                    if param >= 0:
                        equation += f"+ {param:.4f} {data_name}_{{t-{i+1}}} "
                    else:
                        equation += f"- {abs(param):.4f} {data_name}_{{t-{i+1}}} "
                else:
                    if param >= 0:
                        equation += f"{param:.4f} {data_name}_{{t-{i+1}}} "
                    else:
                        equation += f"({param:.4f}) {data_name}_{{t-{i+1}}} "
        
        # Add exogenous variables
        if self.armax_model.exog_variables and results.exog_params is not None:
            for i, (var, param) in enumerate(zip(self.armax_model.exog_variables, results.exog_params)):
                if i > 0 or self.armax_model.ar_order > 0 or self.armax_model.include_constant:
                    if param >= 0:
                        equation += f"+ {param:.4f} {var}_t "
                    else:
                        equation += f"- {abs(param):.4f} {var}_t "
                else:
                    if param >= 0:
                        equation += f"{param:.4f} {var}_t "
                    else:
                        equation += f"({param:.4f}) {var}_t "
        
        # Add error term
        equation += "+ \varepsilon_t "
        
        # Add MA terms
        if self.armax_model.ma_order > 0 and results.ma_params is not None:
            for i, param in enumerate(results.ma_params):
                if param >= 0:
                    equation += f"+ {param:.4f} \varepsilon_{{t-{i+1}}} "
                else:
                    equation += f"- {abs(param):.4f} \varepsilon_{{t-{i+1}}} "
        
        return equation
    
    def _create_parameter_table(self, results: ARMAXModelResults) -> pd.DataFrame:
        """Create a DataFrame containing parameter estimates and statistics.
        
        Args:
            results: ARMAX model results
            
        Returns:
            pd.DataFrame: Parameter table
        """
        if results.parameter_table is not None:
            return results.parameter_table
        
        # Create parameter table manually if not available in results
        param_names = []
        param_values = []
        param_std_errs = []
        param_t_stats = []
        param_p_values = []
        
        # Add constant if included
        if self.armax_model and self.armax_model.include_constant and results.constant is not None:
            param_names.append("Constant")
            param_values.append(results.constant)
            
            # Add placeholder for standard error, t-stat, and p-value if not available
            param_std_errs.append(np.nan)
            param_t_stats.append(np.nan)
            param_p_values.append(np.nan)
        
        # Add AR parameters
        if results.ar_params is not None:
            for i, param in enumerate(results.ar_params):
                param_names.append(f"AR({i+1})")
                param_values.append(param)
                
                # Add placeholder for standard error, t-stat, and p-value if not available
                param_std_errs.append(np.nan)
                param_t_stats.append(np.nan)
                param_p_values.append(np.nan)
        
        # Add MA parameters
        if results.ma_params is not None:
            for i, param in enumerate(results.ma_params):
                param_names.append(f"MA({i+1})")
                param_values.append(param)
                
                # Add placeholder for standard error, t-stat, and p-value if not available
                param_std_errs.append(np.nan)
                param_t_stats.append(np.nan)
                param_p_values.append(np.nan)
        
        # Add exogenous parameters
        if self.armax_model and self.armax_model.exog_variables and results.exog_params is not None:
            for i, (var, param) in enumerate(zip(self.armax_model.exog_variables, results.exog_params)):
                param_names.append(var)
                param_values.append(param)
                
                # Add placeholder for standard error, t-stat, and p-value if not available
                param_std_errs.append(np.nan)
                param_t_stats.append(np.nan)
                param_p_values.append(np.nan)
        
        # Create parameter table as DataFrame
        return pd.DataFrame({
            'Parameter': param_names,
            'Estimate': param_values,
            'Std. Error': param_std_errs,
            't-Statistic': param_t_stats,
            'p-Value': param_p_values
        })
    
    def _create_statistics_table(self, results: ARMAXModelResults) -> pd.DataFrame:
        """Create a DataFrame containing model statistics.
        
        Args:
            results: ARMAX model results
            
        Returns:
            pd.DataFrame: Statistics table
        """
        stat_names = []
        stat_values = []
        stat_descriptions = []
        
        # Add log-likelihood
        if results.log_likelihood is not None:
            stat_names.append("Log-Likelihood")
            stat_values.append(results.log_likelihood)
            stat_descriptions.append("Log-likelihood value at the optimum")
        
        # Add AIC
        if results.aic is not None:
            stat_names.append("AIC")
            stat_values.append(results.aic)
            stat_descriptions.append("Akaike Information Criterion (lower values indicate better fit)")
        
        # Add BIC
        if results.bic is not None:
            stat_names.append("BIC")
            stat_values.append(results.bic)
            stat_descriptions.append("Bayesian Information Criterion (lower values indicate better fit)")
        
        # Add HQIC if available
        if results.hqic is not None:
            stat_names.append("HQIC")
            stat_values.append(results.hqic)
            stat_descriptions.append("Hannan-Quinn Information Criterion (lower values indicate better fit)")
        
        # Add residual variance
        if results.sigma2 is not None:
            stat_names.append("Residual Variance")
            stat_values.append(results.sigma2)
            stat_descriptions.append("Estimated variance of the residuals")
        
        # Add number of observations
        if self.armax_model and self.armax_model.data is not None:
            stat_names.append("Observations")
            stat_values.append(len(self.armax_model.data))
            stat_descriptions.append("Number of observations used in estimation")
        
        # Add number of parameters
        n_params = 0
        if self.armax_model:
            if self.armax_model.include_constant:
                n_params += 1
            n_params += self.armax_model.ar_order
            n_params += self.armax_model.ma_order
            n_params += len(self.armax_model.exog_variables)
        
        stat_names.append("Parameters")
        stat_values.append(n_params)
        stat_descriptions.append("Number of estimated parameters")
        
        # Add degrees of freedom
        if self.armax_model and self.armax_model.data is not None:
            df = len(self.armax_model.data) - n_params
            stat_names.append("Degrees of Freedom")
            stat_values.append(df)
            stat_descriptions.append("Degrees of freedom (observations - parameters)")
        
        # Add convergence information
        stat_names.append("Convergence")
        stat_values.append("Yes" if results.convergence else "No")
        stat_descriptions.append("Whether the optimization algorithm converged")
        
        # Add number of iterations
        stat_names.append("Iterations")
        stat_values.append(results.iterations)
        stat_descriptions.append("Number of iterations used in optimization")
        
        # Create statistics table as DataFrame
        return pd.DataFrame({
            'Statistic': stat_names,
            'Value': stat_values,
            'Description': stat_descriptions
        })
    
    def _create_pages(self) -> None:
        """Create pages for the model viewer."""
        if self.armax_model is None or self.armax_model.results is None:
            return
        
        # Create main page with model equation and parameters
        main_page = ModelViewerPage(
            title="Model Overview",
            equation=self.model_equation,
            description="Overview of the estimated ARMAX model"
        )
        
        # Add parameters to the main page
        if self.parameter_table is not None:
            for _, row in self.parameter_table.iterrows():
                param = ModelParameter(
                    name=row['Parameter'],
                    value=row['Estimate'],
                    std_error=row['Std. Error'] if not np.isnan(row['Std. Error']) else None,
                    t_stat=row['t-Statistic'] if not np.isnan(row['t-Statistic']) else None,
                    p_value=row['p-Value'] if not np.isnan(row['p-Value']) else None
                )
                main_page.parameters.append(param)
        
        # Add statistics to the main page
        if self.statistics_table is not None:
            for _, row in self.statistics_table.iterrows():
                # Convert value to float if possible
                try:
                    value = float(row['Value'])
                except (ValueError, TypeError):
                    value = 0.0  # Default value for non-numeric statistics
                
                stat = ModelStatistic(
                    name=row['Statistic'],
                    value=value,
                    description=row['Description']
                )
                main_page.statistics.append(stat)
        
        self.pages.append(main_page)
        
        # Create diagnostic page if diagnostic tests are available
        if self.armax_model.results.diagnostic_tests:
            diagnostic_page = ModelViewerPage(
                title="Diagnostic Tests",
                description="Statistical tests for model diagnostics"
            )
            
            # Add diagnostic tests as statistics
            for test_name, test_result in self.armax_model.results.diagnostic_tests.items():
                if isinstance(test_result, dict):
                    # For tests with multiple statistics
                    for key, value in test_result.items():
                        if key != "Lags":  # Skip lag information
                            if isinstance(value, list):
                                # For lists of values (e.g., Q-statistics at different lags)
                                for i, val in enumerate(value):
                                    lag = test_result.get("Lags", [])[i] if "Lags" in test_result and i < len(test_result["Lags"]) else i + 1
                                    stat = ModelStatistic(
                                        name=f"{test_name} ({key}, lag={lag})",
                                        value=val,
                                        description=f"{test_name} {key} at lag {lag}"
                                    )
                                    diagnostic_page.statistics.append(stat)
                            else:
                                # For single values
                                stat = ModelStatistic(
                                    name=f"{test_name} ({key})",
                                    value=value,
                                    description=f"{test_name} {key}"
                                )
                                diagnostic_page.statistics.append(stat)
                else:
                    # For tests with a single statistic
                    stat = ModelStatistic(
                        name=test_name,
                        value=test_result,
                        description=f"{test_name} test statistic"
                    )
                    diagnostic_page.statistics.append(stat)
            
            self.pages.append(diagnostic_page)
        
        logger.debug(f"Created {len(self.pages)} pages for model viewer")
    
    def get_current_page(self) -> Optional[ModelViewerPage]:
        """Get the currently displayed page.
        
        Returns:
            Optional[ModelViewerPage]: The current page, or None if no pages are available
        """
        if not self.pages:
            return None
        
        return self.pages[self.current_page_index]
    
    def next_page(self) -> Optional[ModelViewerPage]:
        """Move to the next page and return it.
        
        Returns:
            Optional[ModelViewerPage]: The next page, or None if already at the last page
        """
        if not self.pages or self.current_page_index >= len(self.pages) - 1:
            return None
        
        self.current_page_index += 1
        return self.get_current_page()
    
    def previous_page(self) -> Optional[ModelViewerPage]:
        """Move to the previous page and return it.
        
        Returns:
            Optional[ModelViewerPage]: The previous page, or None if already at the first page
        """
        if not self.pages or self.current_page_index <= 0:
            return None
        
        self.current_page_index -= 1
        return self.get_current_page()
    
    def go_to_page(self, index: int) -> Optional[ModelViewerPage]:
        """Go to a specific page by index.
        
        Args:
            index: Index of the page to go to
            
        Returns:
            Optional[ModelViewerPage]: The requested page, or None if the index is invalid
        """
        if not self.pages or index < 0 or index >= len(self.pages):
            return None
        
        self.current_page_index = index
        return self.get_current_page()
    
    def get_page_count(self) -> int:
        """Get the total number of pages.
        
        Returns:
            int: Number of pages
        """
        return len(self.pages)
    
    def get_current_page_index(self) -> int:
        """Get the index of the current page.
        
        Returns:
            int: Current page index
        """
        return self.current_page_index
    
    def get_model_equation(self) -> str:
        """Get the LaTeX representation of the model equation.
        
        Returns:
            str: LaTeX representation of the model equation
        """
        return self.model_equation or ""
    
    def get_parameter_table(self) -> Optional[pd.DataFrame]:
        """Get the parameter table.
        
        Returns:
            Optional[pd.DataFrame]: Parameter table, or None if not available
        """
        return self.parameter_table
    
    def get_statistics_table(self) -> Optional[pd.DataFrame]:
        """Get the statistics table.
        
        Returns:
            Optional[pd.DataFrame]: Statistics table, or None if not available
        """
        return self.statistics_table
    
    def get_parameters(self) -> List[ModelParameter]:
        """Get all parameters from all pages.
        
        Returns:
            List[ModelParameter]: List of all parameters
        """
        parameters = []
        for page in self.pages:
            parameters.extend(page.parameters)
        return parameters
    
    def get_statistics(self) -> List[ModelStatistic]:
        """Get all statistics from all pages.
        
        Returns:
            List[ModelStatistic]: List of all statistics
        """
        statistics = []
        for page in self.pages:
            statistics.extend(page.statistics)
        return statistics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the model viewer model to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the model viewer model
        """
        return {
            "model_equation": self.model_equation,
            "current_page_index": self.current_page_index,
            "pages": [page.to_dict() for page in self.pages]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelViewerModel':
        """Create a model viewer model from a dictionary.
        
        Args:
            data: Dictionary representation of the model viewer model
            
        Returns:
            ModelViewerModel: Model viewer model
        """
        model = cls()
        model.model_equation = data.get("model_equation")
        model.current_page_index = data.get("current_page_index", 0)
        
        # Create pages
        for page_data in data.get("pages", []):
            page = ModelViewerPage(
                title=page_data.get("title", ""),
                equation=page_data.get("equation"),
                description=page_data.get("description")
            )
            
            # Add parameters
            for param_data in page_data.get("parameters", []):
                param = ModelParameter(
                    name=param_data.get("name", ""),
                    value=param_data.get("value", 0.0),
                    std_error=param_data.get("std_error"),
                    t_stat=param_data.get("t_stat"),
                    p_value=param_data.get("p_value"),
                    significant=param_data.get("significant")
                )
                page.parameters.append(param)
            
            # Add statistics
            for stat_data in page_data.get("statistics", []):
                stat = ModelStatistic(
                    name=stat_data.get("name", ""),
                    value=stat_data.get("value", 0.0),
                    description=stat_data.get("description")
                )
                page.statistics.append(stat)
            
            model.pages.append(page)
        
        return model