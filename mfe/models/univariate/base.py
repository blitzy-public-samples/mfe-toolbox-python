# mfe/models/univariate/base.py
"""
Base classes for univariate volatility models.

This module defines the abstract base classes and shared functionality for all
univariate volatility models in the MFE Toolbox. It establishes the common interface
and implementation patterns that ensure consistent behavior across different model types.

The module provides a comprehensive class hierarchy with a base VolatilityModel class
that implements shared functionality like parameter validation, transformation, and
result presentation. Concrete model implementations (GARCH, EGARCH, etc.) inherit from
this base class and override model-specific methods.
"""

import abc
import asyncio
import warnings
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union, cast, overload
)
import numpy as np
import pandas as pd
from scipy import optimize, stats

from mfe.core.base import ModelBase, VolatilityModelBase
from mfe.core.parameters import (
    ParameterBase, UnivariateVolatilityParameters, ParameterError,
    validate_positive, validate_non_negative, validate_probability, validate_range
)
from mfe.core.exceptions import (
    MFEError, ConvergenceError, NumericError, EstimationError, SimulationError,
    NotFittedError, raise_convergence_error, raise_numeric_error, raise_not_fitted_error,
    warn_convergence, warn_numeric, warn_model
)


@dataclass
class UnivariateVolatilityResult:
    """Result container for univariate volatility model estimation.
    
    This class stores the results of univariate volatility model estimation,
    including parameter estimates, standard errors, and diagnostic statistics.
    
    Attributes:
        model_name: Name of the model
        parameters: Estimated model parameters
        std_errors: Standard errors of parameter estimates
        loglikelihood: Log-likelihood value at the optimum
        aic: Akaike Information Criterion
        bic: Bayesian Information Criterion
        convergence: Whether the optimization converged
        iterations: Number of iterations performed during optimization
        num_obs: Number of observations used in estimation
        variance_target: Whether variance targeting was used
        distribution: Name of the error distribution used
        optimization_result: Full optimization result object
    """
    
    model_name: str
    parameters: UnivariateVolatilityParameters
    std_errors: Optional[np.ndarray] = None
    loglikelihood: Optional[float] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    convergence: bool = True
    iterations: int = 0
    num_obs: Optional[int] = None
    variance_target: bool = False
    distribution: str = "Normal"
    optimization_result: Optional[Any] = None
    
    def __post_init__(self) -> None:
        """Validate result object after initialization."""
        if not self.convergence:
            warnings.warn(
                f"Model {self.model_name} did not converge after {self.iterations} iterations.",
                UserWarning
            )
    
    def summary(self) -> str:
        """Generate a text summary of the model results.
        
        Returns:
            str: A formatted string containing the model results summary.
        """
        header = f"Model: {self.model_name}\n"
        header += "=" * (len(header) - 1) + "\n\n"
        
        # Distribution information
        header += f"Distribution: {self.distribution}\n"
        header += f"Number of observations: {self.num_obs}\n"
        header += f"Variance targeting: {'Yes' if self.variance_target else 'No'}\n\n"
        
        # Convergence information
        convergence_info = f"Convergence: {'Yes' if self.convergence else 'No'}\n"
        convergence_info += f"Iterations: {self.iterations}\n\n"
        
        # Parameter estimates
        param_dict = self.parameters.to_dict()
        param_table = "Parameter Estimates:\n"
        param_table += "-" * 60 + "\n"
        param_table += f"{'Parameter':<15} {'Estimate':>12} {'Std. Error':>12} {'t-stat':>10} {'p-value':>10}\n"
        param_table += "-" * 60 + "\n"
        
        for i, (name, value) in enumerate(param_dict.items()):
            if self.std_errors is not None and i < len(self.std_errors):
                std_err = self.std_errors[i]
                t_stat = value / std_err if std_err > 0 else np.nan
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), self.num_obs - len(param_dict))) if not np.isnan(t_stat) else np.nan
                param_table += f"{name:<15} {value:>12.6f} {std_err:>12.6f} {t_stat:>10.4f} {p_value:>10.4f}\n"
            else:
                param_table += f"{name:<15} {value:>12.6f} {'N/A':>12} {'N/A':>10} {'N/A':>10}\n"
        
        param_table += "-" * 60 + "\n\n"
        
        # Fit statistics
        fit_stats = "Model Fit:\n"
        fit_stats += "-" * 30 + "\n"
        if self.loglikelihood is not None:
            fit_stats += f"Log-likelihood: {self.loglikelihood:.6f}\n"
        if self.aic is not None:
            fit_stats += f"AIC: {self.aic:.6f}\n"
        if self.bic is not None:
            fit_stats += f"BIC: {self.bic:.6f}\n"
        fit_stats += "-" * 30 + "\n"
        
        return header + convergence_info + param_table + fit_stats
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result object to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the result object.
        """
        result_dict = {
            "model_name": self.model_name,
            "parameters": self.parameters.to_dict(),
            "convergence": self.convergence,
            "iterations": self.iterations,
            "loglikelihood": self.loglikelihood,
            "aic": self.aic,
            "bic": self.bic,
            "num_obs": self.num_obs,
            "variance_target": self.variance_target,
            "distribution": self.distribution
        }
        
        if self.std_errors is not None:
            result_dict["std_errors"] = self.std_errors.tolist()
        
        return result_dict


class VolatilityModel(VolatilityModelBase[UnivariateVolatilityParameters, UnivariateVolatilityResult]):
    """Abstract base class for univariate volatility models.
    
    This class extends the VolatilityModelBase class to provide specialized functionality
    for univariate volatility models, including methods for parameter estimation,
    simulation, and forecasting.
    
    Attributes:
        name: Model name
        _parameters: Model parameters if set
        _results: Estimation results if the model has been fitted
        _conditional_variances: Conditional variances if the model has been fitted
    """
    
    def __init__(self, 
                 parameters: Optional[UnivariateVolatilityParameters] = None, 
                 name: str = "VolatilityModel") -> None:
        """Initialize the volatility model.
        
        Args:
            parameters: Pre-specified model parameters if available
            name: A descriptive name for the model
        """
        super().__init__(name=name)
        self._parameters = parameters
        self._fitted = parameters is not None
    
    @property
    def parameters(self) -> Optional[UnivariateVolatilityParameters]:
        """Get the model parameters.
        
        Returns:
            Optional[UnivariateVolatilityParameters]: The model parameters if set, None otherwise
        """
        return self._parameters
    
    @abc.abstractmethod
    def compute_variance(self, 
                         parameters: UnivariateVolatilityParameters, 
                         data: np.ndarray, 
                         sigma2: Optional[np.ndarray] = None,
                         backcast: Optional[float] = None) -> np.ndarray:
        """Compute conditional variances for the given parameters and data.
        
        This method must be implemented by all subclasses to compute the
        conditional variances based on the model parameters and data.
        
        Args:
            parameters: Model parameters
            data: Input data (typically residuals)
            sigma2: Pre-allocated array for conditional variances
            backcast: Value to use for initializing the variance process
        
        Returns:
            np.ndarray: Conditional variances
        
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("compute_variance must be implemented by subclass")
    
    @abc.abstractmethod
    def parameter_class(self) -> Type[UnivariateVolatilityParameters]:
        """Get the parameter class for this model.
        
        Returns:
            Type[UnivariateVolatilityParameters]: The parameter class for this model
        
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("parameter_class must be implemented by subclass")
    
    def fit(self, 
            data: np.ndarray, 
            starting_values: Optional[Union[np.ndarray, UnivariateVolatilityParameters]] = None,
            distribution: Literal["normal", "t", "skewed-t", "ged"] = "normal",
            variance_targeting: bool = False,
            backcast: Optional[float] = None,
            method: str = "SLSQP",
            options: Optional[Dict[str, Any]] = None,
            **kwargs: Any) -> UnivariateVolatilityResult:
        """Fit the volatility model to the provided data.
        
        Args:
            data: The data to fit the model to (typically residuals)
            starting_values: Initial parameter values for optimization
            distribution: Error distribution assumption
            variance_targeting: Whether to use variance targeting
            backcast: Value to use for initializing the variance process
            method: Optimization method to use
            options: Additional options for the optimizer
            **kwargs: Additional keyword arguments for model fitting
        
        Returns:
            UnivariateVolatilityResult: The model estimation results
        
        Raises:
            ValueError: If the data is invalid
            ConvergenceError: If the optimization fails to converge
            EstimationError: If there are other issues with model estimation
        """
        # Validate input data
        self.validate_data(data)
        
        # Normalize distribution name
        distribution = distribution.lower()
        if distribution not in ["normal", "t", "skewed-t", "ged"]:
            raise ValueError(
                f"Unknown distribution: {distribution}. "
                f"Supported distributions are 'normal', 't', 'skewed-t', and 'ged'."
            )
        
        # Get parameter class for this model
        param_class = self.parameter_class()
        
        # Compute backcast value if not provided
        if backcast is None:
            backcast = np.mean(data**2)
        
        # Initialize parameters
        if starting_values is None:
            # Generate starting values based on model type
            starting_values = self._generate_starting_values(data, variance_targeting, backcast)
        elif isinstance(starting_values, np.ndarray):
            # Convert array to parameter object
            starting_values = param_class.from_array(starting_values)
        
        # Validate starting values
        try:
            starting_values.validate()
        except ParameterError as e:
            # If starting values are invalid, try to generate new ones
            warnings.warn(
                f"Invalid starting values: {e}. Generating new starting values.",
                UserWarning
            )
            starting_values = self._generate_starting_values(data, variance_targeting, backcast)
        
        # Transform parameters to unconstrained space for optimization
        unconstrained_params = starting_values.transform()
        
        # Set up optimization options
        if options is None:
            options = {}
        default_options = {
            "maxiter": 1000,
            "disp": False,
            "ftol": 1e-8
        }
        for key, value in default_options.items():
            if key not in options:
                options[key] = value
        
        # Define the negative log-likelihood function for optimization
        def neg_loglikelihood(params: np.ndarray) -> float:
            try:
                # Transform parameters back to constrained space
                model_params = param_class.inverse_transform(params)
                
                # Compute conditional variances
                sigma2 = self.compute_variance(model_params, data, backcast=backcast)
                
                # Compute log-likelihood based on distribution
                if distribution == "normal":
                    ll = -0.5 * np.sum(np.log(sigma2) + data**2 / sigma2)
                elif distribution == "t":
                    # TODO: Implement t-distribution log-likelihood
                    raise NotImplementedError("t-distribution not yet implemented")
                elif distribution == "skewed-t":
                    # TODO: Implement skewed t-distribution log-likelihood
                    raise NotImplementedError("skewed t-distribution not yet implemented")
                elif distribution == "ged":
                    # TODO: Implement GED log-likelihood
                    raise NotImplementedError("GED not yet implemented")
                else:
                    raise ValueError(f"Unknown distribution: {distribution}")
                
                # Add constant term for normal distribution
                if distribution == "normal":
                    ll -= 0.5 * len(data) * np.log(2 * np.pi)
                
                return -ll  # Return negative log-likelihood for minimization
            
            except (ValueError, ParameterError, NumericError) as e:
                # Return a large value if parameters are invalid
                return 1e10
        
        # Run optimization
        try:
            optimization = minimize(
                neg_loglikelihood,
                x0=unconstrained_params,
                method=method,
                bounds=None,
                constraints=None,
                options=options
            )
            params = optimization.x
            success = optimization.success
            iterations = optimization.nit if hasattr(optimization, 'nit') else 0
        except Exception as e:
            warnings.warn(f"Optimization failed: {str(e)}")
            params = unconstrained_params
            success = False
            iterations = 0
            optimization = None
        
        # Transform parameters back to constrained space
        constrained_params = param_class.inverse_transform(params)
        
        # Compute standard errors
        std_errors = self._compute_standard_errors(constrained_params, data)
        
        # Compute log-likelihood
        loglikelihood = -neg_loglikelihood(params)
        
        # Compute AIC and BIC
        aic = 2 * len(params) - 2 * loglikelihood
        bic = len(params) * np.log(len(data)) - 2 * loglikelihood
        
        # Create result object
        result = UnivariateVolatilityResult(
            model_name=self.name,
            parameters=constrained_params,
            std_errors=std_errors,
            loglikelihood=loglikelihood,
            aic=aic,
            bic=bic,
            convergence=success,
            iterations=iterations,
            num_obs=len(data),
            variance_target=variance_targeting,
            distribution=distribution
        )
        
        # Set results in model
        self._results = result
        self._conditional_variances = self.compute_variance(constrained_params, data)
        
        return result


# Aliases for backward compatibility
VolatilityModelResult = UnivariateVolatilityResult
VolatilityForecast = dict  # Placeholder for a proper forecast class
VolatilityParameters = UnivariateVolatilityParameters