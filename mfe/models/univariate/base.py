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


class VolatilityModel(VolatilityModelBase[UnivariateVolatilityParameters, UnivariateVolatilityResult, np.ndarray]):
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
            distribution: Literal("normal", "t", "skewed-t", "ged") = "normal",
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
            optimization_result = optimize.minimize(
                neg_loglikelihood,
                unconstrained_params,
                method=method,
                options=options
            )
            
            # Check convergence
            if not optimization_result.success:
                # Try alternative optimization methods if the first one fails
                alternative_methods = ["Nelder-Mead", "Powell", "BFGS"]
                for alt_method in alternative_methods:
                    if alt_method != method:
                        try:
                            optimization_result = optimize.minimize(
                                neg_loglikelihood,
                                unconstrained_params,
                                method=alt_method,
                                options=options
                            )
                            if optimization_result.success:
                                break
                        except Exception:
                            continue
            
            # If still not converged, raise error
            if not optimization_result.success:
                raise_convergence_error(
                    "Optimization failed to converge.",
                    iterations=optimization_result.get("nit", options["maxiter"]),
                    final_value=optimization_result.fun,
                    gradient_norm=np.linalg.norm(optimization_result.get("jac", np.array([np.nan]))),
                    details=optimization_result.message
                )
            
            # Transform parameters back to constrained space
            final_params = param_class.inverse_transform(optimization_result.x)
            
            # Compute conditional variances with final parameters
            sigma2 = self.compute_variance(final_params, data, backcast=backcast)
            
            # Store conditional variances
            self._conditional_variances = sigma2
            
            # Compute log-likelihood
            if distribution == "normal":
                loglikelihood = -0.5 * np.sum(np.log(sigma2) + data**2 / sigma2)
                loglikelihood -= 0.5 * len(data) * np.log(2 * np.pi)
            else:
                # For other distributions, negate the negative log-likelihood
                loglikelihood = -optimization_result.fun
            
            # Compute information criteria
            k = len(final_params.to_array())  # Number of parameters
            n = len(data)  # Number of observations
            aic = -2 * loglikelihood + 2 * k
            bic = -2 * loglikelihood + k * np.log(n)
            
            # Compute standard errors (using numerical Hessian)
            try:
                # Compute Hessian of negative log-likelihood
                hessian = optimize.approx_fprime(
                    optimization_result.x,
                    lambda x: optimize.approx_fprime(x, neg_loglikelihood, 1e-6),
                    1e-6
                )
                
                # Compute covariance matrix
                try:
                    cov_matrix = np.linalg.inv(hessian)
                    # Check if covariance matrix is positive definite
                    if np.any(np.diag(cov_matrix) <= 0):
                        raise np.linalg.LinAlgError("Covariance matrix is not positive definite")
                    
                    # Extract standard errors
                    std_errors = np.sqrt(np.diag(cov_matrix))
                    
                    # Transform standard errors to constrained space
                    # This is a simplification; proper transformation requires the Jacobian
                    # TODO: Implement proper standard error transformation
                    
                except np.linalg.LinAlgError:
                    warn_numeric(
                        "Could not compute standard errors: Hessian is singular.",
                        operation="Standard error computation",
                        issue="Singular Hessian"
                    )
                    std_errors = None
            except Exception as e:
                warn_numeric(
                    f"Could not compute standard errors: {str(e)}",
                    operation="Standard error computation",
                    issue="Computation error"
                )
                std_errors = None
            
            # Create result object
            result = UnivariateVolatilityResult(
                model_name=self.name,
                parameters=final_params,
                std_errors=std_errors,
                loglikelihood=loglikelihood,
                aic=aic,
                bic=bic,
                convergence=True,
                iterations=optimization_result.get("nit", 0),
                num_obs=len(data),
                variance_target=variance_targeting,
                distribution=distribution,
                optimization_result=optimization_result
            )
            
            # Store parameters and results
            self._parameters = final_params
            self._results = result
            self._fitted = True
            
            return result
            
        except ConvergenceError as e:
            # Re-raise convergence errors
            raise e
        except Exception as e:
            # Wrap other exceptions in EstimationError
            raise EstimationError(
                f"Error during model estimation: {str(e)}",
                model_type=self.name,
                estimation_method=method,
                issue=str(e)
            ) from e
    
    async def fit_async(self, 
                       data: np.ndarray, 
                       starting_values: Optional[Union[np.ndarray, UnivariateVolatilityParameters]] = None,
                       distribution: Literal("normal", "t", "skewed-t", "ged") = "normal",
                       variance_targeting: bool = False,
                       backcast: Optional[float] = None,
                       method: str = "SLSQP",
                       options: Optional[Dict[str, Any]] = None,
                       progress_callback: Optional[Callable[[float, str], None]] = None,
                       **kwargs: Any) -> UnivariateVolatilityResult:
        """Asynchronously fit the volatility model to the provided data.
        
        This method provides an asynchronous interface to the fit method,
        allowing for non-blocking model estimation in UI contexts.
        
        Args:
            data: The data to fit the model to (typically residuals)
            starting_values: Initial parameter values for optimization
            distribution: Error distribution assumption
            variance_targeting: Whether to use variance targeting
            backcast: Value to use for initializing the variance process
            method: Optimization method to use
            options: Additional options for the optimizer
            progress_callback: Optional callback function for progress updates
            **kwargs: Additional keyword arguments for model fitting
        
        Returns:
            UnivariateVolatilityResult: The model estimation results
        
        Raises:
            ValueError: If the data is invalid
            ConvergenceError: If the optimization fails to converge
            EstimationError: If there are other issues with model estimation
        """
        # Report initial progress
        if progress_callback:
            await progress_callback(0.0, "Starting model estimation...")
        
        # Run the fit method in a separate thread to avoid blocking
        loop = asyncio.get_event_loop()
        
        # Define a wrapper function that reports progress
        def fit_with_progress():
            result = self.fit(
                data, 
                starting_values, 
                distribution, 
                variance_targeting, 
                backcast, 
                method, 
                options, 
                **kwargs
            )
            return result
        
        # Run the fit method in a separate thread
        try:
            result = await loop.run_in_executor(None, fit_with_progress)
            
            # Report completion
            if progress_callback:
                await progress_callback(1.0, "Model estimation complete.")
            
            return result
        except Exception as e:
            # Report error
            if progress_callback:
                await progress_callback(1.0, f"Error: {str(e)}")
            raise
    
    def simulate(self, 
                n_periods: int, 
                burn: int = 500, 
                initial_values: Optional[np.ndarray] = None,
                random_state: Optional[Union[int, np.random.Generator]] = None,
                distribution: Literal("normal", "t", "skewed-t", "ged") = "normal",
                distribution_params: Optional[Dict[str, Any]] = None,
                **kwargs: Any) -> np.ndarray:
        """Simulate data from the volatility model.
        
        Args:
            n_periods: Number of periods to simulate
            burn: Number of initial observations to discard
            initial_values: Initial values for the simulation
            random_state: Random number generator or seed
            distribution: Error distribution to use for simulation
            distribution_params: Additional parameters for the error distribution
            **kwargs: Additional keyword arguments for simulation
        
        Returns:
            np.ndarray: Simulated data
        
        Raises:
            RuntimeError: If the model has not been fitted
            ValueError: If the simulation parameters are invalid
            SimulationError: If there are issues during simulation
        """
        if not self._fitted or self._parameters is None:
            raise_not_fitted_error(
                "Model must be fitted before simulation.",
                model_type=self.name,
                operation="simulate"
            )
        
        # Set up random number generator
        if isinstance(random_state, np.random.Generator):
            rng = random_state
        else:
            rng = np.random.default_rng(random_state)
        
        # Normalize distribution name
        distribution = distribution.lower()
        if distribution not in ["normal", "t", "skewed-t", "ged"]:
            raise ValueError(
                f"Unknown distribution: {distribution}. "
                f"Supported distributions are 'normal', 't', 'skewed-t', and 'ged'."
            )
        
        # Set up distribution parameters
        if distribution_params is None:
            distribution_params = {}
        
        # Total number of periods to simulate (including burn-in)
        total_periods = n_periods + burn
        
        try:
            # Initialize arrays for simulation
            innovations = np.zeros(total_periods)
            sigma2 = np.zeros(total_periods)
            
            # Set initial variance
            if initial_values is not None and len(initial_values) > 0:
                # Use provided initial values
                sigma2[0] = initial_values[0]**2 if len(initial_values.shape) == 1 else initial_values[0]
            else:
                # Use unconditional variance as initial value
                sigma2[0] = self._compute_unconditional_variance()
            
            # Generate random innovations based on distribution
            if distribution == "normal":
                std_innovations = rng.standard_normal(total_periods)
            elif distribution == "t":
                df = distribution_params.get("df", 5.0)
                std_innovations = rng.standard_t(df, total_periods)
                # Scale to have unit variance
                std_innovations = std_innovations * np.sqrt((df - 2) / df)
            elif distribution == "skewed-t":
                # TODO: Implement skewed t-distribution random generation
                raise NotImplementedError("skewed t-distribution not yet implemented")
            elif distribution == "ged":
                # TODO: Implement GED random generation
                raise NotImplementedError("GED not yet implemented")
            else:
                raise ValueError(f"Unknown distribution: {distribution}")
            
            # Generate first innovation
            innovations[0] = np.sqrt(sigma2[0]) * std_innovations[0]
            
            # Simulate the process
            for t in range(1, total_periods):
                # Compute conditional variance for time t
                sigma2[t] = self._simulate_variance(t, innovations[:t], sigma2[:t])
                
                # Generate innovation for time t
                innovations[t] = np.sqrt(sigma2[t]) * std_innovations[t]
            
            # Discard burn-in period
            return innovations[burn:]
            
        except Exception as e:
            # Wrap exceptions in SimulationError
            raise SimulationError(
                f"Error during simulation: {str(e)}",
                model_type=self.name,
                n_periods=n_periods,
                issue=str(e)
            ) from e
    
    async def simulate_async(self, 
                           n_periods: int, 
                           burn: int = 500, 
                           initial_values: Optional[np.ndarray] = None,
                           random_state: Optional[Union[int, np.random.Generator]] = None,
                           distribution: Literal("normal", "t", "skewed-t", "ged") = "normal",
                           distribution_params: Optional[Dict[str, Any]] = None,
                           progress_callback: Optional[Callable[[float, str], None]] = None,
                           **kwargs: Any) -> np.ndarray:
        """Asynchronously simulate data from the volatility model.
        
        This method provides an asynchronous interface to the simulate method,
        allowing for non-blocking simulation in UI contexts.
        
        Args:
            n_periods: Number of periods to simulate
            burn: Number of initial observations to discard
            initial_values: Initial values for the simulation
            random_state: Random number generator or seed
            distribution: Error distribution to use for simulation
            distribution_params: Additional parameters for the error distribution
            progress_callback: Optional callback function for progress updates
            **kwargs: Additional keyword arguments for simulation
        
        Returns:
            np.ndarray: Simulated data
        
        Raises:
            RuntimeError: If the model has not been fitted
            ValueError: If the simulation parameters are invalid
            SimulationError: If there are issues during simulation
        """
        # Report initial progress
        if progress_callback:
            await progress_callback(0.0, "Starting simulation...")
        
        # Run the simulate method in a separate thread to avoid blocking
        loop = asyncio.get_event_loop()
        
        # Define a wrapper function that reports progress
        def simulate_with_progress():
            result = self.simulate(
                n_periods, 
                burn, 
                initial_values, 
                random_state, 
                distribution, 
                distribution_params, 
                **kwargs
            )
            return result
        
        # Run the simulate method in a separate thread
        try:
            result = await loop.run_in_executor(None, simulate_with_progress)
            
            # Report completion
            if progress_callback:
                await progress_callback(1.0, "Simulation complete.")
            
            return result
        except Exception as e:
            # Report error
            if progress_callback:
                await progress_callback(1.0, f"Error: {str(e)}")
            raise
    
    def forecast(self, 
                steps: int, 
                data: Optional[np.ndarray] = None,
                method: Literal("analytic", "simulation") = "analytic",
                n_simulations: int = 1000,
                random_state: Optional[Union[int, np.random.Generator]] = None,
                distribution: Literal("normal", "t", "skewed-t", "ged") = "normal",
                distribution_params: Optional[Dict[str, Any]] = None,
                **kwargs: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate volatility forecasts from the fitted model.
        
        Args:
            steps: Number of steps to forecast
            data: Historical data to condition the forecast on (if different from fitting data)
            method: Forecasting method ('analytic' or 'simulation')
            n_simulations: Number of simulations for simulation-based forecasting
            random_state: Random number generator or seed for simulation
            distribution: Error distribution to use for simulation
            distribution_params: Additional parameters for the error distribution
            **kwargs: Additional keyword arguments for forecasting
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Point forecasts, lower bounds, and upper bounds
        
        Raises:
            RuntimeError: If the model has not been fitted
            ValueError: If the forecast parameters are invalid
        """
        if not self._fitted or self._parameters is None:
            raise_not_fitted_error(
                "Model must be fitted before forecasting.",
                model_type=self.name,
                operation="forecast"
            )
        
        # If data is provided, compute conditional variances for the data
        if data is not None:
            self.validate_data(data)
            sigma2 = self.compute_variance(self._parameters, data)
            last_variance = sigma2[-1]
        elif self._conditional_variances is not None:
            # Use the last conditional variance from the fitted model
            last_variance = self._conditional_variances[-1]
        else:
            # Use unconditional variance as fallback
            last_variance = self._compute_unconditional_variance()
        
        if method == "analytic":
            # Analytic forecasting (model-specific implementation)
            forecasts = self._forecast_analytic(steps, last_variance)
            
            # Compute forecast intervals (assuming normal distribution)
            # For volatility, we use log-normal confidence intervals
            alpha = 0.05  # 95% confidence interval
            z_value = stats.norm.ppf(1 - alpha / 2)
            
            # Approximate standard errors for volatility forecasts
            # This is a simplification; proper intervals require model-specific derivation
            std_errors = np.sqrt(forecasts) * 0.5  # Approximate standard error
            
            lower_bounds = forecasts * np.exp(-z_value * std_errors / forecasts)
            upper_bounds = forecasts * np.exp(z_value * std_errors / forecasts)
            
            return forecasts, lower_bounds, upper_bounds
            
        elif method == "simulation":
            # Simulation-based forecasting
            # Set up random number generator
            if isinstance(random_state, np.random.Generator):
                rng = random_state
            else:
                rng = np.random.default_rng(random_state)
            
            # Initialize arrays for simulation
            sim_forecasts = np.zeros((n_simulations, steps))
            
            # Run simulations
            for i in range(n_simulations):
                # Simulate future paths
                sim_data = self.simulate(
                    steps, 
                    burn=0, 
                    initial_values=np.array([np.sqrt(last_variance)]), 
                    random_state=rng, 
                    distribution=distribution, 
                    distribution_params=distribution_params
                )
                
                # Compute conditional variances for the simulated path
                sim_sigma2 = self.compute_variance(self._parameters, sim_data)
                
                # Store the forecasted variances
                sim_forecasts[i, :] = sim_sigma2
            
            # Compute point forecasts and intervals
            forecasts = np.mean(sim_forecasts, axis=0)
            lower_bounds = np.percentile(sim_forecasts, 2.5, axis=0)
            upper_bounds = np.percentile(sim_forecasts, 97.5, axis=0)
            
            return forecasts, lower_bounds, upper_bounds
            
        else:
            raise ValueError(
                f"Unknown forecasting method: {method}. "
                f"Supported methods are 'analytic' and 'simulation'."
            )
    
    async def forecast_async(self, 
                           steps: int, 
                           data: Optional[np.ndarray] = None,
                           method: Literal("analytic", "simulation") = "analytic",
                           n_simulations: int = 1000,
                           random_state: Optional[Union[int, np.random.Generator]] = None,
                           distribution: Literal("normal", "t", "skewed-t", "ged") = "normal",
                           distribution_params: Optional[Dict[str, Any]] = None,
                           progress_callback: Optional[Callable[[float, str], None]] = None,
                           **kwargs: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Asynchronously generate volatility forecasts from the fitted model.
        
        This method provides an asynchronous interface to the forecast method,
        allowing for non-blocking forecasting in UI contexts.
        
        Args:
            steps: Number of steps to forecast
            data: Historical data to condition the forecast on (if different from fitting data)
            method: Forecasting method ('analytic' or 'simulation')
            n_simulations: Number of simulations for simulation-based forecasting
            random_state: Random number generator or seed for simulation
            distribution: Error distribution to use for simulation
            distribution_params: Additional parameters for the error distribution
            progress_callback: Optional callback function for progress updates
            **kwargs: Additional keyword arguments for forecasting
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Point forecasts, lower bounds, and upper bounds
        
        Raises:
            RuntimeError: If the model has not been fitted
            ValueError: If the forecast parameters are invalid
        """
        # Report initial progress
        if progress_callback:
            await progress_callback(0.0, "Starting forecasting...")
        
        # Run the forecast method in a separate thread to avoid blocking
        loop = asyncio.get_event_loop()
        
        # Define a wrapper function that reports progress
        def forecast_with_progress():
            result = self.forecast(
                steps, 
                data, 
                method, 
                n_simulations, 
                random_state, 
                distribution, 
                distribution_params, 
                **kwargs
            )
            return result
        
        # Run the forecast method in a separate thread
        try:
            result = await loop.run_in_executor(None, forecast_with_progress)
            
            # Report completion
            if progress_callback:
                await progress_callback(1.0, "Forecasting complete.")
            
            return result
        except Exception as e:
            # Report error
            if progress_callback:
                await progress_callback(1.0, f"Error: {str(e)}")
            raise
    
    def _generate_starting_values(self, 
                                 data: np.ndarray, 
                                 variance_targeting: bool = False,
                                 backcast: Optional[float] = None) -> UnivariateVolatilityParameters:
        """Generate starting values for parameter estimation.
        
        This method should be implemented by subclasses to generate appropriate
        starting values for the specific model type.
        
        Args:
            data: Input data (typically residuals)
            variance_targeting: Whether to use variance targeting
            backcast: Value to use for initializing the variance process
        
        Returns:
            UnivariateVolatilityParameters: Starting parameter values
        
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("_generate_starting_values must be implemented by subclass")
    
    def _compute_unconditional_variance(self) -> float:
        """Compute the unconditional variance of the process.
        
        This method should be implemented by subclasses to compute the
        unconditional variance based on the model parameters.
        
        Returns:
            float: Unconditional variance
        
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("_compute_unconditional_variance must be implemented by subclass")
    
    def _simulate_variance(self, 
                          t: int, 
                          innovations: np.ndarray, 
                          sigma2: np.ndarray) -> float:
        """Simulate the conditional variance for time t.
        
        This method should be implemented by subclasses to compute the
        conditional variance for time t based on past innovations and variances.
        
        Args:
            t: Time index
            innovations: Past innovations up to t-1
            sigma2: Past conditional variances up to t-1
        
        Returns:
            float: Conditional variance for time t
        
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("_simulate_variance must be implemented by subclass")
    
    def _forecast_analytic(self, 
                          steps: int, 
                          last_variance: float) -> np.ndarray:
        """Generate analytic volatility forecasts.
        
        This method should be implemented by subclasses to compute analytic
        volatility forecasts based on the model parameters.
        
        Args:
            steps: Number of steps to forecast
            last_variance: Last observed conditional variance
        
        Returns:
            np.ndarray: Volatility forecasts
        
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("_forecast_analytic must be implemented by subclass")
