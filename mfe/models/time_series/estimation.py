'''
Specialized estimation methods for time series models in finance.

This module implements various estimation methods for time series models,
including maximum likelihood estimation (MLE), conditional sum-of-squares (CSS),
and robust estimation approaches. It provides both synchronous and asynchronous
interfaces for model estimation, with support for different optimization algorithms
and error distributions.

The module leverages Numba's just-in-time compilation for performance-critical
operations, particularly likelihood evaluation and gradient computation. It also
implements robust optimization strategies with fallback mechanisms to handle
convergence issues.

Key features:
- Maximum likelihood estimation for ARMA/ARMAX models
- Conditional sum-of-squares estimation
- Robust parameter estimation with automatic starting value selection
- Numba-accelerated likelihood functions for performance
- Asynchronous estimation support for responsive UIs
- Comprehensive parameter validation and error handling
- Progress reporting for long-running estimations
'''

import asyncio
import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Literal, Optional, Protocol, Tuple, 
    Type, TypeVar, Union, cast, overload
)

import numpy as np
import pandas as pd
from scipy import optimize, stats
import numba
from numba import jit

from mfe.core.base import ModelBase
from mfe.core.parameters import (
    ParameterBase, TimeSeriesParameters, ARMAParameters,
    transform_positive, inverse_transform_positive,
    transform_probability, inverse_transform_probability
)
from mfe.core.exceptions import (
    ParameterError, DimensionError, ConvergenceError, NumericError,
    EstimationError, warn_convergence, warn_numeric, warn_model
)
from mfe.models.time_series.base import TimeSeriesModel, TimeSeriesConfig, TimeSeriesResult

# Set up module-level logger
logger = logging.getLogger("mfe.models.time_series.estimation")

# Type variables for generic functions
T = TypeVar('T', bound=TimeSeriesParameters)
ProgressCallback = Callable[[float, str], None]


@dataclass
class EstimationResult:
    """Container for estimation results.
    
    This class provides a standardized container for estimation results,
    including parameter estimates, convergence information, and diagnostics.
    
    Attributes:
        params: Estimated parameters
        success: Whether the optimization converged
        iterations: Number of iterations performed
        function_value: Final function value
        gradient: Final gradient
        hessian: Hessian matrix at the solution
        message: Message from the optimizer
        standard_errors: Standard errors of parameter estimates
        covariance_matrix: Covariance matrix of parameter estimates
    """
    
    params: np.ndarray
    success: bool = True
    iterations: int = 0
    function_value: Optional[float] = None
    gradient: Optional[np.ndarray] = None
    hessian: Optional[np.ndarray] = None
    message: Optional[str] = None
    standard_errors: Optional[np.ndarray] = None
    covariance_matrix: Optional[np.ndarray] = None


class EstimationMethod(Protocol):
    """Protocol defining the interface for estimation methods.
    
    This protocol defines the contract that all estimation methods must follow,
    ensuring consistent behavior across different estimation approaches.
    """
    
    def estimate(self, 
                model: TimeSeriesModel[T], 
                data: np.ndarray,
                exog: Optional[np.ndarray] = None,
                start_params: Optional[np.ndarray] = None,
                **kwargs: Any) -> EstimationResult:
        """Estimate model parameters.
        
        Args:
            model: The time series model to estimate
            data: The data to fit the model to
            exog: Exogenous variables (if applicable)
            start_params: Initial parameter values
            **kwargs: Additional keyword arguments for estimation
        
        Returns:
            EstimationResult: The estimation results
        
        Raises:
            EstimationError: If the estimation fails
        """
        ...
    
    async def estimate_async(self, 
                            model: TimeSeriesModel[T], 
                            data: np.ndarray,
                            exog: Optional[np.ndarray] = None,
                            start_params: Optional[np.ndarray] = None,
                            progress_callback: Optional[ProgressCallback] = None,
                            **kwargs: Any) -> EstimationResult:
        """Asynchronously estimate model parameters.
        
        Args:
            model: The time series model to estimate
            data: The data to fit the model to
            exog: Exogenous variables (if applicable)
            start_params: Initial parameter values
            progress_callback: Callback function for reporting progress
            **kwargs: Additional keyword arguments for estimation
        
        Returns:
            EstimationResult: The estimation results
        
        Raises:
            EstimationError: If the estimation fails
        """
        ...


class MaximumLikelihoodEstimation(EstimationMethod):
    """Maximum likelihood estimation for time series models.
    
    This class implements maximum likelihood estimation for time series models,
    using SciPy's optimization routines for parameter estimation.
    """
    
    def __init__(self) -> None:
        """Initialize the maximum likelihood estimation method."""
        self._cancel_requested = False
    
    def estimate(self, 
                model: TimeSeriesModel[T], 
                data: np.ndarray,
                exog: Optional[np.ndarray] = None,
                start_params: Optional[np.ndarray] = None,
                **kwargs: Any) -> EstimationResult:
        """Estimate model parameters using maximum likelihood.
        
        Args:
            model: The time series model to estimate
            data: The data to fit the model to
            exog: Exogenous variables (if applicable)
            start_params: Initial parameter values
            **kwargs: Additional keyword arguments for estimation
        
        Returns:
            EstimationResult: The estimation results
        
        Raises:
            EstimationError: If the estimation fails
        """
        # Get configuration from model or kwargs
        config = model.config
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Get or generate starting values
        if start_params is None:
            start_params = self._get_starting_values(model, data, exog)
        
        # Define the negative log-likelihood function for optimization
        def neg_loglikelihood(params: np.ndarray) -> float:
            try:
                # Convert parameters from unconstrained to constrained space
                param_obj = model._params.__class__.inverse_transform(params)
                
                # Compute log-likelihood
                llf = model.loglikelihood(param_obj, data, exog=exog)
                
                # Return negative log-likelihood for minimization
                return -llf
            except Exception as e:
                logger.warning(f"Error in negative log-likelihood: {e}")
                return np.inf
        
        # Define the gradient function if analytical gradient is available
        def gradient(params: np.ndarray) -> np.ndarray:
            try:
                # Convert parameters from unconstrained to constrained space
                param_obj = model._params.__class__.inverse_transform(params)
                
                # Compute gradient if the model provides it
                if hasattr(model, "loglikelihood_gradient"):
                    grad = model.loglikelihood_gradient(param_obj, data, exog=exog)
                    return -grad  # Negative for minimization
                else:
                    # Use numerical gradient if analytical gradient is not available
                    return optimize.approx_fprime(params, neg_loglikelihood, 1e-8)
            except Exception as e:
                logger.warning(f"Error in gradient computation: {e}")
                return np.zeros_like(params)
        
        # Set up optimization options
        options = {
            "maxiter": config.max_iter,
            "disp": config.display_progress
        }
        
        # Determine whether to use analytical gradient
        use_gradient = hasattr(model, "loglikelihood_gradient")
        
        # Perform optimization
        try:
            result = optimize.minimize(
                neg_loglikelihood,
                start_params,
                method=config.solver,
                jac=gradient if use_gradient else None,
                options=options,
                tol=config.tol
            )
            
            # Check for convergence
            if not result.success:
                # Try alternative optimization methods if the first one fails
                alternative_methods = ["L-BFGS-B", "Powell", "Nelder-Mead"]
                for method in alternative_methods:
                    if method != config.solver:
                        logger.info(f"Trying alternative optimization method: {method}")
                        alt_result = optimize.minimize(
                            neg_loglikelihood,
                            start_params,
                            method=method,
                            options=options,
                            tol=config.tol
                        )
                        if alt_result.success:
                            logger.info(f"Converged using alternative method: {method}")
                            result = alt_result
                            break
            
            # Compute standard errors and covariance matrix
            if result.success or result.nfev > 0:
                try:
                    # Compute Hessian at the solution
                    hessian = optimize.approx_fprime(
                        result.x,
                        lambda p: optimize.approx_fprime(p, neg_loglikelihood, 1e-6),
                        1e-6
                    )
                    
                    # Compute covariance matrix as inverse of Hessian
                    cov_params = np.linalg.inv(hessian)
                    
                    # Compute standard errors
                    std_errors = np.sqrt(np.diag(cov_params))
                except Exception as e:
                    logger.warning(f"Error computing standard errors: {e}")
                    hessian = None
                    cov_params = None
                    std_errors = None
            else:
                hessian = None
                cov_params = None
                std_errors = None
            
            # Create estimation result
            estimation_result = EstimationResult(
                params=result.x,
                success=result.success,
                iterations=result.nit if hasattr(result, 'nit') else result.nfev,
                function_value=result.fun,
                gradient=result.jac if hasattr(result, 'jac') else None,
                hessian=hessian,
                message=result.message,
                standard_errors=std_errors,
                covariance_matrix=cov_params
            )
            
            # Issue warning if convergence is questionable
            if not result.success:
                warn_convergence(
                    f"Optimization did not converge: {result.message}",
                    iterations=estimation_result.iterations,
                    tolerance=config.tol
                )
            
            return estimation_result
            
        except Exception as e:
            raise EstimationError(
                f"Maximum likelihood estimation failed: {e}",
                model_type=model._name,
                estimation_method="MLE",
                details=str(e)
            )
    
    async def estimate_async(self, 
                            model: TimeSeriesModel[T], 
                            data: np.ndarray,
                            exog: Optional[np.ndarray] = None,
                            start_params: Optional[np.ndarray] = None,
                            progress_callback: Optional[ProgressCallback] = None,
                            **kwargs: Any) -> EstimationResult:
        """Asynchronously estimate model parameters using maximum likelihood.
        
        Args:
            model: The time series model to estimate
            data: The data to fit the model to
            exog: Exogenous variables (if applicable)
            start_params: Initial parameter values
            progress_callback: Callback function for reporting progress
            **kwargs: Additional keyword arguments for estimation
        
        Returns:
            EstimationResult: The estimation results
        
        Raises:
            EstimationError: If the estimation fails
        """
        # Reset cancellation flag
        self._cancel_requested = False
        
        # Get configuration from model or kwargs
        config = model.config
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Get or generate starting values
        if start_params is None:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            start_params = await loop.run_in_executor(
                None, self._get_starting_values, model, data, exog
            )
            
            if progress_callback:
                progress_callback(0.1, "Starting values computed")
        
        # Define the negative log-likelihood function with progress reporting
        function_evaluations = 0
        max_evaluations = config.max_iter * 2  # Estimate of maximum evaluations
        
        def neg_loglikelihood(params: np.ndarray) -> float:
            nonlocal function_evaluations
            
            # Check for cancellation
            if self._cancel_requested:
                raise asyncio.CancelledError("Estimation cancelled by user")
            
            try:
                # Convert parameters from unconstrained to constrained space
                param_obj = model._params.__class__.inverse_transform(params)
                
                # Compute log-likelihood
                llf = model.loglikelihood(param_obj, data, exog=exog)
                
                # Update progress
                function_evaluations += 1
                if progress_callback and max_evaluations > 0:
                    # Estimate progress as a percentage of expected evaluations
                    progress = min(0.1 + 0.8 * (function_evaluations / max_evaluations), 0.9)
                    progress_callback(progress, f"Optimizing: iteration {function_evaluations}")
                
                # Return negative log-likelihood for minimization
                return -llf
            except asyncio.CancelledError:
                raise  # Re-raise cancellation
            except Exception as e:
                logger.warning(f"Error in negative log-likelihood: {e}")
                return np.inf
        
        # Define the gradient function if analytical gradient is available
        def gradient(params: np.ndarray) -> np.ndarray:
            try:
                # Convert parameters from unconstrained to constrained space
                param_obj = model._params.__class__.inverse_transform(params)
                
                # Compute gradient if the model provides it
                if hasattr(model, "loglikelihood_gradient"):
                    grad = model.loglikelihood_gradient(param_obj, data, exog=exog)
                    return -grad  # Negative for minimization
                else:
                    # Use numerical gradient if analytical gradient is not available
                    return optimize.approx_fprime(params, neg_loglikelihood, 1e-8)
            except asyncio.CancelledError:
                raise  # Re-raise cancellation
            except Exception as e:
                logger.warning(f"Error in gradient computation: {e}")
                return np.zeros_like(params)
        
        # Set up optimization options
        options = {
            "maxiter": config.max_iter,
            "disp": config.display_progress
        }
        
        # Determine whether to use analytical gradient
        use_gradient = hasattr(model, "loglikelihood_gradient")
        
        # Perform optimization asynchronously
        try:
            # Run optimization in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: optimize.minimize(
                    neg_loglikelihood,
                    start_params,
                    method=config.solver,
                    jac=gradient if use_gradient else None,
                    options=options,
                    tol=config.tol
                )
            )
            
            # Check for cancellation
            if self._cancel_requested:
                raise asyncio.CancelledError("Estimation cancelled by user")
            
            # Check for convergence
            if not result.success:
                # Try alternative optimization methods if the first one fails
                alternative_methods = ["L-BFGS-B", "Powell", "Nelder-Mead"]
                for method in alternative_methods:
                    if method != config.solver and not self._cancel_requested:
                        if progress_callback:
                            progress_callback(0.85, f"Trying alternative method: {method}")
                        
                        logger.info(f"Trying alternative optimization method: {method}")
                        alt_result = await loop.run_in_executor(
                            None,
                            lambda: optimize.minimize(
                                neg_loglikelihood,
                                start_params,
                                method=method,
                                options=options,
                                tol=config.tol
                            )
                        )
                        
                        if alt_result.success:
                            logger.info(f"Converged using alternative method: {method}")
                            result = alt_result
                            break
            
            # Report progress for final computations
            if progress_callback:
                progress_callback(0.9, "Computing standard errors and diagnostics")
            
            # Compute standard errors and covariance matrix
            if result.success or result.nfev > 0:
                try:
                    # Compute Hessian at the solution
                    hessian_func = lambda p: optimize.approx_fprime(p, neg_loglikelihood, 1e-6)
                    hessian = await loop.run_in_executor(
                        None,
                        lambda: optimize.approx_fprime(result.x, hessian_func, 1e-6)
                    )
                    
                    # Compute covariance matrix as inverse of Hessian
                    cov_params = np.linalg.inv(hessian)
                    
                    # Compute standard errors
                    std_errors = np.sqrt(np.diag(cov_params))
                except Exception as e:
                    logger.warning(f"Error computing standard errors: {e}")
                    hessian = None
                    cov_params = None
                    std_errors = None
            else:
                hessian = None
                cov_params = None
                std_errors = None
            
            # Create estimation result
            estimation_result = EstimationResult(
                params=result.x,
                success=result.success,
                iterations=result.nit if hasattr(result, 'nit') else result.nfev,
                function_value=result.fun,
                gradient=result.jac if hasattr(result, 'jac') else None,
                hessian=hessian,
                message=result.message,
                standard_errors=std_errors,
                covariance_matrix=cov_params
            )
            
            # Issue warning if convergence is questionable
            if not result.success:
                warn_convergence(
                    f"Optimization did not converge: {result.message}",
                    iterations=estimation_result.iterations,
                    tolerance=config.tol
                )
            
            # Final progress update
            if progress_callback:
                progress_callback(1.0, "Estimation complete")
            
            return estimation_result
            
        except asyncio.CancelledError:
            logger.info("Estimation cancelled by user")
            raise
        except Exception as e:
            raise EstimationError(
                f"Maximum likelihood estimation failed: {e}",
                model_type=model._name,
                estimation_method="MLE",
                details=str(e)
            )
    
    def cancel(self) -> None:
        """Cancel the ongoing estimation process."""
        self._cancel_requested = True
    
    def _get_starting_values(self, 
                            model: TimeSeriesModel[T], 
                            data: np.ndarray,
                            exog: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate starting values for parameter estimation.
        
        Args:
            model: The time series model
            data: The data to fit the model to
            exog: Exogenous variables (if applicable)
        
        Returns:
            np.ndarray: Starting parameter values in unconstrained space
        
        Raises:
            EstimationError: If starting value generation fails
        """
        try:
            # If the model provides a method for generating starting values, use it
            if hasattr(model, "generate_starting_values"):
                params = model.generate_starting_values(data, exog=exog)
                return params.transform()
            
            # Otherwise, use a simple heuristic based on the model type
            if isinstance(model._params, ARMAParameters):
                # For ARMA models, use OLS or Yule-Walker for AR parameters
                ar_order = len(model._params.ar_params)
                ma_order = len(model._params.ma_params)
                
                # Initialize parameters
                ar_params = np.zeros(ar_order)
                ma_params = np.zeros(ma_order)
                constant = 0.0
                sigma2 = np.var(data)
                
                # Estimate AR parameters using Yule-Walker if applicable
                if ar_order > 0:
                    try:
                        from statsmodels.tsa.ar_model import AutoReg
                        ar_model = AutoReg(data, lags=ar_order, trend='c')
                        ar_result = ar_model.fit()
                        
                        # Extract AR parameters
                        ar_params = ar_result.params[1:ar_order+1]
                        
                        # Extract constant
                        constant = ar_result.params[0]
                        
                        # Update sigma2 based on residuals
                        sigma2 = np.var(ar_result.resid)
                    except Exception as e:
                        logger.warning(f"Error estimating AR parameters: {e}")
                        # Fall back to simple autocorrelation-based estimates
                        for i in range(ar_order):
                            ar_params[i] = np.corrcoef(data[i+1:], data[:-i-1])[0, 1]
                
                # Create parameter object
                params = ARMAParameters(
                    ar_params=ar_params,
                    ma_params=ma_params,
                    sigma2=sigma2,
                    constant=constant
                )
                
                # Transform to unconstrained space
                return params.transform()
            
            # For other model types, use default values
            # This is a placeholder - specific model types should provide their own methods
            logger.warning(f"Using default starting values for {model._name}")
            return model._params.transform()
            
        except Exception as e:
            raise EstimationError(
                f"Failed to generate starting values: {e}",
                model_type=model._name,
                estimation_method="starting values",
                details=str(e)
            )


class ConditionalSumOfSquaresEstimation(EstimationMethod):
    """Conditional sum-of-squares estimation for time series models.
    
    This class implements conditional sum-of-squares estimation for time series models,
    which minimizes the sum of squared residuals conditional on initial values.
    """
    
    def __init__(self) -> None:
        """Initialize the conditional sum-of-squares estimation method."""
        self._cancel_requested = False
    
    def estimate(self, 
                model: TimeSeriesModel[T], 
                data: np.ndarray,
                exog: Optional[np.ndarray] = None,
                start_params: Optional[np.ndarray] = None,
                **kwargs: Any) -> EstimationResult:
        """Estimate model parameters using conditional sum-of-squares.
        
        Args:
            model: The time series model to estimate
            data: The data to fit the model to
            exog: Exogenous variables (if applicable)
            start_params: Initial parameter values
            **kwargs: Additional keyword arguments for estimation
        
        Returns:
            EstimationResult: The estimation results
        
        Raises:
            EstimationError: If the estimation fails
        """
        # Get configuration from model or kwargs
        config = model.config
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Get or generate starting values
        if start_params is None:
            start_params = self._get_starting_values(model, data, exog)
        
        # Define the sum-of-squares function for optimization
        def sum_of_squares(params: np.ndarray) -> float:
            try:
                # Convert parameters from unconstrained to constrained space
                param_obj = model._params.__class__.inverse_transform(params)
                
                # Compute residuals if the model provides a method for it
                if hasattr(model, "compute_residuals"):
                    residuals = model.compute_residuals(param_obj, data, exog=exog)
                    
                    # Return sum of squared residuals
                    return np.sum(residuals**2)
                else:
                    # Fall back to negative log-likelihood if residuals method is not available
                    llf = model.loglikelihood(param_obj, data, exog=exog)
                    return -llf
            except Exception as e:
                logger.warning(f"Error in sum-of-squares: {e}")
                return np.inf
        
        # Set up optimization options
        options = {
            "maxiter": config.max_iter,
            "disp": config.display_progress
        }
        
        # Perform optimization
        try:
            result = optimize.minimize(
                sum_of_squares,
                start_params,
                method=config.solver,
                options=options,
                tol=config.tol
            )
            
            # Check for convergence
            if not result.success:
                # Try alternative optimization methods if the first one fails
                alternative_methods = ["L-BFGS-B", "Powell", "Nelder-Mead"]
                for method in alternative_methods:
                    if method != config.solver:
                        logger.info(f"Trying alternative optimization method: {method}")
                        alt_result = optimize.minimize(
                            sum_of_squares,
                            start_params,
                            method=method,
                            options=options,
                            tol=config.tol
                        )
                        if alt_result.success:
                            logger.info(f"Converged using alternative method: {method}")
                            result = alt_result
                            break
            
            # Compute standard errors and covariance matrix
            if result.success or result.nfev > 0:
                try:
                    # Convert parameters to constrained space
                    param_obj = model._params.__class__.inverse_transform(result.x)
                    
                    # Compute residuals
                    if hasattr(model, "compute_residuals"):
                        residuals = model.compute_residuals(param_obj, data, exog=exog)
                    else:
                        # Fall back to a simple approximation if residuals method is not available
                        residuals = data - np.mean(data)
                    
                    # Compute Hessian at the solution
                    hessian = optimize.approx_fprime(
                        result.x,
                        lambda p: optimize.approx_fprime(p, sum_of_squares, 1e-6),
                        1e-6
                    )
                    
                    # Compute covariance matrix
                    n = len(residuals)
                    sigma2 = np.sum(residuals**2) / n
                    cov_params = sigma2 * np.linalg.inv(hessian)
                    
                    # Compute standard errors
                    std_errors = np.sqrt(np.diag(cov_params))
                except Exception as e:
                    logger.warning(f"Error computing standard errors: {e}")
                    hessian = None
                    cov_params = None
                    std_errors = None
            else:
                hessian = None
                cov_params = None
                std_errors = None
            
            # Create estimation result
            estimation_result = EstimationResult(
                params=result.x,
                success=result.success,
                iterations=result.nit if hasattr(result, 'nit') else result.nfev,
                function_value=result.fun,
                gradient=result.jac if hasattr(result, 'jac') else None,
                hessian=hessian,
                message=result.message,
                standard_errors=std_errors,
                covariance_matrix=cov_params
            )
            
            # Issue warning if convergence is questionable
            if not result.success:
                warn_convergence(
                    f"Optimization did not converge: {result.message}",
                    iterations=estimation_result.iterations,
                    tolerance=config.tol
                )
            
            return estimation_result
            
        except Exception as e:
            raise EstimationError(
                f"Conditional sum-of-squares estimation failed: {e}",
                model_type=model._name,
                estimation_method="CSS",
                details=str(e)
            )
    
    async def estimate_async(self, 
                            model: TimeSeriesModel[T], 
                            data: np.ndarray,
                            exog: Optional[np.ndarray] = None,
                            start_params: Optional[np.ndarray] = None,
                            progress_callback: Optional[ProgressCallback] = None,
                            **kwargs: Any) -> EstimationResult:
        """Asynchronously estimate model parameters using conditional sum-of-squares.
        
        Args:
            model: The time series model to estimate
            data: The data to fit the model to
            exog: Exogenous variables (if applicable)
            start_params: Initial parameter values
            progress_callback: Callback function for reporting progress
            **kwargs: Additional keyword arguments for estimation
        
        Returns:
            EstimationResult: The estimation results
        
        Raises:
            EstimationError: If the estimation fails
        """
        # Reset cancellation flag
        self._cancel_requested = False
        
        # Get configuration from model or kwargs
        config = model.config
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Get or generate starting values
        if start_params is None:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            start_params = await loop.run_in_executor(
                None, self._get_starting_values, model, data, exog
            )
            
            if progress_callback:
                progress_callback(0.1, "Starting values computed")
        
        # Define the sum-of-squares function with progress reporting
        function_evaluations = 0
        max_evaluations = config.max_iter * 2  # Estimate of maximum evaluations
        
        def sum_of_squares(params: np.ndarray) -> float:
            nonlocal function_evaluations
            
            # Check for cancellation
            if self._cancel_requested:
                raise asyncio.CancelledError("Estimation cancelled by user")
            
            try:
                # Convert parameters from unconstrained to constrained space
                param_obj = model._params.__class__.inverse_transform(params)
                
                # Compute residuals if the model provides a method for it
                if hasattr(model, "compute_residuals"):
                    residuals = model.compute_residuals(param_obj, data, exog=exog)
                    
                    # Return sum of squared residuals
                    result = np.sum(residuals**2)
                else:
                    # Fall back to negative log-likelihood if residuals method is not available
                    llf = model.loglikelihood(param_obj, data, exog=exog)
                    result = -llf
                
                # Update progress
                function_evaluations += 1
                if progress_callback and max_evaluations > 0:
                    # Estimate progress as a percentage of expected evaluations
                    progress = min(0.1 + 0.8 * (function_evaluations / max_evaluations), 0.9)
                    progress_callback(progress, f"Optimizing: iteration {function_evaluations}")
                
                return result
            except asyncio.CancelledError:
                raise  # Re-raise cancellation
            except Exception as e:
                logger.warning(f"Error in sum-of-squares: {e}")
                return np.inf
        
        # Set up optimization options
        options = {
            "maxiter": config.max_iter,
            "disp": config.display_progress
        }
        
        # Perform optimization asynchronously
        try:
            # Run optimization in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: optimize.minimize(
                    sum_of_squares,
                    start_params,
                    method=config.solver,
                    options=options,
                    tol=config.tol
                )
            )
            
            # Check for cancellation
            if self._cancel_requested:
                raise asyncio.CancelledError("Estimation cancelled by user")
            
            # Check for convergence
            if not result.success:
                # Try alternative optimization methods if the first one fails
                alternative_methods = ["L-BFGS-B", "Powell", "Nelder-Mead"]
                for method in alternative_methods:
                    if method != config.solver and not self._cancel_requested:
                        if progress_callback:
                            progress_callback(0.85, f"Trying alternative method: {method}")
                        
                        logger.info(f"Trying alternative optimization method: {method}")
                        alt_result = await loop.run_in_executor(
                            None,
                            lambda: optimize.minimize(
                                sum_of_squares,
                                start_params,
                                method=method,
                                options=options,
                                tol=config.tol
                            )
                        )
                        
                        if alt_result.success:
                            logger.info(f"Converged using alternative method: {method}")
                            result = alt_result
                            break
            
            # Report progress for final computations
            if progress_callback:
                progress_callback(0.9, "Computing standard errors and diagnostics")
            
            # Compute standard errors and covariance matrix
            if result.success or result.nfev > 0:
                try:
                    # Convert parameters to constrained space
                    param_obj = model._params.__class__.inverse_transform(result.x)
                    
                    # Compute residuals
                    if hasattr(model, "compute_residuals"):
                        residuals = await loop.run_in_executor(
                            None,
                            lambda: model.compute_residuals(param_obj, data, exog=exog)
                        )
                    else:
                        # Fall back to a simple approximation if residuals method is not available
                        residuals = data - np.mean(data)
                    
                    # Compute Hessian at the solution
                    hessian_func = lambda p: optimize.approx_fprime(p, sum_of_squares, 1e-6)
                    hessian = await loop.run_in_executor(
                        None,
                        lambda: optimize.approx_fprime(result.x, hessian_func, 1e-6)
                    )
                    
                    # Compute covariance matrix
                    n = len(residuals)
                    sigma2 = np.sum(residuals**2) / n
                    cov_params = sigma2 * np.linalg.inv(hessian)
                    
                    # Compute standard errors
                    std_errors = np.sqrt(np.diag(cov_params))
                except Exception as e:
                    logger.warning(f"Error computing standard errors: {e}")
                    hessian = None
                    cov_params = None
                    std_errors = None
            else:
                hessian = None
                cov_params = None
                std_errors = None
            
            # Create estimation result
            estimation_result = EstimationResult(
                params=result.x,
                success=result.success,
                iterations=result.nit if hasattr(result, 'nit') else result.nfev,
                function_value=result.fun,
                gradient=result.jac if hasattr(result, 'jac') else None,
                hessian=hessian,
                message=result.message,
                standard_errors=std_errors,
                covariance_matrix=cov_params
            )
            
            # Issue warning if convergence is questionable
            if not result.success:
                warn_convergence(
                    f"Optimization did not converge: {result.message}",
                    iterations=estimation_result.iterations,
                    tolerance=config.tol
                )
            
            # Final progress update
            if progress_callback:
                progress_callback(1.0, "Estimation complete")
            
            return estimation_result
            
        except asyncio.CancelledError:
            logger.info("Estimation cancelled by user")
            raise
        except Exception as e:
            raise EstimationError(
                f"Conditional sum-of-squares estimation failed: {e}",
                model_type=model._name,
                estimation_method="CSS",
                details=str(e)
            )
    
    def cancel(self) -> None:
        """Cancel the ongoing estimation process."""
        self._cancel_requested = True
    
    def _get_starting_values(self, 
                            model: TimeSeriesModel[T], 
                            data: np.ndarray,
                            exog: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate starting values for parameter estimation.
        
        Args:
            model: The time series model
            data: The data to fit the model to
            exog: Exogenous variables (if applicable)
        
        Returns:
            np.ndarray: Starting parameter values in unconstrained space
        
        Raises:
            EstimationError: If starting value generation fails
        """
        # Use the same starting value generation as MLE
        mle = MaximumLikelihoodEstimation()
        return mle._get_starting_values(model, data, exog)


class OrdinaryLeastSquaresEstimation(EstimationMethod):
    """Ordinary least squares estimation for time series models.
    
    This class implements ordinary least squares estimation for time series models,
    which is applicable to models that can be expressed as linear regressions.
    """
    
    def __init__(self) -> None:
        """Initialize the ordinary least squares estimation method."""
        self._cancel_requested = False
    
    def estimate(self, 
                model: TimeSeriesModel[T], 
                data: np.ndarray,
                exog: Optional[np.ndarray] = None,
                start_params: Optional[np.ndarray] = None,
                **kwargs: Any) -> EstimationResult:
        """Estimate model parameters using ordinary least squares.
        
        Args:
            model: The time series model to estimate
            data: The data to fit the model to
            exog: Exogenous variables (if applicable)
            start_params: Initial parameter values (not used in OLS)
            **kwargs: Additional keyword arguments for estimation
        
        Returns:
            EstimationResult: The estimation results
        
        Raises:
            EstimationError: If the estimation fails
        """
        # Get configuration from model or kwargs
        config = model.config
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        try:
            # Check if the model provides a method for creating design matrix
            if hasattr(model, "create_design_matrix"):
                # Create design matrix
                X, y = model.create_design_matrix(data, exog=exog)
                
                # Perform OLS estimation
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                
                # Compute residuals
                residuals = y - X @ beta
                
                # Compute sigma2
                n, k = X.shape
                sigma2 = np.sum(residuals**2) / (n - k)
                
                # Compute covariance matrix
                try:
                    XX_inv = np.linalg.inv(X.T @ X)
                    cov_params = sigma2 * XX_inv
                    std_errors = np.sqrt(np.diag(cov_params))
                except Exception as e:
                    logger.warning(f"Error computing covariance matrix: {e}")
                    cov_params = None
                    std_errors = None
                
                # Convert parameters to model's parameter format
                if hasattr(model, "ols_to_model_params"):
                    params = model.ols_to_model_params(beta, sigma2)
                else:
                    # Default conversion - this is a placeholder
                    # Specific model types should provide their own conversion methods
                    params = model._params.__class__(
                        ar_params=beta[1:model.ar_order+1] if hasattr(model, 'ar_order') else np.array([]),
                        ma_params=np.zeros(model.ma_order) if hasattr(model, 'ma_order') else np.array([]),
                        sigma2=sigma2,
                        constant=beta[0] if len(beta) > 0 else 0.0
                    )
                
                # Transform parameters to unconstrained space
                unconstrained_params = params.transform()
                
                # Create estimation result
                estimation_result = EstimationResult(
                    params=unconstrained_params,
                    success=True,
                    iterations=1,  # OLS is a direct solution
                    function_value=np.sum(residuals**2),
                    gradient=None,
                    hessian=None,
                    message="OLS estimation successful",
                    standard_errors=std_errors,
                    covariance_matrix=cov_params
                )
                
                return estimation_result
            else:
                raise EstimationError(
                    "Model does not support OLS estimation (create_design_matrix method not found)",
                    model_type=model._name,
                    estimation_method="OLS"
                )
                
        except Exception as e:
            raise EstimationError(
                f"Ordinary least squares estimation failed: {e}",
                model_type=model._name,
                estimation_method="OLS",
                details=str(e)
            )
    
    async def estimate_async(self, 
                            model: TimeSeriesModel[T], 
                            data: np.ndarray,
                            exog: Optional[np.ndarray] = None,
                            start_params: Optional[np.ndarray] = None,
                            progress_callback: Optional[ProgressCallback] = None,
                            **kwargs: Any) -> EstimationResult:
        """Asynchronously estimate model parameters using ordinary least squares.
        
        Args:
            model: The time series model to estimate
            data: The data to fit the model to
            exog: Exogenous variables (if applicable)
            start_params: Initial parameter values (not used in OLS)
            progress_callback: Callback function for reporting progress
            **kwargs: Additional keyword arguments for estimation
        
        Returns:
            EstimationResult: The estimation results
        
        Raises:
            EstimationError: If the estimation fails
        """
        # Reset cancellation flag
        self._cancel_requested = False
        
        # Report initial progress
        if progress_callback:
            progress_callback(0.1, "Starting OLS estimation")
        
        try:
            # Run OLS estimation in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self.estimate, model, data, exog, start_params, **kwargs
            )
            
            # Check for cancellation
            if self._cancel_requested:
                raise asyncio.CancelledError("Estimation cancelled by user")
            
            # Final progress update
            if progress_callback:
                progress_callback(1.0, "OLS estimation complete")
            
            return result
            
        except asyncio.CancelledError:
            logger.info("Estimation cancelled by user")
            raise
        except Exception as e:
            raise EstimationError(
                f"Ordinary least squares estimation failed: {e}",
                model_type=model._name,
                estimation_method="OLS",
                details=str(e)
            )
    
    def cancel(self) -> None:
        """Cancel the ongoing estimation process."""
        self._cancel_requested = True


# Numba-accelerated likelihood functions for ARMA models

@jit(nopython=True, cache=True)
def _arma_recursion(data: np.ndarray, 
                   ar_params: np.ndarray, 
                   ma_params: np.ndarray, 
                   constant: float) -> np.ndarray:
    """Compute ARMA recursion for residuals.
    
    This function computes the residuals of an ARMA model using a recursive approach.
    It is accelerated with Numba for performance.
    
    Args:
        data: The data array
        ar_params: Autoregressive parameters
        ma_params: Moving average parameters
        constant: Constant term
    
    Returns:
        np.ndarray: Residuals from the ARMA recursion
    """
    n = len(data)
    ar_order = len(ar_params)
    ma_order = len(ma_params)
    max_order = max(ar_order, ma_order)
    
    # Initialize arrays
    residuals = np.zeros(n)
    y_hat = np.zeros(n)
    
    # Initial values
    for t in range(max_order):
        residuals[t] = data[t] - constant
    
    # Main recursion
    for t in range(max_order, n):
        # Add constant term
        y_hat[t] = constant
        
        # Add AR terms
        for i in range(ar_order):
            if t - i - 1 >= 0:
                y_hat[t] += ar_params[i] * data[t - i - 1]
        
        # Add MA terms
        for i in range(ma_order):
            if t - i - 1 >= 0:
                y_hat[t] += ma_params[i] * residuals[t - i - 1]
        
        # Compute residual
        residuals[t] = data[t] - y_hat[t]
    
    return residuals


@jit(nopython=True, cache=True)
def _arma_loglikelihood(data: np.ndarray, 
                       ar_params: np.ndarray, 
                       ma_params: np.ndarray, 
                       sigma2: float, 
                       constant: float) -> float:
    """Compute log-likelihood for ARMA model.
    
    This function computes the log-likelihood of an ARMA model using a recursive approach.
    It is accelerated with Numba for performance.
    
    Args:
        data: The data array
        ar_params: Autoregressive parameters
        ma_params: Moving average parameters
        sigma2: Innovation variance
        constant: Constant term
    
    Returns:
        float: Log-likelihood value
    """
    n = len(data)
    
    # Compute residuals
    residuals = _arma_recursion(data, ar_params, ma_params, constant)
    
    # Compute log-likelihood
    llf = -0.5 * n * np.log(2 * np.pi * sigma2) - 0.5 * np.sum(residuals**2) / sigma2
    
    return llf


@jit(nopython=True, cache=True)
def _arma_conditional_loglikelihood(data: np.ndarray, 
                                   ar_params: np.ndarray, 
                                   ma_params: np.ndarray, 
                                   sigma2: float, 
                                   constant: float) -> float:
    """Compute conditional log-likelihood for ARMA model.
    
    This function computes the conditional log-likelihood of an ARMA model,
    which conditions on the first max(p,q) observations.
    It is accelerated with Numba for performance.
    
    Args:
        data: The data array
        ar_params: Autoregressive parameters
        ma_params: Moving average parameters
        sigma2: Innovation variance
        constant: Constant term
    
    Returns:
        float: Conditional log-likelihood value
    """
    n = len(data)
    ar_order = len(ar_params)
    ma_order = len(ma_params)
    max_order = max(ar_order, ma_order)
    
    # Compute residuals
    residuals = _arma_recursion(data, ar_params, ma_params, constant)
    
    # Compute conditional log-likelihood (excluding first max_order observations)
    llf = -0.5 * (n - max_order) * np.log(2 * np.pi * sigma2)
    llf -= 0.5 * np.sum(residuals[max_order:]**2) / sigma2
    
    return llf


# Factory function for creating estimation methods

def create_estimation_method(method: str) -> EstimationMethod:
    """Create an estimation method instance based on the method name.
    
    Args:
        method: The estimation method name ('mle', 'css', or 'ols')
    
    Returns:
        EstimationMethod: The estimation method instance
    
    Raises:
        ValueError: If the method name is invalid
    """
    method = method.lower()
    if method == 'mle':
        return MaximumLikelihoodEstimation()
    elif method == 'css':
        return ConditionalSumOfSquaresEstimation()
    elif method == 'ols':
        return OrdinaryLeastSquaresEstimation()
    else:
        raise ValueError(f"Invalid estimation method: {method}")


# Main estimation function for time series models

def estimate_model(model: TimeSeriesModel[T], 
                  data: Union[np.ndarray, pd.Series],
                  exog: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                  method: Optional[str] = None,
                  start_params: Optional[np.ndarray] = None,
                  **kwargs: Any) -> TimeSeriesResult:
    """Estimate a time series model.
    
    This function provides a unified interface for estimating time series models
    using different estimation methods.
    
    Args:
        model: The time series model to estimate
        data: The data to fit the model to
        exog: Exogenous variables (if applicable)
        method: The estimation method ('mle', 'css', or 'ols')
        start_params: Initial parameter values
        **kwargs: Additional keyword arguments for estimation
    
    Returns:
        TimeSeriesResult: The model estimation results
    
    Raises:
        EstimationError: If the estimation fails
    """
    # Validate and prepare data
    if isinstance(data, pd.Series):
        data_array = data.values
        index = data.index
    else:
        data_array = np.asarray(data)
        index = None
    
    # Validate and prepare exogenous variables
    exog_array = None
    if exog is not None:
        if isinstance(exog, pd.DataFrame):
            exog_array = exog.values
        elif isinstance(exog, pd.Series):
            exog_array = exog.values.reshape(-1, 1)
        else:
            exog_array = np.asarray(exog)
            if exog_array.ndim == 1:
                exog_array = exog_array.reshape(-1, 1)
    
    # Determine estimation method
    if method is None:
        method = model.config.method
    
    # Create estimation method
    estimator = create_estimation_method(method)
    
    # Estimate model
    result = estimator.estimate(model, data_array, exog=exog_array, start_params=start_params, **kwargs)
    
    # Convert estimation result to model result
    model_result = _create_model_result(model, data_array, exog_array, result, index)
    
    return model_result


async def estimate_model_async(model: TimeSeriesModel[T], 
                              data: Union[np.ndarray, pd.Series],
                              exog: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                              method: Optional[str] = None,
                              start_params: Optional[np.ndarray] = None,
                              progress_callback: Optional[ProgressCallback] = None,
                              **kwargs: Any) -> TimeSeriesResult:
    """Asynchronously estimate a time series model.
    
    This function provides a unified interface for asynchronously estimating time series models
    using different estimation methods.
    
    Args:
        model: The time series model to estimate
        data: The data to fit the model to
        exog: Exogenous variables (if applicable)
        method: The estimation method ('mle', 'css', or 'ols')
        start_params: Initial parameter values
        progress_callback: Callback function for reporting progress
        **kwargs: Additional keyword arguments for estimation
    
    Returns:
        TimeSeriesResult: The model estimation results
    
    Raises:
        EstimationError: If the estimation fails
    """
    # Validate and prepare data
    if isinstance(data, pd.Series):
        data_array = data.values
        index = data.index
    else:
        data_array = np.asarray(data)
        index = None
    
    # Validate and prepare exogenous variables
    exog_array = None
    if exog is not None:
        if isinstance(exog, pd.DataFrame):
            exog_array = exog.values
        elif isinstance(exog, pd.Series):
            exog_array = exog.values.reshape(-1, 1)
        else:
            exog_array = np.asarray(exog)
            if exog_array.ndim == 1:
                exog_array = exog_array.reshape(-1, 1)
    
    # Determine estimation method
    if method is None:
        method = model.config.method
    
    # Create estimation method
    estimator = create_estimation_method(method)
    
    # Estimate model asynchronously
    result = await estimator.estimate_async(
        model, data_array, exog=exog_array, start_params=start_params, 
        progress_callback=progress_callback, **kwargs
    )
    
    # Convert estimation result to model result
    model_result = _create_model_result(model, data_array, exog_array, result, index)
    
    return model_result


def _create_model_result(model: TimeSeriesModel[T], 
                        data: np.ndarray,
                        exog: Optional[np.ndarray],
                        estimation_result: EstimationResult,
                        index: Optional[pd.Index] = None) -> TimeSeriesResult:
    """Convert an estimation result to a model result.
    
    Args:
        model: The time series model
        data: The data array
        exog: Exogenous variables (if applicable)
        estimation_result: The estimation result
        index: The data index (if available)
    
    Returns:
        TimeSeriesResult: The model result
    """
    # Convert parameters from unconstrained to constrained space
    params = model._params.__class__.inverse_transform(estimation_result.params)
    
    # Compute residuals and fitted values
    if hasattr(model, "compute_residuals"):
        residuals = model.compute_residuals(params, data, exog=exog)
        fitted_values = data - residuals
    else:
        # Fall back to a simple approximation if residuals method is not available
        residuals = data - np.mean(data)
        fitted_values = np.mean(data) * np.ones_like(data)
    
    # Compute log-likelihood
    if hasattr(model, "loglikelihood"):
        loglikelihood = model.loglikelihood(params, data, exog=exog)
    else:
        # Fall back to a simple approximation if loglikelihood method is not available
        n = len(data)
        sigma2 = np.var(residuals)
        loglikelihood = -0.5 * n * np.log(2 * np.pi * sigma2) - 0.5 * np.sum(residuals**2) / sigma2
    
    # Compute information criteria
    n = len(data)
    k = len(estimation_result.params)
    aic = -2 * loglikelihood + 2 * k
    bic = -2 * loglikelihood + k * np.log(n)
    hqic = -2 * loglikelihood + 2 * k * np.log(np.log(n))
    
    # Convert parameters to dictionary
    param_dict = params.to_dict()
    
    # Create parameter-related dictionaries
    param_names = list(param_dict.keys())
    param_values = list(param_dict.values())
    
    params_dict = {name: value for name, value in zip(param_names, param_values)}
    
    # Create standard errors dictionary
    std_errors_dict = {}
    if estimation_result.standard_errors is not None:
        for i, name in enumerate(param_names):
            if i < len(estimation_result.standard_errors):
                std_errors_dict[name] = estimation_result.standard_errors[i]
            else:
                std_errors_dict[name] = np.nan
    else:
        for name in param_names:
            std_errors_dict[name] = np.nan
    
    # Compute t-statistics and p-values
    t_stats_dict = {}
    p_values_dict = {}
    
    for name, value in params_dict.items():
        std_err = std_errors_dict.get(name, np.nan)
        if std_err > 0:
            t_stat = value / std_err
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - k))
        else:
            t_stat = np.nan
            p_value = np.nan
        
        t_stats_dict[name] = t_stat
        p_values_dict[name] = p_value
    
    # Create result object
    result = TimeSeriesResult(
        model_name=model._name,
        params=params_dict,
        std_errors=std_errors_dict,
        t_stats=t_stats_dict,
        p_values=p_values_dict,
        log_likelihood=loglikelihood,
        aic=aic,
        bic=bic,
        hqic=hqic,
        residuals=residuals,
        fitted_values=fitted_values,
        convergence=estimation_result.success,
        iterations=estimation_result.iterations,
        cov_type=model.config.cov_type,
        cov_params=estimation_result.covariance_matrix,
        nobs=n,
        df_model=k,
        df_resid=n - k
    )
    
    return result
