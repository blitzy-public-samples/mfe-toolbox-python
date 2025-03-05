'''
Scalar Variance Targeting VECH multivariate GARCH model.

This module implements the Scalar Variance Targeting VECH multivariate GARCH model,
which uses a parsimonious parameterization for the dynamics of the vectorized
covariance matrix with variance targeting for improved stability and convergence.

The model is defined as:
    vech(H_t) = (1 - a - b) * vech(Σ) + a * vech(ε_{t-1}ε_{t-1}') + b * vech(H_{t-1})

where:
    - H_t is the conditional covariance matrix at time t
    - vech is the half-vectorization operator that stacks the lower triangular
      portion of a symmetric matrix
    - Σ is the unconditional covariance matrix (estimated from the data)
    - a and b are scalar parameters controlling the ARCH and GARCH effects
    - ε_t is the innovation vector at time t

The variance targeting approach uses the sample covariance matrix to estimate
the unconditional covariance matrix Σ, which reduces the number of parameters
to estimate and improves numerical stability.

Classes:
    ScalarVTVECHParameters: Parameter container for Scalar VT-VECH model
    ScalarVTVECHModel: Implementation of Scalar VT-VECH model
''' 

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
from scipy import optimize

from mfe.core.base import ModelBase
from mfe.core.parameters import (
    MultivariateVolatilityParameters, ParameterError,
    validate_non_negative, validate_positive_definite,
    transform_probability, inverse_transform_probability
)
from mfe.core.results import MultivariateVolatilityResult
from mfe.core.types import (
    AsyncProgressCallback, CovarianceMatrix, Matrix, ProgressCallback, Vector
)
from mfe.models.multivariate.base import CovarianceModelBase
from mfe.models.multivariate.utils import (
    compute_sample_covariance, compute_persistence, compute_half_life,
    ensure_positive_definite, validate_multivariate_data
)
from mfe.utils.matrix_ops import vech, ivech

# Set up module-level logger
logger = logging.getLogger("mfe.models.multivariate.scalar_vt_vech")

# Try to import numba for JIT compilation
try:
    from numba import jit
    from mfe.models.multivariate._numba_core import (
        _scalar_vt_vech_recursion_numba, _scalar_vt_vech_likelihood_numba,
        _scalar_vt_vech_simulate_numba, _scalar_vt_vech_forecast_numba
    )
    HAS_NUMBA = True
    logger.debug("Numba available for Scalar VT-VECH acceleration")
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    HAS_NUMBA = False
    logger.info("Numba not available. Scalar VT-VECH will use pure NumPy implementations.")


@dataclass
class ScalarVTVECHParameters(MultivariateVolatilityParameters):
    """
    Parameters for Scalar Variance Targeting VECH model.
    
    Attributes:
        a: ARCH parameter (must be non-negative)
        b: GARCH parameter (must be non-negative)
    """
    
    a: float
    b: float
    
    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """
        Validate Scalar VT-VECH parameter constraints.
        
        Raises:
            ParameterError: If parameter constraints are violated
        """
        super().validate()
        
        # Validate individual parameters
        validate_non_negative(self.a, "a")
        validate_non_negative(self.b, "b")
        
        # Validate stationarity constraint
        if self.a + self.b >= 1:
            raise ParameterError(
                f"Scalar VT-VECH stationarity constraint violated: a + b = {self.a + self.b} >= 1"
            )
    
    def to_array(self) -> np.ndarray:
        """
        Convert parameters to a NumPy array.
        
        Returns:
            np.ndarray: Array representation of parameters [a, b]
        """
        return np.array([self.a, self.b])
    
    @classmethod
    def from_array(cls, array: np.ndarray, **kwargs: Any) -> 'ScalarVTVECHParameters':
        """
        Create parameters from a NumPy array.
        
        Args:
            array: Array representation of parameters [a, b]
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            ScalarVTVECHParameters: Parameter object
        
        Raises:
            ValueError: If the array length is not 2
        """
        if len(array) != 2:
            raise ValueError(f"Array length must be 2, got {len(array)}")
        
        return cls(a=array[0], b=array[1])
    
    def transform(self) -> np.ndarray:
        """
        Transform parameters to unconstrained space for optimization.
        
        Returns:
            np.ndarray: Transformed parameters in unconstrained space
        """
        # Similar to GARCH transformation
        if self.a + self.b >= 1:
            # If constraint is violated, adjust parameters slightly
            sum_ab = self.a + self.b
            self.a = self.a / (sum_ab + 0.01)
            self.b = self.b / (sum_ab + 0.01)
        
        # Use logit-like transformation for a and b
        lambda_param = self.a + self.b
        delta_param = self.a / lambda_param if lambda_param > 0 else 0.5
        
        transformed_lambda = transform_probability(lambda_param)
        transformed_delta = transform_probability(delta_param)
        
        return np.array([transformed_lambda, transformed_delta])
    
    @classmethod
    def inverse_transform(cls, array: np.ndarray, **kwargs: Any) -> 'ScalarVTVECHParameters':
        """
        Transform parameters from unconstrained space back to constrained space.
        
        Args:
            array: Parameters in unconstrained space [lambda*, delta*]
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            ScalarVTVECHParameters: Parameter object with constrained parameters
        
        Raises:
            ValueError: If the array length is not 2
        """
        if len(array) != 2:
            raise ValueError(f"Array length must be 2, got {len(array)}")
        
        # Extract transformed parameters
        transformed_lambda, transformed_delta = array
        
        # Inverse transform lambda and delta
        lambda_param = inverse_transform_probability(transformed_lambda)
        delta_param = inverse_transform_probability(transformed_delta)
        
        # Compute a and b
        a = lambda_param * delta_param
        b = lambda_param * (1 - delta_param)
        
        return cls(a=a, b=b)


def _scalar_vt_vech_recursion(
    data: np.ndarray,
    a: float,
    b: float,
    unconditional_vech: np.ndarray,
    sigma_vech: Optional[np.ndarray] = None,
    backcast_vech: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Pure NumPy implementation of Scalar VT-VECH recursion.
    
    This function computes the conditional covariance matrices for the
    Scalar Variance Targeting VECH model using a pure NumPy implementation.
    
    Args:
        data: Input data array with shape (T, n_assets)
        a: ARCH parameter
        b: GARCH parameter
        unconditional_vech: Vectorized unconditional covariance matrix
        sigma_vech: Pre-allocated array for vectorized conditional covariances
        backcast_vech: Vectorized backcast value for initializing the recursion
    
    Returns:
        np.ndarray: Vectorized conditional covariance matrices with shape (n_vech, T)
    """
    T, n_assets = data.shape
    n_vech = n_assets * (n_assets + 1) // 2
    
    # Initialize sigma_vech if not provided
    if sigma_vech is None:
        sigma_vech = np.zeros((n_vech, T))
    
    # Initialize backcast if not provided
    if backcast_vech is None:
        # Use unconditional covariance as backcast
        backcast_vech = unconditional_vech
    
    # Compute intercept term
    omega_vech = (1 - a - b) * unconditional_vech
    
    # Initialize first period with backcast
    sigma_vech[:, 0] = backcast_vech
    
    # Compute vectorized outer products of data
    outer_vech = np.zeros((n_vech, T))
    for t in range(T):
        # Compute outer product
        outer = np.outer(data[t], data[t])
        # Vectorize using vech
        outer_vech[:, t] = vech(outer)
    
    # Perform recursion
    for t in range(1, T):
        sigma_vech[:, t] = omega_vech + a * outer_vech[:, t-1] + b * sigma_vech[:, t-1]
    
    return sigma_vech


def _scalar_vt_vech_likelihood(
    data: np.ndarray,
    a: float,
    b: float,
    unconditional_vech: np.ndarray,
    sigma_vech: Optional[np.ndarray] = None,
    backcast_vech: Optional[np.ndarray] = None,
    individual: bool = False
) -> Union[float, np.ndarray]:
    """
    Pure NumPy implementation of Scalar VT-VECH likelihood function.
    
    This function computes the log-likelihood for the Scalar Variance Targeting
    VECH model using a pure NumPy implementation.
    
    Args:
        data: Input data array with shape (T, n_assets)
        a: ARCH parameter
        b: GARCH parameter
        unconditional_vech: Vectorized unconditional covariance matrix
        sigma_vech: Pre-allocated array for vectorized conditional covariances
        backcast_vech: Vectorized backcast value for initializing the recursion
        individual: Whether to return individual log-likelihood contributions
    
    Returns:
        Union[float, np.ndarray]: Log-likelihood value or individual contributions
    """
    T, n_assets = data.shape
    n_vech = n_assets * (n_assets + 1) // 2
    
    # Compute conditional covariance matrices
    sigma_vech = _scalar_vt_vech_recursion(
        data, a, b, unconditional_vech, sigma_vech, backcast_vech
    )
    
    # Initialize log-likelihood
    loglik = np.zeros(T)
    
    # Compute log-likelihood for each time point
    for t in range(T):
        # Convert vectorized covariance to matrix form
        sigma_t = ivech(sigma_vech[:, t])
        
        try:
            # Try Cholesky decomposition for numerical stability
            chol = np.linalg.cholesky(sigma_t)
            # Compute log determinant using Cholesky factor
            log_det = 2 * np.sum(np.log(np.diag(chol)))
            # Compute quadratic form using Cholesky factor
            quad_form = np.sum(np.linalg.solve(chol, data[t]) ** 2)
        except np.linalg.LinAlgError:
            # If Cholesky decomposition fails, use eigenvalue decomposition
            eigvals = np.linalg.eigvalsh(sigma_t)
            # Ensure all eigenvalues are positive
            eigvals = np.maximum(eigvals, 1e-8)
            # Compute log determinant using eigenvalues
            log_det = np.sum(np.log(eigvals))
            # Compute quadratic form using matrix inverse
            quad_form = data[t] @ np.linalg.solve(sigma_t, data[t])
        
        # Compute log-likelihood contribution
        loglik[t] = -0.5 * (n_assets * np.log(2 * np.pi) + log_det + quad_form)
    
    if individual:
        return loglik
    
    # Return sum of log-likelihood contributions
    return np.sum(loglik)


def _scalar_vt_vech_simulate(
    n_periods: int,
    a: float,
    b: float,
    unconditional_vech: np.ndarray,
    n_assets: int,
    burn: int = 0,
    initial_value_vech: Optional[np.ndarray] = None,
    random_state: Optional[Union[int, np.random.Generator]] = None,
    return_covariances: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Pure NumPy implementation of Scalar VT-VECH simulation.
    
    This function simulates data from the Scalar Variance Targeting VECH model
    using a pure NumPy implementation.
    
    Args:
        n_periods: Number of periods to simulate
        a: ARCH parameter
        b: GARCH parameter
        unconditional_vech: Vectorized unconditional covariance matrix
        n_assets: Number of assets
        burn: Number of initial observations to discard
        initial_value_vech: Vectorized initial covariance matrix
        random_state: Random number generator or seed
        return_covariances: Whether to return conditional covariances
    
    Returns:
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: 
            Simulated data and optionally conditional covariances
    """
    # Set up random number generator
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)
    
    # Total number of periods including burn-in
    total_periods = n_periods + burn
    
    # Initialize arrays
    n_vech = n_assets * (n_assets + 1) // 2
    sigma_vech = np.zeros((n_vech, total_periods))
    data = np.zeros((total_periods, n_assets))
    
    # Initialize first period
    if initial_value_vech is None:
        # Use unconditional covariance as initial value
        sigma_vech[:, 0] = unconditional_vech
    else:
        sigma_vech[:, 0] = initial_value_vech
    
    # Convert first covariance to matrix form
    sigma_0 = ivech(sigma_vech[:, 0])
    
    # Generate first observation
    try:
        # Try Cholesky decomposition
        chol_0 = np.linalg.cholesky(sigma_0)
        # Generate random normal variables
        z_0 = rng.standard_normal(n_assets)
        # Transform to correlated random variables
        data[0] = chol_0 @ z_0
    except np.linalg.LinAlgError:
        # If Cholesky decomposition fails, use eigenvalue decomposition
        eigvals_0, eigvecs_0 = np.linalg.eigh(sigma_0)
        # Ensure all eigenvalues are positive
        eigvals_0 = np.maximum(eigvals_0, 1e-8)
        # Compute square root of covariance matrix
        sqrt_sigma_0 = eigvecs_0 @ np.diag(np.sqrt(eigvals_0)) @ eigvecs_0.T
        # Generate random normal variables
        z_0 = rng.standard_normal(n_assets)
        # Transform to correlated random variables
        data[0] = sqrt_sigma_0 @ z_0
    
    # Compute intercept term
    omega_vech = (1 - a - b) * unconditional_vech
    
    # Perform simulation
    for t in range(1, total_periods):
        # Compute outer product of previous observation
        outer_t_1 = np.outer(data[t-1], data[t-1])
        # Vectorize using vech
        outer_vech_t_1 = vech(outer_t_1)
        
        # Update conditional covariance
        sigma_vech[:, t] = omega_vech + a * outer_vech_t_1 + b * sigma_vech[:, t-1]
        
        # Convert to matrix form
        sigma_t = ivech(sigma_vech[:, t])
        
        # Generate observation
        try:
            # Try Cholesky decomposition
            chol_t = np.linalg.cholesky(sigma_t)
            # Generate random normal variables
            z_t = rng.standard_normal(n_assets)
            # Transform to correlated random variables
            data[t] = chol_t @ z_t
        except np.linalg.LinAlgError:
            # If Cholesky decomposition fails, use eigenvalue decomposition
            eigvals_t, eigvecs_t = np.linalg.eigh(sigma_t)
            # Ensure all eigenvalues are positive
            eigvals_t = np.maximum(eigvals_t, 1e-8)
            # Compute square root of covariance matrix
            sqrt_sigma_t = eigvecs_t @ np.diag(np.sqrt(eigvals_t)) @ eigvecs_t.T
            # Generate random normal variables
            z_t = rng.standard_normal(n_assets)
            # Transform to correlated random variables
            data[t] = sqrt_sigma_t @ z_t
    
    # Discard burn-in periods
    data = data[burn:]
    sigma_vech = sigma_vech[:, burn:]
    
    if return_covariances:
        # Convert vectorized covariances to matrix form
        sigma = np.zeros((n_assets, n_assets, n_periods))
        for t in range(n_periods):
            sigma[:, :, t] = ivech(sigma_vech[:, t])
        
        return data, sigma
    
    return data


def _scalar_vt_vech_forecast(
    a: float,
    b: float,
    unconditional_vech: np.ndarray,
    last_sigma_vech: np.ndarray,
    last_data: np.ndarray,
    steps: int,
    n_assets: int
) -> np.ndarray:
    """
    Pure NumPy implementation of Scalar VT-VECH forecasting.
    
    This function forecasts conditional covariance matrices for the
    Scalar Variance Targeting VECH model using a pure NumPy implementation.
    
    Args:
        a: ARCH parameter
        b: GARCH parameter
        unconditional_vech: Vectorized unconditional covariance matrix
        last_sigma_vech: Vectorized last conditional covariance matrix
        last_data: Last observation
        steps: Number of steps to forecast
        n_assets: Number of assets
    
    Returns:
        np.ndarray: Forecasted conditional covariance matrices with shape (n_assets, n_assets, steps)
    """
    # Initialize forecast array
    n_vech = n_assets * (n_assets + 1) // 2
    forecast_vech = np.zeros((n_vech, steps))
    
    # Compute intercept term
    omega_vech = (1 - a - b) * unconditional_vech
    
    # Compute outer product of last observation
    last_outer = np.outer(last_data, last_data)
    last_outer_vech = vech(last_outer)
    
    # First step forecast
    forecast_vech[:, 0] = omega_vech + a * last_outer_vech + b * last_sigma_vech
    
    # Multi-step forecasts
    for h in range(1, steps):
        # For h > 1, E[ε_{t+h-1}ε_{t+h-1}'] = H_{t+h-1}
        # So we use the previous forecast as the expected outer product
        forecast_vech[:, h] = omega_vech + (a + b) * forecast_vech[:, h-1]
    
    # Convert vectorized forecasts to matrix form
    forecast = np.zeros((n_assets, n_assets, steps))
    for h in range(steps):
        forecast[:, :, h] = ivech(forecast_vech[:, h])
    
    return forecast


class ScalarVTVECHModel(CovarianceModelBase):
    """
    Scalar Variance Targeting VECH multivariate GARCH model.
    
    This class implements the Scalar Variance Targeting VECH multivariate GARCH model,
    which uses a parsimonious parameterization for the dynamics of the vectorized
    covariance matrix with variance targeting for improved stability and convergence.
    
    Attributes:
        name: A descriptive name for the model
        n_assets: Number of assets in the model
        parameters: Model parameters if fitted
        unconditional_covariance: Unconditional covariance matrix
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        n_assets: Optional[int] = None
    ):
        """
        Initialize the Scalar VT-VECH model.
        
        Args:
            name: A descriptive name for the model (if None, uses class name)
            n_assets: Number of assets in the model (if None, determined from data)
        """
        if name is None:
            name = "ScalarVTVECHModel"
        super().__init__(name=name, n_assets=n_assets)
        
        self._unconditional_covariance: Optional[np.ndarray] = None
        self._unconditional_vech: Optional[np.ndarray] = None
    
    @property
    def unconditional_covariance(self) -> Optional[np.ndarray]:
        """
        Get the unconditional covariance matrix.
        
        Returns:
            Optional[np.ndarray]: The unconditional covariance matrix if the model
                                 has been fitted, None otherwise
        
        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self._unconditional_covariance
    
    def fit(
        self,
        data: np.ndarray,
        starting_values: Optional[Union[np.ndarray, ScalarVTVECHParameters]] = None,
        backcast: Optional[np.ndarray] = None,
        method: str = "SLSQP",
        options: Optional[Dict[str, Any]] = None,
        constraints: Optional[List[Dict[str, Any]]] = None,
        callback: Optional[ProgressCallback] = None,
        **kwargs: Any
    ) -> MultivariateVolatilityResult:
        """
        Fit the Scalar VT-VECH model to the provided data.
        
        Args:
            data: Residual data for model fitting (T x n_assets)
            starting_values: Initial parameter values for optimization
            backcast: Value to use for initializing the covariance process
            method: Optimization method to use
            options: Additional options for the optimizer
            constraints: Constraints for the optimizer
            callback: Callback function for reporting optimization progress
            **kwargs: Additional keyword arguments for model fitting
        
        Returns:
            MultivariateVolatilityResult: The model estimation results
        
        Raises:
            ValueError: If the data is invalid
            RuntimeError: If the model fails to converge
        """
        # Validate data
        T, n_assets = validate_multivariate_data(data)
        
        # Update n_assets if not already set
        if self._n_assets is None:
            self._n_assets = n_assets
        elif self._n_assets != n_assets:
            raise ValueError(
                f"Data has {n_assets} assets, but model was initialized with {self._n_assets} assets"
            )
        
        # Compute unconditional covariance matrix
        self._unconditional_covariance = compute_sample_covariance(data)
        
        # Ensure positive definiteness
        self._unconditional_covariance = ensure_positive_definite(
            self._unconditional_covariance, method="nearest"
        )
        
        # Vectorize unconditional covariance
        self._unconditional_vech = vech(self._unconditional_covariance)
        
        # Compute backcast if not provided
        if backcast is None:
            backcast = self._unconditional_covariance
        
        # Vectorize backcast
        backcast_vech = vech(backcast)
        
        # Create starting values if not provided
        if starting_values is None:
            # Default starting values: a = 0.05, b = 0.85
            starting_values = np.array([0.05, 0.85])
        
        # Convert starting_values to array if it's a parameter object
        if isinstance(starting_values, ScalarVTVECHParameters):
            starting_values = starting_values.to_array()
        
        # Transform parameters to unconstrained space for optimization
        starting_values_t = ScalarVTVECHParameters(
            a=starting_values[0], b=starting_values[1]
        ).transform()
        
        # Create constraints if not provided
        if constraints is None:
            constraints = self._create_constraints()
        
        # Create options if not provided
        if options is None:
            options = {'disp': False, 'maxiter': 1000}
        
        # Define objective function for optimization
        def objective(params_t: np.ndarray) -> float:
            # Convert parameters from unconstrained space
            params = ScalarVTVECHParameters.inverse_transform(params_t).to_array()
            a, b = params
            
            # Compute negative log-likelihood
            try:
                if HAS_NUMBA:
                    # Use Numba-accelerated implementation if available
                    loglik = _scalar_vt_vech_likelihood_numba(
                        data, a, b, self._unconditional_vech, None, backcast_vech, False
                    )
                else:
                    # Use pure NumPy implementation
                    loglik = _scalar_vt_vech_likelihood(
                        data, a, b, self._unconditional_vech, None, backcast_vech, False
                    )
                
                return -loglik
            except (ValueError, ParameterError, np.linalg.LinAlgError) as e:
                # Return a large value if parameters are invalid
                return 1e10
        
        # Define callback function for progress reporting
        iteration = [0]
        max_iterations = options.get('maxiter', 1000)
        
        def progress_callback(params_t: np.ndarray) -> None:
            iteration[0] += 1
            if callback is not None:
                progress = min(iteration[0] / max_iterations, 0.99)
                callback(progress, f"Iteration {iteration[0]}/{max_iterations}")
        
        # Run optimization
        optimization_callback = progress_callback if callback is not None else None
        
        try:
            result = optimize.minimize(
                objective,
                starting_values_t,
                method=method,
                options=options,
                constraints=constraints,
                callback=optimization_callback
            )
            
            # Check convergence
            if not result.success:
                warnings.warn(
                    f"Optimization did not converge: {result.message}",
                    RuntimeWarning
                )
            
            # Convert parameters from unconstrained space
            params = ScalarVTVECHParameters.inverse_transform(result.x)
            
            # Store parameters
            self._parameters = params
            
            # Compute conditional covariances
            self._conditional_covariances = self.compute_covariance(
                params, data, backcast=backcast
            )
            
            # Compute conditional correlations
            self._conditional_correlations = self.compute_correlation(
                params, data, backcast=backcast
            )
            
            # Compute standard errors
            std_errors = self._compute_std_errors(result.x, data, backcast_vech)
            
            # Mark as fitted
            self._fitted = True
            self._residuals = data
            self._backcast = backcast
            
            # Create result object
            model_result = self._create_result_object(
                parameters=params,
                data=data,
                covariance=self._conditional_covariances,
                correlation=self._conditional_correlations,
                loglikelihood=-result.fun,
                std_errors=std_errors,
                iterations=result.nit,
                convergence=result.success,
                optimization_message=result.message,
                unconditional_covariance=self._unconditional_covariance
            )
            
            return model_result
        
        except Exception as e:
            raise RuntimeError(f"Model estimation failed: {str(e)}") from e
    
    async def fit_async(
        self,
        data: np.ndarray,
        starting_values: Optional[Union[np.ndarray, ScalarVTVECHParameters]] = None,
        backcast: Optional[np.ndarray] = None,
        method: str = "SLSQP",
        options: Optional[Dict[str, Any]] = None,
        constraints: Optional[List[Dict[str, Any]]] = None,
        callback: Optional[AsyncProgressCallback] = None,
        **kwargs: Any
    ) -> MultivariateVolatilityResult:
        """
        Asynchronously fit the Scalar VT-VECH model to the provided data.
        
        This method provides an asynchronous interface to the fit method,
        allowing for non-blocking model estimation in UI contexts.
        
        Args:
            data: Residual data for model fitting (T x n_assets)
            starting_values: Initial parameter values for optimization
            backcast: Value to use for initializing the covariance process
            method: Optimization method to use
            options: Additional options for the optimizer
            constraints: Constraints for the optimizer
            callback: Async callback function for reporting optimization progress
            **kwargs: Additional keyword arguments for model fitting
        
        Returns:
            MultivariateVolatilityResult: The model estimation results
        
        Raises:
            ValueError: If the data is invalid
            RuntimeError: If the model fails to converge
        """
        # Create a synchronous callback that calls the async callback
        sync_callback = None
        if callback is not None:
            import asyncio
            
            def sync_callback_wrapper(progress: float, message: str) -> None:
                asyncio.create_task(callback(progress, message))
            
            sync_callback = sync_callback_wrapper
        
        # Run the synchronous fit method in an executor to avoid blocking
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.fit(
                data=data,
                starting_values=starting_values,
                backcast=backcast,
                method=method,
                options=options,
                constraints=constraints,
                callback=sync_callback,
                **kwargs
            )
        )
        
        return result
    
    def compute_covariance(
        self,
        parameters: ScalarVTVECHParameters,
        data: np.ndarray,
        sigma: Optional[np.ndarray] = None,
        backcast: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute conditional covariance matrices for the given parameters and data.
        
        Args:
            parameters: Model parameters
            data: Input data (typically residuals)
            sigma: Pre-allocated array for conditional covariances
            backcast: Matrix to use for initializing the covariance process
        
        Returns:
            np.ndarray: Conditional covariance matrices (n_assets x n_assets x T)
        """
        # Validate parameters
        parameters.validate()
        
        # Extract parameters
        a, b = parameters.a, parameters.b
        
        # Validate data
        T, n_assets = validate_multivariate_data(data)
        
        # Update n_assets if not already set
        if self._n_assets is None:
            self._n_assets = n_assets
        elif self._n_assets != n_assets:
            raise ValueError(
                f"Data has {n_assets} assets, but model was initialized with {self._n_assets} assets"
            )
        
        # Use stored unconditional covariance if available, otherwise compute it
        if self._unconditional_covariance is None or self._unconditional_vech is None:
            self._unconditional_covariance = compute_sample_covariance(data)
            self._unconditional_covariance = ensure_positive_definite(
                self._unconditional_covariance, method="nearest"
            )
            self._unconditional_vech = vech(self._unconditional_covariance)
        
        # Compute backcast if not provided
        if backcast is None:
            backcast = self._unconditional_covariance
        
        # Vectorize backcast
        backcast_vech = vech(backcast)
        
        # Initialize sigma if not provided
        if sigma is None:
            sigma = np.zeros((n_assets, n_assets, T))
        
        # Compute vectorized conditional covariances
        n_vech = n_assets * (n_assets + 1) // 2
        sigma_vech = np.zeros((n_vech, T))
        
        if HAS_NUMBA:
            # Use Numba-accelerated implementation if available
            sigma_vech = _scalar_vt_vech_recursion_numba(
                data, a, b, self._unconditional_vech, sigma_vech, backcast_vech
            )
        else:
            # Use pure NumPy implementation
            sigma_vech = _scalar_vt_vech_recursion(
                data, a, b, self._unconditional_vech, sigma_vech, backcast_vech
            )
        
        # Convert vectorized covariances to matrix form
        for t in range(T):
            sigma[:, :, t] = ivech(sigma_vech[:, t])
        
        return sigma
    
    def simulate(
        self,
        n_periods: int,
        burn: int = 0,
        initial_value: Optional[np.ndarray] = None,
        random_state: Optional[Union[int, np.random.Generator]] = None,
        return_covariances: bool = False,
        **kwargs: Any
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Simulate data from the Scalar VT-VECH model.
        
        Args:
            n_periods: Number of periods to simulate
            burn: Number of initial observations to discard
            initial_value: Initial covariance matrix for the simulation
            random_state: Random number generator or seed
            return_covariances: Whether to return conditional covariances
            **kwargs: Additional keyword arguments for simulation
        
        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: 
                Simulated data and optionally conditional covariances
        
        Raises:
            RuntimeError: If the model has not been fitted
            ValueError: If the simulation parameters are invalid
        """
        if not self._fitted or self._parameters is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        # Extract parameters
        params = cast(ScalarVTVECHParameters, self._parameters)
        a, b = params.a, params.b
        
        # Check if n_assets is set
        if self._n_assets is None:
            raise ValueError("Number of assets is not set")
        
        # Check if unconditional covariance is available
        if self._unconditional_vech is None:
            raise ValueError("Unconditional covariance is not available")
        
        # Vectorize initial value if provided
        initial_value_vech = None
        if initial_value is not None:
            # Validate initial value
            if initial_value.shape != (self._n_assets, self._n_assets):
                raise ValueError(
                    f"Initial value must have shape ({self._n_assets}, {self._n_assets}), "
                    f"got {initial_value.shape}"
                )
            
            # Ensure positive definiteness
            initial_value = ensure_positive_definite(initial_value, method="nearest")
            
            # Vectorize
            initial_value_vech = vech(initial_value)
        
        # Simulate data
        if HAS_NUMBA:
            # Use Numba-accelerated implementation if available
            result = _scalar_vt_vech_simulate_numba(
                n_periods, a, b, self._unconditional_vech, self._n_assets,
                burn, initial_value_vech, random_state, return_covariances
            )
        else:
            # Use pure NumPy implementation
            result = _scalar_vt_vech_simulate(
                n_periods, a, b, self._unconditional_vech, self._n_assets,
                burn, initial_value_vech, random_state, return_covariances
            )
        
        return result
    
    async def simulate_async(
        self,
        n_periods: int,
        burn: int = 0,
        initial_value: Optional[np.ndarray] = None,
        random_state: Optional[Union[int, np.random.Generator]] = None,
        return_covariances: bool = False,
        callback: Optional[AsyncProgressCallback] = None,
        **kwargs: Any
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Asynchronously simulate data from the Scalar VT-VECH model.
        
        This method provides an asynchronous interface to the simulate method,
        allowing for non-blocking simulation in UI contexts.
        
        Args:
            n_periods: Number of periods to simulate
            burn: Number of initial observations to discard
            initial_value: Initial covariance matrix for the simulation
            random_state: Random number generator or seed
            return_covariances: Whether to return conditional covariances
            callback: Async callback function for reporting simulation progress
            **kwargs: Additional keyword arguments for simulation
        
        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: 
                Simulated data and optionally conditional covariances
        
        Raises:
            RuntimeError: If the model has not been fitted
            ValueError: If the simulation parameters are invalid
        """
        # Create a synchronous callback that calls the async callback
        sync_callback = None
        if callback is not None:
            import asyncio
            
            def sync_callback_wrapper(progress: float, message: str) -> None:
                asyncio.create_task(callback(progress, message))
            
            # Wrap the simulate method to call the callback
            original_simulate = self.simulate
            
            def simulate_with_callback(*args: Any, **kwargs: Any) -> Any:
                total_steps = n_periods + burn
                for i in range(total_steps):
                    if i % max(1, total_steps // 10) == 0:
                        progress = i / total_steps
                        sync_callback_wrapper(progress, f"Simulating period {i+1}/{total_steps}")
                return original_simulate(*args, **kwargs)
            
            # Run the wrapped simulate method in an executor
            import asyncio
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: simulate_with_callback(
                    n_periods=n_periods,
                    burn=burn,
                    initial_value=initial_value,
                    random_state=random_state,
                    return_covariances=return_covariances,
                    **kwargs
                )
            )
        else:
            # Run the original simulate method in an executor
            import asyncio
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.simulate(
                    n_periods=n_periods,
                    burn=burn,
                    initial_value=initial_value,
                    random_state=random_state,
                    return_covariances=return_covariances,
                    **kwargs
                )
            )
        
        return result
    
    def forecast(
        self,
        steps: int,
        initial_value: Optional[np.ndarray] = None,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Forecast conditional covariance matrices.
        
        Args:
            steps: Number of steps to forecast
            initial_value: Initial covariance matrix for forecasting
            **kwargs: Additional keyword arguments for forecasting
        
        Returns:
            np.ndarray: Forecasted conditional covariance matrices (n_assets x n_assets x steps)
        
        Raises:
            RuntimeError: If the model has not been fitted
            ValueError: If the forecast parameters are invalid
        """
        if not self._fitted or self._parameters is None or self._residuals is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        # Extract parameters
        params = cast(ScalarVTVECHParameters, self._parameters)
        a, b = params.a, params.b
        
        # Check if n_assets is set
        if self._n_assets is None:
            raise ValueError("Number of assets is not set")
        
        # Check if unconditional covariance is available
        if self._unconditional_vech is None:
            raise ValueError("Unconditional covariance is not available")
        
        # Use last conditional covariance if initial_value is not provided
        if initial_value is None:
            if self._conditional_covariances is None:
                raise ValueError("Conditional covariances are not available")
            
            initial_value = self._conditional_covariances[:, :, -1]
        
        # Validate initial value
        if initial_value.shape != (self._n_assets, self._n_assets):
            raise ValueError(
                f"Initial value must have shape ({self._n_assets}, {self._n_assets}), "
                f"got {initial_value.shape}"
            )
        
        # Ensure positive definiteness
        initial_value = ensure_positive_definite(initial_value, method="nearest")
        
        # Vectorize initial value
        initial_value_vech = vech(initial_value)
        
        # Get last observation
        last_data = self._residuals[-1]
        
        # Forecast conditional covariances
        if HAS_NUMBA:
            # Use Numba-accelerated implementation if available
            forecast = _scalar_vt_vech_forecast_numba(
                a, b, self._unconditional_vech, initial_value_vech,
                last_data, steps, self._n_assets
            )
        else:
            # Use pure NumPy implementation
            forecast = _scalar_vt_vech_forecast(
                a, b, self._unconditional_vech, initial_value_vech,
                last_data, steps, self._n_assets
            )
        
        return forecast
    
    async def forecast_async(
        self,
        steps: int,
        initial_value: Optional[np.ndarray] = None,
        callback: Optional[AsyncProgressCallback] = None,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Asynchronously forecast conditional covariance matrices.
        
        This method provides an asynchronous interface to the forecast method,
        allowing for non-blocking forecasting in UI contexts.
        
        Args:
            steps: Number of steps to forecast
            initial_value: Initial covariance matrix for forecasting
            callback: Async callback function for reporting forecast progress
            **kwargs: Additional keyword arguments for forecasting
        
        Returns:
            np.ndarray: Forecasted conditional covariance matrices (n_assets x n_assets x steps)
        
        Raises:
            RuntimeError: If the model has not been fitted
            ValueError: If the forecast parameters are invalid
        """
        # Create a synchronous callback that calls the async callback
        sync_callback = None
        if callback is not None:
            import asyncio
            
            def sync_callback_wrapper(progress: float, message: str) -> None:
                asyncio.create_task(callback(progress, message))
            
            # Wrap the forecast method to call the callback
            original_forecast = self.forecast
            
            def forecast_with_callback(*args: Any, **kwargs: Any) -> Any:
                for i in range(steps):
                    if i % max(1, steps // 10) == 0:
                        progress = i / steps
                        sync_callback_wrapper(progress, f"Forecasting step {i+1}/{steps}")
                return original_forecast(*args, **kwargs)
            
            # Run the wrapped forecast method in an executor
            import asyncio
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: forecast_with_callback(
                    steps=steps,
                    initial_value=initial_value,
                    **kwargs
                )
            )
        else:
            # Run the original forecast method in an executor
            import asyncio
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.forecast(
                    steps=steps,
                    initial_value=initial_value,
                    **kwargs
                )
            )
        
        return result
    
    def _array_to_parameters(self, array: np.ndarray) -> ScalarVTVECHParameters:
        """
        Convert a parameter array to a parameter object.
        
        Args:
            array: Parameter array [a, b]
        
        Returns:
            ScalarVTVECHParameters: Parameter object
        
        Raises:
            ValueError: If the array length is not 2
        """
        if len(array) != 2:
            raise ValueError(f"Array length must be 2, got {len(array)}")
        
        return ScalarVTVECHParameters(a=array[0], b=array[1])
    
    def _parameters_to_array(self, parameters: ScalarVTVECHParameters) -> np.ndarray:
        """
        Convert a parameter object to a parameter array.
        
        Args:
            parameters: Parameter object
        
        Returns:
            np.ndarray: Parameter array [a, b]
        """
        return parameters.to_array()
    
    def _create_constraints(self) -> List[Dict[str, Any]]:
        """
        Create constraints for optimization.
        
        Returns:
            List[Dict[str, Any]]: List of constraints for optimization
        """
        # No explicit constraints needed for transformed parameters
        # The transformation ensures a + b < 1 and a, b >= 0
        return []
    
    def _create_starting_values(self, data: np.ndarray) -> np.ndarray:
        """
        Create starting values for optimization.
        
        Args:
            data: Input data (typically residuals)
        
        Returns:
            np.ndarray: Starting values for optimization
        """
        # Default starting values: a = 0.05, b = 0.85
        return np.array([0.05, 0.85])
    
    def _compute_std_errors(
        self,
        parameters: np.ndarray,
        data: np.ndarray,
        backcast_vech: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute standard errors for parameter estimates.
        
        Args:
            parameters: Parameter array in unconstrained space
            data: Input data (typically residuals)
            backcast_vech: Vectorized backcast value for initializing the recursion
        
        Returns:
            np.ndarray: Standard errors for parameter estimates
        """
        # Compute Hessian using finite differences
        def objective(params: np.ndarray) -> float:
            # Convert parameters from unconstrained space
            params_obj = ScalarVTVECHParameters.inverse_transform(params)
            a, b = params_obj.a, params_obj.b
            
            # Compute negative log-likelihood
            try:
                if HAS_NUMBA:
                    # Use Numba-accelerated implementation if available
                    loglik = _scalar_vt_vech_likelihood_numba(
                        data, a, b, self._unconditional_vech, None, backcast_vech, False
                    )
                else:
                    # Use pure NumPy implementation
                    loglik = _scalar_vt_vech_likelihood(
                        data, a, b, self._unconditional_vech, None, backcast_vech, False
                    )
                
                return -loglik
            except (ValueError, ParameterError, np.linalg.LinAlgError) as e:
                # Return a large value if parameters are invalid
                return 1e10
        
        try:
            # Compute Hessian using finite differences
            hessian = optimize.approx_fprime(
                parameters,
                lambda params: optimize.approx_fprime(
                    params,
                    objective,
                    epsilon=1e-5
                ),
                epsilon=1e-5
            )
            
            # Compute covariance matrix as inverse of Hessian
            try:
                cov_matrix = np.linalg.inv(hessian)
                # Extract standard errors as square root of diagonal elements
                std_errors = np.sqrt(np.diag(cov_matrix))
                
                # Convert standard errors to original parameter space
                # This is an approximation using the delta method
                params_obj = ScalarVTVECHParameters.inverse_transform(parameters)
                a, b = params_obj.a, params_obj.b
                
                # Compute Jacobian of the inverse transformation
                lambda_param = a + b
                delta_param = a / lambda_param if lambda_param > 0 else 0.5
                
                # Partial derivatives of a and b with respect to lambda and delta
                da_dlambda = delta_param
                da_ddelta = lambda_param
                db_dlambda = 1 - delta_param
                db_ddelta = -lambda_param
                
                # Jacobian matrix
                jacobian = np.array([
                    [da_dlambda, da_ddelta],
                    [db_dlambda, db_ddelta]
                ])
                
                # Transform covariance matrix to original parameter space
                cov_matrix_orig = jacobian @ cov_matrix @ jacobian.T
                
                # Extract standard errors in original parameter space
                std_errors_orig = np.sqrt(np.diag(cov_matrix_orig))
                
                return std_errors_orig
            except np.linalg.LinAlgError:
                # If Hessian is singular, use pseudo-inverse
                cov_matrix = np.linalg.pinv(hessian)
                std_errors = np.sqrt(np.diag(cov_matrix))
                warnings.warn(
                    "Hessian matrix is singular. Standard errors may be unreliable.",
                    UserWarning
                )
                return std_errors
        except Exception as e:
            # If Hessian computation fails, return NaN values
            warnings.warn(
                f"Failed to compute standard errors: {str(e)}",
                UserWarning
            )
            return np.full(len(parameters), np.nan)
    
    def _compute_persistence(self) -> Optional[float]:
        """
        Compute the persistence of the model.
        
        Returns:
            Optional[float]: Persistence value
        """
        if not self._fitted or self._parameters is None:
            return None
        
        # Extract parameters
        params = cast(ScalarVTVECHParameters, self._parameters)
        
        # For Scalar VT-VECH, persistence is a + b
        return params.a + params.b
    
    def _compute_unconditional_covariance(self) -> Optional[np.ndarray]:
        """
        Compute the unconditional covariance matrix implied by the model.
        
        Returns:
            Optional[np.ndarray]: Unconditional covariance matrix
        """
        # For Scalar VT-VECH, the unconditional covariance is estimated from the data
        # and used in the model specification (variance targeting)
        return self._unconditional_covariance
