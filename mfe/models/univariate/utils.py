# mfe/models/univariate/utils.py
"""
Utility functions for univariate volatility models.

This module provides shared utility functions for univariate volatility models in the MFE Toolbox,
including starting value generation, parameter validation, backcast computation, and diagnostic
statistics. These functions are used across different volatility model implementations to ensure
consistent behavior and avoid code duplication.

The module leverages NumPy's vectorized operations for efficient computation and Numba's JIT
compilation for performance-critical functions. All functions include comprehensive type hints
and input validation to ensure reliability and proper error handling.
"""

import logging
import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union, cast, overload
import numpy as np
from numba import jit, float64
from scipy import optimize, stats

from mfe.core.parameters import (
    UnivariateVolatilityParameters, GARCHParameters, EGARCHParameters, TARCHParameters, APARCHParameters,
    validate_positive, validate_non_negative, validate_probability, validate_range, ParameterError
)
from mfe.core.exceptions import (
    MFEError, NumericError, EstimationError, raise_numeric_error, warn_numeric, warn_model
)
from mfe.utils.matrix_ops import vech, ivech

# Set up module-level logger
logger = logging.getLogger("mfe.models.univariate.utils")


@jit(nopython=True, cache=True)
def _compute_backcast_numba(data: np.ndarray, decay: float = 0.94) -> float:
    """
    Compute backcast value for volatility initialization using exponential decay.

    This Numba-accelerated function computes a weighted average of squared returns
    with exponentially decaying weights to provide a robust initial value for
    volatility recursions.

    Args:
        data: Input data (typically residuals)
        decay: Decay factor for exponential weighting (0 < decay < 1)

    Returns:
        float: Backcast value for volatility initialization
    """
    n = len(data)
    weights = np.zeros(n)

    # Compute exponentially decaying weights
    for i in range(n):
        weights[i] = decay ** (n - i - 1)

    # Normalize weights to sum to 1
    weights = weights / np.sum(weights)

    # Compute weighted average of squared returns
    backcast = 0.0
    for i in range(n):
        backcast += weights[i] * data[i] ** 2

    return backcast


@jit(float64(float64[:], float64), nopython=True, cache=True)
def _compute_backcast_numba(data, decay):
    """Compute a variance backcast using exponential smoothing.
    
    Args:
        data: Input data (typically squared returns)
        decay: Decay factor for exponential smoothing
    
    Returns:
        float: Backcast value
    """
    n = len(data)
    weights = np.ones(n)
    
    # Compute exponentially decaying weights
    for i in range(1, n):
        weights[i] = weights[i-1] * decay
    
    # Normalize weights to sum to 1
    weights = weights / np.sum(weights)
    
    # Compute weighted average
    backcast = 0.0
    for i in range(n):
        backcast += weights[n-i-1] * data[i]
    
    return backcast


def compute_backcast(data: np.ndarray, method: str = "exponential", **kwargs: Any) -> float:
    """
    Compute backcast value for volatility initialization.

    This function computes an initial value for the conditional variance process
    using various methods. The backcast value is used to initialize the recursion
    for computing conditional variances in volatility models.

    Args:
        data: Input data (typically residuals)
        method: Method to use for computing the backcast value
            - "simple": Simple average of squared returns
            - "exponential": Exponentially weighted average of squared returns
            - "garch": GARCH-consistent backcast (weighted average based on persistence)
        **kwargs: Additional keyword arguments for specific methods
            - decay: Decay factor for exponential weighting (default: 0.94)
            - persistence: Persistence parameter for GARCH-consistent method

    Returns:
        float: Backcast value for volatility initialization

    Raises:
        ValueError: If the method is not recognized or if data is invalid

    Examples:
        >>> import numpy as np
        >>> from mfe.models.univariate.utils import compute_backcast
        >>> data = np.random.normal(0, 1, 100)
        >>> backcast = compute_backcast(data, method="simple")
        >>> backcast_exp = compute_backcast(data, method="exponential", decay=0.95)
    """
    # Validate input data
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

    if data.ndim != 1:
        raise ValueError(f"data must be a 1-dimensional array, got shape {data.shape}")

    if len(data) == 0:
        raise ValueError("data must not be empty")

    # Compute backcast based on method
    if method == "simple":
        # Simple average of squared returns
        return np.mean(data ** 2)

    elif method == "exponential":
        # Exponentially weighted average of squared returns
        decay = kwargs.get("decay", 0.94)

        # Validate decay parameter
        if not 0 < decay < 1:
            raise ValueError(f"decay must be between 0 and 1, got {decay}")

        # Use Numba-accelerated implementation
        return _compute_backcast_numba(data, decay)

    elif method == "garch":
        # GARCH-consistent backcast (weighted average based on persistence)
        persistence = kwargs.get("persistence", 0.94)

        # Validate persistence parameter
        if not 0 <= persistence < 1:
            raise ValueError(f"persistence must be between 0 and 1, got {persistence}")

        # Compute unconditional variance
        unconditional_variance = np.mean(data ** 2)

        # For very high persistence, use more weight on the sample variance
        if persistence > 0.98:
            return unconditional_variance

        # Compute weighted average based on persistence
        weights = np.zeros_like(data)
        for i in range(len(data)):
            weights[i] = persistence ** (len(data) - i - 1)

        # Normalize weights
        weights = weights / np.sum(weights)

        # Compute weighted average
        return np.sum(weights * data ** 2)

    else:
        raise ValueError(f"Unknown method: {method}. Supported methods are 'simple', 'exponential', and 'garch'.")


def backcast_variance(data: np.ndarray, 
                     model_type: str = "garch", 
                     persistence: Optional[float] = None, 
                     power: float = 2.0) -> float:
    """
    Compute a model-specific backcast value for initializing variance processes.
    
    This function computes an appropriate backcast value for initializing the
    conditional variance process in different volatility models. It takes into
    account the specific characteristics of each model type.
    
    Args:
        data: Input data (typically residuals)
        model_type: Type of volatility model ("garch", "egarch", "tarch", "aparch", etc.)
        persistence: Model persistence parameter (alpha + beta for GARCH)
        power: Power parameter for models like APARCH
    
    Returns:
        float: Backcast value for initializing the variance process
    
    Raises:
        ValueError: If the model type is not recognized or if data is invalid
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.univariate.utils import backcast_variance
        >>> data = np.random.normal(0, 1, 100)
        >>> backcast_garch = backcast_variance(data, model_type="garch", persistence=0.95)
        >>> backcast_egarch = backcast_variance(data, model_type="egarch")
    """
    # Validate input data
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

    if data.ndim != 1:
        raise ValueError(f"data must be a 1-dimensional array, got shape {data.shape}")

    if len(data) == 0:
        raise ValueError("data must not be empty")
    
    # Default persistence if not provided
    if persistence is None:
        persistence = 0.94
    
    # Validate persistence
    if not 0 <= persistence <= 1:
        raise ValueError(f"persistence must be between 0 and 1, got {persistence}")
    
    # Compute model-specific backcast
    model_type = model_type.lower()
    
    if model_type in ["garch", "igarch", "figarch"]:
        # For GARCH-type models, use GARCH-consistent backcast
        return compute_backcast(data, method="garch", persistence=persistence)
    
    elif model_type == "egarch":
        # For EGARCH, compute backcast for log-variance
        # First, compute variance backcast
        variance_backcast = compute_backcast(data, method="exponential", decay=persistence)
        
        # Convert to log-variance, with a small constant to avoid log(0)
        return np.log(max(variance_backcast, 1e-6))
    
    elif model_type == "tarch":
        # For TARCH, use standard GARCH backcast
        return compute_backcast(data, method="garch", persistence=persistence)
    
    elif model_type == "aparch":
        # For APARCH, compute backcast for power-transformed variance
        # First, compute absolute returns raised to power
        abs_returns_power = np.abs(data) ** power
        
        # Then compute backcast using these transformed returns
        return compute_backcast(abs_returns_power, method="exponential", decay=persistence)
    
    else:
        # For unknown models, use simple exponential backcast
        return compute_backcast(data, method="exponential", decay=persistence)


@jit(nopython=True, cache=True)
def _grid_search_garch_numba(data: np.ndarray,
                             omega_values: np.ndarray,
                             alpha_values: np.ndarray,
                             beta_values: np.ndarray,
                             backcast: float) -> Tuple[float, float, float, float]:
    """
    Perform grid search for GARCH(1,1) starting values using Numba acceleration.

    This function evaluates the log-likelihood of a GARCH(1,1) model for different
    parameter combinations and returns the parameters with the highest likelihood.

    Args:
        data: Input data (typically residuals)
        omega_values: Array of omega values to try
        alpha_values: Array of alpha values to try
        beta_values: Array of beta values to try
        backcast: Value to use for initializing the variance process

    Returns:
        Tuple[float, float, float, float]: Best omega, alpha, beta, and corresponding log-likelihood
    """
    n = len(data)
    best_loglik = -np.inf
    best_omega = omega_values[0]
    best_alpha = alpha_values[0]
    best_beta = beta_values[0]

    # Pre-allocate array for conditional variances
    sigma2 = np.zeros(n)

    # Try all parameter combinations
    for omega in omega_values:
        for alpha in alpha_values:
            for beta in beta_values:
                # Skip if stationarity constraint is violated
                if alpha + beta >= 1:
                    continue

                # Initialize first variance with backcast
                sigma2[0] = backcast

                # Compute conditional variances
                for t in range(1, n):
                    sigma2[t] = omega + alpha * data[t-1]**2 + beta * sigma2[t-1]

                # Check for invalid variances
                if np.any(sigma2 <= 0):
                    continue

                # Compute log-likelihood
                loglik = -0.5 * np.sum(np.log(sigma2) + data**2 / sigma2)

                # Update best parameters if likelihood is higher
                if loglik > best_loglik:
                    best_loglik = loglik
                    best_omega = omega
                    best_alpha = alpha
                    best_beta = beta

    return best_omega, best_alpha, best_beta, best_loglik


def grid_search_starting_values(data: np.ndarray,
                                model_type: str = "garch",
                                p: int = 1,
                                q: int = 1,
                                **kwargs: Any) -> Dict[str, Any]:
    """
    Perform grid search to find good starting values for volatility model estimation.

    This function evaluates the log-likelihood of a volatility model for different
    parameter combinations and returns the parameters with the highest likelihood.

    Args:
        data: Input data (typically residuals)
        model_type: Type of volatility model ("garch", "egarch", "tarch", "aparch")
        p: Order of the ARCH component
        q: Order of the GARCH component
        **kwargs: Additional keyword arguments
            - n_points: Number of grid points for each parameter (default: 5)
            - backcast: Value to use for initializing the variance process
            - omega_range: Tuple of (min, max) for omega grid
            - alpha_range: Tuple of (min, max) for alpha grid
            - beta_range: Tuple of (min, max) for beta grid
            - gamma_range: Tuple of (min, max) for gamma grid (for asymmetric models)
            - delta_range: Tuple of (min, max) for delta grid (for APARCH model)

    Returns:
        Dict[str, Any]: Dictionary of best parameter values and log-likelihood

    Raises:
        ValueError: If the model type is not recognized or if data is invalid

    Examples:
        >>> import numpy as np
        >>> from mfe.models.univariate.utils import grid_search_starting_values
        >>> data = np.random.normal(0, 1, 1000)
        >>> params = grid_search_starting_values(data, model_type="garch")
        >>> print(f"Best omega: {params['omega']:.4f}, alpha: {params['alpha']:.4f}, beta: {params['beta']:.4f}")
    """
    # Validate input data
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

    if data.ndim != 1:
        raise ValueError(f"data must be a 1-dimensional array, got shape {data.shape}")

    if len(data) == 0:
        raise ValueError("data must not be empty")

    # Get number of grid points
    n_points = kwargs.get("n_points", 5)

    # Compute backcast if not provided
    backcast = kwargs.get("backcast", compute_backcast(data))

    # Compute sample variance for scaling
    sample_variance = np.var(data)

    # Handle different model types
    model_type = model_type.lower()

    if model_type == "garch" and p == 1 and q == 1:
        # GARCH(1,1) case - use Numba-accelerated grid search

        # Define parameter grids
        omega_range = kwargs.get("omega_range", (0.01 * sample_variance, 0.2 * sample_variance))
        alpha_range = kwargs.get("alpha_range", (0.01, 0.2))
        beta_range = kwargs.get("beta_range", (0.7, 0.98))

        omega_values = np.linspace(omega_range[0], omega_range[1], n_points)
        alpha_values = np.linspace(alpha_range[0], alpha_range[1], n_points)
        beta_values = np.linspace(beta_range[0], beta_range[1], n_points)

        # Perform grid search
        best_omega, best_alpha, best_beta, best_loglik = _grid_search_garch_numba(
            data, omega_values, alpha_values, beta_values, backcast
        )

        # Return best parameters
        return {
            "omega": best_omega,
            "alpha": best_alpha,
            "beta": best_beta,
            "loglikelihood": best_loglik
        }

    elif model_type == "garch":
        # GARCH(p,q) case - use reasonable defaults based on experience

        # For higher-order models, distribute alpha and beta with exponential decay
        alpha_total = 0.05
        beta_total = 0.90

        alpha = np.zeros(p)
        beta = np.zeros(q)

        # Distribute alpha values with exponential decay
        if p > 0:
            alpha_weights = np.exp(-np.arange(p))
            alpha_weights = alpha_weights / np.sum(alpha_weights)
            alpha = alpha_weights * alpha_total

        # Distribute beta values with exponential decay
        if q > 0:
            beta_weights = np.exp(-np.arange(q))
            beta_weights = beta_weights / np.sum(beta_weights)
            beta = beta_weights * beta_total

        # Set omega based on unconditional variance
        omega = sample_variance * (1 - np.sum(alpha) - np.sum(beta))

        # Return parameters
        return {
            "omega": omega,
            "alpha": alpha,
            "beta": beta
        }

    elif model_type == "egarch" and p == 1 and q == 1:
        # EGARCH(1,1) case

        # Define parameter grids
        omega_range = kwargs.get("omega_range", (-0.1, 0.1))
        alpha_range = kwargs.get("alpha_range", (0.01, 0.2))
        gamma_range = kwargs.get("gamma_range", (-0.2, 0.2))
        beta_range = kwargs.get("beta_range", (0.7, 0.98))

        # Create parameter grids
        omega_values = np.linspace(omega_range[0], omega_range[1], n_points)
        alpha_values = np.linspace(alpha_range[0], alpha_range[1], n_points)
        gamma_values = np.linspace(gamma_range[0], gamma_range[1], n_points)
        beta_values = np.linspace(beta_range[0], beta_range[1], n_points)

        # Initialize best parameters
        best_params = {
            "omega": omega_values[n_points // 2],
            "alpha": alpha_values[n_points // 2],
            "gamma": gamma_values[n_points // 2],
            "beta": beta_values[n_points // 2],
            "loglikelihood": -np.inf
        }

        # Simple grid search (not Numba-accelerated for EGARCH)
        for omega in omega_values:
            for alpha in alpha_values:
                for gamma in gamma_values:
                    for beta in beta_values:
                        # Skip if |beta| >= 1 (stationarity constraint)
                        if abs(beta) >= 1:
                            continue

                        # Compute log-variance
                        log_sigma2 = np.zeros(len(data))
                        log_sigma2[0] = np.log(backcast)

                        # EGARCH recursion
                        for t in range(1, len(data)):
                            z_t_1 = data[t-1] / np.sqrt(np.exp(log_sigma2[t-1]))
                            log_sigma2[t] = omega + beta * log_sigma2[t-1] + alpha * \
                                (abs(z_t_1) - np.sqrt(2/np.pi)) + gamma * z_t_1

                        # Convert to variance
                        sigma2 = np.exp(log_sigma2)

                        # Compute log-likelihood
                        loglik = -0.5 * np.sum(np.log(sigma2) + data**2 / sigma2)

                        # Update best parameters if likelihood is higher
                        if loglik > best_params["loglikelihood"]:
                            best_params["omega"] = omega
                            best_params["alpha"] = alpha
                            best_params["gamma"] = gamma
                            best_params["beta"] = beta
                            best_params["loglikelihood"] = loglik

        return best_params

    elif model_type == "tarch" and p == 1 and q == 1:
        # TARCH(1,1) case

        # Define parameter grids
        omega_range = kwargs.get("omega_range", (0.01 * sample_variance, 0.2 * sample_variance))
        alpha_range = kwargs.get("alpha_range", (0.01, 0.1))
        gamma_range = kwargs.get("gamma_range", (0.01, 0.2))
        beta_range = kwargs.get("beta_range", (0.7, 0.95))

        # Create parameter grids
        omega_values = np.linspace(omega_range[0], omega_range[1], n_points)
        alpha_values = np.linspace(alpha_range[0], alpha_range[1], n_points)
        gamma_values = np.linspace(gamma_range[0], gamma_range[1], n_points)
        beta_values = np.linspace(beta_range[0], beta_range[1], n_points)

        # Initialize best parameters
        best_params = {
            "omega": omega_values[n_points // 2],
            "alpha": alpha_values[n_points // 2],
            "gamma": gamma_values[n_points // 2],
            "beta": beta_values[n_points // 2],
            "loglikelihood": -np.inf
        }

        # Simple grid search
        for omega in omega_values:
            for alpha in alpha_values:
                for gamma in gamma_values:
                    for beta in beta_values:
                        # Skip if stationarity constraint is violated
                        if alpha + beta + 0.5 * gamma >= 1:
                            continue

                        # Compute conditional variances
                        sigma2 = np.zeros(len(data))
                        sigma2[0] = backcast

                        # TARCH recursion
                        for t in range(1, len(data)):
                            # Indicator for negative returns
                            I_t_1 = 1 if data[t-1] < 0 else 0
                            sigma2[t] = omega + alpha * data[t-1]**2 + gamma * data[t-1]**2 * I_t_1 + beta * sigma2[t-1]

                        # Check for invalid variances
                        if np.any(sigma2 <= 0):
                            continue

                        # Compute log-likelihood
                        loglik = -0.5 * np.sum(np.log(sigma2) + data**2 / sigma2)

                        # Update best parameters if likelihood is higher
                        if loglik > best_params["loglikelihood"]:
                            best_params["omega"] = omega
                            best_params["alpha"] = alpha
                            best_params["gamma"] = gamma
                            best_params["beta"] = beta
                            best_params["loglikelihood"] = loglik

        return best_params

    elif model_type == "aparch" and p == 1 and q == 1:
        # APARCH(1,1) case

        # Define parameter grids
        omega_range = kwargs.get("omega_range", (0.01 * sample_variance, 0.2 * sample_variance))
        alpha_range = kwargs.get("alpha_range", (0.01, 0.1))
        gamma_range = kwargs.get("gamma_range", (-0.5, 0.5))
        beta_range = kwargs.get("beta_range", (0.7, 0.95))
        delta_range = kwargs.get("delta_range", (1.0, 2.0))

        # Use fewer grid points for APARCH due to computational complexity
        n_points_aparch = min(n_points, 3)

        # Create parameter grids
        omega_values = np.linspace(omega_range[0], omega_range[1], n_points_aparch)
        alpha_values = np.linspace(alpha_range[0], alpha_range[1], n_points_aparch)
        gamma_values = np.linspace(gamma_range[0], gamma_range[1], n_points_aparch)
        beta_values = np.linspace(beta_range[0], beta_range[1], n_points_aparch)
        delta_values = np.linspace(delta_range[0], delta_range[1], n_points_aparch)

        # Initialize best parameters
        best_params = {
            "omega": omega_values[n_points_aparch // 2],
            "alpha": alpha_values[n_points_aparch // 2],
            "gamma": gamma_values[n_points_aparch // 2],
            "beta": beta_values[n_points_aparch // 2],
            "delta": delta_values[n_points_aparch // 2],
            "loglikelihood": -np.inf
        }

        # Simple grid search
        for omega in omega_values:
            for alpha in alpha_values:
                for gamma in gamma_values:
                    for beta in beta_values:
                        for delta in delta_values:
                            # Skip if stationarity constraint is violated
                            if alpha + beta >= 1:
                                continue

                            # Skip if gamma is outside [-1, 1]
                            if abs(gamma) >= 1:
                                continue

                            # Compute conditional variances
                            sigma_delta = np.zeros(len(data))
                            sigma_delta[0] = backcast ** (delta / 2)

                            # APARCH recursion
                            for t in range(1, len(data)):
                                abs_return = abs(data[t-1])
                                leverage = gamma * np.sign(data[t-1])
                                sigma_delta[t] = omega + alpha * \
                                    (abs_return * (1 + leverage)) ** delta + beta * sigma_delta[t-1]

                            # Convert to variance
                            sigma2 = sigma_delta ** (2 / delta)

                            # Check for invalid variances
                            if np.any(sigma2 <= 0):
                                continue

                            # Compute log-likelihood
                            loglik = -0.5 * np.sum(np.log(sigma2) + data**2 / sigma2)

                            # Update best parameters if likelihood is higher
                            if loglik > best_params["loglikelihood"]:
                                best_params["omega"] = omega
                                best_params["alpha"] = alpha
                                best_params["gamma"] = gamma
                                best_params["beta"] = beta
                                best_params["delta"] = delta
                                best_params["loglikelihood"] = loglik

        return best_params

    else:
        # For other models or higher orders, use reasonable defaults
        if model_type == "egarch":
            return {
                "omega": -0.1,
                "alpha": 0.1,
                "gamma": 0.1,
                "beta": 0.9
            }
        elif model_type == "tarch":
            return {
                "omega": 0.05 * sample_variance,
                "alpha": 0.05,
                "gamma": 0.1,
                "beta": 0.8
            }
        elif model_type == "aparch":
            return {
                "omega": 0.05 * sample_variance,
                "alpha": 0.05,
                "gamma": 0.1,
                "beta": 0.8,
                "delta": 1.5
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")


def compute_persistence(parameters: UnivariateVolatilityParameters) -> float:
    """
    Compute the persistence of a volatility model.

    This function computes the persistence of a volatility model, which measures
    how long shocks to volatility persist. For GARCH-type models, persistence is
    typically the sum of ARCH and GARCH parameters.

    Args:
        parameters: Model parameters

    Returns:
        float: Persistence value

    Raises:
        TypeError: If the parameter type is not recognized

    Examples:
        >>> from mfe.core.parameters import GARCHParameters
        >>> from mfe.models.univariate.utils import compute_persistence
        >>> params = GARCHParameters(omega=0.05, alpha=0.1, beta=0.85)
        >>> persistence = compute_persistence(params)
        >>> print(f"Persistence: {persistence:.4f}")
        Persistence: 0.9500
    """
    if isinstance(parameters, GARCHParameters):
        # For GARCH, persistence is alpha + beta
        alpha = parameters.alpha
        beta = parameters.beta

        # Handle both scalar and array parameters
        if isinstance(alpha, (int, float)):
            alpha_sum = alpha
        else:
            alpha_sum = np.sum(alpha)

        if isinstance(beta, (int, float)):
            beta_sum = beta
        else:
            beta_sum = np.sum(beta)

        return alpha_sum + beta_sum

    elif isinstance(parameters, EGARCHParameters):
        # For EGARCH, persistence is beta
        return abs(parameters.beta)

    elif isinstance(parameters, TARCHParameters):
        # For TARCH, persistence is alpha + beta + 0.5*gamma
        return parameters.alpha + parameters.beta + 0.5 * parameters.gamma

    elif isinstance(parameters, APARCHParameters):
        # For APARCH, persistence is alpha + beta
        return parameters.alpha + parameters.beta

    else:
        raise TypeError(f"Unsupported parameter type: {type(parameters)}")


def compute_half_life(persistence: float) -> float:
    """
    Compute the half-life of volatility shocks.

    This function computes the half-life of volatility shocks, which is the time
    it takes for a shock to volatility to decay to half its initial value.

    Args:
        persistence: Persistence of the volatility model

    Returns:
        float: Half-life in time periods

    Raises:
        ValueError: If persistence is not between 0 and 1

    Examples:
        >>> from mfe.models.univariate.utils import compute_half_life
        >>> half_life = compute_half_life(0.95)
        >>> print(f"Half-life: {half_life:.2f} periods")
        Half-life: 13.51 periods
    """
    if not 0 <= persistence < 1:
        raise ValueError(f"Persistence must be between 0 and 1, got {persistence}")

    # For persistence close to 1, the half-life approaches infinity
    if persistence > 0.9999:
        return float('inf')

    # Half-life formula: log(0.5) / log(persistence)
    return np.log(0.5) / np.log(persistence)


def compute_unconditional_variance(parameters: UnivariateVolatilityParameters) -> float:
    """
    Compute the unconditional variance for a volatility model.
    
    This function computes the long-run unconditional variance implied by the
    parameters of a volatility model, which represents the steady-state level
    of volatility to which the process reverts.
    
    Args:
        parameters: Volatility model parameters
        
    Returns:
        float: Unconditional variance
        
    Raises:
        NumericError: If the model is non-stationary or parameters are invalid
        TypeError: If parameters is not a recognized volatility parameter type
    """
    try:
        # Validate parameters
        validate_volatility_parameters(parameters)
        
        # Compute persistence
        persistence = compute_persistence(parameters)
        
        # Check stationarity
        if persistence >= 1.0:
            raise NumericError(
                f"Cannot compute unconditional variance for non-stationary model with persistence {persistence:.4f}"
            )
        
        # Compute unconditional variance based on model type
        if isinstance(parameters, GARCHParameters):
            return parameters.omega / (1.0 - persistence)
        elif isinstance(parameters, EGARCHParameters):
            # For EGARCH, unconditional variance is more complex due to log formulation
            # This is an approximation
            return np.exp(parameters.omega / (1.0 - persistence))
        elif isinstance(parameters, TARCHParameters):
            # For TARCH, need to account for asymmetry effect
            # Assuming standard normal innovations, E[|z_t|] = sqrt(2/pi)
            expected_abs_z = np.sqrt(2.0 / np.pi)
            gamma_term = 0.5 * parameters.gamma * expected_abs_z
            return parameters.omega / (1.0 - parameters.beta - parameters.alpha - gamma_term)
        elif isinstance(parameters, APARCHParameters):
            # For APARCH, need to account for power transformation and asymmetry
            # This is an approximation
            expected_term = np.exp(0.5 * np.log(2.0) + 0.5 * np.log(1.0 / np.pi))
            gamma_term = parameters.gamma * expected_term
            return (parameters.omega / (1.0 - parameters.beta - parameters.alpha * expected_term - gamma_term)) ** (2.0 / parameters.delta)
        else:
            raise TypeError(f"Unsupported parameter type: {type(parameters)}")
    except Exception as e:
        if isinstance(e, NumericError):
            raise
        raise NumericError(f"Error computing unconditional variance: {str(e)}")


def volatility_process_to_variance(volatility_process: np.ndarray, power: float = 2.0) -> np.ndarray:
    """
    Convert a volatility process to variance process.
    
    This function converts a volatility process (e.g., conditional standard deviations)
    to a variance process (conditional variances) by raising the volatility to the
    specified power. This is particularly useful for models like APARCH where the
    power parameter may not be 2.
    
    Args:
        volatility_process: Array of volatility values (e.g., conditional standard deviations)
        power: Power to raise the volatility to (default is 2.0 for variance)
    
    Returns:
        np.ndarray: Variance process (volatility raised to the specified power)
    
    Raises:
        ValueError: If the volatility process contains negative values
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.univariate.utils import volatility_process_to_variance
        >>> vol = np.array([0.1, 0.2, 0.15])
        >>> volatility_process_to_variance(vol)
        array([0.01, 0.04, 0.0225])
        
        >>> volatility_process_to_variance(vol, power=1.0)  # No change
        array([0.1, 0.2, 0.15])
        
        >>> volatility_process_to_variance(vol, power=0.5)  # Square root
        array([0.31622777, 0.4472136, 0.38729833])
    """
    # Validate input
    if not isinstance(volatility_process, np.ndarray):
        volatility_process = np.asarray(volatility_process)
    
    if volatility_process.ndim != 1:
        raise ValueError(f"volatility_process must be a 1-dimensional array, got shape {volatility_process.shape}")
    
    if len(volatility_process) == 0:
        raise ValueError("volatility_process must not be empty")
    
    if np.any(volatility_process < 0):
        raise ValueError("volatility_process must contain non-negative values")
    
    # Convert volatility to variance
    return np.power(volatility_process, power)


def variance_to_volatility_process(variance_process: np.ndarray, power: float = 2.0) -> np.ndarray:
    """
    Convert a variance process to volatility process.
    
    This function converts a variance process (e.g., conditional variances)
    to a volatility process (conditional standard deviations) by taking the
    appropriate root of the variance. This is particularly useful for models
    like APARCH where the power parameter may not be 2.
    
    Args:
        variance_process: Array of variance values (e.g., conditional variances)
        power: Power to which the volatility was raised to get the variance (default is 2.0)
    
    Returns:
        np.ndarray: Volatility process (variance raised to the power of 1/power)
    
    Raises:
        ValueError: If the variance process contains negative values
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.univariate.utils import variance_to_volatility_process
        >>> var = np.array([0.01, 0.04, 0.0225])
        >>> variance_to_volatility_process(var)
        array([0.1, 0.2, 0.15])
        
        >>> variance_to_volatility_process(var, power=1.0)  # No change
        array([0.01, 0.04, 0.0225])
        
        >>> var_pow = np.array([0.31622777, 0.4472136, 0.38729833])
        >>> variance_to_volatility_process(var_pow, power=0.5)
        array([0.1, 0.2, 0.15])
    """
    # Validate input
    if not isinstance(variance_process, np.ndarray):
        variance_process = np.asarray(variance_process)
    
    if variance_process.ndim != 1:
        raise ValueError(f"variance_process must be a 1-dimensional array, got shape {variance_process.shape}")
    
    if len(variance_process) == 0:
        raise ValueError("variance_process must not be empty")
    
    if np.any(variance_process < 0):
        raise ValueError("variance_process must contain non-negative values")
    
    # Convert variance to volatility
    return np.power(variance_process, 1.0 / power)


def compute_news_impact_curve(parameters: UnivariateVolatilityParameters,
                              points: int = 100,
                              std_range: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the news impact curve for a volatility model.

    The news impact curve shows how past returns affect current volatility,
    holding constant the effect of older observations.

    Args:
        parameters: Model parameters
        points: Number of points to compute
        std_range: Range of standardized returns to consider (in standard deviations)

    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays of returns and corresponding variances

    Raises:
        TypeError: If the parameter type is not recognized

    Examples:
        >>> from mfe.core.parameters import GARCHParameters
        >>> from mfe.models.univariate.utils import compute_news_impact_curve
        >>> params = GARCHParameters(omega=0.05, alpha=0.1, beta=0.85)
        >>> returns, variances = compute_news_impact_curve(params)
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(returns, variances)
        >>> plt.xlabel('Return')
        >>> plt.ylabel('Conditional Variance')
        >>> plt.title('News Impact Curve')
        >>> plt.show()
    """
    # Create array of returns
    returns = np.linspace(-std_range, std_range, points)

    # Compute unconditional variance (for scaling)
    try:
        uncond_var = compute_unconditional_variance(parameters)
    except ValueError:
        # If model is not stationary, use a default value
        uncond_var = 1.0

    # Initialize array for variances
    variances = np.zeros_like(returns)

    if isinstance(parameters, GARCHParameters):
        # For GARCH, news impact is symmetric
        omega = parameters.omega
        alpha = parameters.alpha[0] if hasattr(parameters.alpha, "__len__") else parameters.alpha
        beta = parameters.beta[0] if hasattr(parameters.beta, "__len__") else parameters.beta

        # Compute variances
        for i, ret in enumerate(returns):
            variances[i] = omega + alpha * ret**2 + beta * uncond_var

    elif isinstance(parameters, EGARCHParameters):
        # For EGARCH, news impact can be asymmetric
        omega = parameters.omega
        alpha = parameters.alpha
        gamma = parameters.gamma
        beta = parameters.beta

        # Compute log-variances
        log_uncond_var = np.log(uncond_var)
        log_variances = np.zeros_like(returns)

        for i, ret in enumerate(returns):
            # Standardized return
            z = ret / np.sqrt(uncond_var)
            log_variances[i] = omega + beta * log_uncond_var + alpha * (abs(z) - np.sqrt(2/np.pi)) + gamma * z

        # Convert to variances
        variances = np.exp(log_variances)

    elif isinstance(parameters, TARCHParameters):
        # For TARCH, news impact is asymmetric
        omega = parameters.omega
        alpha = parameters.alpha
        gamma = parameters.gamma
        beta = parameters.beta

        # Compute variances
        for i, ret in enumerate(returns):
            # Indicator for negative returns
            I = 1 if ret < 0 else 0
            variances[i] = omega + alpha * ret**2 + gamma * ret**2 * I + beta * uncond_var

    elif isinstance(parameters, APARCHParameters):
        # For APARCH, news impact is asymmetric
        omega = parameters.omega
        alpha = parameters.alpha
        gamma = parameters.gamma
        beta = parameters.beta
        delta = parameters.delta

        # Compute power variances
        uncond_power = uncond_var ** (delta / 2)
        power_variances = np.zeros_like(returns)

        for i, ret in enumerate(returns):
            abs_ret = abs(ret)
            leverage = 1 + gamma * np.sign(ret)
            power_variances[i] = omega + alpha * (abs_ret * leverage) ** delta + beta * uncond_power

        # Convert to variances
        variances = power_variances ** (2 / delta)

    else:
        raise TypeError(f"Unsupported parameter type: {type(parameters)}")

    return returns, variances


def compute_model_diagnostics(data: np.ndarray,
                              conditional_variances: np.ndarray,
                              parameters: UnivariateVolatilityParameters,
                              log_likelihood: float) -> Dict[str, Any]:
    """
    Compute diagnostic statistics for a fitted volatility model.

    This function computes various diagnostic statistics for a fitted volatility
    model, including information criteria, standardized residuals, and tests for
    remaining ARCH effects.

    Args:
        data: Original data (residuals)
        conditional_variances: Conditional variances from the fitted model
        parameters: Model parameters
        log_likelihood: Log-likelihood of the fitted model

    Returns:
        Dict[str, Any]: Dictionary of diagnostic statistics

    Raises:
        ValueError: If input dimensions are inconsistent

    Examples:
        >>> import numpy as np
        >>> from mfe.core.parameters import GARCHParameters
        >>> from mfe.models.univariate.utils import compute_model_diagnostics
        >>> data = np.random.normal(0, 1, 1000)
        >>> params = GARCHParameters(omega=0.05, alpha=0.1, beta=0.85)
        >>> # Assume we have computed conditional variances and log-likelihood
        >>> conditional_variances = np.ones_like(data)  # Placeholder
        >>> log_likelihood = -1500  # Placeholder
        >>> diagnostics = compute_model_diagnostics(data, conditional_variances, params, log_likelihood)
        >>> print(f"AIC: {diagnostics['aic']:.2f}, BIC: {diagnostics['bic']:.2f}")
    """
    # Validate input dimensions
    if len(data) != len(conditional_variances):
        raise ValueError(f"data and conditional_variances must have the same length, "
                         f"got {len(data)} and {len(conditional_variances)}")

    # Get number of parameters
    if hasattr(parameters, "to_array"):
        n_params = len(parameters.to_array())
    else:
        # Fallback for non-standard parameter objects
        n_params = len(parameters.__dict__)

    # Get number of observations
    n_obs = len(data)

    # Compute information criteria
    aic = -2 * log_likelihood + 2 * n_params
    bic = -2 * log_likelihood + n_params * np.log(n_obs)
    hqic = -2 * log_likelihood + 2 * n_params * np.log(np.log(n_obs))

    # Compute standardized residuals
    std_residuals = data / np.sqrt(conditional_variances)

    # Compute statistics of standardized residuals
    mean = np.mean(std_residuals)
    std = np.std(std_residuals)
    skewness = stats.skew(std_residuals)
    kurtosis = stats.kurtosis(std_residuals, fisher=True) + 3  # Convert to raw kurtosis

    # Compute Jarque-Bera test for normality
    jb_stat, jb_pval = stats.jarque_bera(std_residuals)

    # Compute Ljung-Box test for autocorrelation in standardized residuals
    lb_stat, lb_pval = stats.acorr_ljungbox(std_residuals, lags=[10], return_df=False)

    # Compute Ljung-Box test for autocorrelation in squared standardized residuals
    lb_squared_stat, lb_squared_pval = stats.acorr_ljungbox(std_residuals**2, lags=[10], return_df=False)

    # Compute persistence
    persistence = compute_persistence(parameters)

    # Compute half-life if model is stationary
    if persistence < 1:
        half_life = compute_half_life(persistence)
    else:
        half_life = float('inf')

    # Compute unconditional variance if model is stationary
    try:
        uncond_var = compute_unconditional_variance(parameters)
    except ValueError:
        uncond_var = None

    # Return dictionary of diagnostics
    return {
        "aic": aic,
        "bic": bic,
        "hqic": hqic,
        "log_likelihood": log_likelihood,
        "n_params": n_params,
        "n_obs": n_obs,
        "std_residuals_mean": mean,
        "std_residuals_std": std,
        "std_residuals_skewness": skewness,
        "std_residuals_kurtosis": kurtosis,
        "jarque_bera_stat": jb_stat,
        "jarque_bera_pval": jb_pval,
        "ljung_box_stat": lb_stat[0],
        "ljung_box_pval": lb_pval[0],
        "ljung_box_squared_stat": lb_squared_stat[0],
        "ljung_box_squared_pval": lb_squared_pval[0],
        "persistence": persistence,
        "half_life": half_life,
        "unconditional_variance": uncond_var
    }


def validate_volatility_parameters(parameters: UnivariateVolatilityParameters) -> None:
    """
    Validate parameters for a volatility model.

    This function validates the parameters of a volatility model, checking that
    they satisfy model-specific constraints such as positivity and stationarity.

    Args:
        parameters: Model parameters to validate

    Raises:
        ParameterError: If parameter constraints are violated
        TypeError: If the parameter type is not recognized

    Examples:
        >>> from mfe.core.parameters import GARCHParameters
        >>> from mfe.models.univariate.utils import validate_volatility_parameters
        >>> params = GARCHParameters(omega=0.05, alpha=0.1, beta=0.85)
        >>> validate_volatility_parameters(params)  # No error if parameters are valid
    """
    # Use the validate method of the parameters object
    if hasattr(parameters, "validate"):
        parameters.validate()
    else:
        # Fallback for non-standard parameter objects
        if isinstance(parameters, dict):
            # If parameters is a dictionary, check common constraints
            if "omega" in parameters:
                validate_positive(parameters["omega"], "omega")

            if "alpha" in parameters:
                alpha = parameters["alpha"]
                if isinstance(alpha, (list, np.ndarray)):
                    for i, a in enumerate(alpha):
                        validate_non_negative(a, f"alpha[{i}]")
                else:
                    validate_non_negative(alpha, "alpha")

            if "beta" in parameters:
                beta = parameters["beta"]
                if isinstance(beta, (list, np.ndarray)):
                    for i, b in enumerate(beta):
                        validate_non_negative(b, f"beta[{i}]")
                else:
                    validate_non_negative(beta, "beta")

            # Check stationarity constraint for GARCH-type models
            if "alpha" in parameters and "beta" in parameters:
                alpha = parameters["alpha"]
                beta = parameters["beta"]

                alpha_sum = np.sum(alpha) if isinstance(alpha, (list, np.ndarray)) else alpha
                beta_sum = np.sum(beta) if isinstance(beta, (list, np.ndarray)) else beta

                if alpha_sum + beta_sum >= 1:
                    raise ParameterError(
                        f"Stationarity constraint violated: alpha + beta = {alpha_sum + beta_sum} >= 1"
                    )
        else:
            raise TypeError(f"Unsupported parameter type: {type(parameters)}")


def parameter_decorator(validation_func: Callable) -> Callable:
    """
    Decorator for parameter validation functions.

    This decorator wraps parameter validation functions to provide consistent
    error handling and logging.

    Args:
        validation_func: Function that validates parameters

    Returns:
        Callable: Decorated validation function

    Examples:
        >>> from mfe.models.univariate.utils import parameter_decorator
        >>> @parameter_decorator
        ... def validate_my_params(params):
        ...     if params['alpha'] < 0:
        ...         raise ValueError("alpha must be non-negative")
        ...     return True
        >>> params = {'alpha': -0.1}
        >>> validate_my_params(params)  # Raises ParameterError with detailed message
    """
    def wrapper(*args, **kwargs):
        try:
            return validation_func(*args, **kwargs)
        except (ValueError, TypeError) as e:
            # Convert to ParameterError with more context
            param_name = None
            param_value = None

            # Try to extract parameter name and value from error message
            error_msg = str(e)
            if "must be" in error_msg and "got" in error_msg:
                parts = error_msg.split("must be")
                if len(parts) > 1:
                    param_name = parts[0].strip()

                    value_parts = error_msg.split("got")
                    if len(value_parts) > 1:
                        try:
                            param_value = eval(value_parts[1].strip())
                        except:
                            param_value = value_parts[1].strip()

            raise ParameterError(
                str(e),
                param_name=param_name,
                param_value=param_value,
                details=f"Error in {validation_func.__name__}"
            )

    # Preserve function metadata
    wrapper.__name__ = validation_func.__name__
    wrapper.__doc__ = validation_func.__doc__

    return wrapper


@parameter_decorator
def validate_garch_parameters(omega: float, alpha: Union[float, np.ndarray], beta: Union[float, np.ndarray]) -> bool:
    """
    Validate parameters for a GARCH model.

    This function validates the parameters of a GARCH model, checking that
    they satisfy constraints such as positivity and stationarity.

    Args:
        omega: Constant term in variance equation (must be positive)
        alpha: ARCH parameter(s) (must be non-negative)
        beta: GARCH parameter(s) (must be non-negative)

    Returns:
        bool: True if parameters are valid

    Raises:
        ValueError: If parameter constraints are violated

    Examples:
        >>> from mfe.models.univariate.utils import validate_garch_parameters
        >>> validate_garch_parameters(0.05, 0.1, 0.85)  # Returns True if parameters are valid
    """
    # Validate omega
    validate_positive(omega, "omega")

    # Convert alpha and beta to arrays if they are scalars
    if isinstance(alpha, (int, float)):
        alpha = np.array([alpha])
    elif not isinstance(alpha, np.ndarray):
        alpha = np.array(alpha)

    if isinstance(beta, (int, float)):
        beta = np.array([beta])
    elif not isinstance(beta, np.ndarray):
        beta = np.array(beta)

    # Validate alpha and beta
    for i, a in enumerate(alpha):
        validate_non_negative(a, f"alpha[{i}]")

    for i, b in enumerate(beta):
        validate_non_negative(b, f"beta[{i}]")

    # Validate stationarity constraint
    persistence = np.sum(alpha) + np.sum(beta)
    if persistence >= 1:
        raise ValueError(
            f"GARCH stationarity constraint violated: sum(alpha) + sum(beta) = {persistence} >= 1"
        )

    return True


@parameter_decorator
def validate_egarch_parameters(omega: float, alpha: float, gamma: float, beta: float) -> bool:
    """
    Validate parameters for an EGARCH model.

    This function validates the parameters of an EGARCH model, checking that
    they satisfy constraints such as stationarity.

    Args:
        omega: Constant term in log-variance equation
        alpha: ARCH parameter
        gamma: Asymmetry parameter
        beta: GARCH parameter (must be between -1 and 1 for stationarity)

    Returns:
        bool: True if parameters are valid

    Raises:
        ValueError: If parameter constraints are violated

    Examples:
        >>> from mfe.models.univariate.utils import validate_egarch_parameters
        >>> validate_egarch_parameters(-0.1, 0.1, 0.1, 0.9)  # Returns True if parameters are valid
    """
    # EGARCH has fewer constraints than GARCH
    # The key constraint is |beta| < 1 for stationarity
    if abs(beta) >= 1:
        raise ValueError(
            f"EGARCH stationarity constraint violated: |beta| = {abs(beta)} >= 1"
        )

    return True


@parameter_decorator
def validate_tarch_parameters(omega: float, alpha: float, gamma: float, beta: float) -> bool:
    """
    Validate parameters for a TARCH model.

    This function validates the parameters of a TARCH model, checking that
    they satisfy constraints such as positivity and stationarity.

    Args:
        omega: Constant term in variance equation (must be positive)
        alpha: ARCH parameter (must be non-negative)
        gamma: Asymmetry parameter (must be non-negative)
        beta: GARCH parameter (must be non-negative)

    Returns:
        bool: True if parameters are valid

    Raises:
        ValueError: If parameter constraints are violated

    Examples:
        >>> from mfe.models.univariate.utils import validate_tarch_parameters
        >>> validate_tarch_parameters(0.05, 0.05, 0.1, 0.8)  # Returns True if parameters are valid
    """
    # Validate individual parameters
    validate_positive(omega, "omega")
    validate_non_negative(alpha, "alpha")
    validate_non_negative(gamma, "gamma")
    validate_non_negative(beta, "beta")

    # Validate stationarity constraint
    # For TARCH, the constraint is alpha + beta + 0.5*gamma < 1
    persistence = alpha + beta + 0.5 * gamma
    if persistence >= 1:
        raise ValueError(
            f"TARCH stationarity constraint violated: "
            f"alpha + beta + 0.5*gamma = {persistence} >= 1"
        )

    return True


@parameter_decorator
def validate_aparch_parameters(omega: float, alpha: float, gamma: float, beta: float, delta: float) -> bool:
    """
    Validate parameters for an APARCH model.

    This function validates the parameters of an APARCH model, checking that
    they satisfy constraints such as positivity and stationarity.

    Args:
        omega: Constant term in power variance equation (must be positive)
        alpha: ARCH parameter (must be non-negative)
        gamma: Asymmetry parameter (must be between -1 and 1)
        beta: GARCH parameter (must be non-negative)
        delta: Power parameter (must be positive)

    Returns:
        bool: True if parameters are valid

    Raises:
        ValueError: If parameter constraints are violated

    Examples:
        >>> from mfe.models.univariate.utils import validate_aparch_parameters
        >>> validate_aparch_parameters(0.05, 0.05, 0.1, 0.8, 1.5)  # Returns True if parameters are valid
    """
    # Validate individual parameters
    validate_positive(omega, "omega")
    validate_non_negative(alpha, "alpha")
    validate_range(gamma, "gamma", -1, 1)
    validate_non_negative(beta, "beta")
    validate_positive(delta, "delta")

    # Validate stationarity constraint
    if alpha + beta >= 1:
        raise ValueError(
            f"APARCH stationarity constraint violated: alpha + beta = {alpha + beta} >= 1"
        )

    return True


def compare_volatility_models(models: Dict[str, Dict[str, Any]],
                              data: np.ndarray) -> Dict[str, Any]:
    """
    Compare multiple volatility models fitted to the same data.

    This function compares multiple volatility models based on various criteria
    such as log-likelihood, AIC, BIC, and out-of-sample performance.

    Args:
        models: Dictionary of model results, where keys are model names and values
               are dictionaries containing model parameters, conditional variances,
               and log-likelihood
        data: Original data (residuals) used for model fitting

    Returns:
        Dict[str, Any]: Dictionary of comparison results

    Raises:
        ValueError: If input dimensions are inconsistent

    Examples:
        >>> import numpy as np
        >>> from mfe.models.univariate.utils import compare_volatility_models
        >>> data = np.random.normal(0, 1, 1000)
        >>> # Assume we have fitted multiple models
        >>> models = {
        ...     "GARCH(1,1)": {
        ...         "parameters": {"omega": 0.05, "alpha": 0.1, "beta": 0.85},
        ...         "conditional_variances": np.ones_like(data),  # Placeholder
        ...         "log_likelihood": -1500  # Placeholder
        ...     },
        ...     "EGARCH(1,1)": {
        ...         "parameters": {"omega": -0.1, "alpha": 0.1, "gamma": 0.1, "beta": 0.9},
        ...         "conditional_variances": np.ones_like(data),  # Placeholder
        ...         "log_likelihood": -1490  # Placeholder
        ...     }
        ... }
        >>> comparison = compare_volatility_models(models, data)
        >>> print(f"Best model by AIC: {comparison['best_model_aic']}")
    """
    # Validate input
    if not models:
        raise ValueError("models dictionary must not be empty")

    # Initialize results dictionary
    results = {
        "model_names": list(models.keys()),
        "log_likelihood": {},
        "aic": {},
        "bic": {},
        "hqic": {},
        "persistence": {},
        "n_params": {}
    }

    # Compute diagnostics for each model
    for model_name, model_dict in models.items():
        # Extract model information
        parameters = model_dict["parameters"]
        conditional_variances = model_dict["conditional_variances"]
        log_likelihood = model_dict["log_likelihood"]

        # Compute diagnostics
        diagnostics = compute_model_diagnostics(
            data, conditional_variances, parameters, log_likelihood
        )

        # Store results
        results["log_likelihood"][model_name] = log_likelihood
        results["aic"][model_name] = diagnostics["aic"]
        results["bic"][model_name] = diagnostics["bic"]
        results["hqic"][model_name] = diagnostics["hqic"]
        results["persistence"][model_name] = diagnostics["persistence"]
        results["n_params"][model_name] = diagnostics["n_params"]

    # Find best model by different criteria
    results["best_model_loglik"] = max(results["log_likelihood"], key=results["log_likelihood"].get)
    results["best_model_aic"] = min(results["aic"], key=results["aic"].get)
    results["best_model_bic"] = min(results["bic"], key=results["bic"].get)
    results["best_model_hqic"] = min(results["hqic"], key=results["hqic"].get)

    # Create ranking by AIC
    aic_items = list(results["aic"].items())
    aic_items.sort(key=lambda x: x[1])
    results["aic_ranking"] = [item[0] for item in aic_items]

    # Create ranking by BIC
    bic_items = list(results["bic"].items())
    bic_items.sort(key=lambda x: x[1])
    results["bic_ranking"] = [item[0] for item in bic_items]

    return results


@jit(nopython=True, cache=True)
def _simulate_garch_path_numba(n_periods: int,
                               omega: float,
                               alpha: float,
                               beta: float,
                               initial_variance: float,
                               random_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a GARCH(1,1) path using Numba acceleration.

    Args:
        n_periods: Number of periods to simulate
        omega: Constant term in variance equation
        alpha: ARCH parameter
        beta: GARCH parameter
        initial_variance: Initial variance value
        random_values: Array of random values (e.g., from normal distribution)

    Returns:
        Tuple[np.ndarray, np.ndarray]: Simulated returns and conditional variances
    """
    # Initialize arrays
    returns = np.zeros(n_periods)
    variances = np.zeros(n_periods)

    # Set initial variance
    variances[0] = initial_variance

    # Generate first return
    returns[0] = np.sqrt(variances[0]) * random_values[0]

    # Generate remaining returns and variances
    for t in range(1, n_periods):
        # Compute variance
        variances[t] = omega + alpha * returns[t-1]**2 + beta * variances[t-1]

        # Generate return
        returns[t] = np.sqrt(variances[t]) * random_values[t]

    return returns, variances


def simulate_volatility_path(parameters: Union[Dict[str, Any], UnivariateVolatilityParameters],
                             n_periods: int,
                             model_type: str = "garch",
                             distribution: str = "normal",
                             random_state: Optional[Union[int, np.random.Generator]] = None,
                             initial_variance: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a path from a volatility model.

    This function simulates returns and conditional variances from a volatility
    model with the specified parameters.

    Args:
        parameters: Model parameters (dictionary or parameter object)
        n_periods: Number of periods to simulate
        model_type: Type of volatility model ("garch", "egarch", "tarch", "aparch")
        distribution: Distribution of innovations ("normal", "t", "skewed-t", "ged")
        random_state: Random number generator or seed
        initial_variance: Initial variance value (if None, computed from parameters)

    Returns:
        Tuple[np.ndarray, np.ndarray]: Simulated returns and conditional variances

    Raises:
        ValueError: If parameters or model type are invalid

    Examples:
        >>> import numpy as np
        >>> from mfe.models.univariate.utils import simulate_volatility_path
        >>> params = {"omega": 0.05, "alpha": 0.1, "beta": 0.85}
        >>> returns, variances = simulate_volatility_path(params, n_periods=1000)
        >>> import matplotlib.pyplot as plt
        >>> plt.figure(figsize=(12, 6))
        >>> plt.subplot(2, 1, 1)
        >>> plt.plot(returns)
        >>> plt.title('Simulated Returns')
        >>> plt.subplot(2, 1, 2)
        >>> plt.plot(np.sqrt(variances))
        >>> plt.title('Simulated Volatility')
        >>> plt.tight_layout()
        >>> plt.show()
    """
    # Set up random number generator
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    # Extract parameters
    if isinstance(parameters, dict):
        # If parameters is a dictionary, extract values
        if model_type == "garch":
            omega = parameters["omega"]
            alpha = parameters["alpha"]
            beta = parameters["beta"]
        elif model_type == "egarch":
            omega = parameters["omega"]
            alpha = parameters["alpha"]
            gamma = parameters["gamma"]
            beta = parameters["beta"]
        elif model_type == "tarch":
            omega = parameters["omega"]
            alpha = parameters["alpha"]
            gamma = parameters["gamma"]
            beta = parameters["beta"]
        elif model_type == "aparch":
            omega = parameters["omega"]
            alpha = parameters["alpha"]
            gamma = parameters["gamma"]
            beta = parameters["beta"]
            delta = parameters["delta"]
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    else:
        # If parameters is a parameter object, extract attributes
        if model_type == "garch":
            omega = parameters.omega
            alpha = parameters.alpha
            beta = parameters.beta
        elif model_type == "egarch":
            omega = parameters.omega
            alpha = parameters.alpha
            gamma = parameters.gamma
            beta = parameters.beta
        elif model_type == "tarch":
            omega = parameters.omega
            alpha = parameters.alpha
            gamma = parameters.gamma
            beta = parameters.beta
        elif model_type == "aparch":
            omega = parameters.omega
            alpha = parameters.alpha
            gamma = parameters.gamma
            beta = parameters.beta
            delta = parameters.delta
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    # Handle array parameters for GARCH
    if model_type == "garch":
        if isinstance(alpha, (list, np.ndarray)) and len(alpha) > 1:
            warn_model(
                "simulate_volatility_path currently only supports GARCH(1,1) for simulation. "
                "Using first alpha and beta values.",
                model_type="GARCH",
                issue="higher_order_simulation"
            )
            alpha = alpha[0] if isinstance(alpha, np.ndarray) else alpha[0]
            beta = beta[0] if isinstance(beta, np.ndarray) else beta[0]

    # Compute initial variance if not provided
    if initial_variance is None:
        # Try to compute unconditional variance
        try:
            if model_type == "garch":
                persistence = alpha + beta
                initial_variance = omega / (1 - persistence) if persistence < 1 else 1.0
            elif model_type == "egarch":
                initial_variance = np.exp(omega / (1 - beta)) if abs(beta) < 1 else 1.0
            elif model_type == "tarch":
                persistence = alpha + beta + 0.5 * gamma
                initial_variance = omega / (1 - persistence) if persistence < 1 else 1.0
            elif model_type == "aparch":
                persistence = alpha + beta
                initial_variance = (omega / (1 - persistence)) ** (2 / delta) if persistence < 1 else 1.0
            else:
                initial_variance = 1.0
        except:
            # Fallback to a reasonable value
            initial_variance = 1.0

    # Generate random innovations based on distribution
    if distribution == "normal":
        random_values = rng.standard_normal(n_periods)
    elif distribution == "t":
        df = 5.0  # Default degrees of freedom
        random_values = rng.standard_t(df, n_periods)
        # Scale to have unit variance
        random_values = random_values * np.sqrt((df - 2) / df)
    elif distribution == "skewed-t":
        # Simple approximation of skewed t-distribution
        df = 5.0  # Default degrees of freedom
        skew = 0.5  # Default skewness parameter

        # Generate t-distributed random values
        t_values = rng.standard_t(df, n_periods)

        # Apply skewness transformation
        random_values = np.zeros_like(t_values)
        mask = t_values < 0
        random_values[mask] = skew * t_values[mask]
        random_values[~mask] = (1 / skew) * t_values[~mask]

        # Scale to have unit variance
        random_values = random_values / np.std(random_values)
    elif distribution == "ged":
        # Approximate GED using normal distribution
        warn_model(
            "GED distribution not fully implemented for simulation. Using normal distribution.",
            model_type=model_type,
            issue="distribution_approximation"
        )
        random_values = rng.standard_normal(n_periods)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    # Simulate path based on model type
    if model_type == "garch":
        # Use Numba-accelerated implementation for GARCH(1,1)
        returns, variances = _simulate_garch_path_numba(
            n_periods, omega, alpha, beta, initial_variance, random_values
        )
    elif model_type == "egarch":
        # Initialize arrays
        returns = np.zeros(n_periods)
        log_variances = np.zeros(n_periods)

        # Set initial log-variance
        log_variances[0] = np.log(initial_variance)

        # Generate first return
        returns[0] = np.sqrt(np.exp(log_variances[0])) * random_values[0]

        # Generate remaining returns and log-variances
        for t in range(1, n_periods):
            # Standardized return
            z_t_1 = returns[t-1] / np.sqrt(np.exp(log_variances[t-1]))

            # Compute log-variance
            log_variances[t] = omega + beta * log_variances[t-1] + \
                alpha * (abs(z_t_1) - np.sqrt(2/np.pi)) + gamma * z_t_1

            # Generate return
            returns[t] = np.sqrt(np.exp(log_variances[t])) * random_values[t]

        # Convert log-variances to variances
        variances = np.exp(log_variances)
    elif model_type == "tarch":
        # Initialize arrays
        returns = np.zeros(n_periods)
        variances = np.zeros(n_periods)

        # Set initial variance
        variances[0] = initial_variance

        # Generate first return
        returns[0] = np.sqrt(variances[0]) * random_values[0]

        # Generate remaining returns and variances
        for t in range(1, n_periods):
            # Indicator for negative returns
            I_t_1 = 1 if returns[t-1] < 0 else 0

            # Compute variance
            variances[t] = omega + alpha * returns[t-1]**2 + gamma * returns[t-1]**2 * I_t_1 + beta * variances[t-1]

            # Generate return
            returns[t] = np.sqrt(variances[t]) * random_values[t]

    elif model_type == "aparch":
        # Initialize arrays
        returns = np.zeros(n_periods)
        power_variances = np.zeros(n_periods)

        # Set initial power variance
        power_variances[0] = initial_variance ** (delta / 2)

        # Generate first return
        returns[0] = np.sqrt(power_variances[0] ** (2 / delta)) * random_values[0]

        # Generate remaining returns and power variances
        for t in range(1, n_periods):
            # Compute power variance
            abs_return = abs(returns[t-1])
            leverage = 1 + gamma * np.sign(returns[t-1])
            power_variances[t] = omega + alpha * (abs_return * leverage) ** delta + beta * power_variances[t-1]

            # Convert to variance
            variance = power_variances[t] ** (2 / delta)

            # Generate return
            returns[t] = np.sqrt(variance) * random_values[t]

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return returns, variances


def simulate_univariate_volatility(
    T: int,
    model_type: str = "garch",
    params: Optional[Dict[str, float]] = None,
    dist_type: str = "normal",
    dist_params: Optional[Dict[str, float]] = None,
    initial_value: Optional[float] = None,
    burn_in: int = 500,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate data from a univariate volatility model.
    
    This function simulates returns and conditional variances from various
    univariate volatility models, including GARCH, EGARCH, TARCH, and APARCH.
    
    Args:
        T: Number of observations to simulate
        model_type: Type of volatility model ("garch", "egarch", "tarch", "aparch", etc.)
        params: Dictionary of model parameters (e.g., {"omega": 0.1, "alpha": 0.1, "beta": 0.8})
        dist_type: Distribution type for innovations ("normal", "t", "skewt", "ged")
        dist_params: Dictionary of distribution parameters (e.g., {"df": 5} for t-distribution)
        initial_value: Initial value for the variance process
        burn_in: Number of burn-in observations to discard
        seed: Random seed for reproducibility
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Simulated returns, conditional variances, and standardized innovations
    
    Raises:
        ValueError: If the model type is not recognized or if parameters are invalid
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.univariate.utils import simulate_univariate_volatility
        >>> returns, variances, innovations = simulate_univariate_volatility(
        ...     T=1000,
        ...     model_type="garch",
        ...     params={"omega": 0.05, "alpha": 0.1, "beta": 0.8},
        ...     dist_type="normal"
        ... )
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Set default parameters if not provided
    if params is None:
        if model_type.lower() == "garch":
            params = {"omega": 0.05, "alpha": 0.1, "beta": 0.8}
        elif model_type.lower() == "egarch":
            params = {"omega": -0.1, "alpha": 0.1, "gamma": 0.05, "beta": 0.9}
        elif model_type.lower() == "tarch":
            params = {"omega": 0.05, "alpha": 0.05, "gamma": 0.1, "beta": 0.8}
        elif model_type.lower() == "aparch":
            params = {"omega": 0.05, "alpha": 0.05, "gamma": 0.1, "beta": 0.8, "delta": 1.5}
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    # Set default distribution parameters if not provided
    if dist_params is None:
        if dist_type.lower() == "normal":
            dist_params = {}
        elif dist_type.lower() == "t":
            dist_params = {"df": 5}
        elif dist_type.lower() == "skewt":
            dist_params = {"df": 5, "skew": 0.9}
        elif dist_type.lower() == "ged":
            dist_params = {"nu": 1.5}
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")
    
    # Generate innovations based on the specified distribution
    total_length = T + burn_in
    
    if dist_type.lower() == "normal":
        innovations = np.random.normal(0, 1, total_length)
    elif dist_type.lower() == "t":
        df = dist_params.get("df", 5)
        innovations = np.random.standard_t(df, total_length) / np.sqrt(df / (df - 2))
    elif dist_type.lower() == "skewt":
        # Simple approximation of skewed t-distribution
        df = 5.0  # Default degrees of freedom
        skew = 0.5  # Default skewness parameter

        # Generate t-distributed random values
        t_values = np.random.standard_t(df, total_length)

        # Apply skewness transformation
        random_values = np.zeros_like(t_values)
        mask = t_values < 0
        random_values[mask] = skew * t_values[mask]
        random_values[~mask] = (1 / skew) * t_values[~mask]

        # Scale to have unit variance
        random_values = random_values / np.std(random_values)
    elif dist_type.lower() == "ged":
        nu = dist_params.get("nu", 1.5)
        # Generate GED innovations (simplified approach)
        u = np.random.normal(0, 1, total_length)
        innovations = np.sign(u) * np.abs(u)**(1/nu) * np.sqrt(gamma_function(1/nu) / gamma_function(3/nu))
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")
    
    # Initialize arrays for returns and variances
    returns = np.zeros(total_length)
    variances = np.zeros(total_length)
    
    # Set initial variance
    if initial_value is None:
        initial_value = params.get("omega", 0.05) / (1 - params.get("alpha", 0.1) - params.get("beta", 0.8))
    
    variances[0] = initial_value
    
    # Simulate the process based on the model type
    model_type = model_type.lower()
    
    if model_type == "garch":
        omega = params.get("omega", 0.05)
        alpha = params.get("alpha", 0.1)
        beta = params.get("beta", 0.8)
        
        for t in range(1, total_length):
            variances[t] = omega + alpha * returns[t-1]**2 + beta * variances[t-1]
            returns[t] = np.sqrt(variances[t]) * innovations[t]
    
    elif model_type == "egarch":
        omega = params.get("omega", -0.1)
        alpha = params.get("alpha", 0.1)
        gamma = params.get("gamma", 0.05)
        beta = params.get("beta", 0.9)
        
        # For EGARCH, we work with log-variances
        log_variances = np.zeros(total_length)
        log_variances[0] = np.log(initial_value)
        
        for t in range(1, total_length):
            # Compute standardized innovation
            if t == 1:
                z = innovations[0]
            else:
                z = returns[t-1] / np.sqrt(variances[t-1])
            
            # Update log-variance
            log_variances[t] = omega + beta * log_variances[t-1] + \
                alpha * (abs(z) - np.sqrt(2/np.pi)) + gamma * z
            
            # Convert to variance
            variances[t] = np.exp(log_variances[t])
            
            # Generate return
            returns[t] = np.sqrt(variances[t]) * innovations[t]
    
    elif model_type == "tarch":
        omega = params.get("omega", 0.05)
        alpha = params.get("alpha", 0.05)
        gamma = params.get("gamma", 0.1)
        beta = params.get("beta", 0.8)
        
        for t in range(1, total_length):
            # Asymmetric term: I(r_{t-1} < 0)
            asym = 1 if returns[t-1] < 0 else 0
            
            # Update variance
            variances[t] = omega + alpha * returns[t-1]**2 + gamma * returns[t-1]**2 * asym + beta * variances[t-1]
            
            # Generate return
            returns[t] = np.sqrt(variances[t]) * innovations[t]
    
    elif model_type == "aparch":
        omega = params.get("omega", 0.05)
        alpha = params.get("alpha", 0.05)
        gamma = params.get("gamma", 0.1)
        beta = params.get("beta", 0.8)
        delta = params.get("delta", 1.5)
        
        # For APARCH, we work with power-transformed variances
        power_variances = np.zeros(total_length)
        power_variances[0] = initial_value**(delta/2)
        
        for t in range(1, total_length):
            # Update power variance
            abs_return = np.abs(returns[t-1])
            sign = -1 if returns[t-1] < 0 else 1
            power_variances[t] = omega + alpha * (abs_return - gamma * sign * abs_return)**delta + beta * power_variances[t-1]
            
            # Convert to variance
            variance = power_variances[t]**(2/delta)
            
            # Generate return
            returns[t] = np.sqrt(variance) * innovations[t]
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Discard burn-in period
    returns = returns[burn_in:]
    variances = variances[burn_in:]
    innovations = innovations[burn_in:]
    
    return returns, variances, innovations


def forecast_volatility(
    model_type: str,
    params: Dict[str, float],
    last_returns: np.ndarray,
    last_variances: np.ndarray,
    steps: int = 10,
    return_volatility: bool = False,
    power: float = 2.0
) -> np.ndarray:
    """
    Generate multi-step ahead forecasts for volatility models.
    
    This function computes forecasts for various univariate volatility models,
    including GARCH, EGARCH, TARCH, and APARCH.
    
    Args:
        model_type: Type of volatility model ("garch", "egarch", "tarch", "aparch", etc.)
        params: Dictionary of model parameters (e.g., {"omega": 0.1, "alpha": 0.1, "beta": 0.8})
        last_returns: Recent returns for initializing the forecast
        last_variances: Recent conditional variances for initializing the forecast
        steps: Number of steps to forecast
        return_volatility: Whether to return volatility (True) or variance (False)
        power: Power parameter for models like APARCH
    
    Returns:
        np.ndarray: Forecasted conditional variances or volatilities
    
    Raises:
        ValueError: If the model type is not recognized or if parameters are invalid
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.univariate.utils import forecast_volatility
        >>> last_returns = np.array([0.1, -0.2, 0.3])
        >>> last_variances = np.array([0.01, 0.02, 0.015])
        >>> forecasts = forecast_volatility(
        ...     model_type="garch",
        ...     params={"omega": 0.05, "alpha": 0.1, "beta": 0.8},
        ...     last_returns=last_returns,
        ...     last_variances=last_variances,
        ...     steps=5
        ... )
    """
    # Validate inputs
    if not isinstance(last_returns, np.ndarray):
        last_returns = np.asarray(last_returns)
    
    if not isinstance(last_variances, np.ndarray):
        last_variances = np.asarray(last_variances)
    
    if last_returns.ndim != 1:
        raise ValueError(f"last_returns must be a 1-dimensional array, got shape {last_returns.shape}")
    
    if last_variances.ndim != 1:
        raise ValueError(f"last_variances must be a 1-dimensional array, got shape {last_variances.shape}")
    
    if len(last_returns) == 0 or len(last_variances) == 0:
        raise ValueError("last_returns and last_variances must not be empty")
    
    if len(last_returns) != len(last_variances):
        raise ValueError(f"last_returns and last_variances must have the same length, got {len(last_returns)} and {len(last_variances)}")
    
    # Generate forecasts based on the model type
    model_type = model_type.lower()
    
    if model_type == "garch":
        # Extract parameters
        omega = params.get("omega", 0.05)
        alpha = params.get("alpha", 0.1)
        beta = params.get("beta", 0.8)
        
        # Compute persistence
        persistence = alpha + beta
        
        # Compute unconditional variance
        unconditional_variance = omega / (1 - persistence) if persistence < 1 else last_variances[-1]
        
        # Initialize forecasts
        forecasts = np.zeros(steps)
        
        # First step forecast
        forecasts[0] = omega + alpha * last_returns[-1]**2 + beta * last_variances[-1]
        
        # Multi-step forecasts
        for h in range(1, steps):
            forecasts[h] = omega + persistence * forecasts[h-1]
            
            # For long horizons, approach the unconditional variance
            if persistence < 1 and h > 100:
                forecasts[h] = unconditional_variance - (unconditional_variance - forecasts[h]) * 0.1
    
    elif model_type == "egarch":
        # Extract parameters
        omega = params.get("omega", -0.1)
        alpha = params.get("alpha", 0.1)
        gamma = params.get("gamma", 0.05)
        beta = params.get("beta", 0.9)
        
        # Compute standardized residual
        z = last_returns[-1] / np.sqrt(last_variances[-1])
        
        # Compute log-variance for the first step
        log_variance = omega + alpha * (np.abs(z) - np.sqrt(2/np.pi)) + gamma * z + beta * np.log(last_variances[-1])
        
        # Initialize forecasts
        forecasts = np.zeros(steps)
        forecasts[0] = np.exp(log_variance)
        
        # Multi-step forecasts
        for h in range(1, steps):
            # For EGARCH, the expected value of the news impact is 0
            log_variance = omega + beta * np.log(forecasts[h-1])
            forecasts[h] = np.exp(log_variance)
    
    elif model_type == "tarch":
        # Extract parameters
        omega = params.get("omega", 0.05)
        alpha = params.get("alpha", 0.05)
        gamma = params.get("gamma", 0.1)
        beta = params.get("beta", 0.8)
        
        # Compute asymmetric term
        asym = 1 if last_returns[-1] < 0 else 0
        
        # Compute persistence
        persistence = alpha + 0.5 * gamma + beta
        
        # Compute unconditional variance
        unconditional_variance = omega / (1 - persistence) if persistence < 1 else last_variances[-1]
        
        # Initialize forecasts
        forecasts = np.zeros(steps)
        
        # First step forecast
        forecasts[0] = omega + alpha * last_returns[-1]**2 + gamma * last_returns[-1]**2 * asym + beta * last_variances[-1]
        
        # Multi-step forecasts
        for h in range(1, steps):
            forecasts[h] = omega + persistence * forecasts[h-1]
            
            # For long horizons, approach the unconditional variance
            if persistence < 1 and h > 100:
                forecasts[h] = unconditional_variance - (unconditional_variance - forecasts[h]) * 0.1
    
    elif model_type == "aparch":
        # Extract parameters
        omega = params.get("omega", 0.05)
        alpha = params.get("alpha", 0.05)
        gamma = params.get("gamma", 0.1)
        beta = params.get("beta", 0.8)
        delta = params.get("delta", 1.5)
        
        # Compute power-transformed variance for the first step
        abs_return = np.abs(last_returns[-1])
        sign = -1 if last_returns[-1] < 0 else 1
        power_variance = omega + alpha * (abs_return - gamma * sign * abs_return)**delta + beta * last_variances[-1]**(delta/2)
        
        # Initialize forecasts
        forecasts = np.zeros(steps)
        forecasts[0] = power_variance**(2/delta)
        
        # Compute persistence
        persistence = alpha * (1 + gamma)**delta + beta
        
        # Compute unconditional power variance
        unconditional_power_variance = omega / (1 - persistence) if persistence < 1 else power_variance
        
        # Multi-step forecasts
        for h in range(1, steps):
            # For APARCH, the expected value of the news impact is more complex
            # We use a simplified approach here
            power_variance = omega + persistence * power_variance
            forecasts[h] = power_variance**(2/delta)
            
            # For long horizons, approach the unconditional variance
            if persistence < 1 and h > 100:
                unconditional_variance = unconditional_power_variance**(2/delta)
                forecasts[h] = unconditional_variance - (unconditional_variance - forecasts[h]) * 0.1
    
    elif model_type == "igarch":
        # Extract parameters
        omega = params.get("omega", 0.05)
        beta = params.get("beta", 0.8)
        alpha = 1.0 - beta  # In IGARCH, alpha + beta = 1
        
        # Initialize forecasts
        forecasts = np.zeros(steps)
        
        # First step forecast
        forecasts[0] = omega + alpha * last_returns[-1]**2 + beta * last_variances[-1]
        
        # Multi-step forecasts
        # For IGARCH, E[σ²_{t+h}] = σ²_t + h*ω
        for h in range(1, steps):
            forecasts[h] = forecasts[0] + h * omega
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Convert to volatility if requested
    if return_volatility:
        forecasts = np.power(forecasts, 1.0 / power)
    
    return forecasts


# Register Numba-accelerated functions
def _register_numba_functions() -> None:
    """
    Register Numba JIT-compiled functions for univariate volatility utilities.

    This function is called during module initialization to register
    performance-critical functions for JIT compilation.
    """
    # The functions are already decorated with @jit, so no additional
    # registration is needed here. This function is kept for consistency
    # with the module structure and potential future enhancements.
    logger.debug("Univariate volatility utilities Numba JIT functions registered")


# Initialize the module
_register_numba_functions()
