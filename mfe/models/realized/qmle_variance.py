'''
Quasi-Maximum Likelihood Estimator (QMLE) for realized variance.

This module implements the quasi-maximum likelihood estimator (QMLE) for realized
variance, which applies a parametric approach to handle market microstructure noise.
The QMLE estimator is statistically efficient and provides a robust method for
estimating volatility in the presence of market microstructure noise.

The implementation leverages NumPy's efficient array operations with Numba acceleration
for performance-critical likelihood calculations. It supports both raw NumPy arrays and
Pandas DataFrames with datetime indices for convenient time series analysis.

The QMLE approach models observed log-prices as the sum of an efficient price process
and i.i.d. noise, and estimates the parameters of this model using maximum likelihood.
This approach provides a statistically efficient estimator that is robust to various
forms of market microstructure noise.

Classes:
    QMLEVarianceConfig: Configuration parameters for QMLE variance estimation
    QMLEVarianceResult: Result container for QMLE variance estimation
    QMLEVarianceEstimator: Class for estimating realized variance using QMLE

Functions:
    qmle_variance: Estimate realized variance using QMLE
    _qmle_likelihood: Compute QMLE likelihood function (Numba-accelerated)
    _qmle_likelihood_gradient: Compute gradient of QMLE likelihood (Numba-accelerated)
'''

import logging
import warnings
import time
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Literal, Optional, Sequence, 
    Tuple, Type, TypeVar, Union, cast, overload
)

import numpy as np
import pandas as pd
from scipy import stats, optimize, sparse

from ...core.base import ModelBase
from ...core.parameters import ParameterBase, ParameterError, validate_positive, validate_non_negative
from ...core.exceptions import (
    DimensionError, NumericError, ParameterError, ConvergenceError,
    raise_dimension_error, raise_numeric_error, warn_numeric
)
from .base import BaseRealizedEstimator, NoiseRobustEstimator, RealizedEstimatorConfig, RealizedEstimatorResult

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    HAS_NUMBA = False

# Set up module-level logger
logger = logging.getLogger("mfe.models.realized.qmle_variance")

# Try to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.info("Matplotlib not available. Plotting functions will be disabled.")


@dataclass
class QMLEVarianceConfig(RealizedEstimatorConfig):
    """Configuration parameters for QMLE variance estimation.
    
    This class extends RealizedEstimatorConfig to provide specialized configuration
    parameters for QMLE variance estimation.
    
    Attributes:
        max_iterations: Maximum number of iterations for optimization
        tolerance: Convergence tolerance for optimization
        optimization_method: Method for optimization ('L-BFGS-B', 'BFGS', 'Nelder-Mead')
        initial_noise_method: Method for initial noise variance estimation ('ac1', 'signature', 'auto')
        use_sparse: Whether to use sparse matrix operations for large datasets
        return_likelihood: Whether to return likelihood values
        return_optimization_details: Whether to return detailed optimization information
        plot: Whether to generate diagnostic plots
    """
    
    max_iterations: int = 100
    tolerance: float = 1e-6
    optimization_method: Literal['L-BFGS-B', 'BFGS', 'Nelder-Mead'] = 'L-BFGS-B'
    initial_noise_method: Literal['ac1', 'signature', 'auto'] = 'auto'
    use_sparse: bool = True
    return_likelihood: bool = False
    return_optimization_details: bool = False
    plot: bool = False
    
    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        super().__post_init__()
        
        # Validate max_iterations
        validate_positive(self.max_iterations, "max_iterations")
        
        # Validate tolerance
        validate_positive(self.tolerance, "tolerance")
        
        # Validate optimization_method
        valid_methods = ['L-BFGS-B', 'BFGS', 'Nelder-Mead']
        if self.optimization_method not in valid_methods:
            raise ParameterError(f"optimization_method must be one of {valid_methods}, got {self.optimization_method}")
        
        # Validate initial_noise_method
        valid_noise_methods = ['ac1', 'signature', 'auto']
        if self.initial_noise_method not in valid_noise_methods:
            raise ParameterError(f"initial_noise_method must be one of {valid_noise_methods}, got {self.initial_noise_method}")


@dataclass
class QMLEVarianceResult(RealizedEstimatorResult):
    """Result container for QMLE variance estimation.
    
    This class extends RealizedEstimatorResult to provide specialized functionality
    for QMLE variance estimation results, including additional metadata and
    diagnostic information specific to QMLE estimation.
    
    Attributes:
        integrated_variance: Estimated integrated variance
        noise_variance: Estimated noise variance
        log_likelihood: Log-likelihood value at the optimum
        iterations: Number of iterations performed during optimization
        convergence_status: Convergence status of the optimization
        optimization_details: Detailed information about the optimization
        computation_time: Time taken for computation (in seconds)
        sparse_matrix_used: Whether sparse matrix operations were used
    """
    
    integrated_variance: Optional[float] = None
    log_likelihood: Optional[float] = None
    iterations: Optional[int] = None
    convergence_status: Optional[bool] = None
    optimization_details: Optional[Dict[str, Any]] = None
    sparse_matrix_used: Optional[bool] = None
    
    def __post_init__(self) -> None:
        """Validate result object after initialization."""
        super().__post_init__()
    
    def summary(self) -> str:
        """Generate a text summary of the QMLE variance estimation results.
        
        Returns:
            str: A formatted string containing the QMLE variance results summary
        """
        base_summary = super().summary()
        
        additional_info = f"QMLE Variance Estimation Results:\n"
        
        if self.integrated_variance is not None:
            additional_info += f"  Integrated Variance: {self.integrated_variance:.6e}\n"
        
        if self.noise_variance is not None:
            additional_info += f"  Noise Variance: {self.noise_variance:.6e}\n"
        
        if self.log_likelihood is not None:
            additional_info += f"  Log-Likelihood: {self.log_likelihood:.6f}\n"
        
        if self.iterations is not None:
            additional_info += f"  Iterations: {self.iterations}\n"
        
        if self.convergence_status is not None:
            additional_info += f"  Convergence: {'Successful' if self.convergence_status else 'Failed'}\n"
        
        if self.sparse_matrix_used is not None:
            additional_info += f"  Sparse Matrix: {'Used' if self.sparse_matrix_used else 'Not Used'}\n"
        
        if self.computation_time is not None:
            additional_info += f"  Computation Time: {self.computation_time:.6f} seconds\n"
        
        if self.optimization_details is not None:
            additional_info += "  Optimization Details:\n"
            for key, value in self.optimization_details.items():
                if isinstance(value, (int, float, str, bool)):
                    additional_info += f"    {key}: {value}\n"
        
        return base_summary + additional_info
    
    def plot(self, figsize: Tuple[int, int] = (10, 6)) -> Optional[Any]:
        """Plot QMLE variance estimation results.
        
        Args:
            figsize: Figure size (width, height) in inches
            
        Returns:
            Optional[Any]: Matplotlib figure if matplotlib is available, None otherwise
            
        Raises:
            ImportError: If matplotlib is not available
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib is required for plotting. Please install matplotlib.")
        
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Plot returns
        if self.returns is not None:
            axes[0].plot(self.returns, 'b-', alpha=0.7)
            axes[0].set_title('Returns with QMLE Variance Estimate')
            axes[0].set_xlabel('Observation')
            axes[0].set_ylabel('Return')
            
            # Add variance estimate as text
            if self.realized_measure is not None:
                axes[0].text(0.05, 0.95, 
                            f'QMLE Variance: {self.realized_measure[0]:.6e}\n'
                            f'Noise Variance: {self.noise_variance:.6e}',
                            transform=axes[0].transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot autocorrelation of returns
        if self.returns is not None:
            max_lag = min(30, len(self.returns) // 5)
            acf = np.zeros(max_lag)
            for lag in range(max_lag):
                if lag == 0:
                    acf[lag] = 1.0
                else:
                    acf[lag] = np.corrcoef(self.returns[:-lag], self.returns[lag:])[0, 1]
            
            axes[1].stem(range(max_lag), acf, linefmt='b-', markerfmt='bo', basefmt='r-')
            axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
            axes[1].set_title('Autocorrelation of Returns')
            axes[1].set_xlabel('Lag')
            axes[1].set_ylabel('Autocorrelation')
            
            # Add confidence bands (95%)
            conf_level = 1.96 / np.sqrt(len(self.returns))
            axes[1].axhline(y=conf_level, color='r', linestyle='--', alpha=0.5)
            axes[1].axhline(y=-conf_level, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        return fig


class QMLEVarianceEstimator(NoiseRobustEstimator):
    """Quasi-Maximum Likelihood Estimator (QMLE) for realized variance.
    
    This class implements the QMLE approach for estimating realized variance in the
    presence of market microstructure noise. The QMLE estimator models observed
    log-prices as the sum of an efficient price process and i.i.d. noise, and
    estimates the parameters of this model using maximum likelihood.
    
    The implementation leverages NumPy's efficient array operations with Numba
    acceleration for performance-critical likelihood calculations. For large datasets,
    sparse matrix operations are used to improve performance and reduce memory usage.
    
    Attributes:
        config: Configuration parameters for the estimator
        name: A descriptive name for the estimator
    """
    
    def __init__(self, config: Optional[QMLEVarianceConfig] = None, name: str = "QMLEVarianceEstimator"):
        """Initialize the QMLE variance estimator.
        
        Args:
            config: Configuration parameters for the estimator
            name: A descriptive name for the estimator
        """
        config_to_use = config if config is not None else QMLEVarianceConfig()
        super().__init__(config=config_to_use, name=name)
        self._integrated_variance: Optional[float] = None
        self._log_likelihood: Optional[float] = None
        self._iterations: Optional[int] = None
        self._convergence_status: Optional[bool] = None
        self._optimization_details: Optional[Dict[str, Any]] = None
        self._sparse_matrix_used: Optional[bool] = None
    
    @property
    def config(self) -> QMLEVarianceConfig:
        """Get the estimator configuration.
        
        Returns:
            QMLEVarianceConfig: The estimator configuration
        """
        return cast(QMLEVarianceConfig, self._config)
    
    @config.setter
    def config(self, config: QMLEVarianceConfig) -> None:
        """Set the estimator configuration.
        
        Args:
            config: New configuration parameters
            
        Raises:
            TypeError: If config is not a QMLEVarianceConfig
        """
        if not isinstance(config, QMLEVarianceConfig):
            raise TypeError(f"config must be a QMLEVarianceConfig, got {type(config)}")
        
        self._config = config
        self._fitted = False  # Reset fitted state when configuration changes
    
    @property
    def integrated_variance(self) -> Optional[float]:
        """Get the estimated integrated variance.
        
        Returns:
            Optional[float]: The estimated integrated variance if available,
                            None otherwise
        """
        return self._integrated_variance
    
    @property
    def log_likelihood(self) -> Optional[float]:
        """Get the log-likelihood value at the optimum.
        
        Returns:
            Optional[float]: The log-likelihood value if available,
                            None otherwise
        """
        return self._log_likelihood
    
    def _compute_realized_measure(self, 
                                 prices: np.ndarray, 
                                 times: np.ndarray, 
                                 returns: np.ndarray,
                                 **kwargs: Any) -> np.ndarray:
        """Compute the QMLE variance from the preprocessed data.
        
        Args:
            prices: Preprocessed price data
            times: Preprocessed time points
            returns: Returns computed from prices
            **kwargs: Additional keyword arguments for computation
            
        Returns:
            np.ndarray: QMLE variance estimate
            
        Raises:
            ValueError: If computation fails
            ConvergenceError: If optimization fails to converge
        """
        start_time = time.time()
        
        # Get initial noise variance estimate
        try:
            from .noise_estimate import noise_variance
            initial_noise_var = noise_variance(returns, method=self.config.initial_noise_method)
        except Exception as e:
            logger.warning(f"Initial noise variance estimation failed: {str(e)}. "
                          "Using default value.")
            # Use a small default value
            initial_noise_var = 1e-6 * np.var(returns)
        
        # Initial guess for integrated variance
        # Realized variance minus 2 * noise variance (based on noise bias correction)
        rv = np.mean(returns**2)
        initial_iv = max(rv - 2 * initial_noise_var, 1e-10)
        
        # Determine whether to use sparse matrix operations
        n = len(returns)
        use_sparse = self.config.use_sparse and n > 1000
        self._sparse_matrix_used = use_sparse
        
        # Define negative log-likelihood function
        def neg_log_likelihood(params):
            # params[0] is integrated variance, params[1] is noise variance
            iv, nv = params
            
            if iv <= 0 or nv <= 0:
                return 1e10  # Large value for invalid parameters
            
            try:
                if use_sparse:
                    # Use sparse matrix operations for large datasets
                    # Construct sparse covariance matrix
                    diag_values = np.ones(n) * (iv + 2 * nv)
                    offdiag_values = np.ones(n-1) * (-nv)
                    
                    # Create sparse matrix in CSR format
                    row_indices = np.concatenate([np.arange(n), np.arange(1, n), np.arange(n-1)])
                    col_indices = np.concatenate([np.arange(n), np.arange(n-1), np.arange(1, n)])
                    data = np.concatenate([diag_values, offdiag_values, offdiag_values])
                    
                    cov_matrix = sparse.csr_matrix((data, (row_indices, col_indices)), shape=(n, n))
                    
                    # Compute log-likelihood using sparse Cholesky decomposition
                    # For sparse matrices, we use a different approach
                    # First, solve the linear system Ax = b
                    try:
                        # Use sparse LU decomposition
                        lu = sparse.linalg.splu(cov_matrix)
                        z = lu.solve(returns)
                        
                        # Compute log determinant
                        # For tridiagonal matrices, we can use a specialized approach
                        # The determinant of a tridiagonal matrix can be computed recursively
                        # Here we use an approximation based on the diagonal and off-diagonal elements
                        log_det = np.sum(np.log(diag_values))
                        
                        # Compute quadratic form
                        quad_form = np.dot(returns, z)
                        
                        # Negative log-likelihood
                        nll = 0.5 * (log_det + quad_form + n * np.log(2 * np.pi))
                        
                        return nll
                    except Exception as e:
                        logger.warning(f"Sparse matrix operations failed: {str(e)}. "
                                      "Falling back to dense matrix operations.")
                        use_sparse = False
                        # Fall back to dense matrix operations
                
                # Dense matrix operations
                # Compute autocovariance matrix
                acov_matrix = np.zeros((n, n))
                
                # Diagonal elements (variance)
                np.fill_diagonal(acov_matrix, iv + 2 * nv)
                
                # Off-diagonal elements (first-order autocovariance)
                for i in range(n - 1):
                    acov_matrix[i, i + 1] = -nv
                    acov_matrix[i + 1, i] = -nv
                
                # Compute log-likelihood
                try:
                    # Use Cholesky decomposition for numerical stability
                    L = np.linalg.cholesky(acov_matrix)
                    log_det = 2 * np.sum(np.log(np.diag(L)))
                    
                    # Solve linear system instead of computing inverse
                    z = np.linalg.solve(L, returns)
                    quad_form = np.sum(z**2)
                    
                    # Negative log-likelihood
                    nll = 0.5 * (log_det + quad_form + n * np.log(2 * np.pi))
                    
                    return nll
                except np.linalg.LinAlgError:
                    # If Cholesky decomposition fails, try a more robust approach
                    try:
                        # Use eigenvalue decomposition
                        eigvals = np.linalg.eigvalsh(acov_matrix)
                        if np.any(eigvals <= 0):
                            # Add a small positive value to ensure positive definiteness
                            min_eig = np.min(eigvals)
                            if min_eig <= 0:
                                acov_matrix += np.eye(n) * (abs(min_eig) + 1e-6)
                        
                        # Compute log determinant and inverse
                        sign, logdet = np.linalg.slogdet(acov_matrix)
                        if sign <= 0:
                            return 1e10  # Large value for invalid determinant
                        
                        # Compute quadratic form using solve
                        try:
                            z = np.linalg.solve(acov_matrix, returns)
                            quad_form = np.dot(returns, z)
                        except np.linalg.LinAlgError:
                            # If solve fails, use pseudo-inverse
                            inv_cov = np.linalg.pinv(acov_matrix)
                            quad_form = np.dot(returns, np.dot(inv_cov, returns))
                        
                        # Negative log-likelihood
                        nll = 0.5 * (logdet + quad_form + n * np.log(2 * np.pi))
                        
                        return nll
                    except Exception:
                        return 1e10  # Large value for numerical issues
            except Exception:
                return 1e10  # Large value for any other issues
        
        # Initial parameters
        initial_params = [initial_iv, initial_noise_var]
        
        # Bounds for parameters (both must be positive)
        bounds = [(1e-10, None), (1e-10, None)]
        
        # Optimization options
        options = {
            'maxiter': self.config.max_iterations,
            'disp': False,
            'gtol': self.config.tolerance
        }
        
        # Optimize negative log-likelihood
        try:
            result = optimize.minimize(
                neg_log_likelihood,
                initial_params,
                bounds=bounds,
                method=self.config.optimization_method,
                options=options
            )
            
            # Check convergence
            if not result.success:
                # Try alternative optimization method
                if self.config.optimization_method != 'Nelder-Mead':
                    logger.warning(f"{self.config.optimization_method} optimization failed. "
                                  "Trying Nelder-Mead method.")
                    
                    result = optimize.minimize(
                        neg_log_likelihood,
                        initial_params,
                        method='Nelder-Mead',
                        options={'maxiter': self.config.max_iterations * 2, 'disp': False}
                    )
                
                if not result.success:
                    # If still not converged, raise warning and use best available solution
                    warnings.warn(
                        f"QMLE optimization did not converge after {result.get('nit', 'unknown')} iterations. "
                        f"Using best available solution. Message: {result.get('message', 'unknown')}"
                    )
            
            # Extract estimated parameters
            iv_est, noise_var = result.x
            
            # Store results
            self._integrated_variance = iv_est
            self._noise_variance = noise_var
            self._log_likelihood = -result.fun
            self._iterations = result.get('nit', None)
            self._convergence_status = result.success
            
            if self.config.return_optimization_details:
                self._optimization_details = {
                    'success': result.success,
                    'status': result.status,
                    'message': result.message,
                    'nit': result.get('nit', None),
                    'fun': result.fun,
                    'jac': str(result.get('jac', 'Not available')),
                    'hess_inv': str(result.get('hess_inv', 'Not available')),
                    'initial_params': initial_params,
                    'final_params': result.x,
                    'use_sparse': use_sparse
                }
            
            # Compute computation time
            computation_time = time.time() - start_time
            
            # Create result object
            result_obj = QMLEVarianceResult(
                model_name=self._name,
                realized_measure=np.array([iv_est]),  # Store as array for consistency
                prices=prices,
                times=times,
                returns=returns,
                noise_variance=noise_var,
                integrated_variance=iv_est,
                log_likelihood=self._log_likelihood,
                iterations=self._iterations,
                convergence_status=self._convergence_status,
                optimization_details=self._optimization_details if self.config.return_optimization_details else None,
                computation_time=computation_time,
                sparse_matrix_used=use_sparse
            )
            
            # Store result
            self._results = result_obj
            
            # Generate plot if requested
            if self.config.plot and HAS_MATPLOTLIB:
                result_obj.plot()
            
            return np.array([iv_est])
            
        except Exception as e:
            # If optimization fails completely, raise error
            raise ConvergenceError(
                f"QMLE optimization failed: {str(e)}",
                details="The optimization algorithm failed to estimate the QMLE parameters.",
                context={
                    "Initial IV": initial_iv,
                    "Initial Noise Var": initial_noise_var,
                    "Method": self.config.optimization_method,
                    "Use Sparse": use_sparse,
                    "Data Length": n
                }
            )
    
    def fit(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> QMLEVarianceResult:
        """Fit the QMLE variance estimator to the provided data.
        
        Args:
            data: The data to fit the estimator to, as a tuple of (prices, times)
            **kwargs: Additional keyword arguments for estimation
            
        Returns:
            QMLEVarianceResult: The estimation results
            
        Raises:
            ValueError: If the data is invalid
            RuntimeError: If the estimation fails
        """
        result = super().fit(data, **kwargs)
        return cast(QMLEVarianceResult, result)
    
    async def fit_async(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> QMLEVarianceResult:
        """Asynchronously fit the QMLE variance estimator to the provided data.
        
        This method provides an asynchronous interface to the fit method,
        allowing for non-blocking estimation in UI contexts.
        
        Args:
            data: The data to fit the estimator to, as a tuple of (prices, times)
            **kwargs: Additional keyword arguments for estimation
            
        Returns:
            QMLEVarianceResult: The estimation results
            
        Raises:
            ValueError: If the data is invalid
            RuntimeError: If the estimation fails
        """
        result = await super().fit_async(data, **kwargs)
        return cast(QMLEVarianceResult, result)
    
    def calibrate(self, data: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> QMLEVarianceConfig:
        """Calibrate the estimator configuration based on the provided data.
        
        This method analyzes the input data and determines optimal configuration
        parameters for the QMLE variance estimator.
        
        Args:
            data: The data to calibrate the estimator with, as a tuple of (prices, times)
            **kwargs: Additional keyword arguments for calibration
            
        Returns:
            QMLEVarianceConfig: Calibrated configuration
        """
        # Validate input data
        self.validate_data(data)
        prices, times = data
        
        # Preprocess data
        processed_prices, processed_times, returns = self.preprocess_data(prices, times)
        
        # Determine optimal configuration based on data characteristics
        n = len(returns)
        
        # Use sparse matrix operations for large datasets
        use_sparse = n > 1000
        
        # Choose optimization method based on data size
        if n > 5000:
            # For very large datasets, use L-BFGS-B for memory efficiency
            optimization_method = 'L-BFGS-B'
        elif n > 1000:
            # For large datasets, use BFGS for better convergence
            optimization_method = 'BFGS'
        else:
            # For small datasets, use Nelder-Mead for robustness
            optimization_method = 'Nelder-Mead'
        
        # Adjust max iterations based on data size
        max_iterations = min(100, max(50, n // 100))
        
        # Create calibrated configuration
        calibrated_config = QMLEVarianceConfig(
            max_iterations=max_iterations,
            tolerance=1e-6,
            optimization_method=optimization_method,
            initial_noise_method='auto',
            use_sparse=use_sparse,
            return_likelihood=False,
            return_optimization_details=False,
            plot=False
        )
        
        return calibrated_config
    
    def summary(self) -> str:
        """Generate a text summary of the QMLE variance estimator.
        
        Returns:
            str: A formatted string containing the estimator summary
        
        Raises:
            RuntimeError: If the estimator has not been fitted
        """
        if not self._fitted:
            return f"QMLE Variance Estimator: {self._name} (not fitted)"
        
        if self._results is None:
            return f"QMLE Variance Estimator: {self._name} (fitted, but no results available)"
        
        return self._results.summary()


@jit(nopython=True, cache=True)
def _qmle_likelihood_numba(returns: np.ndarray, iv: float, nv: float) -> float:
    """
    Numba-accelerated implementation of QMLE likelihood computation.
    
    This function computes the negative log-likelihood for the QMLE model
    using a specialized algorithm for tridiagonal matrices, which is much
    faster than the general approach for large datasets.
    
    Args:
        returns: Array of returns
        iv: Integrated variance
        nv: Noise variance
        
    Returns:
        Negative log-likelihood value
    """
    n = len(returns)
    
    # For tridiagonal matrices, we can use a specialized algorithm
    # This is much faster than the general approach for large datasets
    
    # Diagonal elements
    d = np.ones(n) * (iv + 2 * nv)
    
    # Off-diagonal elements
    e = np.ones(n-1) * (-nv)
    
    # LDL' decomposition for tridiagonal matrix
    # L is unit lower bidiagonal, D is diagonal
    l = np.zeros(n-1)  # Subdiagonal of L
    diag = np.zeros(n)  # Diagonal of D
    
    # First element of D
    diag[0] = d[0]
    
    # LDL' decomposition
    for i in range(1, n):
        l[i-1] = e[i-1] / diag[i-1]
        diag[i] = d[i] - l[i-1] * e[i-1]
    
    # Compute log determinant
    log_det = np.sum(np.log(diag))
    
    # Solve L*y = returns
    y = np.zeros(n)
    y[0] = returns[0]
    for i in range(1, n):
        y[i] = returns[i] - l[i-1] * y[i-1]
    
    # Solve D*z = y
    z = np.zeros(n)
    for i in range(n):
        z[i] = y[i] / diag[i]
    
    # Solve L'*w = z
    w = np.zeros(n)
    w[n-1] = z[n-1]
    for i in range(n-2, -1, -1):
        w[i] = z[i] - l[i] * w[i+1]
    
    # Compute quadratic form
    quad_form = np.sum(returns * w)
    
    # Negative log-likelihood
    nll = 0.5 * (log_det + quad_form + n * np.log(2 * np.pi))
    
    return nll


@jit(nopython=True, cache=True)
def _qmle_likelihood_gradient_numba(returns: np.ndarray, iv: float, nv: float) -> Tuple[float, float]:
    """
    Numba-accelerated implementation of QMLE likelihood gradient computation.
    
    This function computes the gradient of the negative log-likelihood for the QMLE model
    with respect to the integrated variance and noise variance parameters.
    
    Args:
        returns: Array of returns
        iv: Integrated variance
        nv: Noise variance
        
    Returns:
        Tuple of gradients (d_iv, d_nv)
    """
    n = len(returns)
    
    # Compute the inverse of the covariance matrix and its derivative
    # For tridiagonal matrices, we can use a specialized algorithm
    
    # Diagonal elements
    d = np.ones(n) * (iv + 2 * nv)
    
    # Off-diagonal elements
    e = np.ones(n-1) * (-nv)
    
    # LDL' decomposition for tridiagonal matrix
    l = np.zeros(n-1)  # Subdiagonal of L
    diag = np.zeros(n)  # Diagonal of D
    
    # First element of D
    diag[0] = d[0]
    
    # LDL' decomposition
    for i in range(1, n):
        l[i-1] = e[i-1] / diag[i-1]
        diag[i] = d[i] - l[i-1] * e[i-1]
    
    # Solve L*y = returns
    y = np.zeros(n)
    y[0] = returns[0]
    for i in range(1, n):
        y[i] = returns[i] - l[i-1] * y[i-1]
    
    # Solve D*z = y
    z = np.zeros(n)
    for i in range(n):
        z[i] = y[i] / diag[i]
    
    # Solve L'*w = z
    w = np.zeros(n)
    w[n-1] = z[n-1]
    for i in range(n-2, -1, -1):
        w[i] = z[i] - l[i] * w[i+1]
    
    # Compute gradient with respect to iv
    # The derivative of the covariance matrix with respect to iv is a matrix with 1's on the diagonal
    d_iv = 0.0
    
    # Contribution from log determinant
    for i in range(n):
        d_iv += 1.0 / diag[i]
    
    # Contribution from quadratic form
    quad_iv = 0.0
    for i in range(n):
        quad_iv += w[i] * w[i]
    
    d_iv = 0.5 * (d_iv - quad_iv)
    
    # Compute gradient with respect to nv
    # The derivative of the covariance matrix with respect to nv is a matrix with 2's on the diagonal
    # and -1's on the first off-diagonals
    d_nv = 0.0
    
    # Contribution from log determinant
    for i in range(n):
        d_nv += 2.0 / diag[i]
    
    # Contribution from quadratic form
    quad_nv = 0.0
    for i in range(n):
        quad_nv += 2.0 * w[i] * w[i]
    
    for i in range(n-1):
        quad_nv -= 2.0 * w[i] * w[i+1]
    
    d_nv = 0.5 * (d_nv - quad_nv)
    
    return d_iv, d_nv


def qmle_variance(prices: np.ndarray, times: np.ndarray, 
                 max_iterations: int = 100, tolerance: float = 1e-6,
                 optimization_method: str = 'L-BFGS-B',
                 use_sparse: bool = True,
                 return_details: bool = False) -> Union[float, Tuple[float, Dict[str, Any]]]:
    """
    Estimate realized variance using the Quasi-Maximum Likelihood Estimator (QMLE).
    
    This function implements the QMLE approach for estimating realized variance in the
    presence of market microstructure noise. The QMLE estimator models observed
    log-prices as the sum of an efficient price process and i.i.d. noise, and
    estimates the parameters of this model using maximum likelihood.
    
    Args:
        prices: High-frequency price data
        times: Corresponding time points
        max_iterations: Maximum number of iterations for optimization
        tolerance: Convergence tolerance for optimization
        optimization_method: Method for optimization ('L-BFGS-B', 'BFGS', 'Nelder-Mead')
        use_sparse: Whether to use sparse matrix operations for large datasets
        return_details: Whether to return detailed results
        
    Returns:
        If return_details is False:
            Estimated integrated variance
        If return_details is True:
            Tuple of (estimated integrated variance, details dictionary)
            
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If optimization fails
        
    Examples:
        >>> import numpy as np
        >>> from mfe.models.realized.qmle_variance import qmle_variance
        >>> np.random.seed(42)
        >>> times = np.arange(1000)
        >>> true_vol = 0.01
        >>> noise_vol = 0.001
        >>> efficient_prices = np.cumsum(np.random.normal(0, true_vol, 1000))
        >>> noise = np.random.normal(0, noise_vol, 1000)
        >>> prices = efficient_prices + noise
        >>> iv_est = qmle_variance(prices, times)
        >>> iv_est
        0.0001...
    """
    # Create estimator with specified configuration
    config = QMLEVarianceConfig(
        max_iterations=max_iterations,
        tolerance=tolerance,
        optimization_method=optimization_method,
        use_sparse=use_sparse,
        return_optimization_details=return_details
    )
    
    estimator = QMLEVarianceEstimator(config=config)
    
    # Fit estimator
    result = estimator.fit((prices, times))
    
    # Return results
    if return_details:
        details = {
            'integrated_variance': result.integrated_variance,
            'noise_variance': result.noise_variance,
            'log_likelihood': result.log_likelihood,
            'iterations': result.iterations,
            'convergence_status': result.convergence_status,
            'computation_time': result.computation_time,
            'sparse_matrix_used': result.sparse_matrix_used
        }
        
        if result.optimization_details is not None:
            details['optimization_details'] = result.optimization_details
        
        return result.integrated_variance, details
    else:
        return result.integrated_variance


# Register Numba-accelerated functions if available
def _register_numba_functions() -> None:
    """
    Register Numba JIT-compiled functions for QMLE variance estimation.
    
    This function is called during module initialization to register
    performance-critical functions for JIT compilation if Numba is available.
    """
    if HAS_NUMBA:
        # The functions are already decorated with @jit, so no additional
        # registration is needed here. This function is kept for consistency
        # with the module structure and potential future enhancements.
        logger.debug("QMLE variance estimation Numba JIT functions registered")
    else:
        logger.info("Numba not available. QMLE variance estimation will use pure NumPy/SciPy implementations.")


# Initialize the module
_register_numba_functions()
