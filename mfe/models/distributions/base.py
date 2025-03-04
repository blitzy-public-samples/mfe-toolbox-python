'''
Base classes for probability distributions in the MFE Toolbox.

This module defines the foundational classes for all probability distributions
implemented in the MFE Toolbox. It provides a consistent interface for density
functions, cumulative distribution functions, quantile functions, random number
generation, and log-likelihood evaluation.

The base classes implement common functionality and parameter validation,
ensuring that all distribution implementations follow the same patterns and
provide a consistent user experience. The module leverages Python's dataclasses
for parameter containers with validation, and Numba's JIT compilation for
performance-critical operations.
'''

import abc
import math
import warnings
from dataclasses import dataclass, field, asdict
from typing import (
    Any, Callable, Dict, Generic, List, Literal, Optional, Tuple, Type, 
    TypeVar, Union, cast, overload, ClassVar
)

import numpy as np
from scipy import stats, optimize
from numba import jit

from mfe.core.base import DistributionBase
from mfe.core.parameters import (
    ParameterBase, ParameterError, validate_positive, validate_range,
    validate_degrees_of_freedom, transform_positive, inverse_transform_positive,
    transform_correlation, inverse_transform_correlation
)
from mfe.core.exceptions import (
    DistributionError, NumericError, raise_parameter_error, warn_numeric
)
from mfe.core.types import (
    Vector, DistributionType, DistributionLike, ParameterVector,
    PDFFunction, CDFFunction, PPFFunction, RVSFunction
)

# Type variables for generic classes
T = TypeVar('T', bound=ParameterBase)  # Parameter type
D = TypeVar('D')  # Data type


class BaseDistribution(DistributionBase, Generic[T]):
    """Base class for all probability distributions in the MFE Toolbox.
    
    This abstract class defines the common interface that all distribution
    implementations must follow, establishing a consistent API across the
    entire toolbox. It provides default implementations for common methods
    and enforces a consistent structure for parameter handling and validation.
    
    Attributes:
        name: A descriptive name for the distribution
        params: Distribution parameters
    """
    
    def __init__(self, name: str = "Distribution", params: Optional[T] = None):
        """Initialize the distribution.
        
        Args:
            name: A descriptive name for the distribution
            params: Distribution parameters
        """
        super().__init__(name=name)
        self._params = params
        self._validate_params()
    
    @property
    def params(self) -> Optional[T]:
        """Get the distribution parameters.
        
        Returns:
            Optional[T]: The distribution parameters
        """
        return self._params
    
    @params.setter
    def params(self, value: T) -> None:
        """Set the distribution parameters.
        
        Args:
            value: The distribution parameters to set
            
        Raises:
            ParameterError: If the parameters are invalid
        """
        self._params = value
        self._validate_params()
    
    def _validate_params(self) -> None:
        """Validate the distribution parameters.
        
        Raises:
            ParameterError: If the parameters are invalid
        """
        if self._params is not None:
            self._params.validate()
    
    @abc.abstractmethod
    def pdf(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute the probability density function.
        
        This method must be implemented by all subclasses to compute the
        probability density function for the given values.
        
        Args:
            x: Values to compute the PDF for
            **kwargs: Additional keyword arguments for the PDF
        
        Returns:
            np.ndarray: PDF values
            
        Raises:
            DistributionError: If the parameters are not set
            ValueError: If x contains invalid values
        """
        pass
    
    @abc.abstractmethod
    def cdf(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute the cumulative distribution function.
        
        This method must be implemented by all subclasses to compute the
        cumulative distribution function for the given values.
        
        Args:
            x: Values to compute the CDF for
            **kwargs: Additional keyword arguments for the CDF
        
        Returns:
            np.ndarray: CDF values
            
        Raises:
            DistributionError: If the parameters are not set
            ValueError: If x contains invalid values
        """
        pass
    
    @abc.abstractmethod
    def ppf(self, q: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute the percent point function (inverse of CDF).
        
        This method must be implemented by all subclasses to compute the
        percent point function for the given probabilities.
        
        Args:
            q: Probabilities to compute the PPF for
            **kwargs: Additional keyword arguments for the PPF
        
        Returns:
            np.ndarray: PPF values
            
        Raises:
            DistributionError: If the parameters are not set
            ValueError: If q contains values outside [0, 1]
        """
        pass
    
    @abc.abstractmethod
    def rvs(self, 
           size: Union[int, Tuple[int, ...]], 
           random_state: Optional[Union[int, np.random.Generator]] = None,
           **kwargs: Any) -> np.ndarray:
        """Generate random variates from the distribution.
        
        This method must be implemented by all subclasses to generate random
        variates from the distribution.
        
        Args:
            size: Number of random variates to generate
            random_state: Random number generator or seed
            **kwargs: Additional keyword arguments for random variate generation
        
        Returns:
            np.ndarray: Random variates
            
        Raises:
            DistributionError: If the parameters are not set
            ValueError: If size is invalid
        """
        pass
    
    @abc.abstractmethod
    def loglikelihood(self, x: np.ndarray, **kwargs: Any) -> float:
        """Compute the log-likelihood of the data under the distribution.
        
        This method must be implemented by all subclasses to compute the
        log-likelihood of the data under the distribution.
        
        Args:
            x: Data to compute the log-likelihood for
            **kwargs: Additional keyword arguments for the log-likelihood
        
        Returns:
            float: Log-likelihood value
            
        Raises:
            DistributionError: If the parameters are not set
            ValueError: If x contains invalid values
        """
        pass
    
    def fit(self, data: np.ndarray, method: str = "MLE", **kwargs: Any) -> T:
        """Fit the distribution to data.
        
        This method estimates the distribution parameters from data using
        the specified method.
        
        Args:
            data: Data to fit the distribution to
            method: Estimation method (default: "MLE")
            **kwargs: Additional keyword arguments for the estimation method
        
        Returns:
            T: Estimated parameters
            
        Raises:
            ValueError: If data contains invalid values
            NotImplementedError: If the method is not supported
        """
        if method.upper() != "MLE":
            raise NotImplementedError(f"Method {method} is not supported")
        
        # Validate data
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        
        if data.ndim != 1:
            raise ValueError(f"Data must be 1-dimensional, got {data.ndim} dimensions")
        
        if np.isnan(data).any() or np.isinf(data).any():
            raise ValueError("Data contains NaN or infinite values")
        
        # Define the negative log-likelihood function for optimization
        def neg_loglikelihood(params_vec: np.ndarray) -> float:
            try:
                # Convert parameter vector to parameter object
                params = self._vector_to_params(params_vec)
                
                # Create a temporary distribution with these parameters
                temp_dist = self.__class__(params=params)
                
                # Compute log-likelihood
                ll = temp_dist.loglikelihood(data)
                
                # Return negative log-likelihood for minimization
                return -ll
            except (ValueError, ParameterError, NumericError) as e:
                # Return a large value if parameters are invalid
                return 1e10
        
        # Get initial parameter vector
        if self._params is not None:
            initial_params = self._params_to_vector()
        else:
            initial_params = self._get_initial_params(data)
        
        # Optimize using SciPy's minimize function
        result = optimize.minimize(
            neg_loglikelihood,
            initial_params,
            method="BFGS",
            options={"disp": False},
            **kwargs
        )
        
        if not result.success:
            warnings.warn(
                f"Parameter estimation did not converge: {result.message}",
                UserWarning
            )
        
        # Convert optimized parameter vector to parameter object
        estimated_params = self._vector_to_params(result.x)
        
        # Update the distribution parameters
        self._params = estimated_params
        
        return estimated_params
    
    async def fit_async(self, data: np.ndarray, method: str = "MLE", **kwargs: Any) -> T:
        """Asynchronously fit the distribution to data.
        
        This method provides an asynchronous interface to the fit method,
        allowing for non-blocking parameter estimation.
        
        Args:
            data: Data to fit the distribution to
            method: Estimation method (default: "MLE")
            **kwargs: Additional keyword arguments for the estimation method
        
        Returns:
            T: Estimated parameters
            
        Raises:
            ValueError: If data contains invalid values
            NotImplementedError: If the method is not supported
        """
        # This is a simple implementation that just calls the synchronous version
        # In a real implementation, this would use asyncio to avoid blocking
        return self.fit(data, method, **kwargs)
    
    def _params_to_vector(self) -> np.ndarray:
        """Convert parameters to a vector for optimization.
        
        Returns:
            np.ndarray: Parameter vector
            
        Raises:
            DistributionError: If parameters are not set
        """
        if self._params is None:
            raise DistributionError(
                "Parameters are not set",
                distribution_type=self.name
            )
        
        return self._params.transform()
    
    @abc.abstractmethod
    def _vector_to_params(self, vector: np.ndarray) -> T:
        """Convert a parameter vector to a parameter object.
        
        Args:
            vector: Parameter vector
        
        Returns:
            T: Parameter object
            
        Raises:
            ValueError: If the vector has incorrect length
        """
        pass
    
    @abc.abstractmethod
    def _get_initial_params(self, data: np.ndarray) -> np.ndarray:
        """Get initial parameter values for optimization.
        
        Args:
            data: Data to estimate initial parameters from
        
        Returns:
            np.ndarray: Initial parameter vector
        """
        pass
    
    def __str__(self) -> str:
        """Generate a string representation of the distribution.
        
        Returns:
            str: A string representation of the distribution
        """
        params_str = str(self._params) if self._params is not None else "not set"
        return f"{self.name} distribution with parameters: {params_str}"
    
    def __repr__(self) -> str:
        """Generate a detailed string representation of the distribution.
        
        Returns:
            str: A detailed string representation of the distribution
        """
        params_repr = repr(self._params) if self._params is not None else "None"
        return f"{self.__class__.__name__}(name='{self.name}', params={params_repr})"


class ContinuousDistribution(BaseDistribution[T], stats.rv_continuous):
    """Base class for continuous probability distributions.
    
    This class extends both BaseDistribution and scipy.stats.rv_continuous,
    providing a bridge between the MFE Toolbox's distribution interface and
    SciPy's distribution framework. It allows MFE distributions to leverage
    SciPy's functionality while maintaining the MFE Toolbox's consistent API.
    
    Attributes:
        name: A descriptive name for the distribution
        params: Distribution parameters
    """
    
    def __init__(self, name: str = "ContinuousDistribution", params: Optional[T] = None):
        """Initialize the continuous distribution.
        
        Args:
            name: A descriptive name for the distribution
            params: Distribution parameters
        """
        # Initialize BaseDistribution
        BaseDistribution.__init__(self, name=name, params=params)
        
        # Initialize scipy.stats.rv_continuous with a dummy _pdf method
        # The actual PDF implementation will be provided by subclasses
        stats.rv_continuous.__init__(self, name=name.lower())
    
    def _pdf(self, x: np.ndarray, *args: Any) -> np.ndarray:
        """Probability density function for scipy.stats.rv_continuous.
        
        This is a placeholder that should be overridden by subclasses.
        
        Args:
            x: Points at which to evaluate the PDF
            *args: Distribution parameters
        
        Returns:
            np.ndarray: PDF values
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("_pdf must be implemented by subclass")
    
    def pdf(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute the probability density function.
        
        Args:
            x: Values to compute the PDF for
            **kwargs: Additional keyword arguments for the PDF
        
        Returns:
            np.ndarray: PDF values
            
        Raises:
            DistributionError: If the parameters are not set
            ValueError: If x contains invalid values
        """
        if self._params is None:
            raise DistributionError(
                "Parameters are not set",
                distribution_type=self.name
            )
        
        # Convert input to numpy array if needed
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        
        # Check for invalid values
        if np.isnan(x).any() or np.isinf(x).any():
            raise ValueError("Input contains NaN or infinite values")
        
        # Use the scipy.stats implementation
        return self._custom_pdf(x, **kwargs)
    
    @abc.abstractmethod
    def _custom_pdf(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Custom implementation of the PDF.
        
        This method should be implemented by subclasses to provide a
        potentially optimized implementation of the PDF.
        
        Args:
            x: Values to compute the PDF for
            **kwargs: Additional keyword arguments for the PDF
        
        Returns:
            np.ndarray: PDF values
        """
        pass
    
    def cdf(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute the cumulative distribution function.
        
        Args:
            x: Values to compute the CDF for
            **kwargs: Additional keyword arguments for the CDF
        
        Returns:
            np.ndarray: CDF values
            
        Raises:
            DistributionError: If the parameters are not set
            ValueError: If x contains invalid values
        """
        if self._params is None:
            raise DistributionError(
                "Parameters are not set",
                distribution_type=self.name
            )
        
        # Convert input to numpy array if needed
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        
        # Check for invalid values
        if np.isnan(x).any() or np.isinf(x).any():
            raise ValueError("Input contains NaN or infinite values")
        
        # Use the scipy.stats implementation or custom implementation
        return self._custom_cdf(x, **kwargs)
    
    @abc.abstractmethod
    def _custom_cdf(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Custom implementation of the CDF.
        
        This method should be implemented by subclasses to provide a
        potentially optimized implementation of the CDF.
        
        Args:
            x: Values to compute the CDF for
            **kwargs: Additional keyword arguments for the CDF
        
        Returns:
            np.ndarray: CDF values
        """
        pass
    
    def ppf(self, q: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute the percent point function (inverse of CDF).
        
        Args:
            q: Probabilities to compute the PPF for
            **kwargs: Additional keyword arguments for the PPF
        
        Returns:
            np.ndarray: PPF values
            
        Raises:
            DistributionError: If the parameters are not set
            ValueError: If q contains values outside [0, 1]
        """
        if self._params is None:
            raise DistributionError(
                "Parameters are not set",
                distribution_type=self.name
            )
        
        # Convert input to numpy array if needed
        if not isinstance(q, np.ndarray):
            q = np.asarray(q)
        
        # Check for invalid values
        if np.isnan(q).any() or np.isinf(q).any():
            raise ValueError("Input contains NaN or infinite values")
        
        if np.any((q < 0) | (q > 1)):
            raise ValueError("Probabilities must be between 0 and 1")
        
        # Use the scipy.stats implementation or custom implementation
        return self._custom_ppf(q, **kwargs)
    
    @abc.abstractmethod
    def _custom_ppf(self, q: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Custom implementation of the PPF.
        
        This method should be implemented by subclasses to provide a
        potentially optimized implementation of the PPF.
        
        Args:
            q: Probabilities to compute the PPF for
            **kwargs: Additional keyword arguments for the PPF
        
        Returns:
            np.ndarray: PPF values
        """
        pass
    
    def rvs(self, 
           size: Union[int, Tuple[int, ...]], 
           random_state: Optional[Union[int, np.random.Generator]] = None,
           **kwargs: Any) -> np.ndarray:
        """Generate random variates from the distribution.
        
        Args:
            size: Number of random variates to generate
            random_state: Random number generator or seed
            **kwargs: Additional keyword arguments for random variate generation
        
        Returns:
            np.ndarray: Random variates
            
        Raises:
            DistributionError: If the parameters are not set
            ValueError: If size is invalid
        """
        if self._params is None:
            raise DistributionError(
                "Parameters are not set",
                distribution_type=self.name
            )
        
        # Handle random state
        if random_state is None:
            rng = np.random.default_rng()
        elif isinstance(random_state, int):
            rng = np.random.default_rng(random_state)
        else:
            rng = random_state
        
        # Use the scipy.stats implementation or custom implementation
        return self._custom_rvs(size, rng, **kwargs)
    
    @abc.abstractmethod
    def _custom_rvs(self, 
                   size: Union[int, Tuple[int, ...]], 
                   random_state: np.random.Generator,
                   **kwargs: Any) -> np.ndarray:
        """Custom implementation of random variate generation.
        
        This method should be implemented by subclasses to provide a
        potentially optimized implementation of random variate generation.
        
        Args:
            size: Number of random variates to generate
            random_state: Random number generator
            **kwargs: Additional keyword arguments for random variate generation
        
        Returns:
            np.ndarray: Random variates
        """
        pass
    
    def loglikelihood(self, x: np.ndarray, **kwargs: Any) -> float:
        """Compute the log-likelihood of the data under the distribution.
        
        Args:
            x: Data to compute the log-likelihood for
            **kwargs: Additional keyword arguments for the log-likelihood
        
        Returns:
            float: Log-likelihood value
            
        Raises:
            DistributionError: If the parameters are not set
            ValueError: If x contains invalid values
        """
        if self._params is None:
            raise DistributionError(
                "Parameters are not set",
                distribution_type=self.name
            )
        
        # Convert input to numpy array if needed
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        
        # Check for invalid values
        if np.isnan(x).any() or np.isinf(x).any():
            raise ValueError("Input contains NaN or infinite values")
        
        # Compute log-likelihood using PDF
        log_pdf = np.log(self.pdf(x, **kwargs))
        
        # Handle potential numerical issues
        if np.isnan(log_pdf).any() or np.isinf(log_pdf).any():
            raise NumericError(
                "Log-likelihood computation resulted in NaN or infinite values",
                operation="log-likelihood",
                values=log_pdf,
                error_type="numerical instability"
            )
        
        return np.sum(log_pdf)


class NumbaDistribution(BaseDistribution[T]):
    """Base class for distributions with Numba-accelerated implementations.
    
    This class extends BaseDistribution to provide a framework for implementing
    distributions with performance-critical operations accelerated using Numba's
    JIT compilation. It defines the interface for Numba-accelerated methods and
    provides default implementations that delegate to the JIT-compiled functions.
    
    Attributes:
        name: A descriptive name for the distribution
        params: Distribution parameters
    """
    
    # Class variables for JIT-compiled functions
    _jit_pdf: ClassVar[Optional[Callable]] = None
    _jit_cdf: ClassVar[Optional[Callable]] = None
    _jit_ppf: ClassVar[Optional[Callable]] = None
    _jit_loglikelihood: ClassVar[Optional[Callable]] = None
    
    def __init__(self, name: str = "NumbaDistribution", params: Optional[T] = None):
        """Initialize the Numba-accelerated distribution.
        
        Args:
            name: A descriptive name for the distribution
            params: Distribution parameters
        """
        super().__init__(name=name, params=params)
        
        # Ensure JIT-compiled functions are initialized
        self._initialize_jit_functions()
    
    @classmethod
    def _initialize_jit_functions(cls) -> None:
        """Initialize JIT-compiled functions if not already done.
        
        This method should be implemented by subclasses to initialize
        the JIT-compiled functions for the distribution.
        """
        pass
    
    def pdf(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute the probability density function using Numba acceleration.
        
        Args:
            x: Values to compute the PDF for
            **kwargs: Additional keyword arguments for the PDF
        
        Returns:
            np.ndarray: PDF values
            
        Raises:
            DistributionError: If the parameters are not set
            ValueError: If x contains invalid values
            RuntimeError: If JIT-compiled function is not initialized
        """
        if self._params is None:
            raise DistributionError(
                "Parameters are not set",
                distribution_type=self.name
            )
        
        # Convert input to numpy array if needed
        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=np.float64)
        elif x.dtype != np.float64:
            x = x.astype(np.float64)
        
        # Check for invalid values
        if np.isnan(x).any() or np.isinf(x).any():
            raise ValueError("Input contains NaN or infinite values")
        
        # Ensure JIT-compiled function is initialized
        if self.__class__._jit_pdf is None:
            self._initialize_jit_functions()
            if self.__class__._jit_pdf is None:
                raise RuntimeError("JIT-compiled PDF function is not initialized")
        
        # Extract parameters for JIT function
        param_tuple = self._params_to_tuple()
        
        # Call JIT-compiled function
        return self.__class__._jit_pdf(x, *param_tuple)
    
    def cdf(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute the cumulative distribution function using Numba acceleration.
        
        Args:
            x: Values to compute the CDF for
            **kwargs: Additional keyword arguments for the CDF
        
        Returns:
            np.ndarray: CDF values
            
        Raises:
            DistributionError: If the parameters are not set
            ValueError: If x contains invalid values
            RuntimeError: If JIT-compiled function is not initialized
        """
        if self._params is None:
            raise DistributionError(
                "Parameters are not set",
                distribution_type=self.name
            )
        
        # Convert input to numpy array if needed
        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=np.float64)
        elif x.dtype != np.float64:
            x = x.astype(np.float64)
        
        # Check for invalid values
        if np.isnan(x).any() or np.isinf(x).any():
            raise ValueError("Input contains NaN or infinite values")
        
        # Ensure JIT-compiled function is initialized
        if self.__class__._jit_cdf is None:
            self._initialize_jit_functions()
            if self.__class__._jit_cdf is None:
                raise RuntimeError("JIT-compiled CDF function is not initialized")
        
        # Extract parameters for JIT function
        param_tuple = self._params_to_tuple()
        
        # Call JIT-compiled function
        return self.__class__._jit_cdf(x, *param_tuple)
    
    def ppf(self, q: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute the percent point function using Numba acceleration.
        
        Args:
            q: Probabilities to compute the PPF for
            **kwargs: Additional keyword arguments for the PPF
        
        Returns:
            np.ndarray: PPF values
            
        Raises:
            DistributionError: If the parameters are not set
            ValueError: If q contains values outside [0, 1]
            RuntimeError: If JIT-compiled function is not initialized
        """
        if self._params is None:
            raise DistributionError(
                "Parameters are not set",
                distribution_type=self.name
            )
        
        # Convert input to numpy array if needed
        if not isinstance(q, np.ndarray):
            q = np.asarray(q, dtype=np.float64)
        elif q.dtype != np.float64:
            q = q.astype(np.float64)
        
        # Check for invalid values
        if np.isnan(q).any() or np.isinf(q).any():
            raise ValueError("Input contains NaN or infinite values")
        
        if np.any((q < 0) | (q > 1)):
            raise ValueError("Probabilities must be between 0 and 1")
        
        # Ensure JIT-compiled function is initialized
        if self.__class__._jit_ppf is None:
            self._initialize_jit_functions()
            if self.__class__._jit_ppf is None:
                raise RuntimeError("JIT-compiled PPF function is not initialized")
        
        # Extract parameters for JIT function
        param_tuple = self._params_to_tuple()
        
        # Call JIT-compiled function
        return self.__class__._jit_ppf(q, *param_tuple)
    
    def rvs(self, 
           size: Union[int, Tuple[int, ...]], 
           random_state: Optional[Union[int, np.random.Generator]] = None,
           **kwargs: Any) -> np.ndarray:
        """Generate random variates from the distribution.
        
        Args:
            size: Number of random variates to generate
            random_state: Random number generator or seed
            **kwargs: Additional keyword arguments for random variate generation
        
        Returns:
            np.ndarray: Random variates
            
        Raises:
            DistributionError: If the parameters are not set
            ValueError: If size is invalid
        """
        if self._params is None:
            raise DistributionError(
                "Parameters are not set",
                distribution_type=self.name
            )
        
        # Handle random state
        if random_state is None:
            rng = np.random.default_rng()
        elif isinstance(random_state, int):
            rng = np.random.default_rng(random_state)
        else:
            rng = random_state
        
        # Generate uniform random numbers
        u = rng.random(size=size)
        
        # Use inverse transform sampling (PPF of uniform -> PPF of target)
        return self.ppf(u)
    
    def loglikelihood(self, x: np.ndarray, **kwargs: Any) -> float:
        """Compute the log-likelihood using Numba acceleration.
        
        Args:
            x: Data to compute the log-likelihood for
            **kwargs: Additional keyword arguments for the log-likelihood
        
        Returns:
            float: Log-likelihood value
            
        Raises:
            DistributionError: If the parameters are not set
            ValueError: If x contains invalid values
            RuntimeError: If JIT-compiled function is not initialized
        """
        if self._params is None:
            raise DistributionError(
                "Parameters are not set",
                distribution_type=self.name
            )
        
        # Convert input to numpy array if needed
        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=np.float64)
        elif x.dtype != np.float64:
            x = x.astype(np.float64)
        
        # Check for invalid values
        if np.isnan(x).any() or np.isinf(x).any():
            raise ValueError("Input contains NaN or infinite values")
        
        # Use JIT-compiled function if available
        if self.__class__._jit_loglikelihood is not None:
            # Extract parameters for JIT function
            param_tuple = self._params_to_tuple()
            
            # Call JIT-compiled function
            return self.__class__._jit_loglikelihood(x, *param_tuple)
        else:
            # Fallback to computing log-likelihood using PDF
            log_pdf = np.log(self.pdf(x))
            
            # Handle potential numerical issues
            if np.isnan(log_pdf).any() or np.isinf(log_pdf).any():
                raise NumericError(
                    "Log-likelihood computation resulted in NaN or infinite values",
                    operation="log-likelihood",
                    values=log_pdf,
                    error_type="numerical instability"
                )
            
            return np.sum(log_pdf)
    
    @abc.abstractmethod
    def _params_to_tuple(self) -> Tuple[float, ...]:
        """Convert parameters to a tuple for JIT functions.
        
        Returns:
            Tuple[float, ...]: Parameter tuple
            
        Raises:
            DistributionError: If parameters are not set
        """
        pass


@dataclass
class NormalParams(ParameterBase):
    """Parameters for the normal distribution.
    
    Attributes:
        mu: Mean parameter
        sigma: Standard deviation parameter (must be positive)
    """
    
    mu: float = 0.0
    sigma: float = 1.0
    
    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate normal distribution parameter constraints.
        
        Raises:
            ParameterError: If parameter constraints are violated
        """
        # Validate sigma (must be positive)
        validate_positive(self.sigma, "sigma")
    
    def to_array(self) -> np.ndarray:
        """Convert parameters to a NumPy array.
        
        Returns:
            np.ndarray: Array representation of parameters
        """
        return np.array([self.mu, self.sigma])
    
    @classmethod
    def from_array(cls, array: np.ndarray, **kwargs: Any) -> 'NormalParams':
        """Create parameters from a NumPy array.
        
        Args:
            array: Array representation of parameters [mu, sigma]
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            NormalParams: Parameter object
            
        Raises:
            ValueError: If the array length is not 2
        """
        if len(array) != 2:
            raise ValueError(f"Array length must be 2, got {len(array)}")
        
        return cls(mu=array[0], sigma=array[1])
    
    def transform(self) -> np.ndarray:
        """Transform parameters to unconstrained space for optimization.
        
        Returns:
            np.ndarray: Transformed parameters in unconstrained space
        """
        # mu is already unconstrained
        # Transform sigma to unconstrained space (log)
        transformed_sigma = transform_positive(self.sigma)
        
        return np.array([self.mu, transformed_sigma])
    
    @classmethod
    def inverse_transform(cls, array: np.ndarray, **kwargs: Any) -> 'NormalParams':
        """Transform parameters from unconstrained space back to constrained space.
        
        Args:
            array: Parameters in unconstrained space [mu, log(sigma)]
            **kwargs: Additional keyword arguments for parameter creation
        
        Returns:
            NormalParams: Parameter object with constrained parameters
            
        Raises:
            ValueError: If the array length is not 2
        """
        if len(array) != 2:
            raise ValueError(f"Array length must be 2, got {len(array)}")
        
        # Extract transformed parameters
        mu = array[0]  # mu is already unconstrained
        transformed_sigma = array[1]
        
        # Inverse transform sigma
        sigma = inverse_transform_positive(transformed_sigma)
        
        return cls(mu=mu, sigma=sigma)


# Initialize Numba JIT-compiled functions for normal distribution
@jit(nopython=True, cache=True)
def _normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Numba-accelerated PDF for normal distribution.
    
    Args:
        x: Values to compute the PDF for
        mu: Mean parameter
        sigma: Standard deviation parameter
    
    Returns:
        np.ndarray: PDF values
    """
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))


@jit(nopython=True, cache=True)
def _normal_loglikelihood(x: np.ndarray, mu: float, sigma: float) -> float:
    """Numba-accelerated log-likelihood for normal distribution.
    
    Args:
        x: Data to compute the log-likelihood for
        mu: Mean parameter
        sigma: Standard deviation parameter
    
    Returns:
        float: Log-likelihood value
    """
    n = len(x)
    z = (x - mu) / sigma
    return -0.5 * n * np.log(2.0 * np.pi) - n * np.log(sigma) - 0.5 * np.sum(z * z)


class Normal(NumbaDistribution[NormalParams]):
    """Normal distribution implementation with Numba acceleration.
    
    This class implements the normal (Gaussian) distribution with parameters
    mu (mean) and sigma (standard deviation). It provides Numba-accelerated
    implementations of the PDF, CDF, PPF, and log-likelihood functions.
    
    Attributes:
        name: A descriptive name for the distribution
        params: Distribution parameters (mu, sigma)
    """
    
    # Class variables for JIT-compiled functions
    _jit_pdf = staticmethod(_normal_pdf)
    _jit_loglikelihood = staticmethod(_normal_loglikelihood)
    
    def __init__(self, 
                name: str = "Normal", 
                params: Optional[NormalParams] = None):
        """Initialize the normal distribution.
        
        Args:
            name: A descriptive name for the distribution
            params: Distribution parameters (mu, sigma)
        """
        if params is None:
            params = NormalParams()
        
        super().__init__(name=name, params=params)
    
    def _params_to_tuple(self) -> Tuple[float, float]:
        """Convert parameters to a tuple for JIT functions.
        
        Returns:
            Tuple[float, float]: Parameter tuple (mu, sigma)
            
        Raises:
            DistributionError: If parameters are not set
        """
        if self._params is None:
            raise DistributionError(
                "Parameters are not set",
                distribution_type=self.name
            )
        
        return (self._params.mu, self._params.sigma)
    
    def _vector_to_params(self, vector: np.ndarray) -> NormalParams:
        """Convert a parameter vector to a parameter object.
        
        Args:
            vector: Parameter vector [mu, log(sigma)]
        
        Returns:
            NormalParams: Parameter object
            
        Raises:
            ValueError: If the vector has incorrect length
        """
        return NormalParams.inverse_transform(vector)
    
    def _get_initial_params(self, data: np.ndarray) -> np.ndarray:
        """Get initial parameter values for optimization.
        
        Args:
            data: Data to estimate initial parameters from
        
        Returns:
            np.ndarray: Initial parameter vector [mu, log(sigma)]
        """
        # Estimate mu and sigma from data
        mu_est = np.mean(data)
        sigma_est = np.std(data, ddof=1)
        
        # Ensure sigma is positive
        if sigma_est <= 0:
            sigma_est = 0.1
        
        # Transform to unconstrained space
        return np.array([mu_est, np.log(sigma_est)])
    
    def cdf(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute the cumulative distribution function.
        
        Args:
            x: Values to compute the CDF for
            **kwargs: Additional keyword arguments for the CDF
        
        Returns:
            np.ndarray: CDF values
            
        Raises:
            DistributionError: If the parameters are not set
            ValueError: If x contains invalid values
        """
        if self._params is None:
            raise DistributionError(
                "Parameters are not set",
                distribution_type=self.name
            )
        
        # Convert input to numpy array if needed
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        
        # Check for invalid values
        if np.isnan(x).any() or np.isinf(x).any():
            raise ValueError("Input contains NaN or infinite values")
        
        # Use scipy.stats.norm for CDF computation
        z = (x - self._params.mu) / self._params.sigma
        return stats.norm.cdf(z)
    
    def ppf(self, q: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute the percent point function (inverse of CDF).
        
        Args:
            q: Probabilities to compute the PPF for
            **kwargs: Additional keyword arguments for the PPF
        
        Returns:
            np.ndarray: PPF values
            
        Raises:
            DistributionError: If the parameters are not set
            ValueError: If q contains values outside [0, 1]
        """
        if self._params is None:
            raise DistributionError(
                "Parameters are not set",
                distribution_type=self.name
            )
        
        # Convert input to numpy array if needed
        if not isinstance(q, np.ndarray):
            q = np.asarray(q)
        
        # Check for invalid values
        if np.isnan(q).any() or np.isinf(q).any():
            raise ValueError("Input contains NaN or infinite values")
        
        if np.any((q < 0) | (q > 1)):
            raise ValueError("Probabilities must be between 0 and 1")
        
        # Use scipy.stats.norm for PPF computation
        return self._params.mu + self._params.sigma * stats.norm.ppf(q)