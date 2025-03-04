# mfe/core/parameters.py

"""
Parameter containers and validation infrastructure for the MFE Toolbox.

This module provides a comprehensive parameter management system using Python's
dataclasses with field validation. It implements parameter containers for different
model families, constraint validation decorators, and helper functions for parameter
transformation between constrained and unconstrained spaces.

The parameter system ensures type safety, enforces model-specific constraints,
and provides a consistent interface for parameter handling across the toolbox.
"""

from dataclasses import fields
import math
import warnings
from dataclasses import dataclass, field, asdict, is_dataclass
from typing import (
    Any, Callable, Dict, Generic, List, Literal, Optional, Protocol,
    Sequence, Tuple, Type, TypeVar, Union, cast, get_type_hints, overload
)
import numpy as np
from scipy import stats

# Type variables for generic parameter classes
T = TypeVar('T')  # Generic type for parameters
P = TypeVar('P', bound='ParameterBase')  # Generic type for parameter subclasses


class ParameterError(Exception):
    """Exception raised for parameter constraint violations."""
    pass


class ParameterBase:
    """Base class for all parameter containers.

    This class provides common functionality for parameter validation,
    transformation, and serialization that is shared across all parameter types.
    """

    def validate(self) -> None:
        """Validate parameter constraints.

        This method should be implemented by subclasses to enforce
        model-specific parameter constraints.

        Raises:
            ParameterError: If parameter constraints are violated
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of parameters
        """
        if is_dataclass(self):
            return asdict(self)
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def to_array(self) -> np.ndarray:
        """Convert parameters to a NumPy array.

        Returns:
            np.ndarray: Array representation of parameters

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("to_array must be implemented by subclass")

    @classmethod
    def from_array(cls: Type[P], array: np.ndarray, **kwargs: Any) -> P:
        """Create parameters from a NumPy array.

        Args:
            array: Array representation of parameters
            **kwargs: Additional keyword arguments for parameter creation

        Returns:
            P: Parameter object

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("from_array must be implemented by subclass")

    def transform(self) -> np.ndarray:
        """Transform parameters to unconstrained space for optimization.

        Returns:
            np.ndarray: Transformed parameters in unconstrained space

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("transform must be implemented by subclass")

    @classmethod
    def inverse_transform(cls: Type[P], array: np.ndarray, **kwargs: Any) -> P:
        """Transform parameters from unconstrained space back to constrained space.

        Args:
            array: Parameters in unconstrained space
            **kwargs: Additional keyword arguments for parameter creation

        Returns:
            P: Parameter object with constrained parameters

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("inverse_transform must be implemented by subclass")

    def copy(self: P) -> P:
        """Create a copy of the parameter object.

        Returns:
            P: Copy of the parameter object
        """
        if is_dataclass(self):
            # For dataclasses, use the constructor with the current values
            return type(self)(**self.to_dict())
        # For regular classes, create a new instance and copy attributes
        new_instance = type(self)()
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                setattr(new_instance, k, v)
        return new_instance


# Parameter validation decorators

def validate_positive(value: float, param_name: str) -> float:
    """Validate that a parameter is positive.

    Args:
        value: Parameter value to validate
        param_name: Name of the parameter for error messages

    Returns:
        float: The validated parameter value

    Raises:
        ParameterError: If the parameter is not positive
    """
    if value <= 0:
        raise ParameterError(f"Parameter {param_name} must be positive, got {value}")
    return value


def validate_non_negative(value: float, param_name: str) -> float:
    """Validate that a parameter is non-negative.

    Args:
        value: Parameter value to validate
        param_name: Name of the parameter for error messages

    Returns:
        float: The validated parameter value

    Raises:
        ParameterError: If the parameter is negative
    """
    if value < 0:
        raise ParameterError(f"Parameter {param_name} must be non-negative, got {value}")
    return value


def validate_probability(value: float, param_name: str) -> float:
    """Validate that a parameter is a probability (between 0 and 1).

    Args:
        value: Parameter value to validate
        param_name: Name of the parameter for error messages

    Returns:
        float: The validated parameter value

    Raises:
        ParameterError: If the parameter is not between 0 and 1
    """
    if value < 0 or value > 1:
        raise ParameterError(
            f"Parameter {param_name} must be between 0 and 1, got {value}"
        )
    return value


def validate_range(value: float, param_name: str,
                   min_value: Optional[float] = None,
                   max_value: Optional[float] = None) -> float:
    """Validate that a parameter is within a specified range.

    Args:
        value: Parameter value to validate
        param_name: Name of the parameter for error messages
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)

    Returns:
        float: The validated parameter value

    Raises:
        ParameterError: If the parameter is outside the specified range
    """
    if min_value is not None and value < min_value:
        raise ParameterError(
            f"Parameter {param_name} must be at least {min_value}, got {value}"
        )
    if max_value is not None and value > max_value:
        raise ParameterError(
            f"Parameter {param_name} must be at most {max_value}, got {value}"
        )
    return value


def validate_degrees_of_freedom(value: float, param_name: str) -> float:
    """Validate that a degrees of freedom parameter is valid (> 2).

    Args:
        value: Parameter value to validate
        param_name: Name of the parameter for error messages

    Returns:
        float: The validated parameter value

    Raises:
        ParameterError: If the parameter is not greater than 2
    """
    if value <= 2:
        raise ParameterError(
            f"Parameter {param_name} (degrees of freedom) must be greater than 2, got {value}"
        )
    return value


def validate_positive_definite(matrix: np.ndarray, param_name: str) -> np.ndarray:
    """Validate that a matrix is positive definite.

    Args:
        matrix: Matrix to validate
        param_name: Name of the parameter for error messages

    Returns:
        np.ndarray: The validated matrix

    Raises:
        ParameterError: If the matrix is not positive definite
    """
    try:
        # Attempt Cholesky decomposition, which only works for positive definite matrices
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        raise ParameterError(f"Matrix {param_name} must be positive definite")
    return matrix


# Parameter transformation functions

def transform_positive(value: float) -> float:
    """Transform a positive parameter to unconstrained space using log.

    Args:
        value: Positive parameter value

    Returns:
        float: Transformed parameter in unconstrained space
    """
    return np.log(value)


def inverse_transform_positive(value: float) -> float:
    """Transform a parameter from unconstrained space to positive space using exp.

    Args:
        value: Parameter value in unconstrained space

    Returns:
        float: Positive parameter value
    """
    return np.exp(value)


def transform_probability(value: float) -> float:
    """Transform a probability parameter to unconstrained space using logit.

    Args:
        value: Probability parameter value (between 0 and 1)

    Returns:
        float: Transformed parameter in unconstrained space
    """
    # Add small epsilon to avoid log(0) or log(1)
    eps = np.finfo(float).eps
    value = np.clip(value, eps, 1 - eps)
    return np.log(value / (1 - value))


def inverse_transform_probability(value: float) -> float:
    """Transform a parameter from unconstrained space to probability space using sigmoid.

    Args:
        value: Parameter value in unconstrained space

    Returns:
        float: Probability parameter value (between 0 and 1)
    """
    return 1.0 / (1.0 + np.exp(-value))


def transform_correlation(value: float) -> float:
    """Transform a correlation parameter to unconstrained space.

    Args:
        value: Correlation parameter value (between -1 and 1)

    Returns:
        float: Transformed parameter in unconstrained space
    """
    # Add small epsilon to avoid extreme values
    eps = np.finfo(float).eps
    value = np.clip(value, -1 + eps, 1 - eps)
    return np.arctanh(value)  # Fisher's z-transformation


def inverse_transform_correlation(value: float) -> float:
    """Transform a parameter from unconstrained space to correlation space.

    Args:
        value: Parameter value in unconstrained space

    Returns:
        float: Correlation parameter value (between -1 and 1)
    """
    return np.tanh(value)  # Inverse of Fisher's z-transformation


# Base parameter classes for different model types

@dataclass
class UnivariateVolatilityParameters(ParameterBase):
    """Base class for univariate volatility model parameters.

    This class provides common functionality for univariate volatility model
    parameters, including validation and transformation methods.
    """

    def validate(self) -> None:
        """Validate parameter constraints for univariate volatility models.

        Raises:
            ParameterError: If parameter constraints are violated
        """
        # Base class doesn't enforce specific constraints
        # Subclasses should implement model-specific constraints
        pass

    def to_array(self) -> np.ndarray:
        """Convert parameters to a NumPy array.

        Returns:
            np.ndarray: Array representation of parameters
        """
        return np.array(list(self.to_dict().values()))

    @classmethod
    def from_array(cls, array: np.ndarray, **kwargs: Any) -> 'UnivariateVolatilityParameters':
        """Create parameters from a NumPy array.

        Args:
            array: Array representation of parameters
            **kwargs: Additional keyword arguments for parameter creation

        Returns:
            UnivariateVolatilityParameters: Parameter object

        Raises:
            ValueError: If the array length doesn't match the number of parameters
        """
        # Get parameter names from dataclass fields
        param_names = list(field.name for field in fields(cls))

        if len(array) != len(param_names):
            raise ValueError(
                f"Array length ({len(array)}) doesn't match number of parameters ({len(param_names)})"
            )

        # Create parameter dictionary from array
        param_dict = {name: array[i] for i, name in enumerate(param_names)}

        # Update with any additional kwargs
        param_dict.update(kwargs)

        # Create parameter object
        return cls(**param_dict)


@dataclass
class GARCHParameters(UnivariateVolatilityParameters):
    """Parameters for GARCH(1,1) model.

    Attributes:
        omega: Constant term in variance equation (must be positive)
        alpha: ARCH parameter (must be non-negative)
        beta: GARCH parameter (must be non-negative)
    """

    omega: float
    alpha: float
    beta: float

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate GARCH parameter constraints.

        Raises:
            ParameterError: If parameter constraints are violated
        """
        super().validate()

        # Validate individual parameters
        validate_positive(self.omega, "omega")
        validate_non_negative(self.alpha, "alpha")
        validate_non_negative(self.beta, "beta")

        # Validate stationarity constraint
        if self.alpha + self.beta >= 1:
            raise ParameterError(
                f"GARCH stationarity constraint violated: alpha + beta = {self.alpha + self.beta} >= 1"
            )

    def transform(self) -> np.ndarray:
        """Transform parameters to unconstrained space for optimization.

        Returns:
            np.ndarray: Transformed parameters in unconstrained space
        """
        # Transform omega to unconstrained space (log)
        transformed_omega = transform_positive(self.omega)

        # Transform alpha and beta to unconstrained space
        # We use a special transformation to ensure alpha + beta < 1
        if self.alpha + self.beta >= 1:
            # If constraint is violated, adjust parameters slightly
            sum_ab = self.alpha + self.beta
            self.alpha = self.alpha / (sum_ab + 0.01)
            self.beta = self.beta / (sum_ab + 0.01)

        # Use logit-like transformation for alpha and beta
        gamma = self.alpha + self.beta
        delta = self.alpha / gamma if gamma > 0 else 0.5

        transformed_gamma = transform_probability(gamma)
        transformed_delta = transform_probability(delta)

        return np.array([transformed_omega, transformed_gamma, transformed_delta])

    @classmethod
    def inverse_transform(cls, array: np.ndarray, **kwargs: Any) -> 'GARCHParameters':
        """Transform parameters from unconstrained space back to constrained space.

        Args:
            array: Parameters in unconstrained space [omega*, gamma*, delta*]
            **kwargs: Additional keyword arguments for parameter creation

        Returns:
            GARCHParameters: Parameter object with constrained parameters

        Raises:
            ValueError: If the array length is not 3
        """
        if len(array) != 3:
            raise ValueError(f"Array length must be 3, got {len(array)}")

        # Extract transformed parameters
        transformed_omega, transformed_gamma, transformed_delta = array

        # Inverse transform omega
        omega = inverse_transform_positive(transformed_omega)

        # Inverse transform gamma (alpha + beta) and delta (alpha / (alpha + beta))
        gamma = inverse_transform_probability(transformed_gamma)
        delta = inverse_transform_probability(transformed_delta)

        # Compute alpha and beta
        alpha = gamma * delta
        beta = gamma * (1 - delta)

        return cls(omega=omega, alpha=alpha, beta=beta)


@dataclass
class EGARCHParameters(UnivariateVolatilityParameters):
    """Parameters for EGARCH(1,1) model.

    Attributes:
        omega: Constant term in log-variance equation
        alpha: ARCH parameter
        gamma: Asymmetry parameter
        beta: GARCH parameter (must be between -1 and 1 for stationarity)
    """

    omega: float
    alpha: float
    gamma: float
    beta: float

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate EGARCH parameter constraints.

        Raises:
            ParameterError: If parameter constraints are violated
        """
        super().validate()

        # EGARCH has fewer constraints than GARCH
        # The key constraint is |beta| < 1 for stationarity
        if abs(self.beta) >= 1:
            raise ParameterError(
                f"EGARCH stationarity constraint violated: |beta| = {abs(self.beta)} >= 1"
            )

    def transform(self) -> np.ndarray:
        """Transform parameters to unconstrained space for optimization.

        Returns:
            np.ndarray: Transformed parameters in unconstrained space
        """
        # omega, alpha, and gamma have no constraints, so they remain unchanged
        transformed_omega = self.omega
        transformed_alpha = self.alpha
        transformed_gamma = self.gamma

        # Transform beta to ensure |beta| < 1
        transformed_beta = transform_correlation(self.beta)

        return np.array([
            transformed_omega,
            transformed_alpha,
            transformed_gamma,
            transformed_beta
        ])

    @classmethod
    def inverse_transform(cls, array: np.ndarray, **kwargs: Any) -> 'EGARCHParameters':
        """Transform parameters from unconstrained space back to constrained space.

        Args:
            array: Parameters in unconstrained space [omega, alpha, gamma, beta*]
            **kwargs: Additional keyword arguments for parameter creation

        Returns:
            EGARCHParameters: Parameter object with constrained parameters

        Raises:
            ValueError: If the array length is not 4
        """
        if len(array) != 4:
            raise ValueError(f"Array length must be 4, got {len(array)}")

        # Extract transformed parameters
        transformed_omega, transformed_alpha, transformed_gamma, transformed_beta = array

        # omega, alpha, and gamma have no constraints
        omega = transformed_omega
        alpha = transformed_alpha
        gamma = transformed_gamma

        # Inverse transform beta
        beta = inverse_transform_correlation(transformed_beta)

        return cls(omega=omega, alpha=alpha, gamma=gamma, beta=beta)


@dataclass
class TARCHParameters(UnivariateVolatilityParameters):
    """Parameters for TARCH(1,1) model.

    Attributes:
        omega: Constant term in variance equation (must be positive)
        alpha: ARCH parameter (must be non-negative)
        gamma: Asymmetry parameter (must be non-negative)
        beta: GARCH parameter (must be non-negative)
    """

    omega: float
    alpha: float
    gamma: float
    beta: float

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate TARCH parameter constraints.

        Raises:
            ParameterError: If parameter constraints are violated
        """
        super().validate()

        # Validate individual parameters
        validate_positive(self.omega, "omega")
        validate_non_negative(self.alpha, "alpha")
        validate_non_negative(self.gamma, "gamma")
        validate_non_negative(self.beta, "beta")

        # Validate stationarity constraint
        # For TARCH, the constraint is alpha + beta + 0.5*gamma < 1
        if self.alpha + self.beta + 0.5 * self.gamma >= 1:
            raise ParameterError(
                f"TARCH stationarity constraint violated: "
                f"alpha + beta + 0.5*gamma = {self.alpha + self.beta + 0.5 * self.gamma} >= 1"
            )

    def transform(self) -> np.ndarray:
        """Transform parameters to unconstrained space for optimization.

        Returns:
            np.ndarray: Transformed parameters in unconstrained space
        """
        # Transform omega to unconstrained space (log)
        transformed_omega = transform_positive(self.omega)

        # Transform alpha, gamma, and beta to unconstrained space
        # We use a special transformation to ensure alpha + beta + 0.5*gamma < 1
        sum_params = self.alpha + self.beta + 0.5 * self.gamma
        if sum_params >= 1:
            # If constraint is violated, adjust parameters slightly
            factor = 0.99 / sum_params
            self.alpha *= factor
            self.beta *= factor
            self.gamma *= factor

        # Use a transformation that preserves the constraint
        # We parameterize in terms of:
        # lambda = alpha + beta + 0.5*gamma (must be < 1)
        # delta1 = alpha / lambda
        # delta2 = gamma / (2 * lambda)
        # This ensures beta = lambda * (1 - delta1 - delta2)

        lambda_param = self.alpha + self.beta + 0.5 * self.gamma
        delta1 = self.alpha / lambda_param if lambda_param > 0 else 0.33
        delta2 = 0.5 * self.gamma / lambda_param if lambda_param > 0 else 0.33

        transformed_lambda = transform_probability(lambda_param)
        transformed_delta1 = transform_probability(delta1)
        transformed_delta2 = transform_probability(delta2)

        return np.array([
            transformed_omega,
            transformed_lambda,
            transformed_delta1,
            transformed_delta2
        ])

    @classmethod
    def inverse_transform(cls, array: np.ndarray, **kwargs: Any) -> 'TARCHParameters':
        """Transform parameters from unconstrained space back to constrained space.

        Args:
            array: Parameters in unconstrained space [omega*, lambda*, delta1*, delta2*]
            **kwargs: Additional keyword arguments for parameter creation

        Returns:
            TARCHParameters: Parameter object with constrained parameters

        Raises:
            ValueError: If the array length is not 4
        """
        if len(array) != 4:
            raise ValueError(f"Array length must be 4, got {len(array)}")

        # Extract transformed parameters
        transformed_omega, transformed_lambda, transformed_delta1, transformed_delta2 = array

        # Inverse transform omega
        omega = inverse_transform_positive(transformed_omega)

        # Inverse transform lambda, delta1, and delta2
        lambda_param = inverse_transform_probability(transformed_lambda)
        delta1 = inverse_transform_probability(transformed_delta1)
        delta2 = inverse_transform_probability(transformed_delta2)

        # Ensure delta1 + delta2 <= 1 (for numerical stability)
        if delta1 + delta2 > 1:
            sum_deltas = delta1 + delta2
            delta1 = delta1 / sum_deltas * 0.99
            delta2 = delta2 / sum_deltas * 0.99

        # Compute alpha, gamma, and beta
        alpha = lambda_param * delta1
        gamma = lambda_param * delta2 * 2
        beta = lambda_param * (1 - delta1 - delta2)

        return cls(omega=omega, alpha=alpha, gamma=gamma, beta=beta)


@dataclass
class APARCHParameters(UnivariateVolatilityParameters):
    """Parameters for APARCH(1,1) model.

    Attributes:
        omega: Constant term in power variance equation (must be positive)
        alpha: ARCH parameter (must be non-negative)
        gamma: Asymmetry parameter (must be between -1 and 1)
        beta: GARCH parameter (must be non-negative)
        delta: Power parameter (must be positive)
    """

    omega: float
    alpha: float
    gamma: float
    beta: float
    delta: float

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate APARCH parameter constraints.

        Raises:
            ParameterError: If parameter constraints are violated
        """
        super().validate()

        # Validate individual parameters
        validate_positive(self.omega, "omega")
        validate_non_negative(self.alpha, "alpha")
        validate_range(self.gamma, "gamma", -1, 1)
        validate_non_negative(self.beta, "beta")
        validate_positive(self.delta, "delta")

        # Validate stationarity constraint
        if self.alpha + self.beta >= 1:
            raise ParameterError(
                f"APARCH stationarity constraint violated: alpha + beta = {self.alpha + self.beta} >= 1"
            )

    def transform(self) -> np.ndarray:
        """Transform parameters to unconstrained space for optimization.

        Returns:
            np.ndarray: Transformed parameters in unconstrained space
        """
        # Transform omega and delta to unconstrained space (log)
        transformed_omega = transform_positive(self.omega)
        transformed_delta = transform_positive(self.delta)

        # Transform gamma to unconstrained space (arctanh)
        transformed_gamma = transform_correlation(self.gamma)

        # Transform alpha and beta to unconstrained space
        # Similar to GARCH transformation
        if self.alpha + self.beta >= 1:
            # If constraint is violated, adjust parameters slightly
            sum_ab = self.alpha + self.beta
            self.alpha = self.alpha / (sum_ab + 0.01)
            self.beta = self.beta / (sum_ab + 0.01)

        # Use logit-like transformation for alpha and beta
        lambda_param = self.alpha + self.beta
        delta_param = self.alpha / lambda_param if lambda_param > 0 else 0.5

        transformed_lambda = transform_probability(lambda_param)
        transformed_delta_param = transform_probability(delta_param)

        return np.array([
            transformed_omega,
            transformed_lambda,
            transformed_delta_param,
            transformed_gamma,
            transformed_delta
        ])

    @classmethod
    def inverse_transform(cls, array: np.ndarray, **kwargs: Any) -> 'APARCHParameters':
        """Transform parameters from unconstrained space back to constrained space.

        Args:
            array: Parameters in unconstrained space
            **kwargs: Additional keyword arguments for parameter creation

        Returns:
            APARCHParameters: Parameter object with constrained parameters

        Raises:
            ValueError: If the array length is not 5
        """
        if len(array) != 5:
            raise ValueError(f"Array length must be 5, got {len(array)}")

        # Extract transformed parameters
        (transformed_omega, transformed_lambda, transformed_delta_param,
         transformed_gamma, transformed_delta) = array

        # Inverse transform omega and delta
        omega = inverse_transform_positive(transformed_omega)
        delta = inverse_transform_positive(transformed_delta)

        # Inverse transform gamma
        gamma = inverse_transform_correlation(transformed_gamma)

        # Inverse transform lambda and delta_param
        lambda_param = inverse_transform_probability(transformed_lambda)
        delta_param = inverse_transform_probability(transformed_delta_param)

        # Compute alpha and beta
        alpha = lambda_param * delta_param
        beta = lambda_param * (1 - delta_param)

        return cls(omega=omega, alpha=alpha, gamma=gamma, beta=beta, delta=delta)


@dataclass
class MultivariateVolatilityParameters(ParameterBase):
    """Base class for multivariate volatility model parameters.

    This class provides common functionality for multivariate volatility model
    parameters, including validation and transformation methods.
    """

    def validate(self) -> None:
        """Validate parameter constraints for multivariate volatility models.

        Raises:
            ParameterError: If parameter constraints are violated
        """
        # Base class doesn't enforce specific constraints
        # Subclasses should implement model-specific constraints
        pass


@dataclass
class DCCParameters(MultivariateVolatilityParameters):
    """Parameters for DCC(1,1) model.

    Attributes:
        a: News parameter (must be non-negative)
        b: Decay parameter (must be non-negative)
    """

    a: float
    b: float

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate DCC parameter constraints.

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
                f"DCC stationarity constraint violated: a + b = {self.a + self.b} >= 1"
            )

    def to_array(self) -> np.ndarray:
        """Convert parameters to a NumPy array.

        Returns:
            np.ndarray: Array representation of parameters
        """
        return np.array([self.a, self.b])

    @classmethod
    def from_array(cls, array: np.ndarray, **kwargs: Any) -> 'DCCParameters':
        """Create parameters from a NumPy array.

        Args:
            array: Array representation of parameters
            **kwargs: Additional keyword arguments for parameter creation

        Returns:
            DCCParameters: Parameter object

        Raises:
            ValueError: If the array length is not 2
        """
        if len(array) != 2:
            raise ValueError(f"Array length must be 2, got {len(array)}")

        return cls(a=array[0], b=array[1])

    def transform(self) -> np.ndarray:
        """Transform parameters to unconstrained space for optimization.

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
    def inverse_transform(cls, array: np.ndarray, **kwargs: Any) -> 'DCCParameters':
        """Transform parameters from unconstrained space back to constrained space.

        Args:
            array: Parameters in unconstrained space [lambda*, delta*]
            **kwargs: Additional keyword arguments for parameter creation

        Returns:
            DCCParameters: Parameter object with constrained parameters

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


@dataclass
class TimeSeriesParameters(ParameterBase):
    """Base class for time series model parameters.

    This class provides common functionality for time series model
    parameters, including validation and transformation methods.
    """

    def validate(self) -> None:
        """Validate parameter constraints for time series models.

        Raises:
            ParameterError: If parameter constraints are violated
        """
        # Base class doesn't enforce specific constraints
        # Subclasses should implement model-specific constraints
        pass


@dataclass
class ARMAParameters(TimeSeriesParameters):
    """Parameters for ARMA(p,q) model.

    Attributes:
        ar_params: Autoregressive parameters
        ma_params: Moving average parameters
        sigma2: Innovation variance (must be positive)
        constant: Constant term
    """

    ar_params: np.ndarray
    ma_params: np.ndarray
    sigma2: float
    constant: float = 0.0

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        # Ensure ar_params and ma_params are NumPy arrays
        if not isinstance(self.ar_params, np.ndarray):
            self.ar_params = np.array(self.ar_params)
        if not isinstance(self.ma_params, np.ndarray):
            self.ma_params = np.array(self.ma_params)

        self.validate()

    def validate(self) -> None:
        """Validate ARMA parameter constraints.

        Raises:
            ParameterError: If parameter constraints are violated
        """
        super().validate()

        # Validate sigma2
        validate_positive(self.sigma2, "sigma2")

        # Check stationarity of AR parameters
        if len(self.ar_params) > 0:
            # Create companion matrix for AR polynomial
            if np.any(np.abs(self.ar_params) > 0):
                companion = np.zeros((len(self.ar_params), len(self.ar_params)))
                companion[0, :] = self.ar_params
                companion[1:, :-1] = np.eye(len(self.ar_params) - 1)

                # Check eigenvalues
                eigenvalues = np.linalg.eigvals(companion)
                if np.any(np.abs(eigenvalues) >= 1):
                    warnings.warn(
                        "AR parameters may not satisfy stationarity condition",
                        UserWarning
                    )

        # Check invertibility of MA parameters
        if len(self.ma_params) > 0:
            # Create companion matrix for MA polynomial
            if np.any(np.abs(self.ma_params) > 0):
                companion = np.zeros((len(self.ma_params), len(self.ma_params)))
                companion[0, :] = self.ma_params
                companion[1:, :-1] = np.eye(len(self.ma_params) - 1)

                # Check eigenvalues
                eigenvalues = np.linalg.eigvals(companion)
                if np.any(np.abs(eigenvalues) >= 1):
                    warnings.warn(
                        "MA parameters may not satisfy invertibility condition",
                        UserWarning
                    )

    def to_array(self) -> np.ndarray:
        """Convert parameters to a NumPy array.

        Returns:
            np.ndarray: Array representation of parameters
        """
        return np.concatenate([
            [self.constant],
            self.ar_params,
            self.ma_params,
            [self.sigma2]
        ])

    @classmethod
    def from_array(cls, array: np.ndarray, ar_order: int = 0, ma_order: int = 0,
                   **kwargs: Any) -> 'ARMAParameters':
        """Create parameters from a NumPy array.

        Args:
            array: Array representation of parameters
            ar_order: Order of the AR component
            ma_order: Order of the MA component
            **kwargs: Additional keyword arguments for parameter creation

        Returns:
            ARMAParameters: Parameter object

        Raises:
            ValueError: If the array length doesn't match the expected length
        """
        expected_length = 1 + ar_order + ma_order + 1  # constant, AR, MA, sigma2
        if len(array) != expected_length:
            raise ValueError(
                f"Array length ({len(array)}) doesn't match expected length ({expected_length})"
            )

        # Extract parameters
        constant = array[0]
        ar_params = array[1:1+ar_order] if ar_order > 0 else np.array([])
        ma_params = array[1+ar_order:1+ar_order+ma_order] if ma_order > 0 else np.array([])
        sigma2 = array[-1]

        return cls(
            ar_params=ar_params,
            ma_params=ma_params,
            sigma2=sigma2,
            constant=constant
        )

    def transform(self) -> np.ndarray:
        """Transform parameters to unconstrained space for optimization.

        Returns:
            np.ndarray: Transformed parameters in unconstrained space
        """
        # Transform sigma2 to unconstrained space (log)
        transformed_sigma2 = transform_positive(self.sigma2)

        # AR and MA parameters don't have simple constraints
        # We could use a more complex transformation to ensure stationarity/invertibility,
        # but for simplicity, we'll leave them untransformed
        transformed_params = np.concatenate([
            [self.constant],
            self.ar_params,
            self.ma_params,
            [transformed_sigma2]
        ])

        return transformed_params

    @classmethod
    def inverse_transform(cls, array: np.ndarray, ar_order: int = 0, ma_order: int = 0,
                          **kwargs: Any) -> 'ARMAParameters':
        """Transform parameters from unconstrained space back to constrained space.

        Args:
            array: Parameters in unconstrained space
            ar_order: Order of the AR component
            ma_order: Order of the MA component
            **kwargs: Additional keyword arguments for parameter creation

        Returns:
            ARMAParameters: Parameter object with constrained parameters

        Raises:
            ValueError: If the array length doesn't match the expected length
        """
        expected_length = 1 + ar_order + ma_order + 1  # constant, AR, MA, sigma2
        if len(array) != expected_length:
            raise ValueError(
                f"Array length ({len(array)}) doesn't match expected length ({expected_length})"
            )

        # Extract parameters
        constant = array[0]
        ar_params = array[1:1+ar_order] if ar_order > 0 else np.array([])
        ma_params = array[1+ar_order:1+ar_order+ma_order] if ma_order > 0 else np.array([])
        transformed_sigma2 = array[-1]

        # Inverse transform sigma2
        sigma2 = inverse_transform_positive(transformed_sigma2)

        return cls(
            ar_params=ar_params,
            ma_params=ma_params,
            sigma2=sigma2,
            constant=constant
        )


@dataclass
class DistributionParameters(ParameterBase):
    """Base class for distribution parameters.

    This class provides common functionality for distribution parameters,
    including validation and transformation methods.
    """

    def validate(self) -> None:
        """Validate parameter constraints for distributions.

        Raises:
            ParameterError: If parameter constraints are violated
        """
        # Base class doesn't enforce specific constraints
        # Subclasses should implement distribution-specific constraints
        pass


@dataclass
class StudentTParameters(DistributionParameters):
    """Parameters for Student's t distribution.

    Attributes:
        df: Degrees of freedom (must be greater than 2)
    """

    df: float

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate Student's t parameter constraints.

        Raises:
            ParameterError: If parameter constraints are violated
        """
        super().validate()

        # Validate degrees of freedom
        validate_degrees_of_freedom(self.df, "df")

    def to_array(self) -> np.ndarray:
        """Convert parameters to a NumPy array.

        Returns:
            np.ndarray: Array representation of parameters
        """
        return np.array([self.df])

    @classmethod
    def from_array(cls, array: np.ndarray, **kwargs: Any) -> 'StudentTParameters':
        """Create parameters from a NumPy array.

        Args:
            array: Array representation of parameters
            **kwargs: Additional keyword arguments for parameter creation

        Returns:
            StudentTParameters: Parameter object

        Raises:
            ValueError: If the array length is not 1
        """
        if len(array) != 1:
            raise ValueError(f"Array length must be 1, got {len(array)}")

        return cls(df=array[0])

    def transform(self) -> np.ndarray:
        """Transform parameters to unconstrained space for optimization.

        Returns:
            np.ndarray: Transformed parameters in unconstrained space
        """
        # Transform df to unconstrained space
        # We use a transformation that ensures df > 2
        transformed_df = np.log(self.df - 2)

        return np.array([transformed_df])

    @classmethod
    def inverse_transform(cls, array: np.ndarray, **kwargs: Any) -> 'StudentTParameters':
        """Transform parameters from unconstrained space back to constrained space.

        Args:
            array: Parameters in unconstrained space
            **kwargs: Additional keyword arguments for parameter creation

        Returns:
            StudentTParameters: Parameter object with constrained parameters

        Raises:
            ValueError: If the array length is not 1
        """
        if len(array) != 1:
            raise ValueError(f"Array length must be 1, got {len(array)}")

        # Inverse transform df
        df = np.exp(array[0]) + 2

        return cls(df=df)


@dataclass
class SkewedTParameters(DistributionParameters):
    """Parameters for Hansen's skewed t distribution.

    Attributes:
        df: Degrees of freedom (must be greater than 2)
        lambda_: Skewness parameter (must be between -1 and 1)
    """

    df: float
    lambda_: float

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate skewed t parameter constraints.

        Raises:
            ParameterError: If parameter constraints are violated
        """
        super().validate()

        # Validate degrees of freedom
        validate_degrees_of_freedom(self.df, "df")

        # Validate skewness parameter
        validate_range(self.lambda_, "lambda_", -1, 1)

    def to_array(self) -> np.ndarray:
        """Convert parameters to a NumPy array.

        Returns:
            np.ndarray: Array representation of parameters
        """
        return np.array([self.df, self.lambda_])

    @classmethod
    def from_array(cls, array: np.ndarray, **kwargs: Any) -> 'SkewedTParameters':
        """Create parameters from a NumPy array.

        Args:
            array: Array representation of parameters
            **kwargs: Additional keyword arguments for parameter creation

        Returns:
            SkewedTParameters: Parameter object

        Raises:
            ValueError: If the array length is not 2
        """
        if len(array) != 2:
            raise ValueError(f"Array length must be 2, got {len(array)}")

        return cls(df=array[0], lambda_=array[1])

    def transform(self) -> np.ndarray:
        """Transform parameters to unconstrained space for optimization.

        Returns:
            np.ndarray: Transformed parameters in unconstrained space
        """
        # Transform df to unconstrained space
        transformed_df = np.log(self.df - 2)

        # Transform lambda to unconstrained space
        transformed_lambda = transform_correlation(self.lambda_)

        return np.array([transformed_df, transformed_lambda])

    @classmethod
    def inverse_transform(cls, array: np.ndarray, **kwargs: Any) -> 'SkewedTParameters':
        """Transform parameters from unconstrained space back to constrained space.

        Args:
            array: Parameters in unconstrained space
            **kwargs: Additional keyword arguments for parameter creation

        Returns:
            SkewedTParameters: Parameter object with constrained parameters

        Raises:
            ValueError: If the array length is not 2
        """
        if len(array) != 2:
            raise ValueError(f"Array length must be 2, got {len(array)}")

        # Inverse transform df
        df = np.exp(array[0]) + 2

        # Inverse transform lambda
        lambda_ = inverse_transform_correlation(array[1])

        return cls(df=df, lambda_=lambda_)


@dataclass
class GEDParameters(DistributionParameters):
    """Parameters for Generalized Error Distribution (GED).

    Attributes:
        nu: Shape parameter (must be positive)
    """

    nu: float

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate GED parameter constraints.

        Raises:
            ParameterError: If parameter constraints are violated
        """
        super().validate()

        # Validate shape parameter
        validate_positive(self.nu, "nu")

    def to_array(self) -> np.ndarray:
        """Convert parameters to a NumPy array.

        Returns:
            np.ndarray: Array representation of parameters
        """
        return np.array([self.nu])

    @classmethod
    def from_array(cls, array: np.ndarray, **kwargs: Any) -> 'GEDParameters':
        """Create parameters from a NumPy array.

        Args:
            array: Array representation of parameters
            **kwargs: Additional keyword arguments for parameter creation

        Returns:
            GEDParameters: Parameter object

        Raises:
            ValueError: If the array length is not 1
        """
        if len(array) != 1:
            raise ValueError(f"Array length must be 1, got {len(array)}")

        return cls(nu=array[0])

    def transform(self) -> np.ndarray:
        """Transform parameters to unconstrained space for optimization.

        Returns:
            np.ndarray: Transformed parameters in unconstrained space
        """
        # Transform nu to unconstrained space
        transformed_nu = transform_positive(self.nu)

        return np.array([transformed_nu])

    @classmethod
    def inverse_transform(cls, array: np.ndarray, **kwargs: Any) -> 'GEDParameters':
        """Transform parameters from unconstrained space back to constrained space.

        Args:
            array: Parameters in unconstrained space
            **kwargs: Additional keyword arguments for parameter creation

        Returns:
            GEDParameters: Parameter object with constrained parameters

        Raises:
            ValueError: If the array length is not 1
        """
        if len(array) != 1:
            raise ValueError(f"Array length must be 1, got {len(array)}")

        # Inverse transform nu
        nu = inverse_transform_positive(array[0])

        return cls(nu=nu)


# Helper functions for parameter creation and manipulation

def create_parameter_object(param_class: Type[P], *args: Any, **kwargs: Any) -> P:
    """Create a parameter object of the specified class.

    Args:
        param_class: Parameter class to create
        *args: Positional arguments for parameter creation
        **kwargs: Keyword arguments for parameter creation

    Returns:
        P: Parameter object

    Raises:
        TypeError: If param_class is not a subclass of ParameterBase
    """
    if not issubclass(param_class, ParameterBase):
        raise TypeError(f"param_class must be a subclass of ParameterBase, got {param_class}")

    return param_class(*args, **kwargs)


def from_dict(param_class: Type[P], param_dict: Dict[str, Any]) -> P:
    """Create a parameter object from a dictionary.

    Args:
        param_class: Parameter class to create
        param_dict: Dictionary containing parameter values

    Returns:
        P: Parameter object

    Raises:
        TypeError: If param_class is not a subclass of ParameterBase
    """
    if not issubclass(param_class, ParameterBase):
        raise TypeError(f"param_class must be a subclass of ParameterBase, got {param_class}")

    return param_class(**param_dict)


def from_dataclass(dataclass_obj: Any) -> ParameterBase:
    """Convert a dataclass object to a parameter object.

    Args:
        dataclass_obj: Dataclass object to convert

    Returns:
        ParameterBase: Parameter object

    Raises:
        TypeError: If dataclass_obj is not a dataclass
    """
    if not is_dataclass(dataclass_obj):
        raise TypeError(f"dataclass_obj must be a dataclass, got {type(dataclass_obj)}")

    # Get the dataclass fields
    param_dict = asdict(dataclass_obj)

    # Find the appropriate parameter class
    class_name = dataclass_obj.__class__.__name__

    # Try to find a matching parameter class in this module
    import sys
    current_module = sys.modules[__name__]

    for name in dir(current_module):
        obj = getattr(current_module, name)
        if (isinstance(obj, type) and issubclass(obj, ParameterBase) and
                obj.__name__ == class_name):
            return obj(**param_dict)

    # If no matching class is found, create a generic parameter object
    return from_dict(ParameterBase, param_dict)


# Import dataclasses.fields at the end to avoid circular imports
