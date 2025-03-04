# mfe/core/validation.py

"""
Validation utilities and decorators for the MFE Toolbox.

This module provides a comprehensive set of validation functions and decorators
for enforcing constraints throughout the MFE Toolbox. It implements validation
utilities for common requirements such as dimensional compatibility, matrix
properties, parameter bounds, and data quality.

The validation components ensure consistent validation behavior across the
toolbox and reduce duplicate validation code. They leverage Python's decorator
pattern and type system to provide clear, informative error messages when
validation fails.
"""

import functools
import inspect
import warnings
from dataclasses import is_dataclass
from typing import (
    Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, TypeVar, Union,
    cast, overload, get_type_hints
)

import numpy as np
import pandas as pd
from scipy import linalg

from mfe.core.exceptions import (
    DimensionError, ParameterError, NumericError, DataError,
    raise_dimension_error, raise_parameter_error, raise_numeric_error,
    raise_data_error, warn_numeric
)
from mfe.core.types import (
    Vector, Matrix, TimeSeriesData, TimeSeriesDataFrame, ParameterVector,
    CovarianceMatrix, CorrelationMatrix
)

# Type variables for generic validation functions
T = TypeVar('T')  # Generic type
F = TypeVar('F', bound=Callable[..., Any])  # Function type


def validate_array_shape(
    array: np.ndarray,
    expected_shape: Union[Tuple[int, ...], List[Tuple[int, ...]]],
    array_name: str = "array",
    allow_none: bool = False
) -> np.ndarray:
    """Validate that an array has the expected shape.

    Args:
        array: Array to validate
        expected_shape: Expected shape or list of valid shapes
        array_name: Name of the array for error messages
        allow_none: Whether to allow None as a valid input

    Returns:
        np.ndarray: The validated array

    Raises:
        TypeError: If array is not a NumPy array
        DimensionError: If array shape doesn't match expected shape
    """
    if array is None:
        if allow_none:
            return None
        raise TypeError(f"{array_name} cannot be None")

    if not isinstance(array, np.ndarray):
        raise TypeError(f"{array_name} must be a NumPy array, got {type(array).__name__}")

    if isinstance(expected_shape, tuple):
        expected_shapes = [expected_shape]
    else:
        expected_shapes = expected_shape

    # Check if any of the expected shapes match
    shape_matches = False
    for shape in expected_shapes:
        if len(shape) != array.ndim:
            continue

        match = True
        for i, dim in enumerate(shape):
            if dim != -1 and dim != array.shape[i]:
                match = False
                break

        if match:
            shape_matches = True
            break

    if not shape_matches:
        shapes_str = " or ".join(str(s) for s in expected_shapes)
        raise_dimension_error(
            f"{array_name} has invalid shape {array.shape}, expected {shapes_str}",
            array_name=array_name,
            expected_shape=shapes_str,
            actual_shape=array.shape
        )

    return array


def validate_matrix_shape(
    matrix: np.ndarray,
    expected_rows: Optional[int] = None,
    expected_cols: Optional[int] = None,
    matrix_name: str = "matrix",
    allow_none: bool = False
) -> np.ndarray:
    """Validate that a matrix has the expected dimensions.

    Args:
        matrix: Matrix to validate
        expected_rows: Expected number of rows, or None for any
        expected_cols: Expected number of columns, or None for any
        matrix_name: Name of the matrix for error messages
        allow_none: Whether to allow None as a valid input

    Returns:
        np.ndarray: The validated matrix

    Raises:
        TypeError: If matrix is not a NumPy array
        DimensionError: If matrix dimensions don't match expected dimensions
    """
    if matrix is None:
        if allow_none:
            return None
        raise TypeError(f"{matrix_name} cannot be None")

    if not isinstance(matrix, np.ndarray):
        raise TypeError(f"{matrix_name} must be a NumPy array, got {type(matrix).__name__}")

    if matrix.ndim != 2:
        raise_dimension_error(
            f"{matrix_name} must be 2-dimensional, got {matrix.ndim} dimensions",
            array_name=matrix_name,
            expected_shape="2D matrix",
            actual_shape=matrix.shape
        )

    if expected_rows is not None and matrix.shape[0] != expected_rows:
        raise_dimension_error(
            f"{matrix_name} has {matrix.shape[0]} rows, expected {expected_rows}",
            array_name=matrix_name,
            expected_shape=f"({expected_rows}, {expected_cols if expected_cols else 'any'})",
            actual_shape=matrix.shape
        )

    if expected_cols is not None and matrix.shape[1] != expected_cols:
        raise_dimension_error(
            f"{matrix_name} has {matrix.shape[1]} columns, expected {expected_cols}",
            array_name=matrix_name,
            expected_shape=f"({expected_rows if expected_rows else 'any'}, {expected_cols})",
            actual_shape=matrix.shape
        )

    return matrix


def validate_square_matrix(
    matrix: np.ndarray,
    matrix_name: str = "matrix",
    allow_none: bool = False
) -> np.ndarray:
    """Validate that a matrix is square.

    Args:
        matrix: Matrix to validate
        matrix_name: Name of the matrix for error messages
        allow_none: Whether to allow None as a valid input

    Returns:
        np.ndarray: The validated matrix

    Raises:
        TypeError: If matrix is not a NumPy array
        DimensionError: If matrix is not square
    """
    if matrix is None:
        if allow_none:
            return None
        raise TypeError(f"{matrix_name} cannot be None")

    if not isinstance(matrix, np.ndarray):
        raise TypeError(f"{matrix_name} must be a NumPy array, got {type(matrix).__name__}")

    if matrix.ndim != 2:
        raise_dimension_error(
            f"{matrix_name} must be 2-dimensional, got {matrix.ndim} dimensions",
            array_name=matrix_name,
            expected_shape="square matrix",
            actual_shape=matrix.shape
        )

    if matrix.shape[0] != matrix.shape[1]:
        raise_dimension_error(
            f"{matrix_name} must be square, got shape {matrix.shape}",
            array_name=matrix_name,
            expected_shape=f"({matrix.shape[0]}, {matrix.shape[0]})",
            actual_shape=matrix.shape
        )

    return matrix


def validate_vector(
    vector: np.ndarray,
    expected_length: Optional[int] = None,
    vector_name: str = "vector",
    allow_none: bool = False
) -> np.ndarray:
    """Validate that an array is a vector with the expected length.

    Args:
        vector: Vector to validate
        expected_length: Expected length, or None for any
        vector_name: Name of the vector for error messages
        allow_none: Whether to allow None as a valid input

    Returns:
        np.ndarray: The validated vector

    Raises:
        TypeError: If vector is not a NumPy array
        DimensionError: If vector is not 1-dimensional or has wrong length
    """
    if vector is None:
        if allow_none:
            return None
        raise TypeError(f"{vector_name} cannot be None")

    if not isinstance(vector, np.ndarray):
        raise TypeError(f"{vector_name} must be a NumPy array, got {type(vector).__name__}")

    # Handle both 1D arrays and column/row vectors
    if vector.ndim == 2:
        if vector.shape[0] == 1 or vector.shape[1] == 1:
            # Convert to 1D array for consistency
            vector = vector.ravel()
        else:
            raise_dimension_error(
                f"{vector_name} must be 1-dimensional or a column/row vector, got shape {vector.shape}",
                array_name=vector_name,
                expected_shape="1D vector",
                actual_shape=vector.shape
            )
    elif vector.ndim != 1:
        raise_dimension_error(
            f"{vector_name} must be 1-dimensional, got {vector.ndim} dimensions",
            array_name=vector_name,
            expected_shape="1D vector",
            actual_shape=vector.shape
        )

    if expected_length is not None and len(vector) != expected_length:
        raise_dimension_error(
            f"{vector_name} has length {len(vector)}, expected {expected_length}",
            array_name=vector_name,
            expected_shape=f"vector of length {expected_length}",
            actual_shape=vector.shape
        )

    return vector


def validate_time_series(
    data: Union[np.ndarray, pd.Series, pd.DataFrame],
    min_length: int = 2,
    data_name: str = "data",
    allow_none: bool = False
) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
    """Validate that data is a valid time series.

    Args:
        data: Time series data to validate
        min_length: Minimum required length
        data_name: Name of the data for error messages
        allow_none: Whether to allow None as a valid input

    Returns:
        Union[np.ndarray, pd.Series, pd.DataFrame]: The validated time series

    Raises:
        TypeError: If data is not a valid time series type
        DataError: If data is too short or contains invalid values
    """
    if data is None:
        if allow_none:
            return None
        raise TypeError(f"{data_name} cannot be None")

    if not isinstance(data, (np.ndarray, pd.Series, pd.DataFrame)):
        raise TypeError(
            f"{data_name} must be a NumPy array, Pandas Series, or DataFrame, "
            f"got {type(data).__name__}"
        )

    # Get data length
    if isinstance(data, np.ndarray):
        length = data.shape[0]
    else:  # pd.Series or pd.DataFrame
        length = len(data)

    if length < min_length:
        raise_data_error(
            f"{data_name} is too short (length {length}), minimum required length is {min_length}",
            data_name=data_name,
            issue=f"insufficient length: {length} < {min_length}"
        )

    # Check for NaN/Inf values
    if isinstance(data, np.ndarray):
        if np.isnan(data).any():
            raise_data_error(
                f"{data_name} contains NaN values",
                data_name=data_name,
                issue="contains NaN values"
            )
        if np.isinf(data).any():
            raise_data_error(
                f"{data_name} contains infinite values",
                data_name=data_name,
                issue="contains infinite values"
            )
    else:  # pd.Series or pd.DataFrame
        if data.isna().any().any():
            raise_data_error(
                f"{data_name} contains NaN values",
                data_name=data_name,
                issue="contains NaN values"
            )
        if np.isinf(data.values).any():
            raise_data_error(
                f"{data_name} contains infinite values",
                data_name=data_name,
                issue="contains infinite values"
            )

    return data


def validate_positive_definite(
    matrix: np.ndarray,
    matrix_name: str = "matrix",
    allow_none: bool = False,
    tol: float = 1e-8
) -> np.ndarray:
    """Validate that a matrix is positive definite.

    Args:
        matrix: Matrix to validate
        matrix_name: Name of the matrix for error messages
        allow_none: Whether to allow None as a valid input
        tol: Tolerance for eigenvalue positivity check

    Returns:
        np.ndarray: The validated matrix

    Raises:
        TypeError: If matrix is not a NumPy array
        DimensionError: If matrix is not square
        NumericError: If matrix is not positive definite
    """
    if matrix is None:
        if allow_none:
            return None
        raise TypeError(f"{matrix_name} cannot be None")

    # Validate that matrix is square
    matrix = validate_square_matrix(matrix, matrix_name)

    try:
        # Attempt Cholesky decomposition, which only works for positive definite matrices
        linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        # If Cholesky fails, check eigenvalues to provide more informative error
        eigenvalues = linalg.eigvalsh(matrix)
        min_eig = np.min(eigenvalues)

        if min_eig < -tol:
            raise_numeric_error(
                f"{matrix_name} is not positive definite (min eigenvalue: {min_eig})",
                operation="positive definite check",
                error_type="negative eigenvalue",
                values=min_eig
            )
        elif min_eig < tol:
            # Matrix is positive semi-definite (has zero eigenvalues)
            warn_numeric(
                f"{matrix_name} is nearly singular (min eigenvalue: {min_eig})",
                operation="positive definite check",
                issue="near-zero eigenvalue",
                value=min_eig
            )
            # Add a small regularization to make it positive definite
            n = matrix.shape[0]
            matrix = matrix + np.eye(n) * tol

    return matrix


def validate_correlation_matrix(
    matrix: np.ndarray,
    matrix_name: str = "correlation matrix",
    allow_none: bool = False,
    tol: float = 1e-8
) -> np.ndarray:
    """Validate that a matrix is a valid correlation matrix.

    Args:
        matrix: Matrix to validate
        matrix_name: Name of the matrix for error messages
        allow_none: Whether to allow None as a valid input
        tol: Tolerance for validation checks

    Returns:
        np.ndarray: The validated correlation matrix

    Raises:
        TypeError: If matrix is not a NumPy array
        DimensionError: If matrix is not square
        NumericError: If matrix is not a valid correlation matrix
    """
    if matrix is None:
        if allow_none:
            return None
        raise TypeError(f"{matrix_name} cannot be None")

    # Validate that matrix is square and positive definite
    matrix = validate_positive_definite(matrix, matrix_name, tol=tol)

    # Check diagonal elements are 1
    diag = np.diag(matrix)
    if not np.allclose(diag, 1.0, rtol=tol, atol=tol):
        raise_numeric_error(
            f"{matrix_name} diagonal elements must be 1.0",
            operation="correlation matrix check",
            error_type="invalid diagonal",
            values=diag
        )

    # Check off-diagonal elements are between -1 and 1
    if np.any(matrix < -1.0 - tol) or np.any(matrix > 1.0 + tol):
        raise_numeric_error(
            f"{matrix_name} elements must be between -1 and 1",
            operation="correlation matrix check",
            error_type="out-of-range values",
            values=matrix
        )

    # Check symmetry
    if not np.allclose(matrix, matrix.T, rtol=tol, atol=tol):
        raise_numeric_error(
            f"{matrix_name} must be symmetric",
            operation="correlation matrix check",
            error_type="asymmetric matrix",
            values=matrix - matrix.T
        )

    return matrix


def validate_parameter_bounds(
    value: float,
    param_name: str,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
    lower_inclusive: bool = True,
    upper_inclusive: bool = True
) -> float:
    """Validate that a parameter value is within specified bounds.

    Args:
        value: Parameter value to validate
        param_name: Name of the parameter for error messages
        lower_bound: Lower bound, or None for no lower bound
        upper_bound: Upper bound, or None for no upper bound
        lower_inclusive: Whether the lower bound is inclusive
        upper_inclusive: Whether the upper bound is inclusive

    Returns:
        float: The validated parameter value

    Raises:
        ParameterError: If the parameter value is outside the specified bounds
    """
    if lower_bound is not None:
        if lower_inclusive:
            if value < lower_bound:
                raise_parameter_error(
                    f"Parameter {param_name} must be >= {lower_bound}, got {value}",
                    param_name=param_name,
                    param_value=value,
                    constraint=f">= {lower_bound}"
                )
        else:
            if value <= lower_bound:
                raise_parameter_error(
                    f"Parameter {param_name} must be > {lower_bound}, got {value}",
                    param_name=param_name,
                    param_value=value,
                    constraint=f"> {lower_bound}"
                )

    if upper_bound is not None:
        if upper_inclusive:
            if value > upper_bound:
                raise_parameter_error(
                    f"Parameter {param_name} must be <= {upper_bound}, got {value}",
                    param_name=param_name,
                    param_value=value,
                    constraint=f"<= {upper_bound}"
                )
        else:
            if value >= upper_bound:
                raise_parameter_error(
                    f"Parameter {param_name} must be < {upper_bound}, got {value}",
                    param_name=param_name,
                    param_value=value,
                    constraint=f"< {upper_bound}"
                )

    return value


def validate_numeric_array(
    array: np.ndarray,
    array_name: str = "array",
    allow_none: bool = False,
    allow_nan: bool = False,
    allow_inf: bool = False
) -> np.ndarray:
    """Validate that an array contains valid numeric values.

    Args:
        array: Array to validate
        array_name: Name of the array for error messages
        allow_none: Whether to allow None as a valid input
        allow_nan: Whether to allow NaN values
        allow_inf: Whether to allow infinite values

    Returns:
        np.ndarray: The validated array

    Raises:
        TypeError: If array is not a NumPy array
        DataError: If array contains invalid values
    """
    if array is None:
        if allow_none:
            return None
        raise TypeError(f"{array_name} cannot be None")

    if not isinstance(array, np.ndarray):
        raise TypeError(f"{array_name} must be a NumPy array, got {type(array).__name__}")

    if not allow_nan and np.isnan(array).any():
        raise_data_error(
            f"{array_name} contains NaN values",
            data_name=array_name,
            issue="contains NaN values"
        )

    if not allow_inf and np.isinf(array).any():
        raise_data_error(
            f"{array_name} contains infinite values",
            data_name=array_name,
            issue="contains infinite values"
        )

    return array


def validate_compatible_shapes(
    arrays: List[np.ndarray],
    array_names: List[str],
    axis: int = 0
) -> None:
    """Validate that arrays have compatible shapes along a specified axis.

    Args:
        arrays: List of arrays to validate
        array_names: Names of the arrays for error messages
        axis: Axis along which shapes should be compatible

    Raises:
        DimensionError: If arrays have incompatible shapes
    """
    if len(arrays) != len(array_names):
        raise ValueError("arrays and array_names must have the same length")

    if len(arrays) < 2:
        return  # Nothing to compare

    # Get the reference shape
    ref_shape = arrays[0].shape[axis]
    ref_name = array_names[0]

    for i in range(1, len(arrays)):
        if arrays[i].shape[axis] != ref_shape:
            raise_dimension_error(
                f"{array_names[i]} has shape {arrays[i].shape} which is incompatible with "
                f"{ref_name} shape {arrays[0].shape} along axis {axis}",
                array_name=array_names[i],
                expected_shape=f"compatible with {ref_name} (dim {axis} = {ref_shape})",
                actual_shape=arrays[i].shape
            )


def validate_dataclass_fields(obj: Any) -> None:
    """Validate all fields in a dataclass object.

    This function calls the validate method on the dataclass object if it exists,
    or validates individual fields based on type annotations.

    Args:
        obj: Dataclass object to validate

    Raises:
        TypeError: If obj is not a dataclass
        ParameterError: If validation fails
    """
    if not is_dataclass(obj):
        raise TypeError(f"Object must be a dataclass, got {type(obj).__name__}")

    # If the object has a validate method, call it
    if hasattr(obj, "validate") and callable(getattr(obj, "validate")):
        obj.validate()
        return

    # Otherwise, validate fields based on type annotations
    type_hints = get_type_hints(type(obj))

    for field_name, field_type in type_hints.items():
        field_value = getattr(obj, field_name)

        # Skip None values
        if field_value is None:
            continue

        # Validate numeric types
        if field_type in (int, float) and not isinstance(field_value, (int, float)):
            raise_parameter_error(
                f"Field {field_name} must be a {field_type.__name__}, got {type(field_value).__name__}",
                param_name=field_name,
                param_value=field_value
            )

        # Validate array types
        if field_type in (np.ndarray, Vector, Matrix) and not isinstance(field_value, np.ndarray):
            raise_parameter_error(
                f"Field {field_name} must be a NumPy array, got {type(field_value).__name__}",
                param_name=field_name,
                param_value=field_value
            )

        # Validate time series types
        if field_type in (TimeSeriesData, TimeSeriesDataFrame) and not isinstance(field_value, (np.ndarray, pd.Series, pd.DataFrame)):
            raise_parameter_error(
                f"Field {field_name} must be a time series type, got {type(field_value).__name__}",
                param_name=field_name,
                param_value=field_value
            )


# Decorator factories for validation

def validate_input_shape(
    param_index: int,
    expected_shape: Union[Tuple[int, ...], List[Tuple[int, ...]]],
    param_name: Optional[str] = None
) -> Callable[[F], F]:
    """Decorator factory for validating the shape of an array input parameter.

    Args:
        param_index: Index of the parameter to validate
        expected_shape: Expected shape or list of valid shapes
        param_name: Name of the parameter for error messages (defaults to parameter name)

    Returns:
        Callable: Decorator function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get the parameter value
            if param_index < len(args):
                value = args[param_index]
                # Get parameter name if not provided
                if param_name is None:
                    sig = inspect.signature(func)
                    param_names = list(sig.parameters.keys())
                    if param_index < len(param_names):
                        name = param_names[param_index]
                    else:
                        name = f"parameter_{param_index}"
                else:
                    name = param_name
            else:
                # Parameter might be in kwargs
                sig = inspect.signature(func)
                param_names = list(sig.parameters.keys())
                if param_index < len(param_names):
                    name = param_names[param_index]
                    if name in kwargs:
                        value = kwargs[name]
                    else:
                        # Parameter not provided, use default
                        return func(*args, **kwargs)
                else:
                    # Invalid parameter index
                    return func(*args, **kwargs)

            # Validate the parameter
            if isinstance(value, np.ndarray):
                validate_array_shape(value, expected_shape, name)

            return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


def validate_input_matrix(
    param_index: int,
    expected_rows: Optional[int] = None,
    expected_cols: Optional[int] = None,
    param_name: Optional[str] = None
) -> Callable[[F], F]:
    """Decorator factory for validating a matrix input parameter.

    Args:
        param_index: Index of the parameter to validate
        expected_rows: Expected number of rows, or None for any
        expected_cols: Expected number of columns, or None for any
        param_name: Name of the parameter for error messages (defaults to parameter name)

    Returns:
        Callable: Decorator function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get the parameter value
            if param_index < len(args):
                value = args[param_index]
                # Get parameter name if not provided
                if param_name is None:
                    sig = inspect.signature(func)
                    param_names = list(sig.parameters.keys())
                    if param_index < len(param_names):
                        name = param_names[param_index]
                    else:
                        name = f"parameter_{param_index}"
                else:
                    name = param_name
            else:
                # Parameter might be in kwargs
                sig = inspect.signature(func)
                param_names = list(sig.parameters.keys())
                if param_index < len(param_names):
                    name = param_names[param_index]
                    if name in kwargs:
                        value = kwargs[name]
                    else:
                        # Parameter not provided, use default
                        return func(*args, **kwargs)
                else:
                    # Invalid parameter index
                    return func(*args, **kwargs)

            # Validate the parameter
            if isinstance(value, np.ndarray):
                validate_matrix_shape(value, expected_rows, expected_cols, name)

            return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


def validate_input_vector(
    param_index: int,
    expected_length: Optional[int] = None,
    param_name: Optional[str] = None
) -> Callable[[F], F]:
    """Decorator factory for validating a vector input parameter.

    Args:
        param_index: Index of the parameter to validate
        expected_length: Expected length, or None for any
        param_name: Name of the parameter for error messages (defaults to parameter name)

    Returns:
        Callable: Decorator function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get the parameter value
            if param_index < len(args):
                value = args[param_index]
                # Get parameter name if not provided
                if param_name is None:
                    sig = inspect.signature(func)
                    param_names = list(sig.parameters.keys())
                    if param_index < len(param_names):
                        name = param_names[param_index]
                    else:
                        name = f"parameter_{param_index}"
                else:
                    name = param_name
            else:
                # Parameter might be in kwargs
                sig = inspect.signature(func)
                param_names = list(sig.parameters.keys())
                if param_index < len(param_names):
                    name = param_names[param_index]
                    if name in kwargs:
                        value = kwargs[name]
                    else:
                        # Parameter not provided, use default
                        return func(*args, **kwargs)
                else:
                    # Invalid parameter index
                    return func(*args, **kwargs)

            # Validate the parameter
            if isinstance(value, np.ndarray):
                validate_vector(value, expected_length, name)

            return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


def validate_input_time_series(
    param_index: int,
    min_length: int = 2,
    param_name: Optional[str] = None
) -> Callable[[F], F]:
    """Decorator factory for validating a time series input parameter.

    Args:
        param_index: Index of the parameter to validate
        min_length: Minimum required length
        param_name: Name of the parameter for error messages (defaults to parameter name)

    Returns:
        Callable: Decorator function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get the parameter value
            if param_index < len(args):
                value = args[param_index]
                # Get parameter name if not provided
                if param_name is None:
                    sig = inspect.signature(func)
                    param_names = list(sig.parameters.keys())
                    if param_index < len(param_names):
                        name = param_names[param_index]
                    else:
                        name = f"parameter_{param_index}"
                else:
                    name = param_name
            else:
                # Parameter might be in kwargs
                sig = inspect.signature(func)
                param_names = list(sig.parameters.keys())
                if param_index < len(param_names):
                    name = param_names[param_index]
                    if name in kwargs:
                        value = kwargs[name]
                    else:
                        # Parameter not provided, use default
                        return func(*args, **kwargs)
                else:
                    # Invalid parameter index
                    return func(*args, **kwargs)

            # Validate the parameter
            if isinstance(value, (np.ndarray, pd.Series, pd.DataFrame)):
                validate_time_series(value, min_length, name)

            return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


def validate_input_positive_definite(
    param_index: int,
    param_name: Optional[str] = None,
    tol: float = 1e-8
) -> Callable[[F], F]:
    """Decorator factory for validating a positive definite matrix input parameter.

    Args:
        param_index: Index of the parameter to validate
        param_name: Name of the parameter for error messages (defaults to parameter name)
        tol: Tolerance for eigenvalue positivity check

    Returns:
        Callable: Decorator function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get the parameter value
            if param_index < len(args):
                value = args[param_index]
                # Get parameter name if not provided
                if param_name is None:
                    sig = inspect.signature(func)
                    param_names = list(sig.parameters.keys())
                    if param_index < len(param_names):
                        name = param_names[param_index]
                    else:
                        name = f"parameter_{param_index}"
                else:
                    name = param_name
            else:
                # Parameter might be in kwargs
                sig = inspect.signature(func)
                param_names = list(sig.parameters.keys())
                if param_index < len(param_names):
                    name = param_names[param_index]
                    if name in kwargs:
                        value = kwargs[name]
                    else:
                        # Parameter not provided, use default
                        return func(*args, **kwargs)
                else:
                    # Invalid parameter index
                    return func(*args, **kwargs)

            # Validate the parameter
            if isinstance(value, np.ndarray):
                validate_positive_definite(value, name, tol=tol)

            return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


def validate_input_bounds(
    param_index: int,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
    lower_inclusive: bool = True,
    upper_inclusive: bool = True,
    param_name: Optional[str] = None
) -> Callable[[F], F]:
    """Decorator factory for validating numeric bounds on an input parameter.

    Args:
        param_index: Index of the parameter to validate
        lower_bound: Lower bound, or None for no lower bound
        upper_bound: Upper bound, or None for no upper bound
        lower_inclusive: Whether the lower bound is inclusive
        upper_inclusive: Whether the upper bound is inclusive
        param_name: Name of the parameter for error messages (defaults to parameter name)

    Returns:
        Callable: Decorator function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get the parameter value
            if param_index < len(args):
                value = args[param_index]
                # Get parameter name if not provided
                if param_name is None:
                    sig = inspect.signature(func)
                    param_names = list(sig.parameters.keys())
                    if param_index < len(param_names):
                        name = param_names[param_index]
                    else:
                        name = f"parameter_{param_index}"
                else:
                    name = param_name
            else:
                # Parameter might be in kwargs
                sig = inspect.signature(func)
                param_names = list(sig.parameters.keys())
                if param_index < len(param_names):
                    name = param_names[param_index]
                    if name in kwargs:
                        value = kwargs[name]
                    else:
                        # Parameter not provided, use default
                        return func(*args, **kwargs)
                else:
                    # Invalid parameter index
                    return func(*args, **kwargs)

            # Validate the parameter
            if isinstance(value, (int, float)):
                validate_parameter_bounds(
                    value, name, lower_bound, upper_bound,
                    lower_inclusive, upper_inclusive
                )

            return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


def validate_input_numeric_array(
    param_index: int,
    param_name: Optional[str] = None,
    allow_nan: bool = False,
    allow_inf: bool = False
) -> Callable[[F], F]:
    """Decorator factory for validating numeric values in an array input parameter.

    Args:
        param_index: Index of the parameter to validate
        param_name: Name of the parameter for error messages (defaults to parameter name)
        allow_nan: Whether to allow NaN values
        allow_inf: Whether to allow infinite values

    Returns:
        Callable: Decorator function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get the parameter value
            if param_index < len(args):
                value = args[param_index]
                # Get parameter name if not provided
                if param_name is None:
                    sig = inspect.signature(func)
                    param_names = list(sig.parameters.keys())
                    if param_index < len(param_names):
                        name = param_names[param_index]
                    else:
                        name = f"parameter_{param_index}"
                else:
                    name = param_name
            else:
                # Parameter might be in kwargs
                sig = inspect.signature(func)
                param_names = list(sig.parameters.keys())
                if param_index < len(param_names):
                    name = param_names[param_index]
                    if name in kwargs:
                        value = kwargs[name]
                    else:
                        # Parameter not provided, use default
                        return func(*args, **kwargs)
                else:
                    # Invalid parameter index
                    return func(*args, **kwargs)

            # Validate the parameter
            if isinstance(value, np.ndarray):
                validate_numeric_array(value, name, allow_nan=allow_nan, allow_inf=allow_inf)

            return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


def validate_dataclass_input(
    param_index: int,
    param_name: Optional[str] = None
) -> Callable[[F], F]:
    """Decorator factory for validating a dataclass input parameter.

    Args:
        param_index: Index of the parameter to validate
        param_name: Name of the parameter for error messages (defaults to parameter name)

    Returns:
        Callable: Decorator function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get the parameter value
            if param_index < len(args):
                value = args[param_index]
                # Get parameter name if not provided
                if param_name is None:
                    sig = inspect.signature(func)
                    param_names = list(sig.parameters.keys())
                    if param_index < len(param_names):
                        name = param_names[param_index]
                    else:
                        name = f"parameter_{param_index}"
                else:
                    name = param_name
            else:
                # Parameter might be in kwargs
                sig = inspect.signature(func)
                param_names = list(sig.parameters.keys())
                if param_index < len(param_names):
                    name = param_names[param_index]
                    if name in kwargs:
                        value = kwargs[name]
                    else:
                        # Parameter not provided, use default
                        return func(*args, **kwargs)
                else:
                    # Invalid parameter index
                    return func(*args, **kwargs)

            # Validate the parameter
            if is_dataclass(value):
                validate_dataclass_fields(value)

            return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


def validate_input_type(
    param_index: int,
    expected_type: Union[Type, Tuple[Type, ...]],
    param_name: Optional[str] = None
) -> Callable[[F], F]:
    """Decorator factory for validating the type of an input parameter.

    Args:
        param_index: Index of the parameter to validate
        expected_type: Expected type or tuple of types
        param_name: Name of the parameter for error messages (defaults to parameter name)

    Returns:
        Callable: Decorator function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get the parameter value
            if param_index < len(args):
                value = args[param_index]
                # Get parameter name if not provided
                if param_name is None:
                    sig = inspect.signature(func)
                    param_names = list(sig.parameters.keys())
                    if param_index < len(param_names):
                        name = param_names[param_index]
                    else:
                        name = f"parameter_{param_index}"
                else:
                    name = param_name
            else:
                # Parameter might be in kwargs
                sig = inspect.signature(func)
                param_names = list(sig.parameters.keys())
                if param_index < len(param_names):
                    name = param_names[param_index]
                    if name in kwargs:
                        value = kwargs[name]
                    else:
                        # Parameter not provided, use default
                        return func(*args, **kwargs)
                else:
                    # Invalid parameter index
                    return func(*args, **kwargs)

            # Validate the parameter
            if value is not None and not isinstance(value, expected_type):
                if isinstance(expected_type, tuple):
                    type_names = " or ".join(t.__name__ for t in expected_type)
                    raise TypeError(f"Parameter {name} must be {type_names}, got {type(value).__name__}")
                else:
                    raise TypeError(f"Parameter {name} must be {expected_type.__name__}, got {type(value).__name__}")

            return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


def validate_custom_condition(
    param_index: int,
    condition: Callable[[Any], bool],
    error_message: str,
    param_name: Optional[str] = None
) -> Callable[[F], F]:
    """Decorator factory for validating a custom condition on an input parameter.

    Args:
        param_index: Index of the parameter to validate
        condition: Function that takes the parameter value and returns True if valid
        error_message: Error message to use if validation fails
        param_name: Name of the parameter for error messages (defaults to parameter name)

    Returns:
        Callable: Decorator function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get the parameter value
            if param_index < len(args):
                value = args[param_index]
                # Get parameter name if not provided
                if param_name is None:
                    sig = inspect.signature(func)
                    param_names = list(sig.parameters.keys())
                    if param_index < len(param_names):
                        name = param_names[param_index]
                    else:
                        name = f"parameter_{param_index}"
                else:
                    name = param_name
            else:
                # Parameter might be in kwargs
                sig = inspect.signature(func)
                param_names = list(sig.parameters.keys())
                if param_index < len(param_names):
                    name = param_names[param_index]
                    if name in kwargs:
                        value = kwargs[name]
                    else:
                        # Parameter not provided, use default
                        return func(*args, **kwargs)
                else:
                    # Invalid parameter index
                    return func(*args, **kwargs)

            # Validate the parameter
            if not condition(value):
                raise_parameter_error(
                    error_message.format(param=name, value=value),
                    param_name=name,
                    param_value=value
                )

            return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator
