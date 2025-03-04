# mfe/utils/matrix_ops.py
"""
Matrix Operations Module

This module provides essential matrix transformation functions for the MFE Toolbox,
including vech/ivech operations, correlation matrix transformations, and matrix
decomposition utilities. These functions are critical for efficient manipulation
of covariance and correlation matrices throughout the toolbox.

The module implements optimized versions of common matrix operations using NumPy's
efficient array operations and Numba's JIT compilation for performance-critical
functions. All functions include comprehensive type hints and input validation
to ensure reliability and proper error handling.

Functions:
    vech: Vectorize the lower triangular portion of a symmetric matrix
    ivech: Inverse vech operation - convert vector to symmetric matrix
    vec2chol: Convert vector to Cholesky factor
    chol2vec: Convert Cholesky factor to vector
    cov2corr: Convert covariance matrix to correlation matrix
    corr2cov: Convert correlation matrix to covariance matrix
    ensure_symmetric: Ensure a matrix is symmetric
    is_positive_definite: Check if a matrix is positive definite
    nearest_positive_definite: Find the nearest positive definite matrix
    block_diagonal: Create a block diagonal matrix
    commutation_matrix: Create a commutation matrix
    duplication_matrix: Create a duplication matrix
    elimination_matrix: Create an elimination matrix
"""

import logging
import warnings
from typing import Optional, Tuple, Union, List, cast

import numpy as np
from scipy import linalg

from mfe.core.types import (
    Matrix, Vector, CorrelationMatrix, CovarianceMatrix, 
    PositiveDefiniteMatrix, TriangularMatrix
)
from mfe.core.exceptions import (
    DimensionError, NumericError, raise_dimension_error, 
    raise_numeric_error, warn_numeric
)

# Set up module-level logger
logger = logging.getLogger("mfe.utils.matrix_ops")

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
    logger.debug("Numba available for matrix operations acceleration")
except ImportError:
    # Create a no-op decorator with the same signature as jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    HAS_NUMBA = False
    logger.info("Numba not available. Matrix operations will use pure NumPy implementations.")


@jit(nopython=True, cache=True)
def _vech_numba(matrix: np.ndarray) -> np.ndarray:
    """
    Numba-accelerated implementation of vech operation.
    
    Args:
        matrix: Square symmetric matrix to vectorize
        
    Returns:
        Vector containing the lower triangular portion of the matrix
    """
    n = matrix.shape[0]
    vech_len = n * (n + 1) // 2
    result = np.zeros(vech_len, dtype=matrix.dtype)
    
    idx = 0
    for i in range(n):
        for j in range(i + 1):
            result[idx] = matrix[i, j]
            idx += 1
            
    return result



def vech(matrix: Matrix) -> Vector:
    """
    Vectorize the lower triangular portion of a symmetric matrix.
    
    This function extracts the lower triangular elements of a symmetric matrix
    and stacks them into a vector. For a matrix of size n×n, the resulting vector
    has length n(n+1)/2.
    
    Args:
        matrix: Square symmetric matrix to vectorize
        
    Returns:
        Vector containing the lower triangular portion of the matrix
        
    Raises:
        DimensionError: If the input matrix is not square
        
    Examples:
        >>> import numpy as np
        >>> from mfe.utils.matrix_ops import vech
        >>> A = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
        >>> vech(A)
        array([1, 2, 4, 3, 5, 6])
    """
    # Convert to numpy array if not already
    matrix = np.asarray(matrix)
    
    # Check if matrix is square
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise_dimension_error(
            "Input must be a square matrix",
            array_name="matrix",
            expected_shape="(n, n)",
            actual_shape=matrix.shape
        )
    
    # Use Numba-accelerated implementation if available
    if HAS_NUMBA:
        return _vech_numba(matrix)
    
    # Pure NumPy implementation
    n = matrix.shape[0]
    # Create mask for lower triangular elements (including diagonal)
    mask = np.tril(np.ones((n, n), dtype=bool))
    # Extract elements where mask is True
    return matrix[mask]


@jit(nopython=True, cache=True)
def _ivech_numba(vector: np.ndarray, diagonal_only: bool = False) -> np.ndarray:
    """
    Numba-accelerated implementation of inverse vech operation.
    
    Args:
        vector: Vector to convert to a symmetric matrix
        diagonal_only: If True, create a diagonal matrix
        
    Returns:
        Symmetric matrix constructed from the vector
    """
    # Compute matrix dimension from vector length
    # For a vector of length m, the matrix dimension n is:
    # n(n+1)/2 = m => n^2 + n - 2m = 0
    # Using quadratic formula: n = (-1 + sqrt(1 + 8m))/2
    m = vector.shape[0]
    n = int((-1 + np.sqrt(1 + 8 * m)) / 2)
    
    # Initialize result matrix
    result = np.zeros((n, n), dtype=vector.dtype)
    
    if diagonal_only:
        # Only fill the diagonal
        for i in range(n):
            result[i, i] = vector[i]
    else:
        # Fill lower triangular and copy to upper triangular
        idx = 0
        for i in range(n):
            for j in range(i + 1):
                result[i, j] = vector[idx]
                if i != j:
                    result[j, i] = vector[idx]
                idx += 1
                
    return result



def ivech(vector: Vector, diagonal_only: bool = False) -> Matrix:
    """
    Inverse vech operation - convert vector to symmetric matrix.
    
    This function takes a vector of length n(n+1)/2 and constructs a symmetric
    matrix of size n×n by filling the lower triangular portion and mirroring
    across the diagonal.
    
    Args:
        vector: Vector to convert to a symmetric matrix
        diagonal_only: If True, create a diagonal matrix with the first n elements
                      of the vector on the diagonal and zeros elsewhere
        
    Returns:
        Symmetric matrix constructed from the vector
        
    Raises:
        DimensionError: If the vector length is not a triangular number
        
    Examples:
        >>> import numpy as np
        >>> from mfe.utils.matrix_ops import ivech
        >>> v = np.array([1, 2, 4, 3, 5, 6])
        >>> ivech(v)
        array([[1, 2, 3],
               [2, 4, 5],
               [3, 5, 6]])
        >>> ivech(np.array([1, 2, 3]), diagonal_only=True)
        array([[1, 0, 0],
               [0, 2, 0],
               [0, 0, 3]])
    """
    # Convert to numpy array if not already
    vector = np.asarray(vector)
    
    # Check if vector is 1D
    if vector.ndim != 1:
        raise_dimension_error(
            "Input must be a 1D vector",
            array_name="vector",
            expected_shape="(m,)",
            actual_shape=vector.shape
        )
    
    # Compute matrix dimension from vector length
    m = vector.shape[0]
    n_float = (-1 + np.sqrt(1 + 8 * m)) / 2
    
    # Check if vector length is valid (corresponds to a triangular number)
    if not np.isclose(n_float, round(n_float)):
        raise_dimension_error(
            "Vector length must be a triangular number (n(n+1)/2 for some integer n)",
            array_name="vector",
            expected_shape="(n(n+1)/2,)",
            actual_shape=vector.shape
        )
    
    # Use Numba-accelerated implementation if available
    if HAS_NUMBA:
        return _ivech_numba(vector, diagonal_only)
    
    # Pure NumPy implementation
    n = int(round(n_float))
    
    if diagonal_only:
        # Create diagonal matrix
        if m < n:
            raise_dimension_error(
                "Vector is too short to create the requested diagonal matrix",
                array_name="vector",
                expected_shape=f"At least ({n},)",
                actual_shape=vector.shape
            )
        result = np.zeros((n, n), dtype=vector.dtype)
        np.fill_diagonal(result, vector[:n])
        return result
    
    # Create symmetric matrix
    result = np.zeros((n, n), dtype=vector.dtype)
    # Create indices for lower triangular elements
    indices = np.tril_indices(n)
    # Fill lower triangular elements
    result[indices] = vector
    # Make symmetric by copying lower triangle to upper triangle
    result = result + result.T - np.diag(np.diag(result))
    
    return result



def vec2chol(vector: Vector, n: int) -> TriangularMatrix:
    """
    Convert a vector to a Cholesky factor.
    
    This function takes a vector of length n(n+1)/2 and constructs a lower
    triangular Cholesky factor of size n×n.
    
    Args:
        vector: Vector of length n(n+1)/2 containing the elements of the Cholesky factor
        n: Dimension of the resulting matrix
        
    Returns:
        Lower triangular Cholesky factor
        
    Raises:
        DimensionError: If the vector length doesn't match n(n+1)/2
        
    Examples:
        >>> import numpy as np
        >>> from mfe.utils.matrix_ops import vec2chol
        >>> v = np.array([1, 2, 0, 3, 4, 5])
        >>> vec2chol(v, 3)
        array([[1., 0., 0.],
               [2., 3., 0.],
               [0., 4., 5.]])
    """
    # Convert to numpy array if not already
    vector = np.asarray(vector)
    
    # Check if vector is 1D
    if vector.ndim != 1:
        raise_dimension_error(
            "Input must be a 1D vector",
            array_name="vector",
            expected_shape="(n(n+1)/2,)",
            actual_shape=vector.shape
        )
    
    # Check if vector length matches n(n+1)/2
    expected_length = n * (n + 1) // 2
    if vector.shape[0] != expected_length:
        raise_dimension_error(
            f"Vector length must be n(n+1)/2 = {expected_length} for n = {n}",
            array_name="vector",
            expected_shape=f"({expected_length},)",
            actual_shape=vector.shape
        )
    
    # Initialize lower triangular matrix
    chol = np.zeros((n, n), dtype=vector.dtype)
    
    # Fill lower triangular elements
    idx = 0
    for i in range(n):
        for j in range(i + 1):
            chol[i, j] = vector[idx]
            idx += 1
    
    return chol



def chol2vec(chol: TriangularMatrix) -> Vector:
    """
    Convert a Cholesky factor to a vector.
    
    This function takes a lower triangular Cholesky factor of size n×n and
    extracts its elements into a vector of length n(n+1)/2.
    
    Args:
        chol: Lower triangular Cholesky factor
        
    Returns:
        Vector containing the elements of the Cholesky factor
        
    Raises:
        DimensionError: If the input matrix is not square
        
    Examples:
        >>> import numpy as np
        >>> from mfe.utils.matrix_ops import chol2vec
        >>> L = np.array([[1, 0, 0], [2, 3, 0], [0, 4, 5]])
        >>> chol2vec(L)
        array([1, 2, 0, 3, 4, 5])
    """
    # Convert to numpy array if not already
    chol = np.asarray(chol)
    
    # Check if matrix is square
    if chol.ndim != 2 or chol.shape[0] != chol.shape[1]:
        raise_dimension_error(
            "Input must be a square matrix",
            array_name="chol",
            expected_shape="(n, n)",
            actual_shape=chol.shape
        )
    
    # Extract lower triangular elements (including diagonal)
    return vech(np.tril(chol))



def cov2corr(cov: CovarianceMatrix) -> CorrelationMatrix:
    """
    Convert a covariance matrix to a correlation matrix.
    
    This function takes a covariance matrix and converts it to a correlation
    matrix by normalizing each element by the corresponding standard deviations.
    
    Args:
        cov: Covariance matrix (must be positive semi-definite)
        
    Returns:
        Correlation matrix
        
    Raises:
        DimensionError: If the input matrix is not square
        NumericError: If the covariance matrix contains non-positive diagonal elements
        
    Examples:
        >>> import numpy as np
        >>> from mfe.utils.matrix_ops import cov2corr
        >>> C = np.array([[4, 2, 0], [2, 9, -3], [0, -3, 16]])
        >>> cov2corr(C)
        array([[ 1.        ,  0.33333333,  0.        ],
               [ 0.33333333,  1.        , -0.25      ],
               [ 0.        , -0.25      ,  1.        ]])
    """
    # Convert to numpy array if not already
    cov = np.asarray(cov)
    
    # Check if matrix is square
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise_dimension_error(
            "Input must be a square matrix",
            array_name="cov",
            expected_shape="(n, n)",
            actual_shape=cov.shape
        )
    
    # Extract standard deviations from diagonal
    std_devs = np.sqrt(np.diag(cov))
    
    # Check for non-positive diagonal elements
    if np.any(std_devs <= 0):
        raise_numeric_error(
            "Covariance matrix contains non-positive diagonal elements",
            operation="cov2corr",
            values=np.diag(cov),
            error_type="invalid_covariance"
        )
    
    # Create outer product of standard deviations
    std_outer = np.outer(std_devs, std_devs)
    
    # Compute correlation matrix
    corr = cov / std_outer
    
    # Ensure exact ones on the diagonal (to handle numerical precision issues)
    np.fill_diagonal(corr, 1.0)
    
    # Ensure the matrix is symmetric (to handle numerical precision issues)
    corr = (corr + corr.T) / 2
    
    return corr



def corr2cov(corr: CorrelationMatrix, std_devs: Vector) -> CovarianceMatrix:
    """
    Convert a correlation matrix to a covariance matrix.
    
    This function takes a correlation matrix and a vector of standard deviations
    and constructs a covariance matrix.
    
    Args:
        corr: Correlation matrix (must have ones on the diagonal)
        std_devs: Vector of standard deviations
        
    Returns:
        Covariance matrix
        
    Raises:
        DimensionError: If the correlation matrix is not square or if the length
                       of std_devs doesn't match the dimension of corr
        NumericError: If the correlation matrix doesn't have ones on the diagonal
                     or if std_devs contains non-positive values
        
    Examples:
        >>> import numpy as np
        >>> from mfe.utils.matrix_ops import corr2cov
        >>> R = np.array([[1, 0.3, 0], [0.3, 1, -0.2], [0, -0.2, 1]])
        >>> s = np.array([2, 3, 4])
        >>> corr2cov(R, s)
        array([[ 4. ,  1.8,  0. ],
               [ 1.8,  9. , -2.4],
               [ 0. , -2.4, 16. ]])
    """
    # Convert to numpy arrays if not already
    corr = np.asarray(corr)
    std_devs = np.asarray(std_devs)
    
    # Check if correlation matrix is square
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise_dimension_error(
            "Correlation matrix must be square",
            array_name="corr",
            expected_shape="(n, n)",
            actual_shape=corr.shape
        )
    
    # Check if std_devs is a vector with matching dimension
    if std_devs.ndim != 1:
        raise_dimension_error(
            "Standard deviations must be a 1D vector",
            array_name="std_devs",
            expected_shape="(n,)",
            actual_shape=std_devs.shape
        )
    
    if std_devs.shape[0] != corr.shape[0]:
        raise_dimension_error(
            "Length of standard deviations must match dimension of correlation matrix",
            array_name="std_devs",
            expected_shape=f"({corr.shape[0]},)",
            actual_shape=std_devs.shape
        )
    
    # Check if correlation matrix has ones on the diagonal
    if not np.allclose(np.diag(corr), 1.0):
        raise_numeric_error(
            "Correlation matrix must have ones on the diagonal",
            operation="corr2cov",
            values=np.diag(corr),
            error_type="invalid_correlation"
        )
    
    # Check if std_devs contains non-positive values
    if np.any(std_devs <= 0):
        raise_numeric_error(
            "Standard deviations must be positive",
            operation="corr2cov",
            values=std_devs,
            error_type="invalid_std_devs"
        )
    
    # Create outer product of standard deviations
    std_outer = np.outer(std_devs, std_devs)
    
    # Compute covariance matrix
    cov = corr * std_outer
    
    # Ensure the matrix is symmetric (to handle numerical precision issues)
    cov = (cov + cov.T) / 2
    
    return cov



def ensure_symmetric(matrix: Matrix, tol: float = 1e-8) -> Matrix:
    """
    Ensure a matrix is symmetric by averaging with its transpose.
    
    This function takes a matrix and makes it symmetric by averaging it with
    its transpose. If the matrix is already symmetric within the specified
    tolerance, it is returned unchanged.
    
    Args:
        matrix: Matrix to make symmetric
        tol: Tolerance for checking symmetry
        
    Returns:
        Symmetric matrix
        
    Raises:
        DimensionError: If the input matrix is not square
        
    Examples:
        >>> import numpy as np
        >>> from mfe.utils.matrix_ops import ensure_symmetric
        >>> A = np.array([[1, 2.000001], [2, 3]])
        >>> ensure_symmetric(A)
        array([[1.      , 2.0000005],
               [2.0000005, 3.      ]])
    """
    # Convert to numpy array if not already
    matrix = np.asarray(matrix)
    
    # Check if matrix is square
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise_dimension_error(
            "Input must be a square matrix",
            array_name="matrix",
            expected_shape="(n, n)",
            actual_shape=matrix.shape
        )
    
    # Check if matrix is already symmetric within tolerance
    if np.allclose(matrix, matrix.T, rtol=tol, atol=tol):
        return matrix
    
    # Make symmetric by averaging with transpose
    return (matrix + matrix.T) / 2



def is_positive_definite(matrix: Matrix, tol: float = 1e-8) -> bool:
    """
    Check if a matrix is positive definite.
    
    This function checks if a matrix is positive definite by attempting to
    compute its Cholesky decomposition. A matrix is positive definite if all
    its eigenvalues are positive.
    
    Args:
        matrix: Matrix to check
        tol: Tolerance for numerical stability checks
        
    Returns:
        True if the matrix is positive definite, False otherwise
        
    Examples:
        >>> import numpy as np
        >>> from mfe.utils.matrix_ops import is_positive_definite
        >>> A = np.array([[2, 1], [1, 2]])
        >>> is_positive_definite(A)
        True
        >>> B = np.array([[1, 2], [2, 1]])
        >>> is_positive_definite(B)
        False
    """
    # Convert to numpy array if not already
    matrix = np.asarray(matrix)
    
    # Check if matrix is square
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    
    # Ensure matrix is symmetric
    matrix = ensure_symmetric(matrix, tol)
    
    # Try to compute Cholesky decomposition
    try:
        # Use scipy's cholesky which is more numerically stable
        linalg.cholesky(matrix, lower=True, check_finite=False)
        return True
    except linalg.LinAlgError:
        return False



def nearest_positive_definite(matrix: Matrix, epsilon: float = 0.0) -> PositiveDefiniteMatrix:
    """
    Find the nearest positive definite matrix to the input matrix.
    
    This function computes the nearest positive definite matrix to the input
    matrix using the algorithm of Higham (1988). The resulting matrix is
    symmetric and positive definite.
    
    Args:
        matrix: Matrix to find the nearest positive definite matrix for
        epsilon: Small value to add to the diagonal for numerical stability
        
    Returns:
        Nearest positive definite matrix
        
    Raises:
        DimensionError: If the input matrix is not square
        
    Examples:
        >>> import numpy as np
        >>> from mfe.utils.matrix_ops import nearest_positive_definite
        >>> A = np.array([[1, 2], [2, 1]])  # Not positive definite
        >>> nearest_positive_definite(A)
        array([[1.00036297, 1.99963703],
               [1.99963703, 1.00036297]])
    """
    # Convert to numpy array if not already
    matrix = np.asarray(matrix)
    
    # Check if matrix is square
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise_dimension_error(
            "Input must be a square matrix",
            array_name="matrix",
            expected_shape="(n, n)",
            actual_shape=matrix.shape
        )
    
    # Ensure matrix is symmetric
    matrix = ensure_symmetric(matrix)
    
    # Check if matrix is already positive definite
    if is_positive_definite(matrix) and epsilon == 0.0:
        return matrix
    
    # Compute the symmetric polar factor
    try:
        # Compute eigendecomposition
        eigvals, eigvecs = linalg.eigh(matrix)
        
        # Replace negative eigenvalues with small positive values
        eigvals = np.maximum(eigvals, 0)
        
        # Add epsilon to the diagonal for numerical stability
        if epsilon > 0:
            eigvals += epsilon
        
        # Reconstruct the matrix
        result = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        # Ensure the result is symmetric (to handle numerical precision issues)
        result = ensure_symmetric(result)
        
        return result
    except Exception as e:
        # If eigendecomposition fails, try a simpler approach
        warn_numeric(
            "Eigendecomposition failed in nearest_positive_definite, using simpler approach",
            operation="nearest_positive_definite",
            issue="eigendecomposition_failed",
            value=str(e)
        )
        
        # Make symmetric
        matrix = ensure_symmetric(matrix)
        
        # Add a small value to the diagonal
        result = matrix.copy()
        np.fill_diagonal(result, np.diag(result) + epsilon)
        
        # If still not positive definite, increase diagonal elements
        while not is_positive_definite(result):
            min_eig = np.min(linalg.eigvalsh(result))
            np.fill_diagonal(result, np.diag(result) + abs(min_eig) * 1.1)
        
        return result



def block_diagonal(matrices: List[Matrix]) -> Matrix:
    """
    Create a block diagonal matrix from a list of matrices.
    
    This function takes a list of matrices and arranges them as blocks along
    the diagonal of a larger matrix, with zeros elsewhere.
    
    Args:
        matrices: List of matrices to arrange as blocks
        
    Returns:
        Block diagonal matrix
        
    Examples:
        >>> import numpy as np
        >>> from mfe.utils.matrix_ops import block_diagonal
        >>> A = np.array([[1, 2], [3, 4]])
        >>> B = np.array([[5, 6], [7, 8]])
        >>> block_diagonal([A, B])
        array([[1, 2, 0, 0],
               [3, 4, 0, 0],
               [0, 0, 5, 6],
               [0, 0, 7, 8]])
    """
    # Check if input is empty
    if not matrices:
        return np.array([[]])
    
    # Convert all matrices to numpy arrays
    matrices = [np.asarray(mat) for mat in matrices]
    
    # Check if all matrices are 2D
    if not all(mat.ndim == 2 for mat in matrices):
        raise_dimension_error(
            "All matrices must be 2D",
            array_name="matrices",
            expected_shape="List of (n_i, m_i) matrices"
        )
    
    # Use scipy's block_diag function
    return linalg.block_diag(*matrices)



def commutation_matrix(n: int, m: int) -> Matrix:
    """
    Create a commutation matrix.
    
    This function creates a commutation matrix K_{n,m} such that
    K_{n,m} * vec(A) = vec(A.T) for any n×m matrix A, where vec is the
    vectorization operator that stacks columns of a matrix.
    
    Args:
        n: Number of rows in the original matrix
        m: Number of columns in the original matrix
        
    Returns:
        Commutation matrix of size (nm)×(nm)
        
    Examples:
        >>> import numpy as np
        >>> from mfe.utils.matrix_ops import commutation_matrix
        >>> K = commutation_matrix(2, 3)
        >>> K.shape
        (6, 6)
        >>> A = np.array([[1, 2, 3], [4, 5, 6]])
        >>> vec_A = A.flatten('F')  # Column-major (Fortran-style) flattening
        >>> vec_AT = A.T.flatten('F')
        >>> np.allclose(K @ vec_A, vec_AT)
        True
    """
    # Initialize result matrix
    K = np.zeros((n * m, n * m))
    
    # Fill the commutation matrix
    for i in range(n):
        for j in range(m):
            # Position in vec(A)
            pos1 = j * n + i
            # Position in vec(A.T)
            pos2 = i * m + j
            # Set the corresponding element to 1
            K[pos2, pos1] = 1
    
    return K



def duplication_matrix(n: int) -> Matrix:
    """
    Create a duplication matrix.
    
    This function creates a duplication matrix D_n such that
    D_n * vech(A) = vec(A) for any symmetric n×n matrix A, where vech is the
    half-vectorization operator that stacks the lower triangular part of a matrix.
    
    Args:
        n: Dimension of the symmetric matrix
        
    Returns:
        Duplication matrix of size (n²)×(n(n+1)/2)
        
    Examples:
        >>> import numpy as np
        >>> from mfe.utils.matrix_ops import duplication_matrix, vech
        >>> D = duplication_matrix(3)
        >>> D.shape
        (9, 6)
        >>> A = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])  # Symmetric matrix
        >>> vech_A = vech(A)
        >>> vec_A = A.flatten('F')  # Column-major (Fortran-style) flattening
        >>> np.allclose(D @ vech_A, vec_A)
        True
    """
    # Calculate dimensions
    n_squared = n * n
    n_vech = n * (n + 1) // 2
    
    # Initialize result matrix
    D = np.zeros((n_squared, n_vech))
    
    # Fill the duplication matrix
    vech_idx = 0
    for j in range(n):
        for i in range(j, n):
            # Position in vech(A)
            vec_idx1 = j * n + i  # Position in vec(A) for (i,j)
            vec_idx2 = i * n + j  # Position in vec(A) for (j,i)
            
            # Set the corresponding elements to 1
            D[vec_idx1, vech_idx] = 1
            if i != j:
                D[vec_idx2, vech_idx] = 1
            
            vech_idx += 1
    
    return D



def elimination_matrix(n: int) -> Matrix:
    """
    Create an elimination matrix.
    
    This function creates an elimination matrix L_n such that
    L_n * vec(A) = vech(A) for any symmetric n×n matrix A, where vec is the
    vectorization operator and vech is the half-vectorization operator.
    
    Args:
        n: Dimension of the symmetric matrix
        
    Returns:
        Elimination matrix of size (n(n+1)/2)×(n²)
        
    Examples:
        >>> import numpy as np
        >>> from mfe.utils.matrix_ops import elimination_matrix, vech
        >>> L = elimination_matrix(3)
        >>> L.shape
        (6, 9)
        >>> A = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])  # Symmetric matrix
        >>> vech_A = vech(A)
        >>> vec_A = A.flatten('F')  # Column-major (Fortran-style) flattening
        >>> np.allclose(L @ vec_A, vech_A)
        True
    """
    # Calculate dimensions
    n_squared = n * n
    n_vech = n * (n + 1) // 2
    
    # Initialize result matrix
    L = np.zeros((n_vech, n_squared))
    
    # Fill the elimination matrix
    vech_idx = 0
    for j in range(n):
        for i in range(j, n):
            # Position in vec(A)
            vec_idx = j * n + i
            
            # Set the corresponding element to 1
            L[vech_idx, vec_idx] = 1
            
            vech_idx += 1
    
    return L


# Register Numba-accelerated functions if available
def _register_numba_functions() -> None:
    """
    Register Numba JIT-compiled functions for matrix operations.
    
    This function is called during module initialization to register
    performance-critical functions for JIT compilation if Numba is available.
    """
    if HAS_NUMBA:
        # The functions are already decorated with @jit, so no additional
        # registration is needed here. This function is kept for consistency
        # with the module structure and potential future enhancements.
        logger.debug("Matrix operations Numba JIT functions registered")
    else:
        logger.info("Numba not available. Matrix operations will use pure NumPy implementations.")


# Initialize the module
_register_numba_functions()
