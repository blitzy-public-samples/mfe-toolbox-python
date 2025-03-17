import numpy as np
from mfe.utils.matrix_ops import vech, ivech

def test_vech_operations():
    """
    Demonstrate vech (Vector-half) operations with examples
    """
    print("Vector-half (vech) Operations Examples\n")

    # Example 1: Basic 2x2 matrix
    print("Example 1: 2x2 Matrix")
    matrix_2x2 = np.array([[1.0, 0.5],
                          [0.5, 2.0]])
    print("\nInput matrix:")
    print(matrix_2x2)
    
    vech_result = vech(matrix_2x2)
    print("\nvech operation result:")
    print(vech_result)
    
    restored_matrix = ivech(vech_result)
    print("\nRestored matrix using ivech:")
    print(restored_matrix)
    print("\n" + "-"*50 + "\n")

    # Example 2: 3x3 correlation matrix
    print("Example 2: 3x3 Correlation Matrix")
    matrix_3x3 = np.array([[1.0, 0.5, 0.3],
                          [0.5, 1.0, 0.6],
                          [0.3, 0.6, 1.0]])
    print("\nInput correlation matrix:")
    print(matrix_3x3)
    
    vech_result = vech(matrix_3x3)
    print("\nvech operation result:")
    print(vech_result)
    
    restored_matrix = ivech(vech_result)
    print("\nRestored matrix using ivech:")
    print(restored_matrix)
    print("\n" + "-"*50 + "\n")

    # Example 3: Creating diagonal matrix
    print("Example 3: Creating Diagonal Matrix")
    diagonal_elements = np.array([1.0, 2.0, 3.0])
    print("\nInput diagonal elements:")
    print(diagonal_elements)
    
    diagonal_matrix = ivech(diagonal_elements, diagonal_only=True)
    print("\nDiagonal matrix created using ivech:")
    print(diagonal_matrix)
    
    vech_result = vech(diagonal_matrix)
    print("\nvech operation on diagonal matrix:")
    print(vech_result)

if __name__ == "__main__":
    test_vech_operations()
