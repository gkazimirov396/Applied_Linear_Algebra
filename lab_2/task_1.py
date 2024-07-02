import numpy as np

def eigenvalues_and_eigenvectors(matrix):
  eigenvalues, eigenvectors = np.linalg.eig(matrix)

  return eigenvalues, eigenvectors

def check_eigen(matrix, eigenvalues, eigenvectors):
    for i in range(len(eigenvalues)):
      v = eigenvectors[:, i];
      lv = eigenvalues[i] * v;
      Av = np.dot(matrix, v);

      if not np.allclose(Av, lv):
          return False
        
    return True

# Example 
A = np.array([[3, 2], [2,  6]])

eigenvalues, eigenvectors = eigenvalues_and_eigenvectors(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

is_equal = check_eigen(A, eigenvalues, eigenvectors)
print(is_equal)