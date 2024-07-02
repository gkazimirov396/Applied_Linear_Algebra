import numpy as np

def my_svd(A):
    AtA = np.dot(A.T, A)
    AAt = np.dot(A, A.T)
    
    eigenvalues_V, V = np.linalg.eig(AtA)
    eigenvalues_U, U = np.linalg.eig(AAt)
    
    sorted_indices_V = np.argsort(-eigenvalues_V)
    eigenvalues_V = eigenvalues_V[sorted_indices_V]
    V = V[:, sorted_indices_V]
    
    sorted_indices_U = np.argsort(-eigenvalues_U)
    eigenvalues_U = eigenvalues_U[sorted_indices_U]
    U = U[:, sorted_indices_U]
    
    sigma = np.sqrt(eigenvalues_V)
    
    m, n = A.shape
    Sigma = np.zeros((m, n))
    min_dim = min(m, n)
    Sigma[:min_dim, :min_dim] = np.diag(sigma)
    
    return U, Sigma, V.T

A = np.array([[1, 2], [3, 4], [5, 6]])
U, Sigma, VT = my_svd(A)

print("U:\n", U)
print("Sigma:\n", Sigma)
print("V^T:\n", VT)

A_reconstructed = np.dot(U, np.dot(Sigma, VT))

print("Reconstructed A:\n", A_reconstructed)
print("Original A:\n", A)