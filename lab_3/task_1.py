import numpy as np

def my_svd(A: np.ndarray):
    AtA = np.dot(A.T, A)
    AAt = np.dot(A, A.T)

    eigenvalues_AtA, eigenvectors_AtA = np.linalg.eig(AtA)

    sorted_eigenvalues = np.sort(eigenvalues_AtA)[::-1][:min(A.shape)]
    singular_values = np.sqrt(np.maximum(sorted_eigenvalues, 0))

    Σ = np.zeros(A.shape)
    Σ[:min(A.shape), :min(A.shape)] = np.diag(singular_values)

    U = np.zeros((A.shape[0], A.shape[0]))
    V = eigenvectors_AtA[:, np.argsort(eigenvalues_AtA)[::-1]]

    for i in range(len(singular_values)):
        if singular_values[i] != 0:
            U[:, i] = np.dot(A, V[:, i]) / singular_values[i]
        else:
            U[:, i] = np.zeros(A.shape[0])

    return U, Σ, V.T

A = np.array([[-12.6, 2], [-43.1, 7], [53, 166]])
#A= np.array([[1, 2, 3], [4, 5, 6]])
#A = np.array([[2, 0], [12, 33]])
U, Σ, Vt = my_svd(A)

print("U: \n", U)
print("Σ: \n", Σ)
print("V^T: \n", Vt)

A_reconstructed = np.dot(U, np.dot(Σ, Vt)).round(1)

print("Reconstructed matrix: \n", A_reconstructed)
print("Original matrix: \n", A)