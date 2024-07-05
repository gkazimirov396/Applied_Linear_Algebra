import numpy as np

def my_svd(A: np.ndarray):
    AtA = np.dot(A.T, A)
    AAt = np.dot(A, A.T)

    eigenvalues_AAt, eigenvectors_AAt = np.linalg.eig(AAt)
    U = eigenvectors_AAt[:, np.argsort(eigenvalues_AAt)[::-1]]

    eigenvalues_AtA, eigenvectors_AtA = np.linalg.eig(AtA)
    V = eigenvectors_AtA[:, np.argsort(eigenvalues_AtA)[::-1]]

    singular_values = np.sqrt(np.maximum(eigenvalues_AtA, 0))
    Σ = np.zeros(A.shape)
    Σ[:min(A.shape), :min(A.shape)] = np.diag(singular_values)

    for i in range(len(singular_values)):
        if singular_values[i] != 0:
            U[:, i] = np.dot(A, V[:, i]) / singular_values[i]
        else:
            U[:, i] = np.zeros(A.shape[0])

    return U, Σ, V.T

A = np.array([[-12.6, 2], [-43.1, -9], [53, 166]])
U, Σ, Vt = my_svd(A)

print("U: \n", U)
print("Σ: \n", Σ)
print("V^T: \n", Vt)

A_reconstructed = np.dot(U, np.dot(Σ, Vt))

print("Reconstructed matrix: \n", A_reconstructed)
print("Original matrix: \n", A)