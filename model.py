import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def custom_cholesky(A):
    """
    Decomposition de Cholesky : A = L @ L.T
    A doit etre symetrique definie positive.
    Retourne L (triangulaire inferieure).
    """
    n = A.shape[0]
    L = np.zeros_like(A)
    for i in range(n):
        for j in range(i + 1):
            s = np.dot(L[i, :j], L[j, :j])
            if i == j:
                L[i, j] = np.sqrt(A[i, i] - s)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]
    return L


def forward_substitution(L, B):
    """
    Resout L @ X = B par substitution avant.
    L : triangulaire inferieure (n, n)
    B : matrice (n, n)
    Retourne X = L^-1 @ B
    """
    n = L.shape[0]
    X = np.zeros_like(B)
    for i in range(n):
        X[i] = (B[i] - L[i, :i] @ X[:i]) / L[i, i]
    return X


def invert_lower_triangular(L):
    """Inverse une matrice triangulaire inferieure."""
    return forward_substitution(L, np.eye(L.shape[0]))


def jacobi_eigen(A, tol=1e-10, max_iter=10000):
    """
    Algorithme de Jacobi (cyclic sweep) pour la decomposition en valeurs propres
    d'une matrice symetrique.
    Parcourt systematiquement toutes les paires (p, q) par sweep complet,
    ce qui converge plus rapidement que la selection du max.

    Retourne (eigenvalues, eigenvectors) tries par ordre croissant.
    """
    n = A.shape[0]
    A = A.copy()
    V = np.eye(n)

    for iteration in range(max_iter):
        # Verifier la convergence (norme des elements hors-diagonale)
        off_norm = np.sqrt(np.sum(np.triu(A, k=1) ** 2))
        if off_norm < tol:
            break

        # Sweep complet : parcourir toutes les paires (p, q) avec p < q
        for p in range(n):
            for q in range(p + 1, n):
                if abs(A[p, q]) < tol / n:
                    continue

                # Calcul de l'angle de rotation
                if abs(A[p, p] - A[q, q]) < 1e-15:
                    theta = np.pi / 4
                else:
                    theta = 0.5 * np.arctan2(2 * A[p, q], A[p, p] - A[q, q])

                c = np.cos(theta)
                s = np.sin(theta)

                # Mise a jour des elements diagonaux
                A_pp = c * c * A[p, p] + 2 * s * c * A[p, q] + s * s * A[q, q]
                A_qq = s * s * A[p, p] - 2 * s * c * A[p, q] + c * c * A[q, q]

                # Mise a jour des colonnes p et q
                A_ip = A[:, p].copy()
                A_iq = A[:, q].copy()

                A[:, p] = c * A_ip + s * A_iq
                A[:, q] = -s * A_ip + c * A_iq
                A[p, :] = A[:, p]
                A[q, :] = A[:, q]

                A[p, p] = A_pp
                A[q, q] = A_qq
                A[p, q] = 0.0
                A[q, p] = 0.0

                # Mise a jour des vecteurs propres
                V_p = V[:, p].copy()
                V_q = V[:, q].copy()
                V[:, p] = c * V_p + s * V_q
                V[:, q] = -s * V_p + c * V_q

    eigenvalues = np.diag(A)
    # Tri par ordre croissant (meme convention que scipy.linalg.eigh)
    idx = np.argsort(eigenvalues)
    return eigenvalues[idx], V[:, idx]


def custom_eigh(A, B):
    """
    Resout le probleme aux valeurs propres generalisees : A @ v = lambda * B @ v
    Meme signature que scipy.linalg.eigh(A, B).

    Methode : reduction via decomposition de Cholesky.
    1. B = L @ L.T  (Cholesky)
    2. A' = L^-1 @ A @ L^-T  (probleme standard)
    3. A' @ z = lambda * z  (Jacobi)
    4. v = L^-T @ z  (reconstruction)
    """
    # Regularisation pour stabilite numerique
    B_reg = B + 1e-10 * np.eye(B.shape[0])

    # 1. Cholesky de B
    L = custom_cholesky(B_reg)

    # 2. Reduction au probleme standard
    L_inv = invert_lower_triangular(L)
    A_prime = L_inv @ A @ L_inv.T

    # Symetriser (corriger les erreurs d'arrondi)
    A_prime = 0.5 * (A_prime + A_prime.T)

    # 3. Decomposition par Jacobi
    eigenvalues, Z = jacobi_eigen(A_prime)

    # 4. Reconstruction des vecteurs propres originaux
    eigenvectors = L_inv.T @ Z

    return eigenvalues, eigenvectors

class CustomCSP(BaseEstimator, TransformerMixin):
    """
    Implémentation algorithmique du Common Spatial Patterns (CSP).
    Transforme les signaux EEG bruts en features basées sur la variance spatiale.
    """
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.filters_ = None

    def fit(self, X, y):
        """
        Calcule la matrice de projection W basée sur les matrices de covariance.
        X shape: (n_epochs, n_channels, n_times)
        """
        # Séparation des données par classe (supposant classes 0 et 1)
        X_class_0 = X[y == 0]
        X_class_1 = X[y == 1]

        # Fonction pour calculer la matrice de covariance spatiale moyenne
        def compute_covariance(X_class):
            cov_matrices = []
            for epoch in X_class:
                # X_e shape : (channels, time)
                # Covariance : X_e * X_e.T / trace(X_e * X_e.T)
                cov = np.dot(epoch, epoch.T)
                cov /= np.trace(cov)
                cov_matrices.append(cov)
            return np.mean(cov_matrices, axis=0)

        C_0 = compute_covariance(X_class_0)
        C_1 = compute_covariance(X_class_1)

        # Résolution du problème aux valeurs propres généralisées
        # C_0 * W = lambda * (C_0 + C_1) * W
        evals, evecs = custom_eigh(C_0, C_0 + C_1)

        # Tri des valeurs propres par ordre décroissant
        sorted_indices = np.argsort(evals)[::-1]
        evecs = evecs[:, sorted_indices]

        # On sélectionne les composantes extrêmes (qui maximisent la séparation)
        # Ex pour n=4 : les 2 premiers et les 2 derniers vecteurs propres
        half_n = self.n_components // 2
        selected = list(range(half_n)) + list(range(-half_n, 0))

        self.filters_ = evecs[:, selected].T # Shape : (n_components, n_channels)
        return self

    def transform(self, X):
        """
        Applique les filtres spatiaux et extrait la log-variance comme feature.
        """
        features = []
        for epoch in X:
            # Projection du signal : Z = W * X
            projected = np.dot(self.filters_, epoch)
            
            # Calcul de la variance sur le temps pour chaque composante
            var = np.var(projected, axis=1)
            
            # Application du logarithme pour normaliser la distribution (Standard en BCI)
            log_var = np.log(var)
            features.append(log_var)
            
        return np.array(features)
