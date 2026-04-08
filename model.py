import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin

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
        evals, evecs = eigh(C_0, C_0 + C_1)

        # Tri des valeurs propres par ordre décroissant
        sorted_indices = np.argsort(evals)[::-1]
        evecs = evecs[:, sorted_indices]

        # On sélectionne les composantes extrêmes (qui maximisent la séparation)
        # Ex pour n=4 : les 2 premiers et les 2 derniers vecteurs propres
        half_n = self.n_components // 2
        filter_indices = np.concatenate([sorted_indices[:half_n], sorted_indices[-half_n:]])
        
        self.filters_ = evecs[:, filter_indices].T # Shape : (n_components, n_channels)
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
