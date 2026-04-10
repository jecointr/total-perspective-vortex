import numpy as np
from sklearn.base import BaseEstimator


class CustomLDA(BaseEstimator):
    """
    Implementation from scratch de l'Analyse Discriminante Lineaire (Fisher LDA).
    Classification binaire basee sur la projection sur l'axe discriminant optimal.
    """
    def __init__(self):
        self.w_ = None
        self.threshold_ = None
        self.classes_ = None

    def fit(self, X, y):
        """
        Calcule la direction discriminante optimale de Fisher.
        X shape: (n_samples, n_features)
        y shape: (n_samples,) avec 2 classes
        """
        self.classes_ = np.unique(y)

        X_0 = X[y == self.classes_[0]]
        X_1 = X[y == self.classes_[1]]

        # Moyennes par classe
        mu_0 = np.mean(X_0, axis=0)
        mu_1 = np.mean(X_1, axis=0)

        # Matrices de dispersion intra-classe (within-class scatter)
        S_0 = (X_0 - mu_0).T @ (X_0 - mu_0)
        S_1 = (X_1 - mu_1).T @ (X_1 - mu_1)
        S_W = S_0 + S_1

        # Regularisation pour eviter la singularite
        S_W += 1e-6 * np.eye(S_W.shape[0])

        # Direction discriminante : w = S_W^-1 @ (mu_1 - mu_0)
        self.w_ = np.linalg.solve(S_W, mu_1 - mu_0)

        # Seuil = point median entre les projections des deux moyennes
        self.threshold_ = 0.5 * (self.w_ @ mu_0 + self.w_ @ mu_1)

        return self

    def predict(self, X):
        """
        Projette X sur l'axe discriminant et classifie par rapport au seuil.
        """
        projections = X @ self.w_
        predictions = np.where(projections > self.threshold_,
                               self.classes_[1], self.classes_[0])
        return predictions

    def score(self, X, y):
        """Retourne l'accuracy de la classification."""
        return np.mean(self.predict(X) == y)
