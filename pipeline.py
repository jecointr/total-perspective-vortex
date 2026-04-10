import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from custom_lda import CustomLDA
from sklearn.model_selection import StratifiedKFold, cross_val_score

from preprocessing import load_and_epoch_data
from model import CustomCSP

def create_bci_pipeline(n_components=4):
    return Pipeline([
        ('csp', CustomCSP(n_components=n_components)),
        ('classifier', CustomLDA())
    ])

def evaluate_subject(subject, runs, n_components=4):
    """ Évaluation rigoureuse en cross-validation """
    try:
        X, y = load_and_epoch_data(subject, runs)
        pipeline = create_bci_pipeline(n_components)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X, y, cv=cv, n_jobs=-1)
        return np.mean(scores)
    except Exception as e:
        # Certains fichiers PhysioNet sont parfois corrompus ou manquants
        return None

def evaluate_with_data(X, y, n_components=4):
    """ Évaluation en cross-validation sur des données pré-chargées (pour datasets alternatifs) """
    try:
        pipe = create_bci_pipeline(n_components)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(pipe, X, y, cv=cv, n_jobs=-1)
        return np.mean(scores)
    except Exception:
        return None


def train_and_save(subject, target_run, all_runs, output_filename="bci_model.pkl"):
    """
    ENTRAÎNEMENT ANTI-DATA LEAK :
    S'entraîne sur les runs de 'all_runs' SAUF le 'target_run'.
    """
    # On filtre pour ne garder que les données d'entraînement (données jamais vues en prédiction)
    train_runs = [r for r in all_runs if r != target_run]
    
    print(f"Entraînement sur les runs {train_runs} (Exclut le run de test {target_run})...")
    X_train, y_train = load_and_epoch_data(subject, train_runs)
    
    pipeline = create_bci_pipeline()
    pipeline.fit(X_train, y_train)
    
    joblib.dump(pipeline, output_filename)
    print(f"Modèle sauvegardé : {output_filename}")

if __name__ == "__main__":
    # Test unitaire de la pipeline sur les tâches motrices (poings gauche/droite)
    # Les runs 4, 8, 12 correspondent à l'imagination motrice dans le dataset
    evaluate_subject(subject=1, runs=[4, 8, 12])
