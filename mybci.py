import sys
import time
import numpy as np
import joblib
from preprocessing import load_and_epoch_data
import pipeline

MODEL_FILENAME = "bci_model.pkl"

# Définition des 6 expériences de l'énoncé (V.1.4)
EXPERIMENTS = {
    0: [3, 7, 11],    # Motor Execution: Left/Right Hand
    1: [4, 8, 12],    # Motor Imagery: Left/Right Hand
    2: [5, 9, 13],    # Motor Execution: Both Hands/Both Feet
    3: [6, 10, 14],   # Motor Imagery: Both Hands/Both Feet
    4: [3, 7, 11, 5, 9, 13], # All Motor Execution
    5: [4, 8, 12, 6, 10, 14]  # All Motor Imagery
}

def get_experiment_group(run_id):
    """ Trouve à quel groupe de runs appartient le run_id fourni """
    for group in EXPERIMENTS.values():
        if run_id in group:
            return group
    return [run_id]

def global_evaluation():
    """ 
    Calcule la 'Mean accuracy' sur les 109 sujets pour les 6 expériences.
    Répond à la contrainte de l'énoncé.
    """
    print("Démarrage de l'évaluation globale (109 sujets, 6 expériences)...")
    total_accs = []
    
    for exp_id in range(6):
        runs = EXPERIMENTS[exp_id]
        exp_scores = []
        for sub in range(1, 110):
            score = pipeline.evaluate_subject(sub, runs)
            if score is not None:
                exp_scores.append(score)
        
        mean_exp = np.mean(exp_scores) if exp_scores else 0
        total_accs.append(mean_exp)
        print(f"experiment {exp_id}: accuracy = {mean_exp:.4f}")
    
    print("-" * 30)
    print(f"Mean accuracy of 6 experiments: {np.mean(total_accs):.4f}")

def simulate_data_stream(subject, run):
    """ Flux temps réel < 2s """
    try:
        model = joblib.load(MODEL_FILENAME)
        X, y = load_and_epoch_data(subject, [run])
    except Exception as e:
        print(f"Erreur : {e}")
        return

    print(f"Simulation du flux (Sujet {subject}, Run {run})...")
    correct = 0
    for i in range(len(y)):
        chunk = X[i:i+1]
        start = time.time()
        pred = model.predict(chunk)[0]
        duration = time.time() - start
        
        if duration > 2.0: print(f"LATE: {duration:.2f}s")
        
        is_correct = (pred == y[i])
        if is_correct: correct += 1
        print(f"epoch {i:02d}: [{pred+1}] [{y[i]+1}] {is_correct}")
        time.sleep(0.05)
        
    print(f"Accuracy: {correct/len(y):.4f}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        global_evaluation()
    elif len(sys.argv) == 4:
        sub, run, mode = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3].lower()
        if mode == "train":
            group = get_experiment_group(run)
            # On passe le groupe pour que train_and_save puisse exclure le run cible
            pipeline.train_and_save(sub, run, group, MODEL_FILENAME)
            # Affichage du score de validation (V.1.4)
            acc = pipeline.evaluate_subject(sub, group)
            print(f"cross_val_score: {acc:.4f}")
        elif mode == "predict":
            simulate_data_stream(sub, run)
    else:
        print("Usage: python mybci.py <subject> <run> <train|predict>")