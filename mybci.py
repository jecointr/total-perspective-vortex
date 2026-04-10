import sys
import time
import numpy as np
import joblib
from preprocessing import load_and_epoch_data
from dataset_loader import load_dataset, get_n_subjects
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

def parse_args(argv):
    """Parse les arguments CLI en extrayant le flag --dataset."""
    dataset = "physionet"
    args = list(argv[1:])

    if "--dataset" in args:
        idx = args.index("--dataset")
        dataset = args[idx + 1]
        args.pop(idx)  # --dataset
        args.pop(idx)  # valeur

    return dataset, args

def global_evaluation(dataset="physionet"):
    """
    Calcule la 'Mean accuracy' sur tous les sujets pour les expériences disponibles.
    """
    n_subjects = get_n_subjects(dataset)

    if dataset == "physionet":
        print(f"Évaluation globale ({n_subjects} sujets, 6 expériences, dataset: {dataset})...")
        total_accs = []

        for exp_id in range(6):
            runs = EXPERIMENTS[exp_id]
            exp_scores = []
            for sub in range(1, n_subjects + 1):
                score = pipeline.evaluate_subject(sub, runs)
                if score is not None:
                    exp_scores.append(score)

            mean_exp = np.mean(exp_scores) if exp_scores else 0
            total_accs.append(mean_exp)
            print(f"experiment {exp_id}: accuracy = {mean_exp:.4f}")

        print("-" * 30)
        print(f"Mean accuracy of 6 experiments: {np.mean(total_accs):.4f}")
    else:
        print(f"Évaluation globale ({n_subjects} sujets, dataset: {dataset})...")
        scores = []
        for sub in range(1, n_subjects + 1):
            try:
                X, y = load_dataset(dataset, sub)
                score = pipeline.evaluate_with_data(X, y)
                if score is not None:
                    scores.append(score)
                    print(f"subject {sub:03d}: accuracy = {score:.4f}")
            except Exception as e:
                print(f"subject {sub:03d}: erreur - {e}")

        if scores:
            print("-" * 30)
            print(f"Mean accuracy ({dataset}): {np.mean(scores):.4f}")

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
    dataset, args = parse_args(sys.argv)

    if len(args) == 0:
        global_evaluation(dataset)
    elif len(args) == 3:
        sub, run, mode = int(args[0]), int(args[1]), args[2].lower()
        if mode == "train":
            group = get_experiment_group(run)
            pipeline.train_and_save(sub, run, group, MODEL_FILENAME)
            acc = pipeline.evaluate_subject(sub, group)
            print(f"cross_val_score: {acc:.4f}")
        elif mode == "predict":
            simulate_data_stream(sub, run)
    else:
        print("Usage: python mybci.py [--dataset physionet|bnci2014] <subject> <run> <train|predict>")
