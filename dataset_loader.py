import numpy as np
from preprocessing import load_and_epoch_data


def load_physionet(subject, runs, use_wavelet=False):
    """Charge les donnees PhysioNet EEG BCI (dataset par defaut)."""
    return load_and_epoch_data(subject, runs, use_wavelet=use_wavelet)


def load_bnci2014(subject):
    """
    Charge le dataset BNCI2014_001 (BCI Competition IV 2a).
    9 sujets, 22 canaux EEG, 250Hz, classification motrice.
    On selectionne 2 classes (left hand vs right hand) pour rester en binaire.
    """
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery

    dataset = BNCI2014_001()
    paradigm = MotorImagery(events=["left_hand", "right_hand"], n_classes=2, fmin=8, fmax=35, resample=160)
    X, y, _ = paradigm.get_data(dataset=dataset, subjects=[subject])

    # Normalisation des labels en 0/1
    label_map = {"left_hand": 0, "right_hand": 1}
    y_numeric = np.array([label_map[label] for label in y])

    return X, y_numeric


def load_dataset(dataset_name, subject, **kwargs):
    """
    Interface unifiee pour charger differents datasets EEG.

    Args:
        dataset_name: "physionet" ou "bnci2014"
        subject: numero du sujet
        **kwargs: arguments supplementaires (runs pour physionet, etc.)

    Returns:
        X: (n_epochs, n_channels, n_times)
        y: (n_epochs,) labels binaires 0/1
    """
    if dataset_name == "physionet":
        runs = kwargs.get('runs', [4, 8, 12])
        use_wavelet = kwargs.get('use_wavelet', False)
        return load_physionet(subject, runs, use_wavelet=use_wavelet)
    elif dataset_name == "bnci2014":
        return load_bnci2014(subject)
    else:
        raise ValueError(f"Dataset inconnu : {dataset_name}. "
                         f"Disponibles : physionet, bnci2014")


def get_n_subjects(dataset_name):
    """Retourne le nombre de sujets disponibles dans le dataset."""
    if dataset_name == "physionet":
        return 109
    elif dataset_name == "bnci2014":
        return 9
    else:
        raise ValueError(f"Dataset inconnu : {dataset_name}")
