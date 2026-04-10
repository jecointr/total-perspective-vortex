import mne
import numpy as np
from mne.datasets import eegbci
from mne.io import concatenate_raws

def load_and_epoch_data(subject, runs, tmin=-1.0, tmax=4.0, use_wavelet=False):
    """
    Charge, filtre et découpe les données EEG pour un sujet donné.
    
    Args:
        subject (int): Numéro du sujet (1 à 109).
        runs (list): Liste des runs à charger (ex: [4, 8, 12] pour l'imagination motrice).
        tmin (float): Début de l'epoch par rapport au stimulus (en secondes).
        tmax (float): Fin de l'epoch par rapport au stimulus (en secondes).
        
    Returns:
        X (np.ndarray): Matrice des signaux de forme (n_epochs, n_channels, n_times).
        y (np.ndarray): Vecteur des labels des classes.
    """
    # 1. Chargement des données brutes
    raw_fnames = eegbci.load_data(subject, runs)
    raw = concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in raw_fnames])
    
    # Standardisation des noms des électrodes (channels)
    eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage)

    # 2. Filtrage spatial et fréquentiel (V.1.1)
    # On isole les fréquences liées au mouvement (rythmes Mu et Beta)
    if use_wavelet:
        from wavelet_preprocessing import wavelet_bandpass
        annotations = raw.annotations
        raw_data = raw.get_data()
        filtered_data = wavelet_bandpass(raw_data, sfreq=raw.info['sfreq'])
        raw = mne.io.RawArray(filtered_data, raw.info)
        raw.set_annotations(annotations)
    else:
        raw.filter(8., 35., fir_design='firwin', skip_by_annotation='edge')

    # 3. Extraction des événements
    # T0 (repos), T1 (action/imagination gauche), T2 (action/imagination droite)
    events, event_dict = mne.events_from_annotations(raw)
    
    # On ne garde que les événements d'action (T1 et T2)
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    
    # 4. Epoching (Découpage)
    epochs = mne.Epochs(raw, events, event_id=[2, 3], tmin=tmin, tmax=tmax, 
                        proj=True, picks=picks, baseline=None, preload=True)
    
    # 5. Extraction des matrices pour Sklearn
    X = epochs.get_data() # Format : (epochs, channels, time)
    y = epochs.events[:, -1] - 2 # Transformation des labels 2,3 en 0,1

    return X, y

if __name__ == "__main__":
    # Test unitaire rapide du fichier
    print("Test de preprocessing.py sur le Sujet 1, Runs [4, 8, 12]...")
    X, y = load_and_epoch_data(subject=1, runs=[4, 8, 12])
    print(f"Shape de X (données) : {X.shape}")
    print(f"Shape de y (labels)  : {y.shape}")
