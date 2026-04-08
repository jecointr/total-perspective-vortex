import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws
import matplotlib.pyplot as plt

def load_and_preprocess(subject=1, runs=[4, 8, 12]):
    """
    Charge les données pour un sujet et des runs spécifiques.
    Ex: Runs 4, 8, 12 = Imagined movement (Left/Right hand)
    """
    # 1. Téléchargement des fichiers
    raw_fnames = eegbci.load_data(subject, runs)
    raw = concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in raw_fnames])
    
    # Nettoyage des noms de channels (ex: 'Fc5.' -> 'FC5')
    eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage)

    # 2. Visualisation du signal BRUT
    print("Affichage du signal brut...")
    raw.plot(n_channels=10, title="Raw EEG Data")

    # 3. FILTRAGE (Crucial pour la partie obligatoire V.1.1)
    # On garde les fréquences entre 8 et 35 Hz
    raw.filter(8., 35., fir_design='firwin', skip_by_annotation='edge')

    # 4. Visualisation du signal FILTRÉ
    print("Affichage du signal filtré...")
    raw.plot(n_channels=10, title="Filtered EEG Data (8-35Hz)")
    
    plt.show()
    return raw

if __name__ == "__main__":
    # Test sur le sujet 1
    raw_data = load_and_preprocess(subject=1)
