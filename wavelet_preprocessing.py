import numpy as np
import pywt


def wavelet_bandpass(raw_data, sfreq=160, wavelet='db4', level=5):
    """
    Filtrage passe-bande par transformee en ondelettes discrete (DWT).
    Isole les bandes de frequence liees a l'activite motrice (mu: 8-13Hz, beta: 13-30Hz).

    Avec fs=160Hz et 5 niveaux de decomposition :
        - A5  : 0 - 2.5 Hz   -> supprime (drift)
        - D5  : 2.5 - 5 Hz   -> supprime
        - D4  : 5 - 10 Hz    -> supprime
        - D3  : 10 - 20 Hz   -> GARDE (rythme mu/alpha)
        - D2  : 20 - 40 Hz   -> GARDE (rythme beta)
        - D1  : 40 - 80 Hz   -> supprime (bruit haute frequence)

    Args:
        raw_data: np.ndarray de shape (n_channels, n_times)
        sfreq: frequence d'echantillonnage en Hz
        wavelet: type d'ondelette (defaut: Daubechies 4)
        level: nombre de niveaux de decomposition

    Returns:
        filtered: np.ndarray de meme shape que raw_data
    """
    n_channels, n_times = raw_data.shape
    filtered = np.zeros_like(raw_data)

    for ch in range(n_channels):
        # Decomposition en ondelettes
        coeffs = pywt.wavedec(raw_data[ch], wavelet, level=level, mode='symmetric')

        # coeffs = [cA5, cD5, cD4, cD3, cD2, cD1]
        # On met a zero les bandes non pertinentes
        coeffs[0] = np.zeros_like(coeffs[0])   # A5 : 0-2.5 Hz
        coeffs[1] = np.zeros_like(coeffs[1])   # D5 : 2.5-5 Hz
        coeffs[2] = np.zeros_like(coeffs[2])   # D4 : 5-10 Hz
        # coeffs[3] = D3 : 10-20 Hz -> on GARDE
        # coeffs[4] = D2 : 20-40 Hz -> on GARDE
        coeffs[5] = np.zeros_like(coeffs[5])   # D1 : 40-80 Hz

        # Reconstruction du signal filtre
        reconstructed = pywt.waverec(coeffs, wavelet, mode='symmetric')
        filtered[ch] = reconstructed[:n_times]

    return filtered
