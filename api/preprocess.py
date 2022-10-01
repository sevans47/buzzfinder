from librosa import load
from librosa.feature import mfcc, delta
import numpy as np
from const import SAMPLES_TO_CONSIDER, N_MFCC, N_FFT, HOP_LENGTH


def preprocess(file_path, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH):

    # load audio file
    signal, sr = load(file_path)

    # ensure consistency in the audio file length
    if len(signal) > SAMPLES_TO_CONSIDER:
        signal = signal[:SAMPLES_TO_CONSIDER]

    # extract the MFCCs
    MFCCs = mfcc(y=signal, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)

    # calculate delta and delta delta
    delta_mfccs = delta(MFCCs)
    delta2_mfccs = delta(MFCCs, order=2)

    # concatenate MFCCs, delta MFCCs, and delta delta MFCCs
    data = np.concatenate([MFCCs, delta_mfccs, delta2_mfccs]).T  # comprehensive MFCCs

    # # transpose data
    # data = data.T

    # convert 2d MFCCs array into 4d array -> (# samples, # segments, # coefficients, # channels)
    data = data[np.newaxis, ..., np.newaxis]

    return data
