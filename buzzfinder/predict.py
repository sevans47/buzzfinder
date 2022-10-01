import tensorflow.keras as keras
import numpy as np
import librosa
import os
from buzzfinder.const import ROOT_DIR, SAMPLES_TO_CONSIDER, N_MFCC, HOP_LENGTH, N_FFT

MODEL_PATH = os.path.join(ROOT_DIR, "data", "model.h5")

class _Buzz_Finder_Service:

    model = None
    _mappings = [
        "buzzy",
        "clean"
    ]
    _instance = None


    def predict(self, file_path):

        # extract MFCCs
        MFCCs = self.preprocess(file_path)  # (# segments, # coefficients)

        ### do this step in self.preprocess()
        # # convert 2d MFCCs array into 4d array -> (# samples, # segments, # coefficients, # channels)
        # MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # make prediction
        predictions = self.model.predict(MFCCs) # [ [0.2, 0.6] ] - 2d array with predictions for buzzy / clean
        predicted_index = np.argmax(predictions)
        predicted_value = self._mappings[predicted_index]

        return predicted_value


    def preprocess(self, file_path, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH):

        # load audio file
        signal, sr = librosa.load(file_path)

        # ensure consistency in the audio file length
        if len(signal) > SAMPLES_TO_CONSIDER:
            signal = signal[:SAMPLES_TO_CONSIDER]

        # extract the MFCCs
        MFCCs = librosa.feature.mfcc(y=signal, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)

        # calculate delta and delta delta
        delta_mfccs = librosa.feature.delta(MFCCs)
        delta2_mfccs = librosa.feature.delta(MFCCs, order=2)

        # concatenate MFCCs, delta MFCCs, and delta delta MFCCs
        data = np.concatenate([MFCCs, delta_mfccs, delta2_mfccs])  # comprehensive MFCCs

        ### can end here if adding axis in self.predict instead of here
        # return data.T

        # transpose data
        data = data.T

        # convert 2d MFCCs array into 4d array -> (# samples, # segments, # coefficients, # channels)
        data = data[np.newaxis, ..., np.newaxis]

        return data





def Buzz_Finder_Service():

    # ensure that we only have 1 instance of BFS
    if _Buzz_Finder_Service._instance is None:
        _Buzz_Finder_Service._instance = _Buzz_Finder_Service()
        _Buzz_Finder_Service.model = keras.models.load_model(MODEL_PATH)
    return _Buzz_Finder_Service._instance


if __name__ == "__main__":

    bfs = Buzz_Finder_Service()

    test_audio_path = os.path.join(ROOT_DIR, "audio", "test_audio")
    tone1 = bfs.predict(os.path.join(test_audio_path, "test_buzzy1.wav"))
    tone2 = bfs.predict(os.path.join(test_audio_path, "test_clean1.wav"))

    print(f"Predicted tone: {tone1}, {tone2}")
