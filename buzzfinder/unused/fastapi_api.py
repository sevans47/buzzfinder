from fastapi import FastAPI
import os
import numpy as np
from buzzfinder.const import ROOT_DIR
from buzzfinder.predict import Buzz_Finder_Service

# TEST_AUDIO_FILE_PATH = os.path.join(ROOT_DIR, "audio", "buzz_finder_audio", "test_audio", "test_buzzy1.wav")

app = FastAPI()

@app.get('/')
def greeting():
    return {'greeting':'welcome to the buzzfinder api!'}

@app.get('/predict')
def predict(audio_file):

    # invoke buzz finder service
    bfs = Buzz_Finder_Service()

    # make a prediction
    predicted_tone = bfs.predict(audio_file)

    # send the predicted keyword back
    data = {"tone": predicted_tone}
    return data
