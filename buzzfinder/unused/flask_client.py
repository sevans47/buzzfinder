from buzzfinder.const import ROOT_DIR
import os
import requests

URL = "http://127.0.0.1:5050/predict"
TEST_AUDIO_FILE_PATH = os.path.join(ROOT_DIR, "audio", "buzz_finder_audio", "test_audio", "test_buzzy1.wav")

if __name__ == "__main__":

    audio_file = open(TEST_AUDIO_FILE_PATH, "rb")
    values = {"file": (TEST_AUDIO_FILE_PATH, audio_file, "audio/wav")}
    response = requests.post(URL, files=values)
    data = response.json()

    print(f"Predicted tone is: {data['tone']}")
