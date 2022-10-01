import json
import os
import io
import random
import numpy as np
import requests
from api.preprocess import preprocess
from api.const import ROOT_DIR

SERVICE_URL = "http://localhost:3000/classify"

def sample_random_audio_clip():
    s_type = random.choice(['clean', 'buzzy'])

    if s_type == 'clean':
        clean_path = os.path.join(ROOT_DIR, 'audio', 'buzz_finder_audio', 'clean')
        clean_file = random.choice(os.listdir(clean_path))
        random_path = os.path.join(clean_path, clean_file)

    else:
        buzzy_path = os.path.join(ROOT_DIR, 'audio', 'buzz_finder_audio', 'buzzy')
        buzzy_file = random.choice(os.listdir(buzzy_path))
        random_path = os.path.join(buzzy_path, buzzy_file)

    return random_path, s_type


def make_request_to_bento_service(
    service_url: str, input_array: np.ndarray
) -> str:
    serialized_input_data = json.dumps(input_array.tolist())
    response = requests.post(
        service_url,
        data = serialized_input_data,
        # header={"content-type": "application/json"}
    )

    return response.text


def main():
    audio_clip_path, expected_output = sample_random_audio_clip()
    with open(audio_clip_path, "rb") as f:
        binary_clip = io.BytesIO(f.read())
    mfccs = preprocess(binary_clip)
    prediction = make_request_to_bento_service(SERVICE_URL, mfccs)
    print(f"Prediction: {prediction}")
    print(f"Expected output: {expected_output}")


if __name__ == "__main__":
    main()

    # audio_clip_path, expected_output = sample_random_audio_clip()
    # with open(audio_clip_path, "rb") as f:
    #     binary_clip = io.BytesIO(f.read())
    # bfs = Buzz_Finder_Service()
    # mfccs = bfs.preprocess(binary_clip)

    # print(type(binary_clip))
    # print(type(io.BytesIO(binary_clip)))
