"""
server

client -> POST request -> server -> prediction back to client

"""

from flask import Flask, request, jsonify
import random
from buzzfinder.predict import Buzz_Finder_Service
import os

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():

    # get audio file and save it
    audio_file = request.files["file"]
    file_name = str(random.randint(0, 100000))
    audio_file.save(file_name)

    # invoke buzz finder service
    bfs = Buzz_Finder_Service()

    # make a prediction
    predicted_tone = bfs.predict(file_name)

    # remove the audio file
    os.remove(file_name)

    # send back the predicted keyword in json format
    data = {"tone": predicted_tone}
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=False)
