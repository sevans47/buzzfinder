"""This module saves a Keras model to BentoML"""

import os
import sys
from buzzfinder.const import ROOT_DIR
from tensorflow import keras
import bentoml

def load_model_and_save_to_bento(model_file) -> None:
    """Loads a keras model from disk and saves it to BentoML."""
    model = keras.models.load_model(model_file)
    bento_model = bentoml.keras.save_model("buzzfinder_model", model)
    print(f"Bento model tag = {bento_model.tag}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        model_name = 'model.h5'
    model_name = sys.argv[1]
    try:
        model_file = os.path.join(ROOT_DIR, "data", model_name)
    except FileNotFoundError:
        print("""
              No file found. Try checking if the file name was included as a command
              line argument, and if the file is in the data directory.
              """
              )
    load_model_and_save_to_bento(model_file)
