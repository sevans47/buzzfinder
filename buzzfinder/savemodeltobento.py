"""This module saves a Keras model to BentoML"""

import os
from buzzfinder.const import ROOT_DIR
from tensorflow import keras
import bentoml

def load_model_and_save_to_bento(model_file) -> None:
    """Loads a keras model from disk and saves it to BentoML."""
    model = keras.models.load_model(model_file)
    bento_model = bentoml.keras.save_model("buzzfinder_model", model)
    print(f"Bento model tag = {bento_model.tag}")

if __name__ == "__main__":
    model_file = os.path.join(ROOT_DIR, "data", "model.h5")
    load_model_and_save_to_bento(model_file)
