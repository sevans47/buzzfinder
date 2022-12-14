"""This module defines a BentoML service that uses a Keras model to classify
guitar tones.
"""

import numpy as np
import bentoml
from bentoml.io import NumpyNdarray


BENTO_MODEL_TAG = "buzzfinder_model:ab6pufd3kw67pueb"

# use runner to wrap model because it optimizes computation
classifier_runner = bentoml.keras.get(BENTO_MODEL_TAG).to_runner()

buzzfinder_service = bentoml.Service("buzzfinder_classifier", runners=[classifier_runner])

@buzzfinder_service.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_data: np.ndarray) -> np.ndarray:
    return classifier_runner.predict.run(input_data)
