# buzzfinder
buzzfinder is an api that uses a deep learning CNN model to identify whether a note
played on the guitar has a clean, buzzy, or muted tone.  It takes a 2 second audio clip of a single guitar note as input, and outputs the predicted tone.

# About the api
### data creation
There were several steps to create data for the model:
- buzzy, clean, and muted notes were recorded on the guitar (nearly 1000 in total)
- two second clips of each note were extracted from the raw audio using librosa's onset detection, then organized, and saved
- all the clips were split into train / validate / test sets, and each clip in the train set was augmented using audiomentations' compose function (about 2600 clips in total)
- each clip was converted to mfcc data using librosa and saved in a json file

### model building
I created a deep learning model using TensorFlow's Keras library.  It has 3 convolusional layers, each one with normalization and max pooling layers, followed by one dense layer, and a final dense output layer with three outputs for buzzy, clean, or muted predictions.  Using the test set, the resulting model acheived a loss score of 21.8%, and an accuracy score of 95.8%.

### API
The API's classify function takes a numpy array of a 2 second audio clip's mfccs, and returns a numpy array of predictions for each tone type.

### tools used
- librosa: process audio data from raw audio into comprehensive mfccs
- audiomentations: augment training data to make the model more robust
- tensorflow: build, train, and evaluate the CNN model
- mlflow: track and save models and evaluations
- bentoml: create api container
- google cloud platform (gcp): deploy api using container registry and cloud run

# Documentation

### Installation
`pip install buzzfinder`

### Getting 2 second audio clips from longer audio
python:
```
from buzzfinder.make_audio_clips import get_audio_clips, check_audio_clips
get_audio_clips(raw_audio_filename)
```

### How to create dataset as a json file from audio clips
python:
```
from buzzfinder.prepare_dataset import main
main()
```
*note* the default datatype to convert audio clips to is comprehensive mfccs

### How to train a model
```
from buzzfinder.train_model import main
main(use_mlflow=False)
```

### How to save model and results locally using MLflow
cli:
```
touch mlruns
mlflow_create_experiment
make mlflow_launch_tracking_server
```

python:
```
from buzzfinder.train_model import main
main(use_mlflow=True)
```

### How to run tests
make tests
