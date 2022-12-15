# buzzfinder
buzzfinder is an API that uses a deep learning CNN model to identify whether a note
played on the guitar has a clean, buzzy, or muted tone.  It takes a 2 second audio clip of a single guitar note as input, and outputs the predicted tone.

### purpose of buzzfinder

I plan to use the buzzfinder API for two future projects:

- *Identify all the buzzy or muted notes in a recording.*

This would help a guitarist to more quickly locate trouble spots, and help bring awareness to their playing.

- *Train a guitarist to better control their finger pressure.*

If a guitarist pushes on the strings too hard, they risk finger pain and possible injury. It also hurts their technique as they're less able to move smoothly and freely. A great exercise to remedy this problem is to push on the string lightly enough to play a buzzy note.  Then, push down just a bit more to make it a clean note, but using the least amount of pressure. The buzzfinder API could be used to teach new guitarists this important exercise.

# About creating the buzzfinder API
### data creation
There were several steps to create data for the model:
- buzzy, clean, and muted notes were recorded on the guitar (nearly 900 in total)
- two second clips of each note were extracted from the raw audio using librosa's onset detection, then organized, and saved
- all the clips were split into train / validate / test sets, and each clip in the train set was augmented using audiomentations' compose function (about 2600 clips in total)
- each clip was converted to mfcc data using librosa and saved in a json file

### model building
I made buzzfinder's model using a convolutional neural network that acheived 95.8% accuracy when classifying tones from my test set.
I created the model using TensorFlow's Keras library.  It has 3 convolusional layers, each one with normalization and max pooling layers. Next is a dense layer, followed by a final dense softmax output layer with three outputs for buzzy, clean, or muted predictions.

### API
The API's classify function takes a numpy array of a two second audio clip's mfccs, and returns a numpy array of predictions for each tone type.

### tools used
- librosa and numpy: process audio data from raw audio into comprehensive mfccs
- audiomentations: augment training data to make the model more robust
- tensorflow: build, train, and evaluate the CNN model
- mlflow: track and save models and evaluations
- bentoml: serve model over an HTTP API endpoint and create a docker container
- google cloud platform (gcp): deploy api using container registry and cloud run

# Documentation

### Installation
Create a python3 virtualenv and activate it:
```
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:
```
git clone git@github.com:sevans47/buzzfinder.git
cd buzzfinder
pip install -r requirements.txt
make clean install test
```

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
python:
```
from buzzfinder.train_model import main
main(use_mlflow=False)
```

### How to save model and results locally using MLflow
cli:
```
mkdir mlruns
make mlflow_create_experiment
make mlflow_launch_tracking_server
```

python:
```
from buzzfinder.train_model import main
main(use_mlflow=True)
```

### How to run tests
cli:
`make test`
