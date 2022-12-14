# buzzfinder
buzzfinder is an api that uses a deep learning CNN model to identify whether a note
played on the guitar has a clean, buzzy, or muted tone.  It takes a 2 second audio clip of a single guitar note as input, and outputs the predicted tone.

# data creation
The data was collected by recording several hundred notes on the guitar as either buzzy, clean, or muted.  Examples of each note type were saved as 2 second audio clips, which were converted to mfccs and saved in a json file.  Further data was created via data
augmentation using the audiomentations python library.

# How to install
`pip install buzzfinder`

# How to get audio clips from longer audio
```
from buzzfinder.make_audio_clips import get_audio_clips, check_audio_clips
get_audio_clips(raw_audio_filename)
```

# How to create dataset as a json file from audio clips
```
from buzzfinder.prepare_dataset import main
main()
```
*note* the default datatype to convert audio clips to is comprehensive mfccs

# How to train a model

# How to save model and results locally using MLflow

# How to create api using BentoML

# How to reproduce results
from buzzfinder.testing import play_clip
play_clip()

# How to run tests
make tests
