# How to install
pip install BuzzFinder

# How to get audio clips from longer audio
from BuzzFinder.make_audio_clips import get_audio_clips, check_audio_clips
get_audio_clips(raw_audio_filename)

# How to reproduce results
from BuzzFinder.testing import play_clip
play_clip()

# How to run tests
make tests
