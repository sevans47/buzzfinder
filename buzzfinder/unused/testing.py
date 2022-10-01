from pydub import AudioSegment
from pydub.playback import play
import os
import sys
import random
from buzzfinder.const import ROOT_DIR

def play_clip():
    clip_path = os.path.join(ROOT_DIR, 'audio', 'buzz_finder_audio', 'clean', 'clean40.wav')
    clip = AudioSegment.from_wav(clip_path)
    play(clip)
    print("clip played using pydub")

def play_random_clip():
    s_type = random.choice(['clean', 'buzzy'])

    if s_type == 'clean':
        clean_path = os.path.join(ROOT_DIR, 'audio', 'buzz_finder_audio', 'clean')
        clean_file = random.choice(os.listdir(clean_path))
        random_path = os.path.join(clean_path, clean_file)

    else:
        buzzy_path = os.path.join(ROOT_DIR, 'audio', 'buzz_finder_audio', 'buzzy')
        buzzy_file = random.choice(os.listdir(buzzy_path))
        random_path = os.path.join(buzzy_path, buzzy_file)

    clip = AudioSegment.from_wav(random_path)
    play(clip)
    print("random clip played using pydub")

if __name__ == "__main__":
    import pdb; pdb.set_trace()
    if len(sys.argv) == 1:
        mlf = True
    elif len(sys.argv) == 2 and sys.argv[1] == str(1):
        mlf = True
    elif len(sys.argv) == 2 and sys.argv[1] == str(0):
        mlf = False
    elif len(sys.argv) > 2:
        mlf = "Too many command line arguments (expected 1)"
    else:
        mlf = "Please add correct command line argument (0 for no mlflow, 1 for mlflow)"

    print(mlf)
    # play_random_clip()
