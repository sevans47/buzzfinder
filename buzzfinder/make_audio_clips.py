import librosa
import soundfile as sf
import os
import glob
from pydub import AudioSegment
from pydub.playback import play

from buzzfinder.const import ROOT_DIR, SAMPLES_TO_CONSIDER

AUDIO_PATH = os.path.join(ROOT_DIR, 'audio')
RAW_AUDIO_PATH = os.path.join(AUDIO_PATH, "bf_raw_data")
DATASET_PATH = os.path.join(AUDIO_PATH, "buzz_finder_audio", "temp_audio_clips")

def get_audio_clips(raw_audio_filename, raw_audio_path=RAW_AUDIO_PATH, dataset_path=DATASET_PATH, pps=100, pp_delta=0.5, pp_wait=10, counter_start=1):
    """
    Take a long audio file, identify when notes are played, and save 2 second clip
    of that note to a folder.

    Arguments:
    - raw_audio_filename    (string): name of raw audio file (e.g. buzzy.m4a)
    - raw_audio_path        (string): path to directory containing long audio file
    - dataset_path          (string): path to folder for saving clips
    - pps                      (int): peak pick samples (for librosa.util.peak_pick()) - how many samples left and
                                      right of each sample to check for max and mean values
    - pp_delta               (float): delta argument for librosa.util.peak_pick()
    - pp_wait                  (int): wait argument for librosa.util.peak_pick()
    - counter_start             (int): number to start counting clips for file names
    """

    # check that dataset_path is a folder
    if os.path.isdir(dataset_path) == False:
        raise Exception("dataset_path must lead to a directory.")
        return 1

    # load audio
    audio_path = os.path.join(raw_audio_path, raw_audio_filename)
    signal, sr = librosa.load(audio_path)

    # get onset envelope from signal
    o_env = librosa.onset.onset_strength(y=signal, sr=sr)

    # find peaks in onset envelope
    onset_frames = librosa.util.peak_pick(o_env, pre_max=pps, post_max=pps, pre_avg=pps, post_avg=pps, delta=pp_delta, wait=pp_wait)

    # convert peaks from frame numbers to sample numbers
    onset_samples = librosa.frames_to_samples(onset_frames)

    # loop through onset_samples and save 2 second clip for each onset from original signal
    pre_onset = int(sr * 0.1)
    post_onset = int(sr * 2 - pre_onset)

    if pre_onset + post_onset != SAMPLES_TO_CONSIDER:
        raise Exception(f"Audio clip length ({pre_onset + post_onset}) does not equal correct number of samples for model ({SAMPLES_TO_CONSIDER})")
        return 2

    # import pdb; pdb.set_trace()
    raw_audio_name = os.path.basename(raw_audio_path)
    # raw_audio_name = raw_audio_path.split("/")[-1].split(".")[0]
    counter = counter_start
    for sample in onset_samples:
        clip = signal[sample - pre_onset: sample + post_onset]
        clip_filename = raw_audio_name + str(counter) + ".wav"
        clip_path = os.path.join(dataset_path, clip_filename)
        sf.write(clip_path, clip, sr)
        counter += 1


def check_audio_clips(dataset_path=DATASET_PATH):
    """
    Check each audio clip created with get_audio_clips, and prompt user to decide if
    it's ok 'y' or not 'n'.  Good clips will be moved to official folder 'buzz_finder_audio',
    and bad clips will be deleted.

    Arguments:
    - dataset_path  (string): path to directory containing audio clips obtained from "get_audio_clips" function
    """
    # get list of file paths to audio clips
    clip_paths = glob.glob(dataset_path+"/*")

    # loop through clips, playing each one, and checking with user whether they're ok or not
    good_files = []
    bad_files = []

    for file in clip_paths:
        clip = AudioSegment.from_wav(file)
        play(clip)
        print('ok? (y / n / r[eplay])')
        x = input().lower()
        while x == 'replay' or x == 'r':
            play(clip)
            print('ok? (y / n / replay)')
            x = input()
        if x == "y" or x == "yes":
            good_files.append(file)
        else:
            bad_files.append(file)

    print(f"Finished! {len(good_files)} clips ok, {len(bad_files)} clips not ok")

    # move good clips to official folder.
    # NOTE: filename should be 6 chars + num, and must have either "clean" or "buzzy" in the name (eg "cleanf32.wav")
    # TODO: this loop should use the os library instead of split / join
    counter = 1
    for file in good_files:
        path = file.split("/")
        root_folders = "/".join(path[:2]) + "/"
        folder_name = "clean/" if "clean" in path[-1] else "buzzy/"
        os.rename(file, f"{root_folders}{folder_name}{path[-1][:6]}{counter}.wav")
        counter += 1
    print("Good clips moved to official dataset folder")

    # delete bad clips
    bad_files = glob.glob(dataset_path+"/*")
    if len(bad_files) > 0:
        [os.remove(file) for file in bad_files]
    print("Bad clips deleted")


if __name__ == "__main__":
    print(ROOT_DIR)
    print(RAW_AUDIO_PATH)
    print(DATASET_PATH)
