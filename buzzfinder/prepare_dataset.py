import librosa
import glob
import os
import numpy as np
import json
import random
import time
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, HighPassFilter, LowPassFilter, Gain
from buzzfinder.const import ROOT_DIR, SAMPLES_TO_CONSIDER, N_MFCC, HOP_LENGTH, N_FFT, N_MELS

timestr = time.strftime("%Y%m%d-%H%M")
FILE_NAME = f"comprehensive_mfccs_all_5to1_{timestr}"

DATASET_PATH = os.path.join(ROOT_DIR, "audio/buzz_finder_audio/")
JSON_PATH = os.path.join(ROOT_DIR, "data", f"{FILE_NAME}.json")

def check_for_augmented_files(all_files):
    """Check if there are any augmented files in dataset_path"""
    original_files = [file for file in all_files if file.split("/")[-1][:3] != "aug"]
    aug_files = list(set(all_files) - set(original_files))
    return len(aug_files) > 0

def delete_augmented_files(all_files):
    """Deletes all augmented files in dataset_path"""
    original_files = [file for file in all_files if file.split("/")[-1][:3] != "aug"]
    aug_files = list(set(all_files) - set(original_files))
    if len(aug_files) > 0:
        [os.remove(file) for file in aug_files]

def split_train_test_val(train_test_val_split, dataset_path):
    """Split audio files into train, test, and validation sets"""

    # destructure train_test_val list
    train_pct, test_pct, val_pct = train_test_val_split

    # get list of all files
    buzzy_files = glob.glob(os.path.join(dataset_path, "buzzy", "*"))
    clean_files = glob.glob(os.path.join(dataset_path, "clean", "*"))
    muted_files = glob.glob(os.path.join(dataset_path, "muted", "*"))

    # split data for creating augmented data later (any previously augmented data has been deleted)
    buzzy_train = random.sample(buzzy_files, int(len(buzzy_files) * train_pct))
    clean_train = random.sample(clean_files, int(len(clean_files) * train_pct))
    muted_train = random.sample(muted_files, int(len(muted_files) * train_pct))


    buzzy_test_val = list(set(buzzy_files) - set(buzzy_train))
    clean_test_val = list(set(clean_files) - set(clean_train))
    muted_test_val = list(set(muted_files) - set(muted_train))

    buzzy_test = random.sample(buzzy_test_val, int(len(buzzy_files) * test_pct))
    clean_test = random.sample(clean_test_val, int(len(clean_files) * test_pct))
    muted_test = random.sample(muted_test_val, int(len(muted_files) * test_pct))

    buzzy_val = list(set(buzzy_test_val) - set(buzzy_test))
    clean_val = list(set(clean_test_val) - set(clean_test))
    muted_val = list(set(muted_test_val) - set(muted_test))

    train_files = buzzy_train + clean_train + muted_train
    test_files = buzzy_test + clean_test + muted_test
    val_files = buzzy_val + clean_val + muted_val

    return train_files, test_files, val_files


def split_test_val(train_test_val_split, dataset_path):
    """
    Split audio clips into test and validation sets only.  This function is for when augmented data has already been
    created and won't be remade, so the train dataset won't be changed (as augmented data in test and val datasets
    would create data leakage)
    """

    # destructure train_test_val list
    train_pct, test_pct, val_pct = train_test_val_split

    # change test percent so it disregards train_pct
    test_pct_aug = test_pct / (test_pct + val_pct)

    # get list of all files
    buzzy_files = glob.glob(os.path.join(dataset_path, "buzzy", "*"))
    clean_files = glob.glob(os.path.join(dataset_path, "clean", "*"))
    muted_files = glob.glob(os.path.join(dataset_path, "muted", "*"))
    all_files = buzzy_files + clean_files

    # create list of training data filepaths "train_files"
    aug_dict = {"aug_files": [],
                "aug_index": []}

    for i, file in enumerate(all_files):
        if file.split("/")[-1].startswith("aug"):
            aug_dict['aug_files'].append(file.split("-")[-1])
            aug_dict['aug_index'].append(i)

    aug_dict['aug_files'] = list(set(aug_dict['aug_files']))

    for i, file in enumerate(all_files):
        if file.split("/")[-1] in aug_dict['aug_files']:
            aug_dict['aug_index'].append(i)

    train_files = [all_files[i] for i in aug_dict['aug_index']]

    # split remaining data into test and validation data
    test_val_files = list(set(all_files) - set(train_files))

    test_files = random.sample(test_val_files, int(len(test_val_files) * test_pct_aug))
    val_files = list(set(test_val_files) - set(test_files))

    return train_files, test_files, val_files


def augment_training_data(training_data_files, n_augmentations_per_file):
    """
    Augment the training data and save the audio files to dataset_path

    Arguments:
    - training_data_files   (string): list of filepaths to training audio data
    """

    # augment1 - for audio that's 3 or more seconds - time stretch can be faster (ie max_rate > 1)
    augment1 = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.005, p=0.8),
        Gain(min_gain_in_db=-12, max_gain_in_db=12, p=0.8),
        HighPassFilter(min_cutoff_freq=400, max_cutoff_freq=800, p=0.8),
        LowPassFilter(min_cutoff_freq=6000, max_cutoff_freq=8000, p=0.8)
    ])

    # augment2 - for audio that's 2 seconds
    augment2 = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.005, p=0.8),
        Gain(min_gain_in_db=-12, max_gain_in_db=12, p=0.8),
        HighPassFilter(min_cutoff_freq=400, max_cutoff_freq=800, p=0.8),
        LowPassFilter(min_cutoff_freq=6000, max_cutoff_freq=8000, p=0.8)
    ])

    # loop through files and augment data
    augmented_data_files = []
    counter = 0
    for file in training_data_files:

        # load file
        signal, sr = librosa.load(file)

        # create i number of augmentations for file
        for i in range(n_augmentations_per_file):

            # choose correct augmentation function for length of signal
            if signal.size == sr*2:
                augmented_signal = augment2(signal, sr)
            else:
                augmented_signal = augment1(signal, sr)

            # rename the file
            # TODO: should use os library instead of split / join
            aug_filepath = file.split("/")
            aug_filepath[-1] = f"aug{counter}-{aug_filepath[-1][:-4]}.wav"
#             print(aug_filepath)
            aug_filepath = "/".join(aug_filepath)

            augmented_data_files.append(aug_filepath)

            # save augmented file
            sf.write(aug_filepath, augmented_signal, sr)

            counter += 1

            if counter % 50 == 0:
                print(f"{' '*10}{counter} augmented files complete")

    print(f"{len(augmented_data_files)} created from {len(training_data_files)} original files")
    return training_data_files + augmented_data_files


# TODO: fix 'waveform' datatype
def main(dataset_path=DATASET_PATH, json_path=JSON_PATH, datatype='comprehensive_mfccs', train_test_val_split=[0.4, 0.3, 0.3], create_augmented_data=False, n_augmentations_per_file=5, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT, n_mels=N_MELS):
    """
    A function that loads audio clips from dataset_path, splits clips into train, test, and validation sets, transforms
    the clips into the desired datatype, and saves the data as a .json file

    Arguments:
    - dataset_path          (string): path to parent directory of buzzy and clean audio clip directories
    - json_path             (string): path to desired location to save dataset as a .json file
    - datatype              (string): desired datatype to transform the audio clips to.  Choose mfccs, comprehensive_mfccs,
                                    waveform (TBD), spectrogram, mel_spectrogram
    - train_test_val_split    (list): list of percentages for splitting data into train, test, and validation sets.  Sum must equal 1.0
    - create_augmented_data   (bool): if True, deletes any current augmented data and creates new augmented data
    - n_augmentations_per_file (int): how many augmented files to create from one original file
    """

    # ensure valid data type has been selected
    if datatype not in ['mfccs', 'comprehensive_mfccs', 'waveform', 'spectrogram', 'mel_spectrogram']:
        raise Exception("Please select a valid datatype: mfccs, comprehensive_mfccs, waveform, spectrogram, mel_spectrogram")
        return 1

    # ensure train_test_val_split is equal to 1
    if sum(train_test_val_split) != 1.0:
        raise Exception("Please ensure the sume of train_test_val_split is 1.0")
        return 2

    # data dictionary
    data_dict = {
        "mappings": ["buzzy", "clean", "muted"],
        "train_labels": [],
        "train_data": [],
        "val_labels": [],
        "val_data": [],
        "test_labels": [],
        "test_data": [],
        "train_files": [],
        "val_files": [],
        "test_files": []
    }

    # create list of all buzzy and clean filepaths
    buzzy_files = glob.glob(os.path.join(dataset_path, "buzzy", "*"))
    clean_files = glob.glob(os.path.join(dataset_path, "clean", "*"))
    muted_files = glob.glob(os.path.join(dataset_path, "muted", "*"))
    all_files = buzzy_files + clean_files + muted_files

    if create_augmented_data:

        # delete previous augmented files
        print("Deleting previous augmented files...")
        delete_augmented_files(all_files)

        # split data into train, test, and validation
        train_files_pre_augment, test_files, val_files = split_train_test_val(train_test_val_split, dataset_path)

        # augment training data
        print("Creating new augmented files...")
        train_files = augment_training_data(train_files_pre_augment, n_augmentations_per_file)


    else:

        # check if there are any augmented files
        if check_for_augmented_files(all_files) == False:

            # split data as normal if no augmented files are found
            print("No augmented files found")
            train_files, test_files, val_files = split_train_test_val(train_test_val_split, dataset_path)

        else:

            # keep augmented files and create train, test, and val data
            print("Augmented files found")
            train_files, test_files, val_files = split_test_val(train_test_val_split, dataset_path)


    all_files = [train_files, test_files, val_files]

    # loop through train data and extract desired datatype:
    for i, split in enumerate(all_files):
        print(f"preparing data: {i+1} of 3")

        for file in split:

            # load audio file
            signal, sr = librosa.load(file)

            # ensure the audio file is at least 2 sec
            if len(signal) >= SAMPLES_TO_CONSIDER:

                # enforce 2 sec. long signal
                signal = signal[:SAMPLES_TO_CONSIDER]

                # get label
                # label = 1 if "clean" in file.split("/")[-1] else 0
                # label = 1 if "clean" in os.path.basename(file) else 0
                if "buzzy" in os.path.basename(file):
                    label = 0
                elif "clean" in os.path.basename(file):
                    label = 1
                elif "muted" in os.path.basename(file):
                    label = 2
                else:
                    continue


                # extract datatype
                if datatype == 'mfccs':
                    data = librosa.feature.mfcc(y=signal, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)  # MFCCs

                elif datatype == 'comprehensive_mfccs':
                    # extract the MFCCs
                    MFCCs = librosa.feature.mfcc(y=signal, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)

                    # calculate delta and delta delta
                    delta_mfccs = librosa.feature.delta(MFCCs)
                    delta2_mfccs = librosa.feature.delta(MFCCs, order=2)

                    # concatenate MFCCs, delta MFCCs, and delta delta MFCCs
                    data = np.concatenate([MFCCs, delta_mfccs, delta2_mfccs])  # comprehensive MFCCs

                elif datatype == "waveform":
                    data = signal

                elif datatype == "spectrogram":
                    s = librosa.stft(y=signal, n_fft=n_fft, hop_length=hop_length)
                    y = np.abs(s) ** 2

                    # move from power representation of amplitude (linear) to decibels (logarithmic)
                    data = librosa.power_to_db(y)  # spectrogram

                else:
                    data = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

                # store data
                if i == 0:
                    data_dict["train_labels"].append(label)
                    data_dict["train_data"].append(data.T.tolist())  # data shape = (frames, MFCCs)
                    data_dict["train_files"].append(file)

                elif i == 1:
                    data_dict["test_labels"].append(label)
                    data_dict["test_data"].append(data.T.tolist())  # data shape = (frames, MFCCs)
                    data_dict["test_files"].append(file)

                else:
                    data_dict["val_labels"].append(label)
                    data_dict["val_data"].append(data.T.tolist())  # data shape = (frames, MFCCs)
                    data_dict["val_files"].append(file)

    # save as json
    with open(json_path, "w") as fp:
        json.dump(data_dict, fp, indent=4)
    print("Data saved as .json")
    print("Complete")

if __name__ == "__main__":
    # main(create_augmented_data=True, n_augmentations_per_file=5)

    # delete augmented data
    buzzy_files = glob.glob(os.path.join(DATASET_PATH, "buzzy", "*"))
    clean_files = glob.glob(os.path.join(DATASET_PATH, "clean", "*"))
    muted_files = glob.glob(os.path.join(DATASET_PATH, "muted", "*"))
    all_files = buzzy_files + clean_files + muted_files
    delete_augmented_files(all_files)
