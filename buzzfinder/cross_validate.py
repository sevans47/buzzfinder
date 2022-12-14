"""
This module is designed to investigate which augmentation types have the biggest
impact on model performance.  It takes the audio clips, splits them into folds,
duplicate the data using the specified type of data augmentation, train
and evaluates the model using each fold as the test data, and finally saves the
results to a json file.
"""

import glob
import random
import librosa
import json
import os
from os.path import exists
import numpy as np
import matplotlib.pyplot as plt
from audiomentations import Compose, HighPassFilter#, LowPassFilter, Gain, AddGaussianNoise

# from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from buzzfinder.const import ROOT_DIR, SAMPLES_TO_CONSIDER, N_MFCC, HOP_LENGTH, N_FFT

DATASET_PATH = os.path.join(ROOT_DIR, 'audio', 'buzz_finder_audio')
JSON_PATH = os.path.join(ROOT_DIR, "data", "augmentation_results", "aug_eval.json")
SAVED_CV_PATH = os.path.join(ROOT_DIR, "data", "augmentation_results", "aug_learning_curves.png")
TYPE_OF_AUGMENTATION = "HighPassFilter 2"
AUG_PARAMETERS = "min_cutoff_freq=400, max_cutoff_freq=800"

LEARNING_RATE = 0.0001
EPOCHS = 50
BATCH_SIZE = 32
PATIENCE = 10

# TODO: run for Gain (already typed out everything, just need to run), add TYPE_OF_AUGMENTATION to charts?, set y_lim to 1.0?
augment = Compose([
        # AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.005, p=1.0),
        # Gain(min_gain_in_db=-12, max_gain_in_db=12, p=1.0),
        HighPassFilter(min_cutoff_freq=400, max_cutoff_freq=800, p=1.0),
        # LowPassFilter(min_cutoff_freq=6000, max_cutoff_freq=8000, p=1.0)
    ])

def get_data(dataset_path):
    buzzy_files = glob.glob(dataset_path+"/buzzy/*")
    clean_files = glob.glob(dataset_path+"/clean/*")
    muted_files = glob.glob(dataset_path+"/muted/*")
    all_files = buzzy_files + clean_files + muted_files
    random.shuffle(all_files)
    return all_files


def make_folds(all_files, cv):
    q, r = divmod(len(all_files), cv)
    indices = [q*i + min(i, r) for i in range(cv + 1)]
    folds = [all_files[indices[i]:indices[i+1]] for i in range(cv)]
    return folds


def make_comprehensive_MFCCs(signal, n_mfcc, hop_length, n_fft):
    MFCCs = librosa.feature.mfcc(y=signal, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)

    # calculate delta and delta delta
    delta_mfccs = librosa.feature.delta(MFCCs)
    delta2_mfccs = librosa.feature.delta(MFCCs, order=2)

    # concatenate MFCCs, delta MFCCs, and delta delta MFCCs
    return np.concatenate([MFCCs, delta_mfccs, delta2_mfccs])  # comprehensive MFCCs


def make_data_dict(folds, n_mfcc=13, hop_length=512, n_fft=2048):

    data_dict = {
        "mappings": ["buzzy", "clean", "muted"],
        "original_data": {},
        "original_labels": {},
        "augmented_data": {},
        "augmented_labels": {}
    }

    # loop through each fold
    for i, fold in enumerate(folds):

        # add new fold to data_dict
        print(f"Making MFCCs: fold {i + 1} of {len(folds)}...")
        data_dict['original_data'][f'fold_{i}'] = []
        data_dict['original_labels'][f'fold_{i}'] = []
        data_dict['augmented_data'][f'fold_{i}'] = []
        data_dict['augmented_labels'][f'fold_{i}'] = []

        # loop through each file and augment it, and add mfccs and labels to data dict
        for j, file in enumerate(fold):

            # load file
            signal, sr = librosa.load(file)

            if len(signal) >= SAMPLES_TO_CONSIDER:

                # enforce 2 sec. long signal
                signal = signal[:SAMPLES_TO_CONSIDER]

                # create augmented signal - TODO: create augment function
                augmented_signal = augment(signal, sr)

                # create MFCCs for original and augmented signals
                original_mfccs = make_comprehensive_MFCCs(signal, n_mfcc, hop_length, n_fft)
                augmented_mfccs = make_comprehensive_MFCCs(augmented_signal, n_mfcc, hop_length, n_fft)

                # get label
                # label = 1 if "clean" in file.split("/")[-1] else 0
                filename = file.split("/")[-1]
                if "buzzy" in filename:
                    label = 0
                elif "clean" in filename:
                    label = 1
                else:
                    label = 2


                # save to dict
                data_dict["original_data"][f"fold_{i}"].append(original_mfccs.T.tolist())
                data_dict["original_labels"][f"fold_{i}"].append(label)
                data_dict['augmented_data'][f'fold_{i}'].append(augmented_mfccs.T.tolist())
                data_dict['augmented_labels'][f'fold_{i}'].append(label)

    print("Finished preparing data")

    return data_dict


def build_model(input_shape, learning_rate, error="binary_crossentropy"):

    # build network
    model = Sequential()

    # conv layer 1
    model.add(Conv2D(64, (3, 3), activation="relu",
                                 input_shape=input_shape,
                                 kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((3, 3), strides=(2, 2), padding="same"))

    # conv layer 2
    model.add(Conv2D(32, (3, 3), activation="relu",
                                 kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((3, 3), strides=(2, 2), padding="same"))

    # conv layer 3
    model.add(Conv2D(32, (2, 2), activation="relu",
                                 kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=(2, 2), padding="same"))

    # flatten the output and feed it into a dense layer
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))

#     # sigmoid classifier
#     model.add(Dense(1, activation="sigmoid"))

#     # compile the model
#     optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
#     model.compile(optimizer=optimizer, loss=error, metrics=["accuracy"])

    # if doing softmax:
    model.add(Dense(3, activation="softmax"))  #[0.1, 0.7, 0.2]
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#     # print model overview
#     model.summary()

    return model


def build_and_train_model(cv, data_dict):

    evaluations = []
    histories = []

    for i in range(cv):
        print(f"Training model for fold {i+1} of {cv}...")
        # get test data
        X_test = np.array(data_dict["original_data"][f"fold_{i}"])
        y_test = np.array(data_dict["original_labels"][f"fold_{i}"])

        # get train data
        original_data = [data for key, data in sorted(data_dict['original_data'].items()) if key.split("_")[-1] != str(i)]
        augmented_data = [data for key, data in sorted(data_dict['augmented_data'].items()) if key.split("_")[-1] != str(i)]
        original_labels = [data for key, data in sorted(data_dict['original_labels'].items()) if key.split("_")[-1] != str(i)]
        augmented_labels = [data for key, data in sorted(data_dict['augmented_labels'].items()) if key.split("_")[-1] != str(i)]

        X_train = np.concatenate(original_data + augmented_data)
        y_train = np.concatenate(original_labels + augmented_labels)

        # convert inputs from 3d to 4d arrays - new dimension is num of channels (1)
        X_test = X_test[..., np.newaxis]
        X_train = X_train[..., np.newaxis]

        # build model
        input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])  # (# frames, # coefficients (39), # channels (1))
        model = build_model(input_shape, LEARNING_RATE)

        # train model
        # es = EarlyStopping(patience=PATIENCE)
        history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)#, callbacks=[es])
        histories.append(history)

        # evaluate the model
        test_error, test_accuracy = model.evaluate(X_test, y_test)
        evaluations.append([test_error, test_accuracy])
#         print(f"Test error: {test_error}, test accuracy: {test_accuracy}")

    print("Finished training all models")

    return evaluations, histories


def make_loss_acc_plot(evaluations, histories):

    # get loss and accuracy scores for each fold training
    losses = np.array([history.history['loss'] for history in histories])
    accs = np.array([history.history['accuracy'] for history in histories])

    # aggregate loss scores
    mean_losses = losses.mean(axis=0)
    max_losses = losses.max(axis=0)
    min_losses = losses.min(axis=0)

    # aggregate accuracy scores
    mean_accs = accs.mean(axis=0)
    max_accs = accs.max(axis=0)
    min_accs = accs.min(axis=0)

    # average loss and accuracy evaluation scores
    mean_loss = np.array(evaluations)[:,0].mean()
    mean_acc = np.array(evaluations)[:,1].mean()

    # make plot
    plt.figure(figsize=(10, 6))

    plt.plot(mean_losses, label='loss')
    plt.plot(mean_accs, label='accuracy')
    plt.fill_between(range(len(max_losses)), max_losses, min_losses, alpha=0.3)
    plt.fill_between(range(len(max_accs)), max_accs, min_accs, alpha=0.3)

    x = EPOCHS - (EPOCHS / 10)
    plt.plot(x, mean_loss, marker='o', markersize=10, label=f"test loss ({round(mean_loss * 100, 1)}%)", color='blue', alpha=0.5)
    plt.plot(x, mean_acc, marker='o', markersize=10, label=f"test accuracy ({round(mean_acc * 100, 1)}%)", color='orange')

    plt.title("Loss and Accuracy Scores")
    plt.xlabel("epochs")
    plt.ylabel("score")
    plt.legend()

    plt.savefig(SAVED_CV_PATH)


def save_evaluations(cv, evaluations, type_of_augmentation, aug_parameters, json_path):

    # check if file exists at json_path and load it
    if exists(json_path) == True:
        with open(json_path, "r") as fp:
            eval_dict = json.load(fp)

    # create eval_dict if no file exists at json_path
    else:
        eval_dict = {
            "type_of_augmentation": [],
            "fold_num": [],
            "test_loss": [],
            "test_accuracy": [],
            "aug_parameters": []
        }

    evaluations = np.array(evaluations)

    # add data to eval_dict
    for i in range(cv):
        eval_dict["type_of_augmentation"].append(type_of_augmentation)
        eval_dict["fold_num"].append(i)
        eval_dict["test_loss"].append(evaluations[i, 0])
        eval_dict["test_accuracy"].append(evaluations[i, 1])
        eval_dict["aug_parameters"].append(aug_parameters)

    # save data as json file
    with open(json_path, "w") as fp:
        json.dump(eval_dict, fp, indent=4)

    print("Data saved to json_path")


def bf_cross_validate(dataset_path, json_path, type_of_augmentation, aug_parameters, cv=5, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT):

    # get all data
    all_files = get_data(dataset_path)

    # create folds (num of folds = cv)
    folds = make_folds(all_files, cv)

    # make data for model - MFCCs and labels for original and augmented audio
    data_dict = make_data_dict(folds, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)

    # modeling - create models using each fold as test data
    evaluations, histories = build_and_train_model(cv, data_dict)

    # save graph of histories as .png to data/augmentation_results
    make_loss_acc_plot(evaluations, histories)

    # append evaluations to .json in data/augmentation_results
    save_evaluations(cv, evaluations, type_of_augmentation, aug_parameters, json_path)

    # return all_files, folds, evaluations, histories


if __name__ == "__main__":
    # all_files, folds, evaluations, histories = bf_cross_validate(DATASET_PATH, JSON_PATH, TYPE_OF_AUGMENTATION, AUG_PARAMETERS, cv=10)
    bf_cross_validate(DATASET_PATH, JSON_PATH, TYPE_OF_AUGMENTATION, AUG_PARAMETERS, cv=5)
