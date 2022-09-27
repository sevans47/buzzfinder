"""
This module builds, trains, and evaluates a model using the dataset created by
prepare_dataset.py.  The dataset is saved as a .json to the data folder.

When using this module from the command line, you can save the results using mlflow
with the following command line argument:
- use mlflow: python train_model.py 1
- no mlflow: python train_model.py 0
"""

import mlflow
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from buzzfinder.const import ROOT_DIR

# filepath to data for training model
data_filename = 'comprehensive_mfccs_no_augment.json'
DATA_PATH = os.path.join(ROOT_DIR, 'data', data_filename)

# filepaths for saving model artifacts
temp_path = os.path.join(ROOT_DIR, 'temp')
SAVED_MODEL_PATH = os.path.join(temp_path, "model.h5")
SAVED_LC_PATH = os.path.join(temp_path, 'learning_curves.png')
SAVED_SUMMARY_PATH = os.path.join(temp_path, 'model_summary.txt')
SAVED_MODEL_INFO = 'model_info.txt'

# model variables
LEARNING_RATE = 0.0001
EPOCHS = 400
BATCH_SIZE = 32
PATIENCE = 20


def load_dataset(data_path):

    with open(data_path, "r") as fp:
        data = json.load(fp)

    # extract inputs and targets
    X_train = np.array(data["train_data"])
    X_validation = np.array(data["val_data"])
    X_test = np.array(data["test_data"])
    y_train = np.array(data["train_labels"])
    y_validation = np.array(data["val_labels"])
    y_test = np.array(data["test_labels"])

    # convert inputs from 2d to 3d arrays - (# segments, 13) -> (# segments, 13, 1)
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test


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
    model.add(Dense(2, activation="softmax"))  #[0.1, 0.7]
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # pring model overview
    model.summary()

    return model


def save_learning_curves(saved_lc_path, history, error, accuracy):

    # create figure
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    x = len(history['loss']) - PATIENCE

    # plot loss, val loss, and test loss
    axs[0].plot(history['loss'], label='loss')
    axs[0].plot(history['val_loss'], label='val_loss')
    axs[0].plot(x, error, marker='x', markersize=10, markeredgewidth=3, label=f'test loss: {round(error, 3)}')
    axs[0].legend()
    axs[0].set_ylim((0, 1.1))
    axs[0].set_title('Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Score')

    # plot accuracy, val accuracy, and test accuracy
    axs[1].plot(history['accuracy'], label='accuracy')
    axs[1].plot(history['val_accuracy'], label='val_accuracy')
    axs[1].plot(x, accuracy, marker='x', markersize=10, markeredgewidth=3, label=f'test accuracy: {round(accuracy, 3)}')
    axs[1].legend()
    axs[1].set_ylim((0, 1.1))
    axs[1].set_title('Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Score')

    plt.savefig(saved_lc_path)


def save_model_summary(s):
    with open(SAVED_SUMMARY_PATH,'a') as f:
        print(s, file=f)


def main(use_mlflow=True, dataset_path=DATA_PATH, saved_model_path=SAVED_MODEL_PATH, saved_lc_path=SAVED_LC_PATH, epochs=EPOCHS):

    # load train/validation/test data splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = load_dataset(dataset_path)

    # build the CNN model
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])  # (# frames, # coefficients (13), # channels (1)
    model = build_model(input_shape, LEARNING_RATE)


    # store model with mlflow
    if use_mlflow == True:
        with mlflow.start_run():

            # train the model
            es = EarlyStopping(patience=PATIENCE)
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=BATCH_SIZE,
                    validation_data=(X_validation, y_validation), callbacks=[es])

            # evaluate the model
            test_error, test_accuracy = model.evaluate(X_test, y_test)
            print(f"Test error: {test_error}, test accuracy: {test_accuracy}")

            # log parameters and metrics
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", BATCH_SIZE)
            mlflow.log_param("learning_rate", LEARNING_RATE)

            total_size = len(X_train) + len(X_validation) + len(X_test)
            training_size = len(X_train) / total_size
            val_size = len(X_validation) / total_size
            test_size = len(X_test) / total_size
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("validation_size", val_size)

            mlflow.log_metric("error", test_error)
            mlflow.log_metric("accuracy", test_accuracy)

            # log model summary
            with open(SAVED_SUMMARY_PATH,'w') as f:
                print("", file=f)
            model.summary(print_fn=save_model_summary)
            mlflow.log_artifact(SAVED_SUMMARY_PATH)

            # log learning curves
            save_learning_curves(saved_lc_path, history.history, test_error, test_accuracy)
            mlflow.log_artifact(saved_lc_path)

            # log other info
            mlflow.log_text(f"""
                data used: {DATA_PATH}
                total data size: {total_size}
                training data size: {len(X_train)}, {round(training_size * 100, 2)}% of data
                validation data size: {len(X_validation)}, {round(val_size * 100, 2)}% of data
                test data size: {len(X_test)}, {round(test_size * 100, 2)}% of data
            """, SAVED_MODEL_INFO)

            # log model
            model.save(saved_model_path)
            mlflow.log_artifact(saved_model_path)

    # train model w/o saving to mlflow
    else:
        # train the model
        es = EarlyStopping(patience=PATIENCE)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=BATCH_SIZE,
                validation_data=(X_validation, y_validation), callbacks=[es])

        # evaluate the model
        test_error, test_accuracy = model.evaluate(X_test, y_test)
        print(f"Test error: {test_error}, test accuracy: {test_accuracy}")

        # save learning curves to temp folder
        save_learning_curves(saved_lc_path, history.history, test_error, test_accuracy)

        # save model to temp folder
        model.save(saved_model_path)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        main(use_mlflow=True)
    elif len(sys.argv) == 2 and sys.argv[1] == str(1):
        main(use_mlflow=True)
    elif len(sys.argv) == 2 and sys.argv[1] == str(0):
        main(use_mlflow=False)
    elif len(sys.argv) > 2:
        print("Too many command line arguments (expected 1)")
    else:
        print("Please add correct command line argument (0 for no mlflow, 1 for mlflow)")
