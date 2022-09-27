from buzzfinder.make_audio_clips import get_audio_clips
from buzzfinder.const import ROOT_DIR, SAMPLES_TO_CONSIDER
from buzzfinder.prepare_dataset import delete_augmented_files, main
from buzzfinder.train_model import main as train_model_main
from buzzfinder.predict import Buzz_Finder_Service, MODEL_PATH
import os
import glob

def test_const_values():
    assert SAMPLES_TO_CONSIDER == 44100

def test_can_make_audio_clips():

    # test that get_audio_clips() in make_audio_clips.py can extract audio clips from longer audio file
    dataset_path = os.path.join(ROOT_DIR, "tests", "test_data", "test_audio_clips", "temp_audio_clips")
    get_audio_clips("test_raw_audio.wav", raw_audio_path=os.path.join(ROOT_DIR, "tests", "test_data"), dataset_path=dataset_path)
    temp_file = os.path.join(dataset_path, "test_data1.wav")
    assert os.path.isfile(temp_file)

    # delete test files in temp_audio_clips
    temp_files = glob.glob(os.path.join(dataset_path, "*.wav"))
    test_files = [file for file in temp_files if "test" in os.path.basename(file)]
    if len(test_files) > 0:
        [os.remove(file) for file in test_files]

def test_can_prepare_dataset():

    # get test paths to dataset and json
    test_dataset_path = os.path.join(ROOT_DIR, "tests", "test_data", "test_audio_clips")
    test_json_path = os.path.join(ROOT_DIR, "data", "test_dataset.json")

    # test that main() in prepare_dataset.py can augment the data and save a json
    main(dataset_path=test_dataset_path, json_path=test_json_path, create_augmented_data=True, n_augmentations_per_file=1)
    assert os.path.isfile(test_json_path)
    os.remove(test_json_path)

    # test main() using preexisting augmented data
    main(dataset_path=test_dataset_path, json_path=test_json_path, create_augmented_data=False)
    assert os.path.isfile(test_json_path)
    os.remove(test_json_path)

    # test main() with no augmented data
    buzzy_files = glob.glob(os.path.join(test_dataset_path, "buzzy", "*"))
    clean_files = glob.glob(os.path.join(test_dataset_path, "clean", "*"))
    all_files = buzzy_files + clean_files
    delete_augmented_files(all_files)
    main(dataset_path=test_dataset_path, json_path=test_json_path, create_augmented_data=False)
    assert os.path.isfile(test_json_path)
    os.remove(test_json_path)

def test_can_train_model():

    # get test paths to dataset, model, and learning curves chart
    test_data_path = os.path.join(ROOT_DIR, "tests", "test_data")
    dataset_path = os.path.join(test_data_path, "test_dataset.json")
    saved_model_path = os.path.join(test_data_path, "test_model.h5")
    saved_lc_path = os.path.join(test_data_path, "test_lc.png")

    # test that main() in train_model.py can train a model, plot learning curves, and save them
    train_model_main(use_mlflow=False, dataset_path=dataset_path, saved_model_path=saved_model_path, saved_lc_path=saved_lc_path, epochs=1)
    assert os.path.isfile(saved_model_path)
    assert os.path.isfile(saved_lc_path)

    # remove unneeded model and chart
    os.remove(saved_model_path)
    os.remove(saved_lc_path)

def test_model_exists():
    assert os.path.isfile(MODEL_PATH)

def test_bf_predict():
    bfs = Buzz_Finder_Service()
    test_audio_path = os.path.join(ROOT_DIR, "tests", "test_data", "test_audio_clips", "buzzy", "buzzy10.wav")
    predicted_tone = bfs.predict(test_audio_path)
    assert predicted_tone in ['buzzy', 'clean']

if __name__ == "__main__":
    # test_can_make_audio_clips()
    # test_can_prepare_dataset()
    # test_can_train_model()
    test_model_exists()
    # test_bf_predict()
