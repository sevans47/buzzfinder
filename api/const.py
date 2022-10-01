import os

my_path = os.path.dirname(os.path.abspath(__file__))  # get path to directory with const.py
ROOT_DIR = os.path.abspath(os.path.join(my_path, os.pardir))  # get path to parent dir of const.py
SAMPLES_TO_CONSIDER = 44100
N_MFCC=13
HOP_LENGTH=512
N_FFT=2048
N_MELS=90

if __name__ == "__main__":
    print(ROOT_DIR)
    print(os.path.dirname(__file__))
    print(os.path.abspath(__file__))
