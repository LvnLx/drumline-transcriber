import os

import librosa
import numpy as np
from soundfile import write

from model.instrument import INSTRUMENTS


def save_audio(file_directory, file_name, y, sr):
    file_path = create_file_path(file_directory, "{}.wav".format(file_name))
    write(file_path, y, sr)


def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=44100)
    return y, sr


def load_chunk(file_path):
    y = load_audio(file_path)[0]
    chunk = librosa.stft(y)
    chunk = chunk.flatten()
    return np.asarray(chunk)


def save_data(file_directory, file_name, data):
    file_path = create_file_path(file_directory, "{}.csv".format(file_name))
    csv_file = open(file_path, "w")
    date_as_strings = [str(value) for value in data]
    csv_file.write(",".join(date_as_strings))
    csv_file.close()


def load_labels(file_path):
    file = open(file_path, "r")
    output = file.readline().split(",")
    output = [int(value) for value in output]
    file.close()

    return np.asarray(output)


def parse_labels(labels):
    output = [0 for i in range(len(INSTRUMENTS))]
    for label in labels:
        output[label] = 1
    return np.asarray(output)


def load_paths(file_path):
    file = open(file_path, "r")
    output = file.readline().split(",")
    file.close

    return output

def create_file_path(file_directory, file_name):
    if not os.path.exists(file_directory):
        os.makedirs(file_directory)

    return "{}/{}".format(file_directory, file_name)


def get_files(file_path):
    return [file_path + "/" + file_name for file_name in os.listdir(file_path)]


def get_file_name(file_path):
    return os.path.basename(file_path).split(".")[0]