from argparse import ArgumentParser
import datetime
import random

import numpy as np
from tensorflow import keras
from keras import layers
from tqdm import tqdm

import util.file as file
from model.instrument import INSTRUMENTS


def main():
    parser = ArgumentParser(description="Generate neural network trained on marching percussion audio chunks")
    parser.add_argument("--hidden_layers", type=int, default=3, metavar="", help="set the number of hidden layers")
    args = parser.parse_args()

    chunks = list()
    labels = list()
    for path in file.get_files("audio/extracted"):
        audio_paths = file.get_files("{}/audio".format(path))
        random.shuffle(audio_paths)
        file.save_data(path, "dataset_list", audio_paths)
        dataset_paths = audio_paths[:round(len(audio_paths) * 0.8)]
        for dataset_path in tqdm(dataset_paths):
            chunks.append(file.load_chunk(dataset_path))
            file_name = file.get_file_name(dataset_path)
            labels_path = "{}/labels/{}.csv".format(path, file_name)
            current_labels = file.load_labels(labels_path)
            formatted_labels = file.parse_labels(current_labels)
            labels.append(formatted_labels)
    
    chunks = np.asarray(chunks)
    labels = np.asarray(labels)

    model = generate_model(chunks.shape[1], len(INSTRUMENTS), args.hidden_layers)
    model.fit(chunks, labels, epochs=10)

    model_path = file.create_file_path("models", "{}.keras".format(datetime.datetime.now().strftime("%d-%m-%Y_%H-%M")))
    model.save(model_path)

    model.summary()


def generate_model(input_number, number_of_labels, number_of_hidden_layers):
    model = keras.Sequential()
    
    model.add(layers.Dense(1024, input_dim=input_number, kernel_initializer="he_uniform", activation="relu"))
    for i in range(number_of_hidden_layers - 1):
        model.add(layers.Dense(1024, activation="relu"))
    model.add(layers.Dense(number_of_labels, activation="sigmoid"))
    
    model.compile(loss="binary_crossentropy", optimizer="adam")

    return model


if __name__ == "__main__":
    main()
