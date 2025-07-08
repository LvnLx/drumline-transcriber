import numpy as np
from tensorflow import keras
from tqdm import tqdm

import util.file as file
from model.instrument import INSTRUMENTS


def main():
    model = keras.models.load_model("models/main.keras")

    chunks = list()
    labels = list()
    file_names = list()

    for path in file.get_files("audio/extracted"):
        dataset_list = file.load_paths(path + "/dataset_list.csv")
        audio_paths = dataset_list[round(len(dataset_list) * 0.8):]
        for audio_path in tqdm(audio_paths):
            chunk = file.load_chunk(audio_path)
            chunks.append(chunk)

            file_name = file.get_file_name(audio_path)
            labels_path = "{}/labels/{}.csv".format(path, file_name)
            current_labels = file.load_labels(labels_path)
            formatted_labels = file.parse_labels(current_labels)
            labels.append(formatted_labels)

            file_names.append(file_name)

    chunks = np.asarray(chunks)
    labels = np.asarray(labels)
    file_names = np.asarray(file_names)

    predictions = model.predict(chunks)

    singles = 0 
    doubles = 0
    triples = 0
    single_correctness = 0
    double_correctness = 0
    triple_correctness = 0

    prediction_results = [[0, 0, 0, 0] for i in range(len(predictions[0]))]

    print("Incorrect predictions:")

    for i in range(len(predictions)):
        prediction = np.rint(predictions[i])
        label = labels[i]
        file_name = file_names[i]

        for j in range(len(label)):
            if prediction[j] == 0:
                if label[j] == 0:
                    prediction_results[j][1] += 1
                elif label[j] == 1:
                    prediction_results[j][3] += 1
            elif prediction[j] == 1:
                if label[j] == 0:
                    prediction_results[j][2] += 1
                elif label[j] == 1:
                    prediction_results[j][0] += 1

        if np.sum(label) == 1:
            singles += 1
            if np.array_equal(prediction, label):
                single_correctness += 1
            else:
                print(" - {}\n\tPrediction: {}\n\tActual:     {}".format(file_name, prediction, label))

        if np.sum(label) == 2:
            doubles += 1
            if np.array_equal(prediction, label):
                double_correctness += 1
            else:
                print(" - {}\n\tPrediction: {}\n\tActual:     {}".format(file_name, prediction, label))

        if np.sum(label) == 3:
            triples += 1
            if np.array_equal(prediction, label):
                triple_correctness += 1
            else:
                print(" - {}\n\tPrediction: {}\n\tActual:     {}".format(file_name, prediction, label))

    for i in range(len(prediction_results)):
        tp = prediction_results[i][0]
        tn = prediction_results[i][1]
        fp = prediction_results[i][2]
        fn = prediction_results[i][3]

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (fp + tp)
        recall = tp / (fn + tp)
        f1_score = 2 * (precision * recall) / (precision + recall)

        print("Instrument: {}".format(INSTRUMENTS[i].name))
        print(" - F1 Score: {:.4f} (Precision: {:.4f}, Recall {:.4f})".format(f1_score, precision, recall))
        print(" - Accuracy: {:.4f}\n".format(accuracy))

    print("Overall Accuracy:")
    print(" - Singles: {:.2f}%".format(single_correctness / singles * 100))
    print(" - Doubles: {:.2f}%".format(double_correctness / doubles * 100))
    print(" - Triples: {:.2f}%".format(triple_correctness / triples * 100))


if __name__ == "__main__":
    main()
