# Drumline Transcriber

A Python toolchain for transcribing drumline audio recordings, built with [Tensorflow](https://www.tensorflow.org/) and [Librosa](https://librosa.org/doc/latest/index.html). The accompanying [research paper](Research%20Paper.pdf) provides further details with regards to implementation details, approach, motivation, and effectiveness.

## Requirements

- Python 3 (no later than version 3.11)
- Installing the `requirements.txt`

## Quick Start

Ensure you have met all the requirements above, from the root directory of the project do the following:

1. run `python3 dataset_generator.py --singles -d --divisor 2` to generate a dataset to be used by the neural network
2. run `python3 network_generator.py` to train and generate a neural network on the generated dataset (feel free to checkout the training results that are output from the command)
3. rename the generated model in `/models` to `main.keras`
4. run `python3 network_evaluator.py` to evaluate the neural network's effectiveness. With the data we've generated for our model in this example you'll likely see overfitting for doubles (feel free to experiment with the dataset generation parameters from step 1)
5. run `python3 rhythm_extractor.py clip_1 114` to see an example of rhythm extraction from the sample audio (feel free to confirm the transcription by listening to the [audio file](audio/rhythmic_evaluation/clip_1.wav))

## Components

The toolchain consists of components that can be used independently and in conjunction. [Sample audio files are included](audio), which were utilized for evaluation in the accompanying [research paper](Research%20Paper.pdf).

All of the components expect audio in a mono 16-bit PCM WAV format. The included audio files can be used as a reference.

### Neural Network Generator

Used to generate the neural network used for evaluating audio files.

`python3 network_generator.py`

You can optionally pass the number of hidden layers by using `--hidden_layers <number>`. For example:

`python3 network_generator.py --hidden_layers 2`

The default number of hidden layers is 3.

Note that the program will look for files in `/audio` under `/bass`, `/snare` and `/tenor`. Any files added should be put in their corresponding folders as the dataset generator relies on it to create it's combinations for training and evaluation.

### Neural Network Evaluator

Used to evaluate the given neural network. The network to be evaluated should be stored in `/models` and be named `main.keras`, yielding a file path of `/models/main.keras`.

`python3 network_evaluator.py`

### Rhythm Extractor

Used to transcribe the rhythm of a given audio file. The file to be evaluated should be stored in `/audio/rhythmic_evaluation`.

`python3 rhythm_extractor.py <file_name> <bpm>`

The file name should exclude `.wav`, and the bpm should match that of the recording, for example:

`python3 rhythm_extractor.py clip_1 114`

Note that the program will assume a time signature of 4/4 and the first note to be the downbeat of a measure. The program handles quarter notes, eighth notes, eighth note triplets, and sixteenth notes by default.

### Dataset Generator

Used to create datasets to train and evaluate the neural networks on. 

`python3 dataset_generator.py --singles --doubles --triples --divisor <number>`

Any of the numeric arguments can also be passed using shorthand notation (`-s`, `-d`, `-t`), and the divisor is used to reduce the number of generated combinations by dividing the number of combinations by `2^d` where `d` is the divisor. For example:

`python3 dataset_generator.py --singles -d --divisor 2`

Note that the program will look for files in `/audio` under `/bass`, `/snare` and `/tenor`. Any files added should be put in their corresponding folders as the program relies on it to create it's combinations.