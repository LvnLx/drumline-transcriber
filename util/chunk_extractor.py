import sys

import librosa
import numpy as np

import util.file as file


def extract_chunks(instrument, file_name):
    y, sr = file.load_audio("audio/{}/{}.wav".format(instrument.name, file_name))

    onset_times = get_onset_times(y, sr)
    chunks = list()

    for i in range(len(onset_times)):
        # Create large window
        start_time = onset_times[i] - 0.02
        end_time = start_time + 0.05

        start_sample = librosa.time_to_samples([start_time], sr=sr)[0]
        end_sample = librosa.time_to_samples([end_time], sr=sr)[0]

        # Grab the peak within the sample window
        peak = max(abs(y[start_sample:end_sample + 1]))

        # Remove false positive onset times
        if peak < 0.05:
            continue

        # Scan forward until the main transient starts
        adjusted_start_sample = start_sample
        for sample in range(start_sample, end_sample + 1):
            if abs(y[sample]) >= 0.6 * peak:
                adjusted_start_sample = sample
                break

        # Get adjusted times and finalize window
        adjusted_start_time = librosa.samples_to_time(adjusted_start_sample, sr=sr) - 0.002
        adjusted_end_time = adjusted_start_time + 0.05

        adjusted_start_sample = librosa.time_to_samples([adjusted_start_time], sr=sr)[0]
        adjusted_end_sample = librosa.time_to_samples([adjusted_end_time], sr=sr)[0]

        # Add all windowed samples into a chunk
        chunk = list()
        for sample in range(adjusted_start_sample, adjusted_end_sample + 1):
            chunk.append(y[sample])
        chunk = np.asarray(chunk)

        chunks.append(chunk)

    for i in range(len(chunks[:instrument.limit])):
        chunk_file_name = "{}_{}_chunk_{}".format(instrument.name, file_name, i)
        save_chunk(instrument.name, chunk_file_name, chunks[i], sr, [instrument.label])


def get_onset_times(y, sr):
    onset_times = librosa.onset.onset_detect(y=y, sr=sr, hop_length=32, units="time", backtrack=True,
        pre_max=10.0, post_max=1.0, pre_avg=33.0, post_avg=34.0, wait=10.0, delta=0.05)
    onset_times = np.unique(onset_times)
    return onset_times


def save_chunk(instrument_name, file_name, chunk, sr, labels):
    file.save_audio("audio/extracted/{}/audio".format(instrument_name), file_name, chunk, sr)
    file.save_data("audio/extracted/{}/labels".format(instrument_name), file_name, labels)
