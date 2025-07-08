import sys

import librosa
import numpy as np

import util.file as file


SUBDIVISIONS = (1, 2, 3, 4)


def main():
    file_name = sys.argv[1]
    bpm = int(sys.argv[2])

    y, sr = file.load_audio("audio/rhythmic_evaluation/{}.wav".format(file_name))

    onset_times = get_onset_times(y, sr)
    adjusted_onset_times = get_adjusted_onset_times(y, sr, onset_times)
    beats = generate_beats(bpm, adjusted_onset_times)
    beat_onset_mappings = match_beats_to_onsets(beats, adjusted_onset_times)
    extract_subdivisions(beat_onset_mappings, adjusted_onset_times, bpm)


def extract_subdivisions(mappings, onset_times, bpm):
    for i in range(len(mappings)):
        start_onset = get_beat_time(i + 1, mappings, bpm)
        end_onset = get_beat_time(i + 2, mappings, bpm)
        
        for subdivision in SUBDIVISIONS:
            subdivision_length = (end_onset - start_onset) / subdivision
            subdivision_beats = [start_onset + subdivision_length * i for i in range(0, subdivision)]

            subdivision_grid = dict().fromkeys(range(1, subdivision + 1), "O")
            if mappings[i + 1]: subdivision_grid[1] = "X"

            onset_times_during_beat = [onset_time for onset_time in onset_times if onset_time >= start_onset and onset_time < end_onset]
            if len(onset_times_during_beat) > len(subdivision_beats):
                continue

            for onset_time in onset_times_during_beat:
                for subdivision_beat in subdivision_beats:
                    if time_in_window(subdivision_beat, onset_time):
                        subdivision_grid[subdivision_beats.index(subdivision_beat) + 1] = "X"
            
            if list(subdivision_grid.values()).count("X") == len(onset_times_during_beat):
                print("Subdivision for beat {}: {}".format(i + 1, subdivision))
                print("Grid for subdivision: {}".format(" ".join(subdivision_grid.values())))
                break


def get_beat_time(beat_index, beat_mappings, bpm):
    if beat_index < len(beat_mappings) and beat_mappings[beat_index]:
        return beat_mappings[beat_index]
    else:
        return beat_mappings[beat_index - 1] + 60 / bpm


def match_beats_to_onsets(beats, onset_times):
    mappings = dict()

    beat_number = 1
    for beat in beats:
        for onset_time in onset_times:
            if time_in_window(beat, onset_time):
                mappings[beat_number] = onset_time
                continue

        if beat_number not in mappings.keys():
            mappings[beat_number] = None

        beat_number += 1

    return mappings


def get_onset_times(y, sr):
    onset_times = librosa.onset.onset_detect(y=y, sr=sr, hop_length=32, units="time", backtrack=True,
        pre_max=10.0, post_max=1.0, pre_avg=33.0, post_avg=34.0, wait=10.0, delta=0.1)
    onset_times = np.unique(onset_times)
    
    return onset_times


def get_adjusted_onset_times(y, sr, onset_times):
    adjusted_onset_times = list()
    for i in range(len(onset_times)):
        start_time = onset_times[i] - 0.02
        end_time = start_time + 0.05

        start_sample = librosa.time_to_samples([start_time], sr=sr)[0]
        end_sample = librosa.time_to_samples([end_time], sr=sr)[0]

        peak = max(abs(y[start_sample:end_sample + 1]))

        adjusted_start_sample = start_sample
        for sample in range(start_sample, end_sample + 1):
            if abs(y[sample]) >= 0.6 * peak:
                adjusted_start_sample = sample
                break

        adjusted_start_time = librosa.samples_to_time(adjusted_start_sample, sr=sr) - 0.0001
        adjusted_onset_times.append(adjusted_start_time)

    adjusted_onset_times = list(set(adjusted_onset_times))
    adjusted_onset_times.sort()
    return adjusted_onset_times


def time_in_window(target, time, start_distance=0.02, end_distance=0.02):
    target_window_start = target - start_distance
    target_window_end = target + end_distance

    return time >= target_window_start and time <= target_window_end


def generate_beats(bpm, onset_times):
    current_beat = onset_times[0]
    beats = [current_beat]
    while current_beat <= onset_times[-1]:
        current_beat = beats[-1] + 60 / bpm
        beats.append(current_beat)

    return beats


if __name__ == "__main__":
    main()
