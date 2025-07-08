from sys import argv
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def main():
    file_name = argv[1]
    graph_name = argv[2]
    y, sr = librosa.load(file_name)
    D = librosa.stft(y)
    graph_spectrogram(D, sr, graph_name)


def graph_spectrogram(D, sr, graph_name):
    rp = np.max(np.absolute(D))

    fig, ax = plt.subplots(nrows=1, sharex=True, sharey=True)

    ax.label_outer()
    ax.set(title=graph_name)

    img = librosa.display.specshow(librosa.amplitude_to_db(np.abs(D), ref=rp),
                                   x_axis='time', y_axis='log', sr=sr, ax=ax)

    fig.colorbar(img, ax=ax)
    plt.show()


def graph_onsets(o_env, onset_frames):
    times = librosa.times_like(o_env)

    ax = plt.subplots(nrows=1, sharex=True)[1]
    ax.plot(times, o_env)
    ax.vlines(times[onset_frames], 0, o_env.max(), color='r', linestyle='--')

    plt.show()

if __name__ == "__main__":
    main()