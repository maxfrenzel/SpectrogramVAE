import librosa
import numpy as np
import json

with open('audio_params.json', 'r') as f:
    param = json.load(f)

N_FFT = param['N_FFT']
HOP_LENGTH = param['HOP_LENGTH']
SAMPLING_RATE = param['SAMPLING_RATE']
MELSPEC_BANDS = param['MELSPEC_BANDS']
sample_secs = param['sample_secs']
num_samples_dataset = int(sample_secs * SAMPLING_RATE)


# Function to read in an audio file and return a mel spectrogram
def get_melspec(filepath_or_audio, hop_length=HOP_LENGTH, n_mels=MELSPEC_BANDS, n_samples=num_samples_dataset,
                sample_secs=sample_secs, as_tf_input=False):

    y_tmp = np.zeros(n_samples)

    # Load a little more than necessary as a buffer
    load_duration = None if sample_secs == None else 1.1 * sample_secs

    # Load audio file or take given input
    if type(filepath_or_audio) == str:
        y, sr = librosa.core.load(filepath_or_audio, sr=SAMPLING_RATE, mono=True, duration=load_duration)
    else:
        y = filepath_or_audio
        sr = SAMPLING_RATE

    # Truncate or pad
    if n_samples:
        if len(y) >= n_samples:
            y_tmp = y[:n_samples]
            lentgh_ratio = 1.0
        else:
            y_tmp[:len(y)] = y
            lentgh_ratio = len(y) / n_samples
    else:
        y_tmp = y
        lentgh_ratio = 1.0

    # sfft -> mel conversion
    melspec = librosa.feature.melspectrogram(y=y_tmp, sr=sr,
                                             n_fft=N_FFT, hop_length=hop_length, n_mels=n_mels)
    S = librosa.power_to_db(melspec, np.max)

    if as_tf_input:
        S = spec_to_input(S)

    return S, lentgh_ratio

def spec_to_input(spec):
    specs_out = (spec + 80.0) / 80.0
    specs_out = np.expand_dims(np.expand_dims(specs_out, axis=0), axis=3)
    return np.float32(specs_out)