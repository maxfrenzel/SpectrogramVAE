# Spectrogram VAE
TensorFlow implementation of a Variational Autencoder for encoding spectrograms.

This is the main model I used for my [NeuralFunk project](https://towardsdatascience.com/neuralfunk-combining-deep-learning-with-sound-design-91935759d628).

This code was not really intended to be shared and is quite messy. I might improve it at some point in the future, but for now be aware that everything is quite hacky and badly documented.

## Acknowledgments
* The preprocessing as well as the encoder architecture were heavily inspired by [this iPython Notebook](https://gist.github.com/naotokui/a2b331dd206b13a70800e862cfe7da3c).
* A lot of the data-feeding code and many other bits and pieces were adapted from [this Wavenet implementation](https://github.com/ibab/tensorflow-wavenet).
* The Griffin-Lim algorithm was taken from the [Magenta NSynth utils](https://github.com/tensorflow/magenta/blob/master/magenta/models/nsynth/utils.py).

## Overview
Some random experiments, as well as the creation of the dataset for the VAE can be found in [Preprocessing and Experiments.ipynb](https://github.com/maxfrenzel/SpectrogramVAE/blob/master/Preprocessing%20and%20Experiments.ipynb).

The dataset pickle file has to be a dictionary of the form
```
{
  'filenames' : list_of_filenames,
  'melspecs' : list_of_spectrogram_arrays,
  'actual_lengths' : list_of_audio_len_in_sec
}
```
and be stored as `dataset.pkl` in the root directory.

### Training the VAE 
```python train.py```

### Generating samples 
Based on
* Sampling from latent space: `python generate.py`
* Single input file: `python generate.py --file_in filename`
* Multiple input files: `python generate.py --file_in list_of_filenames`

### Encode audio
* Single file: `python encode_and_reconstruct.py --audio_file filename`
* Full dataset: `python encode_and_reconstruct.py --encode_full true`

### Finding similar files:
```python find_similar.py --target target_audio_file --sample_dirs list_of_dirs_to_search```

All the above scripts have other options and uses as well, look into the code for more details.
