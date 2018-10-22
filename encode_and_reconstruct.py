import matplotlib as mpl
mpl.use('TkAgg')

import tensorflow as tf
import os
import sys
import time
import joblib
from random import shuffle
import numpy as np
import argparse
import json

import librosa
import librosa.display
import matplotlib.pyplot as plt

from spec_reader import *
from util import *
from model_iaf import *
from griffin_lim import griffin_lim

with open('audio_params.json', 'r') as f:
    param = json.load(f)

N_FFT = param['N_FFT']
HOP_LENGTH = param['HOP_LENGTH']
SAMPLING_RATE = param['SAMPLING_RATE']
MELSPEC_BANDS = param['MELSPEC_BANDS']
sample_secs = param['sample_secs']
num_samples_dataset = int(sample_secs * SAMPLING_RATE)

logdir = './test_iaf'
max_checkpoints = 5
num_steps = 10000
checkpoint_every = 500
batch_size = 1
model_params = 'params.json'
num_data = -1
encode_batch_size = 128
dataset_file = 'dataset.pkl'

def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    # TODO: Some of these paramters clash and if not chosen correctly crash the script
    parser = argparse.ArgumentParser(description='Spectrogram VAE')
    parser.add_argument('--num_data', type=int, default=num_data,
                        help='How many data points to process. Default: ' + str(num_data) + '.')
    parser.add_argument('--logdir', type=str, default=None,
                        help='Directory in which to store the logging '
                        'information for TensorBoard. '
                        'If the model already exists, it will restore '
                        'the state and will continue training. '
                        'Cannot use with --logdir_root and --restore_from.')
    parser.add_argument('--model_params', type=str, default=model_params,
                        help='JSON file with the network parameters. Default: ' + model_params + '.')
    parser.add_argument('--audio_file', type=str, default=None,
                        help='Audiofile to encode and reconstruct. If not specified, will use existing dataset instead.')
    parser.add_argument('--dataset_file', type=str, default=dataset_file,
                        help='Dataset pkl file. Default: ' + dataset_file + '.')
    parser.add_argument('--encode_full', type=bool, default=False,
                        help='Encode and save entire dataset? Default: ' + str(False) + '.')
    parser.add_argument('--encode_only_new', type=bool, default=True,
                        help='Encode only new data points? Default: ' + str(True) + '.')
    parser.add_argument('--process_original_audio', type=bool, default=False,
                        help='Process/copy original audio when saving embeddings? Default: ' + str(False) + '.')
    parser.add_argument('--plot_spec', type=bool, default=True,
                        help='Plot reconstructed spectrograms? Default: ' + str(True) + '.')
    parser.add_argument('--rec_audio', type=bool, default=True,
                        help='Reconstruct and save audio? Default: ' + str(True) + '.')
    return parser.parse_args()

def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


def load(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir),
          end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None

def main():

    args = get_arguments()

    # Load data unless input audiofile is specified
    if args.audio_file:
        melspecs, _ = get_melspec(args.audio_file, as_tf_input=True)
        filename = os.path.basename(args.audio_file)
    else:
        melspecs, filenames = load_specs(filename='dataset.pkl', return_filenames=True)
        melspecs = (np.float32(melspecs) + 80.0) / 80.0
        # melspecs = 80.0*(np.random.random((10000,128,126))-1.0)

    # print(melspecs[0].shape)
    # print(np.expand_dims(np.expand_dims(melspecs[0], 0), 3).shape)

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    # Look for original parameters
    print('Loading existing parameters.')
    print(f'{args.logdir}/params.json')
    with open(f'{args.logdir}/params.json', 'r') as f:
        param = json.load(f)

    if args.encode_full:
        batch_size = encode_batch_size
        full_batches = len(filenames) // batch_size
        filename_counter = 0
    else:
        batch_size = 1

    # Set correct batch size in deconvolution shapes
    deconv_shape = param['deconv_shape']
    for k, s in enumerate(deconv_shape):
        actual_shape = s
        actual_shape[0] = batch_size
        deconv_shape[k] = actual_shape
    param['deconv_shape'] = deconv_shape

    # Create coordinator.
    coord = tf.train.Coordinator()

    with tf.name_scope('create_inputs'):
        reader = SpectrogramReader(melspecs, coord)
        spec_batch = reader.dequeue(batch_size)

    # Create network.
    net = VAEModel(param,
                   batch_size)

    # Set up session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    init = tf.global_variables_initializer()
    sess.run(init)

    # Saver for loading checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=max_checkpoints)

    try:
        saved_global_step = load(saver, sess, args.logdir)

    except:
        print("Something went wrong while restoring checkpoint. "
              "We will terminate training to avoid accidentally overwriting "
              "the previous model.")
        raise

    # Check if directory for saving exists
    out_dir = f'{args.logdir}/reconstructed-{saved_global_step}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if args.encode_full:
        out_dir_emb = f'{args.logdir}/embeddings-{saved_global_step}'
        if not os.path.exists(out_dir_emb):
            os.makedirs(out_dir_emb)

    if args.audio_file == None:

        if args.num_data == -1:
            if args.encode_full:
                num_batches = full_batches
            else:
                num_batches = len(filenames)
        else:
            num_batches = args.num_data

        for step in range(num_batches):

            if batch_size == 1:
                spec_in = np.expand_dims(np.expand_dims(melspecs[step],0),3)
            else:
                if step < full_batches:
                    spec_in = melspecs[step * batch_size:(step + 1) * batch_size]
                    batch_filenames = filenames[step * batch_size:(step + 1) * batch_size]
                else:
                    spec_in = melspecs[step * batch_size:-1]
                    batch_size_discrep = batch_size - spec_in.shape[0]
                    spec_in = np.concatenate([spec_in, np.zeros(batch_size_discrep, spec_in.shape[1])])
                    batch_filenames = filenames[step * batch_size:-1]
                spec_in = np.expand_dims(spec_in,3)

            if args.encode_full:

                print(f'Batch {step} of {full_batches}.')

                # Check if all files exist
                exists = []
                filename_counter_check = filename_counter
                for k, name in enumerate(batch_filenames):
                    name_no_path = os.path.splitext(os.path.split(name)[1])[0]

                    dataset_filename_emb = f'{out_dir_emb}/{filename_counter_check} - {name_no_path}.npy'

                    # Skip if already exists
                    if os.path.isfile(dataset_filename_emb):
                        exists.append(True)
                    else:
                        exists.append(False)
                    filename_counter_check += 1

                if all(exists):
                    filename_counter = filename_counter_check
                    continue

                emb, out = net.encode_and_reconstruct(spec_in)
                embedding, output = sess.run([emb, out])

                del spec_in
                print(embedding.shape)

                # Save
                for k, name in enumerate(batch_filenames):

                    if args.process_original_audio:

                        try:

                            name_no_path = os.path.splitext(os.path.split(name)[1])[0]

                            dataset_filename = f'{out_dir_emb}/{filename_counter} - {name_no_path}.wav'
                            dataset_filename_emb = f'{out_dir_emb}/{filename_counter} - {name_no_path}.npy'

                            # Skip if already exists
                            if args.encode_only_new and os.path.isfile(dataset_filename_emb):
                                filename_counter += 1
                                continue

                            # Load audio file
                            y, sr = librosa.core.load(name, sr=SAMPLING_RATE, mono=True, duration=sample_secs)
                            y_tmp = np.zeros(num_samples_dataset)

                            # Truncate or pad
                            if len(y) >= num_samples_dataset:
                                y_tmp = y[:num_samples_dataset]
                            else:
                                y_tmp[:len(y)] = y

                            # Write to file
                            librosa.output.write_wav(dataset_filename, y_tmp, sr, norm=True)
                            np.save(dataset_filename_emb, embedding[k])

                            # # Also plot reconstruction
                            # melspec = (np.squeeze(output[k]) - 1.0) * 80.0
                            # plt.figure()
                            # ax1 = plt.subplot(2, 1, 1)
                            #
                            # librosa.display.specshow((melspecs[step * batch_size + k] - 1.0) * 80.0, sr=SAMPLING_RATE, y_axis='mel',
                            #                          x_axis='time',
                            #                          hop_length=HOP_LENGTH)
                            # plt.title('Original: ' + name_no_path)
                            # ax2 = plt.subplot(2, 1, 2, sharex=ax1)
                            # librosa.display.specshow(melspec, sr=SAMPLING_RATE, y_axis='mel', x_axis='time',
                            #                          hop_length=HOP_LENGTH)
                            # plt.title('Reconstruction')
                            # plt.tight_layout()
                            # plt.savefig(f'{out_dir_emb}/{filename_counter} - {name_no_path}.png')
                            # plt.close()

                            filename_counter += 1

                        except:
                            pass

                    else:
                        dataset_filename_emb = f'{out_dir_emb}/{filename_counter} - {name_no_path}.npy'

                        # Write to file
                        np.save(dataset_filename_emb, embedding[k])

                        filename_counter += 1

                del emb, out
                del embedding, output

            else:

                emb, out = net.encode_and_reconstruct(spec_in)
                embedding, output = sess.run([emb, out])

                melspec = (np.squeeze(output[0])-1.0)*80.0

                if args.plot_spec:
                    plt.figure()
                    ax1 = plt.subplot(2, 1, 1)

                    librosa.display.specshow((melspecs[step]-1.0)*80.0, sr=SAMPLING_RATE, y_axis='mel', x_axis='time',
                                             hop_length=HOP_LENGTH)
                    plt.title('Original: ' + os.path.basename(filenames[step]))
                    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
                    librosa.display.specshow(melspec, sr=SAMPLING_RATE, y_axis='mel', x_axis='time',
                                             hop_length=HOP_LENGTH)
                    plt.title('Reconstruction')
                    plt.tight_layout()
                    plt.savefig(f'{out_dir}/reconstructed-{step}.png')
                    plt.close()

                if args.rec_audio:
                    audio = griffin_lim(melspec)
                    audio_file = f'{out_dir}/reconstructed-{step}.wav'
                    librosa.output.write_wav(audio_file, audio/np.max(audio), sr=SAMPLING_RATE)

    else:

        spec_in = melspecs

        emb, out = net.encode_and_reconstruct(spec_in)
        embedding, output = sess.run([emb, out])

        melspec = (np.squeeze(output[0]) - 1.0) * 80.0
        melspec_in = (np.squeeze(melspecs) - 1.0) * 80.0

        # Save embeddings
        np.save(f'{out_dir}/reconstructed-{filename[:-4]}.npy', embedding[0])

        # Plot
        plt.figure()
        ax1 = plt.subplot(2, 1, 1)

        librosa.display.specshow((melspec_in - 1.0) * 80.0, sr=SAMPLING_RATE, y_axis='mel', x_axis='time',
                                 hop_length=HOP_LENGTH)
        plt.title('Original: ' + os.path.basename(args.audio_file))
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        librosa.display.specshow(melspec, sr=SAMPLING_RATE, y_axis='mel', x_axis='time',
                                 hop_length=HOP_LENGTH)
        plt.title('Reconstruction')
        plt.tight_layout()
        plt.savefig(f'{out_dir}/reconstructed-{filename[:-4]}.png')
        plt.close()

        # Reconstruct and save audio
        audio = griffin_lim(melspec)
        audio_file = f'{out_dir}/reconstructed-{filename[:-4]}.wav'
        librosa.output.write_wav(audio_file, audio / np.max(audio), sr=SAMPLING_RATE)


if __name__ == '__main__':
    main()