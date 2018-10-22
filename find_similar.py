# Give input file and target folder or file, as well as maximum number of most simialr things to return

import os
import sys
import argparse
from collections import deque
from scipy.spatial import distance
from tqdm import tqdm

from griffin_lim import *
from model_iaf import *
from util import *

# Print most similar file whenever a new one was found
# If multiple clips per file enabled, also say which onset it was

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
batch_size = 128
model_params = 'params.json'

def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='Spectrogram VAE')
    parser.add_argument('--logdir', type=str, default=None,
                        help='Directory in which to store the logging '
                             'information for TensorBoard. '
                             'If the model already exists, it will restore '
                             'the state and will continue training. ')
    parser.add_argument('--target', type=str, default=None,
                        help='File for which similar sounds are to be found. ')
    parser.add_argument('--sample_dirs', type=str, nargs='+',
                        help='Root directories in which to look for samples. ')
    parser.add_argument('--num_to_keep', type=int, default=5,
                        help='Keep this many most similar files.')
    parser.add_argument('--batch_size', type=int, default=batch_size,
                        help='Batch Size.')
    parser.add_argument('--detect_onset', type=bool, default=False,
                        help='Remove initial silence.')
    parser.add_argument('--search_within_file', type=bool, default=False,
                        help='If true, not only encode the beginning, but detect transients and treat each separately.')
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

    # Look for original parameters
    print('Loading existing parameters.')
    print(f'{args.logdir}/params.json')
    with open(f'{args.logdir}/params.json', 'r') as f:
        param = json.load(f)

    batch_size = args.batch_size

    # Set correct batch size in deconvolution shapes
    deconv_shape = param['deconv_shape']
    for k, s in enumerate(deconv_shape):
        actual_shape = s
        actual_shape[0] = batch_size
        deconv_shape[k] = actual_shape
    param['deconv_shape'] = deconv_shape

    # Find all audio files in directories
    audio_files = []

    for root_dir in args.sample_dirs:
        for dirName, subdirList, fileList in os.walk(root_dir, topdown=False):
            for fname in fileList:
                if os.path.splitext(fname)[1] in ['.wav', '.aiff', '.WAV']:
                    audio_files.append('%s/%s' % (dirName, fname))

    # Create network.
    net = VAEModel(param,
                   batch_size)

    # Set up session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    init = tf.global_variables_initializer()
    sess.run(init)

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=max_checkpoints)

    try:
        saved_global_step = load(saver, sess, args.logdir)

    except:
        print("Something went wrong while restoring checkpoint. "
              "We will terminate training to avoid accidentally overwriting "
              "the previous model.")
        raise

    # Get target embeddings
    target_spec_single, _ = get_melspec(args.target, as_tf_input=True)
    target_spec = np.float32(np.zeros((batch_size,
                                       target_spec_single.shape[1],
                                       target_spec_single.shape[2],
                                       1)))
    target_spec[0] = target_spec_single[0]

    emb, _ = net.encode_and_reconstruct(target_spec)
    embedding_target = sess.run(emb)[0]

    # Go through all found files and compare distance
    similar_files = deque([None])
    distances = deque([float('inf')])

    full_batches = len(audio_files) // batch_size

    print(f'Starting to compare to {len(audio_files)} files.')

    try:
        for k in tqdm(range(full_batches+1)):

            # Prepare batch
            spec_list = []
            for j in range(batch_size):
                # Exception for last batch where index will be out of range
                try:
                    comparison_spec, _ = get_melspec(audio_files[k*batch_size+j], as_tf_input=True)
                except:
                    comparison_spec = np.zeros_like(target_spec_single)
                spec_list.append(comparison_spec)
            comparison_specs = np.concatenate(spec_list)

            emb_comp, _ = net.encode_and_reconstruct(comparison_specs)
            embedding_comp = sess.run(emb_comp)

            # Compare each individually
            for j in range(batch_size):

                if k*batch_size+j >= len(audio_files):
                    break

                # Get distance
                dist = distance.euclidean(embedding_comp[j], embedding_target)

                if dist >= max(distances):
                    continue

                # Find position where it should go
                for m in range(len(similar_files)):
                    if dist < distances[m]:
                        # print(f'{k*batch_size+j},{k},{j},{m} Inserting Distance: {dist}; File: {audio_files[k*batch_size+j]}')
                        similar_files.insert(m, audio_files[k * batch_size + j])
                        distances.insert(m, dist)
                        if m == 0:
                            print('New most similar file found.')
                            print(f'Distance: {dist}; File: {audio_files[k*batch_size+j]}')
                        break

                # Check if list grew beyond desired size
                if len(similar_files) > args.num_to_keep:
                    similar_files.pop()
                    distances.pop()

    except KeyboardInterrupt:
        print()
    finally:
        print('Search complete. Most similar files:')
        for k, file in enumerate(similar_files):
            # print()
            print(f'{k}  --  Distance: {distances[k]}; File: {file}')


if __name__ == '__main__':
    main()