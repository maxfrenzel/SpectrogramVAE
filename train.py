import tensorflow as tf
import os
from shutil import copyfile
import sys
import time
import joblib
from random import shuffle
import numpy as np
import argparse
import json

from spec_reader import *
from model_iaf import *

logdir = './logdir'
max_checkpoints = 5
num_steps = 10000
checkpoint_every = 500
batch_size = 64
learning_rate = 1e-3
beta=1.0
model_params = 'params.json'

def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='Spectrogram VAE')
    parser.add_argument('--batch_size', type=int, default=batch_size,
                        help='How many wav files to process at once. Default: ' + str(batch_size) + '.')
    parser.add_argument('--logdir', type=str, default=None,
                        help='Directory in which to store the logging '
                        'information for TensorBoard. '
                        'If the model already exists, it will restore '
                        'the state and will continue training. ')
    parser.add_argument('--checkpoint_every', type=int,
                        default=checkpoint_every,
                        help='How many steps to save each checkpoint after. Default: ' + str(checkpoint_every) + '.')
    parser.add_argument('--num_steps', type=int, default=num_steps,
                        help='Number of training steps. Default: ' + str(num_steps) + '.')
    parser.add_argument('--learning_rate', type=float, default=learning_rate,
                        help='Learning rate for training. Default: ' + str(learning_rate) + '.')
    parser.add_argument('--beta', type=float, default=beta,
                        help='Factor for KL divergence term in loss. Default: ' + str(beta) + '.')
    parser.add_argument('--model_params', type=str, default=model_params,
                        help='JSON file with the network parameters. Default: ' + model_params + '.')
    parser.add_argument('--max_checkpoints', type=int, default=max_checkpoints,
                        help='Maximum amount of checkpoints that will be kept alive. Default: '
                             + str(max_checkpoints) + '.')
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

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    # If restarting an existing model, look for original parameters
    if os.path.isfile(f'{args.logdir}/params.json'):
        print('Loading existing parameters.')
        print(f'{args.logdir}/params.json')
        with open(f'{args.logdir}/params.json', 'r') as f:
            param = json.load(f)
    # Otherwise load new one and copy to logdir
    else:
        print('Starting with new parameters.')
        # Load model parameters
        with open(args.model_params, 'r') as f:
            param = json.load(f)
        copyfile(args.model_params, f'{args.logdir}/params.json')

    # Set correct batch size in deconvolution shapes
    deconv_shape = param['deconv_shape']
    for k, s in enumerate(deconv_shape):
        actual_shape = s
        actual_shape[0] = args.batch_size
        deconv_shape[k] = actual_shape
    param['deconv_shape'] = deconv_shape

    # Load data
    melspecs = load_specs()
    # melspecs = 80.0*(np.random.random((10000,128,126))-1.0)

    # Create coordinator.
    coord = tf.train.Coordinator()

    with tf.name_scope('create_inputs'):
        reader = SpectrogramReader(melspecs, coord)
        spec_batch = reader.dequeue(args.batch_size)

    # Create network.
    net = VAEModel(param,
                   args.batch_size)

    loss = net.loss(spec_batch, beta=args.beta)

    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate,
                                       epsilon=1e-4)
    trainable = tf.trainable_variables()
    for var in trainable:
        print(var)
    optim = optimizer.minimize(loss, var_list=trainable)

    # Set up logging for TensorBoard.
    writer = tf.summary.FileWriter(args.logdir)
    writer.add_graph(tf.get_default_graph())
    run_metadata = tf.RunMetadata()
    summaries = tf.summary.merge_all()
    print(summaries)

    # Set up session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    init = tf.global_variables_initializer()
    sess.run(init)

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=args.max_checkpoints)

    try:
        saved_global_step = load(saver, sess, args.logdir)
        if saved_global_step is None:
            # The first training step will be saved_global_step + 1,
            # therefore we put -1 here for new or overwritten trainings.
            saved_global_step = -1

    except:
        print("Something went wrong while restoring checkpoint. "
              "We will terminate training to avoid accidentally overwriting "
              "the previous model.")
        raise

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    reader.start_threads(sess)

    step = None
    last_saved_step = saved_global_step
    try:
        for step in range(saved_global_step + 1, num_steps):
            start_time = time.time()

            # loss_value = sess.run([loss])[0]
            # print(loss_value)
            summary, loss_value, _ = sess.run([summaries, loss, optim])

            writer.add_summary(summary, step)

            duration = time.time() - start_time
            print('step {:d} - loss = {:.3f}, ({:.3f} sec/step)'
                  .format(step, loss_value, duration))

            if step % args.checkpoint_every == 0:
                save(saver, sess, args.logdir, step)
                last_saved_step = step

    except KeyboardInterrupt:
        # Introduce a line break after ^C is displayed so save message
        # is on its own line.
        print()
    finally:
        if step > last_saved_step:
            save(saver, sess, args.logdir, step)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()