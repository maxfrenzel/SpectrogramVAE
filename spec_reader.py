import threading
import random
import tensorflow as tf
import numpy as np
import joblib

def randomize_specs(specs):
    for k in range(specs.shape[0]):
        file_index = random.randint(0, (specs.shape[0] - 1))
        yield specs[file_index]


def return_spec(specs):
    randomized_specs = randomize_specs(specs)
    for spec in randomized_specs:
        # Convert from -80 to 0dB to range [0,1], and add channel dimension
        normalized_spec = np.expand_dims((spec + 80.0) / 80.0, 2)
        yield normalized_spec


class SpectrogramReader(object):
    def __init__(self,
                 specs,
                 coord,
                 queue_size=32):

        self.specs = specs
        self.coord = coord
        self.threads = []
        self.spec_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.queue = tf.PaddingFIFOQueue(queue_size,
                                         ['float32'],
                                         shapes=[(128, 126, 1)])
        self.enqueue = self.queue.enqueue([self.spec_placeholder])

    def dequeue(self, num_elements):
        output = self.queue.dequeue_many(num_elements)
        return output

    def thread_main(self, sess):
        stop = False
        # Go through the dataset multiple times
        while not stop:
            iterator = return_spec(self.specs)
            for spec in iterator:
                if self.coord.should_stop():
                    stop = True
                    break

                sess.run(self.enqueue,
                         feed_dict={self.spec_placeholder: spec})

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads

def load_specs(filename='dataset.pkl', return_filenames=False):
    print('Loading dataset.')
    #     with open('dataset.pkl', 'rb') as handle:
    #         dataset = pkl.load(handle)

    dataset = joblib.load(filename)

    print('Dataset loaded.')

    filenames = dataset['filenames']
    melspecs = dataset['melspecs']
    actual_lengths = dataset['actual_lengths']

    # Convert spectra to array
    melspecs = np.array(melspecs)

    if return_filenames:
        return melspecs, filenames
    else:
        return melspecs