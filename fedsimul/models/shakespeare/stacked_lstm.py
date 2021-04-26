from fedsimul.utils.tf_utils import process_sparse_grad
from fedsimul.utils.tf_utils import graph_size
from fedsimul.utils.language_utils import letter_to_vec, word_to_indices
from fedsimul.utils.model_utils import batch_data
import numpy as np
from tqdm import trange

import os
import sys
import tensorflow as tf

from tensorflow.contrib import rnn


def process_x(raw_x_batch):
    """ Convert raw input.
    Args:
        raw_x_batch: np.ndarray with size (20, )
            raw input

    Returns:
        x_batch: np.ndarray with size (20, 80)
    """
    x_batch = [word_to_indices(word) for word in raw_x_batch]
    x_batch = np.array(x_batch)
    return x_batch


def process_y(raw_y_batch):
    """ Convert raw output.
    Args:
        raw_y_batch: np.ndarray with size (20, ).

    Returns:
        y_batch: list with len 20.
    """
    y_batch = [letter_to_vec(c) for c in raw_y_batch]
    return y_batch


class Model(object):
    def __init__(self, seq_len, num_classes, n_hidden, optimizer, seed=1, gpu_id=0):
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.n_hidden = n_hidden

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(123 + seed)
            self.features, self.labels, self.train_op, self.grads, self.eval_metric_ops, self.loss = self.create_model(
                optimizer)
            self.saver = tf.train.Saver()
        # gpu_options = tf.compat.v1.GPUOptions(visible_device_list="{}".format(gpu_id), allow_growth=True)
        # config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
        config = tf.ConfigProto(device_count={'GPU': 0})
        self.sess = tf.Session(graph=self.graph, config=config)

        self.size = graph_size(self.graph)

        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())

            metadata = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            self.flops = tf.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops

    def create_model(self, optimizer):
        """ Create model instance
        Args:
            optimizer:          tf.compat.v1.train.Optimizer

        Returns: 
            tuple: 
                features:           tf.placeholder
                labels:             tf.placeholder
                train_op:           Operation
                grads:              tf.tensor
                eval_metric_ops:    tf.tensor
                loss:               tf.tensor
        """
        features = tf.placeholder(tf.int32, [None, self.seq_len])
        embedding = tf.get_variable("embedding", [self.num_classes, 8])
        x = tf.nn.embedding_lookup(embedding, features)
        labels = tf.placeholder(tf.int32, [None, self.num_classes])

        stacked_lstm = tf.compat.v1.nn.rnn_cell.MultiRNNCell([rnn.BasicLSTMCell(self.n_hidden) for _ in range(2)])
        outputs, _ = tf.compat.v1.nn.dynamic_rnn(stacked_lstm, x, dtype=tf.float32)
        pred = tf.compat.v1.layers.Dense(units=self.num_classes)(outputs[:, -1, :])

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=labels))

        # train_op = optimizer.minimize(loss)
        grads_and_vars = optimizer.compute_gradients(loss)
        grads, _ = zip(*grads_and_vars)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
        eval_metric_ops = tf.count_nonzero(correct_pred)

        return features, labels, train_op, grads, eval_metric_ops, loss

    def set_params(self, model_params=None, momentum=False, gamma=0.9):
        """ Set parameters to client

        Args:
            model_params: tf.tensor
            momentum: boolean
            gamma: float
                Default: gamma = 0.9
        """
        if model_params is not None:
            with self.graph.as_default():
                all_vars = tf.trainable_variables()
                for variable, value in zip(all_vars, model_params):
                    variable.load(value, self.sess)

    def get_params(self):
        """ Get parameters.

        Returns:
            model_params: tf.tensor
        """
        with self.graph.as_default():
            model_params = self.sess.run(tf.trainable_variables())
        return model_params

    def get_gradients(self, data, model_len):
        '''in order to avoid the OOM error, we need to calculate the gradients on each 
        client batch by batch. batch size here is set to be 100.

        Return: a one-D array (after flattening all gradients)
        '''
        grads = np.zeros(model_len)
        num_samples = len(data['y'])

        processed_samples = 0

        if num_samples < 50:
            input_data = process_x(data['x'])
            target_data = process_y(data['y'])
            with self.graph.as_default():
                model_grads = self.sess.run(self.grads, feed_dict={self.features: input_data, self.labels: target_data})
            grads = process_sparse_grad(model_grads)
            processed_samples = num_samples

        else:  # in order to fit into memory, compute gradients in a batch of size 50, and subsample a subset of points to approximate

            for i in range(min(int(num_samples / 50), 4)):
                input_data = process_x(data['x'][50 * i: 50 * (i + 1)])
                target_data = process_y(data['y'][50 * i: 50 * (i + 1)])

                with self.graph.as_default():
                    model_grads = self.sess.run(self.grads,
                                                feed_dict={self.features: input_data, self.labels: target_data})

                flat_grad = process_sparse_grad(model_grads)
                grads = np.add(grads, flat_grad)

            grads = grads * 1.0 / min(int(num_samples / 50), 4)
            processed_samples = min(int(num_samples / 50), 4) * 50

        return processed_samples, grads

    def solve_inner(self, data, num_epochs=1, batch_size=32):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        Return:
            soln: trainable variables of the lstm model
            comp: number of FLOPs computed while training given data
        '''
        for _ in trange(num_epochs, desc='Epoch: ', leave=False):
            for X, y in batch_data(data, batch_size):
                input_data = process_x(X)
                target_data = process_y(y)
                with self.graph.as_default():
                    self.sess.run(self.train_op,
                                  feed_dict={self.features: input_data, self.labels: target_data})
        soln = self.get_params()
        comp = num_epochs * (len(data['y'])//batch_size) * batch_size * self.flops
        return soln, comp

    def test(self, data):
        '''
        Args:
            data: dict of the form {'x': [], 'y': []}

        Returns:
            tot_correct: int
            loss: float
        '''
        x_vecs = process_x(data['x'])
        labels = process_y(data['y'])
        with self.graph.as_default():
            tot_correct, loss = self.sess.run([self.eval_metric_ops, self.loss],
                                              feed_dict={self.features: x_vecs, self.labels: labels})
        return tot_correct, loss

    def close(self):
        self.sess.close()
