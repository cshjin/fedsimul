import numpy as np
import tensorflow as tf
from tqdm import trange

from fedsimul.utils.model_utils import batch_data
from fedsimul.utils.tf_utils import graph_size
from fedsimul.utils.tf_utils import process_grad


class Model(object):
    '''
    This is the tf model for the MNIST dataset with multiple class learner regression.
    Images are 28px by 28px.
    '''

    def __init__(self, num_classes, optimizer, gpu_id=0, seed=1):
        """ Initialize the learner.

        Args:
            num_classes: int
            optimizer: tf.train.Optimizer
            gpu_id: int, default 0
            seed: int, default 1
        """
        # params
        self.num_classes = num_classes

        # create computation graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(123 + seed)
            _created = self.create_model(optimizer)
            self.features = _created[0]
            self.labels = _created[1]
            self.train_op = _created[2]
            self.grads = _created[3]
            self.eval_metric_ops = _created[4]
            self.loss = _created[5]
            self.saver = tf.train.Saver()

        # set the gpu resources
        gpu_options = tf.compat.v1.GPUOptions(visible_device_list="{}".format(gpu_id), allow_growth=True)
        config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(graph=self.graph, config=config)
        # self.sess = tf.Session(graph=self.graph)

        # REVIEW: find memory footprint and compute cost of the model
        self.size = graph_size(self.graph)
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            metadata = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            self.flops = tf.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops

    def create_model(self, optimizer):
        """ Model function for Logistic Regression.

        Args:
            optimizer: tf.train.Optimizer

        Returns:
            tuple: (features, labels, train_op, grads, eval_metric_ops, loss)
        """
        features = tf.placeholder(tf.float32, shape=[None, 784], name='features')
        labels = tf.placeholder(tf.int64, shape=[None, ], name='labels')
        logits = tf.layers.dense(inputs=features,
                                 units=self.num_classes,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
        predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        grads_and_vars = optimizer.compute_gradients(loss)
        grads, _ = zip(*grads_and_vars)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())
        eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))
        return features, labels, train_op, grads, eval_metric_ops, loss

    def set_params(self, latest_params=None, momentum=False, gamma=0.9):
        """ Set parameters from server

        Args:
            latest_params: list
                list of tf.Variables
            momentum: boolean
            gamma: float

        TODO: update variable with its local variable and the value from
            latest_params

        TODO: DO NOT set_params from the global, instead, use the global gradient to update
        """

        if latest_params is not None:
            with self.graph.as_default():
                # previous gradient
                all_vars = tf.trainable_variables()
                for variable, value in zip(all_vars, latest_params):
                    if momentum:
                        curr_val = self.sess.run(variable)
                        new_val = gamma * curr_val + (1 - gamma) * value
                        # TODO: use `assign` function instead of `load`
                        variable.load(new_val, self.sess)
                    else:
                        variable.load(value, self.sess)

    def get_params(self):
        """ Get model parameters.

        Returns:
            model_params: list
                list of tf.Variables
        """
        with self.graph.as_default():
            model_params = self.sess.run(tf.trainable_variables())
        return model_params

    def get_gradients(self, data, model_len):
        """ Access gradients of a given dataset.

        Args:
            data: dict
            model_len: int

        Returns:
            num_samples: int
            grads: tuple
        """
        grads = np.zeros(model_len)
        num_samples = len(data['y'])

        with self.graph.as_default():
            model_grads = self.sess.run(self.grads, feed_dict={self.features: data['x'],
                                                               self.labels: data['y']})
            grads = process_grad(model_grads)

        return num_samples, grads

    def solve_inner(self, data, num_epochs=1, batch_size=32):
        '''Solves local optimization problem.

        Args:
            data: dict with format {'x':[], 'y':[]}
            num_epochs: int
            batch_size: int

        Returns:
            soln: list
            comp: float
        '''
        for _ in trange(num_epochs, desc='Epoch: ', leave=False, ncols=120):
            for X, y in batch_data(data, batch_size):
                with self.graph.as_default():
                    self.sess.run(self.train_op, feed_dict={self.features: X, self.labels: y})
        soln = self.get_params()
        comp = num_epochs * (len(data['y']) // batch_size) * batch_size * self.flops
        return soln, comp

    def test(self, data):
        '''
        Args:
            data: dict of the form {'x': [], 'y': []}

        Returns:
            tot_correct: int
            loss: float
        '''
        with self.graph.as_default():
            tot_correct, loss = self.sess.run([self.eval_metric_ops, self.loss],
                                              feed_dict={self.features: data['x'], self.labels: data['y']})
        return tot_correct, loss

    def close(self):
        self.sess.close()
