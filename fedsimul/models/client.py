import numpy as np


class Client(object):

    def __init__(self, id, group=None, train_data={'x': [], 'y': []}, test_data={'x': [], 'y': []}, model=None):
        """ Initiate the Client.

        Args:
            id: int
            group: default None
            train_data: dict with keys {x, y}
            test_data: dict with keys {x, y}
            model: Model
        """
        self.model = model
        self.id = id
        self.group = group
        self.train_data = {k: np.array(v) for k, v in train_data.items()}
        self.test_data = {k: np.array(v) for k, v in test_data.items()}
        self.num_samples = len(self.train_data['y'])
        self.test_samples = len(self.test_data['y'])
        # REVIEW: set the default seq_id
        self.seq_id = 0
        # REVIEW: get the norm of gradients
        self.norm_grads = -1

    def set_params(self, model_params, momentum=False):
        '''Set model parameters.

        Args:
            model_params
            momentum: boolean
                Default: False
        '''
        self.model.set_params(model_params, momentum)

    def get_params(self):
        '''Get model parameters.

        Returns:
            list: list of tf.Variable
        '''
        return self.model.get_params()

    def get_grads(self, model_len):
        '''Get model gradient.

        Args:
            model_len: int

        Returns:
            tuple: (int, np.array): (n_samples, gradients)
        '''
        return self.model.get_gradients(self.train_data, model_len)

    def solve_grad(self):
        '''Get model gradient with cost.

        Returns:
            tuple: ((1,1), (2,2,2))
                1: num_samples: number of samples used in training
                1: grads: local gradient
                2: bytes write: number of bytes received
                2: comp: number of FLOPs executed in training process
                2: bytes read: number of bytes transmitted
        '''
        bytes_w = self.model.size
        grads = self.model.get_gradients(self.train_data)

        comp = self.model.flops * self.num_samples
        bytes_r = self.model.size
        return ((self.num_samples, grads), (bytes_w, comp, bytes_r))

    def solve_inner(self, current_seq, num_epochs=1, batch_size=10):
        '''Solves local optimization problem

        Args:
            current_seq: int
            num_epochs: int
            batch_size: int

        Returns:
            tuple: ((1,1), (2,2,2))
                1: num_samples: number of samples used in training
                1: soln: local optimization solution
                1: staleness: int to represent the staleness in training
                2: bytes write: number of bytes received
                2: comp: number of FLOPs executed in training process
                2: bytes read: number of bytes transmitted
        '''
        bytes_w = self.model.size
        soln, comp = self.model.solve_inner(self.train_data, num_epochs, batch_size)
        bytes_r = self.model.size
        staleness = current_seq - self.seq_id
        return (self.num_samples, soln, staleness), (bytes_w, comp, bytes_r)

    # def train_error_and_loss(self):
    #     """ DEPRECATED! Use `evals` instead.
    #     Evaluate the error and loss of training data.

    #     Returns:
    #         n_correct: int
    #         loss: float
    #         train_samples: int
    #     """
    #     n_correct, loss = self.model.test(self.train_data)
    #     return n_correct, loss, self.num_samples

    # def test(self):
    #     ''' DEPRECATED! Use `evals` instead.
    #     Evaluate the error and loss of test data.

    #     Returns:
    #         n_correct: int
    #         loss: float
    #         test_samples: int
    #     '''
    #     n_correct, loss = self.model.test(self.test_data)
    #     return n_correct, loss, self.test_samples

    def evals(self, dataset):
        """ Eval the performance on the dataset

        Args:
            dataset : split of either training or testing

        Returns:
            n_correct : int
            loss : float
            n_sample : int

        NOTE:
            Overwrite `train_error_and_loss` and `test` functions
        """
        n_correct, loss = self.model.test(dataset)
        n_sample = len(dataset['y'])
        return n_correct, loss, n_sample
