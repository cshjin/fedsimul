###############################################################################
# Utils functions for models, including the Metrics class.
#
# NOTE: source from https: // github.com/litian96/FedProx
# Modified by Hongwei Jin, 2020-08
###############################################################################

import json
import numpy as np
import os


def batch_data(data, batch_size):
    '''
    Args:
        data: dict as {'x': [list], 'y': [list]}
        batch_size: int

    Returns:
        x: lists of size-batch_size
        y: lists of size-batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_y = data_y[i:i + batch_size]
        yield (batched_x, batched_y)


def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    Args:
        train_data_dir: str
        test_data_dir: str

    Assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = list(sorted(train_data.keys()))

    return clients, groups, train_data, test_data


class Metrics(object):
    def __init__(self, clients, params):
        """ Initialize system metrics.

        Args:
            clients: list
            params: dict
        """
        self.params = params
        num_rounds = params['num_rounds']
        self.bytes_written = {c.id: [0] * num_rounds for c in clients}
        # self.client_computations = {c.id: [0] * num_rounds for c in clients}
        self.flops_per_round = [0] * num_rounds
        self.bytes_read = {c.id: [0] * num_rounds for c in clients}
        self.accuracies = []
        self.train_accuracies = []

    def update(self, rnd, cid, stats):
        """ Update system metrics

        Args:
            rnd: int
            cid: int
            stats: tuple ( , , )
        """
        bytes_w, comp, bytes_r = stats
        # REVIEW: compute the total flop per round
        self.flops_per_round[rnd] += comp
        self.bytes_written[cid][rnd] += bytes_w
        self.bytes_read[cid][rnd] += bytes_r

    def write(self):
        """ Write the metrics result to json file under ./out folder. """
        metrics = {}
        for key, val in self.params.items():
            metrics[key] = val

        filename = "_".join(["{}".format(self.params[k]) for k in self.params if k is not 'model_params']) + ".json"
        # TODO: adding loss, and variance in gradient, p_client, r_client
        metrics['accuracies'] = self.accuracies
        metrics['train_accuracies'] = self.train_accuracies
        metrics['flops_per_round'] = self.flops_per_round

        metrics_dir = os.path.join('out', self.params['dataset'], filename)
        if not os.path.exists('out'):
            os.mkdir('out')
        if not os.path.exists(os.path.join('out', self.params['dataset'])):
            os.mkdir(os.path.join('out', self.params['dataset']))
        with open(metrics_dir, 'w') as ouf:
            json.dump(metrics, ouf, indent=2)
