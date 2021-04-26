###############################################################################
# Federated averaging aggregator.
#
# NOTE: source from https://github.com/litian96/FedProx
# Modified by Hongwei Jin, 2020-08
#   * rewrite the `aggregate` function in terms of adaptive aggregation with
#       asynchrony and aggregation rate.
###############################################################################
import numpy as np
import tensorflow as tf

from fedsimul.models.client import Client
from fedsimul.utils.model_utils import Metrics


class BaseFedarated(object):
    def __init__(self, params, learner, dataset):
        """ Inintialize the BaseFederated class.

        Args:
            params: dict
                dictionary of parsed arguments
            learning: model class
                example like mclr, stacked_lstm and etc.
            dataset: tuple
                tuple of (user, group, train_data, test_data)
        """
        # transfer parameters to self
        for key, val in params.items():
            setattr(self, key, val)
        n_clients = len(dataset[2])

        # create worker nodes
        tf.reset_default_graph()

        # get client model
        self.client_model = learner(*params['model_params'], self.inner_opt, self.gpu_id, self.seed)

        # get clients
        self.clients = self.setup_clients(dataset, self.client_model)
        print('{} Clients in Total'.format(len(self.clients)))

        # REVIEW: setup asynchronous parameters
        if self.asyn:
            p_clients = int(self.participate_rate * len(self.clients))
            r_clients = int(self.refresh_rate * len(self.clients))
            if self.adp_p:
                # self.clients_p_round = np.arange(p_clients, p_clients + self.num_rounds)
                self.clients_p_round = np.arange(p_clients + self.num_rounds, p_clients, -1)
            else:
                self.clients_p_round = np.maximum([int(n_clients * self.participate_rate)] * self.num_rounds, 1)
            if self.adp_q:
                # self.clients_r_round = np.arange(r_clients, r_clients + self.num_rounds)
                self.clients_r_round = np.arange(r_clients + self.num_rounds, r_clients, -1)
            else:
                self.clients_r_round = np.maximum([int(n_clients * self.refresh_rate)] * self.num_rounds, 1)
        else:
            self.clients_p_round = np.maximum([int(n_clients * self.participate_rate)] * self.num_rounds, 1)
            self.clients_r_round = np.maximum([int(n_clients * self.refresh_rate)] * self.num_rounds, 1)

        # REVIEW: add simulated seq id, initiate with 0
        self.current_seq = 0
        # self.seq_ids = {x.id: 0 for x in self.clients}

        # REVIEW: set latest model to be a queue
        self.latest_model = self.client_model.get_params()
        self.saved_models = {0: self.latest_model}

        # initialize system metrics
        self.metrics = Metrics(self.clients, params)

    def __del__(self):
        """ Close client's session. """
        self.client_model.close()

    def setup_clients(self, dataset, model=None):
        ''' Instantiates clients based on given train and test data directories

        Args:
            dataset:
            model:

        Returns:
            all_clients: list
                list of Clients
        '''
        users, groups, train_data, test_data = dataset
        if len(groups) == 0:
            groups = [None for _ in users]
        all_clients = [Client(u, g, train_data[u], test_data[u], model)
                       for u, g in zip(users, groups)]
        return all_clients

    def train_error_and_loss(self):
        """ Evaluate training.

        Returns:
            ids: list
                list of ids for all clients
            groups: list
                list of groups for all clients
            num_samples: list
                list of number of samples for all clients
            tot_correct: list
                list of total corrected samples for all clients
            losses: list
                list of training losses for all clients
        """
        num_samples = []
        tot_correct = []
        losses = []

        for c in self.clients:
            ct, cl, ns = c.evals(c.train_data)
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]

        return ids, groups, num_samples, tot_correct, losses

    def test(self):
        ''' Tests self.latest_model on given clients

        Returns:
            ids:            list
            groups:         list
            num_samples:    list
            tot_correct:    list

        TODO: test based on different client sets
            all_clients, p_clients, r_clients
        '''
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]
        num_samples = [0 for _ in range(len(self.clients))]
        tot_correct = [0 for _ in range(len(self.clients))]

        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            ct, cl, ns = c.evals(c.test_data)
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
        return ids, groups, num_samples, tot_correct

    def select_clients(self, round, num_clients=20):
        ''' Selects num_clients clients weighted by number of samples from possible_clients

        Args:
            round : int
            num_clients: int
                number of clients to select; default 20

        Notes:
            within function, num_clients is set to be min(num_clients, len(possible_clients))

        Returns:
            np.array : list of selected clients objects
        '''
        num_clients = max(min(num_clients, len(self.clients)), 2)
        np.random.seed(round)
        return np.random.choice(self.clients, num_clients, replace=False)

    def aggregate(self, wsolns, alpha=0., gamma=1.):
        """ Weighted average aggregation on server
        latest_model = weighted sum of s^alpha * theta_c.

        Args:
            wsolns : list of tuples
                (num of samples, weights, staleness)
            alpha: polynomial in the discount function
                Default: 0, no discount
                ..Math: staleness^{-alpha}

        Returns:
            average_soln : list
                weighted average
        """
        total_samples = 0.0
        base = [0] * len(wsolns[0][1])
        for (n, soln, s) in wsolns:
            total_samples += n
            for i, v in enumerate(soln):
                # REVIEW: with discounted sigma
                base[i] += n * (s ** -alpha) * v
        # local average
        averaged_soln = [v / total_samples for v in base]

        # REVIEW: momentum aggregation
        averaged_soln = [(1 - gamma) * x + gamma * y for x, y in zip(self.latest_model, averaged_soln)]

        return averaged_soln
