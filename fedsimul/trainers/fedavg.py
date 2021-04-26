###############################################################################
# Federated averaging aggregator.
#
# NOTE: source from https://github.com/litian96/FedProx
# Modified by Hongwei Jin, 2020-08
#   * adapative fl training with adp_p, adp_q
#   * async aggregation with staleness on clients
###############################################################################
import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf

from .fedbase import BaseFedarated


class Server(BaseFedarated):
    """ Server class"""

    def __init__(self, params, learner, dataset):
        """ Initialize the Server

        Args:
            params: dictionary
            learner: learning model class
            dataset: tuple
                user, group, train_data, test_data
        """
        print('Using Federated Average to Train')
        self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)

    def train(self):
        ''' Train using Federated Averaging. '''

        for i in trange(self.num_rounds, desc='Round: ', ncols=120):
            self.p_clients = self.select_clients(i, num_clients=self.clients_p_round[i])
            self.r_clients = self.select_clients(i, num_clients=self.clients_r_round[i])
            self.current_seq += 1

            csolns = []
            # REVIEW: train with participate clients
            for c in tqdm(self.p_clients, desc='Client: ', leave=False, ncols=120):
                # ignore too old clients
                if self.asyn and self.current_seq - c.seq_id > self.window_size:
                    continue
                # REVIEW: set model from the saved_models
                c.set_params(self.saved_models[c.seq_id])
                # solve minimization locally
                soln, stats = c.solve_inner(self.current_seq, num_epochs=self.num_epochs, batch_size=self.batch_size)
                # gather solutions from client
                csolns.append(soln)
                # track communication cost
                self.metrics.update(rnd=i, cid=c.id, stats=stats)

            # REVIEW: aggregate with momentum
            self.latest_model = self.aggregate(csolns, alpha=self.alpha, gamma=(self.num_rounds - i) / self.num_rounds)
            self.saved_models[self.current_seq] = self.latest_model

            # REVIEW: maintain the size of saved model
            if len(self.saved_models.keys()) > self.window_size:
                del self.saved_models[min(self.saved_models.keys())]

            # eval and std output
            if i % self.eval_every == 0:
                stats_test = self.test()
                stats_train = self.train_error_and_loss()

                test_acc = np.sum(stats_test[3]) * 1.0 / np.sum(stats_test[2])
                train_acc = np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])
                train_loss = np.dot(stats_train[4], stats_train[2]) * 1.0 / np.sum(stats_train[2])

                tqdm.write(
                    'round:{:03d} '.format(i + 1) +
                    'p_clients:{:03d} '.format(len(self.p_clients)) +
                    'r_clients:{:03d} '.format(len(self.r_clients)) +
                    'test_acc:{:.4f} '.format(test_acc) +
                    'train_acc:{:.4f} '.format(train_acc) +
                    'train_loss:{:.4f} '.format(train_loss) +
                    'flops:{:d}'.format(self.metrics.flops_per_round[i]))

                self.metrics.accuracies.append(test_acc)
                self.metrics.train_accuracies.append(train_acc)

            # REVIEW: refresh seq of clients
            for c in self.r_clients:
                c.seq_id = self.current_seq

        # save server model to json
        self.metrics.write()
