###############################################################################
# Copyright (c) Futurewei Technologies, inc. All Rights Reserved.
#
# Implementation of FedMom aggregator.
# Author: Hongwei Jin (hjin@futurewei.com), 2020-08
###############################################################################

import numpy as np
import tensorflow as tf
from tqdm import trange, tqdm

from .fedbase import BaseFedarated
from fedsimul.utils.tf_utils import process_grad


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        """ Initialization of FedMon server

        Args:
            params: dictionary
            learner: learning model class
            dataset: tuple
                user, group, train_data, test_data
        """
        print("Using Federated Momentum to Train")
        self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
        return super(Server, self).__init__(params, learner, dataset)

    def train(self):
        """ Train using Federated Momentum. """
        print("Training with participate rate of {}".format(self.participate_rate))
        for i in trange(self.num_rounds, desc='Round: ', ncols=120):

            self.p_clients = self.select_clients(i, num_clients=self.clients_p_round[i])
            self.r_clients = self.select_clients(i, num_clients=self.clients_r_round[i])
            self.current_seq += 1

            # test model
            if i % self.eval_every == 0:

                stats_test = self.test()
                stats_train = self.train_error_and_loss()
                # self.metrics.accuracies.append(stats)
                # self.metrics.train_accuracies.append(stats_train)

                model_len = process_grad(self.latest_model).size

                global_grads = np.zeros(model_len)
                num_samples = []
                local_grads = []

                for c in self.clients:
                    # set window size
                    num, client_grad = c.get_grads(model_len)
                    local_grads.append(client_grad)
                    num_samples.append(num)
                    global_grads = np.add(global_grads, client_grad * num)
                global_grads = global_grads * 1.0 / sum(num_samples)

                difference = 0
                for lg in local_grads:
                    difference += np.linalg.norm(global_grads - lg)**2
                difference = difference * 1.0 / len(self.clients)

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
                    'grad_diff:{:.4f}'.format(difference))

                self.metrics.accuracies.append(test_acc)
                self.metrics.train_accuracies.append(train_acc)

            csolns = []
            # self.inner_opt.set_params(self.latest_model, self.client_model)
            for c in tqdm(self.p_clients, desc='Client: ', leave=False, ncols=120):
                if self.asyn and self.current_seq - c.seq_id > self.window_size:
                    continue

                # REVIEW: c.set_params(self.saved_models[c.seq_id])
                c.set_params(self.saved_models[c.seq_id], momentum=True)
                # c.set_params(self.latest_model, momentum=True)

                soln, stats = c.solve_inner(self.current_seq, num_epochs=self.num_epochs, batch_size=self.batch_size)

                csolns.append(soln)

                self.metrics.update(rnd=i, cid=c.id, stats=stats)

            # self.latest_model = self.aggregate(csolns, gamma=(1 - len(self.p_clients) / len(self.clients)))
            self.latest_model = self.aggregate(csolns, gamma=0.9)
            self.saved_models[self.current_seq] = self.latest_model
            if len(self.saved_models.keys()) > self.window_size:
                del self.saved_models[min(self.saved_models.keys())]

            for c in self.r_clients:
                c.seq_id = self.current_seq

        # final test model
        stats = self.test()
        stats_train = self.train_error_and_loss()

        tqdm.write('At round {} accuracy: {}'.format(self.num_rounds, np.sum(stats[3]) * 1.0 / np.sum(stats[2])))
        tqdm.write('At round {} training accuracy: {}'.format(
            self.num_rounds, np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])))

        # save server model
        self.metrics.write()
