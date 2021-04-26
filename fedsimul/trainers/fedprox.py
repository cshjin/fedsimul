import numpy as np
from tqdm import tqdm

from .fedbase import BaseFedarated
from fedsimul.optimizer.pgd import PerturbedGradientDescent
from fedsimul.utils.tf_utils import process_grad
import tensorflow as tf


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using Federated Proximal to Train')
        self.inner_opt = PerturbedGradientDescent(params['learning_rate'], params['mu'])
        # self.inner_opt = tf.train.ProximalGradientDescentOptimizer(
        #     params['learning_rate'], l2_regularization_strength=params['mu'])
        # self.seed = 1
        super(Server, self).__init__(params, learner, dataset)

    def train(self):
        '''Train using Federated Proximal'''
        print('Training with {} workers ---'.format(self.clients_per_round))

        for i in range(self.num_rounds):
            # test model
            if i % self.eval_every == 0:
                stats_test = self.test()  # have set the latest model for all clients
                stats_train = self.train_error_and_loss()

                model_len = process_grad(self.latest_model).size
                global_grads = np.zeros(model_len)
                client_grads = np.zeros(model_len)
                num_samples = []
                local_grads = []

                for c in self.clients:
                    num, client_grad = c.get_grads(model_len)
                    local_grads.append(client_grad)
                    num_samples.append(num)
                    global_grads = np.add(global_grads, client_grads * num)
                global_grads = global_grads * 1.0 / sum(num_samples)

                difference = 0
                for idx in range(len(self.clients)):
                    difference += np.sum(np.square(global_grads - local_grads[idx]))
                difference = difference * 1.0 / len(self.clients)

                tqdm.write(
                    'round:{:03d} '.format(i + 1) +
                    'test_acc:{:.4f} '.format(np.sum(stats_test[3]) * 1.0 / np.sum(stats_test[2])) +
                    'train_acc:{:.4f} '.format(np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])) +
                    'train_loss:{:.4f} '.format(np.dot(stats_train[4], stats_train[2]) * 1.0 / np.sum(stats_train[2])) +
                    'grad_diff:{:.4f}'.format(difference))

            selected_clients = self.select_clients(i, num_clients=self.clients_per_round[i])

            csolns = []  # buffer for receiving client solutions

            # self.inner_opt.set_params(self.latest_model, self.client_model)

            for c in selected_clients:
                # communicate the latest model
                last_model = c.get_params()
                self.inner_opt.set_params(self.latest_model, self.client_model)
                c.set_params(last_model)

                # solve minimization locally
                c_soln, c_stats = c.solve_inner(num_epochs=self.num_epochs[i], batch_size=self.batch_size)

                # gather solutions from client
                csolns.append(c_soln)

                # track communication cost
                self.metrics.update(rnd=i, cid=c.id, stats=c_stats)

            # update model
            self.latest_model = self.aggregate(csolns)
            self.client_model.set_params(self.latest_model)

        # final test model
        stats = self.test()
        stats_train = self.train_error_and_loss()
        self.metrics.accuracies.append(stats)
        self.metrics.train_accuracies.append(stats_train)
        tqdm.write('At round {} accuracy: {:.4f}'.format(self.num_rounds, np.sum(stats[3]) * 1.0 / np.sum(stats[2])))
        tqdm.write('At round {} training accuracy: {:.4f}'.format(
            self.num_rounds, np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])))
