###############################################################################
# Copyright (c) Futurewei Technologies, inc. All Rights Reserved.
#
# Preprocess the MNIST dataset
# Author: Hongwei Jin (hjin@futurewei.com), 2020-08
###############################################################################


from sklearn.datasets import fetch_openml
from tqdm import trange, tqdm
import numpy as np
import random
import json
import os
import os.path as osp
import tempfile

tmpdir = tempfile.gettempdir()

# Setup directory for train/test data
train_path = osp.join(tmpdir, 'data', 'mnist', 'train')
train_file = osp.join(train_path, 'all_data_0_niid_0_keep_10_train_9.json')
test_path = os.path.join(tmpdir, 'data', 'mnist', 'test')
test_file = osp.join(test_path, 'all_data_0_niid_0_keep_10_test_9.json')
for path in [train_path, test_path]:
    if not osp.exists(path):
        os.makedirs(path)

# Get MNIST data, normalize, and divide by level
mnist = fetch_openml('mnist_784', data_home=os.path.join(tmpdir, 'data', 'mnist', 'data'))
mu = np.mean(mnist.data.astype(np.float32), 0)
sigma = np.std(mnist.data.astype(np.float32), 0)
mnist.data = (mnist.data.astype(np.float32) - mu) / (sigma + 0.001)
mnist.target = mnist.target.astype(int)
mnist_data = []
for i in trange(10, ncols=120, desc='process labels'):
    idx = mnist.target == i
    mnist_data.append(mnist.data[idx])

# Get MNIST data splitted by labels
# print([len(v) for v in mnist_data])

###### CREATE USER DATA SPLIT #######
# Assign 10 samples to each user
X = [[] for _ in range(1000)]
y = [[] for _ in range(1000)]
idx = np.zeros(10, dtype=np.int64)
for user in trange(1000, ncols=120, desc='split user data'):
    for j in range(2):
        l = (user + j) % 10
        X[user] += mnist_data[l][idx[l]:idx[l] + 5].tolist()
        y[user] += (l * np.ones(5)).tolist()
        idx[l] += 5

# print(idx)

# Assign remaining sample by power law
user = 0
props = np.random.lognormal(0, 2.0, (10, 100, 2))
props = np.array([[[len(v) - 1000]] for v in mnist_data]) * \
    props / np.sum(props, (1, 2), keepdims=True)

# idx = 1000*np.ones(10, dtype=np.int64)
for user in trange(1000, ncols=120, desc='assign samples'):
    for j in range(2):
        l = (user + j) % 10
        num_samples = int(props[l, user // 10, j])
        # print(num_samples)
        if idx[l] + num_samples < len(mnist_data[l]):
            X[user] += mnist_data[l][idx[l]:idx[l] + num_samples].tolist()
            y[user] += (l * np.ones(num_samples)).tolist()
            idx[l] += num_samples

# print(idx)

# Create data structure
train_data = {'users': [], 'user_data': {}, 'num_samples': []}
test_data = {'users': [], 'user_data': {}, 'num_samples': []}

# Setup 1000 users
for i in trange(1000, ncols=120, desc='setup user'):
    uname = 'f_{0:05d}'.format(i)

    combined = list(zip(X[i], y[i]))
    random.shuffle(combined)
    X[i][:], y[i][:] = zip(*combined)
    num_samples = len(X[i])
    train_len = int(0.9 * num_samples)
    test_len = num_samples - train_len

    train_data['users'].append(uname)
    train_data['user_data'][uname] = {
        'x': X[i][:train_len], 'y': y[i][:train_len]}
    train_data['num_samples'].append(train_len)
    test_data['users'].append(uname)
    test_data['user_data'][uname] = {
        'x': X[i][train_len:], 'y': y[i][train_len:]}
    test_data['num_samples'].append(test_len)

# REVIEW: check statistic of samples
# print(train_data['num_samples'])
# print(sum(train_data['num_samples']))

print('write to json files')
with open(train_file, 'w') as outfile:
    json.dump(train_data, outfile)
with open(test_file, 'w') as outfile:
    json.dump(test_data, outfile)
