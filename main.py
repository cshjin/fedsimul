import numpy as np
import argparse
import importlib
import random
import os
import tensorflow as tf
from fedsimul.utils.model_utils import read_data
import tempfile


# GLOBAL PARAMETERS
OPTIMIZERS = ['fedavg', 'fedprox', 'fedmom', 'fedmomprox']
DATASETS = ['mnist', 'nist', 'shakespeare']


MODEL_PARAMS = {
    'mnist.mclr': (10,),  # num_classes
    'nist.mclr': (10,),
    'shakespeare.stacked_lstm': (80, 80, 256),
}


def read_args():
    ''' Parse command line arguments or load defaults. '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer',
                        help='name of optimizer;',
                        type=str,
                        choices=OPTIMIZERS,
                        default='fedavg')
    parser.add_argument('--dataset',
                        help='name of dataset;',
                        type=str,
                        choices=DATASETS,
                        default='mnist')
    parser.add_argument('--model',
                        help='name of model;',
                        type=str,
                        default='mclr')
    parser.add_argument('--num_rounds',
                        help='number of rounds to simulate;',
                        type=int,
                        default=100)
    parser.add_argument('--eval_every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=1)
    parser.add_argument('--batch_size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=10)
    parser.add_argument('--num_epochs',
                        help='number of epochs when clients train on data;',
                        type=int,
                        default=1)
    parser.add_argument('--learning_rate',
                        help='learning rate for inner solver;',
                        type=float,
                        default=0.01)
    parser.add_argument('--gamma',
                        help='constant for momentum',
                        type=float,
                        default=0.9)
    parser.add_argument('--seed',
                        help='seed for randomness;',
                        type=int,
                        default=0)
    parser.add_argument('--gpu_id',
                        help='gpu_id',
                        type=int,
                        default=0)
    parser.add_argument('--verbose',
                        help='toggle the verbose output',
                        action='store_true')
    # Async
    parser.add_argument('--asyn', '-a',
                        help='toggle asynchronous simulation',
                        action='store_true')
    parser.add_argument('--participate_rate', '-p',
                        help='probability of participating',
                        type=float,
                        default=0.02)
    parser.add_argument('--refresh_rate', '-q',
                        help='probability of refreshing',
                        type=float,
                        default=1)
    parser.add_argument('--adp_p',
                        help='toggle adaptive participate rate',
                        action='store_true')
    parser.add_argument('--adp_q',
                        help='toggle adaptive refresh rate',
                        action='store_true')
    parser.add_argument('--window_size', '-w',
                        help='moving window size',
                        type=int,
                        default=5)
    parser.add_argument('--alpha',
                        help='exponent in discount function',
                        type=float,
                        default=0.5)

    try:
        args = vars(parser.parse_args())
    except IOError as msg:
        parser.error(str(msg))

    if not args['asyn']:
        # args['participate_rate'] = None
        # args['refresh_rate'] = None
        args['window_size'] = 1
        args['alpha'] = 0.

    # Set seeds
    random.seed(1 + args['seed'])
    np.random.seed(12 + args['seed'])
    tf.set_random_seed(123 + args['seed'])

    # load selected model
    if args['dataset'].startswith("synthetic"):  # all synthetic datasets use the same model
        model_path = '%s.%s.%s.%s' % ('fedsimul', 'models', 'synthetic', args['model'])
    else:
        model_path = '%s.%s.%s.%s' % ('fedsimul', 'models', args['dataset'], args['model'])
    mod = importlib.import_module(model_path)
    model = getattr(mod, 'Model')

    # load selected trainer
    opt_path = 'fedsimul.trainers.%s' % args['optimizer']
    mod = importlib.import_module(opt_path)
    optimizer = getattr(mod, 'Server')

    # add selected model parameter
    args['model_params'] = MODEL_PARAMS['.'.join(model_path.split('.')[2:])]

    # print and return
    if args['verbose']:
        maxLen = max([len(ii) for ii in args.keys()])
        fmtString = '\t%' + str(maxLen) + 's : %s'
        print('Arguments:')
        for keyPair in sorted(args.items()):
            print(fmtString % keyPair)
    return args, model, optimizer


def main():

    # parse command line arguments
    args, model, optimizer = read_args()
    tmpdir = tempfile.gettempdir()

    # read data
    train_path = os.path.join(tmpdir, 'data', args['dataset'], 'train')
    test_path = os.path.join(tmpdir, 'data', args['dataset'], 'test')
    dataset = read_data(train_path, test_path)

    # REVIEW: distribution of dataset
    # train_data = dataset[2]
    # test_data = dataset[3]
    # dist = []
    # for k in train_data:
    #     dist.append(len(train_data[k]['x']) + len(test_data[k]['x']))
    # # print(dist)
    # import matplotlib.pyplot as plt
    # fig = plt.figure(figsize=(4, 3))
    # plt.hist(dist, bins=21)
    # # plt.title('MNIST')
    # plt.xlabel("number of samples")
    # plt.ylabel("number of users")
    # plt.tight_layout()
    # plt.savefig('mnist_hist.pdf')
    # exit()

    # call appropriate trainer
    t = optimizer(args, model, dataset)
    t.train()


if __name__ == '__main__':
    # suppress tf warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()
