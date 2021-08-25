import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import json
import argparse

font = {'family': 'sans-serif', 'size': 10}
matplotlib.rc('font', **font)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', help='specify the metric', type=str,
                        choices=['accuracy', 'flops'], default='accuracy')
    parser.add_argument('--outfile', help='filename of saved png', type=str, default='tmp')
    parser.add_argument('--file', help='files to plot', nargs="+")
    parser.add_argument('--legend', help='legend of figure', nargs="+")
    try:
        args = vars(parser.parse_args())
    except IOError as msg:
        parser.error(str(msg))

    files = args['file']
    # plot file
    fig = plt.figure(figsize=(4, 3))
    for json_file in files:
        if json_file.endswith('.json'):
            with open(json_file) as f:
                data = json.load(f)
                if args['metric'] == 'accuracy':
                    plt.plot(data['accuracies'], linewidth=1)
                else:
                    plt.semilogy(np.cumsum(data['flops_per_round']), linewidth=1)

    plt.xlabel(r'Round')
    ylabel = r'log of cumulative GFLOPS' if args['metric'] == 'flops' else r'Test accuracy'
    plt.ylabel(ylabel)
    # REVIEW: specify legend
    if args['legend'] is not None:
        plt.legend(args['legend'])
    plt.tight_layout()
    plt.savefig('tmp.png')
