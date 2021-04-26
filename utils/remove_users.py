'''
removes users with less than the given number of samples
'''

import argparse
import json
import os
import os.path as osp
import tempfile

parser = argparse.ArgumentParser()

parser.add_argument('--name',
                    help='name of dataset to parse; default: nist;',
                    type=str,
                    default='nist')

parser.add_argument('--min_samples',
                    help='users with less than x samples are discarded; default: 10;',
                    type=int,
                    default=10)

args = parser.parse_args()


print('------------------------------')

tmpdir = tempfile.gettempdir()
data_path = osp.join(tmpdir, 'data', args.name)
subdir = osp.join(data_path, 'sampled_data')
files = []
if osp.exists(subdir):
    files = os.listdir(subdir)
if len(files) == 0:
    subdir = osp.join(data_path, 'all_data')
    files = os.listdir(subdir)
files = [f for f in files if f.endswith('.json')]

for f in files:
    users = []
    hierarchies = []
    num_samples = []
    user_data = {}

    min_number = 0
    max_number = 0

    file_dir = osp.join(subdir, f)
    with open(file_dir, 'r') as inf:
        data = json.load(inf)

    num_users = len(data['users'])
    for i in range(num_users):
        curr_user = data['users'][i]
        curr_hierarchy = None
        if 'hierarchies' in data:
            curr_hierarchy = data['hierarchies'][i]
        curr_num_samples = data['num_samples'][i]

        if curr_num_samples > args.min_samples:
            user_data[curr_user] = data['user_data'][curr_user]
            users.append(curr_user)
            max_number += 1
            if curr_hierarchy is not None:
                hierarchies.append(curr_hierarchy)
            num_samples.append(data['num_samples'][i])

    all_data = {}
    all_data['users'] = users
    if len(hierarchies) == len(users):
        all_data['hierarchies'] = hierarchies
    all_data['num_samples'] = num_samples
    all_data['user_data'] = user_data

    file_name = '%s_keep_%d.json' % ((f[:-5]), args.min_samples)
    if not osp.exists(osp.join(data_path, 'rem_user_data')):
        os.makedirs(osp.join(data_path, 'rem_user_data'))
    ouf_dir = osp.join(data_path, 'rem_user_data', file_name)

    print('writing %s' % file_name)
    with open(ouf_dir, 'w') as outfile:
        json.dump(all_data, outfile)
