###############################################################################
# Copyright (c) Futurewei Technologies, inc. All Rights Reserved.
#
# Fetch whole FEMNIST dataset with handwritings by labeled writers.
# Author: Hongwei Jin (hjin@futurewei.com), 2020-08
###############################################################################
from tqdm import tqdm
from zipfile import ZipFile
import hashlib
import os
import os.path as osp
import pickle
import requests
import tempfile
import math
from PIL import Image
import numpy as np
import json


def get_file_dirs():
    """ Creates .pkl files for:
        1. list of directories of every image in 'by_class'
        2. list of directories of every image in 'by_write'
        the hierarchal structure of the data is as follows:
        - by_class -> classes -> folders containing images -> images
        - by_write -> folders containing writers -> writer -> types of images -> images
        the directories written into the files are of the form 'raw_data/...'
    """

    class_files = []
    write_files = []

    if not osp.exists(osp.join(data_path, 'intermediate')):
        os.makedirs(osp.join(data_path, 'intermediate'))
    # class file
    class_dir = osp.join(data_path, "by_class")
    rel_class_dir = os.path.join('by_class')
    classes = os.listdir(class_dir)

    for cl in classes:
        cldir = os.path.join(class_dir, cl)
        rel_cldir = os.path.join(rel_class_dir, cl)
        subcls = os.listdir(cldir)

        subcls = [s for s in subcls if (('hsf' in s) and ('mit' not in s))]

        for subcl in subcls:
            subcldir = os.path.join(cldir, subcl)
            rel_subcldir = os.path.join(rel_cldir, subcl)
            images = os.listdir(subcldir)
            image_dirs = [os.path.join(rel_subcldir, i) for i in images]

            for image_dir in image_dirs:
                class_files.append((cl, image_dir))

    write_dir = os.path.join(data_path, 'by_write')
    rel_write_dir = os.path.join('by_write')
    write_parts = os.listdir(write_dir)

    for write_part in write_parts:
        writers_dir = os.path.join(write_dir, write_part)
        rel_writers_dir = os.path.join(rel_write_dir, write_part)
        writers = os.listdir(writers_dir)

        for writer in writers:
            writer_dir = os.path.join(writers_dir, writer)
            rel_writer_dir = os.path.join(rel_writers_dir, writer)
            wtypes = os.listdir(writer_dir)

            for wtype in wtypes:
                type_dir = os.path.join(writer_dir, wtype)
                rel_type_dir = os.path.join(rel_writer_dir, wtype)
                images = os.listdir(type_dir)
                image_dirs = [os.path.join(rel_type_dir, i) for i in images]

                for image_dir in image_dirs:
                    write_files.append((writer, image_dir))

    with open(osp.join(data_path, 'intermediate', 'class_file_dirs.pkl'), 'wb') as f:
        pickle.dump(class_files, f, pickle.HIGHEST_PROTOCOL)
    with open(osp.join(data_path, 'intermediate', 'write_file_dirs.pkl'), 'wb') as f:
        pickle.dump(write_files, f, pickle.HIGHEST_PROTOCOL)


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def get_hashes():
    """ Get hashes from the class and write
    """
    cfd = osp.join(data_path, 'intermediate', 'class_file_dirs')
    wfd = osp.join(data_path, 'intermediate', 'write_file_dirs')

    class_file_dirs = load_obj(cfd)
    write_file_dirs = load_obj(wfd)

    class_file_hashes = []
    write_file_hashes = []

    count = 0
    for tup in tqdm(class_file_dirs, desc="hashed class images"):
        # if (count % 100000 == 0):
        #     print('hashed %d class images' % count)

        (cclass, cfile) = tup
        file_path = os.path.join(data_path, cfile)

        chash = hashlib.md5(open(file_path, 'rb').read()).hexdigest()

        class_file_hashes.append((cclass, cfile, chash))

        count += 1

    cfhd = os.path.join(data_path, 'intermediate', 'class_file_hashes')
    save_obj(class_file_hashes, cfhd)

    count = 0
    for tup in tqdm(write_file_dirs, desc="hashed write images"):
        # if (count % 100000 == 0):
        #     print('hashed %d write images' % count)

        (cclass, cfile) = tup
        file_path = os.path.join(data_path, cfile)

        chash = hashlib.md5(open(file_path, 'rb').read()).hexdigest()

        write_file_hashes.append((cclass, cfile, chash))

        count += 1

    wfhd = os.path.join(data_path, 'intermediate', 'write_file_hashes')
    save_obj(write_file_hashes, wfhd)


def match_hashes():
    cfhd = os.path.join(data_path, 'intermediate', 'class_file_hashes')
    wfhd = os.path.join(data_path, 'intermediate', 'write_file_hashes')
    class_file_hashes = load_obj(cfhd)  # each elem is (class, file dir, hash)
    write_file_hashes = load_obj(wfhd)  # each elem is (writer, file dir, hash)

    class_hash_dict = {}
    for i in range(len(class_file_hashes)):
        (c, f, h) = class_file_hashes[len(class_file_hashes) - i - 1]
        class_hash_dict[h] = (c, f)

    write_classes = []
    for tup in write_file_hashes:
        (w, f, h) = tup
        write_classes.append((w, f, class_hash_dict[h][0]))

    wwcd = os.path.join(data_path, 'intermediate', 'write_with_class')
    save_obj(write_classes, wwcd)


def group_by_writer():
    wwcd = os.path.join(data_path, 'intermediate', 'write_with_class')
    write_class = load_obj(wwcd)

    writers = []  # each entry is a (writer, [list of (file, class)]) tuple
    cimages = []
    (cw, _, _) = write_class[0]
    for (w, f, c) in write_class:
        if w != cw:
            writers.append((cw, cimages))
            cw = w
            cimages = [(f, c)]
        cimages.append((f, c))
    writers.append((cw, cimages))

    ibwd = os.path.join(data_path, 'intermediate', 'images_by_writer')
    save_obj(writers, ibwd)


def data_to_json():
    MAX_WRITERS = 100

    def relabel_class(c):
        ''' Maps hexadecimal class value (string) to a decimal number.
        Returns:
        - 0 through 9 for classes representing respective numbers
        - 10 through 35 for classes representing respective uppercase letters
        - 36 through 61 for classes representing respective lowercase letters
        '''
        if c.isdigit() and int(c) < 40:
            return (int(c) - 30)
        elif int(c, 16) <= 90:  # uppercase
            return (int(c, 16) - 55)
        else:
            return (int(c, 16) - 61)

    ibwd = osp.join(data_path, 'intermediate', 'images_by_writer')
    writers = load_obj(ibwd)

    num_json = int(math.ceil(len(writers) / MAX_WRITERS))

    users = [[] for _ in range(num_json)]
    num_samples = [[] for _ in range(num_json)]
    user_data = [{} for _ in range(num_json)]

    writer_count = 0
    json_index = 0
    for (w, l) in writers:

        users[json_index].append(w)
        num_samples[json_index].append(len(l))
        user_data[json_index][w] = {'x': [], 'y': []}

        size = 28, 28  # original image size is 128, 128
        for (f, c) in l:
            file_path = os.path.join(data_path, f)
            img = Image.open(file_path)
            gray = img.convert('L')
            gray.thumbnail(size, Image.ANTIALIAS)
            arr = np.asarray(gray).copy()
            vec = arr.flatten()
            vec = vec / 255  # scale all pixel values to between 0 and 1
            vec = vec.tolist()

            nc = relabel_class(c)

            user_data[json_index][w]['x'].append(vec)
            user_data[json_index][w]['y'].append(nc)

        writer_count += 1
        if not osp.exists(osp.join(data_path, "all_data")):
            os.makedirs(osp.join(data_path, 'all_data'))
        if writer_count == MAX_WRITERS:

            all_data = {}
            all_data['users'] = users[json_index]
            all_data['num_samples'] = num_samples[json_index]
            all_data['user_data'] = user_data[json_index]

            file_name = 'all_data_%d.json' % json_index
            file_path = osp.join(data_path, 'all_data', file_name)

            print('writing %s' % file_name)

            with open(file_path, 'w') as outfile:
                json.dump(all_data, outfile)

            writer_count = 0
            json_index += 1


if __name__ == "__main__":
    dataset = "FEMNIST"
    tmpdir = tempfile.gettempdir()
    data_path = osp.join(tmpdir, 'data', 'nist')
    if not osp.exists(data_path):
        os.makedirs(data_path)

    files = ['by_class.zip', 'by_write.zip']
    url_base = 'https://s3.amazonaws.com/nist-srd/SD19/'
    for raw_file in files:
        # download file if not exist
        if not osp.exists(osp.join(data_path, raw_file)):
            url = url_base + raw_file
            r = requests.get(url, stream=True)
            content_size = int(r.headers['Content-Length']) / 1024
            with open(osp.join(data_path, raw_file), 'wb') as outfile:
                for data in tqdm(iterable=r.iter_content(1024),
                                 total=content_size,
                                 desc="Download {}".format(raw_file)):
                    outfile.write(data)

        # extract zip file if not exist
        if not osp.isdir(osp.join(data_path, raw_file.split('.')[0])):
            with ZipFile(osp.join(data_path, raw_file)) as zf:
                for f in tqdm(iterable=zf.namelist(),
                              total=len(zf.namelist()),
                              desc="Extract {}".format(raw_file)):
                    zf.extract(member=f,
                               path=osp.join(data_path))

        get_file_dirs()
        get_hashes()
        match_hashes()
        group_by_writer()
        data_to_json()
