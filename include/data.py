# import pickle
import cPickle as pickle
import numpy as np
import os
from urllib import urlretrieve
import tarfile
import zipfile
import sys
import pdb
import math

import matplotlib.pyplot as plt
from scipy import misc
from PIL import Image, ImageOps


def augment(x, aug_list=None, aug_prob=None):

    assert sum(aug_prob) == 1
    assert len(aug_prob) == len(aug_list)
    
    if aug_list == None:
        return x
    
    for idx, item in enumerate(x):
        dice = np.random.uniform(0,1)
        for p in range(len(aug_prob)):
            if dice <= sum(aug_prob[0:p+1]):
                aug = aug_list[p]
                break
        if aug[0] == "none":
            continue

        img = Image.fromarray(item.reshape(32, 32, 3))
        if aug[0] == "flip":
            img = ImageOps.flip(img)
        elif aug[0] == "mirror": 
            img = ImageOps.mirror(img)
        elif aug[0] == "mirror&flip":
            img = ImageOps.flip(img)
            img = ImageOps.mirror(img)
        elif aug[0] == "rotateCW":
            img = img.rotate(270)
        elif aug[0] == "rotateCCW": 
            img = img.rotate(90)
        elif aug[0] == "rotate": 
            img = img.rotate(int(aug[1]))
        x[idx] = np.asarray(img).reshape(-1)

    return x

def get_mbatch(name, mbatch_size, num_classes, aug_list, aug_prob):
    x, y = get_data_set(name)
    x, y = change_num_classes(x, y, num_classes)
    batch_len = x.shape[0]
    mbatch_num = int(math.ceil(float(batch_len)/mbatch_size))

    while 1: 
        order = np.random.permutation(batch_len)
        x_aug = augment(x, aug_list, aug_prob)

        for i in range(mbatch_num): 
            idx = order[i*mbatch_size : min(batch_len, (i+1)*mbatch_size)]
            yield x_aug[idx]/255.0, y[idx]


def get_data_set(name="train"):
    x = None
    y = None

    maybe_download_and_extract()
    folder_name = "cifar_100"

    f = open('./data_set/'+folder_name+'/meta', 'rb')
    f.close()

    if name is "train":
        f = open('./data_set/'+folder_name+'/train', 'rb')
        # datadict = pickle.load(f, encoding='latin1')
        datadict = pickle.load(f)
        f.close()

        x = datadict["data"]
        y = np.array(datadict['fine_labels'])
        
        # x = np.array(x, dtype=float) / 255.0
        x = x.reshape([-1, 3, 32, 32])
        x = x.transpose([0, 2, 3, 1])
        x = x.reshape(-1, 32*32*3)
	
    elif name is "test":
        f = open('./data_set/'+folder_name+'/test', 'rb')
        # datadict = pickle.load(f, encoding='latin1')
        datadict = pickle.load(f)
        f.close()

        x = datadict["data"]
        y = np.array(datadict['fine_labels'])

        x = np.array(x, dtype=float) / 255.0
        x = x.reshape([-1, 3, 32, 32])
        x = x.transpose([0, 2, 3, 1])
        x = x.reshape(-1, 32*32*3)

    return x, y


def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot


def change_num_classes(x, y, num_classes):
    idx = y < num_classes
    x = x[idx] 
    y = dense_to_one_hot(y[idx], num_classes)
    return x, y

def _print_download_progress(count, block_size, total_size):
    pct_complete = float(count * block_size) / total_size
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()

def maybe_download_and_extract():
    main_directory = "./data_set/"
    cifar_100_directory = main_directory+"cifar_100/"
    if not os.path.exists(main_directory):
        os.makedirs(main_directory)

        url = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        filename = url.split('/')[-1]
        file_path = os.path.join(main_directory, filename)
        zip_cifar_100 = file_path
        file_path, _ = urlretrieve(url=url, filename=file_path, reporthook=_print_download_progress)

        print()
        print("Download finished. Extracting files.")
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(main_directory)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(main_directory)
        print("Done.")

        os.rename(main_directory+"./cifar-100-python", cifar_100_directory)
        os.remove(zip_cifar_100)
