import os
import csv
import pickle
import gzip
import numpy as np
import tensorflow as tf
from itertools import imap
import scipy.misc as smp
from skimage.measure import block_reduce

def show_img_2D(img, max_row, max_col):
    data = np.zeros((max_row, max_col, 3), dtype=np.uint8)
    row = 0
    col = 0
    for r in img:
        for i in r:
            data[row, col] = [255 * i, 255 * i, 255 * i]
            col += 1
            if col == max_col:
                row += 1
                col = 0
    img = smp.toimage(data)  # Create a PIL image
    img.show()  # View in default viewer

def show_img_1D(img, max_row, max_col):
    img_data = np.zeros((max_row, max_col, 3), dtype=np.uint8)
    row = 0
    col = 0

    for i in img:
        img_data[row, col] = [255 * i, 255 * i, 255 * i]
        col += 1
        if col == max_col:
            row += 1
            col = 0
    img = smp.toimage(img_data)  # Create a PIL image
    img.show()  # View in default viewer

def load_img_to_nparray(filepath):
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        data = reader.next()
    return np.asarray(list(imap(float, data)))

class Dataset(object):
    position = ['top_left', 'top_right', 'bot_left', 'bot_right']

    training_img = []
    training_label = []

    testing_img = []
    testing_label = []

    def __init__(self, path, cache_path = 'cache'):
        if os.path.isdir(cache_path):
            imgs, labels = self._load_cache(cache_path)
        else:
            imgs, labels = self._load_data(path)
            self._dump_cache(cache_path, imgs=imgs, labels=labels)

        l = len(imgs)
        self.training_img = imgs[range(0, l, 2)]
        self.training_label = labels[range(0, l, 2)]
        self.testing_img = imgs[range(1, l, 2)]
        self.testing_label = labels[range(1, l, 2)]

    def _dump_cache(self, path, imgs, labels):
        print 'Caching to %s' % path
        if not os.path.isdir(path):
            os.mkdir(path)

        with gzip.GzipFile(os.path.join(path, 'images'), 'w') as zipf:
            pickle.dump(imgs, zipf)
        with gzip.GzipFile(os.path.join(path, 'labels'), 'w') as zipf:
            pickle.dump(labels, zipf)

    def _load_cache(self, path):
        print 'Loading cache from %s' % path
        with gzip.open(os.path.join(path, 'images'), 'rb') as zipf:
            imgs = pickle.load(zipf)
        with gzip.open(os.path.join(path, 'labels'), 'rb') as zipf:
            labels = pickle.load(zipf)
        return imgs, labels

    def _load_data(self, path):
        print 'Loading original data from %s' % path
        imgs = []
        labels = []
        for pos in self.position:
            for root, dirs, files in os.walk('%s/%s' % (path, pos), topdown=False):
                for name in files:
                    filepath = os.path.join(root, name)
                    if not filepath.endswith('.csv'):
                        continue
                    img = load_img_to_nparray(filepath)

                    img = img.reshape(200, 300)
                    img_reduce = block_reduce(img, block_size=(6, 6), func=np.mean)
                    img_reduce = img_reduce.flatten()

                    imgs.append(img_reduce)

                    if pos == 'top_left':
                        one_hot = [1.0, 0.0, 0.0, 0.0]
                    elif pos == 'top_right':
                        one_hot = [0.0, 1.0, 0.0, 0.0]
                    elif pos == 'bot_left':
                        one_hot = [0.0, 0.0, 1.0, 0.0]
                    elif pos == 'bot_right':
                        one_hot = [0.0, 0.0, 0.0, 1.0]
                    else:
                        assert False, 'WTF IS THIS POS? %s' % pos

                    labels.append(np.asarray(one_hot))

        imgs = np.array(imgs)
        labels = np.array(labels)
        return imgs, labels

    def training_batches(self, batch_size, count):
        i = 0
        l = len(self.training_img)
        for _ in range(count):
            img = self.training_img[i: i + batch_size]
            label = self.training_label[i: i + batch_size]
            i += batch_size
            if i > l:
                img = np.concatenate([img, self.training_img[0:(i - l)]])
                label = np.concatenate([label, self.training_label[0:(i - l)]])
                i -= l

            yield (img, label)

