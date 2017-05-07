import os
import csv
import pickle
import gzip
import numpy as np
import tensorflow as tf
from itertools import imap
import scipy.misc as smp
from skimage.measure import block_reduce

def show_img(img, max_row, max_col):
    data = np.zeros((max_row, max_col, 3), dtype=np.uint8)
    row = 0
    col = 0
    for r in img:
        if hasattr(r, '__iter__'):
            x = r
        else:
            x = [r]
        for i in x:
            data[row, col] = [255 * i, 255 * i, 255 * i]
            col += 1
            if col == max_col:
                row += 1
                col = 0
    img = smp.toimage(data)  # Create a PIL image
    img.show()  # View in default viewer

def save(path, obj):
    with gzip.GzipFile(path, 'w') as zipf:
        pickle.dump(obj, zipf)
 
def load(path):
    with gzip.open(path, 'rb') as zipf:
        return pickle.load(zipf)

def load_img_to_nparray(filepath):
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        data = reader.next()
    return np.asarray(list(imap(float, data)))

class Dataset(object):
    position = ['top_left', 'top_right', 'bot_left', 'bot_right']

    def __init__(self, path, cache_path = './cache'):
        if os.path.isfile(cache_path):
            print 'Loading cache from %s' % cache_path
            imgs, labels = load(cache_path)
        else:
            print 'Loading original data from %s' % path
            imgs, labels = self._load_data(path)
            save(cache_path, (imgs, labels))

        l = len(imgs)
        self.training_img = imgs[range(0, l, 2)]
        self.training_label = labels[range(0, l, 2)]
        self.testing_img = imgs[range(1, l, 2)]
        self.testing_label = labels[range(1, l, 2)]

        imgs = None
        labels = None

    def _load_data(self, path):
        imgs = []
        labels = []
        for pos in self.position:
            for root, dirs, files in os.walk('%s/%s' % (path, pos), topdown=False):
                for name in files:
                    if not name.endswith('.csv'):
                        continue
                    filepath = os.path.join(root, name)
                    img = load_img_to_nparray(filepath)

                    img = img.reshape(200, 300)
                    img = block_reduce(img, block_size=(6, 6), func=np.mean)
                    img = img.flatten()

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

                    imgs.append(img)
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

