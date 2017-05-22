import os
import csv
import pickle
import gzip
import numpy as np
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

    def __init__(self, path, num_square, row=200, col=300, block_size=6, cache_path=None):
        self.num_square = num_square
        self.row = row
        self.col = col

        if not cache_path:
            cache_path = path + '_cache'
        if os.path.isfile(cache_path):
            print 'Loading cache from %s' % cache_path
            imgs, labels = load(cache_path)
        else:
            print 'Loading original data from %s' % path
            imgs, labels = self._load_data(path, block_size)
            save(cache_path, (imgs, labels))

        l = len(imgs)
        self.training_img = imgs[range(0, l, 2)]
        self.training_label = labels[range(0, l, 2)]
        self.testing_img = imgs[range(1, l, 2)]
        self.testing_label = labels[range(1, l, 2)]

    def _load_data(self, path, block_size):
        imgs = []
        labels = []
        for pos in range(0, self.num_square):
            print pos
            for root, dirs, files in os.walk('%s/%s' % (path, pos), topdown=False):
                for name in files:
                    if not name.endswith('.csv'):
                        continue
                    filepath = os.path.join(root, name)
                    img = load_img_to_nparray(filepath)

                    img = img.reshape(self.row, self.col)
                    img = block_reduce(img, block_size=(block_size, block_size), func=np.mean)
                    img = img.flatten()

                    label = np.zeros(self.num_square)
                    label[pos] = 1.0

                    imgs.append(img)
                    labels.append(label)

        imgs = np.array(imgs)
        labels = np.array(labels)
        return imgs, labels

    def training_batches(self, batch_size, count, all=False):
        if all:
            yield (self.training_img, self.training_label)

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
