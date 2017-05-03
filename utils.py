import os
import csv
import pickle
import gzip
import numpy as np
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
    cache_dir = 'dataset_cache'

    training_cache = []
    training_label = []

    testing_cache = []
    testing_label = []

    def cache_data(self):
        print 'Cacheing to %s' % self.cache_dir
        if not os.path.isdir(self.cache_dir):
            os.mkdir(self.cache_dir)

        with gzip.GzipFile(os.path.join(self.cache_dir, 'training_cache'), 'w') as zipf:
            pickle.dump(self.training_cache, zipf)
        with gzip.GzipFile(os.path.join(self.cache_dir, 'training_label'), 'w') as zipf:
            pickle.dump(self.training_label, zipf)
        with gzip.GzipFile(os.path.join(self.cache_dir, 'testing_cache'), 'w') as zipf:
            pickle.dump(self.testing_cache, zipf)
        with gzip.GzipFile(os.path.join(self.cache_dir, 'testing_label'), 'w') as zipf:
            pickle.dump(self.testing_label, zipf)


    def load_cache(self):
        print 'Loading cache from %s' % self.cache_dir
        with gzip.open(os.path.join(self.cache_dir, 'training_cache'), 'rb') as zipf:
            self.training_cache = pickle.load(zipf)
        with gzip.open(os.path.join(self.cache_dir, 'training_label'), 'rb') as zipf:
            self.training_label = pickle.load(zipf)
        with gzip.open(os.path.join(self.cache_dir, 'testing_cache'), 'rb') as zipf:
            self.testing_cache = pickle.load(zipf)
        with gzip.open(os.path.join(self.cache_dir, 'testing_label'), 'rb') as zipf:
            self.testing_label = pickle.load(zipf)


    def prep_data(self):
        if os.path.isdir(self.cache_dir):
            self.load_cache()
            return

        for _ in ['training', 'testing']:
            print _
            dir = None
            target_cache = None

            if _ == 'training':
                target_cache = self.training_cache
                target_lable = self.training_label
                dir = 'training_data'
            elif _ == 'testing':
                target_cache = self.testing_cache
                target_lable = self.testing_label
                dir = 'testing_data'

            for pos in self.position:
                print pos
                for root, dirs, files in os.walk('%s/%s' % (dir, pos), topdown=False):
                    for name in files:
                        filepath = os.path.join(root, name)
                        if not filepath.endswith('.csv'):
                            continue
                        img = load_img_to_nparray(filepath)

                        img = img.reshape(200, 300)
                        img_reduce = block_reduce(img, block_size=(6, 6), func=np.mean)
                        img_reduce = img_reduce.flatten()
                        target_cache.append(img_reduce)

                        if pos == 'top_left':
                            target_lable.append(np.asarray([1.0, 0.0, 0.0, 0.0]))
                        elif pos == 'top_right':
                            target_lable.append(np.asarray([0.0, 1.0, 0.0, 0.0]))
                        elif pos == 'bot_left':
                            target_lable.append(np.asarray([0.0, 0.0, 1.0, 0.0]))
                        elif pos == 'bot_right':
                            target_lable.append(np.asarray([0.0, 0.0, 0.0, 1.0]))
                        else:
                            assert False, 'WTF IS THIS POS? %s' % pos

        self.training_cache = np.array(self.training_cache)
        self.training_label = np.array(self.training_label)
        self.testing_cache = np.array(self.testing_cache)
        self.testing_label = np.array(self.testing_label)

        self.cache_data()


    def training_batches(self, batch_size, count):
        i = 0
        l = len(self.training_cache)
        for _ in range(count):
            img = self.training_cache[i: i + batch_size]
            label = self.training_label[i: i + batch_size]
            if i + batch_size > l:
                img = np.concatenate([img, self.training_cache[0: (i + batch_size - l)]])
                label = np.concatenate([label, self.training_label[0: (i + batch_size - l)]])
                i = i + batch_size - l
            else:
                i += batch_size

            yield (img, label)

if __name__=='__main__':
    a = Dataset()
    a.prep_data()
    print a.testing_label[123]
    show_img_1D(a.testing_cache[123], 34, 50)
