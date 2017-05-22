import os
import cv2
import csv
import pickle
import gzip
import pygame
import numpy as np
import scipy.misc as smp

from itertools import imap
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


def test_accuracy(model, testing_img, testing_label, row, col):

    total_img = len(testing_img)
    correct_img = 0
    intermediate_total = 0

    filter = lambda x: 0 if x < 1.0 else 1.0
    filter = np.vectorize(filter)

    for id, img in enumerate(testing_img):
        if not id % 500:
            print '********************************************'
            print '%s/%s' % (id, total_img)
            print 'correct:%s, wrong:%s, total:%s, percentage:%s' % \
                  (correct_img, intermediate_total - correct_img,
                   intermediate_total, (correct_img / float(intermediate_total+1e-6)))
            print '***********'

        label = testing_label[id]
        predict_label = np.array(model.predict(img.reshape([-1, row*col]))[0])

        max_pos = -1
        for pos, i in enumerate(label):
            if i == 1.0:
                max_pos = pos
        assert max_pos > -1

        max_value = predict_label[max_pos]

        predict_label = predict_label/max_value
        correct_prediction = all(label == filter(predict_label))

        if correct_prediction:
            correct_img += 1

        intermediate_total += 1

    print 'Final:'
    print 'correct:%s, wrong:%s, total:%s, percentage:%s' % \
          (correct_img, total_img-correct_img, total_img, (correct_img/float(total_img)))


def draw_heatmap(model, num_square, num_img_per_square, img_data_dir):
    board_size = 400
    full_row = 200
    full_col = 300
    row = 34
    col = 50

    normalize = lambda x: x / 255.0
    normalize = np.vectorize(normalize)

    pygame.init()
    screen = pygame.display.set_mode((board_size, board_size))
    done = False

    for pos in range(0, num_square):
        for img_id in range(0, num_img_per_square):
            filepath = os.path.join(img_data_dir, '%s/%s.png' % (pos, img_id))
            img = cv2.imread(filepath)
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imshow('img_gray', img)

            img = img.reshape(full_row * full_col)
            img = normalize(img)

            img = img.reshape(full_row, full_col)
            img = block_reduce(img, block_size=(6, 6), func=np.mean)
            img = img.flatten()

            predict_label = np.array(model.predict(img.reshape([-1, row * col]))[0])

            max_pos = np.argmax(predict_label)
            max_value = predict_label[max_pos]
            predict_label = predict_label / max_value
            print 'Predicted position %s' % max_pos

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            width = int(board_size/np.sqrt(num_square))
            square_width = int(np.sqrt(num_square))

            predict_label = np.resize(predict_label, (square_width, square_width))
            for i, label_row in enumerate(predict_label):
                for j, val in enumerate(label_row):
                    pygame.draw.rect(screen,
                                     (255 * val, 0, 255 * (1.0 - val)),
                                     pygame.Rect(width * j, width * i, width * (j + 1),
                                                 width * (i + 1)))

            k = cv2.waitKey(30)
            pygame.display.flip()