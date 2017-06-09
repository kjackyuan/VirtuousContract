import os
from multiprocessing import Pool

import cv2
import csv
import pickle
import gzip
import functools
import pygame
import time
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
    # with gzip.GzipFile(path, 'w') as zipf:
    #     pickle.dump(obj, zipf)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load(path):
    # with gzip.open(path, 'rb') as zipf:
    #     return pickle.load(zipf)
    with open(path, 'r') as f:
        return pickle.load(f)


def load_img_to_nparray(filepath):
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        data = reader.next()
    return np.asarray(list(imap(float, data)))


def _load_data(args):
    pos, path, block_size, row, col, num_square = args
    print pos

    imgs = []
    labels = []
    for root, dirs, files in os.walk('%s/%s' % (path, pos), topdown=False):
        for name in files:
            filepath = os.path.join(root, name)

            if name.endswith('.csv'):
                img = load_img_to_nparray(filepath)
                img = img.reshape(row, col)
            elif name.endswith('.png'):
                img = cv2.imread(filepath, 0)
            else:
                continue

            img = block_reduce(img, block_size=(block_size, block_size), func=np.mean)

            img = img.flatten()

            label = np.zeros(num_square)
            label[pos] = 1.0

            imgs.append(img)
            labels.append(label)
    return imgs, labels


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

        pool = Pool(processes=4)  # start 4 worker processes
        inputs = []
        for pos in range(self.num_square):
            inputs.append((pos, path, block_size, self.row, self.col, self.num_square))

        results = pool.map(_load_data, inputs)

        imgs, labels = zip(*results)
        imgs = functools.reduce(lambda a,b: a+b, imgs)
        labels = functools.reduce(lambda a,b: a+b, labels)
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


def sobel_img(img):
    edges_x = cv2.Sobel(img, -1, 1, 0, 20)
    edges_y = cv2.Sobel(img, -1, 0, 1, 20)
    edges = edges_x + edges_y
    return edges


def test_accuracy(model, testing_img, testing_label, row, col, flatten=True):

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
        if flatten:
            img = img.reshape([-1, row*col])
        else:
            img = img.reshape([-1, row, col, 1])

        predict_label = np.array(model.predict(img)[0])

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


def draw_heatmap_with_test_data(model, num_square, num_img_per_square, img_data_dir,
                                row, col, block_size,
                                skip=1, quiet=False):
    board_size = 400
    full_row = 200
    full_col = 300

    normalize = lambda x: x / 255.0
    normalize = np.vectorize(normalize)

    if not quiet:
        pygame.init()
        screen = pygame.display.set_mode((board_size, board_size))
        done = False

    for pos in range(0, num_square):
        for img_id in range(0, num_img_per_square, skip):
            filepath = os.path.join(img_data_dir, '%s/%s.png' % (pos, img_id))
            img = cv2.imread(filepath)
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if not quiet:
                cv2.imshow('img_gray', img)

            # Sobel
            # img = sobel_img(img)
            # cv2.imshow('sobel', img)

            img = img.reshape(full_row * full_col)
            img = normalize(img)
            img = img.reshape(full_row, full_col)

            img = block_reduce(img, block_size=(block_size, block_size), func=np.mean)
            img = img.flatten()

            predict_label = np.array(model.predict(img.reshape([-1, row * col]))[0])

            max_pos = np.argmax(predict_label)
            max_value = predict_label[max_pos]
            predict_label = predict_label / max_value
            print 'Predicted position: %s. Actual position: %s' % (max_pos, pos)

            if not quiet:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True

                width = int(board_size/np.sqrt(num_square))
                square_width = int(np.sqrt(num_square))

                predict_label = np.resize(predict_label, (square_width, square_width))

                for i, label_row in enumerate(predict_label):
                    for j, val in enumerate(label_row):
                        if val == 1.0:
                            pygame.draw.rect(screen,
                                             (255, 0, 255),
                                             pygame.Rect(width * j, width * i, width * (j + 1),
                                                         width * (i + 1)))
                        else:
                            pygame.draw.rect(screen,
                                             (255 * val, 0, 255 * (1.0 - val)),
                                             pygame.Rect(width * j, width * i, width * (j + 1),
                                                         width * (i + 1)))

                pygame.display.flip()
            k = cv2.waitKey(3)


def draw_heatmap_with_realtime(model, num_square, row, col, block_size):
    board_size = 400
    full_row = 200
    full_col = 300
    cap = cv2.VideoCapture(0)

    normalize = lambda x: x / 255.0
    normalize = np.vectorize(normalize)

    pygame.init()
    screen = pygame.display.set_mode((board_size, board_size))
    done = False

    # set up phase, press Q to continue
    while True:
        ret, img = cap.read()
        img = cv2.resize(img, (300, 200))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('img_gray', img)

        k = cv2.waitKey(30)
        if k == ord('q'):
            time.sleep(3)
            break

    while True:
        ret, img = cap.read()
        img = cv2.resize(img, (300, 200))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('img_gray', img)

        # Sobel
        # img = sobel_img(img)
        # cv2.imshow('sobel', img)

        img = img.reshape(full_row * full_col)
        img = normalize(img)
        img = img.reshape(full_row, full_col)

        img = block_reduce(img, block_size=(block_size, block_size), func=np.mean)
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