import numpy as np
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
from utils import Dataset_64, load_img_to_nparray

dataset = Dataset_64('./data_64')

row = 34
col = 50
n_classes = 64
n_epoch = 200
learning_rate = 0.0001

X = []
Y = []

for epoch_x, epoch_y in dataset.training_batches(-1, -1, all=True):
    X += list(epoch_x)
    Y += list(epoch_y)

X = np.array(X)
Y = np.array(Y)

test_x = dataset.testing_img
test_y = dataset.testing_label


fnn = input_data(shape=[None, row * col], name='input')

fnn = fully_connected(fnn, 500, activation='sigmoid')
fnn = fully_connected(fnn, 500, activation='sigmoid')
fnn = fully_connected(fnn, 500, activation='sigmoid')
fnn = fully_connected(fnn, 500, activation='sigmoid')

fnn = fully_connected(fnn, n_classes, activation='sigmoid')

fnn = regression(fnn, learning_rate=learning_rate, name='targets')

model = tflearn.DNN(fnn)

# model.fit({'input': X},
#           {'targets': Y},
#           n_epoch=n_epoch,
#           validation_set=({'input': test_x}, {'targets': test_y}),
#           snapshot_step=500,
#           show_metric=True,
#           run_id='2B')
# model.save('touch_screen.model')

model.load('touch_screen.model')

# maxsize = len(dataset.training_img)
# while True:
#     id = int(raw_input("img id: "))
#     if id >= maxsize:
#         print 'id out of scope'
#         continue
#     result = model.predict(dataset.testing_img[id])
#     print result
#     print dataset.testing_label[id]
#################################################
# total_img = len(dataset.real_testing_label)
# correct_img = 0
# intermediate_total = 0
#
# filter = lambda x: 0 if x < 1.0 else 1.0
# filter = np.vectorize(filter)
#
# for id, img in enumerate(dataset.real_testing_img):
#     if not id % 500:
#         print '********************************************'
#         print '%s/%s' % (id, total_img)
#         print 'correct:%s, wrong:%s, total:%s, percentage:%s' % \
#               (correct_img, intermediate_total - correct_img,
#                intermediate_total, (correct_img / float(intermediate_total+1e-6)))
#         print '***********'
#
#     label = dataset.real_testing_label[id]
#     predict_label = np.array(model.predict(img.reshape([-1, row*col]))[0])
#
#     max_pos = -1
#     for pos, i in enumerate(label):
#         if i == 1.0:
#             max_pos = pos
#     assert max_pos > -1
#
#     max_value = predict_label[max_pos]
#
#     predict_label = predict_label/max_value
#     correct_prediction = all(label == filter(predict_label))
#
#     if correct_prediction:
#         correct_img += 1
#
#     intermediate_total += 1
#
# print 'Final:'
# print 'correct:%s, wrong:%s, total:%s, percentage:%s' % \
#       (correct_img, total_img-correct_img, total_img, (correct_img/float(total_img)))

import cv2
import os
import time

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    img = cv2.resize(img, (300, 200))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('img_gray', gray)
    k = cv2.waitKey(30)
    if k == ord('q'):
        time.sleep(3)
        break


import pygame
from skimage.measure import block_reduce

normalize = lambda x: x/255.0
normalize = np.vectorize(normalize)

pygame.init()
screen = pygame.display.set_mode((400, 400))
done = False

while True:
# for pos in range(0, 16):
#     for img_id in range(1, 502):

    ret, img = cap.read()
    #filepath = '/Users/jackyuan/TensorPractice/training_data_64/%s/%s.png' % (pos, img_id)
    #img = cv2.imread(filepath)

    img = cv2.resize(img, (300, 200))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    filepath = 'temp_img.png'
    try:
        os.remove(filepath)
    except:
        pass
    cv2.imwrite(filepath, img)

    img = cv2.imread(filepath, 0)
    cv2.imshow('img_gray', img)

    img = img.reshape(200 * 300)
    img = normalize(img)

    filepath = 'temp_csv.csv'
    try:
        os.remove(filepath)
    except:
        pass
    img.tofile(filepath, sep=',', format='%10.10f')

    img = load_img_to_nparray(filepath)
    img = img.reshape(200, 300)
    img = block_reduce(img, block_size=(6, 6), func=np.mean)
    img = img.flatten()

    predict_label = np.array(model.predict(img.reshape([-1, row*col]))[0])

    max_pos = np.argmax(predict_label)
    max_value = predict_label[max_pos]
    predict_label = predict_label/max_value
    print (-1, max_pos)
    ########
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    width = 50
    square_width = 8

    predict_label = np.resize(predict_label, (square_width, square_width))
    for i, label_row in enumerate(predict_label):
        for j, val in enumerate(label_row):
            pygame.draw.rect(screen,
                             (255*val, 0, 255*(1.0-val)),
                             pygame.Rect(width*j, width*i, width*(j+1), width*(i+1)))

    k = cv2.waitKey(30)
    pygame.display.flip()