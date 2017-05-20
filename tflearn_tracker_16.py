import numpy as np
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
from utils import Dataset_16

dataset = Dataset_16('./data_16')

row = 34
col = 50
n_classes = 16
n_epoch = 100
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
#
# model.save('touch_screen.model_16')

model.load('touch_screen.model_16')

# maxsize = len(dataset.training_img)
# while True:
    # id = int(raw_input("img id: "))
    # if id >= maxsize:
    #     print 'id out of scope'
    #     continue
    # result = model.predict(dataset.testing_img[id].reshape([-1, row*col]))
    # print result
    # print dataset.testing_label[id]

total_img = len(dataset.testing_label)
correct_img = 0

filter = lambda x: 0 if x < 1.0 else 1.0
filter = np.vectorize(filter)

# for id, img in enumerate(dataset.testing_img):
#     print '%s/%s' % (id, total_img)
#
#     label = dataset.testing_label[id]
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
# print 'correct:%s, wrong:%s, total:%s, percentage:%s' % \
#       (correct_img, total_img-correct_img, total_img, (correct_img/float(total_img)))


import pygame

pygame.init()
screen = pygame.display.set_mode((400, 400))
clock = pygame.time.Clock()
done = False

for id, img in enumerate(dataset.testing_img):
    print '%s/%s' % (id, total_img)

    label = dataset.testing_label[id]
    predict_label = np.array(model.predict(img.reshape([-1, row*col]))[0])

    max_pos = -1
    for pos, i in enumerate(label):
        if i == 1.0:
            max_pos = pos
    assert max_pos > -1

    max_value = predict_label[max_pos]

    predict_label = predict_label/max_value

    ########
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    width = 100
    print predict_label
    predict_label = np.resize(predict_label, (4,4))
    for i, label_row in enumerate(predict_label):
        for j, val in enumerate(label_row):
            pygame.draw.rect(screen,
                             (255*val, 0, 255*(1.0-val)),
                             pygame.Rect(width*i, width*j, width*(i+1), width*(j+1)))

    clock.tick(600)
    pygame.display.flip()