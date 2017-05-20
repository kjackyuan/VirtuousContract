import numpy as np
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from utils import Dataset

dataset = Dataset('./data')

row = 34
col = 50
n_classes = 4
n_epoch = 200
learning_rate = 0.001

X = []
Y = []

for epoch_x, epoch_y in dataset.training_batches(100, 10):
    X += list(epoch_x)
    Y += list(epoch_y)

X = np.array(X)
Y = np.array(Y)

test_x = dataset.testing_img
test_y = dataset.testing_label

X = X.reshape([-1, row, col, 1])
test_x = test_x.reshape([-1, row, col, 1])


cnn = input_data(shape=[None, row, col, 1], name='input')

cnn = conv_2d(cnn, 64, 2, activation='sigmoid')
cnn = max_pool_2d(cnn, 2)

cnn = conv_2d(cnn, 64, 2, activation='sigmoid')
cnn = max_pool_2d(cnn, 2)

cnn = conv_2d(cnn, 64, 2, activation='sigmoid')
cnn = max_pool_2d(cnn, 2)

cnn = conv_2d(cnn, 64, 2, activation='sigmoid')
cnn = max_pool_2d(cnn, 2)

cnn = fully_connected(cnn, 500, activation='sigmoid')
cnn = fully_connected(cnn, n_classes, activation='sigmoid')

cnn = regression(cnn, learning_rate=learning_rate, name='targets')

model = tflearn.DNN(cnn)

model.fit({'input': X},
          {'targets': Y},
          n_epoch=n_epoch,
          validation_set=({'input': test_x}, {'targets': test_y}),
          snapshot_step=500,
          show_metric=True,
          run_id='2B')

model.save('touch_screen.model')

