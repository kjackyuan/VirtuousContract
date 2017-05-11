import numpy as np
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
from utils import Dataset

dataset = Dataset('./data')

row = 34
col = 50
n_classes = 4
n_epoch = 200
learning_rate = 0.0001

X = []
Y = []

for epoch_x, epoch_y in dataset.training_batches(100, 10):
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

model.fit({'input': X},
          {'targets': Y},
          n_epoch=n_epoch,
          validation_set=({'input': test_x}, {'targets': test_y}),
          snapshot_step=500,
          show_metric=True,
          run_id='2B')

model.save('touch_screen.model')

