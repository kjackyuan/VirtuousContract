import os
import numpy as np
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
from utils import Dataset, test_accuracy, draw_heatmap

num_square = 16
model_dir = 'tensorflow_models'
img_data_dir = 'image_data'

model_name = os.path.join(model_dir, '2B_%s.model' % num_square)
dataset = Dataset(os.path.join(img_data_dir, 'data_%s' % num_square), num_square)

row = 34
col = 50
n_classes = num_square
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
# model.save(model_name)

model.load(model_name)
#test_accuracy(model, dataset.testing_img, dataset.testing_label, row, col)
draw_heatmap(model, num_square, 502, os.path.join(img_data_dir, 'raw_img_%s' % num_square))

