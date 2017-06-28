import numpy as np
import tflearn
import tensorflow as tf
import os
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from utils import Dataset, draw_heatmap_with_test_data, test_accuracy, draw_heatmap_with_realtime, \
    draw_double_cross_heatmap

load_dataset = False
load_set2 = False
train_model = False

num_square = 12
model_dir = 'tensorflow_models'

raw_img_data_dir = 'image_data/delta_raw_img_16'

img_data_dir = 'image_data/double_cross_12_v'
img_data_dir_2 = 'image_data/delta_data_16'

if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
model_name = os.path.join(model_dir, '2B_%s_v.model' % num_square)

row = 67
col = 100
block_size = 3
n_classes = num_square
n_epoch = 50
learning_rate = 0.0001

X = []
Y = []
test_x_1 = []
test_x_2 = []
test_y_1 = []
test_y_2 = []

if load_dataset:
    dataset = Dataset(img_data_dir, num_square, block_size=block_size)

    if load_set2:
        dataset_2 = Dataset(img_data_dir_2, num_square, block_size=block_size)

    for epoch_x, epoch_y in dataset.training_batches(-1, -1, all=True):
        X += list(epoch_x)
        Y += list(epoch_y)

    if load_set2:
        for epoch_x, epoch_y in dataset_2.training_batches(-1, -1, all=True):
            X += list(epoch_x)
            Y += list(epoch_y)

    X = np.array(X)
    Y = np.array(Y)

    divisor = 10
    test_x = dataset.testing_img
    test_x_1 = [i for idx, i in enumerate(test_x) if int(idx/divisor) % 2 == 0]
    test_x_2 = [i for idx, i in enumerate(test_x) if int(idx/divisor) % 2 == 1]

    test_y = dataset.testing_label
    test_y_1 = [i for idx, i in enumerate(test_y) if int(idx/divisor) % 2 == 0]
    test_y_2 = [i for idx, i in enumerate(test_y) if int(idx / divisor) % 2 == 1]

    if load_set2:
        test_x = dataset_2.testing_img
        test_x_1 += [i for idx, i in enumerate(test_x) if int(idx/divisor) % 2 == 0]
        test_x_2 += [i for idx, i in enumerate(test_x) if int(idx/divisor) % 2 == 1]

        test_y = dataset_2.testing_label
        test_y_1 += [i for idx, i in enumerate(test_y) if int(idx/divisor) % 2 == 0]
        test_y_2 += [i for idx, i in enumerate(test_y) if int(idx/divisor) % 2 == 1]

    test_x_1 = np.array(test_x_1)
    test_x_2 = np.array(test_x_2)

    test_y_1 = np.array(test_y_1)
    test_y_2 = np.array(test_y_2)

    X = X.reshape([-1, row, col, 1])
    test_x_1 = test_x_1.reshape([-1, row, col, 1])
    test_x_2 = test_x_2.reshape([-1, row, col, 1])

# import pdb
# pdb.set_trace()

tf.reset_default_graph()
cnn = input_data(shape=[None, row, col, 1], name='input')

cnn = conv_2d(cnn, 64, 2, activation='relu')
cnn = max_pool_2d(cnn, 2)

cnn = conv_2d(cnn, 64, 2, activation='relu')
cnn = max_pool_2d(cnn, 2)

cnn = conv_2d(cnn, 64, 2, activation='relu')
cnn = max_pool_2d(cnn, 2)

cnn = conv_2d(cnn, 64, 2, activation='relu')
cnn = max_pool_2d(cnn, 2)

cnn = conv_2d(cnn, 64, 2, activation='relu')
cnn = max_pool_2d(cnn, 2)

cnn = conv_2d(cnn, 64, 2, activation='relu')
cnn = max_pool_2d(cnn, 2)

cnn = fully_connected(cnn, 500, activation='relu')
cnn = fully_connected(cnn, 500, activation='relu')
cnn = fully_connected(cnn, 500, activation='relu')
cnn = fully_connected(cnn, n_classes, activation='softmax')
cnn = regression(cnn, learning_rate=learning_rate, name='targets')

model_h = tflearn.DNN(cnn)
model_h.load('tensorflow_models/2B_12_h.model')


tf.reset_default_graph()
cnn2 = input_data(shape=[None, row, col, 1], name='input2')

cnn2 = conv_2d(cnn2, 64, 2, activation='relu')
cnn2 = max_pool_2d(cnn2, 2)

cnn2 = conv_2d(cnn2, 64, 2, activation='relu')
cnn2 = max_pool_2d(cnn2, 2)

cnn2 = conv_2d(cnn2, 64, 2, activation='relu')
cnn2 = max_pool_2d(cnn2, 2)

cnn2 = conv_2d(cnn2, 64, 2, activation='relu')
cnn2 = max_pool_2d(cnn2, 2)

cnn2 = conv_2d(cnn2, 64, 2, activation='relu')
cnn2 = max_pool_2d(cnn2, 2)

cnn2 = conv_2d(cnn2, 64, 2, activation='relu')
cnn2 = max_pool_2d(cnn2, 2)

cnn2 = fully_connected(cnn2, 500, activation='relu')
cnn2 = fully_connected(cnn2, 500, activation='relu')
cnn2 = fully_connected(cnn2, 500, activation='relu')
cnn2 = fully_connected(cnn2, n_classes, activation='softmax')
cnn2 = regression(cnn2, learning_rate=learning_rate, name='targets2')

model_v = tflearn.DNN(cnn2)
model_v.load('tensorflow_models/2B_12_v.model')

# if train_model:
#     model.fit({'input': X},
#               {'targets': Y},
#               n_epoch=n_epoch,
#               validation_set=({'input': test_x_1}, {'targets': test_y_1}),
#               snapshot_step=500,
#               show_metric=True,
#               run_id='2B')
#
#     model.save(model_name)
#     exit(0)
#
# model.load(model_name)


#test_accuracy(model, test_x_2, test_y_2, row, col, flatten=False)
#draw_heatmap_with_test_data(model, num_square, 100, raw_img_data_dir, skip=1)
#draw_heatmap_with_realtime(model, num_square, row, col, block_size)
draw_double_cross_heatmap(model_h, model_v, num_square, row, col, block_size)