import numpy as np
import tflearn
import tensorflow as tf
import os
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from utils import Dataset, draw_heatmap_with_test_data, test_accuracy, draw_heatmap_with_realtime, \
    draw_double_cross_heatmap


num_square = 12
row = 67
col = 100
block_size = 3
n_classes = num_square
n_epoch = 50
learning_rate = 0.0001



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

