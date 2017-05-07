import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from utils import Dataset

dataset = Dataset('data')

X = dataset.training_img
Y = dataset.training_label
test_x = dataset.testing_img
test_y = dataset.testing_label

row = 34
col = 50
n_classes = 4

n_epoch = 10

X = X.reshape([-1, row, col, 1])
test_x = test_x.reshape([-1, row, col, 1])

convnet = input_data(shape=[None, row, col, 1], name='input')

convnet = conv_2d(convnet, 32, 2, activation='sigmoid')
convnet = max_pool_2d(convnet, 2) # mostly just to reduce complutation cost, removing did help accuracy

convnet = conv_2d(convnet, 32, 2, activation='sigmoid')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='sigmoid')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='sigmoid')
#convnet = dropout(convnet, 0.8) # helps against overfitting, not needed at the moment

convnet = fully_connected(convnet, n_classes, activation='softmax')

convnet = regression(convnet,
                     optimizer='adam',
                     learning_rate=0.01,
                     loss='categorical_crossentropy', # also tried mean_square but no improvements
                     name='targets')

model = tflearn.DNN(convnet)

model.fit({'input': X},
          {'targets': Y},
          n_epoch=n_epoch,
          validation_set=({'input': test_x}, {'targets': test_y}),
          snapshot_step=500,
          show_metric=True,
          run_id='2B')

model.save('tflearn_cnn.model')

