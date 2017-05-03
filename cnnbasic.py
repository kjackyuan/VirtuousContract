import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# handwriting number recognition

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# 10 classes, 0-9

'''
definition of one hot:
insetad of 0 = 0, 1 = 1, 2 = 2
0 = [1,0,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0,0]
...
'''

# TODO: how to tweak this
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100 # how many samples you load at a time for training

# height x width
x = tf.placeholder('float', [None, 784]) # image is flattened 28 x 28
y = tf.placeholder('float')


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def convolutional_neural_network(x):
    weights = {
            'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 3])),
            'W_conv2': tf.Variable(tf.random_normal([5, 5, 3, 6])),
            'W_fc': tf.Variable(tf.random_normal([7*7*6, 10])),
            'out': tf.Variable(tf.random_normal([10, n_classes]))
    }

    biases = {
            'W_conv1': tf.Variable(tf.random_normal([3])),
            'W_conv2': tf.Variable(tf.random_normal([6])),
            'W_fc': tf.Variable(tf.random_normal([10])),
            'out': tf.Variable(tf.random_normal([n_classes]))
    }

    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    conv1 = conv2d(x, weights['W_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = conv2d(conv1, weights['W_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, [-1, 7*7*6])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['W_fc'])

    output = tf.matmul(fc, weights['out']) + biases['out']

    return output


def train_neural_network(x, y):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    # learning_rate = 0.001 default
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # cycles of feed forward + back prop
    hm_epochs = 10

    with tf.Session() as s:
        s.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            print 'starting epoch: %s' % epoch
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)

                _, c = s.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print 'Epoch: ', epoch, ' completed out of ', hm_epochs, ' loss: ', epoch_loss

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print 'Accuracy: ', accuracy.eval({x: mnist.test.images, y: mnist.test.labels})

train_neural_network(x, y)