import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

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

hm_epochs = 2
n_classes = 10
batch_size = 128

chunk_size = 28
n_chunks = 28
rnn_size = 16

# height x width
x = tf.placeholder('float', [None, n_chunks, chunk_size]) # image is flattened 28 x 28
y = tf.placeholder('float')

def recurrent_neural_network(x):
    layer = {
            'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
            'biases': tf.Variable(tf.random_normal([n_classes]))
    }

    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output_layer = tf.add(tf.matmul(outputs[-1], layer['weights']), layer['biases'])

    return output_layer


def train_neural_network(x, y):
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    # learning_rate = 0.001 default
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as s:
        s.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            print "epoch %s" % epoch

            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))


                _, c = s.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print 'Epoch: ', epoch, ' completed out of ', hm_epochs, ' loss: ', epoch_loss

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print 'Accuracy: ', accuracy.eval({x: mnist.test.images.reshape((-1, n_chunks,
                                                                         chunk_size)),
                                           y: mnist.test.labels})

train_neural_network(x, y)