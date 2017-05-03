import tensorflow as tf
from utils import Dataset

# handwriting number recognition

dataset = Dataset()
dataset.prep_data()


# TODO: how to tweak this
n_nodes_hl1 = 1500
n_nodes_hl2 = 1500
n_nodes_hl3 = 1500
n_nodes_hl4 = 1500
n_nodes_hl5 = 1500
n_nodes_hl6 = 1500

n_classes = 4
batch_size = 100 # how many samples you load at a time for training

# height x width
xdim = dataset.testing_cache[0].shape[0]
x = tf.placeholder('float', [None, xdim]) # image is flattened 34 x 50
y = tf.placeholder('float')

def neural_network_model(data):
    hidden_1_layer = {
            'weights': tf.Variable(tf.random_normal([xdim, n_nodes_hl1])),
            'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))
    }

    hidden_2_layer = {
            'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
            'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))
    }

    hidden_3_layer = {
            'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
            'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))
    }

    hidden_4_layer = {
            'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])),
            'biases': tf.Variable(tf.random_normal([n_nodes_hl4]))
    }

    hidden_5_layer = {
            'weights': tf.Variable(tf.random_normal([n_nodes_hl4, n_nodes_hl5])),
            'biases': tf.Variable(tf.random_normal([n_nodes_hl5]))
    }

    hidden_6_layer = {
            'weights': tf.Variable(tf.random_normal([n_nodes_hl5, n_nodes_hl6])),
            'biases': tf.Variable(tf.random_normal([n_nodes_hl6]))
    }

    output_layer = {
            'weights': tf.Variable(tf.random_normal([n_nodes_hl6, n_classes])),
            'biases': tf.Variable(tf.random_normal([n_classes]))
    }

    l1 = tf.matmul(data, hidden_1_layer['weights']) + hidden_1_layer['biases']
    l1 = tf.sigmoid(l1) # rectified linear

    l2 = tf.matmul(l1, hidden_2_layer['weights']) + hidden_2_layer['biases']
    l2 = tf.sigmoid(l2)

    l3 = tf.matmul(l2, hidden_3_layer['weights']) + hidden_3_layer['biases']
    l3 = tf.sigmoid(l3)

    l4 = tf.matmul(l3, hidden_4_layer['weights']) + hidden_4_layer['biases']
    l4 = tf.sigmoid(l4)

    l5 = tf.matmul(l4, hidden_5_layer['weights']) + hidden_5_layer['biases']
    l5 = tf.sigmoid(l5)

    l6 = tf.matmul(l5, hidden_6_layer['weights']) + hidden_6_layer['biases']
    l6 = tf.sigmoid(l6)

    output_layer = tf.matmul(l6, output_layer['weights']) + output_layer['biases']

    return output_layer


def train_neural_network(x, y):
    prediction = neural_network_model(x)
    #cost = tf.reduce_mean(tf.square(tf.nn.softmax(prediction) - y))

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    # learning_rate = 0.001 default
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # cycles of feed forward + back prop
    hm_epochs = 10

    with tf.Session() as s:
        s.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            print 'starting: %s' % epoch

            epoch_loss = 0
            for epoch_x, epoch_y in dataset.training_batches(batch_size, 10):
                _, c = s.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print 'Epoch: ', epoch, ' completed out of ', hm_epochs, ' loss: ', epoch_loss
            print 'Accuracy: ', accuracy.eval({x: dataset.testing_cache, y: dataset.testing_label})

train_neural_network(x, y)
