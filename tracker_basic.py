import tensorflow as tf
from utils import Dataset

dataset = Dataset('./data')

n_classes = 4
batch_size = 100 # how many samples you load at a time for training

# height x width
xdim = dataset.testing_img[0].shape[0]
x = tf.placeholder('float', [None, xdim]) # image is flattened 34 x 50
y = tf.placeholder('float')

def neural_network_model(data):
    dims = [xdim, 1500, 1500, 1500, 1500, 1500, 1500, n_classes]
    output = data

    for i in range(len(dims)-1):
        weights = tf.Variable(tf.random_normal([dims[i], dims[i+1]]))
        biases = tf.Variable(tf.random_normal([dims[i + 1]]))
        output = tf.matmul(output, weights) + biases
        if i != len(dims)-2:
            output = tf.sigmoid(output)

    return output


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
            epoch_loss = 0
            for epoch_x, epoch_y in dataset.training_batches(batch_size, 10):
                _, c, acc = s.run([optimizer, cost, accuracy], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print 'Epoch', epoch, ', accuracy:', acc, ', loss:', epoch_loss
            print 'Accuracy: ', accuracy.eval({x: dataset.testing_img, y: dataset.testing_label})

train_neural_network(x, y)
