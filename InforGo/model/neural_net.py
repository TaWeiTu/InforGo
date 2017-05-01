import tensorflow as tf
import numpy as np
import os

from InforGo.environment.global_var import *
from InforGo.util import logger


class NeuralNetwork(object):
    """
    Neural Network Setup
    input layer: state + current_player + pattern
    hidden layer: n_hidden_layer with nodes at each layer = n_node_hidden, activation function is activation_fn
    output layer: value of the input state for current player
    weight and bias are stored in directory
    learning rate = learning_rate
    """ 
    def __init__(self, player_len=1, pattern_len=6, n_hidden_layer=1, n_node_hidden=[32], activation_fn='tanh', learning_rate=0.001, directory='../Data/default/'):
        logger.info('[NeuralNetwork] Start Building Neural Network')
        self.input_state = tf.placeholder(shape=[4, 4, 4], dtype=tf.float64)
        self.state = tf.reshape(self.input_state, [1, 64])
        self.player_node = tf.placeholder(shape=[1, player_len], dtype=tf.float64)
        self.player_len = player_len
        self.pattern = tf.placeholder(shape=[1, pattern_len], dtype=tf.float64)
        self.activation_fn = self.get_fn(activation_fn)
        self.weight = [None for i in range(n_hidden_layer + 1)]
        self.bias = [None for i in range(n_hidden_layer + 1)]
        self.directory = directory
        self.hidden_layer = [{} for i in range(n_hidden_layer)]

        self.weight[0] = self.initialize_weight(64 + player_len + pattern_len, n_node_hidden[0], 0)
        self.bias[0] = self.initialize_bias(n_node_hidden[0], 0)

        for i in range(1, n_hidden_layer):
            self.weight[i] = self.initialize_weight(n_node_hidden[i - 1], n_node_hidden[i], i)
            self.bias[i] = self.initialize_bias(n_node_hidden[i], i)
        self.weight[n_hidden_layer] = self.initialize_weight(n_node_hidden[n_hidden_layer - 1], 1, n_hidden_layer)
        self.bias[n_hidden_layer] = self.initialize_bias(1, n_hidden_layer)

        self.hidden_layer[0]['output'] = tf.add(tf.matmul(tf.concat([self.state, self.player_node, self.pattern], 1), self.weight[0]), self.bias[0])

        for i in range(1, n_hidden_layer):
            self.hidden_layer[i - 1]['activate'] = self.activation_fn(self.hidden_layer[i - 1]['output'])
            self.hidden_layer[i]['output'] = tf.add(tf.matmul(self.hidden_layer[i - 1]['activate'], self.weight[i]), self.bias[i])
        self.hidden_layer[n_hidden_layer - 1]['activate'] = self.activation_fn(self.hidden_layer[n_hidden_layer - 1]['output'])

        self.v = tf.tanh(tf.add(tf.matmul(self.hidden_layer[n_hidden_layer - 1]['activate'], self.weight[n_hidden_layer]), self.bias[n_hidden_layer]))
        self.v_ = tf.placeholder(shape=[1, 1], dtype=tf.float64)
        self.error = tf.reduce_sum(tf.square(self.v_ - self.v))
        self.trainer = tf.train.GradientDescentOptimizer(learning_rate)
        self.opt_model = self.trainer.minimize(self.error)

        # self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        logger.info('[NeuralNetwork] Done Building Neural Network')

    def initialize_weight(self, n, m, _id):
        if os.path.exists(self.directory + 'weight{}'.format(_id)):
            with open(self.directory + 'weight{}'.format(_id), 'r') as f:
                w = np.zeros([n, m])
                for i in range(n):
                    for j in range(m): w[i, j] = float(f.readline())
                f.close()
                return tf.Variable(tf.cast(w, dtype=tf.float64))
        return tf.Variable(tf.truncated_normal(shape=[n, m], mean=0.0, stddev=0.01, dtype=tf.float64))

    def initialize_bias(self, n, _id):
        if os.path.exists(self.directory + 'bias{}'.format(_id)):
            with open(self.directory + 'bias{}'.format(_id), 'r') as f:
                b = np.zeros([1, n])
                for i in range(n): b[0, i] = float(f.readline())
                f.close()
                return tf.Variable(tf.cast(b, dtype=tf.float64))
        return tf.Variable(tf.truncated_normal(shape=[1, n], mean=0.0, stddev=0.01, dtype=tf.float64))

    def get_fn(self, activation_fn):
        if activation_fn == 'tanh': return lambda x: tf.tanh(x)
        if activation_fn == 'relu': return lambda x: tf.nn.relu(x)
        if activation_fn == 'sigmoid': return lambda x: tf.sigmoid(x)
        return lambda x: x

    def predict(self, state, player, pattern):
        player_node = np.reshape(np.array([player for i in range(self.player_len)]), [1, self.player_len])
        v = self.sess.run(self.v, feed_dict={self.input_state: state, self.player_node: player_node, self.pattern: pattern})
        return v[0, 0]

    def update(self, state, player, pattern, v_):
        player_node = np.reshape(np.array([player for i in range(self.player_len)]), [1, self.player_len])
        v_placeholder = np.reshape(np.array(v_), [1, 1])
        self.sess.run(self.opt_model, feed_dict={self.input_state: state, self.player_node: player_node, self.pattern: pattern, self.v_: v_placeholder})

    def store(self):
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        for _id in range(len(self.weight)):
            with open(self.directory + 'weight{}'.format(_id), 'w+') as f:
                w = self.sess.run(self.weight[_id])
                shape = self.sess.run(tf.shape(self.weight[_id]))
                for i in range(shape[0]):
                    for j in range(shape[1]): f.write('{}\n'.format(w[i, j]))
                f.close()
            with open(self.directory + 'bias{}'.format(_id), 'w+') as f:
                b = self.sess.run(self.bias[_id])
                shape = self.sess.run(tf.shape(self.bias[_id]))
                for i in range(shape[1]): f.write('{}\n'.format(b[0, i]))
                f.close()

