"""Basic Neural Network setup"""
import os
import tensorflow as tf
import numpy as np

from InforGo.environment.global_var import *
from InforGo.util import logger, get_pattern, plot_state
from InforGo.environment.bingo import Bingo as State


class NeuralNetwork(object):
    """
    Neural Network Setup
    input layer: state + current_player + pattern
    hidden layer: n_hidden_layer with nodes at each layer = n_node_hidden, activation function is activation_fn
    output layer: value of the input state for current player
    weight and bias are stored in directory
    learning rate = learning_rate
    """
    def __init__(self, player_len=1, pattern_len=8, n_hidden_layer=1, n_node_hidden=[32],
                 activation_fn='tanh', learning_rate=0.001, directory='../Data/default/'):
        logger.info('[NeuralNetwork] Start Building Neural Network')
        self.input_state = tf.placeholder(shape=[None, 4, 4, 4], dtype=tf.float64)
        self.state = tf.reshape(self.input_state, [tf.shape(self.input_state)[0], 64])
        self.player_node = tf.placeholder(shape=[None, player_len], dtype=tf.float64)
        self.player_len = player_len
        self.pattern = tf.placeholder(shape=[None, pattern_len], dtype=tf.float64)
        self.activation_fn = self.get_fn(activation_fn)
        # Initialize weights and biases
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
        # Apply tanh to the output layer, which maps R to [-1, 1]
        self.v = tf.tanh(tf.add(tf.matmul(self.hidden_layer[n_hidden_layer - 1]['activate'], self.weight[n_hidden_layer]), self.bias[n_hidden_layer]))
        self.v_ = tf.placeholder(shape=[None, 1], dtype=tf.float64)
        # square difference of the prediction and label
        self.error = tf.reduce_sum(tf.square(self.v - self.v_))
        self.trainer = tf.train.AdagradOptimizer(learning_rate)
        self.opt_model = self.trainer.minimize(self.error)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        logger.info('[NeuralNetwork] Done Building Neural Network')

    def initialize_weight(self, n, m, _id):
        if os.path.exists(self.directory + 'weight{}'.format(_id)):
            with open(self.directory + 'weight{}'.format(_id), 'r') as f:
                weight = np.zeros([n, m])
                for i in range(n):
                    for j in range(m): 
                        try: weight[i, j] = float(f.readline())
                        except: 
                            f.close()
                            return tf.Variable(tf.truncated_normal(shape=[n, m], mean=0.0, stddev=0.01, dtype=tf.float64))
                f.close()
                return tf.Variable(tf.cast(weight, dtype=tf.float64))
        return tf.Variable(tf.truncated_normal(shape=[n, m], mean=0.0, stddev=0.01, dtype=tf.float64))

    def initialize_bias(self, n, _id):
        if os.path.exists(self.directory + 'bias{}'.format(_id)):
            with open(self.directory + 'bias{}'.format(_id), 'r') as f:
                bias = np.zeros([1, n])
                for i in range(n): 
                    try: bias[0, i] = float(f.readline())
                    except:
                        f.close()
                        return tf.Variable(tf.truncated_normal(shape=[1, n], mean=0.0, stddev=0.01, dtype=tf.float64))
                f.close()
                return tf.Variable(tf.cast(bias, dtype=tf.float64))
        return tf.Variable(tf.truncated_normal(shape=[1, n], mean=0.0, stddev=0.01, dtype=tf.float64))

    def get_fn(self, activation_fn=''):
        """return tensorflow function"""
        if activation_fn == 'tanh': return lambda x: tf.tanh(x)
        if activation_fn == 'relu': return lambda x: tf.nn.relu(x)
        if activation_fn == 'sigmoid': return lambda x: tf.sigmoid(x)
        return lambda x: x

    def predict(self, states, players):
        """return the output value of state for player"""
        pattern = [get_pattern(state, player) for state, player in zip(states, players)]
        player_node = np.reshape(players, [len(players), 1])
        states = np.reshape(states, [len(states), 4, 4, 4])
        pattern = np.reshape(pattern, [len(pattern), 8])
        value = self.sess.run(self.v, feed_dict={self.input_state: states,
                              self.player_node: player_node, self.pattern: pattern})
        return value[:, 0]

    def update(self, states, players, v_):
        """update the value of state to v_"""
        pattern = [get_pattern(state, player) for state, player in zip(states, players)]
        pattern = np.reshape(pattern, [len(pattern), 8])
        player_node = np.reshape(players, [len(players), 1])
        v_placeholder = np.reshape(v_, [len(v_), 1])
        states = np.reshape(states, [len(states), 4, 4, 4])
        err, _ = self.sess.run([self.error, self.opt_model],
                                feed_dict={self.input_state: states, self.player_node: player_node, self.pattern: pattern, self.v_: v_placeholder})
        return err

    def store(self):
        """store weight and bias"""
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        for _id in range(len(self.weight)):
            with open(self.directory + 'weight{}'.format(_id), 'w+') as f:
                weight = self.sess.run(self.weight[_id])
                shape = self.sess.run(tf.shape(self.weight[_id]))
                for i in range(shape[0]):
                    for j in range(shape[1]): f.write('{}\n'.format(weight[i, j]))
                f.close()
            with open(self.directory + 'bias{}'.format(_id), 'w+') as f:
                bias = self.sess.run(self.bias[_id])
                shape = self.sess.run(tf.shape(self.bias[_id]))
                for i in range(shape[1]): f.write('{}\n'.format(bias[0, i]))
                f.close()

