# -*- coding: utf-8 -*-
'''
InforGo: 3D-Bingo AI developed by INFOR 29th
* Game: 3D-Bingo board game, designed by MiccWan.
* MDP: Markov-Decision-Process
    An enviroment which take action as input and generate resulting state and reward, based on the initial state probability distribution and state-action probability distribution, which in 3D-Bingo game, are all deterministic.
* AI: An artificial intelligence train by Reinforcement Learning
    play:
        Use Minimax Tree Search to find the best action, state evaluation is done by neural network approximator.
    train:
        We automatically record the game, and loop through every game while simulating the process, update the neural network base on TD(0) learning.
'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import sys
import os.path
import tempfile
import argparse
import distutils.util
import math


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
LOG_DIR = '../log/tensorboard'
argv = sys.argv


class Bingo(object):

    def __init__(self, board=None):

        if board is None:
            self.board = [[[0 for i in range(4)] for j in range(4)] for k in range(4)]
            self.height = [[0 for i in range(4)] for j in range(4)]
            self.player = 1

        else:
            self.board = [[[0 for i in range(4)] for j in range(4)] for k in range(4)]
            for h in range(4):
                for r in range(4):
                    for c in range(4):
                        self.board[h][r][c] = board[h][r][c][0][0]

            self.height = [[0 for i in range(4)] for j in range(4)]
            cnt = 0
            for h in range(4):
                for r in range(4):
                    for c in range(4):
                        if self.board[h][r][c] != 0:
                            self.height[r][c] = h + 1
                            cnt += 1
            if cnt % 2 == 0:
                self.player = 1
            else:
                self.player = 2
        

    def place(self, row, col):
        '''
        place a cube on position(height, row, col), and return whether the operation is valid
        '''
        if not self.valid_action(row, col):
            return False

        # if the position is already taken
        height = self.height[row][col]
        
        # place the cube
        self.board[height][row][col] = self.player
        self.change_player()

        self.height[row][col] += 1

        return True

    def full(self):
        '''
        Return whether the board is full
        '''
        for r in range(4):
            for c in range(4):
                if self.height[r][c] < 4:
                    return False
        return True

    def valid_action(self, row, col):
        '''
        Return whether the action is valid
        '''
        if row < 0 or row > 4 or col < 0 or col > 4:
            return False

        if self.height[row][col] >= 4:
            return False

        return True

    def play(self, row, col):
        '''
        Current Player play at (row, col)
        if Draw: return 3
        if invalid action: return -1
        if player win: return current player
        if nothing happens: return 0
        '''
        player = self.player

        if self.full():
            return 3

        if not self.place(row, col):
            return -1

        if self.win(player):
            return player
        
        if self.full():
            return 3

        return 0
        
    def change_player(self):
        '''
        switch player
        '''
        if self.player == 1:
            self.player = 2
        else:
            self.player = 1

    def win(self, player):
        '''
        return True if player won, False otherwise by checking all possible winning combination
        '''
        for h in range(4):
            for r in range(4):
                flag = True
                for c in range(4):
                    if self.board[h][r][c] != player:
                        flag = False
                if flag:
                    return True

        for h in range(4):
            for c in range(4):
                flag = True
                for r in range(4):
                    if self.board[h][r][c] != player:
                        flag = False
                if flag:
                    return True

        for r in range(4):
            for c in range(4):
                flag = True
                for h in range(4):
                    if self.board[h][r][c] != player:
                        flag = False
                if flag:
                    return True
        
        for h in range(4):
            flag = True
            for i in range(4):
                if self.board[h][i][i] != player:
                    flag = False
            if flag:
                return True
            flag = True
            for i in range(4):
                if self.board[h][i][3 - i] != player:
                    flag = False
            if flag:
                return True
        
        for r in range(4):
            flag = True
            for i in range(4):
                if self.board[i][r][i] != player:
                    flag = False
            if flag:
                return True
            flag = True
            for i in range(4):
                if self.board[i][r][3 - i] != player:
                    flag = False
            if flag:
                return True

        for c in range(4):
            flag = True
            for i in range(4):
                if self.board[i][i][c] != player:
                    flag = False
            if flag:
                return True
            flag = True
            for i in range(4):
                if self.board[i][3 - i][c] != player:
                    flag = False
            if flag:
                return True
        
        flag = True
        for i in range(4):
            if self.board[i][i][i] != player:
                flag = False
        if flag:
            return True
        flag = True
        for i in range(4):
            if self.board[i][i][3 - i] != player:
                flag = False
        if flag:
            return True  
        flag = True
        for i in range(4):
            if self.board[i][3 - i][i] != player:
                flag = False
        if flag:
            return True
        flag = True
        for i in range(4):
            if self.board[3 - i][i][i] != player:
                flag = False
        if flag:
            return True

        return False

    def restart(self):
        '''
        restart the game
        '''
        self.__init__()

    def undo_action(self, row, col):
        '''
        Undo the last action at (row, col)
        '''
        self.height[row][col] -= 1
        self.board[self.height[row][col]][row][col] = 0
        
    def get_state(self):
        state = np.zeros([4, 4, 4, 1, 1])
        for h in range(4):
            for r in range(4):
                for c in range(4):
                    state[h][r][c][0][0] = self.board[h][r][c]
        return state


class MDP(object):
    '''
    Markov Decision Process
    which can be represented as M(S, D, A, P, gamma, R)
    '''
    def __init__(self, reward_function):
        # R(s, a) is the reward obtaining by taking action a at state s
        self.R = reward_function
        self.bingo = Bingo()

    def get_initial_state(self):
        '''
        Refresh the game and return the Initial state s0 based on D
        '''
        self.bingo.restart()
        return np.zeros(shape=[4, 4, 4, 1, 1])

    def get_state(self):
        return self.bingo.get_state()

    def get_reward(self, s, flag, player):
        '''
        Return the reward of tranforming into state s
        '''
        return self.R(s, flag, player)

    def take_action(self, action, player):
        '''
        Take action and Return whether the action is valid, whether the player win or not, new state and the reward
        '''
        row, col = action
        flag = self.bingo.play(row, col)

        new_state = self.get_state()
        reward = self.get_reward(new_state, flag, player)

        return flag, new_state, reward

    def valid_action(self, action):
        '''
        Return whether the action is valid
        '''
        row, col = action
        return self.bingo.valid_action(row, col)

    def undo_action(self, action):
        '''
        Undo the latest action on position (row, col)
        '''
        row, col = action
        self.bingo.undo_action(row, col)



class InforGo(object):
    '''
    A Neural Network model for training 3D-bingo AI
    input-layer: 4 x 4 x 4 board
    convolution-layer: stride = 1 x 1 x 1
    hidden-layer: relu/sigmoid/tanh function
    output-layer: approximate value for the input state
    '''
    def __init__(self, reward_function, n_epoch=100, n_hidden_layer=1, n_node_hidden=[32], activation_function='relu', output_function=None, learning_rate=0.00000001, alpha=0.00000001, gamma=0.99, td_lambda=0.85, regularization_param=0.001, decay_step=10000, decay_rate=0.96, convolution=True, filter_depth=1, filter_height=1, filter_width=1, out_channel=5, search_depth=3, DEBUG=False, first=True):

        if DEBUG:
            print("[Init] Start setting training parameter")

        self.DEBUG = DEBUG
        self.first = first
        self.n_epoch = n_epoch
        self.alpha = alpha

        # Learning rate decay
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(learning_rate, self.global_step, decay_step, decay_rate, staircase=True)

        # Discount factor between 0 to 1
        self.gamma = gamma

        # lambda of TD-lambda algorithm
        self.td_lambda = td_lambda

        # R(s) is the reward obtain by achieving state s
        self.reward_function = reward_function

        # Number of hidden layers and number of nodes in each hidden layer
        self.n_hidden_layer = n_hidden_layer
        self.n_node_hidden = n_node_hidden
        
        # Maximum search depth in Minimax Tree Search
        self.search_depth = search_depth

        # Activation function at hidden layer
        if activation_function == 'relu':
            self.activation_function = lambda k: tf.nn.relu(k, name='relu')

        elif activation_function == 'sigmoid':
            self.activation_function = lambda k: tf.sigmoid(k, name='sigmoid')

        elif activation_function == 'tanh':
            self.activation_function = lambda k: tf.tanh(k, name='tanh')

        # Output function 
        if output_function is None:
            self.output_function = lambda k: k

        elif output_function == 'softmax':
            self.output_function = lambda k: tf.nn.softmax(k)

        # L2 regularization paramater
        self.regularization_param = regularization_param
        
        self.MDP = MDP(reward_function)
        if self.DEBUG:
            print("[Init] Done setting training parameter")

        # Neural Network Setup

        # input layer: 4 x 4 x 4 Tensor representing the state
        with tf.name_scope('Input-Layer'):
            self.inp = tf.placeholder(shape=[4, 4, 4, 1, 1], dtype=tf.float64, name='input')

        if self.DEBUG:
            print("[Init] Done consturcting input layer")

        # 3D-Convolution layer
        if convolution:
            with tf.name_scope('Convolution-Layer'):
                self.conv_layer_w = tf.cast(tf.Variable(tf.truncated_normal(shape=[filter_depth, filter_height, filter_width, 1, out_channel], mean=0.0, stddev=1.0)), tf.float64, name='weight')
                self.conv_layer = tf.nn.conv3d(input=self.inp, filter=self.conv_layer_w, strides=[1, 1, 1, 1, 1], padding='SAME', name='Conv-Layer')

                # Flatten the convolution layer
                self.conv_layer_output = tf.cast(tf.reshape(self.conv_layer, [1, -1], name='Flattend'), dtype=tf.float64)

                # Caculate the length of flattened layer
                self.conv_layer_length = 4 * 4 * 4 * out_channel
        else:
            self.conv_layer_output = tf.reshape(self.inp, [1, -1], name='Flattened')
            self.conv_layer_length = 4 * 4 * 4

        if self.DEBUG:
            print("[Init] Done constructing convolution layer with out_channel = {}".format(out_channel))
        
        with tf.name_scope('Player-Node'):
            self.player_node = tf.placeholder(shape=[1, 1], dtype=tf.float64, name='Player-Node')

        with tf.name_scope('Pattern'):
            self.pattern = tf.placeholder(shape=[1, 6], dtype=tf.float64, name='Pattern')

        # Store all the weight and bias between each layer
        self.weight_and_bias = [{} for i in range(self.n_hidden_layer + 1)]
        
        with tf.name_scope('Weight_and_Bias'):
            self.weight_and_bias[0] = {
                'Weight': self.get_weight(self.conv_layer_length + 1 + 6, self.n_node_hidden[0], 0),
                'Bias': self.get_bias(self.n_node_hidden[0], 0)
            }
            if self.DEBUG:
                print("[Init] Done initializing weight and bias from convolution layer to hidden layer 0")
            for i in range(1, self.n_hidden_layer):
                self.weight_and_bias[i] = {
                    'Weight': self.get_weight(self.n_node_hidden[i - 1], self.n_node_hidden[i], i),
                    'Bias': self.get_bias(self.n_node_hidden[i], i)
                }
                if self.DEBUG:
                    print("[Init] Done initializing weight and bias from hidden layer {} to hidden layer {}".format(i - 1, i))
            self.weight_and_bias[self.n_hidden_layer] = {
                'Weight': self.get_weight(self.n_node_hidden[self.n_hidden_layer - 1], 1, self.n_hidden_layer),
                'Bias': self.get_bias(1, self.n_hidden_layer)
            }
            if self.DEBUG:
                print("[Init] Done initializing weight and bias from hidden layer {} to output layer".format(self.n_hidden_layer - 1))

        # Store value of every node in each hidden layer
        self.hidden_layer = [{} for i in range(self.n_hidden_layer)]

        with tf.name_scope('Hidden_Layer'):
            self.hidden_layer[0] = {
                'Output': tf.add(tf.matmul(tf.concat([self.conv_layer_output, self.player_node, self.pattern], 1), self.weight_and_bias[0]['Weight']), self.weight_and_bias[0]['Bias'])
            }
            for i in range(1, self.n_hidden_layer):
                self.hidden_layer[i - 1]['Activated_Output'] = self.activation_function(self.hidden_layer[i - 1]['Output'])
                if self.DEBUG:
                    print("[Init] Done activating output of hidden layer {}".format(i - 1))
                self.hidden_layer[i] = {
                    'Output': tf.add(tf.matmul(self.hidden_layer[i - 1]['Activated_Output'], self.weight_and_bias[i]['Weight']), self.weight_and_bias[i]['Bias'])
                }
            self.hidden_layer[self.n_hidden_layer - 1]['Activated_Output'] = self.hidden_layer[self.n_hidden_layer - 1]['Output']
            if self.DEBUG:
                print("[Init] Done activating output of hidden layer {}".format(self.n_hidden_layer - 1))

        with tf.name_scope('Output_Layer'):
            self.output = tf.add(tf.matmul(self.hidden_layer[self.n_hidden_layer - 1]['Activated_Output'], self.weight_and_bias[self.n_hidden_layer]['Weight'], ), self.weight_and_bias[self.n_hidden_layer]['Bias'])
            self.V = self.output_function(self.output)
            if self.DEBUG:
                print("[Init] Done constructing output layer")

        with tf.name_scope('Training_Model'):
            # Q-value to update the weight
            self.V_desired = tf.placeholder(shape=[1, 1], dtype=tf.float64)

            # Cost function
            def L2_Regularization():
                self.L2_value = 0
                for i in range(0, self.n_hidden_layer + 1):
                    self.L2_value += tf.nn.l2_loss(self.weight_and_bias[i]['Weight']) + tf.nn.l2_loss(self.weight_and_bias[i]['Bias'])
                return self.L2_value

            self.loss = tf.reduce_sum(tf.square(self.V_desired - self.V))
            if self.DEBUG:
                print("[Init] Done caculating cost function")
            # use gradient descent to optimize our model
            self.trainer = tf.train.GradientDescentOptimizer(self.learning_rate)
            self.model = self.trainer.minimize(self.loss, global_step=self.global_step)
            if self.DEBUG:
                print("[Init] Done setting up trainer")

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        if self.DEBUG:
            print("[Init] Done initializing all variables")
    
    def get_weight(self, n, m, layer):
        '''
        Weight to the layer-th hidden layer, with size n x m
        '''
        if os.path.exists('../Data/Weight/{}.txt'.format(layer)):
            f = open('../Data/Weight/{}.txt'.format(layer), 'r')
            w = np.zeros([n, m])
            for i in range(n):
                for j in range(m):
                    try:
                        w[i, j] = float(f.readline())
                    except:
                        if self.DEBUG:
                            print("[ERROR] NaN or unstored weight")
                        os.remove('../Data/Weight/{}.txt'.format(layer))
                        return tf.truncated_normal(shape=[n, m], mean=0.0, stddev=0.001, dtype=tf.float64)
            f.close()
            return tf.Variable(tf.cast(w, tf.float64))
        else:
            return tf.Variable(tf.truncated_normal(shape=[n, m], mean=0.0, stddev=0.001, dtype=tf.float64))

    def get_bias(self, n, layer):
        '''
        Bias to the layer-th hidden layer, with size 1 x n
        '''
        if os.path.exists('../Data/Bias/{}.txt'.format(layer)):
            f = open('../Data/Bias/{}.txt'.format(layer), 'r')
            b = np.zeros([1, n])
            for i in range(n):
                try:
                    b[0, i] = float(f.readline())
                except:
                    if self.DEBUG:
                        print("[ERROR] NaN or unstored bias")
                    os.remove('../Data/Bias/{}.txt'.format(layer))
                    return tf.Variable(tf.truncated_normal([1, n], mean=0.0, stddev=0.001, dtype=tf.float64))
            f.close()
            return tf.Variable(tf.cast(b, tf.float64))
        else:
            return tf.Variable(tf.truncated_normal(shape=[1, n], mean=0.0, stddev=0.001, dtype=tf.float64))

    def get_pattern(self, state, player):
        opponent = 0
        if player == 1:
            opponent = 2
        else:
            opponent = 1
        corner = [0, 0]
        two = [0, 0]
        three = [0, 0]
        for i in range(4):
            if state[i][i][i][0][0] == player:
                corner[0] += 1
            elif state[i][i][i][0][0] == opponent:
                corner[1] += 1
            if state[i][3 - i][3 - i][0][0] == player:
                corner[0] += 1
            elif state[i][3 - i][3 - i][0][0] == opponent:
                corner[1] += 1
            if state[i][3 - i][i][0][0] == player:
                corner[0] += 1
            elif state[i][3 - i][i][0][0] == opponent:
                corner[1] += 1
            if state[i][i][3 - i][0][0] == player:
                corner[0] += 1
            elif state[i][i][3 - i][0][0] == opponent:
                corner[1] += 1

        for h in range(4):
            for r in range(4):
                cnt = [0, 0]
                for c in range(4):
                    if state[h][r][c][0][0]:
                        cnt[int(state[h][r][c][0][0]) - 1] += 1
                if cnt[0] == 2 and cnt[1] == 0:
                    two[0] += 1
                if cnt[1] == 2 and cnt[0] == 0:
                    two[1] += 1
                if cnt[0] == 3 and cnt[1] == 0:
                    three[0] += 1
                if cnt[1] == 3 and cnt[0] == 0:
                    three[1] += 1
            for c in range(4):
                cnt = [0, 0]
                for r in range(4):
                    if state[h][r][c][0][0]:
                        cnt[int(state[h][r][c][0][0]) - 1] += 1
                if cnt[0] == 2 and cnt[1] == 0:
                    two[0] += 1
                if cnt[1] == 2 and cnt[0] == 0:
                    two[1] += 1
                if cnt[0] == 3 and cnt[1] == 0:
                    three[0] += 1
                if cnt[1] == 3 and cnt[0] == 0:
                    three[1] += 1
            cnt = [0, 0]
            for i in range(4):
                if state[h][i][i][0][0]:
                    cnt[int(state[h][i][i][0][0]) - 1] += 1
            if cnt[0] == 2 and cnt[1] == 0:
                two[0] += 1
            if cnt[1] == 2 and cnt[0] == 0:
                two[1] += 1
            if cnt[0] == 3 and cnt[1] == 0:
                three[0] += 1
            if cnt[1] == 3 and cnt[0] == 0:
                three[1] += 1        
            cnt = [0, 0]
            for i in range(4):
                if state[h][i][3 - i][0][0]:
                    cnt[int(state[h][i][3 - i][0][0]) - 1] += 1
            if cnt[0] == 2 and cnt[1] == 0:
                two[0] += 1
            if cnt[1] == 2 and cnt[0] == 0:
                two[1] += 1
            if cnt[0] == 3 and cnt[1] == 0:
                three[0] += 1
            if cnt[1] == 3 and cnt[0] == 0:
                three[1] += 1

        for r in range(4):
            for c in range(4):
                cnt = [0, 0]
                for h in range(4):
                    if state[h][r][c][0][0]:
                        cnt[int(state[h][r][c][0][0]) - 1] += 1
                if cnt[0] == 2 and cnt[1] == 0:
                    two[0] += 1
                if cnt[1] == 2 and cnt[0] == 0:
                    two[1] += 1
                if cnt[0] == 3 and cnt[1] == 0:
                    three[0] += 1
                if cnt[1] == 3 and cnt[0] == 0:
                    three[1] += 1
            cnt = [0, 0]
            for i in range(4):
                if state[i][r][i][0][0]:
                    cnt[int(state[i][r][i][0][0]) - 1] += 1
            if cnt[0] == 2 and cnt[1] == 0:
                two[0] += 1
            if cnt[1] == 2 and cnt[0] == 0:
                two[1] += 1
            if cnt[0] == 3 and cnt[1] == 0:
                three[0] += 1
            if cnt[1] == 3 and cnt[0] == 0:
                three[1] += 1
            cnt = [0, 0]
            for i in range(4):
                if state[i][r][3 - i][0][0]:
                    cnt[int(state[i][r][3 - i][0][0]) - 1] += 1
            if cnt[0] == 2 and cnt[1] == 0:
                two[0] += 1
            if cnt[1] == 2 and cnt[0] == 0:
                two[1] += 1
            if cnt[0] == 3 and cnt[1] == 0:
                three[0] += 1
            if cnt[1] == 3 and cnt[0] == 0:
                three[1] += 1
        for c in range(4):
            cnt = [0, 0]
            for i in range(4):
                if state[i][i][c][0][0]:
                    cnt[int(state[i][i][c][0][0]) - 1] += 1
            if cnt[0] == 2 and cnt[1] == 0:
                two[0] += 1
            if cnt[1] == 2 and cnt[0] == 0:
                two[1] += 1
            if cnt[0] == 3 and cnt[1] == 0:
                three[0] += 1
            if cnt[1] == 3 and cnt[0] == 0:
                three[1] += 1
            cnt = [0, 0]
            for i in range(4):
                if state[i][3 - i][c][0][0]:
                    cnt[int(state[i][3 - i][c][0][0]) - 1] += 1
            if cnt[0] == 2 and cnt[1] == 0:
                two[0] += 1
            if cnt[1] == 2 and cnt[0] == 0:
                two[1] += 1
            if cnt[0] == 3 and cnt[1] == 0:
                three[0] += 1
            if cnt[1] == 3 and cnt[0] == 0:
                three[1] += 1
        cnt = [0, 0]
        for i in range(4):
            if state[i][i][i][0][0]:
                cnt[int(state[i][i][i][0][0]) - 1] += 1
        if cnt[0] == 2 and cnt[1] == 0:
            two[0] += 1
        if cnt[1] == 2 and cnt[0] == 0:
            two[1] += 1
        if cnt[0] == 3 and cnt[1] == 0:
            three[0] += 1
        if cnt[1] == 3 and cnt[0] == 0:
            three[1] += 1
        cnt = [0, 0]
        for i in range(4):
            if state[i][i][3 - i][0][0]:
                cnt[int(state[i][i][3 - i][0][0]) - 1] += 1
        if cnt[0] == 2 and cnt[1] == 0:
            two[0] += 1
        if cnt[1] == 2 and cnt[0] == 0:
            two[1] += 1
        if cnt[0] == 3 and cnt[1] == 0:
            three[0] += 1
        if cnt[1] == 3 and cnt[0] == 0:
            three[1] += 1
        cnt = [0, 0]
        for i in range(4):
            if state[3 - i][i][i][0][0]:
                cnt[int(state[3 - i][i][i][0][0]) - 1] += 1
        if cnt[0] == 2 and cnt[1] == 0:
            two[0] += 1
        if cnt[1] == 2 and cnt[0] == 0:
            two[1] += 1
        if cnt[0] == 3 and cnt[1] == 0:
            three[0] += 1
        if cnt[1] == 3 and cnt[0] == 0:
            three[1] += 1
        cnt = [0, 0]
        for i in range(4):
            if state[i][3 - i][i][0][0]:
                cnt[int(state[i][3 - i][i][0][0]) - 1] += 1
        if cnt[0] == 2 and cnt[1] == 0:
            two[0] += 1
        if cnt[1] == 2 and cnt[0] == 0:
            two[1] += 1
        if cnt[0] == 3 and cnt[1] == 0:
            three[0] += 1
        if cnt[1] == 3 and cnt[0] == 0:
            three[1] += 1
        pattern = [corner[0], corner[1], two[0], two[1], three[0], three[1]]
        return np.reshape(np.array(pattern), [1, -1])

    def decode_action(self, action_num):
        action = [0, 0]
        for i in range(2):
            action[i] = action_num % 4
            action_num //= 4
        return action

    def train(self, run_test=True, run_self_play=True, run_generator=True, n_generator=1000, MAX=0):
        '''
        Main Learning Process
        return final score, graph_x, graph_y
        '''
        writer = tf.summary.FileWriter(LOG_DIR)
        writer.add_graph(self.sess.graph)
        if self.DEBUG:
            print("[Train] Done Tensorboard setup")
            print("[Train] Start training")
        percentage = 0
        record = self.get_record(run_test, run_self_play, run_generator, n_generator, MAX)
        if self.DEBUG:
            print("[Train] Done Collecting record")
            print("[Train] Training Complete: {}%".format(percentage))

        loss = []
        for epoch in range(self.n_epoch):
            loss_sum = 0
            update = 0
            for directory in record.keys():
                for file_name in record[directory]:
                    for rotate_time in range(4):
                        f = open('{}/{}'.format(directory, file_name), 'r')
                        s = self.MDP.get_initial_state()

                        while True:
                            try:
                                height, row, col = map(int, f.readline().split())
                            except:
                                if self.DEBUG:
                                    print("[ERROR] Invalid file format or context {}".format(file_name))
                                break

                            if (height, row, col) == (-1, -1, -1):
                                break

                            height, row, col = self.rotate(height, row, col, rotate_time)

                            v = self.sess.run(self.V, feed_dict={self.inp: s, self.player_node: self.cast_player(1), self.pattern: self.get_pattern(s, 1)})
                            flag, new_s, R = self.MDP.take_action((row, col), 1)

                            new_v = self.sess.run(self.V, feed_dict={self.inp: new_s, self.player_node: self.cast_player(1), self.pattern: self.get_pattern(new_s, 1)})
                            v_desired = np.zeros([1, 1])
                            # TD-0 update
                            # v_desired[0][0] = new_v[0][0] * self.td_lambda + self.alpha * (1 - self.td_lambda) * (R + self.gamma * new_v[0][0] - v[0][0]) 
                            v_desired[0][0] = v[0][0] + self.alpha * (R + self.gamma * new_v[0][0] - v[0][0])
                            # loss.append(self.sess.run(self.loss, feed_dict={self.V_desired: v_desired, self.inp: s, self.player_node: self.cast_player(1)}))
                            loss_sum += self.sess.run(self.loss, feed_dict={self.V_desired: v_desired, self.inp: s, self.player_node: self.cast_player(1), self.pattern: self.get_pattern(s, 1)})
                            update += 1
                            self.sess.run(self.model, feed_dict={self.V_desired: v_desired, self.inp: s, self.player_node: self.cast_player(1), self.pattern: self.get_pattern(s, 1)})
                            s = new_s

                            try:
                                height, row, col = map(int, f.readline().split())
                            except:
                                if self.DEBUG:
                                    print("[ERROR] Invalid file format or context {}".format(file_name))
                                break

                            if (height, row, col) == (-1, -1, -1):
                                break

                            height, row, col = self.rotate(height, row, col, rotate_time)

                            v = self.sess.run(self.V, feed_dict={self.inp: s, self.player_node: self.cast_player(2), self.pattern: self.get_pattern(s, 2)})
                            flag, new_s, R = self.MDP.take_action((row, col), 2)

                            new_v = self.sess.run(self.V, feed_dict={self.inp: new_s, self.player_node: self.cast_player(2), self.pattern: self.get_pattern(new_s, 2)})
                            # TD-0 update
                            # v_desired[0][0] = new_v[0][0] * self.td_lambda + self.alpha * (1 - self.td_lambda) * (R + self.gamma * new_v[0][0] - v[0][0]) 
                            v_desired[0][0] = v[0][0] + self.alpha * (R + self.gamma * new_v[0][0] - v[0][0])
                            # loss.append(self.sess.run(self.loss, feed_dict={self.V_desired: v_desired, self.inp: s, self.player_node: self.cast_player(1)}))
                            loss_sum += self.sess.run(self.loss, feed_dict={self.V_desired: v_desired, self.inp: s, self.player_node: self.cast_player(2), self.pattern: self.get_pattern(s, 2)})
                            update += 1
                            self.sess.run(self.model, feed_dict={self.V_desired: v_desired, self.inp: s, self.player_node: self.cast_player(1), self.pattern: self.get_pattern(s, 1)})
                            s = new_s

            loss.append(loss_sum / update)
            if epoch / self.n_epoch > percentage / 100:
                percentage = math.ceil(epoch / self.n_epoch * 100)    
                if self.DEBUG:
                    print("[Train] Training Complete: {}%".format(percentage))
            if percentage % 10 == 0:
                self.store_weight_and_bias()
                
        if self.DEBUG:
            print("[Train] Training Complete: {}%".format(100))
        self.store_weight_and_bias()
        return loss

    def rotate(self, height, row, col, t):
        '''
        Rotate the board 90 x t degree clockwise
        '''
        for i in range(t):
            row, col = col, row
            col = 3 - col
        return height, row, col

    def get_record(self, run_test=True, run_self_play=True, run_generator=True, n_generator=1000, MAX=0):
        '''
        Return every record file under ../Data/record/*
        '''
        directory = [x[0] for x in os.walk('../Data/record')]
        directory = directory[1:]
        filename = {}
        for d in directory:
            if d == '../Data/record/test_record' and not run_test:
                continue
            if d == '../Data/record/self_play' and not run_self_play:
                continue
            if d == '../Data/record/generator':
                continue
            tmp = [x[2] for x in os.walk(d)]
            filename[d] = [x for x in tmp[0]]
        if not run_generator:
            return filename
        s = set([])
        filename['../Data/record/generator'] = []
        for i in range(n_generator):
            game_id = random.randint(0, MAX)
            while game_id in s or not os.path.exists('../Data/record/generator/{}'.format(game_id)):
                game_id = random.randint(0, MAX)
            s.add(game_id)
            filename['../Data/record/generator'].append('{}'.format(game_id))
        return filename           

    def store_weight_and_bias(self):
        '''
        Store weights under ./Data/Weight, biases under ./Data/Bias
        '''
        for i in range(self.n_hidden_layer + 1):
            f = open('../Data/Weight/{}.txt'.format(i), 'w')
            w = self.sess.run(self.weight_and_bias[i]['Weight'])
            for j in range(self.sess.run(tf.shape(self.weight_and_bias[i]['Weight']))[0]):
                for k in range(self.sess.run(tf.shape(self.weight_and_bias[i]['Weight']))[1]):
                    f.write('{}\n'.format(w[j, k]))
            f.close()
            if self.DEBUG:
                print("[Train] Done storing weight {}".format(i))

            f = open('../Data/Bias/{}.txt'.format(i), 'w')
            b = self.sess.run(self.weight_and_bias[i]['Bias'])
            for j in range(self.sess.run(tf.shape(self.weight_and_bias[i]['Bias']))[1]):
                f.write('{}\n'.format(b[0, j]))
            f.close()
            if self.DEBUG:
                print("[Train] Done storing bias {}".format(i))

    def play(self, test_flag=False, bot=None, AI=None):
        if test_flag:
            tmp = tempfile.NamedTemporaryFile(dir='../Data/record/test_record', delete=False)
        elif AI is not None:
            tmp = tempfile.NamedTemporaryFile(dir='../Data/record/self_play', delete=False)
        else:
            tmp = tempfile.NamedTemporaryFile(dir='../Data/record/selfrecord', delete=False)
        winner = 0
        s = self.MDP.get_initial_state()
        record = ''
        if self.DEBUG and not test_flag and AI is None:
            print("[Play] Start playing")

        player = 1
        if self.first is False:
            if self.DEBUG and not self.test_flag and AI is None:
                print("[Play] Enter position")
            try:
                opponent = self.read_opponent_action(test_flag, bot, AI=AI)
            except:
                if self.DEBUG and not self.test_flag and AI is None:
                    print("[ERROR] Fail to read opponent action")
                os.remove(tmp.name)
                return
            while self.MDP.valid_action(opponent) is False:
                if self.DEBUG and not self.test_flag and AI is None:
                    print("[FATAL] Invalid")
                    print("[FATAL] Re-enter position")
                try:
                    opponent = self.read_opponent_action(test_flag, bot, AI=AI)
                except:
                    if not self.test_flag and AI is None:
                        print("[ERROR] Fail to read opponent action")
            row, col = opponent
            height = self.MDP.bingo.height[row][col]
            record += '{} {} {}\n'.format(height, row, col)

            flag, s, _ = self.MDP.take_action(opponent, player)
            if player == 1:
                player = 2
            else:
                player = 1
            if flag == 1:
                record += '-1 -1 -1\n'
                if self.DEBUG and not test_flag and AI is None:
                    print("[Play] User win")
                winner = 1

        while True:
            # Choose the best action using Minimax Tree Search
            _, action = self.Minimax(Bingo(s), self.search_depth, 'Max', player)
            row, col = self.decode_action(action)
            
            height = self.MDP.bingo.height[row][col]
            self.emit_action(height, row, col, test_flag, AI)
            record += '{} {} {}\n'.format(height, row, col)

            action = (row, col)
            flag, s, _ = self.MDP.take_action(action, player)

            if flag == player:
                record += '-1 -1 -1\n'
                if self.DEBUG and not test_flag and AI is None:
                    print("[Play] AI win")
                winner = player
                break

            if player == 1:
                player = 2
            else:
                player = 1

            if self.DEBUG and not test_flag and AI is None:
                print("[Play] Enter position")

            try:
                opponent = self.read_opponent_action(test_flag, bot, AI=AI)
            except:
                if self.DEBUG and not test_flag and AI is None:
                    print("[ERROR] Invalid Opponent Move")
                os.remove(tmp.name)
                break
            
            success = True
            while self.MDP.valid_action(opponent) is False:
                if self.DEBUG and not test_flag and AI is None:
                    print("[FATAL] Invalid input action")
                    print("[FATAL] Re-enter position")
                try:
                    opponent = self.read_opponent_action(test_flag, bot, AI=AI)
                except:
                    if self.DEBUG and not test_flag and AI is None:
                        print("[ERROR] Invalid Opponent Move")
                    os.remove(tmp.name)
                    success = False
                    break
            
            if not success:
                break
            row, col = opponent
            height = self.MDP.bingo.height[row][col]
            record += '{} {} {}\n'.format(height, row, col)

            flag, s, _ = self.MDP.take_action(opponent, player)

            if flag == player:
                record += '-1 -1 -1\n'
                if self.DEBUG and not test_flag:
                    print("[Play] User win")
                winner = player
                break

            if player == 1:
                player = 2
            else:
                player = 1
        # Record the game for future training
        f = open(tmp.name, 'w')
        f.write(record)
        f.close()
        return winner

    def Minimax(self, bingo, depth, level, player, alpha=-np.inf, beta=np.inf):
        '''
        recursively search every possible action with maximum depth = self.search_depth
        use alpha-beta pruning to reduce time complexity
        '''
        state = bingo.get_state()

        if bingo.win(1) or bingo.win(2):
            return self.evaluate(state, player), None

        if depth == 0:
            return self.evaluate(state, player), None
        
        value, action = 0, 0
        next_player = 0
        next_level = 'Osas'
        func = lambda a, b: 0

        if level == 'Max':
            value = -np.inf
            next_level = 'Min'
            func = lambda a, b: max(a, b)

        else:
            value = np.inf
            next_level = 'Max'
            func = lambda a, b: min(a, b)

        if player == 1:
            next_player = 2
        else:
            next_player = 1

        move = False

        for i in random.shuffle(range(16)):
            r, c = self.decode_action(i)
            if bingo.valid_action(r, c):
                move = True
                bingo.place(r, c)
                new_bingo = Bingo(bingo.get_state())
                bingo.undo_action(r, c)

                val, a = self.Minimax(new_bingo, depth - 1, next_level, next_player, alpha, beta)

                if level == 'Min':
                    beta = min(beta, val)
                else:
                    alpha = max(alpha, val)

                value = func(value, val)
                
                # Lowerbound is greater than the upperbound, stop further searching
                if alpha > beta:
                    return value, action

                if value == val:
                    action = i

        if not move:
            return 0, None

        return value, action

    def evaluate(self, state, player):
        '''
        Evaluate the value of input state with neural network as an approximater
        '''
        V = self.sess.run(self.V, feed_dict={self.inp: state, self.player_node: self.cast_player(player)})
        return V[0][0]
    
    def cast_player(self, player):
        tmp = np.zeros([1, 1])
        node = 0
        if player == 1:
            node = 1
        else:
            node = -1
        tmp[0, 0] = node
        return tmp

    def read_opponent_action(self, test_flag, bot, AI=None):
        if AI is not None:
            state = self.MDP.get_state()
            player = 0
            if AI.first:
                player = 1
            else:
                player = 2
            value, action = AI.Minimax(Bingo(state), AI.search_depth, 'Max', 2)
            return self.decode_action(action)

        if test_flag:
            return bot.generate_action(self.MDP.get_state())

        try:
            h, r, c = map(int, input().split())
        except:
            raise
        return r, c
        
    def emit_action(self, height, row, col, test_flag, AI=None):
        if test_flag or AI is not None:
            return
        if self.DEBUG:
            print("[DEBUG] ", end='')
            print("{} {} {}".format(height, row, col))
        else:
            print(height + row * 4 + col * 16)

    def test(self):
        bot = None
        win = 0
        player = 0
        percentage = 0
        if self.first:
            player = 1
            bot = Bot(2)
        else:
            player = 2
            bot = Bot(1)
        if self.DEBUG:
            print("[Test] Test Complete: {}%".format(0))
        for epoch in range(self.n_epoch):
            winner = self.play(test_flag=True, bot=bot)
            if winner == player:
                win += 1

            if epoch / self.n_epoch > percentage / 100:
                percentage = math.ceil(epoch / self.n_epoch * 100)    
                if self.DEBUG:
                    print("[Test] Test Complete: {}%".format(percentage))
        if self.DEBUG:
            print("[Test] Test Complete: {}%".format(100))
        print("[Test] Winning Percentage: {}%".format(win / self.n_epoch * 100.))
        return win / self.n_epoch * 100.


class Bot:

    def __init__(self, player):
        self.player = player
        self.opponent = 0
        if self.player == 1:
            self.opponent = 2
        else:
            self.opponent = 1
    
    def generate_action(self, state):
        bingo = Bingo(state)
        for i in range(4):
            for j in range(4):
                if bingo.valid_action(i, j):
                    bingo.place(i, j)
                    if bingo.win(self.player):
                        return i, j
                    bingo.undo_action(i, j)

        bingo.change_player()
        for i in range(4):
            for j in range(4):
                if bingo.valid_action(i, j):
                    bingo.place(i, j)
                    if bingo.win(self.opponent):
                        return i, j
                    bingo.undo_action(i, j)
        return random.randint(0, 3), random.randint(0, 3)


def self_play(AI, args):
    opponent = InforGo(
        reward_function=AI.reward_function, 
        n_epoch=args.n_epoch, 
        n_hidden_layer=args.n_hidden_layer,
        n_node_hidden=args.n_node_hidden,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        alpha=args.alpha,
        regularization_param=args.regularization_param,
        decay_step=args.decay_step,
        decay_rate=args.decay_rate,
        convolution=args.convolution,
        filter_depth=args.filter_depth,
        filter_height=args.filter_height,
        filter_width=args.filter_width,
        out_channel=args.out_channel,
        DEBUG=args.DEBUG,
        first=not args.first,
        search_depth=args.search_depth,
        activation_function=args.activation_function,
        output_function=args.output_function)

    percentage = 0
    first_win = 0
    second_win = 0

    if args.DEBUG:
        print("[Self-Play] Self-Play Complete: {}%".format(0))
    for epoch in range(args.n_epoch):
        winner = AI.play(AI=opponent)
        if winner == 1:
            first_win += 1
        else:
            second_win += 1
        if epoch / args.n_epoch > percentage / 100:
                percentage = math.ceil(epoch / args.n_epoch * 100)    
                if args.DEBUG:
                    print("[Self-Play] Self-Play Complete: {}%".format(percentage))
    if args.DEBUG:
        print("[Self-Play] Self-Play Complete: {}%".format(100))
    return first_win, second_win


def main():
       
    parser = argparse.ArgumentParser(description='Execution argument')
        
    # Method
    parser.add_argument('method', help='play/train/self-play/test')

    # Log
    parser.add_argument('--logdir', default='./tensorboard', help='Tensorboard log directory')

    # Training parameter
    parser.add_argument('--learning_rate', default=0.00000001, type=float, help='learning rate for the neural network')
    parser.add_argument('--gamma', default=0.99, type=float, help='discount factor')
    parser.add_argument('--alpha', default=0.00000001, type=float, help='learning rate for TD(0)-learning')
    parser.add_argument('--regularization_param', default=0.001, type=float, help='L2 regularization')
    parser.add_argument('--decay_rate', default=0.96, type=float, help='Decay rate')
    parser.add_argument('--decay_step', default=100, type=int, help='Decay step')

    # Model parameter
    parser.add_argument('--n_epoch', default=100, type=int, help='number of epochs')
    parser.add_argument('--n_hidden_layer', default=1, type=int, help='number of hidden layers')
    parser.add_argument('--n_node_hidden', default=[32], type=int, nargs='+', help='nodes in each hidden layers')
        
    # Neuron
    parser.add_argument('--activation_function', default='relu', type=str, help='activation function')
    parser.add_argument('--output_function', default=None, type=str, help='output function')

    # Convolution Layer
    parser.add_argument('--convolution', default=True, type=distutils.util.strtobool, help='With/Without convolution layer')
    parser.add_argument('--filter_depth', default=1, type=int, help='filter depth')
    parser.add_argument('--filter_height', default=1, type=int, help='filter height')
    parser.add_argument('--filter_width', default=1, type=int, help='filter width')
    parser.add_argument('--out_channel', default=5, type=int, help='out channel')

    # DEBUG
    parser.add_argument('--DEBUG', default=False, type=distutils.util.strtobool, help='Debug mode')
        
    # Play
    parser.add_argument('--first', default=True, type=distutils.util.strtobool, help='Play first')
    parser.add_argument('--search_depth', default=3, type=int, help='maximum search depth')

    # Train
    parser.add_argument('--run_test', default=True, type=distutils.util.strtobool, help='Train the model with testing data')
    parser.add_argument('--run_self_play', default=True, type=distutils.util.strtobool, help='Train the model with self-play data')
    parser.add_argument('--run_generator', default=True, type=distutils.util.strtobool, help='Train the model with auto-generated game')
    parser.add_argument('--n_generator', default=1000, type=int, help='Train the model with n_generator auto-generated game')
    parser.add_argument('--MAX', default=12877521, type=int, help='Maximum generated game id')

    args = parser.parse_args()

    def reward_function(state, flag, player):
        if flag == 3 or flag == 0:
            return 0
        if flag == player:
            return 1
        if flag != player:
            return -1
        return 0
        
    LOG_DIR = './log' + args.logdir

    AI = InforGo(
        reward_function=reward_function, 
        n_epoch=args.n_epoch, 
        n_hidden_layer=args.n_hidden_layer,
        n_node_hidden=args.n_node_hidden,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        alpha=args.alpha,
        regularization_param=args.regularization_param,
        decay_step=args.decay_step,
        decay_rate=args.decay_rate,
        convolution=args.convolution,
        filter_depth=args.filter_depth,
        filter_height=args.filter_height,
        filter_width=args.filter_width,
        out_channel=args.out_channel,
        DEBUG=args.DEBUG,
        first=args.first,
        search_depth=args.search_depth,
        activation_function=args.activation_function,
        output_function=args.output_function)
    
    # TODO: run_generator is True even if specified False
    if args.method == 'train':
        loss = AI.train(run_test=args.run_test, run_self_play=args.run_self_play, run_generator=args.run_generator, n_generator=args.n_generator, MAX=args.MAX)
        try:
            f = open('../tmp', 'w')
            for i in loss:
                f.write('{}\n'.format(i))
            f.close()
        except:
            for i in loss:
                print(i, end=' ')
        plt.plot([i for i in range(len(loss))], loss)
        plt.show()
            
    elif args.method == 'play':
        AI.play()

    elif args.method == 'test':
        AI.test()

    elif args.method == 'self-play':
        a, b = self_play(AI, args)
        print("{} {}".format(a, b))


if __name__ == '__main__':
    main()
