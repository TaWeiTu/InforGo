import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import sys
import os.path

argv = sys.argv
f = open("record-" + argv[1] + ".txt", "w")

class Bingo(object):

    def __init__(self):
        self.board = [[[0 for i in range(4)] for j in range(4)] for k in range(4)]
        self.height = [[0 for i in range(4)] for j in range(4)]

        # player = 1 for the player who play first and player = 2 for the opposite
        self.player = 1

    def place(self, row, col):
        '''
        place a cube on position(height, row, col), and return whether the operation is valid
        '''
        # if the position is already taken
        height = self.height[row][col]
        
        if height >= 4:
            return False

        if self.board[height][row][col] != 0:
            return False

        # if there's no placed cube underneath out position
        if height > 0 and self.board[height - 1][row][col] == 0:
            return False

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

    def play(self, row, col):
        '''
        Player1 go first, return 1 if player1 win
        Player2 go second, return 2 if player2 win
        return -1 if the action is invalid
        return 3 if it's tie
        return 0 otherwise
        '''
        if self.full():
            f.write("Draw")
            return 3

        if not self.place(row, col):
            return -1

        f.write("[{}, {}, {}]".format(self.height[row][col] - 1, row, col))

        if self.win(1):
            f.write("Player1 win")
            return 1
        
        if self.full():
            f.write("Draw")
            return 3

        # Simple bot take action
        r, c = self.generate_move()

        while not self.place(r, c):
            r, c = self.generate_move()

        f.write("[{}, {}, {}], ".format(self.height[r][c] - 1, r, c))

        if self.win(2):
            f.write("Player2 win")
            return 2

        if self.full():
            f.write("Draw")
            return 3

        return 0
        

    def plot(self):
        '''
        Debug: plot the board
        '''
        for r in range(4):
            for h in range(4):
                print(" | ", end='')
                for c in range(4):
                    print(self.board[h][r][c], end='')
            print()

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

    def generate_move(self):
        '''
        1. if player2 can win
        2. if player1 is going to win
        3. random
        '''
        for h in range(4):
            for r in range(4):
                for c in range(4):
                    if self.board[h][r][c] == 0 and (h == 0 or self.board[h - 1][r][c] != 0):
                        self.board[h][r][c] = 2
                        move = False
                        if self.win(2):
                            move = True
                        self.board[h][r][c] = 0
                        if move:
                            return r, c

        for h in range(4):
            for r in range(4):
                for c in range(4):
                    if self.board[h][r][c] == 0 and (h == 0 or self.board[h - 1][r][c] != 0):
                        self.board[h][r][c] = 1
                        move = False
                        if self.win(1):
                            move = True
                        self.board[h][r][c] = 0
                        if move:
                            return r, c
        
        return random.randint(0, 3), random.randint(0, 3)


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
        '''
        Return current state, which is a 4x4x4 bingo board, and compress it to a 1D array for the sake of simplicity
        '''
        state = np.zeros(shape=[4, 4, 4, 1, 1])
        ind = 0
        for h in range(4):
            for r in range(4):
                for c in range(4):
                    state[h][r][c][0][0] = self.bingo.board[h][r][c]

        return state

    def get_reward(self, s, flag):
        '''
        Return the reward of taking action a at state s
        '''
        return self.R(s, flag)

    def take_action(self, action):
        '''
        Take action and Return whether the action is valid, whether the player win or not, new state and the reward
        '''
        row, col = action
        flag = self.bingo.play(row, col)

        new_state = self.get_state()
        reward = self.get_reward(new_state, flag)

        return flag, new_state, reward


class Qlearning(object):

    def __init__(self, n_epoch, n_node_hidden, learning_rate, gamma, regularization_param, reward_function, decay_step, decay_rate, filter_depth, filter_height, filter_width, out_channel):

        # number of epoches
        self.n_epoch = n_epoch

        # Learning rate between 0 to 1
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(learning_rate, self.global_step, decay_step, decay_rate, staircase=True)

        # Discount factor between 0 to 1
        self.gamma = gamma
        # R(s) is the reward obtain by achieving state s
        self.reward_function = reward_function
        # Number of nodes in hidden layer
        self.n_node_hidden = n_node_hidden
        self.regularization_param = regularization_param
        
        self.MDP = MDP(reward_function)

        # Neural Network Setup

        # input layer: 1x64 vector representing the state
        self.inp = tf.placeholder(shape=[4, 4, 4, 1, 1], dtype=tf.float64)

        self.conv_layer_w = tf.cast(tf.Variable(tf.random_uniform(shape=[filter_depth, filter_height, filter_width, 1, out_channel])), tf.float64)
        self.conv_layer = tf.nn.conv3d(input=self.inp, filter=self.conv_layer_w, strides=[1, 1, 1, 1, 1], padding='SAME')

        self.conv_layer_output = tf.reshape(self.conv_layer, [1, -1])
        self.conv_layer_length = 4 * 4 * 4 * out_channel

        # Weight: 64x16 matrix
        if os.path.isfile("_weight1.txt"):

            weight1_f = open("_weight1.txt", "r")

            w1 = np.zeros(shape=[self.conv_layer_length, self.n_node_hidden])
            for i in range(self.conv_layer_length):
                for j in range(self.n_node_hidden):
                    w1[i, j] = float(weight1_f.readline())
            
            self.W1 = tf.Variable(tf.cast(w1, tf.float64))

            weight1_f.close()

        else:
            self.W1 = tf.Variable(tf.cast(tf.random_uniform([self.conv_layer_length, self.n_node_hidden], 0, 0.01), dtype=tf.float64))

        
        if os.path.isfile("_weight2.txt"):

            weight2_f = open("_weight2.txt", "r")

            w2 = np.zeros(shape=[self.n_node_hidden, 16])
            for i in range(n_node_hidden):
                for j in range(16):
                    w2[i, j] = float(weight2_f.readline())

            self.W2 = tf.Variable(tf.cast(w2, tf.float64))

            weight2_f.close()

        else:
            self.W2 = tf.Variable(tf.cast(tf.random_uniform([self.n_node_hidden, 16], 0, 0.01), dtype=tf.float64))

        
        if os.path.isfile("_bias1.txt"):
            bias1_f = open("_bias1.txt", "r")

            b1 = np.zeros(shape=[1, self.n_node_hidden])
            for i in range(self.n_node_hidden):
                b1[0, i] = float(bias1_f.readline())

            self.B1 = tf.Variable(tf.cast(b1, tf.float64))
            bias1_f.close()

        else:
            self.B1 = tf.Variable(tf.cast(tf.random_uniform([1, self.n_node_hidden], 0, 0.1), dtype=tf.float64))


        if os.path.isfile("_bias2.txt"):
            bias2_f = open("_bias2.txt", "r")

            b2 = np.zeros(shape=[1, 16])
            for i in range(16):
                b2[0, i] = float(bias2_f.readline())

            self.B2 = tf.Variable(tf.cast(b2, tf.float64))
            bias2_f.close()

        else:
            self.B2 = tf.Variable(tf.cast(tf.random_uniform([1, 16], 0, 0.1), dtype=tf.float64))


        # Output layer: 1x16 vector representing the Q-value obtained by taking the corresponding action

        self.Y1 = tf.add(tf.matmul(self.conv_layer_output, self.W1), self.B1)
        self.activate_Y1 = tf.nn.relu(self.Y1)

        self.Q = tf.add(tf.matmul(self.activate_Y1, self.W2), self.B2)

        # Q-value to update the weight
        self.Q_update = tf.placeholder(shape=[1, 16], dtype=tf.float64)

        # Cost function
        def L2_Regularization():
            return tf.nn.l2_loss(self.W1) + tf.nn.l2_loss(self.W2) + tf.nn.l2_loss(self.B1) + tf.nn.l2_loss(self.B2)

        self.loss = tf.add(tf.reduce_sum(tf.square(self.Q - self.Q_update)), self.regularization_param / self.n_epoch * L2_Regularization())
        
        # use gradient descent to optimize out model
        self.trainer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.model = self.trainer.minimize(self.loss, global_step=self.global_step)
    
    def decode_action(self, action_num):
        action = [0, 0]
        for i in range(2):
            action[i] = action_num % 4
            action_num //= 4
        return action

    def learn(self):
        '''
        Main Learning Process
        return final score, graph_x, graph_y
        '''

        init = tf.initialize_all_variables()

        reward_list = []

        graph_x = np.zeros(self.n_epoch)
        graph_y = np.zeros(self.n_epoch)

        with tf.Session() as sess:
            sess.run(init)

            percentage = 0
            win = 0

            for epoch in range(self.n_epoch):

                f.write("New Game")

                s = self.MDP.get_initial_state()
                reward = 0
                game_over = False

                while game_over is False:

                    # Compute the Q-value of current state
                    Q_val = sess.run(self.Q, feed_dict={self.inp: s})
                    Q_slice = Q_val[0, :]

                    # Sort actions with their Q-value
                    action = sorted([i for i in range(16)], key=lambda k:Q_slice[k], reverse=True)

                    flag, new_s, R = 0, 0, 0
                    valid_action = 0

                    for a in action:
                        flag, new_s, R = self.MDP.take_action(self.decode_action(a))

                        # If valid
                        if flag != -1:
                            valid_action = a
                            break
                
                    if flag == 1 or flag == 2 or flag == 3:

                        # if AI wins
                        if flag == 1:
                            win += 1

                        game_over = True

                    # Compute the Q-value of the new state
                    new_Q = sess.run(self.Q, feed_dict={self.inp: new_s})

                    # Use the max Q-value to update the Network
                    max_Q = np.max(new_Q)
                    opt_Q = Q_val
                    opt_Q[0, valid_action] = R + self.gamma * max_Q
                    
                    reward += R
                    s = new_s
                    
                    # Run the backpropogation to update the model
                    _, W1, W2 = sess.run([self.model, self.W1, self.W2], feed_dict={self.inp: s, self.Q_update: opt_Q})
                
                reward_list.append(reward)
                f.write("\n")

                graph_x[epoch] = epoch
                graph_y[epoch] = win / (epoch + 1) * 100.

                if epoch / self.n_epoch > percentage / 100:
                    print("Training complete: {}%, Winning rate: {}%".format(percentage, graph_y[epoch]))   
                    percentage += 1
    
            self.store_weight_and_bias(sess.run([self.W1, self.W2, self.B1, self.B2]))

        return reward_list, graph_x, graph_y

    def store_weight_and_bias(self, ctx):

        weight1_f = open("_weight1.txt", "w")
        weight2_f = open("_weight2.txt", "w")
        bias1_f = open("_bias1.txt", "w")
        bias2_f = open("_bias2.txt", "w")

        w1, w2, b1, b2 = ctx        

        def parse(number):
            return "%.3f" % number

        for i in range(64):
            for j in range(self.n_node_hidden):
                weight1_f.write(parse(w1[i, j]))
                weight1_f.write("\n")
        
        for i in range(self.n_node_hidden):
            for j in range(16):
                weight2_f.write(parse(w2[i, j]))
                weight2_f.write("\n")

        for i in range(self.n_node_hidden):
            bias1_f.write(parse(b1[0, i]))
            bias1_f.write("\n")

        for i in range(16):
            bias2_f.write(parse(b2[0, i]))
            bias2_f.write("\n") 

        weight1_f.close()
        weight2_f.close()
        bias1_f.close()
        bias2_f.close()


if __name__ == '__main__':

    def main():
        n_epoch = int(argv[2])
        n_node_hidden = int(argv[3])
        lr = float(argv[4])
        gamma = float(argv[5])
        regularization_param = float(argv[6])
        decay_step = int(argv[7])
        decay_rate = float(argv[8])
        filter_depth = int(argv[9])
        filter_height = int(argv[10])
        filter_width = int(argv[11])
        out_channel = int(argv[12])
        
        # Design a reward function
        def reward_function(state, flag):
            if flag == 1:
                return 1
            if flag == 2:
                return -1
            return 0

        Learner = Qlearning(n_epoch, n_node_hidden, lr, gamma, regularization_param, reward_function, decay_step, decay_rate, filter_depth, filter_height, filter_width, out_channel)
        _, graph_x, graph_y = Learner.learn()

        plt.plot(graph_x, graph_y)
        plt.show()

        f.close()

    main()
