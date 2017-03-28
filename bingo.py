import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


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

    def play(self, row, col):
        if not self.place(row, col):
            return -1
        print("AI place at ({}, {}, {})".format(self.height[row][col] - 1, row, col)) 
        if self.win(1):
            print("Player1 win")
            return 1

        self.plot()
        print("Your turn")
        r, c = map(int, input().split())
        while not self.place( r, c):
            print("Invalid Move")
            r, c = map(int, input().split())
        
        if self.win(2):
            print("Player2 win")
            return 2
        self.plot()
        return 0
        

    def plot(self):
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
        return np.zeros(shape=[1, 64])

    def get_state(self):
        '''
        Return current state, which is a 4x4x4 bingo board, and compress it to a 1D array for the sake of simplicity
        '''
        state = np.zeros(shape=[1, 64])
        ind = 0
        for h in range(4):
            for r in range(4):
                for c in range(4):
                    state[0][ind] = self.bingo.board[h][r][c]
                    ind += 1

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
    
    def __init__(self, n_epoch, learning_rate, gamma, reward_function):

        # number of epoches
        self.n_epoch = n_epoch
        # Learning rate between 0 to 1
        self.lr = learning_rate
        # Discount factor between 0 to 1
        self.gamma = gamma
        # R(s) is the reward obtain by achieving state s
        self.reward_function = reward_function
        
        self.MDP = MDP(reward_function)

        # Neuron Network Setup
        self.inp = tf.placeholder(shape=[1,64], dtype=tf.float32)
        self.W = tf.Variable(tf.random_uniform([64, 16], 0, 0.1))
        self.Q = tf.matmul(self.inp, self.W)
        self.predict = tf.argmax(self.Q, 1)

        self.Q_update = tf.placeholder(shape=[1,16], dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(self.Q - self.Q_update))
        self.trainer = tf.train.GradientDescentOptimizer(self.lr)
        self.model = self.trainer.minimize(self.loss)
    
    def decode_action(self, action_num):
        action = [0, 0]
        for i in range(2):
            action[i] = action_num % 4
            action_num //= 4
        return action

    def learn(self):
        
        init = tf.global_variables_initializer()
        reward_list = []

        with tf.Session() as sess:
            sess.run(init)
            for e in range(self.n_epoch):
                s = self.MDP.get_initial_state()
                reward = 0
                game_over = False
                fail = False

                while game_over is False:
                    if not fail:
                        print(sess.run(self.W))
                    a, Q_val = sess.run([self.predict, self.Q], feed_dict={self.inp: s})
                    flag, new_s, R = self.MDP.take_action(self.decode_action(a[0]))
                

                    if flag == -1:

                        fix_Q = Q_val
                        fix_Q[0, a[0]] = -100

                        sess.run(self.model, feed_dict={self.inp: s, self.Q_update: fix_Q})

                        if not fail:
                            print("Invalid")
                        fail = True
                        continue

                    if flag == 1 or flag == 2:
                        print("GameOver")
                        game_over = True

                    new_Q = sess.run(self.Q, feed_dict={self.inp: new_s})
                    max_Q = np.max(new_Q)
                    print("maxQ = {}".format(max_Q))
                    opt_Q = Q_val
                    opt_Q[0, a[0]] = R + self.gamma * max_Q
                    
                    reward += R
                    s = new_s

                    _, new_W = sess.run([self.model, self.W], feed_dict={self.inp: s, self.Q_update: opt_Q})
                
                reward_list.append(reward)
    
        return reward_list


if __name__ == '__main__':

    def main():
        print("n_epoch, lr, gamma:")
        n_epoch = int(input())
        lr = float(input())
        gamma = float(input())

        def reward_function(state, flag):
            if flag == 1:
                return 1
            if flag == 2:
                return -1
            return 0

        Learner = Qlearning(n_epoch, lr, gamma, reward_function)
        Learner.learn()

    main()
