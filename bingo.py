import numpy
import tensorflow
import matplotlib.pyplot as plt


class Bingo(object):

    def __init__(self):
        self.board = [[[0 for i in range(4)] for j in range(4)] for k in range(4)]

        # player = 1 for the player who play first and player = 2 for the opposite
        self.player = 1

    def place(self, height, row, col):
        '''
        place a cube on position(height, row, col), and return whether the operation is valid
        '''
        # if the position is already taken
        if self.board[height][row][col] != 0:
            return False

        # if there's no placed cube underneath out position
        if height > 0 and self.board[height - 1][row][col] == 0:
            return False

        # place the cube
        self.board[height][row][col] = self.player
        self.change_player()

        return True

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
    def __init__(self, distribution, prob, reward_function):

        # D is a probability distribution of the initial state s0
        self.D = distribution
        
        # P(s, a, s1) is the probablity of state s becoming state s1 by taking action a
        self.P = prob

        # R(s, a) is the reward obtaining by taking action a at state s
        self.R = reward_function

        self.bingo = Bingo()

    def get_initial_state(self):
        '''
        Refresh the game and return the Initial state s0 based on D
        '''
        self.bingo.retart()

        # TODO: return initial state based on the probability distribution
        pass

    def get_state(self):
        '''
        Return current state, which is a 4x4x4 bingo board, and compress it to a 1D array for the sake of simplicity
        '''
        state = []
        for h in range(4):
            for r in range(4):
                for c in range(4):
                    state.append(self.bingo.board[h][r][c])

        return state

    def get_reward(self, s, a):
        '''
        Return the reward of taking action a at state s
        '''
        return self.R(s, a)

    def take_action(self, action):
        '''
        Take action and Return whether the action is valid, whether the player win or not, new state and the reward
        '''
        current_player = self.bingo.player
        height, row, col = action
        valid_flag = self.bingo.place(height, row, col)
        win = self.bingo.win(current_player)
        new_state = self.get_state()
        reward = self.get_reward(new_state)

        return valid_flag, win, new_state, reward


class QLearning(object):
    '''
    A QLearning model using Dynamic Programming
    '''
    def __init__(self, n_epoch, lr, gamma, reward_function, state_size, action_size):

        # n_epoch = the number of epoches
        self.n_epoch = n_epoch

        # lr = learning rate
        self.lr = lr

        # gamma = discount factor of MDP
        self.gamma = gamma

        # In bingo game, the probablity of initial state being the board clear is 1
        self.MDP = MDP([[0 for i in range(64)]: 1], [], reward_function)

        # Initial Q-value table
        self.Q = np.zeros([state_size, action_size])

        # size of the state space and the size of the action space
        self.state_size = state_size
        self.action_size = action_size

    def learn(self):
        '''
        The main learning process
        '''
        # store the score obtain by every epoches
        score = []

        for e in range(self.n_epoch):

            # get initial state from MDP
            s = self.MDP.get_initial_state()
            reward = 0
            
            while True:

                # greediy take action according to the Q-value table
                a = np.argmax(Q[s, :]) + np.random.randn(1, self.action_size) * (1. / (e + 1))
                valid_flag, win, s_prime, R = self.MDP.take_action(a)

                # update the Q-value table based on Bellman Equation
                Q[s, a] = Q[s, a] + lr * (R + self.gamma * np.max(Q[s_prime, :]) - Q[s, a])

                # Change the state and add the reward
                s = s_prime
                reward += R

                if win:
                    break

            score.append(reward)

        return score


def main():
    


if __name__ == '__main__':
    main()
