import numpy
import tensorflow
import matplotlib.pyplot as plt


class Bingo(object):

    def __init__(self):
        self.board = [[[0 for i in range(4)] for j in range(4)] for k in range(4)]
        self.player = 1


    def place(self, height, row, col):
        if self.board[height][row][col] != 0:
            return False
        if height > 0 and self.board[height - 1][row][col] == 0:
            return False
        self.board[height][row][col] = self.player
        self.change_player()
        return True

    def change_player(self):
        if self.player == 1:
            self.player = 2
        else:
            self.player = 1

    def win(self, player):
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

    def refresh(self):
        self.__init__()


class MDP(object):
    
    def __init__(self, distribution, prob, reward_function):
        self.D = distribution
        self.P = prob
        self.R = reward_function
        self.bingo = Bingo()

    def get_initial_state(self):
        self.bingo.refresh()
        # TODO: return initial state based on the probability distribution
        pass

    def get_state(self):
        # TODO: compress bingo board to a presentable state
        state = []
        for h in range(4):
            for r in range(4):
                for c in range(4):
                    state.append(self.bingo.board[h][r][c])
        return state

    def get_reward(self, state):
        return self.R(state)

    def take_action(self, action):
        current_player = self.bingo.player
        height, row, col = action
        valid_flag = self.bingo.place(height, row, col)
        win = self.bingo.win(current_player)
        new_state = self.get_state()
        reward = self.get_reward(new_state)

        return valid_flag, win, new_state, reward


class QLearning(object):
    
    def __init__(self, n_epoch, lr, gamma, reward_function, state_size, action_size):
        self.n_epoch = n_epoch
        self.lr = lr
        self.gamma = gamma
        self.MDP = MDP([[0 for i in range(64)]: 1], [], reward_function)
        self.Q = np.zeros([state_size, action_size])
        self.state_size = state_size
        self.action_size = action_size

    def learn(self):
        score = []
        for e in range(self.n_epoch):
            s = self.MDP.get_initial_state()
            reward = 0
            while True:
                a = np.argmax(Q[s, :]) + np.random.randn(1, self.action_size) * (1. / (e + 1))
                valid_flag, win, s_prime, R = self.MDP.take_action(a)
                Q[s, a] = Q[s, a] + lr * (R + self.gamma * np.max(Q[s_prime, :]) - Q[s, a])
                s = s_prime
                reward += R
                if win:
                    break
            score.append(reward)

        return score


def main():
    pass


if __name__ == '__main__':
    main()
