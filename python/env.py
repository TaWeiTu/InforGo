import numpy as np
from game import Bingo
from utils import *


class MDP(object):
    '''
    Markov Decision Process
    which can be represented as M(S, D, A, P, gamma, R)
    '''
    def __init__(self):
        # R(s, a) is the reward obtaining by taking action a at state s
        self.bingo = Bingo()

    def get_initial_state(self):
        '''
        Refresh the game and return the Initial state s0 based on D
        '''
        self.bingo.restart()
        return np.zeros(shape=[4, 4, 4, 1, 1])

    def get_state(self):
        return self.bingo.get_state()

    def get_reward(self, state, flag, player):
        tmp_bingo = Bingo(state)
        np_state = np.zeros([4, 4, 4, 1, 1])
        for h in range(4):
            for r in range(4):
                for c in range(4):
                    np_state[h][r][c][0][0] = state[h][r][c]
        pattern = get_pattern(np_state, player)
        if flag == 3: return 0
        if flag == player: return 50
        if flag != player and flag != 0: return -50
        reward = 0
        for i in range(6):
            if i % 2 == 0: reward += (i // 2 + 1) * pattern[0, i]
            else: reward -= (i // 2 + 1) * pattern[0, i]
        return reward

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
