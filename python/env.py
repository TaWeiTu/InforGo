import numpy as np
from game import Bingo


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
