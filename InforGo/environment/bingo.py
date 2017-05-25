"""3D-Bingo game environment model, implemented with some Markov Decision Process property"""
import numpy as np
import copy
import InforGo.environment.global_var as gv

from InforGo.util import get_pattern


class Bingo(object):

    def __init__(self, board=None):
        """Constructor
        
        Arguments:
        board -- 1. a list of board 2. a numpy array of board 3. a Bingo object
        """
        if board is None:
            self.board = [[[0 for i in range(4)] for j in range(4)] for k in range(4)]
            self.height = [[0 for i in range(4)] for j in range(4)]
            self.line_scoring = [0 for i in range(76)]
            self.player = 1
        elif type(board) != 'list' and type(board).__module__ != np.__name__:
            self.board = copy.deepcopy(board.board)
            self.height = copy.deepcopy(board.height)
            self.line_scoring = copy.deepcopy(board.line_scoring)
            self.player = board.player
        else:
            self.board = copy.deepcopy(board)
            self.height = [[0 for i in range(4)] for j in range(4)]
            cnt = 0
            for h in range(4):
                for r in range(4):
                    for c in range(4):
                        if self.board[h][r][c] != 0:
                            self.height[r][c] = h + 1
                            cnt += 1
            self.player = 1 if cnt % 2 == 0 else -1
            self.line_scoring = [0 for i in range(76)]
            for j in range(4):
                for k in range(4):
                    for i in range(4):
                        if not self.board[i][j][k]:
                            break
                        for ii in gv.scoring_index[i][j][k]:
                            self.line_scoring += self.board[i][j][k]

    def place(self, row, col):
        """place the cube for current player at given position

        Arguments:
        row -- row number of the position
        col -- column number of the position

        Returns:
        None
        """
        if not self.valid_action(row, col): return 0
        # if the position is already taken
        height = self.height[row][col]
        # place the cube
        self.board[height][row][col] = self.player
        for i in gv.scoring_index[self.height[row][col]][row][col]:
            self.line_scoring[i] -= self.player
        self.player = -self.player
        self.height[row][col] += 1
        if self.win(1): return 1
        if self.win(-1): return -1
        if self.full(): return 3
        return 0

    def full(self):
        """return whether the board is full
        
        Arguments:
        None

        Returns:
        a boolean
        """
        for r in range(4):
            for c in range(4):
                if self.height[r][c] < 4: return False
        return True

    def valid_action(self, row, col):
        """return whether the action is valid"""
        if row < 0 or row > 4 or col < 0 or col > 4: return False
        if self.height[row][col] >= 4: return False
        return True

    def win(self, player):
        """return True if player won
        
        Argument:
        player -- player to be checked, 1 for the one who player first, -1 for the opposite

        Returns:
        a boolean
        """
        if player == 0: return self.full()
        for i in range(len(self.line_scoring)):
            if self.line_scoring[i] == player * 4:
                return True
        return False

    def restart(self):
        """restart the game
        
        Arguments:
        None

        Returns:
        None
        """
        self.__init__()

    def undo_action(self, player, row, col):
        """undo the last action at given position

        Arguments:
        row -- row number of undo action
        col -- column number of undo action

        Returns:
        None
        """
        self.height[row][col] -= 1
        self.board[self.height[row][col]][row][col] = 0
        for i in range(global_var['scoring_index'][self.height[row][col]][row][col]):
            self.line_scoring[global_var['scoring_index'][self.height[row][col]][row][col][i]] -= player


    def get_state(self):
        """get current state
        
        Arguments:
        None

        Returns:
        a numpy array representing the board
        """
        return np.reshape(np.array(self.board), [4, 4, 4])
       
    def terminate(self):
        """return True if the state is terminal
        
        Arguments:
        None

        Returns:
        a boolean
        """
        return self.win(1) or self.win(-1) or self.full()

    def get_initial_state(self):
        """refresh the game and return the initial state
        
        Arguments:
        None

        Returns:
        a numpy array representing the initial state
        """
        self.__init__()
        return np.zeros(shape=[4, 4, 4])

    def get_reward(self, state, player):
        """get the reward for state s_

        Arguments:
        state -- successive state of taking action a
        player -- player who receive the reward

        Returns:
        a number in [-1, 1]
        """
        pattern = get_pattern(Bingo(state), player)
        if pattern[0, 6]: return 1
        if pattern[0, 7]: return -1
        reward = 0
        for i in range(6):
            if i % 2 == 0: reward += (i // 2 + 1) * pattern[0, i]
            else: reward -= (i // 2 + 1) * pattern[0, i]
        return reward / 10

    def take_action(self, row, col):
        """take action at given position
        
        Argument:
        row -- row number of action
        col -- column number of action

        Returns:
        flag -- the winner of the game if terminated
        new_state -- new state from taking action a in state s
        reward -- reward of taking this action
        """
        player = self.player
        origin_reward = self.get_reward(self.get_state(), player)
        flag = self.place(row, col)
        if self.win(player): return flag, self.get_state(), 1
        if self.win(-player): return flag, self.get_state(), -1
        new_state = self.get_state()
        new_reward = self.get_reward(new_state, player)
        return flag, new_state, new_reward - origin_reward

    def __getitem__(self, tup):
        """operator overload"""
        i, j, k = tup
        return self.board[i][j][k]

    def get_height(self, row, col):
        """get height for given position
        
        Arguments:
        row -- row number of position
        col -- column number of position

        Returns:
        height of (row, col)
        """
        return self.height[row][col]
