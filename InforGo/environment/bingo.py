import numpy as np

from InforGo.util import get_pattern


class Bingo(object):

    def __init__(self, board=None):
        """
        if board is empty: return empty board
        if board is a list or numpy array: return the same board
        if board is another board: return a copy of it
        """
        if board is None:
            self.board = [[[0 for i in range(4)] for j in range(4)] for k in range(4)]
            self.height = [[0 for i in range(4)] for j in range(4)]
            self.player = 1
        elif type(board) != 'list' and type(board).__module__ != np.__name__:
            self.board = [[[0 for i in range(4)] for j in range(4)] for k in range(4)]
            self.height = [[0 for i in range(4)] for j in range(4)]
            for i in range(4):
                for j in range(4):
                    for k in range(4): self.board[i][j][k] = board.board[i][j][k]
            for i in range(4):
                for j in range(4): self.height[i][j] = board.height[i][j]
            self.player = board.player
        else:
            self.board = [[[0 for i in range(4)] for j in range(4)] for k in range(4)]
            for h in range(4):
                for r in range(4):
                    for c in range(4): self.board[h][r][c] = board[h][r][c]
            self.height = [[0 for i in range(4)] for j in range(4)]
            cnt = 0
            for h in range(4):
                for r in range(4):
                    for c in range(4):
                        if self.board[h][r][c] != 0:
                            self.height[r][c] = h + 1
                            cnt += 1
            self.player = 1 if cnt % 2 == 0 else -1

    def place(self, row, col):
        """place a cube on position(height, row, col), and return whether the operation is valid"""
        if not self.valid_action(row, col): return 0
        # if the position is already taken
        height = self.height[row][col]
        # place the cube
        self.board[height][row][col] = self.player
        self.player = -self.player
        self.height[row][col] += 1
        if self.win(1): return 1
        if self.win(-1): return -1
        if self.full(): return 3
        return 0

    def full(self):
        """Return whether the board is full"""
        for r in range(4):
            for c in range(4):
                if self.height[r][c] < 4: return False
        return True

    def valid_action(self, row, col):
        """Return whether the action is valid"""
        if row < 0 or row > 4 or col < 0 or col > 4: return False
        if self.height[row][col] >= 4: return False
        return True

    def win(self, player):
        """return True if player won"""
        for h in range(4):
            for r in range(4):
                flag = True
                for c in range(4): flag = False if self.board[h][r][c] != player else flag
                if flag: return True

        for h in range(4):
            for c in range(4):
                flag = True
                for r in range(4): flag = False if self.board[h][r][c] != player else flag
                if flag: return True

        for r in range(4):
            for c in range(4):
                flag = True
                for h in range(4): flag = False if self.board[h][r][c] != player else flag
                if flag: return True

        for h in range(4):
            flag = True
            for i in range(4): flag = False if self.board[h][i][i] != player else flag
            if flag: return True
            flag = True
            for i in range(4): flag = False if self.board[h][i][3 - i] != player else flag
            if flag: return True

        for r in range(4):
            flag = True
            for i in range(4): flag = False if self.board[i][r][i] != player else flag
            if flag: return True
            flag = True
            for i in range(4): flag = False if self.board[i][r][3 - i] != player else flag
            if flag: return True

        for c in range(4):
            flag = True
            for i in range(4): flag = False if self.board[i][i][c] != player else flag
            if flag: return True
            flag = True
            for i in range(4): flag = False if self.board[i][3 - i][c] != player else flag
            if flag: return True

        flag = True
        for i in range(4): flag = False if self.board[i][i][i] != player else flag
        if flag: return True
        flag = True
        for i in range(4): flag = False if self.board[i][i][3 - i] != player else flag
        if flag: return True
        flag = True
        for i in range(4): flag = False if self.board[i][3 - i][i] != player else flag
        if flag: return True
        flag = True
        for i in range(4): flag = False if self.board[3 - i][i][i] != player else flag
        if flag: return True
        if player == 0: return self.full()
        return False

    def restart(self):
        """restart the game"""
        self.__init__()

    def undo_action(self, row, col):
        """Undo the last action at (row, col)"""
        self.height[row][col] -= 1
        self.board[self.height[row][col]][row][col] = 0

    def get_state(self):
        """Get current State"""
        return np.reshape(np.array(self.board), [4, 4, 4])
       
    def terminate(self):
        """Return True if the state is terminal"""
        return self.win(1) or self.win(-1) or self.full()

    def get_initial_state(self):
        """Refresh the game and return the initial state"""
        self.__init__()
        return np.zeros(shape=[4, 4, 4])

    def get_reward(self, state, player):
        """
        if player win: return 50
        if opponent win: return -50
        else return pattern: corner * 1 + two * 2 + three * 3
        """
        pattern = get_pattern(Bingo(state), player)
        tmp_state = Bingo(state)
        if tmp_state.win(player): return 1
        if tmp_state.win(-player): return -1
        reward = 0
        for i in range(6):
            if i % 2 == 0: reward += (i // 2 + 1) * pattern[0, i]
            else: reward -= (i // 2 + 1) * pattern[0, i]
        return reward / 10

    def take_action(self, row, col):
        player = self.player
        """Take action and Return whether the action is valid, whether the player win or not, new state and the reward"""
        origin_reward = self.get_reward(self.get_state(), player)
        flag = self.place(row, col)
        if self.win(player): return flag, self.get_state(), 1
        if self.win(player): return flag, self.get_state(), -1
        new_state = self.get_state()
        new_reward = self.get_reward(new_state, player)
        return flag, new_state, new_reward - origin_reward

    def __getitem__(self, tup):
        """operator overload"""
        i, j, k = tup
        return self.board[i][j][k]

    def get_height(self, row, col):
        """return height of (row, col), plus 1 to avoid ZeroDevisionError"""
        return self.height[row][col]
