import numpy as np


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
            self.player = 1 if cnt % 2 == 0 else -1

    def place(self, row, col):
        '''
        place a cube on position(height, row, col), and return whether the operation is valid
        '''
        if not self.valid_action(row, col): return False

        # if the position is already taken
        height = self.height[row][col]

        # place the cube
        self.board[height][row][col] = self.player
        self.player = -self.player

        self.height[row][col] += 1

        return True

    def full(self):
        '''
        Return whether the board is full
        '''
        for r in range(4):
            for c in range(4):
                if self.height[r][c] < 4: return False
        return True

    def valid_action(self, row, col):
        '''
        Return whether the action is valid
        '''
        if row < 0 or row > 4 or col < 0 or col > 4: return False
        if self.height[row][col] >= 4: return False
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
        if self.full(): return 3
        if not self.place(row, col): return -1
        if self.win(player): return player
        if self.full(): return 3
        return 0

    def win(self, player):
        '''
        return True if player won, False otherwise by checking all possible winning combination
        '''
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

    def terminate(self):
        return self.win(1) or self.win(-1) or self.full()
