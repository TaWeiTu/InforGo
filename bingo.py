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


class MDP(object):
    
    def __init__(self, states, distribution, actions, prob, gamma, reward_function):
        self.S = states
        self.D = distribution
        self.P = prob
        self.gamma = gamma
        self.R = reward_function
        self.env = Bingo()

    def get_state(self):
        # TODO: compress bingo board to a presentable state
        pass

    def take_action(self, action):
        # TODO: return new_state, win, reward
        pass



def test():
    bingo = Bingo()
    player = 1
    win = False
    while not win:
        h, r, c = map(int, input().split())
        if not bingo.place(h, r, c):
            print("Invalid")
            continue

        win = bingo.win(player)
        if player == 1:
            player = 2
        else:
            player = 1

        bingo.plot()



if __name__ == '__main__':
    test()
