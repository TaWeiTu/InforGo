import random
from game import Bingo


class Bot(object):

    def __init__(self, player):
        self.player = player
        self.opponent = -player

    def generate_action(self, state):
        bingo = Bingo(state)
        for i in range(4):
            for j in range(4):
                if bingo.valid_action(i, j):
                    bingo.place(i, j)
                    if bingo.win(self.player): return i, j
                    bingo.undo_action(i, j)

        bingo.player = -bingo.player
        for i in range(4):
            for j in range(4):
                if bingo.valid_action(i, j):
                    bingo.place(i, j)
                    if bingo.win(self.opponent): return i, j
                    bingo.undo_action(i, j)
        return random.randint(0, 3), random.randint(0, 3)
