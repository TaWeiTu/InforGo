import numpy as np


class Debugger(object):

    def __init__(self, AI):
        self.AI = AI

    def check(self):
        while True:
            print("[Debugger] Enter player")
            player = int(input())
            if abs(player) > 1: break
            state = [[[0 for i in range(4)] for j in range(4)] for k in range(4)]
            for i in range(4):
                tmp = list(map(int, input().split()))
                for h in range(4):
                    for c in range(4): state[h][i][c] = tmp[h * 4 + c]
            print("[Debugger] Evaluation (player {}): {}".format(player, self.AI.evaluate(np.reshape(state, [4, 4, 4, 1, 1]), player)))
