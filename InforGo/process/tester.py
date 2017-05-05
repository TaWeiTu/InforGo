import random
import math

import InforGo
from InforGo.environment.global_var import *
from InforGo.process.schema import Schema as schema
from InforGo.environment.bingo import Bingo as State
from InforGo.util import decode_action, logger, encode_action


class Bot(object):
    """Simple AI that first choose to win then prevent loss, finally play randomly"""
    def __init__(self, player):
        self.player = player
        self.opponent = -player

    def generate_action(self, state):
        for i in range(4):
            for j in range(4):
                env = State(state)
                if env.valid_action(i, j):
                    env.place(i, j)
                    if env.win(self.player): return i, j
        for i in range(4):
            for j in range(4):
                env = State(state)
                env.player *= -1
                if env.valid_action(i, j):
                    env.place(i, j)
                    if env.win(self.opponent): return i, j
        env = State(state)
        actions = [i for i in range(16) if env.valid_action(*decode_action(i))]
        random.shuffle(actions)
        return_action = decode_action(actions[random.randint(0, len(actions) - 1)])
        return return_action[0], return_action[1]


class Tester(schema):
    
    def __init__(self, **kwargs):
        super().__init__(kwargs['n_epoch'], kwargs['player_len'], kwargs['pattern_len'], kwargs['n_hidden_layer'], kwargs['n_node_hidden'],
                         kwargs['activation_fn'], kwargs['learning_rate'], kwargs['directory'], kwargs['alpha'], kwargs['gamma'], kwargs['lamda'],
                         kwargs['search_depth'], kwargs['c'], kwargs['n_playout'], kwargs['playout_depth'], kwargs['play_first'], kwargs['tree_type'],
                         kwargs['eps'])
        self.player = 1 if kwargs['play_first'] else -1
        self.bot = Bot(-self.player)

    def test(self):
        result = {1: 0, -1: 0}
        percentage = 0
        logger.info("[Test] Testing Complete: 0%")
        for epoch in range(self.n_epoch):
            state = State()
            while True:
                action = self.get_action(state, state.player)
                flag, _, R = state.take_action(*action)
                if state.terminate(): break
                self.AI.tree.step(encode_action(action))
            for i in [1, -1]:
                if state.win(i): result[i] += 1 
            if epoch / self.n_epoch > percentage / 100:
                percentage = math.ceil(epoch / self.n_epoch * 100)
                logger.info("[Test] Testing Complete: {}%".format(percentage))
            self.AI.refresh()
        logger.info("[Test] Testing Complete: 100%")
        return result[1], result[-1]

    def get_action(self, state, player):
        n_state = State(state)
        if player == self.player: return self.AI.get_action(n_state)
        return self.bot.generate_action(n_state)
