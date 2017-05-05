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
        self._player = player
        self._opponent = -player

    def generate_action(self, state):
        for i in range(4):
            for j in range(4):
                env = State(state)
                if env.valid_action(i, j):
                    env.place(i, j)
                    if env.win(self._player): return i, j
        for i in range(4):
            for j in range(4):
                env = State(state)
                env.player *= -1
                if env.valid_action(i, j):
                    env.place(i, j)
                    if env.win(self._opponent): return i, j
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
        self._player = 1 if kwargs['play_first'] else -1
        self._bot = Bot(-self._player)

    def test(self):
        result = {1: 0, -1: 0}
        percentage = 0
        logger.info("[Test] Testing Complete: 0%")
        for epoch in range(self._n_epoch):
            state = State()
            while True:
                action = self._get_action(state, state.player)
                flag, _, R = state.take_action(*action)
                if state.terminate(): break
                self._step(encode_action(action))
            for i in [1, -1]:
                if state.win(i): result[i] += 1 
            if epoch / self._n_epoch > percentage / 100:
                percentage = math.ceil(epoch / self._n_epoch * 100)
                logger.info("[Test] Testing Complete: {}%".format(percentage))
            self._AI.refresh()
        logger.info("[Test] Testing Complete: 100%")
        return result[1], result[-1]

    def _get_action(self, state, player):
        n_state = State(state)
        if player == self._player: return self._AI.get_action(n_state)
        return self._bot.generate_action(n_state)

    def _step(self, act):
        self._AI.step(act)
