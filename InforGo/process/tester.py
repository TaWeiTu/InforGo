import random
import math
import time

import InforGo
from InforGo.environment.global_var import *
from InforGo.process.schema import Schema as schema
from InforGo.environment.bingo import Bingo as State
from InforGo.util import decode_action, logger, encode_action, get_winning_move


class Bot(object):
    """Simple AI that first choose to win then prevent loss, finally play randomly"""
    def __init__(self, player):
        random.seed(7122)
        self._player = player
        self._opponent = -player

    def generate_action(self, state):
        move = get_winning_move(state, self._player)
        if len(move) > 0: return move[0][1], move[0][2]
        move = get_winning_move(state, self._opponent)
        if len(move) > 0: return move[0][1], move[0][2]
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
                         kwargs['rollout_limit'])
        self._player = 1 if kwargs['play_first'] else -1
        self._bot = Bot(-self._player)

    def test(self):
        result = {1: 0, -1: 0}
        percentage = 0
        logger.info("[Test] Testing Complete: 0%")
        t0 = time.time()
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
        return result[1], result[-1], time.time() - t0

    def _get_action(self, state, player):
        n_state = State(state)
        if player == self._player: return self._AI.get_action(n_state)
        return self._bot.generate_action(n_state)

    def _step(self, act):
        self._AI.step(act)
