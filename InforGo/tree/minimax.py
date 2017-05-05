import numpy as np
import random

from InforGo.util import get_pattern, decode_action
from InforGo.environment.bingo import Bingo as State


class Minimax(object):

    def __init__(self, search_depth, evaluator, eps):
        self._search_depth = search_depth
        self._evaluator = evaluator
        self._eps = eps

    def get_action(self, c_state, player):
        val, action = self._search(State(c_state), player, self._search_depth)
        return action

    def _search(self, c_state, player, depth, max_level=True, alpha=-np.inf, beta=np.inf):
        if c_state.terminate():
            if c_state.win(player): return 1, None
            if c_state.win(-player): return -1, None
            return 0, None
        if depth == 0: return self._evaluate(c_state.get_state(), player), None
        comp = (lambda a, b: max(a, b)) if max_level else (lambda a, b: min(a, b))
        return_val = -np.inf if max_level else np.inf
        return_action = -1
        actions = [i for i in range(16) if c_state.valid_action(*decode_action(i))]
        random.shuffle(actions)
        for i in actions:
            n_state = State(c_state)
            r, c = decode_action(i)
            n_state.take_action(r, c)
            val, action = self._search(n_state, -player, depth - 1, not max_level, alpha, beta)
            return_val = comp(return_val, val)
            if val == return_val: return_action = i
            # if random.random() < self.eps: return val, i
            alpha, beta, prune = self._alpha_beta_pruning(return_val, max_level, alpha, beta)
            if prune: break
        # if random.uniform(0, 1) < self.eps: return actions[random.randint(0, len(actions) - 1)]
        return return_val, return_action

    def _alpha_beta_pruning(self, val, max_level, alpha, beta):
        if max_level: alpha = max(alpha, val)
        else: beta = min(beta, val)
        prune = True if alpha >= beta else False
        return alpha, beta, prune
    
    def step(self, action):
        pass

    def _evaluate(self, state, player):
        return self._evaluator.predict(state, player, get_pattern(state, player))
