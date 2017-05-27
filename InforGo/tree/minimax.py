"""Minimax algorithm implementation with alpha-beta pruning"""
import random
import numpy as np

from InforGo.util import get_pattern, decode_action
from InforGo.environment.bingo import Bingo as State


class Minimax(object):
    """Minimax search tree"""
    def __init__(self, search_depth, evaluator):
        self._search_depth = search_depth
        self._evaluator = evaluator

    def get_action(self, c_state, player):
        """get action which minimize opponent's maximum reward

        Arguments:
        c_state -- a copy of current state
        player -- current player

        Returns:
        a number in [1, 16] denoting the selected action
        """
        val, action = self._search(State(c_state), player, self._search_depth)
        return action

    def _search(self, c_state, player, depth, max_level=True, alpha=-np.inf, beta=np.inf):
        """recursively search every possible action and evaluate the leaf nodes

        Arguments:
        c_state -- a copy of current state
        player -- current player
        depth -- depth left for furthur searching
        max_level -- whether the level is maximizer
        alpha -- best value along the path from root to maximizer
        beta -- worst value along the path from root to minimizer

        Returns:
        return_val -- value of this node
        return_action -- where return_val comes from
        """
        if c_state.terminate():
            if c_state.win(player): return 1, None
            if c_state.win(-player): return -1, None
            return 0, None
        if depth == 0: return self._evaluate([c_state.get_state()], [player]), None
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
            alpha, beta, prune = self._alpha_beta_pruning(return_val, max_level, alpha, beta)
            if prune: break
        return return_val, return_action

    def _alpha_beta_pruning(self, val, max_level, alpha, beta):
        """update alpha and beta and decide to prune or not

        Arguments:
        val -- value of evaluated node
        max_level -- whether this level is maximizer
        alpha -- best value along the path from root to maximizer
        beta -- worst value along the path from root to minimizer

        Returns:
        alpha -- new alpha
        beta -- new beta
        prune -- to prune or not
        """
        if max_level: alpha = max(alpha, val)
        else: beta = min(beta, val)
        prune = True if alpha >= beta else False
        return alpha, beta, prune
    
    def step(self, action):
        """just for simplicity"""
        pass

    def _evaluate(self, state, player):
        """state evalutation
        
        Arguments:
        state -- current state
        player -- current player

        Returs:
        evaluation of current state for current player
        """
        return self._evaluator.predict(state, player)
