import random
import numpy as np

from InforGo.util import decode_action, get_pattern, plot_state
from InforGo.environment.bingo import Bingo as State


class TreeNode(object):

    def __init__(self, parent, state):
        self._parent = parent
        self._children = {}
        self._visit = 0
        self._v = 0
        self._state = state
        self._u = 0

    def _expand(self):
        actions = [i for i in range(16) if self._state.valid_action(*decode_action(i))]
        for i in actions:
            if not i in self._children.keys():
                n_state = State(self._state)
                n_state.take_action(*decode_action(i))
                self._children[i] = TreeNode(self, n_state)

    def _select(self):
        return max(self._children.items(), key=lambda child: child[1]._get_value())[0]

    def _get_value(self):
        return self._v + self._u

    def _update(self, leaf_value, c):
        self._visit += 1
        self._v += (leaf_value - self._v) / self._visit
        if not self.is_root():
            self._u = 2 * c * np.sqrt(2 * np.log(self._parent._visit) / self._visit)

    def _back_prop(self, leaf_value, c):
        if self._parent:
            self._parent._back_prop(leaf_value, c)
        self._update(leaf_value, c)

    def is_root(self):
        return self._parent is None

    def is_leaf(self):
        return len(self._children) == 0


class MCTS(object):
    
    def __init__(self, lamda, c, n_playout, evaluator, playout_depth, player):
        self._lamda = lamda
        self._c = c
        self._n_playout = n_playout
        self._evaluator = evaluator
        self._root = TreeNode(None, State(np.zeros([4, 4, 4])))
        self._playout_depth = playout_depth
        self._player = player

    def step(self, last_action):
        if not last_action in self._root._children:
            self._root.expand()
        self._root = self._root._children[last_action]
        self._root._parent = None

    def get_action(self, state, player):
        for n in range(self._n_playout):
            n_state = State(state)
            self._playout(n_state)
        return max(self._root._children.items(), key=lambda child: child[1]._visit)[0]

    def _playout(self, state):
        node = self._root
        for d in range(self._playout_depth):
            if node.is_leaf():
                if state.terminate(): break
                node._expand()
            action = node._select()
            node = node._children[action]
            state.take_action(*decode_action(action))
        v = self._evaluate(state.get_state(), self._player) if self._lamda < 1 else 0
        z = self._rollout(state) if self._lamda > 0 else 0
        leaf_value = (1 - self._lamda) * v + self._lamda * z
        node._back_prop(leaf_value, self._c)

    def _rollout(self, state):
        while not state.terminate():
            actions = [i for i in range(16) if state.valid_action(*decode_action(i))]
            random.shuffle(actions)
            state.take_action(*decode_action(actions[0]))
        if state.win(self._player): return 1
        if state.win(-self._player): return -1
        return 0

    def _evaluate(self, state, player):
        return self._evaluator.predict(state, player, get_pattern(state, player))
