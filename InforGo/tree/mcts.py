import random
import numpy as np

from InforGo.util import decode_action, get_pattern, plot_state
from InforGo.environment.bingo import Bingo as State


class TreeNode(object):

    def __init__(self, parent, state):
        self.parent = parent
        self.children = {}
        self.visit = 0
        self.v = 0
        self.state = state
        self.u = 0

    def expand(self):
        actions = [i for i in range(16) if self.state.valid_action(*decode_action(i))]
        for i in actions:
            if not i in self.children.keys():
                n_state = State(self.state)
                n_state.take_action(*decode_action(i))
                self.children[i] = TreeNode(self, n_state)

    def select(self):
        return max(self.children.items(), key=lambda child: child[1].get_value())[0]

    def get_value(self):
        return self.v + self.u

    def update(self, leaf_value, c):
        self.visit += 1
        self.v += (leaf_value - self.v) / self.visit
        if not self.is_root():
            self.u = 2 * c * np.sqrt(2 * np.log(self.parent.visit) / self.visit)

    def back_prop(self, leaf_value, c):
        if self.parent:
            self.parent.back_prop(leaf_value, c)
        self.update(leaf_value, c)

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return len(self.children) == 0

class MCTS(object):
    
    def __init__(self, lamda, c, n_playout, evaluator, playout_depth, player):
        self.lamda = lamda
        self.c = c
        self.n_playout = n_playout
        self.evaluator = evaluator
        self.root = TreeNode(None, State(np.zeros([4, 4, 4])))
        self.playout_depth = playout_depth
        self.player = player

    def step(self, last_action):
        self.root = self.root.children[last_action]
        self.root.parent = None

    def get_action(self, state, player):
        for n in range(self.n_playout):
            n_state = State(state)
            self.playout(n_state)
        return max(self.root.children.items(), key=lambda child: child[1].visit)[0]

    def playout(self, state):
        node = self.root
        for d in range(self.playout_depth):
            if node.is_leaf():
                if state.terminate(): break
                node.expand()
            action = node.select()
            node = node.children[action]
            state.take_action(*decode_action(action))
        v = self.evaluator.predict(state.get_state(), self.player, get_pattern(state, self.player)) if self.lamda < 1 else 0
        z = self.rollout(state) if self.lamda > 0 else 0
        leaf_value = (1 - self.lamda) * v + self.lamda * z
        node.back_prop(leaf_value, self.c)

    def rollout(self, state):
        while not state.terminate():
            actions = [i for i in range(16) if state.valid_action(*decode_action(i))]
            random.shuffle(actions)
            state.take_action(*decode_action(actions[0]))
        if state.win(self.player): return 1
        if state.win(-self.player): return -1;
        return 0
        



