"""Monte-Carlo Tree Search Implementation"""
import random
import numpy as np

from InforGo.util import decode_action, get_winning_move, encode_action
from InforGo.environment.bingo import Bingo as State


class TreeNode(object):
    """Nodes in MCTS, average value and UCT are maintained"""
    def __init__(self, parent, state, c, p):
        self._parent = parent
        self._children = {}
        self._visit = 0
        self._v = 0
        self._state = state
        self._u = p
        self._c = c
        self._p = p

    def _expand(self):
        """expand the node for all valid action"""
        actions = [i for i in range(16) if self._state.valid_action(*decode_action(i))]
        c_state = State(self._state)
        total = 0
        for i in actions: total += self._get_priority(c_state.get_height(*decode_action(i)))
        for i in actions:
            if not i in self._children.keys():
                n_state = State(self._state)
                h = n_state.get_height(*decode_action(i))
                n_state.take_action(*decode_action(i))
                p = self._get_priority(c_state.get_height(*decode_action(i))) / total
                self._children[i] = TreeNode(self, n_state, self._c, p)

    def _select(self):
        """select child with highest UCT"""
        return max(self._children.items(), key=lambda child: child[1]._get_value())[0]

    def _get_value(self):
        """compute UCT"""
        return self._v + self._u

    def _update(self, leaf_value, c):
        """update the average value and the number of visits"""
        self._visit += 1
        self._v += (leaf_value - self._v) / self._visit
        if not self.is_root():
            self._u = 2 * c * self._p * np.sqrt(2 * np.log(self._parent._visit) / self._visit)

    def _back_prop(self, leaf_value, c):
        """recursively back propagate the leaf value to the root"""
        if self._parent:
            self._parent._back_prop(leaf_value, c)
        self._update(leaf_value, c)

    def _get_priority(self, height):
        if height == 0 or height == 1: return 4
        if height == 2: return 3
        return 2

    def is_root(self):
        return self._parent is None

    def is_leaf(self):
        return len(self._children) == 0


class MCTS(object):
    """Monte-Carlo Search Tree"""
    def __init__(self, lamda, c, n_playout, evaluator, playout_depth, rollout_limit, player):
        self._lamda = lamda
        self._c = c
        self._n_playout = n_playout
        self._evaluator = evaluator
        self._root = TreeNode(None, State(np.zeros([4, 4, 4])), self._c, 1)
        self._playout_depth = playout_depth
        self._player = player
        self._rollout_limit = rollout_limit

    def step(self, last_action):
        """step to the child selected, release the reference to the previous root"""
        if not last_action in self._root._children:
            self._root._expand()
        self._root = self._root._children[last_action]
        self._root._parent = None

    def get_action(self, state, player):
        """play n_playout playouts, choose action greedily on the basis of visits"""
        for n in range(self._n_playout):
            n_state = State(state)
            self._playout(n_state)
        return max(self._root._children.items(), key=lambda child: child[1]._get_value())[0]

    def _playout(self, state):
        """walking down playout_depth step using node.select(), simulate the game result with rollout policy"""
        node = self._root
        for d in range(self._playout_depth):
            if node.is_leaf():
                if state.terminate(): break
                node._expand()
            action = node._select()
            node = node._children[action]
            state.take_action(*decode_action(action))
        # v = TD(0) z = eligibility trace
        v = self._evaluate([state.get_state()], [self._player]) if self._lamda < 1 else 0
        z = self._rollout(state) if self._lamda > 0 else 0
        leaf_value = (1 - self._lamda) * v + self._lamda * z
        node._back_prop(leaf_value, self._c)

    def _rollout(self, state):
        """play simulation with _rollout_policy"""
        c_player = state.player
        step = 0
        while not state.terminate():
            if step > self._rollout_limit: return 0
            act = self._rollout_policy(state, c_player)
            state.take_action(*decode_action(act))
            c_player *= -1
            step += 1
        if state.win(self._player): return 1 - step * 2 / self._rollout_limit
        if state.win(-self._player): return -1 + step * 2 / self._rollout_limit
        return 0

    def _evaluate(self, state, player):
        """state evluation"""
        return self._evaluator.predict(state, player)[0]

    def _rollout_policy(self, state, player):
        """randomized rollout"""
        move = get_winning_move(state, player)
        if len(move) > 0: return encode_action((move[0][1], move[0][2]))
        move = get_winning_move(state, -player)
        if len(move) > 0: return encode_action((move[0][1], move[0][2]))
        valid_action = [i for i in range(16) if state.valid_action(*decode_action(i))]
        random.shuffle(valid_action)
        return valid_action[0]
