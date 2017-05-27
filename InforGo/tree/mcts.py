"""Monte-Carlo Tree Search Implementation"""
import random
import numpy as np

from InforGo.util import decode_action, get_winning_move, encode_action, logger
from InforGo.environment.bingo import Bingo as State


class TreeNode(object):
    """Nodes in Monte-Carlo tree search, average value and UCT are maintained"""
    def __init__(self, parent, p):
        """Constructor

        Arguments:
        parent -- reference to the parent TreeNode
        state -- state that this node represents
        p -- probabilty over choosing the node to other siblings
        """
        self._parent = parent
        self._children = {}
        self._visit = 0
        self._v = 0
        self._u = np.inf
        self._p = p

    def _expand(self, act_prob):
        """expand the node for all valid action
        
        Arguments:
        None

        Returns:
        None
        """
        for (act, prob) in act_prob:
            if not act in self._children:
                self._children[act] = TreeNode(self, prob)

    def _select(self, c):
        """select child with highest UCT
        
        Arguments:
        None

        Returns:
        a number in [1, 16] denoting the selected action
        """
        return max(self._children.items(), key=lambda child: child[1]._get_value(c))[0]

    def _get_value(self, c):
        """compute UCT
        
        Arguments:
        None

        Returns:
        UCT value of this node
        """
        if not self.is_root() and self._visit:
            self._u = 2 * c * self._p * np.sqrt(2 * np.log(self._parent._visit) / self._visit)
        return self._v + self._u

    def _update(self, leaf_value, c):
        """update the average value and the number of visits
        
        Arguments:
        leaf_value -- state evaluation combined with game simulation result
        c -- parameter in [0, inf] controlling exploration/exploitation tradeoff

        Returns:
        None
        """
        self._visit += 1
        self._v += (leaf_value - self._v) / self._visit
        if not self.is_root() and self._visit:
            self._u = 2 * c * self._p * np.sqrt(2 * np.log(self._parent._visit) / self._visit)

    def _back_prop(self, leaf_value, c):
        """recursively back propagate the leaf value to the root
        
        Arguments:
        leaf_value -- state evaluation combined with game simulation result
        c -- parameter in [0, inf] controlling exploration/exploitation tradeoff

        Returns:
        None
        """
        if self._parent:
            self._parent._back_prop(leaf_value, c)
        self._update(leaf_value, c)

    def is_root(self):
        """return whether this node is root"""
        return self._parent is None

    def is_leaf(self):
        """return whether this node is leaf"""
        return len(self._children) == 0


class MCTS(object):
    """Monte-Carlo Search Tree"""
    def __init__(self, lamda, c, n_playout, evaluator, playout_depth, rollout_limit, player):
        self._lamda = lamda
        self._c = c
        self._n_playout = n_playout
        self._evaluator = evaluator
        self._root = TreeNode(None, 1.0)
        self._playout_depth = playout_depth
        self._player = player
        self._rollout_limit = rollout_limit
        self._step = 0

    def step(self, last_action):
        """step to the child selected, release the reference to the previous root"""
        if last_action in self._root._children:
            self._root = self._root._children[last_action]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)
        self._step += 1
        self._decay_rollout_limit()

    def get_action(self, state, player):
        """play n_playout playouts, choose action greedily on the basis of visits"""
        for n in range(self._n_playout):
            if n % 100 == 0: logger.debug("playout: {}".format(n))
            n_state = State(state)
            self._playout(n_state)
        for _id, node in self._root._children.items():
            logger.debug("id = {}, value = {}".format(_id, node._get_value(self._c)))
        return max(self._root._children.items(), key=lambda child: child[1]._get_value(self._c))[0]

    def _playout(self, state):
        """walking down playout_depth step using node.select(), simulate the game result with rollout policy"""
        node = self._root
        for d in range(self._playout_depth):
            if node.is_leaf():
                if state.terminate(): break
                valid_action = [i for i in range(16) if state.valid_action(*decode_action(i))]
                height_total = 0
                for i in valid_action: height_total += self._get_priority(state.get_height(*decode_action(i)))
                act_prob = [(act, self._get_priority(state.get_height(*decode_action(act))) / height_total) for act in valid_action]
                node._expand(act_prob)
            action = node._select(self._c)
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
            # logger.debug("action: {}".format(act))
            state.take_action(*decode_action(act))
            c_player *= -1
            step += 1
        if state.win(self._player): return 1
        if state.win(-self._player): return -1
        return 0

    def _evaluate(self, state, player):
        """state evluation"""
        return self._evaluator.predict(state, player)[0]

    def _rollout_policy(self, state, player):
        """randomized rollout
        
        Arguments:
        state -- current state
        player -- current player

        Returns:
        action chosen by rollout policy
        """
        move = get_winning_move(state, player)
        if len(move) > 0: return encode_action((move[0][1], move[0][2]))
        move = get_winning_move(state, -player)
        if len(move) > 0: return encode_action((move[0][1], move[0][2]))
        valid_action = [i for i in range(16) if state.valid_action(*decode_action(i))]
        return random.choice(valid_action)

    def _get_priority(self, height):
        """compute priority of height of action

        Arguments:
        height -- height of action

        Returns:
        priority of input action
        """
        if height == 0 or height == 1: return 4
        if height == 2: return 3
        return 2

    def _decay_rollout_limit(self):
        if self._step % 5 == 0:
            self._rollout_limit -= 3
            if self._rollout_limit < 3: self._rollout_limit = 3
