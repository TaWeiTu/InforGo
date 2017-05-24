from InforGo.util import decode_action, encode_action
from InforGo.environment.global_var import DEBUG
from InforGo.model.neural_net import NeuralNetwork as NN
from InforGo.tree.minimax import Minimax as minimax
from InforGo.tree.mcts import MCTS as mcts


class InforGo(object):

    def __init__(self, player_len=1, pattern_len=6, n_hidden_layer=1, n_node_hidden=[32], activation_fn='tanh',
                 learning_rate=0.001, directory='./Data/default/', alpha=0.1, gamma=0.99, lamda=0.85, search_depth=3,
                 c=1, n_playout=100, playout_depth=1, play_first=True, tree_type='minimax', rollout_limit=20):
        self._play_first = play_first
        self.alpha = alpha
        self.gamma = gamma
        self.nn = NN(player_len, pattern_len, n_hidden_layer, n_node_hidden, activation_fn, learning_rate, directory)
        self._search_depth = search_depth
        self._lamda = lamda
        self._playout_depth = playout_depth
        self._n_playout = n_playout
        self._c = c
        self._tree_type = tree_type
        self._rollout_limit = rollout_limit
        player = 1 if self._play_first else -1
        self._tree = minimax(search_depth, self.nn) if tree_type == 'minimax' \
                                                    else mcts(lamda, c, n_playout, self.nn, playout_depth, rollout_limit, player)
        
    def get_action(self, state):
        player = 1 if self._play_first else -1
        act = decode_action(self._tree.get_action(state, player))
        # self.step(encode_action(act))
        return act

    def refresh(self):
        self._tree = None
        player = 1 if self._play_first else -1
        self._tree = minimax(self._search_depth, self.nn) if self._tree_type == 'minimax' \
                                                          else mcts(self._lamda, self._c, self._n_playout, self.nn, self._playout_depth, self._rollout_limit, player)

    def step(self, act):
        self._tree.step(act)
