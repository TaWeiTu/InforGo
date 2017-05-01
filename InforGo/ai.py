import InforGo
from InforGo.util import decode_action
from InforGo.environment.global_var import DEBUG
from InforGo.model.neural_net import NeuralNetwork as NN
from InforGo.tree.minimax import Minimax as minimax
from InforGo.tree.mcts import MCTS as mcts


class InforGo(object):

    def __init__(self, player_len=1, pattern_len=6, n_hidden_layer=1, n_node_hidden=[32], activation_fn='tanh', learning_rate=0.001, directory='../../Data/default/', alpha=0.1, gamma=0.99, lamda=0.85, search_depth=3, c=1, n_playout=10000, playout_depth=10, play_first=True, tree_type='minimax', eps=0.1):
        self.play_first = play_first
        self.alpha = alpha
        # Discount factor between 0 to 1
        self.gamma = gamma
        # Maximum search depth in Minimax Tree Search
        self.nn = NN(player_len, pattern_len, n_hidden_layer, n_node_hidden, activation_fn, learning_rate, directory)
        self.search_depth = search_depth
        self.eps = eps
        self.lamda = lamda
        self.playout_depth = playout_depth
        self.c = c
        player = 1 if self.play_first else -1
        self.tree = minimax(search_depth, self.nn, eps) if tree_type == 'minimax' else mcts(lamda, c, n_playout, self.nn, playout_depth, player)
        
    def get_action(self, state):
        player = 1 if self.play_first else -1
        return decode_action(self.tree.get_action(state, player))

    def refresh(self):
        tree_type = type(self.tree)
        self.tree = None
        player = 1 if self.play_first else -1
        self.tree = minimax(self.search_depth, self.nn, self.eps) if tree_type == InforGo.tree.Minimax \
                                                                  else mcts(self.lamda, self.c, self.n_playout, self.nn, self.playout_depth, player)
