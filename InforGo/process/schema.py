from InforGo.ai import InforGo
from InforGo.tree import minimax, mcts


class Schema(object):

    def __init__(self, n_epoch, player_len, pattern_len, n_hidden_layer, n_node_hidden, activation_fn, learning_rate,
                       directory, alpha, gamma, lamda, search_depth, c, n_playout, playout_depth, play_first, tree_type, eps):

        self.AI = InforGo(player_len, pattern_len, n_hidden_layer, n_node_hidden, activation_fn, learning_rate,
                          directory, alpha, gamma, lamda, search_depth, c, n_playout, playout_depth, play_first, tree_type, eps)
        self.n_epoch = n_epoch
