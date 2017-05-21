"""Low level model for execution process"""
from InforGo.ai import InforGo
from InforGo.tree import minimax, mcts


class Schema(object):

    def __init__(self, n_epoch, player_len, pattern_len, n_hidden_layer, n_node_hidden, activation_fn, learning_rate,
                       directory, alpha, gamma, lamda, search_depth, c, n_playout, playout_depth, play_first, tree_type, rollout_limit):
        """Constructor

        Arguments:
        n_epoch -- number of epochs in execution process
        player_len -- length of player nodes in neural network
        pattern_len -- length of pattern nodes in neural network
        n_hidden_layer -- number of hidden layer(s)
        n_node_hidden -- a list of number of nodes in each hidden layer
        activation_fn -- activation function in neural network
        learning_rate -- learning rate in neural network
        directory -- weight and bias directory for neural network
        alpha -- learning rate for TD procedure
        gamma -- discount factor for MDP
        lamda -- lambda in TD(lambda) controlling the eligibility traces
        search_depth -- maximum search depth for minimax algorithm
        c -- a constant in [0, inf] controlling the exploration/exploitation tradeoff
        n_playout -- number of playout(s) in MCTS
        playout_depth -- playout depth in MCTS
        play_first -- a boolean = whether the AI play first
        tree_type -- 'minimax'/'mcts', type of search tree
        rollout_limit -- maximum rollout depth in MCTS
        """
        self._AI = InforGo(player_len, pattern_len, n_hidden_layer, n_node_hidden, activation_fn, learning_rate,
                          directory, alpha, gamma, lamda, search_depth, c, n_playout, playout_depth, play_first, tree_type, rollout_limit)
        self._n_epoch = n_epoch
