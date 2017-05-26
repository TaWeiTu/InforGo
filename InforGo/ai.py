"""InforGo prototype"""
from InforGo.util import decode_action, encode_action
from InforGo.environment.global_var import DEBUG
from InforGo.model.neural_net import NeuralNetwork as NN
from InforGo.tree.minimax import Minimax as minimax
from InforGo.tree.mcts import MCTS as mcts


class InforGo(object):

    def __init__(self, player_len=1, pattern_len=8, n_hidden_layer=1, n_node_hidden=[32], activation_fn='tanh',
                 learning_rate=0.1, directory='./Data/default/', alpha=0.1, gamma=0.99, lamda=0.85, search_depth=3,
                 c=1, n_playout=100, playout_depth=1, play_first=True, tree_type='minimax', rollout_limit=20):
        """Consturcor
        
        Arguments:
        player_len -- length of player node
        pattern_len -- length of pattern length
        n_hidden_layer -- number of hidden layer(s)
        n_node_hidden -- number of nodes in each hidden layer(s)
        activation_fn -- activation function
        learning_rate -- learning rate of neural network
        directory -- directory of weights and biases
        alpha -- learning rate of TD procedure
        gamma -- discount factor
        lamda -- parameter balances between TD and eligibility trace
        search_depth -- maximum search depth of minimax
        c -- parameter controlling exploration/exploitation tradeoff
        n_playout -- number of playout at each action selection
        plaout_depth -- maximum depth of playout
        play_first -- whether InforGo play first
        tree_type -- 'minimax'/'mcts', type of search tree of InforGo
        rollout_limit -- maximum step of rollout
        """
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
        """get action of InforGo given current state
        
        Arguments:
        state -- current state

        Returns:
        (row, col) denoting chosen action
        """
        player = 1 if self._play_first else -1
        act = decode_action(self._tree.get_action(state, player))
        # self.step(encode_action(act))
        return act

    def refresh(self):
        """reconstruct a new InforGo
        
        Arguments:
        None

        Returns:
        None
        """
        self._tree = None
        player = 1 if self._play_first else -1
        self._tree = minimax(self._search_depth, self.nn) if self._tree_type == 'minimax' \
                                                          else mcts(self._lamda, self._c, self._n_playout, self.nn, self._playout_depth, self._rollout_limit, player)

    def step(self, act):
        """step furthur in search tree
        
        Arguments:
        None

        Returns:
        None
        """
        self._tree.step(act)
