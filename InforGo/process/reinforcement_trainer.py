"""Reinforcement training with respect to TD(0) procedure"""
import math

from InforGo.util import logger, get_pattern, decode_action, TD, encode_action
from InforGo.process.schema import Schema as schema
from InforGo.ai import InforGo
from InforGo.environment.bingo import Bingo as State


class ReinforcementTrainer(schema):
    
    def __init__(self, **kwargs):
        """Constructor

        Arguments:
        kwargs -- command line execution arguments
        """
        super().__init__(kwargs['n_epoch'], kwargs['player_len'], kwargs['pattern_len'], kwargs['n_hidden_layer'], kwargs['n_node_hidden'],
                         kwargs['activation_fn'], kwargs['learning_rate'], kwargs['directory'], kwargs['alpha'], kwargs['gamma'], kwargs['lamda'],
                         kwargs['search_depth'], kwargs['c'], kwargs['n_playout'], kwargs['playout_depth'], kwargs['play_first'], kwargs['tree_type'],
                         kwargs['rollout_limit'])
        self._opponent = InforGo(kwargs['player_len'], kwargs['pattern_len'], kwargs['n_hidden_layer'], kwargs['n_node_hidden'],
                         kwargs['activation_fn'], kwargs['learning_rate'], kwargs['directory'], kwargs['alpha'], kwargs['gamma'], kwargs['lamda'],
                         kwargs['search_depth'], kwargs['c'], kwargs['n_playout'], kwargs['playout_depth'], not kwargs['play_first'], 
                         kwargs['opponent_tree_type'], kwargs['rollout_limit'])

    def train(self):
        """reinforcement training process
        
        Arguments:
        None

        Returns:
        None
        """
        percentage = 0
        logger.info("[Reinforcement] Start Training")
        logger.info("[Reinforcement] Training Complete: 0%")
        for epoch in range(self._n_epoch):
            state = [State() for _ in range(4)]
            s = [state[_].get_initial_state() for _ in range(4)]
            c_player = 1
            while True:
                action = self._get_action(state[0], c_player)
                flag, new_s, R = zip(*[state[i].take_action(*(self._rotate(*action, i))) for i in range(4)])
                v = self._evaluate(s + s, [c_player for i in range(4)] + [-c_player for i in range(4)])
                new_v = self._evaluate(new_s + new_s, [c_player for i in range(4)] + [-c_player for i in range(4)])
                self._update(*self._concat_training_data(s, v, new_v, R, c_player))
                self._AI.step(encode_action(action))
                self._opponent.step(encode_action(action))
                if state[0].terminate(): break
                s = new_s
                c_player *= -1
            if epoch / self._n_epoch > percentage / 100:
                percentage = math.ceil(epoch / self._n_epoch * 100)
                logger.info('[Reinforcement] Training Complete: {}%'.format(percentage))
            if percentage % 10 == 0: self._store()
            self._AI.refresh()
            self._opponent.refresh()
        logger.debug('[Reinforcement] Training Complete: 100%')
        self._store()

    def _store(self):
        """store weights and biases
        
        Arguments:
        None

        Returns:
        None
        """
        self._AI.nn.store()

    def _get_action(self, state, player):
        """get action for current player
        
        Arguments:
        state -- current state
        player -- current player

        Returns:
        a (row, col) pair denoting the action
        """
        ai_player = 1 if self._AI._play_first else -1
        if player == ai_player: return self._AI.get_action(state)
        else: return self._opponent.get_action(state)

    def _evaluate(self, state, player):
        """state evaluation
        
        Arguments:
        state -- a list of states to be evaluated
        player -- a list of players to be evaluated

        Returns:
        a list of value for each (state, player) pair
        """
        return self._AI.nn.predict(state, player)

    def _update(self, state, player, value):
        """update the neural network with backpropagation
        
        Arguments:
        state -- a list of input states 
        player -- a list of input players
        value -- a list of true value for each (state, player) pair

        Returns:
        None
        """
        self._AI.nn.update(state, player, value)

    def _rotate(self, row, col, t):
        """action rotation
        
        Arguments:
        row -- row number of action
        col -- column number of action
        t -- rotation degree = 90 * t

        Returns:
        rotated action
        """
        for i in range(t):
            row, col = col, row
            col = 3 - col
        return row, col
    
    def _concat_training_data(self, s, v, new_v, R, c_player):
        """calculate TD(0) return for given state and value
        
        Arguments:
        s -- a list of states
        v -- a list of value of current states
        new_v -- a list of value of successive states
        R -- a list of rewards
        c_player -- a list of current players

        Returns:
        input_data -- a list of input states for neural network
        player -- a list of input players for neural network
        output -- a list of TD(0) value for input states and players
        """
        input_data = s + s
        player = [c_player] * 4 + [-c_player] * 4
        output = [0 for i in range(8)]
        for i in range(8):
            coef = 1 if i // 4 == 0 else -1
            output[i] = TD(v[i % 4], new_v[i % 4], R[i % 4] * coef, self._AI.alpha, self._AI.gamma)
        return input_data, player, output