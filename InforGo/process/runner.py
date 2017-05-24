"""AI execution for online gaming platform, https://bingo.infor.org"""
from InforGo.process.schema import Schema as schema
from InforGo.environment.bingo import Bingo as State
from InforGo.util import encode_action, TD, emit_action, logger


class Runner(schema):

    def __init__(self, **kwargs):
        """Constructor

        Arguments:
        kwargs -- command line execution arguments
        """
        super().__init__(kwargs['n_epoch'], kwargs['player_len'], kwargs['pattern_len'], kwargs['n_hidden_layer'], kwargs['n_node_hidden'],
                         kwargs['activation_fn'], kwargs['learning_rate'], kwargs['directory'], kwargs['alpha'], kwargs['gamma'], kwargs['lamda'],
                         kwargs['search_depth'], kwargs['c'], kwargs['n_playout'], kwargs['playout_depth'], kwargs['play_first'], kwargs['tree_type'],
                         kwargs['rollout_limit'])
        self.player = 1 if kwargs['play_first'] else -1

    def run(self):
        """Game playing process
        for online gaming, the process will be spawn using child_process in Node.js
        communication is done with web socket

        Arguments:
        None

        Returns:
        None
        """
        state = State()
        s = state.get_initial_state()
        c_player = 1
        while True:
            action = self.get_action(state, state.player)
            self._AI._tree.step(encode_action(action))
            logger.debug("position: {} {}".format(action[0], action[1]))
            flag, new_s, R = state.take_action(*action)
            v = self._evaluate([s, s], [c_player, -c_player])
            new_v = self._evaluate([new_s, new_s], [c_player, -c_player])
            self._update([s, s,], [1, -1],
                         [TD(v[0], new_v[0], R, self._AI.alpha, self._AI.gamma), TD(v[1], new_v[1], -R, self._AI.alpha, self._AI.gamma)])
            if state.terminate(): break
            s = new_s
            c_player *= -1
        for i in [-1, 0, 1]:
            if state.win(i): return i

    def get_action(self, state, player):
        """get action, return AI's action if it's AI's turn, read input from stdin otherwise

        Arguments:
        state -- current state
        player -- current player

        Returns:
        a (row, col) pair denoting action
        """
        n_state = State(state)
        if player == self.player:
            action = self._AI.get_action(n_state)
            emit_action(action)
            return action
        return self.read_action()

    def read_action(self):
        """read opponent's action from standard input

        Arguments:
        None

        Returns:
        a (row, col) pair denoting the action
        """
        logger.debug("input (height, row, col):")
        _, row, col = map(int, input().split())
        return row, col

    def _update(self, state, player, value):
        """update the neural network with backpropagation

        Arguments:
        states -- a list of input states
        players -- a list of input players
        v_ -- a list of true value for each (state, player) pair

        Returns:
        None
        """
        self._AI.nn.update(state, player, value)

    def _evaluate(self, state, player):
        """state evaluation
 
        Arguments:
        state -- a list of states to be evaluated
        player -- a list of players to be evaluated

        Returns:
        a list of value for each (state, player) pair
        """
        return self._AI.nn.predict(state, player)