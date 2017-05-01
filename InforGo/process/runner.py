import InforGo
from InforGo.process.schema import Schema as schema
from InforGo.environment.bingo import Bingo as State
from InforGo.util import encode_action
from InforGo.util import emit_action, logger


class Runner(schema):

    def __init__(self, **kwargs):

        super().__init__(kwargs['n_epoch'], kwargs['player_len'], kwargs['pattern_len'], kwargs['n_hidden_layer'], kwargs['n_node_hidden'],
                         kwargs['activation_fn'], kwargs['learning_rate'], kwargs['directory'], kwargs['alpha'], kwargs['gamma'], kwargs['lamda'],
                         kwargs['search_depth'], kwargs['c'], kwargs['n_playout'], kwargs['playout_depth'], kwargs['play_first'], kwargs['tree_type'],
                         kwargs['eps'])
        self.player = 1 if kwargs['play_first'] else -1

    def run(self):
        state = State()
        while True:
            action = self.get_action(state, state.player)
            if type(self.AI.tree) is InforGo.tree.mcts.MCTS: self.AI.tree.step(encode_action(action))
            logger.debug("position: {} {}".format(action[0], action[1]))
            flag, s, R = state.take_action(*action)
            if flag == -state.player: break
        return -state.player

    def get_action(self, state, player):
        n_state = State(state)
        if player == self.player:
            action = self.AI.get_action(n_state)
            emit_action(action)
            return action
        return self.read_action()

    def read_action(self):
        logger.debug("input (height, row, col):")
        height, row, col = map(int, input().split())
        return row, col

