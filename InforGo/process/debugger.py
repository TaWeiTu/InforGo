"""state evaluation debugger"""
import numpy as np

from InforGo.process.schema import Schema as schema
from InforGo.util import logger, get_pattern
from InforGo.environment.bingo import Bingo as State


class Debugger(schema):

    def __init__(self, **kwargs):

        super().__init__(kwargs['n_epoch'], kwargs['player_len'], kwargs['pattern_len'], kwargs['n_hidden_layer'], kwargs['n_node_hidden'],
                         kwargs['activation_fn'], kwargs['learning_rate'], kwargs['directory'], kwargs['alpha'], kwargs['gamma'], kwargs['lamda'],
                         kwargs['search_depth'], kwargs['c'], kwargs['n_playout'], kwargs['playout_depth'], kwargs['play_first'], kwargs['tree_type'],
                         kwargs['rollout_limit'])
        self.player = 1 if kwargs['play_first'] else -1

    def debug(self):
        while True:
            confirm = input('[Debug] Continue [y/n] ')
            if confirm == 'n' or confirm == 'N': break
            board = np.zeros([4, 4, 4])
            for r in range(4):
                try: tmp_list = list(map(int, input().split()))
                except: return
                for h in range(4):
                    for c in range(4): board[r][h][c] = tmp_list[h * 4 + c]
            logger.debug("[Debug] Evaluation for player 1: {}".format(self._evaluate([board], [1])))
            logger.debug("[Debug] Evaluation for player -1: {}".format(self._evaluate([board], [-1])))

    def _evaluate(self, state, player):
        return self._AI.nn.predict(state, player)
