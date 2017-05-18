import math

from InforGo.util import logger, get_pattern, decode_action, TD, encode_action
from InforGo.process.schema import Schema as schema
from InforGo.ai import InforGo
from InforGo.environment.bingo import Bingo as State


class ReinforcementTrainer(schema):
    """Reinforcement trainer, make n_epoch self-play"""
    def __init__(self, **kwargs):
        super().__init__(kwargs['n_epoch'], kwargs['player_len'], kwargs['pattern_len'], kwargs['n_hidden_layer'], kwargs['n_node_hidden'],
                         kwargs['activation_fn'], kwargs['learning_rate'], kwargs['directory'], kwargs['alpha'], kwargs['gamma'], kwargs['lamda'],
                         kwargs['search_depth'], kwargs['c'], kwargs['n_playout'], kwargs['playout_depth'], kwargs['play_first'], kwargs['tree_type'],
                         kwargs['rollout_limit'])
        self._opponent = InforGo(kwargs['player_len'], kwargs['pattern_len'], kwargs['n_hidden_layer'], kwargs['n_node_hidden'],
                         kwargs['activation_fn'], kwargs['learning_rate'], kwargs['directory'], kwargs['alpha'], kwargs['gamma'], kwargs['lamda'],
                         kwargs['search_depth'], kwargs['c'], kwargs['n_playout'], kwargs['playout_depth'], not kwargs['play_first'], 
                         kwargs['opponent_tree_type'], kwargs['rollout_limit'])

    def train(self):
        percentage = 0
        logger.info("[Reinforcement] Start Training")
        logger.info("[Reinforcement] Training Complete: 0%")
        for epoch in range(self._n_epoch):

            state = State()
            s = state.get_initial_state()
            c_player = 1
            while True:
                action = self._get_action(state, c_player)
                flag, new_s, R = state.take_action(*action)
                v = self._evaluate([s, s], [c_player, -c_player])
                new_v = self._evaluate([new_s, new_s], [c_player, -c_player])
                self._update([s, s,], [1, -1],
                             [TD(v[0], new_v[0], R, self._AI.alpha, self._AI.gamma), TD(v[1], new_v[1], -R, self._AI.alpha, self._AI.gamma)])
                self._AI.step(encode_action(action))
                self._opponent.step(encode_action(action))
                if state.terminate(): break
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
        self._AI.nn.store()

    def _get_action(self, state, player):
        ai_player = 1 if self._AI._play_first else -1
        if player == ai_player: return self._AI.get_action(state)
        else: return self._opponent.get_action(state)

    def _evaluate(self, state, player):
        return self._AI.nn.predict(state, player)

    def _update(self, state, player, value):
        self._AI.nn.update(state, player, value)
