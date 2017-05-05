import math

from InforGo.util import logger, get_pattern, decode_action, TD, encode_action
from InforGo.process.schema import Schema as schema
from InforGo.ai import InforGo
from InforGo.environment.bingo import Bingo as State


class ReinforcementTrainer(schema):

    def __init__(self, **kwargs):
        super().__init__(kwargs['n_epoch'], kwargs['player_len'], kwargs['pattern_len'], kwargs['n_hidden_layer'], kwargs['n_node_hidden'],
                         kwargs['activation_fn'], kwargs['learning_rate'], kwargs['directory'], kwargs['alpha'], kwargs['gamma'], kwargs['lamda'],
                         kwargs['search_depth'], kwargs['c'], kwargs['n_playout'], kwargs['playout_depth'], kwargs['play_first'], kwargs['tree_type'],
                         kwargs['eps'])
        self.opponent = InforGo(kwargs['player_len'], kwargs['pattern_len'], kwargs['n_hidden_layer'], kwargs['n_node_hidden'],
                         kwargs['activation_fn'], kwargs['learning_rate'], kwargs['directory'], kwargs['alpha'], kwargs['gamma'], kwargs['lamda'],
                         kwargs['search_depth'], kwargs['c'], kwargs['n_playout'], kwargs['playout_depth'], not kwargs['play_first'], 
                         kwargs['opponent_tree_type'], kwargs['eps'])

    def train(self):
        percentage = 0
        logger.info("[Reinforcement] Start Training")
        logger.info("[Reinforcement] Training Complete: 0%")
        for epoch in range(self.n_epoch):
            state = State()
            s = state.get_initial_state()
            tmp = '\n'
            while True:
                action = self.get_action(state, state.player)
                v = self.AI.nn.predict(s, 1, get_pattern(s, 1))
                v_ = self.AI.nn.predict(s, -1, get_pattern(s, -1))
                flag, new_s, R = state.take_action(*action)
                new_v = self.AI.nn.predict(new_s, 1, get_pattern(new_s, 1))
                new_v_ = self.AI.nn.predict(new_s, -1, get_pattern(new_s, -1))
                self.AI.nn.update(s, 1, get_pattern(s, 1), TD(v, new_v, R, self.AI.alpha, self.AI.gamma))
                self.AI.nn.update(s, -1, get_pattern(s, -1), TD(v_, new_v_, -R, self.AI.alpha, self.AI.gamma))
                self.AI.tree.step(encode_action(action))
                self.opponent.tree.step(encode_action(action))
                if state.terminate(): break
                s = new_s
                action = self.get_action(state, state.player)
                v = self.AI.nn.predict(s, -1, get_pattern(s, -1))
                v_ = self.AI.nn.predict(s, 1, get_pattern(s, 1))
                flag, new_s, R = state.take_action(*action)
                new_v = self.AI.nn.predict(new_s, -1, get_pattern(new_s, -1))
                new_v_ = self.AI.nn.predict(new_s, 1, get_pattern(new_s, 1))
                self.AI.nn.update(s, -1, get_pattern(s, -1), TD(v, new_v, R, self.AI.alpha, self.AI.gamma))
                self.AI.nn.update(s, 1, get_pattern(s, 1), TD(v_, new_v_, -R, self.AI.alpha, self.AI.gamma))
                self.AI.tree.step(encode_action(action))
                self.opponent.tree.step(encode_action(action))
                if state.terminate(): break
                s = new_s
            winner = 1 if state.win(1) else -1 if state.win(-1) else 0
            if epoch / self.n_epoch > percentage / 100:
                percentage = math.ceil(epoch / self.n_epoch * 100)
                logger.info('[Reinforcement] Training Complete: {}%'.format(percentage))
            if percentage % 10 == 0: self.AI.nn.store()
            self.AI.refresh()
            self.opponent.refresh()
        logger.debug('[Reinforcement] Training Complete: 100%')
        self.AI.nn.store()

    def get_action(self, state, player):
        ai_player = 1 if self.AI.play_first else -1
        if player == ai_player: return self.AI.get_action(state)
        else: return self.opponent.get_action(state)

