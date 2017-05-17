import os
import random
import math

from InforGo.process.schema import Schema as schema
from InforGo.environment.bingo import Bingo as State
from InforGo.util import get_pattern, TD
from InforGo.environment.global_var import *
from InforGo.util import logger, log_state, plot_state
from InforGo.model.filesystem import FileSystem


class SupervisedTrainer(schema):

    def __init__(self, **kwargs):
        super().__init__(kwargs['n_epoch'], kwargs['player_len'], kwargs['pattern_len'], kwargs['n_hidden_layer'], kwargs['n_node_hidden'],
                         kwargs['activation_fn'], kwargs['learning_rate'], kwargs['directory'], kwargs['alpha'], kwargs['gamma'], kwargs['lamda'],
                         kwargs['search_depth'], kwargs['c'], kwargs['n_playout'], kwargs['playout_depth'], kwargs['play_first'], kwargs['tree_type'],
                         kwargs['eps'])
        self._n_test = kwargs['n_test']
        self._n_generator = kwargs['n_generator']
        self._n_self_play = kwargs['n_self_play']
        self._training_directory = kwargs['training_directory']
        self._MAX = kwargs['MAX']
        self.fs = FileSystem(self._AI.gamma, self._training_directory, self._n_test, self._n_self_play, self._n_generator, self._MAX)
        self._batch = kwargs['batch']

    def train(self, logfile):
        """Supervised training"""
        percentage = 0
        log = open(logfile, 'w') if logfile else None
        logger.info('[Supervised] Start Training')
        logger.info('[Supervised] Training Complete: 0%')
        errors = []
        for epoch in range(self._n_epoch):
            x, y = self.fs.get_next_batch(self._batch)
            # print("get batch")
            v = self._evaluate(x)
            err = self._update(x, v + self._AI.alpha * (y - v))
            errors.append(err)
            if epoch / self._n_epoch > percentage / 100:
                percentage = math.ceil(epoch / self._n_epoch * 100)
                logger.info('[Supervised] Training Complete: {}%'.format(percentage))
            if percentage % 10 == 0: self._store()

        logger.info('[Supervised] Training Complete: 100%')
        self._store()
        if logfile is not None: log.close()
        return errors
    
    def _store(self):
        """store weight and bias"""
        self._AI.nn.store()

    def _update(self, data, value):
        states = [state for state, player in data]
        players = [player for state, player in data]
        return self._AI.nn.update(states, players, value)

    def _evaluate(self, data):
        states = [state for state, player in data]
        players = [player for state, player in data] 
        return self._AI.nn.predict(states, players)
