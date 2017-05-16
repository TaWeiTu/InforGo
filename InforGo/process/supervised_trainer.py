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
        """Supervised training """
        percentage = 0
        log = open(logfile, 'w') if logfile else None
        logger.info('[Supervised] Start Training')
        logger.info('[Supervised] Training Complete: 0%')
        errors = []
        for epoch in range(self._n_epoch):
            x, y = self.fs.get_next_batch(self._batch)
            """for directory in record.keys():
                for file_name in record[directory]:
                    for rotate_time in range(4):
                        try: f = open('{}/{}'.format(directory, file_name), 'r')
                        except:
                            logger.error('[Error] No such file or directory: {}/{}'.format(directory, file_name))
                            continue
                        state = State()
                        c_player = 1
                        s = state.get_initial_state()
                        tmp.write('New Game\n')
                        # error, t = 0, 0
                        while not state.terminate():
                            try: 
                                height, row, col = map(int, f.readline().split())
                                height, row, col = self._rotate_data(height, row, col, rotate_time)
                            except:
                                logger.error("[Error] Invalid file input")
                                break
                            flag, new_s, R = state.take_action(row, col)
                            # print(R)
                            for p in [1, -1]:
                                v = self._evaluate(s, c_player * p)
                                new_v = self._evaluate(new_s, c_player * p)
                                err = self._update(s, c_player * p, TD(v, new_v, R * p, self._AI.alpha, self._AI.gamma))
                                error += err
                                t += 1
                            s = new_s
                            c_player *= -1
                        # if t: errors.append(error / t)"""
            v = self._evaluate(x)
            err = self._update(x, v + self._AI.alpha * (y - v))
            errors.append(err)
            if epoch / self._n_epoch > percentage / 100:
                percentage = math.ceil(epoch / self._n_epoch * 100)
                logger.info('[Supervised] Training Complete: {}%'.format(percentage))
            if percentage % 10 == 0: self._store()

        logger.info('[Supervised] Training Complete: 100%')
        self._store()
        # tmp.close()
        if logfile is not None: log.close()
        return errors
    
    def _store(self):
        self._AI.nn.store()

    def _update(self, data, value):
        states = [state for state, player in data]
        players = [player for state, player in data]
        return self._AI.nn.update(states, players, value)

    def _evaluate(self, data):
        states = [state for state, player in data]
        players = [player for state, player in data] 
        return self._AI.nn.predict(states, players)
