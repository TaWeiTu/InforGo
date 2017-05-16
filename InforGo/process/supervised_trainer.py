import os
import random
import math

from InforGo.process.schema import Schema as schema
from InforGo.environment.bingo import Bingo as State
from InforGo.util import get_pattern, TD
from InforGo.environment.global_var import *
from InforGo.util import logger, log_state


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

    def _get_record(self):
        # for dirs in self._training_directory: print(dirs)
        logger.info('[Supervised] Start Collecting Record')
        directory = [x[0] for x in os.walk('./Data/record')]
        directory = directory[1:]
        filename = {}
        test = []
        self_play = []
        for d in directory:
            if self._training_directory[0] and not d in self._training_directory: continue
            if d == './Data/record/test_record':
                if self._n_test > 0: test = [x for x in tmp[0]]
                continue
            if d == './Data/record/self_play':
                if self._n_self_play > 0: self_play = [x for x in tmp[0]]
                continue
            if d == './Data/record/generator': continue
            tmp = [x[2] for x in os.walk(d)]
            filename[d] = [x for x in tmp[0]]
        if self._training_directory[0]: return filename
        if self._n_test > 0:
            random.shuffle(test)
            filename['./Data/record/test'] = test[:min(self._n_test, len(test))]
        if self._n_self_play > 0:
            random.shuffle(self_play)
            filename['./Data/record/self_play'] = self_play[:min(self._n_self_play, len(test))]
        if self._n_generator == 0: 
            logger.info('[Supervised] Done Collecting Record')
            return filename
        s = set([])
        filename['./Data/record/generator'] = []
        for i in range(self._n_generator):
            game_id = random.randint(0, self._MAX)
            while game_id in s or not os.path.exists('./Data/record/generator/{}'.format(game_id)) or game_id % 10 != 0: 
                game_id = random.randint(0, self._MAX)
            s.add(game_id)
            filename['./Data/record/generator'].append('{}'.format(game_id))
        logger.info('[Supervised] Done Collecting Record')
        return filename

    def train(self, logfile):
        """Supervised training """
        percentage = 0
        record = self._get_record()
        log = open(logfile, 'w') if logfile else None
        env = State()
        logger.info('[Supervised] Start Training')
        logger.info('[Supervised] Training Complete: 0%')
        errors = []
        tmp = open('./error.log', 'w')
        for epoch in range(self._n_epoch):
            loss_sum = 0
            update = 0
            error, t = 0, 0
            for directory in record.keys():
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
                        # if t: errors.append(error / t)
            if t: errors.append(error / t)
            if epoch / self._n_epoch > percentage / 100:
                percentage = math.ceil(epoch / self._n_epoch * 100)
                logger.info('[Supervised] Training Complete: {}%'.format(percentage))
            if percentage % 10 == 0: self._store()

        logger.info('[Supervised] Training Complete: 100%')
        self._store()
        tmp.close()
        if logfile is not None: log.close()
        return errors

    def _rotate_data(self, height, row, col, t):
        """Rotate the board 90 x t degree clockwise"""
        for i in range(t):
            row, col = col, row
            col = 3 - col
        return height, row, col

    def _read_game_file(self, f):
        state = State()
        actions = []
        while not state.terminate():
            height, row, col = map(int, f.readline().split())
            state.take_action(row, col)
            actions.append((row, col))
        winner = 1 if state.win(1) else -1 if state.win(-1) else 0
        actions = actions[::-1]
        return state, actions, winner
    
    def _store(self):
        self._AI.nn.store()

    def _update(self, state, player, value):
        return self._AI.nn.update(state, player, value)

    def _evaluate(self, state, player):
        return self._AI.nn.predict(state, player)
