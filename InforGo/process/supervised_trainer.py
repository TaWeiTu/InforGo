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
        self.n_test = kwargs['n_test']
        self.n_generator = kwargs['n_generator']
        self.n_self_play = kwargs['n_self_play']
        self.training_directory = kwargs['training_directory']
        self.MAX = kwargs['MAX']

    def get_record(self):
        logger.info('[Supervised] Start Collecting Record')
        directory = [x[0] for x in os.walk('./Data/record')]
        directory = directory[1:]
        filename = {}
        test = []
        self_play = []
        for d in directory:
            if self.training_directory[0] and not d in self.training_directory: continue
            if d == './Data/record/test_record':
                if self.n_test > 0: test = [x for x in tmp[0]]
                continue
            if d == './Data/record/self_play':
                if self.n_self_play > 0: self_play = [x for x in tmp[0]]
                continue
            if d == './Data/record/generator': continue
            tmp = [x[2] for x in os.walk(d)]
            filename[d] = [x for x in tmp[0]]
        if self.n_test > 0:
            random.shuffle(test)
            filename['./Data/record/test'] = test[:min(self.n_test, len(test))]
        if self.n_self_play > 0:
            random.shuffle(self_play)
            filename['./Data/record/self_play'] = self_play[:min(self.n_self_play, len(test))]
        if self.n_generator == 0: 
            logger.info('[Supervised] Done Collecting Record')
            return filename
        s = set([])
        filename['./Data/record/generator'] = []
        for i in range(self.n_generator):
            game_id = random.randint(0, self.MAX)
            while game_id in s or not os.path.exists('./Data/record/generator/{}'.format(game_id)) or game_id % 10 != 0: 
                game_id = random.randint(0,self. MAX)
            s.add(game_id)
            filename['./Data/record/generator'].append('{}'.format(game_id))
        logger.info('[Supervised] Done Collecting Record')
        return filename

    def train(self, logfile):
        """Supervised training """
        percentage = 0
        record = self.get_record()
        log = open(logfile, 'w') if logfile else None
        env = State()
        logger.info('[Supervised] Start Training')
        logger.info('[Supervised] Training Complete: 0%')
        errors = []
        tmp = open('./error.log', 'w')
        for epoch in range(self.n_epoch):
            loss_sum = 0
            update = 0
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
                        while not state.terminate():
                            try: height, row, col = map(int, f.readline().split())
                            except:
                                logger.error("[Error] Invalid file input")
                                break
                            flag, new_s, R = state.take_action(row, col)
                            for p in [1, -1]:
                                tmp.write("Current State for player {}: \n".format(c_player * p))
                                log_state(s, tmp)
                                v = self.AI.nn.predict(s, c_player * p, get_pattern(s, c_player * p))
                                tmp.write("v = {}\n".format(v))
                                new_v = self.AI.nn.predict(new_s, c_player * p, get_pattern(new_s, c_player * p))
                                tmp.write("new_v = {}\n".format(new_v))
                                tmp.write("R = {}\n".format(R * p))
                                err = self.AI.nn.update(s, c_player * p, get_pattern(s, c_player * p), TD(v, new_v, R * p, self.AI.alpha, self.AI.gamma))
                                tmp.write("TD = {}\n".format(TD(v, new_v, R * p, self.AI.alpha, self.AI.gamma)))
                                errors.append(err)
                                tmp.write('[Supervised] error = {}\n'.format(err))
                            s = new_s
                            c_player *= -1
                            
            if epoch / self.n_epoch > percentage / 100:
                percentage = math.ceil(epoch / self.n_epoch * 100)
                logger.info('[Supervised] Training Complete: {}%'.format(percentage))
            if percentage % 10 == 0: self.AI.nn.store()

        logger.info('[Supervised] Training Complete: 100%')
        self.AI.nn.store()
        tmp.close()
        if logfile is not None: log.close()
        return errors

    def rotate_data(self, height, row, col, t):
        """Rotate the board 90 x t degree clockwise"""
        for i in range(t):
            row, col = col, row
            col = 3 - col
        return height, row, col

    def read_game_file(self, f):
        state = State()
        actions = []
        while not state.terminate():
            height, row, col = map(int, f.readline().split())
            state.take_action(row, col)
            actions.append((row, col))
        winner = 1 if state.win(1) else -1 if state.win(-1) else 0
        actions = actions[::-1]
        return state, actions, winner
