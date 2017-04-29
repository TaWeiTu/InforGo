import os
import random
import math

from InforGo.process.schema import Schema as schema
from InforGo.environment.bingo import Bingo as State
from InforGo.util import get_pattern, TD
from InforGo.environment.global_var import *
from InforGo.util import logger


class Trainer(schema):

    def __init__(self, **kwargs):
        super().__init__(kwargs['n_epoch'], kwargs['player_len'], kwargs['pattern_len'], kwargs['n_hidden_layer'], kwargs['n_node_hidden'],
                         kwargs['activation_fn'], kwargs['learning_rate'], kwargs['directory'], kwargs['alpha'], kwargs['gamma'], kwargs['lamda'],
                         kwargs['search_depth'], kwargs['c'], kwargs['n_playout'], kwargs['playout_depth'], kwargs['play_first'], kwargs['tree_type'])
        self.n_test = kwargs['n_test']
        self.n_generator = kwargs['n_generator']
        self.n_self_play = kwargs['n_self_play']
        self.training_directory = kwargs['training_directory']
        self.MAX = kwargs['MAX']

    def get_record(self):
        logger.info('[Train] Start Collecting Record')
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
            logger.info('[Train] Done Collecting Record')
            return filename
        s = set([])
        filename['./Data/record/generator'] = []
        for i in range(self.n_generator):
            game_id = random.randint(0, self.MAX)
            while game_id in s or not os.path.exists('./Data/record/generator/{}'.format(game_id)) or game_id % 10 != 0: game_id = random.randint(0,self. MAX)
            s.add(game_id)
            filename['./Data/record/generator'].append('{}'.format(game_id))
        logger.info('[Train] Done Collecting Record')
        return filename

    def train(self, logfile):
        percentage = 0
        record = self.get_record()
        log = open(logfile, 'w') if logfile else None
        env = State()
        logger.info('[Train] Start Training')
        logger.debug('[Train] Training Complete: 0%')
        for epoch in range(self.n_epoch):
            loss_sum = 0
            update = 0
            for directory in record.keys():
                for file_name in record[directory]:
                    for rotate_time in range(4):
                        f = open('{}/{}'.format(directory, file_name), 'r')
                        s = env.get_initial_state()
                        if logfile: log.write("New Game\n")
                        while True:
                            try: height, row, col = map(int, f.readline().split())
                            except:
                                logger.error('Invalid file format or context {}'.format(file_name))
                                break
                            if (height, row, col) == (-1, -1, -1): break

                            height, row, col = self.rotate_data(height, row, col, rotate_time)
                            # Value of the current state
                            v = self.AI.nn.predict(s, 1, get_pattern(s, 1))
                            v_ = self.AI.nn.predict(s, -1, get_pattern(s, -1))
                            flag, new_s, R = env.take_action(row, col)
                            # Value of the successive state
                            new_v = self.AI.nn.predict(new_s, 1, get_pattern(new_s, 1))
                            new_v_ = self.AI.nn.predict(new_s, -1, get_pattern(new_s, -1))
                            
                            self.AI.nn.update(s, 1, get_pattern(s, 1), TD(v, new_v, R, self.AI.alpha, self.AI.gamma))
                            self.AI.nn.update(s, -1, get_pattern(s, -1), TD(v_, new_v_, -R, self.AI.alpha, self.AI.gamma))
                            
                            s = new_s

                            try: height, row, col = map(int, f.readline().split())
                            except:
                                logger.error('Invalid file format or context {}'.format(file_name))
                                break
                            if (height, row, col) == (-1, -1, -1): break

                            height, row, col = self.rotate_data(height, row, col, rotate_time)

                            # Value of the current state
                            v = self.AI.nn.predict(s, -1, get_pattern(s, -1))
                            v_ = self.AI.nn.predict(s, 1, get_pattern(s, 1))
                            flag, new_s, R = env.take_action(row, col)
                            # Value of the successive state
                            new_v = self.AI.nn.predict(new_s, -1, get_pattern(new_s, -1))
                            new_v_ = self.AI.nn.predict(new_s, 1, get_pattern(new_s, 1))
                            
                            self.AI.nn.update(s, -1, get_pattern(s, -1), TD(v, new_v, R, self.AI.alpha, self.AI.gamma))
                            self.AI.nn.update(s, 1, get_pattern(s, 1), TD(v_, new_v_, -R, self.AI.alpha, self.AI.gamma))
                            s = new_s

            if epoch / self.n_epoch > percentage / 100:
                percentage = math.ceil(epoch / self.n_epoch * 100)
                logger.debug('[Train] Training Complete: {}%'.format(percentage))
            if percentage % 10 == 0: self.AI.nn.store()

        logger.debug('[Train] Training Complete: 100%')
        self.AI.nn.store()
        if logfile is not None: log.close()

    def rotate_data(self, height, row, col, t):
        """Rotate the board 90 x t degree clockwise"""
        for i in range(t):
            row, col = col, row
            col = 3 - col
        return height, row, col

