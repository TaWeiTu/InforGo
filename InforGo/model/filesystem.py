import os
import random
import numpy as np

from InforGo.environment.bingo import Bingo as State
from InforGo.util import logger, plot_state


class FileSystem(object):

    def __init__(self, gamma, training_directory=[None], n_test=0, n_self_play=0, n_generator=0, MAX=0):
        dirs = self._get_record(training_directory, n_test, n_self_play, n_generator, MAX)
        self.files = []
        self.gamma = gamma
        for d, files in dirs.items():
            for f in files: self.files.append('{}/{}'.format(d, f))
        self.shuffle = [i for i in range(len(self.files))]

    def _get_record(self, training_directory, n_test, n_self_play, n_generator, MAX):
        logger.info('[Filesystem] Start Collecting Record')
        directory = [x[0] for x in os.walk('./Data/record')]
        directory = directory[1:]
        filename = {}
        test = []
        self_play = []
        for d in directory:
            if training_directory[0] and not d in training_directory: continue
            if d == './Data/record/test_record':
                if n_test > 0: test = [x for x in tmp[0]]
                continue
            if d == './Data/record/self_play':
                if n_self_play > 0: self_play = [x for x in tmp[0]]
                continue
            if d == './Data/record/generator': continue
            tmp = [x[2] for x in os.walk(d)]
            filename[d] = [x for x in tmp[0]]
        if training_directory: return filename
        if n_test > 0:
            random.shuffle(test)
            filename['./Data/record/test'] = test[:min(n_test, len(test))]
        if n_self_play > 0:
            random.shuffle(self_play)
            filename['./Data/record/self_play'] = self_play[:min(n_self_play, len(self_play))]
        if n_generator == 0: 
            logger.info('[Filesystem] Done Collecting Record')
            return filename
        s = set([])
        filename['./Data/record/generator'] = []
        for i in range(self._n_generator):
            game_id = random.randint(0, MAX)
            while game_id in s or not os.path.exists('./Data/record/generator/{}'.format(game_id)) or game_id % 10 != 0: 
                game_id = random.randint(0, MAX)
            s.add(game_id)
            filename['./Data/record/generator'].append('{}'.format(game_id))
        logger.info('[Filesystem] Done Collecting Record')
        return filename

    def get_next_batch(self, number):
        random.shuffle(self.shuffle)
        files = [self.files[i] for i in self.shuffle[:number]]
        data = self._read_file(files)
        x, y = self._get_value(data)
        return x, y

    def _read_file(self, files):
        data = []
        for fi in files:
            data.append([])
            with open(fi, 'r') as f:
                while True:
                    try: 
                        height, row, col = map(int, f.readline().split())
                        if (height, row, col) == (-1, -1, -1): break
                        data[-1].append((height, row, col))
                    except:
                        logger.error("[Error] Invalid file context, {}".format(fi))
                        break
                f.close()
        return data

    def _get_value(self, data):
        x, y = [], []
        for dat in data:
            for rotate_time in range(4):
                states, rewards = [], {1: [], -1: []}
                s = State()
                c_player = 1
                for (height, row, col) in dat:
                    height, row, col = self._rotate_data(height, row, col, rotate_time)
                    flag, _, r = s.take_action(row, col)
                    rewards[c_player].append(r)
                    rewards[-c_player].append(-r)
                    c_player *= -1
                reward_sum = {1: [0 for i in range(len(rewards[1]) + 1)], -1: [0 for i in range(len(rewards[1]) + 1)]}
                for i in range(len(rewards[1])):
                    reward_sum[1][len(rewards[1]) - i - 1] = reward_sum[1][len(rewards[1]) - i] * self.gamma + \
                                                                rewards[1][len(rewards[1]) - i - 1]
                    reward_sum[-1][len(rewards[-1]) - i - 1] = reward_sum[-1][len(rewards[-1]) - i] * self.gamma + \
                                                                rewards[-1][len(rewards[-1]) - i - 1]
                s = State()
                c_player = 1
                ind = 0
                for (height, row, col) in dat:
                    height, row, col = self._rotate_data(height, row, col, rotate_time)
                    x.append((np.array(s.get_state()), 1))
                    y.append(reward_sum[1][ind])
                    x.append((np.array(s.get_state()), -1))
                    y.append(reward_sum[-1][ind])
                    s.take_action(row, col)
                    ind += 1
        return x, y
                
    def _rotate_data(self, height, row, col, t):
        """Rotate the board 90 x t degree clockwise"""
        for i in range(t):
            row, col = col, row
            col = 3 - col
        return height, row, col
