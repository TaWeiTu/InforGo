import tensorflow as tf
import numpy as np
import os.path
import random
from utils import *
from global_mod import *
from env import MDP


class InforGo(object):
    '''
    A Neural Network model for training 3D-bingo AI
    input-layer: 4 x 4 x 4 board
    convolution-layer: stride = 1 x 1 x 1
    hidden-layer: relu/sigmoid/tanh function
    output-layer: approximate value for the input state
    '''
    def __init__(self, reward_function, n_epoch=100, n_hidden_layer=1, n_node_hidden=[32], activation_function='relu', output_function=None, learning_rate=0.00000001, alpha=0.00000001, gamma=0.99, td_lambda=0.85, regularization_param=0.001, decay_step=10000, decay_rate=0.96, convolution=True, filter_depth=1, filter_height=1, filter_width=1, out_channel=5, search_depth=3, first=True):
        if DEBUG:
            print("[Init] Start setting training parameter")

        self.first = first
        self.n_epoch = n_epoch
        self.alpha = alpha

        # Learning rate decay
        self.global_step = tf.Variable(0, trainable=False)
        # self.learning_rate = tf.train.exponential_decay(learning_rate, self.global_step, decay_step, decay_rate, staircase=True)
        self.learning_rate = learning_rate

        # Discount factor between 0 to 1
        self.gamma = gamma

        # lambda of TD-lambda algorithm
        self.td_lambda = td_lambda

        # R(s) is the reward obtain by achieving state s
        self.reward_function = reward_function

        # Number of hidden layers and number of nodes in each hidden layer
        self.n_hidden_layer = n_hidden_layer
        self.n_node_hidden = n_node_hidden

        # Maximum search depth in Minimax Tree Search
        self.search_depth = search_depth

        # Activation function at hidden layer
        if activation_function == 'relu': self.activation_function = lambda k: tf.nn.relu(k, name='relu')

        elif activation_function == 'sigmoid': self.activation_function = lambda k: tf.sigmoid(k, name='sigmoid')

        elif activation_function == 'tanh': self.activation_function = lambda k: tf.tanh(k, name='tanh')

        # Output function
        if output_function is None: self.output_function = lambda k: k

        elif output_function == 'softmax': self.output_function = lambda k: tf.nn.softmax(k)

        # L2 regularization paramater
        self.regularization_param = regularization_param

        self.MDP = MDP(reward_function)
        if DEBUG: print("[Init] Done setting training parameter")

        # Neural Network Setup

        # input layer: 4 x 4 x 4 Tensor representing the state
        with tf.name_scope('Input-Layer'):
            self.inp = tf.placeholder(shape=[4, 4, 4, 1, 1], dtype=tf.float64, name='input')

        if DEBUG: print("[Init] Done consturcting input layer")

        # 3D-Convolution layer
        if convolution:
            with tf.name_scope('Convolution-Layer'):
                self.conv_layer_w = tf.cast(tf.Variable(tf.truncated_normal(shape=[filter_depth, filter_height, filter_width, 1, out_channel], mean=0.0, stddev=1.0)), tf.float64, name='weight')
                self.conv_layer = tf.nn.conv3d(input=self.inp, filter=self.conv_layer_w, strides=[1, 1, 1, 1, 1], padding='SAME', name='Conv-Layer')

                # Flatten the convolution layer
                self.conv_layer_output = tf.cast(tf.reshape(self.conv_layer, [1, -1], name='Flattend'), dtype=tf.float64)

                # Caculate the length of flattened layer
                self.conv_layer_length = 4 * 4 * 4 * out_channel
        else:
            self.conv_layer_output = tf.reshape(self.inp, [1, -1], name='Flattened')
            self.conv_layer_length = 4 * 4 * 4

        if DEBUG: print("[Init] Done constructing convolution layer with out_channel = {}".format(out_channel))

        with tf.name_scope('Player-Node'):
            self.player_node = tf.placeholder(shape=[1, 1], dtype=tf.float64, name='Player-Node')

        with tf.name_scope('Pattern'):
            self.pattern = tf.placeholder(shape=[1, 6], dtype=tf.float64, name='Pattern')

        # Store all the weight and bias between each layer
        self.weight_and_bias = [{} for i in range(self.n_hidden_layer + 1)]

        with tf.name_scope('Weight_and_Bias'):
            self.weight_and_bias[0] = {
                'weight': self.get_weight(self.conv_layer_length + 1 + 6, self.n_node_hidden[0], 0),
                'bias': self.get_bias(self.n_node_hidden[0], 0)
            }
            if DEBUG: print("[Init] Done initializing weight and bias from convolution layer to hidden layer 0")
            for i in range(1, self.n_hidden_layer):
                self.weight_and_bias[i] = {
                    'weight': self.get_weight(self.n_node_hidden[i - 1], self.n_node_hidden[i], i),
                    'bias': self.get_bias(self.n_node_hidden[i], i)
                }
                if DEBUG: print("[Init] Done initializing weight and bias from hidden layer {} to hidden layer {}".format(i - 1, i))
            self.weight_and_bias[self.n_hidden_layer] = {
                'weight': self.get_weight(self.n_node_hidden[self.n_hidden_layer - 1], 1, self.n_hidden_layer),
                'bias': self.get_bias(1, self.n_hidden_layer)
            }
            if DEBUG: print("[Init] Done initializing weight and bias from hidden layer {} to output layer".format(self.n_hidden_layer - 1))

        # Store value of every node in each hidden layer
        self.hidden_layer = [{} for i in range(self.n_hidden_layer)]

        with tf.name_scope('Hidden_Layer'):
            self.hidden_layer[0] = {
                'output': tf.add(tf.matmul(tf.concat([self.conv_layer_output, self.player_node, self.pattern], 1), self.weight_and_bias[0]['weight']), self.weight_and_bias[0]['bias'])
            }
            for i in range(1, self.n_hidden_layer):
                self.hidden_layer[i - 1]['activated_output'] = self.activation_function(self.hidden_layer[i - 1]['output'])
                if DEBUG: print("[Init] Done activating output of hidden layer {}".format(i - 1))
                self.hidden_layer[i] = {
                    'output': tf.add(tf.matmul(self.hidden_layer[i - 1]['activated_output'], self.weight_and_bias[i]['weight']), self.weight_and_bias[i]['bias'])
                }
            self.hidden_layer[self.n_hidden_layer - 1]['activated_output'] = self.hidden_layer[self.n_hidden_layer - 1]['output']
            if DEBUG: print("[Init] Done activating output of hidden layer {}".format(self.n_hidden_layer - 1))

        with tf.name_scope('Output_Layer'):
            self.output = tf.add(tf.matmul(self.hidden_layer[self.n_hidden_layer - 1]['activated_output'], self.weight_and_bias[self.n_hidden_layer]['weight'], ), self.weight_and_bias[self.n_hidden_layer]['bias'])
            self.V = self.output_function(self.output)
            if DEBUG: print("[Init] Done constructing output layer")

        with tf.name_scope('Training_Model'):
            # Q-value to update the weight
            self.V_desired = tf.placeholder(shape=[1, 1], dtype=tf.float64)

            # Cost function
            def L2_Regularization():
                self.L2_value = 0
                for i in range(0, self.n_hidden_layer + 1):
                    self.L2_value += tf.nn.l2_loss(self.weight_and_bias[i]['weight']) + tf.nn.l2_loss(self.weight_and_bias[i]['bias'])
                return self.L2_value

            self.loss = tf.reduce_sum(tf.square(self.V_desired - self.V))
            if DEBUG: print("[Init] Done caculating cost function")
            # use gradient descent to optimize our model
            self.trainer = tf.train.GradientDescentOptimizer(self.learning_rate)
            # self.model = self.trainer.minimize(self.loss, global_step=global_step)
            self.model = self.trainer.minimize(self.loss)
            if DEBUG: print("[Init] Done setting up trainer")

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        if DEBUG: print("[Init] Done initializing all variables")

    def get_weight(self, n, m, layer):
        '''
        Weight to the layer-th hidden layer, with size n x m
        '''
        if os.path.exists('../Data/Weight/{}.txt'.format(layer)):
            f = open('../Data/Weight/{}.txt'.format(layer), 'r')
            w = np.zeros([n, m])
            for i in range(n):
                for j in range(m):
                    try:
                        w[i, j] = float(f.readline())
                    except:
                        if DEBUG: print("[ERROR] NaN or unstored weight")
                        os.remove('../Data/Weight/{}.txt'.format(layer))
                        return tf.truncated_normal(shape=[n, m], mean=0.0, stddev=0.001, dtype=tf.float64)
            f.close()
            return tf.Variable(tf.cast(w, tf.float64))
        else:
            return tf.Variable(tf.truncated_normal(shape=[n, m], mean=0.0, stddev=0.001, dtype=tf.float64))

    def get_bias(self, n, layer):
        '''
        Bias to the layer-th hidden layer, with size 1 x n
        '''
        if os.path.exists('../Data/Bias/{}.txt'.format(layer)):
            f = open('../Data/Bias/{}.txt'.format(layer), 'r')
            b = np.zeros([1, n])
            for i in range(n):
                try:
                    b[0, i] = float(f.readline())
                except:
                    if DEBUG: print("[ERROR] NaN or unstored bias")
                    os.remove('../Data/Bias/{}.txt'.format(layer))
                    return tf.Variable(tf.truncated_normal([1, n], mean=0.0, stddev=0.001, dtype=tf.float64))
            f.close()
            return tf.Variable(tf.cast(b, tf.float64))
        else:
            return tf.Variable(tf.truncated_normal(shape=[1, n], mean=0.0, stddev=0.001, dtype=tf.float64))


    def decode_action(self, action_num):
        action = [0, 0]
        for i in range(2):
            action[i] = action_num % 4
            action_num //= 4
        return action

    def train(self, run_test=True, run_self_play=True, run_generator=True, n_generator=1000, MAX=0, training_directory=[None]):
        '''
        Main Learning Process
        return final score, graph_x, graph_y
        '''
        writer = tf.summary.FileWriter(LOG_DIR)
        writer.add_graph(self.sess.graph)
        if DEBUG:
            print("[Train] Done Tensorboard setup")
            print("[Train] Start training")
        percentage = 0
        record = self.get_record(run_test, run_self_play, run_generator, n_generator, MAX, training_directory)
        if DEBUG:
            print("[Train] Done Collecting record")
            print("[Train] Training Complete: {}%".format(percentage))

        loss = []
        for epoch in range(self.n_epoch):
            loss_sum = 0
            update = 0
            for directory in record.keys():
                for file_name in record[directory]:
                    for rotate_time in range(4):
                        print("file_name: {}".format(file_name))
                        f = open('{}/{}'.format(directory, file_name), 'r')
                        s = self.MDP.get_initial_state()
                        while True:
                            try:
                                height, row, col = map(int, f.readline().split())
                            except:
                                if DEBUG: print("[ERROR] Invalid file format or context {}".format(file_name))
                                break

                            if (height, row, col) == (-1, -1, -1): break

                            height, row, col = self.rotate(height, row, col, rotate_time)

                            v = self.sess.run(self.V, feed_dict={self.inp: s, self.player_node: self.cast_player(1), self.pattern: get_pattern(s, 1)})
                            v_ = self.sess.run(self.V, feed_dict={self.inp: s, self.player_node: self.cast_player(-1), self.pattern: get_pattern(s, -1)})
                            flag, new_s, R = self.MDP.take_action((row, col), 1)

                            new_v = self.sess.run(self.V, feed_dict={self.inp: new_s, self.player_node: self.cast_player(1), self.pattern: get_pattern(new_s, 1)})
                            new_v_ = self.sess.run(self.V, feed_dict={self.inp: new_s, self.player_node: self.cast_player(-1), self.pattern: get_pattern(new_s, -1)})
                            v_desired = np.zeros([1, 1])
                            v_desired_ = np.zeros([1, 1])
                            # TD-0 update
                            # v_desired[0][0] = new_v[0][0] * self.td_lambda + self.alpha * (1 - self.td_lambda) * (R + self.gamma * new_v[0][0] - v[0][0])
                            v_desired[0][0] = v[0][0] + self.alpha * (R + self.gamma * new_v[0][0] - v[0][0])
                            # loss.append(self.sess.run(self.loss, feed_dict={self.V_desired: v_desired, self.inp: s, self.player_node: self.cast_player(1)}))
                            v_desired_[0][0] = v_[0][0] + self.alpha * (-R + self.gamma * new_v_[0][0] - v_[0][0])
                            loss_sum += self.sess.run(self.loss, feed_dict={self.V_desired: v_desired, self.inp: s, self.player_node: self.cast_player(1), self.pattern: get_pattern(s, 1)})
                            update += 1
                            self.sess.run(self.model, feed_dict={self.V_desired: v_desired, self.inp: s, self.player_node: self.cast_player(1), self.pattern: get_pattern(s, 1)})
                            self.sess.run(self.model, feed_dict={self.V_desired: v_desired_, self.inp: s, self.player_node: self.cast_player(-1), self.pattern: get_pattern(s, -1)})
                            s = new_s

                            try:
                                height, row, col = map(int, f.readline().split())
                            except:
                                if DEBUG: print("[ERROR] Invalid file format or context {}".format(file_name))
                                break

                            if (height, row, col) == (-1, -1, -1): break

                            height, row, col = self.rotate(height, row, col, rotate_time)

                            v = self.sess.run(self.V, feed_dict={self.inp: s, self.player_node: self.cast_player(-1), self.pattern: get_pattern(s, -1)})
                            flag, new_s, R = self.MDP.take_action((row, col), -1)
                            v_ = self.sess.run(self.V, feed_dict={self.inp: s, self.player_node: self.cast_player(1), self.pattern: get_pattern(s, 1)})

                            new_v = self.sess.run(self.V, feed_dict={self.inp: new_s, self.player_node: self.cast_player(-1), self.pattern: get_pattern(new_s, -1)})
                            # TD-0 update
                            # v_desired[0][0] = new_v[0][0] * self.td_lambda + self.alpha * (1 - self.td_lambda) * (R + self.gamma * new_v[0][0] - v[0][0])
                            v_desired[0][0] = v[0][0] + self.alpha * (R + self.gamma * new_v[0][0] - v[0][0])
                            # loss.append(self.sess.run(self.loss, feed_dict={self.V_desired: v_desired, self.inp: s, self.player_node: self.cast_player(1)}))
                            loss_sum += self.sess.run(self.loss, feed_dict={self.V_desired: v_desired, self.inp: s, self.player_node: self.cast_player(-1), self.pattern: get_pattern(s, -1)})
                            update += 1
                            new_v_ = self.sess.run(self.V, feed_dict={self.inp: new_s, self.player_node: self.cast_player(1), self.pattern: get_pattern(new_s, 1)})
                            v_desired_[0][0] = v[0][0] + self.alpha * (-R + self.gamma * new_v_[0][0] - v_[0][0])
                            self.sess.run(self.model, feed_dict={self.V_desired: v_desired, self.inp: s, self.player_node: self.cast_player(-1), self.pattern: get_pattern(s, -1)})

                            self.sess.run(self.model, feed_dict={self.V_desired: v_desired_, self.inp: s, self.player_node: self.cast_player(1), self.pattern: get_pattern(s, 1)})
                            s = new_s

            # loss.append(loss_sum / update)
            if epoch / self.n_epoch > percentage / 100:
                percentage = math.ceil(epoch / self.n_epoch * 100)
                if DEBUG: print("[Train] Training Complete: {}%".format(percentage))
            if percentage % 10 == 0: self.store_weight_and_bias()

        if DEBUG: print("[Train] Training Complete: {}%".format(100))
        self.store_weight_and_bias()
        return loss

    def rotate(self, height, row, col, t):
        '''
        Rotate the board 90 x t degree clockwise
        '''
        for i in range(t):
            row, col = col, row
            col = 3 - col
        return height, row, col

    def get_record(self, run_test=True, run_self_play=True, run_generator=True, n_generator=1000, MAX=0, training_directory=[None]):
        '''
        Return every record file under ../Data/record/*
        '''
        directory = [x[0] for x in os.walk('../Data/record')]
        directory = directory[1:]
        filename = {}
        for d in directory:
            if training_directory[0] and not d in training_directory: continue
            if d == '../Data/record/test_record' and not run_test: continue
            if d == '../Data/record/self_play' and not run_self_play: continue
            if d == '../Data/record/generator': continue
            tmp = [x[2] for x in os.walk(d)]
            filename[d] = [x for x in tmp[0]]
        if not run_generator: return filename
        s = set([])
        filename['../Data/record/generator'] = []
        for i in range(n_generator):
            game_id = random.randint(0, MAX)
            while game_id in s or not os.path.exists('../Data/record/generator/{}'.format(game_id)) or game_id % 10 != 0: game_id = random.randint(0, MAX)
            s.add(game_id)
            filename['../Data/record/generator'].append('{}'.format(game_id))
        return filename

    def store_weight_and_bias(self):
        '''
        Store weights under ./Data/Weight, biases under ./Data/Bias
        '''
        for i in range(self.n_hidden_layer + 1):
            f = open('../Data/Weight/{}.txt'.format(i), 'w')
            w = self.sess.run(self.weight_and_bias[i]['weight'])
            for j in range(self.sess.run(tf.shape(self.weight_and_bias[i]['weight']))[0]):
                for k in range(self.sess.run(tf.shape(self.weight_and_bias[i]['weight']))[1]):
                    f.write('{}\n'.format(w[j, k]))
            f.close()
            if DEBUG: print("[Train] Done storing weight {}".format(i))

            f = open('../Data/Bias/{}.txt'.format(i), 'w')
            b = self.sess.run(self.weight_and_bias[i]['bias'])
            for j in range(self.sess.run(tf.shape(self.weight_and_bias[i]['bias']))[1]):
                f.write('{}\n'.format(b[0, j]))
            f.close()
            if DEBUG: print("[Train] Done storing bias {}".format(i))

    def play(self, test_flag=False, bot=None, AI=None):
        if test_flag: tmp = tempfile.NamedTemporaryFile(dir='../Data/record/test_record', delete=False)
        elif AI is not None: tmp = tempfile.NamedTemporaryFile(dir='../Data/record/self_play', delete=False)
        else: tmp = tempfile.NamedTemporaryFile(dir='../Data/record/selfrecord', delete=False)
        winner = 0
        s = self.MDP.get_initial_state()
        record = ''
        if DEBUG and not test_flag and AI is None: print("[Play] Start playing")

        player = 1
        if self.first is False:
            if DEBUG and not self.test_flag and AI is None: print("[Play] Enter position")
            try:
                opponent = self.read_opponent_action(test_flag, bot, AI=AI)
            except:
                if DEBUG and not self.test_flag and AI is None: print("[ERROR] Fail to read opponent action")
                os.remove(tmp.name)
                return
            while self.MDP.valid_action(opponent) is False:
                if DEBUG and not self.test_flag and AI is None:
                    print("[FATAL] Invalid")
                    print("[FATAL] Re-enter position")
                try:
                    opponent = self.read_opponent_action(test_flag, bot, AI=AI)
                except:
                    if not self.test_flag and AI is None: print("[ERROR] Fail to read opponent action")
            row, col = opponent
            height = self.MDP.bingo.height[row][col]
            record += '{} {} {}\n'.format(height, row, col)

            flag, s, _ = self.MDP.take_action(opponent, player)
            player = -player
            if flag == 1:
                record += '-1 -1 -1\n'
                if DEBUG and not test_flag and AI is None: print("[Play] User win")
                winner = 1

        while True:
            if DEBUG:
                plot_state(s)
                print("[DEBUG] Evaluate (player {}): {}".format(player, self.evaluate(s, player)))
            # Choose the best action using Minimax Tree Search
            _, action = self.Minimax(Bingo(s), self.search_depth, 'Max', player)
            row, col = self.decode_action(action)

            height = self.MDP.bingo.height[row][col]
            self.emit_action(height, row, col, test_flag, AI)
            record += '{} {} {}\n'.format(height, row, col)

            action = (row, col)
            flag, s, _ = self.MDP.take_action(action, player)

            if flag == player:
                record += '-1 -1 -1\n'
                if DEBUG and not test_flag and AI is None: print("[Play] AI win")
                winner = player
                break
            if flag == 3:
                record += '-1 -1 -1\n'
                if DEBUG and not test_flag and AI is None: print("[Play] Draw")
                winner = 3
                break

            player = -player
            if DEBUG:
                plot_state(s)
                print("[DEBUG] Evaluate (player {}): {}".format(player, self.evaluate(s, player)))
            if DEBUG and not test_flag and AI is None: print("[Play] Enter position")

            try:
                opponent = self.read_opponent_action(test_flag, bot, AI=AI)
            except:
                if DEBUG and not test_flag and AI is None: print("[ERROR] Invalid Opponent Move")
                os.remove(tmp.name)
                break

            success = True
            while self.MDP.valid_action(opponent) is False:
                if DEBUG and not test_flag and AI is None:
                    print("[FATAL] Invalid input action")
                    print("[FATAL] Re-enter position")
                try:
                    opponent = self.read_opponent_action(test_flag, bot, AI=AI)
                except:
                    if DEBUG and not test_flag and AI is None: print("[ERROR] Invalid Opponent Move")
                    os.remove(tmp.name)
                    success = False
                    break

            if not success: break
            row, col = opponent
            height = self.MDP.bingo.height[row][col]
            record += '{} {} {}\n'.format(height, row, col)

            flag, s, _ = self.MDP.take_action(opponent, player)

            if flag == player:
                record += '-1 -1 -1\n'
                if DEBUG and not test_flag and AI is None: print("[Play] User win")
                winner = player
                break

            player = -player

            if flag == 3:
                record += '-1 -1 -1\n'
                if DEBUG and not test_flag and AI is None: print("[Play] Draw")
                winner = 3
                break
        # Record the game for future training
        f = open(tmp.name, 'w')
        f.write(record)
        f.close()
        return winner

    def Minimax(self, bingo, depth, level, player, alpha=-np.inf, beta=np.inf):
        '''
        recursively search every possible action with maximum depth = self.search_depth
        use alpha-beta pruning to reduce time complexity
        '''
        state = bingo.get_state()

        if bingo.win(1) or bingo.win(-1): return self.evaluate(state, player), None

        if depth == 0: return self.evaluate(state, player), None

        value = np.inf if level == 'Min' else -np.inf
        action = 0
        next_player = -player
        next_level = 'Max' if level == 'Min' else 'Max'
        func = lambda a, b: max(a, b) if level == 'Max' else lambda a, b: min(a, b)

        move = False
        permutation = [i for i in range(16)]
        random.shuffle(permutation)
        for i in permutation:
            r, c = self.decode_action(i)
            if bingo.valid_action(r, c):
                move = True
                bingo.place(r, c)
                new_bingo = Bingo(bingo.get_state())
                bingo.undo_action(r, c)

                val, a = self.Minimax(new_bingo, depth - 1, next_level, next_player, alpha, beta)

                if level == 'Min': beta = min(beta, val)
                else: alpha = max(alpha, val)

                value = func(value, val)

                # Lowerbound is greater than the upperbound, stop further searching
                if alpha > beta: return value, action

                if value == val: action = i

        if not move: return 0, None
        return value, action

    def evaluate(self, state, player):
        '''
        Evaluate the value of input state with neural network as an approximater
        '''
        V = self.sess.run(self.V, feed_dict={self.inp: state, self.player_node: self.cast_player(player), self.pattern: get_pattern(state, player)})
        return V[0][0]

    def cast_player(self, player):
        tmp = np.zeros([1, 1])
        tmp[0, 0] = player
        return tmp

    def read_opponent_action(self, test_flag, bot, AI=None):
        if AI is not None:
            state = self.MDP.get_state()
            player = 1 if AI.first else -1
            value, action = AI.Minimax(Bingo(state), AI.search_depth, 'Max', player)
            return self.decode_action(action)

        if test_flag: return bot.generate_action(self.MDP.get_state())

        try: h, r, c = map(int, input().split())
        except: raise
        return r, c

    def emit_action(self, height, row, col, test_flag, AI=None):
        if test_flag or AI is not None: return
        if DEBUG:
            print("[DEBUG] ", end='')
            print("{} {} {}".format(height, row, col))
        else:
            print(height + row * 4 + col * 16)

    def test(self):
        win = 0
        percentage = 0
        player = 1 if self.first else -1
        bot = Bot(-player)
        if DEBUG: print("[Test] Test Complete: {}%".format(0))
        for epoch in range(self.n_epoch):
            winner = self.play(test_flag=True, bot=bot)
            if winner == player: win += 1

            if epoch / self.n_epoch > percentage / 100:
                percentage = math.ceil(epoch / self.n_epoch * 100)
                if DEBUG: print("[Test] Test Complete: {}%".format(percentage))
        if DEBUG: print("[Test] Test Complete: {}%".format(100))
        print("[Test] Winning Percentage: {}%".format(win / self.n_epoch * 100.))
        return win / self.n_epoch * 100.
