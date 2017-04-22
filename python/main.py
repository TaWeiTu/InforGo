import os.path
import argparse
import numpy as np
import distutils.util
import global_mod
from utils import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():

    parser = argparse.ArgumentParser(description='Execution argument')

    # Method
    parser.add_argument('method', help='play/train/self-play/test')

    # Log
    parser.add_argument('--logdir', default='tensorboard', help='Tensorboard log directory')

    # Training parameter
    parser.add_argument('--learning_rate', default=0.00000001, type=float, help='learning rate for the neural network')
    parser.add_argument('--gamma', default=0.99, type=float, help='discount factor')
    parser.add_argument('--alpha', default=0.00000001, type=float, help='learning rate for TD(0)-learning')
    parser.add_argument('--regularization_param', default=0.001, type=float, help='L2 regularization')
    parser.add_argument('--decay_rate', default=0.96, type=float, help='Decay rate')
    parser.add_argument('--decay_step', default=100, type=int, help='Decay step')

    # Model parameter
    parser.add_argument('--n_epoch', default=100, type=int, help='number of epochs')
    parser.add_argument('--n_hidden_layer', default=1, type=int, help='number of hidden layers')
    parser.add_argument('--n_node_hidden', default=[32], type=int, nargs='+', help='nodes in each hidden layers')

    # Neuron
    parser.add_argument('--activation_function', default='relu', type=str, help='activation function')
    parser.add_argument('--output_function', default=None, type=str, help='output function')

    # Convolution Layer
    parser.add_argument('--convolution', default=True, type=distutils.util.strtobool, help='With/Without convolution layer')
    parser.add_argument('--filter_depth', default=1, type=int, help='filter depth')
    parser.add_argument('--filter_height', default=1, type=int, help='filter height')
    parser.add_argument('--filter_width', default=1, type=int, help='filter width')
    parser.add_argument('--out_channel', default=5, type=int, help='out channel')

    # DEBUG
    parser.add_argument('--DEBUG', default=False, type=distutils.util.strtobool, help='Debug mode')

    # Play
    parser.add_argument('--first', default=True, type=distutils.util.strtobool, help='Play first')
    parser.add_argument('--search_depth', default=3, type=int, help='maximum search depth')

    # Train
    parser.add_argument('--run_test', default=True, type=distutils.util.strtobool, help='Train the model with testing data')
    parser.add_argument('--run_self_play', default=True, type=distutils.util.strtobool, help='Train the model with self-play data')
    parser.add_argument('--run_generator', default=True, type=distutils.util.strtobool, help='Train the model with auto-generated game')
    parser.add_argument('--n_generator', default=1000, type=int, help='Train the model with n_generator auto-generated game')
    parser.add_argument('--MAX', default=12877521, type=int, help='Maximum generated game id')
    parser.add_argument('--training_directory', default=[None], type=str, nargs='+', help='Specify training data directory')

    args = parser.parse_args()

    def reward_function(state, flag, player):
        np_state = np.zeros([4, 4, 4, 1, 1])
        for h in range(4):
            for r in range(4):
                for c in range(4):
                    np_state[h][r][c][0][0] = state[h][r][c]
        pattern = get_pattern(np_state, player)
        if flag == 3: return 0
        if flag == player: return 50
        if flag != player and flag != 0: return -50
        reward = 0
        for i in range(6):
            if i % 2 == 0: reward += pattern[0, i]
            else: reward -= pattern[0, i]
        return reward

    global_mod.__dict__['LOG_DIR'] = './log/' + args.logdir
    global_mod.__dict__['DEBUG'] = args.DEBUG == 1

    
    from ai import InforGo


    AI = InforGo(
        reward_function=reward_function,
        n_epoch=args.n_epoch,
        n_hidden_layer=args.n_hidden_layer,
        n_node_hidden=args.n_node_hidden,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        alpha=args.alpha,
        regularization_param=args.regularization_param,
        decay_step=args.decay_step,
        decay_rate=args.decay_rate,
        convolution=args.convolution,
        filter_depth=args.filter_depth,
        filter_height=args.filter_height,
        filter_width=args.filter_width,
        out_channel=args.out_channel,
        first=args.first,
        search_depth=args.search_depth,
        activation_function=args.activation_function,
        output_function=args.output_function)

    # TODO: run_generator is True even if specified False
    if args.method == 'train':
        loss = AI.train(run_test=args.run_test, run_self_play=args.run_self_play, run_generator=args.run_generator, n_generator=args.n_generator, MAX=args.MAX, training_directory=args.training_directory)
        try:
            f = open('../tmp', 'w')
            for i in loss: f.write('{}\n'.format(i))
            f.close()
        except:
            for i in loss: print(i, end=' ')
        # plt.plot([i for i in range(len(loss))], loss)
        # plt.show()

    elif args.method == 'play': AI.play()
    elif args.method == 'test': AI.test()
    elif args.method == 'self-play':
        a, b = self_play(AI, args)
        print("{} {}".format(a, b))
    elif args.method == 'check':
        debugger = Debugger(AI)
        debugger.check()


if __name__ == '__main__':
    main()
