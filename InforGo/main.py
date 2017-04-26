import os.path
import argparse
import distutils.util

from InforGo.environment import global_var
from InforGo.process.trainer import Trainer
from InforGo.process.tester import Tester
from InforGo.process.runner import Runner


# Ignore warning and tensorflow stdout
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():

    parser = argparse.ArgumentParser(description='Execution argument')

    # Method
    parser.add_argument('method', help='play/train/self-play/test')

    # Log
    parser.add_argument('--logdir', default='tensorboard', help='Tensorboard log directory')

    # Training parameter
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate for the neural network')
    parser.add_argument('--gamma', default=0.99, type=float, help='discount factor')
    parser.add_argument('--alpha', default=0.1, type=float, help='learning rate for TD(0)-learning')
    parser.add_argument('--lamda', default=0.5, type=float, help='TD(lambda)')

    # Model parameter
    parser.add_argument('--n_epoch', default=100, type=int, help='number of epochs')
    parser.add_argument('--n_hidden_layer', default=1, type=int, help='number of hidden layers')
    parser.add_argument('--n_node_hidden', default=[32], type=int, nargs='+', help='nodes in each hidden layers')

    # Neuron
    parser.add_argument('--activation_fn', default='tanh', type=str, help='activation function')

    # DEBUG
    parser.add_argument('--DEBUG', default=False, type=distutils.util.strtobool, help='Debug mode')

    # Play
    parser.add_argument('--play_first', default=True, type=distutils.util.strtobool, help='Play first')
    parser.add_argument('--search_depth', default=3, type=int, help='maximum search depth')

    # Train
    parser.add_argument('--n_generator', default=1000, type=int, help='Train the model with n_generator auto-generated game')
    parser.add_argument('--MAX', default=12877521, type=int, help='Maximum generated game id')
    parser.add_argument('--training_directory', default=[None], type=str, nargs='+', help='Specify training data directory')
    parser.add_argument('--logfile', default=None, type=str, help='Log file')
    parser.add_argument('--n_test', default=1000, type=int, help='Number of test file to train')
    parser.add_argument('--n_self_play', default=1000, type=int, help='Number of self-play to train')
    parser.add_argument('--player_len', default=1, type=int, help='Number of player nodes in neural network')
    parser.add_argument('--pattern_len', default=6, type=int, help='Number of patterns')
    parser.add_argument('--directory', default='./Data/default/', type=str, help='Directory to store weight and bias')

    args = parser.parse_args()

    global_var.__dict__['LOG_DIR'] = '../log/' + args.logdir
    global_var.__dict__['DEBUG'] = args.DEBUG == 1
    
    if args.method == 'train': Trainer(**vars(args)).train(args.logfile)
    elif args.method == 'run': Runner(**vars(args)).run()
    elif args.method == 'test': Tester(**vars(args)).test()


if __name__ == '__main__':
    main()
