import os.path
import argparse
import distutils.util
import coloredlogs, logging
import matplotlib.pyplot as plt

from InforGo.environment import global_var


# Ignore warning and tensorflow stdout
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    os.environ['COLOREDLOGS_LOG_FORMAT'] = "[%(hostname)s] %(asctime)s - %(message)s"
    logger = logging.getLogger('InforGo')
    coloredlogs.install(level='DEBUG')

    # logger.info('[Main] Start Collecting Arguments')
    parser = argparse.ArgumentParser(description='Execution argument')

    # Method
    parser.add_argument('method', help='run/train/test')

    # Log
    parser.add_argument('--logdir', '-lg', default='tensorboard', help='Tensorboard log directory')

    # Training parameter
    parser.add_argument('--learning_rate', '-lr', default=0.001, type=float, help='learning rate for the neural network')
    parser.add_argument('--gamma', '-g', default=0.99, type=float, help='discount factor')
    parser.add_argument('--alpha', '-a', default=0.1, type=float, help='learning rate for TD(0)-learning')
    parser.add_argument('--lamda', '-ld', default=0.5, type=float, help='TD(lambda)')

    # Model parameter
    parser.add_argument('--n_epoch', '-ne', default=100, type=int, help='number of epochs')
    parser.add_argument('--n_hidden_layer', '-nh', default=1, type=int, help='number of hidden layers')
    parser.add_argument('--n_node_hidden', '-nn', default=[32], type=int, nargs='+', help='nodes in each hidden layers')

    # Neuron
    parser.add_argument('--activation_fn', '-fn', default='tanh', type=str, help='activation function')

    # DEBUG
    parser.add_argument('--debug', '-d', const=True, default=False, nargs='?', type=distutils.util.strtobool, help='Debug mode')
    parser.add_argument('--verbose', '-v', const=True, default=False, nargs='?', help='Verbose Mode')

    # Run
    parser.add_argument('--play_first', '-pf', default=True, type=distutils.util.strtobool, help='Play first')

    # Supervised Training
    parser.add_argument('--n_generator', '-ng', default=1000, type=int, help='Train the model with n_generator auto-generated game')
    parser.add_argument('--MAX', default=12877521, type=int, help='Maximum generated game id')
    parser.add_argument('--training_directory', '-td', default=[None], type=str, nargs='+', help='Specify training data directory')
    parser.add_argument('--logfile', '-lf', default=None, type=str, help='Log file')
    parser.add_argument('--n_test', '-nt', default=1000, type=int, help='Number of test file to train')
    parser.add_argument('--n_self_play', '-ns', default=1000, type=int, help='Number of self-play to train')
    parser.add_argument('--player_len', default=1, type=int, help='Number of player nodes in neural network')
    parser.add_argument('--pattern_len', default=6, type=int, help='Number of patterns')
    parser.add_argument('--directory', '-dir', default='./Data/default/', type=str, help='Directory to store weight and bias')
    
    # Reinforcement Training
    parser.add_argument('--opponent_tree_type', '-ott', default='minimax', help='Tree type for opponent in reinforcement learning')

    # Tree
    parser.add_argument('--eps', '-e', default=0.1, type=float, help='Probability of choosing random action in minimax')
    parser.add_argument('--n_playout', '-np', default=30, type=int, help='Number of playouts at each action selection')
    parser.add_argument('--playout_depth', '-pd', default=1, type=int, help='Depth of playout')
    parser.add_argument('--tree_type', '-tt', default='minimax', type=str, help='minimax/mcts')
    parser.add_argument('--search_depth', '-sd', default=3, type=int, help='Maximum search depth')
    parser.add_argument('--c', '-c', default=0.5, type=float, help='Exploration/Exploitation')
    
    # GPU
    parser.add_argument('--gpu', default=False, const=True, nargs='?', help='Run Tensorflow with/without GPUs')
    args = parser.parse_args()

    global_var.__dict__['LOG_DIR'] = '../log/' + args.logdir
    global_var.__dict__['DEBUG'] = args.debug
    global_var.__dict__['GPU'] = args.gpu
    global_var.__dict__['LOGGER'] = logger
    global_var.__dict__['VERBOSE'] = args.verbose
    
    if args.directory[-1] != '/': args.directory += '/'
    if args.debug: logger.info('[Main] Done Collecting Arguments')

    from InforGo.process.supervised_trainer import SupervisedTrainer
    from InforGo.process.reinforcement_trainer import ReinforcementTrainer
    from InforGo.process.tester import Tester
    from InforGo.process.runner import Runner
    from InforGo.process.debugger import Debugger

    if args.method == 's_train': 
        errors = SupervisedTrainer(**vars(args)).train(args.logfile)
        x = [i for i in range(len(errors))]
        try:
            plt.plot(x, errors)
            plt.show()
        except:
            for i in errors: print(i)
    elif args.method == 'r_train': ReinforcementTrainer(**vars(args)).train()
    elif args.method == 'run': Runner(**vars(args)).run()
    elif args.method == 'debug': Debugger(**vars(args)).debug()
    elif args.method == 'test': 
        win, loss, t = Tester(**vars(args)).test()
        logger.info('[Test] Win: {}/{}'.format(win, args.n_epoch))
        logger.info('[Test] Loss: {}/{}'.format(loss, args.n_epoch))
        logger.info('[Test] Total time: {}s'.format(t))


if __name__ == '__main__':
    main()
