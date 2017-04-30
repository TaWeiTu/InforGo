import os


LOG_DIR = '../log/tensorboard'
DEBUG = False
VERBOSE = False
DEVICE = os.environ['DEVICE'] if 'DEVICE' in os.environ else 'cpu'
GPU = False
LOGGER = None
