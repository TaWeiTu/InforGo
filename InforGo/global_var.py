import os


LOG_DIR = '../log/tensorboard'
DEBUG = False
DEVICE = os.environ['DEVICE'] if 'DEVICE' in os.environ else 'cpu'
