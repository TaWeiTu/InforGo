# InforGo
InforGo is a 3D-Bingo AI developed by [INFOR 29th](https://infor.org)
* **Algorithm:** InforGo is trained using Reinforcement learning, and uses TD(0) algorithm to evaluate the value function.
* **Neural Netork:** InforGo uses neural network to be the approximator of the value function. The architecture of the neural network consists of 3D-Convolution layer, fully-connected layer and one output neuron representing the approximated value of the input state.
* **Game:** The 3D-Bingo game is designed by [MiccWan](https://github.com/MiccWan), to play, visit [https://bingo.infor.org](https://bingo.infor.org)
## Installation
Download the repository
```bash
$ git clone https://github.com/TaWeiTu/InforGo.git
$ cd InforGo
$ ./build.sh
```
InforGo is developed in [Python3](https://www.python.org/), and uses [Tensorflow](https://www.tensorflow.org/) as its machine learning library.
To download Python3, visit the officail website [https://www.python.org/downloads/](https://www.python.org/downloads/)
To download Tensorflow and other required packages, run the following commands:
```bash
$ pip install tensorflow
$ pip install matplotlib
```
## Examples
To execute the project
```bash
python -m InforGo.main [method] [--argument=value]
```
Valid argument are the following:
* **n_epoch:** number of epoches to every training data, default is 1
* **n_hidden_layer:** number of hidden layers, default is 1
* **n_node_hidden:** an array of length n_hidden_layer, representing the number of nodes in corresponding hidden layer, default is [32]
* **activation_fn:** activation function of hidden layers, "relu", "sigmoid", "tanh" are currently available, default is "tanh"
* **learning_rate:** learning rate of the neural network, default is 0.001
* **gamma:** discount factor of the value funciton, default is 0.99
* **alpha:** learning rate of TD, default is 0.1
* **search_depth:** maximum search depth of Minimax Tree Search, default is 3
* **DEBUG:** Debug mode, default is False
* **play_first:** AI go first or second, default is True
* **n_test:** Number of test data to be trained
* **n_self_play:** Number of self-play data to be trained
* **n_generator:** Number of generated data to be trained
* **MAX:** Maximum id of generated data to be trained
* **logdir:** Tensorboard logdir
* **logfile:** Logfile
* **player_len:** Number of player nodes in neural network
* **pattern_len:** Length of pattern recognition
* **directory:** Directory to store weight and bias
* **training_directory:** Directory to be trained, if not specified, train all of the avaible directory
