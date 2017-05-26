# InforGo
InforGo is a 3D-Bingo AI developed by [INFOR 29th](https://infor.org)
* **Algorithm:** InforGo is trained using Reinforcement learning, and uses TD(0) algorithm to evaluate the value function. When state evaluation is done, InforGo selects actions based on Minimax or Monte-Carlo Tree Search.
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
python -m InforGo.main [method] [--argument=value, -arg val]
```
Valid method are: 
* **s_train:** Supervised training of neural network by reading game file
* **r_train:** Reinforcement training by self-playing
* **test:** Testing the training result
* **run:** Play with human
* **debug:** Debug mode for neural network

Valid argument are the following:
* **\-\-n_epoch, -ne:** [s_train, r_train, test] number of epoches to every training data, default is 1
* **\-\-n_hidden_layer, -nh:** number of hidden layers, default is 1
* **\-\-n_node_hidden, -nn:** an array of length n_hidden_layer, representing the number of nodes in corresponding hidden layer, default is [32]
* **\-\-activation_fn, -fn:** activation function of hidden layers, "relu", "sigmoid", "tanh" are currently available, default is "tanh"
* **\-\-tree_type, -tt:** [r_train, test, run] minimax or mcts
* **\-\-opponent_tree_type, -ott:** [r_train] minimax or mcts for opponent
* **\-\-learning_rate, -lr:** [s_train, r_train] learning rate of the neural network, default is 0.001
* **\-\-gamma, -g:** discount factor of the value funciton, default is 0.99
* **\-\-alpha, -a:** learning rate of TD, default is 0.1
* **\-\-search_depth, -sd:** [r_train, test, run] maximum search depth of Minimax Tree Search, default is 3
* **\-\-lamda, -ld:** [r_train, test, run] Parameter that unifies between state evaluation and simulated result
* **\-\-c, -c:** [r_train, test, run] Parameter that controls exploration vs exploitation tradeoff
* **\-\-n_playout, -np:** [r_train, test, run] Number of playouts at each action selection
* **\-\-playout_depth, -pd:** [r_train, test, run] Depth where simulation starts
* **\-\-rollout_limit, -rl:** [r_train, test, run] Limit of rollout
* **\-\-debug:** Debug mode, default is False
* **\-\-play_first, -pf:** [r_train, test, run] AI go first or second, default is True
* **\-\-n_test, -nt:** [s_train] Number of test data to be trained
* **\-\-n_self_play, -ns:** [s_train] Number of self-play data to be trained
* **\-\-n_generator, -ng:** [s_train] Number of generated data to be trained
* **\-\-MAX:** [s_train] Maximum id of generated data to be trained
* **\-\-player_len:** Number of player nodes in neural network
* **\-\-pattern_len:** Length of pattern recognition
* **\-\-directory, -dir:** Directory to store weight and bias
* **\-\-training_directory, -td:** [s_train] Directory to be trained, if not specified, train all of the avaible directory
