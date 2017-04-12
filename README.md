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
To play with InforGo on command line
```bash
$ python bingo.py play [--argument=<value>]
```
For example, play with AI that has a maximum search depth 3.
```base
$ python bingo.py play --search_depth=3
```
To train InforGo, first store the data in ```./Data/record/*```, the format of the file should contains 3 numbers per line, indicating the (height, row, col) of the corresponding position, with "-1 -1 -1" at the last line(without quote).
```bash
$ python bingo.py train [--argument=<value>]
```
For example, train AI with 3 hidden layers, each layer is constructed with 32, 16, 8 nodes, respectively.
```base
$ python bingo.py train --n_hidden_layer=3 --n_node_hidden 32 16 8
```
To test how good InforGo is trained
```bash
python bingo.py test [--argument=<value>]
```
To make InforGo play against itself
```bash
python bingo.py self-play [--argument=<value>]
```
InforGo will be playing with a random bot which win if it can and prevent loss from opponent  

Valid argument are the following:
* **n_epoch:** number of epoches to every training data, default is 1
* **n_hidden_layer:** number of hidden layers, default is 1
* **n_node_hidden:** an array of length n_hidden_layer, representing the number of nodes in corresponding hidden layer, default is [32]
* **activation_function:** activation function of hidden layers, "relu", "sigmoid", "tanh" are currently available, default is "relu"
* **learning_rate:** learning rate of the neural network, default is 0.00000001
* **gamma:** discount factor of the value funciton, default is 0.99
* **regularization_param:** regularization parameter of L2-Regularization, default is 0.001
* **decay_step:** step of learning rate decay, default is 0.96
* **decay_rate:** rate of learning rate decay, default is 100
* **filter_depth:** depth of filter of Convolution layer, default is 1
* **filter_height:** height of filter of Convolution layer, default is 1
* **filter_width:** width of filter of Convolution layer, default is 1
* **out_channel:** the number of output of Convolution layer, default is 5
* **search_depth:** maximum search depth of Minimax Tree Search, default is 3
* **DEBUG:** Debug mode, default is False
* **first:** AI go first or second, default is True
* **run_test:** Train the model with testing data, default is True
