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
$ python bingo.py play [parameter value]
```
To train InforGo, first store the data in ```./save/*```, the format of the file should contains 3 numbers per line, indicating the (height, row, col) of the corresponding position, with "-1 -1 -1" at the last line(without quote).
```bash
$ python bingo.py train [parameter value]
```
Valid parameters are the following:
* **n_epoch:** number of epoches to every training data
* **n_hidden_layer:** number of hidden layers
* **n_node_hidden:** an array of length n_hidden_layer, representing the number of nodes in corresponding hidden layer
* **activation_function:** activation function of hidden layers, "Relu", "Sigmoid", "Tanh" are currently available
* **learning_rate:** learning rate of the neural network
* **gamma:** discount factor of the value funciton
* **regularization_param:** regularization parameter of L2-Regularization
* **decay_step:** step of learning rate decay
* **decay_rate:** rate of learning rate decay
* **filter_depth:** depth of filter of Convolution layer
* **filter_height:** height of filter of Convolution layer
* **filter_width:** width of filter of Convolution layer
* **out_channel:** the number of output of Convolution layer
* **search_depth:** maximum search depth of Minimax Tree Search
* **DEBUG:** if specified DEBUG, debug log will appear
* **first:** if specified first, AI play first
* **second:** if specified second, AI play second
