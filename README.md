# XOR Neural Network

## Network Presentation

This repository contains different implementations of a *Neural Network* resolving the *XOR Problem*. This problem consists of replicating the behavior of a typical *XOR* operation that follows the truth table below:

| **Input** | **Input** | **Output** |
| :-------: | :-------: | :--------: |
|     A     |     B     |  A XOR B   |
|     0     |     0     |     0      |
|     0     |     1     |     1      |
|     1     |     0     |     1      |
|     1     |     1     |     0      |

The implemented neural network is extremely simple as it consists  of an input layer receiving the two entries (*e.g.* True False, or 1 0 as it will be represented), a linear hidden layer performing the reduce operation with correct weights then applying the sigmoid function and a final linear output layer performing the same operations but with only one output. 

The network topology can be seen as follows:

<center>
  <img src="XOR_NN.png" alt="Topology" style="zoom: 67%;" /> 
</center


## Network Implementations

The network is implemented in three different ways:

1. **No ML Module:** This implementation is made from scratch with no additional machine learning module. It is based on **[1]**.
2. **PyTorch:** This implementation is made using the machine learning package *PyTorch*. It is based on **[2]**.
3. **Brevitas**: This implementation is made using the extension of *PyTorch* named *Brevitas* and aiming at *Quantized Neural Networks* and their implementation on *FPGA*s. It is based on **[3]**. 

You can run the training phase for each implementation by using the command:

```bash
$ python <target implementation>.py
```



## Installation

To install the repository:

```bash
$ cd <directory you want to install in>
$ git clone https://github.com/QDucasse/xor_nn
$ python setup.py install
```

You can use the following if you use `virtualenvwrapper`:

```bash
$ cd <directory you want to install in>
$ git clone https://github.com/QDucasse/xor_nn
$ mkvirtualenv -a . -r requirements.txt <your virtualenv name>
```

You will probably need to install *Brevitas* manually with the following command:

```bash
$ pip install git+https://github.com/Xilinx/brevitas.git
```





[1]: https://towardsdatascience.com/implementing-the-xor-gate-using-backpropagation-in-neural-networks-c1f255b4f20d	"XOR NN from scratch"
[2]: https://courses.cs.washington.edu/courses/cse446/18wi/sections/section8/XOR-Pytorch.html	"XOR NN PyTorch"
[3]: https://github.com/kf7lsu/Brevitas-XOR-Test	"XOR NN Brevitas"



