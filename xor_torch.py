# -*- coding: utf-8 -*-

# xor_nn
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

# Neural Network from scratch to solve the XOR Problem
# 1 1 --> 0
# 1 0 --> 1
# 0 1 --> 1
# 0 0 --> 0

# PyTorch version based on: https://courses.cs.washington.edu/courses/cse446/18wi/sections/section8/XOR-Pytorch.html

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt


class XOR(nn.Module):
    # Layers are defined in the init function as instance variables
    def __init__(self, input_dim = 2, output_dim=1):
        super(XOR, self).__init__()
        self.lin1 = nn.Linear(input_dim, 2)
        self.lin2 = nn.Linear(2, output_dim)
        self.weights_init()

    # forward is performed at each iteration of an input through the network
    def forward(self, x):
        x = self.lin1(x)
        x = torch.sigmoid(x)
        x = self.lin2(x)
        return x

    # Initialize the weights at random
    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # initialize the weight tensor, here we use a normal distribution
                m.weight.data.normal_(0, 1)

def train_model(model, input, loss_func, optimizer):
    for i in range(epochs):
        for j in range(steps):
            data_point = np.random.randint(X.size(0))
            x_var = Variable(X[data_point], requires_grad=False)
            y_var = Variable(Y[data_point], requires_grad=False)

            optimizer.zero_grad()
            y_hat = model(x_var)
            loss = loss_func.forward(y_hat, y_var)
            loss.backward()
            optimizer.step()

        if i % 500 == 0:
            print("Epoch: {0}, Loss: {1}, ".format(i, loss.data.numpy()))


def plot_results(X, model_weights, model_bias):
    plt.scatter(X.numpy()[[0,-1], 0], X.numpy()[[0, -1], 1], s=50)
    plt.scatter(X.numpy()[[1,2], 0], X.numpy()[[1, 2], 1], c='red', s=50)

    x_1 = np.arange(-0.1, 1.1, 0.1)
    y_1 = ((x_1 * model_weights[0,0]) + model_bias[0]) / (-model_weights[0,1])
    plt.plot(x_1, y_1)

    x_2 = np.arange(-0.1, 1.1, 0.1)
    y_2 = ((x_2 * model_weights[1,0]) + model_bias[1]) / (-model_weights[1,1])
    plt.plot(x_2, y_2)
    plt.legend(["neuron_1", "neuron_2"], loc=8)
    plt.show()



if __name__ == "__main__":
    # Set the torch random seed
    torch.manual_seed(2)

    # Inputs
    X = torch.Tensor([[0,0],[0,1], [1,0], [1,1]])

    # Expected output
    Y = torch.Tensor([0,1,1,0]).view(-1,1)

    # Instanciation of the model (weights are initialized here as well)
    model = XOR()

    # Function chosen for the loss
    loss_func = nn.MSELoss()

    # Optimizer selection
    optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)

    # Epochs and steps definition
    epochs = 2500
    steps = X.size(0)

    # Training phase
    train_model(model, X, loss_func, optimizer)

    # Gathering of the final results for plotting purposes
    model_params = list(model.parameters())
    model_weights = model_params[0].data.numpy()
    model_bias = model_params[1].data.numpy()
    plot_results(X, model_weights, model_bias)
