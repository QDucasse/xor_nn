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

# No ML module version based on: https://towardsdatascience.com/implementing-the-xor-gate-using-backpropagation-in-neural-networks-c1f255b4f20d

import numpy as np

# Activation fucntion: sigmoid
def sigmoid (x):
    return 1/(1 + np.exp(-x))

# Derivative of the activation function for the backpropagation
def sigmoid_derivative(x):
    return x * (1 - x)

# Random weights and bias initialization
def initialize_random_weights_bias(inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons):
    hidden_weights = np.random.uniform(size=(inputLayerNeurons,hiddenLayerNeurons))
    hidden_bias =np.random.uniform(size=(1,hiddenLayerNeurons))
    output_weights = np.random.uniform(size=(hiddenLayerNeurons,outputLayerNeurons))
    output_bias = np.random.uniform(size=(1,outputLayerNeurons))
    # Print the obtained values
    print("===============================================")
    print("=============== INITIAL VALUES ================")
    print("===============================================")
    print("Initial hidden weights:\n" + str(hidden_weights))
    print("Initial hidden bias:\n" + str(hidden_bias))
    print("Initial output weights:\n" + str(output_weights))
    print("Initial output bias:\n" + str(output_bias))
    return hidden_weights, hidden_bias, output_weights, output_bias

# Training algorithm
def train(epochs, hidden_weights, hidden_bias, output_weights, output_bias):
    print("===============================================")
    print("================== TRAINING ===================")
    print("===============================================")
    for i in range(epochs):
        ## Forward Propagation
        hidden_layer_activation = np.dot(inputs,hidden_weights)
        hidden_layer_activation += hidden_bias
        hidden_layer_output = sigmoid(hidden_layer_activation)

        output_layer_activation = np.dot(hidden_layer_output,output_weights)
        output_layer_activation += output_bias
        predicted_output = sigmoid(output_layer_activation)

        #Backpropagation
        error = expected_output - predicted_output
        d_predicted_output = error * sigmoid_derivative(predicted_output)

        error_hidden_layer = d_predicted_output.dot(output_weights.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

        #Updating Weights and Biases
        output_weights += hidden_layer_output.T.dot(d_predicted_output) * lr
        output_bias += np.sum(d_predicted_output,axis=0,keepdims=True) * lr
        hidden_weights += inputs.T.dot(d_hidden_layer) * lr
        hidden_bias += np.sum(d_hidden_layer,axis=0,keepdims=True) * lr

        if (i % 1000 == 0):
            print("Epoch: {0}, Loss: {1}, ".format(i, error))

    # Print the final weights and biases
    print("===============================================")
    print("================ FINAL VALUES =================")
    print("===============================================")
    print("Final hidden weights:\n" + str(hidden_weights))
    print("Final hidden bias:\n" + str(hidden_bias))
    print("Final output weights:\n" + str(output_weights))
    print("Final output bias:\n" + str(output_bias))

    # Print final report
    print("===============================================")
    print("=================== OUTPUT ====================")
    print("===============================================")
    print("Output from neural network after {0} epochs:\n{1}".format(epochs,predicted_output))

if __name__ == "__main__":
    #Input datasets
    inputs = np.array([[0,0],
                       [0,1],
                       [1,0],
                       [1,1]])
    expected_output = np.array([[0],[1],[1],[0]])

    # Epochs and learning rate
    epochs = 10000
    lr = 0.1

    # Network topology
    inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2,2,1

    # Weights and biases initialization
    hidden_weights, hidden_bias, output_weights, output_bias = initialize_random_weights_bias(inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons)

    # Training phase
    train(epochs, hidden_weights, hidden_bias, output_weights, output_bias)
