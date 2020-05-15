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

# Brevitas version based on: https://github.com/kf7lsu/Brevitas-XOR-Test

import numpy as np

from torch.nn import Module
from torch import nn
from torch import Tensor
from torch import from_numpy
import torch
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import brevitas.nn as qnn
from brevitas.core.quant import QuantType
from brevitas.quant_tensor import pack_quant_tensor


class XorDataset(Dataset):
    def __init__(self):
        a = Tensor([0,0])
        b = Tensor([0,1])
        c = Tensor([1,0])
        d = Tensor([1,1])
        aq = pack_quant_tensor(a,torch.tensor(.125, dtype=torch.float32),torch.tensor(4.0, dtype=torch.float32))
        bq = pack_quant_tensor(b,torch.tensor(.125, dtype=torch.float32),torch.tensor(4.0, dtype=torch.float32))
        cq = pack_quant_tensor(c,torch.tensor(.125, dtype=torch.float32),torch.tensor(4.0, dtype=torch.float32))
        dq = pack_quant_tensor(d,torch.tensor(.125, dtype=torch.float32),torch.tensor(4.0, dtype=torch.float32))
        self.data=[aq,bq,cq,dq]
        self.key=[torch.tensor(0.0, dtype=torch.float32),torch.tensor(1.0, dtype=torch.float32),
                  torch.tensor(1.0, dtype=torch.float32),torch.tensor(0.0, dtype=torch.float32)]

    def __getitem__(self, index):
        x = self.data[index]
        y = self.key[index]

        return x, y

    def __len__(self):
        return len(self.data)

class QuantXORNet(Module):
    def __init__(self):
        super(QuantXORNet, self).__init__()
        self.relu0 = qnn.QuantReLU(quant_type=QuantType.INT,
                                  bit_width=4.0,
                                  max_val=8)
        self.linear1 = qnn.QuantLinear(in_features = 2,
                                       out_features=2,
                                       bias_quant_type=QuantType.INT,
                                       bias=True,
                                       compute_output_scale=True,
                                       compute_output_bit_width=True,
                                       #input_bit_width=32,
                                       weight_quant_type=QuantType.INT)
        self.relu1 = qnn.QuantReLU(quant_type=QuantType.INT,
                                  bit_width=4,
                                  max_val=8)
        self.linear2 = qnn.QuantLinear(in_features = 2,
                                       out_features=1,
                                       bias_quant_type=QuantType.INT,
                                       bias=True,
                                       compute_output_scale=True,
                                       compute_output_bit_width=True,
                                       #bit_width=4,
                                       weight_quant_type=QuantType.INT)

    def forward(self, x):
        res=x
        res = self.relu0(res)
        res = pack_quant_tensor(res,torch.tensor(1.0, dtype=torch.float32),torch.tensor(4.0, dtype=torch.float32))
        res = self.linear1(res)
        res = pack_quant_tensor(res,torch.tensor(1.0, dtype=torch.float32),torch.tensor(4.0, dtype=torch.float32))
        res = self.relu1(res)
        res = pack_quant_tensor(res,torch.tensor(1.0, dtype=torch.float32),torch.tensor(4.0, dtype=torch.float32))
        res = self.linear2(res)
        return res


def train_model(model, loader, loss_func, optimizer):
    model.train()
    loss_func.train()
    running_loss = 0.0
    for i in range(epochs):
        for j,data in enumerate(loader):
            # Zero the parameter gradients
            optimizer.zero_grad()
            inputs, targets = data

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = loss_func(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if j % 500 == 0:
                print("Epoch: {0}, Loss: {1}, ".format(i, running_loss/4))
                running_loss = 0.0

if __name__ == "__main__":
    # Dataset instanciation
    loader = DataLoader(XorDataset())

    # Model instanciation
    model = QuantXORNet()

    # Optimizer instanciation
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.999)
    loss_func = nn.MSELoss()

    # Epochs definition
    epochs = 2000

    # Training phase
    train_model(model, loader, loss_func, optimizer)
