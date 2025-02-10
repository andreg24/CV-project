import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import copy
import random
from tqdm import trange

import os

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

from .training_utils import *

class CNN(nn.Module):
    def __init__(self, kernel_size=3, batch_norm=False, dropout=0, conv=False, linear=False):
        super(CNN, self).__init__()

        self.device = torch.device('cpu')
        self.batch_norm = batch_norm  # boolean
        self.dropout = dropout # probability
        self.kernel_size = kernel_size
        self.conv = conv
        self.linear = linear
        self.padding = kernel_size // 2

        self.network = nn.Sequential(*self.build())

    def build(self):
        """Function returning the list of layers of the network"""

        layers = []

        # first block
        layers.append(nn.Conv2d(in_channels=1, out_channels=8, kernel_size=self.kernel_size, stride=1, padding=self.padding))
        if self.batch_norm:
            layers.append(nn.BatchNorm2d(num_features=8))
        layers.append(nn.ReLU())
        if self.dropout > 0:
            layers.append(nn.Dropout(self.dropout))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # second block
        layers.append(nn.Conv2d(in_channels=8, out_channels=16, kernel_size=self.kernel_size, stride=1, padding=self.padding))
        if self.batch_norm:
            layers.append(nn.BatchNorm2d(num_features=16))
        layers.append(nn.ReLU())
        if self.dropout > 0:
            layers.append(nn.Dropout(self.dropout))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # third block
        layers.append(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=self.kernel_size, stride=1, padding=self.padding))
        if self.batch_norm:
            layers.append(nn.BatchNorm2d(num_features=32))
        layers.append(nn.ReLU())
        if self.dropout > 0:
            layers.append(nn.Dropout(self.dropout))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        if self.conv:
            # fourth optional block
            layers.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel_size, stride=1, padding=self.padding))
            if self.batch_norm:
                layers.append(nn.BatchNorm2d(num_features=32))
            layers.append(nn.ReLU())

        layers.append(nn.Flatten(1))
        if self.linear:
            layers.append(nn.Linear(32*8*8, 32*8*8))
        layers.append(nn.Linear(32*8*8, 15))
        
        return layers

    def forward(self, x):
        x = self.network(x)
        return x
    
    def to(self, device: torch.device) -> None:
        self.device = device
        self.network.to(device)

def initialize_weights(layer):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

class Ensemble:

    def __init__(self, n, config):
        self.n = n # number of models
        self.config = config
        self.models = []
        self.device = torch.device('cpu')

        for i in range(n):
            model = CNN(**config)
            initialize_weights(model.network)
            self.models.append(model)

    def to(self, device):
        for model in self.models:
            model.to(device)
        self.device = device
    
    def train(
        self,
        loaders,
        # optimizer, 
        loss_function=nn.CrossEntropyLoss(),
        num_epochs=100,
        device=torch.device('cuda')
        ):

        data = []
        for model in self.models:
            optimizer = optim.Adam(model.parameters())

            losses, accuracies = training(model, loaders, optimizer, loss_function, num_epochs, early_stopper=EarlyStopping(15), device=device)
            data.append([losses, accuracies])
        return data