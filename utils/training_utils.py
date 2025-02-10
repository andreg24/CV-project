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


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

__all__ = [
    "initialize_weights",
    "compute_accuracy",
    "compute_majority_accuracy",
    "compute_majority_accuracy",
    # "plot_loss",
    # "plot_accuracy",
    "EarlyStopping",
    "training",
    "CNN",
]

# TRAINING

def compute_accuracy(models, loader, device):
    if not isinstance(models, list):
        models = [models]

    correct: int = 0
    total: int = 0

    y_hat = []

    with torch.no_grad():
        for data in loader:
            predictions = []
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            for model in models:
                outputs =  model(images)
                _, predicted = torch.max(outputs.data, 1)
                predictions.append(predicted)
            y_hat += list(predicted.numpy(force=True))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct/total*100, y_hat

def compute_majority2_accuracy(models, loader, device):
    if not isinstance(models, list):
        models = [models]

    total = 0
    global_correct = 0
    majority_predictions = []

    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            predictions = []  # predictions of all models
            for model in models:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                predictions.append(predicted.unsqueeze(0))

            # majority vote
            stacked_preds = torch.cat(predictions, dim=0)
            majority_vote, _ = torch.mode(stacked_preds, dim=0)  # Most common prediction

            majority_predictions += list(majority_vote.cpu().numpy())
            global_correct += (majority_vote == labels).sum().item()
            total += labels.size(0)

    # Compute global accuracy
    majority_accuracy = (global_correct / total) * 100

    return majority_accuracy, majority_predictions

def compute_majority_accuracy(models, loader, device):
    if not isinstance(models, list):
        models = [models]

    total = 0
    global_correct = 0
    avg_predictions = []

    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            avg_logits = None  # Store summed logits
            for model in models:
                outputs = model(images)  # Get raw logits (before softmax)
                if avg_logits is None:
                    avg_logits = outputs
                else:
                    avg_logits += outputs  # Sum logits across models

            avg_logits /= len(models)  # Compute arithmetic mean

            _, final_predicted = torch.max(avg_logits, 1)  # Get final predictions
            avg_predictions += list(final_predicted.cpu().numpy())  # Convert to list
            global_correct += (final_predicted == labels).sum().item()
            total += labels.size(0)

    # Compute global accuracy
    avg_accuracy = (global_correct / total) * 100

    return avg_accuracy, avg_predictions



class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.current = float('inf')

    def __call__(self, value):
        if self.counter > self.patience:
            return True # stop training
        
        if value <= self.current:
            # better value
            self.current = value
            self.counter = 0
        else:
            # worse value
            self.counter += 1
        return False

def training(model, 
             loaders,
             optimizer, 
             loss_function=nn.CrossEntropyLoss(),
             num_epochs=100, 
             early_stopper=EarlyStopping(10),
             device=torch.device('cuda'),
             best_model=False
             ):
    """Function used to train a single model"""

    train_loader, val_loader, test_loader = loaders

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []
    
    if best_model:
        best_val_loss = float('inf')
        best_model_state = None

    model.to(device)

    for epoch in trange(num_epochs):
        # train one epoch

        model.train(True)

        running_train_loss: float = 0.
        running_val_loss:float = 0.

        # train_loss
        for data in train_loader:
            X, y = data
            X, y = X.to(device), y.to(device)
            y_hat = model(X)

            optimizer.zero_grad()

            loss = loss_function(y_hat, y.long())
            running_train_loss += loss.item()

            loss.backward()
            optimizer.step()
        
        train_loss = running_train_loss / len(train_loader)


        # validation loss
        model.eval()

        with torch.no_grad():
            for vdata in val_loader:
                vX, vy = vdata
                vX, vy = vX.to(device), vy.to(device)
                vy_hat = model(vX)

                loss = loss_function(vy_hat, vy.long())
                running_val_loss += loss.item()

        val_loss = running_val_loss / len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # compute accuracies
        train_accuracy, _ = compute_accuracy(model, train_loader, device)
        val_accuracy, _ = compute_accuracy(model, val_loader, device)
        test_accuracy, _ = compute_accuracy(model, test_loader, device)

        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        test_accuracies.append(test_accuracy)

        # save best model
        if best_model:
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
    
        if early_stopper is not None:
            if early_stopper(val_loss):
                break
        
    # load best state
    if best_model:
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
    return (train_losses, val_losses), (train_accuracies, val_accuracies, test_accuracies)

# MODEL

class CNN(nn.Module):
    def __init__(self, kernel_size=3, batch_norm=False, dropout=0, conv=False, linear=False):
        super(CNN, self).__init__()

        self.device = torch.device('cpu')
        self.batch_norm = batch_norm  # boolean
        self.dropout = dropout # probability
        self.kernel_size = kernel_size
        self.linear = False
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