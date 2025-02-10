import numpy as np
import matplotlib.pyplot as plt

import os
from datetime import datetime
import copy
import random
from tqdm import trange

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.io import read_image

from rich import print

__all__ = [
    'prepare_images',
    'ImageDataset',
    'build_loaders',
]

# transform to get images for normal CNN
standard_transform = transforms.Compose([
    transforms.Resize((64, 64)),
])

# transform for images for alexnet
alex_transform = transforms.Compose([
    lambda x: x.repeat(3, 1, 1),
    transforms.Resize((224, 224)),
    lambda x: x/255,
    transforms.Normalize(
        mean=[0.4559, 0.4559, 0.4559],
        std=[0.2355, 0.2355, 0.2355]
    )
])

class ImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        """
        Args:
            images (torch.Tensor): Tensor of shape (N, 1, 64, 64).
            labels (torch.Tensor): Tensor of shape (N,).
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Retrieve the image and label at the given index
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)
        return image, label
    

def load_raw_images(root_dir='images/train', load_transform=None):
    """"""

    images = []
    labels = []

    for class_index, class_name in enumerate(os.listdir(root_dir)): # for each class
        class_folder = os.path.join(root_dir, class_name)
        for image_name in os.listdir(class_folder):  # for each image
            image_path = os.path.join(class_folder, image_name)

            image = read_image(image_path)
            image = load_transform(image)
            images.append(image)
            labels.append(class_index)

    images = torch.stack(images)
    labels = torch.tensor(labels)

    return images, labels

def split_train_val(X, y, split_ratio=0.85, indices=None):
    """Given images X and labels y, divides them into train and validation with the indicated splitting ratio. 
    It returns the splitting plus the splitting indexes"""

    if indices is None:
        n = len(X)

        train_size = int(split_ratio * n)

        val_size = n - train_size
        indices = torch.randperm(n)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
    else:
        train_indices = indices[0]
        val_indices = indices[1]

    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]
    return X_train, X_val, y_train, y_val, (train_indices, val_indices)

def prepare_images():
    """Recover images for cnn and alexnet by applying the correct transformation and returning images and labels as tensors"""
    # cnn images
    X, y = load_raw_images(load_transform=standard_transform)
    X_test, y_test = load_raw_images('images/test', load_transform=standard_transform)
    X_train, X_val, y_train, y_val, ind = split_train_val(X, y)

    X_train = X_train.to(dtype=torch.float32)
    X_val = X_val.to(dtype=torch.float32)
    X_test = X_test.to(dtype=torch.float32)

    # alexnet images

    alex_X, alex_y = load_raw_images(load_transform=alex_transform)
    alex_X_test, alex_y_test = load_raw_images('images/test', load_transform=alex_transform)
    alex_X_train, alex_X_val, alex_y_train, alex_y_val, ind = split_train_val(alex_X, alex_y, indices=ind)

    alex_X_train = alex_X_train.to(dtype=torch.float32)
    alex_X_val = alex_X_val.to(dtype=torch.float32)
    alex_X_test = alex_X_test.to(dtype=torch.float32)

    return (X_train, y_train, X_val, y_val, X_test, y_test), (alex_X_train, alex_y_train, alex_X_val, alex_y_val, alex_X_test, alex_y_test)

def build_loaders(train_dataset, val_dataset, test_dataset):
    # Create data loaders
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return (train_loader, val_loader, test_loader)