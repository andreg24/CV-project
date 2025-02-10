import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision

def build_svm_features(model, loader, device):
    """loader should be an alex loader"""
    features = None
    targets = None
    X, y = loader.dataset[:]
    X = X.to(device)
    targets = y.numpy()
    features = model(X)
    features = features.squeeze().view(features.size(0), -1)
    features = features.to('cpu').numpy()
    return features, targets