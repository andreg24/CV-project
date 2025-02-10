import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import os

from sklearn.metrics import confusion_matrix


__all__ = [
    # "plot_loss",
    # "plot_accuracy",
    "plot_data",
    "plot_confusion",
]

def plot_loss(losses):
    """Function plotting the losses along the epochs on the train and validation dataset"""

    train_losses, validation_losses = losses
    num_epochs = len(train_losses)

    plt.plot(train_losses, color='r', label='train')
    plt.plot(validation_losses, color='b', label='validation')
    plt.xlabel = 'epochs'
    plt.ylabel = 'loss'
    plt.title('Loss plot')
    plt.legend()

def plot_accuracy(accuracies, goals=[], test=False):
    """Function plotting the accuracy over the epochs on the train and validation dataset"""

    train_accuracy, validation_accuracy, test_accuracy = accuracies

    num_epochs = len(train_accuracy)

    plt.plot(train_accuracy, color='r', label='train')
    plt.plot(validation_accuracy, color='b', label='validation')
    if test:
        plt.plot(test_accuracy, color='g', label='Test')
    if len(goals) > 0:
        for goal in goals:
            plt.hlines(goal, xmin=0, xmax=num_epochs, label=f'{goal}% goal', linestyles='--')
    plt.xlabel = 'epochs'
    plt.ylabel = 'accuracy%'
    plt.ylim(0, 100)
    plt.yticks([int(10*i) for i in range(0, 11)])
    plt.title('Accuracy plot')
    plt.legend()

def plot_data(losses, accuracies, goals=[40, 60], test=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # losses
    train_losses, validation_losses = losses
    num_epochs = len(train_losses)

    ax1.plot(train_losses, color='r', label='train')
    ax1.plot(validation_losses, color='b', label='validation')
    ax1.set_xlabel('epochs') 
    ax1.set_ylabel('loss')
    ax1.set_title('Loss plot')
    ax1.legend()

    # accuracies
    train_accuracy, validation_accuracy = accuracies[:2]
    test_accuracy = accuracies[2] if len(accuracies) > 2 else None

    ax2.plot(train_accuracy, color='r', label='train')
    ax2.plot(validation_accuracy, color='b', label='validation')
    if test and test_accuracy is not None:
        ax2.plot(test_accuracy, color='g', label='Test')

    if len(goals) > 0:
        for goal in goals:
            ax2.hlines(goal, xmin=0, xmax=num_epochs - 1, label=f'{goal}% goal', linestyles='--')

    ax2.set_xlabel('epochs')
    ax2.set_ylabel('accuracy%')
    ax2.set_ylim(0, 100)
    ax2.set_yticks([int(10 * i) for i in range(11)])
    ax2.set_title('Accuracy plot')
    ax2.legend()

    plt.show()


def plot_confusion(y, y_hat):
    cm = confusion_matrix(y, y_hat)
    labels = ['Bedroom',
              'Coast',
              'Forest',
              'Highway',
              'Industrial',
              'InsideCity',
              'Kitchen',
              'LivingRoom',
              'Mountain',
              'Office',
              'OpenCountry',
              'Store',
              'Street',
              'Suburb',
              'TallBuilding']
    
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=90)
    plt.title('Confusion matrix')
    plt.show()