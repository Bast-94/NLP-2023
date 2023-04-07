import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import matplotlib.pyplot as plt

import time
from scripts.logistic_regression.model import LogisticRegression

def fit(model: LogisticRegression, optimizer, criterion, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> tuple[list[float], list[float]]:
    """
    Fits the model to the training data.
    Args:
        model (LogisticRegression): The model to fit.
        optimizer: The optimizer to use.
        criterion: The loss function to use.
        X_train (pd.DataFrame): The training data.
        y_train (pd.Series): The training labels.
        X_val (pd.DataFrame): The validation data.
        y_val (pd.Series): The validation labels.
    """
    # Compute the run time of the training loop
    start: float = time.time()

    num_epochs: int = 1000

    # Keeping an eye on the losses
    train_losses: list[float] = []
    test_losses: list[float] = []

    # Training loop
    for epoch in range(num_epochs):
        # Set the gradients to zero
        optimizer.zero_grad()

        # Set the training set through the model
        predictions = model(torch.tensor(X_train.values).float())
        loss = criterion(predictions, torch.tensor(y_train.values).float().unsqueeze(1))
        train_losses.append(loss.item())
        if epoch % 100 == 0:
            print(loss)
        
        # Computing the gradients and gradient descent.
        loss.backward()
        optimizer.step()
        
        # Validation: When computing the validation loss, we do not want to update the weights.
        # torch.no_grad tells PyTorch to not save the necessary data used for
        # gradient descent.
        
        with torch.no_grad():
            predictions = model(torch.tensor(X_val.values).float())
            loss = criterion(predictions, torch.tensor(y_val.values).float().unsqueeze(1))
            test_losses.append(loss.item())

    # Compute the run time of the training loop
    end: float = time.time()
    print(f"Training time: {end - start:.2f} seconds")

    return train_losses, test_losses

def accuracy(predictions, labels) -> float:
    """
    Computes the accuracy of the model.
    Args:
        predictions: the output of the model.
        labels: the ground truth labels.
    Returns:
        The accuracy of the model.
    """
    predictions = predictions.squeeze(1)
    predictions = torch.round(torch.sigmoid(predictions))
    return (predictions == labels).sum().item() / len(labels)

def display_losses(training: list[float], testing: list[float]) -> None:
    """
    Displays the given losses in a single plot.
    Args:
        training (list[float]): The training losses.
        testing (list[float]): The testing losses.
    """
    plt.plot(training, label="Training loss")
    plt.plot(testing, label="Testing loss")
    plt.legend(frameon=False)
    plt.show()

def evaluate(type: str, model: LogisticRegression, X: pd.DataFrame, y: pd.Series) -> None:
    """
    Displays the accuracy of the model ('model') on the given data.
    Args:
        type (str): The type of data.
        model (LogisticRegression): The model to use.
        X (pd.DataFrame): The data.
        y (pd.Series): The labels.
    """
    with torch.no_grad():
        predictions = model(torch.tensor(X.values).float())
        print(f"{type} accuracy: {accuracy(predictions, torch.tensor(y.values).float()) * 100:.2f}%")