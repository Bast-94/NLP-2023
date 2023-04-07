import pandas as pd
import time
from sklearn.linear_model import LogisticRegression

def model(penalty: str ='l2', C=0.001) -> LogisticRegression:
    """
    Creates a LogisticRegression model.
    Args:
        penalty (str): The penalty to use.
        C (float): The regularization strength.
    Returns:
        model (LogisticRegression): The created model.
    """
    return LogisticRegression(penalty=penalty, C=C)

def fit(model: LogisticRegression, X: pd.DataFrame, y: pd.Series) -> None:
    """
    Fits the model to the training data.
    Args:
        model (LogisticRegression): The model to fit.
        X (pd.DataFrame): The training data.
        y (pd.Series): The training labels.
    """
    # Compute the run time of the training loop
    start: float = time.time()

    # Fit the model
    model.fit(X, y)

    # Compute the run time of the training loop
    end: float = time.time()
    print(f"Training time: {end - start:.2f} seconds")

def evaluate(model: LogisticRegression, X: pd.DataFrame, y: pd.Series, type: str) -> float:
    """
    Evaluates the model on the test set.
    Args:
        model (LogisticRegression): The model to evaluate.
        X (pd.DataFrame): The test data.
        y (pd.Series): The test labels.
    Returns:
        accuracy (float): The accuracy of the model.
    """
    accuracy: float = model.score(X, y)
    print(f"{type} accuracy: {accuracy * 100:.2f}%")
    return accuracy