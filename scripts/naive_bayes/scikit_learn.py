import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def pipeline() -> Pipeline:
    """
    Returns a pipeline.

    Returns:
        Pipeline: Scikit-learn pipeline
    """
    return Pipeline([
        ("vect", CountVectorizer()),
        ("clf", MultinomialNB())
    ])

#-----------------------------#
#       Evaluation            #
#-----------------------------#
def print_classification_report(test_dataset: pd.DataFrame, predictions: list) -> None:
    """
    Prints a classification report.

    Args:
        test_dataset (pd.DataFrame): Test dataset
        predictions (list): Predictions on test dataset
    """
    print(classification_report(test_dataset["label"], predictions))

def print_confusion_matrix(test_dataset: pd.DataFrame, predictions: list) -> None:
    """
    Prints a confusion matrix.

    Args:
        test_dataset (pd.DataFrame): Test dataset
        predictions (list): Predictions on test dataset
    """
    print(confusion_matrix(test_dataset["label"], predictions))

def print_accuracy_score(dataset: pd.DataFrame, predictions: list, set_type: str) -> None:
    """
    Prints the accuracy score.

    Args:
        dataset (pd.DataFrame): Dataset
        predictions (list): Predictions on dataset
    """
    print(f'{set_type} accuracy: {accuracy_score(dataset["label"], predictions) * 100:.2f}%')