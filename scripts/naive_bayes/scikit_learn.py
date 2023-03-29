import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from scripts.data import clean_html

def pipeline(stop_words: list[str] = []) -> Pipeline:
    """
    Returns a pipeline.

    Returns:
        Pipeline: Scikit-learn pipeline
    """
    return Pipeline([
        ("vect", CountVectorizer(stop_words=stop_words)),
        ("clf", MultinomialNB())
    ])

def print_most_likely_words(pipeline: Pipeline) -> None:
    # Get the feature names.
    feature_names: np.ndarray = pipeline.named_steps['vect'].get_feature_names_out()

    # Get the log probabilities.
    log_prob = pipeline.named_steps['clf'].feature_log_prob_

    # Get the index of the most likely words.
    most_likely = np.argsort(log_prob, axis=1)[:, -10:]

    # Print the most likely words.
    for i in range(2):
        print("Class: ", i)
        print("Most likely words: ", [feature_names[j] for j in most_likely[i]])
        print("")

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