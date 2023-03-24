import pandas as pd
import numpy as np
import re
from collections import Counter

def tokenize(text: str)-> list:
    """
    Splits the given text into tokens.
    Args:
        text (str): Text to tokenize (pre-processed)
    Returns:
        list: List of tokens
    """
    return [w for w in re.split("\W+", text)]

def build_vocabulary(texts_serie: pd.Series) -> Counter:
    """
    Builds the vocabulary of the given texts serie.
    Args:
        text_serie (pd.Series): Text serie
    Returns:
        Counter: Vocabulary
    """
    vocabulary: Counter = None # Use Counter as a dictionary with word occurrences
    for text in texts_serie:
        word_list: list[str] = tokenize(text=text)
        if vocabulary is None:
            vocabulary = Counter(word_list)
        else:
            vocabulary.update(word_list)
    return vocabulary

def word_count(counter_class: pd.DataFrame, class_name: str, word: str) -> int:
    """
    Returns the number of occurrences of the given word in the given class.
    Args:
        counter_class (pd.DataFrame): DataFrame with the vocabulary of each class
        class_name (str): Class name / label
        word (str): Word
    Returns:
        int: Number of occurrences of the given word in the given class
    """
    return counter_class.loc[class_name]["text"][word]

def total_words(vocabulary: Counter, c: str, counter_class: pd.DataFrame) -> int:
    """
    Returns the total number of words in a class for the given vocabulary.
    Args:
        vocabulary (Counter): Vocabulary
        c (str): Class name / label
        counter_class (pd.DataFrame): DataFrame with the vocabulary of each class
    Returns:
        int: Total number of words in the given class
    """
    total: int = 0
    for w in vocabulary:
        total += word_count(counter_class, c, w)
    return total

def fill_loglikelihood(loglikelihood: dict, word: str, class_value: str, value_to_affect: float) -> None:
    """
    Fills the loglikelihood dictionary with the given values.
    Args:
        loglikelihood (dict): Loglikelihood dictionary
        word (str): Word
        class_value (str): Class name / label
        value_to_affect (float): Value to affect
    """
    if (loglikelihood.get(word) is None):
        loglikelihood[word] = {}
    loglikelihood[word][class_value] = value_to_affect

def classifier(train_data_frame: pd.DataFrame, vocabulary: Counter, counter_class: pd.DataFrame) -> tuple[dict, dict, Counter]:
    """
    Builds the Naive Bayes classifier.
    Args:
        train_data_frame (pd.DataFrame): Training data frame
        vocabulary (Counter): Vocabulary
        counter_class (pd.DataFrame): DataFrame with the vocabulary of each class
    Returns:
        tuple[dict, dict, Counter]: Tuple with the logprior, loglikelihood and vocabulary
    """
    total_document_count: int = train_data_frame.text.count()
    class_label_set: list = list(train_data_frame.groupby("label").groups.keys())
    logprior: dict = {}
    loglikelihood: dict = {}
    
    for current_class in class_label_set:
        class_document_count: int = train_data_frame[train_data_frame.label == current_class].text.count()
        logprior[current_class] = np.log(class_document_count/total_document_count)
        total: int = total_words(vocabulary,current_class,counter_class) + len(vocabulary)
        
        for word in vocabulary:
            count_w_c = word_count(counter_class, current_class, word) + 1
            log_like_value = np.log(count_w_c / total)
            fill_loglikelihood(loglikelihood,word,current_class,log_like_value)
            
    return logprior, loglikelihood, vocabulary

def test_classifier(testdoc: str, logprior: dict, loglikelihood: dict, train_data_frame: pd.DataFrame, vocabulary: Counter) -> tuple:
    """
    Tests the Naive Bayes classifier.
    Args:
        testdoc (str): Test document
        logprior (dict): Logprior
        loglikelihood (dict): Loglikelihood
        train_data_frame (pd.DataFrame): Training data frame
        vocabulary (Counter): Vocabulary
    Returns:
        tuple: Tuple with the predicted class and the loglikelihood
    """
    class_set: list = list(train_data_frame.groupby("label").groups.keys())
    sums: dict = {}
    max_class = None

    for c in class_set:
        sums[c] = logprior[c]
        word_list = tokenize(testdoc)
        for w in word_list:
            if(vocabulary[w] != 0):
                sums[c] = sums[c] + loglikelihood[w][c]

        if (max_class is None or sums[max_class] < sums[c]):
            max_class = c
        
    return max_class

def display_results(dataframe: pd.DataFrame, dataset_name: str) -> None:
    """
    Displays the results of the Naive Bayes classifier.
    Args:
        dataframe (pd.DataFrame): Data frame
    """
    good_predictions_count: int = (dataframe[dataframe.label == dataframe.model_result]).label.count()
    text_count: int = dataframe.text.count()
    accuracy: float = (good_predictions_count / text_count) * 100
    print(f"{dataset_name} accuracy: {accuracy:.2f}%")
