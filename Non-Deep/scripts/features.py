import string
import pandas as pd
import numpy as np
import datasets as ds
import warnings

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ignore warnings from BeautifulSoup
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

def get_features(text: str, sid: SentimentIntensityAnalyzer) -> list[float]:
    """
    Extracts features from the given text.
    Args:
        text (str): Text to extract features from.
        sid (SentimentIntensityAnalyzer): SentimentIntensityAnalyzer object.
    Returns:
        features (list[float]): List of features.
    """
    # Preprocessing
    text = text.lower()
    text = BeautifulSoup(text, "html.parser").get_text()
    text = "".join([c for c in text if c not in string.punctuation or c == "!" or c == "."])

    features: list[float] = []
    positive: int = 0
    negative: int = 0

    # Features
    features.append(1 if "no" in text.split() else 0)
    features.append(text.split().count("i") + text.split().count("we"))
    features.append(text.split().count("you"))
    features.append(1 if "!" in text else 0)
    features.append(np.log(len(text.split())))

    for word in text.split():
        if sid.lexicon.get(word, 0) < -1:
            negative += 1
        elif sid.lexicon.get(word, 0) > 1:
            positive += 1

    features.append(positive)
    features.append(negative)

    # Additional feature(s)
    features.append(text.split().count("it") +\
                    text.split().count("he") +\
                    text.split().count("she") +\
                    text.split().count("they"))
    
    return features


def add_features(dataset: ds.Dataset, sid: SentimentIntensityAnalyzer) -> pd.DataFrame:
    """
    Adds features to the given dataset.
    Args:
        dataset (ds.Dataset): Dataset to add features to.
    Returns:
        df (pd.DataFrame): Dataframe with added features (as columns).
    """
    # Convert to dataframe
    df: pd.DataFrame = pd.DataFrame(dataset)

    # Apply features to dataset as new columns
    df["no"] = df["text"].apply(lambda x: get_features(x, sid=sid)[0])
    df["first_pronouns"] = df["text"].apply(lambda x: get_features(x, sid=sid)[1])
    df["second_pronouns"] = df["text"].apply(lambda x: get_features(x, sid=sid)[2])
    df["exclamation"] = df["text"].apply(lambda x: get_features(x, sid=sid)[3])
    df["log_word_count"] = df["text"].apply(lambda x: get_features(x, sid=sid)[4])
    df["positive"] = df["text"].apply(lambda x: get_features(x, sid=sid)[5])
    df["negative"] = df["text"].apply(lambda x: get_features(x, sid=sid)[6])
    df["third_person"] = df["text"].apply(lambda x: get_features(x, sid=sid)[7])

    # Delete text column
    df = df.drop(columns=["text"])
    # Keep only relevant columns (containing features)
    df = df[["no", "first_pronouns", "second_pronouns", "exclamation", "log_word_count", "positive", "negative", "label", "third_person"]]
    
    return df