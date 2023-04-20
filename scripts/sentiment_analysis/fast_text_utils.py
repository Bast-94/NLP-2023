import pandas as pd
from string import punctuation

def to_fast_text_format(df: pd.DataFrame, label_column_name: str, texts_column_name: str) -> None:
    """
    Convert data of a pandas DataFrame to FastText library format.
    Args:
        df: pandas DataFrame with columns 'label' and 'text'
    """

    mapped_labels = lambda x: 'positive' if x == 1 else 'negative'

    # Append labels to the beginning of the text
    texts: pd.Series = '__label__' + df[label_column_name].apply(mapped_labels)
    # Append text after the label
    texts = texts + ' ' + df[texts_column_name]

    # Replace the original text column with the new formatted text
    df[texts_column_name] = texts

def preprocess_text(text: str) -> str:
    """
    Preprocess text for FastText library. Only lower case and punctuation removal are performed.
    Args:
        text: text to preprocess
    Returns:
        preprocessed text
    """
    # Lower case
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', punctuation))

    return text