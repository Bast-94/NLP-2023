import pandas as pd
import fasttext
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

def save_to_file(df: pd.DataFrame, file_name: str) -> None:
    """
    Save data of a pandas DataFrame to a file.
    Args:
        df: pandas DataFrame with column 'text'
        file_name: name of the file to save
    """
    with open(file_name, 'w') as f:
        f.write(df.to_string(header=False, index=False))

def display_results(resuls: tuple[float]) -> None:
    """
    Display accuracy of the model.
    Args:
        results(tuple[float]): Tuple of (number of examples, precision, recall)
    """
    print(f'Number of examples: {resuls[0]}')
    print(f'Precision: {resuls[1] * 100:.2f}%')
    print(f'Recall: {resuls[2]}')

def display_model_attributes(model: fasttext.FastText._FastText, parameters: list[str]) -> None:
    """
    Display model attributes.
    Args:
        model(fasttext.FastText._FastText): Trained model.
        parameters(list[str]): List of parameters to display.
    """
    for attr in parameters:
        print(f'{attr}: {getattr(model, attr)}')
        