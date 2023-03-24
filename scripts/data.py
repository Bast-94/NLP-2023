import pandas as pd
import datasets as ds
from bs4 import BeautifulSoup
import re

def load_datasets(splits: list[str]) -> list[ds.Dataset]:
    """
    Loads the IMDB dataset from the datasets library.
    Returns:
        datasets: list[ds.Dataset] - List of datasets
    """
    datasets: list[ds.Dataset] = []
    for split in splits:
        dataset: ds.Dataset = ds.load_dataset('imdb', split=split)
        datasets.append(dataset)
    
    return datasets

def datasets_to_dataframes(datasets: list[ds.Dataset]) -> list[pd.DataFrame]:
    """
    Converts datasets to dataframes.
    Returns:
        dataframes: list[pd.DataFrame] - List of dataframes
    """
    dataframes: list[pd.DataFrame] = []
    for dataset in datasets:
        dataframe: pd.DataFrame = dataset.to_pandas()
        dataframes.append(dataframe)
    
    return dataframes

def clean_html(text: str) -> str:
    """
    Removes HTML tags from the given text.
    Args:
        text (str): Text with html tags.
    Returns:
        str: Text from all html tags.
    """
    no_html = BeautifulSoup(text).get_text()
    return no_html

def text_processing(text: str) -> str:
  """
  Pre-processes the given text.
  Args:
      text (str): Text to process
  Returns:
      str: Processed text
  """
  result_text = text
  result_text = clean_html(result_text)
  result_text = result_text.lower()
  pattern = r"(?<![a-zA-Z])[^\w\s]|[^\w\s](?![a-zA-Z])"
  result_text = re.sub(pattern, "", result_text)
  result_text = result_text.strip()
  return re.sub("(\s+)", " ", result_text)

def test_preprocessing(input: str, expected: str) -> None:
    """
    Tests the preprocessing function.
    Args:
        input (str): Input text
        expected (str): Expected output
    """
    result: str = text_processing(input)
    assert text_processing(input) == expected or print(result)

def processed_dataframes(dataframes: list[pd.DataFrame]) -> tuple[pd.DataFrame]:
    """
    Pre-processes the given dataframes.
    Args:
        dataframes (list[pd.DataFrame]): List of dataframes
    Returns:
        tuple[pd.DataFrame]: Tuple of pre-processed dataframes
    """
    # Get supervised dataset from the list of dataframes
    train_df: pd.DataFrame = dataframes[0]
    test_df: pd.DataFrame = dataframes[1]

    # Apply pre-processing
    train_df.text = train_df.text.apply(text_processing)
    test_df.text = test_df.text.apply(text_processing)
    
    return train_df, test_df