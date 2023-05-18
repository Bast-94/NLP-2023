from datasets import load_dataset
import datasets as ds
from torchtext.vocab import Vocab, vocab
from torchtext.data.utils import get_tokenizer
from typing import Callable, Dict, Generator, List, Tuple
from collections import Counter
import torch
import torch.nn as nn
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm.auto import tqdm

def load_imdb_datasets() -> tuple[ds.Dataset]:
    """
    Loads the IMDB dataset from the datasets library.
    Splits the train dataset into train, validation and test sets.
    Returns:
        datasets: list[ds.Dataset] - List of datasets
    """
    dataset = load_dataset("imdb")
    train_dataset = dataset["train"].train_test_split(
        stratify_by_column="label", test_size=0.2, seed=42
    )
    test_df = dataset["test"]
    train_df = train_dataset["train"]
    valid_df = train_dataset["test"]

    return (train_df, valid_df, test_df)

def build_vocabulary(dataset: ds.Dataset, tokenizer: Callable) -> Vocab:
    """
    Builds a vocabulary from a dataset.
    Args:
        dataset: ds.Dataset - Dataset to build vocabulary from
    Returns:
            vocab: Vocab - Vocabulary
    """
    tokens = tokenizer(" ".join(dataset["text"]))
    counter = Counter(tokens)
    vocabulary = vocab(counter, min_freq=10, specials=["<unk>", "<pad>"])
    vocabulary.set_default_index(vocabulary["<unk>"])

    return vocabulary

def vectorize_text(
    text: str, vocabulary: Vocab, tokenizer: Callable[[str], List[str]]
) -> torch.Tensor:
    """
    Generate a tensor of vocabluary IDs for a given text.
    Args:
        text: the input text.
        vocabulary: a Vocab objects.
        tokenizer: a text tokenizer.
    Returns:
        A tensor of IDs (torch.long).
    """
    return torch.tensor(
        [vocabulary[token] for token in tokenizer(text)], dtype=torch.long
    )

def data_generator(
    X: List[torch.tensor], y: List[int], pad_id: int, batch_size: int = 32
) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    """
    Yield batches from given input data and labels.
    Args:
        X: a list of tensor (input features).
        y: the corresponding labels.
        batch_size: the size of every batch [32].
    Returns:
        A tuple of tensors (features, labels).
    """
    X, y = shuffle(X, y)
    for i in range(0, len(X), batch_size):
        X_batch = nn.utils.rnn.pad_sequence(
            X[i : i + batch_size], batch_first=True, padding_value=pad_id
        )
        y_batch = torch.tensor(y[i : i + batch_size], dtype=torch.float32)
        yield X_batch, y_batch

def get_device() -> str:
    """
    Returns the device to use for training.
    Returns:
        device: str - Device to use for training
    """
    # Define CPU as default device
    device = "cpu"

    # Use Cuda acceleration if available (Nvidia GPU)
    if torch.cuda.is_available():
        device = "cuda:0"
    # Use Metal acceleration if available (MacOS)
    elif torch.backends.mps.is_available():
        device = "mps:0"
    
    return device

def evaluate(
    model: nn.Module,
    generator: Generator[Tuple[torch.Tensor, torch.Tensor], None, None],
    device: str,
) -> Tuple[float, float, float, float]:
    """
    Evaluate a model on a given dataset.
    Args:
        model: a pytorch module.
        criterion: a loss function.
        generator: a generator of data.
        device: the device to use.
    Returns:
        A tuple of accuracy, precision, recall and f1-score.
    """
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in generator():
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_true.extend(y_batch.tolist())
            y_pred.extend(model(X_batch).round().squeeze(1).tolist())
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
    
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-score: {f1:.3f}")
    return accuracy, precision, recall, f1


def train(
    model: nn.Module,
    X_train: List[torch.Tensor],
    X_val: List[torch.Tensor],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_generator: Generator[Tuple[torch.Tensor, torch.Tensor], None, None],
    valid_generator: Generator[Tuple[torch.Tensor, torch.Tensor], None, None],
    n_epochs: int,
    device: str,
) -> Tuple[List[float], List[float]]:
    """
    Train a model on a given dataset.
    Args:
        model: a pytorch module.
        criterion: a loss function.
        optimizer: an optimizer.
        train_generator: a generator of training data.
        valid_generator: a generator of validation data.
        n_epochs: the number of epochs.
        device: the device to use.
    Returns:
        A tuple of lists of train and validation losses.
    """
    train_losses = []
    valid_losses = []
    best_valid_loss = float("inf")
    for epoch in range(n_epochs):
        train_loss = 0.0
        model.train()
        for X_batch, y_batch in tqdm(train_generator()):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(X_train)
        valid_loss = 0.0
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in valid_generator():
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch.unsqueeze(1))
                valid_loss += loss.item()
            valid_loss /= len(X_val)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "model.pt")
        print(
            f"Epoch {epoch+1}/{n_epochs} train loss: {train_loss:.3f} valid loss: {valid_loss:.3f}"
        )
    return train_losses, valid_losses