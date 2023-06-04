import torch
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from typing import Iterable, List, Dict, Callable
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor
from torch import nn

from scripts.preprocessing.token_embedding import TokenEmbedding
from scripts.utils import *
from scripts.model.transformer import Seq2SeqTransformer

from tqdm.auto import tqdm

def train_epoch(model: Seq2SeqTransformer,
                optimizer: torch.optim.Adam,
                source_language: str,
                target_language: str,
                loss_fn: torch.nn.CrossEntropyLoss,
                batch_size: int,
                pad_idx: int,
                text_transform: Dict[str, Callable],
                device: str = "cpu"
                ) -> float:
    """
    Train the model for one epoch on the training set.
    Args:
        model (Seq2SeqTransformer): Transformer model.
        optimizer (torch.optim.Adam): Optimizer for the model.
        source_language (str): Source language.
        target_language (str): Target language.
        loss_fn (torch.nn.CrossEntropyLoss): Loss function.
        batch_size (int): Batch size.
        pad_idx (int): Index for padding token.
        text_transform (Dict[str, Callable]): Dictionary with tokenization functions.
        device (str): Device to use for training.
    Returns:
        float: Average loss on the training set.
    """
    model.train()
    losses = 0
    train_iter = Multi30k(split='train', language_pair=(source_language, target_language))
    train_dataloader = DataLoader(train_iter,
                                  batch_size=batch_size,
                                  collate_fn=collate_fn(pad_idx,
                                                        source_language=source_language,
                                                        target_language=target_language,
                                                        text_transform=text_transform)
                                  )

    for src, tgt in train_dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, pad_idx=pad_idx, device=device)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(list(train_dataloader))

def train(model: Seq2SeqTransformer,
            optimizer: torch.optim.Adam,
            source_language: str,
            target_language: str,
            loss_fn: torch.nn.CrossEntropyLoss,
            batch_size: int,
            pad_idx: int,
            text_transform: Dict[str, Callable],
            n_epochs: int,
            device: str = "cpu"
            ) -> tuple[List[float], List[float]]:
    """
    Train the model for a given number of epochs.
    Args:
        model (Seq2SeqTransformer): Transformer model.
        optimizer (torch.optim.Adam): Optimizer for the model.
        source_language (str): Source language.
        target_language (str): Target language.
        loss_fn (torch.nn.CrossEntropyLoss): Loss function.
        batch_size (int): Batch size.
        pad_idx (int): Index for padding token.
        text_transform (Dict[str, Callable]): Dictionary with tokenization functions.
        n_epochs (int): Number of epochs to train.
        device (str): Device to use for training.
    Returns:
        List[float]: List of average losses on the training set for each epoch.
    """
    train_losses = []
    val_losses = []
    for epoch in tqdm(range(1, n_epochs + 1)):
        loss = train_epoch(model, optimizer, source_language, target_language, loss_fn, batch_size, pad_idx, text_transform, device)
        train_losses.append(loss)

        val_loss = evaluate(model, source_language, target_language, loss_fn, batch_size, pad_idx, text_transform, device)
        val_losses.append(val_loss)

        print(f"Epoch: {epoch}, Train loss: {loss:.3f}, Val loss: {val_loss:.3f}")

    return (train_losses, val_losses)

def evaluate(model: Seq2SeqTransformer,
             source_language: str,
             target_language: str,
             loss_fn: torch.nn.CrossEntropyLoss,
             batch_size: int,
             pad_idx: int,
             text_transform: Dict[str, Callable],
             device: str = "cpu"):
    """
    Evaluate the model on the validation set.
    Args:
        model (Seq2SeqTransformer): Transformer model.
        source_language (str): Source language.
        target_language (str): Target language.
        loss_fn (torch.nn.CrossEntropyLoss): Loss function.
        batch_size (int): Batch size.
        pad_idx (int): Index for padding token.
        text_transform (Dict[str, Callable]): Dictionary with tokenization functions.
        device (str): Device to use for training.
    Returns:
        float: Average loss on the validation set.
    """
    model.eval()
    losses = 0

    val_iter = Multi30k(split='valid', language_pair=(source_language, target_language))
    val_dataloader = DataLoader(val_iter, batch_size=batch_size, collate_fn=collate_fn(pad_idx,
                                                                                       source_language=source_language,
                                                                                       target_language=target_language,
                                                                                       text_transform=text_transform)
                                )

    for src, tgt in val_dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, pad_idx=pad_idx, device=device)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(val_dataloader))