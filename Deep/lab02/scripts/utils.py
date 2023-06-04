import torch
from typing import Iterable, List, Dict, Callable

from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k

from torch.nn.utils.rnn import pad_sequence

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

def yield_tokens(token_transform: Dict[str, Callable], data_iter: Iterable, language: str, source_language: str, target_language: str) -> List[str]:
    """
    Yield tokens from dataset whose source/target language is ``language``.
    Args:
        token_transform (Dict[str, Callable]): Dictionary with tokenization functions.
        data_iter (Iterable): Iterable dataset to yield sentences from.
        language (str): Language of the sentences in ``data_iter``.
        source_language (str): Source language.
        target_language (str): Target language.
    Yields:
        List[str]: List of tokens corresponding to a sentence.
    """
    language_index = {source_language: 0, target_language: 1}

    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])

def build_vocabulary(vocab_transform: Dict[str, Callable], token_transform: Dict[str, Callable], source_language: str, target_language: str, unknown_index: int, special_symbols: list[str]) -> None:
    """
    Build vocabulary for source and target language.
    Args:
        vocab_transform (Dict[str, Callable]): Dictionary
        token_transform (Dict[str, Callable]): Dictionary with tokenization functions.
        source_language (str): Source language.
        target_language (str): Target language.
        unknown_index (int): Index for unknown token.
        special_symbols (list[str]): List of special symbols.
    """
    for ln in [source_language, target_language]:
        # Training data Iterator
        train_iter = Multi30k(split='train', language_pair=(source_language, target_language))
        # Create torchtext's Vocab object
        vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(token_transform, train_iter, ln, source_language, target_language),
                                                        min_freq=1,
                                                        specials=special_symbols,
                                                        special_first=True)

    # Set ``UNK_IDX`` as the default index. This index is returned when the token is not found.
    # If not set, it throws ``RuntimeError`` when the queried token is not found in the Vocabulary.
    for ln in [source_language, target_language]:
        vocab_transform[ln].set_default_index(unknown_index)

def generate_square_subsequent_mask(sz: int, device: str = "cpu") -> torch.Tensor:
    """
    Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    Args:
        sz (int): Size of mask.
        device (str): Device to use for training.
    Returns:
        torch.Tensor: Mask tensor.
    """
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, pad_idx: int, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate masks for src and tgt sequences.
    Args:
        src (torch.Tensor): Source
        tgt (torch.Tensor): Target
        pad_idx (int): Index for padding token.
        device (str): Device to use for training.
    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of masks.
    """
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device=device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = (src == pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def sequential_transforms(*transforms, bos_idx: int, eos_idx: int):
    """
    Compose multiple transforms sequentially.
    Args:
        *transforms: Multiple transforms.
    Returns:
        Callable: Sequentially composes multiple transforms.
    """
    def func(txt_input: str) -> Callable:
        for transform in transforms:
            if transform == tensor_transform:
                txt_input = transform(txt_input, bos_idx=bos_idx, eos_idx=eos_idx)
            else:
                txt_input = transform(txt_input)
        return txt_input
    return func

def tensor_transform(token_ids: List[int], bos_idx: int, eos_idx: int) -> torch.Tensor:
    """
    Turn tokenized text into a tensor of token indices.
    Args:
        token_ids (List[int]): List of token ids to be converted to tensor.
        bos_idx (int): Beginning of sentence index.
        eos_idx (int): End of sentence index.
    Returns:
        torch.Tensor: Tensor of token indices.
    """
    return torch.cat((torch.tensor([bos_idx]),
                      torch.tensor(token_ids),
                      torch.tensor([eos_idx])))

def collate_fn(pad_idx: int, source_language: str, target_language: str, text_transform: Dict[str, Callable]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Function to collate data samples into batch tensors.
    Args:
        pad_idx (int): Index for padding token.
        source_language (str): Source language.
        target_language (str): Target language.
        text_transform (Dict[str, Callable]): Dictionary with tokenization functions.
    Returns:
        tuple[torch.Tensor, torch.Tensor]: Tuple of source and target batch tensors.
    """
    def batch_collate_fn(batch: Iterable) -> tuple[torch.Tensor, torch.Tensor]:
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(text_transform[source_language](src_sample.rstrip("\n")))
            tgt_batch.append(text_transform[target_language](tgt_sample.rstrip("\n")))

        src_batch = pad_sequence(src_batch, padding_value=pad_idx)
        tgt_batch = pad_sequence(tgt_batch, padding_value=pad_idx)
        return src_batch, tgt_batch
    
    return batch_collate_fn