import torch
from typing import Dict, Callable

from scripts.utils import generate_square_subsequent_mask
from scripts.model.transformer import Seq2SeqTransformer

def greedy_decode(model: Seq2SeqTransformer,
                  src: torch.Tensor,
                  src_mask: torch.Tensor,
                  max_len: int,
                  start_symbol: int,
                  eos_idx: str,
                  device: str = "cpu") -> torch.Tensor:
    """
    Generate output sequence using greedy algorithm.
    Args:
        model (Seq2SeqTransformer): Transformer model.
        src (torch.Tensor): Source tensor.
        src_mask (torch.Tensor): Source mask.
        max_len (int): Maximum length of output sequence.
        start_symbol (int): Index for start symbol.
        eos_idx (str): Index for end of sentence.
        device (str): Device to use for training.
    Returns:
        torch.Tensor: Generated output sequence.
    """
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len-1):
        memory = memory.to(device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0), device=device)
                    .type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == eos_idx:
            break
    return ys


def greedy_translate(model: Seq2SeqTransformer,
              src_sentence: str,
              text_transform: Dict[str, Callable],
              vocab_transform: Dict[str, Callable],
              source_language: str,
              target_language: str,
              start_symbol: int,
              eos_idx: int,
              device: str = "cpu") -> str:
    """
    Translate input sentence into target language.
    Args:
        model (Seq2SeqTransformer): Transformer model.
        src_sentence (str): Source sentence.
        text_transform (Dict[str, Callable]): Text transformation pipeline.
        vocab_transform (Dict[str, Callable]): Vocabulary transformation pipeline.
        source_language (str): Source language.
        target_language (str): Target language.
        start_symbol (int): Index for start symbol.
        eos_idx (int): Index for end of sentence.
        device (str): Device to use for training.
    Returns:
        str: Translated sentence.
    """
    model.eval()
    src = text_transform[source_language](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=start_symbol, eos_idx=eos_idx, device=device).flatten()
    return " ".join(vocab_transform[target_language].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")