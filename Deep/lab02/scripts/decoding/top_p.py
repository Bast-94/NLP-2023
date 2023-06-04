import torch
from typing import Dict, Callable

from scripts.model.transformer import Seq2SeqTransformer
from scripts.utils import generate_square_subsequent_mask

def top_p_sampling_decode(model: Seq2SeqTransformer,
                          src: torch.Tensor,
                          src_mask: torch.Tensor,
                          max_len: int,
                          start_symbol: int,
                          eos_idx: int,
                          p=0.9,
                          temperature=1.0,
                          device: str="cpu") -> torch.Tensor:
    """
    Generate output sequence using top-p sampling decoding.
    Args:
        model (Seq2SeqTransformer): Transformer model.
        src (torch.Tensor): Source tensor.
        src_mask (torch.Tensor): Source mask.
        max_len (int): Maximum length of output sequence.
        start_symbol (int): Index for start symbol.
        eos_idx (str): Index for end of sentence.
        p (float): Probability threshold for top-p sampling.
        temperature (float): Temperature for top-p sampling.
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
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        logits = model.generator(out[:, -1])

        # Apply temperature
        logits = logits / temperature

        # Convert logits to probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # Sort probabilities
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        # Get the smallest set of tokens whose cumulative probability exceeds p
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Create a new probabilities tensor to sample from
        new_probs = probs.clone()
        new_probs.squeeze()[sorted_indices[sorted_indices_to_remove]] = 0

        next_word = torch.multinomial(new_probs, num_samples=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == eos_idx:
            break
    return ys

def translate_top_p(model: Seq2SeqTransformer,
                    src_sentence: str,
                    text_transform: Dict[str, Callable],
                    vocab_transform: Dict[str, Callable],
                    source_language: str,
                    target_language: str,
                    start_symbol: int,
                    eos_idx: int,
                    p=0.9,
                    temperature=1.0,
                    device: str = "cpu") -> str:
    """
    Translate a source sentence to a target sentence using top-p sampling decoding.
    Args:
        model (Seq2SeqTransformer): A trained sequence to sequence transformer model.
        src_sentence (str): A source sentence to translate.
        text_transform (Dict[str, Callable]): A dictionary containing the text transforms for the source and target languages.
        vocab_transform (Dict[str, Callable]): A dictionary containing the vocabulary transforms for the source and target languages.
        source_language (str): The source language.
        target_language (str): The target language.
        start_symbol (int): The start symbol.
        eos_idx (int): The end of sentence index.
        p (float): The cumulative probability threshold.
        temperature (float): The temperature.
        device (str): The device to run the model on.
    Returns:
        str: The translated sentence.
    """
    model.eval()
    src = text_transform[source_language](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = top_p_sampling_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=start_symbol, eos_idx=eos_idx, p=p, temperature=temperature, device=device).flatten()
    return " ".join(vocab_transform[target_language].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")