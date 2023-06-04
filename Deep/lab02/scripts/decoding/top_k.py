import torch
from torch import Tensor
from typing import Dict, Callable

from scripts.model.transformer import Seq2SeqTransformer
from scripts.utils import generate_square_subsequent_mask

def top_k_sampling_decode(model: Seq2SeqTransformer,
                          src: Tensor,
                          src_mask: Tensor,
                          max_len: int,
                          start_symbol: int,
                          k=10,
                          temperature=1.0,
                          eos_idx: int = 0,
                          device: str = "cpu") -> Tensor:
    """
    Generate output sequence using top-k sampling.
    Args:
        model (Seq2SeqTransformer): Transformer model.
        src (Tensor): Source tensor.
        src_mask (Tensor): Source mask.
        max_len (int): Maximum length of output sequence.
        start_symbol (int): Index for start symbol.
        k (int): Number of samples to draw from the distribution.
        temperature (float): Temperature to apply to the logits.
        eos_idx (int): Index for end of sentence.
        device (str): Device to use for training.
    Returns:
        Tensor: Generated output sequence.
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
        logits = model.generator(out[:, -1])
        
        # Apply temperature
        logits = logits / temperature
        
        # Top-k sampling
        indices_to_remove = logits < torch.topk(logits, k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
        
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_word = torch.multinomial(probs, num_samples=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == eos_idx:
            break
    return ys

# actual function to translate input sentence into target language
def translate_k(model: Seq2SeqTransformer,
                src_sentence: str,
                text_transform: Dict[str, Callable],
                vocab_transform: Dict[str, Callable],
                source_language: str,
                target_language: str,
                start_symbol: int,
                eos_idx: int,
                k=10,
                temperature=1.0,
                device: str = "cpu"
                ) -> str:
    """
    Translate input sentence into target language using top-k sampling.
    Args:
        model (Seq2SeqTransformer): Transformer model.
        src_sentence (str): Source sentence.
        text_transform (Dict[str, Callable]): Text transform.
        vocab_transform (Dict[str, Callable]): Vocabulary transform.
        source_language (str): Source language.
        target_language (str): Target language.
        start_symbol (int): Index for start symbol.
        eos_idx (int): Index for end of sentence.
        k (int): Number of samples to draw from the distribution.
        temperature (float): Temperature to apply to the logits.
        device (str): Device to use for training.
    Returns:
        str: Translated target sentence.
    """

    model.eval()
    src = text_transform[source_language](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = top_k_sampling_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=start_symbol, k=k, temperature=temperature, eos_idx=eos_idx, device=device).flatten()
    return " ".join(vocab_transform[target_language].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")