from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import regex as re
import multiprocessing as mp
import cProfile

from pretokenized_examples import find_chunk_boundaries

 PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
 NUM_PROCESSES = 4

 def split_file_on_tokens(text, tokens) -> list[str]:
    """Split text on any token from the list"""
    # Escape special regex characters and join with |
    pattern = '|'.join(re.escape(token) for token in tokens)
    return re.split(pattern, text)

def pretokenize_chunk(f, start, end) -> dict[tuple[bytes], int]:
    pretokenized_text = {}
    f.seek(start)
    chunk = f.read(end - start).decode("utf-8", errors="ignore")
    split_chunk = split_file_on_tokens(chunk, special_tokens)
    for subchunk in split_chunk:
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        pretokenized_iter = re.finditer(PAT, subchunk)
        for match in pretokenized_iter:
            pretokenized_text[tuple(bytes(match.group(), "utf-8"))] = pretokenized_text.get(tuple(bytes(match.group(), "utf-8")), 0) + 1

    return pretokenized_text

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    num_processes = NUM_PROCESSES

    # read file, break into chunks, and pretokenize chunks in parallel
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens[0].encode("utf-8"))

        # pretokenize
        pretokenized_chunks = multiprocess.Pool(num_processes).map(pretokenize_chunk, f, start, end in zip(boundaries[:-1], boundaries[1:]))

    full_pretokenized_text = {}
    for pretokenized_text in pretokenized_chunks:
        for token, count in pretokenized_text.items():
            full_pretokenized_text[token] = full_pretokenized_text.get(token, 0) + count
    
    # train bpe naive implementation
    # For eﬀiciency during BPE training, we do not consider pairs that cross pre-token boundaries.
    vocab = set(full_pretokenized_text.keys()) 
    merges = []
    
    
    while len(vocab) < vocab_size:
        pair_frequency = {}
        for token, count in full_pretokenized_text.items():
            # get pair counts
            for i in range(len(token) - 1):
                pair = token[i] + token[i+1]
                pair_frequency[pair] = pair_frequency.get(pair, 0) + count
        # get most frequent pair
        pair = max(pair_frequency, key=pair_frequency.get)
        # update merges
        merges.append(pair)
        vocab.add(pair)
        # update pair frequency
        pair_frequency[pair] = max(pair_frequency.values())

        # update full pretokenized text
        for token in full_pretokenized_text.items():
            new_token = []
            count = full_pretokenized_text.get(token, 0)
            for i in range(len(token) - 1):
                current_pair = token[i] + token[i+1]
                if current_pair == pair:
                    new_token.append(pair)
                else:
                    new_token.append(token[i])
            full_pretokenized_text[tuple(new_token)] = full_pretokenized_text.get(tuple(new_token), 0) + count
            del full_pretokenized_text[token]

    return vocab, merges

    



           
    

    
            


            


