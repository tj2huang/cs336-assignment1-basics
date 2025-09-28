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

from .pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
NUM_PROCESSES = 4

def split_file_on_tokens(text, tokens) -> list[str]:
    """Split text on any token from the list"""
    # Escape special regex characters and join with |
    pattern = '|'.join(re.escape(token) for token in tokens)
    return re.split(pattern, text)

def find_chunk_boundaries_from_bytes(file_content: bytes, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """Find chunk boundaries in file content (similar to find_chunk_boundaries but for bytes)"""
    file_size = len(file_content)
    chunk_size = file_size // desired_num_chunks
    
    # Initial guesses for chunk boundary locations, uniformly spaced
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    
    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time
    
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        position = initial_position
        
        while position < file_size:
            # Get a mini chunk
            end_pos = min(position + mini_chunk_size, file_size)
            mini_chunk = file_content[position:end_pos]
            
            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = position + found_at
                break
            position += mini_chunk_size
    
    # Make sure all boundaries are unique
    return sorted(set(chunk_boundaries))

def pretokenize_chunk_from_bytes(chunk_bytes: bytes, special_tokens: list[str]) -> dict[tuple[bytes], int]:
    """Pretokenize a chunk of bytes. Protekenizing summarizes the chunk into the counts of each pretoken without considering tokenizing across pretokens.
     So we can just summarize the text into a map of pretoken frequencies rather than preserving the order.)"""
    pretokenized_text = {}
    chunk = chunk_bytes.decode("utf-8", errors="ignore")
    split_chunk = split_file_on_tokens(chunk, special_tokens)
    for subchunk in split_chunk:
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        pretokenized_iter = re.finditer(PAT, subchunk)
        for match in pretokenized_iter:
            # Keep the original approach - split into individual bytes
            token_bytes = bytes(match.group(), "utf-8")
            token_tuple = tuple(bytes([byte]) for byte in token_bytes)
            pretokenized_text[token_tuple] = pretokenized_text.get(token_tuple, 0) + 1
    
    return pretokenized_text

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    num_processes = NUM_PROCESSES

    # Read the entire file content first
    with open(input_path, "rb") as f:
        file_content = f.read()
    
    # Find boundaries in the file content
    #FIX THIS FROM HARDCODING ONE SPECIAL TOKEN
    boundaries = find_chunk_boundaries_from_bytes(file_content, num_processes, special_tokens[0].encode("utf-8"))
    
    # Create chunks of data to process
    chunk_data = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        chunk_data.append(file_content[start:end])
    
    # Process chunks in parallel
    with mp.Pool(num_processes) as pool:
        pretokenized_chunks = pool.starmap(pretokenize_chunk_from_bytes, [(chunk, special_tokens) for chunk in chunk_data])

        full_pretokenized_text = {}
        for pretokenized_text in pretokenized_chunks:
            for token, count in pretokenized_text.items():
                full_pretokenized_text[token] = full_pretokenized_text.get(token, 0) + count
    
    # train bpe naive implementation
    # For eﬀiciency during BPE training, we do not consider pairs that cross pre-token boundaries.
    vocab = {bytes([i]) for i in range(256)} | {token.encode('utf-8') for token in special_tokens}
    merges = []
    
    
    # naive implementation, full iteration
    while len(vocab) < vocab_size:
        pair_frequency = {}
        for token, count in full_pretokenized_text.items():

            # token is a tuple of bytes
            # increase pair frequency by 1
            for i in range(len(token) - 1):
                pair = (token[i], token[i+1])
                pair_frequency[pair] = pair_frequency.get(pair, 0) + count
        # get most frequent pair
        pair = max(pair_frequency, key=lambda p: (pair_frequency[p], p))
        # update merges as a tuple of bytes
        merges.append(pair)

        merged_bytes = pair[0] + pair[1]  # Concatenate the bytes objects
        vocab.add(merged_bytes)

        new_pretokenized_text = {}

        # update full pretokenized text
        for token, count in full_pretokenized_text.items():
            new_token = []
            i = 0
            # if there are fewer than 2 bytes left then there is no pair to check
            while i < len(token) - 1:
                current_pair = (token[i], token[i+1])
                if current_pair == pair:
                    # Merge the pair: concatenate the two bytes objects
                    merged_bytes = pair[0] + pair[1]
                    new_token.append(merged_bytes)
                    i += 2
                else:
                    new_token.append(token[i])
                    i += 1
            # Add the last byte if we didn't process it
            if i < len(token):
                new_token.append(token[i])
            new_token_tuple = tuple(new_token)
            new_pretokenized_text[new_token_tuple] = count
        
        full_pretokenized_text = new_pretokenized_text

    # Convert vocab set to dict with int keys and bytes values
    vocab_dict = {i: token for i, token in enumerate(vocab)}
    return vocab_dict, merges

    



           
    

    
            


            


