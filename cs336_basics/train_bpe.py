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


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    num_processes = NUM_PROCESSES
    

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>"s)
        text = f.read().decode("utf-8", errors="ignore")
        # change to use multiprocess

        

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            pretokenized_iter = re.finditer(PAT, chunk)
            for match in pretokenized_iter:
                pretokenized_text[tuple(bytes(match.group(), "utf-8"))] = pretokenized_text.get(tuple(bytes(match.group(), "utf-8")), 0) + 1
    

    
            


            


