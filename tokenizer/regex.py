"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Unlike BasicTokenizer:
- RegexTokenizer handles an optional regex splitting pattern.
- RegexTokenizer handles optional special tokens.
"""

import regex as re
from .base import Tokenizer
from .helper import get_stats, merge
from tqdm import tqdm


# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(Tokenizer):

    def __init__(self, pattern=None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text, vocab_size, verbose=False):
        """
        Train the tokenizer on the provided text.
        Use batching and parallel processing to speed up training.
        """
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # Split the text into text chunks
        text_chunks = re.findall(self.compiled_pattern, text)

        # Input text preprocessing
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        # Iteratively merge the most common pairs to create new tokens
        merges = {}  # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)}  # idx -> bytes

        # Add tqdm progress bar for the merge process
        for i in tqdm(range(num_merges), desc="Merging pairs", unit="merge"):
            # Count the number of times every consecutive pair appears
            stats = {}
            for chunk_ids in tqdm(ids, desc="Processing chunks", leave=False, unit="chunk"):
                # Passing in stats will update it in place, adding up counts
                get_stats(chunk_ids, stats)

            # Find the pair with the highest count
            pair = max(stats, key=stats.get)

            # Mint a new token: assign it the next available id
            idx = 256 + i

            # Replace all occurrences of pair in ids with idx
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]

            # Save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            # Prints verbose info if needed
            if verbose:
                print(f"merge {i + 1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # Save class variables
        self.merges = merges  # used in encode()
        self.vocab = vocab  # used in decode()

    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def _encode_chunk(self, text_bytes):
        # given a string text, return the token ids
        ids = list(text_bytes) # list of integers in range 0..255
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            # Key point: Here, we are not selecting the byte pair based on frequency, but based on the merge index in the self.merges dictionary.
            # self.merges stores the merge priority or order for each byte pair. The min(stats, key=...) selects the byte pair with the smallest merge index,
            # which corresponds to the pair that needs to be merged first, not the pair with the lowest frequency.
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8") # raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text, allowed_special="none_raise"):
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """
        # decode the user desire w.r.t. handling of special tokens
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            # shortcut: if no special tokens, just use the ordinary encoding
            return self.encode_ordinary(text)
        # otherwise, we have to be careful with potential special tokens in text
        # we handle special tokens by splitting the text
        # based on the occurrence of any exact match with any of the special tokens
        # we can use re.split for this. note that surrounding the pattern with ()
        # makes it into a capturing group, so the special tokens will be included
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        # now all the special characters are separated from the rest of the text
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for part in special_chunks:
            if part in special:
                # this is a special token, encode it separately as a special case
                ids.append(special[part])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self.encode_ordinary(part))
        return ids

    def decode(self, ids):
        # given ids (list of integers), return Python string
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text


    # The following code represents an attempt to implement parallel training, which is not fully implemented yet.
    # def train(self, text, vocab_size, verbose=False, batch_size=10000, num_workers=4):
    #     assert vocab_size >= 256
    #     num_merges = vocab_size - 256
    #
    #     # Split the text into chunks
    #     text_chunks = re.findall(self.compiled_pattern, text)
    #
    #     # Input text preprocessing (encoding each chunk into bytes)
    #     ids = [list(ch.encode("utf-8")) for ch in text_chunks]
    #
    #     # Initialize vocab with single-byte tokens (0..255)
    #     vocab = {idx: bytes([idx]) for idx in range(256)}
    #
    #     # Use Manager to share merges and vocab between processes
    #     with Manager() as manager:
    #         merges = manager.dict()  # Shared dictionary for merges
    #         vocab = manager.dict(vocab)  # Shared dictionary for vocab
    #
    #         # Use a process pool to parallelize the merge steps
    #         with ProcessPoolExecutor(max_workers=num_workers) as executor:
    #             futures = []
    #             for i in range(num_merges):
    #                 futures.append(executor.submit(self._merge_step, i, ids, merges, vocab, verbose))
    #
    #             # Collect results from all parallel merges
    #             for future in tqdm(as_completed(futures), total=num_merges, desc="Merging pairs", unit="merge"):
    #                 local_merges, local_vocab = future.result()  # Unwrap the result from each process
    #                 # Merge the local results into the global merges and vocab
    #                 merges.update(local_merges)
    #                 vocab.update(local_vocab)
    #
    #         # After merging, transfer from manager dicts to normal dicts
    #         self.merges = dict(merges)
    #         self.vocab = dict(vocab)
    #
    # def _merge_step(self, i, ids, merges, vocab, verbose):
    #     """
    #     Perform a single merge step for the training process.
    #     This method is designed to run in parallel for each merge step.
    #     """
    #     stats = defaultdict(int)
    #     for chunk_ids in ids:
    #         get_stats(chunk_ids, stats)
    #
    #     # Find the most frequent pair
    #     pair = max(stats, key=stats.get)
    #     idx = 256 + i  # New token id
    #
    #     # Merge pairs in the ids and update vocab
    #     new_ids = []
    #     for chunk_ids in ids:
    #         new_ids.append(merge(chunk_ids, pair, idx))
    #
    #     # Update merges and vocab
    #     local_merges = {pair: idx}
    #     local_vocab = {idx: vocab[pair[0]] + vocab[pair[1]]}
    #
    #     # Optionally print merge information
    #     if verbose:
    #         print(f"merge {i + 1}/{len(merges)}: {pair} -> {idx} ({local_vocab[idx]}) had {stats[pair]} occurrences")
    #
    #     return local_merges, local_vocab




