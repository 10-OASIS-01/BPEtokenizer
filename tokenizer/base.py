from .helper import render_token

class Tokenizer:
    """Base class for Tokenizers"""

    def __init__(self):
        # default: vocab size of 256 (all bytes), no merges, no patterns
        self.merges = {}  # (int, int) -> int
        self.pattern = ""  # str
        self.special_tokens = {}  # str -> int, e.g. {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab()  # int -> bytes

    def _build_vocab(self):
        """
        Build the vocabulary from merges and special tokens.
        This function deterministically constructs vocab from merges and special_tokens.
        """
        try:
            vocab = {idx: bytes([idx]) for idx in range(256)}
            for (p0, p1), idx in self.merges.items():
                if p0 not in vocab or p1 not in vocab:
                    raise ValueError(f"[build_vocab] Merge parents {p0}, {p1} not in vocab.")
                vocab[idx] = vocab[p0] + vocab[p1]
            for special, idx in self.special_tokens.items():
                if idx in vocab:
                    raise ValueError(f"[build_vocab] Special token index conflict: {special} at {idx}")
                vocab[idx] = special.encode("utf-8")
            return vocab
        except Exception as e:
            print(f"[ERROR][_build_vocab] Error building vocab: {e}")
            raise

    def train(self, text, vocab_size, verbose=False):
        # Tokenizer can train a vocabulary of size vocab_size from text
        raise NotImplementedError

    def encode(self, text):
        # Tokenizer can encode a string into a list of integers
        raise NotImplementedError

    def decode(self, ids):
        # Tokenizer can decode a list of integers into a string
        raise NotImplementedError

    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        try:
            # write the model: to be used in load() later
            model_file = file_prefix + ".model"
            with open(model_file, 'w', encoding="utf-8") as f:
                # write the version, pattern and merges, that's all that's needed
                f.write("BPEtokenizer Tokenizer v1\n")
                f.write(f"{self.pattern}\n")
                # write the special tokens, first the number of them, then each one
                f.write(f"{len(self.special_tokens)}\n")
                for special, idx in self.special_tokens.items():
                    f.write(f"{special} {idx}\n")
                # the merges dict
                for idx1, idx2 in self.merges:
                    f.write(f"{idx1} {idx2}\n")

            # write the vocab: for the human to look at
            vocab_file = file_prefix + ".vocab"
            inverted_merges = {idx: pair for pair, idx in self.merges.items()}
            with open(vocab_file, "w", encoding="utf-8") as f:
                for idx, token in self.vocab.items():
                    s = render_token(token)
                    # find the children of this token, if any
                    if idx in inverted_merges:
                        # if this token has children, render it nicely as a merge
                        idx0, idx1 = inverted_merges[idx]
                        s0 = render_token(self.vocab.get(idx0, b""))
                        s1 = render_token(self.vocab.get(idx1, b""))
                        f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                    else:
                        # otherwise this is leaf token, just print it
                        # (this should just be the first 256 tokens, the bytes)
                        f.write(f"[{s}] {idx}\n")
        except Exception as e:
            print(f"[ERROR][save] Error saving model or vocab: {e}")
            raise

    def load(self, model_file):
        """
        Inverse of save() but only for the model file,
        note: many tokens may be partial utf-8 sequences
        and cannot be decoded into valid strings. Here we're using
        errors='replace' to replace them with the replacement char �.
        this also means that we couldn't possibly use .vocab in load()
        because decoding in this way is a lossy operation!
        """
        try:
            assert model_file.endswith(".model"), "[load] Model file must end with .model"
            merges = {}
            special_tokens = {}
            idx = 256
            with open(model_file, 'r', encoding="utf-8") as f:
                version = f.readline().strip()
                if version != "BPEtokenizer Tokenizer v1":
                    raise ValueError(f"[load] Model version mismatch: got '{version}'")
                self.pattern = f.readline().strip()
                num_special_line = f.readline()
                if not num_special_line:
                    raise ValueError("[load] Missing special token count line.")
                num_special = int(num_special_line.strip())
                for _ in range(num_special):
                    line = f.readline()
                    if not line:
                        raise ValueError("[load] Special token count does not match file content.")
                    special, special_idx = line.strip().split()
                    special_tokens[special] = int(special_idx)
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        idx1, idx2 = map(int, line.split())
                        merges[(idx1, idx2)] = idx
                        idx += 1
                    except Exception as e:
                        print(f"[ERROR][load] Error parsing merge line '{line.strip()}': {e}")
                        raise
            self.merges = merges
            self.special_tokens = special_tokens
            self.vocab = self._build_vocab()
        except Exception as e:
            print(f"[ERROR][load] Error loading model file '{model_file}': {e}")
            raise
