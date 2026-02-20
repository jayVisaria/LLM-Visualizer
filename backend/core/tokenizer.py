"""
BPE (Byte Pair Encoding) Tokenizer — built from scratch.

This implements the core BPE algorithm used by GPT-2/GPT-3/GPT-4:
1. Start with individual characters (bytes) as the initial vocabulary.
2. Count all adjacent pairs of tokens in the corpus.
3. Merge the most frequent pair into a new token.
4. Repeat for a desired number of merges.

The result is a subword vocabulary that balances between character-level
and word-level tokenization, achieving good compression while handling
rare/unseen words gracefully.
"""

import os
import json
import urllib.request
from collections import Counter
from typing import Optional

# ---------------------------------------------------------------------------
# Data download helper
# ---------------------------------------------------------------------------
TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def get_training_text() -> str:
    """Download Tiny Shakespeare if needed and return its contents."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "input.txt")
    if not os.path.exists(path):
        print(f"Downloading Tiny Shakespeare to {path} …")
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# BPE Tokenizer
# ---------------------------------------------------------------------------
class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer implemented from scratch.

    Attributes
    ----------
    merges : list[tuple[tuple[int,int], int]]
        Ordered list of (pair, new_token_id) merges performed during training.
    vocab : dict[int, bytes]
        Mapping from token id → byte string.
    merge_history : list[dict]
        Detailed history of every merge step (for visualization).
    """

    def __init__(self) -> None:
        # Base vocabulary: single bytes 0-255
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.merges: list[tuple[tuple[int, int], int]] = []
        self.merge_history: list[dict] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, text: str, num_merges: int = 500, verbose: bool = False) -> None:
        """
        Train BPE on *text* by performing *num_merges* merge operations.

        Algorithm
        ---------
        1. Encode text to raw bytes → list of ints (0-255).
        2. Count frequencies of every adjacent pair.
        3. Pick the pair with the highest frequency.
        4. Replace every occurrence of that pair in the token list with a
           new token whose id = 256 + merge_step.
        5. Record the merge and repeat.
        """
        # Encode entire text to bytes and convert to list of ints
        tokens: list[int] = list(text.encode("utf-8"))

        for i in range(num_merges):
            # Count adjacent pairs
            pair_counts = self._count_pairs(tokens)
            if not pair_counts:
                break

            # Most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)  # type: ignore[arg-type]
            freq = pair_counts[best_pair]
            if freq < 2:
                break  # No pair appears more than once — stop

            # Assign new token id
            new_id = 256 + i
            # Merge in token list
            tokens = self._merge_pair(tokens, best_pair, new_id)

            # Update vocab: new token = concat of the two merged tokens
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.merges.append((best_pair, new_id))

            # Record for visualization
            self.merge_history.append(
                {
                    "step": i,
                    "pair": [best_pair[0], best_pair[1]],
                    "pair_text": [
                        self.vocab[best_pair[0]].decode("utf-8", errors="replace"),
                        self.vocab[best_pair[1]].decode("utf-8", errors="replace"),
                    ],
                    "new_id": new_id,
                    "new_text": self.vocab[new_id].decode("utf-8", errors="replace"),
                    "frequency": freq,
                    "vocab_size": 256 + i + 1,
                    "corpus_length": len(tokens),
                }
            )

            if verbose and i % 50 == 0:
                pair_str = (
                    self.vocab[best_pair[0]].decode("utf-8", errors="replace")
                    + self.vocab[best_pair[1]].decode("utf-8", errors="replace")
                )
                print(
                    f"  merge {i:>4d}: ({best_pair[0]:>4d}, {best_pair[1]:>4d}) "
                    f"→ {new_id:>4d}  freq={freq:>6d}  '{pair_str}'"
                )

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of BPE token ids."""
        tokens = list(text.encode("utf-8"))
        for (p0, p1), new_id in self.merges:
            tokens = self._merge_pair(tokens, (p0, p1), new_id)
        return tokens

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token ids back to a string."""
        byte_seq = b"".join(self.vocab[idx] for idx in ids)
        return byte_seq.decode("utf-8", errors="replace")

    def encode_with_segments(self, text: str) -> list[dict]:
        """
        Encode and return per-token metadata (for visualization).

        Returns list of dicts: {id, text, byte_length}
        """
        ids = self.encode(text)
        segments = []
        for token_id in ids:
            token_bytes = self.vocab[token_id]
            segments.append(
                {
                    "id": token_id,
                    "text": token_bytes.decode("utf-8", errors="replace"),
                    "byte_length": len(token_bytes),
                }
            )
        return segments

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save merges + vocab mapping to JSON."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "merges": [
                {"pair": list(pair), "new_id": new_id}
                for pair, new_id in self.merges
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str) -> None:
        """Load merges from JSON and rebuild vocab."""
        with open(path, "r") as f:
            data = json.load(f)
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = []
        for entry in data["merges"]:
            pair = (entry["pair"][0], entry["pair"][1])
            new_id = entry["new_id"]
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.merges.append((pair, new_id))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _count_pairs(tokens: list[int]) -> Counter:
        counts: Counter = Counter()
        for i in range(len(tokens) - 1):
            counts[(tokens[i], tokens[i + 1])] += 1
        return counts

    @staticmethod
    def _merge_pair(
        tokens: list[int], pair: tuple[int, int], new_id: int
    ) -> list[int]:
        """Replace every occurrence of *pair* in *tokens* with *new_id*."""
        merged: list[int] = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                merged.append(new_id)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged

    # ------------------------------------------------------------------
    # Vocab stats (for API)
    # ------------------------------------------------------------------
    def get_vocab_stats(self, text: Optional[str] = None) -> dict:
        """Return vocabulary statistics for visualization."""
        stats: dict = {
            "vocab_size": self.vocab_size,
            "num_merges": len(self.merges),
            "base_vocab_size": 256,
        }
        if text is not None:
            raw_bytes = len(text.encode("utf-8"))
            encoded = self.encode(text)
            stats["original_bytes"] = raw_bytes
            stats["encoded_tokens"] = len(encoded)
            stats["compression_ratio"] = round(raw_bytes / len(encoded), 2) if encoded else 0

            # Token frequency distribution (top 50)
            freq = Counter(encoded)
            top = freq.most_common(50)
            stats["token_frequencies"] = [
                {
                    "id": tid,
                    "text": self.vocab[tid].decode("utf-8", errors="replace"),
                    "count": count,
                }
                for tid, count in top
            ]
        return stats
