from typing import Dict, List, Tuple

from streaming.algorithms.bloom_filter import BloomFilter
from streaming.utils.token_handler import split_preprocessed_tokens


def shingles(tokens: List[str], k: int) -> List[str]:
    if k <= 1:
        return tokens
    return [" ".join(tokens[i : i + k]) for i in range(0, max(0, len(tokens) - k + 1))]


class DuplicateDetector:
    """
    Probabilistic duplicate / near-duplicate detector using a Bloom Filter over text shingles.

    A message is flagged as duplicate if a sufficient fraction of its shingles are already present
    in the Bloom Filter.

    Note: Expects messages to be preprocessed by the data loader.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        error_rate: float = 0.01,
        shingle_size: int = 3,
        duplicate_threshold: float = 0.7,
        seed: int = 0,
    ) -> None:
        if not (0.0 < duplicate_threshold <= 1.0):
            raise ValueError("duplicate_threshold must be in (0,1]")
        self.bloom = BloomFilter(capacity=capacity, error_rate=error_rate, seed=seed)
        self.shingle_size = int(shingle_size)
        self.duplicate_threshold = float(duplicate_threshold)

    def _to_shingles(self, text: str) -> List[str]:
        toks = split_preprocessed_tokens(text)
        return shingles(toks, self.shingle_size)

    def is_duplicate(self, text: str) -> Tuple[bool, float]:
        sh = self._to_shingles(text)
        if not sh:
            return False, 0.0
        hits = sum(1 for s in sh if s in self.bloom)
        ratio = hits / len(sh)
        return ratio >= self.duplicate_threshold, ratio

    def observe_message(self, text: str) -> Dict[str, float]:
        """
        Check duplication and then update the Bloom Filter with the message shingles.
        Returns a dict with status and score.
        """
        is_dup, score = self.is_duplicate(text)
        # Update filter after checking
        sh = self._to_shingles(text)
        self.bloom.add_many(sh)
        return {"is_duplicate": is_dup, "duplicate_score": score, "fill_ratio": self.bloom.fill_ratio}

    def __repr__(self) -> str:
        return f"DuplicateDetector(shingle_size={self.shingle_size}, threshold={self.duplicate_threshold})"
