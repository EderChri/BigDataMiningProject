import hashlib
from typing import Set


class MinHashLSH:
    """MinHash-based LSH for grouping similar words into buckets."""

    def __init__(self, num_buckets: int, num_hashes: int = 128, ngram_size: int = 3):
        self.num_buckets = num_buckets
        self.num_hashes = num_hashes
        self.ngram_size = ngram_size
        # Precompute hash seeds
        self.seeds = [i * 2654435761 for i in range(num_hashes)]

    def _get_shingles(self, word: str) -> Set[str]:
        """Extract character n-grams from word."""
        word = word.lower()
        if len(word) < self.ngram_size:
            return {word}
        return {word[i:i + self.ngram_size] for i in range(len(word) - self.ngram_size + 1)}

    def _minhash_signature(self, word: str) -> int:
        """Compute MinHash signature and map to bucket."""
        shingles = self._get_shingles(word)

        # Compute min hash for each hash function
        min_hashes = []
        for seed in self.seeds:
            min_val = float('inf')
            for shingle in shingles:
                h = hashlib.sha256(f"{seed}:{shingle}".encode())
                hash_val = int.from_bytes(h.digest()[:8], 'big')
                min_val = min(min_val, hash_val)
            min_hashes.append(min_val)

        # Combine signature into bucket ID
        signature = hash(tuple(min_hashes))
        return signature % self.num_buckets

    def hash(self, word: str) -> int:
        """Map word to semantic bucket."""
        return self._minhash_signature(word)
