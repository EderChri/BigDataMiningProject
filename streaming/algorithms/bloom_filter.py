import hashlib
import math
from typing import Iterable, Optional


class BloomFilter:
    """
    Bloom Filter for probabilistic membership testing.

    Parameters:
      capacity: expected number of elements to store
      error_rate: desired false positive rate (e.g., 0.01)

    Methods:
      add(item), __contains__(item), add_many(items), fill_ratio
    """

    def __init__(self, capacity: int, error_rate: float = 0.01, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if not (0 < error_rate < 1):
            raise ValueError("error_rate must be in (0,1)")
        self.capacity = int(capacity)
        self.error_rate = float(error_rate)
        self.seed = int(seed)
        # Optimal number of bits and hash functions from https://www.eecs.harvard.edu/~michaelm/postscripts/rsa2008.pdf?utm_source=chatgpt.com
        m = -int(round(self.capacity * math.log(self.error_rate) / (math.log(2) ** 2)))
        k = max(1, int(round((m / self.capacity) * math.log(2))))
        self.m = m
        self.k = k
        self._bits = bytearray((self.m + 7) // 8)

    def _hashes(self, item: str):
        if not isinstance(item, (bytes, bytearray)):
            item_bytes = str(item).encode("utf-8", errors="ignore")
        else:
            item_bytes = item
        # Use double hashing to derive k indices from https://www.eecs.harvard.edu/~michaelm/postscripts/rsa2008.pdf?utm_source=chatgpt.com
        h1 = int.from_bytes(hashlib.blake2b(item_bytes, digest_size=16, key=b"h1").digest(), "big")
        h2 = int.from_bytes(hashlib.blake2b(item_bytes, digest_size=16, key=b"h2").digest(), "big")
        for i in range(self.k):
            yield (h1 + i * h2 + i*i) % self.m

    def _set_bit(self, idx: int) -> None:
        byte_index = idx // 8
        bit_index = idx % 8
        self._bits[byte_index] |= (1 << bit_index)

    def _get_bit(self, idx: int) -> bool:
        byte_index = idx // 8
        bit_index = idx % 8
        return bool(self._bits[byte_index] & (1 << bit_index))

    def add(self, item: str) -> None:
        for idx in self._hashes(item):
            self._set_bit(idx)

    def add_many(self, items: Iterable[str]) -> None:
        for it in items:
            self.add(it)

    def __contains__(self, item: str) -> bool:
        return all(self._get_bit(idx) for idx in self._hashes(item))

    @property
    def fill_ratio(self) -> float:
        set_bits = 0
        for b in self._bits:
            # count bits in byte
            v = b
            # builtin bit_count in Python 3.8+
            set_bits += int(v).bit_count()
        return set_bits / self.m

    def __repr__(self) -> str:
        return f"BloomFilter(capacity={self.capacity}, error_rate={self.error_rate:.4f}, m={self.m}, k={self.k})"
