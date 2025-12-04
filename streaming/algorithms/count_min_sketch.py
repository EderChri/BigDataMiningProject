import hashlib
import math
from typing import Iterable, Optional


class CountMinSketch:
    """
    Count-Min Sketch for approximate frequency estimation.
    Space: O(depth * width)
    Query/Update time: O(depth)
    Error bounds:
      - With width w = ceil(e/epsilon), depth d = ceil(ln(1/delta)):
        estimate(x) <= true(x) + epsilon * total_count with probability >= 1 - delta
    """

    def __init__(self, width: int, depth: int, seed: int = 0) -> None:
        if width <= 0 or depth <= 0:
            raise ValueError("width and depth must be positive")
        self.width = int(width)
        self.depth = int(depth)
        self.seed = int(seed)
        # 2D table: depth rows, width columns
        self.table = [[0] * self.width for _ in range(self.depth)]
        self.total_count = 0
        # Precompute per-row salts for hashing
        self._salts = [hashlib.sha256(f"{self.seed}-{i}".encode()).digest() for i in range(self.depth)]

    @classmethod
    def from_error_delta(cls, epsilon: float, delta: float, seed: int = 0) -> "CountMinSketch":
        """
        Build CMS using error (epsilon) and failure probability (delta).
        epsilon: additive error as a fraction of total count (e.g., 0.01)
        delta: failure probability (e.g., 1e-3)
        """
        if epsilon <= 0 or delta <= 0 or delta >= 1:
            raise ValueError("epsilon must be > 0 and delta in (0,1)")
        width = math.ceil(math.e / epsilon)
        depth = math.ceil(math.log(1.0 / delta))
        return cls(width=width, depth=depth, seed=seed)

    def _hash(self, item: str, row: int) -> int:
        """
        Hash an item to a column index for the given row using row-specific salt.
        """
        if not isinstance(item, (bytes, bytearray)):
            item_bytes = str(item).encode("utf-8", errors="ignore")
        else:
            item_bytes = item
        h = hashlib.blake2b(item_bytes, digest_size=16, key=self._salts[row])
        # Convert 16 bytes to integer then mod width
        return int.from_bytes(h.digest(), "big") % self.width

    def add(self, item: str, count: int = 1) -> None:
        """Increment the count estimate for an item."""
        if count < 0:
            raise ValueError("count must be non-negative")
        if count == 0:
            return
        for r in range(self.depth):
            c = self._hash(item, r)
            self.table[r][c] += count
        self.total_count += count

    def estimate(self, item: str) -> int:
        """
        Estimate the frequency of the item using min across rows.
        """
        mins = []
        for r in range(self.depth):
            c = self._hash(item, r)
            mins.append(self.table[r][c])
        return min(mins) if mins else 0
