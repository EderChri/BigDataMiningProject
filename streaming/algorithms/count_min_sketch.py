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

    def add_many(self, items: Iterable[str], count: int = 1) -> None:
        """Add multiple items with the same count."""
        for it in items:
            self.add(it, count=count)

    def estimate(self, item: str) -> int:
        """
        Estimate the frequency of the item using min across rows.
        """
        mins = []
        for r in range(self.depth):
            c = self._hash(item, r)
            mins.append(self.table[r][c])
        return min(mins) if mins else 0

    def merge(self, other: "CountMinSketch") -> "CountMinSketch":
        """
        Merge another CMS with identical dimensions and seed into this one (in-place).
        """
        if not isinstance(other, CountMinSketch):
            raise TypeError("other must be a CountMinSketch")
        if (self.width, self.depth, self.seed) != (other.width, other.depth, other.seed):
            raise ValueError("Cannot merge CMS with different width/depth/seed")
        for r in range(self.depth):
            row = self.table[r]
            other_row = other.table[r]
            for c in range(self.width):
                row[c] += other_row[c]
        self.total_count += other.total_count
        return self

    @property
    def memory_bytes(self) -> int:
        """Approximate memory used by the table (counts only)."""
        # Assuming small ints but in Python this is not strict; still a useful proxy: number of counters * 8 bytes
        return self.width * self.depth * 8

    def __repr__(self) -> str:
        return f"CountMinSketch(width={self.width}, depth={self.depth}, total={self.total_count})"
