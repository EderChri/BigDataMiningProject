from collections import deque
from typing import List, Tuple, Deque, Optional

Bucket = Tuple[int, int]  # (timestamp, size)


class DGIM:
    """DGIM for a single stream of 1s."""

    def __init__(self, window_size: int):
        self.window_size = window_size
        self.buckets: Deque[Bucket] = deque()  # newest left
        self.current_time = 0

    def _expire(self):
        expire_before = self.current_time - self.window_size + 1
        while self.buckets and self.buckets[-1][0] < expire_before:
            self.buckets.pop()

    def _compress(self):
        sizes_seen = {}
        i = 0
        while i < len(self.buckets):
            _, size = self.buckets[i]
            sizes_seen.setdefault(size, 0)
            sizes_seen[size] += 1
            if sizes_seen[size] == 3:
                idxs = [j for j, (_, s) in enumerate(self.buckets) if s == size]
                a, b = idxs[-2], idxs[-1]
                t_new = self.buckets[a][0]
                tmp = list(self.buckets)
                tmp.pop(b)
                tmp.pop(a)
                tmp.insert(a, (t_new, size * 2))
                self.buckets = deque(tmp)
                sizes_seen = {}
                i = 0
                continue
            i += 1

    def add_one(self):
        self.buckets.appendleft((self.current_time, 1))
        self._compress()
        self._expire()

    def tick(self):
        self.current_time += 1
        self._expire()

    def count_last(self, k: Optional[int] = None) -> int:
        if k is None:
            k = self.window_size
        if k <= 0:
            return 0
        threshold = self.current_time - k + 1
        total = 0
        for ts, size in self.buckets:
            if ts >= threshold:
                total += size
            else:
                total += size // 2
                break
        return total


class DGIMManager:
    """DGIMManager managing multiple DGIM instances (one per CMS bucket)."""

    def __init__(self, num_bins: int, window_size: int):
        self.dgims: List[DGIM] = [DGIM(window_size) for _ in range(num_bins)]

    def tick(self):
        for dgim in self.dgims:
            dgim.tick()

    def add_one(self, bin_idx: int):
        self.dgims[bin_idx].add_one()

    def count_last(self, bin_idx: int, k: Optional[int] = None) -> int:
        return self.dgims[bin_idx].count_last(k)
