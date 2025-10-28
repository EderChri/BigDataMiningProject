from collections import deque
from typing import List, Tuple, Deque, Optional

Bucket = Tuple[int, int]  # (timestamp, size)


class DGIMManager:
    """
    DGIM over multiple bins (indexed 0..num_bins-1).
    timestamp is integer event index (monotonic).
    """

    def __init__(self, num_bins: int, window_size: int):
        self.num_bins = num_bins
        self.window_size = window_size
        # per-bin deque of buckets, newest at left
        self.buckets: List[Deque[Bucket]] = [deque() for _ in range(num_bins)]
        self.current_time = 0

    def _expire(self, bin_idx: int):
        expire_before = self.current_time - self.window_size + 1
        dq = self.buckets[bin_idx]
        while dq and dq[-1][0] < expire_before:
            dq.pop()  # remove oldest bucket entirely

    def _compress(self, bin_idx: int):
        dq = self.buckets[bin_idx]
        # ensure at most 2 buckets of each size; when 3, merge oldest two
        # We'll scan sizes from smallest upward until stable.
        sizes_seen = {}
        i = 0
        while i < len(dq):
            _, size = dq[i]
            sizes_seen.setdefault(size, 0)
            sizes_seen[size] += 1
            if sizes_seen[size] == 3:
                # merge the two oldest buckets of this size (rightmost two)
                # find rightmost two indexes with this size
                idxs = [j for j, (_, s) in enumerate(dq) if s == size]
                # idxs are from 0..n-1 (newest..oldest)
                # take two oldest (largest indices)
                a, b = idxs[-2], idxs[-1]
                # new bucket timestamp = timestamp of the newer of the two merged (a is newer than b)
                t_new = dq[a][0]
                # remove by index b then a (since deque supports rotation, convert to list)
                tmp = list(dq)
                # replace a and b by a single bucket at position a with doubled size
                tmp.pop(b)
                tmp.pop(a)
                tmp.insert(a, (t_new, size * 2))
                dq = deque(tmp)
                self.buckets[bin_idx] = dq
                # restart scanning
                sizes_seen = {}
                i = 0
                continue
            i += 1

    def add_one(self, bin_idx: int):
        """Add a 1-bit to bin at current_time (advance time must be handled externally)."""
        dq = self.buckets[bin_idx]
        dq.appendleft((self.current_time, 1))
        self._compress(bin_idx)
        self._expire(bin_idx)

    def tick(self):
        """Advance global time (call once per message)."""
        self.current_time += 1
        # Optionally expire across bins lazily on query/add; not necessary here.

    def count_last(self, bin_idx: int, k: Optional[int] = None) -> int:
        """
        Estimate number of 1s in last k events (k defaults to full window_size).
        Uses standard DGIM approximation: full buckets + 1/2 of oldest included bucket.
        """
        if k is None:
            k = self.window_size
        if k <= 0:
            return 0
        threshold = self.current_time - k + 1
        total = 0
        dq = self.buckets[bin_idx]
        if not dq:
            return 0
        # iterate from newest to oldest
        for idx, (ts, size) in enumerate(dq):
            if ts >= threshold:
                total += size
            else:
                # oldest partially covered bucket -> add approx half
                total += size // 2
                break
        return total
