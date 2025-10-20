from collections import deque
from typing import Deque, List, Tuple


class DGIM:
    """
    DGIM algorithm for counting number of 1s in the last N bits of a binary stream.

    This implementation tracks buckets where each bucket represents a group of 1s of size 2^k,
    identified by the timestamp (arrival index) of its most recent 1 (bucket 'end time').

    Operations:
      - push(bit): Ingests next bit (0 or 1)
      - estimate(): Returns estimated count of 1s in the last window_size bits

    Error: Estimate over-count <= 50% of the size of the oldest bucket included in the window.
    """

    def __init__(self, window_size: int) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = int(window_size)
        # For each bucket size (1,2,4,...), keep a deque of (size, end_timestamp)
        self.levels: List[Deque[Tuple[int, int]]] = []
        self.time = 0  # arrival index of next bit

    def _expire_old(self) -> None:
        """Drop buckets whose end_timestamp is older than the window start."""
        window_start = self.time - self.window_size
        for dq in self.levels:
            while dq and dq[0][1] <= window_start:
                dq.popleft()

    def push(self, bit: int) -> None:
        """Push the next bit (0 or 1) into the stream."""
        self.time += 1
        if bit not in (0, 1):
            raise ValueError("bit must be 0 or 1")

        # Expire old buckets at this time tick
        self._expire_old()

        if bit == 0:
            return

        # Create a new bucket of size 1 at current time
        if not self.levels:
            self.levels.append(deque())
        self.levels[0].append((1, self.time))

        # For each level, ensure at most 2 buckets; if 3, merge two oldest into next level
        lvl = 0
        while True:
            # Ensure at most two buckets at level lvl
            if len(self.levels[lvl]) <= 2:
                break

            # Merge two oldest
            oldest = self.levels[lvl].popleft()
            second_oldest = self.levels[lvl].popleft()
            merged_size = oldest[0] + second_oldest[0]  # should be 2^lvl + 2^lvl = 2^(lvl+1)
            merged_end_time = max(oldest[1], second_oldest[1])

            # Move to next level
            next_lvl = lvl + 1
            if next_lvl >= len(self.levels):
                self.levels.append(deque())
            self.levels[next_lvl].append((merged_size, merged_end_time))

            lvl = next_lvl

    def estimate(self) -> int:
        """
        Estimate the number of 1s in the last window_size bits.
        """
        self._expire_old()
        total = 0
        # Traverse levels from newest to oldest buckets
        # We will include all but the oldest bucket fully; the oldest bucket contributes half its size
        oldest_bucket_size = 0
        oldest_bucket_found = False

        # Collect all buckets as (end_time, size), newest last
        buckets: List[Tuple[int, int]] = []
        for lvl, dq in enumerate(self.levels):
            for size, end_time in dq:
                buckets.append((end_time, size))
        buckets.sort()  # ascending by end_time (oldest first)

        for idx, (end_time, size) in enumerate(reversed(buckets)):
            # Include fully
            total += size
            oldest_bucket_size = size
            oldest_bucket_found = True

        if oldest_bucket_found:
            # Subtract half of the oldest bucket to bound error
            total -= oldest_bucket_size // 2

        return total

    def __repr__(self) -> str:
        active_buckets = sum(len(dq) for dq in self.levels)
        return f"DGIM(window_size={self.window_size}, time={self.time}, buckets={active_buckets})"
