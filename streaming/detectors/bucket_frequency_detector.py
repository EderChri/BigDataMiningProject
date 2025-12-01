import heapq
from collections import defaultdict
from typing import Optional, Dict, List, Tuple, Iterable

from streaming.algorithms import CountMinSketch
from streaming.algorithms.min_hash_lsh import MinHashLSH
from streaming.detectors.base_frequency_detector import BaseFrequencyDetector


class BucketFrequencyDetector(BaseFrequencyDetector):
    """
    Per-bucket frequency detection using LSH and multiple Count-Min Sketches.
    Each semantic bucket maintains its own CMS for frequency tracking.
    """

    def __init__(
        self,
        lsh: MinHashLSH,
        epsilon: float = 0.1,
        delta: float = 1e-3,
        seed: int = 0,
        top_k: int = 10000,
        window_size: Optional[int] = None,
        reservoirs = None,
    ) -> None:
        super().__init__(epsilon, delta, seed, top_k)
        self.lsh = lsh
        self.reservoirs = reservoirs
        self.window_size = window_size or 0

        # One CMS per semantic bucket
        self.cms_per_bucket = [
            CountMinSketch.from_error_delta(epsilon=epsilon, delta=delta, seed=seed + i)
            for i in range(self.lsh.num_buckets)
        ]
        self.tokens_per_bucket: Dict[int, set] = defaultdict(set)

    def _update_frequencies(self, tokens: List[str]) -> None:
        """Update CMS per bucket based on LSH hashing."""
        for token in tokens:
            bucket_id = self.lsh.hash(token)
            self.cms_per_bucket[bucket_id].add(token)
            self.tokens_per_bucket[bucket_id].add(token)

            freq = self.cms_per_bucket[bucket_id].estimate(token)
            self.reservoirs[bucket_id].add(token, score=freq)

        self.periodic_update(tokens)

    def estimate_frequency(self, term: str, **kwargs) -> int:
        """Estimate bucket frequency by summing all tokens in the bucket."""
        bucket_id = self.lsh.hash(term)
        return self._estimate_bucket_frequency(bucket_id)

    def _estimate_bucket_frequency(self, bucket_id: int) -> int:
        """Sum all token estimates in a bucket (never underestimates)."""
        cms = self.cms_per_bucket[bucket_id]
        tokens = self.tokens_per_bucket[bucket_id]
        return sum(cms.estimate(token) for token in tokens)

    def get_frequency_analysis(self, top_n: int = 10) -> Dict[str, int]:
        """Get top N buckets by total frequency."""
        bucket_freqs = {
            f"bucket_{bid}": self._estimate_bucket_frequency(bid)
            for bid in range(len(self.cms_per_bucket))
        }
        sorted_items = sorted(bucket_freqs.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return dict(sorted_items)

    def get_bucket_representative(self, bucket_id: int) -> Optional[str]:
        """Get the best representative token for a bucket."""
        return self.reservoirs[bucket_id].representative()

    def __repr__(self) -> str:
        total_tokens = sum(len(tokens) for tokens in self.tokens_per_bucket.values())
        return (
            f"BucketFrequencyDetector(buckets={len(self.cms_per_bucket)}, "
            f"total_tokens={total_tokens})"
        )