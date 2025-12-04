from typing import Dict, Iterable, List, Optional

from streaming.algorithms.min_hash_lsh import MinHashLSH
from streaming.detectors.bucket_frequency_detector import BucketFrequencyDetector
from streaming.detectors.burst_detector import BurstDetector
from streaming.detectors.duplicate_detector import DuplicateDetector
from streaming.detectors.token_frequency_detector import TokenFrequencyDetector
from streaming.utils.reservoir import Reservoir
from utils.actual_observer import ActualObserver


class StreamingPipeline:
    """
    Orchestrates streaming detectors:
      - FrequencyDetector (Count-Min Sketch + DGIM for trending)
      - BurstDetector (DGIM for most frequent terms in recent window)
      - DuplicateDetector (Bloom Filter)

    Use process_message to feed data and get aggregated outputs.
    """

    def __init__(
            self,
            burst_detector: Optional[BurstDetector] = None,
            duplicate_detector: Optional[DuplicateDetector] = None,
            window_size: int = 100,
            actual_observer: Optional[ActualObserver] = None,
            lsh=None,
            reservoirs=None,
            epsilon=None,
            delta=None,
            seed=None,
    ) -> None:
        self.lsh = lsh or MinHashLSH(num_buckets=100, num_hashes=128)
        self.reservoirs = reservoirs or [Reservoir() for _ in range(self.lsh.num_buckets)]
        self.bucket_frequency_detector = BucketFrequencyDetector(lsh=self.lsh, reservoirs=self.reservoirs,
                                                                 epsilon=epsilon, delta=delta, seed=seed)
        self.token_frequency_detector = TokenFrequencyDetector(epsilon=epsilon, delta=delta, seed=seed)
        self.burst_detector = burst_detector or BurstDetector(lsh=self.lsh, window_size=window_size,
                                                              reservoirs=self.reservoirs, epsilon=epsilon, delta=delta,
                                                              seed=seed)
        self.duplicate_detector = duplicate_detector or DuplicateDetector()
        self.window_size = window_size
        self.actual_observer = actual_observer or ActualObserver()

    def process_message(self, text: str, frequency_queries: Optional[Iterable[str]] = None) -> Dict:
        """
        Process a single message text and return a dict of detector outputs.

        frequency_queries: optional list of tokens/phrases to query current estimates for.
        """
        # Update detectors
        self.bucket_frequency_detector.observe_message(text)
        self.token_frequency_detector.observe_message(text)
        self.burst_detector.observe_message(text)
        self.actual_observer.observe_message(text)
        dup_info = self.duplicate_detector.observe_message(text)

        # Prepare outputs
        bucket_freq_out = {}
        token_freq_out = {}
        if frequency_queries:
            bucket_freq_out = self.bucket_frequency_detector.estimate_batch(frequency_queries)
            token_freq_out = self.token_frequency_detector.estimate_batch(frequency_queries)

        out = {
            "bucket_frequencies": bucket_freq_out,
            "token_frequencies": token_freq_out,
            "actual_frequencies": self.actual_observer.get_formatted_counts(top_k=25),
            "actual_bucket_counts": self.actual_observer.get_formatted_bucket_counts(top_k=25),
            "duplicate": dup_info
        }
        return out

    def update_frequency_detectors(self, recent_tokens):
        self.bucket_frequency_detector.periodic_update(recent_tokens)
        self.token_frequency_detector.periodic_update(recent_tokens)

    def __repr__(self) -> str:
        return (
            f"StreamingPipeline("
            f"bucket_frequency={self.bucket_frequency_detector}, "
            f"token_frequency={self.token_frequency_detector},"
            f"burst={self.burst_detector}, "
            f"duplicate={self.duplicate_detector})"
        )
