from typing import Dict, Iterable, List, Optional

from streaming.detectors.frequency_detector import FrequencyDetector
from streaming.detectors.burst_detector import BurstDetector
from streaming.detectors.duplicate_detector import DuplicateDetector


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
        frequency_detector: Optional[FrequencyDetector] = None,
        burst_detector: Optional[BurstDetector] = None,
        duplicate_detector: Optional[DuplicateDetector] = None,
    ) -> None:
        self.frequency_detector = frequency_detector or FrequencyDetector()
        self.burst_detector = burst_detector or BurstDetector(window_size=25)
        self.duplicate_detector = duplicate_detector or DuplicateDetector()

    def process_message(self, text: str, frequency_queries: Optional[Iterable[str]] = None) -> Dict:
        """
        Process a single message text and return a dict of detector outputs.

        frequency_queries: optional list of tokens/phrases to query current estimates for.
        """
        # Update detectors
        self.frequency_detector.observe_message(text)
        self.burst_detector.observe_message(text)
        dup_info = self.duplicate_detector.observe_message(text)

        # Prepare outputs
        freq_out = {}
        if frequency_queries:
            freq_out = self.frequency_detector.estimate_batch(frequency_queries)

        burst_summary = self.burst_detector.detect_spikes(25, 25)

        out = {
            "frequencies": freq_out,
            "burst": burst_summary,
            "duplicate": dup_info,
        }
        return out

    def __repr__(self) -> str:
        return (
            f"StreamingPipeline("
            f"frequency={self.frequency_detector}, "
            f"burst={self.burst_detector}, "
            f"duplicate={self.duplicate_detector})"
        )