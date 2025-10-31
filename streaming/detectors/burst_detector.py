# streaming/detectors/burst_detector.py
from typing import Iterable, Tuple, List, Dict, Optional
import random

from streaming.algorithms import CountMinSketch, DGIMManager
from streaming.utils.reservoir import Reservoir, Reservoir


class BurstDetector:
    """
    Uses a provided CountMinSketch instance (cms) and a DGIMManager across CMS columns.
    For each token, we mark the columns corresponding to all CMS rows.
    """


    def __init__(self, window_size: int,  epsilon: float = 0.005, delta: float = 1e-3, seed: int = 0):
        """
        window_size: DGIM window in number of messages/events
        reservoir_size: number of tokens to keep per bin
        """
        self.cms = CountMinSketch.from_error_delta(epsilon=epsilon, delta=delta, seed=seed)

        self.dgim = DGIMManager(num_bins=self.cms.width, window_size=window_size)
        self.reservoirs: List[Reservoir] = [Reservoir() for _ in range(self.cms.width)]
        self.window_size = window_size

    def observe_message(self, message: str):
        tokens = message.split()

        # advance all DGIMs
        self.dgim.tick()

        # update CMS counts
        self.cms.add_many(tokens, count=1)

        for tok in tokens:
            for row in range(self.cms.depth):
                col = self.cms._hash(tok, row) % self.cms.width
                self.dgim.add_one(col)
                freq_estimate = self.cms.estimate(tok)
                self.reservoirs[col].add(tok, score=freq_estimate)

    def detect_spikes(self, recent_k: int = None, prev_k: Optional[int] = None, threshold: float = 2.0, min_count: int = 1) -> List[Dict]:
        """
        Detect bins that have recent activity spike.
        - recent_k: length of the recent window in events/messages
        - prev_k: length of previous window to compare against; if None, prev_k = recent_k
        - threshold: ratio (recent / previous) to flag spike
        - min_count: min recent count to consider (avoid tiny counts)
        Returns list of dicts: {bin, ratio, recent_count, prev_count, representative}
        """
        if recent_k is None:
            recent_k = self.window_size // 2
        if prev_k is None:
            prev_k = recent_k

        results = []
        eps = 1e-6
        for col in range(self.cms.width):
            recent = self.dgim.count_last(col, k=recent_k)
            prev_total = self.dgim.count_last(col, k=recent_k + prev_k)
            prev = max(0, prev_total - recent)  # counts in the previous window
            if recent < min_count:
                continue
            ratio = (recent + eps) / (prev + eps)
            if ratio >= threshold:
                rep = self.reservoirs[col].representative()
                results.append({
                    "bin": col,
                    "ratio": ratio,
                    "recent_count": recent,
                    "prev_count": prev,
                    "representative": rep,
                })
        # sort by significance
        results.sort(key=lambda x: (-x["ratio"], -x["recent_count"]))
        return results
