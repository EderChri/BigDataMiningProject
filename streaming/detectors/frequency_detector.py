from typing import Dict, Iterable, List, Tuple
import heapq

from streaming.algorithms.count_min_sketch import CountMinSketch
from streaming.utils.token_handler import split_preprocessed_tokens


class FrequencyDetector:
    """
    Maintains approximate frequencies of tokens and phrases using Count-Min Sketch.
    Tracks top K tokens for frequency analysis.

    Note: Expects messages to be preprocessed by the data loader.
    """

    def __init__(
            self,
            epsilon: float = 0.005,
            delta: float = 1e-3,
            seed: int = 0,
            top_k: int = 10000,
    ) -> None:
        self.cms = CountMinSketch.from_error_delta(epsilon=epsilon, delta=delta, seed=seed)
        self.top_k = top_k

        # Track top K tokens using a min-heap: (count, token)
        self._top_tokens: Dict[str, int] = {}  # token -> approximate count
        self._heap: List[Tuple[int, str]] = []  # min-heap of (count, token)

        # Message counter for periodic updates
        self._message_count = 0

    def observe_message(self, text: str) -> None:
        """Process a message and update Count-Min Sketch."""
        # Expecting 'text' to be preprocessed (space-separated tokens)
        tokens = split_preprocessed_tokens(text)

        # Track all tokens in CMS
        for token in tokens:
            self.cms.add(token)

        self._message_count += 1

    def _update_top_tokens(self, tokens: Iterable[str]) -> None:
        """Update the top K tokens tracking."""
        for token in tokens:
            current_count = self.cms.estimate(token)

            if token in self._top_tokens:
                # Update existing token
                self._top_tokens[token] = current_count
            elif len(self._top_tokens) < self.top_k:
                # Add new token if we haven't reached top_k yet
                self._top_tokens[token] = current_count
                heapq.heappush(self._heap, (current_count, token))
            else:
                # Check if this token should replace the minimum
                min_count, min_token = self._heap[0]
                if current_count > min_count:
                    # Replace minimum
                    heapq.heapreplace(self._heap, (current_count, token))
                    del self._top_tokens[min_token]
                    self._top_tokens[token] = current_count

        # Rebuild heap to maintain consistency
        self._heap = [(count, token) for token, count in self._top_tokens.items()]
        heapq.heapify(self._heap)

    def get_frequency_analysis(self, top_n: int = 10) -> Dict[str, int]:
        """
        Get approximate counts of current top N tokens from CMS.
        Returns dict sorted by frequency (descending).
        """
        snapshot = {}
        for token in self._top_tokens.keys():
            snapshot[token] = self.cms.estimate(token)

        # Sort by count and take top N
        sorted_items = sorted(snapshot.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return dict(sorted_items)

    def periodic_update(self, recent_tokens: Iterable[str]) -> None:
        """
        Perform periodic update of top K tokens.
        Call this every N messages with tokens seen in recent messages.
        """
        self._update_top_tokens(recent_tokens)

    def estimate_frequency(self, term: str) -> int:
        return self.cms.estimate(term.lower())

    def estimate_batch(self, terms: Iterable[str]) -> Dict[str, int]:
        return {t: self.estimate_frequency(t) for t in terms}

    @property
    def message_count(self) -> int:
        return self._message_count

    def __repr__(self) -> str:
        return (
            f"FrequencyDetector(cms={self.cms},"
            f"top_k={self.top_k}, tracked={len(self._top_tokens)})"
        )