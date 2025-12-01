from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Tuple
import heapq

from streaming.utils.token_handler import split_preprocessed_tokens


class BaseFrequencyDetector(ABC):
    """
    Abstract base class for frequency detection.
    Defines common interface and shared functionality for tracking token frequencies.
    """

    def __init__(
            self,
            epsilon: float = 0.005,
            delta: float = 1e-3,
            seed: int = 0,
            top_k: int = 10000,
    ) -> None:
        self.epsilon = epsilon
        self.delta = delta
        self.seed = seed
        self.top_k = top_k
        self._message_count = 0

        # Shared top-K tracking structures
        self._top_tokens: Dict[str, int] = {}
        self._heap: List[Tuple[int, str]] = []

    def observe_message(self, text: str) -> None:
        """Process a message and update frequency structures."""
        tokens = split_preprocessed_tokens(text)
        self._update_frequencies(tokens)
        self._message_count += 1

    @abstractmethod
    def _update_frequencies(self, tokens: List[str]) -> None:
        """Update internal frequency structures. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def estimate_frequency(self, term: str, **kwargs) -> int:
        """Estimate frequency for a given term."""
        pass

    def _update_top_k(self, token: str, current_count: int) -> None:
        """Shared logic for maintaining top-K heap."""
        if token in self._top_tokens:
            self._top_tokens[token] = current_count
        elif len(self._top_tokens) < self.top_k:
            self._top_tokens[token] = current_count
            heapq.heappush(self._heap, (current_count, token))
        else:
            min_count, min_token = self._heap[0]
            if current_count > min_count:
                heapq.heapreplace(self._heap, (current_count, token))
                del self._top_tokens[min_token]
                self._top_tokens[token] = current_count

    def _rebuild_heap(self) -> None:
        """Rebuild heap to maintain consistency."""
        self._heap = [(count, token) for token, count in self._top_tokens.items()]
        heapq.heapify(self._heap)

    def periodic_update(self, recent_tokens: Iterable[str]) -> None:
        """Perform periodic update of top K tokens."""
        for token in recent_tokens:
            current_count = self.estimate_frequency(token)
            self._update_top_k(token, current_count)
        self._rebuild_heap()

    def estimate_batch(self, terms: Iterable[str], **kwargs) -> Dict[str, int]:
        """Estimate frequencies for multiple terms."""
        return {t: self.estimate_frequency(t, **kwargs) for t in terms}

    @property
    def message_count(self) -> int:
        return self._message_count
