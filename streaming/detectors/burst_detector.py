from typing import Dict, Iterable, List, Optional, Set, Tuple
import heapq

from streaming.algorithms.dgim import DGIM
from streaming.utils.token_handler import split_preprocessed_tokens


class BurstDetector:
    """
    Detects bursts by tracking token frequencies using DGIM instances.
    Maintains a DGIM for each tracked token to identify the most frequent terms
    in the recent window.

    Note: This detector expects messages to be preprocessed by the data loader.
    """

    def __init__(
            self,
            window_size: int = 100,
            top_k_tokens: int = 1000,
            burst_threshold: int = 10,
            report_top_n: int = 20,
            promotion_threshold: int = 3
    ) -> None:
        """
        Parameters:
          window_size: DGIM window size (number of messages to track)
          top_k_tokens: Maximum number of tokens to track with DGIM instances
          burst_threshold: Minimum recent count for a token to be considered in burst
          report_top_n: Number of top tokens to report
        """
        self.window_size = window_size
        self.top_k_tokens = top_k_tokens
        self.burst_threshold = burst_threshold
        self.report_top_n = report_top_n
        self.promotion_threshold = promotion_threshold
        # DGIM instances for each tracked token
        self._dgim_instances: Dict[str, DGIM] = {}

        # Track which tokens we're monitoring (for efficient updates)
        self._tracked_tokens: Set[str] = set()

        # Min-heap to maintain top K tokens: (estimate, token)
        self._heap: List[Tuple[int, str]] = []
        # Candidate buffer
        self._candidate_tokens: Dict[str, int] = {}
        # Message counter
        self._message_count = 0

    def observe_message(self, text: str):
        tokens = split_preprocessed_tokens(text)
        seen_in_message = set(tokens)

        # Update tracked tokens
        for token in self._tracked_tokens:
            self._dgim_instances[token].push(1 if token in seen_in_message else 0)

        # Update candidate buffer
        for token in seen_in_message:
            if token not in self._tracked_tokens:
                self._candidate_tokens[token] = self._candidate_tokens.get(token, 0) + 1

        self._message_count += 1

    def update_tracked_tokens(self):
        """
        Promote candidates that show repeated recent activity.
        If full, replace the least active DGIM token.
        """
        # Compute recent estimates for tracked tokens
        current_estimates = {t: dgim.estimate() for t, dgim in self._dgim_instances.items()}

        for token, count in list(self._candidate_tokens.items()):
            # Promote only if candidate is recently active
            if count < self.promotion_threshold:
                continue

            # Case 1: there's room -> just add
            if len(self._dgim_instances) < self.top_k_tokens:
                self._promote_token(token, count)
                continue

            # Case 2: full -> consider replacement
            min_token = min(current_estimates, key=current_estimates.get)
            min_estimate = current_estimates[min_token]

            # Promote only if candidate has stronger recent activity
            if count > min_estimate:
                # Replace least active token
                del self._dgim_instances[min_token]
                self._tracked_tokens.remove(min_token)
                del current_estimates[min_token]

                self._promote_token(token, count)

            # Remove processed candidate
            del self._candidate_tokens[token]

        # Decay remaining candidates
        for token in self._candidate_tokens:
            self._candidate_tokens[token] = max(0, self._candidate_tokens[token] - 1)

    def _promote_token(self, token: str, count: int):
        dgim = DGIM(window_size=self.window_size)
        # Initialize with `count` appearances
        for _ in range(count):
            dgim.push(1)
        self._dgim_instances[token] = dgim
        self._tracked_tokens.add(token)

    def get_burst_terms(self, top_n: Optional[int] = None) -> List[Tuple[str, int]]:
        """
        Get terms that are currently in a burst (above threshold).
        Returns list of (token, recent_count) sorted by recent_count descending.
        """
        if top_n is None:
            top_n = self.report_top_n

        burst_terms = []
        for token, dgim in self._dgim_instances.items():
            recent_count = dgim.estimate()
            if recent_count >= self.burst_threshold:
                burst_terms.append((token, recent_count))

        # Sort by recent count (descending)
        burst_terms.sort(key=lambda x: x[1], reverse=True)
        return burst_terms[:top_n]

    def is_burst(self) -> bool:
        """
        Check if there's currently a burst (any token above threshold).
        """
        for dgim in self._dgim_instances.values():
            if dgim.estimate() >= self.burst_threshold:
                return True
        return False

    def get_burst_summary(self) -> Dict:
        """
        Get a compressed summary of current burst status.
        """
        burst_terms = self.get_burst_terms()

        return {
            "active": self.is_burst(),
            "token": [f"{token}: {count}" for token, count in burst_terms]
        }

    @property
    def message_count(self) -> int:
        return self._message_count

    def __repr__(self) -> str:
        return (
            f"BurstDetector(window_size={self.window_size}, "
            f"tracked={len(self._dgim_instances)}, messages={self._message_count})"
        )