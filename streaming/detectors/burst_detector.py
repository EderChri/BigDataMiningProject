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

        # DGIM instances for each tracked token
        self._dgim_instances: Dict[str, DGIM] = {}

        # Track which tokens we're monitoring (for efficient updates)
        self._tracked_tokens: Set[str] = set()

        # Min-heap to maintain top K tokens: (estimate, token)
        self._heap: List[Tuple[int, str]] = []

        # Message counter
        self._message_count = 0

    def observe_message(self, text: str) -> None:
        """Process a message and update DGIM instances."""
        tokens = split_preprocessed_tokens(text)
        seen_in_message = set(tokens)

        # Update existing DGIM instances
        for token in self._tracked_tokens:
            if token in self._dgim_instances:
                self._dgim_instances[token].push(1 if token in seen_in_message else 0)

        # Add new tokens if we have capacity
        for token in seen_in_message:
            if token not in self._dgim_instances:
                if len(self._dgim_instances) < self.top_k_tokens:
                    # Add new token
                    self._dgim_instances[token] = DGIM(window_size=self.window_size)
                    self._dgim_instances[token].push(1)
                    self._tracked_tokens.add(token)

        self._message_count += 1

    def update_tracked_tokens(self, candidate_tokens: Iterable[str]) -> None:
        """
        Update which tokens are being tracked based on their current estimates.
        This should be called periodically to add new high-frequency tokens.
        """
        # Get current estimates for all tracked tokens
        current_estimates = {}
        for token in self._tracked_tokens:
            if token in self._dgim_instances:
                current_estimates[token] = self._dgim_instances[token].estimate()

        # Check if we should add any candidate tokens
        for token in candidate_tokens:
            if token in self._dgim_instances:
                continue

            if len(self._dgim_instances) < self.top_k_tokens:
                # Add new token
                dgim = DGIM(window_size=self.window_size)
                self._dgim_instances[token] = dgim
                self._tracked_tokens.add(token)
            else:
                # Find minimum estimate among tracked tokens
                if current_estimates:
                    min_token = min(current_estimates, key=current_estimates.get)
                    min_estimate = current_estimates[min_token]

                    # For new tokens, we can't know their recent count yet,
                    # so we only replace if we're explicitly told to track them
                    # This prevents unnecessary churn
                    pass

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