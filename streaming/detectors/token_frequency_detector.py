from typing import List, Dict

from streaming.algorithms import CountMinSketch
from streaming.detectors.base_frequency_detector import BaseFrequencyDetector


class TokenFrequencyDetector(BaseFrequencyDetector):
    """
    Per-token frequency detection using a single Count-Min Sketch.
    Original implementation for tracking individual token frequencies.
    """

    def __init__(
        self,
        epsilon: float = 0.005,
        delta: float = 1e-3,
        seed: int = 0,
        top_k: int = 100,
    ) -> None:
        super().__init__(epsilon, delta, seed, top_k)
        self.cms = CountMinSketch.from_error_delta(
            epsilon=epsilon, delta=delta, seed=seed
        )

    def _update_frequencies(self, tokens: List[str]) -> None:
        """Update CMS with token frequencies."""
        for token in tokens:
            self.cms.add(token)

    def estimate_frequency(self, term: str, **kwargs) -> int:
        """Estimate frequency for a token from the single CMS."""
        return self.cms.estimate(term.lower())

    def get_frequency_analysis(self, top_n: int = 10) -> Dict[str, int]:
        """Get top N tokens by frequency from tracked tokens."""
        snapshot = {token: self.cms.estimate(token) for token in self._top_tokens}
        sorted_items = sorted(snapshot.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return dict(sorted_items)

    def __repr__(self) -> str:
        return (
            f"TokenFrequencyDetector(cms={self.cms}, "
            f"top_k={self.top_k}, tracked={len(self._top_tokens)})"
        )
