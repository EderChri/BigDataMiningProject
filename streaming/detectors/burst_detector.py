from typing import Iterable, Optional, Set

from streaming.algorithms.dgim import DGIM
from streaming.utils.token_handler import split_preprocessed_tokens


class BurstDetector:
    """
    Detects bursts of scam-related tokens using DGIM over a recent window.

    Modes:
      - per_message: push 1 if a message contains any scam token, else 0
      - per_token: push 1 for each scam token occurrence, 0 otherwise (can be very granular)

    Burst decision uses an exponential moving average (EMA) as baseline.

    Note: This detector expects messages to be preprocessed by the data loader.
    """
    def __init__(
        self,
        window_size: int = 256,
        scam_tokens: Optional[Iterable[str]] = None,
        mode: str = "per_message",
        ema_alpha: float = 0.2,
        min_count: int = 3,
        rel_increase: float = 0.5,
    ) -> None:
        if mode not in ("per_message", "per_token"):
            raise ValueError("mode must be 'per_message' or 'per_token'")
        self.dgim = DGIM(window_size=window_size)
        self.scam_tokens: Set[str] = set((t or "").lower() for t in (scam_tokens or []))
        self.mode = mode
        self.ema_alpha = float(ema_alpha)
        self.ema_baseline: float = 0.0
        self.min_count = int(min_count)
        self.rel_increase = float(rel_increase)

    def observe_message(self, text: str) -> None:
        # Expecting 'text' to be preprocessed (space-separated tokens)
        tokens = split_preprocessed_tokens(text)

        if self.mode == "per_message":
            bit = 1 if any(tok in self.scam_tokens for tok in tokens) else 0
            self.dgim.push(bit)
        else:
            # Per-token mode: push a 1 for each scam token; push a 0 for each non-scam token
            # Warning: this can generate many zeros; depending on use-case you may choose to skip zeros.
            for tok in tokens:
                self.dgim.push(1 if tok in self.scam_tokens else 0)

        estimate = self.dgim.estimate()
        # Update EMA baseline
        self.ema_baseline = self.ema_alpha * estimate + (1.0 - self.ema_alpha) * self.ema_baseline

    def estimated_recent_count(self) -> int:
        return self.dgim.estimate()

    def baseline(self) -> float:
        return self.ema_baseline

    def is_burst(self) -> bool:
        est = self.estimated_recent_count()
        base = max(1.0, self.ema_baseline)
        return est >= self.min_count and est >= base * (1.0 + self.rel_increase)

    def __repr__(self) -> str:
        return f"BurstDetector(mode={self.mode}, window={self.dgim.window_size})"
