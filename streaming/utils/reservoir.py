import random
from typing import List, Optional


class Reservoir:
    """Tracks the best representative token per bin with limited memory."""

    def __init__(self):
        self.best_token: Optional[str] = None
        self.best_score: float = -1  # higher is better

    def add(self, token: str, score: float = 1.0):
        """
        Update the representative if token has higher score.
        score could be:
            - 1 for simple presence
            - CMS-estimated frequency
        """
        if score > self.best_score:
            self.best_token = token
            self.best_score = score

    def representative(self) -> Optional[str]:
        return self.best_token
