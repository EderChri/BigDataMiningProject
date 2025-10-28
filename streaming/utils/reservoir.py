import random
from typing import List, Optional


class Reservoir:
    """Simple reservoir sampler per bin."""
    def __init__(self, capacity: int = 5):
        self.capacity = capacity
        self.samples: List[str] = []
        self.seen = 0

    def add(self, item: str):
        self.seen += 1
        if len(self.samples) < self.capacity:
            self.samples.append(item)
        else:
            # replace with probability capacity/seen
            r = random.randrange(self.seen)
            if r < self.capacity:
                self.samples[r] = item

    def representative(self) -> Optional[str]:
        if not self.samples:
            return None
        return self.samples[0]