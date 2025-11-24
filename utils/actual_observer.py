from collections import defaultdict

class ActualObserver:
    def __init__(self):
        self.actual_counter = defaultdict(int)

    def observe_message(self, message: str) -> None:
        for token in message.split():
            self.actual_counter[token] += 1

    def get_counts(self, top_k: int = 25) -> list[tuple[str, int]]:
        items = sorted(self.actual_counter.items(), key=lambda x: x[1], reverse=True)
        return items[:top_k]
