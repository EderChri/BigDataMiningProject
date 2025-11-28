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

    def get_formatted_counts(self, top_k: int = 25) -> dict[str, int]:
        """
        Return the top_k counts as a dictionary suitable for JSON output.
        Example: {"apple": 42, "banana": 37, ...}
        """
        counts = self.get_counts(top_k)
        return {term: count for term, count in counts}

    def get_token_frequency(self, token: str) -> int:
        return self.actual_counter[token]
