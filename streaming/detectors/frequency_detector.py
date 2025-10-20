from typing import Dict, Iterable, List, Tuple

from streaming.algorithms.count_min_sketch import CountMinSketch
from streaming.utils.token_handler import split_preprocessed_tokens


def generate_ngrams(tokens: List[str], ngram_range: Tuple[int, int]) -> Iterable[str]:
    lo, hi = ngram_range
    n = len(tokens)
    for k in range(max(1, lo), max(1, hi) + 1):
        if k == 1:
            for tok in tokens:
                yield tok
        else:
            for i in range(0, n - k + 1):
                yield " ".join(tokens[i : i + k])


class FrequencyDetector:
    """
    Maintains approximate frequencies of tokens and phrases using Count-Min Sketch.

    Note: Expects messages to be preprocessed by the data loader.
    """

    def __init__(self, epsilon: float = 0.005, delta: float = 1e-3, ngram_range: Tuple[int, int] = (1, 1), seed: int = 0) -> None:
        self.cms = CountMinSketch.from_error_delta(epsilon=epsilon, delta=delta, seed=seed)
        self.ngram_range = ngram_range

    def observe_message(self, text: str) -> None:
        # Expecting 'text' to be preprocessed (space-separated tokens)
        tokens = split_preprocessed_tokens(text)
        for gram in generate_ngrams(tokens, self.ngram_range):
            self.cms.add(gram)

    def estimate_frequency(self, term: str) -> int:
        return self.cms.estimate(term.lower())

    def estimate_batch(self, terms: Iterable[str]) -> Dict[str, int]:
        return {t: self.estimate_frequency(t) for t in terms}

    def __repr__(self) -> str:
        return f"FrequencyDetector(cms={self.cms}, ngram_range={self.ngram_range})"
