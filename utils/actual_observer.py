from collections import defaultdict
from typing import Dict, List, Tuple, Optional


class ActualObserver:
    """
    Tracks actual frequencies for both individual tokens and semantic buckets.
    Uses LSH to hash tokens into buckets for bucket-level aggregation.
    """

    def __init__(self, lsh, reservoirs):
        self.lsh = lsh
        self.reservoirs = reservoirs

        # Per-token counts
        self.actual_counter = defaultdict(int)

        # Per-bucket counts
        self.bucket_counter = defaultdict(int)

    def observe_message(self, message: str) -> None:
        """Update both token and bucket counters."""
        for token in message.split():
            # Update token count
            self.actual_counter[token] += 1

            # Hash to bucket and update bucket count
            bucket_id = self.lsh.hash(token)
            self.bucket_counter[bucket_id] += 1

    def get_counts(self, top_k: int = 25) -> List[Tuple[str, int]]:
        """Get top K tokens by frequency."""
        items = sorted(self.actual_counter.items(), key=lambda x: x[1], reverse=True)
        return items[:top_k]

    def get_formatted_counts(self, top_k: int = 25) -> Dict[str, int]:
        """
        Return the top_k token counts as a dictionary suitable for JSON output.
        Example: {"apple": 42, "banana": 37, ...}
        """
        counts = self.get_counts(top_k)
        return {term: count for term, count in counts}

    def get_bucket_counts(self, top_k: int = 25) -> List[Tuple[int, int]]:
        """
        Get top K buckets by frequency.
        Returns list of (bucket_id, count) tuples.
        """
        items = sorted(self.bucket_counter.items(), key=lambda x: x[1], reverse=True)
        return items[:top_k]

    def get_formatted_bucket_counts(self, top_k: int = 25) -> Dict[str, int]:
        """
        Return the top_k bucket counts as a dictionary with bucket representatives.
        Maps bucket representative to total count.
        Example: {"representative_word1": 1523, "representative_word2": 1204, ...}
        """
        bucket_counts = self.get_bucket_counts(top_k)
        result = {}
        for bucket_id, count in bucket_counts:
            rep = self.reservoirs[bucket_id].representative()
            result[rep] = count
        return result

    def get_token_frequency(self, token: str) -> int:
        """Get frequency count for a specific token."""
        return self.actual_counter[token]

    def get_bucket_frequency(self, bucket_id: int) -> int:
        """Get frequency count for a specific bucket."""
        return self.bucket_counter[bucket_id]

    def get_bucket_frequency_by_token(self, token: str) -> int:
        """Get bucket frequency for the bucket containing a given token."""
        bucket_id = self.lsh.hash(token)
        return self.bucket_counter[bucket_id]

    def get_tokens_in_bucket(self, bucket_id: int, top_k: int = 10) -> Dict[str, int]:
        """
        Get top K tokens that belong to a specific bucket.
        Returns dict of token -> count for tokens hashed to this bucket.
        """
        bucket_tokens = {}
        for token, count in self.actual_counter.items():
            if self.lsh.hash(token) == bucket_id:
                bucket_tokens[token] = count

        # Sort and take top K
        sorted_tokens = sorted(bucket_tokens.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return dict(sorted_tokens)
