from streaming.algorithms import CountMinSketch, DGIMManager
from streaming.algorithms.min_hash_lsh import MinHashLSH
from streaming.utils.reservoir import Reservoir


class BurstDetector:
    def __init__(self, num_semantic_buckets=100, window_size=100):
        # LSH for semantic grouping
        self.lsh = MinHashLSH(num_buckets=num_semantic_buckets, num_hashes=128)
        self.window_size = window_size
        self.counter = 0

        # One CMS per semantic bucket
        self.cms_per_bucket = [
            CountMinSketch.from_error_delta(epsilon=0.01, delta=0.001)
            for _ in range(num_semantic_buckets)
        ]

        # DGIM manager for all buckets
        self.dgim = DGIMManager(num_bins=num_semantic_buckets, window_size=window_size)

        # Reservoirs for representatives
        self.reservoirs = [Reservoir() for _ in range(num_semantic_buckets)]

    def observe_message(self, message: str):
        tokens = message.split()

        # Advance sliding window
        self.dgim.tick()
        self.counter += 1

        for token in tokens:
            # Hash to semantic bucket
            bucket_id = self.lsh.hash(token)

            # Update CMS and get frequency
            self.cms_per_bucket[bucket_id].add(token, count=1)
            freq = self.cms_per_bucket[bucket_id].estimate(token)

            # Update representative
            self.reservoirs[bucket_id].add(token, score=freq)

            # Track activity in DGIM
            self.dgim.add_one(bucket_id)

    def detect_bursts(self, recent_k=100, prev_k=None, threshold=0.8, min_count=10, first=False, actual_observer=None):

        if recent_k >= self.window_size:
            raise ValueError("recent_k must be smaller than window_size")

        if recent_k is None:
            recent_k = self.window_size // 2
        if prev_k is None:
            prev_k = recent_k

        eps = 1e-6
        bursts = []
        for bucket_id in range(len(self.reservoirs)):
            recent = self.dgim.count_last(bucket_id, k=recent_k)
            if recent < min_count:
                continue
            if first:
                ratio = recent
                prev = 0
            else:
                prev_total = self.dgim.count_last(bucket_id, k=recent_k + prev_k)
                prev = max(0, prev_total - recent)  # counts in the previous window
                ratio = min((recent + eps) / (prev + eps), 10.0)
            if ratio >= threshold:
                rep = self.reservoirs[bucket_id].representative()
                bursts.append({
                    'bin': bucket_id,
                    "ratio": ratio,
                    "recent_count": recent,
                    "prev_count": prev,
                    "actual_rep_count": actual_observer.get_token_frequency(rep),
                    'representative': rep
                })
        # sort by significance
        bursts.sort(key=lambda x: (-x["ratio"], -x["recent_count"]))
        return bursts
