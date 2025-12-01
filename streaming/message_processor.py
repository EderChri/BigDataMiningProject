from dataclasses import dataclass, field
from typing import Set, List, Dict, Iterable

from streaming.utils.token_handler import split_preprocessed_tokens


@dataclass
class AnalysisState:
    """Holds all metrics and accumulated results"""
    processed: int = 0
    excluded: int = 0
    duplicate_count: int = 0
    duplicate_score_sum: float = 0.0
    recent_tokens: Set[str] = field(default_factory=set)
    messages_out: List[Dict] = field(default_factory=list)
    snapshots: List[Dict] = field(default_factory=list)


class MessageProcessor:
    def __init__(self, pipeline, exclude_duplicates: bool, show_text: bool, update_interval: int, top_k: int):
        self.pipeline = pipeline
        self.exclude_duplicates = exclude_duplicates
        self.show_text = show_text
        self.update_interval = update_interval
        self.top_k = top_k
        self.state = AnalysisState()

    def process_message(self, text: str, is_first_snapshot: bool = False) -> None:
        """Process single message and update state"""
        result = self.pipeline.process_message(text, frequency_queries=None)
        dup_info = result.get("duplicate", {}) or {}
        is_duplicate = dup_info.get("is_duplicate", False)

        if self.exclude_duplicates and is_duplicate:
            self.state.excluded += 1
            return

        if is_duplicate:
            self.state.duplicate_count += 1
        self.state.duplicate_score_sum += float(dup_info.get("duplicate_score", 0.0))

        if self.show_text:
            self.state.messages_out.append({
                "text": text, "duplicate": dup_info, "burst": result.get("burst", {})
            })

        self.state.recent_tokens.update(split_preprocessed_tokens(text))
        self.state.processed += 1

        # Auto-snapshot at intervals
        if self.state.processed % self.update_interval == 0:
            self._create_snapshot(is_first_snapshot)

    def _create_snapshot(self, first: bool) -> None:
        """Create snapshot and clear recent tokens"""
        self.pipeline.update_frequency_detectors(recent_tokens=self.state.recent_tokens)

        burst_summary = self.pipeline.burst_detector.detect_bursts(
            recent_k=self.update_interval,
            first=first,
            actual_observer=self.pipeline.actual_observer
        )[:5]

        self.state.snapshots.append({
            "message_count": self.state.processed,
            "top_tokens": self.pipeline.token_frequency_detector.get_frequency_analysis(top_n=self.top_k),
            "top_buckets": self.pipeline.bucket_frequency_detector.get_frequency_analysis(top_n=self.top_k),
            "actual_tokens": self.pipeline.actual_observer.get_formatted_counts(top_k=self.top_k),
            "actual_buckets": self.pipeline.actual_observer.get_formatted_bucket_counts(top_k=self.top_k),
            "burst": burst_summary,
            "duplicates_so_far": self.state.duplicate_count,
        })
        self.state.recent_tokens.clear()

    def finalize(self, freq_queries: Iterable[str]) -> Dict:
        """Generate final summary with actual vs estimated comparison"""
        if self.state.recent_tokens:
            self.pipeline.update_frequency_detectors(recent_tokens=self.state.recent_tokens)

        freq_estimates = self.pipeline.token_frequency_detector.estimate_batch(freq_queries) if freq_queries else {}

        final_burst = self.pipeline.burst_detector.detect_bursts(
            recent_k=self.state.processed % self.update_interval,
            first=False,
            actual_observer=self.pipeline.actual_observer
        )[:5]

        summary = {
            "processed": self.state.processed,
            "excluded_duplicates": self.state.excluded if self.exclude_duplicates else 0,
            "update_interval": self.update_interval,
            "frequency_estimates": freq_estimates,
            "duplicates": {
                "total": self.state.duplicate_count,
                "rate": self.state.duplicate_count / self.state.processed if self.state.processed else 0.0,
                "avg_score": self.state.duplicate_score_sum / self.state.processed if self.state.processed else 0.0,
            },
            "periodic_snapshots": self.state.snapshots,
            "final_burst": final_burst,
            "final_top_tokens": self.pipeline.token_frequency_detector.get_frequency_analysis(top_n=self.top_k),
            "final_top_buckets": self.pipeline.bucket_frequency_detector.get_frequency_analysis(top_n=self.top_k),
            "actual_counts": self.pipeline.actual_observer.get_formatted_counts(),
            "actual_bucket_counts": self.pipeline.actual_observer.get_formatted_bucket_counts(top_k=25),
        }

        if self.show_text:
            summary["messages"] = self.state.messages_out

        return summary
