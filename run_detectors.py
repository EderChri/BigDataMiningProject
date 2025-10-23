import json
from collections import Counter
from typing import Iterable, Iterator, List, Dict, Set

import click

from data_loader.dataloader import DataLoader
from data_loader.scc_dataset_loader import SCCDatasetLoader
from plot.bump_chart import plot_bump_chart
from streaming.streaming_pipeline import StreamingPipeline
from streaming.utils.token_handler import split_preprocessed_tokens


def iter_preprocessed_messages(
        conversations: List[Dict],
        limit: int | None = None,
        sort_by_time: bool = True
    ) -> Iterator[str]:
    """
    Yield preprocessed message bodies from a list of conversations up to an optional limit.
    Each message is expected to have 'body' and 'time'.
    If sort_by_time=True, messages are yielded sorted by their 'time' field across all conversations.
    """
    messages = [
        msg for convo in conversations for msg in convo.get("messages", []) if msg.get("body")
    ]

    if sort_by_time:
        messages.sort(key=lambda m: m.get("time", float("inf")))

    for i, msg in enumerate(messages):
        if limit is not None and i >= limit:
            break
        yield msg["body"]



@click.command()
@click.option(
    "--data-dir",
    default="data",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, path_type=str),
    required=True,
    help="Root directory containing the dataset splits.",
)
@click.option(
    "--train-subdir",
    type=str,
    default="train_convs",
    show_default=True,
    help="Subdirectory name for the training split under data-dir.",
)
@click.option(
    "--test-subdir",
    type=str,
    default="test_convs",
    show_default=True,
    help="Subdirectory name for the test split under data-dir.",
)
@click.option(
    "--split",
    type=click.Choice(["train", "test"]),
    default="test",
    show_default=True,
    help="Which split to stream through the detectors.",
)
@click.option(
    "--all-messages/--scammer-only",
    default=False,
    show_default=True,
    help="Include all messages vs. keeping the default filtered subset.",
)
@click.option(
    "--max-messages",
    type=int,
    default=200,
    show_default=True,
    help="Maximum number of messages to process (per run).",
)
@click.option(
    "--freq-query",
    "freq_queries",
    multiple=True,
    help="Add a term to be queried in the frequency detector. Repeat for multiple terms.",
)
@click.option(
    "--show-text/--hide-text",
    "show_text",
    default=False,
    show_default=True,
    help="Include original message text in the final aggregated output.",
)
@click.option(
    "--exclude-duplicates/--include-duplicates",
    "exclude_duplicates",
    default=False,
    show_default=True,
    help="Exclude messages detected as duplicates by the Bloom Filter.",
)
@click.option(
    "--update-interval",
    type=int,
    default=100,
    show_default=True,
    help="Number of messages between periodic updates (for top tokens and burst analysis).",
)
@click.option(
    "--top-frequency",
    type=int,
    default=10,
    show_default=True,
    help="Number of top frequent tokens to report in analysis.",
)
def main(
        data_dir: str,
        train_subdir: str,
        test_subdir: str,
        split: str,
        all_messages: bool,
        max_messages: int,
        freq_queries: Iterable[str],
        show_text: bool,
        exclude_duplicates: bool,
        update_interval: int,
        top_frequency: int,
) -> None:
    """
    Load preprocessed messages using the dataloader and stream them through the detectors.
    Outputs aggregated statistics with periodic snapshots every N messages.
    """
    # Initialize dataset loader and dataloader
    dataset_loader = SCCDatasetLoader(
        data_dir=data_dir,
        train_data_dir=train_subdir,
        test_data_dir=test_subdir,
        use_skipwords=True,
    )
    dl = DataLoader([dataset_loader])
    dl.load_data(force_reload=False, all_messages=all_messages)

    # Select conversations for the chosen split
    conversations = dataset_loader.data.get(split, [])

    # Initialize streaming pipeline with default detectors
    pipeline = StreamingPipeline()

    # Stream messages and aggregate outputs
    processed = 0
    excluded = 0
    duplicate_count = 0
    duplicate_score_sum = 0.0
    messages_out: List[Dict] = []

    # Periodic snapshots
    snapshots: List[Dict] = []
    recent_tokens: Set[str] = set()


    for text in iter_preprocessed_messages(conversations, limit=max_messages):
        # Process message
        out = pipeline.process_message(text, frequency_queries=None)

        dup_info = out.get("duplicate", {}) or {}
        is_duplicate = dup_info.get("is_duplicate", False)

        # Check if we should exclude this message
        if exclude_duplicates and is_duplicate:
            excluded += 1
            continue

        # Track duplicates
        if is_duplicate:
            duplicate_count += 1
        duplicate_score_sum += float(dup_info.get("duplicate_score", 0.0))

        last_burst = out.get("burst", {}) or {}

        # Collect tokens from this message for periodic updates
        tokens = split_preprocessed_tokens(text)
        recent_tokens.update(tokens)

        if show_text:
            messages_out.append(
                {
                    "text": text,
                    "duplicate": dup_info,
                    "burst": last_burst,
                }
            )

        processed += 1

        # Periodic update every N messages
        if processed % update_interval == 0:
            # Update top K tokens tracking in frequency detector
            pipeline.frequency_detector.periodic_update(recent_tokens)

            # Sync burst detector with high-frequency tokens
            pipeline.sync_detectors(recent_tokens)

            # Get snapshots from both detectors
            top_tokens = pipeline.frequency_detector.get_frequency_analysis(top_n=top_frequency)
            burst_summary = pipeline.burst_detector.get_burst_summary()

            snapshot = {
                "message_count": processed,
                "top_10_tokens": top_tokens,
                "burst": burst_summary,
                "duplicates_so_far": duplicate_count,
            }
            snapshots.append(snapshot)

            # Clear recent tokens for next period
            recent_tokens.clear()

    # Final update if there are remaining tokens
    if recent_tokens:
        pipeline.frequency_detector.periodic_update(recent_tokens)
        pipeline.sync_detectors(recent_tokens)

    # Final frequency estimates (aggregated)
    freq_estimates: Dict[str, int] = {}
    if freq_queries:
        freq_estimates = pipeline.frequency_detector.estimate_batch(freq_queries)

    # Get final analysis
    final_top_tokens = pipeline.frequency_detector.get_frequency_analysis(top_n=top_frequency)
    final_burst = pipeline.burst_detector.get_burst_summary()

    # Build aggregated summary
    summary = {
        "split": split,
        "processed": processed,
        "excluded_duplicates": excluded if exclude_duplicates else 0,
        "update_interval": update_interval,
        "frequency_estimates": freq_estimates,
        "duplicates": {
            "total": duplicate_count,
            "rate": (duplicate_count / processed) if processed else 0.0,
            "avg_score": (duplicate_score_sum / processed) if processed else 0.0,
        },
        "periodic_snapshots": snapshots,
        "final burst": final_burst,
        "final top_tokens": final_top_tokens,
    }
    if show_text:
        summary["messages"] = messages_out

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    plot_bump_chart(snapshots, nr_msg_per_step=update_interval)

    # Print summary to stderr
    click.echo(f"Processed {processed} messages from split '{split}'.", err=True)
    if exclude_duplicates:
        click.echo(f"Excluded {excluded} duplicate messages.", err=True)


if __name__ == "__main__":
    main()