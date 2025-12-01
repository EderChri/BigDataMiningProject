import json
from pathlib import Path
from typing import Iterable, Iterator, List, Dict

import click

from data_loader.dataloader import DataLoader
from data_loader.scc_dataset_loader import SCCDatasetLoader
from plot.bump_chart import plot_bump_chart
from plot.line_chart import plot_average_counts_comparison
from plot.size_chart import plot_importance_points
from streaming.algorithms.min_hash_lsh import MinHashLSH
from streaming.message_processor import MessageProcessor
from streaming.streaming_pipeline import StreamingPipeline
from streaming.utils.caching import get_cache_key, load_cached_results, save_results
from streaming.utils.reservoir import Reservoir
from utils.actual_observer import ActualObserver


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
    default=300,
    show_default=True,
    help="Number of top frequent tokens to report in analysis.",
)
@click.option(
    "--force-recalc/--use-cache",
    default=False,
    show_default=True,
    help="Force recalculation even if cached results exist."
)
@click.option(
    "--cache-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=str),
    default=".cache/streaming",
    show_default=True,
    help="Directory to store cached computation results."
)
def main(
        data_dir: str, train_subdir: str, test_subdir: str, split: str,
        all_messages: bool, max_messages: int, freq_queries: Iterable[str],
        show_text: bool, exclude_duplicates: bool, update_interval: int, top_frequency: int,
        cache_dir=str, force_recalc=None) -> None:
    # Setup cache
    cache_key = get_cache_key(data_dir, split, max_messages, all_messages,
                              exclude_duplicates, update_interval)
    cache_file = Path(cache_dir) / f"{cache_key}.pkl"

    # Try loading cache
    if not force_recalc:
        cached = load_cached_results(cache_file)
        if cached:
            click.echo(f"Loaded cached results from {cache_file}", err=True)
            summary = cached['summary']
            snapshots = cached['snapshots']

            # Print and plot
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            plot_bump_chart(snapshots, nr_msg_per_step=update_interval, top_k=5)
            plot_importance_points(snapshots[1:], nr_msg_per_step=update_interval, top_k=5)
            plot_average_counts_comparison(snapshots, update_interval)

            click.echo(f"Processed {summary['processed']} messages from split '{split}'.", err=True)
            return

    dataset_loader = SCCDatasetLoader(data_dir=data_dir, train_data_dir=train_subdir,
                                      test_data_dir=test_subdir, use_skipwords=True)
    dataloader = DataLoader([dataset_loader])
    dataloader.load_data(force_reload=False, all_messages=all_messages)

    conversations = dataset_loader.data.get(split, [])
    lsh = MinHashLSH(num_buckets=100, num_hashes=128)
    reservoirs = [Reservoir() for _ in range(lsh.num_buckets)]
    actual_observer = ActualObserver(lsh=lsh, reservoirs=reservoirs)
    pipeline = StreamingPipeline(window_size=update_interval * 2, actual_observer=actual_observer,
                                 lsh=lsh, reservoirs=reservoirs)

    # Process messages
    processor = MessageProcessor(pipeline, exclude_duplicates, show_text, update_interval, top_frequency)

    for idx, text in enumerate(iter_preprocessed_messages(conversations, limit=max_messages), start=1):
        processor.process_message(text, is_first_snapshot=(processor.state.processed == 0))

    # Generate final results
    summary = processor.finalize(freq_queries)
    summary["split"] = split

    # Save to cache
    save_results(cache_file, summary, processor.state.snapshots)
    click.echo(f"Saved results to {cache_file}", err=True)

    # print(json.dumps(summary, ensure_ascii=False, indent=2))
    plot_bump_chart(processor.state.snapshots, nr_msg_per_step=update_interval, top_k=5)
    plot_importance_points(processor.state.snapshots[1:], nr_msg_per_step=update_interval, top_k=5)
    plot_average_counts_comparison(processor.state.snapshots, update_interval)

    click.echo(f"Processed {processor.state.processed} messages from split '{split}'.", err=True)
    if exclude_duplicates:
        click.echo(f"Excluded {processor.state.excluded} duplicate messages.", err=True)


if __name__ == "__main__":
    main()
