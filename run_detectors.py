import json
from typing import Iterable, Iterator, List, Dict

import click

from data_loader.dataloader import DataLoader
from data_loader.scc_dataset_loader import SCCDatasetLoader
from streaming.streaming_pipeline import StreamingPipeline


def iter_preprocessed_messages(conversations: List[Dict], limit: int | None = None) -> Iterator[str]:
    """
    Yield preprocessed message bodies from a list of conversations up to an optional limit.
    Each conversation is expected to have 'messages', where each message contains 'body' as a preprocessed string.
    """
    count = 0
    for convo in conversations:
        for msg in convo.get("messages", []):
            text = msg.get("body") or ""
            if not text:
                continue
            yield text
            count += 1
            if limit is not None and count >= limit:
                return


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
def main(
    data_dir: str,
    train_subdir: str,
    test_subdir: str,
    split: str,
    all_messages: bool,
    max_messages: int,
    freq_queries: Iterable[str],
) -> None:
    """
    Load preprocessed messages using the dataloader and stream them through the detectors.
    Prints one JSON line per processed message with detector outputs.
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

    # Stream messages
    processed = 0
    for text in iter_preprocessed_messages(conversations, limit=max_messages):
        out = pipeline.process_message(text, frequency_queries=freq_queries if freq_queries else None)
        record = {
            "text": text,
            "frequencies": out.get("frequencies", {}),
            "burst": out.get("burst", {}),
            "duplicate": out.get("duplicate", {}),
        }
        print(json.dumps(record, ensure_ascii=False))
        processed += 1

    # Print summary to stderr-like output using click.echo with err=True
    click.echo(f"Processed {processed} messages from split '{split}'.", err=True)


if __name__ == "__main__":
    main()
