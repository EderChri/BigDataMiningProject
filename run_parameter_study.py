import json
from pathlib import Path
from typing import List, Dict
from itertools import product
import pandas as pd
import click

from data_loader.dataloader import DataLoader
from data_loader.scc_dataset_loader import SCCDatasetLoader
from plot.boxplot import plot_boxplot_comparison_seaborn
from plot.bump_chart import plot_bump_chart
from plot.line_chart import plot_average_counts_comparison
from plot.size_chart import plot_importance_points
from streaming.algorithms.min_hash_lsh import MinHashLSH
from streaming.message_processor import MessageProcessor
from streaming.streaming_pipeline import StreamingPipeline
from streaming.utils.caching import get_cache_key, load_cached_results, save_results
from streaming.utils.reservoir import Reservoir
from utils.actual_observer import ActualObserver


def iter_preprocessed_messages(conversations: List[Dict], limit: int | None = None,
                               sort_by_time: bool = True):
    """Yield preprocessed message bodies from conversations."""
    messages = [msg for convo in conversations for msg in convo.get("messages", []) if msg.get("body")]
    if sort_by_time:
        messages.sort(key=lambda m: m.get("time", float("inf")))
    for i, msg in enumerate(messages):
        if limit is not None and i >= limit:
            break
        yield msg["body"]


def run_single_experiment(epsilon: float, delta: float, conversations: List[Dict],
                          max_messages: int, exclude_duplicates: bool,
                          update_interval: int, top_frequency: int, seed: int = 42) -> Dict:
    """Run pipeline with specific epsilon and delta values."""
    lsh = MinHashLSH(num_buckets=100, num_hashes=128)
    reservoirs = [Reservoir() for _ in range(lsh.num_buckets)]
    actual_observer = ActualObserver(lsh=lsh, reservoirs=reservoirs)

    pipeline = StreamingPipeline(
        window_size=update_interval * 2,
        actual_observer=actual_observer,
        lsh=lsh,
        reservoirs=reservoirs,
        seed=seed,
        epsilon=epsilon,
        delta=delta
    )

    processor = MessageProcessor(pipeline, exclude_duplicates, False, update_interval, top_frequency)

    for idx, text in enumerate(iter_preprocessed_messages(conversations, limit=max_messages), start=1):
        processor.process_message(text, is_first_snapshot=(processor.state.processed == 0))

    summary = processor.finalize([])
    summary["epsilon"] = epsilon
    summary["delta"] = delta

    return {
        "summary": summary,
        "snapshots": processor.state.snapshots,
        "epsilon": epsilon,
        "delta": delta
    }


@click.command()
@click.option("--data-dir", default="data", type=click.Path(exists=True), required=True)
@click.option("--train-subdir", default="train_convs")
@click.option("--test-subdir", default="test_convs")
@click.option("--split", type=click.Choice(["train", "test"]), default="test")
@click.option("--all-messages/--scammer-only", default=False)
@click.option("--max-messages", type=int, default=2000)
@click.option("--exclude-duplicates/--include-duplicates", default=False)
@click.option("--update-interval", type=int, default=100)
@click.option("--top-frequency", type=int, default=300)
@click.option("--output-dir", default="parameter_study", type=click.Path())
@click.option("--cache-dir", default=".cache/parameter_study", type=click.Path())
def main(data_dir: str, train_subdir: str, test_subdir: str, split: str,
         all_messages: bool, max_messages: int, exclude_duplicates: bool,
         update_interval: int, top_frequency: int, output_dir: str, cache_dir: str):
    # Parameter ranges to sweep
    epsilon_values = [0.01, 0.05, 0.1, 0.2]
    delta_values = [1e-4, 1e-3, 0.01, 0.05]

    # Setup directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Load dataset once
    dataset_loader = SCCDatasetLoader(
        data_dir=data_dir,
        train_data_dir=train_subdir,
        test_data_dir=test_subdir,
        use_skipwords=True
    )
    dataloader = DataLoader([dataset_loader])
    dataloader.load_data(force_reload=False, all_messages=all_messages)
    conversations = dataset_loader.data.get(split, [])

    # Store results
    all_results = []

    # Parameter sweep
    for epsilon, delta in product(epsilon_values, delta_values):
        exp_id = f"eps{epsilon}_delta{delta}"
        cache_file = cache_path / f"{exp_id}_{split}_{max_messages}.pkl"

        click.echo(f"Processing epsilon={epsilon}, delta={delta}...", err=True)

        # Check cache
        cached = load_cached_results(cache_file)
        if cached:
            click.echo(f"  Loaded from cache", err=True)
            result = cached
        else:
            result = run_single_experiment(
                epsilon, delta, conversations, max_messages,
                exclude_duplicates, update_interval, top_frequency
            )
            save_results(cache_file, result["summary"], result["snapshots"])
            click.echo(f"  Saved to cache", err=True)

        all_results.append(result)

        # Generate plots with specific naming
        plot_dir = output_path / exp_id
        plot_dir.mkdir(exist_ok=True)

        plot_bump_chart(
            result["snapshots"],
            nr_msg_per_step=update_interval,
            top_k=5,
            file=str(plot_dir / f"bump_chart_{exp_id}.png")
        )
        plot_importance_points(
            result["snapshots"][1:],
            nr_msg_per_step=update_interval,
            top_k=5,
            file=str(plot_dir / f"importance_{exp_id}.png")
        )
        plot_average_counts_comparison(
            result["snapshots"],
            update_interval,
            file=str(plot_dir / f"avg_counts_{exp_id}.png")
        )
        plot_boxplot_comparison_seaborn(
            result["snapshots"],
            update_interval,
            file=str(plot_dir / f"boxplot_{exp_id}.png")
        )

    # Create consolidated dataframe
    consolidated_data = []
    for result in all_results:
        summary = result["summary"]
        row = {
            "epsilon": result["epsilon"],
            "delta": result["delta"],
            "processed": summary.get("processed", 0),
            "excluded": summary.get("excluded", 0),
            "split": summary.get("split", split),
        }
        # Add any other metrics from summary
        for key, value in summary.items():
            if key not in row and isinstance(value, (int, float, str)):
                row[key] = value
        consolidated_data.append(row)

    df = pd.DataFrame(consolidated_data)
    df = df.sort_values(["epsilon", "delta"])

    # Save consolidated results
    csv_path = output_path / "consolidated_results.csv"
    df.to_csv(csv_path, index=False)
    click.echo(f"\nConsolidated results saved to {csv_path}", err=True)

    # Save summary JSON
    json_path = output_path / "all_summaries.json"
    with open(json_path, "w") as f:
        json.dump([r["summary"] for r in all_results], f, indent=2)

    click.echo(f"Summary JSON saved to {json_path}", err=True)
    click.echo(f"\nCompleted parameter sweep: {len(all_results)} experiments", err=True)


if __name__ == "__main__":
    main()
