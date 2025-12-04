from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np


def plot_average_counts_comparison(snapshots: List[Dict], update_interval: int, file="avg_counts_comparison.png") -> None:
    """Plot estimated vs actual average counts for tokens and buckets"""

    # Extract data
    message_counts = [s["message_count"] for s in snapshots]

    # Token stats
    token_estimates_list = [list(s["top_tokens"].values()) for s in snapshots]
    token_actuals_list = [list(s["actual_tokens"].values()) for s in snapshots]

    avg_token_estimates = [np.mean(v) for v in token_estimates_list]
    std_token_estimates = [np.std(v) for v in token_estimates_list]

    avg_token_actuals = [np.mean(v) for v in token_actuals_list]
    std_token_actuals = [np.std(v) for v in token_actuals_list]

    # Bucket stats
    bucket_estimates_list = [list(s["top_buckets"].values()) for s in snapshots]
    bucket_actuals_list = [list(s["actual_buckets"].values()) for s in snapshots]

    avg_bucket_estimates = [np.mean(v) for v in bucket_estimates_list]
    std_bucket_estimates = [np.std(v) for v in bucket_estimates_list]

    avg_bucket_actuals = [np.mean(v) for v in bucket_actuals_list]
    std_bucket_actuals = [np.std(v) for v in bucket_actuals_list]

    # Create single plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Token counts (line + band)
    ax.plot(
        message_counts,
        avg_token_estimates,
        color='#2E86AB',
        linewidth=2,
        label='Token Estimated',
    )
    ax.fill_between(
        message_counts,
        np.array(avg_token_estimates) - np.array(std_token_estimates),
        np.array(avg_token_estimates) + np.array(std_token_estimates),
        color='#2E86AB',
        alpha=0.15,
    )

    ax.plot(
        message_counts,
        avg_token_actuals,
        color='#2E86AB',
        linewidth=2,
        alpha=0.5,
        linestyle='--',
        label='Token Actual',
    )
    ax.fill_between(
        message_counts,
        np.array(avg_token_actuals) - np.array(std_token_actuals),
        np.array(avg_token_actuals) + np.array(std_token_actuals),
        color='#2E86AB',
        alpha=0.08,
        hatch='//',  # Dotted/hatching pattern for actual
        edgecolor='#2E86AB',
        linewidth=0.5,
    )

    # Bucket counts (line + band)
    ax.plot(
        message_counts,
        avg_bucket_estimates,
        color='#A23B72',
        linewidth=2,
        label='Bucket Estimated',
    )
    ax.fill_between(
        message_counts,
        np.array(avg_bucket_estimates) - np.array(std_bucket_estimates),
        np.array(avg_bucket_estimates) + np.array(std_bucket_estimates),
        color='#A23B72',
        alpha=0.15,
    )

    ax.plot(
        message_counts,
        avg_bucket_actuals,
        color='#A23B72',
        linewidth=2,
        alpha=0.5,
        linestyle='--',
        label='Bucket Actual',
    )
    ax.fill_between(
        message_counts,
        np.array(avg_bucket_actuals) - np.array(std_bucket_actuals),
        np.array(avg_bucket_actuals) + np.array(std_bucket_actuals),
        color='#A23B72',
        alpha=0.08,
        hatch='//',  # Dotted/hatching pattern for actual
        edgecolor='#A23B72',
        linewidth=0.5,
    )

    ax.set_xlabel(f'Messages Processed (interval={update_interval})')
    ax.set_ylabel('Average Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.title('Average Estimations vs Actual Counts')
    plt.tight_layout()
    plt.savefig(f"{file}", dpi=150, bbox_inches='tight')
