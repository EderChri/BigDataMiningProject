from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np


def plot_average_counts_comparison(snapshots: List[Dict], update_interval: int) -> None:
    """Plot estimated vs actual average counts for tokens and buckets"""

    # Extract data
    message_counts = [s["message_count"] for s in snapshots]

    # Calculate averages
    avg_token_estimates = [np.mean(list(s["top_tokens"].values())) for s in snapshots]
    avg_token_actuals = [np.mean(list(s["actual_tokens"].values())) for s in snapshots]
    avg_bucket_estimates = [np.mean(list(s["top_buckets"].values())) for s in snapshots]
    avg_bucket_actuals = [np.mean(list(s["actual_buckets"].values())) for s in snapshots]

    # Create single plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Token counts
    ax.plot(message_counts, avg_token_estimates, color='#2E86AB', linewidth=2, label='Token Estimated')
    ax.plot(message_counts, avg_token_actuals, color='#2E86AB', linewidth=2, alpha=0.5,
            linestyle='--', label='Token Actual')

    # Bucket counts
    ax.plot(message_counts, avg_bucket_estimates, color='#A23B72', linewidth=2, label='Bucket Estimated')
    ax.plot(message_counts, avg_bucket_actuals, color='#A23B72', linewidth=2, alpha=0.5,
            linestyle='--', label='Bucket Actual')

    ax.set_xlabel(f'Messages Processed (interval={update_interval})')
    ax.set_ylabel('Average Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.title('Average Estimations vs Actual Counts')
    plt.tight_layout()
    plt.savefig('avg_counts_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
