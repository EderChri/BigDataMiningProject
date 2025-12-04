from typing import List, Dict

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_boxplot_comparison_seaborn(snapshots: List[Dict], update_interval: int, file="boxplot_comparison.png") -> None:
    """Seaborn version with better handling of many timepoints"""

    records = []
    for s in snapshots:
        msg_count = s["message_count"]
        for val in s["top_tokens"].values():
            records.append({'Messages': msg_count, 'Category': 'Token Est', 'Count': val})
        for val in s["actual_tokens"].values():
            records.append({'Messages': msg_count, 'Category': 'Token Act', 'Count': val})
        for val in s["top_buckets"].values():
            records.append({'Messages': msg_count, 'Category': 'Bucket Est', 'Count': val})
        for val in s["actual_buckets"].values():
            records.append({'Messages': msg_count, 'Category': 'Bucket Act', 'Count': val})

    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(max(12, len(snapshots) * 2), 6))
    sns.boxplot(data=df, x='Messages', y='Count', hue='Category',
                palette=['#2E86AB', '#A0D2E7', '#A23B72', '#D8A5C8'],
                showfliers=False, ax=ax)
    ax.grid(True, alpha=0.3, axis='y', zorder=0)
    ax.set_xlabel(f'Messages Processed (interval={update_interval})')
    plt.xticks(rotation=45)
    plt.title('Count Distributions: Estimated vs Actual')
    plt.tight_layout()
    plt.savefig(f"{file}", dpi=150, bbox_inches='tight')
