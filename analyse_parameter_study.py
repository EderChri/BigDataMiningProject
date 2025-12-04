import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load the detailed results
with open("parameter_study/all_summaries.json", "r") as f:
    all_summaries = json.load(f)

output_dir = Path("results/analysis")
output_dir.mkdir(parents=True, exist_ok=True)


## EXTRACT METRICS FROM PERIODIC SNAPSHOTS
def extract_accuracy_metrics(summaries):
    """Extract token and bucket estimation accuracy per epsilon/delta"""
    results = []

    for summary in summaries:
        eps = summary['epsilon']
        delta = summary['delta']

        for snapshot in summary.get('periodic_snapshots', []):
            msg_count = snapshot['message_count']

            # Token estimation errors
            top_tokens = snapshot.get('top_tokens', {})
            actual_tokens = snapshot.get('actual_tokens', {})

            for token in top_tokens.keys():
                if token in actual_tokens:
                    estimated = top_tokens[token]
                    actual = actual_tokens[token]
                    abs_error = abs(estimated - actual)
                    rel_error = abs_error / actual if actual > 0 else 0

                    results.append({
                        'epsilon': eps,
                        'delta': delta,
                        'message_count': msg_count,
                        'type': 'token',
                        'item': token,
                        'estimated': estimated,
                        'actual': actual,
                        'abs_error': abs_error,
                        'rel_error': rel_error
                    })

            # Bucket estimation errors
            top_buckets = snapshot.get('top_buckets', {})
            actual_buckets = snapshot.get('actual_buckets', {})

            top_bucket_ids = list(top_buckets.keys())
            actual_bucket_reps = list(actual_buckets.keys())

            for i, bucket_id in enumerate(top_bucket_ids):
                if i < len(actual_bucket_reps):
                    representative = actual_bucket_reps[i]
                    estimated = top_buckets[bucket_id]
                    actual = actual_buckets[representative]
                    abs_error = abs(estimated - actual)
                    rel_error = abs_error / actual if actual > 0 else 0

                    results.append({
                        'epsilon': eps,
                        'delta': delta,
                        'message_count': msg_count,
                        'type': 'bucket',
                        'item': f"{bucket_id}→{representative}",
                        'estimated': estimated,
                        'actual': actual,
                        'abs_error': abs_error,
                        'rel_error': rel_error
                    })

    return pd.DataFrame(results)


def extract_duplicate_metrics(summaries):
    """Extract duplicate detection stats (parameter-independent)"""
    # Just take the first one since they're all the same
    if len(summaries) > 0:
        dup_info = summaries[0].get('duplicates', {})
        return {
            'total_duplicates': dup_info.get('total', 0),
            'duplicate_rate': dup_info.get('rate', 0),
            'avg_dup_score': dup_info.get('avg_score', 0)
        }
    return {}


def extract_burst_metrics(summaries):
    """Extract burst detection patterns"""
    results = []

    for summary in summaries:
        eps = summary['epsilon']
        delta = summary['delta']

        for snapshot in summary.get('periodic_snapshots', []):
            msg_count = snapshot['message_count']
            bursts = snapshot.get('burst', [])

            for burst in bursts:
                burst_ratio = burst.get('ratio', 0)
                # Only include bursts below the cutoff (< 10)
                results.append({
                    'epsilon': eps,
                    'delta': delta,
                    'message_count': msg_count,
                    'burst_ratio': burst_ratio,
                    'recent_count': burst.get('recent_count', 0),
                    'prev_count': burst.get('prev_count', 0),
                    'actual_rep_count': burst.get('actual_rep_count', 0),
                    'representative': burst.get('representative', ''),
                    'is_cutoff': burst_ratio >= 10.0
                })

    return pd.DataFrame(results)


# Extract all metrics
df_accuracy = extract_accuracy_metrics(all_summaries)
dup_stats = extract_duplicate_metrics(all_summaries)
df_bursts = extract_burst_metrics(all_summaries)

# Save extracted data
df_accuracy.to_csv(output_dir / 'accuracy_metrics.csv', index=False)
df_bursts.to_csv(output_dir / 'burst_metrics.csv', index=False)


## 1. ACCURACY ANALYSIS
def plot_accuracy_heatmaps(df, output_dir):
    """Heatmaps showing mean relative error by epsilon/delta"""

    for item_type in ['token', 'bucket']:
        df_type = df[df['type'] == item_type]

        pivot = df_type.groupby(['epsilon', 'delta'])['rel_error'].mean().reset_index()
        pivot_table = pivot.pivot(index='delta', columns='epsilon', values='rel_error')

        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd',
                    cbar_kws={'label': 'Mean Relative Error'})
        plt.title(f'{item_type.capitalize()} Estimation: Mean Relative Error')
        plt.xlabel('Epsilon (ε)')
        plt.ylabel('Delta (δ)')
        plt.tight_layout()
        plt.savefig(output_dir / f'heatmap_{item_type}_error.png', dpi=300)
        plt.close()


def plot_accuracy_over_time(df, output_dir):
    """Line plots showing how error evolves over message count"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, item_type in enumerate(['token', 'bucket']):
        df_type = df[df['type'] == item_type]

        # Mean absolute error over time
        for (eps, delta), group in df_type.groupby(['epsilon', 'delta']):
            avg_by_msg = group.groupby('message_count')['abs_error'].mean()
            axes[0, idx].plot(avg_by_msg.index, avg_by_msg.values,
                              marker='o', label=f'ε={eps},δ={delta}', alpha=0.7)

        axes[0, idx].set_xlabel('Message Count')
        axes[0, idx].set_ylabel('Mean Absolute Error')
        axes[0, idx].set_title(f'{item_type.capitalize()}: Absolute Error over Time')
        axes[0, idx].legend(fontsize=7, ncol=2)
        axes[0, idx].grid(alpha=0.3)

        # Mean relative error over time
        for (eps, delta), group in df_type.groupby(['epsilon', 'delta']):
            avg_by_msg = group.groupby('message_count')['rel_error'].mean()
            axes[1, idx].plot(avg_by_msg.index, avg_by_msg.values,
                              marker='s', label=f'ε={eps},δ={delta}', alpha=0.7)

        axes[1, idx].set_xlabel('Message Count')
        axes[1, idx].set_ylabel('Mean Relative Error')
        axes[1, idx].set_title(f'{item_type.capitalize()}: Relative Error over Time')
        axes[1, idx].legend(fontsize=7, ncol=2)
        axes[1, idx].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_over_time.png', dpi=300)
    plt.close()


## 2. BURST ANALYSIS - Focus on burst patterns and representatives
def plot_burst_analysis(df, output_dir):
    """Analyze burst detection patterns focusing on representatives and temporal patterns"""

    if len(df) == 0:
        print("No burst data available")
        return

    df_below_cutoff = df[df['burst_ratio'] < 10.0]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Burst Analysis: Bucket Patterns and Distributions from 16 Runs of Parameter Study', fontsize=14, y=0.995)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])

    # 1. Top bursting words (representatives) - SPANS LEFT SIDE
    ax1 = fig.add_subplot(gs[:, 0])  # Spans both rows, left column
    rep_counts = df['representative'].value_counts().head(50)
    ax1.barh(range(len(rep_counts)), rep_counts.values, color='steelblue')
    ax1.set_yticks(range(len(rep_counts)))
    ax1.set_yticklabels(rep_counts.index, fontsize=9)
    ax1.set_xlabel('Burst Count', fontsize=11)
    ax1.set_title('Top 50 Most Frequently Bursting Words', fontsize=12)
    ax1.invert_yaxis()
    ax1.grid(alpha=0.3, axis='x')

    # 2. Burst ratio by top representatives (boxplot) - TOP RIGHT
    ax2 = fig.add_subplot(gs[0, 1])
    top_reps = df['representative'].value_counts().head(10).index
    df_top_reps = df[df['representative'].isin(top_reps)]

    if len(df_top_reps) > 0:
        sns.boxplot(data=df_top_reps, y='representative', x='burst_ratio',
                    order=top_reps, ax=ax2, orient='h')
        ax2.set_xlabel('Burst Ratio')
        ax2.set_ylabel('Representative Word')
        ax2.set_title('Burst Ratio Distribution by Top Words')
        ax2.axvline(x=10, color='red', linestyle='--', alpha=0.5, label='Cutoff')
        ax2.legend()
        ax2.grid(alpha=0.3, axis='x')

    # 3. Histogram: All non-10 burst ratios - BOTTOM RIGHT
    ax3 = fig.add_subplot(gs[1, 1])
    if len(df_below_cutoff) > 0:
        ax3.hist(df_below_cutoff['burst_ratio'], bins=30,
                 color='steelblue', edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Burst Ratio')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Histogram of Non-10 Burst Ratios')
        ax3.axvline(df_below_cutoff['burst_ratio'].mean(), color='red',
                    linestyle='--', linewidth=2,
                    label=f"Mean: {df_below_cutoff['burst_ratio'].mean():.2f}")
        ax3.axvline(df_below_cutoff['burst_ratio'].median(), color='orange',
                    linestyle='--', linewidth=2,
                    label=f"Median: {df_below_cutoff['burst_ratio'].median():.2f}")
        ax3.legend()
        ax3.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'burst_analysis.png', dpi=300)
    plt.close()

    # Additional: Word cloud style table of burst representatives
    burst_rep_summary = df.groupby('representative').agg({
        'burst_ratio': ['count', 'mean', 'max'],
        'recent_count': 'mean',
        'actual_rep_count': 'mean'
    }).reset_index()
    burst_rep_summary.columns = ['representative', 'burst_count', 'avg_ratio',
                                 'max_ratio', 'avg_recent_count', 'avg_actual_count']
    burst_rep_summary = burst_rep_summary.sort_values('burst_count', ascending=False)
    burst_rep_summary.to_csv(output_dir / 'burst_representatives_summary.csv', index=False)

    # Per-config analysis: which words burst in each config
    config_word_analysis = []
    for (eps, delta), group in df.groupby(['epsilon', 'delta']):
        top_words = group['representative'].value_counts().head(5)
        for word, count in top_words.items():
            avg_ratio = group[group['representative'] == word]['burst_ratio'].mean()
            config_word_analysis.append({
                'epsilon': eps,
                'delta': delta,
                'word': word,
                'burst_count': count,
                'avg_burst_ratio': avg_ratio
            })

    config_word_df = pd.DataFrame(config_word_analysis)
    config_word_df.to_csv(output_dir / 'top_burst_words_per_config.csv', index=False)

    print("\n=== Top 10 Most Frequently Bursting Words ===")
    print(burst_rep_summary.head(10).to_string(index=False))


## 3. SUMMARY TABLES
def create_summary_tables(df_accuracy, dup_stats, df_bursts, output_dir):
    """Generate comprehensive summary tables"""

    # Accuracy summary
    accuracy_summary = df_accuracy.groupby(['epsilon', 'delta', 'type']).agg({
        'abs_error': ['mean', 'std', 'max'],
        'rel_error': ['mean', 'std', 'max']
    }).reset_index()
    accuracy_summary.columns = ['_'.join(col).strip('_') for col in accuracy_summary.columns]
    accuracy_summary.to_csv(output_dir / 'accuracy_summary_table.csv', index=False)

    # Best configurations
    token_error = df_accuracy[df_accuracy['type'] == 'token'].groupby(['epsilon', 'delta'])['rel_error'].mean()
    bucket_error = df_accuracy[df_accuracy['type'] == 'bucket'].groupby(['epsilon', 'delta'])['rel_error'].mean()

    comparison = pd.DataFrame({
        'epsilon': token_error.index.get_level_values(0),
        'delta': token_error.index.get_level_values(1),
        'token_rel_error': token_error.values,
        'bucket_rel_error': bucket_error.values
    })
    comparison['combined_error'] = comparison['token_rel_error'] + comparison['bucket_rel_error']
    comparison = comparison.sort_values('combined_error')
    comparison.to_csv(output_dir / 'best_configs_by_accuracy.csv', index=False)

    # Duplicate stats as simple table (parameter-independent)
    dup_df = pd.DataFrame([dup_stats])
    dup_df.to_csv(output_dir / 'duplicate_stats.csv', index=False)

    # Burst summary
    if len(df_bursts) > 0:
        burst_summary = df_bursts.groupby(['epsilon', 'delta']).agg({
            'burst_ratio': ['count', 'mean', 'std', 'max'],
        }).reset_index()
        burst_summary.columns = ['_'.join(col).strip('_') for col in burst_summary.columns]

        # Add count of high bursts
        high_burst_counts = df_bursts[df_bursts['burst_ratio'] >= 8.0].groupby(['epsilon', 'delta']).size()
        burst_summary['high_bursts_count'] = burst_summary.apply(
            lambda row: high_burst_counts.get((row['epsilon'], row['delta']), 0), axis=1
        )
        burst_summary.to_csv(output_dir / 'burst_summary_table.csv', index=False)

    print("\n=== Top 3 Configurations by Accuracy ===")
    print(comparison.head(3).to_string(index=False))

    print("\n=== Duplicate Detection (Parameter-Independent) ===")
    print(dup_df.to_string(index=False))

    return comparison


# Run all analyses
plot_accuracy_heatmaps(df_accuracy, output_dir)
plot_accuracy_over_time(df_accuracy, output_dir)
plot_burst_analysis(df_bursts, output_dir)
summary = create_summary_tables(df_accuracy, dup_stats, df_bursts, output_dir)

print(f"\nAnalysis complete! Results saved to {output_dir}/")
