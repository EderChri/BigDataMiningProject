import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools


def prepare_time_dfs(data_points, top_k=10):
    time_dfs = []
    for i, d in enumerate(data_points):
        top_k_tok = sorted(d["burst"], key=lambda x: x["ratio"], reverse=True)[:top_k]
        burst_dict = {b["representative"]: b["ratio"] for b in top_k_tok}
        df = pd.DataFrame(list(burst_dict.items()), columns=["term", f"time_{i}"]).set_index("term")
        time_dfs.append(df)
    return time_dfs


def compute_ranks(time_dfs):
    df_all = pd.concat(time_dfs, axis=1, sort=False)
    ranks = df_all.rank(axis=0, ascending=False, method="first")
    return ranks


def plot_segments(ranks, ax, time_dfs, max_cap):
    unique_terms = ranks.index.tolist()
    col_name_to_idx = {col: idx for idx, col in enumerate(ranks.columns)}

    for term in ranks.index:
        vals = ranks.loc[term]
        valid = vals.notna() & (vals <= 10)
        if not valid.any():
            continue

        segment_x, segment_y = [], []

        for col_name in vals.index:
            if valid[col_name]:
                t_idx = int(col_name.replace("time_", ""))
                ratio = time_dfs[t_idx].loc[term, col_name] if term in time_dfs[t_idx].index else 0

                x_pos = col_name_to_idx[col_name]
                y_pos = vals[col_name]

                # Check if ratio is exactly 10 - display as red square
                if ratio == 10:
                    size = 100  # fixed size for ratio 10
                    color = 'red'
                    marker = 's'  # square
                else:
                    # define normal range
                    min_ratio, max_ratio = 2, 8
                    min_size, max_size = 20, 1000
                    if ratio > max_ratio:
                        size = max_size  # cap outliers
                    else:
                        # linear scaling within normal range
                        size = min_size + (ratio - min_ratio) / (max_ratio - min_ratio) * (max_size - min_size)
                        size = max(size, min_size)
                    color = 'black'
                    marker = 'o'

                # Plot each point immediately with its specific color and marker
                ax.scatter([x_pos], [y_pos], s=size, color=color, marker=marker, zorder=3)
                ax.text(x_pos, y_pos - 0.10, term, ha='center', va='bottom', fontsize=7)


def plot_importance_points(data_points, nr_msg_per_step=None, top_k=10, file="size_chart.png"):
    time_dfs = prepare_time_dfs(data_points, top_k=top_k)
    ranks = compute_ranks(time_dfs)
    ratios = []
    for t_df in time_dfs:
        ratios.extend(t_df.values.flatten())
    max_cap = np.percentile(ratios, 95) * 10
    fig, ax = plt.subplots(figsize=(11, 6))
    plot_segments(ranks, ax, time_dfs, max_cap)
    ax.invert_yaxis()
    ax.set_title(f"Sliding Top {top_k} Bursting Terms Over Time")
    ax.set_xlabel("Time Point")
    ax.set_ylabel("Rank (1 = Most Frequent)")
    if nr_msg_per_step:
        x_ticks = range(len(data_points))
        x_labels = [nr_msg_per_step * (i + 2) for i in x_ticks]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
    ax.set_yticks(range(1, top_k + 1))
    fig.tight_layout()
    plt.savefig(f"{file}")
