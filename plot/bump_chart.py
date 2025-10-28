import pandas as pd
import matplotlib.pyplot as plt
import itertools

def prepare_time_dfs(data_points):
    time_dfs = []
    for i, d in enumerate(data_points):
        top10 = sorted(d["burst"], key=lambda x: x["ratio"], reverse=True)[:10]
        burst_dict = {b["representative"]: b["ratio"] for b in top10}
        df = pd.DataFrame(list(burst_dict.items()), columns=["term", f"time_{i}"]).set_index("term")
        time_dfs.append(df)
    return time_dfs

def compute_ranks(time_dfs):
    df_all = pd.concat(time_dfs, axis=1, sort=False)
    ranks = df_all.rank(axis=0, ascending=False, method="first")
    return ranks

def plot_segments(ranks, ax):
    unique_terms = ranks.index.tolist()
    color_map = {term: c for term, c in zip(unique_terms, itertools.cycle(plt.cm.tab20.colors))}
    col_name_to_idx = {col: idx for idx, col in enumerate(ranks.columns)}

    for term in ranks.index:
        vals = ranks.loc[term]
        valid = vals.notna() & (vals <= 10)
        if not valid.any():
            continue
        color = color_map[term]
        segment_x, segment_y = [], []
        for col_name, v in zip(vals.index, vals):
            if valid[col_name]:
                segment_x.append(col_name_to_idx[col_name])  # Use numeric index
                segment_y.append(v)
            else:
                if segment_x:
                    ax.plot(segment_x, segment_y, marker='o', linewidth=2, color=color)
                    add_labels(ax, segment_x, segment_y, term)
                    segment_x, segment_y = [], []
        if segment_x:
            ax.plot(segment_x, segment_y, marker='o', linewidth=2, color=color)
            add_labels(ax, segment_x, segment_y, term)


def add_labels(ax, segment_x, segment_y, term):
    if len(segment_x) == 1:
        ax.text(segment_x[0], segment_y[0] - 0.15, term, ha='center', va='bottom', fontsize=8)
    else:
        ax.text(segment_x[0], segment_y[0] - 0.15, term, ha='center', va='bottom', fontsize=8)
        ax.text(segment_x[-1], segment_y[-1] - 0.15, term, ha='center', va='bottom', fontsize=8)

def plot_bump_chart(data_points, nr_msg_per_step=None):
    time_dfs = prepare_time_dfs(data_points)
    ranks = compute_ranks(time_dfs)
    fig, ax = plt.subplots(figsize=(11, 6))
    plot_segments(ranks, ax)
    ax.invert_yaxis()
    ax.set_title("Sliding Top 10 Terms Over Time")
    ax.set_xlabel("Time Point")
    ax.set_ylabel("Rank (1 = Most Frequent)")
    if nr_msg_per_step:
        x_ticks = range(len(data_points))
        x_labels = [nr_msg_per_step * (i + 1) for i in x_ticks]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
    ax.set_yticks(range(1, 11))
    fig.tight_layout()
    plt.show()
