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
        total_count = ranks.loc[term].count()
        marker = get_marker_by_count(total_count)
        for col_name, v in zip(vals.index, vals):
            if valid[col_name]:
                segment_x.append(col_name_to_idx[col_name])  # Use numeric index
                segment_y.append(v)
            else:
                if segment_x:
                    ax.plot(segment_x, segment_y, marker=marker, linewidth=2, color=color)
                    add_labels(ax, segment_x, segment_y, term)
                    segment_x, segment_y = [], []
        if segment_x:
            ax.plot(segment_x, segment_y, marker=marker, linewidth=2, color=color)
            add_labels(ax, segment_x, segment_y, term)


def add_labels(ax, segment_x, segment_y, term):
    if len(segment_x) == 1:
        ax.text(segment_x[0], segment_y[0] - 0.05, term, ha='center', va='bottom', fontsize=7)
    else:
        ax.text(segment_x[0], segment_y[0] - 0.05, term, ha='center', va='bottom', fontsize=7)
        ax.text(segment_x[-1], segment_y[-1] - 0.05, term, ha='center', va='bottom', fontsize=7)


def get_marker_by_count(count):
    """Return a marker symbol based on total appearance count."""
    if count == 1:
        return 'o'  # circle
    elif count == 2:
        return 's'  # square
    elif count == 3:
        return 'D'  # diamond
    elif count == 4:
        return '^'  # triangle_up
    else:
        return '*'  # star for 5 or more


def add_legend_for_markers(ax):
    """Add a horizontal legend below the plot explaining marker shapes."""
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='black', linestyle='None', label='Appears once'),
        Line2D([0], [0], marker='s', color='black', linestyle='None', label='Appears twice'),
        Line2D([0], [0], marker='D', color='black', linestyle='None', label='Appears thrice'),
        Line2D([0], [0], marker='^', color='black', linestyle='None', label='Appears four times'),
        Line2D([0], [0], marker='*', color='black', linestyle='None', label='Appears five times or more')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=5, fontsize=8, frameon=False)



def plot_bump_chart(data_points, nr_msg_per_step=None, top_k=10):
    time_dfs = prepare_time_dfs(data_points, top_k=top_k)
    ranks = compute_ranks(time_dfs)
    fig, ax = plt.subplots(figsize=(11, 6))
    plot_segments(ranks, ax)
    ax.invert_yaxis()
    ax.set_title(f"Sliding Top {top_k} Bursting Terms Over Time")
    ax.set_xlabel("Time Point")
    ax.set_ylabel("Rank (1 = Most Frequent)")
    if nr_msg_per_step:
        x_ticks = range(len(data_points))
        x_labels = [nr_msg_per_step * (i + 1) for i in x_ticks]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
    ax.set_yticks(range(1, top_k + 1))
    add_legend_for_markers(ax)
    fig.tight_layout()
    plt.show()
