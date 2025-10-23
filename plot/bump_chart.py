import pandas as pd
import matplotlib.pyplot as plt
import itertools

def plot_bump_chart(data_points, nr_msg_per_step=None):
    time_dfs = []
    time_labels = [f"time_{i}" for i in range(len(data_points))]

    # collect top10 per time point
    for i, d in enumerate(data_points):
        burst_dict = {k.split(":")[0]: int(k.split(":")[1]) for k in d["burst"]["token"]}
        df = pd.DataFrame(
            sorted(burst_dict.items(), key=lambda x: x[1], reverse=True)[:10],
            columns=["term", f"time_{i}"]
        ).set_index("term")
        time_dfs.append(df)

    df_all = pd.concat(time_dfs, axis=1).fillna(0)
    ranks = df_all.rank(axis=0, ascending=False, method="first")

    plt.figure(figsize=(11, 6))

    # unique color per term
    unique_terms = ranks.index.tolist()
    color_map = {term: c for term, c in zip(unique_terms, itertools.cycle(plt.cm.tab20.colors))}

    for term in ranks.index:
        vals = ranks.loc[term]
        valid = vals.notna() & (vals <= 10)
        if not valid.any():
            continue
        color = color_map[term]
        segment_x, segment_y = [], []
        for x, v in zip(vals.index, vals):
            if valid[x]:
                segment_x.append(x)
                segment_y.append(v)
            else:
                if segment_x:
                    plt.plot(segment_x, segment_y, marker='o', linewidth=2, color=color)
                    # labeling
                    if len(segment_x) == 1:
                        plt.text(segment_x[0], segment_y[0]-0.15, term, ha='center', va='bottom', fontsize=8)
                    else:
                        plt.text(segment_x[0], segment_y[0]-0.15, term, ha='center', va='bottom', fontsize=8)
                        plt.text(segment_x[-1], segment_y[-1]-0.15, term, ha='center', va='bottom', fontsize=8)
                    segment_x, segment_y = [], []
        if segment_x:
            plt.plot(segment_x, segment_y, marker='o', linewidth=2, color=color)
            if len(segment_x) == 1:
                plt.text(segment_x[0], segment_y[0]-0.15, term, ha='center', va='bottom', fontsize=8)
            else:
                plt.text(segment_x[0], segment_y[0]-0.15, term, ha='center', va='bottom', fontsize=8)
                plt.text(segment_x[-1], segment_y[-1]-0.15, term, ha='center', va='bottom', fontsize=8)

    plt.gca().invert_yaxis()
    plt.title("Sliding Top 10 Terms Over Time")
    plt.xlabel("Time Point")
    plt.ylabel("Rank (1 = Most Frequent)")
    if nr_msg_per_step:
        x_ticks = range(len(data_points))
        x_labels = [nr_msg_per_step * (i + 1) for i in x_ticks]
        plt.xticks(x_ticks, x_labels)
    plt.yticks(range(1, 11, 1))
    plt.tight_layout()
    plt.show()
