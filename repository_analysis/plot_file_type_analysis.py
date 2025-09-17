import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


file_types = ["core", "config", "test", "docs", "bench", "build"]


def plot_token_distribution_pie(df: pd.DataFrame, repair_cap: int | None = None):
    work_df = df.copy()

    if repair_cap is not None and repair_cap > 0:
        capped_nl = []
        capped_code = []
        capped_repair = []
        for _, row in work_df.iterrows():
            repair_mean = float(row.get("repair_input_mean", 0.0))
            nl_mean = float(row.get("nl_mean", 0.0))
            code_mean = float(row.get("code_mean", 0.0))
            if repair_mean <= 0:
                capped_repair.append(0.0)
                capped_nl.append(0.0)
                capped_code.append(0.0)
                continue
            cap = min(repair_mean, float(repair_cap))
            scale = cap / repair_mean
            capped_repair.append(cap)
            capped_nl.append(nl_mean * scale)
            capped_code.append(code_mean * scale)
        work_df["repair_capped"] = capped_repair
        work_df["nl_capped"] = capped_nl
        work_df["code_capped"] = capped_code
        total_repair = work_df["repair_capped"].sum()
        total_nl = work_df["nl_capped"].sum()
        total_code = work_df["code_capped"].sum()
    else:
        total_repair = float(work_df["repair_input_mean"].sum())
        total_nl = float(work_df["nl_mean"].sum())
        total_code = float(work_df["code_mean"].sum())

    total_ranking = float(work_df["ranking_input_mean"].sum())

    total_sum = total_ranking + total_nl + total_code
    if total_sum <= 0:
        raise ValueError("No tokens available to plot.")

    sizes = [total_ranking, total_nl, total_code]
    labels = ["Ranking", "Repair NL", "Repair Code"]
    colors = ["#4C78A8", "#E45756", "#72B7B2"]

    sns.set(style="white", context="talk")
    fig, ax = plt.subplots(figsize=(6, 4))

    # Ensure Repair NL slice has a minimum visible size (visual only)
    visual_sizes = sizes.copy()
    if total_sum > 0:
        min_visible_frac = 0.005
        min_visible_size = min_visible_frac * total_sum
        # indices: 0 Ranking, 1 Repair NL, 2 Repair Code
        if visual_sizes[1] < min_visible_size:
            delta = min_visible_size - visual_sizes[1]
            # borrow from Repair Code first, then Ranking
            borrow_from_code = min(delta, max(0.0, visual_sizes[2] - 1e-9))
            visual_sizes[2] -= borrow_from_code
            visual_sizes[1] += borrow_from_code
            remaining = delta - borrow_from_code
            if remaining > 0:
                borrow_from_rank = min(remaining, max(0.0, visual_sizes[0] - 1e-9))
                visual_sizes[0] -= borrow_from_rank
                visual_sizes[1] += borrow_from_rank

    wedges, _ = ax.pie(
        visual_sizes,
        labels=None,
        colors=colors,
        startangle=90,
        radius=1.0,
        wedgeprops=dict(width=0.5, edgecolor="white"),
    )

    def pct(v):
        return f"{(100.0 * v / total_sum):.1f}%" if total_sum > 0 else "0.0%"

    legend_labels = [
        f"Ranking ({pct(total_ranking)})",
        f"Repair NL ({pct(total_nl)})",
        f"Repair Code ({pct(total_code)})",
    ]
    ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.suptitle("Distribution of Ranking and Repair Tokens", fontsize=16)

    centre_circle = plt.Circle((0, 0), 0.32, fc="white")
    fig.gca().add_artist(centre_circle)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()


def plot_mean_file_type_distribution(df):
    # Sum over repositories: get total mean tokens for each file type
    file_type_means = {
        file_type: df[f"{file_type}_mean"].sum() for file_type in file_types
    }

    total = sum(file_type_means.values())
    if total == 0:
        raise ValueError("Total token count for file types is zero.")

    # Convert to percentage
    percentages = {
        file_type: 100 * count / total for file_type, count in file_type_means.items()
    }

    data = pd.DataFrame(
        {
            "File Type": list(percentages.keys()),
            "Percentage": list(percentages.values()),
        }
    ).sort_values(by="Percentage", ascending=True)

    sns.set(style="whitegrid", context="talk")
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(
        data=data, x="File Type", y="Percentage", hue="File Type", palette="pastel"
    )

    for p, pct in zip(ax.patches, data["Percentage"]):
        ax.annotate(
            f"{pct:.2f}%",
            (p.get_x() + p.get_width() / 2.0, pct),
            ha="center",
            va="bottom",
            fontsize=12,
        )

    ax.set_ylim(0, max(data["Percentage"]) * 1.1)
    ax.set_title("File-Type Token Distribution", fontsize=16)

    plt.tight_layout()
    plt.show()


def plot_stacked_token_counts_by_repo(df):
    # Filter relevant columns
    cols = ["repository", "code_mean"] + [f"{ft}_mean" for ft in file_types]
    df = df[cols].copy()

    # Normalize: ensure file-type counts sum to code_mean (minor correction for rounding errors)
    for idx, row in df.iterrows():
        ft_sum = sum(row[f"{ft}_mean"] for ft in file_types)
        code_total = row["code_mean"]
        if ft_sum == 0:
            raise ValueError(
                f"Repository {row['repository']} has zero file-type token count."
            )
        scale = code_total / ft_sum
        for ft in file_types:
            df.at[idx, f"{ft}_mean"] *= scale

    # Sort by total code_mean ascending
    df = df.sort_values("code_mean", ascending=True).reset_index(drop=True)

    df_stacked = df.set_index("repository")[[f"{ft}_mean" for ft in file_types]]
    df_stacked.columns = file_types  # drop "_mean" suffix

    sns.set(style="whitegrid", context="talk")
    ax = df_stacked.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="tab20c")

    ax.set_ylabel("Token Count")
    ax.set_title("Token Count per Repository by File Type", fontsize=16)
    ax.legend(title="File Type", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        help="CSV file with repository statistics",
        default="/Users/nicolashrubec/dev/agent-state-management/data/token_analysis/file_type_SWE-bench_Verified_test.csv",
    )
    parser.add_argument(
        "--repair_cap",
        type=int,
        default=None,
        help="Optional per-repository cap for repair tokens (scales NL/Code proportionally). Default: unlimited",
    )
    args = parser.parse_args()
    data = pd.read_csv(args.file)
    plot_token_distribution_pie(data, repair_cap=args.repair_cap)
    plot_mean_file_type_distribution(data)
    plot_stacked_token_counts_by_repo(data)
