import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


file_types = ["core", "config", "test", "docs", "bench", "build"]


def plot_token_type_percentages(df):
    include_file_types = ["core", "config"]
    exclude_file_types = list(set(file_types) - set(include_file_types))

    for file_type in exclude_file_types:
        file_type_col = f"{file_type}_mean"
        df["code_mean"] = df["code_mean"] - df[file_type_col]

    total_nl = df["nl_mean"].sum()
    total_code = df["code_mean"].sum()
    total_tokens = total_nl + total_code

    data = pd.DataFrame(
        {
            "Token Type": ["Natural Language", "Code"],
            "Percentage": [
                100 * total_nl / total_tokens,
                100 * total_code / total_tokens,
            ],
        }
    )

    sns.set(style="whitegrid", context="talk")
    plt.figure(figsize=(6, 6))
    ax = sns.barplot(
        data=data, x="Token Type", y="Percentage", hue="Token Type", palette="muted"
    )

    # Set y-axis to max of data + small margin
    max_pct = data["Percentage"].max()
    ax.set_ylim(0, max_pct * 1.1)

    ax.set_title(
        "Natural Language Instructions vs Code Tokens", fontsize=16
    )

    # Show 3 decimal places
    for p, pct in zip(ax.patches, data["Percentage"]):
        ax.annotate(
            f"{pct:.3f}%",
            (p.get_x() + p.get_width() / 2.0, pct),
            ha="center",
            va="bottom",
            fontsize=12,
        )

    plt.tight_layout()
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
    args = parser.parse_args()
    data = pd.read_csv(args.file)
    plot_token_type_percentages(data)
    plot_mean_file_type_distribution(data)
    plot_stacked_token_counts_by_repo(data)
