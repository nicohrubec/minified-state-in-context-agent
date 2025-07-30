import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


file_types = ["core", "test", "docs", "config", "bench", "build"]


def plot_token_type_percentages(df):
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
    ax = sns.barplot(data=data, x="Token Type", y="Percentage", palette="muted")

    # Set y-axis to max of data + small margin
    max_pct = data["Percentage"].max()
    ax.set_ylim(0, max_pct * 1.1)

    ax.set_title(
        "Percentage of Natural Language Instructions vs Code Tokens", fontsize=16
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

    data = pd.DataFrame({
        "File Type": list(percentages.keys()),
        "Percentage": list(percentages.values()),
    })

    sns.set(style="whitegrid", context="talk")
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(data=data, x="File Type", y="Percentage", palette="pastel")

    for p, pct in zip(ax.patches, data["Percentage"]):
        ax.annotate(
            f"{pct:.2f}%",
            (p.get_x() + p.get_width() / 2.0, pct),
            ha="center",
            va="bottom",
            fontsize=12,
        )

    ax.set_ylim(0, max(data["Percentage"]) * 1.1)
    ax.set_title("Mean File-Type Token Distribution Across Repositories", fontsize=16)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        help="CSV file with repository statistics",
        default="/Users/nicolashrubec/dev/agent-state-management/data/token-analysis/SWE-bench_Verified_test_s0_01.csv",
    )
    args = parser.parse_args()
    data = pd.read_csv(args.file)
    plot_token_type_percentages(data)
    plot_mean_file_type_distribution(data)
