import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_token_type_percentages(file_path):
    df = pd.read_csv(file_path)

    total_nl = df["nl_mean"].sum()
    total_code = df["token_mean"].sum()
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
        "Percentage of Natural Language vs Code Tokens", fontsize=16
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        help="CSV file with repository statistics",
        default="/Users/nicolashrubec/dev/agent-state-management/data/token-analysis/nl_code_SWE-bench_Verified_test_s0_01.csv",
    )
    args = parser.parse_args()
    plot_token_type_percentages(args.file)
