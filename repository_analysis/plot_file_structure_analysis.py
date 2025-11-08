import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid", context="talk")


def get_token_columns(df):
    return [
        col
        for col in df.columns
        if col not in ("problem", "repository", "total_chars") and df[col].dtype != "O"
    ]


def drop_zero_columns(df, columns):
    return [col for col in columns if df[col].sum() > 0]


def plot_bar(data, title, ylabel, save_path):
    data = data.sort_values(by="Count", ascending=False)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=data, x="Category", y="Count", hue="Category", palette="muted"
    )
    ax.set_title(title, fontsize=16)
    ax.set_ylabel(ylabel)

    # Rotate x-axis labels for better spacing
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    for p, val in zip(ax.patches, data["Count"]):
        ax.annotate(
            f"{val:,.2f}",
            (p.get_x() + p.get_width() / 2.0, val),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(save_path, format="svg")
    plt.show()


def plot_distribution(df, category_cols, title, ylabel, save_path):
    totals = df[category_cols].sum()
    data = pd.DataFrame({"Category": totals.index, "Count": totals.values})
    plot_bar(data, title, ylabel, save_path)


def plot_top5_mean_distribution(df, category_cols, total_col, title, ylabel, save_path):
    top5 = df.nlargest(5, total_col)
    mean_vals = top5[category_cols].mean()
    data = pd.DataFrame({"Category": mean_vals.index, "Count": mean_vals.values})
    plot_bar(data, title, ylabel, save_path)


def main(lexical_path, structural_path):
    df_lex = pd.read_csv(lexical_path)
    df_struct = pd.read_csv(structural_path)

    # Lexical
    lex_cols = get_token_columns(df_lex)
    lex_cols = drop_zero_columns(df_lex, lex_cols)
    plot_distribution(
        df_lex,
        lex_cols,
        "Lexical Token Distribution",
        "Character Count",
        "../plots/lexical.svg",
    )
    plot_top5_mean_distribution(
        df_lex,
        lex_cols,
        total_col="total_chars",
        title="Lexical Token Distribution (Large Instances)",
        ylabel="Character Count",
        save_path="../plots/lexical_top5.svg",
    )

    # Structural
    struct_cols = get_token_columns(df_struct)
    struct_cols = drop_zero_columns(df_struct, struct_cols)
    plot_distribution(
        df_struct,
        struct_cols,
        "Structural Token Distribution",
        "Character Count",
        "../plots/structural.svg",
    )
    plot_top5_mean_distribution(
        df_struct,
        struct_cols,
        total_col="total_chars",
        title="Structural Token Distribution (Large Instances)",
        ylabel="Character Count",
        save_path="../plots/structural_top5.svg",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lexical",
        default="/Users/nicolashrubec/dev/minified-state-in-context-agent/data/token_analysis/lexical_SWE-bench_Verified_test.csv",
        help="Path to lexical token CSV",
    )
    parser.add_argument(
        "--structural",
        default="/Users/nicolashrubec/dev/minified-state-in-context-agent/data/token_analysis/structural_SWE-bench_Verified_test.csv",
        help="Path to structural token CSV",
    )
    args = parser.parse_args()
    main(args.lexical, args.structural)
