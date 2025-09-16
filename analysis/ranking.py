import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import ast

sns.set_theme(style="whitegrid", context="talk")

results_path = Path(
    "/Users/nicolashrubec/dev/agent-state-management/data/agent_results"
)
list_results_path = (
    results_path
    / "metrics"
    / "ranking_metrics_gpt-4.1_SWE-bench_Verified_test_list_.csv"
)
trie_results_path = (
    results_path
    / "metrics"
    / "ranking_metrics_gpt-4.1_SWE-bench_Verified_test_trie_.csv"
)
int_folder_results_path = (
    results_path
    / "metrics"
    / "ranking_metrics_gpt-4.1_SWE-bench_Verified_test_int_folder_.csv"
)
int_path_results_path = (
    results_path
    / "metrics"
    / "ranking_metrics_gpt-4.1_SWE-bench_Verified_test_int_path_.csv"
)

list_results = pd.read_csv(list_results_path)
trie_results = pd.read_csv(trie_results_path)
int_folder_results = pd.read_csv(int_folder_results_path)
int_path_results = pd.read_csv(int_path_results_path)

# gpt 4.1
input_dollar_cost_per_token = 2.0 / 1e6
output_dollar_cost_per_token = 8.0 / 1e6

results = {
    "list": list_results,
    "trie": trie_results,
    "int_folder": int_folder_results,
    "int_path": int_path_results,
}
results_data = []
cost_rows = []

for encoding, df in results.items():
    df["encoding"] = encoding
    avg_recall = (df["recall"] * 100).mean()
    avg_input_tokens = df["num_ranking_input_tokens"].mean()
    avg_output_tokens = df["num_ranking_output_tokens"].mean()
    df["input_cost"] = df["num_ranking_input_tokens"] * input_dollar_cost_per_token
    df["output_cost"] = df["num_ranking_output_tokens"] * output_dollar_cost_per_token
    df["cost"] = df["input_cost"] + df["output_cost"]
    avg_cost = df["cost"].mean()

    print(f"Avg recall for {encoding}: {avg_recall:.2f} %")
    print(f"Avg number of input tokens for {encoding}: {avg_input_tokens:.0f}")
    print(f"Avg number of output tokens for {encoding}: {avg_output_tokens:.0f}")
    print(f"Avg cost for {encoding}: {avg_cost:.4f} $")

    print()

    cost_rows.append(
        {
            "encoding": encoding,
            "metric": "Input cost",
            "value": df["input_cost"].mean(),
        }
    )
    cost_rows.append(
        {
            "encoding": encoding,
            "metric": "Output cost",
            "value": df["output_cost"].mean(),
        }
    )
    cost_rows.append(
        {
            "encoding": encoding,
            "metric": "Overall cost",
            "value": df["cost"].mean(),
        }
    )
    results_data.append(df)

cost_summary = pd.DataFrame(cost_rows)
all_results = pd.concat(results_data, ignore_index=True)

# Cost bar plot
encoding_order = (
    cost_summary.loc[cost_summary["metric"] == "Overall cost"]
    .sort_values("value", ascending=False)["encoding"]
    .tolist()
)  # order encodings by overall cost

plt.figure(figsize=(12, 7))
ax1 = sns.barplot(
    data=cost_summary,
    x="metric",
    y="value",
    hue="encoding",
    hue_order=encoding_order,
    order=["Overall cost", "Input cost", "Output cost"],
    edgecolor="w",
)
ax1.set_xlabel("")
ax1.set_ylabel("Average cost (USD)")
ax1.set_title("Average cost by encoding and cost type")
ax1.legend(title="Encoding", ncols=2, frameon=True)
sns.despine()
plt.tight_layout()
plt.show()

# Average cost vs recall scatter plot
avg_metrics = all_results.groupby("encoding", as_index=False).agg(
    avg_cost=("cost", "mean"), avg_recall=("recall", "mean")
)

plt.figure(figsize=(12, 7))
ax2 = sns.scatterplot(
    data=avg_metrics,
    x="avg_cost",
    y="avg_recall",
    hue="encoding",
    style="encoding",
    s=200,
    alpha=0.9,
    edgecolor="w",
)
ax2.set_xlabel("Average cost (USD)")
ax2.set_ylabel("Average recall")
ax2.set_title("Average cost vs average recall by encoding")
ax2.legend(title="Encoding", ncols=2, frameon=True)
sns.despine()
plt.tight_layout()
plt.show()


# Target file position density
def _parse_positions(cell):
    if pd.isna(cell):
        return []
    if isinstance(cell, list):
        return [int(x) for x in cell]

    s = str(cell).strip()
    val = ast.literal_eval(s)
    return [int(x) for x in val]


density_rows = []
for encoding, df in results.items():
    df_local = df.copy()
    positions_lists = df_local["target_file_positions"].apply(_parse_positions)
    exploded = positions_lists.explode().dropna()
    if not exploded.empty:
        density_rows.append(
            pd.DataFrame(
                {
                    "encoding": encoding,
                    "rank": exploded.astype(int).values,
                }
            )
        )

density_df = pd.concat(density_rows, ignore_index=True)

plt.figure(figsize=(12, 7))
for enc, sub in density_df.groupby("encoding"):
    sns.kdeplot(
        data=sub,
        x="rank",
        fill=True,
        common_norm=False,
        alpha=0.3,
        linewidth=2,
        label=enc,
    )

plt.xlabel("Target file rank position")
plt.ylabel("Density")
plt.title("Density of target file positions in ranking")
plt.legend(title="Encoding", ncols=2, frameon=True)
sns.despine()
plt.tight_layout()
plt.show()
