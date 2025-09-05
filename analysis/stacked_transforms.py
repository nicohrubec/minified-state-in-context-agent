import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict

sns.set_theme(style="whitegrid", context="talk")

TRANSFORMATION_COMBINATIONS = OrderedDict(
    [
        ("No Compression", {"eval_name": "no-compression", "transformations": []}),
        (
            "+ Obscure",
            {
                "eval_name": "obscure",
                "transformations": [
                    "short_vars_map",
                    "short_funcs_map",
                    "short_classes",
                ],
            },
        ),
        (
            "+ Remove Comments",
            {
                "eval_name": "obscure-remove-comments",
                "transformations": [
                    "short_vars_map",
                    "short_funcs_map",
                    "short_classes",
                    "remove_comments",
                ],
            },
        ),
    ]
)

base_path_evals = Path(
    "/Users/nicolashrubec/dev/agent-state-management/data/eval_results"
)
base_path_metrics = Path(
    "/Users/nicolashrubec/dev/agent-state-management/data/agent_results/metrics"
)

# gpt 4.1
input_dollar_cost_per_token = 2.0 / 1e6
output_dollar_cost_per_token = 8.0 / 1e6


results_data = []

for combination_key, combination_data in TRANSFORMATION_COMBINATIONS.items():
    combination_name = combination_data["eval_name"]
    transformations = combination_data["transformations"]

    eval_filename = f"gpt-4.1.{combination_name}-n100.json"
    metrics_filename = f"metrics_gpt-4.1_temp0.8_SWE-bench_Verified_test_s0_2_trie_{'_'.join(transformations)}.csv"

    eval_path = base_path_evals / eval_filename
    metrics_path = base_path_metrics / metrics_filename

    # check if files exist
    if not eval_path.exists():
        print(f"  Warning: Eval file not found: {eval_path}")
        continue
    if not metrics_path.exists():
        print(f"  Warning: Metrics file not found: {metrics_path}")
        continue

    with open(eval_path, "r") as f:
        eval_results = json.load(f)
    metrics_df = pd.read_csv(metrics_path)

    submitted_instances = eval_results["submitted_instances"]
    resolved_instances = eval_results["resolved_instances"]
    resolved_percentage = (resolved_instances / submitted_instances) * 100

    # calculate costs
    avg_input_tokens = metrics_df["num_repair_input_tokens"].mean()
    avg_output_tokens = metrics_df["num_repair_output_tokens"].mean()
    input_cost = avg_input_tokens * input_dollar_cost_per_token
    output_cost = avg_output_tokens * output_dollar_cost_per_token
    total_cost = input_cost + output_cost

    results_data.append(
        {
            "combination_key": combination_key,
            "combination_name": combination_name,
            "transformations": transformations,
            "resolved_percentage": resolved_percentage,
            "resolved_count": resolved_instances,
            "submitted_count": submitted_instances,
            "avg_input_tokens": avg_input_tokens,
            "avg_output_tokens": avg_output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
        }
    )

df_results = pd.DataFrame(results_data)

# cost performance line plot
plt.figure(figsize=(12, 8))

plt.plot(
    df_results["total_cost"],
    df_results["resolved_count"],
    marker="o",
    linewidth=2,
    markersize=8,
    alpha=0.8,
)

for i, row in df_results.iterrows():
    plt.annotate(
        row["combination_key"],
        (row["total_cost"], row["resolved_count"]),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=9,
        alpha=0.8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )

plt.xlabel("Total Cost (USD)")
plt.ylabel("Number of Resolved Instances")
plt.title("Resolved Instances vs Cost for Stacked Transformations")
plt.gca().invert_xaxis()
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x)}"))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# summary table
print("Summary Table:")
print("=" * 120)
print(
    f"{'Combination Name':<30} {'Resolved':<10} {'Input Cost ($)':<15} {'Output Cost ($)':<15} {'Total Cost ($)':<15} {'Transformations':<30}"
)
print("-" * 120)

for result in results_data:
    transformations_str = (
        ", ".join(result["transformations"]) if result["transformations"] else "None"
    )
    print(
        f"{result['combination_key']:<30} {result['resolved_count']:<10} {result['input_cost']:<15.4f} {result['output_cost']:<15.4f} {result['total_cost']:<15.4f} {transformations_str:<30}"
    )

print("-" * 120)
