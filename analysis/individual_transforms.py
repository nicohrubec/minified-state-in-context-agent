import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from agent.minify import DEFINED_TRANSFORMATIONS

sns.set_theme(style="whitegrid", context="talk")


def transform_name_for_display(transformation_name):
    if transformation_name.endswith("-map-with-map"):
        return transformation_name.replace("-map-with-map", "-map")
    elif transformation_name.endswith("-map") and not transformation_name.endswith(
        "-map-with-map"
    ):
        return transformation_name.replace("-map", "")
    else:
        return transformation_name


base_path_evals = Path(
    "/Users/nicolashrubec/dev/agent-state-management/data/eval_results"
)
base_path_metrics = Path(
    "/Users/nicolashrubec/dev/agent-state-management/data/agent_results/metrics"
)
eval_transform_names = ["no-compression"] + [
    t.replace("_", "-") for t in DEFINED_TRANSFORMATIONS
]
metrics_transform_names = [""] + DEFINED_TRANSFORMATIONS

performance_data = {}
token_usage_data = {}
for eval_transform, metric_transform in zip(
    eval_transform_names, metrics_transform_names
):
    eval_path = base_path_evals / f"gpt-4.1.{eval_transform}-n100.json"
    with open(eval_path, "r") as f:
        results = json.load(f)
    performance_data[eval_transform] = results

    metric_path = (
        base_path_metrics
        / f"metrics_gpt-4.1_temp0.8_SWE-bench_Verified_test_s0_2_trie_{metric_transform}.csv"
    )
    token_usage_data[metric_transform] = pd.read_csv(metric_path)

# gpt 4.1
input_dollar_cost_per_token = 2.0 / 1e6
output_dollar_cost_per_token = 8.0 / 1e6

excluded_transformations = ["short-vars-map", "short-funcs-map", "short-classes-map"]

results_data = []
individual_results_data = []
renaming_ablation_data = []

for eval_transform, metric_transform in zip(
    eval_transform_names, metrics_transform_names
):
    performance = performance_data[eval_transform]
    submitted_instances = performance["submitted_instances"]
    resolved_instances = performance["resolved_instances"]
    resolved_percentage = (resolved_instances / submitted_instances) * 100

    tokens_df = token_usage_data[metric_transform]
    avg_input_tokens = tokens_df["num_repair_input_tokens"].mean()
    avg_output_tokens = tokens_df["num_repair_output_tokens"].mean()

    input_cost = avg_input_tokens * input_dollar_cost_per_token
    output_cost = avg_output_tokens * output_dollar_cost_per_token
    total_cost = input_cost + output_cost

    result_dict = {
        "transformation": eval_transform,
        "resolved_percentage": resolved_percentage,
        "resolved_count": resolved_instances,
        "submitted_count": submitted_instances,
        "avg_input_tokens": avg_input_tokens,
        "avg_output_tokens": avg_output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }

    results_data.append(result_dict)

    # Separate data for different plots
    if eval_transform not in excluded_transformations:
        individual_results_data.append(result_dict)

    # For renaming ablation: include transformations with and without map
    if any(
        transform in eval_transform
        for transform in ["short-vars", "short-funcs", "short-classes"]
    ):
        renaming_ablation_data.append(result_dict)

print("Individual Transformations Analysis")
print("=" * 80)
print(
    f"{'Transformation':<35} {'Resolved %':<12} {'Resolved':<10} {'Input Tokens':<15} {'Output Tokens':<15} {'Per-Instance Cost ($)':<15}"
)
print("-" * 95)

for result in results_data:
    print(
        f"{transform_name_for_display(result['transformation']):<35} {result['resolved_percentage']:<12.2f} {result['resolved_count']:<10} {result['avg_input_tokens']:<15.0f} {result['avg_output_tokens']:<15.0f} {result['total_cost']:<15.4f}"
    )

print("-" * 95)
print()

df_results = pd.DataFrame(individual_results_data)

# performance vs cost scatter plot
plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    df_results["total_cost"],
    df_results["resolved_percentage"],
    s=100,
    alpha=0.7,
    c=range(len(df_results)),
    cmap="viridis",
)

# label points
for i, row in df_results.iterrows():
    has_close_point = False
    close_partner_index = None
    for j, other_row in df_results.iterrows():
        if i != j:
            cost_diff = abs(row["total_cost"] - other_row["total_cost"])
            perf_diff = abs(
                row["resolved_percentage"] - other_row["resolved_percentage"]
            )
            if cost_diff < 0.0025 and perf_diff < 2.0:
                has_close_point = True
                close_partner_index = j
                break

    x_offset = 5
    y_offset = 5
    # if there is a close point select one to push aside
    if has_close_point and i < close_partner_index:
        y_offset = -10
    elif has_close_point and i > close_partner_index:
        x_offset = -10

    plt.annotate(
        transform_name_for_display(row["transformation"]),
        (row["total_cost"], row["resolved_percentage"]),
        xytext=(x_offset, y_offset),
        textcoords="offset points",
        fontsize=9,
        alpha=0.8,
    )

plt.xlabel("Per-Instance Cost (USD)")
plt.ylabel("Resolved Percentage (%)")
plt.title("Performance vs Cost by Transformation")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# resolved percentage bar chart
plt.figure(figsize=(14, 8))
df_results = df_results.sort_values("resolved_percentage", ascending=False)
bars = plt.bar(
    df_results["transformation"].apply(transform_name_for_display),
    df_results["resolved_percentage"],
    color="skyblue",
    edgecolor="navy",
    alpha=0.8,
)

# add value labels on bars
for bar, percentage in zip(bars, df_results["resolved_percentage"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{percentage:.1f}%",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

plt.xlabel("Transformation")
plt.ylabel("Resolved Percentage (%)")
plt.title("Resolved Percentage by Transformation")
plt.xticks(rotation=45, ha="right")
plt.ylim(0, max(df_results["resolved_percentage"]) * 1.15)
plt.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.show()

# stacked bar chart for cost breakdown
plt.figure(figsize=(14, 8))
df_results = df_results.sort_values("total_cost", ascending=False)

# create stacked bars
x_pos = range(len(df_results))
input_costs = df_results["input_cost"]
output_costs = df_results["output_cost"]

bars1 = plt.bar(x_pos, input_costs, label="Input Cost", color="lightcoral", alpha=0.8)
bars2 = plt.bar(
    x_pos,
    output_costs,
    bottom=input_costs,
    label="Output Cost",
    color="lightblue",
    alpha=0.8,
)

# add value labels on bars
for i, (input_cost, output_cost, total_cost) in enumerate(
    zip(input_costs, output_costs, df_results["total_cost"])
):
    plt.text(
        i,
        total_cost + 0.0001,
        f"${total_cost:.4f}",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=9,
    )

plt.xlabel("Transformation")
plt.ylabel("Per-Instance Cost (USD)")
plt.title("Input vs Output Costs by Transformation")
plt.xticks(
    x_pos,
    df_results["transformation"].apply(transform_name_for_display),
    rotation=45,
    ha="right",
)
plt.legend()
plt.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.show()

# detailed performance breakdown for each transformation
plt.figure(figsize=(16, 10))

breakdown_data = []
for eval_transform in eval_transform_names:
    if eval_transform in excluded_transformations:
        continue
    performance = performance_data[eval_transform]
    submitted_instances = performance["submitted_instances"]

    resolved = performance["resolved_instances"]
    unresolved = performance["unresolved_instances"]
    errors = performance["error_instances"]
    empty_patches = performance["empty_patch_instances"]

    resolved_percentage = (resolved / submitted_instances) * 100
    unresolved_percentage = (unresolved / submitted_instances) * 100
    error_percentage = (errors / submitted_instances) * 100
    empty_patch_percentage = (empty_patches / submitted_instances) * 100

    breakdown_data.append(
        {
            "transformation": eval_transform,
            "resolved_percentage": resolved_percentage,
            "unresolved_percentage": unresolved_percentage,
            "error_percentage": error_percentage,
            "empty_patch_percentage": empty_patch_percentage,
        }
    )

breakdown_data.sort(key=lambda x: x["resolved_percentage"], reverse=True)

# extract sorted data
transformations = [item["transformation"] for item in breakdown_data]
resolved_percentages = [item["resolved_percentage"] for item in breakdown_data]
unresolved_percentages = [item["unresolved_percentage"] for item in breakdown_data]
error_percentages = [item["error_percentage"] for item in breakdown_data]
empty_patch_percentages = [item["empty_patch_percentage"] for item in breakdown_data]

# create stacked bar chart
x_pos = range(len(transformations))

bars1 = plt.bar(
    x_pos, resolved_percentages, label="Resolved", color="#2ecc71", alpha=0.8
)
bars2 = plt.bar(
    x_pos,
    unresolved_percentages,
    bottom=resolved_percentages,
    label="Unresolved",
    color="#e74c3c",
    alpha=0.8,
)
bars3 = plt.bar(
    x_pos,
    error_percentages,
    bottom=[r + u for r, u in zip(resolved_percentages, unresolved_percentages)],
    label="Errors",
    color="#f39c12",
    alpha=0.8,
)
bars4 = plt.bar(
    x_pos,
    empty_patch_percentages,
    bottom=[
        r + u + e
        for r, u, e in zip(
            resolved_percentages, unresolved_percentages, error_percentages
        )
    ],
    label="Empty Patches",
    color="#9b59b6",
    alpha=0.8,
)

plt.xlabel("Transformation")
plt.ylabel("Percentage of Instances (%)")
plt.title("Breakdown of Instance Outcomes by Transformation")
plt.xticks(
    x_pos,
    [transform_name_for_display(t) for t in transformations],
    rotation=45,
    ha="right",
)
plt.legend(title="Outcome", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.ylim(0, 100)
plt.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.show()

# Renaming ablation plot
plt.figure(figsize=(12, 8))
df_renaming = pd.DataFrame(renaming_ablation_data)

with_map = df_renaming[df_renaming["transformation"].str.contains("map-with-map")]
without_map = df_renaming[
    ~df_renaming["transformation"].str.contains("map-with-map")
    & df_renaming["transformation"].str.contains("map")
]

plt.scatter(
    with_map["total_cost"],
    with_map["resolved_count"],
    label="With in-context map",
    s=100,
    alpha=0.7,
    color="blue",
)
plt.scatter(
    without_map["total_cost"],
    without_map["resolved_count"],
    label="Without in-context map",
    s=100,
    alpha=0.7,
    color="red",
)

# add labels
for _, row in df_renaming.iterrows():
    plt.annotate(
        transform_name_for_display(row["transformation"]),
        (row["total_cost"], row["resolved_count"]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9,
        alpha=0.8,
    )

plt.xlabel("Per-Instance Cost (USD)")
plt.ylabel("Number of Resolved Instances")
plt.title("Renaming Identifiers with vs without in-context map")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nDetailed Performance Breakdown:")
print(
    f"{'Transformation':<35} {'Resolved':<10} {'Unresolved':<12} {'Errors':<8} {'Empty Patches':<15}"
)
print("-" * 85)
for item in breakdown_data:
    eval_transform = item["transformation"]
    performance = performance_data[eval_transform]
    submitted_instances = performance["submitted_instances"]

    resolved = performance["resolved_instances"]
    unresolved = performance["unresolved_instances"]
    errors = performance["error_instances"]
    empty_patches = performance["empty_patch_instances"]

    print(
        f"{transform_name_for_display(eval_transform):<35} {resolved:<10} {unresolved:<12} {errors:<8} {empty_patches:<15}"
    )
