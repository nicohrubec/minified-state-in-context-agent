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
            "+ Obfuscate",
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
        (
            "+ Remove Docstrings",
            {
                "eval_name": "obscure-remove-comments-docstrings",
                "transformations": [
                    "short_vars_map",
                    "short_funcs_map",
                    "short_classes",
                    "remove_comments",
                    "remove_docstrings",
                ],
            },
        ),
        (
            "+ Line/Operator Whitespace Removal",
            {
                "eval_name": "obscure-remove-comments-docstrings-blank-lines-reduce-operators",
                "transformations": [
                    "short_vars_map",
                    "short_funcs_map",
                    "short_classes",
                    "remove_comments",
                    "remove_docstrings",
                    "remove_blank_lines",
                    "reduce_operators",
                ],
            },
        ),
        (
            "+ Remove Imports",
            {
                "eval_name": "obscure-remove-comments-docstrings-blank-lines-reduce-operators-remove-imports",
                "transformations": [
                    "short_vars_map",
                    "short_funcs_map",
                    "short_classes",
                    "remove_comments",
                    "remove_docstrings",
                    "remove_blank_lines",
                    "reduce_operators",
                    "remove_imports",
                ],
            },
        ),
        (
            "+ Dedent",
            {
                "eval_name": "obscure-remove-comments-docstrings-blank-lines-reduce-operators-remove-imports-dedent",
                "transformations": [
                    "short_vars_map",
                    "short_funcs_map",
                    "short_classes",
                    "remove_comments",
                    "remove_docstrings",
                    "remove_blank_lines",
                    "reduce_operators",
                    "remove_imports",
                    "dedent",
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


gpt_4_results_data = []
all_results_data = []

models = {
    "gpt-4.1": {
        "eval_pattern": "gpt-4.1.{combination_name}-n100.json",
        "metrics_pattern": "metrics_gpt-4.1_temp0.8_SWE-bench_Verified_test_s0_2_trie_{transformations}.csv",
    },
    "gpt-5-mini": {
        "eval_pattern": "gpt-5-mini.{combination_name}-n100.json",
        "metrics_pattern": "metrics_gpt-5-mini_SWE-bench_Verified_test_s0_2_trie_{transformations}.csv",
    },
}

for model_name, patterns in models.items():
    for combination_key, combination_data in TRANSFORMATION_COMBINATIONS.items():
        combination_name = combination_data["eval_name"]
        transformations = combination_data["transformations"]

        eval_filename = patterns["eval_pattern"].format(
            combination_name=combination_name
        )
        metrics_filename = patterns["metrics_pattern"].format(
            transformations="_".join(transformations)
        )

        eval_path = base_path_evals / eval_filename
        metrics_path = base_path_metrics / metrics_filename

        if not eval_path.exists() or not metrics_path.exists():
            continue

        try:
            with open(eval_path, "r") as f:
                eval_results = json.load(f)
            metrics_df = pd.read_csv(metrics_path)

            submitted_instances = eval_results["submitted_instances"]
            resolved_instances = eval_results["resolved_instances"]
            resolved_percentage = (resolved_instances / submitted_instances) * 100

            avg_input_tokens = metrics_df["num_repair_input_tokens"].mean()
            avg_output_tokens = metrics_df["num_repair_output_tokens"].mean()
            total_tokens = avg_input_tokens + avg_output_tokens
            std_total_tokens = (
                metrics_df["num_repair_input_tokens"]
                + metrics_df["num_repair_output_tokens"]
            ).std()

            result_data = {
                "combination_key": combination_key,
                "combination_name": combination_name,
                "transformations": transformations,
                "resolved_percentage": resolved_percentage,
                "resolved_count": resolved_instances,
                "submitted_count": submitted_instances,
                "avg_input_tokens": avg_input_tokens,
                "avg_output_tokens": avg_output_tokens,
                "total_tokens": total_tokens,
                "std_total_tokens": std_total_tokens,
                "model": model_name,
            }

            # Add cost calculations for gpt-4.1
            if model_name == "gpt-4.1":
                input_cost = avg_input_tokens * input_dollar_cost_per_token
                output_cost = avg_output_tokens * output_dollar_cost_per_token
                total_cost = input_cost + output_cost
                result_data.update(
                    {
                        "input_cost": input_cost,
                        "output_cost": output_cost,
                        "total_cost": total_cost,
                    }
                )
                gpt_4_results_data.append(result_data)

            all_results_data.append(result_data)

        except Exception as e:
            print(f"  Warning: Error processing {model_name} {combination_key}: {e}")
            continue

df_gpt_4_results = pd.DataFrame(gpt_4_results_data)

# cost performance line plot
plt.figure(figsize=(12, 8))

plt.plot(
    df_gpt_4_results["total_cost"],
    df_gpt_4_results["resolved_count"],
    marker="o",
    linewidth=2,
    markersize=8,
    alpha=0.8,
)

for i, row in df_gpt_4_results.iterrows():
    plt.annotate(
        row["combination_key"],
        (row["total_cost"], row["resolved_count"]),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=9,
        alpha=0.8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )

# comparing gpt-4.1 and gpt-5-mini
plt.xlabel("Per-instance Cost (USD)")
plt.ylabel("Number of Resolved Instances")
plt.title("Resolved Instances vs Cost for Stacked Transformations")
plt.gca().invert_xaxis()
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

df_all_results = pd.DataFrame(all_results_data)

# input tokens vs resolved instances comparison plot
plt.figure(figsize=(14, 8))

models_available = df_all_results["model"].unique()
colors = ["#1f77b4", "#ff7f0e"]
markers = ["o", "s"]

# Create legend handles for transformation combinations
legend_handles = []
legend_labels = []

# Get unique combinations in order
combinations_order = list(TRANSFORMATION_COMBINATIONS.keys())

for i, model in enumerate(models_available):
    model_data = df_all_results[df_all_results["model"] == model].copy()
    model_data = model_data.sort_values("avg_input_tokens")

    plt.plot(
        model_data["avg_input_tokens"],
        model_data["resolved_count"],
        marker=markers[i % len(markers)],
        linewidth=2,
        markersize=15,
        alpha=0.8,
        color=colors[i % len(colors)],
        label=model,
        markeredgecolor="white",
        markeredgewidth=1,
    )

    # Add numbered labels to points
    for _, row in model_data.iterrows():
        combination_number = combinations_order.index(row["combination_key"]) + 1
        plt.annotate(
            str(combination_number),
            (row["avg_input_tokens"], row["resolved_count"]),
            xytext=(0, 0),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center",
            color="white"
            if i == 0
            else "black",  # White text for blue markers, black for orange
        )

# Create legend for transformation combinations
for j, combination_key in enumerate(combinations_order):
    legend_handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="",
            color="gray",
            linestyle="None",
            markersize=0,
            alpha=0,
            label=f"{j + 1}. {combination_key}",
        )
    )

plt.xlabel("Average Input Tokens per Instance")
plt.ylabel("Number of Resolved Instances")
plt.title("Resolved Instances vs Input Tokens for Stacked Transformations")

# Create two legends: one for models, one for combinations
model_legend = plt.legend(loc="lower left", bbox_to_anchor=(0, 0))
combination_legend = plt.legend(
    handles=legend_handles, loc="center left", bbox_to_anchor=(1, 0.5)
)
plt.gca().add_artist(model_legend)  # Add model legend back

plt.gca().invert_xaxis()
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Summary Table:")
print("=" * 130)
print(
    f"{'Combination Name':<40} {'Resolved':<10} {'Input Cost ($)':<15} {'Output Cost ($)':<15} {'Total Cost ($)':<15} {'Transformations':<30}"
)
print("-" * 130)

for result in gpt_4_results_data:
    transformations_str = (
        ", ".join(result["transformations"]) if result["transformations"] else "None"
    )
    print(
        f"{result['combination_key']:<40} {result['resolved_count']:<10} {result['input_cost']:<15.4f} {result['output_cost']:<15.4f} {result['total_cost']:<15.4f} {transformations_str:<30}"
    )

print("-" * 130)

print("\n\nToken Usage Summary Table (sorted by total tokens):")
print("=" * 170)
print(
    f"{'Combination Name':<40} {'Model':<12} {'Resolved':<10} {'Input Tokens':<15} {'Output Tokens':<15} {'Total Tokens':<15} {'Total Tokens Std':<20}"
)
print("-" * 170)

df_sorted = df_all_results.sort_values(
    ["model", "total_tokens"], ascending=[True, False]
)

for _, row in df_sorted.iterrows():
    print(
        f"{row['combination_key']:<40} {row['model']:<12} {row['resolved_count']:<10} {row['avg_input_tokens']:<15.0f} {row['avg_output_tokens']:<15.0f} {row['total_tokens']:<15.0f} {row['std_total_tokens']:<20.0f}"
    )

print("-" * 170)
