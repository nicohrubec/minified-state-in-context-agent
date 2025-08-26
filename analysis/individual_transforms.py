import json
from pathlib import Path
import pandas as pd

from agent.minify import DEFINED_TRANSFORMATIONS

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
        / f"metrics_gpt-4.1_SWE-bench_Verified_test_s0_2_trie_{metric_transform}.csv"
    )
    token_usage_data[metric_transform] = pd.read_csv(metric_path)

# gpt 4.1
input_dollar_cost_per_token = 2.0 / 1e6
output_dollar_cost_per_token = 8.0 / 1e6

results_data = []

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

    results_data.append(
        {
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
    )

print("Individual Transformations Analysis")
print("=" * 80)
print(
    f"{'Transformation':<20} {'Resolved %':<12} {'Resolved':<10} {'Input Tokens':<15} {'Output Tokens':<15} {'Total Cost ($)':<15}"
)
print("-" * 80)

for result in results_data:
    print(
        f"{result['transformation']:<20} {result['resolved_percentage']:<12.2f} {result['resolved_count']:<10} {result['avg_input_tokens']:<15.0f} {result['avg_output_tokens']:<15.0f} {result['total_cost']:<15.4f}"
    )

print("-" * 80)
print()
