import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

sns.set_theme(style="whitegrid", context="talk")

base_path_evals = Path("/Users/nicolashrubec/dev/agent-state-management/data/eval_results")
base_path_metrics = Path("/Users/nicolashrubec/dev/agent-state-management/data/agent_results/metrics")

eval_files = {
    "no-compression": "gpt-5-mini.no-compression-n500.json",
    "compression": "gpt-5-mini.compression-n500.json"
}

metrics_files = {
    "no-compression": "metrics_gpt-5-mini_SWE-bench_Verified_test_trie_.csv",
    "compression": "metrics_gpt-5-mini_SWE-bench_Verified_test_trie_short_vars_map_short_funcs_map_short_classes_remove_comments_remove_docstrings_remove_blank_lines_reduce_operators_remove_imports.csv"
}

# get difficulty and repository metadata
swe_bench_data = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
difficulty_map = {row["instance_id"]: row["difficulty"] for row in swe_bench_data}
repo_map = {row["instance_id"]: row["repo"] for row in swe_bench_data}

# load all data into a single dataframe
all_data = []
resolved_ids_by_run = {}

for run_type, eval_filename in eval_files.items():
    eval_path = base_path_evals / eval_filename
    metrics_path = base_path_metrics / metrics_files[run_type]
    
    with open(eval_path, "r") as f:
        eval_results = json.load(f)
    
    metrics_df = pd.read_csv(metrics_path)
    resolved_ids_by_run[run_type] = set(eval_results["resolved_ids"])
    
    # add metadata
    metrics_df["run_type"] = run_type
    metrics_df["difficulty"] = metrics_df["instance_id"].map(difficulty_map)
    metrics_df["repo"] = metrics_df["instance_id"].map(repo_map)
    metrics_df["resolved"] = metrics_df["instance_id"].isin(resolved_ids_by_run[run_type])
    
    all_data.append(metrics_df)

df_all = pd.concat(all_data, ignore_index=True)

# calculate overall results
df_results = df_all.groupby("run_type").agg({
    "resolved": ["sum", "count", "mean"],
    "num_repair_input_tokens": ["mean", "std"]
}).round(2)

df_results.columns = ["resolved_instances", "total_instances", "resolved_percentage", "avg_input_tokens", "std_input_tokens"]
df_results["resolved_percentage"] *= 100
df_results = df_results.reset_index()

# calculate results by difficulty
df_difficulty = df_all.groupby(["run_type", "difficulty"]).agg({
    "resolved": ["sum", "count", "mean"],
    "num_repair_input_tokens": ["mean", "std"]
}).round(2)

df_difficulty.columns = ["resolved_instances", "total_instances", "resolved_percentage", "avg_input_tokens", "std_input_tokens"]
df_difficulty["resolved_percentage"] *= 100
df_difficulty = df_difficulty.reset_index()

# calculate results by repository
repo_counts = df_all.groupby("repo").size()
repos_with_min_instances = repo_counts[repo_counts >= 10].index.tolist() # filter for repos with at least 10 instances

df_repo = df_all[df_all["repo"].isin(repos_with_min_instances)].groupby(["run_type", "repo"]).agg({
    "resolved": ["sum", "count", "mean"],
    "num_repair_input_tokens": ["mean", "std"]
}).round(2)

df_repo.columns = ["resolved_instances", "total_instances", "resolved_percentage", "avg_input_tokens", "std_input_tokens"]
df_repo["resolved_percentage"] *= 100
df_repo = df_repo.reset_index()

# plot overall results
plt.figure(figsize=(10, 8))

colors = ["#1f77b4", "#ff7f0e"]
markers = ["o", "s"]

for i, (_, row) in enumerate(df_results.iterrows()):
    plt.errorbar(
        row["resolved_percentage"],
        row["avg_input_tokens"],
        yerr=row["std_input_tokens"],
        marker=markers[i],
        markersize=15,
        linewidth=2,
        color=colors[i],
        label=row["run_type"].replace("-", " ").title(),
        markeredgecolor="white",
        markeredgewidth=1,
        capsize=5,
        capthick=2
    )

plt.xlabel("Resolved Instances (%)")
plt.ylabel("Average Input Tokens per Instance")
plt.title("Performance vs Input Tokens")
plt.xlim(-1, 100)
plt.ylim(0, None)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# plot split by difficulty
plt.figure(figsize=(14, 8))

run_type_colors = {"no-compression": "#1f77b4", "compression": "#ff7f0e"}
run_type_markers = {"no-compression": "o", "compression": "s"}
difficulty_colors = {
    "<15 min fix": "#2ca02c", 
    "15 min - 1 hour": "#ff7f0e", 
    "1-4 hours": "#ff6b35", 
    ">4 hours": "#d62728"
}

combined_colors = {}
for run_type in ["no-compression", "compression"]:
    for difficulty in ["<15 min fix", "15 min - 1 hour", "1-4 hours", ">4 hours"]:
        base_color = run_type_colors[run_type]
        diff_color = difficulty_colors[difficulty]
        combined_colors[f"{run_type}_{difficulty}"] = diff_color

for run_type in ["no-compression", "compression"]:
    run_data = df_difficulty[df_difficulty["run_type"] == run_type]
    
    for _, row in run_data.iterrows():
        plt.errorbar(
            row["resolved_percentage"],
            row["avg_input_tokens"],
            yerr=row["std_input_tokens"],
            marker=run_type_markers[run_type],
            markersize=12,
            linewidth=2,
            color=combined_colors[f"{run_type}_{row['difficulty']}"],
            label=f"{run_type.replace('-', ' ').title()} - {row['difficulty']}",
            markeredgecolor="white",
            markeredgewidth=1,
            capsize=3,
            capthick=1.5,
            alpha=0.8
        )

plt.xlabel("Resolved Instances (%)")
plt.ylabel("Average Input Tokens per Instance")
plt.title("Performance vs Input Tokens by Difficulty Level")
plt.xlim(-1, 100)
plt.ylim(0, None)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# plot split by repository
plt.figure(figsize=(16, 8))

run_type_colors = {"no-compression": "#1f77b4", "compression": "#ff7f0e"}
run_type_markers = {"no-compression": "o", "compression": "s"}

unique_repos = sorted(df_repo["repo"].unique())
repo_colors = plt.cm.Set3(range(len(unique_repos)))

for run_type in ["no-compression", "compression"]:
    run_data = df_repo[df_repo["run_type"] == run_type]
    
    for _, row in run_data.iterrows():
        repo_idx = unique_repos.index(row["repo"])
        plt.errorbar(
            row["resolved_percentage"],
            row["avg_input_tokens"],
            yerr=row["std_input_tokens"],
            marker=run_type_markers[run_type],
            markersize=10,
            linewidth=2,
            color=repo_colors[repo_idx],
            label=f"{run_type.replace('-', ' ').title()} - {row['repo']}",
            markeredgecolor="white",
            markeredgewidth=1,
            capsize=3,
            capthick=1.5,
            alpha=0.8
        )

plt.xlabel("Resolved Instances (%)")
plt.ylabel("Average Input Tokens per Instance")
plt.title("Performance vs Input Tokens by Repository")
plt.xlim(-1, 100)
plt.ylim(0, None)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Summary Table:")
print("=" * 80)
print(f"{'Run Type':<15} {'Resolved':<10} {'Resolved %':<12} {'Avg Input Tokens':<18} {'Std Input Tokens':<18}")
print("-" * 80)

for _, row in df_results.iterrows():
    print(f"{row['run_type']:<15} {row['resolved_instances']:<10} {row['resolved_percentage']:<12.1f} {row['avg_input_tokens']:<18.0f} {row['std_input_tokens']:<18.0f}")

print("-" * 80)

print("\nDifficulty Analysis:")
print("=" * 120)
print(f"{'Run Type':<15} {'Difficulty':<15} {'Resolved %':<12} {'Avg Input Tokens':<18} {'Std Input Tokens':<18}")
print("-" * 120)

for _, row in df_difficulty.iterrows():
    print(f"{row['run_type']:<15} {row['difficulty']:<15} {row['resolved_percentage']:<12.1f} {row['avg_input_tokens']:<18.0f} {row['std_input_tokens']:<18.0f}")

print("-" * 120)

print("\nRepository Analysis:")
print("=" * 130)
print(f"{'Run Type':<15} {'Repository':<25} {'Resolved %':<12} {'Avg Input Tokens':<18} {'Std Input Tokens':<18}")
print("-" * 130)

for _, row in df_repo.iterrows():
    print(f"{row['run_type']:<15} {row['repo']:<25} {row['resolved_percentage']:<12.1f} {row['avg_input_tokens']:<18.0f} {row['std_input_tokens']:<18.0f}")

print("-" * 130)
