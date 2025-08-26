import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid", context="talk")

base_path = Path("/Users/nicolashrubec/dev/agent-state-management/data/eval_results")
temp08_path = base_path / "gpt-4.1.no-compression-n100.json"
temp02_path = base_path / "gpt-4.1.no-compression-temp02-n100.json"
temp00_path = base_path / "gpt-4.1.no-compression-temp00-n100.json"

with open(temp08_path, "r") as f:
    temp08_results = json.load(f)
with open(temp02_path, "r") as f:
    temp02_results = json.load(f)
with open(temp00_path, "r") as f:
    temp00_results = json.load(f)

submitted_instances = temp08_results["submitted_instances"]

# number of resolved instances
temp08_resolved = temp08_results["resolved_instances"]
temp02_resolved = temp02_results["resolved_instances"]
temp00_resolved = temp00_results["resolved_instances"]

# number of unresolved instances
temp08_unresolved = temp08_results["unresolved_instances"]
temp02_unresolved = temp02_results["unresolved_instances"]
temp00_unresolved = temp00_results["unresolved_instances"]

# number of error instances
temp08_errors = temp08_results["error_instances"]
temp02_errors = temp02_results["error_instances"]
temp00_errors = temp00_results["error_instances"]

# number of empty patches
temp08_empty = temp08_results["empty_patch_instances"]
temp02_empty = temp02_results["empty_patch_instances"]
temp00_empty = temp00_results["empty_patch_instances"]

# percentage of resolved instances
temp08_percentage = (temp08_resolved / submitted_instances) * 100
temp02_percentage = (temp02_resolved / submitted_instances) * 100
temp00_percentage = (temp00_resolved / submitted_instances) * 100


print("Temperature Ablation Results:")
print(f"Temperature 0.8: {temp08_resolved}/{submitted_instances} = {temp08_percentage:.2f}% resolved")
print(f"Temperature 0.2: {temp02_resolved}/{submitted_instances} = {temp02_percentage:.2f}% resolved")
print(f"Temperature 0.0: {temp00_resolved}/{submitted_instances} = {temp00_percentage:.2f}% resolved")

# data for resolved instances
temperature_data = {
    "Temperature": ["0.0", "0.2", "0.8"],
    "Resolved_Percentage": [temp00_percentage, temp02_percentage, temp08_percentage],
    "Resolved_Count": [temp00_resolved, temp02_resolved, temp08_resolved],
    "Submitted_Count": [submitted_instances, submitted_instances, submitted_instances]
}

df = pd.DataFrame(temperature_data)

# basic bar plot for resolved instances
plt.figure(figsize=(10, 6))
ax = sns.barplot(
    data=df,
    x="Temperature",
    y="Resolved_Percentage",
    edgecolor="w",
    color="skyblue"
)

for i, (percentage, count, submitted) in enumerate(zip(df["Resolved_Percentage"], df["Resolved_Count"], df["Submitted_Count"])):
    ax.text(i, percentage + 0.5, f"{percentage:.1f}%", 
            ha='center', va='bottom', fontweight='bold')

ax.set_xlabel("Temperature")
ax.set_ylabel("Percentage of Resolved Instances (%)")
ax.set_title("Temperature Ablation: Percentage of Resolved Instances")
ax.set_ylim(0, max(df["Resolved_Percentage"]) * 1.15)

sns.despine()
plt.tight_layout()
plt.show()

# more detailed breakdown of all categories
plt.figure(figsize=(12, 8))

stacked_data = {
    "Temperature": ["0.0", "0.2", "0.8"] * 4,
    "Category": ["Resolved"] * 3 + ["Unresolved"] * 3 + ["Errors"] * 3 + ["Empty Patches"] * 3,
    "Percentage": [
        (temp00_resolved / submitted_instances) * 100,
        (temp02_resolved / submitted_instances) * 100,
        (temp08_resolved / submitted_instances) * 100,
        (temp00_unresolved / submitted_instances) * 100,
        (temp02_unresolved / submitted_instances) * 100,
        (temp08_unresolved / submitted_instances) * 100,
        (temp00_errors / submitted_instances) * 100,
        (temp02_errors / submitted_instances) * 100,
        (temp08_errors / submitted_instances) * 100,
        (temp00_empty / submitted_instances) * 100,
        (temp02_empty / submitted_instances) * 100,
        (temp08_empty / submitted_instances) * 100
    ],
    "Count": [
        temp00_resolved, temp02_resolved, temp08_resolved,
        temp00_unresolved, temp02_unresolved, temp08_unresolved,
        temp00_errors, temp02_errors, temp08_errors,
        temp00_empty, temp02_empty, temp08_empty
    ]
}

stacked_df = pd.DataFrame(stacked_data)

ax2 = sns.barplot(
    data=stacked_df,
    x="Temperature",
    y="Percentage",
    hue="Category",
    palette=["#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"],  # Green, Red, Orange, Purple
    edgecolor="w"
)

ax2.set_xlabel("Temperature")
ax2.set_ylabel("Percentage of Instances (%)")
ax2.set_title("Temperature Ablation: Breakdown of Instance Outcomes")
ax2.legend(title="Outcome", bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.set_ylim(0, 100)

sns.despine()
plt.tight_layout()
plt.show()

print("\nDetailed Breakdown:")
print("Temperature | Resolved | Unresolved | Errors | Empty Patches")
print("-" * 60)
print(f"0.0         | {temp00_resolved:8d} | {temp00_unresolved:10d} | {temp00_errors:6d} | {temp00_empty:13d}")
print(f"0.2         | {temp02_resolved:8d} | {temp02_unresolved:10d} | {temp02_errors:6d} | {temp02_empty:13d}")
print(f"0.8         | {temp08_resolved:8d} | {temp08_unresolved:10d} | {temp08_errors:6d} | {temp08_empty:13d}")
