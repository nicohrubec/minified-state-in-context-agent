# agent-state-management

Repository for my master thesis on SWE agent state management.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Set environment variables:
```bash
export AZURE_OPENAI_API_KEY="your_azure_openai_api_key"
export AZURE_OPENAI_ENDPOINT="your_azure_openai_endpoint"
export HF_TOKEN="your_huggingface_token"  # Optional: for dataset access
```

## Run

### Prepare SWE-Bench repositories
```bash
python dataset_loader/main.py [OPTIONS]
```

**Arguments:**
- `--repo_directory PATH`: Local directory to store cloned repositories
- `--dataset_directory PATH`: Local directory to store datasets
- `--swe_bench_split STR`: SWE-Bench split to use (default: `SWE-bench_Verified`)
- `--split STR`: Dataset split - train, dev, or test (default: `test`)
- `--sample FLOAT`: Fraction of dataset to sample (default: `1.0`)
- `--push_to_hub`: Push dataset to Hugging Face Hub
- `--output_dataset_prefix STR`: Prefix for output dataset name on HF (default: `nicohrubec/codebase-content`)

### File structure analysis
1. Collect data:
```bash
python collect_file_structure_stats.py [OPTIONS]
```

**Arguments:**
- `--output_directory PATH`: Directory to save results
- `--swe_bench_split STR`: SWE-Bench split to use (default: `SWE-bench_Verified`)
- `--split STR`: Dataset split (default: `test`)
- `--dataset_directory PATH`: Local directory containing datasets, if datasets are not available here they will be loaded from HF
- `--repository_dataset_name STR`: Name of repository dataset on HF

2. Plot results:
```bash
python token_analysis/plot_file_structure_analysis.py [OPTIONS]
```

**Arguments:**
- `--lexical_path PATH`: Path to lexical analysis CSV file
- `--structural_path PATH`: Path to structural analysis CSV file

### File type analysis
1. Collect data:
```bash
python collect_file_type_stats.py [OPTIONS]
```

**Arguments:**
- `--output_directory PATH`: Directory to save results
- `--swe_bench_split STR`: SWE-Bench split to use (default: `SWE-bench_Verified`)
- `--split STR`: Dataset split (default: `test`)
- `--dataset_directory PATH`: Local directory containing datasets, if datasets are not available here they will be loaded from HF
- `--repository_dataset_name STR`: Name of repository dataset on HF

2. Plot results:
```bash
python token_analysis/plot_file_type_analysis.py [OPTIONS]
```

**Arguments:**
- `--file_type_path PATH`: Path to file type analysis CSV file

### Run the agent
```bash
python run_agent.py [OPTIONS]
```

**Arguments:**

- `--swe_bench_split STR`: SWE-Bench split to use (default: `SWE-bench_Verified`)
- `--split STR`: Dataset split (default: `test_s0_01`)
- `--dataset_directory PATH`: Directory containing datasets, if datasets are not available here they will be loaded from HF
- `--output_directory PATH`: Directory to save results
- `--repository_directory PATH`: Directory containing repositories
- `--skip_repair`: Skip the repair stage and only run file ranking
- `--rank_encoding STR`: Encoding method for file ranking - list, trie, int_folder, or int_path (default: `list`)
- `--transformations LIST`: List of transformations to apply (choices: transformations defined in agent.minify)
- `--repository_dataset_name STR`: Name of repository dataset (default: `nicohrubec/codebase-content-SWE-bench_Verified`)

The agent uses a two-stage approach:
1. **File ranking**: Ranks relevant files for the given problem
2. **Code repair**: Generates patches to fix the identified issues

### Analyze ranking results
```bash
python analysis/ranking.py
```

This script analyzes the ranking performance and cost metrics across different encoding methods. It generates:
- Cost comparison plots (input, output, and overall costs)
- Cost vs recall scatter plots
- Target file position density plots

The script expects ranking metrics CSV files in the `data/agent_results/` directory with the naming pattern `ranking_metrics_SWE-bench_Verified_test_{encoding}_.csv`.

### SWE-Bench Evaluation
After running the agent, evaluate the predictions using SWE-Bench, e.g.:

```bash
python3 -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --predictions_path data/agent_results/predictions_SWE-bench_Verified_test.csv \
    --max_workers 8 \
    --run_id my_evaluation_run
```

The agent predictions are stored in the `data/agent_results/` directory. The evaluation will test the generated patches against the actual repositories to verify if the issues are resolved.
