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
python dataset_loader/main.py
```

### File structure analysis
1. Collect data:
```bash
python collect_file_structure_stats.py
```

2. Plot results:
```bash
python token_analysis/plot_file_structure_analysis.py
```

### File type analysis
1. Collect data:
```bash
python collect_file_type_stats.py
```

2. Plot results:
```bash
python token_analysis/plot_file_type_analysis.py
```

### Run the agent
```bash
python run_agent.py
```

The agent uses a two-stage approach:
1. **File ranking**: Ranks relevant files for the given problem
2. **Code repair**: Generates patches to fix the identified issues

### Evaluate results
After running the agent, evaluate the predictions using SWE-Bench, e.g.:

```bash
python3 -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --predictions_path data/agent_results/predictions_SWE-bench_Verified_test.csv \
    --max_workers 8 \
    --run_id my_evaluation_run
```

The predictions are stored in the `data/agent_results/` directory. The evaluation will test the generated patches against the actual repositories to verify if the issues are resolved.
