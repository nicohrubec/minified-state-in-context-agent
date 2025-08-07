import argparse
from datasets import load_dataset, load_from_disk
import os
from pathlib import Path
import pandas as pd
import json

from agent.run import run_agent


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository_dataset_name",
        type=str,
        default="nicohrubec/codebase-content-SWE-bench_Verified",
    )
    parser.add_argument("--swe_bench_split", type=str, default="SWE-bench_Verified")
    parser.add_argument("--split", type=str, default="test_s0_01")
    parser.add_argument(
        "--dataset_directory",
        type=Path,
        default="/Users/nicolashrubec/dev/agent-state-management/data/hf_datasets",
    )
    parser.add_argument(
        "--output_directory",
        type=Path,
        default="/Users/nicolashrubec/dev/agent-state-management/data/agent_results",
    )
    parser.add_argument(
        "--repository_directory",
        default="/Users/nicolashrubec/dev/agent-state-management/data/repositories",
    )
    return parser.parse_args()


def load_data(args):
    local_base_path = args.dataset_directory / args.repository_dataset_name / args.split

    local_file_content_path = local_base_path / "file_content"
    local_problem_files_path = local_base_path / "problem_files"

    if local_file_content_path.exists() and local_problem_files_path.exists():
        print(
            f"Loading file_content and problem_files from local disk: {local_base_path}"
        )
        file_content = load_from_disk(str(local_file_content_path))
        problem_files = load_from_disk(str(local_problem_files_path))
    else:
        print(
            f"Loading file_content and problem_files from Hugging Face: {args.repository_dataset_name}"
        )
        file_content = load_dataset(
            args.repository_dataset_name,
            "file_content",
            split=args.split,
            token=os.environ.get("HF_TOKEN"),
        )
        problem_files = load_dataset(
            args.repository_dataset_name,
            "problem_files",
            split=args.split,
            token=os.environ.get("HF_TOKEN"),
        )

    hash_to_content = {row["hash"]: row["content"] for row in file_content}

    problems = load_dataset(
        f"princeton-nlp/{args.swe_bench_split}",
        split="test",
        token=os.environ.get("HF_TOKEN"),
    )

    # select relevant instance_ids from problems and ensure it is ordered correctly
    instance_ids = list(problem_files["instance_id"])
    id_to_row_index = {p["instance_id"]: i for i, p in enumerate(problems)}
    sorted_indices = [
        id_to_row_index[iid] for iid in instance_ids if iid in id_to_row_index
    ]
    problems = problems.select(sorted_indices)

    return hash_to_content, problem_files, problems


def write_jsonl(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in data:
            json.dump(entry, f)
            f.write("\n")


def main():
    args = parse_arguments()
    hash_to_content, problem_files, problems = load_data(args)

    output_dir = args.output_directory
    output_dir.mkdir(exist_ok=True, parents=True)

    predictions = []
    cots = []
    metrics = []
    raw_responses = []

    for problem, files in zip(problems, problem_files):
        prediction, instance_cots, instance_metrics, raw_response = run_agent(
            problem, files, hash_to_content, args.repository_directory
        )

        predictions.append(prediction)
        cots.append(instance_cots)
        metrics.append(instance_metrics)
        raw_responses.append(raw_response)

    predictions_output_file = (
        output_dir / f"predictions_{args.swe_bench_split}_{args.split}.jsonl"
    )
    write_jsonl(predictions, predictions_output_file)

    cots_output_file = output_dir / f"cots_{args.swe_bench_split}_{args.split}.jsonl"
    write_jsonl(cots, cots_output_file)

    metrics_df = pd.DataFrame(metrics)
    metrics_output_file = (
        output_dir / f"metrics_{args.swe_bench_split}_{args.split}.csv"
    )
    metrics_df.to_csv(metrics_output_file, index=False)

    raw_responses_output_file = (
        output_dir / f"responses_{args.swe_bench_split}_{args.split}.json"
    )
    with open(raw_responses_output_file, "w") as f:
        json.dump(raw_responses, f, indent=2)


if __name__ == "__main__":
    main()
