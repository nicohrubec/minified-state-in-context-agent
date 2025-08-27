import argparse
from datasets import load_dataset, load_from_disk
import os
from pathlib import Path
import pandas as pd
import json

from agent.run import run_agent
from agent.minify import DEFINED_TRANSFORMATIONS


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository_dataset_name",
        type=str,
        default="nicohrubec/codebase-content-SWE-bench_Verified",
    )
    parser.add_argument("--swe_bench_split", type=str, default="SWE-bench_Verified")
    parser.add_argument("--split", type=str, default="test_s0_01")
    parser.add_argument("--model", type=str, default="gpt-4.1")
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
    parser.add_argument("--skip_repair", action="store_true")
    parser.add_argument(
        "--rank_encoding",
        default="list",
        choices=["list", "trie", "int_folder", "int_path"],
    )
    parser.add_argument(
        "--transformations",
        nargs="*",
        default=[],
        choices=DEFINED_TRANSFORMATIONS,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Temperature parameter for LLM calls (only used for non-gpt-5 models)",
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

    metrics_dir = output_dir / "metrics"
    predictions_dir = output_dir / "predictions"
    cots_dir = output_dir / "chain_of_thought"
    responses_dir = output_dir / "raw_responses"

    for subdir in [metrics_dir, predictions_dir, cots_dir, responses_dir]:
        subdir.mkdir(exist_ok=True, parents=True)

    predictions = []
    cots = []
    metrics = []
    raw_responses = []

    transformations = args.transformations
    transformations_suffix = "_".join(transformations)

    if "gpt-5" in args.model:
        model_suffix = args.model
    else:
        model_suffix = f"{args.model}_temp{args.temperature}"

    num_failed_predictions = 0
    for i, (problem, files) in enumerate(zip(problems, problem_files), 1):
        print(f"Processing problem {i}/{len(problems)}: {problem['instance_id']}")

        if args.skip_repair:
            instance_metrics = run_agent(
                problem,
                files,
                hash_to_content,
                args.repository_directory,
                args.skip_repair,
                args.rank_encoding,
                transformations,
                args.model,
                temperature=args.temperature,
            )
            metrics.append(instance_metrics)
        else:
            prediction, instance_cots, instance_metrics, raw_response = run_agent(
                problem,
                files,
                hash_to_content,
                args.repository_directory,
                args.skip_repair,
                args.rank_encoding,
                transformations,
                args.model,
                temperature=args.temperature,
            )

            predictions.append(prediction)
            cots.append(instance_cots)
            metrics.append(instance_metrics)
            raw_responses.append(raw_response)

            if prediction["model_patch"] == "":
                num_failed_predictions += 1

    if args.skip_repair:
        metrics_file_name = f"ranking_metrics_{model_suffix}_{args.swe_bench_split}_{args.split}_{args.rank_encoding}_{transformations_suffix}.csv"
    else:
        metrics_file_name = f"metrics_{model_suffix}_{args.swe_bench_split}_{args.split}_{args.rank_encoding}_{transformations_suffix}.csv"

    metrics_df = pd.DataFrame(metrics)
    metrics_output_file = metrics_dir / metrics_file_name
    metrics_df.to_csv(metrics_output_file, index=False)

    if args.skip_repair:
        return

    print(f"Number of failed predictions: {num_failed_predictions}")
    predictions_output_file = (
        predictions_dir
        / f"predictions_{model_suffix}_{args.swe_bench_split}_{args.split}_{args.rank_encoding}_{transformations_suffix}.jsonl"
    )
    write_jsonl(predictions, predictions_output_file)

    cots_output_file = (
        cots_dir
        / f"cots_{model_suffix}_{args.swe_bench_split}_{args.split}_{args.rank_encoding}_{transformations_suffix}.jsonl"
    )
    write_jsonl(cots, cots_output_file)

    raw_responses_output_file = (
        responses_dir
        / f"responses_{model_suffix}_{args.swe_bench_split}_{args.split}_{args.rank_encoding}_{transformations_suffix}.json"
    )
    with open(raw_responses_output_file, "w") as f:
        json.dump(raw_responses, f, indent=2)


if __name__ == "__main__":
    main()
