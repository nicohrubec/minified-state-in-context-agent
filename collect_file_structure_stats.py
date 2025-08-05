import os
import argparse

import pandas as pd
from datasets import load_dataset, load_from_disk
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path


from token_analysis.analyze_file_structure import analyze_file_structure


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_directory",
        type=Path,
        default="/Users/nicolashrubec/dev/agent-state-management/data/token-analysis",
    )
    parser.add_argument(
        "--repository_dataset_name",
        type=str,
        default="nicohrubec/codebase-content-SWE-bench_Verified",
    )
    parser.add_argument("--swe_bench_split", type=str, default="SWE-bench_Verified")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--dataset_directory",
        type=Path,
        default="/Users/nicolashrubec/dev/agent-state-management/data/hf_datasets",
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


def main():
    args = parse_arguments()
    hash_to_content, problem_files, problems = load_data(args)

    output_dir = args.output_directory
    output_dir.mkdir(exist_ok=True, parents=True)

    structural_token_usage = []
    lexical_token_usage = []
    repositories = set()

    num_skipped_files = 0
    num_files = 0

    for problem, files in tqdm(
        zip(problems, problem_files), total=len(problems), desc="Collecting data"
    ):
        instance_id = problem["instance_id"]
        repository = problem["repo"].split("/")[-1]
        repositories.add(repository)

        include_file_types = ["core", "config"]
        problem_structural_token_usage = {
            "problem": instance_id,
            "repository": repository,
            "token_usage": defaultdict(int),
        }
        problem_lexical_token_usage = {
            "problem": instance_id,
            "repository": repository,
            "token_usage": defaultdict(int),
        }
        problem_total_chars = 0

        for file in files["files"]:
            file_type = file["file_type"]
            num_files += 1

            if file_type not in include_file_types:
                continue

            file_hash = file["content_hash"]
            content = hash_to_content[file_hash]
            try:
                (
                    file_lexical_token_usage,
                    file_structural_token_usage,
                    file_total_chars,
                ) = analyze_file_structure(content)
            except (IndentationError, SyntaxError):
                num_skipped_files += 1
                continue

            for lexical_unit, num_tokens in file_lexical_token_usage.items():
                problem_lexical_token_usage["token_usage"][lexical_unit] += num_tokens
            for structural_unit, num_tokens in file_structural_token_usage.items():
                problem_structural_token_usage["token_usage"][structural_unit] += (
                    num_tokens
                )
            problem_total_chars += file_total_chars

        # unwrap token usage dicts into separate keys, so they are stored as individual columns in df
        for token_type in problem_lexical_token_usage["token_usage"]:
            problem_lexical_token_usage[token_type] = problem_lexical_token_usage[
                "token_usage"
            ][token_type]
        del problem_lexical_token_usage["token_usage"]
        for token_type in problem_structural_token_usage["token_usage"]:
            problem_structural_token_usage[token_type] = problem_structural_token_usage[
                "token_usage"
            ][token_type]
        del problem_structural_token_usage["token_usage"]

        problem_lexical_token_usage["total_chars"] = problem_total_chars
        problem_structural_token_usage["total_chars"] = problem_total_chars

        lexical_token_usage.append(problem_lexical_token_usage)
        structural_token_usage.append(problem_structural_token_usage)

    # store data
    lexical_df = pd.DataFrame(lexical_token_usage)
    lexical_output_file = (
        output_dir / f"lexical_{args.swe_bench_split}_{args.split}.csv"
    )
    lexical_df.to_csv(lexical_output_file, index=False)

    structural_df = pd.DataFrame(structural_token_usage)
    structural_output_file = (
        output_dir / f"structural_{args.swe_bench_split}_{args.split}.csv"
    )
    structural_df.to_csv(structural_output_file, index=False)

    print(
        f"Skipped {(num_skipped_files / num_files) * 100:.2f}% of files due to parsing issues."
    )


if __name__ == "__main__":
    main()
