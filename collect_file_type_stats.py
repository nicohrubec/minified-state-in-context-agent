import os
import argparse
from datasets import load_dataset, load_from_disk
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pandas as pd


from agent.prompt import build_repair_prompt
from shared.tokens import count_tokens


file_types = ["core", "test", "docs", "config", "bench", "build"]


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

    # select relevant instance_ids from problems
    instance_ids = set(problem_files["instance_id"])
    problems = problems.filter(lambda x: x["instance_id"] in instance_ids)

    return hash_to_content, problem_files, problems


def main():
    args = parse_arguments()
    hash_to_content, problem_files, problems = load_data(args)

    output_dir = args.output_directory
    output_dir.mkdir(exist_ok=True, parents=True)

    repository_to_nl_tokens = defaultdict(list)
    repository_to_code_tokens = defaultdict(list)
    repository_to_file_type_tokens = defaultdict(lambda: defaultdict(list))
    repositories = set()

    for problem, files in tqdm(
        zip(problems, problem_files),
        total=len(problems),
        desc="Collecting token counts",
    ):
        repository = problem["repo"].split("/")[-1]
        repositories.add(repository)

        system_prompt, user_prompt, code_input = build_repair_prompt(
            problem, files, hash_to_content
        )
        full_prompt = system_prompt + user_prompt

        all_tokens = count_tokens(full_prompt)
        code_tokens = count_tokens(code_input)
        nl_tokens = all_tokens - code_tokens

        repository_to_nl_tokens[repository].append(nl_tokens)
        repository_to_code_tokens[repository].append(code_tokens)

        instance_filetype_tokens = defaultdict(int)

        for file in files["files"]:
            file_type = file["file_type"]
            file_hash = file["content_hash"]
            content = hash_to_content[file_hash]
            file_tokens = count_tokens(content)
            instance_filetype_tokens[file_type] += file_tokens

        for file_type, token_sum in instance_filetype_tokens.items():
            repository_to_file_type_tokens[repository][file_type].append(token_sum)

    rows = []
    for repository in tqdm(repositories, desc="Store aggregate statistics"):
        nl_values = repository_to_nl_tokens[repository]
        code_values = repository_to_code_tokens[repository]

        nl_mean = np.mean(nl_values)
        nl_std = np.std(nl_values)
        code_mean = np.mean(code_values)
        code_std = np.std(code_values)

        row = {
            "repository": repository,
            "nl_mean": nl_mean,
            "nl_std": nl_std,
            "code_mean": code_mean,
            "code_std": code_std,
        }

        for file_type in file_types:
            token_counts = repository_to_file_type_tokens[repository].get(file_type, [])
            row[f"{file_type}_mean"] = np.mean(token_counts) if token_counts else 0.0
            row[f"{file_type}_std"] = np.std(token_counts) if token_counts else 0.0

        rows.append(row)

    df = pd.DataFrame(rows)
    output_file = output_dir / f"file_type_{args.swe_bench_split}_{args.split}.csv"
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
