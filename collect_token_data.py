import os
import argparse
import datasets
from datasets import load_dataset
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pandas as pd


from agent.prompt import build_repair_prompt
from shared.tokens import count_tokens


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
    parser.add_argument("--split", type=str, default="test_s0_01")
    return parser.parse_args()


def load_data(args):
    file_content = load_dataset(
        args.repository_dataset_name,
        "file_content",
        split=args.split,
        token=os.environ.get("HF_TOKEN"),
    )
    hash_to_content = {row["hash"]: row["content"] for row in file_content}
    problem_files = load_dataset(
        args.repository_dataset_name,
        "problem_files",
        split=args.split,
        token=os.environ.get("HF_TOKEN"),
    )
    problems = datasets.load_dataset(
        f"princeton-nlp/{args.swe_bench_split}", split="test"
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

    # analyze natural language to code token ratio in prompts
    repository_to_nl_tokens = defaultdict(list)
    repository_to_code_tokens = defaultdict(list)
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

    rows = []
    for repository in tqdm(repositories, desc="Store mean and std"):
        nl_values = repository_to_nl_tokens[repository]
        code_values = repository_to_code_tokens[repository]

        nl_mean = np.mean(nl_values)
        nl_std = np.std(nl_values)
        code_mean = np.mean(code_values)
        code_std = np.std(code_values)

        rows.append(
            {
                "repository": repository,
                "nl_mean": nl_mean,
                "nl_std": nl_std,
                "token_mean": code_mean,
                "token_std": code_std,
            }
        )

    df = pd.DataFrame(rows)
    output_file = output_dir / f"nl_code_{args.swe_bench_split}_{args.split}.csv"
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
