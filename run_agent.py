import argparse
from datasets import load_dataset
import os
import datasets

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

    return hash_to_content, problem_files, problems


def main():
    args = parse_arguments()
    hash_to_content, problem_files, problems = load_data(args)

    for problem, files in zip(problems, problem_files):
        run_agent(problem, files, hash_to_content)


if __name__ == "__main__":
    main()
