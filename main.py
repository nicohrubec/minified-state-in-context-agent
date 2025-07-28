import argparse
from datasets import load_dataset
import os


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="nicohrubec/codebase-content-SWE-bench_Verified",
        token=os.environ.get("HF_TOKEN"),
    )
    parser.add_argument("--split", type=str, default="test")
    return parser.parse_args()


def load_data(args):
    file_content = load_dataset(args.dataset_name, "file_content", split=args.split)
    hash_to_content = {row["hash"]: row["content"] for row in file_content}
    problem_files = load_dataset(args.dataset_name, "problem_files", split=args.split)

    return file_content, hash_to_content, problem_files


def main():
    args = parse_arguments()
    file_content, hash_to_content, problem_files = load_data(args)
    print("Done!")


if __name__ == "__main__":
    main()
