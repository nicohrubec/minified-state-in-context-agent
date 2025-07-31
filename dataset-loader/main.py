import datasets
import numpy as np
import tqdm
import os
from pathlib import Path
from dataclasses import asdict

from swebench_problem import _SWEBenchProblem
from helper import parse_arguments, clone_repos, get_codebase_content


def main():
    np.random.seed(42)
    args = parse_arguments()
    output_dataset_name = f"{args.output_dataset_prefix}-{args.swe_bench_split}"

    split_name = (
        f"{args.split}_s{str(args.sample).replace('.', '_')}"
        if args.sample != 1.0
        else args.split
    )

    local_dir = Path(args.dataset_directory) / output_dataset_name / split_name
    local_file_content_path = local_dir / "file_content"
    local_problem_files_path = local_dir / "problem_files"

    if local_file_content_path.exists() and local_problem_files_path.exists():
        print(f"Loading cached dataset from {local_dir}")
        file_content_dataset = datasets.load_from_disk(str(local_file_content_path))
        problems_dataset = datasets.load_from_disk(str(local_problem_files_path))
    else:
        print(f"No cache found. Creating dataset for split '{split_name}'")

        dataset = datasets.load_dataset(
            f"princeton-nlp/{args.swe_bench_split}", split=args.split
        )
        sample_size = int(args.sample * len(dataset))
        problems = np.random.choice(
            [_SWEBenchProblem(row) for row in dataset], size=sample_size, replace=False
        )

        clone_repos(problems, args.repo_directory)

        hash_to_content = {}
        codebase_content_per_problem = [
            get_codebase_content(problem, args.repo_directory, hash_to_content)
            for problem in tqdm.tqdm(problems, desc="Fetching codebase content")
        ]

        hash_to_content_in_hf_form = [
            {"hash": hash_, "content": content}
            for hash_, content in hash_to_content.items()
        ]
        codebase_content_in_hf_form = [
            asdict(problem) for problem in codebase_content_per_problem
        ]

        file_content_dataset = datasets.Dataset.from_list(
            hash_to_content_in_hf_form, split=split_name
        )
        problems_dataset = datasets.Dataset.from_list(
            codebase_content_in_hf_form, split=split_name
        )

        print(f"Saving dataset to disk at {local_dir}")
        local_file_content_path.mkdir(parents=True, exist_ok=True)
        local_problem_files_path.mkdir(parents=True, exist_ok=True)
        file_content_dataset.save_to_disk(str(local_file_content_path))
        problems_dataset.save_to_disk(str(local_problem_files_path))

    if args.push_to_hub:
        print("Pushing dataset to Hugging Face Hub...")
        file_content_dataset.push_to_hub(
            output_dataset_name,
            config_name="file_content",
            split=split_name,
            private=True,
            max_shard_size="256MB",
            token=os.environ.get("HF_TOKEN"),
        )
        problems_dataset.push_to_hub(
            output_dataset_name,
            config_name="problem_files",
            split=split_name,
            private=True,
            max_shard_size="256MB",
            token=os.environ.get("HF_TOKEN"),
        )
    else:
        print("Skip hub upload.")


if __name__ == "__main__":
    main()
