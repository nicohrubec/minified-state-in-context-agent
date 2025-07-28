from dataclasses import asdict
import datasets
from datasets import Dataset
import tqdm
import os

from swebench_problem import _SWEBenchProblem
from helper import parse_arguments, clone_repos, get_codebase_content

split_to_hf_split = {
    "train": datasets.Split.TRAIN,
    "dev": datasets.Split.VALIDATION,
    "test": datasets.Split.TEST,
}


def main():
    args = parse_arguments()
    output_dataset_name = f"{args.output_dataset_prefix}-{args.swe_bench_split}"
    hf_split = split_to_hf_split[args.split]

    dataset = datasets.load_dataset(
        f"princeton-nlp/{args.swe_bench_split}", split=args.split
    )
    problems = [_SWEBenchProblem(row) for row in dataset]

    clone_repos(problems, args.repo_directory)
    hash_to_content = {}
    codebase_content_per_problem = [
        get_codebase_content(problem, args.repo_directory, hash_to_content)
        for problem in tqdm.tqdm(problems, desc="Fetching codebase content")
    ]

    hash_to_content_in_hf_form = [
        {
            "hash": hash_,
            "content": content,
        }
        for (hash_, content) in hash_to_content.items()
    ]

    # for each problem we get a list of files as (hash, file_name)
    codebase_content_in_hf_form = [
        asdict(problem) for problem in codebase_content_per_problem
    ]

    file_content_dataset = Dataset.from_list(hash_to_content_in_hf_form, split=hf_split)
    problems_dataset = Dataset.from_list(codebase_content_in_hf_form, split=hf_split)

    print("Pushing dataset to hf ...")
    file_content_dataset.push_to_hub(
        output_dataset_name,
        "file_content",
        private=True,
        max_shard_size="256MB",
        token=os.environ.get("HF_TOKEN"),
    )
    problems_dataset.push_to_hub(
        output_dataset_name,
        "problem_files",
        private=True,
        max_shard_size="256MB",
        token=os.environ.get("HF_TOKEN"),
    )


if __name__ == "__main__":
    main()
