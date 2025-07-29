import argparse
import hashlib
from pathlib import Path
import subprocess
from typing import Dict
import tqdm
import tokenize
import token
import io
import copy

from swebench_problem import _SWEBenchProblem, CodebaseContent, FileInCodebase


VALID_EXTENSIONS = {"py"}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_directory",
        type=Path,
        default="/Users/nicolashrubec/dev/agent-state-management/data/repositories",
    )
    parser.add_argument(
        "--output_dataset_prefix",
        type=str,
        default="nicohrubec/codebase-content",
    )
    parser.add_argument("--swe_bench_split", type=str, default="SWE-bench_Verified")
    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "dev", "test"]
    )
    parser.add_argument("--sample", type=float, default=1.0)
    return parser.parse_args()


def hash_file_content(file_content: str) -> str:
    return hashlib.sha256(file_content.encode()).hexdigest()


def clone_repos(problems: list[_SWEBenchProblem], repos_dir: Path):
    """Clones all the repos needed for SWE-bench Verified."""
    repos_dir.mkdir(exist_ok=True, parents=True)

    if len(list(repos_dir.iterdir())):
        print("Skip clone, local copy of repositories already exists.")
        return

    repos = {problem.repo for problem in problems}
    for repo in tqdm.tqdm(repos, desc="Cloning repos"):
        output = subprocess.run(
            ["git", "clone", f"https://github.com/{repo}.git"],
            cwd=repos_dir,
            capture_output=True,
        )
        assert output.returncode == 0


# Copied from: https://gist.github.com/BroHui/aca2b8e6e6bdf3cb4af4b246c9837fa3
def remove_comments(input_source: str) -> str:
    source_flag = copy.copy(input_source)
    source = io.StringIO(input_source)
    ls = input_source.split('\n')
    prev_toktype = token.INDENT
    readline = source.readline

    def get_char_index(lineno, col):
        # find the index of the char in the source code
        if lineno == 1:
            return len('\n'.join(ls[:(lineno-1)])) + col
        else:
            return len('\n'.join(ls[:(lineno-1)])) + col + 1

    def replace_char_between(start_lineno, start_col, end_lineno, end_col, source, replace_char, ls):
        # replace char between start_lineno, start_col and end_lineno, end_col with replace_char, but keep '\n' and ' '
        b = get_char_index(start_lineno, start_col)
        e = get_char_index(end_lineno, end_col)
        for i in range(b, e):
            if source[i] == '\n':
                source = source[:i] + '\n' + source[i+1:]
            elif source[i] == ' ':
                source = source[:i] + ' ' + source[i+1:]
            else:
                source = source[:i] + replace_char + source[i+1:]
        return source

    tokgen = tokenize.generate_tokens(readline)
    for toktype, ttext, (slineno, scol), (elineno, ecol), ltext in tokgen:
        if toktype == token.STRING and prev_toktype == token.INDENT:
            source_flag = replace_char_between(slineno, scol, elineno, ecol, source_flag, ' ', ls)
        elif toktype == tokenize.COMMENT:
            source_flag = replace_char_between(slineno, scol, elineno, ecol, source_flag, ' ', ls)
        prev_toktype = toktype
    return source_flag


def get_codebase_content(
    problem: _SWEBenchProblem, repos_dir: Path, hash_to_content: Dict[str, str]
) -> CodebaseContent:
    """Gets the content of the codebase for a specific problem.

    Updates the hash_to_content map in place with hashes of the content of each file.
    """
    repo = problem.repo.split("/")[-1]
    repo_path = repos_dir / repo

    subprocess.run(
        ["git", "checkout", problem.base_commit], cwd=repo_path, capture_output=True
    )

    contexts = []

    for file_path in repo_path.rglob("*"):
        if not file_path.is_file:
            continue

        if file_path.suffix[1:] not in VALID_EXTENSIONS:  # [1:] excludes the '.'
            continue

        try:
            content = file_path.read_text()
        except UnicodeDecodeError:
            # Ignore these files.
            continue

        content = remove_comments(content)
        content_hash = hash_file_content(content)
        if content_hash not in hash_to_content:
            hash_to_content[content_hash] = content

        contexts.append(
            FileInCodebase(
                file_path=str(file_path.relative_to(repo_path)),
                content_hash=content_hash,
            )
        )

    return CodebaseContent(instance_id=problem.instance_id, files=contexts)
