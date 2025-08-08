import re
from pathlib import Path
import subprocess

from agent.prompt import build_file_ranking_prompt, build_repair_prompt
from agent.llm import call_gpt
from shared.tokens import count_tokens


def filter_problem_files_with_ranking(
    problem_files, hash_to_content, ranked_paths, token_limit
):
    selected_paths = []
    total_tokens = 0

    # greedily pack context window based on ranking up to token limit
    for path in ranked_paths:
        # locate the corresponding file entry
        file_entry = next(
            (f for f in problem_files["files"] if f["file_path"] == path), None
        )
        if not file_entry:
            continue  # skip files not in the current problem_files list

        content = hash_to_content[file_entry["content_hash"]]
        file_tokens = count_tokens(f"### {path}\n{content}")

        if total_tokens + file_tokens <= token_limit:
            selected_paths.append(path)
            total_tokens += file_tokens
        else:
            break

    # filter problem files based on selected paths
    problem_files["files"] = [
        file for file in problem_files["files"] if file["file_path"] in selected_paths
    ]

    return problem_files, selected_paths


def extract_target_files_from_patch(patch_text):
    target_files = set()
    for line in patch_text.splitlines():
        if line.startswith("+++ b/"):
            path = line[len("+++ b/") :].strip()
            if path:
                target_files.add(path)
    return target_files


def extract_cot_sections(response):
    localization, repair = "", ""
    localization_match = re.search(
        r"Chain-of-Thought for Localization\s*-+\n(.*?)\n(?=Restated Relevant Code)",
        response,
        re.DOTALL,
    )
    repair_match = re.search(
        r"Chain-of-Thought for Repairing the Code\s*-+\n(.*?)\n(?=Final Patch)",
        response,
        re.DOTALL,
    )

    if localization_match:
        localization = localization_match.group(1).strip()
    if repair_match:
        repair = repair_match.group(1).strip()

    return {
        "localization": localization,
        "repair": repair,
    }


def extract_final_patch_as_diff(response_text, repo_dir):
    file_match = re.search(
        r"\*{0,2}Final Patch\*{0,2}\s*-+\s*\n\s*(\S+)", response_text
    )
    if not file_match:
        print("File path could not be matched!")
        return None
    file_path = file_match.group(1).strip()
    repo_path = Path(repo_dir)
    target_file = repo_path / file_path

    if not target_file.exists():
        print("Target file does not exist!")
        return None

    original_content = target_file.read_text()
    modified_content = original_content

    patch_block_match = re.search(
        r"Final Patch\s*-+\s*" + re.escape(file_path) + r"\s*\n(.*)",
        response_text,
        re.DOTALL,
    )
    if not patch_block_match:
        print("Final patch block not found!")
        return None

    patch_block = patch_block_match.group(1)

    edit_pattern = re.compile(
        r"<{2,}\s*SEARCH\s*\n(.*?)\n=+\n(.*?)\n>+.*REPLACE",
        re.DOTALL,
    )
    edits = edit_pattern.findall(patch_block)

    if not edits:
        print("No edits found in final patch.")
        return None

    for search_block, replace_block in edits:
        search_block = search_block.strip()
        replace_block = replace_block.strip()
        if search_block not in modified_content:
            print("Search block could not be found in file content.")
            return None
        modified_content = modified_content.replace(search_block, replace_block)

    target_file.write_text(modified_content)

    try:
        result = subprocess.run(
            ["git", "diff", "--", str(target_file)],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        diff_output = result.stdout
    finally:
        target_file.write_text(original_content)

    return diff_output


def get_target_file_positions_in_ranking(ranked_paths, target_files):
    target_file_positions = []
    num_target_files_in_ranking = 0

    for idx, path in enumerate(ranked_paths):
        if path in target_files:
            target_file_positions.append(idx + 1)
            num_target_files_in_ranking += 1

    return target_file_positions, num_target_files_in_ranking


def run_agent(
    problem, problem_files, hash_to_content, repo_base_dir, token_limit=10000
):
    print(f"Running agent for problem {problem['instance_id']}")
    problem_files["files"] = [
        file
        for file in problem_files["files"]
        if file["file_type"] in ["core", "config"]
    ]

    repo_dir = Path(repo_base_dir) / problem["repo"].split("/")[-1]

    # preprocessing step
    # 1. file ranking
    prompt = build_file_ranking_prompt(problem, problem_files)
    num_ranking_input_tokens = count_tokens(prompt)
    response = call_gpt("", prompt)
    num_ranking_output_tokens = count_tokens(response)

    # 2. filter based on ranking
    ranked_paths = [line.strip() for line in response.splitlines() if line.strip()]
    problem_files, selected_paths = filter_problem_files_with_ranking(
        problem_files, hash_to_content, ranked_paths, token_limit
    )

    patch_text = problem["patch"]
    target_files = extract_target_files_from_patch(patch_text)
    target_file_positions, num_target_files_in_ranking = (
        get_target_file_positions_in_ranking(ranked_paths, target_files)
    )

    # get recall of preprocessing
    num_found_target_files = len(set(selected_paths).intersection(target_files))
    recall = num_found_target_files / len(target_files)

    # problem solving
    system_prompt, user_prompt, _ = build_repair_prompt(
        problem, problem_files, hash_to_content
    )
    num_repair_input_tokens = count_tokens(system_prompt + user_prompt)
    response = call_gpt(system_prompt, user_prompt)
    num_repair_output_tokens = count_tokens(response)

    chain_of_thoughts = extract_cot_sections(response)
    patch = extract_final_patch_as_diff(response, repo_dir)

    prediction = (
        {
            "instance_id": problem["instance_id"],
            "model_name_or_path": "gpt-4.1",
            "model_patch": patch,
        }
        if patch
        else {
            "instance_id": problem["instance_id"],
            "model_name_or_path": "gpt-4.1",
            "model_patch": "",
        }
    )

    print(f"The ranking prompt has {num_ranking_input_tokens} input tokens")
    print(f"The ranking output has {num_ranking_output_tokens} output tokens")
    print(f"The repair prompt has {num_repair_input_tokens} tokens")
    print(f"The repair output has {num_repair_output_tokens} output tokens")
    print(f"{len(problem_files['files'])} files were selected")
    print(f"Recall: {recall:.3f}")

    metrics = {
        "num_ranking_input_tokens": num_ranking_input_tokens,
        "num_ranking_output_tokens": num_ranking_output_tokens,
        "num_repair_input_tokens": num_repair_input_tokens,
        "num_repair_output_tokens": num_repair_output_tokens,
        "num_selected_files": len(problem_files["files"]),
        "num_target_files": len(target_files),
        "num_selected_target_files": num_found_target_files,
        "target_file_positions": target_file_positions,
        "num_target_files_in_ranking": num_target_files_in_ranking,
        "recall": recall,
    }

    return prediction, chain_of_thoughts, metrics, response
