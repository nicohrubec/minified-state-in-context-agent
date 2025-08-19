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
    file_path = re.sub(r"^#+\s*", "", file_path).strip()
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

    num_applied_edits = 0
    for search_block, replace_block in edits:
        search_block = search_block.strip()
        replace_block = replace_block.strip()
        if search_block not in modified_content:
            print("Search block could not be found in file content.")
            continue

        modified_content = modified_content.replace(search_block, replace_block)
        num_applied_edits += 1

    if num_applied_edits <= 0:
        print("No edits were applied.")
        return None

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


def decode_paths(paths, replacements):
    if len(replacements) <= 0:
        return paths

    inv_replacements = {
        str(replacement_int): folder for folder, replacement_int in replacements.items()
    }

    for path_idx, path in enumerate(paths):
        parts = path.split("/")
        for part_idx, part in enumerate(parts):
            if part in inv_replacements:
                parts[part_idx] = inv_replacements[part]

        paths[path_idx] = "/".join(parts)

    return paths


def run_agent(
    problem,
    problem_files,
    hash_to_content,
    repo_base_dir,
    skip_repair,
    rank_encoding,
    transformations,
    token_limit=10000,
):
    print(f"Running agent for problem {problem['instance_id']}")
    problem_files["files"] = [
        file
        for file in problem_files["files"]
        if file["file_type"] in ["core", "config"]
    ]
    patch_text = problem["patch"]
    instance_id = problem["instance_id"]

    repo_dir = Path(repo_base_dir) / problem["repo"].split("/")[-1]

    # preprocessing step
    # 1. file ranking
    prompt, replacements = build_file_ranking_prompt(
        problem, problem_files, rank_encoding
    )
    num_ranking_input_tokens = count_tokens(prompt)
    response = call_gpt("", prompt)
    num_ranking_output_tokens = count_tokens(response)

    # 2. filter based on ranking
    ranked_paths = [line.strip() for line in response.splitlines() if line.strip()]
    decoded_ranked_paths = decode_paths(ranked_paths, replacements)
    problem_files, selected_paths = filter_problem_files_with_ranking(
        problem_files, hash_to_content, decoded_ranked_paths, token_limit
    )

    target_files = extract_target_files_from_patch(patch_text)
    target_file_positions, num_target_files_in_ranking = (
        get_target_file_positions_in_ranking(ranked_paths, target_files)
    )

    # get recall of preprocessing
    num_found_target_files = len(set(selected_paths).intersection(target_files))
    recall = num_found_target_files / len(target_files)

    print(f"The ranking prompt has {num_ranking_input_tokens} input tokens")
    print(f"The ranking output has {num_ranking_output_tokens} output tokens")
    print(f"{len(problem_files['files'])} files were selected")
    print(f"Recall: {recall:.3f}")

    if skip_repair:
        return {
            "instance_id": instance_id,
            "num_ranking_input_tokens": num_ranking_input_tokens,
            "num_ranking_output_tokens": num_ranking_output_tokens,
            "num_selected_files": len(problem_files["files"]),
            "num_target_files": len(target_files),
            "num_selected_target_files": num_found_target_files,
            "target_file_positions": target_file_positions,
            "num_target_files_in_ranking": num_target_files_in_ranking,
            "recall": recall,
        }

    # repair
    system_prompt, user_prompt, _, source_maps = build_repair_prompt(
        problem, problem_files, hash_to_content, transformations
    )
    num_repair_input_tokens = count_tokens(system_prompt + user_prompt)
    response = call_gpt(system_prompt, user_prompt)
    num_repair_output_tokens = count_tokens(response)

    chain_of_thoughts = extract_cot_sections(response)
    chain_of_thoughts["instance_id"] = instance_id

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

    print(f"The repair prompt has {num_repair_input_tokens} tokens")
    print(f"The repair output has {num_repair_output_tokens} output tokens")

    metrics = {
        "instance_id": instance_id,
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

    response_json = {
        "instance_id": instance_id,
        "response": response,
    }

    return prediction, chain_of_thoughts, metrics, response_json
