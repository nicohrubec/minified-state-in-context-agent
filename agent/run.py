import re
import difflib

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


def extract_final_patch_as_diff(response_text, files, hash_to_content):
    file_path_to_content_hash = {
        file["file_path"]: file["content_hash"] for file in files
    }

    file_match = re.search(
        r"\*{0,2}Final Patch\*{0,2}\s*-+\s*\n\s*(\S+)", response_text
    )
    if not file_match:
        return None
    file_path = file_match.group(1).strip()

    search_match = re.search(
        r"SEARCH:\s*\n(?:(?:```(?:[^\n]*)\n)?(.*?)(?:```)?)(?:\nREPLACE:|\Z)",
        response_text,
        re.DOTALL,
    )
    replace_match = re.search(
        r"REPLACE:\s*\n(?:(?:```(?:[^\n]*)\n)?(.*?)(?:```)?)(?:\n[A-Z]+:|\Z)",
        response_text,
        re.DOTALL,
    )
    if not (search_match and replace_match):
        return None

    search_block = search_match.group(1).strip()
    replace_block = replace_match.group(1).strip()

    search_lines = search_block.splitlines()
    replace_lines = replace_block.splitlines()

    if file_path not in file_path_to_content_hash:
        return None

    original_lines = hash_to_content[file_path_to_content_hash[file_path]].splitlines(
        keepends=True
    )

    # Locate the block to replace
    match_index = None
    for i in range(len(original_lines) - len(search_lines) + 1):
        original_block = [
            line.strip() for line in original_lines[i : i + len(search_lines)]
        ]
        if original_block == [line.strip() for line in search_lines]:
            match_index = i
            break
    if match_index is None:
        return None

    new_lines = (
        original_lines[:match_index]
        + [line + "\n" for line in replace_lines]
        + original_lines[match_index + len(search_lines) :]
    )

    diff = list(
        difflib.unified_diff(
            original_lines,
            new_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm="",
        )
    )

    full_diff = [f"diff --git a/{file_path} b/{file_path}"] + diff
    return "\n".join(full_diff)


def run_agent(problem, problem_files, hash_to_content, token_limit=10000):
    print(f"Running agent for problem {problem['instance_id']}")
    problem_files["files"] = [
        file
        for file in problem_files["files"]
        if file["file_type"] in ["core", "config"]
    ]

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

    # get recall of preprocessing
    patch_text = problem["patch"]
    target_files = extract_target_files_from_patch(patch_text)

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

    # convert response to patch
    patch = extract_final_patch_as_diff(
        response, problem_files["files"], hash_to_content
    )
    prediction = (
        {
            "instance_id": problem["instance_id"],
            "model": "gpt-4.1",
            "prediction": patch,
        }
        if patch
        else None
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
        "recall": recall,
    }

    return prediction, chain_of_thoughts, metrics
