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

    return problem_files


def run_agent(problem, problem_files, hash_to_content, token_limit=60000):
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
    print(f"The ranking prompt has {num_ranking_input_tokens} input tokens")
    response = call_gpt("", prompt)
    num_ranking_output_tokens = count_tokens(response)
    print(f"The ranking output has {num_ranking_output_tokens} output tokens")

    # 2. filter based on ranking
    ranked_paths = [line.strip() for line in response.splitlines() if line.strip()]
    problem_files = filter_problem_files_with_ranking(
        problem_files, hash_to_content, ranked_paths, token_limit
    )
    print(f"{len(problem_files['files'])} files were selected")

    # problem solving
    system_prompt, user_prompt, _ = build_repair_prompt(
        problem, problem_files, hash_to_content
    )
    num_repair_input_tokens = count_tokens(system_prompt + user_prompt)
    print(f"The repair prompt has {num_repair_input_tokens} tokens")
    response = call_gpt(system_prompt, user_prompt)
    num_repair_output_tokens = count_tokens(response)
    print(f"The repair output has {num_repair_output_tokens} output tokens")
    print(response)

    print(f"The input has {num_ranking_input_tokens + num_repair_input_tokens} tokens")
    print(
        f"the output has {num_ranking_output_tokens + num_repair_output_tokens} tokens"
    )
    print(
        f"The preprocessing has {num_ranking_input_tokens + num_ranking_output_tokens} tokens"
    )
    print(
        f"The repair has {num_repair_input_tokens + num_repair_output_tokens} tokens"
    )
    # TODO: get patch in correct format and return it
