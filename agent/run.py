from agent.prompt import build_repair_prompt
from agent.llm import call_gpt
from shared.tokens import count_tokens


def run_agent(problem, problem_files, hash_to_content):
    print(f"Running agent for problem {problem['instance_id']}")
    system_prompt, user_prompt, _ = build_repair_prompt(
        problem, problem_files, hash_to_content
    )
    num_tokens = count_tokens(system_prompt + user_prompt)
    print(f"The prompt has {num_tokens} tokens")
    response = call_gpt(system_prompt, user_prompt)
    print(response)
    # TODO: apply patch
