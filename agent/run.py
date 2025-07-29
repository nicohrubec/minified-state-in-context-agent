from agent.prompt import build_repair_prompt
from agent.llm import call_gpt


def run_agent(problem, problem_files, hash_to_content):
    print(f"Running agent for problem {problem['instance_id']}")
    system_prompt, user_prompt = build_repair_prompt(
        problem, problem_files, hash_to_content
    )
    print(user_prompt)
    response = call_gpt(system_prompt, user_prompt)
    print(response)
    # TODO: apply patch
