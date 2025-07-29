from agent.prompt import build_repair_prompt
from agent.llm import call_gpt


def run_agent(problem, problem_files, hash_to_content):
    print(f"Running agent for problem {problem['instance_id']}")
    prompt = build_repair_prompt(problem, problem_files, hash_to_content)
    response = call_gpt(prompt)
    print(response)
    # TODO: apply patch
