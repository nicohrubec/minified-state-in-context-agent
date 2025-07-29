def build_repair_prompt(problem, problem_files, hash_to_content):
    problem_statement = problem["problem_statement"]

    # collect source files
    all_sources = []
    for file in problem_files["files"]:
        path, content_hash = file["file_path"], file["content_hash"]
        content = hash_to_content[content_hash]
        all_sources.append(f"### {path}\n{content}")
    all_sources_str = "\n\n".join(all_sources)

    prompt = f"""You are a senior software engineer tasked with analyzing and resolving a repository issue. You have been provided with the complete repository structure and the specific issue description.

# REPOSITORY STRUCTURE:
--------------------
{all_sources_str}
--------------------

###########################
# ISSUE DESCRIPTION:
-----------------
{problem_statement}

ANALYSIS INSTRUCTIONS:
--------------------
Your task is to perform the following steps in order:

1. **Chain‑of‑Thought for Localization**
- Analyze the provided repository structure and issue description to identify the relevant code sections.
- Explain your reasoning and process for localizing the relevant code.

2. **Restated Relevant Code**
- Provide the exact code snippet that you have identified as relevant to the issue.
- Include a few lines of context before and after the critical section.
- **IMPORTANT:** Enclose this section in a code block using the tag **relevant code**
  (do not use any markdown language tags like “python”).
- If necessary, copy a longer context from the file to ensure that the location where the revision is needed is fully included, even if only a part of the code will be modified.
- If you would like to add the line ’ print(x)’, you must fully write that out, with all those spaces before the code! Please literally copy the code from the file.

The format should be as follows:

‘‘‘relevant code
### path/to/file.py
[Exact code snippet with proper indentation, including sufficient context]
‘‘‘

3. **Chain‑of‑Thought for Repairing the Code**
- Explain your reasoning and analysis for repairing the identified issue.
- Describe the necessary modifications, why they are needed, and include any edge‑case considerations.

4. **Final Patch**
- Provide the final patch using the following exact *SEARCH/REPLACE* format:
  1. The file path.
  2. The exact lines to search for (i.e. the buggy region).
  3. The replacement lines (i.e. the fixed code).

Ensure that your chain‑of‑thought reasoning is clearly separated from the final patch. Do not include any evaluation or commentary beyond the four requested steps.
"""
    return prompt
