def build_file_ranking_prompt(problem, problem_files):
    problem_statement = problem["problem_statement"]

    # build repo structure
    structure_lines = []
    for file in problem_files["files"]:
        structure_lines.append(file["file_path"])
    structure = "\n".join(structure_lines)

    prompt = f"""Please look through the following GitHub problem description and 
    Repository structure and provide a ranked list of files or 
    subfolders from the most relevant to the least relevant for 
    fixing the problem.

Note that you should focus on providing specific files or the 
    lowest subfolder in the tree. Avoid listing a folder that 
    contains many files; instead, break it down to the most 
    granular and relevant components.

### GitHub Problem Description ###
{problem_statement}

### Repository Structure ###
{structure}

###

Please provide the ranked list with the most relevant item first
    and the least relevant item last.
Always output all files that were given to you in the structure.
    Do not omit any files.
The returned list should be separated by new lines and wrapped
    with```.
    For example:
    ```
    file1 . py
    folder2 / file3 . py
    folder4 / subfolder5 /
    folder6 / file7 . py
    ```
    """

    return prompt


def build_repair_prompt(problem, problem_files, hash_to_content):
    problem_statement = problem["problem_statement"]

    # collect source files
    all_sources = []
    for file in problem_files["files"]:
        path, content_hash = file["file_path"], file["content_hash"]
        content = hash_to_content[content_hash]
        all_sources.append(f"### {path}\n{content}")
    all_sources_str = "\n\n".join(all_sources)

    system_prompt = "You are a senior software engineer tasked with analyzing and resolving a repository issue. You have been provided with the complete repository structure and the specific issue description."
    user_prompt = f"""# REPOSITORY STRUCTURE:
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
- Prefix this with "Chain-of-Thought for Localization\n---------------------------------".
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
- Prefix this with "Chain-of-Thought for Repairing the Code\n---------------------------------------".
- Explain your reasoning and analysis for repairing the identified issue.
- Describe the necessary modifications, why they are needed, and include any edge‑case considerations.

4. **Final Patch**
- Provide the final patch using the following exact *SEARCH/REPLACE* format:
  1. Start with "Final Patch\n-----------" ONCE.
  2. The file path.
  3. Provide patches for the file in the SEARCH/REPLACE edit format. For each edit provide ONLY the following:
    - Start with "<<<<<<< SEARCH"
    - Then provide the exact lines to search for (i.e. the buggy region). 
      It is important that this must match the original source EXACTLY, if even a single letter deviates, everything breaks.
      Keep search regions minimal. Only search for lines that need replacement. For instance, prefer multiple smaller edits over one large edit that regenerates a full function.
    - End the search region with a new line "=======".
    - Then provide the replacement lines (i.e. the fixed code).
    - End the replacement region (and with this also the patch) with ">>>>>>> REPLACE".
    - Do not add the file path for each SEARCH/REPLACE edit.
    - Focus patches on code changes, i.e. a patch should never only consist of a comment edit.

Ensure that your chain‑of‑thought reasoning is clearly separated from the final patch. Do not include any evaluation or commentary beyond the four requested steps.
"""
    return system_prompt, user_prompt, all_sources_str
