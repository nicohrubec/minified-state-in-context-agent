from agent.minify import minify
from shared.tokens import count_tokens


def build_list_repo_structure(problem_files):
    structure_lines = []
    for file in problem_files["files"]:
        structure_lines.append(file["file_path"])
    structure = "\n".join(structure_lines)

    return structure


def build_trie(problem_files):
    trie = {}
    for file in problem_files["files"]:
        parts = file["file_path"].split("/")
        node = trie
        for part in parts:
            node = node.setdefault(part, {})
    return trie


def build_trie_repo_structure(problem_files):
    trie = build_trie(problem_files)

    def render(node, depth=0):
        lines = []
        for name in sorted(node.keys()):
            lines.append("  " * depth + name)
            if node[name]:  # has children
                lines.extend(render(node[name], depth + 1))
        return lines

    structure_lines = render(trie)
    structure = "\n".join(structure_lines)

    return structure


def build_int_folder_repo_structure(problem_files):
    structure_lines = []
    folder_counts_and_tokens_used = {}

    for file in problem_files["files"]:
        parts = file["file_path"].split("/")
        for part in parts:
            if part in folder_counts_and_tokens_used:
                folder_counts_and_tokens_used[part]["count"] += 1
            else:
                folder_counts_and_tokens_used[part] = {
                    "count": 1,
                    "tokens": count_tokens(part),
                }

    # check if it is worth it to do a replacement for the folder name
    # if adding an integer mapping entry and then replacing occurrences with the int saves tokens
    # we do that
    structure_lines, replacements = get_replacements_saving_tokens(
        structure_lines, folder_counts_and_tokens_used
    )

    # add instructions for how to use the mapping
    structure_lines.append(
        "Below you can find the full list of file paths for the repository. "
        "If a folder name is an int, you can look up the actual name in the folder mapping above."
    )
    structure_lines.append(
        "For instance, if the listed name is 1/2 and there are entries 1:abc 2:def.py, "
        "then the actual full path would be abc/def.py."
    )

    # build file structure with folder names replaced by ints if they have an entry at the top
    for file in problem_files["files"]:
        parts = file["file_path"].split("/")
        for idx, part in enumerate(parts):
            if part in replacements:
                parts[idx] = str(replacements[part])
        structure_lines.append("/".join(parts))
    structure = "\n".join(structure_lines)

    return structure, replacements


def get_replacements_saving_tokens(structure_lines, counts_and_tokens_used):
    replacement_int = 0
    replacements = {}
    for path in counts_and_tokens_used:
        path_count = counts_and_tokens_used[path]["count"]
        path_tokens_used = counts_and_tokens_used[path]["tokens"]
        total_tokens_used = path_count * path_tokens_used
        replacement_str = f"{replacement_int}:{path}"

        # int encode
        if total_tokens_used > count_tokens(replacement_str):
            structure_lines.append(replacement_str)
            replacements[path] = replacement_int
            replacement_int += 1

    return structure_lines, replacements


def build_int_path_repo_structure(problem_files):
    structure_lines = []
    path_counts_and_tokens_used = {}

    for file in problem_files["files"]:
        current_path = ""
        parts = file["file_path"].split("/")
        for idx, part in enumerate(parts):
            current_path += part

            if current_path in path_counts_and_tokens_used:
                path_counts_and_tokens_used[current_path]["count"] += 1
            else:
                path_counts_and_tokens_used[current_path] = {
                    "count": 1,
                    "tokens": count_tokens(current_path),
                }

            if idx != len(parts) - 1:
                current_path += "/"

    # check if it is worth it to do a replacement for subpaths
    # if adding an integer mapping entry and then replacing occurrences with the int saves tokens
    # we do that
    structure_lines, replacements = get_replacements_saving_tokens(
        structure_lines, path_counts_and_tokens_used
    )

    # add instructions for how to use mapping
    structure_lines.append(
        "Below you can find the full list of file paths for the repository. "
        "If a path name is an int, you can look up the actual path in the mapping above."
    )
    structure_lines.append(
        "For instance, if the listed name is 1/def.py and is an entry 1:abc, "
        "then the actual full path would be abc/def.py."
    )

    # build file structure with folder names replaced by ints if they have an entry at the top
    # we first check deeper paths, because in case that multiple replacements are available for that path
    # the deeper ones should strictly save more tokens
    for file in problem_files["files"]:
        parts = file["file_path"].split("/")
        idx = len(parts) - 1

        while idx > 0:
            candidate = "/".join(parts[: idx + 1])  # join up to and including idx
            if candidate in replacements:
                # replace all contributing entries [0...idx] with a single entry
                parts = [str(replacements[candidate])] + parts[idx + 1 :]
                break
            idx -= 1
        structure_lines.append("/".join(parts))

    structure = "\n".join(structure_lines)

    return structure, replacements


def build_file_ranking_prompt(problem, problem_files, rank_encoding="list"):
    problem_statement = problem["problem_statement"]
    replacements = {}

    match rank_encoding:
        case "list":
            structure = build_list_repo_structure(problem_files)
        case "trie":
            structure = build_trie_repo_structure(problem_files)
        case "int_folder":
            structure, replacements = build_int_folder_repo_structure(problem_files)
        case "int_path":
            structure, replacements = build_int_path_repo_structure(problem_files)
        case _:
            raise NotImplementedError

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
    file1.py
    folder2/file3.py
    folder4/subfolder5/
    folder6/file7.py
    ```
    """

    return prompt, replacements


def build_repair_prompt(problem, problem_files, hash_to_content, transformations):
    problem_statement = problem["problem_statement"]

    # collect source files
    all_sources = []
    for file in problem_files["files"]:
        path, content_hash = file["file_path"], file["content_hash"]
        content = hash_to_content[content_hash]
        all_sources.append(f"### {path}\n{content}")

    all_sources, source_maps = minify(all_sources, transformations)
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
