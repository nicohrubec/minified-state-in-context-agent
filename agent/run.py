import re
from pathlib import Path
import subprocess
import tokenize
from typing import List, Tuple, Optional
import io

from agent.prompt import build_file_ranking_prompt, build_repair_prompt
from agent.llm import call_gpt
from shared.tokens import count_tokens
from agent.transformations import reverse_shortening
from agent.minify import (
    SHORT_VARS_TRANSFORMATION_CONST,
    SHORT_FUNCS_TRANSFORMATION_CONST,
    SHORT_CLASSES_TRANSFORMATION_CONST,
    SHORT_VARS_MAP_TRANSFORMATION_CONST,
    SHORT_FUNCS_MAP_TRANSFORMATION_CONST,
    SHORT_CLASSES_MAP_TRANSFORMATION_CONST,
    SHORT_VARS_MAP_WITH_MAP_TRANSFORMATION_CONST,
    SHORT_FUNCS_MAP_WITH_MAP_TRANSFORMATION_CONST,
    SHORT_CLASSES_MAP_WITH_MAP_TRANSFORMATION_CONST,
)

MAX_ATTEMPTS = 5


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


IGNORABLE_TOKEN_TYPES = {
    tokenize.NL,
    tokenize.NEWLINE,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.COMMENT,
    tokenize.ENCODING,
}


class MatchError(Exception):
    pass


def extract_paths_from_response(response):
    code_block_pattern = r"```(?:[\w]*\n)?(.*?)```"
    code_block_match = re.search(code_block_pattern, response, re.DOTALL)

    if code_block_match:
        code_block_content = code_block_match.group(1).strip()
        return [
            line.strip() for line in code_block_content.splitlines() if line.strip()
        ]
    else:
        return [line.strip() for line in response.splitlines() if line.strip()]


def _line_starts(text: str) -> List[int]:
    # Absolute index of each line start
    starts = [0]
    acc = 0
    for line in text.splitlines(keepends=True):
        acc += len(line)
        starts.append(acc)
    return starts


def _abs_index(line_starts: List[int], line: int, col: int) -> int:
    # tokenize lines are 1-based
    return line_starts[line - 1] + col


def _tokenize_with_positions(code: str) -> List[tokenize.TokenInfo]:
    # Generate all tokens; allow partial tokenization
    return list(tokenize.generate_tokens(io.StringIO(code).readline))


def _likely_docstring_positions(tokens: List[tokenize.TokenInfo]) -> set:
    """
    Heuristic: a STRING token is considered a docstring using pyminifier logic:
    - If previous token is INDENT, it's definitely a docstring
    - If previous token is NL and next token is NEWLINE, it's a module-level docstring
    We avoid AST to stay resilient to malformed code.
    """
    doc_indices = set()
    prev_tok_type = None
    for i, t in enumerate(tokens):
        if t.type == tokenize.STRING:
            if prev_tok_type == tokenize.INDENT:
                # Definitely a docstring
                doc_indices.add(i)
            elif prev_tok_type == tokenize.NL:
                # This captures whole-module docstrings:
                if i + 1 < len(tokens) and tokens[i + 1].type == tokenize.NEWLINE:
                    doc_indices.add(i)
        prev_tok_type = t.type
    return doc_indices


def _normalize(code: str):
    """
    Produce:
      - tokens_keep: list of strings we compare on (no whitespace/comments/docstrings)
      - spans: list of (abs_start, abs_end) in original code for each token_keep element
      - joined: concatenated comparable string (no separators) for fast substring search
      - char_map: for each character index in `joined`, the absolute index in `code`
    We preserve STRING contents; we strip comments; we drop likely docstrings.
    """
    tokens = _tokenize_with_positions(code)
    doc_idx = _likely_docstring_positions(tokens)
    line_starts = _line_starts(code)

    tokens_keep: List[str] = []
    spans: List[Tuple[int, int]] = []

    for idx, t in enumerate(tokens):
        if t.type in IGNORABLE_TOKEN_TYPES:
            continue
        if t.type == tokenize.STRING and idx in doc_idx:
            # drop likely docstring
            continue
        # Keep significant tokens exactly as in source
        s_line, s_col = t.start
        e_line, e_col = t.end
        abs_s = _abs_index(line_starts, s_line, s_col)
        abs_e = _abs_index(line_starts, e_line, e_col)
        lexeme = code[abs_s:abs_e]
        tokens_keep.append(lexeme)
        spans.append((abs_s, abs_e))

    # Join without separators: operator-space insensitivity + blank-line insensitivity
    joined_parts = []
    char_map: List[int] = []
    for lex, (abs_s, _abs_e) in zip(tokens_keep, spans):
        joined_parts.append(lex)
        # map each char of lex back to original absolute index
        for k in range(len(lex)):
            char_map.append(abs_s + k)
    joined = "".join(joined_parts)

    return tokens_keep, spans, joined, char_map


def _find_in_joined(hay_joined: str, needle_joined: str) -> int:
    return hay_joined.find(needle_joined)


def _spans_for_join_index(
    char_map: List[int], start_idx: int, length: int
) -> Tuple[int, int]:
    # Map [start_idx, start_idx+length) in joined back to absolute [abs_s, abs_e)
    abs_s = char_map[start_idx]
    abs_e = char_map[start_idx + length - 1] + 1
    return abs_s, abs_e


def _fallback_normalize(code: str) -> Tuple[str, List[int]]:
    """
    Extremely robust fallback: remove comments and triple-quoted strings,
    drop all whitespace *outside* of quoted strings approximately,
    and return joined text and approximate char map.
    Note: this is a last resort and may over/under-match. Prefer the tokenizer path.
    """
    # Remove comments
    code_nc = re.sub(r"[ \t]*#[^\n]*", "", code)
    # Remove triple-quoted strings (docstrings) conservatively
    code_nc = re.sub(r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')', "", code_nc)

    # Keep quoted strings intact; strip whitespace elsewhere
    out = []
    cmap = []
    i = 0
    n = len(code_nc)
    while i < n:
        c = code_nc[i]
        if c in ('"', "'"):
            # string literal; handle escapes and triple quotes
            quote = c
            # check for triple
            triple = False
            if i + 2 < n and code_nc[i : i + 3] == quote * 3:
                triple = True
                q = quote * 3
                j = i + 3
                while j < n and code_nc[j : j + 3] != q:
                    # handle simple escapes
                    if code_nc[j] == "\\" and j + 1 < n:
                        j += 2
                    else:
                        j += 1
                j = min(n, j + 3)  # include closing quotes if present
            else:
                j = i + 1
                while j < n and code_nc[j] != quote:
                    if code_nc[j] == "\\" and j + 1 < n:
                        j += 2
                    else:
                        j += 1
                j = min(n, j + 1)
            out.append(code_nc[i:j])
            cmap.extend(range(i, j))
            i = j
            continue
        # outside strings: drop whitespace
        if c.isspace():
            i += 1
            continue
        out.append(c)
        cmap.append(i)
        i += 1
    return "".join(out), cmap


def search_replace_robust(
    original_content: str,
    search_block: str,
    replace_block: str,
) -> Optional[str]:
    """
    Robust SEARCH->REPLACE that is insensitive to blank lines, comments,
    likely docstrings, and operator whitespace, while preserving strings.
    Returns modified content, or None if not found / ambiguous.
    """
    search_block = search_block.strip()
    replace_block = replace_block.strip()

    # Fast path: exact match
    idx = original_content.find(search_block)
    if idx != -1:
        return (
            original_content[:idx]
            + replace_block
            + original_content[idx + len(search_block) :]
        )

    # Token-based normalization
    try:
        _, hay_spans, hay_joined, hay_cmap = _normalize(original_content)
        _, _, needle_joined, _ = _normalize(search_block)
        pos = _find_in_joined(hay_joined, needle_joined)
        if pos != -1:
            abs_s, abs_e = _spans_for_join_index(hay_cmap, pos, len(needle_joined))
            return original_content[:abs_s] + replace_block + original_content[abs_e:]
    except (tokenize.TokenError, IndentationError):
        pass  # fall through to fallback

    # Fallback: ultra-robust, approximate normalization
    try:
        hay_joined_fb, hay_cmap_fb = _fallback_normalize(original_content)
        needle_joined_fb, _ = _fallback_normalize(search_block)
        pos = hay_joined_fb.find(needle_joined_fb)
        if pos != -1:
            abs_s = hay_cmap_fb[pos]
            abs_e = hay_cmap_fb[pos + len(needle_joined_fb) - 1] + 1
            return original_content[:abs_s] + replace_block + original_content[abs_e:]
    except Exception:
        pass

    return None


def _extract_blocks_from_final_patch(response_text: str):
    m = re.search(
        r"(?is)^\s*(?:\d+\.\s*)?\*{0,2}Final\s+Patch\*{0,2}\b.*?$",
        response_text,
        re.MULTILINE,
    )
    if not m:
        print("Final Patch section not found!")
        return []

    tail = response_text[m.end() :]

    # collect all fenced python blocks after Final Patch
    code_blocks = re.findall(
        r"```[ \t]*python[ \t]*\n(.*?)\n```", tail, flags=re.IGNORECASE | re.DOTALL
    )
    if not code_blocks:
        print("No ```python``` patch blocks found after Final Patch.")
        return []

    blocks = []
    for block in code_blocks:
        lines = block.splitlines()
        # first non-empty line must be the '### path/to/file' heading
        path_line = next((ln for ln in lines if ln.strip()), None)
        if not path_line:
            print("Patch block missing path line; skipping.")
            continue

        file_path = re.sub(r"^#+\s*", "", path_line).strip()

        # extract SEARCH/REPLACE pairs
        edit_pattern = re.compile(
            r"<<{2,}\s*SEARCH\s*\n(.*?)\n=+\n(.*?)\n>+.*REPLACE",
            re.DOTALL | re.IGNORECASE,
        )
        edits = edit_pattern.findall(block)
        if not edits:
            print(f"No SEARCH/REPLACE markers found for file '{file_path}'; skipping.")
            continue

        edits = [(s.strip(), r.strip()) for (s, r) in edits]
        blocks.append({"file_path": file_path, "edits": edits})

    return blocks


def reverse_shortening_in_blocks(blocks, source_maps):
    new_blocks = []
    for block in blocks:
        new_block = {
            "file_path": block["file_path"],
            "edits": [],
        }
        for edit in block["edits"]:
            search_block, replace_block = edit
            reversed_search = reverse_shortening(search_block, source_maps)
            reversed_replace = reverse_shortening(replace_block, source_maps)
            new_block["edits"].append((reversed_search, reversed_replace))
        new_blocks.append(new_block)

    return new_blocks


def extract_final_patches_as_diff(response_text, repo_dir, source_maps=None):
    repo_path = Path(repo_dir)

    blocks = _extract_blocks_from_final_patch(response_text)
    if not blocks:
        return None

    # reverse shortening transformations
    if source_maps and SHORT_VARS_TRANSFORMATION_CONST in source_maps:
        print("Reversing variable shortening transformations...")
        blocks = reverse_shortening_in_blocks(
            blocks, source_maps[SHORT_VARS_TRANSFORMATION_CONST]
        )
    if source_maps and SHORT_FUNCS_TRANSFORMATION_CONST in source_maps:
        print("Reversing function shortening transformations...")
        blocks = reverse_shortening_in_blocks(
            blocks, source_maps[SHORT_FUNCS_TRANSFORMATION_CONST]
        )
    if source_maps and SHORT_CLASSES_TRANSFORMATION_CONST in source_maps:
        print("Reversing class shortening transformations...")
        blocks = reverse_shortening_in_blocks(
            blocks, source_maps[SHORT_CLASSES_TRANSFORMATION_CONST]
        )
    if source_maps and SHORT_VARS_MAP_TRANSFORMATION_CONST in source_maps:
        print("Reversing variable shortening map transformations...")
        blocks = reverse_shortening_in_blocks(
            blocks, source_maps[SHORT_VARS_MAP_TRANSFORMATION_CONST]
        )
    if source_maps and SHORT_FUNCS_MAP_TRANSFORMATION_CONST in source_maps:
        print("Reversing function shortening map transformations...")
        blocks = reverse_shortening_in_blocks(
            blocks, source_maps[SHORT_FUNCS_MAP_TRANSFORMATION_CONST]
        )
    if source_maps and SHORT_CLASSES_MAP_TRANSFORMATION_CONST in source_maps:
        print("Reversing class shortening map transformations...")
        blocks = reverse_shortening_in_blocks(
            blocks, source_maps[SHORT_CLASSES_MAP_TRANSFORMATION_CONST]
        )
    if source_maps and SHORT_VARS_MAP_WITH_MAP_TRANSFORMATION_CONST in source_maps:
        print("Reversing variable shortening map transformations...")
        blocks = reverse_shortening_in_blocks(
            blocks, source_maps[SHORT_VARS_MAP_WITH_MAP_TRANSFORMATION_CONST]
        )
    if source_maps and SHORT_FUNCS_MAP_WITH_MAP_TRANSFORMATION_CONST in source_maps:
        print("Reversing function shortening map transformations...")
        blocks = reverse_shortening_in_blocks(
            blocks, source_maps[SHORT_FUNCS_MAP_WITH_MAP_TRANSFORMATION_CONST]
        )
    if source_maps and SHORT_CLASSES_MAP_WITH_MAP_TRANSFORMATION_CONST in source_maps:
        print("Reversing class shortening map transformations...")
        blocks = reverse_shortening_in_blocks(
            blocks, source_maps[SHORT_CLASSES_MAP_WITH_MAP_TRANSFORMATION_CONST]
        )

    originals = {}
    modified = {}
    targets = []

    for blk in blocks:
        rel_path = blk["file_path"]
        if not rel_path:
            print("Empty file path in a patch block; skipping.")
            continue

        target = repo_path / rel_path
        if not target.exists():
            print(f"Target file does not exist: {rel_path}")
            continue

        try:
            content = target.read_text()
        except IsADirectoryError:
            print(f"Target is a directory, not a file: {rel_path}")
            continue

        originals[rel_path] = content
        modified[rel_path] = content
        targets.append((rel_path, blk["edits"]))

    if not targets:
        print("No valid target files to modify.")
        return None

    num_applied_total = 0
    for rel_path, edits in targets:
        current = modified[rel_path]
        for search_block, replace_block in edits:
            new_content = search_replace_robust(current, search_block, replace_block)
            if new_content is None:
                print(f"[{rel_path}] Search block not found; skipping this edit.")
                continue
            current = new_content
            num_applied_total += 1

        modified[rel_path] = current

    if num_applied_total <= 0:
        print("No edits were applied to any file.")
        return None

    written_paths = []
    try:
        for rel_path in modified:
            target = repo_path / rel_path
            target.write_text(modified[rel_path])
            written_paths.append(rel_path)

        try:
            cmd = ["git", "diff", "--"] + [str(Path(p)) for p in written_paths]
            result = subprocess.run(
                cmd, cwd=repo_path, capture_output=True, text=True, check=True
            )
            diff_output = result.stdout
        except subprocess.CalledProcessError as e:
            print(f"git diff failed: {e}")
            diff_output = None
    finally:
        for rel_path, content in originals.items():
            (repo_path / rel_path).write_text(content)

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
    model="gpt-4.1",
    token_limit=100000,
    temperature=0.8,
):
    problem_files["files"] = [
        file
        for file in problem_files["files"]
        if file["file_type"] in ["core", "config"]
    ]
    patch_text = problem["patch"]
    instance_id = problem["instance_id"]

    repo_dir = Path(repo_base_dir) / problem["repo"].split("/")[-1]

    try:
        # ensure we're in a clean state
        subprocess.run(
            ["git", "reset", "--hard", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            check=True,
        )

        # checkout the target commit
        subprocess.run(
            ["git", "checkout", problem["base_commit"]],
            cwd=repo_dir,
            capture_output=True,
            check=True,
        )
        print(
            f"Successfully checked out commit {problem['base_commit']} for {problem['repo']}"
        )
    except subprocess.CalledProcessError as e:
        print(f"Failed to checkout commit {problem['base_commit']}: {e}")

    # preprocessing step
    # 1. file ranking
    prompt, replacements = build_file_ranking_prompt(
        problem, problem_files, rank_encoding
    )
    num_ranking_input_tokens = count_tokens(prompt)
    response = call_gpt("", prompt, model, temperature=temperature)
    num_ranking_output_tokens = count_tokens(response)

    # 2. filter based on ranking
    ranked_paths = extract_paths_from_response(response)
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

    attempt = 1
    while attempt <= MAX_ATTEMPTS:
        response = call_gpt(system_prompt, user_prompt, model, temperature=temperature)
        num_repair_output_tokens = count_tokens(response)

        chain_of_thoughts = extract_cot_sections(response)
        chain_of_thoughts["instance_id"] = instance_id

        patch = extract_final_patches_as_diff(response, repo_dir, source_maps)

        if patch is not None:
            break

        print(f"Attempt {attempt} of obtaining a patch failed. Retry ...")
        attempt += 1

    if attempt > MAX_ATTEMPTS:
        print("Max number of attempts reached without obtaining a valid patch.")

    prediction = (
        {
            "instance_id": problem["instance_id"],
            "model_name_or_path": model,
            "model_patch": patch,
        }
        if patch
        else {
            "instance_id": problem["instance_id"],
            "model_name_or_path": model,
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
