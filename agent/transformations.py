from typing import List, Tuple, Dict
import tokenize
import re

from agent.helpers import is_import_line
import pyminifier.minification as mini
import pyminifier.token_utils as token_utils
import pyminifier.obfuscate as obfuscate
from shared.tokens import count_tokens


def remove_imports(source_files: List[str]):
    new_sources = []

    for src in source_files:
        lines = src.splitlines()
        if not lines:
            continue

        path = lines[0]
        body_lines = []

        for line in lines[1:]:
            if is_import_line(line):
                continue

            body_lines.append(line)

        new_sources.append("\n".join([path] + body_lines))

    return new_sources


def merge_imports(source_files: List[str]):
    merged_imports = []
    seen = set()
    new_sources = []

    for src in source_files:
        lines = src.splitlines()
        if not lines:
            continue

        path = lines[0]
        body_lines = []

        for line in lines[1:]:
            if is_import_line(line):
                if line not in seen:
                    seen.add(line)
                    merged_imports.append(line)
            else:
                body_lines.append(line)

        new_sources.append("\n".join([path] + body_lines))

    if not merged_imports:
        return source_files

    merged_block = "\n".join(merged_imports)
    merged_file = "\n".join(["MERGED_IMPORTS", merged_block])
    return [merged_file] + new_sources


def remove_blank_lines(source_files: List[str]):
    source_paths = [src.splitlines()[0] for src in source_files]
    source_bodies = ["\n".join(src.splitlines()[1:]) for src in source_files]
    source_bodies_no_blank_lines = [
        mini.remove_blank_lines(src_body) for src_body in source_bodies
    ]
    new_sources: List[str] = [
        path + "\n" + body
        for path, body in zip(source_paths, source_bodies_no_blank_lines)
    ]
    return new_sources


def remove_comments(source_files: List[str]):
    new_sources: List[str] = []

    for src in source_files:
        lines = src.splitlines()
        if not lines:
            continue

        path = lines[0]
        body = "\n".join(lines[1:])

        try:
            tokenized_body = token_utils.listified_tokenizer(body)
            mini.remove_comments(tokenized_body)
            new_sources.append(path + "\n" + token_utils.untokenize(tokenized_body))
        except (tokenize.TokenError, IndentationError):
            new_sources.append(path + "\n" + body)

    return new_sources


def remove_docstrings(source_files: List[str]):
    new_sources: List[str] = []

    for src in source_files:
        lines = src.splitlines()
        if not lines:
            continue

        path = lines[0]
        body = "\n".join(lines[1:])

        try:
            tokenized_body = token_utils.listified_tokenizer(body)
            mini.remove_docstrings(tokenized_body)
            new_sources.append(path + "\n" + token_utils.untokenize(tokenized_body))
        except (tokenize.TokenError, IndentationError):
            new_sources.append(path + "\n" + body)

    return new_sources


def dedent(source_files: List[str]):
    source_paths = [src.splitlines()[0] for src in source_files]
    source_bodies = ["\n".join(src.splitlines()[1:]) for src in source_files]

    new_source_bodies = []
    for src_body in source_bodies:
        try:
            src_body_dedent = mini.dedent(src_body)
            new_source_bodies.append(src_body_dedent)
        except (tokenize.TokenError, IndentationError):
            new_source_bodies.append(src_body)

    new_sources: List[str] = [
        path + "\n" + body for path, body in zip(source_paths, new_source_bodies)
    ]
    return new_sources


def reduce_operators(source_files: List[str]):
    source_paths = [src.splitlines()[0] for src in source_files]
    source_bodies = ["\n".join(src.splitlines()[1:]) for src in source_files]

    new_source_bodies = []
    for src_body in source_bodies:
        try:
            src_body_with_reduced_operators = mini.reduce_operators(src_body)
            new_source_bodies.append(src_body_with_reduced_operators)
        except (tokenize.TokenError, IndentationError):
            new_source_bodies.append(src_body)

    new_sources: List[str] = [
        path + "\n" + body for path, body in zip(source_paths, new_source_bodies)
    ]
    return new_sources


def collect_existing_names(source_files: List[str]):
    existing_names = set()

    for src in source_files:
        lines = src.splitlines()
        if not lines:
            continue

        body = "\n".join(lines[1:])

        try:
            tokenized_body = token_utils.listified_tokenizer(body)

            for token in tokenized_body:
                token_type = token[0]
                token_string = token[1]

                if token_type == tokenize.NAME:
                    existing_names.add(token_string)

        except (tokenize.TokenError, IndentationError):
            pass

    return existing_names


def create_conflict_avoiding_generator(
    existing_names, use_unicode=False, identifier_length=1
):
    original_generator = obfuscate.obfuscation_machine(
        use_unicode=use_unicode, identifier_length=identifier_length
    )

    def conflict_avoiding_generator():
        for name in original_generator:
            if name not in existing_names:
                yield name

    return conflict_avoiding_generator()


def shorten(source_files: List[str], find_obfunc, replace_obfunc):
    existing_names = collect_existing_names(source_files)
    safe_name_generator = create_conflict_avoiding_generator(
        existing_names, use_unicode=False, identifier_length=2
    )

    new_sources: List[str] = []

    for src in source_files:
        lines = src.splitlines()
        if not lines:
            continue

        path = lines[0]
        body = "\n".join(lines[1:])

        try:
            tokenized_body = token_utils.listified_tokenizer(body)

            obfuscatables = obfuscate.find_obfuscatables(
                tokenized_body, find_obfunc, ignore_length=False
            )
            obfuscatables = [obfuscatable for obfuscatable in obfuscatables if count_tokens(obfuscatable) > 1]

            for obfuscatable in obfuscatables:
                obfuscate.replace_obfuscatables(
                    path,
                    tokenized_body,
                    replace_obfunc,
                    obfuscatable,
                    safe_name_generator,
                )

            new_sources.append(path + "\n" + token_utils.untokenize(tokenized_body))
        except (tokenize.TokenError, IndentationError):
            new_sources.append(path + "\n" + body)

    return new_sources


def shorten_with_source_map(
    source_files: List[str], find_obfunc
) -> Tuple[List[str], Dict[str, str]]:
    existing_names = collect_existing_names(source_files)
    safe_name_generator = create_conflict_avoiding_generator(
        existing_names, use_unicode=False, identifier_length=2
    )

    new_sources: List[str] = []
    source_map: Dict[str, str] = {}

    for src in source_files:
        lines = src.splitlines()
        if not lines:
            continue

        path = lines[0]
        body = "\n".join(lines[1:])

        try:
            tokenized_body = token_utils.listified_tokenizer(body)

            obfuscatables = obfuscate.find_obfuscatables(
                tokenized_body, find_obfunc, ignore_length=False
            )

            # analyze each obfuscatable to determine if replacement is worth it
            for obfuscatable in obfuscatables:
                if obfuscatable is None:
                    continue

                # count the occurrences of this obfuscatable in the current file
                occurrence_count = 0
                for token in tokenized_body:
                    if token[0] == tokenize.NAME and token[1] == obfuscatable:
                        occurrence_count += 1

                if occurrence_count > 1 and count_tokens(obfuscatable) > 2:
                    shortened_name = next(safe_name_generator)

                    for i, token in enumerate(tokenized_body):
                        if token[0] == tokenize.NAME and token[1] == obfuscatable:
                            tokenized_body[i] = (
                                token[0],
                                shortened_name,
                                token[2],
                                token[3],
                                token[4],
                            )

                    source_map[shortened_name] = obfuscatable

            new_sources.append(path + "\n" + token_utils.untokenize(tokenized_body))
        except (tokenize.TokenError, IndentationError):
            new_sources.append(path + "\n" + body)

    return new_sources, source_map


def reverse_shortening(source_file: str, replacements: dict):
    try:
        lines = source_file.splitlines()
        result_lines = []

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Check if this line starts a multiline string
            if '"""' in stripped or "'''" in stripped:
                # Find the delimiter
                delimiter = '"""' if '"""' in stripped else "'''"
                start_count = stripped.count(delimiter)

                if start_count >= 2:
                    # Single-line docstring - keep unchanged
                    result_lines.append(line)
                else:
                    # Multiline docstring - collect all lines until closing delimiter
                    result_lines.append(line)
                    i += 1
                    while i < len(lines) and delimiter not in lines[i]:
                        result_lines.append(lines[i])
                        i += 1
                    if i < len(lines):
                        result_lines.append(lines[i])
            else:
                # Regular line - check for inline comments and apply replacements
                comment_pos = line.find("#")
                if comment_pos != -1:
                    # Split line into code and comment parts
                    code_part = line[:comment_pos]
                    comment_part = line[comment_pos:]

                    # Apply replacements only to code part
                    modified_code = code_part
                    for shortened_name, original_name in replacements.items():
                        pattern = r"\b" + re.escape(shortened_name) + r"\b"
                        modified_code = re.sub(pattern, original_name, modified_code)

                    # Reconstruct line with modified code and unchanged comment
                    result_lines.append(modified_code + comment_part)
                else:
                    # No comment - apply replacements to entire line
                    modified_line = line
                    for shortened_name, original_name in replacements.items():
                        pattern = r"\b" + re.escape(shortened_name) + r"\b"
                        modified_line = re.sub(pattern, original_name, modified_line)
                    result_lines.append(modified_line)

            i += 1

        return "\n".join(result_lines)

    except Exception as e:
        print(f"Error in reverse_shortening: {e}")
        return source_file
