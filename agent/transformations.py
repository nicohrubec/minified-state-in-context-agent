from typing import List
import tokenize
import re

from agent.helpers import is_import_line
import pyminifier.minification as mini
import pyminifier.token_utils as token_utils
import pyminifier.obfuscate as obfuscate


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


def shorten_vars(source_files: List[str]):
    new_sources: List[str] = []
    obfuscate.VAR_REPLACEMENTS.clear()

    name_generator = obfuscate.obfuscation_machine(
        use_unicode=False, identifier_length=1
    )

    for src in source_files:
        lines = src.splitlines()
        if not lines:
            continue

        path = lines[0]
        body = "\n".join(lines[1:])

        try:
            tokenized_body = token_utils.listified_tokenizer(body)

            variables = obfuscate.find_obfuscatables(
                tokenized_body, obfuscate.obfuscatable_variable, ignore_length=False
            )

            for variable in variables:
                obfuscate.replace_obfuscatables(
                    "module",
                    tokenized_body,
                    obfuscate.obfuscate_variable,
                    variable,
                    name_generator,
                )

            new_sources.append(path + "\n" + token_utils.untokenize(tokenized_body))
        except (tokenize.TokenError, IndentationError):
            new_sources.append(path + "\n" + body)

    return new_sources, obfuscate.VAR_REPLACEMENTS.copy()


def reverse_variable_shortening(source_file: str, replacements: dict):
    lines = source_file.splitlines()
    if not lines:
        return source_file

    path = lines[0]
    body = "\n".join(lines[1:])

    reversed_body = body
    for shortened_name, original_name in replacements.items():
        pattern = r"\b" + re.escape(shortened_name) + r"\b"
        reversed_body = re.sub(pattern, original_name, reversed_body)

    return path + "\n" + reversed_body


def shorten_funcs(source_files: List[str]):
    pass


def shorten_classes(source_files: List[str]):
    pass
