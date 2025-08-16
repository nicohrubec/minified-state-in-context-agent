from typing import List, Dict

from agent.helpers import is_import_line, is_blank_line


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
    new_sources: List[str] = []

    for src in source_files:
        lines = src.splitlines()
        if not lines:
            continue

        path = lines[0]
        body = lines[1:]

        kept_body: List[str] = [line for line in body if not is_blank_line(line)]
        new_sources.append("\n".join([path] + kept_body))

    return new_sources
