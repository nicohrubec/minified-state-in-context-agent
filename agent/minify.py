from typing import List, Dict
from dataclasses import dataclass


IMPORT_TRANSFORMATION_CONST = "imports"
BLANK_LINE_TRANSFORMATION_CONST = "blank_lines"


@dataclass
class SourceMapEntry:
    original: str
    transformed: str
    line_no: int
    offset: int
    type: str


def is_blank_line(s: str) -> bool:
    # Treat whitespace-only lines as blank
    return s.strip() == ""


def plan_remove_blank_lines(source_files: List[str]) -> Dict[str, List[SourceMapEntry]]:
    plan: Dict[str, List[SourceMapEntry]] = {}

    for src in source_files:
        lines = src.splitlines()
        if not lines:
            # Skip malformed entry
            continue

        path = lines[0]
        body = lines[1:]
        removals: List[SourceMapEntry] = []

        for i, line in enumerate(body, start=1):
            if is_blank_line(line):
                removals.append(
                    SourceMapEntry(
                        original=line,
                        transformed="",
                        line_no=i,
                        offset=0,
                        type="blank_line",
                    )
                )

        if removals:
            plan[path] = removals

    return plan


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
            if line.startswith("import ") or line.startswith("from "):
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


def remove_blank_lines(source_files: List[str], plan: Dict[str, List[SourceMapEntry]]):
    new_sources: List[str] = []

    for src in source_files:
        lines = src.splitlines()
        if not lines:
            continue

        path = lines[0]
        body = lines[1:]

        if path not in plan:
            # No changes required
            new_sources.append("\n".join([path] + body))
            continue

        removals = plan[path]
        to_remove = {e.line_no for e in removals}

        kept_body: List[str] = []
        for i, line in enumerate(body, start=1):
            if i in to_remove and is_blank_line(line):
                continue
            kept_body.append(line)

        new_sources.append("\n".join([path] + kept_body))

    return new_sources


def _merge_maps(main_map, merge_map):
    for p, lst in merge_map.items():
        main_map.setdefault(p, []).extend(lst)


def minify(source_files: List[str], transformations: List[str]):
    source_maps = {}

    # planning
    if BLANK_LINE_TRANSFORMATION_CONST in transformations:
        blank_line_maps = plan_remove_blank_lines(source_files)
        _merge_maps(source_maps, blank_line_maps)

    # execute
    if IMPORT_TRANSFORMATION_CONST in transformations:
        source_files = merge_imports(source_files)
        transformations.remove(IMPORT_TRANSFORMATION_CONST)
    if BLANK_LINE_TRANSFORMATION_CONST in transformations:
        source_files = remove_blank_lines(source_files, blank_line_maps)
        transformations.remove(BLANK_LINE_TRANSFORMATION_CONST)

    if transformations:
        raise NotImplementedError(
            f"Missing implementation for the following transformations: {transformations}"
        )

    return source_files, source_maps
