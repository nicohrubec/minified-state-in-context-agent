from typing import List, Dict

from agent.helpers import is_blank_line
from agent.source_map import SourceMapEntry


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
