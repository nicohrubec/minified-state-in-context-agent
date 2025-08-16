from typing import List

from agent.transformations import (
    merge_imports,
    remove_imports,
    remove_blank_lines,
    remove_comments,
)


IMPORT_MERGE_TRANSFORMATION_CONST = "imports_merge"
IMPORT_REMOVE_TRANSFORMATION_CONST = "imports_remove"
BLANK_LINE_TRANSFORMATION_CONST = "blank_lines"
COMMENTS_REMOVE_TRANSFORMATION_CONST = "comments_remove"
DEFINED_TRANSFORMATIONS = [
    IMPORT_REMOVE_TRANSFORMATION_CONST,
    IMPORT_REMOVE_TRANSFORMATION_CONST,
    BLANK_LINE_TRANSFORMATION_CONST,
    COMMENTS_REMOVE_TRANSFORMATION_CONST,
]


def minify(source_files: List[str], transformations: List[str]):
    source_maps = {}

    if IMPORT_MERGE_TRANSFORMATION_CONST in transformations:
        source_files = merge_imports(source_files)
    if IMPORT_REMOVE_TRANSFORMATION_CONST in transformations:
        source_files = remove_imports(source_files)
    if BLANK_LINE_TRANSFORMATION_CONST in transformations:
        source_files = remove_blank_lines(source_files)
    if COMMENTS_REMOVE_TRANSFORMATION_CONST in transformations:
        source_files = remove_comments(source_files)

    unknown_transformations = [
        t for t in transformations if t not in DEFINED_TRANSFORMATIONS
    ]
    if unknown_transformations:
        raise NotImplementedError(
            f"Missing implementation for the following transformations: {unknown_transformations}"
        )

    return source_files, source_maps
