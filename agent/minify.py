from typing import List

from agent.transformations import (
    merge_imports,
    remove_imports,
    remove_blank_lines,
    remove_comments,
    remove_docstrings,
)


IMPORT_MERGE_TRANSFORMATION_CONST = "merge_imports"
IMPORT_REMOVE_TRANSFORMATION_CONST = "remove_imports"
BLANK_LINE_TRANSFORMATION_CONST = "remove_blank_lines"
COMMENTS_REMOVE_TRANSFORMATION_CONST = "remove_comments"
DOCSTRINGS_REMOVE_TRANSFORMATION_CONST = "remove_docstrings"
DEFINED_TRANSFORMATIONS = [
    IMPORT_REMOVE_TRANSFORMATION_CONST,
    IMPORT_REMOVE_TRANSFORMATION_CONST,
    BLANK_LINE_TRANSFORMATION_CONST,
    COMMENTS_REMOVE_TRANSFORMATION_CONST,
    DOCSTRINGS_REMOVE_TRANSFORMATION_CONST,
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
    if DOCSTRINGS_REMOVE_TRANSFORMATION_CONST in transformations:
        source_files = remove_docstrings(source_files)

    unknown_transformations = [
        t for t in transformations if t not in DEFINED_TRANSFORMATIONS
    ]
    if unknown_transformations:
        raise NotImplementedError(
            f"Missing implementation for the following transformations: {unknown_transformations}"
        )

    return source_files, source_maps
