from typing import List

from agent.transformations import merge_imports, remove_imports, remove_blank_lines


IMPORT_MERGE_TRANSFORMATION_CONST = "imports_merge"
IMPORT_REMOVE_TRANSFORMATION_CONST = "imports_remove"
BLANK_LINE_TRANSFORMATION_CONST = "blank_lines"


def minify(source_files: List[str], transformations: List[str]):
    source_maps = {}

    # execute
    if IMPORT_MERGE_TRANSFORMATION_CONST in transformations:
        source_files = merge_imports(source_files)
        transformations.remove(IMPORT_MERGE_TRANSFORMATION_CONST)
    if IMPORT_REMOVE_TRANSFORMATION_CONST in transformations:
        source_files = remove_imports(source_files)
        transformations.remove(IMPORT_REMOVE_TRANSFORMATION_CONST)
    if BLANK_LINE_TRANSFORMATION_CONST in transformations:
        source_files = remove_blank_lines(source_files)
        transformations.remove(BLANK_LINE_TRANSFORMATION_CONST)

    if transformations:
        raise NotImplementedError(
            f"Missing implementation for the following transformations: {transformations}"
        )

    return source_files, source_maps
