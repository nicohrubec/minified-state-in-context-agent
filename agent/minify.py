from typing import List

from agent.transformations import (
    merge_imports,
    remove_imports,
    remove_blank_lines,
    remove_comments,
    remove_docstrings,
    dedent,
    reduce_operators,
    shorten,
    shorten_with_source_map,
)
import pyminifier.obfuscate as obfuscate


IMPORT_MERGE_TRANSFORMATION_CONST = "merge_imports"
IMPORT_REMOVE_TRANSFORMATION_CONST = "remove_imports"
BLANK_LINE_TRANSFORMATION_CONST = "remove_blank_lines"
COMMENTS_REMOVE_TRANSFORMATION_CONST = "remove_comments"
DOCSTRINGS_REMOVE_TRANSFORMATION_CONST = "remove_docstrings"
DEDENT_TRANSFORMATION_CONST = "dedent"
REDUCE_OPERATORS_TRANSFORMATION_CONST = "reduce_operators"
SHORT_VARS_TRANSFORMATION_CONST = "short_vars"
SHORT_FUNCS_TRANSFORMATION_CONST = "short_funcs"
SHORT_CLASSES_TRANSFORMATION_CONST = "short_classes"
SHORT_VARS_MAP_TRANSFORMATION_CONST = "short_vars_map"
SHORT_FUNCS_MAP_TRANSFORMATION_CONST = "short_funcs_map"
SHORT_CLASSES_MAP_TRANSFORMATION_CONST = "short_classes_map"
DEFINED_TRANSFORMATIONS = [
    IMPORT_REMOVE_TRANSFORMATION_CONST,
    IMPORT_MERGE_TRANSFORMATION_CONST,
    BLANK_LINE_TRANSFORMATION_CONST,
    COMMENTS_REMOVE_TRANSFORMATION_CONST,
    DOCSTRINGS_REMOVE_TRANSFORMATION_CONST,
    DEDENT_TRANSFORMATION_CONST,
    REDUCE_OPERATORS_TRANSFORMATION_CONST,
    SHORT_VARS_TRANSFORMATION_CONST,
    SHORT_FUNCS_TRANSFORMATION_CONST,
    SHORT_CLASSES_TRANSFORMATION_CONST,
    SHORT_VARS_MAP_TRANSFORMATION_CONST,
    SHORT_FUNCS_MAP_TRANSFORMATION_CONST,
    SHORT_CLASSES_MAP_TRANSFORMATION_CONST,
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
    if DEDENT_TRANSFORMATION_CONST in transformations:
        source_files = dedent(source_files)
    if REDUCE_OPERATORS_TRANSFORMATION_CONST in transformations:
        source_files = reduce_operators(source_files)
    if SHORT_VARS_TRANSFORMATION_CONST in transformations:
        obfuscate.VAR_REPLACEMENTS.clear()
        source_files = shorten(
            source_files, obfuscate.obfuscatable_variable, obfuscate.obfuscate_variable
        )
        source_maps[SHORT_VARS_TRANSFORMATION_CONST] = obfuscate.VAR_REPLACEMENTS.copy()
    if SHORT_FUNCS_TRANSFORMATION_CONST in transformations:
        obfuscate.FUNC_REPLACEMENTS.clear()
        source_files = shorten(
            source_files, obfuscate.obfuscatable_function, obfuscate.obfuscate_function
        )
        source_maps[SHORT_FUNCS_TRANSFORMATION_CONST] = (
            obfuscate.FUNC_REPLACEMENTS.copy()
        )
    if SHORT_CLASSES_TRANSFORMATION_CONST in transformations:
        obfuscate.CLASS_REPLACEMENTS.clear()
        source_files = shorten(
            source_files, obfuscate.obfuscatable_class, obfuscate.obfuscate_class
        )
        source_maps[SHORT_CLASSES_TRANSFORMATION_CONST] = (
            obfuscate.CLASS_REPLACEMENTS.copy()
        )
    if SHORT_VARS_MAP_TRANSFORMATION_CONST in transformations:
        source_files, var_source_map = shorten_with_source_map(
            source_files, obfuscate.obfuscatable_variable
        )
        source_maps[SHORT_VARS_MAP_TRANSFORMATION_CONST] = var_source_map
    if SHORT_FUNCS_MAP_TRANSFORMATION_CONST in transformations:
        source_files, func_source_map = shorten_with_source_map(
            source_files, obfuscate.obfuscatable_function
        )
        source_maps[SHORT_FUNCS_MAP_TRANSFORMATION_CONST] = func_source_map
    if SHORT_CLASSES_MAP_TRANSFORMATION_CONST in transformations:
        source_files, class_source_map = shorten_with_source_map(
            source_files, obfuscate.obfuscatable_class
        )
        source_maps[SHORT_CLASSES_MAP_TRANSFORMATION_CONST] = class_source_map

    unknown_transformations = [
        t for t in transformations if t not in DEFINED_TRANSFORMATIONS
    ]
    if unknown_transformations:
        raise NotImplementedError(
            f"Missing implementation for the following transformations: {unknown_transformations}"
        )

    return source_files, source_maps
