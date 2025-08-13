from typing import List


IMPORT_TRANSFORMATION_CONST = "imports"


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


def minify(source_files: List[str], transformations: List[str]):
    source_maps = {}

    if IMPORT_TRANSFORMATION_CONST in transformations:
        source_files = merge_imports(source_files)
        transformations.remove(IMPORT_TRANSFORMATION_CONST)

    if transformations:
        raise NotImplementedError(
            f"Missing implementation for the following transformations: {transformations}"
        )

    return source_files, source_maps
