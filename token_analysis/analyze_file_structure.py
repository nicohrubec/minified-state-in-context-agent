import ast
import tokenize
import token
import io
from collections import defaultdict


def analyze_lexical_characters(source: str) -> dict:
    lexical_usage = defaultdict(int)
    tokgen = tokenize.generate_tokens(io.StringIO(source).readline)

    for tok_type, tok_string, _, _, _ in tokgen:
        char_count = len(tok_string)
        match tok_type:
            case token.ENCODING:
                lexical_usage["encoding"] += char_count
            case token.ENDMARKER:
                lexical_usage["endmarker"] += char_count
            case token.ERRORTOKEN:
                lexical_usage["error"] += char_count
            case token.STRING:
                lexical_usage["string_literals"] += char_count
            case token.NUMBER:
                lexical_usage["numeric_literals"] += char_count
            case token.COMMENT:
                lexical_usage["comments"] += char_count
            case token.NAME:
                lexical_usage["identifiers"] += char_count
            case token.NEWLINE | token.NL | token.INDENT | token.DEDENT:
                lexical_usage["whitespace"] += char_count
            case token.OP:
                lexical_usage["operators"] += char_count
            case _:
                lexical_usage["other"] += char_count

    return lexical_usage


def analyze_structural_characters(source: str) -> dict:
    structural_usage = defaultdict(int)
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            start = node.lineno
            end = getattr(node, "end_lineno", node.lineno)
            block = "".join(lines[start - 1 : end])
            structural_usage["import"] += len(block)

        elif isinstance(node, ast.ClassDef):
            sig_start = node.lineno
            sig_end = node.body[0].lineno - 1 if node.body else node.lineno
            sig = "".join(lines[sig_start - 1 : sig_end])
            structural_usage["class_signatures"] += len(sig)

        elif isinstance(node, ast.FunctionDef):
            sig_start = node.lineno
            sig_end = node.body[0].lineno - 1 if node.body else node.lineno
            sig = "".join(lines[sig_start - 1 : sig_end])
            structural_usage["method_signatures"] += len(sig)

        elif isinstance(node, ast.Assign):
            start = node.lineno
            end = getattr(node, "end_lineno", node.lineno)
            assignment = "".join(lines[start - 1 : end])
            structural_usage["assignments"] += len(assignment)

    return structural_usage


def analyze_file_structure(source: str):
    lexical = analyze_lexical_characters(source)
    structural = analyze_structural_characters(source)
    total_chars = len(source)

    return lexical, structural, total_chars


def main():
    path = "examples/example_input.py"

    with open(path, "r", encoding="utf-8") as f:
        source = f.read()

    lexical_usage, structural_usage, total = analyze_file_structure(source)

    print("Lexical Character Usage:")
    for k, v in sorted(lexical_usage.items(), key=lambda x: -x[1]):
        print(f"{k:20}: {v}")

    print("\nStructural Character Usage:")
    for k, v in sorted(structural_usage.items(), key=lambda x: -x[1]):
        print(f"{k:20}: {v}")

    print(f"\nTotal Characters in File: {total}")


if __name__ == "__main__":
    main()
