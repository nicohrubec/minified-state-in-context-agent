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

    def line_col_to_offset(line: int, col: int) -> int:
        return sum(len(lines[i]) for i in range(line - 1)) + col

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
        elif isinstance(node, ast.AnnAssign):
            start = line_col_to_offset(node.lineno, node.col_offset)
            end = line_col_to_offset(node.end_lineno, node.end_col_offset)
            structural_usage["annotated_assignments"] += len(source[start:end])
        elif isinstance(node, (ast.If, ast.While)):
            test = node.test
            start = line_col_to_offset(test.lineno, test.col_offset)
            end = line_col_to_offset(test.end_lineno, test.end_col_offset)
            condition = source[start:end]
            structural_usage["conditions"] += len(condition)
        elif isinstance(node, ast.For):
            target = node.target
            iter_ = node.iter
            start = line_col_to_offset(target.lineno, target.col_offset)
            end = line_col_to_offset(iter_.end_lineno, iter_.end_col_offset)
            condition = source[start:end]
            structural_usage["conditions"] += len(condition)
        elif isinstance(node, ast.Call):
            start = line_col_to_offset(node.lineno, node.col_offset)
            end = line_col_to_offset(node.end_lineno, node.end_col_offset)
            call_text = source[start:end]

            if isinstance(node.func, ast.Name) and node.func.id == "print":
                structural_usage["print_statements"] += len(call_text)

            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in {
                    "debug",
                    "info",  # annotated assignments
                    "warning",
                    "error",
                    "critical",
                    "exception",
                    "log",
                }:
                    structural_usage["logger_calls"] += len(call_text)
                else:
                    structural_usage["function_calls"] += len(call_text)

            else:
                structural_usage["function_calls"] += len(call_text)

        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            for deco in node.decorator_list:
                start = line_col_to_offset(deco.lineno, deco.col_offset)
                end = line_col_to_offset(deco.end_lineno, deco.end_col_offset)
                structural_usage["decorators"] += len(source[start:end])

    blank_lines = [line for line in lines if line.strip() == ""]
    structural_usage["blank_lines"] = sum(len(line) for line in blank_lines)

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
