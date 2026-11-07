"""Version élève des notebooks : vide les fonctions marquées # SOLUTION.

Une fonction est une solution si la ligne non vide qui précède sa définition (ou son premier
décorateur) est un commentaire `# SOLUTION`. Son corps est remplacé par un TODO ; la signature et
la docstring sont gardées.

Usage (depuis la racine du toolchain) :
    python scripts/make_student.py                 écrit dans student/tps/
    python scripts/make_student.py --out autre/dir
"""

import argparse
import ast
from pathlib import Path

MARKER = "# SOLUTION"


def strip_solutions(source: str) -> tuple[str, int]:
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    targets = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        first = min([node.lineno] + [d.lineno for d in node.decorator_list])
        k = first - 2
        while k >= 0 and not lines[k].strip():
            k -= 1
        if k >= 0 and lines[k].strip().startswith(MARKER):
            targets.append(node)
    # Remplacer de bas en haut pour garder les numéros de ligne valides
    for node in sorted(targets, key=lambda n: n.lineno, reverse=True):
        body = node.body
        has_doc = isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str)
        start = (body[1].lineno if has_doc and len(body) > 1 else body[0].lineno) - 1
        if has_doc and len(body) == 1:
            start = body[0].end_lineno
        end = node.end_lineno
        indent = " " * (body[0].col_offset)
        lines[start:end] = [f"{indent}# TODO : à compléter\n", f"{indent}raise NotImplementedError\n"]
    return "".join(lines), len(targets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="tps")
    parser.add_argument("--out", default="student/tps")
    args = parser.parse_args()
    src, out = Path(args.src), Path(args.out)
    for nb in sorted(src.rglob("[0-9]*.py")):
        text, n = strip_solutions(nb.read_text(encoding="utf-8"))
        dest = out / nb.relative_to(src)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8", newline="\n")
        print(f"{nb} -> {dest} ({n} fonction(s) vidée(s))")


if __name__ == "__main__":
    main()
