import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("make_student", ROOT / "scripts" / "make_student.py")
make_student = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_student)

SAMPLE = '''
def keep():
    return 1


# SOLUTION
@decorator
def solved(x):
    """Doc gardee."""
    y = x + 1
    return y


def cell():
    # SOLUTION
    def nested(a):
        return a * 2

    return nested
'''


def test_strip_solutions_on_sample():
    out, n = make_student.strip_solutions(SAMPLE)
    assert n == 2
    ast.parse(out)
    assert "return 1" in out and "Doc gardee." in out
    assert "y = x + 1" not in out and "return a * 2" not in out
    assert out.count("raise NotImplementedError") == 2


def test_real_notebooks_stay_valid_python():
    for nb in (ROOT / "notebooks").rglob("[0-9]*.py"):
        out, _ = make_student.strip_solutions(nb.read_text(encoding="utf-8"))
        ast.parse(out)
