"""Span extraction, including the parent-attribution bug that containment fixed."""
from __future__ import annotations

import textwrap

import pytest

from dwago.spans import Span, _assign_parents, extract_spans, language_for


def test_language_detection():
    assert language_for("a/b/c.py") == "python"
    assert language_for("x.TSX") == "typescript"
    assert language_for("Makefile") is None


def test_python_spans_and_docstrings(tmp_path):
    src = textwrap.dedent('''
        def alpha(a, b):
            """Adds two numbers."""
            return a + b


        class Widget:
            """A widget."""

            def render(self):
                return 1

            def hide(self):
                return 2


        def omega():
            return 0
    ''').lstrip()
    f = tmp_path / "m.py"
    f.write_text(src)

    fs = extract_spans(f, tmp_path)
    by = {s.name: s for s in fs.spans}

    assert by["alpha"].kind == "function"
    assert by["alpha"].docstring == "Adds two numbers."
    assert by["Widget"].kind == "class"
    # Methods must be attributed to the enclosing class...
    assert by["render"].kind == "method"
    assert by["render"].parent == "Widget"
    # ...and a later top-level function must NOT be.
    assert by["omega"].parent is None
    assert by["omega"].kind == "function"


def test_parents_come_from_containment_not_depth():
    """Regression: depth-based tracking leaked containers across sibling subtrees.

    A class ending at line 40 was being attached to a function starting at line
    700 purely because both sat at the same tree depth. Containment cannot do
    that, and this pins the behaviour.
    """
    spans = [
        Span(name="Early", kind="class", start_line=10, end_line=40),
        Span(name="inside", kind="function", start_line=15, end_line=20),
        Span(name="far_away", kind="function", start_line=700, end_line=710),
    ]
    _assign_parents(spans)
    by = {s.name: s for s in spans}
    assert by["inside"].parent == "Early"
    assert by["inside"].kind == "method"
    assert by["far_away"].parent is None, "a non-enclosing container must not be attached"


def test_innermost_container_wins():
    spans = [
        Span(name="Outer", kind="class", start_line=1, end_line=100),
        Span(name="Inner", kind="class", start_line=10, end_line=50),
        Span(name="meth", kind="function", start_line=20, end_line=25),
    ]
    _assign_parents(spans)
    assert {s.name: s.parent for s in spans}["meth"] == "Outer.Inner"


def test_by_name_prefers_top_level_over_method(tmp_path):
    src = textwrap.dedent('''
        def run():
            return 1

        class A:
            def run(self):
                return 2
    ''').lstrip()
    f = tmp_path / "n.py"
    f.write_text(src)
    fs = extract_spans(f, tmp_path)
    idx = fs.by_name()
    assert idx["run"].parent is None
    assert "A.run" in idx


@pytest.mark.parametrize("prefix", ["", "async "])
def test_python_decorated_function_has_one_span(tmp_path, prefix):
    f = tmp_path / "tools.py"
    f.write_text(
        "@first\n"
        "@second('console')\n"
        f"{prefix}def read_console():\n"
        '    """Read console messages."""\n'
        "    def nested():\n"
        "        return 1\n"
        "    return nested()\n")

    fs = extract_spans(f, tmp_path)
    matches = [s for s in fs.spans if s.name == "read_console"]
    assert len(matches) == 1, "the decorator wrapper and its definition are one symbol"
    assert (matches[0].start_line, matches[0].end_line) == (1, 7)
    assert matches[0].signature == f"{prefix}def read_console():"
    assert matches[0].docstring == "Read console messages."
    assert [s.name for s in fs.spans] == ["read_console", "nested"], \
        "skipping the inner declaration must not skip its descendants"


def test_python_decorated_classes_keep_distinct_methods(tmp_path):
    f = tmp_path / "classes.py"
    f.write_text(
        "@decorate\n"
        "class Outer:\n"
        "    @classmethod\n"
        "    async def run(cls):\n"
        "        return 1\n"
        "\n"
        "    @decorate\n"
        "    class Inner:\n"
        "        @staticmethod\n"
        "        def run():\n"
        "            return 2\n")

    fs = extract_spans(f, tmp_path)
    assert [(s.qualified_name, s.kind, s.start_line, s.end_line) for s in fs.spans] == [
        ("Outer", "class", 1, 11),
        ("Outer.run", "method", 3, 5),
        ("Outer.Inner", "class", 7, 11),
        ("Outer.Inner.run", "method", 9, 11),
    ], "retain both same-name methods and the real nested class, once each"


def test_oversized_file_is_skipped(tmp_path, monkeypatch):
    import dwago.spans as sp

    monkeypatch.setattr(sp, "MAX_FILE_BYTES", 10)
    f = tmp_path / "big.py"
    f.write_text("def f():\n    return 1\n" * 20)
    assert sp.extract_spans(f, tmp_path) is None


def test_unknown_language_returns_none(tmp_path):
    f = tmp_path / "notes.xyz"
    f.write_text("hello")
    assert extract_spans(f, tmp_path) is None
