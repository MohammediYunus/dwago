"""Tree-sitter span extraction — the primary source of symbol line ranges.

Why this module exists at all: graphify's ``graph.json`` gives a node only
``source_location: "L<n>"`` — a *start* line, no end line, no signature, no body
(verified in ``graphify/extract.py``, every node emission site). LLM-derived
nodes carry ``null``. That is not enough to build a retrieval document, cite a
range, or map a diff hunk onto a symbol.

The obvious workaround — sort a file's symbols by start line and treat
``[start_i, start_i+1)`` as the range — is wrong, not merely rough. Symbol ranges
*nest*: a method's range lies inside its class's range, so sorted starts do not
partition a file. A file-level node sitting at L1 would additionally swallow
everything above the first symbol. So dwago parses for real ranges and keeps
the sort trick only as a last-resort fallback for languages with no grammar
(see :func:`approximate_spans`).

This deliberately does not import graphify's ``LanguageConfig`` or its private
``_*_CONFIG`` objects. Those are internal, they have changed shape across
releases, and dwago only needs "where does each definition start and end" —
a far smaller question than graphify's full extraction. Binding to them would
recreate the version-drift coupling this design is meant to avoid.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .source import resolve_source_file

log = logging.getLogger(__name__)

__all__ = ["Span", "FileSpans", "extract_spans", "extract_repo_spans", "approximate_spans"]

# Guard against pathological inputs (minified bundles, vendored blobs, generated
# parsers). graphify applies size caps in security.py for the same reason.
MAX_FILE_BYTES = 2_000_000


@dataclass(slots=True)
class Span:
    """One definition's extent in a file. Lines are 1-based and inclusive."""

    name: str
    kind: str                      # function | class | method | interface | struct | ...
    start_line: int
    end_line: int
    signature: str = ""            # the declaration line(s), trimmed
    docstring: str = ""            # docstring or leading comment block
    parent: str | None = None      # enclosing definition name, if nested

    @property
    def qualified_name(self) -> str:
        return f"{self.parent}.{self.name}" if self.parent else self.name


@dataclass(slots=True)
class FileSpans:
    path: str
    language: str
    spans: list[Span] = field(default_factory=list)
    lines: int = 0

    def by_name(self) -> dict[str, Span]:
        """Index by bare name, then by qualified name.

        Bare names are inserted first and never overwritten, so when a file has
        both ``foo`` and ``Bar.foo`` a lookup for ``foo`` resolves to the
        top-level one rather than to whichever method happened to parse last.
        """
        out: dict[str, Span] = {}
        for s in self.spans:
            out.setdefault(s.name, s)
        for s in self.spans:
            if s.parent:
                out.setdefault(s.qualified_name, s)
        return out


# ── Grammar registry ─────────────────────────────────────────────────────────
#
# Each entry: extensions -> (tree-sitter module, {node_type: kind}).
# The node types are the *definition* forms we want ranges for. Anything not
# listed is walked through but not recorded, which keeps this table small and
# means an unfamiliar grammar degrades to "no spans" rather than to garbage.

_LANGUAGES: dict[str, tuple[str, dict[str, str]]] = {
    "python": ("tree_sitter_python", {
        "function_definition": "function",
        "class_definition": "class",
        "decorated_definition": "decorated",
    }),
    "javascript": ("tree_sitter_javascript", {
        "function_declaration": "function",
        "generator_function_declaration": "function",
        "class_declaration": "class",
        "method_definition": "method",
        "lexical_declaration": "binding",
    }),
    "typescript": ("tree_sitter_typescript", {
        "function_declaration": "function",
        "generator_function_declaration": "function",
        "class_declaration": "class",
        "method_definition": "method",
        "interface_declaration": "interface",
        "type_alias_declaration": "type",
        "enum_declaration": "enum",
        "lexical_declaration": "binding",
    }),
    "go": ("tree_sitter_go", {
        "function_declaration": "function",
        "method_declaration": "method",
        "type_declaration": "type",
    }),
    "rust": ("tree_sitter_rust", {
        "function_item": "function",
        "struct_item": "struct",
        "enum_item": "enum",
        "trait_item": "trait",
        "impl_item": "impl",
        "mod_item": "module",
    }),
    "java": ("tree_sitter_java", {
        "class_declaration": "class",
        "interface_declaration": "interface",
        "method_declaration": "method",
        "enum_declaration": "enum",
        "record_declaration": "record",
    }),
    "c": ("tree_sitter_c", {
        "function_definition": "function",
        "struct_specifier": "struct",
        "enum_specifier": "enum",
    }),
    "cpp": ("tree_sitter_cpp", {
        "function_definition": "function",
        "class_specifier": "class",
        "struct_specifier": "struct",
        "namespace_definition": "namespace",
    }),
    "c_sharp": ("tree_sitter_c_sharp", {
        "class_declaration": "class",
        "interface_declaration": "interface",
        "method_declaration": "method",
        "struct_declaration": "struct",
        "record_declaration": "record",
    }),
    "ruby": ("tree_sitter_ruby", {
        "method": "method",
        "class": "class",
        "module": "module",
        "singleton_method": "method",
    }),
    "php": ("tree_sitter_php", {
        "function_definition": "function",
        "class_declaration": "class",
        "method_declaration": "method",
        "interface_declaration": "interface",
        "trait_declaration": "trait",
    }),
    "swift": ("tree_sitter_swift", {
        "function_declaration": "function",
        "class_declaration": "class",
        "protocol_declaration": "protocol",
    }),
    "kotlin": ("tree_sitter_kotlin", {
        "function_declaration": "function",
        "class_declaration": "class",
        "object_declaration": "object",
    }),
    "scala": ("tree_sitter_scala", {
        "function_definition": "function",
        "class_definition": "class",
        "object_definition": "object",
        "trait_definition": "trait",
    }),
    "lua": ("tree_sitter_lua", {
        "function_declaration": "function",
    }),
    "bash": ("tree_sitter_bash", {
        "function_definition": "function",
    }),
    "elixir": ("tree_sitter_elixir", {
        "call": "function",
    }),
    "zig": ("tree_sitter_zig", {
        "function_declaration": "function",
    }),
    "julia": ("tree_sitter_julia", {
        "function_definition": "function",
        "struct_definition": "struct",
    }),
}

_EXT_TO_LANG: dict[str, str] = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".mts": "typescript", ".cts": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp",
    ".cs": "c_sharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin", ".kts": "kotlin",
    ".scala": "scala",
    ".lua": "lua",
    ".sh": "bash", ".bash": "bash",
    ".ex": "elixir", ".exs": "elixir",
    ".zig": "zig",
    ".jl": "julia",
}

# Populated lazily; a missing grammar is cached as None so we attempt the import
# once per process rather than once per file.
_PARSER_CACHE: dict[str, object | None] = {}


def language_for(path: str | Path) -> str | None:
    return _EXT_TO_LANG.get(Path(path).suffix.lower())


def _get_parser(language: str):
    """Return a tree-sitter Parser for ``language``, or None if unavailable."""
    if language in _PARSER_CACHE:
        return _PARSER_CACHE[language]

    parser = None
    entry = _LANGUAGES.get(language)
    if entry:
        module_name, _ = entry
        try:
            import importlib

            from tree_sitter import Language, Parser

            mod = importlib.import_module(module_name)
            # tree-sitter-typescript ships two grammars in one module and has no
            # bare `language()`; the rest expose a single `language()`.
            if language == "typescript":
                raw = mod.language_typescript()
            elif hasattr(mod, "language"):
                raw = mod.language()
            else:
                raw = None
            if raw is not None:
                parser = Parser(Language(raw))
        except Exception as exc:  # noqa: BLE001 - a missing grammar is not fatal
            log.debug("no grammar for %s: %s", language, exc)
            parser = None

    _PARSER_CACHE[language] = parser
    return parser


def _node_text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _find_name(node, src: bytes) -> str | None:
    """Best-effort name for a definition node.

    Tries the grammar's ``name`` field first (most grammars have one), then falls
    back to the first identifier-ish child. Returning None is fine — the caller
    drops unnamed definitions rather than inventing a label.
    """
    named = node.child_by_field_name("name")
    if named is not None:
        return _node_text(named, src).strip()

    # Go/C/C++ put the name inside a declarator subtree.
    decl = node.child_by_field_name("declarator")
    while decl is not None:
        inner = decl.child_by_field_name("declarator")
        if inner is None:
            break
        decl = inner
    if decl is not None and decl.type in {"identifier", "field_identifier", "type_identifier"}:
        return _node_text(decl, src).strip()

    for child in node.children:
        if child.type in {"identifier", "type_identifier", "field_identifier",
                          "constant", "property_identifier", "word"}:
            return _node_text(child, src).strip()
    return None


def _signature(node, src: bytes, body_field: str = "body") -> str:
    """The declaration text with the body stripped off.

    Cutting at the body start keeps a 400-line function's signature to one line,
    which is what a retrieval document wants.
    """
    body = node.child_by_field_name(body_field)
    end = body.start_byte if body is not None else node.end_byte
    text = src[node.start_byte:min(end, node.start_byte + 600)].decode("utf-8", errors="replace")
    return " ".join(text.split())[:400]


def _python_docstring(node, src: bytes) -> str:
    body = node.child_by_field_name("body")
    if body is None or not body.children:
        return ""
    first = body.children[0]
    if first.type == "expression_statement" and first.children:
        lit = first.children[0]
        if lit.type == "string":
            raw = _node_text(lit, src)
            return " ".join(raw.strip("\"'").split())[:600]
    return ""


def _leading_comment(node, src_lines: list[str]) -> str:
    """Comment block immediately above a definition, for non-Python languages.

    Walks upward from the declaration while lines look like comments. Blank lines
    terminate the block, so an unrelated comment further up is not absorbed.
    """
    out: list[str] = []
    i = node.start_point[0] - 1
    while i >= 0 and len(out) < 20:
        stripped = src_lines[i].strip()
        if not stripped:
            break
        if stripped.startswith(("//", "#", "*", "/*", "///", "--")):
            out.append(stripped.lstrip("/*#- ").rstrip("*/").strip())
            i -= 1
            continue
        break
    return " ".join(reversed(out))[:600]


def _walk(node, depth: int = 0) -> Iterator[tuple[object, int]]:
    yield node, depth
    for child in node.children:
        yield from _walk(child, depth + 1)


def extract_spans(path: Path, root: Path | None = None) -> FileSpans | None:
    """Parse one file and return its definition spans, or None if unparseable."""
    language = language_for(path)
    if language is None:
        return None

    source = path
    rel = str(path)
    if root is not None:
        source = resolve_source_file(root, path.absolute())
        if source is None:
            return None
        try:
            rel = path.absolute().relative_to(root.absolute()).as_posix()
        except ValueError:
            # An explicitly supplied path may use a symlinked spelling of the
            # selected root. Its resolved target has already passed containment.
            rel = source.relative_to(root.resolve()).as_posix()

    try:
        raw = source.read_bytes()
    except OSError as exc:
        log.debug("unreadable %s: %s", path, exc)
        return None
    if len(raw) > MAX_FILE_BYTES:
        log.debug("skipping oversized file %s (%d bytes)", path, len(raw))
        return None

    parser = _get_parser(language)
    if parser is None:
        return None

    try:
        tree = parser.parse(raw)
    except Exception as exc:  # noqa: BLE001 - a parse failure is per-file, not fatal
        log.debug("parse failed for %s: %s", path, exc)
        return None

    _, wanted = _LANGUAGES[language]
    text_lines = raw.decode("utf-8", errors="replace").splitlines()
    result = FileSpans(path=rel, language=language, lines=len(text_lines))

    # Parents are assigned by RANGE CONTAINMENT in a second pass, not by tree
    # depth during the walk. Depth-based tracking leaks: when the walk moves from
    # one subtree to an unrelated sibling at the same depth, a container recorded
    # earlier is still in scope and gets attached to symbols it does not enclose.
    # That produced real misattributions on graphify's own serve.py (a method at
    # L704 inheriting a class that ends at L448). Containment cannot do that.
    collected: list[Span] = []

    for node, _depth in _walk(tree.root_node):
        kind = wanted.get(node.type)
        if kind is None:
            continue

        # The wrapper below already emits this definition with its decorators.
        # Skip only the inner declaration; _walk still visits its descendants.
        if (language == "python"
                and node.type in {"function_definition", "class_definition"}
                and node.parent is not None
                and node.parent.type == "decorated_definition"):
            continue

        # Python decorators wrap the real definition; descend to it so the span
        # covers the decorators but the name/kind come from the inner node.
        target = node
        if kind == "decorated":
            inner = next((c for c in node.children
                          if c.type in {"function_definition", "class_definition"}), None)
            if inner is None:
                continue
            target = inner
            kind = wanted.get(inner.type, "function")

        name = _find_name(target, raw)
        if not name:
            continue

        # A `lexical_declaration` is only interesting when it binds a function.
        if kind == "binding":
            if not any(c.type in {"arrow_function", "function_expression", "function"}
                       for c in _iter_descendants(target, max_depth=3)):
                continue
            kind = "function"

        doc = (_python_docstring(target, raw) if language == "python"
               else _leading_comment(node, text_lines))

        collected.append(Span(
            name=name,
            kind=kind,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            signature=_signature(target, raw),
            docstring=doc,
        ))

    result.spans = _assign_parents(collected)
    return result


CONTAINER_KINDS = frozenset({
    "class", "struct", "interface", "trait", "impl", "module",
    "object", "namespace", "record", "enum",
})


def _assign_parents(spans: list[Span]) -> list[Span]:
    """Attach each span to the innermost container whose range encloses it.

    Containment is the only sound basis here: nesting in source is expressed by
    range inclusion, and a symbol belongs to the tightest container around it.
    Ties (a container starting on the same line) resolve to the smaller range.

    A plain `function` that turns out to sit inside a container is reclassified
    as a `method`, which is what callers and retrieval documents expect.
    """
    containers = sorted(
        (s for s in spans if s.kind in CONTAINER_KINDS),
        key=lambda s: (s.end_line - s.start_line),
    )
    for s in spans:
        best: Span | None = None
        for c in containers:
            if c is s:
                continue
            if c.start_line <= s.start_line and s.end_line <= c.end_line:
                # `containers` is sorted smallest-first, so the first hit is the
                # innermost one.
                best = c
                break
        if best is not None:
            s.parent = best.qualified_name
            if s.kind == "function":
                s.kind = "method"
    return spans


def _iter_descendants(node, max_depth: int = 3, _depth: int = 0):
    if _depth > max_depth:
        return
    for child in node.children:
        yield child
        yield from _iter_descendants(child, max_depth, _depth + 1)


def extract_repo_spans(
    root: Path,
    files: list[Path] | None = None,
    max_workers: int | None = None,
) -> dict[str, FileSpans]:
    """Parse every supported code file under ``root``.

    Runs in a process pool because tree-sitter parsing is CPU-bound and releases
    nothing useful to threads. Falls back to serial on any pool failure so a
    restricted environment still produces spans.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    root = Path(root).resolve()
    if files is None:
        files = [p for p in _iter_source_files(root)]

    if not files:
        return {}

    workers = max_workers or min(8, (os.cpu_count() or 2))
    out: dict[str, FileSpans] = {}

    if workers <= 1 or len(files) < 16:
        for p in files:
            fs = extract_spans(p, root)
            if fs is not None:
                out[fs.path] = fs
        return out

    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(extract_spans, p, root): p for p in files}
            for fut in as_completed(futures):
                try:
                    fs = fut.result()
                except Exception as exc:  # noqa: BLE001
                    log.debug("span worker failed for %s: %s", futures[fut], exc)
                    continue
                if fs is not None:
                    out[fs.path] = fs
    except Exception as exc:  # noqa: BLE001 - pools fail in sandboxes; degrade
        log.warning("process pool unavailable (%s); parsing serially", exc)
        for p in files:
            fs = extract_spans(p, root)
            if fs is not None:
                out[fs.path] = fs

    return out


_SKIP_DIRS = {".git", ".hg", "node_modules", "__pycache__", ".venv", "venv",
              "dist", "build", "target", "vendor", ".mypy_cache", ".pytest_cache",
              "graphify-out", "dwago-out", ".tox", ".next", ".cache"}


def _iter_source_files(root: Path) -> Iterator[Path]:
    for p in root.rglob("*"):
        if resolve_source_file(root, p) is None:
            continue
        if any(part in _SKIP_DIRS for part in p.parts):
            continue
        if p.suffix.lower() in _EXT_TO_LANG:
            yield p


def approximate_spans(starts: list[tuple[str, int]], total_lines: int) -> dict[str, tuple[int, int]]:
    """Last-resort ranges for files with no grammar.

    Explicitly NOT the general strategy — see the module docstring. This assumes
    a flat, non-nesting symbol list, which is only safe for the config-ish
    formats that reach it. Each symbol runs to the line before the next one.
    """
    ordered = sorted(starts, key=lambda t: t[1])
    out: dict[str, tuple[int, int]] = {}
    for i, (name, line) in enumerate(ordered):
        end = ordered[i + 1][1] - 1 if i + 1 < len(ordered) else total_lines
        out[name] = (line, max(line, end))
    return out
