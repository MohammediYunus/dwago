"""Standalone extraction: a code knowledge graph with no external extractor.

Walks the repository, parses every supported source file with tree-sitter
(:mod:`dwago.spans` owns the grammars), resolves import edges for the
languages where that is reliable, detects communities with Louvain, and
returns a graph dict in the same node-link shape :func:`dwago.ingest.ingest`
already consumes.

Nodes: one per file (documents included) and one per parsed definition.
Edges: ``contains`` (file→symbol, outer→inner definition), ``imports``
(file→file, Python/TypeScript/JavaScript resolution), and everything else —
co-change, diffusion — layers on later from git history.

Communities are named after the deepest directory the members share, which
is honest and stable; ``dwago summarize`` can replace the name with an LLM
sentence later.
"""
from __future__ import annotations

import logging
import posixpath
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from .source import resolve_source_file
from .spans import FileSpans, extract_repo_spans, language_for

log = logging.getLogger(__name__)

_DOC_EXTS = {".md", ".rst", ".txt"}
_SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "env",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "dist", "build", "target", "vendor", ".next", ".nuxt", ".output",
    ".terraform", ".idea", ".vscode", "coverage", ".tox", ".eggs",
    ".worktrees", ".codex_worktrees", "dwago-out", "graphify-out",
}
_MAX_DOC_BYTES = 1_500_000

_TS_EXTS = (".ts", ".tsx", ".js", ".jsx", ".mts", ".cts", ".mjs", ".cjs")
_TS_INDEXES = tuple(f"/index{e}" for e in _TS_EXTS)


def _iter_files(root: Path):
    for p in sorted(root.rglob("*")):
        if resolve_source_file(root, p) is None:
            continue
        rel = p.relative_to(root)
        if any(part in _SKIP_DIRS or part.startswith(".") and part not in (".github",)
               for part in rel.parts[:-1]):
            continue
        name = p.name
        if name.startswith(".") and p.suffix not in _DOC_EXTS:
            continue
        # Graph keys use the same slash-separated paths as Git on every OS.
        yield p, rel.as_posix()


# ── import resolution ────────────────────────────────────────────────────────

_PY_IMPORT = re.compile(
    r"^[ \t]*(?:from[ \t]+(\.*[\w\.]*)[ \t]+import[ \t]+([^\n]+)|import[ \t]+([\w\.]+))", re.M)
_TS_IMPORT = re.compile(
    r"""(?:import|export)\s+(?:[\w\s{},*$]+\s+from\s+)?['"]([^'"]+)['"]"""
    r"""|require\(\s*['"]([^'"]+)['"]\s*\)""")


def _resolve_py(module: str, importer: str, files: set[str]) -> str | None:
    """a.b.c → a/b/c.py or a/b/c/__init__.py, tried from the repo root and
    from the importer's package."""
    dotted = module.lstrip(".")
    rel_up = len(module) - len(module.lstrip("."))
    bases: list[PurePosixPath] = [PurePosixPath("")]
    if rel_up:
        base = PurePosixPath(importer).parent
        for _ in range(rel_up - 1):
            base = base.parent
        bases = [base]
    else:
        bases.append(PurePosixPath(importer).parent)
    parts = dotted.split(".") if dotted else []
    for base in bases:
        for take in range(len(parts), 0, -1):
            stem = base.joinpath(*parts[:take])
            for cand in (f"{stem}.py", f"{stem}/__init__.py"):
                cand = str(PurePosixPath(cand))
                if cand in files:
                    return cand
    return None


def _resolve_ts(spec: str, importer: str, files: set[str]) -> str | None:
    if not spec.startswith("."):
        return None                                    # external package
    # Collapse parent segments lexically, without consulting the filesystem.
    base_s = posixpath.normpath(str(PurePosixPath(importer).parent / spec))
    if base_s == ".." or base_s.startswith("../"):
        return None
    # exact file, with extension already
    if base_s in files:
        return base_s
    for ext in _TS_EXTS:
        if base_s + ext in files:
            return base_s + ext
    for idx in _TS_INDEXES:
        if base_s + idx in files:
            return base_s + idx
    return None


def _imports_of(rel: str, text: str, files: set[str]) -> set[str]:
    lang = language_for(rel)
    out: set[str] = set()
    if lang == "python":
        for m in _PY_IMPORT.finditer(text):
            module = m.group(1) if m.group(1) is not None else m.group(3)
            if not module:
                module = "."
            candidates = [module]
            # `from X import a, b` may mean modules X.a, X.b
            if m.group(2):
                names = [x.strip().split(" as ")[0]
                         for x in m.group(2).replace("(", "").replace(")", "").split(",")]
                sep = "" if module.endswith(".") else "."
                candidates += [f"{module}{sep}{nm}" for nm in names if nm]
            for cand in candidates:
                hit = _resolve_py(cand, rel, files)
                if hit and hit != rel:
                    out.add(hit)
    elif rel.endswith(_TS_EXTS):
        for m in _TS_IMPORT.finditer(text):
            spec = m.group(1) or m.group(2)
            hit = _resolve_ts(spec, rel, files) if spec else None
            if hit and hit != rel:
                out.add(hit)
    return out


# ── communities ──────────────────────────────────────────────────────────────

def _communities(files: list[str], fedges: list[tuple[str, str]]) -> dict[str, tuple[int, str]]:
    """file → (community id, community name). Louvain over the import graph;
    files the graph says nothing about are grouped by their top directory.

    Canonical ordering keeps unchanged graphs stable within the same
    Python/NetworkX runtime and default backend, regardless of hash seed.
    """
    import networkx as nx

    g = nx.Graph()
    # A fixed RNG seed alone does not remove Louvain's input-order sensitivity.
    g.add_nodes_from(sorted(files))
    g.add_edges_from(sorted({(min(a, b), max(a, b)) for a, b in fedges}))

    try:
        comms = nx.community.louvain_communities(g, seed=42)
    except Exception:  # noqa: BLE001 - tiny graphs
        comms = [{f} for f in files]

    def topdir(f: str) -> str:
        parts = PurePosixPath(f).parts
        return parts[0] if len(parts) > 1 else "(root)"

    # merge singleton communities into per-directory buckets
    merged: dict[str, set] = defaultdict(set)
    real: list[set] = []
    for c in comms:
        if len(c) <= 2:
            for f in c:
                merged[topdir(f)].add(f)
        else:
            real.append(set(c))
    real.extend(s for s in merged.values() if s)

    def name_of(members: set) -> str:
        paths = [PurePosixPath(f).parts[:-1] for f in members]
        if not paths:
            return "misc"
        common: list[str] = []
        for level in zip(*paths):
            if all(x == level[0] for x in level):
                common.append(level[0])
            else:
                break
        if common:
            return "/".join(common)
        counts = Counter(topdir(f) for f in members)
        return min(counts, key=lambda directory: (-counts[directory], directory))

    out: dict[str, tuple[int, str]] = {}
    ordered = sorted(real, key=lambda members: (-len(members), sorted(members)))
    for i, members in enumerate(ordered):
        nm = name_of(members)
        for f in sorted(members):
            out[f] = (i, nm)
    return out


# ── the graph ────────────────────────────────────────────────────────────────

def extract_repo(root: str | Path, *, spans_by_file: dict[str, FileSpans] | None = None) -> dict:
    """Build the node-link graph for ``root``. Pass ``spans_by_file`` to reuse
    an existing parse (ingest parses again for line ranges; sharing saves a
    full pass)."""
    root = Path(root).resolve()
    listed = list(_iter_files(root))
    all_rel = {rel for _, rel in listed}

    code_paths = [p for p, rel in listed if language_for(rel) is not None]
    if spans_by_file is None:
        spans_by_file = extract_repo_spans(root, files=code_paths)

    nodes: list[dict] = []
    links: list[dict] = []

    file_texts: dict[str, str] = {}
    for p, rel in listed:
        is_doc = p.suffix.lower() in _DOC_EXTS
        is_code = language_for(rel) is not None
        if not (is_doc or is_code):
            continue
        source = resolve_source_file(root, p)
        if source is None:
            continue
        try:
            if source.stat().st_size > _MAX_DOC_BYTES:
                continue
            text = source.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        file_texts[rel] = text
        nodes.append({
            "id": rel,
            "label": PurePosixPath(rel).name,
            "file_type": "document" if is_doc else "code",
            "source_file": rel,
            "source_location": "L1",
            "_origin": "dwago",
        })

    # symbols + containment
    for rel, fs in spans_by_file.items():
        if rel not in file_texts:
            continue
        first_at: dict[str, object] = {}
        for s in fs.spans:
            first_at.setdefault(s.name, s)
        for s in fs.spans:
            sid = f"{rel}::{s.name}::{s.start_line}"
            nodes.append({
                "id": sid,
                "label": s.name,
                "file_type": "code",
                "kind": s.kind,
                "source_file": rel,
                "source_location": f"L{s.start_line}",
                "_origin": "dwago",
            })
            parent_id = rel
            if s.parent and s.parent in first_at:
                ps = first_at[s.parent]
                parent_id = f"{rel}::{ps.name}::{ps.start_line}"
            links.append({"source": parent_id, "target": sid,
                          "relation": "contains", "weight": 1.0,
                          "confidence_score": 1.0})

    # imports
    fedges: list[tuple[str, str]] = []
    for rel, text in file_texts.items():
        for hit in _imports_of(rel, text, all_rel):
            if hit in file_texts:
                links.append({"source": rel, "target": hit,
                              "relation": "imports", "weight": 1.0,
                              "confidence_score": 1.0})
                fedges.append((rel, hit))

    comm = _communities(sorted(file_texts), fedges)
    for n in nodes:
        c = comm.get(n["source_file"])
        if c:
            n["community"], n["community_name"] = c

    log.info("extracted %d nodes, %d links, %d communities",
             len(nodes), len(links), len({c for c, _ in comm.values()}))
    return {"directed": True, "nodes": nodes, "links": links}
