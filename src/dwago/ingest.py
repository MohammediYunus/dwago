"""Read graphify's ``graph.json`` into dwago's own store.

This is the seam between the two projects, and it is deliberately defensive.
graphify moves fast (issue numbers past #2775 at time of writing) and its
``graph.json`` has changed shape across releases: ``_src``/``_tgt`` direction
carriers appeared in 0.9.x, ``community_name`` is present in some versions and
not others, ``hyperedges`` semantics were hardened in #2485, and a future
multigraph mode will change link multiplicity. So ingest validates what it
found, tolerates absent optional keys, and fails loudly on a shape it does not
recognise rather than silently producing a half-empty index.

The other job here is enrichment that graphify structurally cannot provide:
every node gets real line ranges, a signature, a docstring, and a content hash,
sourced from :mod:`dwago.spans`. Without that the retrieval documents in
Phase 2 would be little more than identifiers.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .identity import content_hash, node_key
from .spans import FileSpans, extract_repo_spans

log = logging.getLogger(__name__)

__all__ = ["IngestResult", "ingest", "load_graph_json", "GraphJsonError"]

# graphify caps graph.json at 512 MiB by default; mirror that so a corrupt or
# adversarial file cannot exhaust memory here either.
MAX_GRAPH_BYTES = 512 * 1024 * 1024


class GraphJsonError(RuntimeError):
    """graph.json is missing, unreadable, or not in a shape dwago understands."""


@dataclass
class IngestResult:
    nodes: int = 0
    edges: int = 0
    spans_matched: int = 0
    spans_total: int = 0
    files_parsed: int = 0
    dropped_edges: int = 0
    duplicate_nodes: int = 0      # collapsed onto an identical content key
    external_refs: int = 0        # nodes naming a symbol defined outside this file
    unparsed_lang_nodes: int = 0  # nodes in languages with no bundled grammar
    built_at_commit: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def span_coverage(self) -> float:
        """Fraction of code nodes that got a real parsed range.

        Reported by `dwago stats` because it is the single best health signal
        for the substrate: if this collapses, retrieval documents degrade to
        bare identifiers and every downstream number gets worse.
        """
        return (self.spans_matched / self.spans_total) if self.spans_total else 0.0


def load_graph_json(path: str | Path) -> dict:
    """Load and structurally validate graphify's graph.json."""
    p = Path(path)
    if not p.exists():
        raise GraphJsonError(
            f"{p} not found. Build the substrate first:\n"
            f"  graphify update {p.parent.parent}    # code-only, no LLM\n"
            f"  graphify extract {p.parent.parent}   # full, needs an LLM backend"
        )

    size = p.stat().st_size
    if size > MAX_GRAPH_BYTES:
        raise GraphJsonError(
            f"{p} is {size / 1e6:.0f}MB, above the {MAX_GRAPH_BYTES / 1e6:.0f}MB cap."
        )

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GraphJsonError(f"cannot parse {p}: {exc}") from exc

    if not isinstance(data, dict):
        raise GraphJsonError(f"{p}: expected a JSON object, got {type(data).__name__}")
    if "nodes" not in data:
        raise GraphJsonError(
            f"{p}: no 'nodes' key. This does not look like a graphify graph.json "
            f"(found keys: {sorted(data)[:8]})."
        )
    # networkx node-link format calls them 'links'; some graphify paths emit
    # 'edges'. Accept either rather than pinning to one.
    if "links" not in data and "edges" not in data:
        raise GraphJsonError(f"{p}: no 'links' or 'edges' key.")
    if not isinstance(data["nodes"], list):
        raise GraphJsonError(f"{p}: 'nodes' is not a list.")

    return data


def _links_of(data: dict) -> list[dict]:
    return data.get("links") or data.get("edges") or []


# Function-ish labels arrive as `foo()`, `Foo.bar()`, sometimes quoted test
# names, and — for methods whose receiver graphify could not name — as `.foo()`
# with a bare leading dot. That leading dot silently broke every method match
# against parsed spans, so it is stripped here rather than at the call site.
_LABEL_CLEAN = re.compile(r"\(\)\s*$")


def _clean_label(label: str) -> str:
    out = _LABEL_CLEAN.sub("", (label or "").strip()).strip('"\'')
    return out.lstrip(".")


def _infer_kind(node: dict, span_kind: str | None) -> str:
    """Best available kind for a node.

    A parsed span always wins: it comes from the grammar and knows the
    difference between a class and a method. Only when there is no span do we
    fall back to graphify's coarse ``file_type``.
    """
    if span_kind:
        return span_kind
    ft = node.get("file_type") or ""
    label = node.get("label") or ""
    if ft == "code":
        # A label that is exactly a filename is graphify's file-level node.
        if "." in label and "/" not in label and not label.endswith(")"):
            return "file"
        return "symbol"
    return ft or "unknown"


def _start_line(node: dict) -> int | None:
    loc = node.get("source_location")
    if not loc:
        return None
    m = re.match(r"^L(\d+)", str(loc))
    return int(m.group(1)) if m else None


def _read_slice(root: Path, rel_path: str, start: int, end: int,
                _cache: dict[str, list[str]]) -> str:
    """Return source lines [start, end] for content hashing and doc building."""
    if rel_path not in _cache:
        p = root / rel_path
        try:
            _cache[rel_path] = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            _cache[rel_path] = []
    lines = _cache[rel_path]
    if not lines:
        return ""
    lo = max(0, start - 1)
    hi = min(len(lines), end)
    # Cap the slice: a 3000-line class should not put 3000 lines into a hash or
    # a retrieval document. The head carries the identifying content.
    return "\n".join(lines[lo:hi][:200])


def _extend_flat_ranges(rows: list[dict], root: Path,
                        line_cache: dict[str, list[str]]) -> None:
    """Give span-less non-code nodes a range reaching to the next node.

    Only applied where nesting is not a concern. Code nodes are excluded
    outright: their ranges nest, and this rule would make a class swallow its
    own methods' text.
    """
    from collections import defaultdict

    by_file: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if (r["file_type"] != "code"
                and r["start_line"]
                and (r["end_line"] or 0) <= r["start_line"]):
            by_file[r["source_file"]].append(r)

    for path, group in by_file.items():
        if not path:
            continue
        if path not in line_cache:
            try:
                line_cache[path] = (root / path).read_text(
                    encoding="utf-8", errors="replace").splitlines()
            except OSError:
                line_cache[path] = []
        total = len(line_cache[path])
        if not total:
            continue
        group.sort(key=lambda r: r["start_line"])
        for i, r in enumerate(group):
            nxt = group[i + 1]["start_line"] - 1 if i + 1 < len(group) else total
            # Cap the reach: an unbounded final section can otherwise pull in
            # hundreds of lines for one citation.
            r["end_line"] = max(r["start_line"], min(nxt, r["start_line"] + 60, total))


def ingest(
    project_root: str | Path,
    store,
    graph_json: str | Path | None = None,
    *,
    parse_spans: bool = True,
    data: dict | None = None,
) -> IngestResult:
    """Load a graph (own extraction by default, graph.json when given)
    plus parsed spans into ``store``."""
    root = Path(project_root).resolve()

    if data is None:
        if graph_json is not None:
            data = load_graph_json(Path(graph_json))
        else:
            from .extract import extract_repo
            data = extract_repo(root)
    raw_nodes = data["nodes"]
    raw_links = _links_of(data)
    result = IngestResult(built_at_commit=data.get("built_at_commit"))

    if data.get("multigraph"):
        result.warnings.append(
            "graph.json is a multigraph; parallel relations between the same pair "
            "are preserved as separate edges."
        )

    # ── spans ────────────────────────────────────────────────────────────────
    spans_by_file: dict[str, FileSpans] = {}
    if parse_spans:
        spans_by_file = extract_repo_spans(root)
        result.files_parsed = len(spans_by_file)
        log.info("parsed spans for %d files", len(spans_by_file))

    span_index: dict[str, dict] = {f: fs.by_name() for f, fs in spans_by_file.items()}

    # ── nodes ────────────────────────────────────────────────────────────────
    line_cache: dict[str, list[str]] = {}
    rows: list[dict] = []
    idx_of_gid: dict[str, int] = {}
    seen_keys: dict[str, int] = {}

    for i, n in enumerate(raw_nodes):
        gid = n.get("id")
        if gid is None:
            continue
        label = n.get("label") or str(gid)
        src_file = n.get("source_file") or ""
        clean = _clean_label(label)
        parsed_file = spans_by_file.get(src_file)
        source_line = _start_line(n)

        # Repeated names need the original location before the name-only fallback.
        # A declaration line may follow attributes/decorators, so accept its
        # enclosing span, preferring an exact start and then the tightest range.
        span = None
        if parsed_file is not None and source_line is not None:
            span = min(
                (s for s in parsed_file.spans
                 if clean in (s.name, s.qualified_name)
                 and s.start_line <= source_line <= s.end_line),
                key=lambda s: (s.start_line != source_line, s.end_line - s.start_line),
                default=None,
            )
        if span is None:
            span = span_index.get(src_file, {}).get(clean)
        if span is None and src_file in span_index:
            # Second chance: graphify sometimes labels a method bare while the
            # parser qualified it (`Class.method`). Match on the trailing part.
            for qname, s in span_index[src_file].items():
                if qname.rsplit(".", 1)[-1] == clean:
                    span = s
                    break

        kind = _infer_kind(n, span.kind if span else None)

        start = span.start_line if span else source_line
        end = span.end_line if span else start

        # A file-level node has no *symbol* span, but it does have a real
        # extent: the whole file. Giving it one lets citations and diff-hunk
        # mapping treat file nodes like any other.
        if span is None and kind == "file" and parsed_file is not None:
            start, end = 1, parsed_file.lines

        # Coverage is measured only over nodes that could plausibly have a
        # parsed span: symbol nodes in files we actually have a grammar for.
        # File nodes are excluded (they are covered by the branch above) and so
        # are files in languages with no grammar — counting those as misses
        # would report a parser gap as a matching failure.
        if n.get("file_type") == "code" and parsed_file is not None and kind != "file":
            result.spans_total += 1
            if span is not None:
                result.spans_matched += 1
            else:
                # Almost always an imported name (`Path`, `ndarray`) for which
                # graphify makes a node but whose definition lives elsewhere.
                result.external_refs += 1
        elif n.get("file_type") == "code" and parsed_file is None:
            result.unparsed_lang_nodes += 1

        text = ""
        if src_file and start:
            text = _read_slice(root, src_file, start, end or start, line_cache)

        key = node_key(src_file, clean, kind)

        # Distinct graphify nodes can collapse onto one content key (an overload,
        # or the same name at class and module level). Disambiguate with the
        # start line rather than dropping either node.
        if key in seen_keys:
            key = node_key(src_file, clean, kind, start)
            if key in seen_keys:
                # Same file, same name, same kind, same line: graphify emitted
                # the node twice. Collapsing is right, but it is counted and
                # surfaced rather than dropped quietly.
                result.duplicate_nodes += 1
                continue
        seen_keys[key] = i

        idx = len(rows)
        idx_of_gid[gid] = idx
        rows.append({
            "idx": idx,
            "node_key": key,
            "graphify_id": gid,
            "label": label,
            "kind": kind,
            "file_type": n.get("file_type"),
            "source_file": src_file,
            "start_line": start,
            "end_line": end,
            "signature": span.signature if span else "",
            "docstring": span.docstring if span else "",
            "community": n.get("community"),
            "community_name": n.get("community_name"),
            "content_hash": content_hash(text or f"{src_file}:{label}"),
            "doc": None,          # filled by index/docs.py in Phase 2
        })

    # Document nodes (markdown headings, mostly) have a start line and nothing
    # else, so a citation renders as one line of heading text with no content
    # under it. Markdown sections do not nest the way code does, so the flat
    # "runs until the next one starts" rule is sound here — which is exactly the
    # case `approximate_spans` exists for.
    _extend_flat_ranges(rows, root, line_cache)

    store.write_nodes(rows)
    result.nodes = len(rows)

    # ── edges ────────────────────────────────────────────────────────────────
    edge_rows: list[dict] = []
    for e in raw_links:
        # 0.9.x carries true pre-collapse direction in _src/_tgt; older graphs
        # only have source/target. Prefer the explicit pair when present.
        s = e.get("_src") or e.get("source")
        t = e.get("_tgt") or e.get("target")
        si, ti = idx_of_gid.get(s), idx_of_gid.get(t)
        if si is None or ti is None or si == ti:
            # Endpoint missing means the node was deduped away upstream; a
            # self-loop adds nothing to a random walk. Count, do not crash.
            result.dropped_edges += 1
            continue
        edge_rows.append({
            "src": si,
            "dst": ti,
            "relation": e.get("relation") or "related",
            "confidence": e.get("confidence") or "EXTRACTED",
            "confidence_score": float(e.get("confidence_score") or 1.0),
            "weight": float(e.get("weight") or 1.0),
            "source_file": e.get("source_file"),
            "source_location": e.get("source_location"),
        })

    store.write_edges(edge_rows, channel="structural")
    store.build_csr("structural", n_nodes=len(rows), symmetric=True)
    result.edges = len(edge_rows)

    # ── file table ───────────────────────────────────────────────────────────
    file_rows = [
        (fs.path, fs.lines, fs.language)
        for fs in spans_by_file.values()
    ]
    if file_rows:
        store.conn.executemany(
            "INSERT INTO files (path, lines, language) VALUES (?, ?, ?) "
            "ON CONFLICT(path) DO UPDATE SET lines=excluded.lines, "
            "language=excluded.language",
            file_rows,
        )

    store.set_meta("built_at_commit", result.built_at_commit)
    store.set_meta("graph_json_path", str(graph_json) if graph_json else "(own extraction)")
    store.set_meta("span_coverage", round(result.span_coverage, 4))
    store.set_meta("ingest_breakdown", {
        "spans_matched": result.spans_matched,
        "spans_expected": result.spans_total,
        "external_refs": result.external_refs,
        "unparsed_lang_nodes": result.unparsed_lang_nodes,
        "duplicate_nodes": result.duplicate_nodes,
        "files_parsed": result.files_parsed,
    })
    store.set_meta("content_hashes", {r["node_key"]: r["content_hash"] for r in rows})
    store.commit()

    if result.duplicate_nodes:
        result.warnings.append(
            f"{result.duplicate_nodes} duplicate nodes collapsed onto existing keys"
        )
    if result.dropped_edges:
        result.warnings.append(
            f"{result.dropped_edges} edges dropped (missing endpoint or self-loop)"
        )
    if result.spans_total and result.span_coverage < 0.5:
        result.warnings.append(
            f"span coverage is only {result.span_coverage:.0%} — retrieval documents "
            f"will be weak. Check that graph.json was built from this same checkout."
        )

    return result
