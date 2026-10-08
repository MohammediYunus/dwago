"""The brain cockpit over real data.

Collapses the store to file level and injects the payload into the
self-contained WebGL brain (assets/brain.html): the eight largest
communities become the cortical regions, files become neurons, structural
and co-change edges become the axons the strikes travel, and hotspot
scores pick the KEY FILES overlay.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from html import escape
from pathlib import Path

from ..store import Store

_TEMPLATE = Path(__file__).parent / "assets" / "brain.html"
_MARKER = "/*__DWAGO_DATA__*/"

MAX_EDGES_PER_FILE = 8
MAX_EDGES_TOTAL = 12_000
KEY_FILES = 12


def build_payload(store: Store, *, max_files: int | None = None) -> dict:
    con = store.conn

    # ── files: majority community per source_file ────────────────────────
    file_comm: dict[str, Counter] = defaultdict(Counter)
    file_cname: dict[str, Counter] = defaultdict(Counter)
    idx_file: dict[int, str] = {}
    for row in con.execute(
        "SELECT idx, source_file, community, community_name FROM nodes "
        "WHERE source_file IS NOT NULL AND source_file != ''"
    ):
        f = row["source_file"]
        idx_file[row["idx"]] = f
        if row["community"] is not None:
            file_comm[f][row["community"]] += 1
        if row["community_name"]:
            file_cname[f][row["community_name"]] += 1
    files = sorted(file_comm)

    # ── hotspots ─────────────────────────────────────────────────────────
    hot: dict[str, float] = {}
    try:
        for row in con.execute("SELECT path, hotspot FROM files"):
            hot[row["path"]] = row["hotspot"] or 0.0
    except Exception:
        pass

    # ── regions: 7 largest communities + everything else ─────────────────
    comm_files: Counter = Counter()
    for f in files:
        comm_files[file_comm[f].most_common(1)[0][0]] += 1
    top = [c for c, _ in comm_files.most_common(7)]
    region_of_comm = {c: i for i, c in enumerate(top)}

    def region_name(comm: int) -> str:
        names = Counter()
        for f in files:
            if file_comm[f].most_common(1)[0][0] == comm and file_cname[f]:
                names[file_cname[f].most_common(1)[0][0]] += 1
        return (names.most_common(1)[0][0] if names else f"community {comm}")[:26]

    else_region = len(top)
    regions = [{"name": region_name(c)} for c in top] + [{"name": "everything else"}]

    # ── rank files (hotspot desc, then degree later); optional cap ───────
    if max_files and len(files) > max_files:
        files = sorted(files, key=lambda f: -(hot.get(f, 0.0)))[:max_files]
        files.sort()
    file_idx = {f: i for i, f in enumerate(files)}

    payload_files = []
    for f in files:
        comm = file_comm[f].most_common(1)[0][0]
        payload_files.append({
            "p": f,
            "r": region_of_comm.get(comm, else_region),
            "h": round(hot.get(f, 0.0), 2),
        })

    # ── edges: structural + temporal collapsed to file pairs ─────────────
    pair_w: Counter = Counter()
    for row in con.execute("SELECT src, dst FROM edges"):
        fa, fb = idx_file.get(row["src"]), idx_file.get(row["dst"])
        if not fa or not fb or fa == fb:
            continue
        ia, ib = file_idx.get(fa), file_idx.get(fb)
        if ia is None or ib is None:
            continue
        pair_w[(min(ia, ib), max(ia, ib))] += 1

    per_file: Counter = Counter()
    edges: list[list[int]] = []
    for (a, b), _w in pair_w.most_common():
        if per_file[a] >= MAX_EDGES_PER_FILE and per_file[b] >= MAX_EDGES_PER_FILE:
            continue
        edges.append([a, b])
        per_file[a] += 1
        per_file[b] += 1
        if len(edges) >= MAX_EDGES_TOTAL:
            break

    # ── key files: hotspot first, degree as the tie-breaker ──────────────
    deg = Counter()
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    key = sorted(range(len(files)),
                 key=lambda i: (-(payload_files[i]["h"]), -deg[i]))[:KEY_FILES]

    return {"regions": regions, "files": payload_files,
            "edges": edges, "key": key}


def write_brain(store: Store, out: Path, *, title: str = "dwago",
                max_files: int | None = None) -> int:
    payload = build_payload(store, max_files=max_files)
    # Keep graph text inside the script element, including HTML comment markers.
    payload_json = json.dumps(payload, separators=(',', ':')).replace("<", "\\u003c")
    html = _TEMPLATE.read_text(encoding="utf-8")
    html = html.replace(_MARKER,
                        f"window.DWAGO_DATA={payload_json};")
    html = html.replace("<title>The Brain</title>", f"<title>{escape(title)} · brain</title>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return len(payload["files"])
