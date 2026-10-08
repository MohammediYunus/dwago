"""MCP server exposing dwago to coding agents.

This is a superset of graphify's server, not a first one. graphify already
ships ten tools (`query_graph`, `get_node`, `get_neighbors`, `get_community`,
`god_nodes`, `graph_stats`, `shortest_path`, and three PR tools) over stdio and
HTTP. What it cannot expose is anything behind them: its retrieval is lexical,
and it has no temporal layer at all.

So the tools here fall into two groups, and the docstrings say which is which:
upgrades of something graphify already does (`search`, `explain`, `path`), and
capabilities with no upstream equivalent (`co_change`, `hotspots`, `owners`,
and the temporal half of `impact_of`).

Tool results are returned as compact text rather than JSON blobs. An agent pays
for every token of a tool result, and prose with `file:line` citations is both
smaller and directly usable in an answer.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

__all__ = ["serve", "build_server"]

MAX_RESULTS = 50


def _fmt_hits(hits, limit: int = 20) -> str:
    if not hits:
        return "No results."
    out = []
    for h in hits[:limit]:
        loc = h.location() or "—"
        why = f"  ({h.why})" if h.why else ""
        out.append(f"{h.label}  [{h.kind}]  {loc}{why}")
    return "\n".join(out)


class _Ctx:
    """Lazily opens the store so the server starts even before a build exists."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self._store = None
        self._retriever = None

    @property
    def store(self):
        from .store import Store

        if self._store is None:
            self._store = Store.open(self.root)
        return self._store

    @property
    def retriever(self):
        from .retrieve.hybrid import Retriever

        if self._retriever is None:
            self._retriever = Retriever(self.store)
        return self._retriever


def build_server(root: Path):
    """Construct the MCP server. Imported lazily so `mcp` stays an optional extra."""
    from mcp.server import MCPServer

    ctx = _Ctx(root)
    server = MCPServer(
        name="dwago",
        instructions=(
            "Semantic code search, change-impact analysis and git-history "
            "intelligence for this repository. Prefer context_pack when you need "
            "code to reason over, search when you need to locate something, and "
            "impact_of before changing a symbol."
        ),
    )

    @server.tool(
        description=(
            "Assemble a token-budgeted, citation-carrying context pack for a task. "
            "This is the primary tool: it runs the full retrieval pipeline (exact "
            "match, BM25 + embeddings fused by RRF, then graph diffusion across "
            "code structure and change history) and returns source slices with "
            "file:line citations, packed to fit the budget."
        )
    )
    def context_pack(task: str, token_budget: int = 8000) -> str:
        from .retrieve.pack import build_pack

        res = ctx.retriever.search(task, k=40)
        pack = build_pack(ctx.store, ctx.root, task, res.hits,
                          budget=int(token_budget))
        return pack.render()

    @server.tool(
        description=(
            "Find symbols, files or concepts. Semantic: matches meaning rather "
            "than substrings, so natural-language questions work. Upgrade of "
            "graphify's lexical query."
        )
    )
    def search(query: str, k: int = 15, include_tests: bool = True) -> str:
        r = ctx.retriever
        r.include_tests = include_tests
        res = r.search(query, k=min(k, MAX_RESULTS))
        if not res.hits:
            return f"No results for: {query}"
        return f"{len(res.hits)} results for: {query}\n\n" + _fmt_hits(res.hits, k)

    @server.tool(
        description=(
            "What a change to this symbol reaches: callers and dependents from code "
            "structure, PLUS files connected through co-change history and who "
            "owns them. The historical half has no static equivalent - it catches "
            "coupling no parser can see."
        )
    )
    def impact_of(symbol: str, k: int = 20) -> str:
        k = min(k, MAX_RESULTS)
        res = ctx.retriever.search(symbol, k=k, reverse=True)
        if not res.hits:
            return f"Nothing found for {symbol}."
        static = [h for h in res.hits if "co-change" not in h.why]
        temporal = [h for h in res.hits if "co-change" in h.why]

        parts = [f"Impact of {symbol}:", "", "Reached through code structure:",
                 _fmt_hits(static, k) if static else "  (none)"]
        if temporal:
            parts += ["", "Reached through co-change history:",
                      _fmt_hits(temporal, k)]

        files = {h.source_file for h in res.hits[:10] if h.source_file}
        if files:
            rows = ctx.store.conn.execute(
                "SELECT path, primary_owner, bus_factor FROM files WHERE path IN "
                f"({','.join('?' * len(files))}) AND primary_owner IS NOT NULL",
                list(files)).fetchall()
            if rows:
                parts += ["", "Ownership:"]
                for r in rows:
                    flag = "   <- bus factor 1" if r["bus_factor"] == 1 else ""
                    parts.append(f"  {r['path']}: {r['primary_owner']}{flag}")
        return "\n".join(parts)

    @server.tool(
        description=(
            "Files that historically change together with this one, with the "
            "statistical evidence (times together, lift, p-value). Answers 'what "
            "else do I need to touch?'. No graphify equivalent."
        )
    )
    def co_change(path: str, n: int = 15) -> str:
        rows = ctx.store.conn.execute(
            "SELECT path_a, path_b, support, n_ab, lift, p_value FROM cochange "
            "WHERE path_a = ? OR path_b = ? ORDER BY support DESC LIMIT ?",
            (path, path, min(n, MAX_RESULTS))).fetchall()
        if not rows:
            return (f"No statistically significant co-change partners for {path}. "
                    "Either it changes independently, or history is too short.")
        out = [f"Files that historically change with {path}:", ""]
        for r in rows:
            other = r["path_b"] if r["path_a"] == path else r["path_a"]
            out.append(f"  {other}   (together {r['n_ab']}x, lift {r['lift']:.1f}, "
                       f"p={r['p_value']:.1e})")
        return "\n".join(out)

    @server.tool(
        description=(
            "Files where change concentrates (time-decayed churn x size), with "
            "ownership and bus factor. Use to find risky areas before editing."
        )
    )
    def hotspots(n: int = 20) -> str:
        rows = ctx.store.conn.execute(
            "SELECT path, n_commits, churn, hotspot, bus_factor, primary_owner "
            "FROM files WHERE hotspot > 0 ORDER BY hotspot DESC LIMIT ?",
            (min(n, MAX_RESULTS),)).fetchall()
        if not rows:
            return "No hotspot data. Run `dwago build` inside a git repository."
        out = ["Files where change concentrates (decayed churn x size):", ""]
        for r in rows:
            flag = "   <- bus factor 1" if r["bus_factor"] == 1 else ""
            out.append(f"  {r['path']}  ({r['n_commits']} commits, "
                       f"hotspot {r['hotspot']:.0f}, owner {r['primary_owner']}){flag}")
        return "\n".join(out)

    @server.tool(description="Who has historically worked on a file, and its bus factor.")
    def owners(path: str) -> str:
        f = ctx.store.conn.execute(
            "SELECT bus_factor, n_authors, primary_owner FROM files WHERE path = ?",
            (path,)).fetchone()
        if not f:
            return f"No history recorded for {path}."
        rows = ctx.store.conn.execute(
            "SELECT author, lines FROM ownership WHERE path = ? "
            "ORDER BY lines DESC LIMIT 10", (path,)).fetchall()
        out = [f"{path}: {f['n_authors']} authors, bus factor {f['bus_factor']}", ""]
        out += [f"  {r['author']}  ({r['lines']} commits touching it)" for r in rows]
        if f["bus_factor"] == 1:
            out.append("\nBus factor 1: one person accounts for most of this file.")
        return "\n".join(out)

    @server.tool(description="Describe one symbol: location, signature, docs, neighbours.")
    def explain(symbol: str) -> str:
        res = ctx.retriever.search(symbol, k=1)
        if not res.hits:
            return f"Nothing found for {symbol}."
        h = res.hits[0]
        row = ctx.store.conn.execute(
            "SELECT signature, docstring FROM nodes WHERE idx = ?", (h.idx,)).fetchone()
        out = [f"{h.label}  [{h.kind}]  {h.location()}"]
        if row and row["signature"]:
            out += ["", row["signature"]]
        if row and row["docstring"]:
            out += ["", row["docstring"]]
        nb = ctx.store.conn.execute(
            "SELECT n.label, e.relation FROM edges e JOIN nodes n ON n.idx = e.dst "
            "WHERE e.src = ? AND e.channel='structural' LIMIT 15", (h.idx,)).fetchall()
        if nb:
            out += ["", "Connects to:"]
            out += [f"  {r['relation']} -> {r['label']}" for r in nb]
        return "\n".join(out)

    @server.tool(
        description="Shortest connection between two symbols or files - how A "
                    "reaches B through the graph. Upgrade of graphify's "
                    "shortest_path: BFS over structural + temporal channels.")
    def path(from_symbol: str, to_symbol: str) -> str:
        import numpy as np
        from collections import deque

        st = ctx.store
        con = st.conn

        def find(q: str) -> int | None:
            row = con.execute(
                "SELECT idx FROM nodes WHERE label = ? OR source_file = ? "
                "ORDER BY (kind = 'file') DESC LIMIT 1", (q, q)).fetchone()
            if row:
                return row["idx"]
            row = con.execute(
                "SELECT idx FROM nodes WHERE label LIKE ? OR source_file LIKE ? "
                "LIMIT 1", (f"%{q}%", f"%{q}%")).fetchone()
            return row["idx"] if row else None

        a, b = find(from_symbol), find(to_symbol)
        if a is None or b is None:
            missing = from_symbol if a is None else to_symbol
            return f"No node matching '{missing}'."
        mats = [m for m in (st.load_csr("structural"), st.load_csr("temporal"))
                if m is not None]
        if not mats:
            return "No graph channels built."
        adj = mats[0] if len(mats) == 1 else (mats[0] + mats[1]).tocsr()
        prev = {a: None}
        dq = deque([a])
        while dq:
            u = dq.popleft()
            if u == b:
                break
            row = adj.indices[adj.indptr[u]:adj.indptr[u + 1]]
            for v in row:
                v = int(v)
                if v not in prev:
                    prev[v] = u
                    dq.append(v)
        if b not in prev:
            return f"No path between '{from_symbol}' and '{to_symbol}'."
        chain = []
        cur = b
        while cur is not None:
            chain.append(cur)
            cur = prev[cur]
        chain.reverse()
        out = []
        for idx in chain:
            r = con.execute(
                "SELECT label, kind, source_file, start_line FROM nodes "
                "WHERE idx = ?", (idx,)).fetchone()
            loc = f"{r['source_file']}:{r['start_line']}" if r["start_line"]                 else (r["source_file"] or "-")
            out.append(f"{r['label']}  [{r['kind'] or '?'}]  {loc}")
        return f"{len(chain) - 1} hops:\n" + "\n".join(out)

    @server.tool(
        description="Strongly-connected file clusters - dependency cycles at "
                    "file level. No upstream equivalent at this granularity.")
    def cycles(min_size: int = 2, n: int = 10) -> str:
        from collections import defaultdict

        st = ctx.store
        con = st.conn
        idx_file = {}
        for row in con.execute(
                "SELECT idx, source_file FROM nodes WHERE source_file != ''"):
            idx_file[row["idx"]] = row["source_file"]
        fadj: dict[str, set] = defaultdict(set)
        for row in con.execute("SELECT src, dst FROM edges"):
            fa, fb = idx_file.get(row["src"]), idx_file.get(row["dst"])
            if fa and fb and fa != fb:
                fadj[fa].add(fb)
        # Tarjan SCC, iterative
        index: dict[str, int] = {}
        low: dict[str, int] = {}
        on: set = set()
        stack: list[str] = []
        sccs: list[list[str]] = []
        counter = [0]

        def strongconnect(root: str) -> None:
            work = [(root, iter(sorted(fadj[root])))]
            index[root] = low[root] = counter[0]; counter[0] += 1
            stack.append(root); on.add(root)
            while work:
                v, it = work[-1]
                advanced = False
                for w in it:
                    if w not in index:
                        index[w] = low[w] = counter[0]; counter[0] += 1
                        stack.append(w); on.add(w)
                        work.append((w, iter(sorted(fadj[w]))))
                        advanced = True
                        break
                    elif w in on:
                        low[v] = min(low[v], index[w])
                if advanced:
                    continue
                work.pop()
                if work:
                    low[work[-1][0]] = min(low[work[-1][0]], low[v])
                if low[v] == index[v]:
                    comp = []
                    while True:
                        w = stack.pop(); on.discard(w)
                        comp.append(w)
                        if w == v:
                            break
                    if len(comp) >= min_size:
                        sccs.append(sorted(comp))

        for f in sorted(fadj):
            if f not in index:
                strongconnect(f)
        if not sccs:
            return "No file-level dependency cycles."
        sccs.sort(key=len, reverse=True)
        out = [f"{len(sccs)} cycle cluster(s):"]
        for comp in sccs[:n]:
            out.append(f"  [{len(comp)} files] " + " <-> ".join(comp[:6])
                       + (" ..." if len(comp) > 6 else ""))
        return "\n".join(out)

    @server.tool(
        description="Blast radius of a revision range (e.g. 'HEAD~5..HEAD' or "
                    "a branch): changed files plus their structural and "
                    "co-change neighbourhood, ranked. No upstream equivalent.")
    def diff_impact(rev_range: str = "HEAD~1..HEAD", n: int = 25) -> str:
        import subprocess

        try:
            raw = subprocess.run(
                ["git", "diff", "--name-only", rev_range],
                cwd=ctx.root, capture_output=True, text=True, timeout=30,
                check=True).stdout
        except subprocess.CalledProcessError as e:
            return f"git diff failed: {e.stderr.strip() or rev_range}"
        changed = [f for f in raw.splitlines() if f.strip()]
        if not changed:
            return f"No files changed in {rev_range}."
        con = ctx.store.conn
        changed_set = set(changed)
        idx_file = {}
        file_idxs: dict[str, list[int]] = {}
        for row in con.execute(
                "SELECT idx, source_file FROM nodes WHERE source_file != ''"):
            idx_file[row["idx"]] = row["source_file"]
            file_idxs.setdefault(row["source_file"], []).append(row["idx"])
        from collections import Counter
        impact: Counter = Counter()
        for row in con.execute("SELECT src, dst FROM edges"):
            fa, fb = idx_file.get(row["src"]), idx_file.get(row["dst"])
            if not fa or not fb or fa == fb:
                continue
            if fa in changed_set and fb not in changed_set:
                impact[fb] += 1
            elif fb in changed_set and fa not in changed_set:
                impact[fa] += 1
        out = [f"{len(changed)} files changed in {rev_range}:"]
        out += [f"  {f}" for f in changed[:n]]
        known = [f for f in changed if f in file_idxs]
        if len(known) < len(changed):
            out.append(f"  ({len(changed) - len(known)} not in the graph - "
                       "new or unindexed)")
        if impact:
            out.append("")
            out.append("Ripples into (structural + co-change neighbours):")
            for f, w in impact.most_common(n):
                out.append(f"  {f}  ({w} connection(s) to the change)")
        return "\n".join(out)

    @server.tool(
        description="Test files most coupled to a symbol or file - via imports "
                    "and co-change history. Heuristic until coverage ingestion "
                    "lands; says so in the output.")
    def tests_for(symbol: str, n: int = 10) -> str:
        con = ctx.store.conn
        row = con.execute(
            "SELECT idx, source_file FROM nodes WHERE label = ? OR source_file = ? "
            "ORDER BY (kind = 'file') DESC LIMIT 1", (symbol, symbol)).fetchone()
        if row is None:
            row = con.execute(
                "SELECT idx, source_file FROM nodes WHERE label LIKE ? "
                "OR source_file LIKE ? LIMIT 1",
                (f"%{symbol}%", f"%{symbol}%")).fetchone()
        if row is None:
            return f"No node matching '{symbol}'."
        target = row["source_file"]

        def is_test(f: str) -> bool:
            fl = f.lower()
            return ("test" in fl or "spec" in fl) and not fl.endswith(".md")

        idx_file = {}
        for r2 in con.execute(
                "SELECT idx, source_file FROM nodes WHERE source_file != ''"):
            idx_file[r2["idx"]] = r2["source_file"]
        from collections import Counter
        hits: Counter = Counter()
        for r2 in con.execute("SELECT src, dst FROM edges"):
            fa, fb = idx_file.get(r2["src"]), idx_file.get(r2["dst"])
            if not fa or not fb:
                continue
            if fa == target and is_test(fb):
                hits[fb] += 1
            elif fb == target and is_test(fa):
                hits[fa] += 1
        if not hits:
            return (f"No test files coupled to {target} in the graph. "
                    "(Heuristic: imports + co-change; coverage ingestion "
                    "would make this exact.)")
        out = [f"Tests coupled to {target} (heuristic - imports + co-change):"]
        for f, w in hits.most_common(n):
            out.append(f"  {f}  ({w} link(s))")
        return "\n".join(out)

    @server.tool(
        description="Architecture overview: the largest communities with their "
                    "current cached LLM summaries (run dwago summarize to populate).")
    def overview(n: int = 12) -> str:
        from .summarize import get_summaries

        rows = get_summaries(ctx.store, n)
        if not rows:
            return ("No current community summaries. Run dwago summarize "
                    "with your configured summary backend to generate them.")
        out = []
        for r in rows:
            out.append(f"[{r['community']}] {r['name']}")
            out.append(f"  {r['summary']}")
        return "\n".join(out)

    @server.tool(description="Index health: node/edge counts, span coverage, history depth.")
    def graph_stats() -> str:
        s = ctx.store.stats()
        lines = [f"{k}: {v}" for k, v in s.items()]
        lines.append(f"span_coverage: {ctx.store.get_meta('span_coverage', 0):.1%}")
        model = ctx.store.get_meta("embedding_model")
        if model:
            lines.append(f"embedding_model: {model}")
        commits = ctx.store.get_meta("temporal_commits")
        if commits:
            lines.append(f"history_commits: {commits}")
        return "\n".join(lines)

    return server


def serve(root: Path) -> None:
    """Run the stdio MCP server."""
    try:
        server = build_server(root)
    except ImportError:
        raise SystemExit(
            "MCP support is not installed. See the GitHub installation guide: "
            "https://github.com/MohammediYunus/dwago#install"
        ) from None
    server.run()
