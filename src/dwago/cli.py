"""dwago command line — the single place orchestration lives.

The skill file deliberately contains no shell pipelines. graphify's SKILL.md is
~41KB of bash that the model executes step by step, which costs context on every
invocation, drifts from the library it drives, and fails in ways the model has
to diagnose mid-task. Here the agent calls one command and reads one result;
everything sequencing-related is code, where it can be tested.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

log = logging.getLogger("dwago")

__all__ = ["main"]


# Libraries that log at INFO on import or first use. huggingface_hub in
# particular narrates every HTTP request, which buries the build output.
_NOISY = ("httpx", "httpcore", "urllib3", "huggingface_hub", "filelock",
          "sentence_transformers", "transformers", "jieba", "numba")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        stream=sys.stderr,
    )
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)


def _fmt_int(n: int) -> str:
    return f"{n:,}"


# ── build ────────────────────────────────────────────────────────────────────

class _EmbeddingConfirmationRequired(Exception):
    """Roll back an unfinished index before requesting a longer build."""


def cmd_build(args) -> int:
    from .enrich.git_temporal import (TemporalConfig, build_temporal_edges,
                                      enrich_store)
    from .index.dense import Embedder, build_dense_index, resolve_backend
    from .index.docs import build_documents
    from .index.lexical import LexicalIndex
    from .ingest import GraphJsonError, ingest
    from .store import Store

    root = Path(args.path).resolve()
    t0 = time.time()

    backend = resolve_backend("fast" if args.fast else args.embed_backend, args.model)
    if backend.kind == "none":
        print("! no embedding backend available — retrieval will be BM25 only.")
        print("  install one:  pip install 'dwago[dense]'   (best quality)")
        print("                pip install 'dwago[fast]'    (pure numpy, ~30MB)")
    else:
        print(f"Embeddings: {backend.model} ({backend.kind} on {backend.device})")

    try:
        with Store.begin(root, inherit=not args.force) as st:
            if args.graph:
                print(f"Ingesting {args.graph}...")
            else:
                print(f"Extracting {root}...")
            res = ingest(root, st, args.graph)
            print(f"  {_fmt_int(res.nodes)} nodes · {_fmt_int(res.edges)} edges · "
                  f"span coverage {res.span_coverage:.0%}")
            for w in res.warnings:
                print(f"  ! {w}")

            if not args.no_git:
                print("Mining git history...")
                tr = enrich_store(root, st, TemporalConfig(max_commits=args.max_commits))
                if tr.warnings:
                    for w in tr.warnings:
                        print(f"  ! {w}")
                else:
                    print(f"  {_fmt_int(tr.commits_scanned)} commits · "
                          f"{_fmt_int(tr.files_seen)} files · {tr.authors} authors")
                    print(f"  {_fmt_int(tr.pairs_kept)} significant co-change pairs "
                          f"(of {_fmt_int(tr.pairs_considered)} considered)")
                    n = build_temporal_edges(st)
                    print(f"  {_fmt_int(n)} temporal edges")

            print("Building retrieval documents...")
            n_docs = build_documents(st)
            print(f"  {_fmt_int(n_docs)} documents")

            print("Building BM25 index...")
            LexicalIndex.build(st, st.paths.bm25())

            if backend.kind != "none":
                emb = Embedder(backend)
                est = emb.estimate_seconds(n_docs)
                if est > 120 and not args.yes:
                    print(f"  embedding ~{_fmt_int(n_docs)} docs on {backend.device} "
                          f"will take roughly {est/60:.0f} min.")
                    print("  re-run with --yes to proceed, or --fast for the "
                          "pure-numpy encoder.")
                    raise _EmbeddingConfirmationRequired
                print("Embedding...")
                info = build_dense_index(st, emb, force=args.force)
                emb.release()
                if info.get("embedded"):
                    print(f"  {_fmt_int(info['embedded'])} embedded, "
                          f"{_fmt_int(info.get('reused', 0))} reused "
                          f"({info.get('seconds', 0)}s)")
                else:
                    print(f"  all {_fmt_int(info.get('reused', 0))} vectors reused")
    except _EmbeddingConfirmationRequired:
        return 2
    except GraphJsonError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"\nBuilt in {time.time() - t0:.1f}s → {root / 'dwago-out'}")
    return 0


def cmd_refresh(args) -> int:
    """Incremental rebuild. Same pipeline; the dirty-set logic does the saving."""
    args.force = False
    return cmd_build(args)


# ── query ────────────────────────────────────────────────────────────────────

def _load_retriever(root: Path, args):
    from .retrieve.hybrid import Retriever
    from .store import Store

    st = Store.open(root)
    return st, Retriever(
        st,
        temporal_weight=getattr(args, "temporal_weight", 0.3),
        include_tests=not getattr(args, "no_tests", False),
    )


def cmd_ask(args) -> int:
    root = Path(args.path).resolve()
    st, r = _load_retriever(root, args)
    res = r.search(args.query, k=args.k, use_ppr=not args.no_ppr, rerank=args.rerank)

    if args.json:
        print(json.dumps({
            "query": args.query,
            "stages": res.stages,
            "hits": [vars(h) for h in res.hits],
            "warnings": res.warnings,
        }, indent=2))
        return 0

    for w in res.warnings:
        print(f"! {w}")
    if not res.hits:
        return 1

    stages = " · ".join(f"{k}={v}" for k, v in res.stages.items())
    print(f"{len(res.hits)} results   [{stages}]\n")
    for i, h in enumerate(res.hits, 1):
        loc = h.location()
        print(f"{i:2}. {h.label[:56]:58} {loc}")
        if h.why:
            print(f"    {h.kind} · {h.why} · {h.score:.3f}")
    return 0


def cmd_pack(args) -> int:
    from .retrieve.pack import build_pack

    root = Path(args.path).resolve()
    st, r = _load_retriever(root, args)
    res = r.search(args.query, k=args.k, rerank=args.rerank)
    pack = build_pack(st, root, args.query, res.hits, budget=args.budget)
    print(pack.render())
    return 0


def cmd_impact(args) -> int:
    """What a change to this symbol reaches — statically and historically."""
    root = Path(args.path).resolve()
    st, r = _load_retriever(root, args)
    # Reverse orientation: we want what depends on this, not what it depends on.
    res = r.search(args.symbol, k=args.k, reverse=True)

    print(f"Impact of: {args.symbol}\n")
    static, temporal = [], []
    for h in res.hits:
        (temporal if "co-change" in h.why else static).append(h)

    if static:
        print("Reached through code structure:")
        for h in static[:args.k]:
            print(f"  {h.label[:50]:52} {h.location()}")
    if temporal:
        print("\nHistorically changes alongside:")
        for h in temporal[:args.k]:
            print(f"  {h.label[:50]:52} {h.location()}")
    if not res.hits:
        print("  nothing found — is the symbol name right?")
        return 1
    return 0


# ── temporal reporting ───────────────────────────────────────────────────────

def cmd_cochange(args) -> int:
    from .store import Store

    st = Store.open(Path(args.path).resolve())
    rows = st.conn.execute(
        "SELECT path_a, path_b, support, n_ab, lift, g_stat, p_value FROM cochange "
        "WHERE path_a = ? OR path_b = ? ORDER BY support DESC LIMIT ?",
        (args.file, args.file, args.n),
    ).fetchall()
    if not rows:
        print(f"no significant co-change partners for {args.file}")
        return 1
    print(f"Files that historically change with {args.file}:\n")
    print(f"  {'partner':<52} {'support':>8} {'together':>9} {'lift':>7} {'p':>9}")
    for r in rows:
        other = r["path_b"] if r["path_a"] == args.file else r["path_a"]
        print(f"  {other[:50]:<52} {r['support']:>8.2f} {r['n_ab']:>9} "
              f"{r['lift']:>7.1f} {r['p_value']:>9.2e}")
    return 0


def cmd_hotspots(args) -> int:
    from .store import Store

    st = Store.open(Path(args.path).resolve())
    rows = st.conn.execute(
        "SELECT path, n_commits, churn, complexity, hotspot, bus_factor, "
        "n_authors, primary_owner, owner_share FROM files "
        "WHERE hotspot > 0 ORDER BY hotspot DESC LIMIT ?", (args.n,),
    ).fetchall()
    if not rows:
        print("no hotspot data — run `dwago build` in a git repository")
        return 1
    print("Files where change concentrates (decayed churn x size):\n")
    print(f"  {'file':<46} {'commits':>8} {'churn':>7} {'hotspot':>8} {'bus':>4} {'owner':>18}")
    for r in rows:
        owner = (r["primary_owner"] or "")[:16]
        print(f"  {r['path'][:44]:<46} {r['n_commits']:>8} {r['churn']:>7.1f} "
              f"{r['hotspot']:>8.1f} {r['bus_factor']:>4} {owner:>18}")
    risky = [r for r in rows if r["bus_factor"] == 1]
    if risky:
        print(f"\n  {len(risky)} of these have a bus factor of 1.")
    return 0


def cmd_stats(args) -> int:
    from .store import Store

    st = Store.open(Path(args.path).resolve())
    s = st.stats()
    print(f"dwago index — epoch {s['epoch']}\n")
    for k in ("nodes", "edges_structural", "edges_temporal", "vectors",
              "files", "cochange_pairs", "communities", "summaries"):
        print(f"  {k:<20} {_fmt_int(s[k])}")
    bd = st.get_meta("ingest_breakdown") or {}
    if bd:
        print(f"\n  span coverage        {st.get_meta('span_coverage', 0):.1%} "
              f"({_fmt_int(bd.get('spans_matched', 0))}/"
              f"{_fmt_int(bd.get('spans_expected', 0))})")
        print(f"  external refs        {_fmt_int(bd.get('external_refs', 0))}")
        print(f"  unparsed-language    {_fmt_int(bd.get('unparsed_lang_nodes', 0))}")
    model = st.get_meta("embedding_model")
    if model:
        print(f"\n  embeddings           {model}")
    commits = st.get_meta("temporal_commits")
    if commits:
        print(f"  history              {_fmt_int(commits)} commits")
    return 0


def cmd_serve(args) -> int:
    from .serve import serve

    root = Path(args.path).resolve()
    from .store import Store
    if not Store.exists(root):
        print(f"error: no index at {root}. Run `dwago build` first.", file=sys.stderr)
        return 1
    log.info("dwago MCP server on %s", root)
    serve(root)
    return 0


def cmd_eval(args) -> int:
    from .enrich.git_temporal import (TemporalConfig, build_temporal_edges,
                                      enrich_store)
    from .eval.harness import (build_eval_set, format_report, run_ladder,
                               split_commit)
    from .store import Store

    root = Path(args.path).resolve()

    try:
        cutoff_sha, cutoff_ts = split_commit(root, fraction=args.split)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    items = build_eval_set(root, after_ts=cutoff_ts, limit=args.n, seed=args.seed)
    if len(items) < 20:
        print(f"error: only {len(items)} scoreable changes after the cutoff — "
              f"lower --split or raise -n", file=sys.stderr)
        return 1

    print(f"Held-out set: {len(items)} changes after {cutoff_sha[:8]} "
          f"({args.split:.0%} of history used for the index)")

    if not args.no_rebuild:
        # Rebuild the temporal layer with the cutoff applied, so co-change has
        # not seen any commit being evaluated.
        print("Rebuilding temporal layer with the evaluation window excluded...")
        with Store.begin(root) as st:
            tr = enrich_store(root, st,
                              TemporalConfig(max_commits=args.max_commits,
                                             before_ts=cutoff_ts))
            build_temporal_edges(st)
            print(f"  {tr.commits_scanned:,} commits before cutoff · "
                  f"{tr.pairs_kept:,} co-change pairs")

    st = Store.open(root)
    rungs = args.rungs.split(",") if args.rungs else None
    results = run_ladder(st, root, items, rungs=rungs,
                         temporal_weight=args.temporal_weight)
    report = format_report(results)
    print(report)

    if args.out:
        Path(args.out).write_text(json.dumps({
            "cutoff": cutoff_sha, "n_items": len(items),
            "rungs": [
                {"name": r.name, "recall": r.recall_at, "mrr": r.mrr,
                 "seconds_per_query": r.seconds_per_query, "n": r.n, "note": r.note}
                for r in results
            ],
        }, indent=2))
        print(f"wrote {args.out}")
    return 0


def cmd_summarize(args) -> int:
    from .store import Store
    from .summarize import summarize_communities

    root = Path(args.path).resolve()
    st = Store.open(root)
    r = summarize_communities(st, top=args.n, backend=args.backend,
                              model=args.model)
    print(f"summaries: {r['written']} written, {r['cached']} cached")
    for e in r["errors"][:5]:
        print(f"  ! {e}")
    if r["errors"] and r["written"] == 0:
            print("  (no backend available — set OPENAI_API_KEY or "
              "ANTHROPIC_API_KEY, or authenticate a local claude CLI)")
    return 0 if not r["errors"] or r["written"] else 1


def cmd_map(args) -> int:
    from .store import Store

    root = Path(args.path).resolve()
    st = Store.open(root)
    if args.galaxy:
        from .viz.build_html import write_html
        out = Path(args.out) if args.out else root / "dwago-out" / "map.html"
        n = write_html(st, out, title=args.title or root.name,
                       max_nodes=args.max_nodes)
        print(f"Wrote {out} — {_fmt_int(n)} nodes, "
              f"{out.stat().st_size / 1e6:.1f}MB, self-contained.")
        return 0
    from .viz.build_brain import write_brain
    out = Path(args.out) if args.out else root / "dwago-out" / "brain.html"
    n = write_brain(st, out, title=args.title or root.name,
                    max_files=args.max_files)
    print(f"Wrote {out} — {_fmt_int(n)} files as neurons, "
          f"{out.stat().st_size / 1e6:.1f}MB, self-contained.")
    return 0


# ── argument wiring ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dwago",
        description="Retrieval, git-temporal intelligence and GPU visualization "
                    "over a code knowledge graph it builds itself.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("path", nargs="?", default=".", help="project root")
        return sp

    b = add_common(sub.add_parser("build", help="build or rebuild the index"))
    b.add_argument("--graph", help="ingest an existing graph.json instead of extracting")
    b.add_argument("--fast", action="store_true",
                   help="static embeddings, no torch — minutes instead of hours")
    b.add_argument("--embed-backend", default="auto",
                   choices=["auto", "st", "fast", "none"])
    b.add_argument("--model", help="override the embedding model")
    b.add_argument("--no-git", action="store_true", help="skip the temporal layer")
    b.add_argument("--max-commits", type=int, default=20_000)
    b.add_argument("--force", action="store_true", help="rebuild from scratch")
    b.add_argument("--yes", action="store_true", help="skip the long-build prompt")
    b.set_defaults(func=cmd_build)

    r = add_common(sub.add_parser("refresh", help="incremental rebuild after commits"))
    for a in ("--graph", "--model"):
        r.add_argument(a)
    r.add_argument("--fast", action="store_true")
    r.add_argument("--embed-backend", default="auto",
                   choices=["auto", "st", "fast", "none"])
    r.add_argument("--no-git", action="store_true")
    r.add_argument("--max-commits", type=int, default=20_000)
    r.add_argument("--yes", action="store_true")
    r.set_defaults(func=cmd_refresh)

    a = sub.add_parser("ask", help="search the graph")
    a.add_argument("query")
    a.add_argument("path", nargs="?", default=".")
    a.add_argument("-k", type=int, default=20)
    a.add_argument("--no-ppr", action="store_true", help="skip graph diffusion")
    a.add_argument("--rerank", action="store_true", help="cross-encoder rerank (slow on CPU)")
    a.add_argument("--temporal-weight", type=float, default=0.3)
    a.add_argument("--no-tests", action="store_true",
                       help="exclude test files from results")
    a.add_argument("--json", action="store_true")
    a.set_defaults(func=cmd_ask)

    pk = sub.add_parser("pack", help="build a token-budgeted context pack")
    pk.add_argument("query")
    pk.add_argument("path", nargs="?", default=".")
    pk.add_argument("--budget", type=int, default=8000)
    pk.add_argument("-k", type=int, default=40)
    pk.add_argument("--rerank", action="store_true")
    pk.add_argument("--temporal-weight", type=float, default=0.3)
    pk.add_argument("--no-tests", action="store_true",
                       help="exclude test files from results")
    pk.set_defaults(func=cmd_pack)

    im = sub.add_parser("impact", help="what a change to this symbol reaches")
    im.add_argument("symbol")
    im.add_argument("path", nargs="?", default=".")
    im.add_argument("-k", type=int, default=15)
    im.set_defaults(func=cmd_impact)

    cc = sub.add_parser("co-change", help="files that historically change together")
    cc.add_argument("file")
    cc.add_argument("path", nargs="?", default=".")
    cc.add_argument("-n", type=int, default=15)
    cc.set_defaults(func=cmd_cochange)

    hs = add_common(sub.add_parser("hotspots", help="where change concentrates"))
    hs.add_argument("-n", type=int, default=20)
    hs.set_defaults(func=cmd_hotspots)

    stt = add_common(sub.add_parser("stats", help="index health"))
    stt.set_defaults(func=cmd_stats)

    ev = add_common(sub.add_parser("eval", help="benchmark retrieval on this repo's history"))
    ev.add_argument("-n", type=int, default=200, help="held-out changes to score")
    ev.add_argument("--split", type=float, default=0.8,
                    help="fraction of history used to build the index")
    ev.add_argument("--rungs", help="comma-separated (bm25,dense,hybrid,hybrid+ppr)")
    ev.add_argument("--temporal-weight", type=float, default=0.3)
    ev.add_argument("--max-commits", type=int, default=20_000)
    ev.add_argument("--no-rebuild", action="store_true",
                    help="skip the leak-free temporal rebuild (results will be optimistic)")
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--out", help="write JSON results here")
    ev.set_defaults(func=cmd_eval)

    sv = add_common(sub.add_parser("serve", help="run the MCP server (stdio)"))
    sv.add_argument("--mcp", action="store_true", help="accepted for symmetry; stdio is the only transport")
    sv.set_defaults(func=cmd_serve)

    mp = add_common(sub.add_parser("map", help="write the WebGL visualization"))
    mp.add_argument("--out", help="output HTML path")
    mp.add_argument("--title")
    mp.add_argument("--max-nodes", type=int, default=250_000)
    sm = add_common(sub.add_parser("summarize",
                                    help="LLM summaries for the largest communities"))
    sm.add_argument("-n", type=int, default=20)
    sm.add_argument("--backend", default="auto",
                    choices=["auto", "openai", "anthropic", "claude-cli", "none"])
    sm.add_argument("--model", default=None,
                    help="model id for the chosen backend")
    sm.set_defaults(func=cmd_summarize)

    mp.add_argument("--galaxy", action="store_true",
                    help="the symbol-level galaxy map instead of the brain")
    mp.add_argument("--max-files", type=int, default=4_000,
                    help="cap the brain's neurons to the hottest N files")
    mp.set_defaults(func=cmd_map)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
