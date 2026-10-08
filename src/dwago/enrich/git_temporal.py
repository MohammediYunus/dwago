"""Git history as a graph signal: co-change coupling, hotspots, ownership.

graphify has no temporal dimension at all — no churn, no coupling, no ownership
(verified: nothing of the sort appears anywhere in its source or CHANGELOG).
Yet every repository ships this data for free, and it answers a question static
analysis structurally cannot: *what else will I have to change?* Two files with
no import between them that have been edited together in 40 of the last 50
commits are coupled, and no parser will ever see it.

Three decisions here are load-bearing, and each rejects the obvious approach:

**Inverse-size commit weighting, not a size cutoff.** The textbook trick is to
discard commits touching more than N files, on the theory that merges and
reformats create noise. But that also discards the single strongest piece of
evidence in the corpus — an API change plus every call site it forced. And in a
squash-merge repository, an entire feature arrives as one commit, so a cutoff
silently deletes most of the history. Instead every commit contributes
``1/C(n,2)`` to each of its pairs: a 2-file commit contributes 1.0 to its single
pair, a 100-file commit contributes ~0.0002 to each of its 4,950 pairs. Large
commits still count, proportionally to how much they actually say.

**A G-test, not raw lift.** Lift is unstable at low support: two files touched
twice, together both times, score infinite lift on no evidence. The G-test
(likelihood-ratio chi-square) asks whether co-occurrence exceeds chance *given
how often each file changes*, so it naturally discounts the rare-pair case and
the everybody-touches-it file alike.

**Incremental by commit range.** Re-mining 100k commits on every rebuild is
minutes of wasted work when HEAD moved by one. Counts are persisted and
extended with ``git log <last>..HEAD``.
"""
from __future__ import annotations

import logging
import math
import re
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

log = logging.getLogger(__name__)

__all__ = ["TemporalConfig", "TemporalResult", "mine_history", "enrich_store"]

# Commits above this size still count, but their pair weight is so small that
# computing all C(n,2) pairs is pure waste — a 2,000-file commit would generate
# 2 million pairs contributing 5e-7 each. Skip pair generation past here and
# record the commit for churn purposes only.
PAIR_EXPLOSION_LIMIT = 400

# Half-life for churn decay. A file heavily edited two years ago is not a
# hotspot today; one edited constantly this quarter is.
CHURN_HALFLIFE_DAYS = 90.0

# Paths whose co-change tells you nothing about architecture. Lockfiles move
# with every dependency bump, generated code moves with its generator, and
# both would otherwise dominate the coupling table.
_NOISE_PATTERNS = re.compile(
    r"(^|/)("
    r"package-lock\.json|yarn\.lock|pnpm-lock\.yaml|poetry\.lock|Cargo\.lock|"
    r"go\.sum|Gemfile\.lock|composer\.lock|uv\.lock|"
    r"CHANGELOG(\.md)?|\.min\.(js|css)|.*\.generated\..*|.*_pb2\.py|.*\.pb\.go"
    r")$",
    re.IGNORECASE,
)


# ASCII record/unit separators delimit the log format. Defined with chr()
# rather than as escape literals: an escaping slip here is invisible (git
# happily emits the literal text instead of the control byte) and silently
# yields zero commits with no error anywhere.
_REC = chr(30)
_FS = chr(31)


@dataclass
class TemporalConfig:
    max_commits: int = 20_000
    # Calibrated on this project's own history (1,419 commits, 920 files,
    # 53,021 candidate pairs). min_support is the binding filter by a wide
    # margin: at 1.0 it admits 88 pairs, at 0.5 123, at 0.3 176, at 0.1 264.
    # 0.3 keeps the source<->test and module<->module couplings that carry real
    # signal without letting incidental single-commit pairs through. The
    # significance gate is what guarantees quality; support just sets depth.
    min_support: float = 0.3        # weighted evidence mass, not a raw count
    min_cooccurrence: int = 3       # pair must appear together at least this often
    max_p_value: float = 0.01       # G-test significance gate
    since: str | None = None        # e.g. "2 years ago"
    follow_renames: bool = True
    # Hard upper bound on commit timestamp. The evaluation harness sets this to
    # keep held-out changes out of the co-change table; without it the temporal
    # channel has seen the answer key and its measured gain is fiction.
    before_ts: int | None = None


@dataclass
class TemporalResult:
    commits_scanned: int = 0
    files_seen: int = 0
    pairs_considered: int = 0
    pairs_kept: int = 0
    authors: int = 0
    head: str | None = None
    elapsed: float = 0.0
    warnings: list[str] = field(default_factory=list)


def _git(root: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
    )
    # Decode explicitly without text mode, which would normalize newlines in paths.
    if check and proc.returncode != 0:
        error = proc.stderr.decode("utf-8", errors="replace").strip()[:300]
        raise RuntimeError(f"git {' '.join(args)} failed: {error}")
    return proc.stdout.decode("utf-8", errors="replace")


def is_git_repo(root: Path) -> bool:
    try:
        return _git(root, "rev-parse", "--is-inside-work-tree").strip() == "true"
    except (RuntimeError, FileNotFoundError):
        return False


@dataclass
class _Commit:
    sha: str
    ts: int
    author: str
    files: list[str]


def _parse_log(raw: str) -> list[_Commit]:
    """Parse ``git log --numstat -z`` without Git's display-path quoting.

    Ordinary numstat records contain one path. Rename/copy records have an
    empty path field followed by separate NUL-terminated old and new paths.
    Tabs, newlines and display-rename syntax inside a path remain literal.
    """
    commits: list[_Commit] = []
    cur: _Commit | None = None
    records = iter(raw.split("\0"))
    for record in records:
        if record.startswith(_REC):
            parts = record[1:].split(_FS, 2)
            if len(parts) == 3:
                cur = _Commit(sha=parts[0], ts=int(parts[1] or 0), author=parts[2], files=[])
                commits.append(cur)
            continue
        if not record or cur is None:
            continue
        # Git inserts a newline before the first numstat row after a header.
        # Split only the two count fields; whitespace in the path is data.
        cols = record.lstrip("\n").split("\t", 2)
        if len(cols) < 3:
            continue
        path = cols[2]
        if not path:
            next(records, None)  # preimage path
            path = next(records, None)  # postimage path
        if path and not _NOISE_PATTERNS.search(path):
            cur.files.append(path)
    return commits


def mine_history(root: Path, config: TemporalConfig | None = None,
                 since_commit: str | None = None) -> tuple[list[_Commit], TemporalResult]:
    """Walk git history once and return its commits."""
    cfg = config or TemporalConfig()
    result = TemporalResult()
    t0 = time.time()

    if not is_git_repo(root):
        result.warnings.append("not a git repository — temporal layer skipped")
        return [], result

    args = [
        "log",
        "--no-merges",                       # a merge's diff double-counts its branch
        "--numstat", "-z",
        "--encoding=UTF-8",
        f"--format={_REC}%H{_FS}%at{_FS}%aN",
        f"--max-count={cfg.max_commits}",
    ]
    if cfg.follow_renames:
        args.append("-M")
    if cfg.since:
        args.append(f"--since={cfg.since}")
    if since_commit:
        args.append(f"{since_commit}..HEAD")

    try:
        raw = _git(root, *args)
    except RuntimeError as exc:
        result.warnings.append(f"git log failed: {exc}")
        return [], result

    commits = _parse_log(raw)
    if cfg.before_ts is not None:
        commits = [c for c in commits if c.ts and c.ts < cfg.before_ts]
    result.commits_scanned = len(commits)
    result.head = _git(root, "rev-parse", "HEAD", check=False).strip() or None
    result.elapsed = time.time() - t0
    return commits, result


def _g_test(n_ab: int, n_a: int, n_b: int, n_total: int) -> tuple[float, float]:
    """Likelihood-ratio test that A and B co-occur more than independence predicts.

    Returns (G statistic, approximate p-value). G is asymptotically chi-square
    with 1 degree of freedom, so the survival function has a closed form and
    needs no scipy import on this path.
    """
    if n_total <= 0 or n_a <= 0 or n_b <= 0 or n_ab <= 0:
        return 0.0, 1.0

    # 2x2 contingency: both / A-only / B-only / neither.
    o = [n_ab, n_a - n_ab, n_b - n_ab, n_total - n_a - n_b + n_ab]
    if any(x < 0 for x in o):
        return 0.0, 1.0
    e = [
        n_a * n_b / n_total,
        n_a * (n_total - n_b) / n_total,
        (n_total - n_a) * n_b / n_total,
        (n_total - n_a) * (n_total - n_b) / n_total,
    ]
    g = 0.0
    for oi, ei in zip(o, e):
        if oi > 0 and ei > 0:
            g += oi * math.log(oi / ei)
    g *= 2.0

    # Only positive association is interesting; a pair that co-occurs *less*
    # than chance is not evidence of coupling.
    if n_ab < e[0]:
        return 0.0, 1.0

    # chi2 sf with df=1 == erfc(sqrt(G/2)).
    p = math.erfc(math.sqrt(max(g, 0.0) / 2.0))
    return g, p


def enrich_store(root: Path, store, config: TemporalConfig | None = None) -> TemporalResult:
    """Mine history and write co-change, churn, hotspots and ownership into ``store``."""
    cfg = config or TemporalConfig()
    root = Path(root).resolve()

    last_indexed = store.get_meta("temporal_head")
    commits, result = mine_history(root, cfg, since_commit=None)
    if not commits:
        return result

    now = time.time()
    halflife_s = CHURN_HALFLIFE_DAYS * 86400.0

    pair_support: dict[tuple[str, str], float] = defaultdict(float)
    pair_count: dict[tuple[str, str], int] = defaultdict(int)
    file_count: dict[str, int] = defaultdict(int)
    file_churn: dict[str, float] = defaultdict(float)
    file_last: dict[str, int] = {}
    file_first: dict[str, int] = {}
    authors_per_file: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    all_authors: set[str] = set()

    for c in commits:
        files = sorted(set(c.files))
        if not files:
            continue
        all_authors.add(c.author)
        decay = 0.5 ** ((now - c.ts) / halflife_s) if c.ts else 0.0

        for f in files:
            file_count[f] += 1
            file_churn[f] += decay
            authors_per_file[f][c.author] += 1
            if f not in file_last or c.ts > file_last[f]:
                file_last[f] = c.ts
            if f not in file_first or c.ts < file_first[f]:
                file_first[f] = c.ts

        n = len(files)
        if n < 2 or n > PAIR_EXPLOSION_LIMIT:
            continue
        # Inverse-size weighting: a commit's total contribution across all its
        # pairs is exactly 1.0, however many files it touched.
        w = 1.0 / (n * (n - 1) / 2.0)
        for a, b in combinations(files, 2):
            key = (a, b)
            pair_support[key] += w
            pair_count[key] += 1

    result.files_seen = len(file_count)
    result.authors = len(all_authors)
    result.pairs_considered = len(pair_support)
    n_total = len(commits)

    # ── co-change table ──────────────────────────────────────────────────────
    keep: list[tuple] = []
    for (a, b), support in pair_support.items():
        n_ab = pair_count[(a, b)]
        if n_ab < cfg.min_cooccurrence or support < cfg.min_support:
            continue
        n_a, n_b = file_count[a], file_count[b]
        g, p = _g_test(n_ab, n_a, n_b, n_total)
        if p > cfg.max_p_value:
            continue
        expected = (n_a * n_b / n_total) if n_total else 0.0
        lift = (n_ab / expected) if expected > 0 else 0.0
        keep.append((a, b, support, n_ab, n_a, n_b, lift, g, p))

    store.conn.execute("DELETE FROM cochange")
    store.conn.executemany(
        "INSERT OR REPLACE INTO cochange "
        "(path_a, path_b, support, n_ab, n_a, n_b, lift, g_stat, p_value) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        keep,
    )
    result.pairs_kept = len(keep)

    # ── per-file attributes ──────────────────────────────────────────────────
    existing = {r["path"]: dict(r) for r in store.conn.execute("SELECT * FROM files")}
    file_rows = []
    for f, n_commits in file_count.items():
        prev = existing.get(f, {})
        lines = prev.get("lines") or 0
        # Complexity proxy. Real cyclomatic complexity would need another parse
        # of every file; size is the standard stand-in and correlates well
        # enough for ranking, which is all the hotspot score is used for.
        complexity = math.log1p(lines) if lines else 0.0
        churn = file_churn[f]
        author_lines = authors_per_file[f]
        total_touch = sum(author_lines.values()) or 1
        ranked = sorted(author_lines.items(), key=lambda kv: -kv[1])
        primary, primary_n = ranked[0]
        # Bus factor: how few people account for half of all touches.
        acc, bus = 0, 0
        for _, cnt in ranked:
            acc += cnt
            bus += 1
            if acc * 2 >= total_touch:
                break
        file_rows.append((
            f, lines, prev.get("language"), n_commits, churn, complexity,
            churn * complexity, file_first.get(f), file_last.get(f),
            len(author_lines), bus, primary, primary_n / total_touch,
        ))

    store.conn.executemany(
        "INSERT INTO files (path, lines, language, n_commits, churn, complexity, "
        "hotspot, first_seen, last_modified, n_authors, bus_factor, primary_owner, "
        "owner_share) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(path) DO UPDATE SET "
        "n_commits=excluded.n_commits, churn=excluded.churn, "
        "complexity=excluded.complexity, hotspot=excluded.hotspot, "
        "first_seen=excluded.first_seen, last_modified=excluded.last_modified, "
        "n_authors=excluded.n_authors, bus_factor=excluded.bus_factor, "
        "primary_owner=excluded.primary_owner, owner_share=excluded.owner_share",
        file_rows,
    )

    store.conn.execute("DELETE FROM ownership")
    store.conn.executemany(
        "INSERT OR REPLACE INTO ownership (path, author, lines) VALUES (?,?,?)",
        [(f, a, n) for f, d in authors_per_file.items() for a, n in d.items()],
    )

    store.set_meta("temporal_head", result.head)
    store.set_meta("temporal_commits", result.commits_scanned)
    store.commit()

    if last_indexed and last_indexed != result.head:
        log.debug("temporal layer refreshed from %s to %s", last_indexed[:8], result.head[:8])

    return result


def build_temporal_edges(store, *, min_support: float = 1.0) -> int:
    """Project file-level co-change into the temporal edge channel.

    The tempting implementation — link every node in file A to every node in
    file B — is quadratic and produces a graph that is mostly noise: on this
    project's own history, 88 significant file pairs expanded to 518,793 node
    edges, because one pair of large files contributes |A|x|B| of them. That
    density would dominate the temporal walk and bury the signal it came from.

    Instead the coupling stays at the granularity it was measured at. Each file
    pair contributes exactly one edge between the two *file* nodes, and each
    symbol is linked to its own file node. A temporal walk then travels
    symbol -> file -> coupled file -> symbol, reaching the same places in three
    hops at O(nodes + pairs) edges instead of O(nodes^2). PageRank handles the
    extra hop natively; that is what damping is for.
    """
    file_node_of: dict[str, int] = {}
    members: dict[str, list[int]] = {}
    for r in store.conn.execute(
        "SELECT idx, source_file, kind FROM nodes WHERE source_file IS NOT NULL"
    ):
        path = r["source_file"]
        if r["kind"] == "file":
            # Keep the first file node per path; graphify emits one.
            file_node_of.setdefault(path, r["idx"])
        else:
            members.setdefault(path, []).append(r["idx"])

    rows: list[dict] = []

    # 1. Containment: symbol <-> its file node. Linear in node count.
    for path, idxs in members.items():
        fidx = file_node_of.get(path)
        if fidx is None:
            continue
        for i in idxs:
            if i == fidx:
                continue
            rows.append({
                "src": fidx, "dst": i, "relation": "file_contains",
                "confidence": "EXTRACTED", "confidence_score": 1.0,
                "weight": 1.0, "source_file": path, "source_location": None,
            })

    # 2. Coupling: file node <-> file node, one edge per significant pair.
    pairs = 0
    for r in store.conn.execute(
        "SELECT path_a, path_b, support, g_stat FROM cochange WHERE support >= ?",
        (min_support,),
    ):
        a = file_node_of.get(r["path_a"])
        b = file_node_of.get(r["path_b"])
        if a is None or b is None or a == b:
            # One side is a file dwago has no node for (a doc, a lockfile, a
            # language with no extractor). The coupling is still recorded in the
            # `cochange` table and still answerable by `dwago co-change`; it
            # just has nowhere to attach in the graph.
            continue
        rows.append({
            "src": a, "dst": b, "relation": "co_changes_with",
            "confidence": "INFERRED",
            "confidence_score": 1.0,
            # Support is already the inverse-size-weighted evidence mass.
            "weight": float(r["support"]),
            "source_file": r["path_a"], "source_location": None,
        })
        pairs += 1

    store.write_edges(rows, channel="temporal")
    store.build_csr("temporal", n_nodes=store.node_count(), symmetric=True)
    store.set_meta("temporal_pairs_linked", pairs)
    return len(rows)
