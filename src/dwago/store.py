"""On-disk store: SQLite metadata + numpy CSR graph + float16 vector memmap.

Three deliberate choices, all made against alternatives that look better on paper:

*No embedded graph database.* Kùzu was the obvious candidate and is now archived
(Apple acquisition, Oct 2025); its forks are months old. Meanwhile the only graph
operation on the hot path is a sparse matrix-vector product for Personalized
PageRank, which scipy does in single-digit milliseconds at 100k nodes. A Cypher
engine would add a dependency and a migration story to buy nothing.

*Epoch directories with an atomic symlink swap.* graphify rewrites graph.json on
every commit via its post-commit hook, so a rebuild can easily overlap a query.
Writing in place would let `ask` read a half-written index. Instead each build
lands in ``epochs/<n>/`` and becomes visible only when a symlink flips, which is
a single atomic rename. Readers either see the whole old epoch or the whole new
one, never a mix.

*Vectors in a memmap keyed by content-addressed node key, with tombstones.*
Re-embedding 100k nodes with a 0.6B model takes hours, so a refresh must touch
only what changed. Rows are appended and stale rows tombstoned rather than
compacted, because brute-force cosine over a masked array costs nothing extra
and compaction would invalidate every row index.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

log = logging.getLogger(__name__)

__all__ = ["Store", "OUT_DIR_NAME", "SCHEMA_VERSION"]

OUT_DIR_NAME = "dwago-out"
SCHEMA_VERSION = 1

_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;

-- One row per graph node. `idx` is the dense row index used by the CSR arrays
-- and the vector memmap, so it must stay contiguous from 0.
CREATE TABLE IF NOT EXISTS nodes (
    idx            INTEGER PRIMARY KEY,
    node_key       TEXT    NOT NULL UNIQUE,
    graphify_id    TEXT,
    label          TEXT    NOT NULL,
    kind           TEXT,
    file_type      TEXT,
    source_file    TEXT,
    start_line     INTEGER,
    end_line       INTEGER,
    signature      TEXT,
    docstring      TEXT,
    community      INTEGER,
    community_name TEXT,
    content_hash   TEXT,
    doc            TEXT           -- the assembled retrieval document
);
CREATE INDEX IF NOT EXISTS idx_nodes_file  ON nodes(source_file);
CREATE INDEX IF NOT EXISTS idx_nodes_gid   ON nodes(graphify_id);
CREATE INDEX IF NOT EXISTS idx_nodes_label ON nodes(label);
CREATE INDEX IF NOT EXISTS idx_nodes_comm  ON nodes(community);

-- graphify id -> content key. graphify ids are unstable across rebuilds
-- (its own ids.py documents the bug class), so nothing durable keys on them;
-- this table exists only to translate an incoming graph.json.
CREATE TABLE IF NOT EXISTS id_remap (
    graphify_id TEXT PRIMARY KEY,
    node_key    TEXT NOT NULL
);

-- Edges are stored once here for provenance and rendered into CSR for traversal.
-- `channel` separates structural edges from git-derived temporal ones; they are
-- diffused independently (see retrieve/ppr.py) rather than being mixed.
CREATE TABLE IF NOT EXISTS edges (
    src              INTEGER NOT NULL,
    dst              INTEGER NOT NULL,
    relation         TEXT,
    confidence       TEXT,
    confidence_score REAL,
    weight           REAL,
    source_file      TEXT,
    source_location  TEXT,   -- the call site, when graphify recorded one
    channel          TEXT    NOT NULL DEFAULT 'structural'
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src, channel);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst, channel);

-- Per-file git-derived attributes.
CREATE TABLE IF NOT EXISTS files (
    path          TEXT PRIMARY KEY,
    lines         INTEGER,
    language      TEXT,
    n_commits     INTEGER DEFAULT 0,
    churn         REAL    DEFAULT 0,   -- time-decayed
    complexity    REAL    DEFAULT 0,
    hotspot       REAL    DEFAULT 0,   -- churn * complexity
    first_seen    INTEGER,             -- unix ts
    last_modified INTEGER,
    n_authors     INTEGER DEFAULT 0,
    bus_factor    INTEGER,
    primary_owner TEXT,
    owner_share   REAL
);
CREATE INDEX IF NOT EXISTS idx_files_hotspot ON files(hotspot DESC);

-- File-level co-change, with the statistics needed to justify each pair.
CREATE TABLE IF NOT EXISTS cochange (
    path_a   TEXT NOT NULL,
    path_b   TEXT NOT NULL,
    support  REAL NOT NULL,   -- inverse-commit-size weighted
    n_ab     INTEGER NOT NULL,
    n_a      INTEGER NOT NULL,
    n_b      INTEGER NOT NULL,
    lift     REAL,
    g_stat   REAL,
    p_value  REAL,
    PRIMARY KEY (path_a, path_b)
);
CREATE INDEX IF NOT EXISTS idx_cochange_a ON cochange(path_a, support DESC);

-- Per-author line ownership, retained so bus factor can be recomputed
-- without re-walking history.
CREATE TABLE IF NOT EXISTS ownership (
    path   TEXT NOT NULL,
    author TEXT NOT NULL,
    lines  INTEGER NOT NULL,
    PRIMARY KEY (path, author)
);

-- Cached community summaries, keyed by member-content hash so only dirty
-- communities are re-summarized.
CREATE TABLE IF NOT EXISTS summaries (
    community    INTEGER PRIMARY KEY,
    member_hash  TEXT NOT NULL,
    name         TEXT,
    summary      TEXT,
    entry_points TEXT,
    risks        TEXT
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


@dataclass(slots=True)
class EpochPaths:
    root: Path

    @property
    def db(self) -> Path:
        return self.root / "nodes.db"

    @property
    def vectors(self) -> Path:
        return self.root / "vectors.f16.npy"

    @property
    def vector_keys(self) -> Path:
        return self.root / "vector_keys.json"

    @property
    def meta(self) -> Path:
        return self.root / "meta.json"

    def csr(self, channel: str) -> Path:
        return self.root / f"csr_{channel}.npz"

    def bm25(self) -> Path:
        return self.root / "bm25"


class Store:
    """Read/write access to one dwago output directory.

    Typical write flow::

        with Store.begin(project_root) as st:   # allocates a new epoch
            st.write_nodes(...)
            st.write_edges(...)
        # exiting the context publishes the epoch atomically

    Read flow::

        st = Store.open(project_root)           # follows the `current` symlink
    """

    def __init__(self, out_dir: Path, epoch_dir: Path, *, writable: bool = False):
        self.out_dir = Path(out_dir)
        self.paths = EpochPaths(Path(epoch_dir))
        self.writable = writable
        self._conn: sqlite3.Connection | None = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    @staticmethod
    def out_dir_for(project_root: str | Path) -> Path:
        return Path(project_root).resolve() / OUT_DIR_NAME

    @classmethod
    def open(cls, project_root: str | Path) -> "Store":
        """Open the currently published epoch. Raises if none exists."""
        out = cls.out_dir_for(project_root)
        epoch = cls._resolve_current(out)
        if epoch is None:
            raise FileNotFoundError(
                f"no dwago index at {out}. Run `dwago build` first."
            )
        return cls(out, epoch, writable=False)

    @staticmethod
    def _resolve_current(out: Path) -> Path | None:
        """Locate the published epoch via the symlink, else the pointer file.

        Both paths must be understood on read: `_publish` falls back to a plain
        text pointer on filesystems that reject symlinks, and an index written
        that way has to remain openable.
        """
        current = out / "current"
        if current.is_symlink() or current.exists():
            resolved = current.resolve()
            if resolved.exists():
                return resolved
        pointer = out / "current_epoch.txt"
        if pointer.exists():
            candidate = out / "epochs" / pointer.read_text().strip()
            if candidate.exists():
                return candidate
        return None

    @classmethod
    def exists(cls, project_root: str | Path) -> bool:
        return cls._resolve_current(cls.out_dir_for(project_root)) is not None

    @classmethod
    @contextmanager
    def begin(cls, project_root: str | Path, *, inherit: bool = True) -> Iterator["Store"]:
        """Allocate a new epoch, yield a writable Store, publish on clean exit.

        ``inherit`` copies the previous epoch's contents first, which is what
        makes an incremental refresh cheap: unchanged vectors and CSR files are
        already in place and only the dirty parts get rewritten. On any exception
        the half-built epoch is deleted and ``current`` still points at the last
        good one.
        """
        out = cls.out_dir_for(project_root)
        epochs = out / "epochs"
        epochs.mkdir(parents=True, exist_ok=True)

        n = cls._next_epoch_number(epochs)
        epoch_dir = epochs / f"{n:06d}"

        prev = cls._resolve_current(out)
        if inherit and prev is not None and prev.exists():
            shutil.copytree(prev, epoch_dir)
            log.debug("epoch %06d inherited from %s", n, prev.name)
        else:
            epoch_dir.mkdir(parents=True)

        st = cls(out, epoch_dir, writable=True)
        try:
            st.init_schema()
            yield st
            st.commit()
        except BaseException:
            st.close()
            shutil.rmtree(epoch_dir, ignore_errors=True)
            raise
        else:
            st.close()
            st._publish(epoch_dir)
            cls._prune_epochs(epochs, keep=3)

    @staticmethod
    def _next_epoch_number(epochs: Path) -> int:
        existing = [int(p.name) for p in epochs.iterdir()
                    if p.is_dir() and p.name.isdigit()]
        return (max(existing) + 1) if existing else 1

    def _publish(self, epoch_dir: Path) -> None:
        """Publish ``epoch_dir`` via an atomic symlink or text-pointer swap.

        A symlink cannot be retargeted in place, so build a temporary one beside
        it and rename over the top: ``os.replace`` on a symlink is atomic, so a
        concurrent reader resolves either the old target or the new one.
        """
        current = self.out_dir / "current"
        tmp = self.out_dir / f".current.{os.getpid()}.tmp"
        if tmp.exists() or tmp.is_symlink():
            tmp.unlink()
        try:
            # Relative to out_dir, where `current` lives — NOT the bare
            # basename, which would resolve to out_dir/<n> instead of
            # out_dir/epochs/<n> and leave a dangling link.
            target = os.path.relpath(epoch_dir, self.out_dir)
            try:
                os.symlink(target, tmp, target_is_directory=True)
            except OSError:
                # Publish the complete fallback before removing an old link:
                # readers prefer that link until the new pointer is ready.
                tmp.unlink(missing_ok=True)
                tmp.write_text(epoch_dir.name, encoding="utf-8")
                os.replace(tmp, self.out_dir / "current_epoch.txt")
                current.unlink(missing_ok=True)
                log.warning("symlinks unavailable; published via current_epoch.txt")
            else:
                # A failed rename is a publication failure, not evidence that
                # symlinks are unsupported. Leave the old publication intact.
                os.replace(tmp, current)
                # Retain any inactive text pointer: a reader that already saw
                # no link can still finish its lookup on the previous epoch.
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError as exc:
                log.warning("could not clean temporary pointer %s: %s", tmp, exc)

    @staticmethod
    def _prune_epochs(epochs: Path, keep: int = 3) -> None:
        """Keep the last few epochs so a bad build can be rolled back by hand."""
        dirs = sorted((p for p in epochs.iterdir() if p.is_dir() and p.name.isdigit()),
                      key=lambda p: int(p.name))
        for old in dirs[:-keep]:
            shutil.rmtree(old, ignore_errors=True)

    # ── sqlite ───────────────────────────────────────────────────────────────

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.paths.db)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
        return self._conn

    def init_schema(self) -> None:
        self.conn.executescript(_SCHEMA)
        self.set_meta("schema_version", SCHEMA_VERSION)
        self.conn.commit()

    def commit(self) -> None:
        if self._conn is not None:
            self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def set_meta(self, key: str, value: Any) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(value)),
        )

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    # ── nodes ────────────────────────────────────────────────────────────────

    def write_nodes(self, rows: list[dict]) -> None:
        """Replace the node table wholesale.

        Node rows are cheap to rewrite (they are just metadata); the expensive
        artifacts keyed off them — vectors, summaries — survive because they are
        addressed by ``node_key``, not by ``idx``.
        """
        cur = self.conn
        cur.execute("DELETE FROM nodes")
        cur.execute("DELETE FROM id_remap")
        cur.executemany(
            "INSERT INTO nodes (idx, node_key, graphify_id, label, kind, file_type, "
            "source_file, start_line, end_line, signature, docstring, community, "
            "community_name, content_hash, doc) "
            "VALUES (:idx, :node_key, :graphify_id, :label, :kind, :file_type, "
            ":source_file, :start_line, :end_line, :signature, :docstring, :community, "
            ":community_name, :content_hash, :doc)",
            rows,
        )
        cur.executemany(
            "INSERT OR REPLACE INTO id_remap (graphify_id, node_key) VALUES (?, ?)",
            [(r["graphify_id"], r["node_key"]) for r in rows if r.get("graphify_id")],
        )
        self.commit()

    def node_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) AS n FROM nodes").fetchone()["n"]

    def iter_nodes(self, columns: str = "*") -> Iterator[sqlite3.Row]:
        yield from self.conn.execute(f"SELECT {columns} FROM nodes ORDER BY idx")

    def node_keys(self) -> list[str]:
        return [r["node_key"] for r in
                self.conn.execute("SELECT node_key FROM nodes ORDER BY idx")]

    def content_hashes(self) -> dict[str, str]:
        return {r["node_key"]: r["content_hash"] for r in
                self.conn.execute("SELECT node_key, content_hash FROM nodes")}

    # ── edges + CSR ──────────────────────────────────────────────────────────

    def write_edges(self, rows: list[dict], channel: str = "structural") -> None:
        self.conn.execute("DELETE FROM edges WHERE channel = ?", (channel,))
        self.conn.executemany(
            "INSERT INTO edges (src, dst, relation, confidence, confidence_score, "
            "weight, source_file, source_location, channel) "
            "VALUES (:src, :dst, :relation, :confidence, :confidence_score, "
            ":weight, :source_file, :source_location, :channel)",
            [{**r, "channel": channel} for r in rows],
        )
        self.commit()

    def build_csr(self, channel: str, n_nodes: int, *, symmetric: bool = True) -> None:
        """Render one edge channel into a CSR matrix on disk.

        ``symmetric`` controls whether the reverse edge is added. Context
        retrieval wants an undirected walk (a caller is relevant to its callee
        and vice versa); impact analysis re-orients at query time instead of
        keeping a second copy here.
        """
        from scipy import sparse

        rows = self.conn.execute(
            "SELECT src, dst, weight, confidence_score FROM edges WHERE channel = ?",
            (channel,),
        ).fetchall()

        if not rows:
            m = sparse.csr_matrix((n_nodes, n_nodes), dtype=np.float32)
            sparse.save_npz(self.paths.csr(channel), m)
            return

        src = np.fromiter((r["src"] for r in rows), dtype=np.int32, count=len(rows))
        dst = np.fromiter((r["dst"] for r in rows), dtype=np.int32, count=len(rows))
        w = np.fromiter(
            ((r["weight"] if r["weight"] is not None else 1.0)
             * (r["confidence_score"] if r["confidence_score"] is not None else 1.0)
             for r in rows),
            dtype=np.float32, count=len(rows),
        )
        # A non-positive prior would silently delete the edge from the walk;
        # floor it instead so provenance stays visible.
        w = np.maximum(w, 1e-4)

        if symmetric:
            src, dst = np.concatenate([src, dst]), np.concatenate([dst, src])
            w = np.concatenate([w, w])

        m = sparse.coo_matrix((w, (src, dst)), shape=(n_nodes, n_nodes), dtype=np.float32)
        # Parallel relations between the same pair sum, which is the behaviour we
        # want: two independent kinds of link is stronger evidence than one.
        m = m.tocsr()
        m.sum_duplicates()
        sparse.save_npz(self.paths.csr(channel), m)

    def load_csr(self, channel: str):
        from scipy import sparse

        p = self.paths.csr(channel)
        if not p.exists():
            return None
        return sparse.load_npz(p)

    # ── vectors ──────────────────────────────────────────────────────────────

    def load_vector_keys(self) -> dict[str, int]:
        """node_key -> row index in the vector memmap."""
        if not self.paths.vector_keys.exists():
            return {}
        return json.loads(self.paths.vector_keys.read_text())

    def load_vectors(self, mode: str = "r") -> np.ndarray | None:
        if not self.paths.vectors.exists():
            return None
        return np.load(self.paths.vectors, mmap_mode=mode)

    def write_vectors(self, keys: list[str], mat: np.ndarray) -> None:
        """Persist the vector matrix and its key->row map.

        Stored as float16: at 1024 dims that is 2KB per node, so a 100k-node repo
        costs ~200MB. The precision loss is well below the noise floor of cosine
        ranking, and halving the bytes matters more when the matrix is memmapped.
        """
        if mat.dtype != np.float16:
            mat = mat.astype(np.float16)
        tmp = self.paths.vectors.with_suffix(".tmp.npy")
        np.save(tmp, mat)
        os.replace(tmp, self.paths.vectors)
        self.paths.vector_keys.write_text(json.dumps({k: i for i, k in enumerate(keys)}))

    def dirty_keys(self, new_hashes: dict[str, str]) -> tuple[list[str], list[str]]:
        """Split incoming nodes into (needs embedding, reusable).

        A key is dirty when it is new or when its content hash moved. Everything
        else already has a vector in the inherited epoch and is left alone --
        this is the difference between a refresh taking seconds and taking hours.
        """
        have = self.load_vector_keys()
        old_hashes = self.get_meta("content_hashes", {}) or {}
        dirty, reusable = [], []
        for key, h in new_hashes.items():
            if key in have and old_hashes.get(key) == h:
                reusable.append(key)
            else:
                dirty.append(key)
        return dirty, reusable

    # ── introspection ────────────────────────────────────────────────────────

    def stats(self) -> dict:
        c = self.conn
        out = {
            "epoch": self.paths.root.name,
            "nodes": self.node_count(),
            "edges_structural": c.execute(
                "SELECT COUNT(*) n FROM edges WHERE channel='structural'").fetchone()["n"],
            "edges_temporal": c.execute(
                "SELECT COUNT(*) n FROM edges WHERE channel='temporal'").fetchone()["n"],
            "files": c.execute("SELECT COUNT(*) n FROM files").fetchone()["n"],
            "cochange_pairs": c.execute("SELECT COUNT(*) n FROM cochange").fetchone()["n"],
            "communities": c.execute(
                "SELECT COUNT(DISTINCT community) n FROM nodes "
                "WHERE community IS NOT NULL").fetchone()["n"],
            "summaries": c.execute("SELECT COUNT(*) n FROM summaries").fetchone()["n"],
        }
        vk = self.load_vector_keys()
        out["vectors"] = len(vk)
        return out
