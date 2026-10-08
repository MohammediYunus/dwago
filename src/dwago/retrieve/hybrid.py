"""Query pipeline: exact match, then BM25 + dense fused, then PPR diffusion.

Stage order matters and is not arbitrary.

**Stage 0 — exact and prefix symbol match.** A query that names a symbol
(``_pick_seeds``, ``IdempotencyKey``) should resolve immediately. graphify's own
search gets this right with tiered matching; skipping it would mean paying the
whole pipeline to rediscover something a dictionary lookup already knew, and
occasionally ranking it *below* a fuzzy match.

**Stage 1 — lexical and dense, fused by Reciprocal Rank Fusion.** RRF rather
than score addition because BM25 scores and cosine similarities live on
incomparable scales; normalizing them into agreement requires per-corpus tuning
that RRF avoids entirely by using ranks. It is also robust when one retriever
returns nothing, which happens whenever dense is unavailable.

**Stage 2 — PPR diffusion** from the fused hits, over both channels.

Reranking sits between stages 1 and 2 by default, but the position is a
parameter, because it is genuinely unclear which is better: rerank-then-diffuse
gives the walk better seeds, while diffuse-then-rerank lets the reranker see
candidates the walk found. The eval harness settles it per repo instead of this
module asserting an answer.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np

from ..index.dense import DenseIndex, Embedder, resolve_backend
from ..index.lexical import LexicalIndex
from .ppr import combine_channels, ppr

log = logging.getLogger(__name__)

__all__ = ["Retriever", "RetrievalResult", "Hit"]

RRF_K = 60  # standard RRF damping; rank 1 scores 1/61, rank 10 scores 1/70

# Share of the final score from direct retrieval vs graph diffusion. Fitted
# per repo by the eval harness; this is the starting point, not a claim.
SEED_WEIGHT = 0.6

# Discount applied to nodes living in test files. Not zero: tests genuinely
# document behaviour and often belong in the answer.
TEST_PRIOR = 0.55

# How many results may share one label. Vendored copies otherwise
# crowd out everything else; 2 keeps a genuine near-tie visible.
DUPLICATE_LABEL_LIMIT = 2

_TEST_PATH_RE = re.compile(
    r"(^|/)(tests?|spec|__tests__|testing)/|(^|/)test_[^/]*$|_test\.[a-z]+$|\.(spec|test)\.[a-z]+$",
    re.IGNORECASE,
)


@dataclass
class Hit:
    idx: int
    score: float
    label: str = ""
    source_file: str = ""
    start_line: int | None = None
    end_line: int | None = None
    kind: str = ""
    why: str = ""          # provenance, surfaced to the user

    def location(self) -> str:
        if self.source_file and self.start_line:
            return f"{self.source_file}:{self.start_line}"
        return self.source_file or ""


@dataclass
class RetrievalResult:
    hits: list[Hit] = field(default_factory=list)
    seeds: list[Hit] = field(default_factory=list)
    stages: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _rrf(rankings: list[list[tuple[int, float]]], weights: list[float] | None = None
         ) -> dict[int, float]:
    """Reciprocal Rank Fusion over several ranked lists."""
    weights = weights or [1.0] * len(rankings)
    fused: dict[int, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, (idx, _score) in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + w / (RRF_K + rank)
    return fused


class Retriever:
    """Holds the indexes for one store and answers queries."""

    def __init__(self, store, *, embed_backend: str = "auto",
                 temporal_weight: float = 0.3, include_tests: bool = True):
        self.store = store
        self.temporal_weight = temporal_weight
        self.include_tests = include_tests
        self.lexical = LexicalIndex.load(store.paths.bm25())
        self.dense = DenseIndex.load(store)
        self._embedder: Embedder | None = None
        self._embed_backend = embed_backend
        self._meta = {
            r["idx"]: dict(r) for r in store.conn.execute(
                "SELECT idx, label, kind, source_file, start_line, end_line "
                "FROM nodes"
            )
        }
        # Match the opened Store snapshot, as with node metadata and indexes.
        # Containment in the temporal channel is not evidence of co-change.
        self._cochange_neighbors: dict[str, set[str]] = {}
        for edge in store.conn.execute(
            "SELECT src, dst FROM edges WHERE channel='temporal' "
            "AND relation='co_changes_with'"
        ):
            a = self._meta.get(edge["src"], {}).get("source_file")
            b = self._meta.get(edge["dst"], {}).get("source_file")
            if a and b and a != b:
                self._cochange_neighbors.setdefault(a, set()).add(b)
                self._cochange_neighbors.setdefault(b, set()).add(a)

    # ── stage 0 ──────────────────────────────────────────────────────────────

    def exact_match(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """Symbol-name lookup for queries that name something directly."""
        q = query.strip().rstrip("()").lstrip(".")
        if not q or " " in q or len(q) < 3:
            return []
        rows = self.store.conn.execute(
            "SELECT idx, label FROM nodes "
            "WHERE label = ? OR label = ? OR label = ? COLLATE NOCASE LIMIT ?",
            (q, q + "()", q, k),
        ).fetchall()
        out = [(r["idx"], 1.0) for r in rows]
        if len(out) < k and re.fullmatch(r"[\w.]+", q):
            like = self.store.conn.execute(
                "SELECT idx FROM nodes WHERE label LIKE ? COLLATE NOCASE LIMIT ?",
                (q + "%", k - len(out)),
            ).fetchall()
            seen = {i for i, _ in out}
            out.extend((r["idx"], 0.7) for r in like if r["idx"] not in seen)
        return out

    # ── stage 1 ──────────────────────────────────────────────────────────────

    def _embed_query(self, query: str) -> np.ndarray:
        if not self.dense.available:
            return np.zeros(0, dtype=np.float32)
        if self._embedder is None:
            model = self.store.get_meta("embedding_model")
            backend = self.store.get_meta("embedding_backend") or self._embed_backend
            kind = {"model2vec": "model2vec",
                    "sentence-transformers": "st"}.get(backend, "auto")
            self._embedder = Embedder(resolve_backend(kind, model))
        vecs = self._embedder.encode([query], progress=False)
        return vecs[0] if len(vecs) else np.zeros(0, dtype=np.float32)

    def candidates(self, query: str, k: int = 100) -> tuple[dict[int, float], dict]:
        lex = self.lexical.search(query, k=k) if self.lexical.available else []
        dense: list[tuple[int, float]] = []
        if self.dense.available:
            qv = self._embed_query(query)
            if qv.size:
                dense = self.dense.search(qv, k=k)

        stages = {"bm25": len(lex), "dense": len(dense)}
        if not lex and not dense:
            return {}, stages
        fused = _rrf([r for r in (lex, dense) if r])
        stages["fused"] = len(fused)
        return fused, stages

    # ── full pipeline ────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        k: int = 20,
        candidate_k: int = 100,
        seed_k: int = 15,
        use_ppr: bool = True,
        reverse: bool = False,
        rerank: bool = False,
    ) -> RetrievalResult:
        res = RetrievalResult()

        exact = self.exact_match(query)
        fused, stages = self.candidates(query, k=candidate_k)
        res.stages = stages
        res.stages["exact"] = len(exact)

        # Exact hits are injected above the fused scores rather than merged into
        # them: naming a symbol is an unambiguous signal and should not have to
        # out-compete a hundred fuzzy neighbours on rank.
        top_fused = max(fused.values()) if fused else 0.0
        for idx, w in exact:
            fused[idx] = top_fused + w

        if not fused:
            res.warnings.append(
                "no lexical or semantic match — the corpus may not cover this topic"
            )
            return res

        # Apply the kind prior BEFORE seeds are chosen, not only at final
        # scoring. Seeds decide where the diffusion starts, so a test-heavy seed
        # set walks outward through the test suite and never reaches the
        # implementation it exercises — the prior has to be in place first or it
        # only reorders a result set that was already wrong.
        ranked = sorted(
            ((i, sc * self._prior(i)) for i, sc in fused.items()),
            key=lambda kv: -kv[1],
        )
        ranked = [(i, sc) for i, sc in ranked if sc > 0]
        if rerank:
            ranked = self._rerank(query, ranked, top_n=min(50, len(ranked)))

        seeds = ranked[:seed_k]
        res.seeds = [self._hit(i, s, "seed") for i, s in seeds]

        if not use_ppr:
            res.hits = [self._hit(i, s, "lexical/dense") for i, s in ranked[:k]]
            return res

        struct = self.store.load_csr("structural")
        temp = self.store.load_csr("temporal")
        seed_map = {i: max(s, 0.0) for i, s in seeds}

        s_scores = ppr(struct, seed_map, reverse=reverse).scores if struct is not None else None
        t_scores = ppr(temp, seed_map, reverse=reverse).scores if temp is not None and temp.nnz else None

        if s_scores is None and t_scores is None:
            res.hits = [self._hit(i, s, "lexical/dense") for i, s in ranked[:k]]
            return res

        combined = combine_channels(s_scores, t_scores,
                                    temporal_weight=self.temporal_weight)
        res.stages["ppr_channels"] = (
            ("structural" if s_scores is not None else "")
            + ("+temporal" if t_scores is not None else "")
        )

        # Blend retrieval score with diffusion mass, each max-normalized.
        #
        # The obvious alternative — give seeds an absolute bonus so they always
        # outrank diffused nodes — is what this originally did, and it silently
        # made PPR a no-op: with seed_k (15) above the result count (k), every
        # slot was already taken by a seed before diffusion was consulted. A
        # blend keeps direct hits on top (they score on both terms) while still
        # letting a strongly-connected non-seed outrank a marginal one.
        seed_idx = {i for i, _ in seeds}
        seed_files = {self._meta[i]["source_file"] for i in seed_idx
                      if self._meta.get(i, {}).get("source_file")}
        # Explanations follow the channels actually used by combine_channels.
        # A sole available channel contributes even at the opposite endpoint.
        explain_s, explain_t = s_scores, t_scores
        if s_scores is not None and t_scores is not None:
            weight = float(np.clip(self.temporal_weight, 0.0, 1.0))
            if weight <= 0:
                explain_t = None
            if weight >= 1:
                explain_s = None
        history_files = self._history_files(seed_files) if explain_t is not None else set()
        fused_peak = max((s for _, s in ranked), default=0.0) or 1.0
        ppr_peak = float(combined.max()) or 1.0
        fused_norm = {i: s / fused_peak for i, s in ranked}

        scored: list[tuple[int, float, str]] = []
        for i, mass in enumerate(combined):
            f = fused_norm.get(i, 0.0)
            pm = float(mass) / ppr_peak
            if f <= 0 and pm <= 0:
                continue
            # fused_norm already carries the prior (applied at candidate
            # selection); apply it to the diffusion term only, so a node the
            # walk reached is discounted consistently with one retrieved directly.
            score = SEED_WEIGHT * f + (1.0 - SEED_WEIGHT) * pm * self._prior(i)
            why = "matched query" if i in seed_idx else self._explain(
                i, explain_s, explain_t, seed_files, history_files)
            scored.append((i, score, why))

        scored.sort(key=lambda t: -t[1])

        # Suppress near-duplicates. Real corpora vendor the same file into many
        # places — graphify itself copies one reference doc into 14 per-platform
        # skill directories — and without this a single document can occupy most
        # of the result list. Identical label plus identical kind is a
        # deliberately conservative test: it catches copies without merging two
        # genuinely distinct symbols that happen to share a name in different
        # files, because those differ in the path shown to the user anyway.
        seen_labels: dict[tuple[str, str], int] = {}
        deduped: list[tuple[int, float, str]] = []
        for i, sc, why in scored:
            m = self._meta.get(i, {})
            sig = ((m.get("label") or "").strip().lower(), m.get("kind") or "")
            n_seen = seen_labels.get(sig, 0)
            if n_seen >= DUPLICATE_LABEL_LIMIT:
                res.stages["suppressed_duplicates"] = \
                    res.stages.get("suppressed_duplicates", 0) + 1
                continue
            seen_labels[sig] = n_seen + 1
            deduped.append((i, sc, why))
            if len(deduped) >= k:
                break

        res.hits = [self._hit(i, s, why) for i, s, why in deduped[:k]]
        return res

    def _prior(self, idx: int) -> float:
        """Mild ranking prior over node kinds.

        Test files are systematically over-retrieved by lexical search because
        test names are literally descriptive sentences: a query like "how are
        duplicate nodes merged" matches `test_exact_duplicates_merged` far more
        strongly than the implementation it exercises. That is a real signal —
        the test does describe the behaviour — but someone asking how something
        works wants the implementation first. A modest discount reorders them
        without hiding them; `include_tests=False` removes them outright.
        """
        f = self._meta.get(idx, {}).get("source_file") or ""
        if _TEST_PATH_RE.search(f):
            return 0.0 if not self.include_tests else TEST_PRIOR
        return 1.0

    def _history_files(self, seed_files: set[str]) -> set[str]:
        """Files connected to query seeds by retained co-change edges."""
        reached = set(seed_files)
        pending = list(seed_files)
        while pending:
            for other in self._cochange_neighbors.get(pending.pop(), ()):
                if other not in reached:
                    reached.add(other)
                    pending.append(other)
        return reached - seed_files

    def _explain(self, idx: int, s_scores, t_scores, seed_files: set[str],
                 history_files: set[str]) -> str:
        """Describe active graph support without treating containment as history."""
        s = float(s_scores[idx]) if s_scores is not None else 0.0
        t = float(t_scores[idx]) if t_scores is not None else 0.0
        source = self._meta.get(idx, {}).get("source_file")
        if t > 0 and source in seed_files:
            return "in the same file as a match"
        if t > 0 and source in history_files:
            if self._cochange_neighbors.get(source, set()) & seed_files:
                history = "file co-changes with a matched file"
                return f"connected in code; {history}" if s > 0 else history
            return ("connected through code and co-change history" if s > 0
                    else "connected through co-change history")
        if s > 0 or t > 0:
            return "connected to matches"
        # A retrieved candidate outside seed_k can survive without graph mass.
        return "matched query"

    def _hit(self, idx: int, score: float, why: str = "") -> Hit:
        m = self._meta.get(idx, {})
        return Hit(
            idx=idx, score=float(score),
            label=m.get("label") or "", source_file=m.get("source_file") or "",
            start_line=m.get("start_line"), end_line=m.get("end_line"),
            kind=m.get("kind") or "", why=why,
        )

    def _rerank(self, query: str, ranked: list[tuple[int, float]], top_n: int
                ) -> list[tuple[int, float]]:
        """Cross-encoder rerank of the top candidates.

        Off by default: on CPU a cross-encoder over 50 documents costs seconds,
        not milliseconds, which is the wrong trade for an interactive query. The
        eval harness measures whether it earns its latency on a given repo.
        """
        try:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415
        except ImportError:
            log.debug("no sentence-transformers; skipping rerank")
            return ranked

        head, tail = ranked[:top_n], ranked[top_n:]
        docs = {
            r["idx"]: r["doc"] for r in self.store.conn.execute(
                "SELECT idx, doc FROM nodes WHERE idx IN "
                f"({','.join('?' * len(head))})", [i for i, _ in head]
            )
        }
        pairs = [(query, docs.get(i, "")) for i, _ in head]
        if not pairs:
            return ranked
        try:
            from ..index.dense import CACHE_DIR  # noqa: PLC0415

            model = CrossEncoder("BAAI/bge-reranker-v2-m3", cache_folder=CACHE_DIR)
            scores = model.predict(pairs)
        except Exception as exc:  # noqa: BLE001
            log.warning("rerank failed (%s); using fused order", exc)
            return ranked
        rescored = sorted(zip((i for i, _ in head), scores),
                          key=lambda kv: -float(kv[1]))
        return [(i, float(s)) for i, s in rescored] + tail
