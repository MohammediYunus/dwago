"""BM25 index over node documents.

bm25s rather than SQLite FTS5, for one reason: control of tokenization. FTS5
would treat ``parseURLPath`` as a single token, which is precisely the failure
that makes lexical code search feel broken. Here the same splitting used to
build documents is applied to queries, so ``url parsing`` reaches
``parseURLPath`` without anyone hand-writing a synonym list.

CJK text is segmented with jieba when available. graphify deliberately handles
Chinese identifiers and comments (``serve.py`` carries a ``_segment_chinese``
path), and silently regressing that for non-ASCII codebases would be a step
backwards, not forwards.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

from .docs import split_identifier

log = logging.getLogger(__name__)

__all__ = ["LexicalIndex", "tokenize"]

_WORD_RE = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)
_CJK_RE = re.compile(r"[一-鿿぀-ヿ가-힯]")

_jieba = None
_jieba_tried = False


def _maybe_jieba():
    global _jieba, _jieba_tried
    if not _jieba_tried:
        _jieba_tried = True
        try:
            import jieba  # noqa: PLC0415

            _jieba = jieba
        except ImportError:
            _jieba = None
    return _jieba


def tokenize(text: str) -> list[str]:
    """Tokenize a document or query the same way, which is what makes them match."""
    if not text:
        return []
    out: list[str] = []
    for raw in _WORD_RE.findall(text):
        low = raw.lower()
        out.append(low)
        # Re-emit the split form so a compound identifier is reachable by its
        # parts. Cheap: most tokens are already single words and split to
        # themselves, which dict.fromkeys collapses at document build time.
        parts = split_identifier(raw)
        if len(parts) > 1:
            out.extend(parts)

    if _CJK_RE.search(text):
        jb = _maybe_jieba()
        if jb is not None:
            out.extend(t for t in jb.cut_for_search(text) if t.strip())
        else:
            # Without a segmenter, character bigrams are the standard fallback
            # and are far better than treating a whole CJK run as one token.
            for run in re.findall(r"[一-鿿぀-ヿ가-힯]+", text):
                out.extend(run[i:i + 2] for i in range(max(1, len(run) - 1)))
    return out


class LexicalIndex:
    """Thin wrapper over bm25s with dwago's tokenizer bolted on."""

    def __init__(self, retriever=None, idx_to_node: list[int] | None = None):
        self._r = retriever
        self.idx_to_node = idx_to_node or []

    @property
    def available(self) -> bool:
        return self._r is not None

    @classmethod
    def build(cls, store, path: Path) -> "LexicalIndex":
        try:
            import bm25s  # noqa: PLC0415
        except ImportError:
            log.warning("bm25s not installed; lexical retrieval disabled. "
                        "See the GitHub installation guide: "
                        "https://github.com/MohammediYunus/dwago#install")
            return cls()

        rows = list(store.conn.execute(
            "SELECT idx, doc FROM nodes WHERE doc IS NOT NULL AND doc != '' ORDER BY idx"
        ))
        if not rows:
            return cls()

        idx_to_node = [r["idx"] for r in rows]
        corpus = [tokenize(r["doc"]) for r in rows]

        retriever = bm25s.BM25()
        retriever.index(corpus)

        path.mkdir(parents=True, exist_ok=True)
        retriever.save(str(path))
        np.save(path / "idx_to_node.npy", np.array(idx_to_node, dtype=np.int32))
        return cls(retriever, idx_to_node)

    @classmethod
    def load(cls, path: Path) -> "LexicalIndex":
        try:
            import bm25s  # noqa: PLC0415
        except ImportError:
            return cls()
        if not (path / "idx_to_node.npy").exists():
            return cls()
        try:
            retriever = bm25s.BM25.load(str(path))
        except Exception as exc:  # noqa: BLE001
            log.warning("could not load BM25 index: %s", exc)
            return cls()
        idx_to_node = np.load(path / "idx_to_node.npy").tolist()
        return cls(retriever, idx_to_node)

    def search(self, query: str, k: int = 100) -> list[tuple[int, float]]:
        """Return [(node_idx, score)] best-first."""
        if self._r is None:
            return []
        toks = tokenize(query)
        if not toks:
            return []
        k = min(k, len(self.idx_to_node))
        if k <= 0:
            return []
        try:
            results, scores = self._r.retrieve([toks], k=k)
        except Exception as exc:  # noqa: BLE001 - an empty/odd query is not fatal
            log.debug("bm25 retrieve failed: %s", exc)
            return []
        out: list[tuple[int, float]] = []
        for pos, score in zip(results[0], scores[0]):
            if score <= 0:
                continue
            out.append((self.idx_to_node[int(pos)], float(score)))
        return out
