# Build pipeline

`dwago build` runs six stages. Each writes into a new *epoch* directory; the
index only becomes visible when the whole build succeeds and a symlink flips, so
a query running during a rebuild never sees a half-written index.

## Stages

1. **Extract & ingest** — walk the repository, tree-sitter every supported
   file into file + symbol nodes, resolve Python/TS/JS imports into edges,
   detect communities (Louvain, named from shared directories), then remap node
   IDs onto content-addressed keys.
2. **Spans** — parse every source file with tree-sitter for real line ranges,
   signatures and docstrings. A bare graph records only a *start* line and no
   signature or body, which is not enough to build a retrieval document or cite
   a range.
3. **Git** — mine history for co-change, hotspots, ownership (see
   `temporal.md`).
4. **Documents** — assemble one retrieval document per node (see `query.md`).
5. **BM25** — build the lexical index.
6. **Embeddings** — encode documents; reuse vectors for nodes whose content
   hash did not change.

## Source files and project scope

Source reads for indexing and context packs stay within the selected project's
resolved root. File symlinks are supported when their targets remain inside that
root; their original names remain in citations. Outside targets, missing files
and broken or looping links supply no source text.

An explicitly selected `--graph` JSON file may live elsewhere. Its nodes and
edges are still imported when their source files are unavailable or outside the
project, but those references receive no local source enrichment or snippets.

This path check is not a sandbox against concurrent filesystem changes or an
untrusted index directory. Upgrading does not remove content already stored in
old indexes: rebuild to create current records, and restart an MCP server to use
the rebuilt index. Older epochs remain on disk.

## Embedding tiers

Resolved at runtime, best available first:

| Tier | Extra | Notes |
|---|---|---|
| `Qwen/Qwen3-Embedding-0.6B` | **dense** | Default. Apache-2.0. Best quality. |
| `minishlab/potion-base-8M` | **fast** | `--fast`. Pure numpy, ~30MB, no torch. Measurably weaker. |
| none | — | BM25 only. Still works. |

Install the extra you want from GitHub in your activated virtual environment.
These commands retain lexical search and MCP support:

```bash
# Small, pure-NumPy encoder:
python -m pip install "dwago[lexical,fast,mcp] @ git+https://github.com/MohammediYunus/dwago"

# Sentence-transformer encoder with PyTorch:
python -m pip install "dwago[lexical,dense,mcp] @ git+https://github.com/MohammediYunus/dwago"
```

For a first build without model downloads, follow the
[no-model quickstart](../README.md#use).

`jinaai/jina-code-embeddings-0.5b` is available via `--model` and is often
stronger on code-heavy corpora, but check its licence before using it at work —
several Jina checkpoints ship CC-BY-NC.

Cost is real and stated up front: encoding ~100k documents with a 0.6B model on
CPU is hours. `build` estimates and stops rather than starting silently. It
auto-detects CUDA and MPS. The encoder and any reranker are loaded one at a
time, never together, so an 8GB machine survives.

## Incremental refresh

`dwago refresh` re-runs the pipeline but re-embeds only nodes whose content
hash moved. Everything else is copied forward from the previous epoch.

This works because identity and freshness are separate hashes. A node's *key*
comes from its file, name and kind, so editing a function body leaves its
identity intact; its *content hash* comes from the source text, so the edit
marks it dirty. Upstream extractor IDs are unsuitable as keys — they
documents a recurring drift bug class, and its dedup pass can collapse
same-named symbols across files during an update.

Chain it to a post-commit hook rather than running a second
watcher.

## Failure modes

| Symptom | Cause |
|---|---|
| `no 'nodes' key` | `--graph` was given a file that is not a node-link graph. |
| span coverage below 50% | `graph.json` was built from a different checkout. Rebuild it. |
| `all vectors reused` after real edits | Stale index — run `dwago build . --force`. |
| build stops asking for `--yes` | The time estimate exceeded two minutes. Intended. |
