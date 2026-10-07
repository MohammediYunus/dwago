# The benchmark

Use dwago eval to compare retrieval configurations on tasks drawn from your
repository's history. Results describe this benchmark, with the limitations below.

```bash
dwago eval .              # ladder over up to 200 held-out changes
dwago eval . -n 500 --split 0.7 --out evaluation.json
```

## Ground truth

For each historical change, the query is what the author wrote about the work
(commit subject and body) and the expected files are those they actually touched.
This provides a task proxy without manual annotation; it does not measure whether
an assistant solves the task correctly.

## Leakage control

History is split by time: temporal co-change edges come only from commits before
a cutoff, and commits after it are scored. The structural graph and source index
still reflect the checkout used for the build. This controls temporal co-change
leakage; it is not a reconstruction of the entire repository at the cutoff.

Passing --no-rebuild skips that temporal-layer rebuild. Interpret those results
as optimistic if the existing layer contains evaluated commits.

## Known biases

- Commit subjects are written after the work, in the vocabulary of the
  implementation. This can favor lexical retrieval. Gains on these queries do
  not establish gains on questions people ask while working.
- Files that are no longer tracked in the current checkout are omitted from the labels.
- Lockfiles and common generated paths are excluded because they often co-occur
  with many unrelated files.
- Merge, release, dependency-bump and some formatting changes are skipped by
  commit-message filters.

## Reading the output

Each rung is a retrieval configuration evaluated on the same queries.

- R@k is the mean fraction of changed files found in the first k retrieved files.
- MRR is the mean reciprocal rank of the first matching file.
- s/query is the mean query time for that run.
- Paired bootstrap intervals compare recall differences on the same tasks.
  The report compares consecutive rungs at R@20 and the first and last rungs
  at each reported k. An interval spanning zero does not establish an improvement.

These metrics do not include precision or end-to-end coding task success.
Compare recall at the result count you plan to use, together with query latency.
The evaluation reports results; it does not change retrieval defaults for you.

## Sharing a result

To make your results reproducible, include:

- The public repository URL and commit, plus the dwago version or commit.
- Build and evaluation commands, including split, seed, model and history limits.
- The evaluation JSON saved with --out, alongside the printed report.
- Hardware and model-cache state when reporting timings.

Report uncertainty and regressions as well as improvements. A result from one
repository does not establish that an encoder is better for every codebase.
