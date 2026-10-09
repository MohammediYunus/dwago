# dwago

Ask your codebase anything.

`dwago build .` indexes supported source files and definitions with tree-sitter,
infers import links, groups files into communities, and mines your git history. Then
you ask questions in plain English and get answers that cite `file:line`.
Indexing and search run locally without an API key. Embedding modes may download
model weights on first use; optional summarization uses the backend you choose.

![the brain map](docs/images/brain.jpg)

That is the map. Files are neurons, the biggest communities are lobes, imports
are the wiring, and the gold spray marks where the code is churning. It is one
HTML file that works offline. I wanted a map you could actually use, and every
code map I had seen was a screenshot for slide decks. So the map and the search
share the same index.

## Why this exists

Parsers can't see time. The file that breaks in prod is usually the one that
changed forty times this quarter, in lockstep with a config file nobody
documented. That link is invisible in the code and obvious in git. dwago mines
it, with a significance test so coincidences don't count, and ranks files by
recent churn and bus factor while it's there.

The second reason is that substring search can't answer "where do we prevent
double charging". Semantic search can. dwago runs BM25 and embeddings
together, fuses the rankings, then lets relevance flow along import and
co-change edges so the neighborhood of a good hit surfaces too.

`dwago eval` compares retrieval modes using tasks from your git history. It
excludes held-out commits from the co-change layer; the source index still
reflects the checkout you built. Results include recall, mean reciprocal rank,
query timing, and paired confidence intervals for recall differences.
See [how the evaluation works](references/eval.md) before interpreting a result.

## The strike

![selecting a file](docs/images/strike.jpg)

Click a file, or pick it from the panel. The rest of the scene dims, strikes
run along its real edges, and every connected file gets a reticle and a name.
The same information as `dwago impact`, drawn instead of printed.

![neurons view and key files](docs/images/neurons.jpg)

The panel on the left switches views. Neurons strips the shell away and shows
the bare network. Key files rings the hotspots git complains about most.

## What you can ask

| question | command |
|---|---|
| where is X, how does Y work | `dwago ask "how do we prevent double charging?"` |
| give me everything for this task, budgeted, cited | `dwago pack "migrate the session store" --budget 8000` |
| what breaks if I touch this | `dwago impact src/auth/session.ts` |
| what always changes with this file | `dwago co-change src/db/store.ts` |
| where is the risk concentrated | `dwago hotspots` |
| what did the last ten commits touch, really | `diff_impact` over MCP |
| which tests cover this | `tests_for` over MCP |
| how are these two files connected | `path`, `cycles` over MCP |
| show me | `dwago map` |

## Install

Requires Python 3.10 or newer and Git. Install from GitHub; dwago is not
currently published on PyPI.

Start with lexical search and MCP in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install "dwago[lexical,mcp] @ git+https://github.com/MohammediYunus/dwago"
```

On Windows, activate with **.venv\Scripts\activate** instead. Installation
downloads Python packages; this setup does not download embedding models or
require an API key.

From a clone of this repository, you can instead install in editable mode:

```bash
python -m pip install -e ".[lexical,mcp]"
```

## Use

Run these commands from the project you want to explore. Replace the example
question with one about your own code:

```bash
dwago build . --embed-backend none
dwago ask "where is the OIDC issuer configured?"
dwago stats
```

This builds a local index and searches it with BM25. To refresh it after edits
without adding embeddings, run:

```bash
dwago refresh . --embed-backend none
```

Indexes built with **0.3.5 or earlier** can retain incorrect Git history records
for non-ASCII filenames. After upgrading, rebuild once to remove those old records;
ordinary refresh does not remove them:

```bash
dwago build . --force --embed-backend none
```

Indexes built with **0.3.7 or earlier** under a non-UTF-8 default text encoding
may also be missing links for Unicode imports in UTF-8 source files. If affected,
use the same forced rebuild after upgrading. The **0.3.8** change-impact filename
fix works with existing valid indexes without a rebuild.

If you use embeddings, replace **none** with your usual backend. Restart any
running MCP server to load the rebuilt index.

Maps and history-based evaluation are separate commands:

```bash
dwago map                # writes dwago-out/brain.html, open it
dwago eval               # benchmark it on your own history first
```

After a commit, `dwago refresh` re-embeds only what changed. Readers never see
a half-built index; a build publishes atomically or not at all.

## Optional embeddings

To add the small embedding model to the virtual environment above:

```bash
python -m pip install "dwago[lexical,fast,mcp] @ git+https://github.com/MohammediYunus/dwago"
dwago build . --fast
```

The model is downloaded on first use. The **dense** extra enables a
sentence-transformer encoder with PyTorch instead; see the
[embedding tiers](references/build.md#embedding-tiers) for installation commands.
Retrieval quality and runtime depend on the model and repository; compare them
with **dwago eval**.

## Works with your agent, or without one

The MCP server speaks the standard protocol, so Cursor, Claude Code, Codex
CLI, Cline, Windsurf and Zed all get the same 13 tools:

```bash
dwago serve /path/to/repo
```

See the [MCP setup guide](references/mcp.md) to configure the installed
executable in your agent.

`SKILL.md` is plain markdown instructions with a command table. Claude Code
users copy it to `~/.claude/skills/dwago/`. Everyone else can paste it into
AGENTS.md or point their rules file at it. There is nothing vendor-specific
in it.

The CLI also works without an agent. Optional summaries can use OpenAI,
Anthropic, or an installed Claude CLI. They send community file paths and symbol
names to the selected backend. Indexing, search, and graph tools work without
summaries.

MCP's architecture overview only returns cached summaries whose members and
content still match the opened index. After rebuilding, run **dwago summarize**
with your chosen backend to regenerate summaries for changed communities.
Reading an overview does not call a model or regenerate missing summaries.
Restart an already-running MCP server after rebuilding to load the new index;
its opened index remains a snapshot until then.

## Languages

Python, TypeScript, TSX, JavaScript, Go, Rust, Java, C, C++, C#, Ruby, PHP,
plus Markdown and RST as document nodes. Import edges use text-based resolution
for Python and TypeScript/JavaScript. It can miss imports or infer false links:
for example, Python multi-import statements, dynamic JavaScript imports, and
commented import text are not handled fully. The other languages still get containment,
communities and the whole git layer. If you already have a node-link
`graph.json` from another tool, `dwago build --graph path.json` ingests it.

## Rough edges

`tests_for` is a coupling heuristic and says so in its own output; coverage
ingestion would make it exact. Co-change is correlation, and the tool reports
lift and p-values so you can judge. Retrieval quality depends on the repository
and embedding model; use `dwago eval` to compare configurations.
Above 200k nodes, nearest-neighbor search still needs an
ANN index I haven't wired in. Tests cover extraction, retrieval, storage,
graph tools, and visualization.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the local setup, test commands,
and guidance for bug reports and focused pull requests.

## License

Apache-2.0.
