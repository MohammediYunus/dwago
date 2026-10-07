# MCP server

```bash
dwago serve
```

Stdio MCP server. Register it with any MCP-capable agent:

```json
{
  "mcpServers": {
    "dwago": {
      "command": "dwago",
      "args": ["serve", "/absolute/path/to/project"]
    }
  }
}
```

Requires `pip install 'dwago[mcp]'`.

## Tools

Results are compact text with `file:line` citations rather than JSON blobs — an
agent pays for every token of a tool result, and prose is both smaller and
directly usable in an answer.

| Tool | What it does |
|---|---|
| `context_pack(task, token_budget)` | **Primary.** Full pipeline → source slices packed to budget, cited. |
| `search(query, k, include_tests)` | Semantic symbol/file/concept search. |
| `impact_of(symbol, k)` | Dependents from code structure **plus** historical co-changers and owners. |
| `co_change(path, n)` | Files that change together, with lift and p-value. |
| `hotspots(n)` | Churn × size ranking, with bus factor. |
| `owners(path)` | Who knows this file. |
| `explain(symbol)` | Location, signature, docs, neighbours. |
| `graph_stats()` | Index health. |

## Design notes

Typical graph servers expose raw graph plumbing (`query_graph`, `get_node`,
`get_neighbors`, `get_community`, `god_nodes`, `graph_stats`, `shortest_path`,
plus three PR tools) over stdio and HTTP. This is a **superset that replaces
it**, not a first agent surface.

- `search` / `explain` are upgrades of `query_graph` / `explain` — same job,
  semantic retrieval behind them instead of substring matching.
- `impact_of` unions the static dependency cone with the temporal overlay.
- `co_change`, `hotspots` and `owners` have no upstream equivalent.

Run one or the other. Running both gives the agent two tools for every job and
it will pick the weaker one about half the time. See `coexist.md`.


## Added in v0.2

| tool | what it answers |
|---|---|
| `path(a, b)` | how A reaches B — BFS over structural + temporal channels |
| `cycles(min_size)` | file-level dependency cycles (Tarjan SCC) |
| `diff_impact(rev_range)` | files changed in a range plus the neighbourhood they ripple into |
| `tests_for(symbol)` | test files coupled by imports + co-change (heuristic until coverage ingestion) |
| **overview(n)** | largest communities with current cached LLM summaries (**dwago summarize**); stale summaries are omitted without a model call |
