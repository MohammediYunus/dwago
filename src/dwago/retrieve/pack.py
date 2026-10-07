"""Assemble retrieval results into a token-budgeted, citable context pack.

This is what an agent actually consumes, and the constraint that shapes it is
that a context window is a hard budget: going over does not degrade the answer,
it truncates it somewhere arbitrary. So the pack is built greedily against an
explicit budget and every included item carries a ``file:line`` citation.

Two rules stop a pack from being technically full and practically useless:

*File dedup.* Ten methods from one class are ten citations of the same file. The
first few earn their place; the rest crowd out other parts of the codebase.

*Community diversity floor.* Highly-ranked results cluster, so a naive greedy
fill can spend the whole budget inside one subsystem and never mention the
caller in another that the change also touches. A minimum share of the budget is
reserved for results from communities not yet represented.

Token counting is approximate by design — tiktoken would tie the pack to one
tokenizer family and add a dependency for a number used only to decide when to
stop. Characters-per-token is estimated conservatively so the pack undershoots
rather than overshoots.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..source import read_source_lines

__all__ = ["ContextPack", "build_pack", "estimate_tokens"]

# Conservative: code tokenizes worse than prose (punctuation, long identifiers),
# so 3.2 chars/token undershoots for English and is about right for source.
CHARS_PER_TOKEN = 3.2


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


@dataclass
class PackItem:
    label: str
    location: str
    kind: str
    why: str
    text: str
    tokens: int
    community: int | None = None


@dataclass
class ContextPack:
    query: str
    items: list[PackItem] = field(default_factory=list)
    budget: int = 0
    used: int = 0
    truncated: int = 0            # results that did not fit
    communities: int = 0
    warnings: list[str] = field(default_factory=list)

    def render(self) -> str:
        """Markdown, because every agent reads it and it survives copy-paste."""
        if not self.items:
            return (f"No context found for: {self.query}\n\n"
                    + "\n".join(f"- {w}" for w in self.warnings))

        out = [f"# Context for: {self.query}", ""]
        out.append(
            f"_{len(self.items)} results across {self.communities} subsystems · "
            f"~{self.used} tokens_"
        )
        if self.truncated:
            out.append(f"_{self.truncated} further results omitted to stay in budget._")
        out.append("")

        for it in self.items:
            head = f"## {it.label}"
            if it.location:
                head += f"  `{it.location}`"
            out.append(head)
            meta = " · ".join(x for x in (it.kind, it.why) if x)
            if meta:
                out.append(f"_{meta}_")
            if it.text:
                out.append("")
                out.append("```")
                out.append(it.text)
                out.append("```")
            out.append("")
        return "\n".join(out)


def _read_slice(root: Path, rel: str, start: int | None, end: int | None,
                cache: dict, max_lines: int = 40) -> str:
    if not rel or not start:
        return ""
    if rel not in cache:
        cache[rel] = read_source_lines(root, rel)
    lines = cache[rel]
    if not lines:
        return ""
    lo = max(0, start - 1)
    hi = min(len(lines), end or start)
    chunk = lines[lo:hi]
    if len(chunk) > max_lines:
        # Head plus an explicit marker beats a silent cut: the reader can see
        # that the body continues and go look if it matters.
        omitted = len(chunk) - max_lines
        chunk = chunk[:max_lines] + [f"    ... {omitted} more lines ..."]
    return "\n".join(chunk)


def build_pack(
    store,
    project_root: Path,
    query: str,
    hits: list,
    *,
    budget: int = 8000,
    max_per_file: int = 3,
    diversity_reserve: float = 0.25,
    max_lines_per_item: int = 40,
) -> ContextPack:
    """Greedily fill a pack from ranked hits, respecting dedup and diversity."""
    pack = ContextPack(query=query, budget=budget)
    if not hits:
        pack.warnings.append("retrieval returned no results")
        return pack

    comm_of = {
        r["idx"]: r["community"] for r in store.conn.execute(
            "SELECT idx, community FROM nodes"
        )
    }

    root = Path(project_root)
    cache: dict[str, list[str]] = {}
    per_file: dict[str, int] = {}
    seen_comms: set[int] = set()
    used = 0
    reserve = int(budget * diversity_reserve)

    deferred: list = []

    for hit in hits:
        comm = comm_of.get(hit.idx)
        f = hit.source_file or ""

        # Once a file has contributed its share, hold further hits from it back
        # rather than dropping them: if budget remains after the diverse pass,
        # they are still better than nothing.
        if f and per_file.get(f, 0) >= max_per_file:
            deferred.append(hit)
            continue

        # Guard the reserve: when the remaining budget is down to the reserved
        # slice, only admit results from subsystems not yet represented.
        remaining = budget - used
        if remaining <= reserve and comm in seen_comms:
            deferred.append(hit)
            continue

        text = _read_slice(root, f, hit.start_line, hit.end_line, cache,
                           max_lines=max_lines_per_item)
        item_text = text or ""
        cost = estimate_tokens(f"{hit.label}{hit.location()}{item_text}") + 12

        if used + cost > budget:
            deferred.append(hit)
            continue

        pack.items.append(PackItem(
            label=hit.label, location=hit.location(), kind=hit.kind,
            why=hit.why, text=item_text, tokens=cost, community=comm,
        ))
        used += cost
        if f:
            per_file[f] = per_file.get(f, 0) + 1
        if comm is not None:
            seen_comms.add(comm)

    # Second pass: spend anything left on what was held back.
    for hit in deferred:
        remaining = budget - used
        if remaining <= 0:
            pack.truncated += 1
            continue
        text = _read_slice(root, hit.source_file, hit.start_line, hit.end_line,
                           cache, max_lines=max_lines_per_item)
        cost = estimate_tokens(f"{hit.label}{hit.location()}{text or ''}") + 12
        if used + cost > budget:
            pack.truncated += 1
            continue
        pack.items.append(PackItem(
            label=hit.label, location=hit.location(), kind=hit.kind,
            why=hit.why, text=text or "", tokens=cost,
            community=comm_of.get(hit.idx),
        ))
        used += cost
        if hit.source_file:
            per_file[hit.source_file] = per_file.get(hit.source_file, 0) + 1

    pack.used = used
    pack.communities = len({i.community for i in pack.items if i.community is not None})
    return pack
