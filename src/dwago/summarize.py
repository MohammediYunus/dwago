"""Community summaries, cached by member-content hash.

Each of the largest communities gets a two-sentence architectural summary.
Summaries are cached in the store keyed by a hash of the member node keys
and content hashes, so an unchanged community never hits the model twice.

Backends (any LLM works; none is required):
  auto        -- first of: OPENAI_API_KEY, ANTHROPIC_API_KEY, claude CLI
  openai      -- OPENAI_API_KEY via raw HTTPS, no SDK dependency
  anthropic   -- ANTHROPIC_API_KEY via raw HTTPS, no SDK dependency
  claude-cli  -- shells out to a local `claude -p`
  none        -- skip quietly (build stays LLM-free by default)
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import urllib.request
from collections import defaultdict

from .store import Store

_PROMPT = (
    "You are summarizing one module cluster of a codebase for an engineer "
    "seeing it for the first time. Files and symbols:\n\n{members}\n\n"
    "Reply with exactly two sentences: what this cluster is responsible "
    "for, and what its most important file or entry point is. No preamble."
)


def _members_digest(rows) -> str:
    h = hashlib.blake2b(digest_size=16)
    for r in rows:
        h.update((r["node_key"] or "").encode())
        h.update((r["content_hash"] or "").encode())
    return h.hexdigest()


def _community_members(store: Store) -> tuple[dict[int, list], dict[int, str]]:
    groups: dict[int, list] = defaultdict(list)
    names: dict[int, str] = {}
    for row in store.conn.execute(
            "SELECT node_key, content_hash, label, kind, source_file,"
            " community, community_name FROM nodes"
            " WHERE community IS NOT NULL ORDER BY community, idx"):
        groups[row["community"]].append(row)
        if row["community_name"]:
            names.setdefault(row["community"], row["community_name"])
    return groups, names


def _call_claude_cli(prompt: str, model: str, timeout: int = 90) -> str:
    r = subprocess.run(
        ["claude", "-p", prompt, "--model", model],
        capture_output=True, text=True, timeout=timeout,
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:200] or "claude CLI failed")
    return r.stdout.strip()


def _call_openai(prompt: str, model: str, timeout: int = 90) -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    body = json.dumps({
        "model": model,
        "max_tokens": 300,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}",
                 "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    return data["choices"][0]["message"]["content"].strip()


def _call_anthropic(prompt: str, model: str, timeout: int = 90) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    body = json.dumps({
        "model": model,
        "max_tokens": 300,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body,
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    return "".join(b.get("text", "") for b in data.get("content", [])).strip()


def summarize_communities(store: Store, *, top: int = 20,
                          backend: str = "auto",
                          model: str | None = None,
                          caller=None) -> dict:
    """Returns {written, cached, skipped, errors}. ``caller`` overrides the
    backend entirely (tests use a stub)."""
    con = store.conn
    con.execute(
        "CREATE TABLE IF NOT EXISTS community_summaries ("
        " community INTEGER PRIMARY KEY, name TEXT, member_hash TEXT,"
        " summary TEXT)")

    if caller is None:
        if backend == "none":
            return {"written": 0, "cached": 0, "skipped": top, "errors": []}
        if backend == "auto":
            if os.environ.get("OPENAI_API_KEY"):
                backend = "openai"
            elif os.environ.get("ANTHROPIC_API_KEY"):
                backend = "anthropic"
            else:
                backend = "claude-cli"
        if backend == "openai":
            m = model or "gpt-5-mini"
            caller = lambda p: _call_openai(p, m)  # noqa: E731
        elif backend == "anthropic":
            m = model or "claude-haiku-4-5-20251001"
            caller = lambda p: _call_anthropic(p, m)  # noqa: E731
        else:
            m = model or "haiku"
            caller = lambda p: _call_claude_cli(p, m)  # noqa: E731

    groups, names = _community_members(store)
    biggest = sorted(groups, key=lambda c: -len(groups[c]))[:top]
    written = cached = 0
    errors: list[str] = []
    for comm in biggest:
        rows = groups[comm]
        digest = _members_digest(rows)
        row = con.execute(
            "SELECT member_hash FROM community_summaries WHERE community=?",
            (comm,)).fetchone()
        if row and row["member_hash"] == digest:
            cached += 1
            continue
        sample = rows[:40]
        members = "\n".join(
            f"- {r['label']} ({r['kind'] or 'node'}) {r['source_file'] or ''}"
            for r in sample)
        try:
            text = caller(_PROMPT.format(members=members))
        except Exception as e:  # noqa: BLE001 — record, keep going
            errors.append(f"community {comm}: {e}")
            continue
        con.execute(
            "INSERT INTO community_summaries (community, name, member_hash,"
            " summary) VALUES (?,?,?,?) ON CONFLICT(community) DO UPDATE SET"
            " name=excluded.name, member_hash=excluded.member_hash,"
            " summary=excluded.summary",
            (comm, names.get(comm, f"community {comm}"), digest, text))
        written += 1
    con.commit()
    return {"written": written, "cached": cached,
            "skipped": len(biggest) - written - cached, "errors": errors}


def get_summaries(store: Store, n: int = 20) -> list[dict]:
    """Return up to n current summaries, largest communities first.

    Cached text must match the members and content of this store's epoch.
    Reading never regenerates summaries or changes the stored cache.
    """
    con = store.conn
    try:
        rows = con.execute(
            "SELECT community, member_hash, summary FROM community_summaries"
        ).fetchall()
    except Exception:
        return []
    if not rows or n == 0:
        return []
    cached = {row["community"]: row for row in rows}
    groups, names = _community_members(store)
    summaries = [
        {"community": comm, "name": names.get(comm, f"community {comm}"),
         "summary": cached[comm]["summary"]}
        for comm in sorted(groups, key=lambda c: -len(groups[c]))
        if comm in cached and cached[comm]["member_hash"] == _members_digest(groups[comm])
    ]
    return summaries[:n] if n >= 0 else summaries
