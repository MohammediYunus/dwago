"""The graph tools (path/cycles/diff_impact/tests_for), summaries and the
brain payload, exercised against a small synthetic store."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dwago.store import Store
from dwago.summarize import summarize_communities, get_summaries
from dwago.viz.build_brain import build_payload, write_brain


@pytest.fixture()
def toy_store(tmp_path: Path) -> Store:
    with Store.begin(tmp_path, inherit=False) as st:
        rows = []
        files = ["src/a.ts", "src/b.ts", "src/c.ts", "test/a.test.ts"]
        for i, f in enumerate(files):
            rows.append(dict(idx=i, node_key=f"k{i}", graphify_id=str(i),
                             label=f.split("/")[-1],
                             kind="file", file_type="code", source_file=f,
                             start_line=1, end_line=10, signature=None,
                             docstring=None, community=i % 2,
                             community_name=f"comm{i % 2}", content_hash=f"h{i}",
                             doc=f"doc {f}"))
        st.write_nodes(rows)
        st.write_edges([
            dict(src=0, dst=1, relation="imports", confidence="high",
                 confidence_score=1.0, weight=1.0, source_file=None,
                 source_location=None),
            dict(src=1, dst=2, relation="imports", confidence="high",
                 confidence_score=1.0, weight=1.0, source_file=None,
                 source_location=None),
            dict(src=2, dst=0, relation="imports", confidence="high",
                 confidence_score=1.0, weight=1.0, source_file=None,
                 source_location=None),
            dict(src=3, dst=0, relation="imports", confidence="high",
                 confidence_score=1.0, weight=1.0, source_file=None,
                 source_location=None),
        ], channel="structural")
        st.build_csr("structural", len(rows))
    return Store.open(tmp_path)


def test_brain_payload(toy_store):
    p = build_payload(toy_store)
    assert len(p["files"]) == 4
    assert len(p["regions"]) == 3          # 2 communities + everything-else
    assert p["edges"], "file edges collapsed"
    assert all(0 <= f["r"] < len(p["regions"]) for f in p["files"])


def test_brain_html(toy_store, tmp_path):
    out = tmp_path / "brain.html"
    n = write_brain(toy_store, out, title="toy")
    assert n == 4
    html = out.read_text()
    assert "window.DWAGO_DATA=" in html
    payload = html.split("window.DWAGO_DATA=")[1].split(";\n")[0].split(";const")[0]
    data = json.loads(payload.rstrip(";"))
    assert data["files"][0]["p"].endswith(".ts")


def test_summaries_cached(toy_store):
    calls = []

    def stub(prompt):
        calls.append(prompt)
        return "Does X. Entry point is a.ts."

    r1 = summarize_communities(toy_store, top=2, caller=stub)
    assert r1["written"] == 2 and not r1["errors"]
    r2 = summarize_communities(toy_store, top=2, caller=stub)
    assert r2["cached"] == 2 and r2["written"] == 0
    assert len(calls) == 2, "cache prevented re-calls"
    assert len(get_summaries(toy_store)) == 2


@pytest.mark.parametrize("change", [
    "UPDATE nodes SET content_hash='changed' WHERE community=0",
    "UPDATE nodes SET node_key='replacement' WHERE idx=0",
    "UPDATE nodes SET community=2 WHERE community=0",
    "DELETE FROM nodes WHERE community=0",
])
def test_inherited_summaries_match_current_members(toy_store, change):
    summarize_communities(toy_store, caller=lambda _: "Original summary.")
    with Store.begin(toy_store.out_dir.parent) as rebuilt:
        rebuilt.conn.execute(change)

    current = Store.open(toy_store.out_dir.parent)
    try:
        changes_before = current.conn.total_changes
        # A stale first row must not consume the requested result slot.
        assert get_summaries(current, 1) == [
            {"community": 1, "name": "comm1", "summary": "Original summary."},
        ]
        assert get_summaries(current, 0) == []
        assert get_summaries(current, -1) == get_summaries(current, 1)
        assert current.conn.total_changes == changes_before
        assert current.conn.execute(
            "SELECT COUNT(*) FROM community_summaries").fetchone()[0] == 2
        # Existing readers retain the complete, valid previous epoch.
        assert len(get_summaries(toy_store)) == 2
    finally:
        current.close()


def test_cached_summaries_use_current_names(toy_store):
    calls = []
    summarize_communities(toy_store, caller=lambda prompt: calls.append(prompt) or "Summary.")
    toy_store.conn.execute("UPDATE nodes SET community_name='' WHERE community=0")
    toy_store.conn.execute("UPDATE nodes SET community_name='renamed' WHERE idx=2")
    toy_store.conn.execute("UPDATE nodes SET community_name=NULL WHERE community=1")
    rows = get_summaries(toy_store, -1)
    assert [row["name"] for row in rows] == ["renamed", "community 1"]
    result = summarize_communities(toy_store, caller=lambda prompt: calls.append(prompt) or "New.")
    assert result["cached"] == 2 and result["written"] == 0
    assert len(calls) == 2


def test_failed_summary_refresh_does_not_expose_old_text(toy_store):
    summarize_communities(toy_store, caller=lambda _: "Old summary.")
    toy_store.conn.execute("UPDATE nodes SET content_hash='changed' WHERE community=0")

    def fail(_):
        raise RuntimeError("summary backend unavailable")

    result = summarize_communities(toy_store, caller=fail)
    assert result["cached"] == 1 and result["skipped"] == 1
    assert len(result["errors"]) == 1
    assert [row["community"] for row in get_summaries(toy_store)] == [1]

    result = summarize_communities(toy_store, caller=lambda _: "Updated summary.")
    assert result["written"] == result["cached"] == 1
    assert [row["summary"] for row in get_summaries(toy_store)] == [
        "Updated summary.", "Old summary.",
    ]


def test_overview_omits_stale_summaries(toy_store):
    pytest.importorskip("mcp")
    import asyncio
    from dwago.serve import build_server

    summarize_communities(toy_store, caller=lambda _: "Obsolete architecture.")
    toy_store.conn.execute("UPDATE nodes SET content_hash='changed'")
    toy_store.conn.commit()
    server = build_server(toy_store.out_dir.parent)

    async def run():
        result = await server.call_tool("overview", {})
        content = getattr(result, "content", result)
        return "\n".join(getattr(c, "text", str(c)) for c in content)

    text = asyncio.run(run())
    assert "Obsolete architecture." not in text
    assert "No current community summaries" in text
    assert "dwago summarize" in text


def test_summaries_limit_selects_largest_current_community(toy_store):
    toy_store.conn.execute("DELETE FROM nodes WHERE idx=0")
    summarize_communities(toy_store, caller=lambda _: "Summary.")
    assert [row["community"] for row in get_summaries(toy_store, 1)] == [1]
    assert [row["community"] for row in get_summaries(toy_store, -1)] == [1, 0]


@pytest.mark.parametrize("tool,kwargs,expect", [
    ("path", dict(from_symbol="a.ts", to_symbol="c.ts"), "hops"),
    ("cycles", dict(min_size=2), "cycle"),
    ("tests_for", dict(symbol="src/a.ts"), "a.test.ts"),
])
def test_graph_tools(toy_store, tmp_path, tool, kwargs, expect):
    mcp = pytest.importorskip("mcp")  # noqa: F841
    import asyncio
    from dwago.serve import build_server

    server = build_server(toy_store.out_dir.parent)

    async def run():
        r = await server.call_tool(tool, kwargs)
        content = getattr(r, "content", r)
        if isinstance(content, list):
            return "\n".join(getattr(c, "text", str(c)) for c in content)
        return str(content)

    text = asyncio.run(run())
    assert expect in text
