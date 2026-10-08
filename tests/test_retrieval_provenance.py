"""Explanations reflect query-connected history, not containment-channel mass."""
from __future__ import annotations

import pytest

from dwago.enrich.git_temporal import build_temporal_edges
from dwago.index.docs import build_documents
from dwago.index.lexical import LexicalIndex
from dwago.ingest import ingest
from dwago.retrieve.hybrid import Retriever
from dwago.store import Store


@pytest.fixture()
def project(tmp_path):
    opened = []

    def build(pairs=(), *, imports=False):
        files = {
            "alpha.py": ('import beta\n\n' if imports else '')
                        + 'def needle():\n    """Locate a unique invoice marker."""\n    return 1\n',
            "beta.py": 'def nearby():\n    return 2\n',
            "gamma.py": 'def distant():\n    return 3\n',
            "unrelated_a.py": 'def separate_a():\n    return 4\n',
            "unrelated_b.py": 'def separate_b():\n    return 5\n',
        }
        for name, text in files.items():
            (tmp_path / name).write_text(text, encoding="utf-8")
        with Store.begin(tmp_path, inherit=False) as store:
            ingest(tmp_path, store)
            for a, b in pairs:
                store.conn.execute(
                    "INSERT INTO cochange VALUES (?,?,3,3,3,3,2,10,0.001)", (a, b))
            build_temporal_edges(store)
            build_documents(store)
            LexicalIndex.build(store, store.paths.bm25())
        store = Store.open(tmp_path)
        opened.append(store)
        return store

    yield build
    for store in opened:
        store.close()


def _hits(store, **kwargs):
    retriever = Retriever(store, embed_backend="none", **kwargs)
    assert not retriever.dense.available
    return retriever


def _search(retriever, **kwargs):
    result = retriever.search("needle", seed_k=1, k=20, **kwargs)
    assert result.seeds[0].label == "needle"
    assert result.stages["dense"] == 0
    return {hit.label: hit for hit in result.hits}


@pytest.mark.parametrize("unrelated_history", [False, True])
def test_same_file_containment_is_not_cochange(project, unrelated_history):
    pairs = [("unrelated_a.py", "unrelated_b.py")] if unrelated_history else []
    store = project(pairs)
    hits = _search(_hits(store))
    assert hits["alpha.py"].score > 0
    assert hits["alpha.py"].why == "in the same file as a match"
    assert not any("co-change" in hit.why for hit in hits.values())
    assert "unrelated_a.py" not in hits


@pytest.mark.parametrize("reverse", [False, True])
def test_direct_and_multihop_history_have_distinct_explanations(project, reverse):
    store = project([("alpha.py", "beta.py"), ("beta.py", "gamma.py")])
    hits = _search(_hits(store), reverse=reverse)
    assert hits["alpha.py"].why == "in the same file as a match"
    assert hits["beta.py"].why == "file co-changes with a matched file"
    assert hits["nearby"].why == "file co-changes with a matched file"
    assert hits["gamma.py"].why == "connected through co-change history"
    assert hits["distant"].why == "connected through co-change history"


@pytest.mark.parametrize("weight,confidence", [(0, 1), (3, 0), (-1, 1), (3, -1), (-1, -1)])
def test_history_evidence_includes_edges_retained_by_csr_floor(project, weight, confidence):
    store = project([("alpha.py", "beta.py")])
    store.conn.execute(
        "UPDATE edges SET weight=?, confidence_score=? "
        "WHERE channel='temporal' AND relation='co_changes_with'", (weight, confidence))
    # Store retains these edges, flooring their effective weight at a minimum of 1e-4.
    store.build_csr("temporal", store.node_count())
    hits = _search(_hits(store))
    assert hits["beta.py"].score > 0
    assert hits["beta.py"].why == "file co-changes with a matched file"


def test_zero_temporal_weight_does_not_claim_history(project):
    store = project([("alpha.py", "beta.py")], imports=True)
    hits = _search(_hits(store, temporal_weight=0))
    assert hits["beta.py"].score > 0
    assert hits["beta.py"].why == "connected to matches"
    assert not any("co-change" in hit.why for hit in hits.values())


def test_one_temporal_weight_does_not_claim_code_contribution(project):
    store = project([("alpha.py", "beta.py")], imports=True)
    hits = _search(_hits(store, temporal_weight=1))
    assert hits["beta.py"].why == "file co-changes with a matched file"


def test_both_channels_preserve_code_and_history_explanations(project):
    store = project([("alpha.py", "beta.py")], imports=True)
    hits = _search(_hits(store))
    assert hits["beta.py"].why == "connected in code; file co-changes with a matched file"


def test_temporal_only_fallback_still_contributes_at_zero_weight(project):
    store = project([("alpha.py", "beta.py")])
    store.paths.csr("structural").unlink()
    hits = _search(_hits(store, temporal_weight=0))
    assert hits["beta.py"].why == "file co-changes with a matched file"


def test_structural_only_fallback_still_contributes_at_one_weight(project):
    store = project(imports=True)
    store.paths.csr("temporal").unlink()
    hits = _search(_hits(store, temporal_weight=1))
    assert hits["beta.py"].why == "connected to matches"
    assert not any("co-change" in hit.why for hit in hits.values())


def test_no_ppr_does_not_report_graph_evidence(project):
    store = project([("alpha.py", "beta.py")], imports=True)
    hits = _search(_hits(store), use_ppr=False)
    assert hits
    assert all(hit.why == "lexical/dense" for hit in hits.values())


def test_history_explanations_remain_bound_to_opened_epoch(project):
    first = project([("alpha.py", "beta.py")])
    old_reader = _hits(first)
    second = project([("alpha.py", "gamma.py")])
    old_hits = _search(old_reader)
    new_hits = _search(_hits(second))
    assert old_hits["beta.py"].why == "file co-changes with a matched file"
    assert "gamma.py" not in old_hits
    assert new_hits["gamma.py"].why == "file co-changes with a matched file"
    assert "beta.py" not in new_hits


def test_reverse_explanation_uses_actual_structural_direction(project):
    store = project([("alpha.py", "beta.py")], imports=True)
    # With beta importing alpha, a reverse walk from alpha's symbol reaches beta.
    store.conn.execute("UPDATE edges SET src=dst, dst=src WHERE relation='imports'")
    store.build_csr("structural", store.node_count(), symmetric=False)
    forward = _search(_hits(store))
    reverse = _search(_hits(store), reverse=True)
    assert forward["beta.py"].why == "file co-changes with a matched file"
    assert reverse["beta.py"].why == "connected in code; file co-changes with a matched file"


@pytest.mark.parametrize("transport", ["cli", "mcp"])
def test_impact_output_does_not_deny_existing_code_links(project, capsys, transport):
    store = project([("alpha.py", "beta.py"), ("beta.py", "gamma.py")], imports=True)
    root = store.out_dir.parent
    if transport == "cli":
        from dwago.cli import main

        assert main(["impact", "needle", str(root)]) == 0
        text = capsys.readouterr().out
    else:
        import asyncio
        from dwago.serve import build_server

        server = build_server(root)

        async def call():
            response = await server.call_tool("impact_of", {"symbol": "needle"})
            return "\n".join(getattr(part, "text", str(part))
                             for part in getattr(response, "content", response))

        text = asyncio.run(call())
    assert "Reached through co-change history:" in text
    assert "beta.py" in text and "gamma.py" in text
    assert "no static link" not in text
    assert "Historically changes alongside" not in text
