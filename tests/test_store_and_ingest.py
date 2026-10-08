"""Store lifecycle (epochs, atomic publish) and graph.json ingestion."""
from __future__ import annotations

import errno
import json
import os
from pathlib import Path

import pytest

from dwago.ingest import GraphJsonError, ingest, load_graph_json
from dwago.store import Store


def _graph(nodes, links) -> dict:
    return {"directed": False, "multigraph": False, "graph": {},
            "nodes": nodes, "links": links}


def _write_graph(root, data):
    d = root / "graphify-out"
    d.mkdir(parents=True, exist_ok=True)
    (d / "graph.json").write_text(json.dumps(data))
    return d / "graph.json"


def test_publish_is_visible_and_resolves(tmp_path):
    """Regression: the symlink was written relative to the wrong base.

    `current -> 000001` resolved to out/000001 while the epoch lived at
    out/epochs/000001, so every read after a successful build failed.
    """
    _write_graph(tmp_path, _graph(
        [{"id": "a", "label": "a.py", "file_type": "code",
          "source_file": "a.py", "source_location": "L1"}], []))
    assert not Store.exists(tmp_path)
    with Store.begin(tmp_path) as st:
        ingest(tmp_path, st, tmp_path / 'graphify-out' / 'graph.json')
    assert Store.exists(tmp_path)
    st = Store.open(tmp_path)
    assert st.node_count() == 1
    assert st.paths.root.exists()


def test_failed_build_leaves_previous_epoch_intact(tmp_path):
    _write_graph(tmp_path, _graph(
        [{"id": "a", "label": "a.py", "file_type": "code",
          "source_file": "a.py", "source_location": "L1"}], []))
    with Store.begin(tmp_path) as st:
        ingest(tmp_path, st, tmp_path / 'graphify-out' / 'graph.json')
    first = Store.open(tmp_path).paths.root.name

    with pytest.raises(RuntimeError):
        with Store.begin(tmp_path) as st:
            raise RuntimeError("build blew up")

    assert Store.open(tmp_path).paths.root.name == first, \
        "a failed build must not change what `current` points at"


def _publish_generation(root, generation):
    with Store.begin(root) as st:
        st.set_meta("generation", generation)


def _read_generation(root):
    st = Store.open(root)
    try:
        return st.get_meta("generation")
    finally:
        st.close()


def test_successive_publications_keep_existing_reader_snapshot(tmp_path):
    _publish_generation(tmp_path, 1)
    old = Store.open(tmp_path)
    try:
        assert old.get_meta("generation") == 1
        _publish_generation(tmp_path, 2)
        assert _read_generation(tmp_path) == 2
        assert old.get_meta("generation") == 1
    finally:
        old.close()


@pytest.mark.parametrize("failure", [None, "pointer", "unlink"])
def test_windows_directory_symlink_replace_uses_pointer(tmp_path, monkeypatch,
                                                       failure):
    _publish_generation(tmp_path, 1)
    old = Store.open(tmp_path)
    out = Store.out_dir_for(tmp_path)
    current = out / "current"
    pointer = out / "current_epoch.txt"
    replace = os.replace
    unlink = Path.unlink
    observed = []

    def windows_replace(src, dst):
        if Path(dst) == current:
            error = OSError(errno.EACCES, "synthetic Windows directory symlink replacement")
            error.winerror = 5
            raise error
        if Path(dst) == pointer:
            assert Path(src).read_text() == "000002"
            assert _read_generation(tmp_path) == 1
            if failure == "pointer":
                raise OSError(errno.ENOSPC, "synthetic pointer failure")
        return replace(src, dst)

    def observe_unlink(path, *args, **kwargs):
        if path == current:
            assert pointer.read_text() == "000002"
            observed.append(_read_generation(tmp_path))
            if failure == "unlink":
                raise OSError(errno.EACCES, "synthetic unlink failure")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "replace", windows_replace)
    monkeypatch.setattr(Path, "unlink", observe_unlink)
    try:
        assert old.get_meta("generation") == 1
        if failure is None:
            _publish_generation(tmp_path, 2)
        else:
            with pytest.raises(OSError, match=f"synthetic {failure} failure"):
                _publish_generation(tmp_path, 2)
        assert observed == ([] if failure == "pointer" else [1])
        assert _read_generation(tmp_path) == (2 if failure is None else 1)
        assert old.get_meta("generation") == 1
        assert current.is_symlink() is (failure is not None)
        assert not list(out.glob(".current*.tmp"))
    finally:
        old.close()


def _no_symlinks(*args, **kwargs):
    raise OSError(errno.EPERM, "synthetic symlink restriction")


def test_publication_switches_to_pointer_and_back(tmp_path, monkeypatch):
    _publish_generation(tmp_path, 1)
    old = Store.open(tmp_path)
    assert old.get_meta("generation") == 1
    out = Store.out_dir_for(tmp_path)
    try:
        with monkeypatch.context() as m:
            m.setattr(os, "symlink", _no_symlinks)
            _publish_generation(tmp_path, 2)
            assert _read_generation(tmp_path) == 2
            assert not (out / "current").is_symlink()
            assert (out / "current_epoch.txt").read_text() == "000002"
            assert old.get_meta("generation") == 1

        _publish_generation(tmp_path, 3)
        assert (out / "current").is_symlink()
        assert _read_generation(tmp_path) == 3
        assert old.get_meta("generation") == 1
    finally:
        old.close()


def test_pointer_publication_replaces_complete_file(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "symlink", _no_symlinks)
    _publish_generation(tmp_path, 1)
    out = Store.out_dir_for(tmp_path)
    pointer = out / "current_epoch.txt"
    replace = os.replace
    observed = []

    def observe_replace(src, dst):
        if Path(dst) == pointer:
            assert Path(src).parent == out
            assert Path(src).read_text() == "000002"
            assert pointer.read_text() == "000001"
            observed.append(_read_generation(tmp_path))
            replace(src, dst)
            observed.append(_read_generation(tmp_path))
        else:
            replace(src, dst)

    monkeypatch.setattr(os, "replace", observe_replace)
    _publish_generation(tmp_path, 2)
    assert observed == [1, 2]
    assert not list(out.glob(".current*.tmp"))


def test_symlink_replace_failure_does_not_publish_pointer(tmp_path, monkeypatch):
    _publish_generation(tmp_path, 1)
    out = Store.out_dir_for(tmp_path)
    current = out / "current"
    replace = os.replace

    def fail_current_replace(src, dst):
        if Path(dst) == current:
            raise OSError(errno.EACCES, "synthetic rename restriction")
        return replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_current_replace)
    with pytest.raises(OSError, match="synthetic rename restriction"):
        _publish_generation(tmp_path, 2)
    assert _read_generation(tmp_path) == 1
    assert not (out / "current_epoch.txt").exists()
    assert not list(out.glob(".current*.tmp"))


@pytest.mark.parametrize("initial_pointer", [False, True])
def test_pointer_replace_failure_preserves_publication(tmp_path, monkeypatch,
                                                       initial_pointer):
    if initial_pointer:
        monkeypatch.setattr(os, "symlink", _no_symlinks)
    _publish_generation(tmp_path, 1)
    out = Store.out_dir_for(tmp_path)
    pointer = out / "current_epoch.txt"
    monkeypatch.setattr(os, "symlink", _no_symlinks)
    replace = os.replace

    def fail_pointer_replace(src, dst):
        if Path(dst) == pointer:
            raise OSError(errno.EACCES, "synthetic pointer rename restriction")
        return replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_pointer_replace)
    with pytest.raises(OSError, match="synthetic pointer rename restriction"):
        _publish_generation(tmp_path, 2)
    assert _read_generation(tmp_path) == 1
    if initial_pointer:
        assert pointer.read_text() == "000001"
    else:
        assert not pointer.exists()
    assert not list(out.glob(".current*.tmp"))


def test_fallback_does_not_report_success_if_old_link_cannot_be_removed(
        tmp_path, monkeypatch):
    _publish_generation(tmp_path, 1)
    current = Store.out_dir_for(tmp_path) / "current"
    unlink = Path.unlink

    def fail_current_unlink(path, *args, **kwargs):
        if path == current:
            raise OSError(errno.EACCES, "synthetic unlink restriction")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "symlink", _no_symlinks)
    monkeypatch.setattr(Path, "unlink", fail_current_unlink)
    with pytest.raises(OSError, match="synthetic unlink restriction"):
        _publish_generation(tmp_path, 2)
    assert _read_generation(tmp_path) == 1


def test_reader_can_finish_pointer_lookup_during_switch_to_symlink(
        tmp_path, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(os, "symlink", _no_symlinks)
        _publish_generation(tmp_path, 1)
    current = Store.out_dir_for(tmp_path) / "current"
    exists = Path.exists
    published = False

    def publish_after_link_check(path):
        nonlocal published
        found = exists(path)
        if path == current and not published:
            assert not found
            published = True
            _publish_generation(tmp_path, 2)
        return found

    monkeypatch.setattr(Path, "exists", publish_after_link_check)
    assert _read_generation(tmp_path) == 1
    assert published
    assert _read_generation(tmp_path) == 2


def test_missing_graph_json_explains_how_to_fix(tmp_path):
    with pytest.raises(GraphJsonError) as e:
        load_graph_json(tmp_path / "graphify-out" / "graph.json")
    assert "graphify" in str(e.value)


def test_wrong_shape_is_rejected_loudly(tmp_path):
    p = tmp_path / "g.json"
    p.write_text(json.dumps({"something": "else"}))
    with pytest.raises(GraphJsonError):
        load_graph_json(p)


def test_edges_accept_either_links_or_edges_key(tmp_path):
    p = tmp_path / "g.json"
    p.write_text(json.dumps({"nodes": [], "edges": []}))
    assert load_graph_json(p)["edges"] == []


def test_leading_dot_method_labels_match_spans(tmp_path):
    """Regression: graphify emits `.update()` for receiver-less methods.

    Leaving the dot on meant those labels never matched a parsed span, which
    cost 18 points of span coverage on a real repository.
    """
    (tmp_path / "m.py").write_text("class A:\n    def update(self):\n        return 1\n")
    _write_graph(tmp_path, _graph([
        {"id": "m", "label": "m.py", "file_type": "code",
         "source_file": "m.py", "source_location": "L1"},
        {"id": "m_update", "label": ".update()", "file_type": "code",
         "source_file": "m.py", "source_location": "L2"},
    ], []))
    with Store.begin(tmp_path) as st:
        res = ingest(tmp_path, st, tmp_path / 'graphify-out' / 'graph.json')
    st = Store.open(tmp_path)
    row = st.conn.execute(
        "SELECT signature, start_line, end_line FROM nodes WHERE label = '.update()'"
    ).fetchone()
    assert row["signature"], "leading-dot label should still resolve to its span"
    assert row["end_line"] >= row["start_line"]


def test_file_nodes_get_whole_file_ranges(tmp_path):
    (tmp_path / "m.py").write_text("a = 1\nb = 2\nc = 3\n")
    _write_graph(tmp_path, _graph([
        {"id": "m", "label": "m.py", "file_type": "code",
         "source_file": "m.py", "source_location": "L1"}], []))
    with Store.begin(tmp_path) as st:
        ingest(tmp_path, st, tmp_path / 'graphify-out' / 'graph.json')
    row = Store.open(tmp_path).conn.execute(
        "SELECT start_line, end_line, kind FROM nodes WHERE label='m.py'").fetchone()
    assert row["kind"] == "file"
    assert row["start_line"] == 1 and row["end_line"] == 3


def test_dangling_and_self_edges_are_counted_not_crashed(tmp_path):
    _write_graph(tmp_path, _graph(
        [{"id": "a", "label": "a.py", "file_type": "code",
          "source_file": "a.py", "source_location": "L1"}],
        [{"source": "a", "target": "ghost", "relation": "calls"},
         {"source": "a", "target": "a", "relation": "calls"}]))
    with Store.begin(tmp_path) as st:
        res = ingest(tmp_path, st, tmp_path / 'graphify-out' / 'graph.json')
    assert res.edges == 0
    assert res.dropped_edges == 2


def test_dirty_keys_reuses_unchanged_vectors(tmp_path):
    _write_graph(tmp_path, _graph(
        [{"id": "a", "label": "a.py", "file_type": "code",
          "source_file": "a.py", "source_location": "L1"}], []))
    with Store.begin(tmp_path) as st:
        ingest(tmp_path, st, tmp_path / 'graphify-out' / 'graph.json')
        st.set_meta("content_hashes", {"k1": "h1", "k2": "h2"})
        import numpy as np
        st.write_vectors(["k1", "k2"], np.zeros((2, 4), dtype=np.float16))
        dirty, reusable = st.dirty_keys({"k1": "h1", "k2": "CHANGED", "k3": "new"})
    assert set(dirty) == {"k2", "k3"}
    assert reusable == ["k1"]
