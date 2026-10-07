"""Repository-scoped reads must not follow source references outside the project."""
from contextlib import closing
import json
from pathlib import Path

import pytest

from dwago.extract import extract_repo
from dwago.identity import node_key
from dwago.ingest import ingest
from dwago.retrieve.hybrid import Hit
from dwago.retrieve.pack import build_pack
from dwago.spans import extract_repo_spans, extract_spans
from dwago.store import Store


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "repo"
    # The shared prefix catches containment checks based on string prefixes.
    outside = tmp_path / "repo-outside"
    root.mkdir()
    outside.mkdir()
    (root / "safe.py").write_text("def safe():\n    return 'local source'\n")
    (root / "README.md").write_text("# Local notes\nProject content.\n")
    return root, outside


def _link(path, target, *, directory=False):
    try:
        path.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")


def _guard_outside_reads(monkeypatch, outside):
    """Fail at the actual open, even if the caller would discard its contents."""
    original = Path.open
    outside = outside.resolve()

    def guarded(path, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if "r" in mode or "+" in mode:
            try:
                target = path.resolve()
            except (OSError, RuntimeError):  # broken/looping paths cannot be read
                target = path
            assert not target.is_relative_to(outside), f"outside read attempted: {path}"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)


def _reference(root, outside, form, filename="escape.py"):
    if form == "parent":
        return f"../{outside.name}/{filename}"
    if form == "absolute":
        return str(outside / filename)
    if form == "file-link":
        alias = root / filename
        _link(alias, outside / filename)
        return alias.name
    _link(root / "linked", outside, directory=True)
    return f"linked/{filename}"


def _graph(source, *, kind="code", label=None):
    return {"nodes": [{
        "id": "source", "label": label or ("escape.py" if kind == "code" else "Heading"),
        "file_type": kind, "source_file": source, "source_location": "L1",
    }], "links": []}


def _hits(store):
    return [Hit(idx=row["idx"], score=1, label=row["label"],
                source_file=row["source_file"], start_line=row["start_line"],
                end_line=row["end_line"], kind=row["kind"], why="fixture")
            for row in store.iter_nodes()]


@pytest.mark.parametrize("reader", [
    "graph", "discovered-spans", "supplied-spans", "direct-spans",
])
def test_walkers_do_not_read_external_symlinks(project, monkeypatch, reader):
    root, outside = project
    (outside / "escape.py").write_text("def outside_marker():\n    return 123\n")
    (outside / "escape.md").write_text("# Outside marker\nNot project content.\n")
    _link(root / "escape.py", outside / "escape.py")
    _link(root / "escape.md", outside / "escape.md")
    _link(root / "linked", outside, directory=True)
    _guard_outside_reads(monkeypatch, outside)

    if reader == "graph":
        data = extract_repo(root)
        assert {n["source_file"] for n in data["nodes"]} == {"safe.py", "README.md"}
    elif reader == "direct-spans":
        assert extract_spans(outside / "escape.py", root) is None
    else:
        files = None if reader == "discovered-spans" else [
            root / "safe.py", root / "escape.py", root / "linked" / "escape.py",
            outside / "escape.py",
        ]
        spans = extract_repo_spans(root, files=files, max_workers=1)
        assert set(spans) == {"safe.py"}


@pytest.mark.parametrize("form", ["parent", "absolute", "file-link", "dir-link"])
@pytest.mark.parametrize("kind", ["code", "document"])
@pytest.mark.parametrize("parse_spans", [False, True])
def test_ingest_keeps_outside_nodes_without_reading(
        project, monkeypatch, form, kind, parse_spans):
    root, outside = project
    filename = "escape.py" if kind == "code" else "escape.md"
    (outside / filename).write_text("# Outside marker\nNot project content.\n")
    source = _reference(root, outside, form, filename)
    _guard_outside_reads(monkeypatch, outside)

    with Store.begin(root) as store:
        result = ingest(root, store, data=_graph(source, kind=kind), parse_spans=parse_spans)
        rows = list(store.iter_nodes())
        assert result.nodes == len(rows) == 1
        assert rows[0]["source_file"] == source
        assert rows[0]["signature"] == rows[0]["docstring"] == ""
        assert rows[0]["end_line"] == rows[0]["start_line"] == 1


@pytest.mark.parametrize("form", ["parent", "absolute", "file-link", "dir-link"])
def test_pack_does_not_read_outside_index_sources(project, monkeypatch, form):
    root, outside = project
    source = _reference(root, outside, form)
    # Create a real index while the outside source is absent. It can represent
    # graph metadata imported earlier, without having read an external body.
    with Store.begin(root) as store:
        ingest(root, store, data=_graph(source), parse_spans=False)
    (outside / "escape.py").write_text("OUTSIDE_MARKER = 'not project source'\n")
    _guard_outside_reads(monkeypatch, outside)

    with closing(Store.open(root)) as store:
        pack = build_pack(store, root, "fixture", _hits(store))
    assert len(pack.items) == 1
    assert pack.items[0].location == f"{source}:1"
    assert pack.items[0].text == ""
    assert "OUTSIDE_MARKER" not in pack.render()


@pytest.mark.parametrize("directory", [False, True])
def test_pack_rechecks_link_retargeted_after_index(project, monkeypatch, directory):
    root, outside = project
    (outside / "safe.py").write_text("OUTSIDE_MARKER = 'new target'\n")
    alias = root / ("alias" if directory else "alias.py")
    target = root if directory else root / "safe.py"
    _link(alias, target, directory=directory)
    source = "alias/safe.py" if directory else "alias.py"
    with Store.begin(root) as store:
        ingest(root, store, data=_graph(source), parse_spans=False)
    with closing(Store.open(root)) as store:
        before = build_pack(store, root, "fixture", _hits(store))
        assert "def safe():" in before.items[0].text
        alias.unlink()
        _link(alias, outside if directory else outside / "safe.py", directory=directory)
        _guard_outside_reads(monkeypatch, outside)
        after = build_pack(store, root, "fixture", _hits(store))
    assert after.items[0].location == f"{source}:1"
    assert after.items[0].text == ""


@pytest.mark.parametrize("root_link", [False, True])
def test_in_root_aliases_keep_text_and_citations(project, tmp_path, monkeypatch, root_link):
    root, outside = project
    (root / "sub").mkdir()
    _link(root / "alias.py", root / "safe.py")
    selected = root
    if root_link:
        selected = tmp_path / "selected-project"
        _link(selected, root, directory=True)
    sources = ["safe.py", "alias.py", "sub/../safe.py", str(root / "safe.py")]
    data = {"nodes": [], "links": []}
    for i, source in enumerate(sources):
        node = _graph(source, label="safe.py")["nodes"][0]
        node["id"] = str(i)
        data["nodes"].append(node)
    # An explicitly selected graph file may live outside the source root.
    graph = tmp_path / "imported-graph.json"
    graph.write_text(json.dumps(data))
    _guard_outside_reads(monkeypatch, outside)

    spans = extract_repo_spans(selected, max_workers=1)
    assert {"safe.py", "alias.py"} <= spans.keys()
    extracted = extract_repo(selected)
    assert {"safe.py", "alias.py"} <= {n["source_file"] for n in extracted["nodes"]}
    with Store.begin(selected) as store:
        ingest(selected, store, graph)
        rows = list(store.iter_nodes())
        assert [r["source_file"] for r in rows] == sources
        assert [r["node_key"] for r in rows] == [node_key(p, "safe.py", "file") for p in sources]
        pack = build_pack(store, selected, "fixture", _hits(store))
    assert [item.location for item in pack.items] == [f"{p}:1" for p in sources]
    assert all("def safe():" in item.text for item in pack.items)


def test_in_root_alias_extension_controls_parser_and_citation(project):
    root, _ = project
    source = root / "implementation.txt"
    source.write_text("def from_alias():\n    return 'local implementation'\n")
    alias = root / "alias.py"
    _link(alias, source)
    parsed = extract_spans(alias, root)
    assert parsed is not None
    assert parsed.path == "alias.py" and parsed.language == "python"
    assert [span.name for span in parsed.spans] == ["from_alias"]

    with Store.begin(root) as store:
        ingest(root, store)
        hits = [hit for hit in _hits(store) if hit.label == "from_alias"]
        pack = build_pack(store, root, "from_alias", hits)
    assert len(pack.items) == 1
    assert pack.items[0].location == "alias.py:1"
    assert "return 'local implementation'" in pack.items[0].text


def test_missing_sources_keep_metadata_and_empty_packs(project):
    root, _ = project
    with Store.begin(root) as store:
        result = ingest(root, store, data=_graph("missing.py"))
        assert result.nodes == 1
        pack = build_pack(store, root, "fixture", _hits(store))
    assert len(pack.items) == 1
    assert pack.items[0].location == "missing.py:1"
    assert pack.items[0].text == ""


def test_broken_and_looping_links_are_skipped(project):
    root, _ = project
    _link(root / "broken.py", root / "absent.py")
    _link(root / "loop.py", root / "loop.py")
    for reader in (extract_repo, extract_repo_spans):
        reader(root)
    assert extract_repo_spans(
        root, files=[root / "broken.py", root / "loop.py"], max_workers=1) == {}
    with Store.begin(root) as store:
        ingest(root, store, data=_graph("loop.py"), parse_spans=False)
        pack = build_pack(store, root, "fixture", _hits(store))
    assert pack.items[0].text == ""


def test_standalone_span_parser_without_root_keeps_explicit_file_api(project):
    _, outside = project
    source = outside / "standalone.py"
    source.write_text("def standalone():\n    return 1\n")
    parsed = extract_spans(source)
    assert parsed is not None
    assert parsed.path == str(source)
    assert [span.name for span in parsed.spans] == ["standalone"]
