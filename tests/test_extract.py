"""Standalone extraction: walker, symbols, import resolution, communities."""
from __future__ import annotations

from pathlib import Path

import pytest

from dwago.extract import _resolve_ts, extract_repo
from dwago.ingest import ingest
from dwago.store import Store


def _fixture_repo(tmp_path: Path) -> Path:
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "auth.py").write_text(
        "from app import db\n\n"
        "class Session:\n"
        "    def refresh(self):\n"
        "        return db.get()\n")
    (tmp_path / "app" / "db.py").write_text(
        "def get():\n    return 1\n")
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "api.ts").write_text(
        "import { helper } from './util'\n"
        "export function handler() { return helper() }\n")
    (tmp_path / "web" / "util.ts").write_text(
        "export function helper() { return 1 }\n")
    (tmp_path / "web" / "components").mkdir()
    (tmp_path / "web" / "components" / "button.tsx").write_text(
        "import { helper } from '../util'\n"
        "export function Button() { return helper() }\n")
    (tmp_path / "README.md").write_text("# fixture\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.js").write_text("x")
    return tmp_path


def test_extract_repo(tmp_path):
    data = extract_repo(_fixture_repo(tmp_path))
    ids = {n["id"] for n in data["nodes"]}
    assert "app/auth.py" in ids and "README.md" in ids
    assert not any("node_modules" in i for i in ids), "junk dirs skipped"
    labels = {n["label"] for n in data["nodes"]}
    assert {"Session", "refresh", "get", "handler", "helper"} <= labels
    rels = {(l["source"], l["target"]) for l in data["links"]
            if l["relation"] == "imports"}
    assert ("app/auth.py", "app/db.py") in rels, "python import resolved"
    assert ("web/api.ts", "web/util.ts") in rels, "ts relative import resolved"
    assert ("web/components/button.tsx", "web/util.ts") in rels, \
        "ts parent-relative import resolved"
    contains = [l for l in data["links"] if l["relation"] == "contains"]
    assert any(l["source"].startswith("app/auth.py::Session") for l in contains), \
        "method nested under its class"
    assert all("community" in n for n in data["nodes"] if n["source_file"])


@pytest.mark.parametrize(("spec", "importer", "expected"), [
    ("../util", "web/components/button.tsx", "web/util.ts"),
    ("../../util", "web/components/nested/button.tsx", "web/util.ts"),
    ("../shared", "web/components/button.tsx", "web/shared/index.ts"),
    ("../util.ts", "web/components/button.tsx", "web/util.ts"),
    ("./util", "web/api.ts", "web/util.ts"),
    ("./util.ts", "web/api.ts", "web/util.ts"),
    ("../../outside", "web/api.ts", None),
    ("../outside", "api.ts", None),
    ("react", "web/api.ts", None),
])
def test_resolve_ts_relative_paths(spec, importer, expected):
    files = {"web/util.ts", "web/shared/index.ts", "outside.ts", "../outside.ts"}
    assert _resolve_ts(spec, importer, files) == expected


def test_extract_end_to_end(tmp_path):
    root = _fixture_repo(tmp_path)
    with Store.begin(root, inherit=False) as st:
        res = ingest(root, st)          # no graph.json anywhere: own extraction
    assert res.nodes >= 9
    st = Store.open(root)
    row = st.conn.execute(
        "SELECT community_name FROM nodes WHERE source_file='app/auth.py' "
        "AND kind='file'").fetchone()
    assert row is not None and row["community_name"]
