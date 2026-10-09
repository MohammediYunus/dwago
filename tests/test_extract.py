"""Standalone extraction: walker, symbols, import resolution, communities."""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
import subprocess
import sys

import pytest

from dwago.extract import _communities, _resolve_ts, extract_repo
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
    ("./util", r"literal\folder/api.ts", r"literal\folder/util.ts"),
])
def test_resolve_ts_relative_paths(spec, importer, expected):
    files = {"web/util.ts", "web/shared/index.ts", "outside.ts", "../outside.ts",
             r"literal\folder/util.ts"}
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


def test_ingest_utf8_imports_under_non_utf8_locale(tmp_path):
    modules = ["plain", "café", "日本語"]
    for module in modules:
        (tmp_path / f"{module}.py").write_text(
            "def answer():\n    return 1\n", encoding="utf-8")
    imports = "".join(f"from {module} import answer\n" for module in modules)
    (tmp_path / "consumer.py").write_text(imports, encoding="utf-8")

    script = """
import json
import locale
import sys
from dwago.ingest import ingest
from dwago.store import Store

# Change the text locale after startup to preserve Unicode filesystem support.
locale.setlocale(locale.LC_CTYPE, "C")
with Store.begin(sys.argv[1], inherit=False) as store:
    ingest(sys.argv[1], store)
    imports = [list(row) for row in store.conn.execute(
        "SELECT source.source_file, target.source_file FROM edges "
        "JOIN nodes AS source ON source.idx = edges.src "
        "JOIN nodes AS target ON target.idx = edges.dst "
        "WHERE edges.relation = 'imports' ORDER BY target.source_file")]
print(json.dumps({"imports": imports, "utf8_mode": sys.flags.utf8_mode,
                  "encoding": locale.getpreferredencoding(False)}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "PYTHONUTF8": "0"},
        capture_output=True, check=True, timeout=30,
    )
    result = json.loads(proc.stdout)
    assert result["utf8_mode"] == 0
    assert imports.encode("utf-8").decode(result["encoding"], errors="replace") != imports
    assert result["imports"] == [
        ["consumer.py", f"{module}.py"] for module in sorted(modules)
    ]


def test_communities_ignore_logical_input_order():
    files = [f"group{i}/f{i}.py" for i in range(9)]
    edges = [(files[i], files[(i + 1) % len(files)]) for i in range(len(files))]
    expected = _communities(files, edges)
    for seed in range(12):
        rng = random.Random(seed)
        reordered_files = files.copy()
        # Reversing or repeating an edge does not change an undirected graph.
        reordered_edges = edges + [(b, a) for a, b in edges]
        rng.shuffle(reordered_files)
        rng.shuffle(reordered_edges)
        assert _communities(reordered_files, reordered_edges) == expected


@pytest.mark.parametrize(("files", "expected_name"), [
    (["beta/a.py", "beta/b.py", "alpha/c.py", "alpha/d.py"], "alpha"),
    (["alpha/a.py", "beta/b.py", "beta/c.py", "beta/d.py"], "beta"),
    (["pkg/sub/a.py", "pkg/sub/b.py", "pkg/sub/c.py"], "pkg/sub"),
    (["a.py", "b.py", "c.py"], "(root)"),
])
def test_community_names_keep_common_paths_and_majorities(files, expected_name):
    edges = [(a, b) for i, a in enumerate(files) for b in files[i + 1:]]
    assert set(_communities(files, edges).values()) == {(0, expected_name)}


def test_isolated_communities_keep_size_order_with_stable_ties():
    assert _communities([], []) == {}
    files = ["z/a.py", "z/b.py", "beta/a.py", "alpha/a.py"]
    assert _communities(files, []) == {
        "z/a.py": (0, "z"), "z/b.py": (0, "z"),
        "alpha/a.py": (1, "alpha"), "beta/a.py": (2, "beta"),
    }


@pytest.mark.parametrize("fixture", ["pair", "clique", "cycle"])
def test_extracted_community_metadata_survives_hash_randomization(tmp_path, fixture):
    if fixture == "pair":
        files = ["alpha/a.py", "beta/b.py"]
        edges = [(0, 1)]
    elif fixture == "clique":
        files = ["alpha/a.py", "alpha/b.py", "beta/c.py", "beta/d.py"]
        edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    else:
        files = [f"group{i}/f{i}.py" for i in range(9)]
        edges = [(i, (i + 1) % len(files)) for i in range(len(files))]

    for i, rel in enumerate(files):
        neighbors = sorted({b if a == i else a for a, b in edges if i in (a, b)})
        imports = [f"import {files[j][:-3].replace('/', '.')}\n" for j in neighbors]
        path = tmp_path / rel
        path.parent.mkdir(exist_ok=True)
        path.write_text("".join(imports) + "\ndef marker():\n    pass\n")

    script = """
import json
import sys
from dwago.extract import extract_repo
graph = extract_repo(sys.argv[1])
print(json.dumps({
    "communities": {n["id"]: [n["community"], n["community_name"]]
                    for n in graph["nodes"]},
    "imports": sorted([e["source"], e["target"]] for e in graph["links"]
                      if e["relation"] == "imports"),
}))
"""
    results = []
    for seed in (0, 1, 2, 7):
        proc = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path)],
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
            capture_output=True, text=True, check=True,
        )
        results.append(json.loads(proc.stdout))
    expected_imports = sorted(
        [files[a], files[b]] for a, b in edges + [(b, a) for a, b in edges]
    )
    assert results[0]["imports"] == expected_imports
    assert len(results[0]["communities"]) == 2 * len(files)  # files and symbols
    assert all(result == results[0] for result in results[1:])
