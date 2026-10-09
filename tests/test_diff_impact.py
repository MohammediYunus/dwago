"""Diff impact joins literal Git paths to indexed files and their neighbours."""
from __future__ import annotations

import asyncio
import os
import subprocess

import pytest

from dwago.ingest import ingest
from dwago.store import Store


@pytest.fixture()
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")

    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args],
                              check=True, capture_output=True)

    def commit(message):
        git("add", "--all")
        git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
            "-c", "commit.gpgsign=false", "commit", "-qm", message)

    def build(filename, *, rename=False, indexed=True):
        git("init", "-q")
        original = "old.py" if rename else filename
        source = tmp_path / original
        source.write_text("def answer():\n    return 1\n", encoding="utf-8")
        module = filename.removesuffix(".py")
        caller = (f"from {module} import answer\n\n" if module.isidentifier() else "")
        (tmp_path / "consumer.py").write_text(
            caller + "def use_answer():\n    return answer()\n", encoding="utf-8")
        commit("Add source and caller")
        if rename:
            git("mv", original, filename)
        else:
            source.write_text("def answer():\n    return 2\n", encoding="utf-8")
        commit("Update source")
        with Store.begin(tmp_path, inherit=False) as store:
            if indexed:
                ingest(tmp_path, store)
                assert store.conn.execute(
                    "SELECT 1 FROM nodes WHERE source_file=?", (filename,)).fetchone()
        return tmp_path

    return build


def call_diff(root, rev_range="HEAD~1..HEAD"):
    pytest.importorskip("mcp")
    from dwago.serve import build_server

    server = build_server(root)

    async def call():
        response = await server.call_tool("diff_impact", {"rev_range": rev_range})
        return "\n".join(getattr(part, "text", str(part))
                         for part in getattr(response, "content", response))

    return asyncio.run(call())


POSIX_ONLY = pytest.mark.skipif(os.name == "nt", reason="filename is invalid on Windows")


@pytest.mark.parametrize("filename", [
    "plain.py", "monthly report.py", " leading name.py", "café.py", "日本語.py",
    pytest.param("quarter\treport.py", marks=POSIX_ONLY),
    pytest.param("new\nline.py", marks=POSIX_ONLY),
    pytest.param("carriage\rreturn.py", marks=POSIX_ONLY),
    pytest.param('quoted"name.py', marks=POSIX_ONLY),
    pytest.param("back\\slash.py", marks=POSIX_ONLY),
])
def test_diff_impact_matches_literal_indexed_paths(project, filename):
    root = project(filename)
    text = call_diff(root)

    assert text.startswith("1 files changed in HEAD~1..HEAD:\n")
    assert f"\n  {filename}" in text
    assert "not in the graph" not in text
    if filename.removesuffix(".py").isidentifier():
        assert "Ripples into (structural + co-change neighbours):" in text
        assert "consumer.py  (1 connection(s) to the change)" in text
    config = subprocess.run(["git", "-C", str(root), "config", "--local",
                             "--get", "core.quotePath"], capture_output=True)
    assert config.returncode == 1


def test_diff_impact_unicode_rename_uses_destination(project):
    text = call_diff(project("café.py", rename=True))
    assert text.startswith("1 files changed in HEAD~1..HEAD:\n  café.py")
    assert "old.py" not in text
    assert "not in the graph" not in text
    assert "consumer.py  (1 connection(s) to the change)" in text


def test_diff_impact_empty_diff(project):
    assert call_diff(project("plain.py"), "HEAD..HEAD") == "No files changed in HEAD..HEAD."


def test_diff_impact_invalid_revision(project):
    text = call_diff(project("plain.py"), "missing_revision..HEAD")
    assert text.startswith("git diff failed: ")
    assert "missing_revision..HEAD" in text
    assert not text.startswith("git diff failed: b'")


def test_diff_impact_unindexed_file(project):
    text = call_diff(project("café.py", indexed=False))
    assert "\n  café.py" in text
    assert "(1 not in the graph - new or unindexed)" in text
    assert "Ripples into" not in text
