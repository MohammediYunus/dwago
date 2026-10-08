"""Git mining: log parsing, weighting, and the significance gate."""
from __future__ import annotations

import subprocess

import pytest

from dwago.enrich.git_temporal import (_FS, _REC, TemporalConfig, _g_test,
                                         _parse_log, enrich_store, is_git_repo, mine_history)
from dwago.ingest import ingest
from dwago.store import Store


def _log(*commits: tuple[str, int, str, list[str]]) -> str:
    """Build synthetic `git log --numstat` output."""
    out = []
    for sha, ts, author, files in commits:
        out.append(f"{_REC}{sha}{_FS}{ts}{_FS}{author}")
        out.append("")
        for f in files:
            out.append(f"1\t1\t{f}")
    return "\n".join(out)


def test_parse_log_reads_commits_and_files():
    raw = _log(("abc", 1000, "alice", ["a.py", "b.py"]),
               ("def", 900, "bob", ["c.py"]))
    commits = _parse_log(raw)
    assert [c.sha for c in commits] == ["abc", "def"]
    assert commits[0].files == ["a.py", "b.py"]
    assert commits[1].author == "bob"


def test_record_separator_survives_parsing():
    """Regression: str.splitlines() treats \\x1e as a line break.

    Using splitlines() here consumed the record separator itself, so no line
    ever started with it and the parser silently returned zero commits — with
    no error and a perfectly healthy-looking git invocation.
    """
    raw = _log(("abc", 1000, "alice", ["a.py"]))
    assert _REC in raw
    assert len(raw.splitlines()) > len(raw.split("\n")), \
        "this test is meaningless if splitlines() stops splitting on \\x1e"
    assert len(_parse_log(raw)) == 1


def test_lockfiles_and_changelogs_are_filtered():
    raw = _log(("abc", 1000, "a", ["src/x.py", "uv.lock", "CHANGELOG.md",
                                   "package-lock.json"]))
    assert _parse_log(raw)[0].files == ["src/x.py"]


def test_renames_follow_to_the_new_path():
    raw = _log(("abc", 1000, "a", ["src/{old.py => new.py}"]))
    assert _parse_log(raw)[0].files == ["src/new.py"]


def test_g_test_rejects_independence():
    """Two files that co-occur exactly as often as chance predicts score ~0."""
    g, p = _g_test(n_ab=10, n_a=100, n_b=100, n_total=1000)
    assert g == pytest.approx(0.0, abs=1e-6)
    assert p > 0.9


def test_g_test_detects_real_coupling():
    g, p = _g_test(n_ab=40, n_a=50, n_b=50, n_total=1000)
    assert g > 50
    assert p < 1e-6


def test_g_test_ignores_negative_association():
    """Co-occurring *less* than chance is not evidence of coupling."""
    g, p = _g_test(n_ab=1, n_a=100, n_b=100, n_total=1000)
    assert g == 0.0
    assert p == 1.0


def test_g_test_handles_degenerate_input():
    assert _g_test(0, 0, 0, 0) == (0.0, 1.0)
    assert _g_test(5, 3, 5, 10)[0] >= 0.0   # n_ab > n_a must not explode


@pytest.mark.skipif(not hasattr(subprocess, "run"), reason="needs subprocess")
def test_non_git_directory_degrades_cleanly(tmp_path):
    assert is_git_repo(tmp_path) is False
    commits, result = mine_history(tmp_path, TemporalConfig())
    assert commits == []
    assert any("not a git repository" in w for w in result.warnings)


def test_before_ts_excludes_the_evaluation_window(tmp_path, monkeypatch):
    """The leakage guard the eval harness depends on."""
    import dwago.enrich.git_temporal as gt

    raw = _log(("new", 2000, "a", ["x.py"]), ("old", 500, "a", ["y.py"]))
    monkeypatch.setattr(gt, "is_git_repo", lambda p: True)
    monkeypatch.setattr(gt, "_git", lambda root, *a, **k: raw if a[0] == "log" else "HEAD")
    commits, _ = gt.mine_history(tmp_path, TemporalConfig(before_ts=1000))
    assert [c.sha for c in commits] == ["old"]


def test_extracted_paths_join_native_git_history(tmp_path):
    """Extraction and Git must address the same nested files on every OS."""
    source = tmp_path / "app" / "nested" / "module.py"
    source.parent.mkdir(parents=True)
    source.write_text("def answer():\n    return 42\n", encoding="utf-8")
    for args in (("init",), ("add", "app"),
                 ("-c", "user.name=Fixture", "-c", "user.email=fixture@example.com",
                  "-c", "commit.gpgsign=false", "commit", "-m", "Add module")):
        subprocess.run(["git", "-C", str(tmp_path), *args],
                       check=True, capture_output=True)

    with Store.begin(tmp_path, inherit=False) as store:
        result = ingest(tmp_path, store)
        assert result.spans_matched > 0
        temporal = enrich_store(tmp_path, store)
        assert temporal.commits_scanned == 1
        rows = store.conn.execute(
            "SELECT n.source_file, f.language, f.lines, f.n_commits, f.hotspot "
            "FROM nodes n JOIN files f ON n.source_file=f.path WHERE n.kind='file'"
        ).fetchall()
        assert len(rows) == 1
        row = rows[0]
        assert row["source_file"] == "app/nested/module.py"
        assert row["language"] == "python"
        assert row["lines"] == 2
        assert row["n_commits"] == 1
        assert row["hotspot"] > 0
        assert store.conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 1
