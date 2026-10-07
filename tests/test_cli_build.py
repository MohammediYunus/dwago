"""CLI build confirmation must preserve the store's atomic publication."""
from contextlib import closing

import numpy as np
import pytest

from dwago.cli import main
from dwago.index import dense
from dwago.store import Store


@pytest.fixture
def slow_embeddings(monkeypatch):
    """Exercise the real build with a small encoder that needs confirmation."""
    calls = []

    class Encoder:
        available = True

        def __init__(self, info):
            self.info = info

        def estimate_seconds(self, n_docs):
            return 121

        def load(self):
            return self

        def encode(self, texts, **kwargs):
            calls.append(len(texts))
            return np.ones((len(texts), self.info.dim), dtype=np.float32)

        def release(self):
            pass

    monkeypatch.setattr(dense, "resolve_backend", lambda *args:
                        dense.BackendInfo("model2vec", "test-encoder", "cpu", 2))
    monkeypatch.setattr(dense, "Embedder", Encoder)
    return calls


@pytest.mark.parametrize("force", [False, True])
def test_declined_rebuild_preserves_current_index(tmp_path, slow_embeddings, force):
    source = tmp_path / "app.py"
    source.write_text("def original():\n    return 1\n")
    assert main(["build", str(tmp_path), "--no-git", "--yes"]) == 0
    with closing(Store.open(tmp_path)) as st:
        before = st.stats()
        epoch = st.paths.root
        vectors = st.paths.vectors.read_bytes()
        keys = st.load_vector_keys()
    assert before["vectors"] > 0
    slow_embeddings.clear()

    source.write_text("def replacement():\n    return 2\n\ndef extra():\n    return 3\n")
    args = ["build", str(tmp_path), "--no-git"] + (["--force"] if force else [])
    assert main(args) == 2

    with closing(Store.open(tmp_path)) as st:
        assert st.paths.root == epoch
        assert st.stats() == before
        assert st.paths.vectors.read_bytes() == vectors
        assert st.load_vector_keys() == keys
    assert list(epoch.parent.iterdir()) == [epoch], "discard the unfinished epoch"
    assert slow_embeddings == [], "declining must not start the encoder"


def test_declined_first_build_does_not_publish(tmp_path, slow_embeddings, capsys):
    (tmp_path / "app.py").write_text("def answer():\n    return 42\n")
    assert main(["build", str(tmp_path), "--no-git"]) == 2
    assert not Store.exists(tmp_path)
    assert list((Store.out_dir_for(tmp_path) / "epochs").iterdir()) == []
    assert slow_embeddings == []
    assert "re-run with --yes" in capsys.readouterr().out


def test_confirmed_long_build_publishes_embeddings(tmp_path, slow_embeddings):
    (tmp_path / "app.py").write_text("def answer():\n    return 42\n")
    assert main(["build", str(tmp_path), "--no-git", "--yes"]) == 0
    assert slow_embeddings
    with closing(Store.open(tmp_path)) as st:
        assert st.stats()["vectors"] == st.node_count() > 0
        assert st.get_meta("embedding_model") == "test-encoder"
