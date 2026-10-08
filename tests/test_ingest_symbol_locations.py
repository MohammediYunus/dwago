"""Repeated symbol names must retain their own parsed source ranges."""
from contextlib import closing
from pathlib import Path

import pytest

from dwago.ingest import ingest
from dwago.store import Store


def test_csharp_overloads_keep_source_ranges_and_graph_ids(tmp_path):
    (tmp_path / "Reader.cs").write_text(
        "class Reader\n"
        "{\n"
        "    public int Read(int value)\n"
        "    {\n"
        "        return value;\n"
        "    }\n"
        "\n"
        "    public string Read(string value)\n"
        "    {\n"
        "        return value;\n"
        "    }\n"
        "\n"
        "    public bool Read(bool value)\n"
        "    {\n"
        "        return value;\n"
        "    }\n"
        "}\n")
    with Store.begin(tmp_path) as st:
        result = ingest(tmp_path, st)
    with closing(Store.open(tmp_path)) as st:
        rows = st.conn.execute(
            "SELECT graphify_id, start_line, end_line, signature FROM nodes "
            "WHERE label='Read' ORDER BY start_line").fetchall()
        incoming = st.conn.execute(
            "SELECT COUNT(*) FROM edges e JOIN nodes n ON e.dst=n.idx "
            "WHERE n.label='Read' AND e.relation='contains'").fetchone()[0]
    assert [tuple(r) for r in rows] == [
        ("Reader.cs::Read::3", 3, 6, "public int Read(int value)"),
        ("Reader.cs::Read::8", 8, 11, "public string Read(string value)"),
        ("Reader.cs::Read::13", 13, 16, "public bool Read(bool value)"),
    ]
    assert incoming == 3, "each original overload must remain reachable"
    assert result.duplicate_nodes == result.dropped_edges == 0


def test_same_name_methods_in_nested_and_sibling_classes(tmp_path):
    (tmp_path / "Readers.cs").write_text(
        "class Outer\n"
        "{\n"
        "    public int Run() { return 1; }\n"
        "    class Inner\n"
        "    {\n"
        "        public int Run() { return 2; }\n"
        "    }\n"
        "}\n"
        "class Other\n"
        "{\n"
        "    public int Run() { return 3; }\n"
        "}\n")
    with Store.begin(tmp_path) as st:
        ingest(tmp_path, st)
    with closing(Store.open(tmp_path)) as st:
        rows = st.conn.execute(
            "SELECT graphify_id, start_line, end_line FROM nodes "
            "WHERE label='Run' ORDER BY start_line").fetchall()
    assert [tuple(r) for r in rows] == [
        ("Readers.cs::Run::3", 3, 3),
        ("Readers.cs::Run::6", 6, 6),
        ("Readers.cs::Run::11", 11, 11),
    ]


@pytest.mark.parametrize("label", ["Read()", "Reader.Read()"])
def test_imported_graph_line_inside_attributed_declaration(tmp_path, label):
    (tmp_path / "Reader.cs").write_text(
        "class Reader\n"
        "{\n"
        "    [Marker]\n"
        "    public int Read(int value) { return value; }\n"
        "\n"
        "    [Marker]\n"
        "    public string Read(string value) { return value; }\n"
        "}\n")
    data = {"nodes": [
        {"id": "integer", "label": label, "file_type": "code",
         "source_file": "Reader.cs", "source_location": "L4"},
        {"id": "string", "label": label, "file_type": "code",
         "source_file": "Reader.cs", "source_location": "L7"},
    ], "links": [{"source": "integer", "target": "string", "relation": "related"}]}
    with Store.begin(tmp_path) as st:
        ingest(tmp_path, st, data=data)
    with closing(Store.open(tmp_path)) as st:
        rows = st.conn.execute(
            "SELECT graphify_id, start_line, end_line, signature FROM nodes "
            "ORDER BY start_line").fetchall()
    assert [tuple(r) for r in rows] == [
        ("integer", 3, 4, "[Marker] public int Read(int value)"),
        ("string", 6, 7, "[Marker] public string Read(string value)"),
    ]


def test_imported_native_paths_keep_identity_and_span_enrichment(tmp_path):
    path = Path("app") / "module.py"
    source = tmp_path / path
    source.parent.mkdir()
    source.write_text("def answer(value):\n    return value\n", encoding="utf-8")
    native_path = str(path)
    data = {"nodes": [
        {"id": "external-symbol", "label": "answer", "file_type": "code",
         "source_file": native_path, "source_location": "L1"},
    ], "links": []}
    with Store.begin(tmp_path) as st:
        result = ingest(tmp_path, st, data=data)
        row = st.conn.execute(
            "SELECT graphify_id, source_file, start_line, end_line, signature "
            "FROM nodes").fetchone()
        assert tuple(row) == ("external-symbol", native_path, 1, 2, "def answer(value):")
        assert result.spans_matched == 1
