"""Graph metadata must remain text across HTML and script boundaries."""
from contextlib import closing
from html.parser import HTMLParser
import json

import pytest

from dwago.store import Store
from dwago.viz.build_brain import build_payload, write_brain
from dwago.viz.build_html import collect_graph_data, write_html


class Document(HTMLParser):
    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.scripts = []
        self.title = ""
        self.markers = []
        self.current = None
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        if tag.startswith("x-"):
            self.markers.append(tag)
        if tag == "script":
            self.scripts.append("")
        if tag in ("script", "title"):
            self.current = tag

    def handle_endtag(self, tag):
        if tag == self.current:
            self.current = None

    def handle_data(self, data):
        if self.current == "script":
            self.scripts[-1] += data
        elif self.current == "title":
            self.title += data


@pytest.fixture
def metadata_store(tmp_path):
    marker = '</ScRiPt><x-file> & <!--<script> snowman: \u2603'
    with Store.begin(tmp_path) as st:
        st.write_nodes([dict(
            idx=0, node_key="fixture", graphify_id="fixture", label=marker,
            kind="file", file_type="code", source_file=f"src/{marker}.py",
            start_line=1, end_line=1, signature=None, docstring=None,
            community=0, community_name="</ScRiPt><x-community>",
            content_hash="fixture", doc=marker,
        )])
    with closing(Store.open(tmp_path)) as st:
        yield st


@pytest.mark.parametrize("writer,suffix,count", [
    (write_brain, "brain", 1), (write_html, "dwago", 2),
])
def test_html_boundaries_and_metadata_roundtrip(
    tmp_path, metadata_store, writer, suffix, count,
):
    title = '</title><x-title> & </ScRiPt><x-script> "quoted" \u2603'
    out = tmp_path / "map.html"
    writer(metadata_store, out, title=title)
    document = Document(out.read_text())
    assert document.markers == [], "metadata must not create HTML elements"
    assert len(document.scripts) == count
    assert document.title == f"{title} \u00b7 {suffix}"
    if writer is write_brain:
        raw = document.scripts[0].split("window.DWAGO_DATA=", 1)[1]
        data, end = json.JSONDecoder().raw_decode(raw)
        assert "<" not in raw[:end], "avoid script end and escaped-comment states"
        assert data == build_payload(metadata_store)
    else:
        assert "<" not in document.scripts[0]
        assert json.loads(document.scripts[0]) == collect_graph_data(metadata_store)
        raw = document.scripts[1].split("document.getElementById('brandsub').textContent = ", 1)[1]
        decoded_title, end = json.JSONDecoder().raw_decode(raw)
        assert "<" not in raw[:end]
        assert decoded_title == title


def test_title_template_markers_are_literal(tmp_path, metadata_store):
    title = "__TITLE_JS__ __TITLE__ __DATA__"
    out = tmp_path / "map.html"
    write_html(metadata_store, out, title=title)
    document = Document(out.read_text())
    assert document.title == f"{title} \u00b7 dwago"
    raw = document.scripts[1].split("document.getElementById('brandsub').textContent = ", 1)[1]
    assert json.JSONDecoder().raw_decode(raw)[0] == title
