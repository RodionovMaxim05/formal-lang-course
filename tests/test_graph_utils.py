from pathlib import Path

import networkx as nx
import pytest

from project.graph_utils import (
    GraphMetadata,
    get_graph_metadata,
    build_two_cycles_graph,
)

GRAPHS_EXAMPLES_PATH = Path(__file__).parent / "graphs_examples"


def test_get_graph_metadata_for_existing_graph():
    bzip_md = GraphMetadata(node_count=632, edge_count=556, edge_labels={"d", "a"})

    actual_md = get_graph_metadata("bzip")

    assert actual_md.node_count == bzip_md.node_count
    assert actual_md.edge_count == bzip_md.edge_count
    assert actual_md.edge_labels == bzip_md.edge_labels


def test_get_graph_metadata_for_nonexistent_graph():
    with pytest.raises(FileNotFoundError):
        get_graph_metadata("aaaaa")


def test_build_two_cycles_graph(tmp_path: Path):
    filename = "two_cycles.dot"

    build_two_cycles_graph(5, 7, ("A", "B"), tmp_path / filename)
    actual_graph = nx.nx_pydot.read_dot(tmp_path / filename)

    expected_graph = nx.nx_pydot.read_dot(GRAPHS_EXAMPLES_PATH / filename)

    assert nx.utils.graphs_equal(actual_graph, expected_graph)
