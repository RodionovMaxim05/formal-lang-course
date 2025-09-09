from pyformlang.finite_automaton import Symbol

import networkx as nx
import pytest
import cfpq_data as cd

from project.graph_utils import get_graph_metadata
from project.automata_builder import graph_to_nfa
from tests.test_graph_utils import GRAPHS_EXAMPLES_PATH


@pytest.mark.parametrize(
    "nodes, start_states, final_states, expected_states",
    [
        ([], set(), set(), set()),
        ([], {0}, {1}, {0, 1}),
        ([0], {0, 999}, {0, 1}, {0, 1, 999}),
        ([1, 2], {1}, {2}, {1, 2}),
    ],
)
def test_graph_to_nfa_extreme_cases(
    nodes: list, start_states: set, final_states: set, expected_states: set
):
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(nodes)

    nfa = graph_to_nfa(graph, start_states, final_states)

    assert nfa.states == expected_states
    assert nfa.start_states == start_states
    assert nfa.final_states == final_states

# Testing with graphs using the functionality implemented in Task 1

def test_existing_graph_to_nfa():
    graph_name = "bzip"

    graph = cd.graph_from_csv(cd.download(graph_name))
    graph_metadata = get_graph_metadata(graph_name)

    nfa = graph_to_nfa(graph, set(), set())

    assert nfa.start_states == nfa.final_states == nfa.states
    assert len(nfa.states) == graph_metadata.node_count
    assert nfa.symbols == graph_metadata.edge_labels


def test_two_cycles_graph_to_nfa():
    filename = "two_cycles.dot"
    labels = ("A", "B")

    graph = nx.nx_pydot.read_dot(GRAPHS_EXAMPLES_PATH / filename)

    nfa = graph_to_nfa(graph, set(), set())

    assert nfa.start_states == nfa.final_states == nfa.states

    # Checking the correctness of cycles (one full cycle)
    word_cycle0 = [Symbol(labels[0])] * 6
    word_cycle1 = [Symbol(labels[1])] * 8

    assert nfa.accepts(word_cycle0)
    assert nfa.accepts(word_cycle1)
