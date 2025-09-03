from dataclasses import dataclass
from typing import Tuple
from pathlib import Path

import cfpq_data as cd
import networkx as nx


@dataclass
class GraphMetadata:
    node_count: int
    edge_count: int
    edge_labels: set


def get_graph_metadata(graph_name: str) -> GraphMetadata:
    path = cd.download(graph_name)
    graph = cd.graph_from_csv(path)

    return GraphMetadata(
        node_count=graph.number_of_nodes(),
        edge_count=graph.number_of_edges(),
        edge_labels={label for _, _, label in graph.edges(data="label")},
    )


def build_two_cycles_graph(
    fst_cycle_node_count: int,
    snd_cycle_node_count: int,
    labels: Tuple[str, str],
    output_path: Path,
):
    graph = cd.labeled_two_cycles_graph(
        fst_cycle_node_count, snd_cycle_node_count, labels=labels
    )

    nx.nx_pydot.write_dot(graph, output_path)
