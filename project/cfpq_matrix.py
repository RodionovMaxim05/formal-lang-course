from pyformlang.cfg import CFG, Terminal
import networkx as nx
from typing import Set, Tuple, Dict
from scipy import sparse as sp
from project.cfpq_hellings import cfg_to_weak_normal_form


def build_rev_production_maps(
    wcnf: CFG,
) -> Tuple[Dict[Terminal, Set], Dict[Tuple, Set]]:
    term_to_vars = {}
    pair_to_heads = {}

    for prod in wcnf.productions:
        body = prod.body

        if len(body) == 1 and isinstance(body[0], Terminal):
            term_to_vars.setdefault(body[0], set()).add(prod.head)
        elif len(body) == 2:
            pair_to_heads.setdefault((body[0], body[1]), set()).add(prod.head)

    return term_to_vars, pair_to_heads


def matrix_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: Set[int] = None,
    final_nodes: Set[int] = None,
) -> set[tuple[int, int]]:
    graph_idx_to_node = list(graph.nodes)
    graph_node_to_idx = {node: i for i, node in enumerate(graph_idx_to_node)}

    wcnf = cfg_to_weak_normal_form(cfg)
    term_to_vars, pair_to_heads = build_rev_production_maps(wcnf)

    node_count = len(graph.nodes)
    decomposition = {
        var: sp.csr_matrix((node_count, node_count), dtype=bool)
        for var in wcnf.variables
    }

    for u, v, label in graph.edges(data="label"):
        terminal = Terminal(label)
        if terminal in term_to_vars:
            for var in term_to_vars[terminal]:
                decomposition[var][graph_node_to_idx[u], graph_node_to_idx[v]] = True

    for var in wcnf.get_nullable_symbols():
        decomposition[var].setdiag(True)

    changed = True
    while changed:
        changed = False

        for (N_i, N_j), heads in pair_to_heads.items():
            product = decomposition[N_i] @ decomposition[N_j]

            for head in heads:
                prev_count = decomposition[head].count_nonzero()

                decomposition[head] += product

                if decomposition[head].count_nonzero() != prev_count:
                    changed = True

    start_nodes = start_nodes if start_nodes else graph.nodes
    final_nodes = final_nodes if final_nodes else graph.nodes

    result = set()
    start_matrix = decomposition[wcnf.start_symbol]
    rows, cols = start_matrix.nonzero()
    for i, j in zip(rows, cols):
        u, v = graph_idx_to_node[i], graph_idx_to_node[j]
        if u in start_nodes and v in final_nodes:
            result.add((u, v))

    return result
