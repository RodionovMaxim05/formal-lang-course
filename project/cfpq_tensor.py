import networkx as nx
from pyformlang.rsa import RecursiveAutomaton
from pyformlang.cfg import CFG
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton
from scipy import sparse
from typing import Set
from collections import defaultdict

from project.adjacency_matrix_fa import AdjacencyMatrixFA, intersect_automata
from project.automata_builder import graph_to_nfa


def rsm_to_nfa(rsm: RecursiveAutomaton) -> NondeterministicFiniteAutomaton:
    nfa = NondeterministicFiniteAutomaton()

    for nonterm, box in rsm.boxes.items():
        for u, v, label in box.dfa.to_networkx().edges(data="label"):
            nfa.add_transition((nonterm, u), label, (nonterm, v))

        for s in box.dfa.start_states:
            nfa.add_start_state((nonterm, s))
        for f in box.dfa.final_states:
            nfa.add_final_state((nonterm, f))

    return nfa


def msbfs(
    inter_nfa: AdjacencyMatrixFA,
    graph_nfa: AdjacencyMatrixFA,
    rsm_nfa: AdjacencyMatrixFA,
) -> dict[str, set[tuple[int, int]]]:
    num_inter_states = inter_nfa.states_count

    comb_trans_matrix = sparse.csr_array(
        (num_inter_states, num_inter_states), dtype=bool
    )
    for matrix in inter_nfa.transition_matrices.values():
        comb_trans_matrix += matrix
    comb_trans_matrix = comb_trans_matrix.transpose()

    idx_to_state = {idx: st for st, idx in inter_nfa.state_to_idx.items()}

    inter_start_idxs = set()
    inter_final_idxs = set()
    for idx, st in idx_to_state.items():
        nonterm, rsm_state = st.value[1]

        if rsm_nfa.state_to_idx[(nonterm, rsm_state)] in rsm_nfa.start_idxs:
            inter_start_idxs.add(idx)
        if rsm_nfa.state_to_idx[(nonterm, rsm_state)] in rsm_nfa.final_idxs:
            inter_final_idxs.add(idx)

    inter_start_idxs_list = list(inter_start_idxs)
    num_start_idxs = len(inter_start_idxs_list)

    blocks = []
    for start_idx in inter_start_idxs_list:
        block = sparse.csr_array((num_inter_states, 1), dtype=bool)
        block[start_idx, 0] = True
        blocks.append(block)

    front = sparse.vstack(blocks)
    visited = front.copy()

    while front.count_nonzero() > 0:
        new_front_parts = []
        for i in range(num_start_idxs):
            new_slice = (
                comb_trans_matrix
                @ front[i * num_inter_states : (i + 1) * num_inter_states]
            )
            new_front_parts.append(new_slice)

        new_front = sparse.vstack(new_front_parts)
        front = new_front > visited
        visited += front

    reachable_pairs_by_nonterm: dict[str, set[tuple[int, int]]] = defaultdict(set)

    for i, start_idx in enumerate(inter_start_idxs_list):
        for final_idx in inter_final_idxs:
            if visited[i * num_inter_states + final_idx, 0]:
                start_graph_state, (start_nonterm, _) = idx_to_state[start_idx].value
                final_graph_state, (final_nonterm, _) = idx_to_state[final_idx].value

                if start_nonterm != final_nonterm:
                    continue

                start_graph_idx = graph_nfa.state_to_idx[start_graph_state]
                final_graph_idx = graph_nfa.state_to_idx[final_graph_state]

                reachable_pairs_by_nonterm[start_nonterm].add(
                    (start_graph_idx, final_graph_idx)
                )

    return reachable_pairs_by_nonterm


def tensor_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: Set[int] = None,
    final_nodes: Set[int] = None,
) -> set[tuple[int, int]]:
    rsm_nfa = AdjacencyMatrixFA(rsm_to_nfa(rsm))
    graph_nfa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))

    for nonterm in rsm.boxes:
        if nonterm not in graph_nfa.transition_matrices:
            graph_nfa.transition_matrices[nonterm] = sparse.csr_array(
                (graph_nfa.states_count, graph_nfa.states_count), dtype=bool
            )
    graph_nfa.alphabet = graph_nfa.alphabet.union(rsm.labels)

    changed = True
    while changed:
        changed = False

        intersection_nfa = intersect_automata(graph_nfa, rsm_nfa)

        new_reachable_pairs_by_nonterm = msbfs(intersection_nfa, graph_nfa, rsm_nfa)

        for nonterm, pairs in new_reachable_pairs_by_nonterm.items():
            for u, v in pairs:
                if not graph_nfa.transition_matrices[nonterm][u, v]:
                    graph_nfa.transition_matrices[nonterm][u, v] = True
                    changed = True

    result = set()
    idx_to_state_graph = {idx: st for st, idx in graph_nfa.state_to_idx.items()}
    for u, v in zip(*graph_nfa.transition_matrices[rsm.initial_label].nonzero()):
        if u in graph_nfa.start_idxs and v in graph_nfa.final_idxs:
            result.add((idx_to_state_graph[u].value, idx_to_state_graph[v].value))

    return result


def cfg_to_rsm(cfg: CFG) -> RecursiveAutomaton:
    return RecursiveAutomaton.from_text(cfg.to_text())


def ebnf_to_rsm(ebnf: str) -> RecursiveAutomaton:
    return RecursiveAutomaton.from_text(ebnf)
