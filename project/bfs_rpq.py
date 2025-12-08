from networkx import MultiDiGraph

from project.automata_builder import regex_to_dfa, graph_to_nfa
from project.adjacency_matrix_fa import AdjacencyMatrixFA, MatrixType
from scipy.sparse import csr_array


def ms_bfs_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
    matrix_type: MatrixType = csr_array,
) -> set[tuple[int, int]]:
    graph_mfa = AdjacencyMatrixFA(
        graph_to_nfa(graph, start_nodes, final_nodes), matrix_type
    )
    regex_mfa = AdjacencyMatrixFA(regex_to_dfa(regex), matrix_type)

    common_alphabet = graph_mfa.alphabet.intersection(regex_mfa.alphabet)

    fronts = []
    reachable = []
    for graph_start in graph_mfa.start_idxs:
        front = matrix_type(
            (graph_mfa.states_count, regex_mfa.states_count), dtype=bool
        )

        for regex_start in regex_mfa.start_idxs:
            front[graph_start, regex_start] = True
        fronts.append(front)
        reachable.append(front.copy())

    graph_mfa_trans_matrices_transposed = {
        symbol: matrix.transpose()
        for symbol, matrix in graph_mfa.transition_matrices.items()
    }

    while any(f.count_nonzero() != 0 for f in fronts):
        for idx, front in enumerate(fronts):
            if front.count_nonzero() == 0:
                continue

            new_front = matrix_type(front.shape, dtype=bool)
            for symbol in common_alphabet:
                new_front += (
                    graph_mfa_trans_matrices_transposed[symbol]
                    @ front
                    @ regex_mfa.transition_matrices[symbol]
                )

            fronts[idx] = new_front > reachable[idx]
            reachable[idx] += fronts[idx]

    graph_mfa_idx_to_state = {v: k for k, v in graph_mfa.state_to_idx.items()}

    result = set()
    for idx, graph_start in enumerate(graph_mfa.start_idxs):
        for regex_final in regex_mfa.final_idxs:
            reached = reachable[idx][:, [regex_final]].nonzero()[0]

            for graph_idx in reached:
                if graph_idx in graph_mfa.final_idxs:
                    result.add(
                        (
                            graph_mfa_idx_to_state[graph_start].value,
                            graph_mfa_idx_to_state[graph_idx].value,
                        )
                    )

    return result
