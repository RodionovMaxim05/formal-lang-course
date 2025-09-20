from typing import Iterable, Set, Tuple, Dict
from scipy import sparse
from pyformlang.finite_automaton import (
    NondeterministicFiniteAutomaton,
    Symbol,
    State,
)
from networkx import MultiDiGraph

from project.automata_builder import regex_to_dfa, graph_to_nfa


class AdjacencyMatrixFA:
    states_count: int
    alphabet: Set[Symbol]
    state_to_idx: Dict[State, int]
    start_idxs: Set[int]
    final_idxs: Set[int]
    transition_matrices: Dict[Symbol, sparse.csr_array]

    def __init__(self, automaton: NondeterministicFiniteAutomaton = None):
        if automaton is None:
            self._init_empty()
        else:
            self._init_from_automaton(automaton)

    def _init_empty(self) -> None:
        self.states_count = 0
        self.alphabet = set()
        self.transition_matrices = {}
        self.state_to_idx = {}
        self.start_idxs = set()
        self.final_idxs = set()

    def _init_from_automaton(self, automaton: NondeterministicFiniteAutomaton) -> None:
        self.transition_matrices = {}
        self.states_count = len(automaton.states)
        self.alphabet = automaton.symbols

        self.state_to_idx = {state: idx for idx, state in enumerate(automaton.states)}
        self.start_idxs = {self.state_to_idx[state] for state in automaton.start_states}
        self.final_idxs = {self.state_to_idx[state] for state in automaton.final_states}

        graph = automaton.to_networkx()

        for symbol in self.alphabet:
            self.transition_matrices[symbol] = sparse.lil_array(
                (self.states_count, self.states_count), dtype=bool
            )

        for src, dst, label in graph.edges(data="label"):
            if label is not None:
                self.transition_matrices[label][
                    self.state_to_idx[src], self.state_to_idx[dst]
                ] = True

        for symbol in self.alphabet:
            self.transition_matrices[symbol] = self.transition_matrices[symbol].tocsr()

    def accepts(self, word: Iterable[Symbol]) -> bool:
        cur_vector = sparse.lil_array((1, self.states_count), dtype=bool)

        for start_idx in self.start_idxs:
            cur_vector[0, start_idx] = True

        cur_vector = cur_vector.tocsr()

        for symbol in word:
            if symbol not in self.alphabet:
                return False

            cur_vector = cur_vector @ self.transition_matrices[symbol]
            if cur_vector.count_nonzero() == 0:
                return False

        for final_idx in self.final_idxs:
            if cur_vector[0, final_idx]:
                return True

        return False

    def transitive_closure(self) -> sparse.csr_array:
        matrix_tc = sparse.eye_array(self.states_count, dtype=bool, format="csr")

        for symbol in self.alphabet:
            matrix_tc += self.transition_matrices[symbol]

        for _ in range(self.states_count):
            prev_count = matrix_tc.count_nonzero()
            matrix_tc = matrix_tc @ matrix_tc

            if prev_count == matrix_tc.count_nonzero():
                return matrix_tc

        return matrix_tc

    def is_empty(self) -> bool:
        trans_closure = self.transitive_closure()

        for start_idx in self.start_idxs:
            for final_idx in self.final_idxs:
                if trans_closure[start_idx, final_idx]:
                    return False

        return True


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    intersect = AdjacencyMatrixFA()

    intersect.states_count = automaton1.states_count * automaton2.states_count
    intersect.alphabet = automaton1.alphabet.intersection(automaton2.alphabet)

    for symbol in intersect.alphabet:
        intersect.transition_matrices[symbol] = sparse.kron(
            automaton1.transition_matrices[symbol],
            automaton2.transition_matrices[symbol],
            format="csr",
        )

    for state1 in automaton1.state_to_idx.keys():
        for state2 in automaton2.state_to_idx.keys():
            idx1, idx2 = (
                automaton1.state_to_idx[state1],
                automaton2.state_to_idx[state2],
            )
            inter_idx = idx1 * automaton2.states_count + idx2
            intersect.state_to_idx[State((state1.value, state2.value))] = inter_idx

            if idx1 in automaton1.start_idxs and idx2 in automaton2.start_idxs:
                intersect.start_idxs.add(inter_idx)

            if idx1 in automaton1.final_idxs and idx2 in automaton2.final_idxs:
                intersect.final_idxs.add(inter_idx)

    return intersect


def tensor_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: Set[int], final_nodes: Set[int]
) -> Set[Tuple[int, int]]:
    graph_nfa = graph_to_nfa(graph, start_nodes, final_nodes)
    graph_mfa = AdjacencyMatrixFA(graph_nfa)

    regex_dfa = regex_to_dfa(regex)
    regex_mfa = AdjacencyMatrixFA(regex_dfa)

    inter_mfa = intersect_automata(graph_mfa, regex_mfa)
    inter_tc = inter_mfa.transitive_closure()

    result = set()

    for graph_start in graph_nfa.start_states:
        for graph_final in graph_nfa.final_states:
            for regex_start in regex_dfa.start_states:
                for regex_final in regex_dfa.final_states:
                    if inter_tc[
                        inter_mfa.state_to_idx[
                            State((graph_start.value, regex_start.value))
                        ],
                        inter_mfa.state_to_idx[
                            State((graph_final.value, regex_final.value))
                        ],
                    ]:
                        result.add((graph_start.value, graph_final.value))

    return result
