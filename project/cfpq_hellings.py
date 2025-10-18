from pyformlang.cfg import CFG, Production, Variable, Epsilon, Terminal
import networkx as nx
from collections import deque


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    nullable_symbols = cfg.get_nullable_symbols()
    cfg_in_cnf = cfg.to_normal_form()

    cfg_prod = set(cfg_in_cnf.productions)
    for symbol in nullable_symbols:
        cfg_prod.add(Production(Variable(symbol.value), [Epsilon()]))

    return CFG(
        start_symbol=cfg.start_symbol, productions=cfg_prod
    ).remove_useless_symbols()


def hellings_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    wcnf = cfg_to_weak_normal_form(cfg)

    cfpq_result = set()
    queue = deque()

    wcnf_nullable_symbols = wcnf.get_nullable_symbols()
    for node in graph.nodes:
        for sym in wcnf_nullable_symbols:
            cfpq_result.add((sym, node, node))
            queue.append((sym, node, node))

    for u, v, label in graph.edges(data="label"):
        for production in wcnf.productions:
            if [Terminal(label)] == production.body:
                cfpq_result.add((production.head, u, v))
                queue.append((production.head, u, v))

    def helper(triple1, triple2, temp: set):
        (N, a, b) = triple1
        (M, c, d) = triple2

        if b == c:
            for production in wcnf.productions:
                if [N, M] == production.body:
                    triple = (production.head, a, d)
                    if triple not in cfpq_result:
                        queue.append(triple)
                        temp.add(triple)

    while queue:
        triple1 = queue.popleft()
        temp = set()

        for triple2 in cfpq_result:
            helper(triple1, triple2, temp)
            helper(triple2, triple1, temp)

        cfpq_result |= temp

    result_pairs = set()
    for sym, u, v in cfpq_result:
        if sym == wcnf.start_symbol:
            if (not start_nodes or u in start_nodes) and (
                not final_nodes or v in final_nodes
            ):
                result_pairs.add((u, v))

    return result_pairs
