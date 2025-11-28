from collections import defaultdict, deque
from typing import Callable, Iterable, TypeVar

Node = TypeVar("Node")


def kahn(
    nodes: set[Node],
    adjs: Callable[[Node], Iterable[Node]],
) -> list[Node]:
    """Run Kahn's algorithm for topological sorting.

    Args:
        nodes: Set of all nodes in the directed graph
        adjs: Function that yields adjacent nodes (outgoing edges) for a given node

    Returns:
        Topologically sorted list of nodes. If the graph contains a cycle,
        returns a partial ordering (fewer nodes than input) because some nodes
        will be waiting on each other.

    """
    in_degree: dict[Node, int] = defaultdict(int)
    for node in nodes:
        for adj in adjs(node):
            in_degree[adj] += 1
    order: list[Node] = []
    q: deque[Node] = deque(node for node in nodes if in_degree[node] == 0)
    while q:
        node = q.popleft()
        order.append(node)
        for adj in adjs(node):
            in_degree[adj] -= 1
            if in_degree[adj] == 0:
                q.append(adj)
    return order
