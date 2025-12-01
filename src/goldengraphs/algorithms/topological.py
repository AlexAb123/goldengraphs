from collections import defaultdict, deque
from typing import Callable, Iterable, TypeVar

Node = TypeVar("Node")

# Step callback receives algorithm state at each iteration:
# - current: the node being processed
# - order: nodes processed so far (in order)
# - in_degree: current in-degree for each node
# - queue: current queue contents
StepCallback = Callable[[Node, list[Node], dict[Node, int], list[Node]], None]


def kahn(
    nodes: set[Node],
    adjs: Callable[[Node], Iterable[Node]],
    on_step: StepCallback[Node] | None = None,
) -> list[Node]:
    """Run Kahn's algorithm for topological sorting.

    Args:
        nodes: Set of all nodes in the directed graph
        adjs: Function that yields adjacent nodes (outgoing edges) for a given node
        on_step: Optional callback called at each step with algorithm state

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

        if on_step:
            on_step(node, order.copy(), dict(in_degree), list(q))

        for adj in adjs(node):
            in_degree[adj] -= 1
            if in_degree[adj] == 0:
                q.append(adj)
    return order
