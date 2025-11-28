from heapq import heappop, heappush
from typing import Callable, Iterable, TypeVar

Node = TypeVar("Node")


def dijkstra(
    source: Node,
    is_goal: Callable[[Node, float], bool],
    adjs: Callable[[Node], Iterable[tuple[Node, float]]],
) -> tuple[list[Node], float]:
    """Run Dijkstra's shortest path algorithm.

    Args:
        source: Starting node
        is_goal: Function that returns True if a node is the goal (takes node and distance)
        adjs: Function that yields (neighbor, distance) tuples for a given node

    Returns:
        Tuple of (path from source to goal, distance from source to goal)

    """
    start = source
    dists: dict[Node, float] = {start: 0}
    parent: dict[Node, Node] = {}
    visited: set[Node] = set()
    counter: int = 0
    q: list[tuple[float, int, Node]] = [(0, counter, start)]
    while q:
        _, _, curr = heappop(q)

        if curr in visited:
            continue
        visited.add(curr)

        if is_goal(curr, dists[curr]):
            return (reconstruct_path(source, curr, parent), dists[curr])

        for adj, adj_dist in adjs(curr):
            dist = dists[curr] + adj_dist
            if dist < dists.get(adj, float("inf")):
                dists[adj] = dist
                counter += 1
                heappush(q, (dists[adj], counter, adj))
                parent[adj] = curr

    return [], float("inf")


def dijkstra_all_paths(
    source: Node,
    adjs: Callable[[Node], Iterable[tuple[Node, float]]],
) -> tuple[dict[Node, float], dict[Node, Node]]:
    """Find shortest paths from source to all reachable nodes.

    Args:
        source: Starting node
        adjs: Function that yields (neighbor, distance) tuples for a given node

    Returns:
        Tuple of (distances dict, parent dict) where:
        - distances: Maps each reachable node to its shortest distance from source
        - parent: Maps each reachable node to its predecessor in the shortest path tree

    """
    start = source
    dists: dict[Node, float] = {start: 0}
    parent: dict[Node, Node] = {}
    visited: set[Node] = set()
    counter: int = 0
    q: list[tuple[float, int, Node]] = [(0, counter, start)]
    while q:
        _, _, curr = heappop(q)

        if curr in visited:
            continue
        visited.add(curr)

        for adj, adj_dist in adjs(curr):
            dist = dists[curr] + adj_dist
            if dist < dists.get(adj, float("inf")):
                dists[adj] = dist
                counter += 1
                heappush(q, (dists[adj], counter, adj))
                parent[adj] = curr

    return dists, parent


# TODO: multi_dijkstra that takes in a list of nodes (waypoints)
# and finds shortest path from a to b that goes through each waypoint


def a_star(
    source: Node,
    is_goal: Callable[[Node, float], bool],
    adjs: Callable[[Node], Iterable[tuple[Node, float]]],
    h: Callable[[Node], float],
) -> tuple[list[Node], float]:
    """Run the A* pathfinding algorithm.

    Args:
        source: Starting node
        is_goal: Function that returns True if a node is the goal (takes node and distance)
        adjs: Function that yields (neighbor, distance) tuples for a given node
        h: Heuristic function estimating distance from a node to the goal.
        Must never overestimate the actual distance.

    Returns:
        Tuple of (path from source to goal, distance from source to goal)

    """
    start = source
    dists: dict[Node, float] = {start: 0}
    parent: dict[Node, Node] = {}
    visited: set[Node] = set()
    counter: int = 0
    q: list[tuple[float, int, Node]] = [(h(start), counter, start)]
    while q:
        _, _, curr = heappop(q)

        if curr in visited:
            continue
        visited.add(curr)

        if is_goal(curr, dists[curr]):
            return (reconstruct_path(source, curr, parent), dists[curr])

        for adj, adj_dist in adjs(curr):
            dist = dists[curr] + adj_dist
            if dist < dists.get(adj, float("inf")):
                dists[adj] = dist
                counter += 1
                heappush(q, (dists[adj] + h(adj), counter, adj))
                parent[adj] = curr

    return ([], float("inf"))


def reconstruct_path(
    source: Node,
    target: Node,
    parent: dict[Node, Node],
) -> list[Node]:
    curr = target
    path = [curr]
    while curr != source:
        curr = parent[curr]
        path.append(curr)
    return path[::-1]
