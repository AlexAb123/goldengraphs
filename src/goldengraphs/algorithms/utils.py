from typing import TypeVar

Node = TypeVar("Node")


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
