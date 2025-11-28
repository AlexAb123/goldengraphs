from typing import TypeVar

from .graph import DirectedGraph
from .union_find import UnionFind

Node = TypeVar("Node")


def kruskal(
    graph: DirectedGraph[Node],
) -> DirectedGraph[Node]:
    uf: UnionFind[Node] = UnionFind()
    mst: DirectedGraph[Node] = DirectedGraph()
    for src, dst, weight in sorted(graph.get_edges(), key=lambda e: e[2]):
        if uf.union(src, dst):
            mst.add_edge(src, dst, weight)
    return mst
