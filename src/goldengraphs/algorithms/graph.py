from collections.abc import ItemsView, Iterator, KeysView
from typing import Generic, Optional, TypeVar

Node = TypeVar("Node")


class DirectedGraph(Generic[Node]):
    def __init__(
        self,
        nodes: Optional[set[Node]] = None,
        edges: Optional[set[tuple[Node, Node, float]]] = None,
    ) -> None:
        self.adjs: dict[Node, dict[Node, float]] = {}
        if nodes:
            for node in nodes:
                self.adjs[node] = {}
        if edges:
            for src, dst, weight in edges:
                self.add_edge(src, dst, weight)

    def add_edge(self, src: Node, dst: Node, weight: float) -> None:
        for node in src, dst:
            if node not in self.adjs:
                self.adjs[node] = {}
        self.adjs[src][dst] = weight

    def remove_edge(self, src: Node, dst: Node) -> bool:
        if src not in self.adjs or dst not in self.adjs[src]:
            return False
        self.adjs[src].pop(dst)
        return True

    def add_node(self, node: Node) -> bool:
        if node in self.adjs:
            return False
        self.adjs[node] = {}
        return True

    def remove_node(self, node: Node) -> bool:
        if node not in self.adjs:
            return False
        for other in self.adjs:
            self.adjs[other].pop(node, None)
        self.adjs.pop(node)
        return True

    def get_nodes(self) -> KeysView[Node]:
        return self.adjs.keys()

    def get_adjs(self, node: Node) -> ItemsView[Node, float]:
        return self.adjs[node].items()

    def get_edges(self) -> Iterator[tuple[Node, Node, float]]:
        for src in self.adjs:
            for dst, weight in self.adjs[src].items():
                yield src, dst, weight

    def __len__(self) -> int:
        return len(self.adjs)

    def __contains__(self, node: Node) -> bool:
        return node in self.adjs

    def has_edge(self, src: Node, dst: Node) -> bool:
        return src in self.adjs and dst in self.adjs[src]

    def get_weight(self, src: Node, dst: Node) -> float | None:
        if self.has_edge(src, dst):
            return self.adjs[src][dst]
        return None

    def in_degree(self, node: Node) -> int:
        total = 0
        for src in self.adjs:
            if node in self.adjs[src]:
                total += 1
        return total

    def out_degree(self, node: Node) -> int:
        return len(self.adjs.get(node, {}))

    def num_edges(self) -> int:
        return sum(len(self.adjs[src]) for src in self.adjs)


class Graph(DirectedGraph[Node]):
    def add_edge(self, src: Node, dst: Node, weight: float) -> None:
        super().add_edge(src, dst, weight)
        super().add_edge(dst, src, weight)

    def remove_edge(self, src: Node, dst: Node) -> bool:
        a = super().remove_edge(src, dst)
        b = super().remove_edge(dst, src)
        return a or b

    def degree(self, node: Node) -> int:
        return self.out_degree(node)

    def num_edges(self) -> int:
        return super().num_edges() // 2
