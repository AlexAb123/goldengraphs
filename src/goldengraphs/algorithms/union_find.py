from typing import Generic, TypeVar

T = TypeVar("T")


class UnionFind(Generic[T]):
    """
    Disjoint Set Union data structure.

    Tracks elements partitioned into non-overlapping sets.
    Supports near O(1) amortized union and find operations
    using path compression.
    """

    def __init__(self) -> None:
        """Initialize an empty UnionFind structure."""
        self.parent: dict[T, T] = {}
        self.rank: dict[T, int] = {}

    def find(self, x: T) -> T:
        """
        Find the root representative of x's set.

        Creates a new set if x hasn't been seen.
        """
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            # Path compression
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: T, y: T) -> bool:
        """
        Merge the sets containing x and y.

        Returns True if x and y were in different sets,
        False if already in the same set.
        """
        root1, root2 = self.find(x), self.find(y)
        if root1 == root2:
            return False

        if self.rank[root1] < self.rank[root2]:
            root1, root2 = root2, root1  # Make root1 taller
        self.parent[root2] = root1

        # If trees are equal rank (approximate height), increment rank by one
        if self.rank[root1] == self.rank[root2]:
            self.rank[root1] += 1

        return True
