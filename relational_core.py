from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import math
import random
from typing import Dict, List, Tuple, Optional

import numpy as np

# Four quarter-turn relations on U(1):
# 0 -> 1
# 1 -> i
# 2 -> -1
# 3 -> -i
CANONICAL_COMPLEX = np.array([1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j], dtype=np.complex128)
RELATION_NAMES = ["0° / 1", "+90° / i", "180° / -1", "-90° / -i"]


def compose_relations(relations: List[int]) -> int:
    """Exact Z4/U(1) composition. Equivalent to multiplying complex phases."""
    return int(sum(int(r) for r in relations) % 4)


def complex_compose(relations: List[int]) -> complex:
    z = 1 + 0j
    for r in relations:
        z *= CANONICAL_COMPLEX[int(r)]
    # Round tiny floating point noise for display.
    return complex(round(z.real, 12), round(z.imag, 12))


def loop_is_consistent(relations: List[int]) -> bool:
    """A closed loop is consistent when the net relation is the identity."""
    return compose_relations(relations) == 0


@dataclass
class Edge:
    a: int
    b: int
    relation: int


class RelationalGraph:
    """
    Directed relational graph with inverse edges.

    add_edge(a, b, r) means:
        potential[b] = potential[a] + r  (mod 4)

    Therefore the inverse relation b -> a is -r (mod 4).
    """

    def __init__(self, node_count: int):
        self.node_count = node_count
        self._edges: Dict[Tuple[int, int], int] = {}

    def add_edge(self, a: int, b: int, relation: int) -> None:
        relation = int(relation) % 4
        self._edges[(a, b)] = relation
        self._edges[(b, a)] = (-relation) % 4

    def remove_edge(self, a: int, b: int) -> None:
        self._edges.pop((a, b), None)
        self._edges.pop((b, a), None)

    def relation(self, a: int, b: int) -> Optional[int]:
        return self._edges.get((a, b))

    def neighbors(self, a: int):
        for (u, v), r in self._edges.items():
            if u == a:
                yield v, r

    def infer(self, start: int, goal: int) -> Optional[int]:
        """
        Infer the net relation along any available path.
        If the graph is consistent, every path gives the same answer.
        """
        if start == goal:
            return 0

        q = deque([(start, 0)])
        seen = {start}
        while q:
            node, accumulated = q.popleft()
            for nxt, rel in self.neighbors(node):
                if nxt in seen:
                    continue
                new_acc = (accumulated + rel) % 4
                if nxt == goal:
                    return new_acc
                seen.add(nxt)
                q.append((nxt, new_acc))
        return None

    def check_consistency(self) -> Tuple[bool, Optional[str]]:
        """
        Assign a potential to each node in each connected component and verify
        that every edge agrees with those potentials.
        """
        potentials: Dict[int, int] = {}

        for root in range(self.node_count):
            if root in potentials:
                continue
            potentials[root] = 0
            q = deque([root])

            while q:
                u = q.popleft()
                for v, rel_uv in self.neighbors(u):
                    implied_v = (potentials[u] + rel_uv) % 4
                    if v not in potentials:
                        potentials[v] = implied_v
                        q.append(v)
                    elif potentials[v] != implied_v:
                        return (
                            False,
                            f"Contradiction on edge {u}->{v}: "
                            f"stored relation={rel_uv}, but current graph implies "
                            f"{(potentials[v] - potentials[u]) % 4}.",
                        )
        return True, None

    def unique_edges(self) -> List[Edge]:
        out = []
        for (a, b), r in self._edges.items():
            if a < b:
                out.append(Edge(a, b, r))
        return out


def generate_consistent_graph(
    node_count: int = 16,
    extra_edges: int = 12,
    seed: int = 7,
) -> Tuple[RelationalGraph, List[int]]:
    """
    Generate a graph from hidden node potentials. Every edge is therefore
    mutually consistent by construction.
    """
    rng = random.Random(seed)
    potentials = [rng.randrange(4) for _ in range(node_count)]
    g = RelationalGraph(node_count)

    # Random spanning tree so the graph is connected.
    for node in range(1, node_count):
        parent = rng.randrange(node)
        relation = (potentials[node] - potentials[parent]) % 4
        g.add_edge(parent, node, relation)

    attempts = 0
    while len(g.unique_edges()) < (node_count - 1 + extra_edges) and attempts < 10000:
        a, b = rng.sample(range(node_count), 2)
        if g.relation(a, b) is None:
            relation = (potentials[b] - potentials[a]) % 4
            g.add_edge(a, b, relation)
        attempts += 1

    return g, potentials


def topology_change_demo(seed: int = 7) -> str:
    """
    Demonstrates:
      1) inference on a consistent relational graph,
      2) a topology change by deleting an edge,
      3) a new relation being added,
      4) contradiction detection without retraining.
    """
    rng = random.Random(seed)
    g, hidden = generate_consistent_graph(node_count=16, extra_edges=10, seed=seed)

    lines = []
    lines.append("RELATIONAL TOPOLOGY DEMO")
    lines.append("=" * 72)
    ok, reason = g.check_consistency()
    lines.append(f"Initial graph: {len(g.unique_edges())} undirected relations.")
    lines.append(f"Initial consistency: {ok}")
    lines.append("")

    a, b = 0, 15
    inferred = g.infer(a, b)
    true_rel = (hidden[b] - hidden[a]) % 4
    lines.append(
        f"Infer node {a} -> node {b}: {inferred} ({RELATION_NAMES[inferred]})"
    )
    lines.append(
        f"Hidden generating relation: {true_rel} ({RELATION_NAMES[true_rel]})"
    )
    lines.append(f"Matches: {inferred == true_rel}")
    lines.append("")

    # Remove a non-tree-ish edge if possible.
    edges = g.unique_edges()
    remove_edge = rng.choice(edges)
    before = g.infer(remove_edge.a, remove_edge.b)
    g.remove_edge(remove_edge.a, remove_edge.b)
    after = g.infer(remove_edge.a, remove_edge.b)
    lines.append(
        f"TOPOLOGY CHANGE 1: removed relation "
        f"{remove_edge.a}<->{remove_edge.b}."
    )
    lines.append(f"Relation before deletion: {before}")
    lines.append(
        "Relation after deletion via remaining paths: "
        + ("disconnected" if after is None else str(after))
    )
    ok, reason = g.check_consistency()
    lines.append(f"Graph remains consistent: {ok}")
    lines.append("")

    # Add a correct relation between two nodes not directly connected.
    candidates = [
        (i, j)
        for i in range(g.node_count)
        for j in range(i + 1, g.node_count)
        if g.relation(i, j) is None
    ]
    x, y = rng.choice(candidates)
    correct = (hidden[y] - hidden[x]) % 4
    g.add_edge(x, y, correct)
    lines.append(
        f"TOPOLOGY CHANGE 2: added new relation {x}->{y} = {correct} "
        f"({RELATION_NAMES[correct]})."
    )
    ok, reason = g.check_consistency()
    lines.append(f"Graph after correct relation: consistent={ok}")
    lines.append("")

    # Add an intentionally contradictory relation.
    candidates = [
        (i, j)
        for i in range(g.node_count)
        for j in range(i + 1, g.node_count)
        if g.relation(i, j) is None
    ]
    c, d = rng.choice(candidates)
    correct_cd = (hidden[d] - hidden[c]) % 4
    wrong_cd = (correct_cd + rng.choice([1, 2, 3])) % 4
    g.add_edge(c, d, wrong_cd)
    lines.append(
        f"TOPOLOGY CHANGE 3: injected contradictory relation "
        f"{c}->{d} = {wrong_cd}; geometry requires {correct_cd}."
    )
    ok, reason = g.check_consistency()
    lines.append(f"Graph after contradiction: consistent={ok}")
    if reason:
        lines.append("Detected reason:")
        lines.append("  " + reason)

    lines.append("")
    lines.append(
        "No model was retrained during any topology change. "
        "The answer changed because the relational structure changed."
    )
    return "\n".join(lines)
