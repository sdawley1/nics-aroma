#!/usr/bin/env python3

"""
Ring perception (iterative, no recursion).

Finds the smallest set of smallest rings (SSSR) from a molecular adjacency list.
Replaces the legacy recursive cycle search with an explicit, bounded approach:
for each edge, find the shortest cycle through it, then keep a GF(2)-independent
subset whose count equals the graph's cycle rank (E - V + components).

Sam Dawley
06/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from collections import deque
from typing import Dict, FrozenSet, List, Optional, Tuple

# ----- numerics -----
import numpy as np
import numpy.typing as npt

# ----- local modules -----
from aroma.constants import MAX_RING_SIZE, PLANARITY_TOLERANCE_DEG
from aroma.geometry import dihedral

# ============================================================
# GRAPH HELPERS
# ============================================================


def _edges(adj: List[List[int]]) -> List[Tuple[int, int]]:
    """Return the undirected edges (i < j) of an adjacency list."""
    return [(i, j) for i, nbrs in enumerate(adj) for j in nbrs if i < j]


def _count_components(adj: List[List[int]]) -> int:
    """Count connected components via iterative breadth-first search."""
    seen = [False] * len(adj)
    components = 0
    for start in range(len(adj)):
        if seen[start]:
            continue
        components += 1
        queue = deque([start])
        seen[start] = True
        while queue:
            node = queue.popleft()
            for nbr in adj[node]:
                if not seen[nbr]:
                    seen[nbr] = True
                    queue.append(nbr)
    return components


def _shortest_cycle_through_edge(
    adj: List[List[int]], u: int, v: int
) -> Optional[List[int]]:
    """Shortest cycle containing edge (u, v), as an ordered node list.

    Performs a BFS from ``u`` to ``v`` with the direct edge (u, v) excluded; the
    returned path plus that edge forms the cycle. Returns None if no cycle exists.
    """
    parent: Dict[int, int] = {u: u}
    queue = deque([u])
    while queue:
        node = queue.popleft()
        for nbr in adj[node]:
            if node == u and nbr == v:
                continue
            if nbr not in parent:
                parent[nbr] = node
                if nbr == v:
                    queue.clear()
                    break
                queue.append(nbr)
    if v not in parent:
        return None

    path = [v]
    while path[-1] != u:
        path.append(parent[path[-1]])
    return path


# ============================================================
# CYCLE-SPACE SELECTION
# ============================================================


def _edge_index(edges: List[Tuple[int, int]]) -> Dict[FrozenSet[int], int]:
    """Map each undirected edge to a unique bit position."""
    return {frozenset(edge): bit for bit, edge in enumerate(edges)}


def _cycle_mask(ring: List[int], edge_bit: Dict[FrozenSet[int], int]) -> int:
    """GF(2) edge bitmask for a ring given as an ordered, closed node list."""
    mask = 0
    for k in range(len(ring)):
        edge = frozenset((ring[k], ring[(k + 1) % len(ring)]))
        assert edge in edge_bit, f"ring uses unknown edge {tuple(edge)}"
        mask |= 1 << edge_bit[edge]
    return mask


def _is_independent(mask: int, basis: List[int]) -> Optional[int]:
    """Reduce ``mask`` against ``basis``; return its pivot, or None if dependent."""
    for vec in basis:
        mask = min(mask, mask ^ vec)
    return None if mask == 0 else mask


# ============================================================
# PUBLIC API
# ============================================================


def find_rings(
    adj: List[List[int]], max_ring: int = MAX_RING_SIZE
) -> List[List[int]]:
    """Find the smallest set of smallest rings.

    Parameters
    ----------
    adj : list of lists
        Adjacency list (``adj[i]`` are the neighbors of atom i).
    max_ring : int
        Largest ring size to retain.

    Returns
    -------
    rings : list of ordered node lists
        One entry per ring, each a cyclically ordered list of atom indices.
    """
    assert adj, "empty adjacency list"
    assert max_ring >= 3, "max_ring must be at least 3"

    edges = _edges(adj)
    rank = len(edges) - len(adj) + _count_components(adj)
    if rank <= 0:
        return []

    candidates: Dict[FrozenSet[int], List[int]] = {}
    for u, v in edges:
        cycle = _shortest_cycle_through_edge(adj, u, v)
        if cycle is not None and 3 <= len(cycle) <= max_ring:
            candidates.setdefault(frozenset(cycle), cycle)

    edge_bit = _edge_index(edges)
    basis: List[int] = []
    rings: List[List[int]] = []
    for ring in sorted(candidates.values(), key=len):
        pivot = _is_independent(_cycle_mask(ring, edge_bit), basis)
        if pivot is not None:
            basis.append(pivot)
            rings.append(ring)
        if len(rings) == rank:
            break
    assert len(rings) == rank, f"found {len(rings)} rings, expected rank {rank}"
    return rings


def order_ring(adj: List[List[int]], ring: List[int]) -> List[int]:
    """Order an unordered ring's atoms by walking its connectivity.

    Parameters
    ----------
    adj : list of lists
        Adjacency list for the whole molecule.
    ring : list of int
        Atom indices forming a ring, in any order.

    Returns
    -------
    ordered : list of int
        The same atoms ordered so consecutive entries (and the last-to-first
        wrap) are bonded.
    """
    assert len(ring) >= 3, "a ring needs at least three atoms"
    members = set(ring)
    ordered = [ring[0]]
    while len(ordered) < len(ring):
        current = ordered[-1]
        nxt = next(
            (n for n in adj[current] if n in members and n not in ordered), None
        )
        assert nxt is not None, f"ring is not a closed cycle near atom {current}"
        ordered.append(nxt)
    assert ordered[0] in adj[ordered[-1]], "ring does not close"
    return ordered


def is_planar(
    coords: npt.NDArray[np.float64],
    ring: List[int],
    tol_deg: float = PLANARITY_TOLERANCE_DEG,
) -> bool:
    """Test ring planarity via consecutive dihedral angles.

    Parameters
    ----------
    coords : (N, 3) float array
        Coordinates of all atoms.
    ring : list of int
        Ring atoms in cyclic order.
    tol_deg : float
        Maximum tolerated deviation (degrees) of each dihedral from 0 or 180.

    Returns
    -------
    planar : bool
        True if every four-atom dihedral around the ring is flat.
    """
    assert len(ring) >= 3, "a ring needs at least three atoms"
    assert 0.0 < tol_deg < 90.0, "planarity tolerance must be in (0, 90) degrees"
    if len(ring) < 4:
        return True
    n = len(ring)
    for i in range(n):
        quad = [ring[(i + k) % n] for k in range(4)]
        angle = abs(dihedral(*(coords[a] for a in quad)))
        if angle > tol_deg and abs(angle - 180.0) > tol_deg:
            return False
    return True
