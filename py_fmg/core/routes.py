"""
Routes generation module (land and sea), parity-focused baseline.

Implements:
- Land routes between settlements using a costed shortest-path over cells
  (distance + biome movement cost + mild uphill penalty), then MST + local spurs.
- Sea routes between ports using a water-only shortest-path.

Outputs are polylines as sequences of coordinates (cell centers) suitable for
GeoJSON export.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import heapq
import math

import numpy as np

from .settlements import Settlement
from .biomes import BiomeClassifier


@dataclass
class RouteOptions:
    # Land routing
    land_k_neighbors: int = 5
    land_max_neighbor_k: int = 12
    uphill_penalty_per_20m: float = 0.25
    # River crossing penalty (set <=0 to disable shapely checks for speed)
    river_cross_penalty: float = 0.0
    spur_fraction: float = 0.15  # fraction of nodes to add as local spurs

    # Sea routing
    sea_k_neighbors: int = 4


@dataclass
class Route:
    id: int
    kind: str  # "land" or "sea"
    start_settlement: int
    end_settlement: int
    distance: float
    coords: List[Tuple[float, float]]
    cls: str = "primary"  # class: primary/secondary
    cells: Optional[List[int]] = None  # path as cell indices when available


class RoutesGenerator:
    def __init__(
        self,
        graph,
        settlements: Dict[int, Settlement],
        biome_classifier: BiomeClassifier,
        options: Optional[RouteOptions] = None,
        rivers: Optional[Dict[int, any]] = None,
    ) -> None:
        self.graph = graph
        self.settlements = settlements
        self.biome_classifier = biome_classifier
        self.options = options or RouteOptions()

        # Movement cost per cell from biomes (fallback if missing)
        self.cell_biomes = getattr(graph, "biomes", None)
        if self.cell_biomes and hasattr(self.cell_biomes, "cell_biomes"):
            self.cell_biomes = self.cell_biomes.cell_biomes
        else:
            self.cell_biomes = np.zeros(len(graph.points), dtype=np.uint8)
        self.movement_cost = self._build_movement_cost()

        # Build a prepared MultiLineString of rivers for crossing detection (optional)
        self._river_lines = None
        self._river_prep = None
        if rivers and self.options.river_cross_penalty > 0:
            try:
                from shapely.geometry import LineString, MultiLineString
                from shapely.prepared import prep
                lines = []
                for r in rivers.values():
                    cells = getattr(r, "cells", [])
                    coords: List[Tuple[float, float]] = []
                    for c in cells:
                        if 0 <= c < len(self.graph.points):
                            p = self.graph.points[c]
                            coords.append((float(p[0]), float(p[1])))
                    if len(coords) >= 2:
                        lines.append(LineString(coords))
                if lines:
                    self._river_lines = MultiLineString(lines)
                    self._river_prep = prep(self._river_lines)
            except Exception:
                self._river_lines = None
                self._river_prep = None

    def _build_movement_cost(self) -> np.ndarray:
        n = len(self.graph.points)
        costs = np.zeros(n, dtype=np.float32)
        md = self.biome_classifier.biome_data.movement_cost
        for i in range(n):
            bid = int(self.cell_biomes[i]) if i < len(self.cell_biomes) else 0
            c = md[bid] if 0 <= bid < len(md) else 50
            costs[i] = float(c)
        return costs

    def _edge_cost(self, a: int, b: int) -> float:
        # Base distance between centers
        pa = self.graph.points[a]
        pb = self.graph.points[b]
        dx = float(pb[0] - pa[0])
        dy = float(pb[1] - pa[1])
        base = math.hypot(dx, dy)
        # Movement penalty based on biome movement cost (scaled)
        m = (self.movement_cost[a] + self.movement_cost[b]) * 0.5 / 100.0
        # Uphill penalty on ascent only
        ha = float(self.graph.heights[a])
        hb = float(self.graph.heights[b])
        uphill = max(0.0, hb - ha) / 20.0 * self.options.uphill_penalty_per_20m

        penalty = 0.0
        # River crossing penalty: if enabled and shared edge intersects any river line
        if (
            self.options.river_cross_penalty > 0
            and self._river_prep is not None
            and self.graph.heights[a] >= 20
            and self.graph.heights[b] >= 20
        ):
            try:
                from shapely.geometry import LineString
                # Try shared edge between a and b via cell vertex overlap
                try:
                    va = set(self.graph.cell_vertices[a])
                    vb = set(self.graph.cell_vertices[b])
                    vs = list(va.intersection(vb))
                    if len(vs) >= 2:
                        v1, v2 = vs[0], vs[1]
                        p1 = self.graph.vertex_coordinates[v1]
                        p2 = self.graph.vertex_coordinates[v2]
                        edge = LineString([(float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))])
                        if self._river_prep.intersects(edge):
                            penalty += float(self.options.river_cross_penalty)
                    else:
                        # Fallback: center-to-center segment intersection
                        seg = LineString([(float(pa[0]), float(pa[1])), (float(pb[0]), float(pb[1]))])
                        if self._river_prep.intersects(seg):
                            penalty += float(self.options.river_cross_penalty)
                except Exception:
                    pass
            except Exception:
                pass

        return base * (1.0 + m + uphill) + penalty

    def _edge_cost_sea(self, a: int, b: int) -> float:
        # Water-only: prefer shorter steps
        pa = self.graph.points[a]
        pb = self.graph.points[b]
        return math.hypot(float(pb[0] - pa[0]), float(pb[1] - pa[1]))

    def _dijkstra_path(self, start: int, goal: int, water_only: bool = False) -> Optional[List[int]]:
        nbs = self.graph.cell_neighbors
        heights = self.graph.heights
        pq: List[Tuple[float, int]] = [(0.0, start)]
        dist: Dict[int, float] = {start: 0.0}
        prev: Dict[int, int] = {}
        sea = water_only
        while pq:
            d, u = heapq.heappop(pq)
            if u == goal:
                # reconstruct
                path = [u]
                while u in prev:
                    u = prev[u]
                    path.append(u)
                path.reverse()
                return path
            if d > dist.get(u, float("inf")):
                continue
            for v in nbs[u]:
                if sea:
                    # skip land
                    if heights[v] >= 20:
                        continue
                    w = self._edge_cost_sea(u, v)
                else:
                    w = self._edge_cost(u, v)
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        return None

    def _path_coords(self, cells: List[int]) -> List[Tuple[float, float]]:
        pts: List[Tuple[float, float]] = []
        for c in cells:
            if 0 <= c < len(self.graph.points):
                p = self.graph.points[c]
                pts.append((float(p[0]), float(p[1])))
        return pts

    def _mst_from_candidates(self, nodes: List[int], candidates: List[Tuple[float, int, int, List[int]]]) -> List[Tuple[int, int, List[int]]]:
        # Kruskal MST on precomputed candidate edges: (weight, u, v, path)
        parent = {x: x for x in nodes}
        rank = {x: 0 for x in nodes}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            if rank[rx] < rank[ry]:
                parent[rx] = ry
            elif rank[rx] > rank[ry]:
                parent[ry] = rx
            else:
                parent[ry] = rx
                rank[rx] += 1
            return True

        edges_sorted = sorted(candidates, key=lambda t: t[0])
        mst: List[Tuple[int, int, List[int]]] = []
        for w, u, v, path in edges_sorted:
            if union(u, v):
                mst.append((u, v, path))
        return mst

    def build_land_routes(self) -> List[Route]:
        # Gather settlement indices and cells
        s_ids = sorted(self.settlements.keys())
        if not s_ids:
            return []
        # Coordinates for neighbor search
        coords = np.array([[self.settlements[i].x, self.settlements[i].y] for i in s_ids], dtype=float)
        # Simple kNN using brute force distances to avoid adding sklearn for routes
        def knn(idx: int, k: int) -> List[int]:
            p = coords[idx]
            d2 = np.sum((coords - p) ** 2, axis=1)
            order = np.argsort(d2)
            res = [int(o) for o in order if o != idx][:k]
            return res

        # Candidate edges between near neighbors with shortest paths
        k = self.options.land_k_neighbors
        candidates: List[Tuple[float, int, int, List[int]]] = []
        def cell_of_idx(idx: int) -> int:
            sid = s_ids[idx]
            return int(self.settlements[sid].cell_id)
        while True:
            candidates.clear()
            for i in range(len(s_ids)):
                src_cell = cell_of_idx(i)
                for j in knn(i, k):
                    if j < i:
                        continue
                    dst_cell = cell_of_idx(j)
                    path = self._dijkstra_path(src_cell, dst_cell, water_only=False)
                    if not path:
                        continue
                    # Total path cost as sum of geometric distances
                    coords_path = self._path_coords(path)
                    dist = 0.0
                    for a in range(len(coords_path) - 1):
                        x1, y1 = coords_path[a]
                        x2, y2 = coords_path[a + 1]
                        dist += math.hypot(x2 - x1, y2 - y1)
                    candidates.append((dist, i, j, path))
            # Check connectivity via MST
            mst = self._mst_from_candidates(list(range(len(s_ids))), candidates)
            comp = self._components_from_edges(len(s_ids), [(u, v) for u, v, _ in mst])
            if len(comp) == 1 or k >= self.options.land_max_neighbor_k:
                break
            k = min(self.options.land_max_neighbor_k, k + 2)

        # Build routes from MST
        routes: List[Route] = []
        rid = 1
        for u, v, path in mst:
            start_id = s_ids[u]
            end_id = s_ids[v]
            coords_path = self._path_coords(path)
            length = 0.0
            for a in range(len(coords_path) - 1):
                x1, y1 = coords_path[a]
                x2, y2 = coords_path[a + 1]
                length += math.hypot(x2 - x1, y2 - y1)
            # Classify by importance (capitals / large cities on MST)
            sA = self.settlements[start_id]
            sB = self.settlements[end_id]
            if getattr(sA, 'is_capital', False) or getattr(sB, 'is_capital', False) or (getattr(sA, 'population', 0) + getattr(sB, 'population', 0) > 15000):
                cls = "highway"
            else:
                cls = "road"
            routes.append(Route(id=rid, kind="land", start_settlement=start_id, end_settlement=end_id, distance=length, coords=coords_path, cls=cls, cells=path))
            rid += 1

        # Add local spurs: connect each node to its nearest non-MST neighbor if short
        if candidates:
            used_pairs = {(min(s_ids[u], s_ids[v]), max(s_ids[u], s_ids[v])) for u, v, _ in mst}
            added = 0
            target_spurs = max(1, int(len(s_ids) * self.options.spur_fraction))
            for w, u, v, path in sorted(candidates, key=lambda t: t[0]):
                if added >= target_spurs:
                    break
                pair = (min(s_ids[u], s_ids[v]), max(s_ids[u], s_ids[v]))
                if pair in used_pairs:
                    continue
                coords_path = self._path_coords(path)
                routes.append(Route(id=rid, kind="land", start_settlement=pair[0], end_settlement=pair[1], distance=w, coords=coords_path, cls="trail", cells=path))
                rid += 1
                added += 1

        return routes

    def _components_from_edges(self, n: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
        adj: List[List[int]] = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        seen = [False] * n
        comps: List[List[int]] = []
        for i in range(n):
            if seen[i]:
                continue
            stack = [i]
            seen[i] = True
            comp = []
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in adj[u]:
                    if not seen[v]:
                        seen[v] = True
                        stack.append(v)
            comps.append(comp)
        return comps

    def build_sea_routes(self) -> List[Route]:
        # Find ports
        ports = [s for s in self.settlements.values() if getattr(s, "port_id", 0) > 0]
        if len(ports) < 2:
            return []
        # Map each port to a water cell to start (nearest neighboring water cell)
        water_heights = self.graph.heights < 20
        def nearest_water_cell(cell_id: int) -> Optional[int]:
            if water_heights[cell_id]:
                return int(cell_id)
            from collections import deque
            dq = deque([cell_id])
            seen = {cell_id}
            while dq:
                u = dq.popleft()
                for v in self.graph.cell_neighbors[u]:
                    if v in seen:
                        continue
                    if water_heights[v]:
                        return int(v)
                    seen.add(v)
                    dq.append(v)
            return None

        port_cells: Dict[int, int] = {}
        for s in ports:
            wc = nearest_water_cell(int(s.cell_id))
            if wc is not None:
                port_cells[s.id] = wc

        # Connect ports to their K nearest other ports by sea path
        if len(port_cells) < 2:
            return []
        p_ids = sorted(port_cells.keys())
        coords = np.array([[self.settlements[i].x, self.settlements[i].y] for i in p_ids], dtype=float)
        def knn(idx: int, k: int) -> List[int]:
            p = coords[idx]
            d2 = np.sum((coords - p) ** 2, axis=1)
            order = np.argsort(d2)
            return [int(o) for o in order if o != idx][:k]

        routes: List[Route] = []
        rid = 100000  # separate id range for sea
        k = self.options.sea_k_neighbors
        for i in range(len(p_ids)):
            a_sid = p_ids[i]
            a_cell = port_cells[a_sid]
            for j in knn(i, k):
                b_sid = p_ids[j]
                b_cell = port_cells[b_sid]
                path = self._dijkstra_path(a_cell, b_cell, water_only=True)
                if not path:
                    continue
                coords_path = self._path_coords(path)
                # Snap endpoints to port coordinates
                axy = (float(self.settlements[a_sid].x), float(self.settlements[a_sid].y))
                bxy = (float(self.settlements[b_sid].x), float(self.settlements[b_sid].y))
                if coords_path and coords_path[0] != axy:
                    coords_path = [axy] + coords_path
                if coords_path and coords_path[-1] != bxy:
                    coords_path = coords_path + [bxy]
                length = 0.0
                for t in range(len(coords_path) - 1):
                    x1, y1 = coords_path[t]
                    x2, y2 = coords_path[t + 1]
                    length += math.hypot(x2 - x1, y2 - y1)
                # Classify by port significance (population & harbor_score)
                sA = self.settlements[a_sid]
                sB = self.settlements[b_sid]
                score = (getattr(sA, 'population', 1) * (getattr(sA, 'port_id', 0) > 0 and 1 or 0)) + (getattr(sB, 'population', 1) * (getattr(sB, 'port_id', 0) > 0 and 1 or 0))
                cls = "sea-lane" if score > 10000 else "coastal"
                routes.append(Route(id=rid, kind="sea", start_settlement=a_sid, end_settlement=b_sid, distance=length, coords=coords_path, cls=cls, cells=path))
                rid += 1

        return routes
