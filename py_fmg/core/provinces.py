"""
Provinces generator — port of FMG Provinces.generate() adapted for headless py‑fmg.

Creates administrative subdivisions inside each state, seeding from major
settlements and expanding regions with a simple terrain-aware cost. Outputs:

- provinces: dict[int, Province]
- cell_province: np.ndarray mapping cell_id -> province_id (0 = none)
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Province:
    id: int
    state_id: int
    center_cell: int
    burg_id: int = 0
    name: str = ""
    full_name: str = ""
    color: Optional[str] = None
    cells: List[int] = field(default_factory=list)


@dataclass
class ProvinceOptions:
    provinces_ratio: int = 50  # 0..100: % of major centers promoted (2..20 caps)
    max_growth_cost: Optional[float] = None  # if None, derive ~ FMG: 20 * sqrt(ratio) (100 -> 1000)


class ProvincesGenerator:
    def __init__(self, graph, states: Dict[int, any], settlements: Dict[int, any], options: Optional[ProvinceOptions] = None):
        self.graph = graph
        self.states = states
        self.settlements = settlements
        self.options = options or ProvinceOptions()

        self.cell_state = getattr(graph, "cell_state", None)
        if self.cell_state is None:
            # Derive from states territory if not provided: fallback to 0s
            self.cell_state = np.zeros(len(graph.points), dtype=np.uint16)
            for st in states.values():
                for c in getattr(st, "territory_cells", []):
                    if 0 <= c < len(self.cell_state):
                        self.cell_state[c] = int(st.id)

        self.cell_province = np.zeros(len(graph.points), dtype=np.uint16)
        self.provinces: Dict[int, Province] = {}
        self.next_prov_id = 1
        # derive growth if not provided
        if self.options.max_growth_cost is None:
            r = max(1, min(100, int(self.options.provinces_ratio)))
            self.options.max_growth_cost = 1000.0 if r == 100 else 20.0 * (r ** 0.5)

    def _is_land(self, cid: int) -> bool:
        return int(self.graph.heights[cid]) >= 20

    def _major_burgs_for_state(self, state_id: int) -> List[any]:
        burgs: List[any] = []
        for b in self.settlements.values():
            if int(getattr(b, "state_id", 0)) != int(state_id):
                continue
            if getattr(b, "is_capital", False) or float(getattr(b, "population", 0.0)) >= 1.0:
                burgs.append(b)
        if not burgs:
            # fallback: pick most populous inside state
            burgs = [b for b in self.settlements.values() if int(getattr(b, "state_id", 0)) == int(state_id)]
        # Sort capital first, then population
        burgs.sort(key=lambda b: (not getattr(b, "is_capital", False), -float(getattr(b, "population", 0.0))))
        return burgs

    def _seed_state(self, st) -> List[Province]:
        centers = self._major_burgs_for_state(st.id)
        if len(centers) < 2:
            # Fallback: use most populous burgs within the state territory
            burgs = [b for b in self.settlements.values() if int(getattr(b, "state_id", 0)) == int(st.id)]
            burgs.sort(key=lambda b: -float(getattr(b, "population", 0.0)))
            centers = burgs[: max(0, min(10, len(burgs)))]
            # Ensure at least 2 centers per state by adding a far cell as synthetic center when needed
            if len(centers) == 1:
                first_cell = centers[0].cell_id
                # choose farthest land cell within state
                far_cell = first_cell
                fx, fy = self.graph.points[first_cell]
                max_d2 = -1.0
                for i in range(len(self.graph.points)):
                    if not self._is_land(i) or self.cell_state[i] != st.id:
                        continue
                    x, y = self.graph.points[i]
                    d2 = (x - fx) ** 2 + (y - fy) ** 2
                    if d2 > max_d2:
                        max_d2 = d2
                        far_cell = i
                # create a synthetic pseudo-burg object with required attrs
                class P:
                    pass
                p = P()
                p.cell_id = int(far_cell)
                p.id = 0
                p.name = getattr(st, "name", f"State {st.id}")
                p.population = 0.5
                p.is_capital = False
                centers.append(p)
            if len(centers) < 2:
                return []
        target = max(2, min(20, int(np.ceil(len(centers) * self.options.provinces_ratio / 100.0))))
        centers = centers[:target]

        items: List[Province] = []
        for b in centers:
            pid = self.next_prov_id
            self.next_prov_id += 1
            name = getattr(b, "name", "") or f"Province {pid}"
            full_name = name
            prov = Province(id=pid, state_id=st.id, center_cell=b.cell_id, burg_id=b.id, name=name, full_name=full_name)
            self.provinces[pid] = prov
            items.append(prov)
        return items

    def _terrain_cost(self, cid: int, state_id: int) -> float:
        h = int(self.graph.heights[cid])
        # basic cost tiers like FMG
        if h >= 70:
            base = 100
        elif h >= 50:
            base = 30
        elif h >= 20:
            base = 10
        else:
            base = 100  # water prohibitive
        # discourage leaving the state
        if self.cell_state[cid] != state_id:
            base += 1000
        return float(base)

    def _expand(self, seeds: List[Province]) -> None:
        max_cost = float(self.options.max_growth_cost)
        q = []
        cost = {}
        # init queue
        for p in seeds:
            c = int(p.center_cell)
            self.cell_province[c] = p.id
            heapq.heappush(q, (0.0, c, p.id, p.state_id))
            cost[c] = 0.0

        while q:
            p_cost, cell, pid, sid = heapq.heappop(q)
            for nb in self.graph.cell_neighbors[cell]:
                if not self._is_land(nb):
                    continue
                if self.cell_state[nb] != sid:
                    continue
                step = self._terrain_cost(nb, sid)
                ncost = p_cost + step
                if ncost > max_cost:
                    continue
                if nb not in cost or ncost < cost[nb]:
                    self.cell_province[nb] = pid
                    cost[nb] = ncost
                    heapq.heappush(q, (ncost, nb, pid, sid))

        # populate per-province cell lists
        for i, pid in enumerate(self.cell_province):
            if pid > 0:
                self.provinces[pid].cells.append(i)

    def _justify(self) -> None:
        # one pass reassignment by neighborhood majority
        prov_ids = self.cell_province
        for i in range(len(prov_ids)):
            if prov_ids[i] == 0 or not self._is_land(i):
                continue
            nbs = [prov_ids[n] for n in self.graph.cell_neighbors[i] if self._is_land(n) and self.cell_state[n] == self.cell_state[i]]
            if not nbs:
                continue
            cur = prov_ids[i]
            candidates = [p for p in nbs if p != cur]
            if len(candidates) < 2:
                continue
            # pick the province that appears most among neighbors
            best = max(set(candidates), key=candidates.count)
            # only switch when best strictly outranks current by 2+ occurrences
            if candidates.count(best) >= nbs.count(cur) + 2:
                prov_ids[i] = best

    def _fill_gaps(self) -> None:
        # Assign any land cells within a state that are still 0 to the nearest province in that state
        from collections import deque

        for sid in {s.id for s in self.states.values()}:
            frontier = deque()
            dist = {}
            # seed from assigned cells
            for i in range(len(self.cell_province)):
                if self.cell_state[i] == sid and self.cell_province[i] > 0:
                    frontier.append(i)
                    dist[i] = 0
            while frontier:
                c = frontier.popleft()
                for n in self.graph.cell_neighbors[c]:
                    if self.cell_state[n] != sid or not self._is_land(n):
                        continue
                    if self.cell_province[n] == 0:
                        self.cell_province[n] = self.cell_province[c]
                    if n not in dist:
                        dist[n] = dist[c] + 1
                        frontier.append(n)

        # refresh province cell lists
        for p in self.provinces.values():
            p.cells = []
        for i, pid in enumerate(self.cell_province):
            if pid > 0:
                self.provinces[pid].cells.append(i)

    def generate(self) -> Tuple[Dict[int, Province], np.ndarray]:
        # seed provinces per state
        seeds: List[Province] = []
        per_state_counts: Dict[int, int] = {}
        for st in self.states.values():
            if not getattr(st, "id", 0):
                continue
            s = self._seed_state(st)
            per_state_counts[st.id] = len(s)
            seeds += s
        # expand and clean
        if seeds:
            self._expand(seeds)
            self._justify()
            self._fill_gaps()
        return self.provinces, self.cell_province
