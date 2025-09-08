"""
Markers generator — parity-aligned with FMG markers-generator.js.

Implements a configurable set of marker types with:
- type, icon, pixel offsets (dx, dy, px)
- min/each/multiplier sampling policy
- list() candidate selector per type
- add() marker legend/name generator per type

Outputs: in-memory marker dicts with fields
  {i, type, icon, x, y, cell, name, legend, dx, dy, px}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import math
import random

import numpy as np


@dataclass
class Marker:
    i: int
    type: str
    icon: str
    x: float
    y: float
    cell: int
    name: str
    legend: str
    dx: Optional[int] = None
    dy: Optional[int] = None
    px: Optional[int] = None


class MarkersGenerator:
    def __init__(
        self,
        graph,
        settlements: Optional[Dict[int, Any]] = None,
        rivers: Optional[Dict[int, Any]] = None,
        routes: Optional[List[Any]] = None,
        seed: Optional[str] = None,
    ) -> None:
        self.g = graph
        self.settlements = settlements or {}
        self.rivers = rivers or {}
        self.routes = routes or []
        if seed is not None:
            random.seed(seed)

        # Precompute arrays for quick checks
        n = len(graph.points)
        self.cell_has_settlement = np.zeros(n, dtype=bool)
        for s in self.settlements.values():
            cid = int(getattr(s, "cell_id", -1))
            if 0 <= cid < n:
                self.cell_has_settlement[cid] = True
        self.river_cell = np.zeros(n, dtype=bool)
        if hasattr(graph, "river_ids"):
            rr = getattr(graph, "river_ids")
            if rr is not None and len(rr) == n:
                self.river_cell = (np.array(rr) > 0)
        self.harbor = getattr(graph, "harbor", np.zeros(n, dtype=np.uint8))

        # Routes crossroad degree by cell index (approximate)
        self.route_degree = np.zeros(n, dtype=int)
        for r in (routes or []):
            cells: List[int] = getattr(r, "cells", [])
            for c in cells:
                if 0 <= c < n:
                    self.route_degree[c] += 1

        self.occupied = np.zeros(n, dtype=bool)

    # ---- Public API ----
    def generate(self) -> List[Marker]:
        cfg = self._default_config()
        markers: List[Marker] = []
        for entry in cfg:
            if entry.get("multiplier", 1) == 0:
                continue
            candidates = list(entry["list"](self))
            qty = self._get_quantity(candidates, entry["min"], entry["each"], entry.get("multiplier", 1))
            while qty and candidates:
                # pick random candidate cell
                idx = random.randrange(len(candidates))
                cell = int(candidates.pop(idx))
                m = self._add_marker(markers, entry, cell)
                if m is not None:
                    entry["add"](self, m)
                    qty -= 1
        # reset occupancy for next run
        self.occupied[:] = False
        return markers

    # ---- Sampling and add helpers ----
    def _get_quantity(self, array: List[int], min_candidates: int, each: int, multiplier: float) -> int:
        if not array or len(array) < min_candidates / max(1.0, multiplier):
            return 0
        req = math.ceil((len(array) / each) * multiplier)
        return min(len(array), req)

    def _marker_xy(self, cell: int) -> Tuple[float, float]:
        # Prefer settlement location if present
        for s in self.settlements.values():
            if int(getattr(s, "cell_id", -1)) == cell:
                return (float(getattr(s, "x", self.g.points[cell][0])), float(getattr(s, "y", self.g.points[cell][1])))
        p = self.g.points[cell]
        return (float(p[0]), float(p[1]))

    def _add_marker(self, markers: List[Marker], base: Dict[str, Any], cell: int) -> Optional[Marker]:
        if cell < 0 or cell >= len(self.g.points):
            return None
        i = (markers[-1].i + 1) if markers else 0
        x, y = self._marker_xy(cell)
        m = Marker(
            i=i,
            type=str(base["type"]),
            icon=str(base["icon"]),
            x=x,
            y=y,
            cell=int(cell),
            name="",
            legend="",
            dx=base.get("dx"),
            dy=base.get("dy"),
            px=base.get("px"),
        )
        markers.append(m)
        self.occupied[cell] = True
        return m

    # ---- Candidate selectors and adders ----
    # Each list_* returns iterable of cell indices; add_* sets name/legend
    def list_volcanoes(self) -> Iterable[int]:
        return [i for i in range(len(self.g.points)) if not self.occupied[i] and self.g.heights[i] >= 70]

    def add_volcano(self, m: Marker) -> None:
        m.name = f"Volcano"
        m.legend = "Dormant volcano."

    def list_hot_springs(self) -> Iterable[int]:
        return [i for i in range(len(self.g.points)) if not self.occupied[i] and self.g.heights[i] > 50]

    def add_hot_spring(self, m: Marker) -> None:
        m.name = "Hot Springs"
        m.legend = "Geothermal springs with naturally heated water."

    def list_water_sources(self) -> Iterable[int]:
        return [i for i in range(len(self.g.points)) if not self.occupied[i] and self.g.heights[i] > 30 and self.river_cell[i]]

    def add_water_source(self, m: Marker) -> None:
        m.name = "Legendary Water Source"
        m.legend = "A spring believed to possess mystical properties."

    def list_mines(self) -> Iterable[int]:
        return [i for i in range(len(self.g.points)) if not self.occupied[i] and self.g.heights[i] > 47 and self.cell_has_settlement[i]]

    def add_mine(self, m: Marker) -> None:
        m.name = "Mining Town"
        m.legend = "A town near a productive mine."

    def list_bridges(self) -> Iterable[int]:
        fl = getattr(self.g, "flux", None)
        if fl is None or len(fl) != len(self.g.points):
            return []
        mean_flux = float(np.mean([v for v in fl if v and v > 0])) if np.any(fl) else 0.0
        return [
            i
            for i in range(len(self.g.points))
            if not self.occupied[i]
            and self.cell_has_settlement[i]
            and self.river_cell[i]
            and fl[i] > mean_flux
        ]

    def add_bridge(self, m: Marker) -> None:
        m.name = "Bridge"
        m.legend = "An important crossing over a major river."

    def list_inns(self) -> Iterable[int]:
        # crossroads: route degree >= 3 and nearby population
        pop = getattr(self.g, "cell_population", None)
        if pop is None:
            return []
        return [i for i in range(len(self.g.points)) if not self.occupied[i] and pop[i] > 5 and self.route_degree[i] >= 3]

    def add_inn(self, m: Marker) -> None:
        m.name = "Roadside Inn"
        m.legend = "A famous roadside inn serving travelers."

    def list_lighthouses(self) -> Iterable[int]:
        # harbor > 6 and adjacent to water
        res = []
        for i in range(len(self.g.points)):
            if self.occupied[i]:
                continue
            if int(self.harbor[i]) <= 6:
                continue
            if any(self.g.heights[n] < 20 for n in self.g.cell_neighbors[i]):
                res.append(i)
        return res

    def add_lighthouse(self, m: Marker) -> None:
        m.name = "Lighthouse"
        m.legend = "A lighthouse serving as a beacon for ships."

    def list_waterfalls(self) -> Iterable[int]:
        res = []
        for i in range(len(self.g.points)):
            if self.occupied[i]:
                continue
            if not self.river_cell[i] or self.g.heights[i] < 50:
                continue
            if any(self.g.heights[c] < 40 and self.river_cell[c] for c in self.g.cell_neighbors[i]):
                res.append(i)
        return res

    def add_waterfall(self, m: Marker) -> None:
        m.name = "Waterfall"
        m.legend = "An impressive waterfall cascades here."

    def list_battlefields(self) -> Iterable[int]:
        pop = getattr(self.g, "cell_population", None)
        state = getattr(self.g, "cell_state", None)
        if pop is None or state is None:
            return []
        return [i for i in range(len(self.g.points)) if not self.occupied[i] and state[i] > 0 and pop[i] > 2 and 25 < self.g.heights[i] < 50]

    def add_battlefield(self, m: Marker) -> None:
        m.name = "Battlefield"
        m.legend = "A historical battlefield."

    def list_dungeons(self) -> Iterable[int]:
        pop = getattr(self.g, "cell_population", None)
        if pop is None:
            return []
        return [i for i in range(len(self.g.points)) if not self.occupied[i] and 0 < pop[i] < 3]

    def add_dungeon(self, m: Marker) -> None:
        m.name = "Dungeon"
        m.legend = "An undiscovered dungeon."

    def list_lake_monsters(self) -> Iterable[int]:
        res = []
        feats = getattr(self.g, "features", None)
        fids = getattr(self.g, "feature_ids", None)
        if not feats or fids is None:
            return res
        for i in range(1, len(feats)):
            f = feats[i]
            if not f or getattr(f, "type", None) != "lake":
                continue
            c = int(getattr(f, "first_cell", getattr(f, "firstCell", -1)))
            if c >= 0 and not self.occupied[c]:
                res.append(c)
        return res

    def add_lake_monster(self, m: Marker) -> None:
        m.name = "Lake Monster"
        m.legend = "Rumors say a relic monster inhabits this lake."

    def list_sea_monsters(self) -> Iterable[int]:
        feats = getattr(self.g, "features", None)
        fids = getattr(self.g, "feature_ids", None)
        if fids is None or not feats:
            return []
        return [i for i in range(len(self.g.points)) if not self.occupied[i] and self.g.heights[i] < 20 and feats[int(fids[i])].type == "ocean"]

    def add_sea_monster(self, m: Marker) -> None:
        m.name = "Sea Monster"
        m.legend = "Old sailors tell stories of a gigantic sea monster."

    def list_sacred_mountains(self) -> Iterable[int]:
        res = []
        for i in range(len(self.g.points)):
            if self.occupied[i]:
                continue
            if self.g.heights[i] < 70:
                continue
            if any(self.g.heights[c] >= 60 for c in self.g.cell_neighbors[i]):
                continue
            res.append(i)
        return res

    def add_sacred_mountain(self, m: Marker) -> None:
        m.name = "Sacred Mountain"
        m.legend = "A sacred mountain revered by locals."

    def list_sacred_forests(self) -> Iterable[int]:
        biomes = getattr(self.g, "biomes", None)
        if not biomes or not hasattr(biomes, "cell_biomes"):
            return []
        b = biomes.cell_biomes
        return [i for i in range(len(self.g.points)) if not self.occupied[i] and (b[i] in (6, 8))]

    def add_sacred_forest(self, m: Marker) -> None:
        m.name = "Sacred Forest"
        m.legend = "A forest sacred to local faith."

    def list_sacred_pineries(self) -> Iterable[int]:
        biomes = getattr(self.g, "biomes", None)
        if not biomes or not hasattr(biomes, "cell_biomes"):
            return []
        b = biomes.cell_biomes
        return [i for i in range(len(self.g.points)) if not self.occupied[i] and (b[i] == 9)]

    def add_sacred_pinery(self, m: Marker) -> None:
        m.name = "Sacred Pinery"
        m.legend = "A pinery sacred to local faith."

    # ---- Default config mirroring FMG set (subset; extend as needed) ----
    def _default_config(self) -> List[Dict[str, Any]]:
        return [
            {"type": "volcanoes", "icon": "🌋", "dx": 52, "px": 13, "min": 10, "each": 500, "multiplier": 1, "list": MarkersGenerator.list_volcanoes, "add": MarkersGenerator.add_volcano},
            {"type": "hot-springs", "icon": "♨️", "dy": 52, "min": 30, "each": 1200, "multiplier": 1, "list": MarkersGenerator.list_hot_springs, "add": MarkersGenerator.add_hot_spring},
            {"type": "water-sources", "icon": "💧", "min": 1, "each": 1000, "multiplier": 1, "list": MarkersGenerator.list_water_sources, "add": MarkersGenerator.add_water_source},
            {"type": "mines", "icon": "⛏️", "dx": 48, "px": 13, "min": 1, "each": 15, "multiplier": 1, "list": MarkersGenerator.list_mines, "add": MarkersGenerator.add_mine},
            {"type": "bridges", "icon": "🌉", "px": 14, "min": 1, "each": 5, "multiplier": 1, "list": MarkersGenerator.list_bridges, "add": MarkersGenerator.add_bridge},
            {"type": "inns", "icon": "🍻", "px": 14, "min": 1, "each": 10, "multiplier": 1, "list": MarkersGenerator.list_inns, "add": MarkersGenerator.add_inn},
            {"type": "lighthouses", "icon": "🚨", "px": 14, "min": 1, "each": 2, "multiplier": 1, "list": MarkersGenerator.list_lighthouses, "add": MarkersGenerator.add_lighthouse},
            {"type": "waterfalls", "icon": "⟱", "dy": 54, "px": 16, "min": 1, "each": 5, "multiplier": 1, "list": MarkersGenerator.list_waterfalls, "add": MarkersGenerator.add_waterfall},
            {"type": "battlefields", "icon": "⚔️", "dy": 52, "min": 50, "each": 700, "multiplier": 1, "list": MarkersGenerator.list_battlefields, "add": MarkersGenerator.add_battlefield},
            {"type": "dungeons", "icon": "🗝️", "dy": 51, "px": 13, "min": 30, "each": 200, "multiplier": 1, "list": MarkersGenerator.list_dungeons, "add": MarkersGenerator.add_dungeon},
            {"type": "lake-monsters", "icon": "🐉", "dy": 48, "min": 2, "each": 10, "multiplier": 1, "list": MarkersGenerator.list_lake_monsters, "add": MarkersGenerator.add_lake_monster},
            {"type": "sea-monsters", "icon": "🦑", "min": 50, "each": 700, "multiplier": 1, "list": MarkersGenerator.list_sea_monsters, "add": MarkersGenerator.add_sea_monster},
            {"type": "sacred-mountains", "icon": "🗻", "dy": 48, "min": 1, "each": 5, "multiplier": 1, "list": MarkersGenerator.list_sacred_mountains, "add": MarkersGenerator.add_sacred_mountain},
            {"type": "sacred-forests", "icon": "🌳", "min": 30, "each": 1000, "multiplier": 1, "list": MarkersGenerator.list_sacred_forests, "add": MarkersGenerator.add_sacred_forest},
            {"type": "sacred-pineries", "icon": "🌲", "px": 13, "min": 30, "each": 800, "multiplier": 1, "list": MarkersGenerator.list_sacred_pineries, "add": MarkersGenerator.add_sacred_pinery},
        ]

