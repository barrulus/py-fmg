"""
Military/regiments generator — parity-aligned with FMG military-generator.js.

Generates platoons from burgs (settlements), aggregates into regiments (expected size ~300),
assigns names/icons, and returns per-state regiment lists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math

import numpy as np


@dataclass
class UnitOption:
    icon: str
    name: str
    rural: float
    urban: float
    crew: int
    power: float
    type: str
    separate: int  # 0 or 1


DEFAULT_UNITS: List[UnitOption] = [
    UnitOption("⚔️", "infantry", rural=0.25, urban=0.2, crew=1, power=1, type="melee", separate=0),
    UnitOption("🏹", "archers", rural=0.12, urban=0.2, crew=1, power=1, type="ranged", separate=0),
    UnitOption("🐴", "cavalry", rural=0.12, urban=0.03, crew=2, power=2, type="mounted", separate=0),
    UnitOption("💣", "artillery", rural=0.0, urban=0.03, crew=8, power=12, type="machinery", separate=0),
    UnitOption("🌊", "fleet", rural=0.0, urban=0.015, crew=100, power=50, type="naval", separate=1),
]


@dataclass
class Regiment:
    i: int
    state: int
    a: int  # total troops
    cell: int
    x: float
    y: float
    bx: float
    by: float
    u: Dict[str, int]  # composition by unit name
    n: int  # naval flag (1 fleet / 0 land)
    name: str
    icon: str


class MilitaryGenerator:
    def __init__(
        self,
        graph,
        settlements: Dict[int, Any],
        states: Dict[int, Any],
        units: Optional[List[UnitOption]] = None,
    ) -> None:
        self.g = graph
        self.settlements = settlements
        self.states = states
        self.units = units or DEFAULT_UNITS

    def generate(self) -> Dict[int, List[Regiment]]:
        # Build per-state platoon lists
        per_state_nodes: Dict[int, List[Dict[str, Any]]] = {}

        # For parity simplicity, generate from burgs only
        for s in self.settlements.values():
            if getattr(s, "population", 0) <= 0:
                continue
            state_id = int(getattr(s, "state_id", 0) or 0)
            if state_id <= 0:
                continue
            if state_id not in per_state_nodes:
                per_state_nodes[state_id] = []
            # Actual population
            actual_pop = float(s.population)
            if actual_pop < 500:
                continue
            # Base mobilization ~2.5%
            m = actual_pop / 40.0
            if getattr(s, "is_capital", False):
                m *= 1.2
            # Apply simple landmass/culture/religion modifiers if present
            # Skipped for brevity; parity can be extended using culture/religion arrays

            # Cell type modifier: highland / wetland / generic
            cell = int(getattr(s, "cell_id", -1))
            typ = self._cell_type(cell)

            for u in self.units:
                perc = float(u.urban)
                if perc <= 0:
                    continue
                # Naval units only for ports
                if u.type == "naval" and not getattr(s, "is_port", False):
                    continue
                mod = self._burg_type_modifier(typ, u.type)
                total = int(round(m * perc * mod))
                if total <= 0:
                    continue
                x, y = float(s.x), float(s.y)
                nflag = 1 if u.type == "naval" else 0
                per_state_nodes[state_id].append({
                    "cell": cell,
                    "a": total,
                    "t": total,
                    "x": x,
                    "y": y,
                    "u": u.name,
                    "n": nflag,
                    "s": u.separate,
                    "type": u.type,
                })

        # Aggregate nodes into regiments
        regiments_by_state: Dict[int, List[Regiment]] = {}
        expected = 300
        rid_base = 0
        for sid, nodes in per_state_nodes.items():
            if not nodes:
                regiments_by_state[sid] = []
                continue
            # Spatial index: naive O(n^2) merging with pruning by radius; acceptable for typical sizes
            # Overlap merge: radius 20
            nodes = sorted(nodes, key=lambda n: n["a"])  # ascending

            def can_merge(n0, n1):
                return (not n0["s"] and not n1["s"]) or (n0["u"] == n1["u"])  # separate units cannot mix

            for i in range(len(nodes)):
                n0 = nodes[i]
                if n0["t"] <= 0:
                    continue
                # Find overlapping within 20
                for j in range(i + 1, len(nodes)):
                    n1 = nodes[j]
                    if n1["t"] <= 0:
                        continue
                    if not can_merge(n0, n1):
                        continue
                    if (n1["x"] - n0["x"]) ** 2 + (n1["y"] - n0["y"]) ** 2 <= 20 ** 2:
                        n0["t"] += n1["t"]
                        n1["t"] = 0
                # Grow to expected by merging nearest small nodes
                if n0["t"] > expected:
                    continue
                r = (expected - n0["t"]) / (40 if n0["s"] else 20)
                r2 = r * r
                for j in range(i + 1, len(nodes)):
                    n1 = nodes[j]
                    if n1["t"] <= 0 or n0["t"] >= expected:
                        continue
                    if not can_merge(n0, n1):
                        continue
                    if (n1["x"] - n0["x"]) ** 2 + (n1["y"] - n0["y"]) ** 2 <= r2 and n1["t"] < expected:
                        n0["t"] += n1["t"]
                        n1["t"] = 0

            # Build regiments list
            regs: List[Regiment] = []
            for idx, r in enumerate(sorted([n for n in nodes if n["t"] > 0], key=lambda n: -n["t"])):
                comp: Dict[str, int] = {r["u"]: r["a"]}
                # Could carry childen merges; we already summed into t
                # Assign name and icon
                name = self._regiment_name(r, idx, naval=(r["n"] == 1))
                icon = self._regiment_icon(r, sid)
                regs.append(
                    Regiment(
                        i=idx,
                        state=sid,
                        a=int(r["t"]),
                        cell=int(r["cell"]),
                        x=float(r["x"]),
                        y=float(r["y"]),
                        bx=float(r["x"]),
                        by=float(r["y"]),
                        u=comp,
                        n=int(r["n"]),
                        name=name,
                        icon=icon,
                    )
                )
            regiments_by_state[sid] = regs

        return regiments_by_state

    # ---- Helpers ----
    def _cell_type(self, cell: int) -> str:
        if cell < 0:
            return "generic"
        biomes = getattr(self.g, "biomes", None)
        if biomes and hasattr(biomes, "cell_biomes"):
            b = int(biomes.cell_biomes[cell])
            if b in (1, 2, 3, 4):
                return "nomadic"
            if b in (7, 8, 9, 12):
                return "wetland"
        if self.g.heights[cell] >= 70:
            return "highland"
        return "generic"

    def _burg_type_modifier(self, typ: str, unit_type: str) -> float:
        table = {
            "nomadic": {"melee": 0.3, "ranged": 0.8, "mounted": 3.0, "machinery": 0.4, "naval": 1.0, "armored": 1.6, "aviation": 1.0, "magical": 0.5},
            "wetland": {"melee": 1.0, "ranged": 1.6, "mounted": 0.2, "machinery": 1.2, "naval": 1.0, "armored": 0.2, "aviation": 0.5, "magical": 0.5},
            "highland": {"melee": 1.2, "ranged": 2.0, "mounted": 0.3, "machinery": 3.0, "naval": 1.0, "armored": 0.8, "aviation": 0.3, "magical": 2.0},
            "generic": {"melee": 1.0, "ranged": 1.0, "mounted": 1.0, "machinery": 1.0, "naval": 1.0, "armored": 1.0, "aviation": 1.0, "magical": 1.0},
        }
        return float(table.get(typ, table["generic"]).get(unit_type, 1.0))

    def _regiment_name(self, node: Dict[str, Any], ordinal: int, naval: bool) -> str:
        form = "Fleet" if naval else "Regiment"
        number = self._ordinal(ordinal + 1)
        return f"{number} {form}"

    def _regiment_icon(self, node: Dict[str, Any], state_id: int) -> str:
        if not node.get("n") and node.get("u") is None:
            return "🔰"
        # Royal: if state form monarchy and burg capital; not tracked -> return unit icon
        main_unit = node.get("u", "infantry")
        for u in self.units:
            if u.name == main_unit:
                return u.icon
        return "⚔️"

    @staticmethod
    def _ordinal(n: int) -> str:
        # simple English ordinals
        if 10 <= n % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"

