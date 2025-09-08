"""
FMG .map exporter

Builds a CRLF-delimited .map file compatible with Azgaar's FMG loader
by mirroring modules/io/save.js:prepareMapData output structure.

This focuses on core geometry/climate/biome/river parity; higher-level
entities (cultures, states, burgs, etc.) are emitted as empty structures
until implemented.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .core.voronoi_graph import VoronoiGraph
from .core.biomes import BiomeClassifier


def _csv(a: Iterable[Any]) -> str:
    return ",".join(str(int(x)) for x in a)


def _csv_float(a: Iterable[float], ndigits: int = 4) -> str:
    fmt = f"{{:.{ndigits}f}}"
    return ",".join(fmt.format(float(x)) for x in a)


def _ensure_len(arr: np.ndarray, n: int, fill: int = 0, dtype: Optional[str] = None) -> np.ndarray:
    if arr is None:
        return np.full(n, fill, dtype=dtype or np.int32)
    if len(arr) == n:
        return arr
    out = np.full(n, fill, dtype=arr.dtype if dtype is None else dtype)
    m = min(n, len(arr))
    out[:m] = arr[:m]
    return out


def _build_svg(width: float, height: float) -> str:
    """
    Build an FMG-compatible SVG skeleton.

    FMG expects a fairly rich structure under #map, including defs (with #deftemp
    and masks) and a set of named layer groups under #viewbox. If these groups
    are missing, the loader will not be able to attach layers and styles, which
    can manifest as a striped / semi-rendered background.

    This skeleton mirrors FMG's default structure closely, while keeping the
    content minimal. The important bit is that all expected ids exist so the
    loader can select and populate them.
    """

    w = int(width)
    h = int(height)

    # Keep this as a single string (no external assets required)
    svg = f"""
<svg id=\"map\" xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\">
  <defs>
    <g id=\"deftemp\">
      <g id=\"featurePaths\"></g>
      <g id=\"textPaths\"></g>
      <g id=\"statePaths\"></g>
      <g id=\"defs-emblems\"></g>
      <mask id=\"land\"></mask>
      <mask id=\"water\"></mask>
      <mask id=\"fog\" style=\"stroke-width:10;stroke:black;stroke-linejoin:round;stroke-opacity:0.1\">
        <rect x=\"0\" y=\"0\" width=\"100%\" height=\"100%\" fill=\"white\" stroke=\"none\" />
      </mask>
    </g>
    <!-- Lightweight placeholder pattern so references to url(#oceanic) resolve -->
    <pattern id=\"oceanic\" width=\"100\" height=\"100\" patternUnits=\"userSpaceOnUse\">
      <rect x=\"0\" y=\"0\" width=\"100\" height=\"100\" fill=\"#466eab\" opacity=\"0.2\" />
    </pattern>
    <mask id=\"vignette-mask\">
      <rect x=\"0\" y=\"0\" width=\"100%\" height=\"100%\" fill=\"white\"></rect>
      <rect id=\"vignette-rect\" fill=\"black\"></rect>
    </mask>
  </defs>
  <g id=\"viewbox\">
    <g id=\"ocean\">
      <g id=\"oceanLayers\"></g>
      <g id=\"oceanPattern\"></g>
    </g>
    <g id=\"lakes\">
      <g id=\"freshwater\"></g>
      <g id=\"salt\"></g>
      <g id=\"sinkhole\"></g>
      <g id=\"frozen\"></g>
      <g id=\"lava\"></g>
      <g id=\"dry\"></g>
    </g>
    <g id=\"landmass\"></g>
    <g id=\"texture\"></g>
    <g id=\"terrs\">
      <g id=\"oceanHeights\"></g>
      <g id=\"landHeights\"></g>
    </g>
    <g id=\"topography\"></g>
    <g id=\"biomes\"></g>
    <g id=\"cells\"></g>
    <g id=\"gridOverlay\"></g>
    <g id=\"coordinates\"></g>
    <g id=\"compass\"></g>
    <g id=\"rivers\"></g>
    <g id=\"terrain\"></g>
    <g id=\"relig\"></g>
    <g id=\"cults\"></g>
    <g id=\"regions\">
      <g id=\"statesBody\"></g>
      <g id=\"statesHalo\"></g>
    </g>
    <g id=\"provs\"></g>
    <g id=\"zones\"></g>
    <g id=\"borders\">
      <g id=\"stateBorders\"></g>
      <g id=\"provinceBorders\"></g>
    </g>
    <g id=\"routes\">
      <g id=\"roads\"></g>
      <g id=\"trails\"></g>
      <g id=\"searoutes\"></g>
    </g>
    <g id=\"temperature\"></g>
    <g id=\"coastline\">
      <g id=\"sea_island\"></g>
      <g id=\"lake_island\"></g>
    </g>
    <g id=\"ice\"></g>
    <g id=\"prec\"></g>
    <g id=\"population\">
      <g id=\"rural\"></g>
      <g id=\"urban\"></g>
    </g>
    <g id=\"emblems\">
      <g id=\"burgEmblems\"></g>
      <g id=\"provinceEmblems\"></g>
      <g id=\"stateEmblems\"></g>
    </g>
    <g id=\"labels\">
      <g id=\"burgLabels\"></g>
      <g id=\"states\"></g>
      <g id=\"addedLabels\"></g>
    </g>
    <g id=\"icons\">
      <g id=\"burgIcons\">
        <g id=\"cities\"></g>
        <g id=\"towns\"></g>
      </g>
      <g id=\"anchors\">
        <g id=\"cities\"></g>
        <g id=\"towns\"></g>
      </g>
    </g>
    <g id=\"armies\"></g>
    <g id=\"markers\"></g>
    <g id=\"fogging-cont\"><g id=\"fogging\"></g></g>
    <g id=\"ruler\"></g>
    <g id=\"debug\"></g>
  </g>
  <g id=\"scaleBar\"><rect id=\"scaleBarBack\"></rect></g>
  <g id=\"vignette\" mask=\"url(#vignette-mask)\"><rect x=\"0\" y=\"0\" width=\"100%\" height=\"100%\" /></g>
</svg>
""".strip()

    return svg


def export_fmg_map(
    out_path: str | Path,
    graph: VoronoiGraph,
    *,
    map_name: str = "py-fmg",
    seed: str = "seed",
    map_id: Optional[int] = None,
    temperatures: Optional[np.ndarray] = None,
    precipitation: Optional[np.ndarray] = None,
    biomes: Optional[np.ndarray] = None,
    rivers_json: Optional[List[Dict[str, Any]]] = None,
    features_list: Optional[List[Any]] = None,
    classifier: Optional[BiomeClassifier] = None,
    version: str = "1.108.0",
    minimal: bool = False,
    # Extended entities for full export
    settlements: Optional[Dict[int, Any]] = None,
    states: Optional[Dict[int, Any]] = None,
    provinces: Optional[Dict[int, Any]] = None,
    cell_provinces: Optional[Sequence[int]] = None,
    cultures: Optional[Dict[int, Any]] = None,
    cell_cultures: Optional[Sequence[int]] = None,
    religions: Optional[Dict[int, Any]] = None,
    cell_religions: Optional[Sequence[int]] = None,
    land_routes: Optional[List[Any]] = None,
    sea_routes: Optional[List[Any]] = None,
    markers: Optional[List[Any]] = None,
    regiments_by_state: Optional[Dict[int, List[Any]]] = None,
) -> Path:
    n = len(graph.points)
    map_id = map_id or int(np.random.randint(10**9))

    # Grid arrays (FMG names)
    heights = _ensure_len(graph.heights.astype(np.uint8), n, 0, dtype=np.uint8)
    prec = _ensure_len(precipitation if precipitation is not None else np.zeros(n, dtype=np.uint8), n, 0, dtype=np.uint8)
    feats = _ensure_len(getattr(graph, "feature_ids", None) if getattr(graph, "feature_ids", None) is not None else np.zeros(n, dtype=np.uint16), n, 0, dtype=np.uint16)
    dist = _ensure_len(getattr(graph, "distance_field", None) if getattr(graph, "distance_field", None) is not None else np.zeros(n, dtype=np.int8), n, 0, dtype=np.int8)
    temps = _ensure_len(temperatures if temperatures is not None else np.zeros(n, dtype=np.int8), n, 0, dtype=np.int8)

    # Pack cells arrays
    biome_ids = _ensure_len(biomes if biomes is not None else np.zeros(n, dtype=np.uint8), n, 0, dtype=np.uint8)
    burg = np.zeros(n, dtype=np.uint16)
    conf = _ensure_len(getattr(graph, "confluences", None) if getattr(graph, "confluences", None) is not None else np.zeros(n, dtype=np.uint8), n, 0, dtype=np.uint8)
    culture = np.zeros(n, dtype=np.uint16)
    fl = _ensure_len(getattr(graph, "flux", None) if getattr(graph, "flux", None) is not None else np.zeros(n, dtype=np.uint16), n, 0, dtype=np.uint16)
    pop = np.zeros(n, dtype=np.float32)
    rivers_cell = _ensure_len(getattr(graph, "river_ids", None) if getattr(graph, "river_ids", None) is not None else np.zeros(n, dtype=np.uint16), n, 0, dtype=np.uint16)
    slope = np.zeros(n, dtype=np.int16)
    state = np.zeros(n, dtype=np.uint16)
    religion = np.zeros(n, dtype=np.uint16)
    province = np.zeros(n, dtype=np.uint16)

    # Populate pack arrays where data is available
    if hasattr(graph, "cell_population") and graph.cell_population is not None:
        try:
            pop = _ensure_len(np.asarray(graph.cell_population, dtype=float), n, 0.0, dtype=np.float32)
        except Exception:
            pass
    if hasattr(graph, "cell_state") and graph.cell_state is not None:
        try:
            state = _ensure_len(np.asarray(graph.cell_state, dtype=np.uint16), n, 0, dtype=np.uint16)
        except Exception:
            pass
    if cell_cultures is not None:
        try:
            culture = _ensure_len(np.asarray(cell_cultures, dtype=np.uint16), n, 0, dtype=np.uint16)
        except Exception:
            pass
    if cell_religions is not None:
        try:
            religion = _ensure_len(np.asarray(cell_religions, dtype=np.uint16), n, 0, dtype=np.uint16)
        except Exception:
            pass
    if cell_provinces is not None:
        try:
            province = _ensure_len(np.asarray(cell_provinces, dtype=np.uint16), n, 0, dtype=np.uint16)
        except Exception:
            pass
    if settlements:
        # Fill burg array: point cell -> settlement id
        try:
            for sid, s in settlements.items():
                cid = int(getattr(s, "cell_id", -1))
                if 0 <= cid < n:
                    burg[cid] = int(sid)
        except Exception:
            pass

    # Rivers JSON: build from our hydrology structures if not provided
    if rivers_json is None and hasattr(graph, "rivers"):
        rivers_json = []
        for rid, r in getattr(graph, "rivers").items():
            rivers_json.append(
                {
                    "i": int(rid),
                    "cells": [int(c) for c in (r.cells or [])],
                    "width": float(getattr(r, "width", 1.0)),
                    "length": float(getattr(r, "length", 0.0)),
                    "discharge": float(getattr(r, "discharge", 0.0)),
                    "parent": int(getattr(r, "parent_id", 0) or 0),
                    "name": f"River {rid}",
                    "type": 1,
                }
            )
    rivers_json = rivers_json if rivers_json is not None else []

    # Biomes dictionary (colors, habitability, names)
    if classifier is None:
        classifier = BiomeClassifier()
    bcolors = ",".join(list(map(str, list(classifier.biome_data.colors))))
    bhabit = ",".join(str(int(x)) for x in list(classifier.biome_data.habitability))
    bnames = ",".join(list(map(str, list(classifier.biome_data.names))))
    biomes_dict_line = f"{bcolors}|{bhabit}|{bnames}"

    # Settings: mirror array size; fill with basics
    settings = [
        "km", 1, "km2", "m", 1.5, "°C",
        "", "", "", "", "", "",
        1.0, 1.0, 100, 50, "", "", 100,
        json.dumps({}),
        map_name,
        0, "", 1, 1.0, 50, "1.0",
    ]
    settings_line = "|".join(str(x) for x in settings)

    # Params
    license_text = "File can be loaded in azgaar.github.io/Fantasy-Map-Generator"
    params_line = "|".join([version, license_text, "2025-01-01", seed, str(int(graph.graph_width)), str(int(graph.graph_height)), str(map_id)])

    # Coordinates
    coords_line = json.dumps({"lat_n": 90, "lat_s": -90})

    # Notes, rulers, fonts, namesData
    notes_line = json.dumps([])
    rulers_line = ""
    fonts_line = json.dumps([])
    names_line = ""  # keep default name bases

    # SVG
    svg_line = _build_svg(graph.graph_width, graph.graph_height)

    # Grid general JSON (features filled after we build flist)
    grid_general = {
        "spacing": float(graph.spacing),
        "cellsX": int(graph.cells_x),
        "cellsY": int(graph.cells_y),
        "boundary": graph.boundary_points.tolist() if hasattr(graph, "boundary_points") else [],
        "points": graph.points.tolist(),
        # Will set to a dense features array later
        "features": [],
        "cellsDesired": int(graph.cells_desired),
    }
    # grid_general_line will be created after features are normalized

    # JSON blocks for higher-level entities (empty for now)
    def _to_serializable(obj: Any) -> Any:
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if isinstance(obj, (list, tuple)):
            return [_to_serializable(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): _to_serializable(v) for k, v in obj.items()}
        # Numpy
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if hasattr(obj, "__dict__"):
            return {k: _to_serializable(v) for k, v in obj.__dict__.items()}
        return str(obj)

    # features array (pack.features), index 0 must be an object
    flist = features_list
    if minimal:
        # Minimal safe set: all cells reference feature 0
        feats = np.zeros(n, dtype=np.uint16)
        flist = [{"id": 0, "type": "ocean", "land": False, "border": True}]
    else:
        if flist is None:
            flist = getattr(graph, "features", None)
        if flist is None or not isinstance(flist, list):
            flist = []
        # Ensure index 0 exists as an object
        if not flist:
            flist = [{"id": 0, "type": "ocean", "land": False, "border": True}]
        elif flist[0] in (None, False):
            flist[0] = {"id": 0, "type": "ocean", "land": False, "border": True}
        # Ensure length covers all feature ids referenced in feats
        max_id = int(np.max(feats)) if feats.size else 0
        if len(flist) <= max_id:
            # pad with oceans
            for i in range(len(flist), max_id + 1):
                flist.append({"id": i, "type": "ocean", "land": False, "border": True})
    features_json = json.dumps(_to_serializable(flist))

    # Also expose the same dense list on grid.general.features so
    # FMG reGraph (which reads grid.features) can safely access .type
    grid_general["features"] = flist
    grid_general_line = json.dumps(_to_serializable(grid_general))
    # Build extended JSON blocks
    def build_burgs()-> List[Dict[str, Any]]:
        if not settlements:
            return []
        out: List[Dict[str, Any]] = []
        for sid, s in sorted(settlements.items(), key=lambda kv: int(kv[0])):
            out.append(
                {
                    "i": int(sid),
                    "name": getattr(s, "name", f"Burg {sid}"),
                    "cell": int(getattr(s, "cell_id", 0)),
                    "x": float(getattr(s, "x", 0.0)),
                    "y": float(getattr(s, "y", 0.0)),
                    "population": float(getattr(s, "population", 0.0)),
                    "capital": bool(getattr(s, "is_capital", False)),
                    "state": int(getattr(s, "state_id", 0)),
                    "culture": int(getattr(s, "culture_id", 0)),
                    "port": int(1 if getattr(s, "is_port", False) or getattr(s, "port_id", 0) > 0 else 0),
                    "citadel": bool(getattr(s, "citadel", False)),
                    "plaza": bool(getattr(s, "plaza", False)),
                    "walls": bool(getattr(s, "walls", False)),
                    "shanty": bool(getattr(s, "shanty", False)),
                    "temple": bool(getattr(s, "temple", False)),
                }
            )
        return out

    def build_states() -> List[Dict[str, Any]]:
        if not states:
            return []
        out: List[Dict[str, Any]] = []
        for sid, st in sorted(states.items(), key=lambda kv: int(kv[0])):
            out.append(
                {
                    "i": int(sid),
                    "name": getattr(st, "name", f"State {sid}"),
                    "capital": int(getattr(st, "capital_id", 0)),
                    "color": getattr(st, "color", "#888888"),
                    "expansionism": float(getattr(st, "expansionism", 1.0)),
                    "type": getattr(st, "type", "Generic"),
                    "center": int(getattr(st, "center_cell", 0)),
                    "cells": [int(c) for c in getattr(st, "territory_cells", [])],
                }
            )
        return out

    def build_provinces() -> List[Dict[str, Any]]:
        if not provinces:
            return []
        out: List[Dict[str, Any]] = []
        for pid, p in sorted(provinces.items(), key=lambda kv: int(kv[0])):
            out.append(
                {
                    "i": int(pid),
                    "name": getattr(p, "name", f"Province {pid}"),
                    "state": int(getattr(p, "state_id", 0)),
                    "center": int(getattr(p, "center_cell", 0)),
                    "burg": int(getattr(p, "burg_id", 0)),
                }
            )
        return out

    def build_cultures() -> List[Dict[str, Any]]:
        if not cultures:
            return []
        out: List[Dict[str, Any]] = []
        for cid, c in sorted(cultures.items(), key=lambda kv: int(kv[0])):
            out.append(
                {
                    "i": int(cid),
                    "name": getattr(c, "name", f"Culture {cid}"),
                    "color": getattr(c, "color", "#999999"),
                    "type": getattr(c, "type", "Generic"),
                    "center": int(getattr(c, "center", 0)),
                }
            )
        return out

    def build_religions() -> List[Dict[str, Any]]:
        if not religions:
            return []
        out: List[Dict[str, Any]] = []
        for rid, r in sorted(religions.items(), key=lambda kv: int(kv[0])):
            out.append(
                {
                    "i": int(rid),
                    "name": getattr(r, "name", f"Religion {rid}"),
                    "color": getattr(r, "color", "#aaaaaa"),
                    "type": getattr(r, "type", "Organized"),
                    "form": getattr(r, "form", "Polytheism"),
                    "center": int(getattr(r, "center", 0)),
                    "expansion": getattr(r, "expansion", "global"),
                    "expansionism": float(getattr(r, "expansionism", 1.0)),
                    "code": getattr(r, "code", f"REL{rid}"),
                    "origins": [int(x) for x in getattr(r, "origins", [])],
                }
            )
        return out

    def build_markers() -> List[Dict[str, Any]]:
        if not markers:
            return []
        out: List[Dict[str, Any]] = []
        for m in markers:
            out.append(
                {
                    "i": int(getattr(m, "i", 0)),
                    "type": getattr(m, "type", "marker"),
                    "icon": getattr(m, "icon", ""),
                    "x": float(getattr(m, "x", 0.0)),
                    "y": float(getattr(m, "y", 0.0)),
                    "cell": int(getattr(m, "cell", 0)),
                    "name": getattr(m, "name", ""),
                    "legend": getattr(m, "legend", ""),
                    "dx": getattr(m, "dx", None),
                    "dy": getattr(m, "dy", None),
                    "px": getattr(m, "px", None),
                }
            )
        return out

    def build_routes() -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for src in (land_routes or []):
            out.append(
                {
                    "i": int(getattr(src, "id", 0)),
                    "kind": getattr(src, "kind", "land"),
                    "class": getattr(src, "cls", "road"),
                    "start": int(getattr(src, "start_settlement", 0)),
                    "end": int(getattr(src, "end_settlement", 0)),
                    "cells": [int(c) for c in (getattr(src, "cells", None) or [])],
                    "points": [[float(x), float(y)] for (x, y) in getattr(src, "coords", [])],
                }
            )
        for src in (sea_routes or []):
            out.append(
                {
                    "i": int(getattr(src, "id", 0)),
                    "kind": getattr(src, "kind", "sea"),
                    "class": getattr(src, "cls", "coastal"),
                    "start": int(getattr(src, "start_settlement", 0)),
                    "end": int(getattr(src, "end_settlement", 0)),
                    "cells": [int(c) for c in (getattr(src, "cells", None) or [])],
                    "points": [[float(x), float(y)] for (x, y) in getattr(src, "coords", [])],
                }
            )
        return out

    def build_regiments() -> List[Dict[str, Any]]:
        if not regiments_by_state:
            return []
        out: List[Dict[str, Any]] = []
        for sid, regs in regiments_by_state.items():
            for r in regs:
                out.append(
                    {
                        "i": int(getattr(r, "i", 0)),
                        "state": int(getattr(r, "state", sid)),
                        "a": int(getattr(r, "a", 0)),
                        "cell": int(getattr(r, "cell", 0)),
                        "x": float(getattr(r, "x", 0.0)),
                        "y": float(getattr(r, "y", 0.0)),
                        "u": getattr(r, "u", {}),
                        "n": int(getattr(r, "n", 0)),
                        "name": getattr(r, "name", "Regiment"),
                        "icon": getattr(r, "icon", ""),
                    }
                )
        return out

    burgs_json = json.dumps(_to_serializable(build_burgs()))
    states_json = json.dumps(_to_serializable(build_states()))
    cultures_json = json.dumps(_to_serializable(build_cultures()))
    religions_json = json.dumps(_to_serializable(build_religions()))
    provinces_json = json.dumps(_to_serializable(build_provinces()))
    markers_json = json.dumps(_to_serializable(build_markers()))
    routes_json = json.dumps(_to_serializable(build_routes()))
    cell_routes_json = json.dumps([])  # keep as placeholder (per-cell paths optional)
    zones_json = json.dumps([])

    # Lines in FMG order
    lines: List[str] = [
        params_line,
        settings_line,
        coords_line,
        biomes_dict_line,
        notes_line,
        svg_line,
        grid_general_line,
        _csv(heights),
        _csv(prec),
        _csv(feats),
        _csv(dist),
        _csv(temps),
        features_json,
        cultures_json,
        states_json,
        burgs_json,
        _csv(biome_ids),
        _csv(burg),
        _csv(conf),
        _csv(culture),
        _csv(fl),
        _csv_float(pop),
        _csv(rivers_cell),
        json.dumps(_to_serializable(build_regiments())),
        _csv(slope),
        _csv(state),
        _csv(religion),
        _csv(province),
        json.dumps([]),
        religions_json,
        provinces_json,
        names_line,
        json.dumps(rivers_json),
        rulers_line,
        fonts_line,
        markers_json,
        cell_routes_json,
        routes_json,
        zones_json,
    ]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # FMG uses CRLF
    out_path.write_text("\r\n".join(lines), encoding="utf-8")
    return out_path
