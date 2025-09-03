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
    # Minimal valid FMG SVG skeleton
    layers = [
        "ocean", "coastline", "lakes", "rivers", "terrain", "relief",
        "states", "provinces", "routes", "labels", "burgs", "markers",
    ]
    groups = "\n".join(f'<g id="{gid}"></g>' for gid in layers)
    return (
        f"<svg id=\"map\" xmlns=\"http://www.w3.org/2000/svg\" width=\"{int(width)}\" height=\"{int(height)}\">"
        f"<g id=\"viewbox\">{groups}</g></svg>"
    )


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
    cultures_json = json.dumps([])
    states_json = json.dumps([])
    burgs_json = json.dumps([])
    religions_json = json.dumps([])
    provinces_json = json.dumps([])
    markers_json = json.dumps([])
    cell_routes_json = json.dumps([])
    routes_json = json.dumps([])
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
        json.dumps([]),
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
