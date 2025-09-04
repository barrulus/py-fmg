"""
GeoJSON exporter for Voronoi-derived layers.

Provides helpers to write cells as GeoJSON FeatureCollections and optional
Leaflet preview pages. Focuses on cells first; can be extended for rivers,
routes, and thematic overlays later.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .core.voronoi_graph import VoronoiGraph
from .core.biomes import BiomeClassifier
from .core.cultures import Culture
from .core.settlements import Settlement


def _order_polygon_vertices(coords: np.ndarray) -> np.ndarray:
    """Return vertices ordered around centroid (clockwise).

    SciPy ridge iteration order is not guaranteed to be circular for a cell.
    We sort by angle around the centroid for a consistent polygon ring.
    """
    if len(coords) <= 2:
        return coords
    c = coords.mean(axis=0)
    angles = np.arctan2(coords[:, 1] - c[1], coords[:, 0] - c[0])
    order = np.argsort(angles)
    return coords[order]


def _cell_polygon(graph: VoronoiGraph, cell_id: int) -> List[List[float]]:
    """Build a closed polygon ring for a given cell.

    Returns a list of [x, y] pairs (closed: last == first). Skips invalid
    or underdefined cells (fewer than 3 vertices) by returning an empty list.
    """
    v_ids = graph.cell_vertices[cell_id] if cell_id < len(graph.cell_vertices) else []
    if not v_ids:
        return []
    coords = []
    for v in v_ids:
        if 0 <= v < len(graph.vertex_coordinates):
            xy = graph.vertex_coordinates[v]
            coords.append([float(xy[0]), float(xy[1])])
    if len(coords) < 3:
        return []
    ordered = _order_polygon_vertices(np.asarray(coords))
    ring = ordered.tolist()
    # close ring if needed
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def build_cells_fc(
    graph: VoronoiGraph,
    map_id: str,
    include_heights: bool = True,
) -> Dict[str, Any]:
    """Build Voronoi cells FeatureCollection (no write)."""
    features: List[Dict[str, Any]] = []
    for i in range(len(graph.points)):
        ring = _cell_polygon(graph, i)
        if not ring:
            continue

        centroid_x = float(np.mean([p[0] for p in ring[:-1]]))  # exclude duplicate last point
        centroid_y = float(np.mean([p[1] for p in ring[:-1]]))

        neighbors = graph.cell_neighbors[i] if i < len(graph.cell_neighbors) else []
        neighbors = [int(n) for n in neighbors]

        props: Dict[str, Any] = {
            "map_id": map_id,
            "cell_id": i,
            "neighbors": neighbors,
            "border": int(graph.cell_border_flags[i]) if i < len(graph.cell_border_flags) else 0,
            "centroid": [centroid_x, centroid_y],
        }
        if include_heights and i < len(graph.heights):
            h = int(graph.heights[i])
            props["height"] = h
            props["is_land"] = bool(h >= 20)

        # Coastline flag if distance_field present
        if getattr(graph, "distance_field", None) is not None:
            df = int(graph.distance_field[i])
            props["is_coast"] = df in (1, -1)

        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": props,
            }
        )

    return {"type": "FeatureCollection", "features": features}


def export_cells_geojson(
    graph: VoronoiGraph,
    out_dir: str | os.PathLike,
    map_id: str,
    include_heights: bool = True,
) -> Path:
    """Export Voronoi cells as a GeoJSON FeatureCollection.

    - geometry: Polygon (Voronoi cell)
    - properties: cell_id, neighbors, border, centroid, height (optional)
    """
    out_dir = Path(out_dir)
    layer_dir = out_dir / "geojson" / map_id
    layer_dir.mkdir(parents=True, exist_ok=True)

    fc = build_cells_fc(graph, map_id, include_heights)
    out_path = layer_dir / "cells.geojson"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False)
    return out_path


def export_coastlines_geojson(
    graph: VoronoiGraph,
    out_dir: str | os.PathLike,
    map_id: str,
) -> Path:
    """Export coastline segments as a MultiLineString.

    For each edge between a land cell and a water cell, output the shared
    Voronoi edge as a linestring. Deduplicate segments by vertex ids.
    """
    out_dir = Path(out_dir)
    layer_dir = out_dir / "geojson" / map_id
    layer_dir.mkdir(parents=True, exist_ok=True)

    seg_keys = set()
    segments: List[List[List[float]]] = []

    n = len(graph.points)
    is_land = [False] * n
    if getattr(graph, "heights", None) is not None:
        is_land = [bool(int(h) >= 20) for h in graph.heights[:n]]

    for i in range(n):
        for j in graph.cell_neighbors[i]:
            if j <= i:
                continue  # avoid duplicates
            if is_land[i] == is_land[j]:
                continue

            # Find the two shared vertices between cells i and j
            candidates = []
            for v in graph.cell_vertices[i]:
                # vertex_cells[v] contains adjacent cells to this vertex
                if j in graph.vertex_cells[v]:
                    candidates.append(v)
            if len(candidates) < 2:
                continue
            # There may be more than 2 due to ordering; pick first 2 distinct
            v1, v2 = candidates[0], candidates[1]

            key = tuple(sorted((v1, v2)))
            if key in seg_keys:
                continue
            seg_keys.add(key)

            p1 = graph.vertex_coordinates[v1]
            p2 = graph.vertex_coordinates[v2]
            segments.append([[float(p1[0]), float(p1[1])], [float(p2[0]), float(p2[1])]])

    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "MultiLineString", "coordinates": segments},
                "properties": {"map_id": map_id, "layer": "coastlines"},
            }
        ],
    }

    out_path = layer_dir / "coastlines.geojson"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False)
    return out_path


def build_coastlines_fc(graph: VoronoiGraph, map_id: str) -> Dict[str, Any]:
    """Build coastline MultiLineString FC without writing to disk."""
    seg_keys = set()
    segments: List[List[List[float]]] = []

    n = len(graph.points)
    is_land = [False] * n
    if getattr(graph, "heights", None) is not None:
        is_land = [bool(int(h) >= 20) for h in graph.heights[:n]]

    for i in range(n):
        for j in graph.cell_neighbors[i]:
            if j <= i:
                continue
            if is_land[i] == is_land[j]:
                continue
            candidates = []
            for v in graph.cell_vertices[i]:
                if j in graph.vertex_cells[v]:
                    candidates.append(v)
            if len(candidates) < 2:
                continue
            v1, v2 = candidates[0], candidates[1]
            key = tuple(sorted((v1, v2)))
            if key in seg_keys:
                continue
            seg_keys.add(key)
            p1 = graph.vertex_coordinates[v1]
            p2 = graph.vertex_coordinates[v2]
            segments.append([[float(p1[0]), float(p1[1])], [float(p2[0]), float(p2[1])]])

    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "MultiLineString", "coordinates": segments},
                "properties": {"map_id": map_id, "layer": "coastlines"},
            }
        ],
    }


def export_watermask_geojson(
    graph: VoronoiGraph,
    out_dir: str | os.PathLike,
    map_id: str,
) -> Path:
    """Export a simple land/water/coast mask as polygons.

    Properties per feature:
    - is_land: bool
    - is_ocean: bool
    - is_lake: bool
    - is_coast: bool (touches opposite type)
    """
    out_dir = Path(out_dir)
    layer_dir = out_dir / "geojson" / map_id
    layer_dir.mkdir(parents=True, exist_ok=True)

    features: List[Dict[str, Any]] = []

    # Build quick lookup for feature types if available
    feature_types: Dict[int, str] = {}
    if getattr(graph, "features", None) is not None and getattr(graph, "feature_ids", None) is not None:
        for f in graph.features:
            if not f:
                continue
            feature_types[int(f.id)] = str(f.type)

    n = len(graph.points)
    for i in range(n):
        ring = _cell_polygon(graph, i)
        if not ring:
            continue

        h = int(graph.heights[i]) if getattr(graph, "heights", None) is not None else 0
        is_land = bool(h >= 20)
        is_coast = False
        if getattr(graph, "distance_field", None) is not None:
            df = int(graph.distance_field[i])
            is_coast = df in (1, -1)

        is_ocean = False
        is_lake = False
        if feature_types and getattr(graph, "feature_ids", None) is not None:
            fid = int(graph.feature_ids[i]) if i < len(graph.feature_ids) else 0
            ftype = feature_types.get(fid, None)
            is_ocean = ftype == "ocean"
            is_lake = ftype == "lake"
        else:
            # Fallback: classify lakes as water cells not on border and not coast if we cannot resolve features
            is_ocean = (not is_land) and bool(graph.cell_border_flags[i])
            is_lake = (not is_land) and (not is_ocean)

        props = {
            "map_id": map_id,
            "cell_id": i,
            "is_land": is_land,
            "is_ocean": is_ocean,
            "is_lake": is_lake,
            "is_coast": is_coast,
        }

        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": props,
            }
        )

    fc = {"type": "FeatureCollection", "features": features}
    out_path = layer_dir / "cells_watermask.geojson"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False)
    return out_path


def build_watermask_fc(graph: VoronoiGraph, map_id: str) -> Dict[str, Any]:
    """Build watermask FeatureCollection without writing to disk."""
    features: List[Dict[str, Any]] = []

    feature_types: Dict[int, str] = {}
    if getattr(graph, "features", None) is not None and getattr(graph, "feature_ids", None) is not None:
        for f in graph.features:
            if not f:
                continue
            feature_types[int(f.id)] = str(f.type)

    n = len(graph.points)
    for i in range(n):
        ring = _cell_polygon(graph, i)
        if not ring:
            continue
        h = int(graph.heights[i]) if getattr(graph, "heights", None) is not None else 0
        is_land = bool(h >= 20)
        is_coast = False
        if getattr(graph, "distance_field", None) is not None:
            df = int(graph.distance_field[i])
            is_coast = df in (1, -1)

        is_ocean = False
        is_lake = False
        if feature_types and getattr(graph, "feature_ids", None) is not None:
            fid = int(graph.feature_ids[i]) if i < len(graph.feature_ids) else 0
            ftype = feature_types.get(fid, None)
            is_ocean = ftype == "ocean"
            is_lake = ftype == "lake"
        else:
            is_ocean = (not is_land) and bool(graph.cell_border_flags[i])
            is_lake = (not is_land) and (not is_ocean)

        props = {
            "map_id": map_id,
            "cell_id": i,
            "is_land": is_land,
            "is_ocean": is_ocean,
            "is_lake": is_lake,
            "is_coast": is_coast,
        }
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": props,
        })

    return {"type": "FeatureCollection", "features": features}


def export_climate_geojson(
    graph: VoronoiGraph,
    temperatures: np.ndarray,
    precipitation: np.ndarray,
    out_dir: str | os.PathLike,
    map_id: str,
) -> Path:
    """Export per-cell climate as polygons with temperature and precipitation."""
    out_dir = Path(out_dir)
    layer_dir = out_dir / "geojson" / map_id
    layer_dir.mkdir(parents=True, exist_ok=True)

    features: List[Dict[str, Any]] = []
    n = len(graph.points)
    for i in range(n):
        ring = _cell_polygon(graph, i)
        if not ring:
            continue
        props = {
            "map_id": map_id,
            "cell_id": int(i),
            "temperature": float(temperatures[i]) if i < len(temperatures) else None,
            "precipitation": float(precipitation[i]) if i < len(precipitation) else None,
        }
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": props,
        })

    fc = {"type": "FeatureCollection", "features": features}
    out_path = layer_dir / "cells_climate.geojson"
    out_path.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
    return out_path


def build_climate_fc(
    graph: VoronoiGraph,
    temperatures: np.ndarray,
    precipitation: np.ndarray,
    map_id: str,
) -> Dict[str, Any]:
    features: List[Dict[str, Any]] = []
    n = len(graph.points)
    for i in range(n):
        ring = _cell_polygon(graph, i)
        if not ring:
            continue
        props = {
            "map_id": map_id,
            "cell_id": int(i),
            "temperature": float(temperatures[i]) if i < len(temperatures) else None,
            "precipitation": float(precipitation[i]) if i < len(precipitation) else None,
        }
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": props,
        })
    return {"type": "FeatureCollection", "features": features}


def export_biomes_geojson(
    graph: VoronoiGraph,
    biome_ids: np.ndarray,
    classifier: BiomeClassifier,
    out_dir: str | os.PathLike,
    map_id: str,
) -> Path:
    """Export per-cell biomes as polygons with id, name, and color."""
    out_dir = Path(out_dir)
    layer_dir = out_dir / "geojson" / map_id
    layer_dir.mkdir(parents=True, exist_ok=True)

    features: List[Dict[str, Any]] = []
    n = len(graph.points)
    for i in range(n):
        ring = _cell_polygon(graph, i)
        if not ring:
            continue
        bid = int(biome_ids[i]) if i < len(biome_ids) else 0
        props = {
            "map_id": map_id,
            "cell_id": int(i),
            "biome_id": bid,
            "biome_name": classifier.get_biome_name(bid),
            "color": classifier.get_biome_color(bid),
        }
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": props,
        })

    fc = {"type": "FeatureCollection", "features": features}
    out_path = layer_dir / "cells_biomes.geojson"
    out_path.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
    return out_path


def build_biomes_fc(
    graph: VoronoiGraph,
    biome_ids: np.ndarray,
    classifier: BiomeClassifier,
    map_id: str,
) -> Dict[str, Any]:
    features: List[Dict[str, Any]] = []
    n = len(graph.points)
    for i in range(n):
        ring = _cell_polygon(graph, i)
        if not ring:
            continue
        bid = int(biome_ids[i]) if i < len(biome_ids) else 0
        props = {
            "map_id": map_id,
            "cell_id": int(i),
            "biome_id": bid,
            "biome_name": classifier.get_biome_name(bid),
            "color": classifier.get_biome_color(bid),
        }
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": props,
        })
    return {"type": "FeatureCollection", "features": features}


def export_rivers_geojson(
    graph: VoronoiGraph,
    rivers: Dict[int, Any],
    out_dir: str | os.PathLike,
    map_id: str,
) -> Path:
    """Export rivers as a FeatureCollection of LineStrings.

    Each river feature contains properties: river_id, discharge, width, length,
    source_distance, and cells (the cell indices path).
    """
    out_dir = Path(out_dir)
    layer_dir = out_dir / "geojson" / map_id
    layer_dir.mkdir(parents=True, exist_ok=True)

    feats: List[Dict[str, Any]] = []
    for rid, river in rivers.items():
        cells = getattr(river, "cells", [])
        if not cells or len(cells) < 2:
            continue
        coords: List[List[float]] = []
        for c in cells:
            if c < 0 or c >= len(graph.points):
                continue
            p = graph.points[c]
            coords.append([float(p[0]), float(p[1])])
        if len(coords) < 2:
            continue
        props = {
            "map_id": map_id,
            "river_id": int(rid),
            "discharge": float(getattr(river, "discharge", 0.0)),
            "width": float(getattr(river, "width", 0.0)),
            "length": float(getattr(river, "length", 0.0)),
            "source_distance": float(getattr(river, "source_distance", 0.0)),
            "cells": [int(x) for x in cells],
        }
        feats.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": props,
        })

    fc = {"type": "FeatureCollection", "features": feats}
    out_path = layer_dir / "rivers.geojson"
    out_path.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
    return out_path


def build_rivers_fc(graph: VoronoiGraph, rivers: Dict[int, Any], map_id: str) -> Dict[str, Any]:
    feats: List[Dict[str, Any]] = []
    for rid, river in rivers.items():
        cells = getattr(river, "cells", [])
        if not cells or len(cells) < 2:
            continue
        coords: List[List[float]] = []
        for c in cells:
            if c < 0 or c >= len(graph.points):
                continue
            p = graph.points[c]
            coords.append([float(p[0]), float(p[1])])
        if len(coords) < 2:
            continue
        props = {
            "map_id": map_id,
            "river_id": int(rid),
            "discharge": float(getattr(river, "discharge", 0.0)),
            "width": float(getattr(river, "width", 0.0)),
            "length": float(getattr(river, "length", 0.0)),
            "source_distance": float(getattr(river, "source_distance", 0.0)),
            "cells": [int(x) for x in cells],
        }
        feats.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": props,
        })
    return {"type": "FeatureCollection", "features": feats}


def export_cell_cultures_geojson(
    graph: VoronoiGraph,
    cell_cultures: np.ndarray,
    cultures: Dict[int, Culture],
    out_dir: str | os.PathLike,
    map_id: str,
) -> Path:
    """Export per-cell culture assignment as polygons colored by culture."""
    out_dir = Path(out_dir)
    layer_dir = out_dir / "geojson" / map_id
    layer_dir.mkdir(parents=True, exist_ok=True)

    feats: List[Dict[str, Any]] = []
    n = len(graph.points)
    for i in range(n):
        ring = _cell_polygon(graph, i)
        if not ring:
            continue
        cid = int(cell_cultures[i]) if i < len(cell_cultures) else 0
        c = cultures.get(cid)
        props = {
            "map_id": map_id,
            "cell_id": i,
            "culture_id": cid,
            "culture_name": getattr(c, "name", f"Culture {cid}") if c else f"Culture {cid}",
            "color": getattr(c, "color", "#888888") if c else "#888888",
            "type": getattr(c, "type", "Generic") if c else "Generic",
        }
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": props,
        })

    fc = {"type": "FeatureCollection", "features": feats}
    out_path = layer_dir / "cells_cultures.geojson"
    out_path.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
    return out_path


def build_cell_cultures_fc(
    graph: VoronoiGraph,
    cell_cultures: np.ndarray,
    cultures: Dict[int, Culture],
    map_id: str,
) -> Dict[str, Any]:
    feats: List[Dict[str, Any]] = []
    n = len(graph.points)
    for i in range(n):
        ring = _cell_polygon(graph, i)
        if not ring:
            continue
        cid = int(cell_cultures[i]) if i < len(cell_cultures) else 0
        c = cultures.get(cid)
        props = {
            "map_id": map_id,
            "cell_id": i,
            "culture_id": cid,
            "culture_name": getattr(c, "name", f"Culture {cid}") if c else f"Culture {cid}",
            "color": getattr(c, "color", "#888888") if c else "#888888",
            "type": getattr(c, "type", "Generic") if c else "Generic",
        }
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": props,
        })
    return {"type": "FeatureCollection", "features": feats}


def export_cultures_points_geojson(
    graph: VoronoiGraph,
    cultures: Dict[int, Culture],
    out_dir: str | os.PathLike,
    map_id: str,
) -> Path:
    """Export culture centers as point features."""
    out_dir = Path(out_dir)
    layer_dir = out_dir / "geojson" / map_id
    layer_dir.mkdir(parents=True, exist_ok=True)

    feats: List[Dict[str, Any]] = []
    for cid, c in cultures.items():
        center = int(getattr(c, "center", 0))
        if 0 <= center < len(graph.points):
            x, y = graph.points[center]
        else:
            x, y = 0.0, 0.0
        props = {
            "map_id": map_id,
            "culture_id": int(cid),
            "name": getattr(c, "name", f"Culture {cid}"),
            "color": getattr(c, "color", "#888888"),
            "type": getattr(c, "type", "Generic"),
            "center_cell": center,
        }
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(x), float(y)]},
            "properties": props,
        })

    fc = {"type": "FeatureCollection", "features": feats}
    out_path = layer_dir / "cultures.geojson"
    out_path.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
    return out_path


def build_cultures_points_fc(
    graph: VoronoiGraph,
    cultures: Dict[int, Culture],
    map_id: str,
) -> Dict[str, Any]:
    feats: List[Dict[str, Any]] = []
    for cid, c in cultures.items():
        center = int(getattr(c, "center", 0))
        if 0 <= center < len(graph.points):
            x, y = graph.points[center]
        else:
            x, y = 0.0, 0.0
        props = {
            "map_id": map_id,
            "culture_id": int(cid),
            "name": getattr(c, "name", f"Culture {cid}"),
            "color": getattr(c, "color", "#888888"),
            "type": getattr(c, "type", "Generic"),
            "center_cell": center,
        }
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(x), float(y)]},
            "properties": props,
        })
    return {"type": "FeatureCollection", "features": feats}


def export_burgs_points_geojson(
    settlements: Dict[int, Settlement],
    out_dir: str | os.PathLike,
    map_id: str,
) -> Path:
    """Export settlements (burgs) as point GeoJSON."""
    out_dir = Path(out_dir)
    layer_dir = out_dir / "geojson" / map_id
    layer_dir.mkdir(parents=True, exist_ok=True)

    feats: List[Dict[str, Any]] = []
    for sid, s in settlements.items():
        props = {
            "map_id": map_id,
            "burg_id": int(sid),
            "name": s.name,
            "population": float(s.population),
            "is_capital": bool(s.is_capital),
            "state_id": int(s.state_id),
        }
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(s.x), float(s.y)]},
            "properties": props,
        })

    fc = {"type": "FeatureCollection", "features": feats}
    out_path = layer_dir / "burgs.geojson"
    out_path.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
    return out_path


def build_burgs_points_fc(
    settlements: Dict[int, Settlement],
    map_id: str,
) -> Dict[str, Any]:
    feats: List[Dict[str, Any]] = []
    for sid, s in settlements.items():
        props = {
            "map_id": map_id,
            "burg_id": int(sid),
            "name": s.name,
            "population": float(s.population),
            "is_capital": bool(s.is_capital),
            "state_id": int(s.state_id),
        }
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(s.x), float(s.y)]},
            "properties": props,
        })
    return {"type": "FeatureCollection", "features": feats}


def _shared_edge_midpoint(graph: VoronoiGraph, a: int, b: int) -> List[float]:
    """Return midpoint of the shared Voronoi edge between two neighboring cells.

    Falls back to cell center if shared edge cannot be determined.
    """
    try:
        va = set(graph.cell_vertices[a])
        vb = set(graph.cell_vertices[b])
        shared = list(va.intersection(vb))
        if len(shared) >= 2:
            v1, v2 = shared[0], shared[1]
            p1 = graph.vertex_coordinates[v1]
            p2 = graph.vertex_coordinates[v2]
            return [float((p1[0] + p2[0]) / 2.0), float((p1[1] + p2[1]) / 2.0)]
    except Exception:
        pass
    # Fallback: midpoint of cell centers
    pa = graph.points[a]
    pb = graph.points[b]
    return [float((pa[0] + pb[0]) / 2.0), float((pa[1] + pb[1]) / 2.0)]


def _catmull_rom_spline(
    points: List[List[float]],
    alpha: float = 0.5,
    segments: int = 8,
) -> List[List[float]]:
    """Centripetal Catmull–Rom spline through points.

    - alpha=0.5 gives centripetal parameterization (avoids loops/overshoot)
    - segments is the number of samples per original segment
    """
    n = len(points)
    if n < 2:
        return points

    def tj(ti: float, pi: List[float], pj: List[float]) -> float:
        dx = pj[0] - pi[0]
        dy = pj[1] - pi[1]
        d = math.sqrt(dx * dx + dy * dy) ** alpha
        if d < 1e-6:
            d = 1e-6
        return ti + d

    out: List[List[float]] = []
    for i in range(n - 1):
        p0 = points[i - 1] if i - 1 >= 0 else points[i]
        p1 = points[i]
        p2 = points[i + 1]
        p3 = points[i + 2] if i + 2 < n else points[i + 1]

        t0 = 0.0
        t1 = tj(t0, p0, p1)
        t2 = tj(t1, p1, p2)
        t3 = tj(t2, p2, p3)

        # First point on the segment
        if not out:
            out.append([p1[0], p1[1]])

        # Sample between t1 and t2
        for k in range(1, segments + 1):
            t = t1 + (t2 - t1) * (k / segments)

            def lerp_pa(pa, pb, ta, tb, t):
                denom = (tb - ta)
                if abs(denom) < 1e-9:
                    return [pa[0], pa[1]]
                return [
                    (tb - t) / denom * pa[0] + (t - ta) / denom * pb[0],
                    (tb - t) / denom * pa[1] + (t - ta) / denom * pb[1],
                ]

            A1 = lerp_pa(p0, p1, t0, t1, t)
            A2 = lerp_pa(p1, p2, t1, t2, t)
            A3 = lerp_pa(p2, p3, t2, t3, t)
            B1 = lerp_pa(A1, A2, t0, t2, t)
            B2 = lerp_pa(A2, A3, t1, t3, t)
            C = lerp_pa(B1, B2, t1, t2, t)

            out.append([C[0], C[1]])

    return out


def export_rivers_smooth_geojson(
    graph: VoronoiGraph,
    rivers: Dict[int, Any],
    out_dir: str | os.PathLike,
    map_id: str,
    alpha: float = 0.5,
    segments_per_edge: int = 8,
) -> Path:
    """Export smoothed, edge-following rivers as LineStrings.

    Builds a path using midpoints of shared Voronoi edges between consecutive
    river cells, then applies Chaikin smoothing.
    """
    out_dir = Path(out_dir)
    layer_dir = out_dir / "geojson" / map_id
    layer_dir.mkdir(parents=True, exist_ok=True)

    feats: List[Dict[str, Any]] = []
    for rid, river in rivers.items():
        cells = getattr(river, "cells", [])
        if not cells or len(cells) < 2:
            continue
        waypoints: List[List[float]] = []
        # start at center of first cell for a natural source
        p0 = graph.points[cells[0]]
        waypoints.append([float(p0[0]), float(p0[1])])
        for i in range(len(cells) - 1):
            a, b = cells[i], cells[i + 1]
            mid = _shared_edge_midpoint(graph, a, b)
            # avoid duplicates
            if not waypoints or mid != waypoints[-1]:
                waypoints.append(mid)
        # end at center of mouth cell to extend to the sea
        p_last = graph.points[cells[-1]]
        last = [float(p_last[0]), float(p_last[1])]
        if waypoints[-1] != last:
            waypoints.append(last)

        smooth = _catmull_rom_spline(waypoints, alpha=alpha, segments=segments_per_edge)
        if len(smooth) < 2:
            continue

        props = {
            "map_id": map_id,
            "river_id": int(rid),
            "discharge": float(getattr(river, "discharge", 0.0)),
            "width": float(getattr(river, "width", 0.0)),
            "length": float(getattr(river, "length", 0.0)),
            "source_distance": float(getattr(river, "source_distance", 0.0)),
            "cells": [int(x) for x in cells],
        }
        feats.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": smooth},
            "properties": props,
        })

    fc = {"type": "FeatureCollection", "features": feats}
    out_path = layer_dir / "rivers_smooth.geojson"
    out_path.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
    return out_path


def build_rivers_smooth_fc(
    graph: VoronoiGraph,
    rivers: Dict[int, Any],
    map_id: str,
    alpha: float = 0.5,
    segments_per_edge: int = 8,
) -> Dict[str, Any]:
    feats: List[Dict[str, Any]] = []
    for rid, river in rivers.items():
        cells = getattr(river, "cells", [])
        if not cells or len(cells) < 2:
            continue
        waypoints: List[List[float]] = []
        p0 = graph.points[cells[0]]
        waypoints.append([float(p0[0]), float(p0[1])])
        for i in range(len(cells) - 1):
            a, b = cells[i], cells[i + 1]
            mid = _shared_edge_midpoint(graph, a, b)
            if not waypoints or mid != waypoints[-1]:
                waypoints.append(mid)
        p_last = graph.points[cells[-1]]
        last = [float(p_last[0]), float(p_last[1])]
        if waypoints[-1] != last:
            waypoints.append(last)
        smooth = _catmull_rom_spline(waypoints, alpha=alpha, segments=segments_per_edge)
        if len(smooth) < 2:
            continue
        props = {
            "map_id": map_id,
            "river_id": int(rid),
            "discharge": float(getattr(river, "discharge", 0.0)),
            "width": float(getattr(river, "width", 0.0)),
            "length": float(getattr(river, "length", 0.0)),
            "source_distance": float(getattr(river, "source_distance", 0.0)),
            "cells": [int(x) for x in cells],
        }
        feats.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": smooth},
            "properties": props,
        })
    return {"type": "FeatureCollection", "features": feats}
