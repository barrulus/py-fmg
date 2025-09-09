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
from .core.provinces import Province
from .core.routes import Route
from .core.markers import Marker
from .core.military import Regiment
from .core.hydrology import RiverData


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


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
    r, g, b = [max(0, min(255, int(round(c)))) for c in rgb]
    return f"#{r:02x}{g:02x}{b:02x}"


def _mix(c1: str, c2: str, t: float) -> str:
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    return _rgb_to_hex((_lerp(r1, r2, t), _lerp(g1, g2, t), _lerp(b1, b2, t)))


def _hypsometric_color(h: float) -> str:
    """Return a hypsometric tint for a given height (0..100). Water is blue."""
    if h < 20:
        return "#69a7ff"  # ocean blue
    # Normalize land [20..100] -> [0..1]
    t = max(0.0, min(1.0, (h - 20.0) / 80.0))
    # Two-stage ramp: green -> brown -> white
    if t < 0.6:
        t2 = t / 0.6
        return _mix("#d8f0c0", "#8c6b4f", t2)  # light green to brown
    else:
        t2 = (t - 0.6) / 0.4
        return _mix("#8c6b4f", "#ffffff", t2)  # brown to white


def _compute_hillshade(graph: VoronoiGraph, cell_id: int, vertical_exaggeration: float = 3.0,
                        sun_azimuth_deg: float = 315.0, sun_altitude_deg: float = 45.0) -> float:
    """Approximate hillshade per cell using neighbor gradients.

    Returns a value in [0,1]. Higher = brighter (facing the sun).
    """
    p = graph.points[cell_id]
    h = float(graph.heights[cell_id]) if getattr(graph, "heights", None) is not None else 0.0
    gx = 0.0
    gy = 0.0
    # Gradient from neighbors
    for n in graph.cell_neighbors[cell_id]:
        if n < 0 or n >= len(graph.points):
            continue
        q = graph.points[n]
        hn = float(graph.heights[n]) if n < len(graph.heights) else h
        dx = float(q[0] - p[0])
        dy = float(q[1] - p[1])
        dist = math.hypot(dx, dy)
        if dist <= 0:
            continue
        dh = (hn - h) * vertical_exaggeration
        # Contribution along direction
        ux = dx / dist
        uy = dy / dist
        gx += dh * ux
        gy += dh * uy

    # Slope magnitude and aspect
    slope = math.atan(math.hypot(gx, gy))  # radians
    aspect = math.atan2(gy, -gx)  # facing downhill

    az = math.radians(sun_azimuth_deg)
    alt = math.radians(sun_altitude_deg)
    zen = (math.pi / 2.0) - alt
    # Standard hillshade model
    intensity = math.cos(zen) * math.cos(slope) + math.sin(zen) * math.sin(slope) * math.cos(az - aspect)
    return max(0.0, min(1.0, (intensity + 1.0) / 2.0))


def build_topography_fc(graph: VoronoiGraph, map_id: str) -> Dict[str, Any]:
    """Build hypsometric + hillshade topography layer as polygons with precomputed color."""
    features: List[Dict[str, Any]] = []
    n = len(graph.points)
    for i in range(n):
        ring = _cell_polygon(graph, i)
        if not ring:
            continue
        h = float(graph.heights[i]) if getattr(graph, "heights", None) is not None else 0.0
        base = _hypsometric_color(h)
        shade = _compute_hillshade(graph, i)
        # Mix toward black for shade; keep subtle
        mix_t = 0.35 * (0.5 - (shade - 0.5))  # darken for low shade, lighten slightly for high
        color = _mix(base, "#000000", max(0.0, min(1.0, mix_t)))
        props = {
            "map_id": map_id,
            "cell_id": i,
            "height": h,
            "shade": round(float(shade), 3),
            "color": color,
            "layer": "topography",
        }
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": props,
        })
    return {"type": "FeatureCollection", "features": features}


def build_hillshade_fc(graph: VoronoiGraph, map_id: str) -> Dict[str, Any]:
    """Build hillshade-only layer as polygons with per-cell shade value.

    Shade is in [0,1]; consumers can style using fillOpacity or grayscale.
    """
    features: List[Dict[str, Any]] = []
    n = len(graph.points)
    for i in range(n):
        ring = _cell_polygon(graph, i)
        if not ring:
            continue
        shade = _compute_hillshade(graph, i)
        props = {
            "map_id": map_id,
            "cell_id": i,
            "shade": round(float(shade), 3),
            "layer": "hillshade",
        }
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": props,
        })
    return {"type": "FeatureCollection", "features": features}


def export_hillshade_geojson(
    graph: VoronoiGraph,
    out_dir: str | os.PathLike,
    map_id: str,
) -> Path:
    out_dir = Path(out_dir)
    layer_dir = out_dir / "geojson" / map_id
    layer_dir.mkdir(parents=True, exist_ok=True)
    fc = build_hillshade_fc(graph, map_id)
    out_path = layer_dir / "hillshade.geojson"
    out_path.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
    return out_path


def export_topography_geojson(
    graph: VoronoiGraph,
    out_dir: str | os.PathLike,
    map_id: str,
) -> Path:
    out_dir = Path(out_dir)
    layer_dir = out_dir / "geojson" / map_id
    layer_dir.mkdir(parents=True, exist_ok=True)
    fc = build_topography_fc(graph, map_id)
    out_path = layer_dir / "topography.geojson"
    out_path.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
    return out_path


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
        lake_kind = None
        if feature_types and getattr(graph, "feature_ids", None) is not None:
            fid = int(graph.feature_ids[i]) if i < len(graph.feature_ids) else 0
            ftype = feature_types.get(fid, None)
            is_ocean = ftype == "ocean"
            is_lake = ftype == "lake"
            # If lake, try to propagate kind from feature object
            try:
                if is_lake and graph.features and 0 <= fid < len(graph.features):
                    f = graph.features[fid]
                    lake_kind = getattr(f, "kind", None)
            except Exception:
                lake_kind = None
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
        if lake_kind is not None:
            props["lake_kind"] = lake_kind

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
        lake_kind = None
        if feature_types and getattr(graph, "feature_ids", None) is not None:
            fid = int(graph.feature_ids[i]) if i < len(graph.feature_ids) else 0
            ftype = feature_types.get(fid, None)
            is_ocean = ftype == "ocean"
            is_lake = ftype == "lake"
            try:
                if is_lake and graph.features and 0 <= fid < len(graph.features):
                    f = graph.features[fid]
                    lake_kind = getattr(f, "kind", None)
            except Exception:
                lake_kind = None
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
        if lake_kind is not None:
            props["lake_kind"] = lake_kind
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

    # Build a map of confluence anchors: for any internal step (b -> c) along any river,
    # store the midpoint of the shared edge. Tributaries that end at cell b will use this
    # exact point as their last waypoint so lines join perfectly.
    anchor_midpoint: Dict[int, List[float]] = {}
    for r in rivers.values():
        cells = getattr(r, "cells", [])
        for i in range(len(cells) - 1):
            b, c = cells[i], cells[i + 1]
            anchor_midpoint[b] = _shared_edge_midpoint(graph, b, c)
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
        # If this river is a tributary, end at the downstream anchor midpoint for its last cell
        last_id = cells[-1]
        if getattr(river, "parent_id", None) and last_id in anchor_midpoint:
            j = anchor_midpoint[last_id]
            if waypoints[-1] != j:
                waypoints.append(j)
        # For rivers reaching the ocean, extend to mouth cell center
        last_h = int(graph.heights[last_id]) if getattr(graph, "heights", None) is not None and last_id < len(graph.heights) else 100
        if last_h < 20:
            p_last = graph.points[last_id]
            last_pt = [float(p_last[0]), float(p_last[1])]
            if waypoints[-1] != last_pt:
                waypoints.append(last_pt)

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
    # Same anchoring as file export
    anchor_midpoint: Dict[int, List[float]] = {}
    for r in rivers.values():
        cells = getattr(r, "cells", [])
        for i in range(len(cells) - 1):
            b, c = cells[i], cells[i + 1]
            anchor_midpoint[b] = _shared_edge_midpoint(graph, b, c)
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
        # Tributary end at anchor
        last_id = cells[-1]
        if getattr(river, "parent_id", None) and last_id in anchor_midpoint:
            j = anchor_midpoint[last_id]
            if waypoints[-1] != j:
                waypoints.append(j)
        last_h = int(graph.heights[last_id]) if getattr(graph, "heights", None) is not None and last_id < len(graph.heights) else 100
        if last_h < 20:
            p_last = graph.points[last_id]
            last_pt = [float(p_last[0]), float(p_last[1])]
            if waypoints[-1] != last_pt:
                waypoints.append(last_pt)
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

def export_rivers_polygons_geojson(
    graph: VoronoiGraph,
    rivers: Dict[int, RiverData],
    out_dir: str | os.PathLike,
    map_id: str,
) -> Path:
    """Export polygonal rivers as a FeatureCollection of Polygons.

    Prefer polygons precomputed in Hydrology (river.polygon). If missing, fall back
    to buffering a smoothed centerline with constant width.
    """
    out_dir = Path(out_dir)
    layer_dir = out_dir / "geojson" / map_id
    layer_dir.mkdir(parents=True, exist_ok=True)

    feats: List[Dict[str, Any]] = []
    try:
        from shapely.geometry import LineString
        from shapely.validation import make_valid
    except Exception:
        LineString = None  # type: ignore
        make_valid = None  # type: ignore

    for rid, river in rivers.items():
        ring: List[List[float]] = []
        if getattr(river, "polygon", None):
            ring = [[float(x), float(y)] for x, y in river.polygon]  # type: ignore[arg-type]
        else:
            # Fallback: buffer a simple centerline with constant width
            coords: List[List[float]] = []
            for c in getattr(river, "cells", []):
                if 0 <= c < len(graph.points):
                    p = graph.points[c]
                    coords.append([float(p[0]), float(p[1])])
            if len(coords) >= 2 and LineString is not None:
                try:
                    ls = LineString(coords)
                    poly = ls.buffer(max(1.0, float(getattr(river, "width", 2.0))) / 2.0, cap_style=1, join_style=1)
                    poly = make_valid(poly) if make_valid else poly
                    # Build exterior ring
                    ring = [[float(x), float(y)] for x, y in getattr(poly, "exterior", ls).coords]  # type: ignore[attr-defined]
                except Exception:
                    ring = []
        if not ring or len(ring) < 4:
            continue
        props = {
            "map_id": map_id,
            "river_id": int(rid),
            "discharge": float(getattr(river, "discharge", 0.0)),
            "width": float(getattr(river, "width", 0.0)),
            "length": float(getattr(river, "length", 0.0)),
            "source_distance": float(getattr(river, "source_distance", 0.0)),
        }
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": props,
        })

    fc = {"type": "FeatureCollection", "features": feats}
    out_path = layer_dir / "rivers_polygons.geojson"
    out_path.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
    return out_path

def build_rivers_polygons_fc(
    graph: VoronoiGraph,
    rivers: Dict[int, RiverData],
    map_id: str,
) -> Dict[str, Any]:
    feats: List[Dict[str, Any]] = []
    for rid, river in rivers.items():
        ring: List[List[float]] = []
        if getattr(river, "polygon", None):
            ring = [[float(x), float(y)] for x, y in river.polygon]  # type: ignore[arg-type]
        if not ring or len(ring) < 4:
            continue
        props = {
            "map_id": map_id,
            "river_id": int(rid),
            "discharge": float(getattr(river, "discharge", 0.0)),
            "width": float(getattr(river, "width", 0.0)),
            "length": float(getattr(river, "length", 0.0)),
            "source_distance": float(getattr(river, "source_distance", 0.0)),
        }
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": props,
        })
    return {"type": "FeatureCollection", "features": feats}

def export_routes_geojson(
    routes: List[Route],
    out_dir: str | os.PathLike,
    map_id: str,
    filename: str = "routes.geojson",
) -> Path:
    out_dir = Path(out_dir)
    layer_dir = out_dir / "geojson" / map_id
    layer_dir.mkdir(parents=True, exist_ok=True)
    feats: List[Dict[str, Any]] = []
    for r in routes:
        coords = [[float(x), float(y)] for x, y in r.coords]
        props = {
            "map_id": map_id,
            "route_id": int(r.id),
            "kind": r.kind,
            "class": r.cls,
            "start_settlement": int(r.start_settlement),
            "end_settlement": int(r.end_settlement),
            "distance": float(r.distance),
        }
        feats.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords}, "properties": props})
    fc = {"type": "FeatureCollection", "features": feats}
    out_path = layer_dir / filename
    out_path.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
    return out_path

def build_routes_fc(routes: List[Route], map_id: str) -> Dict[str, Any]:
    feats: List[Dict[str, Any]] = []
    for r in routes:
        coords = [[float(x), float(y)] for x, y in r.coords]
        props = {
            "map_id": map_id,
            "route_id": int(r.id),
            "kind": r.kind,
            "class": r.cls,
            "start_settlement": int(r.start_settlement),
            "end_settlement": int(r.end_settlement),
            "distance": float(r.distance),
        }
        feats.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords}, "properties": props})
    return {"type": "FeatureCollection", "features": feats}

def export_markers_geojson(
    markers: List[Marker],
    out_dir: str | os.PathLike,
    map_id: str,
) -> Path:
    out_dir = Path(out_dir)
    layer_dir = out_dir / "geojson" / map_id
    layer_dir.mkdir(parents=True, exist_ok=True)
    feats: List[Dict[str, Any]] = []
    for m in markers:
        props = {
            "map_id": map_id,
            "marker_id": int(m.i),
            "type": m.type,
            "icon": m.icon,
            "cell": int(m.cell),
            "name": m.name,
            "legend": m.legend,
            "dx": m.dx,
            "dy": m.dy,
            "px": m.px,
        }
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [float(m.x), float(m.y)]}, "properties": props})
    fc = {"type": "FeatureCollection", "features": feats}
    out_path = layer_dir / "markers.geojson"
    out_path.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
    return out_path

def build_markers_fc(markers: List[Marker], map_id: str) -> Dict[str, Any]:
    feats: List[Dict[str, Any]] = []
    for m in markers:
        props = {
            "map_id": map_id,
            "marker_id": int(m.i),
            "type": m.type,
            "icon": m.icon,
            "cell": int(m.cell),
            "name": m.name,
            "legend": m.legend,
        }
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [float(m.x), float(m.y)]}, "properties": props})
    return {"type": "FeatureCollection", "features": feats}

def export_regiments_geojson(
    regiments_by_state: Dict[int, List[Regiment]],
    out_dir: str | os.PathLike,
    map_id: str,
) -> Path:
    out_dir = Path(out_dir)
    layer_dir = out_dir / "geojson" / map_id
    layer_dir.mkdir(parents=True, exist_ok=True)
    feats: List[Dict[str, Any]] = []
    for sid, regs in regiments_by_state.items():
        for r in regs:
            props = {
                "map_id": map_id,
                "state_id": int(sid),
                "regiment_id": int(r.i),
                "name": r.name,
                "icon": r.icon,
                "naval": int(r.n),
                "total": int(r.a),
                "cell": int(r.cell),
                "units": r.u,
            }
            feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [float(r.x), float(r.y)]}, "properties": props})
    fc = {"type": "FeatureCollection", "features": feats}
    out_path = layer_dir / "regiments.geojson"
    out_path.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
    return out_path

def build_regiments_fc(regiments_by_state: Dict[int, List[Regiment]], map_id: str) -> Dict[str, Any]:
    feats: List[Dict[str, Any]] = []
    for sid, regs in regiments_by_state.items():
        for r in regs:
            props = {
                "map_id": map_id,
                "state_id": int(sid),
                "regiment_id": int(r.i),
                "name": r.name,
                "icon": r.icon,
                "naval": int(r.n),
                "total": int(r.a),
                "cell": int(r.cell),
                "units": r.u,
            }
            feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [float(r.x), float(r.y)]}, "properties": props})
    return {"type": "FeatureCollection", "features": feats}
def build_provinces_fc(
    graph: VoronoiGraph,
    provinces: Dict[int, Province],
    cell_province: np.ndarray,
    map_id: str,
) -> Dict[str, Any]:
    """Build a provinces FeatureCollection as MultiPolygon per province.

    Simpler and faster approach: represent each province as MultiPolygon of cell polygons
    without dissolving. For cartography and queries this is acceptable and deterministic.
    """
    feats: List[Dict[str, Any]] = []
    for pid, prov in provinces.items():
        if pid == 0 or prov is None:
            continue
        polys: List[List[List[float]]] = []
        # use assigned cells mapping to collect rings
        for i in range(len(graph.points)):
            if int(cell_province[i]) != pid:
                continue
            ring = _cell_polygon(graph, i)
            if ring:
                polys.append(ring)
        if not polys:
            continue
        # Ensure a distinct color per province when not provided
        def _prov_color(pid: int, fallback: str) -> str:
            palette = [
                "#5b8ff9", "#61d9a8", "#65789b", "#f6bd16", "#7262fd",
                "#78d3f8", "#9661bc", "#f6903d", "#008685", "#f08bb4",
            ]
            c = getattr(prov, "color", None)
            if not c or c == "#cccccc":
                return palette[pid % len(palette)]
            return c

        props = {
            "map_id": map_id,
            "province_id": int(pid),
            "state_id": int(getattr(prov, "state_id", 0)),
            "name": getattr(prov, "name", f"Province {pid}"),
            "full_name": getattr(prov, "full_name", getattr(prov, "name", "")),
            "color": _prov_color(int(pid), "#cccccc"),
        }
        feats.append({
            "type": "Feature",
            "geometry": {"type": "MultiPolygon", "coordinates": [[poly] for poly in polys]},
            "properties": props,
        })
    return {"type": "FeatureCollection", "features": feats}


def export_provinces_geojson(
    graph: VoronoiGraph,
    provinces: Dict[int, Province],
    cell_province: np.ndarray,
    out_dir: str | os.PathLike,
    map_id: str,
) -> Path:
    out_dir = Path(out_dir)
    layer_dir = out_dir / "geojson" / map_id
    layer_dir.mkdir(parents=True, exist_ok=True)
    fc = build_provinces_fc(graph, provinces, cell_province, map_id)
    out_path = layer_dir / "provinces.geojson"
    out_path.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
    return out_path
