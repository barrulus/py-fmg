"""
Lakes subsystem — a faithful port of FMG lakes.js behavior adapted to py-fmg.

Implements:
- detect_close_lakes: mark lakes that are in deep depressions as closed (no outlets)
- define_climate_data: compute per-lake flux, temperature, evaporation, and outCell
- get_height: compute lake surface elevation (slightly below lowest shore)

This module operates on the Features detected by py_fmg.core.features.Features.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import math
import numpy as np


LAKE_ELEVATION_DELTA = 0.1


def _ensure_shoreline(graph, lake_feature) -> List[int]:
    """Compute and cache lake shoreline (land cells adjacent to this lake)."""
    if getattr(lake_feature, "shoreline", None):
        return lake_feature.shoreline  # type: ignore[return-value]

    shoreline: List[int] = []
    fid = int(lake_feature.id)
    for i in range(len(graph.points)):
        if graph.feature_ids[i] != fid:
            continue
        # i is a lake cell, record neighboring land cells as shoreline
        for n in graph.cell_neighbors[i]:
            if graph.heights[n] >= 20 and n not in shoreline:
                shoreline.append(n)
    lake_feature.shoreline = shoreline
    return shoreline


def get_height(graph, lake_feature) -> float:
    """Return lake surface elevation as min shore height minus delta."""
    shoreline = _ensure_shoreline(graph, lake_feature)
    if shoreline:
        min_shore_h = float(np.min([graph.heights[c] for c in shoreline]))
    else:
        # Fallback: use first cell neighbors
        lc = int(lake_feature.first_cell)
        neigh = graph.cell_neighbors[lc]
        min_shore_h = float(np.min([graph.heights[c] for c in neigh])) if neigh else 20.0
    return round(min_shore_h - LAKE_ELEVATION_DELTA, 2)


def detect_close_lakes(graph, elevation_limit: float = 22.0) -> None:
    """Mark lakes as closed if they are in deep depressions per FMG logic.

    A lake is closed if from its lowest shoreline cell there is no path to ocean or
    to another lake with lower surface under the max elevation threshold.
    """
    n = len(graph.points)
    heights = graph.heights

    for feature in getattr(graph, "features", []) or []:
        if not feature or feature.type != "lake":
            continue
        if getattr(feature, "height", None) is None:
            feature.height = get_height(graph, feature)

        # Reset state
        if hasattr(feature, "closed"):
            delattr(feature, "closed")

        max_elev = float(feature.height) + float(elevation_limit)
        if max_elev > 99:
            feature.closed = False
            continue

        shoreline = _ensure_shoreline(graph, feature)
        if not shoreline:
            feature.closed = True
            continue

        # pick lowest shoreline cell
        lowest = min(shoreline, key=lambda c: heights[c])
        queue = [lowest]
        checked = set([lowest])
        is_deep = True

        while queue and is_deep:
            cell = queue.pop()
            for nb in graph.cell_neighbors[cell]:
                if nb in checked:
                    continue
                if float(heights[nb]) >= max_elev:
                    continue

                if heights[nb] < 20:
                    # Check feature type of neighboring water
                    fid = int(graph.feature_ids[nb]) if nb < len(graph.feature_ids) else 0
                    if 0 < fid < len(graph.features):
                        nfeat = graph.features[fid]
                        if nfeat and (nfeat.type == "ocean" or float(feature.height) > float(getattr(nfeat, "height", 0))):
                            is_deep = False
                            break
                checked.add(nb)
                queue.append(nb)

        feature.closed = is_deep


def define_climate_data(graph, climate, elevation_limit: float = 22.0) -> Dict[int, List]:
    """Compute per-lake flux, temp, evaporation, and outCell.

    Returns mapping of outlet shore cell -> list of lake features draining through it.
    """
    # Ensure "closed" flags are set
    detect_close_lakes(graph, elevation_limit=elevation_limit)

    lake_out_cells: Dict[int, List] = {}

    for feature in getattr(graph, "features", []) or []:
        if not feature or feature.type != "lake":
            continue
        # Ensure shoreline and height
        shoreline = _ensure_shoreline(graph, feature)
        if getattr(feature, "height", None) is None:
            feature.height = get_height(graph, feature)

        # Flux = sum of precipitation over shoreline grid cells
        flux = 0.0
        if shoreline and getattr(climate, "precipitation", None) is not None:
            for c in shoreline:
                gid = c
                if hasattr(graph, "grid_indices") and graph.grid_indices is not None:
                    gid = graph.grid_indices[c]
                if isinstance(climate.precipitation, dict):
                    flux += float(climate.precipitation.get(gid, 0.0))
                else:
                    if gid < len(climate.precipitation):
                        flux += float(climate.precipitation[gid])
        feature.flux = flux

        # Lake temperature = mean of shoreline temps for larger lakes, else single
        temp = 0.0
        if hasattr(climate, "temperatures") and climate.temperatures is not None:
            if getattr(feature, "cells", 0) and int(feature.cells) >= 6 and shoreline:
                vals = []
                for c in shoreline:
                    gid = c
                    if hasattr(graph, "grid_indices") and graph.grid_indices is not None:
                        gid = graph.grid_indices[c]
                    if gid < len(climate.temperatures):
                        vals.append(float(climate.temperatures[gid]))
                temp = round(float(np.mean(vals)) if vals else 0.0, 1)
            else:
                gid = feature.first_cell
                if hasattr(graph, "grid_indices") and graph.grid_indices is not None:
                    gid = graph.grid_indices[gid]
                if gid < len(climate.temperatures):
                    temp = round(float(climate.temperatures[gid]), 1)
        feature.temp = temp

        # Evaporation based on Penman-like formula from FMG
        # height_m = ((h - 18) ** height_exponent)
        height_exponent = getattr(getattr(climate, "options", None), "height_exponent", 1.5)
        height_m = ((float(feature.height) - 18.0) ** float(height_exponent))
        evaporation = ((700.0 * (temp + 0.006 * height_m)) / 50.0 + 75.0) / (80.0 - temp if (80.0 - temp) != 0 else 1.0)
        # Scale by lake size (cells count)
        cells_count = int(getattr(feature, "cells", 0))
        feature.evaporation = float(round(evaporation * max(1, cells_count)))

        # Outlet cell is lowest shoreline cell unless lake is closed
        if getattr(feature, "closed", False):
            # Classify dry/sinkhole if evaporation overwhelms flux
            try:
                if feature.flux <= 0 or feature.flux < 0.5 * feature.evaporation:
                    # Very low inflow and high evaporation
                    feature.kind = "dry" if feature.flux <= 0 else "sinkhole"
                else:
                    feature.kind = getattr(feature, "kind", None) or "closed"
            except Exception:
                feature.kind = getattr(feature, "kind", None) or "closed"
            continue
        if shoreline:
            out_cell = min(shoreline, key=lambda c: graph.heights[c])
            feature.outCell = int(out_cell)
            lake_out_cells.setdefault(int(out_cell), []).append(feature)

        # Lake type classification (parity-leaning heuristics)
        # - frozen: cold temperatures around shoreline
        # - salt: near sea level with high evaporation vs inflow
        # - lava: very high elevation and warm (proxy for volcanic proximity)
        # - default: "fresh"
        try:
            t = float(getattr(feature, "temp", 0.0))
            min_shore_h = float(min(graph.heights[s] for s in shoreline)) if shoreline else 30.0
            is_frozen = t <= -5.0
            is_lava = (min_shore_h >= 80.0 and t >= 15.0)
            is_salt = (min_shore_h < 30.0 and feature.evaporation > max(1.5 * feature.flux, 5.0))
            if is_frozen:
                feature.kind = "frozen"
            elif is_lava:
                feature.kind = "lava"
            elif is_salt:
                feature.kind = "salt"
            else:
                feature.kind = getattr(feature, "kind", None) or "fresh"
        except Exception:
            # Best-effort default
            feature.kind = getattr(feature, "kind", None) or "fresh"

    return lake_out_cells
