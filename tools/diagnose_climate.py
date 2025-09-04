from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.heightmap_generator import HeightmapConfig, HeightmapGenerator
from py_fmg.core.features import Features
from py_fmg.core.climate import Climate, ClimateOptions
from py_fmg.core.hydrology import Hydrology, HydrologyOptions


@dataclass
class Diagnostics:
    lat_bins: List[Tuple[float, float]]
    temp_avg: List[float]
    precip_avg: List[float]
    flux_avg: List[float]
    river_cells_pct: List[float]


def lat_from_y(y: float, height: float) -> float:
    # Map y in [0, height] to latitude +90 .. -90
    return 90.0 - (y / height) * 180.0


def run(width=800, height=600, cells=10000, seed: str | None = "diag1") -> Diagnostics:
    # 1) Graph and heights
    gconf = GridConfig(width=width, height=height, cells_desired=cells)
    graph = generate_voronoi_graph(gconf, seed=seed)
    hconf = HeightmapConfig(
        width=int(width), height=int(height), cells_x=graph.cells_x, cells_y=graph.cells_y,
        cells_desired=cells, spacing=float(graph.spacing)
    )
    hm = HeightmapGenerator(hconf, graph, seed=seed)
    graph.heights = hm.from_template("continents", seed=seed)

    # 2) Features (distance/coasts/lakes)
    feat = Features(graph, seed=seed)
    feat.markup_grid()
    graph.distance_field = feat.distance_field
    graph.feature_ids = feat.feature_ids
    graph.features = feat.features

    # 3) Climate
    climate = Climate(graph, options=ClimateOptions())
    climate.calculate_temperatures()
    climate.generate_precipitation()
    graph.temperatures = climate.temperatures
    graph.precipitation = climate.precipitation

    # 4) Hydrology (rivers + flux)
    hyd = Hydrology(graph, feat, climate, options=HydrologyOptions(min_river_flux=30.0))
    hyd.generate_rivers()
    graph.river_ids = hyd.river_ids
    graph.flux = hyd.flux

    # 5) Aggregate by latitude bins (10° bands)
    n = len(graph.points)
    lats = np.array([lat_from_y(graph.points[i][1], height) for i in range(n)])
    temps = climate.temperatures.astype(float)
    prec = climate.precipitation.astype(float)
    flux = hyd.flux.astype(float)
    has_river = (hyd.river_ids > 0)

    # Consider only land for river/flux summaries to avoid ocean skew
    is_land = (graph.heights >= 20)

    bins = list(range(-90, 91, 10))  # -90..90 step 10
    lat_bins: List[Tuple[float, float]] = []
    temp_avg: List[float] = []
    precip_avg: List[float] = []
    flux_avg: List[float] = []
    river_cells_pct: List[float] = []

    for b0, b1 in zip(bins[:-1], bins[1:]):
        mask = (lats >= b0) & (lats < b1)
        if not mask.any():
            lat_bins.append((b0, b1))
            temp_avg.append(float("nan"))
            precip_avg.append(float("nan"))
            flux_avg.append(float("nan"))
            river_cells_pct.append(float("nan"))
            continue
        temp_avg.append(float(np.mean(temps[mask])))
        precip_avg.append(float(np.mean(prec[mask])))
        land_mask = mask & is_land
        if land_mask.any():
            flux_avg.append(float(np.mean(flux[land_mask])))
            total_land = int(np.sum(land_mask))
            river_land = int(np.sum(has_river[land_mask]))
            pct = 100.0 * river_land / max(total_land, 1)
            river_cells_pct.append(pct)
        else:
            flux_avg.append(float("nan"))
            river_cells_pct.append(float("nan"))
        lat_bins.append((b0, b1))

    return Diagnostics(lat_bins, temp_avg, precip_avg, flux_avg, river_cells_pct)


if __name__ == "__main__":
    d = run()
    print("Latitude band, tempC, precip, avg_flux(land), river_cells_%")
    for (b0, b1), t, p, f, r in zip(d.lat_bins, d.temp_avg, d.precip_avg, d.flux_avg, d.river_cells_pct):
        print(f"[{b0:>3},{b1:>3})  {t:>6.2f}  {p:>6.2f}  {f:>8.2f}  {r:>6.2f}")

