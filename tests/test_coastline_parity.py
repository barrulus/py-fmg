import numpy as np

from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.heightmap_generator import HeightmapConfig, HeightmapGenerator
from py_fmg.core.features import Features
from py_fmg.core.cell_packing import regraph


def _build_packed_with_features(seed: str = "parity1"):
    cfg = GridConfig(width=800, height=600, cells_desired=8000)
    g = generate_voronoi_graph(cfg, seed=seed, apply_relaxation=True)
    hm = HeightmapGenerator(
        HeightmapConfig(
            width=int(cfg.width),
            height=int(cfg.height),
            cells_x=g.cells_x,
            cells_y=g.cells_y,
            cells_desired=cfg.cells_desired,
            spacing=g.spacing,
        ),
        g,
        seed=seed,
    )
    g.heights = hm.from_template("continents", seed=seed)
    feat = Features(g, seed=seed)
    feat.markup_grid()
    pg = regraph(g)
    # apply packed markup
    feat.markup_pack(pg)
    # expose fields on packed graph for checks
    pg.distance_field = feat.distance_field
    pg.feature_ids = feat.feature_ids
    pg.features = feat.features
    return pg


def test_coastline_counts_consistent_across_seeds():
    seeds = ["parity1", "parity2", "parity3"]
    counts = []
    for s in seeds:
        pg = _build_packed_with_features(seed=s)
        df = pg.distance_field
        land_coast = int(np.sum(df == 1))
        water_coast = int(np.sum(df == -1))
        # Record total coastline cells (both sides)
        counts.append(land_coast + water_coast)
        # Invariants per seed
        assert land_coast > 0 and water_coast > 0
        # No inland-water mislabeled as coast: each WATER_COAST must neighbor land
        bad = 0
        for i in range(len(df)):
            if int(df[i]) != -1:
                continue
            if not any(pg.heights[n] >= 20 for n in pg.cell_neighbors[i]):
                bad += 1
        assert bad == 0

    # Coastline totals should not vary wildly between seeds
    arr = np.array(counts, dtype=float)
    mean = float(arr.mean())
    # standard deviation within 15% of mean gives a stability check
    if mean > 0:
        assert float(arr.std()) <= 0.15 * mean

