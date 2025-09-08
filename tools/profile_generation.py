#!/usr/bin/env python3
"""
Simple profiling harness for end-to-end map generation time.

Usage:
  python tools/profile_generation.py --width 1000 --height 800 --cells 10000 --seed demo
"""
import argparse
import time

from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.heightmap_generator import HeightmapConfig, HeightmapGenerator
from py_fmg.core.features import Features
from py_fmg.core.cell_packing import regraph
from py_fmg.core.climate import Climate
from py_fmg.core.biomes import BiomeClassifier
from py_fmg.core.cultures import CultureGenerator
from py_fmg.core.hydrology import Hydrology, HydrologyOptions
from py_fmg.core.settlements import Settlements, SettlementOptions
from py_fmg.core.name_generator import NameGenerator
from py_fmg.core.provinces import ProvincesGenerator, ProvinceOptions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=float, default=1000)
    ap.add_argument("--height", type=float, default=800)
    ap.add_argument("--cells", type=int, default=10000)
    ap.add_argument("--seed", type=str, default="benchmark-seed")
    args = ap.parse_args()

    t0 = time.time()
    cfg = GridConfig(width=args.width, height=args.height, cells_desired=args.cells)
    g = generate_voronoi_graph(cfg, seed=args.seed, apply_relaxation=True)
    hm_cfg = HeightmapConfig(width=int(args.width), height=int(args.height), cells_x=g.cells_x, cells_y=g.cells_y, cells_desired=args.cells, spacing=g.spacing)
    hm = HeightmapGenerator(hm_cfg, g, seed=args.seed)
    g.heights = hm.from_template("continents", seed=args.seed)
    t_heights = time.time()

    feat = Features(g, seed=args.seed); feat.markup_grid()
    g = regraph(g); feat.markup_pack(g)
    t_regraph = time.time()

    clim = Climate(g); clim.calculate_temperatures(); clim.generate_precipitation()
    bc = BiomeClassifier()
    cult = CultureGenerator(g, feat, bc)
    cultures, cell_cultures, cell_population, cell_suitability = cult.generate()
    g.cell_population = cell_population
    g.cell_suitability = cell_suitability
    t_climate = time.time()

    hyd = Hydrology(g, feat, clim, options=HydrologyOptions(topo_guided_flow=False, snap_to_coast_steps=0))
    rivers = hyd.generate_rivers()
    t_rivers = time.time()

    st = Settlements(g, feat, type("CWrap", (), {"cultures": cultures, "cell_cultures": cell_cultures})(), bc, name_generator=NameGenerator(), options=SettlementOptions(states_number=30, burgs_number=1000))
    settlements, states = st.generate()
    g.cell_state = st.cell_state
    prov = ProvincesGenerator(g, states, settlements, options=ProvinceOptions())
    provinces, cell_prov = prov.generate()
    t_end = time.time()

    print(f"Heights: {t_heights - t0:.2f}s, reGraph: {t_regraph - t_heights:.2f}s, climate/culture: {t_climate - t_regraph:.2f}s, rivers: {t_rivers - t_climate:.2f}s, settlements+provinces: {t_end - t_rivers:.2f}s")
    print(f"Total: {t_end - t0:.2f}s for {args.cells} cells")


if __name__ == "__main__":
    main()

