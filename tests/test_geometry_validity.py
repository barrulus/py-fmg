import numpy as np
from shapely.geometry import Polygon

from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.heightmap_generator import HeightmapConfig, HeightmapGenerator
from py_fmg.core.features import Features
from py_fmg.core.cell_packing import regraph
from py_fmg.core.climate import Climate
from py_fmg.core.hydrology import Hydrology, HydrologyOptions
from py_fmg.exporter_geojson import build_cells_fc


def test_cell_polygons_valid():
    # Smaller graph for quick validity check
    cfg = GridConfig(width=300, height=220, cells_desired=1200)
    g = generate_voronoi_graph(cfg, seed="geom-seed", apply_relaxation=True)
    hm_cfg = HeightmapConfig(width=300, height=220, cells_x=g.cells_x, cells_y=g.cells_y, cells_desired=1200, spacing=g.spacing)
    hm = HeightmapGenerator(hm_cfg, g, seed="geom-seed")
    g.heights = hm.from_template("continents", seed="geom-seed")
    feat = Features(g, seed="geom-seed"); feat.markup_grid()
    g = regraph(g); feat.markup_pack(g)

    fc = build_cells_fc(g, map_id="geom-test")
    # Validate polygons
    count = 0
    for f in fc["features"]:
        ring = f["geometry"]["coordinates"][0]
        if len(ring) < 4:
            continue
        poly = Polygon(ring)
        assert poly.is_valid, "Invalid Voronoi cell polygon"
        count += 1
        if count >= 500:  # sample cap for speed
            break


def test_river_polygons_valid():
    cfg = GridConfig(width=300, height=220, cells_desired=1200)
    g = generate_voronoi_graph(cfg, seed="geom-seed2", apply_relaxation=True)
    hm_cfg = HeightmapConfig(width=300, height=220, cells_x=g.cells_x, cells_y=g.cells_y, cells_desired=1200, spacing=g.spacing)
    hm = HeightmapGenerator(hm_cfg, g, seed="geom-seed2")
    g.heights = hm.from_template("continents", seed="geom-seed2")
    feat = Features(g, seed="geom-seed2"); feat.markup_grid()
    g = regraph(g); feat.markup_pack(g)
    clim = Climate(g); clim.calculate_temperatures(); clim.generate_precipitation()
    hyd = Hydrology(g, feat, clim, options=HydrologyOptions(topo_guided_flow=False, snap_to_coast_steps=0))
    rivers = hyd.generate_rivers()

    # Ensure any generated river polygons are valid
    polys = 0
    for r in rivers.values():
        ring = getattr(r, "polygon", None)
        if not ring or len(ring) < 4:
            continue
        poly = Polygon(ring)
        assert poly.is_valid, "Invalid river polygon"
        polys += 1
        if polys >= 50:
            break
