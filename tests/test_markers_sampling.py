from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.heightmap_generator import HeightmapConfig, HeightmapGenerator
from py_fmg.core.features import Features
from py_fmg.core.cell_packing import regraph
from py_fmg.core.climate import Climate
from py_fmg.core.biomes import BiomeClassifier
from py_fmg.core.cultures import CultureGenerator
from py_fmg.core.settlements import Settlements, SettlementOptions
from py_fmg.core.name_generator import NameGenerator
from py_fmg.core.hydrology import Hydrology, HydrologyOptions
from py_fmg.core.markers import MarkersGenerator


def _pipeline(seed: str = "markers1"):
    cfg = GridConfig(width=700, height=500, cells_desired=7000)
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
    feat.markup_pack(pg)
    pg.distance_field = feat.distance_field
    pg.feature_ids = feat.feature_ids
    pg.features = feat.features
    clim = Climate(pg)
    clim.calculate_temperatures()
    clim.generate_precipitation()
    pg.temperatures = clim.temperatures
    pg.precipitation = clim.precipitation
    hd = Hydrology(pg, feat, clim, options=HydrologyOptions(topo_guided_flow=False, snap_to_coast_steps=0))
    hd.generate_rivers()
    pg.river_ids = hd.river_ids
    pg.flux = hd.flux
    biome = BiomeClassifier()
    cultures, cell_cultures, cell_population, cell_suitability = CultureGenerator(pg, feat, biome).generate()
    pg.cell_population = cell_population
    settlements, _ = Settlements(pg, feat, type("C", (), {"cultures": cultures, "cell_cultures": cell_cultures})(), biome, NameGenerator(), options=SettlementOptions()).generate()
    return pg, settlements


def test_lighthouses_on_coast_and_nonzero():
    pg, settlements = _pipeline()
    markers = MarkersGenerator(pg, settlements, seed="m1").generate()
    lighthouses = [m for m in markers if m.type == "lighthouses"]
    # Should have at least one if coasts exist
    assert len(lighthouses) >= 1
    # Each lighthouse should be coastal (neighbor water)
    for m in lighthouses:
        cell = int(m.cell)
        assert any(pg.heights[n] < 20 for n in pg.cell_neighbors[cell])

