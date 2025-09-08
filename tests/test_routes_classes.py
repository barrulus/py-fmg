from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.heightmap_generator import HeightmapConfig, HeightmapGenerator
from py_fmg.core.features import Features
from py_fmg.core.cell_packing import regraph
from py_fmg.core.climate import Climate
from py_fmg.core.biomes import BiomeClassifier
from py_fmg.core.cultures import CultureGenerator
from py_fmg.core.settlements import Settlements, SettlementOptions
from py_fmg.core.name_generator import NameGenerator
from py_fmg.core.routes import RoutesGenerator


def _pipeline(seed: str = "routes1"):
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
    biome = BiomeClassifier()
    cultures, cell_cultures, cell_population, cell_suitability = CultureGenerator(pg, feat, biome).generate()
    pg.cell_population = cell_population
    pg.cell_suitability = cell_suitability
    settlements, states = Settlements(pg, feat, type("C", (), {"cultures": cultures, "cell_cultures": cell_cultures})(), biome, NameGenerator(), options=SettlementOptions()).generate()
    return pg, settlements


def test_routes_have_hierarchy_and_capitals_connected():
    pg, settlements = _pipeline()
    routes = RoutesGenerator(pg, settlements, BiomeClassifier()).build_land_routes()
    assert routes, "expected some land routes"
    # Must contain at least one highway edge
    assert any(r.cls == "highway" for r in routes)
    # Distribution sanity: trails should not dominate
    total = len(routes)
    trails = sum(1 for r in routes if r.cls == "trail")
    assert trails < max(1, int(0.6 * total))

