import json
from pathlib import Path

import numpy as np

from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.heightmap_generator import HeightmapConfig, HeightmapGenerator
from py_fmg.core.features import Features
from py_fmg.core.cell_packing import regraph
from py_fmg.core.climate import Climate
from py_fmg.core.biomes import BiomeClassifier
from py_fmg.core.cultures import CultureGenerator
from py_fmg.core.settlements import Settlements, SettlementOptions
from py_fmg.core.provinces import ProvincesGenerator, ProvinceOptions
from py_fmg.core.routes import RoutesGenerator
from py_fmg.core.markers import MarkersGenerator
from py_fmg.core.hydrology import Hydrology, HydrologyOptions
from py_fmg.core.military import MilitaryGenerator
from py_fmg.core.religions import ReligionGenerator
from py_fmg.exporter_map import export_fmg_map


def build_full_pipeline(seed: str = "rt1"):
    cfg = GridConfig(width=700, height=520, cells_desired=7000)
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
    try:
        feat.add_lakes_in_deep_depressions()
        feat.open_near_sea_lakes()
    except Exception:
        pass
    pg = regraph(g)
    feat.markup_pack(pg)
    # Attach fields
    pg.distance_field = feat.distance_field
    pg.feature_ids = feat.feature_ids
    pg.features = feat.features

    clim = Climate(pg)
    clim.calculate_temperatures()
    clim.generate_precipitation()
    pg.temperatures = clim.temperatures
    pg.precipitation = clim.precipitation

    hd = Hydrology(pg, feat, clim, options=HydrologyOptions(topo_guided_flow=False, snap_to_coast_steps=0))
    rivers = hd.generate_rivers()
    pg.river_ids = hd.river_ids
    pg.flux = hd.flux

    biome = BiomeClassifier()
    cultures, cell_cultures, cell_population, cell_suitability = CultureGenerator(pg, feat, biome).generate()
    pg.cell_population = cell_population
    pg.cell_suitability = cell_suitability

    religions, cell_religions = ReligionGenerator(pg, cultures, cell_cultures, {}, {}, name_generator=None).generate()

    from py_fmg.core.name_generator import NameGenerator
    settlements_eng = Settlements(pg, feat, type("C", (), {"cultures": cultures, "cell_cultures": cell_cultures})(), biome, NameGenerator(), options=SettlementOptions(), cell_religions=cell_religions)
    settlements, states = settlements_eng.generate()
    pg.cell_state = settlements_eng.cell_state

    provinces, cell_provinces = ProvincesGenerator(pg, states, settlements, options=ProvinceOptions()).generate()

    routes = RoutesGenerator(pg, settlements, biome)
    land_routes = routes.build_land_routes()
    sea_routes = routes.build_sea_routes()

    markers = MarkersGenerator(pg, settlements, rivers).generate()

    regiments_by_state = MilitaryGenerator(pg, settlements, states).generate()

    return {
        "graph": pg,
        "feat": feat,
        "biomes": biome,
        "cultures": cultures,
        "cell_cultures": cell_cultures,
        "religions": religions,
        "cell_religions": cell_religions,
        "settlements": settlements,
        "states": states,
        "provinces": provinces,
        "cell_provinces": cell_provinces,
        "land_routes": land_routes,
        "sea_routes": sea_routes,
        "markers": markers,
        "regiments_by_state": regiments_by_state,
        "rivers_features": [
            {
                'i': int(rid),
                'cells': [int(c) for c in r.cells],
                'width': float(getattr(r, 'width', 1.0)),
                'length': float(getattr(r, 'length', 0.0)),
                'discharge': float(getattr(r, 'discharge', 0.0)),
                'name': f"River {int(rid)}",
                'type': 1,
            }
            for rid, r in rivers.items()
        ],
    }


def test_roundtrip_loader_simulation(tmp_path: Path):
    ctx = build_full_pipeline(seed="rt2")

    out_path = tmp_path / "roundtrip.map"
    path = export_fmg_map(
        out_path,
        ctx["graph"],
        map_name="Roundtrip",
        seed="rt2",
        biomes=None,
        rivers_json=ctx["rivers_features"],
        features_list=ctx["graph"].features,
        minimal=False,
        settlements=ctx["settlements"],
        states=ctx["states"],
        provinces=ctx["provinces"],
        cell_provinces=ctx["cell_provinces"],
        cultures=ctx["cultures"],
        cell_cultures=ctx["cell_cultures"],
        religions=ctx["religions"],
        cell_religions=ctx["cell_religions"],
        land_routes=ctx["land_routes"],
        sea_routes=ctx["sea_routes"],
        markers=ctx["markers"],
        regiments_by_state=ctx["regiments_by_state"],
    )

    assert path.exists(), "map file not written"

    content = path.read_text(encoding="utf-8")
    lines = content.split("\r\n")
    assert len(lines) > 20

    # Parse known JSON lines from our exporter ordering
    grid_general = json.loads(lines[6])
    features = json.loads(lines[12])
    cultures = json.loads(lines[13])
    states = json.loads(lines[14])
    burgs = json.loads(lines[15])
    # pack arrays
    biomes_ids = [int(x) for x in lines[16].split(",")]
    burg_arr = [int(x) for x in lines[17].split(",")]
    # regiments JSON at 23 (skip)
    state_arr = [int(x) for x in lines[25].split(",")]
    province_arr = [int(x) for x in lines[27].split(",")]
    markers = json.loads(lines[35])
    routes = json.loads(lines[37])

    n = len(grid_general["points"])
    assert len(biomes_ids) == n
    assert len(state_arr) == n
    assert len(province_arr) == n

    # Sanity: at least one item in each entity group
    assert isinstance(features, list) and len(features) >= 1
    assert isinstance(burgs, list) and len(burgs) >= 1
    assert isinstance(states, list) and len(states) >= 1
    assert isinstance(markers, list)
    assert isinstance(routes, list)
