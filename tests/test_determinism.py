import numpy as np

from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.heightmap_generator import HeightmapConfig, HeightmapGenerator
from py_fmg.core.features import Features
from py_fmg.core.cell_packing import regraph
from py_fmg.core.climate import Climate
from py_fmg.core.biomes import BiomeClassifier
from py_fmg.core.cultures import CultureGenerator
from py_fmg.core.hydrology import Hydrology, HydrologyOptions
from py_fmg.core.settlements import Settlements, SettlementOptions
from py_fmg.core.provinces import ProvincesGenerator, ProvinceOptions
from py_fmg.core.name_generator import NameGenerator


def build_graph(seed: str, width=400, height=300, cells=2000):
    cfg = GridConfig(width=width, height=height, cells_desired=cells)
    graph = generate_voronoi_graph(cfg, seed=seed, apply_relaxation=True)
    hm_cfg = HeightmapConfig(width=int(width), height=int(height), cells_x=graph.cells_x, cells_y=graph.cells_y, cells_desired=cells, spacing=graph.spacing)
    hm = HeightmapGenerator(hm_cfg, graph, seed=seed)
    graph.heights = hm.from_template("continents", seed=seed)
    feat = Features(graph, seed=seed)
    feat.markup_grid()
    graph = regraph(graph)
    feat.markup_pack(graph)
    clim = Climate(graph)
    clim.calculate_temperatures(); clim.generate_precipitation()
    bc = BiomeClassifier()
    graph.biomes = type("B", (), {"cell_biomes": bc.classify_biomes(clim.temperatures, clim.precipitation, graph.heights, neighbors=graph.cell_neighbors)})
    hyd = Hydrology(graph, feat, clim, options=HydrologyOptions(topo_guided_flow=False, snap_to_coast_steps=0))
    rivers = hyd.generate_rivers()
    cult = CultureGenerator(graph, feat, bc)
    cultures, cell_cultures, cell_population, cell_suitability = cult.generate()
    graph.cell_population = cell_population
    graph.cell_suitability = cell_suitability
    sett = Settlements(graph, feat, type("CWrap", (), {"cultures": cultures, "cell_cultures": cell_cultures})(), bc, name_generator=NameGenerator(), options=SettlementOptions(states_number=8, burgs_number=150))
    settlements, states = sett.generate()
    graph.cell_state = sett.cell_state
    prov = ProvincesGenerator(graph, states, settlements, options=ProvinceOptions(provinces_ratio=50))
    provinces, cell_prov = prov.generate()
    return {
        "graph": graph,
        "rivers": rivers,
        "settlements": settlements,
        "states": states,
        "provinces": provinces,
    }


def test_seed_determinism_counts():
    seed = "determinism-seed"
    a = build_graph(seed)
    b = build_graph(seed)

    # Number of rivers, settlements, states, provinces must match
    assert len(a["rivers"]) == len(b["rivers"])  # river count
    assert len(a["settlements"]) == len(b["settlements"])  # burgs count
    assert len(a["states"]) == len(b["states"])  # includes neutrals; stable across runs
    assert len(a["provinces"]) == len(b["provinces"])  # province count


def test_seed_determinism_first_entities():
    seed = "determinism-seed-2"
    a = build_graph(seed)
    b = build_graph(seed)

    # Compare first few sorted entity IDs / keys deterministically
    def head3_keys(d):
        return sorted(d.keys())[:3]

    assert head3_keys(a["rivers"]) == head3_keys(b["rivers"])  # first river ids
    assert head3_keys(a["settlements"]) == head3_keys(b["settlements"])  # first settlement ids
    assert head3_keys(a["states"]) == head3_keys(b["states"])  # first state ids
    assert head3_keys(a["provinces"]) == head3_keys(b["provinces"])  # first province ids

