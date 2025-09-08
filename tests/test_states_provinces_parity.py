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
from py_fmg.core.name_generator import NameGenerator


def build_pipeline(width=400, height=300, cells=2000, seed="state-province-seed"):
    # Graph
    gcfg = GridConfig(width=width, height=height, cells_desired=cells)
    graph = generate_voronoi_graph(gcfg, seed=seed, apply_relaxation=True)
    # Heights
    hcfg = HeightmapConfig(width=int(width), height=int(height), cells_x=graph.cells_x, cells_y=graph.cells_y, cells_desired=cells, spacing=graph.spacing)
    hm = HeightmapGenerator(hcfg, graph, seed=seed)
    graph.heights = hm.from_template("continents", seed=seed)
    # Features and regraph
    feat = Features(graph, seed=seed)
    feat.markup_grid()
    graph = regraph(graph)
    feat.markup_pack(graph)
    # Climate for river preference
    clim = Climate(graph)
    clim.calculate_temperatures(); clim.generate_precipitation()
    # Biomes + cultures
    bc = BiomeClassifier()
    graph.biomes = type("B", (), {"cell_biomes": bc.classify_biomes(clim.temperatures, clim.precipitation, graph.heights, neighbors=graph.cell_neighbors)})
    cult = CultureGenerator(graph, feat, bc)
    cultures, cell_cultures, cell_population, cell_suitability = cult.generate()
    graph.cell_population = cell_population
    graph.cell_suitability = cell_suitability
    # Settlements & states
    opts = SettlementOptions(states_number=10, burgs_number=200)
    settlements_engine = Settlements(graph, feat, type("CWrap", (), {"cultures": cultures, "cell_cultures": cell_cultures})(), bc, name_generator=NameGenerator(), options=opts)
    settlements, states = settlements_engine.generate()
    graph.cell_state = settlements_engine.cell_state
    return graph, feat, settlements, states, bc


def test_non_naval_states_do_not_cross_ocean_landmasses():
    graph, feat, settlements, states, bc = build_pipeline()
    # For each non-Naval state, land cells should be on same feature id as center (allow lakes)
    for sid, st in states.items():
        if not getattr(st, 'id', 0):
            continue
        stype = getattr(st, 'type', 'Generic')
        if stype == 'Naval':
            continue
        center = int(getattr(st, 'center_cell', 0))
        if center >= len(feat.feature_ids):
            continue
        center_fid = int(feat.feature_ids[center])
        for i, state_id in enumerate(graph.cell_state):
            if state_id != sid:
                continue
            if graph.heights[i] < 20:
                continue
            cell_fid = int(feat.feature_ids[i]) if i < len(feat.feature_ids) else center_fid
            if cell_fid != center_fid:
                # allow mismatches if either feature is a lake
                f1 = feat.features[center_fid] if center_fid < len(feat.features) else None
                f2 = feat.features[cell_fid] if cell_fid < len(feat.features) else None
                if not (getattr(f1, 'type', None) == 'lake' or getattr(f2, 'type', None) == 'lake'):
                    raise AssertionError(f"State {sid} ({stype}) crosses ocean to cell {i}")


def test_provinces_cover_state_land_cells():
    graph, feat, settlements, states, bc = build_pipeline()
    from py_fmg.core.provinces import ProvincesGenerator, ProvinceOptions
    prov_gen = ProvincesGenerator(graph, states, settlements, options=ProvinceOptions(provinces_ratio=50))
    provinces, cell_prov = prov_gen.generate()
    # For each land cell with a state, there should be a province assignment >0
    for i in range(len(graph.points)):
        if graph.heights[i] < 20:
            continue
        if graph.cell_state[i] > 0:
            assert cell_prov[i] > 0, f"Land cell {i} in state {graph.cell_state[i]} has no province"

