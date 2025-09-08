import numpy as np
from pathlib import Path
from PIL import Image

from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.heightmap_generator import HeightmapConfig, HeightmapGenerator


def build_graph(width=400, height=300, cells=2000, seed="parity-seed"):
    cfg = GridConfig(width=width, height=height, cells_desired=cells)
    graph = generate_voronoi_graph(cfg, seed=seed, apply_relaxation=True)
    return graph


def build_hm_config(graph):
    return HeightmapConfig(
        width=int(graph.graph_width),
        height=int(graph.graph_height),
        cells_x=int(graph.cells_x),
        cells_y=int(graph.cells_y),
        cells_desired=int(graph.cells_desired),
        spacing=float(graph.spacing),
    )


def test_from_template_deterministic_and_bounds():
    graph = build_graph()
    hm_cfg = build_hm_config(graph)
    seed = "template-seed"

    hm1 = HeightmapGenerator(hm_cfg, graph, seed=seed)
    h1 = hm1.from_template("continents", seed=seed)

    # Determinism with same seed and graph
    hm2 = HeightmapGenerator(hm_cfg, graph, seed=seed)
    h2 = hm2.from_template("continents", seed=seed)
    assert np.array_equal(h1, h2), "from_template must be deterministic for same seed and graph"

    # Basic bounds
    assert h1.dtype == np.uint8
    assert len(h1) == len(graph.points)
    assert int(h1.min()) >= 0 and int(h1.max()) <= 100

    # Land fraction should be sensible (broad band to avoid flakiness)
    land_frac = float(np.mean(h1 >= 20))
    assert 0.1 <= land_frac <= 0.95


def test_from_template_varies_with_seed():
    graph = build_graph()
    hm_cfg = build_hm_config(graph)

    hm1 = HeightmapGenerator(hm_cfg, graph, seed="seed-a").from_template("continents", seed="seed-a")
    hm2 = HeightmapGenerator(hm_cfg, graph, seed="seed-b").from_template("continents", seed="seed-b")

    # Not guaranteed to differ at every cell, but very unlikely to be identical arrays
    assert not np.array_equal(hm1, hm2), "Different seeds should produce different heightmaps"


def test_from_precreated_matches_power_mapping():
    # Uses FMG-bundled precreated image 'europe.png'
    graph = build_graph()
    hm_cfg = build_hm_config(graph)

    gen = HeightmapGenerator(hm_cfg, graph)
    h = gen.from_precreated("europe")

    # Manual mapping following FMG logic
    img_path = Path("Fantasy-Map-Generator/heightmaps/europe.png")
    assert img_path.exists(), "Expected bundled FMG precreated PNG missing"
    with Image.open(img_path) as img:
        img2 = img.convert("RGB").resize((graph.cells_x, graph.cells_y), Image.BILINEAR)
        arr = np.asarray(img2, dtype=np.uint8)
        red = arr[:, :, 0].astype(np.float32) / 255.0
    powered = np.where(red < 0.2, red, 0.2 + (red - 0.2) ** 0.8)
    expected = np.floor(np.clip(powered * 100.0, 0.0, 100.0)).astype(np.uint8).reshape(-1)

    assert np.array_equal(h, expected), "from_precreated must match FMG power-curve mapping"

