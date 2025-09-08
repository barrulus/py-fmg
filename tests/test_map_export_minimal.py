import json
from pathlib import Path

from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.heightmap_generator import HeightmapConfig, HeightmapGenerator
from py_fmg.core.features import Features
from py_fmg.core.cell_packing import regraph
from py_fmg.exporter_map import export_fmg_map


def test_minimal_map_export_structure(tmp_path: Path):
    cfg = GridConfig(width=500, height=400, cells_desired=4000)
    g = generate_voronoi_graph(cfg, seed="mapexp1", apply_relaxation=True)
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
        seed="mapexp1",
    )
    g.heights = hm.from_template("continents", seed="mapexp1")
    feat = Features(g, seed="mapexp1")
    feat.markup_grid()
    pg = regraph(g)
    feat.markup_pack(pg)
    pg.distance_field = feat.distance_field
    pg.feature_ids = feat.feature_ids
    pg.features = feat.features

    out = tmp_path / "test.map"
    path = export_fmg_map(
        out,
        pg,
        map_name="TestMap",
        seed="mapexp1",
        biomes=None,
        rivers_json=[],
        features_list=pg.features,
        minimal=True,
    )
    assert path.exists()
    # Parse CRLF chunks and spot-check critical blocks
    content = path.read_text(encoding="utf-8")
    lines = content.split("\r\n")
    assert len(lines) > 10
    # biomes dict line present (index 3)
    assert "|" in lines[3]
    # grid.general JSON (index 6)
    grid = json.loads(lines[6])
    assert "cellsX" in grid and "points" in grid
    # features array JSON present later
    features = json.loads(lines[10])
    assert isinstance(features, list)
    # Array lengths align with points
    n = len(grid["points"])
    heights = [int(x) for x in lines[7].split(",")]
    assert len(heights) == n

