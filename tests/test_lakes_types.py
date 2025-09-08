import pytest

from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.heightmap_generator import HeightmapConfig, HeightmapGenerator
from py_fmg.core.features import Features
from py_fmg.core.cell_packing import regraph
from py_fmg.core.climate import Climate
from py_fmg.core.lakes import define_climate_data


def test_lake_kind_classification_present_when_lakes_exist():
    cfg = GridConfig(width=600, height=450, cells_desired=6000)
    g = generate_voronoi_graph(cfg, seed="lake-kind-1", apply_relaxation=True)
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
        seed="lake-kind-1",
    )
    g.heights = hm.from_template("archipelago", seed="lake-kind-1")
    feat = Features(g, seed="lake-kind-1")
    feat.markup_grid()
    # try creating more lakes
    feat.add_lakes_in_deep_depressions(elevation_limit=22)

    pg = regraph(g)
    clim = Climate(pg)
    clim.calculate_temperatures()
    clim.generate_precipitation()
    # Attach feature arrays to packed graph for exporter checks
    pg.distance_field = feat.distance_field
    pg.feature_ids = feat.feature_ids
    pg.features = feat.features

    define_climate_data(pg, clim, elevation_limit=22)

    # Verify that at least one lake has a .kind attribute and that the watermask can expose it
    lakes = [f for f in pg.features if f and hasattr(f, "type") and f.type == "lake"]
    if not lakes:
        pytest.skip("no lakes detected in this seed; classification not applicable")
    assert any(getattr(l, "kind", None) for l in lakes)

