"""
Minimal CLI to generate a Voronoi grid and export cells as GeoJSON
with a Leaflet preview. Uses argparse to avoid extra deps.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.heightmap_generator import HeightmapConfig, HeightmapGenerator
from py_fmg.core.features import Features
from py_fmg.core.hydrology import Hydrology, HydrologyOptions
from py_fmg.exporter_geojson import (
    export_cells_geojson,
    export_coastlines_geojson,
    export_watermask_geojson,
    export_climate_geojson,
    export_biomes_geojson,
    export_rivers_geojson,
    export_rivers_smooth_geojson,
)
from py_fmg.core.climate import Climate, ClimateOptions
from py_fmg.core.biomes import BiomeClassifier
from py_fmg.core.cultures import CultureGenerator
from py_fmg.core.settlements import Settlements, SettlementOptions
from py_fmg.core.name_generator import NameGenerator
from py_fmg.preview.leaflet_template import write_inline_leaflet, write_inline_leaflet_multi
from py_fmg.exporter_map import export_fmg_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Voronoi cells and export GeoJSON")
    parser.add_argument("--width", type=float, default=1000, help="Map width")
    parser.add_argument("--height", type=float, default=800, help="Map height")
    parser.add_argument("--cells", type=int, default=10000, help="Target number of cells")
    parser.add_argument("--seed", type=str, default=None, help="Random seed")
    parser.add_argument("--out", type=str, default="out", help="Output directory root")
    parser.add_argument("--template", type=str, default="continents", help="Heightmap template name")
    parser.add_argument("--target-land", type=float, default=None, help="Target land fraction [0-1] (auto sea-level shift)")
    parser.add_argument("--preview-width", type=int, default=1200, help="Preview width in px")
    parser.add_argument("--preview-height", type=int, default=800, help="Preview height in px")
    parser.add_argument("--no-relax", action="store_true", help="Disable Lloyd relaxation")
    parser.add_argument("--export-map", type=str, default=None, help="Write FMG .map file to this path")
    parser.add_argument("--export-map-minimal", action="store_true", help="Export a minimal .map with safe defaults for FMG import")
    # Hydrology tuning
    parser.add_argument("--min-river-flux", type=float, default=30.0, help="Minimum flux to form a visible river")
    parser.add_argument("--precip-mult", type=float, default=1.0, help="Multiplier for precipitation in hydrology")
    parser.add_argument("--snap-to-coast-steps", type=int, default=3, help="Steps to snap river mouths to coast")
    # Settlement tuning
    parser.add_argument("--states-number", type=int, default=30, help="Target number of states (capitals)")
    parser.add_argument("--burgs-number", type=int, default=1000, help="Target number of towns (1000 = auto)")
    parser.add_argument("--town-spacing-base", type=int, default=150, help="Base divisor for town spacing")
    parser.add_argument("--town-spacing-power", type=float, default=0.7, help="Power adjustment for town spacing")
    parser.add_argument("--urbanization-rate", type=float, default=0.1, help="Urbanization rate (0..1) used in settlement sizing")
    args = parser.parse_args()

    map_id = (args.seed or datetime.utcnow().strftime("%Y%m%d%H%M%S"))[:16]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    config = GridConfig(width=args.width, height=args.height, cells_desired=args.cells)
    graph = generate_voronoi_graph(config, seed=args.seed, apply_relaxation=not args.no_relax)

    # Generate heightmap and attach to graph
    hm_cfg = HeightmapConfig(
        width=int(args.width),
        height=int(args.height),
        cells_x=int(graph.cells_x),
        cells_y=int(graph.cells_y),
        cells_desired=int(args.cells),
        spacing=float(graph.spacing),
    )
    hm = HeightmapGenerator(hm_cfg, graph, seed=args.seed)
    graph.heights = hm.from_template(args.template, seed=args.seed)

    # Optional: auto sea-level shift to hit target land fraction
    if args.target_land is not None:
        import numpy as np
        heights = graph.heights.astype(float)
        target = max(0.0, min(1.0, args.target_land))
        # Land is height >= 20; shift heights so that (1 - target) quantile maps to 20
        q = float(np.percentile(heights, (1.0 - target) * 100.0))
        delta = 20.0 - q
        if abs(delta) > 0.1:
            heights = np.clip(heights + delta, 0, 100)
            graph.heights = heights.astype(np.uint8)

    # Detect features (coastlines, oceans, lakes) for is_coast flags
    feat = Features(graph, seed=args.seed)
    feat.markup_grid()
    # Persist fields on graph for exporter convenience
    graph.distance_field = feat.distance_field
    graph.feature_ids = feat.feature_ids
    graph.features = feat.features

    # Climate: temperatures + precipitation
    climate = Climate(graph, options=ClimateOptions())
    climate.calculate_temperatures()
    climate.generate_precipitation()
    graph.temperatures = climate.temperatures
    graph.precipitation = climate.precipitation

    # Biomes
    biome_classifier = BiomeClassifier()
    cell_biomes = biome_classifier.classify_biomes(
        graph.temperatures,
        graph.precipitation,
        graph.heights,
        neighbors=graph.cell_neighbors,
    )

    geo_path = export_cells_geojson(graph, out_root, map_id)
    print(f"Wrote cells: {geo_path}")

    coast_path = export_coastlines_geojson(graph, out_root, map_id)
    print(f"Wrote coastlines: {coast_path}")
    watermask_path = export_watermask_geojson(graph, out_root, map_id)
    print(f"Wrote watermask: {watermask_path}")
    climate_path = export_climate_geojson(graph, graph.temperatures, graph.precipitation, out_root, map_id)
    print(f"Wrote climate: {climate_path}")
    biomes_path = export_biomes_geojson(graph, cell_biomes, biome_classifier, out_root, map_id)
    # Rivers (after climate and features)
    hyd = Hydrology(
        graph,
        feat,
        climate,
        options=HydrologyOptions(
            min_river_flux=args.min_river_flux,
            precip_multiplier=args.precip_mult,
            snap_to_coast_steps=args.snap_to_coast_steps,
        ),
    )
    rivers = hyd.generate_rivers()
    rivers_path = export_rivers_geojson(graph, rivers, out_root, map_id)
    rivers_smooth_path = export_rivers_smooth_geojson(graph, rivers, out_root, map_id)
    print(f"Wrote rivers: {rivers_path}")
    print(f"Wrote rivers (smooth): {rivers_smooth_path}")
    # Attach hydro arrays to graph for downstream modules
    graph.river_ids = hyd.river_ids
    graph.confluences = hyd.confluences
    graph.flux = hyd.flux
    print(f"Wrote biomes: {biomes_path}")

    # Cultures
    culture_gen = CultureGenerator(graph, feat, biome_classifier)
    cultures, cell_cultures, cell_population, cell_suitability = culture_gen.generate()
    # Expose population/suitability on graph for downstream modules
    graph.cell_population = cell_population
    graph.cell_suitability = cell_suitability
    from py_fmg.exporter_geojson import export_cell_cultures_geojson, export_cultures_points_geojson
    cell_cultures_path = export_cell_cultures_geojson(graph, cell_cultures, cultures, out_root, map_id)
    cultures_points_path = export_cultures_points_geojson(graph, cultures, out_root, map_id)
    print(f"Wrote cultures: {cultures_points_path}")

    # Burgs (settlements) - minimal placement using settlement system
    # Wrap cultures to match expected interface in Settlements
    class CulturesWrapper:
        def __init__(self, cultures_dict, cell_cults):
            self.cultures = cultures_dict
            self.cell_cultures = cell_cults

    cultures_wrapped = CulturesWrapper(cultures, cell_cultures)

    settlement_opts = SettlementOptions(
        states_number=args.states_number,
        manors_number=args.burgs_number,
        town_spacing_base=args.town_spacing_base,
        town_spacing_power=args.town_spacing_power,
        urbanization_rate=args.urbanization_rate,
    )

    settlements_engine = Settlements(
        graph,
        feat,
        cultures_wrapped,
        biome_classifier,
        name_generator=NameGenerator(),
        options=settlement_opts,
    )
    settlements, states = settlements_engine.generate()
    from py_fmg.exporter_geojson import export_burgs_points_geojson
    burgs_path = export_burgs_points_geojson(settlements, out_root, map_id)
    print(f"Wrote burgs: {burgs_path}")

    # Also write a small manifest with basic metadata
    manifest_dir = out_root / "geojson" / map_id
    manifest = {
        "map_id": map_id,
        "width": args.width,
        "height": args.height,
        "cells_desired": args.cells,
        "seed": args.seed,
        "artifacts": {"cells": str(geo_path)},
    }
    manifest["artifacts"]["coastlines"] = str(coast_path)
    manifest["artifacts"]["watermask"] = str(watermask_path)
    manifest["artifacts"]["climate"] = str(climate_path)
    manifest["artifacts"]["biomes"] = str(biomes_path)
    manifest["artifacts"]["rivers"] = str(rivers_path)
    manifest["artifacts"]["rivers_smooth"] = str(rivers_smooth_path)
    manifest["artifacts"]["cultures_cells"] = str(cell_cultures_path)
    manifest["artifacts"]["cultures_points"] = str(cultures_points_path)
    manifest["artifacts"]["burgs"] = str(burgs_path)
    (manifest_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {manifest_dir / 'manifest.json'}")

    # Write a simple Leaflet preview with inlined GeoJSON
    try:
        fc = json.loads(geo_path.read_text(encoding="utf-8"))
        preview_dir = out_root / "preview"
        html_path = preview_dir / f"{map_id}_cells.html"
        write_inline_leaflet(fc, html_path, title=f"Cells Preview — {map_id}", width=args.preview_width, height=args.preview_height)
        print(f"Wrote preview: {html_path}")

        # Multi-layer preview
        with open(coast_path, 'r', encoding='utf-8') as f2:
            coast_fc = json.load(f2)
        html2 = preview_dir / f"{map_id}_layers.html"
        write_inline_leaflet_multi(
            {
                "cells": fc,
                "climate": json.loads(Path(climate_path).read_text(encoding='utf-8')),
                "biomes": json.loads(Path(biomes_path).read_text(encoding='utf-8')),
                "cultures_cells": json.loads(Path(cell_cultures_path).read_text(encoding='utf-8')),
                "watermask": json.loads(Path(watermask_path).read_text(encoding='utf-8')),
                "rivers": json.loads(Path(rivers_path).read_text(encoding='utf-8')),
                "burgs": json.loads(Path(burgs_path).read_text(encoding='utf-8')),
                "rivers_smooth": json.loads(Path(rivers_smooth_path).read_text(encoding='utf-8')),
                "coastlines": coast_fc
            },
            html2,
            title=f"Map Preview — {map_id}",
            width=args.preview_width,
            height=args.preview_height,
        )
        print(f"Wrote preview: {html2}")
    except Exception as e:
        print(f"Could not write preview HTML: {e}")

    # Optional: FMG .map export
    if args.export_map:
        try:
            # Build rivers JSON if available
            rivers_json = None
            try:
                rivers_json = json.loads(Path(rivers_path).read_text(encoding="utf-8"))['features']
                # Convert features back to FMG-like river objects with id & cells
                rivers_json = [
                    {
                        'i': int(f['properties'].get('river_id', idx + 1)),
                        'cells': f['properties'].get('cells', []),
                        'width': f['properties'].get('width', 1.0),
                        'length': f['properties'].get('length', 0.0),
                        'name': f"River {int(f['properties'].get('river_id', idx + 1))}",
                        'type': 1
                    }
                    for idx, f in enumerate(rivers_json)
                ]
            except Exception:
                rivers_json = []

            export_fmg_map(
                args.export_map,
                graph,
                map_name=f"Map {map_id}",
                seed=args.seed or map_id,
                map_id=None,
                temperatures=graph.temperatures,
                precipitation=graph.precipitation,
                biomes=cell_biomes,
                rivers_json=rivers_json,
                features_list=feat.features,
                minimal=args.export_map_minimal,
            )
            print(f"Wrote FMG map: {args.export_map}")
        except Exception as e:
            print(f"FMG map export failed: {e}")


if __name__ == "__main__":
    main()
