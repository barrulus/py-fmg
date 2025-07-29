#!/usr/bin/env python3
"""
Direct comparison test between Python heightmap generation and FMG export data.
This script helps identify at what stage the discrepancy occurs.
"""

import json
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from py_fmg.core.voronoi_graph import generate_voronoi_graph, GridConfig
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig
from py_fmg.utils.random import set_random_seed


def analyze_fmg_export():
    """Analyze the FMG export file to understand its structure and data."""
    
    # Load the reference data
    reference_path = Path("tests/Mateau Full 2025-07-27-14-53.json")
    with open(reference_path, "r") as f:
        fmg_data = json.load(f)
    
    print("=" * 80)
    print("FMG Export Data Analysis")
    print("=" * 80)
    
    # Basic info
    info = fmg_data["info"]
    print(f"\nMap Info:")
    print(f"  Name: {info.get('name', 'unknown')}")
    print(f"  Seed: {info['seed']}")
    print(f"  Dimensions: {info['width']}x{info['height']}")
    print(f"  Template: {info.get('template', 'unknown')}")
    
    # Grid data
    grid = fmg_data["grid"]
    print(f"\nGrid Data:")
    print(f"  Cells: {len(grid['cells'])}")
    print(f"  Points: {len(grid['points'])}")
    print(f"  Cells desired: {grid['cellsDesired']}")
    print(f"  Spacing: {grid['spacing']}")
    
    # Check if grid cells have heights
    grid_heights = [cell["h"] for cell in grid["cells"]]
    print(f"\nGrid Cell Heights:")
    print(f"  Min: {min(grid_heights)}")
    print(f"  Max: {max(grid_heights)}")
    print(f"  Mean: {np.mean(grid_heights):.1f}")
    print(f"  Land cells (h>=20): {sum(1 for h in grid_heights if h >= 20)} ({sum(1 for h in grid_heights if h >= 20)/len(grid_heights)*100:.1f}%)")
    
    # Pack data (after reGraph)
    pack = fmg_data["pack"]
    pack_heights = [cell["h"] for cell in pack["cells"]]
    print(f"\nPack Data (after reGraph):")
    print(f"  Cells: {len(pack['cells'])}")
    print(f"  Min height: {min(pack_heights)}")
    print(f"  Max height: {max(pack_heights)}")
    print(f"  Mean height: {np.mean(pack_heights):.1f}")
    print(f"  Land cells (h>=20): {sum(1 for h in pack_heights if h >= 20)} ({sum(1 for h in pack_heights if h >= 20)/len(pack_heights)*100:.1f}%)")
    
    # Check features
    if "features" in pack:
        features = pack["features"]
        print(f"\nFeatures: {len(features)} total")
        # Features might be a list of indices or objects
        if features and isinstance(features[0], dict):
            for i, feature in enumerate(features[:5]):  # First 5 features
                print(f"  Feature {i}: type={feature.get('type', 'unknown')}, cells={len(feature.get('cells', []))}")
    
    return fmg_data


def test_heightmap_generation_stages():
    """Test heightmap generation at different stages to find where differences occur."""
    
    print("\n" + "=" * 80)
    print("Python Heightmap Generation Test")
    print("=" * 80)
    
    # Use the same parameters as FMG
    width, height = 300, 300
    cells_desired = 10000
    grid_seed = "1234567"  # From console analysis
    map_seed = "651658815"  # From FMG export
    template_name = "lowIsland"
    
    # Stage 1: Generate Voronoi graph
    print(f"\nStage 1: Generating Voronoi graph with seed '{grid_seed}'...")
    config = GridConfig(width=width, height=height, cells_desired=cells_desired)
    graph = generate_voronoi_graph(config, seed=grid_seed)
    
    print(f"  Grid cells: {len(graph.points)}")
    print(f"  Cells X/Y: {graph.cells_x}x{graph.cells_y}")
    print(f"  Spacing: {graph.spacing}")
    
    # Check pre-allocated heights
    print(f"\nPre-allocated heights (before heightmap generation):")
    print(f"  Min: {np.min(graph.heights)}")
    print(f"  Max: {np.max(graph.heights)}")
    print(f"  All zeros: {np.all(graph.heights == 0)}")
    
    # Stage 2: Generate heightmap
    print(f"\nStage 2: Generating heightmap with template '{template_name}' and seed '{map_seed}'...")
    heightmap_config = HeightmapConfig(
        width=width,
        height=height,
        cells_x=graph.cells_x,
        cells_y=graph.cells_y,
        cells_desired=cells_desired,
        spacing=graph.spacing,
    )
    
    generator = HeightmapGenerator(heightmap_config, graph)
    heights = generator.from_template(template_name, seed=map_seed)
    
    print(f"\nGenerated heights:")
    print(f"  Shape: {heights.shape}")
    print(f"  Min: {np.min(heights)}")
    print(f"  Max: {np.max(heights)}")
    print(f"  Mean: {np.mean(heights):.1f}")
    print(f"  Land cells (h>=20): {np.sum(heights >= 20)} ({np.sum(heights >= 20)/len(heights)*100:.1f}%)")
    
    # Test without template (just hills)
    print(f"\nTest with single hill (no template):")
    graph2 = generate_voronoi_graph(config, seed=grid_seed)
    generator2 = HeightmapGenerator(heightmap_config, graph2)
    generator2.add_hill(1, 50, "40-60", "40-60")
    
    single_hill_heights = generator2.heights
    print(f"  Min: {np.min(single_hill_heights)}")
    print(f"  Max: {np.max(single_hill_heights)}")
    print(f"  Non-zero cells: {np.sum(single_hill_heights > 0)} ({np.sum(single_hill_heights > 0)/len(single_hill_heights)*100:.1f}%)")
    
    return heights


def compare_template_effects():
    """Compare the effects of the lowIsland template steps."""
    
    print("\n" + "=" * 80)
    print("LowIsland Template Step-by-Step Analysis")
    print("=" * 80)
    
    # lowIsland template steps:
    template_steps = [
        "Hill 1 90-99 60-80 45-55",
        "Hill 1-2 20-30 10-30 10-90",
        "Smooth 2 0 0 0",
        "Hill 6-7 25-35 20-70 30-70",
        "Range 1 40-50 45-55 45-55",
        "Trough 2-3 20-30 15-85 20-30",
        "Trough 2-3 20-30 15-85 70-80",
        "Hill 1.5 10-15 5-15 20-80",
        "Hill 1 10-15 85-95 70-80",
        "Pit 5-7 15-25 15-85 20-80",
        "Multiply 0.4 20-100 0 0",
        "Mask 4 0 0 0",
    ]
    
    # Generate initial graph
    config = GridConfig(width=300, height=300, cells_desired=10000)
    graph = generate_voronoi_graph(config, seed="1234567")
    
    heightmap_config = HeightmapConfig(
        width=300,
        height=300,
        cells_x=graph.cells_x,
        cells_y=graph.cells_y,
        cells_desired=10000,
        spacing=graph.spacing,
    )
    
    generator = HeightmapGenerator(heightmap_config, graph)
    set_random_seed("651658815")
    
    print("\nApplying template steps:")
    
    # Track heights after each step
    for i, step in enumerate(template_steps):
        parts = step.strip().split()
        command = parts[0]
        args = parts[1:]
        
        # Apply the command
        if command == "Hill":
            generator.add_hill(*args)
        elif command == "Smooth":
            generator.smooth(*args)
        elif command == "Range":
            generator.add_range(*args)
        elif command == "Trough":
            generator.add_trough(*args)
        elif command == "Pit":
            generator.add_pit(*args)
        elif command == "Multiply":
            generator.modify(args[1], multiply=float(args[0]))
        elif command == "Mask":
            generator.mask(float(args[0]))
        
        # Analyze current state
        heights = np.floor(generator.heights).astype(np.uint8)
        land_pct = np.sum(heights >= 20) / len(heights) * 100
        
        print(f"  Step {i+1}: {step}")
        print(f"    Heights: min={np.min(heights)}, max={np.max(heights)}, mean={np.mean(heights):.1f}")
        print(f"    Land: {land_pct:.1f}%")
        
        # Critical step: Multiply 0.4 20-100
        if command == "Multiply" and float(args[0]) == 0.4:
            print(f"    >>> This step reduces land heights by 60%!")


def main():
    # Analyze FMG export
    fmg_data = analyze_fmg_export()
    
    # Test Python generation
    python_heights = test_heightmap_generation_stages()
    
    # Compare template effects
    compare_template_effects()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nThe discrepancy appears to be that the FMG export contains heights")
    print("at a different stage of map generation, possibly after additional")
    print("processing steps beyond the initial heightmap template application.")
    print("\nThe key difference: FMG export shows 31.9% land vs Python's 93.2%")
    print("This suggests FMG may have applied additional water/erosion steps.")


if __name__ == "__main__":
    main()