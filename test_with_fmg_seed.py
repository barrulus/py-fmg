#!/usr/bin/env python3
"""
Test our heightmap generation with the exact seed FMG used.
"""

import json
import numpy as np
from py_fmg.core import GridConfig, generate_voronoi_graph, HeightmapGenerator, HeightmapConfig
from py_fmg.config import get_template
from py_fmg.utils.random import set_random_seed


def main():
    # Use the exact seed from FMG
    seed = "854906727"
    
    # Generate with same parameters as FMG
    # Note: FMG shows 123x82 grid, which suggests different dimensions
    # Let's calculate the actual dimensions
    cells_x, cells_y = 123, 82
    spacing = 3.0  # Typical spacing
    width = int(cells_x * spacing)
    height = int(cells_y * spacing)
    
    print(f"Testing with FMG parameters:")
    print(f"  Seed: {seed}")
    print(f"  Grid: {cells_x}x{cells_y}")
    print(f"  Estimated dimensions: {width}x{height}")
    print(f"  Total cells: {cells_x * cells_y}")
    
    # Generate Voronoi graph
    config = GridConfig(width=width, height=height, cells_desired=10086)
    graph = generate_voronoi_graph(config, seed=seed)
    
    print(f"\nActual graph generated:")
    print(f"  Grid: {graph.cells_x}x{graph.cells_y}")
    print(f"  Total cells: {len(graph.points)}")
    
    # Create heightmap
    heightmap_config = HeightmapConfig(
        width=width,
        height=height,
        cells_x=graph.cells_x,
        cells_y=graph.cells_y,
        cells_desired=10086,
        spacing=graph.spacing
    )
    
    generator = HeightmapGenerator(heightmap_config, graph)
    
    # Generate with lowIsland template
    template = get_template("lowIsland")
    heights = generator.from_template(template, seed=seed)
    
    print(f"\nPython heightmap results:")
    print(f"  Min: {heights.min()}")
    print(f"  Max: {heights.max()}")
    print(f"  Mean: {heights.mean():.1f}")
    print(f"  Land cells: {np.sum(heights >= 20)} ({np.sum(heights >= 20) / len(heights) * 100:.1f}%)")
    print(f"  First 20: {heights[:20].tolist()}")
    
    # Load FMG data for comparison
    with open('heights.json') as f:
        fmg_data = json.load(f)
    
    fmg_heights = fmg_data['heights']['first20']
    
    print(f"\nFMG first 20: {fmg_heights}")
    print(f"\nComparison of first 20 values:")
    print("Index | Python | FMG | Diff")
    print("------|--------|-----|-----")
    for i in range(20):
        diff = heights[i] - fmg_heights[i]
        print(f"{i:5d} | {heights[i]:6d} | {fmg_heights[i]:3d} | {diff:+4d}")
    
    # Check if we need different template
    # FMG might have used a different template
    print(f"\n\nTrying to detect which template FMG used...")
    
    # The max height of 100 and high land percentage suggests a different template
    templates = ["lowIsland", "highIsland", "continents", "archipelago", "mediterranean"]
    
    for template_name in templates:
        print(f"\nTrying {template_name}...")
        generator2 = HeightmapGenerator(heightmap_config, graph)
        template = get_template(template_name)
        heights2 = generator2.from_template(template, seed=seed)
        
        land_pct = np.sum(heights2 >= 20) / len(heights2) * 100
        print(f"  Land: {land_pct:.1f}%, Max: {heights2.max()}")


if __name__ == "__main__":
    main()