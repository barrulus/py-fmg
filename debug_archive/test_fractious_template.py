#!/usr/bin/env python3
"""
Test heightmap generation with the fractious template and FMG's seed.
"""

import json
import numpy as np
from py_fmg.core import GridConfig, generate_voronoi_graph, HeightmapGenerator, HeightmapConfig
from py_fmg.config import get_template
from py_fmg.utils.random import set_random_seed


def main():
    # FMG used seed 854906727 for heightmap generation  
    # BUT the grid was from seed 651658815!
    grid_seed = "651658815"
    heightmap_seed = "854906727"
    
    print("Testing FMG seed behavior:")
    print(f"  Grid seed: {grid_seed}")
    print(f"  Heightmap seed: {heightmap_seed}")
    print(f"  Template: fractious")
    
    # Generate Voronoi graph with GRID seed
    config = GridConfig(width=1200, height=799, cells_desired=10000)
    graph = generate_voronoi_graph(config, seed=grid_seed)
    
    print(f"\nGrid generated:")
    print(f"  Cells: {len(graph.points)} (grid: {graph.cells_x}x{graph.cells_y})")
    
    # Create heightmap with HEIGHTMAP seed
    heightmap_config = HeightmapConfig(
        width=1200,
        height=799,
        cells_x=graph.cells_x,
        cells_y=graph.cells_y,
        cells_desired=10000,
        spacing=graph.spacing
    )
    
    generator = HeightmapGenerator(heightmap_config, graph)
    
    # Generate with fractious template
    template = get_template("fractious")
    heights = generator.from_template(template, seed=heightmap_seed)
    
    print(f"\nPython heightmap (fractious template):")
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
    print("Index | Python | FMG | Match")
    print("------|--------|-----|------")
    
    matches = 0
    for i in range(20):
        match = heights[i] == fmg_heights[i]
        matches += match
        print(f"{i:5d} | {heights[i]:6d} | {fmg_heights[i]:3d} | {'YES' if match else 'NO'}")
    
    print(f"\nMatches: {matches}/20 ({matches/20*100:.0f}%)")
    
    # Check overall match percentage
    fmg_all = np.array(fmg_data['heights']['data'])
    exact_matches = np.sum(heights == fmg_all)
    print(f"\nOverall exact matches: {exact_matches}/{len(heights)} ({exact_matches/len(heights)*100:.1f}%)")


if __name__ == "__main__":
    main()