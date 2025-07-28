#!/usr/bin/env python3
"""
Test heightmap generation with the original map parameters.
"""

import json
import numpy as np
from py_fmg.core import GridConfig, generate_voronoi_graph, HeightmapGenerator, HeightmapConfig
from py_fmg.config import get_template
from py_fmg.utils.random import set_random_seed


def main():
    # Load original map data
    with open('tests/Mateau Full 2025-07-27-14-53.json') as f:
        fmg_data = json.load(f)
    
    seed = fmg_data['info']['seed']
    width = fmg_data['info']['width']
    height = fmg_data['info']['height']
    
    print(f"Testing with original map parameters:")
    print(f"  Seed: {seed}")
    print(f"  Dimensions: {width}x{height}")
    
    # Get FMG heights
    fmg_heights = [cell['h'] for cell in fmg_data['pack']['cells']]
    print(f"\nFMG map has {len(fmg_heights)} cells")
    print(f"  Min: {min(fmg_heights)}")
    print(f"  Max: {max(fmg_heights)}")
    print(f"  Mean: {np.mean(fmg_heights):.1f}")
    print(f"  First 20: {fmg_heights[:20]}")
    
    # Generate Voronoi graph
    config = GridConfig(width=width, height=height, cells_desired=10000)
    graph = generate_voronoi_graph(config, seed=seed)
    
    print(f"\nPython grid generated:")
    print(f"  Cells: {len(graph.points)} (grid: {graph.cells_x}x{graph.cells_y})")
    
    # Create heightmap
    heightmap_config = HeightmapConfig(
        width=width,
        height=height,
        cells_x=graph.cells_x,
        cells_y=graph.cells_y,
        cells_desired=10000,
        spacing=graph.spacing
    )
    
    generator = HeightmapGenerator(heightmap_config, graph)
    
    # Test with lowIsland template (which was in the original tests)
    template = get_template("lowIsland")
    heights = generator.from_template(template, seed=seed)
    
    print(f"\nPython heightmap (lowIsland template):")
    print(f"  Min: {heights.min()}")
    print(f"  Max: {heights.max()}")
    print(f"  Mean: {heights.mean():.1f}")
    print(f"  First 20: {heights[:20].tolist()}")
    
    # Check if we can match the exact cell count
    if len(heights) == len(fmg_heights):
        print(f"\n✓ Cell counts match!")
        
        # Compare heights
        matches = 0
        for i in range(min(100, len(heights))):
            if heights[i] == fmg_heights[i]:
                matches += 1
        
        print(f"\nFirst 100 cells: {matches} matches ({matches}%)")
        
        # Overall statistics
        exact_matches = np.sum(heights == np.array(fmg_heights))
        print(f"Total exact matches: {exact_matches}/{len(heights)} ({exact_matches/len(heights)*100:.1f}%)")
    else:
        print(f"\n✗ Cell count mismatch: Python {len(heights)} vs FMG {len(fmg_heights)}")
        
        # Still compare first 20
        print(f"\nComparison of first 20 values:")
        print("Index | Python | FMG | Match")
        print("------|--------|-----|------")
        for i in range(20):
            match = heights[i] == fmg_heights[i]
            print(f"{i:5d} | {heights[i]:6d} | {fmg_heights[i]:3d} | {'YES' if match else 'NO'}")


if __name__ == "__main__":
    main()