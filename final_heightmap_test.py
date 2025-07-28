#!/usr/bin/env python3
"""
Final test to generate heightmap and compare with FMG.
"""

import json
import numpy as np
from py_fmg.core import GridConfig, generate_voronoi_graph, HeightmapGenerator, HeightmapConfig
from py_fmg.config import get_template
from py_fmg.utils.random import set_random_seed


def main():
    # Load FMG reference
    with open('tests/Mateau Full 2025-07-27-14-53.json') as f:
        fmg_data = json.load(f)
    
    fmg_heights = np.array([cell['h'] for cell in fmg_data['grid']['cells']])
    
    # Generate heightmap
    grid_seed = "1234567"
    map_seed = "651658815"
    
    config = GridConfig(width=300, height=300, cells_desired=10000)
    graph = generate_voronoi_graph(config, seed=grid_seed)
    
    heightmap_config = HeightmapConfig(
        width=300,
        height=300,
        cells_x=graph.cells_x,
        cells_y=graph.cells_y,
        cells_desired=10000,
        spacing=graph.spacing
    )
    
    generator = HeightmapGenerator(heightmap_config, graph)
    
    # Generate with template
    template = get_template("lowIsland")
    heights = generator.from_template(template, seed=map_seed)
    
    print("Final Comparison")
    print("=" * 60)
    
    print("\nPython implementation:")
    print(f"  Heights: min={heights.min()}, max={heights.max()}, mean={heights.mean():.1f}")
    print(f"  Land percentage: {np.sum(heights >= 20) / len(heights) * 100:.1f}%")
    
    print("\nFMG reference:")
    print(f"  Heights: min={fmg_heights.min()}, max={fmg_heights.max()}, mean={fmg_heights.mean():.1f}")
    print(f"  Land percentage: {np.sum(fmg_heights >= 20) / len(fmg_heights) * 100:.1f}%")
    
    # Compare distributions
    print("\nHeight distribution comparison:")
    print("Height | Python | FMG")
    print("-------|--------|--------")
    for h in range(0, 55, 5):
        py_count = np.sum((heights >= h) & (heights < h+5))
        fmg_count = np.sum((fmg_heights >= h) & (fmg_heights < h+5))
        print(f"{h:2d}-{h+4:2d} | {py_count:6d} | {fmg_count:6d}")
    
    # Check exact matches
    matches = heights == fmg_heights
    print(f"\nExact matches: {np.sum(matches)} / {len(heights)} ({np.sum(matches)/len(heights)*100:.1f}%)")
    
    # Save our heights for analysis
    np.save('python_heights.npy', heights)
    np.save('fmg_heights.npy', fmg_heights)
    
    print("\nHeights saved to python_heights.npy and fmg_heights.npy")


if __name__ == "__main__":
    main()