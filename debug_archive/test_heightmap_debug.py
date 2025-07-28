#!/usr/bin/env python3
"""
Debug script to understand heightmap generation differences.
"""

import json
import numpy as np
from py_fmg.core import GridConfig, generate_voronoi_graph, HeightmapGenerator, HeightmapConfig
from py_fmg.config import get_template
from py_fmg.utils.random import set_random_seed, get_prng


def main():
    # Load reference data
    with open('tests/Mateau Full 2025-07-27-14-53.json') as f:
        fmg_data = json.load(f)
    
    # Set up parameters
    grid_seed = "1234567"  # Grid was reused from this seed
    map_seed = "651658815"  # The actual map seed
    width, height = 300, 300
    cells_desired = 10000
    
    # Step 1: Generate Voronoi graph
    print(f"Generating Voronoi graph with seed '{grid_seed}'...")
    config = GridConfig(width=width, height=height, cells_desired=cells_desired)
    graph = generate_voronoi_graph(config, seed=grid_seed)
    
    print(f"Graph dimensions: {graph.cells_x}x{graph.cells_y}, spacing={graph.spacing}")
    
    # Step 2: Test Alea PRNG
    print(f"\nTesting Alea PRNG with seed '{map_seed}':")
    set_random_seed(map_seed)
    prng = get_prng()
    
    # Generate first 10 random numbers
    print("First 10 random values from Alea PRNG:")
    for i in range(10):
        print(f"  {i}: {prng.random():.10f}")
    
    # Reset and test again
    set_random_seed(map_seed)
    prng2 = get_prng()
    print("\nAfter reset, first 5 values:")
    for i in range(5):
        print(f"  {i}: {prng2.random():.10f}")
    
    # Step 3: Generate heightmap step by step
    print(f"\n\nGenerating heightmap with lowIsland template...")
    heightmap_config = HeightmapConfig(
        width=width,
        height=height,
        cells_x=graph.cells_x,
        cells_y=graph.cells_y,
        cells_desired=cells_desired,
        spacing=graph.spacing
    )
    
    generator = HeightmapGenerator(heightmap_config, graph)
    
    # Get template and parse it
    template = get_template("lowIsland")
    print("\nlowIsland template commands:")
    for i, line in enumerate(template.strip().split('\n')):
        print(f"  {i+1}: {line.strip()}")
    
    # Generate with fresh seed
    set_random_seed(map_seed)
    generator._prng = None  # Reset generator's PRNG
    
    # Execute template
    heights = generator.from_template(template, seed=map_seed)
    
    # Compare with FMG
    fmg_heights = np.array([cell['h'] for cell in fmg_data['grid']['cells']])
    
    print(f"\n\nHeight comparison:")
    print(f"Python: min={heights.min()}, max={heights.max()}, mean={heights.mean():.1f}")
    print(f"FMG:    min={fmg_heights.min()}, max={fmg_heights.max()}, mean={fmg_heights.mean():.1f}")
    
    print(f"\nLand percentage:")
    print(f"Python: {np.sum(heights >= 20) / len(heights) * 100:.1f}%")
    print(f"FMG:    {np.sum(fmg_heights >= 20) / len(fmg_heights) * 100:.1f}%")
    
    # Check first few height values
    print(f"\nFirst 20 height values:")
    print(f"Python: {heights[:20]}")
    print(f"FMG:    {fmg_heights[:20]}")
    
    # Check if heights are exactly equal
    matches = heights == fmg_heights
    print(f"\nExact matches: {np.sum(matches)} / {len(heights)} ({np.sum(matches)/len(heights)*100:.1f}%)")
    
    if not np.all(matches):
        # Find first mismatch
        first_mismatch = np.where(~matches)[0][0]
        print(f"\nFirst mismatch at index {first_mismatch}:")
        print(f"  Python: {heights[first_mismatch]}")
        print(f"  FMG:    {fmg_heights[first_mismatch]}")


if __name__ == "__main__":
    main()