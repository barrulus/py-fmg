#!/usr/bin/env python3
"""
Test exact FMG process to find discrepancy.
"""

import numpy as np
from py_fmg.core import GridConfig, generate_voronoi_graph, HeightmapGenerator, HeightmapConfig
from py_fmg.utils.random import set_random_seed, get_prng


def main():
    # Set up exactly as FMG
    grid_seed = "1234567"
    map_seed = "651658815"
    
    config = GridConfig(width=300, height=300, cells_desired=10000)
    graph = generate_voronoi_graph(config, seed=grid_seed)
    
    heightmap_config = HeightmapConfig(
        width=300, height=300,
        cells_x=graph.cells_x, cells_y=graph.cells_y,
        cells_desired=10000, spacing=graph.spacing
    )
    
    generator = HeightmapGenerator(heightmap_config, graph)
    
    # Test hypothesis: Maybe FMG starts with different initial heights?
    print("Testing initial state:")
    print(f"  Initial heights all zero: {np.all(generator.heights == 0)}")
    print(f"  Heights dtype: {generator.heights.dtype}")
    
    # Set seed for generation
    set_random_seed(map_seed)
    generator._prng = None
    
    # Test first hill parameters
    print("\n\nFirst hill: Hill 1 90-99 60-80 45-55")
    
    # Get exact random values
    prng = get_prng()
    
    # Height: 90-99
    r1 = prng.random()
    height = 90 + r1 * 9
    print(f"  Random for height: {r1:.10f}")
    print(f"  Height: {height:.2f}")
    
    # X position: 60-80% of 300
    r2 = prng.random()
    x = 300 * (0.6 + r2 * 0.2)
    print(f"  Random for x: {r2:.10f}")
    print(f"  X position: {x:.2f}")
    
    # Y position: 45-55% of height value (not map height!)
    r3 = prng.random()
    y = height * (0.45 + r3 * 0.1)
    print(f"  Random for y: {r3:.10f}")
    print(f"  Y position: {y:.2f}")
    
    # Find cell
    col = min(int(x / 3.0), 99)
    row = min(int(y / 3.0), 99)
    cell = row * 100 + col
    print(f"  Grid cell: row={row}, col={col}, index={cell}")
    
    # Test hypothesis: Maybe the issue is with clamping?
    print("\n\nTesting clamping behavior:")
    print(f"  lim(92) = {generator._lim(92)}")
    print(f"  lim(150) = {generator._lim(150)}")
    print(f"  lim(-10) = {generator._lim(-10)}")
    
    # Actually run the first hill
    print("\n\nExecuting first hill:")
    set_random_seed(map_seed)
    generator._prng = None
    generator.heights[:] = 0  # Reset
    
    generator.add_hill(1, "90-99", "60-80", "45-55")
    
    print(f"  Non-zero cells: {np.sum(generator.heights > 0)}")
    print(f"  Heights: min={generator.heights.min()}, max={generator.heights.max()}, mean={generator.heights.mean():.1f}")
    print(f"  First 20 heights: {generator.heights[:20]}")
    
    # Check if the spreading is different
    print("\n\nChecking spreading pattern:")
    # Count cells by height
    for h in [80, 60, 40, 20, 10, 5, 1]:
        count = np.sum(generator.heights >= h)
        print(f"  Cells with height >= {h}: {count}")


if __name__ == "__main__":
    main()