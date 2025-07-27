#!/usr/bin/env python3
"""
Trace heightmap generation step by step.
"""

import numpy as np
from py_fmg.core import GridConfig, generate_voronoi_graph, HeightmapGenerator, HeightmapConfig
from py_fmg.utils.random import set_random_seed, get_prng


def main():
    # Set up parameters
    grid_seed = "1234567"
    map_seed = "651658815"
    width, height = 300, 300
    cells_desired = 10000
    
    # Generate Voronoi graph
    print(f"Generating Voronoi graph...")
    config = GridConfig(width=width, height=height, cells_desired=cells_desired)
    graph = generate_voronoi_graph(config, seed=grid_seed)
    
    # Create heightmap generator
    heightmap_config = HeightmapConfig(
        width=width,
        height=height,
        cells_x=graph.cells_x,
        cells_y=graph.cells_y,
        cells_desired=cells_desired,
        spacing=graph.spacing
    )
    
    generator = HeightmapGenerator(heightmap_config, graph)
    
    # Set seed for heightmap generation
    set_random_seed(map_seed)
    generator._prng = None
    
    print(f"\nInitial heights (first 20): {generator.heights[:20]}")
    print(f"Initial land percentage: {np.sum(generator.heights >= 20) / len(generator.heights) * 100:.1f}%")
    
    # Execute first command: Hill 1 90-99 60-80 45-55
    print("\n\nExecuting: Hill 1 90-99 60-80 45-55")
    
    # Get the exact parameters
    count = 1
    height_range = "90-99"
    x_range = "60-80"
    y_range = "45-55"
    
    # Parse height
    height = generator._get_number_in_range(height_range)
    print(f"  Height selected: {height}")
    
    # Get position
    x = generator._get_point_in_range(x_range, width)
    y = generator._get_point_in_range(y_range, height)
    print(f"  Position: ({x:.2f}, {y:.2f})")
    
    # Find cell
    cell = generator._find_grid_cell(x, y)
    print(f"  Starting cell: {cell}")
    print(f"  Initial height at cell: {generator.heights[cell]}")
    
    # Apply the hill
    generator.add_hill(count, height_range, x_range, y_range)
    
    print(f"\nAfter first hill:")
    print(f"  Heights (first 20): {generator.heights[:20]}")
    print(f"  Height at starting cell: {generator.heights[cell]}")
    print(f"  Land percentage: {np.sum(generator.heights >= 20) / len(generator.heights) * 100:.1f}%")
    print(f"  Min: {generator.heights.min()}, Max: {generator.heights.max()}, Mean: {generator.heights.mean():.1f}")
    
    # Try with smaller test case
    print("\n\n=== SMALLER TEST ===")
    
    # Create tiny map
    small_config = GridConfig(width=30, height=30, cells_desired=100)
    small_graph = generate_voronoi_graph(small_config, seed="test")
    
    small_heightmap_config = HeightmapConfig(
        width=30,
        height=30,
        cells_x=small_graph.cells_x,
        cells_y=small_graph.cells_y,
        cells_desired=100,
        spacing=small_graph.spacing
    )
    
    small_gen = HeightmapGenerator(small_heightmap_config, small_graph)
    
    # Set seed
    set_random_seed("test123")
    small_gen._prng = None
    
    print(f"Small map initial heights: {small_gen.heights}")
    print(f"Small map dimensions: {small_graph.cells_x}x{small_graph.cells_y}")
    
    # Add one hill
    small_gen.add_hill(1, 50, "40-60", "40-60")
    
    print(f"After hill: min={small_gen.heights.min()}, max={small_gen.heights.max()}, "
          f"mean={small_gen.heights.mean():.1f}")
    print(f"Non-zero cells: {np.sum(small_gen.heights > 0)}")


if __name__ == "__main__":
    main()