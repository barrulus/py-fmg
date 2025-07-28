#!/usr/bin/env python3
"""
Test just the first hill operation to debug differences.
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
    
    print("Initial state:")
    print(f"  Heights all zero: {np.all(generator.heights == 0)}")
    print(f"  Heights shape: {generator.heights.shape}")
    print(f"  Heights dtype: {generator.heights.dtype}")
    
    # Test random number generation
    print("\nRandom number generation:")
    prng = get_prng()
    for i in range(5):
        print(f"  Random {i}: {prng.random():.10f}")
    
    # Reset seed and generator
    set_random_seed(map_seed)
    generator._prng = None
    
    # Execute first command: Hill 1 90-99 60-80 45-55
    print("\n\nExecuting: Hill 1 90-99 60-80 45-55")
    
    # Manually trace the operation
    count = 1
    height_range = "90-99"
    x_range = "60-80"
    y_range = "45-55"
    
    # Get height
    height = generator._get_number_in_range(height_range)
    print(f"\nHeight calculation:")
    print(f"  Range: {height_range}")
    print(f"  Selected height: {height}")
    
    # Get position
    x = generator._get_point_in_range(x_range, width)
    y = generator._get_point_in_range(y_range, height)
    print(f"\nPosition calculation:")
    print(f"  X range: {x_range}, width: {width}")
    print(f"  Y range: {y_range}, height: {height}")
    print(f"  Selected position: ({x:.2f}, {y:.2f})")
    
    # Find cell
    cell = generator._find_grid_cell(x, y)
    print(f"\nCell calculation:")
    print(f"  Grid spacing: {graph.spacing}")
    print(f"  Grid dimensions: {graph.cells_x}x{graph.cells_y}")
    print(f"  Starting cell: {cell}")
    
    # Let's trace the blob spreading
    print(f"\nBlob spreading parameters:")
    print(f"  Blob power: {generator.blob_power}")
    print(f"  Cells desired: {cells_desired}")
    
    # Actually execute the hill
    generator.add_hill(1, height_range, x_range, y_range)
    
    print(f"\nAfter hill:")
    print(f"  Non-zero cells: {np.sum(generator.heights > 0)}")
    print(f"  Heights: min={generator.heights.min()}, max={generator.heights.max()}, mean={generator.heights.mean():.1f}")
    print(f"  First 20 heights: {generator.heights[:20]}")
    
    # Check height at starting cell
    print(f"\nStarting cell height: {generator.heights[cell]}")
    
    # Check some neighbors
    if hasattr(graph, 'cell_neighbors') and cell < len(graph.cell_neighbors):
        neighbors = graph.cell_neighbors[cell]
        print(f"\nNeighbor heights:")
        for i, n in enumerate(neighbors[:5]):
            print(f"  Neighbor {i} (cell {n}): {generator.heights[n]}")


if __name__ == "__main__":
    main()