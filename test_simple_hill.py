#!/usr/bin/env python3
"""
Test with exact same random numbers to see where divergence happens.
"""

import numpy as np
from py_fmg.core import GridConfig, generate_voronoi_graph, HeightmapGenerator, HeightmapConfig
from py_fmg.utils.random import set_random_seed, get_prng


def main():
    # Generate small grid
    config = GridConfig(width=30, height=30, cells_desired=100)
    graph = generate_voronoi_graph(config, seed="test")
    
    heightmap_config = HeightmapConfig(
        width=30,
        height=30,
        cells_x=graph.cells_x,
        cells_y=graph.cells_y,
        cells_desired=100,
        spacing=graph.spacing
    )
    
    generator = HeightmapGenerator(heightmap_config, graph)
    
    # Set seed
    set_random_seed("test123")
    generator._prng = None
    
    print(f"Grid: {graph.cells_x}x{graph.cells_y} = {len(graph.points)} cells")
    print(f"Blob power: {generator.blob_power}")
    
    # Add a simple hill with fixed parameters
    generator.add_hill(1, 50, "40-60", "40-60")
    
    print(f"\nAfter hill:")
    print(f"  Non-zero cells: {np.sum(generator.heights > 0)}")
    print(f"  Heights: min={generator.heights.min()}, max={generator.heights.max()}, mean={generator.heights.mean():.1f}")
    
    # Show height distribution
    unique, counts = np.unique(generator.heights, return_counts=True)
    print(f"\nHeight distribution:")
    for h, c in zip(unique, counts):
        if c > 0:
            print(f"  Height {h}: {c} cells")
    
    # Check the actual spread pattern
    print(f"\nFirst 20 heights: {generator.heights[:20]}")
    
    # Now let's manually implement hill with explicit logging
    print("\n\n=== Manual hill implementation ===")
    
    # Reset
    generator.heights[:] = 0
    set_random_seed("test123")
    generator._prng = None
    
    # Get parameters
    h = generator._lim(generator._get_number_in_range(50))
    x = generator._get_point_in_range("40-60", 30)
    y = generator._get_point_in_range("40-60", 30)
    start = generator._find_grid_cell(x, y)
    
    print(f"Hill parameters:")
    print(f"  Height: {h}")
    print(f"  Position: ({x:.2f}, {y:.2f})")
    print(f"  Start cell: {start}")
    
    # Manual spreading
    change = np.zeros(generator.n_cells, dtype=np.float32)
    change[start] = h
    queue = [start]
    visited = set([start])
    
    spread_count = 0
    while queue and spread_count < 5:  # Limit for debugging
        current = queue.pop(0)
        current_height = change[current]
        
        print(f"\nProcessing cell {current} (height={current_height:.2f}):")
        
        for neighbor in graph.cell_neighbors[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                rand_val = generator._random()
                new_height = (current_height ** generator.blob_power) * (rand_val * 0.2 + 0.9)
                
                print(f"  Neighbor {neighbor}: {current_height:.2f}^{generator.blob_power} * ({rand_val:.3f}*0.2+0.9) = {new_height:.2f}")
                
                if new_height > 1:
                    change[neighbor] = new_height
                    queue.append(neighbor)
        
        spread_count += 1


if __name__ == "__main__":
    main()