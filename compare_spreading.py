#!/usr/bin/env python3
"""
Compare spreading behavior between Python and expected FMG behavior.
"""

import numpy as np
from py_fmg.core import GridConfig, generate_voronoi_graph, HeightmapGenerator, HeightmapConfig
from py_fmg.utils.random import set_random_seed


def trace_spreading():
    """Trace the spreading algorithm step by step."""
    # Small grid for analysis
    config = GridConfig(width=30, height=30, cells_desired=100)
    graph = generate_voronoi_graph(config, seed="test")
    
    heightmap_config = HeightmapConfig(
        width=30, height=30,
        cells_x=graph.cells_x, cells_y=graph.cells_y,
        cells_desired=100, spacing=graph.spacing
    )
    
    generator = HeightmapGenerator(heightmap_config, graph)
    
    # Reset seed
    set_random_seed("test123")
    generator._prng = None
    
    # Manually implement spreading with detailed logging
    change = np.zeros(generator.n_cells, dtype=np.float32)
    h = 50  # Initial height
    start = 50  # Start cell
    
    change[start] = h
    queue = [start]
    visited = set([start])
    
    blob_power = generator.blob_power
    print(f"Blob power: {blob_power}")
    print(f"Starting height: {h}")
    print(f"Starting cell: {start}")
    
    # Track spreading by level
    level = 0
    current_level = [start]
    next_level = []
    
    while current_level and level < 10:
        print(f"\n\nLevel {level}: {len(current_level)} cells")
        
        for cell in current_level:
            cell_height = change[cell]
            print(f"\n  Cell {cell} (height={cell_height:.2f}):")
            
            neighbors = graph.cell_neighbors[cell]
            print(f"    {len(neighbors)} neighbors: {neighbors}")
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    
                    # Calculate new height
                    rand_val = generator._random()
                    rand_factor = rand_val * 0.2 + 0.9
                    new_height = (cell_height ** blob_power) * rand_factor
                    
                    print(f"      Neighbor {neighbor}: {cell_height:.2f}^{blob_power} * {rand_factor:.3f} = {new_height:.2f}")
                    
                    if new_height > 1:
                        change[neighbor] = new_height
                        next_level.append(neighbor)
                    else:
                        print(f"        (stopped - height <= 1)")
        
        current_level = next_level
        next_level = []
        level += 1
        
        print(f"\n  Level {level} summary: {len(visited)} total cells visited")
    
    # Apply changes and show result
    generator.heights = generator._lim(generator.heights + change)
    
    print(f"\n\nFinal result:")
    print(f"  Non-zero cells: {np.sum(generator.heights > 0)}")
    print(f"  Total cells affected: {len(visited)}")
    print(f"  Heights: min={generator.heights.min()}, max={generator.heights.max()}, mean={generator.heights.mean():.1f}")


def test_blob_power_effect():
    """Test how blob power affects spreading."""
    print("\n\n" + "=" * 60)
    print("Testing blob power effect on spreading")
    print("=" * 60)
    
    for blob_power in [0.90, 0.93, 0.95, 0.98]:
        # Calculate how many steps to decay from 92 to 1
        value = 92
        steps = 0
        min_factor = 0.9  # Minimum random factor
        
        while value > 1 and steps < 100:
            value = (value ** blob_power) * min_factor
            steps += 1
        
        print(f"\nBlob power {blob_power}:")
        print(f"  Steps to decay from 92 to 1: {steps}")
        print(f"  Final value: {value:.3f}")
        
        # Estimate affected cells (rough approximation)
        # In a grid, each level adds ~4 * level cells
        affected = 1
        for level in range(1, min(steps, 50)):
            affected += min(4 * level, 10000 - affected)
        
        print(f"  Estimated cells affected: {affected}")


if __name__ == "__main__":
    trace_spreading()
    test_blob_power_effect()