#!/usr/bin/env python3
"""
Trace blob spreading to understand why it spreads to all cells.
"""

import numpy as np
from py_fmg.core import GridConfig, generate_voronoi_graph, HeightmapGenerator, HeightmapConfig
from py_fmg.utils.random import set_random_seed, get_prng


def main():
    # Small test case
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
    
    # Manually implement the spreading to trace it
    change = np.zeros(generator.n_cells, dtype=np.float32)
    h = 50  # Start with height 50
    start = 50  # Middle cell
    
    change[start] = h
    queue = [start]
    visited = set([start])
    
    iteration = 0
    while queue and iteration < 10:  # Limit iterations for debugging
        current = queue.pop(0)
        current_height = change[current]
        
        print(f"\nIteration {iteration}, processing cell {current}:")
        print(f"  Current height in change array: {current_height:.2f}")
        
        # Spread to neighbors
        for neighbor in graph.cell_neighbors[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                # Calculate new height with power decay and randomness
                new_height = (current_height ** generator.blob_power) * (generator._random() * 0.2 + 0.9)
                
                print(f"    Neighbor {neighbor}: {current_height:.2f} ^ {generator.blob_power} * rand = {new_height:.2f}")
                
                if new_height > 1:
                    change[neighbor] = new_height
                    queue.append(neighbor)
        
        iteration += 1
        print(f"  Queue size: {len(queue)}, Visited: {len(visited)}")
    
    print(f"\n\nFinal statistics:")
    print(f"  Cells affected: {np.sum(change > 0)}")
    print(f"  Total cells: {generator.n_cells}")
    print(f"  Percentage affected: {np.sum(change > 0) / generator.n_cells * 100:.1f}%")
    
    # Test decay rate
    print(f"\n\nDecay test:")
    value = 92
    for i in range(20):
        value = value ** 0.98 * 0.9  # Minimum random factor
        print(f"  Step {i}: {value:.2f}")
        if value <= 1:
            print(f"  Stopped at step {i}")
            break


if __name__ == "__main__":
    main()