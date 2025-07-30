#!/usr/bin/env python3
"""Debug which cells are being processed during blob spreading."""

import sys
sys.path.append('/home/user/py-fmg')

import numpy as np
from py_fmg.core.voronoi_graph import generate_voronoi_graph, GridConfig
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig
from py_fmg.utils.random import set_random_seed, get_prng

# Small test case
WIDTH = 120  # Much smaller for debugging
HEIGHT = 100
CELLS_DESIRED = 100  # Very few cells
SEED = "debug-spread"

# Set random seed
set_random_seed(SEED)

# Generate graph
config = GridConfig(WIDTH, HEIGHT, CELLS_DESIRED)
graph = generate_voronoi_graph(config, seed=SEED)

print(f"Graph has {len(graph.points)} cells")
print(f"Grid: {graph.cells_x}x{graph.cells_y}")

# Create heightmap generator
heightmap_config = HeightmapConfig(
    width=WIDTH,
    height=HEIGHT,
    cells_x=graph.cells_x,
    cells_y=graph.cells_y,
    cells_desired=CELLS_DESIRED,
    spacing=graph.spacing,
)

generator = HeightmapGenerator(heightmap_config, graph)

# Instrument the _add_one_hill method to track what's happening
original_add_one_hill = generator._add_one_hill

def instrumented_add_one_hill(height, range_x, range_y):
    """Instrumented version to track spreading."""
    change = np.zeros(generator.n_cells, dtype=np.uint8)
    h = generator._lim(generator._get_number_in_range(height))
    
    # Find starting point
    limit = 0
    while limit < 50:
        x = generator._get_point_in_range(range_x, generator.config.width)
        y = generator._get_point_in_range(range_y, generator.config.height)
        start = generator._find_grid_cell(x, y)
        
        if generator.heights[start] + h <= 90:
            break
        limit += 1
    
    print(f"Starting from cell {start} with height {h}")
    
    # Track spreading
    change[start] = int(h)
    queue = [start]
    
    cells_processed = set()
    cells_with_change = set([start])
    prng_calls_for_spreading = 0
    
    while queue:
        if len(cells_processed) > 200:  # Safety limit
            print("  WARNING: Too many cells processed, breaking")
            break
            
        current = queue.pop(0)
        if current in cells_processed:
            print(f"  WARNING: Cell {current} already processed!")
            continue
            
        cells_processed.add(current)
        
        # Check neighbors
        for neighbor in generator.graph.cell_neighbors[current]:
            if change[neighbor] > 0:
                continue
            
            # Calculate new height
            rand_val = generator._random()
            prng_calls_for_spreading += 1
            new_height = (change[current] ** generator.blob_power) * (rand_val * 0.2 + 0.9)
            
            if new_height > 1:
                change[neighbor] = int(new_height)
                cells_with_change.add(neighbor)
                queue.append(neighbor)
    
    print(f"  Cells processed: {len(cells_processed)}")
    print(f"  Cells changed: {len(cells_with_change)}")
    print(f"  PRNG calls for spreading: {prng_calls_for_spreading}")
    
    # Apply changes
    generator.heights = generator._lim(generator.heights + change)

# Replace method
generator._add_one_hill = instrumented_add_one_hill

# Test
prng = get_prng()
prng.call_count = 0

print("\nAdding single hill...")
generator.add_hill("1", "50", "40-60", "40-60")

print(f"\nTotal PRNG calls: {prng.call_count}")