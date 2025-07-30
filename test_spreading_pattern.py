#!/usr/bin/env python3
"""Test spreading pattern to understand the issue."""

import sys
sys.path.append('/home/user/py-fmg')

import numpy as np
from py_fmg.core.voronoi_graph import generate_voronoi_graph, GridConfig
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig
from py_fmg.utils.random import set_random_seed, get_prng

# Use the exact same parameters as debug_prng_tracking.py
WIDTH = 1200
HEIGHT = 1000
CELLS_DESIRED = 10000
SEED = "654321"  # Same as in the test

# Set random seed
set_random_seed(SEED)

# Generate graph
config = GridConfig(WIDTH, HEIGHT, CELLS_DESIRED)
graph = generate_voronoi_graph(config, seed=SEED)

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

# Get PRNG
prng = get_prng()

# Instrument add_hill to see what's happening
def instrumented_add_hill(count, height, range_x, range_y):
    """Track hill generation in detail."""
    count_val = int(generator._get_number_in_range(count))
    print(f"Adding {count_val} hills")
    
    total_cells_changed = 0
    total_prng_calls = 0
    
    for i in range(count_val):
        prng_start = prng.call_count
        
        # Track one hill
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
        
        # Spread
        change[start] = int(h)
        queue = [start]
        cells_visited = 0
        max_distance = 0
        
        # Track distance from start
        distances = {start: 0}
        
        while queue:
            current = queue.pop(0)
            cells_visited += 1
            current_dist = distances[current]
            
            for neighbor in graph.cell_neighbors[current]:
                if change[neighbor] > 0:
                    continue
                
                # Calculate new height
                rand_val = generator._random()
                new_height = (change[current] ** generator.blob_power) * (rand_val * 0.2 + 0.9)
                
                if new_height > 1:
                    change[neighbor] = int(new_height)
                    queue.append(neighbor)
                    distances[neighbor] = current_dist + 1
                    max_distance = max(max_distance, current_dist + 1)
        
        # Apply changes
        cells_changed = np.sum(change > 0)
        prng_calls = prng.call_count - prng_start
        
        print(f"  Hill {i+1}: start={start}, height={h:.0f}, cells_changed={cells_changed}, max_dist={max_distance}, prng_calls={prng_calls}")
        
        total_cells_changed += cells_changed
        total_prng_calls += prng_calls
        
        # Actually apply the changes
        generator.heights = generator._lim(generator.heights + change)
    
    return total_cells_changed, total_prng_calls

# Test the first Hill command from isthmus template
print("Testing: Hill 5-10 15-30 0-30 0-20")
print("-" * 60)

prng.call_count = 0
cells, calls = instrumented_add_hill("5-10", "15-30", "0-30", "0-20")

print(f"\nTotal cells changed: {cells}")
print(f"Total PRNG calls: {calls}")
print(f"Average cells per hill: {cells / 5.0:.1f}")
print(f"Average PRNG calls per hill: {calls / 5.0:.1f}")

# Compare with FMG which showed ~25 calls for this command
print(f"\nFMG used ~25 calls for this command")
print(f"We're using {calls / 25:.1f}x more PRNG calls!")