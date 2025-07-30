#!/usr/bin/env python3
"""Debug the exact spreading behavior."""

import sys
sys.path.append('/home/user/py-fmg')

import numpy as np
from py_fmg.core.voronoi_graph import generate_voronoi_graph, GridConfig
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig
from py_fmg.utils.random import set_random_seed, get_prng

# Small test
WIDTH = 1200
HEIGHT = 1000
CELLS_DESIRED = 10000
SEED = "debug-exact"

set_random_seed(SEED)

config = GridConfig(WIDTH, HEIGHT, CELLS_DESIRED)
graph = generate_voronoi_graph(config, seed=SEED)

heightmap_config = HeightmapConfig(
    width=WIDTH,
    height=HEIGHT,
    cells_x=graph.cells_x,
    cells_y=graph.cells_y,
    cells_desired=CELLS_DESIRED,
    spacing=graph.spacing,
)

generator = HeightmapGenerator(heightmap_config, graph)

# Manually trace through one hill
change = np.zeros(generator.n_cells, dtype=np.uint8)
h = 50  # Fixed height for testing

# Find a starting cell
start = 5000  # Middle of the map

print(f"Starting cell: {start}")
print(f"Initial height: {h}")
print(f"Blob power: {generator.blob_power}")

# Initialize
change[start] = int(h)
queue = [start]

cells_visited = 0
prng_calls = 0
max_dist = 0

# Track distances
distances = {start: 0}

# Process first few iterations manually
iteration = 0
while queue and iteration < 5:
    print(f"\nIteration {iteration}:")
    print(f"  Queue size: {len(queue)}")
    
    # Process one cell
    current = queue.pop(0)
    cells_visited += 1
    current_dist = distances[current]
    
    print(f"  Processing cell {current}, change[{current}] = {change[current]}")
    
    # Check each neighbor
    for i, neighbor in enumerate(graph.cell_neighbors[current]):
        print(f"    Neighbor {i}: cell {neighbor}")
        
        # Check if already processed
        if change[neighbor] > 0:
            print(f"      -> Already has change value {change[neighbor]}, skipping")
            continue
        
        # Calculate new height
        current_height = change[current]
        rand_val = generator._random()
        prng_calls += 1
        new_height_float = (current_height ** generator.blob_power) * (rand_val * 0.2 + 0.9)
        
        print(f"      -> {current_height} ** {generator.blob_power} * {rand_val * 0.2 + 0.9:.3f} = {new_height_float:.3f}")
        
        # Assign to uint8
        change[neighbor] = int(new_height_float)
        print(f"      -> Stored as uint8: {change[neighbor]}")
        
        # Check if continues
        if change[neighbor] > 1:
            queue.append(neighbor)
            distances[neighbor] = current_dist + 1
            max_dist = max(max_dist, current_dist + 1)
            print(f"      -> Added to queue (distance {current_dist + 1})")
        else:
            print(f"      -> Too low, not added to queue")
    
    iteration += 1

print(f"\nAfter {iteration} iterations:")
print(f"  Cells visited: {cells_visited}")
print(f"  PRNG calls: {prng_calls}")
print(f"  Max distance: {max_dist}")
print(f"  Queue size: {len(queue)}")

# Now check if the issue is with how numpy handles the assignment
print("\n\nTesting uint8 assignment edge cases:")
test_arr = np.zeros(10, dtype=np.uint8)

test_values = [1.0, 1.1, 1.5, 1.9, 2.0, 2.1]
for val in test_values:
    test_arr[0] = val
    test_arr[1] = int(val)
    print(f"  {val} -> direct: {test_arr[0]}, int(): {test_arr[1]}, check > 1: {test_arr[0] > 1}, {test_arr[1] > 1}")