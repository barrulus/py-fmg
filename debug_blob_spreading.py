#!/usr/bin/env python3
"""Debug blob spreading to understand PRNG call count."""

import sys
sys.path.append('/home/user/py-fmg')

import numpy as np
from py_fmg.core.voronoi_graph import generate_voronoi_graph, GridConfig
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig
from py_fmg.utils.random import set_random_seed, get_prng

# Test parameters
WIDTH = 1200
HEIGHT = 1000
CELLS_DESIRED = 10000
SEED = "blob-debug"

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

# Get PRNG and reset counter
prng = get_prng()
prng.call_count = 0

# Manually implement hill spreading with detailed tracking
def debug_add_hill(generator, height_val=50, x_val=600, y_val=500):
    """Add a single hill with detailed tracking."""
    change = np.zeros(generator.n_cells, dtype=np.uint8)
    h = int(height_val)
    
    # Find cell at specific position
    start = generator._find_grid_cell(x_val, y_val)
    print(f"Starting cell: {start}")
    print(f"Initial height: {h}")
    
    # Initialize spreading
    change[start] = h
    queue = [start]
    
    cells_processed = 0
    prng_calls_start = prng.call_count
    
    # Track spreading iteration by iteration
    iteration = 0
    while queue and iteration < 10:  # Limit iterations for debugging
        print(f"\nIteration {iteration}:")
        print(f"  Queue size: {len(queue)}")
        
        # Process one cell
        if queue:
            current = queue.pop(0)
            cells_processed += 1
            
            neighbors_added = 0
            for neighbor in graph.cell_neighbors[current]:
                if change[neighbor] > 0:
                    continue
                
                # Calculate new height
                prng_before = prng.call_count
                rand_val = generator._random()
                new_height = (change[current] ** generator.blob_power) * (rand_val * 0.2 + 0.9)
                
                print(f"    Cell {current} -> {neighbor}: change[{current}]={change[current]}, new_height={new_height:.2f}")
                
                if new_height > 1:
                    change[neighbor] = int(new_height)
                    queue.append(neighbor)
                    neighbors_added += 1
                else:
                    print(f"      -> Height too low, stopping spread")
            
            print(f"  Added {neighbors_added} neighbors")
        
        iteration += 1
    
    prng_calls = prng.call_count - prng_calls_start
    print(f"\nSummary:")
    print(f"  Cells processed: {cells_processed}")
    print(f"  PRNG calls: {prng_calls}")
    print(f"  Final queue size: {len(queue)}")
    
    # Count cells with changes
    cells_changed = np.sum(change > 0)
    print(f"  Cells changed: {cells_changed}")
    
    return change

# Test with a single fixed hill
print("Testing single hill with fixed parameters...")
print("="*60)

change_array = debug_add_hill(generator)

# Now test with actual add_hill method
print("\n\nTesting with actual add_hill method...")
print("="*60)

generator.heights[:] = 0
prng.call_count = 0

generator.add_hill("1", "50", "45-55", "45-55")

print(f"PRNG calls for single hill: {prng.call_count}")
land = sum(1 for h in generator.heights if h > 20)
print(f"Land cells: {land}/{len(generator.heights)} ({land/len(generator.heights)*100:.1f}%)")