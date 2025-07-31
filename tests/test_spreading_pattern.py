#!/usr/bin/env python3
"""Test spreading pattern to understand the issue.

This test uses FMG's seeding pattern where a single seed is reseeded
at each major pipeline stage:
1. Graph generation reseeds with the seed
2. Heightmap generation reseeds with the same seed
3. Features markup reseeds with the same seed (if needed)
"""

import numpy as np
from py_fmg.core.voronoi_graph import generate_voronoi_graph, GridConfig
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig

# Use the exact same parameters as debug_prng_tracking.py
WIDTH = 1200
HEIGHT = 1000
CELLS_DESIRED = 10000
SEED = "123456789"  # Single seed for all stages

# Generate graph (this will reseed internally)
config = GridConfig(WIDTH, HEIGHT, CELLS_DESIRED)
graph = generate_voronoi_graph(config, seed=SEED)

# Create heightmap generator (this will reseed internally)
heightmap_config = HeightmapConfig(
    width=WIDTH,
    height=HEIGHT,
    cells_x=graph.cells_x,
    cells_y=graph.cells_y,
    cells_desired=CELLS_DESIRED,
    spacing=graph.spacing,
)

# Pass seed to heightmap generator for proper reseeding
generator = HeightmapGenerator(heightmap_config, graph, seed=SEED)

# Get PRNG from generator
prng = generator._prng

# Instrument add_hill to see what's happening
def instrumented_add_hill(count, height, range_x, range_y):
    """Track hill generation in detail."""
    count_val = int(generator._get_number_in_range(count))
    print(f"Adding {count_val} hills")
    
    total_cells_changed = 0
    total_prng_calls = 0
    
    for i in range(count_val):
        prng_start = prng.call_count
        
        # Track one hill (use uint8 to match FMG)
        change = np.zeros(generator.n_cells, dtype=np.uint8)
        h = generator._lim(generator._get_number_in_range(height))
        
        # Count PRNG calls for height generation
        height_prng_calls = prng.call_count - prng_start
        
        # Find starting point
        limit = 0
        start = -1
        while limit < 50:
            x = generator._get_point_in_range(range_x, generator.config.width)
            y = generator._get_point_in_range(range_y, generator.config.height)
            start = generator._find_grid_cell(x, y)
            
            if generator.heights[start] + h <= 90:
                break
            limit += 1
        
        if start == -1:
            print(f"  Hill {i+1}: FAILED to find valid start point")
            continue
            
        # Count PRNG calls for finding start
        start_prng_calls = prng.call_count - prng_start - height_prng_calls
        
        # Spread
        change[start] = h  # Store as float, not int
        queue = [start]
        cells_visited = 0
        max_distance = 0
        spreading_prng_calls = 0
        cells_below_threshold = 0
        
        # Track distance from start
        distances = {start: 0}
        
        # Debug: track height decay
        height_samples = []
        
        while queue:
            current = queue.pop(0)
            cells_visited += 1
            current_dist = distances[current]
            
            for neighbor in graph.cell_neighbors[current]:
                if change[neighbor] > 0:
                    continue
                
                # Calculate new height
                rand_val = generator._random()
                spreading_prng_calls += 1
                new_height = (change[current] ** generator.blob_power) * (rand_val * 0.2 + 0.9)
                
                # Store the value (will be truncated to int by uint8 array)
                change[neighbor] = new_height
                
                # Track height decay (show both float and stored int)
                if current_dist < 10:  # Sample first 10 distances
                    height_samples.append((current_dist, change[current], new_height, change[neighbor]))
                
                # Check the STORED truncated value!
                if change[neighbor] > 1:
                    queue.append(neighbor)
                    distances[neighbor] = current_dist + 1
                    max_distance = max(max_distance, current_dist + 1)
                else:
                    cells_below_threshold += 1
        
        # Count actual cells changed
        cells_changed = np.sum(change > 0)
        prng_calls = prng.call_count - prng_start
        
        print(f"  Hill {i+1}: start={start}, height={h:.0f}, cells_changed={cells_changed}, max_dist={max_distance}")
        print(f"    Cells visited: {cells_visited}, neighbors checked: {spreading_prng_calls}")
        print(f"    PRNG calls: height={height_prng_calls}, start={start_prng_calls}, spreading={spreading_prng_calls}, total={prng_calls}")
        print(f"    Cells below threshold: {cells_below_threshold}, blob_power={generator.blob_power}")
        
        # Calculate theoretical decay distance
        threshold = 1.0
        theoretical_dist = 0
        test_height = float(h)
        while test_height > threshold and theoretical_dist < 200:
            test_height = test_height ** generator.blob_power * 0.9  # Use minimum factor
            theoretical_dist += 1
        print(f"    Theoretical max distance (at min random): {theoretical_dist}")
        
        if height_samples:
            print(f"    Height decay samples (first few):")
            for dist, curr_h, calc_h, stored_h in height_samples[:5]:
                print(f"      dist={dist}: current={curr_h} -> calc={calc_h:.2f} -> stored={stored_h}")
        
        total_cells_changed += cells_changed
        total_prng_calls += prng_calls
        
        # Actually apply the changes
        generator.heights = generator._lim(generator.heights + change)
    
    return total_cells_changed, total_prng_calls

# Test the first Hill command from isthmus template
print("Testing: Hill 5-10 15-30 0-30 0-20")
print(f"Graph cells: {generator.n_cells}, blob_power: {generator.blob_power}")
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